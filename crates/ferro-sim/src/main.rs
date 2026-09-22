//! `ferro-sim` -- run the scheduling experiments and write the results out.
//!
//! One command produces every number the term project reports, from a clean
//! checkout, with no GPU. Nothing is hard-coded into a document: the tables in
//! the write-up are generated from `summary.csv`, and if the code changes the
//! tables change with it.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ferro_sched::{PlacementWeights, SchedulerConfig};
use ferro_sim::metrics::summarise;
use ferro_sim::sim::{Dispatch, ExecutionModel, SimOutcome};
use ferro_sim::{catalogue, git_commit, now_unix_s, RunMetadata, RunSummary, Scenario};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "ferro-sim",
    version,
    about = "Offline comparison of FerroGrid's scheduling policies"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the whole catalogue against every policy pair and write the results.
    Run(RunArgs),
    /// List the scenarios and policies without running anything.
    List,
    /// Print one run's scheduler trace, for drawing a Gantt chart.
    Trace(TraceArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Where to write results.csv, summary.csv and runs.json.
    #[arg(long, default_value = "outputs/benchmarks")]
    out: PathBuf,

    /// Only these scenario letters, e.g. --scenario A --scenario C.
    #[arg(long = "scenario")]
    scenarios: Vec<String>,

    /// Queue policies to compare. Default: all of them.
    #[arg(long = "queue")]
    queue: Vec<String>,

    /// Placement policies to compare. Default: performance only, which keeps
    /// the queue comparison from being confounded by placement.
    #[arg(long = "placement", default_values_t = [String::from("performance")])]
    placement: Vec<String>,

    /// Also run every scenario under strict head-of-line dispatch, to measure
    /// what FerroGrid's opportunistic dispatcher is worth.
    #[arg(long)]
    with_strict: bool,

    /// Also run every scenario under reservation-based backfilling. Worth
    /// reading next to `--with-strict`: reservation is what bounds the large-job
    /// tail that opportunistic dispatch leaves open, and it can only do so for
    /// jobs whose submitters declared a duration. A scenario where few of them
    /// did (`estimate_fraction`) is a scenario where reservation degrades
    /// towards strict, which is the finding rather than a flaw in the run.
    #[arg(long)]
    with_reserved: bool,

    /// A GPU must have this much free VRAM to be placed on.
    #[arg(long, default_value_t = 8)]
    min_free_vram_gib: u64,

    /// Aging: seconds of waiting that earn one step of priority.
    ///
    /// The default is tuned to these workloads rather than to the controller's
    /// default, and deliberately so: aging only does anything when its interval
    /// is short relative to how long jobs actually wait, and a comparison run
    /// with an interval nobody reaches would show aging doing nothing and
    /// invite the conclusion that aging does nothing.
    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    aging_interval_secs: u32,

    /// Aging: priority gained per completed interval.
    #[arg(long, default_value_t = 5)]
    aging_increment: u32,

    /// Ignore which GPUs a job landed on: every placement then runs at the same
    /// speed. Use it to isolate the queue comparison from placement, and to
    /// demonstrate that without an execution model the placement strategies are
    /// indistinguishable.
    #[arg(long)]
    flat_execution: bool,

    /// Fraction of a multi-node job's step spent communicating, at the
    /// reference link speed.
    #[arg(long, default_value_t = 0.35, value_name = "F")]
    communication_share: f64,
}

#[derive(clap::Args, Debug)]
struct TraceArgs {
    #[arg(long, default_value = "A")]
    scenario: String,
    #[arg(long, default_value = "fifo")]
    queue: String,
    #[arg(long, default_value = "performance")]
    placement: String,
    /// Stop after this many events. 0 prints them all.
    #[arg(long, default_value_t = 60)]
    limit: usize,
}

fn config(min_free_vram_gib: u64) -> SchedulerConfig {
    SchedulerConfig {
        master_port: 29500,
        min_free_vram_b: min_free_vram_gib << 30,
        // The simulator's measurements are all taken at t=0 and the clock runs
        // for hours, so expiry would silently discard the whole fabric.
        network_max_age_s: 0,
        placement_weights: PlacementWeights::default(),
    }
}

fn pick(all: Vec<Scenario>, wanted: &[String]) -> Vec<Scenario> {
    if wanted.is_empty() {
        return all;
    }
    all.into_iter()
        .filter(|s| wanted.iter().any(|w| w.eq_ignore_ascii_case(s.letter)))
        .collect()
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::List => list(),
        Cmd::Run(args) => run(args),
        Cmd::Trace(args) => trace(args),
    }
}

fn list() -> Result<()> {
    println!("Scenarios:");
    for s in catalogue() {
        println!(
            "  {}  {:<22} {} jobs, {} GPUs -- {}",
            s.letter,
            s.workload.name,
            s.workload.jobs,
            s.cluster.total_gpus(),
            s.about
        );
    }
    println!(
        "\nQueue policies:     {}",
        ferro_sched::QUEUE_POLICIES.join(", ")
    );
    println!(
        "Placement policies: {}",
        ferro_sched::PLACEMENT_POLICIES.join(", ")
    );
    println!(
        "Dispatch modes:     {} (--with-strict, --with-reserved)",
        ferro_sched::DISPATCH_MODES.join(", ")
    );
    Ok(())
}

fn one_run(
    scenario: &Scenario,
    queue_name: &str,
    placement_name: &str,
    dispatch: Dispatch,
    cfg: &SchedulerConfig,
    tuning: &ferro_sched::QueueTuning,
    execution: &ExecutionModel,
) -> Result<(RunSummary, SimOutcome)> {
    let queue = ferro_sched::queue_policy(queue_name, tuning)
        .with_context(|| format!("queue policy `{queue_name}`"))?;
    let placement = ferro_sched::placement_policy(placement_name)
        .with_context(|| format!("placement policy `{placement_name}`"))?;

    let outcome = ferro_sim::sim::run(
        &scenario.workload,
        &scenario.cluster,
        &*queue,
        &*placement,
        cfg,
        dispatch,
        execution,
    );
    let metrics = summarise(&outcome);
    let meta = RunMetadata {
        scenario: scenario.workload.name.clone(),
        scenario_about: scenario.about.to_string(),
        queue_policy: queue_name.to_string(),
        placement_policy: placement_name.to_string(),
        dispatch: dispatch.label(),
        seed: scenario.workload.seed,
        jobs: scenario.workload.jobs,
        cluster_nodes: scenario.cluster.nodes.len(),
        cluster_gpus: scenario.cluster.total_gpus(),
        min_free_vram_gib: cfg.min_free_vram_b >> 30,
        git_commit: git_commit(),
        generated_unix_s: now_unix_s(),
    };
    Ok((RunSummary { meta, metrics }, outcome))
}

fn run(args: RunArgs) -> Result<()> {
    let cfg = config(args.min_free_vram_gib);
    let tuning = ferro_sched::QueueTuning {
        aging: ferro_sched::queue::aging::AgingConfig {
            interval_s: args.aging_interval_secs,
            increment: args.aging_increment,
            ceiling: ferro_sched::MAX_PRIORITY,
        },
        ..Default::default()
    };
    let execution = if args.flat_execution {
        ExecutionModel::flat()
    } else {
        ExecutionModel {
            communication_share: args.communication_share,
            ..Default::default()
        }
    };
    let scenarios = pick(catalogue(), &args.scenarios);
    anyhow::ensure!(!scenarios.is_empty(), "no scenario matched");

    let queues: Vec<String> = if args.queue.is_empty() {
        ferro_sched::QUEUE_POLICIES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        args.queue.clone()
    };
    let mut dispatches = vec![Dispatch::Opportunistic];
    if args.with_strict {
        dispatches.push(Dispatch::Strict);
    }
    if args.with_reserved {
        dispatches.push(Dispatch::Reserved);
    }

    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    let mut summaries: Vec<RunSummary> = Vec::new();
    let mut per_job = String::from(
        "scenario,queue_policy,placement_policy,dispatch,job_id,user,class,priority,gpus,\
         arrival_s,start_s,end_s,duration_s,nominal_duration_s,wait_s,turnaround_s,overtaken_by,failed,nodes\n",
    );

    for scenario in &scenarios {
        for placement in &args.placement {
            for queue in &queues {
                for dispatch in &dispatches {
                    let (summary, outcome) = one_run(
                        scenario, queue, placement, *dispatch, &cfg, &tuning, &execution,
                    )?;
                    for j in &outcome.jobs {
                        per_job.push_str(&format!(
                            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                            scenario.workload.name,
                            queue,
                            placement,
                            dispatch.label(),
                            j.id,
                            j.user,
                            j.class,
                            j.priority,
                            j.gpus,
                            j.arrival_s,
                            j.start_s.map(|v| v.to_string()).unwrap_or_default(),
                            j.end_s.map(|v| v.to_string()).unwrap_or_default(),
                            j.duration_s,
                            j.nominal_duration_s,
                            j.wait_s,
                            j.turnaround_s,
                            j.overtaken_by,
                            j.failed,
                            j.nodes_used.join(" ")
                        ));
                    }
                    println!(
                        "{:<20} {:<11} {:<12} {:<14} wait avg {:>7.0}s  p95 {:>7.0}s  \
                         util {:>5.1}%  gpu-fair {:.3}  wait-fair {:.3}  slowdown {:.2}x  starved {:>3}",
                        scenario.workload.name,
                        queue,
                        placement,
                        dispatch.label(),
                        summary.metrics.avg_wait_s,
                        summary.metrics.p95_wait_s,
                        summary.metrics.gpu_utilisation * 100.0,
                        summary.metrics.jain_fairness,
                        summary.metrics.wait_fairness,
                        summary.metrics.mean_slowdown,
                        summary.metrics.starved_jobs,
                    );
                    summaries.push(summary);
                }
            }
        }
    }

    write(&args.out.join("results.csv"), &per_job)?;
    write(&args.out.join("summary.csv"), &summary_csv(&summaries))?;
    write(
        &args.out.join("runs.json"),
        &serde_json::to_string_pretty(&summaries)?,
    )?;

    println!(
        "\n{} run(s) written to {}",
        summaries.len(),
        args.out.display()
    );
    Ok(())
}

/// The table the write-up reads from.
fn summary_csv(rows: &[RunSummary]) -> String {
    let mut out = String::from(
        "scenario,queue_policy,placement_policy,dispatch,seed,jobs,cluster_gpus,\
         jobs_completed,jobs_failed,jobs_never_started,\
         avg_wait_s,p50_wait_s,p95_wait_s,max_wait_s,\
         avg_turnaround_s,p95_turnaround_s,makespan_s,gpu_utilisation,throughput_per_hour,\
         jain_fairness,wait_fairness,heavy_user_wait_ratio,starved_jobs,jobs_overtaken_often,max_overtaken_by,\
         fragmentation,mean_slowdown,scheduler_us_per_pass,scheduling_passes,git_commit\n",
    );
    for r in rows {
        let m = &r.metrics;
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{},{:.4},{:.2},\
             {:.4},{:.4},{:.3},{},{},{},{:.3},{:.3},{:.1},{},{}\n",
            r.meta.scenario,
            r.meta.queue_policy,
            r.meta.placement_policy,
            r.meta.dispatch,
            r.meta.seed,
            r.meta.jobs,
            r.meta.cluster_gpus,
            m.jobs_completed,
            m.jobs_failed,
            m.jobs_never_started,
            m.avg_wait_s,
            m.p50_wait_s,
            m.p95_wait_s,
            m.max_wait_s,
            m.avg_turnaround_s,
            m.p95_turnaround_s,
            m.makespan_s,
            m.gpu_utilisation,
            m.throughput_per_hour,
            m.jain_fairness,
            m.wait_fairness,
            m.heavy_user_wait_ratio,
            m.starved_jobs,
            m.jobs_overtaken_often,
            m.max_overtaken_by,
            m.fragmentation,
            m.mean_slowdown,
            m.scheduler_us_per_pass,
            m.scheduling_passes,
            r.meta.git_commit,
        ));
    }
    out
}

fn write(path: &Path, contents: &str) -> Result<()> {
    let mut f =
        std::fs::File::create(path).with_context(|| format!("writing {}", path.display()))?;
    f.write_all(contents.as_bytes())?;
    Ok(())
}

fn trace(args: TraceArgs) -> Result<()> {
    let scenarios = pick(catalogue(), std::slice::from_ref(&args.scenario));
    let scenario = scenarios
        .first()
        .with_context(|| format!("no scenario `{}`", args.scenario))?;
    let cfg = config(8);
    let (_, outcome) = one_run(
        scenario,
        &args.queue,
        &args.placement,
        Dispatch::Opportunistic,
        &cfg,
        &ferro_sched::QueueTuning::default(),
        &ExecutionModel::default(),
    )?;

    let mut events = outcome.trace;
    events.sort_by_key(|e| e.t);
    let limit = if args.limit == 0 {
        events.len()
    } else {
        args.limit.min(events.len())
    };
    println!(
        "# {} under queue={} placement={}",
        scenario.workload.name, args.queue, args.placement
    );
    for e in events.iter().take(limit) {
        println!(
            "t={:<7} {:<14} {:<7} {:<8} {}",
            e.t, e.event, e.job, e.user, e.detail
        );
    }
    if limit < events.len() {
        println!("... {} more events", events.len() - limit);
    }
    Ok(())
}
