//! Paired, multi-seed evaluation of hard per-user GPU quotas.

use crate::git_commit;
use crate::metrics::{self, Metrics};
use crate::rng::Rng;
use crate::sim::{self, Dispatch, ExecutionModel};
use crate::workload::{ClusterSpec, JobClass, SimJob, UserSpec, WorkloadSpec};
use anyhow::{Context, Result};
use ferro_admission::QuotaTable;
use ferro_sched::{PlacementWeights, SchedulerConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;

const JOBS_PER_SCENARIO: usize = 240;
const MIN_DURATION_S: u64 = 120;
const MAX_DURATION_S: u64 = 240;
const QUOTAS: [Option<u32>; 4] = [None, Some(1), Some(2), Some(4)];

/// One workload definition and its paired generated jobs for all quota values.
#[derive(Debug, Clone)]
pub struct QuotaWorkload {
    pub scenario: String,
    pub description: String,
    pub heavy_user: Option<String>,
    pub workload_parameters: Value,
    pub spec: WorkloadSpec,
    pub jobs: Vec<SimJob>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuotaUserMetrics {
    pub jobs_submitted: usize,
    pub jobs_completed: usize,
    pub mean_wait_s: Option<f64>,
    pub p50_wait_s: Option<f64>,
    pub p95_wait_s: Option<f64>,
    pub gpu_seconds_received: f64,
    pub gpu_seconds_share: f64,
}

/// One deterministic row per (scenario, seed, quota) run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuotaRawRun {
    pub schema_version: String,
    pub scenario: String,
    pub scenario_description: String,
    pub seed: u64,
    pub jobs_submitted: usize,
    pub user_count: usize,
    pub cluster_gpus: usize,
    pub quota_gpus: Option<u32>,
    pub queue_policy: String,
    pub dispatch: String,
    pub placement_policy: String,
    pub workload_parameters: Value,
    pub git_commit: String,
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    pub jobs_never_started: usize,
    pub mean_wait_s: f64,
    pub p50_wait_s: f64,
    pub p95_wait_s: f64,
    pub mean_turnaround_s: f64,
    pub p95_turnaround_s: f64,
    pub makespan_s: i64,
    pub gpu_utilisation: f64,
    pub throughput_per_hour: f64,
    pub jain_gpu_seconds: f64,
    pub jain_mean_wait: f64,
    pub heavy_user: Option<String>,
    pub heavy_user_mean_wait_s: Option<f64>,
    pub heavy_user_p95_wait_s: Option<f64>,
    pub normal_users_mean_wait_s: Option<f64>,
    pub normal_users_p95_wait_s: Option<f64>,
    pub heavy_to_normal_wait_ratio: Option<f64>,
    pub quota_blocked_jobs: usize,
    pub quota_block_events: usize,
    pub quota_block_attempts: usize,
    pub quota_blocked_time_s: i64,
    pub hard_rejected_jobs: usize,
    pub quota_unused_gpu_seconds: f64,
    pub per_user: BTreeMap<String, QuotaUserMetrics>,
    pub per_job_quota: BTreeMap<String, crate::sim::QuotaJobMetrics>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Aggregate {
    pub n: usize,
    pub mean: f64,
    pub standard_deviation: Option<f64>,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct QuotaAggregateRow {
    pub scenario: String,
    pub quota_gpus: Option<u32>,
    pub metric: String,
    #[serde(flatten)]
    pub aggregate: Aggregate,
}

fn user(name: &str, weight: f64) -> UserSpec {
    UserSpec {
        name: name.into(),
        weight,
        priority: 50,
    }
}

fn one_gpu_class() -> JobClass {
    JobClass {
        name: "single-gpu".into(),
        weight: 1.0,
        nodes: 1,
        gpus_per_node: 1,
        min_duration_s: MIN_DURATION_S,
        max_duration_s: MAX_DURATION_S,
    }
}

fn spec(
    name: &str,
    seed: u64,
    jobs: usize,
    mean_interarrival_s: f64,
    users: Vec<UserSpec>,
) -> WorkloadSpec {
    WorkloadSpec {
        name: name.into(),
        seed,
        jobs,
        mean_interarrival_s,
        users,
        classes: vec![one_gpu_class()],
        estimate_fraction: 1.0,
        estimate_error: 0.0,
        fail_node: None,
    }
}

/// The four main scenarios. The resulting job list is created once for each
/// seed, then reused unchanged for each quota in the paired comparison.
pub fn workloads_for_seed(seed: u64) -> Vec<QuotaWorkload> {
    let balanced = spec(
        "A-balanced",
        seed,
        JOBS_PER_SCENARIO,
        5.0,
        vec![
            user("alice", 1.0),
            user("bob", 1.0),
            user("carol", 1.0),
            user("dave", 1.0),
        ],
    );
    let balanced_jobs = balanced.generate();

    let heavy = spec(
        "B-heavy",
        seed,
        JOBS_PER_SCENARIO,
        5.0,
        vec![
            user("hog", 7.0),
            user("alice", 1.0),
            user("bob", 1.0),
            user("carol", 1.0),
        ],
    );
    let heavy_jobs = heavy.generate();

    let mut burst = spec(
        "C-burst",
        seed,
        250,
        5.0,
        vec![
            user("hog", 1.0),
            user("alice", 1.0),
            user("bob", 1.0),
            user("carol", 1.0),
        ],
    );
    let mut burst_jobs = burst.generate();
    let mut arrivals = Rng::seeded(seed ^ 0xa5a5_5a5a_d3c1_b7e9);
    let mut heavy_clock = 0.0;
    let mut light_counts: HashMap<&str, usize> = HashMap::new();
    for (index, job) in burst_jobs.iter_mut().enumerate() {
        if index < 160 {
            heavy_clock += arrivals.exponential(0.15);
            job.user = "hog".into();
            job.arrival_s = heavy_clock as i64;
        } else {
            let slot = index - 160;
            let name = ["alice", "bob", "carol"][slot / 30];
            let per_user = light_counts.entry(name).or_default();
            job.user = name.into();
            job.arrival_s = 20 + (*per_user as i64 * 3) + arrivals.range(0, 2) as i64;
            *per_user += 1;
        }
        job.priority = 50;
    }
    burst_jobs.sort_by_key(|job| job.arrival_s);
    burst.jobs = burst_jobs.len();

    let single = spec(
        "D-single-user",
        seed,
        JOBS_PER_SCENARIO,
        5.0,
        vec![user("hog", 1.0)],
    );
    let single_jobs = single.generate();

    vec![
        QuotaWorkload {
            scenario: "A-balanced".into(),
            description: "four users with equal Poisson arrivals and one-GPU jobs".into(),
            heavy_user: None,
            workload_parameters: json!({
                "jobs": JOBS_PER_SCENARIO,
                "arrival_process": "Poisson; exponential interarrival mean 5 s",
                "user_arrival_weights": {"alice": 1, "bob": 1, "carol": 1, "dave": 1},
                "job_gpu_count": 1,
                "duration_s_uniform_inclusive": [MIN_DURATION_S, MAX_DURATION_S]
            }),
            spec: balanced,
            jobs: balanced_jobs,
        },
        QuotaWorkload {
            scenario: "B-heavy".into(),
            description: "one user receives 70% of Poisson arrivals; three users share 30%".into(),
            heavy_user: Some("hog".into()),
            workload_parameters: json!({
                "jobs": JOBS_PER_SCENARIO,
                "arrival_process": "Poisson; exponential interarrival mean 5 s",
                "heavy_arrival_share": 0.70,
                "light_users_total_arrival_share": 0.30,
                "user_arrival_weights": {"hog": 7, "alice": 1, "bob": 1, "carol": 1},
                "job_gpu_count": 1,
                "duration_s_uniform_inclusive": [MIN_DURATION_S, MAX_DURATION_S]
            }),
            spec: heavy,
            jobs: heavy_jobs,
        },
        QuotaWorkload {
            scenario: "C-burst".into(),
            description: "160 early heavy-user jobs followed by three users arriving from t=20 s"
                .into(),
            heavy_user: Some("hog".into()),
            workload_parameters: json!({
                "jobs": 250,
                "heavy_jobs": 160,
                "light_jobs_per_user": 30,
                "heavy_arrivals": "seeded exponential interarrival mean 0.15 s, cumulative arrival times floored to the simulator's integer-second clock",
                "light_arrivals": "each user starts at 20 s, then 3 s intervals plus seeded integer jitter [0,2] s",
                "job_gpu_count": 1,
                "duration_s_uniform_inclusive": [MIN_DURATION_S, MAX_DURATION_S]
            }),
            spec: burst,
            jobs: burst_jobs,
        },
        QuotaWorkload {
            scenario: "D-single-user".into(),
            description: "one user's queue is the only source of work on the 8-GPU cluster".into(),
            heavy_user: Some("hog".into()),
            workload_parameters: json!({
                "jobs": JOBS_PER_SCENARIO,
                "arrival_process": "Poisson; exponential interarrival mean 5 s",
                "active_users": ["hog"],
                "job_gpu_count": 1,
                "duration_s_uniform_inclusive": [MIN_DURATION_S, MAX_DURATION_S]
            }),
            spec: single,
            jobs: single_jobs,
        },
    ]
}

/// Run one fixed-policy matrix cell and retain only deterministic metrics.
pub fn run_case(
    case: &QuotaWorkload,
    quota_gpus: Option<u32>,
    source_commit: &str,
) -> Result<QuotaRawRun> {
    let cluster = ClusterSpec::homogeneous(2, 4);
    let config = SchedulerConfig {
        master_port: 29500,
        min_free_vram_b: 8 << 30,
        network_max_age_s: 0,
        placement_weights: PlacementWeights::default(),
    };
    let policy_tuning = ferro_sched::QueueTuning::default();
    let queue = ferro_sched::queue_policy("fifo", &policy_tuning)?;
    let placement = ferro_sched::placement_policy("performance")?;
    let quota_table = match quota_gpus {
        Some(gpus) => {
            QuotaTable::from_specs(case.spec.users.iter().map(|user| (user.name.clone(), gpus)))?
        }
        None => QuotaTable::default(),
    };
    let result = sim::run_with_jobs_quota(
        &case.spec,
        case.jobs.clone(),
        &cluster,
        &*queue,
        &*placement,
        &config,
        Dispatch::Opportunistic,
        &ExecutionModel::default(),
        &quota_table,
    );
    let metrics = metrics::summarise(&result.simulation);
    let per_user = per_user_metrics(&case.spec, &result.simulation.jobs, &metrics);
    let heavy_stats = case
        .heavy_user
        .as_deref()
        .and_then(|name| per_user.get(name));
    let normal_stats: Vec<&QuotaUserMetrics> = per_user
        .iter()
        .filter(|(name, _)| Some(name.as_str()) != case.heavy_user.as_deref())
        .map(|(_, metrics)| metrics)
        .collect();
    let normal_mean = optional_average(normal_stats.iter().filter_map(|m| m.mean_wait_s));
    let normal_p95 = optional_average(normal_stats.iter().filter_map(|m| m.p95_wait_s));
    let heavy_mean = heavy_stats.and_then(|m| m.mean_wait_s);
    let heavy_p95 = heavy_stats.and_then(|m| m.p95_wait_s);
    let heavy_ratio = match (heavy_mean, normal_mean) {
        (Some(heavy), Some(normal)) if normal > 0.0 => Some(heavy / normal),
        _ => None,
    };

    Ok(QuotaRawRun {
        schema_version: "quota-raw-v1".into(),
        scenario: case.scenario.clone(),
        scenario_description: case.description.clone(),
        seed: case.spec.seed,
        jobs_submitted: result.simulation.jobs.len(),
        user_count: case.spec.users.len(),
        cluster_gpus: cluster.total_gpus(),
        quota_gpus,
        queue_policy: "fifo".into(),
        dispatch: Dispatch::Opportunistic.label().into(),
        placement_policy: "performance".into(),
        workload_parameters: case.workload_parameters.clone(),
        git_commit: source_commit.into(),
        jobs_completed: metrics.jobs_completed,
        jobs_failed: metrics.jobs_failed,
        jobs_never_started: metrics.jobs_never_started,
        mean_wait_s: metrics.avg_wait_s,
        p50_wait_s: metrics.p50_wait_s,
        p95_wait_s: metrics.p95_wait_s,
        mean_turnaround_s: metrics.avg_turnaround_s,
        p95_turnaround_s: metrics.p95_turnaround_s,
        makespan_s: metrics.makespan_s,
        gpu_utilisation: metrics.gpu_utilisation,
        throughput_per_hour: metrics.throughput_per_hour,
        jain_gpu_seconds: metrics.jain_fairness,
        jain_mean_wait: metrics.wait_fairness,
        heavy_user: case.heavy_user.clone(),
        heavy_user_mean_wait_s: heavy_mean,
        heavy_user_p95_wait_s: heavy_p95,
        normal_users_mean_wait_s: normal_mean,
        normal_users_p95_wait_s: normal_p95,
        heavy_to_normal_wait_ratio: heavy_ratio,
        quota_blocked_jobs: result.quota.blocked_jobs,
        quota_block_events: result.quota.block_events,
        quota_block_attempts: result.quota.block_attempts,
        quota_blocked_time_s: result.quota.blocked_time_s,
        hard_rejected_jobs: result.quota.hard_rejected_jobs,
        quota_unused_gpu_seconds: result.quota.quota_unused_gpu_seconds,
        per_user,
        per_job_quota: result.quota.per_job,
    })
}

fn per_user_metrics(
    workload: &WorkloadSpec,
    jobs: &[sim::JobRecord],
    summary: &Metrics,
) -> BTreeMap<String, QuotaUserMetrics> {
    let total_gpu_seconds: f64 = summary.per_user_gpu_seconds.values().sum();
    workload
        .users
        .iter()
        .map(|user| {
            let records: Vec<_> = jobs.iter().filter(|job| job.user == user.name).collect();
            let mut waits: Vec<f64> = records
                .iter()
                .filter(|job| job.start_s.is_some())
                .map(|job| job.wait_s as f64)
                .collect();
            waits.sort_by(f64::total_cmp);
            let gpu_seconds = summary
                .per_user_gpu_seconds
                .get(&user.name)
                .copied()
                .unwrap_or(0.0);
            (
                user.name.clone(),
                QuotaUserMetrics {
                    jobs_submitted: records.len(),
                    jobs_completed: records
                        .iter()
                        .filter(|job| job.end_s.is_some() && !job.failed)
                        .count(),
                    mean_wait_s: (!waits.is_empty()).then(|| average(&waits)),
                    p50_wait_s: (!waits.is_empty()).then(|| percentile(&waits, 50.0)),
                    p95_wait_s: (!waits.is_empty()).then(|| percentile(&waits, 95.0)),
                    gpu_seconds_received: gpu_seconds,
                    gpu_seconds_share: if total_gpu_seconds > 0.0 {
                        gpu_seconds / total_gpu_seconds
                    } else {
                        0.0
                    },
                },
            )
        })
        .collect()
}

fn optional_average(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    (!values.is_empty()).then(|| average(&values))
}

fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// Aggregate per-seed values with a two-sided Student-t 95% interval.
/// For df > 30, 1.96 is used as the documented normal approximation.
pub fn aggregate_samples(values: &[f64]) -> Result<Aggregate> {
    anyhow::ensure!(!values.is_empty(), "cannot aggregate an empty sample");
    anyhow::ensure!(
        values.iter().all(|v| v.is_finite()),
        "sample values must be finite"
    );
    let mean = average(values);
    if values.len() == 1 {
        return Ok(Aggregate {
            n: 1,
            mean,
            standard_deviation: None,
            ci_low: None,
            ci_high: None,
        });
    }
    let df = values.len() - 1;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / df as f64;
    let sd = variance.sqrt();
    let margin = t_critical_95(df) * sd / (values.len() as f64).sqrt();
    Ok(Aggregate {
        n: values.len(),
        mean,
        standard_deviation: Some(sd),
        ci_low: Some(mean - margin),
        ci_high: Some(mean + margin),
    })
}

fn t_critical_95(df: usize) -> f64 {
    const VALUES: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if df == 0 {
        0.0
    } else if df <= VALUES.len() {
        VALUES[df - 1]
    } else {
        1.96
    }
}

fn metric_values(run: &QuotaRawRun) -> BTreeMap<String, Option<f64>> {
    let mut values = BTreeMap::from([
        ("jobs_completed".into(), Some(run.jobs_completed as f64)),
        (
            "jobs_never_started".into(),
            Some(run.jobs_never_started as f64),
        ),
        ("mean_wait_s".into(), Some(run.mean_wait_s)),
        ("p50_wait_s".into(), Some(run.p50_wait_s)),
        ("p95_wait_s".into(), Some(run.p95_wait_s)),
        ("mean_turnaround_s".into(), Some(run.mean_turnaround_s)),
        ("p95_turnaround_s".into(), Some(run.p95_turnaround_s)),
        ("makespan_s".into(), Some(run.makespan_s as f64)),
        ("gpu_utilisation".into(), Some(run.gpu_utilisation)),
        (
            "gpu_utilisation_percent".into(),
            Some(run.gpu_utilisation * 100.0),
        ),
        ("throughput_per_hour".into(), Some(run.throughput_per_hour)),
        ("jain_gpu_seconds".into(), Some(run.jain_gpu_seconds)),
        ("jain_mean_wait".into(), Some(run.jain_mean_wait)),
        ("heavy_user_mean_wait_s".into(), run.heavy_user_mean_wait_s),
        ("heavy_user_p95_wait_s".into(), run.heavy_user_p95_wait_s),
        (
            "normal_users_mean_wait_s".into(),
            run.normal_users_mean_wait_s,
        ),
        (
            "normal_users_p95_wait_s".into(),
            run.normal_users_p95_wait_s,
        ),
        (
            "heavy_to_normal_wait_ratio".into(),
            run.heavy_to_normal_wait_ratio,
        ),
        (
            "quota_blocked_jobs".into(),
            Some(run.quota_blocked_jobs as f64),
        ),
        (
            "quota_block_events".into(),
            Some(run.quota_block_events as f64),
        ),
        (
            "quota_block_attempts".into(),
            Some(run.quota_block_attempts as f64),
        ),
        (
            "quota_blocked_time_s".into(),
            Some(run.quota_blocked_time_s as f64),
        ),
        (
            "hard_rejected_jobs".into(),
            Some(run.hard_rejected_jobs as f64),
        ),
        (
            "quota_unused_gpu_seconds".into(),
            Some(run.quota_unused_gpu_seconds),
        ),
    ]);
    for (user, stats) in &run.per_user {
        values.insert(format!("user.{user}.mean_wait_s"), stats.mean_wait_s);
        values.insert(format!("user.{user}.p50_wait_s"), stats.p50_wait_s);
        values.insert(format!("user.{user}.p95_wait_s"), stats.p95_wait_s);
        values.insert(
            format!("user.{user}.gpu_seconds_received"),
            Some(stats.gpu_seconds_received),
        );
        values.insert(
            format!("user.{user}.gpu_seconds_share"),
            Some(stats.gpu_seconds_share),
        );
        values.insert(
            format!("user.{user}.jobs_completed"),
            Some(stats.jobs_completed as f64),
        );
    }
    values
}

pub fn aggregate_runs(runs: &[QuotaRawRun]) -> Result<Vec<QuotaAggregateRow>> {
    anyhow::ensure!(!runs.is_empty(), "cannot aggregate an empty run set");
    let mut samples: BTreeMap<(String, Option<u32>, String), Vec<f64>> = BTreeMap::new();
    for run in runs {
        for (metric, value) in metric_values(run) {
            if let Some(value) = value {
                samples
                    .entry((run.scenario.clone(), run.quota_gpus, metric))
                    .or_default()
                    .push(value);
            }
        }
    }
    samples
        .into_iter()
        .map(|((scenario, quota_gpus, metric), values)| {
            Ok(QuotaAggregateRow {
                scenario,
                quota_gpus,
                metric,
                aggregate: aggregate_samples(&values)?,
            })
        })
        .collect()
}

/// Run the full seed/scenario/quota matrix and write auditable raw and summary
/// files. The source commit is captured before creating or replacing outputs.
pub fn run_sweep(out_dir: &Path, seed_count: usize) -> Result<Vec<QuotaRawRun>> {
    anyhow::ensure!(seed_count > 0, "seed count must be positive");
    let source_commit = git_commit();
    let mut raw = Vec::with_capacity(seed_count * 4 * QUOTAS.len());
    for seed in 1001..(1001 + seed_count as u64) {
        for case in workloads_for_seed(seed) {
            for quota in QUOTAS {
                let run = run_case(&case, quota, &source_commit).with_context(|| {
                    format!("running {} seed {seed} quota {quota:?}", case.scenario)
                })?;
                println!(
                    "{} seed {:>4} quota {:>7}: wait {:>7.1}s p95 {:>7.1}s util {:>5.1}% throughput {:>6.2}/h",
                    run.scenario,
                    seed,
                    quota.map(|n| n.to_string()).unwrap_or_else(|| "unlimited".into()),
                    run.mean_wait_s,
                    run.p95_wait_s,
                    run.gpu_utilisation * 100.0,
                    run.throughput_per_hour
                );
                raw.push(run);
            }
        }
    }
    let raw_json = serde_json::to_string_pretty(&raw)?;
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    write(&out_dir.join("raw.json"), &raw_json)?;
    write(&out_dir.join("raw.csv"), &raw_csv(&raw)?)?;
    // Aggregate the exact serialized raw representation so `quota-aggregate`
    // reproduces the same summary rather than using pre-serialization floats.
    let persisted_raw: Vec<QuotaRawRun> = serde_json::from_str(&raw_json)?;
    let summary = aggregate_runs(&persisted_raw)?;
    write_summary(out_dir, &summary)?;
    println!(
        "\n{} paired runs across {} seeds written to {}",
        raw.len(),
        seed_count,
        out_dir.display()
    );
    Ok(raw)
}

/// Recompute aggregate files from previously generated raw JSON.
pub fn summarise_file(raw_path: &Path, out_dir: &Path) -> Result<Vec<QuotaAggregateRow>> {
    let raw: Vec<QuotaRawRun> = serde_json::from_slice(
        &std::fs::read(raw_path).with_context(|| format!("reading {}", raw_path.display()))?,
    )
    .with_context(|| format!("parsing raw quota runs from {}", raw_path.display()))?;
    let summary = aggregate_runs(&raw)?;
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    write_summary(out_dir, &summary)?;
    Ok(summary)
}

fn write_summary(out_dir: &Path, summary: &[QuotaAggregateRow]) -> Result<()> {
    write(
        &out_dir.join("summary.json"),
        &serde_json::to_string_pretty(summary)?,
    )?;
    write(&out_dir.join("summary.csv"), &summary_csv(summary))
}

fn raw_csv(runs: &[QuotaRawRun]) -> Result<String> {
    let mut csv = String::from(
        "schema_version,scenario,scenario_description,seed,jobs_submitted,user_count,cluster_gpus,quota_gpus,queue_policy,dispatch,placement_policy,workload_parameters_json,git_commit,jobs_completed,jobs_failed,jobs_never_started,mean_wait_s,p50_wait_s,p95_wait_s,mean_turnaround_s,p95_turnaround_s,makespan_s,gpu_utilisation,throughput_per_hour,jain_gpu_seconds,jain_mean_wait,heavy_user,heavy_user_mean_wait_s,heavy_user_p95_wait_s,normal_users_mean_wait_s,normal_users_p95_wait_s,heavy_to_normal_wait_ratio,quota_blocked_jobs,quota_block_events,quota_block_attempts,quota_blocked_time_s,hard_rejected_jobs,quota_unused_gpu_seconds,per_user_json,per_job_quota_json\n",
    );
    for run in runs {
        let values = [
            csv_field(&run.schema_version),
            csv_field(&run.scenario),
            csv_field(&run.scenario_description),
            run.seed.to_string(),
            run.jobs_submitted.to_string(),
            run.user_count.to_string(),
            run.cluster_gpus.to_string(),
            run.quota_gpus.map(|v| v.to_string()).unwrap_or_default(),
            csv_field(&run.queue_policy),
            csv_field(&run.dispatch),
            csv_field(&run.placement_policy),
            csv_field(&serde_json::to_string(&run.workload_parameters)?),
            csv_field(&run.git_commit),
            run.jobs_completed.to_string(),
            run.jobs_failed.to_string(),
            run.jobs_never_started.to_string(),
            run.mean_wait_s.to_string(),
            run.p50_wait_s.to_string(),
            run.p95_wait_s.to_string(),
            run.mean_turnaround_s.to_string(),
            run.p95_turnaround_s.to_string(),
            run.makespan_s.to_string(),
            run.gpu_utilisation.to_string(),
            run.throughput_per_hour.to_string(),
            run.jain_gpu_seconds.to_string(),
            run.jain_mean_wait.to_string(),
            csv_field(run.heavy_user.as_deref().unwrap_or("")),
            optional_field(run.heavy_user_mean_wait_s),
            optional_field(run.heavy_user_p95_wait_s),
            optional_field(run.normal_users_mean_wait_s),
            optional_field(run.normal_users_p95_wait_s),
            optional_field(run.heavy_to_normal_wait_ratio),
            run.quota_blocked_jobs.to_string(),
            run.quota_block_events.to_string(),
            run.quota_block_attempts.to_string(),
            run.quota_blocked_time_s.to_string(),
            run.hard_rejected_jobs.to_string(),
            run.quota_unused_gpu_seconds.to_string(),
            csv_field(&serde_json::to_string(&run.per_user)?),
            csv_field(&serde_json::to_string(&run.per_job_quota)?),
        ];
        csv.push_str(&values.join(","));
        csv.push('\n');
    }
    Ok(csv)
}

fn summary_csv(rows: &[QuotaAggregateRow]) -> String {
    let mut csv =
        String::from("scenario,quota_gpus,metric,n,mean,standard_deviation,ci95_low,ci95_high\n");
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            csv_field(&row.scenario),
            row.quota_gpus.map(|v| v.to_string()).unwrap_or_default(),
            csv_field(&row.metric),
            row.aggregate.n,
            row.aggregate.mean,
            optional_field(row.aggregate.standard_deviation),
            optional_field(row.aggregate.ci_low),
            optional_field(row.aggregate.ci_high),
        ));
    }
    csv
}

fn optional_field(value: Option<f64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write(path: &Path, contents: &str) -> Result<()> {
    let mut file =
        std::fs::File::create(path).with_context(|| format!("writing {}", path.display()))?;
    file.write_all(contents.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_workloads_are_seeded_and_cover_the_four_required_shapes() {
        let first = workloads_for_seed(17);
        let second = workloads_for_seed(17);
        let names: Vec<_> = first.iter().map(|case| case.scenario.as_str()).collect();
        assert_eq!(names, ["A-balanced", "B-heavy", "C-burst", "D-single-user"]);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(
                serde_json::to_string(&a.jobs).unwrap(),
                serde_json::to_string(&b.jobs).unwrap()
            );
            assert!(a.jobs.iter().all(|job| job.gpus() == 1));
        }
    }

    #[test]
    fn aggregation_reports_mean_sample_sd_and_student_t_confidence_interval() {
        let result = aggregate_samples(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(result.n, 4);
        assert!((result.mean - 2.5).abs() < 1e-12);
        assert!((result.standard_deviation.unwrap() - (5.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        let margin = result.ci_high.unwrap() - result.mean;
        assert!((margin - 2.054).abs() < 0.002);
    }

    #[test]
    fn empty_and_single_sample_summaries_are_explicit() {
        assert!(aggregate_samples(&[]).is_err());
        let one = aggregate_samples(&[7.0]).unwrap();
        assert_eq!(one.n, 1);
        assert_eq!(one.standard_deviation, None);
        assert_eq!(one.ci_low, None);
        assert_eq!(one.ci_high, None);
    }

    #[test]
    fn each_raw_run_contains_the_reproduction_configuration() {
        let case = workloads_for_seed(23).remove(1);
        let raw = run_case(&case, Some(2), "abc123").unwrap();
        assert_eq!(raw.seed, 23);
        assert_eq!(raw.scenario, "B-heavy");
        assert_eq!(raw.cluster_gpus, 8);
        assert_eq!(raw.quota_gpus, Some(2));
        assert_eq!(raw.queue_policy, "fifo");
        assert_eq!(raw.dispatch, "opportunistic");
        assert_eq!(raw.placement_policy, "performance");
        assert_eq!(raw.git_commit, "abc123");
        assert!(raw.workload_parameters.get("heavy_arrival_share").is_some());
        let encoded = serde_json::to_vec(std::slice::from_ref(&raw)).unwrap();
        let decoded: Vec<QuotaRawRun> = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded[0].scenario, raw.scenario);
        assert_eq!(decoded[0].seed, raw.seed);
        assert_eq!(decoded[0].quota_gpus, raw.quota_gpus);
        let same_raw_again: Vec<QuotaRawRun> = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            aggregate_runs(&decoded).unwrap(),
            aggregate_runs(&same_raw_again).unwrap()
        );
    }

    #[test]
    fn a_single_user_with_a_small_quota_leaves_measurable_gpu_capacity_idle() {
        let case = workloads_for_seed(31).remove(3);
        let unlimited = run_case(&case, None, "test").unwrap();
        let limited = run_case(&case, Some(2), "test").unwrap();
        assert!(limited.gpu_utilisation < unlimited.gpu_utilisation * 0.5);
        assert!(limited.quota_unused_gpu_seconds > 0.0);
        assert!(limited.quota_blocked_jobs > 0);
    }
}
