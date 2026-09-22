//! The discrete-event simulator.
//!
//! It drives the *same* `QueuePolicy` and `PlacementPolicy` objects the
//! controller runs, over the same `NodeState` snapshot type, with the same
//! reservation semantics. Nothing about scheduling is reimplemented here; if a
//! policy behaves differently in simulation than on the cluster, that is a bug
//! in this file and not a finding.
//!
//! ## What the dispatcher does, and why it matters
//!
//! The controller's `run_queue` walks the whole waiting list each tick and
//! starts anything that fits, not just the head. That is *opportunistic*
//! dispatch: it already backfills, without reservations. The consequence is
//! worth being explicit about, because it is a real property of the system
//! rather than a modelling choice -- a four-GPU job can be passed over
//! indefinitely by a stream of one-GPU jobs, which is precisely the case
//! reservation (§34) exists to fix.
//!
//! So [`Dispatch`] offers both, and the default is what FerroGrid actually
//! does. A simulator that modelled an idealised strict FIFO would produce
//! numbers about a scheduler nobody is running.

use crate::workload::{ClusterSpec, SimJob, WorkloadSpec};
use ferro_proto::{JobPlan, NodeState};
use ferro_sched::queue::{QueueContext, QueuePolicy, QueuedJob, UsageSnapshot, UserUsage};
use ferro_sched::{NetworkSnapshot, PlacementPolicy, PlacementRequest, SchedulerConfig, Shape};
use serde::Serialize;
use std::collections::HashMap;

/// How a placement decision turns into a running time.
///
/// **This is the simulator's one substantive model assumption, and every
/// placement comparison rests on it.** Without it a job takes the same time
/// wherever it lands, and all five placement strategies produce byte-identical
/// results -- which says nothing about placement and everything about the
/// model.
///
/// Two effects, both of them consequences of how synchronous data-parallel
/// training actually works:
///
/// * **The slowest rank sets the pace.** A collective steps together, so a set
///   of GPUs runs at the speed of its worst member, not its average. This is
///   why mixing a 82 TFLOP/s card with a 35 TFLOP/s one wastes the fast one,
///   and why the homogeneity axis exists at all.
/// * **Crossing the network costs.** A fraction of each step is spent in
///   all-gather and reduce-scatter; halve the link and that fraction doubles.
///   FerroGrid measured this on real hardware: 1 GbE between nodes cost about
///   55x throughput against a single GPU (see the README), which is the scale
///   these defaults are set to reproduce.
///
/// What it does **not** model: PCIe contention between jobs on one node,
/// container startup, data loading, or any of the reasons a real job misses
/// its own estimate. Durations here are clean.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionModel {
    /// The GPU throughput a workload's nominal durations are quoted at.
    pub reference_tflops: f64,
    /// The link speed those durations assume for a multi-node job.
    pub reference_mbps: f64,
    /// Fraction of a multi-node job's step spent communicating, at the
    /// reference link speed. 0 disables the network effect entirely.
    pub communication_share: f64,
}

impl Default for ExecutionModel {
    fn default() -> Self {
        Self {
            reference_tflops: 82.0,
            reference_mbps: 940.0,
            communication_share: 0.35,
        }
    }
}

impl ExecutionModel {
    /// A model that ignores placement, for isolating the queue comparison from
    /// it. Useful precisely because it makes the placement policies identical
    /// again, which is a fact worth being able to demonstrate on purpose.
    pub fn flat() -> Self {
        Self {
            reference_tflops: 0.0,
            reference_mbps: 0.0,
            communication_share: 0.0,
        }
    }

    /// How long a job of nominal duration `nominal` takes on this placement.
    pub fn duration(
        &self,
        nominal: i64,
        slowest_gpu_tflops: f64,
        slowest_hop_mbps: Option<f64>,
    ) -> i64 {
        if self.reference_tflops <= 0.0 {
            return nominal.max(1);
        }
        let compute = if slowest_gpu_tflops > 0.0 {
            self.reference_tflops / slowest_gpu_tflops
        } else {
            1.0
        };
        let network = match slowest_hop_mbps {
            Some(mbps) if mbps > 0.0 && self.reference_mbps > 0.0 => {
                let ratio = (self.reference_mbps / mbps).max(1.0);
                (1.0 - self.communication_share) + self.communication_share * ratio
            }
            // Single node, or a hop nobody measured: no penalty is applied, and
            // the placement score reports the missing measurement separately.
            _ => 1.0,
        };
        ((nominal as f64) * compute * network).round().max(1.0) as i64
    }
}

/// How the dispatcher treats a job at the head of the queue that does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Dispatch {
    /// Keep walking the ranked list and start anything that fits. What the
    /// controller does today.
    Opportunistic,
    /// Stop at the first job that does not fit, so nothing overtakes it.
    /// Classic FIFO behaviour, with the head-of-line blocking that comes with
    /// it. Useful as a comparison point, not as a recommendation.
    Strict,
}

impl Dispatch {
    pub fn label(self) -> &'static str {
        match self {
            Dispatch::Opportunistic => "opportunistic",
            Dispatch::Strict => "strict",
        }
    }
}

/// What happened to one job.
#[derive(Debug, Clone, Serialize)]
pub struct JobRecord {
    pub id: String,
    pub user: String,
    pub class: String,
    pub priority: u32,
    pub gpus: u32,
    pub arrival_s: i64,
    /// `None` if it never started, which happens when a node failure takes the
    /// only hardware that could have run it.
    pub start_s: Option<i64>,
    pub end_s: Option<i64>,
    /// What it actually ran for on the GPUs it was given.
    pub duration_s: i64,
    /// What it would have run for on reference hardware. The ratio of the two
    /// is what the placement decision cost this job.
    pub nominal_duration_s: i64,
    pub wait_s: i64,
    pub turnaround_s: i64,
    /// How many jobs that arrived *after* this one had already started before
    /// it did. The direct measure of being passed over.
    pub overtaken_by: u32,
    pub nodes_used: Vec<String>,
    pub failed: bool,
}

/// One line of the scheduler trace (§57), enough to draw a Gantt chart.
#[derive(Debug, Clone, Serialize)]
pub struct TraceEvent {
    pub t: i64,
    pub event: &'static str,
    pub job: String,
    pub user: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimOutcome {
    pub jobs: Vec<JobRecord>,
    pub trace: Vec<TraceEvent>,
    /// Total wall-clock nanoseconds spent inside the policies. The scheduler's
    /// own overhead (§48), measured rather than assumed negligible.
    pub scheduler_ns: u128,
    pub scheduling_passes: u64,
    /// Time-weighted mean of free GPUs that no single node could offer as one
    /// block -- capacity that exists but cannot take a single-node job.
    pub fragmentation: f64,
    pub total_gpus: usize,
}

struct Running {
    job: SimJob,
    plan: JobPlan,
    start_s: i64,
    end_s: i64,
}

/// Mark or clear a plan's GPUs on the mutable cluster view.
fn set_allocation(nodes: &mut [NodeState], plan: &JobPlan, job_id: &str) {
    for p in &plan.placements {
        for node in nodes.iter_mut() {
            let Some(info) = node.info.as_mut() else {
                continue;
            };
            if info.node_id != p.node_id {
                continue;
            }
            for gpu in info.gpus.iter_mut() {
                if p.gpu_indices.contains(&gpu.index) {
                    gpu.allocated_job_id = job_id.to_string();
                }
            }
        }
    }
}

fn clear_allocation(nodes: &mut [NodeState], job_id: &str) {
    for node in nodes.iter_mut() {
        let Some(info) = node.info.as_mut() else {
            continue;
        };
        for gpu in info.gpus.iter_mut() {
            if gpu.allocated_job_id == job_id {
                gpu.allocated_job_id.clear();
            }
        }
    }
}

/// The slowest card in a plan, which is the pace the whole job runs at.
fn slowest_gpu_tflops(nodes: &[NodeState], plan: &JobPlan) -> f64 {
    let mut slowest = f64::INFINITY;
    for p in &plan.placements {
        let Some(info) = nodes
            .iter()
            .filter_map(|n| n.info.as_ref())
            .find(|i| i.node_id == p.node_id)
        else {
            continue;
        };
        for gpu in info
            .gpus
            .iter()
            .filter(|g| p.gpu_indices.contains(&g.index))
        {
            if gpu.bench_tflops > 0.0 {
                slowest = slowest.min(gpu.bench_tflops);
            }
        }
    }
    if slowest.is_finite() {
        slowest
    } else {
        0.0
    }
}

/// The slowest hop a multi-node plan has to cross, where it is known.
fn slowest_hop(network: &NetworkSnapshot, plan: &JobPlan, now: i64) -> Option<f64> {
    if plan.placements.len() < 2 {
        return None;
    }
    let ids: Vec<&str> = plan.placements.iter().map(|p| p.node_id.as_str()).collect();
    network.slowest_among(&ids, now, 0)
}

/// Free GPUs that cannot be gathered into one block on any single node.
fn stranded(nodes: &[NodeState], min_free_b: u64) -> usize {
    let per_node: Vec<usize> = nodes
        .iter()
        .filter(|n| n.healthy)
        .map(|n| ferro_sched::placement::engine::placeable(n, min_free_b).len())
        .collect();
    let total: usize = per_node.iter().sum();
    let largest = per_node.iter().copied().max().unwrap_or(0);
    total.saturating_sub(largest)
}

/// Run one workload under one pair of policies.
pub fn run(
    workload: &WorkloadSpec,
    cluster: &ClusterSpec,
    queue_policy: &dyn QueuePolicy,
    placement: &dyn PlacementPolicy,
    config: &SchedulerConfig,
    dispatch: Dispatch,
    execution: &ExecutionModel,
) -> SimOutcome {
    let mut nodes = cluster.to_nodes();
    let mut network = NetworkSnapshot::default();
    for link in &cluster.links {
        network.record(&link.from, &link.to, link.mbps, 0);
    }

    let mut arriving: Vec<SimJob> = workload.generate();
    arriving.reverse(); // pop from the back, cheapest way to take them in order

    let mut waiting: Vec<(u64, SimJob)> = Vec::new(); // (submission sequence, job)
    let mut running: Vec<Running> = Vec::new();
    let mut records: HashMap<String, JobRecord> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut trace: Vec<TraceEvent> = Vec::new();
    let mut usage: HashMap<String, UserUsage> = HashMap::new();

    let mut seq: u64 = 0;
    let mut now: i64 = 0;
    let mut scheduler_ns: u128 = 0;
    let mut passes: u64 = 0;
    let mut frag_area: f64 = 0.0;
    let mut node_failed = false;

    // Start the clock at the first arrival rather than at zero, so an idle
    // lead-in does not deflate the utilisation figure.
    if let Some(first) = arriving.last() {
        now = first.arrival_s;
    }

    loop {
        // --- a node disappears, if this experiment says so.
        if let Some(failure) = &workload.fail_node {
            if !node_failed && now >= failure.at_s {
                node_failed = true;
                for node in nodes.iter_mut() {
                    if node.info.as_ref().map(|i| i.node_id.as_str()) == Some(&failure.node_id) {
                        node.healthy = false;
                    }
                }
                // Every rank of a job touching that node dies with it, which is
                // what the controller does: survivors would otherwise sit in a
                // collective holding GPUs.
                let doomed: Vec<String> = running
                    .iter()
                    .filter(|r| {
                        r.plan
                            .placements
                            .iter()
                            .any(|p| p.node_id == failure.node_id)
                    })
                    .map(|r| r.job.id.clone())
                    .collect();
                for id in &doomed {
                    if let Some(pos) = running.iter().position(|r| r.job.id == *id) {
                        let r = running.remove(pos);
                        clear_allocation(&mut nodes, &r.job.id);
                        if let Some(rec) = records.get_mut(id) {
                            rec.failed = true;
                            rec.end_s = Some(now);
                            rec.turnaround_s = now - rec.arrival_s;
                        }
                        trace.push(TraceEvent {
                            t: now,
                            event: "JOB_FAILED",
                            job: r.job.id.clone(),
                            user: r.job.user.clone(),
                            detail: format!("node {} lost", failure.node_id),
                        });
                    }
                }
                trace.push(TraceEvent {
                    t: now,
                    event: "NODE_LOST",
                    job: String::new(),
                    user: String::new(),
                    detail: failure.node_id.clone(),
                });
            }
        }

        // --- completions.
        let finished: Vec<usize> = running
            .iter()
            .enumerate()
            .filter(|(_, r)| r.end_s <= now)
            .map(|(i, _)| i)
            .collect();
        for i in finished.into_iter().rev() {
            let r = running.remove(i);
            clear_allocation(&mut nodes, &r.job.id);
            let charged = (r.end_s - r.start_s) as f64 * r.job.gpus() as f64;
            usage.entry(r.job.user.clone()).or_default().gpu_seconds += charged;
            if let Some(rec) = records.get_mut(&r.job.id) {
                rec.end_s = Some(r.end_s);
                rec.turnaround_s = r.end_s - rec.arrival_s;
            }
            trace.push(TraceEvent {
                t: r.end_s,
                event: "JOB_COMPLETED",
                job: r.job.id.clone(),
                user: r.job.user.clone(),
                detail: format!("{}s", r.end_s - r.start_s),
            });
        }

        // --- arrivals.
        while arriving.last().map(|j| j.arrival_s <= now).unwrap_or(false) {
            let job = arriving.pop().expect("checked");
            records.insert(
                job.id.clone(),
                JobRecord {
                    id: job.id.clone(),
                    user: job.user.clone(),
                    class: job.class.clone(),
                    priority: job.priority,
                    gpus: job.gpus(),
                    arrival_s: job.arrival_s,
                    start_s: None,
                    end_s: None,
                    duration_s: job.duration_s,
                    nominal_duration_s: job.duration_s,
                    wait_s: 0,
                    turnaround_s: 0,
                    overtaken_by: 0,
                    nodes_used: Vec::new(),
                    failed: false,
                },
            );
            order.push(job.id.clone());
            trace.push(TraceEvent {
                t: now,
                event: "JOB_QUEUED",
                job: job.id.clone(),
                user: job.user.clone(),
                detail: format!("{} GPU", job.gpus()),
            });
            waiting.push((seq, job));
            seq += 1;
        }

        // --- one scheduling pass.
        if !waiting.is_empty() {
            let started = schedule_pass(
                &mut waiting,
                &mut nodes,
                &network,
                queue_policy,
                placement,
                config,
                dispatch,
                &usage,
                now,
                &mut scheduler_ns,
            );
            passes += 1;
            for (job, plan) in started {
                // What the placement decision actually costs this job.
                let effective = execution.duration(
                    job.duration_s,
                    slowest_gpu_tflops(&nodes, &plan),
                    slowest_hop(&network, &plan, now),
                );
                let end_s = now + effective;
                // Count the overtakes before taking a mutable borrow: everything
                // that arrived later than this job and has already started.
                let arrival = records.get(&job.id).map(|r| r.arrival_s).unwrap_or(now);
                let overtaken = records_started_after(&records, arrival, &job.id);
                if let Some(rec) = records.get_mut(&job.id) {
                    rec.start_s = Some(now);
                    rec.duration_s = effective;
                    rec.wait_s = now - rec.arrival_s;
                    rec.nodes_used = plan.placements.iter().map(|p| p.node_id.clone()).collect();
                    rec.overtaken_by = overtaken;
                }
                trace.push(TraceEvent {
                    t: now,
                    event: "JOB_STARTED",
                    job: job.id.clone(),
                    user: job.user.clone(),
                    detail: plan
                        .placements
                        .iter()
                        .map(|p| format!("{}{:?}", p.node_id, p.gpu_indices))
                        .collect::<Vec<_>>()
                        .join(" "),
                });
                running.push(Running {
                    job,
                    plan,
                    start_s: now,
                    end_s,
                });
            }
        }

        // --- advance to the next thing that can happen.
        let next_arrival = arriving.last().map(|j| j.arrival_s);
        let next_completion = running.iter().map(|r| r.end_s).min();
        let next_failure = workload
            .fail_node
            .as_ref()
            .filter(|_| !node_failed)
            .map(|f| f.at_s);
        let next = [next_arrival, next_completion, next_failure]
            .into_iter()
            .flatten()
            .filter(|t| *t > now)
            .min();

        let Some(next) = next else {
            // Nothing left to happen. Anything still waiting never ran.
            break;
        };
        // Fragmentation is a time-weighted average, so it is accumulated over
        // the interval the cluster spent in this state.
        frag_area += stranded(&nodes, config.min_free_vram_b) as f64 * (next - now) as f64;
        now = next;
    }

    let span = span_of(&records).max(1);
    let mut jobs: Vec<JobRecord> = order.iter().filter_map(|id| records.remove(id)).collect();
    jobs.sort_by(|a, b| a.arrival_s.cmp(&b.arrival_s).then_with(|| a.id.cmp(&b.id)));

    SimOutcome {
        jobs,
        trace,
        scheduler_ns,
        scheduling_passes: passes,
        fragmentation: frag_area / span as f64,
        total_gpus: cluster.total_gpus(),
    }
}

fn span_of(records: &HashMap<String, JobRecord>) -> i64 {
    let first = records.values().map(|r| r.arrival_s).min().unwrap_or(0);
    let last = records
        .values()
        .filter_map(|r| r.end_s)
        .max()
        .unwrap_or(first);
    last - first
}

/// How many jobs that arrived after `arrival` had already started.
fn records_started_after(records: &HashMap<String, JobRecord>, arrival: i64, self_id: &str) -> u32 {
    records
        .values()
        .filter(|r| r.id != self_id)
        .filter(|r| r.arrival_s > arrival)
        .filter(|r| r.start_s.is_some())
        .count() as u32
}

/// Rank the waiting list and start whatever fits, returning what was started.
#[allow(clippy::too_many_arguments)]
fn schedule_pass(
    waiting: &mut Vec<(u64, SimJob)>,
    nodes: &mut [NodeState],
    network: &NetworkSnapshot,
    queue_policy: &dyn QueuePolicy,
    placement: &dyn PlacementPolicy,
    config: &SchedulerConfig,
    dispatch: Dispatch,
    usage: &HashMap<String, UserUsage>,
    now: i64,
    scheduler_ns: &mut u128,
) -> Vec<(SimJob, JobPlan)> {
    let snapshot = UsageSnapshot {
        per_user: usage.clone(),
    };
    let queued: Vec<QueuedJob> = waiting
        .iter()
        .map(|(seq, j)| QueuedJob {
            job_id: j.id.clone(),
            order_seq: *seq,
            submitted_unix_s: j.arrival_s,
            submitted_by: j.user.clone(),
            priority: j.priority,
            estimated_duration_s: j.estimated_duration_s,
            gpus: j.gpus(),
        })
        .collect();

    let started_at = std::time::Instant::now();
    let ranked = queue_policy.rank(&queued, &QueueContext::new(now, &snapshot));
    *scheduler_ns += started_at.elapsed().as_nanos();

    let mut started = Vec::new();
    for ranking in ranked {
        let Some(index) = waiting.iter().position(|(_, j)| j.id == ranking.job_id) else {
            continue;
        };
        let job = &waiting[index].1;
        let req = PlacementRequest {
            shape: Shape::Explicit {
                nodes: job.nodes.max(1),
                gpus_per_node: job.gpus_per_node.max(1),
            },
            node_filter: Vec::new(),
        };

        let placed = {
            let ctx = ferro_sched::SchedulingContext::new(now, nodes, config).with_network(network);
            let t = std::time::Instant::now();
            let outcome = placement.place(&req, &ctx);
            *scheduler_ns += t.elapsed().as_nanos();
            outcome
        };

        match placed {
            Ok(decision) => {
                let (_, job) = waiting.remove(index);
                set_allocation(nodes, &decision.plan, &job.id);
                started.push((job, decision.plan));
            }
            Err(_) => {
                if dispatch == Dispatch::Strict {
                    // Nothing may overtake the head of the queue.
                    break;
                }
            }
        }
    }
    started
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::{JobClass, UserSpec};
    use ferro_sched::placement::FirstFit;
    use ferro_sched::queue::{Fifo, Priority};
    use ferro_sched::PlacementWeights;

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            master_port: 29500,
            min_free_vram_b: 1 << 30,
            network_max_age_s: 0,
            placement_weights: PlacementWeights::default(),
        }
    }

    fn one_class(nodes: u32, gpus: u32, dur: u64) -> JobClass {
        JobClass {
            name: "c".into(),
            weight: 1.0,
            nodes,
            gpus_per_node: gpus,
            min_duration_s: dur,
            max_duration_s: dur,
        }
    }

    fn workload(jobs: usize, interarrival: f64, class: JobClass) -> WorkloadSpec {
        WorkloadSpec {
            name: "t".into(),
            seed: 7,
            jobs,
            mean_interarrival_s: interarrival,
            users: vec![UserSpec {
                name: "alice".into(),
                weight: 1.0,
                priority: 50,
            }],
            classes: vec![class],
            estimate_fraction: 1.0,
            estimate_error: 0.0,
            fail_node: None,
        }
    }

    #[test]
    fn every_job_runs_when_the_cluster_is_never_full() {
        let w = workload(20, 1000.0, one_class(1, 1, 10));
        let out = run(
            &w,
            &ClusterSpec::homogeneous(1, 4),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        assert_eq!(out.jobs.len(), 20);
        assert!(out.jobs.iter().all(|j| j.start_s.is_some()));
        assert!(out.jobs.iter().all(|j| j.end_s.is_some()));
        assert!(
            out.jobs.iter().all(|j| j.wait_s == 0),
            "an empty cluster should make nobody wait"
        );
    }

    #[test]
    fn a_gpu_is_never_in_two_jobs_at_once() {
        // The invariant the whole simulation rests on. Four GPUs, jobs
        // arriving far faster than they finish.
        let w = workload(60, 1.0, one_class(1, 1, 100));
        let out = run(
            &w,
            &ClusterSpec::homogeneous(1, 4),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        let mut events: Vec<(i64, i64)> = out
            .jobs
            .iter()
            .filter_map(|j| Some((j.start_s?, j.end_s?)))
            .collect();
        events.sort();
        // At no instant may more than four jobs overlap.
        for (start, _) in &events {
            let concurrent = events
                .iter()
                .filter(|(s, e)| s <= start && start < e)
                .count();
            assert!(concurrent <= 4, "{concurrent} jobs overlapped at t={start}");
        }
    }

    #[test]
    fn a_job_never_starts_before_it_arrives() {
        let w = workload(40, 5.0, one_class(1, 2, 60));
        let out = run(
            &w,
            &ClusterSpec::homogeneous(2, 2),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        for j in &out.jobs {
            if let Some(start) = j.start_s {
                assert!(start >= j.arrival_s, "{} started before arriving", j.id);
                assert_eq!(j.wait_s, start - j.arrival_s);
            }
        }
    }

    #[test]
    fn strict_dispatch_lets_nobody_overtake() {
        // The definition of the mode, asserted rather than assumed.
        let w = workload(40, 2.0, one_class(1, 1, 50));
        let out = run(
            &w,
            &ClusterSpec::homogeneous(1, 2),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Strict,
            &ExecutionModel::flat(),
        );
        assert!(
            out.jobs.iter().all(|j| j.overtaken_by == 0),
            "strict FIFO must not let a later arrival start first"
        );
    }

    #[test]
    fn opportunistic_dispatch_lets_small_jobs_past_a_big_one() {
        // FerroGrid's actual behaviour, and the reason reservation is on the
        // roadmap: a four-GPU job can be walked past by one-GPU jobs.
        let mut w = workload(40, 3.0, one_class(1, 1, 40));
        w.classes = vec![
            JobClass {
                name: "big".into(),
                weight: 1.0,
                nodes: 1,
                gpus_per_node: 4,
                min_duration_s: 200,
                max_duration_s: 200,
            },
            JobClass {
                name: "small".into(),
                weight: 6.0,
                nodes: 1,
                gpus_per_node: 1,
                min_duration_s: 40,
                max_duration_s: 40,
            },
        ];
        let out = run(
            &w,
            &ClusterSpec::homogeneous(1, 4),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        let passed_over: u32 = out.jobs.iter().map(|j| j.overtaken_by).sum();
        assert!(
            passed_over > 0,
            "opportunistic dispatch should let somebody overtake"
        );
    }

    #[test]
    fn the_same_inputs_give_the_same_run() {
        let w = workload(50, 4.0, one_class(1, 2, 70));
        let cluster = ClusterSpec::homogeneous(2, 2);
        let first = run(
            &w,
            &cluster,
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        for _ in 0..4 {
            let again = run(
                &w,
                &cluster,
                &Fifo,
                &FirstFit,
                &config(),
                Dispatch::Opportunistic,
                &ExecutionModel::flat(),
            );
            let a: Vec<_> = first
                .jobs
                .iter()
                .map(|j| (&j.id, j.start_s, j.end_s))
                .collect();
            let b: Vec<_> = again
                .jobs
                .iter()
                .map(|j| (&j.id, j.start_s, j.end_s))
                .collect();
            assert_eq!(a, b, "identical inputs must give an identical run");
        }
    }

    #[test]
    fn the_queue_policy_changes_who_goes_first() {
        // If swapping the policy changed nothing, the whole comparison would be
        // measuring one scheduler under several names.
        let mut w = workload(40, 2.0, one_class(1, 1, 60));
        w.users = vec![
            UserSpec {
                name: "quiet".into(),
                weight: 1.0,
                priority: 10,
            },
            UserSpec {
                name: "loud".into(),
                weight: 1.0,
                priority: 90,
            },
        ];
        let cluster = ClusterSpec::homogeneous(1, 2);
        let fifo = run(
            &w,
            &cluster,
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        let prio = run(
            &w,
            &cluster,
            &Priority,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );

        let wait = |o: &SimOutcome, user: &str| -> f64 {
            let xs: Vec<i64> = o
                .jobs
                .iter()
                .filter(|j| j.user == user)
                .map(|j| j.wait_s)
                .collect();
            xs.iter().sum::<i64>() as f64 / xs.len() as f64
        };
        assert!(
            wait(&prio, "loud") < wait(&fifo, "loud"),
            "priority should shorten the loud user's wait"
        );
    }

    #[test]
    fn a_node_failure_kills_the_jobs_on_it() {
        let mut w = workload(30, 3.0, one_class(1, 1, 120));
        w.fail_node = Some(crate::workload::NodeFailure {
            node_id: "gpu-a".into(),
            at_s: 40,
        });
        let out = run(
            &w,
            &ClusterSpec::homogeneous(2, 2),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        assert!(
            out.jobs.iter().any(|j| j.failed),
            "losing a node mid-run should fail whatever was on it"
        );
        assert!(
            out.trace.iter().any(|e| e.event == "NODE_LOST"),
            "the trace should record the failure"
        );
        // And nothing may be placed on it afterwards.
        for j in out.jobs.iter().filter(|j| j.start_s.unwrap_or(0) > 40) {
            assert!(
                !j.nodes_used.iter().any(|n| n == "gpu-a"),
                "{} landed on a dead node",
                j.id
            );
        }
    }

    #[test]
    fn the_trace_tells_a_coherent_story() {
        let w = workload(15, 5.0, one_class(1, 1, 30));
        let out = run(
            &w,
            &ClusterSpec::homogeneous(1, 2),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        for job in &out.jobs {
            let events: Vec<&str> = out
                .trace
                .iter()
                .filter(|e| e.job == job.id)
                .map(|e| e.event)
                .collect();
            assert_eq!(
                events,
                vec!["JOB_QUEUED", "JOB_STARTED", "JOB_COMPLETED"],
                "{} had {events:?}",
                job.id
            );
        }
    }

    #[test]
    fn scheduler_overhead_is_measured_not_assumed() {
        let w = workload(30, 3.0, one_class(1, 1, 50));
        let out = run(
            &w,
            &ClusterSpec::homogeneous(1, 2),
            &Fifo,
            &FirstFit,
            &config(),
            Dispatch::Opportunistic,
            &ExecutionModel::flat(),
        );
        assert!(out.scheduling_passes > 0);
        assert!(out.scheduler_ns > 0, "the policies must have been timed");
    }
}
