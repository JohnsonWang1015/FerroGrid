//! In-memory registry of nodes, jobs, and their logs.
//!
//! MVP scope: state lives in the controller process only. Agents re-register
//! automatically after a controller restart, so node state self-heals; job
//! history does not survive a restart, which is an accepted trade-off here.

use ferro_proto::{
    Gpu, GpuEntry, GpuOccupant, GpuProcess, JobPhase, JobPlan, JobStatus, JobSummary, LogLine,
    NodeInfo, NodeState, NodeVerdict, PlacementExplanation, QueueScore, ScoreComponent,
    SubmitJobRequest, TrainingMetrics,
};
use ferro_sched::{
    NetworkSnapshot, QueueContext, QueuePolicy, QueueRanking, QueuedJob, UsageSnapshot, UserUsage,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

/// A node is considered unhealthy after this many seconds without a heartbeat.
pub const HEARTBEAT_TIMEOUT_S: i64 = 15;
/// Ring-buffer size for per-job logs kept for `ferro logs`.
const LOG_BUFFER_LINES: usize = 20_000;
const MAX_NCCL_ERRORS: usize = 50;

pub fn now_s() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub struct Node {
    pub info: NodeInfo,
    pub last_seen: i64,
}

impl Node {
    pub fn healthy(&self) -> bool {
        now_s() - self.last_seen <= HEARTBEAT_TIMEOUT_S
    }

    /// Cards that could actually take a job: no FerroGrid allocation *and*
    /// enough VRAM left. Counting a card somebody else is already using as
    /// free is how "5/5 nodes ready, 8/8 GPUs free" ends in a job that will
    /// not place -- the scheduler applies the same VRAM floor.
    fn free_gpus(&self, min_free_vram_b: u64) -> u32 {
        self.info
            .gpus
            .iter()
            .filter(|g| g.allocated_job_id.is_empty())
            .filter(|g| g.memory_total_b.saturating_sub(g.memory_used_b) >= min_free_vram_b)
            .count() as u32
    }

    pub fn to_state(&self, min_free_vram_b: u64) -> NodeState {
        NodeState {
            info: Some(self.info.clone()),
            healthy: self.healthy(),
            last_seen_unix_s: self.last_seen,
            free_gpus: self.free_gpus(min_free_vram_b),
        }
    }
}

pub struct Job {
    pub job_id: String,
    pub name: String,
    pub submitted_by: String,
    /// Accounting bucket, orthogonal to the submitter.
    pub project: String,
    /// 0-100 as submitted, with the controller's default already substituted.
    pub priority: u32,
    /// What the submitter said this would take. `None` means they did not say,
    /// which is a different thing from saying zero.
    pub estimated_duration_s: Option<u32>,
    /// Wall-clock limit in seconds; 0 disables it.
    pub timeout_s: u32,
    pub plan: JobPlan,
    pub per_node: HashMap<String, JobStatus>,
    pub submitted: i64,
    pub logs: std::collections::VecDeque<LogLine>,
    pub nccl_errors: Vec<String>,
    pub metrics: TrainingMetrics,
    /// GPU utilisation samples accumulated from heartbeats, for the average.
    pub util_sum: f64,
    pub util_n: u64,
    pub tx: broadcast::Sender<LogLine>,
    /// Waiting for capacity: no plan yet, and the original request kept so the
    /// dispatcher can place it once the cluster frees up.
    pub queued: bool,
    pub queue_req: Option<SubmitJobRequest>,
    /// Unix seconds after which to give up queueing. 0 = wait indefinitely.
    pub queue_deadline: i64,
    /// Latest per-node explanation from the scheduler, refreshed while queued.
    pub node_verdicts: Vec<NodeVerdict>,
    pub warnings: Vec<String>,
    /// Latest aggregate scheduler message while this job is queued.
    pub queue_message: String,
    /// Why the scheduler chose these GPUs. Set when the job gets a plan, so a
    /// decision can still be explained long after the cluster has moved on.
    pub placement: Option<PlacementExplanation>,
}

impl Job {
    /// A job is only as good as its worst rank: any failure fails the job, and
    /// it only succeeds once every rank has succeeded.
    pub fn phase(&self) -> JobPhase {
        // A queued job has no placements at all, which the vote below would
        // read as "every rank succeeded".
        if self.queued {
            return JobPhase::Pending;
        }
        let expected = self.plan.placements.len();
        if self.per_node.len() < expected {
            return JobPhase::Launching;
        }
        let phases: Vec<JobPhase> = self.per_node.values().map(|s| s.phase()).collect();
        if phases.contains(&JobPhase::Failed) {
            JobPhase::Failed
        } else if phases.contains(&JobPhase::Cancelled) {
            JobPhase::Cancelled
        } else if phases.iter().all(|p| *p == JobPhase::Succeeded) {
            JobPhase::Succeeded
        } else if phases.contains(&JobPhase::Running) {
            JobPhase::Running
        } else {
            JobPhase::Pending
        }
    }

    pub fn to_summary(&self) -> JobSummary {
        let mut per_node: Vec<JobStatus> = self.per_node.values().cloned().collect();
        per_node.sort_by_key(|s| s.node_rank);
        let mut metrics = self.metrics;
        if self.util_n > 0 {
            metrics.avg_gpu_util_pct = self.util_sum / self.util_n as f64;
        }
        JobSummary {
            job_id: self.job_id.clone(),
            name: self.name.clone(),
            submitted_by: self.submitted_by.clone(),
            phase: self.phase() as i32,
            plan: Some(self.plan.clone()),
            per_node,
            submitted_unix_s: self.submitted,
            nccl_errors: self.nccl_errors.clone(),
            metrics: Some(metrics),
            queued: self.queued,
            // Filled in by the registry, which is the only place that knows
            // about the other jobs in line.
            queue_position: 0,
            node_verdicts: self.node_verdicts.clone(),
            warnings: self.warnings.clone(),
            queue_message: self.queue_message.clone(),
            priority: self.priority,
            project: self.project.clone(),
            estimated_duration_s: self.estimated_duration_s.unwrap_or(0),
            placement: self.placement.clone(),
            // Filled in by the registry, which is the only place that can run
            // the queue policy over the whole waiting list.
            queue_score: None,
        }
    }

    /// GPU-seconds this job has consumed as of `now`.
    ///
    /// Derived from the job records rather than kept in a running ledger, so
    /// there is one source of truth: a counter maintained alongside the jobs
    /// would be a second one, and the two would eventually disagree.
    ///
    /// A job that never started has consumed nothing, however long it sat in
    /// the queue -- waiting is not usage, and charging for it would penalise
    /// exactly the users fair share is meant to protect.
    pub fn gpu_seconds(&self, now: i64) -> f64 {
        let gpus: usize = self
            .plan
            .placements
            .iter()
            .map(|p| p.gpu_indices.len())
            .sum();
        if gpus == 0 {
            return 0.0;
        }
        let Some(started) = self
            .per_node
            .values()
            .map(|s| s.started_unix_s)
            .filter(|t| *t > 0)
            .min()
        else {
            return 0.0;
        };
        let ended = if self.phase().is_terminal() {
            self.per_node
                .values()
                .map(|s| s.ended_unix_s)
                .filter(|t| *t > 0)
                .max()
                .unwrap_or(now)
        } else {
            now
        };
        (ended - started).max(0) as f64 * gpus as f64
    }

    pub fn record_util(&mut self, pct: f64) {
        self.util_sum += pct;
        self.util_n += 1;
    }
}

pub struct RegistryInner {
    /// GPU uuid -> measured TFLOP/s. Heartbeats overwrite the GPU list, so the
    /// scores live here and are re-applied on every update.
    pub bench: HashMap<String, (f64, i64)>,
    pub nodes: HashMap<String, Node>,
    pub jobs: HashMap<String, Job>,
    /// Submission order, so `ferro jobs` lists newest first.
    pub job_order: Vec<String>,
    /// What `ferro net` measured between nodes, kept so the scheduler can see
    /// it. Heartbeats replace the GPU list wholesale, and these survive that
    /// the same way the benchmark scores do.
    pub net: NetworkSnapshot,
    /// Decides which queued job runs next. Lives here rather than beside the
    /// dispatcher so that the position a user is quoted and the order they are
    /// actually served in are computed by the same object -- they are the same
    /// promise, and a scheduler that breaks it is worse than one with no queue.
    pub queue_policy: Arc<dyn QueuePolicy>,
}

impl Default for RegistryInner {
    fn default() -> Self {
        Self {
            bench: HashMap::new(),
            nodes: HashMap::new(),
            jobs: HashMap::new(),
            job_order: Vec::new(),
            net: NetworkSnapshot::default(),
            queue_policy: Arc::new(ferro_sched::queue::Fifo),
        }
    }
}

pub struct Registry {
    pub inner: Mutex<RegistryInner>,
    /// The scheduler's VRAM floor, kept here too so "free" means the same
    /// thing in `ferro nodes` as it does at placement time.
    pub min_free_vram_b: u64,
}

impl Registry {
    /// A registry ordering its queue first come, first served.
    pub fn new(min_free_vram_b: u64) -> Self {
        Self {
            inner: Mutex::new(RegistryInner::default()),
            min_free_vram_b,
        }
    }

    pub fn with_queue_policy(min_free_vram_b: u64, queue_policy: Arc<dyn QueuePolicy>) -> Self {
        Self {
            inner: Mutex::new(RegistryInner {
                queue_policy,
                ..Default::default()
            }),
            min_free_vram_b,
        }
    }

    pub async fn upsert_node(&self, info: NodeInfo) {
        let mut g = self.inner.lock().await;
        let id = info.node_id.clone();
        g.nodes.insert(
            id,
            Node {
                info,
                last_seen: now_s(),
            },
        );
    }

    pub async fn heartbeat(
        &self,
        node_id: &str,
        gpus: Vec<Gpu>,
        processes: Vec<GpuProcess>,
    ) -> bool {
        let mut g = self.inner.lock().await;
        let bench = g.bench.clone();
        let Some(node) = g.nodes.get_mut(node_id) else {
            return false;
        };
        node.last_seen = now_s();
        node.info.gpus = gpus;
        node.info.processes = processes;
        for gpu in node.info.gpus.iter_mut() {
            if let Some((tflops, at)) = bench.get(&gpu.uuid) {
                gpu.bench_tflops = *tflops;
                gpu.bench_unix_s = *at;
            }
        }

        // Fold this node's GPU utilisation into the average of every job that
        // currently holds one of its devices.
        let mut per_job: HashMap<String, (f64, u32)> = HashMap::new();
        for gpu in &node.info.gpus {
            if gpu.allocated_job_id.is_empty() {
                continue;
            }
            let e = per_job
                .entry(gpu.allocated_job_id.clone())
                .or_insert((0.0, 0));
            e.0 += gpu.utilization_pct as f64;
            e.1 += 1;
        }
        for (job_id, (sum, n)) in per_job {
            if let Some(job) = g.jobs.get_mut(&job_id) {
                job.record_util(sum / n as f64);
            }
        }
        true
    }

    /// Remember what `ferro net` measured, so a placement decision can prefer
    /// a path somebody has actually tested over one that merely negotiated a
    /// fast link speed.
    pub async fn record_network(&self, pairs: &[ferro_proto::NetPair]) {
        let mut g = self.inner.lock().await;
        let now = now_s();
        for pair in pairs {
            // A failed probe reports 0 Mb/s. Recording that would make an
            // unreachable pair look like the slowest link rather than an
            // untested one, and the scheduler would then rank it.
            if pair.error.is_empty() {
                g.net.record(&pair.from_node, &pair.to_node, pair.mbps, now);
            }
        }
    }

    /// The measurements as they stand, for handing to a placement policy.
    pub async fn network_snapshot(&self) -> NetworkSnapshot {
        self.inner.lock().await.net.clone()
    }

    /// Remember measured throughput so the scheduler keeps seeing it after the
    /// next heartbeat replaces the GPU list.
    pub async fn record_benchmarks(&self, results: &[ferro_proto::GpuBenchmark]) {
        let mut g = self.inner.lock().await;
        let now = now_s();
        for r in results {
            if r.tflops > 0.0 && !r.uuid.is_empty() {
                g.bench.insert(r.uuid.clone(), (r.tflops, now));
            }
        }
        let bench = g.bench.clone();
        for node in g.nodes.values_mut() {
            for gpu in node.info.gpus.iter_mut() {
                if let Some((t, at)) = bench.get(&gpu.uuid) {
                    gpu.bench_tflops = *t;
                    gpu.bench_unix_s = *at;
                }
            }
        }
    }

    pub async fn node_states(&self) -> Vec<NodeState> {
        let g = self.inner.lock().await;
        let mut v: Vec<NodeState> = g
            .nodes
            .values()
            .map(|n| n.to_state(self.min_free_vram_b))
            .collect();
        v.sort_by(|a, b| node_id_of(a).cmp(node_id_of(b)));
        v
    }

    pub async fn gpu_entries(&self) -> Vec<GpuEntry> {
        let g = self.inner.lock().await;
        let live = g.live_job_ids();
        let mut nodes: Vec<&Node> = g.nodes.values().collect();
        nodes.sort_by(|a, b| a.info.node_id.cmp(&b.info.node_id));
        nodes
            .iter()
            .flat_map(|n| {
                let healthy = n.healthy();
                let live = &live;
                let floor = self.min_free_vram_b;
                n.info.gpus.iter().map(move |gpu| GpuEntry {
                    node_id: n.info.node_id.clone(),
                    gpu: Some(gpu.clone()),
                    healthy,
                    occupants: occupants_of(&n.info.processes, gpu.index, live),
                    schedulable: gpu.allocated_job_id.is_empty()
                        && gpu.memory_total_b.saturating_sub(gpu.memory_used_b) >= floor,
                })
            })
            .collect()
    }

    /// Jobs still waiting for capacity, oldest first, with the request the
    /// dispatcher needs to place them.
    pub async fn queued_jobs(&self) -> Vec<(String, SubmitJobRequest, i64)> {
        let g = self.inner.lock().await;
        let mut out: Vec<(String, SubmitJobRequest, i64)> = g
            .jobs
            .values()
            .filter(|j| j.queued)
            .filter_map(|j| {
                j.queue_req
                    .clone()
                    .map(|r| (j.job_id.clone(), r, j.queue_deadline))
            })
            .collect();
        // Order is the queue policy's decision, not this function's. The same
        // ranking backs `queue_position`, so what a user is told and what the
        // dispatcher does cannot drift apart.
        let ranked = g.queue_order(now_s());
        out.sort_by_key(|(id, _, _)| ranked.iter().position(|j| j == id).unwrap_or(usize::MAX));
        out
    }

    /// A queued job has been placed: it now has a plan and stops being queued.
    pub async fn promote(
        &self,
        job_id: &str,
        plan: JobPlan,
        placement: Option<PlacementExplanation>,
    ) {
        let mut g = self.inner.lock().await;
        if let Some(job) = g.jobs.get_mut(job_id) {
            job.plan = plan;
            job.placement = placement;
            job.queued = false;
            job.queue_req = None;
            job.queue_message.clear();
            // The wall clock starts when the job starts, not when it queued.
            job.submitted = now_s();
        }
    }

    /// Atomically refresh everything a queued job exposes about its latest
    /// placement attempt.
    pub async fn update_queue_assessment(
        &self,
        job_id: &str,
        verdicts: Vec<NodeVerdict>,
        message: String,
        warnings: Vec<String>,
    ) {
        let mut g = self.inner.lock().await;
        if let Some(job) = g.jobs.get_mut(job_id).filter(|j| j.queued) {
            job.node_verdicts = verdicts;
            job.queue_message = message;
            job.warnings = warnings;
        }
    }

    /// Take a job out of the queue without ever launching it. The status is
    /// what makes it terminal -- a queued job has no ranks to report one.
    pub async fn dequeue(&self, job_id: &str, phase: JobPhase, message: &str) -> bool {
        let mut g = self.inner.lock().await;
        let Some(job) = g.jobs.get_mut(job_id) else {
            return false;
        };
        if !job.queued {
            return false;
        }
        job.queued = false;
        job.queue_req = None;
        job.queue_message = message.to_string();
        job.per_node.insert(
            String::new(),
            JobStatus {
                job_id: job_id.to_string(),
                phase: phase as i32,
                message: message.to_string(),
                ended_unix_s: now_s(),
                ..Default::default()
            },
        );
        true
    }

    /// Where a job sits in the queue, 1-based. 0 when it is not queued.
    pub async fn queue_position(&self, job_id: &str) -> u32 {
        let g = self.inner.lock().await;
        g.queue_position(job_id)
    }

    /// Jobs past their wall-clock limit, as (job_id, agent addresses).
    ///
    /// A hung distributed job never reports failure -- every rank sits in a
    /// collective waiting for a peer -- so nothing else reclaims its GPUs.
    /// On a shared cluster that is the difference between a wasted afternoon
    /// and a wasted week.
    pub async fn expired_jobs(&self) -> Vec<(String, Vec<String>)> {
        let g = self.inner.lock().await;
        let now = now_s();
        g.jobs
            .values()
            // A queued job is not burning anything; its own deadline applies.
            .filter(|j| j.timeout_s > 0 && !j.queued && !j.phase().is_terminal())
            .filter(|j| now - j.submitted > j.timeout_s as i64)
            .map(|j| {
                (
                    j.job_id.clone(),
                    j.plan
                        .placements
                        .iter()
                        .map(|p| p.address.clone())
                        .collect(),
                )
            })
            .collect()
    }

    pub async fn insert_job(&self, job: Job) {
        let mut g = self.inner.lock().await;
        g.job_order.push(job.job_id.clone());
        g.jobs.insert(job.job_id.clone(), job);
    }

    pub async fn append_logs(&self, lines: Vec<LogLine>) {
        let mut g = self.inner.lock().await;
        for line in lines {
            let Some(job) = g.jobs.get_mut(&line.job_id) else {
                continue;
            };
            if crate::metrics::is_nccl_error(&line.line) && job.nccl_errors.len() < MAX_NCCL_ERRORS
            {
                job.nccl_errors.push(format!(
                    "[rank{}/{}] {}",
                    line.node_rank, line.node_id, line.line
                ));
            }
            if let Some(m) = crate::metrics::parse_metric_line(&line.line) {
                crate::metrics::merge(&mut job.metrics, m);
            }
            // Subscribers may be absent; a send error just means nobody is following.
            let _ = job.tx.send(line.clone());
            job.logs.push_back(line);
            if job.logs.len() > LOG_BUFFER_LINES {
                job.logs.pop_front();
            }
        }
    }

    pub async fn update_job_status(&self, status: JobStatus) {
        let mut g = self.inner.lock().await;
        let Some(job) = g.jobs.get_mut(&status.job_id) else {
            return;
        };
        // Agents send a start event with started_unix_s and a finish event
        // without it; keep the earlier start time.
        let started = job
            .per_node
            .get(&status.node_id)
            .map(|s| s.started_unix_s)
            .unwrap_or(0);
        let merged = JobStatus {
            started_unix_s: if status.started_unix_s > 0 {
                status.started_unix_s
            } else {
                started
            },
            ..status
        };
        job.per_node.insert(merged.node_id.clone(), merged);
    }

    /// Free the GPUs a finished job was holding. Agents do this too via their
    /// own allocation table, but the controller must not wait a heartbeat.
    pub async fn release_if_done(&self, job_id: &str) {
        let mut g = self.inner.lock().await;
        let done = g
            .jobs
            .get(job_id)
            .map(|j| j.phase().is_terminal())
            .unwrap_or(false);
        if !done {
            return;
        }
        for node in g.nodes.values_mut() {
            for gpu in node.info.gpus.iter_mut() {
                if gpu.allocated_job_id == job_id {
                    gpu.allocated_job_id.clear();
                }
            }
        }
    }

    /// Take exactly the GPUs in `plan`, or take none of them.
    ///
    /// Placement runs outside this lock -- it is pure CPU work over a snapshot,
    /// and holding the lock across it would serialise every queued job's
    /// planning for no benefit. That leaves a window in which another
    /// submission can claim a card this plan wanted, so the reservation
    /// verifies its own precondition here rather than assuming it: a
    /// compare-and-swap against the real ownership, not against a version
    /// counter that every heartbeat would bump.
    ///
    /// All-or-nothing matters as much as the check. A partial reservation would
    /// leave this job owning cards it will never launch on, and the release
    /// that follows a failure clears by job id -- which is how an unconditional
    /// overwrite used to end with one job releasing another job's GPUs.
    pub async fn reserve_exact(&self, plan: &JobPlan, job_id: &str) -> Result<(), ReserveConflict> {
        let mut g = self.inner.lock().await;

        // Check everything before touching anything.
        for p in &plan.placements {
            let node = g
                .nodes
                .get(&p.node_id)
                .ok_or_else(|| ReserveConflict::NodeGone {
                    node_id: p.node_id.clone(),
                })?;
            for index in &p.gpu_indices {
                let gpu = node
                    .info
                    .gpus
                    .iter()
                    .find(|g| g.index == *index)
                    .ok_or_else(|| ReserveConflict::GpuGone {
                        node_id: p.node_id.clone(),
                        gpu_index: *index,
                    })?;
                if !gpu.allocated_job_id.is_empty() && gpu.allocated_job_id != job_id {
                    return Err(ReserveConflict::AlreadyHeld {
                        node_id: p.node_id.clone(),
                        gpu_index: *index,
                        holder: gpu.allocated_job_id.clone(),
                    });
                }
            }
        }

        for p in &plan.placements {
            if let Some(node) = g.nodes.get_mut(&p.node_id) {
                for gpu in node.info.gpus.iter_mut() {
                    if p.gpu_indices.contains(&gpu.index) {
                        gpu.allocated_job_id = job_id.to_string();
                    }
                }
            }
        }
        Ok(())
    }
}

/// Why a reservation was refused. Every variant means the same thing to a
/// caller -- the snapshot this plan was built from is stale -- but they are
/// worth telling apart in a log when a cluster is misbehaving.
#[derive(Debug, thiserror::Error)]
pub enum ReserveConflict {
    #[error("{node_id} GPU {gpu_index} was taken by {holder} while this job was being planned")]
    AlreadyHeld {
        node_id: String,
        gpu_index: u32,
        holder: String,
    },
    #[error("{node_id} is no longer registered")]
    NodeGone { node_id: String },
    #[error("{node_id} no longer reports GPU {gpu_index}")]
    GpuGone { node_id: String, gpu_index: u32 },
}

impl RegistryInner {
    /// Ids of jobs this controller still considers alive. A process tagged
    /// with anything else -- another controller's job, or one this one has
    /// forgotten across a restart -- is somebody else's as far as the GPU
    /// tables are concerned.
    fn live_job_ids(&self) -> HashSet<String> {
        self.jobs
            .values()
            .filter(|j| !j.phase().is_terminal())
            .map(|j| j.job_id.clone())
            .collect()
    }

    /// What each user has consumed, as of `now`.
    ///
    /// Recomputed from the job records on demand. That is O(jobs) per call and
    /// entirely affordable at this scale, and it buys the property that
    /// matters: the numbers a scheduling decision used and the numbers
    /// `ferro usage` reports cannot drift apart, because they are the same
    /// numbers.
    pub fn usage_snapshot(&self, now: i64) -> UsageSnapshot {
        let mut per_user: HashMap<String, UserUsage> = HashMap::new();
        for job in self.jobs.values() {
            let entry = per_user.entry(job.submitted_by.clone()).or_default();
            entry.gpu_seconds += job.gpu_seconds(now);
            if !job.phase().is_terminal() && !job.queued {
                entry.running_jobs += 1;
                entry.gpus_held += job
                    .plan
                    .placements
                    .iter()
                    .map(|p| p.gpu_indices.len() as u32)
                    .sum::<u32>();
            }
        }
        UsageSnapshot { per_user }
    }

    /// Every waiting job, scored and ordered by the queue policy.
    pub fn ranked_queue(&self, now: i64) -> Vec<QueueRanking> {
        let waiting: Vec<QueuedJob> = self
            .job_order
            .iter()
            .enumerate()
            .filter_map(|(seq, id)| {
                let job = self.jobs.get(id)?;
                job.queued.then(|| QueuedJob {
                    job_id: job.job_id.clone(),
                    // Position in `job_order` is the submission sequence: two
                    // jobs submitted in the same second still have an order.
                    order_seq: seq as u64,
                    submitted_unix_s: job.submitted,
                    submitted_by: job.submitted_by.clone(),
                    priority: job.priority,
                    estimated_duration_s: job.estimated_duration_s,
                    gpus: job.queue_req.as_ref().map(requested_gpus).unwrap_or(0),
                })
            })
            .collect();
        let usage = self.usage_snapshot(now);
        self.queue_policy
            .rank(&waiting, &QueueContext::new(now, &usage))
    }

    /// Just the ids, in served order.
    pub fn queue_order(&self, now: i64) -> Vec<String> {
        self.ranked_queue(now)
            .into_iter()
            .map(|r| r.job_id)
            .collect()
    }

    /// Fill in the queue view of these summaries from a single ranking pass.
    ///
    /// Position and score come from the same evaluation on purpose. Computing
    /// them separately would let a heartbeat land in between and produce a
    /// position that the score next to it does not justify -- which is exactly
    /// the sort of inconsistency that makes users stop believing the queue.
    pub fn annotate_queue(&self, summaries: &mut [JobSummary], now: i64) {
        let policy = self.queue_policy.name();
        for (index, ranking) in self.ranked_queue(now).into_iter().enumerate() {
            let Some(summary) = summaries.iter_mut().find(|s| s.job_id == ranking.job_id) else {
                continue;
            };
            summary.queue_position = index as u32 + 1;
            summary.queue_score = Some(QueueScore {
                policy: policy.to_string(),
                total: ranking.score,
                components: ranking
                    .components
                    .iter()
                    .map(|(name, value)| ScoreComponent {
                        name: (*name).to_string(),
                        value: *value,
                    })
                    .collect(),
            });
        }
    }

    /// Where a job sits in line, 1-based. 0 when it is not queued.
    pub fn queue_position(&self, job_id: &str) -> u32 {
        if !self.jobs.get(job_id).map(|j| j.queued).unwrap_or(false) {
            return 0;
        }
        self.queue_order(now_s())
            .iter()
            .position(|id| id == job_id)
            .map(|i| i as u32 + 1)
            .unwrap_or(0)
    }
}

/// How many GPUs a request is asking for, for the fair-share weighting.
/// Auto mode has not chosen a shape yet, so it counts as the one GPU it is
/// guaranteed to take.
fn requested_gpus(req: &SubmitJobRequest) -> u32 {
    if req.auto_place {
        req.gpus_per_node.max(1)
    } else {
        req.nodes.max(1) * req.gpus_per_node.max(1)
    }
}

/// Foreign processes on one GPU, folded per user.
fn occupants_of(procs: &[GpuProcess], index: u32, live: &HashSet<String>) -> Vec<GpuOccupant> {
    let mut by_user: HashMap<&str, GpuOccupant> = HashMap::new();
    for p in procs.iter().filter(|p| p.gpu_index == index) {
        if live.contains(&p.job_id) {
            continue; // ours: the JOB column already names it
        }
        let e = by_user
            .entry(p.user.as_str())
            .or_insert_with(|| GpuOccupant {
                user: p.user.clone(),
                ..Default::default()
            });
        e.processes += 1;
        e.memory_used_b += p.memory_used_b;
        // The user is as busy as their busiest process here.
        e.busy_unix_s = e.busy_unix_s.max(p.busy_unix_s);
    }
    let mut out: Vec<GpuOccupant> = by_user.into_values().collect();
    out.sort_by_key(|o| std::cmp::Reverse(o.memory_used_b));
    out
}

fn node_id_of(n: &NodeState) -> &str {
    n.info.as_ref().map(|i| i.node_id.as_str()).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(index: u32, used_b: u64, job: &str) -> Gpu {
        Gpu {
            index,
            uuid: format!("uuid-{index}"),
            memory_total_b: 24 << 30,
            memory_used_b: used_b,
            allocated_job_id: job.into(),
            ..Default::default()
        }
    }

    fn proc(index: u32, user: &str, mem_b: u64, job: &str, busy: i64) -> GpuProcess {
        GpuProcess {
            gpu_index: index,
            pid: 100 + index,
            memory_used_b: mem_b,
            user: user.into(),
            job_id: job.into(),
            busy_unix_s: busy,
            utilization_known: true,
            ..Default::default()
        }
    }

    fn job(id: &str, queued: bool, submitted: i64) -> Job {
        let (tx, _) = broadcast::channel(4);
        Job {
            job_id: id.into(),
            name: id.into(),
            submitted_by: "tester".into(),
            project: String::new(),
            priority: ferro_sched::DEFAULT_PRIORITY,
            estimated_duration_s: None,
            timeout_s: 0,
            plan: JobPlan::default(),
            per_node: Default::default(),
            submitted,
            logs: Default::default(),
            nccl_errors: Vec::new(),
            metrics: Default::default(),
            util_sum: 0.0,
            util_n: 0,
            tx,
            queued,
            queue_req: queued.then(SubmitJobRequest::default),
            queue_deadline: 0,
            node_verdicts: Vec::new(),
            warnings: Vec::new(),
            queue_message: String::new(),
            placement: None,
        }
    }

    #[test]
    fn a_card_somebody_else_filled_is_not_free() {
        let node = Node {
            info: NodeInfo {
                gpus: vec![
                    gpu(0, 1 << 30, ""),     // free
                    gpu(1, 20 << 30, ""),    // someone else's work
                    gpu(2, 1 << 30, "jabc"), // ours
                ],
                ..Default::default()
            },
            last_seen: now_s(),
        };
        assert_eq!(node.free_gpus(8 << 30), 1);
        // Without a VRAM floor, only our own allocation counts.
        assert_eq!(node.free_gpus(0), 2);
    }

    #[test]
    fn occupants_fold_per_user_and_skip_our_own_jobs() {
        let live: HashSet<String> = ["jabc".to_string()].into_iter().collect();
        let procs = vec![
            proc(0, "alice", 4 << 30, "", 100),
            proc(0, "alice", 2 << 30, "", 500),
            proc(0, "bob", 1 << 30, "", 0),
            proc(0, "us", 8 << 30, "jabc", 0), // ours: named by the JOB column
            proc(1, "carol", 3 << 30, "", 0),  // another card
        ];
        let out = occupants_of(&procs, 0, &live);
        assert_eq!(out.len(), 2);
        // Biggest holder first.
        assert_eq!(out[0].user, "alice");
        assert_eq!(out[0].processes, 2);
        assert_eq!(out[0].memory_used_b, 6 << 30);
        // A user is as busy as their busiest process on the card.
        assert_eq!(out[0].busy_unix_s, 500);
        assert_eq!(out[1].user, "bob");
    }

    #[test]
    fn a_queued_job_is_pending_not_succeeded() {
        // Its plan is empty, which the per-rank vote would otherwise read as
        // "every rank succeeded".
        let j = job("j1", true, 10);
        assert_eq!(j.phase(), JobPhase::Pending);
        assert!(!j.phase().is_terminal());
        assert!(j.to_summary().queued);
    }

    #[tokio::test]
    async fn the_queue_is_fifo_and_cancellable() {
        let r = Registry::new(8 << 30);
        // Same submission second on purpose: the order still has to hold.
        r.insert_job(job("first", true, 10)).await;
        r.insert_job(job("second", true, 10)).await;
        r.insert_job(job("running", false, 10)).await;

        let queued: Vec<String> = r
            .queued_jobs()
            .await
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();
        assert_eq!(queued, vec!["first", "second"]);
        assert_eq!(r.queue_position("first").await, 1);
        assert_eq!(r.queue_position("second").await, 2);
        assert_eq!(r.queue_position("running").await, 0);

        assert!(
            r.dequeue("second", JobPhase::Cancelled, "cancelled while queued")
                .await
        );
        // Cancelling a queued job must actually take it out of the line, or
        // the dispatcher launches something the user has already given up on.
        assert!(!r.dequeue("second", JobPhase::Cancelled, "again").await);
        let g = r.inner.lock().await;
        assert_eq!(g.jobs["second"].phase(), JobPhase::Cancelled);
        assert_eq!(g.queue_position("first"), 1);
    }

    #[tokio::test]
    async fn queued_verdicts_are_replaced_by_the_latest_attempt() {
        let r = Registry::new(8 << 30);
        r.insert_job(job("queued", true, 10)).await;
        r.update_queue_assessment(
            "queued",
            vec![NodeVerdict {
                node_id: "gpu-a".into(),
                eligible: false,
                reasons: vec!["GPU(s) allocated by FerroGrid".into()],
                free_gpus: 0,
                free_vram_b: 0,
            }],
            "only 0 free GPU(s), need 1".into(),
            vec!["node is busy".into()],
        )
        .await;
        let first = r.inner.lock().await.jobs["queued"].to_summary();
        assert_eq!(first.node_verdicts[0].node_id, "gpu-a");
        assert_eq!(first.queue_message, "only 0 free GPU(s), need 1");
        assert_eq!(first.warnings, vec!["node is busy"]);

        r.update_queue_assessment(
            "queued",
            vec![NodeVerdict {
                node_id: "gpu-b".into(),
                eligible: false,
                reasons: vec!["node is unhealthy or heartbeat is stale".into()],
                free_gpus: 2,
                free_vram_b: 40 << 30,
            }],
            "only 2 free GPU(s), need 3".into(),
            Vec::new(),
        )
        .await;
        let g = r.inner.lock().await;
        let latest = g.jobs["queued"].to_summary();
        assert_eq!(latest.node_verdicts[0].node_id, "gpu-b");
        assert_eq!(latest.queue_message, "only 2 free GPU(s), need 3");
        assert!(latest.warnings.is_empty());
    }
}
