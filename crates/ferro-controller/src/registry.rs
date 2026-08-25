//! In-memory registry of nodes, jobs, and their logs.
//!
//! MVP scope: state lives in the controller process only. Agents re-register
//! automatically after a controller restart, so node state self-heals; job
//! history does not survive a restart, which is an accepted trade-off here.

use ferro_proto::{
    Gpu, GpuEntry, GpuOccupant, GpuProcess, JobPhase, JobPlan, JobStatus, JobSummary, LogLine,
    NodeInfo, NodeState, TrainingMetrics,
};
use std::collections::{HashMap, HashSet};
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
}

impl Job {
    /// A job is only as good as its worst rank: any failure fails the job, and
    /// it only succeeds once every rank has succeeded.
    pub fn phase(&self) -> JobPhase {
        let expected = self.plan.placements.len();
        if self.per_node.len() < expected {
            return JobPhase::Launching;
        }
        let phases: Vec<JobPhase> = self.per_node.values().map(|s| s.phase()).collect();
        if phases.iter().any(|p| *p == JobPhase::Failed) {
            JobPhase::Failed
        } else if phases.iter().any(|p| *p == JobPhase::Cancelled) {
            JobPhase::Cancelled
        } else if phases.iter().all(|p| *p == JobPhase::Succeeded) {
            JobPhase::Succeeded
        } else if phases.iter().any(|p| *p == JobPhase::Running) {
            JobPhase::Running
        } else {
            JobPhase::Pending
        }
    }

    pub fn to_summary(&self) -> JobSummary {
        let mut per_node: Vec<JobStatus> = self.per_node.values().cloned().collect();
        per_node.sort_by_key(|s| s.node_rank);
        let mut metrics = self.metrics.clone();
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
        }
    }

    pub fn record_util(&mut self, pct: f64) {
        self.util_sum += pct;
        self.util_n += 1;
    }
}

#[derive(Default)]
pub struct RegistryInner {
    /// GPU uuid -> measured TFLOP/s. Heartbeats overwrite the GPU list, so the
    /// scores live here and are re-applied on every update.
    pub bench: HashMap<String, (f64, i64)>,
    pub nodes: HashMap<String, Node>,
    pub jobs: HashMap<String, Job>,
    /// Submission order, so `ferro jobs` lists newest first.
    pub job_order: Vec<String>,
}

pub struct Registry {
    pub inner: Mutex<RegistryInner>,
    /// The scheduler's VRAM floor, kept here too so "free" means the same
    /// thing in `ferro nodes` as it does at placement time.
    pub min_free_vram_b: u64,
}

impl Registry {
    pub fn new(min_free_vram_b: u64) -> Self {
        Self { inner: Mutex::new(RegistryInner::default()), min_free_vram_b }
    }

    pub async fn upsert_node(&self, info: NodeInfo) {
        let mut g = self.inner.lock().await;
        let id = info.node_id.clone();
        g.nodes.insert(id, Node { info, last_seen: now_s() });
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
            let e = per_job.entry(gpu.allocated_job_id.clone()).or_insert((0.0, 0));
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
        let mut v: Vec<NodeState> =
            g.nodes.values().map(|n| n.to_state(self.min_free_vram_b)).collect();
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
            .filter(|j| j.timeout_s > 0 && !j.phase().is_terminal())
            .filter(|j| now - j.submitted > j.timeout_s as i64)
            .map(|j| {
                (
                    j.job_id.clone(),
                    j.plan.placements.iter().map(|p| p.address.clone()).collect(),
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
                job.nccl_errors
                    .push(format!("[rank{}/{}] {}", line.node_rank, line.node_id, line.line));
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
            started_unix_s: if status.started_unix_s > 0 { status.started_unix_s } else { started },
            ..status
        };
        job.per_node.insert(merged.node_id.clone(), merged);
    }

    /// Free the GPUs a finished job was holding. Agents do this too via their
    /// own allocation table, but the controller must not wait a heartbeat.
    pub async fn release_if_done(&self, job_id: &str) {
        let mut g = self.inner.lock().await;
        let done = g.jobs.get(job_id).map(|j| j.phase().is_terminal()).unwrap_or(false);
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

    /// Optimistically mark GPUs busy at submit time so two back-to-back
    /// submissions cannot both be handed the same devices.
    pub async fn reserve(&self, plan: &JobPlan, job_id: &str) {
        let mut g = self.inner.lock().await;
        for p in &plan.placements {
            if let Some(node) = g.nodes.get_mut(&p.node_id) {
                for gpu in node.info.gpus.iter_mut() {
                    if p.gpu_indices.contains(&gpu.index) {
                        gpu.allocated_job_id = job_id.to_string();
                    }
                }
            }
        }
    }
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
}

/// Foreign processes on one GPU, folded per user.
fn occupants_of(procs: &[GpuProcess], index: u32, live: &HashSet<String>) -> Vec<GpuOccupant> {
    let mut by_user: HashMap<&str, GpuOccupant> = HashMap::new();
    for p in procs.iter().filter(|p| p.gpu_index == index) {
        if live.contains(&p.job_id) {
            continue; // ours: the JOB column already names it
        }
        let e = by_user.entry(p.user.as_str()).or_insert_with(|| GpuOccupant {
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

    #[test]
    fn a_card_somebody_else_filled_is_not_free() {
        let node = Node {
            info: NodeInfo {
                gpus: vec![
                    gpu(0, 1 << 30, ""),        // free
                    gpu(1, 20 << 30, ""),       // someone else's work
                    gpu(2, 1 << 30, "jabc"),    // ours
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
            proc(0, "us", 8 << 30, "jabc", 0),   // ours: named by the JOB column
            proc(1, "carol", 3 << 30, "", 0),    // another card
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
}
