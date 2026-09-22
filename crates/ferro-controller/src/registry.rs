//! In-memory registry of nodes, jobs, and their logs.
//!
//! Reads and writes all go through here; the durable copy is write-behind.
//! Agents re-register automatically after a controller restart, so node state
//! self-heals and is never written down. Jobs, their submission order, GPU
//! benchmarks and `ferro net` measurements exist nowhere else, so they are --
//! see `store` for which fields and why.

use ferro_proto::{
    Gpu, GpuEntry, GpuOccupant, GpuProcess, JobPhase, JobPlacement, JobPlan, JobStatus, JobSummary,
    LogLine, NodeInfo, NodeState, NodeVerdict, PlacementExplanation, QueueScore, ScoreComponent,
    SubmitJobRequest, TrainingMetrics,
};
use ferro_sched::{
    NetworkSnapshot, QueueContext, QueuePolicy, QueueRanking, QueuedJob, UsageSnapshot, UserUsage,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

use crate::store::{
    Change, Event, EventKind, JobRecord, LoadedJob, LoadedState, Store, StoreError, EVENT_RING,
};

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
            // Likewise: whether a job is still being reconciled is recovery
            // bookkeeping the registry holds, not something the job knows.
            reconciling: false,
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

    /// The durable half of this job. `order_seq` is its index in `job_order`,
    /// which the caller holds the lock for and this job does not know.
    pub fn to_record(&self, order_seq: i64) -> JobRecord {
        JobRecord {
            job_id: self.job_id.clone(),
            order_seq,
            name: self.name.clone(),
            submitted_by: self.submitted_by.clone(),
            project: self.project.clone(),
            priority: self.priority,
            estimated_duration_s: self.estimated_duration_s,
            timeout_s: self.timeout_s,
            submitted: self.submitted,
            queued: self.queued,
            queue_deadline: self.queue_deadline,
            plan: self.plan.clone(),
            queue_req: self.queue_req.clone(),
            placement: self.placement.clone(),
        }
    }

    /// Rebuild a job from disk, exactly as it was written.
    ///
    /// Nothing here is reconciled against what the cluster currently reports: a
    /// job that was Running when the controller died comes back Running, even
    /// if its ranks are long gone. Deciding what is still true is a separate
    /// job from remembering what was, and mixing the two would mean a restart
    /// silently rewriting history it has no evidence about.
    ///
    /// The broadcast channel is new because the old subscribers died with the
    /// process, and the log ring starts empty because it was never stored.
    pub fn from_record(record: JobRecord, per_node: HashMap<String, JobStatus>) -> Self {
        let (tx, _) = broadcast::channel(4096);
        Job {
            job_id: record.job_id,
            name: record.name,
            submitted_by: record.submitted_by,
            project: record.project,
            priority: record.priority,
            estimated_duration_s: record.estimated_duration_s,
            timeout_s: record.timeout_s,
            plan: record.plan,
            per_node,
            submitted: record.submitted,
            logs: std::collections::VecDeque::new(),
            nccl_errors: Vec::new(),
            metrics: Default::default(),
            util_sum: 0.0,
            util_n: 0,
            tx,
            queued: record.queued,
            queue_req: record.queue_req,
            queue_deadline: record.queue_deadline,
            // A live assessment of a cluster that has moved on. The next queue
            // tick writes the current one.
            node_verdicts: Vec::new(),
            warnings: Vec::new(),
            queue_message: String::new(),
            placement: record.placement,
        }
    }
}

/// What became of the jobs that came back off disk.
///
/// Counted rather than inferred afterwards, because nothing reconstructs it:
/// once a lost job has been failed it is indistinguishable from a job that
/// failed on its own, and once a claimed one is running it looks like any
/// other running job.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Recovery {
    /// Every job read off disk, terminal ones included -- the same number
    /// `CONTROLLER_RECOVERED` reports, so the two lines can be read together.
    pub restored: usize,
    /// Non-terminal restored jobs an agent accounted for.
    pub claimed: usize,
    /// Restored as queued, found already running, and given back the plan the
    /// crash lost.
    pub adopted: usize,
    /// Nobody accounted for these before the window closed.
    pub failed: usize,
    /// The window has closed and the counts above are final.
    pub settled: bool,
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
    /// The timeline, oldest first. This is the read path: `ferro events`
    /// answers from here, so it works under `--no-state` exactly as it does
    /// with a database behind it, and a read never touches a disk.
    pub events: std::collections::VecDeque<Event>,
    /// Number for the next event. Carries on across restarts, from the last id
    /// read back off disk.
    next_event_id: u64,
    /// Whether each node was healthy the last time anybody looked. Health is a
    /// duration without a heartbeat rather than anything a node sends, so the
    /// only way to notice it changing is to remember the previous answer.
    pub node_health: HashMap<String, bool>,
    /// Decides which queued job runs next. Lives here rather than beside the
    /// dispatcher so that the position a user is quoted and the order they are
    /// actually served in are computed by the same object -- they are the same
    /// promise, and a scheduler that breaks it is worse than one with no queue.
    pub queue_policy: Arc<dyn QueuePolicy>,
    /// Restored jobs no agent has accounted for yet.
    ///
    /// A set beside the jobs rather than a field on one: this is a claim about
    /// how much the *controller* currently knows, not about the job, and it is
    /// empty for the entire life of a controller that never restarted.
    pub reconciling: HashSet<String>,
    /// Jobs adopted during this recovery, remembered only until the window
    /// closes.
    ///
    /// A job is adopted from the cards the agents have reported *so far*, and
    /// a peer that has not reconnected yet is a placement missing from the
    /// rebuilt plan -- a rank nobody could afterwards cancel or time out. So a
    /// later heartbeat naming the same job may still complete it, and only
    /// while the window that owns this question is open.
    pub adopted: HashSet<String>,
    pub recovery: Recovery,
}

impl Default for RegistryInner {
    fn default() -> Self {
        Self {
            bench: HashMap::new(),
            nodes: HashMap::new(),
            jobs: HashMap::new(),
            job_order: Vec::new(),
            net: NetworkSnapshot::default(),
            events: std::collections::VecDeque::new(),
            next_event_id: 1,
            node_health: HashMap::new(),
            queue_policy: Arc::new(ferro_sched::queue::Fifo),
            reconciling: HashSet::new(),
            adopted: HashSet::new(),
            recovery: Recovery::default(),
        }
    }
}

pub struct Registry {
    pub inner: Mutex<RegistryInner>,
    /// The scheduler's VRAM floor, kept here too so "free" means the same
    /// thing in `ferro nodes` as it does at placement time.
    pub min_free_vram_b: u64,
    /// Where durable state goes. `None` is a controller running entirely in
    /// memory, which is what `--no-state` and every test that does not care
    /// about restarts get.
    store: Option<Store>,
}

impl Registry {
    /// A registry ordering its queue first come, first served.
    pub fn new(min_free_vram_b: u64) -> Self {
        Self {
            inner: Mutex::new(RegistryInner::default()),
            min_free_vram_b,
            store: None,
        }
    }

    pub fn with_queue_policy(min_free_vram_b: u64, queue_policy: Arc<dyn QueuePolicy>) -> Self {
        Self {
            inner: Mutex::new(RegistryInner {
                queue_policy,
                ..Default::default()
            }),
            min_free_vram_b,
            store: None,
        }
    }

    /// A registry that picks up where the last one left off, and keeps writing.
    ///
    /// Nodes are deliberately absent: they are back within a heartbeat, and
    /// restoring them would mean claiming a node is present on the strength of
    /// a file rather than a heartbeat.
    pub fn restore(
        min_free_vram_b: u64,
        queue_policy: Arc<dyn QueuePolicy>,
        store: Store,
        state: LoadedState,
    ) -> Self {
        // The ring comes back with the tail of the log in it, so `ferro events`
        // after a restart still answers "what happened last night" rather than
        // starting the history at the restart. The sequence continues from the
        // last id: two events with the same number would be two events nobody
        // can put in order.
        let next_event_id = state.events.last().map(|e| e.id + 1).unwrap_or(1);
        let mut inner = RegistryInner {
            queue_policy,
            bench: state.bench,
            events: state.events.into_iter().collect(),
            next_event_id,
            ..Default::default()
        };
        for link in &state.network {
            inner
                .net
                .record(&link.a, &link.b, link.mbps, link.measured_unix_s);
        }
        // `LoadedState.jobs` arrives in `order_seq` order, which is what
        // `job_order` means: rebuilding it by pushing preserves both the queue
        // order and the tie-break between two jobs submitted in one second.
        for LoadedJob { record, per_node } in state.jobs {
            let job = Job::from_record(record, per_node);
            inner.recovery.restored += 1;
            // Every non-terminal job starts unaccounted for. A terminal one is
            // finished business: no agent will mention it again, it owns
            // nothing, and there is nothing left to decide about it.
            if !job.phase().is_terminal() {
                inner.reconciling.insert(job.job_id.clone());
            }
            inner.job_order.push(job.job_id.clone());
            inner.jobs.insert(job.job_id.clone(), job);
        }
        Self {
            inner: Mutex::new(inner),
            min_free_vram_b,
            store: Some(store),
        }
    }

    /// Return once every change queued so far has been committed.
    ///
    /// Callers must not hold `inner` across this: that is the whole point of
    /// the write-behind channel. A registry without a store has nothing to
    /// wait for and says so immediately.
    pub async fn flush(&self) -> Result<(), StoreError> {
        match &self.store {
            Some(store) => store.flush().await,
            None => Ok(()),
        }
    }

    /// Queue a change for the store. Cheap enough to call under the lock --
    /// that is the contract the store's writer thread exists to honour.
    fn write(&self, change: Change) {
        if let Some(store) = &self.store {
            store.write(change);
        }
    }

    /// Queue a job's durable half, looking its submission sequence up from the
    /// order it is already in.
    fn write_job(&self, inner: &RegistryInner, job_id: &str) {
        if self.store.is_none() {
            return;
        }
        let Some(job) = inner.jobs.get(job_id) else {
            return;
        };
        let order_seq = inner
            .job_order
            .iter()
            .position(|id| id == job_id)
            .unwrap_or(inner.job_order.len()) as i64;
        self.write(Change::Job(Box::new(job.to_record(order_seq))));
    }

    /// Record something that has already happened: onto the ring, which is
    /// what anyone reads, and down the same write-behind channel as every
    /// other change.
    ///
    /// Never flushed. A job record goes to disk before its ranks go out
    /// because the user is being promised something; an event is a note about
    /// the past, and waiting on a disk to finish writing one would put a
    /// syscall in front of the thing it describes.
    fn emit(&self, inner: &mut RegistryInner, event: Event) {
        let event = Event {
            id: inner.next_event_id,
            unix_s: now_s(),
            ..event
        };
        inner.next_event_id += 1;
        self.write(Change::Event(event.clone()));
        inner.events.push_back(event);
        if inner.events.len() > EVENT_RING {
            inner.events.pop_front();
        }
    }

    /// Record an event about something the registry itself cannot see -- the
    /// controller process starting, which no call through here would produce.
    pub async fn record_event(&self, event: Event) {
        let mut g = self.inner.lock().await;
        self.emit(&mut g, event);
    }

    /// The timeline, oldest first, so it reads forwards.
    ///
    /// Filtered before it is truncated: `--limit 50 --kind JOB_FAILED` means
    /// the last fifty failures, not the failures among the last fifty events.
    /// A zero limit means everything still kept.
    pub async fn events(&self, limit: usize, job_id: &str, kind: Option<EventKind>) -> Vec<Event> {
        let g = self.inner.lock().await;
        let mut matching: Vec<Event> = g
            .events
            .iter()
            .filter(|e| job_id.is_empty() || e.job_id == job_id)
            .filter(|e| kind.map(|k| e.kind == k).unwrap_or(true))
            .cloned()
            .collect();
        if limit > 0 && matching.len() > limit {
            matching.drain(..matching.len() - limit);
        }
        matching
    }

    /// Emit `NODE_LOST` and `NODE_RECOVERED` for nodes whose health has
    /// changed since the last sweep.
    ///
    /// On the edge only: a node that has been down all weekend is one event,
    /// not one per sweep, which is the whole reason the previous answer is
    /// remembered rather than recomputed and reported.
    pub async fn sweep_node_health(&self) {
        let mut g = self.inner.lock().await;
        let changed: Vec<(String, bool, i64)> = g
            .nodes
            .values()
            .filter(|n| {
                g.node_health
                    .get(&n.info.node_id)
                    .copied()
                    .unwrap_or(n.healthy())
                    != n.healthy()
            })
            .map(|n| (n.info.node_id.clone(), n.healthy(), n.last_seen))
            .collect();

        for (node_id, healthy, last_seen) in changed {
            g.node_health.insert(node_id.clone(), healthy);
            let event = if healthy {
                Event::new(EventKind::NodeRecovered).node(&node_id)
            } else {
                Event::new(EventKind::NodeLost)
                    .node(&node_id)
                    .detail(format!("silent for {}s", (now_s() - last_seen).max(0)))
            };
            self.emit(&mut g, event);
        }
    }

    fn write_status(&self, status: &JobStatus) {
        self.write(Change::Status {
            job_id: status.job_id.clone(),
            node_id: status.node_id.clone(),
            status: Box::new(status.clone()),
        });
    }

    pub async fn upsert_node(&self, info: NodeInfo) {
        let mut g = self.inner.lock().await;
        let id = info.node_id.clone();
        let detail = format!("{} GPU(s)", info.gpus.len());
        g.nodes.insert(
            id.clone(),
            Node {
                info,
                last_seen: now_s(),
            },
        );
        // A node that re-registers has just come back by definition, so the
        // health watch starts again from healthy rather than reporting a
        // recovery the registration already said.
        g.node_health.insert(id.clone(), true);
        self.emit(
            &mut g,
            Event::new(EventKind::NodeRegistered)
                .node(&id)
                .detail(detail),
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

        // A card carrying a restored job's id is that job accounted for, even
        // when this heartbeat reports no status for it: ownership is the
        // agent's own answer to "what is running here". Skipped entirely once
        // nothing is being reconciled, which is every heartbeat of a
        // controller that has not just restarted.
        if !g.reconciling.is_empty() || !g.adopted.is_empty() {
            let mut named: Vec<String> = g
                .nodes
                .get(node_id)
                .map(|n| {
                    n.info
                        .gpus
                        .iter()
                        .map(|gpu| gpu.allocated_job_id.clone())
                        .filter(|id| {
                            !id.is_empty() && (g.reconciling.contains(id) || g.adopted.contains(id))
                        })
                        .collect()
                })
                .unwrap_or_default();
            named.sort();
            named.dedup();
            for job_id in named {
                if g.reconciling.contains(&job_id) {
                    self.account_for(&mut g, &job_id);
                } else {
                    self.complete_adopted_plan(&mut g, &job_id);
                }
            }
        }
        true
    }

    /// A peer of an adopted job has reconnected: put it in the plan.
    ///
    /// The plan a job is adopted with names the nodes that had reported by
    /// then, which for a multi-node job is not necessarily all of them --
    /// agents come back one at a time. A rank missing from the plan is a rank
    /// `ferro cancel` and the reaper cannot reach, so the picture is allowed
    /// to fill in for as long as the recovery window is open.
    fn complete_adopted_plan(&self, g: &mut RegistryInner, job_id: &str) {
        let Some(job) = g.jobs.get(job_id) else {
            return;
        };
        let rebuilt = adopt_plan(&g.nodes, job_id, &job.plan);
        if rebuilt.placements.len() <= job.plan.placements.len() {
            return;
        }
        let actor = job.submitted_by.clone();
        let placed = where_it_runs(&rebuilt);
        if let Some(job) = g.jobs.get_mut(job_id) {
            job.plan = rebuilt;
        }
        self.write_job(g, job_id);
        // Its own line rather than a silent correction: the log is what says
        // how the controller arrived at the plan it is now using, and it
        // arrived at this one in two steps.
        self.emit(
            g,
            Event::new(EventKind::JobScheduled)
                .job(job_id)
                .actor(&actor)
                .detail(format!("adopted after restart: {placed}")),
        );
    }

    /// An agent has accounted for a restored job: stop reconciling it, and if
    /// the database and the cluster disagree about whether it was ever
    /// launched, believe the cluster.
    ///
    /// The disagreement is narrow but real. A job is promoted, dispatched, and
    /// the controller dies before the promote reaches disk -- write-behind is
    /// milliseconds wide, and a crash can land in it. What comes back is a
    /// queued job that is in fact already running, and re-placing it would put
    /// a second copy on cards the first one is still holding.
    ///
    /// So it is adopted rather than failed: the GPUs the agents say it holds
    /// are enough to rebuild the plan, and from there the agent's own status
    /// reports drive its phase exactly as they would for any other job.
    fn account_for(&self, g: &mut RegistryInner, job_id: &str) {
        if !g.reconciling.contains(job_id) {
            return;
        }
        let Some(job) = g.jobs.get(job_id) else {
            return;
        };
        if !job.queued {
            g.reconciling.remove(job_id);
            g.recovery.claimed += 1;
            return;
        }

        let actor = job.submitted_by.clone();
        let plan = adopt_plan(&g.nodes, job_id, &job.plan);
        if plan.placements.is_empty() {
            // Something named this job without holding a card for it, so there
            // is no plan to rebuild. Leave it queued and still reconciling: a
            // later heartbeat may name the GPUs, and if none ever does the
            // dispatcher places it as it would any other waiting job.
            return;
        }
        let placed = where_it_runs(&plan);

        g.reconciling.remove(job_id);
        g.adopted.insert(job_id.to_string());
        g.recovery.adopted += 1;
        if let Some(job) = g.jobs.get_mut(job_id) {
            job.plan = plan;
            job.queued = false;
            job.queue_req = None;
            job.queue_message.clear();
            // The same clock `promote` starts, for the same reason: the
            // wall-clock limit measures a run, not a wait. When this job
            // actually started is not recoverable -- it was some time before
            // the crash -- and now is the closest estimate that cannot reap a
            // healthy job the moment it is adopted.
            job.submitted = now_s();
        }
        self.write_job(g, job_id);
        self.emit(
            g,
            Event::new(EventKind::JobScheduled)
                .job(job_id)
                .actor(&actor)
                .detail(format!("adopted after restart: {placed}")),
        );
    }

    /// Decide what became of the restored jobs nobody accounted for.
    ///
    /// Runs once, when the recovery window closes, and is idempotent
    /// afterwards: a second call returns the same counts and writes nothing.
    ///
    /// Everything still unaccounted for and still supposed to be running is
    /// failed. That is a decision rather than an observation, and it is made
    /// because the alternative is worse: a job left `Running` forever is a
    /// lie an operator plans around -- they wait for output that is not
    /// coming, and they do not resubmit. A job failed in error costs one
    /// resubmission. The message says which of the two cases it was, because
    /// they are not the same news: a node that is up and never mentioned the
    /// job is not running it, while a node that never came back leaves the
    /// question genuinely open.
    ///
    /// The GPUs go back through the ordinary terminal path -- a terminal job
    /// owns nothing -- which is why the releases happen outside the lock.
    pub async fn close_recovery_window(&self, elapsed_s: i64) -> Recovery {
        let (recovery, lost) = {
            let mut g = self.inner.lock().await;
            if g.recovery.settled {
                return g.recovery;
            }
            g.recovery.settled = true;
            let stranded = std::mem::take(&mut g.reconciling);
            // Whatever an adopted job's plan says now is what it says: the
            // agents have had their window, and a node that has not reported
            // by now is not going to complete anybody's plan.
            g.adopted.clear();

            // Read the whole verdict out first: the reason depends on the node
            // table and the statuses depend on the plan, and both live in the
            // map the failures are about to write to.
            let mut lost: Vec<(String, String, String, Vec<JobStatus>)> = stranded
                .iter()
                .filter_map(|id| g.jobs.get(id))
                // A job still waiting for capacity is not missing. It holds
                // nothing, it claims nothing, and the dispatcher will place it
                // on its next tick exactly as if it had just been submitted --
                // failing it would throw away the queue position that
                // persistence exists to keep.
                .filter(|j| !j.queued && !j.phase().is_terminal())
                .map(|j| {
                    let reason = lost_reason(&g, j);
                    let ranks = lost_ranks(j, &reason);
                    (j.job_id.clone(), j.submitted_by.clone(), reason, ranks)
                })
                .collect();
            // A set has no order, and two runs of the same recovery should not
            // number the same failures differently.
            lost.sort_by(|a, b| a.0.cmp(&b.0));

            for (job_id, actor, reason, ranks) in &lost {
                if let Some(job) = g.jobs.get_mut(job_id) {
                    for status in ranks {
                        job.per_node.insert(status.node_id.clone(), status.clone());
                    }
                }
                for status in ranks {
                    self.write_status(status);
                }
                self.write_job(&g, job_id);
                self.emit(
                    &mut g,
                    Event::new(EventKind::JobFailed)
                        .job(job_id)
                        .actor(actor)
                        .detail(reason),
                );
            }
            g.recovery.failed = lost.len();
            let ids: Vec<String> = lost.into_iter().map(|(id, ..)| id).collect();
            (g.recovery, ids)
        };

        for job_id in &lost {
            self.release_if_done(job_id).await;
        }

        // Nothing came back off disk, so there was no recovery to time and no
        // jobs to account for. A controller starting fresh does not need a
        // line saying it reconciled nothing.
        if recovery.restored > 0 {
            self.record_event(Event::new(EventKind::Reconciled).detail(format!(
                "restored {}, claimed {}, adopted {}, failed {} in {elapsed_s}s",
                recovery.restored, recovery.claimed, recovery.adopted, recovery.failed
            )))
            .await;
        }
        recovery
    }

    /// The recovery counters as they stand.
    pub async fn recovery(&self) -> Recovery {
        self.inner.lock().await.recovery
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
                // Read the pair back rather than storing what was reported:
                // the snapshot keeps the slower of the two directions, and the
                // number worth reloading is the one it decided to keep.
                if let Some(mbps) = g.net.between(&pair.from_node, &pair.to_node, now, 0) {
                    self.write(Change::Network {
                        a: pair.from_node.clone(),
                        b: pair.to_node.clone(),
                        mbps,
                        measured_unix_s: g
                            .net
                            .measured_at(&pair.from_node, &pair.to_node)
                            .unwrap_or(now),
                    });
                }
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
                self.write(Change::Benchmark {
                    uuid: r.uuid.clone(),
                    tflops: r.tflops,
                    measured_unix_s: now,
                });
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
        let mut scheduled = None;
        if let Some(job) = g.jobs.get_mut(job_id) {
            job.plan = plan;
            job.placement = placement;
            job.queued = false;
            job.queue_req = None;
            job.queue_message.clear();
            // The wall clock starts when the job starts, not when it queued.
            job.submitted = now_s();
            scheduled = Some((job.submitted_by.clone(), where_it_runs(&job.plan)));
        }
        self.write_job(&g, job_id);
        if let Some((actor, placed)) = scheduled {
            self.emit(
                &mut g,
                Event::new(EventKind::JobScheduled)
                    .job(job_id)
                    .actor(&actor)
                    .detail(placed),
            );
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
        let actor = job.submitted_by.clone();
        job.queued = false;
        job.queue_req = None;
        job.queue_message = message.to_string();
        let status = JobStatus {
            job_id: job_id.to_string(),
            phase: phase as i32,
            message: message.to_string(),
            ended_unix_s: now_s(),
            ..Default::default()
        };
        job.per_node.insert(String::new(), status.clone());
        self.write_job(&g, job_id);
        self.write_status(&status);
        // This is how a queued job ends whatever the reason -- cancelled by
        // its owner, or given up on by the dispatcher when it ran out of
        // patience. Record the one that happened, not the one the caller's
        // name suggests.
        let kind = phase_event(phase).unwrap_or(EventKind::JobCancelled);
        self.emit(
            &mut g,
            Event::new(kind).job(job_id).actor(&actor).detail(message),
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
        let job_id = job.job_id.clone();
        let actor = job.submitted_by.clone();
        let queued = job.queued;
        g.job_order.push(job_id.clone());
        g.jobs.insert(job_id.clone(), job);
        self.write_job(&g, &job_id);
        self.emit(
            &mut g,
            Event::new(EventKind::JobSubmitted)
                .job(&job_id)
                .actor(&actor),
        );
        if queued {
            // The position is the promise the queue makes to this user, so it
            // belongs in the record of the promise being made -- by the time
            // anyone reads this back, the queue has moved on.
            let position = g.queue_position(&job_id);
            self.emit(
                &mut g,
                Event::new(EventKind::JobQueued)
                    .job(&job_id)
                    .actor(&actor)
                    .detail(format!("position {position}")),
            );
        }
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
        let job_id = status.job_id.clone();
        if !g.jobs.contains_key(&job_id) {
            return;
        }
        // An agent reporting on a job is that job accounted for, whatever
        // phase it reports: what reconciliation asks is whether anybody still
        // knows about it, not how it is going.
        self.account_for(&mut g, &job_id);
        let Some(job) = g.jobs.get_mut(&job_id) else {
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
        self.write_status(&merged);

        // The job's phase, not this rank's: ranks report the same phase over
        // and over -- every heartbeat carries one -- and a job only starts,
        // finishes or fails once. The transition is the event; the reports are
        // just how it is discovered.
        let before = job.phase();
        let actor = job.submitted_by.clone();
        job.per_node.insert(merged.node_id.clone(), merged);
        let after = job.phase();

        if after != before {
            if let Some(kind) = phase_event(after) {
                // A start needs no explanation; an ending does.
                let detail = if after.is_terminal() {
                    ending_detail(job, after)
                } else {
                    String::new()
                };
                self.emit(
                    &mut g,
                    Event::new(kind).job(&job_id).actor(&actor).detail(detail),
                );
            }
        }
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
        let mut freed: Vec<String> = Vec::new();
        for node in g.nodes.values_mut() {
            let mut indices = Vec::new();
            for gpu in node.info.gpus.iter_mut() {
                if gpu.allocated_job_id == job_id {
                    gpu.allocated_job_id.clear();
                    indices.push(gpu.index);
                }
            }
            if !indices.is_empty() {
                freed.push(name_gpus(&node.info.node_id, &indices));
            }
        }
        // Only when cards actually went back. This runs on every terminal
        // report and on every cancellation, and most of those free nothing --
        // a second rank reporting the same failure, a job that never placed.
        if !freed.is_empty() {
            let actor = g
                .jobs
                .get(job_id)
                .map(|j| j.submitted_by.clone())
                .unwrap_or_default();
            self.emit(
                &mut g,
                Event::new(EventKind::GpuReleased)
                    .job(job_id)
                    .actor(&actor)
                    .detail(freed.join(" ")),
            );
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

        // Which cards, by name: the plan that asked for them is not in front
        // of whoever reads this afterwards, and "somebody took two GPUs" is
        // not an answer to "who had gpu-a 1 at four in the morning".
        let actor = g
            .jobs
            .get(job_id)
            .map(|j| j.submitted_by.clone())
            .unwrap_or_default();
        self.emit(
            &mut g,
            Event::new(EventKind::GpuAllocated)
                .job(job_id)
                .actor(&actor)
                .detail(where_it_runs(plan)),
        );
        Ok(())
    }
}

/// Rebuild the plan a lost promote never wrote down, from the cards the agents
/// say this job holds.
///
/// `master_addr` and `master_port` are carried over from the record rather
/// than invented -- empty, for a plan that never reached disk. They exist to
/// bring ranks up at a rendezvous, and these ranks are already up.
///
/// The rank numbering is not recoverable either: the dispatcher assigned it
/// and only the ranks themselves still know it. What the plan is needed for
/// from here is which node to stop and which cards to give back, so the nodes
/// are ordered by id -- an arbitrary order that is at least the same every
/// time, rather than a `HashMap`'s, which is not.
fn adopt_plan(nodes: &HashMap<String, Node>, job_id: &str, restored: &JobPlan) -> JobPlan {
    let mut holders: Vec<&Node> = nodes
        .values()
        .filter(|n| n.info.gpus.iter().any(|g| g.allocated_job_id == job_id))
        .collect();
    holders.sort_by(|a, b| a.info.node_id.cmp(&b.info.node_id));

    let placements: Vec<JobPlacement> = holders
        .into_iter()
        .enumerate()
        .map(|(rank, n)| {
            let held: Vec<&Gpu> = n
                .info
                .gpus
                .iter()
                .filter(|g| g.allocated_job_id == job_id)
                .collect();
            JobPlacement {
                node_id: n.info.node_id.clone(),
                address: n.info.address.clone(),
                node_rank: rank as u32,
                gpu_indices: held.iter().map(|g| g.index).collect(),
                gpu_uuids: held.iter().map(|g| g.uuid.clone()).collect(),
            }
        })
        .collect();

    JobPlan {
        master_addr: restored.master_addr.clone(),
        master_port: restored.master_port,
        world_size: placements.iter().map(|p| p.gpu_indices.len() as u32).sum(),
        placements,
    }
}

/// Why a stranded job is being failed, in the words that tell the two cases
/// apart.
///
/// A node that is up and did not mention the job is evidence: the job is gone.
/// A node that never reported back is not, and saying so is the honest answer.
/// Where a job straddles both, the silent node decides -- one node we cannot
/// hear from is enough to make the whole job's fate a guess.
fn lost_reason(g: &RegistryInner, job: &Job) -> String {
    let mut placed: Vec<&str> = job
        .plan
        .placements
        .iter()
        .map(|p| p.node_id.as_str())
        .collect();
    placed.sort_unstable();
    placed.dedup();
    let silent: Vec<&str> = placed
        .iter()
        .copied()
        .filter(|id| !g.nodes.get(*id).map(|n| n.healthy()).unwrap_or(false))
        .collect();

    if silent.is_empty() {
        format!(
            "lost while the controller was down: {} reported in without it",
            placed.join(", ")
        )
    } else {
        format!(
            "fate unknown: {} has not reported since the controller restarted",
            silent.join(", ")
        )
    }
}

/// The per-rank statuses that make a stranded job terminal.
///
/// One per placement rather than a single summary entry, because a job's phase
/// is a vote of its ranks: one status under one key would leave a two-rank job
/// `Launching` forever, which is the same lie in a different column. A rank
/// that did report a terminal result keeps it -- that report was real, and
/// this one is a decision.
fn lost_ranks(job: &Job, reason: &str) -> Vec<JobStatus> {
    let now = now_s();
    job.plan
        .placements
        .iter()
        .filter(|p| {
            !job.per_node
                .get(&p.node_id)
                .map(|s| s.phase().is_terminal())
                .unwrap_or(false)
        })
        .map(|p| JobStatus {
            job_id: job.job_id.clone(),
            node_id: p.node_id.clone(),
            node_rank: p.node_rank,
            phase: JobPhase::Failed as i32,
            message: reason.to_string(),
            started_unix_s: job
                .per_node
                .get(&p.node_id)
                .map(|s| s.started_unix_s)
                .unwrap_or(0),
            ended_unix_s: now,
            ..Default::default()
        })
        .collect()
}

/// `gpu-a[0,1]`, the shortest true answer to "which cards".
fn name_gpus(node_id: &str, indices: &[u32]) -> String {
    let list: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
    format!("{node_id}[{}]", list.join(","))
}

/// Where a plan puts a job: `gpu-a[0,1] gpu-b[0,1]`.
fn where_it_runs(plan: &JobPlan) -> String {
    plan.placements
        .iter()
        .map(|p| name_gpus(&p.node_id, &p.gpu_indices))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The event a job reaching this phase is worth recording as. Phases that are
/// steps along the way rather than news -- pending, launching -- have none.
fn phase_event(phase: JobPhase) -> Option<EventKind> {
    match phase {
        JobPhase::Running => Some(EventKind::JobStarted),
        JobPhase::Succeeded => Some(EventKind::JobCompleted),
        JobPhase::Failed => Some(EventKind::JobFailed),
        JobPhase::Cancelled => Some(EventKind::JobCancelled),
        _ => None,
    }
}

/// The short half of a status. Next to "failed", what a reader wants is the
/// exit code; where there is none, the message says what happened instead.
fn status_detail(status: &JobStatus) -> String {
    if status.exit_code != 0 {
        return format!("exit {}", status.exit_code);
    }
    status
        .message
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Why a job ended, read off a rank that ended that way rather than off
/// whichever report happened to complete the picture: a rank can die before
/// its peer has reported anything at all, and then it is the peer's *start*
/// that turns the job failed. The answer worth printing is the dead rank's
/// exit code, not the peer's progress message. Lowest rank first, because
/// report order is not recorded and a `HashMap`'s order would make the same
/// failure read differently every run.
fn ending_detail(job: &Job, phase: JobPhase) -> String {
    let mut ranks: Vec<&JobStatus> = job
        .per_node
        .values()
        .filter(|s| s.phase() == phase)
        .collect();
    ranks.sort_by_key(|s| s.node_rank);
    ranks
        .into_iter()
        .map(status_detail)
        .find(|d| !d.is_empty())
        .unwrap_or_default()
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

    /// Mark the jobs the controller is still making up its mind about.
    ///
    /// Alongside `annotate_queue` and for the same reason: this is something
    /// only the registry knows, and a job asked for its own summary cannot
    /// answer it.
    pub fn annotate_recovery(&self, summaries: &mut [JobSummary]) {
        if self.reconciling.is_empty() {
            return;
        }
        for summary in summaries.iter_mut() {
            summary.reconciling = self.reconciling.contains(&summary.job_id);
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
