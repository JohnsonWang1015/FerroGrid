//! FerroGrid scheduling core.
//!
//! Two questions, deliberately kept apart:
//!
//! * **Which job runs next?** -- [`QueuePolicy`], over the waiting queue.
//! * **Where should it run?** -- [`PlacementPolicy`], over the cluster snapshot.
//!
//! Every policy here is a pure function of its inputs. It gets an immutable
//! snapshot and an injected `now`, and returns a decision. Nothing in this
//! crate awaits, opens a socket, or calls `SystemTime::now()`.
//!
//! That constraint is not fastidiousness. The term project has to compare
//! policies over synthetic workloads *and* run them on a real cluster, and a
//! policy allowed to do I/O would either not run offline at all or have to be
//! stubbed there -- at which point the simulator is measuring a second
//! implementation rather than the one that ships.

pub mod placement;
pub mod queue;

pub use placement::{
    node_verdicts, PerformancePlacement, PlacementDecision, PlacementPolicy, PlacementRequest,
    Shape,
};
pub use queue::{QueueContext, QueuePolicy, QueuedJob};

/// Why a placement could not be made.
///
/// The distinction that matters to callers is capacity versus shape: a job that
/// does not fit *right now* can wait in the queue, a job asking for zero GPUs
/// never will.
#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error("no nodes are registered")]
    NoNodes,
    #[error("requested {requested} nodes with {per_node} free GPU(s) each, but only {available} node(s) qualify")]
    NotEnoughNodes {
        requested: u32,
        per_node: u32,
        available: usize,
    },
    #[error("nodes must be >= 1 and gpus_per_node must be >= 1")]
    BadShape,
}

impl ScheduleError {
    /// Whether waiting could plausibly fix this. Capacity frees up; a malformed
    /// request does not, so `--wait` must not retry one forever.
    pub fn is_capacity(&self) -> bool {
        matches!(self, Self::NoNodes | Self::NotEnoughNodes { .. })
    }
}

/// Knobs shared by every policy.
///
/// One struct rather than a widening argument list, so adding a weight in a
/// later phase does not touch every call site.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Rendezvous port written into the plan for rank 0.
    pub master_port: u32,
    /// A GPU needs at least this much free VRAM before it may be placed on.
    pub min_free_vram_b: u64,
}

/// Everything a policy is allowed to look at.
///
/// Held by reference and never mutated: a policy reads this and returns a
/// decision. Later phases widen it (per-user usage for fair share, running
/// jobs for preemption) rather than letting policies reach outside it -- if
/// something genuinely cannot be snapshotted, it belongs in admission control,
/// which runs before the queue and may do I/O.
pub struct SchedulingContext<'a> {
    /// Injected, never read from the system clock, so aging and fair-share
    /// decisions replay identically in the simulator and in tests.
    pub now: i64,
    pub nodes: &'a [ferro_proto::NodeState],
    pub config: &'a SchedulerConfig,
}

impl<'a> SchedulingContext<'a> {
    pub fn new(now: i64, nodes: &'a [ferro_proto::NodeState], config: &'a SchedulerConfig) -> Self {
        Self { now, nodes, config }
    }
}

/// Build a queue policy by name.
///
/// One registry rather than a `match` at each call site, so adding a policy is
/// a one-line change and the error message can always list what actually
/// exists instead of guessing at what the operator meant.
pub fn queue_policy(name: &str) -> Result<std::sync::Arc<dyn QueuePolicy>, UnknownPolicy> {
    match name {
        "fifo" => Ok(std::sync::Arc::new(queue::Fifo)),
        _ => Err(UnknownPolicy {
            kind: "queue",
            given: name.to_string(),
            known: QUEUE_POLICIES,
        }),
    }
}

/// Build a placement policy by name.
pub fn placement_policy(name: &str) -> Result<std::sync::Arc<dyn PlacementPolicy>, UnknownPolicy> {
    match name {
        "performance" => Ok(std::sync::Arc::new(placement::PerformancePlacement)),
        _ => Err(UnknownPolicy {
            kind: "placement",
            given: name.to_string(),
            known: PLACEMENT_POLICIES,
        }),
    }
}

/// Every queue policy this build knows, for `--help` and error messages.
pub const QUEUE_POLICIES: &[&str] = &["fifo"];
/// Every placement policy this build knows.
pub const PLACEMENT_POLICIES: &[&str] = &["performance"];

#[derive(Debug, thiserror::Error)]
#[error("unknown {kind} policy `{given}` (known: {})", known.join(", "))]
pub struct UnknownPolicy {
    pub kind: &'static str,
    pub given: String,
    pub known: &'static [&'static str],
}
