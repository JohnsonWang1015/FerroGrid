//! Placement: *where* a job runs.
//!
//! A placement policy turns a requested shape plus a cluster snapshot into a
//! concrete [`JobPlan`] -- which nodes, which GPU indices, who is rank 0.

use crate::{ScheduleError, SchedulingContext};
use ferro_proto::JobPlan;

pub mod gpu;

pub use gpu::node_verdicts;

/// What the caller asked for, before any policy has looked at it.
#[derive(Debug, Clone)]
pub struct PlacementRequest {
    pub shape: Shape,
    /// Restrict placement to these node ids. Empty means "anywhere".
    pub node_filter: Vec<String>,
}

/// How many GPUs, and whether the caller or the scheduler decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Exactly this many nodes with this many GPUs each.
    Explicit { nodes: u32, gpus_per_node: u32 },
    /// Let the policy pick the shape, taking at most `max_gpus` devices.
    /// `u32::MAX` means uncapped.
    Auto { max_gpus: u32 },
}

/// A placement, plus which policy produced it.
///
/// The policy name is carried because a cluster that can be reconfigured owes
/// the operator an answer to "which scheduler made this decision?".
#[derive(Debug, Clone)]
pub struct PlacementDecision {
    pub plan: JobPlan,
    pub policy: &'static str,
}

/// "Where should this job run?"
///
/// Implementations must be pure: same request, same snapshot, same `now` =>
/// same plan, every time. Several call sites depend on that, and so does every
/// reproducible experiment.
pub trait PlacementPolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn place(
        &self,
        req: &PlacementRequest,
        ctx: &SchedulingContext<'_>,
    ) -> Result<PlacementDecision, ScheduleError>;
}

/// The policy FerroGrid has always used, now behind the trait.
///
/// Ranks hardware by measured TFLOP/s where `ferro bench` has run, prefers a
/// set of identical GPU models within and across nodes, and for multi-node jobs
/// weighs the negotiated link speed ahead of GPU throughput -- a collective
/// runs at the pace of its slowest hop.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerformancePlacement;

impl PlacementPolicy for PerformancePlacement {
    fn name(&self) -> &'static str {
        "performance"
    }

    fn place(
        &self,
        req: &PlacementRequest,
        ctx: &SchedulingContext<'_>,
    ) -> Result<PlacementDecision, ScheduleError> {
        let plan = match req.shape {
            Shape::Explicit {
                nodes,
                gpus_per_node,
            } => gpu::plan(
                ctx.nodes,
                nodes,
                gpus_per_node,
                &req.node_filter,
                ctx.config.master_port,
                ctx.config.min_free_vram_b,
            ),
            Shape::Auto { max_gpus } => gpu::plan_auto(
                ctx.nodes,
                &req.node_filter,
                ctx.config.master_port,
                ctx.config.min_free_vram_b,
                max_gpus,
            ),
        }?;

        Ok(PlacementDecision {
            plan,
            policy: self.name(),
        })
    }
}
