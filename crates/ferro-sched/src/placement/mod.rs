//! Placement: *where* a job runs.
//!
//! A placement policy turns a requested shape plus a cluster snapshot into a
//! concrete [`JobPlan`] -- which nodes, which GPU indices, who is rank 0.

use crate::{ScheduleError, SchedulingContext};
use ferro_proto::JobPlan;

pub mod best_fit;
pub mod engine;
pub mod first_fit;
pub mod gpu;
pub mod score;
pub mod topology;
pub mod vram;

pub use best_fit::BestFit;
pub use first_fit::FirstFit;
pub use gpu::node_verdicts;
pub use score::{rate, PlacementScore, PlacementWeights};
pub use topology::TopologyAware;
pub use vram::VramAware;

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

/// A placement, which policy produced it, and what it is worth.
///
/// The policy name is carried because a cluster that can be reconfigured owes
/// the operator an answer to "which scheduler made this decision?", and the
/// score because "why these GPUs?" is the next question and it should not
/// require reading the source to answer.
#[derive(Debug, Clone)]
pub struct PlacementDecision {
    pub plan: JobPlan,
    pub policy: &'static str,
    pub score: PlacementScore,
}

impl PlacementDecision {
    /// Rate a plan as it is returned, so every strategy is described in the
    /// same vocabulary however it chose.
    pub fn new(plan: JobPlan, policy: &'static str, ctx: &SchedulingContext<'_>) -> Self {
        let score = score::rate(&plan, ctx);
        Self {
            plan,
            policy,
            score,
        }
    }
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

        Ok(PlacementDecision::new(plan, self.name(), ctx))
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use crate::{NetworkSnapshot, SchedulerConfig};
    use ferro_proto::{Gpu, NodeInfo, NodeState};

    pub const VRAM_FLOOR: u64 = 8 << 30;
    /// 24 GiB cards, so `used` doubles as "how much somebody else took".
    pub const CARD_BYTES: u64 = 24 << 30;

    pub fn config() -> SchedulerConfig {
        SchedulerConfig {
            master_port: 29500,
            min_free_vram_b: VRAM_FLOOR,
            network_max_age_s: 86_400,
            placement_weights: PlacementWeights::default(),
        }
    }

    pub fn gpu_named(index: u32, used_b: u64, model: &str, tflops: f64) -> Gpu {
        Gpu {
            index,
            uuid: format!("uuid-{model}-{index}"),
            name: model.into(),
            memory_total_b: CARD_BYTES,
            memory_used_b: used_b,
            bench_tflops: tflops,
            ..Default::default()
        }
    }

    pub fn node_with(id: &str, gpus: &[Gpu]) -> NodeState {
        linked(id, 0, gpus)
    }

    /// A node advertising a negotiated link speed.
    pub fn linked(id: &str, link_mbps: u32, gpus: &[Gpu]) -> NodeState {
        NodeState {
            info: Some(NodeInfo {
                node_id: id.into(),
                address: format!("http://{id}:7071"),
                nccl_address: format!("10.0.0.{}", id.len()),
                link_mbps,
                gpus: gpus.to_vec(),
                ..Default::default()
            }),
            healthy: true,
            last_seen_unix_s: 0,
            free_gpus: gpus.len() as u32,
        }
    }

    pub fn cluster(nodes: &[NodeState]) -> Vec<NodeState> {
        nodes.to_vec()
    }

    /// Place with no network measurements at all.
    pub fn place(
        policy: &dyn PlacementPolicy,
        nodes: &[NodeState],
        want_nodes: u32,
        per_node: u32,
    ) -> PlacementDecision {
        place_on(policy, nodes, NetworkSnapshot::none(), want_nodes, per_node)
    }

    pub fn place_on(
        policy: &dyn PlacementPolicy,
        nodes: &[NodeState],
        network: &NetworkSnapshot,
        want_nodes: u32,
        per_node: u32,
    ) -> PlacementDecision {
        place_at(policy, nodes, network, 0, want_nodes, per_node)
    }

    /// Place at a specific instant, for anything that depends on how old a
    /// measurement is.
    pub fn place_at(
        policy: &dyn PlacementPolicy,
        nodes: &[NodeState],
        network: &NetworkSnapshot,
        now: i64,
        want_nodes: u32,
        per_node: u32,
    ) -> PlacementDecision {
        let config = config();
        let ctx = SchedulingContext::new(now, nodes, &config).with_network(network);
        let req = PlacementRequest {
            shape: Shape::Explicit {
                nodes: want_nodes,
                gpus_per_node: per_node,
            },
            node_filter: Vec::new(),
        };
        policy.place(&req, &ctx).expect("placement should succeed")
    }

    /// "node:[indices]" per placement, in rank order -- the whole decision in
    /// one line, so a test asserts what it means to assert.
    pub fn chosen(d: &PlacementDecision) -> Vec<String> {
        d.plan
            .placements
            .iter()
            .map(|p| format!("{}:{:?}", p.node_id, p.gpu_indices))
            .collect()
    }

    pub fn plan_of(d: &PlacementDecision) -> ferro_proto::JobPlan {
        d.plan.clone()
    }

    pub fn rate_against(plan: &ferro_proto::JobPlan, nodes: &[NodeState]) -> PlacementScore {
        let config = config();
        let ctx = SchedulingContext::new(0, nodes, &config);
        score::rate(plan, &ctx)
    }
}
