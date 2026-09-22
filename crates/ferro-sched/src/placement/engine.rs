//! Shared machinery every placement strategy builds on.
//!
//! The strategies differ in *preference*, not in what counts as possible. This
//! module owns the "possible" half -- which GPUs may take work, how a plan is
//! assembled from a set of picks -- so that a new strategy is a comparator and
//! nothing more, and so that a bug in feasibility is fixed in one place rather
//! than five.

use crate::{ScheduleError, SchedulingContext};
use ferro_proto::{Gpu, JobPlacement, JobPlan, NodeState};

/// A GPU's unused memory.
pub fn free_bytes(g: &Gpu) -> u64 {
    g.memory_total_b.saturating_sub(g.memory_used_b)
}

/// Cards on this node that could actually take work, in NVML index order.
///
/// "Could" means both unallocated by FerroGrid **and** holding enough free
/// VRAM. Those are different conditions and both matter: FerroGrid shares
/// these machines with work it does not manage, and a card somebody else
/// filled will OOM at the first forward pass however free our own table says
/// it is.
///
/// Index order, not preference order: a strategy that wants the fastest cards
/// sorts them itself, and one that wants the first available needs them in the
/// order the operator sees in `nvidia-smi`.
pub fn placeable(node: &NodeState, min_free_b: u64) -> Vec<&Gpu> {
    let Some(info) = node.info.as_ref() else {
        return Vec::new();
    };
    let mut v: Vec<&Gpu> = info
        .gpus
        .iter()
        .filter(|g| g.allocated_job_id.is_empty() && free_bytes(g) >= min_free_b)
        .collect();
    v.sort_by_key(|g| g.index);
    v
}

/// Nodes a request is allowed to land on, in node-id order.
///
/// Sorted so that every strategy starts from the same sequence: "first fit"
/// has to mean something stable, and two runs of the same scheduler over the
/// same cluster must agree.
pub fn eligible<'a>(ctx: &SchedulingContext<'a>, node_filter: &[String]) -> Vec<&'a NodeState> {
    let mut nodes: Vec<&NodeState> = ctx
        .nodes
        .iter()
        .filter(|n| n.healthy)
        .filter(|n| {
            node_filter.is_empty()
                || n.info
                    .as_ref()
                    .map(|i| node_filter.contains(&i.node_id))
                    .unwrap_or(false)
        })
        .collect();
    nodes.sort_by_key(|n| node_id(n).to_string());
    nodes
}

pub fn node_id(n: &NodeState) -> &str {
    n.info.as_ref().map(|i| i.node_id.as_str()).unwrap_or("")
}

/// Measured throughput of a GPU set, in TFLOP/s.
///
/// Cards `ferro bench` has never measured contribute their free VRAM on a
/// scale small enough that any real measurement outranks them, so an
/// unbenchmarked cluster still orders sensibly instead of treating every card
/// as equally fast.
pub fn compute_of<'a>(gpus: impl Iterator<Item = &'a Gpu>) -> f64 {
    gpus.map(|g| {
        if g.bench_tflops > 0.0 {
            g.bench_tflops
        } else {
            free_bytes(g) as f64 / (1u64 << 40) as f64
        }
    })
    .sum()
}

/// One node's contribution to a plan.
pub struct Pick<'a> {
    pub node: &'a NodeState,
    pub indices: Vec<u32>,
}

/// Turn picks into a plan, rank 0 first.
///
/// Rank 0 hosts the rendezvous, so whichever node the strategy put first
/// becomes `MASTER_ADDR`. Its NCCL address is used where it has one: on these
/// boxes the management IP and the wire NCCL uses are frequently different.
pub fn assemble(picks: &[Pick<'_>], master_port: u32) -> Result<JobPlan, ScheduleError> {
    if picks.is_empty() {
        return Err(ScheduleError::NoNodes);
    }
    let world_size: u32 = picks.iter().map(|p| p.indices.len() as u32).sum();

    let placements: Vec<JobPlacement> = picks
        .iter()
        .enumerate()
        .map(|(rank, pick)| {
            let info = pick.node.info.as_ref().expect("eligible nodes have info");
            let uuids = pick
                .indices
                .iter()
                .filter_map(|idx| {
                    info.gpus
                        .iter()
                        .find(|g| g.index == *idx)
                        .map(|g| g.uuid.clone())
                })
                .collect();
            JobPlacement {
                node_id: info.node_id.clone(),
                address: info.address.clone(),
                node_rank: rank as u32,
                gpu_indices: pick.indices.clone(),
                gpu_uuids: uuids,
            }
        })
        .collect();

    let master_addr = picks[0]
        .node
        .info
        .as_ref()
        .map(|i| {
            if i.nccl_address.is_empty() {
                i.address
                    .rsplit_once(':')
                    .map(|(h, _)| h.to_string())
                    .unwrap_or_default()
            } else {
                i.nccl_address.clone()
            }
        })
        .unwrap_or_default();

    Ok(JobPlan {
        master_addr,
        master_port,
        world_size,
        placements,
    })
}

/// Not enough nodes could offer the requested shape.
pub fn shortfall(requested: u32, per_node: u32, available: usize) -> ScheduleError {
    ScheduleError::NotEnoughNodes {
        requested,
        per_node,
        available,
    }
}

/// What one node is prepared to contribute.
pub struct Offer<'a> {
    pub node: &'a NodeState,
    pub gpus: Vec<&'a Gpu>,
}

impl Offer<'_> {
    pub fn id(&self) -> &str {
        node_id(self.node)
    }

    pub fn free_vram_b(&self) -> u64 {
        self.gpus.iter().map(|g| free_bytes(g)).sum()
    }

    pub fn compute(&self) -> f64 {
        compute_of(self.gpus.iter().copied())
    }

    /// GPUs this node would still have spare afterwards. The quantity best-fit
    /// minimises: leaving one card idle on each of four nodes is worse than
    /// leaving four idle on one, because the next four-GPU job can use the
    /// second and not the first.
    pub fn leftover(&self, placeable_total: usize) -> usize {
        placeable_total.saturating_sub(self.gpus.len())
    }

    pub fn indices(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.gpus.iter().map(|g| g.index).collect();
        v.sort_unstable();
        v
    }
}

/// The shape almost every strategy has: each node offers the `per_node` GPUs
/// it likes best, nodes are ranked, and the top `want_nodes` win.
///
/// The two comparators are the whole of a strategy's opinion. Everything else
/// -- what is feasible, how ties are broken, how a plan is built -- is fixed
/// here, so two strategies cannot disagree about the facts, only about the
/// preference. Ties always fall through to node id and GPU index, which is
/// what makes a placement reproducible.
pub fn pick_independently<'a>(
    ctx: &SchedulingContext<'a>,
    node_filter: &[String],
    want_nodes: usize,
    per_node: usize,
    gpu_pref: impl Fn(&Gpu, &Gpu) -> std::cmp::Ordering,
    node_pref: impl Fn(&Offer<'a>, &Offer<'a>) -> std::cmp::Ordering,
) -> Result<Vec<Pick<'a>>, ScheduleError> {
    let mut offers: Vec<Offer<'a>> = Vec::new();
    for node in eligible(ctx, node_filter) {
        let mut free = placeable(node, ctx.config.min_free_vram_b);
        if free.len() < per_node {
            continue;
        }
        // Stable sort on the strategy's preference, then take the head. The
        // input is already in index order, so equal-preference cards resolve
        // to the lowest index without the comparator having to say so.
        free.sort_by(|a, b| gpu_pref(a, b));
        free.truncate(per_node);
        offers.push(Offer { node, gpus: free });
    }

    if offers.len() < want_nodes {
        return Err(shortfall(want_nodes as u32, per_node as u32, offers.len()));
    }

    offers.sort_by(|a, b| node_pref(a, b).then_with(|| a.id().cmp(b.id())));
    offers.truncate(want_nodes);

    Ok(offers
        .into_iter()
        .map(|o| Pick {
            indices: o.indices(),
            node: o.node,
        })
        .collect())
}

/// Compare two floats best-first, treating NaN as equal rather than panicking.
pub fn desc(a: f64, b: f64) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

/// Compare two floats smallest-first.
pub fn asc(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Auto mode for the strategies that do not define their own.
///
/// One node, and as many of its GPUs as it can place, capped. Staying on a
/// single node is not timidity: on this cluster crossing the network costs
/// roughly 55x throughput and sharding a model that already fits costs about
/// 3x, so "use every GPU in the cluster" is usually the wrong answer to a
/// question nobody asked precisely.
///
/// Which node, when several offer the same count, is the strategy's business
/// -- that is what `node_pref` decides.
pub fn auto_on_one_node<'a>(
    ctx: &SchedulingContext<'a>,
    node_filter: &[String],
    max_gpus: u32,
    gpu_pref: impl Fn(&Gpu, &Gpu) -> std::cmp::Ordering,
    node_pref: impl Fn(&Offer<'a>, &Offer<'a>) -> std::cmp::Ordering,
) -> Result<Vec<Pick<'a>>, ScheduleError> {
    let cap = max_gpus.max(1) as usize;
    let widest = eligible(ctx, node_filter)
        .into_iter()
        .map(|n| placeable(n, ctx.config.min_free_vram_b).len().min(cap))
        .max()
        .unwrap_or(0);

    if widest == 0 {
        return Err(if ctx.nodes.is_empty() {
            ScheduleError::NoNodes
        } else {
            shortfall(1, 1, 0)
        });
    }
    pick_independently(ctx, node_filter, 1, widest, gpu_pref, node_pref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::tests_support::{gpu_named, node_with};

    #[test]
    fn a_card_somebody_else_filled_is_not_placeable() {
        let node = node_with(
            "a",
            &[
                gpu_named(0, 1 << 30, "X", 0.0),  // free
                gpu_named(1, 23 << 30, "X", 0.0), // someone else's work
            ],
        );
        let free: Vec<u32> = placeable(&node, 8 << 30).iter().map(|g| g.index).collect();
        assert_eq!(free, vec![0]);
    }

    #[test]
    fn placeable_is_in_index_order_not_preference_order() {
        // First-fit depends on this: the operator's `nvidia-smi` order is the
        // one that has to be reproduced.
        let node = node_with(
            "a",
            &[
                gpu_named(2, 0, "X", 99.0),
                gpu_named(0, 0, "X", 1.0),
                gpu_named(1, 0, "X", 50.0),
            ],
        );
        let order: Vec<u32> = placeable(&node, 0).iter().map(|g| g.index).collect();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn a_real_measurement_outranks_the_vram_proxy() {
        // The proxy is free-TiB, so an unbenchmarked 24 GiB card scores 0.023.
        // Any GPU worth scheduling benchmarks in the tens of TFLOP/s, so the
        // measured card wins by three orders of magnitude.
        let measured = gpu_named(0, 0, "X", 40.0);
        let unmeasured = gpu_named(1, 0, "X", 0.0);
        assert!(compute_of([&measured].into_iter()) > compute_of([&unmeasured].into_iter()));
    }

    #[test]
    fn the_vram_proxy_sits_just_below_a_plausible_measurement() {
        // Worth pinning down rather than hand-waving: the proxy is not zero,
        // so a measurement *below* free-TiB would lose to an unbenchmarked
        // card. That is 0.023 TFLOP/s on a 24 GiB card -- far under anything a
        // working GPU reports, but it is a boundary and not a guarantee.
        let unmeasured = gpu_named(0, 0, "X", 0.0);
        let proxy = compute_of([&unmeasured].into_iter());
        assert!((proxy - 0.0234375).abs() < 1e-9, "got {proxy}");
        assert!(compute_of([&gpu_named(1, 0, "X", 0.01)].into_iter()) < proxy);
    }

    #[test]
    fn assembling_nothing_is_an_error_not_an_empty_plan() {
        assert!(matches!(assemble(&[], 29500), Err(ScheduleError::NoNodes)));
    }
}
