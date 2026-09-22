//! Best fit: use up the tightest node, keep the roomy ones whole.
//!
//! Given a choice between a node with exactly enough cards and one with plenty
//! to spare, take the tight one. The spare capacity on the roomy node stays
//! contiguous and can still take a large job later; scattering small jobs
//! across every node is how a cluster ends up with six free GPUs and nowhere
//! to put a four-GPU job.
//!
//! That is the classic bin-packing argument, and the classic caveat applies:
//! best fit optimises the *next* allocation at the cost of leaving slivers
//! behind. Whether it beats first fit here is exactly what the placement
//! comparison is meant to measure rather than assume.

use super::engine::{asc, assemble, auto_on_one_node, free_bytes, pick_independently, Offer, Pick};
use super::{PlacementDecision, PlacementPolicy, PlacementRequest, Shape};
use crate::placement::engine::placeable;
use crate::{ScheduleError, SchedulingContext};
use ferro_proto::Gpu;
use std::cmp::Ordering;

#[derive(Debug, Default, Clone, Copy)]
pub struct BestFit;

/// Within a node, prefer the cards with the *least* spare memory that still
/// clear the floor, for the same reason: keep the roomy cards available for a
/// job that needs the room.
fn tightest_gpu(a: &Gpu, b: &Gpu) -> Ordering {
    asc(free_bytes(a) as f64, free_bytes(b) as f64)
}

impl PlacementPolicy for BestFit {
    fn name(&self) -> &'static str {
        "best-fit"
    }

    fn place(
        &self,
        req: &PlacementRequest,
        ctx: &SchedulingContext<'_>,
    ) -> Result<PlacementDecision, ScheduleError> {
        let min_free = ctx.config.min_free_vram_b;
        // Fewest spare GPUs left behind wins. Computed against what the node
        // could have placed, not its total card count, so a node whose other
        // cards are already busy correctly reads as tight.
        let tightest_node = move |a: &Offer<'_>, b: &Offer<'_>| {
            let leftover = |o: &Offer<'_>| o.leftover(placeable(o.node, min_free).len());
            leftover(a)
                .cmp(&leftover(b))
                // Then the node with less spare memory on the cards it offered.
                .then_with(|| asc(a.free_vram_b() as f64, b.free_vram_b() as f64))
        };

        let picks: Vec<Pick<'_>> = match req.shape {
            Shape::Explicit {
                nodes,
                gpus_per_node,
            } => {
                if nodes == 0 || gpus_per_node == 0 {
                    return Err(ScheduleError::BadShape);
                }
                pick_independently(
                    ctx,
                    &req.node_filter,
                    nodes as usize,
                    gpus_per_node as usize,
                    tightest_gpu,
                    tightest_node,
                )?
            }
            // Auto asks for as many GPUs as one node can give, so there is no
            // leftover to minimise; the tie-break still prefers the tighter node.
            Shape::Auto { max_gpus } => {
                auto_on_one_node(ctx, &req.node_filter, max_gpus, tightest_gpu, tightest_node)?
            }
        };

        Ok(PlacementDecision::new(
            assemble(&picks, ctx.config.master_port)?,
            self.name(),
            ctx,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::first_fit::FirstFit;
    use crate::placement::tests_support::{chosen, cluster, gpu_named, node_with, place};

    /// The whole argument for best fit, in one case.
    #[test]
    fn spends_the_tight_node_and_keeps_the_roomy_one_whole() {
        let nodes = cluster(&[
            // `big` could take a four-GPU job later; `small` could not.
            node_with(
                "big",
                &[
                    gpu_named(0, 0, "X", 1.0),
                    gpu_named(1, 0, "X", 1.0),
                    gpu_named(2, 0, "X", 1.0),
                    gpu_named(3, 0, "X", 1.0),
                ],
            ),
            node_with("small", &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        assert_eq!(chosen(&place(&BestFit, &nodes, 1, 1)), vec!["small:[0]"]);
        // First fit would have taken `big`, because "big" sorts first.
        assert_eq!(chosen(&place(&FirstFit, &nodes, 1, 1)), vec!["big:[0]"]);
    }

    #[test]
    fn prefers_the_card_with_less_spare_memory() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                // 16 GiB free
                gpu_named(0, 8 << 30, "X", 1.0),
                // 4 GiB free -- below the 8 GiB floor, not placeable at all
                gpu_named(1, 20 << 30, "X", 1.0),
                // 12 GiB free: the tightest card that still clears the floor
                gpu_named(2, 12 << 30, "X", 1.0),
            ],
        )]);
        assert_eq!(chosen(&place(&BestFit, &nodes, 1, 1)), vec!["a:[2]"]);
    }

    #[test]
    fn falls_back_to_node_id_when_nodes_are_equally_tight() {
        let nodes = cluster(&[
            node_with("b", &[gpu_named(0, 0, "X", 1.0)]),
            node_with("a", &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        assert_eq!(chosen(&place(&BestFit, &nodes, 1, 1)), vec!["a:[0]"]);
    }

    #[test]
    fn is_deterministic() {
        let nodes = cluster(&[
            node_with("a", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
            node_with("b", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
        ]);
        let first = chosen(&place(&BestFit, &nodes, 2, 1));
        for _ in 0..8 {
            assert_eq!(chosen(&place(&BestFit, &nodes, 2, 1)), first);
        }
    }
}
