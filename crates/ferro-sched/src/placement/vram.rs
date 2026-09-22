//! VRAM-aware: take the roomiest cards available.
//!
//! The opposite of best fit, and it exists because the two optimise against
//! different failure modes. Best fit protects the *cluster* from
//! fragmentation; this protects the *job* from running out of memory.
//!
//! On a cluster FerroGrid does not own exclusively, that is not a theoretical
//! concern. Somebody else's notebook can grow into a card between the
//! placement decision and the first forward pass, and the job that OOMs is the
//! one that was given the tightest fit. Where headroom matters more than
//! packing density, this is the strategy to run.

use super::engine::{
    assemble, auto_on_one_node, desc, free_bytes, pick_independently, Offer, Pick,
};
use super::{PlacementDecision, PlacementPolicy, PlacementRequest, Shape};
use crate::{ScheduleError, SchedulingContext};
use ferro_proto::Gpu;
use std::cmp::Ordering;

#[derive(Debug, Default, Clone, Copy)]
pub struct VramAware;

fn roomiest_gpu(a: &Gpu, b: &Gpu) -> Ordering {
    desc(free_bytes(a) as f64, free_bytes(b) as f64)
}

/// Rank nodes by the *smallest* card each one offered, not by the total.
///
/// A job is limited by its worst rank: eight GPUs where one has 9 GiB free is
/// an eight-GPU job that OOMs on rank three, however impressive the sum looks.
fn roomiest_node(a: &Offer<'_>, b: &Offer<'_>) -> Ordering {
    let floor = |o: &Offer<'_>| o.gpus.iter().map(|g| free_bytes(g)).min().unwrap_or(0);
    desc(floor(a) as f64, floor(b) as f64)
        .then_with(|| desc(a.free_vram_b() as f64, b.free_vram_b() as f64))
}

impl PlacementPolicy for VramAware {
    fn name(&self) -> &'static str {
        "vram"
    }

    fn place(
        &self,
        req: &PlacementRequest,
        ctx: &SchedulingContext<'_>,
    ) -> Result<PlacementDecision, ScheduleError> {
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
                    roomiest_gpu,
                    roomiest_node,
                )?
            }
            Shape::Auto { max_gpus } => {
                auto_on_one_node(ctx, &req.node_filter, max_gpus, roomiest_gpu, roomiest_node)?
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
    use crate::placement::best_fit::BestFit;
    use crate::placement::tests_support::{chosen, cluster, gpu_named, node_with, place};

    #[test]
    fn takes_the_card_with_the_most_headroom() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                gpu_named(0, 12 << 30, "X", 1.0), // 12 GiB free
                gpu_named(1, 2 << 30, "X", 1.0),  // 22 GiB free
                gpu_named(2, 8 << 30, "X", 1.0),  // 16 GiB free
            ],
        )]);
        assert_eq!(chosen(&place(&VramAware, &nodes, 1, 1)), vec!["a:[1]"]);
        // Best fit reaches the opposite conclusion from the same cluster,
        // which is the point of having both.
        assert_eq!(chosen(&place(&BestFit, &nodes, 1, 1)), vec!["a:[0]"]);
    }

    #[test]
    fn a_node_is_judged_by_its_worst_card_not_its_total() {
        let nodes = cluster(&[
            // 22 + 2... but rank 1 would only have 2 GiB: the job dies there.
            node_with(
                "lopsided",
                &[
                    gpu_named(0, 2 << 30, "X", 1.0),
                    gpu_named(1, 14 << 30, "X", 1.0),
                ],
            ),
            // Less in total, but nothing below 13 GiB.
            node_with(
                "even",
                &[
                    gpu_named(0, 11 << 30, "X", 1.0),
                    gpu_named(1, 11 << 30, "X", 1.0),
                ],
            ),
        ]);
        assert_eq!(
            chosen(&place(&VramAware, &nodes, 1, 2)),
            vec!["even:[0, 1]"],
            "a job is limited by its worst rank, so the node should be too"
        );
    }

    #[test]
    fn a_card_below_the_floor_is_never_offered() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                gpu_named(0, 20 << 30, "X", 1.0), // 4 GiB free, below the floor
                gpu_named(1, 10 << 30, "X", 1.0), // 14 GiB free
            ],
        )]);
        assert_eq!(chosen(&place(&VramAware, &nodes, 1, 1)), vec!["a:[1]"]);
    }

    #[test]
    fn is_deterministic() {
        let nodes = cluster(&[
            node_with("a", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
            node_with("b", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
        ]);
        let first = chosen(&place(&VramAware, &nodes, 2, 1));
        for _ in 0..8 {
            assert_eq!(chosen(&place(&VramAware, &nodes, 2, 1)), first);
        }
    }
}
