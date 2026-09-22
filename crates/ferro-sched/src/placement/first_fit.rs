//! First fit: the first nodes that qualify, and their lowest-numbered cards.
//!
//! No preference at all beyond a stable scan order. That is the point: it is
//! the cheapest possible placement decision and the baseline the cleverer
//! strategies have to justify themselves against. If best-fit or
//! performance-aware placement cannot beat "take whatever comes first" on a
//! given workload, the extra machinery is not earning its keep.
//!
//! Its weakness is fragmentation. Scanning from the same end every time fills
//! the first nodes and leaves scattered single cards behind, so the cluster
//! ends up with plenty of free GPUs and nowhere to put a four-GPU job.

use super::engine::{assemble, auto_on_one_node, pick_independently, Offer, Pick};
use super::{PlacementDecision, PlacementPolicy, PlacementRequest, Shape};
use crate::{ScheduleError, SchedulingContext};
use ferro_proto::Gpu;
use std::cmp::Ordering;

#[derive(Debug, Default, Clone, Copy)]
pub struct FirstFit;

/// No opinion: the caller hands GPUs over in index order and a stable sort
/// keeps them there.
fn any_gpu(_: &Gpu, _: &Gpu) -> Ordering {
    Ordering::Equal
}

/// No opinion: `pick_independently` breaks the tie on node id, which is the
/// scan order first fit is defined by.
fn any_node(_: &Offer<'_>, _: &Offer<'_>) -> Ordering {
    Ordering::Equal
}

impl PlacementPolicy for FirstFit {
    fn name(&self) -> &'static str {
        "first-fit"
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
                    any_gpu,
                    any_node,
                )?
            }
            Shape::Auto { max_gpus } => {
                auto_on_one_node(ctx, &req.node_filter, max_gpus, any_gpu, any_node)?
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
    use crate::placement::tests_support::{chosen, cluster, gpu_named, node_with, place};

    #[test]
    fn takes_the_first_node_in_id_order() {
        let nodes = cluster(&[
            node_with("zulu", &[gpu_named(0, 0, "X", 99.0)]),
            node_with("alpha", &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        // Explicitly *not* the fastest: first fit does not look.
        assert_eq!(chosen(&place(&FirstFit, &nodes, 1, 1)), vec!["alpha:[0]"]);
    }

    #[test]
    fn takes_the_lowest_numbered_cards() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                gpu_named(0, 0, "X", 1.0),
                gpu_named(1, 0, "X", 99.0),
                gpu_named(2, 0, "X", 50.0),
            ],
        )]);
        assert_eq!(chosen(&place(&FirstFit, &nodes, 1, 2)), vec!["a:[0, 1]"]);
    }

    #[test]
    fn skips_a_node_that_cannot_satisfy_the_shape() {
        let nodes = cluster(&[
            node_with("a", &[gpu_named(0, 0, "X", 1.0)]),
            node_with("b", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
        ]);
        assert_eq!(chosen(&place(&FirstFit, &nodes, 1, 2)), vec!["b:[0, 1]"]);
    }

    #[test]
    fn is_deterministic() {
        let nodes = cluster(&[
            node_with("a", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
            node_with("b", &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 1.0)]),
        ]);
        let first = chosen(&place(&FirstFit, &nodes, 2, 1));
        for _ in 0..8 {
            assert_eq!(chosen(&place(&FirstFit, &nodes, 2, 1)), first);
        }
    }
}
