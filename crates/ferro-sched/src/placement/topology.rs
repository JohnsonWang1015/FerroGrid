//! Topology-aware: pick the set of nodes with the fastest slowest hop.
//!
//! A collective runs at the pace of its worst link, so the quantity to
//! maximise is the *minimum* throughput across every pair in the chosen set,
//! not the average and certainly not the sum. Two nodes on the same switch
//! beat three spread across a building even when the third has the faster
//! cards: on 1 GbE, crossing the network costs this cluster roughly 55x.
//!
//! Where `ferro net` has measured a pair, that measurement is used. Where it
//! has not, the negotiated link speed stands in -- and the decision says which
//! it used, because they are different claims. A negotiated 1000 Mb/s covers
//! the node-to-switch hop only; a NIC reporting no errors can still sit behind
//! a 100 Mb/s path, and that is precisely the case this strategy exists to
//! catch.

use super::engine::{assemble, auto_on_one_node, desc, eligible, node_id, placeable, Offer, Pick};
use super::{PlacementDecision, PlacementPolicy, PlacementRequest, Shape};
use crate::{ScheduleError, SchedulingContext};
use ferro_proto::{Gpu, NodeState};
use std::cmp::Ordering;

/// Combination search is exhaustive below this many candidate sets, which
/// covers any cluster this is meant for. Beyond it the search degrades to
/// picking the individually best-connected nodes, and says so.
const MAX_COMBINATIONS: usize = 200_000;

#[derive(Debug, Default, Clone, Copy)]
pub struct TopologyAware;

fn fastest_gpu(a: &Gpu, b: &Gpu) -> Ordering {
    desc(a.bench_tflops, b.bench_tflops)
}

fn fastest_node(a: &Offer<'_>, b: &Offer<'_>) -> Ordering {
    desc(a.compute(), b.compute())
}

/// How good the evidence for a hop is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evidence {
    /// Only the negotiated link speed, which covers the node-to-switch hop.
    Negotiated,
    /// Measured, but longer ago than the configured window.
    Stale,
    /// Measured recently enough to still describe the link.
    Fresh,
}

/// What one hop is worth, and how good the evidence for it is.
///
/// A stale measurement is capped by what was last actually seen. Letting it
/// expire all the way back to the negotiated speed would mean **forgetting
/// makes a link look faster** -- a pair measured at 90 Mb/s would quietly
/// become a 1000 Mb/s candidate a day later, and the scheduler would pick
/// exactly the path `ferro net` was run to expose. Old evidence is weaker than
/// new evidence; it is not weaker than no evidence.
fn hop(ctx: &SchedulingContext<'_>, a: &NodeState, b: &NodeState) -> (f64, Evidence) {
    let (ida, idb) = (node_id(a), node_id(b));

    // Negotiated speed of the slower end. A path is never faster than its
    // slowest NIC, though it is frequently slower than both.
    let negotiated = |n: &NodeState| n.info.as_ref().map(|i| i.link_mbps).unwrap_or(0);
    let link = match (negotiated(a), negotiated(b)) {
        (0, v) | (v, 0) => v,
        (x, y) => x.min(y),
    } as f64;

    if let Some(mbps) = ctx
        .network
        .between(ida, idb, ctx.now, ctx.config.network_max_age_s)
    {
        return (mbps, Evidence::Fresh);
    }
    // Expired, but we did once see it.
    if let Some(mbps) = ctx.network.between(ida, idb, ctx.now, 0) {
        let capped = if link > 0.0 { mbps.min(link) } else { mbps };
        return (capped, Evidence::Stale);
    }
    (link, Evidence::Negotiated)
}

/// The slowest hop in a set, and the weakest evidence behind any of them.
fn weakest_link(ctx: &SchedulingContext<'_>, set: &[&Offer<'_>]) -> (f64, Evidence) {
    let mut slowest = f64::INFINITY;
    let mut evidence = Evidence::Fresh;
    for (i, a) in set.iter().enumerate() {
        for b in set.iter().skip(i + 1) {
            let (mbps, e) = hop(ctx, a.node, b.node);
            slowest = slowest.min(mbps);
            evidence = evidence.min(e);
        }
    }
    if slowest.is_infinite() {
        // A single node has no hops at all, which is the best possible answer.
        return (f64::INFINITY, Evidence::Fresh);
    }
    (slowest, evidence)
}

fn combinations(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1usize, |acc, i| acc.saturating_mul(n - i) / (i + 1).max(1))
}

/// Every k-subset of `0..n`, in lexicographic order so the search is
/// reproducible.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

impl PlacementPolicy for TopologyAware {
    fn name(&self) -> &'static str {
        "topology"
    }

    fn place(
        &self,
        req: &PlacementRequest,
        ctx: &SchedulingContext<'_>,
    ) -> Result<PlacementDecision, ScheduleError> {
        let (want_nodes, per_node) = match req.shape {
            Shape::Explicit {
                nodes,
                gpus_per_node,
            } => {
                if nodes == 0 || gpus_per_node == 0 {
                    return Err(ScheduleError::BadShape);
                }
                (nodes as usize, gpus_per_node as usize)
            }
            // One node means no hops to be slow, so there is nothing for this
            // strategy to say beyond "the fastest node that fits".
            Shape::Auto { max_gpus } => {
                let picks =
                    auto_on_one_node(ctx, &req.node_filter, max_gpus, fastest_gpu, fastest_node)?;
                return Ok(PlacementDecision::new(
                    assemble(&picks, ctx.config.master_port)?,
                    self.name(),
                    ctx,
                ));
            }
        };

        // Each node offers its fastest `per_node` cards; which nodes go
        // together is the question this strategy actually answers.
        let mut offers: Vec<Offer<'_>> = Vec::new();
        for node in eligible(ctx, &req.node_filter) {
            let mut free = placeable(node, ctx.config.min_free_vram_b);
            if free.len() < per_node {
                continue;
            }
            free.sort_by(|a, b| fastest_gpu(a, b));
            free.truncate(per_node);
            offers.push(Offer { node, gpus: free });
        }
        if offers.len() < want_nodes {
            return Err(super::engine::shortfall(
                want_nodes as u32,
                per_node as u32,
                offers.len(),
            ));
        }

        let chosen: Vec<&Offer<'_>> =
            if want_nodes == 1 || combinations(offers.len(), want_nodes) > MAX_COMBINATIONS {
                // Nothing to combine, or too much to combine exhaustively: fall
                // back to ranking nodes individually.
                let mut ranked: Vec<&Offer<'_>> = offers.iter().collect();
                ranked.sort_by(|a, b| fastest_node(a, b).then_with(|| a.id().cmp(b.id())));
                ranked.truncate(want_nodes);
                ranked
            } else {
                let mut best: Option<(f64, f64, Vec<&Offer<'_>>)> = None;
                for combo in subsets(offers.len(), want_nodes) {
                    let set: Vec<&Offer<'_>> = combo.iter().map(|i| &offers[*i]).collect();
                    let (slowest, _) = weakest_link(ctx, &set);
                    let compute: f64 = set.iter().map(|o| o.compute()).sum();
                    let better = match &best {
                        None => true,
                        // Fastest weakest link first; equal links fall through to
                        // GPU throughput, then to node ids for reproducibility.
                        Some((b_link, b_compute, b_set)) => {
                            (slowest, compute) > (*b_link, *b_compute)
                                || ((slowest, compute) == (*b_link, *b_compute)
                                    && set.iter().map(|o| o.id()).collect::<Vec<_>>()
                                        < b_set.iter().map(|o| o.id()).collect::<Vec<_>>())
                        }
                    };
                    if better {
                        best = Some((slowest, compute, set));
                    }
                }
                best.expect("at least one combination exists").2
            };

        // Rank 0 on the fastest of the chosen nodes: it hosts the rendezvous
        // and runs the same steps as everyone else.
        let mut ordered = chosen;
        ordered.sort_by(|a, b| fastest_node(a, b).then_with(|| a.id().cmp(b.id())));

        let picks: Vec<Pick<'_>> = ordered
            .into_iter()
            .map(|o| Pick {
                node: o.node,
                indices: o.indices(),
            })
            .collect();

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
    use crate::placement::tests_support::{
        chosen, cluster, gpu_named, linked, place, place_at, place_on,
    };
    use crate::NetworkSnapshot;

    #[test]
    fn prefers_the_pair_somebody_measured_as_fast() {
        // `slow` has the fastest cards by a mile, but sits behind a 90 Mb/s
        // path. A collective across it would run at the pace of that hop.
        let nodes = cluster(&[
            linked("a", 1000, &[gpu_named(0, 0, "X", 10.0)]),
            linked("b", 1000, &[gpu_named(0, 0, "X", 10.0)]),
            linked("slow", 1000, &[gpu_named(0, 0, "X", 99.0)]),
        ]);
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 0);
        net.record("a", "slow", 90.0, 0);
        net.record("b", "slow", 90.0, 0);

        assert_eq!(
            chosen(&place_on(&TopologyAware, &nodes, &net, 2, 1)),
            vec!["a:[0]", "b:[0]"],
            "the measured fast pair, not the fast GPU"
        );
    }

    #[test]
    fn negotiated_speed_stands_in_where_nothing_was_measured() {
        // No `ferro net` results at all: the only evidence is the link speed
        // each NIC negotiated, which is better than nothing and worse than a
        // measurement.
        let nodes = cluster(&[
            linked("fast-a", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("fast-b", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("slow", 100, &[gpu_named(0, 0, "X", 99.0)]),
        ]);
        assert_eq!(
            chosen(&place(&TopologyAware, &nodes, 2, 1)),
            vec!["fast-a:[0]", "fast-b:[0]"]
        );
    }

    #[test]
    fn a_measurement_overrides_a_flattering_negotiated_speed() {
        // Every NIC says 1000. Only the measurement knows that a-to-b is
        // actually 90, which is the whole reason `ferro net` exists.
        let nodes = cluster(&[
            linked("a", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("b", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("c", 1000, &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 90.0, 0);
        net.record("a", "c", 940.0, 0);
        net.record("b", "c", 940.0, 0);

        let picked = chosen(&place_on(&TopologyAware, &nodes, &net, 2, 1));
        assert!(
            picked == vec!["a:[0]", "c:[0]"] || picked == vec!["b:[0]", "c:[0]"],
            "must avoid the a-b pair; got {picked:?}"
        );
    }

    #[test]
    fn forgetting_a_slow_link_does_not_make_it_look_fast() {
        // Every NIC negotiated 1000. a-b was measured at 90 a week ago and the
        // measurement has since expired. If expiry fell all the way back to
        // the negotiated speed, a-b would become the *best* looking pair -- and
        // the scheduler would choose the one path `ferro net` was run to
        // expose. Old evidence is weaker than new evidence, not than none.
        let nodes = cluster(&[
            linked("a", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("b", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("c", 1000, &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        let now = 7 * 86_400 + 3_600;
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 90.0, 0); // stale, and slow
        net.record("a", "c", 200.0, now - 600); // fresh
        net.record("b", "c", 200.0, now - 600); // fresh

        let picked = chosen(&place_at(&TopologyAware, &nodes, &net, now, 2, 1));
        assert!(
            picked != vec!["a:[0]", "b:[0]"],
            "the expired 90 must still count against a-b; got {picked:?}"
        );
    }

    #[test]
    fn a_fresh_measurement_replaces_a_stale_one_entirely() {
        let nodes = cluster(&[
            linked("a", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("b", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("c", 1000, &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        let now = 7 * 86_400 + 3_600;
        let mut net = NetworkSnapshot::default();
        // The cable was fixed: a-b re-measured today at 940.
        net.record("a", "b", 940.0, now - 60);
        net.record("a", "c", 200.0, now - 60);
        net.record("b", "c", 200.0, now - 60);

        assert_eq!(
            chosen(&place_at(&TopologyAware, &nodes, &net, now, 2, 1)),
            vec!["a:[0]", "b:[0]"]
        );
    }

    #[test]
    fn a_single_node_request_ignores_the_network_entirely() {
        let nodes = cluster(&[
            linked("slow-link-fast-gpu", 100, &[gpu_named(0, 0, "X", 99.0)]),
            linked("fast-link-slow-gpu", 1000, &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        assert_eq!(
            chosen(&place(&TopologyAware, &nodes, 1, 1)),
            vec!["slow-link-fast-gpu:[0]"],
            "with no hops to cross, the cards are all that matter"
        );
    }

    #[test]
    fn is_deterministic() {
        let nodes = cluster(&[
            linked("a", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("b", 1000, &[gpu_named(0, 0, "X", 1.0)]),
            linked("c", 1000, &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        let first = chosen(&place(&TopologyAware, &nodes, 2, 1));
        for _ in 0..8 {
            assert_eq!(chosen(&place(&TopologyAware, &nodes, 2, 1)), first);
        }
    }

    #[test]
    fn subsets_are_complete_and_ordered() {
        assert_eq!(subsets(4, 2).len(), 6);
        assert_eq!(subsets(4, 2)[0], vec![0, 1]);
        assert_eq!(subsets(4, 2)[5], vec![2, 3]);
        assert_eq!(subsets(3, 3), vec![vec![0, 1, 2]]);
        assert!(subsets(2, 3).is_empty());
        assert_eq!(combinations(20, 4), 4845);
    }
}
