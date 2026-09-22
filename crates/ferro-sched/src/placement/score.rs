//! One scoring vocabulary, shared by every placement strategy.
//!
//! The strategies disagree about what to optimise. They must not disagree
//! about how to *describe* a placement, or `ferro explain` would be comparing
//! apples with pears and an operator could not tell whether switching strategy
//! helped.
//!
//! So scoring is separate from selection. Each strategy picks by its own rule;
//! this module then measures what it picked, on five axes, against what the
//! cluster could have offered. A `compute` of 0.62 means "the chosen cards are
//! 62% as fast as the fastest set of that size available at the time" -- a
//! statement about this decision, in this cluster, at this instant, which is
//! the only kind worth printing.
//!
//! The weighted total exists for §13 and for the `cost` strategy, which is the
//! one that actually selects by it. For every other strategy the total is a
//! summary, not the thing being maximised, and the components are what to read.

use super::engine::{compute_of, eligible, free_bytes, node_id, placeable};
use crate::SchedulingContext;
use ferro_proto::{Gpu, JobPlan, NodeState};

/// How much each axis counts towards the weighted total.
///
/// Named rather than positional so a config file cannot silently transpose
/// them, and all configurable because the right trade-off depends on the
/// cluster: headroom matters more where FerroGrid shares the cards, network
/// matters more where jobs actually span nodes.
#[derive(Debug, Clone, Copy)]
pub struct PlacementWeights {
    pub compute: f64,
    pub vram: f64,
    pub homogeneity: f64,
    pub network: f64,
    pub load: f64,
}

impl Default for PlacementWeights {
    fn default() -> Self {
        // Compute and homogeneity lead because a collective runs at the pace
        // of its slowest rank, and network matters just as much the moment a
        // job spans nodes. Load is a tiebreak, not a goal.
        Self {
            compute: 1.0,
            vram: 0.5,
            homogeneity: 1.0,
            network: 1.0,
            load: 0.25,
        }
    }
}

/// What a placement is worth, on axes that mean the same thing for every
/// strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementScore {
    /// Each axis in 0..1, best at 1. `network` is absent for a single-node
    /// placement -- there is no hop to rate, and reporting 1.0 would claim a
    /// measurement nobody made.
    pub components: Vec<(&'static str, f64)>,
    /// The weighted mean of the components present, also 0..1.
    pub total: f64,
    /// Things worth saying in words rather than numbers.
    pub reasons: Vec<String>,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Every GPU the plan claims, resolved against the cluster snapshot.
fn chosen_gpus<'a>(plan: &JobPlan, nodes: &'a [NodeState]) -> Vec<&'a Gpu> {
    plan.placements
        .iter()
        .flat_map(|p| {
            nodes
                .iter()
                .filter(move |n| node_id(n) == p.node_id)
                .filter_map(|n| n.info.as_ref())
                .flat_map(move |info| {
                    info.gpus
                        .iter()
                        .filter(|g| p.gpu_indices.contains(&g.index))
                })
        })
        .collect()
}

/// The best `want` cards the cluster could have offered, by throughput.
fn best_available<'a>(ctx: &SchedulingContext<'a>, want: usize) -> Vec<&'a Gpu> {
    let mut all: Vec<&Gpu> = eligible(ctx, &[])
        .into_iter()
        .flat_map(|n| placeable(n, ctx.config.min_free_vram_b))
        .collect();
    all.sort_by(|a, b| {
        compute_of([*b].into_iter())
            .partial_cmp(&compute_of([*a].into_iter()))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all.truncate(want);
    all
}

/// Rate a plan against the cluster it was chosen from.
pub fn rate(plan: &JobPlan, ctx: &SchedulingContext<'_>) -> PlacementScore {
    let weights = ctx.config.placement_weights;
    let picked = chosen_gpus(plan, ctx.nodes);
    let mut components: Vec<(&'static str, f64)> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();

    if picked.is_empty() {
        return PlacementScore {
            components,
            total: 0.0,
            reasons: vec!["the plan names no GPUs this cluster still reports".into()],
        };
    }

    // --- compute: how close to the fastest set of this size.
    let got = compute_of(picked.iter().copied());
    let ceiling = compute_of(best_available(ctx, picked.len()).into_iter());
    let compute = if ceiling > 0.0 {
        (got / ceiling).clamp(0.0, 1.0)
    } else {
        0.0
    };
    components.push(("compute", compute));
    let benchmarked = picked.iter().filter(|g| g.bench_tflops > 0.0).count();
    if benchmarked == picked.len() {
        reasons.push(format!("{got:.1} TFLOP/s measured across the set"));
    } else if benchmarked == 0 {
        reasons.push("no GPU here has been benchmarked; ranked by free VRAM instead".into());
    } else {
        reasons.push(format!(
            "{benchmarked} of {} GPUs benchmarked; the rest ranked by free VRAM",
            picked.len()
        ));
    }

    // --- vram: the worst rank's headroom, against the roomiest card going.
    let worst = picked.iter().map(|g| free_bytes(g)).min().unwrap_or(0);
    let roomiest = eligible(ctx, &[])
        .into_iter()
        .flat_map(|n| placeable(n, ctx.config.min_free_vram_b))
        .map(free_bytes)
        .max()
        .unwrap_or(0);
    let vram = if roomiest > 0 {
        (worst as f64 / roomiest as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    components.push(("vram", vram));
    reasons.push(format!("{:.1} GiB free on the tightest card", gib(worst)));

    // --- homogeneity: a mixed set runs at the pace of its slowest model.
    let mut models: std::collections::BTreeMap<&str, usize> = Default::default();
    for g in &picked {
        *models.entry(g.name.as_str()).or_default() += 1;
    }
    let largest = models.values().copied().max().unwrap_or(0);
    let homogeneity = largest as f64 / picked.len() as f64;
    components.push(("homogeneity", homogeneity));
    if models.len() == 1 {
        reasons.push(format!(
            "all {} GPUs are {}",
            picked.len(),
            models.keys().next().copied().unwrap_or("unknown")
        ));
    } else {
        reasons.push(format!(
            "mixed GPU models ({}); the job runs at the pace of the slowest",
            models
                .iter()
                .map(|(m, n)| format!("{n}x {m}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // --- network: only meaningful once the job spans nodes.
    if plan.placements.len() > 1 {
        let ids: Vec<&str> = plan.placements.iter().map(|p| p.node_id.as_str()).collect();
        match ctx
            .network
            .slowest_among(&ids, ctx.now, ctx.config.network_max_age_s)
        {
            Some(slowest) => {
                // Against the fastest link anybody has measured here, so the
                // number means "as good as this cluster gets", not "as good as
                // Ethernet gets".
                let best = best_measured_link(ctx);
                let network = if best > 0.0 {
                    (slowest / best).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                components.push(("network", network));
                reasons.push(format!("slowest measured hop {slowest:.0} Mb/s"));
            }
            None => {
                reasons
                    .push("spans nodes whose link has not been measured; run `ferro net`".into());
            }
        }
    } else {
        reasons.push("single node, so no network hop to cross".into());
    }

    // --- load: how much of the chosen nodes was already spoken for.
    let (held, total) = plan
        .placements
        .iter()
        .filter_map(|p| ctx.nodes.iter().find(|n| node_id(n) == p.node_id))
        .filter_map(|n| n.info.as_ref())
        .fold((0usize, 0usize), |(held, total), info| {
            (
                held + info
                    .gpus
                    .iter()
                    .filter(|g| !g.allocated_job_id.is_empty())
                    .count(),
                total + info.gpus.len(),
            )
        });
    let load = if total > 0 {
        1.0 - (held as f64 / total as f64)
    } else {
        0.0
    };
    components.push(("load", load));

    // --- weighted mean over whatever axes applied.
    let weight_of = |name: &str| match name {
        "compute" => weights.compute,
        "vram" => weights.vram,
        "homogeneity" => weights.homogeneity,
        "network" => weights.network,
        "load" => weights.load,
        _ => 0.0,
    };
    let sum_w: f64 = components.iter().map(|(n, _)| weight_of(n)).sum();
    let total = if sum_w > 0.0 {
        components
            .iter()
            .map(|(n, v)| weight_of(n) * v)
            .sum::<f64>()
            / sum_w
    } else {
        0.0
    };

    PlacementScore {
        components,
        total,
        reasons,
    }
}

/// The fastest hop anybody has measured in this cluster, as the yardstick the
/// network axis is scaled against.
fn best_measured_link(ctx: &SchedulingContext<'_>) -> f64 {
    let ids: Vec<&str> = ctx.nodes.iter().map(node_id).collect();
    let mut best: f64 = 0.0;
    for (i, a) in ids.iter().enumerate() {
        for b in ids.iter().skip(i + 1) {
            if let Some(mbps) = ctx
                .network
                .between(a, b, ctx.now, ctx.config.network_max_age_s)
            {
                best = best.max(mbps);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::tests_support::{cluster, gpu_named, node_with, place, plan_of};
    use crate::placement::{first_fit::FirstFit, vram::VramAware};

    fn component(score: &PlacementScore, name: &str) -> Option<f64> {
        score
            .components
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
    }

    #[test]
    fn taking_the_fastest_cards_scores_a_perfect_compute() {
        let nodes = cluster(&[node_with(
            "a",
            &[gpu_named(0, 0, "X", 100.0), gpu_named(1, 0, "X", 1.0)],
        )]);
        let d = place(&VramAware, &nodes, 1, 2);
        // Both cards taken, so the set is by definition the best available.
        assert_eq!(component(&d.score, "compute"), Some(1.0));
    }

    #[test]
    fn leaving_the_fastest_card_behind_costs_compute() {
        let nodes = cluster(&[node_with(
            "a",
            &[gpu_named(0, 0, "X", 1.0), gpu_named(1, 0, "X", 99.0)],
        )]);
        // First fit takes index 0, the slow one.
        let d = place(&FirstFit, &nodes, 1, 1);
        let compute = component(&d.score, "compute").unwrap();
        assert!(
            compute < 0.02,
            "expected a poor compute score, got {compute}"
        );
    }

    #[test]
    fn a_uniform_set_is_perfectly_homogeneous_and_says_so() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                gpu_named(0, 0, "RTX 4090", 1.0),
                gpu_named(1, 0, "RTX 4090", 1.0),
            ],
        )]);
        let d = place(&FirstFit, &nodes, 1, 2);
        assert_eq!(component(&d.score, "homogeneity"), Some(1.0));
        assert!(d
            .score
            .reasons
            .iter()
            .any(|r| r.contains("all 2 GPUs are RTX 4090")));
    }

    #[test]
    fn a_mixed_set_is_marked_down_and_named() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                gpu_named(0, 0, "RTX 4090", 1.0),
                gpu_named(1, 0, "A6000", 1.0),
            ],
        )]);
        let d = place(&FirstFit, &nodes, 1, 2);
        assert_eq!(component(&d.score, "homogeneity"), Some(0.5));
        assert!(d
            .score
            .reasons
            .iter()
            .any(|r| r.contains("mixed GPU models")));
    }

    #[test]
    fn a_single_node_plan_has_no_network_axis() {
        // Rather than scoring 1.0, which would claim a measurement nobody made.
        let nodes = cluster(&[node_with("a", &[gpu_named(0, 0, "X", 1.0)])]);
        let d = place(&FirstFit, &nodes, 1, 1);
        assert_eq!(component(&d.score, "network"), None);
        assert!(d.score.reasons.iter().any(|r| r.contains("no network hop")));
    }

    #[test]
    fn an_unmeasured_multi_node_plan_says_so_instead_of_guessing() {
        let nodes = cluster(&[
            node_with("a", &[gpu_named(0, 0, "X", 1.0)]),
            node_with("b", &[gpu_named(0, 0, "X", 1.0)]),
        ]);
        let d = place(&FirstFit, &nodes, 2, 1);
        assert_eq!(component(&d.score, "network"), None);
        assert!(d
            .score
            .reasons
            .iter()
            .any(|r| r.contains("has not been measured")));
    }

    #[test]
    fn the_total_stays_inside_zero_to_one() {
        let nodes = cluster(&[node_with(
            "a",
            &[
                gpu_named(0, 4 << 30, "X", 7.5),
                gpu_named(1, 9 << 30, "Y", 2.0),
            ],
        )]);
        for d in [
            place(&FirstFit, &nodes, 1, 2),
            place(&VramAware, &nodes, 1, 1),
        ] {
            assert!(
                (0.0..=1.0).contains(&d.score.total),
                "total {} out of range",
                d.score.total
            );
        }
    }

    #[test]
    fn an_unbenchmarked_cluster_says_what_it_ranked_by() {
        let nodes = cluster(&[node_with("a", &[gpu_named(0, 0, "X", 0.0)])]);
        let d = place(&FirstFit, &nodes, 1, 1);
        assert!(d
            .score
            .reasons
            .iter()
            .any(|r| r.contains("has been benchmarked")));
    }

    #[test]
    fn rating_a_plan_whose_gpus_vanished_does_not_panic() {
        let nodes = cluster(&[node_with("a", &[gpu_named(0, 0, "X", 1.0)])]);
        let gone = cluster(&[node_with("a", &[])]);
        let plan = plan_of(&place(&FirstFit, &nodes, 1, 1));
        let score = crate::placement::tests_support::rate_against(&plan, &gone);
        assert_eq!(score.total, 0.0);
        assert!(score.reasons[0].contains("names no GPUs"));
    }
}
