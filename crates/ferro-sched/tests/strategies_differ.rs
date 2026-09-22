//! Do the placement strategies actually disagree?
//!
//! Five strategies that all pick the same GPUs would be five names for one
//! policy, and the placement comparison in the experiments would measure
//! nothing. This drives every strategy over one deliberately awkward cluster
//! and asserts that they reach different answers -- and that each one reaches
//! the answer its own documentation claims.
//!
//! The cluster is built so that no single choice is best on every axis, and so
//! that the negotiated link speed is *uninformative*: every NIC reports 1000
//! Mb/s, and only the measurement knows that alpha sits behind a 90 Mb/s path.
//! That is precisely the case `ferro net` exists to catch, and the one that
//! separates measured topology awareness from the negotiated-speed heuristic.
//!
//! ```text
//!   node      link    GPUs (free VRAM, model, TFLOP/s)   measured to others
//!   alpha    1000     0: 22 GiB RTX4090  40.0            90 Mb/s  <- fastest cards
//!   bravo    1000     0: 10 GiB RTX4090   9.0           940 Mb/s  <- tight, mixed
//!                     1: 10 GiB A6000     9.0
//!   charlie  1000     0: 23 GiB RTX4090  12.0           940 Mb/s  <- roomiest
//!                     1: 23 GiB RTX4090  12.0
//!                     2: 23 GiB RTX4090  12.0
//! ```

use ferro_proto::{Gpu, NodeInfo, NodeState};
use ferro_sched::placement::{BestFit, FirstFit, TopologyAware, VramAware};
use ferro_sched::{
    NetworkSnapshot, PerformancePlacement, PlacementPolicy, PlacementRequest, PlacementWeights,
    SchedulerConfig, SchedulingContext, Shape,
};

const CARD: u64 = 24 << 30;
const FLOOR: u64 = 8 << 30;

fn gpu(index: u32, free_gib: u64, model: &str, tflops: f64) -> Gpu {
    Gpu {
        index,
        uuid: format!("uuid-{model}-{index}"),
        name: model.into(),
        memory_total_b: CARD,
        memory_used_b: CARD - (free_gib << 30),
        bench_tflops: tflops,
        ..Default::default()
    }
}

fn node(id: &str, link_mbps: u32, gpus: Vec<Gpu>) -> NodeState {
    NodeState {
        info: Some(NodeInfo {
            node_id: id.into(),
            address: format!("http://{id}:7071"),
            nccl_address: format!("10.0.0.{}", id.len()),
            link_mbps,
            gpus,
            ..Default::default()
        }),
        healthy: true,
        last_seen_unix_s: 0,
        free_gpus: 0,
    }
}

fn cluster() -> Vec<NodeState> {
    vec![
        node("alpha", 1000, vec![gpu(0, 22, "RTX4090", 40.0)]),
        node(
            "bravo",
            1000,
            vec![gpu(0, 10, "RTX4090", 9.0), gpu(1, 10, "A6000", 9.0)],
        ),
        node(
            "charlie",
            1000,
            vec![
                gpu(0, 23, "RTX4090", 12.0),
                gpu(1, 23, "RTX4090", 12.0),
                gpu(2, 23, "RTX4090", 12.0),
            ],
        ),
    ]
}

fn measured() -> NetworkSnapshot {
    let mut net = NetworkSnapshot::default();
    // alpha is behind a slow path however good its NIC claims to be.
    net.record("alpha", "bravo", 90.0, 0);
    net.record("alpha", "charlie", 90.0, 0);
    net.record("bravo", "charlie", 940.0, 0);
    net
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        master_port: 29500,
        min_free_vram_b: FLOOR,
        network_max_age_s: 86_400,
        placement_weights: PlacementWeights::default(),
    }
}

/// "node:[gpus]" per rank -- a whole decision in one comparable line.
fn decide(policy: &dyn PlacementPolicy, nodes: u32, per_node: u32) -> Vec<String> {
    let snapshot = cluster();
    let net = measured();
    let config = config();
    let ctx = SchedulingContext::new(1_000, &snapshot, &config).with_network(&net);
    let req = PlacementRequest {
        shape: Shape::Explicit {
            nodes,
            gpus_per_node: per_node,
        },
        node_filter: Vec::new(),
    };
    policy
        .place(&req, &ctx)
        .expect("this cluster can satisfy the request")
        .plan
        .placements
        .iter()
        .map(|p| format!("{}:{:?}", p.node_id, p.gpu_indices))
        .collect()
}

fn policies() -> Vec<(&'static str, Box<dyn PlacementPolicy>)> {
    vec![
        ("first-fit", Box::new(FirstFit)),
        ("best-fit", Box::new(BestFit)),
        ("vram", Box::new(VramAware)),
        ("performance", Box::new(PerformancePlacement)),
        ("topology", Box::new(TopologyAware)),
    ]
}

#[test]
fn each_strategy_picks_what_its_documentation_claims() {
    // One GPU, one node: the axes pull in genuinely different directions.
    assert_eq!(
        decide(&FirstFit, 1, 1),
        vec!["alpha:[0]"],
        "first fit takes the first node in id order and looks no further"
    );
    assert_eq!(
        decide(&BestFit, 1, 1),
        vec!["alpha:[0]"],
        "best fit takes the node with the least left over -- alpha has one card"
    );
    assert_eq!(
        decide(&VramAware, 1, 1),
        vec!["charlie:[0]"],
        "vram takes the roomiest card, 23 GiB on charlie"
    );
    assert_eq!(
        decide(&PerformancePlacement, 1, 1),
        vec!["alpha:[0]"],
        "performance takes the fastest measured card, 40 TFLOP/s on alpha"
    );
}

#[test]
fn the_strategies_do_not_all_agree() {
    // If they did, four of them would be dead weight and the comparison in the
    // experiments would be measuring one policy under five names.
    let outcomes: Vec<Vec<String>> = policies().iter().map(|(_, p)| decide(&**p, 1, 1)).collect();
    let distinct: std::collections::BTreeSet<Vec<String>> = outcomes.iter().cloned().collect();
    assert!(
        distinct.len() > 1,
        "every strategy chose the same GPUs: {outcomes:?}"
    );
}

#[test]
fn topology_avoids_the_node_behind_the_slow_path() {
    // alpha has by far the fastest card and a NIC that negotiated the same
    // 1000 Mb/s as everyone else. Only the measurement condemns it.
    let picked = decide(&TopologyAware, 2, 1);
    assert!(
        !picked.iter().any(|p| p.starts_with("alpha")),
        "a two-node job must not span the 90 Mb/s path; got {picked:?}"
    );
    assert_eq!(picked, vec!["charlie:[0]", "bravo:[0]"]);
}

#[test]
fn performance_and_topology_disagree_about_a_two_node_job() {
    // The disagreement worth having, and the one that justifies storing what
    // `ferro net` measures. Every NIC negotiated 1000 Mb/s, so the existing
    // performance policy -- which ranks multi-node placements by negotiated
    // link before GPU throughput -- sees no reason to avoid alpha and takes its
    // 40 TFLOP/s card. The measurement says alpha is 90 Mb/s to anywhere.
    let perf = decide(&PerformancePlacement, 2, 1);
    let topo = decide(&TopologyAware, 2, 1);
    assert!(
        perf.iter().any(|p| p.starts_with("alpha")),
        "negotiated speed cannot distinguish alpha, so performance should take it: {perf:?}"
    );
    assert!(
        !topo.iter().any(|p| p.starts_with("alpha")),
        "the measurement should keep topology away from alpha: {topo:?}"
    );
    assert_ne!(perf, topo);
}

#[test]
fn every_strategy_is_deterministic() {
    for (name, policy) in policies() {
        let first = decide(&*policy, 2, 1);
        for _ in 0..8 {
            assert_eq!(
                decide(&*policy, 2, 1),
                first,
                "{name} gave a different answer for identical input"
            );
        }
    }
}

#[test]
fn every_strategy_respects_the_vram_floor() {
    // A card below the floor is not placeable for anyone, whatever they are
    // optimising: it is the one rule that is not a preference.
    let tight = vec![node(
        "only",
        1000,
        vec![
            gpu(0, 4, "RTX4090", 99.0), // below the 8 GiB floor
            gpu(1, 20, "RTX4090", 1.0), // the only usable card
        ],
    )];
    let net = NetworkSnapshot::default();
    let config = config();
    let ctx = SchedulingContext::new(0, &tight, &config).with_network(&net);
    let req = PlacementRequest {
        shape: Shape::Explicit {
            nodes: 1,
            gpus_per_node: 1,
        },
        node_filter: Vec::new(),
    };

    for (name, policy) in policies() {
        let plan = policy.place(&req, &ctx).expect("one card is usable").plan;
        assert_eq!(
            plan.placements[0].gpu_indices,
            vec![1],
            "{name} placed onto a card below the VRAM floor"
        );
    }
}

#[test]
fn every_strategy_scores_its_own_decision() {
    // `ferro explain` depends on this: whatever the strategy, the decision
    // comes back described in the same vocabulary.
    let snapshot = cluster();
    let net = measured();
    let config = config();
    let ctx = SchedulingContext::new(1_000, &snapshot, &config).with_network(&net);
    let req = PlacementRequest {
        shape: Shape::Explicit {
            nodes: 2,
            gpus_per_node: 1,
        },
        node_filter: Vec::new(),
    };

    for (name, policy) in policies() {
        let d = policy.place(&req, &ctx).expect("placeable");
        assert_eq!(d.policy, name);
        assert!(
            (0.0..=1.0).contains(&d.score.total),
            "{name}: total {} out of range",
            d.score.total
        );
        for (axis, value) in &d.score.components {
            assert!(
                (0.0..=1.0).contains(value),
                "{name}: {axis} = {value} out of range"
            );
        }
        assert!(!d.score.reasons.is_empty(), "{name} explained nothing");
    }
}
