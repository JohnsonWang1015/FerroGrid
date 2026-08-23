//! GPU placement.
//!
//! The MVP policy is deliberately simple and predictable: pick the N healthiest
//! nodes that each have at least `gpus_per_node` free GPUs, preferring nodes
//! with more free VRAM, and take the lowest-numbered free devices on each. Rank
//! 0 lands on the first chosen node and its NCCL address becomes MASTER_ADDR.
//!
//! "Free" means both unallocated by FerroGrid *and* holding at least
//! `min_free_vram_b` of unused VRAM. FerroGrid shares these machines with
//! workloads it does not manage, and those hold memory without holding a
//! FerroGrid allocation -- scheduling onto a device with 0.5 GiB left just
//! OOMs at the first forward pass.
//!
//! Two things beyond capacity shape the placement:
//!
//! * **Model homogeneity.** Collectives run at the pace of the slowest rank,
//!   so a job spread over a fast card and a slow one wastes the fast one. The
//!   scheduler prefers a set of identical GPUs and only mixes models when it
//!   cannot avoid it.
//! * **Measured throughput.** Where several placements are equally valid, the
//!   one with the higher benchmarked TFLOP/s wins. Names are a poor proxy:
//!   `ferro bench` measures what the hardware actually does today.

use ferro_proto::{JobPlacement, JobPlan, NodeState};


#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error("no nodes are registered")]
    NoNodes,
    #[error("requested {requested} nodes with {per_node} free GPU(s) each, but only {available} node(s) qualify")]
    NotEnoughNodes { requested: u32, per_node: u32, available: usize },
    #[error("nodes must be >= 1 and gpus_per_node must be >= 1")]
    BadShape,
}

fn gpu_free_bytes(g: &ferro_proto::Gpu) -> u64 {
    g.memory_total_b.saturating_sub(g.memory_used_b)
}

/// Schedulable GPUs on a node, best first.
fn free_gpus(node: &NodeState, min_free_b: u64) -> Vec<&ferro_proto::Gpu> {
    let Some(info) = node.info.as_ref() else {
        return Vec::new();
    };
    let mut v: Vec<&ferro_proto::Gpu> = info
        .gpus
        .iter()
        .filter(|g| g.allocated_job_id.is_empty() && gpu_free_bytes(g) >= min_free_b)
        .collect();
    // Fastest first when we have measurements, then most free VRAM, then index
    // so the choice is reproducible.
    v.sort_by(|a, b| {
        b.bench_tflops
            .partial_cmp(&a.bench_tflops)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| gpu_free_bytes(b).cmp(&gpu_free_bytes(a)))
            .then_with(|| a.index.cmp(&b.index))
    });
    v
}

/// The largest set of `want` identical GPUs on this node, or the best mixed
/// set if no single model has enough.
fn pick_homogeneous(node: &NodeState, want: usize, min_free_b: u64) -> Vec<u32> {
    let free = free_gpus(node, min_free_b);
    if free.len() < want {
        return Vec::new();
    }

    let mut by_model: std::collections::HashMap<&str, Vec<&ferro_proto::Gpu>> = Default::default();
    for g in &free {
        by_model.entry(g.name.as_str()).or_default().push(g);
    }

    // Among models with enough cards, take the fastest.
    let best = by_model
        .values()
        .filter(|v| v.len() >= want)
        .max_by(|a, b| {
            score(a[..want].iter().copied())
                .partial_cmp(&score(b[..want].iter().copied()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    let chosen: Vec<&ferro_proto::Gpu> = match best {
        Some(v) => v[..want].to_vec(),
        // Nothing homogeneous is big enough: fall back to the fastest mix.
        None => free[..want].to_vec(),
    };

    let mut idx: Vec<u32> = chosen.iter().map(|g| g.index).collect();
    idx.sort_unstable();
    idx
}

/// Total measured throughput of a GPU set. Unbenchmarked cards score by free
/// VRAM instead, on a scale small enough that any real measurement outranks
/// them -- so an unbenchmarked cluster still behaves as it did before.
fn score<'a>(gpus: impl Iterator<Item = &'a ferro_proto::Gpu>) -> f64 {
    gpus.map(|g| {
        if g.bench_tflops > 0.0 {
            g.bench_tflops
        } else {
            gpu_free_bytes(g) as f64 / (1u64 << 40) as f64
        }
    })
    .sum()
}

fn node_score(node: &NodeState, want: usize, min_free_b: u64) -> f64 {
    let free = free_gpus(node, min_free_b);
    if free.len() < want {
        return f64::MIN;
    }
    score(free[..want].iter().copied())
}

fn free_gpu_indices(node: &NodeState, min_free_b: u64) -> Vec<u32> {
    let mut idx: Vec<u32> = free_gpus(node, min_free_b).iter().map(|g| g.index).collect();
    idx.sort_unstable();
    idx
}

fn free_vram_bytes(node: &NodeState, min_free_b: u64) -> u64 {
    node.info
        .as_ref()
        .map(|i| {
            i.gpus
                .iter()
                .filter(|g| g.allocated_job_id.is_empty() && gpu_free_bytes(g) >= min_free_b)
                .map(gpu_free_bytes)
                .sum()
        })
        .unwrap_or(0)
}

/// Choose a shape as well as a placement.
///
/// Policy, in priority order, and shaped by what this cluster actually
/// measures: crossing the network costs ~55x throughput and sharding a model
/// that already fits costs ~3x, so "use every GPU" is usually the wrong
/// answer. Auto therefore keeps a job on **one** node and takes the largest
/// homogeneous set of GPUs there, preferring the node that benchmarks fastest.
pub fn plan_auto(
    nodes: &[NodeState],
    node_filter: &[String],
    master_port: u32,
    min_free_vram_b: u64,
    max_gpus: u32,
) -> Result<JobPlan, ScheduleError> {
    let candidates: Vec<&NodeState> = nodes
        .iter()
        .filter(|n| n.healthy)
        .filter(|n| {
            node_filter.is_empty()
                || n.info
                    .as_ref()
                    .map(|i| node_filter.contains(&i.node_id))
                    .unwrap_or(false)
        })
        .filter(|n| !free_gpus(n, min_free_vram_b).is_empty())
        .collect();

    if candidates.is_empty() {
        return Err(if nodes.is_empty() {
            ScheduleError::NoNodes
        } else {
            ScheduleError::NotEnoughNodes { requested: 1, per_node: 1, available: 0 }
        });
    }

    // For each node, the biggest identical-model group it can offer.
    let mut best: Option<(&NodeState, usize, f64)> = None;
    for n in &candidates {
        let free = free_gpus(n, min_free_vram_b);
        let mut counts: std::collections::HashMap<&str, Vec<&ferro_proto::Gpu>> = Default::default();
        for g in &free {
            counts.entry(g.name.as_str()).or_default().push(g);
        }
        let Some(group) = counts.values().max_by_key(|v| v.len()) else { continue };
        let take = group.len().min(max_gpus.max(1) as usize);
        let sc = score(group[..take].iter().copied());
        // More GPUs wins; equal counts are broken by measured throughput.
        let better = match best {
            None => true,
            Some((_, bt, bs)) => take > bt || (take == bt && sc > bs),
        };
        if better {
            best = Some((n, take, sc));
        }
    }

    let (node, take, _) = best.ok_or(ScheduleError::NoNodes)?;
    let node_id = node.info.as_ref().map(|i| i.node_id.clone()).unwrap_or_default();
    plan(nodes, 1, take as u32, &[node_id], master_port, min_free_vram_b)
}

pub fn plan(
    nodes: &[NodeState],
    want_nodes: u32,
    gpus_per_node: u32,
    node_filter: &[String],
    master_port: u32,
    min_free_vram_b: u64,
) -> Result<JobPlan, ScheduleError> {
    if want_nodes == 0 || gpus_per_node == 0 {
        return Err(ScheduleError::BadShape);
    }
    if nodes.is_empty() {
        return Err(ScheduleError::NoNodes);
    }

    let mut candidates: Vec<&NodeState> = nodes
        .iter()
        .filter(|n| n.healthy)
        .filter(|n| {
            node_filter.is_empty()
                || n.info
                    .as_ref()
                    .map(|i| node_filter.contains(&i.node_id))
                    .unwrap_or(false)
        })
        .filter(|n| free_gpu_indices(n, min_free_vram_b).len() as u32 >= gpus_per_node)
        .collect();

    if (candidates.len() as u32) < want_nodes {
        return Err(ScheduleError::NotEnoughNodes {
            requested: want_nodes,
            per_node: gpus_per_node,
            available: candidates.len(),
        });
    }

    // Most free VRAM first; node_id breaks ties so placement is deterministic
    // across runs, which makes debugging a flaky job much easier.
    // Fastest measured node first, then most free VRAM, then node id so the
    // placement is reproducible across runs.
    candidates.sort_by(|a, b| {
        node_score(b, gpus_per_node as usize, min_free_vram_b)
            .partial_cmp(&node_score(a, gpus_per_node as usize, min_free_vram_b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                free_vram_bytes(b, min_free_vram_b).cmp(&free_vram_bytes(a, min_free_vram_b))
            })
            .then_with(|| node_id(a).cmp(node_id(b)))
    });
    candidates.truncate(want_nodes as usize);

    let placements: Vec<JobPlacement> = candidates
        .iter()
        .enumerate()
        .map(|(rank, node)| {
            let info = node.info.as_ref().expect("filtered above");
            // Prefer identical cards within a node: a collective runs at the
            // pace of its slowest member.
            let chosen: Vec<u32> = pick_homogeneous(node, gpus_per_node as usize, min_free_vram_b);
            let uuids = chosen
                .iter()
                .filter_map(|idx| {
                    info.gpus.iter().find(|g| g.index == *idx).map(|g| g.uuid.clone())
                })
                .collect();
            JobPlacement {
                node_id: info.node_id.clone(),
                address: info.address.clone(),
                node_rank: rank as u32,
                gpu_indices: chosen,
                gpu_uuids: uuids,
            }
        })
        .collect();

    let master_addr = candidates[0]
        .info
        .as_ref()
        .map(|i| {
            if i.nccl_address.is_empty() {
                i.address.rsplit_once(':').map(|(h, _)| h.to_string()).unwrap_or_default()
            } else {
                i.nccl_address.clone()
            }
        })
        .unwrap_or_default();

    Ok(JobPlan {
        master_addr,
        master_port,
        world_size: want_nodes * gpus_per_node,
        placements,
    })
}

fn node_id(n: &NodeState) -> &str {
    n.info.as_ref().map(|i| i.node_id.as_str()).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferro_proto::{Gpu, NodeInfo};

    /// Low enough that the fixtures' "free" GPUs qualify.
    const TEST_MIN: u64 = 2 << 30;

    /// Fixture GPUs default to one model; `node_mixed` varies it.
    fn node(id: &str, gpus: &[(u32, u64, &str)]) -> NodeState {
        NodeState {
            info: Some(NodeInfo {
                node_id: id.into(),
                address: format!("{id}:7071"),
                nccl_address: format!("10.0.0.{}", id.len()),
                gpus: gpus
                    .iter()
                    .map(|(i, free, job)| Gpu {
                        index: *i,
                        uuid: format!("{id}-gpu{i}"),
                        memory_total_b: 24 << 30,
                        memory_used_b: (24u64 << 30) - free,
                        allocated_job_id: job.to_string(),
                        name: "RTX 4090".into(),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            healthy: true,
            last_seen_unix_s: 0,
            free_gpus: gpus.iter().filter(|g| g.2.is_empty()).count() as u32,
        }
    }

    #[test]
    fn picks_two_nodes_two_gpus() {
        let nodes = vec![
            node("a", &[(0, 20 << 30, ""), (1, 20 << 30, "")]),
            node("b", &[(0, 20 << 30, ""), (1, 20 << 30, "")]),
        ];
        let p = plan(&nodes, 2, 2, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.world_size, 4);
        assert_eq!(p.placements.len(), 2);
        assert_eq!(p.placements[0].node_rank, 0);
        assert_eq!(p.placements[1].node_rank, 1);
        assert_eq!(p.placements[0].gpu_indices, vec![0, 1]);
        assert!(!p.master_addr.is_empty());
    }

    #[test]
    fn skips_allocated_gpus() {
        let nodes = vec![
            node("a", &[(0, 20 << 30, "busy"), (1, 20 << 30, ""), (2, 20 << 30, "")]),
        ];
        let p = plan(&nodes, 1, 2, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].gpu_indices, vec![1, 2]);
    }

    #[test]
    fn rejects_when_not_enough_free() {
        let nodes = vec![node("a", &[(0, 20 << 30, "busy"), (1, 20 << 30, "")])];
        assert!(matches!(
            plan(&nodes, 1, 2, &[], 29500, TEST_MIN),
            Err(ScheduleError::NotEnoughNodes { .. })
        ));
    }

    #[test]
    fn unhealthy_nodes_are_not_scheduled() {
        let mut n = node("a", &[(0, 20 << 30, ""), (1, 20 << 30, "")]);
        n.healthy = false;
        assert!(matches!(plan(&[n], 1, 2, &[], 29500, TEST_MIN), Err(ScheduleError::NotEnoughNodes { .. })));
    }

    #[test]
    fn node_filter_restricts_placement() {
        let nodes = vec![
            node("a", &[(0, 20 << 30, ""), (1, 20 << 30, "")]),
            node("b", &[(0, 20 << 30, ""), (1, 20 << 30, "")]),
        ];
        let p = plan(&nodes, 1, 2, &["b".to_string()], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].node_id, "b");
    }

    #[test]
    fn rank0_has_most_free_vram() {
        let nodes = vec![
            node("a", &[(0, 6 << 30, ""), (1, 6 << 30, "")]),
            node("b", &[(0, 22 << 30, ""), (1, 22 << 30, "")]),
        ];
        let p = plan(&nodes, 2, 2, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].node_id, "b");
    }

    #[test]
    fn gpu_held_by_an_external_process_is_not_free() {
        // GPU 1 has only 0.5 GiB left because something outside FerroGrid is
        // using it, even though no FerroGrid job has allocated it.
        let nodes = vec![node(
            "a",
            &[(0, 20 << 30, ""), (1, 512 << 20, ""), (2, 20 << 30, "")],
        )];
        let p = plan(&nodes, 1, 2, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].gpu_indices, vec![0, 2]);
    }

    #[test]
    fn node_without_enough_usable_vram_is_skipped() {
        let nodes = vec![
            node("busy", &[(0, 256 << 20, ""), (1, 256 << 20, "")]),
            node("free", &[(0, 20 << 30, ""), (1, 20 << 30, "")]),
        ];
        let p = plan(&nodes, 1, 2, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].node_id, "free");
    }

    /// (index, free bytes, job, model, tflops)
    fn node_mixed(id: &str, gpus: &[(u32, u64, &str, &str, f64)]) -> NodeState {
        NodeState {
            info: Some(NodeInfo {
                node_id: id.into(),
                address: format!("{id}:7071"),
                nccl_address: format!("10.0.0.{}", id.len()),
                gpus: gpus
                    .iter()
                    .map(|(i, free, job, model, tf)| Gpu {
                        index: *i,
                        uuid: format!("{id}-gpu{i}"),
                        name: (*model).into(),
                        memory_total_b: 48 << 30,
                        memory_used_b: (48u64 << 30) - free,
                        allocated_job_id: job.to_string(),
                        bench_tflops: *tf,
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            }),
            healthy: true,
            last_seen_unix_s: 0,
            free_gpus: gpus.iter().filter(|g| g.2.is_empty()).count() as u32,
        }
    }

    #[test]
    fn prefers_identical_gpus_within_a_node() {
        // Two 4090s and one A6000 free: a 2-GPU job must take the matching
        // pair, not the fastest card plus a mismatched one.
        let nodes = vec![node_mixed(
            "a",
            &[
                (0, 40 << 30, "", "RTX A6000", 90.0),
                (1, 40 << 30, "", "RTX 4090", 80.0),
                (2, 40 << 30, "", "RTX 4090", 80.0),
            ],
        )];
        let p = plan(&nodes, 1, 2, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].gpu_indices, vec![1, 2]);
    }

    #[test]
    fn ranks_nodes_by_measured_throughput() {
        // Same VRAM everywhere, so only the benchmark can break the tie.
        let nodes = vec![
            node_mixed("slow", &[(0, 40 << 30, "", "RTX 4090", 40.0)]),
            node_mixed("fast", &[(0, 40 << 30, "", "RTX 5090", 130.0)]),
        ];
        let p = plan(&nodes, 1, 1, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements[0].node_id, "fast");
    }

    #[test]
    fn auto_keeps_the_job_on_one_node() {
        let nodes = vec![
            node_mixed("a", &[(0, 40 << 30, "", "RTX 4090", 80.0), (1, 40 << 30, "", "RTX 4090", 80.0)]),
            node_mixed("b", &[(0, 40 << 30, "", "RTX 4090", 80.0)]),
        ];
        let p = plan_auto(&nodes, &[], 29500, TEST_MIN, u32::MAX).unwrap();
        assert_eq!(p.placements.len(), 1, "auto must not span nodes");
        assert_eq!(p.world_size, 2);
        assert_eq!(p.placements[0].node_id, "a");
    }

    #[test]
    fn auto_takes_the_largest_identical_group_not_the_most_gpus() {
        // 'mixed' has three free cards but only two alike; 'pair' has two
        // alike. Both offer a 2-GPU homogeneous job, and 'pair' benchmarks
        // faster, so it should win.
        let nodes = vec![
            node_mixed("mixed", &[
                (0, 40 << 30, "", "RTX A6000", 50.0),
                (1, 40 << 30, "", "RTX 4090", 60.0),
                (2, 40 << 30, "", "RTX 4090", 60.0),
            ]),
            node_mixed("pair", &[
                (0, 40 << 30, "", "RTX 5090", 130.0),
                (1, 40 << 30, "", "RTX 5090", 130.0),
            ]),
        ];
        let p = plan_auto(&nodes, &[], 29500, TEST_MIN, u32::MAX).unwrap();
        assert_eq!(p.placements[0].node_id, "pair");
        assert_eq!(p.world_size, 2);
    }

    #[test]
    fn auto_respects_a_gpu_cap() {
        let nodes = vec![node_mixed("a", &[
            (0, 40 << 30, "", "RTX 4090", 80.0),
            (1, 40 << 30, "", "RTX 4090", 80.0),
            (2, 40 << 30, "", "RTX 4090", 80.0),
        ])];
        let p = plan_auto(&nodes, &[], 29500, TEST_MIN, 2).unwrap();
        assert_eq!(p.world_size, 2);
    }

    #[test]
    fn auto_fails_cleanly_with_nothing_free() {
        let nodes = vec![node_mixed("a", &[(0, 40 << 30, "busy", "RTX 4090", 80.0)])];
        assert!(plan_auto(&nodes, &[], 29500, TEST_MIN, u32::MAX).is_err());
    }
}
