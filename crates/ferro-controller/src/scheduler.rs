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

use ferro_proto::NodeVerdict;
use ferro_proto::{JobPlacement, JobPlan, NodeState};

#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error("no nodes are registered")]
    NoNodes,
    #[error("requested {requested} nodes with {per_node} free GPU(s) each, but only {available} node(s) qualify")]
    NotEnoughNodes {
        requested: u32,
        per_node: u32,
        available: usize,
    },
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

#[derive(Clone, Debug)]
struct GpuSelection {
    model: Option<String>,
    indices: Vec<u32>,
    score: f64,
    free_vram_b: u64,
}

/// Every homogeneous set of `want` cards this node can offer.
fn homogeneous_options(node: &NodeState, want: usize, min_free_b: u64) -> Vec<GpuSelection> {
    let free = free_gpus(node, min_free_b);
    if free.len() < want {
        return Vec::new();
    }

    let mut by_model: std::collections::HashMap<&str, Vec<&ferro_proto::Gpu>> = Default::default();
    for g in &free {
        by_model.entry(g.name.as_str()).or_default().push(g);
    }

    // HashMap iteration is deliberately unordered; model names make ties
    // deterministic while the cards in each group already follow free_gpus'
    // benchmark/VRAM order.
    let mut models: Vec<&str> = by_model.keys().copied().collect();
    models.sort_unstable();
    models
        .into_iter()
        .filter_map(|model| {
            let cards = by_model.get(model)?;
            (cards.len() >= want).then(|| selection(cards, Some(model.to_string()), want))
        })
        .collect()
}

fn selection(cards: &[&ferro_proto::Gpu], model: Option<String>, want: usize) -> GpuSelection {
    let chosen = &cards[..want];
    let mut indices: Vec<u32> = chosen.iter().map(|g| g.index).collect();
    indices.sort_unstable();
    GpuSelection {
        model,
        indices,
        score: score(chosen.iter().copied()),
        free_vram_b: chosen.iter().map(|g| gpu_free_bytes(g)).sum(),
    }
}

fn compare_selection(a: &GpuSelection, b: &GpuSelection) -> std::cmp::Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.free_vram_b.cmp(&a.free_vram_b))
        .then_with(|| a.model.cmp(&b.model))
        .then_with(|| a.indices.cmp(&b.indices))
}

/// The fastest homogeneous set, or the fastest mixed set if no model has
/// enough cards on this node.
fn best_selection(node: &NodeState, want: usize, min_free_b: u64) -> Option<GpuSelection> {
    let mut options = homogeneous_options(node, want, min_free_b);
    if options.is_empty() {
        let free = free_gpus(node, min_free_b);
        if free.len() < want {
            return None;
        }
        return Some(selection(&free, None, want));
    }
    options.sort_by(compare_selection);
    options.into_iter().next()
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

/// Explain every scheduling filter without changing the placement policy.
/// `eligible` means this node can satisfy the requested per-node shape; the
/// caller still needs enough eligible nodes to satisfy `want_nodes`.
pub fn node_verdicts(
    nodes: &[NodeState],
    gpus_per_node: u32,
    node_filter: &[String],
    min_free_vram_b: u64,
) -> Vec<NodeVerdict> {
    nodes
        .iter()
        .map(|node| {
            let node_id = node_id(node).to_string();
            let mut reasons = Vec::new();
            let mut free_gpus = 0u32;
            let mut free_vram_b = 0u64;

            if !node.healthy {
                reasons.push("node is unhealthy or heartbeat is stale".to_string());
            }

            let in_filter = node_filter.is_empty() || node_filter.iter().any(|id| id == &node_id);
            if !in_filter {
                reasons.push("node is not in the requested node filter".to_string());
            }

            if gpus_per_node == 0 {
                reasons.push("gpus_per_node must be at least 1".to_string());
            }

            match node.info.as_ref() {
                None => reasons.push("node has no info".to_string()),
                Some(info) => {
                    let allocated = info
                        .gpus
                        .iter()
                        .filter(|g| !g.allocated_job_id.is_empty())
                        .count();
                    let below_vram = info
                        .gpus
                        .iter()
                        .filter(|g| {
                            g.allocated_job_id.is_empty() && gpu_free_bytes(g) < min_free_vram_b
                        })
                        .count();
                    free_gpus = info
                        .gpus
                        .iter()
                        .filter(|g| {
                            g.allocated_job_id.is_empty() && gpu_free_bytes(g) >= min_free_vram_b
                        })
                        .count() as u32;
                    free_vram_b = info
                        .gpus
                        .iter()
                        .filter(|g| {
                            g.allocated_job_id.is_empty() && gpu_free_bytes(g) >= min_free_vram_b
                        })
                        .map(gpu_free_bytes)
                        .sum();

                    if gpus_per_node > 0 && free_gpus < gpus_per_node {
                        if allocated > 0 {
                            reasons.push(format!("{allocated} GPU(s) allocated by FerroGrid"));
                        }
                        if below_vram > 0 {
                            reasons
                                .push(format!("{below_vram} GPU(s) below the free VRAM threshold"));
                        }
                        reasons.push(format!(
                            "only {free_gpus} free GPU(s), need {gpus_per_node}"
                        ));
                    }
                }
            }

            NodeVerdict {
                node_id,
                eligible: reasons.is_empty(),
                reasons,
                free_gpus,
                free_vram_b,
            }
        })
        .collect()
}

struct Candidate<'a> {
    node: &'a NodeState,
    local: GpuSelection,
    options: Vec<GpuSelection>,
}

fn link_mbps(node: &NodeState) -> u32 {
    node.info.as_ref().map(|i| i.link_mbps).unwrap_or(0)
}

fn candidate_nodes<'a>(
    nodes: &'a [NodeState],
    node_filter: &[String],
    gpus_per_node: usize,
    min_free_b: u64,
) -> Vec<Candidate<'a>> {
    nodes
        .iter()
        .filter(|n| n.healthy)
        .filter(|n| {
            node_filter.is_empty()
                || n.info
                    .as_ref()
                    .map(|i| node_filter.contains(&i.node_id))
                    .unwrap_or(false)
        })
        .filter_map(|node| {
            let local = best_selection(node, gpus_per_node, min_free_b)?;
            Some(Candidate {
                node,
                local,
                options: homogeneous_options(node, gpus_per_node, min_free_b),
            })
        })
        .collect()
}

/// Sort one node choice from best to worst. For a multi-node job, the
/// negotiated link is considered before GPU throughput because the slowest
/// interconnect becomes the collective's ceiling.
fn node_choice_order(
    a_node: &NodeState,
    a_selection: &GpuSelection,
    b_node: &NodeState,
    b_selection: &GpuSelection,
    network_first: bool,
) -> std::cmp::Ordering {
    let network = if network_first {
        link_mbps(b_node).cmp(&link_mbps(a_node))
    } else {
        std::cmp::Ordering::Equal
    };
    network
        .then_with(|| {
            b_selection
                .score
                .partial_cmp(&a_selection.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| b_selection.free_vram_b.cmp(&a_selection.free_vram_b))
        .then_with(|| node_id(a_node).cmp(node_id(b_node)))
}

fn selection_for_model<'a>(candidate: &'a Candidate<'a>, model: &str) -> Option<&'a GpuSelection> {
    candidate
        .options
        .iter()
        .find(|selection| selection.model.as_deref() == Some(model))
}

struct PlacementChoice {
    target_model: Option<String>,
    selected: Vec<(usize, GpuSelection)>,
}

/// Build one candidate set. A target model is a preference, never a
/// requirement: if fewer than `want` nodes can provide it, the remaining
/// nodes use their local best selection.
fn placement_choice(
    candidates: &[Candidate<'_>],
    want: usize,
    target_model: Option<&str>,
    network_first: bool,
) -> PlacementChoice {
    let mut selected = Vec::with_capacity(want);
    let mut matching: Vec<usize> = target_model
        .map(|model| {
            candidates
                .iter()
                .enumerate()
                .filter(|(_, candidate)| selection_for_model(candidate, model).is_some())
                .map(|(index, _)| index)
                .collect()
        })
        .unwrap_or_default();

    if let Some(model) = target_model {
        matching.sort_by(|a, b| {
            node_choice_order(
                candidates[*a].node,
                selection_for_model(&candidates[*a], model).expect("matching model"),
                candidates[*b].node,
                selection_for_model(&candidates[*b], model).expect("matching model"),
                network_first,
            )
        });
        for index in matching.into_iter().take(want) {
            selected.push((
                index,
                selection_for_model(&candidates[index], model)
                    .expect("matching model")
                    .clone(),
            ));
        }
    }

    let mut remaining: Vec<usize> = (0..candidates.len())
        .filter(|index| !selected.iter().any(|(chosen, _)| chosen == index))
        .collect();
    remaining.sort_by(|a, b| {
        node_choice_order(
            candidates[*a].node,
            &candidates[*a].local,
            candidates[*b].node,
            &candidates[*b].local,
            network_first,
        )
    });
    for index in remaining
        .into_iter()
        .take(want.saturating_sub(selected.len()))
    {
        selected.push((index, candidates[index].local.clone()));
    }

    // Rank order should describe the same preference as node selection, even
    // when a target-model group was assembled before the fallback nodes.
    selected.sort_by(|(a, a_selection), (b, b_selection)| {
        node_choice_order(
            candidates[*a].node,
            a_selection,
            candidates[*b].node,
            b_selection,
            network_first,
        )
    });

    PlacementChoice {
        target_model: target_model.map(str::to_string),
        selected,
    }
}

fn choice_order(
    a: &PlacementChoice,
    b: &PlacementChoice,
    candidates: &[Candidate<'_>],
    network_first: bool,
) -> std::cmp::Ordering {
    let matches = |choice: &PlacementChoice| {
        choice
            .target_model
            .as_deref()
            .map(|model| {
                choice
                    .selected
                    .iter()
                    .filter(|(_, selection)| selection.model.as_deref() == Some(model))
                    .count()
            })
            .unwrap_or(0)
    };
    let order = matches(a).cmp(&matches(b));
    if order != std::cmp::Ordering::Equal {
        return order;
    }

    if network_first {
        let min_link = |choice: &PlacementChoice| {
            choice
                .selected
                .iter()
                .map(|(index, _)| link_mbps(candidates[*index].node))
                .min()
                .unwrap_or(0)
        };
        let order = min_link(a).cmp(&min_link(b));
        if order != std::cmp::Ordering::Equal {
            return order;
        }
        let total_link = |choice: &PlacementChoice| {
            choice
                .selected
                .iter()
                .map(|(index, _)| link_mbps(candidates[*index].node) as u64)
                .sum::<u64>()
        };
        let order = total_link(a).cmp(&total_link(b));
        if order != std::cmp::Ordering::Equal {
            return order;
        }
    }

    let total_score = |choice: &PlacementChoice| {
        choice
            .selected
            .iter()
            .map(|(_, selection)| selection.score)
            .sum::<f64>()
    };
    let order = total_score(a)
        .partial_cmp(&total_score(b))
        .unwrap_or(std::cmp::Ordering::Equal);
    if order != std::cmp::Ordering::Equal {
        return order;
    }

    let total_vram = |choice: &PlacementChoice| {
        choice
            .selected
            .iter()
            .map(|(_, selection)| selection.free_vram_b)
            .sum::<u64>()
    };
    let order = total_vram(a).cmp(&total_vram(b));
    if order != std::cmp::Ordering::Equal {
        return order;
    }

    // Prefer the lexicographically smaller node set for reproducibility.
    let ids = |choice: &PlacementChoice| {
        let mut ids: Vec<&str> = choice
            .selected
            .iter()
            .map(|(index, _)| node_id(candidates[*index].node))
            .collect();
        ids.sort_unstable();
        ids
    };
    ids(b).cmp(&ids(a))
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
            ScheduleError::NotEnoughNodes {
                requested: 1,
                per_node: 1,
                available: 0,
            }
        });
    }

    // For each node, the biggest identical-model group it can offer.
    let mut best: Option<(&NodeState, usize, f64)> = None;
    for n in &candidates {
        let free = free_gpus(n, min_free_vram_b);
        let mut counts: std::collections::HashMap<&str, Vec<&ferro_proto::Gpu>> =
            Default::default();
        for g in &free {
            counts.entry(g.name.as_str()).or_default().push(g);
        }
        let Some(group) = counts.values().max_by_key(|v| v.len()) else {
            continue;
        };
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
    let node_id = node
        .info
        .as_ref()
        .map(|i| i.node_id.clone())
        .unwrap_or_default();
    plan(
        nodes,
        1,
        take as u32,
        &[node_id],
        master_port,
        min_free_vram_b,
    )
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

    let candidates = candidate_nodes(nodes, node_filter, gpus_per_node as usize, min_free_vram_b);

    if (candidates.len() as u32) < want_nodes {
        return Err(ScheduleError::NotEnoughNodes {
            requested: want_nodes,
            per_node: gpus_per_node,
            available: candidates.len(),
        });
    }

    let network_first = want_nodes > 1;
    let mut target_models: Vec<String> = candidates
        .iter()
        .flat_map(|candidate| candidate.options.iter())
        .filter_map(|selection| selection.model.clone())
        .collect();
    target_models.sort_unstable();
    target_models.dedup();

    // The baseline is the old local policy. Each target-model choice then
    // gets a chance to keep the selected cards identical across nodes.
    let mut choices = vec![placement_choice(
        &candidates,
        want_nodes as usize,
        None,
        network_first,
    )];
    for model in &target_models {
        choices.push(placement_choice(
            &candidates,
            want_nodes as usize,
            Some(model),
            network_first,
        ));
    }
    let choice = choices
        .into_iter()
        .max_by(|a, b| choice_order(a, b, &candidates, network_first))
        .expect("baseline placement choice");

    let placements: Vec<JobPlacement> = choice
        .selected
        .iter()
        .enumerate()
        .map(|(rank, (candidate_index, selection))| {
            let node = candidates[*candidate_index].node;
            let info = node.info.as_ref().expect("filtered above");
            let uuids = selection
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
                gpu_indices: selection.indices.clone(),
                gpu_uuids: uuids,
            }
        })
        .collect();

    let master_addr = candidates[choice.selected[0].0]
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
        let nodes = vec![node(
            "a",
            &[(0, 20 << 30, "busy"), (1, 20 << 30, ""), (2, 20 << 30, "")],
        )];
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
        assert!(matches!(
            plan(&[n], 1, 2, &[], 29500, TEST_MIN),
            Err(ScheduleError::NotEnoughNodes { .. })
        ));
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
    fn prefers_the_same_gpu_model_across_nodes() {
        let nodes = vec![
            node_mixed("5090-a", &[(0, 40 << 30, "", "RTX 5090", 130.0)]),
            node_mixed("4090", &[(0, 40 << 30, "", "RTX 4090", 100.0)]),
            node_mixed("5090-b", &[(0, 40 << 30, "", "RTX 5090", 90.0)]),
        ];
        let p = plan(&nodes, 2, 1, &[], 29500, TEST_MIN).unwrap();
        let chosen: Vec<&str> = p.placements.iter().map(|p| p.node_id.as_str()).collect();
        assert_eq!(chosen, vec!["5090-a", "5090-b"]);
    }

    #[test]
    fn mixed_gpu_models_are_still_schedulable_when_no_pair_matches() {
        let nodes = vec![
            node_mixed("5090", &[(0, 40 << 30, "", "RTX 5090", 130.0)]),
            node_mixed("4090", &[(0, 40 << 30, "", "RTX 4090", 100.0)]),
        ];
        let p = plan(&nodes, 2, 1, &[], 29500, TEST_MIN).unwrap();
        assert_eq!(p.placements.len(), 2);
    }

    #[test]
    fn prefers_a_fast_network_combination_over_faster_gpu_scores() {
        let mut slow_link = node_mixed("slow-link", &[(0, 40 << 30, "", "RTX 5090", 200.0)]);
        let mut fast_a = node_mixed("fast-a", &[(0, 40 << 30, "", "RTX 4090", 100.0)]);
        let mut fast_b = node_mixed("fast-b", &[(0, 40 << 30, "", "RTX A6000", 90.0)]);
        slow_link.info.as_mut().unwrap().link_mbps = 100;
        fast_a.info.as_mut().unwrap().link_mbps = 1_000;
        fast_b.info.as_mut().unwrap().link_mbps = 1_000;

        let p = plan(&[slow_link, fast_a, fast_b], 2, 1, &[], 29500, TEST_MIN).unwrap();
        let chosen: Vec<&str> = p.placements.iter().map(|p| p.node_id.as_str()).collect();
        assert_eq!(chosen, vec!["fast-a", "fast-b"]);
    }

    #[test]
    fn node_verdicts_keep_the_reason_each_filter_found() {
        let mut unhealthy = node("unhealthy", &[(0, 20 << 30, "")]);
        unhealthy.healthy = false;
        let filtered = node("filtered", &[(0, 20 << 30, "")]);
        let held = node("held", &[(0, 20 << 30, "job-a")]);
        let low_vram = node("low-vram", &[(0, 1 << 30, "")]);

        let verdicts = node_verdicts(
            &[unhealthy, filtered, held, low_vram],
            1,
            &["filtered".to_string()],
            TEST_MIN,
        );

        assert!(verdicts[0].reasons.iter().any(|r| r.contains("unhealthy")));
        assert!(verdicts[1].eligible);
        assert!(verdicts[2]
            .reasons
            .iter()
            .any(|r| r.contains("allocated by FerroGrid")));
        assert!(verdicts[3]
            .reasons
            .iter()
            .any(|r| r.contains("free VRAM threshold")));
    }

    #[test]
    fn auto_keeps_the_job_on_one_node() {
        let nodes = vec![
            node_mixed(
                "a",
                &[
                    (0, 40 << 30, "", "RTX 4090", 80.0),
                    (1, 40 << 30, "", "RTX 4090", 80.0),
                ],
            ),
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
            node_mixed(
                "mixed",
                &[
                    (0, 40 << 30, "", "RTX A6000", 50.0),
                    (1, 40 << 30, "", "RTX 4090", 60.0),
                    (2, 40 << 30, "", "RTX 4090", 60.0),
                ],
            ),
            node_mixed(
                "pair",
                &[
                    (0, 40 << 30, "", "RTX 5090", 130.0),
                    (1, 40 << 30, "", "RTX 5090", 130.0),
                ],
            ),
        ];
        let p = plan_auto(&nodes, &[], 29500, TEST_MIN, u32::MAX).unwrap();
        assert_eq!(p.placements[0].node_id, "pair");
        assert_eq!(p.world_size, 2);
    }

    #[test]
    fn auto_respects_a_gpu_cap() {
        let nodes = vec![node_mixed(
            "a",
            &[
                (0, 40 << 30, "", "RTX 4090", 80.0),
                (1, 40 << 30, "", "RTX 4090", 80.0),
                (2, 40 << 30, "", "RTX 4090", 80.0),
            ],
        )];
        let p = plan_auto(&nodes, &[], 29500, TEST_MIN, 2).unwrap();
        assert_eq!(p.world_size, 2);
    }

    #[test]
    fn auto_fails_cleanly_with_nothing_free() {
        let nodes = vec![node_mixed("a", &[(0, 40 << 30, "busy", "RTX 4090", 80.0)])];
        assert!(plan_auto(&nodes, &[], 29500, TEST_MIN, u32::MAX).is_err());
    }
}
