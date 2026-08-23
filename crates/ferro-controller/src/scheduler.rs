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

/// A GPU is schedulable when no live FerroGrid job holds it *and* it has
/// enough VRAM actually free for a training rank.
fn free_gpu_indices(node: &NodeState, min_free_b: u64) -> Vec<u32> {
    let Some(info) = node.info.as_ref() else {
        return Vec::new();
    };
    let mut idx: Vec<u32> = info
        .gpus
        .iter()
        .filter(|g| g.allocated_job_id.is_empty() && gpu_free_bytes(g) >= min_free_b)
        .map(|g| g.index)
        .collect();
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
    candidates.sort_by(|a, b| {
        free_vram_bytes(b, min_free_vram_b)
            .cmp(&free_vram_bytes(a, min_free_vram_b))
            .then_with(|| node_id(a).cmp(node_id(b)))
    });
    candidates.truncate(want_nodes as usize);

    let placements: Vec<JobPlacement> = candidates
        .iter()
        .enumerate()
        .map(|(rank, node)| {
            let info = node.info.as_ref().expect("filtered above");
            let chosen: Vec<u32> = free_gpu_indices(node, min_free_vram_b)
                .into_iter()
                .take(gpus_per_node as usize)
                .collect();
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
}
