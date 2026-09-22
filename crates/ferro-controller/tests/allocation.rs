//! Invariants about who owns a GPU.
//!
//! These live outside the controller binary on purpose: they are the first
//! tests that could not have been written at all before the crate grew a
//! library target, and the scheduler harness in later phases depends on the
//! same seam.
//!
//! The property under test is the one a resource manager cannot get wrong:
//! **a GPU belongs to at most one job, and a finished job owns nothing.**

use ferro_controller::registry::{now_s, Job, Registry, ReserveConflict};
use ferro_proto::{Gpu, JobPhase, JobPlacement, JobPlan, JobStatus, NodeInfo};
use ferro_sched::{
    PerformancePlacement, PlacementPolicy, PlacementRequest, SchedulerConfig, SchedulingContext,
    Shape,
};
use std::collections::HashSet;
use std::sync::Arc;

const VRAM_FLOOR: u64 = 8 << 30;

fn config() -> SchedulerConfig {
    SchedulerConfig {
        master_port: 29500,
        min_free_vram_b: VRAM_FLOOR,
    }
}

fn gpu(index: u32) -> Gpu {
    Gpu {
        index,
        uuid: format!("uuid-{index}"),
        name: "Test GPU".into(),
        memory_total_b: 24 << 30,
        memory_used_b: 0,
        ..Default::default()
    }
}

fn node(id: &str, gpus: u32) -> NodeInfo {
    NodeInfo {
        node_id: id.into(),
        address: format!("http://{id}:7071"),
        nccl_address: format!("10.0.0.{}", 10 + gpus),
        gpus: (0..gpus).map(gpu).collect(),
        ..Default::default()
    }
}

fn job(id: &str) -> Job {
    let (tx, _) = tokio::sync::broadcast::channel(4);
    Job {
        job_id: id.into(),
        name: id.into(),
        submitted_by: "tester".into(),
        project: String::new(),
        priority: ferro_sched::DEFAULT_PRIORITY,
        estimated_duration_s: None,
        timeout_s: 0,
        plan: JobPlan::default(),
        per_node: Default::default(),
        submitted: now_s(),
        logs: Default::default(),
        nccl_errors: Vec::new(),
        metrics: Default::default(),
        util_sum: 0.0,
        util_n: 0,
        tx,
        queued: false,
        queue_req: None,
        queue_deadline: 0,
        node_verdicts: Vec::new(),
        warnings: Vec::new(),
        queue_message: String::new(),
    }
}

/// Plan one single-GPU job against the registry's current view.
async fn plan_one_gpu(registry: &Registry) -> Option<JobPlan> {
    let nodes = registry.node_states().await;
    let config = config();
    let ctx = SchedulingContext::new(now_s(), &nodes, &config);
    let request = PlacementRequest {
        shape: Shape::Explicit {
            nodes: 1,
            gpus_per_node: 1,
        },
        node_filter: Vec::new(),
    };
    PerformancePlacement
        .place(&request, &ctx)
        .ok()
        .map(|d| d.plan)
}

/// Every (node, gpu index) a plan asks for.
fn claimed(plan: &JobPlan) -> Vec<(String, u32)> {
    plan.placements
        .iter()
        .flat_map(|p| p.gpu_indices.iter().map(move |i| (p.node_id.clone(), *i)))
        .collect()
}

/// The defect this phase exists to fix.
///
/// Planning happens outside the registry lock, so concurrent submissions all
/// see the same free cards. Before `reserve_exact`, every one of them would
/// "succeed" and the last write would win -- two jobs granted the same GPU,
/// and the loser's release later clearing the winner's ownership.
///
/// With four GPUs and sixteen simultaneous submissions, at most four may be
/// granted, and no two of them may be granted the same device.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_submissions_never_share_a_gpu() {
    let registry = Arc::new(Registry::new(VRAM_FLOOR));
    registry.upsert_node(node("gpu-a", 4)).await;

    let mut tasks = Vec::new();
    for n in 0..16 {
        let registry = registry.clone();
        tasks.push(tokio::spawn(async move {
            let job_id = format!("j{n:02}");
            let plan = plan_one_gpu(&registry).await?;
            registry
                .reserve_exact(&plan, &job_id)
                .await
                .ok()
                .map(|()| claimed(&plan))
        }));
    }

    let mut granted: Vec<(String, u32)> = Vec::new();
    for task in tasks {
        if let Some(cards) = task.await.expect("task panicked") {
            granted.extend(cards);
        }
    }

    let unique: HashSet<&(String, u32)> = granted.iter().collect();
    assert_eq!(
        unique.len(),
        granted.len(),
        "a GPU was granted to more than one job: {granted:?}"
    );
    assert!(
        granted.len() <= 4,
        "granted {} cards but the node only has 4",
        granted.len()
    );
    assert!(!granted.is_empty(), "nobody got a GPU at all");
}

/// A refused reservation must leave the cluster exactly as it found it.
///
/// Partial reservations are what made the old code dangerous: a job left
/// owning cards it never launched on, whose release then cleared by job id.
#[tokio::test]
async fn a_refused_reservation_takes_nothing() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.upsert_node(node("gpu-a", 2)).await;

    // Somebody already holds GPU 0.
    let held = JobPlan {
        placements: vec![JobPlacement {
            node_id: "gpu-a".into(),
            gpu_indices: vec![0],
            ..Default::default()
        }],
        ..Default::default()
    };
    registry.reserve_exact(&held, "incumbent").await.unwrap();

    // A plan wanting both cards must be refused outright.
    let both = JobPlan {
        placements: vec![JobPlacement {
            node_id: "gpu-a".into(),
            gpu_indices: vec![0, 1],
            ..Default::default()
        }],
        ..Default::default()
    };
    let err = registry
        .reserve_exact(&both, "latecomer")
        .await
        .expect_err("GPU 0 is taken, so this must not succeed");
    assert!(matches!(
        err,
        ReserveConflict::AlreadyHeld { gpu_index: 0, .. }
    ));

    let owners: Vec<String> = registry.node_states().await[0]
        .info
        .as_ref()
        .unwrap()
        .gpus
        .iter()
        .map(|g| g.allocated_job_id.clone())
        .collect();
    assert_eq!(
        owners,
        vec!["incumbent".to_string(), String::new()],
        "GPU 1 must still be free: a refused reservation takes none of the cards"
    );
}

/// Re-reserving the same plan for the same job is not a conflict.
///
/// The queue dispatcher can legitimately reach this path twice for one job, and
/// a scheduler that deadlocks against its own reservation is no use.
#[tokio::test]
async fn a_job_does_not_conflict_with_itself() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.upsert_node(node("gpu-a", 2)).await;
    let plan = plan_one_gpu(&registry).await.unwrap();

    registry.reserve_exact(&plan, "mine").await.unwrap();
    registry
        .reserve_exact(&plan, "mine")
        .await
        .expect("re-reserving my own cards must be a no-op, not a conflict");
}

/// The standing invariant: **a terminal job owns no GPUs.**
///
/// Checked for each terminal phase, because the release path keys off
/// `is_terminal` and a phase added later must not quietly escape it.
#[tokio::test]
async fn a_terminal_job_owns_no_gpus() {
    for phase in [JobPhase::Succeeded, JobPhase::Failed, JobPhase::Cancelled] {
        let registry = Registry::new(VRAM_FLOOR);
        registry.upsert_node(node("gpu-a", 2)).await;

        let plan = plan_one_gpu(&registry).await.unwrap();
        let mut j = job("done");
        j.plan = plan.clone();
        registry.insert_job(j).await;
        registry.reserve_exact(&plan, "done").await.unwrap();

        // It is holding a card while it runs.
        assert!(
            registry.node_states().await[0]
                .info
                .as_ref()
                .unwrap()
                .gpus
                .iter()
                .any(|g| g.allocated_job_id == "done"),
            "{phase:?}: the job should hold a GPU before it finishes"
        );

        registry
            .update_job_status(JobStatus {
                job_id: "done".into(),
                node_id: "gpu-a".into(),
                phase: phase as i32,
                ended_unix_s: now_s(),
                ..Default::default()
            })
            .await;
        registry.release_if_done("done").await;

        let still_held: Vec<u32> = registry.node_states().await[0]
            .info
            .as_ref()
            .unwrap()
            .gpus
            .iter()
            .filter(|g| g.allocated_job_id == "done")
            .map(|g| g.index)
            .collect();
        assert!(
            still_held.is_empty(),
            "{phase:?}: terminal job still owns GPUs {still_held:?}"
        );
    }
}
