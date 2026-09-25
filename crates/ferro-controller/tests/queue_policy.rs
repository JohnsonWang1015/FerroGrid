//! The wiring between the registry and the queue policy.
//!
//! The policies themselves are unit-tested in `ferro-sched`. What is tested
//! here is that the controller actually *uses* them: that a swapped policy
//! changes the order jobs are served in, that the position a user is quoted is
//! the position the dispatcher honours, and that the usage numbers feeding
//! fair share are derived from real job records rather than a ledger that can
//! drift away from them.

use ferro_controller::registry::{Job, Registry};
use ferro_proto::{Gpu, JobPhase, JobPlacement, JobPlan, JobStatus, NodeInfo, SubmitJobRequest};
use ferro_sched::queue::{Fifo, Priority};
use ferro_sched::DEFAULT_PRIORITY;
use std::sync::Arc;

const VRAM_FLOOR: u64 = 8 << 30;

/// A queued job: no plan yet, and the request kept so it can be placed later.
fn queued(id: &str, user: &str, priority: u32, submitted: i64) -> Job {
    let (tx, _) = tokio::sync::broadcast::channel(4);
    Job {
        job_id: id.into(),
        name: id.into(),
        submitted_by: user.into(),
        project: String::new(),
        priority,
        estimated_duration_s: None,
        timeout_s: 0,
        plan: JobPlan::default(),
        per_node: Default::default(),
        submitted,
        logs: Default::default(),
        nccl_errors: Vec::new(),
        metrics: Default::default(),
        util_sum: 0.0,
        util_n: 0,
        tx,
        queued: true,
        queue_req: Some(SubmitJobRequest {
            nodes: 1,
            gpus_per_node: 1,
            ..Default::default()
        }),
        queue_deadline: 0,
        node_verdicts: Vec::new(),
        warnings: Vec::new(),
        queue_message: String::new(),
        placement: None,
    }
}

/// A job that ran on `gpus` devices from `started` to `ended`.
fn finished(id: &str, user: &str, gpus: u32, started: i64, ended: i64) -> Job {
    let mut job = queued(id, user, DEFAULT_PRIORITY, started);
    job.queued = false;
    job.queue_req = None;
    job.plan = JobPlan {
        world_size: gpus,
        placements: vec![JobPlacement {
            node_id: "gpu-a".into(),
            gpu_indices: (0..gpus).collect(),
            ..Default::default()
        }],
        ..Default::default()
    };
    job.per_node.insert(
        "gpu-a".into(),
        JobStatus {
            job_id: id.into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Succeeded as i32,
            started_unix_s: started,
            ended_unix_s: ended,
            ..Default::default()
        },
    );
    job
}

#[tokio::test]
async fn the_queue_policy_decides_the_order_not_arrival() {
    let registry = Registry::with_queue_policy(VRAM_FLOOR, Arc::new(Priority));
    registry
        .insert_job(queued("early-and-dull", "alice", 10, 0))
        .await;
    registry
        .insert_job(queued("late-and-urgent", "bob", 90, 0))
        .await;

    // Arrival order would put the dull job first; the policy must not.
    assert_eq!(registry.queue_position("late-and-urgent").await, 1);
    assert_eq!(registry.queue_position("early-and-dull").await, 2);

    let served: Vec<String> = registry
        .queued_jobs()
        .await
        .into_iter()
        .map(|(id, _, _)| id)
        .collect();
    assert_eq!(served, vec!["late-and-urgent", "early-and-dull"]);
}

#[tokio::test]
async fn the_same_queue_under_fifo_is_served_by_arrival() {
    // The control for the test above: same jobs, different policy, and the
    // only thing that changed is the order.
    let registry = Registry::with_queue_policy(VRAM_FLOOR, Arc::new(Fifo));
    registry
        .insert_job(queued("early-and-dull", "alice", 10, 0))
        .await;
    registry
        .insert_job(queued("late-and-urgent", "bob", 90, 0))
        .await;

    assert_eq!(registry.queue_position("early-and-dull").await, 1);
    assert_eq!(registry.queue_position("late-and-urgent").await, 2);
}

/// The promise `ferro queue` makes: the number you are shown is the order you
/// are served in. These come from one ranking pass precisely so they cannot
/// disagree, and this asserts it end to end.
#[tokio::test]
async fn the_quoted_position_matches_the_served_order() {
    let registry = Registry::with_queue_policy(VRAM_FLOOR, Arc::new(Priority));
    for (n, priority) in [(0u32, 30u32), (1, 80), (2, 55), (3, 80), (4, 5)] {
        registry
            .insert_job(queued(&format!("j{n}"), "alice", priority, 0))
            .await;
    }

    let served: Vec<String> = registry
        .queued_jobs()
        .await
        .into_iter()
        .map(|(id, _, _)| id)
        .collect();

    for (index, job_id) in served.iter().enumerate() {
        assert_eq!(
            registry.queue_position(job_id).await,
            index as u32 + 1,
            "{job_id} is served {} but was told a different position",
            index + 1
        );
    }
    // Equal priorities still resolve by arrival, so the order is total.
    assert_eq!(served, vec!["j1", "j3", "j2", "j0", "j4"]);
}

#[tokio::test]
async fn usage_is_charged_in_gpu_seconds_per_user() {
    let registry = Registry::new(VRAM_FLOOR);
    // alice: 2 GPUs for 100s = 200. bob: 1 GPU for 50s = 50.
    registry
        .insert_job(finished("a1", "alice", 2, 1_000, 1_100))
        .await;
    registry
        .insert_job(finished("b1", "bob", 1, 1_000, 1_050))
        .await;

    let usage = registry.inner.lock().await.usage_snapshot(2_000);
    assert_eq!(usage.gpu_seconds("alice"), 200.0);
    assert_eq!(usage.gpu_seconds("bob"), 50.0);
    assert_eq!(usage.peak_gpu_seconds(), 200.0);
}

#[tokio::test]
async fn waiting_is_not_usage() {
    // A job that never started has consumed nothing, however long it queued.
    // Charging for the wait would penalise exactly the users fair share is
    // meant to protect.
    let registry = Registry::new(VRAM_FLOOR);
    registry
        .insert_job(queued("patient", "alice", DEFAULT_PRIORITY, 0))
        .await;

    let usage = registry.inner.lock().await.usage_snapshot(100_000);
    assert_eq!(usage.gpu_seconds("alice"), 0.0);
}

#[tokio::test]
async fn a_running_job_accrues_usage_as_it_goes() {
    let registry = Registry::new(VRAM_FLOOR);
    registry
        .upsert_node(NodeInfo {
            node_id: "gpu-a".into(),
            gpus: (0..2)
                .map(|index| Gpu {
                    index,
                    memory_total_b: 24 << 30,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        })
        .await;
    let mut running = finished("live", "alice", 2, 1_000, 0);
    // Still going: no end time, and the rank reports Running.
    running.per_node.get_mut("gpu-a").unwrap().phase = JobPhase::Running as i32;
    running.per_node.get_mut("gpu-a").unwrap().ended_unix_s = 0;
    let plan = running.plan.clone();
    registry.insert_job(running).await;
    registry.reserve_exact(&plan, "live").await.unwrap();

    let g = registry.inner.lock().await;
    assert_eq!(g.usage_snapshot(1_100).gpu_seconds("alice"), 200.0);
    assert_eq!(
        g.usage_snapshot(1_200).gpu_seconds("alice"),
        400.0,
        "a job still holding GPUs keeps accruing"
    );
    let snapshot = g.usage_snapshot(1_200);
    let alice = snapshot.per_user.get("alice").unwrap();
    assert_eq!(alice.running_jobs, 1);
    assert_eq!(alice.gpus_held, 2);
}

#[tokio::test]
async fn usage_is_never_negative() {
    // An invariant worth asserting rather than assuming: a clock that went
    // backwards between the start and end stamps must not produce a credit.
    let registry = Registry::new(VRAM_FLOOR);
    registry
        .insert_job(finished("weird", "alice", 4, 5_000, 1_000))
        .await;

    let usage = registry.inner.lock().await.usage_snapshot(9_000);
    assert_eq!(usage.gpu_seconds("alice"), 0.0);
}
