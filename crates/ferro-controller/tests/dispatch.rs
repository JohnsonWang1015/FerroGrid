//! The wiring between the dispatcher and the dispatch modes.
//!
//! The modes themselves are unit-tested in `ferro-sched`, over a queue and a
//! GPU count. What is tested here is that `run_queue` actually *obeys* them --
//! that swapping the mode changes which jobs it starts against a real registry,
//! with real reservations and a real placement policy underneath.
//!
//! The scenario is the one the whole feature exists for: four GPUs, two of them
//! held by a running job, a four-GPU job at the head of the queue that cannot
//! fit, and a one-GPU job behind it that can. Whether the small job goes is the
//! entire question, and each mode answers it differently.

use ferro_controller::registry::{now_s, Job, Registry};
use ferro_controller::service::queue_pass;
use ferro_proto::{Gpu, JobPhase, JobPlacement, JobPlan, JobStatus, NodeInfo, SubmitJobRequest};
use ferro_sched::{Dispatch, PlacementPolicy, SchedulerConfig};
use std::sync::Arc;

const VRAM_FLOOR: u64 = 8 << 30;

fn config() -> SchedulerConfig {
    SchedulerConfig {
        master_port: 29500,
        min_free_vram_b: VRAM_FLOOR,
        network_max_age_s: 86_400,
        placement_weights: Default::default(),
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

/// A node with nowhere to launch on.
///
/// The launch that follows promotion therefore fails, which is fine and
/// deliberate: what is under test is which jobs the pass *decided* to start.
/// The address is empty rather than a closed port because tonic retries a
/// refused connection with backoff for over two minutes before giving up, and
/// a test suite cannot pay that per job.
fn node(id: &str, gpus: u32) -> NodeInfo {
    NodeInfo {
        node_id: id.into(),
        address: String::new(),
        nccl_address: "127.0.0.1".into(),
        gpus: (0..gpus).map(gpu).collect(),
        ..Default::default()
    }
}

fn blank(id: &str) -> Job {
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
        placement: None,
    }
}

/// A job waiting for `gpus` cards, declaring `estimate` seconds if it declares
/// anything at all.
fn queued(id: &str, gpus: u32, estimate: Option<u32>) -> Job {
    Job {
        queued: true,
        estimated_duration_s: estimate,
        queue_req: Some(SubmitJobRequest {
            nodes: 1,
            gpus_per_node: gpus,
            ..Default::default()
        }),
        ..blank(id)
    }
}

/// A job running on `gpu-a`'s first two cards, due to release them in
/// `estimate` seconds.
fn incumbent(estimate: Option<u32>) -> Job {
    let plan = JobPlan {
        world_size: 2,
        placements: vec![JobPlacement {
            node_id: "gpu-a".into(),
            gpu_indices: vec![0, 1],
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut job = Job {
        estimated_duration_s: estimate,
        plan,
        ..blank("incumbent")
    };
    job.per_node.insert(
        "gpu-a".into(),
        JobStatus {
            job_id: "incumbent".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Running as i32,
            started_unix_s: job.submitted,
            ..Default::default()
        },
    );
    job
}

/// Four GPUs, two of them already held, `big` at the head asking for all four
/// and `small` behind it asking for one.
async fn contended(incumbent_estimate: Option<u32>, small_estimate: Option<u32>) -> Arc<Registry> {
    let registry = Arc::new(Registry::new(VRAM_FLOOR));
    registry.upsert_node(node("gpu-a", 4)).await;

    let held = incumbent(incumbent_estimate);
    registry
        .reserve_exact(&held.plan, "incumbent")
        .await
        .expect("the cluster is empty, so this cannot conflict");
    registry.insert_job(held).await;
    registry.insert_job(queued("big", 4, Some(600))).await;
    registry
        .insert_job(queued("small", 1, small_estimate))
        .await;
    registry
}

async fn pass(registry: &Arc<Registry>, mode: Dispatch) -> Vec<String> {
    let placement: Arc<dyn PlacementPolicy> = Arc::new(ferro_sched::PerformancePlacement);
    queue_pass(registry, &placement, &config(), mode).await
}

/// Why a job says it is waiting, which is what `ferro queue`'s WHY WAITING
/// column shows.
async fn why(registry: &Registry, job_id: &str) -> String {
    registry.inner.lock().await.jobs[job_id]
        .queue_message
        .clone()
}

#[tokio::test]
async fn opportunistic_dispatch_starts_the_small_job_behind_the_big_one() {
    // What FerroGrid does today, and the baseline the other two are read
    // against: `big` cannot fit, and `small` goes anyway.
    let registry = contended(Some(1_000), Some(60)).await;
    assert_eq!(pass(&registry, Dispatch::Opportunistic).await, ["small"]);
}

#[tokio::test]
async fn strict_dispatch_starts_nothing_behind_a_job_that_does_not_fit() {
    // Same registry, same cluster, same queue. Only the mode changed.
    let registry = contended(Some(1_000), Some(60)).await;
    assert!(pass(&registry, Dispatch::Strict).await.is_empty());
    assert!(
        why(&registry, "small").await.contains("strict dispatch"),
        "the job should say the dispatcher is why it is waiting"
    );
}

#[tokio::test]
async fn reserved_dispatch_backfills_a_job_that_finishes_in_time() {
    // `incumbent` frees two cards in 1000s, which is when `big` can start.
    // `small` says 60s, so it is provably out of the way first.
    let registry = contended(Some(1_000), Some(60)).await;
    assert_eq!(pass(&registry, Dispatch::Reserved).await, ["small"]);
}

#[tokio::test]
async fn reserved_dispatch_refuses_a_job_that_declared_nothing() {
    // The one difference from the test above: `small` never said how long it
    // would run, so it can prove nothing and does not go. This is the case
    // that makes reservation a regression on a cluster where nobody declares
    // anything, and the reason it is not the default.
    let registry = contended(Some(1_000), None).await;
    assert!(pass(&registry, Dispatch::Reserved).await.is_empty());
    assert!(
        why(&registry, "small")
            .await
            .contains("no declared duration"),
        "the job should say what it failed to prove"
    );
}

#[tokio::test]
async fn reserved_dispatch_refuses_everything_when_the_running_job_declared_nothing() {
    // Nobody knows when the incumbent releases its cards, so nobody knows when
    // `big` can start, so nothing may be let past it. Reservation degrades to
    // strict rather than guessing -- even though `small` declared a duration
    // that would have been short enough against any plausible guess.
    let registry = contended(None, Some(60)).await;
    assert!(pass(&registry, Dispatch::Reserved).await.is_empty());
    assert!(
        why(&registry, "small")
            .await
            .contains("earliest start is unknown"),
        "the job should say whose answer was missing"
    );
}

#[tokio::test]
async fn a_timeout_the_controller_enforces_is_a_declaration_too() {
    // No estimate anywhere, but wall-clock limits on both sides. The timeout
    // is the stronger claim of the two -- `reap_expired` actually kills the
    // job -- so reservation can work from it alone.
    let registry = Arc::new(Registry::new(VRAM_FLOOR));
    registry.upsert_node(node("gpu-a", 4)).await;
    let mut held = incumbent(None);
    held.timeout_s = 1_000;
    registry
        .reserve_exact(&held.plan, "incumbent")
        .await
        .unwrap();
    registry.insert_job(held).await;
    registry.insert_job(queued("big", 4, Some(600))).await;
    let mut small = queued("small", 1, None);
    small.timeout_s = 60;
    registry.insert_job(small).await;

    assert_eq!(pass(&registry, Dispatch::Reserved).await, ["small"]);
}

#[tokio::test]
async fn every_mode_starts_the_head_when_it_fits() {
    // The control: with nothing held, `big` fits and goes first under all
    // three modes. A mode that changed this would not be a dispatch mode --
    // none of them reorders the queue, they only decide who may overtake.
    //
    // Only the head is asserted. Under opportunistic dispatch `small` starts
    // too, because `big`'s launch fails against a node with no address and
    // hands its four cards back inside the same pass; that is the launch path
    // reacting to a dead agent, not the dispatcher choosing anything.
    for mode in [
        Dispatch::Opportunistic,
        Dispatch::Strict,
        Dispatch::Reserved,
    ] {
        let registry = Arc::new(Registry::new(VRAM_FLOOR));
        registry.upsert_node(node("gpu-a", 4)).await;
        registry.insert_job(queued("big", 4, Some(600))).await;
        registry.insert_job(queued("small", 1, Some(60))).await;

        assert_eq!(
            pass(&registry, mode).await.first().map(String::as_str),
            Some("big"),
            "{} refused the head of an empty cluster",
            mode.label()
        );
    }
}

#[tokio::test]
async fn patience_runs_out_whatever_the_mode_decided() {
    // A queued job past its deadline has to end. Under strict dispatch it is
    // never even offered to the placement policy, so checking deadlines only
    // for the admissible jobs would strand it in the queue forever.
    let registry = contended(Some(1_000), Some(60)).await;
    {
        let mut g = registry.inner.lock().await;
        g.jobs.get_mut("small").unwrap().queue_deadline = now_s() - 1;
    }
    assert!(pass(&registry, Dispatch::Strict).await.is_empty());

    let g = registry.inner.lock().await;
    let small = &g.jobs["small"];
    assert!(
        !small.queued,
        "the deadline should have taken it out of line"
    );
    assert_eq!(small.phase(), JobPhase::Failed);
}
