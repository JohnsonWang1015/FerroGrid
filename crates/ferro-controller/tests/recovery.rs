//! What a restarted controller decides about the jobs it just read off disk.
//!
//! Persistence answers "what did this controller know"; this answers "how much
//! of it is still true". The two are different questions and the gap between
//! them is the whole of recovery: a job written down as `Running` is a claim
//! about a moment that has passed, and until an agent says otherwise it is
//! only a claim.
//!
//! The properties worth pinning down are that an agent's word settles the
//! question either way, that nothing is decided before the agents have had
//! time to speak, and that once the window has closed nothing is left
//! undecided -- a job the controller is still unsure about at that point is a
//! job `ferro jobs` is lying about.

use ferro_controller::registry::Registry;
use ferro_controller::service::{self, ControllerService};
use ferro_controller::store::{Change, Event, EventKind, JobRecord, Store};
use ferro_proto::controller_server::Controller;
use ferro_proto::{
    GetJobRequest, Gpu, HeartbeatRequest, JobPhase, JobPlacement, JobPlan, JobStatus, NodeInfo,
    SubmitJobRequest,
};
use ferro_sched::SchedulerConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tonic::Request;

const VRAM_FLOOR: u64 = 8 << 30;

/// A database of its own per test, removed afterwards.
struct TempDb {
    dir: PathBuf,
}

impl TempDb {
    fn new(tag: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrogrid-recovery-{}-{tag}-{nonce}",
            std::process::id()
        ));
        Self { dir }
    }

    fn path(&self) -> PathBuf {
        self.dir.join("state/controller.db")
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn plan(nodes: &[(&str, u32)]) -> JobPlan {
    JobPlan {
        world_size: nodes.iter().map(|(_, gpus)| gpus).sum(),
        master_addr: "10.0.0.10".into(),
        master_port: 29500,
        placements: nodes
            .iter()
            .enumerate()
            .map(|(rank, (node, gpus))| JobPlacement {
                node_id: (*node).to_string(),
                node_rank: rank as u32,
                address: format!("http://{node}:7071"),
                gpu_indices: (0..*gpus).collect(),
                gpu_uuids: (0..*gpus).map(|i| format!("uuid-{node}-{i}")).collect(),
            })
            .collect(),
    }
}

/// A job as the last controller left it on disk.
fn record(job_id: &str, plan: JobPlan) -> JobRecord {
    JobRecord {
        job_id: job_id.into(),
        order_seq: 0,
        name: format!("{job_id}-name"),
        submitted_by: "alice".into(),
        project: "vision".into(),
        priority: 50,
        estimated_duration_s: None,
        timeout_s: 0,
        submitted: 1_700_000_000,
        queued: false,
        queue_deadline: 0,
        plan,
        queue_req: None,
        placement: None,
    }
}

/// A job that was still waiting for capacity when the controller died. It has
/// no plan, and the request that described it is what the dispatcher would
/// need to place it.
fn queued_record(job_id: &str) -> JobRecord {
    JobRecord {
        queued: true,
        plan: JobPlan::default(),
        queue_req: Some(SubmitJobRequest {
            nodes: 1,
            gpus_per_node: 2,
            script: "train.py".into(),
            ..Default::default()
        }),
        ..record(job_id, JobPlan::default())
    }
}

fn status(job_id: &str, node_id: &str, rank: u32, phase: JobPhase) -> JobStatus {
    JobStatus {
        job_id: job_id.into(),
        node_id: node_id.into(),
        node_rank: rank,
        phase: phase as i32,
        message: format!("{node_id} says {phase:?}"),
        started_unix_s: 1_700_000_100,
        ended_unix_s: if phase.is_terminal() {
            1_700_000_900
        } else {
            0
        },
        ..Default::default()
    }
}

/// `count` cards, each either free or held by `holder`.
fn gpus(node: &str, count: u32, holder: &str) -> Vec<Gpu> {
    (0..count)
        .map(|index| Gpu {
            index,
            uuid: format!("uuid-{node}-{index}"),
            name: "NVIDIA RTX 4090".into(),
            memory_total_b: 24 << 30,
            memory_used_b: if holder.is_empty() { 0 } else { 20 << 30 },
            allocated_job_id: holder.to_string(),
            ..Default::default()
        })
        .collect()
}

fn node_info(node: &str, count: u32) -> NodeInfo {
    NodeInfo {
        node_id: node.into(),
        hostname: node.into(),
        address: format!("{node}:7071"),
        nccl_address: format!("10.0.0.{}", 10 + count),
        gpus: gpus(node, count, ""),
        ..Default::default()
    }
}

/// Write a database the way a controller that then died would have left it,
/// and restore a registry from it. Closing the store is what commits: the
/// writer thread drains what is queued before it is joined.
fn seed(path: &Path, jobs: &[(JobRecord, Vec<JobStatus>)]) -> Registry {
    {
        let store = Store::open(path).expect("open");
        for (seq, (rec, statuses)) in jobs.iter().enumerate() {
            let mut rec = rec.clone();
            rec.order_seq = seq as i64;
            store.write(Change::Job(Box::new(rec)));
            for s in statuses {
                store.write(Change::Status {
                    job_id: s.job_id.clone(),
                    node_id: s.node_id.clone(),
                    status: Box::new(s.clone()),
                });
            }
        }
    }
    let state = Store::load(path).expect("load");
    let store = Store::open(path).expect("reopen");
    Registry::restore(VRAM_FLOOR, Arc::new(ferro_sched::queue::Fifo), store, state)
}

fn service(registry: Arc<Registry>) -> ControllerService {
    ControllerService {
        registry,
        plugins: Default::default(),
        heartbeat_interval_s: 3,
        sched: SchedulerConfig {
            master_port: 29500,
            min_free_vram_b: VRAM_FLOOR,
            network_max_age_s: 0,
            placement_weights: Default::default(),
        },
        placement: ferro_sched::placement_policy("performance").expect("policy"),
    }
}

/// A heartbeat as an agent sends one: the cards and what holds them, plus
/// whatever it is running.
async fn heartbeat(svc: &ControllerService, node: &str, gpus: Vec<Gpu>, jobs: Vec<JobStatus>) {
    svc.heartbeat(Request::new(HeartbeatRequest {
        node_id: node.into(),
        gpus,
        jobs,
        processes: Vec::new(),
    }))
    .await
    .expect("heartbeat");
}

async fn phase_of(registry: &Registry, job_id: &str) -> JobPhase {
    registry.inner.lock().await.jobs[job_id].phase()
}

async fn reconciling(registry: &Registry, job_id: &str) -> bool {
    registry.inner.lock().await.reconciling.contains(job_id)
}

async fn message_of(registry: &Registry, job_id: &str) -> String {
    let g = registry.inner.lock().await;
    let job = &g.jobs[job_id];
    let mut ranks: Vec<&JobStatus> = job.per_node.values().collect();
    ranks.sort_by_key(|s| s.node_rank);
    ranks.first().map(|s| s.message.clone()).unwrap_or_default()
}

async fn events_for(registry: &Registry, job_id: &str, kind: EventKind) -> Vec<Event> {
    registry.events(0, job_id, Some(kind)).await
}

/// 1. An agent that reports the job settles it: it stays running, and the
///    controller stops wondering.
#[tokio::test]
async fn a_restored_job_an_agent_reports_stays_running() {
    let db = TempDb::new("claimed");
    let registry = Arc::new(seed(
        &db.path(),
        &[(
            record("jrun", plan(&[("gpu-a", 2)])),
            vec![status("jrun", "gpu-a", 0, JobPhase::Running)],
        )],
    ));
    assert!(reconciling(&registry, "jrun").await, "restored unconfirmed");
    assert_eq!(phase_of(&registry, "jrun").await, JobPhase::Running);

    let svc = service(registry.clone());
    registry.upsert_node(node_info("gpu-a", 2)).await;
    heartbeat(
        &svc,
        "gpu-a",
        gpus("gpu-a", 2, "jrun"),
        vec![status("jrun", "gpu-a", 0, JobPhase::Running)],
    )
    .await;

    assert!(
        !reconciling(&registry, "jrun").await,
        "an agent reporting the job is the job accounted for"
    );

    let r = registry.close_recovery_window(30).await;
    assert_eq!(r.claimed, 1);
    assert_eq!(r.failed, 0, "a job somebody claimed is not lost");
    assert_eq!(
        phase_of(&registry, "jrun").await,
        JobPhase::Running,
        "and the window closing does not touch it"
    );
}

/// 2. Nobody claims it, so it is failed -- and the message says which of the
///    two reasons it was, because they are not the same news.
#[tokio::test]
async fn a_restored_job_nobody_reports_is_failed_with_the_reason() {
    let db = TempDb::new("unclaimed");
    let registry = Arc::new(seed(
        &db.path(),
        &[
            (
                record("jgone", plan(&[("gpu-a", 2)])),
                vec![status("jgone", "gpu-a", 0, JobPhase::Running)],
            ),
            (
                record("jdark", plan(&[("gpu-b", 2)])),
                vec![status("jdark", "gpu-b", 0, JobPhase::Running)],
            ),
        ],
    ));

    // gpu-a came back and said nothing about its job. gpu-b never came back.
    let svc = service(registry.clone());
    registry.upsert_node(node_info("gpu-a", 2)).await;
    heartbeat(&svc, "gpu-a", gpus("gpu-a", 2, ""), Vec::new()).await;

    let r = registry.close_recovery_window(30).await;
    assert_eq!(r.failed, 2);
    assert_eq!(r.claimed, 0);

    assert_eq!(phase_of(&registry, "jgone").await, JobPhase::Failed);
    let gone = message_of(&registry, "jgone").await;
    assert!(
        gone.contains("lost while the controller was down") && gone.contains("gpu-a"),
        "a node that is up and did not mention the job is evidence: {gone}"
    );

    assert_eq!(phase_of(&registry, "jdark").await, JobPhase::Failed);
    let dark = message_of(&registry, "jdark").await;
    assert!(
        dark.contains("fate unknown") && dark.contains("gpu-b"),
        "a node that never came back is not evidence, and says so: {dark}"
    );

    // Terminal means the cards are back, through the ordinary release path.
    let g = registry.inner.lock().await;
    assert!(
        g.nodes["gpu-a"]
            .info
            .gpus
            .iter()
            .all(|gpu| gpu.allocated_job_id.is_empty()),
        "a terminal job owns no GPUs"
    );
    drop(g);

    for job_id in ["jgone", "jdark"] {
        assert_eq!(
            events_for(&registry, job_id, EventKind::JobFailed)
                .await
                .len(),
            1,
            "{job_id} failed once, with its reason on the line"
        );
    }
}

/// 3. And it is not failed early. An agent that reconnects late is an agent
///    that reconnects, not a job that died.
#[tokio::test]
async fn a_slow_agent_does_not_lose_its_job() {
    let db = TempDb::new("slow");
    let registry = Arc::new(seed(
        &db.path(),
        &[(
            record("jslow", plan(&[("gpu-a", 2)])),
            vec![status("jslow", "gpu-a", 0, JobPhase::Running)],
        )],
    ));

    // The real background task, with the window it is actually given.
    let window = tokio::spawn(service::reconcile_recovery(registry.clone(), 2));

    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    assert!(
        reconciling(&registry, "jslow").await,
        "the window is still open"
    );
    assert_eq!(
        phase_of(&registry, "jslow").await,
        JobPhase::Running,
        "nothing is decided while it is"
    );

    // The agent turns up late, but inside the window.
    let svc = service(registry.clone());
    registry.upsert_node(node_info("gpu-a", 2)).await;
    heartbeat(
        &svc,
        "gpu-a",
        gpus("gpu-a", 2, "jslow"),
        vec![status("jslow", "gpu-a", 0, JobPhase::Running)],
    )
    .await;

    window.await.expect("the window closes on its own");
    assert_eq!(
        phase_of(&registry, "jslow").await,
        JobPhase::Running,
        "a job whose agent arrived late is not a lost job"
    );
    assert_eq!(registry.recovery().await.failed, 0);
}

/// 4. Ownership alone is a claim. An agent that holds the cards for a job is
///    running it, whether or not this particular heartbeat said so.
#[tokio::test]
async fn a_gpu_carrying_the_job_id_counts_as_a_claim() {
    let db = TempDb::new("ownership");
    let registry = Arc::new(seed(
        &db.path(),
        &[(
            record("jheld", plan(&[("gpu-a", 2)])),
            vec![status("jheld", "gpu-a", 0, JobPhase::Running)],
        )],
    ));

    let svc = service(registry.clone());
    registry.upsert_node(node_info("gpu-a", 2)).await;
    // No JobStatus at all: only the allocation table.
    heartbeat(&svc, "gpu-a", gpus("gpu-a", 2, "jheld"), Vec::new()).await;

    assert!(!reconciling(&registry, "jheld").await);
    let r = registry.close_recovery_window(30).await;
    assert_eq!(r.claimed, 1);
    assert_eq!(r.failed, 0);
    assert_eq!(phase_of(&registry, "jheld").await, JobPhase::Running);
}

/// 5. The mirror image: the database says queued, the agent says running. The
///    promote reached the cluster and not the disk, so the plan is rebuilt
///    from what the cards say rather than the job being placed a second time.
#[tokio::test]
async fn a_queued_job_found_running_is_adopted() {
    let db = TempDb::new("adopt");
    let registry = Arc::new(seed(&db.path(), &[(queued_record("jlost"), Vec::new())]));
    assert!(
        registry.inner.lock().await.jobs["jlost"].queued,
        "it came back queued, which is what was written down"
    );

    let svc = service(registry.clone());
    registry.upsert_node(node_info("gpu-a", 2)).await;
    heartbeat(
        &svc,
        "gpu-a",
        gpus("gpu-a", 2, "jlost"),
        vec![status("jlost", "gpu-a", 0, JobPhase::Running)],
    )
    .await;

    let g = registry.inner.lock().await;
    let job = &g.jobs["jlost"];
    assert!(!job.queued, "it is running, so it is not waiting");
    assert!(job.queue_req.is_none(), "and must not be placed again");
    assert_eq!(job.plan.placements.len(), 1);
    assert_eq!(job.plan.placements[0].node_id, "gpu-a");
    assert_eq!(job.plan.placements[0].gpu_indices, vec![0, 1]);
    assert_eq!(job.plan.placements[0].address, "gpu-a:7071");
    assert_eq!(job.plan.world_size, 2);
    assert_eq!(
        job.phase(),
        JobPhase::Running,
        "and the agent's report drives it from here"
    );
    drop(g);

    assert!(!reconciling(&registry, "jlost").await);
    assert!(
        registry.queued_jobs().await.is_empty(),
        "the dispatcher has nothing left to place"
    );

    let r = registry.close_recovery_window(30).await;
    assert_eq!(r.adopted, 1);
    assert_eq!(r.failed, 0);
}

/// 6. Reconciliation happens once. Running the sweep again must not fail an
///    already-failed job a second time or write the summary twice.
#[tokio::test]
async fn closing_the_window_twice_changes_nothing() {
    let db = TempDb::new("idempotent");
    let registry = Arc::new(seed(
        &db.path(),
        &[(
            record("jonce", plan(&[("gpu-a", 1)])),
            vec![status("jonce", "gpu-a", 0, JobPhase::Running)],
        )],
    ));

    let first = registry.close_recovery_window(30).await;
    let after_first = registry.events(0, "", None).await;
    let second = registry.close_recovery_window(30).await;
    let after_second = registry.events(0, "", None).await;

    assert_eq!(first, second, "the counts are settled, not recomputed");
    assert_eq!(first.failed, 1);
    assert_eq!(
        after_first, after_second,
        "and the second sweep records nothing"
    );
    assert_eq!(
        events_for(&registry, "jonce", EventKind::JobFailed)
            .await
            .len(),
        1,
        "one failure, however often it is swept"
    );
    assert_eq!(
        registry
            .events(0, "", Some(EventKind::Reconciled))
            .await
            .len(),
        1,
        "and one summary of the recovery"
    );
}

/// 7. The invariant the whole window exists to establish: afterwards, nothing
///    non-terminal is still waiting on an answer.
#[tokio::test]
async fn nothing_is_left_undecided_once_the_window_closes() {
    let db = TempDb::new("invariant");
    let registry = Arc::new(seed(
        &db.path(),
        &[
            (
                record("jkept", plan(&[("gpu-a", 1)])),
                vec![status("jkept", "gpu-a", 0, JobPhase::Running)],
            ),
            (
                record("jlost", plan(&[("gpu-b", 1)])),
                vec![status("jlost", "gpu-b", 0, JobPhase::Running)],
            ),
            (
                record("jdone", plan(&[("gpu-a", 1)])),
                vec![status("jdone", "gpu-a", 0, JobPhase::Succeeded)],
            ),
            (queued_record("jwait"), Vec::new()),
        ],
    ));

    let svc = service(registry.clone());
    registry.upsert_node(node_info("gpu-a", 2)).await;
    heartbeat(
        &svc,
        "gpu-a",
        gpus("gpu-a", 1, "jkept"),
        vec![status("jkept", "gpu-a", 0, JobPhase::Running)],
    )
    .await;

    // The flag is visible to an operator while it is up.
    let summary = svc
        .get_job(Request::new(GetJobRequest {
            job_id: "jlost".into(),
        }))
        .await
        .expect("get_job")
        .into_inner();
    assert!(summary.reconciling, "still making up its mind about it");

    let r = registry.close_recovery_window(30).await;

    let g = registry.inner.lock().await;
    assert!(g.reconciling.is_empty(), "nothing is still being decided");
    for job in g.jobs.values() {
        assert!(
            job.phase().is_terminal() || !g.reconciling.contains(&job.job_id),
            "{} is neither finished nor decided",
            job.job_id
        );
    }
    // A job still waiting for capacity was never missing: it holds nothing and
    // the dispatcher owns it. Failing it would throw away the queue position
    // that persistence exists to keep.
    assert!(g.jobs["jwait"].queued, "the queue survives recovery");
    assert_eq!(g.jobs["jdone"].phase(), JobPhase::Succeeded);
    assert_eq!(g.jobs["jkept"].phase(), JobPhase::Running);
    assert_eq!(g.jobs["jlost"].phase(), JobPhase::Failed);
    drop(g);

    assert_eq!(r.restored, 4, "every row read off disk is counted");
    assert_eq!(r.claimed, 1);
    assert_eq!(r.failed, 1);
    assert!(r.settled);

    let summary = svc
        .get_job(Request::new(GetJobRequest {
            job_id: "jlost".into(),
        }))
        .await
        .expect("get_job")
        .into_inner();
    assert!(!summary.reconciling, "and it has stopped saying so");

    let done = registry.events(0, "", Some(EventKind::Reconciled)).await;
    assert_eq!(done.len(), 1);
    assert!(
        done[0].detail.contains("restored 4")
            && done[0].detail.contains("claimed 1")
            && done[0].detail.contains("failed 1"),
        "the recovery is measurable from the log alone: {}",
        done[0].detail
    );
}

/// 8. A job that had already finished is finished business. Nobody will
///    mention it again, it owns nothing, and there is nothing left to decide.
#[tokio::test]
async fn a_terminal_restored_job_is_left_alone() {
    let db = TempDb::new("terminal");
    let registry = Arc::new(seed(
        &db.path(),
        &[(
            record("jold", plan(&[("gpu-a", 2)])),
            vec![status("jold", "gpu-a", 0, JobPhase::Succeeded)],
        )],
    ));

    assert!(
        !reconciling(&registry, "jold").await,
        "it was never in question"
    );

    let before = registry.events(0, "jold", None).await;
    let r = registry.close_recovery_window(30).await;
    let after = registry.events(0, "jold", None).await;

    assert_eq!(phase_of(&registry, "jold").await, JobPhase::Succeeded);
    assert_eq!(r.failed, 0);
    assert_eq!(r.claimed, 0);
    assert_eq!(r.restored, 1, "it was still restored, and still counted");
    assert_eq!(before, after, "and nothing was recorded about it");
    assert_eq!(
        registry.inner.lock().await.jobs["jold"].per_node["gpu-a"].phase(),
        JobPhase::Succeeded,
        "its rank's own report is untouched"
    );
}

/// A follow-on to 5, for the shape it does not cover: agents come back one at
/// a time, so a multi-node job can be adopted before its peer has said
/// anything. A rank missing from the rebuilt plan is a rank nobody could
/// afterwards cancel or time out, so the plan is allowed to fill in -- until
/// the window closes and the question is over.
#[tokio::test]
async fn a_peer_reconnecting_late_completes_an_adopted_plan() {
    let db = TempDb::new("adopt-peer");
    let registry = Arc::new(seed(&db.path(), &[(queued_record("jpair"), Vec::new())]));
    let svc = service(registry.clone());

    registry.upsert_node(node_info("gpu-a", 1)).await;
    heartbeat(&svc, "gpu-a", gpus("gpu-a", 1, "jpair"), Vec::new()).await;
    assert_eq!(
        registry.inner.lock().await.jobs["jpair"]
            .plan
            .placements
            .len(),
        1,
        "only one node has reported, so only one is known"
    );

    registry.upsert_node(node_info("gpu-b", 1)).await;
    heartbeat(&svc, "gpu-b", gpus("gpu-b", 1, "jpair"), Vec::new()).await;

    let g = registry.inner.lock().await;
    let plan = &g.jobs["jpair"].plan;
    assert_eq!(plan.world_size, 2);
    let nodes: Vec<&str> = plan.placements.iter().map(|p| p.node_id.as_str()).collect();
    assert_eq!(
        nodes,
        vec!["gpu-a", "gpu-b"],
        "both ranks are reachable now"
    );
    drop(g);

    // One adoption, told in the two steps it actually happened in.
    assert_eq!(
        events_for(&registry, "jpair", EventKind::JobScheduled)
            .await
            .len(),
        2
    );
    assert_eq!(registry.recovery().await.adopted, 1, "still one job");

    // And the window closing ends it: a node reporting afterwards is an
    // ordinary heartbeat, not more evidence about the restart.
    registry.close_recovery_window(30).await;
    registry.upsert_node(node_info("gpu-c", 1)).await;
    heartbeat(&svc, "gpu-c", gpus("gpu-c", 1, "jpair"), Vec::new()).await;
    assert_eq!(
        registry.inner.lock().await.jobs["jpair"]
            .plan
            .placements
            .len(),
        2,
        "the plan is settled"
    );
}
