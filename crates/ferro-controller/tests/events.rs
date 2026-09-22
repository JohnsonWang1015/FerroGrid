//! What the controller remembers *happening*, as opposed to what is true now.
//!
//! The job table answers "what is this job"; the event log answers "what did
//! it go through". Nothing reconstructs the second from the first -- a card
//! that was taken and handed back leaves no trace in any current state, and a
//! job that failed at 04:00 looks exactly like one that failed at noon -- so
//! the properties worth pinning down are that an event is written when the
//! fact occurs, exactly once per fact, and that it outlives the process.

use ferro_controller::registry::{Job, Registry};
use ferro_controller::service::ControllerService;
use ferro_controller::store::{Event, EventKind, JobRecord, Store};
use ferro_proto::controller_server::Controller;
use ferro_proto::{
    JobPhase, JobPlacement, JobPlan, JobStatus, ListEventsRequest, NodeInfo, SubmitJobRequest,
};
use ferro_sched::SchedulerConfig;
use std::collections::HashMap;
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
            "ferrogrid-events-{}-{tag}-{nonce}",
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

fn plan(node: &str, gpus: u32) -> JobPlan {
    JobPlan {
        world_size: gpus,
        master_addr: "10.0.0.10".into(),
        master_port: 29500,
        placements: vec![JobPlacement {
            node_id: node.into(),
            node_rank: 0,
            address: format!("http://{node}:7071"),
            gpu_indices: (0..gpus).collect(),
            gpu_uuids: (0..gpus).map(|i| format!("uuid-{node}-{i}")).collect(),
        }],
    }
}

fn record(job_id: &str, queued: bool) -> JobRecord {
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
        queued,
        queue_deadline: 0,
        plan: if queued {
            JobPlan::default()
        } else {
            plan("gpu-a", 2)
        },
        queue_req: queued.then(SubmitJobRequest::default),
        placement: None,
    }
}

fn job(job_id: &str, queued: bool) -> Job {
    Job::from_record(record(job_id, queued), HashMap::new())
}

/// A job with a rank on each of two nodes, which is how a real one fails: one
/// rank dies while the other is still reporting.
fn two_node_job(job_id: &str) -> Job {
    let mut r = record(job_id, false);
    r.plan = JobPlan {
        world_size: 2,
        master_addr: "10.0.0.10".into(),
        master_port: 29500,
        placements: ["gpu-a", "gpu-b"]
            .iter()
            .enumerate()
            .map(|(rank, node)| JobPlacement {
                node_id: (*node).into(),
                node_rank: rank as u32,
                address: format!("http://{node}:7071"),
                gpu_indices: vec![0],
                gpu_uuids: vec![format!("uuid-{node}-0")],
            })
            .collect(),
    };
    Job::from_record(r, HashMap::new())
}

fn reports(node: &str, rank: u32, phase: JobPhase, exit_code: i32) -> JobStatus {
    JobStatus {
        job_id: "jpair".into(),
        node_id: node.into(),
        node_rank: rank,
        phase: phase as i32,
        exit_code,
        message: format!("{node} says {phase:?}"),
        started_unix_s: 1_700_000_100,
        ended_unix_s: if phase.is_terminal() {
            1_700_000_900
        } else {
            0
        },
        ..Default::default()
    }
}

/// The service with nothing behind it but the registry. `ferro events` reaches
/// the ring through here, and this layer is the only one that turns the kind
/// back from the string it travels as.
fn service(registry: Arc<Registry>) -> ControllerService {
    ControllerService {
        registry,
        plugins: Default::default(),
        heartbeat_interval_s: 5,
        sched: SchedulerConfig {
            master_port: 29500,
            min_free_vram_b: VRAM_FLOOR,
            network_max_age_s: 0,
            placement_weights: Default::default(),
        },
        placement: ferro_sched::placement_policy("performance").expect("policy"),
    }
}

async fn list(svc: &ControllerService, req: ListEventsRequest) -> Vec<ferro_proto::Event> {
    svc.list_events(Request::new(req))
        .await
        .expect("list_events")
        .into_inner()
        .events
}

fn registry_at(path: &Path) -> Registry {
    let state = Store::load(path).expect("load");
    let store = Store::open(path).expect("open");
    Registry::restore(VRAM_FLOOR, Arc::new(ferro_sched::queue::Fifo), store, state)
}

/// Everything recorded, in order.
async fn timeline(registry: &Registry) -> Vec<Event> {
    registry.events(0, "", None).await
}

fn kinds(events: &[Event]) -> Vec<EventKind> {
    events.iter().map(|e| e.kind).collect()
}

#[tokio::test]
async fn submitting_a_job_is_recorded_and_queueing_it_says_where_in_line() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(job("jrun", false)).await;
    registry.insert_job(job("jwait", true)).await;

    let events = timeline(&registry).await;
    assert_eq!(
        kinds(&events),
        vec![
            EventKind::JobSubmitted,
            EventKind::JobSubmitted,
            EventKind::JobQueued,
        ],
        "a job that runs straight away was never queued"
    );
    assert_eq!(events[0].job_id, "jrun");
    assert_eq!(events[0].actor, "alice");
    assert_eq!(events[2].job_id, "jwait");
    // The position is the promise the queue made, so it is what was recorded.
    assert_eq!(events[2].detail, "position 1");
}

#[tokio::test]
async fn a_rank_repeating_itself_is_one_event() {
    // Ranks report their phase on every heartbeat. A job starts once.
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(job("jrun", false)).await;

    let running = JobStatus {
        job_id: "jrun".into(),
        node_id: "gpu-a".into(),
        node_rank: 0,
        phase: JobPhase::Running as i32,
        started_unix_s: 1_700_000_100,
        ..Default::default()
    };
    for _ in 0..3 {
        registry.update_job_status(running.clone()).await;
    }

    let started: Vec<Event> = registry
        .events(0, "", Some(EventKind::JobStarted))
        .await
        .to_vec();
    assert_eq!(started.len(), 1, "one start, however often it is reported");

    // And an ending is its own transition, with the exit code attached.
    registry
        .update_job_status(JobStatus {
            phase: JobPhase::Failed as i32,
            exit_code: 137,
            ended_unix_s: 1_700_000_900,
            ..running.clone()
        })
        .await;
    registry
        .update_job_status(JobStatus {
            phase: JobPhase::Failed as i32,
            exit_code: 137,
            ended_unix_s: 1_700_000_900,
            ..running
        })
        .await;

    let failed = registry.events(0, "", Some(EventKind::JobFailed)).await;
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].detail, "exit 137");
}

#[tokio::test]
async fn a_job_across_two_nodes_starts_once() {
    // The transition belongs to the job, not to a rank: the first rank going
    // Running is not the job running, and the second one saying it again is
    // not it running twice.
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(two_node_job("jpair")).await;

    registry
        .update_job_status(reports("gpu-a", 0, JobPhase::Running, 0))
        .await;
    assert!(
        registry
            .events(0, "", Some(EventKind::JobStarted))
            .await
            .is_empty(),
        "half a job is not a started job"
    );

    registry
        .update_job_status(reports("gpu-b", 1, JobPhase::Running, 0))
        .await;
    for _ in 0..3 {
        registry
            .update_job_status(reports("gpu-a", 0, JobPhase::Running, 0))
            .await;
    }
    let started = registry.events(0, "", Some(EventKind::JobStarted)).await;
    assert_eq!(started.len(), 1);
    assert_eq!(started[0].detail, "", "a start needs no explanation");
}

#[tokio::test]
async fn an_ending_quotes_the_rank_that_ended() {
    // A rank can die before its peer has reported anything, and then it is the
    // peer's *start* that turns the job failed. What belongs next to "failed"
    // is the dead rank's exit code, not the peer's progress message.
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(two_node_job("jpair")).await;

    registry
        .update_job_status(reports("gpu-a", 0, JobPhase::Failed, 137))
        .await;
    assert!(
        registry
            .events(0, "", Some(EventKind::JobFailed))
            .await
            .is_empty(),
        "nothing is known about the job until every rank has reported"
    );

    registry
        .update_job_status(reports("gpu-b", 1, JobPhase::Running, 0))
        .await;
    let failed = registry.events(0, "", Some(EventKind::JobFailed)).await;
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].detail, "exit 137");
}

#[tokio::test]
async fn allocation_and_release_name_the_cards() {
    let registry = Registry::new(VRAM_FLOOR);
    registry
        .upsert_node(NodeInfo {
            node_id: "gpu-a".into(),
            gpus: (0..2)
                .map(|index| ferro_proto::Gpu {
                    index,
                    uuid: format!("uuid-gpu-a-{index}"),
                    memory_total_b: 24 << 30,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        })
        .await;
    registry.insert_job(job("jrun", false)).await;

    let plan = plan("gpu-a", 2);
    registry
        .reserve_exact(&plan, "jrun")
        .await
        .expect("reserve");
    registry
        .update_job_status(JobStatus {
            job_id: "jrun".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Succeeded as i32,
            ended_unix_s: 1_700_000_900,
            ..Default::default()
        })
        .await;
    registry.release_if_done("jrun").await;
    // Nothing is holding anything now, so a second pass has nothing to say.
    registry.release_if_done("jrun").await;

    let events = timeline(&registry).await;
    assert_eq!(
        kinds(&events),
        vec![
            EventKind::NodeRegistered,
            EventKind::JobSubmitted,
            EventKind::GpuAllocated,
            EventKind::JobCompleted,
            EventKind::GpuReleased,
        ]
    );
    let allocated = &events[2];
    assert_eq!(allocated.detail, "gpu-a[0,1]");
    assert_eq!(allocated.actor, "alice");
    assert_eq!(events[4].detail, "gpu-a[0,1]");
    assert_eq!(events[0].node_id, "gpu-a");
    assert_eq!(events[0].detail, "2 GPU(s)");
}

#[tokio::test]
async fn cancelling_a_queued_job_records_the_cancellation() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(job("jwait", true)).await;
    assert!(
        registry
            .dequeue("jwait", JobPhase::Cancelled, "cancelled while queued")
            .await
    );

    let cancelled = registry.events(0, "", Some(EventKind::JobCancelled)).await;
    assert_eq!(cancelled.len(), 1);
    assert_eq!(cancelled[0].job_id, "jwait");
    assert_eq!(cancelled[0].detail, "cancelled while queued");
}

#[tokio::test]
async fn a_node_is_lost_once_and_recovered_once() {
    let registry = Registry::new(VRAM_FLOOR);
    registry
        .upsert_node(NodeInfo {
            node_id: "gpu-a".into(),
            ..Default::default()
        })
        .await;

    // Still fresh: sweeping changes nothing.
    registry.sweep_node_health().await;
    assert!(registry
        .events(0, "", Some(EventKind::NodeLost))
        .await
        .is_empty());

    // Age the heartbeat past the timeout, then sweep repeatedly: a node that
    // stays down is one event, not one per sweep.
    {
        let mut g = registry.inner.lock().await;
        let node = g.nodes.get_mut("gpu-a").expect("node");
        node.last_seen -= ferro_controller::registry::HEARTBEAT_TIMEOUT_S + 30;
    }
    for _ in 0..3 {
        registry.sweep_node_health().await;
    }
    let lost = registry.events(0, "", Some(EventKind::NodeLost)).await;
    assert_eq!(lost.len(), 1);
    assert_eq!(lost[0].node_id, "gpu-a");

    // Coming back is the other edge.
    {
        let mut g = registry.inner.lock().await;
        g.nodes.get_mut("gpu-a").expect("node").last_seen = ferro_controller::registry::now_s();
    }
    for _ in 0..3 {
        registry.sweep_node_health().await;
    }
    assert_eq!(
        registry
            .events(0, "", Some(EventKind::NodeRecovered))
            .await
            .len(),
        1
    );
}

#[tokio::test]
async fn filters_answer_about_one_job_and_one_kind() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(job("jone", false)).await;
    registry.insert_job(job("jtwo", true)).await;

    let one = registry.events(0, "jone", None).await;
    assert_eq!(kinds(&one), vec![EventKind::JobSubmitted]);
    assert!(one.iter().all(|e| e.job_id == "jone"));

    let queued = registry.events(0, "", Some(EventKind::JobQueued)).await;
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].job_id, "jtwo");

    // Both at once, and a combination nothing matches.
    assert_eq!(
        registry
            .events(0, "jtwo", Some(EventKind::JobQueued))
            .await
            .len(),
        1
    );
    assert!(registry
        .events(0, "jone", Some(EventKind::JobQueued))
        .await
        .is_empty());

    // Filtered before truncated: the last submission, not the submissions
    // among the last event.
    let last = registry.events(1, "", Some(EventKind::JobSubmitted)).await;
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].job_id, "jtwo");
}

#[tokio::test]
async fn the_rpc_filters_the_same_way_the_registry_does() {
    // The filter a user actually types goes through here: `--kind` is a string
    // on the wire, and this is the only layer that turns it back into a kind.
    let registry = Arc::new(Registry::new(VRAM_FLOOR));
    registry.insert_job(job("jone", false)).await;
    registry.insert_job(job("jtwo", true)).await;
    let svc = service(registry.clone());

    let all = list(&svc, ListEventsRequest::default()).await;
    assert_eq!(all.len(), 3);
    // Kinds go out as strings so a client built against an older proto still
    // prints one it has never heard of.
    assert_eq!(all[2].kind, "JOB_QUEUED");
    assert_eq!(all[2].job_id, "jtwo");
    assert_eq!(all[2].actor, "alice");
    assert_eq!(all[2].detail, "position 1");
    assert!(all[0].id < all[2].id, "oldest first, so it reads forwards");

    let by_job = list(
        &svc,
        ListEventsRequest {
            job_id: "jtwo".into(),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(by_job.len(), 2);
    assert!(by_job.iter().all(|e| e.job_id == "jtwo"));

    let by_kind = list(
        &svc,
        ListEventsRequest {
            kind: "JOB_QUEUED".into(),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(by_kind.len(), 1);
    assert_eq!(by_kind[0].job_id, "jtwo");

    let both = list(
        &svc,
        ListEventsRequest {
            limit: 1,
            job_id: "jone".into(),
            kind: "JOB_SUBMITTED".into(),
        },
    )
    .await;
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].job_id, "jone");

    // A kind nobody emits is a typo. Answering one with an empty timeline
    // reads as "nothing has happened", which is a much more alarming answer.
    let refused = svc
        .list_events(Request::new(ListEventsRequest {
            kind: "JOB_EXPLODED".into(),
            ..Default::default()
        }))
        .await
        .expect_err("an unknown kind is rejected");
    assert_eq!(refused.code(), tonic::Code::InvalidArgument);
    assert!(
        refused.message().contains("JOB_FAILED"),
        "it lists the kinds"
    );
}

#[tokio::test]
async fn a_start_says_whether_anything_was_recovered() {
    // CONTROLLER_RECOVERED is a claim about jobs that came back, not about
    // having had a database to look in: a controller that read an empty one
    // started fresh, whatever its --state said.
    let registry = Registry::new(VRAM_FLOOR);
    registry.record_event(Event::controller_start(0)).await;
    registry.record_event(Event::controller_start(3)).await;

    let events = timeline(&registry).await;
    assert_eq!(
        kinds(&events),
        vec![EventKind::ControllerStarted, EventKind::ControllerRecovered,]
    );
    assert_eq!(events[0].detail, "");
    assert_eq!(events[1].detail, "restored 3 job(s)");
}

#[tokio::test]
async fn the_timeline_survives_a_restart() {
    let db = TempDb::new("restart");
    let before = {
        let registry = registry_at(&db.path());
        registry.insert_job(job("jwait", true)).await;
        registry
            .dequeue("jwait", JobPhase::Cancelled, "cancelled while queued")
            .await;
        registry.flush().await.expect("flush");
        timeline(&registry).await
    };
    assert_eq!(
        kinds(&before),
        vec![
            EventKind::JobSubmitted,
            EventKind::JobQueued,
            EventKind::JobCancelled,
        ]
    );

    let registry = registry_at(&db.path());
    let after = timeline(&registry).await;
    assert_eq!(after, before, "same events, same ids, same order");

    // And the sequence carries on rather than starting again, which would
    // leave two events sharing a number and no way to order them.
    registry
        .record_event(Event::new(EventKind::ControllerRecovered).detail("restored 1 job(s)"))
        .await;
    let events = timeline(&registry).await;
    assert_eq!(events.last().map(|e| e.id), before.last().map(|e| e.id + 1));
    let mut ids: Vec<u64> = events.iter().map(|e| e.id).collect();
    ids.dedup();
    assert_eq!(ids.len(), events.len(), "ids are unique");
}

#[tokio::test]
async fn without_a_database_the_ring_still_answers() {
    // `--no-state`: nothing is written down, but `ferro events` is a view onto
    // the ring, so it works exactly as it does with a database behind it.
    let registry = Registry::new(VRAM_FLOOR);
    registry.insert_job(job("jrun", false)).await;
    registry
        .record_event(Event::new(EventKind::ControllerStarted))
        .await;

    let events = timeline(&registry).await;
    assert_eq!(
        kinds(&events),
        vec![EventKind::JobSubmitted, EventKind::ControllerStarted]
    );
    assert_eq!(events[0].id, 1, "the sequence starts at one");
}

#[test]
fn a_version_1_database_migrates_and_keeps_its_jobs() {
    let db = TempDb::new("migrate");
    let path = db.path();

    // A file as the previous release left it: the v1 tables, v1 in the header,
    // and no events table at all. Closing the store is what commits the job --
    // the writer thread drains what is queued before it is joined.
    {
        let store = Store::open(&path).expect("open");
        store.write(ferro_controller::store::Change::Job(Box::new(record(
            "jold", false,
        ))));
    }
    {
        let conn = rusqlite::Connection::open(&path).expect("raw open");
        conn.execute("DROP TABLE events", []).expect("drop events");
        conn.pragma_update(None, "user_version", 1).expect("v1");
    }

    let state = Store::load(&path).expect("a v1 database must still open");
    assert_eq!(state.jobs.len(), 1, "the job rows are not touched");
    assert_eq!(state.jobs[0].record.job_id, "jold");
    assert!(state.events.is_empty(), "there were no events to read");

    let conn = rusqlite::Connection::open(&path).expect("raw open");
    let version: i32 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("version");
    assert_eq!(version, 2, "opening it is what upgrades it");
    drop(conn);

    // And it takes events from here on.
    let store = Store::open(&path).expect("reopen");
    store.write(ferro_controller::store::Change::Event(
        Event::new(EventKind::ControllerRecovered).detail("restored 1 job(s)"),
    ));
    drop(store);
    let state = Store::load(&path).expect("load");
    assert_eq!(state.jobs.len(), 1);
    assert_eq!(state.events.len(), 1);
    assert_eq!(state.events[0].kind, EventKind::ControllerRecovered);
}
