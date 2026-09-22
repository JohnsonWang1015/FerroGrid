//! What the controller must still know after it has been restarted.
//!
//! The property under test is narrow and total: **a job the controller
//! acknowledged is a job it can still describe after a crash**, with the same
//! fields, the same per-rank statuses, the same computed phase and the same
//! place in line. A store that remembers most of a job is not a smaller version
//! of this; it is a controller that lies about a different set of jobs.
//!
//! Reconciliation against what the cluster currently reports is explicitly not
//! tested here, because it is explicitly not done: a job that was running when
//! the process died comes back running.

use ferro_controller::registry::{Job, Registry};
use ferro_controller::store::{Change, JobRecord, LoadedJob, Store};
use ferro_proto::{
    GpuBenchmark, JobPhase, JobPlacement, JobPlan, JobStatus, NetPair, PlacementExplanation,
    SubmitJobRequest,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const VRAM_FLOOR: u64 = 8 << 30;

/// A database of its own per test, removed afterwards. The parent directory is
/// one level down so the store's own `create_dir_all` is exercised too.
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
            "ferrogrid-persistence-{}-{tag}-{nonce}",
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
        world_size: nodes.iter().map(|(_, ranks)| ranks).sum(),
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

/// A job with every scalar field set to something distinguishable, so a column
/// swapped for its neighbour shows up as a wrong value rather than as nothing.
fn record(job_id: &str, order_seq: i64) -> JobRecord {
    JobRecord {
        job_id: job_id.into(),
        order_seq,
        name: format!("{job_id}-name"),
        submitted_by: "alice".into(),
        project: "vision".into(),
        priority: 73,
        estimated_duration_s: Some(1234),
        timeout_s: 4321,
        submitted: 1_700_000_000,
        queued: false,
        queue_deadline: 0,
        plan: plan(&[("gpu-a", 2), ("gpu-b", 2)]),
        queue_req: None,
        placement: Some(PlacementExplanation {
            policy: "performance".into(),
            total: 0.87,
            reasons: vec!["both ranks on measured 940 Mb/s".into()],
            components: vec![ferro_proto::ScoreComponent {
                name: "network".into(),
                value: 0.94,
            }],
        }),
    }
}

/// Write some changes, close the store so the writer thread drains, and read
/// the file back.
fn round_trip(path: &Path, changes: Vec<Change>) -> ferro_controller::store::LoadedState {
    {
        let store = Store::open(path).expect("open");
        for change in changes {
            store.write(change);
        }
    }
    Store::load(path).expect("load")
}

fn only_job(state: &ferro_controller::store::LoadedState) -> &LoadedJob {
    assert_eq!(state.jobs.len(), 1, "expected exactly one job");
    &state.jobs[0]
}

/// A registry backed by the database at `path`, picking up whatever is already
/// in it. This is what the controller does on startup.
fn registry_at(path: &Path) -> Registry {
    let state = Store::load(path).expect("load");
    let store = Store::open(path).expect("open");
    Registry::restore(VRAM_FLOOR, Arc::new(ferro_sched::queue::Fifo), store, state)
}

#[test]
fn a_job_survives_with_every_field_intact() {
    let db = TempDb::new("fields");
    let mut original = record("jaaa", 0);
    original.queued = true;
    original.queue_deadline = 1_700_009_999;
    original.plan = JobPlan::default();
    original.queue_req = Some(SubmitJobRequest {
        script: "train.py".into(),
        nodes: 2,
        gpus_per_node: 4,
        submitted_by: "alice".into(),
        queue: true,
        ..Default::default()
    });

    let state = round_trip(&db.path(), vec![Change::Job(Box::new(original.clone()))]);
    let loaded = &only_job(&state).record;

    assert_eq!(loaded.job_id, original.job_id);
    assert_eq!(loaded.order_seq, original.order_seq);
    assert_eq!(loaded.name, original.name);
    assert_eq!(loaded.submitted_by, original.submitted_by);
    assert_eq!(loaded.project, original.project);
    assert_eq!(loaded.priority, original.priority);
    assert_eq!(loaded.estimated_duration_s, original.estimated_duration_s);
    assert_eq!(loaded.timeout_s, original.timeout_s);
    assert_eq!(loaded.submitted, original.submitted);
    assert_eq!(loaded.queued, original.queued);
    assert_eq!(loaded.queue_deadline, original.queue_deadline);
    assert_eq!(loaded.plan, original.plan);
    assert_eq!(loaded.queue_req, original.queue_req);
    assert_eq!(loaded.placement, original.placement);
}

#[test]
fn an_undeclared_duration_stays_undeclared() {
    // SJF ranks a job that said nothing differently from one that said zero,
    // so a NULL that reloads as Some(0) would silently re-order the queue.
    let db = TempDb::new("null-duration");
    let mut said_nothing = record("jnone", 0);
    said_nothing.estimated_duration_s = None;
    let mut said_zero = record("jzero", 1);
    said_zero.estimated_duration_s = Some(0);

    let state = round_trip(
        &db.path(),
        vec![
            Change::Job(Box::new(said_nothing)),
            Change::Job(Box::new(said_zero)),
        ],
    );
    assert_eq!(state.jobs[0].record.estimated_duration_s, None);
    assert_eq!(state.jobs[1].record.estimated_duration_s, Some(0));
}

#[tokio::test]
async fn queue_position_is_the_same_after_a_restart() {
    let db = TempDb::new("order");
    {
        let registry = registry_at(&db.path());
        // Submitted in the same second on purpose: only the submission order
        // can tell these apart, and the position each was quoted has to hold.
        for id in ["first", "second", "third"] {
            let mut r = record(id, 0);
            r.queued = true;
            r.submitted = 1_700_000_000;
            r.plan = JobPlan::default();
            r.queue_req = Some(SubmitJobRequest::default());
            registry
                .insert_job(Job::from_record(r, HashMap::new()))
                .await;
        }
        assert_eq!(registry.queue_position("second").await, 2);
        registry.flush().await.expect("flush");
    }

    let state = Store::load(&db.path()).expect("load");
    let ids: Vec<&str> = state
        .jobs
        .iter()
        .map(|j| j.record.job_id.as_str())
        .collect();
    assert_eq!(ids, vec!["first", "second", "third"]);

    let registry = registry_at(&db.path());
    assert_eq!(registry.queue_position("first").await, 1);
    assert_eq!(registry.queue_position("second").await, 2);
    assert_eq!(registry.queue_position("third").await, 3);
    let queued: Vec<String> = registry
        .queued_jobs()
        .await
        .into_iter()
        .map(|(id, _, _)| id)
        .collect();
    assert_eq!(queued, vec!["first", "second", "third"]);
}

#[tokio::test]
async fn benchmarks_and_measured_links_survive() {
    let db = TempDb::new("measurements");
    {
        let registry = registry_at(&db.path());
        registry
            .record_benchmarks(&[
                GpuBenchmark {
                    node_id: "gpu-a".into(),
                    uuid: "GPU-aaaa".into(),
                    tflops: 41.5,
                    ..Default::default()
                },
                GpuBenchmark {
                    node_id: "gpu-b".into(),
                    uuid: "GPU-bbbb".into(),
                    tflops: 12.25,
                    ..Default::default()
                },
                // A failed probe: recorded nowhere, so not stored either.
                GpuBenchmark {
                    node_id: "gpu-c".into(),
                    uuid: "GPU-cccc".into(),
                    error: "no CUDA device".into(),
                    ..Default::default()
                },
            ])
            .await;
        registry
            .record_network(&[
                NetPair {
                    from_node: "gpu-b".into(),
                    to_node: "gpu-a".into(),
                    mbps: 942.0,
                    ..Default::default()
                },
                // Unreachable, so unmeasured -- not the slowest link there is.
                NetPair {
                    from_node: "gpu-a".into(),
                    to_node: "gpu-c".into(),
                    error: "connection refused".into(),
                    ..Default::default()
                },
            ])
            .await;
        registry.flush().await.expect("flush");
    }

    let state = Store::load(&db.path()).expect("load");
    assert_eq!(state.bench.len(), 2);
    assert_eq!(state.bench["GPU-aaaa"].0, 41.5);
    assert_eq!(state.bench["GPU-bbbb"].0, 12.25);
    assert!(state.bench["GPU-aaaa"].1 > 0, "measurement time is stored");
    // Reported b->a but keyed on the unordered pair, so it is one row.
    assert_eq!(state.network.len(), 1);
    assert_eq!(state.network[0].mbps, 942.0);

    // And the restored registry answers about the pair in either direction.
    let registry = registry_at(&db.path());
    let net = registry.network_snapshot().await;
    let now = state.network[0].measured_unix_s;
    assert_eq!(net.between("gpu-a", "gpu-b", now, 0), Some(942.0));
    assert_eq!(net.between("gpu-b", "gpu-a", now, 0), Some(942.0));
}

#[tokio::test]
async fn per_rank_statuses_survive_and_the_phase_is_unchanged() {
    let db = TempDb::new("statuses");
    let mut running = record("jrun", 0);
    running.plan = plan(&[("gpu-a", 2), ("gpu-b", 2)]);

    let before = {
        let registry = registry_at(&db.path());
        registry
            .insert_job(Job::from_record(running.clone(), HashMap::new()))
            .await;
        for (node, rank) in [("gpu-a", 0), ("gpu-b", 1)] {
            registry
                .update_job_status(status("jrun", node, rank, JobPhase::Running))
                .await;
        }
        registry.flush().await.expect("flush");
        let g = registry.inner.lock().await;
        g.jobs["jrun"].phase()
    };
    assert_eq!(before, JobPhase::Running);

    let state = Store::load(&db.path()).expect("load");
    let loaded = only_job(&state);
    assert_eq!(loaded.per_node.len(), 2);
    assert_eq!(
        loaded.per_node["gpu-a"],
        status("jrun", "gpu-a", 0, JobPhase::Running)
    );
    assert_eq!(
        loaded.per_node["gpu-b"],
        status("jrun", "gpu-b", 1, JobPhase::Running)
    );

    // Restored, not reconciled: it was running when the process died and it is
    // running now, whatever the agents would say about those ranks.
    let registry = registry_at(&db.path());
    let g = registry.inner.lock().await;
    assert_eq!(g.jobs["jrun"].phase(), before);
    assert_eq!(g.jobs["jrun"].plan, running.plan);
    // Logs and the live queue assessment were never written down.
    assert!(g.jobs["jrun"].logs.is_empty());
    assert!(g.jobs["jrun"].node_verdicts.is_empty());
}

#[test]
fn terminal_jobs_survive_too() {
    // History is the point: a controller that keeps only what is still running
    // forgets exactly the jobs somebody wants to look up afterwards.
    let db = TempDb::new("history");
    let mut done = record("jdone", 0);
    done.plan = plan(&[("gpu-a", 1)]);
    let mut failed = record("jfail", 1);
    failed.plan = plan(&[("gpu-b", 1)]);

    let state = round_trip(
        &db.path(),
        vec![
            Change::Job(Box::new(done)),
            Change::Status {
                job_id: "jdone".into(),
                node_id: "gpu-a".into(),
                status: Box::new(status("jdone", "gpu-a", 0, JobPhase::Succeeded)),
            },
            Change::Job(Box::new(failed)),
            Change::Status {
                job_id: "jfail".into(),
                node_id: "gpu-b".into(),
                status: Box::new(status("jfail", "gpu-b", 0, JobPhase::Failed)),
            },
        ],
    );

    assert_eq!(state.jobs.len(), 2);
    let phases: Vec<JobPhase> = state
        .jobs
        .iter()
        .map(|j| Job::from_record(j.record.clone(), j.per_node.clone()).phase())
        .collect();
    assert_eq!(phases, vec![JobPhase::Succeeded, JobPhase::Failed]);
}

#[tokio::test]
async fn flush_waits_for_the_commit() {
    // The point of the marker: after flush returns, a connection that knows
    // nothing about this process's channel can already see the row. Without it
    // the write is merely queued, which is what `submit_job` must not report as
    // accepted.
    let db = TempDb::new("flush");
    let store = Store::open(&db.path()).expect("open");

    assert!(Store::load(&db.path()).expect("load").jobs.is_empty());

    store.write(Change::Job(Box::new(record("jflush", 0))));
    store.flush().await.expect("flush");

    // A separate connection, while the store above is still open.
    let state = Store::load(&db.path()).expect("load");
    assert_eq!(state.jobs.len(), 1);
    assert_eq!(state.jobs[0].record.job_id, "jflush");
}

#[test]
fn a_database_from_the_future_is_refused() {
    let db = TempDb::new("future");
    let path = db.path();
    drop(Store::open(&path).expect("open"));

    // Something a later FerroGrid wrote: opening it read-write would mean
    // dropping whatever columns this build does not know about.
    let conn = rusqlite::Connection::open(&path).expect("raw open");
    conn.pragma_update(None, "user_version", 99).expect("bump");
    drop(conn);

    let err = Store::load(&path).expect_err("must refuse a newer schema");
    let message = err.to_string();
    assert!(
        message.contains("newer FerroGrid") && message.contains("v99"),
        "unhelpful error: {message}"
    );
    assert!(Store::open(&path).is_err(), "open must refuse it too");
}
