//! Controller state that outlives the process.
//!
//! What gets written down is decided by where the information comes from, not
//! by how expensive it was to obtain. Anything an agent re-reports on its next
//! heartbeat -- the node list, the GPU inventory, utilisation, the processes on
//! a card, which job holds which GPU -- is cheaper to rediscover than to
//! reconcile, and a stale copy of it read off disk at startup is worse than no
//! copy at all: for one heartbeat interval it looks exactly like current truth.
//! None of it is stored.
//!
//! What is stored is what nobody else remembers. A job record exists in the
//! controller and nowhere else, and so does the order jobs were submitted in,
//! which is the promise the queue makes to its users. GPU benchmarks and
//! `ferro net` measurements are the other two: each costs a cluster-wide
//! measurement run, and nothing re-reports them.
//!
//! The event log is the other thing nobody else remembers, and it is kept for
//! a different reason from the rest: a job record says what a job is now, and
//! only the log says what it went through to get there. Nothing reconstructs
//! that afterwards -- a card that was taken and given back leaves no trace in
//! any current state -- so it is written down as it happens.
//!
//! Some of what is ours is still left out on purpose. The per-job log ring is
//! 20k lines and exists to be tailed live, not archived; the NCCL errors,
//! metrics and utilisation averages are derived from those same lines. The
//! queue assessment -- per-node verdicts, warnings, the queue message -- is a
//! *live* judgement about a cluster that has since moved on, and the next queue
//! tick regenerates it; restoring it would mean answering "why is my job
//! waiting?" with last week's reasons.
//!
//! The hard rule in the implementation is that the controller never waits for a
//! disk while holding the registry lock. A writer hands a small [`Change`] to a
//! channel and carries on; a dedicated thread drains the channel and commits
//! whatever it finds in one transaction, so a burst of updates costs one fsync
//! rather than one each. [`Store::flush`] is the exception that proves the rule:
//! it is awaited, outside the lock, on the one path where write-behind is not
//! good enough.

use ferro_proto::{JobPlan, JobStatus, PlacementExplanation, SubmitJobRequest};
use prost::Message;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};

/// Schema version this build writes and understands.
const SCHEMA_VERSION: i32 = 2;

/// How many events are kept: the ring in memory and the tail read back off
/// disk at startup are the same number, because the ring is the read path and
/// loading rows it has no room for would be reading rows nobody can ask for.
pub const EVENT_RING: usize = 2000;

/// Most changes one transaction will swallow. The channel is unbounded because
/// a producer must never block, but a transaction that grows without limit
/// would trade the fsync it saves for a commit nobody can predict the cost of.
const MAX_BATCH: usize = 1024;

/// How long to wait for another connection's write lock before giving up.
/// Readers (`Store::load` on a live database) and the writer thread do overlap.
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS jobs (
    job_id                TEXT PRIMARY KEY,
    order_seq             INTEGER NOT NULL,
    name                  TEXT NOT NULL,
    submitted_by          TEXT NOT NULL,
    project               TEXT NOT NULL,
    priority              INTEGER NOT NULL,
    -- NULL means the submitter did not say, which the SJF policy reads
    -- differently from a declared zero. Never collapse the two.
    estimated_duration_s  INTEGER,
    timeout_s             INTEGER NOT NULL,
    submitted_unix_s      INTEGER NOT NULL,
    queued                INTEGER NOT NULL,
    queue_deadline_unix_s INTEGER NOT NULL,
    plan                  BLOB NOT NULL,
    queue_req             BLOB,
    placement             BLOB
);
CREATE TABLE IF NOT EXISTS job_status (
    job_id  TEXT NOT NULL,
    node_id TEXT NOT NULL,
    status  BLOB NOT NULL,
    PRIMARY KEY (job_id, node_id)
);
CREATE TABLE IF NOT EXISTS benchmarks (
    uuid            TEXT PRIMARY KEY,
    tflops          REAL NOT NULL,
    measured_unix_s INTEGER NOT NULL
);
-- Ids are assigned by the controller rather than by SQLite, so the ring in
-- memory and the rows on disk agree about which event is which. Only the tail
-- is ever read back, in id order, which the primary key already serves; the
-- filtering `ferro events` does happens in memory, so no other index earns
-- its write cost.
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY,
    unix_s  INTEGER NOT NULL,
    kind    TEXT NOT NULL,
    job_id  TEXT NOT NULL,
    node_id TEXT NOT NULL,
    actor   TEXT NOT NULL,
    detail  TEXT NOT NULL
);
-- Keyed by the pair sorted, so a probe in either direction lands on one row.
CREATE TABLE IF NOT EXISTS network (
    node_a          TEXT NOT NULL,
    node_b          TEXT NOT NULL,
    mbps            REAL NOT NULL,
    measured_unix_s INTEGER NOT NULL,
    PRIMARY KEY (node_a, node_b)
);
";

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("{0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("cannot create the directory for {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },

    /// Opening a newer schema read-only and hoping for the best is how a
    /// downgrade silently drops columns it does not know about. Refuse instead
    /// and let the operator choose.
    #[error(
        "{path} was written by a newer FerroGrid (schema v{found}; this build knows v{known}). \
         Upgrade the controller, or point --state at a different file."
    )]
    FutureSchema {
        path: String,
        found: i32,
        known: i32,
    },

    #[error("could not commit to {path}: {message}")]
    Write { path: String, message: String },

    #[error("the state writer for {path} is gone; nothing more will be persisted")]
    WriterGone { path: String },
}

/// The durable half of a `Job`. Everything a restarted controller needs to
/// show the same job it showed before, and nothing that a heartbeat or the next
/// queue tick would supply anyway.
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub job_id: String,
    /// Position in the registry's `job_order`. Queue position is a promise, so
    /// it has to survive a restart, and submission timestamps cannot carry it:
    /// two jobs submitted in the same second still have an order.
    pub order_seq: i64,
    pub name: String,
    pub submitted_by: String,
    pub project: String,
    pub priority: u32,
    pub estimated_duration_s: Option<u32>,
    pub timeout_s: u32,
    pub submitted: i64,
    pub queued: bool,
    pub queue_deadline: i64,
    pub plan: JobPlan,
    pub queue_req: Option<SubmitJobRequest>,
    pub placement: Option<PlacementExplanation>,
}

/// What happened. These spellings are the wire format, the CLI's `--kind`
/// filter and the column in the table all at once, which is why they are an
/// enum rather than a string every caller gets to spell for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    JobSubmitted,
    JobQueued,
    JobScheduled,
    JobStarted,
    JobCompleted,
    JobFailed,
    JobCancelled,
    NodeRegistered,
    NodeLost,
    NodeRecovered,
    GpuAllocated,
    GpuReleased,
    ControllerStarted,
    ControllerRecovered,
    /// The recovery window closed: what became of the jobs that came back off
    /// disk, counted. One line rather than a mechanism of its own, because
    /// "how long did recovery take and how many jobs did it lose" is a
    /// question about something that happened, which is what the log is for.
    Reconciled,
}

impl EventKind {
    /// Every kind, for the "did you mean" an unknown filter deserves.
    pub const ALL: [EventKind; 15] = [
        EventKind::JobSubmitted,
        EventKind::JobQueued,
        EventKind::JobScheduled,
        EventKind::JobStarted,
        EventKind::JobCompleted,
        EventKind::JobFailed,
        EventKind::JobCancelled,
        EventKind::NodeRegistered,
        EventKind::NodeLost,
        EventKind::NodeRecovered,
        EventKind::GpuAllocated,
        EventKind::GpuReleased,
        EventKind::ControllerStarted,
        EventKind::ControllerRecovered,
        EventKind::Reconciled,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::JobSubmitted => "JOB_SUBMITTED",
            EventKind::JobQueued => "JOB_QUEUED",
            EventKind::JobScheduled => "JOB_SCHEDULED",
            EventKind::JobStarted => "JOB_STARTED",
            EventKind::JobCompleted => "JOB_COMPLETED",
            EventKind::JobFailed => "JOB_FAILED",
            EventKind::JobCancelled => "JOB_CANCELLED",
            EventKind::NodeRegistered => "NODE_REGISTERED",
            EventKind::NodeLost => "NODE_LOST",
            EventKind::NodeRecovered => "NODE_RECOVERED",
            EventKind::GpuAllocated => "GPU_ALLOCATED",
            EventKind::GpuReleased => "GPU_RELEASED",
            EventKind::ControllerStarted => "CONTROLLER_STARTED",
            EventKind::ControllerRecovered => "CONTROLLER_RECOVERED",
            EventKind::Reconciled => "RECONCILED",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        EventKind::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

/// Something that already happened, as recorded.
///
/// Every field but the kind is optional in practice: a node registering has no
/// job, a controller starting has neither, and `detail` is whatever short
/// thing a reader would otherwise have to go and look up -- which GPUs, which
/// exit code, which position in line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// Monotonic, and monotonic across restarts: the sequence picks up from
    /// the last id read back off disk.
    pub id: u64,
    pub unix_s: i64,
    pub kind: EventKind,
    pub job_id: String,
    pub node_id: String,
    /// Whoever submitted the job this is about, where there is one.
    pub actor: String,
    pub detail: String,
}

impl Event {
    /// A bare fact. `id` and `unix_s` are stamped by the registry when it
    /// records this: an event is dated by when it was recorded and numbered by
    /// the registry that owns the sequence, neither of which a caller knows.
    pub fn new(kind: EventKind) -> Self {
        Self {
            id: 0,
            unix_s: 0,
            kind,
            job_id: String::new(),
            node_id: String::new(),
            actor: String::new(),
            detail: String::new(),
        }
    }

    /// The timeline's own provenance, and the first thing in it. An empty
    /// history means one thing after a fresh start and quite another after a
    /// restart that had a database to read, and only this line tells them
    /// apart -- which is a claim about jobs that actually came back, not about
    /// having had a file to look in.
    pub fn controller_start(restored_jobs: usize) -> Self {
        match restored_jobs {
            0 => Event::new(EventKind::ControllerStarted),
            n => Event::new(EventKind::ControllerRecovered).detail(format!("restored {n} job(s)")),
        }
    }

    pub fn job(mut self, job_id: &str) -> Self {
        self.job_id = job_id.to_string();
        self
    }

    pub fn node(mut self, node_id: &str) -> Self {
        self.node_id = node_id.to_string();
        self
    }

    pub fn actor(mut self, actor: &str) -> Self {
        self.actor = actor.to_string();
        self
    }

    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }
}

/// One unit of work for the writer thread, small enough that queueing it under
/// the registry lock costs a move and a channel push.
#[derive(Debug)]
pub enum Change {
    /// The whole job row, rewritten. Jobs change rarely enough that a diff
    /// would be more machinery than it saves.
    Job(Box<JobRecord>),
    Status {
        job_id: String,
        node_id: String,
        status: Box<JobStatus>,
    },
    Benchmark {
        uuid: String,
        tflops: f64,
        measured_unix_s: i64,
    },
    Network {
        a: String,
        b: String,
        mbps: f64,
        measured_unix_s: i64,
    },
    /// One line of the timeline. Deliberately never flushed for: it is a note
    /// about something that has already happened, and putting a disk in front
    /// of the control plane for one would undo the point of write-behind.
    Event(Event),
    /// A marker, not a write. The writer answers it once everything queued
    /// ahead of it is committed, which is what makes ordering the guarantee
    /// rather than timing.
    Flush(oneshot::Sender<Result<(), String>>),
}

/// A job as it came back off disk.
#[derive(Debug, Clone)]
pub struct LoadedJob {
    pub record: JobRecord,
    pub per_node: HashMap<String, JobStatus>,
}

/// One `ferro net` measurement, as stored.
#[derive(Debug, Clone)]
pub struct LinkRow {
    pub a: String,
    pub b: String,
    pub mbps: f64,
    pub measured_unix_s: i64,
}

/// Everything needed to repopulate a registry.
#[derive(Debug, Clone, Default)]
pub struct LoadedState {
    /// In `order_seq` order, which is the `job_order` to rebuild.
    pub jobs: Vec<LoadedJob>,
    /// GPU uuid -> (TFLOP/s, when it was measured).
    pub bench: HashMap<String, (f64, i64)>,
    pub network: Vec<LinkRow>,
    /// The tail of the timeline, oldest first, ready to become the ring.
    pub events: Vec<Event>,
}

/// A handle onto the writer thread. Cloning is deliberately not offered: the
/// registry owns exactly one, and a second one would make "flushed" ambiguous.
pub struct Store {
    /// `Option` only so `Drop` can close the channel before joining.
    tx: Option<mpsc::UnboundedSender<Change>>,
    writer: Option<std::thread::JoinHandle<()>>,
    path: PathBuf,
}

impl Store {
    /// Open (creating as needed) and start the writer thread.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        let conn = prepare(path)?;
        let (tx, rx) = mpsc::unbounded_channel();
        let owned = path.to_path_buf();
        let writer = std::thread::Builder::new()
            .name("ferro-state".into())
            .spawn(move || writer_loop(conn, rx, &owned))
            .map_err(|source| StoreError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Ok(Self {
            tx: Some(tx),
            writer: Some(writer),
            path: path.to_path_buf(),
        })
    }

    /// Read the whole database back. Uses its own connection, so this is also
    /// how another process -- or a test -- observes what a live store committed.
    pub fn load(path: &Path) -> Result<LoadedState, StoreError> {
        let conn = prepare(path)?;
        Ok(LoadedState {
            jobs: load_jobs(&conn)?,
            bench: load_benchmarks(&conn)?,
            network: load_network(&conn)?,
            events: load_events(&conn)?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Queue a change. Never blocks and never fails the caller: this is what
    /// runs under the registry lock, and a store that cannot keep up must not
    /// be able to stall the scheduler.
    pub fn write(&self, change: Change) {
        let Some(tx) = &self.tx else { return };
        if tx.send(change).is_err() {
            tracing::warn!(
                state = %self.path.display(),
                "state writer stopped; this change was not persisted"
            );
        }
    }

    /// Return once everything queued so far is committed.
    ///
    /// Must be awaited outside the registry lock. It is ordered rather than
    /// timed: the marker goes through the same channel as the writes, so it
    /// cannot overtake them or be overtaken.
    pub async fn flush(&self) -> Result<(), StoreError> {
        let (tx, rx) = oneshot::channel();
        let gone = || StoreError::WriterGone {
            path: self.path.display().to_string(),
        };
        self.tx
            .as_ref()
            .ok_or_else(gone)?
            .send(Change::Flush(tx))
            .map_err(|_| gone())?;
        match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(message)) => Err(StoreError::Write {
                path: self.path.display().to_string(),
                message,
            }),
            Err(_) => Err(gone()),
        }
    }
}

impl Drop for Store {
    /// Closing the channel ends the writer's loop, but only after it has
    /// drained what is still queued -- a controller shutting down should not
    /// lose the last few seconds of write-behind.
    fn drop(&mut self) {
        self.tx.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").field("path", &self.path).finish()
    }
}

fn prepare(path: &Path) -> Result<Connection, StoreError> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|source| StoreError::Io {
            path: dir.display().to_string(),
            source,
        })?;
    }
    let conn = Connection::open(path)?;
    // WAL so a reader never blocks the writer; NORMAL because the writer
    // batches and the one place that needs a durability barrier asks for it
    // explicitly, by flushing.
    conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get::<_, String>(0))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    migrate(&conn, path)?;
    Ok(conn)
}

fn migrate(conn: &Connection, path: &Path) -> Result<(), StoreError> {
    let found: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if found > SCHEMA_VERSION {
        return Err(StoreError::FutureSchema {
            path: path.display().to_string(),
            found,
            known: SCHEMA_VERSION,
        });
    }
    // Every version so far has only ever *added* a table, so replaying the
    // whole schema is the upgrade: the older tables are left exactly as they
    // are and the new ones appear. A change that altered an existing table
    // would need its own step here, keyed on `found` -- this shortcut is the
    // reward for additive changes, not a licence to skip the question.
    conn.execute_batch(SCHEMA)?;
    if found < SCHEMA_VERSION {
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }
    Ok(())
}

fn writer_loop(mut conn: Connection, mut rx: mpsc::UnboundedReceiver<Change>, path: &Path) {
    let mut batch: Vec<Change> = Vec::new();
    // Returns 0 only once the channel is closed *and* drained, so a shutdown
    // still commits whatever was in flight.
    while rx.blocking_recv_many(&mut batch, MAX_BATCH) > 0 {
        let outcome = apply(&mut conn, &batch).map_err(|e| e.to_string());
        if let Err(message) = &outcome {
            tracing::error!(state = %path.display(), "state write failed: {message}");
        }
        // After the commit, never before: a waiter is being told its data is
        // on disk.
        for change in batch.drain(..) {
            if let Change::Flush(reply) = change {
                let _ = reply.send(outcome.clone());
            }
        }
    }
}

fn apply(conn: &mut Connection, batch: &[Change]) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;
    for change in batch {
        match change {
            Change::Job(job) => {
                tx.execute(
                    "INSERT OR REPLACE INTO jobs \
                     (job_id, order_seq, name, submitted_by, project, priority, \
                      estimated_duration_s, timeout_s, submitted_unix_s, queued, \
                      queue_deadline_unix_s, plan, queue_req, placement) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                    params![
                        job.job_id,
                        job.order_seq,
                        job.name,
                        job.submitted_by,
                        job.project,
                        job.priority,
                        job.estimated_duration_s,
                        job.timeout_s,
                        job.submitted,
                        job.queued,
                        job.queue_deadline,
                        job.plan.encode_to_vec(),
                        job.queue_req.as_ref().map(Message::encode_to_vec),
                        job.placement.as_ref().map(Message::encode_to_vec),
                    ],
                )?;
            }
            Change::Status {
                job_id,
                node_id,
                status,
            } => {
                tx.execute(
                    "INSERT OR REPLACE INTO job_status (job_id, node_id, status) \
                     VALUES (?1, ?2, ?3)",
                    params![job_id, node_id, status.encode_to_vec()],
                )?;
            }
            Change::Benchmark {
                uuid,
                tflops,
                measured_unix_s,
            } => {
                tx.execute(
                    "INSERT OR REPLACE INTO benchmarks (uuid, tflops, measured_unix_s) \
                     VALUES (?1, ?2, ?3)",
                    params![uuid, tflops, measured_unix_s],
                )?;
            }
            Change::Network {
                a,
                b,
                mbps,
                measured_unix_s,
            } => {
                let (first, second) = if a <= b { (a, b) } else { (b, a) };
                tx.execute(
                    "INSERT OR REPLACE INTO network (node_a, node_b, mbps, measured_unix_s) \
                     VALUES (?1, ?2, ?3, ?4)",
                    params![first, second, mbps, measured_unix_s],
                )?;
            }
            Change::Event(event) => {
                tx.execute(
                    "INSERT OR REPLACE INTO events \
                     (id, unix_s, kind, job_id, node_id, actor, detail) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        // SQLite integers are signed, and at one event per
                        // microsecond the difference runs out in 292,000
                        // years. The cast is the whole of the mismatch.
                        event.id as i64,
                        event.unix_s,
                        event.kind.as_str(),
                        event.job_id,
                        event.node_id,
                        event.actor,
                        event.detail,
                    ],
                )?;
            }
            Change::Flush(_) => {}
        }
    }
    tx.commit()
}

/// A blob that will not decode means the file is damaged, not that one job is
/// odd. Surface it and let the operator decide rather than starting up with a
/// silently shorter history.
fn decode<T: Message + Default>(blob: Vec<u8>, column: usize) -> rusqlite::Result<T> {
    T::decode(&blob[..]).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(column, rusqlite::types::Type::Blob, Box::new(e))
    })
}

fn load_jobs(conn: &Connection) -> Result<Vec<LoadedJob>, StoreError> {
    let mut per_node: HashMap<String, HashMap<String, JobStatus>> = HashMap::new();
    let mut stmt = conn.prepare("SELECT job_id, node_id, status FROM job_status")?;
    let rows = stmt.query_map([], |row| {
        let job_id: String = row.get(0)?;
        let node_id: String = row.get(1)?;
        let status: JobStatus = decode(row.get(2)?, 2)?;
        Ok((job_id, node_id, status))
    })?;
    for row in rows {
        let (job_id, node_id, status) = row?;
        per_node.entry(job_id).or_default().insert(node_id, status);
    }

    let mut stmt = conn.prepare(
        "SELECT job_id, order_seq, name, submitted_by, project, priority, \
                estimated_duration_s, timeout_s, submitted_unix_s, queued, \
                queue_deadline_unix_s, plan, queue_req, placement \
         FROM jobs ORDER BY order_seq",
    )?;
    let jobs = stmt
        .query_map([], |row| {
            let queue_req: Option<Vec<u8>> = row.get(12)?;
            let placement: Option<Vec<u8>> = row.get(13)?;
            Ok(JobRecord {
                job_id: row.get(0)?,
                order_seq: row.get(1)?,
                name: row.get(2)?,
                submitted_by: row.get(3)?,
                project: row.get(4)?,
                priority: row.get(5)?,
                estimated_duration_s: row.get(6)?,
                timeout_s: row.get(7)?,
                submitted: row.get(8)?,
                queued: row.get(9)?,
                queue_deadline: row.get(10)?,
                plan: decode(row.get(11)?, 11)?,
                queue_req: queue_req.map(|b| decode(b, 12)).transpose()?,
                placement: placement.map(|b| decode(b, 13)).transpose()?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(jobs
        .into_iter()
        .map(|record| LoadedJob {
            per_node: per_node.remove(&record.job_id).unwrap_or_default(),
            record,
        })
        .collect())
}

fn load_benchmarks(conn: &Connection) -> Result<HashMap<String, (f64, i64)>, StoreError> {
    let mut stmt = conn.prepare("SELECT uuid, tflops, measured_unix_s FROM benchmarks")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, (row.get(1)?, row.get(2)?)))
    })?;
    Ok(rows.collect::<rusqlite::Result<HashMap<_, _>>>()?)
}

/// The newest [`EVENT_RING`] events, returned oldest first so they can be
/// pushed straight onto the ring in the order they happened.
fn load_events(conn: &Connection) -> Result<Vec<Event>, StoreError> {
    let mut stmt = conn.prepare(
        "SELECT id, unix_s, kind, job_id, node_id, actor, detail \
         FROM events ORDER BY id DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([EVENT_RING as i64], |row| {
        Ok((
            row.get::<_, i64>(0)? as u64,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, String>(6)?,
        ))
    })?;

    let mut events = Vec::new();
    for row in rows {
        let (id, unix_s, kind, job_id, node_id, actor, detail) = row?;
        // A kind this build has no name for can only come from a later one
        // that added kinds without changing the schema. Leaving it out is
        // better than a row in the timeline that says nothing.
        let Some(kind) = EventKind::parse(&kind) else {
            tracing::debug!("ignoring event {id}: unknown kind {kind}");
            continue;
        };
        events.push(Event {
            id,
            unix_s,
            kind,
            job_id,
            node_id,
            actor,
            detail,
        });
    }
    events.reverse();
    Ok(events)
}

fn load_network(conn: &Connection) -> Result<Vec<LinkRow>, StoreError> {
    let mut stmt = conn.prepare("SELECT node_a, node_b, mbps, measured_unix_s FROM network")?;
    let rows = stmt.query_map([], |row| {
        Ok(LinkRow {
            a: row.get(0)?,
            b: row.get(1)?,
            mbps: row.get(2)?,
            measured_unix_s: row.get(3)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}
