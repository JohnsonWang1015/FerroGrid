# Persistence

What the controller writes down, what it deliberately does not, and why the distinction is the whole design.

> Status: the design below is what Phase 5 implements. What actually landed, and what did not, is recorded in [`progress.md`](progress.md).

---

## 1. The problem

Controller state lives in one `Mutex<RegistryInner>`. Restart the process and every job record is gone. Node registrations self-heal within a heartbeat — agents re-register on every reconnect — but the jobs those nodes are running become invisible: no `ferro job`, no `ferro logs`, and `ferro cancel` answers `not_found` while the work carries on burning GPUs.

That is the gap. It is narrower than "we lose everything", and saying so precisely is what keeps the fix small.

---

## 2. What is durable, and what is not

This split is the design. Everything else follows from it.

### Durable

| State | Why |
|---|---|
| Job records — id, name, owner, project, priority, estimated duration, timeout, submission time | The thing that is actually lost today |
| Queue membership, the deadline, and the original `SubmitJobRequest` | A queued job cannot be re-placed without the request that described it |
| `job_order` | Queue position must be stable across a restart, or the number a user was quoted stops meaning anything |
| The plan, and the placement explanation | "Why these GPUs?" should still be answerable next week |
| Per-rank `JobStatus` | The job's phase is a vote of its ranks; without them a restored job has no phase |
| GPU benchmarks | `ferro bench` is slow and its results do not change between reboots |
| Network measurements | Same, and the scheduler now reads them |

### Ephemeral

| State | Why not |
|---|---|
| Node registrations, GPU lists, utilisation, processes | Rebuilt within one heartbeat, and writing them is exactly the per-second telemetry flood §20 warns against |
| `allocated_job_id` on each GPU | The agent is the real allocator; its heartbeat is authoritative |
| The 20,000-line log ring per job | High volume, and persisting it properly is a different problem — see §6 |
| `nccl_errors`, metrics, utilisation averages | Derived from logs and heartbeats |
| Queue verdicts, warnings, `queue_message` | A *live* assessment. The next queue tick regenerates it in at most five seconds, and a stale explanation is worse than none |

The line between the two columns is "can a heartbeat rebuild it?". Where the answer is yes, writing it down buys nothing and costs a disk write every few seconds per node.

---

## 3. Never hold the lock over the disk

§79 is explicit about this, and it is the constraint that shapes the implementation:

> Persistence 不應 hold global scheduler lock while waiting for slow disk.

The registry's mutex is taken by every scheduling decision, every heartbeat and every CLI read. A synchronous `INSERT` inside that critical section would serialise the entire control plane behind fsync.

So writes are **queued, not performed**, under the lock:

```
registry method  ──(lock)──▶ mutate memory
                             push a small Change onto a channel   ← non-blocking
                 ──(unlock)─▶

writer task      ──▶ drain the channel
                 ──▶ apply a batch inside one transaction
                 ──▶ one commit, not N
```

Two properties matter and are worth stating rather than assuming:

- The channel send is **non-blocking**. A bounded channel that filled would block a caller that is holding the lock, which is the failure this design exists to avoid — smuggled in through the back door.
- Batching means a burst of status updates costs one fsync rather than one each. A distributed job reporting four ranks at once is one write.

### The cost, stated plainly

Write-behind means a crash can lose the last few milliseconds of changes. That is an accepted trade, with one exception.

---

## 4. The one place durability is synchronous

`SubmitJob` flushes before it answers.

Telling a user their job was accepted and then losing the record in a crash is the specific failure this phase exists to prevent — the job would keep running, holding GPUs, with nothing in the controller that knows about it. So the submit path queues the insert, releases the lock, and *then* awaits a flush before returning `accepted: true`.

Everything else — status updates, benchmarks, network measurements — is write-behind. Those are recoverable from the cluster or cheap to redo; a lost job record is neither.

---

## 5. Storage choice

SQLite, with:

- **WAL** (`journal_mode=WAL`) so a reader never blocks the writer. `ferro jobs` should not wait behind a commit.
- `synchronous=NORMAL` rather than `FULL`. With WAL that risks losing the last transaction on an OS crash, not on a process crash, which is the right point on the curve for a single-host control plane.
- `PRAGMA user_version` for schema versioning. A database written by a newer build makes the controller **refuse to start**, rather than opening it and writing something the newer build cannot read back.

Protobuf messages that already have a wire format — `JobPlan`, `SubmitJobRequest`, `JobStatus`, `PlacementExplanation` — are stored as prost-encoded blobs rather than shredded into columns. They are already versioned by the proto's field numbering, which is the compatibility story this project has committed to; re-encoding them into columns would create a second, weaker one.

Scalars that need to be *queried* — owner, project, priority, queue state, submission time — are real columns.

One deliberate detail: `estimated_duration_s` is `NULL` when the submitter declared nothing, never `0`. The SJF policy treats those differently on purpose, and a storage layer that collapsed them would quietly undo it.

---

## 6. What this phase does not do

**Reconciliation.** Loading is not reconciling. A job that was `Running` at shutdown loads as `Running`, and no attempt is made to check that against what the agents say. Resolving *DB says running / agent says missing* — and its mirror image — is Phase 6, and doing it properly needs the loaded state to exist first.

The audit found this is cheaper than it looks: agents already report their running jobs and GPU ownership on every heartbeat, and the controller currently discards reports for jobs it does not recognise. The desired-state half is what was missing, and this phase supplies it.

**Log persistence.** The log ring stays in memory. Persisting it well means deciding a retention policy, a write path that does not amplify every stdout line into a disk write, and what to do when a job produces a gigabyte. That is its own piece of work, and the honest position is that `ferro logs` still loses history on a restart.

**Quota, accounting tables, per-user history.** Usage is derived from the job records on demand rather than stored separately, which keeps one source of truth (§90.3). Now that the job records survive, so does the usage history, without a second ledger to drift.
