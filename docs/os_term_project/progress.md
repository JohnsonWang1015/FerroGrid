# FerroGrid — OS Term Project Progress Log

Updated at the end of every phase, in the format fixed by the project
specification (§91). Newest entry first.

---

## Per-user GPU quota admission and usage accounting

**Status:** ✅ Controller-configured concurrent GPU limits, atomic admission,
queue retry behavior, `Controller.GetUsage`, and `ferro usage` are implemented.

`ferro-controller --user-quota USER=N` is repeatable. Duplicate identities and
malformed limits fail startup; an omitted user is unlimited and zero is a valid
limit. Exact requests that can never fit are rejected, temporary quota blocks
are rejected without `--wait` or retained in the queue with it, and automatic
placement is capped by remaining quota. Quota blocking has its own message,
separate from cluster capacity.

The race was prevented at the allocation boundary: `reserve_exact_with_quota`
holds the registry mutex while it checks physical GPU ownership, counts GPUs
already allocated to the submitter, checks the quota, and applies the full
reservation or none of it. The preflight check only supplies early feedback;
both immediate and queued jobs repeat admission at the atomic reservation.
Placement remains outside the critical section, and the agent's launch-time
GPU validation remains in place.

Usage reuses `RegistryInner::usage_snapshot`, the same job-derived data used by
fair-share, rather than maintaining a second cumulative ledger. The API reports
held GPUs, current nonqueued jobs, accumulated GPU-seconds, and an optional
quota; the CLI displays an absent quota as `unlimited` and preserves quota zero
as zero in JSON. Job-derived GPU-seconds remain available after restart from
persisted job records. The controller's quota flags are startup configuration
and must be supplied again after a restart.

### Verification

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | ✅ clean |
| `PROTOC=/tmp/ferro-protoc/bin/protoc cargo clippy --workspace --all-targets -- -D warnings` | ✅ clean |
| `PROTOC=/tmp/ferro-protoc/bin/protoc cargo test --workspace` | ✅ 289 passed, 0 failed |
| `uv run --all-extras pytest -q` | ✅ 14 passed, 5 skipped; one CUDA driver-version warning |

The first full Rust test run hit two unchanged `ferro-agent` process-fixture
assertions before `setsid` had moved the child into its own session. Both tests
passed when isolated and on the complete workspace rerun. The new mandatory
16-way same-user reservation test admits exactly two GPUs under a quota of two.
A mocked NodeAgent gRPC integration test also leaves a quota-blocked job queued,
releases the incumbent job, then dispatches and launches the waiting job. RPC
and renderer tests verify current usage and the difference between unlimited
and a zero-GPU limit. No real GPU result is claimed.

### Known limitations

- `submitted_by` is client supplied; quota enforcement is resource management,
  not an authentication or security boundary. The controller gRPC endpoint is
  still unauthenticated.
- Quota definitions are supplied at controller startup rather than persisted
  in the state database; operators must pass the same limits after restart.
- GPU-seconds and current job counts are derived from retained durable job
  records and the existing registry lifecycle semantics. There is no separate
  accounting ledger, identity provider, project quota, or GPU preemption.

---

## Graceful cancellation, and the workers that never died

**Status:** ✅ Closes the `--no-docker` leak found during the RQ3 experiment, and
§29 from the Phase 0 audit, which had been open since the beginning.

### The bug was worse than either description of it

The audit recorded §29 as "immediate SIGKILL". The RQ3 write-up recorded that
`child.start_kill()` "does not reach the workers". Both were too kind.

`launcher.rs` inserted `child: None` and moved the `Child` into the supervisor,
so `stop_job`'s `child.start_kill()` was **unreachable dead code**. Under
`--no-docker`, cancelling a job killed *nothing* — not the workers, not even
torchrun. A baseline run with the fix stashed shows both processes still alive
after `ferro cancel`, the rendezvous port still bound, and `ferro nodes`
cheerfully reporting the GPU as free.

### The process group was not the answer either

The brief for this work assumed the fix was to signal the existing process
group. It was not: `torch/distributed/elastic/.../subprocess_handler.py:66` sets
`start_new_session=True`, so every worker is the leader of its own session and
group. `kill(-pgid)` reaches torchrun and nothing else.

So teardown walks the descendant tree **before** signalling anything — once
torchrun dies its workers are reparented to init and parentage finds nothing —
keeping `(pid, start_time)` pairs so pid reuse cannot make it kill a stranger.

### Verified live, on real torchrun

| | SIGTERM-deaf script | SIGTERM-honouring script |
|---|---|---|
| teardown | **10 s** (the full grace) | **0.5 s** |
| surviving processes | none | none |
| GPU released | yes | yes |
| message | `cancelled by controller; did not exit in 10 s and was killed` | `cancelled by controller; exited on SIGTERM` |

The message distinction is the useful part: it tells an operator whether their
training script honours signals at all, which is the prerequisite for any
cooperative checkpointing (§31) later.

Independently re-verified by watching the status message over a timeline:
`cancelled by controller` from t+0.5 s through t+9 s, becoming
`…did not exit in 10 s and was killed` at t+11 s. The grace period is real.

### Docker path

`docker kill` became `docker stop --time <grace>`, so the same SIGTERM → wait →
SIGKILL contract applies under Docker, where the runtime implements it.

### Known limitations

1. **The grace period is per-job, not per-cluster.** Cancelling twenty jobs whose
   scripts all ignore SIGTERM takes ten seconds, not two hundred — teardowns run
   concurrently — but a slow node still holds its own for the full window.
2. **Nothing verifies the worker actually honoured the signal**, only that it
   exited. A script that exits on SIGTERM without checkpointing looks identical
   to one that saved its state first.
3. **`--no-docker` remains the weaker path.** Docker gives cgroup-level
   containment; the descendant walk is a best effort that a process deliberately
   escaping its tree could still defeat.

---

## RQ3 measurement — controller crash recovery

**Status:** ✅ The last research question now has data.

`scripts/run_recovery_experiment.sh` drives the real binaries — controller,
agent, 20 queued jobs and one genuinely running job — **SIGKILLs** the
controller and restarts it. SIGKILL rather than SIGTERM because a clean
shutdown is not the failure being studied.

| Metric | persistent | in-memory |
|---|---|---|
| Jobs known after restart | **21 / 21** | **0 / 21** |
| Lost jobs | **0** | **21** |
| Incorrect allocations | **0** | **1** |
| Queue order preserved | yes | n/a |
| Recovery time (controller-timed) | **30 s** | never reconciles |

Three runs agree on every count. Written up in
[`experiments.md`](experiments.md) §5, including two caveats that matter more
than the numbers:

- **`failed_by_reconcile = 0` does not mean reconciliation found nothing.** The
  agent survives the crash, so the running job is claimed and the queued ones
  are exempt. The "job nobody claims" path is *not* exercised by this
  experiment.
- **Two recovery times are reported, not one.** The controller's monotonic
  measure says 30 s; wall clock says 32–35 s, because this WSL2 host inflates
  30.000 s of monotonic time to 35.6 s of wall time. Quoting only the wall
  figure would report a scheduler overrunning its own configuration.

### A FerroGrid gap the harness exposed

Under `--no-docker`, torchrun puts its workers in a fresh session, so neither
the agent's `child.start_kill()` nor a process-group kill reaches them.
**Stopping an agent leaves its workers running and its GPUs held.** Docker mode
is covered by `docker kill`, which is presumably why it has not bitten. Worse,
the leftover worker holds the rendezvous port, so the next run's rank 0 dies
with `EADDRINUSE` — which from outside is indistinguishable from a job lost to
a crash. Not fixed; recorded.

---

## Phase 7 — Backfilling and reservation

**Phase:** 7 — Dispatch modes, EASY backfilling with reservation
**Status:** ✅ Complete. The feature works; the measurement says not to use it by default.

### Implemented

- **`crates/ferro-sched/src/dispatch.rs`** — one `Dispatch` enum shared by the
  controller and the simulator, so there cannot be two definitions of what the
  scheduler does.
- **`opportunistic`** (default, unchanged), **`strict`** (nothing overtakes the
  head), **`reserved`** (EASY backfilling: the first job that does not fit holds
  a reservation, and a later job may start only if it can *prove* it finishes
  before that reservation's earliest start).
- **The proof requirement is enforced, not waived.** A running job with neither
  a declared estimate nor a timeout is unknowable; if the reservation's earliest
  start cannot be computed, nothing past it is admitted. `reserved` then
  collapses onto `strict` rather than guessing.
- `--dispatch <MODE>` on the controller; `--with-strict` / `--with-reserved` in
  the experiment runner.

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **262 passed, 0 failed** (was 229; +33) |
| `cargo fmt --check` / `clippy -D warnings` | ✅ clean |
| Pre-existing assertions modified | **none** |

`reservation_is_given_no_information_the_controller_would_not_have` is the guard
that matters: the simulator knows every job's true duration, and reserving
against it would hand the algorithm information the controller can never have.
The test sets `estimate_fraction = 0` and asserts `reserved` collapses onto
`strict` — if it were reading the future it would still backfill.

### The result, and a correction to Phase 4

Workload D, split by job class, which the earlier write-up did not do:

| Dispatch | class | mean wait | overtaken by |
|---|---|---|---|
| `opportunistic` | distributed (36) | 7 844 s | **70.9** |
| `opportunistic` | small (164) | **142 s** | 0.0 |
| `strict` | distributed | 7 174 s | 0.0 |
| `strict` | small | 7 537 s | 0.0 |
| `reserved` | distributed | **6 951 s** | 3.2 |
| `reserved` | small | 6 997 s | 0.7 |

**Phase 4 claimed opportunistic dispatch "costs an unbounded tail for large
jobs" and that reservation "is what bounds it". That was wrong.** The claim came
from an aggregate p95 of 10 801 s against a mean of 1 529 s, read as starvation.
It is not a tail — it is a bimodal population, 82 % small and fast against 18 %
large and slow, and the 95th percentile simply lands in the second group.

The overtaking is real: 70 later arrivals start before a large job, on average.
The *consequence* is not. Forbid overtaking entirely and those jobs wait 9 %
less. Their wait is set by cluster saturation, not queue position.

Reservation does exactly what it promises — overtaking falls to 3.2, large jobs
improve 11 %, utilisation stays at 97.0 % — and charges the small jobs **49×**
for it. On this workload that is a bad trade, which is why it ships opt-in with
the default unmoved.

The first implementation had only EASY's first backfill condition; the second —
a candidate may use resources the reservation will not need anyway — was found
missing *after* this conclusion was drawn. It was implemented and everything
re-measured, because a verdict against half an algorithm is not a verdict.
Completing it moved reservation from 90 % of the way to strict to 92 %: the
arithmetic changed, the answer did not.

### Known limitations

1. **One workload shaped the conclusion.** A cluster with fewer, larger jobs, or
   one where a delayed large job costs more than many delayed small ones, would
   reach a different answer. The mechanism is implemented and available; the
   recommendation is workload-specific and says so.
2. **Reservation is only as good as the declarations.** With no estimates and no
   timeouts it is strict FIFO, which is measurably worse than the default.
3. **The earliest-start calculation ignores placement shape.** It counts free
   GPUs, not whether they can be assembled into the shape the reserved job
   needs, so a reservation can be satisfied on paper and still not place.
4. **No preemption.** Reservation holds a place; it never takes one back.

### Next

§85's minimum feature set and §86's strong target are both complete. What
remains from the specification is stretch work — preemption and cooperative
checkpointing, power and thermal awareness, quota, authentication — none of
which the measurements so far have shown a need for.

---

## Phase 6 — Recovery

**Phase:** 6 — Reconciling restored state against the cluster
**Status:** ✅ Complete. Gate met.

### Implemented

- **A reconciliation window** (`--reconcile-window-secs`, default 30) rather
  than an instant verdict. Agents reconnect on a 3 s heartbeat behind a 3 s
  backoff and a node is not called unhealthy for 15 s; deciding at startup
  would kill work that is merely slow to be reported.
- **Claiming from either signal** — a `JobStatus` report *or* a GPU carrying
  the job's `allocated_job_id`. Requiring both would lose jobs whose rank had
  nothing to say in that particular heartbeat.
- **Unclaimed jobs are failed with the distinction that matters**: a healthy
  node that did not mention the job (`lost while the controller was down`)
  versus a node that never reconnected (`fate unknown`). Both terminal, but
  only one is worth walking to a machine about.
- **Queued-but-running jobs are adopted**, with the plan rebuilt from the GPU
  ownership the agents report, and multi-node plans completed as later peers
  reconnect.
- **`reconciling` on `JobSummary`** (new field number) so an operator can see
  the controller is still making up its mind.
- **`RECONCILED` event** carrying restored / claimed / adopted / failed and the
  elapsed time — §69's recovery-time and lost-job measurements become two event
  rows rather than a benchmark harness.

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **229 passed, 0 failed** (was 220; +9 recovery) |
| `cargo fmt --check` / `clippy -D warnings` | ✅ clean |
| Pre-existing tests modified | **none** |

### Verified live

A real controller and agent, a genuinely running job, then **both** processes
killed and the controller alone restarted with a 6 s window:

```
  t+2s (window open):   longrunner: running  reconciling=True
  t+8s (window closed): longrunner: failed   reconciling=False

  JOB_FAILED   fate unknown: local-gpu has not reported since the controller restarted
  RECONCILED   restored 1, claimed 0, adopted 0, failed 1 in 6s
```

The job was not failed early, and the reason picked the correct branch — the
agent never came back, so the honest answer is "unknown", not "gone".

### A specification bug this phase caught

The brief said *every* job still reconciling and non-terminal becomes failed. A
restored **queued** job is non-terminal, so taken literally that would fail
every job waiting in line — destroying exactly the queue position Phase 5
persists. A queued job holds nothing and claims nothing; the dispatcher owns
it. It is now un-flagged at window close and left in line.

### Known limitations

1. **No elastic restart.** A job failed here is not resubmitted. Retry policy
   (§43) needs failure classification to be meaningful and is a separate
   decision.
2. **No agent-side cleanup.** If an agent is holding a container for a job the
   controller has failed, the controller does not reach in and kill it.
   Deciding to stop something on a node the operator did not ask about is a
   bigger call than recovery should make alone; §25's orphan classification is
   where that belongs.
3. **Restored jobs have no logs**, because logs are still not persisted.
4. **The window is fixed**, not adaptive. A cluster of fifty nodes with one slow
   machine waits the same 30 s as a cluster of two.

### Next

Phase 7 — advanced scheduling, by remaining time: backfilling with reservation
(the experiments showed FerroGrid already backfills without one, and what that
costs the tail), user quota, multi-resource requests, or orphan classification.

---

## Phase 5 — Persistence

**Phase:** 5 — SQLite state, event log
**Status:** ✅ Complete. Gate met.

### Implemented

- **`crates/ferro-controller/src/store.rs`** — SQLite with WAL, `synchronous=NORMAL`,
  `PRAGMA user_version` for schema versioning. A database from a future version
  makes the controller refuse to start rather than write something it cannot
  read back.
- **Durable:** job records, queue membership and the original request, `job_order`,
  the plan, the placement explanation, per-rank statuses, GPU benchmarks,
  network measurements, events.
  **Ephemeral:** node registrations, GPU lists, utilisation, `allocated_job_id`,
  the 20k-line log ring, queue verdicts. The dividing line is "can a heartbeat
  rebuild it?" — see [`persistence.md`](persistence.md).
- **Write-behind through a channel.** Registry methods queue a small `Change`
  under the lock — a non-blocking send on an unbounded channel — and a writer
  thread applies batches in one transaction. No disk I/O ever happens under the
  registry lock (§79).
- **One synchronous exception:** `submit_job` flushes before answering
  `accepted: true`. A flush marker travels the same channel, so it cannot
  overtake the writes it is waiting for, and the reply is sent after the commit.
- **Event log** (§22) — all fourteen kinds, write-behind, with a 2000-entry
  in-memory ring so `ferro events` works under `--no-state`. The ring is seeded
  from the table at startup.
- **`ferro events`** with `--job`, `--kind`, `--limit`, `--json` and the shared
  watch flags.
- **`--state <PATH>` / `--no-state`** on the controller, defaulting to
  `$XDG_STATE_HOME/ferrogrid/controller.db`.

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **220 passed, 0 failed** (was 199; +8 persistence, +13 events) |
| `cargo fmt --check` / `clippy -D warnings` | ✅ clean |
| Pre-existing tests modified | **none** |
| `ferro-sched` purity | ✅ gained no dependencies |

### Verified live, not just in unit tests

A controller was started against a state file, two jobs queued, then **killed
with SIGTERM** — which the controller does not catch, so it had no chance to
shut down cleanly. Both jobs came back with priority, project and queue order
intact. That is the synchronous-flush-on-submit decision earning its place:
write-behind alone would have lost them.

Restart also showed `CONTROLLER_RECOVERED  restored 2 job(s)` where a fresh
start says `CONTROLLER_STARTED`, and the event history survived in the ring.

### A bug found while testing the event log

A terminal job's event detail was taken from whichever report completed the
picture, which is not always the rank that ended. A rank can fail before its
peer has reported anything; the peer's *start* is then what flips the job to
failed, and `JOB_FAILED` was labelled with the peer's progress message —
`gpu-b says Running` instead of `exit 137`. The fix reads the detail off a rank
actually in that phase, lowest `node_rank` first, because `per_node` is a
`HashMap` and without the ordering the same failure would print differently
between runs.

### An effect that came for free

Restoring jobs closed half of §21's reconciliation without any code written for
it. Before, an agent's heartbeat reporting a job the controller did not
recognise was discarded. Now the job is restored, so the heartbeat applies: a
job that finished while the controller was down is correctly recorded as
finished. What remains for Phase 6 is the case where *the agent no longer knows
about the job either* — nothing reports, and the job stays `Running` forever.

### Known limitations

1. **No reconciliation.** Loading is not reconciling. A job that was running at
   shutdown loads as running and stays that way until a heartbeat says
   otherwise, or forever if none comes. This is deliberate: Phase 6.
2. **Logs are still not persisted.** `ferro logs` loses history on a restart,
   and the audit's finding that logs are dropped outright while the controller
   is unreachable is untouched.
3. **Write-behind loses its tail on a crash** for everything except job
   submission. Status updates, benchmarks and events queued in the last
   milliseconds go with the process.
4. **`rusqlite` is vendored (`bundled`)**, which adds a C compile to the build.
   The portable build container (`rust:1-bullseye`) already has a compiler, so
   `scripts/build.sh portable` needed no change — but it is now a build-time
   dependency worth knowing about.
5. **Queue verdicts and warnings are not persisted**, so a restored queued job
   shows no explanation until the next queue tick regenerates one, up to five
   seconds later.

### Next

Phase 6 — recovery: reconcile the loaded state against what the agents report,
in both directions, and handle the job nobody claims any more.

---

## Phase 4 — Evaluation

**Phase:** 4 — Simulator, workload generator, experiment runner
**Status:** ✅ Complete. Gate met, with one documented deviation.

### Implemented

- **`crates/ferro-sim`** — a discrete-event simulator that links `ferro-sched`
  and drives the *same* policy objects the controller runs. No second
  scheduler exists.
- **Workload generator** — Poisson arrivals, weighted users, weighted job
  classes, and a deliberate split between a job's *true* duration and the one
  its submitter *declared*, so SJF can be evaluated against estimates that are
  absent or wrong.
- **Own PRNG** (PCG-XSH-RR, 40 lines). A reproducibility claim that depends on
  a third party not changing an unspecified algorithm is not a reproducibility
  claim.
- **Eight scenarios** (§51 A–E, G, plus I for mixed priorities and H for bad
  estimates). F, controller failure, is absent rather than faked: a simulator
  with no controller cannot restart one.
- **Metrics** — avg/p50/p95/max wait, turnaround, makespan, utilisation,
  throughput, Jain fairness over GPU-seconds *and* over waiting time, the
  heavy-user wait ratio, starvation, fragmentation, mean slowdown, and measured
  scheduler overhead.
- **`scripts/run_os_experiments.sh`** — one command, five sweeps, CSV + JSON
  with seed, cluster, policy config and git commit on every row.
- **`python/examples/gpu_burn.py`** (§74) — a compute-bound demo workload that
  starts instantly, with `--idle` for producing the squatter case on purpose.
- **`docs/os_term_project/experiments.md`** — the write-up, every figure read
  off the generated CSVs.

### Files changed

```
NEW  crates/ferro-sim/                          simulator, generator, metrics, runner
NEW  scripts/run_os_experiments.sh
NEW  python/examples/gpu_burn.py
NEW  docs/os_term_project/experiments.md
NEW  outputs/benchmarks/                        generated results, committed
```

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **199 passed, 0 failed** (was 154) |
| `uv run --all-extras pytest -q` | **19 passed** |
| `cargo fmt --check` / `clippy -D warnings` | ✅ clean |

### Three measurement bugs the experiments exposed

Each was found by running the thing and disbelieving the output.

1. **Starvation was defined as being overtaken.** FIFO scored zero by
   construction and every reordering policy scored badly for reordering. A
   definition that decides the comparison before it runs is not a metric. Now:
   waited more than ten times your own runtime, which is scale-free and
   policy-neutral.
2. **Jain fairness over GPU-seconds was identical under every policy.** On a
   run where the queue drains, what a user received equals what they demanded.
   The index was describing the workload. Per-user waiting time and a
   directional heavy-user ratio were added; fair share moves the latter from
   0.99 to 4.69.
3. **All five placement strategies produced byte-identical output.** Of course
   they did — a job took the same time wherever it landed. The simulator now
   models the two effects that make placement matter, and `--flat-execution`
   keeps the old behaviour as the control proving it.

### Headline results

- **Aging halves what priority costs the queue's tail.** Strict priority makes
  the low-priority user wait 11.72× longer than everyone else; aging brings
  that to 4.62× while still serving urgent work first.
- **Fair share corrects a flooded cluster**: the greedy user's wait rises 33 %,
  the three polite users' fall 3–4×.
- **SJF is worth 4.6× with honest estimates and 1.04× with realistic ones.**
  Its entire advantage is a claim about information quality.
- **Performance-aware placement walks into the slow wire**: 35 of 36 multi-node
  jobs across a pair measured at 90 Mb/s, running 9.3× slow, because it ranks
  by *negotiated* speed and every NIC claims 1000. Topology-aware placement
  avoids it deliberately and gets the best utilisation of the five.
- **FerroGrid already backfills.** `run_queue` walks the whole waiting list, so
  dispatch is opportunistic, not strict FIFO. That is worth 4.9× on mean
  waiting time — and costs an unbounded tail for large jobs, which is what
  reservation would fix.
- **Policy choice costs 0.2 µs per scheduling pass.** Measured, not assumed.

### Gate

The roadmap's gate asked for results covering "FIFO / priority / aging /
fair-share / SJF / backfill". Five of the six are covered. **Backfill is not
implemented** (it is Phase 7), so rather than fabricate a row, the dispatch
sweep measures the opportunistic-versus-strict question FerroGrid actually
poses today. That is a deviation from the stated gate and is recorded as one.

### Known limitations

1. **One seed per scenario.** No confidence intervals. The runner supports
   sweeping `seed`; the script does not yet.
2. **The execution model's constants are calibrated, not derived.** Absolute
   slowdowns are "this scale", not "this figure".
3. **Aging is tuned to the simulated timescales** (30 s / 5 points) rather than
   the controller's defaults (60 s / 1 point), because at the defaults no job
   in these workloads waits long enough for aging to do anything.
4. **Fragmentation is reported but does not yet discriminate.** The workloads
   lack the packing pressure that would separate best-fit from first-fit on it.
5. **No real-cluster validation.** Every number here is simulated. The
   placement path was demonstrated end to end on one GPU in Phase 3, but
   nothing has cross-checked a simulated waiting time against a measured one.

### Next

Phase 5 — persistence. SQLite with WAL for jobs, queue, history, usage,
benchmarks, network measurements and events; ephemeral telemetry stays in
memory. Then Phase 6's reconciliation, which the audit found is already half
done on the wire: agents report running jobs and GPU ownership on every
heartbeat, and the controller currently discards what it does not recognise.

---

## Phase 3 — Placement Algorithms

**Phase:** 3 — Named placement strategies, measured topology, explainability
**Status:** ✅ Complete. Gate met.

### Implemented

**Four new placement strategies**, alongside the existing `performance`, all
selectable with `--placement-policy`:

| Strategy | Prefers | The failure mode it addresses |
|---|---|---|
| `first-fit` | first node in id order, lowest indices | none — it is the baseline the others must beat |
| `best-fit` | the node with least left over | cluster fragmentation: six free GPUs, nowhere to put four |
| `vram` | roomiest cards, judged by the *worst* one | a job OOMing because it was given the tightest fit |
| `topology` | the set whose slowest **measured** hop is fastest | a collective running at the pace of an untested wire |

They share feasibility and plan construction (`placement/engine.rs`) and differ
only in two comparators, so a bug in "what is possible" is fixed once rather
than five times, while each algorithm stays readable on its own.

**`ferro net` results are now kept.** Audit finding §5.3 is closed: measured
pairwise throughput reaches the scheduler through `NetworkSnapshot` on the
scheduling context instead of being printed and discarded.

**A shared scoring vocabulary** (`placement/score.rs`, §13). Every strategy
reports the same five axes — compute, vram, homogeneity, network, load — each
0..1, plus a weighted total with configurable weights. Selection and
description are deliberately separate: strategies disagree about what to
optimise, but must not disagree about how to describe what they chose, or
switching strategy would produce incomparable numbers.

**`ferro explain <job>`** (§46) shows both decisions and the arithmetic behind
each, with `--json`.

### Files changed

```
NEW  crates/ferro-sched/src/topology.rs              NetworkSnapshot
NEW  crates/ferro-sched/src/placement/engine.rs      shared feasibility + assembly
NEW  crates/ferro-sched/src/placement/score.rs       the five-axis vocabulary
NEW  crates/ferro-sched/src/placement/{first_fit,best_fit,vram,topology}.rs
NEW  crates/ferro-sched/tests/strategies_differ.rs   cross-strategy comparison
MOD  crates/ferro-sched/src/{lib,placement/mod}.rs
MOD  crates/ferro-controller/src/{registry,service,main}.rs
MOD  crates/ferro-cli/src/{main,render}.rs           `ferro explain`
MOD  proto/ferrogrid.proto                           PlacementExplanation
MOD  README.md
```

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **154 passed, 0 failed** (was 103) |
| `uv run --all-extras pytest -q` | **19 passed** |
| `cargo fmt --check` / `clippy -D warnings` | ✅ clean |

Every strategy has determinism tests and tests asserting it picks what its own
documentation claims. `strategies_differ.rs` drives all five over one
deliberately awkward cluster and asserts they **disagree** — five strategies
that always agreed would be five names for one policy, and the placement
comparison in Phase 4 would measure nothing.

### Demonstrated end to end

A real controller and agent against this machine's single RTX 3060:

```
Where: placement policy `performance`
  rank 0 -> local-gpu GPU [0]

  compute                    1.00
  vram                       1.00
  homogeneity                1.00
  load                       1.00
                          --------
  placement score            1.00

  Reasons:
  - no GPU here has been benchmarked; ranked by free VRAM instead
  - 5.9 GiB free on the tightest card
  - all 1 GPUs are NVIDIA GeForce RTX 3060 Laptop GPU
  - single node, so no network hop to cross
```

Note what it does *not* claim: the network axis is absent rather than scored
1.0, and the compute reason says plainly that nothing has been benchmarked.

### A bug the tests found

The first implementation let an expired network measurement fall all the way
back to the negotiated link speed. That meant **forgetting made a link look
faster**: a pair measured at 90 Mb/s would quietly become a 1000 Mb/s candidate
a day later, and the scheduler would then pick exactly the path `ferro net` had
been run to expose. Stale evidence is now capped by what was last actually
seen — weaker than fresh evidence, but not weaker than none.

### Known limitations

1. **No `cost` strategy that *selects* by the weighted total.** The scoring
   framework and its weights exist and every decision is scored by them, but no
   policy currently maximises the total; each picks by its own rule. That is a
   small addition on top of the subset search `topology` already uses.
2. **`topology` searches node subsets exhaustively** below 200,000 combinations
   and degrades to per-node ranking above it. Fine for any cluster this is
   meant for; the fallback is not itself topology-aware.
3. **Auto mode is uniform across the new strategies** — one node, as many GPUs
   as it can place, capped — rather than each strategy defining its own. Only
   `performance` retains the original "largest identical-model group" rule.
4. **The score is computed against the cluster as it was at placement time** and
   then frozen on the job. That is deliberate — it explains the decision that
   was made — but it means `ferro explain` on an old job describes an old
   cluster.
5. **Fragmentation is not yet measured** (§47), so the best-fit-versus-first-fit
   argument is still theoretical. That number arrives with the Phase 4 metrics.

### Next

Phase 4 — evaluation: `crates/ferro-sim`, the workload generator, and the
experiment runner. The scheduler core is now importable, pure, deterministic and
policy-plural in both dimensions, which is everything a simulator needs.

---

## Phase 2 — OS Scheduling Algorithms

**Phase:** 2 — Priority, aging, fair share, SJF
**Status:** ✅ Complete. Gate met.

### Implemented

Four new queue policies in `ferro-sched`, all pure, all selectable by name:

| Policy | Orders by | The property it is there to demonstrate |
|---|---|---|
| `priority` | priority desc, then arrival | urgent work gets through a full cluster; the bottom can starve |
| `aging` | `min(base + floor(wait/interval)*increment, ceiling)` | bounded waiting — starvation is impossible |
| `fair-share` | `a*priority + b*wait − c*usage`, each term normalised 0..1 | evens out between users over time |
| `sjf` | shortest declared duration, **unestimated last** | lowest mean wait where estimates exist; long-job starvation |

Supporting work:

- **Protocol** — `SubmitJobRequest` gains `priority`, `estimated_duration_s`
  (both `optional`, so "unset" is distinguishable from a real zero) and
  `project`; `JobSummary` gains those plus a `QueueScore` breakdown. All at new
  field numbers; old clients are unaffected.
- **Usage accounting** — `RegistryInner::usage_snapshot` derives GPU-seconds per
  user from the job records themselves rather than maintaining a parallel
  ledger, so the numbers a scheduling decision used and the numbers a report
  shows cannot drift apart.
- **Jain fairness index** — `ferro_sched::jain_index`, returning `None` rather
  than a number for input it cannot describe (empty, negative, NaN).
- **`ferro queue`** — the waiting list in served order, with the score
  breakdown underneath. `--json` and `-w` supported.
- **`ferro train --priority / --estimated-duration / --project`**; the
  controller gains `--aging-interval-secs`, `--aging-increment`,
  `--aging-ceiling` and the three `--fair-*-weight` knobs. No magic numbers.

### Files changed

```
NEW  crates/ferro-sched/src/queue/{priority,aging,fair_share,sjf}.rs
MOD  crates/ferro-sched/src/queue/{mod,fifo}.rs     QueuedJob, QueueContext, QueueRanking
MOD  crates/ferro-sched/src/lib.rs                  policy registry + QueueTuning
NEW  crates/ferro-controller/tests/queue_policy.rs  registry <-> policy wiring
MOD  crates/ferro-controller/src/{registry,service,main}.rs
MOD  crates/ferro-cli/src/{main,render}.rs          flags + `ferro queue`
MOD  proto/ferrogrid.proto                          new fields, new numbers
MOD  README.md
```

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **103 passed, 0 failed** (was 64; +32 policy, +7 wiring) |
| `uv run --all-extras pytest -q` | **19 passed** |
| `cargo fmt --check` / `clippy -D warnings` | ✅ clean |

The §5 requirements are covered explicitly: a waiting job's priority rises
(`a_waiting_jobs_priority_rises`), a newer high-priority job still overtakes
(`a_newer_high_priority_job_still_overtakes`), an old low-priority job
eventually runs (`an_old_low_priority_job_eventually_runs`), and ties are
deterministic (`ties_break_on_arrival_and_stay_deterministic`). Strict priority
carries the opposite assertion — `a_low_priority_job_never_advances_on_its_own`
— so the baseline aging has to beat is pinned down rather than assumed.

### Demonstrated end to end

A live controller, `--queue-policy aging --aging-interval-secs 1
--aging-increment 5`, with `starved` (base 10) queued two seconds before
`urgent` (base 90):

```
  --- immediately after urgent arrives ---
    #1 urgent   base=90  +aging=0    effective=90
    #2 starved  base=10  +aging=10   effective=20
  --- 16s later ---
    #1 starved  base=10  +aging=90   effective=100
    #2 urgent   base=90  +aging=10   effective=100
```

Anti-starvation on a real system, not only in a unit test.

### Known limitations

1. **No `ferro usage` or `ferro stats fairness` yet.** The data exists and
   fair share consumes it; exposing it over gRPC belongs with the accounting
   work in a later phase.
2. **`--priority` is inert under the default FIFO policy**, by design. The CLI
   help says so, but a cluster running FIFO will silently ignore the flag.
3. **Usage is recomputed from all job records on every ranking pass** — O(jobs)
   per call. Correct and affordable here, and a deliberate choice over a
   running counter that could drift; it will want revisiting once history is
   persisted rather than bounded by process lifetime.
4. **SJF depends on self-reported estimates.** Nothing verifies them, and a
   user who under-declares is rewarded. Recorded rather than papered over:
   this is why SJF ships as an experiment.
5. **Policy is fixed at controller startup.** §67 hot switching is untouched.

### Next

Phase 3 — placement algorithms: refactor the existing composite policy into
named `first-fit`, `best-fit`, `vram-aware`, `performance-aware`,
`homogeneous` and `topology-aware` strategies, persist `ferro net` results so
measured throughput reaches the scheduler, and add the unified cost function
behind `ferro explain`.

---

## Phase 1 — Scheduler Refactor

**Phase:** 1 — Pluggable scheduler architecture, no behaviour change
**Status:** ✅ Complete. Gate met.

### Implemented

1. **Quality gates made green first**, so that everything after this point can
   detect a regression: `cargo fmt` applied across 12 files, and all 17
   pre-existing `clippy -D warnings` lints fixed (`clone_on_copy` ×8,
   `useless_format` ×4, `manual_contains` ×3, `manual_checked_div`,
   `items_after_test_module`). No behaviour touched.

2. **`crates/ferro-sched`** — the scheduling core, extracted as its own crate.
   `scheduler.rs` moved here **verbatim** (git records it as a rename; the only
   edit is lifting `ScheduleError` to the crate root). All 18 placement tests
   moved with it and pass byte-identically unmodified, which is the evidence
   that placement behaviour did not change.

3. **Two policy traits**, separating the two questions:
   - `PlacementPolicy` — "where?" — with `PerformancePlacement` wrapping the
     existing `plan`/`plan_auto` behind `PlacementRequest`/`Shape`.
   - `QueuePolicy` — "who next?" — with `Fifo` replacing the `sort_by_key` that
     used to be buried in `registry.rs::queued_jobs`.
   Both selectable by name (`--queue-policy`, `--placement-policy`), with a
   registry that lists the known policies in its error message.

4. **`crates/ferro-controller/src/lib.rs`** — the crate now has a library
   target. Blocker B1 is resolved: integration tests, the benchmark harness and
   the Phase 4 simulator can link the controller and the scheduler.

5. **Blocker B2 fixed.** `reserve` became `reserve_exact`: an all-or-nothing
   compare-and-swap against real GPU ownership, refusing rather than
   overwriting. `start_job` split into reservation plus `dispatch_ranks`, so the
   queue dispatcher reserves *before* promoting — a queued job that loses a race
   goes back in line instead of failing for something it did not cause.

### Files changed

```
NEW  crates/ferro-sched/{Cargo.toml,src/lib.rs}
NEW  crates/ferro-sched/src/placement/mod.rs        trait + PerformancePlacement
MOV  crates/ferro-controller/src/scheduler.rs
       -> crates/ferro-sched/src/placement/gpu.rs   verbatim (rename, -14 lines)
NEW  crates/ferro-sched/src/queue/{mod.rs,fifo.rs}  trait + FIFO
NEW  crates/ferro-sched/tests/purity.rs             enforces the purity rule
NEW  crates/ferro-controller/src/lib.rs             library target
NEW  crates/ferro-controller/tests/allocation.rs    allocation invariants
MOD  crates/ferro-controller/src/{main,service,registry}.rs
MOD  Cargo.toml, crates/ferro-controller/Cargo.toml
FMT  crates/ferro-{agent,cli,gpu,controller}/...    rustfmt + clippy only
```

### Tests

| Check | Result |
|---|---|
| `cargo test --workspace` | **64 passed, 0 failed** (was 54; +4 FIFO, +4 allocation, +2 purity) |
| `uv run --all-extras pytest -q` | **19 passed** |
| `cargo fmt --check` | ✅ clean (was 32 hunks) |
| `cargo clippy --workspace --all-targets -- -D warnings` | ✅ clean (was 17 errors) |
| 18 moved placement tests | pass **unmodified** |
| `only_capacity_errors_are_retryable` | passes **unmodified** |

**The concurrency test was verified to have teeth.** With `reserve_exact`
temporarily reverted to the old unconditional overwrite, it fails loudly:

```
a GPU was granted to more than one job:
  [("gpu-a", 0), ("gpu-a", 0), ("gpu-a", 1), ("gpu-a", 0), ... ]
  left: 2   right: 16
```

Sixteen concurrent submissions, four GPUs, and GPU 0 handed to fifteen jobs at
once. That is the §5.2 defect reproduced, and it passes with the fix restored.

### Benchmark

Still none, and still for the reason recorded in Phase 0: no cluster is
reachable. But B1 is now resolved, so a scheduler micro-benchmark and the
offline simulator have somewhere to live. That is Phase 4's work.

### Design revisions made during implementation

- **Queue ordering does not see the hardware.** The roadmap originally gave both
  traits one shared `SchedulingContext`. Implementation showed that a queue
  policy needs only the jobs and the clock, and that letting it see free
  capacity would let `queue_position` disagree with the order the dispatcher
  actually serves. `QueueContext { now }` is therefore separate and deliberately
  narrow. Backfilling is not a counter-example: it fills idle capacity from the
  *already ranked* order, which is a dispatcher concern.
- **The queue policy lives in `RegistryInner`**, not beside the dispatcher, so
  the position a user is quoted and the order they are served in are computed by
  the same object.
- **Purity is enforced, not just documented.** `ferro-sched/tests/purity.rs`
  scans the crate for `async fn`, `.await`, clock reads and `std::{fs,net,process}`,
  and checks the manifest for runtime dependencies. The rule is load-bearing for
  the whole evaluation plan, so it should fail a build rather than erode quietly.

### Known limitations

1. **A lost race still fails a non-queued submission.** `--wait` jobs return to
   the queue, but a plain `ferro train` that loses a reservation race gets a
   clear error rather than an automatic re-plan. A bounded retry is a small
   Phase 3 addition; today's behaviour is no worse than before (the agent
   rejected the loser anyway) and is now explained properly.
2. **`tokio::sync::Mutex` is still the registry's lock** although no path awaits
   while holding it. `std::sync::Mutex` would be the honest type. Deliberately
   left as a separate mechanical commit.
3. **Only one policy of each kind exists.** `--queue-policy` accepts `fifo` and
   `--placement-policy` accepts `performance`. The seam is real, the menu is not
   yet — that is Phase 2 and 3.
4. **Citations in `current_state.md` are as of `4d1d2f4`** and some have shifted.
   That document is a point-in-time audit and is not rewritten as code moves.

### Next

Phase 2 — OS scheduling algorithms, in order: `priority`, then `aging`, then
`fair-share`, then experimental `sjf`. First step is the proto change
(`priority`, `estimated_duration_s`, `project` at new field numbers, with
"unspecified" distinguishable from a real zero) plus the matching `ferro train`
flags. Each policy ships with the §5 unit tests: a waiting job's priority rises,
a newer high-priority job overtakes, an old low-priority job eventually runs, and
ties stay deterministic.

---

## Phase 0 — Baseline Audit

**Phase:** 0 — Baseline Audit
**Status:** ✅ Complete. Gate met.

### Implemented

No production code was changed. Phase 0 is an audit, and its deliverables are
documents plus a measured baseline.

- Full read of all 8,428 lines of Rust, `proto/ferrogrid.proto`, the Python
  examples, the deployment scripts and the README.
- Feature matrix of all 93 specification sections, each marked DONE / PARTIAL /
  MISSING against verified source evidence.
- Minimum architecture refactor proposed (`roadmap.md` §2): a new pure,
  synchronous `ferro-sched` crate holding `QueuePolicy` and `PlacementPolicy`
  traits, with time injected rather than read.
- OS concept mapping written against the real code, with unimplemented rows
  labelled as such.

### Files changed

```
docs/os_term_project/current_state.md   (new)
docs/os_term_project/roadmap.md         (new)
docs/os_term_project/os_mapping.md      (new)
docs/os_term_project/progress.md        (new, this file)
```

No file under `crates/`, `proto/`, `python/` or `scripts/` was modified.

### Tests

Baseline measured on commit `4d1d2f4`, 2026-09-22, Rust 1.98:

| Check | Result |
|---|---|
| `cargo test --workspace` | **54 passed, 0 failed** — exit 0 (37 controller · 15 agent · 2 CLI) |
| `uv run --all-extras pytest -q` | **19 passed, 1 warning** — exit 0 |
| `cargo fmt --check` | ❌ 32 diff hunks across 8 files — exit 1 |
| `cargo clippy --workspace --all-targets -- -D warnings` | ❌ 17 lint errors across 3 crates — exit 101 |

**Gate ("all existing tests PASS"): met.** The §80 quality bar is not yet met;
see Known Limitations.

### Benchmark

**None established, and this is a finding rather than an omission.**

- No GPU cluster is reachable from the development machine (one RTX 3060 Laptop,
  6 GiB, WSL2; no controller or agent running), so no live scheduling baseline
  could be taken.
- No scheduler micro-benchmark could be written either, because
  `ferro-controller` is a binary-only crate and its scheduler cannot be imported
  from a test or benchmark target (`current_state.md` §5.1).

The measurable Phase 0 baseline is therefore the test and lint table above. A
real scheduling baseline becomes possible in Phase 4, once `ferro-sched` exists
and the simulator can drive it without hardware.

### Known limitations

Carried into Phase 1 as work items, in priority order:

1. **`ferro-controller` is not importable** (no `lib.rs`). Blocks the simulator,
   the benchmark harness and every integration test. Must be fixed first.
2. **Plan and reserve are not atomic.** `submit_job` reads the cluster, plans and
   reserves in three separate critical sections, and `reserve` overwrites
   `allocated_job_id` unconditionally (`registry.rs:482`). The agent's launch-time
   re-validation prevents two jobs actually running on one card, but the
   controller's ledger loses writes: it releases the wrong job's GPUs and reports
   running cards as free for up to one heartbeat. No existing test would catch it.
3. **Lint and format gates are red** — 32 `rustfmt` hunks, 17 `clippy` lints. All
   mechanical (`clone_on_copy`, `useless_format`, `manual_contains`,
   `manual_checked_div`, `items_after_test_module`), none are bugs, but a red
   gate cannot detect a regression.
4. **`ferro net` measurements are discarded** — pairwise throughput is printed to
   the CLI and never stored, so topology-aware placement currently rests on the
   *negotiated* link speed only.
5. **Container uid is the agent's, not the submitter's**
   (`ferro-agent/src/launcher.rs:401-405`). Jobs are isolated from the host and
   from each other's devices and mounts, but every FerroGrid job on a node runs
   as the same account. §61 must describe this honestly.
6. **README is stale** — "Scope and limitations" still claims there is no
   queueing, but `--wait` has existed since commit `7363af6`.

### Independent review

The proposed refactor was reviewed by an external model instructed to disagree.
One objection was **accepted and changed the design**: putting placement inside
the registry mutex fixes the concurrency defect by serialising the dispatcher,
which is the wrong direction. The fix is now an all-or-nothing `reserve_exact`
compare-and-swap with caller-side re-plan, leaving placement outside the lock
(`roadmap.md` §2.4). Two further objections — allowing async policies, and making
queue policies stateful — were **rejected** for reasons recorded in
`roadmap.md` §2.6, chiefly that both would break the single-scheduler-core
requirement or create a second source of truth for usage accounting.

Two external audit passes over the CLI and agent crates were also run. Their
factual claims were re-verified against the source before being used; one was
wrong (`bench` and `net` *do* honour `--json`, `main.rs:392,431`) and was
discarded.

### Notable positive findings

Worth recording so that later phases do not rebuild what works:

- The existing placement heuristic is good: measured-TFLOP/s ranking, GPU-model
  homogeneity *within and across* nodes, network-first ordering for multi-node
  jobs, and a total tiebreak order so decisions are reproducible.
- `node_verdicts` already produces a per-node explanation ledger, refreshed on
  every queue tick. §46 is an extension of an idea the codebase already holds.
- **Agents already report running jobs and GPU ownership on every heartbeat**
  (`ferro-agent/src/main.rs:148-152`), and the controller discards reports for
  jobs it does not know (`registry.rs:432-434`). The "actual state" half of
  §21's reconciliation is already on the wire — Phase 6 is cheaper than it looks.

### Next

Phase 1 — scheduler refactor, no behaviour change. In order:

1. Clean `cargo fmt` and `cargo clippy -D warnings` so the §80 gate goes green
   and can detect regressions afterwards.
2. Add `crates/ferro-sched` (pure, sync) and `ferro-controller/src/lib.rs`.
3. Move `plan`/`plan_auto` verbatim behind `PlacementPolicy`; move the
   `job_order` sort behind `QueuePolicy`.
4. Replace `reserve` with an all-or-nothing `reserve_exact` that refuses if any
   target GPU is already spoken for, and have the caller re-plan on conflict.
   (Placement deliberately stays *outside* the lock — see `roadmap.md` §2.4/§2.6.)
5. Add the concurrency test that fails before step 4 and passes after, plus the
   invariant test that a terminal job owns no allocations.

**Phase 1 gate:** all 54 existing tests pass *unmodified*, `fmt` and `clippy`
green, new concurrency and invariant tests pass, and `ferro-sched` has no tokio
or gRPC-client dependency.
