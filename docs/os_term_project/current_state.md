# FerroGrid — Phase 0 Baseline Audit

**Date:** 2026-09-22
**Commit audited:** `4d1d2f4` ("Make placement legible, and per-node images first-class")
**Auditor:** repository read of every Rust source file, `proto/ferrogrid.proto`,
the Python examples, the scripts and the README, plus a full test/lint run.

> **Line numbers in this document are as of `4d1d2f4`.** This is a point-in-time
> audit and is deliberately not rewritten as the code moves: it records what was
> found. Phase 1 has since reformatted every crate, moved `scheduler.rs` to
> `crates/ferro-sched/src/placement/gpu.rs`, and replaced `reserve` with
> `reserve_exact`, so several cited sites have shifted or no longer exist.
> [`progress.md`](progress.md) tracks what has changed since.

This document records **what exists today**, verified against the source. It does
not propose work; the plan lives in [`roadmap.md`](roadmap.md).

---

## 0. Scale of the thing being audited

| Component | Files | Lines |
|---|---|---|
| `crates/ferro-controller` | 6 | 3,456 |
| `crates/ferro-cli` | 2 | 2,479 |
| `crates/ferro-agent` | 7 | 2,177 |
| `crates/ferro-gpu` | 1 | 280 |
| `crates/ferro-proto` | 2 | 36 |
| `proto/ferrogrid.proto` | 1 | ~470 |
| Python (`python/`, `tests/`) | 12 | 2,409 |
| **Rust total** | **18** | **8,428** |

Small enough that the whole control plane fits in one reading. That is a feature
worth protecting: the project's stated character is *small, lightweight,
understandable, testable, measurable*.

---

## 1. Baseline measurements (the Phase 0 gate)

Run on 2026-09-22, WSL2, Rust 1.98, on commit `4d1d2f4`.

| Check | Command | Result |
|---|---|---|
| Rust tests | `cargo test --workspace` | **54 passed, 0 failed** (exit 0) — 37 `ferro-controller`, 15 `ferro-agent`, 2 `ferro-cli` |
| Python tests | `uv run --all-extras pytest -q` | **19 passed, 1 warning** (exit 0) |
| Formatting | `cargo fmt --check` | ❌ **FAILS** — 32 diff hunks across 8 files (exit 1) |
| Lints | `cargo clippy --workspace --all-targets -- -D warnings` | ❌ **FAILS** — 17 lint errors across 3 crates (exit 101) |

**The Phase 0 gate ("all existing tests PASS") is met. The §80 quality bar is
not.** The lint failures are all mechanical and pre-existing — they are newer
`clippy` lints firing on code written against an older toolchain, not latent
bugs:

| Lint | Count | Where |
|---|---|---|
| `clone_on_copy` (`TrainingMetrics` / `Option<TrainingMetrics>` are `Copy`) | 8 | `ferro-cli/src/render.rs` ×7, `ferro-controller/src/registry.rs:119` |
| `useless_format` (`format!("{}", x)`) | 4 | `ferro-controller/src/service.rs:1230,1257,1271`, `ferro-cli/src/render.rs:530` |
| `manual_contains` (`contains()` vs `iter().any()`) | 3 | `ferro-controller/src/registry.rs:103,105,109` |
| `manual_checked_div` | 1 | `ferro-controller/src/service.rs:547` |
| `items_after_test_module` | 1 | `ferro-agent/src/service.rs:236` |

`cargo fmt` diffs are confined to `ferro-agent` (bench, launcher, main, net,
service, state), `ferro-gpu/src/lib.rs`, `ferro-controller` (main, metrics,
plugins, service) and one hunk in `ferro-cli/src/render.rs`.

### What could not be measured

No GPU cluster is reachable from this machine (one RTX 3060 Laptop, 6 GiB, under
WSL2; no `ferro-controller` or `ferro-agent` process running). Therefore:

- **No live scheduling baseline exists.** Throughput figures in the README
  (§"Measured results") were taken on real lab hardware over 1 GbE and are
  historical, not reproducible here.
- **No scheduler micro-benchmark can be written yet** — see §4.1 below; the
  scheduler is not importable from outside its own binary.

The honest Phase 0 baseline is therefore the table above: test counts, lint
counts, and the structural findings in §4.

---

## 2. Already implemented (verified in source)

### 2.1 Cluster membership and telemetry

| Capability | Evidence |
|---|---|
| Node registration, agent → controller | `service.rs:27` `register_node`; `registry.rs:175` `upsert_node` |
| Heartbeat with GPU + process payload | `service.rs:55`; `registry.rs:187` `heartbeat` |
| Binary health from heartbeat age | `registry.rs:33` `Node::healthy()`, `HEARTBEAT_TIMEOUT_S = 15` (`registry.rs:15`) |
| NVML GPU discovery (name, VRAM, util, temp, power, CUDA capability, UUID) | `crates/ferro-gpu/src/lib.rs`; `proto` message `Gpu` |
| Per-process GPU discovery incl. **foreign** processes | `ferro-agent/src/procs.rs` (681 lines); `proto` `GpuProcess`, `GpuOccupant` |
| Process attribution (cgroup + cached `docker ps`, or pid ancestry for `--no-docker`) | `procs.rs` |
| Credential redaction before a command line leaves the node | `procs::redact_secrets` |
| Sampled per-process utilisation with explicit *unknown* (`utilization_known`) | `proto` `GpuProcess.utilization_pct/utilization_known/busy_unix_s` |
| Measured GPU throughput (`ferro bench`, bf16 TFLOP/s), cached by UUID | `ferro-agent/src/bench.rs`; `registry.rs:231` `record_benchmarks` |
| Measured pairwise network throughput (`ferro net`), one pair at a time | `service.rs:459` `measure_network`, `829` `measure_pair`; `ferro-agent/src/net.rs` |
| Negotiated link speed per node (`link_iface`, `link_mbps`) | `proto` `NodeInfo:15,16` |

### 2.2 Job lifecycle

| Capability | Evidence |
|---|---|
| Submit → plan → reserve → dispatch to every rank | `service.rs:88` `submit_job`, `1043` `start_job` |
| Rank-0-first launch ordering (rendezvous host up before peers) | `service.rs:1062-1065` |
| Partial-launch rollback (stop already-started ranks) | `service.rs:1096-1110` |
| Phase aggregation from per-rank votes | `registry.rs:92` `Job::phase()` |
| Queued jobs are `Pending`, not falsely `Succeeded` | `registry.rs:95-98` (explicit special case + test) |
| Log streaming with backlog replay, lag-tolerant | `service.rs:716` `stream_logs` |
| Metric ingestion from `FERRO_METRIC {json}` on stdout | `metrics.rs:38` `parse_metric_line` |
| NCCL error classification (deliberately excludes bare `ProcessGroupNCCL`) | `metrics.rs:16-33` (`NCCL_ERROR_MARKERS`) |
| Job cancellation | `service.rs:665` `cancel_job` |
| **Peer teardown on rank failure** (survivors would otherwise hold GPUs forever) | `service.rs:773` `report_job_status`, survivors at `:790` |
| Wall-clock job timeout (`--timeout`), reaped every 15 s | `registry.rs:378` `expired_jobs`; `service.rs:1190` `reap_expired` |
| Docker isolation, GPU pinning, explicit bind mounts | `ferro-agent/src/launcher.rs` |
| Per-node image overrides (`--image-for`) | `proto` `SubmitJobRequest.node_images:16`; `service.rs:915,929` |

### 2.3 Queueing (exists — the README is stale on this point)

A FIFO waiting queue **is implemented**, contrary to README §"Scope and
limitations" which still claims "No queueing".

| Capability | Evidence |
|---|---|
| `--wait [give-up-after]` queues instead of failing | `proto` `SubmitJobRequest.queue:14, queue_timeout_s:15` |
| FIFO order by `job_order`, **not** by timestamp | `registry.rs:286` `queued_jobs`, FIFO comment at `:298-301` + comment |
| Stable 1-based queue position | `registry.rs:367` `queue_position` |
| Dispatcher re-reads the cluster per job per tick (5 s) | `service.rs:1127` `run_queue` |
| Queue deadline expiry | `service.rs:1132-1136` |
| Cancel while queued (dequeue, never launched) | `registry.rs:342` `dequeue`; `service.rs:670` |
| Live per-node verdict refresh while queued | `registry.rs:325` `update_queue_assessment` |
| Only *capacity* errors retry; shape errors fail immediately | `service.rs:908` `retryable_schedule_error` |

### 2.3b What already survives a controller restart (better than the README claims)

This matters disproportionately for Phase 6, so it is worth stating precisely.

The agent's heartbeat loop reconnects forever and **re-registers on every
reconnect**, and the controller's `HeartbeatResponse.known = false` explicitly
tells an agent to re-register (`ferro-agent/src/main.rs:123-168`). More
importantly, every heartbeat already carries two things a recovering controller
needs:

- `HeartbeatRequest.jobs` — the status of every job the agent is *currently
  running* (`main.rs:148-152`, `state.rs:149` `job_statuses`).
- `HeartbeatRequest.gpus` — each GPU's `allocated_job_id`, derived from the
  agent's own allocation table (`state.rs:96` `allocations`, `:110`
  `gpu_snapshot`).

Consequences, verified by reading the handlers:

| After a controller restart | What happens today |
|---|---|
| Node registrations | **Recover automatically** — agents re-register within ~3 s |
| GPU allocations | **Recover automatically** — the controller stores `node.info.gpus` wholesale (`registry.rs:199`), so `allocated_job_id` comes back from the agent. The cards correctly read as not schedulable |
| Job records | **Lost.** `update_job_status` early-returns for an unknown `job_id` (`registry.rs:432-434`), so the agent's reports are silently discarded |
| The running job itself | **Keeps running**, untracked: no `ferro job`, no `ferro logs`, and `ferro cancel` returns `not_found` (`service.rs:687`). Only the agent can still stop it |

So the "actual state" half of §21's reconciliation is **already flowing over the
wire**; what is missing is the "desired state" half (durable job records) and
somewhere to put what arrives. That makes Phase 6 considerably cheaper than a
blank-sheet reading of the spec suggests — and it means the honest description
of today's behaviour is *"jobs become unmanageable"*, not *"GPUs leak"*.

### 2.4 Placement policy (a single, already fairly sophisticated function)

`scheduler.rs` implements one composite placement policy:

1. **Feasibility.** Healthy node, inside `--node` filter, GPU unallocated **and**
   ≥ `min_free_vram_b` free (`free_gpus`, `scheduler.rs:46`).
2. **Homogeneity, within a node.** Group free cards by model, only offer groups
   of ≥ `want` (`homogeneous_options`, `:76`). Mixed sets are a fallback only
   (`best_selection`, `:124`).
3. **Homogeneity, across nodes.** Build one candidate placement per target GPU
   model plus a baseline, score them, keep the best (`plan`, `:559`;
   `placement_choice`, `:316`). A target model is a *preference*, never a
   requirement.
4. **Measured performance.** Rank by benchmarked TFLOP/s; unbenchmarked cards
   fall back to a free-VRAM proxy scaled so any real measurement outranks them
   (`score`, `:140`).
5. **Network.** For multi-node jobs only, negotiated `link_mbps` is compared
   *before* GPU score, and choices are compared on min-link then total-link
   (`node_choice_order`, `:278`; `choice_order`, `:391`).
6. **Determinism.** Every comparison chain ends in a total order — model name,
   GPU index, node id (`compare_selection`, `:113`; `choice_order` tail).
7. **Auto shape selection.** `plan_auto` (`:487`) keeps a job on **one** node and
   takes the largest identical-model group, because on this cluster crossing the
   network costs ~55× and sharding a fitting model costs ~3×.

### 2.5 Explainability (partially there, and genuinely good)

`node_verdicts` (`scheduler.rs:154`) produces a **per-node** ledger — eligible
yes/no, the reasons each filter fired, free GPU count, free VRAM — surfaced in
`SubmitJobResponse.node_verdicts` and `JobSummary.node_verdicts`, and refreshed
on every queue tick. `compatibility_warnings` (`service.rs:943`) adds driver-age
and mixed-compute-capability warnings as *warnings, not gates*.

This is the seed of §46 "Scheduling explainability" and is a real asset: the
abstraction to build is one the codebase already believes in.

### 2.6 CLI surface (the backward-compatibility contract)

Global: `--controller <URL>` (env `FERRO_CONTROLLER`), `--json`.
Shared `WatchArgs` on read-only views: `-w/--watch`, `-n/--interval <SECONDS>`.

| Command | Flags |
|---|---|
| `ferro nodes` | `WatchArgs` |
| `ferro gpu` | `WatchArgs` |
| `ferro ps [PID]` | `--idle [FOR]`, `--by-user`, `WatchArgs` |
| `ferro watch` | `-n/--interval` |
| `ferro jobs` | `--limit` (20), `WatchArgs` |
| `ferro job <ID>` | `WatchArgs` |
| `ferro logs <ID>` | `-f/--follow` |
| `ferro cancel <ID>` | — |
| `ferro train <SCRIPT> [ARGS…]` | `--nodes`, `--gpus-per-node`, `--auto`, `--image`, `--image-for`, `--workdir`, `--env`, `--node`, `--mount`, `--name`, `-f/--follow`, `--sync`, `--timeout`, `--wait [GIVE-UP-AFTER]` |
| `ferro bench` | `--node`, `--force` |
| `ferro net` | `--node`, `-s/--seconds`, `--both-ways` |
| `ferro sync [PATH]` | `--node`, `--delete`, `--dry-run` |
| `ferro plugins` / `fetch` / `push` | plugin/remote/local positional, `--node`, `--timeout` |

Rendering is frame-based: renderers build a frame and return it, only `main`
prints, and `Screen` overwrites the previous frame in one write rather than
clearing first (`render.rs`). Data age is surfaced alongside numbers throughout.

`--json` is honoured by `nodes`, `gpu`, `ps`, `net`, `plugins`, `fetch`, `push`,
`bench`, `jobs`, `job` and `train`. It is **not** honoured by `sync`, `watch`,
`logs` or `cancel` (`main.rs:293-489`); `--watch` and `--json` are explicitly
rejected together (`main.rs:610-611`). For §63, only `sync` and `cancel` are real
gaps — `watch` and `logs` are streaming views where JSON has no obvious meaning.

---

## 3. Partially implemented

| # | Area | What exists | What is missing |
|---|---|---|---|
| P1 | **Queue policy** | FIFO, deterministic, `job_order`-based | Not a policy — it is a `sort_by_key` in `registry.rs:286`. No abstraction, no alternative, no priority field |
| P2 | **Placement policy** | One composite function, high quality | Not swappable. Cannot express first-fit / best-fit; no weight configuration; no fragmentation awareness |
| P3 | **Explainability** | Per-node *feasibility* verdicts + warnings | No *scores*. The user is told why a node was excluded, never why the chosen one won. No `ferro explain` |
| P4 | **Topology awareness** | Negotiated `link_mbps` used for multi-node ordering; `ferro net` measures real pairwise throughput | Measured `ferro net` results are **not stored** and **not used by the scheduler** — they are printed and discarded (`service.rs:459` returns them to the CLI only) |
| P5 | **Fault detection** | Binary healthy/unhealthy at 15 s | No `Suspected`/`Dead`/`Recovering` states; threshold is a `const`, not configurable |
| P6 | **Node failure handling** | Rank failure tears down peers and releases GPUs (`service.rs:787`) | A node that simply *disappears* (no rank ever reports) leaves its job running forever: nothing watches `healthy()` transitions |
| P7 | **Job timeout** | Implemented and reaped | Terminal phase is `Cancelled`; no distinction from a user cancel |
| P8 | **Queue timeout** | Implemented | Expiry is recorded as `JobPhase::Failed` (`service.rs:1135`) — §28 asks for a distinct `EXPIRED` |
| P9 | **Cancellation** | Works | **Immediate SIGKILL**: `docker kill` then `child.start_kill()` (`ferro-agent/src/state.rs:169-179`). No SIGTERM → grace → SIGKILL |
| P10 | **Accounting** | `submitted_by` is recorded per job; rolling GPU-util average per job (`registry.rs:143` `record_util`) | No per-user aggregation, no GPU-seconds, no history, no `ferro usage` |
| P11 | **Node capacity** | `cpu_count`, `memory_total_b` are advertised (`proto` `NodeInfo:8,9`) | Never used by the scheduler, never requested by a job |

---

## 4. Missing entirely

Verified by repository-wide search: **zero occurrences** of `priority`,
`fair.?share`, `quota`, `sqlite`/`rusqlite`, `event.?log`, `backfill`,
`preempt`, or `drain` anywhere in `*.rs`, `*.proto` or `*.toml` (the only
matches are the English word "priority" in two comments and "drains" describing
a TCP sink).

- Queue policies: priority, aging, fair-share, SJF, DRF
- Placement policies: first-fit, best-fit as *named, selectable* strategies
- Scheduling cost function with configurable weights; `ferro explain`
- Multi-resource requests (CPU, RAM, min-VRAM) and multi-resource placement
- Admission control as a distinct stage (today: two ad-hoc validations inside `submit_job`)
- User quota, per-user/per-project accounting, `ferro usage` / `ferro stats`
- Persistent state (SQLite/WAL), controller restart recovery, reconciliation
- Event log and `ferro events`; audit log
- Orphan / GPU-leak detection as an explicit, reported state
- Backfilling, reservation, gang-scheduling as a *named* guarantee, preemption, checkpointing
- Retry policy, OOM detection, adaptive scheduling feedback
- Power-/thermal-aware placement; GPU health states; `drain`/`undrain`
- Job dependencies
- Scheduler metrics (waiting time, turnaround, makespan, fairness, starvation, scheduler latency); Prometheus export
- **Scheduler simulator, workload generator, benchmark harness, experiment output**
- Authentication, TLS, RBAC
- `ferro queue`, `ferro usage`, `ferro stats`, `ferro events`, `ferro explain`, `ferro quota`, `ferro node drain`

---

## 5. Technical debt and defects found

These are ordered by how much they block the roadmap.

### 5.1 `ferro-controller` is a binary-only crate — **structural blocker**

`crates/ferro-controller/` declares `[[bin]]` with `path = "src/main.rs"` and has
**no `lib.rs`**. Every module (`scheduler`, `registry`, `service`, `metrics`) is
private to that binary.

Consequences, all of which the roadmap depends on:

- No integration tests can import the scheduler (`crates/*/tests/` does not exist
  anywhere in the workspace — all 54 tests are inline `#[cfg(test)] mod tests`).
- **No scheduler simulator is possible** (§53) — it could not link the policy code.
- **No benchmark harness is possible** (§51, §52) — same reason.
- No `criterion`-style micro-benchmark of scheduler latency (§48).

This must be fixed before anything in Phase 2 onward can be evaluated, and it is
cheap: add `src/lib.rs` re-exporting the modules and point the binary at it.

### 5.2 Plan and reserve are not atomic — lost update on the allocation table

`submit_job` reads the cluster, plans, and reserves in three separate critical
sections:

```
service.rs:97    let nodes = self.registry.node_states().await;   // lock → clone → unlock
service.rs:98    let (plan_result, …) = plan_for(&nodes, …);      // NO LOCK HELD
service.rs:1060  registry.reserve(plan, job_id).await;            // lock → mark → unlock
```

and `reserve` overwrites unconditionally:

```rust
// registry.rs:482
if p.gpu_indices.contains(&gpu.index) {
    gpu.allocated_job_id = job_id.to_string();   // no check that it is empty
}
```

Two concurrent `SubmitJob` RPCs (tonic serves each in its own task; nothing
serialises them) can both observe GPU 0 as free and both plan onto it.

**What this does *not* cause.** Two jobs do not end up running on the same GPU.
The agent re-validates every launch against its **own** allocation table and
rejects a conflict outright:

```rust
// ferro-agent/src/service.rs:69-73
// Re-validate the controller's placement locally. The controller's view
// is up to one heartbeat stale, so this is the authoritative check.
let busy = self.state.busy_gpus().await;
if let Some(conflict) = req.gpu_indices.iter().find(|i| busy.contains(i)) { … }
```

plus a GPU-UUID re-check before that (`service.rs:57-67`). The agent, not the
controller, is the real allocator.

**What it does cause**, which is still a genuine defect:

1. **Lost update.** Job B's `reserve` overwrites job A's ownership of A's cards
   in the controller's table.
2. **Wrong release.** B's launch is then rejected by the agent, `start_job`
   marks B failed and calls `release_if_done(B)`, which clears every GPU whose
   `allocated_job_id == "B"` — those are **A's cards**. The controller now
   reports A's running GPUs as free.
3. **Cascading spurious failures.** A third submission can be planned onto those
   phantom-free cards and will also be rejected by the agent.
4. **Self-healing, slowly.** The next heartbeat overwrites `node.info.gpus`
   wholesale from the agent's authoritative table (`registry.rs:199`), so the
   window closes after up to one heartbeat interval (~3 s default).

So the controller's allocation table is **soft state**: an optimistic hint that
exists to stop two submissions in the same instant from planning onto the same
device, reconciled from the agent every few seconds. It currently fails at the
one job it has. The comment at `service.rs:1058-1059` ("Reserve before
dispatching, so a second submission racing this one sees the GPUs as taken")
states an intent the code does not achieve.

This is §36 (atomic allocation) and §37 (concurrency safety), and **no existing
test would catch it**. Framing matters for the fix: the goal is not to make the
controller the authority — the agent should stay authoritative — but to make
*plan-and-reserve* a single critical section and to make `reserve` refuse a
GPU that is already spoken for instead of silently taking it.

### 5.3 `ferro net` measurements are discarded

`MeasureNetwork` returns pairwise Mbps straight to the CLI. Nothing persists them
and the scheduler cannot see them, so §12 (topology-aware placement) currently
rests on the *negotiated* link speed only — which, as CLAUDE.md records, "covers
the node-to-switch hop only, and a 1000 Mb/s NIC with zero errors can still sit
behind a 100 Mb/s path". The measured number is the one worth scheduling on.

### 5.4 Lint and format gates are red

32 `rustfmt` hunks and 17 `clippy -D warnings` errors (itemised in §1). None are
bugs, but a red gate cannot detect a regression, so §80 cannot be enforced until
this is cleaned once.

### 5.5 README is stale

"Scope and limitations" still says *"No queueing — a job that cannot be placed is
rejected rather than held"*. `--wait` has existed since commit `7363af6`.

### 5.6 Smaller items

- `HEARTBEAT_TIMEOUT_S` is a `const` (`registry.rs:14`), not configurable (§23, §66).
- Queue tick (5 s) and reap tick (15 s) are hard-coded literals in `service.rs:1128,1191`.
- Admission checks are scattered: `script.is_empty()` raises a gRPC `Status`
  (`service.rs:93`) while image-override errors return a *typed non-accepted
  response* (`service.rs:103`). Two different failure channels for the same class
  of problem.
- `Job::phase()` recomputes from a `HashMap` on every call, including inside
  `live_job_ids()` which runs per `gpu_entries()` call.
- A `Job` holds an unbounded-ish `VecDeque` of 20,000 log lines **per job**, in
  memory, forever — with no persistence this is also the only copy.
- Containers run as the agent's uid/gid rather than the submitter's
  (`ferro-agent/src/launcher.rs:401-405`). Jobs are isolated from the host and
  from each other's devices and mounts, but **not from each other by user** —
  every FerroGrid job on a node runs as the same account. §61 needs this stated
  honestly rather than implied otherwise.

---

### 5.7 Operational findings from the agent side

Verified first-hand after an external audit pass flagged them.

- **Job logs are silently dropped while the controller is unreachable.** The
  agent batches log lines and, if it cannot connect, calls `batch.clear()` and
  returns (`ferro-agent/src/launcher.rs:178-181`). Combined with the fact that
  the controller's 20,000-line ring is the only copy (§5.6), a controller
  restart loses output permanently. Phase 5 persistence should account for this.
- **No CPU or RAM limit is placed on a container.** The `docker run` argv sets
  `--shm-size 8g` and `--ulimit memlock=-1` and nothing else — no `--memory`,
  `--cpus` or `--cpu-shares` (`launcher.rs:378-399`). A job can therefore starve
  a node of RAM while its GPU accounting looks healthy, which is exactly the
  failure mode §16 describes.
- **Containers run with `--network host --ipc host`** (`launcher.rs:383-387`).
  Both are load-bearing — host networking keeps NCCL and the rendezvous port
  reachable without publishing a port range per job — but they mean network and
  IPC namespaces are *not* isolated. §61 must state this rather than imply
  container isolation is complete.
- **CPU and RAM capacity are computed** from `std::thread::available_parallelism()`
  and `/proc/meminfo` `MemTotal` (`ferro-agent/src/state.rs:129-132`, `:246`) and
  sent in `NodeInfo` at registration — then never read by anything. The
  advertisement side of §16 is already done.
- **Agent reconnect uses a fixed 3-second delay, no backoff**
  (`ferro-agent/src/main.rs:143,173`). Harmless at this cluster size; worth
  noting before §23's graded failure detection is designed on top of it.

## 6. Feature matrix against the term-project specification

`DONE` = implemented and tested · `PARTIAL` = present but not in the required
form · `MISSING` = no code exists.

| § | Feature | Status | Note |
|---|---|---|---|
| 1 | Layered controller architecture | PARTIAL | Components exist, not named or separated |
| 2 | Pluggable scheduler (traits) | MISSING | Single function; no trait |
| 3 | FIFO baseline | PARTIAL | Behaviour exists, not as a selectable policy |
| 4 | Priority scheduling | MISSING | No `priority` field anywhere |
| 5 | Priority aging | MISSING | |
| 6 | Fair share | MISSING | `submitted_by` exists as raw material |
| 7 | Jain fairness index | MISSING | |
| 8 | SJF / estimated runtime | MISSING | |
| 9 | Placement: first-fit / best-fit / VRAM-aware | PARTIAL | VRAM floor is enforced; fit strategies unnamed |
| 10 | Performance-aware placement | **DONE** | `ferro bench` → TFLOP/s → `score()` |
| 11 | Homogeneous GPU scheduling | **DONE** | Within *and* across nodes; mixed-capability warning exists |
| 12 | Topology-aware placement | PARTIAL | Uses negotiated link; ignores measured `ferro net` |
| 13 | Unified cost function + `ferro explain` | MISSING | Scores are computed but never surfaced |
| 14 | Extended JobSpec | PARTIAL | nodes/gpus/image/mounts/timeout/queue exist; priority/vram/duration/cpu/memory do not |
| 15 | Admission control | PARTIAL | Ad-hoc, two inconsistent channels |
| 16 | Multi-resource scheduling | MISSING | CPU/RAM advertised, never used |
| 17 | DRF | MISSING | |
| 18 | User quota | MISSING | |
| 19 | Per-user/project accounting | PARTIAL | `submitted_by` only |
| 20 | Persistent state (SQLite) | MISSING | Explicitly in-memory (`registry.rs:1-6`) |
| 21 | Controller restart recovery | MISSING | Nodes self-heal via re-registration; jobs do not |
| 22 | Event log / `ferro events` | MISSING | |
| 23 | Graded fault detection | PARTIAL | Binary at 15 s |
| 24 | Node failure handling | PARTIAL | Rank-failure path is solid; node-disappearance path absent |
| 25 | Orphan process detection | PARTIAL | The *data* is already complete: foreign processes carry an empty `job_id`, and a stray `ferro-<job>-r<rank>` container is parsed back to its job id (`procs.rs:475-482` `parse_ferro_container`), so "ours but unknown to the controller" is distinguishable from "somebody else's". Only the classification and the `managed`/`foreign`/`orphan` label are missing |
| 26 | GPU leak detection | PARTIAL | `release_if_done` covers the happy path; no invariant test |
| 27 | Job timeout | **DONE** | |
| 28 | Queue timeout | PARTIAL | Recorded as `Failed`, not `EXPIRED` |
| 29 | Graceful cancellation | **DONE** *(after the audit)* | SIGTERM → configurable grace → SIGKILL, over the whole descendant tree. The audit's diagnosis was incomplete: `job.child` was always `None`, so under `--no-docker` the stop path killed nothing at all |
| 30–32 | Preemption / checkpointing / cost model | MISSING | |
| 33–35 | Backfill / reservation / gang scheduling | MISSING | Gang behaviour is implicit in the all-or-nothing plan, undocumented and untested |
| 36 | Atomic allocation | **DEFECT** | Lost update under concurrent submit; the agent prevents actual double-execution. See §5.2 |
| 37 | Concurrency safety tests | MISSING | |
| 38–41 | Power / thermal / GPU health / drain | MISSING | Telemetry exists; no policy |
| 42 | Job dependency | MISSING | |
| 43–45 | Retry / OOM detection / adaptive | MISSING | NCCL errors are classified; OOM is not |
| 46 | Explainability | PARTIAL | Feasibility yes, scores no |
| 47–50 | Fragmentation / scheduler metrics / Prometheus / history | MISSING | |
| 51–57 | Benchmarks, generator, simulator, experiments | MISSING | **Blocked by §5.1** |
| 58–60 | Auth / TLS / RBAC | MISSING | |
| 61 | Multi-tenant isolation | PARTIAL | Containers run as the **agent's** uid, not the submitter's (`launcher.rs:401-405`), so all jobs on a node share one uid — no isolation *between FerroGrid users*. Device visibility and mounts are isolated |
| 62 | Audit log | MISSING | |
| 63–65 | CLI additions / queue view / TUI | PARTIAL | Strong existing CLI; new verbs missing |
| 66 | Config system | MISSING | Flags only, no file, no validation |
| 67 | Policy hot switching | MISSING | |
| 68–70 | Failure injection / recovery experiments | MISSING | |
| 71–72 | Scheduler + placement comparison | MISSING | |
| 73–74 | Real + lightweight workloads | PARTIAL | `train_fsdp2.py` exists; no GPU-burn/sleep job |
| 77 | Testing strategy | PARTIAL | 54 unit tests, good quality; no integration/property/failure/concurrency tests |
| 78 | No regression | n/a | Contract captured in §2.6 |

**Tally by table row** (several rows cover more than one spec section): **3 DONE · 18 PARTIAL · 28 MISSING · 1 DEFECT · 1 n/a.**

---

## 7. What the audit concludes

Three things are true at once, and the roadmap has to respect all of them.

1. **The hard part is already built.** Node registration, NVML telemetry,
   process attribution, Docker execution, torchrun/NCCL integration, log and
   metric plumbing, and a genuinely thoughtful placement heuristic all work and
   are tested. None of this should be rewritten.

2. **The scheduling *research* surface is nearly empty.** One placement function,
   one implicit FIFO, no priority, no fairness, no persistence, no metrics, no
   simulator. That is where the term project's contribution lies, and it is
   mostly greenfield — which is good news for not breaking things.

3. **Two structural problems gate everything else**: the controller cannot be
   imported (§5.1), so no experiment can be run against it; and plan-and-reserve is
   not atomic (§5.2), so the controller's ledger of who holds what loses writes
   under concurrency — and that ledger is exactly what every utilisation, waiting
   time and fairness number would be read off.

Phase 1 should therefore do exactly three things: make the crate importable,
make allocation atomic, and put the existing behaviour behind policy traits
without changing a single placement decision.
