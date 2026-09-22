# FerroGrid — OS Term Project Progress Log

Updated at the end of every phase, in the format fixed by the project
specification (§91). Newest entry first.

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
