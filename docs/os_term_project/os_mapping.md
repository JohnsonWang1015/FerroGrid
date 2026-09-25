# FerroGrid ↔ Operating Systems: concept mapping

FerroGrid is a **distributed resource manager**. The machine it manages is a cluster rather than a single host, and the processor it schedules is a GPU rather than a core, but the problems are the textbook ones: who runs, when, on which processor, with how much memory, under what isolation, and what happens when something dies.

This document maps each OS concept onto the concrete FerroGrid code that implements it. Where a row is aspirational rather than implemented, it says so — a mapping table that quietly promises unbuilt features would be worse than none.

---

## 1. Process and resource abstractions

| OS concept | FerroGrid | Where |
|---|---|---|
| Process / task | **Job** — a named, owned, distributed unit of work with a lifecycle and an exit code | `registry.rs::Job`; `proto` `JobSummary` |
| Thread / worker within a process | **Rank** — one torchrun process pinned to one GPU | `proto` `JobPlacement`, `JobStatus.node_rank` |
| PID | `job_id` (`j` + 10 hex chars), unique per controller | `service.rs::new_job_id` |
| PCB (process control block) | `registry.rs::Job` — plan, per-rank status, submitter, timeout, logs, metrics, queue state | `registry.rs:60-87` |
| Process state diagram | `JobPhase`: Pending → Launching → Running → {Succeeded, Failed, Cancelled} | `proto` `JobPhase`; `JobPhase::is_terminal` |
| Processor / CPU | **GPU** — the scarce compute resource actually being scheduled | `proto` `Gpu`; `crates/ferro-gpu` |
| Heterogeneous cores (big.LITTLE) | **Heterogeneous GPUs** — RTX 4090 / 5090 / A6000 in one cluster, ranked by *measured* TFLOP/s rather than by model name | `scheduler.rs::score`; `ferro bench` |
| Main memory | **VRAM** — the binding capacity constraint | `Gpu.memory_total_b/memory_used_b`; `--min-free-vram-gib` |
| Memory protection / OOM | VRAM floor before placement; a card without headroom is not offered | `scheduler.rs::free_gpus` |
| Machine / node | **Node**, running an agent | `proto` `NodeInfo`; `crates/ferro-agent` |

## 2. Scheduling

| OS concept | FerroGrid | Where |
|---|---|---|
| Ready queue | **Job queue** — jobs submitted with `--wait` that do not yet fit | `registry.rs::queued_jobs`; `service.rs::run_queue` |
| Short-term (CPU) scheduler | **Queue policy** — "which job runs next" | today: FIFO by `job_order` (`registry.rs:286`). Phase 1 makes it a `QueuePolicy` trait |
| Processor affinity / assignment | **Placement policy** — "which GPUs on which nodes" | `scheduler.rs::plan`, `plan_auto` |
| FCFS | FIFO queue, strictly by submission order, not timestamp | `registry.rs:298-301` |
| Priority scheduling | *Not implemented* — Phase 2 | — |
| Aging (anti-starvation) | *Not implemented* — Phase 2 | — |
| Fair-share / proportional scheduling | *Not implemented* — Phase 2. `submitted_by` is the raw material | `proto` `SubmitJobRequest.submitted_by` |
| SJF / SRTF | *Not implemented* — Phase 2, deliberately with **no fabricated duration estimate** | — |
| Gang / co-scheduling | Implicit today: a plan is all-or-nothing, so every rank starts together or none does. **Not named, not tested** — Phase 3 makes the guarantee explicit | `service.rs::start_job` |
| Head-of-line blocking | Present, and the motivation for backfilling (§33) | `service.rs::run_queue` processes the queue in order each tick |
| Backfilling / reservation | *Not implemented* — Phase 7 | — |
| Preemption | *Not implemented* — Phase 8. Would require cooperative checkpointing; FerroGrid must not claim transparent checkpointing of arbitrary PyTorch programs | — |
| Scheduling latency | *Not measured* — Phase 4 | — |

## 3. Memory and multi-resource allocation

| OS concept | FerroGrid | Where |
|---|---|---|
| Contiguous allocation / fragmentation | GPUs are allocated as whole devices; fragmentation appears as "4 free GPUs, but no node has 2 of the same model" | `scheduler.rs::homogeneous_options` |
| First-fit / best-fit | *Not named as strategies* — Phase 3 | — |
| Multi-resource allocation (CPU + RAM + GPU) | Node CPU and RAM **are advertised** but never used in placement — Phase 7 | `proto` `NodeInfo.cpu_count:8, memory_total_b:9` |
| Dominant Resource Fairness | *Not implemented* — Phase 7 | — |
| Quota / rlimit | Optional per-user concurrent GPU limit; controller admission and reservation share one atomic registry check. Unconfigured users are unlimited; requests larger than the hard limit are rejected and temporary quota blocks can queue | `quota.rs::QuotaTable`; `registry.rs::reserve_exact_with_quota`; controller `--user-quota USER=N` |

## 4. Isolation and protection

| OS concept | FerroGrid | Where |
|---|---|---|
| Address-space isolation | **Container isolation** — one Docker container per rank | `ferro-agent/src/launcher.rs` |
| Capability restriction / device visibility | GPU pinning: only the allocated device indices are visible to the container | `launcher.rs` (`--gpus "device=0,1"`) |
| File-system namespace | Explicit bind mounts; only the agent workspace is mounted by default, datasets need `--mount` | `proto` `SubmitJobRequest.mounts:10` |
| User / uid | Containers run **not as root**, but as the *agent's* uid/gid (`getuid()`/`getgid()` on the node), **not** the submitter's. All FerroGrid jobs on a node therefore share one uid: no per-user isolation between them | `launcher.rs:401-405`; `proto` `ProcessEntry.runs_as` |
| Information leak prevention | Credential-looking flags are **redacted** before a command line leaves the node | `procs::redact_secrets` |
| Authentication / access control | *Not implemented* — Phase 8. gRPC is unauthenticated; `submitted_by` is client supplied, so quotas are resource management, not a security boundary | README §"Scope and limitations" |

## 5. Inter-process communication

| OS concept | FerroGrid | Where |
|---|---|---|
| IPC / RPC | **gRPC** between CLI ↔ controller ↔ agents | `proto/ferrogrid.proto`, services `Controller` and `NodeAgent` |
| Pipes / stdout redirection | Agent captures rank stdout/stderr and streams it upward | `ReportLogs`, `StreamLogs` |
| Message passing between workers | **NCCL** collectives between ranks — the distributed-IPC layer FerroGrid configures but does not implement | `NCCL_SOCKET_IFNAME` pinning in `ferro-agent/src/state.rs::detect_ifname` |
| Rendezvous / barrier | torchrun rendezvous on rank 0; rank 0 is launched first so peers do not burn retry timeout | `service.rs::start_job` |
| Shared-memory telemetry channel | Structured metrics on stdout: `FERRO_METRIC {json}` | `metrics.rs::parse_metric_line` |

## 6. Failure detection and recovery

| OS concept | FerroGrid | Where |
|---|---|---|
| Watchdog / failure detector | **Heartbeat**, 3 s interval, 15 s timeout → unhealthy | `registry.rs::HEARTBEAT_TIMEOUT_S`, `Node::healthy` |
| Phi-accrual / graded suspicion | *Not implemented* — binary healthy/unhealthy only. Phase 7 adds Healthy/Suspected/Dead/Recovering | — |
| Process timeout / `ulimit -t` | **Job timeout** (`--timeout`), reaped every 15 s | `registry.rs::expired_jobs`; `service.rs::reap_expired` |
| Orphan / zombie reaping | A failed rank tears down its surviving peers, so nobody sits in a collective holding GPUs | `service.rs::report_job_status` |
| Orphan detection | Foreign GPU processes **are** discovered and shown, but are not classified as `ORPHAN` against the job table — Phase 7 | `procs.rs`; `proto` `ProcessEntry.external:15` |
| Resource leak detection | `release_if_done` frees GPUs on terminal phase; **no invariant test** that a terminal job owns nothing — Phase 1 adds one | `registry.rs::release_if_done` |
| Crash recovery / journaling | *Not implemented* — state is in memory. Phases 5–6 add SQLite + WAL and reconciliation | `registry.rs:1-6` |
| Graceful termination (SIGTERM → grace → SIGKILL) | *Not implemented* — `docker kill` is an immediate SIGKILL | `ferro-agent/src/state.rs:169-179` |

## 7. Observability — the `ps`/`top` layer

| OS tool | FerroGrid | Notes |
|---|---|---|
| `ps` | `ferro ps` | Lists **every** process on every GPU, not just FerroGrid's, so a card held by somebody's notebook is not read as free |
| `ps -p <pid>` | `ferro ps <pid>` | Reads the node live rather than using the heartbeat, because heartbeat numbers are up to one interval old |
| `top` / `htop` | `ferro watch` | Live dashboard; always surfaces data age, because a stale reading looks exactly like an idle GPU |
| `lscpu` / `nvidia-smi -L` | `ferro nodes`, `ferro gpu` | |
| `uptime` / load | *Not implemented* — cluster utilisation metrics arrive in Phase 4 | |
| `dmesg` / audit log | *Not implemented* — event log is Phase 5 | |
| `iperf` | `ferro net` | Measures real pairwise TCP throughput, strictly one pair at a time |
| accounting (`sa`, `acct`) | `ferro usage` reports current GPU holdings and running jobs plus job-derived GPU-seconds and optional per-user quota; usage and fair-share share `RegistryInner::usage_snapshot` | `Controller.GetUsage`; `ferro usage [--json] [--watch]` |

---

## 8. Where FerroGrid differs from a classic OS

Worth stating plainly, because the differences are what make the project interesting rather than a re-implementation:

1. **No preemption by default.** A GPU job holds its device until it finishes. There is no cheap context switch: swapping out a training job means checkpointing gigabytes. This is why aging and backfilling matter *more* here than in a CPU scheduler — starvation cannot be fixed by a time slice.
2. **The scheduler does not own the machine.** Other users, notebooks and desktop compositors hold GPUs FerroGrid never allocated. "Free" therefore means *placeable* — unallocated **and** with enough VRAM headroom — not merely "not ours".
3. **Heterogeneous processors are the normal case**, and their speed is *measured*, not inferred from a model name.
4. **The interconnect is part of the scheduling decision.** On 1 GbE, crossing the network costs ~55× throughput, so placement must prefer one node — a consideration with no real analogue in single-host CPU scheduling.
