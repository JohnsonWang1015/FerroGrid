# Recovery

What happens when the controller comes back, and why "the database is right" is the wrong place to start.

> Status: the design below is what Phase 6 implements. What landed, and what did not, is in [`progress.md`](progress.md).

---

## 1. Two halves, and only one of them was missing

§21 asks the controller to reconcile *desired* state against *actual* state. It is worth being precise about which half FerroGrid was short of, because the answer made this phase much smaller than the specification implies.

**Actual state was already on the wire.** Every agent heartbeat carries, and always has carried:

- `jobs: Vec<JobStatus>` — what that node is running right now
- `gpus[].allocated_job_id` — which job owns each card, from the agent's own allocation table

The agent is the authority on both. Nothing needed building.

**Desired state was what was missing**, and [persistence](persistence.md) supplied it. Before that, the controller's handler for an agent reporting a job it did not recognise was to discard the report — so the two halves could never be compared, because one of them did not survive a restart.

The immediate consequence, which cost no code: once jobs are restored, an agent's report lands on a job the controller knows about, so **"database says running, agent says finished" resolves itself on the next heartbeat.** That is the common case, and it was fixed by Phase 5 without being aimed at.

---

## 2. What is genuinely left

### 2.1 The job nobody claims

A restored job is `Running`, and no agent mentions it. Three ways to get there:

- the agent restarted too and has forgotten it
- the node never came back
- it died while the controller was down, and nothing remains to report the death

They are indistinguishable from the controller's side at the moment of asking, which is the whole difficulty.

### 2.2 The database says queued, the agent says running

A job was promoted and dispatched, then the controller died before the promote reached disk. Write-behind makes that window milliseconds wide, and the dispatch has to succeed inside it — but it is reachable, and a scheduler that silently runs a job it believes is still queued would eventually place something else on the same cards.

---

## 3. Deciding is better than not deciding

The tempting answer to §2.1 is to leave the job `Running` and wait. It is the wrong answer: `ferro jobs` would report work that is not happening, and it would do so indefinitely.

So the controller decides, and the design is about **decides when** and **decides what**.

### When: a window, not an instant

Agents reconnect on their own schedule — a 3 s heartbeat, a 3 s reconnect backoff, and a 15 s health timeout before a node is even called unhealthy. Declaring jobs lost at startup would kill work that is merely slow to be reported.

Reconciliation therefore runs over a window (`--reconcile-window-secs`, default
30) chosen to comfortably exceed the health timeout. Inside it, a job is **claimed** the moment an agent mentions it — by a `JobStatus`, or by a GPU carrying its `allocated_job_id`. Either is proof that somebody is still running it; requiring both would lose jobs whose rank had nothing to report in that particular heartbeat.

### What: failed, with the reason

At the close of the window, anything still unclaimed becomes `Failed`, and the message distinguishes the cases the controller *can* tell apart:

| Situation | What it means |
|---|---|
| The node is healthy and did not report the job | The job is gone. The agent is talking to us and does not have it. |
| The node never reconnected | We do not know. The job may still be running on an unreachable machine. |

Both are terminal, because both leave the controller unable to manage the job — but an operator reading the second one knows to go and look, and reading the first one knows not to bother. Recording only "failed" would throw that away.

Making the job terminal also releases its allocation through the existing path, which is the §26 invariant: a terminal job owns no GPUs.

### The case for reconstructing rather than failing

§2.2 gets the opposite treatment. A job the agents *are* running should not be failed for a bookkeeping gap — and it does not have to be, because the heartbeat says which node and which GPU indices carry that job id. That is a `JobPlan`, minus the rendezvous address and port, which only matter at launch and the job is already launched.

So the controller adopts it: clears `queued`, installs the reconstructed plan, and lets the agent's reports drive the phase from there. The information was already arriving; it just had nowhere to go.

---

## 4. Not assuming the database is right

§21 is explicit that the persisted state is not automatically the truth, and the design reflects it in three places:

1. **The agent wins on allocation.** `allocated_job_id` is overwritten wholesale from each heartbeat. The controller's copy is a cache.
2. **The agent wins on phase.** A restored `Running` job that an agent reports as `Succeeded` becomes succeeded.
3. **Silence is not agreement.** An unclaimed job is not assumed to be running just because the database said so — that is the whole point of §3.

The database is authoritative for exactly one thing: what was *asked for*. The cluster is authoritative for what is *happening*.

---

## 5. Measuring it (§69)

The event log already timestamps everything, so recovery is measurable without new machinery:

- **Recovery time** — `CONTROLLER_RECOVERED` to `RECONCILED`
- **Lost job count** — the failures the `RECONCILED` event counts
- **Incorrect allocation count** — GPUs whose `allocated_job_id` names a job the controller considers terminal, after reconciliation

`RECONCILED` carries the counts: restored, claimed, adopted, failed. That makes the §69 comparison between an in-memory registry and a persistent one a matter of reading two event rows rather than instrumenting a benchmark.

---

## 6. What this does not do

- **No elastic restart.** A job the controller fails here is not resubmitted. Retry policy (§43) is a separate decision and needs failure classification to be meaningful.
- **No agent-side reconciliation.** If the agent is holding a container for a job the controller has failed, the controller does not kill it. Deciding to reach into a node and stop something the operator did not ask about is a bigger call than recovery should make on its own; §25's orphan classification is where that conversation belongs.
- **Logs are still lost.** A restored job has no log history, because logs are not persisted.
