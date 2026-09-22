# Scheduling experiments

Every number in this document is generated. Reproduce all of it with:

```bash
./scripts/run_os_experiments.sh
```

No GPU and no cluster are needed. The tables below are read off
`outputs/benchmarks/*/summary.csv`, which carries the seed, cluster, policy
configuration and git commit for every run.

---

## 1. What is being measured, and what is not

`ferro-sim` links `ferro-sched` and drives the **same policy objects the
controller runs**, over the same `NodeState` snapshot type, with the same
reservation semantics. It does not contain a second scheduler. A change that
improved these numbers without changing `ferro-sched` would be a bug in the
experiment.

It models two things below scheduling, and nothing else:

* **A collective runs at the pace of its slowest rank.** A set of GPUs
  progresses at the speed of its worst member.
* **Crossing the network costs.** A configurable fraction of a multi-node
  job's step is communication; halve the link and that fraction doubles.

It does **not** model PCIe contention between jobs sharing a node, container
startup, data loading, or any other reason a real job misses its estimate.
Durations here are clean. That makes waiting time, fairness and starvation
trustworthy; it makes absolute throughput a statement about the scheduler
rather than about hardware.

### The definitions that matter

| Metric | Definition | Why this one |
|---|---|---|
| **Starvation** | waited > 10× your own runtime, or never ran at all | The first version counted how often a job was *overtaken*. That made FIFO score zero by construction and penalised priority and fair share for doing what they were asked to — a definition that decides the comparison before it runs. |
| **`jain_fairness`** | Jain's index over per-user GPU-seconds | What §7 asks for. Note its limit: on a run where the queue drains, what a user *received* equals what they *demanded*, so it describes the workload more than the policy. |
| **`wait_fairness`** | Jain's index over per-user mean waiting time | What a queue policy can actually move. Low is not automatically bad — see the next row. |
| **`heavy_user_wait_ratio`** | mean wait of the heaviest consumer ÷ mean wait of everyone else | Says which *direction* the unfairness went. ≈1.0 = the cluster does not care what you have used; >1.0 = the heavy user waits longer, which is what fair share is for. |
| **`mean_slowdown`** | actual runtime ÷ runtime on reference hardware | What the placement decision cost. 1.00× means every job landed on hardware as good as the workload was quoted against. |

---

## 2. RQ1 — queue policies, waiting time and fairness

### 2.1 Priority works, and starves; aging fixes most of it

Workload **I**, where three users' work is genuinely of different urgency
(`routine` priority 20 and the heaviest submitter, `normal` 50, `urgent` 90),
on 2 nodes × 4 GPUs.

| Queue policy | avg wait | p95 wait | wait fairness | heavy user waits | GPU util |
|---|---|---|---|---|---|
| `fifo` | 147 s | 289 s | 0.999 | 0.93× | 96.1 % |
| `priority` | 147 s | **445 s** | **0.450** | **11.72×** | 96.8 % |
| `aging` | 147 s | 372 s | 0.613 | 4.62× | 96.6 % |
| `fair-share` | 147 s | 405 s | 0.593 | 5.12× | 96.7 % |
| `sjf` | 108 s | 471 s | 0.977 | 1.37× | 95.9 % |

Read across the row for `priority`: the mean wait is unchanged — reordering a
saturated queue of equal-sized jobs is a permutation, and permutations do not
move the mean — but the *distribution* changes completely. The low-priority
user now waits **11.7× longer** than everyone else, and p95 wait rises by 54 %.

Aging, with a 30 s interval and 5 points per interval, cuts that disparity from
11.72× to **4.62×** and recovers a third of the p95 regression, while still
serving urgent work first. That is the anti-starvation result, and it is the
core of RQ1: strict priority buys responsiveness with the queue's tail, and
aging buys most of the tail back without giving up the responsiveness.

**Mean waiting time is the wrong headline for a queue policy.** Every policy in
that table averages 147 s. If the comparison stopped at the mean, it would
conclude that queue policy does not matter.

### 2.2 Fair share corrects a flooded cluster

Workload **C**: one user (`hog`) submits six times as often as each of three
others, on the same cluster.

| Queue policy | hog waits | alice | bob | carol | heavy/others |
|---|---|---|---|---|---|
| `fifo` | 363 s | 306 s | 415 s | 378 s | 0.99× |
| `fair-share` | **483 s** | 124 s | 86 s | 100 s | **4.69×** |

FIFO treats the flood as 240 individually reasonable requests, because that is
what it is. Fair share charges the hog for what they have already consumed: the
three polite users' waits fall by 3–4×, and the hog's rises by a third.

Note that `jain_fairness` over GPU-seconds is **0.513 under every policy**. The
queue cannot give anyone more GPU than they asked for, only make them wait for
it, so that index is blind to the effect entirely. This is the measurement
trap §2's table describes, and the reason two fairness numbers are reported.

### 2.3 SJF's advantage is entirely a claim about estimate quality

Workload **B** and workload **H** are the same job mix. B's submitters declare
their durations exactly; H's declare nothing 40 % of the time and are wrong by
up to ±50 % the rest.

| | `fifo` avg wait | `sjf` avg wait | improvement | `fifo` starved | `sjf` starved |
|---|---|---|---|---|---|
| **B** — honest estimates | 1308 s | **286 s** | **4.6×** | 124 | **0** |
| **H** — realistic estimates | 1272 s | 1225 s | 1.04× | 105 | 47 |

With perfect information SJF is transformative. With the information a real
cluster actually has, it is worth 4 %. Nothing verifies a declared duration,
and a user who under-declares is rewarded, so the honest summary is that SJF is
an instructive experiment rather than a policy to run.

Its p95 also worsens where its mean improves — 591 s → 1451 s on workload C —
which is the long jobs paying for the short ones' gain.

### 2.4 Scheduler overhead

| Policy | µs per scheduling pass |
|---|---|
| `fifo` | 1.7 |
| `priority` | 1.8 |
| `aging` | 1.8 |
| `fair-share` | 1.9 |
| `sjf` | 1.8 |

Measured, over 335 passes with up to 240 jobs in flight. The difference between
the cheapest and the most expensive policy is 0.2 µs per pass. Policy choice
costs nothing; it is worth saying so with a number rather than assuming it.

---

## 3. RQ2 — placement on heterogeneous hardware

Workload **E** on a cluster where node order is deliberately not performance
order: `gpu-a` has one RTX 3090 (35 TFLOP/s), `gpu-b` two A6000 (38),
`gpu-c` two RTX 4090 (82). Every NIC negotiated 1000 Mb/s, but `ferro net`
measured `gpu-a↔gpu-b` at 940 and everything touching `gpu-c` at 90.

| Placement | avg wait | p95 wait | GPU util | mean slowdown | starved |
|---|---|---|---|---|---|
| `performance` | 18 758 s | 77 230 s | 83.4 % | **3.13×** | 122 |
| `best-fit` | 18 973 s | 81 017 s | 83.3 % | 3.20× | 119 |
| `first-fit` | 9 198 s | 26 554 s | 84.5 % | 1.58× | 117 |
| `vram` | 9 198 s | 26 554 s | 84.5 % | 1.58× | 117 |
| `topology` | 9 417 s | **25 672 s** | **89.6 %** | 1.65× | **114** |

Where the multi-node jobs landed:

| Placement | dominant pair | measured link | slowdown on those jobs |
|---|---|---|---|
| `performance` | `gpu-c + gpu-b` (35 of 36) | 90 Mb/s | **9.31×** |
| `best-fit` | split, mostly across `gpu-c` | 90 Mb/s | 9.73× |
| `first-fit` | `gpu-a + gpu-b` (32 of 36) | 940 Mb/s | 3.11× |
| `topology` | `gpu-b + gpu-a` (30 of 36) | 940 Mb/s | 3.50× |

**The existing performance-aware policy walks straight into the slow wire.** It
ranks multi-node placements by *negotiated* link speed before GPU throughput,
and every NIC here claims 1000 Mb/s — so the negotiated figure is
uninformative, GPU speed decides, and it pairs the two fastest nodes, which are
the two connected at 90 Mb/s. Those jobs run 9.3× slow.

**First-fit beating it is luck, not skill.** Alphabetical order happens to
start at `gpu-a`, which happens to be on the fast side of the fabric. Change
the node names and that advantage disappears; it knows nothing about either
GPUs or wires.

**Topology-aware placement is the only one that avoids the pair on purpose**,
and it produces the best p95 wait, the best utilisation (89.6 % against
83–85 %) and the fewest starved jobs. It is *not* the best on mean slowdown,
because when its preferred pair is busy it falls back to the slow one — which
is a real limitation, not a rounding error.

This is the case that justifies storing `ferro net` results at all
(`current_state.md` §5.3): the negotiated link speed describes the
node-to-switch hop and nothing else.

### The control

Re-running the same comparison with `--flat-execution`, which makes every
placement run at the same speed, gives **identical results for all five
strategies** — avg wait 4182 s, util 89.7 %, slowdown 1.00×. That is the
control: it confirms the differences above come from the execution model
representing real effects, and not from the strategies diverging by accident.

---

## 4. An unplanned finding: FerroGrid already backfills

The controller's `run_queue` walks the entire waiting list each tick and starts
anything that fits, not just the head. That is opportunistic dispatch — it
backfills, without reservations — and it is not what "FIFO queue" usually
implies.

Workload **D** (large distributed jobs against a stream of small ones):

| Dispatch | avg wait | p95 wait | GPU util | starved |
|---|---|---|---|---|
| opportunistic (what FerroGrid does) | **1 529 s** | 10 801 s | **98.1 %** | **15** |
| strict head-of-line | 7 471 s | 14 169 s | 94.0 % | 165 |

Strict FIFO is **4.9× worse** on mean waiting time and starves eleven times as
many jobs. On workload A, where every job is the same size, the two are
identical — head-of-line blocking needs jobs of different sizes to bite.

The other half of that finding is the cost: with no reservation, a four-GPU job
can be walked past indefinitely by one-GPU jobs. Workload D's p95 wait is
10 801 s against a mean of 1 529 s, and that tail is the large jobs. Reservation
(§34) is what bounds it, and is not implemented.

---

## 5. Node failure

Workload **G** removes `gpu-b` 1 500 s into the run. Utilisation drops to 55 %
— half the cluster is gone and the makespan stretches to cover the same work on
what remains — and the jobs that were on it are recorded as failed rather than
silently restarted, which is what the controller does.

This is the scheduling half of fault handling. The recovery half — a controller
restart, reconciliation against what the agents report — cannot be answered by
a simulator with no controller to restart, and is why §51's workload F is
absent from the catalogue rather than faked. It belongs with the persistence
work.

---

## 6. Limitations of these results

1. **One seed per scenario.** Each row is a single run. Confidence intervals
   would need repeated seeds, which the runner supports (`seed` is a workload
   field) but the script does not yet sweep.
2. **The execution model is an assumption.** The slowest-rank and
   communication-share effects are real, but the specific constants
   (82 TFLOP/s reference, 35 % communication share) are calibrated to the
   README's measured 1 GbE numbers rather than derived from these workloads.
   Absolute slowdowns should be read as "this scale", not "this figure".
3. **Aging's parameters are tuned to these workloads** (30 s interval, 5 points)
   and differ from the controller's defaults (60 s, 1 point). Aging only does
   anything when its interval is short relative to how long jobs actually wait;
   quoting a result at parameters nobody reaches would show aging doing nothing
   and invite the wrong conclusion.
4. **No backfill or reservation policy exists yet**, so the §51 comparison that
   includes them is incomplete. What is measured instead is the dispatcher
   FerroGrid already has, which turns out to be the more interesting question.
5. **Fragmentation is reported but not yet interesting.** The workloads do not
   create enough packing pressure for `best-fit` and `first-fit` to separate on
   it; that needs a workload of mixed-width jobs on a cluster sized to fragment.
