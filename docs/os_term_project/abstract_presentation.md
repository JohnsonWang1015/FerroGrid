# Abstract Presentation Outline

- **Target length:** about 7 minutes 50 seconds (within the 5–10 minute limit)
- **Team members:** [Add presenter and team member names]

## Slide 1 — A shared GPU cluster is a scheduling problem (0:50)

- Show several users submitting jobs to one limited GPU pool.
- A burst can delay other users; a strict limit can leave devices idle.
- Research question: **How should a shared GPU cluster allocate limited GPU resources among competing users while balancing waiting time, fairness, and resource utilization?**

**Speaker cue:** Set up the conflict between serving demand quickly and sharing scarce devices predictably.

## Slide 2 — Why this is an Operating Systems project (1:00)

- Queue scheduling decides which waiting job runs next.
- Placement allocates concrete GPUs across heterogeneous nodes.
- Admission control decides whether a request may reserve resources; accounting exposes who currently holds and has consumed them.
- Persistence and recovery preserve the scheduler's view when the controller restarts.

**Visual:** Map each mechanism to the OS resource-management responsibility it implements.

## Slide 3 — FerroGrid as the experimental platform (1:10)

- Controller maintains job and resource state; node agents report GPUs and launch jobs.
- A shared, pure scheduling core serves both the live controller and the offline simulator.
- SQLite-backed job and queue state plus restart reconciliation support persistence and recovery experiments.

**Visual:** One compact flow: users → controller/admission/queue → scheduler → agents → GPUs; connect the scheduler core to the simulator.

## Slide 4 — Policies already in the prototype (1:00)

- Queue: FIFO, priority, aging, fair-share, and SJF.
- Placement: performance, first-fit, best-fit, VRAM-aware, and topology-aware.
- Dispatch: opportunistic backfilling and reservation-based comparison.
- Per-user concurrent GPU quotas extend the system with admission control and usage accounting.
- Quota admission/accounting is implemented in PR #1; the initial quota study is complete in PR #2 and depends on #1. Neither PR has merged as of September 24.

**Speaker cue:** Emphasize the OS mechanisms and use FerroGrid only to run and measure them.

## Slide 5 — Early evidence: policies trade one outcome for another (1:30)

- In a synthetic priority workload, aging reduced the low-priority-to-other mean-wait ratio from 11.72 to 4.62.
- In a simulated burst workload, quota 4 lowered later-user mean wait from 4,456 to 1,886 seconds.
- In that same quota comparison, utilization fell from 98.26% to 77.12%; in a heavy-user workload, quotas also reduced equal-wait treatment.
- The quota experiment used 20 paired seeds and an eight-GPU simulator; these are not real-cluster results.

**Visual:** Simple paired bars for later-user wait and utilization, with a note that each point is a simulator mean over 20 seeds.

## Slide 6 — Evaluation plan for the rest of the term (1:25)

- Consolidate the existing policy, placement, dispatch, recovery, and quota experiments under fixed workloads and clearly defined metrics.
- Report mean and tail wait, per-user wait fairness, throughput, utilization, starvation, and quota-blocked idle capacity with paired-seed confidence intervals.
- If time permits, add one focused fair-share × quota comparison and a mixed-size quota workload.
- Validate a small matched workload on a real multi-GPU cluster if one is available; otherwise keep claims explicitly simulation-only.

**Speaker cue:** Describe a bounded experiment plan, not a promise to add a new scheduler feature for every metric.

## Slide 7 — Contribution and course milestones (0:55)

- Deliver a reproducible, workload-specific account of the trade-offs among fairness, waiting time, throughput, and GPU utilization.
- Sep 29: abstract presentation; Oct 20: progress report; Nov 10 onward: 25-minute presentation/demo; Dec 29: final report, at most 15 pages, with teammate roles.
- Close with the key point: the right allocation policy depends on workload and on which cost users are willing to accept.

**Speaker cue:** End on the expected research result: evidence for choosing a policy, not a claim that one policy wins everywhere.
