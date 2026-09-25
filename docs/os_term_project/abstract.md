# FerroGrid: Evaluating Scheduling and Resource Allocation Policies for Shared GPU Clusters

- **Course:** Operating Systems Term Project
- **Team members:** 王嘉暐、張健勳、陳育哲
- **Abstract submission:** September 29, 2026

## Project abstract

GPU clusters are shared by users whose jobs differ in size, duration, urgency, and arrival pattern. A scheduler that keeps devices busy may let a burst of work from one user dominate the queue; a policy that limits that user may protect others while leaving GPUs idle. GPU heterogeneity and multi-GPU jobs add a placement problem: choosing a device set affects both the work that can run now and the capacity left for later jobs. These are operating-system resource management questions, not simply questions of building a training platform.

Our central research question is: **How should a shared GPU cluster allocate limited GPU resources among competing users while balancing waiting time, fairness, and resource utilization?** We study how queue order, GPU placement, dispatch, and per-user admission limits change that balance under different workloads. We distinguish fairness of access from equality of waiting time: a hard quota can isolate users by limiting concurrent GPU holdings, while fair-share scheduling can prefer users with less accumulated service without forbidding a user from using otherwise idle capacity.

## Operating-systems perspective and prototype

FerroGrid is the experimental vehicle for this study. It is a small distributed resource manager: a controller tracks jobs and GPU availability, agents report node resources and launch jobs, and a pure scheduling library makes queue and placement decisions. The offline simulator uses the same scheduling policies as the controller, allowing repeatable workload comparisons without treating a second, separately implemented scheduler as evidence. Durable job and queue state, event history, and restart reconciliation make persistence and recovery part of the resource-management setting as well.

The prototype already has FIFO, priority, aging, fair-share, and shortest-job-first queue policies; performance, first-fit, best-fit, VRAM-aware, and topology-aware placement; and opportunistic backfilling with an optional reservation-based dispatch mode. Per-user GPU quotas add admission control and usage accounting: the controller checks the quota at the atomic reservation boundary and reports current GPU holdings and job-derived GPU-seconds. The quota identity is the client-supplied submitted-by field. It is not authenticated and is not a security boundary.

The queue-policy, placement, dispatch, persistence, and recovery work is implemented and its simulator studies are recorded in the [project progress log](progress.md) and [scheduling experiments](experiments.md). The quota admission implementation and its first multi-user evaluation are complete in the working prototype but remain under review in [PR #1](https://github.com/JohnsonWang1015/FerroGrid/pull/1) and [PR #2](https://github.com/JohnsonWang1015/FerroGrid/pull/2), respectively; PR #2 depends on PR #1. As of September 24, 2026, neither has merged. The [roadmap](roadmap.md) describes the intended system, while the older [Phase 0 audit](current_state.md) is a dated baseline rather than a current feature inventory.

## Proposed work and evaluation

Existing work already includes simulator comparisons of queue policies, placement strategies, dispatch modes, and per-user quotas, plus a controller-crash recovery experiment. The remaining term work is to synthesize them into one coherent OS evaluation: align workload definitions and fairness measures, reproduce the key comparisons from a clean checkout, and explain how each result bears on the central question. Where existing studies do not share a comparable workload, we will rerun only the rows needed to make the trade-offs interpretable. The completed quota study remains a fixed-FIFO baseline; a limited fair-share-by-quota comparison and mixed-size quota workload are extensions only if time and implementation stability permit.

For synthesis and any reruns, we will report mean and tail waiting time, turnaround, completed jobs per hour, GPU utilization, starvation, and per-user waiting-time distributions. We will use confidence intervals over paired seeds where the experiment design supports them and preserve workload and policy configurations so another student can reproduce each result. GPU-seconds will describe service consumed, but will not stand alone as a fairness measure when all jobs eventually complete. We will interpret utilization and throughput alongside waiting-time fairness and idle capacity blocked by quotas.

Where a real multi-GPU cluster is available, we will run a small matched set of workloads to check whether the simulator's qualitative policy trade-offs hold under real launch, communication, and hardware effects. This is an external validation step, not the source of the existing quota results. If cluster time is unavailable, we will report that limitation explicitly and make no claim that simulated values are real-cluster measurements.

## Preliminary observations

Existing scheduling experiments show why a single average is insufficient. In one synthetic priority workload, strict priority made the low-priority user's mean wait 11.72 times that of other users; aging reduced the ratio to 4.62 while retaining service for urgent jobs. In the completed quota study, 20 paired seeds on an eight-GPU simulated cluster showed a workload-specific trade-off. When an early hog stream could overlap with light-user arrivals, a four-GPU per-user quota reduced light users' mean wait from 4,456 to 1,886 seconds, while utilization fell from 98.26% to 77.12% and throughput from 156.93 to 123.16 jobs per hour. In a different heavy-user workload, the same quota reduced other users' waiting time but also reduced mean-wait equality and utilization. These are synthetic simulator results, not real-cluster measurements; the quota matrix holds FIFO, opportunistic dispatch, and performance placement fixed. The [quota evaluation PR](https://github.com/JohnsonWang1015/FerroGrid/pull/2) records the full scenarios, confidence intervals, and limitations.

## Expected contribution and schedule

The expected contribution is a practical shared-GPU resource-management platform with multiple OS-inspired policies in a common framework, plus a reproducible evaluation of their trade-offs. The analysis will identify when a policy improves access for some users and what it costs in waiting time, throughput, or idle capacity. We expect a workload-specific account rather than a universal winning algorithm, supported by experiment configurations and, where practical, a real-cluster validation.

- **September 29:** submit this two-page-style abstract and give a 5–10 minute in-class presentation.
- **October 20:** present a progress report with the selected experiment matrix and any real-cluster validation status.
- **November 10 onward:** give the 25-minute project presentation and demonstration.
- **December 29:** submit a final report of at most 15 pages, including each teammate's role.
