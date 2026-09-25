# GPU Quota Evaluation

## Research Questions

- **RQ1:** How do hard per-user concurrent GPU limits change waiting-time treatment when users have equal demand, one user submits most jobs, or one user arrives in a burst?
- **RQ2:** What utilization and throughput cost accompanies those changes?
- **RQ3:** How sensitive is the trade-off to limits of 1, 2, or 4 GPUs on an 8-GPU cluster?

The results do not support a blanket claim that a quota improves fairness. They show a workload-dependent exchange: a hard limit can protect other users from a dominant or early-arriving queue, while moving delay to the limited user and leaving GPUs idle.

## Experimental Setup

The experiment uses the simulator's shared `ferro-admission::QuotaTable` and `QuotaDecision` model, the same pure admission semantics used by `ferro-controller`. The controller's atomic GPU reservation and registry lock remain controller-owned. This experiment does not implement a second quota rule or scheduler.

| Setting | Value |
|---|---|
| Cluster | 2 nodes × 4 homogeneous GPUs = 8 GPUs |
| Queue policy | FIFO |
| Dispatch | Opportunistic |
| Placement | Performance |
| Quotas | Unlimited, 1, 2, and 4 concurrent GPUs per user |
| Seeds | 20 paired seeds per scenario and quota; the same seed set is reused for every quota |
| Jobs | 1 GPU each; duration uniform from 120 through 240 seconds |
| Execution | Simulator default model; no node failures |
| Runs | 4 scenarios × 4 quota settings × 20 seeds = 320 |

For each scenario and seed, the workload is generated once and the same job identities, users, arrival times, and durations are reused for all four quota settings. The quota is the only scheduling input that changes within that paired group. The main matrix uses one-GPU jobs, so every job fits the smallest tested quota; no main-matrix run rejects an oversized job.

| Scenario | Workload |
|---|---|
| A — Balanced | 240 jobs; four users receive equal arrival weights in a Poisson stream with mean interarrival time 5 s. |
| B — Heavy user | 240 jobs; one user receives 70% of arrivals and three other users share 30%, with the same one-GPU job and duration distributions. |
| C — Overlapping burst | 250 jobs; 160 jobs from `hog` arrive with seeded exponential interarrival mean 0.15 s. Their cumulative timestamps are floored to the simulator's integer-second clock, so the hog stream can continue after the three light-user streams begin at t=20 s; those users then submit 30 jobs each at 3 s intervals plus seeded 0–2 s jitter. |
| D — Single user | 240 jobs from one user on the same 8-GPU cluster, so no other user can borrow capacity left idle by the quota. |

The fixed FIFO baseline matters to interpretation. Scenario B assigns user identity to a shared arrival stream; FIFO serves the resulting arrivals in order, so the 70%-demand user is not intentionally placed at the front of the queue. Scenario C starts one user's burst early, but the burst can overlap with the other users' arrivals.

## Metrics and Statistical Method

- **Mean, p50, and p95 waiting time:** submission-to-start time for jobs that started. Percentiles use nearest rank. All 320 main-matrix runs start every submitted job, so these values do not conceal jobs rejected or left waiting.
- **Turnaround:** submission to completion. **Makespan:** first arrival to last completion.
- **GPU utilization:** GPU-seconds used divided by `8 × makespan`.
- **Throughput:** completed jobs per hour of makespan. Every run completes all submitted jobs, so throughput and utilization move together for this fixed workload; they are related views of the same longer or shorter drain time.
- **Per-user service and waiting:** raw output includes submitted/completed jobs, mean/p50/p95 wait, GPU-seconds received, and share of total GPU-seconds.
- **Wait fairness:** Jain's index over each user's mean wait. A lower value means users' mean waits differ more; it does not by itself say which user was treated better. The heavy-to-normal ratio supplies that direction.
- **Heavy-to-normal wait ratio:** the heavy user's mean wait divided by the unweighted average of the other users' mean waits. A value above 1 means the heavy user waits longer; below 1 means the later/normal users wait longer.
- **Jain GPU-seconds:** reported in the raw data, but not used as the primary policy fairness result. Since every job completes and quotas do not change the submitted jobs or their durations, this index primarily describes the workload's demand mix.
- **Quota observations:** blocked jobs count distinct jobs ever delayed by a quota; block events count continuous blocked intervals per job; attempts count scheduling passes while blocked; block time ends when a job starts or the run ends. Hard rejects are requests larger than a user's whole limit. `quota_unused_gpu_seconds` counts placeable free GPU-seconds during intervals with at least one quota-blocked queued job. It is a direct underutilization measure for scenario D; in multi-user scenarios it is an overlap measure and does not prove that every idle GPU was caused only by quota enforcement.

Each table cell below is **mean ± sample standard deviation [95% confidence interval]** across the 20 seed-level runs. These per-cell intervals use a two-sided Student-t critical value with 19 degrees of freedom (2.093), computed as `mean ± t × SD / sqrt(20)`. The checked-in `summary.csv` and `summary.json` retain full precision and include all metrics; values here are rounded for reading.

The table below reports **paired differences (quota minus unlimited)** for headline outcomes. For each scenario and seed, we subtract the unlimited result from the quota result before calculating the mean and two-sided 95% Student-t CI (n=20, df=19). Every quota cell uses the same 20 seeds as its scenario's unlimited baseline. Wait differences use overall mean wait for A/D and heavy-user plus normal-user mean waits for B/C; utilization is shown as percentage-point change.

| Scenario | Quota | Paired wait difference, s (95% CI) | Utilization difference, pp (95% CI) | Throughput difference, jobs/h (95% CI) |
|---|---:|---|---:|---:|
| A — Balanced | 1 | +2,777.4 [2,742.7, 2,812.1] | −54.35 [−55.54, −53.15] | −86.86 [−88.95, −84.77] |
| A — Balanced | 2 | +42.7 [27.3, 58.0] | −11.46 [−13.88, −9.03] | −18.34 [−22.25, −14.43] |
| A — Balanced | 4 | +0.2 [−0.1, 0.5] | +0.01 [−0.15, +0.17] | +0.02 [−0.24, +0.27] |
| B — Heavy user | 1 | Heavy +12,645 [12,272, 13,017]; normal −565 [−704, −427] | −80.43 [−80.90, −79.96] | −128.53 [−129.43, −127.63] |
| B — Heavy user | 2 | Heavy +4,963 [4,777, 5,148]; normal −1,576 [−1,677, −1,476] | −62.92 [−63.78, −62.05] | −100.54 [−101.89, −99.18] |
| B — Heavy user | 4 | Heavy +1,131 [1,037, 1,224]; normal −1,102 [−1,219, −984] | −28.32 [−29.93, −26.71] | −45.24 [−47.79, −42.70] |
| C — Overlapping burst | 1 | Hog +12,637 [12,529, 12,744]; light users −1,894 [−1,931, −1,857] | −78.76 [−78.98, −78.54] | −125.78 [−126.54, −125.02] |
| C — Overlapping burst | 2 | Hog +5,412 [5,365, 5,458]; light users −3,241 [−3,272, −3,210] | −59.37 [−59.66, −59.07] | −94.81 [−95.46, −94.15] |
| C — Overlapping burst | 4 | Hog +1,800 [1,783, 1,816]; light users −2,570 [−2,603, −2,536] | −21.15 [−21.64, −20.65] | −33.77 [−34.54, −33.00] |
| D — Single user | 1 | +18,930.6 [18,794.0, 19,067.3] | −85.57 [−85.78, −85.37] | −136.75 [−137.55, −135.95] |
| D — Single user | 2 | +8,105.3 [8,046.3, 8,164.4] | −73.12 [−73.32, −72.93] | −116.85 [−117.54, −116.17] |
| D — Single user | 4 | +2,696.5 [2,676.7, 2,716.3] | −48.42 [−48.60, −48.24] | −77.38 [−77.86, −76.89] |

## Results

### Scenario A — Balanced users

| Quota | Mean wait (s) | p95 wait (s) | GPU utilization (%) | Throughput (jobs/h) |
|---:|---:|---:|---:|---:|
| Unlimited | 2,026 ± 59 [1,999, 2,054] | 3,900 ± 106 [3,851, 3,950] | 98.07 ± 0.44 [97.87, 98.28] | 156.72 ± 1.92 [155.83, 157.62] |
| 1 | 4,804 ± 107 [4,754, 4,854] | 9,607 ± 552 [9,349, 9,865] | 43.72 ± 2.67 [42.47, 44.98] | 69.86 ± 4.14 [67.92, 71.80] |
| 2 | 2,069 ± 60 [2,041, 2,097] | 4,231 ± 304 [4,089, 4,374] | 86.61 ± 5.33 [84.12, 89.11] | 138.39 ± 8.24 [134.53, 142.24] |
| 4 | 2,027 ± 59 [1,999, 2,054] | 3,903 ± 109 [3,852, 3,954] | 98.08 ± 0.50 [97.85, 98.32] | 156.74 ± 1.80 [155.90, 157.58] |

For this balanced control, no difference was detected between quota 4 and unlimited access. The paired 95% CIs bound the changes to +0.2 s mean wait [−0.1, +0.5], +0.01 percentage points of utilization [−0.15, +0.17], and +0.02 jobs/h throughput [−0.24, +0.27]. Limits of 1 and 2 increase the mean wait and reduce capacity use even though the users have equal arrival weights.

### Scenario B — One user receives 70% of arrivals

| Quota | Heavy-user mean wait (s) | Other users' mean wait (s) | Heavy/normal ratio | Jain wait index | GPU util. (%) | Throughput (jobs/h) |
|---:|---:|---:|---:|---:|---:|---:|
| Unlimited | 2,030 ± 58 [2,003, 2,057] | 2,018 ± 169 [1,939, 2,097] | 1.01 ± 0.10 [0.96, 1.06] | 0.991 ± 0.007 [0.988, 0.995] | 98.07 ± 0.44 [97.87, 98.28] | 156.72 ± 1.92 [155.83, 157.62] |
| 1 | 14,674 ± 813 [14,294, 15,055] | 1,453 ± 225 [1,347, 1,558] | 10.39 ± 2.00 [9.45, 11.32] | 0.409 ± 0.035 [0.393, 0.426] | 17.64 ± 0.87 [17.23, 18.05] | 28.20 ± 1.51 [27.49, 28.91] |
| 2 | 6,992 ± 412 [6,800, 7,185] | 442 ± 113 [389, 494] | 17.00 ± 5.02 [14.65, 19.35] | 0.351 ± 0.033 [0.335, 0.366] | 35.16 ± 1.75 [34.33, 35.98] | 56.19 ± 3.04 [54.77, 57.61] |
| 4 | 3,160 ± 211 [3,061, 3,259] | 916 ± 172 [836, 997] | 3.59 ± 0.83 [3.20, 3.98] | 0.693 ± 0.085 [0.654, 0.733] | 69.75 ± 3.41 [68.16, 71.35] | 111.48 ± 5.87 [108.73, 114.23] |

Without a quota, mean waits are already similar under FIFO: the heavy user's mean is 2,030 s and the average of the three other users is 2,018 s. The quota does reduce the other users' waits, but it does so by shifting delay to the heavy user's queue. At quota 4, the other-user mean falls about 55% while the heavy-user mean rises about 56%; utilization falls 28.3 percentage points and throughput falls 28.9% from the unlimited baseline. Quotas 1 and 2 shift more delay, but cost roughly 82% and 64% of throughput, respectively.

This is **not an improvement in equal mean-wait treatment** in scenario B: the wait Jain index falls from 0.991 without a quota to 0.693 at quota 4 and 0.351 at quota 2. A quota can provide admission isolation to lighter users, but this workload gives no evidence that it improves their aggregate wait without substantial cost to the heavy user and cluster throughput.

### Scenario C — Overlapping burst and light-user arrivals

| Quota | Hog mean wait (s) | Light users' mean wait (s) | Hog/light ratio | Jain wait index | GPU util. (%) | Throughput (jobs/h) |
|---:|---:|---:|---:|---:|---:|---:|
| Unlimited | 1,708 ± 31 [1,694, 1,722] | 4,456 ± 66 [4,425, 4,487] | 0.383 ± 0.006 [0.381, 0.386] | 0.909 ± 0.002 [0.909, 0.910] | 98.26 ± 0.38 [98.09, 98.44] | 156.93 ± 2.00 [155.99, 157.87] |
| 1 | 14,345 ± 260 [14,223, 14,466] | 2,562 ± 54 [2,537, 2,587] | 5.603 ± 0.184 [5.517, 5.689] | 0.538 ± 0.009 [0.534, 0.543] | 19.50 ± 0.17 [19.42, 19.58] | 31.15 ± 0.53 [30.90, 31.40] |
| 2 | 7,120 ± 129 [7,059, 7,180] | 1,215 ± 26 [1,202, 1,227] | 5.865 ± 0.196 [5.774, 5.957] | 0.526 ± 0.009 [0.521, 0.530] | 38.90 ± 0.37 [38.73, 39.07] | 62.13 ± 1.07 [61.62, 62.63] |
| 4 | 3,508 ± 64 [3,478, 3,538] | 1,886 ± 40 [1,868, 1,905] | 1.861 ± 0.061 [1.832, 1.889] | 0.914 ± 0.009 [0.910, 0.918] | 77.12 ± 0.86 [76.72, 77.52] | 123.16 ± 2.31 [122.08, 124.24] |

The unlimited baseline leaves light users waiting much longer than the hog user. Quota 4 reduces their mean wait by about 58% and raises the hog user's mean by about 105%; the paired differences are −2,570 s [−2,603, −2,536] for light users and +1,800 s [1,783, 1,816] for the hog user. It moves the mean-wait Jain index only slightly (0.909 to 0.914). Quotas 1 and 2 push the ratio past 5.6 and reduce wait fairness substantially. Among these tested limits, 4 is the least disruptive setting that still changes the light users' outcome materially; it costs 21.15 utilization points [20.65, 21.64] and 33.77 jobs/h [33.00, 34.54] in paired comparisons.

### Scenario D — One user is the only source of demand

| Quota | Mean wait (s) | GPU utilization (%) | Throughput (jobs/h) | Placeable idle GPU-seconds while quota-blocked, per run |
|---:|---:|---:|---:|---:|
| Unlimited | 2,026 ± 59 [1,999, 2,054] | 98.07 ± 0.44 [97.87, 98.28] | 156.72 ± 1.92 [155.83, 157.62] | 0 |
| 1 | 20,957 ± 342 [20,797, 21,117] | 12.50 ± 0.00 [12.50, 12.50] | 19.98 ± 0.23 [19.87, 20.08] | 301,496 ± 3,604 [299,809, 303,182] |
| 2 | 10,132 ± 178 [10,048, 10,215] | 24.95 ± 0.03 [24.94, 24.96] | 39.87 ± 0.48 [39.65, 40.10] | 128,879 ± 1,637 [128,113, 129,644] |
| 4 | 4,723 ± 97 [4,678, 4,768] | 49.65 ± 0.14 [49.59, 49.72] | 79.35 ± 0.97 [78.90, 79.80] | 42,743 ± 571 [42,475, 43,010] |

This directly exposes the hard ceiling: with one active user, limits of 1, 2, and 4 cap useful concurrency at 1/8, 2/8, and 4/8 of the GPUs. The waiting queue cannot borrow idle GPUs because no other user has jobs. The per-run idle GPU-second measure agrees with that mechanism.

### Figures

- [Heavy-user mean wait](../../outputs/benchmarks/quota/figures/quota_heavy_user_wait.svg)
- [Normal/light-user mean wait](../../outputs/benchmarks/quota/figures/quota_normal_user_wait.svg)
- [GPU utilization](../../outputs/benchmarks/quota/figures/quota_gpu_utilisation.svg)
- [Heavy-to-normal waiting-time ratio](../../outputs/benchmarks/quota/figures/quota_wait_ratio.svg)
- [Completed-job throughput](../../outputs/benchmarks/quota/figures/quota_throughput.svg)

Points show across-seed means and bars show the two-sided 95% t confidence intervals. The checked-in `summary.csv` contains the plotted means and intervals.

## Interpretation

**RQ1 — Does quota improve fairness under dominant arrivals?** It depends on what fairness means. In B, FIFO already gives the 70%-arrival user and the other users nearly equal mean wait. Quotas 1/2/4 reduce light-user waits but make the wait distribution less equal and cause the heavy user to wait longer. In C, quota 4 moves the light users' wait substantially toward the hog user's, while quotas 1 and 2 overshoot and make the hog user wait much longer. The data support workload-specific isolation, not a general fairness improvement claim.

**RQ2 — What does it cost?** B's quota-4 setting gives lighter users a lower mean wait, with 69.8% utilization and 111.5 jobs/h versus 98.1% and 156.7/h without quota. C's quota-4 setting retains 77.1% utilization and 123.2 jobs/h while cutting light-user mean wait from 4,456 s to 1,886 s. Quotas 1 and 2 have much larger costs in all four workloads.

**RQ3 — How sensitive is the trade-off?** Strongly. Quota 4 has little effect on balanced A, is a possible isolation compromise for B/C when that objective justifies the throughput cost, and still wastes half the cluster in D. Quota 1 or 2 is not a sensible general default from these runs.

For this simulated B/C mix, **quota 4 is the most defensible compromise among the tested limits only when reducing light-user waiting is an explicit priority**. It is not a universal recommendation: B's mean-wait equality worsens, its throughput drops about 29%, and D demonstrates a much larger cost when demand is not shared.

Quota and fair-share are different mechanisms. A hard quota is a concurrent resource ceiling: it prevents one user from holding more than N GPUs at once and can leave capacity idle. Fair-share is a soft queue preference based on historical/current usage; it can choose other users' work first without forbidding the heavy user from using otherwise-idle GPUs. This PR's primary matrix keeps FIFO fixed and does **not** measure their interaction.

## Threats to Validity and Limitations

- These are simulator results, not measurements on a real multi-GPU cluster.
- Workloads are synthetic and the conclusions depend on their arrival rates, user mix, job duration, and the chosen 8-GPU topology.
- Scenario C's subsecond exponential interarrivals are floored to an integer-second simulator clock; this creates tied arrival timestamps and is part of the recorded workload definition.
- Quota identity uses the job's `submitted_by` user string. The controller endpoint has no authentication in this study; quota is not an identity or security boundary.
- There is no preemption. Running jobs keep their GPUs until completion, so quotas govern future admission and queued work.
- All jobs complete and retain their submitted GPU-seconds. Jain GPU-seconds therefore describes demand, not who received service sooner; the wait metrics are more informative for these runs.
- The wait-time ratio intentionally compares the named dominant/burst user to an unweighted average of the other users. It is directional and does not encode policy entitlements.
- Throughput and utilization are coupled because every run drains the same fixed jobs and service durations; neither is an independent validation of the other.
- The optional fair-share × quota interaction experiment and mixed-size workload were not included. The primary matrix isolates quota under a fixed FIFO baseline.

## Reproduction

From a clean checkout with Rust and the repository's Python environment:

```bash
./scripts/run_quota_experiments.sh
```

The script builds the release simulator, runs the 320 cases, writes local raw and aggregate files, then generates the five SVGs. The large raw CSV/JSON are ignored by Git and can be regenerated by this script. The committed results are:

- [`summary.csv`](../../outputs/benchmarks/quota/summary.csv) and [`summary.json`](../../outputs/benchmarks/quota/summary.json) — 1,176 scenario/quota/metric aggregates with n, mean, sample SD, and 95% CI, including paired quota-minus-unlimited differences;
- five figures under [`figures/`](../../outputs/benchmarks/quota/figures/).

To regenerate only the aggregates from raw JSON:

```bash
./target/release/ferro-sim quota-aggregate \
  --input outputs/benchmarks/quota/raw.json \
  --out outputs/benchmarks/quota
```

The independent reaggregation reproduced both summary files byte-for-byte. The full experiment was run twice from the same clean code revision; raw CSV/JSON, summary CSV/JSON, and all five SVGs were byte-identical. Every main-matrix row completed its submitted jobs with zero failures, never-started jobs, or hard quota rejections.

The raw `git_commit` value (`61a188a`) is captured before the experiment creates its output directory. A later rerun from a newer checkout records that newer revision in this metadata field, so raw files will differ there even when all simulated fields, summaries, and plots are unchanged. For byte-for-byte comparison, rerun from the recorded code revision; otherwise compare the simulation fields and regenerated summaries.
