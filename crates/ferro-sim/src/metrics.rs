//! Turning a simulated run into the numbers §54 asks for.
//!
//! Every definition here is a choice, and a comparison is only meaningful if
//! the choices are stated. Where a metric could reasonably mean two things,
//! the doc comment says which one this is and why.

use crate::sim::SimOutcome;
use ferro_sched::jain_index;
use serde::Serialize;
use std::collections::BTreeMap;

/// A job is starved if it waited more than this many times its own runtime.
///
/// The first attempt at this counted how often a job was overtaken, which
/// turned out to measure the wrong thing: **any** policy that reorders the
/// queue overtakes somebody, so priority and fair share scored badly for doing
/// exactly what they were asked to do, while FIFO scored zero by construction.
/// That is a definition that decides the result before the experiment runs.
///
/// Waiting ten times as long as you will run for is bad treatment whoever
/// caused it, is scale-free across job sizes, and is policy-neutral -- so it
/// can be compared across policies without flattering any of them.
pub const STARVATION_WAIT_RATIO: f64 = 10.0;

/// Being overtaken this often is still worth reporting as a mechanism, but it
/// is a diagnostic rather than a verdict.
pub const OVERTAKEN_OFTEN: u32 = 5;

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub jobs_submitted: usize,
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    /// Still in the queue when the run ended, having never started.
    pub jobs_never_started: usize,

    pub avg_wait_s: f64,
    pub p50_wait_s: f64,
    pub p95_wait_s: f64,
    pub max_wait_s: f64,
    pub avg_turnaround_s: f64,
    pub p95_turnaround_s: f64,

    /// First arrival to last completion.
    pub makespan_s: i64,
    /// GPU-seconds actually used over GPU-seconds available across the
    /// makespan. Capacity that existed but sat idle counts against it.
    pub gpu_utilisation: f64,
    pub throughput_per_hour: f64,

    /// Jain's index over per-user GPU-seconds, as §7 asks for. Note what it
    /// can and cannot show: once a queue has drained, what each user *received*
    /// equals what they *demanded*, so on a completed run this number describes
    /// the workload more than the policy.
    pub jain_fairness: f64,
    /// Jain's index over per-user **mean waiting time**. This is the one a
    /// queue policy can actually move: fair share cannot give a user more GPU
    /// than they asked for, only make them wait longer for it.
    ///
    /// Read it with care. A low value means users were treated *differently*,
    /// not that they were treated *unjustly* -- fair share deliberately makes
    /// the heaviest consumer wait longer, and scores worse here for doing so.
    /// [`Metrics::heavy_user_wait_ratio`] says which direction the difference
    /// went, which is the question actually being asked.
    pub wait_fairness: f64,
    /// Mean wait of the user who consumed the most GPU-seconds, divided by the
    /// mean wait of everyone else.
    ///
    /// About 1.0 means the cluster does not care how much you have already
    /// used. Above 1.0 means the heaviest user waits longer than the others,
    /// which is what fair share is for. Below 1.0 means the greediest user is
    /// also the best served, which is the failure worth catching.
    pub heavy_user_wait_ratio: f64,
    pub starved_jobs: usize,
    pub jobs_overtaken_often: usize,
    pub max_overtaken_by: u32,

    /// Mean microseconds per scheduling pass, measured.
    pub scheduler_us_per_pass: f64,
    pub scheduling_passes: u64,
    /// Time-weighted mean of free GPUs no single node could offer as a block.
    pub fragmentation: f64,
    /// Mean of (time actually run) / (time on reference hardware).
    ///
    /// 1.0 means every job was placed on hardware as good as the workload was
    /// quoted against. Above 1.0 is what the placement decision cost -- slower
    /// cards, or a network hop the job had to cross. This is the number a
    /// placement comparison turns on, and it is 1.0 for every strategy when the
    /// run is made with a flat execution model.
    pub mean_slowdown: f64,

    pub per_user_gpu_seconds: BTreeMap<String, f64>,
    pub per_user_avg_wait_s: BTreeMap<String, f64>,
    pub per_user_jobs: BTreeMap<String, usize>,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    // Nearest-rank, which needs no interpolation and cannot invent a value
    // that no job actually had.
    let rank = (p / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

pub fn summarise(out: &SimOutcome) -> Metrics {
    let completed: Vec<_> = out
        .jobs
        .iter()
        .filter(|j| j.end_s.is_some() && !j.failed)
        .collect();

    // Waiting time is only defined for a job that started. Counting a job that
    // never ran as "waited until the end" would quietly reward a policy for
    // refusing to start things.
    let mut waits: Vec<f64> = out
        .jobs
        .iter()
        .filter(|j| j.start_s.is_some())
        .map(|j| j.wait_s as f64)
        .collect();
    waits.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut turnarounds: Vec<f64> = completed.iter().map(|j| j.turnaround_s as f64).collect();
    turnarounds.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let first_arrival = out.jobs.iter().map(|j| j.arrival_s).min().unwrap_or(0);
    let last_end = out
        .jobs
        .iter()
        .filter_map(|j| j.end_s)
        .max()
        .unwrap_or(first_arrival);
    let makespan = (last_end - first_arrival).max(0);

    let gpu_seconds_used: f64 = out
        .jobs
        .iter()
        .filter_map(|j| Some((j.start_s?, j.end_s?, j.gpus)))
        .map(|(s, e, g)| (e - s).max(0) as f64 * g as f64)
        .sum();
    let gpu_seconds_available = out.total_gpus as f64 * makespan as f64;
    let utilisation = if gpu_seconds_available > 0.0 {
        (gpu_seconds_used / gpu_seconds_available).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut per_user: BTreeMap<String, f64> = BTreeMap::new();
    for j in &out.jobs {
        if let (Some(s), Some(e)) = (j.start_s, j.end_s) {
            *per_user.entry(j.user.clone()).or_default() += (e - s).max(0) as f64 * j.gpus as f64;
        } else {
            // Present with zero, so a user who got nothing still counts
            // against fairness rather than vanishing from it.
            per_user.entry(j.user.clone()).or_default();
        }
    }
    let shares: Vec<f64> = per_user.values().copied().collect();

    // Per-user waiting, which is where a queue policy's effect actually shows.
    let mut waits_by_user: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut jobs_by_user: BTreeMap<String, usize> = BTreeMap::new();
    for j in &out.jobs {
        *jobs_by_user.entry(j.user.clone()).or_default() += 1;
        if j.start_s.is_some() {
            waits_by_user
                .entry(j.user.clone())
                .or_default()
                .push(j.wait_s as f64);
        }
    }
    let per_user_wait: BTreeMap<String, f64> = waits_by_user
        .iter()
        .map(|(u, xs)| (u.clone(), mean(xs)))
        .collect();
    let wait_shares: Vec<f64> = per_user_wait.values().copied().collect();

    // Does the heaviest consumer wait longer than everyone else, or less?
    let heaviest = per_user
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(u, _)| u.clone());
    let heavy_ratio = match heaviest {
        Some(ref who) if per_user_wait.len() > 1 => {
            let theirs = per_user_wait.get(who).copied().unwrap_or(0.0);
            let others: Vec<f64> = per_user_wait
                .iter()
                .filter(|(u, _)| *u != who)
                .map(|(_, w)| *w)
                .collect();
            let other = mean(&others);
            if other > 0.0 {
                theirs / other
            } else {
                1.0
            }
        }
        // One user, or nobody: there is nobody to be treated differently from.
        _ => 1.0,
    };

    Metrics {
        jobs_submitted: out.jobs.len(),
        jobs_completed: completed.len(),
        jobs_failed: out.jobs.iter().filter(|j| j.failed).count(),
        jobs_never_started: out.jobs.iter().filter(|j| j.start_s.is_none()).count(),

        avg_wait_s: mean(&waits),
        p50_wait_s: percentile(&waits, 50.0),
        p95_wait_s: percentile(&waits, 95.0),
        max_wait_s: waits.last().copied().unwrap_or(0.0),
        avg_turnaround_s: mean(&turnarounds),
        p95_turnaround_s: percentile(&turnarounds, 95.0),

        makespan_s: makespan,
        gpu_utilisation: utilisation,
        throughput_per_hour: if makespan > 0 {
            completed.len() as f64 * 3600.0 / makespan as f64
        } else {
            0.0
        },

        jain_fairness: jain_index(&shares).unwrap_or(0.0),
        wait_fairness: jain_index(&wait_shares).unwrap_or(1.0),
        heavy_user_wait_ratio: heavy_ratio,
        starved_jobs: out
            .jobs
            .iter()
            .filter(|j| j.start_s.is_some())
            .filter(|j| j.wait_s as f64 > STARVATION_WAIT_RATIO * j.duration_s.max(1) as f64)
            .count()
            // A job that never ran at all is the limiting case of starvation.
            + out.jobs.iter().filter(|j| j.start_s.is_none()).count(),
        jobs_overtaken_often: out
            .jobs
            .iter()
            .filter(|j| j.overtaken_by >= OVERTAKEN_OFTEN)
            .count(),
        max_overtaken_by: out.jobs.iter().map(|j| j.overtaken_by).max().unwrap_or(0),

        scheduler_us_per_pass: if out.scheduling_passes > 0 {
            out.scheduler_ns as f64 / out.scheduling_passes as f64 / 1_000.0
        } else {
            0.0
        },
        scheduling_passes: out.scheduling_passes,
        fragmentation: out.fragmentation,
        mean_slowdown: {
            let ratios: Vec<f64> = out
                .jobs
                .iter()
                .filter(|j| j.start_s.is_some() && j.nominal_duration_s > 0)
                .map(|j| j.duration_s as f64 / j.nominal_duration_s as f64)
                .collect();
            if ratios.is_empty() {
                1.0
            } else {
                mean(&ratios)
            }
        },

        per_user_gpu_seconds: per_user,
        per_user_avg_wait_s: per_user_wait,
        per_user_jobs: jobs_by_user,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::JobRecord;

    fn record(id: &str, user: &str, arrival: i64, start: i64, dur: i64, gpus: u32) -> JobRecord {
        JobRecord {
            id: id.into(),
            user: user.into(),
            class: "c".into(),
            priority: 50,
            gpus,
            arrival_s: arrival,
            start_s: Some(start),
            end_s: Some(start + dur),
            duration_s: dur,
            nominal_duration_s: dur,
            wait_s: start - arrival,
            turnaround_s: start + dur - arrival,
            overtaken_by: 0,
            nodes_used: vec!["gpu-a".into()],
            failed: false,
        }
    }

    fn outcome(jobs: Vec<JobRecord>, total_gpus: usize) -> SimOutcome {
        SimOutcome {
            jobs,
            trace: Vec::new(),
            scheduler_ns: 2_000_000,
            scheduling_passes: 4,
            fragmentation: 1.5,
            total_gpus,
        }
    }

    #[test]
    fn a_worked_example_comes_out_as_hand_computed() {
        // Two jobs on a 2-GPU cluster. a: arrives 0, starts 0, runs 10, 1 GPU.
        //                              b: arrives 0, starts 10, runs 10, 1 GPU.
        // makespan 20; used = 10 + 10 = 20 GPU-s; available = 2 * 20 = 40.
        let m = summarise(&outcome(
            vec![
                record("a", "u", 0, 0, 10, 1),
                record("b", "u", 0, 10, 10, 1),
            ],
            2,
        ));
        assert_eq!(m.makespan_s, 20);
        assert!(
            (m.gpu_utilisation - 0.5).abs() < 1e-9,
            "{}",
            m.gpu_utilisation
        );
        assert!((m.avg_wait_s - 5.0).abs() < 1e-9);
        assert_eq!(m.p50_wait_s, 0.0);
        assert_eq!(m.max_wait_s, 10.0);
        assert!((m.avg_turnaround_s - 15.0).abs() < 1e-9);
        assert_eq!(m.jobs_completed, 2);
    }

    #[test]
    fn utilisation_is_one_when_nothing_is_idle() {
        // Two GPUs, both busy for the whole span.
        let m = summarise(&outcome(
            vec![
                record("a", "u", 0, 0, 100, 1),
                record("b", "u", 0, 0, 100, 1),
            ],
            2,
        ));
        assert!((m.gpu_utilisation - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fairness_is_one_when_two_users_got_the_same() {
        let m = summarise(&outcome(
            vec![
                record("a", "alice", 0, 0, 50, 1),
                record("b", "bob", 0, 0, 50, 1),
            ],
            2,
        ));
        assert!((m.jain_fairness - 1.0).abs() < 1e-9);
        assert_eq!(m.per_user_gpu_seconds["alice"], 50.0);
    }

    #[test]
    fn fairness_falls_when_one_user_takes_everything() {
        let m = summarise(&outcome(
            vec![
                record("a", "alice", 0, 0, 100, 2),
                record("b", "bob", 0, 0, 0, 1),
            ],
            2,
        ));
        assert!(m.jain_fairness < 0.6, "got {}", m.jain_fairness);
    }

    #[test]
    fn a_user_who_got_nothing_still_counts_against_fairness() {
        // Dropping them would flatter the index by measuring only the winners.
        let mut jobs = vec![record("a", "alice", 0, 0, 100, 1)];
        let mut starved = record("b", "bob", 0, 0, 0, 1);
        starved.start_s = None;
        starved.end_s = None;
        jobs.push(starved);
        let m = summarise(&outcome(jobs, 2));
        assert!(m.per_user_gpu_seconds.contains_key("bob"));
        assert_eq!(m.per_user_gpu_seconds["bob"], 0.0);
        assert!(m.jain_fairness < 0.6, "got {}", m.jain_fairness);
        assert_eq!(m.jobs_never_started, 1);
    }

    #[test]
    fn a_job_that_never_started_has_no_waiting_time_to_report() {
        // Counting it as "waited forever" would reward a policy for refusing
        // to start things, which is the opposite of what we are measuring.
        let mut never = record("b", "bob", 0, 0, 0, 1);
        never.start_s = None;
        never.end_s = None;
        never.wait_s = 0;
        let m = summarise(&outcome(vec![record("a", "u", 0, 0, 10, 1), never], 2));
        assert!((m.avg_wait_s - 0.0).abs() < 1e-9);
        assert_eq!(m.jobs_never_started, 1);
        assert_eq!(m.jobs_completed, 1);
    }

    #[test]
    fn percentiles_never_invent_a_value() {
        let waits = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(percentile(&waits, 50.0), 3.0);
        assert_eq!(percentile(&waits, 95.0), 100.0);
        assert_eq!(percentile(&waits, 100.0), 100.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
        for p in [1.0, 25.0, 50.0, 75.0, 99.0, 100.0] {
            assert!(waits.contains(&percentile(&waits, p)));
        }
    }

    #[test]
    fn starvation_is_waiting_far_longer_than_you_will_run_for() {
        // 10s of work after a 10,000s wait is starvation whoever caused it.
        let starved = record("starved", "u", 0, 10_000, 10, 1);
        // A long wait in absolute terms, but proportionate to a long job.
        let patient = record("patient", "u", 0, 200, 5_000, 1);
        let m = summarise(&outcome(vec![starved, patient], 2));
        assert_eq!(m.starved_jobs, 1);
    }

    #[test]
    fn reordering_the_queue_is_not_by_itself_starvation() {
        // The definition must not decide the comparison in advance: a policy
        // that overtakes somebody and then runs them promptly has not starved
        // anyone, and an earlier version of this metric said otherwise.
        let mut overtaken = record("overtaken", "u", 0, 5, 100, 1);
        overtaken.overtaken_by = 20;
        let m = summarise(&outcome(vec![overtaken], 2));
        assert_eq!(m.starved_jobs, 0);
        assert_eq!(m.jobs_overtaken_often, 1, "still reported as a mechanism");
        assert_eq!(m.max_overtaken_by, 20);
    }

    #[test]
    fn a_job_that_never_ran_is_the_limiting_case_of_starvation() {
        let mut never = record("never", "u", 0, 0, 10, 1);
        never.start_s = None;
        never.end_s = None;
        let m = summarise(&outcome(vec![never], 2));
        assert_eq!(m.starved_jobs, 1);
    }

    #[test]
    fn the_heavy_user_ratio_says_which_way_the_difference_went() {
        // alice consumed the most and waited the longest for it: above 1.0,
        // which is what a fair-share policy is supposed to produce.
        let mut alice = record("a", "alice", 0, 400, 1_000, 1);
        alice.wait_s = 400;
        let mut bob = record("b", "bob", 0, 100, 10, 1);
        bob.wait_s = 100;
        let m = summarise(&outcome(vec![alice, bob], 2));
        assert!(
            m.heavy_user_wait_ratio > 3.0,
            "got {}",
            m.heavy_user_wait_ratio
        );

        // The reverse: the greediest user is also the best served.
        let mut greedy = record("a", "alice", 0, 10, 1_000, 1);
        greedy.wait_s = 10;
        let mut polite = record("b", "bob", 0, 500, 10, 1);
        polite.wait_s = 500;
        let m = summarise(&outcome(vec![greedy, polite], 2));
        assert!(
            m.heavy_user_wait_ratio < 0.5,
            "got {}",
            m.heavy_user_wait_ratio
        );
    }

    #[test]
    fn a_single_user_is_not_treated_unequally() {
        let m = summarise(&outcome(vec![record("a", "solo", 0, 50, 10, 1)], 2));
        assert_eq!(m.heavy_user_wait_ratio, 1.0);
    }

    #[test]
    fn wait_fairness_sees_what_gpu_second_fairness_cannot() {
        // Both users ran the same total work, so GPU-second fairness is
        // perfect -- but one of them waited ten times as long for it, which is
        // exactly the unfairness a queue policy creates and fixes.
        let mut alice = record("a", "alice", 0, 10, 100, 1);
        alice.wait_s = 10;
        let mut bob = record("b", "bob", 0, 1_000, 100, 1);
        bob.wait_s = 1_000;
        let m = summarise(&outcome(vec![alice, bob], 2));
        assert!(
            (m.jain_fairness - 1.0).abs() < 1e-9,
            "GPU-seconds look fair"
        );
        assert!(
            m.wait_fairness < 0.7,
            "waiting was not fair at all: {}",
            m.wait_fairness
        );
        assert_eq!(m.per_user_avg_wait_s["bob"], 1_000.0);
    }

    #[test]
    fn scheduler_overhead_is_reported_per_pass() {
        let m = summarise(&outcome(vec![record("a", "u", 0, 0, 10, 1)], 2));
        // 2_000_000 ns over 4 passes = 500 us each.
        assert!((m.scheduler_us_per_pass - 500.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_run_does_not_divide_by_zero() {
        let m = summarise(&SimOutcome {
            jobs: Vec::new(),
            trace: Vec::new(),
            scheduler_ns: 0,
            scheduling_passes: 0,
            fragmentation: 0.0,
            total_gpus: 4,
        });
        assert_eq!(m.jobs_submitted, 0);
        assert_eq!(m.gpu_utilisation, 0.0);
        assert_eq!(m.throughput_per_hour, 0.0);
        assert_eq!(m.scheduler_us_per_pass, 0.0);
    }
}
