//! Shortest job first, over *declared* durations.
//!
//! SJF minimises mean waiting time when the durations are known. On a GPU
//! cluster they are not: the only source is `--estimated-duration`, which is a
//! claim, not a measurement.
//!
//! So this policy is explicit about what it does not know. A job with no
//! estimate is ordered **last**, as though it were maximally long. That is a
//! stated, conservative fallback rather than a guess: guessing short would let
//! any job jump the queue by declining to answer, and guessing an average would
//! be inventing data. It follows that unestimated jobs can starve behind a
//! stream of short ones — which is the honest result, and the reason this
//! policy is offered for experiments rather than as a default.
//!
//! The experiment worth running is SJF against FIFO on mean waiting time, and
//! then SJF's long-job starvation against aging.

use super::{by_score_then_arrival, QueueContext, QueuePolicy, QueueRanking, QueuedJob};

/// What an unestimated job is treated as lasting. Not an estimate of anything:
/// a sentinel chosen so such jobs sort behind every declared duration.
pub const UNKNOWN_DURATION_S: u32 = u32::MAX;

#[derive(Debug, Default, Clone, Copy)]
pub struct Sjf;

impl QueuePolicy for Sjf {
    fn name(&self) -> &'static str {
        "sjf"
    }

    fn rank(&self, jobs: &[QueuedJob], _ctx: &QueueContext<'_>) -> Vec<QueueRanking> {
        let scored = jobs
            .iter()
            .map(|j| {
                let declared = j.estimated_duration_s;
                let assumed = declared.unwrap_or(UNKNOWN_DURATION_S);
                (
                    j.order_seq,
                    QueueRanking {
                        job_id: j.job_id.clone(),
                        // Negated: shorter is better, and "higher runs sooner"
                        // has to hold across every policy.
                        score: -(assumed as f64),
                        components: vec![(
                            if declared.is_some() {
                                "estimated seconds"
                            } else {
                                "no estimate; assumed longest"
                            },
                            assumed as f64,
                        )],
                    },
                )
            })
            .collect();
        by_score_then_arrival(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::tests_support::{ids, job, rank_with};

    fn lasting(id: &str, seq: u64, seconds: Option<u32>) -> QueuedJob {
        QueuedJob {
            estimated_duration_s: seconds,
            ..job(id, seq)
        }
    }

    #[test]
    fn the_shortest_declared_job_runs_first() {
        let jobs = vec![
            lasting("long", 0, Some(3_600)),
            lasting("short", 1, Some(60)),
            lasting("medium", 2, Some(600)),
        ];
        assert_eq!(ids(&rank_with(&Sjf, &jobs, 0)), ["short", "medium", "long"]);
    }

    #[test]
    fn an_unestimated_job_goes_last_and_is_labelled_as_such() {
        // The honesty requirement: no fabricated duration, and the reason is
        // visible in the explanation rather than buried in the ordering.
        let jobs = vec![lasting("unknown", 0, None), lasting("hour", 1, Some(3_600))];
        let ranked = rank_with(&Sjf, &jobs, 0);
        assert_eq!(ids(&ranked), ["hour", "unknown"]);
        assert_eq!(
            ranked[1].components[0].0, "no estimate; assumed longest",
            "the policy must say that it is guessing nothing"
        );
    }

    #[test]
    fn unestimated_jobs_keep_arrival_order_among_themselves() {
        let jobs = vec![
            lasting("c", 2, None),
            lasting("a", 0, None),
            lasting("b", 1, None),
        ];
        assert_eq!(ids(&rank_with(&Sjf, &jobs, 0)), ["a", "b", "c"]);
    }

    #[test]
    fn declaring_zero_is_not_the_same_as_declaring_nothing() {
        // proto3 would collapse these if the field were not `optional`; the
        // ordering proves the distinction survived all the way down.
        let jobs = vec![lasting("silent", 0, None), lasting("instant", 1, Some(0))];
        assert_eq!(ids(&rank_with(&Sjf, &jobs, 0)), ["instant", "silent"]);
    }

    #[test]
    fn a_long_job_starves_behind_a_stream_of_short_ones() {
        // Asserted, not glossed over: this is the weakness the aging experiment
        // is meant to expose, so it should be visible in the tests too.
        let long = lasting("long", 0, Some(86_400));
        let mut queue = vec![long.clone()];
        for n in 1..=20 {
            queue.push(lasting(&format!("short{n}"), n, Some(30)));
        }
        let ranked = rank_with(&Sjf, &queue, 100_000);
        assert_eq!(
            ranked.last().unwrap().job_id,
            "long",
            "however long it waits, SJF alone never advances it"
        );
    }

    #[test]
    fn ranking_is_deterministic() {
        let jobs = vec![
            lasting("a", 0, Some(100)),
            lasting("b", 1, Some(100)),
            lasting("c", 2, None),
        ];
        let first = rank_with(&Sjf, &jobs, 0);
        for _ in 0..8 {
            assert_eq!(rank_with(&Sjf, &jobs, 0), first);
        }
    }
}
