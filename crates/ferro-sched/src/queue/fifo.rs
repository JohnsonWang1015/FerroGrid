//! First come, first served.
//!
//! The baseline every other policy is measured against: deterministic, trivial
//! to explain, and incapable of starving anyone through priority. Its known
//! weakness is head-of-line blocking -- a job wanting four GPUs holds up the
//! one-GPU job behind it even while a card sits idle -- which is exactly what
//! the backfilling experiments later in the project are for.

use super::{by_score_then_arrival, QueueContext, QueuePolicy, QueueRanking, QueuedJob};

#[derive(Debug, Default, Clone, Copy)]
pub struct Fifo;

impl QueuePolicy for Fifo {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn rank(&self, jobs: &[QueuedJob], _ctx: &QueueContext<'_>) -> Vec<QueueRanking> {
        let scored = jobs
            .iter()
            .map(|j| {
                (
                    j.order_seq,
                    QueueRanking {
                        job_id: j.job_id.clone(),
                        // Submission sequence, negated so that "higher runs
                        // sooner" holds here as it does everywhere else.
                        score: -(j.order_seq as f64),
                        components: vec![("arrival", j.order_seq as f64)],
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

    #[test]
    fn orders_by_submission_sequence() {
        let jobs = vec![job("c", 2), job("a", 0), job("b", 1)];
        assert_eq!(ids(&rank_with(&Fifo, &jobs, 1_000)), ["a", "b", "c"]);
    }

    #[test]
    fn a_shared_timestamp_still_has_an_order() {
        // Both submitted in the same second. Timestamp ordering would be
        // ambiguous here; the sequence number is not.
        let mut first = job("first", 0);
        let mut second = job("second", 1);
        first.submitted_unix_s = 42;
        second.submitted_unix_s = 42;
        let ranked = rank_with(&Fifo, &[second, first], 1_000);
        assert_eq!(ids(&ranked), ["first", "second"]);
    }

    #[test]
    fn priority_is_ignored() {
        // The whole point of the baseline: it does not matter who shouts.
        let mut loud = job("loud", 1);
        loud.priority = 100;
        let mut quiet = job("quiet", 0);
        quiet.priority = 0;
        assert_eq!(
            ids(&rank_with(&Fifo, &[loud, quiet], 1_000)),
            ["quiet", "loud"]
        );
    }

    #[test]
    fn ranking_is_stable_across_repeated_calls() {
        let jobs = vec![job("x", 5), job("y", 3), job("z", 9)];
        let first = rank_with(&Fifo, &jobs, 1_000);
        for _ in 0..8 {
            assert_eq!(
                rank_with(&Fifo, &jobs, 1_000),
                first,
                "equal inputs must rank identically"
            );
        }
    }

    #[test]
    fn an_empty_queue_ranks_to_nothing() {
        assert!(rank_with(&Fifo, &[], 1_000).is_empty());
    }
}
