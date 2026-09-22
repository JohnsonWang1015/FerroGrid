//! Strict priority: the highest number goes first.
//!
//! The classic trade. It gets urgent work through a full cluster, and it will
//! starve the bottom of the queue for as long as anyone keeps submitting above
//! it -- nothing here ever raises a waiting job. That is not an oversight; it
//! is the property [`super::aging`] exists to fix, and keeping the two apart is
//! what lets the experiments show the difference.

use super::{
    by_score_then_arrival, QueueContext, QueuePolicy, QueueRanking, QueuedJob, MAX_PRIORITY,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct Priority;

impl QueuePolicy for Priority {
    fn name(&self) -> &'static str {
        "priority"
    }

    fn rank(&self, jobs: &[QueuedJob], _ctx: &QueueContext<'_>) -> Vec<QueueRanking> {
        let scored = jobs
            .iter()
            .map(|j| {
                let p = j.priority.min(MAX_PRIORITY) as f64;
                (
                    j.order_seq,
                    QueueRanking {
                        job_id: j.job_id.clone(),
                        score: p,
                        components: vec![("priority", p)],
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

    fn at(id: &str, seq: u64, priority: u32) -> QueuedJob {
        QueuedJob {
            priority,
            ..job(id, seq)
        }
    }

    #[test]
    fn higher_priority_runs_first() {
        let jobs = vec![at("low", 0, 10), at("high", 1, 90), at("mid", 2, 50)];
        assert_eq!(
            ids(&rank_with(&Priority, &jobs, 1_000)),
            ["high", "mid", "low"]
        );
    }

    #[test]
    fn equal_priority_falls_back_to_arrival() {
        // Without this the order would depend on hash iteration, and a user
        // would be quoted a position the dispatcher does not honour.
        let jobs = vec![at("third", 2, 50), at("first", 0, 50), at("second", 1, 50)];
        assert_eq!(
            ids(&rank_with(&Priority, &jobs, 1_000)),
            ["first", "second", "third"]
        );
    }

    #[test]
    fn priority_is_clamped_to_the_documented_range() {
        // A caller that sends 4 billion does not get to outrank everyone
        // forever; the dial is 0-100 and the scheduler holds it to that.
        let jobs = vec![at("sane", 0, 100), at("absurd", 1, u32::MAX)];
        let ranked = rank_with(&Priority, &jobs, 1_000);
        assert_eq!(ranked[0].score, ranked[1].score);
        assert_eq!(ids(&ranked), ["sane", "absurd"], "tie breaks on arrival");
    }

    #[test]
    fn a_low_priority_job_never_advances_on_its_own() {
        // The starvation property, asserted rather than assumed: this is the
        // baseline that aging has to beat.
        let jobs = vec![at("starved", 0, 1), at("loud", 1, 99)];
        for now in [0, 3_600, 86_400, 30 * 86_400] {
            assert_eq!(
                ids(&rank_with(&Priority, &jobs, now)),
                ["loud", "starved"],
                "strict priority must not age; at t={now} it still has not"
            );
        }
    }

    #[test]
    fn ranking_is_deterministic() {
        let jobs = vec![at("a", 0, 70), at("b", 1, 70), at("c", 2, 30)];
        let first = rank_with(&Priority, &jobs, 500);
        for _ in 0..8 {
            assert_eq!(rank_with(&Priority, &jobs, 500), first);
        }
    }
}
