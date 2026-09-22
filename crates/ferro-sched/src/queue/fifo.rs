//! First come, first served.
//!
//! The baseline every other policy is measured against: deterministic, trivial
//! to explain, and incapable of starving anyone through priority. Its known
//! weakness is head-of-line blocking -- a job wanting four GPUs holds up the
//! one-GPU job behind it even while a card sits idle -- which is exactly what
//! the backfilling experiments later in the project are for.

use super::{QueueContext, QueuePolicy, QueuedJob};

#[derive(Debug, Default, Clone, Copy)]
pub struct Fifo;

impl QueuePolicy for Fifo {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn rank(&self, jobs: &[QueuedJob], _ctx: &QueueContext) -> Vec<String> {
        let mut ordered: Vec<&QueuedJob> = jobs.iter().collect();
        // Submission sequence, not timestamp: jobs submitted within the same
        // second must still have one unambiguous order.
        ordered.sort_by_key(|j| j.order_seq);
        ordered.into_iter().map(|j| j.job_id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, seq: u64, submitted: i64) -> QueuedJob {
        QueuedJob {
            job_id: id.into(),
            order_seq: seq,
            submitted_unix_s: submitted,
            submitted_by: "tester".into(),
        }
    }

    fn rank(jobs: &[QueuedJob]) -> Vec<String> {
        Fifo.rank(jobs, &QueueContext { now: 1_000 })
    }

    #[test]
    fn orders_by_submission_sequence() {
        let jobs = vec![job("c", 2, 10), job("a", 0, 10), job("b", 1, 10)];
        assert_eq!(rank(&jobs), vec!["a", "b", "c"]);
    }

    #[test]
    fn a_shared_timestamp_still_has_an_order() {
        // Every job submitted in the same second. Timestamp ordering would be
        // ambiguous here; the sequence number is not.
        let jobs = vec![job("second", 1, 42), job("first", 0, 42)];
        assert_eq!(rank(&jobs), vec!["first", "second"]);
    }

    #[test]
    fn ranking_is_stable_across_repeated_calls() {
        let jobs = vec![job("x", 5, 1), job("y", 3, 1), job("z", 9, 1)];
        let first = rank(&jobs);
        for _ in 0..8 {
            assert_eq!(rank(&jobs), first, "equal inputs must rank identically");
        }
    }

    #[test]
    fn an_empty_queue_ranks_to_nothing() {
        assert!(rank(&[]).is_empty());
    }
}
