//! Priority with aging: waiting is itself a claim on the cluster.
//!
//! ```text
//! effective = min(base + floor(waited / interval) * increment, ceiling)
//! ```
//!
//! The floor division matters. A continuous bonus would reorder the queue on
//! every tick as fractional seconds accumulated, so two jobs would swap places
//! repeatedly without either making progress. Stepping once per interval means
//! a position holds until something real changes.
//!
//! The ceiling matters too: without it a job that waited long enough would
//! outrank a genuine emergency, which is the failure mode aging is supposed to
//! prevent, arrived at from the other direction.

use super::{
    by_score_then_arrival, QueueContext, QueuePolicy, QueueRanking, QueuedJob, MAX_PRIORITY,
};

/// All three knobs are configurable because the right values depend entirely on
/// how long the cluster's jobs run. An interval of ten minutes is generous for
/// a queue of five-minute jobs and meaningless for a queue of three-day ones.
#[derive(Debug, Clone, Copy)]
pub struct AgingConfig {
    /// Seconds of waiting that earn one step of `increment`.
    pub interval_s: u32,
    /// Priority added per completed interval.
    pub increment: u32,
    /// Effective priority may not exceed this, however long the wait.
    pub ceiling: u32,
}

impl Default for AgingConfig {
    fn default() -> Self {
        Self {
            interval_s: 60,
            increment: 1,
            ceiling: MAX_PRIORITY,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Aging {
    pub config: AgingConfig,
}

impl Aging {
    pub fn new(config: AgingConfig) -> Self {
        Self { config }
    }

    /// The bonus a job has earned by waiting, before the ceiling is applied.
    fn bonus(&self, job: &QueuedJob, now: i64) -> u32 {
        if self.config.interval_s == 0 || self.config.increment == 0 {
            return 0;
        }
        let steps = job.waited_for(now) as u64 / self.config.interval_s as u64;
        // Saturating, because a job left queued over a long weekend should not
        // wrap its way to the bottom of the queue.
        steps
            .saturating_mul(self.config.increment as u64)
            .min(u32::MAX as u64) as u32
    }

    /// What this job effectively ranks as right now.
    pub fn effective_priority(&self, job: &QueuedJob, now: i64) -> u32 {
        job.priority
            .min(MAX_PRIORITY)
            .saturating_add(self.bonus(job, now))
            .min(self.config.ceiling)
    }
}

impl QueuePolicy for Aging {
    fn name(&self) -> &'static str {
        "aging"
    }

    fn rank(&self, jobs: &[QueuedJob], ctx: &QueueContext<'_>) -> Vec<QueueRanking> {
        let scored = jobs
            .iter()
            .map(|j| {
                let base = j.priority.min(MAX_PRIORITY);
                let effective = self.effective_priority(j, ctx.now);
                (
                    j.order_seq,
                    QueueRanking {
                        job_id: j.job_id.clone(),
                        score: effective as f64,
                        components: vec![
                            ("base priority", base as f64),
                            // Reported after the ceiling, so the numbers on
                            // screen add up to the score next to them.
                            ("aging bonus", (effective - base) as f64),
                        ],
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

    /// One point per minute, up to 100.
    fn policy() -> Aging {
        Aging::new(AgingConfig {
            interval_s: 60,
            increment: 1,
            ceiling: MAX_PRIORITY,
        })
    }

    #[test]
    fn a_waiting_jobs_priority_rises() {
        let j = at("waiting", 0, 50);
        let p = policy();
        assert_eq!(p.effective_priority(&j, 0), 50);
        assert_eq!(p.effective_priority(&j, 59), 50, "not yet a full interval");
        assert_eq!(p.effective_priority(&j, 60), 51);
        assert_eq!(p.effective_priority(&j, 600), 60);
    }

    #[test]
    fn a_newer_high_priority_job_still_overtakes() {
        // Aging must not turn into "whoever waited longest always wins", or an
        // urgent job could never get through a busy queue.
        let mut old = at("old", 0, 40);
        old.submitted_unix_s = 0;
        let mut urgent = at("urgent", 1, 95);
        urgent.submitted_unix_s = 3_000;

        let ranked = rank_with(&policy(), &[old, urgent], 3_000);
        assert_eq!(ids(&ranked), ["urgent", "old"]);
        assert_eq!(ranked[1].score, 90.0, "old aged 40 -> 90 in 50 minutes");
    }

    #[test]
    fn an_old_low_priority_job_eventually_runs() {
        // The property the whole policy exists for: no permanent starvation.
        //
        // Modelled the way it actually happens -- a low-priority job waits
        // while a stream of urgent work keeps arriving. Both jobs ageing from
        // the same instant would prove nothing, since they would rise together.
        let p = policy();
        let starved = at("starved", 0, 1);

        let freshly_submitted_loud = |now: i64| {
            let mut loud = at("loud", 1, 99);
            loud.submitted_unix_s = now;
            loud
        };

        assert_eq!(
            ids(&rank_with(
                &p,
                &[starved.clone(), freshly_submitted_loud(0)],
                0
            )),
            ["loud", "starved"],
            "at first the urgent job wins, as it should"
        );

        // starved needs 99 points of bonus to pass a just-submitted 99, which
        // at one point per minute takes 99 minutes.
        let overtakes_at = 99 * 60;
        assert_eq!(
            ids(&rank_with(
                &p,
                &[starved, freshly_submitted_loud(overtakes_at)],
                overtakes_at
            )),
            ["starved", "loud"],
            "after waiting long enough it must get its turn even against new urgent work"
        );
    }

    #[test]
    fn the_ceiling_is_respected() {
        let j = at("patient", 0, 90);
        let p = Aging::new(AgingConfig {
            interval_s: 60,
            increment: 10,
            ceiling: 95,
        });
        assert_eq!(p.effective_priority(&j, 600), 95, "capped, not 190");
    }

    #[test]
    fn components_sum_to_the_score() {
        // `ferro queue` prints these under the total; they have to add up.
        let ranked = rank_with(&policy(), &[at("j", 0, 30)], 1_800);
        let sum: f64 = ranked[0].components.iter().map(|(_, v)| v).sum();
        assert_eq!(sum, ranked[0].score);
        assert_eq!(ranked[0].score, 60.0);
    }

    #[test]
    fn a_backwards_clock_does_not_grant_a_bonus() {
        let mut j = at("future", 0, 50);
        j.submitted_unix_s = 10_000;
        assert_eq!(policy().effective_priority(&j, 0), 50);
    }

    #[test]
    fn zero_increment_degenerates_to_strict_priority() {
        // Useful for the experiments: the same policy object with aging turned
        // off must behave exactly like the baseline it is compared against.
        let p = Aging::new(AgingConfig {
            interval_s: 60,
            increment: 0,
            ceiling: MAX_PRIORITY,
        });
        let jobs = vec![at("starved", 0, 1), at("loud", 1, 99)];
        assert_eq!(ids(&rank_with(&p, &jobs, 30 * 86_400)), ["loud", "starved"]);
    }

    #[test]
    fn ties_break_on_arrival_and_stay_deterministic() {
        let jobs = vec![at("b", 1, 50), at("a", 0, 50), at("c", 2, 50)];
        let first = rank_with(&policy(), &jobs, 600);
        assert_eq!(ids(&first), ["a", "b", "c"]);
        for _ in 0..8 {
            assert_eq!(rank_with(&policy(), &jobs, 600), first);
        }
    }
}
