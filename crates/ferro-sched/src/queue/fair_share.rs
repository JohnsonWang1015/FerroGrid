//! Fair share: what you have already used counts against you.
//!
//! ```text
//! score = a * priority_norm + b * wait_norm - c * usage_norm
//! ```
//!
//! Each term is normalised to 0..1 **against the queue being ranked**, not
//! against an absolute scale. That is what keeps the weights meaningful: `a`,
//! `b` and `c` trade off three quantities in the same units, so setting them
//! to 1/1/1 really does mean "these matter equally". Normalising against fixed
//! constants instead would make the weights depend on how busy the cluster
//! happens to be that week.
//!
//! Usage is per *user*, not per job, because that is the unit fairness is owed
//! to. A user who submits fifty small jobs should not thereby out-compete one
//! who submitted a single large one.

use super::{
    by_score_then_arrival, QueueContext, QueuePolicy, QueueRanking, QueuedJob, MAX_PRIORITY,
};

/// The three weights, named rather than positional so a config file cannot
/// silently transpose them.
#[derive(Debug, Clone, Copy)]
pub struct FairShareConfig {
    pub priority_weight: f64,
    pub wait_weight: f64,
    pub usage_weight: f64,
}

impl Default for FairShareConfig {
    fn default() -> Self {
        // Equal thirds: a starting point to measure from, not a claim that
        // these are the right numbers for any particular cluster.
        Self {
            priority_weight: 1.0,
            wait_weight: 1.0,
            usage_weight: 1.0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FairShare {
    pub config: FairShareConfig,
}

impl FairShare {
    pub fn new(config: FairShareConfig) -> Self {
        Self { config }
    }
}

/// Scale into 0..1 against a maximum, treating "everything is zero" as "this
/// term tells us nothing" rather than dividing by zero.
fn normalise(value: f64, max: f64) -> f64 {
    if max <= 0.0 {
        0.0
    } else {
        (value / max).clamp(0.0, 1.0)
    }
}

impl QueuePolicy for FairShare {
    fn name(&self) -> &'static str {
        "fair-share"
    }

    fn rank(&self, jobs: &[QueuedJob], ctx: &QueueContext<'_>) -> Vec<QueueRanking> {
        let longest_wait = jobs
            .iter()
            .map(|j| j.waited_for(ctx.now))
            .max()
            .unwrap_or(0) as f64;
        // Normalise against the heaviest user in the *cluster*, not merely the
        // heaviest currently queued: someone who used a fortnight of GPU time
        // and then stopped submitting should still weigh on their next job.
        let heaviest_usage = ctx.usage.peak_gpu_seconds();

        let scored = jobs
            .iter()
            .map(|j| {
                let priority = normalise(j.priority.min(MAX_PRIORITY) as f64, MAX_PRIORITY as f64);
                let wait = normalise(j.waited_for(ctx.now) as f64, longest_wait);
                let used = normalise(ctx.usage.gpu_seconds(&j.submitted_by), heaviest_usage);

                let p = self.config.priority_weight * priority;
                let w = self.config.wait_weight * wait;
                let u = -self.config.usage_weight * used;

                (
                    j.order_seq,
                    QueueRanking {
                        job_id: j.job_id.clone(),
                        score: p + w + u,
                        components: vec![("priority", p), ("waiting", w), ("usage", u)],
                    },
                )
            })
            .collect();
        by_score_then_arrival(scored)
    }
}

/// Jain's fairness index over any non-negative allocation.
///
/// ```text
/// J(x) = (sum x)^2 / (n * sum x^2)
/// ```
///
/// 1.0 means everyone received the same; 1/n means one participant took
/// everything. Returns `None` for an empty input, and for a negative one --
/// the index is only defined over non-negative shares, and silently accepting
/// a negative GPU-second total would report a fairness number for data that is
/// already wrong.
///
/// Nobody having used anything is reported as perfectly fair, which is both
/// the limit of the formula and the honest answer.
pub fn jain_index(shares: &[f64]) -> Option<f64> {
    if shares.is_empty() || shares.iter().any(|x| *x < 0.0 || x.is_nan()) {
        return None;
    }
    let sum: f64 = shares.iter().sum();
    let sum_sq: f64 = shares.iter().map(|x| x * x).sum();
    if sum_sq == 0.0 {
        return Some(1.0);
    }
    Some(sum * sum / (shares.len() as f64 * sum_sq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::tests_support::{ids, job, owned_by, rank_using, usage};
    use crate::queue::UsageSnapshot;

    fn equal_weights() -> FairShare {
        FairShare::new(FairShareConfig::default())
    }

    #[test]
    fn a_heavy_user_yields_to_a_light_one() {
        let jobs = vec![owned_by("hog", 0, "alice"), owned_by("newcomer", 1, "bob")];
        let used = usage(&[("alice", 10_000.0), ("bob", 0.0)]);
        assert_eq!(
            ids(&rank_using(&equal_weights(), &jobs, 100, &used)),
            ["newcomer", "hog"],
            "alice has had her turn"
        );
    }

    #[test]
    fn with_equal_usage_it_falls_back_to_arrival() {
        let jobs = vec![owned_by("second", 1, "bob"), owned_by("first", 0, "alice")];
        let used = usage(&[("alice", 500.0), ("bob", 500.0)]);
        assert_eq!(
            ids(&rank_using(&equal_weights(), &jobs, 100, &used)),
            ["first", "second"]
        );
    }

    #[test]
    fn an_unknown_user_is_treated_as_having_used_nothing() {
        // Charging a first-time submitter for usage nobody recorded would be a
        // fabricated penalty.
        let jobs = vec![
            owned_by("veteran", 0, "alice"),
            owned_by("stranger", 1, "nobody-has-heard-of-them"),
        ];
        let used = usage(&[("alice", 9_000.0)]);
        assert_eq!(
            ids(&rank_using(&equal_weights(), &jobs, 100, &used)),
            ["stranger", "veteran"]
        );
    }

    #[test]
    fn the_weights_decide_between_a_long_wait_and_a_heavy_history() {
        // alice has used the cluster hard and has been waiting longest; bob has
        // used nothing and just arrived. Each term is normalised against the
        // queue, so alice takes the full wait credit and the full usage
        // penalty: under equal weights they cancel exactly, and arrival order
        // decides. Which of them *should* win is precisely what the weights are
        // for, so assert that they control it rather than picking a favourite.
        let mut old = owned_by("alice-old", 0, "alice");
        old.submitted_unix_s = 0;
        let mut fresh = owned_by("bob-fresh", 1, "bob");
        fresh.submitted_unix_s = 10_000;
        let used = usage(&[("alice", 1_000.0), ("bob", 0.0)]);
        let queue = [old, fresh];

        let usage_matters_more = FairShare::new(FairShareConfig {
            priority_weight: 1.0,
            wait_weight: 1.0,
            usage_weight: 2.0,
        });
        assert_eq!(
            ids(&rank_using(&usage_matters_more, &queue, 10_000, &used))[0],
            "bob-fresh",
            "weighted towards fairness, the clean record goes first"
        );

        let waiting_matters_more = FairShare::new(FairShareConfig {
            priority_weight: 1.0,
            wait_weight: 2.0,
            usage_weight: 1.0,
        });
        assert_eq!(
            ids(&rank_using(&waiting_matters_more, &queue, 10_000, &used))[0],
            "alice-old",
            "weighted towards waiting, the heavy user still gets her turn back"
        );
    }

    #[test]
    fn weights_actually_weigh() {
        let jobs = vec![owned_by("hog", 0, "alice"), owned_by("newcomer", 1, "bob")];
        let used = usage(&[("alice", 10_000.0), ("bob", 0.0)]);
        // Turn the usage term off entirely and the ordering reverts to arrival.
        let ignore_usage = FairShare::new(FairShareConfig {
            priority_weight: 1.0,
            wait_weight: 1.0,
            usage_weight: 0.0,
        });
        assert_eq!(
            ids(&rank_using(&ignore_usage, &jobs, 100, &used)),
            ["hog", "newcomer"]
        );
    }

    #[test]
    fn components_sum_to_the_score() {
        let jobs = vec![owned_by("j", 0, "alice")];
        let used = usage(&[("alice", 100.0)]);
        let ranked = rank_using(&equal_weights(), &jobs, 100, &used);
        let sum: f64 = ranked[0].components.iter().map(|(_, v)| v).sum();
        assert!((sum - ranked[0].score).abs() < 1e-12);
    }

    #[test]
    fn an_empty_cluster_history_does_not_divide_by_zero() {
        let jobs = vec![job("a", 0), job("b", 1)];
        let ranked = rank_using(&equal_weights(), &jobs, 0, &UsageSnapshot::default());
        assert_eq!(ids(&ranked), ["a", "b"]);
        assert!(ranked.iter().all(|r| r.score.is_finite()));
    }

    #[test]
    fn jain_is_one_when_everyone_gets_the_same() {
        assert_eq!(jain_index(&[5.0, 5.0, 5.0]), Some(1.0));
        assert_eq!(jain_index(&[42.0]), Some(1.0));
    }

    #[test]
    fn jain_falls_to_one_over_n_when_one_user_takes_everything() {
        let j = jain_index(&[10.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((j - 0.25).abs() < 1e-12, "got {j}");
    }

    #[test]
    fn jain_matches_a_worked_example() {
        // (4.2 + 5.1 + 6.2)^2 / (3 * (4.2^2 + 5.1^2 + 6.2^2))
        let j = jain_index(&[4.2, 5.1, 6.2]).unwrap();
        assert!((j - 0.975_555).abs() < 1e-6, "got {j}");
    }

    #[test]
    fn jain_rejects_input_it_cannot_describe() {
        assert_eq!(jain_index(&[]), None, "no users, no fairness to report");
        assert_eq!(
            jain_index(&[1.0, -1.0]),
            None,
            "usage cannot be negative; report nothing rather than a number"
        );
        assert_eq!(jain_index(&[f64::NAN]), None);
    }

    #[test]
    fn jain_treats_an_idle_cluster_as_fair() {
        assert_eq!(jain_index(&[0.0, 0.0, 0.0]), Some(1.0));
    }
}
