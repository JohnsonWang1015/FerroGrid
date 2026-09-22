//! Queue ordering: *which* job runs next.
//!
//! Separate from placement on purpose. "Who goes next" is a fairness question
//! answered over the waiting list; "where do they go" is a fit question
//! answered over the hardware. Conflating them is what makes a scheduler
//! impossible to explain.

use std::collections::HashMap;

pub mod aging;
pub mod fair_share;
pub mod fifo;
pub mod priority;
pub mod sjf;

pub use aging::Aging;
pub use fair_share::jain_index;
pub use fair_share::FairShare;
pub use fifo::Fifo;
pub use priority::Priority;
pub use sjf::Sjf;

/// Priority is a 0-100 dial, and 50 is the middle of it.
pub const DEFAULT_PRIORITY: u32 = 50;
pub const MAX_PRIORITY: u32 = 100;

/// One job waiting for capacity, reduced to what ordering policies may use.
///
/// Deliberately *not* the controller's `Job`: a policy has no business seeing
/// log buffers or broadcast channels, and keeping this narrow is what lets the
/// simulator synthesise a queue without building a controller.
#[derive(Debug, Clone)]
pub struct QueuedJob {
    pub job_id: String,
    /// Monotonic submission sequence. This, not the timestamp, is what makes
    /// FIFO well defined: two jobs submitted in the same second still have an
    /// order, and the queue position each was told has to agree with it.
    pub order_seq: u64,
    pub submitted_unix_s: i64,
    pub submitted_by: String,
    /// 0-100 as submitted; the controller substitutes its default when the
    /// user said nothing.
    pub priority: u32,
    /// What the submitter expects this to run for. `None` means **unknown**,
    /// and no policy may quietly turn that into a number.
    pub estimated_duration_s: Option<u32>,
    /// GPUs this job is asking for, used to weigh what granting it would cost.
    pub gpus: u32,
}

impl QueuedJob {
    /// Seconds spent waiting. Never negative: a clock that went backwards must
    /// not hand out an aging bonus.
    pub fn waited_for(&self, now: i64) -> i64 {
        (now - self.submitted_unix_s).max(0)
    }
}

/// What one user has consumed, as of the snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UserUsage {
    /// GPU-seconds charged so far, finished jobs plus time accrued by running
    /// ones. Never negative.
    pub gpu_seconds: f64,
    pub running_jobs: u32,
    pub gpus_held: u32,
}

/// Consumption per user at a single instant.
///
/// Computed by the controller and handed in, rather than looked up by the
/// policy: the policy stays pure, and the numbers a scheduling decision used
/// are the same ones `ferro usage` will report.
#[derive(Debug, Clone, Default)]
pub struct UsageSnapshot {
    pub per_user: HashMap<String, UserUsage>,
}

impl UsageSnapshot {
    pub fn gpu_seconds(&self, user: &str) -> f64 {
        self.per_user
            .get(user)
            .map(|u| u.gpu_seconds)
            .unwrap_or(0.0)
    }

    /// The heaviest consumer's total, which is what the others are normalised
    /// against. Zero when nobody has run anything yet.
    pub fn peak_gpu_seconds(&self) -> f64 {
        self.per_user
            .values()
            .map(|u| u.gpu_seconds)
            .fold(0.0, f64::max)
    }
}

/// What an ordering decision may depend on.
///
/// Deliberately narrow: the waiting jobs, the clock, and what each user has
/// already consumed. Nothing about which GPUs happen to be free right now --
/// letting that in would make the queue position a user was quoted disagree
/// with the order they are actually served in.
///
/// Backfilling, when it arrives, is not a counter-example: it does not reorder
/// the queue, it takes the ranked order and fills idle capacity from further
/// down without displacing anyone. That is a dispatcher concern, not this one.
pub struct QueueContext<'a> {
    /// Injected, never read from the system clock, so aging replays
    /// identically in tests and in the simulator.
    pub now: i64,
    pub usage: &'a UsageSnapshot,
}

impl<'a> QueueContext<'a> {
    pub fn new(now: i64, usage: &'a UsageSnapshot) -> Self {
        Self { now, usage }
    }
}

/// One job's place in line, and the arithmetic that put it there.
///
/// The components are kept as numbers rather than a rendered sentence because
/// `ferro queue` and `ferro explain` need to show the sum, and a scheduler
/// whose reasoning cannot be audited is one nobody should trust.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueRanking {
    pub job_id: String,
    /// Higher runs sooner. Comparable only within one policy.
    pub score: f64,
    pub components: Vec<(&'static str, f64)>,
}

/// "Which job runs next?"
///
/// Returns rankings best-first. Implementations must produce a **total order**:
/// equal inputs must give an identical sequence, every time, or a queue
/// position shown to a user stops meaning anything. In practice that means
/// every comparison chain ends in `order_seq`, which is unique.
pub trait QueuePolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn rank(&self, jobs: &[QueuedJob], ctx: &QueueContext<'_>) -> Vec<QueueRanking>;
}

/// Sort descending by score, breaking ties by submission order.
///
/// Shared by every score-based policy so that "equal scores fall back to first
/// come, first served" is one decision made once, rather than four subtly
/// different ones.
pub(crate) fn by_score_then_arrival(mut scored: Vec<(u64, QueueRanking)>) -> Vec<QueueRanking> {
    scored.sort_by(|(a_seq, a), (b_seq, b)| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_seq.cmp(b_seq))
    });
    scored.into_iter().map(|(_, r)| r).collect()
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// A plain job: mid priority, no duration estimate, one GPU, submitted at
    /// t=0 so `now` doubles as "seconds waited".
    pub fn job(id: &str, seq: u64) -> QueuedJob {
        QueuedJob {
            job_id: id.into(),
            order_seq: seq,
            submitted_unix_s: 0,
            submitted_by: "tester".into(),
            priority: DEFAULT_PRIORITY,
            estimated_duration_s: None,
            gpus: 1,
        }
    }

    pub fn owned_by(id: &str, seq: u64, user: &str) -> QueuedJob {
        QueuedJob {
            submitted_by: user.into(),
            ..job(id, seq)
        }
    }

    pub fn rank_with(policy: &dyn QueuePolicy, jobs: &[QueuedJob], now: i64) -> Vec<QueueRanking> {
        let usage = UsageSnapshot::default();
        policy.rank(jobs, &QueueContext::new(now, &usage))
    }

    pub fn rank_using(
        policy: &dyn QueuePolicy,
        jobs: &[QueuedJob],
        now: i64,
        usage: &UsageSnapshot,
    ) -> Vec<QueueRanking> {
        policy.rank(jobs, &QueueContext::new(now, usage))
    }

    pub fn ids(ranked: &[QueueRanking]) -> Vec<String> {
        ranked.iter().map(|r| r.job_id.clone()).collect()
    }

    pub fn usage(entries: &[(&str, f64)]) -> UsageSnapshot {
        UsageSnapshot {
            per_user: entries
                .iter()
                .map(|(u, s)| {
                    (
                        u.to_string(),
                        UserUsage {
                            gpu_seconds: *s,
                            ..Default::default()
                        },
                    )
                })
                .collect(),
        }
    }
}
