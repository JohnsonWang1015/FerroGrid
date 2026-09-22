//! Dispatch: *which* of the jobs the queue has already ranked may attempt to
//! run this tick.
//!
//! Ordering is the queue policy's question and fit is the placement policy's.
//! This is the third one, and the one FerroGrid currently answers by accident:
//! `run_queue` walks the entire ranked waiting list each tick and starts
//! anything that fits, not just the head. That is *opportunistic* dispatch --
//! backfilling with no reservation -- and it is not what "FIFO queue" is
//! usually taken to mean.
//!
//! ## What it is worth, measured
//!
//! On workload D (large distributed jobs against a stream of small ones),
//! opportunistic dispatch beat strict head-of-line FIFO by **4.9x on mean
//! waiting time** -- 1 529 s against 7 471 s -- and starved eleven times fewer
//! jobs. The cost is the tail: with nothing reserved, a four-GPU job can be
//! walked past indefinitely by a stream of one-GPU jobs, and D's p95 wait of
//! 10 801 s against that 1 529 s mean *is* the large jobs. Reservation is what
//! bounds that tail; see `docs/os_term_project/experiments.md` §4.
//!
//! ## Why reservation is not the default
//!
//! EASY backfilling has to know when the *running* jobs will end, and
//! FerroGrid has exactly two sources for that, both optional: the submitter's
//! `--estimated-duration`, and `--timeout`. A running job with neither is
//! **unknowable**, and this module refuses to pretend otherwise -- if the
//! reservation holder's earliest start cannot be computed, nothing may be
//! backfilled past it and the tick degrades to strict FIFO. That is the safe
//! failure, and it is also, on the evidence above, a large regression.
//!
//! So on a cluster where nobody declares anything, switching the default to
//! [`Dispatch::Reserved`] would turn a 1 529 s mean wait into a 7 471 s one in
//! exchange for a tail nobody measured. Most clusters are that cluster.
//! Reservation is therefore opt-in, and the default stays what FerroGrid
//! already does.
//!
//! ## What this module deliberately does not decide
//!
//! Whether a job *actually fits*: that needs the cluster snapshot and is the
//! placement policy's answer. [`admissible`] works from a single count of free
//! GPUs, so it can say "there is not enough room" but never "there is room, of
//! the wrong shape". A job it admits may still fail to place, in which case the
//! GPUs it notionally consumed were not consumed after all and the rest of the
//! pass is more pessimistic than it needed to be. The next tick starts again
//! from the real cluster, so the error does not accumulate.

use crate::queue::QueuedJob;
use ferro_proto::NodeState;

/// How the dispatcher treats a ranked job that does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dispatch {
    /// Keep walking the ranked list and start anything that fits. What the
    /// controller does today, and the default for the reasons in the module
    /// documentation.
    #[default]
    Opportunistic,
    /// Stop at the first job that does not fit, so nothing overtakes it.
    /// Classic FIFO, with the head-of-line blocking that comes with it. A
    /// comparison point, not a recommendation.
    Strict,
    /// Backfill, but only past a reservation the backfilled job can prove it
    /// will not delay. Bounds the large-job tail that [`Self::Opportunistic`]
    /// leaves unbounded, at the price of needing declared durations.
    Reserved,
}

impl Dispatch {
    pub fn label(self) -> &'static str {
        match self {
            Dispatch::Opportunistic => "opportunistic",
            Dispatch::Strict => "strict",
            Dispatch::Reserved => "reserved",
        }
    }
}

/// A job that currently holds GPUs, reduced to what reservation needs of it.
#[derive(Debug, Clone)]
pub struct RunningJob {
    pub job_id: String,
    pub gpus: u32,
    /// When this is expected to free its GPUs. `None` means nobody knows --
    /// no estimate, no timeout -- and the algorithm must treat that as
    /// unknowable rather than guessing.
    pub expected_end_s: Option<i64>,
}

/// Whether one job may attempt to place this tick, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    pub job_id: String,
    pub allowed: bool,
    /// Why, for `ferro queue` and `ferro explain`. A queued job that was never
    /// offered to the placement policy has no placement verdict to show, and
    /// showing the previous tick's would be a lie about what just happened.
    pub reason: &'static str,
}

// These end up in `ferro queue`'s WHY WAITING column next to the scheduler's
// own messages, so they are kept to about the same length: a reason that wraps
// the table is a reason nobody reads.
const OPPORTUNISTIC: &str = "opportunistic dispatch: anything that fits may start";
const AT_THE_HEAD: &str = "at the head of the queue";
const NOBODY_BLOCKED_AHEAD: &str = "fits, and so did everything ahead of it";
const BLOCKED_AHEAD: &str = "strict dispatch: a job ahead of it does not fit";
const FITS_NOW: &str = "fits in the GPUs free right now";
const HOLDS_RESERVATION: &str = "does not fit; holds the reservation others backfill around";
const NO_ROOM_LEFT: &str = "does not fit in the GPUs left free";
const NO_DECLARED_DURATION: &str = "no declared duration, so it cannot prove it would finish \
                                    in time";
const WOULD_OVERRUN: &str = "would still be running when the reserved job is due to start";
const RESERVATION_UNKNOWN: &str = "the reservation's earliest start is unknown: a running job \
                                   declared nothing";

/// Free, placeable GPUs across the healthy nodes.
///
/// One definition shared by the controller and the simulator, because "free"
/// has to mean the same thing here as it does everywhere else: unallocated
/// *and* with enough VRAM headroom left by whatever else is on the card. A
/// dispatcher counting cards the placement policy would refuse promises
/// backfills that then fail.
pub fn free_gpus(nodes: &[NodeState], min_free_b: u64) -> u32 {
    nodes
        .iter()
        .filter(|n| n.healthy)
        .map(|n| crate::placement::engine::placeable(n, min_free_b).len() as u32)
        .sum()
}

/// Who, out of `ranked`, may attempt to place this tick.
///
/// `ranked` is best-first, as the queue policy left it; `free_gpus` is the
/// whole cluster's placeable count; `running` is every job currently holding
/// GPUs. The result is one [`Admission`] per input job, in the same order.
///
/// Pure: the same inputs give the same answer, every time. `now` is passed in
/// rather than read, like everywhere else in this crate.
pub fn admissible(
    ranked: &[QueuedJob],
    running: &[RunningJob],
    free_gpus: u32,
    now: i64,
    mode: Dispatch,
) -> Vec<Admission> {
    match mode {
        Dispatch::Opportunistic => ranked
            .iter()
            .map(|j| verdict(j, true, OPPORTUNISTIC))
            .collect(),
        Dispatch::Strict => strict(ranked, free_gpus),
        Dispatch::Reserved => reserved(ranked, running, free_gpus, now),
    }
}

fn verdict(job: &QueuedJob, allowed: bool, reason: &'static str) -> Admission {
    Admission {
        job_id: job.job_id.clone(),
        allowed,
        reason,
    }
}

/// Head-of-line FIFO: the queue is served strictly in order, and stops dead at
/// the first job that does not fit.
///
/// The head is always offered to the placement policy, even when the free-GPU
/// count says it cannot fit. Two reasons: the count is cluster-wide and cannot
/// see shape, and the attempt is what produces the per-node verdicts that tell
/// the user *why* they are waiting. Everything after the first job that does
/// not fit is refused, which is the head-of-line blocking this mode exists to
/// demonstrate.
///
/// Note what this is not: "only the head may ever start". A run of jobs that
/// all fit overtakes nobody -- each one was served in its turn -- so refusing
/// them would not be stricter, only slower, at one start per tick. That would
/// make the strict baseline a claim about tick rates rather than about
/// head-of-line blocking.
fn strict(ranked: &[QueuedJob], free_gpus: u32) -> Vec<Admission> {
    let mut out = Vec::with_capacity(ranked.len());
    let mut free = free_gpus;
    let mut blocked = false;

    for (i, job) in ranked.iter().enumerate() {
        if blocked {
            out.push(verdict(job, false, BLOCKED_AHEAD));
            continue;
        }
        let fits = job.gpus <= free;
        if fits {
            free -= job.gpus;
        } else {
            // Whether or not this one is offered to the placer, it is the job
            // the queue now waits behind.
            blocked = true;
        }
        match (i, fits) {
            (0, _) => out.push(verdict(job, true, AT_THE_HEAD)),
            (_, true) => out.push(verdict(job, true, NOBODY_BLOCKED_AHEAD)),
            (_, false) => out.push(verdict(job, false, BLOCKED_AHEAD)),
        }
    }
    out
}

/// EASY backfilling: one reservation, and only provably harmless overtaking.
fn reserved(
    ranked: &[QueuedJob],
    running: &[RunningJob],
    free_gpus: u32,
    now: i64,
) -> Vec<Admission> {
    let mut out = Vec::with_capacity(ranked.len());
    let mut free = free_gpus;
    // `None` until somebody fails to fit; then `Some(earliest)`, where the
    // inner `None` is an earliest start nobody can compute.
    let mut reservation: Option<Option<i64>> = None;

    for job in ranked {
        let Some(earliest) = reservation else {
            // Still ahead of any reservation: fitting is the only question.
            if job.gpus <= free {
                free -= job.gpus;
                out.push(verdict(job, true, FITS_NOW));
            } else {
                reservation = Some(earliest_start(running, free, job.gpus));
                out.push(verdict(job, false, HOLDS_RESERVATION));
            }
            continue;
        };

        // Past the reservation. Overtaking is now something a job has to earn.
        let Some(earliest) = earliest else {
            out.push(verdict(job, false, RESERVATION_UNKNOWN));
            continue;
        };
        if job.gpus > free {
            out.push(verdict(job, false, NO_ROOM_LEFT));
            continue;
        }
        let Some(duration) = declared_duration_s(job) else {
            out.push(verdict(job, false, NO_DECLARED_DURATION));
            continue;
        };
        if now.saturating_add(duration as i64) <= earliest {
            free -= job.gpus;
            out.push(verdict(job, true, FITS_NOW));
        } else {
            out.push(verdict(job, false, WOULD_OVERRUN));
        }
    }
    out
}

/// The soonest `needed` GPUs could be free, given `free` now and what the
/// running jobs say they will release.
///
/// `None` means unknowable, which is the answer whenever the running jobs run
/// out before the count is reached -- including the case where they run out
/// because one of them declared nothing. An unknown end sorts last and is
/// never passed: a job that will free its GPUs at a time nobody knows cannot
/// be counted on to free them at all.
fn earliest_start(running: &[RunningJob], free: u32, needed: u32) -> Option<i64> {
    let mut order: Vec<&RunningJob> = running.iter().collect();
    // `Option`'s own ordering puts `None` first, which is exactly backwards
    // here. Ties break on the job id so that two runs over the same cluster
    // agree.
    order.sort_by(|a, b| {
        a.expected_end_s
            .is_none()
            .cmp(&b.expected_end_s.is_none())
            .then_with(|| a.expected_end_s.cmp(&b.expected_end_s))
            .then_with(|| a.job_id.cmp(&b.job_id))
    });

    let mut available = free as u64;
    for r in order {
        let end = r.expected_end_s?;
        available += r.gpus as u64;
        if available >= needed as u64 {
            return Some(end);
        }
    }
    None
}

/// How long a waiting job says it will run for, or `None` if it never said.
///
/// The smaller of the two declarations when both exist. They mean different
/// things -- the estimate is a claim, the timeout is a ceiling the controller
/// actually enforces by killing the job -- and the smaller of them is the
/// earliest point past which the job either has finished or has been stopped.
/// Taking the smaller is also the conservative direction here: it can only
/// ever make a backfill candidate look *better* than the enforced ceiling
/// would, but the same rule applied to running jobs makes the reservation's
/// earliest start earlier, and so admits less.
///
/// What this does not survive is a wrong estimate. FerroGrid does not kill a
/// job at its estimate, only at its timeout, so a job backfilled on the
/// strength of an estimate it then overruns *will* delay the reservation.
/// Phase 4 measured estimates as frequently absent and frequently wrong; that
/// is a real limit on what this mode can promise, not a detail.
fn declared_duration_s(job: &QueuedJob) -> Option<u32> {
    match (job.estimated_duration_s, job.timeout_s) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::tests_support::job;

    /// `gpus` wide, and saying nothing about how long it will run.
    fn wanting(id: &str, seq: u64, gpus: u32) -> QueuedJob {
        QueuedJob {
            gpus,
            ..job(id, seq)
        }
    }

    /// The same, but declaring a duration.
    fn lasting(id: &str, seq: u64, gpus: u32, seconds: u32) -> QueuedJob {
        QueuedJob {
            estimated_duration_s: Some(seconds),
            ..wanting(id, seq, gpus)
        }
    }

    fn running(id: &str, gpus: u32, end: Option<i64>) -> RunningJob {
        RunningJob {
            job_id: id.into(),
            gpus,
            expected_end_s: end,
        }
    }

    fn allowed(out: &[Admission]) -> Vec<&str> {
        out.iter()
            .filter(|a| a.allowed)
            .map(|a| a.job_id.as_str())
            .collect()
    }

    fn reason_for<'a>(out: &'a [Admission], id: &str) -> &'a str {
        out.iter()
            .find(|a| a.job_id == id)
            .expect("job was not judged at all")
            .reason
    }

    #[test]
    fn opportunistic_admits_everything() {
        // Including jobs that plainly do not fit: whether they fit is the
        // placement policy's answer, and this mode declines to pre-empt it.
        let queue = vec![wanting("big", 0, 8), wanting("small", 1, 1)];
        let out = admissible(&queue, &[], 2, 1_000, Dispatch::Opportunistic);
        assert_eq!(allowed(&out), ["big", "small"]);
    }

    #[test]
    fn strict_admits_only_the_head() {
        // The head does not fit, so it is the whole of the queue that may run:
        // this is head-of-line blocking, asserted rather than assumed.
        let queue = vec![wanting("big", 0, 8), wanting("small", 1, 1)];
        let out = admissible(&queue, &[], 2, 1_000, Dispatch::Strict);
        assert_eq!(allowed(&out), ["big"]);
        assert_eq!(reason_for(&out, "small"), BLOCKED_AHEAD);
    }

    #[test]
    fn strict_still_serves_a_queue_nobody_is_blocking() {
        // Three jobs, all of which fit, served in order. None of them overtook
        // anything, so refusing them would not be stricter -- only one start
        // per tick, which would make the strict baseline a claim about the
        // tick rate instead of about head-of-line blocking.
        let queue = vec![wanting("a", 0, 1), wanting("b", 1, 1), wanting("c", 2, 1)];
        let out = admissible(&queue, &[], 4, 1_000, Dispatch::Strict);
        assert_eq!(allowed(&out), ["a", "b", "c"]);
    }

    #[test]
    fn strict_stops_at_the_first_job_that_does_not_fit() {
        // `c` fits in what is left, and is refused anyway: that is the whole
        // difference between strict and opportunistic dispatch.
        let queue = vec![wanting("a", 0, 1), wanting("b", 1, 8), wanting("c", 2, 1)];
        let out = admissible(&queue, &[], 4, 1_000, Dispatch::Strict);
        assert_eq!(allowed(&out), ["a"]);
        assert_eq!(reason_for(&out, "c"), BLOCKED_AHEAD);
        assert_eq!(
            allowed(&admissible(&queue, &[], 4, 1_000, Dispatch::Opportunistic)),
            ["a", "b", "c"],
            "the control: opportunistic dispatch lets `c` past"
        );
    }

    #[test]
    fn reserved_backfills_a_job_that_finishes_before_the_reservation() {
        // Four GPUs, two free, and a running job that says it will release its
        // other two at t=2000. `big` reserves them; `small` declares 500s,
        // finishes at 1500, and so cannot delay anybody.
        let queue = vec![wanting("big", 0, 4), lasting("small", 1, 1, 500)];
        let out = admissible(
            &queue,
            &[running("incumbent", 2, Some(2_000))],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert_eq!(allowed(&out), ["small"]);
        assert_eq!(reason_for(&out, "big"), HOLDS_RESERVATION);
    }

    #[test]
    fn reserved_refuses_a_job_that_declared_no_duration() {
        // The same queue, with the one difference that matters: `small` says
        // nothing, so it can prove nothing.
        let queue = vec![wanting("big", 0, 4), wanting("small", 1, 1)];
        let out = admissible(
            &queue,
            &[running("incumbent", 2, Some(2_000))],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert!(allowed(&out).is_empty());
        assert_eq!(reason_for(&out, "small"), NO_DECLARED_DURATION);
    }

    #[test]
    fn reserved_refuses_a_job_that_would_still_be_running() {
        // It fits, and it declared -- but 1000 + 1500 is past 2000, so
        // starting it is exactly the decision that produced the tail.
        let queue = vec![wanting("big", 0, 4), lasting("small", 1, 1, 1_500)];
        let out = admissible(
            &queue,
            &[running("incumbent", 2, Some(2_000))],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert!(allowed(&out).is_empty());
        assert_eq!(reason_for(&out, "small"), WOULD_OVERRUN);
    }

    #[test]
    fn reserved_refuses_everything_past_an_unknowable_reservation() {
        // Nobody declared anything, so there is no answer to "when do these
        // GPUs come back". Guessing here is what would make the mode a lie.
        let queue = vec![wanting("big", 0, 4), lasting("small", 1, 1, 1)];
        let out = admissible(
            &queue,
            &[running("incumbent", 2, None)],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert!(allowed(&out).is_empty());
        assert_eq!(reason_for(&out, "small"), RESERVATION_UNKNOWN);
    }

    #[test]
    fn a_known_end_behind_an_unknown_one_is_still_not_reached() {
        // Two GPUs free, `big` wants six, and only one running job declares.
        // Four is as far as the declarations get, so the earliest start is
        // unknown however tidy the one number looks.
        let queue = vec![wanting("big", 0, 6), lasting("small", 1, 1, 1)];
        let out = admissible(
            &queue,
            &[
                running("declares", 2, Some(2_000)),
                running("silent", 4, None),
            ],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert!(allowed(&out).is_empty());
        assert_eq!(reason_for(&out, "small"), RESERVATION_UNKNOWN);
    }

    #[test]
    fn reserved_on_an_empty_queue_decides_nothing() {
        let out = admissible(
            &[],
            &[running("a", 4, Some(2_000))],
            0,
            1_000,
            Dispatch::Reserved,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn reserved_admits_everybody_when_everybody_fits() {
        // No reservation is created at all, so reservation costs nothing on an
        // uncontended cluster -- not even from jobs that declared nothing.
        let queue = vec![wanting("a", 0, 2), wanting("b", 1, 2), wanting("c", 2, 4)];
        let out = admissible(&queue, &[], 8, 1_000, Dispatch::Reserved);
        assert_eq!(allowed(&out), ["a", "b", "c"]);
        assert!(out.iter().all(|a| a.reason == FITS_NOW));
    }

    #[test]
    fn a_backfilled_job_consumes_the_room_it_took() {
        // Two free. The first short job takes one; the second is identical and
        // may have the other; the third must not be handed a card twice.
        let queue = vec![
            wanting("big", 0, 4),
            lasting("s1", 1, 1, 100),
            lasting("s2", 2, 1, 100),
            lasting("s3", 3, 1, 100),
        ];
        let out = admissible(
            &queue,
            &[running("incumbent", 4, Some(9_000))],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert_eq!(allowed(&out), ["s1", "s2"]);
        assert_eq!(reason_for(&out, "s3"), NO_ROOM_LEFT);
    }

    #[test]
    fn the_reservation_is_computed_from_what_is_left_not_what_there_was() {
        // `first` takes the one free GPU before `big` is reached, so `big`
        // needs three more from the running jobs, not two. That pushes its
        // earliest start out to 3000 and lets `tail` -- which would have been
        // refused against 2000 -- in.
        let queue = vec![
            wanting("first", 0, 1),
            wanting("big", 1, 4),
            lasting("tail", 2, 1, 1_800),
        ];
        let out = admissible(
            &queue,
            &[
                running("early", 2, Some(2_000)),
                running("late", 2, Some(3_000)),
            ],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert_eq!(allowed(&out), ["first", "tail"]);
    }

    #[test]
    fn a_timeout_is_a_declaration_too() {
        // No estimate, but a wall-clock limit the controller enforces: the job
        // is provably gone by then whatever it thinks it is doing.
        let queue = vec![
            wanting("big", 0, 4),
            QueuedJob {
                timeout_s: Some(400),
                ..wanting("small", 1, 1)
            },
        ];
        let out = admissible(
            &queue,
            &[running("incumbent", 2, Some(2_000))],
            2,
            1_000,
            Dispatch::Reserved,
        );
        assert_eq!(allowed(&out), ["small"]);
    }

    #[test]
    fn identical_input_gives_identical_output() {
        // Including the sort inside `earliest_start`, which is why two running
        // jobs share an end time here.
        let queue = vec![
            wanting("big", 0, 6),
            lasting("a", 1, 1, 100),
            wanting("b", 2, 1),
            lasting("c", 3, 2, 9_000),
        ];
        let running = vec![
            running("r1", 2, Some(2_000)),
            running("r2", 2, Some(2_000)),
            running("r3", 1, None),
        ];
        let first = admissible(&queue, &running, 2, 1_000, Dispatch::Reserved);
        for _ in 0..8 {
            assert_eq!(
                admissible(&queue, &running, 2, 1_000, Dispatch::Reserved),
                first,
                "identical input must give an identical answer"
            );
        }
    }

    #[test]
    fn every_mode_judges_every_job_exactly_once_and_in_order() {
        // The caller pairs these up with the ranked list positionally.
        let queue = vec![
            wanting("a", 0, 1),
            wanting("b", 1, 9),
            lasting("c", 2, 1, 10),
        ];
        for mode in [
            Dispatch::Opportunistic,
            Dispatch::Strict,
            Dispatch::Reserved,
        ] {
            let out = admissible(&queue, &[], 2, 0, mode);
            let ids: Vec<&str> = out.iter().map(|a| a.job_id.as_str()).collect();
            assert_eq!(ids, ["a", "b", "c"], "{} reordered the queue", mode.label());
        }
    }
}
