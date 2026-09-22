//! Queue ordering: *which* job runs next.
//!
//! Separate from placement on purpose. "Who goes next" is a fairness question
//! answered over the waiting list; "where do they go" is a fit question
//! answered over the hardware. Conflating them is what makes a scheduler
//! impossible to explain.

pub mod fifo;

pub use fifo::Fifo;

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
}

/// What an ordering decision may depend on.
///
/// Deliberately narrow: the jobs themselves and the clock, nothing about the
/// hardware. "Who goes next" is a fairness question, and letting it depend on
/// what is free right now would make the queue position a user was quoted
/// disagree with the order they are actually served in.
///
/// Backfilling, when it arrives, is not a counter-example: it does not reorder
/// the queue, it takes the ranked order and fills idle capacity from further
/// down without displacing anyone. That is a dispatcher concern, not this one.
pub struct QueueContext {
    /// Injected, never read from the system clock, so aging replays
    /// identically in tests and in the simulator.
    pub now: i64,
}

/// "Which job runs next?"
///
/// Returns job ids best-first. Implementations must produce a **total order**:
/// equal inputs must give an identical sequence, every time, or a queue
/// position shown to a user stops meaning anything.
pub trait QueuePolicy: Send + Sync {
    fn name(&self) -> &'static str;

    fn rank(&self, jobs: &[QueuedJob], ctx: &QueueContext) -> Vec<String>;
}
