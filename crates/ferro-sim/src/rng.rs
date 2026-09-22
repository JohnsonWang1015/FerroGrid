//! A small, fixed, seeded generator.
//!
//! Written out rather than pulled in, for one reason: §56 requires every
//! experiment to be reproducible from its seed, and "reproducible" has to mean
//! *by someone else, later*. A dependency whose algorithm is an implementation
//! detail can change it in a patch release and silently invalidate every number
//! already published. Forty lines of PCG here is cheaper than that risk.
//!
//! This is PCG-XSH-RR 64/32, which is small, well studied, and more than good
//! enough for arrival times and job sizes. It is not cryptographic and nothing
//! here should ever be used as though it were.

/// Deterministic for a given seed, on every platform, for as long as this file
/// does not change.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    inc: u64,
}

const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        // The stream constant is fixed so that a seed alone identifies the
        // sequence; two runs quoting the same seed must mean the same thing.
        let mut rng = Self {
            state: 0,
            inc: 0xda3e_39cb_94b9_5bdb,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.inc | 1);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        // 24 bits is ample and keeps the value exactly representable.
        (self.next_u32() >> 8) as f64 / (1u32 << 24) as f64
    }

    /// Uniform integer in `[low, high]`, inclusive.
    pub fn range(&mut self, low: u64, high: u64) -> u64 {
        if high <= low {
            return low;
        }
        low + (self.unit() * ((high - low + 1) as f64)) as u64 % (high - low + 1)
    }

    /// An exponential interval with the given mean, which is what makes
    /// arrivals Poisson.
    pub fn exponential(&mut self, mean: f64) -> f64 {
        if mean <= 0.0 {
            return 0.0;
        }
        // `unit()` can return 0, and ln(0) is -inf; nudge it off the boundary.
        let u = 1.0 - self.unit().min(1.0 - f64::EPSILON);
        -mean * u.ln()
    }

    /// Pick an index from a weighted list. Weights need not sum to anything in
    /// particular; non-positive weights are simply never chosen.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().filter(|w| **w > 0.0).sum();
        if total <= 0.0 {
            return 0;
        }
        let mut point = self.unit() * total;
        for (i, w) in weights.iter().enumerate() {
            if *w <= 0.0 {
                continue;
            }
            point -= w;
            if point <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        // The property every published result rests on.
        let a: Vec<u32> = (0..64).map(|_| Rng::seeded(42).next_u32()).collect();
        let b: Vec<u32> = (0..64).map(|_| Rng::seeded(42).next_u32()).collect();
        assert_eq!(a, b);

        let mut one = Rng::seeded(7);
        let mut two = Rng::seeded(7);
        for _ in 0..1_000 {
            assert_eq!(one.next_u32(), two.next_u32());
        }
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        let mut a = Rng::seeded(1);
        let mut b = Rng::seeded(2);
        let differ = (0..32).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(differ > 28, "only {differ}/32 draws differed");
    }

    #[test]
    fn unit_stays_in_range_and_spreads_out() {
        let mut rng = Rng::seeded(99);
        let mut buckets = [0usize; 10];
        for _ in 0..10_000 {
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u), "{u} out of range");
            buckets[(u * 10.0) as usize] += 1;
        }
        // Not a statistical test, just a check that it is not stuck.
        for (i, n) in buckets.iter().enumerate() {
            assert!(*n > 700, "bucket {i} had only {n} of 10000");
        }
    }

    #[test]
    fn range_is_inclusive_at_both_ends() {
        let mut rng = Rng::seeded(3);
        let mut seen_low = false;
        let mut seen_high = false;
        for _ in 0..1_000 {
            let v = rng.range(5, 8);
            assert!((5..=8).contains(&v), "{v} out of range");
            seen_low |= v == 5;
            seen_high |= v == 8;
        }
        assert!(seen_low && seen_high);
    }

    #[test]
    fn a_degenerate_range_is_its_own_answer() {
        let mut rng = Rng::seeded(3);
        assert_eq!(rng.range(7, 7), 7);
        assert_eq!(rng.range(9, 2), 9);
    }

    #[test]
    fn exponential_has_roughly_the_mean_it_was_asked_for() {
        let mut rng = Rng::seeded(11);
        let n = 50_000;
        let mean: f64 = (0..n).map(|_| rng.exponential(30.0)).sum::<f64>() / n as f64;
        assert!((mean - 30.0).abs() < 1.0, "mean was {mean}");
    }

    #[test]
    fn weighted_respects_its_weights() {
        let mut rng = Rng::seeded(5);
        let mut counts = [0usize; 3];
        for _ in 0..10_000 {
            counts[rng.weighted(&[1.0, 3.0, 0.0])] += 1;
        }
        assert_eq!(counts[2], 0, "a zero weight must never be chosen");
        let ratio = counts[1] as f64 / counts[0] as f64;
        assert!((ratio - 3.0).abs() < 0.3, "ratio was {ratio}");
    }
}
