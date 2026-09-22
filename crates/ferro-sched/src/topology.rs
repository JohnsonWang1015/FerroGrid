//! What the wires between nodes actually do.
//!
//! Two different claims live here and they must not be confused:
//!
//! * **Negotiated link speed** (`NodeInfo.link_mbps`) covers the node-to-switch
//!   hop only. A 1000 Mb/s NIC reporting zero errors can still sit behind a
//!   100 Mb/s path.
//! * **Measured throughput** (`ferro net`) is what a cross-node job runs at.
//!
//! Placement should prefer the measured number where one exists and fall back
//! to the negotiated one where it does not, rather than pretending the two are
//! the same quantity.

use std::collections::HashMap;

/// One pair measurement and when it was taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkMeasurement {
    pub mbps: f64,
    pub measured_unix_s: i64,
}

/// Measured throughput between nodes, as of some instant.
///
/// Keyed by unordered pair: a collective runs at the pace of its slowest
/// direction, so when both directions have been measured the lower one is the
/// number worth scheduling on.
#[derive(Debug, Clone, Default)]
pub struct NetworkSnapshot {
    pairs: HashMap<(String, String), LinkMeasurement>,
}

fn key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

impl NetworkSnapshot {
    /// Record a measurement, keeping the **slower** of the two directions when
    /// both have been probed. Asymmetric links are rare and worth knowing
    /// about, but a job spanning the pair is limited by the slow way round.
    pub fn record(&mut self, from: &str, to: &str, mbps: f64, measured_unix_s: i64) {
        // `<= 0.0` would let NaN through; this rejects it too.
        if from == to || !matches!(mbps.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return;
        }
        let entry = self.pairs.entry(key(from, to)).or_insert(LinkMeasurement {
            mbps,
            measured_unix_s,
        });
        // A newer probe of the same direction replaces the old one; the other
        // direction only wins if it is slower.
        if measured_unix_s > entry.measured_unix_s || mbps < entry.mbps {
            entry.mbps = entry.mbps.min(mbps);
            entry.measured_unix_s = entry.measured_unix_s.max(measured_unix_s);
        }
    }

    /// Measured throughput between two nodes, if it was measured recently
    /// enough to still mean something.
    ///
    /// `max_age_s` of 0 disables the expiry. A stale number is not
    /// automatically wrong, but scheduling a week-old measurement as though it
    /// were current is exactly the mistake the data-age convention exists to
    /// prevent.
    pub fn between(&self, a: &str, b: &str, now: i64, max_age_s: i64) -> Option<f64> {
        let m = self.pairs.get(&key(a, b))?;
        if max_age_s > 0 && now.saturating_sub(m.measured_unix_s) > max_age_s {
            return None;
        }
        Some(m.mbps)
    }

    /// When the pair was last measured, regardless of age.
    pub fn measured_at(&self, a: &str, b: &str) -> Option<i64> {
        self.pairs.get(&key(a, b)).map(|m| m.measured_unix_s)
    }

    /// The slowest measured hop among a set of nodes.
    ///
    /// `None` when any pair has never been measured: a set whose weakest link
    /// is unknown cannot honestly be called fast, and guessing the missing hop
    /// is how a scheduler ends up preferring the one path nobody has tested.
    pub fn slowest_among(&self, nodes: &[&str], now: i64, max_age_s: i64) -> Option<f64> {
        let mut slowest = f64::INFINITY;
        for (i, a) in nodes.iter().enumerate() {
            for b in nodes.iter().skip(i + 1) {
                slowest = slowest.min(self.between(a, b, now, max_age_s)?);
            }
        }
        slowest.is_finite().then_some(slowest)
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }
}

impl NetworkSnapshot {
    /// A shared empty snapshot, so a caller with no measurements does not have
    /// to keep one alive just to borrow it.
    pub fn none() -> &'static NetworkSnapshot {
        static EMPTY: std::sync::OnceLock<NetworkSnapshot> = std::sync::OnceLock::new();
        EMPTY.get_or_init(NetworkSnapshot::default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400;

    #[test]
    fn a_pair_reads_the_same_in_either_direction() {
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 100);
        assert_eq!(net.between("a", "b", 100, 0), Some(940.0));
        assert_eq!(net.between("b", "a", 100, 0), Some(940.0));
    }

    #[test]
    fn an_asymmetric_link_is_remembered_at_its_slower_end() {
        // A job spanning the pair runs at the slow way round, so that is the
        // number a placement decision should see.
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 100);
        net.record("b", "a", 92.0, 100);
        assert_eq!(net.between("a", "b", 100, 0), Some(92.0));
    }

    #[test]
    fn a_stale_measurement_is_withheld_rather_than_used() {
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 0);
        assert_eq!(net.between("a", "b", DAY - 1, DAY), Some(940.0));
        assert_eq!(
            net.between("a", "b", DAY + 1, DAY),
            None,
            "older than the limit must read as unmeasured, not as fast"
        );
        // The timestamp survives so a caller can still say how old it was.
        assert_eq!(net.measured_at("a", "b"), Some(0));
    }

    #[test]
    fn expiry_can_be_switched_off() {
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 0);
        assert_eq!(net.between("a", "b", 10 * DAY, 0), Some(940.0));
    }

    #[test]
    fn the_slowest_hop_is_what_a_set_is_worth() {
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 100);
        net.record("b", "c", 90.0, 100);
        net.record("a", "c", 940.0, 100);
        assert_eq!(net.slowest_among(&["a", "b", "c"], 100, 0), Some(90.0));
        assert_eq!(net.slowest_among(&["a", "c"], 100, 0), Some(940.0));
    }

    #[test]
    fn an_unmeasured_hop_makes_the_whole_set_unknown() {
        // Rather than scoring the set on the pairs that happen to be known,
        // which would favour exactly the path nobody has tested.
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 940.0, 100);
        assert_eq!(net.slowest_among(&["a", "b", "c"], 100, 0), None);
    }

    #[test]
    fn a_single_node_has_no_hops_to_be_slow() {
        let net = NetworkSnapshot::default();
        assert_eq!(net.slowest_among(&["only"], 0, 0), None);
    }

    #[test]
    fn failed_probes_are_not_recorded_as_zero() {
        // `ferro net` reports an error as mbps = 0. Storing that would make an
        // unreachable pair look like the slowest possible link rather than an
        // unknown one, and the scheduler would then rank it.
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 0.0, 100);
        assert!(net.is_empty());
        assert_eq!(net.between("a", "b", 100, 0), None);
    }

    #[test]
    fn a_node_is_not_measured_against_itself() {
        let mut net = NetworkSnapshot::default();
        net.record("a", "a", 10_000.0, 100);
        assert!(net.is_empty());
    }

    #[test]
    fn a_newer_probe_replaces_an_older_one() {
        let mut net = NetworkSnapshot::default();
        net.record("a", "b", 90.0, 100);
        net.record("a", "b", 940.0, 200);
        assert_eq!(
            net.between("a", "b", 200, 0),
            Some(90.0),
            "the slower direction still governs"
        );
        assert_eq!(net.measured_at("a", "b"), Some(200));
    }
}
