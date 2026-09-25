//! Pure admission decisions shared by the controller and offline simulator.

use std::collections::BTreeMap;

/// A pure source of one user and their GPU limit. CLI parsing stays in the
/// controller; the simulator supplies tuples directly.
pub trait QuotaLimitSpec {
    fn into_quota_limit(self) -> (String, u32);
}

impl QuotaLimitSpec for (String, u32) {
    fn into_quota_limit(self) -> (String, u32) {
        self
    }
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum QuotaTableError {
    #[error("duplicate user quota for `{user}`")]
    DuplicateUser { user: String },
}

/// Immutable per-user GPU limits. Atomic reservations remain controller-owned.
#[derive(Debug, Clone, Default)]
pub struct QuotaTable {
    per_user: BTreeMap<String, u32>,
}

impl QuotaTable {
    /// Build a table, rejecting duplicate names instead of silently replacing one.
    pub fn from_specs<S: QuotaLimitSpec>(
        specs: impl IntoIterator<Item = S>,
    ) -> Result<Self, QuotaTableError> {
        let mut per_user = BTreeMap::new();
        for spec in specs {
            let (user, gpus) = spec.into_quota_limit();
            if per_user.insert(user.clone(), gpus).is_some() {
                return Err(QuotaTableError::DuplicateUser { user });
            }
        }
        Ok(Self { per_user })
    }

    pub fn quota_for(&self, user: &str) -> Option<u32> {
        self.per_user.get(user).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.per_user.is_empty()
    }

    pub fn users(&self) -> impl Iterator<Item = (&str, u32)> {
        self.per_user
            .iter()
            .map(|(user, gpus)| (user.as_str(), *gpus))
    }

    /// GPU headroom before a new request. `None` means unlimited.
    pub fn remaining(&self, user: &str, held: u32) -> Option<u32> {
        self.quota_for(user).map(|limit| limit.saturating_sub(held))
    }

    /// Classify a request against a snapshot of held GPUs. A later real
    /// reservation must repeat this check atomically with GPU ownership changes.
    pub fn decide(&self, user: &str, held: u32, requested: u32) -> QuotaDecision {
        let Some(limit) = self.quota_for(user) else {
            return QuotaDecision::Unlimited;
        };
        if requested > limit {
            return QuotaDecision::Reject { requested, limit };
        }
        if held.saturating_add(requested) > limit {
            return QuotaDecision::Wait {
                held,
                requested,
                limit,
            };
        }
        QuotaDecision::Admitted {
            remaining: limit - held - requested,
        }
    }
}

/// Result of comparing a job request with its user's configured quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaDecision {
    Unlimited,
    Admitted {
        remaining: u32,
    },
    Wait {
        held: u32,
        requested: u32,
        limit: u32,
    },
    Reject {
        requested: u32,
        limit: u32,
    },
}

impl QuotaDecision {
    pub fn message(self, user: &str) -> Option<String> {
        match self {
            Self::Unlimited | Self::Admitted { .. } => None,
            Self::Wait {
                held,
                requested,
                limit,
            } => Some(format!(
                "GPU quota for user `{user}` is {limit}; {held} GPU(s) are currently held and this job needs {requested} more"
            )),
            Self::Reject { requested, limit } => Some(format!(
                "GPU quota for user `{user}` is {limit}, but this job needs {requested} GPU(s) and can never fit"
            )),
        }
    }

    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Wait { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quota_classifies_unlimited_admitted_wait_and_permanent_reject() {
        let table =
            QuotaTable::from_specs([("alice".to_string(), 2), ("bob".to_string(), 1)]).unwrap();

        assert_eq!(table.decide("carol", 99, 1), QuotaDecision::Unlimited);
        assert_eq!(
            table.decide("alice", 0, 1),
            QuotaDecision::Admitted { remaining: 1 }
        );
        assert_eq!(
            table.decide("alice", 2, 1),
            QuotaDecision::Wait {
                held: 2,
                requested: 1,
                limit: 2
            }
        );
        assert_eq!(
            table.decide("bob", 0, 2),
            QuotaDecision::Reject {
                requested: 2,
                limit: 1
            }
        );
    }

    #[test]
    fn quotas_are_independent_and_duplicate_limits_are_rejected() {
        let table =
            QuotaTable::from_specs([("alice".to_string(), 1), ("bob".to_string(), 3)]).unwrap();

        assert!(matches!(
            table.decide("alice", 1, 1),
            QuotaDecision::Wait { .. }
        ));
        assert_eq!(
            table.decide("bob", 0, 3),
            QuotaDecision::Admitted { remaining: 0 }
        );
        assert!(
            QuotaTable::from_specs([("alice".to_string(), 1), ("alice".to_string(), 2),]).is_err()
        );
    }
}
