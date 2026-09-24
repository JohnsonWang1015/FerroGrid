//! Controller-side per-user GPU quota configuration and admission decisions.
//!
//! Quotas are a cooperative resource-management policy. `submitted_by` is
//! supplied by the client, so these limits are not an authentication boundary.

use std::collections::BTreeMap;
use std::str::FromStr;

/// One parsed `--user-quota USER=N` option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserQuotaSpec {
    pub user: String,
    pub gpus: u32,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum QuotaConfigError {
    #[error("invalid user quota `{spec}`; expected USER=N with a non-empty user and a non-negative integer GPU limit")]
    InvalidSpec { spec: String },
    #[error("duplicate user quota for `{user}`")]
    DuplicateUser { user: String },
}

impl FromStr for UserQuotaSpec {
    type Err = QuotaConfigError;

    fn from_str(spec: &str) -> Result<Self, Self::Err> {
        let invalid = || QuotaConfigError::InvalidSpec {
            spec: spec.to_string(),
        };
        let (user, gpus) = spec.split_once('=').ok_or_else(invalid)?;
        if user.is_empty()
            || user.trim() != user
            || user.chars().any(char::is_whitespace)
            || gpus.is_empty()
            || !gpus.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(invalid());
        }
        let gpus = gpus.parse().map_err(|_| invalid())?;
        Ok(Self {
            user: user.to_string(),
            gpus,
        })
    }
}

/// Immutable quota settings owned by the controller/registry.
#[derive(Debug, Clone, Default)]
pub struct QuotaTable {
    per_user: BTreeMap<String, u32>,
}

impl QuotaTable {
    /// Build a table and reject duplicate names instead of silently replacing
    /// the earlier value. A zero-GPU quota is valid and prevents all placement.
    pub fn from_specs(
        specs: impl IntoIterator<Item = UserQuotaSpec>,
    ) -> Result<Self, QuotaConfigError> {
        let mut per_user = BTreeMap::new();
        for spec in specs {
            if per_user.insert(spec.user.clone(), spec.gpus).is_some() {
                return Err(QuotaConfigError::DuplicateUser { user: spec.user });
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

    /// Classify a request against a single snapshot of held GPUs.
    ///
    /// The result is useful for immediate feedback and queueing decisions. A
    /// later reservation must still repeat this check atomically with GPU
    /// ownership changes; this snapshot alone never grants resources.
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

/// Result of comparing a job's GPU request with its user's configured quota.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaDecision {
    /// No setting exists for this user.
    Unlimited,
    /// The job fits now; `remaining` is quota headroom after granting it.
    Admitted { remaining: u32 },
    /// The request can fit later, after the user's existing jobs release GPUs.
    Wait {
        held: u32,
        requested: u32,
        limit: u32,
    },
    /// The request is larger than the user's entire configured quota.
    Reject { requested: u32, limit: u32 },
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
