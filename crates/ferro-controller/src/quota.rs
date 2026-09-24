//! Controller-side quota CLI parsing and compatibility exports.
//!
//! The table and pure admission rule are shared with `ferro-sim`; parsing
//! `--user-quota USER=N` remains controller-owned. Atomic reservations and
//! registry locking also remain in the controller.

pub use ferro_admission::{QuotaDecision, QuotaTable, QuotaTableError};

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

impl From<QuotaTableError> for QuotaConfigError {
    fn from(error: QuotaTableError) -> Self {
        match error {
            QuotaTableError::DuplicateUser { user } => Self::DuplicateUser { user },
        }
    }
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

impl ferro_admission::QuotaLimitSpec for UserQuotaSpec {
    fn into_quota_limit(self) -> (String, u32) {
        (self.user, self.gpus)
    }
}
