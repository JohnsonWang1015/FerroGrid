//! Generated gRPC bindings for the FerroGrid control plane.

pub mod v1 {
    tonic::include_proto!("ferrogrid.v1");
}

pub use v1::*;

impl JobPhase {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobPhase::Succeeded | JobPhase::Failed | JobPhase::Cancelled
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            JobPhase::Unspecified => "unknown",
            JobPhase::Pending => "pending",
            JobPhase::Launching => "launching",
            JobPhase::Running => "running",
            JobPhase::Succeeded => "succeeded",
            JobPhase::Failed => "failed",
            JobPhase::Cancelled => "cancelled",
        }
    }
}
