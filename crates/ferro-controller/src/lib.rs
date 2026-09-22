//! FerroGrid controller: cluster registry, job orchestrator, gRPC surface.
//!
//! This is a library first and a binary second. The binary in `main.rs` is a
//! thin wrapper that parses flags and starts the server; everything worth
//! testing lives here, so integration tests, benchmarks and the offline
//! scheduler harness can link it directly instead of shelling out to a process.
//!
//! Scheduling policy itself is not here -- it lives in the `ferro-sched` crate,
//! which is pure and synchronous so the same policy objects can be driven
//! without a cluster.

pub mod metrics;
pub mod plugins;
pub mod registry;
pub mod service;
pub mod store;
