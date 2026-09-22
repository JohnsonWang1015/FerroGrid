//! Offline evaluation of FerroGrid's scheduling policies.
//!
//! The point of this crate is a comparison nobody has to take on trust: the
//! policies under test are the ones the controller runs, the workloads are
//! generated from a seed anybody can repeat, and every number written out
//! carries the configuration that produced it.
//!
//! What it does **not** model is everything below scheduling -- no NCCL, no
//! container startup, no contention between jobs sharing a node's PCIe. A job
//! here runs for exactly the duration it was given. That makes waiting time,
//! fairness and starvation trustworthy, and makes throughput a statement about
//! the *scheduler*, not about the hardware.

pub mod metrics;
pub mod rng;
pub mod sim;
pub mod workload;

pub use sim::{Dispatch, JobRecord, SimOutcome, TraceEvent};

use serde::Serialize;
use workload::{ClusterSpec, JobClass, NodeFailure, UserSpec, WorkloadSpec};

/// One workload from the catalogue, with the cluster it belongs on.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub letter: &'static str,
    pub about: &'static str,
    pub workload: WorkloadSpec,
    pub cluster: ClusterSpec,
}

fn user(name: &str, weight: f64, priority: u32) -> UserSpec {
    UserSpec {
        name: name.into(),
        weight,
        priority,
    }
}

fn class(name: &str, weight: f64, nodes: u32, gpus: u32, lo: u64, hi: u64) -> JobClass {
    JobClass {
        name: name.into(),
        weight,
        nodes,
        gpus_per_node: gpus,
        min_duration_s: lo,
        max_duration_s: hi,
    }
}

fn base(name: &str, seed: u64, jobs: usize, interarrival: f64) -> WorkloadSpec {
    WorkloadSpec {
        name: name.into(),
        seed,
        jobs,
        mean_interarrival_s: interarrival,
        users: vec![user("alice", 1.0, 50), user("bob", 1.0, 50)],
        classes: vec![class("small", 1.0, 1, 1, 60, 300)],
        estimate_fraction: 1.0,
        estimate_error: 0.0,
        fail_node: None,
    }
}

/// The workloads of §51.
///
/// Workload F (controller failure) is deliberately absent: it is a question
/// about persistence and reconciliation, not about scheduling, and it cannot
/// be answered by a simulator that has no controller to restart. It belongs
/// with the recovery work and is measured on the real cluster.
pub fn catalogue() -> Vec<Scenario> {
    vec![
        Scenario {
            letter: "A",
            about: "uniform small jobs on a homogeneous cluster",
            workload: base("A-uniform", 1001, 200, 25.0),
            cluster: ClusterSpec::homogeneous(2, 4),
        },
        Scenario {
            letter: "B",
            about: "short and long jobs mixed, with honest duration estimates",
            workload: WorkloadSpec {
                classes: vec![
                    class("short", 5.0, 1, 1, 30, 120),
                    class("long", 1.0, 1, 1, 900, 1800),
                ],
                ..base("B-mixed", 1002, 200, 25.0)
            },
            cluster: ClusterSpec::homogeneous(2, 4),
        },
        Scenario {
            letter: "C",
            about: "one user floods the cluster; three others are polite",
            workload: WorkloadSpec {
                users: vec![
                    user("hog", 6.0, 50),
                    user("alice", 1.0, 50),
                    user("bob", 1.0, 50),
                    user("carol", 1.0, 50),
                ],
                ..base("C-unfair", 1003, 240, 20.0)
            },
            cluster: ClusterSpec::homogeneous(2, 4),
        },
        Scenario {
            letter: "D",
            about: "large distributed jobs competing with a stream of small ones",
            workload: WorkloadSpec {
                classes: vec![
                    class("distributed", 1.0, 2, 2, 600, 1200),
                    class("small", 6.0, 1, 1, 60, 180),
                ],
                ..base("D-large-and-small", 1004, 200, 22.0)
            },
            cluster: ClusterSpec::homogeneous(2, 4),
        },
        Scenario {
            letter: "E",
            about: "heterogeneous GPUs and an uneven fabric",
            workload: WorkloadSpec {
                classes: vec![
                    class("single", 4.0, 1, 1, 120, 600),
                    class("paired", 1.0, 2, 1, 300, 900),
                ],
                ..base("E-heterogeneous", 1005, 180, 30.0)
            },
            cluster: ClusterSpec::heterogeneous(),
        },
        Scenario {
            letter: "G",
            about: "a node disappears halfway through the run",
            workload: WorkloadSpec {
                fail_node: Some(NodeFailure {
                    node_id: "gpu-b".into(),
                    at_s: 1_500,
                }),
                ..base("G-node-failure", 1007, 200, 25.0)
            },
            cluster: ClusterSpec::homogeneous(2, 4),
        },
        Scenario {
            letter: "I",
            about: "users whose work is genuinely of different urgency",
            workload: WorkloadSpec {
                // Without this the priority, aging and FIFO policies are
                // handed identical input and produce identical output, and the
                // comparison says nothing about any of them.
                users: vec![
                    user("routine", 4.0, 20),
                    user("normal", 2.0, 50),
                    user("urgent", 1.0, 90),
                ],
                ..base("I-mixed-priority", 1009, 240, 20.0)
            },
            cluster: ClusterSpec::homogeneous(2, 4),
        },
        Scenario {
            letter: "H",
            about: "duration estimates that are often absent and frequently wrong",
            workload: WorkloadSpec {
                classes: vec![
                    class("short", 5.0, 1, 1, 30, 120),
                    class("long", 1.0, 1, 1, 900, 1800),
                ],
                estimate_fraction: 0.6,
                estimate_error: 0.5,
                ..base("H-bad-estimates", 1008, 200, 25.0)
            },
            cluster: ClusterSpec::homogeneous(2, 4),
        },
    ]
}

/// Everything needed to repeat a run (§56).
#[derive(Debug, Clone, Serialize)]
pub struct RunMetadata {
    pub scenario: String,
    pub scenario_about: String,
    pub queue_policy: String,
    pub placement_policy: String,
    pub dispatch: &'static str,
    pub seed: u64,
    pub jobs: usize,
    pub cluster_nodes: usize,
    pub cluster_gpus: usize,
    pub min_free_vram_gib: u64,
    pub git_commit: String,
    pub generated_unix_s: i64,
}

/// One row of `summary.csv`: the metadata and the metrics together, because a
/// metric without its configuration is not a result.
#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    #[serde(flatten)]
    pub meta: RunMetadata,
    #[serde(flatten)]
    pub metrics: metrics::Metrics,
}

/// The commit the numbers were produced at, or a marker saying it is unknown.
///
/// Recorded rather than assumed, and explicitly flagged when the tree is dirty:
/// a result attributed to a commit that does not contain the code that produced
/// it is worse than one attributed to nothing.
pub fn git_commit() -> String {
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    match rev {
        Some(r) if dirty => format!("{r}-dirty"),
        Some(r) => r,
        None => "unknown".into(),
    }
}

pub fn now_unix_s() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scenario_is_runnable_and_distinct() {
        let all = catalogue();
        assert!(all.len() >= 6);
        let letters: std::collections::BTreeSet<&str> = all.iter().map(|s| s.letter).collect();
        assert_eq!(letters.len(), all.len(), "duplicate scenario letters");
        for s in &all {
            assert!(s.workload.jobs > 0, "{} has no jobs", s.letter);
            assert!(s.cluster.total_gpus() > 0, "{} has no GPUs", s.letter);
            assert!(!s.about.is_empty());
            let jobs = s.workload.generate();
            assert_eq!(jobs.len(), s.workload.jobs);
            // Every job must be satisfiable by the cluster it is paired with,
            // or the scenario measures nothing but rejection.
            let widest = s
                .cluster
                .nodes
                .iter()
                .map(|n| n.gpus.len() as u32)
                .max()
                .unwrap_or(0);
            for j in &jobs {
                assert!(
                    j.gpus_per_node <= widest && j.nodes as usize <= s.cluster.nodes.len(),
                    "{}: job {} wants {}x{} which {} cannot satisfy",
                    s.letter,
                    j.id,
                    j.nodes,
                    j.gpus_per_node,
                    s.workload.name
                );
            }
        }
    }

    #[test]
    fn scenario_c_really_is_unfair() {
        // The workload has to contain the problem before fair share can be
        // shown to fix it.
        let c = catalogue().into_iter().find(|s| s.letter == "C").unwrap();
        let jobs = c.workload.generate();
        let hog = jobs.iter().filter(|j| j.user == "hog").count();
        assert!(
            hog as f64 / jobs.len() as f64 > 0.5,
            "the hog submitted only {hog}/{}",
            jobs.len()
        );
    }

    #[test]
    fn scenario_h_really_has_bad_estimates() {
        let h = catalogue().into_iter().find(|s| s.letter == "H").unwrap();
        let jobs = h.workload.generate();
        let absent = jobs
            .iter()
            .filter(|j| j.estimated_duration_s.is_none())
            .count();
        let wrong = jobs
            .iter()
            .filter_map(|j| j.estimated_duration_s.map(|e| (e as i64, j.duration_s)))
            .filter(|(e, d)| e != d)
            .count();
        assert!(absent > 40, "only {absent} jobs declined to estimate");
        assert!(wrong > 40, "only {wrong} estimates were wrong");
    }

    #[test]
    fn the_commit_is_recorded_or_honestly_unknown() {
        let c = git_commit();
        assert!(!c.is_empty());
        assert!(
            c == "unknown" || c.len() >= 7,
            "unexpected commit string {c:?}"
        );
    }
}
