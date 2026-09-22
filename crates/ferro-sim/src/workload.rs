//! Synthetic workloads, and the cluster they run on.
//!
//! Every field here is part of the experiment's identity: a result quoted
//! without its workload is not a result. The whole spec serialises into the
//! run metadata (§56) so a number can always be traced back to what produced
//! it.
//!
//! One deliberate asymmetry: a job has both a **true** duration and the
//! duration its submitter *declared*. They are allowed to differ, because on a
//! real cluster they do, and a shortest-job-first policy evaluated only against
//! honest estimates would be evaluated against a workload that does not exist.

use crate::rng::Rng;
use ferro_proto::{Gpu, NodeInfo, NodeState};
use serde::{Deserialize, Serialize};

/// One GPU model in the synthetic cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpec {
    pub model: String,
    pub vram_gib: u64,
    /// What `ferro bench` would have measured, in TFLOP/s.
    pub tflops: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    pub id: String,
    /// Negotiated link speed, Mb/s. 0 means unknown.
    pub link_mbps: u32,
    pub gpus: Vec<GpuSpec>,
}

/// The cluster a workload is run against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSpec {
    pub nodes: Vec<NodeSpec>,
    /// Measured pairwise throughput, as `ferro net` would have found it.
    #[serde(default)]
    pub links: Vec<MeasuredLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredLink {
    pub from: String,
    pub to: String,
    pub mbps: f64,
}

impl ClusterSpec {
    pub fn total_gpus(&self) -> usize {
        self.nodes.iter().map(|n| n.gpus.len()).sum()
    }

    /// The cluster as the scheduler sees it, with nothing allocated.
    ///
    /// Deliberately the same `NodeState` the controller hands to a placement
    /// policy, so the simulator exercises the real code path rather than a
    /// parallel one that happens to look similar.
    pub fn to_nodes(&self) -> Vec<NodeState> {
        self.nodes
            .iter()
            .map(|n| NodeState {
                info: Some(NodeInfo {
                    node_id: n.id.clone(),
                    address: format!("http://{}:7071", n.id),
                    nccl_address: format!("10.0.0.{}", n.id.len()),
                    link_mbps: n.link_mbps,
                    gpus: n
                        .gpus
                        .iter()
                        .enumerate()
                        .map(|(i, g)| Gpu {
                            index: i as u32,
                            uuid: format!("{}-{i}", n.id),
                            name: g.model.clone(),
                            memory_total_b: g.vram_gib << 30,
                            memory_used_b: 0,
                            bench_tflops: g.tflops,
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                }),
                healthy: true,
                last_seen_unix_s: 0,
                free_gpus: n.gpus.len() as u32,
            })
            .collect()
    }

    /// A homogeneous cluster: `nodes` machines of `per_node` identical cards.
    pub fn homogeneous(nodes: usize, per_node: usize) -> Self {
        Self {
            nodes: (0..nodes)
                .map(|i| NodeSpec {
                    id: format!("gpu-{}", (b'a' + i as u8) as char),
                    link_mbps: 1000,
                    gpus: (0..per_node)
                        .map(|_| GpuSpec {
                            model: "RTX 4090".into(),
                            vram_gib: 24,
                            tflops: 82.0,
                        })
                        .collect(),
                })
                .collect(),
            links: Vec::new(),
        }
    }

    /// A cluster of genuinely different hardware, which is FerroGrid's actual
    /// situation and the case §51's workload E is about.
    pub fn heterogeneous() -> Self {
        let card = |model: &str, vram: u64, tflops: f64| GpuSpec {
            model: model.into(),
            vram_gib: vram,
            tflops,
        };
        // Node order is deliberately *not* performance order. If the
        // alphabetically first node also held the fastest cards, first-fit and
        // performance-aware placement would coincide by accident and the
        // comparison between them would measure nothing.
        Self {
            nodes: vec![
                NodeSpec {
                    id: "gpu-a".into(),
                    // Slowest cards, and behind the slow path. Everything a
                    // naive scan order would reach for first.
                    link_mbps: 1000,
                    gpus: vec![card("RTX 3090", 24, 35.0)],
                },
                NodeSpec {
                    id: "gpu-b".into(),
                    link_mbps: 1000,
                    gpus: vec![card("RTX A6000", 48, 38.0), card("RTX A6000", 48, 38.0)],
                },
                NodeSpec {
                    id: "gpu-c".into(),
                    link_mbps: 1000,
                    gpus: vec![card("RTX 4090", 24, 82.0), card("RTX 4090", 24, 82.0)],
                },
            ],
            links: vec![
                MeasuredLink {
                    from: "gpu-a".into(),
                    to: "gpu-b".into(),
                    mbps: 940.0,
                },
                MeasuredLink {
                    from: "gpu-a".into(),
                    to: "gpu-c".into(),
                    mbps: 90.0,
                },
                MeasuredLink {
                    from: "gpu-b".into(),
                    to: "gpu-c".into(),
                    mbps: 90.0,
                },
            ],
        }
    }
}

/// One shape of job the generator can emit, and how often.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobClass {
    pub name: String,
    /// Relative frequency. Classes with weight 0 never appear.
    pub weight: f64,
    pub nodes: u32,
    pub gpus_per_node: u32,
    pub min_duration_s: u64,
    pub max_duration_s: u64,
}

/// How each user behaves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSpec {
    pub name: String,
    /// Share of submissions. A user with three times the weight submits
    /// roughly three times as many jobs, which is how workload C creates the
    /// unfairness fair share is supposed to correct.
    pub weight: f64,
    /// Priority every job of theirs carries.
    pub priority: u32,
}

/// Everything that defines a run's input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub name: String,
    pub seed: u64,
    pub jobs: usize,
    /// Mean seconds between arrivals. Exponential intervals make the arrival
    /// process Poisson.
    pub mean_interarrival_s: f64,
    pub users: Vec<UserSpec>,
    pub classes: Vec<JobClass>,
    /// Fraction of jobs whose submitter declared an estimated duration at all.
    /// The rest declare nothing, which SJF must handle without inventing one.
    #[serde(default = "one")]
    pub estimate_fraction: f64,
    /// How wrong a declared estimate may be, as a fraction either way. 0.0
    /// means everybody is honest; 0.5 means estimates land anywhere from half
    /// to one and a half times the truth.
    #[serde(default)]
    pub estimate_error: f64,
    /// Remove this node partway through the run, for the node-failure
    /// experiment. `None` for a healthy cluster.
    #[serde(default)]
    pub fail_node: Option<NodeFailure>,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFailure {
    pub node_id: String,
    pub at_s: i64,
}

/// One synthetic job.
#[derive(Debug, Clone, Serialize)]
pub struct SimJob {
    pub id: String,
    pub user: String,
    pub arrival_s: i64,
    /// How long it will actually run for once it starts.
    pub duration_s: i64,
    /// What the submitter claimed, which may be wrong or absent.
    pub estimated_duration_s: Option<u32>,
    pub priority: u32,
    pub nodes: u32,
    pub gpus_per_node: u32,
    pub class: String,
}

impl SimJob {
    pub fn gpus(&self) -> u32 {
        self.nodes.max(1) * self.gpus_per_node.max(1)
    }
}

impl WorkloadSpec {
    /// Generate the jobs. Pure in the seed: the same spec always produces the
    /// same list, which is what makes a comparison between policies a
    /// comparison between policies.
    pub fn generate(&self) -> Vec<SimJob> {
        let mut rng = Rng::seeded(self.seed);
        let user_weights: Vec<f64> = self.users.iter().map(|u| u.weight).collect();
        let class_weights: Vec<f64> = self.classes.iter().map(|c| c.weight).collect();

        let mut clock = 0.0f64;
        let mut jobs = Vec::with_capacity(self.jobs);
        for n in 0..self.jobs {
            clock += rng.exponential(self.mean_interarrival_s);
            let user = &self.users[rng.weighted(&user_weights).min(self.users.len() - 1)];
            let class = &self.classes[rng.weighted(&class_weights).min(self.classes.len() - 1)];
            let duration = rng.range(class.min_duration_s, class.max_duration_s) as i64;

            // What the submitter says, if anything.
            let estimated = if rng.unit() < self.estimate_fraction {
                let error = if self.estimate_error > 0.0 {
                    1.0 + (rng.unit() * 2.0 - 1.0) * self.estimate_error
                } else {
                    1.0
                };
                Some(((duration as f64 * error).max(1.0)) as u32)
            } else {
                None
            };

            jobs.push(SimJob {
                id: format!("j{n:04}"),
                user: user.name.clone(),
                arrival_s: clock as i64,
                duration_s: duration.max(1),
                estimated_duration_s: estimated,
                priority: user.priority,
                nodes: class.nodes,
                gpus_per_node: class.gpus_per_node,
                class: class.name.clone(),
            });
        }
        jobs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> WorkloadSpec {
        WorkloadSpec {
            name: "test".into(),
            seed: 1234,
            jobs: 200,
            mean_interarrival_s: 10.0,
            users: vec![
                UserSpec {
                    name: "alice".into(),
                    weight: 3.0,
                    priority: 50,
                },
                UserSpec {
                    name: "bob".into(),
                    weight: 1.0,
                    priority: 50,
                },
            ],
            classes: vec![
                JobClass {
                    name: "small".into(),
                    weight: 4.0,
                    nodes: 1,
                    gpus_per_node: 1,
                    min_duration_s: 30,
                    max_duration_s: 120,
                },
                JobClass {
                    name: "large".into(),
                    weight: 1.0,
                    nodes: 2,
                    gpus_per_node: 2,
                    min_duration_s: 600,
                    max_duration_s: 1200,
                },
            ],
            estimate_fraction: 1.0,
            estimate_error: 0.0,
            fail_node: None,
        }
    }

    #[test]
    fn the_same_spec_generates_the_same_jobs() {
        // Without this, comparing two policies would compare two workloads.
        let a = spec().generate();
        let b = spec().generate();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                (&x.id, &x.user, x.arrival_s, x.duration_s, x.nodes),
                (&y.id, &y.user, y.arrival_s, y.duration_s, y.nodes)
            );
        }
    }

    #[test]
    fn a_different_seed_generates_a_different_workload() {
        let mut other = spec();
        other.seed = 4321;
        let a = spec().generate();
        let b = other.generate();
        assert_ne!(
            a.iter().map(|j| j.duration_s).collect::<Vec<_>>(),
            b.iter().map(|j| j.duration_s).collect::<Vec<_>>()
        );
    }

    #[test]
    fn arrivals_are_ordered_and_start_near_zero() {
        let jobs = spec().generate();
        assert!(jobs[0].arrival_s < 60);
        for pair in jobs.windows(2) {
            assert!(
                pair[0].arrival_s <= pair[1].arrival_s,
                "arrivals must not go backwards"
            );
        }
    }

    #[test]
    fn user_weights_shape_who_submits() {
        let jobs = spec().generate();
        let alice = jobs.iter().filter(|j| j.user == "alice").count();
        let bob = jobs.iter().filter(|j| j.user == "bob").count();
        let ratio = alice as f64 / bob as f64;
        assert!((ratio - 3.0).abs() < 0.8, "alice/bob was {ratio}");
    }

    #[test]
    fn durations_stay_inside_their_class() {
        for j in spec().generate() {
            let (lo, hi) = match j.class.as_str() {
                "small" => (30, 120),
                _ => (600, 1200),
            };
            assert!(
                (lo..=hi).contains(&j.duration_s),
                "{} ran {} outside {lo}..{hi}",
                j.id,
                j.duration_s
            );
        }
    }

    #[test]
    fn an_honest_workload_declares_exactly_the_truth() {
        for j in spec().generate() {
            assert_eq!(j.estimated_duration_s, Some(j.duration_s as u32));
        }
    }

    #[test]
    fn estimates_can_be_absent_or_wrong() {
        let mut s = spec();
        s.estimate_fraction = 0.5;
        s.estimate_error = 0.5;
        let jobs = s.generate();

        let silent = jobs
            .iter()
            .filter(|j| j.estimated_duration_s.is_none())
            .count();
        assert!(
            (60..140).contains(&silent),
            "expected about half to say nothing, got {silent}/200"
        );
        let wrong = jobs
            .iter()
            .filter_map(|j| j.estimated_duration_s.map(|e| (e as i64, j.duration_s)))
            .filter(|(e, d)| e != d)
            .count();
        assert!(wrong > 0, "with 50% error, some estimate should be wrong");
    }

    #[test]
    fn a_cluster_renders_as_the_scheduler_sees_it() {
        let nodes = ClusterSpec::homogeneous(2, 4).to_nodes();
        assert_eq!(nodes.len(), 2);
        let info = nodes[0].info.as_ref().unwrap();
        assert_eq!(info.gpus.len(), 4);
        assert_eq!(info.gpus[0].index, 0);
        assert_eq!(info.gpus[3].index, 3);
        assert!(info.gpus.iter().all(|g| g.allocated_job_id.is_empty()));
        assert_eq!(ClusterSpec::homogeneous(2, 4).total_gpus(), 8);
    }

    #[test]
    fn the_heterogeneous_cluster_is_actually_heterogeneous() {
        // Workload E depends on this being true rather than nominally so.
        let spec = ClusterSpec::heterogeneous();
        let models: std::collections::BTreeSet<&str> = spec
            .nodes
            .iter()
            .flat_map(|n| n.gpus.iter().map(|g| g.model.as_str()))
            .collect();
        assert!(models.len() >= 3, "got {models:?}");
        assert!(spec.links.iter().any(|l| l.mbps < 100.0), "no slow path");
    }
}
