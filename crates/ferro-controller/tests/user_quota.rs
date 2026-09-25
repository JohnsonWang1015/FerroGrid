use ferro_controller::quota::{QuotaDecision, QuotaTable, UserQuotaSpec};
use ferro_controller::registry::{now_s, Job, QuotaReservationError, Registry};
use ferro_controller::service::{queue_pass, ControllerService};
use ferro_controller::store::{Change, Store};
use ferro_proto::controller_server::Controller;
use ferro_proto::node_agent_server::{NodeAgent, NodeAgentServer};
use ferro_proto::{Gpu, JobPhase, JobPlacement, JobPlan, JobStatus, NodeInfo, SubmitJobRequest};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const VRAM_FLOOR: u64 = 8 << 30;

struct TempDb(PathBuf);

impl TempDb {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "ferrogrid-user-quota-{}-{nonce}",
            std::process::id()
        )))
    }

    fn path(&self) -> PathBuf {
        self.0.join("state/controller.db")
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn quota(user: &str, gpus: u32) -> QuotaTable {
    QuotaTable::from_specs([UserQuotaSpec {
        user: user.into(),
        gpus,
    }])
    .unwrap()
}

fn node(gpus: u32) -> NodeInfo {
    NodeInfo {
        node_id: "gpu-a".into(),
        address: String::new(),
        gpus: (0..gpus)
            .map(|index| Gpu {
                index,
                uuid: format!("uuid-{index}"),
                name: "Test GPU".into(),
                memory_total_b: 24 << 30,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }
}

fn planned_job(id: &str, user: &str, gpu_indices: &[u32]) -> (Job, JobPlan) {
    let plan = JobPlan {
        world_size: gpu_indices.len() as u32,
        placements: vec![JobPlacement {
            node_id: "gpu-a".into(),
            gpu_indices: gpu_indices.to_vec(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let (tx, _) = tokio::sync::broadcast::channel(4);
    let job = Job {
        job_id: id.into(),
        name: id.into(),
        submitted_by: user.into(),
        project: String::new(),
        priority: ferro_sched::DEFAULT_PRIORITY,
        estimated_duration_s: None,
        timeout_s: 0,
        plan: plan.clone(),
        per_node: Default::default(),
        submitted: now_s(),
        logs: Default::default(),
        nccl_errors: Vec::new(),
        metrics: Default::default(),
        util_sum: 0.0,
        util_n: 0,
        tx,
        queued: false,
        queue_req: None,
        queue_deadline: 0,
        node_verdicts: Vec::new(),
        warnings: Vec::new(),
        queue_message: String::new(),
        placement: None,
    };
    (job, plan)
}

fn service(registry: Arc<Registry>) -> ControllerService {
    ControllerService {
        registry,
        plugins: Default::default(),
        heartbeat_interval_s: 3,
        sched: ferro_sched::SchedulerConfig {
            master_port: 29500,
            min_free_vram_b: VRAM_FLOOR,
            network_max_age_s: 86_400,
            placement_weights: Default::default(),
        },
        placement: Arc::new(ferro_sched::PerformancePlacement),
    }
}

fn request(user: &str, gpus: u32, queue: bool) -> SubmitJobRequest {
    SubmitJobRequest {
        script: "train.py".into(),
        nodes: 1,
        gpus_per_node: gpus,
        submitted_by: user.into(),
        queue,
        ..Default::default()
    }
}

#[derive(Clone)]
struct MockAgent {
    launches: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl NodeAgent for MockAgent {
    async fn get_node_info(
        &self,
        _request: tonic::Request<ferro_proto::GetNodeInfoRequest>,
    ) -> Result<tonic::Response<NodeInfo>, tonic::Status> {
        Ok(tonic::Response::new(NodeInfo::default()))
    }

    async fn describe_process(
        &self,
        _request: tonic::Request<ferro_proto::DescribeProcessRequest>,
    ) -> Result<tonic::Response<ferro_proto::ProcessDetail>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }

    async fn launch_job(
        &self,
        _request: tonic::Request<ferro_proto::LaunchJobRequest>,
    ) -> Result<tonic::Response<ferro_proto::LaunchJobResponse>, tonic::Status> {
        self.launches.fetch_add(1, Ordering::SeqCst);
        Ok(tonic::Response::new(ferro_proto::LaunchJobResponse {
            launched: true,
            message: "accepted by test agent".into(),
        }))
    }

    async fn stop_job(
        &self,
        _request: tonic::Request<ferro_proto::StopJobRequest>,
    ) -> Result<tonic::Response<ferro_proto::StopJobResponse>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }

    async fn ping(
        &self,
        _request: tonic::Request<ferro_proto::PingRequest>,
    ) -> Result<tonic::Response<ferro_proto::PingResponse>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }

    async fn benchmark(
        &self,
        _request: tonic::Request<ferro_proto::BenchmarkRequest>,
    ) -> Result<tonic::Response<ferro_proto::BenchmarkResponse>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }

    async fn net_sink(
        &self,
        _request: tonic::Request<ferro_proto::NetSinkRequest>,
    ) -> Result<tonic::Response<ferro_proto::NetSinkResponse>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }

    async fn net_probe(
        &self,
        _request: tonic::Request<ferro_proto::NetProbeRequest>,
    ) -> Result<tonic::Response<ferro_proto::NetProbeResponse>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }

    async fn exec_plugin(
        &self,
        _request: tonic::Request<ferro_proto::ExecPluginRequest>,
    ) -> Result<tonic::Response<ferro_proto::ExecPluginResponse>, tonic::Status> {
        Ok(tonic::Response::new(Default::default()))
    }
}

async fn start_mock_agent() -> (String, tokio::sync::oneshot::Sender<()>, Arc<AtomicUsize>) {
    let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = socket.local_addr().unwrap();
    drop(socket);
    let launches = Arc::new(AtomicUsize::new(0));
    let agent_launches = launches.clone();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(NodeAgentServer::new(MockAgent {
                launches: agent_launches,
            }))
            .serve_with_shutdown(address, async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
    });
    let endpoint = format!("http://{address}");
    for _ in 0..100 {
        if let Ok(mut client) =
            ferro_proto::node_agent_client::NodeAgentClient::connect(endpoint.clone()).await
        {
            if client.ping(ferro_proto::PingRequest {}).await.is_ok() {
                return (endpoint, stop, launches);
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("test node agent did not start listening");
}

#[tokio::test]
async fn no_configured_quota_keeps_reservations_unlimited() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.upsert_node(node(4)).await;
    let (job, plan) = planned_job("job", "alice", &[0, 1, 2, 3]);
    registry.insert_job(job).await;

    registry
        .reserve_exact_with_quota(&plan, "job")
        .await
        .unwrap();
    assert_eq!(registry.user_gpus_held("alice").await, 4);
}

#[tokio::test]
async fn reservation_admits_a_job_below_the_limit() {
    let registry = Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 2)));
    registry.upsert_node(node(4)).await;
    let (job, plan) = planned_job("job", "alice", &[0, 1]);
    registry.insert_job(job).await;

    registry
        .reserve_exact_with_quota(&plan, "job")
        .await
        .unwrap();
    assert_eq!(registry.user_gpus_held("alice").await, 2);
}

#[tokio::test]
async fn restored_running_job_counts_towards_quota_before_its_node_returns() {
    let db = TempDb::new();
    let (mut running, _) = planned_job("restored", "alice", &[0, 1]);
    let started = now_s() - 60;
    let status = JobStatus {
        job_id: "restored".into(),
        node_id: "gpu-a".into(),
        phase: JobPhase::Running as i32,
        started_unix_s: started,
        ..Default::default()
    };
    running.per_node.insert("gpu-a".into(), status.clone());
    {
        let store = Store::open(&db.path()).unwrap();
        store.write(Change::Job(Box::new(running.to_record(0))));
        store.write(Change::Status {
            job_id: "restored".into(),
            node_id: "gpu-a".into(),
            status: Box::new(status),
        });
    }

    let state = Store::load(&db.path()).unwrap();
    let store = Store::open(&db.path()).unwrap();
    let registry = Arc::new(
        Registry::restore(VRAM_FLOOR, Arc::new(ferro_sched::queue::Fifo), store, state)
            .with_quotas(Arc::new(quota("alice", 2))),
    );
    let (address, stop, _) = start_mock_agent().await;
    let mut reappeared_node = node(2);
    reappeared_node.node_id = "gpu-b".into();
    reappeared_node.address = address;
    registry.upsert_node(reappeared_node).await;

    let response = Controller::submit_job(
        &service(registry.clone()),
        tonic::Request::new(request("alice", 2, false)),
    )
    .await
    .unwrap()
    .into_inner();

    assert_eq!(registry.user_gpus_held("alice").await, 2);
    assert!(
        !response.accepted,
        "the restored two-GPU plan already consumes Alice's quota: {}",
        response.message
    );
    assert!(
        response.message.contains("GPU quota"),
        "{}",
        response.message
    );
    let (mut candidate, mut candidate_plan) = planned_job("attempt", "alice", &[0, 1]);
    candidate_plan.placements[0].node_id = "gpu-b".into();
    candidate.plan = candidate_plan.clone();
    registry.insert_job(candidate).await;
    assert!(matches!(
        registry
            .reserve_exact_with_quota(&candidate_plan, "attempt")
            .await,
        Err(QuotaReservationError::Quota {
            decision: QuotaDecision::Wait {
                held: 2,
                requested: 2,
                limit: 2
            },
            ..
        })
    ));
    let usage = registry.inner.lock().await.usage_snapshot(now_s());
    assert_eq!(usage.per_user["alice"].gpus_held, 2);
    let _ = stop.send(());
}

#[tokio::test]
async fn reservation_rejects_a_request_larger_than_the_hard_limit() {
    let registry = Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 2)));
    registry.upsert_node(node(4)).await;
    let (job, plan) = planned_job("job", "alice", &[0, 1, 2]);
    registry.insert_job(job).await;

    let error = registry
        .reserve_exact_with_quota(&plan, "job")
        .await
        .expect_err("three GPUs can never fit under a quota of two");
    assert!(matches!(
        error,
        QuotaReservationError::Quota {
            decision: QuotaDecision::Reject {
                requested: 3,
                limit: 2
            },
            ..
        }
    ));
}

#[tokio::test]
async fn same_user_cannot_reserve_past_the_quota_but_another_user_can() {
    let registry = Registry::new(VRAM_FLOOR).with_quotas(Arc::new(
        QuotaTable::from_specs([
            UserQuotaSpec {
                user: "alice".into(),
                gpus: 1,
            },
            UserQuotaSpec {
                user: "bob".into(),
                gpus: 1,
            },
        ])
        .unwrap(),
    ));
    registry.upsert_node(node(4)).await;

    for (id, user, index) in [("a1", "alice", 0), ("a2", "alice", 1), ("b1", "bob", 2)] {
        let (job, plan) = planned_job(id, user, &[index]);
        registry.insert_job(job).await;
        let result = registry.reserve_exact_with_quota(&plan, id).await;
        if id == "a2" {
            assert!(matches!(
                result,
                Err(QuotaReservationError::Quota {
                    decision: QuotaDecision::Wait {
                        held: 1,
                        requested: 1,
                        limit: 1
                    },
                    ..
                })
            ));
        } else {
            result.unwrap();
        }
    }
    assert_eq!(registry.user_gpus_held("alice").await, 1);
    assert_eq!(registry.user_gpus_held("bob").await, 1);
}

#[tokio::test]
async fn quota_capacity_is_atomic_under_concurrent_reservations() {
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 2))));
    registry.upsert_node(node(16)).await;
    let mut plans = Vec::new();
    for index in 0..16 {
        let id = format!("j{index:02}");
        let (job, plan) = planned_job(&id, "alice", &[index]);
        registry.insert_job(job).await;
        plans.push((id, plan));
    }

    let tasks: Vec<_> = plans
        .into_iter()
        .map(|(id, plan)| {
            let registry = registry.clone();
            tokio::spawn(async move { registry.reserve_exact_with_quota(&plan, &id).await.is_ok() })
        })
        .collect();
    let mut granted = 0;
    for task in tasks {
        granted += usize::from(task.await.unwrap());
    }

    assert_eq!(granted, 2, "only the two quota slots may be reserved");
    assert_eq!(registry.user_gpus_held("alice").await, 2);
}

#[tokio::test]
async fn terminal_job_release_frees_its_quota_for_the_next_job() {
    let registry = Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 1)));
    registry.upsert_node(node(2)).await;
    let (first, first_plan) = planned_job("first", "alice", &[0]);
    registry.insert_job(first).await;
    registry
        .reserve_exact_with_quota(&first_plan, "first")
        .await
        .unwrap();
    let (second, second_plan) = planned_job("second", "alice", &[1]);
    registry.insert_job(second).await;
    assert!(matches!(
        registry
            .reserve_exact_with_quota(&second_plan, "second")
            .await,
        Err(QuotaReservationError::Quota {
            decision: QuotaDecision::Wait { .. },
            ..
        })
    ));

    registry
        .update_job_status(JobStatus {
            job_id: "first".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Cancelled as i32,
            ended_unix_s: now_s(),
            ..Default::default()
        })
        .await;
    registry.release_if_done("first").await;
    registry
        .reserve_exact_with_quota(&second_plan, "second")
        .await
        .unwrap();
    assert_eq!(registry.user_gpus_held("alice").await, 1);
}

#[tokio::test]
async fn a_quota_blocked_submission_without_wait_gets_a_useful_error() {
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 1))));
    registry.upsert_node(node(2)).await;
    let (holder, plan) = planned_job("holder", "alice", &[0]);
    registry.insert_job(holder).await;
    registry
        .reserve_exact_with_quota(&plan, "holder")
        .await
        .unwrap();

    let response = Controller::submit_job(
        &service(registry),
        tonic::Request::new(request("alice", 1, false)),
    )
    .await
    .unwrap()
    .into_inner();

    assert!(!response.accepted);
    assert!(
        response.message.contains("GPU quota"),
        "{}",
        response.message
    );
    assert!(
        response.message.contains("currently held"),
        "{}",
        response.message
    );
}

#[tokio::test]
async fn a_quota_blocked_submission_with_wait_remains_queued() {
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 1))));
    registry.upsert_node(node(2)).await;
    let (holder, plan) = planned_job("holder", "alice", &[0]);
    registry.insert_job(holder).await;
    registry
        .reserve_exact_with_quota(&plan, "holder")
        .await
        .unwrap();

    let response = Controller::submit_job(
        &service(registry.clone()),
        tonic::Request::new(request("alice", 1, true)),
    )
    .await
    .unwrap()
    .into_inner();

    assert!(response.accepted, "{}", response.message);
    assert!(response.queue_position > 0);
    let g = registry.inner.lock().await;
    let queued = g.jobs.get(&response.job_id).unwrap();
    assert!(queued.queued);
    assert!(queued.queue_message.contains("GPU quota"));
}

#[tokio::test]
async fn an_exact_request_larger_than_quota_is_rejected_even_with_wait() {
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 2))));
    registry.upsert_node(node(4)).await;

    let response = Controller::submit_job(
        &service(registry),
        tonic::Request::new(request("alice", 3, true)),
    )
    .await
    .unwrap()
    .into_inner();

    assert!(!response.accepted);
    assert!(
        response.message.contains("can never fit"),
        "{}",
        response.message
    );
}

#[tokio::test]
async fn auto_placement_is_capped_by_the_users_quota() {
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 2))));
    let (address, stop, launches) = start_mock_agent().await;
    let mut available = node(4);
    available.address = address;
    registry.upsert_node(available).await;
    let mut req = request("alice", 0, false);
    req.auto_place = true;

    let response = Controller::submit_job(&service(registry.clone()), tonic::Request::new(req))
        .await
        .unwrap()
        .into_inner();

    assert!(response.accepted, "{}", response.message);
    let plan = response.plan.expect("a feasible auto request has a plan");
    assert_eq!(plan.world_size, 2, "auto chose {} GPUs", plan.world_size);
    assert!(launches.load(Ordering::SeqCst) > 0, "job was not launched");
    assert_eq!(registry.user_gpus_held("alice").await, 2);
    let _ = stop.send(());
}

#[tokio::test]
async fn auto_queue_dispatch_uses_remaining_quota_for_strict_and_reserved_modes() {
    for dispatch in [
        ferro_sched::Dispatch::Strict,
        ferro_sched::Dispatch::Reserved,
    ] {
        let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 2))));
        let service = service(registry.clone());
        let mut bob = request("bob", 1, true);
        bob.name = "bob-first".into();
        let first = Controller::submit_job(&service, tonic::Request::new(bob))
            .await
            .unwrap()
            .into_inner();
        assert!(first.accepted, "{}", first.message);

        let mut alice = request("alice", 4, true);
        alice.name = "alice-auto".into();
        alice.auto_place = true;
        let auto = Controller::submit_job(&service, tonic::Request::new(alice))
            .await
            .unwrap()
            .into_inner();
        assert!(auto.accepted, "{}", auto.message);

        let queued = registry.waiting_jobs().await;
        let auto_queued = queued
            .iter()
            .find(|(job, _, _)| job.job_id == auto.job_id)
            .expect("auto job remains queued until GPUs return");
        assert_eq!(
            auto_queued.0.gpus, 2,
            "{dispatch:?} saw the uncapped request"
        );

        let (address, stop, launches) = start_mock_agent().await;
        let mut available = node(3);
        available.address = address;
        registry.upsert_node(available).await;
        let started = queue_pass(&registry, &service.placement, &service.sched, dispatch).await;

        assert!(
            started.contains(&auto.job_id),
            "{dispatch:?} left auto job waiting"
        );
        assert_eq!(registry.user_gpus_held("alice").await, 2);
        assert!(launches.load(Ordering::SeqCst) >= 2);
        let _ = stop.send(());
    }
}

#[tokio::test]
async fn usage_rpc_reports_current_holds_jobs_gpu_seconds_and_quotas() {
    let quotas = QuotaTable::from_specs([
        UserQuotaSpec {
            user: "alice".into(),
            gpus: 2,
        },
        UserQuotaSpec {
            user: "bob".into(),
            gpus: 3,
        },
    ])
    .unwrap();
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quotas)));
    registry.upsert_node(node(4)).await;
    let (mut running, plan) = planned_job("running", "alice", &[0, 1]);
    running.per_node.insert(
        "gpu-a".into(),
        JobStatus {
            job_id: "running".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Running as i32,
            started_unix_s: now_s() - 30,
            ..Default::default()
        },
    );
    registry.insert_job(running).await;
    registry
        .reserve_exact_with_quota(&plan, "running")
        .await
        .unwrap();

    let response = Controller::get_usage(
        &service(registry),
        tonic::Request::new(ferro_proto::GetUsageRequest {}),
    )
    .await
    .unwrap()
    .into_inner();
    let alice = response.users.iter().find(|u| u.user == "alice").unwrap();
    assert_eq!(alice.gpus_held, 2);
    assert_eq!(alice.running_jobs, 1);
    assert!(alice.gpu_seconds >= 60.0);
    assert_eq!(alice.gpu_quota, Some(2));
    let bob = response.users.iter().find(|u| u.user == "bob").unwrap();
    assert_eq!(bob.gpus_held, 0);
    assert_eq!(bob.running_jobs, 0);
    assert_eq!(bob.gpu_quota, Some(3));
}

#[tokio::test]
async fn queued_quota_blocked_job_launches_after_the_users_job_releases_gpus() {
    let (address, stop_agent, launches) = start_mock_agent().await;
    let registry = Arc::new(Registry::new(VRAM_FLOOR).with_quotas(Arc::new(quota("alice", 1))));
    let mut node_info = node(2);
    node_info.address = address;
    registry.upsert_node(node_info).await;

    let (mut holder, holder_plan) = planned_job("holder", "alice", &[0]);
    holder.per_node.insert(
        "gpu-a".into(),
        JobStatus {
            job_id: "holder".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Running as i32,
            started_unix_s: now_s() - 10,
            ..Default::default()
        },
    );
    registry.insert_job(holder).await;
    registry
        .reserve_exact_with_quota(&holder_plan, "holder")
        .await
        .unwrap();
    let (mut waiting, _) = planned_job("waiting", "alice", &[]);
    waiting.plan = JobPlan::default();
    waiting.queued = true;
    waiting.queue_req = Some(request("alice", 1, true));
    registry.insert_job(waiting).await;

    let placement: Arc<dyn ferro_sched::PlacementPolicy> =
        Arc::new(ferro_sched::PerformancePlacement);
    let sched = ferro_sched::SchedulerConfig {
        master_port: 29500,
        min_free_vram_b: VRAM_FLOOR,
        network_max_age_s: 86_400,
        placement_weights: Default::default(),
    };
    assert!(queue_pass(
        &registry,
        &placement,
        &sched,
        ferro_sched::Dispatch::Opportunistic
    )
    .await
    .is_empty());
    {
        let g = registry.inner.lock().await;
        assert!(g.jobs["waiting"].queued);
        assert!(g.jobs["waiting"].queue_message.contains("GPU quota"));
    }

    registry
        .update_job_status(JobStatus {
            job_id: "holder".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Succeeded as i32,
            ended_unix_s: now_s(),
            ..Default::default()
        })
        .await;
    registry.release_if_done("holder").await;
    assert_eq!(
        queue_pass(
            &registry,
            &placement,
            &sched,
            ferro_sched::Dispatch::Opportunistic
        )
        .await,
        ["waiting"]
    );
    assert_eq!(launches.load(Ordering::SeqCst), 1);
    let g = registry.inner.lock().await;
    assert!(!g.jobs["waiting"].queued);
    assert_eq!(g.jobs["waiting"].plan.world_size, 1);
    assert!(g.nodes["gpu-a"]
        .info
        .gpus
        .iter()
        .any(|gpu| gpu.allocated_job_id == "waiting"));
    drop(g);
    let _ = stop_agent.send(());
}

#[tokio::test]
async fn cancelled_job_stops_accruing_gpu_seconds() {
    let registry = Registry::new(VRAM_FLOOR);
    registry.upsert_node(node(2)).await;
    let started = now_s() - 30;
    let (mut job, plan) = planned_job("cancelled", "alice", &[0, 1]);
    job.per_node.insert(
        "gpu-a".into(),
        JobStatus {
            job_id: "cancelled".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Running as i32,
            started_unix_s: started,
            ..Default::default()
        },
    );
    registry.insert_job(job).await;
    registry.reserve_exact(&plan, "cancelled").await.unwrap();

    let ended = now_s();
    registry
        .update_job_status(JobStatus {
            job_id: "cancelled".into(),
            node_id: "gpu-a".into(),
            phase: JobPhase::Cancelled as i32,
            ended_unix_s: ended,
            ..Default::default()
        })
        .await;
    registry.release_if_done("cancelled").await;

    let usage = registry.inner.lock().await.usage_snapshot(ended + 300);
    assert_eq!(usage.gpu_seconds("alice"), ((ended - started) * 2) as f64);
    assert_eq!(usage.per_user["alice"].gpus_held, 0);
    assert_eq!(usage.per_user["alice"].running_jobs, 0);
}

#[test]
fn quota_spec_accepts_a_user_and_zero_or_positive_gpu_limit() {
    assert_eq!(
        UserQuotaSpec::from_str("alice=2").unwrap(),
        UserQuotaSpec {
            user: "alice".into(),
            gpus: 2,
        }
    );
    assert_eq!(UserQuotaSpec::from_str("bob=0").unwrap().gpus, 0);
}

#[test]
fn quota_spec_rejects_malformed_names_and_limits() {
    for invalid in [
        "=2",
        "alice=",
        "alice=-1",
        "alice=1.5",
        "alice =2",
        "alice=2=3",
    ] {
        assert!(
            UserQuotaSpec::from_str(invalid).is_err(),
            "accepted malformed quota {invalid:?}"
        );
    }
}

#[test]
fn quota_table_rejects_duplicate_users_instead_of_overriding() {
    let a = UserQuotaSpec::from_str("alice=2").unwrap();
    let b = UserQuotaSpec::from_str("alice=4").unwrap();
    assert!(QuotaTable::from_specs([a, b]).is_err());
}

#[test]
fn quota_decision_distinguishes_unlimited_admitted_wait_and_impossible() {
    let table = QuotaTable::from_specs([UserQuotaSpec::from_str("alice=2").unwrap()]).unwrap();

    assert_eq!(table.decide("bob", 99, 4), QuotaDecision::Unlimited);
    assert_eq!(
        table.decide("alice", 0, 1),
        QuotaDecision::Admitted { remaining: 1 }
    );
    assert_eq!(
        table.decide("alice", 2, 1),
        QuotaDecision::Wait {
            held: 2,
            requested: 1,
            limit: 2,
        }
    );
    assert_eq!(
        table.decide("alice", 0, 3),
        QuotaDecision::Reject {
            requested: 3,
            limit: 2,
        }
    );
}
