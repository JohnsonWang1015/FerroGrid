//! The `Controller` gRPC service: agent registration plus the CLI-facing API.

use crate::registry::{now_s, Job, Registry};
use crate::scheduler;
use ferro_proto::controller_server::Controller;
use ferro_proto::node_agent_client::NodeAgentClient;
use ferro_proto::*;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::ReceiverStream, Stream};
use tonic::{Request, Response, Status};

pub struct ControllerService {
    pub registry: Arc<Registry>,
    pub plugins: crate::plugins::Registry,
    pub master_port: u32,
    pub heartbeat_interval_s: u32,
    pub min_free_vram_b: u64,
}

type LogStream = Pin<Box<dyn Stream<Item = Result<LogLine, Status>> + Send>>;

#[tonic::async_trait]
impl Controller for ControllerService {
    async fn register_node(
        &self,
        req: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let node = req
            .into_inner()
            .node
            .ok_or_else(|| Status::invalid_argument("missing node info"))?;

        if node.node_id.is_empty() {
            return Err(Status::invalid_argument("node_id must not be empty"));
        }
        tracing::info!(
            node = %node.node_id,
            gpus = node.gpus.len(),
            addr = %node.address,
            "node registered"
        );
        let msg = format!("welcome {}, {} GPU(s) known", node.node_id, node.gpus.len());
        self.registry.upsert_node(node).await;

        Ok(Response::new(RegisterResponse {
            accepted: true,
            message: msg,
            heartbeat_interval_s: self.heartbeat_interval_s,
        }))
    }

    async fn heartbeat(
        &self,
        req: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = req.into_inner();
        let known = self.registry.heartbeat(&req.node_id, req.gpus).await;
        for status in req.jobs {
            self.registry.update_job_status(status).await;
        }
        Ok(Response::new(HeartbeatResponse { known }))
    }

    async fn list_nodes(
        &self,
        _req: Request<ListNodesRequest>,
    ) -> Result<Response<ListNodesResponse>, Status> {
        Ok(Response::new(ListNodesResponse {
            nodes: self.registry.node_states().await,
        }))
    }

    async fn list_gpus(
        &self,
        _req: Request<ListGpusRequest>,
    ) -> Result<Response<ListGpusResponse>, Status> {
        Ok(Response::new(ListGpusResponse {
            gpus: self.registry.gpu_entries().await,
        }))
    }

    async fn submit_job(
        &self,
        req: Request<SubmitJobRequest>,
    ) -> Result<Response<SubmitJobResponse>, Status> {
        let req = req.into_inner();
        if req.script.is_empty() {
            return Err(Status::invalid_argument("script must not be empty"));
        }

        let nodes = self.registry.node_states().await;
        let plan = if req.auto_place {
            scheduler::plan_auto(
                &nodes,
                &req.node_filter,
                self.master_port,
                self.min_free_vram_b,
                // gpus_per_node doubles as a cap in auto mode when set.
                if req.gpus_per_node > 0 { req.gpus_per_node } else { u32::MAX },
            )
        } else {
            scheduler::plan(
                &nodes,
                req.nodes,
                req.gpus_per_node,
                &req.node_filter,
                self.master_port,
                self.min_free_vram_b,
            )
        }
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

        // Auto mode decides the shape, so the launch requests must follow the
        // plan rather than what the caller asked for.
        let nnodes = plan.placements.len() as u32;
        let nproc = plan.placements.first().map(|p| p.gpu_indices.len() as u32).unwrap_or(0);

        let job_id = format!("j{}", &uuid::Uuid::new_v4().simple().to_string()[..10]);
        let name = if req.name.is_empty() {
            std::path::Path::new(&req.script)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "job".into())
        } else {
            req.name.clone()
        };

        // Reserve before dispatching, so a second submission racing this one
        // sees the GPUs as taken instead of planning onto the same devices.
        self.registry.reserve(&plan, &job_id).await;

        let (tx, _) = broadcast::channel(4096);
        self.registry
            .insert_job(Job {
                job_id: job_id.clone(),
                name,
                submitted_by: req.submitted_by.clone(),
                timeout_s: req.timeout_s,
                plan: plan.clone(),
                per_node: Default::default(),
                submitted: now_s(),
                logs: VecDeque::new(),
                nccl_errors: Vec::new(),
                metrics: Default::default(),
                util_sum: Default::default(),
                util_n: Default::default(),
                tx,
            })
            .await;

        // Pass the image through verbatim, empty included. An empty value
        // means "whatever this node's agent is configured to use", which is
        // what lets a cluster with mixed GPU architectures work: a Blackwell
        // node needs a CUDA 12.8 image where an Ampere node is happy on 12.6.
        let image = req.image.clone();

        // Launch rank 0 first: it hosts the rendezvous, and starting the other
        // ranks against a master that is not up yet just burns retry timeout.
        let mut placements = plan.placements.clone();
        placements.sort_by_key(|p| p.node_rank);

        for p in &placements {
            let launch = LaunchJobRequest {
                job_id: job_id.clone(),
                node_rank: p.node_rank,
                nnodes,
                nproc_per_node: nproc,
                master_addr: plan.master_addr.clone(),
                master_port: plan.master_port,
                gpu_indices: p.gpu_indices.clone(),
                image: image.clone(),
                workdir: req.workdir.clone(),
                script: req.script.clone(),
                script_args: req.script_args.clone(),
                env: req.env.clone(),
                torchrun_args: Vec::new(),
                mounts: req.mounts.clone(),
            };

            if let Err(e) = dispatch(&p.address, launch).await {
                // Partial launch: tear down whatever already started so the
                // cluster is not left with orphaned ranks holding GPUs.
                tracing::error!(job = %job_id, node = %p.node_id, "launch failed: {e}");
                for done in placements.iter().take_while(|q| q.node_rank < p.node_rank) {
                    let _ = stop_on(&done.address, &job_id).await;
                }
                self.registry
                    .update_job_status(JobStatus {
                        job_id: job_id.clone(),
                        node_id: p.node_id.clone(),
                        node_rank: p.node_rank,
                        phase: JobPhase::Failed as i32,
                        exit_code: -1,
                        message: format!("launch failed: {e}"),
                        started_unix_s: 0,
                        ended_unix_s: now_s(),
                    })
                    .await;
                self.registry.release_if_done(&job_id).await;

                return Ok(Response::new(SubmitJobResponse {
                    job_id,
                    accepted: false,
                    message: format!("launch on {} failed: {e}", p.node_id),
                    plan: Some(plan),
                }));
            }
        }

        tracing::info!(
            job = %job_id,
            world_size = plan.world_size,
            master = %plan.master_addr,
            "job launched"
        );
        Ok(Response::new(SubmitJobResponse {
            job_id,
            accepted: true,
            message: "launched".into(),
            plan: Some(plan),
        }))
    }

    async fn get_job(&self, req: Request<GetJobRequest>) -> Result<Response<JobSummary>, Status> {
        let id = req.into_inner().job_id;
        let g = self.registry.inner.lock().await;
        g.jobs
            .get(&id)
            .map(|j| Response::new(j.to_summary()))
            .ok_or_else(|| Status::not_found(format!("no such job {id}")))
    }

    async fn list_jobs(
        &self,
        req: Request<ListJobsRequest>,
    ) -> Result<Response<ListJobsResponse>, Status> {
        let limit = req.into_inner().limit;
        let g = self.registry.inner.lock().await;
        let mut jobs: Vec<JobSummary> = g
            .job_order
            .iter()
            .rev()
            .filter_map(|id| g.jobs.get(id))
            .map(|j| j.to_summary())
            .collect();
        if limit > 0 {
            jobs.truncate(limit as usize);
        }
        Ok(Response::new(ListJobsResponse { jobs }))
    }

    async fn list_processes(
        &self,
        _req: Request<ListProcessesRequest>,
    ) -> Result<Response<ListProcessesResponse>, Status> {
        let g = self.registry.inner.lock().await;

        // Live GPU telemetry, keyed by (node, index), so each rank can be
        // shown with the utilisation of the cards it actually holds.
        let mut gpu_by_key: std::collections::HashMap<(&str, u32), &Gpu> = Default::default();
        for node in g.nodes.values() {
            for gpu in &node.info.gpus {
                gpu_by_key.insert((node.info.node_id.as_str(), gpu.index), gpu);
            }
        }

        let mut processes = Vec::new();
        for id in g.job_order.iter().rev() {
            let Some(job) = g.jobs.get(id) else { continue };
            if job.phase().is_terminal() {
                continue;
            }
            for p in &job.plan.placements {
                let st = job.per_node.get(&p.node_id);
                let held: Vec<&Gpu> = p
                    .gpu_indices
                    .iter()
                    .filter_map(|i| gpu_by_key.get(&(p.node_id.as_str(), *i)).copied())
                    .collect();
                let n = held.len().max(1) as f64;
                processes.push(ProcessEntry {
                    job_id: job.job_id.clone(),
                    name: job.name.clone(),
                    node_id: p.node_id.clone(),
                    node_rank: p.node_rank,
                    gpu_indices: p.gpu_indices.clone(),
                    phase: st.map(|s| s.phase).unwrap_or(JobPhase::Pending as i32),
                    started_unix_s: st.map(|s| s.started_unix_s).unwrap_or(0),
                    world_size: job.plan.world_size,
                    image: String::new(),
                    user: job.submitted_by.clone(),
                    // The container runs as the agent's own account, which is
                    // not necessarily whoever submitted the job.
                    runs_as: g
                        .nodes
                        .get(&p.node_id)
                        .map(|n| n.info.user.clone())
                        .unwrap_or_default(),
                    metrics: Some(job.to_summary().metrics.unwrap_or_default()),
                    gpu_util_pct: held.iter().map(|g| g.utilization_pct as f64).sum::<f64>() / n,
                    vram_used_gb: held
                        .iter()
                        .map(|g| g.memory_used_b as f64 / (1u64 << 30) as f64)
                        .sum(),
                });
            }
        }
        Ok(Response::new(ListProcessesResponse { processes }))
    }

    async fn benchmark_nodes(
        &self,
        req: Request<BenchmarkNodesRequest>,
    ) -> Result<Response<BenchmarkNodesResponse>, Status> {
        let req = req.into_inner();
        let nodes = self.registry.node_states().await;

        let targets: Vec<(String, String)> = nodes
            .iter()
            .filter(|n| n.healthy)
            .filter_map(|n| n.info.as_ref())
            .filter(|i| req.node_filter.is_empty() || req.node_filter.contains(&i.node_id))
            .map(|i| (i.node_id.clone(), i.address.clone()))
            .collect();

        if targets.is_empty() {
            return Err(Status::failed_precondition("no healthy nodes to benchmark"));
        }

        // Sequential on purpose: benchmarking several nodes at once is fine,
        // but each node's GPUs must be measured without contending with the
        // others, and the agent already serialises within a node.
        let mut results = Vec::new();
        for (node_id, addr) in targets {
            match NodeAgentClient::connect(endpoint(&addr)).await {
                Ok(mut c) => match c.benchmark(BenchmarkRequest { force: req.force }).await {
                    Ok(r) => results.extend(r.into_inner().results),
                    Err(e) => results.push(GpuBenchmark {
                        node_id,
                        error: format!("{}", e.message()),
                        ..Default::default()
                    }),
                },
                Err(e) => results.push(GpuBenchmark {
                    node_id,
                    error: format!("connect: {e}"),
                    ..Default::default()
                }),
            }
        }

        self.registry.record_benchmarks(&results).await;
        Ok(Response::new(BenchmarkNodesResponse { results }))
    }

    async fn list_plugins(
        &self,
        _req: Request<ListPluginsRequest>,
    ) -> Result<Response<ListPluginsResponse>, Status> {
        Ok(Response::new(ListPluginsResponse {
            plugins: self
                .plugins
                .plugins
                .iter()
                .map(|(name, p)| PluginInfo {
                    name: name.clone(),
                    description: p.description.clone(),
                    can_fetch: !p.fetch.is_empty(),
                    can_push: !p.push.is_empty(),
                })
                .collect(),
        }))
    }

    async fn run_plugin(
        &self,
        req: Request<RunPluginRequest>,
    ) -> Result<Response<RunPluginResponse>, Status> {
        let req = req.into_inner();

        let plugin = self
            .plugins
            .get(&req.plugin)
            .map_err(|e| Status::not_found(format!("{e:#}")))?;
        let argv = plugin
            .argv(&req.action, &req.remote, &req.local)
            .map_err(|e| Status::invalid_argument(format!("{e:#}")))?;

        let targets: Vec<(String, String)> = self
            .registry
            .node_states()
            .await
            .iter()
            .filter(|n| n.healthy)
            .filter_map(|n| n.info.as_ref())
            .filter(|i| req.node_filter.is_empty() || req.node_filter.contains(&i.node_id))
            .map(|i| (i.node_id.clone(), i.address.clone()))
            .collect();

        if targets.is_empty() {
            return Err(Status::failed_precondition("no healthy nodes selected"));
        }

        tracing::info!(
            plugin = %req.plugin,
            action = %req.action,
            nodes = targets.len(),
            "running plugin: {}", argv.join(" ")
        );

        // Concurrently, because the whole point is that each node fetches its
        // own copy rather than queueing behind the others.
        let mut tasks = Vec::new();
        for (node_id, addr) in targets {
            let argv = argv.clone();
            let plugin_name = req.plugin.clone();
            let workdir = plugin.workdir.clone();
            let timeout_s = req.timeout_s;
            tasks.push(tokio::spawn(async move {
                let started = std::time::Instant::now();
                let outcome = exec_plugin_on(&addr, &plugin_name, argv, workdir, timeout_s).await;
                let seconds = started.elapsed().as_secs_f64();
                match outcome {
                    Ok((code, out, err)) => PluginResult {
                        node_id,
                        exit_code: code,
                        output: out,
                        error: err,
                        seconds,
                    },
                    Err(e) => PluginResult {
                        node_id,
                        exit_code: -1,
                        output: String::new(),
                        error: e,
                        seconds,
                    },
                }
            }));
        }

        let mut results = Vec::new();
        for t in tasks {
            match t.await {
                Ok(r) => results.push(r),
                Err(e) => results.push(PluginResult {
                    node_id: "?".into(),
                    exit_code: -1,
                    error: format!("task failed: {e}"),
                    ..Default::default()
                }),
            }
        }
        results.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        Ok(Response::new(RunPluginResponse { results }))
    }

    async fn cancel_job(
        &self,
        req: Request<CancelJobRequest>,
    ) -> Result<Response<CancelJobResponse>, Status> {
        let id = req.into_inner().job_id;
        let addrs: Vec<String> = {
            let g = self.registry.inner.lock().await;
            let Some(job) = g.jobs.get(&id) else {
                return Err(Status::not_found(format!("no such job {id}")));
            };
            job.plan.placements.iter().map(|p| p.address.clone()).collect()
        };

        let mut errs = Vec::new();
        for a in &addrs {
            if let Err(e) = stop_on(a, &id).await {
                errs.push(format!("{a}: {e}"));
            }
        }
        self.registry.release_if_done(&id).await;

        Ok(Response::new(CancelJobResponse {
            cancelled: errs.is_empty(),
            message: if errs.is_empty() { format!("cancelled {id}") } else { errs.join("; ") },
        }))
    }

    type StreamLogsStream = LogStream;

    async fn stream_logs(
        &self,
        req: Request<LogRequest>,
    ) -> Result<Response<Self::StreamLogsStream>, Status> {
        let req = req.into_inner();
        let (backlog, mut rx) = {
            let g = self.registry.inner.lock().await;
            let Some(job) = g.jobs.get(&req.job_id) else {
                return Err(Status::not_found(format!("no such job {}", req.job_id)));
            };
            // Subscribe before releasing the lock so no line slips through
            // between replaying the backlog and attaching to the live feed.
            (job.logs.iter().cloned().collect::<Vec<_>>(), job.tx.subscribe())
        };

        let (tx, out_rx) = tokio::sync::mpsc::channel(256);
        let follow = req.follow;
        tokio::spawn(async move {
            for line in backlog {
                if tx.send(Ok(line)).await.is_err() {
                    return;
                }
            }
            if !follow {
                return;
            }
            loop {
                match rx.recv().await {
                    Ok(line) => {
                        if tx.send(Ok(line)).await.is_err() {
                            return;
                        }
                    }
                    // A slow client that fell behind keeps following rather
                    // than having its stream torn down.
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(out_rx)) as LogStream))
    }

    async fn report_logs(
        &self,
        req: Request<ReportLogsRequest>,
    ) -> Result<Response<ReportLogsResponse>, Status> {
        self.registry.append_logs(req.into_inner().lines).await;
        Ok(Response::new(ReportLogsResponse {}))
    }

    async fn report_job_status(
        &self,
        req: Request<ReportJobStatusRequest>,
    ) -> Result<Response<ReportJobStatusResponse>, Status> {
        let Some(status) = req.into_inner().status else {
            return Err(Status::invalid_argument("missing status"));
        };
        let job_id = status.job_id.clone();
        let failed_node = status.node_id.clone();
        let rank_failed = status.phase() == JobPhase::Failed;
        self.registry.update_job_status(status).await;

        // A distributed job is dead once any rank dies, but the survivors do
        // not know that: they sit in a collective waiting for a peer that
        // will never answer, holding their GPUs indefinitely. Tear them down
        // so the cards go back to the pool.
        if rank_failed {
            let survivors: Vec<(String, String)> = {
                let g = self.registry.inner.lock().await;
                match g.jobs.get(&job_id) {
                    Some(job) => job
                        .plan
                        .placements
                        .iter()
                        .filter(|p| p.node_id != failed_node)
                        .filter(|p| {
                            job.per_node
                                .get(&p.node_id)
                                .map(|s| !s.phase().is_terminal())
                                .unwrap_or(true)
                        })
                        .map(|p| (p.node_id.clone(), p.address.clone()))
                        .collect(),
                    None => Vec::new(),
                }
            };
            for (node_id, addr) in survivors {
                tracing::info!(
                    job = %job_id,
                    node = %node_id,
                    "rank on {failed_node} failed; stopping surviving rank"
                );
                if let Err(e) = stop_on(&addr, &job_id).await {
                    tracing::warn!(job = %job_id, node = %node_id, "could not stop: {e}");
                }
            }
        }

        self.registry.release_if_done(&job_id).await;
        Ok(Response::new(ReportJobStatusResponse {}))
    }
}

/// Cancel jobs that have outrun their wall-clock limit.
pub async fn reap_expired(registry: std::sync::Arc<Registry>) {
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(15));
    loop {
        tick.tick().await;
        for (job_id, addrs) in registry.expired_jobs().await {
            tracing::warn!(job = %job_id, "wall-clock timeout exceeded, cancelling");
            for a in &addrs {
                if let Err(e) = stop_on(a, &job_id).await {
                    tracing::warn!(job = %job_id, "could not stop on {a}: {e}");
                }
            }
            registry
                .update_job_status(JobStatus {
                    job_id: job_id.clone(),
                    phase: JobPhase::Cancelled as i32,
                    message: "cancelled: wall-clock timeout exceeded".into(),
                    ended_unix_s: now_s(),
                    ..Default::default()
                })
                .await;
            registry.release_if_done(&job_id).await;
        }
    }
}

fn endpoint(addr: &str) -> String {
    if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    }
}

async fn dispatch(addr: &str, req: LaunchJobRequest) -> Result<(), String> {
    let mut client = NodeAgentClient::connect(endpoint(addr))
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let resp = client.launch_job(req).await.map_err(|e| format!("{}", e.message()))?;
    let resp = resp.into_inner();
    if resp.launched {
        Ok(())
    } else {
        Err(resp.message)
    }
}

async fn exec_plugin_on(
    addr: &str,
    plugin: &str,
    argv: Vec<String>,
    workdir: String,
    timeout_s: u32,
) -> Result<(i32, String, String), String> {
    let mut client = NodeAgentClient::connect(endpoint(addr))
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let resp = client
        .exec_plugin(ExecPluginRequest {
            plugin: plugin.to_string(),
            argv,
            workdir,
            timeout_s,
        })
        .await
        .map_err(|e| format!("{}", e.message()))?
        .into_inner();
    Ok((resp.exit_code, resp.output, resp.error))
}

async fn stop_on(addr: &str, job_id: &str) -> Result<(), String> {
    let mut client = NodeAgentClient::connect(endpoint(addr))
        .await
        .map_err(|e| format!("connect: {e}"))?;
    client
        .stop_job(StopJobRequest { job_id: job_id.to_string() })
        .await
        .map_err(|e| format!("{}", e.message()))?;
    Ok(())
}
