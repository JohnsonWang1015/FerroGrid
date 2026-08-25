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
        let known = self
            .registry
            .heartbeat(&req.node_id, req.gpus, req.processes)
            .await;
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
        let plan = match plan_for(&nodes, &req, self.master_port, self.min_free_vram_b) {
            Ok(plan) => plan,
            // Nothing fits right now. With `--wait` that is a queue rather
            // than a failure: on a shared cluster "full" is the normal state,
            // and resubmitting by hand at 03:00 is not a scheduling policy.
            Err(e) if req.queue => {
                let job_id = new_job_id();
                let deadline = match req.queue_timeout_s {
                    0 => 0,
                    t => now_s() + t as i64,
                };
                self.registry
                    .insert_job(new_job(&job_id, &req, JobPlan::default(), Some(deadline)))
                    .await;
                let queue_position = self.registry.queue_position(&job_id).await;
                tracing::info!(job = %job_id, "queued at #{queue_position}: {e}");
                return Ok(Response::new(SubmitJobResponse {
                    job_id,
                    accepted: true,
                    message: e.to_string(),
                    plan: None,
                    queue_position,
                }));
            }
            Err(e) => return Err(Status::failed_precondition(e.to_string())),
        };

        let job_id = new_job_id();
        self.registry
            .insert_job(new_job(&job_id, &req, plan.clone(), None))
            .await;

        match start_job(&self.registry, &req, &job_id, &plan).await {
            Ok(()) => {
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
                    queue_position: 0,
                }))
            }
            Err(message) => Ok(Response::new(SubmitJobResponse {
                job_id,
                accepted: false,
                message,
                plan: Some(plan),
                queue_position: 0,
            })),
        }
    }

    async fn get_job(&self, req: Request<GetJobRequest>) -> Result<Response<JobSummary>, Status> {
        let id = req.into_inner().job_id;
        let g = self.registry.inner.lock().await;
        g.jobs
            .get(&id)
            .map(|j| {
                let mut summary = j.to_summary();
                summary.queue_position = g.queue_position(&id);
                Response::new(summary)
            })
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
            .map(|j| {
                let mut summary = j.to_summary();
                summary.queue_position = g.queue_position(&j.job_id);
                summary
            })
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
                    external: false,
                    pid: 0,
                    command: String::new(),
                    container: String::new(),
                    kind: String::new(),
                    // Our own ranks own whole GPUs, so the device figure above
                    // already is this rank's utilisation.
                    proc_util_pct: 0.0,
                    proc_util_known: false,
                    busy_unix_s: 0,
                    node_last_seen_unix_s: g
                        .nodes
                        .get(&p.node_id)
                        .map(|n| n.last_seen)
                        .unwrap_or(0),
                });
            }
        }

        // Everything else the agents found on the cards. A GPU with no rank on
        // it is not necessarily a free GPU: someone's notebook, another
        // scheduler's container or a leftover of ours after a controller
        // restart hold VRAM just the same, and `ferro ps` is where people look
        // before asking why the cluster is full.
        let ours: std::collections::HashSet<&str> = g
            .jobs
            .values()
            .filter(|j| !j.phase().is_terminal())
            .map(|j| j.job_id.as_str())
            .collect();

        let mut nodes: Vec<&crate::registry::Node> = g.nodes.values().collect();
        nodes.sort_by(|a, b| a.info.node_id.cmp(&b.info.node_id));
        for node in nodes {
            // NVML reports a process once per device it holds; one row per pid
            // keeps a 4-GPU trainer from looking like four separate squatters.
            let mut by_pid: std::collections::BTreeMap<u32, Vec<&GpuProcess>> = Default::default();
            for p in &node.info.processes {
                if ours.contains(p.job_id.as_str()) {
                    continue; // already listed above, as the rank that owns it
                }
                by_pid.entry(p.pid).or_default().push(p);
            }

            for (pid, procs) in by_pid {
                let first = procs[0];
                let indices: Vec<u32> = procs.iter().map(|p| p.gpu_index).collect();
                let held: Vec<&Gpu> = indices
                    .iter()
                    .filter_map(|i| gpu_by_key.get(&(node.info.node_id.as_str(), *i)).copied())
                    .collect();
                let n = held.len().max(1) as f64;
                processes.push(ProcessEntry {
                    // Non-empty only for a job of ours the controller has
                    // forgotten -- the stray worth naming.
                    job_id: first.job_id.clone(),
                    name: String::new(),
                    node_id: node.info.node_id.clone(),
                    node_rank: 0,
                    gpu_indices: indices,
                    phase: JobPhase::Unspecified as i32,
                    started_unix_s: first.started_unix_s,
                    world_size: 0,
                    image: String::new(),
                    user: first.user.clone(),
                    runs_as: String::new(),
                    metrics: None,
                    // Device utilisation, as for our own ranks: NVML cannot
                    // attribute SM time to a pid. The VRAM figure, however, is
                    // this process's own.
                    gpu_util_pct: held.iter().map(|g| g.utilization_pct as f64).sum::<f64>() / n,
                    vram_used_gb: procs
                        .iter()
                        .map(|p| p.memory_used_b as f64 / (1u64 << 30) as f64)
                        .sum(),
                    external: true,
                    pid,
                    command: first.command.clone(),
                    container: first.container.clone(),
                    kind: first.kind.clone(),
                    proc_util_pct: first.utilization_pct as f64,
                    proc_util_known: first.utilization_known,
                    busy_unix_s: first.busy_unix_s,
                    node_last_seen_unix_s: node.last_seen,
                });
            }
        }

        Ok(Response::new(ListProcessesResponse { processes }))
    }

    async fn describe_process(
        &self,
        req: Request<DescribeProcessRequest>,
    ) -> Result<Response<DescribeProcessResponse>, Status> {
        let pid = req.into_inner().pid;

        // Ask the node whose last heartbeat mentions the pid; if none does,
        // ask every healthy node. Pids are unique per machine, so "which node
        // is this even on" is half the question -- and the last heartbeat only
        // lists processes that hold a GPU.
        let nodes = self.registry.node_states().await;
        let healthy: Vec<(String, String)> = nodes
            .iter()
            .filter(|n| n.healthy)
            .filter_map(|n| n.info.as_ref())
            .map(|i| (i.node_id.clone(), i.address.clone()))
            .collect();
        let known: Vec<(String, String)> = nodes
            .iter()
            .filter_map(|n| n.info.as_ref())
            .filter(|i| i.processes.iter().any(|p| p.pid == pid))
            .map(|i| (i.node_id.clone(), i.address.clone()))
            .collect();
        let targets = if known.is_empty() { healthy } else { known };

        let mut matches = Vec::new();
        for (node_id, address) in targets {
            match describe_on(&address, pid).await {
                Ok(detail) if detail.found => matches.push(detail),
                Ok(_) => {}
                Err(e) => tracing::debug!(node = %node_id, "describe pid {pid} failed: {e}"),
            }
        }
        Ok(Response::new(DescribeProcessResponse { matches }))
    }

    async fn measure_network(
        &self,
        req: Request<MeasureNetworkRequest>,
    ) -> Result<Response<MeasureNetworkResponse>, Status> {
        let req = req.into_inner();
        let seconds = if req.seconds == 0 { 3 } else { req.seconds };

        let nodes = self.registry.node_states().await;
        let targets: Vec<NodeInfo> = nodes
            .iter()
            .filter(|n| n.healthy)
            .filter_map(|n| n.info.clone())
            .filter(|i| req.node_filter.is_empty() || req.node_filter.contains(&i.node_id))
            .collect();

        if targets.len() < 2 {
            return Err(Status::failed_precondition(
                "need at least two healthy nodes to measure a link between",
            ));
        }

        let mut ordered: Vec<(&NodeInfo, &NodeInfo)> = Vec::new();
        for (i, a) in targets.iter().enumerate() {
            for b in targets.iter().skip(i + 1) {
                ordered.push((a, b));
                if req.both_ways {
                    ordered.push((b, a));
                }
            }
        }

        let mut pairs = Vec::new();
        {
            for (from, to) in ordered {
                // Strictly one pair at a time: two probes at once share the
                // same switch and each would measure the other's traffic.
                let mut pair = NetPair {
                    from_node: from.node_id.clone(),
                    to_node: to.node_id.clone(),
                    // The path is only ever as fast as its slower end.
                    link_mbps: match (from.link_mbps, to.link_mbps) {
                        (0, b) => b,
                        (a, 0) => a,
                        (a, b) => a.min(b),
                    },
                    ..Default::default()
                };
                // Measure the interface NCCL would use, not the management IP:
                // on these boxes they are frequently not the same wire.
                match measure_pair(&from.address, &to.address, &to.nccl_address, seconds).await {
                    Ok(mbps) => pair.mbps = mbps,
                    Err(e) => pair.error = e,
                }
                pairs.push(pair);
            }
        }
        Ok(Response::new(MeasureNetworkResponse { pairs }))
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

        // A queued job has no ranks to stop; taking it out of the queue is the
        // whole cancellation.
        if self
            .registry
            .dequeue(&id, JobPhase::Cancelled, "cancelled while queued")
            .await
        {
            return Ok(Response::new(CancelJobResponse {
                cancelled: true,
                message: format!("cancelled {id} before it started"),
            }));
        }

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
/// One direction, one pair: open a sink on the receiver, then have the sender
/// blast at it for `seconds`.
async fn measure_pair(
    from_addr: &str,
    to_addr: &str,
    to_host: &str,
    seconds: u32,
) -> Result<f64, String> {
    let mut receiver = NodeAgentClient::connect(endpoint(to_addr))
        .await
        .map_err(|e| format!("connect receiver: {e}"))?;
    let port = receiver
        .net_sink(NetSinkRequest { seconds })
        .await
        .map_err(|e| format!("sink: {}", e.message()))?
        .into_inner()
        .port;

    let mut sender = NodeAgentClient::connect(endpoint(from_addr))
        .await
        .map_err(|e| format!("connect sender: {e}"))?;
    let result = sender
        .net_probe(NetProbeRequest {
            host: to_host.to_string(),
            port,
            seconds,
        })
        .await
        .map_err(|e| format!("probe: {}", e.message()))?
        .into_inner();
    Ok(result.mbps)
}

async fn describe_on(addr: &str, pid: u32) -> Result<ProcessDetail, String> {
    let mut client = NodeAgentClient::connect(endpoint(addr))
        .await
        .map_err(|e| format!("connect: {e}"))?;
    Ok(client
        .describe_process(DescribeProcessRequest { pid })
        .await
        .map_err(|e| e.message().to_string())?
        .into_inner())
}

/// Ask the scheduler where this job goes. Auto mode picks the shape itself,
/// which is why the caller cannot precompute it.
fn plan_for(
    nodes: &[NodeState],
    req: &SubmitJobRequest,
    master_port: u32,
    min_free_vram_b: u64,
) -> Result<JobPlan, scheduler::ScheduleError> {
    if req.auto_place {
        scheduler::plan_auto(
            nodes,
            &req.node_filter,
            master_port,
            min_free_vram_b,
            // gpus_per_node doubles as a cap in auto mode when set.
            if req.gpus_per_node > 0 { req.gpus_per_node } else { u32::MAX },
        )
    } else {
        scheduler::plan(
            nodes,
            req.nodes,
            req.gpus_per_node,
            &req.node_filter,
            master_port,
            min_free_vram_b,
        )
    }
}

fn new_job_id() -> String {
    format!("j{}", &uuid::Uuid::new_v4().simple().to_string()[..10])
}

/// `queue_deadline` set means the job is queued: no plan yet, and the request
/// is kept so the dispatcher can place it later.
fn new_job(job_id: &str, req: &SubmitJobRequest, plan: JobPlan, queue_deadline: Option<i64>) -> Job {
    let name = if req.name.is_empty() {
        std::path::Path::new(&req.script)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "job".into())
    } else {
        req.name.clone()
    };
    let (tx, _) = broadcast::channel(4096);
    Job {
        job_id: job_id.to_string(),
        name,
        submitted_by: req.submitted_by.clone(),
        timeout_s: req.timeout_s,
        plan,
        per_node: Default::default(),
        submitted: now_s(),
        logs: VecDeque::new(),
        nccl_errors: Vec::new(),
        metrics: Default::default(),
        util_sum: Default::default(),
        util_n: Default::default(),
        tx,
        queued: queue_deadline.is_some(),
        queue_req: queue_deadline.map(|_| req.clone()),
        queue_deadline: queue_deadline.unwrap_or(0),
    }
}

/// Reserve the GPUs and launch every rank. Shared by `submit_job` and the
/// queue dispatcher, so a job that waited starts exactly like one that did not.
async fn start_job(
    registry: &Registry,
    req: &SubmitJobRequest,
    job_id: &str,
    plan: &JobPlan,
) -> Result<(), String> {
    // Auto mode decides the shape, so the launch requests must follow the
    // plan rather than what the caller asked for.
    let nnodes = plan.placements.len() as u32;
    let nproc = plan.placements.first().map(|p| p.gpu_indices.len() as u32).unwrap_or(0);

    // Reserve before dispatching, so a second submission racing this one
    // sees the GPUs as taken instead of planning onto the same devices.
    registry.reserve(plan, job_id).await;

    // Launch rank 0 first: it hosts the rendezvous, and starting the other
    // ranks against a master that is not up yet just burns retry timeout.
    let mut placements = plan.placements.clone();
    placements.sort_by_key(|p| p.node_rank);

    for p in &placements {
        let launch = LaunchJobRequest {
            job_id: job_id.to_string(),
            node_rank: p.node_rank,
            nnodes,
            nproc_per_node: nproc,
            master_addr: plan.master_addr.clone(),
            master_port: plan.master_port,
            gpu_indices: p.gpu_indices.clone(),
            // Pass the image through verbatim, empty included. An empty value
            // means "whatever this node's agent is configured to use", which
            // is what lets a cluster with mixed GPU architectures work: a
            // Blackwell node needs a CUDA 12.8 image where an Ampere node is
            // happy on 12.6.
            image: req.image.clone(),
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
                let _ = stop_on(&done.address, job_id).await;
            }
            registry
                .update_job_status(JobStatus {
                    job_id: job_id.to_string(),
                    node_id: p.node_id.clone(),
                    node_rank: p.node_rank,
                    phase: JobPhase::Failed as i32,
                    exit_code: -1,
                    message: format!("launch failed: {e}"),
                    started_unix_s: 0,
                    ended_unix_s: now_s(),
                })
                .await;
            registry.release_if_done(job_id).await;
            return Err(format!("launch on {} failed: {e}", p.node_id));
        }
    }
    Ok(())
}

/// Places jobs submitted with `--wait` as the cluster frees up.
///
/// Deliberately FIFO and one pass per tick: fancier policies (backfill,
/// priorities) need a fairness story this cluster has not asked for, and
/// "whoever waited longest goes next" is the one rule nobody argues with.
pub async fn run_queue(
    registry: std::sync::Arc<Registry>,
    master_port: u32,
    min_free_vram_b: u64,
) {
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
    loop {
        tick.tick().await;
        for (job_id, req, deadline) in registry.queued_jobs().await {
            if deadline > 0 && now_s() > deadline {
                tracing::warn!(job = %job_id, "gave up waiting for capacity");
                registry
                    .dequeue(&job_id, JobPhase::Failed, "gave up waiting for capacity")
                    .await;
                continue;
            }

            // Re-read the cluster for every job: the one placed a moment ago
            // took GPUs the next one must not be handed as well.
            let nodes = registry.node_states().await;
            let Ok(plan) = plan_for(&nodes, &req, master_port, min_free_vram_b) else {
                continue;
            };

            registry.promote(&job_id, plan.clone()).await;
            tracing::info!(job = %job_id, world_size = plan.world_size, "capacity freed, launching");
            if let Err(e) = start_job(&registry, &req, &job_id, &plan).await {
                tracing::error!(job = %job_id, "queued job failed to launch: {e}");
            }
        }
    }
}

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
