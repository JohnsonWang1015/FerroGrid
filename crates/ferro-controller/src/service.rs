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
    pub default_image: String,
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
        let plan = scheduler::plan(
            &nodes,
            req.nodes,
            req.gpus_per_node,
            &req.node_filter,
            self.master_port,
            self.min_free_vram_b,
        )
        .map_err(|e| Status::failed_precondition(e.to_string()))?;

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

        let image = if req.image.is_empty() { self.default_image.clone() } else { req.image.clone() };

        // Launch rank 0 first: it hosts the rendezvous, and starting the other
        // ranks against a master that is not up yet just burns retry timeout.
        let mut placements = plan.placements.clone();
        placements.sort_by_key(|p| p.node_rank);

        for p in &placements {
            let launch = LaunchJobRequest {
                job_id: job_id.clone(),
                node_rank: p.node_rank,
                nnodes: req.nodes,
                nproc_per_node: req.gpus_per_node,
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
        self.registry.update_job_status(status).await;
        self.registry.release_if_done(&job_id).await;
        Ok(Response::new(ReportJobStatusResponse {}))
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
