//! `NodeAgent` gRPC service: the controller's entry point into this node.

use crate::launcher;
use crate::state::SharedState;
use ferro_proto::node_agent_server::NodeAgent;
use ferro_proto::{
    BenchmarkRequest, BenchmarkResponse, DescribeProcessRequest, ExecPluginRequest,
    ExecPluginResponse, GetNodeInfoRequest, LaunchJobRequest, LaunchJobResponse, NetProbeRequest,
    NetProbeResponse, NetSinkRequest, NetSinkResponse, NodeInfo, PingRequest, PingResponse,
    ProcessDetail, StopJobRequest, StopJobResponse,
};
use tonic::{Request, Response, Status};

pub struct AgentService {
    state: SharedState,
}

impl AgentService {
    pub fn new(state: SharedState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl NodeAgent for AgentService {
    async fn get_node_info(
        &self,
        _req: Request<GetNodeInfoRequest>,
    ) -> Result<Response<NodeInfo>, Status> {
        Ok(Response::new(self.state.node_info().await))
    }

    async fn describe_process(
        &self,
        req: Request<DescribeProcessRequest>,
    ) -> Result<Response<ProcessDetail>, Status> {
        let pid = req.into_inner().pid;
        Ok(Response::new(crate::procs::describe(&self.state, pid).await))
    }

    async fn launch_job(
        &self,
        req: Request<LaunchJobRequest>,
    ) -> Result<Response<LaunchJobResponse>, Status> {
        let req = req.into_inner();

        if req.gpu_indices.len() as u32 != req.nproc_per_node {
            return Err(Status::invalid_argument(format!(
                "nproc_per_node={} but {} GPU(s) assigned",
                req.nproc_per_node,
                req.gpu_indices.len()
            )));
        }

        // Re-validate the controller's placement locally. The controller's view
        // is up to one heartbeat stale, so this is the authoritative check.
        let busy = self.state.busy_gpus().await;
        if let Some(conflict) = req.gpu_indices.iter().find(|i| busy.contains(i)) {
            return Err(Status::failed_precondition(format!(
                "GPU {conflict} on {} is already allocated",
                self.state.node_id
            )));
        }
        if self.state.jobs.lock().await.contains_key(&req.job_id) {
            return Err(Status::already_exists(format!("job {} exists", req.job_id)));
        }

        let job_id = req.job_id.clone();
        match launcher::launch(self.state.clone(), req).await {
            Ok(()) => Ok(Response::new(LaunchJobResponse {
                launched: true,
                message: format!("launched {job_id} on {}", self.state.node_id),
            })),
            Err(e) => Err(Status::internal(format!("launch failed: {e:#}"))),
        }
    }

    async fn stop_job(
        &self,
        req: Request<StopJobRequest>,
    ) -> Result<Response<StopJobResponse>, Status> {
        let (stopped, message) = self.state.stop_job(&req.into_inner().job_id).await;
        Ok(Response::new(StopJobResponse { stopped, message }))
    }

    async fn benchmark(
        &self,
        req: Request<BenchmarkRequest>,
    ) -> Result<Response<BenchmarkResponse>, Status> {
        let force = req.into_inner().force;
        match crate::bench::run(self.state.clone(), force).await {
            Ok(results) => Ok(Response::new(BenchmarkResponse { results })),
            Err(e) => Err(Status::internal(format!("benchmark failed: {e:#}"))),
        }
    }

    async fn net_sink(
        &self,
        req: Request<NetSinkRequest>,
    ) -> Result<Response<NetSinkResponse>, Status> {
        match crate::net::sink(req.into_inner().seconds).await {
            Ok(port) => Ok(Response::new(NetSinkResponse { port: port as u32 })),
            Err(e) => Err(Status::internal(format!("open sink failed: {e:#}"))),
        }
    }

    async fn net_probe(
        &self,
        req: Request<NetProbeRequest>,
    ) -> Result<Response<NetProbeResponse>, Status> {
        let req = req.into_inner();
        match crate::net::probe(&req.host, req.port as u16, req.seconds).await {
            Ok((bytes_sent, seconds)) => Ok(Response::new(NetProbeResponse {
                bytes_sent,
                seconds,
                mbps: crate::net::mbps(bytes_sent, seconds),
            })),
            Err(e) => Err(Status::internal(format!("probe failed: {e:#}"))),
        }
    }

    async fn exec_plugin(
        &self,
        req: Request<ExecPluginRequest>,
    ) -> Result<Response<ExecPluginResponse>, Status> {
        let req = req.into_inner();
        let Some((program, rest)) = req.argv.split_first() else {
            return Err(Status::invalid_argument("empty argv"));
        };

        // No shell: argv is passed through as-is, so a path containing spaces
        // or `;` is a path and not a command.
        let mut cmd = tokio::process::Command::new(program);
        cmd.args(rest);

        // The plugin reads its own credentials from this directory. FerroGrid
        // never handles them, which is why they are not in the request.
        if !req.workdir.is_empty() {
            let dir = expand_home(&req.workdir);
            if !std::path::Path::new(&dir).is_dir() {
                return Err(Status::failed_precondition(format!(
                    "plugin `{}` wants workdir {dir}, which does not exist on {}",
                    req.plugin, self.state.node_id
                )));
            }
            cmd.current_dir(dir);
        }

        tracing::info!(plugin = %req.plugin, "exec: {}", req.argv.join(" "));

        let timeout = std::time::Duration::from_secs(if req.timeout_s == 0 {
            3600
        } else {
            req.timeout_s as u64
        });

        let output = match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return Err(Status::internal(format!(
                    "could not run `{program}` on {}: {e}. Is the plugin installed there?",
                    self.state.node_id
                )))
            }
            Err(_) => {
                return Ok(Response::new(ExecPluginResponse {
                    exit_code: -1,
                    output: String::new(),
                    error: format!("timed out after {}s", timeout.as_secs()),
                }))
            }
        };

        Ok(Response::new(ExecPluginResponse {
            exit_code: output.status.code().unwrap_or(-1),
            output: tail(&String::from_utf8_lossy(&output.stdout)),
            error: tail(&String::from_utf8_lossy(&output.stderr)),
        }))
    }

    async fn ping(&self, _req: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(PingResponse {
            agent_version: crate::AGENT_VERSION.to_string(),
        }))
    }
}

fn expand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => path.to_string(),
        },
        None => path.to_string(),
    }
}

/// Keep the end of a transfer tool's output: progress bars can run to
/// megabytes, and the part that says what went wrong is at the bottom.
fn tail(s: &str) -> String {
    const MAX: usize = 8 * 1024;
    let s = s.trim_end();
    if s.len() <= MAX {
        return s.to_string();
    }
    let cut = s.len() - MAX;
    let start = s
        .char_indices()
        .find(|(i, _)| *i >= cut)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    format!("...(truncated)\n{}", &s[start..])
}
