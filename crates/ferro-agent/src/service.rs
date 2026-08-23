//! `NodeAgent` gRPC service: the controller's entry point into this node.

use crate::launcher;
use crate::state::SharedState;
use ferro_proto::node_agent_server::NodeAgent;
use ferro_proto::{
    BenchmarkRequest, BenchmarkResponse, GetNodeInfoRequest, LaunchJobRequest, LaunchJobResponse,
    NodeInfo, PingRequest, PingResponse, StopJobRequest, StopJobResponse,
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

    async fn ping(&self, _req: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(PingResponse {
            agent_version: crate::AGENT_VERSION.to_string(),
        }))
    }
}
