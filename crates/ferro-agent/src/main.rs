//! FerroGrid node agent.
//!
//! One process per GPU server. It:
//!   1. enumerates local GPUs via NVML and registers with the controller,
//!   2. heartbeats live GPU telemetry on a fixed interval,
//!   3. serves the `NodeAgent` gRPC service so the controller can launch and
//!      stop torchrun jobs inside Docker.

mod bench;
mod launcher;
mod procs;
mod service;
mod state;

use anyhow::{Context, Result};
use clap::Parser;
use ferro_proto::controller_client::ControllerClient;
use ferro_proto::node_agent_server::NodeAgentServer;
use ferro_proto::{HeartbeatRequest, RegisterRequest};
use state::AgentState;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub const AGENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug, Clone)]
#[command(name = "ferro-agent", version, about = "FerroGrid node agent")]
pub struct Args {
    /// Controller gRPC endpoint, e.g. http://10.0.0.1:7070
    #[arg(long, env = "FERRO_CONTROLLER")]
    controller: String,

    /// Address this agent's gRPC server binds to.
    #[arg(long, env = "FERRO_AGENT_BIND", default_value = "0.0.0.0:7071")]
    bind: SocketAddr,

    /// IP:port the controller should dial back on. Defaults to <advertise-ip>:<bind port>.
    #[arg(long, env = "FERRO_AGENT_ADVERTISE")]
    advertise: Option<String>,

    /// IP other nodes use for NCCL traffic (becomes MASTER_ADDR when this node is rank 0).
    /// Defaults to the advertised IP.
    #[arg(long, env = "FERRO_NCCL_IP")]
    nccl_ip: Option<String>,

    /// Network interface NCCL should use. Defaults to whichever interface
    /// holds the NCCL IP.
    #[arg(long, env = "FERRO_NCCL_IFNAME")]
    nccl_ifname: Option<String>,

    /// Stable node id. Defaults to the hostname.
    #[arg(long, env = "FERRO_NODE_ID")]
    node_id: Option<String>,

    /// Seconds between heartbeats.
    #[arg(long, default_value_t = 3)]
    heartbeat_secs: u64,

    /// Docker image used when the job does not specify one.
    #[arg(long, env = "FERRO_DEFAULT_IMAGE", default_value = "pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")]
    default_image: String,

    /// Root directory for job workspaces on this node. Relative script paths
    /// from the controller resolve against it, so nodes may have different
    /// home directories.
    #[arg(long, env = "FERRO_WORKSPACE")]
    workspace: Option<String>,

    /// Run training directly on the host instead of inside Docker.
    #[arg(long, env = "FERRO_NO_DOCKER", default_value_t = false)]
    no_docker: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let state = Arc::new(AgentState::new(&args)?);

    tracing::info!(
        node_id = %state.node_id,
        gpus = state.monitor.gpu_count(),
        advertise = %state.advertise,
        nccl_ip = %state.nccl_ip,
        "starting ferro-agent"
    );
    if let Some(err) = state.monitor.init_error() {
        tracing::warn!("no GPUs will be advertised: {err}");
    }

    // gRPC server for controller-initiated actions.
    let svc = service::AgentService::new(state.clone());
    let bind = args.bind;
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(NodeAgentServer::new(svc))
            .serve(bind)
            .await
    });

    // Registration + heartbeat loop. Reconnects forever so an agent survives a
    // controller restart without operator intervention.
    let hb = tokio::spawn(heartbeat_loop(state.clone(), args.clone()));

    tokio::select! {
        r = server => { r.context("agent gRPC server task")??; }
        r = hb => { r.context("heartbeat task")?; }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown requested, stopping running jobs");
            state.stop_all().await;
        }
    }
    Ok(())
}

async fn heartbeat_loop(state: Arc<AgentState>, args: Args) {
    let mut interval = Duration::from_secs(args.heartbeat_secs.max(1));

    // Every reconnect re-registers: it is idempotent on the controller and it
    // is exactly what a controller that restarted and lost us needs anyway.
    loop {
        match ControllerClient::connect(args.controller.clone()).await {
            Ok(mut client) => {
                let node = state.node_info().await;
                match client.register_node(RegisterRequest { node: Some(node) }).await {
                    Ok(resp) => {
                        let resp = resp.into_inner();
                        if resp.heartbeat_interval_s > 0 {
                            interval = Duration::from_secs(resp.heartbeat_interval_s as u64);
                        }
                        tracing::info!("registered with controller: {}", resp.message);
                    }
                    Err(e) => {
                        tracing::warn!("register failed: {e}");
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        continue;
                    }
                }

                // Stay in this inner loop while the connection is healthy.
                loop {
                    tokio::time::sleep(interval).await;
                    let req = HeartbeatRequest {
                        node_id: state.node_id.clone(),
                        gpus: state.gpu_snapshot().await,
                        jobs: state.job_statuses().await,
                        processes: state.process_snapshot().await,
                    };
                    match client.heartbeat(req).await {
                        Ok(resp) => {
                            if !resp.into_inner().known {
                                tracing::info!("controller does not know us, re-registering");
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("heartbeat failed, reconnecting: {e}");
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("cannot reach controller {}: {e}", args.controller);
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
    }
}
