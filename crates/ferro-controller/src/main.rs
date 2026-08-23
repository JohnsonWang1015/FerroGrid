//! FerroGrid controller: cluster registry, GPU scheduler, job orchestrator.

mod metrics;
mod registry;
mod scheduler;
mod service;

use anyhow::Result;
use clap::Parser;
use ferro_proto::controller_server::ControllerServer;
use registry::Registry;
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "ferro-controller", version, about = "FerroGrid controller")]
struct Args {
    /// gRPC listen address.
    #[arg(long, env = "FERRO_CONTROLLER_BIND", default_value = "0.0.0.0:7070")]
    bind: SocketAddr,

    /// Port used for the torch.distributed rendezvous on the rank-0 node.
    #[arg(long, env = "FERRO_MASTER_PORT", default_value_t = 29500)]
    master_port: u32,

    /// Heartbeat interval handed to agents at registration.
    #[arg(long, default_value_t = 3)]
    heartbeat_secs: u32,

    /// A GPU must have at least this much free VRAM (GiB) to be scheduled.
    /// Guards against GPUs busy with workloads FerroGrid does not manage.
    #[arg(long, default_value_t = 8)]
    min_free_vram_gib: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let registry = Arc::new(Registry::new());

    let svc = service::ControllerService {
        registry: registry.clone(),
        master_port: args.master_port,
        heartbeat_interval_s: args.heartbeat_secs,
        min_free_vram_b: args.min_free_vram_gib << 30,
    };

    tracing::info!(
        bind = %args.bind,
        master_port = args.master_port,
        min_free_vram_gib = args.min_free_vram_gib,
        "ferro-controller listening"
    );

    tonic::transport::Server::builder()
        .add_service(ControllerServer::new(svc))
        .serve_with_shutdown(args.bind, async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
