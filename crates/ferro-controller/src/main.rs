//! FerroGrid controller: cluster registry, GPU scheduler, job orchestrator.

use anyhow::Result;
use clap::Parser;
use ferro_controller::registry::Registry;
use ferro_controller::{plugins, service};
use ferro_proto::controller_server::ControllerServer;
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

    /// Plugin definitions. Defaults to ~/.config/ferrogrid/plugins.toml, then
    /// ./plugins.toml.
    #[arg(long, env = "FERRO_PLUGINS")]
    plugins: Option<std::path::PathBuf>,

    /// A GPU must have at least this much free VRAM (GiB) to be scheduled.
    /// Guards against GPUs busy with workloads FerroGrid does not manage.
    #[arg(long, default_value_t = 8)]
    min_free_vram_gib: u64,

    /// Which job runs next, out of those waiting for capacity.
    #[arg(long, default_value = "fifo", value_name = "POLICY")]
    queue_policy: String,

    /// Where a job runs, once it has been chosen to run.
    #[arg(long, default_value = "performance", value_name = "POLICY")]
    placement_policy: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let min_free_vram_b = args.min_free_vram_gib << 30;
    let queue_policy = ferro_sched::queue_policy(&args.queue_policy)?;
    let registry = Arc::new(Registry::with_queue_policy(min_free_vram_b, queue_policy));
    let plugins = plugins::Registry::load(args.plugins.as_deref())?;
    match &plugins.source {
        Some(p) => tracing::info!(
            "loaded {} plugin(s) from {}: {}",
            plugins.plugins.len(),
            p.display(),
            plugins
                .plugins
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => tracing::info!("no plugin config found; `ferro fetch`/`push` unavailable"),
    }

    let placement = ferro_sched::placement_policy(&args.placement_policy)?;
    let sched = ferro_sched::SchedulerConfig {
        master_port: args.master_port,
        min_free_vram_b,
    };

    let svc = service::ControllerService {
        registry: registry.clone(),
        plugins,
        heartbeat_interval_s: args.heartbeat_secs,
        sched,
        placement: placement.clone(),
    };

    tracing::info!(
        bind = %args.bind,
        master_port = args.master_port,
        min_free_vram_gib = args.min_free_vram_gib,
        queue_policy = %args.queue_policy,
        placement_policy = %args.placement_policy,
        "ferro-controller listening"
    );

    tokio::spawn(service::reap_expired(registry.clone()));
    // Jobs submitted with `--wait` sit here until the cluster frees up.
    tokio::spawn(service::run_queue(registry.clone(), placement, sched));

    tonic::transport::Server::builder()
        .add_service(ControllerServer::new(svc))
        .serve_with_shutdown(args.bind, async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}
