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

    /// Aging: seconds a job must wait to earn one step of priority.
    #[arg(long, default_value_t = 60, value_name = "SECONDS")]
    aging_interval_secs: u32,

    /// Aging: priority added per completed interval. 0 disables aging, which
    /// makes `--queue-policy aging` behave exactly like `priority`.
    #[arg(long, default_value_t = 1)]
    aging_increment: u32,

    /// Aging: effective priority may not exceed this, however long the wait.
    #[arg(long, default_value_t = ferro_sched::MAX_PRIORITY)]
    aging_ceiling: u32,

    /// Fair share: weight on the submitted priority.
    #[arg(long, default_value_t = 1.0, value_name = "W")]
    fair_priority_weight: f64,

    /// Fair share: weight on how long the job has waited.
    #[arg(long, default_value_t = 1.0, value_name = "W")]
    fair_wait_weight: f64,

    /// Fair share: weight on the submitter's GPU-seconds already consumed.
    #[arg(long, default_value_t = 1.0, value_name = "W")]
    fair_usage_weight: f64,

    /// Where a job runs, once it has been chosen to run.
    #[arg(long, default_value = "performance", value_name = "POLICY")]
    placement_policy: String,

    /// Ignore `ferro net` measurements older than this when placing. 0 keeps
    /// them forever. The default is a day: a link that was 940 Mb/s last week
    /// is not evidence about the link today.
    #[arg(long, default_value_t = 86_400, value_name = "SECONDS")]
    network_max_age_secs: i64,

    /// Weight on measured GPU throughput when scoring a placement.
    #[arg(long, default_value_t = 1.0, value_name = "W")]
    weight_compute: f64,

    /// Weight on free VRAM headroom on the tightest chosen card.
    #[arg(long, default_value_t = 0.5, value_name = "W")]
    weight_vram: f64,

    /// Weight on the chosen GPUs all being the same model.
    #[arg(long, default_value_t = 1.0, value_name = "W")]
    weight_homogeneity: f64,

    /// Weight on the slowest hop between the chosen nodes.
    #[arg(long, default_value_t = 1.0, value_name = "W")]
    weight_network: f64,

    /// Weight on how little of the chosen nodes was already spoken for.
    #[arg(long, default_value_t = 0.25, value_name = "W")]
    weight_load: f64,
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
    let tuning = ferro_sched::QueueTuning {
        aging: ferro_sched::queue::aging::AgingConfig {
            interval_s: args.aging_interval_secs,
            increment: args.aging_increment,
            ceiling: args.aging_ceiling,
        },
        fair_share: ferro_sched::queue::fair_share::FairShareConfig {
            priority_weight: args.fair_priority_weight,
            wait_weight: args.fair_wait_weight,
            usage_weight: args.fair_usage_weight,
        },
    };
    let queue_policy = ferro_sched::queue_policy(&args.queue_policy, &tuning)?;
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
        network_max_age_s: args.network_max_age_secs,
        placement_weights: ferro_sched::PlacementWeights {
            compute: args.weight_compute,
            vram: args.weight_vram,
            homogeneity: args.weight_homogeneity,
            network: args.weight_network,
            load: args.weight_load,
        },
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
