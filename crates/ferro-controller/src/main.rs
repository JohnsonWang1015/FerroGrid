//! FerroGrid controller: cluster registry, GPU scheduler, job orchestrator.

use anyhow::Result;
use clap::Parser;
use ferro_controller::quota::{QuotaTable, UserQuotaSpec};
use ferro_controller::registry::Registry;
use ferro_controller::store::{Event, Store};
use ferro_controller::{plugins, service};
use ferro_proto::controller_server::ControllerServer;
use std::net::SocketAddr;
use std::path::PathBuf;
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

    /// Concurrent GPU limit for a submitted_by identity. Repeatable, e.g.
    /// --user-quota alice=2 --user-quota bob=4. Missing users are unlimited.
    #[arg(long = "user-quota", value_name = "USER=N")]
    user_quotas: Vec<UserQuotaSpec>,

    /// Where jobs, queue order, GPU benchmarks and `ferro net` measurements
    /// are kept across restarts. Defaults to
    /// $XDG_STATE_HOME/ferrogrid/controller.db, i.e.
    /// ~/.local/state/ferrogrid/controller.db. Parent directories are created.
    #[arg(long, env = "FERRO_STATE", value_name = "PATH")]
    state: Option<PathBuf>,

    /// Keep everything in memory, as before persistence existed: nothing is
    /// written and a restart starts empty.
    #[arg(long, conflicts_with = "state")]
    no_state: bool,

    /// How long after startup to keep waiting for an agent to account for each
    /// restored job before writing it off as lost. Must comfortably exceed the
    /// 15 s a node may go unheard before it counts as unhealthy: an agent that
    /// is merely slow to reconnect -- 3 s heartbeats behind a 3 s reconnect
    /// backoff, plus whatever the network was doing -- must not lose its job
    /// for it.
    #[arg(long, default_value_t = 30, value_name = "SECONDS")]
    reconcile_window_secs: u64,

    /// Which job runs next, out of those waiting for capacity.
    #[arg(long, default_value = "fifo", value_name = "POLICY")]
    queue_policy: String,

    /// Who, out of the ranked queue, may start on a given tick.
    ///
    /// `opportunistic` starts anything that fits, which is what FerroGrid has
    /// always done and measured 4.9x better than `strict` on mean waiting
    /// time. `strict` serves the queue in order and stops at the first job
    /// that does not fit. `reserved` backfills as `opportunistic` does, but
    /// only past a reservation the backfilled job can prove it will not
    /// delay -- which it can only do when the jobs involved declared
    /// `--estimated-duration` or `--timeout`. On a cluster where they do not,
    /// `reserved` degrades to `strict`, which is why it is not the default.
    #[arg(long, default_value = "opportunistic", value_name = "MODE")]
    dispatch: String,

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
    let quotas = Arc::new(QuotaTable::from_specs(args.user_quotas)?);
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
    let dispatch = ferro_sched::dispatch_mode(&args.dispatch)?;
    let mut restored_jobs = 0;
    let registry = Arc::new(if args.no_state {
        tracing::warn!("--no-state: jobs and measurements will not survive a restart");
        Registry::with_queue_policy(min_free_vram_b, queue_policy).with_quotas(quotas.clone())
    } else {
        let path = args.state.clone().unwrap_or_else(default_state_path);
        let state = Store::load(&path)?;
        let store = Store::open(&path)?;
        // Restored as written down, not as verified: a job that was running
        // when this process died comes back running. What is still true is
        // settled afterwards, by the agents, inside the reconcile window.
        tracing::info!(
            state = %path.display(),
            "restored {} job(s), {} GPU benchmark(s), {} measured link(s)",
            state.jobs.len(),
            state.bench.len(),
            state.network.len(),
        );
        restored_jobs = state.jobs.len();
        Registry::restore(min_free_vram_b, queue_policy, store, state).with_quotas(quotas.clone())
    });

    registry
        .record_event(Event::controller_start(restored_jobs))
        .await;
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
        dispatch = dispatch.label(),
        user_quotas = quotas.users().count(),
        "ferro-controller listening"
    );

    tokio::spawn(service::reap_expired(registry.clone()));
    // Restored jobs come back exactly as they were written down. This is what
    // decides, once the agents have had their say, which of them are still
    // true.
    tokio::spawn(service::reconcile_recovery(
        registry.clone(),
        args.reconcile_window_secs,
    ));
    // Nothing else notices a node going quiet: heartbeats simply stop.
    tokio::spawn(service::watch_node_health(registry.clone()));
    // Jobs submitted with `--wait` sit here until the cluster frees up.
    tokio::spawn(service::run_queue(
        registry.clone(),
        placement,
        sched,
        dispatch,
    ));

    tonic::transport::Server::builder()
        .add_service(ControllerServer::new(svc))
        .serve_with_shutdown(args.bind, async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}

/// `~/.local/state/ferrogrid/controller.db`, spelled the way the XDG base
/// directory spec does so a machine that has moved its state directory is
/// obeyed rather than second-guessed.
fn default_state_path() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ferrogrid/controller.db")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_quota_flag_is_repeatable() {
        let args = Args::try_parse_from([
            "ferro-controller",
            "--user-quota",
            "alice=2",
            "--user-quota",
            "bob=4",
        ])
        .unwrap();
        let quotas = QuotaTable::from_specs(args.user_quotas).unwrap();
        assert_eq!(quotas.quota_for("alice"), Some(2));
        assert_eq!(quotas.quota_for("bob"), Some(4));
    }

    #[test]
    fn malformed_and_duplicate_user_quota_flags_are_rejected() {
        assert!(Args::try_parse_from(["ferro-controller", "--user-quota", "alice=-1"]).is_err());

        let args = Args::try_parse_from([
            "ferro-controller",
            "--user-quota",
            "alice=2",
            "--user-quota",
            "alice=4",
        ])
        .unwrap();
        assert!(QuotaTable::from_specs(args.user_quotas).is_err());
    }
}
