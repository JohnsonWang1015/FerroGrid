//! `ferro` — operator CLI for the FerroGrid control plane.

mod render;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ferro_proto::controller_client::ControllerClient;
use ferro_proto::*;
use std::collections::HashMap;
use tonic::transport::Channel;

#[derive(Parser, Debug)]
#[command(name = "ferro", version, about = "FerroGrid multi-server GPU training CLI")]
struct Cli {
    /// Controller endpoint.
    #[arg(long, global = true, env = "FERRO_CONTROLLER", default_value = "http://127.0.0.1:7070")]
    controller: String,

    /// Emit JSON instead of tables.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

/// Flags shared by the read-only views, so any of them can be watched live.
#[derive(clap::Args, Debug, Clone, Copy)]
struct WatchArgs {
    /// Redraw continuously, like `watch nvidia-smi`. Ctrl-C to stop.
    #[arg(short = 'w', long)]
    watch: bool,

    /// Seconds between redraws.
    #[arg(short = 'n', long, default_value_t = 2.0, value_name = "SECONDS")]
    interval: f64,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// List registered servers.
    Nodes {
        #[command(flatten)]
        watch: WatchArgs,
    },
    /// List every GPU across the cluster.
    Gpu {
        #[command(flatten)]
        watch: WatchArgs,
    },
    /// Copy the current project to every node's workspace over rsync/SSH.
    Sync(SyncArgs),
    /// Show what is running right now, per rank, like `docker ps`.
    Ps {
        #[command(flatten)]
        watch: WatchArgs,
    },
    /// Measure each GPU's bf16 throughput so the scheduler can rank hardware.
    Bench {
        /// Only these node ids, repeatable. Default: every healthy node.
        #[arg(long = "node")]
        node_filter: Vec<String>,
        /// Re-measure even where a cached result exists.
        #[arg(long)]
        force: bool,
    },
    /// Live dashboard: nodes, GPUs and running jobs on one screen.
    Watch {
        /// Seconds between redraws.
        #[arg(short = 'n', long, default_value_t = 2.0, value_name = "SECONDS")]
        interval: f64,
    },
    /// Launch a distributed training job.
    Train(TrainArgs),
    /// List recent jobs.
    Jobs {
        #[arg(long, default_value_t = 20)]
        limit: u32,
        #[command(flatten)]
        watch: WatchArgs,
    },
    /// Show one job in detail.
    Job {
        job_id: String,
        #[command(flatten)]
        watch: WatchArgs,
    },
    /// Stream a job's logs.
    Logs {
        job_id: String,
        /// Keep following until the job ends.
        #[arg(short, long)]
        follow: bool,
    },
    /// Cancel a running job.
    Cancel { job_id: String },
}

#[derive(clap::Args, Debug, Clone)]
struct SyncArgs {
    /// Directory to copy. Defaults to the current directory.
    #[arg(default_value = ".")]
    path: String,

    /// Only sync these node ids, repeatable. Default: every healthy node.
    #[arg(long = "node")]
    node_filter: Vec<String>,

    /// Delete files on the node that no longer exist locally.
    #[arg(long)]
    delete: bool,

    /// Print the rsync commands without running them.
    #[arg(long)]
    dry_run: bool,
}

#[derive(clap::Args, Debug)]
struct TrainArgs {
    /// Python entrypoint, as seen inside the container.
    script: String,

    /// Arguments forwarded to the script. Everything after the script path is
    /// passed through verbatim, so `ferro` flags must come before it.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    script_args: Vec<String>,

    /// Number of servers to use.
    #[arg(long, default_value_t = 1)]
    nodes: u32,

    /// GPUs per server (becomes torchrun --nproc_per_node). In --auto mode
    /// this caps how many GPUs may be used rather than requesting exactly N.
    #[arg(long, default_value_t = 1)]
    gpus_per_node: u32,

    /// Let the scheduler choose the shape. It keeps the job on one node and
    /// takes the largest set of identical GPUs there, preferring whichever
    /// node benchmarks fastest -- because crossing the network or sharding
    /// unnecessarily both cost far more than they gain here.
    #[arg(long)]
    auto: bool,

    /// Docker image override.
    #[arg(long)]
    image: Option<String>,

    /// Directory bind-mounted into the container and used as the working
    /// directory. Relative to the agent's workspace root; defaults to it.
    #[arg(long)]
    workdir: Option<String>,

    /// Extra environment, repeatable: --env NCCL_DEBUG=INFO
    #[arg(long = "env", value_parser = parse_kv)]
    envs: Vec<(String, String)>,

    /// Restrict placement to these node ids, repeatable.
    #[arg(long = "node")]
    node_filter: Vec<String>,

    /// Extra bind mount, repeatable: --mount /mnt/adni_data
    /// Accepts HOST, HOST:CONTAINER or HOST:CONTAINER:ro. A bare path is
    /// mounted at the same path inside the container.
    #[arg(long = "mount")]
    mounts: Vec<String>,

    /// Job name shown in `ferro jobs`.
    #[arg(long)]
    name: Option<String>,

    /// Stream logs after submitting.
    #[arg(short, long)]
    follow: bool,

    /// rsync the current directory to every target node before launching.
    #[arg(long)]
    sync: bool,
}

fn parse_kv(s: &str) -> Result<(String, String), String> {
    s.split_once('=')
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .ok_or_else(|| format!("expected KEY=VALUE, got `{s}`"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut client = ControllerClient::connect(cli.controller.clone())
        .await
        .with_context(|| format!("cannot reach controller at {}", cli.controller))?;

    match cli.cmd {
        Cmd::Nodes { watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_nodes(ListNodesRequest {}).await?.into_inner();
                render::nodes(&r.nodes, cli.json);
                Ok(())
            })
            .await?;
        }
        Cmd::Gpu { watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_gpus(ListGpusRequest {}).await?.into_inner();
                render::gpus(&r.gpus, cli.json);
                Ok(())
            })
            .await?;
        }
        Cmd::Watch { interval } => {
            let w = WatchArgs { watch: true, interval };
            repeat(w, false, || async {
                let mut c = client.clone();
                let nodes = c.list_nodes(ListNodesRequest {}).await?.into_inner();
                let gpus = c.list_gpus(ListGpusRequest {}).await?.into_inner();
                let jobs = c.list_jobs(ListJobsRequest { limit: 8 }).await?.into_inner();
                render::dashboard(&nodes.nodes, &gpus.gpus, &jobs.jobs);
                Ok(())
            })
            .await?;
        }
        Cmd::Ps { watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_processes(ListProcessesRequest {}).await?.into_inner();
                render::processes(&r.processes, cli.json);
                Ok(())
            })
            .await?;
        }
        Cmd::Bench { node_filter, force } => {
            eprintln!("benchmarking (a few seconds per GPU)...");
            let r = client
                .benchmark_nodes(BenchmarkNodesRequest { node_filter, force })
                .await?
                .into_inner();
            render::benchmarks(&r.results, cli.json);
        }
        Cmd::Jobs { limit, watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_jobs(ListJobsRequest { limit }).await?.into_inner();
                render::jobs(&r.jobs, cli.json);
                Ok(())
            })
            .await?;
        }
        Cmd::Job { job_id, watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c
                    .get_job(GetJobRequest { job_id: job_id.clone() })
                    .await?
                    .into_inner();
                render::job_detail(&r, cli.json);
                Ok(())
            })
            .await?;
        }
        Cmd::Logs { job_id, follow } => {
            stream_logs(&mut client, &job_id, follow).await?;
        }
        Cmd::Cancel { job_id } => {
            let r = client
                .cancel_job(CancelJobRequest { job_id })
                .await?
                .into_inner();
            println!("{}", r.message);
        }
        Cmd::Sync(args) => {
            let nodes = client.list_nodes(ListNodesRequest {}).await?.into_inner().nodes;
            sync_project(&nodes, &args)?;
        }
        Cmd::Train(args) => {
            if args.sync {
                let nodes = client.list_nodes(ListNodesRequest {}).await?.into_inner().nodes;
                sync_project(
                    &nodes,
                    &SyncArgs {
                        path: ".".into(),
                        node_filter: args.node_filter.clone(),
                        delete: false,
                        dry_run: false,
                    },
                )?;
            }
            train(&mut client, args, cli.json).await?;
        }
    }
    Ok(())
}

/// Everything that should never be shipped to a training node: build output,
/// virtualenvs, caches, and the dataset if it happens to live in the project.
const SYNC_EXCLUDES: &[&str] = &[
    ".git", "target", ".venv", "venv", "__pycache__", "*.pyc", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", ".cargo-container-registry",
    "*.nii", "*.nii.gz", "*.dcm", "*.pt", "*.pth", "*.ckpt", "*.safetensors",
];

/// rsync the project to each node's workspace.
///
/// The nodes tell us their own user and workspace root in their registration,
/// so this needs no host list and copes with the nodes having different home
/// directories -- which they generally do.
fn sync_project(nodes: &[NodeState], args: &SyncArgs) -> Result<()> {
    let src = std::path::Path::new(&args.path)
        .canonicalize()
        .with_context(|| format!("cannot resolve {}", args.path))?;

    let targets: Vec<&NodeState> = nodes
        .iter()
        .filter(|n| n.healthy)
        .filter(|n| {
            args.node_filter.is_empty()
                || n.info
                    .as_ref()
                    .map(|i| args.node_filter.contains(&i.node_id))
                    .unwrap_or(false)
        })
        .collect();

    if targets.is_empty() {
        anyhow::bail!("no healthy nodes to sync to");
    }

    for n in targets {
        let Some(i) = n.info.as_ref() else { continue };
        let host = i.address.rsplit_once(':').map(|(h, _)| h).unwrap_or(&i.address);
        if i.workspace.is_empty() || i.user.is_empty() {
            eprintln!(
                "  {} is running an agent too old to report its workspace; \
                 redeploy it or use scripts/sync_workspace.sh",
                i.node_id
            );
            continue;
        }
        // Trailing slash on the source: copy the *contents* into the
        // workspace, not the directory itself.
        let dest = format!("{}@{}:{}/", i.user, host, i.workspace.trim_end_matches('/'));

        let mut cmd = std::process::Command::new("rsync");
        cmd.arg("-az");
        // --mkpath needs rsync >= 3.2.3 and Ubuntu 20.04 ships 3.1.3, so
        // create the workspace through the remote shell instead.
        cmd.arg("--rsync-path")
            .arg(format!("mkdir -p '{}' && rsync", i.workspace.trim_end_matches('/')));
        if args.delete {
            cmd.arg("--delete");
        }
        for e in SYNC_EXCLUDES {
            cmd.arg("--exclude").arg(e);
        }
        cmd.arg(format!("{}/", src.display())).arg(&dest);

        if args.dry_run {
            println!("rsync {:?}", cmd.get_args().collect::<Vec<_>>());
            continue;
        }

        println!("  {} <- {}", dest, src.display());
        let status = cmd
            .status()
            .context("failed to run rsync (is it installed locally and on the node?)")?;
        if !status.success() {
            anyhow::bail!("rsync to {} failed with {status}", i.node_id);
        }
    }
    Ok(())
}

/// Run `f` once, or repeatedly with a cleared screen when watching.
///
/// Note the refresh rate is bounded below by the agents' heartbeat interval
/// (controller `--heartbeat-secs`, default 3): asking for -n 1 redraws every
/// second but the underlying GPU counters only move every heartbeat, which is
/// why the header shows how stale the data is.
async fn repeat<F, Fut>(w: WatchArgs, json: bool, mut f: F) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    if !w.watch {
        return f().await;
    }
    if json {
        anyhow::bail!("--watch and --json cannot be combined");
    }
    let period = std::time::Duration::from_secs_f64(w.interval.max(0.1));
    loop {
        // Clear screen + home the cursor, the same trick `watch` uses.
        print!("\x1b[2J\x1b[H");
        println!(
            "FerroGrid  {}   every {:.1}s   (Ctrl-C to exit)\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            w.interval
        );
        f().await?;
        tokio::time::sleep(period).await;
    }
}

async fn train(client: &mut ControllerClient<Channel>, args: TrainArgs, json: bool) -> Result<()> {
    // A relative script path is resolved by each agent against its own
    // workspace root (`--workspace`, default ~/ferrogrid), because lab nodes
    // rarely share a home directory. An absolute path is passed through for
    // shared-NFS setups.
    let env: HashMap<String, String> = args.envs.into_iter().collect();
    let workdir = args.workdir.clone().unwrap_or_default();
    let req = SubmitJobRequest {
        script: args.script.clone(),
        script_args: args.script_args,
        nodes: args.nodes,
        // In auto mode this is a cap; 1 is the flag default, which would cap
        // every auto job to a single GPU, so treat "not set" as unlimited.
        gpus_per_node: if args.auto && args.gpus_per_node == 1 { 0 } else { args.gpus_per_node },
        auto_place: args.auto,
        image: args.image.unwrap_or_default(),
        workdir,
        env,
        name: args.name.unwrap_or_default(),
        node_filter: args.node_filter,
        mounts: args.mounts,
        // Whose job this is, for `ferro ps` on a shared cluster.
        submitted_by: std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .unwrap_or_default(),
    };

    let resp = client.submit_job(req).await?.into_inner();
    render::submit(&resp, json);

    if !resp.accepted {
        std::process::exit(1);
    }
    if args.follow {
        stream_logs(client, &resp.job_id, true).await?;
        let final_job = client
            .get_job(GetJobRequest { job_id: resp.job_id.clone() })
            .await?
            .into_inner();
        println!();
        render::job_detail(&final_job, json);
        if final_job.phase() != JobPhase::Succeeded {
            std::process::exit(1);
        }
    }
    Ok(())
}

async fn stream_logs(
    client: &mut ControllerClient<Channel>,
    job_id: &str,
    follow: bool,
) -> Result<()> {
    let mut stream = client
        .stream_logs(LogRequest { job_id: job_id.to_string(), follow })
        .await?
        .into_inner();

    // When following, stop once every rank has reported a terminal phase;
    // otherwise the stream would hang open after training finishes.
    let mut poll = tokio::time::interval(std::time::Duration::from_secs(2));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut done_since: Option<std::time::Instant> = None;
    let mut poller = client.clone();

    loop {
        tokio::select! {
            msg = stream.message() => {
                match msg? {
                    Some(l) => render::log_line(&l),
                    None => break,
                }
            }
            _ = poll.tick(), if follow => {
                let s = poller
                    .get_job(GetJobRequest { job_id: job_id.to_string() })
                    .await?
                    .into_inner();
                if s.phase().is_terminal() {
                    // Give the agents a moment to flush their last log batch.
                    match done_since {
                        Some(t) if t.elapsed() > std::time::Duration::from_secs(3) => break,
                        Some(_) => {}
                        None => done_since = Some(std::time::Instant::now()),
                    }
                } else {
                    done_since = None;
                }
            }
        }
    }
    Ok(())
}
