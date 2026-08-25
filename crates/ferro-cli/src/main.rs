//! `ferro` — operator CLI for the FerroGrid control plane.

mod render;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ferro_proto::controller_client::ControllerClient;
use ferro_proto::*;
use std::collections::HashMap;
use std::io::Write;
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
    /// List the transfer plugins the controller has configured.
    Plugins,
    /// Download data onto the nodes with a plugin.
    Fetch(TransferArgs),
    /// Upload data from a node with a plugin.
    Push(TransferArgs),
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
struct TransferArgs {
    /// Plugin name, as shown by `ferro plugins`.
    plugin: String,

    /// Remote path, as the plugin understands it.
    remote: String,

    /// Path on the node.
    local: String,

    /// Only these nodes, repeatable. Default for fetch: every healthy node,
    /// each pulling its own copy in parallel.
    #[arg(long = "node")]
    node_filter: Vec<String>,

    /// Give up after this long: 90s, 30m, 2h.
    #[arg(long, value_parser = parse_duration, default_value = "1h")]
    timeout: u32,
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

    /// Cancel the job if it is still running after this long: 90s, 30m, 2h.
    /// A hung distributed job never fails on its own, so on a shared cluster
    /// this is what stops it holding GPUs indefinitely.
    #[arg(long, value_parser = parse_duration)]
    timeout: Option<u32>,
}

/// Accepts a bare number of seconds, or a value suffixed s/m/h.
fn parse_duration(s: &str) -> Result<u32, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('s') => (&s[..s.len() - 1], 1),
        Some('m') => (&s[..s.len() - 1], 60),
        Some('h') => (&s[..s.len() - 1], 3600),
        _ => (s, 1),
    };
    num.parse::<u32>()
        .map(|n| n * mult)
        .map_err(|_| format!("expected a duration like 90s, 30m or 2h, got `{s}`"))
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
                Ok(render::nodes(&r.nodes, cli.json))
            })
            .await?;
        }
        Cmd::Gpu { watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_gpus(ListGpusRequest {}).await?.into_inner();
                Ok(render::gpus(&r.gpus, cli.json))
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
                Ok(render::dashboard(&nodes.nodes, &gpus.gpus, &jobs.jobs))
            })
            .await?;
        }
        Cmd::Ps { watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_processes(ListProcessesRequest {}).await?.into_inner();
                Ok(render::processes(&r.processes, cli.json))
            })
            .await?;
        }
        Cmd::Plugins => {
            let r = client.list_plugins(ListPluginsRequest {}).await?.into_inner();
            print!("{}", render::plugins(&r.plugins, cli.json));
        }
        // Bind the action before destructuring, so the match arm can move `a`.
        ref c @ (Cmd::Fetch(ref a) | Cmd::Push(ref a)) => {
            let action = if matches!(c, Cmd::Fetch(_)) { "fetch" } else { "push" };
            eprintln!("{action}: {} <-> {} via {}", a.remote, a.local, a.plugin);
            let r = client
                .run_plugin(RunPluginRequest {
                    plugin: a.plugin.clone(),
                    action: action.to_string(),
                    remote: a.remote.clone(),
                    local: a.local.clone(),
                    node_filter: a.node_filter.clone(),
                    timeout_s: a.timeout,
                })
                .await?
                .into_inner();
            print!("{}", render::transfer(&r.results, cli.json));
            if r.results.iter().any(|x| x.exit_code != 0) {
                std::process::exit(1);
            }
        }
        Cmd::Bench { node_filter, force } => {
            eprintln!("benchmarking (a few seconds per GPU)...");
            let r = client
                .benchmark_nodes(BenchmarkNodesRequest { node_filter, force })
                .await?
                .into_inner();
            print!("{}", render::benchmarks(&r.results, cli.json));
        }
        Cmd::Jobs { limit, watch } => {
            repeat(watch, cli.json, || async {
                let mut c = client.clone();
                let r = c.list_jobs(ListJobsRequest { limit }).await?.into_inner();
                Ok(render::jobs(&r.jobs, cli.json))
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
                Ok(render::job_detail(&r, cli.json))
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

/// Run `f` once, or repeatedly as a live view when watching.
///
/// Note the refresh rate is bounded below by the agents' heartbeat interval
/// (controller `--heartbeat-secs`, default 3): asking for -n 1 redraws every
/// second but the underlying GPU counters only move every heartbeat, which is
/// why the header shows how stale the data is.
async fn repeat<F, Fut>(w: WatchArgs, json: bool, mut f: F) -> Result<()>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    if !w.watch {
        print!("{}", f().await?);
        return Ok(());
    }
    if json {
        anyhow::bail!("--watch and --json cannot be combined");
    }

    let period = std::time::Duration::from_secs_f64(w.interval.max(0.1));
    let mut screen = Screen::new();
    loop {
        // Build the whole frame before anything reaches the terminal: the fetch
        // below is most of the interval at -n 1, and a screen cleared before it
        // is a screen that spends most of its life blank.
        let frame = format!(
            "FerroGrid  {}   every {:.1}s   (Ctrl-C to exit)\n\n{}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            w.interval,
            f().await?
        );
        screen.draw(&frame);

        tokio::select! {
            _ = tokio::time::sleep(period) => {}
            // Leave the last frame up; `Screen`'s drop puts the cursor back.
            // Whatever we leave behind, the next shell inherits.
            _ = interrupted() => break,
        }
    }
    Ok(())
}

/// Ctrl-C, or SIGTERM: `timeout 10 ferro watch` is a normal thing to script,
/// and exiting without restoring the cursor leaves the operator without one.
async fn interrupted() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

/// A live view repainted in place.
///
/// Nothing is ever cleared in advance. Each frame is written over the previous
/// one in a single write, erasing the tail of every line it lands on
/// (`\x1b[K`) and whatever a shorter frame left below (`\x1b[J`). Clearing
/// first -- what `watch(1)` does -- is what makes a one-second refresh flicker.
struct Screen {
    /// Terminal size the previous frame was laid out for. A resize rewraps the
    /// tables, so the old and new lines no longer line up and the screen does
    /// need one honest clear.
    size: Option<(u16, u16)>,
}

impl Screen {
    fn new() -> Self {
        // A cursor parked mid-table and blinking through every repaint reads as
        // flicker of its own.
        print!("\x1b[?25l");
        Self { size: None }
    }

    fn draw(&mut self, frame: &str) {
        let size = crossterm::terminal::size().ok();
        // A pty with no size set (a pipe, a `script` capture) reports zero;
        // that is "unknown", not "no room", so nothing gets trimmed.
        let rows = size
            .filter(|(_, r)| *r >= 3)
            .map(|(_, r)| r as usize)
            .unwrap_or(usize::MAX);

        let mut buf = String::with_capacity(frame.len() + 128);
        if size != self.size {
            buf.push_str("\x1b[2J");
            self.size = size;
        }
        buf.push_str("\x1b[H");

        // A frame taller than the window would scroll, and then the next
        // repaint would land a row too high and smear. Stop one row short and
        // say what was dropped instead.
        let total = frame.lines().count();
        for (drawn, l) in frame.lines().enumerate() {
            if drawn + 1 >= rows {
                let hidden = total - drawn;
                buf.push_str(&format!(
                    "\x1b[7m+{hidden} more line(s) -- enlarge the window\x1b[0m"
                ));
                break;
            }
            buf.push_str(l);
            buf.push_str("\x1b[K\n");
        }
        buf.push_str("\x1b[J");

        let mut out = std::io::stdout().lock();
        let _ = out.write_all(buf.as_bytes());
        let _ = out.flush();
    }

}

impl Drop for Screen {
    /// Also covers the error paths: a controller that goes away mid-watch must
    /// not take the cursor with it.
    fn drop(&mut self) {
        print!("\x1b[?25h");
        let _ = std::io::stdout().flush();
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
        timeout_s: args.timeout.unwrap_or(0),
        submitted_by: std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .unwrap_or_default(),
    };

    let resp = client.submit_job(req).await?.into_inner();
    print!("{}", render::submit(&resp, json));

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
        print!("{}", render::job_detail(&final_job, json));
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
