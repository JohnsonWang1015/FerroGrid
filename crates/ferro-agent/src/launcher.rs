//! Builds and supervises one torchrun process per job, normally inside Docker.
//!
//! We deliberately do not reimplement any part of torchrun, NCCL or FSDP: the
//! agent's whole job is to compute the rendezvous environment the controller
//! decided on, hand it to stock `torchrun`, and stream the result back.

use crate::state::{RunningJob, SharedState};
use anyhow::{Context, Result};
use ferro_proto::controller_client::ControllerClient;
use ferro_proto::{
    JobPhase, JobStatus, LaunchJobRequest, LogLine, ReportJobStatusRequest, ReportLogsRequest,
};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

fn now_s() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Spawn the job and return immediately; supervision continues in the background.
pub async fn launch(state: SharedState, req: LaunchJobRequest) -> Result<()> {
    let container = format!("ferro-{}-r{}", req.job_id, req.node_rank);
    let (program, argv) = build_command(&state, &req, &container);

    tracing::info!(job = %req.job_id, rank = req.node_rank, "launching: {program} {}", argv.join(" "));

    let mut cmd = tokio::process::Command::new(&program);
    cmd.args(&argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        // Put the child in its own process group so a Ctrl-C in the agent's
        // terminal does not race our explicit container teardown.
        .process_group(0);

    // When running without Docker the env has to go on the host process.
    if state.no_docker {
        for (k, v) in torch_env(&state, &req) {
            cmd.env(k, v);
        }
        cmd.current_dir(resolve_workdir(&state, &req));
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {program} for job {}", req.job_id))?;

    let stdout = child.stdout.take().context("capture stdout")?;
    let stderr = child.stderr.take().context("capture stderr")?;

    let status = JobStatus {
        job_id: req.job_id.clone(),
        node_id: state.node_id.clone(),
        node_rank: req.node_rank,
        phase: JobPhase::Running as i32,
        exit_code: 0,
        message: String::new(),
        started_unix_s: now_s(),
        ended_unix_s: 0,
    };

    {
        let mut jobs = state.jobs.lock().await;
        jobs.insert(
            req.job_id.clone(),
            RunningJob {
                job_id: req.job_id.clone(),
                node_rank: req.node_rank,
                gpu_indices: req.gpu_indices.clone(),
                status: status.clone(),
                container: (!state.no_docker).then_some(container.clone()),
                child: None, // ownership moves into the supervisor below
            },
        );
    }

    // One buffered channel per job feeds a single uploader task, so a slow
    // controller applies backpressure to the readers instead of unbounded RAM.
    let (tx, rx) = mpsc::channel::<LogLine>(4096);
    tokio::spawn(upload_logs(state.clone(), rx));

    spawn_reader(stdout, "stdout", state.clone(), req.clone(), tx.clone());
    spawn_reader(stderr, "stderr", state.clone(), req.clone(), tx.clone());

    report_status(&state, status).await;

    tokio::spawn(supervise(state, req, child, container, tx));
    Ok(())
}

fn spawn_reader<R>(
    reader: R,
    stream: &'static str,
    state: SharedState,
    req: LaunchJobRequest,
    tx: mpsc::Sender<LogLine>,
) where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let msg = LogLine {
                job_id: req.job_id.clone(),
                node_id: state.node_id.clone(),
                node_rank: req.node_rank,
                stream: stream.to_string(),
                line,
                unix_ms: now_ms(),
            };
            if tx.send(msg).await.is_err() {
                break;
            }
        }
    });
}

/// Batches log lines and ships them to the controller. Drops lines rather than
/// blocking the training process if the controller is unreachable.
async fn upload_logs(state: SharedState, mut rx: mpsc::Receiver<LogLine>) {
    let mut client = None;
    let mut batch: Vec<LogLine> = Vec::new();

    loop {
        let deadline = tokio::time::sleep(Duration::from_millis(250));
        tokio::pin!(deadline);

        tokio::select! {
            maybe = rx.recv() => {
                match maybe {
                    Some(l) => {
                        batch.push(l);
                        if batch.len() < 256 { continue; }
                    }
                    None => {
                        flush(&state, &mut client, &mut batch).await;
                        return;
                    }
                }
            }
            _ = &mut deadline => {}
        }

        if !batch.is_empty() {
            flush(&state, &mut client, &mut batch).await;
        }
    }
}

async fn flush(
    state: &SharedState,
    client: &mut Option<ControllerClient<tonic::transport::Channel>>,
    batch: &mut Vec<LogLine>,
) {
    if batch.is_empty() {
        return;
    }
    if client.is_none() {
        *client = ControllerClient::connect(state.controller.clone()).await.ok();
    }
    let Some(c) = client.as_mut() else {
        batch.clear();
        return;
    };
    let lines = std::mem::take(batch);
    if let Err(e) = c.report_logs(ReportLogsRequest { lines }).await {
        tracing::debug!("log upload failed: {e}");
        *client = None;
    }
}

async fn supervise(
    state: SharedState,
    req: LaunchJobRequest,
    mut child: tokio::process::Child,
    container: String,
    tx: mpsc::Sender<LogLine>,
) {
    let result = child.wait().await;
    // Readers hold clones; dropping ours lets the uploader finish once they end.
    drop(tx);

    let (code, phase, message) = match result {
        Ok(s) if s.success() => (0, JobPhase::Succeeded, String::new()),
        Ok(s) => (
            s.code().unwrap_or(-1),
            JobPhase::Failed,
            format!("exited with status {s}"),
        ),
        Err(e) => (-1, JobPhase::Failed, format!("wait failed: {e}")),
    };

    // A cancelled job reports as cancelled even though the process exited non-zero.
    let cancelled = {
        let jobs = state.jobs.lock().await;
        jobs.get(&req.job_id)
            .map(|j| j.status.phase() == JobPhase::Cancelled)
            .unwrap_or(false)
    };
    let phase = if cancelled { JobPhase::Cancelled } else { phase };

    if !state.no_docker {
        // Best-effort: the container is normally gone thanks to --rm.
        let _ = tokio::process::Command::new("docker")
            .args(["rm", "-f", &container])
            .output()
            .await;
    }

    let status = JobStatus {
        job_id: req.job_id.clone(),
        node_id: state.node_id.clone(),
        node_rank: req.node_rank,
        phase: phase as i32,
        exit_code: code,
        message,
        started_unix_s: 0,
        ended_unix_s: now_s(),
    };

    {
        let mut jobs = state.jobs.lock().await;
        if let Some(j) = jobs.get_mut(&req.job_id) {
            let started = j.status.started_unix_s;
            j.status = JobStatus { started_unix_s: started, ..status.clone() };
            j.child = None;
        }
    }

    tracing::info!(job = %req.job_id, rank = req.node_rank, code, "job finished: {}", phase.label());
    report_status(&state, status).await;
}

async fn report_status(state: &SharedState, status: JobStatus) {
    if let Ok(mut c) = ControllerClient::connect(state.controller.clone()).await {
        let _ = c
            .report_job_status(ReportJobStatusRequest { status: Some(status) })
            .await;
    }
}

/// The distributed-training environment. This is the contract with torchrun:
/// the controller owns the placement decision, the agent only materialises it.
fn torch_env(state: &SharedState, req: &LaunchJobRequest) -> Vec<(String, String)> {
    let devices = req
        .gpu_indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let world_size = req.nnodes * req.nproc_per_node;

    let mut env = vec![
        ("MASTER_ADDR".to_string(), req.master_addr.clone()),
        ("MASTER_PORT".to_string(), req.master_port.to_string()),
        ("WORLD_SIZE".to_string(), world_size.to_string()),
        ("NODE_RANK".to_string(), req.node_rank.to_string()),
        ("NNODES".to_string(), req.nnodes.to_string()),
        ("NPROC_PER_NODE".to_string(), req.nproc_per_node.to_string()),
        ("FERRO_JOB_ID".to_string(), req.job_id.clone()),
        ("FERRO_NODE_ID".to_string(), state.node_id.clone()),
        // Inside the container CUDA_VISIBLE_DEVICES is already narrowed by
        // `--gpus`, so we only set it on the host path.
        ("PYTHONUNBUFFERED".to_string(), "1".to_string()),
        // Lab machines have no InfiniBand; asking NCCL for it wastes 30s of
        // probing before it falls back to sockets.
        ("NCCL_IB_DISABLE".to_string(), "1".to_string()),
    ];

    // Pin both NCCL and the gloo/TCPStore rendezvous to the real LAN
    // interface, so neither wanders onto a docker bridge the peer cannot reach.
    if let Some(ifname) = &state.nccl_ifname {
        env.push(("NCCL_SOCKET_IFNAME".to_string(), ifname.clone()));
        env.push(("GLOO_SOCKET_IFNAME".to_string(), ifname.clone()));
    }

    if state.no_docker {
        env.push(("CUDA_VISIBLE_DEVICES".to_string(), devices));
    }

    // Caller-supplied values win, so a job can override e.g. NCCL_DEBUG.
    for (k, v) in &req.env {
        env.retain(|(ek, _)| ek != k);
        env.push((k.clone(), v.clone()));
    }
    env
}

fn torchrun_argv(req: &LaunchJobRequest) -> Vec<String> {
    let mut v = vec![
        "torchrun".to_string(),
        format!("--nnodes={}", req.nnodes),
        format!("--nproc_per_node={}", req.nproc_per_node),
        format!("--node_rank={}", req.node_rank),
        format!("--master_addr={}", req.master_addr),
        format!("--master_port={}", req.master_port),
    ];
    v.extend(req.torchrun_args.iter().cloned());
    v.push(req.script.clone());
    v.extend(req.script_args.iter().cloned());
    v
}

/// Nodes in a lab rarely share a home directory, so the controller sends a
/// path relative to each agent's own workspace root. An absolute path is
/// honoured as-is, for shared NFS setups.
fn resolve_workdir(state: &SharedState, req: &LaunchJobRequest) -> String {
    if req.workdir.is_empty() {
        return state.workspace.clone();
    }
    if req.workdir.starts_with('/') {
        return req.workdir.clone();
    }
    format!("{}/{}", state.workspace.trim_end_matches('/'), req.workdir)
}

/// Returns (program, argv) for the supervised child.
fn build_command(
    state: &SharedState,
    req: &LaunchJobRequest,
    container: &str,
) -> (String, Vec<String>) {
    let torchrun = torchrun_argv(req);

    if state.no_docker {
        let mut it = torchrun.into_iter();
        let prog = it.next().unwrap();
        return (prog, it.collect());
    }

    let image = if req.image.is_empty() { &state.default_image } else { &req.image };
    let devices = req
        .gpu_indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let mut argv: Vec<String> = vec![
        "run".into(),
        "--rm".into(),
        "--name".into(),
        container.to_string(),
        // Host networking keeps the NCCL/rendezvous ports reachable between
        // nodes without publishing a port range per job.
        "--network".into(),
        "host".into(),
        "--ipc".into(),
        "host".into(),
        "--shm-size".into(),
        "8g".into(),
        "--ulimit".into(),
        "memlock=-1".into(),
        "--gpus".into(),
        // The quotes are part of the VALUE, not shell syntax: docker CSV-parses
        // this flag, so an unquoted `device=0,1` splits into `device=0` plus a
        // bare `1` that it reads as a device *count* -- "cannot set both Count
        // and DeviceIDs". Quoting keeps it one device list.
        format!("\"device={devices}\""),
    ];

    // Run as the invoking user so checkpoints written to the bind mount are
    // not left root-owned on the host.
    let (uid, gid) = (unsafe { libc_getuid() }, unsafe { libc_getgid() });
    argv.push("--user".into());
    argv.push(format!("{uid}:{gid}"));

    let workdir = resolve_workdir(state, req);
    argv.push("-v".into());
    argv.push(format!("{workdir}:{workdir}"));
    argv.push("-w".into());
    argv.push(workdir.clone());
    // HOME must be writable for torch/triton caches when running as --user.
    argv.push("-e".into());
    argv.push(format!("HOME={workdir}"));

    for (k, v) in torch_env(state, req) {
        argv.push("-e".into());
        argv.push(format!("{k}={v}"));
    }

    argv.push(image.clone());
    argv.extend(torchrun);

    ("docker".to_string(), argv)
}

// Avoid pulling in the whole `libc` crate for two calls.
extern "C" {
    #[link_name = "getuid"]
    fn c_getuid() -> u32;
    #[link_name = "getgid"]
    fn c_getgid() -> u32;
}
unsafe fn libc_getuid() -> u32 { c_getuid() }
unsafe fn libc_getgid() -> u32 { c_getgid() }
