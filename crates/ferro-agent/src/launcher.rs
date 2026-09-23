//! Builds and supervises one torchrun process per job, normally inside Docker.
//!
//! We deliberately do not reimplement any part of torchrun, NCCL or FSDP: the
//! agent's whole job is to compute the rendezvous environment the controller
//! decided on, hand it to stock `torchrun`, and stream the result back.

use crate::procs::{self, ProcId};
use crate::state::{RunningJob, SharedState};
use anyhow::{Context, Result};
use ferro_proto::controller_client::ControllerClient;
use ferro_proto::{
    JobPhase, JobStatus, LaunchJobRequest, LogLine, ReportJobStatusRequest, ReportLogsRequest,
};
use std::process::Stdio;
use std::time::{Duration, Instant};
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
    let image = effective_image(&state, &req).to_string();
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

    let launcher_pid = child.id();
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
        image: image.clone(),
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
                launcher_pid,
                stopper: None,
                stopping: false,
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

    tokio::spawn(supervise(state, req, child, container, tx, image));
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
        *client = ControllerClient::connect(state.controller.clone())
            .await
            .ok();
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
    image: String,
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

    // A cancelled job reports as cancelled even though the process exited
    // non-zero, and its message is the teardown's verdict rather than the
    // signal the launcher happened to die of.
    let stopper = {
        let mut jobs = state.jobs.lock().await;
        match jobs.get_mut(&req.job_id) {
            Some(j) if j.status.phase() == JobPhase::Cancelled => Some(j.stopper.take()),
            _ => None,
        }
    };
    let (phase, message) = match stopper {
        // The launcher is reaped by now, so all that is left to wait for is
        // the escalation to SIGKILL -- bounded by the grace period.
        Some(handle) => {
            let outcome = match handle {
                Some(h) => h.await.ok(),
                None => None,
            };
            (JobPhase::Cancelled, cancel_message(outcome, state.grace))
        }
        None => (phase, message),
    };

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
        image,
    };

    {
        let mut jobs = state.jobs.lock().await;
        if let Some(j) = jobs.get_mut(&req.job_id) {
            let started = j.status.started_unix_s;
            j.status = JobStatus {
                started_unix_s: started,
                ..status.clone()
            };
            // The GPUs are genuinely free now, which they were not while the
            // teardown was still running.
            j.stopping = false;
        }
    }

    tracing::info!(job = %req.job_id, rank = req.node_rank, code, "job finished: {}", phase.label());
    report_status(&state, status).await;
}

async fn report_status(state: &SharedState, status: JobStatus) {
    if let Ok(mut c) = ControllerClient::connect(state.controller.clone()).await {
        let _ = c
            .report_job_status(ReportJobStatusRequest {
                status: Some(status),
            })
            .await;
    }
}

/// How a stop ended. An operator reading "cancelled" learns nothing about
/// whether their training script honours SIGTERM at all; this is the
/// difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOutcome {
    /// Everything was gone before the grace period ran out.
    Graceful,
    /// It was not, and got SIGKILL.
    Forced,
}

/// How often the teardown re-checks whether the job's processes are gone.
const TEARDOWN_POLL: Duration = Duration::from_millis(100);
/// How often it re-walks /proc looking for workers it has not seen yet.
const RESCAN_EVERY: Duration = Duration::from_secs(1);

/// Tear a job down: ask politely, wait up to `grace`, then insist.
///
/// Caller-driven rather than a method on the job so that it can run as its own
/// task: the controller stops each rank of a job in turn, and waiting out the
/// grace period inside the gRPC handler would multiply it by the rank count.
pub async fn terminate(
    container: Option<String>,
    launcher_pid: Option<u32>,
    grace: Duration,
) -> StopOutcome {
    match (container, launcher_pid) {
        (Some(name), _) => stop_container(&name, grace).await,
        (None, Some(pid)) => stop_process_tree(pid, grace).await,
        // Nothing was ever spawned, so nothing is left to signal.
        (None, None) => StopOutcome::Graceful,
    }
}

async fn stop_container(name: &str, grace: Duration) -> StopOutcome {
    let started = Instant::now();
    let _ = tokio::process::Command::new("docker")
        .args(docker_stop_argv(name, grace))
        .output()
        .await;
    // `docker stop` returns the moment the container is down and only reaches
    // for SIGKILL once the whole timeout has passed, so the elapsed time is
    // the runtime telling us which of the two happened.
    if started.elapsed() >= grace {
        StopOutcome::Forced
    } else {
        StopOutcome::Graceful
    }
}

/// `docker stop`, not `docker kill`: stop is SIGTERM, wait, SIGKILL, done by
/// the runtime. Either way the container is addressed by name -- killing the
/// docker CLI process instead would orphan it.
fn docker_stop_argv(name: &str, grace: Duration) -> Vec<String> {
    vec![
        "stop".into(),
        "--time".into(),
        grace.as_secs().to_string(),
        name.to_string(),
    ]
}

async fn stop_process_tree(launcher_pid: u32, grace: Duration) -> StopOutcome {
    // The workers are not in the launcher's process group -- torchrun gives
    // each one its own session -- and they are unreachable through parentage
    // as soon as the launcher dies. So they are enumerated first and remembered.
    let mut workers = procs::descendants(launcher_pid);

    let _ = signal_group(launcher_pid, SIGTERM);
    for w in &workers {
        // torchrun's own handler signals each worker group, so this is usually
        // redundant -- but a torchrun that is wedged or already gone would
        // otherwise strand exactly the processes holding the GPUs.
        signal_worker(*w, SIGTERM);
    }

    let deadline = Instant::now() + grace;
    let mut rescanned = Instant::now();
    loop {
        let launcher_up = group_alive(launcher_pid);
        // Re-walking catches a worker spawned just after the first snapshot,
        // and is only possible while the parent links still exist. It reads
        // the whole of /proc, so it is throttled rather than run every poll:
        // a launcher deaf to SIGTERM would otherwise be scanned for the whole
        // grace period.
        if launcher_up && rescanned.elapsed() >= RESCAN_EVERY {
            rescanned = Instant::now();
            merge(&mut workers, procs::descendants(launcher_pid));
        }
        if !launcher_up && !workers.iter().any(|w| procs::is_alive(*w)) {
            return StopOutcome::Graceful;
        }
        if Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(TEARDOWN_POLL).await;
    }

    let _ = signal_group(launcher_pid, SIGKILL);
    for w in &workers {
        if procs::is_alive(*w) {
            signal_worker(*w, SIGKILL);
        }
    }
    StopOutcome::Forced
}

/// Signal one worker: its group first, because a worker is a group leader and
/// its own children (dataloader workers) belong to that group, then the pid
/// itself in case it is not a leader after all.
fn signal_worker(w: ProcId, sig: i32) {
    let _ = signal_group(w.pid, sig);
    let _ = signal_pid(w.pid, sig);
}

fn merge(known: &mut Vec<ProcId>, found: Vec<ProcId>) {
    for f in found {
        if !known.iter().any(|k| k.pid == f.pid) {
            known.push(f);
        }
    }
}

/// What the job's final status says once the teardown has resolved.
fn cancel_message(outcome: Option<StopOutcome>, grace: Duration) -> String {
    match outcome {
        Some(StopOutcome::Graceful) => "cancelled by controller; exited on SIGTERM".into(),
        Some(StopOutcome::Forced) => format!(
            "cancelled by controller; did not exit in {} s and was killed",
            grace.as_secs()
        ),
        // The teardown was awaited elsewhere -- agent shutdown takes the handle
        // -- or never ran at all.
        None => "cancelled by controller".into(),
    }
}

const SIGTERM: i32 = 15;
const SIGKILL: i32 = 9;
const ESRCH: i32 = 3;

/// Signal every process in the group led by `pgid`. The negative pid is the
/// whole point: the process we spawn is a launcher, and the processes actually
/// holding the GPUs are the ones underneath it.
fn signal_group(pgid: u32, sig: i32) -> Result<(), i32> {
    kill_raw(-(pgid as i32), sig)
}

fn signal_pid(pid: u32, sig: i32) -> Result<(), i32> {
    kill_raw(pid as i32, sig)
}

/// True while any process remains in the group.
fn group_alive(pgid: u32) -> bool {
    // Signal 0 delivers nothing and only reports whether the target exists.
    // Anything but ESRCH is read as "still there": a SIGKILL aimed at a group
    // that has already gone is harmless, whereas calling a live group dead is
    // exactly the leak being fixed.
    signal_group(pgid, 0) != Err(ESRCH)
}

/// `kill(2)`, with errno captured at the call so a later syscall cannot
/// overwrite it.
fn kill_raw(target: i32, sig: i32) -> Result<(), i32> {
    if unsafe { c_kill(target, sig) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or_default())
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

/// Accepts "HOST", "HOST:CONTAINER" and "HOST:CONTAINER:ro", and expands the
/// bare form to "HOST:HOST" so docker mounts it at the same path.
fn normalise_mount(spec: &str) -> String {
    if spec.contains(':') {
        spec.to_string()
    } else {
        format!("{spec}:{spec}")
    }
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

    let image = effective_image(state, req);
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

    // Datasets and checkpoint dirs live outside the workspace, typically on
    // shared NFS. Mount them at the same path inside the container unless the
    // caller asked for a different one, so a path in the job's config file
    // means the same thing on the host and in the container.
    for m in &req.mounts {
        argv.push("-v".into());
        argv.push(normalise_mount(m));
    }

    for (k, v) in torch_env(state, req) {
        argv.push("-e".into());
        argv.push(format!("{k}={v}"));
    }

    argv.push(image.to_string());
    argv.extend(torchrun);

    ("docker".to_string(), argv)
}

/// Resolve the image once at launch time. An agent running without Docker has
/// no image to report because it executes torchrun directly on the host.
fn effective_image<'a>(state: &'a SharedState, req: &'a LaunchJobRequest) -> &'a str {
    resolve_image(&req.image, &state.default_image, state.no_docker)
}

fn resolve_image<'a>(requested: &'a str, default: &'a str, no_docker: bool) -> &'a str {
    if no_docker {
        ""
    } else if requested.is_empty() {
        default
    } else {
        requested
    }
}

// Avoid pulling in the whole `libc` crate for three calls.
extern "C" {
    #[link_name = "getuid"]
    fn c_getuid() -> u32;
    #[link_name = "getgid"]
    fn c_getgid() -> u32;
    #[link_name = "kill"]
    fn c_kill(pid: i32, sig: i32) -> i32;
}
unsafe fn libc_getuid() -> u32 {
    c_getuid()
}
unsafe fn libc_getgid() -> u32 {
    c_getgid()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stand-in for torchrun: a launcher whose worker sits in a session of its
    /// own, exactly as `start_new_session=True` leaves a real one, so a kill
    /// aimed at the launcher's process group cannot reach it.
    async fn spawn_detached_worker(deaf_to_sigterm: bool) -> (tokio::process::Child, u32) {
        let script = if deaf_to_sigterm {
            r#"trap "" TERM; setsid sleep 30 & wait"#
        } else {
            "setsid sleep 30 & wait"
        };
        let child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()
            .expect("spawn test launcher");
        let pid = child.id().expect("launcher pid");
        (child, pid)
    }

    /// Process group id: field 5 of `/proc/<pid>/stat`, counted from the last
    /// `)` for the same reason `parse_stat` is.
    fn read_pgid(pid: u32) -> Option<u32> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        stat[stat.rfind(')')? + 1..]
            .split_whitespace()
            .nth(2)?
            .parse()
            .ok()
    }

    async fn wait_for_worker(launcher: u32) -> Vec<ProcId> {
        for _ in 0..100 {
            let found = procs::descendants(launcher);
            if !found.is_empty() {
                // If the fixture ever stops detaching the worker the tests
                // below would pass against a plain process-group kill, which
                // is the bug they exist to catch.
                assert!(
                    found.iter().any(|w| read_pgid(w.pid) != Some(launcher)),
                    "fixture is not reproducing torchrun's detached session"
                );
                return found;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("test launcher never spawned its worker");
    }

    async fn all_gone(workers: &[ProcId]) -> bool {
        for _ in 0..100 {
            if workers.iter().all(|w| !procs::is_alive(*w)) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        false
    }

    #[tokio::test]
    async fn terminate_reaches_a_worker_outside_the_launchers_group() {
        let (mut child, pid) = spawn_detached_worker(false).await;
        let workers = wait_for_worker(pid).await;

        // The supervisor is what reaps the launcher in production; without a
        // waiter here it would linger as a zombie and never look gone.
        let waiter = tokio::spawn(async move { child.wait().await });
        let outcome = terminate(None, Some(pid), Duration::from_secs(5)).await;
        let _ = waiter.await;

        assert_eq!(outcome, StopOutcome::Graceful);
        assert!(all_gone(&workers).await, "a worker survived SIGTERM");
    }

    #[tokio::test]
    async fn terminate_kills_a_job_that_ignores_sigterm() {
        let (mut child, pid) = spawn_detached_worker(true).await;
        let workers = wait_for_worker(pid).await;

        let waiter = tokio::spawn(async move { child.wait().await });
        let outcome = terminate(None, Some(pid), Duration::from_secs(1)).await;
        let _ = waiter.await;

        assert_eq!(outcome, StopOutcome::Forced);
        assert!(all_gone(&workers).await, "a worker survived SIGKILL");
    }

    #[test]
    fn the_stop_message_says_whether_the_job_honoured_sigterm() {
        assert_eq!(
            cancel_message(Some(StopOutcome::Graceful), Duration::from_secs(10)),
            "cancelled by controller; exited on SIGTERM"
        );
        assert_eq!(
            cancel_message(Some(StopOutcome::Forced), Duration::from_secs(10)),
            "cancelled by controller; did not exit in 10 s and was killed"
        );
        // No verdict is still an honest message, not a guess at one.
        assert_eq!(
            cancel_message(None, Duration::from_secs(10)),
            "cancelled by controller"
        );
    }

    #[test]
    fn the_grace_period_reaches_docker_as_its_stop_timeout() {
        assert_eq!(
            docker_stop_argv("ferro-j1-r0", Duration::from_secs(30)),
            vec!["stop", "--time", "30", "ferro-j1-r0"]
        );
    }

    #[test]
    fn reports_the_effective_docker_image() {
        assert_eq!(
            resolve_image("", "node/default:tag", false),
            "node/default:tag"
        );
        assert_eq!(
            resolve_image("requested:tag", "node/default:tag", false),
            "requested:tag"
        );
        assert_eq!(resolve_image("requested:tag", "node/default:tag", true), "");
    }
}
