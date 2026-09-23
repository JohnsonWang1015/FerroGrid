//! Shared agent state: GPU monitor, node identity, and the running job table.

use crate::launcher::{self, StopOutcome};
use crate::procs::{self, Lookups};
use crate::Args;
use anyhow::{Context, Result};
use ferro_gpu::GpuMonitor;
use ferro_proto::{Gpu, GpuProcess, JobPhase, JobStatus, NodeInfo};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

pub struct RunningJob {
    pub job_id: String,
    #[allow(dead_code)] // kept for debugging / future per-rank routing
    pub node_rank: u32,
    pub gpu_indices: Vec<u32>,
    pub status: JobStatus,
    /// Container name when running under Docker; the container is stopped by
    /// name because killing the docker CLI would orphan it.
    pub container: Option<String>,
    /// Pid of the process we spawned (torchrun, or the docker CLI); also its
    /// process group id, since the launcher spawns with `process_group(0)`.
    /// Only the no-docker case is useful for attribution or teardown: a
    /// container's workers descend from containerd, not from us.
    pub launcher_pid: Option<u32>,
    /// The in-flight teardown, taken by whoever waits for it -- normally the
    /// supervisor, so that the final status can say how the job stopped.
    pub stopper: Option<JoinHandle<StopOutcome>>,
    /// Set from the moment a stop is asked for until the teardown finishes.
    pub stopping: bool,
}

impl RunningJob {
    /// Whether this job's GPUs are still spoken for. The phase turns terminal
    /// the instant a cancellation is accepted, but the processes holding the
    /// devices are only on their way out: "free" means *placeable*, and a card
    /// whose previous tenant is still exiting is not.
    pub fn holds_gpus(&self) -> bool {
        !self.status.phase().is_terminal() || self.stopping
    }
}

pub struct AgentState {
    pub node_id: String,
    pub hostname: String,
    pub advertise: String,
    pub nccl_ip: String,
    /// Interface that owns `nccl_ip`; pinned into NCCL_SOCKET_IFNAME.
    pub nccl_ifname: Option<String>,
    pub default_image: String,
    pub workspace: String,
    pub no_docker: bool,
    /// How long a cancelled job gets to exit on SIGTERM before it is killed.
    pub grace: Duration,
    pub controller: String,
    pub monitor: GpuMonitor,
    pub jobs: Mutex<HashMap<String, RunningJob>>,
    /// Caches behind `ferro ps`: container names and uid -> login name.
    pub lookups: Mutex<Lookups>,
}

impl AgentState {
    pub fn new(args: &Args) -> Result<Self> {
        let hostname = hostname::get()
            .context("read hostname")?
            .to_string_lossy()
            .to_string();
        let node_id = args.node_id.clone().unwrap_or_else(|| hostname.clone());

        // Work out the address the controller and peer nodes should use. We
        // prefer an explicit flag; otherwise we ask the kernel which local IP
        // it would use to reach the controller, which is the right answer on
        // multi-homed lab boxes covered in docker bridges.
        let port = args.bind.port();
        let advertise = match &args.advertise {
            Some(a) => a.clone(),
            None => format!("{}:{}", detect_local_ip(&args.controller)?, port),
        };
        let advertise_ip = advertise
            .rsplit_once(':')
            .map(|(h, _)| h.to_string())
            .unwrap_or_else(|| advertise.clone());
        let nccl_ip = args.nccl_ip.clone().unwrap_or(advertise_ip);
        let nccl_ifname = args.nccl_ifname.clone().or_else(|| detect_ifname(&nccl_ip));

        match &nccl_ifname {
            Some(n) => tracing::info!("NCCL will use interface {n} ({nccl_ip})"),
            None => tracing::warn!(
                "could not map {nccl_ip} to an interface; NCCL will auto-select, \
                 which often picks a docker bridge and fails between nodes"
            ),
        }

        Ok(Self {
            node_id,
            hostname,
            advertise,
            nccl_ip,
            nccl_ifname,
            default_image: args.default_image.clone(),
            workspace: args.workspace.clone().unwrap_or_else(default_workspace),
            no_docker: args.no_docker,
            grace: Duration::from_secs(args.termination_grace_period_secs),
            controller: args.controller.clone(),
            monitor: GpuMonitor::new(),
            jobs: Mutex::new(HashMap::new()),
            lookups: Mutex::new(Lookups::default()),
        })
    }

    /// GPU index -> job id, for every GPU currently held by a non-terminal job.
    async fn allocations(&self) -> HashMap<u32, String> {
        let jobs = self.jobs.lock().await;
        let mut map = HashMap::new();
        for job in jobs.values() {
            if !job.holds_gpus() {
                continue;
            }
            for idx in &job.gpu_indices {
                map.insert(*idx, job.job_id.clone());
            }
        }
        map
    }

    pub async fn gpu_snapshot(&self) -> Vec<Gpu> {
        let allocs = self.allocations().await;
        self.monitor.snapshot(&allocs)
    }

    /// Every process holding a GPU here, ours and everyone else's.
    pub async fn process_snapshot(&self) -> Vec<GpuProcess> {
        procs::snapshot(self).await
    }

    pub async fn node_info(&self) -> NodeInfo {
        NodeInfo {
            node_id: self.node_id.clone(),
            hostname: self.hostname.clone(),
            address: self.advertise.clone(),
            nccl_address: self.nccl_ip.clone(),
            driver_version: self.monitor.driver_version(),
            cuda_version: self.monitor.cuda_driver_version(),
            agent_version: crate::AGENT_VERSION.to_string(),
            cpu_count: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(0),
            memory_total_b: read_mem_total_bytes(),
            gpus: self.gpu_snapshot().await,
            gpu_error: self.monitor.init_error().unwrap_or_default().to_string(),
            workspace: self.workspace.clone(),
            user: std::env::var("USER").unwrap_or_default(),
            processes: self.process_snapshot().await,
            // Re-read rather than cached: a link that renegotiated after a
            // cable was moved should not need an agent restart to show up.
            link_iface: self.nccl_ifname.clone().unwrap_or_default(),
            link_mbps: self
                .nccl_ifname
                .as_deref()
                .and_then(crate::net::link_speed_mbps)
                .unwrap_or(0),
        }
    }

    pub async fn job_statuses(&self) -> Vec<JobStatus> {
        self.jobs
            .lock()
            .await
            .values()
            .map(|j| j.status.clone())
            .collect()
    }

    /// GPUs held by live jobs; the controller schedules around these but the
    /// agent re-checks so two controllers can't double-book a device.
    pub async fn busy_gpus(&self) -> Vec<u32> {
        self.allocations().await.into_keys().collect()
    }

    pub async fn stop_job(&self, job_id: &str) -> (bool, String) {
        let mut jobs = self.jobs.lock().await;
        let Some(job) = jobs.get_mut(job_id) else {
            return (false, format!("unknown job {job_id}"));
        };
        if job.status.phase().is_terminal() {
            return (false, "job already finished".into());
        }

        job.status.phase = JobPhase::Cancelled as i32;
        job.status.message = "cancelled by controller".into();
        // The teardown runs as its own task and the lock is released here: it
        // may sit through the whole grace period, and neither the other jobs'
        // supervision nor the controller -- which stops each rank in turn --
        // can afford to wait for it.
        job.stopping = true;
        job.stopper = Some(tokio::spawn(launcher::terminate(
            job.container.clone(),
            job.launcher_pid,
            self.grace,
        )));
        (true, format!("stopped {job_id}"))
    }

    pub async fn stop_all(&self) {
        let ids: Vec<String> = self.jobs.lock().await.keys().cloned().collect();
        for id in &ids {
            self.stop_job(id).await;
        }

        // The agent is on its way out, so it waits here rather than exiting
        // part-way through a grace period and leaving the workers behind. The
        // teardowns are already running in parallel, so this costs one grace
        // period, not one per job.
        let stoppers: Vec<(String, JoinHandle<StopOutcome>)> = {
            let mut jobs = self.jobs.lock().await;
            jobs.iter_mut()
                .filter_map(|(id, j)| j.stopper.take().map(|s| (id.clone(), s)))
                .collect()
        };
        for (id, stopper) in stoppers {
            let _ = stopper.await;
            if let Some(job) = self.jobs.lock().await.get_mut(&id) {
                job.stopping = false;
            }
        }
    }
}

/// Find the interface holding `ip`.
///
/// GPU boxes in a lab are covered in docker/calico bridges. Left to itself
/// NCCL enumerates them all and frequently binds one that the peer node cannot
/// reach, which surfaces as a bare `ncclSystemError` at init. Pinning
/// NCCL_SOCKET_IFNAME to the interface that actually carries `nccl_ip` avoids
/// the whole class of failure.
fn detect_ifname(ip: &str) -> Option<String> {
    let out = std::process::Command::new("ip")
        .args(["-o", "-4", "addr", "show"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    // Lines look like: "2: enp6s0    inet 10.0.0.2/24 brd ... scope global ..."
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(name) = tokens.get(1) else { continue };
        let Some(pos) = tokens.iter().position(|t| *t == "inet") else {
            continue;
        };
        let Some(cidr) = tokens.get(pos + 1) else {
            continue;
        };
        if cidr.split('/').next() == Some(ip) {
            return Some(name.to_string());
        }
    }
    None
}

/// Where job files live on this node when the controller sends a relative path.
fn default_workspace() -> String {
    std::env::var("HOME")
        .map(|h| format!("{h}/ferrogrid"))
        .unwrap_or_else(|_| "/tmp/ferrogrid".to_string())
}

/// Ask the routing table which source IP reaches `endpoint`. Uses a connected
/// UDP socket, which sets up no traffic but resolves the route.
fn detect_local_ip(endpoint: &str) -> Result<String> {
    let hostport = endpoint
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/');
    let addr = if hostport.contains(':') {
        hostport.to_string()
    } else {
        format!("{hostport}:80")
    };

    let sock = std::net::UdpSocket::bind("0.0.0.0:0").context("bind probe socket")?;
    sock.connect(&addr)
        .with_context(|| format!("route probe to {addr}"))?;
    Ok(sock
        .local_addr()
        .context("probe local_addr")?
        .ip()
        .to_string())
}

fn read_mem_total_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

pub type SharedState = Arc<AgentState>;
