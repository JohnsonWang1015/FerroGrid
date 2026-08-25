//! Who is actually holding each GPU on this node -- including everything
//! FerroGrid never launched.
//!
//! A GPU with no FerroGrid job on it is not necessarily a free GPU: on a shared
//! lab box somebody's interactive notebook, another scheduler's container or a
//! desktop session hold VRAM just as effectively. NVML lists the pids; the rest
//! of the work here is turning a pid into something a human recognises (owner,
//! command, container) and deciding whether it is one of ours.
//!
//! Everything is best-effort: `/proc` entries disappear between the NVML call
//! and the read, and a hardened host may hide other users' processes entirely.
//! A process we cannot describe is still reported -- "pid 4711 is holding 20
//! GiB" is the useful half of the answer.

use crate::state::AgentState;
use ferro_proto::GpuProcess;
use std::collections::HashMap;
use std::sync::OnceLock;

/// `docker ps` is far too expensive to run on every heartbeat, and a container
/// id never changes name, so the map is only rebuilt when we meet an id we do
/// not know -- and then at most this often.
const CONTAINER_REFRESH_S: i64 = 5;
/// Enough to recognise the command; a torchrun line can run to kilobytes.
const MAX_COMMAND_LEN: usize = 256;

/// Lookups that are expensive to redo every heartbeat and stable once known.
#[derive(Default)]
pub struct Lookups {
    /// Full container id -> container name.
    containers: HashMap<String, String>,
    containers_refreshed_s: i64,
    /// uid -> login name.
    users: HashMap<u32, String>,
    /// pid -> last time it was seen using the GPU. Idleness is a duration, and
    /// NVML only ever reports an instant, so somebody has to keep the clock.
    busy: HashMap<u32, i64>,
}

fn now_s() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Every GPU-holding process on this node, ready to ship in a heartbeat.
pub async fn snapshot(state: &AgentState) -> Vec<GpuProcess> {
    let raw = state.monitor.processes();
    if raw.is_empty() {
        return Vec::new();
    }

    // Our own jobs, so their ranks are not reported as somebody else's work.
    // Containerised jobs are matched by container, host jobs by ancestry: the
    // torchrun workers are children of the process we spawned, but a container's
    // processes hang off containerd, not off us.
    let (by_container, launcher_pids) = {
        let jobs = state.jobs.lock().await;
        let mut by_container = HashMap::new();
        let mut pids = HashMap::new();
        for job in jobs.values() {
            if job.status.phase().is_terminal() {
                continue;
            }
            if let Some(c) = &job.container {
                by_container.insert(c.clone(), job.job_id.clone());
            }
            if let Some(pid) = job.launcher_pid {
                pids.insert(pid, job.job_id.clone());
            }
        }
        (by_container, pids)
    };

    let mut details: HashMap<u32, Details> = HashMap::new();
    for p in &raw {
        details.entry(p.pid).or_insert_with(|| read_details(p.pid));
    }

    // Per-pid utilisation, when the driver can attribute it. `None` is "no
    // answer", and is reported as such rather than as 0%.
    let util = state.monitor.process_utilization();
    let now = now_s();

    let mut lookups = state.lookups.lock().await;
    resolve_containers(&mut lookups, &details).await;

    if let Some(util) = util.as_ref() {
        let live: Vec<u32> = raw.iter().map(|p| p.pid).collect();
        for pid in &live {
            let busy = util.get(pid).copied().unwrap_or(0) > 0;
            // A pid we have never seen starts its idle clock now: the agent
            // may have just started, and "unknown since boot" must not be
            // reported as "idle since boot".
            let e = lookups.busy.entry(*pid).or_insert(now);
            if busy {
                *e = now;
            }
        }
        lookups.busy.retain(|pid, _| live.contains(pid));
    }

    let mut out = Vec::with_capacity(raw.len());
    for p in raw {
        let d = details.get(&p.pid).cloned().unwrap_or_default();
        let container = d
            .container_id
            .as_ref()
            .and_then(|id| lookups.containers.get(id).cloned())
            .unwrap_or_default();

        // Ours by container name, ours by descent, or somebody else's. A
        // `ferro-` container we have no live job for still names its job id:
        // that is exactly the stray left behind by a restarted agent, and
        // saying so beats reporting it as an anonymous squatter.
        let job_id = by_container
            .get(&container)
            .cloned()
            .or_else(|| parse_ferro_container(&container))
            .or_else(|| d.ancestors.iter().find_map(|a| launcher_pids.get(a).cloned()))
            .unwrap_or_default();

        out.push(GpuProcess {
            gpu_index: p.gpu_index,
            pid: p.pid,
            memory_used_b: p.memory_used_b,
            user: user_name(&mut lookups, d.uid),
            command: d.command.clone(),
            started_unix_s: d.started_unix_s,
            container,
            job_id,
            kind: if p.graphics { "graphics" } else { "compute" }.into(),
            utilization_pct: util.as_ref().and_then(|u| u.get(&p.pid).copied()).unwrap_or(0),
            utilization_known: util.is_some(),
            busy_unix_s: lookups.busy.get(&p.pid).copied().unwrap_or(0),
        });
    }
    out
}

#[derive(Clone, Default)]
struct Details {
    uid: Option<u32>,
    command: String,
    started_unix_s: i64,
    container_id: Option<String>,
    /// The process itself and everything between it and init, nearest first.
    ancestors: Vec<u32>,
}

fn read_details(pid: u32) -> Details {
    let mut d = Details {
        uid: read_uid(pid),
        command: read_command(pid),
        ..Default::default()
    };
    if let Some((_, starttime_ticks)) = read_stat(pid) {
        d.started_unix_s = start_time_unix_s(starttime_ticks);
    }
    d.ancestors = ancestry(pid);
    d.container_id = read_container_id(pid);
    d
}

/// argv joined for display; `comm` when the process is not ours to read.
fn read_command(pid: u32) -> String {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let joined = String::from_utf8_lossy(&raw)
        .split('\0')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    // `python -c` puts a whole script in one argv element; kept verbatim it
    // turns one table row into twenty.
    let joined = one_line(&joined);
    let cmd = if joined.is_empty() {
        // Kernel threads have an empty cmdline, and so do zombies.
        std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .unwrap_or_default()
            .trim()
            .to_string()
    } else {
        joined
    };
    truncate(&cmd, MAX_COMMAND_LEN)
}

/// Collapse embedded newlines and runs of whitespace so a command stays one
/// table row.
fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}...")
}

fn read_uid(pid: u32) -> Option<u32> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    parse_uid(&status)
}

/// The real uid is the first of the four on the `Uid:` line.
fn parse_uid(status: &str) -> Option<u32> {
    status
        .lines()
        .find(|l| l.starts_with("Uid:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

fn read_stat(pid: u32) -> Option<(u32, u64)> {
    parse_stat(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// `(ppid, starttime)` from `/proc/<pid>/stat`.
///
/// The second field is the executable name in parentheses and may itself
/// contain spaces and parentheses (`(a b) c`), so everything is counted from
/// the *last* `)` rather than by splitting the whole line.
fn parse_stat(stat: &str) -> Option<(u32, u64)> {
    let rest = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // fields[0] is field 3 (state), so field N lives at index N - 3.
    let ppid = fields.get(1)?.parse().ok()?;
    let starttime = fields.get(19)?.parse().ok()?;
    Some((ppid, starttime))
}

/// Ticks since boot -> wall clock. `USER_HZ` is 100 on Linux regardless of the
/// kernel's internal tick rate; it is part of the userspace ABI.
fn start_time_unix_s(ticks: u64) -> i64 {
    const USER_HZ: u64 = 100;
    match boot_time_unix_s() {
        Some(btime) => btime + (ticks / USER_HZ) as i64,
        None => 0,
    }
}

fn boot_time_unix_s() -> Option<i64> {
    static BTIME: OnceLock<Option<i64>> = OnceLock::new();
    *BTIME.get_or_init(|| {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        stat.lines()
            .find(|l| l.starts_with("btime "))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
    })
}

fn ancestry(mut pid: u32) -> Vec<u32> {
    let mut out = Vec::new();
    // Bounded: a /proc that lies about parentage must not spin the agent.
    for _ in 0..32 {
        if pid <= 1 {
            break;
        }
        out.push(pid);
        match read_stat(pid) {
            Some((ppid, _)) => pid = ppid,
            None => break,
        }
    }
    out
}

fn read_container_id(pid: u32) -> Option<String> {
    parse_cgroup(&std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?)
}

/// Pull a container id out of a cgroup file. Covers the shapes docker and
/// containerd produce under both cgroup versions:
///   `0::/system.slice/docker-<id>.scope`
///   `12:memory:/docker/<id>`
///   `0::/kubepods/burstable/pod<uid>/<id>`
fn parse_cgroup(cgroup: &str) -> Option<String> {
    for line in cgroup.lines() {
        let path = line.rsplit(':').next()?;
        for token in path.split(['/', '-', '.']) {
            if token.len() == 64 && token.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Some(token.to_string());
            }
        }
    }
    None
}

/// `ferro-<job-id>-r<rank>` -> job id. Used when the container outlived the
/// agent that started it, so it is no longer in the job table.
fn parse_ferro_container(name: &str) -> Option<String> {
    let rest = name.strip_prefix("ferro-")?;
    let (job, rank) = rest.rsplit_once("-r")?;
    if job.is_empty() || rank.is_empty() || !rank.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(job.to_string())
}

/// Fill in any container ids we have not seen before, at most every
/// `CONTAINER_REFRESH_S`. Nodes running without docker never pay for this:
/// no process has a container id, so the command never runs.
async fn resolve_containers(lookups: &mut Lookups, details: &HashMap<u32, Details>) {
    let unknown = details
        .values()
        .filter_map(|d| d.container_id.as_ref())
        .any(|id| !lookups.containers.contains_key(id));
    let now = now_s();
    if !unknown || now - lookups.containers_refreshed_s < CONTAINER_REFRESH_S {
        return;
    }
    lookups.containers_refreshed_s = now;

    let out = tokio::process::Command::new("docker")
        .args(["ps", "--no-trunc", "--format", "{{.ID}} {{.Names}}"])
        .output()
        .await;
    let Ok(out) = out else { return };
    if !out.status.success() {
        tracing::debug!("docker ps failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        return;
    }
    // Rebuild rather than merge, so ids of long-dead containers do not
    // accumulate for the lifetime of the agent.
    lookups.containers = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(id, name)| (id.trim().to_string(), name.trim().to_string()))
        .collect();
}

/// uid -> login name. `/etc/passwd` first; `getent` covers LDAP/NIS lab
/// accounts that are not in the local file. Both answers are cached forever,
/// since a uid does not change hands underneath a running process.
fn user_name(lookups: &mut Lookups, uid: Option<u32>) -> String {
    let Some(uid) = uid else { return String::new() };
    if let Some(name) = lookups.users.get(&uid) {
        return name.clone();
    }
    let name = passwd_file_name(uid)
        .or_else(|| getent_name(uid))
        .unwrap_or_else(|| format!("uid:{uid}"));
    lookups.users.insert(uid, name.clone());
    name
}

fn passwd_file_name(uid: u32) -> Option<String> {
    let passwd = std::fs::read_to_string("/etc/passwd").ok()?;
    lookup_passwd(&passwd, uid)
}

/// `name:x:uid:gid:...`
fn lookup_passwd(passwd: &str, uid: u32) -> Option<String> {
    passwd.lines().find_map(|l| {
        let mut f = l.split(':');
        let name = f.next()?;
        let _ = f.next()?;
        (f.next()?.parse::<u32>().ok()? == uid).then(|| name.to_string())
    })
}

fn getent_name(uid: u32) -> Option<String> {
    let out = std::process::Command::new("getent")
        .args(["passwd", &uid.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    lookup_passwd(&String::from_utf8_lossy(&out.stdout), uid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_survives_a_command_containing_spaces_and_parens() {
        // Field 22 (starttime) is 8371, ppid is 1234.
        let mut fields: Vec<String> = (3..=52).map(|n| n.to_string()).collect();
        fields[1] = "1234".into(); // field 4: ppid
        fields[19] = "8371".into(); // field 22: starttime
        let stat = format!("42 (py (a b) thon) {}", fields.join(" "));
        assert_eq!(parse_stat(&stat), Some((1234, 8371)));
    }

    #[test]
    fn uid_is_the_real_one() {
        let status = "Name:\tpython\nUid:\t1001\t1001\t1001\t1001\nGid:\t100\n";
        assert_eq!(parse_uid(status), Some(1001));
    }

    #[test]
    fn container_id_from_both_cgroup_layouts() {
        let id = "a".repeat(64);
        assert_eq!(parse_cgroup(&format!("0::/system.slice/docker-{id}.scope")), Some(id.clone()));
        assert_eq!(parse_cgroup(&format!("12:memory:/docker/{id}\n11:cpu:/docker/{id}")), Some(id.clone()));
        assert_eq!(parse_cgroup(&format!("0::/kubepods/burstable/podabc/{id}")), Some(id));
        assert_eq!(parse_cgroup("0::/user.slice/user-1000.slice/session-3.scope"), None);
    }

    #[test]
    fn ferro_container_names_carry_their_job_id() {
        assert_eq!(parse_ferro_container("ferro-abc123-r0").as_deref(), Some("abc123"));
        // A job id may itself contain the separator.
        assert_eq!(parse_ferro_container("ferro-job-r2-x-r11").as_deref(), Some("job-r2-x"));
        assert_eq!(parse_ferro_container("someone-elses-container"), None);
        assert_eq!(parse_ferro_container("ferro-abc-rx"), None);
    }

    #[test]
    fn passwd_lookup_matches_on_uid() {
        let passwd = "root:x:0:0:root:/root:/bin/bash\njohnson:x:1001:1001::/home/johnson:/bin/bash\n";
        assert_eq!(lookup_passwd(passwd, 1001).as_deref(), Some("johnson"));
        assert_eq!(lookup_passwd(passwd, 4242), None);
    }

    #[test]
    fn long_commands_are_trimmed() {
        assert_eq!(truncate("abcdef", 3), "abc...");
        assert_eq!(truncate("abc", 3), "abc");
    }

    #[test]
    fn embedded_scripts_collapse_to_one_line() {
        assert_eq!(one_line("python -c \nimport torch\n\nx = 1\n"), "python -c import torch x = 1");
    }
}
