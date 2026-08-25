//! Table and JSON rendering for the CLI.
//!
//! Every view builds a whole frame as a `String` and hands it back rather than
//! printing as it goes. `ferro watch` needs the finished screen in one write:
//! drawing it piecemeal, after the round-trip to the controller that produced
//! it, is what a redraw at `-n 1` looks like from the other side -- a flicker.

use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use ferro_gpu::fmt_gib;
use ferro_proto::*;

fn table(headers: &[&str]) -> Table {
    let mut t = Table::new();
    t.load_style(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(headers.iter().map(|h| Cell::new(h).fg(Color::Cyan)));
    t
}

fn dump<T: serde::Serialize>(v: &T) -> String {
    format!("{}\n", serde_json::to_string_pretty(v).unwrap_or_default())
}

/// `println!` for a frame under construction.
macro_rules! line {
    ($out:ident) => {
        $out.push('\n')
    };
    ($out:ident, $($arg:tt)*) => {{
        use std::fmt::Write as _;
        let _ = writeln!($out, $($arg)*);
    }};
}

/// How long a process must sit on its VRAM doing nothing before `ferro ps`
/// says so. Short enough to catch a forgotten notebook, long enough not to
/// libel a job between epochs.
const IDLE_AFTER_S: i64 = 3600;

/// Compact duration for a table cell: 45s, 12m, 3h, 5d.
fn short_dur(secs: i64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m", s / 60),
        s if s < 86_400 => format!("{}h", s / 3600),
        s => format!("{}d", s / 86_400),
    }
}

fn now_s() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Seconds since this node last reported. In a live view the numbers are only
/// as fresh as the heartbeat, so showing the age stops a frozen agent from
/// looking like an idle GPU.
fn age_cell(last_seen: i64) -> Cell {
    if last_seen <= 0 {
        return Cell::new("-").fg(Color::Grey);
    }
    let age = (now_s() - last_seen).max(0);
    let c = match age {
        0..=5 => Color::Green,
        6..=15 => Color::Yellow,
        _ => Color::Red,
    };
    Cell::new(format!("{age}s")).fg(c)
}

fn bar(pct: u32, width: usize) -> String {
    let filled = (pct as usize * width / 100).min(width);
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(width - filled))
}

fn health_cell(healthy: bool) -> Cell {
    if healthy {
        Cell::new("ready").fg(Color::Green)
    } else {
        Cell::new("stale").fg(Color::Red)
    }
}

pub fn nodes(nodes: &[NodeState], json: bool) -> String {
    if json {
        let v: Vec<_> = nodes
            .iter()
            .map(|n| {
                let i = n.info.clone().unwrap_or_default();
                serde_json::json!({
                    "node_id": i.node_id,
                    "hostname": i.hostname,
                    "address": i.address,
                    "nccl_address": i.nccl_address,
                    "healthy": n.healthy,
                    "gpus": i.gpus.len(),
                    "free_gpus": n.free_gpus,
                    "driver": i.driver_version,
                    "cuda": i.cuda_version,
                    "cpus": i.cpu_count,
                    "gpu_error": i.gpu_error,
                })
            })
            .collect();
        return dump(&v);
    }

    if nodes.is_empty() {
        return "No nodes registered. Start ferro-agent on your servers.\n".into();
    }

    let mut t = table(&["NODE", "ADDRESS", "NCCL IP", "STATUS", "AGE", "GPUS", "FREE", "DRIVER", "CUDA", "CPU"]);
    for n in nodes {
        let i = n.info.clone().unwrap_or_default();
        t.add_row(vec![
            Cell::new(&i.node_id),
            Cell::new(&i.address),
            Cell::new(&i.nccl_address),
            health_cell(n.healthy),
            age_cell(n.last_seen_unix_s),
            Cell::new(i.gpus.len()),
            Cell::new(n.free_gpus),
            Cell::new(&i.driver_version),
            Cell::new(&i.cuda_version),
            Cell::new(i.cpu_count),
        ]);
    }
    let mut out = String::new();
    line!(out, "{t}");

    for n in nodes {
        if let Some(i) = &n.info {
            if !i.gpu_error.is_empty() {
                line!(out, "  ! {}: GPU detection failed: {}", i.node_id, i.gpu_error);
            }
        }
    }
    out
}

pub fn gpus(entries: &[GpuEntry], json: bool) -> String {
    if json {
        let v: Vec<_> = entries
            .iter()
            .map(|e| {
                let g = e.gpu.clone().unwrap_or_default();
                serde_json::json!({
                    "node_id": e.node_id,
                    "index": g.index,
                    "uuid": g.uuid,
                    "name": g.name,
                    "memory_total_b": g.memory_total_b,
                    "memory_used_b": g.memory_used_b,
                    "utilization_pct": g.utilization_pct,
                    "temperature_c": g.temperature_c,
                    "power_usage_w": g.power_usage_mw / 1000,
                    "power_limit_w": g.power_limit_mw / 1000,
                    "cuda_capability": g.cuda_capability,
                    "allocated_job_id": g.allocated_job_id,
                    "healthy": e.healthy,
                })
            })
            .collect();
        return dump(&v);
    }

    if entries.is_empty() {
        return "No GPUs reported yet.\n".into();
    }

    let mut t = table(&["NODE", "IDX", "NAME", "VRAM USED / TOTAL", "UTIL", "TEMP", "POWER", "CC", "JOB"]);
    for e in entries {
        let g = e.gpu.clone().unwrap_or_default();
        let util = Cell::new(format!("{}%", g.utilization_pct)).fg(match g.utilization_pct {
            0..=10 => Color::Grey,
            11..=70 => Color::Yellow,
            _ => Color::Green,
        });
        t.add_row(vec![
            Cell::new(&e.node_id),
            Cell::new(g.index),
            Cell::new(&g.name),
            Cell::new(format!("{} / {}", fmt_gib(g.memory_used_b), fmt_gib(g.memory_total_b))),
            util,
            Cell::new(format!("{}C", g.temperature_c)),
            Cell::new(format!("{}/{}W", g.power_usage_mw / 1000, g.power_limit_mw / 1000)),
            Cell::new(&g.cuda_capability),
            owner_cell(&g, &e.occupants, e.schedulable),
        ]);
    }
    let total = entries.len();
    // "Free" means placeable, which is the controller's call: no job of ours
    // and enough VRAM left for one.
    let free = entries.iter().filter(|e| e.schedulable).count();

    let mut out = String::new();
    line!(out, "{t}");
    line!(out, "{free}/{total} GPU(s) free");
    out
}

fn phase_cell(p: JobPhase) -> Cell {
    let c = match p {
        JobPhase::Succeeded => Color::Green,
        JobPhase::Failed => Color::Red,
        JobPhase::Running => Color::Cyan,
        JobPhase::Cancelled => Color::Yellow,
        _ => Color::Grey,
    };
    Cell::new(p.label()).fg(c)
}

fn ts(unix: i64) -> String {
    if unix <= 0 {
        return "-".into();
    }
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|d| d.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
        .unwrap_or_else(|| "-".into())
}

fn num(v: f64) -> String {
    if v.is_nan() || v == 0.0 {
        "-".into()
    } else {
        format!("{v:.2}")
    }
}

pub fn jobs(list: &[JobSummary], json: bool) -> String {
    if json {
        let v: Vec<_> = list
            .iter()
            .map(|j| {
                let m = j.metrics.clone().unwrap_or_default();
                serde_json::json!({
                    "job_id": j.job_id,
                    "name": j.name,
                    "phase": j.phase().label(),
                    "world_size": j.plan.as_ref().map(|p| p.world_size).unwrap_or(0),
                    "submitted_unix_s": j.submitted_unix_s,
                    "step": m.step,
                    "samples_per_s": m.samples_per_s,
                    "nccl_errors": j.nccl_errors.len(),
                })
            })
            .collect();
        return dump(&v);
    }

    if list.is_empty() {
        return "No jobs submitted yet.\n".into();
    }

    let mut t = table(&["JOB ID", "NAME", "PHASE", "WORLD", "NODES", "STEP", "SAMPLES/S", "NCCL ERR", "SUBMITTED"]);
    for j in list {
        let p = j.plan.clone().unwrap_or_default();
        let m = j.metrics.clone().unwrap_or_default();
        t.add_row(vec![
            Cell::new(&j.job_id),
            Cell::new(&j.name),
            phase_cell(j.phase()),
            Cell::new(p.world_size),
            Cell::new(p.placements.len()),
            Cell::new(m.step),
            Cell::new(num(m.samples_per_s)),
            if j.nccl_errors.is_empty() {
                Cell::new("0")
            } else {
                Cell::new(j.nccl_errors.len()).fg(Color::Red)
            },
            Cell::new(ts(j.submitted_unix_s)),
        ]);
    }
    let mut out = String::new();
    line!(out, "{t}");
    out
}

pub fn job_detail(j: &JobSummary, json: bool) -> String {
    if json {
        let m = j.metrics.clone().unwrap_or_default();
        let p = j.plan.clone().unwrap_or_default();
        return dump(&serde_json::json!({
            "job_id": j.job_id,
            "name": j.name,
            "phase": j.phase().label(),
            "master_addr": p.master_addr,
            "master_port": p.master_port,
            "world_size": p.world_size,
            "placements": p.placements.iter().map(|pl| serde_json::json!({
                "node_id": pl.node_id,
                "node_rank": pl.node_rank,
                "gpu_indices": pl.gpu_indices,
                "gpu_uuids": pl.gpu_uuids,
            })).collect::<Vec<_>>(),
            "per_node": j.per_node.iter().map(|s| serde_json::json!({
                "node_id": s.node_id,
                "node_rank": s.node_rank,
                "phase": s.phase().label(),
                "exit_code": s.exit_code,
                "message": s.message,
            })).collect::<Vec<_>>(),
            "metrics": {
                "step": m.step,
                "loss": m.loss,
                "samples_per_s": m.samples_per_s,
                "tokens_per_s": m.tokens_per_s,
                "step_time_ms": m.step_time_ms,
                "peak_vram_gb": m.peak_vram_gb,
                "avg_gpu_util_pct": m.avg_gpu_util_pct,
            },
            "nccl_errors": j.nccl_errors,
        }));
    }

    let mut out = String::new();
    let p = j.plan.clone().unwrap_or_default();
    line!(out, "Job     {}  ({})", j.job_id, j.name);
    line!(out, "Phase   {}", j.phase().label());
    line!(
        out,
        "Rendez  MASTER_ADDR={} MASTER_PORT={} WORLD_SIZE={}",
        p.master_addr,
        p.master_port,
        p.world_size
    );
    line!(out);

    let mut t = table(&["RANK", "NODE", "GPUS", "PHASE", "EXIT", "STARTED", "ENDED", "MESSAGE"]);
    for pl in &p.placements {
        let st = j.per_node.iter().find(|s| s.node_id == pl.node_id);
        let gpus = pl.gpu_indices.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",");
        t.add_row(vec![
            Cell::new(pl.node_rank),
            Cell::new(&pl.node_id),
            Cell::new(gpus),
            st.map(|s| phase_cell(s.phase())).unwrap_or_else(|| Cell::new("pending").fg(Color::Grey)),
            Cell::new(st.map(|s| s.exit_code.to_string()).unwrap_or_else(|| "-".into())),
            Cell::new(ts(st.map(|s| s.started_unix_s).unwrap_or(0))),
            Cell::new(ts(st.map(|s| s.ended_unix_s).unwrap_or(0))),
            Cell::new(st.map(|s| s.message.clone()).unwrap_or_default()),
        ]);
    }
    line!(out, "{t}");

    let m = j.metrics.clone().unwrap_or_default();
    let mut mt = table(&["STEP", "LOSS", "SAMPLES/S", "TOKENS/S", "STEP MS", "PEAK VRAM GB", "AVG GPU UTIL"]);
    mt.add_row(vec![
        Cell::new(m.step),
        Cell::new(num(m.loss)),
        Cell::new(num(m.samples_per_s)),
        Cell::new(num(m.tokens_per_s)),
        Cell::new(num(m.step_time_ms)),
        Cell::new(num(m.peak_vram_gb)),
        Cell::new(format!("{:.0}%", m.avg_gpu_util_pct)),
    ]);
    line!(out, "{mt}");

    if !j.nccl_errors.is_empty() {
        line!(out, "\nNCCL / distributed errors ({}):", j.nccl_errors.len());
        for e in j.nccl_errors.iter().take(20) {
            line!(out, "  {e}");
        }
    }
    out
}

pub fn submit(r: &SubmitJobResponse, json: bool) -> String {
    if json {
        let p = r.plan.clone().unwrap_or_default();
        return dump(&serde_json::json!({
            "job_id": r.job_id,
            "accepted": r.accepted,
            "message": r.message,
            "master_addr": p.master_addr,
            "master_port": p.master_port,
            "world_size": p.world_size,
            "placements": p.placements.iter().map(|pl| serde_json::json!({
                "node_id": pl.node_id,
                "node_rank": pl.node_rank,
                "gpu_indices": pl.gpu_indices,
            })).collect::<Vec<_>>(),
        }));
    }

    if !r.accepted {
        // A rejection is not a view; it belongs on stderr next to the exit code.
        eprintln!("submit failed: {}", r.message);
        return String::new();
    }

    let mut out = String::new();

    let p = r.plan.clone().unwrap_or_default();
    line!(out, "Submitted {}", r.job_id);
    line!(
        out,
        "  MASTER_ADDR={}  MASTER_PORT={}  WORLD_SIZE={}",
        p.master_addr,
        p.master_port,
        p.world_size
    );
    for pl in &p.placements {
        line!(
            out,
            "  NODE_RANK={}  node={}  gpus=[{}]",
            pl.node_rank,
            pl.node_id,
            pl.gpu_indices.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
        );
    }
    out
}

pub fn log_line(l: &LogLine) {
    let tag = format!("[r{}|{}]", l.node_rank, l.node_id);
    if l.stream == "stderr" {
        eprintln!("{tag} {}", l.line);
    } else {
        println!("{tag} {}", l.line);
    }
}

/// One-screen live view: GPUs with utilisation bars, plus anything running.
pub fn dashboard(nodes: &[NodeState], gpus: &[GpuEntry], jobs: &[JobSummary]) -> String {
    let ready = nodes.iter().filter(|n| n.healthy).count();
    let stale: Vec<&str> = nodes
        .iter()
        .filter(|n| !n.healthy)
        .filter_map(|n| n.info.as_ref().map(|i| i.node_id.as_str()))
        .collect();

    let free = gpus.iter().filter(|e| e.schedulable).count();
    let used_b: u64 = gpus.iter().filter_map(|e| e.gpu.as_ref()).map(|g| g.memory_used_b).sum();
    let total_b: u64 = gpus.iter().filter_map(|e| e.gpu.as_ref()).map(|g| g.memory_total_b).sum();

    // The screen can redraw faster than the agents report, so say how old the
    // numbers are -- otherwise a wedged agent looks like an idle GPU.
    let oldest = nodes
        .iter()
        .map(|n| (now_s() - n.last_seen_unix_s).max(0))
        .max()
        .unwrap_or(0);

    let mut out = String::new();
    line!(
        out,
        "nodes {ready}/{} ready   GPUs {free}/{} free   VRAM {} / {}   data age <={oldest}s",
        nodes.len(),
        gpus.len(),
        fmt_gib(used_b),
        fmt_gib(total_b)
    );
    if !stale.is_empty() {
        line!(out, "  ! not reporting: {}", stale.join(", "));
    }
    line!(out);

    let mut t = table(&["NODE", "IDX", "GPU", "UTIL", "", "VRAM", "TEMP", "POWER", "JOB"]);
    for e in gpus {
        let g = e.gpu.clone().unwrap_or_default();
        let util = g.utilization_pct;
        let util_colour = match util {
            0..=10 => Color::Grey,
            11..=70 => Color::Yellow,
            _ => Color::Green,
        };
        let mem_pct = if g.memory_total_b > 0 {
            (g.memory_used_b * 100 / g.memory_total_b) as u32
        } else {
            0
        };
        t.add_row(vec![
            Cell::new(&e.node_id),
            Cell::new(g.index),
            // The marketing name is long and identical across rows; the model
            // suffix is the part that distinguishes cards in a mixed cluster.
            Cell::new(g.name.replace("NVIDIA ", "").replace("GeForce ", "")),
            Cell::new(format!("{util:>3}%")).fg(util_colour),
            Cell::new(bar(util, 12)).fg(util_colour),
            Cell::new(format!(
                "{:>5.1}/{:.0}G {}",
                g.memory_used_b as f64 / (1u64 << 30) as f64,
                g.memory_total_b as f64 / (1u64 << 30) as f64,
                bar(mem_pct, 8)
            )),
            Cell::new(format!("{}C", g.temperature_c)),
            Cell::new(format!("{}W", g.power_usage_mw / 1000)),
            owner_cell(&g, &e.occupants, e.schedulable),
        ]);
    }
    line!(out, "{t}");

    let live: Vec<&JobSummary> = jobs.iter().filter(|j| !j.phase().is_terminal()).collect();
    if live.is_empty() {
        line!(out, "\nNo running jobs.");
        return out;
    }

    line!(out);
    let mut jt = table(&["JOB", "NAME", "PHASE", "WORLD", "STEP", "LOSS", "TOKENS/S", "STEP MS", "VRAM GB", "NCCL ERR"]);
    for j in live {
        let m = j.metrics.clone().unwrap_or_default();
        let p = j.plan.clone().unwrap_or_default();
        jt.add_row(vec![
            Cell::new(&j.job_id),
            Cell::new(&j.name),
            phase_cell(j.phase()),
            Cell::new(p.world_size),
            Cell::new(m.step),
            Cell::new(num(m.loss)),
            Cell::new(num(m.tokens_per_s)),
            Cell::new(num(m.step_time_ms)),
            Cell::new(num(m.peak_vram_gb)),
            if j.nccl_errors.is_empty() {
                Cell::new("0")
            } else {
                Cell::new(j.nccl_errors.len()).fg(Color::Red)
            },
        ]);
    }
    line!(out, "{jt}");
    out
}

fn elapsed(started: i64) -> String {
    if started <= 0 {
        return "-".into();
    }
    let secs = (now_s() - started).max(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// `ferro ps`: one row per rank, so you can see which node and which cards a
/// job is actually occupying rather than just that it exists -- plus a row for
/// every other process holding a GPU, which is what tells "free" apart from
/// "idle but occupied by somebody else".
pub fn processes(procs: &[ProcessEntry], json: bool) -> String {
    if json {
        let v: Vec<_> = procs
            .iter()
            .map(|p| {
                let m = p.metrics.clone().unwrap_or_default();
                serde_json::json!({
                    "job_id": p.job_id,
                    "external": p.external,
                    "pid": p.pid,
                    "command": p.command,
                    "container": p.container,
                    "kind": p.kind,
                    "user": p.user,
                    "runs_as": p.runs_as,
                    "name": p.name,
                    "node_id": p.node_id,
                    "node_last_seen_unix_s": p.node_last_seen_unix_s,
                    "node_rank": p.node_rank,
                    "gpu_indices": p.gpu_indices,
                    "phase": if p.external { external_phase(p) } else { p.phase().label() },
                    "world_size": p.world_size,
                    "started_unix_s": p.started_unix_s,
                    "gpu_util_pct": p.gpu_util_pct,
                    "proc_util_pct": p.proc_util_known.then_some(p.proc_util_pct),
                    "idle_s": idle_for(p),
                    "vram_used_gb": p.vram_used_gb,
                    "step": m.step,
                    "tokens_per_s": m.tokens_per_s,
                })
            })
            .collect();
        return dump(&v);
    }

    if procs.is_empty() {
        return "Nothing running, and no other process is holding a GPU.\n".into();
    }

    // AGE is how long ago the node last reported: every other number on the row
    // is that old, and a wedged agent looks exactly like an idle GPU without it.
    let mut t = table(&["JOB", "USER", "NAME", "NODE", "AGE", "RANK", "GPUS", "PHASE", "UPTIME", "UTIL", "VRAM", "STEP", "TOKENS/S"]);
    for p in procs {
        let m = p.metrics.clone().unwrap_or_default();
        let util = if p.external && p.proc_util_known {
            p.proc_util_pct.round() as u32
        } else {
            p.gpu_util_pct.round() as u32
        };
        // Submitter, plus the node account the container runs as when they
        // differ -- on a shared cluster "whose job" and "which uid wrote this
        // checkpoint" are different questions.
        let who = match (p.user.as_str(), p.runs_as.as_str()) {
            ("", "") => "-".to_string(),
            ("", r) => r.to_string(),
            (u, r) if u == r || r.is_empty() => u.to_string(),
            (u, r) => format!("{u}→{r}"),
        };
        t.add_row(vec![
            if p.external {
                // Nothing owns it as far as FerroGrid is concerned, so name the
                // one handle that does: its pid on that node.
                Cell::new(format!("pid {}", p.pid)).fg(Color::Grey)
            } else {
                Cell::new(&p.job_id)
            },
            Cell::new(who).fg(Color::Blue),
            if p.external {
                Cell::new(describe(p)).fg(Color::Grey)
            } else {
                Cell::new(&p.name)
            },
            Cell::new(&p.node_id),
            age_cell(p.node_last_seen_unix_s),
            if p.external {
                Cell::new("-")
            } else {
                Cell::new(format!("{}/{}", p.node_rank, p.world_size.max(1)))
            },
            Cell::new(p.gpu_indices.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")),
            if p.external {
                // An orphan is one of ours that outlived its job record --
                // usually a container a restarted controller lost track of,
                // and the one external row somebody has to act on. An idle
                // squatter is the other: VRAM held, nothing computing.
                match (external_phase(p), idle_for(p)) {
                    ("orphan", _) => Cell::new("orphan").fg(Color::Yellow),
                    (_, Some(i)) if i >= IDLE_AFTER_S => {
                        Cell::new(format!("idle {}", short_dur(i))).fg(Color::Yellow)
                    }
                    (other, _) => Cell::new(other).fg(Color::Grey),
                }
            } else {
                phase_cell(p.phase())
            },
            Cell::new(elapsed(p.started_unix_s)),
            // For an external row this is the process's own utilisation where
            // the driver can attribute it, and the device's otherwise --
            // greyed in that case to say it is not about this pid. Never red:
            // an idle rank is our stall, an idle squatter is somebody else's.
            Cell::new(format!("{util}%")).fg(match (p.external, p.proc_util_known) {
                (true, true) if util > 0 => Color::Green,
                (true, _) => Color::Grey,
                _ => match util {
                    0..=10 => Color::Red, // running but idle GPUs means a stall
                    11..=70 => Color::Yellow,
                    _ => Color::Green,
                },
            }),
            Cell::new(format!("{:.1}G", p.vram_used_gb)),
            if p.external { Cell::new("-") } else { Cell::new(m.step) },
            if p.external { Cell::new("-") } else { Cell::new(num(m.tokens_per_s)) },
        ]);
    }
    let ranks = procs.iter().filter(|p| !p.external).count();
    let external = procs.len() - ranks;
    // Deliberately not called "other users": the process may well be your own
    // shell, and VRAM does not care whose it is.
    let others = match external {
        0 => String::new(),
        n => format!(", {n} other process(es) holding GPUs"),
    };

    let mut out = String::new();
    line!(out, "{t}");
    line!(out, "{ranks} rank(s) running{others}");
    out
}

/// The JOB column of a GPU table: our job when we placed one, otherwise
/// whoever else is on the card. "-" has to mean *nobody*, or the whole table
/// reads as an empty cluster while somebody's training runs on it.
fn owner_cell(g: &Gpu, occupants: &[GpuOccupant], schedulable: bool) -> Cell {
    if !g.allocated_job_id.is_empty() {
        return Cell::new(&g.allocated_job_id).fg(Color::Magenta);
    }
    let Some(top) = occupants.first() else {
        return Cell::new("-").fg(Color::Grey);
    };
    let more = match occupants.len() {
        1 => String::new(),
        n => format!(" +{}", n - 1),
    };
    let label = format!(
        "ext:{}{more} {:.0}G",
        if top.user.is_empty() { "?" } else { &top.user },
        occupants.iter().map(|o| o.memory_used_b as f64 / (1u64 << 30) as f64).sum::<f64>()
    );
    // Idle is the actionable case: somebody is holding the card, not using
    // it. A card with room left is greyed -- the compositor's 80 MB is worth
    // knowing about but is not why your job did not place.
    let idle = occupants
        .iter()
        .all(|o| o.busy_unix_s > 0 && now_s() - o.busy_unix_s >= IDLE_AFTER_S);
    Cell::new(label).fg(match (schedulable, idle) {
        (true, _) => Color::Grey,
        (false, true) => Color::Yellow,
        (false, false) => Color::Blue,
    })
}

/// `ferro ps --by-user`: who is holding what, across the cluster.
///
/// The question a shared cluster actually asks is not "which processes exist"
/// but "whose are they, and can I ask for the card back".
pub fn processes_by_user(procs: &[ProcessEntry], json: bool) -> String {
    #[derive(Default)]
    struct Tally {
        nodes: std::collections::BTreeSet<String>,
        gpus: usize,
        vram_gb: f64,
        procs: usize,
        idle_procs: usize,
        oldest: i64,
        ours: usize,
    }

    let mut by_user: std::collections::BTreeMap<String, Tally> = Default::default();
    for p in procs {
        let who = match (p.user.as_str(), p.runs_as.as_str()) {
            ("", "") => "-",
            ("", r) => r,
            (u, _) => u,
        };
        let t = by_user.entry(who.to_string()).or_default();
        t.nodes.insert(p.node_id.clone());
        t.gpus += p.gpu_indices.len();
        t.vram_gb += p.vram_used_gb;
        t.procs += 1;
        if idle_for(p).is_some_and(|i| i >= IDLE_AFTER_S) {
            t.idle_procs += 1;
        }
        if !p.external {
            t.ours += 1;
        }
        if p.started_unix_s > 0 && (t.oldest == 0 || p.started_unix_s < t.oldest) {
            t.oldest = p.started_unix_s;
        }
    }

    if json {
        let v: Vec<_> = by_user
            .iter()
            .map(|(user, t)| {
                serde_json::json!({
                    "user": user,
                    "nodes": t.nodes.iter().collect::<Vec<_>>(),
                    "gpus": t.gpus,
                    "vram_gb": t.vram_gb,
                    "processes": t.procs,
                    "ferro_ranks": t.ours,
                    "idle_processes": t.idle_procs,
                    "oldest_unix_s": t.oldest,
                })
            })
            .collect();
        return dump(&v);
    }

    if by_user.is_empty() {
        return "Nobody is holding a GPU.\n".into();
    }

    // Biggest holder first: that is who you go and talk to.
    let mut rows: Vec<(&String, &Tally)> = by_user.iter().collect();
    rows.sort_by(|a, b| b.1.vram_gb.partial_cmp(&a.1.vram_gb).unwrap_or(std::cmp::Ordering::Equal));

    let mut t = table(&["USER", "NODES", "GPUS", "VRAM", "PROCS", "VIA FERRO", "IDLE", "OLDEST"]);
    for (user, v) in &rows {
        t.add_row(vec![
            Cell::new(user).fg(Color::Blue),
            Cell::new(v.nodes.iter().cloned().collect::<Vec<_>>().join(",")),
            Cell::new(v.gpus),
            Cell::new(format!("{:.1}G", v.vram_gb)),
            Cell::new(v.procs),
            if v.ours > 0 { Cell::new(v.ours).fg(Color::Green) } else { Cell::new("-").fg(Color::Grey) },
            if v.idle_procs > 0 {
                Cell::new(v.idle_procs).fg(Color::Yellow)
            } else {
                Cell::new("-").fg(Color::Grey)
            },
            Cell::new(elapsed(v.oldest)),
        ]);
    }

    let mut out = String::new();
    line!(out, "{t}");
    line!(
        out,
        "{} user(s), {:.1}G held in total",
        rows.len(),
        rows.iter().map(|(_, v)| v.vram_gb).sum::<f64>()
    );
    out
}

/// How long a process has been holding VRAM without computing, when the
/// driver could tell us. `None` means the question is unanswerable here, which
/// is not the same as "busy" -- and only one of the two justifies calling
/// somebody's job a squatter.
pub fn idle_for(p: &ProcessEntry) -> Option<i64> {
    if !p.proc_util_known || p.busy_unix_s <= 0 {
        return None;
    }
    // A compositor holding 50 MB is idle by construction and nobody's problem;
    // listing it as a squatter buries the ones that are.
    if p.kind == "graphics" {
        return None;
    }
    Some((now_s() - p.busy_unix_s).max(0))
}

/// Keep only processes idle for at least `secs`. Our own ranks never qualify:
/// nothing attributes utilisation to them, and an idle rank is a stall to
/// debug rather than a squatter to evict.
pub fn only_idle(procs: Vec<ProcessEntry>, secs: Option<u32>) -> Vec<ProcessEntry> {
    let Some(secs) = secs else { return procs };
    procs
        .into_iter()
        .filter(|p| idle_for(p).is_some_and(|i| i >= secs as i64))
        .collect()
}

/// What an external row is: a leftover of ours, somebody's compute job, or
/// just the machine's display server -- which holds VRAM but is nobody's
/// problem, and saying so keeps people from hunting it down.
fn external_phase(p: &ProcessEntry) -> &'static str {
    if !p.job_id.is_empty() {
        "orphan"
    } else if p.kind == "graphics" {
        "display"
    } else {
        "external"
    }
}

/// Command line for an external process, prefixed with its container when it
/// has one -- `docker kill <name>` and `kill <pid>` are different fixes.
fn describe(p: &ProcessEntry) -> String {
    let cmd = if p.command.is_empty() { "?" } else { p.command.as_str() };
    let cmd = shorten(cmd, 64);
    if p.container.is_empty() {
        cmd
    } else {
        format!("[{}] {cmd}", p.container)
    }
}

/// Make a command fit a table cell. argv[0] is usually an absolute path into a
/// venv or a conda env -- most of the width and none of the information, since
/// the script name right after it is what identifies the work. The untouched
/// command line is still in `--json`.
fn shorten(cmd: &str, max: usize) -> String {
    let (head, rest) = match cmd.split_once(' ') {
        Some((h, r)) => (h, r),
        None => (cmd, ""),
    };
    let head = head.rsplit('/').next().unwrap_or(head);
    let short = if rest.is_empty() { head.to_string() } else { format!("{head} {rest}") };
    if short.chars().count() <= max {
        return short;
    }
    format!("{}...", short.chars().take(max).collect::<String>())
}


/// `ferro bench`: measured throughput per GPU, with a relative column so it is
/// obvious which cards the scheduler will favour.
pub fn benchmarks(results: &[GpuBenchmark], json: bool) -> String {
    if json {
        let v: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "node_id": r.node_id, "index": r.index, "uuid": r.uuid,
                    "name": r.name, "tflops": r.tflops, "error": r.error,
                })
            })
            .collect();
        return dump(&v);
    }

    if results.is_empty() {
        return "No results.\n".into();
    }

    let best = results.iter().map(|r| r.tflops).fold(0.0_f64, f64::max);
    let mut t = table(&["NODE", "IDX", "GPU", "BF16 TFLOP/S", "RELATIVE", "NOTE"]);
    for r in results {
        let rel = if best > 0.0 { r.tflops / best } else { 0.0 };
        t.add_row(vec![
            Cell::new(&r.node_id),
            Cell::new(r.index),
            Cell::new(r.name.replace("NVIDIA ", "").replace("GeForce ", "")),
            if r.tflops > 0.0 {
                Cell::new(format!("{:.1}", r.tflops))
            } else {
                Cell::new("-").fg(Color::Grey)
            },
            if r.tflops > 0.0 {
                Cell::new(format!("{:.0}%  {}", rel * 100.0, bar((rel * 100.0) as u32, 10)))
                    .fg(if rel > 0.85 { Color::Green } else { Color::Yellow })
            } else {
                Cell::new("-").fg(Color::Grey)
            },
            if r.error.is_empty() {
                Cell::new("")
            } else {
                Cell::new(&r.error).fg(Color::Red)
            },
        ]);
    }
    let mut out = String::new();
    line!(out, "{t}");
    line!(out, "Scores are cached on each node and used to rank placements.");
    out
}

pub fn plugins(list: &[PluginInfo], json: bool) -> String {
    if json {
        let v: Vec<_> = list
            .iter()
            .map(|p| {
                serde_json::json!({
                    "name": p.name, "description": p.description,
                    "fetch": p.can_fetch, "push": p.can_push,
                })
            })
            .collect();
        return dump(&v);
    }

    if list.is_empty() {
        return "No plugins configured.\n\
                Copy plugins.example.toml to ~/.config/ferrogrid/plugins.toml on the\n\
                controller host and restart it.\n"
            .into();
    }

    let mut t = table(&["PLUGIN", "FETCH", "PUSH", "DESCRIPTION"]);
    for p in list {
        let mark = |ok: bool| {
            if ok {
                Cell::new("yes").fg(Color::Green)
            } else {
                Cell::new("-").fg(Color::Grey)
            }
        };
        t.add_row(vec![
            Cell::new(&p.name),
            mark(p.can_fetch),
            mark(p.can_push),
            Cell::new(&p.description),
        ]);
    }
    let mut out = String::new();
    line!(out, "{t}");
    out
}

pub fn transfer(results: &[PluginResult], json: bool) -> String {
    if json {
        let v: Vec<_> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "node_id": r.node_id, "exit_code": r.exit_code,
                    "seconds": r.seconds, "output": r.output, "error": r.error,
                })
            })
            .collect();
        return dump(&v);
    }

    let mut t = table(&["NODE", "RESULT", "TOOK"]);
    for r in results {
        t.add_row(vec![
            Cell::new(&r.node_id),
            if r.exit_code == 0 {
                Cell::new("ok").fg(Color::Green)
            } else {
                Cell::new(format!("failed ({})", r.exit_code)).fg(Color::Red)
            },
            Cell::new(format!("{:.1}s", r.seconds)),
        ]);
    }
    let mut out = String::new();
    line!(out, "{t}");

    // Only the failures' output, and only the tail of it: a successful
    // transfer's progress bars are noise, a failure's last lines are the reason.
    for r in results.iter().filter(|r| r.exit_code != 0) {
        line!(out, "\n--- {} ---", r.node_id);
        for l in r.error.lines().rev().take(12).collect::<Vec<_>>().iter().rev() {
            line!(out, "  {l}");
        }
        if r.error.trim().is_empty() {
            for l in r.output.lines().rev().take(8).collect::<Vec<_>>().iter().rev() {
                line!(out, "  {l}");
            }
        }
    }

    let ok = results.iter().filter(|r| r.exit_code == 0).count();
    line!(out, "\n{ok}/{} node(s) succeeded", results.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_lose_their_interpreter_path_first() {
        assert_eq!(shorten("/opt/conda/envs/x/bin/python train.py --lr 3e-4", 64), "python train.py --lr 3e-4");
        assert_eq!(shorten("./gpu_loop /work/poly.cado", 64), "gpu_loop /work/poly.cado");
        assert_eq!(shorten("nvidia-smi", 64), "nvidia-smi");
    }

    #[test]
    fn what_is_left_is_capped() {
        assert_eq!(shorten("python a.py --flag value", 12), "python a.py ...");
    }
}
