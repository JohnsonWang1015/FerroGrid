//! Table and JSON rendering for the CLI.

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

fn dump<T: serde::Serialize>(v: &T) {
    println!("{}", serde_json::to_string_pretty(v).unwrap_or_default());
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

pub fn nodes(nodes: &[NodeState], json: bool) {
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
        println!("No nodes registered. Start ferro-agent on your servers.");
        return;
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
    println!("{t}");

    for n in nodes {
        if let Some(i) = &n.info {
            if !i.gpu_error.is_empty() {
                println!("  ! {}: GPU detection failed: {}", i.node_id, i.gpu_error);
            }
        }
    }
}

pub fn gpus(entries: &[GpuEntry], json: bool) {
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
        println!("No GPUs reported yet.");
        return;
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
            if g.allocated_job_id.is_empty() {
                Cell::new("-").fg(Color::Grey)
            } else {
                Cell::new(&g.allocated_job_id).fg(Color::Magenta)
            },
        ]);
    }
    println!("{t}");

    let total = entries.len();
    let free = entries
        .iter()
        .filter(|e| e.gpu.as_ref().map(|g| g.allocated_job_id.is_empty()).unwrap_or(false))
        .count();
    println!("{free}/{total} GPU(s) free");
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

pub fn jobs(list: &[JobSummary], json: bool) {
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
        println!("No jobs submitted yet.");
        return;
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
    println!("{t}");
}

pub fn job_detail(j: &JobSummary, json: bool) {
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

    let p = j.plan.clone().unwrap_or_default();
    println!("Job     {}  ({})", j.job_id, j.name);
    println!("Phase   {}", j.phase().label());
    println!(
        "Rendez  MASTER_ADDR={} MASTER_PORT={} WORLD_SIZE={}",
        p.master_addr, p.master_port, p.world_size
    );
    println!();

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
    println!("{t}");

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
    println!("{mt}");

    if !j.nccl_errors.is_empty() {
        println!("\nNCCL / distributed errors ({}):", j.nccl_errors.len());
        for e in j.nccl_errors.iter().take(20) {
            println!("  {e}");
        }
    }
}

pub fn submit(r: &SubmitJobResponse, json: bool) {
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
        eprintln!("submit failed: {}", r.message);
        return;
    }

    let p = r.plan.clone().unwrap_or_default();
    println!("Submitted {}", r.job_id);
    println!(
        "  MASTER_ADDR={}  MASTER_PORT={}  WORLD_SIZE={}",
        p.master_addr, p.master_port, p.world_size
    );
    for pl in &p.placements {
        println!(
            "  NODE_RANK={}  node={}  gpus=[{}]",
            pl.node_rank,
            pl.node_id,
            pl.gpu_indices.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
        );
    }
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
pub fn dashboard(nodes: &[NodeState], gpus: &[GpuEntry], jobs: &[JobSummary]) {
    let ready = nodes.iter().filter(|n| n.healthy).count();
    let stale: Vec<&str> = nodes
        .iter()
        .filter(|n| !n.healthy)
        .filter_map(|n| n.info.as_ref().map(|i| i.node_id.as_str()))
        .collect();

    let free = gpus
        .iter()
        .filter(|e| e.gpu.as_ref().map(|g| g.allocated_job_id.is_empty()).unwrap_or(false))
        .count();
    let used_b: u64 = gpus.iter().filter_map(|e| e.gpu.as_ref()).map(|g| g.memory_used_b).sum();
    let total_b: u64 = gpus.iter().filter_map(|e| e.gpu.as_ref()).map(|g| g.memory_total_b).sum();

    // The screen can redraw faster than the agents report, so say how old the
    // numbers are -- otherwise a wedged agent looks like an idle GPU.
    let oldest = nodes
        .iter()
        .map(|n| (now_s() - n.last_seen_unix_s).max(0))
        .max()
        .unwrap_or(0);

    println!(
        "nodes {ready}/{} ready   GPUs {free}/{} free   VRAM {} / {}   data age <={oldest}s",
        nodes.len(),
        gpus.len(),
        fmt_gib(used_b),
        fmt_gib(total_b)
    );
    if !stale.is_empty() {
        println!("  ! not reporting: {}", stale.join(", "));
    }
    println!();

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
            if g.allocated_job_id.is_empty() {
                Cell::new("-").fg(Color::Grey)
            } else {
                Cell::new(&g.allocated_job_id).fg(Color::Magenta)
            },
        ]);
    }
    println!("{t}");

    let live: Vec<&JobSummary> = jobs.iter().filter(|j| !j.phase().is_terminal()).collect();
    if live.is_empty() {
        println!("\nNo running jobs.");
        return;
    }

    println!();
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
    println!("{jt}");
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
/// job is actually occupying rather than just that it exists.
pub fn processes(procs: &[ProcessEntry], json: bool) {
    if json {
        let v: Vec<_> = procs
            .iter()
            .map(|p| {
                let m = p.metrics.clone().unwrap_or_default();
                serde_json::json!({
                    "job_id": p.job_id,
                    "name": p.name,
                    "node_id": p.node_id,
                    "node_rank": p.node_rank,
                    "gpu_indices": p.gpu_indices,
                    "phase": p.phase().label(),
                    "world_size": p.world_size,
                    "started_unix_s": p.started_unix_s,
                    "gpu_util_pct": p.gpu_util_pct,
                    "vram_used_gb": p.vram_used_gb,
                    "step": m.step,
                    "tokens_per_s": m.tokens_per_s,
                })
            })
            .collect();
        return dump(&v);
    }

    if procs.is_empty() {
        println!("Nothing running.");
        return;
    }

    let mut t = table(&["JOB", "NAME", "NODE", "RANK", "GPUS", "PHASE", "UPTIME", "UTIL", "VRAM", "STEP", "TOKENS/S"]);
    for p in procs {
        let m = p.metrics.clone().unwrap_or_default();
        let util = p.gpu_util_pct.round() as u32;
        t.add_row(vec![
            Cell::new(&p.job_id),
            Cell::new(&p.name),
            Cell::new(&p.node_id),
            Cell::new(format!("{}/{}", p.node_rank, p.world_size.max(1))),
            Cell::new(p.gpu_indices.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")),
            phase_cell(p.phase()),
            Cell::new(elapsed(p.started_unix_s)),
            Cell::new(format!("{util}%")).fg(match util {
                0..=10 => Color::Red,      // running but idle GPUs means a stall
                11..=70 => Color::Yellow,
                _ => Color::Green,
            }),
            Cell::new(format!("{:.1}G", p.vram_used_gb)),
            Cell::new(m.step),
            Cell::new(num(m.tokens_per_s)),
        ]);
    }
    println!("{t}");
    println!("{} rank(s) running", procs.len());
}

/// `ferro bench`: measured throughput per GPU, with a relative column so it is
/// obvious which cards the scheduler will favour.
pub fn benchmarks(results: &[GpuBenchmark], json: bool) {
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
        println!("No results.");
        return;
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
    println!("{t}");
    println!("Scores are cached on each node and used to rank placements.");
}
