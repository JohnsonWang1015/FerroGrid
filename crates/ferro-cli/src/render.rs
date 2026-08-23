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

    let mut t = table(&["NODE", "ADDRESS", "NCCL IP", "STATUS", "GPUS", "FREE", "DRIVER", "CUDA", "CPU"]);
    for n in nodes {
        let i = n.info.clone().unwrap_or_default();
        t.add_row(vec![
            Cell::new(&i.node_id),
            Cell::new(&i.address),
            Cell::new(&i.nccl_address),
            health_cell(n.healthy),
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
