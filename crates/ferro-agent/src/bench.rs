//! Per-GPU throughput measurement.
//!
//! The scheduler needs to compare real hardware, and a model name is a poor
//! proxy: an RTX 4090 and an A6000 have similar VRAM but very different bf16
//! throughput, and a card being thermally throttled or shared with someone
//! else's job is slower still than its datasheet.
//!
//! We measure rather than guess: a bf16 matmul in the node's own training
//! image, which also proves the image can actually drive the GPU. Results are
//! cached on disk because the number barely moves between reboots.

use crate::state::SharedState;
use anyhow::{Context, Result};
use ferro_proto::GpuBenchmark;
use std::collections::HashMap;
use std::path::PathBuf;

/// Big enough that the measurement is compute-bound rather than launch-bound.
const BENCH_PY: &str = r#"
import json, os, time, torch
idx = int(os.environ["FERRO_BENCH_GPU"])
torch.cuda.set_device(idx)
dev = torch.device("cuda", idx)
n = 8192
a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
for _ in range(5):
    a @ b
torch.cuda.synchronize(dev)
t0 = time.perf_counter()
iters = 30
for _ in range(iters):
    a @ b
torch.cuda.synchronize(dev)
dt = (time.perf_counter() - t0) / iters
# One matmul is 2*n^3 FLOPs.
print("FERRO_BENCH " + json.dumps({"tflops": (2 * n ** 3) / dt / 1e12}))
"#;

fn cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".ferrogrid-bench.json")
}

fn load_cache() -> HashMap<String, f64> {
    std::fs::read_to_string(cache_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_cache(c: &HashMap<String, f64>) {
    if let Ok(s) = serde_json::to_string_pretty(c) {
        let _ = std::fs::write(cache_path(), s);
    }
}

/// Benchmark every GPU on this node. Cached results are reused unless `force`.
pub async fn run(state: SharedState, force: bool) -> Result<Vec<GpuBenchmark>> {
    let gpus = state.gpu_snapshot().await;
    let mut cache = load_cache();
    let mut out = Vec::new();

    for gpu in &gpus {
        if !force {
            if let Some(tflops) = cache.get(&gpu.uuid) {
                out.push(GpuBenchmark {
                    node_id: state.node_id.clone(),
                    index: gpu.index,
                    uuid: gpu.uuid.clone(),
                    name: gpu.name.clone(),
                    tflops: *tflops,
                    error: String::new(),
                });
                continue;
            }
        }

        // The 8192^2 bf16 matmul needs roughly 1 GiB of working set. A card
        // that cannot spare it is not a card we would schedule onto either,
        // so report why rather than letting the container OOM with an opaque
        // CUDA error.
        const NEEDED_B: u64 = 2 << 30;
        let free_b = gpu.memory_total_b.saturating_sub(gpu.memory_used_b);
        if gpu.allocated_job_id.is_empty() && free_b < NEEDED_B {
            out.push(GpuBenchmark {
                node_id: state.node_id.clone(),
                index: gpu.index,
                uuid: gpu.uuid.clone(),
                name: gpu.name.clone(),
                tflops: cache.get(&gpu.uuid).copied().unwrap_or(0.0),
                error: format!(
                    "only {:.1} GiB free; another process is using this GPU",
                    free_b as f64 / (1u64 << 30) as f64
                ),
            });
            continue;
        }

        // Never benchmark a GPU someone is training on: the number would be
        // wrong and we would be stealing their throughput to get it.
        if !gpu.allocated_job_id.is_empty() {
            out.push(GpuBenchmark {
                node_id: state.node_id.clone(),
                index: gpu.index,
                uuid: gpu.uuid.clone(),
                name: gpu.name.clone(),
                tflops: cache.get(&gpu.uuid).copied().unwrap_or(0.0),
                error: format!("busy with job {}", gpu.allocated_job_id),
            });
            continue;
        }

        let result = measure(&state, gpu.index).await;
        let (tflops, error) = match result {
            Ok(t) => {
                cache.insert(gpu.uuid.clone(), t);
                (t, String::new())
            }
            Err(e) => (0.0, format!("{e:#}")),
        };
        tracing::info!(gpu = gpu.index, tflops, "benchmark: {}", if error.is_empty() { "ok" } else { &error });
        out.push(GpuBenchmark {
            node_id: state.node_id.clone(),
            index: gpu.index,
            uuid: gpu.uuid.clone(),
            name: gpu.name.clone(),
            tflops,
            error,
        });
    }

    save_cache(&cache);
    Ok(out)
}

async fn measure(state: &SharedState, index: u32) -> Result<f64> {
    let output = if state.no_docker {
        tokio::process::Command::new("python3")
            .args(["-c", BENCH_PY])
            .env("FERRO_BENCH_GPU", index.to_string())
            .env("CUDA_VISIBLE_DEVICES", index.to_string())
            .output()
            .await
            .context("run benchmark on host")?
    } else {
        tokio::process::Command::new("docker")
            .args([
                "run", "--rm",
                "--gpus", &format!("\"device={index}\""),
                // The container sees one GPU, so it is always index 0 inside.
                "-e", "FERRO_BENCH_GPU=0",
                &state.default_image,
                "python", "-c", BENCH_PY,
            ])
            .output()
            .await
            .context("run benchmark container")?
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Some(line) = stdout.lines().find(|l| l.contains("FERRO_BENCH")) {
        let json = &line[line.find('{').context("malformed benchmark output")?..];
        let v: serde_json::Value = serde_json::from_str(json).context("parse benchmark json")?;
        return v["tflops"].as_f64().context("missing tflops");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!(
        "benchmark produced no result: {}",
        stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("(no output)")
    )
}
