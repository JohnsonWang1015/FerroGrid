//! GPU inventory and live telemetry via NVML.
//!
//! NVML is opened once and kept alive for the lifetime of the agent; each
//! snapshot re-reads the volatile counters (utilisation, VRAM, power, temp).
//! If NVML is unavailable (no driver, version mismatch, container without
//! `--gpus`) the agent still starts and reports the error upstream rather than
//! crashing, so the controller can show the node as GPU-less.

use anyhow::{Context, Result};
use ferro_proto::Gpu;
use nvml_wrapper::enum_wrappers::device::{Clock, ClockId, TemperatureSensor};
use nvml_wrapper::Nvml;
use std::collections::HashMap;

pub struct GpuMonitor {
    nvml: Option<Nvml>,
    init_error: Option<String>,
    /// Static per-device data that never changes; read once at startup.
    statics: Vec<GpuStatic>,
}

#[derive(Clone, Debug)]
struct GpuStatic {
    index: u32,
    uuid: String,
    name: String,
    memory_total_b: u64,
    power_limit_mw: u32,
    cuda_capability: String,
}

impl GpuMonitor {
    /// Never fails: a node without a usable NVML is still a valid (GPU-less) node.
    pub fn new() -> Self {
        match Nvml::init() {
            Ok(nvml) => {
                let statics = match Self::read_statics(&nvml) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("failed to enumerate GPUs: {e:#}");
                        Vec::new()
                    }
                };
                tracing::info!("NVML initialised, {} GPU(s) detected", statics.len());
                Self { nvml: Some(nvml), init_error: None, statics }
            }
            Err(e) => {
                tracing::warn!("NVML unavailable: {e}");
                Self { nvml: None, init_error: Some(e.to_string()), statics: Vec::new() }
            }
        }
    }

    fn read_statics(nvml: &Nvml) -> Result<Vec<GpuStatic>> {
        let count = nvml.device_count().context("nvml device_count")?;
        let mut out = Vec::with_capacity(count as usize);
        for index in 0..count {
            let d = nvml.device_by_index(index).with_context(|| format!("device {index}"))?;
            let cc = d
                .cuda_compute_capability()
                .map(|c| format!("{}.{}", c.major, c.minor))
                .unwrap_or_default();
            out.push(GpuStatic {
                index,
                uuid: d.uuid().unwrap_or_else(|_| format!("nvml-index-{index}")),
                name: d.name().unwrap_or_else(|_| "unknown".into()),
                memory_total_b: d.memory_info().map(|m| m.total).unwrap_or(0),
                power_limit_mw: d.enforced_power_limit().unwrap_or(0),
                cuda_capability: cc,
            });
        }
        Ok(out)
    }

    pub fn init_error(&self) -> Option<&str> {
        self.init_error.as_deref()
    }

    pub fn gpu_count(&self) -> usize {
        self.statics.len()
    }

    pub fn driver_version(&self) -> String {
        self.nvml
            .as_ref()
            .and_then(|n| n.sys_driver_version().ok())
            .unwrap_or_default()
    }

    pub fn cuda_driver_version(&self) -> String {
        self.nvml
            .as_ref()
            .and_then(|n| n.sys_cuda_driver_version().ok())
            // NVML encodes this as major*1000 + minor*10.
            .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10))
            .unwrap_or_default()
    }

    /// Read live counters for every GPU. `allocations` maps GPU index -> job id
    /// so the controller can see which devices the agent considers busy.
    pub fn snapshot(&self, allocations: &HashMap<u32, String>) -> Vec<Gpu> {
        let Some(nvml) = self.nvml.as_ref() else {
            return Vec::new();
        };
        self.statics
            .iter()
            .map(|s| {
                let mut gpu = Gpu {
                    index: s.index,
                    uuid: s.uuid.clone(),
                    name: s.name.clone(),
                    memory_total_b: s.memory_total_b,
                    power_limit_mw: s.power_limit_mw,
                    cuda_capability: s.cuda_capability.clone(),
                    allocated_job_id: allocations.get(&s.index).cloned().unwrap_or_default(),
                    ..Default::default()
                };
                // A device that vanishes mid-run (fell off the bus, driver reset)
                // is reported with zeroed counters instead of taking the agent down.
                if let Ok(d) = nvml.device_by_index(s.index) {
                    if let Ok(m) = d.memory_info() {
                        gpu.memory_used_b = m.used;
                    }
                    if let Ok(u) = d.utilization_rates() {
                        gpu.utilization_pct = u.gpu;
                        gpu.memory_util_pct = u.memory;
                    }
                    if let Ok(t) = d.temperature(TemperatureSensor::Gpu) {
                        gpu.temperature_c = t;
                    }
                    if let Ok(p) = d.power_usage() {
                        gpu.power_usage_mw = p;
                    }
                }
                gpu
            })
            .collect()
    }

    /// SM clock in MHz, used by the benchmark path. Best-effort.
    pub fn sm_clock_mhz(&self, index: u32) -> Option<u32> {
        let nvml = self.nvml.as_ref()?;
        let d = nvml.device_by_index(index).ok()?;
        d.clock(Clock::SM, ClockId::Current).ok()
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Human-friendly byte formatting used by the CLI tables.
pub fn fmt_gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}
