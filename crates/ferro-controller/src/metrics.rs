//! Parsing of training telemetry out of the job's stdout.
//!
//! The contract with the training script is one line of the form
//!   FERRO_METRIC {"step": 10, "loss": 6.9, "samples_per_s": 128.0}
//! Anything the script does not report is simply left at its previous value.

use ferro_proto::TrainingMetrics;

pub const METRIC_PREFIX: &str = "FERRO_METRIC";

/// Substrings that indicate a genuine distributed failure.
///
/// Deliberately excludes a bare "ProcessGroupNCCL": torch logs many routine
/// `[W...] ProcessGroupNCCL.cpp:...` warnings, and matching them all buries
/// the real errors in noise.
const NCCL_ERROR_MARKERS: &[&str] = &[
    "NCCL WARN",
    "NCCL error",
    "ncclInternalError",
    "ncclUnhandledCudaError",
    "ncclSystemError",
    "ncclInvalidUsage",
    "ncclRemoteError",
    "unhandled system error",
    "torch.distributed.DistBackendError",
    "Watchdog caught collective operation timeout",
    "Socket Timeout",
    "NCCL communicator was aborted",
    "Connection reset by peer",
];

pub fn is_nccl_error(line: &str) -> bool {
    NCCL_ERROR_MARKERS.iter().any(|m| line.contains(m))
}

/// Extract the JSON payload following the marker, wherever it appears in the
/// line (torchrun prefixes ranks, so the marker is rarely at column 0).
pub fn parse_metric_line(line: &str) -> Option<TrainingMetrics> {
    let idx = line.find(METRIC_PREFIX)?;
    let rest = line[idx + METRIC_PREFIX.len()..].trim();
    let start = rest.find('{')?;
    let json = &rest[start..];
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let o = v.as_object()?;

    let f = |k: &str| o.get(k).and_then(|x| x.as_f64());
    Some(TrainingMetrics {
        step: o.get("step").and_then(|x| x.as_u64()).unwrap_or(0),
        loss: f("loss").unwrap_or(f64::NAN),
        samples_per_s: f("samples_per_s").unwrap_or(f64::NAN),
        tokens_per_s: f("tokens_per_s").unwrap_or(f64::NAN),
        step_time_ms: f("step_time_ms").unwrap_or(f64::NAN),
        peak_vram_gb: f("peak_vram_gb").unwrap_or(f64::NAN),
        updated_unix_s: crate::registry::now_s(),
        avg_gpu_util_pct: 0.0,
    })
}

/// Fold a freshly parsed sample into the job's running metrics. NaN means
/// "not reported in this line", so the previous value survives.
pub fn merge(dst: &mut TrainingMetrics, src: TrainingMetrics) {
    if src.step > 0 {
        dst.step = src.step;
    }
    let keep = |old: f64, new: f64| if new.is_nan() { old } else { new };
    dst.loss = keep(dst.loss, src.loss);
    dst.samples_per_s = keep(dst.samples_per_s, src.samples_per_s);
    dst.tokens_per_s = keep(dst.tokens_per_s, src.tokens_per_s);
    dst.step_time_ms = keep(dst.step_time_ms, src.step_time_ms);
    dst.peak_vram_gb = dst.peak_vram_gb.max(keep(dst.peak_vram_gb, src.peak_vram_gb));
    dst.updated_unix_s = src.updated_unix_s;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metric_with_rank_prefix() {
        let l = r#"[rank0]: FERRO_METRIC {"step": 12, "loss": 6.5, "samples_per_s": 64.5}"#;
        let m = parse_metric_line(l).unwrap();
        assert_eq!(m.step, 12);
        assert!((m.loss - 6.5).abs() < 1e-9);
        assert!((m.samples_per_s - 64.5).abs() < 1e-9);
        assert!(m.tokens_per_s.is_nan());
    }

    #[test]
    fn ignores_plain_lines() {
        assert!(parse_metric_line("epoch 1 done").is_none());
        assert!(parse_metric_line("FERRO_METRIC not-json").is_none());
    }

    #[test]
    fn merge_keeps_unreported_fields() {
        let mut dst = TrainingMetrics {
            step: 5,
            loss: 1.0,
            tokens_per_s: 900.0,
            ..Default::default()
        };
        let src = parse_metric_line(r#"FERRO_METRIC {"step": 6, "loss": 0.5}"#).unwrap();
        merge(&mut dst, src);
        assert_eq!(dst.step, 6);
        assert!((dst.loss - 0.5).abs() < 1e-9);
        // tokens_per_s was not in the new line, so it must be preserved.
        assert!((dst.tokens_per_s - 900.0).abs() < 1e-9);
    }

    #[test]
    fn peak_vram_is_monotonic() {
        let mut dst = TrainingMetrics { peak_vram_gb: 10.0, ..Default::default() };
        merge(&mut dst, parse_metric_line(r#"FERRO_METRIC {"peak_vram_gb": 4.0}"#).unwrap());
        assert!((dst.peak_vram_gb - 10.0).abs() < 1e-9);
        merge(&mut dst, parse_metric_line(r#"FERRO_METRIC {"peak_vram_gb": 18.0}"#).unwrap());
        assert!((dst.peak_vram_gb - 18.0).abs() < 1e-9);
    }

    #[test]
    fn detects_nccl_failures() {
        assert!(is_nccl_error("[rank1]: NCCL WARN Connect to 10.0.0.2 failed"));
        assert!(is_nccl_error("torch.distributed.DistBackendError: NCCL error"));
        assert!(is_nccl_error("[rank0]: ncclSystemError: System call failed"));
        assert!(!is_nccl_error("training step 3 loss 1.2"));
    }

    #[test]
    fn routine_warnings_are_not_reported_as_errors() {
        // These are emitted by healthy runs; treating them as failures made
        // every successful job look like it had NCCL problems.
        assert!(!is_nccl_error(
            "[rank0]:[W823 15:22:27.17] ProcessGroupNCCL.cpp:5072] Guessing device ID \
             based on global rank."
        ));
        assert!(!is_nccl_error(
            "[rank0]:[W823] ProcessGroupNCCL.cpp:1524] Warning: destroy_process_group() \
             was not called before program exit"
        ));
    }
}
