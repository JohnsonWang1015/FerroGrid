#!/usr/bin/env python3
"""Benchmark the Mojo custom kernel against its PyTorch baseline.

Emits `FERRO_METRIC` lines, so running it through `ferro train` shows the
numbers in `ferro job` exactly like a training run:

    ferro train --nodes 1 --gpus-per-node 1 -f python/examples/bench_mojo.py

Standalone:

    uv run --all-extras python python/examples/bench_mojo.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ferro_mojo  # noqa: E402


def log(msg):
    print(msg, flush=True)


def metric(**kw):
    print("FERRO_METRIC " + json.dumps(kw), flush=True)


def timed(fn, x, iters, warmup, device):
    for _ in range(warmup):
        fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iters


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    args = p.parse_args()

    info = ferro_mojo.describe()
    log(f"Mojo kernels: {info['reason']}")
    if info["versions"]:
        log("versions: " + ", ".join(f"{k}={v}" for k, v in info["versions"].items()))
    if not info["available"]:
        log("Mojo unavailable -- nothing to compare against, exiting cleanly.")
        metric(step=0)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    log(f"device={device} dtype={args.dtype} shape=({args.rows}, {args.cols})")
    log("measuring the forward kernel only (no autograd tape)")

    x = torch.randn(args.rows, args.cols, device=device, dtype=dtype)

    torch_gelu = lambda t: torch.nn.functional.gelu(t, approximate="tanh")  # noqa: E731

    # Correctness first: a faster kernel that computes the wrong thing is worse
    # than no kernel at all.
    # strict=True: a silent fallback would make this a torch-vs-torch
    # benchmark reporting a meaningless 1.0x.
    mojo_gelu = lambda t: ferro_mojo.gelu(t, strict=True)  # noqa: E731
    got, want = mojo_gelu(x), torch_gelu(x)
    err = (got.float() - want.float()).abs().max().item()
    tol = 1e-5 if dtype is torch.float32 else 1e-2
    log(f"max|mojo - torch| = {err:.3e} (tolerance {tol:g}) "
        f"{'PASS' if err < tol else 'FAIL'}")
    if err >= tol:
        log("numerical mismatch -- refusing to report a speedup")
        return 1

    t_torch = timed(torch_gelu, x, args.iters, args.warmup, device)
    t_mojo = timed(mojo_gelu, x, args.iters, args.warmup, device)

    elems = args.rows * args.cols
    speedup = t_torch / t_mojo if t_mojo > 0 else float("nan")

    log("=" * 58)
    log(f"PyTorch GELU   {t_torch * 1e6:9.1f} us   {elems / t_torch / 1e9:7.2f} Gelem/s")
    log(f"Mojo GELU      {t_mojo * 1e6:9.1f} us   {elems / t_mojo / 1e9:7.2f} Gelem/s")
    log(f"speedup        {speedup:9.2f}x")
    log("=" * 58)

    metric(
        step=args.iters,
        step_time_ms=round(t_mojo * 1e3, 4),
        samples_per_s=round(1.0 / t_mojo, 1),
        tokens_per_s=round(elems / t_mojo, 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
