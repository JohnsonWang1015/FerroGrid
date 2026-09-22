"""A GPU job that does nothing useful, for exactly as long as you ask.

The training examples are the point of the cluster, but they are the wrong
thing to demonstrate a *scheduler* with: a demo that spends thirty minutes
loading ADNI before anything is visible is a demo of a data loader. This burns
a configurable amount of GPU for a configurable time, reports metrics in the
usual format, and starts instantly.

    ferro train --auto python/examples/gpu_burn.py --seconds 120

It is deliberately compute-bound rather than a sleep: a sleeping job holds its
GPU allocation without showing up in `ferro gpu` utilisation, so a queue of
them would make the cluster look idle while being completely full -- which is
the one thing a scheduling demo must not do.

Use `--idle` when you *want* that: it holds the allocation without doing any
work, which is how to produce the "somebody is squatting on a card" case that
`ferro ps --idle` exists to find.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def emit(**fields: object) -> None:
    """One metric line, in the format the controller parses off stdout."""
    print(f"FERRO_METRIC {json.dumps(fields)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0, help="how long to run")
    ap.add_argument(
        "--size",
        type=int,
        default=4096,
        help="matrix side; the square of this sets how much VRAM and compute is used",
    )
    ap.add_argument(
        "--report-every",
        type=float,
        default=5.0,
        help="seconds between FERRO_METRIC lines",
    )
    ap.add_argument(
        "--idle",
        action="store_true",
        help="hold the GPU without computing, to simulate a squatting process",
    )
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed in this image", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        # Refuse rather than silently burning CPU: a GPU benchmark that ran on
        # the CPU would report numbers nobody could interpret.
        print("no CUDA device visible to this process", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    rank = torch.cuda.current_device()
    name = torch.cuda.get_device_name(rank)
    print(f"burning {name} (cuda:{rank}) for {args.seconds:.0f}s", flush=True)

    a = torch.randn(args.size, args.size, device=device, dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device=device, dtype=torch.bfloat16)

    # 2*n^3 floating-point operations per matmul.
    flop_per_step = 2.0 * args.size**3

    start = time.monotonic()
    last_report = start
    steps = 0

    while True:
        now = time.monotonic()
        if now - start >= args.seconds:
            break

        if args.idle:
            # Holding memory, doing nothing. Deliberately visible as 0%.
            time.sleep(0.2)
        else:
            a = (a @ b).to(torch.bfloat16)
            steps += 1

        now = time.monotonic()
        if now - last_report >= args.report_every:
            elapsed = now - start
            tflops = (steps * flop_per_step / elapsed / 1e12) if elapsed > 0 else 0.0
            emit(
                step=steps,
                loss=0.0,
                step_time_ms=(elapsed / steps * 1000.0) if steps else 0.0,
                samples_per_s=steps / elapsed if elapsed > 0 else 0.0,
                peak_vram_gb=torch.cuda.max_memory_allocated() / (1 << 30),
                tflops=tflops,
            )
            last_report = now

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    tflops = (steps * flop_per_step / elapsed / 1e12) if elapsed > 0 and steps else 0.0
    emit(
        step=steps,
        loss=0.0,
        step_time_ms=(elapsed / steps * 1000.0) if steps else 0.0,
        samples_per_s=steps / elapsed if elapsed > 0 else 0.0,
        peak_vram_gb=torch.cuda.max_memory_allocated() / (1 << 30),
        tflops=tflops,
    )
    print(
        f"done: {steps} matmuls in {elapsed:.1f}s ({tflops:.1f} TFLOP/s)",
        flush=True,
    )
    # Keep the result alive so the compiler cannot decide none of this was
    # needed.
    return 0 if a.numel() else 1


if __name__ == "__main__":
    raise SystemExit(main())
