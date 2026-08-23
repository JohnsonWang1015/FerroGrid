# FerroGrid Mojo kernels (interface only)

Phase 1 ships the **interface**, not the kernels. The training path is stock
PyTorch + FSDP2 + NCCL; nothing here is on that path yet.

What is defined now, so phase 2 can drop kernels in without reshaping anything:

- `kernels/kernel_api.mojo` — the signature every FerroGrid custom kernel implements.
- `kernels/benchmark.mojo` — the harness that will time a kernel against its
  PyTorch baseline and emit `FERRO_METRIC` lines, so Mojo benchmarks show up in
  `ferro job` exactly like training runs do.
- `python/ferro_mojo.py` — the Python side of the boundary. It loads a Mojo
  kernel when one is present and **falls back to the PyTorch op otherwise**, so
  the same training script runs with or without a Mojo toolchain installed.

## Why the fallback matters

`ferro train` must work on a node with no Mojo installed. `ferro_mojo.load()`
returns `None` in that case and callers use the torch path. Kernels are an
optimisation, never a dependency.

## Adding a kernel in phase 2

1. Implement `kernel_api.mojo`'s entry point in `kernels/<name>.mojo`.
2. Build a shared object: `mojo build --emit shared-lib kernels/<name>.mojo`.
3. Point `FERRO_MOJO_KERNEL_DIR` at the output directory.
4. `ferro_mojo.load("<name>")` picks it up; the benchmark harness compares it
   against the baseline and reports speedup.

Install the toolchain with `curl -fsSL https://pixi.sh/install.sh | sh` and
`pixi global install mojo` (see modular.com/mojo for current instructions).
