# FerroGrid — working notes

Rust control plane + stock PyTorch FSDP2/NCCL for multi-server GPU training.
See README.md for architecture, deployment and measured results.

## Build

- `uv run --all-extras ferro-setup` does everything from scratch (venv, torch,
  Mojo/MAX, cargo build, PATH links). Idempotent.
- `cargo build --workspace` / `cargo test --workspace` for local work.
- `uv run --all-extras pytest -q` for the Python/Mojo side.
- `./scripts/build.sh portable` before deploying: builds in a glibc-2.31
  container so the binaries run on the older Ubuntu releases in the lab.
- The agent must stay **dynamically linked** — NVML is `dlopen`ed, so a static
  musl build silently reports zero GPUs.

## Hard-won details worth not rediscovering

- `--gpus` is CSV-parsed by docker: the value must be `"device=0,1"` *with*
  the quotes, or docker reads the `1` as a device count and errors.
- NCCL must be pinned to the LAN interface (`NCCL_SOCKET_IFNAME`); left alone
  it binds a docker bridge and fails between nodes. The agent auto-detects it.
- `systemctl --user enable --now` does not restart a running unit; redeploys
  need `restart`.
- Do not match a bare `ProcessGroupNCCL` when classifying NCCL errors — torch
  logs routine warnings with that string.
- `uv run` does not source your shell profile, so `~/.cargo/bin` is usually
  absent from PATH there; `ferro_setup.find_cargo()` looks it up explicitly.
- MAX custom ops **reject tensors that require grad**. Calling one straight
  from a model raises, and a bare try/except fallback turns a "Mojo" run into
  a silent PyTorch run. `ferro_mojo` wraps the op in an `autograd.Function`;
  `used_backend()` and `strict=True` exist so tests can prove which path ran.
- `CustomOpLibrary` wants a directory containing `__init__.mojo`, not a
  `.mojo` file, and it must be a `Path`, not a `str`.
- Mojo 1.0 removed `fn` (use `def`) and deprecated `alias` (use `comptime`).
  Kernel API is in `extensibility`; stdlib imports are `std.`-prefixed.

## Conventions

- Training scripts report metrics by printing `FERRO_METRIC {json}` on stdout.
- Agents resolve relative script paths against their own `--workspace`
  (default `~/ferrogrid`), because lab nodes have different home directories.
- Never reimplement FSDP/NCCL/torchrun; the platform only wraps them.
- Mojo kernels stay optional and must always have a working PyTorch fallback.
  Measure before adopting one: as of MAX 26.5 the custom-op bridge is slower
  than PyTorch's fused elementwise kernels (see mojo/README.md).
