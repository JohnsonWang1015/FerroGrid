# FerroGrid — working notes

Rust control plane + stock PyTorch FSDP2/NCCL for multi-server GPU training.
See README.md for architecture, deployment and measured results.

## Build

- `cargo build --workspace` / `cargo test --workspace` for local work.
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

## Conventions

- Training scripts report metrics by printing `FERRO_METRIC {json}` on stdout.
- Agents resolve relative script paths against their own `--workspace`
  (default `~/ferrogrid`), because lab nodes have different home directories.
- Never reimplement FSDP/NCCL/torchrun; the platform only wraps them.
