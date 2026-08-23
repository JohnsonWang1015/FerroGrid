# FerroGrid

A small multi-server GPU training platform for a lab: a Rust control plane that
discovers every GPU in the cluster, picks devices for a job, and launches
**stock PyTorch FSDP2 + NCCL** through `torchrun` inside Docker on each node.

FerroGrid does **not** reimplement FSDP, NCCL or torchrun. Its job is the part
around them: knowing what hardware exists, choosing where a job runs, setting
`MASTER_ADDR` / `NODE_RANK` / `WORLD_SIZE` correctly, starting the containers,
and collecting logs, GPU telemetry, throughput and NCCL errors in one place.

```
                    ferro (CLI)
                         │ gRPC
                 ┌───────▼────────┐
                 │ ferro-controller│  registry · scheduler · job state · logs
                 └───┬────────┬───┘
              gRPC   │        │   gRPC
              ┌──────▼──┐  ┌──▼──────┐
              │ferro-   │  │ferro-   │   NVML telemetry · docker run · torchrun
              │agent    │  │agent    │
              └────┬────┘  └────┬────┘
                   │            │
              docker run   docker run     ← NCCL over the LAN between them
              torchrun     torchrun
              rank 0..n    rank n..m
```

## Components

| Path | What it is |
|---|---|
| `crates/ferro-proto` | gRPC contract (`proto/ferrogrid.proto`) and generated code |
| `crates/ferro-gpu` | NVML wrapper: model, VRAM, utilisation, temperature, power |
| `crates/ferro-agent` | Per-server agent: reports GPUs, launches/supervises torchrun |
| `crates/ferro-controller` | Registry, GPU scheduler, job orchestration, log/metric collection |
| `crates/ferro-cli` | The `ferro` command |
| `python/examples/train_fsdp2.py` | FSDP2 reference model (synthetic data) |
| `python/ferro_mojo.py` | Mojo kernel loader with a PyTorch fallback |
| `mojo/` | Phase-2 custom-kernel interface and benchmark harness (no kernels yet) |
| `docker/Dockerfile.train` | Optional custom training image |
| `scripts/` | Build, deploy, sync, prepare, benchmark |

## Requirements

**Controller host** (can be any machine, including one of the GPU servers):
Rust 1.80+, `protobuf-compiler`, Docker (only for `build.sh portable`).

**Each GPU server** (Ubuntu 20.04 / 22.04 / 24.04):

- NVIDIA driver **≥ 525** (CUDA 12.x minor-version compatibility)
- Docker, plus the NVIDIA Container Toolkit (`docker info | grep nvidia`)
- The login user in the `docker` group
- SSH access from the controller host
- **No root required** — the agent runs as a `systemd --user` service

---

## Deploying on two Ubuntu servers

The example below uses two servers, `gpu-a` and `gpu-b`, with the controller on
a third machine at `10.0.0.1`. Put `gpu-a` / `gpu-b` in your `~/.ssh/config`.

### 1. Check the servers

```bash
./scripts/prepare_node.sh gpu-a
./scripts/prepare_node.sh gpu-b
```

This verifies `nvidia-smi`, Docker, the NVIDIA runtime and docker-group
membership, pre-pulls the training image, and confirms that CUDA and FSDP2
work **inside a container** — the failure mode you want to find now, not
during your first job.

### 2. Build

```bash
./scripts/build.sh portable
```

`portable` builds inside a glibc-2.31 container so one binary runs on Ubuntu
20.04 and newer. Use plain `./scripts/build.sh` if the controller and the GPU
servers run the same distro.

> **Why not a static musl binary?** NVML is `dlopen`ed at runtime, and a
> statically linked musl binary has no dynamic loader — the agent starts fine
> but reports **zero GPUs**. The agent must be dynamically linked, so it is
> built against the oldest glibc in the fleet instead.

### 3. Put `ferro` on your PATH

`build.sh` leaves the binaries under `target/`; nothing installs them for you.
Symlink rather than copy, so a later rebuild is picked up automatically:

```bash
mkdir -p ~/.local/bin
ln -sf "$PWD/target/release/ferro"            ~/.local/bin/ferro
ln -sf "$PWD/target/release/ferro-controller" ~/.local/bin/ferro-controller
ferro --version
```

If that prints `command not found`, `~/.local/bin` is not on your `PATH` --
add `export PATH="$HOME/.local/bin:$PATH"` to your shell rc file.

### 4. Start the controller

```bash
./target/portable/release/ferro-controller --bind 0.0.0.0:7070
```

Useful flags: `--master-port` (rendezvous port, default 29500),
`--min-free-vram-gib` (default 8, see *Scheduling* below),
`--default-image`.

For a permanent setup, run it under systemd the same way the agents do.

### 5. Deploy the agents

```bash
./scripts/deploy_agent.sh gpu-a 10.0.0.1:7070
./scripts/deploy_agent.sh gpu-b 10.0.0.1:7070
```

Each agent is installed to `~/.local/bin/ferro-agent` and enabled as a user
service with `Restart=always` and lingering turned on, so it survives logout
and reboots. Optional 3rd/4th arguments override the NCCL IP and the node id:

```bash
./scripts/deploy_agent.sh gpu-a 10.0.0.1:7070 192.168.50.11 gpu-a
```

Set the node id when a machine has an unhelpful hostname — it is what
`ferro nodes` shows and what `--node` filters match.

### 6. Ship the training code

```bash
./scripts/sync_workspace.sh gpu-a gpu-b
```

Agents resolve **relative** script paths against their own workspace root
(`~/ferrogrid`, override with `--workspace`), so the nodes do not need
identical home directories. Absolute paths are passed through untouched for
shared-NFS setups.

### 7. Verify

```bash
export FERRO_CONTROLLER=http://10.0.0.1:7070
ferro nodes
ferro gpu
```

---

## Using the CLI

```bash
ferro nodes                     # servers, health, driver, free GPU count
ferro gpu                       # every GPU: VRAM, utilisation, temp, power, owning job
ferro train --nodes 2 --gpus-per-node 2 train.py
ferro jobs                      # recent jobs
ferro job <job-id>              # placement, per-rank status, metrics, NCCL errors
ferro logs <job-id> [-f]        # merged logs, tagged by rank and node
ferro cancel <job-id>           # stop every rank and free the GPUs
```

Add `--json` to any command for scripting.

### `ferro train`

```bash
ferro train --nodes 2 --gpus-per-node 2 --follow \
    python/examples/train_fsdp2.py --steps 100 --layers 12
```

Flags **before** the script path belong to `ferro`; everything **after** it is
forwarded verbatim to the script.

| Flag | Meaning |
|---|---|
| `--nodes N` | number of servers |
| `--gpus-per-node K` | GPUs per server → `torchrun --nproc_per_node` |
| `--follow` / `-f` | stream logs, exit non-zero if the job fails |
| `--image` | override the Docker image |
| `--node ID` | restrict placement (repeatable) |
| `--env K=V` | extra environment, e.g. `--env NCCL_DEBUG=INFO` (repeatable) |
| `--workdir` | working directory, relative to the agent workspace |
| `--name` | label shown in `ferro jobs` |

The controller computes the placement and prints it before launching:

```
Submitted j52c9839cf7
  MASTER_ADDR=140.123.105.18  MASTER_PORT=29500  WORLD_SIZE=4
  NODE_RANK=0  node=lab18  gpus=[0,1]
  NODE_RANK=1  node=ccu2   gpus=[0,1]
```

---

## How it works

**Discovery.** Each agent opens NVML once, then reports live counters on every
heartbeat (default 3s). A node with no usable NVML still registers, and
`ferro nodes` shows why it has no GPUs instead of the agent crashing. Agents
re-register automatically after a controller restart.

**Scheduling.** First-fit over healthy nodes, preferring the most free VRAM,
ties broken by node id so placement is reproducible. Rank 0 goes to the first
chosen node and its NCCL IP becomes `MASTER_ADDR`.

A GPU counts as free only when *both* no FerroGrid job holds it **and** it has
at least `--min-free-vram-gib` (default 8) actually free. FerroGrid shares
these machines with workloads it does not manage; without the VRAM check it
would happily schedule onto a card with 0.5 GiB left and OOM immediately.

GPUs are reserved at submit time, so two back-to-back submissions cannot be
handed the same devices, and each agent re-validates the placement locally
before launching.

**Launching.** The agent runs, per node:

```
docker run --rm --network host --ipc host --shm-size 8g \
  --gpus '"device=0,1"' --user $(id -u):$(id -g) \
  -v <workdir>:<workdir> -w <workdir> \
  -e MASTER_ADDR=... -e MASTER_PORT=... -e WORLD_SIZE=... -e NODE_RANK=... \
  -e NCCL_SOCKET_IFNAME=<lan-if> -e GLOO_SOCKET_IFNAME=<lan-if> \
  -e NCCL_IB_DISABLE=1 \
  <image> torchrun --nnodes=N --nproc_per_node=K --node_rank=R \
                   --master_addr=... --master_port=... <script> <args...>
```

`--network host` keeps the rendezvous and NCCL ports reachable between nodes.
`--user` keeps checkpoints written to the bind mount owned by you, not root.

**Metrics.** The training script prints one line per interval:

```python
print('FERRO_METRIC ' + json.dumps({"step": 10, "loss": 6.9, "tokens_per_s": 1234}))
```

The controller parses those lines out of stdout (the rank prefix does not
matter) and folds them into the job summary. Recognised keys: `step`, `loss`,
`samples_per_s`, `tokens_per_s`, `step_time_ms`, `peak_vram_gb`. Anything a
line omits keeps its previous value; `peak_vram_gb` only ever increases. GPU
utilisation is averaged from NVML heartbeats over the GPUs the job holds.

**NCCL errors.** Log lines matching known failure signatures (`NCCL WARN`,
`ncclSystemError`, `DistBackendError`, watchdog timeouts, …) are collected and
shown by `ferro job`. Routine `[W...] ProcessGroupNCCL.cpp` warnings are
deliberately not matched — they appear in healthy runs and would bury real
failures.

---

## Measured results

Real hardware, 1 GbE between nodes. Model: 8 layers, `d_model` 1024, seq 512,
per-GPU batch 8, ~166M parameters, FSDP2 with bf16 all-gather and fp32
gradient reduction.

| Shape | GPUs | tokens/s | step time | peak VRAM/rank |
|---|---|---|---|---|
| 1 node × 1 GPU (RTX 4090) | 1 | **85,065** | 48 ms | 4.69 GiB |
| 1 node × 2 GPU (RTX 4090, PCIe) | 2 | 63,724 | 129 ms | 3.76 GiB |
| 2 nodes × 1 GPU (1 GbE) | 2 | 1,522 | 5,384 ms | 3.76 GiB |
| 2 nodes × 1 GPU (1 GbE, both idle) | 2 | 1,583 | 5,174 ms | 3.76 GiB |

The 4-GPU target configuration (2 servers × 2 GPUs, `world_size=4`,
heterogeneous RTX 4090 / RTX PRO 5000 Blackwell / A6000) **runs FSDP2 to
completion with zero NCCL errors**, which is what phase 1 set out to prove.

### Read this before trusting the throughput numbers

**The 1 GbE interconnect, not the GPUs, is the limit.** FSDP2 all-gathers
parameters every forward and backward and reduce-scatters gradients — roughly
1.3 GB per step for this model. At ~125 MB/s that is seconds of communication
against ~50 ms of compute, which is exactly the 56× drop from one GPU to two
GPUs on separate nodes.

Consequences worth planning around:

- Multi-node FSDP2 on 1 GbE is for **correctness and capacity** (fitting a
  model that does not fit on one card), not for speed.
- For throughput, **10/25 GbE or InfiniBand** is the single highest-value
  upgrade. Nothing in FerroGrid changes; NCCL picks up the faster fabric.
- Sharding a *small* model hurts even inside one node (85k → 64k tokens/s
  above): FSDP2 pays off when the model does not fit, not when it does.
- The measured 4-GPU numbers were taken on a node whose GPUs were already
  saturated by other users' jobs, so they reflect contention, not the
  platform's ceiling. The two 2-node rows above were measured on different
  server pairs, one of them fully idle, and agree within 4% -- the inter-node
  cost is the fabric, not contention.

Reproduce with:

```bash
./scripts/benchmark.sh gpu-a gpu-b
```

---

## Testing

```bash
cargo test --workspace          # scheduler + metric-parser unit tests
```

End-to-end, after deploying two agents:

```bash
ferro nodes && ferro gpu                      # discovery

# single node, single GPU
ferro train --nodes 1 --gpus-per-node 1 -f python/examples/train_fsdp2.py --steps 10

# two nodes -- the real multi-node NCCL path
ferro train --nodes 2 --gpus-per-node 1 -f python/examples/train_fsdp2.py --steps 10

# the phase-1 target
ferro train --nodes 2 --gpus-per-node 2 -f python/examples/train_fsdp2.py --steps 20
```

`--follow` exits non-zero when a job fails, so these work in CI. Check
`ferro job <id>` afterwards for per-rank exit codes and the NCCL error list.

To verify cancellation and GPU release:

```bash
ferro train --nodes 2 --gpus-per-node 1 python/examples/train_fsdp2.py --steps 100000
ferro cancel <job-id>
ferro gpu        # the JOB column should be empty again
```

---

## Troubleshooting

**`ncclSystemError` / hang at `init_process_group`.** Almost always the wrong
network interface. GPU boxes are covered in `docker0` / `br-*` / calico
bridges, and NCCL will happily bind one the peer cannot reach. The agent
detects the interface holding its NCCL IP and pins `NCCL_SOCKET_IFNAME`
automatically; override with `FERRO_NCCL_IFNAME=<iface>` in the unit file if
the guess is wrong. Diagnose with `--env NCCL_DEBUG=INFO`.

**`cannot set both Count and DeviceIDs on device request`.** Docker CSV-parses
`--gpus`, so `device=0,1` splits into `device=0` plus a bare `1` read as a
count. The value must be quoted — the agent does this; the same applies if you
run `docker run` by hand.

**`ferro nodes` shows a node with 0 GPUs.** Look at the `gpu_error` line
printed under the table. A statically linked agent cannot `dlopen` NVML (see
*Build*); a driver/library version mismatch after an unattended upgrade needs
a reboot.

**Agent not registering.** `./scripts/logs_agent.sh gpu-a`. Check the
controller address, and that port 7070 is reachable from the server.

**Job dies instantly with exit 125.** That is `docker run` refusing to start —
the log line carries docker's own message. Common causes: image not pulled on
that node, or the user not in the `docker` group.

**Redeploying an agent seems to change nothing.** `deploy_agent.sh` restarts
the service; if you install by hand, remember `systemctl --user restart
ferro-agent` — `enable --now` will not restart an already-running unit.

**Throughput far below expectations on multi-node.** See *Measured results* —
on 1 GbE this is expected, not a bug.

---

## Mojo (phase 2)

The interface is defined, no kernels are written yet, and nothing on the
training path depends on Mojo:

- `mojo/kernels/kernel_api.mojo` — the contract every kernel implements
- `mojo/kernels/benchmark.mojo` — harness that emits `FERRO_METRIC` lines, so
  kernel benchmarks show up in `ferro job` like any training run
- `python/ferro_mojo.py` — loader that returns `None` when no Mojo toolchain or
  compiled kernel is present, so callers fall back to the PyTorch op

```bash
python python/ferro_mojo.py     # report toolchain and available kernels
```

See `mojo/README.md` for how to add a kernel.

## Scope and limitations

Phase 1 is deliberately small. Known gaps, in rough priority order:

- Controller state is in memory: node registrations self-heal after a restart,
  job history does not.
- No queueing — a job that cannot be placed is rejected rather than held.
- No authentication or TLS on the gRPC endpoints; run it on a trusted network.
- No multi-tenant fair sharing, quotas or preemption.
- Elastic/fault-tolerant training is not wired up; a rank failure fails the job.
- `--gpus-per-node` is uniform across nodes, as torchrun expects.
