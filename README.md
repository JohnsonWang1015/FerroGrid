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

## Quick start

```bash
git clone <this repo> && cd FerroGrid
uv run --all-extras ferro-setup
```

That single command creates the virtualenv, installs PyTorch and Mojo/MAX,
builds the Rust binaries, and links `ferro` / `ferro-agent` /
`ferro-controller` into `~/.local/bin`. Re-run it any time; it is idempotent.

Lighter variants:

```bash
uv run ferro-setup                    # control plane only, no torch/Mojo
uv sync --extra mojo                  # Python side only, with Mojo/MAX
uv run --all-extras ferro-setup --portable   # binaries for older servers
```

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain
(`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`).

## Components

| Path | What it is |
|---|---|
| `crates/ferro-proto` | gRPC contract (`proto/ferrogrid.proto`) and generated code |
| `crates/ferro-gpu` | NVML wrapper: model, VRAM, utilisation, temperature, power |
| `crates/ferro-agent` | Per-server agent: reports GPUs, launches/supervises torchrun |
| `crates/ferro-controller` | Registry, GPU scheduler, job orchestration, log/metric collection |
| `crates/ferro-cli` | The `ferro` command |
| `python/examples/train_fsdp2.py` | FSDP2 reference model (synthetic data) |
| `python/ferro_mojo.py` | Mojo kernel loader, autograd wrapper, PyTorch fallback |
| `python/examples/bench_mojo.py` | Mojo-vs-PyTorch correctness check and benchmark |
| `mojo/kernels/` | Mojo custom kernels (`ferro_gelu`), compiled via MAX |
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

### 2. Build and install

```bash
uv run --all-extras ferro-setup
```

This builds the binaries and links them into `~/.local/bin` (see *Quick
start*). If `ferro --version` then prints `command not found`, `~/.local/bin`
is not on your `PATH` -- add `export PATH="$HOME/.local/bin:$PATH"` to your
shell rc file.

**Deploying to servers older than the controller host** needs the portable
build instead, which compiles inside a glibc-2.31 container so one binary runs
on Ubuntu 20.04 and newer:

```bash
uv run --all-extras ferro-setup --portable   # or: ./scripts/build.sh portable
```

> **Why not a static musl binary?** NVML is `dlopen`ed at runtime, and a
> statically linked musl binary has no dynamic loader — the agent starts fine
> but reports **zero GPUs**. The agent must be dynamically linked, so it is
> built against the oldest glibc in the fleet instead.

### 3. Start the controller

```bash
./target/portable/release/ferro-controller --bind 0.0.0.0:7070
```

Useful flags: `--master-port` (rendezvous port, default 29500),
`--min-free-vram-gib` (default 8, see *Scheduling* below),
`--default-image`.

For a permanent setup, run it under systemd the same way the agents do.

### 4. Register the agents

```bash
./scripts/register_node.sh gpu-a 10.0.0.1:7070
./scripts/register_node.sh gpu-b 10.0.0.1:7070
```

One command per node: it checks prerequisites, pre-pulls the training image,
installs the agent as a `systemd --user` service, and then waits for the node
to actually appear in `ferro nodes` rather than just reporting that a process
started.

**Password-only SSH is fine.** Pass `user@host` directly — no entry in
`~/.ssh/config` and no key required:

```bash
./scripts/register_node.sh johnson@10.0.0.12 10.0.0.1:7070 rtx5090
```

You are prompted for the password **once**. The script opens a single
authenticated SSH connection and multiplexes every later `scp`/`ssh` over it
through a control socket, which is closed when the run finishes. The password
is typed into `ssh` itself: nothing is stored, and nothing is placed on a
command line or in an environment variable where `ps` could read it. That is
deliberately not `sshpass`.

To stop typing it altogether, install your key on the first run:

```bash
./scripts/register_node.sh --copy-id johnson@10.0.0.12 10.0.0.1:7070 rtx5090
```

After that, redeploys and `ferro sync` need no password at all.

Other options: `--no-image` skips the image pull; a third positional argument
sets the node id, which is worth doing when a machine's hostname is unhelpful
(`user`, `gpu`) — it is what `ferro nodes` shows and what `--node` matches.

`scripts/deploy_agent.sh` is the lower-level version if you only want to push
a new binary to a node that is already registered:

```bash
./scripts/deploy_agent.sh gpu-a 10.0.0.1:7070
```

Each agent is installed to `~/.local/bin/ferro-agent` and enabled as a user
service with `Restart=always` and lingering turned on, so it survives logout
and reboots. Optional 3rd/4th arguments override the NCCL IP and the node id:

```bash
./scripts/deploy_agent.sh gpu-a 10.0.0.1:7070 192.168.50.11 gpu-a
```

Set the node id when a machine has an unhelpful hostname — it is what
`ferro nodes` shows and what `--node` filters match.

### 5. Ship the training code

```bash
./scripts/sync_workspace.sh gpu-a gpu-b
```

Agents resolve **relative** script paths against their own workspace root
(`~/ferrogrid`, override with `--workspace`), so the nodes do not need
identical home directories. Absolute paths are passed through untouched for
shared-NFS setups.

### 6. Verify

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
ferro watch                     # live dashboard: GPUs + running jobs, one screen
ferro train --nodes 2 --gpus-per-node 2 train.py
ferro jobs                      # recent jobs
ferro job <job-id>              # placement, per-rank status, metrics, NCCL errors
ferro logs <job-id> [-f]        # merged logs, tagged by rank and node
ferro cancel <job-id>           # stop every rank and free the GPUs
```

Add `--json` to any command for scripting.

### Live monitoring

`ferro watch` is the cluster-wide equivalent of `watch -n 1 nvidia-smi`: every
GPU on every node with utilisation and VRAM bars, which job owns each card, and
a row per running job with its live step, loss, throughput and NCCL error count.

```bash
ferro watch            # refresh every 2s
ferro watch -n 1       # every second
```

`nodes`, `gpu`, `jobs` and `job` each take the same `-w/--watch` and
`-n/--interval` flags if you want just one of those views:

```bash
ferro gpu -w -n 1
ferro job <job-id> -w
```

**The refresh rate is not the data rate.** GPU counters reach the controller
on the agents' heartbeat (controller `--heartbeat-secs`, default 3), so
`-n 1` redraws every second over data that changes every three. Both views
show how stale the numbers are — an `AGE` column in `ferro nodes`, a
`data age` figure in `ferro watch` — so a wedged agent reads as stale rather
than as an idle GPU. Start the controller with `--heartbeat-secs 1` if you
want the dashboard to genuinely track second by second.

### Running your own project

Agents resolve a relative script path against **their own** workspace root
(`~/ferrogrid`), so the code has to be on the nodes before a job can run.
`--sync` does that in the same command:

```bash
cd ~/my-experiment
ferro train --nodes 1 --gpus-per-node 2 --sync -f my_train.py
```

That rsyncs the current directory to every target node's workspace, then
launches `my_train.py` from there. Sync separately when you would rather not
re-copy on every run:

```bash
ferro sync                      # current directory -> every healthy node
ferro sync --node gpu-a         # just one node
ferro sync --delete             # also remove files deleted locally
ferro sync --dry-run            # show the rsync commands only
```

`ferro sync` needs no host list: the nodes report their own login user and
workspace root when they register, which is also why nodes with different
home directories need no special handling.

Build artefacts, virtualenvs, caches and common weight/volume file types are
excluded automatically — **your dataset should not be synced**, put it on
shared storage and `--mount` it instead.

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
  MASTER_ADDR=10.0.0.11  MASTER_PORT=29500  WORLD_SIZE=4
  NODE_RANK=0  node=gpu-a  gpus=[0,1]
  NODE_RANK=1  node=gpu-b  gpus=[0,1]
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
cargo test --workspace                     # scheduler + metric-parser unit tests
uv run --all-extras pytest -q              # Mojo fallback + gradient correctness
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

## Choosing a parallelism strategy

More GPUs is not automatically faster here, and on this fabric it is often
much slower. Pick by asking **why** you need them:

| Situation | Use | Why |
|---|---|---|
| Model fits on one GPU | `--nodes 1 --gpus-per-node 1` | No communication at all |
| Want a bigger batch, model still fits | `--nodes 1 --gpus-per-node 2` | Gradient sync stays on PCIe |
| Model does not fit on one GPU | `--nodes 1 --gpus-per-node 2` with FSDP2 | Sharding cost stays inside one box |
| Model does not fit on one *node* | `--nodes 2 ...` | Only now is the network worth paying for |
| Many independent runs (CV folds, sweeps) | one 1-GPU job per fold | Perfectly parallel, zero communication |

On 1 GbE, sharding one model across two nodes costs ~55× throughput (see
*Measured results*). The cluster's real value for most lab work is the last
row: run five folds as five jobs, not one job on five GPUs.

Check before you assume you need sharding:

```bash
ferro train --nodes 1 --gpus-per-node 1 -f your_train.py --max-steps 5
ferro job <id>        # look at PEAK VRAM GB against the card's capacity
```

### Worked example: how big an LLM actually fits

Yes, FSDP2 shards one model across GPUs and FerroGrid drives it. What that
buys you on this hardware is worth knowing before you plan a run. All measured
on one node with 2x RTX 4090 (24 GiB each, **no NVLink**), batch 1, seq 512,
AdamW, bf16 all-gather with fp32 reduction:

| Params | Setup | Peak VRAM/rank | tokens/s |
|---|---|---|---|
| 1.34B | 1 GPU | 20.64 GiB | **3,316** |
| 1.34B | 2 GPU, FSDP2 | 10.66 GiB | 1,013 |
| 2.68B | 1 GPU | **OOM** | — |
| 2.68B | 2 GPU, FSDP2 | **OOM** | — |
| 2.68B | 2 GPU, FSDP2 + `--offload` | 2.63 GiB | 188 |

Three things fall out of this:

**Sharding halves memory and costs 3.3x throughput.** FSDP2 did exactly what
it promises — 20.64 → 10.66 GiB — but consumer RTX 4090s have no NVLink and
NVIDIA disables peer-to-peer over PCIe, so every all-gather round-trips
through host memory. Sharding a model that already fits is a bad trade; use
the second GPU for a bigger batch, or for a second experiment.

**The full-finetune ceiling on 2x24 GiB is roughly 1.5B parameters.** AdamW
needs about 16 bytes per parameter (fp32 weights, gradients, and two moments)
and sharding divides that but does not remove it. 2.68B needs ~43 GiB of
optimiser state alone, which is why it OOMs even sharded across 48 GiB.

**CPU offload changes what is possible, not what is fast.** `--offload` keeps
the sharded state in host RAM and pulls in one block at a time: 2.68B drops to
2.63 GiB per rank and trains, at 188 tokens/s. That is a checkpoint-recovery
or a proof-of-concept, not a training run.

To train larger models here, in order of how much they buy:

- **LoRA / QLoRA** — trains adapters instead of weights, removing almost all
  of the optimiser state. This is the realistic path to 7B+ on these cards.
- **8-bit optimiser** (`bitsandbytes`) — cuts the two moments from 8 bytes per
  parameter to 2, roughly doubling the ceiling for a small quality cost.
- **Activation checkpointing** — helps activations, not optimiser state, so it
  raises the batch size you can use rather than the model size you can hold.
- **More GPUs** — 4 GPUs is 96 GiB and ~4B parameters, but only within one
  node. Across nodes at 1 GbE the throughput loss (see *Measured results*)
  makes it impractical for anything but a correctness check.

Sharded checkpoints need `torch.distributed.checkpoint`; a plain
`torch.save` of a sharded model saves one rank's shard, not the model.

### Worked example: 3D MRI (CNN + Video-Swin)

`python/examples/train_mri_3d.py` is a template for volumetric classification:
a strided 3D conv stem, transformer blocks, activation checkpointing, bf16,
and FSDP2 wrapping applied per block. Swap in your own `build_model` and
`build_dataset`; everything else is ready.

Measured on one RTX 3090, 95M parameters, activation checkpointing on:

| Volume | Batch/GPU | Peak VRAM | Step time |
|---|---|---|---|
| 128³ | 2 | 2.51 GiB | 257 ms |
| 160×192×160 | 1 | 2.68 GiB | 408 ms |

Under 3 GiB on a 24 GiB card — so this **fits on a single GPU with room for a
much larger batch**, and multi-node FSDP would be a pure loss. In 3D imaging
the memory pressure is *activations*, not parameters, which is why the conv
stem's stride and activation checkpointing matter far more than sharding.

Running it:

```bash
# 1. Dataset on storage every node can see, at the same path.
#    Bind-mount it explicitly -- only the workspace is mounted by default.
ferro train --nodes 1 --gpus-per-node 2 --follow \
    --mount /mnt/adni_data:/mnt/adni_data:ro \
    --mount /mnt/adni_work \
    --image ferrogrid/train:mri \
    python/examples/train_mri_3d.py \
        --data-root /mnt/adni_data --out-dir /mnt/adni_work/runs/exp1 \
        --volume 128 128 128 --batch-size 4 --accum 4
```

Build the image with the medical-imaging dependencies uncommented in
`docker/requirements.txt`:

```bash
docker build -f docker/Dockerfile.train -t ferrogrid/train:mri .
```

#### Watch the data path, not just the GPUs

For imaging the bottleneck usually moves off the GPU. From the numbers above,
a 128³ float32 volume is 8.4 MB and a step takes 257 ms at batch 2 — about
**65 MB/s per GPU**. Two GPUs already want ~130 MB/s, which exceeds what a
1 GbE link to an NFS server can deliver (~125 MB/s), and it gets worse with
every GPU you add.

So a job can be perfectly sized for the GPUs and still crawl, with `ferro
watch` showing utilisation flat at 20% while the cards sit waiting on the
network. Check there first when throughput disappoints.

Fixes, most effective first:

- **Preprocess once, cache on each node's local NVMe.** Resample and convert
  the raw DICOM/NIfTI to `.npy`/`.pt` ahead of time; the result is far smaller
  than the originals and reads at NVMe speed instead of network speed.
- **Cache in RAM** with MONAI's `CacheDataset` — these machines have 60 GB+,
  so after the first epoch the network is out of the loop entirely.
- **Store volumes as float16.** Halves the bytes on the wire, and the data is
  cast to bf16 for the forward pass anyway.

Cross-validation as parallel jobs rather than one distributed job:

```bash
for fold in 0 1 2 3 4; do
    ferro train --nodes 1 --gpus-per-node 1 --name "cv-fold$fold" \
        --mount /mnt/adni_data:/mnt/adni_data:ro --mount /mnt/adni_work \
        --image ferrogrid/train:mri \
        python/examples/train_mri_3d.py --data-root /mnt/adni_data --fold $fold
done
ferro jobs
```

The scheduler spreads them over whatever GPUs are free and refuses the ones it
cannot place, so you can queue more folds than you have cards and re-run the
rejected ones.

### Mounting data

Only the agent's workspace is mounted into the container by default. Datasets
and checkpoint directories need `--mount`:

| Form | Result |
|---|---|
| `--mount /mnt/data` | mounted at `/mnt/data` inside the container |
| `--mount /host/path:/container/path` | mounted at a different path |
| `--mount /mnt/data:/mnt/data:ro` | read-only, which is what you want for a dataset |

Mount at the **same path** wherever you can: then a path in a config file
means the same thing on the host and inside the container.

Jobs run as **your uid**, not root. A dataset mounted `:ro` is fine either
way, but an output directory has to be writable by you — a fresh NFS export is
usually `root:root 755`, which silently is not. Check before a long run:

```bash
ferro train --nodes 1 --gpus-per-node 1 -f \
    --mount /mnt/adni_data:/mnt/adni_data:ro --mount /mnt/adni_work \
    python/examples/check_mounts.py /mnt/adni_data /mnt/adni_work
```

It reports each path's filesystem type, ownership, and whether your uid can
write to it.

### Network storage: NFS and Samba/CIFS

FerroGrid needs no special support for either. `--mount` bind-mounts a host
path, and the container does not care what backs it — ext4, NFS or CIFS all
behave the same once mounted on the host. Verified end to end with a CIFS
share bind-mounted into a job container running as a non-root uid.

What does need care is the **host-side mount**, because NFS and CIFS both fix
ownership at mount time rather than honouring the on-disk owner:

- **CIFS** takes `uid=`/`gid=` as mount options. Set them to the account the
  agent runs as, or your jobs cannot write to the share at all.
- **NFS** maps by uid, and a fresh export is typically `root:root 755` — the
  dataset reads fine, but an `--out-dir` on it will fail. Fix ownership on the
  server, not the client.

`scripts/mount_smb.sh` sets a Samba share up correctly on a node — installs
`cifs-utils`, writes a root-owned `0600` credentials file (the password never
reaches a command line), matches uid/gid to the agent's account, adds an
`_netdev,nofail` fstab entry so it survives reboot, and write-tests the result:

```bash
./scripts/mount_smb.sh gpu-a //fileserver/mri /mnt/mri labuser
ferro train ... --mount /mnt/mri:/mnt/mri:ro ...
```

It needs sudo on the target and will prompt for it.

## Mojo / MAX

Mojo and MAX are supported and working: `mojo/kernels/gelu.mojo` compiles to a
MAX custom op that PyTorch calls on CPU and GPU, with gradients verified by
`torch.autograd.gradcheck`.

```bash
uv sync --all-extras
uv run ferro-mojo-info                              # what loaded, and why
uv run --all-extras python python/examples/bench_mojo.py
ferro train --nodes 1 --gpus-per-node 1 -f python/examples/train_fsdp2.py \
    --activation mojo
```

Kernels are always optional. `ferro_mojo.gelu()` falls back to
`torch.nn.functional.gelu` when MAX is absent, the kernel fails to compile, or
the input is unsupported, so the same script runs unchanged on a node with no
Mojo toolchain. Pass `strict=True` when you need it to fail loudly instead.

### The honest performance picture

On an RTX 3090 the Mojo custom op is currently **slower** than PyTorch's fused
GELU — 6.6× slower at 33M elements, and 43× slower at 4K where a fixed ~180 µs
bridge cost dominates. End to end: 100,175 tok/s with PyTorch vs 85,008 tok/s
with Mojo. `--activation torch` is therefore the default.

This is a statement about *replacing one already-optimal PyTorch op*, not
about Mojo. Custom kernels pay off when they fuse several ops into one bridge
crossing, or implement something PyTorch has no fused kernel for. See
`mojo/README.md` for the full measurement table, the Mojo 1.0 syntax notes,
and how to add a kernel.

## Scope and limitations

Phase 1 is deliberately small. Known gaps, in rough priority order:

- Controller state is in memory: node registrations self-heal after a restart,
  job history does not.
- No queueing — a job that cannot be placed is rejected rather than held.
- No authentication or TLS on the gRPC endpoints; run it on a trusted network.
- No multi-tenant fair sharing, quotas or preemption.
- Elastic/fault-tolerant training is not wired up; a rank failure fails the job.
- `--gpus-per-node` is uniform across nodes, as torchrun expects.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2026 Wang Chia Wei.

FerroGrid orchestrates PyTorch, NCCL and Mojo/MAX rather than vendoring them;
those remain under their own licences and are not redistributed here.
