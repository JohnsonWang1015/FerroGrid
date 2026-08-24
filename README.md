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
| `python/examples/train_pp.py` | Pipeline-parallel example (works intra-node, see below) |
| `python/examples/p2p_probe.py` | Checks which NCCL primitives your fabric supports |
| `python/tools/preprocess_adni.py` | ADNI T1 DICOM zips → compact volumes + manifest |
| `mojo/kernels/` | Mojo custom kernels (`ferro_gelu`), compiled via MAX |
| `docker/Dockerfile.train` | Optional custom training image |
| `scripts/` | Build, deploy, sync, prepare, benchmark |

## Requirements

**Controller host** (can be any machine, including one of the GPU servers):

- [uv](https://docs.astral.sh/uv/) and a Rust toolchain — `ferro-setup` uses both
- `protobuf-compiler` (for the gRPC codegen)
- `rsync` and an SSH client (for registering nodes and `ferro sync`)
- Docker — only if you build portable binaries for older servers

**Each GPU server** (Ubuntu 20.04 / 22.04 / 24.04):

- NVIDIA driver **≥ 525** (CUDA 12.x minor-version compatibility)
- Docker, plus the NVIDIA Container Toolkit (`docker info | grep nvidia`)
- The login user in the `docker` group
- SSH access from the controller host
- **No root required** — the agent runs as a `systemd --user` service

---

## Deploying on two Ubuntu servers

Two GPU servers, `gpu-a` and `gpu-b`, with the controller on a third machine
at `10.0.0.1`. The controller can equally run on one of the GPU servers.

You need SSH access to each server and nothing else — no root, no
`~/.ssh/config` entry, not even an SSH key (see *Password-only SSH* below).

### 1. Build and install

```bash
uv run --all-extras ferro-setup
```

This builds the binaries and links them into `~/.local/bin` (see *Quick
start*). If `ferro --version` then prints `command not found`, `~/.local/bin`
is not on your `PATH` — add `export PATH="$HOME/.local/bin:$PATH"` to your
shell rc file.

**If the GPU servers are older than the controller host**, build portable
binaries instead, inside a glibc-2.31 container, so one binary runs on Ubuntu
20.04 and newer:

```bash
uv run --all-extras ferro-setup --portable   # or: ./scripts/build.sh portable
```

> **Why not a static musl binary?** NVML is `dlopen`ed at runtime, and a
> statically linked musl binary has no dynamic loader — the agent starts fine
> but reports **zero GPUs**. The agent must be dynamically linked, so it is
> built against the oldest glibc in the fleet instead.

### 2. Start the controller

```bash
ferro-controller --bind 0.0.0.0:7070
```

Useful flags: `--master-port` (rendezvous port, default 29500),
`--min-free-vram-gib` (default 8, see *Scheduling* below), `--default-image`,
and `--heartbeat-secs` (default 3; lower it for a snappier `ferro watch`).

Start it before registering nodes — registration confirms itself by waiting
for the node to appear in the controller's registry. For a permanent setup,
run it under systemd the same way the agents do.

### 3. Register the nodes

One command per server:

```bash
./scripts/register_node.sh gpu-a 10.0.0.1:7070
./scripts/register_node.sh gpu-b 10.0.0.1:7070
```

Each run:

1. checks `nvidia-smi`, Docker, the NVIDIA container runtime and docker-group
   membership — the failure modes you want to find now, not during your first
   training job;
2. pre-pulls the training image (`--no-image` to skip);
3. installs the agent to `~/.local/bin/ferro-agent` and enables it as a
   `systemd --user` service with `Restart=always` and lingering on, so it
   survives logout and reboot;
4. **waits for the node to appear in `ferro nodes`** and prints the table —
   so a green result means it really registered, not merely that a process
   started.

#### Password-only SSH

Pass `user@host` directly. No `~/.ssh/config` entry and no key needed:

```bash
./scripts/register_node.sh user@10.0.0.12 10.0.0.1:7070 rtx5090
```

You are prompted for the password **once**. The script opens a single
authenticated SSH connection and multiplexes every later `scp`/`ssh` over it
through a control socket, closed when the run finishes. The password goes into
`ssh` itself: nothing is stored, and nothing lands on a command line or in an
environment variable where `ps` could read it. That is deliberately not
`sshpass`.

To stop typing it, install your key on the first run:

```bash
./scripts/register_node.sh --copy-id user@10.0.0.12 10.0.0.1:7070 rtx5090
```

After that, re-registration and `ferro sync` need no password at all — worth
doing, because `ferro sync` uses SSH on every launch.

#### Naming a node

The third argument sets the node id, which is what `ferro nodes` shows and
what `--node` matches. It defaults to the machine's hostname, so set it
explicitly when that is unhelpful (`user`, `gpu`, `localhost`):

```bash
./scripts/register_node.sh user@10.0.0.12 10.0.0.1:7070 rtx5090
```

#### Re-running it

`register_node.sh` is idempotent — re-run it to upgrade a node after
`ferro-setup`, and it will restart the service and re-verify. Two narrower
scripts exist for when that is more than you want:

| Script | Use when |
|---|---|
| `deploy_agent.sh <host> <controller>` | pushing a new binary to a node that is already registered |
| `prepare_node.sh <host>` | checking a machine's prerequisites before you are ready to register it |

Both take an SSH alias or `user@host`, and `deploy_agent.sh` takes optional
NCCL-IP and node-id arguments for multi-homed machines:

```bash
./scripts/deploy_agent.sh gpu-a 10.0.0.1:7070 192.168.50.11 gpu-a
```

### 4. Ship the training code

```bash
export FERRO_CONTROLLER=http://10.0.0.1:7070
ferro sync
```

`ferro sync` copies the current directory to every registered node, using the
login user and workspace root each node reported at registration — so no host
list, and nodes with different home directories need no special handling. See
*Running your own project*.

### 5. Verify

```bash
ferro nodes
ferro gpu
ferro watch     # live dashboard

# end to end, on two nodes
ferro train --nodes 2 --gpus-per-node 1 -f python/examples/train_fsdp2.py --steps 10
```

---

## Using the CLI

```bash
ferro nodes                     # servers, health, driver, free GPU count
ferro gpu                       # every GPU: VRAM, utilisation, temp, power, owning job
ferro ps                        # what is running right now, per rank
ferro watch                     # live dashboard: GPUs + running jobs, one screen
ferro bench                     # measure each GPU, so the scheduler can rank hardware
ferro plugins                   # transfer plugins the controller knows about
ferro fetch <plugin> <remote> <local>   # pull data onto the nodes, in parallel
ferro push  <plugin> <local> <remote>   # send results back
ferro train --nodes 2 --gpus-per-node 2 train.py
ferro jobs                      # recent jobs
ferro job <job-id>              # placement, per-rank status, metrics, NCCL errors
ferro logs <job-id> [-f]        # merged logs, tagged by rank and node
ferro cancel <job-id>           # stop every rank and free the GPUs
```

Add `--json` to any command for scripting.

### Timeouts

```bash
ferro train --timeout 2h -f my_train.py      # 90s, 30m and 2h all parse
```

A distributed job that **hangs** never fails: every rank sits in a collective
waiting for a peer, reporting nothing, holding its GPUs. Nothing reclaims
them, because from the outside a wedged job and a slow one look identical.

`--timeout` is the backstop. The controller cancels the job once it exceeds
the limit and the GPUs return to the pool. On a shared cluster, put one on
anything you are not watching — it is the difference between losing an
afternoon and losing a week. A rank that *fails* is handled without it: the
controller tears down its surviving peers immediately.

### Moving data: plugins

FerroGrid does not speak WebDAV, S3 or anything else — moving bytes is a
solved problem. What it adds is running an existing tool **on every node at
once**, so each node pulls its own copy instead of relaying a dataset through
the controller.

A plugin is an argv template in `~/.config/ferrogrid/plugins.toml` on the
controller host (see `plugins.example.toml`):

```toml
[nextcloud]
description = "Nextcloud NAS over WebDAV (NextcloudFetcher)"
fetch = ["ncfetch", "mirror", "{remote}", "--out", "{local}"]
push  = ["ncfetch", "upload-folder", "{local}", "{remote}"]
workdir = "~/.config/ferrogrid"
```

```bash
ferro plugins
ferro fetch nextcloud Datasets/adni /data/adni            # every healthy node
ferro fetch nextcloud Datasets/adni /data/adni --node gpu-a
ferro push  nextcloud runs/exp1 Backups/ferrogrid/exp1 --node gpu-a
```

`{remote}` and `{local}` are substituted as **whole argv elements** and the
command is exec'd directly, never through a shell — a path containing spaces
or `;` is a path, not an injection. Anything with a command line works;
`plugins.example.toml` also sketches rclone and rsync.

**FerroGrid never handles your credentials.** The tool reads its own config
from `workdir` on each node — for NextcloudFetcher that is the `.env` its
README describes, found by searching upward from the working directory.
`scripts/install_plugin_creds.sh` will distribute one for you, but think
first: it copies a secret to every host you name, those hosts are shared, and
anyone with root on them can read it. Prefer a credential scoped to the share
in question and revocable on its own — for Nextcloud, an app password — over
your account password.

#### Transfer archives, not directory trees

Measured on this cluster, fetching the same ADNI scan two ways:

| Shape | Throughput |
|---|---|
| One 1.8 GB archive | **32.5 MB/s** |
| 208 DICOM files (56 MB total) | 0.44 MB/s |

**74× apart.** WebDAV pays a round trip per file, so a scan's ~200 small
DICOMs spend all their time on latency and none on bandwidth. `ncfetch
folder` does not help — it fetches file by file and zips locally.

The consequence is concrete: the 46.8 GB ADNI archive on Nextcloud takes about
**24 minutes** as one file, where the same data as a tree, or from the lab SMB
server at 1.1 MB/s, takes most of a day. Fetch the archive with the
`nextcloud-file` plugin and unpack on the node.

Two things worth knowing:

- **Install the tool on the nodes**, not just the controller. Nothing is
  shipped for you: `uv tool install` (or pip) it on each node.
- The agent runs under `systemd --user`, whose `PATH` is minimal. The unit
  written by `register_node.sh` puts `~/.local/bin` and `~/.cargo/bin` on it
  so user-installed tools are visible; a node deployed before that change will
  report `could not run ...: No such file or directory` until it is
  re-registered.

### Live monitoring

`ferro watch` is the cluster-wide equivalent of `watch -n 1 nvidia-smi`: every
GPU on every node with utilisation and VRAM bars, which job owns each card, and
a row per running job with its live step, loss, throughput and NCCL error count.

```bash
ferro watch            # refresh every 2s
ferro watch -n 1       # every second
```

`ferro ps` is the per-rank view — one row per rank with the node and GPUs it
holds, uptime, live utilisation, VRAM, step and throughput. A `UTIL` in red
means a rank is running but its GPUs are idle, which is what a stall or a
starved dataloader looks like.

`nodes`, `gpu`, `ps`, `jobs` and `job` each take the same `-w/--watch` and
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

### Letting the scheduler choose: `--auto`

```bash
ferro train --auto -f my_train.py
ferro train --auto --gpus-per-node 2 -f my_train.py    # cap at 2 GPUs
```

Auto keeps the job on **one node** and takes the largest set of **identical**
GPUs there, preferring whichever node benchmarks fastest. Both parts follow
from the measurements above: crossing the network costs ~55x, and a collective
runs at the pace of its slowest rank, so a 4090 paired with an A6000 wastes
the 4090.

**What auto cannot know is whether your script shards.** It hands you GPUs; it
cannot tell FSDP2 (where a second GPU makes a small model *slower*, ~3.3x
here) from DDP (where it nearly doubles throughput). If your model fits on one
card and you are sharding it, cap the shape yourself:

```bash
ferro train --auto --gpus-per-node 1 -f my_train.py
```

### Ranking hardware: `ferro bench`

The scheduler prefers faster GPUs, but a model name is a poor proxy for speed.
`ferro bench` measures a bf16 matmul on every free GPU, through that node's own
training image — which also proves the image can actually drive the card:

```bash
ferro bench                # every healthy node, cached results reused
ferro bench --force        # re-measure
ferro bench --node gpu-a
```

Measured on this cluster:

| GPU | bf16 TFLOP/s | relative |
|---|---|---|
| RTX 5090 | 238.0 | 100% |
| RTX 4090 | ~162 | 68% |
| RTX PRO 5000 Blackwell | 105.9 | 44% |
| RTX A6000 | 58.5 | 25% |

The A6000 has twice the VRAM of a 4090 and roughly a third of the throughput —
exactly the kind of thing that makes "most free VRAM" the wrong ranking on its
own. Scores are cached per node and survive restarts; a GPU busy with someone
else's work is skipped rather than measured wrongly.

### Pipeline parallelism: promising in theory, blocked in practice

Pipeline parallelism should be the answer to a slow interconnect. FSDP
all-gathers every parameter each step; PP only ships the activations crossing a
stage boundary — for the model below, ~1.3 GB versus ~8 MB. Two orders of
magnitude, exactly where 1 GbE hurts.

`python/examples/train_pp.py` implements it with
`torch.distributed.pipelining`. It works, but the measurements do not support
using it here:

| Shape | Strategy | tokens/s |
|---|---|---|
| 1 node × 2 GPU | FSDP2 | **63,789** |
| 1 node × 2 GPU | Pipeline (1F1B, 8 microbatches) | 46,632 |
| 2 nodes × 1 GPU | FSDP2 | 1,582 |
| 2 nodes × 1 GPU | Pipeline | **hangs** |

Two separate results:

**Intra-node, PP loses to FSDP** — 46.6k vs 63.8k tokens/s. That is the
expected outcome: inside one box bandwidth is plentiful, so PP's smaller
transfers buy nothing while its pipeline bubble and sequential stage
dependency cost real time.

**Cross-node, PP hangs** — and the network is not at fault. A probe
(`python/examples/p2p_probe.py`) confirms every primitive works between these
nodes: `all_reduce` 90 ms, `send`/`recv` 52 ms, and `batch_isend_irecv` — the
one pipelining actually uses — 65 ms. The hang is inside
`torch.distributed.pipelining` at the first `schedule.step()`, with both GPUs
spinning at 100%.

One real bug was found and fixed along the way: passing `input_args` alone
opts into deprecated init-time shape inference, whose metadata exchange breaks
over NCCL's socket transport (`message truncated: receiving 8 bytes instead of
4`). Passing `output_args` too removes that exchange entirely. It only
reproduces between nodes, which is why the intra-node run looked fine.

These were tried and all hang identically between nodes, while all succeed
inside one:

| Variant | Result |
|---|---|
| `Schedule1F1B`, 8 / 4 / 2 microbatches | hangs |
| `ScheduleGPipe`, 4 / 2 microbatches | hangs |
| 1 microbatch | rejected: microbatches must be >= stages |
| Explicit `input_args` + `output_args` | fixes the truncation, still hangs |
| No `device_id=` on `init_process_group` | hangs |
| `NCCL_PROTO=Simple` | hangs |

A watchdog (`FERRO_STACK_TIMEOUT=75`, which arms `faulthandler` in the
example) shows rank 0 stalled inside `nn.Embedding` on the **first** forward
of the **first** microbatch — before it has sent anything. The GPU is pinned
at 100% by a spinning NCCL kernel, so the next kernel launch never lands.

The evidence points upstream, not at this cluster: every NCCL primitive works
between these two nodes, and the identical code path is fine within one node.
**FSDP2 remains the recommendation here.** Worth retrying on a newer torch;
the theory — two orders of magnitude less traffic — still holds if the
implementation cooperates.

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

A full run on 2× RTX 4090 with FSDP2 — 128³ volumes, 15 epochs, 14.8M
parameters, synthetic but *learnable* data:

```
FSDP2 sharding 6 blocks over 2 ranks
LR schedule: warmup 96 steps, then cosine over 960
epoch  3/15  train_loss 0.3173  val_loss 0.0326  val_acc 100.0%
epoch 15/15  train_loss 0.0027  val_loss 0.0023  val_acc 100.0%
avg step time      116 ms
peak VRAM per rank 0.76 GiB
```

Sizing, measured on one RTX 3090 at 95M parameters with activation
checkpointing on:

| Volume | Batch/GPU | Peak VRAM | Step time |
|---|---|---|---|
| 128³ | 2 | 2.51 GiB | 257 ms |
| 160×192×160 | 1 | 2.68 GiB | 408 ms |

Under 3 GiB on a 24 GiB card — so this **fits on a single GPU with room for a
much larger batch**, and multi-node FSDP would be a pure loss. In 3D imaging
the memory pressure is *activations*, not parameters, which is why the conv
stem's stride and activation checkpointing matter far more than sharding.

#### Do not pool away the position

The single most important line in this model is how the token grid reaches
the classifier. An ablation on identical data:

| Head | train loss | val accuracy |
|---|---|---|
| conv stem → **keep the grid** → linear | 0.0000 | **100%** |
| conv stem → global mean → linear | 1.0719 | 24% (chance) |
| full model, positional embeddings + attention pooling | 1.1000 | 34% (chance) |

Anything that collapses the grid to one vector — a global mean, or attention
with a learned query — discards *where* a feature was. Adding positional
embeddings does not rescue it. The model here pools to a coarse 4×4×4 grid and
flattens, which keeps location and still keeps the head small.

This is not an artefact of synthetic data. In volumetric imaging the location
of an abnormality is most of the diagnosis; a head that cannot represent
position cannot represent the task.

Two other things this run needed, both of which look like "the model cannot
learn" when missing: **LR warmup** (a transformer trained from scratch
otherwise collapses to the class prior and can sit there for tens of epochs)
and **a task that is actually separable** (an early version jittered the signal
by half the class spacing, capping even a perfect oracle at 80%).

#### Getting ADNI onto the cluster

ADNI ships as DICOM: one directory of ~160 `.dcm` files per scan, inside zips
that expand to well over 100 GB. Decoding that in the dataloader would make
every epoch re-do work whose result never changes, and none of these nodes has
the disk for the extracted tree anyway.

`python/tools/preprocess_adni.py` reads the `.dcm` members **straight out of
the zip** — no extraction — and writes one `<image_id>.npy` per scan plus a
`manifest.csv` carrying the label and the cohort's own split:

```bash
uv run --with pandas --with pydicom --with numpy --with scipy python \
    python/tools/preprocess_adni.py \
        --zip    /path/to/FedUQ_T1_MRI.zip \
        --cohort /path/to/cohort_scans.csv \
        --out    ~/adni_t1_128 --shape 128 128 128 --workers 14
```

It handles both archive flavours ADNI ships, and they are not interchangeable:

| Archive | Contents | Per scan |
|---|---|---|
| raw (`FedUQ_T1_MRI.zip`) | DICOM series | ~200 files |
| `ADNI1_Complete *` | one preprocessed NIfTI, gradwarp/B1/N3-corrected | 1 file |

Prefer the NIfTI collections: better input, and two orders of magnitude fewer
files to move.

**A preprocessed collection will not join on `image_id`.** ADNI assigns
derivatives their own IDA image IDs, distinct from the raw series they came
from, so matching a "Complete" archive to a cohort table by `image_id` finds
exactly nothing. The tool falls back to subject + scan date, both of which are
in the archive path, and reports which key it used — on the 46.8 GB ADNI1
collection that is 1,705 of 2,294 scans matched, all by `ptid+date`.

At 128³ float16 a volume is 4.2 MB, so a few hundred scans fit in a couple of
GB and stream comfortably. Three things it gets right that are easy to get wrong:

- **Orientation.** NIfTI volumes are reoriented to canonical RAS before
  resampling. ADNI scans arrive in assorted orientations, and stacking them
  as-stored trains the model on whichever way each scanner wrote its axes.

- **Slice ordering** comes from each slice's position along the slice normal,
  not from the filename or `InstanceNumber`. ADNI mixes conventions across
  sites and decades, and a wrongly ordered stack looks perfectly plausible
  while being anatomically scrambled.
- **Slice spacing** is taken from the gap between the first two slices, not
  `SliceThickness`, which ignores any inter-slice gap and distorts the aspect
  ratio.

**Copy the zip to local disk first.** Reading it over CIFS/NFS means ~160 small
random reads per scan across the network: measured here at 0.8 scans/min from
an SMB share versus minutes for the whole set once local.

The manifest is read with the stdlib `csv` module, not pandas: this runs
inside the training container and the stock PyTorch images do not ship pandas.
Reading a manifest is not worth making every user build a custom image for.

Then point the trainer at the output — it reads the manifest, honours the
cohort's train/val split (never reshuffle it: the same subject appears in
several scans and would leak across the boundary), and augments with
left-right flips only:

```bash
ferro train --nodes 1 --gpus-per-node 2 -f \
    --mount /home/you/adni_t1_128:/data/adni:ro \
    python/examples/train_mri_3d.py \
        --data-root /data/adni --label-set cn-ad --classes 2
```

`--label-set cn-ad` drops MCI. MCI sits between the two classes by definition
and needs far more data to separate; a handful of MCI scans adds noise rather
than a third class.

**Read the balanced accuracy, not the accuracy.** ADNI splits are imbalanced,
and a model that has learned nothing still scores the prior: on this split
validation is 17 CN and 4 AD, so predicting "CN" every time gives 81%. The
trainer reports balanced accuracy (the mean of per-class recalls) and each
class's recall alongside it, because always-predicting-the-majority scores
`1 / n_classes` there however lopsided the split is. A run whose accuracy is
high while its balanced accuracy sits at chance has learned nothing. Measured
here, on 97 training scans:

```
epoch  5/30  train_loss 0.7586  val_loss 0.6227  acc 81.0%  balanced 50.0%  [CN=100%  AD=0%]
epoch 30/30  train_loss 0.5160  val_loss 0.4793  acc 76.2%  balanced 47.1%  [CN=94%   AD=0%]
best balanced acc  50.0%  (chance is 50%)
```

81% accuracy, and AD recall never leaves zero. The pipeline is sound — real
DICOM through preprocessing, FSDP2 across two GPUs, honest metrics — but 97
scans (67 CN / 30 AD) cannot train an AD classifier, and the run says so
plainly instead of reporting a number that flatters it.

**Validation is sharded by hand, not with `DistributedSampler`.**
`DistributedSampler` pads the set so every rank gets an equal count, which
duplicates samples. On a small validation set that is not a rounding detail:
21 scans over 2 ranks became 22 evaluations, one of them counted twice, and
the reported accuracy was of a set that does not exist.

Running the synthetic version:

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
