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
- Node registration supports password-only SSH via an SSH ControlMaster socket
  (one prompt, multiplexed thereafter) rather than sshpass, which would expose
  the password through `ps`. `%C` in a ControlPath is an ssh token, so build
  the path by hand -- mktemp rejects it.
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

- `ferro ps` lists every process on every GPU, not just ours, so a card held
  by somebody's notebook is not read as free. NVML's process queries are
  `_v3` on driver 510+; the older lab drivers need the `_v2` fallback, which
  lives behind nvml-wrapper's `legacy-functions` feature. Attribution is by
  cgroup id + a cached `docker ps` for containers and by the pid ancestry
  chain for `--no-docker` jobs: a container's workers descend from containerd,
  never from the agent, so ancestry alone finds nothing under Docker.
- NVML's per-process utilisation is a *sampled* API: ask for samples newer
  than the last one you saw (feed its own timestamps back -- they are its
  clock, not yours), and treat `NotFound` as "nothing happened since", not as
  an error. It only ever reports instants, so the agent keeps the "last seen
  busy" clock that makes idleness a duration. A driver that cannot attribute
  utilisation at all must report *unknown*, never 0%: one of those is grounds
  for calling somebody's job a squatter and the other is not.
- `ferro ps <pid>` reads the node live (`DescribeProcess`) instead of using the
  heartbeat: the heartbeat's command is trimmed for the table and its numbers
  are up to one interval old. The controller asks whichever node's last
  heartbeat mentions the pid and falls back to asking everyone, because pids
  are unique per machine and the heartbeat only lists GPU holders.
- Command lines reported upstream are redacted for credential-looking flags in
  `procs::redact_secrets`. `ps` on the node shows the same string to anyone
  logged in there, but FerroGrid fans it out to every client and JSON dump; a
  real lab box had an inference server running with `--api-key` in argv.
- `/proc/<pid>/cmdline` is NUL-separated, not space-separated: splitting it on
  whitespace leaves embedded NULs in the string and the terminal eats them.
- `/proc/<pid>/stat` field 2 is the executable name and may itself contain
  spaces and parentheses, so parse from the *last* `)`. Process start time is
  `/proc/stat`'s `btime` + `starttime`/100 -- USER_HZ is 100 in the userspace
  ABI whatever the kernel's tick rate is.

- A queued job (`--wait`) has an empty plan, so `Job::phase()` has to special
  case it: the per-rank vote reads "no placements" as "every rank succeeded".
  Queue order comes from `job_order`, not from `submitted` -- two jobs
  submitted in the same second must still have an order, and it has to agree
  with the position each was told.

- `ferro net` measures pairs strictly one at a time: two probes at once share
  the switch and each ends up measuring the other. It sends to the peer's
  `nccl_address`, not its management address -- on these boxes they are
  frequently different wires, and the one NCCL uses is the one worth knowing.
  The negotiated link speed in `ferro nodes` is a different claim from measured
  throughput: it covers the node-to-switch hop only, and a 1000 Mb/s NIC with
  zero errors can still sit behind a 100 Mb/s path.

- `scripts/migrate.sh` moves the whole setup to another machine. The node
  inventory it carries is read live from the controller (`ferro --json nodes`),
  because that registry only ever exists in the controller's memory -- there is
  no file to copy, and a host list would go stale. It therefore refuses to run
  with the controller down rather than migrating a guess.
- `ssh_config` is **first-obtained-value-wins**, so the migrated block is
  *prepended*: land it after somebody's `Host *` and its per-host `IdentityFile`
  is silently ignored. The block also names the key explicitly only where the
  original block did not, so a host with its own key keeps it.
- The migration bundle contains a private key. It is removed from the target as
  soon as the install finishes; the local copy lives in a `mktemp -d` cleaned by
  the exit trap.
- Reaching the controller and reaching the nodes are separate questions on the
  target. A machine on the VPN only can talk to the controller (give it
  `--controller <this machine's VPN address>`; the LAN address the script
  otherwise guesses is unreachable there) and still have no route to the node
  network at all, which breaks `ferro sync` and nothing else. `--proxy-jump
  [user@]host` puts a `ProxyJump` in every migrated block that does not already
  route itself; the script probes the first node's SSH port from the target
  before shipping and suggests the flag when there is no route.

- A failed rank must tear down its peers: survivors sit in a collective
  forever holding GPUs. The controller does this in `report_job_status`.
- `torch.distributed.pipelining` hangs cross-node here even though every NCCL
  primitive works (`p2p_probe.py` proves it). Pass both `input_args` and
  `output_args` to `PipelineStage` regardless -- `input_args` alone uses a
  deprecated shape-inference path that corrupts metadata over sockets.

- Plugins are argv templates, never shell strings, and are exec'd directly:
  substitution happens per argv element so paths cannot inject. Credentials
  never travel through FerroGrid -- the tool reads its own config from the
  plugin `workdir` on each node.
- `systemd --user` has a minimal PATH; the agent unit must extend it or
  user-installed tools (ncfetch, rclone) are invisible to plugins.

## Conventions

- "Free" means *placeable* everywhere it appears (`ferro nodes` FREE, the
  `n/m free` counts, `GpuEntry.schedulable`): no job of ours **and** at least
  `--min-free-vram-gib` left, which is exactly what the scheduler applies. A
  view that counts a card somebody else filled as free promises placements
  that then fail. "Occupied" is a separate answer from "unusable" -- a
  compositor holding 80 MB is worth showing but does not make a 24 GB card
  unavailable.
- Live views (`ferro watch`, `-w` on the read-only commands) redraw faster than
  agents report. Always surface data age alongside the numbers -- a stale
  reading looks exactly like an idle GPU otherwise.
- Renderers build a frame and return it; only `main` prints. Clearing the
  screen and then fetching leaves it blank for the whole round-trip, which at
  `-n 1` is the flicker. `Screen` overwrites the previous frame in one write
  (`\x1b[K` per line, `\x1b[J` at the end) and clears only on the first frame
  and on resize, and it trims the frame to the window: a view that scrolls puts
  the next repaint a row off and smears.

- Training scripts report metrics by printing `FERRO_METRIC {json}` on stdout.
- Agents resolve relative script paths against their own `--workspace`
  (default `~/ferrogrid`), because lab nodes have different home directories.
  Code must therefore reach the nodes first: `ferro sync`, or `ferro train
  --sync`. Nodes advertise their login user and workspace at registration, so
  neither needs a host list.
- Never reimplement FSDP/NCCL/torchrun; the platform only wraps them.
- Only the agent workspace is mounted by default; datasets need `--mount`.
  Bind mounts are filesystem-agnostic, so NFS and Samba/CIFS need no code --
  but both fix uid/gid at host mount time, and jobs run as the user's uid, so
  "cannot write to the output dir" is almost always a host-mount problem.
- Mojo kernels stay optional and must always have a working PyTorch fallback.
  Measure before adopting one: as of MAX 26.5 the custom-op bridge is slower
  than PyTorch's fused elementwise kernels (see mojo/README.md).
