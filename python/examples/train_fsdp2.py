#!/usr/bin/env python3
"""FSDP2 reference model for FerroGrid.

Trains a small decoder-only transformer on synthetic tokens, sharded across
every rank with PyTorch's FSDP2 (`torch.distributed.fsdp.fully_shard`) over
NCCL. Nothing here reimplements FSDP or NCCL -- it is stock PyTorch, launched
by torchrun, with the rendezvous environment supplied by the FerroGrid
controller.

Rank 0 prints one `FERRO_METRIC {...}` line per logging interval; the agent
forwards stdout to the controller, which parses those lines into the
throughput/VRAM figures shown by `ferro job`.

Run it through the platform:
    ferro train --nodes 2 --gpus-per-node 2 python/examples/train_fsdp2.py

or standalone for a single-node smoke test:
    torchrun --nproc_per_node=2 python/examples/train_fsdp2.py --steps 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# The training container has no FerroGrid package installed, so make the
# sibling modules importable from the checkout itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ferro_mojo  # noqa: E402
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def metric(**kw) -> None:
    """Emit a line the controller's metric parser understands."""
    print("FERRO_METRIC " + json.dumps(kw), flush=True)


class MojoGELU(nn.Module):
    """GELU backed by the Mojo custom kernel, falling back to PyTorch.

    The fallback lives in ferro_mojo, so this module behaves identically on a
    node without a Mojo toolchain -- just without the custom kernel.
    """

    def forward(self, x):
        return ferro_mojo.gelu(x)


class Block(nn.Module):
    """One pre-norm transformer block -- the unit we shard on."""

    def __init__(self, d_model: int, n_heads: int, activation: str = "torch"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            MojoGELU() if activation == "mojo" else nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, attn_mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        seq_len: int,
        activation: str = "torch",
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, activation) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # Causal mask, registered so `.to(device)` moves it with the model.
        self.register_buffer(
            "mask",
            torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        mask = self.mask[:t, :t]
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.norm(x))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU micro batch")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=3,
                   help="steps excluded from the throughput average")
    p.add_argument("--no-fsdp", action="store_true", help="debug: skip sharding")
    p.add_argument(
        "--offload",
        action="store_true",
        help="keep sharded params and optimizer state in host RAM. Trades a "
             "lot of speed for fitting a model the GPUs cannot otherwise hold.",
    )
    p.add_argument(
        "--activation",
        choices=("torch", "mojo"),
        default="torch",
        help="MLP activation. 'mojo' uses the Mojo custom GELU kernel when "
             "MAX is installed, and silently falls back to PyTorch when not.",
    )
    p.add_argument(
        "--param-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="FSDP2 all-gather dtype. bf16 halves inter-node traffic, which "
             "dominates step time on a 1 GbE cluster.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.cuda.is_available():
        log("CUDA is not available inside the container; aborting")
        sys.exit(2)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Passing device_id lets NCCL bind the rank to its GPU up front. Without
    # it PyTorch guesses from the global rank, which warns and can hang when
    # the rank->GPU mapping is not uniform across nodes.
    dist.init_process_group(backend="nccl", device_id=device)
    is_master = rank == 0

    if is_master:
        log(
            f"FerroGrid FSDP2 example | world_size={world_size} "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
            f"torch={torch.__version__} nccl={'.'.join(map(str, torch.cuda.nccl.version()))}"
        )
    log(
        f"[rank {rank}] node={os.environ.get('FERRO_NODE_ID', '?')} "
        f"local_rank={local_rank} gpu={torch.cuda.get_device_name(local_rank)}"
    )

    torch.manual_seed(1234)
    if is_master and args.activation == "mojo":
        info = ferro_mojo.describe()
        log(f"Mojo kernels: {'available' if info['available'] else 'UNAVAILABLE'} "
            f"-- {info['reason']}")

    model = TinyLM(
        args.vocab, args.d_model, args.layers, args.heads, args.seq_len, args.activation
    ).to(device)

    if not args.no_fsdp:
        # Sharded params are all-gathered in param_dtype and gradients are
        # reduced in fp32. On a 1 GbE fabric the all-gather is the step-time
        # bottleneck, so bf16 params roughly halve it; fp32 reduction keeps
        # the optimiser update numerically stable.
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if args.param_dtype == "bf16" else torch.float32,
            reduce_dtype=torch.float32,
        )
        # CPU offload moves the *sharded* params, grads and optimizer state to
        # host RAM, leaving only the transient all-gathered block on the GPU.
        # It is the difference between "does not fit" and "fits but is slow".
        offload = CPUOffloadPolicy() if args.offload else None
        shard_kw = {"mp_policy": mp}
        if offload is not None:
            shard_kw["offload_policy"] = offload
        # Shard each block, then the root. Sharding the blocks first is what
        # lets FSDP2 overlap the all-gather of block N+1 with block N.
        for block in model.blocks:
            fully_shard(block, **shard_kw)
        fully_shard(model, **shard_kw)
        if is_master:
            log(f"FSDP2 fully_shard applied to all blocks + root "
                f"(param_dtype={args.param_dtype}, reduce_dtype=fp32"
                f"{', cpu_offload' if args.offload else ''})")

    params = sum(p.numel() for p in model.parameters())
    if is_master:
        log(f"model parameters: {params / 1e6:.1f}M (sharded over {world_size} rank(s))")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic data keeps the benchmark about the platform, not the dataloader.
    gen = torch.Generator(device="cuda").manual_seed(rank)

    tokens_per_step = args.batch_size * args.seq_len * world_size
    samples_seen, timed_steps, timed_seconds = 0, 0, 0.0
    torch.cuda.reset_peak_memory_stats(device)

    dist.barrier()
    if is_master:
        log("starting training loop")

    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()

        idx = torch.randint(
            0, args.vocab, (args.batch_size, args.seq_len), device=device, generator=gen
        )
        targets = torch.roll(idx, shifts=-1, dims=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(idx)
            loss = loss_fn(logits.reshape(-1, args.vocab), targets.reshape(-1))

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_start
        samples_seen += args.batch_size * world_size

        # Warmup steps pay for CUDA graph capture, autotuning and the first
        # NCCL all-gathers; averaging them in understates steady-state speed.
        if step > args.warmup_steps:
            timed_steps += 1
            timed_seconds += elapsed

        if is_master and (step % args.log_every == 0 or step == args.steps):
            avg = timed_seconds / timed_steps if timed_steps else elapsed
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            metric(
                step=step,
                loss=round(loss.item(), 4),
                step_time_ms=round(avg * 1000, 2),
                samples_per_s=round(args.batch_size * world_size / avg, 2),
                tokens_per_s=round(tokens_per_step / avg, 1),
                peak_vram_gb=round(peak_gb, 2),
            )

    dist.barrier()

    if is_master and timed_steps:
        avg = timed_seconds / timed_steps
        log("=" * 62)
        log(f"steps                {args.steps} ({timed_steps} timed)")
        log(f"world size           {world_size}")
        log(f"avg step time        {avg * 1000:.1f} ms")
        log(f"throughput           {tokens_per_step / avg:,.0f} tokens/s "
            f"({args.batch_size * world_size / avg:.1f} samples/s)")
        log(f"peak VRAM per rank   {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
        log(f"total samples        {samples_seen}")
        log("=" * 62)

    dist.destroy_process_group()
    if is_master:
        log("training completed cleanly")


if __name__ == "__main__":
    main()
