#!/usr/bin/env python3
"""Template: 3D MRI classification with a CNN stem + Video-Swin-style encoder.

This is a *skeleton to adapt*, not a published architecture. Swap `build_model`
for your real 3D-CNN-VSwinFormer and `build_dataset` for your ADNI loader; the
rest -- FSDP2 wrapping, activation checkpointing, bf16, metric reporting -- is
the part that matters for running it on FerroGrid and is ready to use.

    ferro train --nodes 1 --gpus-per-node 2 --follow \
        --mount /mnt/adni_data --mount /mnt/adni_work \
        --image ferrogrid/train:mri \
        python/examples/train_mri_3d.py \
            --data-root /mnt/adni_data --out-dir /mnt/adni_work/runs/exp1

Read the "Choosing a parallelism strategy" note in the README before reaching
for --nodes > 1: for a model that fits on one GPU, sharding across the 1 GbE
fabric is much slower than simply running on one node.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ferro_mojo  # noqa: E402


def log(msg):
    print(msg, flush=True)


def metric(**kw):
    print("FERRO_METRIC " + json.dumps(kw), flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SyntheticMRI(Dataset):
    """Stand-in so the template runs before your loader exists.

    Replace with your ADNI dataset. For NIfTI volumes, MONAI's
    `CacheDataset` + `LoadImaged` is the usual choice; keep the returned
    shapes identical to this: volume (1, D, H, W) float32, label int64.
    """

    def __init__(self, n: int, shape: tuple[int, int, int], classes: int):
        self.n, self.shape, self.classes = n, shape, classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        vol = torch.randn(1, *self.shape, generator=g)
        return vol, torch.randint(0, self.classes, (1,), generator=g).item()


def build_dataset(args, train: bool):
    if args.data_root:
        # >>> Your ADNI dataset goes here. Something like:
        #   from monai.data import CacheDataset
        #   from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
        #                                 Orientationd, Spacingd, NormalizeIntensityd,
        #                                 RandSpatialCropd, RandFlipd)
        #   return CacheDataset(data=records, transform=Compose([...]))
        raise NotImplementedError(
            f"--data-root={args.data_root} given, but build_dataset() is still "
            "the template stub. Plug your ADNI loader in here."
        )
    n = args.train_size if train else args.val_size
    return SyntheticMRI(n, tuple(args.volume), args.classes)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvStem(nn.Module):
    """Strided 3D conv stem: cuts the volume down before attention.

    Attention over a raw MRI volume is hopeless -- 128^3 voxels is 2M tokens.
    The stem is what makes the transformer affordable, so its stride is the
    single most important knob for both speed and memory here.
    """

    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, dim // 4, 3, stride=2, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.GELU(),
            nn.Conv3d(dim // 4, dim // 2, 3, stride=2, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.GELU(),
            nn.Conv3d(dim // 2, dim, 3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SwinLikeBlock(nn.Module):
    """Placeholder for one Video-Swin stage.

    Real Video Swin does 3D *shifted-window* attention. This uses full
    attention over the (already downsampled) token grid so the template is
    self-contained and correct -- swap in your windowed implementation and the
    FSDP2 wrapping below still applies unchanged.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class CNNVSwinFormer(nn.Module):
    def __init__(self, dim=384, depth=6, heads=6, classes=3, in_ch=1):
        super().__init__()
        self.stem = ConvStem(in_ch, dim)
        self.blocks = nn.ModuleList(SwinLikeBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x):
        x = self.stem(x)                       # (B, C, d, h, w)
        b, c = x.shape[:2]
        x = x.flatten(2).transpose(1, 2)       # (B, tokens, C)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))


def build_model(args):
    return CNNVSwinFormer(
        dim=args.dim, depth=args.depth, heads=args.heads,
        classes=args.classes, in_ch=args.in_channels,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="", help="ADNI root; omit for synthetic data")
    p.add_argument("--out-dir", default="", help="checkpoint dir (shared storage)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    p.add_argument("--batch-size", type=int, default=1, help="per GPU")
    p.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    p.add_argument("--volume", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--val-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--no-fsdp", action="store_true",
                   help="skip sharding; correct when the model fits on one GPU")
    p.add_argument("--no-checkpointing", action="store_true",
                   help="disable activation checkpointing (uses much more VRAM)")
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.cuda.is_available():
        log("CUDA unavailable in the container; aborting")
        sys.exit(2)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    is_master = rank == 0

    if is_master:
        log(f"world_size={world_size} volume={tuple(args.volume)} "
            f"batch/gpu={args.batch_size} accum={args.accum} "
            f"effective_batch={args.batch_size * args.accum * world_size}")
        log(f"mojo kernels: {ferro_mojo.describe()['reason']}")

    model = build_model(args).to(device)

    # Activation checkpointing before sharding. 3D volumes are activation-
    # bound, not parameter-bound: this is usually what decides whether a
    # sensible batch size fits at all.
    if not args.no_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
        )
        apply_activation_checkpointing(
            model, check_fn=lambda m: isinstance(m, SwinLikeBlock)
        )
        if is_master:
            log("activation checkpointing enabled on transformer blocks")

    if not args.no_fsdp and world_size > 1:
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        # Shard the transformer blocks individually so FSDP2 can overlap each
        # all-gather with the previous block's compute. The conv stem is small;
        # it rides along with the root.
        for blk in model.blocks:
            fully_shard(blk, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        if is_master:
            log(f"FSDP2 sharding {len(model.blocks)} blocks over {world_size} ranks")

    params = sum(p.numel() for p in model.parameters())
    if is_master:
        log(f"parameters: {params / 1e6:.1f}M")

    train_ds = build_dataset(args, train=True)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
        # Volumes are large; prefetching too far ahead just burns host RAM.
        prefetch_factor=2 if args.workers > 0 else None,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    loss_fn = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(device)
    step = 0
    timed_steps, timed_seconds = 0, 0.0
    stop = False

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.perf_counter()

        for i, (vol, label) in enumerate(loader):
            vol = vol.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(vol), label) / args.accum
            loss.backward()

            if (i + 1) % args.accum:
                continue

            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)

            step += 1
            elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()
            if step > args.warmup_steps:
                timed_steps += 1
                timed_seconds += elapsed

            if is_master and step % args.log_every == 0:
                avg = timed_seconds / timed_steps if timed_steps else elapsed
                vox = args.batch_size * args.accum * world_size
                metric(
                    step=step,
                    loss=round(loss.item() * args.accum, 4),
                    step_time_ms=round(avg * 1000, 1),
                    samples_per_s=round(vox / avg, 3),
                    peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
                )

            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if args.out_dir and is_master:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # For a sharded model use torch.distributed.checkpoint instead; this
        # simple path is correct for the single-node / unsharded case.
        torch.save({"step": step, "args": vars(args)}, out / "last.pt")
        log(f"wrote {out / 'last.pt'}")

    dist.barrier()
    if is_master and timed_steps:
        avg = timed_seconds / timed_steps
        log("=" * 60)
        log(f"optimizer steps    {step}")
        log(f"avg step time      {avg * 1000:.0f} ms")
        log(f"peak VRAM per rank {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
        log("=" * 60)
    dist.destroy_process_group()
    if is_master:
        log("training completed cleanly")


if __name__ == "__main__":
    main()
