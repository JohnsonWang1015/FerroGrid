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

import numpy as np
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

    The signal is *learnable*: each class puts a soft blob at a different
    place in the volume, under noise. That matters -- with random labels the
    loss cannot fall, so a run proves only that nothing crashed. With a real
    signal, a falling loss and rising validation accuracy show that gradients,
    the optimizer and FSDP2's sharding are all actually correct.

    Replace with your ADNI dataset. For NIfTI volumes, MONAI's `CacheDataset`
    + `LoadImaged` is the usual choice; keep the returned shapes identical:
    volume (1, D, H, W) float32, label int64.
    """

    def __init__(self, n: int, shape: tuple[int, int, int], classes: int, seed: int = 0):
        self.n, self.shape, self.classes, self.seed = n, shape, classes, seed
        d, h, w = shape
        # One centre per class, spread along the depth axis.
        self.centres = [
            (int(d * (c + 1) / (classes + 1)), h // 2, w // 2) for c in range(classes)
        ]
        self.radius = max(2, min(shape) // 12)
        # Jitter must stay well inside the class spacing. At half the spacing
        # the classes overlap so much that even an oracle that knows exactly
        # what to look for tops out around 80%, and the model never gets a
        # clean gradient signal -- which is what stalled an earlier version of
        # this at chance.
        self.jitter = max(1, d // (classes + 1) // 4)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        label = int(torch.randint(0, self.classes, (1,), generator=g).item())
        vol = torch.randn(1, *self.shape, generator=g) * 0.5

        cz, cy, cx = self.centres[label]
        d, h, w = self.shape
        r = self.radius
        # Jitter so the model cannot memorise an exact coordinate, but not so
        # far that neighbouring classes become genuinely ambiguous.
        j = self.jitter
        jz, jy, jx = (int(torch.randint(-j, j + 1, (1,), generator=g).item()) for _ in range(3))
        z0, z1 = max(0, cz + jz - r), min(d, cz + jz + r)
        y0, y1 = max(0, cy + jy - r), min(h, cy + jy + r)
        x0, x1 = max(0, cx + jx - r), min(w, cx + jx + r)
        vol[0, z0:z1, y0:y1, x0:x1] += 3.0

        return vol, label


class ADNIVolumes(Dataset):
    """Preprocessed ADNI T1 volumes, as written by tools/preprocess_adni.py.

    Deliberately dumb: one `.npy` per scan, memory-mapped, no DICOM parsing at
    train time. Decoding DICOM in the dataloader would make every epoch pay for
    work whose result never changes, and on this cluster the data path is
    already the tighter constraint (see "Watch the data path").
    """

    LABEL_SETS = {
        "all": ("CN", "MCI", "AD"),
        # CN vs AD is the standard ADNI benchmark: MCI sits between the two by
        # definition and needs far more data to separate, so including a
        # handful of MCI scans adds noise rather than a third class.
        "cn-ad": ("CN", "AD"),
    }

    def __init__(self, root: Path, split: str, augment: bool, label_set: str = "all"):
        # csv, not pandas: this runs inside the training container, and the
        # stock PyTorch images do not ship pandas. Reading a manifest is not
        # worth making every user build a custom image for.
        import csv

        self.root = Path(root)
        manifest = self.root / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"no manifest.csv in {self.root}. Run python/tools/preprocess_adni.py first."
            )
        self.labels = self.LABEL_SETS[label_set]
        with manifest.open(newline="") as fh:
            # The cohort ships its own split; honour it rather than
            # reshuffling, which would leak a subject across the boundary --
            # one subject contributes several scans.
            self.rows = [
                r
                for r in csv.DictReader(fh)
                if r["split"] == split and r["label"] in self.labels
            ]
        self.augment = augment
        self.index = {name: i for i, name in enumerate(self.labels)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        vol = np.load(self.root / row["file"], mmap_mode="r")
        vol = torch.from_numpy(np.asarray(vol, dtype=np.float32)).unsqueeze(0)

        if self.augment:
            # Left-right flip only. Brains are near-symmetric, so it is the one
            # augmentation that is clearly label-preserving here; rotations and
            # crops risk moving the structures the diagnosis depends on.
            if torch.rand(()) < 0.5:
                vol = torch.flip(vol, dims=(3,))

        return vol, self.index[row["label"]]


def build_dataset(args, train: bool):
    if args.data_root:
        ds = ADNIVolumes(
            Path(args.data_root), split="train" if train else "val",
            augment=train, label_set=args.label_set,
        )
        if args.classes != len(ds.labels):
            raise SystemExit(
                f"--classes {args.classes} but --label-set {args.label_set} "
                f"has {len(ds.labels)}: {ds.labels}"
            )
        return ds
    n = args.train_size if train else args.val_size
    # Different seeds so validation volumes are genuinely unseen.
    return SyntheticMRI(n, tuple(args.volume), args.classes, seed=0 if train else 7)


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
    def __init__(self, dim=384, depth=6, heads=6, classes=3, in_ch=1, volume=(128, 128, 128)):
        super().__init__()
        self.stem = ConvStem(in_ch, dim)
        # The stem has three stride-2 convs, so the token grid is volume / 8.
        grid = tuple(max(1, v // 8) for v in volume)
        self.n_tokens = grid[0] * grid[1] * grid[2]

        # Positional embedding is not optional here. Attention and the mean
        # pool that follows are both permutation-invariant, so without it the
        # model cannot tell *where* a feature is -- only that it exists. In
        # neuroimaging that throws away most of the signal: which structure
        # has atrophied is the diagnosis.
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList(SwinLikeBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

        # Pool to a coarse grid, do not collapse to a single vector.
        #
        # This is the most important line in the model. Any pooling that
        # reduces the token grid to one vector -- a global mean, or attention
        # with a learned query -- throws away *where* a feature was. On this
        # task an ablation was unambiguous: keeping the grid reached 100%
        # validation accuracy while global mean pooling sat at chance, on
        # identical data. The same holds for real volumetric work, where the
        # location of an abnormality is most of the diagnosis.
        self.grid = grid
        self.pool_to = tuple(min(g, 4) for g in grid)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim * self.pool_to[0] * self.pool_to[1] * self.pool_to[2], classes),
        )

    def forward(self, x):
        x = self.stem(x)                       # (B, C, d, h, w)
        x = x.flatten(2).transpose(1, 2)       # (B, tokens, C)
        if x.shape[1] != self.pos.shape[1]:
            raise ValueError(
                f"got {x.shape[1]} tokens but the positional embedding has "
                f"{self.pos.shape[1]}; --volume must match what the model was built with"
            )
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # tokens -> (B, C, d, h, w) -> coarse grid -> flat
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, *self.grid)
        x = nn.functional.adaptive_avg_pool3d(x, self.pool_to)
        return self.head(x)


def build_model(args):
    return CNNVSwinFormer(
        dim=args.dim, depth=args.depth, heads=args.heads,
        classes=args.classes, in_ch=args.in_channels, volume=tuple(args.volume),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class PaddedShard(Dataset):
    """One rank's slice of a validation set, padded to a common length.

    Both constraints have to hold at once:

    * every rank must run the **same number of forward passes**, because a
      sharded model all-gathers on each one and a rank that stops early leaves
      its peers waiting in a collective until the watchdog kills the job;
    * the metric must count each scan **once**, so the padding cannot be
      allowed to inflate it.

    So the shard is padded like `DistributedSampler` does, and each item
    carries a 0/1 weight that removes the padding from the totals.
    """

    def __init__(self, base: Dataset, rank: int, world_size: int):
        idx = list(range(rank, len(base), world_size))
        per_rank = -(-len(base) // world_size)  # ceil
        self.valid = [1.0] * len(idx) + [0.0] * (per_rank - len(idx))
        # Repeat the last real index for padding; its weight is zero.
        self.idx = idx + [idx[-1] if idx else 0] * (per_rank - len(idx))
        self.base = base

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        vol, label = self.base[self.idx[i]]
        return vol, label, self.valid[i]


@torch.no_grad()
def evaluate(model, loader, device, world_size, n_classes):
    """Validation accuracy and loss, reduced across ranks.

    Each rank sees a disjoint shard, so the totals have to be summed globally
    before dividing -- averaging per-rank accuracies would be wrong whenever
    the shards differ in size.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    totals = torch.zeros(3, device=device)  # loss, correct, count
    # Per-class correct/count, so a model that has simply learned the class
    # prior cannot hide behind overall accuracy. On an imbalanced split those
    # two numbers are far apart and only one of them is informative.
    per_class = torch.zeros(2, n_classes, device=device)

    for vol, label, weight in loader:
        vol = vol.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(vol)
        hit = (logits.argmax(-1) == label).float() * weight
        totals[0] += (loss_fn(logits.float(), label) * weight).sum()
        totals[1] += hit.sum()
        totals[2] += weight.sum()
        per_class[0].index_add_(0, label, hit)
        per_class[1].index_add_(0, label, weight)

    if world_size > 1:
        dist.all_reduce(totals)
        dist.all_reduce(per_class)
    model.train()

    n = max(totals[2].item(), 1)
    seen = per_class[1] > 0
    # Balanced accuracy: the mean of the per-class recalls. Always-predict-the
    # majority scores 1/n_classes here, however lopsided the split is.
    balanced = (per_class[0][seen] / per_class[1][seen]).mean().item() if seen.any() else 0.0
    recalls = [
        (per_class[0][c] / per_class[1][c]).item() if per_class[1][c] > 0 else float("nan")
        for c in range(n_classes)
    ]
    return totals[0].item() / n, totals[1].item() / n, balanced, recalls


def save_checkpoint(model, opt, out_dir: Path, tag, sharded: bool, is_master: bool):
    """Persist the model.

    A sharded model must go through torch.distributed.checkpoint: each rank
    holds only a slice, so a plain torch.save writes one shard and silently
    loses the rest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if sharded:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_sd, opt_sd = get_state_dict(model, opt)
        dcp.save({"model": model_sd, "optim": opt_sd}, checkpoint_id=str(out_dir / str(tag)))
        return out_dir / str(tag)

    if is_master:
        path = out_dir / f"{tag}.pt"
        torch.save({"tag": tag, "model": model.state_dict()}, path)
        return path
    return None


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
    p.add_argument("--classes", type=int, default=3, help="CN / MCI / AD")
    p.add_argument("--label-set", choices=("all", "cn-ad"), default="all",
                   help="which ADNI diagnoses to train on")
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
    p.add_argument("--eval-every", type=int, default=1, help="validate every N epochs")
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="fraction of training spent warming the LR up from 0")
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many evaluations without improvement (0 = never)")
    p.add_argument("--weight-decay", type=float, default=0.05)
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

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
    )
    loss_fn = nn.CrossEntropyLoss()

    # Linear warmup then cosine decay. Without warmup a transformer trained
    # from scratch collapses to predicting the class prior and can sit there
    # for tens of epochs before escaping -- which looks exactly like "the model
    # cannot learn this" and is the single easiest way to waste a day.
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = max(1, steps_per_epoch * args.epochs if not args.max_steps else args.max_steps)
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    def lr_at(s: int) -> float:
        if s < warmup_steps:
            return s / warmup_steps
        import math
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    if is_master:
        log(f"LR schedule: warmup {warmup_steps} steps, then cosine over {total_steps}")

    val_ds = build_dataset(args, train=False)
    # See PaddedShard: padding keeps the ranks' collective counts aligned,
    # the weights keep the metric honest.
    val_loader = DataLoader(
        PaddedShard(val_ds, rank, world_size), batch_size=args.batch_size,
        shuffle=False, num_workers=max(1, args.workers // 2), pin_memory=True,
    )

    # The held-out test split, scored only at the epoch validation picks as
    # best and never used to choose anything. Selecting on a 268-scan
    # validation set is itself noisy -- balanced accuracy swings 10 points
    # between epochs here -- so the validation figure is optimistic by
    # construction. This is the number to quote.
    test_loader = None
    if args.data_root:
        try:
            test_ds = ADNIVolumes(
                Path(args.data_root), split="test", augment=False, label_set=args.label_set
            )
            if len(test_ds):
                test_loader = DataLoader(
                    PaddedShard(test_ds, rank, world_size), batch_size=args.batch_size,
                    shuffle=False, num_workers=max(1, args.workers // 2), pin_memory=True,
                )
                if is_master:
                    log(f"test split held out: {len(test_ds)} scans")
        except FileNotFoundError:
            pass
    if is_master:
        log(f"data: {len(train_ds)} train, {len(val_ds)} val "
            f"({', '.join(val_ds.labels) if hasattr(val_ds, 'labels') else 'synthetic'})")

    torch.cuda.reset_peak_memory_stats(device)
    step = 0
    timed_steps, timed_seconds = 0, 0.0
    best_acc, best_epoch, since_best = 0.0, 0, 0
    best_test = None
    stop = False

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.perf_counter()
        epoch_loss, epoch_batches = 0.0, 0

        for i, (vol, label) in enumerate(loader):
            vol = vol.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(vol), label) / args.accum
            loss.backward()
            epoch_loss += loss.item() * args.accum
            epoch_batches += 1

            if (i + 1) % args.accum:
                continue

            opt.step()
            sched.step()
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
                seen = args.batch_size * args.accum * world_size
                metric(
                    step=step,
                    loss=round(epoch_loss / max(epoch_batches, 1), 4),
                    step_time_ms=round(avg * 1000, 1),
                    samples_per_s=round(seen / avg, 3),
                    peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
                )

            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1 or stop:
            val_loss, val_acc, balanced, recalls = evaluate(
                model, val_loader, device, world_size, args.classes
            )
            improved = balanced > best_acc
            if improved:
                best_acc, best_epoch, since_best = balanced, epoch + 1, 0
                if test_loader is not None:
                    best_test = evaluate(
                        model, test_loader, device, world_size, args.classes
                    )
            else:
                since_best += 1
            if is_master:
                names = getattr(val_ds, "labels", tuple(str(i) for i in range(args.classes)))
                per = "  ".join(
                    f"{n}={r * 100:.0f}%" for n, r in zip(names, recalls) if r == r
                )
                log(f"epoch {epoch + 1}/{args.epochs}  "
                    f"train_loss {epoch_loss / max(epoch_batches, 1):.4f}  "
                    f"val_loss {val_loss:.4f}  acc {val_acc * 100:.1f}%  "
                    f"balanced {balanced * 100:.1f}%  [{per}]")
                metric(step=step, loss=round(val_loss, 4))

            # Only checkpoint improvements. This model peaked around epoch 24
            # and kept training to 40; saving every evaluation means the last
            # file on disk is the most overfit one, which is what you would
            # then deploy.
            if args.out_dir and improved:
                sharded = not args.no_fsdp and world_size > 1
                path = save_checkpoint(
                    model, opt, Path(args.out_dir), "best", sharded, is_master
                )
                if is_master and path:
                    log(f"  new best ({balanced * 100:.1f}%) -> {path}")

            if args.patience and since_best >= args.patience:
                if is_master:
                    log(f"no improvement for {since_best} evaluations; stopping "
                        f"(best {best_acc * 100:.1f}% at epoch {best_epoch})")
                stop = True

        if stop:
            break

    dist.barrier()
    if is_master and timed_steps:
        avg = timed_seconds / timed_steps
        log("=" * 60)
        log(f"optimizer steps    {step}")
        log(f"avg step time      {avg * 1000:.0f} ms")
        log(f"val  balanced acc  {best_acc * 100:.1f}% at epoch {best_epoch}  "
            f"(chance is {100 / args.classes:.0f}%)")
        if best_test is not None:
            _, t_acc, t_bal, t_rec = best_test
            names = getattr(val_ds, "labels", tuple(str(i) for i in range(args.classes)))
            per = "  ".join(f"{n}={r * 100:.0f}%" for n, r in zip(names, t_rec) if r == r)
            log(f"TEST balanced acc  {t_bal * 100:.1f}%  acc {t_acc * 100:.1f}%  [{per}]")
            log("                   (held out; never used to select anything)")
        log(f"peak VRAM per rank {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
        log("=" * 60)
    dist.destroy_process_group()
    if is_master:
        log("training completed cleanly")


if __name__ == "__main__":
    main()
