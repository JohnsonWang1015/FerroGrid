#!/usr/bin/env python3
"""Pipeline-parallel training: the same model split across ranks by layer.

Why this exists: on a slow interconnect, pipeline parallelism moves far less
data than FSDP. FSDP all-gathers every parameter each step; PP only sends the
activations crossing a stage boundary. For the model below that is roughly
1.3 GB versus 8 MB per step -- two orders of magnitude, which is the
difference between usable and not on 1 GbE.

    ferro train --nodes 2 --gpus-per-node 1 -f python/examples/train_pp.py

Compare against FSDP at the same shape:

    ferro train --nodes 2 --gpus-per-node 1 -f python/examples/train_fsdp2.py

Nothing here reimplements pipelining: it is torch.distributed.pipelining,
launched by torchrun with the rendezvous FerroGrid computed.
"""

import argparse
import faulthandler
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.train_fsdp2 import Block  # noqa: E402  reuse the same block


def log(msg):
    print(msg, flush=True)


def metric(**kw):
    print("FERRO_METRIC " + json.dumps(kw), flush=True)


class Stage(nn.Module):
    """One pipeline stage: a slice of the layer stack.

    The embedding rides on the first stage and the head on the last, which is
    the usual split -- they are the two ends of the network and keeping them
    with their neighbours avoids an extra boundary crossing.
    """

    def __init__(self, args, first: bool, last: bool, n_layers: int):
        super().__init__()
        self.first, self.last = first, last
        d = args.d_model
        if first:
            self.tok = nn.Embedding(args.vocab, d)
            self.pos = nn.Embedding(args.seq_len, d)
        self.blocks = nn.ModuleList(Block(d, args.heads) for _ in range(n_layers))
        if last:
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, args.vocab, bias=False)
        self.register_buffer(
            "mask",
            torch.triu(torch.full((args.seq_len, args.seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        if self.first:
            t = x.shape[1]
            x = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        mask = self.mask[: x.shape[1], : x.shape[1]]
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.norm(x)) if self.last else x


def split_layers(total: int, stages: int) -> list[int]:
    """Layers per stage, remainder spread over the earliest stages.

    Even splits matter: the pipeline runs at the pace of its slowest stage.
    """
    base, extra = divmod(total, stages)
    return [base + (1 if i < extra else 0) for i in range(stages)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="global batch per step")
    p.add_argument("--microbatches", type=int, default=8,
                   help="more microbatches means a smaller pipeline bubble")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--schedule", choices=("1f1b", "gpipe"), default="1f1b")
    return p.parse_args()


def main():
    args = parse_args()

    # A hang inside a collective gives no clue on its own. Arm a watchdog that
    # prints every thread's stack and exits, so the stall shows up in the job
    # log instead of just sitting at 100% GPU forever.
    if (secs := os.environ.get("FERRO_STACK_TIMEOUT")):
        faulthandler.dump_traceback_later(int(secs), exit=True)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        log("pipeline parallelism needs at least 2 ranks")
        sys.exit(2)
    if not torch.cuda.is_available():
        log("CUDA unavailable in the container")
        sys.exit(2)
    if args.batch_size % args.microbatches:
        log(f"batch-size {args.batch_size} must divide by microbatches {args.microbatches}")
        sys.exit(2)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    is_last = rank == world_size - 1
    is_master = rank == 0

    counts = split_layers(args.layers, world_size)
    if is_master:
        log(f"FerroGrid pipeline-parallel | world_size={world_size} "
            f"stages={counts} microbatches={args.microbatches} "
            f"schedule={args.schedule} torch={torch.__version__}")

    torch.manual_seed(1234)
    # Each rank builds only its own slice, so the full model never has to fit
    # on one GPU -- that is the other half of what PP buys you.
    module = Stage(args, first=rank == 0, last=is_last, n_layers=counts[rank]).to(device)
    params = sum(p.numel() for p in module.parameters())
    log(f"[rank {rank}] stage with {counts[rank]} layer(s), {params / 1e6:.1f}M params "
        f"on {torch.cuda.get_device_name(local_rank)}")

    # Declare both input and output shapes so PipelineStage does no shape
    # inference at all.
    #
    # Neither inference mode survives this cluster: the init-time one (pass
    # `input_args` alone) exchanges metadata over NCCL's socket transport and
    # dies with "message truncated: receiving 8 bytes instead of 4", hanging
    # the run; the runtime one (pass neither) fails with "EOFError: Ran out of
    # input". Both work intra-node and break between nodes. Since the shapes
    # here are known analytically there is no reason to infer them.
    micro_bs = args.batch_size // args.microbatches
    hidden = (micro_bs, args.seq_len, args.d_model)

    stage_input = (
        torch.zeros(micro_bs, args.seq_len, dtype=torch.long, device=device)
        if rank == 0
        else torch.zeros(*hidden, device=device)
    )
    stage_output = (
        torch.zeros(micro_bs, args.seq_len, args.vocab, device=device)
        if is_last
        else torch.zeros(*hidden, device=device)
    )

    stage = PipelineStage(
        module,
        rank,
        world_size,
        device,
        input_args=(stage_input,),
        output_args=(stage_output,),
    )
    loss_fn = nn.CrossEntropyLoss()

    def stage_loss(output, target):
        return loss_fn(output.reshape(-1, args.vocab), target.reshape(-1))

    Sched = Schedule1F1B if args.schedule == "1f1b" else ScheduleGPipe
    schedule = Sched(stage, n_microbatches=args.microbatches, loss_fn=stage_loss)

    opt = torch.optim.AdamW(module.parameters(), lr=args.lr, fused=True)
    gen = torch.Generator(device="cuda").manual_seed(0)

    tokens_per_step = args.batch_size * args.seq_len
    timed_steps, timed_seconds = 0, 0.0
    torch.cuda.reset_peak_memory_stats(device)

    dist.barrier()
    if is_master:
        log("starting training loop")

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        # Only the ends of the pipeline touch data: rank 0 feeds inputs, the
        # last rank supplies targets and receives the losses.
        if rank == 0:
            idx = torch.randint(0, args.vocab, (args.batch_size, args.seq_len),
                                device=device, generator=gen)
            schedule.step(idx)
        elif is_last:
            targets = torch.randint(0, args.vocab, (args.batch_size, args.seq_len),
                                    device=device, generator=gen)
            losses = []
            schedule.step(target=targets, losses=losses)
        else:
            schedule.step()

        opt.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        if step > args.warmup_steps:
            timed_steps += 1
            timed_seconds += elapsed

        if is_last and step % args.log_every == 0:
            avg = timed_seconds / timed_steps if timed_steps else elapsed
            metric(
                step=step,
                loss=round(float(torch.stack(losses).mean()), 4) if losses else float("nan"),
                step_time_ms=round(avg * 1000, 2),
                samples_per_s=round(args.batch_size / avg, 2),
                tokens_per_s=round(tokens_per_step / avg, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
            )

    dist.barrier()
    if is_last and timed_steps:
        avg = timed_seconds / timed_steps
        log("=" * 62)
        log(f"schedule             {args.schedule}, {args.microbatches} microbatches")
        log(f"stages               {counts}")
        log(f"avg step time        {avg * 1000:.1f} ms")
        log(f"throughput           {tokens_per_step / avg:,.0f} tokens/s")
        log(f"peak VRAM per rank   {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")
        log("=" * 62)

    dist.destroy_process_group()
    if is_master:
        log("training completed cleanly")


if __name__ == "__main__":
    main()
