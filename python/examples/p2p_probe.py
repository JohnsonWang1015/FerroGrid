"""Minimal cross-node NCCL point-to-point probe.

Collectives (all_reduce) and point-to-point (send/recv) take different NCCL
code paths. FSDP only needs the former; pipeline parallelism needs the latter.
This tells you which of the two your fabric actually supports.
"""
import os, time, torch, torch.distributed as dist

lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dev = torch.device("cuda", lr)
dist.init_process_group("nccl", device_id=dev)
rank, world = dist.get_rank(), dist.get_world_size()

def stage(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        torch.cuda.synchronize(dev)
        print(f"[rank {rank}] {name}: OK ({(time.perf_counter()-t0)*1e3:.0f} ms)", flush=True)
    except Exception as e:
        print(f"[rank {rank}] {name}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

x = torch.ones(1024, 1024, device=dev)
stage("all_reduce (collective)", lambda: dist.all_reduce(x))

def p2p():
    if rank == 0:
        dist.send(x, dst=1)
        dist.recv(x, src=world - 1)
    elif rank == world - 1:
        dist.recv(x, src=rank - 1)
        dist.send(x, dst=0)
print(f"[rank {rank}] starting send/recv...", flush=True)
stage("send/recv (point-to-point)", p2p)

# What torch.distributed.pipelining actually uses is the batched form, not
# plain send/recv, so probe that separately.
def batched():
    ops = []
    if rank == 0:
        ops.append(dist.P2POp(dist.isend, x, 1))
    elif rank == world - 1:
        ops.append(dist.P2POp(dist.irecv, x, rank - 1))
    if ops:
        for w in dist.batch_isend_irecv(ops):
            w.wait()
print(f"[rank {rank}] starting batch_isend_irecv...", flush=True)
stage("batch_isend_irecv (what pipelining uses)", batched)

dist.barrier()
if rank == 0:
    print("probe complete", flush=True)
dist.destroy_process_group()
