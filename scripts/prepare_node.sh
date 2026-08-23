#!/usr/bin/env bash
# Verify a GPU server can run FerroGrid jobs and pre-pull the training image.
#
#   ./scripts/prepare_node.sh <ssh-host> [image]
set -euo pipefail
HOST="${1:?usage: prepare_node.sh <ssh-host> [image]}"
IMAGE="${2:-pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime}"

echo "==> [$HOST] checking prerequisites"
ssh "$HOST" "set -e
    command -v nvidia-smi >/dev/null || { echo 'FAIL: nvidia-smi missing'; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    command -v docker >/dev/null || { echo 'FAIL: docker missing'; exit 1; }
    docker info >/dev/null 2>&1 || { echo 'FAIL: cannot talk to docker (add \$USER to the docker group)'; exit 1; }
    docker info 2>/dev/null | grep -q nvidia || { echo 'FAIL: nvidia container runtime not registered'; exit 1; }
    echo 'prerequisites OK'"

echo "==> [$HOST] pulling $IMAGE (may take a while)"
ssh "$HOST" "docker pull '$IMAGE' >/dev/null && echo 'image ready'"

echo "==> [$HOST] verifying GPUs are visible inside a container"
ssh "$HOST" "docker run --rm --gpus all '$IMAGE' python -c \"
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA not visible in container'
from torch.distributed.fsdp import fully_shard  # FSDP2 must exist
print('FSDP2 available')\""
echo "==> [$HOST] ready"
