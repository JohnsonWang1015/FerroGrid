#!/usr/bin/env bash
# Measure FSDP2 scaling across the cluster and print a comparison table.
#
#   ./scripts/benchmark.sh <node-a> <node-b>
#
# Runs the same model at three shapes -- 1 GPU, 2 GPUs on one node, and
# 1 GPU on each of two nodes -- so the cost of crossing the network is
# visible next to the intra-node number.
set -euo pipefail
cd "$(dirname "$0")/.."

NODE_A="${1:?usage: benchmark.sh <node-a> <node-b>}"
NODE_B="${2:?missing second node id}"
FERRO="${FERRO:-target/release/ferro}"
CFG="--steps 12 --layers 8 --d-model 1024 --batch-size 8 --seq-len 512 --log-every 4 --warmup-steps 3"

run() {
    local name="$1" nodes="$2" per_node="$3"; shift 3
    local filters=() n
    for n in "$@"; do filters+=(--node "$n"); done

    local job
    job=$("$FERRO" --json train --nodes "$nodes" --gpus-per-node "$per_node" \
        "${filters[@]}" --name "$name" python/examples/train_fsdp2.py $CFG \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

    # Poll until the job leaves the running states.
    local phase=running
    for _ in $(seq 1 120); do
        sleep 5
        phase=$("$FERRO" --json job "$job" | python3 -c 'import json,sys; print(json.load(sys.stdin)["phase"])')
        case "$phase" in succeeded|failed|cancelled) break;; esac
    done

    "$FERRO" --json job "$job" | python3 -c "
import json,sys
d=json.load(sys.stdin); m=d['metrics']; p=d
print('%-22s %-10s %6s %12.0f %10.0f %9.2f %6d' % (
    '$name', d['phase'], m['step'], m['tokens_per_s'], m['step_time_ms'],
    m['peak_vram_gb'], len(d['nccl_errors'])))"
}

printf '%-22s %-10s %6s %12s %10s %9s %6s\n' \
    SHAPE PHASE STEP TOKENS/S STEP_MS VRAM_GB NCCLERR
run "1node-1gpu"  1 1 "$NODE_A"
run "1node-2gpu"  1 2 "$NODE_A"
run "2node-1gpu"  2 1 "$NODE_A" "$NODE_B"
