#!/usr/bin/env bash
# Produce every number the OS term project reports, from a clean checkout.
#
#   ./scripts/run_os_experiments.sh [OUTDIR]
#
# No GPU, no cluster, no network. The scheduling policies under test are the
# ones the controller runs -- `ferro-sim` links `ferro-sched` rather than
# reimplementing it -- so a result here is a statement about the shipped
# scheduler and not about a model of it.
#
# Writes into OUTDIR (default outputs/benchmarks/):
#
#   queue/       every queue policy over every workload
#   placement/   every placement policy over the heterogeneous cluster
#   dispatch/    opportunistic vs strict head-of-line dispatch
#   flat/        the same placement comparison with placement made invisible,
#                which is the control showing the execution model is doing the
#                work rather than the strategies differing by accident
#   trace/       one scheduler trace per policy, for Gantt charts
#
# Each directory gets results.csv (one row per job), summary.csv (one row per
# run) and runs.json (the same plus full metadata: seed, cluster, commit).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-outputs/benchmarks}"
SIM="${SIM:-./target/release/ferro-sim}"

if [[ ! -x "$SIM" ]]; then
    echo "building ferro-sim (release, so the scheduler-overhead numbers mean something)"
    cargo build --release -p ferro-sim
fi

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "note: the working tree is dirty, so runs will be tagged <commit>-dirty." >&2
    echo "      Commit first if these numbers are going into the write-up." >&2
fi

mkdir -p "$OUT"

echo
echo "=== 1/5  queue policies, every workload ======================================"
# One placement policy throughout, so the queue comparison is not confounded by
# placement decisions varying underneath it.
"$SIM" run --out "$OUT/queue" --placement performance

echo
echo "=== 2/5  placement policies, heterogeneous cluster ==========================="
# FIFO throughout, for the mirror-image reason.
"$SIM" run --out "$OUT/placement" --scenario E --queue fifo \
    --placement performance --placement first-fit --placement best-fit \
    --placement vram --placement topology

echo
echo "=== 3/5  the same, with placement made invisible (control) ==================="
# A flat execution model makes every placement run at the same speed. If the
# strategies still differed here, something other than placement would be
# driving the comparison above.
"$SIM" run --out "$OUT/flat" --scenario E --queue fifo --flat-execution \
    --placement performance --placement first-fit --placement best-fit \
    --placement vram --placement topology

echo
echo "=== 4/5  opportunistic vs strict dispatch ===================================="
# What FerroGrid's walk-the-whole-queue dispatcher is worth against the strict
# head-of-line FIFO it is usually assumed to be.
"$SIM" run --out "$OUT/dispatch" --scenario D --scenario A --queue fifo --with-strict

echo
echo "=== 5/5  scheduler traces ===================================================="
mkdir -p "$OUT/trace"
for policy in fifo priority aging fair-share sjf; do
    "$SIM" trace --scenario I --queue "$policy" --limit 0 > "$OUT/trace/I-$policy.txt"
done
echo "wrote $(ls "$OUT/trace" | wc -l) traces to $OUT/trace/"

echo
echo "=== done ===================================================================="
echo "Summary tables:"
for d in queue placement flat dispatch; do
    echo "  $OUT/$d/summary.csv"
done
echo
echo "Reproduce any single row with the seed and policies in its runs.json."
