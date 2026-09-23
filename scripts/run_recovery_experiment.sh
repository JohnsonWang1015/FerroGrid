#!/usr/bin/env bash
# Measure what a controller crash costs, with and without persistent state.
#
#   ./scripts/run_recovery_experiment.sh [OUTDIR]
#
# This is the one experiment in the OS term project that `ferro-sim` cannot
# answer: there is no controller in a simulator to kill. So it runs the real
# binaries -- a controller and one agent -- fills the controller with state,
# kills it, and restarts it. Twice: once on `--state <db>` and once on
# `--no-state`, which is the in-memory registry as it was before persistence.
#
# SIGKILL, not SIGTERM. A clean shutdown is not the failure being studied, and
# write-behind (persistence.md §3) has its worst case precisely when the
# process is not given the chance to finish what it queued.
#
# Three numbers per arm, per docs/os_term_project/experiments.md §69:
#
#   recovery time             CONTROLLER_RECOVERED -> RECONCILED in the event
#                             log, and wall-clock from process start to each
#   lost job count            jobs known before the kill, minus jobs known after
#   incorrect allocations     GPUs whose allocated_job_id names a job the
#                             controller now calls terminal or does not know
#
# ONE GPU IS ENOUGH, and this box has one. A job submitted with `--wait` that
# cannot be placed exercises the entire persistence path -- job record, queue
# order, the original request -- without touching hardware, so a batch of those
# is the bulk of the measurement. What they never reach is reconciliation,
# which is about jobs the agents are actually running; that is what the single
# running job is for. With no GPU free the script runs the queued half alone
# and says so in every line it prints and every file it writes.
#
# Knobs, all environment variables:
#   QUEUED_JOBS       how many queued jobs to submit (default 20)
#   RECONCILE_WINDOW  --reconcile-window-secs for both arms (default 30, the
#                     shipped default, so the number means something)
#   CTRL_PORT         controller port (default 7170)
#   AGENT_PORT        agent port (default 7171)
#   MASTER_PORT       base torchrun rendezvous port (default 29500; the
#                     in-memory arm uses this + 100)
#
# Writes into OUTDIR (default outputs/recovery/):
#
#   summary.csv   one row per arm, the three metrics plus how they were taken
#   runs.json     the same, plus every before/after job list and the commit
#   logs/         the controller and agent logs each arm actually produced,
#                 because a recovery time is a claim about a process and the
#                 process wrote down when it did what
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-outputs/recovery}"
QUEUED_JOBS="${QUEUED_JOBS:-20}"
RECONCILE_WINDOW="${RECONCILE_WINDOW:-30}"
CTRL_PORT="${CTRL_PORT:-7170}"
AGENT_PORT="${AGENT_PORT:-7171}"
# Base rendezvous port. Each arm gets its own so the two arms cannot collide
# on it, and both are checked free before anything is launched: rank 0 dying of
# EADDRINUSE looks exactly like a job lost to the crash, and this experiment
# exists to count jobs lost to the crash.
MASTER_PORT="${MASTER_PORT:-29500}"

# The window is the design's own deadline for deciding. Waiting it out plus
# slack before calling RECONCILED missing keeps a slow machine from being
# recorded as a controller that failed to reconcile.
DEADLINE=$(( RECONCILE_WINDOW + 30 ))
# Two agent heartbeats. GPU ownership is the agent's to report (persistence.md
# §2), so the allocation table is only worth reading once it has reported since
# reconciliation finished.
SETTLE=7
# Longer than the whole experiment, so the running job never ends on its own
# and turn a measurement of recovery into a measurement of job duration.
HOLD_SECONDS=1800

CTRL=./target/release/ferro-controller
AGENT=./target/release/ferro-agent
FERRO=./target/release/ferro
ENDPOINT="http://127.0.0.1:$CTRL_PORT"

for bin in "$CTRL" "$AGENT" "$FERRO"; do
    if [[ ! -x "$bin" ]]; then
        echo "building release binaries (a debug controller would put its own"
        echo "startup cost inside the recovery time)"
        cargo build --release -p ferro-controller -p ferro-agent -p ferro-cli
        break
    fi
done

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    COMMIT="$COMMIT-dirty"
    echo "note: the working tree is dirty, so runs are tagged $COMMIT." >&2
fi

for port in "$CTRL_PORT" "$AGENT_PORT" "$MASTER_PORT" $(( MASTER_PORT + 100 )); do
    # Under a timeout, because a bare `/dev/tcp` probe can hang: on WSL2 a SYN
    # to a closed loopback port is sometimes dropped rather than refused, and
    # the connect then blocks forever. Only a completed connection proves the
    # port is taken -- a refusal and a timeout both mean nothing answered.
    if timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
        echo "port $port is already in use." >&2
        echo "Controller and agent: CTRL_PORT/AGENT_PORT. Rendezvous: MASTER_PORT" >&2
        echo "(this arm's and the next arm's, which is MASTER_PORT + 100)." >&2
        echo "A stale torchrun from an interrupted run is the usual holder." >&2
        exit 1
    fi
done

TMP="$(mktemp -d)"
CTRL_PID=""
AGENT_PID=""
# Named after this invocation so cleanup can find its own leftovers and only
# its own. `--no-docker` torchrun puts its workers in a fresh session, so
# neither the agent's SIGKILL of torchrun nor a process-group kill reaches
# them; one that outlives the run holds the rendezvous port and the next run's
# rank 0 dies at launch with EADDRINUSE. That failure looks exactly like a job
# lost to the crash, which is the one thing this script must not get wrong.
JOB_SCRIPT="recovery_job_$$.py"

stop_controller() {
    if [[ -n "$CTRL_PID" ]]; then
        kill -9 "$CTRL_PID" 2>/dev/null || true
        wait "$CTRL_PID" 2>/dev/null || true
        CTRL_PID=""
    fi
}

stop_agent() {
    if [[ -n "$AGENT_PID" ]]; then
        # The agent runs under setsid so this can signal the whole group. The
        # torchrun it launched is a child, not a supervised subprocess of ours:
        # killing the agent alone leaves a python holding the GPU for the next
        # arm to trip over. SIGINT first, which is the agent's own "stop
        # everything you started" path.
        kill -INT -- "-$AGENT_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$AGENT_PID" 2>/dev/null || break
            sleep 0.25
        done
        kill -9 -- "-$AGENT_PID" 2>/dev/null || true
        wait "$AGENT_PID" 2>/dev/null || true
        AGENT_PID=""
    fi
    stop_job_processes
}

# The backstop for the workers that left the process group. Matched on this
# invocation's own script name and restricted to this user, so it cannot reach
# anybody else's python.
stop_job_processes() {
    pkill -9 -u "$(id -u)" -f "$JOB_SCRIPT" 2>/dev/null || true
}

cleanup() {
    local code=$?
    stop_controller
    stop_agent
    rm -rf "$TMP"
    exit "$code"
}
trap cleanup EXIT INT TERM

mkdir -p "$TMP/ws" "$OUT/logs"

# Holds its allocation and does nothing else. What is being measured is the
# controller's bookkeeping, not throughput -- but it claims a little VRAM
# anyway, so the GPU it is recorded against is a GPU it is genuinely on.
cat > "$TMP/ws/$JOB_SCRIPT" <<'PY'
import os
import time

try:
    import torch

    if torch.cuda.is_available():
        hold = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
        print(f"holding {hold.numel() >> 20} MiB on cuda:0", flush=True)
    else:
        print("no CUDA device visible; holding the allocation only", flush=True)
except Exception as exc:  # torch missing or the device refused us
    print(f"no CUDA tensor ({exc}); holding the allocation only", flush=True)

print('FERRO_METRIC {"step": 1}', flush=True)
time.sleep(int(os.environ.get("HOLD_SECONDS", "900")))
PY

ferro_json() { "$FERRO" --controller "$ENDPOINT" --json "$@"; }

jq_count() { python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'; }

start_controller() {  # start_controller <arm> <log-suffix>
    local arm="$1" tag="$2" args=()
    if [[ "$arm" == persistent ]]; then
        args=(--state "$TMP/controller.db")
    else
        args=(--no-state)
    fi
    "$CTRL" --bind "127.0.0.1:$CTRL_PORT" \
        --master-port "$(arm_master_port "$arm")" \
        --min-free-vram-gib 1 \
        --reconcile-window-secs "$RECONCILE_WINDOW" \
        "${args[@]}" > "$TMP/$arm-controller-$tag.log" 2>&1 &
    CTRL_PID=$!
}

arm_master_port() {
    if [[ "$1" == persistent ]]; then echo "$MASTER_PORT"; else echo $(( MASTER_PORT + 100 )); fi
}

wait_for_controller() {
    local deadline=$(( SECONDS + 60 ))
    while (( SECONDS < deadline )); do
        if ferro_json nodes >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    echo "controller never answered on $ENDPOINT" >&2
    return 1
}

wait_for_node() {
    local deadline=$(( SECONDS + 60 )) n
    while (( SECONDS < deadline )); do
        n="$(ferro_json nodes 2>/dev/null | jq_count 2>/dev/null || echo 0)"
        if [[ "$n" -gt 0 ]]; then
            return 0
        fi
        sleep 0.5
    done
    echo "no agent registered within 60s" >&2
    return 1
}

snapshot() {  # snapshot <arm> <before|after>
    ferro_json jobs --limit 0     > "$TMP/$1-$2-jobs.json"
    ferro_json queue              > "$TMP/$1-$2-queue.json"
    ferro_json gpu                > "$TMP/$1-$2-gpu.json"
    ferro_json events --limit 0   > "$TMP/$1-$2-events.json"
}

count_reconciled() {
    ferro_json events --kind RECONCILED --limit 0 2>/dev/null | jq_count 2>/dev/null || echo 0
}

job_phase() {  # job_phase <job-id>
    ferro_json jobs --limit 0 | python3 -c \
        'import json,sys; jid=sys.argv[1]; print(next((j["phase"] for j in json.load(sys.stdin) if j["job_id"] == jid), "gone"))' "$1"
}

# The step the job has reported. `running` on its own is not proof of life: the
# controller says running the moment it has dispatched, which is before the
# agent has execed torchrun, let alone before python has imported torch. The
# job prints one FERRO_METRIC once it is past CUDA init, and that round trip
# -- job to agent to controller -- is the first thing that could not have
# happened unless the job really is up.
job_step() {  # job_step <job-id>
    ferro_json jobs --limit 0 | python3 -c \
        'import json,sys; jid=sys.argv[1]; print(next((j["step"] for j in json.load(sys.stdin) if j["job_id"] == jid), 0))' "$1"
}

gpus_held_by() {  # gpus_held_by <job-id>
    ferro_json gpu | python3 -c \
        'import json,sys; jid=sys.argv[1]; print(sum(1 for g in json.load(sys.stdin) if g["allocated_job_id"] == jid))' "$1"
}

# Seconds between two `date +%s.%N` stamps, to two decimals. Bash has no
# floating point and the interesting part of a restart is sub-second.
elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", b - a }'; }

run_arm() {  # run_arm <persistent|in-memory>
    local arm="$1"
    echo
    echo "=== arm: $arm ==============================================================="

    start_controller "$arm" pre
    wait_for_controller

    FERRO_CONTROLLER="$ENDPOINT" setsid "$AGENT" \
        --bind "127.0.0.1:$AGENT_PORT" \
        --advertise "127.0.0.1:$AGENT_PORT" \
        --workspace "$TMP/ws" \
        --no-docker > "$TMP/$arm-agent.log" 2>&1 &
    AGENT_PID=$!
    wait_for_node

    local free_gpus max_gpus
    free_gpus="$(ferro_json nodes | python3 -c 'import json,sys; print(sum(n["free_gpus"] for n in json.load(sys.stdin)))')"
    max_gpus="$(ferro_json nodes | python3 -c 'import json,sys; ns=json.load(sys.stdin); print(max((n["gpus"] for n in ns), default=0))')"

    # A running job, if there is anywhere to put one. This is the only part of
    # the experiment that reaches reconciliation: queued jobs are restored, but
    # nothing has to be claimed back for them.
    # `runner_id` is whatever was submitted; `running_job` is what was verified
    # to be genuinely up. They come apart exactly when the interesting failure
    # happens, and the logs are wanted under the first name, not the second.
    local running_job="" runner_id="" gpu_half=false skip_reason=""
    if (( free_gpus == 0 )); then
        skip_reason="no free GPU on any registered node"
    else
        ferro_json train "$JOB_SCRIPT" \
            --nodes 1 --gpus-per-node 1 \
            --env "HOLD_SECONDS=$HOLD_SECONDS" \
            --name recovery-runner > "$TMP/$arm-runner.json"
        runner_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$TMP/$arm-runner.json")"
        running_job="$runner_id"
        echo "running job $running_job submitted; waiting for it to report a step"
        # All three, not any: a job with a card assigned but no python behind it
        # is a launch failure, and killing the controller on top of one measures
        # a harness bug rather than recovery.
        local deadline=$(( SECONDS + 180 )) held=0 step=0 phase=""
        while (( SECONDS < deadline )); do
            phase="$(job_phase "$running_job")"
            step="$(job_step "$running_job")"
            held="$(gpus_held_by "$running_job")"
            if [[ "$phase" == running && "$step" -ge 1 && "$held" -gt 0 ]]; then break; fi
            case "$phase" in
                failed|succeeded|cancelled|gone) break ;;
            esac
            sleep 1
        done
        if [[ "$phase" != running || "$step" -lt 1 || "$held" -eq 0 ]]; then
            skip_reason="the running job reached phase '$phase', step $step, $held GPU(s) held"
            running_job=""
        else
            echo "running job is up: step $step, $held GPU(s) held"
        fi
    fi
    if [[ -n "$skip_reason" ]]; then
        echo "!! the running-job half is SKIPPED for this arm: $skip_reason"
        echo "!! only the queued half of the experiment is being measured here."
    else
        gpu_half=true
    fi

    # Queued jobs ask for one more GPU than any node has, so they queue for a
    # reason that has nothing to do with the running job. Queueing them behind
    # it instead would make the queued half depend on the GPU half, and the two
    # halves are meant to be independent -- not least because the GPU half is
    # the one that can be skipped.
    local want=$(( max_gpus + 1 )) i
    echo "submitting $QUEUED_JOBS queued job(s), each asking for $want GPU(s) on a node"
    for i in $(seq 1 "$QUEUED_JOBS"); do
        ferro_json train "$JOB_SCRIPT" \
            --nodes 1 --gpus-per-node "$want" --wait \
            --name "recovery-q$i" > "$TMP/$arm-q$i.json"
    done

    # The job has had a minute or two of queue submissions to die in. What the
    # arm is recorded as having tested is what was true when the kill landed,
    # not what was true when the job was launched.
    if [[ -n "$running_job" ]]; then
        local phase_now
        phase_now="$(job_phase "$running_job")"
        if [[ "$phase_now" != running ]]; then
            skip_reason="the running job fell to '$phase_now' before the kill"
            gpu_half=false
            running_job=""
            echo "!! the running-job half is SKIPPED for this arm: $skip_reason"
        fi
    fi

    # Taken now, not at the end: the log ring is not persisted (persistence.md
    # §6), so after the restart `ferro logs` has nothing for a restored job and
    # nothing at all for a forgotten one. Now is the only time these exist, and
    # a running-job half that was skipped is exactly when they are wanted.
    if [[ -n "$runner_id" ]]; then
        "$FERRO" --controller "$ENDPOINT" logs "$runner_id" \
            > "$OUT/logs/$arm-runner-$runner_id.log" 2>&1 || true
    fi

    snapshot "$arm" before
    local before_reconciled
    before_reconciled="$(count_reconciled)"
    echo "before the kill: $(jq_count < "$TMP/$arm-before-jobs.json") job(s) known"

    echo "SIGKILL to the controller (pid $CTRL_PID)"
    kill -9 "$CTRL_PID"
    wait "$CTRL_PID" 2>/dev/null || true
    CTRL_PID=""

    local t0 t_serving t_reconciled=""
    t0="$(date +%s.%N)"
    start_controller "$arm" post
    wait_for_controller
    t_serving="$(date +%s.%N)"
    echo "controller answering again after $(elapsed "$t0" "$t_serving")s"

    local deadline=$(( SECONDS + DEADLINE ))
    while (( SECONDS < deadline )); do
        if [[ "$(count_reconciled)" -gt "$before_reconciled" ]]; then
            t_reconciled="$(date +%s.%N)"
            break
        fi
        sleep 0.2
    done
    if [[ -n "$t_reconciled" ]]; then
        echo "RECONCILED after $(elapsed "$t0" "$t_reconciled")s"
    else
        echo "no RECONCILED event within ${DEADLINE}s"
    fi

    # Let the agent report at least twice more: the GPU allocation table this
    # reads next is the agent's, not the controller's memory of it.
    sleep "$SETTLE"
    snapshot "$arm" after

    {
        printf '{"arm":"%s","commit":"%s","queued_jobs":%s,"reconcile_window_s":%s,' \
            "$arm" "$COMMIT" "$QUEUED_JOBS" "$RECONCILE_WINDOW"
        printf '"gpu_half":%s,"gpu_half_skipped_because":"%s","running_job":"%s","free_gpus_at_start":%s,' \
            "$gpu_half" "$skip_reason" "$running_job" "$free_gpus"
        printf '"wall_to_serving_s":%s,' "$(elapsed "$t0" "$t_serving")"
        if [[ -n "$t_reconciled" ]]; then
            printf '"wall_to_reconciled_s":%s}' "$(elapsed "$t0" "$t_reconciled")"
        else
            printf '"wall_to_reconciled_s":null}'
        fi
    } > "$TMP/$arm-timing.json"

    stop_agent
    stop_controller
    # Colour stripped on the way out: these are kept as evidence and read in a
    # diff as often as in a terminal.
    local log
    for log in "$arm-controller-pre" "$arm-controller-post" "$arm-agent"; do
        sed 's/\x1b\[[0-9;]*m//g' "$TMP/$log.log" > "$OUT/logs/$log.log"
    done
    # The persistent arm gets a fresh database per invocation, or a second run
    # would be measuring the first run's leftovers.
    rm -f "$TMP/controller.db" "$TMP/controller.db-wal" "$TMP/controller.db-shm"
}

echo "=== FerroGrid controller-recovery experiment (RQ3) ==========================="
echo "commit $COMMIT, $QUEUED_JOBS queued job(s), reconcile window ${RECONCILE_WINDOW}s"

run_arm persistent
run_arm in-memory

python3 - "$TMP" "$OUT" <<'PY'
import csv
import json
import pathlib
import re
import sys

tmp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ARMS = ["persistent", "in-memory"]
TERMINAL = {"succeeded", "failed", "cancelled"}


def load(arm, name):
    return json.loads((tmp / f"{arm}-{name}.json").read_text())


def recovery_events(events):
    """The restart's own two lines: the last start, and the reconcile that
    closed it. Taking the *last* start matters -- the persistent arm reads the
    pre-kill timeline back off disk, so the log holds both boots."""
    start = None
    for e in events:
        if e["kind"] in ("CONTROLLER_RECOVERED", "CONTROLLER_STARTED"):
            start = e
    reconciled = next(
        (e for e in events if start and e["kind"] == "RECONCILED" and e["id"] > start["id"]),
        None,
    )
    return start, reconciled


rows, runs = [], []
for arm in ARMS:
    timing = load(arm, "timing")
    before_jobs = load(arm, "before-jobs")
    after_jobs = load(arm, "after-jobs")
    before_queue = load(arm, "before-queue")
    after_queue = load(arm, "after-queue")
    after_gpus = load(arm, "after-gpu")
    after_events = load(arm, "after-events")

    before_ids = [j["job_id"] for j in before_jobs]
    after = {j["job_id"]: j for j in after_jobs}
    lost = [i for i in before_ids if i not in after]

    # An allocation the controller cannot account for: the card says it is
    # working for somebody, and the controller either has never heard of them
    # or has already written them off.
    bad_alloc = []
    for g in after_gpus:
        jid = g["allocated_job_id"]
        if not jid:
            continue
        job = after.get(jid)
        if job is None:
            bad_alloc.append({"node_id": g["node_id"], "index": g["index"],
                              "allocated_job_id": jid, "why": "unknown job"})
        elif job["phase"] in TERMINAL:
            bad_alloc.append({"node_id": g["node_id"], "index": g["index"],
                              "allocated_job_id": jid, "why": f"job is {job['phase']}"})

    start, reconciled = recovery_events(after_events)
    log_recovery_s = (reconciled["unix_s"] - start["unix_s"]) if reconciled else None
    # What the controller timed the window as, against its own monotonic clock.
    # Worth carrying separately from the two wall-clock numbers: a host whose
    # monotonic clock lags wall time -- WSL2 does, by about 12% over 30s --
    # makes every wall measurement here longer than the window it configured,
    # and only this number says what the controller itself waited.
    m = re.search(r"in (\d+)s", reconciled["detail"]) if reconciled else None
    self_timed_s = int(m.group(1)) if m else None

    before_phase = {j["job_id"]: j["phase"] for j in before_jobs}
    changed = {i: [before_phase[i], after[i]["phase"]]
               for i in before_ids if i in after and after[i]["phase"] != before_phase[i]}

    q_before = [j["job_id"] for j in sorted(before_queue, key=lambda j: j["position"])]
    q_after = [j["job_id"] for j in sorted(after_queue, key=lambda j: j["position"])]
    # Order is only comparable over the jobs that are still queued at all.
    q_kept = [i for i in q_before if i in set(q_after)]
    # An arm that kept nothing kept nothing in order either. Reporting True
    # there would read as "the queue survived" when the queue is empty.
    order_held = (q_kept == [i for i in q_after if i in set(q_before)]) if q_kept else None

    rows.append({
        "arm": arm,
        "commit": timing["commit"],
        "jobs_before": len(before_ids),
        "jobs_after": len(after_jobs),
        "lost_jobs": len(lost),
        "failed_by_reconcile": sum(1 for e in after_events
                                   if start and e["kind"] == "JOB_FAILED" and e["id"] > start["id"]),
        "incorrect_allocations": len(bad_alloc),
        "recovery_time_log_s": "" if log_recovery_s is None else log_recovery_s,
        "window_self_timed_s": "" if self_timed_s is None else self_timed_s,
        "wall_to_serving_s": timing["wall_to_serving_s"],
        "wall_to_reconciled_s": "" if timing["wall_to_reconciled_s"] is None
                                else timing["wall_to_reconciled_s"],
        "reconciled": bool(reconciled),
        "queue_order_held": "" if order_held is None else order_held,
        "queued_before": len(q_before),
        "queued_after": len(q_after),
        "running_job_half": timing["gpu_half"],
        "reconcile_window_s": timing["reconcile_window_s"],
    })
    runs.append({
        **rows[-1],
        "running_job_half_skipped_because": timing["gpu_half_skipped_because"] or None,
        "reconciled_detail": reconciled["detail"] if reconciled else None,
        "start_event": start["kind"] if start else None,
        "running_job": timing["running_job"] or None,
        "lost_job_ids": lost,
        "phase_changes": changed,
        "incorrect_allocations_detail": bad_alloc,
        "before_jobs": before_jobs,
        "after_jobs": after_jobs,
        "queue_before": q_before,
        "queue_after": q_after,
        "gpus_after": after_gpus,
    })

out.mkdir(parents=True, exist_ok=True)
with (out / "summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")

print()
print("=== results =================================================================")
head = f"{'metric':<28}" + "".join(f"{a:>16}" for a in ARMS)
print(head)
print("-" * len(head))


def line(label, key, fmt=str):
    print(f"{label:<28}" + "".join(f"{fmt(r[key]):>16}" for r in rows))


def dash(v):
    return "n/a" if v == "" else str(v)


line("jobs known before kill", "jobs_before")
line("jobs known after restart", "jobs_after")
line("lost jobs", "lost_jobs")
line("incorrect allocations", "incorrect_allocations")
line("recovery time (event log)", "recovery_time_log_s", dash)
line("  window, controller-timed", "window_self_timed_s", dash)
line("wall: start -> serving", "wall_to_serving_s", lambda v: f"{v}s")
line("wall: start -> RECONCILED", "wall_to_reconciled_s", dash)
line("reconciled at all", "reconciled")
line("queue order held", "queue_order_held", dash)
line("running-job half ran", "running_job_half")
print()
for r, run in zip(rows, runs):
    if not r["running_job_half"]:
        print(f"!! {r['arm']}: PARTIAL. The running-job half was skipped --")
        print(f"!! {run['running_job_half_skipped_because']}.")
        print("!! Nothing needed claiming back, so this arm's recovery time and")
        print("!! incorrect-allocation count cover queued jobs only.")
    if run["reconciled_detail"]:
        print(f"{r['arm']}: {run['reconciled_detail']}")
    else:
        print(f"{r['arm']}: no RECONCILED event -- nothing was restored to reconcile")
print()
print(f"wrote {out/'summary.csv'} and {out/'runs.json'}")
PY
