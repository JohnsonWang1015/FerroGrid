#!/usr/bin/env bash
# Keep this machine at the tip of origin/main -- once CI has passed on it --
# and roll the build out to the controller and every node's agent.
#
#   ./scripts/auto_update.sh             one pass
#   ./scripts/auto_update.sh --dry-run   say what a pass would do, change nothing
#
# Normally run every couple of minutes by the ferro-autoupdate timer that
# scripts/install_autoupdate.sh installs. Each pass:
#
#   1. fetches origin/main and, if every GitHub check on the new tip passed,
#      fast-forwards this checkout to it;
#   2. builds that commit -- native `ferro`/`ferro-controller`, portable agent --
#      unless it has already been built;
#   3. restarts the controller if it runs a binary other than the one built,
#      unless a job is pending, launching or running;
#   4. installs the portable agent on every healthy node whose agent runs
#      something else, unless that node holds a job or anything is queued or
#      starting and could be dispatched into the restart. An agent that does
#      not come back registered is rolled back to the binary it replaced.
#
# Steps 3 and 4 compare what is *running* (the sha256 of /proc/<pid>/exe) with
# what step 2 built, instead of remembering what was done: a node that was
# busy, offline or unreachable is picked up by a later pass, and a pass that
# died half way is finished by the next one.
#
# It leaves the checkout alone when it is not on main, has uncommitted changes
# to tracked files, or has commits origin does not -- that is somebody working,
# and building their tree would ship code CI never saw. For the same reason it
# only rolls out binaries it built itself: a `cargo build --release` by hand
# leaves the running services where they are.
#
# Environment:
#   FERRO_UPDATE_BRANCH   branch to follow (default main)
#   FERRO_CONTROLLER      controller endpoint (default http://127.0.0.1:7070)
#
# State -- the last build, a commit that failed to build, nodes where a build
# failed to start -- lives in ${XDG_STATE_HOME:-~/.local/state}/ferrogrid/autoupdate.
set -euo pipefail

say()  { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
sha()  { sha256sum "$1" | cut -d' ' -f1; }

ferro_json() { "$FERRO" --json "$@"; }

# key -> value from the manifest step 2 writes after a successful build.
built() { awk -v k="$1" '$1 == k {print $2}' "$STATE/built" 2>/dev/null; }

# The checkout is ours to move only while nobody is working in it.
checkout_is_ours() {
    local branch
    branch="$(git symbolic-ref --quiet --short HEAD || echo 'a detached HEAD')"
    [[ "$branch" == "$BRANCH" ]] || { WHY="it is on $branch, not $BRANCH"; return 1; }
    git diff --quiet HEAD -- || { WHY="it has uncommitted changes to tracked files"; return 1; }
}

# <commit> -> success | pending | failure. Every check run GitHub has for the
# commit must have finished and passed; none at all means CI has not started.
ci_status() {
    GH_NO_UPDATE_NOTIFIER=1 gh api "repos/$SLUG/commits/$1/check-runs?per_page=100" --jq '
        .check_runs
        | if length == 0 or any(.[]; .status != "completed") then "pending"
          elif all(.[]; .conclusion == "success" or .conclusion == "skipped" or .conclusion == "neutral") then "success"
          else "failure" end' 2>/dev/null
}

# Number of jobs in any of the given phases, plus any a restarted controller
# is still reconciling.
count_jobs() {
    ferro_json jobs --limit 0 | python3 -c '
import json, sys
want = set(sys.argv[1:])
print(sum(1 for j in json.load(sys.stdin) if j.get("phase") in want or j.get("reconciling")))' "$@"
}

healthy_nodes() {
    ferro_json nodes | python3 -c 'import json, sys; print(sum(1 for n in json.load(sys.stdin) if n.get("healthy")))'
}

# node_id <TAB> user@host <TAB> GPUs holding a job <TAB> healthy|stale|no-user
node_table() {
    local nodes gpus
    nodes="$(ferro_json nodes)" && gpus="$(ferro_json gpu)" || return 1
    python3 - "$nodes" "$gpus" <<'PY'
import json, sys
nodes, gpus = json.loads(sys.argv[1]), json.loads(sys.argv[2])
held = {}
for g in gpus:
    if g.get("allocated_job_id"):
        held[g["node_id"]] = held.get(g["node_id"], 0) + 1
for n in nodes:
    # Same target `ferro sync` uses: the login user the agent reported, at its
    # management address.
    host = n["address"].rsplit(":", 1)[0]
    state = "no-user" if not n.get("user") else "healthy" if n.get("healthy") else "stale"
    print(f'{n["node_id"]}\t{n.get("user", "")}@{host}\t{held.get(n["node_id"], 0)}\t{state}')
PY
}

# ---------------------------------------------------------------------------
# 1. Follow origin
# ---------------------------------------------------------------------------

advance_checkout() {
    git fetch --quiet origin "$BRANCH" || { note "fetch failed; carrying on with what is here"; return 0; }
    local head remote ci
    head="$(git rev-parse HEAD)"
    remote="$(git rev-parse FETCH_HEAD)"
    if [[ "$head" == "$remote" ]]; then
        note "checkout is at origin/$BRANCH (${head:0:7})"
        return 0
    fi
    if ! git merge-base --is-ancestor HEAD "$remote"; then
        note "checkout has commits origin/$BRANCH does not; leaving it alone"
        return 0
    fi
    ci="$(ci_status "$remote")" || ci=unknown
    case "$ci" in
        success) ;;
        pending) note "origin/$BRANCH is at ${remote:0:7}, still waiting for CI"; return 0 ;;
        failure) note "CI failed on ${remote:0:7}; not deploying it"; return 0 ;;
        *)       note "cannot read CI results for ${remote:0:7} (is gh logged in?); trying again next pass"; return 0 ;;
    esac
    say "fast-forwarding ${head:0:7} -> ${remote:0:7}"
    git log --oneline "$head..$remote" | sed 's/^/    /'
    if (( DRY_RUN )); then
        note "(dry run) not merging"
        return 0
    fi
    git merge --quiet --ff-only "$remote" \
        || note "fast-forward failed -- an untracked file in the way? trying again next pass"
}

# ---------------------------------------------------------------------------
# 2. Build what was checked out
# ---------------------------------------------------------------------------

build_head() {
    local head ci
    head="$(git rev-parse HEAD)"
    [[ "$head" == "$(built commit)" ]] && return 0
    if [[ "$head" == "$(cat "$STATE/build-failed" 2>/dev/null)" ]]; then
        note "${head:0:7} failed to build earlier; waiting for a new commit"
        return 0
    fi
    # The checkout can reach a commit without step 1 -- a first run, or a
    # `git pull` by hand -- so CI is asked again here rather than assumed.
    ci="$(ci_status "$head")" || ci=unknown
    if [[ "$ci" != success ]]; then
        note "not building ${head:0:7}: CI is $ci"
        return 0
    fi
    say "building ${head:0:7}"
    if (( DRY_RUN )); then
        note "(dry run) would build native binaries and the portable agent"
        return 0
    fi
    if cargo build --release --workspace && ./scripts/build.sh portable; then
        {
            echo "commit $head"
            echo "ferro-controller $(sha "$CTRL_BIN")"
            echo "ferro-agent $(sha "$AGENT_BIN")"
        } >"$STATE/built.tmp"
        mv "$STATE/built.tmp" "$STATE/built"
        rm -f "$STATE/build-failed"
        note "built ${head:0:7}"
    else
        echo "$head" >"$STATE/build-failed"
        note "build of ${head:0:7} failed; the running services keep their binaries"
    fi
}

# ---------------------------------------------------------------------------
# 3. The controller
# ---------------------------------------------------------------------------

converge_controller() {
    local want exec_path pid active before now
    want="$(built ferro-controller)"
    if [[ "$(sha "$CTRL_BIN" 2>/dev/null)" != "$want" ]]; then
        note "target/release/ferro-controller is not the binary this updater built (rebuilt by hand?); leaving the controller alone"
        return 0
    fi
    # A unit that runs a copy elsewhere would restart straight back onto the
    # old binary, and every later pass would do it again.
    exec_path="$(systemctl --user show -p ExecStart --value ferro-controller 2>/dev/null \
        | sed -n 's/.*path=\([^ ;]*\).*/\1/p')"
    if [[ -z "$exec_path" ]]; then
        note "no ferro-controller user unit; not managing the controller"
        return 0
    fi
    if [[ "$(readlink -f "$exec_path")" != "$(readlink -f "$CTRL_BIN")" ]]; then
        note "ferro-controller.service runs $exec_path, not $CTRL_BIN; not managing the controller"
        return 0
    fi
    pid="$(systemctl --user show -p MainPID --value ferro-controller)"
    if [[ "${pid:-0}" == 0 ]]; then
        note "ferro-controller is not running; systemd will start it on the new binary"
        return 0
    fi
    [[ "$(sha "/proc/$pid/exe")" == "$want" ]] && return 0

    # A restarted controller does recover running jobs, but that is a safety
    # net, not something to lean on for a routine upgrade.
    active="$(count_jobs pending launching running)" \
        || { note "cannot list jobs; not restarting the controller blind"; return 0; }
    if (( active > 0 )); then
        note "controller update waiting: $active job(s) pending, launching or running"
        return 0
    fi
    before="$(healthy_nodes)" || before=0
    say "restarting the controller onto ${want:0:12}"
    if (( DRY_RUN )); then
        note "(dry run) not restarting"
        return 0
    fi
    # Readable while the process runs even though cargo has replaced the file:
    # the one copy of the old binary left, for a manual rollback.
    cp "/proc/$pid/exe" "$STATE/ferro-controller.prev"
    systemctl --user restart ferro-controller
    # Agents reconnect by themselves: a 3 s heartbeat behind a 3 s backoff.
    now=0
    for _ in $(seq 1 20); do
        sleep 3
        now="$(healthy_nodes 2>/dev/null)" || now=0
        (( now >= before )) && break
    done
    if ! systemctl --user is-active --quiet ferro-controller; then
        die "the new controller is not running -- journalctl --user -u ferro-controller; the previous binary is $STATE/ferro-controller.prev"
    fi
    note "controller restarted, $now of $before node(s) back"
}

# ---------------------------------------------------------------------------
# 4. The agents
# ---------------------------------------------------------------------------

# Prints the sha256 of the agent systemd is running on <target>, provided it
# has not restarted by itself since it was last (re)started -- a crash loop
# would otherwise pass for a healthy agent between two crashes.
remote_agent() {
    "${SSH[@]}" -n "$1" 'pid=$(systemctl --user show -p MainPID --value ferro-agent)
        n=$(systemctl --user show -p NRestarts --value ferro-agent)
        [ "${pid:-0}" -gt 0 ] && [ "${n:-0}" = 0 ] && sha256sum "/proc/$pid/exe" | cut -d" " -f1'
}

node_healthy() {
    ferro_json nodes | python3 -c '
import json, sys
print(any(n["node_id"] == sys.argv[1] and n.get("healthy") for n in json.load(sys.stdin)))' "$1" \
        | grep -qx True
}

deploy_agent() {  # <node id> <user@host> <sha256>
    local id="$1" target="$2" want="$3"
    say "$id: installing agent ${want:0:12} on $target"
    if (( DRY_RUN )); then
        note "(dry run) not installing"
        return 0
    fi
    "${SCP[@]}" "$AGENT_BIN" "$target:~/.local/bin/ferro-agent.new" \
        || { note "$id: copy failed; trying again next pass"; return 0; }
    # Swap atomically so a restart cannot catch half a file, and keep the old
    # binary beside it for the rollback below.
    "${SSH[@]}" -n "$target" 'set -e; cd ~/.local/bin
        chmod +x ferro-agent.new
        cp -p ferro-agent ferro-agent.bak
        mv ferro-agent.new ferro-agent
        systemctl --user restart ferro-agent' \
        || { note "$id: install failed; trying again next pass"; return 0; }
    # Past the controller's 15 s health timeout, so "healthy" can only mean
    # the new agent has been heartbeating, not that the old one's last
    # heartbeat has not expired yet.
    sleep 18
    if [[ "$(remote_agent "$target" 2>/dev/null)" == "$want" ]] && node_healthy "$id"; then
        rm -f "$STATE/failed-$id"
        note "$id: new agent running and registered"
        return 0
    fi
    note "$id: new agent did not come back registered; restoring the previous one"
    "${SSH[@]}" -n "$target" 'cd ~/.local/bin && mv ferro-agent.bak ferro-agent && systemctl --user restart ferro-agent' \
        || note "$id: ROLLBACK FAILED -- log in and check ferro-agent there"
    # Not retried every pass: the next build gets a fresh attempt.
    echo "$want" >"$STATE/failed-$id"
}

converge_agents() {
    local want nodes waiting id target held state running
    want="$(built ferro-agent)"
    if [[ "$(sha "$AGENT_BIN" 2>/dev/null)" != "$want" ]]; then
        note "target/portable/release/ferro-agent is not the binary this updater built; leaving the agents alone"
        return 0
    fi
    nodes="$(node_table)" || { note "cannot read the node list from the controller; skipping the agents"; return 0; }
    # A queued job can be dispatched onto a node while its agent restarts.
    waiting="$(count_jobs queued pending launching)" || { note "cannot list jobs; skipping the agents"; return 0; }
    while IFS=$'\t' read -r id target held state; do
        [[ -n "$id" ]] || continue
        case "$state" in
            stale)   note "$id: not heartbeating; skipping it"; continue ;;
            no-user) note "$id: agent too old to report its login user; update it once with scripts/deploy_agent.sh"; continue ;;
        esac
        running="$(remote_agent "$target" 2>/dev/null)" \
            || { note "$id: cannot read its agent over SSH ($target); skipping it"; continue; }
        [[ "$running" == "$want" ]] && continue
        if [[ "$(cat "$STATE/failed-$id" 2>/dev/null)" == "$want" ]]; then
            note "$id: this build failed to come up there before; skipping it until the next build"
            continue
        fi
        if (( held > 0 )); then
            note "$id: agent update waiting, a job holds $held GPU(s) there"
            continue
        fi
        if (( waiting > 0 )); then
            note "$id: agent update waiting, $waiting job(s) queued or starting"
            continue
        fi
        deploy_agent "$id" "$target" "$want"
    done <<<"$nodes"
}

# ---------------------------------------------------------------------------

main() {
    DRY_RUN=0
    case "${1:-}" in
        "")        ;;
        --dry-run) DRY_RUN=1 ;;
        # Print the header comment, however long it grows.
        -h|--help) awk 'NR>1 && /^#/ {sub(/^#[[:space:]]?/, ""); print; next} NR>1 {exit}' "$0"; return 0 ;;
        *)         die "unknown argument: $1 (try --help)" ;;
    esac

    cd "$(dirname "$0")/.."
    BRANCH="${FERRO_UPDATE_BRANCH:-main}"
    STATE="${XDG_STATE_HOME:-$HOME/.local/state}/ferrogrid/autoupdate"
    CTRL_BIN="$PWD/target/release/ferro-controller"
    AGENT_BIN="$PWD/target/portable/release/ferro-agent"
    FERRO="$PWD/target/release/ferro"
    [[ -x "$FERRO" ]] || FERRO=ferro
    SLUG="$(git remote get-url origin | sed -E 's#^(git@github\.com:|ssh://git@github\.com/|https://github\.com/)##; s#\.git$##')"
    # Never prompt: under systemd there is nobody to answer.
    SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
    SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
    # `uv run` and systemd --user both start without ~/.cargo/bin on PATH.
    export PATH="$HOME/.cargo/bin:$PATH"

    mkdir -p "$STATE"
    exec 9>"$STATE/lock"
    flock -n 9 || { say "another pass is already running"; return 0; }

    say "auto-update pass$( (( DRY_RUN )) && echo ' (dry run)')"
    if checkout_is_ours; then
        advance_checkout
        build_head
    else
        note "leaving the checkout alone: $WHY"
    fi
    # Rolling out what was already built does not depend on the checkout.
    if [[ -f "$STATE/built" ]]; then
        converge_controller
        converge_agents
    else
        note "nothing built by this updater yet; not touching the services"
    fi
}

# Everything runs from one function called on the last line: a pass can
# fast-forward this very file, and bash reads a script while running it.
main "$@"; exit
