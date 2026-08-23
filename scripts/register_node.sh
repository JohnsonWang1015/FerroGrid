#!/usr/bin/env bash
# Register a GPU server with FerroGrid over SSH, password auth included.
#
#   ./scripts/register_node.sh <user@host|ssh-alias> <controller-ip:port> [node-id]
#
#   ./scripts/register_node.sh user@10.0.0.12 10.0.0.1:7070 rtx5090
#
# Checks prerequisites, installs the agent as a systemd --user service, and
# waits for the node to appear in `ferro nodes`. Needs no root on the target.
#
# Password authentication is fine: the password is typed **once**, into ssh
# itself, and every later scp/ssh reuses that authenticated connection through
# an SSH control socket. Nothing is stored, and nothing is passed on a command
# line where `ps` would show it -- which is why this does not use sshpass.
#
# Options:
#   --copy-id     install your public key on the target first, so future
#                 deploys and `ferro sync` need no password at all
#   --no-image    skip pre-pulling the training image
set -euo pipefail
cd "$(dirname "$0")/.."

COPY_ID=0
PULL_IMAGE=1
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --copy-id)  COPY_ID=1; shift ;;
        --no-image) PULL_IMAGE=0; shift ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        *)          POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]}"

TARGET="${1:?usage: register_node.sh <user@host|ssh-alias> <controller-ip:port> [node-id]}"
CONTROLLER="${2:?missing controller address, e.g. 10.0.0.1:7070}"
NODE_ID="${3:-}"
IMAGE="${FERRO_DEFAULT_IMAGE:-pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime}"

BIN="target/portable/release/ferro-agent"
[[ -f "$BIN" ]] || { echo "missing $BIN -- run ./scripts/build.sh portable first"; exit 1; }

if [[ $COPY_ID -eq 1 ]]; then
    echo "==> installing your public key on $TARGET (one password prompt)"
    ssh-copy-id "$TARGET"
fi

# One authenticated connection, reused by everything below. ControlPersist
# keeps it alive briefly so the whole run needs a single prompt.
# %C is an SSH ControlPath token (a hash of user/host/port), so it has to
# reach ssh literally -- mktemp would reject it.
CTLDIR="${TMPDIR:-/tmp}/ferro-ssh-$(id -u)"
mkdir -p "$CTLDIR" && chmod 700 "$CTLDIR"
CTL="$CTLDIR/%C"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=120)
SCP=(scp -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=120)

cleanup() { ssh -o "ControlPath=$CTL" -O exit "$TARGET" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> connecting to $TARGET (you may be prompted for a password once)"
"${SSH[@]}" "$TARGET" true

echo "==> checking prerequisites"
"${SSH[@]}" "$TARGET" 'set -e
    command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi missing"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed "s/^/    /"
    command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }
    docker info >/dev/null 2>&1 || { echo "FAIL: cannot talk to docker (add $USER to the docker group)"; exit 1; }
    docker info 2>/dev/null | grep -q nvidia || { echo "FAIL: nvidia container runtime not registered"; exit 1; }
    echo "    prerequisites OK"'

if [[ $PULL_IMAGE -eq 1 ]]; then
    echo "==> pre-pulling $IMAGE (skip with --no-image)"
    "${SSH[@]}" "$TARGET" "docker pull '$IMAGE' >/dev/null && echo '    image ready'"
fi

echo "==> installing the agent"
"${SSH[@]}" "$TARGET" 'mkdir -p ~/.local/bin ~/.config/systemd/user ~/ferrogrid'
"${SCP[@]}" -q "$BIN" "$TARGET:.local/bin/ferro-agent.new"
"${SSH[@]}" "$TARGET" 'chmod +x ~/.local/bin/ferro-agent.new && mv ~/.local/bin/ferro-agent.new ~/.local/bin/ferro-agent'

EXTRA=""
[[ -n "$NODE_ID" ]] && EXTRA="Environment=FERRO_NODE_ID=$NODE_ID"

"${SSH[@]}" "$TARGET" "cat > ~/.config/systemd/user/ferro-agent.service" <<UNIT
[Unit]
Description=FerroGrid node agent
After=network-online.target docker.service

[Service]
Type=simple
Environment=FERRO_CONTROLLER=http://$CONTROLLER
Environment=FERRO_AGENT_BIND=0.0.0.0:7071
Environment=RUST_LOG=info
$EXTRA
ExecStart=%h/.local/bin/ferro-agent
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT

"${SSH[@]}" "$TARGET" 'loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable ferro-agent >/dev/null 2>&1
systemctl --user restart ferro-agent
sleep 2
systemctl --user is-active ferro-agent >/dev/null && echo "    agent active" || {
    echo "    agent failed to start:"; journalctl --user -u ferro-agent -n 20 --no-pager; exit 1; }'

# Confirm it actually reached the controller, rather than just started.
echo "==> waiting for the node to register"
WANT="${NODE_ID:-$("${SSH[@]}" "$TARGET" hostname)}"
FERRO="${FERRO:-ferro}"
command -v "$FERRO" >/dev/null || FERRO=./target/release/ferro
for _ in $(seq 1 20); do
    if FERRO_CONTROLLER="http://$CONTROLLER" "$FERRO" --json nodes 2>/dev/null \
        | grep -q "\"node_id\": \"$WANT\""; then
        echo "    registered as '$WANT'"
        FERRO_CONTROLLER="http://$CONTROLLER" "$FERRO" nodes
        exit 0
    fi
    sleep 1
done

echo "    node did not appear in \`ferro nodes\` within 20s."
echo "    check the agent log:  ssh $TARGET journalctl --user -u ferro-agent -n 30"
exit 1
