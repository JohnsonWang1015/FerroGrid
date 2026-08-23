#!/usr/bin/env bash
# Ship the agent to a GPU server and run it under systemd --user.
#
#   ./scripts/deploy_agent.sh <ssh-host> <controller-ip:port> [nccl-ip] [node-id]
#
# Requires only SSH + docker-group membership on the target. No root needed:
# the agent runs as a user service.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:?usage: deploy_agent.sh <ssh-host> <controller-ip:port> [nccl-ip] [node-id]}"
CONTROLLER="${2:?missing controller address, e.g. 10.0.0.1:7070}"
NCCL_IP="${3:-}"
NODE_ID="${4:-}"
# Per-node training image. A node whose GPUs are newer than the default image
# supports (Blackwell needs CUDA >= 12.8) must override this.
IMAGE="${FERRO_DEFAULT_IMAGE:-pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime}"

BIN="target/portable/release/ferro-agent"
[[ -f "$BIN" ]] || { echo "missing $BIN -- run ./scripts/build.sh portable first"; exit 1; }

echo "==> [$HOST] copying agent"
ssh "$HOST" 'mkdir -p ~/.local/bin ~/.config/systemd/user ~/ferrogrid'
scp -q "$BIN" "$HOST:~/.local/bin/ferro-agent.new"
# Replace atomically so an upgrade cannot catch the binary mid-copy.
ssh "$HOST" 'chmod +x ~/.local/bin/ferro-agent.new && mv ~/.local/bin/ferro-agent.new ~/.local/bin/ferro-agent'

EXTRA=""
[[ -n "$NCCL_IP" ]] && EXTRA="Environment=FERRO_NCCL_IP=$NCCL_IP"
# Some lab boxes have unhelpful hostnames ("user", "gpu"); a stable node id
# keeps `ferro nodes` readable and lets --node filters target them.
[[ -n "$NODE_ID" ]] && EXTRA="$EXTRA
Environment=FERRO_NODE_ID=$NODE_ID"

echo "==> [$HOST] installing user service"
ssh "$HOST" "cat > ~/.config/systemd/user/ferro-agent.service" <<UNIT
[Unit]
Description=FerroGrid node agent
After=network-online.target docker.service

[Service]
Type=simple
Environment=FERRO_CONTROLLER=http://$CONTROLLER
Environment=FERRO_AGENT_BIND=0.0.0.0:7071
Environment=RUST_LOG=info
Environment=FERRO_DEFAULT_IMAGE=$IMAGE
$EXTRA
ExecStart=%h/.local/bin/ferro-agent
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT

# linger keeps the agent running after the deploying SSH session logs out.
ssh "$HOST" 'loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable ferro-agent
# restart, not "enable --now": on a redeploy the service is already running and
# would otherwise keep executing the old binary.
systemctl --user restart ferro-agent
sleep 2
systemctl --user is-active ferro-agent && echo "   agent active" || {
    echo "   agent failed to start:"; journalctl --user -u ferro-agent -n 20 --no-pager; exit 1; }'

echo "==> [$HOST] done"
