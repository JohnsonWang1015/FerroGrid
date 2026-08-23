#!/usr/bin/env bash
# Tail an agent's journal:  ./scripts/logs_agent.sh <ssh-host> [lines]
set -euo pipefail
HOST="${1:?usage: logs_agent.sh <ssh-host> [lines]}"
ssh -t "$HOST" "journalctl --user -u ferro-agent -n ${2:-50} --no-pager"
