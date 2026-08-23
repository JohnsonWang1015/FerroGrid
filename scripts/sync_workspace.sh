#!/usr/bin/env bash
# Copy the training code into each node's FerroGrid workspace.
#
#   ./scripts/sync_workspace.sh <ssh-host> [more hosts...]
#
# Agents resolve relative script paths against their own workspace root
# (~/ferrogrid by default), so every node ends up with the same *relative*
# layout even though their home directories differ.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ $# -ge 1 ]] || { echo "usage: sync_workspace.sh <ssh-host> [more hosts...]"; exit 1; }

for HOST in "$@"; do
    echo "==> [$HOST] syncing python/ and mojo/ to ~/ferrogrid"
    ssh "$HOST" 'mkdir -p ~/ferrogrid'
    rsync -az --delete \
        --exclude '__pycache__' --exclude '*.pyc' \
        python mojo "$HOST:~/ferrogrid/"
    ssh "$HOST" 'ls ~/ferrogrid/python/examples/' | sed 's/^/    /'
done
echo "==> workspace synced"
