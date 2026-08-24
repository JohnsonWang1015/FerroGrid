#!/usr/bin/env bash
# Copy a plugin's credentials file to each node's plugin workdir.
#
#   ./scripts/install_plugin_creds.sh <local-file> <ssh-host> [more hosts...]
#   ./scripts/install_plugin_creds.sh ~/NextcloudFetcher/.env lab199 lab127
#
# Installs to ~/.config/ferrogrid/<basename> with mode 600.
#
# THINK BEFORE RUNNING THIS. It copies a secret to every host you list. Those
# machines are shared: anyone with root on them, now or later, can read it.
# Prefer a credential scoped to what the job needs -- for Nextcloud, an app
# password limited to the dataset share, revocable on its own -- over your
# account password.
set -euo pipefail

SRC="${1:?usage: install_plugin_creds.sh <local-file> <ssh-host> [more hosts...]}"
shift
[[ -f "$SRC" ]] || { echo "no such file: $SRC"; exit 1; }
[[ $# -ge 1 ]] || { echo "name at least one host"; exit 1; }

NAME="$(basename "$SRC")"
echo "about to copy $SRC -> ~/.config/ferrogrid/$NAME on: $*"
read -rp "continue? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || { echo "aborted"; exit 1; }

for HOST in "$@"; do
    echo "==> [$HOST]"
    ssh "$HOST" 'mkdir -p ~/.config/ferrogrid && chmod 700 ~/.config/ferrogrid'
    # Land it 600 from the start rather than fixing the mode afterwards.
    ssh "$HOST" "umask 077 && cat > ~/.config/ferrogrid/$NAME" < "$SRC"
    ssh "$HOST" "ls -l ~/.config/ferrogrid/$NAME | awk '{print \"    \", \$1, \$NF}'"
done

echo
echo "==> done. Verify with:  ferro fetch <plugin> <remote> <local>"
