#!/usr/bin/env bash
# Unpacks a bundle written by scripts/migrate.sh. Runs on the *new* machine,
# from ~/.ferro-migrate, and is not meant to be invoked by hand.
#
#   bash ~/.ferro-migrate/install.sh                      # install everything
#   bash ~/.ferro-migrate/install.sh --controller-service PORT
#
# Nothing here needs root: binaries go to ~/.local/bin and the controller, if
# asked for, runs as a `systemd --user` service exactly like the agents do.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/manifest.env"

note() { printf '    %s\n' "$*"; }

MARK_BEGIN="# >>> ferrogrid >>>"
MARK_END="# <<< ferrogrid <<<"

# Replace our managed block in a file, leaving everything else alone. `where`
# is `top` for ssh_config, where first-obtained-value wins and a block landing
# after somebody's `Host *` would never take effect.
replace_block() {  # <file> <where: top|bottom> < block on stdin
    local file="$1" where="$2" block tmp
    block="$(cat)"
    tmp="$(mktemp)"
    if [[ -f "$file" ]]; then
        awk -v b="$MARK_BEGIN" -v e="$MARK_END" '
            $0 == b { skip = 1; next }
            $0 == e { skip = 0; next }
            !skip' "$file" > "$tmp"
    fi
    if [[ "$where" == top ]]; then
        { printf '%s\n%s\n%s\n\n' "$MARK_BEGIN" "$block" "$MARK_END"; cat "$tmp"; } > "$file.new"
    else
        { cat "$tmp"; printf '\n%s\n%s\n%s\n' "$MARK_BEGIN" "$block" "$MARK_END"; } > "$file.new"
    fi
    rm -f "$tmp"
    mv "$file.new" "$file"
}

# ---------------------------------------------------------------------------
# --controller-service: the takeover half, run after the main install
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--controller-service" ]]; then
    PORT="${2:-7070}"
    mkdir -p ~/.config/systemd/user
    cat > ~/.config/systemd/user/ferro-controller.service <<UNIT
[Unit]
Description=FerroGrid controller
After=network-online.target

[Service]
Type=simple
Environment=RUST_LOG=info
Environment=PATH=%h/.local/bin:%h/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=%h/.local/bin/ferro-controller --bind 0.0.0.0:$PORT
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
    # linger, so the controller outlives the SSH session that installed it.
    loginctl enable-linger "$USER" 2>/dev/null || true
    systemctl --user daemon-reload
    systemctl --user enable ferro-controller >/dev/null 2>&1 || true
    # restart, not `enable --now`: that does not restart a unit already running
    # an older binary.
    systemctl --user restart ferro-controller
    sleep 2
    if systemctl --user is-active ferro-controller >/dev/null; then
        note "ferro-controller active on 0.0.0.0:$PORT"
    else
        note "ferro-controller failed to start:"
        journalctl --user -u ferro-controller -n 20 --no-pager
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Binaries
# ---------------------------------------------------------------------------
mkdir -p ~/.local/bin ~/.config/ferrogrid ~/.ssh
chmod 700 ~/.ssh
for b in ferro ferro-agent ferro-controller; do
    # Replace atomically: an upgrade must not catch a binary mid-copy, and
    # `ferro watch` may well be running while this lands.
    install -m 755 "$HERE/bin/$b" ~/.local/bin/"$b.new"
    mv ~/.local/bin/"$b.new" ~/.local/bin/"$b"
done
note "installed $(~/.local/bin/ferro --version)"

# ---------------------------------------------------------------------------
# SSH: key, host blocks, known_hosts
# ---------------------------------------------------------------------------
KEY_PATH=""
if [[ -f "$HERE/id_key" && -n "${FERRO_MIGRATE_KEYNAME:-}" ]]; then
    KEY_PATH="$HOME/.ssh/$FERRO_MIGRATE_KEYNAME"
    if [[ -f "$KEY_PATH" ]] && ! cmp -s "$KEY_PATH" "$HERE/id_key"; then
        # A different key already lives under that name. Taking it would lock
        # this machine out of whatever it was for, so the migrated one gets a
        # name of its own and the host blocks point at it explicitly.
        KEY_PATH="$HOME/.ssh/id_ferrogrid"
        note "$FERRO_MIGRATE_KEYNAME already exists and differs; installing as id_ferrogrid"
    fi
    install -m 600 "$HERE/id_key" "$KEY_PATH"
    [[ -f "$HERE/id_key.pub" ]] && install -m 644 "$HERE/id_key.pub" "$KEY_PATH.pub"
    note "ssh key -> $KEY_PATH"
fi

if [[ -f "$HERE/ssh_config" ]]; then
    BLOCK="$(mktemp)"
    if [[ -n "$KEY_PATH" ]]; then
        sed "s|@@FERRO_KEY@@|$KEY_PATH|g" "$HERE/ssh_config" > "$BLOCK"
    else
        # No key travelled: leave the default identity search alone rather than
        # pointing every host at a file that is not there.
        sed '/@@FERRO_KEY@@/d; /^[[:space:]]*IdentitiesOnly yes$/d' "$HERE/ssh_config" > "$BLOCK"
    fi
    [[ -f ~/.ssh/config ]] && cp -n ~/.ssh/config ~/.ssh/config.bak 2>/dev/null || true
    replace_block ~/.ssh/config top < "$BLOCK"
    chmod 600 ~/.ssh/config
    note "$(grep -c '^Host ' "$BLOCK") ssh host block(s) merged into ~/.ssh/config"
    rm -f "$BLOCK"
fi

if [[ -s "$HERE/known_hosts" ]]; then
    touch ~/.ssh/known_hosts
    added=0
    while read -r line; do
        [[ -n "$line" ]] || continue
        grep -qxF "$line" ~/.ssh/known_hosts || { printf '%s\n' "$line" >> ~/.ssh/known_hosts; added=$((added + 1)); }
    done < "$HERE/known_hosts"
    chmod 600 ~/.ssh/known_hosts
    note "$added new known_hosts entr(ies)"
fi

# ---------------------------------------------------------------------------
# Controller address and plugins
# ---------------------------------------------------------------------------
if [[ -f "$HERE/plugins.toml" ]]; then
    if [[ -f ~/.config/ferrogrid/plugins.toml ]] && ! cmp -s ~/.config/ferrogrid/plugins.toml "$HERE/plugins.toml"; then
        cp ~/.config/ferrogrid/plugins.toml ~/.config/ferrogrid/plugins.toml.bak
        note "kept the previous plugins.toml as plugins.toml.bak"
    fi
    install -m 600 "$HERE/plugins.toml" ~/.config/ferrogrid/plugins.toml
    note "plugins.toml installed (the plugin's own credentials are not in it -- they live in its workdir on each node)"
fi

cat > ~/.config/ferrogrid/env.sh <<EOF
# FerroGrid environment. Written by scripts/migrate.sh; edit freely, it is only
# rewritten by another migration.
case ":\$PATH:" in
    *":\$HOME/.local/bin:"*) ;;
    *) PATH="\$HOME/.local/bin:\$PATH"; export PATH ;;
esac
export FERRO_CONTROLLER="$FERRO_MIGRATE_CONTROLLER"
EOF
note "FERRO_CONTROLLER=$FERRO_MIGRATE_CONTROLLER"

for rc in ~/.bashrc ~/.zshrc ~/.profile; do
    # ~/.profile is the fallback for a login shell with no bashrc; skip it when
    # one of the interactive rc files is already there.
    [[ "$rc" == ~/.profile && ( -f ~/.bashrc || -f ~/.zshrc ) ]] && continue
    [[ -f "$rc" ]] || continue
    replace_block "$rc" bottom <<'RC'
[ -f "$HOME/.config/ferrogrid/env.sh" ] && . "$HOME/.config/ferrogrid/env.sh"
RC
    note "wired into $(basename "$rc")"
done

# ---------------------------------------------------------------------------
# The checkout
# ---------------------------------------------------------------------------
if [[ "${FERRO_MIGRATE_SOURCE:-0}" -eq 1 && -f "$HERE/src.tar.gz" ]]; then
    # Expanded here rather than on the sending side: $HOME is this machine's.
    DEST="${FERRO_MIGRATE_DEST/#\$HOME/$HOME}"
    DEST="${DEST/#\~/$HOME}"
    if [[ -e "$DEST" && -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
        # A re-run is the normal way to upgrade, so an existing FerroGrid
        # checkout is refreshed in place: tar overwrites what this machine
        # ships and leaves everything else, so nothing there is deleted. A
        # directory that is *not* a FerroGrid checkout is somebody else's, and
        # gets unpacked beside instead of into.
        if [[ -f "$DEST/Cargo.toml" ]] && grep -q ferro-proto "$DEST/Cargo.toml" 2>/dev/null; then
            note "refreshing the checkout at $DEST (this machine's copy wins where they differ)"
        else
            note "leaving $DEST alone: it is not a FerroGrid checkout"
            DEST="$DEST.migrated"
            rm -rf "$DEST"
        fi
    fi
    mkdir -p "$DEST"
    tar xzf "$HERE/src.tar.gz" -C "$DEST"
    note "checkout -> $DEST"
fi

# ---------------------------------------------------------------------------
# What this machine still lacks
# ---------------------------------------------------------------------------
for tool in rsync ssh; do
    command -v "$tool" >/dev/null || note "WARNING: $tool is not installed -- \`ferro sync\` needs it"
done
