#!/usr/bin/env bash
# Move FerroGrid onto another machine over SSH, in one command.
#
#   ./scripts/migrate.sh <user@newhost|ssh-alias> [options]
#
#   ./scripts/migrate.sh esl@10.0.0.5              # new machine drives this cluster
#   ./scripts/migrate.sh esl@10.0.0.5 --takeover   # ...and becomes the controller
#
# What actually has to travel for `ferro nodes` to work somewhere else is not
# the repository -- it is the three binaries, the way in to each node, and the
# address of the controller. This ships all of it:
#
#   * ferro / ferro-agent / ferro-controller  -> ~/.local/bin
#   * the SSH config blocks, private key and known_hosts entries for every
#     registered node, so `ferro sync` works from the new machine too
#   * ~/.config/ferrogrid/plugins.toml
#   * FERRO_CONTROLLER + PATH, wired into the login shell
#   * the checkout itself, so training code and scripts are there as well
#
# The node inventory is read live from the controller rather than from a host
# list, the same way `ferro sync` does it: a node that registered is a node
# that gets carried over, and one that never did is not silently assumed.
#
# Options:
#   --controller HOST:PORT  controller the new machine should talk to.
#                           Default: this machine's address on the lab network.
#   --takeover              also move the controller: run it on the new machine
#                           under systemd and re-point every agent at it. The
#                           new machine then needs nothing from this one.
#   --dest DIR              where to put the checkout (default ~/FerroGrid)
#   --no-key                do not copy the SSH private key. The new machine
#                           can still reach nodes it already has a key for.
#   --no-source             binaries and settings only, no checkout
#   --proxy-jump [user@]host
#                           reach the nodes through a jump host, for a new
#                           machine that can see this one but not the node
#                           network (a laptop on the VPN, say). Added to every
#                           migrated Host block that does not already name one.
#   --ssh-all               carry every Host block in ~/.ssh/config, not just
#                           the ones belonging to registered nodes
#   --yes                   do not ask before copying the private key
#   --dry-run               build the bundle, print what is in it, send nothing
#
# Re-runnable: everything it writes on the far side is replaced in place, and
# nothing it finds there is overwritten without a .bak beside it.
set -euo pipefail
cd "$(dirname "$0")/.."

TAKEOVER=0
WITH_KEY=1
WITH_SOURCE=1
SSH_ALL=0
ASSUME_YES=0
DRY_RUN=0
DEST='$HOME/FerroGrid'
CONTROLLER_OVERRIDE=""
PROXY_JUMP=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --takeover)   TAKEOVER=1; shift ;;
        --no-key)     WITH_KEY=0; shift ;;
        --no-source)  WITH_SOURCE=0; shift ;;
        --ssh-all)    SSH_ALL=1; shift ;;
        --yes|-y)     ASSUME_YES=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --dest)       DEST="${2:?--dest needs a directory}"; shift 2 ;;
        --controller) CONTROLLER_OVERRIDE="${2:?--controller needs HOST:PORT}"; shift 2 ;;
        --proxy-jump) PROXY_JUMP="${2:?--proxy-jump needs [user@]host}"; shift 2 ;;
        # Print the header comment, however long it grows.
        -h|--help)    awk 'NR>1 && /^#/ {sub(/^#[[:space:]]?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]:-}"

TARGET="${1:?usage: migrate.sh <user@newhost|ssh-alias> [--takeover] [--controller HOST:PORT]}"

say()  { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. What we have here
# ---------------------------------------------------------------------------

FERRO="${FERRO:-ferro}"
command -v "$FERRO" >/dev/null 2>&1 || FERRO=./target/release/ferro
[[ -x "$FERRO" ]] || command -v "$FERRO" >/dev/null 2>&1 \
    || die "no \`ferro\` on PATH and no ./target/release/ferro -- run \`uv run --all-extras ferro-setup\` first"

SRC_CONTROLLER="${FERRO_CONTROLLER:-http://127.0.0.1:7070}"
CONTROLLER_PORT="${SRC_CONTROLLER##*:}"
[[ "$CONTROLLER_PORT" =~ ^[0-9]+$ ]] || CONTROLLER_PORT=7070

# Prefer the portable build: the new machine may be an older Ubuntu than this
# one, and a binary built here against a newer glibc would not start there.
BINDIR="target/portable/release"
[[ -x "$BINDIR/ferro" ]] || BINDIR="target/release"
for b in ferro ferro-agent ferro-controller; do
    [[ -x "$BINDIR/$b" ]] || die "missing $BINDIR/$b -- run \`./scripts/build.sh portable\` (or \`cargo build --release\`) first"
done
note "shipping binaries from $BINDIR"
[[ "$BINDIR" == target/release ]] && note "note: not the portable build; if the new machine is older than this one, run ./scripts/build.sh portable first"

say "reading the node inventory from $SRC_CONTROLLER"
NODES_JSON="$(FERRO_CONTROLLER="$SRC_CONTROLLER" "$FERRO" --json nodes 2>/dev/null)" \
    || die "cannot reach the controller at $SRC_CONTROLLER.
    Start it (\`ferro-controller\`) or point FERRO_CONTROLLER at it: the node
    list is what this script migrates, so there is nothing to do without it."

# node_id <TAB> ip <TAB> login user <TAB> workspace <TAB> healthy
NODE_TSV="$(printf '%s' "$NODES_JSON" | python3 -c '
import json, sys
for n in json.load(sys.stdin):
    ip = (n.get("nccl_address") or n.get("address", "")).split(":")[0]
    print("\t".join([n["node_id"], ip, n.get("user") or "", n.get("workspace") or "",
                     "healthy" if n.get("healthy") else "stale"]))
')"
[[ -n "$NODE_TSV" ]] || die "the controller has no registered nodes -- nothing to migrate"
NODE_COUNT="$(printf '%s\n' "$NODE_TSV" | wc -l)"
while IFS=$'\t' read -r id ip user _ health; do
    note "$id  $ip  ${user:-<unknown login>}  $health"
done <<<"$NODE_TSV"

# The address the nodes can actually reach us on. Asking the routing table for
# the route to a node beats guessing a "primary" IP on a box with half a dozen
# docker bridges and a tailscale interface.
FIRST_NODE_IP="$(printf '%s\n' "$NODE_TSV" | head -1 | cut -f2)"
lan_ip_towards() {  # <ip> -> source address the kernel would use
    ip route get "$1" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1
}

# ---------------------------------------------------------------------------
# 2. One authenticated SSH connection to the new machine, reused throughout
#    (the register_node.sh pattern: a single password prompt, then a control
#    socket -- never sshpass, which would put the password in `ps`).
# ---------------------------------------------------------------------------

CTLDIR="${TMPDIR:-/tmp}/ferro-ssh-$(id -u)"
mkdir -p "$CTLDIR" && chmod 700 "$CTLDIR"
CTL="$CTLDIR/%C"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=180)
SCP=(scp -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=180)

BUNDLE="$(mktemp -d "${TMPDIR:-/tmp}/ferro-migrate.XXXXXX")"
SHIPPED=0
cleanup() {
    rm -rf "$BUNDLE" "$BUNDLE.tar.gz"
    # The bundle carries a private key, so it comes off the target whatever
    # happened -- including a run that failed half way through the install.
    [[ $SHIPPED -eq 1 ]] && ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o "ControlPath=$CTL" \
        "$TARGET" 'rm -rf ~/.ferro-migrate' 2>/dev/null
    ssh -o "ControlPath=$CTL" -O exit "$TARGET" 2>/dev/null || true
}
trap cleanup EXIT

if [[ $DRY_RUN -eq 0 ]]; then
    say "connecting to $TARGET (you may be prompted for a password once)"
    "${SSH[@]}" "$TARGET" true || die "cannot ssh to $TARGET"
    note "target: $("${SSH[@]}" "$TARGET" 'printf "%s, home %s" "$(id -un)" "$HOME"')"

    # Whether the node network is routable from there at all. Learning this in
    # the final verification instead means a bundle has already shipped and an
    # ssh config has already been written for a route that does not exist.
    if [[ -z "$PROXY_JUMP" ]] \
       && ! "${SSH[@]}" -n "$TARGET" "timeout 5 bash -c 'cat </dev/null >/dev/tcp/$FIRST_NODE_IP/22'" 2>/dev/null; then
        note "warning: $TARGET cannot open $FIRST_NODE_IP:22, so it has no way to"
        note "         reach the nodes and \`ferro sync\` will not work from there."
        note "         Re-run with --proxy-jump $(id -un)@<an address of this machine"
        note "         it can see> to send them through here."
    fi
fi

# ---------------------------------------------------------------------------
# 3. Which controller the new machine will talk to
# ---------------------------------------------------------------------------

if [[ -n "$CONTROLLER_OVERRIDE" ]]; then
    NEW_CONTROLLER="http://$CONTROLLER_OVERRIDE"
elif [[ $TAKEOVER -eq 1 ]]; then
    [[ $DRY_RUN -eq 1 ]] && die "--dry-run --takeover needs --controller HOST:PORT: the new machine's address is something only it can answer"
    # The new machine's own address, as the *nodes* would route to it.
    NEW_IP="$("${SSH[@]}" "$TARGET" "ip route get $FIRST_NODE_IP 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -1")"
    [[ -n "$NEW_IP" ]] || die "could not work out $TARGET's address towards $FIRST_NODE_IP -- pass --controller HOST:PORT"
    NEW_CONTROLLER="http://$NEW_IP:$CONTROLLER_PORT"
    note "new controller will be $NEW_CONTROLLER"
else
    HERE_IP="$(lan_ip_towards "$FIRST_NODE_IP")"
    [[ -n "$HERE_IP" ]] || die "could not work out this machine's address towards $FIRST_NODE_IP -- pass --controller HOST:PORT"
    NEW_CONTROLLER="http://$HERE_IP:$CONTROLLER_PORT"
    note "new machine will use the controller already running here: $NEW_CONTROLLER"
fi

# ---------------------------------------------------------------------------
# 4. Build the bundle
# ---------------------------------------------------------------------------

say "packing settings"
mkdir -p "$BUNDLE/bin"
for b in ferro ferro-agent ferro-controller; do cp "$BINDIR/$b" "$BUNDLE/bin/$b"; done
printf '%s\n' "$NODES_JSON" > "$BUNDLE/nodes.json"

PLUGINS="${FERRO_PLUGINS:-$HOME/.config/ferrogrid/plugins.toml}"
if [[ -f "$PLUGINS" ]]; then
    cp "$PLUGINS" "$BUNDLE/plugins.toml"
    note "plugins.toml ($(grep -c '^\[' "$PLUGINS") plugin(s))"
else
    note "no plugins.toml here; \`ferro fetch\`/\`push\` will be unavailable there too"
fi

# --- SSH: config blocks, key, known_hosts ---------------------------------
SSH_CONFIG="$HOME/.ssh/config"
# Aliases the local config defines, wildcards excluded -- `Host *` carries
# defaults, not a machine.
local_aliases() {
    [[ -f "$SSH_CONFIG" ]] || return 0
    awk '/^[[:space:]]*[Hh]ost[[:space:]]/{for(i=2;i<=NF;i++) if ($i !~ /[*?!]/) print $i}' "$SSH_CONFIG"
}
# The block the user wrote for one alias, verbatim: ProxyJump, a per-host
# IdentityFile, an odd port, whatever else is in there travels unchanged.
literal_block() {
    awk -v want="$1" '
        /^[[:space:]]*([Hh]ost|[Mm]atch)[[:space:]]/ {
            p = 0
            if (tolower($1) == "host") for (i = 2; i <= NF; i++) if ($i == want) p = 1
        }
        p' "$SSH_CONFIG"
}

declare -A ALIAS_FOR_IP=()
declare -a MIGRATED_HOSTS=()
while read -r a; do
    [[ -n "$a" ]] || continue
    hn="$(ssh -G "$a" 2>/dev/null | sed -n 's/^hostname //p' | head -1)"
    [[ -n "$hn" ]] && ALIAS_FOR_IP["$hn"]="$a"
done < <(local_aliases)

{
    echo "# FerroGrid nodes, migrated from $(hostname) on $(date -u +%Y-%m-%dT%H:%M:%SZ)."
    echo "# Managed block: scripts/migrate.sh replaces it wholesale on a re-run."
} > "$BUNDLE/ssh_config"

emit_host() {  # <alias> <ip> <login-user>
    local alias="$1" ip="$2" user="$3" block=""
    MIGRATED_HOSTS+=("$alias" "$ip")
    if [[ -f "$SSH_CONFIG" ]]; then
        block="$(literal_block "$alias")"
    fi
    if [[ -z "$block" ]]; then
        # No entry here either: synthesise one from what the node itself
        # reported at registration, which is where its login user comes from.
        block="Host $alias"$'\n'"    HostName $ip"
        [[ -n "$user" ]] && block+=$'\n'"    User $user"
    fi
    # The node network may be routable from here and not from there. A block
    # that already routes itself somehow -- ProxyJump, ProxyCommand -- keeps
    # what it has rather than being sent through ours as well.
    if [[ -n "$PROXY_JUMP" ]] && ! grep -qiE '^[[:space:]]*proxy(jump|command)' <<<"$block"; then
        block+=$'\n'"    ProxyJump $PROXY_JUMP"
    fi
    # Only name the migrated key where the block does not already choose one:
    # a host with its own IdentityFile keeps it. `IdentitiesOnly` stops an
    # agent holding a dozen keys from exhausting MaxAuthTries before ours.
    if ! grep -qi '^[[:space:]]*identityfile' <<<"$block"; then
        block+=$'\n'"    IdentityFile @@FERRO_KEY@@"$'\n'"    IdentitiesOnly yes"
    fi
    { echo; printf '%s\n' "$block"; } >> "$BUNDLE/ssh_config"
}

declare -A EMITTED=()
while IFS=$'\t' read -r id ip user _ _; do
    alias="${ALIAS_FOR_IP[$ip]:-$id}"
    [[ -n "${EMITTED[$alias]:-}" ]] && continue
    EMITTED["$alias"]=1
    emit_host "$alias" "$ip" "$user"
done <<<"$NODE_TSV"

if [[ $SSH_ALL -eq 1 ]]; then
    while read -r a; do
        [[ -n "$a" && -z "${EMITTED[$a]:-}" ]] || continue
        EMITTED["$a"]=1
        hn="$(ssh -G "$a" 2>/dev/null | sed -n 's/^hostname //p' | head -1)"
        [[ -n "$hn" ]] && emit_host "$a" "$hn" ""
    done < <(local_aliases)
fi
note "$(( ${#EMITTED[@]} )) SSH host block(s)"

# known_hosts, so the first connection from the new machine is verified rather
# than merely accepted.
: > "$BUNDLE/known_hosts"
for h in "${MIGRATED_HOSTS[@]}"; do
    [[ -n "$h" ]] || continue
    ssh-keygen -F "$h" -f "$HOME/.ssh/known_hosts" 2>/dev/null | grep -v '^#' >> "$BUNDLE/known_hosts" || true
done
sort -u "$BUNDLE/known_hosts" -o "$BUNDLE/known_hosts"
note "$(wc -l < "$BUNDLE/known_hosts") known_hosts entr(ies)"

KEY=""
KEYNAME=""
if [[ $WITH_KEY -eq 1 ]]; then
    # Whatever key the config points these hosts at, rather than assuming
    # id_ed25519.
    # `|| true`: finding no key is an answer, not a failure -- without it the
    # loop's non-zero status ends the whole script under `set -e`, and the
    # "skipping" note below never gets to say so.
    KEY="$(ssh -G "${MIGRATED_HOSTS[0]}" 2>/dev/null | sed -n 's/^identityfile //p' \
           | sed "s|^~|$HOME|" | while read -r k; do [[ -f "$k" ]] && echo "$k" && break; done || true)"
    if [[ -z "$KEY" ]]; then
        note "no SSH private key found for these hosts; skipping (the new machine will prompt for passwords)"
    else
        if [[ $ASSUME_YES -eq 0 ]]; then
            printf '    copy the private key %s to %s? [y/N] ' "$KEY" "$TARGET"
            read -r reply < /dev/tty || reply=""
            [[ "$reply" =~ ^[Yy] ]] || { KEY=""; note "skipping the key (--no-key)"; }
        fi
        if [[ -n "$KEY" ]]; then
            cp "$KEY" "$BUNDLE/id_key"
            [[ -f "$KEY.pub" ]] && cp "$KEY.pub" "$BUNDLE/id_key.pub"
            chmod 600 "$BUNDLE/id_key"
            KEYNAME="$(basename "$KEY")"
            note "key $KEYNAME"
        fi
    fi
fi

# --- the checkout ---------------------------------------------------------
if [[ $WITH_SOURCE -eq 1 ]]; then
    say "packing the checkout"
    tar czf "$BUNDLE/src.tar.gz" \
        --exclude=./target --exclude=./.venv --exclude=./.cargo-container-registry \
        --exclude=__pycache__ --exclude='*.pyc' --exclude=./.pytest_cache \
        --exclude=./mojo/build \
        -C . .
    note "$(du -h "$BUNDLE/src.tar.gz" | cut -f1)"
fi

cat > "$BUNDLE/manifest.env" <<EOF
FERRO_MIGRATE_CONTROLLER='$NEW_CONTROLLER'
FERRO_MIGRATE_DEST="$DEST"
FERRO_MIGRATE_TAKEOVER=$TAKEOVER
FERRO_MIGRATE_PORT=$CONTROLLER_PORT
FERRO_MIGRATE_HOSTS='$(printf '%s,' "${!EMITTED[@]}" | sed 's/,$//')'
FERRO_MIGRATE_KEYNAME='${KEYNAME:-}'
FERRO_MIGRATE_SOURCE=$WITH_SOURCE
EOF

cp scripts/migrate_install.sh "$BUNDLE/install.sh"

if [[ $DRY_RUN -eq 1 ]]; then
    say "dry run: this is the bundle, and it is going nowhere"
    ( cd "$BUNDLE" && find . -type f | sed 's|^\./|    |' | sort )
    echo
    say "ssh_config block"
    sed 's/^/    /' "$BUNDLE/ssh_config"
    echo
    say "manifest"
    sed 's/^/    /' "$BUNDLE/manifest.env"
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. Ship and install
# ---------------------------------------------------------------------------

say "copying to $TARGET"
tar czf "$BUNDLE.tar.gz" -C "$(dirname "$BUNDLE")" "$(basename "$BUNDLE")"
"${SSH[@]}" "$TARGET" 'rm -rf ~/.ferro-migrate && mkdir -p ~/.ferro-migrate && chmod 700 ~/.ferro-migrate'
"${SCP[@]}" -q "$BUNDLE.tar.gz" "$TARGET:.ferro-migrate/bundle.tar.gz"
SHIPPED=1
rm -f "$BUNDLE.tar.gz"
"${SSH[@]}" "$TARGET" "cd ~/.ferro-migrate && tar xzf bundle.tar.gz --strip-components=1 && rm -f bundle.tar.gz"

say "installing on $TARGET"
"${SSH[@]}" "$TARGET" 'bash ~/.ferro-migrate/install.sh'

# ---------------------------------------------------------------------------
# 6. Takeover: move the controller too
# ---------------------------------------------------------------------------

if [[ $TAKEOVER -eq 1 ]]; then
    say "starting the controller on $TARGET"
    "${SSH[@]}" "$TARGET" "bash ~/.ferro-migrate/install.sh --controller-service $CONTROLLER_PORT"

    say "re-pointing the agents at $NEW_CONTROLLER"
    # `systemctl --user enable --now` does not restart a running unit, so this
    # rewrites the unit and restarts explicitly.
    REPOINT='set -e
        u=~/.config/systemd/user/ferro-agent.service
        [ -f "$u" ] || { echo "    no ferro-agent unit -- register this node instead"; exit 1; }
        sed -i "s|^Environment=FERRO_CONTROLLER=.*|Environment=FERRO_CONTROLLER=__NEW__|" "$u"
        systemctl --user daemon-reload
        systemctl --user restart ferro-agent
        systemctl --user is-active ferro-agent >/dev/null && echo "    restarted"'
    REPOINT="${REPOINT//__NEW__/$NEW_CONTROLLER}"

    LOCAL_IPS="$(ip -4 -o addr show | awk '{split($4,a,"/"); print a[1]}')"
    FAILED=()
    while IFS=$'\t' read -r id ip user _ _; do
        printf '    %s ... ' "$id"
        if grep -qx "$ip" <<<"$LOCAL_IPS"; then
            # This machine is itself a node: no SSH round trip to reach it.
            bash -c "$REPOINT" </dev/null >/dev/null 2>&1 && echo "restarted (local)" || { echo "FAILED"; FAILED+=("$id"); }
            continue
        fi
        # -n: without it ssh reads the loop's stdin and swallows every node
        # after the first.
        alias="${ALIAS_FOR_IP[$ip]:-${user:+$user@}$ip}"
        if ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$alias" "$REPOINT" >/dev/null 2>&1; then
            echo "restarted"
        else
            echo "FAILED"
            FAILED+=("$id")
        fi
    done <<<"$NODE_TSV"

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        note "could not re-point: ${FAILED[*]}"
        note "re-register them from the new machine instead:"
        note "  ./scripts/register_node.sh <host> ${NEW_CONTROLLER#http://}"
    fi
fi

"${SSH[@]}" "$TARGET" 'rm -rf ~/.ferro-migrate'
SHIPPED=0

# ---------------------------------------------------------------------------
# 7. Verify from the new machine, which is the only opinion that counts
# ---------------------------------------------------------------------------

say "verifying from $TARGET"

SEEN=0
for _ in $(seq 1 20); do
    SEEN="$("${SSH[@]}" -n "$TARGET" "FERRO_CONTROLLER='$NEW_CONTROLLER' ~/.local/bin/ferro --json nodes 2>/dev/null | grep -c '\"node_id\"' || true" 2>/dev/null || true)"
    SEEN="${SEEN:-0}"
    [[ "$SEEN" -ge "$NODE_COUNT" ]] && break
    sleep 2
done

"${SSH[@]}" "$TARGET" "FERRO_CONTROLLER='$NEW_CONTROLLER' ~/.local/bin/ferro nodes" || true

if [[ "$SEEN" -lt "$NODE_COUNT" ]]; then
    note "$SEEN of $NODE_COUNT nodes visible from $TARGET."
    if [[ $TAKEOVER -eq 1 ]]; then
        note "agents that did not come back are still pointed at the old controller."
        note "check one:  ssh <node> journalctl --user -u ferro-agent -n 20"
    else
        note "the new machine cannot reach $NEW_CONTROLLER."
        note "check the controller is bound to 0.0.0.0 (not 127.0.0.1) and the port is open."
    fi
fi

say "checking the new machine can reach the nodes over SSH"
[[ -n "$PROXY_JUMP" ]] && note "through $PROXY_JUMP"
SSH_FAILED=0
while IFS=$'\t' read -r id ip user _ _; do
    alias="${ALIAS_FOR_IP[$ip]:-$id}"
    printf '    %s ... ' "$alias"
    "${SSH[@]}" -n "$TARGET" "ssh -n -o BatchMode=yes -o ConnectTimeout=10 '$alias' 'command -v rsync >/dev/null && echo ok || echo no-rsync'" 2>/dev/null \
        || { echo "unreachable (password auth, or the key was not copied)"; SSH_FAILED=1; }
done <<<"$NODE_TSV"
if [[ $SSH_FAILED -eq 1 && -z "$PROXY_JUMP" ]]; then
    note "if the node network is not routable from $TARGET at all, re-run with"
    note "  --proxy-jump $(id -un)@<an address of this machine it can see>"
fi

echo
say "done"
note "on $TARGET, open a new shell and run:"
note "  ferro nodes"
note "  ferro watch"
[[ $WITH_SOURCE -eq 1 ]] && note "the checkout is at $DEST -- \`ferro sync\` from inside it"
if [[ $TAKEOVER -eq 1 ]]; then
    note "the controller here no longer has any agents; stop it when you are satisfied."
    note "this machine's CLI should now use:  export FERRO_CONTROLLER=$NEW_CONTROLLER"
fi
