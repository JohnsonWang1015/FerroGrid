#!/usr/bin/env bash
# Run scripts/auto_update.sh from a systemd --user timer on the controller host.
#
#   ./scripts/install_autoupdate.sh                 a pass every 2 minutes
#   ./scripts/install_autoupdate.sh --every 10min   any systemd time span
#   ./scripts/install_autoupdate.sh --uninstall
#
# The timer polls GitHub rather than GitHub pushing here: nothing has to reach
# this machine from outside, and no Actions runner -- which a fork's pull
# request can point a workflow at -- runs on the box holding SSH keys to every
# node. A pass deploys only commits whose CI passed, so CI is the gate either way.
#
# Follow it with: journalctl --user -u ferro-autoupdate -f
set -euo pipefail
cd "$(dirname "$0")/.."

EVERY=2min
UNINSTALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --every)     EVERY="${2:?--every needs a time span, e.g. 5min}"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        # Print the header comment, however long it grows.
        -h|--help)   awk 'NR>1 && /^#/ {sub(/^#[[:space:]]?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *)           echo "unknown argument: $1 (try --help)" >&2; exit 1 ;;
    esac
done

say()  { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

UNIT_DIR="$HOME/.config/systemd/user"

if (( UNINSTALL )); then
    say "removing the ferro-autoupdate timer"
    systemctl --user disable --now ferro-autoupdate.timer 2>/dev/null || true
    rm -f "$UNIT_DIR/ferro-autoupdate.service" "$UNIT_DIR/ferro-autoupdate.timer"
    systemctl --user daemon-reload
    exit 0
fi

# What a pass needs, found now rather than in the journal later.
command -v gh >/dev/null && gh auth status >/dev/null 2>&1 \
    || die "gh is not logged in -- passes read CI results through it (gh auth login)"
command -v docker >/dev/null || die "docker is needed for the portable agent build"
systemctl --user cat ferro-controller.service >/dev/null 2>&1 \
    || note "warning: no ferro-controller user unit -- passes will build and update agents, but not the controller"

mkdir -p "$UNIT_DIR"
cat >"$UNIT_DIR/ferro-autoupdate.service" <<UNIT
[Unit]
Description=FerroGrid auto-update (follow CI-green origin/main, rebuild, roll out)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
# systemd --user starts with a minimal PATH: cargo, gh and docker must be on it.
Environment=PATH=%h/.local/bin:%h/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$PWD/scripts/auto_update.sh
# A cold build of both targets takes minutes; a hung SSH must not take forever.
TimeoutStartSec=1h
Nice=10
UNIT

cat >"$UNIT_DIR/ferro-autoupdate.timer" <<UNIT
[Unit]
Description=Run the FerroGrid auto-update every $EVERY

[Timer]
OnActiveSec=30s
# Counted from the end of the previous pass, so a long build never overlaps
# the next one.
OnUnitInactiveSec=$EVERY
RandomizedDelaySec=15s

[Install]
WantedBy=timers.target
UNIT

# Lingering keeps user timers running with nobody logged in.
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable ferro-autoupdate.timer
# restart, not "enable --now": a re-install must re-arm with the new interval.
systemctl --user restart ferro-autoupdate.timer

say "ferro-autoupdate installed: a pass every $EVERY"
note "status:   systemctl --user list-timers ferro-autoupdate.timer"
note "logs:     journalctl --user -u ferro-autoupdate -f"
note "one pass: $PWD/scripts/auto_update.sh [--dry-run]"
