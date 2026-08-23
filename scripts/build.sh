#!/usr/bin/env bash
# Build FerroGrid binaries.
#
#   ./scripts/build.sh              release build for this machine
#   ./scripts/build.sh portable     build inside a glibc-2.31 container, so the
#                                   binaries run on Ubuntu 20.04 and newer
#
# Why not a static musl build? nvml-wrapper dlopen()s libnvidia-ml.so at
# runtime, and a statically linked musl binary has no dynamic loader -- the
# agent starts but reports zero GPUs. The agent must therefore be dynamically
# linked; building against the OLDEST glibc in the fleet is what makes it
# portable across the lab.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-native}"

if [[ "$MODE" == "portable" ]]; then
    echo "==> building portable binaries in a glibc-2.31 container"
    docker run --rm \
        -v "$PWD":/src -w /src \
        -v "$PWD/.cargo-container-registry":/usr/local/cargo/registry \
        -e CARGO_TARGET_DIR=/src/target/portable \
        rust:1-bullseye \
        bash -c 'set -e
            apt-get update -qq
            apt-get install -y -qq protobuf-compiler >/dev/null
            cargo build --release'
    OUT="target/portable/release"
    # The container builds as root; hand the artefacts back to the caller.
    if [[ -n "${SUDO_USER:-}" ]] || [[ ! -w "$OUT/ferro-agent" ]]; then
        sudo chown -R "$(id -u):$(id -g)" target/portable .cargo-container-registry 2>/dev/null || true
    fi
else
    echo "==> building release binaries for this machine"
    cargo build --release
    OUT="target/release"
fi

echo
echo "binaries in $OUT:"
for b in ferro ferro-agent ferro-controller; do
    [[ -f "$OUT/$b" ]] && ls -lh "$OUT/$b" | awk '{printf "  %-14s %s\n", "'"$b"'", $5}'
done

if [[ "$MODE" == "portable" ]]; then
    echo
    echo "glibc requirement:"
    objdump -T "$OUT/ferro-agent" 2>/dev/null \
        | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 | sed 's/^/  /'
fi
