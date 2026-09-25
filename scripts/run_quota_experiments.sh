#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

seed_count="${QUOTA_SEEDS:-20}"
output_dir="outputs/benchmarks/quota"
protoc_wrapper_dir=""

if [[ -z "${PROTOC:-}" ]] && ! command -v protoc >/dev/null 2>&1; then
  torch_protoc="$(find "$repo_root/.venv/lib" -path '*/site-packages/torch/bin/protoc' -type f -print -quit 2>/dev/null || true)"
  if [[ -n "$torch_protoc" ]]; then
    protoc_wrapper_dir="$(mktemp -d)"
    cat > "$protoc_wrapper_dir/protoc" <<EOF
#!/usr/bin/env bash
exec "$torch_protoc" --experimental_allow_proto3_optional "\$@"
EOF
    chmod +x "$protoc_wrapper_dir/protoc"
    export PROTOC="$protoc_wrapper_dir/protoc"
  fi
fi

cleanup() {
  if [[ -n "$protoc_wrapper_dir" ]]; then
    rm -rf "$protoc_wrapper_dir"
  fi
}
trap cleanup EXIT

cargo build --release -p ferro-sim
./target/release/ferro-sim quota --out "$output_dir" --seeds "$seed_count"
python3 python/tools/plot_quota_results.py \
  --input "$output_dir/summary.csv" \
  --out "$output_dir/figures"
