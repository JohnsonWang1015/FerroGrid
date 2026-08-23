"""One-command setup for FerroGrid: `uv run --all-extras ferro-setup`.

Builds the Rust binaries, links them onto PATH, and reports whether the
Mojo/MAX kernels loaded. Safe to re-run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BINARIES = ("ferro", "ferro-agent", "ferro-controller")


def repo_root() -> Path:
    # Installed into the venv, so locate the repo from the CWD upwards rather
    # than from __file__, which lives in site-packages.
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "Cargo.toml").is_file() and (candidate / "crates").is_dir():
            return candidate
    sys.exit("error: run ferro-setup from inside the FerroGrid checkout")


def find_cargo() -> str | None:
    """Locate cargo.

    `uv run` does not source your shell profile, so rustup's ~/.cargo/bin is
    usually missing from PATH here even though `cargo` works in your terminal.
    Check the standard rustup location before giving up.
    """
    found = shutil.which("cargo")
    if found:
        return found
    for candidate in (
        Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "bin" / "cargo",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run(cmd: list[str], **kw) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, **kw)


def build_rust(root: Path, portable: bool) -> Path:
    cargo = find_cargo()
    if cargo is None:
        sys.exit(
            "error: cargo not found. Install Rust:\n"
            "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n"
            "then re-run this command."
        )

    # scripts/build.sh shells out to cargo, so make sure it can find the same
    # one we did even when PATH came from `uv run` rather than a login shell.
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(cargo).parent), env.get("PATH", "")])

    if portable:
        print("==> building portable binaries (glibc 2.31 container)")
        if run([str(root / "scripts" / "build.sh"), "portable"], cwd=root, env=env) != 0:
            sys.exit("error: portable build failed")
        return root / "target" / "portable" / "release"

    print("==> building release binaries")
    if run([cargo, "build", "--release", "--workspace"], cwd=root, env=env) != 0:
        sys.exit("error: cargo build failed")
    return root / "target" / "release"


def link_binaries(out_dir: Path, bin_dir: Path) -> None:
    print(f"==> linking binaries into {bin_dir}")
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in BINARIES:
        src = out_dir / name
        if not src.is_file():
            print(f"  ! {name} not found in {out_dir}, skipping")
            continue
        dst = bin_dir / name
        # Symlink so a later `cargo build --release` is picked up for free.
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
        print(f"  {name} -> {src}")

    if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"\n  ! {bin_dir} is not on your PATH. Add this to your shell rc:")
        print(f'      export PATH="{bin_dir}:$PATH"')


def report_mojo() -> None:
    print("==> Mojo / MAX kernels")
    try:
        import ferro_mojo
    except ImportError:
        print("  ferro_mojo not importable (run inside `uv run`)")
        return
    info = ferro_mojo.describe()
    if info["available"]:
        versions = ", ".join(f"{k} {v}" for k, v in info["versions"].items())
        print(f"  available ({versions})")
    else:
        print(f"  not available: {info['reason']}")
        print("  -> training still works; it falls back to PyTorch ops")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up FerroGrid")
    ap.add_argument(
        "--portable",
        action="store_true",
        help="build in a glibc-2.31 container for deploying to older servers",
    )
    ap.add_argument(
        "--bin-dir",
        type=Path,
        default=Path.home() / ".local" / "bin",
        help="where to link the binaries (default: ~/.local/bin)",
    )
    args = ap.parse_args()

    root = repo_root()
    print(f"FerroGrid at {root}\n")
    out_dir = build_rust(root, args.portable)
    link_binaries(out_dir, args.bin_dir)
    report_mojo()
    print("\nDone. Try:  ferro --help")


if __name__ == "__main__":
    main()
