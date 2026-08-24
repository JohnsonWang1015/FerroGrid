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

MARKER = "# added by ferro-setup"


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


def on_path(bin_dir: Path, path: str | None = None) -> bool:
    """Is bin_dir already reachable through PATH?

    Compare resolved paths, not strings: an entry can be spelled with a
    trailing slash, through a symlinked home, or relative, and still be the
    same directory. A string compare there warns about a PATH that works.
    """
    if path is None:
        path = os.environ.get("PATH", "")
    target = bin_dir.expanduser().resolve()
    for entry in path.split(os.pathsep):
        if not entry:
            continue
        try:
            if Path(entry).expanduser().resolve() == target:
                return True
        except OSError:
            continue
    return False


def shell_rc(shell: str | None = None) -> tuple[Path, str] | None:
    """Pick the rc file for the user's shell and the line that extends PATH.

    Returns None for a shell we do not know how to edit, so the caller can
    fall back to printing the hint.
    """
    if shell is None:
        shell = os.environ.get("SHELL", "")
    name = Path(shell).name
    home = Path.home()
    if name == "bash":
        return home / ".bashrc", 'export PATH="{bin_dir}:$PATH"'
    if name == "zsh":
        zdotdir = os.environ.get("ZDOTDIR")
        return Path(zdotdir or home) / ".zshrc", 'export PATH="{bin_dir}:$PATH"'
    if name == "fish":
        return home / ".config" / "fish" / "config.fish", 'fish_add_path "{bin_dir}"'
    return None


def add_to_path(bin_dir: Path) -> Path | None:
    """Append a PATH line to the shell rc file. Idempotent.

    Returns the file written, or None if the shell is unsupported or the line
    is already there.
    """
    target = shell_rc()
    if target is None:
        return None
    rc, template = target
    line = template.format(bin_dir=bin_dir)
    if rc.is_file() and line in rc.read_text():
        print(f"  {rc} already extends PATH, leaving it alone")
        return None
    rc.parent.mkdir(parents=True, exist_ok=True)
    with rc.open("a") as fh:
        fh.write(f"\n{MARKER}\n{line}\n")
    print(f"  appended to {rc}:  {line}")
    return rc


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
    ap.add_argument(
        "--add-to-path",
        action="store_true",
        help="append the PATH line to your shell rc instead of just printing it",
    )
    args = ap.parse_args()

    root = repo_root()
    print(f"FerroGrid at {root}\n")
    out_dir = build_rust(root, args.portable)
    link_binaries(out_dir, args.bin_dir)
    report_mojo()

    # PATH last: it is the one thing the user may still have to act on, and a
    # warning printed mid-run scrolls away above `Done. Try: ferro --help`.
    reachable = on_path(args.bin_dir)
    written = None
    if not reachable and args.add_to_path:
        print("==> PATH")
        written = add_to_path(args.bin_dir)

    if reachable:
        print("\nDone. Try:  ferro --help")
    elif written is not None:
        print(f"\nDone. Open a new shell (or `source {written}`), then:  ferro --help")
    else:
        print(f"\nDone, but {args.bin_dir} is not on your PATH.")
        print("  Add this to your shell rc:")
        print(f'      export PATH="{args.bin_dir}:$PATH"')
        print("  or re-run with --add-to-path to have it appended for you.")
        print(f"\nUntil then:  {args.bin_dir}/ferro --help")


if __name__ == "__main__":
    main()
