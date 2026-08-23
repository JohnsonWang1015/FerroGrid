"""Python side of the FerroGrid Mojo kernel boundary (phase 2 interface).

The whole point of this module is that it is safe to import and call on a node
with no Mojo toolchain: `load()` returns None and callers use the PyTorch path.
Custom kernels are an optimisation, never a dependency of `ferro train`.

    from ferro_mojo import load

    kernel = load("fused_rmsnorm")
    y = kernel(x) if kernel is not None else torch.nn.functional.rms_norm(x, ...)
"""

from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

#: Directory searched for compiled kernels (`<name>.so`).
KERNEL_DIR_ENV = "FERRO_MOJO_KERNEL_DIR"

_cache: dict[str, Optional[Callable]] = {}


def toolchain_available() -> bool:
    """True when a `mojo` compiler is on PATH."""
    return shutil.which("mojo") is not None


def kernel_dir() -> Path:
    return Path(os.environ.get(KERNEL_DIR_ENV, Path(__file__).parent.parent / "mojo" / "build"))


def available_kernels() -> list[str]:
    d = kernel_dir()
    return sorted(p.stem for p in d.glob("*.so")) if d.is_dir() else []


def load(name: str) -> Optional[Callable]:
    """Load a compiled Mojo kernel by name, or return None if unavailable.

    Never raises: a missing kernel is a normal condition, not an error.
    """
    if name in _cache:
        return _cache[name]

    so = kernel_dir() / f"{name}.so"
    kernel: Optional[Callable] = None
    if so.is_file():
        try:
            lib = ctypes.CDLL(str(so))
            kernel = getattr(lib, f"ferro_{name}", None)
        except OSError:
            kernel = None

    _cache[name] = kernel
    return kernel


def describe() -> dict:
    """Diagnostics for `ferro` / bug reports."""
    return {
        "toolchain": toolchain_available(),
        "kernel_dir": str(kernel_dir()),
        "kernels": available_kernels(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), indent=2))
