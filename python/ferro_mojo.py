"""Mojo/MAX custom kernels for FerroGrid, with a PyTorch fallback.

Kernels are an optimisation, never a dependency: every entry point here works
whether or not Mojo is installed. `gelu(x)` runs the Mojo kernel when MAX is
available and the kernel compiles, and `torch.nn.functional.gelu` otherwise,
so the same training script runs on a node with no Mojo toolchain.

    from ferro_mojo import gelu, describe

    y = gelu(x)              # Mojo if available, PyTorch if not
    print(describe())        # what actually got loaded, and why

Install the toolchain with `uv sync --all-extras` (see pyproject.toml).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Callable, Optional

#: Directory holding the Mojo source package (must contain __init__.mojo).
KERNEL_DIR_ENV = "FERRO_MOJO_KERNEL_DIR"

#: Set to "0" to force the PyTorch path even when Mojo is available.
ENABLE_ENV = "FERRO_MOJO"


def kernel_dir() -> Path:
    override = os.environ.get(KERNEL_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "mojo" / "kernels"


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "1").lower() not in ("0", "false", "no")


@functools.lru_cache(maxsize=1)
def _load_library() -> tuple[Optional[Any], str]:
    """Compile and load the Mojo kernel package.

    Returns (library, reason). `library` is None when kernels are unavailable
    and `reason` explains why, which is what `describe()` reports. Every
    failure mode is caught: a missing toolchain, a kernel that no longer
    compiles, or a MAX version whose API moved are all "fall back to torch",
    never "crash the training job".
    """
    if not enabled():
        return None, f"disabled via {ENABLE_ENV}=0"

    try:
        from max.experimental.torch import CustomOpLibrary
    except ImportError as e:
        return None, f"MAX not installed ({e.__class__.__name__}); pip install modular"

    d = kernel_dir()
    if not (d / "__init__.mojo").is_file():
        return None, f"no Mojo source package at {d}"

    try:
        # Compilation happens here, on first use, and is cached by MAX.
        return CustomOpLibrary(d), f"loaded from {d}"
    except Exception as e:  # noqa: BLE001 - any failure means "use torch"
        return None, f"kernel load failed: {type(e).__name__}: {e}"


def available() -> bool:
    return _load_library()[0] is not None


def describe() -> dict:
    """Diagnostics for `ferro-mojo-info` and bug reports."""
    lib, reason = _load_library()
    info: dict[str, Any] = {
        "available": lib is not None,
        "reason": reason,
        "kernel_dir": str(kernel_dir()),
        "enabled": enabled(),
    }
    try:
        import importlib.metadata as md

        info["versions"] = {
            p: md.version(p) for p in ("modular", "max", "mojo") if _installed(md, p)
        }
    except Exception:  # noqa: BLE001
        info["versions"] = {}
    return info


def _installed(md, pkg: str) -> bool:
    try:
        md.version(pkg)
        return True
    except Exception:  # noqa: BLE001
        return False


def _mojo_op(name: str) -> Optional[Callable]:
    lib, _ = _load_library()
    if lib is None:
        return None
    try:
        return getattr(lib, name)
    except Exception:  # noqa: BLE001
        return None


SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEFF = 0.044715


def _make_autograd_fn():
    """Wrap the Mojo op in an autograd.Function.

    MAX's custom ops refuse tensors that require grad ("Can't export tensors
    that require gradient"), so calling the op directly inside a model would
    raise and silently fall back to PyTorch -- making a `--activation mojo`
    run secretly a PyTorch run. Wrapping it here means the forward pass really
    does execute the Mojo kernel.

    The backward is the analytic tanh-GELU derivative in PyTorch. Writing it
    in Mojo too would be straightforward, but there is no reason to: see the
    benchmark in the README -- the custom-op bridge is currently slower than
    PyTorch's fused kernel, so the less traffic across it the better.
    """
    import torch

    class MojoGeluFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, op):
            ctx.save_for_backward(x)
            out = torch.empty_like(x)
            # detach(): the op reads the raw buffer and rejects grad-tracking
            # tensors. Autograd is handled by this Function, not by the op.
            op(out, x.detach())
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)
            tanh_inner = torch.tanh(inner)
            d_inner = SQRT_2_OVER_PI * (1 + 3 * GELU_COEFF * x * x)
            grad = 0.5 * (1 + tanh_inner) + 0.5 * x * (1 - tanh_inner * tanh_inner) * d_inner
            # Second return value is the grad for `op`, which is not a tensor.
            return grad_out * grad, None

    return MojoGeluFunction


@functools.lru_cache(maxsize=1)
def _autograd_fn():
    return _make_autograd_fn()


def gelu(x, strict: bool = False):
    """GELU (tanh approximation). Mojo kernel when available, else PyTorch.

    Set `strict=True` to raise instead of falling back, which is what you want
    in a benchmark or a test that is supposed to be exercising Mojo.
    """
    import torch

    op = _mojo_op("ferro_gelu")
    if op is not None and x.is_contiguous():
        try:
            return _autograd_fn().apply(x, op)
        except Exception as e:  # noqa: BLE001
            # A runtime failure (unsupported dtype, shape) must not take the
            # training run down -- degrade to torch and keep going.
            if strict:
                raise RuntimeError(f"Mojo gelu failed and strict=True: {e}") from e
    elif strict:
        raise RuntimeError(f"Mojo gelu unavailable: {_load_library()[1]}")
    return torch.nn.functional.gelu(x, approximate="tanh")


def used_backend(x) -> str:
    """Which path `gelu(x)` would actually take. For tests and diagnostics."""
    op = _mojo_op("ferro_gelu")
    if op is None:
        return "torch (unavailable)"
    if not x.is_contiguous():
        return "torch (non-contiguous input)"
    return "mojo"


class MojoGELU:
    """Drop-in `nn.Module`-style callable for use inside a model."""

    def __call__(self, x):
        return gelu(x)


def _cli() -> None:
    import json

    print(json.dumps(describe(), indent=2))


if __name__ == "__main__":
    _cli()
