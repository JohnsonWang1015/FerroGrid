"""Tests for the Mojo/MAX kernel layer.

These run with or without a Mojo toolchain: the fallback tests always run, the
kernel tests skip when MAX or a GPU is unavailable. That is deliberate -- the
promise this module makes is "works either way", so the test suite has to hold
on a machine with no Mojo installed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import ferro_mojo  # noqa: E402

torch = pytest.importorskip("torch")

mojo_only = pytest.mark.skipif(
    not ferro_mojo.available(), reason="Mojo/MAX not installed"
)


def test_describe_is_always_answerable():
    info = ferro_mojo.describe()
    assert set(info) >= {"available", "reason", "kernel_dir", "enabled"}
    assert isinstance(info["available"], bool)
    assert info["reason"], "an unavailable kernel must explain itself"


def test_gelu_matches_torch_on_cpu():
    """Holds whichever backend is used -- that is the point of the fallback."""
    x = torch.randn(32, 64)
    got = ferro_mojo.gelu(x)
    want = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(got, want, atol=1e-5)


def test_disabled_via_env_falls_back(monkeypatch):
    monkeypatch.setenv(ferro_mojo.ENABLE_ENV, "0")
    ferro_mojo._load_library.cache_clear()
    try:
        assert not ferro_mojo.available()
        assert "disabled" in ferro_mojo.describe()["reason"]
        # Still computes the right answer via PyTorch.
        x = torch.randn(8, 8)
        assert torch.allclose(
            ferro_mojo.gelu(x),
            torch.nn.functional.gelu(x, approximate="tanh"),
            atol=1e-6,
        )
    finally:
        ferro_mojo._load_library.cache_clear()


def test_strict_raises_when_unavailable(monkeypatch):
    monkeypatch.setenv(ferro_mojo.ENABLE_ENV, "0")
    ferro_mojo._load_library.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            ferro_mojo.gelu(torch.randn(4, 4), strict=True)
    finally:
        ferro_mojo._load_library.cache_clear()


@mojo_only
def test_kernel_is_actually_used():
    """Guards the failure mode where 'Mojo' silently means PyTorch."""
    assert ferro_mojo.used_backend(torch.randn(16, 16)) == "mojo"
    # A non-contiguous view cannot go to the kernel and must say so.
    assert "torch" in ferro_mojo.used_backend(torch.randn(16, 16).t())


@mojo_only
def test_forward_runs_mojo_under_autograd():
    x = torch.randn(16, 16, requires_grad=True)
    y = ferro_mojo.gelu(x, strict=True)
    # MAX ops reject grad-tracking tensors, so this only passes because of the
    # autograd.Function wrapper -- without it we would have silently used torch.
    assert type(y.grad_fn).__name__ == "MojoGeluFunctionBackward"


@mojo_only
def test_gradients_match_torch():
    x = torch.randn(32, 32, requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)

    ferro_mojo.gelu(x, strict=True).sum().backward()
    torch.nn.functional.gelu(ref, approximate="tanh").sum().backward()

    assert torch.allclose(x.grad, ref.grad, atol=1e-5)


@mojo_only
def test_gradcheck():
    # float32 with loose tolerances: gradcheck prefers float64, but the kernel
    # is float32 and a float64 input would quietly take the fallback path.
    x = torch.randn(6, 6, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: ferro_mojo.gelu(t, strict=True), (x,), eps=1e-3, atol=1e-2, rtol=1e-2
    )


@mojo_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_matches_torch():
    x = torch.randn(128, 256, device="cuda")
    got = ferro_mojo.gelu(x, strict=True)
    want = torch.nn.functional.gelu(x, approximate="tanh")
    assert (got - want).abs().max().item() < 1e-5
