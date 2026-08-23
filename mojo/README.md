# FerroGrid Mojo / MAX kernels

Real, working Mojo custom kernels — compiled, numerically verified, and
callable from PyTorch on both CPU and GPU. Kernels remain **optional**:
everything falls back to the PyTorch op when Mojo is not installed.

## Layout

| File | Purpose |
|---|---|
| `kernels/__init__.mojo` | Makes `kernels/` a Mojo source package — required by `CustomOpLibrary` |
| `kernels/gelu.mojo` | `ferro_gelu`: fused tanh-approximation GELU, one source for CPU and GPU |
| `../python/ferro_mojo.py` | Loader, autograd wrapper, and PyTorch fallback |
| `../python/examples/bench_mojo.py` | Correctness check + benchmark vs PyTorch |

## Install

```bash
uv sync --all-extras          # pulls modular (max + mojo)
uv run ferro-mojo-info        # report what loaded
```

## Use

```python
from ferro_mojo import gelu

y = gelu(x)                   # Mojo when available, PyTorch when not
y = gelu(x, strict=True)      # raise instead of falling back
```

In the reference model: `ferro train ... train_fsdp2.py --activation mojo`.

## How it hangs together

`CustomOpLibrary(Path("mojo/kernels"))` compiles the package on first use and
caches the result. Each `@register("name")` struct becomes an attribute on the
library object.

Two things about this are easy to get wrong:

- **The directory needs `__init__.mojo`.** Without it MAX rejects the path
  with "must be a Mojo source or binary package". Passing the `.mojo` file
  directly does not work either.
- **The op rejects tensors that require grad** (`BufferError: Can't export
  tensors that require gradient`). Calling it straight from a model therefore
  raises, and a naive `try/except` fallback turns a "Mojo" run into a silent
  PyTorch run. `ferro_mojo` wraps the op in a `torch.autograd.Function` that
  detaches for the forward and supplies the analytic derivative, so the
  forward pass genuinely executes the Mojo kernel. Verified with
  `torch.autograd.gradcheck`.

## Measured performance — read this before writing a kernel

RTX 3090, fp32, `torch.no_grad`, forward only:

| Elements | PyTorch | Mojo | Mojo / PyTorch |
|---|---|---|---|
| 4,096 | 4.6 µs | 196.7 µs | 43× slower |
| 262,144 | 4.4 µs | 176.5 µs | 40× slower |
| 4,194,304 | 42.7 µs | 608.0 µs | 14× slower |
| 33,554,432 | 320.8 µs | 2,111 µs | 6.6× slower |
| 134,217,728 | 1,325 µs | 9,317 µs | 7.0× slower |

End-to-end training (4 layers, d_model 768, batch 8, seq 512, single GPU):
**100,175 tok/s with PyTorch GELU vs 85,008 tok/s with the Mojo kernel.**

Two separate effects are visible:

1. **A fixed ~180 µs per-call cost** through the PyTorch↔MAX bridge. It
   dominates small tensors completely (43× at 4K elements) and only amortises
   above ~10M elements.
2. **The generated kernel is ~7× off roofline even when overhead amortises.**
   PyTorch reaches 104 Gelem/s here, close to the RTX 3090's memory-bandwidth
   ceiling for an 8 byte/element elementwise op; this kernel manages 15.9.

So: **replacing a single well-optimised PyTorch elementwise op with a Mojo
custom op is currently a loss.** That is why `--activation torch` is the
default and `mojo` is opt-in.

Where a custom kernel can still win:

- **Fusing several ops into one**, so one bridge crossing replaces many kernel
  launches and several round-trips to HBM.
- **Operations PyTorch has no fused kernel for**, where the baseline is a
  chain of eager ops rather than one tuned CUDA kernel.
- **Large tensors**, where the fixed cost stops mattering.

Measure before adopting; `bench_mojo.py` refuses to report a speedup if the
numerics do not match, and uses `strict=True` so it cannot accidentally
benchmark PyTorch against itself.

## Adding a kernel

1. Write `kernels/<name>.mojo` with a `@register("<name>")` struct exposing a
   static `execute`.
2. Re-export it from `kernels/__init__.mojo`.
3. Add a wrapper in `python/ferro_mojo.py` (with a fallback, and an
   `autograd.Function` if it is used in training).
4. Extend `bench_mojo.py` and confirm both correctness and a real speedup.

### Mojo 1.0 syntax notes

The language changed significantly at 1.0; older examples will not compile.

- `fn` is gone — use `def`.
- `alias` is deprecated — use `comptime`.
- Imports are `std.`-prefixed: `from std.math import tanh`.
- Kernel API lives in `extensibility`: `register`, `InputTensor`,
  `OutputTensor`, `foreach`, `Coord`.
- Unbind unused parameters explicitly: `InputTensor[..., static_spec=_]`.
- Variadic parameters use `Coord[...]`, not `Coord[**_]`.
- Constrain dtypes with a `where` clause, and mark `raises` when calling
  `foreach`.
