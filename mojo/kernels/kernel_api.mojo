# FerroGrid custom-kernel interface (phase 2 placeholder).
#
# Every FerroGrid kernel exposes this shape so the Python loader and the
# benchmark harness can treat kernels uniformly. No kernel is implemented yet;
# this file fixes the contract the implementations will satisfy.

from memory import UnsafePointer


@fieldwise_init
struct TensorView(Copyable, Movable):
    """A borrowed, contiguous view of a device tensor owned by PyTorch.

    FerroGrid never allocates training tensors in Mojo: torch owns the memory,
    Mojo kernels operate on it in place.
    """

    var data: UnsafePointer[Float32]
    var numel: Int
    var rows: Int
    var cols: Int


trait FerroKernel:
    """Contract implemented by each custom kernel."""

    @staticmethod
    fn name() -> String:
        """Stable identifier used by `ferro_mojo.load(name)`."""
        ...

    @staticmethod
    fn run(inout out: TensorView, x: TensorView) raises:
        """Execute the kernel, writing into `out`."""
        ...


struct IdentityKernel(FerroKernel):
    """Reference implementation: proves the plumbing end to end.

    Deliberately trivial -- it exists so the loader, the benchmark harness and
    the fallback path can be tested before any real kernel is written.
    """

    @staticmethod
    fn name() -> String:
        return "identity"

    @staticmethod
    fn run(inout out: TensorView, x: TensorView) raises:
        for i in range(x.numel):
            out.data[i] = x.data[i]
