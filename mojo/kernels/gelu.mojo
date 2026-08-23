# FerroGrid custom kernel: fused tanh-approximation GELU.
#
# Registered as a MAX custom op, so PyTorch can call it through
# `max.experimental.torch.CustomOpLibrary` (see python/ferro_mojo.py).
# The same source compiles for CPU and GPU -- `foreach` handles the
# vectorisation and the device dispatch.

from extensibility import Coord, InputTensor, OutputTensor, foreach, register
from max.gpu.host import DeviceContext
from std.math import tanh


@register("ferro_gelu")
struct FerroGelu:
    """GELU, tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))."""

    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank, static_spec=_],
        ctx: DeviceContext,
    ) raises where output.dtype.is_floating_point():
        @parameter
        def compute[width: Int](coord: Coord[...]) -> SIMD[output.dtype, width]:
            var v = x.load[width](coord)
            # Written with explicit constants rather than a call into a math
            # helper so the whole expression stays in registers.
            comptime SQRT_2_OVER_PI = 0.7978845608028654
            comptime COEFF = 0.044715
            var inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v)
            return 0.5 * v * (1 + tanh(inner))

        foreach[compute, target=target](output, ctx)
