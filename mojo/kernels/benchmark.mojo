# Benchmark harness for FerroGrid Mojo kernels (phase 2 placeholder).
#
# Emits the same `FERRO_METRIC {...}` lines the training scripts use, so a
# kernel benchmark launched through `ferro train` shows its numbers in
# `ferro job` without any special-casing in the controller.

from time import perf_counter_ns
from kernel_api import TensorView, IdentityKernel


fn emit_metric(name: String, iters: Int, ns_per_iter: Float64):
    # Keys match what ferro-controller's metrics parser understands.
    print(
        'FERRO_METRIC {"step": ',
        iters,
        ', "step_time_ms": ',
        ns_per_iter / 1.0e6,
        ', "samples_per_s": ',
        1.0e9 / ns_per_iter,
        "}",
        sep="",
    )


fn bench_identity(numel: Int, iters: Int) raises:
    var buf = UnsafePointer[Float32].alloc(numel)
    var out_buf = UnsafePointer[Float32].alloc(numel)
    for i in range(numel):
        buf[i] = Float32(i)

    var x = TensorView(buf, numel, 1, numel)
    var out = TensorView(out_buf, numel, 1, numel)

    # Warmup, so the timed loop measures steady state.
    for _ in range(8):
        IdentityKernel.run(out, x)

    var start = perf_counter_ns()
    for _ in range(iters):
        IdentityKernel.run(out, x)
    var total = perf_counter_ns() - start

    emit_metric(IdentityKernel.name(), iters, Float64(total) / Float64(iters))

    buf.free()
    out_buf.free()


fn main() raises:
    bench_identity(1 << 16, 100)
