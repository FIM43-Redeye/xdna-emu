"""Shared defaults for perfcnt-anchored trace tooling.

The Rust counterpart lives in `src/trace/compare.rs` as
`DEFAULT_PERFCNT_PERIOD`.  Keep the two values in sync if either changes.
"""


DEFAULT_PERFCNT_PERIOD = 1024
"""Default cycles between PERF_CNT_0 overflows.

Used by `mlir-trace-inject.py` when `--perfcnt-period` is not specified
and by `trace-sweep.py` for the same flag.  `trace-compare`'s Rust side
falls back to the same constant when neither HW nor EMU traces have
enough ticks to estimate the period from observed PCs.

The number is a tradeoff: smaller values give finer cycle bands but
generate more trace volume; larger values reduce volume but coarsen
band resolution.  1024 is wide enough that a kernel iterating thousands
of times doesn't drown the trace buffer, narrow enough that even short
kernels see multiple ticks within one batch.
"""
