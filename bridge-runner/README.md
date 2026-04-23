# bridge-trace-runner

Generic XRT runner for trace-instrumented AIE xclbins.

Replaces the per-test `test.exe` for bridge tests when the test has been
trace-instrumented via `tools/mlir-trace-inject.py`. Instead of hardcoding
`group_id(1..7)` like traditional bridge tests do, this runner reads the
xclbin's kernarg metadata and dispatches each argument by semantic
position.

## Build

    cmake -S bridge-runner -B bridge-runner/build
    cmake --build bridge-runner/build

## Usage

    bridge-trace-runner \
      --xclbin aie-traced.xclbin \
      --instr insts.bin \
      --input in0.bin \
      --output out0.bin \
      --trace-out trace.bin \
      --trace-size 8192

The runner discovers kernargs in the xclbin and classifies them by type:

  - Scalar args (opcode, instr_size) are passed as integers
  - The instruction buffer is loaded from `--instr`
  - Input buffers get data from `--input <path>` in order; buffers beyond
    the supplied inputs are zero-filled
  - Output buffers are zero-initialized and written to `--output <path>`
    paths in order after the kernel completes
  - The LAST buffer kernarg (after accounting for the above) is treated
    as the trace buffer; its contents are written to `--trace-out`

## Trace buffer naming

The installed mlir-aie tags the trace slot with a generic name like `bo4`
rather than the literal string `trace`. We identify the trace buffer by
its position (last buffer arg) instead of by name, so the runner works
with both older and newer mlir-aie versions.

Exit 0 on success.
