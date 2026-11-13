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
      --trace-size 8192 \
      --post-completion-us 100

The runner discovers kernargs in the xclbin and classifies them by type:

  - Scalar args (opcode, instr_size) are passed as integers
  - The instruction buffer is loaded from `--instr`
  - Input buffers get data from `--input <path>` in order; buffers beyond
    the supplied inputs are zero-filled
  - Output buffers are zero-initialized and written to `--output <path>`
    paths in order after the kernel completes
  - The LAST buffer kernarg (after accounting for the above) is treated
    as the trace buffer; its contents are written to `--trace-out`

`--post-completion-us` optionally keeps the completed hardware context alive
before syncing the trace BO. It is a measurement aid for observing autonomous
post-command activity, defaults to zero, and is capped at one second.

## Host-buffer bounds

XRT's kernarg metadata gives an NPU buffer argument only its 8-byte pointer
size, so the runner sizes each data buffer from the instruction stream
itself (`insts_bounds.h`): it decodes the stream completely, evaluates every
shim DMA task the stream pushes (Task_Queue write, BD chain, the BD's length,
dimensions and iteration, and the DdrPatch that binds the BD to a buffer) to
the byte range it reaches, and allocates each buffer up to the largest end.

Before any buffer is allocated or the run submitted, the runner then checks every task
against the buffers it actually allocates (including the trace buffer's
`--trace-size` and buffers sized from `--input` files) and refuses the run,
exit 1 with each offending task named, if any task leaves its buffer. A
stream it cannot decode completely (an opcode it does not know, an op that
does not fit, a task on a BD with no host-buffer patch) is refused outright
(a single-shot run: exit 2, before the device is opened): its tasks' reach is
unknown.

Two device-free modes expose the same computation:

    bridge-trace-runner --print-arg-sizes insts.bin          # "IDX BYTES" per data buffer
    bridge-trace-runner --check-bounds insts.bin 0=4096 1=65536
        # every task's range; exit 0 inside, 3 outside, 2 undecodable

## Trace buffer naming

The installed mlir-aie tags the trace slot with a generic name like `bo4`
rather than the literal string `trace`. We identify the trace buffer by
its position (last buffer arg) instead of by name, so the runner works
with both older and newer mlir-aie versions.

Exit 0 on success.
