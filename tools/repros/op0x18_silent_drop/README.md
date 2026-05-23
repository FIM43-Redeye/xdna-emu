# `op0x18_silent_drop` — `MSG_OP_CHAIN_EXEC_NPU` deterministic reproducer

Minimal C++ reproducer for the firmware silent-drop on op 0x18
(`MSG_OP_CHAIN_EXEC_NPU`) observed on AMD Ryzen AI Phoenix (NPU1).

## What it does

Submits the `add_one_ctrl_packet` kernel (from the mlir-aie test
suite) with the control-packet input buffer (`bo_ctrlIn`) left
zero-filled. The kernel's compute core acquires `input_lock0` and
waits for a real control packet to release it — with `bo_ctrlIn`
zero, the core blocks indefinitely, no data ever reaches the shim
S2MM output channel, and the firmware op 0x18 handler never raises
its completion IRQ.

Each iteration creates a fresh `xrt::hw_context` so each silent-drop
is independently isolated by the driver's TDR; this measures the
per-submission failure rate rather than any cascade behavior.

## Build

```
g++ -std=c++17 -O2 op0x18_repro.cpp -o op0x18_repro \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
    -lxrt_coreutil -lpthread -Wl,-rpath,/opt/xilinx/xrt/lib
```

## Run

```
./op0x18_repro <aie.xclbin> <insts.bin> [iterations=10]
```

Uses `add_one_ctrl_packet`'s pre-built artifacts from a mlir-aie
build tree, e.g.:

```
./op0x18_repro \
  $MLIR_AIE/build/test/npu-xrt/add_one_ctrl_packet/chess/aie.xclbin \
  $MLIR_AIE/build/test/npu-xrt/add_one_ctrl_packet/chess/insts.bin \
  10
```

Run *without* `XDNA_EMU` in the environment — this reproducer
targets real hardware via the stock XRT plugin, not the emulator.

## Expected output

On Phoenix NPU1 with firmware 1.5.5.391 and amdxdna 2.23.0
(`drivers/accel` tree), every iteration returns `state=8`
(`ERT_CMD_STATE_TIMEOUT`) after ~4080ms (the driver's TDR firing).
See `run.log` for verbatim output from a verified run.

dmesg signature for this scenario, per iteration: **+1 `TDR
timeout`, +1 `ret -22`, 0 `status 0x2000003` (NOAVAIL)**. The zero
NOAVAIL is essential — it confirms each silent-drop is isolated and
no column-leak cascade fires (the cascade is a separate failure
mode, observable under sustained load through the same path).

## Context

Backs the upstream xdna-driver issue post (drafting in progress —
see `NEXT-STEPS.md` Thread B and the finding doc
`docs/superpowers/findings/2026-05-22-chain-exec-npu-silent-drop-captured.md`
for the full investigation history).

Verified 2026-05-22 evening: 10/10 deterministic, ~4080ms each.
