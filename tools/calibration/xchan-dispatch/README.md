# Cross-channel dispatch-gate probe (#140)

Two-dispatch microbenchmark that measures the gap between two **independent**
shim DMA channels' `START_TASK` events when dispatched back-to-back. Settles
whether the EMU's single shared controller dispatch gate
(`controller_next_taskq_cycle`, `src/npu/executor.rs`) over-serializes
independent channels, or faithfully models a real HW serialization.

Forked from `_diag_shim_chain_sweep/k1`: one shim MM2S_0 BD and one shim
S2MM_0 BD dispatched back-to-back (no `dma_wait` between). MM2S_0 + S2MM_0 are
data channels (their starts fire after trace setup, so both are captured);
S2MM_1 is left free for the trace-data drain (the trace-drain channel's own
START is unmeasurable -- it starts before the trace timer is alive).

- `mm2s-first.aie.mlir` -- MM2S_0 dispatched first; gate armed by
  `dispatch_mm2s(0)` base.
- `s2mm-first.aie.mlir` -- S2MM_0 dispatched first; gate armed by `dispatch_s2mm`
  flat. This is the lean SP-4a ordering (trace-drain S2MM then of_out S2MM).
- `test.cpp` -- completion-only launcher (reused from `_diag_shim_chain_sweep/k1`).

## Result (2026-06-30, HW NPU1 vs EMU)

| order | EMU gap | HW gap | verdict |
|-------|---------|--------|---------|
| MM2S-first | 1086 | bimodal 920 / 5623 | EMU between HW modes |
| **S2MM-first** | **3050** | **stable 2922-3055** | **faithful ~2%** |

The shared dispatch gate is **HW-faithful** for the SP-4a-relevant S2MM-first
case. Full writeup:
`docs/superpowers/findings/2026-06-29-coldstart-headstart-trace-baseline.md`.

## Build + run recipe

```bash
# Stage a test dir (trace-prepare expects aie.mlir + test.cpp side by side):
D=/home/triple/npu-work/mlir-aie/test/npu-xrt/_diag_xchan_dispatch
mkdir -p "$D"
cp tools/calibration/xchan-dispatch/s2mm-first.aie.mlir "$D/aie.mlir"   # or mm2s-first
cp tools/calibration/xchan-dispatch/test.cpp "$D/test.cpp"

# Inject trace + compile:
python3 tools/trace-prepare.py "$D" --output build/experiments/xchan/build \
  --device npu1_1col --shim-sweep-events all --trace-mode event_time --trace-size 16384
( cd build/experiments/xchan/build && aiecc.py --no-aiesim --aie-generate-xclbin \
  --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential \
  --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_traced.mlir )

# Run (EMU: XDNA_EMU=1; HW: env -u XDNA_EMU), then decode and read off the gap
# between DMA_MM2S_0_START_TASK and DMA_S2MM_0_START_TASK on the shim (row 0).
```
