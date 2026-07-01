# bridge-trace-runner under-allocated output-only BOs -> intermittent IO_PAGE_FAULT

**Date:** 2026-07-01. **Status:** root-caused + fixed + silicon-verified.
**Component:** `bridge-runner/bridge-trace-runner.cpp` (shared HW trace-capture runner).
**Surfaced by:** the SP-5b R1 HW runnability gate (#140), which correctly failed loud.

## Symptom

The SP-5b R1 gate (`build/experiments/sp5-skew/r1_gate.sh`, 20 serial Phoenix
runs) failed on `iommu_delta`: the NPU (`amdxdna 0000:c6:00.1`) emitted
`IO_PAGE_FAULT`s (`domain=0x0001 flags=0x0027`) on 13/20 runs, 1-3 each,
intermittent and clustered. All runs were rc-0, produced valid traces (~2100
events), and never TDR'd. `flags=0x0027` = write to a not-present page.

## Root cause (definitive, address-correlated on silicon)

The runner allocated the kernel's **output data BO at 8 bytes** instead of its
real 8192-byte extent; the kernel's output-drain DMA (2048 int32 = 8192 B)
overran it. A verbose capture logging each BO's device address alongside the
fresh dmesg fault lines showed, every dirty run, the identical signature:

```
output BO (karg 3): 8 bytes at 0x…69000  (one page mapped: …69000–…69fff)
faults:            0x…6a000, 0x…6a700, 0x…6ae00  = BO_base + 0x1000, +0x1700, +0x1e00
```

The 8192-byte drain spans two pages; only one was mapped -> the second page
faults. "Clean" runs (7/20) were clean only because that second page happened
to be mapped by a neighbor -> **silent DDR corruption**, not actual safety.

Why 8 bytes: `middle_slot_size()` returns `karg.size`, and XRT kernel-arg
metadata reports an NPU buffer arg as its **8-byte pointer size**, not its
extent. Input buffers escape the bug (sized from the `--input` file); an
**output-only** buffer (this kernel is Q=0 -- the drain is the only DDR access)
has no file to infer from, so it got 8 bytes.

## Hypotheses falsified first (so the fix wasn't a guess)

- **Trace-buffer overflow (the pre-existing recorded hypothesis).** FALSE. The
  compiled trace shim DMA BD is a single one-shot descriptor with
  `buffer_length = 4096` words = 16384 B = exactly the trace BO
  (`aie_traced.mlir` `writebd bd_id=15`; `memref<16384xi8>` trace arg). It
  cannot overrun by BD length. Its recorded fix (bump `--trace-size`) would not
  even have touched the faulting buffer (`karg 3`), and `--trace-size` sizes
  only the BO, not the compiled BD.
- **Unpatched / stale-address BD.** FALSE. insts.bin has exactly 2 `DDR_PATCH`
  ops; both output (arg_idx 0) and trace (arg_idx 1) BOs get valid device
  addresses.
- **Shim S2MM free-running past `buffer_length` in packet mode.** FALSE per
  aie-rt: the shim S2MM is BD/task-queue-driven, address generation is
  BD-bounded, and an empty task queue idles the channel (no DDR writes). (A
  runner comment claiming "no length check fires" is a real but unrelated
  batch-mode cumulative-offset note -- a red herring here.)

## Fix (toolchain-derived, at the source)

`discover_arg_sizes_from_insts()` recovers each data buffer's real DMA transfer
length from the compiled kernel: mlir-aie's NPU lowering emits, per DDR buffer,
a BLOCKWRITE that programs the shim BD -- **payload word 0 = Buffer_Length in
32-bit words** -- immediately followed by the DdrPatch carrying that buffer's
arg_idx. So `required_bytes = (preceding BLOCKWRITE payload[0]) * 4`, keyed to
the DdrPatch arg_idx (the same 0-based data-buffer index `--trace-buf-idx`
uses). The per-run sizing floors each middle BO by this. Empty on parse failure
-> falls back to the declared size, so already-correct kernels are unaffected.

Verified against sp5_skew_r1's insts.bin: `{arg 0: 8192, arg 1: 16384}` (output,
trace) -- both matching the BDs and the trace-config `size_bytes`.

## Verification (silicon)

- 12 verbose runs at the fault size (16384): output BO now 8192 (2 pages), the
  `[bo-size]` floor fires every run, **0/12 faults** (was ~65%).
- Full R1 gate re-run: **20/20 clean** (`iommu_delta=0`, `tdr_delta=0`), all
  three `d_v` core-core pairs range-0, non-degenerate `dn_core=[2,3,4]`. Intra
  contrast range 8 -> PROVISIONAL, the designed-for Q2-A outcome (`d_v` is the
  gate; contrast is best-effort). The R1 instrument is now green on Phoenix.
- Regression guard: `tools/test_bridge_runner_arg_sizes.py` (synthetic + real
  artifact) locks the extraction against parser regressions and mlir-aie
  lowering drift.

## Scope / audit

The bug is in the **shared** runner, so any bridge-trace-runner capture with an
output buffer larger than the 8-byte placeholder and no `--input` was affected.
`build/experiments/sp3-spike-trace/task3_gate.sh` (and other drivers:
`tools/trace_capture.py`, `trace-sweep.py`, `run_sweep.py`, ...) drive the runner
the same way and **do not check IOMMU faults** (only TDR), so any faults/silent
corruption there went unseen. SP-3's conclusions still stand -- its output is a
Q=0 garbage completion-sink, and its results are about trace-event
reproducibility (stable 20/20), not output data. The fix now protects all
captures. Recommended follow-up (low priority): backport r1_gate.sh's
`iommu_fault_count` check into task3_gate.sh and the other drivers so a future
regression fails loud rather than silently corrupting.

Kept: the `--verbose` `[bo]` device-address logging added for this
investigation -- it is the tool that turns an opaque IOMMU fault into a
buffer-attributable one, and belongs in the runner permanently.
