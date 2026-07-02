# A trace-free in-kernel timer read works on Phoenix silicon (Path A crux + identity confirmed)

**Date:** 2026-07-02  **Issue:** #140 (SP-4a cold-start fill-state; the W1 mechanism)
**Status:** CAPABILITY CONFIRMED on hardware. This is a new, reusable
cycle-accuracy tool for the whole project -- it sidesteps both dead ends that
stalled the SP-4a investigation for 3 sessions (trace-unit perturbation; and the
FW-artifact register readback of `2026-05-26-aie-rw-access-not-a-cycle-probe.md`).
**Artifacts:** `build/experiments/pathA-cntr-spike/` (gitignored): `read_cntr.s`,
`cntr_probe.cc/.py`, `identity/`, `build.sh`.

## What was built

An AIE2 core reads its tile timer (`cntr`) from CORE code under Peano and drains
the value to DDR for host readback -- the trace-free in-kernel-timer oracle
(Path A) whose feasibility was gated on three prerequisites, all now settled:

1. **Peano `cntr` read (Q1): WORKS via a hand-assembled `.s` stub.** No C
   builtin / inline-asm path exists (clang AIE `validateAsmConstraint` returns
   false; inline-asm with a vector-typed output CRASHES the backend -- confirmed,
   not theoretical). The robust route: `read_cntr.s` = `mov r17:r16, cntr; ret lr`
   (+5 delay-slot nops), assembled by `llvm-mc -triple aie2`, called from a `.cc`
   declared to return a 64-bit VECTOR type (`v2int32`), ABI-correct by
   construction via `RetCC_AIE2` `AssignToLReg` -> l0 = r17:r16 (the exact pair
   `MOV_CNTR` writes). Verified byte-for-byte in the final linked per-core ELF:
   `read_cntr` at 0x130 = `59 26 00 18`. `link_with` takes one filename, so the
   two TUs are `ld.lld -r` partial-linked first.
2. **Timer reset (Q2): WORKS via a DIRECT register write.** `Timer_Control`
   @ 0x34000, bit 31 (`0x80000000`), write-only "Reset Timer by writing 1" --
   confirmed from aie-rt `XAie_ResetTimer` (`xaie_timer.c:132`,
   `xaiemlgbl_params.h:2648`) AND the AM025 regdb (`aie_registers_aie2.json:6040`).
   One `npu_write32` in the runtime sequence (via IRON `rt.inline_ops`, since a
   bare `npu_write32` inside `rt.sequence` has no live MLIR Context). No event/
   broadcast arming needed for a single tile.
3. **DDR drain + host readback (Q3): routine** (standard objectfifo -> shim ->
   BO). Note: our trace-centric `bridge-trace-runner` mis-classifies the single
   output BO as the trace buffer, so the data lands in `--trace-out` (harmless;
   read it there).

## Hardware results

**Rate is exact.** Two `cntr` reads across a 1000-iteration delay loop
(predicted `24 + 9*1000 = 9024` cycles from hand-counted VLIW bundles) gave
**delta = 9024, every run.** So `cntr` advances at PRECISELY one tick per core
clock. Combined with "no other 64-bit tile-clock counter exists in the ISA",
`cntr` IS the tile timer -- rate-exact, not approximate.

**Free-running.** Across 4 runs the first read `t0` varied (215752..246323) --
`cntr` free-runs continuously from an undefined per-run zero, not reset at
core-start. A common reference requires an explicit reset (Q2).

**Identity CONFIRMED.** A 64-sample sweep (3631 cy/sample) with a
`Timer_Control` reset fired mid-window: samples climb +3631/sample, then at
sample 7 **DROP 241412 -> 1541** (-239871), then resume climbing at the identical
+3631 slope. The reset zeroes the very counter `cntr` reads. `cntr` = tile timer,
resettable -> the two-core common-zero mechanism Path A needs is validated
end-to-end.

## Bonus W1 lead (not yet the full experiment)

The identity kernel is NOT traced, so its `Timer_Control` reset is the FIRST
runtime-sequence instruction. Arithmetic on the drop:
- reset fired 3631-1541 = 2090 cy after sample 6 (free-running `cntr ~= 243502`);
- sample 0 read at `cntr = 219626`.
So the core executed **>= ~23,900 cy of its sampling loop BEFORE the first
runtime-sequence instruction fired** (and more before, in prologue).

**The core genuinely free-runs ~24k cy before the runtime sequence** -- the
"head-start" the warm-up comment asserts is REAL. So the W1 mechanism is NOT
"core starts at command-dispatch"; the core executes well before the drain. The
gate that keeps HW's pipeline empty (per the write32-sweep Horn-A result) is
therefore on the objectfifo DATAFLOW (locks/DMA/produce), NOT on core execution:
this `cntr`-only kernel runs freely during the head-start, but the lean kernel's
producer apparently does not fill during it. Reframes the fix -- the EMU's "cores
run from CDO" is likely faithful for core execution; the defect is the EMU
letting objectfifo dataflow engage/fill during the head-start where HW holds it
(CDO writes config, but HW may not make DMA/locks live until the workload
dispatches; the EMU treats them live immediately).

## Next: the W1 experiment

Instrument the lean kernel's PRODUCER with `cntr` reads (loop entry, per-iteration
after release), drain timestamps to DDR, reset the timer AT the drain dispatch so
each producer timestamp reads large (pre-drain, free-running) or small
(post-drain, post-reset). Splits A1' (producer produced pre-drain -> gate is
downstream) from A2' (producer dataflow didn't engage until the drain), and the
iter-1->iter-2 gap pins the block duration. Design approved (2026-07-02).
