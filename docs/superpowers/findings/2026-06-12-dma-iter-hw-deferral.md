# DMA `iter` feature deferred: passes emulator, fails NPU1 silicon (2026-06-12)

Part of the 3b DMA work ([[framework-step3a-dma-tenant]] follow-on). The `iter`
(shim BD iteration) and `chain` (memtile multi-BD `next_bd`) features were built
as generator + lowering only -- the 3a spike's claim that each needed an
emulator-engine fix was wrong (both already execute correctly on the emulator;
the spike's stalls were degenerate hand-written MLIR). See the commit message and
the `chain` doc-comment in `table.rs`.

**`chain` shipped** (silicon-clean, 6/6 keys, byte-exact on NPU1). **`iter` is
deferred** -- it runs on the emulator but fails on real silicon. This note
captures the evidence and the redesign plan so the follow-up starts informed.

## What `iter` is

The AIE2 shim DMA BD has 2 hardware transfer dims (D0/D1; no D2 on the shim) plus
a separate ITERATION register (`iteration_wrap`, `iteration_stepsize`) that
repeats the whole dimensional pattern with a per-iteration address offset.
`iter` exercises that register.

Per the toolchain (mlir-aie `AIEXDialect.cpp` `getHardwareStridesWraps` +
`AIEDmaToNpu.cpp:380`), an access-pattern's dims map
`[outermost..innermost] -> [iteration, D2, D1, D0]`, and `Buffer_Length` is the
inner product (excludes the iteration dim). So a shim iteration BD needs a
**4-entry** pattern `[<iter>, <D2=1>, <D1=1>, <D0>]` with a degenerate D2.

Critically, the iteration must be **non-foldable**: if the iteration stride is
contiguous with the inner extent, aiecc folds the whole pattern back to a plain
linear BD and no iteration is ever emitted (the failure that made the spike's
hand-written `iter_shim` case degenerate -- it produced a plain 16-word linear
BD with `iteration_wrap=0`). The built generator used 2 iterations over a
`region/2` inner block with stride `inner/2` (overlapping halves) to stay
non-foldable. **That overlap is the root of the HW failure.**

## HW campaign results (NPU1, target-hits 3)

`chain`: 60 pass, 0 fail, 0 error, 0 CRASH -> all 6 keys silicon-clean.

`iter`: 0 pass, 30 fail (s2mm), 30 error (mm2s), 0 CRASH:

- **`iter/shim/s2mm/{I8,I16,I32}` -- divergent.** The overlapping iteration means
  the two S2MM iterations *write* to `out[0..inner]` and `out[inner/2..inner/2+inner]`
  -- overlapping writes whose result is order-dependent. The emulator and silicon
  legitimately disagree on overlap-overwrite ordering. This is an **ill-defined
  pattern**, not an emulator fidelity bug: overlapping *writes* lose data.
  (Example banked case: pattern `sizes=[2,1,1,32] strides=[16,2,2,1]`, region 64;
  silicon `out` had only 32 nonzero of 64.)

- **`iter/shim/mm2s/{I8,I16,I32}` -- HW Timeout.** The iterated shim *read* hangs
  the kernel on silicon (it runs fine on the emulator). 0 CRASH (no NPU wedge,
  per-run timeout). Root cause NOT yet established -- likely the iteration's
  word-count or completion-token behavior differs on HW so the memtile linear
  passthrough (which expects exactly `region` words) starves, or the degenerate
  D1/D2 dims interact badly with the real BD engine. Needs HW instrumentation
  (`XDNA_EMU_WATCH` analogue on HW, or dmesg/trace) to pin down.

## Redesign plan for the follow-up

1. **Non-overlapping, non-foldable, gapless iteration.** The tension: a useful
   contiguous tiling is foldable; a non-foldable one either overlaps (writes lose
   data) or gaps (holes -> undefined output). The clean resolution is an
   **interleaved** pattern: inner D0 with stride 2 (every-other element),
   iteration stride 1, 2 iterations -> iter0 touches even indices, iter1 odd,
   together gapless and non-overlapping and non-foldable. Verify it still emits a
   live `iteration_wrap` (doesn't fold) and is deterministic for BOTH read (mm2s)
   and write (s2mm).
2. **Root-cause the mm2s timeout** independently -- even a non-overlapping mm2s
   iteration may still hang if the completion/word-count issue is structural.
   Probe a single hand-authored interleaved `iter/mm2s` case on HW first (like the
   `iter_shim_v3` spike), before wiring the generator.
3. Re-flip `table::supported` (`Iter => engine == Engine::Shim`) once both
   directions match silicon. The `Feature::Iter` enum variant and the deferred
   placeholder in `gen::build_pattern` are already in place, so it's a gating
   flip plus the new pattern, not a refactor.

## Key takeaway

The HW gate did exactly its job: `iter` passed the emulator (compile + run +
non-vacuous) but the differential campaign against silicon caught that the
feature is not real-hardware-faithful. Shipping `chain` alone keeps the merged
work 100% silicon-verified; `iter` returns as a focused follow-up with this
evidence in hand.
