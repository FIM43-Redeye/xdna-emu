---
name: '#355a Phase B: per-segment trace anchors decompose the dma_wait gap'
description: Instrumented add_one_using_dma's runtime sequence with four INSTR_EVENT_0 anchors (T0..T3) bracketing the two dma_memcpy_nd ops and the dma_wait. The dma_wait segment dominates (HW=9631 vs EMU=428 cyc, 22.5x). Decomposing further, ~1650 cyc of HW's window is real DMA + kernel work; the remaining ~8000 cyc is pure firmware-mailbox token-path latency. EMU's dma_wait returns too early (before the kernel finishes all 16 acquires), which is a separate correctness issue but explains why EMU's wait segment is 0-padded.
type: project
---

# `#355a` Phase B: runtime-sequence instrumentation

## TL;DR

Created `_diag_phase_b_add_one_instrumented` -- a clone of
`add_one_using_dma` whose runtime sequence fires `INSTR_EVENT_0`
(event ID 33) on compute tile col 0 row 2 at four landmarks:

| Anchor | Where                                |
|--------|--------------------------------------|
| T0     | before first `dma_memcpy_nd`         |
| T1     | after  first `dma_memcpy_nd`         |
| T2     | after  second `dma_memcpy_nd`        |
| T3     | after  `dma_wait`                    |

The compute tile's core trace already has `INSTR_EVENT_0` in slot 1
of its grounding events (PERF_CNT_2, INSTR_EVENT_0, INSTR_EVENT_1),
so writing event ID 33 to the Event_Generate register at offset
0x34008 fires the existing slot.  No trace-config changes needed --
all four firings appear as four trace events on the compute core,
distinguishable by chronological order.

Bridge test passes on both compilers; trace is clean.  Per-segment
deltas:

| Segment | HW (cyc) | EMU (cyc) | Ratio |
|---------|---------:|----------:|------:|
| T0->T1 (1st dma_memcpy_nd) | 1094 | 525 | 2.1x |
| T1->T2 (2nd dma_memcpy_nd) | 1056 | 750 | 1.4x |
| **T2->T3 (dma_wait)**      | **9631** | **428** | **22.5x** |
| Total (T3-T0)              | 11781 | 1703 | 6.9x |

The dma_wait segment dominates the apparent gap.  Going further: of
HW's 9631-cyc T2->T3 window, the actual DMA + kernel work completes
in the **first 1650 cyc**; the remaining **~8000 cyc is pure
firmware/mailbox/token-path latency** with no trace-visible events.

## Decomposition of the dma_wait segment

Cross-referencing T2->T3 events on HW against EMU:

```
HW (T2=333015, gap=9631):
  T2+   0  INSTR_EVENT_0    (T2 anchor)
  T2+   1  LOCK_STALL       (kernel hits 1st acquire, blocks)
  T2+  55  DMA_MM2S_0_FINISHED  (shim input BD done)
  T2+1110  INSTR_LOCK_ACQUIRE_REQ  (1st acquire success after pipeline fill)
  T2+1110-1650  15 more INSTR_LOCK_ACQUIRE_REQ (kernel iter burst @ ~33 cyc/acq)
  T2+1502  DMA_S2MM_0_FINISHED  (shim output BD done)
  T2+1650  (last visible event)
  --- 7981-cyc DEAD WINDOW (no trace events) ---
  T2+9631  INSTR_EVENT_0  (T3 anchor)

EMU (T2=4879, gap=428):
  T2+   0  INSTR_EVENT_0
  T2+ 28-181  6 INSTR_LOCK_ACQUIRE_REQ (kernel started, but only 6 of 16 acquires)
  T2+ 356  DMA_MM2S_0_FINISHED
  T2+ 428  INSTR_EVENT_0  (T3 anchor)
```

Two distinct EMU/HW gaps surface here:

### 1. HW firmware token-path latency (~8000 cyc, EMU=0)

Between "shim S2MM 0 BD finished" and "T3 fires" on HW there is
a 7981-cyc window with no trace activity.  The `dma_wait` blocks on
a token issued by the shim S2MM 0 BD (`issue_token = true` on the
second `dma_memcpy_nd`).  Token-issue and dma_wait-completion go
through the firmware mailbox -- on HW that round-trip is ~8 us at
1 GHz = ~8000 cyc.

EMU has no mailbox model, so this latency is 0.

This is a real EMU/HW gap but it's *outside* the DMA model -- it's
the firmware command-and-response loop, not chain propagation.
Calibrating it would require either modeling the mailbox or adding
a fixed dispatcher-completion latency to dma_wait.

### 2. EMU dma_wait returns too early (correctness issue)

EMU shows 6 INSTR_LOCK_ACQUIRE_REQ events between T2 and T3, but
the kernel does 16 acquires total (4 outer iters x 4 acquires/iter).
The remaining 10 must be firing AFTER T3, meaning the runtime
sequence's dma_wait returned before the kernel's work completed.

The test still passes (output buffer correctness), so kernel work
*does* complete -- just after the runtime sequence has already
moved past dma_wait.  This is structurally fine for functional
correctness but produces misleading cycle measurements: EMU's T3
anchor doesn't represent "all work done," whereas HW's T3 does.

This is an EMU correctness issue independent of #355a calibration
and worth chasing separately.  Likely lives in the runtime
sequence executor's dma_wait handling -- it should block until the
matching shim S2MM channel issues its token, not return on
BD-queued or some early signal.

### 3. Real DMA pipeline propagation gap (~3000 cyc, structural)

The Phase A finding's 11.4x pipeline fill ratio is the *structural*
gap: HW takes 3458 cyc from compute LOCK_STALL to first
INSTR_LOCK_ACQUIRE_REQ; EMU takes 427 cyc.  This is genuinely the
DMA chain.  Even with the firmware overhead and dma_wait bug
backed out, the chain is 8x too fast on EMU.

## Recommended next direction

Three independent threads emerged:

1. **Track down EMU's dma_wait early-return bug.**  Cleanest fix
   first because it perturbs every trace measurement that goes
   through the runtime sequence.  Probably lives in
   `src/npu/executor.rs`'s handling of `aiex.npu.dma_wait`.

2. **Phase C: instrument the kernel itself.**  Add a USER_EVENT
   trigger as the kernel's first instruction (modify the .cc
   source) to bracket "kernel boot" vs "kernel hits first acquire."
   Also add anchors around the kernel's outer-loop iterations so
   we can see per-iter timing on HW vs EMU directly.

3. **Attribute the 3000-cyc structural pipeline gap.**  Even with
   T0..T3 anchors, Phase B shows the gap lives between LOCK_STALL
   and the first INSTR_LOCK_ACQUIRE_REQ.  Finer attribution
   requires instrumenting either the memtile DMA or the data path
   between memtile and compute -- either via memtile-side user
   events (memtile has 2 user events: IDs 159-160) or by reading
   memtile DMA channel state at known times via the existing core
   user-event triggers.  Memtile reads via read_aie_reg are
   firmware-broken on Phoenix (May 7 finding), so this is the
   only feasible path.

## Mechanical approach

The instrumentation pattern is:

```mlir
aiex.npu.write32 {address = 0x34008 : ui32, column = 0 : i32, row = 2 : i32, value = 33 : ui32}
```

- `address = 0x34008` -- Event_Generate register on the core module
  (per AM025 register DB; same offset for shim.  memory module is
  0x14008, memtile is 0x94008).
- `column, row` -- target tile in MLIR-origin coords
  (the runtime adds start_col offset).
- `value` -- event ID to fire (33 = INSTR_EVENT_0,
  34 = INSTR_EVENT_1, 124-127 = USER_EVENT_0..3).

Up to 8 distinct events fit in a single tile's trace slots, but the
default `--core-grounding` reserves slots 0-2 for PERF_CNT_2,
INSTR_EVENT_0, INSTR_EVENT_1.  USER_EVENT_0..3 are not in default
grounding -- adding them requires `--core-grounding=...,USER_EVENT_0,...`.
For Phase B we just used INSTR_EVENT_0 multiple times and
distinguished by chronological order.

## See also

- `2026-05-10-phase-a-trace-cycles-measurement.md` -- prior finding;
  pipeline fill 11.4x ratio still holds (re-measured 8.1x here on
  the slightly-modified instrumented variant; same order of
  magnitude).
- `mlir-aie/test/npu-xrt/_diag_phase_b_add_one_instrumented/` --
  the test variant.  Sibling-style under mlir-aie's tree
  (matches the pattern from the 2026-05-09 bug-6 diagnostics).
- task #355a -- Phase B complete; three follow-up threads listed
  above.
