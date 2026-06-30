# Cold-start head-start: HW cores warm before the trace baseline; EMU captures cold

**Date:** 2026-06-29  **Issue:** #140 (timer-sync arc, SP-4a)
**Status:** PARTIALLY FALSIFIED -- the head-start is REAL but is NOT the cause of
the SP-4a offset (empirically disproven, see correction below). The actual offset
lever is the **fill-cadence / consumer-pacing residual** (known-fidelity-gaps
row 51).
**Branch:** `sp3-validation-kernel`.
**Working notes + raw evidence:** `build/experiments/sp3-spike-trace/lean/`
(decoded HW + EMU traces, 5 stable HW runs, the cold-start sim probe).

> **CORRECTION (2026-06-29, same session -- the causal claim below was tested and
> FALSIFIED).** This doc concluded the cross-domain offset (-52 EMU vs +2 HW) is
> caused by the missing head-start. That is WRONG. Added a configurable warm-up
> floor (`XDNA_EMU_CORE_HEADSTART`, FFI `backend.rs`) and swept N = 0 .. 2000cy:
> the trace baseline genuinely shifts later (total cycles 117556 -> 119449, shim
> base soc 6779 -> 8672, so the cores DO warm longer) but the prod->consA offset
> stays **pinned at -52** for every N. So the head-start does not move the offset.
> Everything in "What the gap is NOT" + the head-start measurement table below is
> still TRUE (HW cores really do run ~795-1068cy before the baseline, 6 runs) --
> but that is a real, ORTHOGONAL behavior, not the offset's cause.
>
> **The actual cause (re-confirmed):** the pipeline FILL-STATE at the baseline.
> The EMU producer reaches the baseline with FREE buffers (its MM2S drained them
> downstream; the pipeline blocks downstream, not at the producer), so after the
> reset+drain cascade it re-fills 2 buffers before stalling -> +52 after consA.
> HW's producer is coupled to one buffer / blocked at the baseline, so it stalls
> ~immediately with consA -> +2. This IS known-fidelity-gaps **row 51** (send-port
> `[16,16,16,16]` EMU vs `[8,8,14]` HW = same producer-fills-both vs
> coupled-to-one fill-cadence). The lever is the consumer-pacing / drain-rate /
> buffer-coupling axis (device-model audit), NOT cold-start sequencing.

This finding **corrects** the prior SP-4a root-cause framing (the
`build/experiments/sp3-spike-trace/SP4A-LEAN-FINDING.md` "memtile relay cadence"
conclusion). It also **characterizes** the open residual in known-fidelity-gaps
**row 51** (the send-cadence "cold-start transient... needs an HW capture of the
first-buffer lock-handoff timing") -- they are the same root cause, seen from
two observables.

## The substrate

The lean SP-4a kernel `of_q0_lean.py` (Q=0, pure lock/DMA handshake, no compute):
ProdCore(0,2) -> MemTile(0,1) -> ConsA(0,3) objectfifo chain, REPS=32, OBJ=64.
The SP-4a oracle is the within-column cross-domain pair: prod -> consA first
`LOCK_STALL` offset, HW **+2** (range 0 across 20 runs), EMU **-52**.

## What the gap is NOT (each ruled out with evidence)

1. **Not a per-tile clock stagger.** A cold-start sim probe (env-gated
   `XDNA_EMU_COLDPROBE` in `coordinator.rs` step loop) shows both cores execute
   in lockstep from sim-cycle 0 -- identical PC every cycle through the prologue.
   The clock ungate is per-column (`dispatch.rs` `assign_partition_columns`), so
   all compute cores in a column become active the same cycle. No stagger.

2. **Not an instruction-timing / VLIW-bundling gap.** Producer loop period: HW
   ~69 trace-ticks/iter, EMU ~67. ConsA: HW 67, EMU 69. The loops run at HW speed.

3. **Not a producer free-buffer lock bug -- the EMU is toolchain-faithful.**
   The lowered MLIR sets `src_prod_lock` (producer free-slot lock) **init = 2**,
   set in the CDO before core-enable and **never reset** by any DMA-config or
   channel-start path (aie-rt: DMA setup writes only BD/channel-control
   registers, never lock *value* registers; the only lock-value writer is
   `XAie_LockSetValue` in `initLocks`, which runs before core-enable). The
   producer core lowers to `acquire_ge(prod_lock, 1)` -- immediately satisfiable
   against 2. So per the toolchain the producer **should** acquire 2 buffers at
   cold-start without stalling, which is exactly what the EMU does. (Derivation:
   `mlir-aie/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp`
   :422-451 lock init, :1628-1641 acquire lock/value; `aie-rt/.../dma/xaie_dma.c`
   channel start touches only the Enable bit; `mlir-aie/lib/Targets/AIERT.cpp`
   :533-557 `initLocks` ordering.)

## What the gap IS: HW cores warm before the trace baseline; EMU captures cold

PERF_CNT_2 fires every 1024 core-clocks, so its first occurrence back-dates each
core's counter start relative to the shim trace base (the shim
`DMA_S2MM_0_START_TASK`). Across **5 stable HW runs + the main run** (consistent
to +/-25):

| | producer counter-start (rel. trace base) | consA counter-start | first-LOCK_STALL offset (consA - prod) |
|---|---|---|---|
| **HW** | **-774 ... -822** (BEFORE the base) | **-561 ... -602** | **+2 every run** |
| **EMU** | **+581** (AFTER the base) | +573 | **-52** |

So on HW the cores have already executed ~795cy (producer) / ~580cy (consA)
**before** the trace/timer baseline is established; by the time the trace starts
they are warm and in steady-state phase-lock (+2). On the EMU the cores start
~580cy **after** the baseline -- the trace captures the cold-start transient
(producer fills its 2 free buffers and stalls on its 3rd acquire ~52cy after
consA's 1st-acquire stall) -> -52.

Two HW behaviors the EMU lacks, both feeding the trace-baseline phase:
- the **~795cy CDO-config -> runtime-sequence head-start** (cores enabled at CDO
  time run freely; the runtime sequence that starts the shim DMA / trace baseline
  is dispatched ~795cy later);
- a **~215cy producer-before-consumer core-start stagger** (HW prod -795 vs consA
  -580); the EMU starts both cores in lockstep.

## The +2 is a phase-locked sample, not a steady-state invariant

Sweeping a hypothetical trace-start time T and computing `(first consA stall >= T)
- (first prod stall >= T)` shows the offset is **noisy (+/-~30)**, not a stable
constant, in BOTH worlds. HW reproduces +2 (range 0 across 20 runs) because its
trace-start event is deterministically phase-locked to the dataflow **at a warm
point**; the EMU samples -52 because its trace-start lands **cold** and at a
different dataflow phase. So "byte-match the raw cross-domain offset" is
reproducing a phase-locked sample on a +/-30 jitter floor, not validating a
within-domain timing invariant. SP-4a's premise (within-domain exactness =>
cross-domain +2) does not hold the way it was framed.

## Unification with known-fidelity-gaps row 51 (send-port cadence)

Row 51's open residual is the same mechanism from the send-port observable:
EMU `PORT_RUNNING_4 = [16,16,16,16]` (producer empties the whole transfer before
any consumer stall) vs HW `[8,8,14,...]` (consumer-paced). That is the **cold**
opening (producer fills both ping-pong buffers, un-paced) vs the **warm**
steady-state (consumer stalling regularly, pacing the producer). The head-start
explains both observables. Testable prediction: warming the EMU cores before the
trace baseline should shift the send port toward `[8,8,14,...]` AND the
cross-domain offset toward +2.

## Resume: build the head-start / trace-baseline model

The fix is to model the CDO-config -> runtime-sequence head-start (~795cy, a
HW-measured constant; SP-5 flavor) so the EMU's cores warm before the trace
baseline is established, as on HW -- and likely the ~215cy producer-first
core-start stagger. This is cold-start sequencing that affects every traced
kernel, so it is HW-calibrated territory: gate any change on the offset moving
-52 -> +2 (lean oracle), the send port moving toward HW (row 51 oracle),
`cargo test --lib`, and a bridge-corpus spot-check for no regression.

Evidence on disk: `build/experiments/sp3-spike-trace/lean/{hw,emu,hw_stab/run_*}/
events.json`; the cold-start sim probe output and the analysis scripts used for
the tables above.
