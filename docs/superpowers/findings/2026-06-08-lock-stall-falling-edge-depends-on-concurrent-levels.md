# LOCK_STALL falling edge depends on concurrent levels (fork B resolution)

**Date:** 2026-06-08
**Status:** Root cause established. Resolves the "second long stall not surfacing"
fork (B) raised after the held-level mechanism (`7767dd8`) landed.
**Predecessors:**
[2026-06-07 over-emission finding](2026-06-07-lock-stall-overemission-interp-vs-hw.md);
[level-emission design spec](../specs/2026-06-07-lock-stall-level-emission-design.md);
known-fidelity-gaps "Trace stall/starvation micro-timing" row.

## Question

After routing LOCK_STALL as a held level, the **count** matches HW (~19) but the
**decomposition** does not: HW's two long stalls (`dur=6354`, `dur=1930`) never
surface in our decoded trace; we emit only short `dur=1` spans. Is the missing
long span a real *mechanism drop* or a benign *DMA-timing* artifact that is stable
Phoenix->Strix and can be documented?

Answer: **both, and they are separable.** There is a genuine mechanism gap
underneath the DMA-timing surface, and it is already in scope as Milestones 2-4
of the held-level plan.

## Evidence (vec_mul_trace_distribute_lateral, peano, interpreter backend)

Decoded LOCK_STALL spans:

| | spans | shape |
|--|--|--|
| HW | 18 | `B=1 E=6355 (6354)`, `B=6363 (1)`, `B=6372 E=8302 (1930)`, then 4 iteration clusters of short 1-cycle pulses (4,4,4,3) |
| EMU | 10-19 | all `dur=1`; iteration clusters match HW well; **no long span**, and a gap where the startup acquire-stall belongs |

Instrumenting `set_event_level` shows the held mechanism fires exactly **one**
genuine long hold in the traced window:

```
assert  cycle=10564  held 0x00->0x80  edge=true
deassert cycle=11507  held 0x80->0x00  edge=true     (943-cycle hold)
```

(Plus an orphan deassert@10548 whose assert@~34 happened before the trace unit
was Running, and an unclosed assert@60993 at end-of-segment.)

Instrumenting the emitted frames shows the **assert emits a frame**
(`pending_cycle=10564 active=0x80`) but the **deassert emits none**: at deassert,
`held_mask -> 0` and no pulse is pending, so `active == 0` and
`commit_pending_frame` early-returns. **The falling edge is silently dropped.**
The decoder can only close a span when a later frame *drops* the bit, so the held
span never closes at its true cycle.

## Why HW does not have this problem

HW closes its long stalls "alone" -- the lock-acquire that ends the stall fires
*later*, not coincidentally:

```
ts=6355 E LOCK_STALL                  <- closes alone
ts=6360 B INSTR_LOCK_ACQUIRE_REQ      <- 5 cycles later, different slot
```

This refutes the design comment's assumption ("the falling edge is carried by the
lock acquire that fires in the same cycle as the deassert"). HW can close the level
alone because **other level events are concurrently asserted**, so the falling-edge
frame still carries a non-empty mask:

```
ts=1    B DMA_S2MM_0_STREAM_STARVATION + DMA_S2MM_1_STREAM_STARVATION + LOCK_STALL
ts=6348 B DMA_S2MM_1_STREAM_STARVATION   (still asserted at 6355)
ts=8295 B PORT_RUNNING_0                  (asserted across the 8302 close)
```

When LOCK_STALL falls at 6355, the frame's new snapshot is
`{DMA_S2MM_1_STREAM_STARVATION, ...}` -- a valid Single/Multiple frame. The
mode-0 encoding has **no empty-frame representation** (Single needs a slot,
Multiple needs >=2 bits), so HW cannot emit an empty frame either. Our
`active == 0` early-return is therefore **correct HW behavior**; it only bites us
because LOCK_STALL is currently our *only* modeled core level, so its deassert
always drives the snapshot to zero.

## Root cause (compound, separable)

1. **Mechanism -- falling-edge drop.** LOCK_STALL being the only modeled core
   level means every deassert drives `active -> 0` and the closing frame is
   suppressed. The held span never closes at the right cycle. **This is structurally
   fixed by Milestones 2-4** (DMA starvation, port-running, memory/stream/cascade
   stall levels): once HW's always-present concurrent levels are modeled, the
   LOCK_STALL falling edge rides on a non-empty frame exactly as on HW. No separate
   "force-close empty frame" hack is needed (and would be anti-HW: HW has no empty
   frame).

2. **Timing -- DMA fill latency.** Even with the mechanism fixed, our acquire
   stall is ~943 cycles vs HW's 1930, and time-shifted, because our DMA fills input
   buffers faster than HW's DDR-backed DMA. Both of HW's long stalls are *startup*
   acquire-stalls (core waiting for the first DMA fill). This is the documented,
   Phoenix->Strix-stable DDR-fill axis (known-fidelity-gaps "Trace stall/starvation
   micro-timing", and 2026-06-07 finding Caution #2).

## Consequences for the plan

- **M1 (LOCK_STALL level) is not a standalone island.** Its falling-edge fidelity
  is *delivered by* M2-M4. Do not try to "close M1 and document the residual" as if
  it were independent -- the residual *is* M2-M4.
- The arb-pulse hack (a 1-cycle LOCK_STALL pulse on immediate acquire) stays
  **reverted**: HW's short non-release LOCK_STALL blips are brief *acquire* stalls
  (held level high ~1 cycle), which the held mechanism would produce naturally if
  our lock model imposed the 1-cycle arbitration stall -- a *timing-model* change,
  not a trace change. Faking them as same-slot pulses corrupts the held level's
  falling edge (the pulse re-asserts the bit in the deassert frame, cancelling the
  edge). The committed *release*-pulse path (`control.rs:367`) is separate and
  HW-validated (2026-05-11) -- keep it.
- Our model does **not** reproduce HW's brief inter-stall acquire blips
  (e.g. HW `B=6363 dur=1`), because our acquires either succeed instantly (no
  level) or stall long. Reproducing them needs a 1-cycle lock-arbitration stall in
  the timing model -- a separate micro-timing gap, not this trace work.

## Validation path (proposed)

Implement M4's DMA starvation level (the concurrent level HW asserts across the
LOCK_STALL window) and confirm the LOCK_STALL falling edge then closes at its true
cycle in the decoded trace -- i.e. the held acquire-stall renders as one long span
bounded by the starvation-level frame, matching HW's close-alone behavior. That is
the decisive test that the mechanism gap is closed; the residual duration/placement
difference is then purely the documented DMA-fill-timing axis.
