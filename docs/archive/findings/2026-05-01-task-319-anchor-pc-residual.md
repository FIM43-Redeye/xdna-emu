# Task 319 investigation: anchor_pc residual is deeper than "lock-acquire poll loop"

## Summary

Task #319 was scoped as "model lock-acquire poll loop so PC advances
through NOPs," with the expectation that pinning EMU's PC at the
LockAcquire instruction (PC=816) was the sole cause of the 16-byte
anchor_pc residual against HW (PC=832). Investigation shows the residual
has **two distinct causes**, only one of which is the lock-acquire poll
loop. A clean fix needs more than a localized PC-advance hack.

## What we observed

For `add_one_using_dma.chess` mode-2 baseline (after #320's fix landed):

```
HW   tile (0,2): Start anchor_pc=832 at cycle 204, first New_PC=368
EMU  tile (0,2): Start anchor_pc=816 at cycle 144, first New_PC=336
```

Two separate divergences:

1. **PC offset 16 bytes**: HW shows PC=832 (last delay slot of the
   acquire-stub RET), EMU shows PC=816 (the ACQ instruction itself).
2. **First-iteration extra New_PC=336**: HW's first New_PC after Start
   is 368 (i.e. it's already in the *second* acquire's epilog by trace
   start). EMU's first New_PC is 336 (i.e. it has only just begun the
   *first* acquire by trace start).

Both can be read off the events list: the first New_PC after Start tells
you which acquire the trace started in (LR=336 ⇒ inside the 1st acquire,
LR=368 ⇒ inside the 2nd acquire).

The cycle delta at trace start is **60 cycles** (HW=204, EMU=144). HW
has progressed an entire acquire iteration further into the kernel than
EMU has, by the time the broadcast fires.

## Two causes, not one

### Cause A: pipeline-PC vs execute-PC

When ACQ stalls in HW, the architectural execute-PC is 816, but the
pipeline has already fetched and decoded the RET (820), four NOPX delay
slots (824–830), and the start of the NOPA cluster delay slot (832). The
core's PC register reads as the fetch/decode PC, not the execute PC, so
trace events that sample core PC see 832.

For the acquire stub specifically:
- 816: ACQ r0, r1 (4 bytes) — execute PC, stalled
- 820: RET lr (4 bytes)
- 824, 826, 828, 830: 4× NOPX (2 bytes each)
- 832: NOPA cluster (16 bytes; last delay slot of RET)

The fetch PC settles at the start of the last delay slot (832) because
that's the point where the pipeline can no longer advance until the
branch (RET) commits, which itself can't commit until ACQ resolves.

EMU pins PC at the LockAcquire instruction (816) on WaitLock and never
advances it through the function epilog, so the PC reported to the trace
unit on broadcast arrival is the execute PC, not the fetch PC.

### Cause B: kernel/broadcast cycle drift

Even if EMU modeled fetch-PC correctly and reported 832, the *first*
New_PC after Start would still be 336 (the LR for the 1st-acquire JL),
not 368 (the LR for the 2nd-acquire JL), because EMU is genuinely 60
cycles behind HW in its progression through the kernel by broadcast
time. HW has already completed the entire 1st acquire (lock acquired,
RET committed, JL #816 to 2nd acquire issued, ACQ at 816 stalling
again); EMU has only just entered the 1st acquire and is still on the
1st ACQ.

This is **not** a pipeline-PC effect — even with cause A fully
modeled, EMU's kernel state at broadcast time is one acquire behind
HW's. Something in EMU either:

- runs the lock-acquire poll loop in fewer simulated cycles than HW
  (e.g. EMU charges 1 cycle per stall iteration where HW charges some
  larger number due to pipeline depth or arbitration latency), or
- delivers BROADCAST_15 to the trace unit faster than HW does (e.g. the
  control-packet → Event_Generate → broadcast → trace-unit-observe path
  is nearly free in EMU but has natural delay in HW), or
- starts the kernel later relative to broadcast firing (kernel takes
  fewer wall-cycles to reach first ACQ in EMU).

Without a smoking-gun diagnostic (e.g. instrumenting the cycle on which
each significant event happens in both EMU and HW), it isn't clear which
of these contributes.

## Why the simple fix doesn't suffice

A targeted patch — "on WaitLock, walk forward in the disassembly to
find the next branch and pin trace-PC at the start of the last delay
slot" — fixes cause A and would land anchor_pc=832. But cause B remains:
the *first New_PC* after Start would still be 336, not 368, leaving an
iter-1 extra-New_PC residual that's the more visible mode-2 delta.

Conversely, fixing only cause B (advancing EMU's broadcast firing or
debiting kernel cycles to match HW's progress) would give matching
iter-1 New_PC=368 but anchor_pc would still be 816, leaving cosmetic
divergence.

To fully close the gap we need both. Fixing A is doable and bounded
(needs disassembly look-ahead during WaitLock). Fixing B is harder and
needs investigation into where the 60-cycle drift comes from.

## Concrete next steps

1. Instrument the absolute cycle at which key kernel-side events fire
   in both EMU and HW: kernel core enabled, first ACQ issued, first
   ACQ granted, first RET committed, BROADCAST_15 arrived at trace
   unit. The diff between EMU and HW for each of these isolates which
   stage owns the 60-cycle drift.
2. Once cause B's source is found, decide whether to fix it
   point-source (e.g. add a startup delay, charge more cycles per
   stall iteration) or accept it.
3. Implement cause A's pipeline-PC modeling if cause B is fixed first
   — there's no reason to ship cause A alone since it leaves the
   visible iter-1 residual.

## Status

Logged as residual for #319. Not blocking #305 (mode-2 sweep)
generalization — the comparison can tolerate small anchor_pc skews
since the trace-event sequence after Start is what matters for
correctness validation. #320 (JNZD New_PC) is the major correctness
item and is fixed.
