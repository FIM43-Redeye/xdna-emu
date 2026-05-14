# Task 321 investigation: EMU trace truncates before kernel halt, not "EMU halts before _fini"

## Summary

Task #321 was scoped as "investigate why EMU halts before _fini's
branches execute," with the assumption that EMU's interpreter was
halting the kernel core somewhere between PC 880 (entry to `_fini`) and
PC 1184 (the `JNZD r23, r23, p6` that closes the `__cxa_finalize`
atexit loop). Investigation shows the kernel doesn't halt early — the
**trace unit stops emitting** before the kernel reaches PC 1184. The
remaining events are simply not captured, even though the kernel keeps
executing for a few more cycles. Same root cause as #319: cycle drift
in when broadcast events fire relative to kernel progression.

## What we observed

For `add_one_using_dma.chess` mode-2 baseline:

```
HW   tail: ...New_PC=144, New_PC=880, E_atom, New_PC=1184, N_atom, Sync
EMU  tail: ...New_PC=144, New_PC=880, Sync
```

EMU is missing three events: the JNZD-taken atom at PC 1184, the
indirect-call destination of the second JL p1 (which lands at 1184 in
this kernel because the atexit array's terminator entry points to a
sentinel that returns immediately), and the JNZD-not-taken atom that
closes the loop.

EMU log evidence:
- `BROADCAST_15 (hw_id=122) ... at cycle 51` — trace start
- `BROADCAST_14 (hw_id=121) ... at cycle 790` — trace stop
- `XDNA_EMU_STATUS: halt_reason=completed cycles=791`

The engine halts at cycle 791. BROADCAST_14 fires at cycle 790, exactly
one cycle before. Trace useful bytes: EMU=92, HW=296 — both stop at
their respective stop_event, but EMU's stop arrives much earlier
relative to the kernel's progress.

## Why this happens (kernel structure)

The `__cxa_finalize` atexit loop at PC 1056-1218:

```
1056-1136 prologue (sets r23 = 1, p7 = atexit array, p6 = #1152)
1152: LDA p1, [p7], #4   (load handler[i])
1156: LDA p0, [p7], #-12 (load handler-arg, rewind)
1170: JL p1              (call handler -- INDIRECT, emits New_PC)
1184: .return_address; JNZD r23, r23, p6   (loop-back, taken if r23 != 0)
1198-1216: epilogue and RET
```

Atexit array has two entries: handler_1 = `_fini` (PC 880) and
handler_2 = a sentinel that points back into the loop (in this kernel,
it lands at PC 1184 itself; calling JL p1 -> 1184 falls through the
JNZD with r23=0 and into the epilogue). So:

- iter 1: JL p1 -> 880 (`_fini`), runs, RET back to 1184. JNZD with
  r23=1 -> taken, decrement r23 to 0, branch to 1152. Trace events:
  `New_PC=880` (JL p1 dest), `E_atom` (JNZD taken).
- iter 2: JL p1 -> 1184 (sentinel), JNZD with r23=0 -> not taken, fall
  through. Trace events: `New_PC=1184` (JL p1 dest), `N_atom` (JNZD
  not taken).

EMU captures only iter 1's `New_PC=880`. It loses everything from
iter 1's E_atom onwards.

## Why the trace stops early (the actual cause)

BROADCAST_14 in this test is fired by a control packet on the shim tile
that writes Event_Generate=126 — and that control packet is
processed during the runtime sequence epilog (after `dma_wait`
completes). In HW, the dma_wait completion is a real-time signal with
PCIe and command-queue latency, so the broadcast fires plenty of cycles
after the kernel finishes its cleanup. In EMU, dma_wait completes
synchronously when the output DMA's last word is written, and the host
fires BROADCAST_14 immediately on the next simulation step — well
before the kernel has finished its post-loop cleanup (atexit, _fini,
DONE).

This is the **same class of issue** as #319: simulation cycle accounting
for the host-side path is too compressed relative to kernel cycles.
Whatever the exact cycle of dma_wait-vs-broadcast in HW, EMU collapses
that gap, causing the trace to truncate.

## Why "EMU halts before _fini" is the wrong framing

The bridge test PASSES (output data is correct), which means the kernel
runs to completion — through `_fini`, through both atexit iterations,
through `__cxa_finalize`'s epilog, to PC 188 DONE. The trace just isn't
listening anymore by the time those branches happen.

If EMU's kernel were halting early in `_fini`, the output DMA wouldn't
have written the last batch of post-processed data (which depends on
the kernel reaching the release calls in the main loop). The fact that
data is correct AND the engine halts at cycle 791 (about 100 cycles
past the 1st JL p1 trace event) implies the kernel finishes its work
across those last 100 cycles — the trace just doesn't see them.

## Concrete next steps

1. Verify the hypothesis: instrument the kernel core to log its PC at
   every step, run the bridge test, check what PCs are visited between
   cycle 790 (BROADCAST_14) and cycle 791 (engine halt). Should see
   the kernel still in `__cxa_finalize` body at cycle 790 and reach
   PC 188 (DONE) by 791.
2. If confirmed, fix-options:
   - Delay BROADCAST_14 firing in EMU by N cycles to match HW's
     dma_wait → control-packet → Event_Generate latency (requires
     finding the right N empirically and matching the HW ordering).
   - Have EMU's kernel core continue running for some cycles past
     halt before declaring engine-halted, so trace events finish
     emitting (less clean — kernel state is technically halted).
   - Have the trace unit's stop_event handling drain *additional*
     pending kernel-side events for some grace period before flushing.
     This is the cleanest option since it isolates the fix to the
     trace unit.
3. Consider whether this and #319 should be addressed by a single
   broader fix that models host-vs-kernel cycle drift correctly,
   rather than two point patches.

## Status

Logged as residual for #321. Re-scoping suggested: not "fix EMU halt"
but "fix EMU broadcast timing or trace-unit stop_event handling so
late-kernel events are captured." Same root cause family as #319 —
both should likely be addressed together.

#305 (mode-2 sweep generalization) remains parked until either #319
and #321 are fixed, or the comparison's tolerance for trace-bracket
skew is increased so it can clear cleanly without these.
