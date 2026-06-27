# Finding: compute MEMMOD (pkt_type 1) trace records no DMA events

**Date:** 2026-06-27 (root cause RESOLVED on HW 2026-06-27)
**Context:** connectivity sub-project (#140), P2 diagnosis.
**Status:** ROOT CAUSE CONFIRMED ON HW. The compute memmod trace unit works; it
records LOCK events fine. The DMA *task* events we watched are structurally
silent (circular-BD objectfifo). See "HW confirmation" below.

## ROOT CAUSE (HW-confirmed 2026-06-27)

A broad-menu HW capture (`build/experiments/two_col_p2_spike/broad_menu_driver.py`,
memmod menu widened to both DMA channels both directions start+finished + 2 lock
events) settled it. Across all four compute tiles, the memmod recorded:

- `LOCK_SEL0_ACQ_GE` and `LOCK_0_REL` — **~16 per run, every tile** (the
  per-iteration buffer handshake).
- **ZERO** DMA events — not `DMA_S2MM_0/1_START_TASK`, not `_FINISHED_TASK`,
  neither channel, neither direction.

So the trace unit arms, the window is correct, the patcher works (H1 and H3 from
the original spike are REFUTED). The compute-tile objectfifo DMA is a **circular
BD ring** (`two_col.config.json`: compute BD0 `use_next_bd:true, next_bd:1`,
`lock_acq_id:0, lock_rel_id:1`) enqueued **once at startup, before the trace
window opens**. `DMA_*_START_TASK`/`FINISHED_TASK` are per-task-ENQUEUE events,
so they fire once outside the window and never again; the per-iteration data
movement is gated by **locks**, which is why lock events fire every iteration and
DMA task events do not. The original menu
(`DMA_MM2S_0_START_TASK, DMA_S2MM_0_START_TASK, EDGE_DETECTION_EVENT_0`) watched
only events that structurally cannot fire in-window for this dataflow.

## CONNECTIVITY IMPLICATION (the grounding gap)

The generator orients the cross-column memtile→compute conversation via
**route reachability over DMA events** — all 30 cross-column ledger entries
couple DMA/PORT_RUNNING endpoints (e.g. `1|1|3|PORT_RUNNING_0 →
2|4|1|DMA_S2MM_0_START_TASK`). The consumer endpoint is always the compute DMA,
which is **silent**. The events that DO fire on the compute side (memmod lock
events; core `INSTR_LOCK_ACQUIRE_REQ`) have **no cross-column ledger entry** —
lock events aren't routable stream ports, and the lock-pair/program-path
orientation is tile-local. So the cross-column conversation cannot ground via
the oriented endpoint, and the observable endpoint isn't oriented. Closing P2 =
giving the cross-column consumer a groundable, observable endpoint (the lock the
consumer DMA uses), which is a real orientation extension, not a menu tweak.

---

## Original spike record (offline isolation, 2026-06-27)

**Status:** OPEN — split to a follow-on (no fix attempted yet). HW-gated.

## Summary

On the two_col kernel, the compute-tile memmod (DMA-module, `pkt_type 1`) trace
unit is **armed and routed** but records **zero events**. This blocks grounding
every cross-column conversation in two_col (the col-2 side of the boundary is
always a compute tile whose dataflow uses its DMA), so the inference engine
honestly classifies those conversations `unobserved`. It is **not** a decode bug
and **not** a routing bug.

## Evidence (offline, from the existing capture)

Source: `build/experiments/two_col_capture/cap/capture_00/run_00/` (no fresh HW).

1. **Packets are emitted and routed.** Raw `trace.bin` header scan finds 7
   `pkt_type 1` packets from all four compute tiles (hw (1,2),(1,3),(2,4),(2,5)),
   alongside 56 core / 13 shim / 46 memtile. The decoder's 8-word stride walk
   parses these headers fine.

2. **The payload contains no events; the decoder is correct.** Each compute
   memmod payload is 28 bytes that decode (forced EVENT_TIME / mode 0, matching
   the tile's declared mode) to `Start + DC + padding` — zero `EventCmd`s:

   ```
   f0 00 00 00 00 32 67 c5   Start (EVENT_TIME anchor)
   dc 00 3c 61               DC    (don't-care, advance 4 bytes)
   fe fe fe ...              idle padding
   ```

   Core/shim/memtile payloads on the same capture decode to 132-449 events
   each; compute memmod decodes to 0.

3. **The trace window is fine.** Traced MLIR `Trace_Control0`: compute core
   (0x340D0) = `0x79780001` (mode 1, EVENT_PC); compute memmod (0x140D0) =
   `0x79780000` (mode 0, EVENT_TIME). Both carry the **same** start event (120)
   and stop event (121). The core captures ~222 events in that window; the
   memmod captures the Start anchor (so the unit armed) and then nothing.

## Conclusion

The memmod trace unit, armed and routed with the same start/stop window as the
working core unit, records none of its selected DMA events
(`Trace_Event0/1` patched with 21=DMA_S2MM_0_START_TASK, 19=DMA_MM2S_0_START_TASK,
11=EDGE_DETECTION_EVENT_0). The remaining ambiguity — one of:

1. patched `Trace_Event` not taking effect on HW,
2. wrong event IDs for an objectfifo compute-tile DMA, or
3. a memmod trigger/window subtlety —

cannot be split from the existing capture; it needs a targeted HW experiment.

## Resume plan (when there is HW appetite)

In order of cheapness:

- **(A)** Fresh two_col capture, then read back the live `Trace_Event0` (0x140E0)
  on a compute memmod and confirm it holds `[21,19,11,0]`. If not → patcher bug
  (offline-fixable once confirmed).
- **(B)** If the register is correct, swap in a memmod event known to fire (e.g.
  a lock event) and re-capture. Records → the DMA event IDs are wrong for this
  dataflow; still nothing → the trigger/window.

Detailed spike log (gitignored): `build/experiments/two_col_p2_spike/P2-FINDINGS.md`.

## Relationship to the connectivity feature

P1 of #140 (the logical connectivity model) is complete and unaffected: it
reports these cross-column conversations as `connectivity_unobserved`, which is
the truthful classification given this gap. Closing this gap is what would let
the engine *ground* a cross-column edge on real hardware.
