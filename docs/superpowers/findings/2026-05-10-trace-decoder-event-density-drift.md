# Trace decoder `ts` is NOT SoC cycle — accumulates +1 cyc drift per event

**Date**: 2026-05-10
**Status**: VERIFIED (reproduced across pre/post #355a closure runs)
**Affects**: every cross-tile timestamp comparison in #355a Phase A/B/C,
the closure commit (`3357b7c`), and any future trace measurement that
treats decoded `ts` as SoC cycle.

## Summary

The `ts` field emitted by `tools/parse-trace.py` (and mlir-aie's reference
`parse_trace` decoder) is not the SoC cycle of the event. It is

```
ts = baseline + sum_over_events_so_far(1 + encoded_delta)
   = SoC_cycle_of_event + (1 + events_before_in_this_tile_stream)
```

Each event in a tile's byte stream contributes **+1 cycle of drift**
in the decoded `ts` of every subsequent event. The encoder writes raw
`delta = pending_cycle - last_event_cycle`; the decoder accumulates
`timer += 1 + delta` per `EventCmd`. The asymmetry is intentional and
mirrors mlir-aie's convention (`python/utils/trace/parse.py` line 260,
`timer = timer + 1` then `timer = timer + cycles`).

For tiles with sparse events (compute core: a couple stalls per kernel
iteration) the drift is tiny. For tiles with dense events (shim with
`STREAM_STARVATION` firing every cycle while a channel waits for data)
the drift accumulates to **hundreds of cycles** before any other event.

## The discovery path

Phase C's closure measurement (commit `3357b7c`) reported a "-314 cyc
residual" using `shim_dispatch -> kernel_acquired` (inter-tile, shim
to compute). When I re-ran the calibration test post-closure with the
same env config, two anchors that should have been identical instead
differed:

| Anchor (pre-closure code, fresh re-run) | Pre `ts` | Post `ts` | Δ |
|----|---:|---:|---:|
| compute T0 (`INSTR_EVENT_0`) | 3496 | 3496 | 0 |
| compute LOCK_STALL | 4135 | 4135 | 0 |
| shim DMA_MM2S_0_START_TASK (`shim_dispatch`) | **4272** | **3916** | **−356** |
| shim DMA_MM2S_0_FINISHED_TASK (`shim_done`) | 5211 | 7716 | +2505 |
| memtile S2MM_SEL0 FINISHED_BD | 4434 | 6945 | +2511 |
| compute INSTR_LOCK_ACQUIRE_REQ | 4454 | 6978 | +2524 |

Compute-side anchors are stable — same SoC cycle pre/post. Downstream
anchors shift by ~+2500 (the cold-start propagating). Shim's
`shim_dispatch` shifts the *wrong direction* by 356 cyc — earlier
post-closure than pre — even though the closure code only adds latency
inside `MemoryLatency`, after `START_TASK` fires.

Bridge log confirms `Idle -> BdSetup cycle=3915` for shim ch2 in
**both** pre- and post-closure runs. The FSM transition is at the same
SoC cycle. The decoded `ts` differs.

The Start-frame `timer_value` is identical too: all four tiles (shim,
memtile, compute core, compute mem) write `timer=3283` to the Start
frame in both runs. Trace timer baseline is not drifting.

So the difference must be in the encoded byte stream between Start and
the `START_TASK` event. Dumping the first 64 bytes of the shim trace
in each run:

```
PRE  (pre-closure):  f0 00 00 00 00 00 0c d3 9c c0 71 71 71 71 71 71 ...
POST (post-closure): f0 00 00 00 00 00 0c d3 8a 78 82 e2 bc 05 22 71 ...
```

`f0 00 00 00 00 00 0c d3` = Start with timer=3283 (identical).
- Pre then has `9c c0` (Single1, slot=7 `STREAM_STARVATION`, delta=192)
  followed by a long run of `0x71` (Single0, slot=7, delta=1) — a
  STARVATION event firing roughly every cycle from SoC=3475 onward,
  filling the byte stream **before** the MM2S_0 START_TASK arrives
  later in the stream.
- Post starts with `8a 78` (Single1, slot=2 `MM2S_0_START_TASK`,
  delta=632) — the first event in the encoded shim stream is the
  START_TASK itself, no STARVATION events ahead of it.

Counting events before `shim_dispatch` in the JSON output of
`parse-trace.py`:

| Run | events_before | decoded ts | corrected SoC = ts − (1 + events_before) |
|-----|---:|---:|---:|
| Pre  | 356 | 4272 | **3915** |
| Post | 0   | 3916 | **3915** |

Both confirm SoC=3915 — matching the bridge log exactly.

## Why pre vs post differ in event density

The shim ch1 (S2MM_1) is the output channel: kernel_out → memtile_out
→ shim_S2MM_1 → host. It is configured (Idle→BdSetup) at SoC=2971 in
both runs. After BdSetup → AcquiringLock → Transferring, the channel
sits in `Transferring` waiting for the kernel to produce output data.
While waiting, `DmaStreamStarvation` fires every cycle.

- **Pre-closure**: the kernel runs immediately (no cold-start delay),
  but the *output* channel's STARVATION still fires from SoC=3475 (the
  first cycle after trace_unit's start_event arrival where ch1 is in
  Transferring) and continues until output data arrives ~SoC=4500.
  ~1000+ STARVATION events, half of them firing **before** ch2's
  MM2S_0 START_TASK appears in the shim stream byte order.
- **Post-closure**: cold-start delays MM2S_0's first-BD payload by
  2500 cyc. The compute kernel waits 2500 cyc longer before producing
  output. So ch1's STARVATION starts firing ~2500 cyc later in SoC
  (around SoC=5969). Critically, this is *after* MM2S_0 START_TASK
  fires at SoC=3915, so MM2S_0 START_TASK is the **first** event
  encoded in the shim byte stream.

The cold-start change reordered which event appears first in the
encoded stream. Drift accumulation differs accordingly.

## Why this isn't a bug in our encoder

The convention `timer += 1 + delta` per event matches mlir-aie's
`parse_trace.convert_commands_to_json` (parse.py:260,275). HW emits
the same encoded stream layout (the Single0/1/2 and Multiple0/1/2
discriminators are AM020-defined). HW's decoded ts under the same
convention has the same drift property.

So the encoder/decoder pair is *internally consistent* with the
upstream reference. The methodology error was assuming decoded `ts`
== SoC cycle when it's actually `SoC + 1 + events_before_in_tile`.

## What this invalidates

### Phase A/B/C closure measurement (`3357b7c` commit message)

The commit reported:

> Stage 1+2 (shim_dispatch -> memtile_s2mm_done): HW=2699 EMU=3029 (gap −330)
> Total (shim_dispatch -> kernel_acquired): HW=2748 EMU=3062 (gap −314)

These are inter-tile subtractions of decoded `ts`. The shim anchor
`shim_dispatch` carries +N cyc of drift where N = STARVATION events
preceding it. The compute anchor `kernel_acquired` carries near-zero
drift on compute core (LOCK_STALL stays one frame, no STARVATION
events). Subtracting them lets the shim drift bleed into the gap.

Pre-closure: shim_dispatch had +356 drift, post: +1. Compute anchors
unchanged. So the apparent gap *closed* by 355 cyc more than the
real change. The "-314 cyc residual" is fictional — most of it is
shim event-density drift artefact.

### Phase C per-stage table

The same decoded-ts-as-SoC assumption affects every cross-tile entry
in the Phase C anchor table (`docs/superpowers/findings/2026-05-10-phase-c-stage-attribution.md`).
HW values are decoded with the same convention, so HW is similarly
inflated by HW-side STARVATION density. The `Stage 1+2 gap = 2537 cyc`
that motivated the cold-start work is itself partly drift, partly real.

## What still works

### Single-tile measurements with no events between anchors

`compute LOCK_STALL → INSTR_LOCK_ACQUIRE_REQ` is clean: both events
fire on the compute core trace unit, and **no other compute-core
events fire between them** (the kernel is stalled — no INSTR_VECTOR,
no other stalls toggling, no INSTR_EVENT_0 since T0 already fired).
Drift = 0. Decoded delta = SoC delta.

| Measurement | HW | EMU pre | EMU post | Closure | Residual |
|---|---:|---:|---:|---:|---|
| LOCK_STALL → ACQUIRE_REQ | 3015 | 319 | 2843 | +2524 | EMU 172 cyc UNDER |

This is the load-bearing closure measurement. **EMU is 172 cyc UNDER
HW**, not 314 OVER as the commit claimed. The cold-start of 2500 cyc
under-corrected; closing this residual would need ~2672.

### Single-tile intra-shim and intra-memtile

Within shim or memtile, *if* events between anchors are sparse,
intra-tile deltas are roughly clean. With dense STARVATION between,
they aren't.

`memtile S2MM_SEL0 FINISHED_BD → MM2S_SEL0 FINISHED_BD`: HW=13, EMU=16
post. Memtile trace has no STARVATION events in this kernel (slot
config doesn't include them), so this is clean.

## Action items

1. **Add SoC-cycle correction to `parse-trace.py`.** Emit an
   `events_before` field per event so consumers can compute
   `soc_cycle = ts - 1 - events_before`. Or emit a corrected `soc`
   field directly. Optional flag, since downstream tools currently
   expect mlir-aie's convention.

2. **Re-base Phase C residual analysis on single-tile compute
   measurement.** The −172 cyc UNDER on `LOCK_STALL → ACQUIRE_REQ`
   is the canonical residual. Update
   `2026-05-10-phase-c-stage-attribution.md` and the closure commit
   message references.

3. **Re-evaluate cold-start tuning.** If we want exact closure on this
   workload, bump `shim_ddr_cold_start_cycles` from 2500 to ~2672.
   But the workload-mix question still applies; defer until (4)
   gives more evidence.

4. **Per-stage attribution on a different-pattern kernel** (originally
   blocked on this discovery being out of the way). When done, use
   single-tile measurements throughout — preferably anchored on the
   compute tile's stall→acquire cycle (drift-free).

5. **Audit other findings.** Phase A/B residency claims (`2026-05-10
   -phase-a-trace-cycles-measurement.md`,
   `-phase-b-runtime-seq-instrumentation.md`,
   `-dma-waterfall-pipeline-fill-decomp.md`) all use decoded ts. They
   need re-checking under the corrected convention.

## Key data files

- `/tmp/claude-1000/preclose-results/.../trace_raw.bin` — pre-closure
  shim trace with 356 STARVATION events before MM2S_0 START_TASK
- `/tmp/claude-1000/decomp-results/.../trace_raw.bin` — post-closure
  shim trace with 0 events before MM2S_0 START_TASK
- `/tmp/claude-1000/preclose-events.json`,
  `/tmp/claude-1000/decomp-events.json` — decoded event lists used
  to compute the drift correction

## See also

- `2026-05-10-phase-c-stage-attribution.md` — the original Phase C
  finding; needs the residual numbers re-stated under the corrected
  convention.
- Commit `3357b7c` — the closure work; the implementation is sound,
  only the residual diagnostic in the commit message is wrong.
- mlir-aie `python/utils/trace/parse.py:223` — reference
  `convert_commands_to_json` showing the +1 implicit increment.
- `tools/trace_decoder/decode.py:198` — our matching python
  implementation.
- `src/device/trace_unit/mod.rs:843` — encoder writing raw delta.
