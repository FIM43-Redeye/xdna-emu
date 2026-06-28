# Send-port consumer-pacing cadence: pipeline-latency fix + ingress-depth/drain-rate diagnosis

**Date:** 2026-06-28  **Issue:** #140  **Status:** root cause 1 FIXED+committed;
root cause 2 DIAGNOSED, implementation queued.
**Branch:** `send-cadence-fidelity` (off master). Latency fix = commit `f4009413`.
**Working notes + raw evidence:** `build/experiments/sendcadence-cotrace/INVESTIGATION.md`

This is the durable handoff: read it cold and implement the faithful model
without re-deriving. The known-fidelity-gaps row (send-port cadence) points here.

## The gap

`add_one_using_dma` memtile MM2S **send** port `PORT_RUNNING_4`:
- **HW:** `[8,8,14,2,14,2,6,8,1]` over ~201cy, `PORT_STALLED_4=[1,1,25,60,66]`.
- **EMU (pre-work):** `[16,16,16,16]` then one end stall -- the producer empties
  the whole 64-word transfer in ~54cy, never throttled by the consumer's
  mid-stream lock-stalls.

The compute consumer (S2MM) lock-stalls ~56-63cy per 8-word buffer in BOTH worlds
(EMU is faithful here -- 7-8 stalls). The defect is purely that the consumer's
stalls do not backpressure the producer in EMU.

## Method that cracked it (reusable)

1. **Co-trace on ONE BROADCAST_15-synced run.** Trace producer + consumer events
   in a single capture so they share a timebase. `events.json` carries a `soc`
   field that is a **shared cross-tile clock** -- align directly on it (no skew
   gymnastics). Recipe:
   ```
   env -u XDNA_EMU -u XDNA_EMU_RUNTIME \
     XDNA_TRACE_MEMTILE_EVENTS="PORT_RUNNING_4,PORT_STALLED_4" \
     XDNA_TRACE_MEMMOD_EVENTS="DMA_S2MM_0_STALLED_LOCK,DMA_S2MM_0_START_TASK,DMA_S2MM_0_STREAM_STARVATION" \
     ./scripts/emu-bridge-test.sh --no-hw --chess-only --trace -v add_one_using_dma
   ```
   (`--no-hw` for EMU-only after the HW reference is captured; drop it for HW.)
   Results: `build/bridge-test-results/<date>/add_one_using_dma.chess.{hw,emu}/events.json`.
   Tiles: (1,0)=shim, (1,1)=memtile=PRODUCER, (1,2)=compute=CONSUMER.
2. **Per-stage cascade probe** `XDNA_EMU_STAGE_PROBE=1` (default-off, committed in
   `coordinator.rs`). Per cycle dumps memtile-output -> port FIFOs -> crossing
   in-flight -> compute port FIFOs -> S2MM ingress, plus consumer FSM phase +
   locks + producer beat/stall. Lands in the EMU run's `.bridge.log`; grep `[STAGE]`.
   Accessors it uses: `DmaEngine::channel_phase`, `TileArray::inflight_to_tile`.

## HW reference values (measured, keep)

- producer `PORT_RUNNING_4=[8,8,14,2,14,2,6,8,1]`, `PORT_STALLED_4=[1,1,25,60,66]`, span ~201cy.
- recv `PORT_RUNNING_0=[16,16,16,16]` / `STALLED_0=[1,1,1]` (relay-fill, already matched).
- consumer lock-stall onset -> producer stall onset lag = **~1-6cy** (= crossing
  latency). HW backpressures the producer within a crossing-time of the consumer
  stalling -- NOT after filling a deep buffer.
- HW "14 = 8 + 6" = consumer-ingress HEADROOM (8, one drained buffer) + crossing (6).

## What was RULED OUT (don't re-investigate)

- **Phase ordering** (consume-before-produce): already handled by the `drained`
  counter (`stream_io.rs`, `reset_cycle_drain_counters` + `can_accept_stream_in_for_routing`).
- **Fix A (FSM-gate accept on `AcquiringLock`)**: WRONG -- breaks the HW-verified
  cold recv absorb (a lock-stalled S2MM legitimately stages a full BD into the skid).
- **aiesim as a timing oracle for this**: it over-buffers too (~30 words vs HW ~14);
  it is a *mechanism* reference (registered occupancy FIFO), not a depth oracle.
- **Crossing model**: already HW-faithful (`e044ac72`, bounded delay-line + FIFO).

## ROOT CAUSE 1 -- switch pipeline double-advanced (FIXED, commit f4009413)

`route_streams` steps every tile's stream switch TWICE per cycle (Step 3 before
inter-tile delivery, Step 5 after, so crossing-delivered words flow same-cycle).
`StreamSwitch::step()` begins with `advance_switch_pipeline()`, which decrements
each in-flight word's `cycles_remaining`. Two `step()` calls -> the AM020
register-slice latency was decremented **twice per cycle = halved**. Words
crossed the fabric at ~2x silicon rate.

**Fix:** split `step()` -> advance ONCE per cycle (first pass), accept in both.
`StreamSwitch::step_accept_only()` + `TileArray::step_tile_switches_accept_only()`;
`route_streams` Step 5 uses accept-only. Tests:
`two_pass_per_cycle_advances_pipeline_once_not_twice`, `step_accept_only_never_advances`.

**Validated:** `--lib` 3552/0 (one test, `inter_tile_crossing_flows_when_dest_drains`,
was a pre-existing compensation -- its `>=40 in 48cy` total was tuned to the halved
latency; rewritten to assert steady-state ~1 word/cy throughput). Bridge chess
sweep 148/148 HW pass, 0 bridge fail; recv `[16,16,16,16]` guards preserved.

**It is a real accuracy win but does NOT close the gap**: producer span only
54 -> 69cy. Cascade DEPTH unchanged. The latency was halved; correcting it does
not reduce capacity.

## ROOT CAUSE 2 -- over-deep per-channel S2MM "ingress" + wrong drain rate (NEXT)

**Device-model misread (the key error).** `s2mm_ingress_fifo_depth=16`
(`model_builder.rs:230`) was derived from the decrypted aiesim model's
`s2mmChannel.buffer_depth=12` (+master 4). But that field is the **memory-side DMA
buffer, NOT the AXI stream skid**: the SHIM has `mm2s.buffer_depth=256`,
`s2mm.buffer_depth=64` with `max_outstanding_transactions=128`,
`write_queue_depth=17`, `read_resp_queue_depth=64` -- those are DDR/NoC
latency-hiding depths. A 256-word AXI skid is absurd. So `buffer_depth` = memory
buffer. The real per-channel AXI skid is small; the 16-deep number belongs on the
**switch central FIFO** (AM020 ch2, compute tile only; memtile has none per ch5),
SHARED across channels, not replicated per S2MM channel.

**Wrong drain rate.** `stepping.rs:1027`: an S2MM transfer `involves_stream()`, so
EMU drains its ingress->memory at `stream_words_per_cycle=1`. That conflates the
stream READ rate (1/cy, correct -- AXI delivers 1/cy) with the memory WRITE rate
(should be `words_per_cycle` ~= 4/cy, the data-memory bus). Because EMU drains the
skid at only 1/cy, the skid builds up and *needed* depth 16 to avoid fragmenting
recv -- a compensation for too-slow drain.

**DEPTH-KNOCKOUT EXPERIMENT (confirms depth is the lever).** Set depth 16 -> 4,
re-ran the co-trace:

| | depth=16 | depth=4 | HW |
|---|---|---|---|
| producer run span | 54cy | **183cy** | ~201cy |
| producer stalls | 1 (end) | **6** (throughout) | 5 |
| PORT_RUNNING_4 intervals | streams free | **7** | 8 |

A shallow skid nearly closes the gap. depth=4 alone breaks recv -- which is why
the fix is the mechanism, not the constant.

## IMPLEMENTATION PLAN (the faithful model)

Goal: recv stays `[16,16,16,16]` (via fast drain, not deep buffering) AND send ->
HW `[8,8,14,2,14,2,6,8,1]`.

1. **Drain the S2MM ingress at the memory-bus rate.** `stepping.rs:1027` -- the
   S2MM ingress->memory write should use `words_per_cycle` (~4/cy), NOT
   `stream_words_per_cycle`. The MM2S *egress* stays `stream_words_per_cycle=1`
   (the relay-fill fix -- correct). Distinguish direction: S2MM drain = memory
   rate; MM2S fill-of-stream = stream rate. The S2MM beat = `min(words_available,
   words_per_cycle)` -- drain up to the memory rate, limited by what the stream
   delivered.
2. **Shrink the per-channel skid** `s2mm_ingress_fifo_depth` 16 -> ~6-8
   (`model_builder.rs:230`). Bracketed: 4 over-throttles slightly (183<201), 16 is
   way under. Pin with HW (step 4).
3. **Relocate the 16-deep FIFO to the switch central FIFO** (shared) if needed for
   recv contiguity once the drain rate is fixed -- may be unnecessary once drain is
   fast. Verify, don't over-build.
4. **Pin the skid with HW** if the co-trace is ambiguous: lock-stall the compute
   consumer with a >8-word feed, watch where the recv port backpressures.

**Validation ladder:**
- recv `[16,16,16,16]` preserved (the two `*_recv_stages_full_bd_while_buffer_lock_stalled`
  unit tests, `dma/engine/tests.rs:1215,1269` -- these assert per-channel absorb-16
  and ARE the compensation; reconcile them to the new mechanism, don't just delete).
- send co-trace -> HW `[8,8,14,...]` (producer span ~201cy, ~8 bursts, lag ~6cy).
- data: bridge chess sweep 148/148, 0 fail.
- `cargo test --lib` green.
- Every moved trace triaged win-vs-regression; a regression that's a compensation
  for this bug gets fixed, not preserved (Maya's principle).

## Key files

- `crates/xdna-archspec/src/model_builder.rs:230` -- `s2mm_ingress_fifo_depth`.
- `src/device/dma/engine/stepping.rs:1025-1031` -- the drain-rate branch.
- `src/device/dma/engine/stream_io.rs:42,349` -- `input_fifo_capacity`,
  `can_accept_stream_in_for_routing`.
- `src/device/dma/engine/tests.rs:1215,1269` -- recv absorb-16 guards (compensation).
- `src/device/stream_switch/mod.rs` -- `step`/`step_accept_only` (latency fix landed).
- `src/device/array/routing.rs` -- `route_streams`, `step_tile_switches{,_accept_only}`.
- `src/interpreter/engine/coordinator.rs` -- `XDNA_EMU_STAGE_PROBE`.
- Decrypted device model: `build/experiments/aiesim-device-decrypt/NPU1.json`.
