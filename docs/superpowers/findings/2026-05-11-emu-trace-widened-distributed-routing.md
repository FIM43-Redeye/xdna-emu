---
name: 'EMU gap: trace routing through widened devices and distributed channels'
description: The three Tier-6 TRACE=ERROR tests (add_one_ctrl_packet, dmabd_task_queue, packet_flow_fanout) all share a single root cause with vec_mul_trace_distribute_lateral -- the emulator does not route trace traffic when the trace planner places it on a widened device column or splits it across distributed shim DMA slots.
type: project
---

# EMU gap: trace routing through widened / distributed shim DMA

## TL;DR

Four tests with HW PASS and EMU PASS produce **zero** bytes of trace
data on the emulator side, all for the same reason. The trace planner
finds the application already occupies the default shim DMA channels,
and rather than failing, it either:

1. **Widens the device** -- adds a spare column so trace traffic can use
   that column's shim DMA, OR
2. **Distributes the trace BD** -- splits trace across multiple shim DMA
   slots (e.g., channel 0 and channel 1) when a single channel can't
   carry it.

Both code paths produce kernels that run correctly on HW (real silicon
handles the alternative routing transparently) and on EMU (kernel
output is identical), but the emulator's trace event capture path
**only handles the default single-channel, origin-column layout**. When
trace events land on a widened column's shim DMA or are routed to a
distributed BD slot, the EMU drops them silently.

The trace summary classifier sees HW with hundreds of events and EMU
with zero, classifies the comparison as "count mismatch", and reports
TRACE=ERROR.

## Evidence -- the failing trio share the same `trace-prepare.log` signature

| Test | trace-prepare verdict |
|------|----------------------|
| `add_one_ctrl_packet` | occupied=[(0,0),(0,1)] → widened to **npu1_2col** |
| `packet_flow_fanout`  | occupied=[(0,0),(0,1)] → widened to **npu1_2col** |
| `dmabd_task_queue`    | occupied=[(2,0)] → **distributed plan across 4 shim DMA slots** |
| `vec_mul_trace_distribute_lateral` | test explicitly opts into distribute-channels + lateral-routing via `-aie-insert-trace-flows="distribute-channels=true lateral-routing=true"` (no occupied-channel pressure -- test design) |

For comparison, CLEAN tests:
- `add_one_cpp_aiecc`: no occupancy → default routing → CLEAN
- `packet_flow_fanin`: occupied=[(0,0)] but **fits** in remaining default channels → CLEAN
- `add_one_ctrl_packet_4_cores`: 4 application packet_flows but fits without widening → CLEAN
- `packet_flow`: 4 packet_flows, fits → CLEAN

Pattern: **widening or distribution → TRACE=ERROR. Default-channel routing → CLEAN.**

## Per-test trace_raw.bin

```
$ md5sum *.emu/trace_raw.bin *.hw/trace_raw.bin
b6d81b3...  add_one_ctrl_packet.chess.emu/trace_raw.bin   # all zeros
7dab1b5...  add_one_ctrl_packet.chess.hw/trace_raw.bin    # HW events
```

EMU trace buffers are 1 MB of zeros; HW buffers contain hundreds to
thousands of trace events.

## Why HW passes

Real silicon's shim DMA + stream switch don't care which column owns the
trace BD -- the packet flow's destination address is encoded in the BD,
and the firmware sets up the BD correctly regardless of which column we
chose. EMU's trace dispatch path checks the *origin column* and the
*default channel slot* and silently skips events that aren't routed
that way.

## What to fix in EMU

Three layers, in increasing order of effort:

1. **Trace BD instantiation on widened columns.** When trace-prepare
   widens the device to npu1_2col and the trace BD lives in column 1
   (the spare column), EMU must instantiate that column's shim tile
   and accept its DMA writeback. Today the emulator's shim DMA model
   may not be reaching the spare column for trace traffic.

2. **Distributed-channel BD writeback.** When the trace planner splits
   trace across two shim DMA channels (channel 0 + channel 1), each
   half-buffer needs the corresponding BD's writeback to land at its
   offset in the trace BO. This is what `vec_mul_trace_distribute_lateral`
   tests directly: channel 0 → bytes 0..N/2-1, channel 1 → bytes N/2..N-1.

3. **Lateral-routing trace ingress.** Trace events generated in column 0
   that get routed laterally (through the stream switch fabric to column 1's
   shim) must propagate through the EMU's stream switch model the same way
   they do on HW. This is essentially "stream switch routing already works
   for data flows -- make sure trace packet flows take the same path."

The first two are local to the shim DMA / DDR patch path. The third is
about stream switch trace traffic.

## Connection to prior work

This unifies what the 2026-05-10 bridge coverage classification doc
flagged as two separate gaps:

- "Forward gap #1: Trace-decoder ERROR on three otherwise-passing
  tests (Tier 6)" -- those three tests, fixed.
- "Forward gap #2: EMU support for distribute-channels + lateral-routing
  trace" -- the underlying mechanism, fixed.

They're the same root cause. Fixing the EMU's widened/distributed trace
routing recovers all four tests.

## Root cause confirmed (2026-05-11 night)

The drop point is upstream of the stream switch -- the trace units on the
application column never **arm**. Trace start uses BROADCAST_15: on a
widened device the trace planner places `Event_Generate` on the spare
column's shim (e.g., (2,0)), generating BROADCAST_15 there, and expects
every tile's trace unit to see the corresponding per-module hw_id
(compute = 122, shim = 125, memtile = 157). EMU's `propagate_broadcasts`
in `src/device/state/effects.rs` only propagates within the source
column. Bridge log:

```
Tile(2,0) Event_Generate: event 127 -> BROADCAST channel 15
Propagating BROADCAST channel 15 from tile (2,0) to column 2 at cycle 5099
```

Column 2 only. Column 1 (where the application + trace units live)
never receives the event, so `(1,2)` / `(1,3)` / `(1,1)` / `(1,0)`
trace units stay in `Idle` and emit no packets. The stream-switch
routing in the original breadcrumb was correct -- there's nothing to
route because the trace units are silent.

`TraceUnit (1,2) Control0: mode=EventTime start=122 stop=121 -> Idle`
is what the debug log shows; the trace unit is configured but parked.

## Why the naive "broadcast to all columns" fix breaks other tests

Attempted 2026-05-11: extended `propagate_broadcasts` to iterate every
column, on the theory that real HW's broadcast network spans the array
and default block masks are clear. Results:

| Test                        | TRACE before | TRACE after | Kernel before | Kernel after |
| --------------------------- | -----------: | ----------: | ------------: | -----------: |
| add_one_ctrl_packet         |        ERROR |   **CLEAN** |          PASS |     **PASS** |
| dmabd_task_queue            |        ERROR |   **CLEAN** |          PASS |     **PASS** |
| packet_flow_fanout          |        ERROR |       CLEAN |          PASS |     **FAIL** |
| vec_mul_trace_distribute_*  |         NONE |        NONE |          FAIL |          FAIL |

`packet_flow_fanout` regressed: kernel output came back as zeros where
HW returns 8. The `chess.trace.log` shows `EDGE_DETECTION_EVENT_0` on
tile `(0,0)` firing **963 times on EMU vs 7 on HW**, with a regular
2-cyc interval -- a classic sign of a signal toggling continuously
where it shouldn't. Some tile's edge detector input is matching the
flooded broadcast hw_id and triggering every cycle, propagating into
the kernel's data path.

The fix needs to do one of:

1. **Honor per-tile per-channel broadcast block masks** (registers
   `EVENT_BROADCAST_BLOCK_{S,W,N,E}_{SET,CLR,VALUE}` at `0x34050+stride*16`).
   On HW, the absence of trace-relevant block configuration means the
   broadcast IS supposed to flood the array, but apparently HW handles
   the flood in a way EMU doesn't model -- need to investigate whether
   trace-prepare actually writes block masks (CDO inspection) or whether
   EMU's edge detector / trace-unit notify paths have spurious activation.
2. **Verify EMU's edge-detector signal modeling.** A 2-cyc-period toggle
   pattern suggests `curr_active` is being set every step by some
   mechanism not present in HW. Possibly the broadcast notify is
   re-firing the same channel each cycle, or the edge detector polarity
   is inverted from HW.
3. **Possibly trace-prepare CDO inspection.** Check whether the trace
   planner emits block-mask register writes that EMU doesn't recognize
   (silently dropped), in which case the planner-intended scoping is
   being lost.

Tracked as task #27. The revert preserves the previous "all four tests
TRACE=ERROR, all but vec_mul_trace_distribute_lateral PASS kernel-side"
state.

## Task #27 investigation (2026-05-11 late)

Three drop-candidate audits ran in parallel: EMU broadcast/mask state,
trace-prepare CDO output, and edge-detector signal modeling. Findings:

- **EMU already parses all block-mask registers and stores `block_mask`
  per direction** in `BroadcastConfig` (`src/device/events/broadcast.rs`),
  with `is_blocked()` and `allowed_directions()` helpers. The bug is
  upstream: `propagate_broadcasts` never consults them.
- **trace-prepare emits zero block-mask writes.** AIEInsertTraceFlows
  programs only `Event_Broadcast_N_A` source-select, never
  `EVENT_BROADCAST_BLOCK_*`. So on HW the masks stay at reset (0 = no
  blocking) and the broadcast does flood the array. The flood is
  benign on HW.
- **EMU has two unrelated bugs that the flood exposes.** Both fixed
  on this iteration as defensive hardening (see "Defensive fixes
  landed" below):
  1. `propagate_broadcasts` calls `notify_mem_trace_event(hw_id=0)` on
     shim tiles and `notify_core_trace_event(hw_id=0)` on memtiles
     (there's no matching module side; hw_id 0 is the EVENT_NONE
     sentinel). The receiving `notify_*_trace_event` didn't guard
     against hw_id=0 and iterated edge detectors, where disabled
     detectors default to `input_event=0`, so the comparison
     `det.input_event == hw_id` (0 == 0) accidentally activated them.
  2. Even with hw_id != 0, the edge-detector match was vulnerable
     because it didn't skip disabled detectors. Added the
     `det.input_event != 0` guard for defense in depth.

### What still blocks #23

Tried the BFS again with the defensive fixes in place. `packet_flow_fanout`
still regresses with a different failure mode: **kernel deadlocks at
~58k cycles** rather than producing zeros. The deadlock signature:

- One broadcast event (channel 15) fires at cycle 5088 from tile (2,0).
- BFS reaches application column 1; trace units on (1,2) and (1,3)
  ARM and start capturing trace events (LOCK_STALL = hw_id 26, etc.).
  Bridge log shows `TraceUnit (1,2): EventPc mode received event hw_id=26`
  warnings beginning shortly after the broadcast.
- DMA(2,0) ch1 (trace destination, 1 MB total) enters Transferring at
  cycle 6782 and never advances.
- Application DMAs continue making progress for ~50k cycles, then
  the last forward DMA activity is `DMA(1,1) ch9` entering
  Transferring at cycle 58049. After 100k cycles of no progress the
  stall detector halts the run.

So the bug surface has *moved* from edge detectors to the trace
plumbing itself: once the application-column trace units start
emitting words, something between trace_unit -> tile stream switch
-> col 2 shim DMA backpressures and eventually deadlocks the
application DMA chain. Candidates:

- Trace stream contention on the application column's stream switch
  arbiter (slave[23]/[24] in packet mode) starving the data flow.
- A timer-reset side effect: trace-prepare configures timer reset on
  BROADCAST events, and when the broadcast arrives on the application
  column the timer reset interacts with the kernel in a way that HW
  doesn't model the same.
- Stream switch packet header / pkt_id mismatch on the cross-column
  routing -- words leave (1,2) but never arrive at (2,0).

This is **task #27's true root cause for #23**: cross-column
propagation is necessary but not sufficient. The downstream trace
flow plumbing has to work end-to-end too.

## Defensive fixes landed (2026-05-11)

These two changes are safe on their own and address the spurious
edge-detector activation surfaced during this investigation:

- `notify_core_trace_event` / `notify_mem_trace_event` in
  `src/device/tile/mod.rs` now early-return when `hw_id == 0`
  (the EVENT_NONE sentinel). This prevents disabled trace units,
  unmatched edge detectors, and timer reset_event=0 sentinels from
  spuriously activating.
- Edge-detector match loop now also skips `input_event == 0`
  detectors explicitly, as defense in depth.

Bridge sweep across the four affected tests plus seven CLEAN controls
(add_one_using_dma, packet_flow_fanin, vec_vec_add_*, etc.) shows
11/11 PASS kernel-side. The pre-existing TRACE=ERROR on the four
victims is preserved -- not fixed -- by this iteration.

The `propagate_broadcasts` BFS attempt was reverted to column-only;
git history preserves the BFS code if/when the trace plumbing bug is
fixed and we can land cross-column propagation.

## Investigation breadcrumb (2026-05-11 evening)

Partial investigation on `add_one_ctrl_packet` (chess, widened to
npu1_2col, physical start_col=1):

- **EMU does configure the col-2 trace BD.** Bridge log:
  `DMA(2,0) ch1 BD15 start: total_bytes=1048576 base_addr=0x800000005000
  dir=S2MM`, channel reaches `Transferring` state at cycle 4545.
- **EMU does configure tile (1,2)'s trace slave ports.** Bridge log:
  `Tile (1,2) stream switch: slave[23] packet mode enabled`,
  `slave[23] slot[0] = 0x011F0102` (and slave[24] similarly). The
  compute trace slave port range is 23-24 (verified against
  generated `gen_stream_ranges.rs`).
- **Tiles (2,1) and (2,2) have no stream switch config.** Only tile
  (2,0) is configured in column 2 (8 stream-switch log lines vs 46 in
  column 1). This is plausibly expected if the pathfinder routes
  trace traffic vertically down column 1 first (1,2)→(1,1)→(1,0)
  and then crosses east into (2,0). Needs confirmation.
- **The DMA starves.** Channel (2,0) ch1 transitions Idle → BdSetup →
  MemoryLatency → HostPipelineLatency → Transferring, then nothing.
  No stream bytes arrive at master[5] (the source that feeds the
  shim mux's S2MM ch1).

Next step is to rerun with debug-level logging on `trace_unit` and
`stream_switch` modules to localize the drop:

```
RUST_LOG=xdna_emu::device::trace_unit=debug,\
xdna_emu::device::stream_switch=debug,\
xdna_emu::device::dma::engine=debug \
./test.exe ...
```

Drop candidates, in order of likelihood:
1. Stream switch packet header mismatch -- the events leaving (1,2)
   slave[23] don't match any master port's slot configuration on
   the route to (2,0).
2. Cross-column propagation -- the per-tile stream switch's master
   port pushes packets, but the East/West neighbor's slave doesn't
   pick them up (column boundary logic missing).
3. Trace unit not actually firing events on this kernel pattern --
   `Tile (1,2) Event_Generate` count is zero in INFO-level log; could
   be the tile's trace events never trigger because the kernel
   structure doesn't hit the configured event triggers (this would
   not be a routing bug, just an empty-trace artifact).

## See also

- `tools/trace-prepare.py` -- the planner that decides to widen vs distribute.
- `tools/mlir-trace-inject.py` -- the MLIR side that emits the post-decision routing.
- `scripts/emu-bridge-test.sh` (Phase 5 trace.summary classifier at line ~3140).
- `build/bridge-test-results/20260510/{add_one_ctrl_packet,dmabd_task_queue,packet_flow_fanout,vec_mul_trace_distribute_lateral}.*.trace.log`
  for per-test trace event count breakdowns (HW has data, EMU has zero).
- `docs/superpowers/findings/2026-05-10-bridge-coverage-classification.md`
  -- the classification doc that originally listed these as separate gaps.
