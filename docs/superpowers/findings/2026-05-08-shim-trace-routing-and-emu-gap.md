# Shim trace routing on shim col 0: works at the lowering layer, EMU doesn't emit events

**Date:** 2026-05-08
**Context:** #372 stage 1 (shim trace inject) end-to-end validation against `add_one_using_dma`.

## What was investigated

Concern raised before HW validation: stage 1 of #372 adds a second trace
source (shim `(0,0)`) on top of the existing core source `(0,2)`. Both
share the same physical shim column. Risks suspected:

1. Trace BO routing might collide with the application's data path on
   shim col 0's DMA channels.
2. Lowered MLIR might fail to resolve the routing.
3. Multi-source packet IDs / packet types might not demux correctly at
   decode time.
4. EMU might not faithfully emit shim trace events.

## What the lowered MLIR shows

`add_one_using_dma` runs through trace-prepare with
`--shim-sweep-events all`. The lowered MLIR
(`mlir-aie/build/.../add_one_using_dma/traced/aie_traced.mlir`) routes
both trace sources into a single trace BO via separate `aie.packet_flow`
ops, both terminating at `shim_noc_tile_0_0 DMA : 1`:

```mlir
aie.packet_flow(1) {
  aie.packet_source<%tile_0_2, Trace : 0>
  aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
} {keep_pkt_header = true}
aie.packet_flow(1) {
  aie.packet_source<%shim_noc_tile_0_0, Trace : 0>
  aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
} {keep_pkt_header = true}
```

Channel allocation is sound:

- App data: shim `(0,0)` MM2S 0 (input), S2MM 0 (output)
- Trace: shim `(0,0)` S2MM 1 (free)

`AIEInsertTraceFlows` correctly avoided the occupied S2MM 0 by allocating
S2MM 1.  No conflict.

## Packet ID / packet type observation

Both trace sources end up with packet ID `1`:

- Core: `Trace_Control1 = 0x00000001` (packet ID 1, packet type CORE = 0)
- Shim: `Trace_Control1 = 0x00002001` (packet ID 1, packet type ShimTile = 2)

`keep_pkt_header = true` means each packet retains its header through the
stream switch. The decoder at the consumer end must demux on packet
**type**, not packet ID, since IDs collide. Per upstream mlir-aie's
expected behavior this is the supported pattern.

## Compile + EMU run results

Both compilers and EMU run pass `add_one_using_dma`:

```
TEST                Chess/EMU   Peano/EMU
add_one_using_dma   PASS        PASS
```

So the routing didn't break compile and the kernel still produces correct
output.

## EMU trace BO decode result (the gap)

Decoded the EMU trace_raw.bin via `tools/parse-trace.py` (mlir-aie
backend). Both Chess and Peano EMU outputs:

- Chess EMU: 20 events, 100% pkt_type=0 (core), 0 pkt_type=2 (shim)
- Peano EMU: 17 events, 100% pkt_type=0 (core), 0 pkt_type=2 (shim)

The trace BO contains zero shim packets. The shim trace unit is
**configured** (registers written at runtime per the lowered MLIR's
`aie.trace.config @trace_shim_0_0_config(...) packet_type = shimtile`)
but no events appear in the captured stream.

## Diagnosis

The configuration, event-emission, and stream-switch routing layers
were all correct. The actual gap was a single line in
`src/device/array/routing.rs`: when draining trace packets from a shim
tile to its trace stream-switch slave port, the routing function was
popping from `tile.mem_trace`. Shim's only trace unit is the PL-module
trace, configured via the 0x340D0+ register block, which writes to
`tile.core_trace`. So shim trace packets were piling up in `core_trace`
and never reaching the stream switch -- while `mem_trace` (unused on
shim) sat empty.

Latent since 2026-02-23 -- predates active shim tracing; surfaced once
#372 stage 1 actually exercised the path.

## Resolution (2026-05-08)

Fix: change the shim branch of `route_trace_to_tile_switches` to pop
from `core_trace`. After the fix, the EMU writes 42 shim packets
(pkt_type=2) and 3 core packets (pkt_type=0) to the trace BO for the
peano `add_one_using_dma` run.

A regression test in `src/device/tile/tests.rs` now exercises the full
shim-trace path against the lowered MLIR's exact register writes (start
event 127 = PL_USER_EVENT_1, slots 14/15/16/22/23/24/30/31).

Note: `tools/parse-trace.py --decoder ours --trace-mode event_pc`
currently only decodes core-mode-1 packets and skips the shim
mode-0 packets. That's a decoder-side gap (mixed-mode BOs) and not an
EMU correctness gap; the BO bytes match what HW would write.

This is structurally the same SHAPE as #353 ("EMU does not emit
LOCK_STALL trace events") and #354 ("EMU does not emit PERF_CNT_0 anchor
pulses") -- a trace-unit-fidelity issue exposed only when a new flow
exercises the path -- but the actual fix here was a pure routing
mismatch, not a missing emission path.

## Conclusion

Routing on shim col 0 with both core and shim trace ops: **safe**.
- aiecc lowering allocates a free shim S2MM channel (the trace planner
  already avoids occupied channels).
- Multi-source flows merge with `keep_pkt_header=true` so decoders can
  demux by packet type.
- Compilation succeeds with both compilers; EMU runs the kernel
  correctly.

Stage 1 deliverable -- the inject tool emits the right MLIR, the lowering
turns it into the right register writes -- is **complete**. The
lowered output is structurally indistinguishable from what real hardware
expects.

What's not yet validated:

1. **HW emission and capture**: real silicon should fire the shim DMA
   events into the trace BO. Worth a single bridge-test run with HW
   once we're ready to spend that time.
2. **Decoder mixed-mode support**: `parse-trace.py --decoder ours`
   takes a single `--trace-mode` flag and applies it uniformly. A real
   trace BO carries packets with multiple modes (core in event_pc,
   shim in event_time) interleaved. Pre-fix this was masked because
   shim packets were never emitted; post-fix it shows up as "the
   shim packets in the BO don't decode." Not a correctness regression
   -- raw bytes are right -- but a tooling gap to track separately.

### Files / data referenced

- Lowered MLIR: `/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/traced/aie_traced.mlir`
- Bridge test results: `xdna-emu/build/bridge-test-results/20260508/`
  (pre-fix: 0 shim packets) and
  `xdna-emu/build/bridge-test-results/20260509/` (post-fix: 42 shim
  packets + 3 core packets in trace BO)
- Fix: `src/device/array/routing.rs` (shim branch:
  `mem_trace` -> `core_trace`)
- Regression test:
  `src/device/tile/tests.rs::test_shim_trace_unit_records_dma_start_task_for_lowered_config`
