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

The configuration and routing layers are correct. The gap is in the
emulator: it isn't faithfully emitting shim-tile trace events for
`DMA_S2MM_*_START_TASK` / `DMA_S2MM_*_FINISHED_TASK` /
`DMA_*_STREAM_STARVATION`.

This is structurally the same shape as #353 ("EMU does not emit
LOCK_STALL trace events") and #354 ("EMU does not emit PERF_CNT_0 anchor
pulses") -- the EMU's trace unit modeling has gaps. Stage 1 of #372
exposes another gap: shim DMA event sourcing.

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

What's not yet validated, and is **not in scope for stage 1**:

1. **HW emission and capture**: real silicon should fire the shim DMA
   events into the trace BO. Worth a single bridge-test run with HW once
   we're ready to spend that time.
2. **EMU gap fix**: emitting shim trace events on the EMU side is its
   own task (similar to how #353 / #354 are tracked separately). This
   doesn't gate #372 -- the EMU calibration target per
   `feedback_emu_calibration_approach.md` is "ballpark + deterministic,
   not perfectly accurate." Treating shim events as another known
   trace-fidelity gap is consistent.

### Files / data referenced

- Lowered MLIR: `/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/traced/aie_traced.mlir`
- Bridge test results: `xdna-emu/build/bridge-test-results/20260508/add_one_using_dma.{chess,peano}.emu/`
- Decoded events: 20 core / 0 shim (Chess), 17 core / 0 shim (Peano)
