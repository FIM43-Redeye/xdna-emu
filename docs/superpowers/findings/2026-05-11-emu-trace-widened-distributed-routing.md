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

## See also

- `tools/trace-prepare.py` -- the planner that decides to widen vs distribute.
- `tools/mlir-trace-inject.py` -- the MLIR side that emits the post-decision routing.
- `scripts/emu-bridge-test.sh` (Phase 5 trace.summary classifier at line ~3140).
- `build/bridge-test-results/20260510/{add_one_ctrl_packet,dmabd_task_queue,packet_flow_fanout,vec_mul_trace_distribute_lateral}.*.trace.log`
  for per-test trace event count breakdowns (HW has data, EMU has zero).
- `docs/superpowers/findings/2026-05-10-bridge-coverage-classification.md`
  -- the classification doc that originally listed these as separate gaps.
