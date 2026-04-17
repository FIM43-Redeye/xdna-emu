# Plan: Shim DMA Channel Conflict Guard for Trace Injection

## Problem

Trace injection adds `aie.packet_flow` ops that route trace data from tiles
to a shim DMA S2MM channel. The planner evaluates (shim_col, channel)
candidates and picks the first where pathfinder routing succeeds. But it
**never checks whether that DMA channel is already used by the application's
own packet_flow ops** (either as source or destination).

Pathfinder correctly routes the flows (packet-switched routing can share
ports), but the HOST-SIDE BD setup conflicts: the application's host code
sets up BDs to receive computation output on that DMA channel, while the
trace system also sets up a BD for trace data on the same channel. Result:
data corruption or TDR.

## Confirmed Regressions

- `packet_flow_fanout`: TDR with trace injection, PASS without.
  Application flow #6 routes to `shim DMA:1`. All 5 trace flows also target
  `shim DMA:1`. Stream switch routing succeeds, but the DMA channel conflict
  causes kernel timeout.

- `add_one_ctrl_packet` (Chess): FAIL with trace, PASS without.
  Control packet routing shares shim DMA resources with trace data.
  DMA output corruption (0xDC07FFFF garbage in data output buffer).

## Investigation Notes

### Initial hypothesis (WRONG): Packet slot overflow

Initially suspected mlir-aie's `MAX_PACKET_STREAM_CAPACITY=32` vs hardware
`NumSlaveSlots=4` was the root cause. Investigation disproved this:

- `amsel<N>(M)` is `<arbiterID>(msel)`, NOT `<msel>(arbiter)` (AIEOps.td:491-492)
- ARBIT field: 3 bits (mask 0x7), MSEL field: 2 bits (mask 0x30) (aie-rt params.h)
- mlir-aie uses 6 arbiters x 4 msels = 24 combinations (within hardware limits)
- The 32 vs 4 discrepancy tracks different things:
  - 32 = max packet flows per channel (time-sharing a single slot via mask)
  - 4 = max routing rules (distinct arbiter+msel destinations) per slave port
- Compiled MLIR for `packet_flow_fanout` shows no port exceeding 4 rules

### Actual root cause: Shim DMA channel conflict

The planner picks `(shim_col=0, channel=1)` for trace. But application flow
#6 already routes `mem_tile DMA:3 -> shim_noc_tile DMA:1`. Both the
application and trace share the same physical S2MM channel, causing:

1. Application BD receives trace packets mixed with computation data
2. Trace BD receives application packets mixed with trace events
3. Both data streams are corrupted

The pathfinder CORRECTLY routes all packet_flow ops (routing is valid).
The conflict is at the HOST/DMA level, not the stream switch level.

## Full Audit Results

A comprehensive audit of trace-inject.py identified 10 potential conflict
dimensions. The planner focuses on **routing feasibility** (can packets reach
the shim?) but has gaps in **resource allocation** (are the target resources
free?).

### Actionable Gaps (fix now)

| # | Gap | Severity | Current Guard |
|---|-----|----------|---------------|
| 1 | Shim DMA dest channel occupied | HIGH | NONE |
| 2 | Shim DMA source channel occupied | MEDIUM-HIGH | NONE |
| 3 | BD 15 hardcoded (may collide) | MEDIUM | NONE |

### Low-Risk Gaps (document only)

| # | Gap | Severity | Why Low Risk |
|---|-----|----------|--------------|
| 4 | Lock conflicts | MEDIUM | Lowering assigns locks; trace token setup implicit |
| 5 | Event broadcast IDs 14-15 | LOW | Apps rarely use broadcast events |
| 6 | Timer configuration | LOW | Apps almost never configure timers directly |
| 7 | Trace port availability | LOW | No mlir-aie API for manual trace port use |

### Already Safe

| # | Dimension | Mechanism |
|---|-----------|-----------|
| 8 | Circuit-switched flow conflicts | Pathfinder single-pass evaluation |
| 9 | Control packet msel/arbiter | Pathfinder (priority_route attribute) |
| 10 | Multi-column widening | Retry logic with 3 widen steps |

## Solution

### Step 1: Shim DMA Channel Occupancy Check (~20 lines)

Single function that scans both source AND dest ports of all packet_flow
ops. Returns the set of occupied (col, channel) pairs.

```python
def find_occupied_shim_dma_channels(mlir_text: str) -> set[tuple[int, int]]:
    """Find (shim_col, channel) pairs used by application packet_flow ops.

    Scans BOTH source and destination ports of aie.packet_flow ops for
    references to shim tile DMA ports. These channels are occupied by the
    application and must not be reused for trace collection.

    Source conflicts: application sends data FROM shim DMA to tiles
    (e.g., control packets, input data via packet routing).

    Dest conflicts: application receives data AT shim DMA from tiles
    (e.g., output data via packet routing).

    Both directions occupy the physical DMA channel and its BD queue.

    Derived from the MLIR source (no compilation needed).
    """
    occupied = set()
    # Match both packet_source and packet_dest targeting shim DMA ports.
    # Shim tiles appear as %shim_noc_tile_C_0 or %tile_C_0 (row 0).
    for m in re.finditer(
        r'aie\.packet_(?:source|dest)\s*<\s*'
        r'%(?:shim_noc_tile|shim_tile|tile)_(\d+)_0\s*,\s*'
        r'DMA\s*:\s*(\d+)\s*>',
        mlir_text,
    ):
        col, ch = int(m.group(1)), int(m.group(2))
        occupied.add((col, ch))
    return occupied
```

### Step 2: Filter Candidates (~5 lines)

In `plan_trace_route()` at line 934, replace the unfiltered candidate
generation:

```python
# -- Generate candidates: all device columns x 2 channels -------------
occupied = find_occupied_shim_dma_channels(mlir_text)
if occupied:
    print(f"  Planner: shim DMA channels occupied by application: "
          f"{sorted(occupied)}", file=sys.stderr)
candidates_params = [
    (col, ch) for col in range(num_cols) for ch in range(2)
    if (col, ch) not in occupied
]

if not candidates_params:
    return TracePlan(
        False,
        f"all shim DMA channels occupied by application packet flows "
        f"(occupied: {sorted(occupied)})",
    )
```

### Step 3: IR-Level Equivalent (~15 lines)

For code paths that already have a parsed Module (the `_evaluate_candidate`
subprocess), add an IR-walking version:

```python
def find_occupied_shim_dma_channels_ir(device_op) -> set[tuple[int, int]]:
    """IR-walking version of find_occupied_shim_dma_channels.

    Walks PacketFlowOp source/dest ports in the DeviceOp.
    """
    from aie.extras.util import find_ops
    import aie.dialects.aie as aiedialect

    occupied = set()
    for op in find_ops(device_op.operation,
                       lambda o: isinstance(o.opview, aiedialect.PacketFlowOp)):
        flow = op.opview
        # Check source port
        src = flow.source
        src_tile = src.owner.opview
        if (hasattr(src_tile, 'row') and src_tile.row.value == 0
                and src.bundle == aiedialect.WireBundle.DMA):
            occupied.add((src_tile.col.value, src.channel.value))
        # Check all destination ports
        for dest_block in flow.dests.blocks:
            for dest_op in dest_block.operations:
                d = dest_op.opview
                d_tile = d.tile.owner.opview
                if (hasattr(d_tile, 'row') and d_tile.row.value == 0
                        and d.bundle == aiedialect.WireBundle.DMA):
                    occupied.add((d_tile.col.value, d.channel.value))
    return occupied
```

### Step 4: BD 15 Safety Check (~10 lines)

The trace system hardcodes BD 15 for shim DMA trace collection
(trace-inject.py line 1411). mlir-aie typically allocates BDs 0..N from the
bottom, making BD 15 usually free. Add a diagnostic check:

```python
def check_bd_availability(device_op, shim_col: int, bd_id: int = 15) -> bool:
    """Verify that the target BD is not already allocated.

    Scans shimDMAAllocationOp and DMABDOp on the target shim tile.
    Returns True if bd_id is free.
    """
    # Walk BDOps on the target shim tile
    # If any existing BD uses bd_id, return False
    ...
```

If BD 15 is occupied, log a warning. This is defensive -- in practice BD 15
is always free because mlir-aie allocates from 0 upward and typical tests
use at most 4-8 BDs. But the check costs nothing and prevents a subtle
failure mode.

### Step 5: Document Assumptions (~comments only)

Add comments in trace-inject.py documenting the low-risk assumptions:

```python
# Trace resource assumptions (validated by audit 2026-03-11):
#
# - Broadcast events 14-15: reserved for trace start/stop signaling.
#   Applications rarely use broadcast events directly.
#
# - Timer: trace configures tile timers for timestamp generation.
#   Applications almost never configure timers (compiler doesn't emit
#   timer configuration).
#
# - Trace ports: Trace:0 (core module) and Trace:1 (mem module) are
#   assumed available on all tiles. No mlir-aie API exposes manual
#   trace port configuration.
#
# - Locks: trace uses token-based DMA sync (enable_token=1) which
#   operates independently of lock-based application synchronization.
#   The aie-assign-lock-ids pass handles lock allocation for the
#   application; trace tokens use a separate mechanism.
```

## Test Cases

```python
class TestShimDMAChannelGuard:
    def test_dest_channel_excluded(self):
        """DMA channel used by packet_flow dest is excluded."""
        mlir = MLIR_WITH_PACKET_DEST_SHIM_DMA_1
        occupied = find_occupied_shim_dma_channels(mlir)
        assert (0, 1) in occupied

    def test_source_channel_excluded(self):
        """DMA channel used by packet_flow source is excluded."""
        mlir = MLIR_WITH_PACKET_SOURCE_SHIM_DMA_0
        occupied = find_occupied_shim_dma_channels(mlir)
        assert (0, 0) in occupied

    def test_both_channels_occupied_infeasible(self):
        """When all channels on all columns are occupied, plan is infeasible."""

    def test_unoccupied_channel_accepted(self):
        """Standard test with no shim DMA packet_flow ops has no exclusions."""
        mlir = STANDARD_ADD_ONE_MLIR  # uses aie.flow, not packet_flow
        occupied = find_occupied_shim_dma_channels(mlir)
        assert len(occupied) == 0

    def test_occupied_on_col0_uses_col1(self):
        """If col 0 channels are occupied, planner can use col 1."""
        # Multi-column test where col 0 has packet flows but col 1 is free
```

## Files to Modify

| File | Changes |
|------|---------|
| `tools/trace-inject.py` | `find_occupied_shim_dma_channels()`, `find_occupied_shim_dma_channels_ir()`, candidate filter in `plan_trace_route()`, BD safety check, assumption comments |
| `tools/test_trace_inject.py` | 5 test cases for the channel guard |
| `scripts/trace-quarantine.txt` | Remove entries once guard catches them automatically |

## Verification

1. `python -m pytest tools/test_trace_inject.py -v` -- new tests pass
2. `python tools/trace-inject.py mlir-aie/test/npu-xrt/packet_flow_fanout -o /tmp/pff-test`
   -- should report "shim DMA channel (0,1) occupied" and use (0,0) or skip
3. `python tools/trace-inject.py mlir-aie/test/npu-xrt/add_one_ctrl_packet -o /tmp/ctrl-test`
   -- should report occupied channels and use an unoccupied one or skip
4. `python tools/trace-inject.py mlir-aie/test/npu-xrt/add_one_using_dma -o /tmp/aoud-test`
   -- should succeed with no exclusions (uses aie.flow, not packet_flow)
5. Bridge run `--compile` on quarantined tests -- either pass (free channel
   found) or skip cleanly with diagnostic
6. Full bridge run -- no new regressions

## Hardware Reference

- Shim tile DMA: 2 S2MM channels (0, 1) for device->host
- Shim tile DMA: 2 MM2S channels (0, 1) for host->device
- Each channel has a BD queue (16 BDs, indices 0-15)
- Packet-switched BDs can filter by packet ID, but the host-side setup
  (xrt::bo, set_arg) assumes exclusive channel ownership
- All toolchain-derived: shim DMA channel count from device model
  (`aie-device-models.json`), packet_flow ops from MLIR source

## Estimated Effort

~1.5 hours including tests:
- Channel occupancy function + candidate filter: 30 min
- IR-level equivalent: 15 min
- BD safety check: 15 min
- Test cases: 20 min
- Assumption documentation: 10 min
