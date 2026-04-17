# Stream Switch Divergence Catalog

Audit by Agent H (re-run, 2026-03-12) against aie-rt stream_switch
sources: xaie_ss_aieml.c, xaie_ss.c, xaiemlgbl_reginit.c, xaie_helper.c,
xaiemlgbl_params.h.

Port layouts for all three tile types match aie-rt AieMlTileStrmSw*PortMap
arrays exactly (verified entry-by-entry). Packet routing register bit
positions match xaiemlgbl_params.h. Circuit-mode slave index computation
is compatible. No CRITICAL or HIGH items.

## [STREAM-1] SLAVE_SELECT_MASK width mismatch (COSMETIC)

- **Severity**: LOW
- **File**: gen_stream_ranges.rs (generated), state.rs line 1018
- **Our behavior**: 5-bit mask (0x1F) for circuit-mode slave select
- **aie-rt behavior**: 7-bit CONFIGURATION field (0x7F, bits 6:0)
- **Impact**: None for AIE2 -- max slave index is 24 (fits in 5 bits).
  Would matter for a hypothetical architecture with >31 slave ports.
- **aie-rt ref**: xaiemlgbl_params.h CONFIGURATION_MASK=0x7F,
  _XAie_GetSlaveIdx writes slave index to full 7-bit field.
- **Suggested fix**: Widen mask to 0x7F for forward compatibility.
  In circuit mode (packet_enable=0), upper bits (6:5) are always 0.
- **Fixed in-place**: no (no functional impact)

## [STREAM-2] Deterministic merge not behaviorally implemented

- **Severity**: LOW
- **File**: stream_switch.rs (no det-merge code), state.rs (registers absorbed)
- **Our behavior**: Deterministic merge register writes are absorbed with
  no behavioral effect on arbitration order.
- **aie-rt behavior**: `XAie_StrmSwDeterministicMergeConfig` programs
  slave ID and packet count per arbiter position. Enable/disable via
  `XAie_StrmSwDeterministicMergeEnable`. 2 arbiters, 4 positions each.
- **Impact**: No bridge test or CDO uses deterministic merge. Standard
  arbitration (lower-index-first) is used for all current test cases.
- **aie-rt ref**: xaiemlgbl_reginit.c AieMlAieTileStrmSwDetMerge struct,
  xaie_ss.c XAie_StrmSwDeterministicMergeConfig
- **Suggested fix**: Implement when a test case exercises it.
- **Fixed in-place**: no

## [STREAM-3] Per-tile-type port validity not enforced

- **Severity**: LOW
- **File**: stream_switch.rs (no PortVerify equivalent)
- **Our behavior**: All port connections accepted regardless of tile type.
  Any slave can connect to any master if indices are valid.
- **aie-rt behavior**: `PortVerify` function pointer per tile type
  enforces connection rules:
  - Compute: TRACE can only go to FIFO/SOUTH/DMA(0). CORE slave cannot
    connect to CORE master. DMA slave can only connect to same-channel
    DMA master or non-DMA masters.
  - MemTile: TRACE only to SOUTH or DMA(5). No WEST/EAST connections.
  - Shim: TRACE only to FIFO/SOUTH/WEST(0)/EAST(0). No CORE/DMA.
- **Impact**: CDO always programs valid routes. Invalid routes (which
  cannot occur from valid CDO) would be silently accepted but would not
  produce incorrect results because the hardware would also not support
  them.
- **aie-rt ref**: xaie_ss_aieml.c _XAieMl_AieTile_StrmSwCheckPortValidity,
  _XAieMl_MemTile_StrmSwCheckPortValidity, _XAieMl_ShimTile_StrmSwCheckPortValidity
- **Suggested fix**: Add validation for defensive error reporting. Not
  needed for correctness.
- **Fixed in-place**: no

## [STREAM-4] Test comment -- RESOLVED

- **Severity**: RESOLVED (was LOW)
- **Our behavior**: Comments in test_stream_switch_compute now correctly
  state 4 South masters for compute tile.
- **Status**: Already fixed in current codebase on dev branch.

## [STREAM-5] Stale module doc comment -- FIXED

- **Severity**: TRIVIAL
- **File**: stream_switch.rs lines 1-37
- **Our behavior**: Module doc header claimed "This stub does NOT model
  packet switching (only circuit switching)". This was false -- packet
  switching is fully implemented with slot matching, arbiter locking,
  drop-header, and backpressure.
- **Impact**: Misleading for developers reading the source.
- **Fixed in-place**: YES -- updated doc comment to accurately describe
  what is and is not modeled.

## [STREAM-6] Circuit route not removed on disable

- **Severity**: LOW
- **File**: state.rs write_stream_switch, write_memtile_stream_switch
- **Our behavior**: When a master config register is written with
  enable=0, the port's `enabled` flag is cleared but the LocalRoute
  entry persists in the `local_routes` vector.
- **aie-rt behavior**: XAie_StrmConnCctDisable writes 0 to both master
  and slave config registers, fully clearing the connection.
- **Impact**: Functionally equivalent -- disabled ports do not forward
  data (the `step()` loop checks `route.enabled`... actually it does NOT
  check port.enabled, it checks route.enabled which is always true once
  created). However, `slave_select=0` with enable=0 would overwrite the
  route to slave 0. Since CDO always writes specific enable=1 configs
  before use and the entire device is reset between runs, stale routes
  from previous runs do not persist.
- **Suggested fix**: Remove LocalRoute entries when master is disabled.
- **Fixed in-place**: no (no observed functional impact)

## Summary

| ID | Severity | Area | Status |
|----|----------|------|--------|
| STREAM-1 | LOW | Slave select mask width | Not fixed (cosmetic) |
| STREAM-2 | LOW | Deterministic merge | Not implemented |
| STREAM-3 | LOW | Port validity checks | Not implemented |
| STREAM-4 | RESOLVED | Test comment | Already correct |
| STREAM-5 | TRIVIAL | Module doc comment | FIXED |
| STREAM-6 | LOW | Route removal on disable | Not fixed |

No CRITICAL or HIGH items. All LOW items are defensive checks or
forward-compatibility concerns with no impact on current test suite.
