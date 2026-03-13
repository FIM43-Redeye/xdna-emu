# Stream Switch Divergence Catalog

Audit by Agent H against aie-rt stream_switch/xaie_ss_aieml.c.
Port layouts for all three tile types match aie-rt AieMlTileStrmSw*PortMap
arrays exactly. Packet routing register bit positions match xaiemlgbl_params.h.

## [STREAM-1] SLAVE_SELECT_MASK width mismatch

- **Severity**: LOW
- **Our behavior**: 5-bit mask (0x1F) for slave select
- **aie-rt behavior**: 7-bit CONFIGURATION field (0x7F) in xaiemlgbl_params.h
- **Impact**: None -- max slave index is 24 for AIE2, fits in 5 bits
- **Suggested fix**: Widen mask to 0x7F for forward compatibility
- **Fixed in-place**: no

## [STREAM-2] Deterministic merge not behaviorally implemented

- **Severity**: LOW
- **Our behavior**: Register write absorbed, no behavioral effect
- **aie-rt behavior**: Deterministic merge enables priority-based arbitration
- **Impact**: No bridge test uses deterministic merge
- **Suggested fix**: Implement priority arbitration when register is set
- **Fixed in-place**: no

## [STREAM-3] Per-tile-type port validity not enforced

- **Severity**: LOW
- **Our behavior**: All port connections accepted regardless of tile type
- **aie-rt behavior**: Validates port availability per tile type
- **Impact**: CDO always programs valid routes; invalid routes silently accepted
- **Suggested fix**: Add port validity checks matching aie-rt tile type tables
- **Fixed in-place**: no

## [STREAM-4] Test comment incorrect compute tile master port count

- **Severity**: LOW
- **Our behavior**: Comment stated 6 South masters for compute tile
- **aie-rt behavior**: 4 South masters for compute tile
- **Impact**: Comment only
- **Suggested fix**: Fix comment
- **Fixed in-place**: no (was fixed in worktree, lost on cleanup)
