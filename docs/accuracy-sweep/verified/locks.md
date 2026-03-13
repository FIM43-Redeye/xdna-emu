# Lock Subsystem: aie-rt Cross-Validation Report

**Audit date**: 2026-03-12
**Auditor**: Agent G (Emulator Accuracy Sweep)
**Reference**: aie-rt `locks/xaie_locks_aieml.c`, `global/xaiemlgbl_reginit.c`

## Summary

The lock subsystem was audited against aie-rt's AIE2/AIEML lock implementation.
Four bugs were found and fixed. Eight cross-validation tests were added.

## Verified Behaviors

### Lock Value State (MATCH)
- Lock value is 6-bit unsigned (0-63): `LOCK_MAX_VALUE = 63` matches
  `LOCK0_VALUE_LOCK_VALUE_MASK = 0x3F` (6-bit width) in xaiemlgbl_params.h.
- Value stored as u8, lower 6 bits valid. Matches hardware register width.

### Lock Counts Per Tile Type (MATCH)
- Compute tile: 16 locks -- matches `AieMlTileLockMod.NumLocks = 16U`
- Memory tile: 64 locks -- matches `AieMlMemTileLockMod.NumLocks = 64U`
- Shim NOC tile: 16 locks -- matches `AieMlShimNocLockMod.NumLocks = 16U`
  (Note: shim tile lock handling is not yet implemented in the emulator)

### Lock Value Bounds (MATCH)
- `LockValUpperBound = 63`, `LockValLowerBound = -64` (for change_value delta)
- Our `LOCK_CHANGE_VALUE_BITS = 7` correctly gives 7-bit signed range (-64..+63)

### Acquire Semantics (MATCH after fix)
- aie-rt: Acquire succeeds when `value + delta >= 0`, then applies delta
- Our `acquire_with_value(expected, delta)` checks `value >= expected` then
  applies delta. When `expected = -delta` (the standard mapping for negative
  deltas), this is equivalent.
- The `XAIEML_LOCK_VALUE_MASK = 0x7F` (7-bit) for change_value in the
  Lock_Request register matches our 7-bit delta encoding.

### Release Semantics (MATCH after fix)
- aie-rt: Release always applies delta (non-blocking), saturating at MAX_VALUE
- Our `release_with_value(delta)` matches this behavior

### Lock Set/Get (MATCH)
- `_XAieMl_LockSetValue` writes value via `LockSetValBase + LockSetValOff * LockId`
- Our `lock.set(value)` clamps to MAX_VALUE, matching hardware 6-bit field

### Lock Acquire/Release Latency (MATCH)
- `LOCK_ACQUIRE_LATENCY = 1`, `LOCK_RELEASE_LATENCY = 1`
- Consistent with aie-rt polling model (one request per attempt)

## Divergences Found and Fixed

See `catalog-locks.md` for full catalog.

1. **CRITICAL**: Lock register stride was 4, should be 0x10 (16 bytes)
2. **HIGH**: DMA lock acquire ignored BD's acquire_value delta
3. **HIGH**: DMA lock release ignored BD's release_value delta
4. **MEDIUM**: Register map/CDO lock addressing used wrong stride

## Test Coverage

8 new cross-validation tests added to `src/device/tile.rs`:
- `test_aiert_lock_value_range`
- `test_aiert_lock_set_value`
- `test_aiert_lock_counts_per_tile_type`
- `test_aiert_acquire_semantics`
- `test_aiert_release_semantics`
- `test_aiert_lock_register_stride`
- `test_aiert_lock_value_bounds`
- `test_aiert_dma_lock_pattern`

3 existing tests updated to use correct aie-rt semantics:
- `test_transfer_with_lock` (engine.rs)
- `test_lock_timing_integration` (engine.rs)
- `test_bd_config_with_locks` (mod.rs)

## Known Remaining Gaps

1. **Mem tile lock register offsets** (0xC0000 range) not handled in
   `RegisterModule::from_offset` -- all mem tile registers map to Unknown.
2. **Shim tile lock register offsets** (0x14000 range) not handled.
3. **Cross-tile lock access** (core lock IDs 48-63 mapping to neighbor tile
   memory module locks) not yet implemented in the interpreter.
4. **MemTile DMA 192-entry lock address space** (W/Own/E at 0/64/128) not
   implemented.

These are broader register-map issues, not lock-specific bugs. They will be
addressed when the full tile-type-aware register dispatch is implemented.
