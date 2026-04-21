# Lock Subsystem: aie-rt Cross-Validation Report

**Audit date**: 2026-03-12 (re-audit)
**Auditor**: Agent G (Emulator Accuracy Sweep, re-run)
**Reference**: aie-rt `locks/xaie_locks_aieml.c`, `global/xaiemlgbl_reginit.c`,
`global/xaiemlgbl_params.h`, `global/xaiegbl.h`

## Summary

The lock subsystem was re-audited against aie-rt's AIE2/AIEML lock
implementation. All seven previously cataloged issues (LOCKS-1 through
LOCKS-7) have been resolved. No new bugs found. 95 lock-related tests pass.

## 1. Acquire/Release Semantics

### Lock_Request Register Interface (MATCH)

aie-rt uses a memory-mapped Lock_Request register to issue acquire/release
operations. The register address encodes the operation parameters:

```
RegOff = BaseAddr + (LockId * LockIdOff) + [RelAcqOff for acquire]
       + ((change_value & 0x7F) << 2)
```

Key constants from `xaiemlgbl_reginit.c`:
- `BaseAddr = LOCK_REQUEST` (0x40000 for compute, 0xD0000 for memtile)
- `LockIdOff = 0x400` (spacing between lock IDs in request space)
- `RelAcqOff = 0x200` (offset from release base to acquire base)
- Change value: 7-bit field (`XAIEML_LOCK_VALUE_MASK = 0x7FU`), shifted
  left by 2 (`XAIEML_LOCK_VALUE_SHIFT = 0x2U`)

The emulator does not model the Lock_Request register interface directly
(it would be an MMIO path). Instead, the lock arbiter receives structured
`LockRequest` values with the same semantic content. This is functionally
equivalent because the emulator controls all requestors (core instructions
and DMA engine).

### Acquire Semantics (MATCH)

aie-rt `_XAieMl_LockAcquire()` writes to the Lock_Request register with
the `RelAcqOff` added, then polls `Request_Result` (bit 0) for success.
The hardware checks the precondition and applies the delta atomically.

The emulator implements two acquire modes matching hardware:
- **acquire_ge** (`acquire_with_value`): checks `value >= expected`, then
  applies delta. Used when BD `Lock_Acq_Value` is negative.
- **acquire_eq** (`acquire_equal`): checks `value == expected`, then applies
  delta. Used when BD `Lock_Acq_Value` is positive.

The BD-to-arbiter mapping in `submit_acquire_request()`:
- `acq_value < 0`: GE mode, `expected = |acq_value|`, `delta = acq_value`
- `acq_value > 0`: EQ mode, `expected = acq_value`, `delta = -acq_value`
- `acq_value == 0`: default GE(1), `delta = -1`

This matches the mlir-aie convention where `AcquireGreaterEqual(1)` compiles
to `acq_val = -1` in the BD word.

### Release Semantics (MATCH)

aie-rt `_XAieMl_LockRelease()` writes to the Lock_Request register without
the `RelAcqOff`, then polls for success. In the emulator, release always
applies the delta and saturates at bounds. The arbiter treats releases as
always granted, consistent with hardware where release is non-blocking and
the delta is applied regardless.

`release_with_value(delta)` saturates at `MIN_VALUE` (-64) and `MAX_VALUE`
(+63), setting overflow/underflow flags. This matches aie-rt's
`LockValUpperBound = 63` and `LockValLowerBound = -64`.

### Lock Value Delta (7-bit signed) (MATCH)

The `XAIEML_LOCK_VALUE_MASK = 0x7FU` (7 bits) in the Lock_Request register
gives a change_value range of -64 to +63 (as signed). This matches:
- `LockValUpperBound = 63`
- `LockValLowerBound = -64`
- BD `Lock_Acq_Value` and `Lock_Rel_Value` fields: 7 bits, sign-extended
  via `sign_extend_7bit()` in `bd.rs`.

## 2. Lock Value Range

### Lock_Value Register (6-bit) (MATCH)

The Lock_Value register (`Lock0_value`, `Lock1_value`, etc.) uses a 6-bit
field for reading/writing the current lock value:
- `LOCK_VALUE_WIDTH = 6`
- `LOCK_VALUE_MASK = 0x3F`
- LSB = 0

This is a different interface from the Lock_Request register. The emulator's
`LockValueLayout::sign_extend()` (on the archspec-side AIE2_LOCK_VALUE_LAYOUT) correctly handles this 6-bit field, deriving the
width and mask from the AM025 register database at startup.

### Value Range (MATCH)

- `Lock::MAX_VALUE = 63` matches `LockValUpperBound = 63`
- `Lock::MIN_VALUE = -64` matches `LockValLowerBound = -64`
- Value type: `i8` in emulator, `s8` (signed 8-bit) in aie-rt `XAie_Lock`

## 3. Lock Counts Per Tile Type (MATCH)

| Tile Type | aie-rt NumLocks | Emulator | Source |
|-----------|-----------------|----------|--------|
| Compute   | 16 | 16 | `AieMlTileLockMod.NumLocks = 16U` |
| MemTile   | 64 | 64 | `AieMlMemTileLockMod.NumLocks = 64U` |
| Shim NOC  | 16 | 16 | `AieMlShimNocLockMod.NumLocks = 16U` |

Cross-validated by build-time extraction (`gen_aiert_locks.rs`) and runtime
tests in `aiert_validation::aiert_locks`.

## 4. Lock Register Stride (MATCH)

Lock value registers (SetVal interface) are spaced 0x10 apart:
- `LockSetValOff = 0x10` for all tile types
- Compute: `Lock0_value = 0x1F000`, `Lock1_value = 0x1F010`, ...
- MemTile: `Lock0_value = 0xC0000`, `Lock1_value = 0xC0010`, ...
- Shim NOC: `Lock0_value = 0x14000`, `Lock1_value = 0x14010`, ...

The emulator derives these from the AM025 register database:
- `memory_lock_base = 0x1F000` (verified: `Lock0_value` offset)
- `memory_lock_stride = 0x10` (verified: `Lock1_value - Lock0_value`)
- `memtile_lock_base = 0xC0000`
- `memtile_lock_stride = 0x10`

Cross-validated at build time (`aiert_locks::compute_lock_set_val_stride_matches_regdb`).

## 5. Quadrant Routing (Core Lock IDs 48-63) (MATCH)

Core instructions access locks via a 6-bit lock ID that maps to four
quadrants of 16 locks each:

| ID Range | Quadrant | Maps to |
|----------|----------|---------|
| 0-15 | South | Row-1 neighbor (MemTile for row 2 compute) |
| 16-31 | West | Col-1 neighbor (not yet connected) |
| 32-47 | North | Row+1 neighbor (not yet connected) |
| 48-63 | East = Internal | Own tile's memory module locks 0-15 |

The emulator implements this in `route_lock()` (`control.rs:429-453`):
- IDs >= 48: own tile locks, `id = (raw_lock_id - 48) % locks.len()`
- IDs 0-15: MemTile locks (when `mem_tile_locks` is provided)
- IDs 16-47: fallback to own tile (West/North not yet connected)

This matches `AIE2TargetModel::isMemEast` (which returns `isInternal`)
and `getLockLocalBaseIndex` in mlir-aie.

## 6. Cross-Tile Lock Access (MemTile 192-Entry Space) (MATCH)

MemTile DMA BDs use an 8-bit lock ID field addressing a 192-entry space
across three adjacent columns:

| ID Range | Target |
|----------|--------|
| 0-63 | West column MemTile (col-1) |
| 64-127 | Own MemTile |
| 128-191 | East column MemTile (col+1) |

The emulator implements this in `resolve_lock_id_static()` (`engine.rs`):
```
if lock_id < num_locks:        -> LockTarget::West(lock_id)
if lock_id < num_locks * 2:    -> LockTarget::Own(lock_id - num_locks)
if lock_id < num_locks * 3:    -> LockTarget::East(lock_id - num_locks * 2)
```

The `NeighborLocks` struct provides mutable access to west and east
neighbor tiles through disjoint borrows, matching the hardware interconnect.

## 7. Lock Arbiter (Round-Robin) (CORRECT, implementation detail)

The emulator implements a round-robin lock arbiter that serializes competing
requests from Core and DMA channels. This is an implementation detail not
directly derived from aie-rt (which uses polling), but matches the AM020
description of lock arbitration behavior:
- One grant per lock per cycle
- Round-robin priority rotation
- Releases processed before acquires in the same cycle
- Core releases deferred by 1 cycle (submitted Phase 2, resolved Phase 3)

## 8. DMA Lock Synchronization (MATCH)

The DMA engine correctly uses BD lock fields:
- `Lock_Acq_Enable`: gates whether acquire is attempted
- `Lock_Acq_ID`: target lock (4-bit for compute, 8-bit for memtile)
- `Lock_Acq_Value`: 7-bit signed delta (sign-extended via `sign_extend_7bit`)
- `Lock_Rel_ID`: target lock for release
- `Lock_Rel_Value`: 7-bit signed delta (release_value != 0 triggers release)

The DMA FSM pipeline matches hardware timing:
```
BdSetup -> AcquiringLock -> MemoryLatency -> Transferring -> ReleasingLock -> BdChaining
```

Lock operations are routed through the arbiter (`submit_lock_requests` +
`resolve_lock_requests`) to handle contention with other channels and core.

## 9. Lock Set/Get (CDO Path) (MATCH)

CDO `WRITE` commands to lock value registers dispatch through
`DeviceState::write_tile_register()` -> `write_lock_value()`:
- Computes `lock_idx = (offset - base) / stride`
- Sign-extends the value from the 6-bit register field
- Sets the lock value (clamped to -64..+63)

Both compute and memtile paths are handled. Shim tile locks are stored
in the raw register map but not yet wired to functional lock state.

`MASK_WRITE` commands are handled by `mask_write_lock_value()` which
does proper read-modify-write on the unsigned representation.

## Known Remaining Gaps

1. **Shim tile locks**: CDO writes stored in register map but not connected
   to functional lock state. Low priority -- shim tile DMA uses locks for
   host buffer synchronization, which the emulator handles differently.

2. **West/North quadrant routing**: Core lock IDs 16-47 fall back to own
   tile locks instead of accessing west/north neighbor tiles. These quadrants
   are rarely used in practice (standard patterns use East=Internal for own
   locks and South for MemTile synchronization).

3. **Lock_Request register MMIO**: The emulator does not model the
   Lock_Request register address space (0x40000+ for compute, 0xD0000+ for
   memtile). Host-side lock operations go through the CDO Set/Get path
   instead. This is functionally equivalent because the emulator controls
   all requestors internally.

## Test Coverage

95 lock-related tests pass across the codebase:
- `device::tile::tests`: 10 tests (Lock operations, flags, acquire modes)
- `device::state::tests`: 1 test (CDO lock write path)
- `device::regdb::tests`: 1 test (Lock register validation)
- `device::aiert_validation::aiert_locks`: 7 tests (Cross-validation)
- `device::registers::tests`: 1 test (Lock offset lookup)
- `interpreter::timing::sync::tests`: 4 tests (Lock timing model)
- `interpreter::timing::deadlock::tests`: 3 tests (Deadlock detection)
- `interpreter::timing::hazards::tests`: 1 test (Lock stall stats)
- `interpreter::execute::memory::tests`: 1 test (Register read for lock value)
- `interpreter::test_runner::tests`: 1 test (Lock-synchronized producer/consumer)
- `ffi::tests`: 3 tests (FFI lock value access)
- `vcd::lock_mapping::tests`: 9 tests (VCD signal mapping)
- `vcd::mapping::tests`: 3 tests (VCD tree resolution)
- `vcd::dma_mapping::tests`: 1 test (DMA lock ID resolution)
- `vcd::tolerance::tests`: 1 test (Lock tolerance defaults)
- `vcd::report::tests`: 1 test (Report subsystem entry)
- `trace::tests`: 2 tests (Lock event IDs)
- `tablegen::types::tests`: 1 test (Intrinsic lock ops)
- `testing::test_cpp_parser::tests`: various (lock patterns in test parsing)
