# Lock Subsystem Divergence Catalog

**Audit date**: 2026-03-12 (re-audit)
**Reference**: aie-rt branch xlnx_rel_v2025.2

---

## [LOCKS-1] Lock register stride wrong (4 instead of 0x10) -- RESOLVED

- **Severity**: CRITICAL
- **Status**: FIXED (previous sweep)
- **Description**: Lock value registers were indexed at 4-byte stride
  instead of the correct 0x10 (16 bytes). CDO lock initialization writes
  went to wrong lock indices.
- **Resolution**: Lock base/stride are now derived from the AM025 register
  database via `DeviceRegLayout` (`memory_lock_base`, `memory_lock_stride`,
  `memtile_lock_base`, `memtile_lock_stride`). Cross-validated at build time
  against aie-rt constants.

---

## [LOCKS-2] DMA lock acquire ignores BD's acquire_value -- RESOLVED

- **Severity**: HIGH
- **Status**: FIXED (previous sweep)
- **Description**: DMA engine used simple `lock.acquire()` (decrement by 1)
  instead of the BD's `acquire_value` delta.
- **Resolution**: DMA lock acquire now routes through the lock arbiter with
  proper delta values derived from the BD's `Lock_Acq_Value` field.
  `submit_acquire_request()` maps the signed `acquire_value` to
  `(expected, delta, equal_mode)` for the arbiter.

---

## [LOCKS-3] DMA lock release ignores BD's release_value -- RESOLVED

- **Severity**: HIGH
- **Status**: FIXED (previous sweep)
- **Description**: DMA engine used simple `lock.release()` (increment by 1)
  instead of the BD's `release_value` delta.
- **Resolution**: DMA lock release now routes through the lock arbiter with
  the BD's `Lock_Rel_Value` as the delta. The `ReleasingLock` FSM state
  carries the `release_value` from the transfer.

---

## [LOCKS-4] DMA lock test values used wrong sign convention -- RESOLVED

- **Severity**: MEDIUM
- **Status**: FIXED (previous sweep)
- **Description**: Tests used incorrect sign conventions for lock values,
  masked by the fact that values were previously ignored.
- **Resolution**: Tests updated to match hardware convention (negative
  acquire values for GE mode, positive release values for produce pattern).

---

## [LOCKS-5] Mem tile / shim tile lock registers not handled -- RESOLVED

- **Severity**: MEDIUM
- **Status**: FIXED
- **Description**: `write_tile_register()` only mapped compute tile lock
  range. Mem tile and shim tile lock offsets fell through to Unknown.
- **Resolution**: `DeviceState::write_tile_register()` now dispatches
  `SubsystemKind::Lock` for all tile types. The `subsystem_from_offset()`
  function uses tile-kind-aware offset ranges from `xdna-archspec`. Compute
  and memtile locks are properly routed to `write_lock_value()` with the
  correct base/stride. Shim locks are stored in the raw register map.

---

## [LOCKS-6] Cross-tile lock access (core IDs 48-63) -- RESOLVED

- **Severity**: MEDIUM
- **Status**: FIXED
- **Description**: Core instruction lock operations used the raw lock ID
  as an index, ignoring the quadrant routing where IDs 48-63 map to the
  tile's own memory module locks 0-15.
- **Resolution**: `route_lock()` in `control.rs` implements full quadrant
  routing:
  - IDs 48-63 -> own tile locks 0-15 (East = Internal)
  - IDs 0-15 -> MemTile locks (South, when provided)
  - IDs 16-47 -> fallback to own tile (West/North not yet connected)

  The coordinator passes MemTile lock copies to the core interpreter,
  enabling proper synchronization between compute tile cores and MemTile
  DMA operations.

---

## [LOCKS-7] MemTile DMA 192-entry lock address space -- RESOLVED

- **Severity**: MEDIUM
- **Status**: FIXED
- **Description**: MemTile locks used a flat 64-entry array with no
  cross-tile lock addressing.
- **Resolution**: `resolve_lock_id_static()` in `engine.rs` implements
  the full 192-entry address space:
  - IDs 0-63: West neighbor (col-1) via `LockTarget::West`
  - IDs 64-127: Own tile via `LockTarget::Own`
  - IDs 128-191: East neighbor (col+1) via `LockTarget::East`

  `NeighborLocks` provides safe disjoint borrows for cross-tile access.
  `submit_lock_requests()` and `check_acquire_granted()` route operations
  to the correct tile's arbiter.

---

## [LOCKS-8] Shim tile locks not connected to functional state -- OPEN

- **Severity**: LOW
- **Status**: OPEN (by design)
- **Description**: Shim tile CDO lock writes are stored in the raw register
  map but do not update functional `Lock` state. The shim tile lock value
  registers (base 0x14000, stride 0x10) are recognized as
  `SubsystemKind::Lock` but the handler only logs and stores.
- **aie-rt behavior**: `AieMlShimNocLockMod` has the same structure as
  compute/memtile lock modules (NumLocks=16, LockSetValOff=0x10,
  LockValUpperBound=63, LockValLowerBound=-64).
- **Impact**: None in practice. Shim tile DMA uses locks for host buffer
  synchronization, which the emulator handles through the XRT plugin's
  synchronous execution model rather than lock polling.
- **Fix when**: Shim tile DMA lock synchronization is needed for a test case.

---

## [LOCKS-9] West/North quadrant routing for core lock instructions -- OPEN

- **Severity**: LOW
- **Status**: OPEN
- **Description**: Core lock IDs 16-31 (West) and 32-47 (North) fall back
  to own tile locks instead of routing to neighbor tiles. Only East
  (Internal, IDs 48-63) and South (IDs 0-15) are properly routed.
- **aie-rt behavior**: The hardware routes all four quadrants to their
  respective neighbor tiles. West goes to col-1 compute tile, North goes
  to row+1 compute tile.
- **Impact**: Minimal. Standard AIE2 patterns use East/Internal for own
  locks and South for MemTile synchronization. West/North cross-tile lock
  access is rare in practice (only used in multi-tile compute kernels with
  horizontal/vertical lock chaining).
- **Fix when**: A test case exercises West or North lock access.

---

## [LOCKS-10] Lock_Request register MMIO not modeled -- OPEN (by design)

- **Severity**: INFORMATIONAL
- **Status**: OPEN (by design)
- **Description**: The emulator does not model the Lock_Request register
  address space (compute: 0x40000+, memtile: 0xD0000+). In hardware, the
  host or a DMA control packet can issue lock operations by writing to
  these MMIO addresses. The emulator handles lock operations through
  structured `LockRequest` values instead.
- **aie-rt behavior**: `_XAieMl_LockAcquire/Release()` write to computed
  offsets within the Lock_Request space, encoding the lock ID and
  change_value in the address bits.
- **Impact**: None for current use cases. CDO-based lock initialization
  uses the Lock_Value registers (SetVal interface). Core instructions and
  DMA BDs use the arbiter directly. Control packets that target lock
  request registers would silently fail.
- **Fix when**: A control packet or CDO command targets the Lock_Request
  address space rather than the Lock_Value registers.
