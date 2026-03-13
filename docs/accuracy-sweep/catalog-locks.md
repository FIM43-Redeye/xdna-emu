# Lock Subsystem Divergence Catalog

**Audit date**: 2026-03-12
**Reference**: aie-rt branch xlnx_rel_v2025.2

---

## [LOCKS-1] Lock register stride wrong (4 instead of 0x10)

- **Severity**: CRITICAL
- **Our behavior**: Lock value registers were indexed at 4-byte stride:
  `lock_idx = (offset - 0x1F000) / 4`. This means offset 0x1F010 decoded
  as lock 4, and lock 5 was at offset 0x1F014. CDO lock initialization
  writes were going to the wrong lock indices.
- **aie-rt behavior**: `LockSetValOff = 0x10` (16 bytes between consecutive
  lock value registers). LOCK0_VALUE = 0x1F000, LOCK1_VALUE = 0x1F010, ...,
  LOCK15_VALUE = 0x1F0F0. See `xaiemlgbl_reginit.c:2455` and
  `xaiemlgbl_params.h:10697`.
- **Impact**: Every CDO that initializes locks via register writes would set
  the wrong lock indices. This affects all bridge tests that use DMA with
  lock synchronization (add_one_using_dma, matrix_mult, etc.). The bug was
  latent because most test patterns only used a few locks and the addressing
  happened to work by accident for lock 0.
- **Suggested fix**: Change `/4` to `/0x10` in state.rs and registers.rs.
- **Fixed in-place**: yes
  - `src/device/state.rs`: two occurrences (write_register, mask_write_register)
  - `src/device/registers.rs`: one occurrence (lookup_lock)
  - `src/device/registers.rs`: test_lookup_lock updated

---

## [LOCKS-2] DMA lock acquire ignores BD's acquire_value

- **Severity**: HIGH
- **Our behavior**: `DmaEngine::try_acquire_lock()` called `lock.acquire()`
  (simple decrement by 1) instead of using the BD's `acquire_value` as the
  signed delta. The transfer's `acquire_value` field was stored but never
  used during acquire.
- **aie-rt behavior**: DMA BD lock acquire uses a signed change_value delta.
  `LockAcqVal` is stored in the BD word and the hardware checks
  `value + delta >= 0` before applying the delta. Negative deltas are the
  typical consume pattern (e.g., delta=-1 means "wait for value >= 1, then
  decrement"). See `xaie_dma_aieml.c:150-151` for BD field setup,
  `xaie_locks_aieml.c:112-133` for acquire semantics.
- **Impact**: DMA transfers with non-default lock values (delta != -1)
  would behave incorrectly. For the common delta=-1 case, the old simple
  `acquire()` happened to produce the same effect by coincidence.
- **Suggested fix**: Use `lock.acquire_with_value(expected, delta)` with
  expected derived from the delta.
- **Fixed in-place**: yes
  - `src/device/dma/engine.rs`: try_acquire_lock() rewritten

---

## [LOCKS-3] DMA lock release ignores BD's release_value

- **Severity**: HIGH
- **Our behavior**: `DmaEngine::complete_transfer()` called `lock.release()`
  (simple increment by 1) instead of using the BD's `release_value` as the
  signed delta. The transfer's `release_value` field was stored but never
  used during release.
- **aie-rt behavior**: DMA BD lock release uses a signed change_value delta.
  `LockRelVal` is stored in the BD word and the hardware applies the delta
  unconditionally (release is non-blocking). See `xaie_dma_aieml.c:389-390`
  for BD word encoding.
- **Impact**: DMA transfers with non-default lock release values (delta != +1)
  would behave incorrectly. For the common delta=+1 case, the old simple
  `release()` happened to produce the same effect.
- **Suggested fix**: Use `lock.release_with_value(delta)` with delta from the
  transfer's release_value.
- **Fixed in-place**: yes
  - `src/device/dma/engine.rs`: complete_transfer() updated

---

## [LOCKS-4] DMA lock test values used wrong sign convention

- **Severity**: MEDIUM
- **Our behavior**: Tests used `with_acquire(5, 1)` (acquire_value=+1) and
  `with_release(5, 0)` (release_value=0), which doesn't match the hardware
  convention. With the old broken code these tests passed by accident because
  the values were ignored.
- **aie-rt behavior**: In hardware DMA BDs, the acquire value for a consume
  pattern is negative (delta=-1 means "wait for value >= 1, then decrement").
  Release for a produce pattern is positive (delta=+1 means "increment by 1").
  See mlir-aie `AcquireGreaterEqual, 1` which compiles to `acq_val = -1` in
  the BD, and `Release, 1` which compiles to `rel_val = +1`.
- **Impact**: Tests were encoding incorrect hardware semantics. Now that the
  engine uses the values, the tests needed to match hardware convention.
- **Suggested fix**: Update test values to use hardware-correct deltas.
- **Fixed in-place**: yes
  - `src/device/dma/engine.rs`: test_transfer_with_lock, test_lock_timing_integration
  - `src/device/dma/mod.rs`: test_bd_config_with_locks

---

## [LOCKS-5] Mem tile / shim tile lock registers not handled

- **Severity**: MEDIUM (deferred -- not lock-specific)
- **Our behavior**: `RegisterModule::from_offset()` only maps the compute tile
  lock range (0x1F000-0x1F0FF). Mem tile locks (base 0xC0000) and shim NOC
  locks (base 0x14000) map to `RegisterModule::Unknown`.
- **aie-rt behavior**: Each tile type has its own lock base address:
  - Compute: `LOCK0_VALUE = 0x1F000` (MEMORY_MODULE)
  - MemTile: `LOCK0_VALUE = 0xC0000` (MEM_TILE_MODULE)
  - ShimNOC: `LOCK0_VALUE = 0x14000` (NOC_MODULE)
  All use the same stride (`LockSetValOff = 0x10`).
  See `xaiemlgbl_params.h:10697,15911,32348`.
- **Impact**: CDO writes to mem tile or shim tile locks are silently ignored.
  This will become critical when mem tile DMA is fully implemented.
- **Suggested fix**: Implement tile-type-aware register dispatch that uses
  different offset maps per tile type.
- **Fixed in-place**: no (requires broader register architecture work)

---

## [LOCKS-6] Cross-tile lock access (core IDs 48-63) not implemented

- **Severity**: LOW (no tests exercise this path yet)
- **Our behavior**: Core instruction lock operations use the lock ID directly
  as an index into `tile.locks[]`. Lock IDs 48-63 are supposed to map to the
  tile's own memory module locks 0-15 (East=Internal quadrant per AM025).
- **aie-rt behavior**: The core uses lock IDs in the instruction encoding.
  IDs 48-63 access the compute tile's own memory module locks (which are the
  same physical locks but addressed differently by the core vs memory module).
  The memory module locks (0-15) and core-visible locks (48-63) refer to the
  same hardware resources.
- **Impact**: None currently -- generated code from mlir-aie uses the memory
  module lock IDs (0-15) in DMA BDs and the remapped IDs (48-63) in core
  instructions. The emulator would access out-of-bounds if a core instruction
  used lock ID >= 64.
- **Suggested fix**: Add ID remapping in the core lock instruction handler:
  if lock_id >= 48 && lock_id <= 63, remap to lock_id - 48.
- **Fixed in-place**: no (needs investigation of real instruction encodings)

---

## [LOCKS-7] MemTile DMA 192-entry lock address space not implemented

- **Severity**: LOW (memtile DMA not yet functional)
- **Our behavior**: MemTile locks use the same 64-entry array as compute tiles
  (just with 64 entries instead of 16). No cross-tile lock addressing.
- **aie-rt behavior**: MemTile DMA BD lock IDs use an 8-bit field addressing
  a 192-entry space: W=0-63, Own=64-127, E=128-191. This allows DMA BDs in
  a mem tile to synchronize with locks in adjacent columns.
  See `AieMlMemTileDmaMod.NumLocks = 192U` (xaiemlgbl_reginit.c:421).
- **Impact**: None currently -- memtile DMA is not yet functional in the
  emulator.
- **Suggested fix**: Implement when memtile DMA cross-column synchronization
  is needed.
- **Fixed in-place**: no
