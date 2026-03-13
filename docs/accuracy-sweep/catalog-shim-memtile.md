# Shim / MemTile / Cascade Divergence Catalog

Audit date: 2026-03-12 (re-audit)
Reference: aie-rt branch xlnx_rel_v2025.2 (`/home/triple/npu-work/aie-rt/driver/src/`)

## Previously Cataloged Issues Now Resolved

The prior audit found 11 issues. Of those, the following are NOW RESOLVED:

1. Shim BD field layout -- RESOLVED: `parse_shim()` is fully data-driven (8 words)
2. Shim DMA channel registers at wrong offset -- RESOLVED: `write_shim_dma_channel()` uses `shim_channel_base`
3. Shim lock registers at different offset -- PARTIALLY RESOLVED: routing is correct (0x14000 -> Lock subsystem), but lock state not wired (see below)
4. MemTile register offsets unhandled -- RESOLVED: `subsystem_from_offset()` is tile-type-aware
5. MemTile has 48 BDs -- RESOLVED: `arch::memtile::NUM_BDS = 48`, Vec-based storage
6. MemTile has 8 words per BD -- RESOLVED: `parse_memtile()` handles all 8 words
7. RegisterModule::from_offset tile-type-unaware -- RESOLVED: replaced by `subsystem_from_offset(offset, tile_kind)`
8. Cascade subsystem unimplemented -- RESOLVED: FIFO state, routing, accumulator control all implemented
9. Accumulator Control register not handled -- RESOLVED: handled in `apply_tile_local_effects()`

---

## Remaining Issues

### [SHIM-1] Shim lock values not wired to lock state machine

- **Severity**: MEDIUM
- **File**: `src/device/state.rs:258-263`
- **Our behavior**: Shim lock writes (offset 0x14000+) are correctly routed to
  `SubsystemKind::Lock` by the tile-type-aware decoder, but the handler only
  stores the value in the raw register map. It does NOT update the tile's lock
  state (`tile.locks[]`), unlike compute and memtile tiles which call
  `write_lock_value()`.
- **aie-rt behavior**: Shim tiles have 16 locks
  (`XAIEMLGBL_NOC_MODULE_LOCK0_VALUE = 0x14000`, 16 locks at stride 0x10).
  These locks synchronize shim DMA transfers with host/NoC.
- **Impact**: Shim DMA BDs that specify lock acquire/release will not
  synchronize correctly. Currently, bridge tests work because the XRT plugin
  path manages synchronization externally (via XRT's buffer sync API), but
  pure-emulation flows that depend on shim lock values will fail.
- **Suggested fix**: Add shim-specific lock base/stride constants (derivable
  from regdb "shim" module Lock0_value register) and call `write_lock_value()`
  for shim tiles. Note: shim lock stride and base may need to be added to
  `DeviceRegLayout`.
- **Owner**: This agent (non-DMA fix)

---

### [SHIM-2] Shim BD d2_wrap field missing from parse_shim()

- **Severity**: LOW
- **File**: `src/device/dma/bd.rs:376`
- **Our behavior**: `parse_shim()` sets `d2_wrap: 0` unconditionally in the
  "Not used in Shim" section. However, the shim BD has no D2_Wrap register
  field in AM025 (confirmed: `NOC_MODULE_DMA_BD0_5` contains only SMID,
  AxCache, AxQoS, and D2_Stepsize -- no D2_Wrap).
- **aie-rt behavior**: `_XAieMl_ShimDmaReadBd()` does not read a D2_Wrap field
  for shim tiles. Shim addressing is 3D (D0, D1, D2 stepsizes) with D0 and D1
  wraps but no D2 wrap. This is correct.
- **Impact**: None. The field is correctly absent in hardware. The comment
  "Not used in Shim" accurately reflects that shim tiles lack D2_Wrap.
- **Action**: No fix needed. Cosmetic only.

---

### [SHIM-3] Shim BD compression_enable hardcoded to false

- **Severity**: LOW
- **File**: `src/device/dma/bd.rs:377`
- **Our behavior**: `parse_shim()` sets `compression_enable: false`
  unconditionally. The ShimBdFieldLayout does not include an
  `enable_compression` field.
- **aie-rt behavior**: Shim BDs do not have a compression field (compression
  is only supported on compute and memtile DMA). This is correct.
- **Impact**: None. Correctly reflects hardware.
- **Action**: No fix needed.

---

### [MEMTILE-1] MemTile DMA lock ID 192-entry address space remapping

- **Severity**: MEDIUM
- **File**: `src/device/dma/bd.rs:289-292`, DMA engine lock acquire/release
- **Our behavior**: MemTile BD lock IDs (`lock_acq_id`, `lock_rel_id`) are
  stored as raw 8-bit values and passed directly to the DMA engine. The DMA
  engine uses them as local lock indices within the tile.
- **aie-rt behavior**: MemTile DMA BDs address locks in a 192-entry virtual
  space. Lock IDs 0-63 map to the West neighbor's locks, 64-127 to the tile's
  own locks, and 128-191 to the East neighbor's locks. The aie-rt API
  `_XAieMl_GetNearestLocks()` in `locks/xaie_locks_aieml.c` maps these
  virtual IDs to physical tile+lock.
- **Impact**: MemTile BDs referencing own-tile locks (IDs 64-127) work by
  accident because bridge tests typically use lock IDs < 16 and the low bits
  match. Cross-tile lock synchronization (IDs 0-63 or 128-191) will fail.
  This primarily affects multi-tile DMA configurations spanning column
  boundaries.
- **Suggested fix**: Add lock ID remapping in the DMA engine for memtile BDs:
  `(lock_id / 64)` determines the tile quadrant (0=West, 1=Own, 2=East);
  `(lock_id % 64)` is the local lock index. This is a DMA engine concern.
- **Owner**: Agent F (DMA subsystem)

---

### [CASCADE-1] Cascade FIFO depth is unbounded

- **Severity**: LOW
- **File**: `src/device/tile.rs:1028,1031`
- **Our behavior**: `cascade_input` and `cascade_output` are `VecDeque<[u64; 6]>`
  with no capacity limit. Data can be pushed without bound.
- **aie-rt behavior**: The hardware cascade FIFO has depth 1 per direction (the
  doc comment says "384-bit width, depth 1"). With backpressure checking in
  `route_cascade()`, the effective depth stays at 1 in practice because
  `cascade_input.is_empty()` is checked before routing. However, if
  `push_cascade_input()` is called multiple times without intervening routing
  (e.g., during multi-core instruction execution), the FIFO can accumulate
  more than 1 entry.
- **Impact**: Functionally correct due to routing backpressure, but does not
  enforce the hardware FIFO depth limit at the push level. A kernel that
  issues multiple cascade_put instructions without intervening cascade routing
  would not stall as it should.
- **Suggested fix**: Add capacity check in `push_cascade_output()` -- if
  FIFO already has 1 entry, reject the push and signal a stall.
- **Owner**: This agent (non-DMA fix)

---

### [CASCADE-2] Cascade instruction handlers not yet connected to FIFO

- **Severity**: MEDIUM
- **File**: `src/interpreter/timing/latency.rs:296` (CascadeRead/CascadeWrite
  semantic ops return latency only)
- **Our behavior**: The interpreter recognizes `SemanticOp::CascadeRead` and
  `SemanticOp::CascadeWrite` for timing purposes but does not appear to
  connect them to the tile's cascade FIFO for actual data movement. The
  cascade FIFO push/pop helpers exist in tile.rs but need to be called from
  the instruction execution path.
- **Impact**: Cascade instructions will execute with correct timing but will
  not move actual data through the cascade FIFOs. Kernels using cascade for
  accumulator forwarding will compute wrong results.
- **Suggested fix**: In the instruction execution handlers for cascade_get and
  cascade_put, call `pop_cascade_input()` and `push_cascade_output()`
  respectively, with stall handling when FIFOs are empty/full.
- **Owner**: Interpreter team (not DMA)

---

### [SHIM-4] Shim DMA status register layout not specialized

- **Severity**: LOW
- **File**: `src/device/regdb.rs` -- `DeviceRegLayout` has no `shim_status` field
- **Our behavior**: Shim DMA status registers are not parsed. The `memory_status`
  and `memtile_status` layouts exist but there is no `shim_status` variant.
- **aie-rt behavior**: Shim DMA status registers exist at
  `NOC_MODULE_DMA_S2MM_Status_0` (0x1D220) with the same field layout as
  compute/memtile status registers (Status, Stalled_*, Channel_Running, etc.).
- **Impact**: DMA status readback for shim tiles will not work. Bridge tests
  do not currently read shim DMA status, so no functional impact yet.
- **Suggested fix**: Add `shim_status: StatusFieldLayout` to DeviceRegLayout,
  built from "shim" module's `DMA_S2MM_Status_0` register.
- **Owner**: Agent F (DMA subsystem)

---

### [DOC-1] (RESOLVED) CLAUDE.md shim BD buffer length was 18-bit

Previously cataloged as "CLAUDE.md states shim BD buffer length is 18-bit".
Now confirmed MEMORY.md correctly states "32-bit field (0xFFFFFFFF)".
No code or documentation fix needed.
