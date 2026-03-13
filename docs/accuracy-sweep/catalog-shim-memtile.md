# Shim / MemTile / Cascade Divergence Catalog

Audit date: 2026-03-12
Reference: aie-rt branch xlnx_rel_v2025.2 (`/home/triple/npu-work/aie-rt/driver/src/`)

---

## [SHIM] Shim DMA BD field layout differs from compute tile

- **Severity**: CRITICAL
- **Our behavior**: `write_dma_bd()` in `src/device/state.rs:252` treats all
  tile types identically: 6 words per BD at offset 0x1D000, with word layout
  `[addr_low, addr_high, length, control, d0, d1]`. The `DmaBufferDescriptor`
  struct has only 6 fields.
- **aie-rt behavior**: Shim BDs have 8 words (`XAIEML_SHIMDMA_NUM_BD_WORDS = 8U`)
  per the `_XAieMl_ShimDmaWriteBd()` function in `dma/xaie_dma_aieml.c:823`.
  The layout is completely different from compute tiles:
  - Word 0: Buffer length (32-bit full word!)
  - Word 1: Address low
  - Word 2: Address high + packet config + OOB BD ID
  - Word 3: D0 wrap/stride + secure access
  - Word 4: D1 wrap/stride + burst length
  - Word 5: SMID, AxQos, AxCache, D2 stride
  - Word 6: Iter current, iter wrap/stride
  - Word 7: Valid BD, lock config, next BD, use next BD
  The emulator writes `value` to `bd.addr_low` for word 0, but aie-rt puts
  buffer length in word 0 for shim tiles.
- **Impact**: All shim DMA transfers will have wrong address and length.
  This affects every test that uses DDR transfers (most bridge tests).
  Currently works because the DMA engine uses the higher-level `BdConfig`
  path, not the raw BD registers.
- **Suggested fix**: Make `write_dma_bd()` tile-type-aware: detect shim tiles
  (row 0) and use the correct 8-word layout. Extend `DmaBufferDescriptor`
  or add a `ShimDmaBufferDescriptor` variant.
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [SHIM] Shim DMA channel registers at different offset than compute tiles

- **Severity**: HIGH
- **Our behavior**: `write_dma_channel()` uses base offset 0x1DE00 for all
  tile types (`src/device/state.rs:291`).
- **aie-rt behavior**: Shim DMA channel control is at 0x1D200
  (`XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_CTRL = 0x1D200`), start queue at 0x1D204
  (`XAIEMLGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE = 0x1D204`). This is a
  completely different offset range than compute tiles.
  File: `global/xaiemlgbl_params.h:18663-18687`
- **Impact**: CDO commands targeting shim DMA channels at 0x1D200 are silently
  ignored (they fall into the DmaBufferDescriptor range 0x1D000-0x1D1FF but
  outside the BD region, so word > 5 causes early return). Shim DMA channels
  are never configured from CDO.
- **Suggested fix**: Add shim DMA channel offset handling to `RegisterModule::from_offset()`
  and `write_dma_channel()` with tile-type-aware dispatch.
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [SHIM] Shim lock registers at different offset than compute tiles

- **Severity**: HIGH
- **Our behavior**: Lock writes use offset 0x1F000 for all tile types
  (`src/device/state.rs:141`).
- **aie-rt behavior**: Shim lock values are at 0x14000
  (`XAIEMLGBL_NOC_MODULE_LOCK0_VALUE = 0x14000`), lock requests at 0x40000
  (`XAIEMLGBL_NOC_MODULE_LOCK_REQUEST = 0x40000`).
  File: `global/xaiemlgbl_params.h:15911,19187`
- **Impact**: CDO commands writing shim locks at offset 0x14000 are silently
  dropped (falls into Unknown register module). Shim locks are never
  initialized from CDO.
- **Suggested fix**: Make `RegisterModule::from_offset()` tile-type-aware, or
  add shim-specific offset ranges.
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [SHIM] Shim BD buffer length field is 32 bits, not 14 bits

- **Severity**: HIGH
- **Our behavior**: `DmaBufferDescriptor.length` is stored as `u32` which is
  correct in width. However, CLAUDE.md states "Shim DMA BD buffer length:
  18-bit field (0x3FFFF)" which is WRONG.
- **aie-rt behavior**: Shim BD buffer length is a full 32-bit field:
  `XAIEMLGBL_NOC_MODULE_DMA_BD0_0_BUFFER_LENGTH_WIDTH = 32`,
  `XAIEMLGBL_NOC_MODULE_DMA_BD0_0_BUFFER_LENGTH_MASK = 0xFFFFFFFF`.
  Compare: compute tile = 14-bit (0x3FFF), memtile = 17-bit (0x1FFFF).
  File: `global/xaiemlgbl_params.h:16299-16300`
- **Impact**: Documentation error. Code is accidentally correct (u32 field),
  but any masking code relying on 18-bit would truncate valid lengths.
- **Suggested fix**: Update CLAUDE.md to state 32-bit, not 18-bit. No code
  change needed since the field already uses u32.
- **Fixed in-place**: no (documentation only)

---

## [MEMTILE] MemTile register offsets completely unhandled

- **Severity**: CRITICAL
- **Our behavior**: `RegisterModule::from_offset()` in
  `src/device/registers.rs:92` maps offsets to modules using compute tile
  ranges. MemTile offsets (DMA BDs at 0xA0000, channels at 0xA0600, locks at
  0xC0000, stream switch at 0xB0000) all fall into `RegisterModule::Unknown`.
- **aie-rt behavior**: MemTile has its own register map:
  - DMA BDs: 0xA0000 (`XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_0`)
  - DMA channels: 0xA0600 (`XAIEMLGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL`)
  - Locks: 0xC0000 (`XAIEMLGBL_MEM_TILE_MODULE_LOCK0_VALUE`)
  - Stream switch: 0xB0000 (`XAIEMLGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0`)
  File: `global/xaiemlgbl_params.h:20772,28452,32348,29792`
- **Impact**: All CDO commands targeting memtile registers are silently
  ignored. MemTile DMA, locks, and stream routing are never configured.
  Tests that rely on memtile data movement will fail or produce wrong results.
- **Suggested fix**: Either:
  (a) Make `RegisterModule::from_offset()` accept a TileType parameter, or
  (b) Add memtile-specific offset ranges to the match statement (they don't
  overlap with compute tile ranges, so both can coexist).
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [MEMTILE] MemTile has 48 BDs, not 16

- **Severity**: HIGH
- **Our behavior**: `NUM_DMA_BUFFER_DESCRIPTORS = 16` in `aie2_spec.rs:227`,
  used for all tile types. `Tile` struct has `dma_bds: [DmaBufferDescriptor; 16]`.
- **aie-rt behavior**: MemTile has 48 BDs (`AieMlMemTileDmaMod.NumBds = 48`).
  Compute and shim tiles have 16 each.
  File: `global/xaiemlgbl_reginit.c:420`
- **Impact**: MemTile can only use 16 of its 48 BDs. Complex memtile
  configurations with BD chaining across more than 16 BDs will fail.
- **Suggested fix**: Make BD count tile-type-dependent. Add
  `MEM_TILE_NUM_DMA_BUFFER_DESCRIPTORS = 48` to `aie2_spec.rs` and use
  dynamic sizing for memtile `dma_bds`.
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [MEMTILE] MemTile has 8 words per BD, not 6

- **Severity**: HIGH
- **Our behavior**: `write_dma_bd()` uses 6 words per BD for all tile types,
  and `DmaBufferDescriptor` has 6 fields.
- **aie-rt behavior**: MemTile BDs have 8 words
  (`XAIEML_MEMTILEDMA_NUM_BD_WORDS = 8U`) with 4D addressing support
  (`NumAddrDim = 4U`), padding fields, and additional lock/iteration config.
  Compute tiles have 6 words with 3D addressing (`NumAddrDim = 3U`).
  File: `dma/xaie_dma_aieml.c:35,285`
- **Impact**: MemTile BD words 6-7 (iteration config, padding) are never
  written or read, even if the CDO contains them.
- **Suggested fix**: Extend BD parsing to be tile-type-aware with 8-word
  support for memtile BDs.
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [MEMTILE] MemTile DMA lock addressing uses 192-entry space

- **Severity**: MEDIUM
- **Our behavior**: The DMA engine and lock state use flat indexing within the
  tile's 64 physical locks.
- **aie-rt behavior**: MemTile DMA BDs address locks in a 192-entry space
  (`NumLocks = 192U`). Lock IDs 0-63 map to the West neighbor's locks, 64-127
  to the tile's own locks, and 128-191 to the East neighbor's locks. This
  allows DMA transfers to synchronize with locks in adjacent tiles.
  File: `global/xaiemlgbl_reginit.c:421`
- **Impact**: MemTile DMA BDs referencing lock IDs >= 64 (own tile) or
  >= 128 (East neighbor) will be misinterpreted. This primarily affects
  multi-tile DMA configurations where data crosses column boundaries.
- **Suggested fix**: Add lock ID remapping in the DMA engine for memtile:
  `lock_id / 64` determines the tile (W/Own/E), `lock_id % 64` is the local
  lock index within that tile.
- **Fixed in-place**: no (DMA file, Agent F owns)

---

## [CASCADE] Cascade subsystem is entirely unimplemented

- **Severity**: MEDIUM
- **Our behavior**: No `cascade.rs` file exists. The cascade interface is
  defined only as constants in `aie2_spec.rs` (width=512 bits, FIFO depth=2)
  and a `Cascade` variant in `PortType` enum. No cascade data movement, no
  cascade stall tracking, no accumulator control register handling.
- **aie-rt behavior**: The cascade subsystem consists of:
  1. **Accumulator Control Register** at offset 0x36060
     (`XAIEMLGBL_CORE_MODULE_ACCUMULATOR_CONTROL`):
     - INPUT bit (0): 0=NORTH, 1=WEST
     - OUTPUT bit (1): 0=SOUTH, 1=EAST
     Configured via `XAie_CoreConfigAccumulatorControl()` in
     `core/xaie_core.c:993`.
  2. **Core status cascade stall bits**:
     - Bit 14: CASCADE_STALL_SCD (scalar cascade data stall)
     - Bit 15: CASCADE_STALL_MCD (master cascade data stall)
     File: `core/xaie_core.h:58-59`
  3. **Cascade events**:
     - Event 25: CASCADE_STALL
     - Event 42: INSTR_CASCADE_GET
     - Event 43: INSTR_CASCADE_PUT
     File: `events/xaie_events_aieml.h:60,77-78`
  4. **Cascade FIFO**: 2-deep, 512-bit wide, between adjacent tiles.
     Direction controlled by accumulator control register.
- **Impact**: Kernels that use cascade instructions (`cascade_get`,
  `cascade_put`, accumulator forwarding between tiles) will not work.
  This is MEDIUM severity because cascade is not used in the current bridge
  test suite (simple DMA-based tests), but is common in production ML
  kernels (e.g., systolic array patterns for matrix multiply).
- **Suggested fix**: Implement cascade in phases:
  1. Add accumulator control register handling in `write_core_register()`
  2. Create per-tile cascade FIFO state (2-deep, 512-bit)
  3. Implement cascade_put/cascade_get instruction handlers
  4. Wire cascade data movement into the multi-tile execution engine
- **Fixed in-place**: no (requires new module creation)

---

## [CASCADE] Accumulator Control register (0x36060) not handled in CDO apply

- **Severity**: MEDIUM
- **Our behavior**: `write_core_register()` in `src/device/state.rs:334`
  handles offsets 0x32000 (CORE_CONTROL), 0x32004 (CORE_STATUS), 0x31100
  (CORE_PC), 0x31120 (CORE_SP), 0x31130 (CORE_LR). Offset 0x36060 falls
  through to the default `_ => {}` branch and is silently ignored.
- **aie-rt behavior**: `XAIEMLGBL_CORE_MODULE_ACCUMULATOR_CONTROL = 0x36060`
  with two 1-bit fields controlling cascade input/output direction.
  File: `global/xaiemlgbl_params.h:3732-3742`
- **Impact**: CDO commands configuring cascade direction are silently dropped.
  Cascade routing between tiles cannot be set up.
- **Suggested fix**: Add 0x36060 handling to `write_core_register()` and store
  the cascade input/output direction in `CoreState` or a new `CascadeState`.
- **Fixed in-place**: no (requires CoreState extension)

---

## [SHIM] Shim stream switch port count is underspecified

- **Severity**: LOW
- **Our behavior**: `StreamSwitch::new_shim_tile()` creates only 2 DMA slave
  ports, 2 DMA master ports, and 1 North master + 1 North slave (total 6).
- **aie-rt behavior**: The shim NOC switchbox has significantly more ports:
  6 North master, 8 North slave, plus DMA and other ports. The ShimMux
  connects 4 south ports (SOUTH2, SOUTH3, SOUTH6, SOUTH7) via mux/demux
  configuration. The total port count depends on whether you count the
  switchbox and shim mux separately.
  File: `global/xaiemlgbl_reginit.c:2244-2259`
- **Impact**: Limited routing flexibility. Multi-path shim routing
  configurations will not work. Current tests mostly use single-path DMA
  flows which still work.
- **Suggested fix**: Implement the full shim switchbox + shim_mux port model
  from the device model JSON.
- **Fixed in-place**: no

---

## [MEMTILE] MemTile stream switch port count is underspecified

- **Severity**: LOW
- **Our behavior**: `StreamSwitch::new_mem_tile()` creates 6 DMA slave + 6 DMA
  master + 1 North + 1 South (total 14).
- **aie-rt behavior**: MemTile has additional DMA ports, trace ports, and
  multiple North/South ports. The full port configuration includes up to
  24 master and 24 slave ports depending on counting method.
  File: `global/xaiemlgbl_reginit.c:29792+`
- **Impact**: Similar to shim: limited routing. Single-path flows work.
- **Suggested fix**: Use the device model JSON for full port enumeration.
- **Fixed in-place**: no

---

## [SHIM] RegisterModule::from_offset does not distinguish tile types

- **Severity**: HIGH
- **Our behavior**: `from_offset()` is a pure function of the offset with no
  tile type parameter. It maps 0x1D000-0x1D1FF to DmaBufferDescriptor
  regardless of tile type.
- **aie-rt behavior**: The same offset can mean different things in different
  tile types. For example, 0x14000 is in the Memory range (0x0-0xFFFF) for
  compute tiles but is the lock value base for shim tiles. The aie-rt uses
  separate module descriptors per tile type (AieMlTileDmaMod vs
  AieMlShimDmaMod vs AieMlMemTileDmaMod).
- **Impact**: Any register offset that differs between tile types will be
  misrouted. This is the root cause of many shim/memtile issues listed above.
- **Suggested fix**: Add a `TileType` parameter to `from_offset()` or create
  separate decoders per tile type.
- **Fixed in-place**: no (architectural change)
