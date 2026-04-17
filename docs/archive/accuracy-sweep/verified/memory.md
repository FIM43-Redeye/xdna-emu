# Memory Model Verification Report

Audit date: 2026-03-12 (re-run)
Agent: Agent J (Memory Model)
Reference: aie-rt driver/src/ (branch xlnx_rel_v2025.2)

## Scope

Comparison of xdna-emu memory model against:
- aie-rt `global/xaiemlgbl_params.h` (register definitions)
- aie-rt `global/xaiemlgbl_reginit.c` (AieMlCoreMod, AieMlTileMemMod, AieMlMemTileMemMod)
- aie-rt `core/xaie_elfloader.c` (_XAie_GetTargetTileLoc)
- aie-rt `memory/xaie_mem.c` (DataMemWrWord, DataMemRdWord, DataMemBlockWrite)
- aie-rt `global/xaiegbl_regdef.h` (XAie_CoreMod, XAie_MemMod struct definitions)
- AM025 register database JSON (`aie_registers_aie2.json`)

Files audited in xdna-emu (dev branch, current):
- `src/device/tile.rs` -- memory allocation, read/write methods, TileParams
- `src/device/banking.rs` -- bank conflict detection (addr_to_bank, banks_for_access)
- `src/device/host_memory.rs` -- DDR/host memory interface (sparse pages)
- `src/interpreter/timing/memory.rs` -- timing model, CardDir, MemoryQuadrant, MemoryModel
- `src/interpreter/execute/memory.rs` -- core load/store execution, NeighborMemory
- `src/device/registers_spec.rs` -- address constants (PROGRAM_MEMORY_BASE, etc.)
- `src/device/registers.rs` -- subsystem routing (data memory region classification)
- `src/device/model.rs` -- device model validation (validate_against_spec)
- `src/device/arch_config.rs` -- ArchConfig trait and ModelConfig implementation
- `src/device/dma/engine.rs` -- DMA address wrapping (0x80000 offset handling)
- `build.rs` -- arch constant generation (PHYSICAL_BANK_WIDTH_BITS, MEMORY_SIZE, etc.)
- `crates/xdna-archspec/src/lib.rs` -- ArchModel construction (physical banking, CoreAddressMap)
- `crates/xdna-archspec/src/types.rs` -- BankingModel, MemoryModel, CoreAddressMap types

## Verified Items

### 1. Data Memory Sizes -- MATCH

| Property | aie-rt | xdna-emu | Source |
|----------|--------|----------|--------|
| Compute tile data memory | 64KB (0x10000) | `arch::compute::MEMORY_SIZE = 65536` | `AieMlTileMemMod.Size = 0x10000` (reginit.c:2348) |
| MemTile data memory | 512KB (0x80000) | `arch::memtile::MEMORY_SIZE = 524288` | `AieMlMemTileMemMod.Size = 0x80000` (reginit.c:2356) |
| Program memory | 16KB | `arch::compute::PROGRAM_MEMORY_SIZE = 16384` | `AieMlCoreMod.ProgMemSize = 16 * 1024` (reginit.c:2322) |
| Shim data memory | 0 (none) | `TileParams::shim().data_memory_size = 0` | aie-rt rejects shim in DataMemWrWord |

All values are generated from the ArchModel at build time. The derivation chain
is: `aie-device-models.json` -> `xdna-archspec::device_model::extract_device_model()`
-> `build.rs` -> `gen_arch.rs` -> `crate::arch::*` constants.

Memory is allocated as `vec![0u8; params.data_memory_size].into_boxed_slice()`
in `Tile::new()` (tile.rs:1334), matching aie-rt's zero-initialization behavior
(`_XAieMl_PartMemZeroInit()`).

### 2. Core Address Space (CardDir) -- MATCH

| Property | aie-rt | xdna-emu |
|----------|--------|----------|
| DataMemAddr | 0x40000 | `arch::compute::DATA_MEM_ADDR = 0x40000` |
| DataMemSize | 64 * 1024 | `arch::compute::MEMORY_SIZE = 65536` |
| DataMemShift | 16 | `arch::compute::DATA_MEM_SHIFT = 16` |
| IsCheckerBoard | 0 (false) | `arch::compute::IS_CHECKERBOARD = false` |
| CardDir formula | `Addr / CoreMod->DataMemSize` | `address / arch::compute::MEMORY_SIZE` |
| CardDir 4 = South | `Loc.Row -= 1` | `MemoryQuadrant::South` |
| CardDir 5 = West | `Loc.Col -= 1` (RowParity=1) | `MemoryQuadrant::West` |
| CardDir 6 = North | `Loc.Row += 1` | `MemoryQuadrant::North` |
| CardDir 7 = East = Local | Same tile (forced RowParity=1) | `MemoryQuadrant::Local` |

The emulator's `MemoryQuadrant::from_address()` (timing/memory.rs:122-141) exactly
mirrors `_XAie_GetTargetTileLoc()` from aie-rt xaie_elfloader.c:124-183.

Key detail: AIE2 forces `RowParity = 1` (because `IsCheckerBoard = 0`),
which means CardDir 5 (West) is always `col-1` and CardDir 7 (East) is
always the local tile. The emulator correctly handles this by checking
`IS_CHECKERBOARD` and mapping East to Local when false.

The OFFSET_MASK in execute/memory.rs (`MEMORY_SIZE - 1 = 0xFFFF`) correctly
extracts the 16-bit local offset within a 64KB tile.

### 3. Cross-Tile Memory Access Resolution -- MATCH

`NeighborMemory` (execute/memory.rs:99-210) resolves cardinal directions to
actual tile coordinates, matching aie-rt's `_XAie_GetTargetTileLoc`:

| Direction | Neighbor formula | Out-of-bounds handling |
|-----------|------------------|----------------------|
| South | (col, row-1) | Returns None if row=0 |
| West | (col-1, row) | Returns None if col=0 |
| North | (col, row+1) | Returns None via device lookup |
| East | (col+1, row) | Returns None via device lookup (unused on AIE2) |

aie-rt returns `XAIE_ERR` for out-of-bounds neighbors. The emulator returns
zeros (read from non-existent memory), which is functionally equivalent for
valid programs. Cross-tile writes are buffered and applied after the core
step completes, modeling the higher latency of cross-tile access.

### 4. Program Memory Host Offset -- MATCH

| Property | aie-rt | xdna-emu |
|----------|--------|----------|
| ProgMemHostOffset | `XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY = 0x20000` | `arch::compute::PROGRAM_MEM_HOST_OFFSET = 0x20000` |

CDO/ELF loading in `state.rs:319` correctly uses `(tile_addr.offset - 0x20000)`
to convert from host address space to program memory offset.

### 5. Data Memory Host Offset -- MATCH

| Property | aie-rt | xdna-emu |
|----------|--------|----------|
| MemAddr (compute) | `XAIEMLGBL_MEMORY_MODULE_DATAMEMORY = 0x00000000` | `arch::DATA_MEM_HOST_OFFSET = 0` |
| MemAddr (memtile) | `XAIEMLGBL_MEM_TILE_MODULE_DATAMEMORY = 0x00000000` | Same 0 offset |

### 6. Bank Organization -- MATCH

| Property | aie-rt / AM020 | xdna-emu |
|----------|----------------|----------|
| Compute physical banks | 8 (AM020 Ch2) | `arch::compute::PHYSICAL_BANKS = 8` |
| Compute physical bank size | 8KB | `arch::compute::PHYSICAL_BANK_SIZE = 8192` |
| MemTile physical banks | 16 (AM020 Ch2) | `arch::memtile::PHYSICAL_BANKS = 16` |
| MemTile physical bank size | 32KB | `arch::memtile::PHYSICAL_BANK_SIZE = 32768` |
| Compute logical banks | 4 | `arch::compute::LOGICAL_BANKS = 4` |
| MemTile logical banks | 8 | `arch::memtile::LOGICAL_BANKS = 8` |

Physical banking constants come from `xdna-archspec::apply_aie2_handcoded_constants()`
which sources them from AM020 Ch2. Structural invariant `num_banks * bank_size == size_bytes`
is enforced at construction time in `MemoryModel::new()`.

### 7. Bank Width -- MATCH

| Property | aie-rt | xdna-emu |
|----------|--------|----------|
| DATAMEMORY_WIDTH (compute) | 128 bits (params.h:7206) | `arch::compute::PHYSICAL_BANK_WIDTH_BITS = 128` |
| DATAMEMORY_WIDTH (memtile) | 128 bits (params.h:19197) | `arch::memtile::PHYSICAL_BANK_WIDTH_BITS = 128` |
| BANK_WIDTH_BYTES | 16 | `PHYSICAL_BANK_WIDTH_BITS / 8 = 16` (derived) |

The bank width drives the interleaving granularity: consecutive 16-byte blocks
map to different banks. `addr_to_bank()` in banking.rs uses `addr >> 4` (shift
by log2(16)=4 bits) which matches the 128-bit bank width.

### 8. Bank Conflict Detection -- MATCH (two implementations)

Two bank conflict implementations exist, both correct:

**banking.rs** (production path for DMA and core bank tracking):
- `addr_to_bank(addr, num_banks)`: `(addr >> 4) % num_banks`
- `banks_for_access(addr, bytes, num_banks)`: returns bitmask of all banks touched
- Parameterized by `num_banks`, works for both compute (8) and memtile (16)
- Used by DMA engine and coordinator for actual conflict detection

**timing/memory.rs** (timing model, compute-tile-only):
- `MemoryAccess::bank()`: `(addr >> 4) & 0x7` (hardcoded 8 banks)
- `MemoryModel::check_conflict()` and `record_access()`
- Only used within the compute tile timing model tests
- `NUM_BANKS` constant is `arch::compute::PHYSICAL_BANKS = 8`
- Consistent within its scope but not parameterized for memtile

Both correctly implement 128-bit interleaved banking. The banking.rs functions
are the production path; the timing model is supplementary.

### 9. Address Space Layout (Host/CDO View) -- MATCH

Register routing in `registers.rs::classify_subsystem()` correctly treats:
- Offsets `0..MEMORY_SIZE` as DataMemory for both compute and memtile
- Offsets `0x20000..0x30000` as ProgramMemory (compute only)

This matches aie-rt's tile address encoding: `MemAddr + Addr + TileAddr`
where `MemAddr` is 0 (data memory starts at offset 0 within the tile).

### 10. DMA Address Wrapping -- CORRECT

`engine.rs` wraps DMA addresses with `(addr as usize) % mem_size`, which
correctly handles the MemTile DMA address space where addresses may carry a
0x80000 offset. aie-rt validates addresses against `MemMod->Size` and
rejects out-of-range; the emulator wraps rather than faulting, which is
functionally equivalent for valid programs and more forgiving for debugging.

### 11. Host Memory Model -- ACCEPTABLE

`host_memory.rs` uses sparse 4KB-page storage (BTreeMap) for DDR simulation.
This is an emulator implementation choice -- real hardware uses the NoC/DDR
subsystem. The API (read_bytes, write_bytes, read_u32, write_u32) is
byte-addressable and correctly handles cross-page accesses.

There is no aie-rt analog for DDR memory modeling since aie-rt operates on
real hardware where DDR is managed by the OS/driver. The host memory model
is correct for emulation purposes.

### 12. Array Topology Constants -- MATCH

| Property | aie-rt | xdna-emu |
|----------|--------|----------|
| ColShift | 25 | `arch::TILE_COL_SHIFT = 25` |
| RowShift | 20 | `arch::TILE_ROW_SHIFT = 20` |
| OffsetMask | 0xFFFFF (20 bits) | `arch::TILE_OFFSET_MASK = 0xFFFFF` |

### 13. Memory Initialization -- MATCH

Both aie-rt (`_XAieMl_PartMemZeroInit`) and the emulator
(`vec![0u8; params.data_memory_size]`) initialize memory to zero.

### 14. AIE1 vs AIE2 Parameter Comparison -- VERIFIED

The codebase correctly distinguishes architecture variants:

| Parameter | AIE1 (xaiegbl_reginit.c) | AIE2 (xaiemlgbl_reginit.c) | Emulator target |
|-----------|--------------------------|----------------------------|-----------------|
| IsCheckerBoard | 1 | 0 | AIE2: false |
| DataMemAddr | 0x20000 | 0x40000 | 0x40000 |
| DataMemSize | 32KB | 64KB | 64KB |
| DataMemShift | 15 | 16 | 16 |

The emulator targets AIE2 (NPU1/Phoenix). The `IS_CHECKERBOARD` flag is
generated at build time from the ArchModel and correctly controls the
East-is-local behavior in `MemoryQuadrant::from_address()`.

## Test Coverage

Memory-related tests verified passing:
- 9 banking tests (bank addressing, bank masks, constants for compute + memtile)
- 35 timing/memory tests (bank conflicts, alignment, cross-tile quadrants, latency)
- 12 host_memory tests (read/write, cross-page, regions, statistics)
- 7 model validation tests (JSON parse, reference value cross-check)
- Full suite: 1,824 library tests passing, 0 failures, 1 ignored

## Changes Made

None -- all items verified as correct on the current dev branch. The prior
audit's BANK_WIDTH_BYTES fix (deriving from PHYSICAL_BANK_WIDTH_BITS instead
of hardcoding 128/8) is confirmed present and tested.

## Design Notes (not bugs)

1. **MemoryAccess struct scope**: The `MemoryAccess` struct in timing/memory.rs
   hardcodes 8 banks (`& 0x7` mask) but is correctly scoped to compute tiles
   only. MemTile bank operations use the parameterized `banking.rs` functions.
   If the timing model is ever extended to memtiles, `MemoryAccess.bank()`
   would need to be parameterized.

2. **check_conflicts() dead code**: The `MemoryModel::check_conflicts()` method
   (plural) is defined but never called. It has a potential double-counting
   issue with inter-cycle vs intra-cycle conflicts. Not harmful since unused.

3. **Cross-tile write latency**: Cross-tile stores are buffered via
   `NeighborMemory::pending_writes` and applied after the core step. This
   models the principle that cross-tile writes have higher latency, though
   the exact cycle at which the write becomes visible is not precisely
   modeled (it becomes visible on the next core step, not after a specific
   number of cycles).
