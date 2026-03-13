# Shim / MemTile / Cascade Verification Report

Audit date: 2026-03-12
Reference: aie-rt branch xlnx_rel_v2025.2 (`/home/triple/npu-work/aie-rt/driver/src/`)

## Summary

The emulator's register address decoder, BD parser, and CDO application
pipeline were compared against the authoritative aie-rt register definitions
and DMA module structures. The comparison reveals that the emulator is
currently built around **compute tile register offsets only**, with shim and
memtile variants not yet implemented.

## Items Verified as Correct

### 1. Compute Tile BD Layout

The emulator's `write_dma_bd()` at `src/device/state.rs:252` correctly
handles compute tile BDs:
- Base offset 0x1D000 matches `XAIEMLGBL_MEMORY_MODULE_DMA_BD0_0`
- BD stride 0x20 matches `AieMlTileDmaMod.IdxOffset = 0x20`
- 6 words per BD matches `XAIEML_TILEDMA_NUM_BD_WORDS = 6U`
- 16 BDs matches `AieMlTileDmaMod.NumBds = 16U`

### 2. Compute Tile DMA Channel Layout

The emulator's `write_dma_channel()` at `src/device/state.rs:280` correctly
handles compute tile DMA channels:
- Base offset 0x1DE00 matches `XAIEMLGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL`
- Channel stride 0x8 matches `AieMlTileDmaMod.ChIdxOffset = 0x8`
- 2 S2MM + 2 MM2S channels match aie-rt (`NumChannels = 2U` per direction)

### 3. Compute Tile Lock Layout

The lock offset 0x1F000 matches `XAIEMLGBL_MEMORY_MODULE_LOCK0_VALUE`.
Lock spacing of 4 bytes matches the register database.
16 locks per compute tile matches `AieMlTileLockMod.NumLocks = 16U`.

### 4. Lock Semaphore Model

The lock value range (6-bit, 0-63) and acquire/release semantics with
signed change_value match aie-rt's `_XAieMl_LockAcquire`/`_XAieMl_LockRelease`.

### 5. Stream Switch Port Types

The `PortType` enum in `src/device/stream_switch.rs` includes all necessary
types: North, South, East, West, Dma, Core, Cascade, Fifo. The Cascade type
exists but is not used in any stream switch constructor.

### 6. DMA Engine Channel Counts

- Compute: 2 S2MM + 2 MM2S = 4 total. Matches aie-rt.
- MemTile: 6 S2MM + 6 MM2S = 12 total. Matches aie-rt.

### 7. Cascade Constants

- CASCADE_STREAM_WIDTH_BITS = 512 matches AM020 cascade description.
- CASCADE_FIFO_DEPTH = 2 matches "two-deep 512-bit wide FIFO".

### 8. Accumulator Control Register Address

The aie-rt defines `XAIEMLGBL_CORE_MODULE_ACCUMULATOR_CONTROL = 0x36060`
with two 1-bit fields:
- INPUT (bit 0): 0=NORTH, 1=WEST
- OUTPUT (bit 1): 0=SOUTH, 1=EAST

The emulator's `aie2_spec.rs` does not reference this register, and
`write_core_register()` does not handle offset 0x36060. However, the
constants are defined correctly and the register offset is known.
