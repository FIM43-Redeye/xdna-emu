# Shim / MemTile / Cascade Verification Report

Audit date: 2026-03-12 (re-audit)
Reference: aie-rt branch xlnx_rel_v2025.2 (`/home/triple/npu-work/aie-rt/driver/src/`)

## Summary

The emulator now has fully data-driven, regdb-based BD parsing for all three
tile types (compute, memtile, shim). The register address decoder is
tile-type-aware. Cascade routing is implemented with tested FIFO state and
accumulator control register handling. The prior audit (also dated 2026-03-12)
found many CRITICAL/HIGH issues; this re-audit confirms the vast majority have
been resolved.

## Items Verified as Correct

### 1. Shim BD Parsing (8 words, regdb-driven)

`parse_shim()` in `src/device/dma/bd.rs:309` correctly implements the 8-word
shim BD layout matching `_XAieMl_ShimDmaWriteBd()` in aie-rt:

| Word | Fields | aie-rt ref | Emulator |
|------|--------|------------|----------|
| 0 | Buffer_Length (32-bit full) | `NOC_MODULE_DMA_BD0_0` | `lay.buffer_length.extract(w0)` |
| 1 | Base_Address_Low [31:2] (30 bits) | `NOC_MODULE_DMA_BD0_1` | `lay.base_address_low.extract(w1)` |
| 2 | Base_Address_High [15:0], packet fields | `NOC_MODULE_DMA_BD0_2` | `lay.base_address_high.extract(w2)` |
| 3 | Secure_Access, D0_Wrap, D0_Stepsize (20-bit) | `NOC_MODULE_DMA_BD0_3` | correct fields extracted |
| 4 | Burst_Length, D1_Wrap, D1_Stepsize (20-bit) | `NOC_MODULE_DMA_BD0_4` | correct fields extracted |
| 5 | SMID, AxCache, AxQoS, D2_Stepsize (20-bit) | `NOC_MODULE_DMA_BD0_5` | correct fields extracted |
| 6 | Iteration_Current, Iteration_Wrap, Iteration_Stepsize (20-bit) | `NOC_MODULE_DMA_BD0_6` | correct fields extracted |
| 7 | TLAST, Next_BD, Use_Next_BD, Valid_BD, lock fields | `NOC_MODULE_DMA_BD0_7` | correct fields extracted |

46-bit word address reconstruction is correct:
`addr_low | (addr_high << 30)` -- where `addr_low` is the 30-bit value
extracted from bits [31:2] (i.e., byte address >> 2), giving the word address.
aie-rt reconstructs by `(extracted << Lsb) | (high << 32)` for a byte address;
dividing by 4 gives the same word address.

Stepsize stored-as-minus-1 convention: all three parsers correctly add 1 to
extracted stepsize values, matching aie-rt's read path.

Iteration wrap stored-as-minus-1: correctly adds 1, matching aie-rt's
`IterDesc.Wrap = 1U + XAie_GetField(...)`.

### 2. MemTile BD Parsing (8 words, regdb-driven)

`parse_memtile()` in `src/device/dma/bd.rs:230` correctly implements the
8-word memtile BD layout matching `_XAieMl_MemTileDmaWriteBd()`:

- Word 0: Enable_Packet, Packet_Type, Packet_ID, OOO_BD_ID, Buffer_Length (17-bit)
- Word 1: D0_Zero_Before, Next_BD (6-bit), Use_Next_BD, Base_Address (19-bit)
- Word 2: TLAST_Suppress, D0_Wrap, D0_Stepsize (17-bit)
- Word 3: D1_Zero_Before, D1_Wrap, D1_Stepsize (17-bit)
- Word 4: Enable_Compression, D2_Zero_Before, D2_Wrap, D2_Stepsize (17-bit)
- Word 5: D2_Zero_After, D1_Zero_After, D0_Zero_After, D3_Stepsize (17-bit)
- Word 6: Iteration_Current, Iteration_Wrap, Iteration_Stepsize (17-bit)
- Word 7: Valid_BD, Lock_Rel_Value, Lock_Rel_ID (8-bit!), Lock_Acq_Enable,
  Lock_Acq_Value, Lock_Acq_ID (8-bit!)

All field widths cross-validated against `xaiemlgbl_params.h`:
- Buffer length: 17 bits (0x1FFFF) -- matches `MEM_TILE_MODULE_DMA_BD0_0_BUFFER_LENGTH_WIDTH=17`
- Base address: 19 bits (0x7FFFF) -- matches `_BASE_ADDRESS_WIDTH=19`
- Lock IDs: 8 bits (0xFF) -- matches `_LOCK_ACQ_ID_WIDTH=8`
- D3_Stepsize present -- matches 4D addressing (`NumAddrDim = 4U`)

Zero-padding fields (D0/D1/D2 before/after) correctly extracted per
`_XAieMl_MemTileDmaWriteBd()` BdWord[1], BdWord[3-5].

### 3. Compute BD Parsing (6 words, regdb-driven)

`parse_compute()` in `src/device/dma/bd.rs:158` matches `_XAieMl_DmaWriteBd()`:
- 6 words per BD, 14-bit buffer length, 14-bit base address
- No zero-padding, no D3, no AXI params
- Cross-validated by `test_compute_bd_cross_validation()` test

### 4. BD Structural Constants (all tile types)

All derived from regdb at runtime, verified in `test_regdb_layout_cross_validate()`:

| Tile Type | Base | Stride | Words | BDs |
|-----------|------|--------|-------|-----|
| Compute | 0x1D000 | 0x20 | 6 | 16 |
| MemTile | 0xA0000 | 0x20 | 8 | 48 |
| Shim | 0x1D000 | 0x20 | 8 | 16 |

Matches aie-rt: `AieMlTileDmaMod`, `AieMlMemTileDmaMod`, `AieMlShimDmaMod`.

### 5. Register Address Decoder (tile-type-aware)

`subsystem_from_offset()` in `src/device/registers.rs:411` correctly dispatches
based on `TileKind` (Compute, Mem, ShimNoc, ShimPl):

- MemTile: DMA at 0xA0000, SS at 0xB0000, Lock at 0xC0000, LockRequest at 0xD0000
- Shim: Lock at 0x14000, DMA at 0x1D000, NoC at 0x1E008, SS at 0x3F000
- Cross-tile sensitivity verified: offset 0x14000 is Timer on compute, Lock on shim

Tests: `test_subsystem_from_offset_compute_primary`, `_memtile`, `_shim`,
`_tile_sensitivity`.

### 6. CDO Write Dispatch (tile-type-aware)

`write_register()` in `src/device/state.rs:230` correctly routes writes based
on the subsystem classification:

- DMA sub-dispatch distinguishes Mem (memtile BD vs channel), ShimNoc/ShimPl
  (shim channel vs BD), and Compute (BD vs channel)
- MemTile stream switch gets its own handler (`write_memtile_stream_switch`)
- Lock writes distinguish MemTile (base 0xC0000), Compute (base 0x1F000),
  and Shim (noted as not yet wired to lock state)

### 7. Shim DMA Channel Registers

`write_shim_dma_channel()` at `src/device/state.rs:816` uses the correct
shim-specific base and stride from regdb (`shim_channel_base = 0x1D200`,
`shim_channel_stride`), not the compute channel base (0x1DE00).

### 8. Shim Mux/Demux Configuration (regdb-driven)

`ShimMuxLayout::from_regdb()` in `src/device/regdb.rs:512` correctly:
- Parses Mux_Config register (offset 0x1F000) for fields South2, South3, South6, South7
- Parses Demux_Config register (offset 0x1F004) for fields South2, South3, South4, South5
- Maps SouthN to switchbox port_index = N + 2
- Select values: 0=South/PL, 1=DMA, 2=NoC

aie-rt reference: `AieMlShimMuxConfig[]` and `AieMlShimDeMuxConfig[]` in
`global/xaiemlgbl_reginit.c:2244-2259`.

Tile-local effect handler dispatches Mux_Config and Demux_Config writes to
`parse_shim_mux_config()` and `parse_shim_demux_config()`.

### 9. Cascade State and FIFO

Tile struct has cascade fields:
- `cascade_input: VecDeque<[u64; 6]>` -- 384-bit SCD FIFO
- `cascade_output: VecDeque<[u64; 6]>` -- 384-bit MCD FIFO
- `cascade_input_dir: u8` -- 0=North, 1=West
- `cascade_output_dir: u8` -- 0=South, 1=East

Push/pop/has helpers implemented and tested.

### 10. Cascade Accumulator Control Register (0x36060)

`apply_tile_local_effects()` in `src/device/state.rs:1662` handles offset
`reg_layout.cascade_config_offset` (verified = 0x36060) for compute tiles:
- Bit 0: cascade input direction
- Bit 1: cascade output direction

Matches `XAIEMLGBL_CORE_MODULE_ACCUMULATOR_CONTROL` in aie-rt.
Tested: `test_cascade_register_write`, `test_cascade_register_ignored_for_non_compute`.

### 11. Cascade Routing

`route_cascade()` in `src/device/array.rs:826` implements two-phase
collect-then-apply cascade data movement:
- Only compute tiles participate
- Output dir 0=South (row-1), 1=East (col+1)
- Destination input dir must match (0=North from South output, 1=West from East)
- Backpressure: skips if destination FIFO non-empty
- Tested: `test_cascade_route_south`, `test_cascade_route_east`,
  `test_cascade_backpressure`, `test_cascade_direction_mismatch`

### 12. MemTile Lock State

MemTile lock writes use memtile-specific base (0xC0000) and stride, correctly
distinguished from compute lock base (0x1F000). MemTile lock count = 64
(from arch constant), matching aie-rt.

### 13. Sign Extension for Lock Values

`sign_extend_7bit()` correctly extends 7-bit lock values to i8.
`sign_extend_lock_value()` derives field width from regdb (6 bits for AIE2).

### 14. BD-to-BdConfig Conversion

`to_bd_config()` correctly converts:
- Word addresses to byte addresses (multiply by 4)
- Stored-minus-1 stepsizes to actual values (already done in parse)
- Iteration wrap/stepsize back to stored-minus-1 convention for IterationConfig
- Lock enable/value gating: acquire only if enable bit set, release only if
  value is non-zero
- Zero padding propagation with `validate_padding()` per aie-rt

Tested: `test_to_bd_config_full`, `test_to_bd_config_no_locks_no_chain`,
`test_to_bd_config_simple_1d`, `test_to_bd_config_shim_round_trip`.
