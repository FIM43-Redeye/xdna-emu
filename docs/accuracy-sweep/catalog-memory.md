# Memory Model Divergence Catalog

Audit date: 2026-03-12
Agent: Agent J (Memory Model)

## Summary

The memory model is in excellent shape. All memory sizes, address space
layout, CardDir routing, banking constants, and host offsets match aie-rt
exactly. One trivial derivation issue was fixed in-place. No functional
divergences were found.

---

## [MEMORY-001] BANK_WIDTH_BYTES was hardcoded instead of derived

- **Severity**: LOW
- **Our behavior**: `BANK_WIDTH_BYTES = 128 / 8` hardcoded with a TODO comment
- **aie-rt behavior**: `XAIEMLGBL_MEMORY_MODULE_DATAMEMORY_WIDTH = 128` (xaiemlgbl_params.h:7206)
- **Impact**: No functional impact (value was already correct). Would not track
  hardware changes automatically if bank width ever changed.
- **Suggested fix**: Generate from ArchModel `bank_width_bits` field
- **Fixed in-place**: YES
  - `build.rs`: Now emits `PHYSICAL_BANK_WIDTH_BITS` constant
  - `memory.rs`: Now uses `arch::compute::PHYSICAL_BANK_WIDTH_BITS / 8`
  - Tests added to validate against aie-rt value (128 bits)

---

## Items Audited With No Divergence Found

The following areas were checked and found to be fully consistent with aie-rt:

1. **Data memory sizes** (compute 64KB, memtile 512KB, shim 0)
   - Source: `AieMlTileMemMod.Size`, `AieMlMemTileMemMod.Size`
   - Our code: generated from ArchModel at build time

2. **Cardinal direction (CardDir) model**
   - Source: `_XAie_GetTargetTileLoc()` in xaie_elfloader.c:124-183
   - Our code: `MemoryQuadrant::from_address()` in timing/memory.rs
   - AIE2 `IsCheckerBoard=0` correctly forces East=Local

3. **Core data address space layout**
   - Source: `AieMlCoreMod.DataMemAddr=0x40000`, `.DataMemShift=16`
   - Our code: `arch::compute::DATA_MEM_ADDR`, `DATA_MEM_SHIFT`

4. **Program memory host offset**
   - Source: `XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY=0x20000`
   - Our code: `arch::compute::PROGRAM_MEM_HOST_OFFSET=0x20000`

5. **Data memory host offset**
   - Source: `XAIEMLGBL_MEMORY_MODULE_DATAMEMORY=0x00000000`
   - Our code: `arch::DATA_MEM_HOST_OFFSET=0`

6. **Physical banking** (8 banks compute, 16 banks memtile)
   - Source: AM020 Ch2, implied by aie-rt bank conflict events
   - Our code: generated from ArchModel physical banking

7. **Bank interleaving** (128-bit / 16-byte granularity)
   - Source: `XAIEMLGBL_MEMORY_MODULE_DATAMEMORY_WIDTH=128`
   - Our code: `addr_to_bank()` shifts right by 4 (16 bytes)

8. **Memory initialization** (zero-fill)
   - Source: `_XAieMl_PartMemZeroInit()` in xaie_device_aieml.c
   - Our code: `vec![0u8; size]` allocation

9. **Host/DDR memory** -- no aie-rt analog (emulator-only)
   - Sparse 4KB-page BTreeMap storage is correct for emulation

10. **Tile address encoding** (col_shift=25, row_shift=20)
    - Source: aie-rt `XAie_Config` RowShift/ColShift fields
    - Our code: generated from ArchModel

## Potential Future Work (not bugs)

- **Cross-tile memory access latency**: Currently modeled as 4 cycles per hop
  (from AM020). Not validated against hardware measurement. This is a timing
  refinement, not a correctness issue.

- **Memory bank conflict events**: The emulator tracks bank conflicts via
  `MemoryModel` and fires `CONFLICT_DM_BANK_N` trace events, but the exact
  arbitration policy (round-robin per AM020) is simplified to a 1-cycle stall.
  Real hardware may have more nuanced arbitration. This affects cycle-accuracy
  but not functional correctness.

- **ECC scrubbing**: aie-rt supports ECC scrubbing events and memory control
  registers. The emulator does not model ECC. No impact on functional
  correctness.
