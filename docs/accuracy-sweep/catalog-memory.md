# Memory Model Divergence Catalog

Audit date: 2026-03-12 (re-run)
Agent: Agent J (Memory Model)

## Summary

The memory model is fully correct. All memory sizes, address space layout,
CardDir routing, banking constants, and host offsets match aie-rt exactly.
No functional divergences were found. No code changes required.

---

## Items Audited With No Divergence Found

The following areas were checked against aie-rt reference sources and found
to be fully consistent:

1. **Data memory sizes** (compute 64KB, memtile 512KB, shim 0)
   - Source: `AieMlTileMemMod.Size=0x10000`, `AieMlMemTileMemMod.Size=0x80000`
   - Our code: generated from ArchModel at build time via `xdna-archspec`
   - Verified: `arch::compute::MEMORY_SIZE=65536`, `arch::memtile::MEMORY_SIZE=524288`

2. **Cardinal direction (CardDir) model**
   - Source: `_XAie_GetTargetTileLoc()` in xaie_elfloader.c:124-183
   - Our code: `MemoryQuadrant::from_address()` in timing/memory.rs
   - AIE2 `IsCheckerBoard=0` correctly forces East=Local (RowParity forced to 1)
   - All four directions (S=4, W=5, N=6, E=7) correctly mapped

3. **Core data address space layout**
   - Source: `AieMlCoreMod.DataMemAddr=0x40000`, `.DataMemShift=16`
   - Our code: `arch::compute::DATA_MEM_ADDR=0x40000`, `DATA_MEM_SHIFT=16`
   - OFFSET_MASK = 0xFFFF correctly extracts 16-bit local offset

4. **Cross-tile memory access resolution**
   - Source: `_XAie_GetTargetTileLoc()` neighbor tile location computation
   - Our code: `NeighborMemory::neighbor_coords()` in execute/memory.rs
   - Out-of-bounds neighbors return None (reads as zero), matching aie-rt error handling

5. **Program memory host offset**
   - Source: `XAIEMLGBL_CORE_MODULE_PROGRAM_MEMORY=0x20000`
   - Our code: `arch::compute::PROGRAM_MEM_HOST_OFFSET=0x20000`

6. **Data memory host offset**
   - Source: `XAIEMLGBL_MEMORY_MODULE_DATAMEMORY=0x0` and `MEM_TILE_MODULE_DATAMEMORY=0x0`
   - Our code: `arch::DATA_MEM_HOST_OFFSET=0`

7. **Physical banking** (8 banks compute, 16 banks memtile)
   - Source: AM020 Ch2, confirmed by aie-rt bank conflict event register names
   - Our code: generated from ArchModel physical banking
   - Structural invariant `num_banks * bank_size == total_size` enforced

8. **Bank width** (128 bits = 16 bytes)
   - Source: `XAIEMLGBL_MEMORY_MODULE_DATAMEMORY_WIDTH=128` (params.h:7206)
   - Our code: `arch::compute::PHYSICAL_BANK_WIDTH_BITS=128` (generated from ArchModel)
   - `BANK_WIDTH_BYTES` derived as `PHYSICAL_BANK_WIDTH_BITS / 8` (not hardcoded)

9. **Bank interleaving** (128-bit / 16-byte granularity)
   - `addr_to_bank()` uses `(addr >> 4) % num_banks` -- correctly interleaves
   - `banks_for_access()` returns bitmask covering all banks touched by a wide access
   - Both functions parameterized by `num_banks` for compute and memtile use

10. **Memory initialization** (zero-fill)
    - Source: `_XAieMl_PartMemZeroInit()` in xaie_device_aieml.c
    - Our code: `vec![0u8; size]` allocation

11. **Host/DDR memory** -- no aie-rt analog (emulator-only)
    - Sparse 4KB-page BTreeMap storage is correct for emulation
    - Cross-page reads/writes handled correctly

12. **Tile address encoding** (col_shift=25, row_shift=20)
    - Source: aie-rt `XAie_Config` RowShift/ColShift fields
    - Our code: generated from ArchModel

13. **DMA address wrapping** for memtile 0x80000 offsets
    - Source: aie-rt `MemMod->Size` bounds check (rejects out of range)
    - Our code: `(addr as usize) % mem_size` wraps to valid range
    - Functionally equivalent for valid programs

14. **AIE1 vs AIE2 architecture distinction**
    - AIE1: IsCheckerBoard=1, DataMemAddr=0x20000, DataMemSize=32KB, DataMemShift=15
    - AIE2: IsCheckerBoard=0, DataMemAddr=0x40000, DataMemSize=64KB, DataMemShift=16
    - Emulator correctly targets AIE2 parameters via build-time code generation

---

## Design Observations (not divergences)

### MEMORY-NOTE-001: MemoryAccess struct scoped to compute tiles

The `MemoryAccess` struct in `timing/memory.rs` hardcodes 8 banks via `& 0x7`
mask in `bank()`. This is correct because the struct is exclusively used in
the compute tile timing model (where `NUM_BANKS=8`). The production bank
conflict path uses the parameterized `banking.rs` functions.

If the timing model were extended to memtiles, `MemoryAccess` would need
to accept `num_banks` as a parameter.

### MEMORY-NOTE-002: check_conflicts() is dead code

`MemoryModel::check_conflicts()` (plural) is defined but never called from
production code. It has a potential double-counting issue where inter-cycle
and intra-cycle bank conflicts could be summed. No impact since unused.

### MEMORY-NOTE-003: Cross-tile write visibility timing

Cross-tile stores are buffered and applied after the core step completes.
This is functionally correct (writes become visible eventually) but does
not model the exact cycle at which the write becomes visible on the remote
tile. For cycle-accuracy, the write should appear after
`BASE_LATENCY + CROSS_TILE_LATENCY` cycles. This is a timing refinement,
not a correctness issue.

---

## Potential Future Work (not bugs)

- **Cross-tile memory access latency measurement**: The 4-cycle-per-hop
  routing latency comes from AM020 prose. It has not been validated against
  hardware measurement. This is a timing refinement.

- **Bank conflict arbitration policy**: The emulator models a 1-cycle stall
  on conflict. Real hardware uses round-robin arbitration per AM020. The
  difference affects cycle-accuracy but not functional correctness.

- **ECC scrubbing**: aie-rt supports ECC scrubbing events and registers.
  The emulator does not model ECC. No impact on functional correctness.
