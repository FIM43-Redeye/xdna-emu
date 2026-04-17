# Full Pipeline Review -- 2026-03-01

Cross-subsystem audit by five parallel Explore agents covering decode,
execution, DMA/locks/streams, device model, and parsers. Findings
deduplicated, cross-checked, and false positives removed.

## Status Key

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` False positive / won't fix
- `[?]` Needs verification against llvm-aie

---

## Positive Findings

Before the issues: the audit found several areas in excellent shape.

- **BD parsing**: Fully data-driven from AM025 JSON regdb. Zero hardcoded
  bit positions. Cross-validated by tests.
- **Register offsets**: All from regdb, generated at build time by build.rs.
- **Lock semantics**: Match aie-rt acquire/release/acquire_equal behavior.
- **Device topology**: Loaded from mlir-aie device model JSON. Validated at
  startup.
- **Stream switch port types**: Generated at build time from AM025 JSON.
- **DMA addressing**: Full 1D/2D/3D/4D support. Zero-padding implemented.
- **Instruction decoding**: Fully TableGen-driven. SemanticOp inference
  covers ~95% of instructions.
- **Latency model**: Cross-validated against TableGen itinerary data.

---

## P0 -- Correctness

### 1. BSS segments not zero-initialized in ELF loader
- **File**: `src/parser/elf.rs:196`
- **Problem**: Filter `ph.p_filesz > 0` skips PT_LOAD segments with zero
  file size (BSS). These should allocate memsz bytes and zero-fill them.
  Stale memory from prior runs leaks into uninitialized data.
- **Fix**: For segments where `p_memsz > p_filesz`, zero-fill the gap.
- **Status**: `[ ]`

### 2. vector_pack() ignores second operand
- **File**: `src/interpreter/execute/vector.rs:1457`
- **Problem**: `_b` parameter unused. Comment explains 256-bit register
  model only packs the first source. For full 512-bit pack (combining
  two 256-bit inputs into one), the second operand is needed.
- **Impact**: Pack operations that need both inputs produce incomplete
  results. Likely masked by current tests using single-input packs.
- **Fix**: Either implement dual-input pack or document limitation
  clearly with an assertion when `_b != zero`.
- **Status**: `[ ]`

---

## P1 -- Accuracy Gaps

### 3. Missing SemanticOp mnemonic inference handlers
- **File**: `src/tablegen/resolver.rs` (infer_semantic_from_mnemonic)
- **Problem**: Several SemanticOp variants defined in types.rs have no
  corresponding mnemonic pattern in the inference function:
  - `PointerMov` -- no "pmov" handler
  - `Accumulate` -- no "vacc" handler (vmac catches vmac* but vacc is
    a distinct accumulator-add without multiply)
  - `Cmp` -- no "cmp" handler (flag-setting compare, no dest register)
  - `Bswap` -- excluded from branch check but no dedicated handler
- **Needs verification** (may not exist as AIE2 mnemonics):
  - `Rotl`/`Rotr` -- no "rotl"/"rotr" handler
  - `Cttz` -- no "ctz" handler
  - `Ctpop` -- no "popcount" handler
- **Impact**: Instructions with these mnemonics get semantic=None, falling
  through to the generic path. May work if structural inference catches
  them, but creates a blind spot.
- **Fix**: Check llvm-aie AIE2InstrInfo.td for actual mnemonics, add
  handlers for any that exist.
- **Status**: `[?]`

### 4. vmul catches vmul.dense (MatMul misclassification)
- **File**: `src/tablegen/resolver.rs:1374`
- **Problem**: `lower.starts_with("vmul")` returns `SemanticOp::Mul` for
  ALL vmul* variants including any dense matrix multiply forms. If
  "vmul.dense" or "vmul_dense" exists as a mnemonic, it maps to
  element-wise multiply instead of MatMul.
- **Note**: Most matmul goes through vmac* (correctly handled). This only
  matters if vmul-prefixed dense matmul mnemonics exist.
- **Fix**: Check llvm-aie for vmul.dense; if it exists, add a check
  before the generic vmul handler.
- **Status**: `[?]`

### 5. vpush mapped to Ups instead of Accumulate
- **File**: `src/tablegen/resolver.rs:1528`
- **Problem**: `vpush` maps to `SemanticOp::Ups` but may be semantically
  an accumulator push (load acc from vector), not an upshift.
- **Fix**: Verify vpush semantics in llvm-aie; if it's acc push, change
  to Accumulate or a dedicated variant.
- **Status**: `[?]`

### 6. MatMul tile geometry hardcoded per element type
- **File**: `src/interpreter/execute/vector_matmul.rs:223,280,335,383`
- **Problem**: `TileGeometry { rows: 4, inner: 8, cols: 4 }` etc. are
  hardcoded per function body. Real hardware selects geometry via the
  permute_mode field in the instruction config word.
- **Impact**: Only the default geometry per type works. Non-default
  permutation modes produce wrong results.
- **Fix**: Extract tile geometry from instruction's permute mode. Use
  aietools `mulmac.py` as reference for geometry-mode mapping.
- **Status**: `[ ]`

### 7. AG field addressing mode heuristic
- **File**: `src/interpreter/decode/decoder.rs:559-598`
- **Problem**: Address generation field mode bits extracted via hardcoded
  bit patterns (0b1010, 0b0110, 0b0010) with fallback heuristic. These
  should come from the ISA encoding specification.
- **Related**: AG immediate scale factors (*4 vs *1) also hardcoded
  at lines 517-521, 530-536, 568-574.
- **Fix**: Extract addressing mode encoding from llvm-aie AG field
  definitions. The ISB format template may specify these.
- **Status**: `[ ]`

### 8. Composite encoder widths not validated against TableGen
- **File**: `src/interpreter/decode/composite.rs:88-102`
- **Problem**: LUT widths (7 bits, 6 bits, etc.) hardcoded in
  `CompositeDecoder::new()`. No assertion that parsed field.width
  matches the hardcoded LUT width.
- **Impact**: If a composite field is wider/narrower than expected,
  upper bits silently truncated or LUT panic.
- **Fix**: Add `debug_assert_eq!(field.width, lut.width())` at each
  decode call site.
- **Status**: `[ ]`

### 9. DMA lock handoff between channels
- **File**: `src/device/dma/engine.rs`
- **Problem**: S2MM releasing -> MM2S acquiring same lock sometimes
  fails. Active investigation with five hypotheses.
- **Status**: `[~]` (see dma-lock-release-investigation.md)

### 10. Token-based DMA sync unimplemented
- **File**: `src/device/dma/engine.rs:109`
- **Problem**: `TaskCompleteToken` struct defined but never emitted when
  `Enable_Token_Issue` BD field is set. Required for interrupt-free
  host-device DMA completion.
- **Status**: `[ ]`

### 11. Packet switching in stream switch minimal
- **File**: `src/device/stream_switch.rs`
- **Problem**: PacketSlot struct and MasterPacketConfig defined but not
  integrated into routing logic. Circuit switching works; packet
  switching does not.
- **Impact**: Multicast and dynamic packet-based routing not functional.
- **Status**: `[ ]`

### 12. Cross-tile MemTile lock access incomplete
- **File**: `src/device/dma/engine.rs:51-68`
- **Problem**: LockTarget enum supports West/East but routing code only
  handles own-tile. MemTile 192-entry lock space (0-63 West, 64-127
  Own, 128-191 East) partially implemented.
- **Status**: `[ ]`

### 13. Stream write/read backpressure not modeled
- **File**: `src/interpreter/execute/stream.rs:95-99`
- **Problem**: `blocking` flag ignored. Stream FIFOs unbounded (VecDeque).
  Blocking writes never stall.
- **Status**: `[ ]`

### 14. MAC permutation modes not implemented
- **File**: `src/interpreter/execute/vector_permute.rs:42-43`
- **Problem**: MAC permutation engine is stub. Matmul operations with
  non-trivial permutation modes produce wrong data routing.
- **Status**: `[ ]`

### 15. Hazard detection disabled
- **File**: `src/interpreter/execute/cycle_accurate.rs:269-280`
- **Problem**: RAW/WAW stalls computed but never applied. Executor trusts
  compiler scheduling. Intentional for now but means "cycle-accurate"
  timing is approximate.
- **Status**: `[-]` (intentional, LLVM scheduler resolves hazards)

---

## P2 -- Maintainability / Hardcoding

### 16. stream_io.rs port mappings duplicate generated data
- **File**: `src/device/dma/stream_io.rs:72-277`
- **Problem**: Compute/MemTile/Shim port type arrays hardcoded as Rust
  constants, duplicating the AM025-generated arrays in aie2_spec.rs.
  Can drift independently.
- **Fix**: Import from generated arrays or add cross-validation.
- **Status**: `[ ]`

### 17. record_memory_access() always uses addr=0
- **File**: `src/interpreter/execute/cycle_accurate.rs:182`
- **Problem**: Memory bank conflict stats receive placeholder addr=0
  instead of actual computed address. Stats are invalid.
- **Fix**: Pass ExecutionContext to resolve actual address.
- **Status**: `[ ]`

### 18. Cross-tile load latency hardcoded
- **File**: `src/interpreter/execute/memory.rs:31-32`
- **Problem**: `CROSS_TILE_LATENCY = 4` cycles per hop, not validated
  against any toolchain source.
- **Status**: `[ ]`

### 19. Cascade FIFO depth hardcoded to 1
- **File**: `src/interpreter/execute/cascade.rs:116`
- **Problem**: Should be a named constant from device model, not a magic
  number in a comparison.
- **Status**: `[ ]`

### 20. Modifier register prefix mapping duplicated 3x
- **File**: `src/interpreter/decode/composite.rs:144-154,186-196,222-232`
- **Problem**: Same prefix→modifier-class logic in decode_lda_scl,
  decode_mv_scl_src, decode_lda_cg. DRY violation.
- **Fix**: Extract to shared helper function.
- **Status**: `[ ]`

### 21. DMA timing constants hardcoded
- **File**: `src/device/aie2_spec.rs:133-149`
- **Problem**: BD_SETUP_CYCLES, CHANNEL_START_CYCLES, etc. are correct
  per AM020 but not machine-readable from any toolchain source. Will
  need updating if AIE2P differs.
- **Status**: `[-]` (no machine-readable source exists)

### 22. CDO opcodes: 35/43 unimplemented
- **File**: `src/parser/cdo.rs`
- **Problem**: Only 8 opcodes implemented (Write, MaskWrite, Write64,
  MaskWrite64, DmaWrite, MaskPoll, Nop, Marker). Most unused opcodes
  are mlir-aie-irrelevant (PM commands, Sync, Proc), but some may be
  needed for advanced binaries.
- **Status**: `[-]` (only implement as needed)

### 23. Vector shuffle returns identity for unrecognized pattern
- **File**: `src/interpreter/execute/vector.rs:1401-1456`
- **Problem**: Unrecognized shuffle patterns silently produce identity
  output. Should at minimum log a warning.
- **Status**: `[ ]`

---

## Summary

| Priority | Count | Key themes |
|----------|-------|------------|
| P0 | 2 | BSS zero-fill, vector_pack second operand |
| P1 | 13 | Mnemonic gaps, matmul geometry, AG heuristic, DMA sync, packet routing |
| P2 | 8 | Port duplication, timing hardcoding, DRY violations |
| Verified OK | 8+ | BD parsing, regdb, locks, device model, latencies |
| False positives | 3 | "ELF not loaded" (is loaded), several agent overreaches |
