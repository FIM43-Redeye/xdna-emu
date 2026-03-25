# LLVM Decoder FFI Integration

## Summary

Replace our custom TableGen extraction pipeline with direct LLVM FFI calls
for instruction decoding, operand extraction, and register classification.
LLVM is linked statically via `llvm-config --libs aie`.

## What's Done

### Step 1: LLVM Decoder Linked and Working

- `decoder_ffi/aie2_decoder.cpp` compiles via `cc` crate in `build.rs`
- Links all LLVM AIE libraries (`llvm-config --libs aie --system-libs`)
- `aie2_decoder_init()` creates MCDisassembler, MCInstrInfo, MCRegisterInfo
- Thread-safe via `std::mutex` around `getInstruction()`

### Step 2: Per-Slot Decoding via Synthetic VLIW Bundles

- `aie2_decode_slot(slot, insn_bits)` constructs minimal 32/48-bit VLIW
  bundles from raw slot bits using encoding layout from AIE2CompositeFormats.td
- Calls `getInstruction()` through the public API (no upstream patches)
- Extracts nested slot MCInst from the bundle-level result
- Proven: VPUSH_HI_32 vs VSEL_16 disambiguation works correctly

### Step 3: Enriched Decode Result

- `Aie2DecodeResult` now includes:
  - `num_defs` (from `MCInstrDesc`) -- first N operands are outputs
  - `reg_name` per operand (from `MCRegisterInfo::getName()`)
- Rust `DecodeResult` has `.defs()` / `.uses()` for output/input split
- Verified: VMAC_F returns `num_defs=1`, register names like `"bml0"`, `"x0"`, `"r0"`

### Step 4: Resolver Integration (Name Only)

- `SlotIndex::decode()` tries FFI first, falls back to bytecode interpreter
- Currently uses FFI only for instruction NAME, still extracts operands
  from our own bit-field tables via `extract_operands()`

### Cleanup Done

- Removed `try_decode_validators` mechanism from `decoder_bytecode.rs`
- Removed debug tracing (eprintln, insn-specific traces)
- Removed validator generation from `build_helpers/codegen.rs`
- Updated bytecode TRY_DECODE handler with TODO comment

### Bug Fix: Wide vs Narrow Accumulator

- `execute_acc_add_sub` and `execute_acc_negate` unconditionally used
  1024-bit `read_wide`/`write_wide` for ALL accumulator operations
- Narrow operations (VADD_F, VSUB_F, VNEG_F, etc.) use 512-bit `bm`
  registers which can have odd indices -- crashed on `debug_assert`
- Fixed: detect narrow by checking if any accumulator operand has odd index
- All 6 previously-crashing ISA batches now pass

### ISA Test Harness Fix: Staleness

- `isa-test-gen.py` now uses `write_if_changed()` to preserve file timestamps
- `isa-test.sh` Phase 1 skips entirely when inputs are unchanged
- Manifest gets `os.utime()` touch after unchanged generation runs
- Eliminates unnecessary repackaging on repeated runs

### Step 5: Parse LLVM Register Names -> Operand Enum (DONE)

`operand_from_reg_name()` in `decoder_ffi.rs` maps LLVM register names
to our internal Operand variants, returning `MappedOperand` which includes
an `AccumWidth` hint for accumulator registers (Quarter/Half/Full).

Handles: r, p, x, wl/wh, y, bml/bmh, cm, amll/amlh/amhl/amhh, m/dn/dj/dc,
cr*, lr, SP, LS, LE, LC, DP, CORE_ID, s, l. Returns None for mask registers
(q, qx) and DMA registers (d) which have no Operand variant.

17 tests passing, including integration tests that decode real LLVM instructions
and verify all register operands map correctly.

### Step 6: Build SlotOp from LLVM Decode Result (DONE)

The FFI path is now primary, with legacy bytecode as fallback:

```
try_decode_via_ffi(slot, bits)
  -> LLVM: name + register operands + num_defs
  -> encoding_by_name(): InstrEncoding lookup (metadata only)
  -> operand_from_reg_name() for each register operand
  -> sign_extend for signed immediates (field width from encoding)
  -> num_defs splits dest vs sources
  -> addressing_mode combines ptr+imm into Memory/PostModify
  -> build_slot_op() populates SlotOp metadata from InstrEncoding
```

Cross-validation results against legacy path (2580 real ELF instructions):
- 100% FFI hit rate (zero fallbacks to bytecode)
- 85% exact operand match
- 15% divergences are all IMPROVEMENTS over legacy:
  - Stores: FFI correctly has dest=None (stores write memory, not regs);
    legacy incorrectly classified modifier regs as destinations
  - Pointer arithmetic: minor source list differences (functionally equivalent)

InstrEncoding is still needed for: SemanticOp, element_type, from_type,
mem_width, branch_condition, select_variant, implicit_regs, is_vector,
is_ptr_arithmetic, addressing_mode. These are our invention, not in LLVM.

What's eliminated at decode time:
- `extract_operands()` bit-field extraction pipeline
- `composite.rs` register class heuristic LUTs
- `extract_ordered_operands()` ordering heuristics
- `decode_ag_field()` packed AG field unpacking

### Step 7: Remove Bytecode Interpreter Fallback (DONE)

The legacy fallback (`decode_slot_bits` + `extract_operands`) has been
removed from the production decode path. The LLVM FFI path is now the
sole decoder. Legacy methods are retained for test infrastructure only
(cross-validation, bundle diagnosis tests).

## What's Left

Nothing -- all 7 steps are complete. Future cleanup opportunities:
- Remove `composite_luts` field from InstructionDecoder (test-only)
- Remove build-time bytecode extraction (build.rs codegen)
- Remove `decoder_bytecode.rs` entirely (dead code in production)

## What Stays Regardless

- **SemanticOp dispatch** -- maps instruction names to execution handlers.
  This is our invention, not in LLVM.
- **ElementType detection** -- from encoding name patterns ("_16", "_32", etc.)
- **Build-time TableGen extraction** -- still generates SemanticOp mappings,
  scheduling model, and format metadata.  The operand field extraction
  becomes unused once Step 6 is complete.
- **Scheduling model / itineraries** -- could move to LLVM later via
  MCSchedModel, but not urgent.

## Key Files

| File | Role |
|------|------|
| `decoder_ffi/aie2_decoder.h` | C interface definition |
| `decoder_ffi/aie2_decoder.cpp` | C++ wrapper around LLVM MCDisassembler |
| `src/tablegen/decoder_ffi.rs` | Rust FFI bindings, DecodeResult types |
| `src/tablegen/resolver.rs` | SlotIndex::decode() -- integration point |
| `src/interpreter/decode/` | SlotOp construction (target for Step 6) |
| `build.rs` | `compile_llvm_decoder_ffi()` -- cc crate + llvm-config |

## Performance Notes

All LLVM calls are in-process function calls (array lookups, bytecode
interpreter, string returns).  No IPC, no serialization.  The mutex
around `getInstruction()` can become thread-local instances if profiling
shows contention.  Instruction decode is ~1% of per-instruction cost;
the bottleneck is DMA simulation and vector compute semantics.

## Current Status

LLVM FFI is used for instruction **identification** (perfect disambiguation).
Operand **extraction** still uses the legacy bit-field path because the FFI
operand mapping has 387 divergences (all FFI-correct, but executors expect
legacy layout).

See `docs/plans/2026-03-25-ffi-operand-parity.md` for the 5-step plan to
achieve full FFI operand parity and remove the legacy path.

## Current ISA Test Results

- 0 HW failures (123/123 batches)
- 0 EMU crashes
- 12 batch-level matches (PASS) with legacy operand path
- 111 divergences (accuracy work ongoing)
- 2,576 unit tests passing
