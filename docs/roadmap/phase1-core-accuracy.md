# Phase 1: Core Accuracy

**Goal**: Make the emulator faithful to real AIE2 hardware behavior.

**Status**: 🟢 Mostly Complete

---

## Architecture Overview

Based on analysis of [llvm-aie](https://github.com/Xilinx/llvm-aie) TableGen files, AIE2 has:

### VLIW Structure
- **8 functional slots**: `lda`, `ldb`, `alu`, `mv`, `st`, `vec`, `lng`, `nop`
- **Variable slot widths**: 16-42 bits per slot
- **Bundle sizes**: 2-byte (nop), 4-byte (standard), 6-byte (long), 16-byte (full VLIW)

### Slot Details

| Slot | Bits | Purpose |
|------|------|---------|
| lda | 21 | Load A channel |
| ldb | 16 | Load B channel |
| alu | 20 | Scalar ALU operations |
| mv | 22 | Move operations |
| st | 21 | Store operations |
| vec | 26 | Vector operations |
| lng | 42 | Long format (6-byte) |
| nop | 1 | NOP marker (artificial) |

---

## Implementation Progress

### 1.1 Instruction Decoder

| Task | Status | Notes |
|------|--------|-------|
| Bundle representation (`VliwBundle`) | ✅ Done | `src/interpreter/bundle/` |
| Slot operation types (`SlotOp`, `Operation`) | ✅ Done | 30+ operation types |
| Pattern-based decoder | ✅ Done | `src/interpreter/decode/patterns.rs` |
| TableGen-based decoder | ✅ Done | `src/interpreter/decode/tablegen_decoder.rs` |
| VLIW slot extraction | ✅ Done | 16-112 bit formats fully supported |
| Full VLIW bundle parsing | ✅ Done | All formats 16-112 bit, 128-bit rare |

**Files created**:
- `src/interpreter/bundle/mod.rs` - VliwBundle struct, disassembler
- `src/interpreter/bundle/slot.rs` - SlotIndex, SlotOp, Operation, Operand
- `src/interpreter/bundle/encoding.rs` - BundleFormat, format detection
- `src/interpreter/bundle/slot_layout.rs` - VLIW slot extraction (16-112 bit)
- `src/interpreter/decode/mod.rs` - Aie2Slot, extraction helpers
- `src/interpreter/decode/patterns.rs` - PatternDecoder
- `src/interpreter/decode/tablegen_decoder.rs` - TableGenDecoder

### 1.2 Scalar Unit

| Task | Status | Notes |
|------|--------|-------|
| GPR file (32 registers) | ✅ Done | `ScalarRegisterFile` in `state/registers.rs` |
| Pointer registers (p0-p7) | ✅ Done | `PointerRegisterFile` |
| Modifier registers (m0-m7) | ✅ Done | `ModifierRegisterFile` |
| ALU operations | ✅ Done | `ScalarAlu` in `execute/scalar.rs` |
| Condition codes | ✅ Done | `Flags` struct in `traits.rs` |
| Address generation | ✅ Done | Post-modify in memory unit |

**Implemented operations**:
- `ScalarAdd`, `ScalarSub`, `ScalarMul`
- `ScalarAnd`, `ScalarOr`, `ScalarXor`
- `ScalarShl`, `ScalarShr`, `ScalarSra`
- `ScalarMov`, `ScalarMovi`, `ScalarCmp`

### 1.3 Vector Unit

| Task | Status | Notes |
|------|--------|-------|
| Vector registers (32 × 256-bit) | ✅ Done | `VectorRegisterFile` in `state/registers.rs` |
| Accumulator registers (8 × 512-bit) | ✅ Done | `AccumulatorRegisterFile` |
| Vector ALU operations | ✅ Done | `VectorAlu` in `execute/vector.rs` |
| Shuffle/permute | ✅ Done | ShufflePattern enum + execution |
| Element types | ✅ Done | i8/u8/i16/u16/i32/u32/bf16/f32 |

**Implemented operations**:
- `VectorAdd`, `VectorSub`, `VectorMul`, `VectorMac`
- `VectorShuffle`, `VectorPack`, `VectorUnpack`
- `VectorCmp`, `VectorMin`, `VectorMax`

### 1.4 Memory System

| Task | Status | Notes |
|------|--------|-------|
| Load/store operations | ✅ Done | `MemoryUnit` in `execute/memory.rs` |
| Memory width variants | ✅ Done | Byte/HalfWord/Word/DoubleWord/QuadWord/Vector256 |
| Post-modify addressing | ✅ Done | None/Immediate/Register |
| Bank conflict detection | 🔲 TODO | |
| Timing model | 🔲 TODO | |

### 1.5 DMA Engine

| Task | Status | Notes |
|------|--------|-------|
| DMA start/wait operations | 🟡 Partial | Decoded, instant completion |
| Multi-dimensional addressing | 🔲 TODO | |
| BD chaining | 🔲 TODO | |
| Transfer simulation | 🔲 TODO | Currently instant |

### 1.6 Synchronization

| Task | Status | Notes |
|------|--------|-------|
| Lock acquire/release | ✅ Done | `ControlUnit` in `execute/control.rs` |
| Lock contention tracking | 🔲 TODO | |
| Deadlock detection | 🔲 TODO | |
| Stream switch routing | 🔲 TODO | |

### 1.7 TableGen Parser

| Task | Status | Notes |
|------|--------|-------|
| Parse slots from AIE2Slots.td | ✅ Done | 8 slots with correct bit widths |
| Parse format classes | ✅ Done | 108 format classes parsed |
| Parse instructions | ✅ Done | 135 instruction definitions |
| Extract semantics | ✅ Done | mayLoad, mayStore, Defs, Uses |
| Extract patterns | ✅ Done | 18 semantic patterns (Add, Sub, etc.) |
| Resolve encodings | ✅ Done | 70/135 instructions → concrete encodings |
| Build decoder tables | ✅ Done | `build_decoder_tables()` API |
| Real binary test | ✅ Done | 20% recognition on real ELF |

**Files created**:
- `src/tablegen/mod.rs` - Public API, `load_from_llvm_aie()`
- `src/tablegen/types.rs` - SlotDef, FormatClass, InstrDef, SemanticOp
- `src/tablegen/parser.rs` - Regex-based .td file parsing
- `src/tablegen/resolver.rs` - Compute fixed bits/masks, operand fields

**Parsing results**:
- **8 slots**: lda, ldb, alu, mv, st, vec, lng, nop (all with correct bit widths)
- **108 format classes**: Encoding patterns with field layouts
- **135 instructions**: Concrete instruction definitions
- **34 instructions** with Defs (implicit register writes)
- **6 instructions** with mayLoad
- **6 instructions** with mayStore
- **18 semantic patterns**: Add, Sub, And, Or, Xor, Shl, Sra, Srl, Br → instructions

**Resolved encodings** (70/135 instructions):
- `mv`: 23 instructions
- `alu`: 17 instructions
- `lda`: 12 instructions
- `st`: 10 instructions
- `ldb`: 5 instructions
- `lng`: 3 instructions

### 1.8 Real Binary Validation

Tested the full pipeline against a real AIE2 ELF from mlir-aie:

```
ELF: add_one_objFifo/main_core_0_2.elf
Architecture: AIE2
Entry point: 0x0000

Recognition rate: 100% (20/20 instructions)
- All bundle formats correctly detected (16/32/48/64/80/96/112/128-bit)
- Slot extraction working for 32-bit, 48-bit, and 64-bit formats
- NOPs, branches, moves, arithmetic, and lock operations all recognized
```

**Improvements made:**
- Added proper VLIW bundle format detection from low nibble
- Bundle sizes now correctly determined (was always assuming 4 bytes)
- 16-bit NOP format (`0x0001` marker) now recognized
- **Slot extraction from VLIW bundles** - new `slot_layout.rs` module
  - Extracts individual slot bits from packed bundles
  - Supports 32-bit single-slot formats (LDA, LDB, ALU, MV, ST, VEC)
  - Supports 48-bit dual-slot formats (LDA+ST, LDA+MV, LDA+ALU, LDB variants, LNG)
  - Supports 64-bit multi-slot formats (ALU+MV, ALU+VEC, LDA+LDB+ST, etc.)
- Decoder now uses slot extraction for accurate instruction recognition

**Remaining work** (areas for further improvement):
1. 80-128 bit format slot extraction (complex hierarchical layouts)
2. Improve operand extraction for specific instruction variants
3. Vector/DMA instruction semantics

---

## Module Structure

```
src/interpreter/
├── mod.rs              # Public API, re-exports
├── traits.rs           # Decoder, Executor, StateAccess traits
├── bundle/             # ✅ DONE
│   ├── mod.rs          # VliwBundle
│   ├── slot.rs         # SlotOp, Operation, Operand
│   ├── encoding.rs     # BundleFormat, detection
│   └── slot_layout.rs  # VLIW slot extraction from bundles
├── decode/             # ✅ DONE
│   ├── mod.rs          # Aie2Slot, helpers
│   ├── patterns.rs     # PatternDecoder
│   └── tablegen_decoder.rs  # TableGenDecoder
├── state/              # ✅ DONE
│   ├── mod.rs          # Module exports
│   ├── registers.rs    # All register files
│   └── context.rs      # ExecutionContext
├── execute/            # ✅ DONE
│   ├── mod.rs          # Module exports
│   ├── scalar.rs       # ScalarAlu
│   ├── vector.rs       # VectorAlu
│   ├── memory.rs       # MemoryUnit
│   ├── control.rs      # ControlUnit (branch, lock, DMA)
│   └── fast_executor.rs # FastExecutor
├── core/               # ✅ DONE
│   ├── mod.rs          # Module exports
│   └── interpreter.rs  # CoreInterpreter
└── engine/             # ✅ DONE
    ├── mod.rs          # Module exports
    └── coordinator.rs  # InterpreterEngine

src/tablegen/           # ✅ DONE
├── mod.rs              # Public API, load_from_llvm_aie()
├── types.rs            # Data structures
├── parser.rs           # Regex-based parsing
└── resolver.rs         # Encoding resolution
```

---

## Test Coverage

**Total: 277 tests passing** (272 unit + 5 doc tests)

| Module | Tests | Notes |
|--------|-------|-------|
| bundle/slot.rs | 8 | SlotIndex, ElementType, Operation |
| bundle/encoding.rs | 6 | BundleFormat, SlotMask |
| bundle/mod.rs | 8 | VliwBundle creation, disassembly |
| decode/mod.rs | 4 | Extract helpers |
| decode/patterns.rs | 8 | Pattern decoding |
| decode/tablegen_decoder.rs | 6 | TableGen-based decoding |
| traits.rs | 5 | Flags operations |
| state/registers.rs | 13 | All register files |
| state/context.rs | 10 | ExecutionContext |
| execute/scalar.rs | 10 | Scalar ALU operations |
| execute/vector.rs | 10 | Vector ALU operations |
| execute/memory.rs | 7 | Load/store operations |
| execute/control.rs | 10 | Branch, lock, DMA |
| execute/fast_executor.rs | 9 | Executor integration |
| core/interpreter.rs | 9 | CoreInterpreter |
| engine/coordinator.rs | 11 | InterpreterEngine |
| tablegen/types.rs | 6 | Data structures |
| tablegen/parser.rs | 11 | Parsing tests |
| tablegen/resolver.rs | 8 | Encoding resolution |
| tablegen/mod.rs | 5 | Integration tests |
| **Interpreter subtotal** | **~140** | |
| **TableGen subtotal** | **~36** | |
| **Legacy (emu_stub)** | **~86** | Preserved |
| **Grand total** | **~262** | All passing |

---

## Next Steps

Current status: **100% recognition rate** on test ELF binaries.

### Completed

1. **VLIW bundle slot extraction** - All 16-112 bit formats
   - 16-bit NOP, 32-bit single-slot, 48-bit dual-slot, 64-bit multi-slot
   - 80-bit (21 format variants)
   - 96-bit (20+ format variants)
   - 112-bit (8 format variants)

2. **TableGen-based decoder** - 70/135 instructions resolved
   - Pattern matching with specificity ordering
   - Operand field extraction

### Remaining Work

1. **128-bit format extraction** (rare, ~0% of test binaries)
   - I128_LDB_LDA_ST_LNG_VEC
   - I128_LDB_LDA_ST_ALU_MV_VEC

2. **Improve operand extraction** for specific instruction variants
   - Some Load operations show as `Load:??? (0x00000)`
   - Vector instruction operands need refinement

3. **Code generation** (optional optimization)
   - Generate static Rust decode tables from resolved encodings
   - Auto-generate executor stubs from semantic patterns

---

## Technical Decisions

### Why Pattern-Based Decoder First?

TableGen parsing is complex and the llvm-aie repo has generated files. Starting with pattern-based decoding:
1. Gets us running quickly
2. Handles common cases accurately
3. Provides infrastructure for TableGen integration later
4. Gracefully falls back to `Unknown` for unrecognized patterns

Now we have **both** decoders:
- `PatternDecoder` - Hand-written patterns for known instructions
- `TableGenDecoder` - Auto-generated from llvm-aie TableGen files

### Slot Mapping

The interpreter uses a simplified 7-slot model internally while the decoder understands the real 8-slot AIE2 architecture:

| AIE2 Slot | Interpreter SlotIndex |
|-----------|----------------------|
| lda, ldb | Load |
| alu | Scalar0 |
| mv | Scalar1 |
| st | Store |
| vec, lng | Vector |
| nop | Control |

---

## Resources

- **llvm-aie TableGen**: `llvm/lib/Target/AIE/` in [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie)
- **Key files**: `AIE2Slots.td`, `AIE2GenInstrFormats.td`, `AIE2GenInstrInfo.td`
- **aie-rt**: Register definitions in [Xilinx/aie-rt](https://github.com/Xilinx/aie-rt)
- **AMD Docs**: AM020 (AIE2 Architecture), AM025 (Register Reference)
- **Assessment**: [tablegen-assessment.md](tablegen-assessment.md)
