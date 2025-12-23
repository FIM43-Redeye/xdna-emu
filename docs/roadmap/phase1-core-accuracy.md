# Phase 1: Core Accuracy

**Goal**: Make the emulator faithful to real AIE2 hardware behavior.

**Status**: 🟡 In Progress

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
| TableGen parser infrastructure | 🔲 Planned | For automated decode table generation |
| Full VLIW bundle parsing | 🟡 Partial | 4-byte works, 16-byte needs refinement |

**Files created**:
- `src/interpreter/bundle/mod.rs` - VliwBundle struct, disassembler
- `src/interpreter/bundle/slot.rs` - SlotIndex, SlotOp, Operation, Operand
- `src/interpreter/bundle/encoding.rs` - BundleFormat, format detection
- `src/interpreter/decode/mod.rs` - Aie2Slot, extraction helpers
- `src/interpreter/decode/patterns.rs` - PatternDecoder

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

---

## Module Structure

```
src/interpreter/
├── mod.rs              # Public API, re-exports
├── traits.rs           # Decoder, Executor, StateAccess traits
├── bundle/             # ✅ DONE
│   ├── mod.rs          # VliwBundle
│   ├── slot.rs         # SlotOp, Operation, Operand
│   └── encoding.rs     # BundleFormat, detection
├── decode/             # ✅ DONE
│   ├── mod.rs          # Aie2Slot, helpers
│   └── patterns.rs     # PatternDecoder
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
```

---

## Test Coverage

| Module | Tests | Notes |
|--------|-------|-------|
| bundle/slot.rs | 8 | SlotIndex, ElementType, Operation |
| bundle/encoding.rs | 6 | BundleFormat, SlotMask |
| bundle/mod.rs | 8 | VliwBundle creation, disassembly |
| decode/mod.rs | 4 | Extract helpers |
| decode/patterns.rs | 8 | Pattern decoding |
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
| **Total new** | **128** | |
| **Legacy (emu_stub)** | **86** | Preserved |
| **Grand total** | **224** | All passing |

---

## Technical Decisions

### Why Pattern-Based Decoder First?

TableGen parsing is complex and the llvm-aie repo has generated files. Starting with pattern-based decoding:
1. Gets us running quickly
2. Handles common cases accurately
3. Provides infrastructure for TableGen integration later
4. Gracefully falls back to `Unknown` for unrecognized patterns

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

## Next Steps

1. **State module** - Full register file implementations
2. **Execute module** - Actual operation execution
3. **CoreInterpreter** - Per-core execution loop
4. **InterpreterEngine** - Multi-core coordination
5. **GUI integration** - Wire new interpreter to visual layerf

---

## Resources

- **llvm-aie TableGen**: `llvm/lib/Target/AIE/` in [Xilinx/llvm-aie](https://github.com/Xilinx/llvm-aie)
- **Key files**: `AIE2Slots.td`, `AIE2InstrFormats.td`, `AIE2InstrInfo.td`
- **aie-rt**: Register definitions in [Xilinx/aie-rt](https://github.com/Xilinx/aie-rt)
- **AMD Docs**: AM020 (AIE2 Architecture), AM025 (Register Reference)
