# Interpreter

Modular AIE2 instruction set interpreter: VLIW decoding, execution, timing, and multi-core coordination.

Read this file when working on anything in `src/interpreter/`.

## Files

### Module Root

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports of all submodule types |
| `traits.rs` | Core abstractions: `Decoder`, `Executor`, `StateAccess`, `Flags` |
| `test_runner.rs` | `TestRunner` -- end-to-end kernel execution harness |

### Bundle (`bundle/`)

VLIW instruction bundle representation.

| File | Purpose |
|------|---------|
| `mod.rs` | `VliwBundle`, `Operation`, `Operand`, `SlotOp` and related types |
| `encoding.rs` | `BundleFormat` -- format class handling for 128-bit bundles |
| `slot.rs` | `SlotIndex`, `SlotMask` -- slot identification and masking |
| `slot_layout.rs` | Physical bit layout of slots within a 128-bit bundle |

### Decode (`decode/`)

TableGen-driven instruction decoder.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `decoder.rs` | `InstructionDecoder` -- O(1) lookup decoder built from TableGen data |

### State (`state/`)

Processor state: registers, flags, execution context.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `registers.rs` | `ScalarRegisterFile`, `VectorRegisterFile`, `AccumulatorRegisterFile`, `PointerRegisterFile`, `ModifierRegisterFile` |
| `context.rs` | `ExecutionContext` -- per-core state (PC, registers, flags, memory view) |

### Execute (`execute/`)

Execution units implementing instruction semantics.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `scalar.rs` | `ScalarAlu` -- GPR operations, ALU, comparisons |
| `vector.rs` | `VectorAlu` -- SIMD operations, element types, accumulators |
| `memory.rs` | `MemoryUnit` -- load/store with post-modify addressing |
| `control.rs` | `ControlUnit` -- branches, calls, loops, delay slots |
| `stream.rs` | Stream put/get instruction execution |
| `semantic.rs` | `SemanticOp`-based execution dispatch |
| `cycle_accurate.rs` | `CycleAccurateExecutor` -- full pipeline model with timing |

### Timing (`timing/`)

Cycle-accurate timing model.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `latency.rs` | `LatencyTable`, `OperationTiming` -- per-operation cycle costs |
| `hazards.rs` | `HazardDetector` -- data/structural hazard detection |
| `memory.rs` | `MemoryModel` -- memory access timing, bank conflicts |
| `sync.rs` | Synchronization timing (lock acquire/release) |
| `deadlock.rs` | Deadlock detection for lock operations |
| `barrier.rs` | Barrier synchronization timing |
| `slots.rs` | Slot-level timing constraints |
| `arbitration.rs` | Multi-core resource arbitration |

### Core (`core/`)

Per-core interpreter.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `interpreter.rs` | `CoreInterpreter` -- fetch/decode/execute loop for one core |

### Engine (`engine/`)

Multi-core coordinator.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `coordinator.rs` | `InterpreterEngine` -- steps all cores, manages global state |

## Key Types

- `VliwBundle` -- decoded 128-bit VLIW instruction (up to 7 slot operations)
- `InstructionDecoder` -- TableGen-driven decoder with O(1) per-slot lookup
- `ExecutionContext` -- all per-core state (registers, PC, flags, memory)
- `CoreInterpreter` -- single-core fetch/decode/execute loop
- `InterpreterEngine` -- multi-core coordinator (steps all cores + DMAs)
- `CycleAccurateExecutor` -- timing-aware executor with pipeline model

## Slot Architecture

AIE2 defines 8 hardware slots. The interpreter maps these to 7 functional slots (the `nop` slot is handled implicitly):

| Slot | Field | Bits | Function |
|------|-------|------|----------|
| lda | lda | 21 | Load A (pointer-based) |
| ldb | ldb | 16 | Load B (pointer-based) |
| alu | alu | 20 | Scalar ALU |
| mv | mv | 22 | Move/register transfer |
| st | st | 21 | Store |
| vec | vec | 26 | Vector ALU |
| lng | lng | 42 | Long instruction (two-slot) |
| nop | nop | 1 | NOP (implicit) |

## Execution Model

1. `CoreInterpreter` fetches a 128-bit bundle from memory at the current PC
2. `InstructionDecoder` decodes each slot into `Operation` values
3. `CycleAccurateExecutor` executes all slot operations respecting:
   - Data hazards (RAW, WAW, WAR)
   - Branch delay slots (AIE2 has 5 delay slots)
   - Memory bank conflicts
   - Lock synchronization
4. `InterpreterEngine` coordinates multiple cores per cycle

## Key Bug Fixes to Preserve

- **Branch delay slots**: AIE2 has 5 delay slots after a branch. Instructions in delay slots execute before the branch takes effect.
- **VLIW execution order**: All slot operations in a bundle read state before any writes. This prevents intra-bundle data hazards.
- **Long instruction handling**: `lng` slot spans two physical slot positions. The decoder must not try to decode the second half as a separate instruction.
