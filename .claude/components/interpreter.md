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
| `loader.rs` | Loads decoder tables from generated artifacts at startup |
| `composite.rs` | Composite-instruction handling (multi-bundle ops) |
| `crossref.rs` | Cross-references decoder output against expected operand layouts |
| `operand_extraction.rs` | Operand extraction helpers (split fields, immediate handling) |
| `register_map.rs` | Maps decoded register identifiers to register-file slots |
| `slot_builder.rs` | Builds `SlotOp` values from raw decoder output |

### State (`state/`)

Processor state: registers, flags, execution context.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `registers.rs` | `ScalarRegisterFile`, `VectorRegisterFile`, `AccumulatorRegisterFile`, `PointerRegisterFile`, `ModifierRegisterFile` |
| `context.rs` | `ExecutionContext` -- per-core state (PC, registers, flags, memory view) |
| `timing_context.rs` | `TimingContext` -- per-core hazard / stall / latency state |
| `event_trace.rs` | Per-core event log used by the trace pipeline |

### Execute (`execute/`)

Execution units implementing instruction semantics.

| File | Purpose |
|------|---------|
| `mod.rs` | Module re-exports |
| `vector_dispatch.rs` | `VectorAlu` -- top-level dispatch, accumulator ops, wide bridges |
| `vector_helpers.rs` | Shared vector helpers used across dispatch paths |
| `vector_arith.rs` | Arithmetic ops (add, sub, mul, min, max, shifts, negate, abs, floor) |
| `vector_compare.rs` | Comparison ops (eq, ge, lt, eqz, select) |
| `vector_misc.rs` | Misc ops (shuffle, broadcast, extract, insert, align, bitwise) |
| `vector_matmul/` | Dense and sparse matrix multiply (config-driven, all type combos) |
| `vmac_hw.rs` | Hardware-faithful sparse vmac pipeline (oracle-verified crossbar routing) |
| `vector_permute.rs` | VSHUFFLE routing tables (40+ modes) |
| `vector_srs.rs` | Shift-Round-Saturate (10 rounding modes) |
| `vector_ups.rs` | UPS widening conversion |
| `vector_pack.rs` | Pack/unpack operations |
| `vector_convert.rs` | Type conversion (int<->float, narrow<->wide) |
| `vector_float.rs` | Float32/BFloat16 compute helpers (NaN, FTZ, PSA) |
| `vector_config.rs` | MAC configuration word parsing and geometry tables |
| `vector_semantic.rs` | SemanticOp-based vector dispatch |
| `vector_validate.rs` | Cross-checks vector outputs against golden tables |
| `memory/` | Load/store with post-modify addressing (split into submodules) |
| `control.rs` | `ControlUnit` -- branches, calls, loops, delay slots |
| `stream.rs` | Stream put/get instruction execution |
| `cascade.rs` | Cascade stream operations |
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

Note: there is no `barrier.rs` companion file outside `timing/`; the
old `interpreter::traits` `Flags` abstraction has been folded into the
register-file types and `TimingContext`.

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

- `VliwBundle` -- decoded 128-bit VLIW instruction (up to 8 slot operations)
- `InstructionDecoder` -- TableGen-driven decoder with O(1) per-slot lookup
- `ExecutionContext` -- all per-core state (registers, PC, flags, memory)
- `CoreInterpreter` -- single-core fetch/decode/execute loop
- `InterpreterEngine` -- multi-core coordinator (steps all cores + DMAs)
- `CycleAccurateExecutor` -- timing-aware executor with pipeline model

## Slot Architecture

AIE2 defines 8 hardware slots. The interpreter's `SlotIndex` enum has 8
matching variants -- `LoadA` and `LoadB` are independent ports that can
issue together in 128-bit bundles, even though they share the load
execution unit.

| AIE2 slot | SlotIndex | Bits | Function |
|-----------|-----------|------|----------|
| lda | LoadA | 21 | Load A (pointer-based) |
| ldb | LoadB | 16 | Load B (pointer-based, only present in 128-bit bundles) |
| alu | Scalar0 | 20 | Scalar ALU |
| mv | Scalar1 | 22 | Move / scalar ALU 1 |
| st | Store | 21 | Store |
| vec | Vector | 26 | Vector ALU |
| lng | Accumulator | 42 | Long instruction (accumulator-class operations) |
| nop | Control | 1 | Branches, calls, control flow |

## Execution Model

1. `CoreInterpreter` fetches a 128-bit bundle from memory at the current PC
2. `InstructionDecoder` decodes each slot into `Operation` values
3. `CycleAccurateExecutor` executes all slot operations respecting:
   - Data hazards (RAW, WAW, WAR)
   - Branch delay slots (AIE2 has 5 delay slots)
   - Memory bank conflicts
   - Lock synchronization
4. `InterpreterEngine` coordinates multiple cores per cycle

## Cross-Tile Memory Access

A core executing on tile T can read its neighbors' data memory (north, south,
east, west) via load instructions. The interpreter mediates this through
`NeighborMemory` (`src/interpreter/execute/memory/neighbor.rs`), a per-core
read cache that holds a snapshot of each neighbor's data memory.

The cache is **gen-aware**. Each `Tile` carries a `data_memory_gen` counter
that bumps on every write to the tile's data memory. `NeighborMemory::ensure_snapshot`
compares the cached gen against the neighbor's current gen and only re-clones
the 64KB / 512KB memory when they differ. Steady-state cost is one
`u64` comparison per access; full clones happen only when the neighbor
actually wrote.

Access goes through a `NeighborView<'a>` (`src/device/state/mod.rs`), a
borrow-safe split-slice view into the tile array with the executing tile's
slot held out as a hole. `DeviceState::split_tile_mut(col, row)` returns
`(&mut Tile, NeighborView<'_>)` in one call -- the executing tile is
mutable, and the rest of the array is read-only through the view.
`NeighborMemory::ensure_snapshot` is generic over a `TileLookup` trait
implemented by both `DeviceState` and `NeighborView<'a>`, so the same
caching path serves both the coordinator's eager refresh and the
read site's lazy access.

`NeighborMemory` lives on `CoreState`, so the cache survives across
interpreter steps (Stage 1 hoist). Stage 2 threads `&NeighborView` through
the read paths so the read site declares its own data dependency rather
than assuming the coordinator pre-decided.

## Key Bug Fixes to Preserve

- **Branch delay slots**: AIE2 has 5 delay slots after a branch. Instructions in delay slots execute before the branch takes effect.
- **VLIW execution order**: All slot operations in a bundle read state before any writes. This prevents intra-bundle data hazards.
- **Long instruction handling**: `lng` slot spans two physical slot positions. The decoder must not try to decode the second half as a separate instruction.
