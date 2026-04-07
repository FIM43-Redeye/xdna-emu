# Phase 1: Core Accuracy

**Goal**: Make the emulator faithful to real AIE2 hardware behavior.

**Status**: **100% ISA accuracy achieved (2026-04-06).** 4815/4815 instruction-level
test points pass against real NPU hardware. 2660+ unit tests. Sparse vmac
pipeline is hardware-faithful with oracle-verified routing.

For the confidence tier system used in this document, see [ROADMAP.md](../../ROADMAP.md).

---

## Component Inventory

Each component is assessed for confidence level based on what evidence exists,
not what we believe to be true.

### Instruction Decoder -- VERIFIED

The TableGen-driven decoder has extensive unit tests and has been validated
against real ELF binaries.

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Bundle format detection (16-128 bit) | VERIFIED | Unit tests for all format sizes |
| Slot extraction from VLIW bundles | VERIFIED | Unit tests in `slot_layout.rs` |
| O(1) opcode lookup | VERIFIED | Unit tests in `decoder.rs` |
| 210+ resolved instruction encodings | VERIFIED | TableGen parser tests |
| 100% recognition on `add_one` ELF | OBSERVED | Worked in a session; no regression test |
| 100% recognition on all mlir-aie ELFs | CLAIMED | Tested on very few binaries |

**Files**: `src/interpreter/bundle/`, `src/interpreter/decode/`

### Scalar Unit -- VERIFIED

| Capability | Confidence | Evidence |
|------------|------------|----------|
| 32 GPRs (r0-r31) | VERIFIED | Unit tests in `registers.rs` |
| Pointer registers (p0-p7) | VERIFIED | Unit tests |
| Modifier registers (m0-m7) | VERIFIED | Unit tests |
| ALU ops (add, sub, mul, and, or, xor, shifts) | VERIFIED | Unit tests in `scalar.rs` |
| Condition codes / flags | VERIFIED | Unit tests in `traits.rs` |
| Division (div, divu, mod) | VERIFIED | Unit tests |
| Conditional select (seleqz, selnez) | VERIFIED | Unit tests |
| Sign/zero extend (s8/s16/u8/u16) | VERIFIED | Unit tests |
| Scalar abs, clz, clb, adc, sbc | VERIFIED | Unit tests |

**Files**: `src/interpreter/state/registers.rs`, `src/interpreter/execute/scalar.rs`

### Vector Unit -- VERIFIED (100% ISA accuracy)

All vector operations validated against real NPU hardware via the ISA test
harness (4815/4815 test points). Coverage includes all element type
combinations, float edge cases, and sparse matrix multiply.

| Capability | Confidence | Evidence |
|------------|------------|----------|
| W registers (24x256-bit) | VERIFIED | Unit tests + ISA harness |
| X registers (12x512-bit wide) | VERIFIED | ISA harness |
| Accumulator registers (8x512-bit, 4x1024-bit) | VERIFIED | ISA harness |
| Arithmetic ops (add, sub, mul, min, max, abs, neg) | VERIFIED | ISA harness 100% |
| Matrix multiply -- dense (i8xi4, i8xi8, i16xi8, i16xi16, bf16) | VERIFIED | ISA harness 100% |
| Matrix multiply -- sparse (all types, hw-faithful vmac pipeline) | VERIFIED | ISA harness 100%, oracle-verified |
| Shift-Round-Saturate (SRS, 10 rounding modes) | VERIFIED | ISA harness 100% |
| UPS (widening conversion) | VERIFIED | ISA harness 100% |
| Type conversion (bf16/f32/int, all directions) | VERIFIED | ISA harness 100% |
| Vector load/store with post-modify | VERIFIED | Unit tests + bridge tests |
| Vector element ops (extract, insert, push) | VERIFIED | ISA harness 100% |
| Vector comparison ops (eq, ge, lt, eqz, select) | VERIFIED | ISA harness 100% |
| Vector bitwise ops (and, or, xor, not) | VERIFIED | ISA harness 100% |
| Vector conditional arithmetic (addsub, negadd, maxdiff) | VERIFIED | ISA harness 100% |
| BFloat16 arithmetic (NaN canonical, FTZ, PSA 30-bit) | VERIFIED | ISA harness 100% |
| Float32 edge cases (NaN, inf, denorm) | VERIFIED | ISA harness + dedicated unit tests |
| SIMD shuffle/permute (40+ modes via routing tables) | VERIFIED | ISA harness 100% |

**Files**: `src/interpreter/execute/vector.rs`, `vector_arith.rs`, `vector_compare.rs`,
`vector_misc.rs`, `vector_matmul.rs`, `vmac_hw.rs`, `vector_permute.rs`

### Memory System -- VERIFIED

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Load/store (byte through vector-256) | VERIFIED | Unit tests in `memory.rs` |
| Post-modify addressing modes | VERIFIED | Unit tests |
| Bank conflict detection | VERIFIED | Unit tests in `timing/memory.rs` |
| Access latency model (5 cycles base) | VERIFIED | Unit tests |
| Alignment checking | VERIFIED | Unit tests |
| Bank mapping (bits[6:4]) | VERIFIED | Unit tests |
| Word-addressed DMA loads/stores | VERIFIED | Unit tests |

**AM020 Memory Architecture** (Ch4):
- Data memory per tile: 64 KB (8 banks x 8 KB)
- Memory tile: 512 KB (16 banks x 32 KB)
- Bank width: 128 bits
- Two 256-bit load ports + one 256-bit store port

**Files**: `src/interpreter/execute/memory.rs`, `src/interpreter/timing/memory.rs`

### DMA Engine -- VERIFIED (unit), OBSERVED (integration)

The DMA engine has good unit test coverage. Multi-tile DMA integration was
observed working in sessions but is covered by very few automated tests.

| Capability | Confidence | Evidence |
|------------|------------|----------|
| BD configuration and interpretation | VERIFIED | Unit tests in `dma/mod.rs` |
| 1D/2D/3D/4D addressing patterns | VERIFIED | Unit tests in `dma/addressing.rs` |
| Transfer state machine with locks | VERIFIED | Unit tests in `dma/transfer.rs` |
| BD chaining (Next_BD from word5) | VERIFIED | Unit tests |
| Repeat count (task queue bits 23:16) | VERIFIED | Unit tests |
| Channel start/stop/pause/resume | VERIFIED | Unit tests in `dma/engine.rs` |
| Per-phase timing model | VERIFIED | Unit tests in `dma/timing.rs` |
| Channel arbitration (round-robin) | VERIFIED | Unit tests |
| Host memory interface (sparse 64-bit) | VERIFIED | Unit tests in `host_memory.rs` |
| Two-tile DMA stream flow (256-byte) | OBSERVED | Worked in session |
| Three-tile pipeline chain (128-byte) | OBSERVED | Worked in session |
| Bidirectional ping-pong DMA | OBSERVED | Worked in session |
| objFifo buffer convention | **BROKEN** | Produces input+41 instead of input+1 |
| Continuous-loop kernel handling | **BROKEN** | Test runner times out |

**Known issues**:
- Dual-abstraction: `channel.rs` and `engine.rs` both define channel state
- 8 `unwrap()` calls in `engine.rs` should use `expect()` with context
- BD field parsing from AM029 has not been cross-checked against aie-rt

**Files**: `src/device/dma/`, `src/device/host_memory.rs`

### Synchronization -- VERIFIED

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Lock acquire/release with value | VERIFIED | Unit tests in `control.rs` |
| Lock value clamping (6-bit, 0-63) | VERIFIED | Unit tests |
| Lock overflow/underflow flags | VERIFIED | Unit tests |
| Lock timing (1 cycle acquire) | VERIFIED | Unit tests in `timing/sync.rs` |
| Lock contention tracking | VERIFIED | Unit tests |
| Deadlock detection (DFS cycle) | VERIFIED | Unit tests in `timing/deadlock.rs` |
| Barrier synchronization | VERIFIED | Unit tests in `timing/barrier.rs` |

**Files**: `src/interpreter/execute/control.rs`, `src/interpreter/timing/sync.rs`,
`src/interpreter/timing/deadlock.rs`, `src/interpreter/timing/barrier.rs`

### Stream Switch -- VERIFIED (unit), OBSERVED (integration)

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Per-tile switch with ports/FIFOs | VERIFIED | Unit tests in `stream_switch.rs` |
| Master/slave port configuration | VERIFIED | Unit tests |
| Backpressure (FIFO full) | VERIFIED | Unit tests |
| Route configuration API | VERIFIED | Unit tests |
| Circuit-switched routing | VERIFIED | Unit tests |
| Packet-switched routing | VERIFIED | Unit tests |
| Packet header/arbitration | VERIFIED | Unit tests |
| Routing latency calculation | VERIFIED | Unit tests in `stream_router.rs` |
| Tile-to-tile data movement | OBSERVED | Worked in session tests |
| Multi-stream routing (>2 inputs) | **BROKEN** | Fails on shim port 12 |

**Files**: `src/device/stream_switch.rs`, `src/device/stream_router.rs`

### Pipeline / Timing -- VERIFIED (infrastructure), CLAIMED (accuracy)

The timing infrastructure is thoroughly tested. But whether it produces
cycle counts that match real hardware is entirely unknown.

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Instruction latency tables | VERIFIED | Unit tests in `timing/latency.rs` |
| RAW/WAW/WAR hazard detection | VERIFIED | Unit tests in `timing/hazards.rs` |
| VLIW structural hazard detection | VERIFIED | Unit tests in `timing/slots.rs` |
| Branch penalty (3 cycles) | VERIFIED | Unit tests |
| Branch delay slots (5-cycle) | VERIFIED | Unit tests |
| Stall cycle modeling | VERIFIED | Unit tests in `cycle_accurate.rs` |
| Event tracing (13 event types) | VERIFIED | Unit tests |
| CycleAccurateExecutor integration | VERIFIED | Unit tests |
| Cycle counts match hardware | CLAIMED | No comparison has been done |
| Full pipeline model (F/D/E/WB) | Not implemented | Uses latency + hazard model |

**Files**: `src/interpreter/timing/`, `src/interpreter/execute/cycle_accurate.rs`

### Multi-Core Coordination -- VERIFIED (unit), CLAIMED (realistic workloads)

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Memory tile arbitration (round-robin) | VERIFIED | Unit tests in `timing/arbitration.rs` |
| Cross-tile memory latency (0/4/8 cycles) | VERIFIED | Unit tests |
| Global cycle counter | VERIFIED | Unit tests |
| DMA-lock timing integration | VERIFIED | Unit tests |
| Realistic multi-tile workloads | CLAIMED | Only 3 unit tests exercise >1 tile |

**Files**: `src/interpreter/timing/arbitration.rs`, `src/interpreter/core/`,
`src/interpreter/engine/`

### Kernel Execution -- VERIFIED (1 kernel), OBSERVED (2 kernels)

| Capability | Confidence | Evidence |
|------------|------------|----------|
| `add_one_using_dma`: correct outputs | VERIFIED | TestRunner regression test |
| `add_314_using_dma_op`: correct outputs | OBSERVED | Worked in session (64/64) |
| 24 xclbins load without unknown opcodes | OBSERVED | Worked in session |
| General kernel correctness | CLAIMED | Only 1-2 kernels tested to completion |

**Files**: `src/interpreter/test_runner.rs`

---

## TableGen Parser

The parser extracts instruction definitions from llvm-aie's TableGen files and
builds O(1) decoder tables.

| Capability | Confidence | Evidence |
|------------|------------|----------|
| Slot parsing (8 slots, correct widths) | VERIFIED | Unit tests |
| Format class parsing (144 classes) | VERIFIED | Unit tests |
| Instruction parsing (135+ definitions) | VERIFIED | Unit tests |
| Encoding resolution (210+ instructions) | VERIFIED | Unit tests |
| Nested template parameter handling | VERIFIED | Unit tests |

**Files**: `src/tablegen/`

---

## Architecture Notes

### VLIW Structure (AM020)
- 8 functional slots: `lda`, `ldb`, `alu`, `mv`, `st`, `vec`, `lng`, `nop`
- Variable slot widths: 16-42 bits per slot
- Bundle sizes: 2-byte (nop) through 16-byte (full VLIW)

### Slot Mapping (decoder to interpreter)

| AIE2 Slot | Interpreter SlotIndex |
|-----------|----------------------|
| lda, ldb | Load |
| alu | Scalar0 |
| mv | Scalar1 |
| st | Store |
| vec, lng | Vector |
| nop | Control |

### Key Bug Fixes (for historical context)

These are documented here because they represent non-obvious hardware behaviors
that were discovered through debugging, not documentation.

**Branch Delay Slots** (discovered Jan 3, 2026):
AIE2 uses 5-cycle branch delay slots -- instructions already in the pipeline
continue executing after a branch is taken. Without this, loops exit too early
(off-by-one on iteration count). Implemented as `PendingBranch` in `context.rs`.

**BD Chaining Field Location** (discovered Jan 1, 2026):
Next_BD and Use_Next_BD fields are in word 5 (d1) at bits 30:27 and bit 26,
not in word 3 (control) as initially assumed. Source: AM029.

**VLIW Execution Order** (discovered Jan 6, 2026):
Store slots must execute before Scalar slots for correct data flow. The
original execution order was wrong.

---

## Documentation Sources

Architecture constants verified against AMD official documentation:
- **AM020**: Versal AI Engine ML (AIE-ML) Architecture Manual
- **AM025**: AIE-ML Register Reference
- **AM027**: AIE-ML v2 (AIE2P) Architecture Manual
- **AM029**: AIE-ML v2 Register Reference

All constants defined in `src/device/aie2_spec.rs` with AM020 section references.

---

## What Would "Done" Look Like?

Phase 1 has no formal exit criteria. Here is a proposed definition of "done"
that would justify moving focus to Phase 2:

1. **All 30 mlir-aie test binaries run without crashes or unknown opcodes**
   (currently: OBSERVED for 24, untested for 6)
2. **At least 10 kernels produce correct output**
   (currently: 1-2 verified)
3. **Timing model validated against aiesimulator on at least 3 kernels**
   (currently: zero comparisons)
4. **No known-broken features in the critical path**
   (currently: objFifo results wrong, multi-stream routing broken, continuous
   kernels timeout)
5. **SemanticOp test coverage above 50%**
   (currently: 17%)

None of these criteria are met today. Phase 1 is functional but not complete.
