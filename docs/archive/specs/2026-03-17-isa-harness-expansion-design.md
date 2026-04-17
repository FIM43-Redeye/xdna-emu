# ISA Harness Expansion: Loads, Stores, Branches

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Extend `tools/isa-test-gen.py` to cover load, store, and branch instructions

## Problem

The ISA validation harness currently tests 162 of 606 AIE2 instructions (27%).
The remaining 444 are skipped because they require addressing modes (loads/stores),
control flow verification (branches), live hardware (streams/locks), or have
unrecognized operand types. This design expands coverage to ~406 instructions (67%)
by adding test strategies for loads, stores, branches, and reclassifying
`dontcare` padding operands.

### Classification Guards

Several instructions straddle categories and need explicit guards:

- **may_load + may_store:** 12 store instructions in the `lda` slot (ST_2D_S16,
  ST_3D_S16, etc.) have both flags set. LoadStrategy must reject instructions
  that also have `may_store=true` to avoid misclassifying stores as loads.
- **Register-to-register "loads":** 6 VLDB_4x* instructions have `may_load=true`
  but are actually register-to-register vector shuffles with no pointer operand.
  LoadStrategy must reject instructions with no pointer-kind operand.
- **Compressed/sparse loads:** 14 VLDB_COMPR_*/VLDB_SPARSE_* instructions have
  pointer operands but read from a hardware FIFO state machine requiring
  compression unit initialization. LoadStrategy rejects these (deferred).
- **Branch immediates:** Branch target operands (`cpmaddr`) are always label
  references in generated assembly, never numeric. The combo generator must
  skip immediate variation for branch target operands.

## Architecture: Pluggable Test Strategies

Refactor the monolithic `classify_instruction()` and `generate_test_asm()` into
pluggable strategies. Each strategy answers three questions:

1. **Can I test this?** -- classification
2. **What setup do I need?** -- register/pointer initialization
3. **How do I capture the result?** -- store to output buffer

### Strategy Interface

```python
class TestStrategy:
    def can_test(self, instr) -> tuple[bool, str]:
        """Return (True, "") or (False, skip_reason)."""

    def setup_code(self, instr, operands, in_offset, ctx) -> tuple[list[str], int]:
        """Return (asm_lines, input_bytes_consumed)."""

    def capture_code(self, instr, operands, out_offset, ctx) -> tuple[list[str], int]:
        """Return (asm_lines, output_bytes_produced)."""

    def execute_code(self, instr, operands) -> list[str]:
        """Return the instruction-under-test (+ latency NOPs)."""
```

### Strategy Dispatch Order

```python
STRATEGIES = [
    BranchStrategy(),    # branches checked first (small, distinct)
    LoadStrategy(),      # may_load=true
    StoreStrategy(),     # may_store=true
    ComputeStrategy(),   # everything else (existing logic)
]
```

First strategy that returns `can_test=True` wins. The existing compute logic
becomes `ComputeStrategy` with no behavioral change.

### Shared Infrastructure (Unchanged)

- Batch packer (greedy bin-packing into 16KB kernels)
- Manifest format (per-test-point in_offset, out_offset, in_size, out_size)
- Assembly header/footer (`.text`, `.globl test_kernel`, `ret lr`)
- Latency lookup from `aie2-sched-latencies.json`
- Operand combination generator (one-at-a-time variation)
- Runner script (`scripts/isa-test.sh`) -- completely unchanged

## Load Strategy

**Coverage target:** ~119 of 168 load instructions (Tiers 1+2).

### Register Convention

- `p0` = input buffer base (harness-reserved, read-only)
- `p1` = output buffer base (harness-reserved, write-only)
- `p6` = scratch pointer for load-under-test (new convention)
- `p7` = scratch pointer for store strategy (new convention)
- `m0` = scratch modifier (zeroed for 2D/3D loads)

**Note:** p6 serves double duty: the setup code uses it to reach the data
region (via `padda`), and then the load-under-test reads through it. This is
safe because setup completes before execution. The existing compute strategy
also uses p6 as a scratch pointer for large-offset input loading, so there is
no conflict -- each test point's setup is self-contained.

### Tier 1: Simple Loads (register + immediate offset)

~47 instructions. Addressing: `[ptr, #imm]` or `[ptr, dj]`.

```asm
// Setup: position scratch pointer at known data
mov p6, p0
padda [p6], #data_offset       // advance to input data region

// Execute: load instruction under test
lda.s16 r0, [p6, #0]          // INSTRUCTION UNDER TEST
nop                            // x result_latency(instr) from sched model
...

// Capture: store loaded value to output
st r0, [p1, #out_offset]
```

For loads with an immediate offset operand in the instruction encoding, we set
`p6` to point at the appropriate base such that `base + encoded_offset` lands
on valid input data.

For loads with a `modifier_dj` index register, we zero `dj0` and position
`p6` directly at the data.

### Tier 2: Modifier Loads (2D/3D with modifier_m)

~72 instructions. Addressing: `[ptr], mod` (auto-increment).

```asm
// Setup: zero modifier step so auto-increment is benign
mov m0, #0
mov p6, p0
padda [p6], #data_offset

// Execute: load with 2D modifier
lda.2d.s16 r0, [p6], m0       // INSTRUCTION UNDER TEST (auto-increments p6)
nop x{result_latency}         // per-instruction from sched model

// Capture
st r0, [p1, #out_offset]
```

The modifier is zeroed so auto-increment doesn't corrupt the pointer. We only
care about the loaded *value*, not pointer arithmetic (which is tested
separately by padda/paddb in the compute strategy).

### Tier 3: Composite Destination Loads (Deferred)

~7 instructions using `LdaScl`/`LdaCg` composite operands. These encode
destination type + data width + size in a 7-bit field that requires a lookup
table to decode. Deferred to the composite register work.

### Vector Load Variants

Same pattern with wider capture:

```asm
mov p6, p0
padda [p6], #data_offset
vlda wl0, [p6, #0]            // vector load under test
nop x{result_latency}         // per-instruction from sched model
vst wl0, [p1, #out_offset]    // vector store to capture (32 bytes)
```

512-bit loads capture both halves. Accumulator-destination loads use the
existing VSRS proxy to convert to vector before storing.

### Data Layout

The input buffer contains deterministic PRNG data (seeded). Each test point's
`in_offset` records where in the buffer the load reads from. The manifest
stores this offset. The comparison is HW-vs-EMU output identity -- no golden
oracle needed.

### Stack-Pointer Loads

~6 instructions using `[sp, #imm]`. These require SP setup, which is risky
in a flat test kernel. Deferred as a stretch goal.

## Store Strategy

**Coverage target:** ~94 of 126 store instructions (Tiers 1+2).

Mirror image of loads: load known data into a register, then execute the
store-under-test to write it to the output buffer.

### Tier 1: Simple Stores

```asm
lda r0, [p0, #in_offset]      // load known value from input
nop x7
mov p7, p1                    // copy output base
padda [p7], #data_offset
st.s16 r0, [p7, #0]           // INSTRUCTION UNDER TEST
```

### Tier 2: Modifier Stores (2D/3D)

```asm
lda r0, [p0, #in_offset]
nop x7
mov m0, #0
mov p7, p1
padda [p7], #data_offset
st.2d.s16 r0, [p7], m0        // INSTRUCTION UNDER TEST
```

### Tier 3: Vector/Accumulator Stores

```asm
vlda wl0, [p0, #in_offset]
nop x7
mov p7, p1
padda [p7], #data_offset
vst wl0, [p7, #0]             // INSTRUCTION UNDER TEST
```

### Data Type Narrowing

A `st.s8` stores 1 byte; `st.s16` stores 2 bytes. The `out_size` for each
test point reflects the actual store width, not the source register width.
The manifest already tracks per-test-point sizes.

## Branch Strategy

**Coverage target:** 4-6 of 8 branch instructions, ~10 test points.

### Marker-Based Verification

Branches can't be verified by data flow alone. Instead, we store marker values
that differ depending on which path executes:

```asm
// Setup: load condition value
lda r0, [p0, #in_offset]       // known value (zero or nonzero)
nop x7

// Store "before" marker
mov r14, #0xAA
st r14, [p1, #out_offset]

// Execute branch under test
jnz r0, .Ltaken_N              // INSTRUCTION UNDER TEST
nop                             // x5 branch delay slots
nop
nop
nop
nop

// Fall-through path
mov r14, #0xBB
st r14, [p1, #out_offset+4]
j .Ldone_N
nop x5

// Taken path
.Ltaken_N:
mov r14, #0xCC
st r14, [p1, #out_offset+4]

.Ldone_N:
```

Output is 8 bytes: `[0xAA, path_marker]`. HW and EMU must produce identical
markers for both taken and not-taken inputs.

**Label uniqueness:** Labels use a globally unique counter suffix (`_N`) across
the entire batch, not just within a single test point. All test points are
concatenated into one `.s` file, so duplicate labels would cause assembler errors.

### Testable Branches

| Instruction | Test Points | Notes |
|-------------|-------------|-------|
| `j addr` | 1 | Unconditional, always taken |
| `jl addr` | 1 | Jump-and-link, verify LR is set (store LR after) |
| `jnz r, addr` | 2 | Conditional: zero input (not taken) + nonzero (taken) |
| `jz r, addr` | 2 | Conditional: zero input (taken) + nonzero (not taken) |
| `j ptr` | 1 | Register-indirect, set ptr to known label |
| `jl ptr` | 1 | Register-indirect jump-and-link |

### Deferred Branches

- `ret lr` -- needs valid LR setup in a flat kernel
- `jnzd` -- delayed branch with different delay slot semantics

## Dontcare Reclassification

~56 instructions are skipped due to `operand_type="unknown"`. Analysis shows
most are `dontcare` padding fields (1-4 bits of VLIW slot alignment).

**Fix:** In `classify_instruction()`, treat `unknown` operand types as
non-blocking when the operand name starts with `dontcare`. Zero-fill them in
assembly output. This reclassifies ~25 instructions into testable categories:

- `padda`, `paddb`, `padds` variants -> compute strategy
- Various ALU instructions with padding bits -> compute strategy
- Lock/stream instructions with padding -> still deferred (other skip reasons)

## Coverage Summary

| Class | Before | After | Delta |
|-------|--------|-------|-------|
| Compute (existing) | 162 | 162 | -- |
| Loads (Tier 1+2) | 0 | ~119 | +119 |
| Stores (Tier 1+2) | 0 | ~94 | +94 |
| Branches | 0 | ~6 | +6 |
| Dontcare reclassification | 0 | ~25 | +25 |
| **Total testable** | **162** | **~406** | **+244** |
| **Coverage** | **27%** | **~67%** | **+40pp** |

Remaining untestable (~200): composites (31), compressed/sparse loads (14),
streams (23), locks (8), NOPs/side-effects (9), stack-pointer loads/stores (~12),
composite-destination loads (~7), register-to-register VLDB_4x (6),
other unknown (~90).

## Deferred Work

- **Composite registers (31):** Needs lookup table for multi-field encodings
- **Compressed/sparse loads (14):** Requires compression unit initialization (FIFO state machine)
- **Register-to-register VLDB_4x (6):** Actually vector shuffles, not memory loads; belongs in compute strategy with composite operand support
- **Streams (23):** Needs live stream switches; separate test infrastructure
- **Locks (8):** Needs real lock hardware; bridge test territory
- **Stack-pointer loads/stores (~12):** Needs SP setup in flat kernel

## Testing

Verification is the same as today:

1. `scripts/isa-test.sh` generates, assembles, packages, runs on HW and EMU
2. Binary comparison of HW vs EMU output per batch
3. Manifest provides per-test-point attribution for any divergence
4. All existing 40 batches (162 instructions) must still pass unchanged

New batches are additive -- the manifest format and runner are unchanged.

**Batch count growth:** Adding ~244 instructions with combo variations will
roughly triple the batch count from 40 to ~120+. The aiecc.py packaging step
is the slowest phase (~10-15s per batch), so total harness run time will
increase proportionally. The runner already parallelizes EMU runs; packaging
could be parallelized in a future optimization.

## Files Modified

| File | Change |
|------|--------|
| `tools/isa-test-gen.py` | Refactor into strategy classes, add Load/Store/Branch strategies |
| `tools/test_isa_test_gen.py` | Add unit tests for new strategies |

No changes to `scripts/isa-test.sh`, `aie2-isa.json`, or the runner infrastructure.
