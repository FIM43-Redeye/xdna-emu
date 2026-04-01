# ISA Harness Memory Safety Fix -- Design Spec

**Date**: 2026-03-31
**Status**: Draft
**File**: `tools/isa-test-gen.py`

## Problem

The ISA test harness has 10 memory safety and correctness bugs that cause 4
batches to be genuinely nondeterministic on hardware (batch_27, batch_44,
batch_77, batch_78). An additional 8 batches show cold-start-only differences
caused by stale data memory from CDO/boot.

Root cause: store and load instructions can write/read anywhere in the tile's
64KB data memory because the harness does not constrain computed addresses to
the allocated buffer regions. When a store escapes its output slot, it
corrupts other tests' memory within the same batch.

## Design Principle: Sequential Output Tape

**Every strategy must write its output to the next sequential position in the
output buffer.** The main generation loop tracks `batch_out_offset` and passes
it to each test point. Each test point writes exactly at that offset, then the
loop advances the offset by `compute_output_size()`. No strategy may write to
a computed or escaped address.

Corollary: input reads are PRNG-driven (offsets chosen by the generator), but
output writes form a clean, predictable, sequential tape. This is what makes
the output buffer verifiable.

## Fix 1: Shared Safe Pointer Setup

**Issues addressed**: 1 (uncontrolled store addresses), 3 (modifier setup
timing), 6 (modifier combos ignored).

### Current Problem

LoadStrategy and StoreStrategy each independently set up pointer registers
(p6 for loads, p7 for stores) and zero modifier registers. Both get the
ordering wrong: they zero modifiers AFTER the pointer is set up. For 2D/3D
addressing modes, the modifier controls the stride/wrap. If the modifier has
a stale value when the pointer is first used, the access goes to the wrong
address.

Additionally, both strategies hardcode which modifier register to zero rather
than using the combo-assigned register name, so if the combo assigns m1 but
the code zeros m0, the instruction uses unzeroed m1.

### Fix

Extract a shared helper:

```python
def _safe_ptr_setup(ptr_reg, base_ptr, offset, op_by_name, regs):
    """Set up a pointer for load/store with all modifiers pre-zeroed.

    Order: zero modifiers FIRST, then set the pointer. This prevents
    stale modifier values from corrupting 2D/3D addressing.
    """
    lines = []
    # 1. Zero ALL modifier registers this instruction references.
    for op_name, op in op_by_name.items():
        kind = op.get("register_kind", "")
        if kind == "modifier_m":
            lines.append(f"  mov {regs.get(op_name, 'm0')}, #0")
        elif kind == "modifier_dj":
            lines.append(f"  mov {regs.get(op_name, 'dj0')}, #0")
    # 2. THEN set the pointer (modifiers are now safe).
    lines.extend(_padda_sequence(ptr_reg, base_ptr, offset))
    return lines
```

Both `LoadStrategy.generate_test_point` and `StoreStrategy.generate_test_point`
call this instead of their inline modifier zeroing + padda sequence.

## Fix 2: Zero All Immediates Including Post-Modify

**Issues addressed**: 2 (post-modify immediates preserved from random data).

### Current Problem

Both LoadStrategy and StoreStrategy check `_is_postmodify_immediate()` and
deliberately preserve combo-specified values for post-modify immediates. After
a post-modify instruction like `vst x0, [p7], #128`, p7 escapes to
`original + 128`. If the next test inherits p7, it starts from the escaped
address.

### Fix

Zero ALL immediates unconditionally in Load and Store strategies:

```python
if op.get("operand_type") == "immediate":
    store_regs[op_name] = "0"
```

The test's purpose is to verify the instruction's data transfer. Post-modify
pointer behavior is PointerArithStrategy's domain. With all immediates zeroed,
every load/store accesses exactly `[ptr + 0]` -- the address set up by
`_safe_ptr_setup` IS the final address. Clean, bounded, deterministic.

## Fix 3: StoreStrategy Output via Sequential Tape

**Issues addressed**: 1 (stores with uncontrolled computed addresses),
reinforces the sequential output tape principle.

### Current Flow (broken)

1. Load source data from input buffer
2. Set up p7 to output region
3. Zero modifiers (TOO LATE)
4. Execute store with combo post-modify (DANGEROUS -- p7 escapes)

### New Flow

1. Load source data from input buffer via `_load_instruction(src_reg, kind, "p0", in_offset)`
2. NOP sled for load latency
3. `_safe_ptr_setup("p7", "p1", out_offset, ...)` -- modifiers zeroed first
4. Execute store with ALL immediates zeroed -- writes to exactly `[p7, #0]`

The store hits exactly the output slot the caller allocated. The output offset
is `out_offset` as given by the main loop. No escape possible.

## Fix 4: LoadStrategy Output via Sequential Tape

**Issues addressed**: 3 (modifier timing), reinforces the sequential output
tape principle.

### New Flow

1. `_safe_ptr_setup("p6", "p0", in_offset, ...)` -- modifiers zeroed, then pointer set
2. Execute load with ALL immediates zeroed -- reads from exactly `[p6, #0]`
3. NOP sled for load latency
4. Store result to output via `_store_instruction(dest_reg, kind, "p1", out_offset)`

The load reads from the designated input slot, the result is written to the
next sequential output position. Both addresses are bounded.

## Fix 5: PointerArithStrategy Cleanup

**Issues addressed**: 4 (pointer arithmetic unbounded).

PointerArithStrategy is different from Load/Store: the pointer value IS the
test output. It loads a pointer from PRNG input, applies the operation, then
stores the resulting pointer value as a scalar to the output buffer.

The pointer is never dereferenced for data access, so there is no memory
corruption risk from the operation itself. The risk is a leaked pointer
register being accidentally used by subsequent tests.

### Fix

After capturing the result, restore the pointer register:

```python
# After: mov r14, ptr_reg / st r14, [p1, #out_offset]
lines.append(f"  mov {ptr_reg}, p1")  # Reset to safe value
```

This is defensive -- subsequent tests set up their own pointers anyway. But
it prevents any future code change from accidentally inheriting a wild pointer.

## Fix 6: `detect_output_operands` for Stores

**Issues addressed**: 5 (inverted output detection for stores).

### Current Problem

`detect_output_operands` treats the first asm_string operand as the
destination for ALL instructions. For store instructions, the first operand is
the source data register, not the destination (which is memory, not a
register). StoreStrategy works around this with `_detect_store_source()`, but
the root function is wrong.

`ComputeStrategy` and `generate_operand_combos` call `detect_output_operands`
and may misclassify a store's source register as an output. This can cause
the combo generator to skip loading input data for the source register.

### Fix

Early-return empty list when the instruction is a pure store:

```python
def detect_output_operands(instr):
    if instr.get("may_store") and not instr.get("may_load"):
        return []
    # ... existing logic ...
```

Store instructions write to memory, not to a register. They have no register
output. StoreStrategy already uses `_detect_store_source()` to find the
data register independently.

## Fix 7: `_padda_sequence` Negative Offsets

**Issues addressed**: 7 (negative offsets silently ignored).

### Current Problem

`_padda_sequence` only handles `remaining > 0`. If `offset < 0`, the while
loop and final if-statement both skip, and the pointer stays at base. This
means any test requesting a negative offset gets the wrong address.

### Fix

Handle both positive and negative offsets:

```python
def _padda_sequence(ptr_reg, base_ptr, offset):
    lines = [f"  mov {ptr_reg}, {base_ptr}"]
    remaining = offset
    while abs(remaining) > 1024:
        step = 1024 if remaining > 0 else -1024
        lines.append(f"  padda [{ptr_reg}], #{step}")
        remaining -= step
    if remaining != 0:
        lines.append(f"  padda [{ptr_reg}], #{remaining}")
    return lines
```

Note: In practice, all current offsets are non-negative (buffers start at 0
and grow upward). This fix is defensive against future use and eliminates the
silent-failure mode.

## Fix 8: Output Buffer Bounds Check

**Issues addressed**: 8 (output offset not bounds-checked).

### Current Problem

`batch_out_offset` increases monotonically in the main generation loop. No
check that it stays within the output buffer allocation. Overflow would cause
a store to write past the output buffer into the input buffer region (or
beyond data memory).

### Fix

Add a generation-time assertion in the main batch loop (lines ~4900-4930):

```python
batch_out_offset = _align(batch_out_offset, 32)
if batch_out_offset + out_size > MAX_OUT_BUFFER_SIZE:
    # Start a new batch -- this test point doesn't fit.
    break
```

This is purely a generation-time safety net. The bin-packing already respects
program memory limits; this adds the same check for data memory. If output
buffer space is exhausted, the test point goes into the next batch.

The output buffer limit is the tile's 64KB data memory minus the input
buffer allocation. Since the MLIR template sizes the objectfifo from
`batch_out_offset`, the check guards against bugs in size computation that
could cause the output region to overflow into unrelated memory. In practice,
program memory (1024 bundles) is the binding constraint -- data memory
overflow would require an implausibly large number of test points. This check
is a safety net, not an expected code path.

## Fix 9: Scratch Region Documentation

**Issues addressed**: 9 (zero-init scratch region overlap with test outputs).

### Assessment

The preamble writes zeros to `[p1, #0..#12]` for mask register loading. Tests
then overwrite those same output offsets with their actual output data. This
is not a bug -- the preamble runs before any tests, and the test data
overwrites the scratch zeros.

### Fix

Add a comment in `_register_zeroing_preamble()` documenting the assumption
that the scratch region (offsets 0-15 of the output buffer) will be
overwritten by test output. No code change needed.

## Fix 10: Implicit Register Offset Bounds Check

**Issues addressed**: 10 (implicit register offsets unchecked).

### Current Problem

Implicit register loads (e.g., r29 for VINSERT, r27 for SELEQZ) read from
`cur_in_offset` without checking bounds against the input buffer size. If
the input buffer is smaller than expected, the load reads garbage.

### Fix

Add bounds check before emitting implicit register loads:

```python
if cur_in_offset + 4 > in_buffer_size:
    # Skip this test point -- not enough input space for implicit regs.
    return None  # Caller skips this test point.
```

This parallels the program memory overflow handling: if the test doesn't fit,
skip it rather than generate broken code.

## Audit: All Strategies Against Sequential Output Principle

Every strategy's `generate_test_point` must write to `out_offset` as given
by the caller. Quick audit of all 16 strategies:

| Strategy | Writes to out_offset? | Action needed |
|----------|----------------------|---------------|
| ComputeStrategy | Yes (via `_store_instruction(reg, kind, "p1", cur_out_offset)`) | None |
| LoadStrategy | Yes, but via p6 indirection | Fix: store result via p1+out_offset (Fix 4) |
| StoreStrategy | Via p7 + combo post-modify | Fix: use _safe_ptr_setup + zero immediates (Fix 3) |
| BranchStrategy | Disabled (LLVM IR) | N/A |
| LockStrategy | Via `_store_instruction` | Verify correct |
| FifoLoadStrategy | Via `_store_instruction` | Verify correct |
| CascadeStrategy | Via `_store_instruction` | Verify correct |
| CascadeReadStrategy | Via `_store_instruction` | Verify correct |
| StreamStrategy | Multi-tile, separate buffers | N/A (different model) |
| ConversionStrategy | LLVM IR | N/A |
| DoneStrategy | Via `_scalar_store` | Verify correct |
| EventStrategy | Via `_scalar_store` | Verify correct |
| PointerArithStrategy | Via `_scalar_store("r14", "p1", out_offset)` | Add pointer restore (Fix 5) |
| PaddaSpStrategy | Via `_scalar_store` | Verify correct |
| AccumArithStrategy | Via `_store_instruction` | Verify correct |
| VmacStrategy | Via `_store_instruction` | Verify correct |

Strategies marked "Verify correct" already use the sequential pattern. During
implementation, confirm each one actually honors `out_offset` and does not
use a computed pointer for output writes.

## Testing

After applying all fixes:

1. **Regenerate all ISA test batches**: `bash scripts/isa-test.sh --generate-only`
2. **Compile**: `bash scripts/isa-test.sh --compile`
3. **Determinism check**: Run every batch 5x on hardware. All batches must be
   deterministic (0 nondeterministic, 0 errors).
4. **ISA accuracy**: Re-run the full ISA accuracy comparison. The fixes should
   not regress accuracy (same or better than 79.0%).

## Non-Goals

- This spec does not redesign the test harness from scratch. It patches the
  existing harness to be memory-safe.
- A proper verification framework for the NPU (with formal memory models,
  cycle-accurate comparison, etc.) is a separate future effort.
- Cold-start nondeterminism (stale data memory from CDO/boot on first run)
  is a separate issue. The preamble could zero data memory regions, but
  that is outside this spec's scope.
