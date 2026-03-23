# SRS/UPS Type Inference and Wide Dispatch Fix

## Problem

SRS (shift-round-saturate) and UPS (upshift) instructions have three bugs
causing 10-20% accuracy on narrow variants and 0% on wide (x2c/cm) variants:

### Bug 1: Wrong element_type for dual-type mnemonics

`infer_element_type()` captures the LAST numeric suffix from the mnemonic.
For dual-type instructions like `vups.s32.d16` and `vsrs.s16.s32`, this
captures the INPUT type, but the execution handlers use `element_type` as
the OUTPUT type. Additionally, `infer_element_type` determines signedness
by checking for `.u` in the mnemonic -- but `.d` (AMD's unsigned prefix)
is not recognized, so unsigned variants are misidentified as signed.

Example: `vups.s32.d16` (output=s32, input=d16)
- `infer_element_type("vups.s32.d16")` returns `Int16` (matches "16",
  and `.d16` does not contain `.u` so it defaults to signed)
- UPS handler uses `et = Int16` as the output accumulator type
- `ups_vector_to_acc()` gets `bits_out = 16` instead of `bits_out = 32`
- Result: accumulator values truncated to 16 bits
- The signedness error compounds with Bug 2 below

### Bug 2: from_type never populated during decoding

`SlotOp::from_type` is only set in unit test code. During real instruction
decoding, it is always `None`. The SRS/UPS handlers fall back to defaults:

- SRS: `from_type.unwrap_or(Int32)` -- happens to be correct for S32 accum,
  wrong for S64 accum
- UPS: `from_type.unwrap_or(Int16)` -- always signed, wrong for D16/D32
  (unsigned) variants

This means unsigned UPS variants (`vups.s32.d16`) sign-extend input values
instead of zero-extending them. Values >= 32768 become negative, producing
wrong accumulator results.

### Bug 3: No wide SRS/UPS handlers in execute_wide()

Wide SRS/UPS (x2c and cm variants) fall through to `execute_wide_fallback()`,
which calls `execute_half()` twice with `increment_vector_regs()`. But
`increment_vector_regs()` only increments `VectorReg`, not `AccumReg`.

For wide UPS (`vups.s32.s16 cm0, x4, s0`):
- First half: reads VectorReg(4) -> writes AccumReg(0) (correct)
- Second half: reads VectorReg(5) -> writes AccumReg(0) again (OVERWRITES)

For wide SRS (`vsrs.d32.s64 x4, cm0, s0`):
- First half: reads AccumReg(0) -> writes VectorReg(4) (correct)
- Second half: reads AccumReg(0) again -> writes VectorReg(5) (DUPLICATES)

## Design

### 1. Add Int64/UInt64 to ElementType

SRS/UPS instructions use 64-bit accumulator types (S64 in encoding names).
`ElementType` currently stops at 32-bit. Add two variants:

```rust
pub enum ElementType {
    Int8, UInt8,
    Int16, UInt16,
    Int32, UInt32,
    Int64,   // new
    UInt64,  // new
    BFloat16,
    Float32,
}
```

Update `bits()` to return 64, `lanes_256()` to return 4, `is_signed()` to
include `Int64`. These are real hardware widths -- the accumulator stores
values in u64 lanes, and the SRS/UPS math uses the bit width to determine
shift ranges, saturation bounds, and lane packing.

### 2. Populate from_type at decode time

Add `from_type: Option<ElementType>` to `InstrEncoding` (in resolver.rs).

Add `infer_dual_element_types(name: &str) -> (Option<ElementType>, Option<ElementType>)`
that parses encoding names matching the pattern `V{SRS|UPS}_{OUT}_{IN}_*`:

| Token | Mapping |
|-------|---------|
| S8    | Int8    |
| D8    | UInt8   |
| S16   | Int16   |
| D16   | UInt16  |
| S32   | Int32   |
| D32   | UInt32  |
| S64   | Int64   |
| D64   | UInt64  |

For `VSRS_S16_S32_mv_w_srs`:
- element_type = Int16 (output vector type)
- from_type = Int32 (input accumulator type)

For `VUPS_S32_D16_mv_ups_w2b`:
- element_type = Int32 (output accumulator type)
- from_type = UInt16 (input vector type)

**Fused variants** use a different naming pattern. Fused load-UPS
(`VLDA_2D_UPS_S32_D16`) and fused store-SRS (`VST_2D_SRS_D8_S32`)
embed the SRS/UPS types after the memory operation prefix. The parser
handles both patterns:
- `V{SRS|UPS}_{OUT}_{IN}_*` (standalone)
- `V{LDA|ST}_{2D|3D}_{UPS|SRS}_{OUT}_{IN}_*` (fused)

For non-SRS/UPS instructions, both return None (no change to existing
behavior; the single-type `infer_element_type()` still runs as fallback).

During `InstrEncoding` construction, if `infer_dual_element_types` returns
values, those override the single `infer_element_type` result.

### 3. Propagate from_type through SlotOp

During SlotOp construction from InstrEncoding (in the decode path), copy
`encoding.from_type` into `slot_op.from_type`. The field already exists
on SlotOp but is never populated from the encoding.

### 4. Add wide SRS/UPS to execute_wide()

Both operations are element-wise: each output lane depends on exactly one
input lane. The wide handlers split into halves and call the existing
narrow math:

**Wide SRS** (`SemanticOp::Srs` in execute_wide):
```
read Acc1024 from AccumReg(base) via read_wide
split into lo [u64; 8] and hi [u64; 8]
SRS lo half -> [u32; 8] (low vector register)
SRS hi half -> [u32; 8] (high vector register)
write Vec512 to VectorReg(base) via write_wide
```

**Wide UPS** (`SemanticOp::Ups` in execute_wide):
```
read Vec512 from VectorReg(base) via read_wide
split into lo [u32; 8] and hi [u32; 8]
UPS lo half -> [u64; 8] (low accumulator register)
UPS hi half -> [u64; 8] (high accumulator register)
write Acc1024 to AccumReg(base) via write_wide
```

Both use existing `vector_srs()` and `ups_vector_to_acc()` functions on
each half. The only new logic is reading/writing wide registers and
splitting/concatenating the halves.

**Lane packing for narrow outputs**: When wide SRS produces narrow types
(e.g., S32->S16: 16 x 32-bit accumulator -> 16 x 16-bit output), each
half's `vector_srs` packs lanes densely in the low words of `[u32; 8]`.
Two halves concatenated fill only the lower portion of the Vec512. The
upper words will be zero. This matches hardware behavior where the
output x-register is partially filled for sub-32-bit SRS results.

### 5. Update narrow SRS/UPS handlers

The narrow handlers in `execute_half` already work for single-half
operations (w2b/bm variants). The only change needed is to use the
now-correct `from_type` and `element_type` values from the decoded
instruction, instead of relying on defaults.

No code change needed here -- the handlers already read `op.from_type`
and `op.element_type`. The fix is upstream (populating them correctly).

### 6. Use from_type in vector_srs for accumulator mode

`vector_srs()` currently ignores its `_from_type` parameter (leading
underscore). In S32 accumulator mode, each u64 lane stores a 32-bit
value in the low bits. The function casts `acc[i] as i64` which works
when upper bits are zero, but is fragile. With `from_type` now correctly
populated, `vector_srs` should mask the accumulator value to
`from_type.bits()` width before processing. This ensures S32 mode values
are properly bounded regardless of upper-bit state.

## Files Modified

| File | Change |
|------|--------|
| `src/tablegen/types.rs` | Add `Int64`, `UInt64` to `ElementType`; update `bits()`, `lanes_256()`, `is_signed()` |
| `src/tablegen/resolver.rs` | Add `from_type` to `InstrEncoding`; add `infer_dual_element_types()` |
| `src/interpreter/bundle/slot.rs` | Populate `SlotOp::from_type` from `InstrEncoding::from_type` during decode |
| `src/interpreter/execute/vector.rs` | Add `Srs` and `Ups` arms to `execute_wide()` |
| `src/interpreter/execute/vector_srs.rs` | Use `from_type` to mask accumulator values to correct width |

## What Does NOT Change
- Register file (`registers.rs`) -- `read_wide`/`write_wide` already exist
- Narrow SRS/UPS path in `execute_half` -- still used for w2b/bm variants
- Decoder bit-level logic -- encoding names already carry type info
- Any other SemanticOp handlers

## Testing

- Existing SRS/UPS unit tests continue to pass (they set from_type explicitly)
- New unit tests for `infer_dual_element_types()` covering all SRS/UPS variants
- New unit tests for wide SRS/UPS in execute_wide()
- ISA test sweep measures overall accuracy improvement
- Expected: SRS family 10% -> significant improvement, UPS family 20% -> significant improvement, x2c variants 0% -> functional
