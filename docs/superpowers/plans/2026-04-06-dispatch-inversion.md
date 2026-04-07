# Vector Dispatch Inversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Invert vector dispatch so operations own their semantics and dispatch is a trivial one-line-per-op routing table.

**Architecture:** Extract helper methods to `vector_helpers.rs`, create `vector_dispatch.rs` with a single match table, then move each operation's match-arm logic into `execute_<op>()` functions in the file that already owns its compute function. Operations handle both narrow and wide paths internally. `vector.rs` is deleted when empty.

**Tech Stack:** Rust, cargo test, ISA test harness (`scripts/isa-test.sh`)

---

## Guiding Principles

1. **One operation at a time.** Extract one SemanticOp, run `cargo test --lib`, commit. Never batch-move operations without testing between each.
2. **Pure move, no behavior changes.** Copy the match arm logic verbatim into the new `execute_<op>()` function. Do not refactor, simplify, or "improve" the logic during extraction.
3. **Both widths in one function.** Each `execute_<op>()` handles narrow and wide internally. The narrow/wide check from the old `execute()` entry point moves into the operation.
4. **Protect 100%.** Run `cargo test --lib` after every structural change. Run `nice -n 19 ./scripts/isa-test.sh` after completing each major task. Expected: 2660+ tests, 4815/4815 ISA.

## File Map

### Files to Create

| File | Purpose |
|------|---------|
| `src/interpreter/execute/vector_dispatch.rs` | Single match table (~150 lines), the only `pub fn execute()` |
| `src/interpreter/execute/vector_helpers.rs` | Shared VectorAlu methods: operand access, dest writes, masks, bridges, bf16 |
| `src/interpreter/execute/vector_convert.rs` | `execute_convert()`, `vector_convert()`, `vector_srs_from_acc()` pipeline |

### Files to Modify

| File | Change |
|------|--------|
| `src/interpreter/execute/vector_arith.rs` | Add `execute_add`, `execute_sub`, ..., `execute_accumulate` (18 functions) |
| `src/interpreter/execute/vector_compare.rs` | Add `execute_cmp`, `execute_setge`, ..., `execute_select` (7 functions) |
| `src/interpreter/execute/vector_misc.rs` | Add `execute_shuffle`, `execute_copy`, ..., `execute_clear` (14 functions) |
| `src/interpreter/execute/vector_srs.rs` | Add `execute_srs()` wrapping existing SRS pipeline |
| `src/interpreter/execute/vector_ups.rs` | Add `execute_ups()` wrapping existing UPS pipeline |
| `src/interpreter/execute/vector_pack.rs` | Add `execute_pack()`, `execute_unpack()` |
| `src/interpreter/execute/mod.rs` | Replace `mod vector;` with `mod vector_dispatch;` + `mod vector_helpers;` + `mod vector_convert;` |

### Files to Delete

| File | Reason |
|------|--------|
| `src/interpreter/execute/vector.rs` | All contents distributed to other files |

---

## Task 1: Extract Helper Methods to vector_helpers.rs

This is the foundation. Move all shared helper methods out of vector.rs into a dedicated file. Zero behavioral change -- same methods, same struct, different file.

**Files:**
- Create: `src/interpreter/execute/vector_helpers.rs`
- Modify: `src/interpreter/execute/vector.rs` (remove helpers)
- Modify: `src/interpreter/execute/mod.rs` (add module)

- [ ] **Step 1: Create vector_helpers.rs with all shared helpers**

Read vector.rs and extract every helper method that is NOT a dispatch function, NOT an operation implementation, and NOT a test. These are the methods that multiple operation files will need.

The file should contain:

```rust
//! Shared helper methods for vector ALU operations.
//!
//! Operand access, result writing, comparison masks, wide bridges,
//! element manipulation, and conversion utilities. Used by all
//! vector operation files.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::{ExecutionContext, Vec512, Acc1024};
use crate::tablegen::SemanticOp;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    // --- Operand access ---
    // get_two_vector_sources, get_vector_source, read_vector_operand,
    // get_wide_vec_source, get_two_wide_vec_sources,
    // get_scalar_source, get_scalar_source_64, get_nth_scalar_source,
    // get_lane_index, get_shift_amount, get_config_register,
    // get_acc_source, get_acc_dest

    // --- Result writing ---
    // write_vector_dest, write_wide_vec_dest, write_wide_acc_dest,
    // write_scalar_dest, write_scalar_dest_wide,
    // write_cmp_dest, write_cmp_dest_wide

    // --- Comparison masks ---
    // condense_comparison_mask, pack_comparison_flags

    // --- Wide bridges ---
    // wide_element_wise_unary, wide_element_wise_binary,
    // execute_wide_fallback, increment_vector_regs

    // --- Wide element manipulation ---
    // wide_vector_push, extract_wide_element,
    // insert_wide_element, insert_wide_element_64,
    // wide_vector_shift

    // --- Conversion utilities ---
    // bf16_to_f32, f32_to_bf16

    // --- Pack/Unpack (simple wrappers) ---
    // vector_pack, vector_unpack_low
}
```

Copy each function verbatim from vector.rs (lines 622-757, 1036-1098, 1372-1387, 1392-1529, 1539-1643, 2687-2921, 1062-1070). Keep all doc comments. Change visibility to `pub(super)` for methods that are currently private but will be called from other files.

- [ ] **Step 2: Add module declaration**

In `src/interpreter/execute/mod.rs`, add after the existing `mod vector;` line:

```rust
mod vector_helpers;
```

- [ ] **Step 3: Remove helpers from vector.rs**

Delete all the functions that were copied to vector_helpers.rs from vector.rs. Leave the dispatch functions (`execute`, `execute_half`, `execute_wide`), accumulator ops (`execute_acc_add_sub`, `execute_acc_negate`), SRS/Convert pipelines (`vector_srs`, `vector_srs_from_acc`, `vector_convert`), and the test module.

- [ ] **Step 4: Fix imports and verify**

Update `use` statements in vector.rs (remove any imports that are only needed by the moved helpers). Add any missing imports to vector_helpers.rs.

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

Expected: 2660+ passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/execute/vector_helpers.rs src/interpreter/execute/vector.rs \
       src/interpreter/execute/mod.rs
git commit -m "refactor: extract shared helpers to vector_helpers.rs"
```

---

## Task 2: Create vector_dispatch.rs (Scaffold)

Create the dispatch file with a single `execute()` function. Initially it still delegates to the old `execute_half`/`execute_wide` -- this is a pure structural move of the entry point.

**Files:**
- Create: `src/interpreter/execute/vector_dispatch.rs`
- Modify: `src/interpreter/execute/vector.rs` (make execute_half/execute_wide `pub(super)`)
- Modify: `src/interpreter/execute/mod.rs` (swap module declaration)

- [ ] **Step 1: Create vector_dispatch.rs**

```rust
//! Vector ALU dispatch table.
//!
//! Single entry point that routes SemanticOp variants to operation
//! implementations. Each match arm is a single function call.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::tablegen::SemanticOp;

/// Vector ALU execution unit.
pub struct VectorAlu;

impl VectorAlu {
    /// Execute a vector operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a vector op.
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        let Some(semantic) = op.semantic else {
            return false;
        };

        if !op.is_vector {
            return false;
        }

        // Skip fused ops that have memory operands -- MemoryUnit handles these.
        if op.slot.is_memory() && matches!(semantic,
            SemanticOp::Ups | SemanticOp::Srs | SemanticOp::Pack
            | SemanticOp::Unpack | SemanticOp::Convert)
            && op.sources.iter().any(|s| matches!(s, Operand::Memory { .. }))
        {
            return false;
        }

        let et = op.element_type.unwrap_or(ElementType::Int32);

        log::trace!("[VECTOR_ALU] Checking semantic={:?} element_type={:?} dest={:?}",
            semantic, op.element_type, op.dest);

        // Temporary: delegate to old dispatch while operations are being extracted.
        // As each operation gets its own execute_<op>(), it replaces its arm here.
        let has_wide_acc_source = matches!(op.accum_width,
            Some(crate::tablegen::decoder_ffi::AccumWidth::Full)
            | Some(crate::tablegen::decoder_ffi::AccumWidth::Half));
        let is_accum_only = matches!(semantic,
            SemanticOp::Accumulate | SemanticOp::AccumSub
            | SemanticOp::AccumNegAdd | SemanticOp::AccumNegSub
            | SemanticOp::Neg | SemanticOp::NegAdd) && {
            let has_acc_source = op.sources.iter()
                .any(|s| matches!(s, Operand::AccumReg(_)));
            let has_vec_source = op.sources.iter()
                .any(|s| matches!(s, Operand::VectorReg(_)));
            has_acc_source && !has_vec_source
        };

        if op.is_wide_vector || is_accum_only || has_wide_acc_source {
            Self::execute_wide(op, ctx, semantic, et)
        } else {
            Self::execute_half(op, ctx, semantic, et)
        }
    }
}
```

This is a temporary scaffold. It preserves the existing half/wide routing while we incrementally extract operations.

- [ ] **Step 2: Move VectorAlu struct definition**

Remove `pub struct VectorAlu;` from vector.rs. It now lives in vector_dispatch.rs. Update the `use super::vector_dispatch::VectorAlu;` import in vector_helpers.rs if needed (or use `use super::vector::VectorAlu;` -> `use super::vector_dispatch::VectorAlu;`).

- [ ] **Step 3: Make execute_half and execute_wide pub(super)**

In vector.rs, change:
```rust
fn execute_half(
```
to:
```rust
pub(super) fn execute_half(
```

Same for `execute_wide`. The dispatch file needs to call them.

- [ ] **Step 4: Update mod.rs**

Replace `mod vector;` with:
```rust
mod vector_dispatch;
mod vector;  // temporary: still has execute_half/execute_wide + SRS/Convert
```

Update any re-exports if needed (check if `VectorAlu` is re-exported from mod.rs).

- [ ] **Step 5: Update all files that import VectorAlu**

Search for `use super::vector::VectorAlu` or `use crate::interpreter::execute::vector::VectorAlu` across the codebase and update to point to `vector_dispatch`.

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

Expected: 2660+ passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/
git commit -m "refactor: create vector_dispatch.rs scaffold with execute() entry point"
```

---

## Task 3: Extract Arithmetic Operations

Move the match arm logic for each arithmetic SemanticOp into `execute_<op>()` functions in vector_arith.rs. Each function handles both narrow and wide paths.

**Files:**
- Modify: `src/interpreter/execute/vector_arith.rs` (add execute_ functions)
- Modify: `src/interpreter/execute/vector_dispatch.rs` (update match arms)
- Modify: `src/interpreter/execute/vector.rs` (remove arms from execute_half/execute_wide)

- [ ] **Step 1: Add execute_add to vector_arith.rs**

Read the Add arms from both `execute_half` (lines ~130-157) and `execute_wide` (lines ~1659-1703) in vector.rs. Combine them into a single function:

```rust
impl VectorAlu {
    /// VADD / VADDSUB: element-wise vector addition.
    ///
    /// Two variants:
    /// 1. VADD: simple vector add (2 sources)
    /// 2. VADDSUB: conditional add/subtract (2 vector + 1 scalar selector)
    pub(super) fn execute_add(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        // Copy the narrow arm logic AND the wide arm logic here,
        // gated on op.is_wide_vector or the accum_width check.
        // ...verbatim from execute_half::Add and execute_wide::Add...
        true
    }
}
```

The function body combines both paths with an `if op.is_wide_vector { ... } else { ... }` guard. Copy the match arm bodies verbatim -- do NOT simplify.

- [ ] **Step 2: Update dispatch to call execute_add**

In vector_dispatch.rs, replace the temporary delegation for Add with a direct call. Since the scaffold still routes everything through execute_half/execute_wide, the simplest approach: remove the Add arms from execute_half and execute_wide in vector.rs, and add Add to a new match block in vector_dispatch.rs before the fallback:

```rust
// In vector_dispatch.rs execute(), add before the half/wide delegation:
match semantic {
    SemanticOp::Add => return Self::execute_add(op, ctx, et),
    _ => {}
}
// ... existing half/wide fallback ...
```

- [ ] **Step 3: Test**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

Expected: 2660+ passed, 0 failed.

- [ ] **Step 4: Repeat for Sub, Mul, Min, Max, Neg**

These are the simple binary/unary arithmetic ops. Each follows the same pattern:
- Read the narrow arm from execute_half
- Read the wide arm from execute_wide
- Combine into `execute_<op>(op, ctx, et) -> bool`
- Add to the dispatch match
- Remove the arms from execute_half/execute_wide
- Run `cargo test --lib` after each

Note: `Neg` has a special case -- if the source is an AccumReg (accumulator-only negate), it calls `execute_acc_negate`. Read the arms carefully and preserve this branching.

- [ ] **Step 5: Extract NegAdd, shift ops (Shl, Srl, Sra)**

Same pattern. Shift ops read a shift amount from operands.

- [ ] **Step 6: Extract conditional arithmetic (AbsGtz, NegGtz, NegLtz, SubLt, SubGe, MaxDiffLt)**

Same pattern. These write to both a vector dest and a comparison dest.

- [ ] **Step 7: Extract Accumulate**

The Accumulate arm handles 4 SemanticOp variants (`Accumulate | AccumSub | AccumNegAdd | AccumNegSub`). It has a complex branch: if the op has an AccumReg source, it calls `execute_acc_add_sub`; otherwise it does legacy vector accumulation. Move `execute_acc_add_sub` and `execute_acc_negate` into vector_arith.rs as well.

- [ ] **Step 8: Move arithmetic tests**

Move tests from vector.rs's `#[cfg(test)] mod tests` to vector_arith.rs:
- `test_vector_add_i32`, `test_vector_add_i16`
- `test_vector_sub`, `test_vector_mul`
- `test_vector_min_max`
- `test_vector_add_f32`, `test_vector_mul_f32`, `test_vector_min_max_f32`
- `test_vector_add_bf16`
- `test_vadd_f_nan_canonical`, `test_vadd_f_denormal_ftz`

- [ ] **Step 9: Test and commit**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

```bash
git add src/interpreter/execute/
git commit -m "refactor: extract arithmetic operations to own dispatch in vector_arith.rs"
```

---

## Task 4: Extract Comparison Operations

**Files:**
- Modify: `src/interpreter/execute/vector_compare.rs`
- Modify: `src/interpreter/execute/vector_dispatch.rs`
- Modify: `src/interpreter/execute/vector.rs`

- [ ] **Step 1: Add execute_cmp, execute_setge, execute_setlt, execute_seteq**

Read the Cmp/SetGe/SetLt/SetEq arms from both execute_half and execute_wide. The wide path for comparisons condenses to a scalar bitmask (32-bit or 64-bit for 8-bit elements). Preserve this logic exactly.

- [ ] **Step 2: Add execute_maxlt, execute_minge**

These combine a min/max operation with a comparison flag write.

- [ ] **Step 3: Add execute_select**

VectorSelect has a complex wide path with 64-bit selector splitting for 8-bit elements. Copy verbatim.

- [ ] **Step 4: Update dispatch, remove old arms, test**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

- [ ] **Step 5: Move comparison tests**

Move from vector.rs tests:
- `test_vector_cmp`
- `test_wide_setge_scalar_dest_i32`
- `test_wide_setlt_scalar_dest_i32`

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/
git commit -m "refactor: extract comparison operations to own dispatch in vector_compare.rs"
```

---

## Task 5: Extract Misc Operations (Data Movement, Bitwise, Pack)

**Files:**
- Modify: `src/interpreter/execute/vector_misc.rs`
- Modify: `src/interpreter/execute/vector_pack.rs`
- Modify: `src/interpreter/execute/vector_dispatch.rs`
- Modify: `src/interpreter/execute/vector.rs`

- [ ] **Step 1: Add execute_shuffle to vector_misc.rs**

Shuffle is wide-only (the narrow arm returns false). The wide path calls `super::vector_permute::shuffle_vectors()`. Also handles VBCSTSHFL variant (broadcast + shuffle).

- [ ] **Step 2: Add execute_broadcast, execute_extract, execute_insert, execute_push**

These have significant wide-path logic (cross-boundary element operations). Copy verbatim from execute_wide arms.

- [ ] **Step 3: Add execute_align**

Wide VSHIFT with pre-shift stage. Uses `wide_vector_shift` helper.

- [ ] **Step 4: Add execute_copy, execute_clear**

Copy handles both vector and accumulator moves. Clear zeros a register.

- [ ] **Step 5: Add execute_and, execute_or, execute_xor, execute_not**

The wide path for bitwise ops manually splits halves (no bridge). Copy verbatim.

- [ ] **Step 6: Add execute_pack, execute_unpack to vector_pack.rs**

Pack (512->256 with split/rejoin in wide) and Unpack (256->512 expansion).

- [ ] **Step 7: Update dispatch, remove old arms, test**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

- [ ] **Step 8: Move misc tests**

Move from vector.rs tests:
- `test_vector_shuffle_mode0`, `test_vector_shuffle_overflow_mode`
- `test_wide_vector_push_lo_32`, `test_wide_vector_push_hi_32`
- `test_wide_vector_push_lo_64`, `test_wide_vector_push_hi_64`
- `test_extract_wide_element_high_half`, `test_extract_wide_element_16bit`
- `test_insert_wide_element_high_half`, `test_insert_wide_element_16bit`
- `test_insert_extract_roundtrip`
- `test_wide_vector_align_no_shift`, `test_wide_vector_align_full_shift`, `test_wide_vector_align_cross_boundary`
- `test_vector_mov`

- [ ] **Step 9: Commit**

```bash
git add src/interpreter/execute/
git commit -m "refactor: extract misc/bitwise/pack operations to own dispatch"
```

---

## Task 6: Extract SRS, UPS, and Convert Pipelines

These are the most complex operations. SRS and UPS already have dedicated files but don't own their dispatch. Convert needs a new file.

**Files:**
- Create: `src/interpreter/execute/vector_convert.rs`
- Modify: `src/interpreter/execute/vector_srs.rs`
- Modify: `src/interpreter/execute/vector_ups.rs`
- Modify: `src/interpreter/execute/vector_dispatch.rs`
- Modify: `src/interpreter/execute/vector.rs`
- Modify: `src/interpreter/execute/mod.rs`

- [ ] **Step 1: Create vector_convert.rs**

Move `vector_convert()` (the compute function, ~100 lines) and `vector_srs_from_acc()` (~130 lines) from vector.rs into a new file. Also move `vector_floor_bf16_to_s32` from vector_arith.rs (it's a conversion, not arithmetic).

Add `execute_convert()` that combines the narrow and wide Convert arms. The wide Convert arm is the most complex single arm in the file (~100+ lines with quarter/half/full accumulator paths, expansion/contraction). Copy it verbatim.

```rust
//! Type conversion operations (VCONV, VFLOOR).

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;

use super::vector_dispatch::VectorAlu;

impl VectorAlu {
    pub(super) fn execute_convert(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        // ... narrow and wide Convert logic from execute_half/execute_wide ...
        true
    }

    // vector_convert(), vector_srs_from_acc(), vector_floor_bf16_to_s32()
    // moved here from vector.rs
}
```

- [ ] **Step 2: Add execute_srs to vector_srs.rs**

Move the SRS dispatch logic from both execute_half and execute_wide into vector_srs.rs. The wide SRS path has AccumWidth-aware reads (Full=1024-bit, Half=512-bit).

```rust
impl VectorAlu {
    pub(super) fn execute_srs(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        et: ElementType,
    ) -> bool {
        // ... narrow and wide SRS logic ...
        true
    }
}
```

- [ ] **Step 3: Add execute_ups to vector_ups.rs**

Move UPS dispatch logic. Wide UPS has two paths: w2c (narrow src -> wide acc) and x2c (wide src -> wide acc).

- [ ] **Step 4: Add module declaration for vector_convert**

In mod.rs:
```rust
mod vector_convert;
```

- [ ] **Step 5: Update dispatch, remove old arms, test**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

- [ ] **Step 6: Move SRS/Convert/UPS tests**

Move from vector.rs tests:
- `test_vector_srs_int32`, `test_vector_ups_srs_roundtrip`
- `test_wide_ups_srs_s8_s32_roundtrip`
- `test_srs_from_acc32_to_d16`, `test_srs_from_acc32_to_d8`
- `test_vector_srs_reads_config`, `test_vector_srs_saturation_from_config`
- `test_vector_srs_from_type_masks_accumulator`
- `test_wide_ups_x2c`, `test_wide_srs_cm_to_x`
- `test_vector_convert_bf16_to_f32`, `test_vector_convert_int32_to_f32`
- `test_vconv_bf16_fp32_acc_source`, `test_vconv_fp32_bf16_acc_dest`
- `test_vfloor_bf16_to_s32_shift0`, `test_vfloor_signed_6bit_shift`, `test_vfloor_nan_saturates_to_int_max`

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/execute/
git commit -m "refactor: extract SRS/UPS/Convert pipelines to own dispatch"
```

---

## Task 7: Collapse Dispatch and Delete vector.rs

All operations have been extracted. The scaffold dispatch can be simplified to the final clean match table, and vector.rs should be empty.

**Files:**
- Modify: `src/interpreter/execute/vector_dispatch.rs` (final form)
- Delete: `src/interpreter/execute/vector.rs`
- Modify: `src/interpreter/execute/mod.rs` (remove `mod vector;`)

- [ ] **Step 1: Verify vector.rs is empty**

At this point, vector.rs should contain only:
- Empty `execute_half` and `execute_wide` functions (all arms removed)
- An empty `#[cfg(test)] mod tests` block (all tests moved)

If anything remains, it was missed in earlier tasks -- move it now.

- [ ] **Step 2: Rewrite vector_dispatch.rs to final form**

Remove the temporary scaffold (the half/wide delegation fallback). Replace with the clean match table from the spec:

```rust
match semantic {
    // Arithmetic
    SemanticOp::Add => Self::execute_add(op, ctx, et),
    SemanticOp::Sub => Self::execute_sub(op, ctx, et),
    // ... all operations, one line each ...
    _ => false,
}
```

Remove the `has_wide_acc_source` / `is_accum_only` logic from dispatch -- each operation now handles this internally.

- [ ] **Step 3: Delete vector.rs**

```bash
git rm src/interpreter/execute/vector.rs
```

- [ ] **Step 4: Remove mod vector from mod.rs**

Delete the `mod vector;` line. All other module declarations should already be present.

- [ ] **Step 5: Fix any remaining imports**

Search for references to `super::vector::` or `vector::VectorAlu` and update to `vector_dispatch::VectorAlu` or just `VectorAlu` if re-exported.

- [ ] **Step 6: Run full test suite**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

Expected: 2660+ passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add -A src/interpreter/execute/
git commit -m "refactor: collapse dispatch to clean match table, delete vector.rs"
```

---

## Task 8: Final Verification Gate

- [ ] **Step 1: Unit tests**

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib -- -q
```

Expected: 2660+ pass, 0 fail.

- [ ] **Step 2: ISA accuracy**

```bash
nice -n 19 ./scripts/isa-test.sh
```

Expected: 4815/4815 (100.0%).

- [ ] **Step 3: Verify dispatch is clean**

```bash
wc -l src/interpreter/execute/vector_dispatch.rs
```

Expected: under 200 lines.

```bash
grep -c 'Self::execute_\|vector_matmul::execute_matmul' \
    src/interpreter/execute/vector_dispatch.rs
```

Expected: equals the number of match arms (every arm is a single function call).

- [ ] **Step 4: Verify vector.rs is gone**

```bash
test ! -f src/interpreter/execute/vector.rs && echo "DELETED"
```

Expected: DELETED.

- [ ] **Step 5: No new warnings**

```bash
cargo build --lib 2>&1 | grep "^warning:" | grep -v llvm | head -5
```

Expected: no Rust warnings.

- [ ] **Step 6: Commit summary**

```bash
git log --oneline HEAD~8..HEAD
```

Review the commit chain. All 7+ commits should tell a coherent story of incremental extraction.

---

## Execution Order & Risk Assessment

| Task | Risk | Dependencies |
|------|------|-------------|
| 1. Extract helpers | Low | None |
| 2. Create dispatch scaffold | Low | Task 1 |
| 3. Arithmetic operations | Medium | Task 2 |
| 4. Comparison operations | Medium | Task 2 |
| 5. Misc operations | Medium | Task 2 |
| 6. SRS/UPS/Convert | High | Task 2 |
| 7. Collapse dispatch | Low | Tasks 3-6 |
| 8. Final verification | Zero | Task 7 |

Tasks 3, 4, and 5 are independent of each other (they extract different SemanticOps). They all depend on Task 2. Task 6 is highest risk because Convert has the most complex wide-path logic. Task 7 is the payoff -- once all ops are extracted, the cleanup is mechanical.
