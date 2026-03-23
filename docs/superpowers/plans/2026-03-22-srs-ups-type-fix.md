# SRS/UPS Type Inference and Wide Dispatch Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix SRS/UPS accuracy from 10-20% to near-correct by populating correct dual types at decode time and adding wide dispatch handlers.

**Architecture:** Add `Int64`/`UInt64` to `ElementType`, parse dual types from encoding names in the resolver, propagate `from_type` through the decode pipeline, add wide SRS/UPS handlers to `execute_wide()`, and use `from_type` to mask accumulator values in `vector_srs`.

**Tech Stack:** Rust, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-22-srs-ups-type-fix-design.md`

---

### Task 1: Add Int64/UInt64 to ElementType

**Files:**
- Modify: `src/tablegen/types.rs:632-670`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `src/tablegen/types.rs` (in the existing `#[cfg(test)]` module, or create one if none exists):

```rust
#[cfg(test)]
mod element_type_tests {
    use super::*;

    #[test]
    fn test_int64_bits() {
        assert_eq!(ElementType::Int64.bits(), 64);
        assert_eq!(ElementType::UInt64.bits(), 64);
    }

    #[test]
    fn test_int64_lanes() {
        assert_eq!(ElementType::Int64.lanes_256(), 4);
        assert_eq!(ElementType::UInt64.lanes_256(), 4);
    }

    #[test]
    fn test_int64_signed() {
        assert!(ElementType::Int64.is_signed());
        assert!(!ElementType::UInt64.is_signed());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib element_type_tests`
Expected: FAIL -- `Int64` and `UInt64` variants don't exist yet.

- [ ] **Step 3: Add the variants and update methods**

In `src/tablegen/types.rs`, add two variants to `ElementType` enum (before `BFloat16`):

```rust
pub enum ElementType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    /// 64-bit signed integer (accumulator lanes in S64 mode).
    Int64,
    /// 64-bit unsigned integer (accumulator lanes in D64 mode).
    UInt64,
    BFloat16,
    Float32,
}
```

Update `bits()`:
```rust
pub fn bits(self) -> u8 {
    match self {
        ElementType::Int8 | ElementType::UInt8 => 8,
        ElementType::Int16 | ElementType::UInt16 | ElementType::BFloat16 => 16,
        ElementType::Int32 | ElementType::UInt32 | ElementType::Float32 => 32,
        ElementType::Int64 | ElementType::UInt64 => 64,
    }
}
```

Update `is_signed()`:
```rust
pub fn is_signed(self) -> bool {
    matches!(self, ElementType::Int8 | ElementType::Int16 | ElementType::Int32 | ElementType::Int64)
}
```

`lanes_256()` needs no change -- it already computes `256 / bits()` generically.

- [ ] **Step 4: Fix exhaustive match compilation errors**

The new variants will cause `match` exhaustiveness errors across many files. The compiler will find every site. The pattern for each is straightforward -- add `Int64 | UInt64` to an existing arm or add a new no-op arm. Key affected files:

- `src/interpreter/execute/vector.rs`: `vector_srs` match on `to_type` (~line 1263) -- add arm identical to Int32 (one value per lane). `vector_convert` has a wildcard arm already.
- `src/interpreter/execute/vector_matmul.rs`: Several `match` blocks.
- `src/interpreter/execute/vector_matmul_sparse.rs`: Several `match` blocks.
- `src/interpreter/execute/vector_semantic.rs`: Several `match` blocks.
- `src/interpreter/execute/vector_config.rs`: Several `match` blocks.
- `src/interpreter/execute/vector_ups.rs`: Uses `bits()` generically, may not need changes.
- `src/fuzzer/runner.rs`: Random element type generation.
- `src/testing/test_cpp_parser.rs`: Has its own ElementType enum (separate, unaffected).

For most match blocks, `Int64 | UInt64` should be added to the `Int32 | UInt32` arm since 64-bit types represent accumulator lanes stored as one-value-per-u64 (same layout as 32-bit in the accumulator register file). Run `cargo check` iteratively to find and fix each site.

- [ ] **Step 5: Run tests to verify everything passes**

Run: `cargo test --lib`
Expected: All tests pass including the new `element_type_tests`.

- [ ] **Step 6: Commit**

```bash
git add src/tablegen/types.rs src/interpreter/execute/vector.rs src/interpreter/execute/vector_ups.rs
git commit -m "feat(types): add Int64/UInt64 to ElementType for SRS/UPS accumulator modes"
```

---

### Task 2: Add infer_dual_element_types and populate from_type at decode time

**Files:**
- Modify: `src/tablegen/resolver.rs:523-768` (add function, add field, wire into construction)
- Modify: `src/interpreter/decode/decoder.rs:853` (propagate to SlotOp)
- Modify: `src/interpreter/decode/decoder.rs:1618,1703,1829` (test fixtures)

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/tablegen/resolver.rs` (find the existing `#[cfg(test)] mod tests` block):

```rust
#[test]
fn test_infer_dual_element_types_srs() {
    // Standalone SRS: VSRS_{OUT}_{IN}_*
    let (et, ft) = infer_dual_element_types("VSRS_S16_S32_mv_w_srs");
    assert_eq!(et, Some(ElementType::Int16));
    assert_eq!(ft, Some(ElementType::Int32));

    let (et, ft) = infer_dual_element_types("VSRS_D32_S64_mv_x_srs");
    assert_eq!(et, Some(ElementType::UInt32));
    assert_eq!(ft, Some(ElementType::Int64));

    let (et, ft) = infer_dual_element_types("VSRS_S8_S32_mv_w_srs");
    assert_eq!(et, Some(ElementType::Int8));
    assert_eq!(ft, Some(ElementType::Int32));
}

#[test]
fn test_infer_dual_element_types_ups() {
    let (et, ft) = infer_dual_element_types("VUPS_S32_D16_mv_ups_w2b");
    assert_eq!(et, Some(ElementType::Int32));
    assert_eq!(ft, Some(ElementType::UInt16));

    let (et, ft) = infer_dual_element_types("VUPS_S64_S32_mv_ups_w2b");
    assert_eq!(et, Some(ElementType::Int64));
    assert_eq!(ft, Some(ElementType::Int32));

    let (et, ft) = infer_dual_element_types("VUPS_S32_S16_mv_ups_x2c");
    assert_eq!(et, Some(ElementType::Int32));
    assert_eq!(ft, Some(ElementType::Int16));
}

#[test]
fn test_infer_dual_element_types_fused() {
    // Fused load-UPS: VLDA_{dim}_UPS_{OUT}_{IN}
    let (et, ft) = infer_dual_element_types("VLDA_2D_UPS_S32_D16");
    assert_eq!(et, Some(ElementType::Int32));
    assert_eq!(ft, Some(ElementType::UInt16));

    // Fused store-SRS: VST_{dim}_SRS_{OUT}_{IN}
    let (et, ft) = infer_dual_element_types("VST_2D_SRS_D8_S32");
    assert_eq!(et, Some(ElementType::UInt8));
    assert_eq!(ft, Some(ElementType::Int32));
}

#[test]
fn test_infer_dual_element_types_non_srs_ups() {
    // Non-SRS/UPS instructions return (None, None)
    let (et, ft) = infer_dual_element_types("VADD_32");
    assert_eq!(et, None);
    assert_eq!(ft, None);

    let (et, ft) = infer_dual_element_types("VMAC_vmac_cm_core_dense");
    assert_eq!(et, None);
    assert_eq!(ft, None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib infer_dual_element_types`
Expected: FAIL -- function doesn't exist yet.

- [ ] **Step 3: Implement infer_dual_element_types**

Add this function near `infer_element_type` in `src/tablegen/resolver.rs` (around line 1191):

```rust
/// Parse a type token like "S16", "D32", "S64" into an ElementType.
fn parse_type_token(token: &str) -> Option<ElementType> {
    match token {
        "S8" => Some(ElementType::Int8),
        "D8" => Some(ElementType::UInt8),
        "S16" => Some(ElementType::Int16),
        "D16" => Some(ElementType::UInt16),
        "S32" => Some(ElementType::Int32),
        "D32" => Some(ElementType::UInt32),
        "S64" => Some(ElementType::Int64),
        "D64" => Some(ElementType::UInt64),
        _ => None,
    }
}

/// Infer both element types for dual-type instructions (SRS/UPS).
///
/// Encoding names follow two patterns:
/// - Standalone: `V{SRS|UPS}_{OUT}_{IN}_*`
/// - Fused: `V{LDA|ST}_{2D|3D}_{UPS|SRS}_{OUT}_{IN}*`
///
/// Returns `(element_type, from_type)` where element_type is the OUTPUT
/// type and from_type is the INPUT type. Returns `(None, None)` for
/// non-SRS/UPS instructions.
pub fn infer_dual_element_types(name: &str) -> (Option<ElementType>, Option<ElementType>) {
    let parts: Vec<&str> = name.split('_').collect();

    // Pattern 1: V{SRS|UPS}_{OUT}_{IN}_*
    // e.g., VSRS_S16_S32_mv_w_srs or VUPS_S32_D16_mv_ups_w2b
    if parts.len() >= 3 && (parts[0] == "VSRS" || parts[0] == "VUPS") {
        if let (Some(out_type), Some(in_type)) = (parse_type_token(parts[1]), parse_type_token(parts[2])) {
            return (Some(out_type), Some(in_type));
        }
    }

    // Pattern 2: V{LDA|ST}_{2D|3D}_{UPS|SRS}_{OUT}_{IN}*
    // e.g., VLDA_2D_UPS_S32_D16 or VST_2D_SRS_D8_S32
    if parts.len() >= 5 {
        let is_fused = (parts[0] == "VLDA" || parts[0] == "VST")
            && (parts[2] == "UPS" || parts[2] == "SRS");
        if is_fused {
            if let (Some(out_type), Some(in_type)) = (parse_type_token(parts[3]), parse_type_token(parts[4])) {
                return (Some(out_type), Some(in_type));
            }
        }
    }

    (None, None)
}
```

- [ ] **Step 4: Add from_type field to InstrEncoding**

In `src/tablegen/resolver.rs`, add this field after `element_type` (around line 523):

```rust
    /// Source element type for dual-type instructions (SRS/UPS).
    /// The output type goes in `element_type`; the input type goes here.
    /// None for single-type instructions.
    pub from_type: Option<ElementType>,
```

- [ ] **Step 5: Wire into InstrEncoding construction**

In the `resolve_instruction` method (around line 738), replace:

```rust
let element_type = infer_element_type(&instr.mnemonic);
```

with:

```rust
let (dual_et, dual_ft) = infer_dual_element_types(&instr.name);
let element_type = dual_et.or_else(|| infer_element_type(&instr.mnemonic));
let from_type = dual_ft;
```

And add `from_type,` to the `InstrEncoding { ... }` struct literal (after the `element_type,` line around 761).

- [ ] **Step 6: Propagate to SlotOp in decoder**

In `src/interpreter/decode/decoder.rs`, after line 853 (`slot_op.element_type = enc.element_type;`), add:

```rust
        slot_op.from_type = enc.from_type;
```

- [ ] **Step 7: Add from_type to test fixture InstrEncoding literals**

In `src/interpreter/decode/decoder.rs`, add `from_type: None,` to each test InstrEncoding literal. There are three locations (around lines 1618, 1703, 1829). Add it after the `element_type` field in each.

- [ ] **Step 8: Run tests**

Run: `cargo test --lib`
Expected: All tests pass, including the new `infer_dual_element_types` tests.

- [ ] **Step 9: Commit**

```bash
git add src/tablegen/resolver.rs src/interpreter/decode/decoder.rs
git commit -m "feat(decode): populate from_type for SRS/UPS dual-type instructions"
```

---

### Task 3: Use from_type in vector_srs for accumulator width masking

**Files:**
- Modify: `src/interpreter/execute/vector.rs:1240-1325` (`vector_srs` function)

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/interpreter/execute/vector.rs`:

```rust
#[test]
fn test_vector_srs_from_type_masks_accumulator() {
    // When from_type is Int32, only the low 32 bits of each u64
    // accumulator lane should be used. Upper bits should be ignored.
    let mut ctx = make_ctx();
    // Set acc lane 0: upper 32 bits = garbage, lower 32 bits = 100.
    // With saturation enabled and signed 16-bit output, value 100
    // should survive. But without masking, the garbage upper bits
    // (0xDEAD_BEEF) make the i64 value huge and saturation clamps
    // to i16::MAX (32767) instead of 100.
    ctx.accumulator.write(0, [
        0xDEAD_BEEF_0000_0064, // upper garbage, lower = 100
        0, 0, 0, 0, 0, 0, 0,
    ]);

    let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
        .as_vector(ElementType::Int16) // 16-bit output
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::AccumReg(0))
        .with_source(Operand::Immediate(0)); // shift=0
    op.from_type = Some(ElementType::Int32); // 32-bit accumulator mode

    // Floor rounding, saturation ON, signed output.
    // Without masking: i64 value is huge -> saturates to 32767.
    // With masking: value is 100 -> passes through as 100.
    ctx.srs_config.rounding_mode = 0;
    ctx.srs_config.saturation_mode = 1; // Saturate
    ctx.srs_config.srs_sign = true;

    VectorAlu::execute(&op, &mut ctx);
    let result = ctx.vector.read(0);
    let lo16 = result[0] as i16;
    assert_eq!(lo16, 100, "from_type=Int32 should mask to low 32 bits");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib test_vector_srs_from_type_masks_accumulator`
Expected: FAIL -- the garbage upper bits produce wrong output since `_from_type` is ignored.

- [ ] **Step 3: Implement accumulator masking**

In `src/interpreter/execute/vector.rs`, update the `vector_srs` function:

1. Rename `_from_type` to `from_type` (remove the leading underscore).
2. Add masking logic before the match on `to_type`. Convert the accumulator value to the correct input width using sign extension:

```rust
fn vector_srs(
    ctx: &ExecutionContext,
    acc_reg: u8,
    shift: u32,
    from_type: ElementType,
    to_type: ElementType,
) -> [u32; 8] {
    let acc = ctx.accumulator.read(acc_reg);
    let mut result = [0u32; 8];

    let cfg = &ctx.srs_config;
    let mode = RoundingMode::from_raw(cfg.rounding_mode)
        .unwrap_or(RoundingMode::PosInf);
    let saturate = cfg.saturate();
    let sym_sat = cfg.symmetric_saturate();

    let signed_output = to_type.is_signed();

    // Mask accumulator values to from_type width.
    // In S32 mode, only low 32 bits are meaningful.
    // In S64 mode (or when from_type is 64-bit), use the full u64.
    let from_bits = from_type.bits() as u32;
    let mask_value = |raw: u64| -> i64 {
        if from_bits >= 64 {
            raw as i64
        } else {
            // Sign-extend from from_bits width.
            let shift = 64 - from_bits;
            ((raw as i64) << shift) >> shift
        }
    };
    // ... rest of function uses mask_value(acc[i]) instead of acc[i] as i64
```

Then in each match arm, replace `acc[i] as i64` with `mask_value(acc[i])`. For example, the Int32 arm becomes:

```rust
ElementType::Int32 | ElementType::UInt32 => {
    for i in 0..8 {
        let val = mask_value(acc[i]);
        let out = vector_srs::srs_lane(
            val, shift, signed_output, 32,
            saturate, sym_sat, mode,
        );
        result[i] = out as u32;
    }
}
```

Apply the same change to the Int16 and Int8 arms (replace `acc[...] as i64` with `mask_value(acc[...])`).

The BFloat16 and Float32 arms use `f64::from_bits()` and don't need masking (they interpret the bits as float).

- [ ] **Step 4: Run tests**

Run: `cargo test --lib`
Expected: All tests pass including the new masking test. Existing SRS tests still pass (they store valid 32-bit values, so masking is a no-op).

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "fix(srs): use from_type to mask accumulator values to correct width"
```

---

### Task 4: Add wide SRS/UPS handlers to execute_wide()

**Files:**
- Modify: `src/interpreter/execute/vector.rs:3774-3949` (execute_wide match block)

- [ ] **Step 1: Write the failing tests**

Add to the test module in `src/interpreter/execute/vector.rs`:

```rust
#[test]
fn test_wide_ups_x2c() {
    // Wide UPS: x-register (512-bit) -> cm-register (1024-bit)
    // vups.s32.s16 cm0, x4, s0 (shift=0)
    let mut ctx = make_ctx();

    // Write 512-bit input to x4 (v4+v5): 32 x i16 packed
    // Low half (v4): lanes 0-15 = [1,2,3,...,16]
    let mut lo = [0u32; 8];
    for i in 0..8u32 {
        let lane0 = (i * 2 + 1) as u16;
        let lane1 = (i * 2 + 2) as u16;
        lo[i as usize] = (lane0 as u32) | ((lane1 as u32) << 16);
    }
    ctx.vector.write(4, lo);
    // High half (v5): lanes 16-31 = [17,18,...,32]
    let mut hi = [0u32; 8];
    for i in 0..8u32 {
        let lane0 = (i * 2 + 17) as u16;
        let lane1 = (i * 2 + 18) as u16;
        hi[i as usize] = (lane0 as u32) | ((lane1 as u32) << 16);
    }
    ctx.vector.write(5, hi);

    let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Ups)
        .as_vector(ElementType::Int32)
        .with_dest(Operand::AccumReg(0))
        .with_source(Operand::VectorReg(4))
        .with_source(Operand::Immediate(0)); // shift=0
    op.from_type = Some(ElementType::Int16);
    op.is_wide_vector = true;

    VectorAlu::execute(&op, &mut ctx);

    // Check low accumulator (acc0): should have lanes 0-7 = [1,2,...,8]
    let acc_lo = ctx.accumulator.read(0);
    for i in 0..8 {
        assert_eq!(acc_lo[i], (i as u64 + 1), "acc_lo lane {i}");
    }
    // Check high accumulator (acc1): should have lanes 8-15 = [17,18,...,24]
    let acc_hi = ctx.accumulator.read(1);
    for i in 0..8 {
        assert_eq!(acc_hi[i], (i as u64 + 17), "acc_hi lane {i}");
    }
}

#[test]
fn test_wide_srs_cm_to_x() {
    // Wide SRS: cm-register (1024-bit) -> x-register (512-bit)
    // vsrs.s32.s64 x4, cm0, s0 (shift=0)
    let mut ctx = make_ctx();

    // Write 1024-bit accumulator to cm0 (acc0+acc1)
    let acc_lo: [u64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let acc_hi: [u64; 8] = [90, 100, 110, 120, 130, 140, 150, 160];
    ctx.accumulator.write(0, acc_lo);
    ctx.accumulator.write(1, acc_hi);

    let mut op = SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::Srs)
        .as_vector(ElementType::Int32)
        .with_dest(Operand::VectorReg(4))
        .with_source(Operand::AccumReg(0))
        .with_source(Operand::Immediate(0)); // shift=0
    op.from_type = Some(ElementType::Int64);
    op.is_wide_vector = true;

    // Floor rounding, no saturation, signed
    ctx.srs_config.rounding_mode = 0;
    ctx.srs_config.saturation_mode = 0;
    ctx.srs_config.srs_sign = true;

    VectorAlu::execute(&op, &mut ctx);

    // Check output: x4 = v4+v5
    let v4 = ctx.vector.read(4);
    assert_eq!(v4, [10, 20, 30, 40, 50, 60, 70, 80]);
    let v5 = ctx.vector.read(5);
    assert_eq!(v5, [90, 100, 110, 120, 130, 140, 150, 160]);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_wide_ups_x2c test_wide_srs_cm_to_x`
Expected: FAIL -- wide SRS/UPS fall through to the broken fallback.

- [ ] **Step 3: Add wide SRS handler to execute_wide()**

In `src/interpreter/execute/vector.rs`, add these arms to the `execute_wide` match block, in the accumulator ops section (after the `Accumulate` arm, around line 3786):

```rust
            // ========== SRS/UPS (element-wise, split into halves) ==========

            SemanticOp::Srs => {
                // Wide SRS: read Acc1024, SRS each half, write Vec512.
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int64);
                let acc_wide = ctx.accumulator.read_wide(acc_reg);
                let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
                let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

                // Temporarily write each half to the base accumulator register
                // so vector_srs can read it via the standard path.
                let saved_lo = ctx.accumulator.read(acc_reg);
                let saved_hi = ctx.accumulator.read(acc_reg + 1);

                ctx.accumulator.write(acc_reg, acc_lo);
                let result_lo = Self::vector_srs(ctx, acc_reg, shift, from, et);
                ctx.accumulator.write(acc_reg, acc_hi);
                let result_hi = Self::vector_srs(ctx, acc_reg, shift, from, et);

                // Restore (the final write_wide will overwrite anyway, but
                // be safe in case vector_srs has side effects).
                ctx.accumulator.write(acc_reg, saved_lo);
                ctx.accumulator.write(acc_reg + 1, saved_hi);

                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&result_lo);
                result[8..].copy_from_slice(&result_hi);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            SemanticOp::Ups => {
                // Wide UPS: read Vec512, UPS each half, write Acc1024.
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int16);
                let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                let src_hi: [u32; 8] = src[8..].try_into().unwrap();

                let acc_lo = vector_ups::ups_vector_to_acc(&src_lo, shift, from, et);
                let acc_hi = vector_ups::ups_vector_to_acc(&src_hi, shift, from, et);

                match &op.dest {
                    Some(Operand::AccumReg(r)) => {
                        let mut acc_wide = [0u64; 16];
                        acc_wide[..8].copy_from_slice(&acc_lo);
                        acc_wide[8..].copy_from_slice(&acc_hi);
                        ctx.accumulator.write_wide(*r, acc_wide);
                    }
                    _ => {
                        log::warn!("[VECTOR_WIDE] UPS with non-AccumReg dest");
                        return Self::execute_wide_fallback(op, ctx, semantic, et);
                    }
                }
                true
            }
```

**Important:** The SRS handler temporarily writes accumulator halves to the base register so `vector_srs` (which takes an acc_reg index and reads from `ctx.accumulator`) can process each half. An alternative is to refactor `vector_srs` to take `&[u64; 8]` directly -- but that's a larger change. The temp-write approach is correct and simple. If it proves awkward, refactor later.

Actually, looking at `vector_srs` more carefully, it reads from `ctx.accumulator.read(acc_reg)` internally. Since we need to avoid mutating ctx (it's `&ExecutionContext` in the function signature, not `&mut`), we need a different approach. Let me check the signature.

The `vector_srs` function takes `ctx: &ExecutionContext` (immutable ref). We can't temporarily write to the accumulator. Instead, we should refactor `vector_srs` to accept `&[u64; 8]` directly. This is a small, clean change:

Replace the SRS handler above with:

```rust
            SemanticOp::Srs => {
                let acc_reg = Self::get_acc_source(op);
                let shift = Self::get_shift_amount(op, ctx);
                let from = op.from_type.unwrap_or(ElementType::Int64);
                let acc_wide = ctx.accumulator.read_wide(acc_reg);
                let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
                let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

                let result_lo = Self::vector_srs_from_acc(&acc_lo, shift, from, et, &ctx.srs_config);
                let result_hi = Self::vector_srs_from_acc(&acc_hi, shift, from, et, &ctx.srs_config);

                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&result_lo);
                result[8..].copy_from_slice(&result_hi);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
```

And extract the core SRS logic into a new function `vector_srs_from_acc` that takes `&[u64; 8]` instead of reading from the register file. The existing `vector_srs` calls into it:

```rust
    fn vector_srs_from_acc(
        acc: &[u64; 8],
        shift: u32,
        from_type: ElementType,
        to_type: ElementType,
        cfg: &SrsConfig,
    ) -> [u32; 8] {
        // ... (same body as current vector_srs, but reads from `acc` parameter
        // instead of ctx.accumulator.read(acc_reg))
    }

    fn vector_srs(
        ctx: &ExecutionContext,
        acc_reg: u8,
        shift: u32,
        from_type: ElementType,
        to_type: ElementType,
    ) -> [u32; 8] {
        let acc = ctx.accumulator.read(acc_reg);
        Self::vector_srs_from_acc(&acc, shift, from_type, to_type, &ctx.srs_config)
    }
```

- [ ] **Step 4: Extract vector_srs_from_acc**

Move the body of `vector_srs` into a new `vector_srs_from_acc` method that takes `acc: &[u64; 8]` and `cfg: &SrsConfig` directly. Make `vector_srs` a thin wrapper that reads the accumulator and calls `vector_srs_from_acc`.

- [ ] **Step 5: Run tests**

Run: `cargo test --lib`
Expected: All tests pass including the two new wide SRS/UPS tests.

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "feat(vector): add wide SRS/UPS handlers to execute_wide()"
```

---

### Task 5: Verify and run ISA sweep

- [ ] **Step 1: Run full test suite**

Run: `cargo test --lib`
Expected: All tests pass.

- [ ] **Step 2: Run ISA accuracy sweep**

Run: `nice -n 19 bash scripts/isa-test.sh`

Compare SRS/UPS batch accuracy against the baseline:
- Baseline SRS: 10.2% → expect significant improvement
- Baseline UPS: 20.6% → expect significant improvement
- x2c variants: 0% → expect functional

- [ ] **Step 3: Commit any fixups**

If any issues surface during the sweep, fix and commit.
