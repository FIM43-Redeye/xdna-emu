# ISA Accuracy Hard Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 categories of ISA test failures to push test-point pass rate from 52.2% (2515/4815) toward 65%+.

**Architecture:** Each task fixes one root-cause category. Fixes are independent and can be committed separately. All changes are in `src/interpreter/execute/` (runtime) and `build_helpers/extract.rs` (build-time semantic mapping). Build-time changes require `cargo build` to regenerate the compiled-in tables.

**Tech Stack:** Rust, build.rs TableGen pipeline, ISA test harness (`scripts/isa-test.sh`)

---

## File Map

| File | Role | Tasks |
|------|------|-------|
| `build_helpers/extract.rs` | Build-time semantic mapping from intrinsics/itineraries | 1, 4 |
| `src/interpreter/execute/vector.rs` | Vector ALU dispatch and handlers | 1, 2, 3 |
| `src/interpreter/execute/vector_ups.rs` | UPS (upshift) implementation | 2 |
| `src/interpreter/execute/vector_pack.rs` | Pack/unpack implementation | 3 |

---

### Task 1: Fix VGE/VLT Semantic Mapping (~154 test points)

VGE and VLT are vector comparison instructions that produce a scalar bitmask.
They are currently mapped to `MinGe`/`MaxLt` (vector min/max), which computes
the wrong operation AND writes to vector dest (discarding the scalar result).

**Root cause:** `semantic_from_intrinsic()` in `build_helpers/extract.rs` maps
`vlt` -> `MaxLt` and `vge` -> `MinGe`. These prefixes collide with the
intended `vmax_lt` / `vmin_ge` mappings. The `vge`/`vlt` intrinsics are
comparisons, not min/max.

**Additionally:** The `execute_wide` SetGe/SetLt handlers write to
`write_wide_vec_dest` which only handles VectorReg. VGE/VLT have ScalarReg
dest. The `execute_half` path's `write_vector_dest` already handles ScalarReg
via `condense_comparison_mask`, but the wide path needs the same treatment.

**Files:**
- Modify: `build_helpers/extract.rs:574-578`
- Modify: `src/interpreter/execute/vector.rs:3886-3897` (execute_wide SetGe/SetLt)

- [ ] **Step 1: Write a test for the build-time semantic fix**

In `build_helpers/extract.rs`, find the `#[cfg(test)]` section. Add:

```rust
#[test]
fn test_vge_vlt_semantic_mapping() {
    // vge/vlt are comparisons, not min/max
    assert_eq!(semantic_from_intrinsic("int_aie2_vge8"), Some("SemanticOp::SetGe".to_string()));
    assert_eq!(semantic_from_intrinsic("int_aie2_vge16"), Some("SemanticOp::SetGe".to_string()));
    assert_eq!(semantic_from_intrinsic("int_aie2_vge32"), Some("SemanticOp::SetGe".to_string()));
    assert_eq!(semantic_from_intrinsic("int_aie2_vgebf16"), Some("SemanticOp::SetGe".to_string()));
    assert_eq!(semantic_from_intrinsic("int_aie2_vlt8"), Some("SemanticOp::SetLt".to_string()));
    assert_eq!(semantic_from_intrinsic("int_aie2_vlt16"), Some("SemanticOp::SetLt".to_string()));
    // vmax_lt and vmin_ge should still map to MaxLt/MinGe
    assert_eq!(semantic_from_intrinsic("int_aie2_vmax_lt8"), Some("SemanticOp::MaxLt".to_string()));
    assert_eq!(semantic_from_intrinsic("int_aie2_vmin_ge8"), Some("SemanticOp::MinGe".to_string()));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && cargo test --lib test_vge_vlt_semantic_mapping 2>&1 | tail -5`
Expected: FAIL -- vge returns `MinGe`, vlt returns `MaxLt`

- [ ] **Step 3: Fix the semantic mapping**

In `build_helpers/extract.rs`, reorder the prefix checks so `vge`/`vlt`
(without `vmin_ge`/`vmax_lt` prefix) map to SetGe/SetLt. The `vmax_lt` and
`vmin_ge` checks are already above on lines 565-573, so they match first.
Change lines 574-578:

```rust
    // vlt (standalone) is a comparison, NOT max. vmax_lt (matched above) IS max.
    if stem.starts_with("vlt") {
        return Some("SemanticOp::SetLt".to_string());
    }
    // vge (standalone) is a comparison, NOT min. vmin_ge (matched above) IS min.
    if stem.starts_with("vge") {
        return Some("SemanticOp::SetGe".to_string());
    }
```

Remove the `|| stem.starts_with("vmax_lt")` and `|| stem.starts_with("vmin_ge")`
from these lines (those are already handled by the checks above at lines 565-573).

- [ ] **Step 4: Run the build-time test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && cargo test --lib test_vge_vlt_semantic_mapping 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Fix execute_wide SetGe/SetLt to handle ScalarReg dest**

In `src/interpreter/execute/vector.rs`, replace the SetGe handler in
`execute_wide` (around line 3886) with:

```rust
            SemanticOp::SetGe => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_ge);
                // VGE writes a scalar bitmask, not a vector result.
                if matches!(op.dest, Some(Operand::ScalarReg(_))) {
                    // Condense both halves into one bitmask.
                    let lo: [u32; 8] = result[..8].try_into().unwrap();
                    let hi: [u32; 8] = result[8..].try_into().unwrap();
                    let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                    let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                    let lanes_per_half = 256 / et.bits() as u32;
                    let full_mask = mask_lo | (mask_hi << lanes_per_half);
                    Self::write_scalar_dest(op, ctx, full_mask);
                } else {
                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }
```

Apply the same pattern for SetLt (around line 3892):

```rust
            SemanticOp::SetLt => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_lt);
                if matches!(op.dest, Some(Operand::ScalarReg(_))) {
                    let lo: [u32; 8] = result[..8].try_into().unwrap();
                    let hi: [u32; 8] = result[8..].try_into().unwrap();
                    let mask_lo = Self::condense_comparison_mask(&lo, op.element_type);
                    let mask_hi = Self::condense_comparison_mask(&hi, op.element_type);
                    let lanes_per_half = 256 / et.bits() as u32;
                    let full_mask = mask_lo | (mask_hi << lanes_per_half);
                    Self::write_scalar_dest(op, ctx, full_mask);
                } else {
                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }
```

- [ ] **Step 6: Write a unit test for wide comparison bitmask**

In vector.rs tests section, add:

```rust
#[test]
fn test_wide_compare_ge_scalar_bitmask() {
    use crate::interpreter::bundle::*;
    use crate::interpreter::state::ExecutionContext;

    let mut ctx = ExecutionContext::new();

    // Set up x0 (lo=wl0, hi=wh0) with known values.
    // 32-bit elements: 8 lanes per half = 16 total.
    // wl0: [1, 2, 3, 4, 5, 6, 7, 8]
    ctx.vector.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
    // wh0: [9, 10, 11, 12, 13, 14, 15, 16]
    ctx.vector.write(1, [9, 10, 11, 12, 13, 14, 15, 16]);

    // Set up x2 (lo=wl2, hi=wh2) with threshold of 5.
    ctx.vector.write(2, [5, 5, 5, 5, 5, 5, 5, 5]);
    ctx.vector.write(3, [5, 5, 5, 5, 5, 5, 5, 5]);

    let op = SlotOp {
        semantic: Some(SemanticOp::SetGe),
        is_vector: true,
        is_wide_vector: true,
        element_type: Some(ElementType::Int32),
        sources: vec![
            Operand::VectorReg(0),  // x0
            Operand::VectorReg(2),  // x2
        ],
        dest: Some(Operand::ScalarReg(17)),  // r17
        ..SlotOp::default()
    };

    let handled = VectorAlu::execute(&op, &mut ctx);
    assert!(handled);

    // Lanes >= 5: indices 4..15 (0-indexed).
    // lo half: lanes 4,5,6,7 -> bits 4-7 set = 0xF0
    // hi half: lanes 8-15 -> bits 8-15 set = 0xFF00
    // Expected: 0xFFF0
    let result = ctx.scalar.read(17);
    assert_eq!(result, 0xFFF0, "bitmask should have bits 4-15 set, got {:#x}", result);
}
```

- [ ] **Step 7: Run the test**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_wide_compare_ge_scalar_bitmask 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add build_helpers/extract.rs src/interpreter/execute/vector.rs
git commit -m "$(cat <<'EOF'
fix(vector): VGE/VLT are comparisons, not min/max

semantic_from_intrinsic mapped vge->MinGe and vlt->MaxLt, causing
vector min/max to execute instead of comparison. The result was also
written to vector dest (discarded for VGE/VLT's scalar dest).

Fix: map vge->SetGe, vlt->SetLt. The existing vmax_lt and vmin_ge
checks (different intrinsics) still correctly map to MaxLt/MinGe.
Also fix execute_wide SetGe/SetLt to condense the per-lane mask
into a scalar bitmask when dest is ScalarReg.

Generated using Claude Code.
EOF
)"
```

---

### Task 2: Fix UPS/SRS Accumulator Packing (~228 test points)

UPS and SRS both fail at 0% because the accumulator packing convention is
wrong. In Acc32 mode, hardware packs TWO 32-bit values per u64 slot:
`u64[i] = lane[2*i] | (lane[2*i+1] << 32)`. The emulator stores one
value per u64 (zero-extended), so UPS writes half the lanes and SRS reads
wrong values.

**Root cause in UPS:** `ups_vector_to_acc()` in `vector_ups.rs` iterates
8 output lanes (one per u64) regardless of Acc32/Acc64 mode. For Acc32
with 16-bit input, there should be 16 output lanes packed 2 per u64.

**Root cause in SRS:** `vector_srs_from_acc()` in `vector.rs` assumes
"8 accumulator lanes, each u64 holds one value" (line 1349). For Acc32
output-to-16-bit, it reads 8 u64 as 8 lanes. It should read 16 lanes
(2 per u64, extracting lo32 and hi32).

**Files:**
- Modify: `src/interpreter/execute/vector_ups.rs:273-294` (ups_vector_to_acc)
- Modify: `src/interpreter/execute/vector.rs:1318-1410` (vector_srs_from_acc)
- Test: unit tests in both files

- [ ] **Step 1: Write a test for Acc32 UPS packing**

In `src/interpreter/execute/vector_ups.rs` tests section, add:

```rust
#[test]
fn test_ups_acc32_packing_two_per_u64() {
    // VUPS S32 from D16: 16 lanes of 16-bit input -> 16 lanes of 32-bit in acc.
    // Hardware packs 2 x 32-bit values per u64: u64[i] = lane[2i] | (lane[2i+1] << 32).
    let mut src = [0u32; 8];
    // 16-bit lanes packed into u32: lane0=0x0001, lane1=0x0002, ...
    src[0] = 0x0002_0001; // lanes 0, 1
    src[1] = 0x0004_0003; // lanes 2, 3
    src[2] = 0x0006_0005; // lanes 4, 5
    src[3] = 0x0008_0007; // lanes 6, 7
    src[4] = 0x000A_0009; // lanes 8, 9
    src[5] = 0x000C_000B; // lanes 10, 11
    src[6] = 0x000E_000D; // lanes 12, 13
    src[7] = 0x0010_000F; // lanes 14, 15

    let result = ups_vector_to_acc(&src, 0, ElementType::Int16, ElementType::Int32);

    // Each u64 should pack two 32-bit values: lo32 = lane[2i], hi32 = lane[2i+1].
    assert_eq!(result[0], 0x0000_0002_0000_0001, "u64[0] = (lane1 << 32) | lane0");
    assert_eq!(result[1], 0x0000_0004_0000_0003, "u64[1] = (lane3 << 32) | lane2");
    assert_eq!(result[2], 0x0000_0006_0000_0005);
    assert_eq!(result[3], 0x0000_0008_0000_0007);
    assert_eq!(result[4], 0x0000_000A_0000_0009);
    assert_eq!(result[5], 0x0000_000C_0000_000B);
    assert_eq!(result[6], 0x0000_000E_0000_000D);
    assert_eq!(result[7], 0x0000_0010_0000_000F);
}

#[test]
fn test_ups_acc64_one_per_u64() {
    // VUPS S64 from S32: 8 lanes of 32-bit -> 8 lanes of 64-bit.
    // One value per u64 (unchanged from current behavior).
    let mut src = [0u32; 8];
    src[0] = 0x0000_0001;
    src[1] = 0x0000_0002;
    src[2] = 0xFFFF_FFFF; // -1 signed

    let result = ups_vector_to_acc(&src, 0, ElementType::Int32, ElementType::Int64);

    assert_eq!(result[0], 1);
    assert_eq!(result[1], 2);
    // -1 sign-extended to 64-bit
    assert_eq!(result[2], 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_ups_acc32_from_d8() {
    // VUPS S32 from D8: 32 lanes of 8-bit -> 32 lanes of 32-bit in acc.
    // For a 256-bit half (8 u64 words), 16 lanes fit (2 per u64).
    // The other 16 lanes go in the second half (handled by caller).
    let mut src = [0u32; 8];
    // 8-bit lanes: lane0=1, lane1=2, lane2=3, lane3=4 packed in word 0
    src[0] = 0x04_03_02_01;
    src[1] = 0x08_07_06_05;

    let result = ups_vector_to_acc(&src, 0, ElementType::Int8, ElementType::Int32);

    // 2 per u64: u64[0] = (lane1 << 32) | lane0 = (2 << 32) | 1
    assert_eq!(result[0], 0x0000_0002_0000_0001);
    assert_eq!(result[1], 0x0000_0004_0000_0003);
    assert_eq!(result[2], 0x0000_0006_0000_0005);
    assert_eq!(result[3], 0x0000_0008_0000_0007);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_ups_acc32_packing 2>&1 | tail -10`
Expected: FAIL -- currently stores one value per u64

- [ ] **Step 3: Fix ups_vector_to_acc**

Replace the function body in `src/interpreter/execute/vector_ups.rs` at line 273:

```rust
pub fn ups_vector_to_acc(
    src: &[u32; 8],
    shift: u32,
    from_type: ElementType,
    to_type: ElementType,
) -> [u64; 8] {
    let bits_in = from_type.bits() as u32;
    let bits_out = to_type.bits() as u32;
    let signed_input = from_type.is_signed();

    let mut result = [0u64; 8];

    if bits_out <= 32 {
        // Acc32 mode: pack two 32-bit values per u64 word.
        // Total input lanes in a 256-bit half: 256 / bits_in.
        // Output lanes that fit in 8 u64 words: 16 (2 per word).
        let in_lanes = (256 / bits_in).min(16);
        for i in 0..in_lanes {
            let val = extract_lane(src, i, bits_in);
            let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
            let out_u32 = (out as u64) & 0xFFFF_FFFF;
            let word_idx = (i / 2) as usize;
            if i % 2 == 0 {
                result[word_idx] |= out_u32;
            } else {
                result[word_idx] |= out_u32 << 32;
            }
        }
    } else {
        // Acc64 mode: one 64-bit value per u64 word.
        let in_lanes = (256 / bits_in).min(8);
        for i in 0..in_lanes {
            let val = extract_lane(src, i, bits_in);
            let out = ups_lane_signed(val, shift, bits_in, bits_out, false, signed_input);
            result[i as usize] = out as u64;
        }
    }

    result
}
```

- [ ] **Step 4: Run the UPS tests**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib vector_ups::tests 2>&1 | tail -15`
Expected: All UPS tests PASS

- [ ] **Step 5: Write a test for SRS Acc32 unpacking**

In `src/interpreter/execute/vector.rs` tests section, add:

```rust
#[test]
fn test_srs_from_acc32_to_d16() {
    // SRS from Acc32 to D16: 8 u64 words hold 16 acc32 lanes (2 per u64).
    // Each lane is SRS'd to 16-bit and packed 2 per u32 word.
    use crate::interpreter::state::SrsConfig;

    let mut acc = [0u64; 8];
    // Pack: u64[i] = lane[2i] | (lane[2i+1] << 32)
    acc[0] = 0x0000_0002_0000_0001; // lanes 0, 1
    acc[1] = 0x0000_0004_0000_0003; // lanes 2, 3
    acc[2] = 0x0000_0006_0000_0005;
    acc[3] = 0x0000_0008_0000_0007;
    acc[4] = 0x0000_000A_0000_0009;
    acc[5] = 0x0000_000C_0000_000B;
    acc[6] = 0x0000_000E_0000_000D;
    acc[7] = 0x0000_0010_0000_000F;

    let cfg = SrsConfig::default();
    let result = VectorAlu::vector_srs_from_acc(&acc, 0, ElementType::Int32, ElementType::Int16, &cfg);

    // 16 lanes of 16-bit packed 2 per u32: word[i] = (lane[2i+1] << 16) | lane[2i]
    assert_eq!(result[0], 0x0002_0001, "word0 = lanes 0,1");
    assert_eq!(result[1], 0x0004_0003, "word1 = lanes 2,3");
    assert_eq!(result[2], 0x0006_0005);
    assert_eq!(result[3], 0x0008_0007);
    assert_eq!(result[4], 0x000A_0009);
    assert_eq!(result[5], 0x000C_000B);
    assert_eq!(result[6], 0x000E_000D);
    assert_eq!(result[7], 0x0010_000F);
}
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_srs_from_acc32_to_d16 2>&1 | tail -10`
Expected: FAIL -- currently reads 8 lanes from 8 u64, packs into 4 words

- [ ] **Step 7: Fix vector_srs_from_acc for Acc32 modes**

In `src/interpreter/execute/vector.rs`, replace the `vector_srs_from_acc`
Int16/UInt16 branch (around line 1361) to unpack 2 acc32 values per u64:

```rust
            ElementType::Int16 | ElementType::UInt16 => {
                // Acc32 mode: 16 lanes packed 2 per u64.
                // Extract lo32 and hi32 from each u64, SRS each to 16-bit,
                // pack 2 per output u32 word.
                for i in 0..8 {
                    let lo_val = mask_value(acc[i] & 0xFFFF_FFFF);
                    let hi_val = mask_value(acc[i] >> 32);
                    let out_lo = vector_srs::srs_lane(
                        lo_val, shift, signed_output, 16,
                        saturate, sym_sat, mode,
                    );
                    let out_hi = vector_srs::srs_lane(
                        hi_val, shift, signed_output, 16,
                        saturate, sym_sat, mode,
                    );
                    result[i] = (out_lo as u16 as u32) | ((out_hi as u16 as u32) << 16);
                }
            }
```

Similarly fix Int8/UInt8 (around line 1377) to unpack 2 acc32 values per u64:

```rust
            ElementType::Int8 | ElementType::UInt8 => {
                // Acc32 mode: 16 lanes packed 2 per u64 -> 16 x 8-bit output.
                // Pack 4 bytes per output word, consuming 2 u64 words per output word.
                for i in 0..4 {
                    let mut word = 0u32;
                    for j in 0..4 {
                        // Lane index within the 16-lane half:
                        let lane_idx = i * 4 + j;
                        let u64_idx = lane_idx / 2;
                        let is_hi = lane_idx % 2 == 1;
                        let raw = if is_hi {
                            acc[u64_idx] >> 32
                        } else {
                            acc[u64_idx] & 0xFFFF_FFFF
                        };
                        let val = mask_value(raw);
                        let out = vector_srs::srs_lane(
                            val, shift, signed_output, 8,
                            saturate, sym_sat, mode,
                        );
                        word |= (out as u8 as u32) << (j * 8);
                    }
                    result[i] = word;
                }
            }
```

- [ ] **Step 8: Run SRS test**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_srs_from_acc32 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 9: Run the full test suite to check for regressions**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: All tests pass (no regressions in bridge tests or other SRS/UPS consumers)

- [ ] **Step 10: Commit**

```bash
git add src/interpreter/execute/vector_ups.rs src/interpreter/execute/vector.rs
git commit -m "$(cat <<'EOF'
fix(srs-ups): Acc32 packs two values per u64, not one

UPS stored one 32-bit value per u64 slot (zero-extended), but hardware
packs two: u64[i] = lane[2i] | (lane[2i+1] << 32). This caused all
VSRS and VUPS tests to fail because the accumulator layout was wrong.

Fix: ups_vector_to_acc now packs 2 per u64 in Acc32 mode (bits_out<=32),
and vector_srs_from_acc extracts lo32/hi32 per u64 for 16-bit and 8-bit
output types. Acc64 mode (1 per u64) is unchanged.

Generated using Claude Code.
EOF
)"
```

---

### Task 3: Add VPACK/VUNPACK Wide Path (~48 test points)

VPACK and VUNPACK need asymmetric register widths: VPACK reads a 512-bit
x-register and writes a 256-bit w-register; VUNPACK does the reverse.
The current `execute_wide` has no Pack/Unpack handlers, so they fall to
`execute_wide_fallback` which runs `execute_half` twice -- incorrect for
these asymmetric operations.

**Root cause:** No `SemanticOp::Pack` or `SemanticOp::Unpack` cases in
`execute_wide()`. The fallback splits into two halves but can't handle
the width mismatch.

**Files:**
- Modify: `src/interpreter/execute/vector_pack.rs` (add helpers)
- Modify: `src/interpreter/execute/vector.rs` (add execute_wide Pack/Unpack cases)

- [ ] **Step 1: Write tests for pack_half and unpack_half helpers**

In `src/interpreter/execute/vector_pack.rs` test section, add:

```rust
#[test]
fn test_pack_half_d16_to_d8() {
    // 16 lanes of 16-bit in 256 bits, pack to 8-bit.
    // Only the first 128 bits of output are meaningful.
    let mut src = [0u32; 8];
    src[0] = 0x0102_0001; // lane0=0x01, lane1=0x02 (as 16-bit: 0x0001, 0x0102)
    // Actually: 16-bit lanes. lane0=0x0001, lane1=0x0102... let me use simpler values.
    // lane0=1, lane1=2 packed as 16-bit: word0 = (2 << 16) | 1 = 0x0002_0001
    src[0] = 0x0002_0001;
    src[1] = 0x0004_0003;
    src[2] = 0x0006_0005;
    src[3] = 0x0008_0007;
    src[4] = 0x000A_0009;
    src[5] = 0x000C_000B;
    src[6] = 0x000E_000D;
    src[7] = 0x0010_000F;

    let result = pack_half(&src, 16, 8, false, PackMode::Truncate);

    // 16 lanes of 8-bit = 128 bits = 4 words.
    // lane0=1, lane1=2, lane2=3, lane3=4 -> word0 = 0x04030201
    assert_eq!(result[0], 0x04_03_02_01);
    assert_eq!(result[1], 0x08_07_06_05);
    assert_eq!(result[2], 0x0C_0B_0A_09);
    assert_eq!(result[3], 0x10_0F_0E_0D);
}

#[test]
fn test_unpack_half_d8_to_d16() {
    // 16 lanes of 8-bit starting at lane 0, widen to 16-bit.
    let mut src = [0u32; 8];
    src[0] = 0x04_03_02_01;
    src[1] = 0x08_07_06_05;
    src[2] = 0x0C_0B_0A_09;
    src[3] = 0x10_0F_0E_0D;
    // lanes 16-31 in src[4..7] (for the high half call)

    let result = unpack_half(&src, 0, 8, 16, false);

    // 16 lanes of 16-bit = 256 bits = 8 words.
    assert_eq!(result[0], 0x0002_0001);
    assert_eq!(result[1], 0x0004_0003);
    assert_eq!(result[2], 0x0006_0005);
    assert_eq!(result[3], 0x0008_0007);
    assert_eq!(result[4], 0x000A_0009);
    assert_eq!(result[5], 0x000C_000B);
    assert_eq!(result[6], 0x000E_000D);
    assert_eq!(result[7], 0x0010_000F);
}
```

- [ ] **Step 2: Implement pack_half and unpack_half**

Add to `src/interpreter/execute/vector_pack.rs` (before the `#[cfg(test)]` section):

```rust
/// Parse VPACK bit widths from encoding name.
/// VPACK_{D|S}{out}_{D|S}{in}: e.g. VPACK_D4_D8 -> (8, 4, false)
pub fn pack_widths_from_name(name: &str) -> (u32, u32, bool) {
    let upper = name.to_uppercase();
    let parts: Vec<&str> = upper.split('_').collect();
    if parts.len() >= 3 {
        let signed = parts[1].starts_with('S') || parts[2].starts_with('S');
        let out_bits: u32 = parts[1][1..].parse().unwrap_or(8);
        let in_bits: u32 = parts[2][1..].parse().unwrap_or(16);
        (in_bits, out_bits, signed)
    } else {
        (16, 8, false)
    }
}

/// Parse VUNPACK bit widths from encoding name.
/// VUNPACK_{D|S}{out}_{D|S}{in}: e.g. VUNPACK_D16_D8 -> (8, 16, false)
pub fn unpack_widths_from_name(name: &str) -> (u32, u32, bool) {
    let upper = name.to_uppercase();
    let parts: Vec<&str> = upper.split('_').collect();
    if parts.len() >= 3 {
        let signed = parts[1].starts_with('S') || parts[2].starts_with('S');
        let out_bits: u32 = parts[1][1..].parse().unwrap_or(16);
        let in_bits: u32 = parts[2][1..].parse().unwrap_or(8);
        (in_bits, out_bits, signed)
    } else {
        (8, 16, false)
    }
}

/// Pack a 256-bit half: narrow each of 256/bits_i lanes from bits_i to bits_o.
/// Result fits in (256/bits_i * bits_o / 32) words (typically 4).
pub fn pack_half(src: &[u32; 8], bits_i: u32, bits_o: u32, signed: bool, mode: PackMode) -> [u32; 8] {
    let lanes = (256 / bits_i) as usize;
    let mut narrowed = vec![0i64; lanes];
    for i in 0..lanes {
        let val = extract_lane(src, i, bits_i, signed);
        narrowed[i] = pack_lane(val, bits_i, bits_o, signed, mode);
    }
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &narrowed, bits_o);
    result
}

/// Unpack lanes from a 256-bit source starting at lane_start,
/// widening from bits_i to bits_o. Produces 256/bits_o output lanes.
pub fn unpack_half(src: &[u32; 8], lane_start: usize, bits_i: u32, bits_o: u32, signed: bool) -> [u32; 8] {
    let out_lanes = (256 / bits_o) as usize;
    let mut widened = vec![0i64; out_lanes];
    for i in 0..out_lanes {
        let val = extract_lane(src, lane_start + i, bits_i, signed);
        widened[i] = unpack_lane(val, bits_i, bits_o, signed);
    }
    let mut result = [0u32; 8];
    insert_lanes(&mut result, &widened, bits_o);
    result
}
```

NOTE: Verify the existing `extract_lane`, `pack_lane`, `unpack_lane`, and
`insert_lanes` function signatures in `vector_pack.rs` match these call
sites. The helper names and parameter counts may need adjustment. Read the
actual file to confirm before implementing.

- [ ] **Step 3: Run pack/unpack helper tests**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib vector_pack::tests 2>&1 | tail -15`
Expected: PASS

- [ ] **Step 4: Add Pack/Unpack to execute_wide**

In `src/interpreter/execute/vector.rs`, add before the `_ => Self::execute_wide_fallback(...)` arm:

```rust
            SemanticOp::Pack => {
                // VPACK: 512-bit x-reg source -> 256-bit w-reg dest.
                let name = op.encoding_name.as_deref().unwrap_or("");
                let (bits_i, bits_o, signed) = vector_pack::pack_widths_from_name(name);
                let src = Self::get_wide_vec_source(op, ctx, 0);
                let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                let src_hi: [u32; 8] = src[8..].try_into().unwrap();

                let packed_lo = vector_pack::pack_half(
                    &src_lo, bits_i, bits_o, signed, vector_pack::PackMode::Truncate,
                );
                let packed_hi = vector_pack::pack_half(
                    &src_hi, bits_i, bits_o, signed, vector_pack::PackMode::Truncate,
                );

                // Each half produces 128 bits (4 words). Concatenate into 256-bit dest.
                let words_per_half = ((256 / bits_i) * bits_o / 32) as usize;
                let mut result = [0u32; 8];
                result[..words_per_half].copy_from_slice(&packed_lo[..words_per_half]);
                result[words_per_half..words_per_half * 2]
                    .copy_from_slice(&packed_hi[..words_per_half]);

                Self::write_vector_dest(op, ctx, result);
                true
            }

            SemanticOp::Unpack => {
                // VUNPACK: 256-bit w-reg source -> 512-bit x-reg dest.
                let name = op.encoding_name.as_deref().unwrap_or("");
                let (bits_i, bits_o, signed) = vector_pack::unpack_widths_from_name(name);

                // Read the 256-bit source (NOT wide -- it's a w-register).
                let src = Self::get_vector_source(op, ctx, 0);

                let lane_start_hi = (256 / bits_o) as usize;
                let result_lo = vector_pack::unpack_half(&src, 0, bits_i, bits_o, signed);
                let result_hi = vector_pack::unpack_half(&src, lane_start_hi, bits_i, bits_o, signed);

                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&result_lo);
                result[8..].copy_from_slice(&result_hi);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
```

- [ ] **Step 5: Run full test suite**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/vector_pack.rs src/interpreter/execute/vector.rs
git commit -m "$(cat <<'EOF'
feat(vector): add VPACK/VUNPACK wide-path handlers

VPACK (512->256 bit) and VUNPACK (256->512 bit) need asymmetric register
widths that the execute_wide_fallback cannot handle. Add dedicated handlers
in execute_wide that split into halves, pack/unpack each half, and
concatenate the results.

Generated using Claude Code.
EOF
)"
```

---

### Task 4: Add Missing Itinerary Semantics for VBCST/VBAND/VBOR (~85 test points)

Several instructions have no semantic because their sched_class is not in
the `ITINERARY_SEMANTICS` table and their intrinsic patterns use compound
Pat<> records that `load_pattern_records()` skips.

| Instruction | sched_class | Needed semantic |
|-------------|-------------|-----------------|
| VBAND | II_VBLOG | And |
| VBOR | II_VBLOG | Or |
| VBCST_8/16/32/64 | II_VBCST | VectorBroadcast |

Note: `II_VEXTBCST` (extract-broadcast) already maps to `VectorBroadcast`.
The plain `II_VBCST` (scalar broadcast) needs its own entry.

**Files:**
- Modify: `build_helpers/extract.rs:1108-1134` (ITINERARY_SEMANTICS table)

- [ ] **Step 1: Add itinerary entries**

In `build_helpers/extract.rs`, add to the `ITINERARY_SEMANTICS` array:

```rust
    ("II_VBCST", "SemanticOp::VectorBroadcast"),
    ("II_VBLOG", "SemanticOp::And"),  // VBAND/VBOR share this class
```

Wait -- VBAND and VBOR share `II_VBLOG` but need different semantics (And
vs Or). A single itinerary entry can only map to one semantic. This means
the itinerary approach cannot distinguish VBAND from VBOR.

**Alternative:** Add a name-based override in the `semantic_override` table
in `decoder.rs`, or extend `semantic_from_intrinsic` to handle the bitwise
instructions.

Actually, the better fix: since VBAND and VBOR are distinguished by name
(not by sched_class), add them to the decoder's `semantic_override` table
in `src/interpreter/decode/decoder.rs`.

- [ ] **Step 1 (revised): Add VBCST to ITINERARY_SEMANTICS**

In `build_helpers/extract.rs`, add to `ITINERARY_SEMANTICS`:

```rust
    ("II_VBCST", "SemanticOp::VectorBroadcast"),
```

- [ ] **Step 2: Add VBAND/VBOR to the decoder's semantic_override**

In `src/interpreter/decode/decoder.rs`, find the `semantic_override` table
(around line 1127). Add entries:

```rust
    ("VBAND", Some(SemanticOp::And)),
    ("VBOR", Some(SemanticOp::Or)),
    ("VBNEG", Some(SemanticOp::Bneg)),  // if Bneg semantic exists; otherwise use a custom handler
```

Check what SemanticOp variants exist for bitwise negation. If `Bneg` doesn't
exist, check if `Not` or `Xor` covers it, or add a new variant.

- [ ] **Step 3: Rebuild and run tests**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: All tests pass (no regressions)

- [ ] **Step 4: Commit**

```bash
git add build_helpers/extract.rs src/interpreter/decode/decoder.rs
git commit -m "$(cat <<'EOF'
fix(decode): add missing semantics for VBCST, VBAND, VBOR

VBCST (scalar broadcast) had no itinerary entry -- only VEXTBCST
(extract-broadcast) was mapped. VBAND/VBOR shared sched_class II_VBLOG
so they need name-based overrides in the decoder.

Generated using Claude Code.
EOF
)"
```

---

### Task 5: Integration Test (ISA harness run)

After all fixes are committed, rebuild and run the ISA test suite to measure
the improvement.

- [ ] **Step 1: Full rebuild**

Run: `cd /home/triple/npu-work/xdna-emu && nice -n 19 cargo build --release 2>&1 | tail -5`

- [ ] **Step 2: Run ISA tests**

Run: `cd /home/triple/npu-work/xdna-emu && env -u XDNA_EMU ./scripts/isa-test.sh`
NOTE: This takes 5-10 minutes. Run directly (not backgrounded) so the user
can watch progress. Do NOT pipe through tail.

- [ ] **Step 3: Examine results**

Read the analysis output at `build/isa-test-results/YYYYMMDD/analysis.log`
and `analysis-failing.log`. Compare against baseline:

| Category | Baseline | Target |
|----------|----------|--------|
| vector-srs | 0/132 | ~120/132 |
| vector-ups | 0/96 | ~90/96 |
| vector-arith (VGE/VLT) | ~30% | ~80%+ |
| vector-pack | 0/24 | ~20/24 |
| vector-unpack | 0/24 | ~20/24 |
| vector-broadcast | 39% | ~70%+ |
| vector-bitwise | 30% | ~60%+ |
| **Overall** | **52.2%** | **65%+** |

---

## Deferred Items (not in this plan)

These require separate plans due to complexity or external dependencies:

| Category | Test Points | Reason for Deferral |
|----------|-------------|---------------------|
| VMAC config register | 226 | Requires understanding aietools `mulmac.py` config word semantics |
| PADDA 2D/3D | 69 | Requires decode changes (count register output) + new execution logic |
| DIVS iterative step | 25 | Requires multi-step emulation model (partial quotient) |
| Batch 121-122 (branches + fused ops) | 30 | Requires investigation of nested call LR handling |
| VINSERT_16/32/64 | 37 | Needs further debugging (8-bit works, wider fails) |
| VPUSH_64 | 22 | Needs further debugging (element type fix applied, still fails) |
