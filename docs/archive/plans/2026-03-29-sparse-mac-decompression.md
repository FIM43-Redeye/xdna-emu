# Sparse MAC Pair-Routing Decompression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 0/452 sparse MAC ISA test failures by implementing pair-routing decompression for compressed B data.

**Architecture:** The hardware's vmac instruction treats the 512-bit B register as 32 byte-pairs, using the 128-bit mask to route each pair to its correct position in a 128-byte dense B matrix. We add a `sparse_pair_route` decompression function, widen the sparse multiply to operate on 128-byte buffers, and read the full 128-bit mask.

**Tech Stack:** Rust, no new dependencies. Changes confined to two files: `registers.rs` (1 method) and `vector_matmul.rs` (rewrite sparse path).

**Spec:** `docs/superpowers/specs/2026-03-29-sparse-mac-decompression-design.md`

---

### Task 1: Add `read_u128` to MaskRegisterFile

**Files:**
- Modify: `src/interpreter/state/registers.rs:666-670`
- Test: inline in same file or `cargo test --lib`

- [ ] **Step 1: Write the test**

Add to the existing test module for registers (or create one at the bottom of `registers.rs`):

```rust
#[cfg(test)]
mod mask_tests {
    use super::*;

    #[test]
    fn test_mask_read_u128() {
        let mut mrf = MaskRegisterFile::new();
        // Write a known 128-bit pattern: 0xDDDDDDDD_CCCCCCCC_BBBBBBBB_AAAAAAAA
        mrf.write(0, [0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD]);
        let val = mrf.read_u128(0);
        assert_eq!(val, 0xDDDDDDDD_CCCCCCCC_BBBBBBBB_AAAAAAAA_u128);
    }

    #[test]
    fn test_mask_read_u128_zero() {
        let mrf = MaskRegisterFile::new();
        assert_eq!(mrf.read_u128(0), 0u128);
        assert_eq!(mrf.read_u128(3), 0u128);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib mask_tests -- --nocapture`

Expected: Compile error -- `read_u128` method does not exist.

- [ ] **Step 3: Implement `read_u128`**

Add this method to the `impl MaskRegisterFile` block in `src/interpreter/state/registers.rs`, right after the existing `read_u64_low` method (after line 670):

```rust
    /// Read the full 128-bit mask register as u128 (used by sparse pair-routing).
    pub fn read_u128(&self, reg: u8) -> u128 {
        let idx = (reg as usize) % NUM_MASK_REGS;
        (self.regs[idx][0] as u128)
            | ((self.regs[idx][1] as u128) << 32)
            | ((self.regs[idx][2] as u128) << 64)
            | ((self.regs[idx][3] as u128) << 96)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib mask_tests -- --nocapture`

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/state/registers.rs
git commit -m "feat(registers): add read_u128 to MaskRegisterFile for sparse MAC"
```

---

### Task 2: Implement `sparse_pair_route` with unit tests

**Files:**
- Modify: `src/interpreter/execute/vector_matmul.rs`

- [ ] **Step 1: Write the tests**

Add a test module at the bottom of `vector_matmul.rs` (or extend the existing one):

```rust
#[cfg(test)]
mod sparse_decompress_tests {
    use super::*;

    /// Helper: build a compressed array with sequential byte values.
    fn seq_compressed() -> [u8; 64] {
        let mut c = [0u8; 64];
        for i in 0..64 {
            c[i] = (i + 1) as u8; // 1, 2, 3, ..., 64
        }
        c
    }

    #[test]
    fn test_pair_route_mask_0000() {
        // All zeros: no data routed, output is all zeros.
        let compressed = seq_compressed();
        // Set group 0 to 0000, all other groups also 0000.
        let mask: u128 = 0;
        let out = sparse_pair_route(&compressed, mask);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pair_route_mask_0011() {
        // Group 0: mask=0011 (bits 0,1 set).
        // Pair: hi=compressed[0]=1, lo=compressed[1]=2.
        // lo -> lowest set bit (pos 0), hi -> highest set bit (pos 1).
        // Output: [2, 1, 0, 0, ...]
        let compressed = seq_compressed();
        let mask: u128 = 0b0011;
        let out = sparse_pair_route(&compressed, mask);
        assert_eq!(out[0], 2, "lo -> pos 0");
        assert_eq!(out[1], 1, "hi -> pos 1");
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_pair_route_mask_0101() {
        // Group 0: mask=0101 (bits 0,2 set).
        // Pair: hi=compressed[0]=1, lo=compressed[1]=2.
        // lo -> pos 0, hi -> pos 2.
        let compressed = seq_compressed();
        let mask: u128 = 0b0101;
        let out = sparse_pair_route(&compressed, mask);
        assert_eq!(out[0], 2, "lo -> pos 0");
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 1, "hi -> pos 2");
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_pair_route_mask_1001() {
        // Group 0: mask=1001 (bits 0,3 set).
        // lo -> pos 0, hi -> pos 3.
        let compressed = seq_compressed();
        let mask: u128 = 0b1001;
        let out = sparse_pair_route(&compressed, mask);
        assert_eq!(out[0], 2, "lo -> pos 0");
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 1, "hi -> pos 3");
    }

    #[test]
    fn test_pair_route_single_bit_0001() {
        // Group 0: mask=0001 (bit 0 only).
        // Only lo is used, hi is ignored. lo -> pos 0.
        let compressed = seq_compressed();
        let mask: u128 = 0b0001;
        let out = sparse_pair_route(&compressed, mask);
        assert_eq!(out[0], 2, "lo -> pos 0");
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_pair_route_single_bit_1000() {
        // Group 0: mask=1000 (bit 3 only).
        // Only one set bit: hi=compressed[0]=1 -> pos 3.
        // (Single bit: use hi slot, lo is 0.)
        let compressed = seq_compressed();
        let mask: u128 = 0b1000;
        let out = sparse_pair_route(&compressed, mask);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 1, "hi -> pos 3");
    }

    #[test]
    fn test_pair_route_two_groups() {
        // Group 0: mask=0011 -> pair (1, 2), lo=2 -> pos 0, hi=1 -> pos 1
        // Group 1: mask=1010 -> pair (3, 4), lo=4 -> pos 5, hi=3 -> pos 7
        // (group 1 positions are 4+{1,3})
        let compressed = seq_compressed();
        let mask: u128 = 0b1010_0011;
        let out = sparse_pair_route(&compressed, mask);
        // Group 0
        assert_eq!(out[0], 2);
        assert_eq!(out[1], 1);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 0);
        // Group 1: mask=1010, bits 1 and 3 set
        assert_eq!(out[4], 0);
        assert_eq!(out[5], 4, "lo -> pos 4+1=5");
        assert_eq!(out[6], 0);
        assert_eq!(out[7], 3, "hi -> pos 4+3=7");
    }

    #[test]
    fn test_pair_route_invalid_3bits() {
        // Group 0: mask=0111 (3 set bits -- invalid 2:4).
        // Our model: lo -> lowest (pos 0), hi -> highest (pos 2).
        // Middle bit (pos 1) gets 0.
        let compressed = seq_compressed();
        let mask: u128 = 0b0111;
        let out = sparse_pair_route(&compressed, mask);
        assert_eq!(out[0], 2, "lo -> lowest set bit (pos 0)");
        assert_eq!(out[1], 0, "middle set bit -> 0 (only 2 data paths)");
        assert_eq!(out[2], 1, "hi -> highest set bit (pos 2)");
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_pair_route_all_groups_1100() {
        // Every group has mask=1100 (bits 2,3 set).
        // For each group g: lo -> pos 4g+2, hi -> pos 4g+3.
        let compressed = seq_compressed();
        let mask: u128 = {
            let mut m: u128 = 0;
            for g in 0..32u128 {
                m |= 0b1100 << (4 * g);
            }
            m
        };
        let out = sparse_pair_route(&compressed, mask);
        for g in 0..32 {
            let hi = compressed[2 * g];     // value for higher pos
            let lo = compressed[2 * g + 1]; // value for lower pos
            assert_eq!(out[4 * g + 0], 0, "group {g} pos 0");
            assert_eq!(out[4 * g + 1], 0, "group {g} pos 1");
            assert_eq!(out[4 * g + 2], lo, "group {g} lo -> pos 2");
            assert_eq!(out[4 * g + 3], hi, "group {g} hi -> pos 3");
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib sparse_decompress_tests -- --nocapture`

Expected: Compile error -- `sparse_pair_route` does not exist.

- [ ] **Step 3: Implement `sparse_pair_route`**

Add this function in `vector_matmul.rs`, above `matmul_sparse_config_driven` (around line 436):

```rust
/// Decompress sparse B data using pair-routing.
///
/// The hardware treats the 512-bit x register as 32 byte-pairs. For each
/// group of 4 mask bits, the pair `(compressed[2*g], compressed[2*g+1])` is
/// routed to the set-bit positions within the group's 4-byte output span:
///
///   - `compressed[2*g]`   (hi) -> highest set-bit position in the group
///   - `compressed[2*g+1]` (lo) -> lowest set-bit position in the group
///   - Clear-bit positions -> 0
///
/// For groups with >2 set bits (violates 2:4 structured sparsity), only the
/// lowest and highest set-bit positions receive data. Middle positions get 0.
/// This matches the hardware's 2-data-path-per-group physical constraint.
///
/// The mask operates at byte granularity (1 bit per byte). For 16-bit
/// elements, each element spans 2 consecutive mask bits. For 4-bit elements,
/// each mask bit controls a byte containing 2 nibbles.
///
/// Reference: aie_doc.hpp partial decompression table (vldb.sparse.fill
/// stage 1), vmac pair-routing (stage 2).
fn sparse_pair_route(compressed: &[u8; 64], mask: u128) -> [u8; 128] {
    let mut out = [0u8; 128];

    for g in 0..32 {
        let mask4 = ((mask >> (4 * g)) & 0xF) as u8;
        if mask4 == 0 {
            continue; // All positions zero, skip.
        }

        let hi = compressed[2 * g];
        let lo = compressed[2 * g + 1];

        // Find lowest and highest set-bit positions within the 4-bit group.
        let lowest = mask4.trailing_zeros() as usize;   // 0..3
        let highest = 3 - mask4.leading_zeros() as usize + 4; // need (7 - leading_zeros)
        // Simpler: highest set bit position.
        let highest = 7 - (mask4 as u8).leading_zeros() as usize;
        // But mask4 is only 4 bits wide stored in u8, so leading_zeros counts
        // from bit 7 down. For a 4-bit value in u8:
        //   highest_bit = 7 - mask4.leading_zeros()
        // But that gives bit position in the u8, not in the 4-bit group.
        // Use: highest = (8 - 1) - mask4.leading_zeros() = 7 - leading_zeros
        // For mask4=0b1001 (in u8: 0b00001001): leading_zeros=4, highest=7-4=3 ✓
        // For mask4=0b0011 (in u8: 0b00000011): leading_zeros=6, highest=7-6=1 ✓

        let popcount = mask4.count_ones();
        let base = 4 * g;

        if popcount == 1 {
            // Single set bit: route hi to that position (lo unused).
            out[base + lowest] = hi;
        } else {
            // 2+ set bits: lo -> lowest, hi -> highest.
            out[base + lowest] = lo;
            out[base + highest] = hi;
        }
    }

    out
}
```

Wait, I made this confusing with the leading_zeros arithmetic. Let me simplify and write it cleanly:

```rust
fn sparse_pair_route(compressed: &[u8; 64], mask: u128) -> [u8; 128] {
    let mut out = [0u8; 128];

    for g in 0..32 {
        let mask4 = ((mask >> (4 * g)) & 0xF) as u8;
        if mask4 == 0 {
            continue;
        }

        let hi = compressed[2 * g];
        let lo = compressed[2 * g + 1];
        let base = 4 * g;

        // Find lowest and highest set-bit positions (0..3) within the group.
        let lowest = mask4.trailing_zeros() as usize;
        // For highest: mask4 is 4 bits in a u8. Bit 3 is the MSB of interest.
        // 31 - (mask4 as u32).leading_zeros() gives the highest set bit in u32.
        // But for u8: 7 - mask4.leading_zeros() works.
        let highest = 7 - mask4.leading_zeros() as usize;

        if lowest == highest {
            // Single set bit: only hi is used.
            out[base + lowest] = hi;
        } else {
            // 2+ set bits: lo -> lowest, hi -> highest.
            out[base + lowest] = lo;
            out[base + highest] = hi;
        }
    }

    out
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib sparse_decompress_tests -- --nocapture`

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/execute/vector_matmul.rs
git commit -m "feat(sparse): add sparse_pair_route decompression function with tests"
```

---

### Task 3: Add `extract_element_bytes` and helper to convert Vec512 to bytes

**Files:**
- Modify: `src/interpreter/execute/vector_matmul.rs`

- [ ] **Step 1: Write the tests**

Add to the test module in `vector_matmul.rs`:

```rust
#[cfg(test)]
mod extract_bytes_tests {
    use super::*;

    #[test]
    fn test_extract_i8_signed() {
        let mut buf = [0u8; 128];
        buf[0] = 0xFF; // -1 as i8
        buf[1] = 0x7F; // 127
        buf[5] = 0x80; // -128
        assert_eq!(extract_element_bytes(&buf, 0, 8, true), -1);
        assert_eq!(extract_element_bytes(&buf, 1, 8, true), 127);
        assert_eq!(extract_element_bytes(&buf, 5, 8, true), -128);
    }

    #[test]
    fn test_extract_u8() {
        let mut buf = [0u8; 128];
        buf[0] = 0xFF;
        assert_eq!(extract_element_bytes(&buf, 0, 8, false), 255);
    }

    #[test]
    fn test_extract_i16_le() {
        let mut buf = [0u8; 128];
        // 0x0102 at byte 0 (LE: buf[0]=0x02, buf[1]=0x01)
        buf[0] = 0x02;
        buf[1] = 0x01;
        assert_eq!(extract_element_bytes(&buf, 0, 16, false), 0x0102);
        assert_eq!(extract_element_bytes(&buf, 0, 16, true), 0x0102);
    }

    #[test]
    fn test_extract_i16_signed_negative() {
        let mut buf = [0u8; 128];
        buf[4] = 0x00;
        buf[5] = 0x80; // 0x8000 = -32768
        assert_eq!(extract_element_bytes(&buf, 4, 16, true), -32768);
        assert_eq!(extract_element_bytes(&buf, 4, 16, false), 0x8000);
    }

    #[test]
    fn test_extract_4bit() {
        let mut buf = [0u8; 128];
        buf[0] = 0xBA; // low nibble = 0xA, high nibble = 0xB
        // 4-bit: byte_idx is element index. Element 0 = low nibble, element 1 = high nibble.
        assert_eq!(extract_element_bytes(&buf, 0, 4, false), 0xA);
        assert_eq!(extract_element_bytes(&buf, 1, 4, false), 0xB);
        // Signed: 0xA = -6 (4-bit signed), 0xB = -5
        assert_eq!(extract_element_bytes(&buf, 0, 4, true), -6);
        assert_eq!(extract_element_bytes(&buf, 1, 4, true), -5);
    }

    #[test]
    fn test_extract_oob_returns_zero() {
        let buf = [0u8; 128];
        assert_eq!(extract_element_bytes(&buf, 200, 8, false), 0);
    }

    #[test]
    fn test_vec512_to_bytes() {
        let mut v: Vec512 = [0u32; 16];
        v[0] = 0x04030201;
        v[1] = 0x08070605;
        let bytes = vec512_to_bytes(&v);
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[1], 0x02);
        assert_eq!(bytes[2], 0x03);
        assert_eq!(bytes[3], 0x04);
        assert_eq!(bytes[4], 0x05);
        assert_eq!(bytes[7], 0x08);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib extract_bytes_tests -- --nocapture`

Expected: Compile error -- functions do not exist.

- [ ] **Step 3: Implement `vec512_to_bytes` and `extract_element_bytes`**

Add these functions in `vector_matmul.rs`:

```rust
/// Convert a Vec512 ([u32; 16], 64 bytes) to a byte array in little-endian order.
fn vec512_to_bytes(v: &Vec512) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    for (i, word) in v.iter().enumerate() {
        let base = i * 4;
        bytes[base] = *word as u8;
        bytes[base + 1] = (*word >> 8) as u8;
        bytes[base + 2] = (*word >> 16) as u8;
        bytes[base + 3] = (*word >> 24) as u8;
    }
    bytes
}

/// Convert a quad vector ([u32; 32], 128 bytes) to a byte array.
fn vec1024_to_bytes(v: &[u32; 32]) -> [u8; 128] {
    let mut bytes = [0u8; 128];
    for (i, word) in v.iter().enumerate() {
        let base = i * 4;
        bytes[base] = *word as u8;
        bytes[base + 1] = (*word >> 8) as u8;
        bytes[base + 2] = (*word >> 16) as u8;
        bytes[base + 3] = (*word >> 24) as u8;
    }
    bytes
}

/// Extract an element from a 128-byte buffer.
///
/// For 4-bit elements, `byte_idx` is the element index (two elements per
/// byte: element 0 = low nibble, element 1 = high nibble). For 8/16/32-bit
/// elements, `byte_idx` is the byte offset.
///
/// Returns 0 for out-of-bounds accesses.
fn extract_element_bytes(src: &[u8; 128], byte_idx: usize, bits: u32, signed: bool) -> i64 {
    match bits {
        4 => {
            let byte_pos = byte_idx / 2;
            let nibble = byte_idx % 2;
            if byte_pos >= 128 { return 0; }
            let raw = src[byte_pos];
            let val = if nibble == 0 { raw & 0xF } else { (raw >> 4) & 0xF };
            if signed && (val & 0x8) != 0 {
                (val as i8 | !0x0Fi8) as i64
            } else {
                val as i64
            }
        }
        8 => {
            if byte_idx >= 128 { return 0; }
            let val = src[byte_idx];
            if signed { val as i8 as i64 } else { val as i64 }
        }
        16 => {
            if byte_idx + 1 >= 128 { return 0; }
            let val = u16::from_le_bytes([src[byte_idx], src[byte_idx + 1]]);
            if signed { val as i16 as i64 } else { val as i64 }
        }
        32 => {
            if byte_idx + 3 >= 128 { return 0; }
            let val = u32::from_le_bytes([
                src[byte_idx], src[byte_idx + 1],
                src[byte_idx + 2], src[byte_idx + 3],
            ]);
            if signed { val as i32 as i64 } else { val as i64 }
        }
        _ => 0,
    }
}
```

- [ ] **Step 4: Run tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib extract_bytes_tests -- --nocapture`

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/execute/vector_matmul.rs
git commit -m "feat(sparse): add vec512_to_bytes, vec1024_to_bytes, extract_element_bytes"
```

---

### Task 4: Rewrite `get_sparse_operands` to return byte arrays + u128 mask

**Files:**
- Modify: `src/interpreter/execute/vector_matmul.rs:366-434`

- [ ] **Step 1: Rewrite `get_sparse_operands`**

Replace the existing function (lines 366-434) with:

```rust
/// Read operands for a sparse MAC instruction.
///
/// Sparse MAC instructions have different operand types from dense:
/// - A (dense input): VectorReg -- 512-bit (xs1) or 1024-bit (ys1)
/// - B (sparse input): ControlReg(28+n) -- composite qx_n = {x_n, q_n}
///
/// Returns:
/// - `a_bytes`: A operand as 128-byte buffer (zero-padded if A < 128 bytes)
/// - `b_compressed`: Raw 64 bytes from the B x-register (before decompression)
/// - `mask`: Full 128-bit mask from the q register
fn get_sparse_operands(op: &SlotOp, ctx: &ExecutionContext) -> ([u8; 128], [u8; 64], u128) {
    let mut a_bytes = [0u8; 128];
    let mut b_compressed = [0u8; 64];
    let mut mask: u128 = 0;
    let mut found_a = false;

    for src in &op.sources {
        match src {
            Operand::VectorReg(r) => {
                if !found_a {
                    found_a = true;
                    if op.is_quad_vector {
                        // y-register (1024-bit): full 128 bytes for sparse wide.
                        let base = *r & !3;
                        let quad = ctx.vector.read_quad(base);
                        a_bytes = vec1024_to_bytes(&quad);
                    } else {
                        // x-register (512-bit): 64 bytes, zero-padded to 128.
                        let base = *r & !1;
                        let wide = ctx.vector.read_wide(base);
                        let narrow = vec512_to_bytes(&wide);
                        a_bytes[..64].copy_from_slice(&narrow);
                        // a_bytes[64..128] remains zero.
                    }
                }
            }
            Operand::ControlReg(id) => {
                if *id >= 28 && *id <= 31 {
                    let qx_idx = *id - 28; // 0..3
                    // B data: raw bytes from x_n register.
                    let x_base = (qx_idx as u8) * 2;
                    let wide = ctx.vector.read_wide(x_base);
                    b_compressed = vec512_to_bytes(&wide);
                    // Mask: full 128 bits from q_n register.
                    mask = ctx.mask.read_u128(qx_idx as u8);
                }
            }
            _ => {}
        }
    }

    if !found_a {
        log::error!("[MATMUL-SPARSE] missing VectorReg source for A operand");
    }

    (a_bytes, b_compressed, mask)
}
```

- [ ] **Step 2: Update call site in `execute_matmul`**

Replace the sparse branch in `execute_matmul` (lines 134-175). Find:

```rust
    // Read input vectors. Sparse instructions have different operand types.
    let (a, b, sparse_mask) = if is_sparse {
        get_sparse_operands(op, ctx)
    } else {
        let (a, b) = get_two_vec512(op, ctx);
        (a, b, 0u64) // mask unused for dense
    };
```

And the sparse call:

```rust
    if is_sparse {
        matmul_sparse_config_driven(&mut acc, &a, &b, sparse_mask, &config);
    } else {
        matmul_config_driven(&mut acc, &a, &b, &config);
    }
```

Replace both with:

```rust
    // Read input vectors and perform multiply.
    if is_sparse {
        let (a_bytes, b_compressed, mask) = get_sparse_operands(op, ctx);
        let b_decompressed = sparse_pair_route(&b_compressed, mask);
        matmul_sparse_config_driven(&mut acc, &a_bytes, &b_decompressed, &config);
    } else {
        let (a, b) = get_two_vec512(op, ctx);
        matmul_config_driven(&mut acc, &a, &b, &config);
    };
```

- [ ] **Step 3: Verify it compiles**

Run: `TMPDIR=/tmp/claude-1000 cargo check`

Expected: Compile error on `matmul_sparse_config_driven` signature mismatch (still takes old types). This is expected -- we fix it in Task 5.

- [ ] **Step 4: Commit (WIP -- will compile after Task 5)**

No commit yet. Proceed to Task 5.

---

### Task 5: Rewrite `matmul_sparse_config_driven` to use byte arrays

**Files:**
- Modify: `src/interpreter/execute/vector_matmul.rs:436-578`

- [ ] **Step 1: Write an integrated test**

Add to the test module:

```rust
#[cfg(test)]
mod sparse_matmul_tests {
    use super::*;
    use crate::interpreter::execute::vector_config::MatMulConfig;

    #[test]
    fn test_sparse_i8xi8_identity() {
        // i8xi8 sparse: 4x16x8, acc32.
        // Set up a simple multiply: A has 1s in one row, B decompressed has
        // a single non-zero column. Result should be the sum of the row-column
        // dot product.
        let mut a = [0u8; 128];
        // Row 0, inner positions 0..15: all set to 1.
        for k in 0..16 {
            a[0 * 16 + k] = 1;
        }

        let mut b = [0u8; 128];
        // Column 0, inner positions 0..15: values 1..16.
        for k in 0..16 {
            b[k * 8 + 0] = (k + 1) as u8;
        }

        let config = MatMulConfig::from_config_word(
            // amode=0, bmode=1 (i8xi8), variant=5 (sparse), zero_acc=1,
            // sgn_x=1, sgn_y=1
            (1 << 0) | (0 << 1) | (1 << 3) | (5 << 5) | (1 << 8) | (1 << 9),
            false,
        ).expect("valid sparse i8xi8 config");

        let mut acc = [0u64; 16];
        matmul_sparse_config_driven(&mut acc, &a, &b, &config);

        // C[0][0] = sum(k=0..15) 1 * (k+1) = 136
        let result = (acc[0] & 0xFFFF_FFFF) as i32;
        assert_eq!(result, 136, "dot product of [1]*16 with [1..16]");
    }

    #[test]
    fn test_sparse_bf16_basic() {
        // bf16 sparse: 4x16x4, acc32(fp32).
        let mut a = [0u8; 128];
        // Row 0, element 0: bf16 1.0 = 0x3F80
        a[0] = 0x80;
        a[1] = 0x3F;

        let mut b = [0u8; 128];
        // Column 0, element 0: bf16 2.0 = 0x4000
        b[0] = 0x00;
        b[1] = 0x40;

        let config = MatMulConfig::from_config_word(
            // bf16 sparse: variant=2, zero_acc=1
            (1 << 0) | (2 << 5),
            true, // is_bf16
        ).expect("valid sparse bf16 config");

        let mut acc = [0u64; 16];
        matmul_sparse_config_driven(&mut acc, &a, &b, &config);

        // C[0][0] = 1.0 * 2.0 = 2.0
        let result_bits = (acc[0] & 0xFFFF_FFFF) as u32;
        let result = f32::from_bits(result_bits);
        assert_eq!(result, 2.0);
    }
}
```

- [ ] **Step 2: Rewrite `matmul_sparse_config_driven`**

Replace the entire function (lines 436-578) with:

```rust
/// Sparse config-driven matrix multiply using pair-routed (decompressed) B.
///
/// Both `a` and `b` are 128-byte buffers. `b` has already been decompressed
/// by `sparse_pair_route` -- the mask was consumed during decompression.
/// The multiply loop shape is identical to the dense version but indexes
/// into the wider byte buffers.
pub fn matmul_sparse_config_driven(
    acc: &mut Acc1024,
    a: &[u8; 128],
    b: &[u8; 128],
    config: &MatMulConfig,
) {
    let rows = config.rows as usize;
    let inner = config.inner as usize;
    let cols = config.cols as usize;
    let bits_x = config.a_type.bits() as u32;
    let bits_y = config.b_type.bits() as u32;
    let bytes_x = if bits_x == 4 { 1 } else { (bits_x / 8) as usize };
    let bytes_y = if bits_y == 4 { 1 } else { (bits_y / 8) as usize };

    if !config.accumulate {
        *acc = [0u64; 16];
    }

    if config.bfloat {
        for r in 0..rows {
            for c in 0..cols {
                let out_idx = r * cols + c;
                let mut sum: f32 = 0.0;

                for k in 0..inner {
                    let a_byte = (r * inner + k) * bytes_x;
                    let b_byte = (k * cols + c) * bytes_y;

                    if a_byte + 1 >= 128 || b_byte + 1 >= 128 {
                        continue;
                    }
                    let a_bits = u16::from_le_bytes([a[a_byte], a[a_byte + 1]]);
                    let b_bits = u16::from_le_bytes([b[b_byte], b[b_byte + 1]]);
                    let a_val = f32::from_bits((a_bits as u32) << 16);
                    let b_val = f32::from_bits((b_bits as u32) << 16);

                    sum += a_val * b_val;
                }

                let prev = read_acc_wide_f32(acc, out_idx);
                if config.subtract {
                    write_acc_wide_f32(acc, out_idx, prev - sum);
                } else {
                    write_acc_wide_f32(acc, out_idx, prev + sum);
                }
            }
        }
        return;
    }

    // Integer sparse path.
    for r in 0..rows {
        for c in 0..cols {
            let out_idx = r * cols + c;
            let mut sum: i64 = 0;

            for k in 0..inner {
                let a_byte = if bits_x == 4 {
                    r * inner + k
                } else {
                    (r * inner + k) * bytes_x
                };
                let b_byte = if bits_y == 4 {
                    k * cols + c
                } else {
                    (k * cols + c) * bytes_y
                };

                let a_val = extract_element_bytes(a, a_byte, bits_x, config.x_signed);
                let b_val = extract_element_bytes(b, b_byte, bits_y, config.y_signed);

                sum += a_val * b_val;
            }

            let prev = read_acc_wide(acc, out_idx, config.acc_width);
            if config.subtract {
                write_acc_wide(acc, out_idx, prev - sum, config.acc_width);
            } else {
                write_acc_wide(acc, out_idx, prev + sum, config.acc_width);
            }
        }
    }
}
```

- [ ] **Step 3: Build and run all tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`

Expected: All tests pass, including the new sparse tests and all existing tests (no regressions).

- [ ] **Step 4: Commit**

```bash
git add src/interpreter/execute/vector_matmul.rs
git commit -m "feat(sparse): rewrite sparse matmul with pair-routing decompression

Replace the broken sparse multiply path (0/452 ISA tests passing) with
proper pair-routing decompression. The hardware treats the B register as
32 byte-pairs, using the 128-bit mask to route each pair to its correct
position in a 128-byte dense matrix. Previous code treated B as dense
data with a mask filter, and only read 64 of 128 mask bits.

Three bugs fixed:
- B indexing OOB (loop indexed up to 127 bytes in a 64-byte buffer)
- Only reading 64 of 128 mask bits (read_u64_low -> read_u128)
- No decompression (dense interpretation -> pair-routing expansion)"
```

---

### Task 6: Build release and run ISA tests

**Files:** None (verification only)

- [ ] **Step 1: Build release**

Run: `TMPDIR=/tmp/claude-1000 nice -n 19 cargo build --release`

Expected: Clean build, no warnings from the changed files.

- [ ] **Step 2: Run ISA tests (EMU-only against existing HW baselines)**

Run: `XDNA_EMU=release ./scripts/isa-test.sh --no-hw`

Expected: Significant improvement in sparse MAC categories. Previous: 0/452. Target: meaningful non-zero pass rate. Any remaining failures will need diagnosis (possibly from the invalid-mask edge cases or bf16 float semantics).

- [ ] **Step 3: Check sparse results specifically**

Run: `grep -i sparse build/isa-test-results/latest/analysis.log`

Record the pass rates. Compare with previous 0/452 baseline.

- [ ] **Step 4: Check for regressions**

Compare overall pass rate with previous 79.5% (3879/4878). The dense MAC and all non-MAC categories should be unchanged.

- [ ] **Step 5: Commit test results (if applicable)**

If the ISA test script produces updated analysis, commit any relevant notes.

```bash
git add -A
git commit -m "test(sparse): ISA test results after pair-routing decompression"
```
