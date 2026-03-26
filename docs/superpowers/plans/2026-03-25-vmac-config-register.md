# VMAC Config Register Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the config register into all MAC instruction handlers so the emulator performs geometry-correct matrix multiplies instead of element-wise operations, targeting ~274 dense test points.

**Architecture:** The config-driven matmul engine (`matmul_config_driven`) and accumulator helpers already exist in `vector_matmul.rs` but are never called. The main work is: (1) route MAC semantics through config parsing to the existing engine, (2) generate permute tables from constants.py to replace naive row-major indexing, and (3) handle the double-accumulator (AddMac/SubMac) variants.

**Tech Stack:** Rust, Python (table generator), aietools constants.py (read-only reference)

---

## Existing Infrastructure (already built, not called)

These exist and are correct but unused:

| File | What exists |
|------|-------------|
| `vector_config.rs` | `MatMulConfig::from_config_word()`, geometry tables, lookup functions |
| `vector_matmul.rs` | `matmul_config_driven()`, `extract_element_512()`, `read_acc_wide()`, `write_acc_wide()`, all accumulator packing helpers |
| `vector_matmul_sparse.rs` | Sparse matmul engine (1330 lines, deferred to phase 2) |
| `vector.rs` | `get_config_register()` reads scalar config operand |

## File Map

| File | Change | Task |
|------|--------|------|
| `src/interpreter/execute/vector.rs` | Route MAC semantics to new `execute_matmul`, delete old element-wise handlers | 2 |
| `src/interpreter/execute/vector_matmul.rs` | Add `execute_matmul()` entry point, add permute-table path | 1, 3 |
| `src/interpreter/execute/vector_matmul_tables.rs` | NEW: generated permute tables | 3 |
| `src/interpreter/execute/mod.rs` | Add `pub mod vector_matmul_tables;` | 3 |
| `tools/gen-matmul-tables.py` | NEW: table generator script | 3 |

---

### Task 1: Add execute_matmul Entry Point

Wire the existing `matmul_config_driven` to MAC instructions via a new
`execute_matmul` function that reads the config register, parses it, reads
512-bit inputs and 1024-bit accumulators, and delegates to the engine.

**Files:**
- Modify: `src/interpreter/execute/vector_matmul.rs`

- [ ] **Step 1: Write a test for config-driven MAC dispatch**

Add to the `#[cfg(test)]` section of `vector_matmul.rs`:

```rust
#[test]
fn test_execute_matmul_i8xi8_identity() {
    // A 4x8 identity-like pattern * B 8x8 pattern should produce
    // predictable accumulator output when config selects i8xi8 dense.
    use crate::interpreter::bundle::*;
    use crate::interpreter::state::ExecutionContext;

    let mut ctx = ExecutionContext::new();

    // Config word: amode=0 (acc32), bmode=1 (8x8), variant=0 (dense 4x8x8)
    // sgn_x=1, sgn_y=1, zero_acc=1 (clear acc before)
    let conf: u32 = (1 << 0)   // zero_acc
                  | (0 << 1)   // amode=0
                  | (1 << 3)   // bmode=1 (8x8)
                  | (0 << 5)   // variant=0
                  | (1 << 8)   // sgn_y=1
                  | (1 << 9);  // sgn_x=1
    ctx.scalar.write(0, conf);

    // Set up A (x0): 64 bytes of i8, all 1s
    for i in 0..16 {
        ctx.vector.write_wide_word(0, i, 0x01010101);
    }
    // Set up B (x2): 64 bytes of i8, all 1s
    for i in 0..16 {
        ctx.vector.write_wide_word(2, i, 0x01010101);
    }

    // Build SlotOp for vmac
    let op = SlotOp {
        semantic: Some(SemanticOp::MatMul),
        is_vector: true,
        is_wide_vector: true,
        element_type: Some(ElementType::Int8),
        sources: vec![
            Operand::VectorReg(0),   // x0 (A)
            Operand::VectorReg(2),   // x2 (B)
            Operand::AccumReg(0),    // acc source
            Operand::ScalarReg(0),   // config register
        ],
        dest: Some(Operand::AccumReg(0)),
        ..SlotOp::default()
    };

    let handled = execute_matmul(&op, &mut ctx);
    assert!(handled, "execute_matmul should handle MatMul");

    // For a 4x8x8 matmul where all A=1 and all B=1:
    // Each output[m][n] = sum(k=0..7) { 1 * 1 } = 8
    // 32 output lanes, each should be 8.
    let acc = ctx.accumulator.read_wide(0);
    for i in 0..16 {
        let lo = (acc[i] & 0xFFFF_FFFF) as i32;
        let hi = (acc[i] >> 32) as i32;
        assert_eq!(lo, 8, "acc32 lane {} (lo) should be 8, got {}", i * 2, lo);
        assert_eq!(hi, 8, "acc32 lane {} (hi) should be 8, got {}", i * 2 + 1, hi);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_execute_matmul_i8xi8_identity 2>&1 | tail -10`
Expected: FAIL -- `execute_matmul` does not exist yet

- [ ] **Step 3: Implement execute_matmul**

Add this public function to `vector_matmul.rs`:

```rust
use crate::interpreter::bundle::{Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::execute::vector_config::MatMulConfig;
use crate::tablegen::SemanticOp;

/// Entry point for all MAC-family instructions.
///
/// Reads the config register, parses the config word, reads 512-bit inputs
/// and 1024-bit accumulator, performs the matrix multiply via
/// `matmul_config_driven`, and writes the result back.
///
/// Returns true if handled, false if the instruction was not a MAC variant.
pub fn execute_matmul(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let semantic = match op.semantic {
        Some(s) => s,
        None => return false,
    };

    // Only handle MAC-family semantics.
    let (negate_product, is_double_acc) = match semantic {
        SemanticOp::Mac | SemanticOp::MatMul => (false, false),
        SemanticOp::NegMul | SemanticOp::NegMatMul | SemanticOp::MatMulSub => (true, false),
        SemanticOp::AddMac => (false, true),
        SemanticOp::SubMac => (false, true),
        _ => return false,
    };

    // Read config register (last scalar source).
    let conf_val = get_config_register(op, ctx).unwrap_or(0);

    // Detect bf16 from encoding name.
    let is_bf16 = op.encoding_name.as_ref()
        .map_or(false, |n| n.contains("_F_") || n.ends_with("_F"));

    // Parse config word.
    let config = match MatMulConfig::from_config_word(conf_val, is_bf16) {
        Some(c) => c,
        None => {
            log::warn!("[MATMUL] Unknown config word {:#x} (bf16={})", conf_val, is_bf16);
            return false;
        }
    };

    // Read 512-bit input vectors.
    let (a, b) = get_two_vec512_sources(op, ctx);

    // Read destination accumulator (1024-bit).
    let dst_reg = get_acc_dest_reg(op);
    let mut acc = ctx.accumulator.read_wide(dst_reg);

    // For double-accumulator modes (AddMac/SubMac), read second accumulator
    // and merge before the multiply.
    if is_double_acc {
        let acc2_reg = get_acc_source_reg(op);
        let acc2 = ctx.accumulator.read_wide(acc2_reg);
        let sub_acc2 = matches!(semantic, SemanticOp::SubMac)
            || ((conf_val >> 12) & 1) != 0;  // sub1 bit
        let output_count = (config.rows * config.cols) as usize;
        for i in 0..output_count {
            let v1 = read_acc_wide(&acc, i, config.acc_width);
            let v2 = read_acc_wide(&acc2, i, config.acc_width);
            let combined = if sub_acc2 { v1 - v2 } else { v1 + v2 };
            write_acc_wide(&mut acc, i, combined, config.acc_width);
        }
    }

    // Apply negate_product by flipping subtract in a local config copy.
    let mut config = config;
    if negate_product {
        config.subtract = !config.subtract;
    }

    // Perform the matrix multiply.
    matmul_config_driven(&mut acc, &a, &b, &config);

    // Write result back.
    ctx.accumulator.write_wide(dst_reg, acc);
    true
}

/// Read the scalar config register from operand sources.
fn get_config_register(op: &SlotOp, ctx: &ExecutionContext) -> Option<u32> {
    for src in op.sources.iter().rev() {
        if let Operand::ScalarReg(r) = src {
            return Some(ctx.scalar.read(*r));
        }
    }
    None
}

/// Read two 512-bit vector sources as Vec512 ([u32; 16]).
fn get_two_vec512_sources(op: &SlotOp, ctx: &ExecutionContext) -> (Vec512, Vec512) {
    let mut vecs = Vec::new();
    for src in &op.sources {
        if let Operand::VectorReg(r) = src {
            let lo = ctx.vector.read(*r);
            let hi = ctx.vector.read(*r + 1);
            let mut v = [0u32; 16];
            v[..8].copy_from_slice(&lo);
            v[8..].copy_from_slice(&hi);
            vecs.push(v);
            if vecs.len() == 2 { break; }
        }
    }
    let a = vecs.get(0).copied().unwrap_or([0u32; 16]);
    let b = vecs.get(1).copied().unwrap_or([0u32; 16]);
    (a, b)
}

/// Get the accumulator destination register index.
fn get_acc_dest_reg(op: &SlotOp) -> u8 {
    match &op.dest {
        Some(Operand::AccumReg(r)) => *r,
        _ => 0,
    }
}

/// Get the first accumulator source register index.
fn get_acc_source_reg(op: &SlotOp) -> u8 {
    for src in &op.sources {
        if let Operand::AccumReg(r) = src {
            return *r;
        }
    }
    0
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_execute_matmul_i8xi8_identity 2>&1 | tail -10`
Expected: PASS

NOTE: This test uses the naive row-major byte offsets in `matmul_config_driven`.
For all-1s inputs, the row-major layout gives the same result as the hardware
permute layout (every A*B product is 1*1=1, so permutation doesn't matter).
Permute-table correctness will be tested in Task 3.

- [ ] **Step 5: Add test for bf16 path**

```rust
#[test]
fn test_execute_matmul_bf16_ones() {
    use crate::interpreter::bundle::*;
    use crate::interpreter::state::ExecutionContext;

    let mut ctx = ExecutionContext::new();

    // Config: amode=2 (FP32), bmode=3 (16x16), variant=0, zero_acc=1, sgn_x=1, sgn_y=1
    let conf: u32 = 1 | (2 << 1) | (3 << 3) | (1 << 8) | (1 << 9);
    ctx.scalar.write(0, conf);

    // bf16 1.0 = 0x3F80
    let bf16_ones = 0x3F80_3F80u32;  // two bf16 1.0 values per u32
    for i in 0..16 {
        ctx.vector.write_wide_word(0, i, bf16_ones);
        ctx.vector.write_wide_word(2, i, bf16_ones);
    }

    let op = SlotOp {
        semantic: Some(SemanticOp::MatMul),
        is_vector: true,
        is_wide_vector: true,
        element_type: Some(ElementType::BFloat16),
        encoding_name: Some("VMAC_F_vmac_bm_core_dense".to_string()),
        sources: vec![
            Operand::VectorReg(0),
            Operand::VectorReg(2),
            Operand::AccumReg(0),
            Operand::ScalarReg(0),
        ],
        dest: Some(Operand::AccumReg(0)),
        ..SlotOp::default()
    };

    let handled = execute_matmul(&op, &mut ctx);
    assert!(handled);

    // bf16 4x8x4: each output = sum(k=0..7) { 1.0 * 1.0 } = 8.0
    let acc = ctx.accumulator.read_wide(0);
    for i in 0..8 {
        let lo_bits = (acc[i] & 0xFFFF_FFFF) as u32;
        let hi_bits = (acc[i] >> 32) as u32;
        let lo = f32::from_bits(lo_bits);
        let hi = f32::from_bits(hi_bits);
        assert!((lo - 8.0).abs() < 0.01, "bf16 acc lane {} lo: expected 8.0, got {}", i*2, lo);
        assert!((hi - 8.0).abs() < 0.01, "bf16 acc lane {} hi: expected 8.0, got {}", i*2+1, hi);
    }
}
```

- [ ] **Step 6: Run tests**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib test_execute_matmul 2>&1 | tail -10`
Expected: Both tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/execute/vector_matmul.rs
git commit -m "feat(matmul): add execute_matmul entry point with config parsing"
```

---

### Task 2: Route MAC Dispatch Through execute_matmul

Replace all element-wise MAC handlers in `vector.rs` with calls to the
new `execute_matmul` from `vector_matmul.rs`.

**Files:**
- Modify: `src/interpreter/execute/vector.rs`

- [ ] **Step 1: Add execute_matmul call to execute_wide**

In `execute_wide()`, BEFORE the existing `SemanticOp::Accumulate` handler
(around line 3900), add a block that intercepts all MAC semantics:

```rust
            // ========== Matrix Multiply (config-driven) ==========
            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                return super::vector_matmul::execute_matmul(op, ctx);
            }
```

This intercepts MAC instructions in the wide path before they can fall
through to the fallback. The `return` means if `execute_matmul` returns
false (unknown config), the instruction is treated as unhandled.

- [ ] **Step 2: Replace execute_half MAC handlers**

In `execute_half()`, replace the existing `SemanticOp::Mac` handler
(around line 147) and all related handlers (MatMul, NegMatMul,
MatMulSub, AddMac, SubMac, NegMul) with the same delegation:

```rust
            SemanticOp::Mac | SemanticOp::MatMul | SemanticOp::MatMulSub
            | SemanticOp::NegMul | SemanticOp::NegMatMul
            | SemanticOp::AddMac | SemanticOp::SubMac => {
                // MAC instructions always operate on 512-bit inputs and
                // 1024-bit accumulators. Route to the config-driven engine
                // regardless of is_wide_vector flag.
                return super::vector_matmul::execute_matmul(op, ctx);
            }
```

This ensures MAC instructions are handled correctly even when `is_wide_vector`
is false (some MAC instruction encodings may not have Vector512 operand
fields, causing the decoder to set `is_wide_vector = false`).

- [ ] **Step 3: Run full test suite**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: All tests pass (existing MAC tests may now produce different output
since the config-driven path reads a config register that may be 0 in old
tests -- verify no regressions)

- [ ] **Step 4: Delete dead code**

Remove the old element-wise MAC functions from `vector.rs` that are no
longer called:
- `vector_mac()` (around line 1722)
- `vector_matmul_dense()` (around line 1003)
- `vector_matmul_bf16()` (search for `fn vector_matmul_bf16`)
- `vector_neg_matmul()` (search for `fn vector_neg_matmul`)
- `vector_neg_mul()` (search for `fn vector_neg_mul`)
- `vector_matmul_sub()` (search for `fn vector_matmul_sub`)
- `vector_double_acc_mac()` (search for `fn vector_double_acc_mac`)

Search for each function name to confirm it's no longer referenced before
deleting. If any are still referenced from tests, keep them.

- [ ] **Step 5: Run tests again after cleanup**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: All tests pass, no dead code warnings for deleted functions

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "fix(matmul): route all MAC semantics through config-driven engine"
```

---

### Task 3: Generate Permute Tables from constants.py

The current `matmul_config_driven` uses naive row-major byte offsets:
`A[r][k]` at `(r * inner + k) * bytes_x`. The real hardware permutes input
bytes differently. Generate the correct mapping from `constants.py`.

**Files:**
- Create: `tools/gen-matmul-tables.py`
- Create: `src/interpreter/execute/vector_matmul_tables.rs`
- Modify: `src/interpreter/execute/mod.rs` (add module declaration)
- Modify: `src/interpreter/execute/vector_matmul.rs` (use tables)

- [ ] **Step 1: Create the table generator script**

Create `tools/gen-matmul-tables.py`:

```python
#!/usr/bin/env python3
"""Generate matmul permute tables from aietools constants.py.

Reads the permute mode tables from the aietools Python model and emits
Rust const arrays mapping (a_byte_offset, b_byte_offset, output_index)
triples for each dense geometry mode.

Usage:
    python3 tools/gen-matmul-tables.py > src/interpreter/execute/vector_matmul_tables.rs
"""

import sys
import os

# Add aietools model to path
AIETOOLS_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "aietools", "data", "aie_ml", "lib", "python_model", "model"
)
sys.path.insert(0, AIETOOLS_MODEL)

import constants as C


def emit_header():
    print("//! Generated matmul permute tables for AIE2.")
    print("//!")
    print("//! DO NOT EDIT -- regenerate with: python3 tools/gen-matmul-tables.py")
    print("//!")
    print("//! Each geometry entry maps (amode, bmode, variant) to a flat array of")
    print("//! (a_byte_offset, b_byte_offset, output_index) triples describing which")
    print("//! input bytes multiply and accumulate into which output lane.")
    print()


def emit_geometry(pmode_idx, pm, mm):
    """Emit a single geometry's lane table."""
    rows = pm.rows
    cols = pm.cols
    inner = pm.inner
    channels = pm.channels
    bits_x = mm.bits_x
    bits_y = mm.bits_y
    acc_cmb = mm.acc_cmb
    bfloat = mm.bfloat

    # Compute amode/bmode/variant from the config word encoding.
    # We derive these from the mult mode and perm mode properties.
    if bfloat:
        amode = 2
        bmode = 3
    elif acc_cmb == 2:
        amode = 1
        if bits_y == 8 and bits_x == 16:
            bmode = 2
        elif bits_y == 16 and bits_x == 16:
            bmode = 3
        elif bits_y == 16 and bits_x == 32:
            bmode = 0  # 32x16 reuses bmode=0 in amode=1
        else:
            return None  # Unknown
    else:
        amode = 0
        if bits_y == 4:
            bmode = 0
        elif bits_y == 8 and bits_x == 8:
            bmode = 1
        elif bits_y == 8 and bits_x == 16:
            bmode = 2
        elif bits_y == 16:
            bmode = 3
        else:
            return None

    # Build the lane table from the permute arrays.
    lanes = []
    for r in range(rows):
        for c in range(cols):
            out_idx = pm.idx_o(r, c, 0)  # channel=0 for dense
            for k in range(inner):
                a_perm_idx = pm.idx_x(r, k, 0)  # byte position in permuted X
                b_perm_idx = pm.idx_y(k, c, 0)  # byte position in permuted Y

                # The permute tables map from logical position to physical byte.
                a_byte = C.permute_x[pmode_idx][a_perm_idx]
                b_byte = C.permute_y[pmode_idx][b_perm_idx]
                lanes.append((a_byte, b_byte, out_idx))

    return {
        "amode": amode,
        "bmode": bmode,
        "variant": pm.variant if hasattr(pm, 'variant') else 0,
        "rows": rows,
        "inner": inner,
        "cols": cols,
        "bits_x": bits_x,
        "bits_y": bits_y,
        "acc_cmb": acc_cmb,
        "bfloat": bfloat,
        "sparse": pm.sparse if hasattr(pm, 'sparse') else False,
        "lanes": lanes,
    }


def main():
    emit_header()

    print("#[allow(clippy::all)]")
    print()

    geometries = []

    # Iterate permute modes, filter to dense (non-sparse, non-convolution).
    for idx, pm in enumerate(C.perm_modes):
        mm = C.mult_modes[pm.mmode]

        # Skip sparse and convolution modes for phase 1.
        if hasattr(pm, 'sparse') and pm.sparse:
            continue
        if hasattr(pm, 'conv') and pm.conv:
            continue
        if pm.channels > 1:
            continue

        geo = emit_geometry(idx, pm, mm)
        if geo and not geo["sparse"]:
            geometries.append(geo)

    # Emit lane arrays as const.
    for i, geo in enumerate(geometries):
        name = f"LANES_{i}"
        print(f"const {name}: &[(u16, u16, u16)] = &[")
        for (a, b, o) in geo["lanes"]:
            print(f"    ({a}, {b}, {o}),")
        print("];")
        print()

    # Emit the geometry table.
    print("/// Dense matmul geometry entries with permute lane tables.")
    print("pub struct MatMulGeometry {")
    print("    pub amode: u32,")
    print("    pub bmode: u32,")
    print("    pub variant: u32,")
    print("    pub rows: u32,")
    print("    pub inner: u32,")
    print("    pub cols: u32,")
    print("    pub bits_x: u32,")
    print("    pub bits_y: u32,")
    print("    pub acc_cmb: u32,")
    print("    pub bfloat: bool,")
    print("    /// (a_byte_offset, b_byte_offset, output_index) triples.")
    print("    pub lanes: &'static [(u16, u16, u16)],")
    print("}")
    print()

    print(f"pub const DENSE_GEOMETRIES: &[MatMulGeometry] = &[")
    for i, geo in enumerate(geometries):
        bf = "true" if geo["bfloat"] else "false"
        print(f"    MatMulGeometry {{")
        print(f"        amode: {geo['amode']}, bmode: {geo['bmode']}, variant: {geo['variant']},")
        print(f"        rows: {geo['rows']}, inner: {geo['inner']}, cols: {geo['cols']},")
        print(f"        bits_x: {geo['bits_x']}, bits_y: {geo['bits_y']}, acc_cmb: {geo['acc_cmb']},")
        print(f"        bfloat: {bf}, lanes: LANES_{i},")
        print(f"    }},")
    print("];")
    print()

    print("/// Look up a dense geometry by (amode, bmode, variant).")
    print("pub fn lookup_geometry(amode: u32, bmode: u32, variant: u32) -> Option<&'static MatMulGeometry> {")
    print("    DENSE_GEOMETRIES.iter().find(|g| g.amode == amode && g.bmode == bmode && g.variant == variant)")
    print("}")


if __name__ == "__main__":
    main()
```

NOTE: The `constants.py` API (attribute names like `idx_x`, `idx_y`,
`idx_o`, `permute_x`, `permute_y`, `perm_modes`, `mult_modes`) must be
verified against the actual file. Read `constants.py` to confirm the
attribute names before running. The script above shows the intended
logic; adapt to the actual API.

- [ ] **Step 2: Run the generator**

```bash
cd /home/triple/npu-work/xdna-emu
python3 tools/gen-matmul-tables.py > src/interpreter/execute/vector_matmul_tables.rs
```

If the script fails due to API mismatches with constants.py, fix the
attribute access patterns. The key data needed:
- Permute tables: `permute_x[pmode_idx]` and `permute_y[pmode_idx]` (512-entry byte arrays)
- Per perm_mode: `rows`, `cols`, `inner`, `channels`, `mmode`, `variant`
- Per mult_mode: `bits_x`, `bits_y`, `acc_cmb`, `bfloat`
- Element index functions: `idx_x(row, k, channel)`, `idx_y(k, col, channel)`, `idx_o(row, col, channel)`

- [ ] **Step 3: Add module declaration**

In `src/interpreter/execute/mod.rs`, add:

```rust
pub mod vector_matmul_tables;
```

- [ ] **Step 4: Verify the generated file compiles**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo build --lib 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 5: Wire permute tables into matmul_config_driven**

In `vector_matmul.rs`, modify `matmul_config_driven` to use the permute
tables when available. Add at the top of the function, before the
existing row-major loop:

```rust
pub fn matmul_config_driven(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    config: &MatMulConfig,
) {
    // Zero accumulator if requested.
    if !config.accumulate {
        *acc = [0u64; 16];
    }

    // Try permute-table path first (hardware-accurate byte offsets).
    let amode = match config.acc_width {
        AccWidth::Acc32 => if config.bfloat { 2 } else { 0 },
        AccWidth::Acc64 => 1,
    };
    let bmode = match (config.a_type.bits(), config.b_type.bits()) {
        (8, 4) | (32, 16) => 0,
        (8, 8) => 1,
        (16, 8) => 2,
        (16, 16) => 3,
        _ => 0,
    };

    if let Some(geo) = super::vector_matmul_tables::lookup_geometry(amode, bmode, 0) {
        // Permute-table-driven path.
        if config.bfloat {
            matmul_permute_bf16(acc, a, b, geo, config);
        } else {
            matmul_permute_int(acc, a, b, geo, config);
        }
        return;
    }

    // Fallback: naive row-major path (existing code below).
    // ... existing row-major implementation unchanged ...
}

/// Permute-table-driven integer matmul.
fn matmul_permute_int(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    geo: &super::vector_matmul_tables::MatMulGeometry,
    config: &MatMulConfig,
) {
    // Each lane triple: (a_byte, b_byte, output_idx)
    // Multiple lanes may share the same output_idx (they sum into it).
    let bits_x = geo.bits_x;
    let bits_y = geo.bits_y;
    let acc_width = config.acc_width;

    // Accumulate products per output lane.
    let output_count = (geo.rows * geo.cols) as usize;
    let mut sums = vec![0i64; output_count];

    for &(a_byte, b_byte, out_idx) in geo.lanes {
        let a_val = extract_element_512(a, a_byte as usize, bits_x, config.x_signed);
        let b_val = extract_element_512(b, b_byte as usize, bits_y, config.y_signed);
        sums[out_idx as usize] += a_val * b_val;
    }

    // Apply to accumulator.
    for i in 0..output_count {
        let prev = read_acc_wide(acc, i, acc_width);
        let result = if config.subtract { prev - sums[i] } else { prev + sums[i] };
        write_acc_wide(acc, i, result, acc_width);
    }
}

/// Permute-table-driven bf16 matmul.
fn matmul_permute_bf16(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    geo: &super::vector_matmul_tables::MatMulGeometry,
    config: &MatMulConfig,
) {
    let output_count = (geo.rows * geo.cols) as usize;
    let mut sums = vec![0.0f32; output_count];

    for &(a_byte, b_byte, out_idx) in geo.lanes {
        // bf16: 2 bytes per element, byte offset / 2 gives element index
        let a_elem = a_byte as usize / 2;
        let b_elem = b_byte as usize / 2;
        let a_word = a_elem / 2;
        let a_half = a_elem % 2;
        let b_word = b_elem / 2;
        let b_half = b_elem % 2;
        let a_bits = ((a[a_word] >> (a_half * 16)) & 0xFFFF) as u16;
        let b_bits = ((b[b_word] >> (b_half * 16)) & 0xFFFF) as u16;
        let a_val = f32::from_bits((a_bits as u32) << 16);
        let b_val = f32::from_bits((b_bits as u32) << 16);
        sums[out_idx as usize] += a_val * b_val;
    }

    for i in 0..output_count {
        let prev = read_acc_wide_f32(acc, i);
        let result = if config.subtract { prev - sums[i] } else { prev + sums[i] };
        write_acc_wide_f32(acc, i, result);
    }
}
```

- [ ] **Step 6: Run full test suite**

Run: `cd /home/triple/npu-work/xdna-emu && TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add tools/gen-matmul-tables.py src/interpreter/execute/vector_matmul_tables.rs \
        src/interpreter/execute/vector_matmul.rs src/interpreter/execute/mod.rs
git commit -m "feat(matmul): add permute-table-driven matmul from constants.py"
```

---

### Task 4: Integration Test

Rebuild the release binary and run the ISA test suite to measure improvement.

**Files:** None (test-only)

- [ ] **Step 1: Build release**

Run: `cd /home/triple/npu-work/xdna-emu && nice -n 19 cargo build --release 2>&1 | tail -5`

- [ ] **Step 2: Rebuild and install XRT plugin**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/rebuild-plugin.sh --release 2>&1 | tail -5`
NOTE: May need `dangerouslyDisableSandbox` for the install step.

- [ ] **Step 3: Run ISA tests**

Run: `cd /home/triple/npu-work/xdna-emu && env -u XDNA_EMU ./scripts/isa-test.sh`
NOTE: Takes 5-10 minutes. Run directly for live output.

- [ ] **Step 4: Compare results**

Read `build/isa-test-results/YYYYMMDD/analysis.log`. Compare:

| Category | Before | Target |
|----------|--------|--------|
| vector-mac | 0/228 | ~170/228 (75%+) |
| Overall | 55.8% | 60%+ |

The dense integer and bf16 modes should now work. Sparse modes will still
fail (phase 2). Some integer tests may fail if the permute tables have
edge cases not covered by the row-major fallback.

---

## Deferred to Phase 2

- Sparse matmul (596 test points) -- `vector_matmul_sparse.rs` exists but needs config-driven wiring
- Element-wise variants (variant != 0) -- not tested by ISA harness
- Convolution / complex modes
- Pure-Rust permute table derivation (replace gen script)
- `HardwareMatMul` trait implementation (sub-cycle accuracy)
