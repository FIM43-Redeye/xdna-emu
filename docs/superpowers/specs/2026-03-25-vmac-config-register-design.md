# VMAC Config Register Implementation Design

## Goal

Implement config-register-driven matrix multiply for all dense VMAC/VMUL
instruction variants, replacing the current element-wise placeholder handlers.
This is the single largest accuracy gap (228+ test points at 0% in the
vector-mac category, with additional MAC-related failures in vector-arith).

## Background

The AIE2 vector unit's multiply instructions (VMAC, VMUL, VADDMAC, VSUBMAC,
VNEGMUL, VNEGMAC, VMSC, VADDMSC, etc.) all share a single hardware multiply
array: 512 physical 8-bit x 4-bit multiplier cells fed through a permutation
network, with a 4-level post-add tree (PSA) reducing products into
accumulator lanes.

The behavior of this array is entirely controlled by a **config word** passed
in a scalar register. The config word encodes:

| Bits | Field | Purpose |
|------|-------|---------|
| 0 | `zero_acc` | Clear accumulator before operation |
| 1-2 | `amode` | Accumulator width (0=I32, 1=I64, 2=FP32) |
| 3-4 | `bmode` | Element widths (0=8x4, 1=8x8, 2=16x8, 3=16x16) |
| 5-7 | `variant` | Geometry variant (dense, element-wise, convolution) |
| 8 | `sgn_y` | Y/B signedness |
| 9 | `sgn_x` | X/A signedness |
| 10 | `shift16` | Left-shift accumulator by 16 before add |
| 11 | `sub0` | Negate product |
| 12 | `sub1` | Negate acc1 (double-accumulator modes) |
| 13 | `sub2` | Negate acc2 (double-accumulator modes) |
| 16+ | `sub_mask` | Per-element subtract mask |

The current emulator ignores this config word entirely. All MAC handlers
perform element-wise `acc[i] += a[i] * b[i]` with no permutation, no
geometry, and no sign extension control. This produces wrong output for
every single MAC instruction.

## What Already Exists

`vector_config.rs` has a complete, correct implementation of:
- `MatMulConfig::from_config_word()` -- parses all config fields
- `CONFIG_GEOMETRY_TABLE` -- maps (amode, bmode, variant) to tile dimensions
- `DENSE_GEOMETRY_TABLE` / `SPARSE_GEOMETRY_TABLE` -- geometry entries
- `lookup_integer_geometry()` / `lookup_bf16_geometry()` -- table lookups

This code is fully tested but **never called from any MAC execution path**.

## Architecture

### Three-Layer Design

```
MatMulEngine (trait)
    |
    +-- FastMatMul (impl)       -- MxKxN loop, correct results, ~100 lines
    |                              Build this now.
    +-- HardwareMatMul (impl)   -- 512 multiplier cells, PSA tree, exact rounding
                                   Future work for sub-cycle accuracy.
```

The trait provides a single `compute()` method. `FastMatMul` implements the
mathematically equivalent operation using straightforward matrix arithmetic.
A future `HardwareMatMul` can model the physical pipeline stages (permute
network, 8x4 multiplier cells, 4-level post-add tree, accumulator combiner)
for sub-cycle accuracy work.

### File Organization

```
src/interpreter/execute/
    vector_matmul.rs        (NEW)  -- Engine trait + FastMatMul + dispatch
    vector_matmul_tables.rs (NEW)  -- Permute tables (generated, checked in)
    vector_config.rs        (EXISTING, minor cleanup)
    vector.rs               (EXISTING, simplify MAC dispatch)

tools/
    gen-matmul-tables.py    (NEW)  -- Table generator from constants.py
```

## Dispatch Flow

All MAC-family semantic ops route through a single entry point:

```
execute_wide()
    |
    SemanticOp::Mac | MatMul | AddMac | SubMac | NegMul | NegMatMul
        |
        v
    execute_matmul(op, ctx, semantic)
        |
        1. Read config register via get_config_register(op, ctx)
        2. Parse: MatMulConfig::from_config_word(conf, is_bf16)
        3. Look up permute table: lookup_geometry(amode, bmode, variant)
        4. Read 512-bit input vectors (X and Y)
        5. Read accumulator (or zero if !accumulate)
        6. Call engine: FastMatMul::compute(geometry, x, y, acc, config)
        7. Apply variant sign flags (sub0/sub1/sub2 from config + semantic)
        8. Write accumulator with correct Acc32/Acc64 packing
```

### Semantic Variant Mapping

The instruction mnemonic determines the base sign behavior. Config word
`sub0`/`sub1`/`sub2` bits provide additional negation control that composes
with the semantic.

| SemanticOp | Product | Acc1 | Acc2 | Description |
|-----------|---------|------|------|-------------|
| Mac / MatMul | + | accumulate | n/a | acc += A*B |
| NegMul / NegMatMul | - | accumulate | n/a | acc -= A*B |
| AddMac | + | +acc1 | +acc2 | acc1 + acc2 + A*B |
| SubMac | + | +acc1 | -acc2 | acc1 - acc2 + A*B |

All variants use the same multiply engine. The only difference is post-
multiply sign application on the product and accumulator terms.

### No execute_half Path

MAC instructions always operate on 512-bit inputs and 1024-bit accumulators.
They must never fall through to `execute_wide_fallback` (which calls
`execute_half` twice). The new `execute_matmul` intercepts all MAC semantics
in `execute_wide` before the fallback.

## Multiply Logic (FastMatMul)

### Permute-Table-Driven Element Extraction

Each geometry has a precomputed table of multiply-accumulate "lanes":

```rust
struct MatMulLane {
    a_byte_offset: u16,  // byte position in 64-byte X vector
    b_byte_offset: u16,  // byte position in 64-byte Y vector
    output_index: u16,   // accumulator lane to add to
}
```

For a 4x8x8 int8xi8 matmul, there are 4*8*8 = 256 lanes. The fast path:

```rust
fn compute(geometry: &MatMulGeometry, x: &[u8; 64], y: &[u8; 64],
           acc: &mut [i64], x_signed: bool, y_signed: bool) {
    for lane in &geometry.lanes {
        let a = sign_ext_element(x, lane.a_byte_offset, geometry.bits_x, x_signed);
        let b = sign_ext_element(y, lane.b_byte_offset, geometry.bits_y, y_signed);
        acc[lane.output_index as usize] += a * b;
    }
}
```

This is mathematically equivalent to the hardware pipeline (permute ->
multiply -> post-add tree -> accumulate) because the lane table encodes the
complete permute-to-output mapping.

### BFloat16 Path

BFloat16 multiply uses f32 arithmetic throughout:

```rust
fn compute_bf16(geometry: &MatMulGeometry, x: &[u8; 64], y: &[u8; 64],
                acc: &mut [f32]) {
    for lane in &geometry.lanes {
        let a = bf16_to_f32(extract_bf16(x, lane.a_byte_offset));
        let b = bf16_to_f32(extract_bf16(y, lane.b_byte_offset));
        acc[lane.output_index as usize] += a * b;
    }
}
```

Accumulator stores f32 bits in the u64 lanes (one f32 per 32-bit half,
same Acc32 packing convention).

### Accumulator Packing

**Confirmed by hardware analysis:** MAC uses the same accumulator packing
convention as UPS/SRS.

- **Acc32 mode:** Two 32-bit values per u64: `u64[i] = lane[2*i] | (lane[2*i+1] << 32)`
- **Acc64 mode:** One 64-bit value per u64: `u64[i] = lane[i]`
- **Output ordering:** Row-major. For 4x8 matmul: output[0..7] = row 0, etc.

Unpacking before compute and repacking after uses the same helpers as
the UPS/SRS fix.

### Post-Multiply Sign Application

After the core multiply:

1. If `sub0` (or semantic is NegMul/NegMatMul): negate all products
2. If `sub1`: negate acc1 contribution
3. If `sub2`: negate acc2 contribution
4. If `shift16`: left-shift existing accumulator by 16 before adding
5. Combine: `result = acc_term + product_term`

Double-negation cancels (NegMul + sub0=1 = positive product).

## Build-Time Table Generation

### Script: `tools/gen-matmul-tables.py`

Reads `constants.py` from aietools and emits `vector_matmul_tables.rs`.

**Input:** `../aietools/data/aie_ml/lib/python_model/model/constants.py`

**Output:** `src/interpreter/execute/vector_matmul_tables.rs`

The output is **checked into the repository** rather than generated in
build.rs because:
1. aietools is not always available (CI, other developers)
2. The tables are hardware constants that change only with new silicon
3. Regeneration is a manual step: `python3 tools/gen-matmul-tables.py`

### Generated Data Structure

```rust
pub struct MatMulGeometry {
    pub amode: u32,
    pub bmode: u32,
    pub variant: u32,
    pub rows: u32,
    pub inner: u32,
    pub cols: u32,
    pub bits_x: u32,
    pub bits_y: u32,
    pub acc_cmb: u32,
    pub bfloat: bool,
    pub sparse: bool,
    pub lanes: &'static [(u16, u16, u16)],  // (a_offset, b_offset, output_idx)
}
```

### Lookup Function

```rust
pub fn lookup_geometry(amode: u32, bmode: u32, variant: u32) -> Option<&'static MatMulGeometry>
```

Replaces `CONFIG_GEOMETRY_TABLE` in `vector_config.rs` with richer data
that includes the actual permutation mapping, not just dimensions.

### Path Away From aietools Dependency

The permute tables encode a mathematical relationship fully determined by
(M, K, N, bits_x, bits_y, packing_order). For dense non-convolution modes:
- `A[m][k]` is at byte offset `(m * K + k) * (bits_x / 8)` in X
- `B[k][n]` is at byte offset `(k * N + n) * (bits_y / 8)` in Y

A future pure-Rust generator can compute these tables from geometry
parameters alone, eliminating the aietools dependency. The generated tables
serve as the verified reference for that transition. Uncommon modes
(convolution, complex, sparse) have more intricate mappings that will need
additional derivation work.

### Estimated Table Size

- ~10 dense geometries (phase 1)
- Largest: 4x16x8 (int8xi4) = 512 lanes = 3KB
- Total: ~15-20KB of const data

## Scope

### Phase 1 (This Plan): Dense Integer + BFloat16

- All dense (amode, bmode, variant=0) geometries
- Config word parsing and geometry lookup
- FastMatMul engine with permute-table-driven multiply
- BFloat16 f32-accumulator path
- Single and double accumulator modes (mac, addmac, submac)
- Sign extension, subtract, shift16, zero_acc flags
- **Target: ~274 test points (166 integer dense + 108 bf16 dense)**

### Phase 2 (Follow-On): Sparse Modes

- Sparse narrow and sparse wide variants
- Sparsity mask decoding from qx registers
- Extended permute tables with wider X-side window
- **Target: ~596 additional test points**

### Phase 3 (Future): Element-wise, Convolution, Complex

- variant != 0 geometries (element-wise, convolution, FFT)
- Not tested by current ISA harness; lower priority
- Complex multiply with sub_mask per-element negation

### Not In Scope

- HardwareMatMul (sub-cycle accurate pipeline model)
- Pure-Rust table derivation (replace gen script)
- Sparse mode support

## Changes Summary

### New Files
- `src/interpreter/execute/vector_matmul.rs` -- engine trait, FastMatMul, dispatch
- `src/interpreter/execute/vector_matmul_tables.rs` -- generated permute tables
- `tools/gen-matmul-tables.py` -- table generator script

### Modified Files
- `src/interpreter/execute/vector.rs` -- route MAC semantics to execute_matmul, delete old handlers
- `src/interpreter/execute/vector_config.rs` -- minor cleanup (geometry table superseded)
- `src/interpreter/execute/mod.rs` -- add `mod vector_matmul;` declaration

### Deleted Code
- `vector_mac()` -- element-wise placeholder
- `vector_matmul_dense()` -- element-wise despite name
- `vector_matmul_bf16()` -- element-wise bf16
- `vector_double_acc_mac()` -- element-wise double-acc

## Testing Strategy

- Unit tests in `vector_matmul.rs` using known input/output pairs
- Golden reference from HW ISA test baseline (batch_69 etc.)
- Table generator `--verify` mode for hand-computed spot checks
- Full ISA test suite run after implementation
