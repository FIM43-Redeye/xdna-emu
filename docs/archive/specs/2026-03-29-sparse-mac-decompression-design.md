# Sparse MAC Pair-Routing Decompression

**Date**: 2026-03-29
**Status**: Approved
**Scope**: Fix 0% pass rate on all 452 sparse MAC ISA test points

## Problem

Every sparse MAC test fails (0/452). Three compounding bugs:

1. **B indexing out of bounds**: Sparse geometries double the inner dimension
   (e.g., i8xi8: inner=16 vs dense inner=8). The multiply loop indexes B up
   to 127 bytes, but `Vec512` is only 64 bytes. `extract_element_512` returns
   0 for out-of-range accesses, so half of all B reads are zero.

2. **Only reading 64 of 128 mask bits**: `read_u64_low()` returns bits [0..63].
   The q register is 128 bits. Every sparse geometry needs all 128 mask bits
   (1 per byte of decompressed B).

3. **No decompression of compressed B data**: The hardware treats B data in
   the x register as compressed (only non-zero values, arranged as pairs by
   `vldb.sparse.fill`). The mask tells the vmac instruction where each pair
   element belongs in the full 128-byte B matrix. Our code treats B as dense
   data and uses the mask as a skip filter.

## Hardware Architecture

Sparse MAC operates in two stages:

### Stage 1: Partial decompression (vldb.sparse.fill)

Loads compressed sparse data from memory into a qx composite register
(512-bit x data + 128-bit q mask). The load processes the memory-format
mask (64 bits per half) in groups of 4 bits, consuming 0-2 compressed
values per group and producing a pair of bytes. Each qx register holds
32 such pairs (64 bytes total) plus the 128-bit mask.

The documented partial decompression table (from aie_doc.hpp):

| Mask [3:0] | Pair[0] (hi) | Pair[1] (lo) | Consumed |
|:----------:|:------------:|:------------:|:--------:|
| 0000       | 0            | 0            | 0        |
| 0001       | 0            | A            | 1        |
| 0010       | 0            | B            | 1        |
| 0011       | B            | A            | 2        |
| 0100       | C            | 0            | 1        |
| 0101       | C            | A            | 2        |
| 0110       | C            | B            | 2        |
| 1000       | D            | 0            | 1        |
| 1001       | D            | A            | 2        |
| 1010       | D            | B            | 2        |
| 1100       | D            | C            | 2        |

A/B/C/D label the compressed value associated with bit 0/1/2/3 respectively.
Pair[0] holds the value for the higher set-bit position; pair[1] holds the
value for the lower set-bit position.

Patterns with >2 set bits (0111, 1011, 1101, 1110, 1111) violate the 2:4
structured sparsity constraint and are not documented. Hardware behavior for
these is deterministic but unspecified; a 16-pattern HW validation test can
determine it empirically.

### Stage 2: Pair routing (vmac instruction)

The vmac instruction reads x as 32 pairs and uses the 128-bit mask to route
each pair's elements to their correct positions in the full 128-byte B matrix:

For each group g (0..31):
- `mask4 = (mask >> (4*g)) & 0xF`
- `hi = x_bytes[2*g]`, `lo = x_bytes[2*g + 1]`
- Find set-bit positions within mask4
- Route `lo` to the lowest set-bit position: `B[4*g + lowest] = lo`
- Route `hi` to the highest set-bit position: `B[4*g + highest] = hi`
- All clear-bit positions: `B[4*g + pos] = 0`

Output: 128 bytes = full dense B matrix, ready for standard matrix multiply.

Note: the mask operates at byte granularity (1 bit per byte), not element
granularity. For 16-bit elements, each element spans 2 consecutive mask bits
(both set or both clear in valid data). For 4-bit elements, each mask bit
controls a byte containing 2 nibbles. The pair routing and decompression
work uniformly at byte level regardless of element width; element
interpretation happens in the multiply loop.

### Why sequential decompression is wrong

Sequential decompression walks the entire 128-bit mask consuming bytes from
a single stream. This makes each group's data depend on previous groups'
popcount. Pair routing reads from fixed positions (2*g, 2*g+1), independent
of other groups. For raw test data loaded via vlda (not vldb.sparse.fill),
these produce different results:

```
Group 0: mask=0001 (1 bit), Group 1: mask=0011 (2 bits)
Sequential:   group0 -> x[0];      group1 -> x[1], x[2]
Pair-routing: group0 -> x[0],x[1]; group1 -> x[2], x[3]
```

Our ISA tests load via vlda+lda, so pair routing is the correct model.

## Design

### 1. Pair-routing decompression function

New pure function in `vector_matmul.rs`:

```rust
fn sparse_pair_route(compressed: &[u8; 64], mask: u128) -> [u8; 128]
```

Processes 32 groups of 4 mask bits. For each group, reads pair from
compressed[2*g..2*g+2] and routes to the two set-bit positions within the
group's 4-byte output span. For invalid groups (>2 set bits), only the
lowest and highest positions receive data; middle positions get 0.

### 2. Full mask register reading

Add `read_u128()` convenience method to `MaskRegisterFile`:

```rust
pub fn read_u128(&self, reg: u8) -> u128
```

Assembles from the existing `[u32; 4]` storage. No new data structures.

### 3. Byte-array element extraction

New function for extracting elements from the 128-byte decompressed buffer:

```rust
fn extract_element_bytes(src: &[u8; 128], byte_idx: usize, bits: u32, signed: bool) -> i64
```

Simpler than the existing `extract_element_512` (which indexes u32 words
and shifts): directly indexes bytes, handles 4/8/16/32-bit element widths
with sign extension.

### 4. Updated sparse multiply signature

```rust
fn matmul_sparse_config_driven(
    acc: &mut Acc1024,
    a: &[u8; 128],     // was &Vec512, now byte array (zero-padded if A < 128 bytes)
    b: &[u8; 128],     // was &Vec512, now pair-routed decompressed output
    config: &MatMulConfig,
)
```

The mask parameter is removed -- mask is consumed by decompression before
this function is called. The multiply loop is unchanged in shape (R x K x C)
but indexes the 128-byte buffers via `extract_element_bytes`.

### 5. Updated operand reading

`get_sparse_operands` changes return type from `(Vec512, Vec512, u64)` to
`([u8; 128], [u8; 64], u128)`:

- **A**: Read from VectorReg (512-bit wide for sparse narrow) or quad
  VectorReg (1024-bit for sparse wide). Extract as bytes into [u8; 128],
  zero-padding if A < 128 bytes. This fixes the existing TODO for sparse
  wide support.
- **B compressed**: Extract 64 raw bytes from the x register in the qx
  composite (ControlReg mapping unchanged).
- **Mask**: `ctx.mask.read_u128(qx_idx)` instead of `read_u64_low`.

### 6. Dense path untouched

The dense multiply path (`matmul_config_driven`) is completely unchanged.
No regression risk for the 107/220 currently-passing dense MAC tests.

## Integration flow

```
execute_matmul()
  |
  |-- is_sparse?
  |     |
  |     yes: get_sparse_operands() -> (a_bytes, b_compressed, mask)
  |     |    sparse_pair_route(&b_compressed, mask) -> b_decompressed
  |     |    matmul_sparse_config_driven(&mut acc, &a_bytes, &b_decompressed, &config)
  |     |
  |     no:  existing dense path (unchanged)
  |
  |-- write accumulator (unchanged)
```

## Sparse geometry reference

All sparse geometries produce 128-byte decompressed B and consume exactly
64 compressed bytes (32 pairs):

| Geometry | A bytes | B decomp bytes | Mask bits | Inner | Cols |
|----------|---------|---------------|-----------|-------|------|
| i8xi4    | 128     | 128           | 128       | 32    | 8    |
| i8xi8    | 64      | 128           | 128       | 16    | 8    |
| i16xi8   | 64      | 128           | 128       | 16    | 8    |
| i16xi16  | 32      | 128           | 128       | 8     | 8    |
| bf16     | 128     | 128           | 128       | 16    | 4    |

A > 64 bytes (i8xi4, bf16) requires sparse wide (y register = 1024 bits).

## Testing strategy

1. **Unit: `sparse_pair_route`** -- All 11 valid 4-bit patterns with known
   data. A few invalid patterns (3-4 set bits) to lock in chosen behavior.
   Full 128-bit mask with mixed groups.

2. **Unit: integrated multiply** -- Known A, known compressed B + mask that
   decompresses to known dense B, verify multiply result. Cover i8xi8 and
   bf16 at minimum.

3. **ISA regression** -- Full suite with `--no-hw` against existing HW
   baselines. Target: significant improvement from 0/452.

4. **Future: 16-pattern HW validation** -- Exercises each of 16 possible
   4-bit mask patterns on real hardware with known data. Definitively
   validates or corrects invalid-mask handling. Not part of this
   implementation.

## Files to change

| File | Change |
|------|--------|
| `src/interpreter/state/registers.rs` | Add `read_u128()` to MaskRegisterFile |
| `src/interpreter/execute/vector_matmul.rs` | `sparse_pair_route`, updated `get_sparse_operands`, updated `matmul_sparse_config_driven`, new `extract_element_bytes` |

## Verification

```bash
TMPDIR=/tmp/claude-1000 cargo test --lib
TMPDIR=/tmp/claude-1000 cargo build --release
# Run ISA tests EMU-only against existing HW baselines:
XDNA_EMU=release ./scripts/isa-test.sh --no-hw
```
