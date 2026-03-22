# Full-Width Vector Execution

## Problem

The AIE2 vector unit operates on three register widths:

| Class | Width | Example | Physical mapping |
|-------|-------|---------|------------------|
| w-register | 256-bit | wl0 | v0 (single) |
| x-register | 512-bit | x0 | v0 + v1 (pair) |
| cm-register | 1024-bit | cm0 | acc0 + acc1 (pair) |

The emulator's `execute_vector()` processes ALL operations by calling
`execute_half()` on 256-bit chunks. For 512-bit operations, it clones the
SlotOp, increments VectorReg indices by 1, and calls `execute_half()` a
second time. This produces correct results for element-wise operations
(vadd, vmul) where each half is independent, but breaks for:

1. **Cross-half data access**: VPUSH shifts elements across the 256-bit
   boundary. VSHUFFLE interleaves lanes from both halves.
2. **Whole-register reductions**: MAC/matmul operations where all 512 input
   bits contribute to all output bits via tiled matrix multiply.
3. **Accumulator width mismatch**: `increment_vector_regs()` only increments
   VectorReg, not AccumReg. CM-class (1024-bit) accumulator operations
   never process their high half correctly.

This is the root cause behind 0% accuracy for VPUSH, VSHUFFLE, and the
inability to wire up config-word-driven MAC geometry.

## Design: Two-Tier Dispatch

Add a new `execute_wide()` dispatch path alongside the existing
`execute_half()`. When `is_wide_vector` is true, `execute_vector()`
routes to `execute_wide()` with full-width data instead of calling
`execute_half()` twice.

### Data Types

```rust
/// 512-bit vector: 16 x u32 lanes (two consecutive 256-bit w-registers).
pub type Vec512 = [u32; 16];

/// 1024-bit accumulator: 16 x u64 lanes (two consecutive 512-bit bm-registers).
pub type Acc1024 = [u64; 16];
```

### Register File Extensions

New methods on VectorRegFile and AccumRegFile. Both include debug
assertions that the base register is even (hardware enforces pair
alignment for x-registers and cm-registers):

```rust
impl VectorRegFile {
    /// Read 512-bit x-register (concatenate two consecutive w-registers).
    /// base_reg must be even (x0 = v0+v1, x2 = v4+v5, etc.).
    pub fn read_wide(&self, base_reg: u8) -> Vec512 {
        debug_assert!(base_reg % 2 == 0, "wide read from odd base register {}", base_reg);
        let lo = self.read(base_reg);
        let hi = self.read(base_reg + 1);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Write 512-bit x-register (split across two consecutive w-registers).
    /// base_reg must be even.
    pub fn write_wide(&mut self, base_reg: u8, data: Vec512) {
        debug_assert!(base_reg % 2 == 0, "wide write to odd base register {}", base_reg);
        let mut lo = [0u32; 8];
        let mut hi = [0u32; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.write(base_reg, lo);
        self.write(base_reg + 1, hi);
    }
}

impl AccumRegFile {
    /// Read 1024-bit cm-register (concatenate two consecutive bm-registers).
    /// base_reg must be even (cm0 = acc0+acc1, cm2 = acc2+acc3, etc.).
    pub fn read_wide(&self, base_reg: u8) -> Acc1024 {
        debug_assert!(base_reg % 2 == 0, "wide accum read from odd base register {}", base_reg);
        let lo = self.read(base_reg);
        let hi = self.read(base_reg + 1);
        let mut result = [0u64; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Write 1024-bit cm-register (split across two consecutive bm-registers).
    /// base_reg must be even.
    pub fn write_wide(&mut self, base_reg: u8, data: Acc1024) {
        debug_assert!(base_reg % 2 == 0, "wide accum write to odd base register {}", base_reg);
        let mut lo = [0u64; 8];
        let mut hi = [0u64; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.write(base_reg, lo);
        self.write(base_reg + 1, hi);
    }
}
```

No changes to the underlying storage layout (32 x 256-bit vector registers,
8 x 512-bit accumulator registers).

### Dispatch Logic

```rust
pub fn execute_vector(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let semantic = match op.semantic { Some(s) => s, None => return false };
    if !op.is_vector { return false; }
    let et = op.element_type.unwrap_or(ElementType::Int32);

    if op.is_wide_vector {
        // Full-width path: read 512-bit inputs, dispatch, write 512-bit outputs.
        Self::execute_wide(op, ctx, semantic, et)
    } else {
        // Narrow path: single 256-bit half (unchanged from current code).
        Self::execute_half(op, ctx, semantic, et)
    }
}
```

The existing `is_wide_vector` logic already removes the `if handled &&
op.is_wide_vector { clone + increment + execute_half }` block. Wide ops
go through `execute_wide()` exclusively.

### Wide Execution Path

`execute_wide()` is a new dispatch function with the same match structure
as `execute_half()`, but its helper functions operate on `Vec512` and
`Acc1024` instead of `[u32; 8]` and `[u64; 8]`.

**Source/dest helpers:**

```rust
fn get_wide_vector_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> Vec512;
fn get_wide_acc_source(op: &SlotOp, ctx: &ExecutionContext) -> (u8, Acc1024);
fn write_wide_vector_dest(op: &SlotOp, ctx: &mut ExecutionContext, data: Vec512);
fn write_wide_acc_dest(op: &SlotOp, ctx: &mut ExecutionContext, data: Acc1024);
```

**Element-wise operations** (Add, Sub, Mul, Min, Max, And, Or, Xor, Shl,
Srl, Sra, Cmp, etc.) call the same per-element math as the narrow path,
just looping over 16 elements:

```rust
SemanticOp::Add => {
    let a = Self::get_wide_vector_source(op, ctx, 0);
    let b = Self::get_wide_vector_source(op, ctx, 1);
    let result = wide_vector_add(&a, &b, et);
    Self::write_wide_vector_dest(op, ctx, result);
    true
}
```

Where `wide_vector_add` either calls the narrow `vector_add` on each half,
or loops over 16 elements directly. The simplest implementation:

```rust
fn wide_element_wise(
    a: &Vec512, b: &Vec512, et: ElementType,
    op_fn: fn(&[u32; 8], &[u32; 8], ElementType) -> [u32; 8],
) -> Vec512 {
    let a_lo: [u32; 8] = a[..8].try_into().unwrap();
    let a_hi: [u32; 8] = a[8..].try_into().unwrap();
    let b_lo: [u32; 8] = b[..8].try_into().unwrap();
    let b_hi: [u32; 8] = b[8..].try_into().unwrap();
    let r_lo = op_fn(&a_lo, &b_lo, et);
    let r_hi = op_fn(&a_hi, &b_hi, et);
    let mut result = [0u32; 16];
    result[..8].copy_from_slice(&r_lo);
    result[8..].copy_from_slice(&r_hi);
    result
}
```

This is a bridge: element-wise ops reuse existing `[u32; 8]` functions
without rewriting them. Only cross-half ops need new implementations.

**Cross-half operations** get full-width implementations:

- `SemanticOp::VectorInsert` (VPUSH): shift-and-insert across full 512 bits
- `SemanticOp::Shuffle` (VSHUFFLE): two-input shuffle with full visibility
- `SemanticOp::Align` (VSHIFT): concatenate-and-shift across full width
- `SemanticOp::VectorExtract`: extract from any position in 512 bits
- `SemanticOp::VectorBroadcast` (VEXTBCST): extract-then-broadcast from full vector
- MAC family: config-word-driven tiled matmul on full inputs

**Accumulator operations** (VADD/VSUB/VNEG on cm-class):

These read `Acc1024`, operate on all 16 lanes, and write `Acc1024`.
The config word (from the scalar register source) controls zero_acc,
shift16, sub_acc1, sub_acc2 as currently implemented.

### Accumulator Register Mapping

The decoder maps cm-class operands via `RegisterKind::Accumulator`,
producing `AccumReg(n)` where `n` is the raw 4-bit field value from
the instruction encoding. For cm0, this is `AccumReg(0)`, meaning
"the pair starting at acc0" (acc0 = low half, acc1 = high half).
The `read_wide(0)` / `write_wide(0)` methods handle the pairing.

The decoder already emits the correct base register index -- it does
NOT need modification. The bug is in the execution path, not the
decoder. Specifically:

1. The `increment_vector_regs()` function skips AccumReg, so the
   high half of cm-class operands is never processed by the old
   clone+increment path.
2. The `execute_half()` handlers read `AccumReg(n)` as a 512-bit
   register, missing the second half entirely.

Both are fixed by routing wide ops through `execute_wide()` which
uses `read_wide()` / `write_wide()` on the AccumReg index directly.

**Which operations are always cm-class (wide accumulator)?**

All accumulator-to-accumulator operations use cm-class registers
exclusively: VADD, VSUB, VNEGADD, VNEGSUB, VNEG (and .f variants).
The `Accumulate` and `Neg` SemanticOps with AccumReg sources always
get the wide path. The `is_wide_vector` flag may or may not be set
for these (since they have no Vector512 operands), so the dispatch
should also check for AccumReg-only operations explicitly.

### SRS/UPS Operations

SRS (shift-round-saturate) reads an accumulator and writes a vector.
UPS (unsigned promote/shift) reads a vector and writes an accumulator.
Both are element-wise: each output lane depends on exactly one input
lane. They are safe for the `wide_element_wise()` bridge -- split
into halves, process each independently, concatenate. No cross-half
access needed.

### Fallback for Unhandled Wide Ops

During phased rollout, `execute_wide()` will not initially cover
every SemanticOp. For any SemanticOp not yet handled in `execute_wide()`,
fall back to the `wide_element_wise()` bridge (split into halves,
call `execute_half()` on each). This preserves the current behavior
for ops that happen to be element-wise, and produces the same
(wrong-but-no-worse) results for ops that need true wide handling
but haven't been ported yet. Log a trace-level message for unhandled
ops to track migration progress.

### VSEL (VectorSelect)

VSEL takes a scalar mask and selects between two vector sources
per-element. In 512-bit mode, the scalar mask covers 16/32/64
elements depending on element type (32-bit mask for 16 x Int32,
etc.). This is element-wise: each output element depends only on
the corresponding input elements and one bit of the mask. Safe for
the bridge.

### Migration Strategy

Not all wide-vector SemanticOps need new implementations on day one.
The bridge function `wide_element_wise()` handles any operation that
is per-element by splitting into halves and calling the existing
narrow function. Only operations that are fundamentally broken by
half-splitting need new full-width implementations.

**Phase 1 (immediate):** Infrastructure + element-wise bridge
- Add Vec512/Acc1024 types and register file methods
- Add execute_wide() dispatch with wide_element_wise() bridge
- Remove the clone+increment+execute_half block
- All element-wise wide ops work correctly via bridge

**Phase 2 (cross-half ops):** Purpose-built wide implementations
- VPUSH: full-width shift-and-insert
- VSHUFFLE: two-input shuffle with modifier
- VSHIFT/Align: full-width concatenate-and-shift
- VEXTBCST: extract from any position in 512-bit source
- Accumulator add/sub/neg: full-width with config word

**Phase 3 (MAC):** Config-word-driven matrix multiply
- Wire MatMulConfig::from_config_word() into MAC dispatch
- Full-width matmul kernel on Vec512 inputs -> Acc1024 output
- Permutation tables (future: derive from hardware observation)

### Files Modified

| File | Change |
|------|--------|
| `src/interpreter/state/registers.rs` | Add `read_wide`/`write_wide` to VectorRegFile and AccumRegFile |
| `src/interpreter/execute/vector.rs` | Add `execute_wide()`, wide source/dest helpers, wide_element_wise bridge, remove clone+increment block |
| `src/interpreter/execute/vector.rs` | Purpose-built wide implementations for cross-half ops |
| `src/interpreter/execute/vector_matmul.rs` | `matmul_config_driven` signature updated: `(acc: &mut Acc1024, a: &Vec512, b: &Vec512, config)`. Internal element extraction uses flat indexing into the 16-element arrays instead of lo/hi pairs. |
| `src/interpreter/execute/vector_config.rs` | No changes (from_config_word already exists) |

### What Does NOT Change

- Register file storage layout (32 x `[u32; 8]`, 8 x `[u64; 8]`)
- SlotOp structure and Operand enum
- Decoder logic (is_wide_vector detection already correct)
- Narrow-only (256-bit) instruction execution path
- Element-wise operation math functions
- Memory operations (load/store)
- Scalar execution

### Testing

- Existing unit tests continue to pass (they test narrow ops)
- New unit tests for Vec512/Acc1024 register read/write
- New unit tests for wide element-wise bridge (verify same results as narrow)
- New unit tests for VPUSH, VSHUFFLE, VSHIFT with cross-half data
- ISA test sweep measures overall accuracy improvement
