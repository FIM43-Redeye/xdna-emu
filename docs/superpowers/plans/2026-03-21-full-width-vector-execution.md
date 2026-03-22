# Full-Width Vector Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the half-splitting architecture so 512-bit vector and 1024-bit accumulator operations process their full width, enabling correct VPUSH, VSHUFFLE, MAC, and accumulator arithmetic.

**Architecture:** Two-tier dispatch in `execute_vector()`. Narrow ops (256-bit, `is_wide_vector=false`) continue through `execute_half()` unchanged. Wide ops (512-bit, `is_wide_vector=true`) go through a new `execute_wide()` that reads/writes full-width data. Element-wise wide ops reuse existing narrow math via a bridge function. Cross-half ops get purpose-built implementations.

**Tech Stack:** Rust, xdna-emu emulator codebase.

**Spec:** `docs/superpowers/specs/2026-03-21-full-width-vector-execution-design.md`

---

### Task 1: Add Vec512 / Acc1024 Types and Register File Methods

**Files:**
- Modify: `src/interpreter/state/registers.rs:337-382` (VectorRegisterFile methods)
- Modify: `src/interpreter/state/registers.rs:438-473` (AccumulatorRegisterFile methods)

- [ ] **Step 1: Write failing tests for read_wide/write_wide**

Add to the existing test module in `registers.rs`:

```rust
#[test]
fn test_vector_read_wide() {
    let mut vrf = VectorRegisterFile::new();
    vrf.write(0, [1, 2, 3, 4, 5, 6, 7, 8]);
    vrf.write(1, [9, 10, 11, 12, 13, 14, 15, 16]);
    let wide = vrf.read_wide(0);
    assert_eq!(wide, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
}

#[test]
fn test_vector_write_wide() {
    let mut vrf = VectorRegisterFile::new();
    let data: [u32; 16] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    vrf.write_wide(0, data);
    assert_eq!(vrf.read(0), [10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(vrf.read(1), [90, 100, 110, 120, 130, 140, 150, 160]);
}

#[test]
fn test_accum_read_wide() {
    let mut arf = AccumulatorRegisterFile::new();
    arf.write(0, [100, 200, 300, 400, 500, 600, 700, 800]);
    arf.write(1, [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]);
    let wide = arf.read_wide(0);
    assert_eq!(wide, [100, 200, 300, 400, 500, 600, 700, 800,
                       900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]);
}

#[test]
fn test_accum_write_wide() {
    let mut arf = AccumulatorRegisterFile::new();
    let mut data = [0u64; 16];
    for i in 0..16 { data[i] = (i as u64 + 1) * 10; }
    arf.write_wide(0, data);
    assert_eq!(arf.read(0), [10, 20, 30, 40, 50, 60, 70, 80]);
    assert_eq!(arf.read(1), [90, 100, 110, 120, 130, 140, 150, 160]);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib registers 2>&1 | tail -10`
Expected: FAIL with "no method named `read_wide`"

- [ ] **Step 3: Add type aliases and methods**

Add type aliases at the top of `registers.rs` (after existing use statements):

```rust
/// 512-bit vector data: two consecutive 256-bit w-registers.
pub type Vec512 = [u32; 16];

/// 1024-bit accumulator data: two consecutive 512-bit bm-registers.
pub type Acc1024 = [u64; 16];
```

Add methods to VectorRegisterFile (after `write_bytes` at line ~382):

```rust
    /// Read a 512-bit x-register (two consecutive w-registers).
    /// The decoder maps x0 -> vreg 0, x1 -> vreg 2, etc. (reg * 2).
    /// base_reg is already the decoded index (0, 2, 4, ...).
    pub fn read_wide(&self, base_reg: u8) -> Vec512 {
        debug_assert!(
            base_reg % 2 == 0,
            "wide vector read from odd base register {}",
            base_reg
        );
        let lo = self.read(base_reg);
        let hi = self.read(base_reg + 1);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Write a 512-bit x-register (split across two consecutive w-registers).
    pub fn write_wide(&mut self, base_reg: u8, data: Vec512) {
        debug_assert!(
            base_reg % 2 == 0,
            "wide vector write to odd base register {}",
            base_reg
        );
        let mut lo = [0u32; 8];
        let mut hi = [0u32; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.write(base_reg, lo);
        self.write(base_reg + 1, hi);
    }
```

Add methods to AccumulatorRegisterFile (after `accumulate` at line ~473):

```rust
    /// Read a 1024-bit cm-register (two consecutive bm-registers).
    /// cm0 = (acc0, acc1), cm2 = (acc2, acc3), etc.
    pub fn read_wide(&self, base_reg: u8) -> Acc1024 {
        debug_assert!(
            base_reg % 2 == 0,
            "wide accum read from odd base register {}",
            base_reg
        );
        let lo = self.read(base_reg);
        let hi = self.read(base_reg + 1);
        let mut result = [0u64; 16];
        result[..8].copy_from_slice(&lo);
        result[8..].copy_from_slice(&hi);
        result
    }

    /// Write a 1024-bit cm-register (split across two consecutive bm-registers).
    pub fn write_wide(&mut self, base_reg: u8, data: Acc1024) {
        debug_assert!(
            base_reg % 2 == 0,
            "wide accum write to odd base register {}",
            base_reg
        );
        let mut lo = [0u64; 8];
        let mut hi = [0u64; 8];
        lo.copy_from_slice(&data[..8]);
        hi.copy_from_slice(&data[8..]);
        self.write(base_reg, lo);
        self.write(base_reg + 1, hi);
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib registers 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/state/registers.rs
git commit -m "feat(emu): add Vec512/Acc1024 types and read_wide/write_wide methods"
```

---

### Task 2: Add Wide Source/Dest Helpers to VectorAlu

**Files:**
- Modify: `src/interpreter/execute/vector.rs:636-717` (helper functions area)

These helpers parallel the existing narrow helpers but operate on `Vec512`
and `Acc1024`. They are needed by `execute_wide()` in the next task.

- [ ] **Step 1: Write the wide helper functions**

Add after the existing `write_vector_dest` function (around line 717).
Import the new types at the top of the file:

```rust
use crate::interpreter::state::registers::{Vec512, Acc1024};
```

Then add the helpers:

```rust
    // ========== Wide (512-bit / 1024-bit) Helpers ==========

    /// Read the nth VectorReg source as a full 512-bit value.
    fn get_wide_vec_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> Vec512 {
        let mut vec_count = 0;
        for src in &op.sources {
            if let Operand::VectorReg(r) = src {
                if vec_count == idx {
                    return ctx.vector.read_wide(*r);
                }
                vec_count += 1;
            }
        }
        [0u32; 16]
    }

    /// Read two wide vector sources.
    fn get_two_wide_vec_sources(
        op: &SlotOp,
        ctx: &ExecutionContext,
    ) -> (Vec512, Vec512) {
        let a = Self::get_wide_vec_source(op, ctx, 0);
        let b = Self::get_wide_vec_source(op, ctx, 1);
        (a, b)
    }

    /// Write a 512-bit result to the vector destination.
    fn write_wide_vec_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: Vec512) {
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write_wide(*r, value);
        } else {
            log::error!(
                "[VECTOR_WIDE] write_wide_vec_dest: expected VectorReg dest, got {:?}",
                op.dest
            );
        }
    }

    /// Read the accumulator destination as a 1024-bit cm-register.
    fn get_wide_acc_dest_value(op: &SlotOp, ctx: &ExecutionContext) -> (u8, Acc1024) {
        let reg = Self::get_acc_dest(op);
        (reg, ctx.accumulator.read_wide(reg))
    }

    /// Write a 1024-bit result to the accumulator destination.
    fn write_wide_acc_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: Acc1024) {
        let reg = Self::get_acc_dest(op);
        ctx.accumulator.write_wide(reg, value);
    }

    /// Read an AccumReg source as a 1024-bit cm-register.
    fn get_wide_acc_source(op: &SlotOp, ctx: &ExecutionContext) -> (u8, Acc1024) {
        let reg = Self::get_acc_source(op);
        (reg, ctx.accumulator.read_wide(reg))
    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check 2>&1 | tail -5`
Expected: compiles (warnings OK)

- [ ] **Step 3: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "feat(emu): add wide (512-bit/1024-bit) source/dest helpers"
```

---

### Task 3: Add wide_element_wise Bridge and execute_wide Dispatch

**Files:**
- Modify: `src/interpreter/execute/vector.rs:50-95` (execute + increment_vector_regs)

This is the core change: replace the clone+increment+execute_half block
with a dispatch to `execute_wide()`. The bridge function lets element-wise
ops work immediately without rewriting them.

- [ ] **Step 1: Write the bridge function**

Add after the wide helpers from Task 2:

```rust
    /// Bridge: apply a narrow element-wise function to a wide vector.
    ///
    /// Splits Vec512 into two [u32; 8] halves, applies the function to
    /// each half independently, and concatenates the results. Works for
    /// any operation where each output element depends only on
    /// corresponding input elements.
    fn wide_element_wise_unary(
        a: &Vec512,
        et: ElementType,
        op_fn: fn(&[u32; 8], ElementType) -> [u32; 8],
    ) -> Vec512 {
        let a_lo: [u32; 8] = a[..8].try_into().unwrap();
        let a_hi: [u32; 8] = a[8..].try_into().unwrap();
        let r_lo = op_fn(&a_lo, et);
        let r_hi = op_fn(&a_hi, et);
        let mut result = [0u32; 16];
        result[..8].copy_from_slice(&r_lo);
        result[8..].copy_from_slice(&r_hi);
        result
    }

    /// Bridge: apply a narrow two-input element-wise function to wide vectors.
    fn wide_element_wise_binary(
        a: &Vec512,
        b: &Vec512,
        et: ElementType,
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

- [ ] **Step 2: Add execute_wide dispatch function**

Add after the bridge functions. This covers element-wise ops via bridge
and accumulator ops directly. Cross-half ops (VPUSH, VSHUFFLE, MAC) get
placeholder entries that fall through to the bridge for now -- they will
be implemented in subsequent tasks.

```rust
    /// Execute a 512-bit wide vector operation.
    ///
    /// Unlike execute_half which processes 256-bit chunks independently,
    /// this reads full 512-bit inputs and writes full 512-bit outputs.
    /// Element-wise ops use the bridge to reuse narrow math. Cross-half
    /// ops have dedicated implementations.
    fn execute_wide(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        match semantic {
            // ========== Element-wise arithmetic (bridge) ==========
            SemanticOp::Add => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_add);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Sub => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_sub);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Mul => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_mul);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Min => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_min);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Max => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_max);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            // Bitwise ops: type-agnostic, no ElementType parameter.
            // Use inline closures rather than the bridge since signatures differ.
            SemanticOp::And => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_and(&a_lo, &b_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_and(&a_hi, &b_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Or => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_or(&a_lo, &b_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_or(&a_hi, &b_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Xor => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let b_lo: [u32; 8] = b[..8].try_into().unwrap();
                let b_hi: [u32; 8] = b[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_xor(&a_lo, &b_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_xor(&a_hi, &b_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Not => {
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let a_lo: [u32; 8] = a[..8].try_into().unwrap();
                let a_hi: [u32; 8] = a[8..].try_into().unwrap();
                let mut result = [0u32; 16];
                result[..8].copy_from_slice(&Self::vector_bitwise_not(&a_lo));
                result[8..].copy_from_slice(&Self::vector_bitwise_not(&a_hi));
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::Neg => {
                let a = Self::get_wide_vec_source(op, ctx, 0);
                let result = Self::wide_element_wise_unary(&a, et, Self::vector_negate);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // ========== Comparison (bridge) ==========
            SemanticOp::Cmp => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_cmp_eq);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::SetGe => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_ge);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
            SemanticOp::SetLt => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_compare_lt);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }

            // ========== Accumulator ops (full-width) ==========
            SemanticOp::Accumulate => {
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));
                if has_acc_source {
                    Self::execute_acc_add_sub(op, ctx);
                } else {
                    // Legacy: vector-into-accumulator path, use bridge
                    let src = Self::get_wide_vec_source(op, ctx, 0);
                    let acc_reg = Self::get_acc_dest(op);
                    let src_lo: [u32; 8] = src[..8].try_into().unwrap();
                    let src_hi: [u32; 8] = src[8..].try_into().unwrap();
                    Self::vector_accumulate(ctx, acc_reg, &src_lo, et);
                    Self::vector_accumulate(ctx, acc_reg + 1, &src_hi, et);
                }
                true
            }

            // ========== Copy / Clear ==========
            SemanticOp::Copy => {
                let a = Self::get_wide_vec_source(op, ctx, 0);
                Self::write_wide_vec_dest(op, ctx, a);
                true
            }
            SemanticOp::VectorClear => {
                Self::write_wide_vec_dest(op, ctx, [0u32; 16]);
                true
            }

            // ========== Fallback: split into halves ==========
            //
            // Any SemanticOp not explicitly handled above falls through
            // here. This calls execute_half on each 256-bit half, which
            // preserves current behavior for ops not yet ported to
            // full-width. Log at trace level to track migration progress.
            _ => {
                log::trace!(
                    "[VECTOR_WIDE] fallback to half-split for {:?} (not yet ported)",
                    semantic
                );
                let handled = Self::execute_half(op, ctx, semantic, et);
                if handled {
                    let mut hi_op = op.clone();
                    Self::increment_vector_regs(&mut hi_op);
                    Self::execute_half(&hi_op, ctx, semantic, et);
                }
                handled
            }
        }
    }
```

- [ ] **Step 3: Modify execute_vector to dispatch to execute_wide**

Replace the current `execute_vector` body (lines 58-82) so that wide
ops go through `execute_wide()` instead of the clone+increment pattern:

```rust
        let et = op.element_type.unwrap_or(ElementType::Int32);

        log::trace!("[VECTOR_ALU] Checking semantic={:?} element_type={:?} dest={:?}",
            semantic, op.element_type, op.dest);

        // Determine if this needs full-width processing:
        // 1. is_wide_vector: instruction has Vector512 operands (x-registers)
        // 2. AccumReg-only ops (VADD, VSUB, VNEG on cm-class accumulators)
        //    These don't have Vector512 operands but always operate on
        //    1024-bit cm registers. Detect by checking for AccumReg sources
        //    without any VectorReg sources.
        let has_acc_source = op.sources.iter()
            .any(|s| matches!(s, Operand::AccumReg(_)));
        let has_vec_source = op.sources.iter()
            .any(|s| matches!(s, Operand::VectorReg(_)));
        let is_accum_only = has_acc_source && !has_vec_source;

        if op.is_wide_vector || is_accum_only {
            // Full-width path: 512-bit vectors, 1024-bit accumulators.
            Self::execute_wide(op, ctx, semantic, et)
        } else {
            // Narrow path: single 256-bit half.
            Self::execute_half(op, ctx, semantic, et)
        }
```

This removes the clone+increment+execute_half block entirely. The
`increment_vector_regs` function is kept for the fallback path inside
`execute_wide()`.

Also remove the separate `execute_mac_legacy` function and its
clone+increment block (around line 1003-1014). MAC ops will go through
`execute_wide()` like all other wide ops. The `execute_half` still has
the MAC handler for narrow-only (non-wide) MAC variants if any exist.

- [ ] **Step 4: Run all tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: all 2536+ tests pass

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "feat(emu): two-tier vector dispatch with execute_wide and element-wise bridge"
```

---

### Task 4: Port Accumulator Ops to execute_wide

**Files:**
- Modify: `src/interpreter/execute/vector.rs` (execute_acc_add_sub and Neg handler)

The existing `execute_acc_add_sub` already handles cm-class registers
with both halves via a `for half in 0..2` loop. Migrate it to use
`Acc1024` for cleaner code and proper integration with the wide path.
Also ensure the `Neg` handler for accumulator negation is routed through
`execute_wide`.

- [ ] **Step 1: Update execute_acc_add_sub to use Acc1024**

Refactor the function to read/write `Acc1024` instead of looping over
halves manually. The core logic (config word parsing, sign handling)
stays the same. See the current implementation at lines 917-997.

The key change: replace the `for half in 0..2u8` loop with single
`read_wide` / `write_wide` calls and loop over 16 lanes.

- [ ] **Step 2: Add Neg (accumulator) to execute_wide**

In the `execute_wide` match, the `Neg` arm currently calls the narrow
`vector_negate` via bridge. For accumulator operations (has AccumReg
source), it should use `Acc1024`:

```rust
            SemanticOp::Neg => {
                let has_acc_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::AccumReg(_)));
                if has_acc_source {
                    // Accumulator negate on cm-class register
                    Self::execute_acc_negate(op, ctx);
                } else {
                    let a = Self::get_wide_vec_source(op, ctx, 0);
                    let result = Self::wide_element_wise_unary(&a, et, Self::vector_negate);
                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }
```

Add `execute_acc_negate` that reads/writes `Acc1024` and handles the
config word (zero_acc, float vs int).

- [ ] **Step 3: Run all tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "refactor(emu): accumulator ops use Acc1024 in execute_wide path"
```

---

### Task 5: Port Cross-Half Vector Ops (VPUSH, VEXTBCST, Align)

**Files:**
- Modify: `src/interpreter/execute/vector.rs`

Move VPUSH, VEXTBCST, and Align (VSHIFT) from the fallback path to
dedicated wide implementations in `execute_wide()`.

- [ ] **Step 1: Add VPUSH to execute_wide**

VPUSH shifts a 512-bit vector and inserts a scalar. It operates on the
FULL 512 bits, not halves. Add a `VectorInsert` arm in `execute_wide`:

```rust
            SemanticOp::VectorInsert => {
                let is_push = op.encoding_name.as_ref().map_or(false, |n| {
                    n.to_lowercase().contains("vpush")
                });
                if is_push {
                    let src = Self::get_wide_vec_source(op, ctx, 0);
                    let value = Self::get_scalar_source(op, ctx);
                    let is_hi = op.encoding_name.as_ref().map_or(false, |n| {
                        n.to_lowercase().contains("_hi")
                    });
                    let result = Self::wide_vector_push(&src, value, is_hi, et);
                    Self::write_wide_vec_dest(op, ctx, result);
                } else {
                    // VINSERT: element insert, delegate to fallback
                    return Self::execute_wide_fallback(op, ctx, semantic, et);
                }
                true
            }
```

Implement `wide_vector_push` that operates on `Vec512` (64 bytes):

```rust
    fn wide_vector_push(src: &Vec512, value: u32, is_hi: bool, et: ElementType) -> Vec512 {
        let mut bytes = [0u8; 64];
        for (i, word) in src.iter().enumerate() {
            let b = word.to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }

        let elem_bytes = (et.bits() as usize / 8).max(1);

        if is_hi {
            // Shift towards low, insert at high end
            bytes.copy_within(elem_bytes.., 0);
            let val_bytes = value.to_le_bytes();
            let insert_pos = 64 - elem_bytes;
            for i in 0..elem_bytes.min(4) {
                bytes[insert_pos + i] = val_bytes[i];
            }
        } else {
            // Shift towards high, insert at low end
            bytes.copy_within(..64 - elem_bytes, elem_bytes);
            let val_bytes = value.to_le_bytes();
            for i in 0..elem_bytes.min(4) {
                bytes[i] = val_bytes[i];
            }
        }

        let mut result = [0u32; 16];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            result[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        result
    }
```

- [ ] **Step 2: Add VEXTBCST to execute_wide**

Extract element from full 512-bit source, then broadcast:

```rust
            SemanticOp::VectorBroadcast => {
                let has_vector_source = op.sources.iter()
                    .any(|s| matches!(s, Operand::VectorReg(_)));
                if has_vector_source {
                    // VEXTBCST: extract from 512-bit source, then broadcast
                    let src = Self::get_wide_vec_source(op, ctx, 0);
                    let index = Self::get_lane_index(op, ctx);
                    let value = Self::extract_wide_element(&src, index, et);
                    let narrow_result = Self::vector_broadcast(value, et);
                    // Broadcast to both halves
                    let mut result = [0u32; 16];
                    result[..8].copy_from_slice(&narrow_result);
                    result[8..].copy_from_slice(&narrow_result);
                    Self::write_wide_vec_dest(op, ctx, result);
                } else {
                    let value = Self::get_scalar_source(op, ctx);
                    let narrow_result = Self::vector_broadcast(value, et);
                    let mut result = [0u32; 16];
                    result[..8].copy_from_slice(&narrow_result);
                    result[8..].copy_from_slice(&narrow_result);
                    Self::write_wide_vec_dest(op, ctx, result);
                }
                true
            }
```

Add `extract_wide_element` that indexes into full 512 bits:

```rust
    fn extract_wide_element(src: &Vec512, index: u32, et: ElementType) -> u32 {
        let max_elems = 512 / et.bits() as u32;
        let idx = index % max_elems;
        let bit_offset = idx * et.bits() as u32;
        let word_idx = (bit_offset / 32) as usize;
        let bit_in_word = bit_offset % 32;
        let mask = (1u64 << et.bits()) - 1;
        ((src[word_idx] as u64 >> bit_in_word) & mask) as u32
    }
```

- [ ] **Step 3: Add Align (VSHIFT) to execute_wide**

The current Align implementation concatenates two 256-bit vectors.
For 512-bit mode, it should concatenate two 512-bit vectors (1024
bits total) and shift. Update the `Align` arm:

```rust
            SemanticOp::Align => {
                let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
                let shift = Self::get_lane_index(op, ctx);
                let result = Self::wide_vector_align(&a, &b, shift);
                Self::write_wide_vec_dest(op, ctx, result);
                true
            }
```

Implement `wide_vector_align` on `Vec512`:

```rust
    fn wide_vector_align(src1: &Vec512, src2: &Vec512, byte_shift: u32) -> Vec512 {
        // Concatenate: [src1 (low 64B) || src2 (high 64B)] = 128 bytes
        // Shift right by byte_shift bytes, take lower 64 bytes
        let shift = (byte_shift & 0x7F) as usize;
        let get_byte = |idx: usize| -> u8 {
            let word_array = if idx < 64 { src1 } else { src2 };
            let adj = idx % 64;
            let w = adj / 4;
            let b = adj % 4;
            ((word_array[w] >> (b * 8)) & 0xFF) as u8
        };

        let mut result = [0u32; 16];
        for i in 0..16 {
            let base = i * 4 + shift;
            let b0 = if base < 128 { get_byte(base) } else { 0 } as u32;
            let b1 = if base + 1 < 128 { get_byte(base + 1) } else { 0 } as u32;
            let b2 = if base + 2 < 128 { get_byte(base + 2) } else { 0 } as u32;
            let b3 = if base + 3 < 128 { get_byte(base + 3) } else { 0 } as u32;
            result[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        result
    }
```

- [ ] **Step 4: Write unit tests for cross-half operations**

Add tests that verify data crosses the 256-bit boundary correctly:

```rust
#[test]
fn test_wide_vector_push_lo_32() {
    // Push value 0xDEADBEEF into low end of a 512-bit vector.
    // All elements should shift up by 4 bytes, last element discarded.
    let mut src = [0u32; 16];
    for i in 0..16 { src[i] = (i as u32 + 1) * 100; }  // 100, 200, ..., 1600
    let result = VectorAlu::wide_vector_push(&src, 0xDEADBEEF, false, ElementType::Int32);
    assert_eq!(result[0], 0xDEADBEEF);  // Inserted value
    assert_eq!(result[1], 100);   // Was src[0]
    assert_eq!(result[8], 800);   // Was src[7] -- crossed the 256-bit boundary!
    assert_eq!(result[15], 1500); // Was src[14]
    // src[15] (1600) is discarded
}

#[test]
fn test_wide_vector_push_hi_16() {
    // Push a 16-bit value into high end of a 512-bit vector (32 x i16).
    let mut src = [0u32; 16];
    src[0] = 0x0002_0001;  // elements 0, 1
    src[15] = 0x0020_001F; // elements 30, 31
    let result = VectorAlu::wide_vector_push(&src, 0x00FF, true, ElementType::Int16);
    // Element 0 discarded (shifted out), elements shift down by one i16 position
    assert_eq!(result[15] >> 16, 0x00FF); // Inserted at highest i16 position
}

#[test]
fn test_extract_wide_element_crosses_boundary() {
    // Extract element 8 (first element in the high 256-bit half)
    let mut src = [0u32; 16];
    src[8] = 0x42424242;
    let val = VectorAlu::extract_wide_element(&src, 8, ElementType::Int32);
    assert_eq!(val, 0x42424242);
}

#[test]
fn test_wide_vector_align_crosses_boundary() {
    // Shift by 32 bytes = move src2 into lower half of result
    let src1 = [1u32; 16];
    let src2 = [2u32; 16];
    let result = VectorAlu::wide_vector_align(&src1, &src2, 64);
    // Shift by 64 bytes: src2 provides all 64 bytes
    assert_eq!(result, [2u32; 16]);
}
```

- [ ] **Step 5: Extract the fallback into a named function**

To keep `execute_wide` clean, extract the fallback pattern:

```rust
    fn execute_wide_fallback(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        semantic: SemanticOp,
        et: ElementType,
    ) -> bool {
        log::trace!(
            "[VECTOR_WIDE] fallback to half-split for {:?}",
            semantic
        );
        let handled = Self::execute_half(op, ctx, semantic, et);
        if handled {
            let mut hi_op = op.clone();
            Self::increment_vector_regs(&mut hi_op);
            Self::execute_half(&hi_op, ctx, semantic, et);
        }
        handled
    }
```

Update the `_` arm in `execute_wide` to call this.

- [ ] **Step 5: Run all tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "feat(emu): full-width VPUSH, VEXTBCST, VSHIFT in execute_wide"
```

---

### Task 6: Port Remaining Wide Ops and Run ISA Sweep

**Files:**
- Modify: `src/interpreter/execute/vector.rs`

Port any remaining SemanticOps that appear in the fallback trace log.
Then build and run the ISA test sweep to measure accuracy improvement.

- [ ] **Step 1: Add remaining element-wise ops to execute_wide**

Review the fallback trace output (from a quick test run) and add any
missing element-wise ops to `execute_wide` using the bridge pattern.
Common candidates: Shl, Srl, Sra, MaxLt, MinGe, SubLt, SubGe,
MaxDiffLt, NegAdd, NegMul, Shuffle, Pack, Unpack, Convert, etc.

For each, the pattern is identical:
```rust
SemanticOp::Foo => {
    let (a, b) = Self::get_two_wide_vec_sources(op, ctx);
    let result = Self::wide_element_wise_binary(&a, &b, et, Self::vector_foo);
    Self::write_wide_vec_dest(op, ctx, result);
    true
}
```

- [ ] **Step 2: Build**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Run ISA test sweep**

Run: `nice -n 19 bash scripts/isa-test.sh 2>&1 | tee /tmp/claude-1000/isa-test-wide.log | tail -10`

- [ ] **Step 4: Analyze results**

Run the per-instruction accuracy analysis and compare against the
baseline (45.5% / 21,177 words). Expected improvements in:
- VPUSH (was 0%)
- VEXTBCST (was 8-32%, should improve with full-width extract)
- VSHIFT (may improve)
- VADD_F/VSUB_F family (was 50-55%, may improve with proper Acc1024)

- [ ] **Step 5: Commit results**

```bash
git add src/interpreter/execute/vector.rs
git commit -m "feat(emu): port remaining wide ops, ISA accuracy improvement"
```

---

### Task 7: Update matmul_config_driven to Vec512/Acc1024

**Files:**
- Modify: `src/interpreter/execute/vector_matmul.rs:177-389`

Update the config-driven matmul to use `Vec512` and `Acc1024` directly
instead of split `(lo, hi)` pairs. This makes it ready to wire into
`execute_wide` once permutation tables are understood.

- [ ] **Step 1: Update matmul_config_driven signature**

Change from:
```rust
pub fn matmul_config_driven(
    acc_lo: &mut [u64; 8], acc_hi: &mut [u64; 8],
    a_lo: &[u32; 8], a_hi: &[u32; 8],
    b_lo: &[u32; 8], b_hi: &[u32; 8],
    config: &MatMulConfig,
)
```

To:
```rust
pub fn matmul_config_driven(
    acc: &mut Acc1024,
    a: &Vec512,
    b: &Vec512,
    config: &MatMulConfig,
)
```

Update `extract_element_512` to index into `&[u32; 16]` directly.
Update `read_acc_wide` / `write_acc_wide` to index into `&[u64; 16]`.

- [ ] **Step 2: Run matmul tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib vector_matmul 2>&1 | tail -10`
Expected: all matmul tests pass

- [ ] **Step 3: Commit**

```bash
git add src/interpreter/execute/vector_matmul.rs
git commit -m "refactor(emu): matmul_config_driven uses Vec512/Acc1024"
```
