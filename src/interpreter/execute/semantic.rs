//! TableGen-driven semantic execution.
//!
//! This module provides execution handlers based on SemanticOp from TableGen,
//! replacing the 133+ manual Operation handlers with ~40 semantic handlers.
//!
//! # Architecture
//!
//! ```text
//! SlotOp { semantic: Some(SemanticOp::Add), sources, dest, implicit_regs }
//!                          |
//!                          v
//!                  execute_semantic()
//!                          |
//!                          v
//!                    execute_add()
//!                          |
//!                          v
//!                  read sources[0], sources[1]
//!                  compute result = a + b
//!                  write to dest
//! ```
//!
//! # Operand Conventions
//!
//! All handlers assume operands are in **canonical order** per TableGen InstrDef.inputs:
//! - sources[0] = first input operand
//! - sources[1] = second input operand (if any)
//! - Implicit registers accessed via `op.implicit_regs` or `op.get_implicit_use()`
//!
//! # CPU Flag Behavior (AIE2 Accurate)
//!
//! AIE2 has a unique flag architecture:
//!
//! | Flag | Hardware? | Storage | Behavior |
//! |------|-----------|---------|----------|
//! | **Carry (C)** | YES | SR bit 0 | Set by ADD, SUB, ADC, SBC, ABS, NEG |
//! | Zero (Z) | NO | Computed | Branch logic tests register == 0 |
//! | Negative (N) | NO | Computed | Branch logic tests register < 0 |
//! | Overflow (V) | NO | Computed | Computed on demand |
//!
//! ## Which Operations Set Flags
//!
//! **Set Carry (from TableGen Defs=[srCarry]):**
//! - `ADD`, `SUB` - arithmetic carry/borrow
//! - `ADC`, `SBC` - add/sub with carry (also Uses=[srCarry])
//! - `ABS` - carry set if input was negative
//! - `NEG` - equivalent to SUB(0, x)
//!
//! **Do NOT set flags (preserve Carry):**
//! - `MUL`, `DIV`, `MOD` - no flag effects
//! - `AND`, `OR`, `XOR`, `NOT` - no flag effects
//! - `SHL`, `SHR`, `SRA` - no flag effects (unlike x86!)
//! - Comparisons (`LT`, `GE`, etc.) - produce 0/1 result, no flags
//! - Moves, loads, stores - no flag effects
//!
//! ## Branch Conditions
//!
//! - `jz`/`jnz` - test **register value** directly (most common)
//! - `jc`/`jnc` - test Carry flag (for ADC/SBC chains)
//! - Other conditions computed from register values, not stored flags

use crate::interpreter::bundle::{Operand, Operation, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::Flags;
use crate::tablegen::SemanticOp;

/// Execute a SlotOp using its SemanticOp if available.
///
/// Returns `true` if execution was handled, `false` if the caller should
/// fall back to Operation-based dispatch.
pub fn execute_semantic(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let Some(semantic) = op.semantic else {
        return false; // No semantic info - fall back to Operation dispatch
    };

    // Vector operations have VectorReg destinations - let VectorAlu handle them
    if let Some(Operand::VectorReg(_)) = op.dest {
        return false; // Fall back to VectorAlu
    }

    // Also check for explicitly vector Operations
    if matches!(
        &op.op,
        Operation::VectorAdd { .. }
            | Operation::VectorSub { .. }
            | Operation::VectorMul { .. }
            | Operation::VectorAnd { .. }
            | Operation::VectorOr { .. }
            | Operation::VectorXor { .. }
    ) {
        return false; // Fall back to VectorAlu
    }

    log::trace!("[SEMANTIC] Handling {:?} via semantic {:?}", op.op, semantic);

    match semantic {
        // Arithmetic operations
        SemanticOp::Add => execute_add(op, ctx),
        SemanticOp::Sub => execute_sub(op, ctx),
        SemanticOp::Mul => execute_mul(op, ctx),
        SemanticOp::SDiv => execute_div(op, ctx, true),  // signed
        SemanticOp::UDiv => execute_div(op, ctx, false), // unsigned
        SemanticOp::SRem => execute_rem(op, ctx, true),  // signed
        SemanticOp::URem => execute_rem(op, ctx, false), // unsigned
        SemanticOp::Abs => execute_abs(op, ctx),
        SemanticOp::Neg => execute_neg(op, ctx),

        // Bitwise operations
        SemanticOp::And => execute_and(op, ctx),
        SemanticOp::Or => execute_or(op, ctx),
        SemanticOp::Xor => execute_xor(op, ctx),
        SemanticOp::Not => execute_not(op, ctx),
        SemanticOp::Shl => execute_shl(op, ctx),
        SemanticOp::Sra => execute_sra(op, ctx), // arithmetic right shift
        SemanticOp::Srl => execute_srl(op, ctx), // logical right shift

        // Comparison operations (produce 0 or 1)
        SemanticOp::SetLt => execute_setcc(op, ctx, CmpOp::Lt, true),
        SemanticOp::SetLe => execute_setcc(op, ctx, CmpOp::Le, true),
        SemanticOp::SetGt => execute_setcc(op, ctx, CmpOp::Gt, true),
        SemanticOp::SetGe => execute_setcc(op, ctx, CmpOp::Ge, true),
        SemanticOp::SetEq => execute_setcc(op, ctx, CmpOp::Eq, true),
        SemanticOp::SetNe => execute_setcc(op, ctx, CmpOp::Ne, true),
        SemanticOp::SetUlt => execute_setcc(op, ctx, CmpOp::Lt, false),
        SemanticOp::SetUle => execute_setcc(op, ctx, CmpOp::Le, false),
        SemanticOp::SetUgt => execute_setcc(op, ctx, CmpOp::Gt, false),
        SemanticOp::SetUge => execute_setcc(op, ctx, CmpOp::Ge, false),

        // Conditional select
        SemanticOp::Select => execute_select(op, ctx),

        // Memory operations
        SemanticOp::Load => execute_load(op, ctx),
        SemanticOp::Store => execute_store(op, ctx),

        // Control flow
        SemanticOp::Br => execute_branch(op, ctx),
        SemanticOp::BrCond => execute_branch_cond(op, ctx),

        // Bit manipulation
        SemanticOp::Ctlz => execute_clz(op, ctx),
        SemanticOp::Cttz => execute_ctz(op, ctx),
        SemanticOp::Ctpop => execute_popcount(op, ctx),

        // Move/copy
        SemanticOp::Copy => execute_mov(op, ctx),

        // Sign/zero extension
        SemanticOp::SignExtend => execute_sign_extend(op, ctx),
        SemanticOp::ZeroExtend => execute_zero_extend(op, ctx),
        SemanticOp::Truncate => execute_truncate(op, ctx),

        // Nop
        SemanticOp::Nop => true, // Nothing to do

        // Unknown/unsupported - fall back to Operation-based dispatch
        _ => {
            log::trace!("[SEMANTIC] Unhandled semantic op: {:?}", semantic);
            false
        }
    }
}

// ============================================================================
// Helper types and functions
// ============================================================================

/// Comparison operation type.
#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Read a source operand as a 32-bit value.
fn read_source(op: &SlotOp, ctx: &ExecutionContext, index: usize) -> u32 {
    op.sources.get(index).map_or(0, |src| read_operand(src, ctx))
}

/// Read an operand value from context.
///
/// Uses VLIW-safe reads where available (scalar_read for scalar regs).
fn read_operand(operand: &Operand, ctx: &ExecutionContext) -> u32 {
    match operand {
        Operand::ScalarReg(r) => ctx.scalar_read(*r), // VLIW-safe read
        Operand::PointerReg(r) => ctx.pointer.read(*r),
        Operand::ModifierReg(r) => ctx.modifier.read(*r),
        Operand::Immediate(v) => *v as u32,
        Operand::VectorReg(_) | Operand::AccumReg(_) => {
            // Vector/accum need special handling - return 0 for scalar context
            0
        }
        _ => 0,
    }
}

/// Write a result to the destination operand.
fn write_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
    if let Some(dest) = &op.dest {
        write_operand(dest, ctx, value);
    }
}

/// Write a value to an operand.
fn write_operand(operand: &Operand, ctx: &mut ExecutionContext, value: u32) {
    match operand {
        Operand::ScalarReg(r) => ctx.scalar.write(*r, value),
        Operand::PointerReg(r) => ctx.pointer.write(*r, value),
        Operand::ModifierReg(r) => ctx.modifier.write(*r, value),
        _ => {
            log::warn!("[SEMANTIC] Cannot write to operand: {:?}", operand);
        }
    }
}

/// Get an implicit register use value.
fn get_implicit_use(op: &SlotOp, ctx: &ExecutionContext, reg_num: u8) -> Option<u32> {
    // Check if this register is in implicit_regs as a use
    if op.implicit_regs.iter().any(|ir| ir.reg_num == reg_num && ir.is_use) {
        Some(ctx.scalar_read(reg_num))
    } else {
        None
    }
}

// ============================================================================
// Arithmetic handlers
// ============================================================================

fn execute_add(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_add(b);
    log::debug!(
        "[SEMANTIC ADD] dest={:?} src0={:?}({}) + src1={:?}({}) = {}",
        op.dest, op.sources.get(0), a, op.sources.get(1), b, result
    );
    write_dest(op, ctx, result);
    // AIE2: ADD sets the Carry flag (C). Z/N/V computed by branch logic.
    ctx.set_flags(Flags::from_add(a, b, result));
    true
}

fn execute_sub(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_sub(b);
    write_dest(op, ctx, result);
    // AIE2: SUB sets the Carry flag (C). Z/N/V computed by branch logic.
    ctx.set_flags(Flags::from_sub(a, b, result));
    true
}

fn execute_mul(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = a.wrapping_mul(b);
    write_dest(op, ctx, result);
    // AIE2: MUL does NOT set any flags (no Defs=[srCarry] in TableGen)
    true
}

fn execute_div(op: &SlotOp, ctx: &mut ExecutionContext, signed: bool) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = if b == 0 {
        // Division by zero: return saturated value
        if signed { i32::MIN as u32 } else { u32::MAX }
    } else if signed {
        ((a as i32).wrapping_div(b as i32)) as u32
    } else {
        a / b
    };
    write_dest(op, ctx, result);
    // AIE2: DIV does NOT set any flags
    true
}

fn execute_rem(op: &SlotOp, ctx: &mut ExecutionContext, signed: bool) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let result = if b == 0 {
        a // Modulo by zero returns dividend unchanged
    } else if signed {
        ((a as i32).wrapping_rem(b as i32)) as u32
    } else {
        a % b
    };
    write_dest(op, ctx, result);
    // AIE2: REM/MOD does NOT set any flags
    true
}

// ============================================================================
// Bitwise handlers
// ============================================================================

fn execute_and(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    write_dest(op, ctx, a & b);
    // AIE2: AND does NOT set any flags (preserves C)
    true
}

fn execute_or(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    write_dest(op, ctx, a | b);
    // AIE2: OR does NOT set any flags (preserves C)
    true
}

fn execute_xor(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    write_dest(op, ctx, a ^ b);
    // AIE2: XOR does NOT set any flags (preserves C)
    true
}

fn execute_not(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, !a);
    // AIE2: NOT does NOT set any flags (preserves C)
    true
}

fn execute_shl(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let shift = b & 0x1F; // Mask to 5 bits for 32-bit shift
    write_dest(op, ctx, a << shift);
    // AIE2: Shifts do NOT set any flags (preserves C)
    true
}

fn execute_sra(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0) as i32;
    let b = read_source(op, ctx, 1);
    let shift = b & 0x1F;
    write_dest(op, ctx, (a >> shift) as u32);
    // AIE2: Shifts do NOT set any flags (preserves C)
    true
}

fn execute_srl(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);
    let shift = b & 0x1F;
    write_dest(op, ctx, a >> shift);
    // AIE2: Shifts do NOT set any flags (preserves C)
    true
}

// ============================================================================
// Comparison handlers
// ============================================================================

fn execute_setcc(op: &SlotOp, ctx: &mut ExecutionContext, cmp: CmpOp, signed: bool) -> bool {
    let a = read_source(op, ctx, 0);
    let b = read_source(op, ctx, 1);

    let result = if signed {
        let a = a as i32;
        let b = b as i32;
        match cmp {
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
        }
    } else {
        match cmp {
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
        }
    };

    write_dest(op, ctx, if result { 1 } else { 0 });
    true
}

// ============================================================================
// Conditional select
// ============================================================================

/// Execute conditional select: dest = (test == 0) ? true_val : false_val
///
/// For `sel.eqz`, r27 is implicit (from TableGen eR27 register class).
/// For generic select, the test value is the 3rd source operand.
fn execute_select(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let src_true = read_source(op, ctx, 0);
    let src_false = read_source(op, ctx, 1);

    // Get test value: first check for implicit r27, then fall back to source[2]
    let test = get_implicit_use(op, ctx, 27).unwrap_or_else(|| read_source(op, ctx, 2));

    log::debug!(
        "[SEMANTIC SELECT] sources={:?} implicit_regs={:?} test={} true={} false={}",
        op.sources, op.implicit_regs, test, src_true, src_false
    );

    // sel.eqz semantics: if test == 0, select true_val
    let result = if test == 0 { src_true } else { src_false };

    log::debug!("[SEMANTIC SELECT] result={} (test==0? {})", result, test == 0);

    write_dest(op, ctx, result);
    true
}

// ============================================================================
// Memory operations
// ============================================================================

fn execute_load(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    // Memory operations need the full memory subsystem - fall back for now
    log::trace!("[SEMANTIC] Load requires memory subsystem - falling back");
    false
}

fn execute_store(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    log::trace!("[SEMANTIC] Store requires memory subsystem - falling back");
    false
}

// ============================================================================
// Control flow
// ============================================================================

fn execute_branch(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    log::trace!("[SEMANTIC] Branch requires control flow handling - falling back");
    false
}

fn execute_branch_cond(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
    log::trace!("[SEMANTIC] BranchCond requires control flow handling - falling back");
    false
}


// ============================================================================
// Special operations
// ============================================================================

fn execute_mov(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let value = read_source(op, ctx, 0);
    log::debug!(
        "[SEMANTIC MOV] dest={:?} src={:?} value={}",
        op.dest, op.sources.get(0), value
    );
    write_dest(op, ctx, value);
    true
}

fn execute_abs(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let result = (a as i32).wrapping_abs() as u32;
    write_dest(op, ctx, result);
    // AIE2: ABS sets the Carry flag (Defs=[srCarry] in TableGen)
    // Carry is set if the input was negative (negation occurred)
    let was_negative = (a as i32) < 0;
    let mut flags = ctx.flags();
    flags.c = was_negative;
    ctx.set_flags(flags);
    true
}

fn execute_neg(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    let result = (a as i32).wrapping_neg() as u32;
    write_dest(op, ctx, result);
    // AIE2: NEG is equivalent to SUB(0, a), sets Carry flag
    // Using SUB semantics: C = (0 >= a) in unsigned, i.e., C = (a == 0)
    ctx.set_flags(Flags::from_sub(0, a, result));
    true
}

fn execute_clz(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.leading_zeros());
    true
}

fn execute_ctz(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.trailing_zeros());
    true
}

fn execute_popcount(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    write_dest(op, ctx, a.count_ones());
    true
}

fn execute_sign_extend(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    // Sign extension width is typically encoded in the instruction
    // For now, assume 8-bit or 16-bit based on mnemonic (handled by caller)
    let a = read_source(op, ctx, 0);
    // Default to 16-bit sign extension
    let extended = ((a as i16) as i32) as u32;
    write_dest(op, ctx, extended);
    true
}

fn execute_zero_extend(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    // Default to 16-bit zero extension
    let extended = a & 0xFFFF;
    write_dest(op, ctx, extended);
    true
}

fn execute_truncate(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
    let a = read_source(op, ctx, 0);
    // Truncation just keeps lower bits - identity for 32-bit
    write_dest(op, ctx, a);
    true
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use crate::interpreter::bundle::Operation;
    use crate::tablegen::ImplicitReg;
    use smallvec::smallvec;

    fn make_test_context() -> ExecutionContext {
        ExecutionContext::new()
    }

    #[test]
    fn test_execute_add() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 10);
        ctx.scalar.write(2, 20);

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarAdd, SemanticOp::Add);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 30);
    }

    #[test]
    fn test_execute_sub() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 50);
        ctx.scalar.write(2, 20);

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSub, SemanticOp::Sub);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 30);
    }

    #[test]
    fn test_execute_select_with_implicit_r27() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 0);  // test value (r27) = 0, so select true

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSelEqz, SemanticOp::Select);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 100); // r27==0, so true value selected
    }

    #[test]
    fn test_execute_select_with_implicit_r27_nonzero() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 100); // true value
        ctx.scalar.write(2, 200); // false value
        ctx.scalar.write(27, 1);  // test value (r27) != 0, so select false

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarSelEqz, SemanticOp::Select);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));
        op.implicit_regs = smallvec![ImplicitReg {
            reg_class: "eR27".to_string(),
            reg_num: 27,
            is_use: true,
        }];

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 200); // r27!=0, so false value selected
    }

    #[test]
    fn test_execute_setcc_lt_signed() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, (-5i32) as u32); // -5
        ctx.scalar.write(2, 3);               // 3

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarLt, SemanticOp::SetLt);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 1); // -5 < 3 is true
    }

    #[test]
    fn test_execute_shift_left() {
        let mut ctx = make_test_context();
        ctx.scalar.write(1, 0b0001);
        ctx.scalar.write(2, 4);

        let mut op = SlotOp::with_semantic(SlotIndex::Scalar0, Operation::ScalarShl, SemanticOp::Shl);
        op.sources = smallvec![Operand::ScalarReg(1), Operand::ScalarReg(2)];
        op.dest = Some(Operand::ScalarReg(3));

        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar_read(3), 0b10000); // 1 << 4 = 16
    }

    #[test]
    fn test_no_semantic_returns_false() {
        let ctx = make_test_context();
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd);
        // No semantic set, should return false
        assert!(!execute_semantic(&op, &mut ctx.clone()));
    }
}
