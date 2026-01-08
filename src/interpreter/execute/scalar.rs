//! Scalar ALU execution unit (LEGACY FALLBACK).
//!
//! # Architecture Note
//!
//! This module is a **legacy fallback** for scalar operations. The preferred
//! execution path is through the TableGen-driven semantic dispatcher in
//! [`execute_semantic`](super::semantic::execute_semantic).
//!
//! ## Execution Flow
//!
//! ```text
//! CycleAccurateExecutor::execute_slot()
//!         |
//!         v
//!   execute_semantic(op, ctx)  <-- TableGen-driven, preferred
//!         |
//!         | returns false (no semantic info)
//!         v
//!   ScalarAlu::execute(op, ctx)  <-- Legacy fallback (this module)
//!         |
//!         v
//!   VectorAlu, MemoryUnit, etc.
//! ```
//!
//! ## When Each Path is Used
//!
//! - **Semantic dispatch**: When `op.semantic` is `Some(SemanticOp::*)` AND the
//!   semantic dispatcher has a handler for that operation. Currently covers
//!   ~33% of instructions via TableGen patterns + mnemonic inference.
//!
//! - **Legacy fallback**: When `op.semantic` is `None`.
//!
//! ## CPU Flag Behavior (AIE2 Accurate)
//!
//! AIE2 has only ONE hardware flag - the **Carry flag (C)** stored in SR bit 0.
//! Zero (Z), Negative (N), and Overflow (V) are computed on-demand by branch logic.
//!
//! **Operations that SET the Carry flag:**
//! - `ADD`, `SUB` - arithmetic carry/borrow
//! - `ADC`, `SBC` - add/sub with carry (also read C)
//! - `ABS` - carry set if input was negative
//!
//! **Operations that do NOT affect flags (preserve C):**
//! - `MUL`, `DIV`, `MOD` - no flag effects
//! - `AND`, `OR`, `XOR`, `NOT` - no flag effects
//! - `SHL`, `SHR`, `SRA` - no flag effects (unlike x86!)
//! - Extensions, moves, comparisons - no flag effects
//!
//! ## Future Direction
//!
//! As TableGen coverage expands, this module should shrink. New operations
//! should be added to `semantic.rs` with proper SemanticOp definitions, not here.
//!
//! # Operations (Legacy)
//!
//! - **Arithmetic**: add, sub (set C), mul (no flags)
//! - **Logic**: and, or, xor (no flags)
//! - **Shift**: shl, shr, sra (no flags)
//! - **Move**: mov, movi (no flags)
//! - **Compare**: cmp (sets C via sub semantics)
//! - **Carry ops**: adc, sbc (read and write C)

use crate::interpreter::bundle::{Operation, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::Flags;

/// Scalar ALU execution unit (legacy fallback).
///
/// **PREFER semantic dispatch** via `execute_semantic()` for new operations.
/// This unit exists for:
/// 1. Operations not yet in semantic dispatch (~67% of instructions)
/// 2. Operations that need to set CPU flags
///
/// See module docs for architecture details.
pub struct ScalarAlu;

impl ScalarAlu {
    /// Execute a scalar operation (legacy fallback).
    ///
    /// Returns `true` if the operation was handled, `false` if not a scalar op.
    ///
    /// **Note**: This is called AFTER `execute_semantic()`. If an operation has
    /// a SemanticOp and a semantic handler, it will be handled there first.
    /// This function only runs for operations that:
    /// 1. Don't have `op.semantic` set, OR
    /// 2. Need flag-setting behavior that semantic handlers don't provide
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        match &op.op {
            // ═══════════════════════════════════════════════════════════════
            // ARITHMETIC OPERATIONS (semantic-covered but we set flags)
            // These have SemanticOp equivalents (Add, Sub, Mul) but we also
            // set CPU flags which semantic handlers don't do.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarAdd => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = a.wrapping_add(b);
                Self::write_dest(op, ctx, result);
                ctx.set_flags(Flags::from_add(a, b, result));
                true
            }

            Operation::ScalarSub => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = a.wrapping_sub(b);
                Self::write_dest(op, ctx, result);
                ctx.set_flags(Flags::from_sub(a, b, result));
                true
            }

            Operation::ScalarMul => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = a.wrapping_mul(b);
                Self::write_dest(op, ctx, result);
                // AIE2: MUL does NOT set any flags
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // BITWISE OPERATIONS (no flag effects in AIE2)
            // These do NOT set any flags - they preserve the Carry flag.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarAnd => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a & b);
                // AIE2: AND does NOT set any flags
                true
            }

            Operation::ScalarOr => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a | b);
                // AIE2: OR does NOT set any flags
                true
            }

            Operation::ScalarXor => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a ^ b);
                // AIE2: XOR does NOT set any flags
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // SHIFT OPERATIONS (no flag effects in AIE2)
            // These do NOT set any flags - they preserve the Carry flag.
            // Unlike x86, AIE2 shifts don't affect the carry flag.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarShl => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let shift = b & 0x1F;
                Self::write_dest(op, ctx, a << shift);
                // AIE2: SHL does NOT set any flags
                true
            }

            Operation::ScalarShr => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let shift = b & 0x1F;
                Self::write_dest(op, ctx, a >> shift);
                // AIE2: SHR does NOT set any flags
                true
            }

            Operation::ScalarSra => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let shift = b & 0x1F;
                Self::write_dest(op, ctx, ((a as i32) >> shift) as u32);
                // AIE2: SRA does NOT set any flags
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // MOVE OPERATIONS (semantic-covered as Copy)
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarMov => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, src);
                true
            }

            Operation::ScalarMovi { value } => {
                Self::write_dest(op, ctx, *value as u32);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // COMPARE-ONLY (legacy-only: sets flags without writing result)
            // No semantic equivalent - this is a flag-setting-only operation.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarCmp => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = a.wrapping_sub(b);
                ctx.set_flags(Flags::from_sub(a, b, result));
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // COMPARISON OPERATIONS (semantic-covered as SetLt, SetGe, etc.)
            // These have SemanticOp equivalents but we keep for completeness.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarLt => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if (a as i32) < (b as i32) { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarLtu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if a < b { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarLe => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if (a as i32) <= (b as i32) { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarLeu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if a <= b { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarGt => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if (a as i32) > (b as i32) { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarGtu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if a > b { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarGe => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if (a as i32) >= (b as i32) { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarGeu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if a >= b { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarEq => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if a == b { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarNe => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if a != b { 1 } else { 0 };
                Self::write_dest(op, ctx, result);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // SELECT OPERATIONS (semantic-covered as Select)
            // sel.eqz and sel.nez test r27 implicitly - handled properly by
            // semantic dispatch when op.implicit_regs contains r27.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarSel => {
                // Generic sel: dst = cond ? src1 : src2
                // Three sources: cond, true_val, false_val
                let cond = Self::get_source(op, ctx, 0);
                let true_val = Self::get_source(op, ctx, 1);
                let false_val = Self::get_source(op, ctx, 2);
                let result = if cond != 0 { true_val } else { false_val };
                Self::write_dest(op, ctx, result);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // UNARY OPERATIONS (semantic-covered as Abs, Ctlz)
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarAbs => {
                let src = Self::get_source(op, ctx, 0);
                let result = (src as i32).wrapping_abs() as u32;
                Self::write_dest(op, ctx, result);
                // AIE2: ABS sets Carry flag (Defs=[srCarry] in TableGen)
                // Carry is set if the input was negative (negation occurred)
                let was_negative = (src as i32) < 0;
                let mut flags = ctx.flags();
                flags.c = was_negative;
                ctx.set_flags(flags);
                true
            }

            Operation::ScalarClz => {
                let src = Self::get_source(op, ctx, 0);
                let result = src.leading_zeros();
                Self::write_dest(op, ctx, result);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // COUNT LEADING BITS (legacy-only)
            // Different from CLZ - counts leading sign extension bits.
            // Not the same as Ctlz semantic operation.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarClb => {
                // Count leading bits: leading sign extension bits
                // For positive: leading zeros - 1 (sign bit doesn't count)
                // For negative: leading ones - 1
                let src = Self::get_source(op, ctx, 0);
                let result = if (src as i32) >= 0 {
                    src.leading_zeros().saturating_sub(1)
                } else {
                    src.leading_ones().saturating_sub(1)
                };
                Self::write_dest(op, ctx, result);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // CARRY OPERATIONS (legacy-only: use CPU carry flag)
            // These read/write the carry flag (C), making them stateful.
            // Cannot be implemented as pure semantic operations.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarAdc => {
                // Add with carry: dst = src1 + src2 + carry_flag
                let (a, b) = Self::get_two_sources(op, ctx);
                let carry_in = if ctx.flags().c { 1u32 } else { 0u32 };
                let (result1, carry1) = a.overflowing_add(b);
                let (result, carry2) = result1.overflowing_add(carry_in);
                Self::write_dest(op, ctx, result);
                // Set flags: carry if either addition overflowed
                let mut flags = Flags::from_result(result);
                flags.c = carry1 || carry2;
                // Overflow: signed overflow in the addition
                let a_sign = (a as i32) < 0;
                let b_sign = (b as i32) < 0;
                let r_sign = (result as i32) < 0;
                flags.v = (a_sign == b_sign) && (a_sign != r_sign);
                ctx.set_flags(flags);
                true
            }

            Operation::ScalarSbc => {
                // Subtract with borrow: dst = src1 - src2 - !carry_flag
                // ARM-style: C=1 means no borrow, C=0 means borrow
                let (a, b) = Self::get_two_sources(op, ctx);
                let borrow_in = if ctx.flags().c { 0u32 } else { 1u32 };
                let (result1, borrow1) = a.overflowing_sub(b);
                let (result, borrow2) = result1.overflowing_sub(borrow_in);
                Self::write_dest(op, ctx, result);
                // Set flags
                let mut flags = Flags::from_result(result);
                flags.c = !(borrow1 || borrow2); // C=1 means no borrow
                let a_sign = (a as i32) < 0;
                let b_sign = (b as i32) < 0;
                let r_sign = (result as i32) < 0;
                flags.v = (a_sign != b_sign) && (a_sign != r_sign);
                ctx.set_flags(flags);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // SIGN/ZERO EXTENSION (no flag effects in AIE2)
            // These do NOT set any flags - they preserve the Carry flag.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarExtendS8 => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, ((src as i8) as i32) as u32);
                // AIE2: Extensions do NOT set any flags
                true
            }

            Operation::ScalarExtendS16 => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, ((src as i16) as i32) as u32);
                // AIE2: Extensions do NOT set any flags
                true
            }

            Operation::ScalarExtendU8 => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, src & 0xFF);
                // AIE2: Extensions do NOT set any flags
                true
            }

            Operation::ScalarExtendU16 => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, src & 0xFFFF);
                // AIE2: Extensions do NOT set any flags
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // DIVISION OPERATIONS (no flag effects in AIE2)
            // Multi-cycle operations (6+ cycles in real hardware).
            // Division by zero returns saturated values, not trap.
            // These do NOT set any flags.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarDiv => {
                let (a, b) = Self::get_two_sources(op, ctx);
                // Signed division: a / b
                // Division by zero returns INT_MIN (saturated)
                let result = if b == 0 {
                    i32::MIN as u32
                } else {
                    ((a as i32).wrapping_div(b as i32)) as u32
                };
                Self::write_dest(op, ctx, result);
                // AIE2: DIV does NOT set any flags
                true
            }

            Operation::ScalarDivu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                // Unsigned division: a / b
                // Division by zero returns MAX (saturated)
                let result = if b == 0 { u32::MAX } else { a / b };
                Self::write_dest(op, ctx, result);
                // AIE2: DIVU does NOT set any flags
                true
            }

            Operation::ScalarMod => {
                let (a, b) = Self::get_two_sources(op, ctx);
                // Signed modulo: a % b
                // Mod by zero returns the dividend unchanged
                let result = if b == 0 {
                    a
                } else {
                    ((a as i32).wrapping_rem(b as i32)) as u32
                };
                Self::write_dest(op, ctx, result);
                // AIE2: MOD does NOT set any flags
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // CONDITIONAL SELECT WITH IMPLICIT R27 (semantic-covered as Select)
            // These test r27 implicitly - the register is not encoded in bits.
            // Semantic dispatch handles this via op.implicit_regs containing r27.
            // Legacy handlers hardcode r27 as fallback.
            // ═══════════════════════════════════════════════════════════════
            Operation::ScalarSelEqz => {
                // Select if equal zero: dest = (r27 == 0) ? src_true : src_false
                // AIE2 sel.eqz ALWAYS tests r27 (hardcoded in instruction definition)
                // Assembly format: sel.eqz dest, true_val, false_val, r27
                // Only 2 actual operand fields: mRx0 (true), mRy (false)
                // TODO: This should be driven by TableGen InstrDef, not hardcoded
                let src_true = Self::get_source(op, ctx, 0);
                let src_false = Self::get_source(op, ctx, 1);
                let test = ctx.scalar_read(27); // r27 is implicit test condition
                log::debug!("[SEL.EQZ] sources={:?} r27={} src_true={} src_false={}",
                           op.sources, test, src_true, src_false);
                let result = if test == 0 { src_true } else { src_false };
                log::debug!("[SEL.EQZ] result={} (r27==0? {})", result, test == 0);
                Self::write_dest(op, ctx, result);
                true
            }

            Operation::ScalarSelNez => {
                // Select if not equal zero: dest = (r27 != 0) ? src_true : src_false
                // AIE2 sel.nez ALWAYS tests r27 (hardcoded in instruction definition)
                // Assembly format: sel.nez dest, true_val, false_val, r27
                // Only 2 actual operand fields: mRx0 (true), mRy (false)
                // TODO: This should be driven by TableGen InstrDef, not hardcoded
                let src_true = Self::get_source(op, ctx, 0);
                let src_false = Self::get_source(op, ctx, 1);
                let test = ctx.scalar_read(27); // r27 is implicit test condition
                let result = if test != 0 { src_true } else { src_false };
                Self::write_dest(op, ctx, result);
                true
            }

            _ => false, // Not a scalar operation
        }
    }

    /// Get two source operands as u32.
    fn get_two_sources(op: &SlotOp, ctx: &ExecutionContext) -> (u32, u32) {
        let a = Self::get_source(op, ctx, 0);
        let b = Self::get_source(op, ctx, 1);
        (a, b)
    }

    /// Get a single source operand.
    fn get_source(op: &SlotOp, ctx: &ExecutionContext, idx: usize) -> u32 {
        op.sources.get(idx).map_or(0, |src| Self::read_operand(src, ctx))
    }

    /// Read an operand value.
    ///
    /// Uses `ctx.scalar_read()` for VLIW-safe reads that respect the
    /// bundle snapshot when inside a VLIW bundle.
    fn read_operand(operand: &Operand, ctx: &ExecutionContext) -> u32 {
        match operand {
            Operand::ScalarReg(r) => ctx.scalar_read(*r),
            Operand::PointerReg(r) => ctx.pointer.read(*r),
            Operand::ModifierReg(r) => ctx.modifier.read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0, // Other operand types not valid for scalar ALU
        }
    }

    /// Write result to destination operand.
    fn write_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
        if let Some(dest) = &op.dest {
            match dest {
                Operand::ScalarReg(r) => ctx.scalar.write(*r, value),
                Operand::PointerReg(r) => ctx.pointer.write(*r, value),
                Operand::ModifierReg(r) => ctx.modifier.write(*r, value),
                _ => {} // Ignore invalid destinations
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_add_op(dest: u8, src1: u8, src2: u8) -> SlotOp {
        SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(dest))
            .with_source(Operand::ScalarReg(src1))
            .with_source(Operand::ScalarReg(src2))
    }

    #[test]
    fn test_scalar_add() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        let op = make_add_op(2, 0, 1);
        assert!(ScalarAlu::execute(&op, &mut ctx));
        assert_eq!(ctx.scalar.read(2), 30);
    }

    #[test]
    fn test_scalar_add_overflow() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 0xFFFF_FFFF);
        ctx.scalar.write(1, 1);

        let op = make_add_op(2, 0, 1);
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0);
        assert!(ctx.flags().c); // Carry set
        assert!(ctx.flags().z); // Zero set
    }

    #[test]
    fn test_scalar_sub() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 30);
        ctx.scalar.write(1, 10);

        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarSub)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));

        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 20);
    }

    #[test]
    fn test_scalar_mul() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 6);
        ctx.scalar.write(1, 7);

        let op = SlotOp::new(SlotIndex::Scalar1, Operation::ScalarMul)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));

        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 42);
    }

    #[test]
    fn test_scalar_logic() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 0xFF00);
        ctx.scalar.write(1, 0x0FF0);

        // AND
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAnd)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0x0F00);

        // OR
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarOr)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0xFFF0);

        // XOR
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarXor)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0xF0F0);
    }

    #[test]
    fn test_scalar_shifts() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 0x0000_00FF);
        ctx.scalar.write(1, 4);

        // SHL
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarShl)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0x0000_0FF0);

        // SHR
        ctx.scalar.write(0, 0x0000_0FF0);
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarShr)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0x0000_00FF);

        // SRA with negative number
        ctx.scalar.write(0, 0x8000_0000u32); // -2147483648
        ctx.scalar.write(1, 4);
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarSra)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0xF800_0000u32); // Sign-extended
    }

    #[test]
    fn test_scalar_mov() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 42);

        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(0));

        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(1), 42);
    }

    #[test]
    fn test_scalar_movi() {
        let mut ctx = make_ctx();

        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMovi { value: 12345 })
            .with_dest(Operand::ScalarReg(5));

        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(5), 12345);
    }

    #[test]
    fn test_scalar_cmp() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 10);

        // Equal
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarCmp)
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));

        ScalarAlu::execute(&op, &mut ctx);
        assert!(ctx.flags().z); // Zero flag set (equal)

        // Less than
        ctx.scalar.write(1, 20);
        ScalarAlu::execute(&op, &mut ctx);
        assert!(!ctx.flags().z);
        assert!(ctx.flags().n); // Negative flag (10 - 20 < 0)
    }

    #[test]
    fn test_immediate_operand() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 10);

        // r1 = r0 + 5
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::Immediate(5));

        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(1), 15);
    }

    #[test]
    fn test_pointer_reg_operand() {
        let mut ctx = make_ctx();
        ctx.pointer.write(0, 0x1000);
        ctx.scalar.write(0, 0x100);

        // r1 = p0 + r0
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::PointerReg(0))
            .with_source(Operand::ScalarReg(0));

        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(1), 0x1100);
    }
}
