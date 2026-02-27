//! Scalar ALU execution unit.
//!
//! # Two Roles
//!
//! This module serves two distinct purposes:
//!
//! 1. **Scalar-only operations** that have no SemanticOp equivalent and are
//!    handled exclusively here: `Movi`, `Cmp`, `Clb`, `Adc`, `Sbc`.
//!
//! 2. **Defensive fallback** for operations that semantic dispatch also covers.
//!    When `op.semantic` is set (the normal case for decoded instructions),
//!    `execute_semantic()` handles them and this code is never reached.
//!    The fallback exists for safety in case an instruction arrives without
//!    semantic info.
//!
//! ## Execution Flow
//!
//! ```text
//! CycleAccurateExecutor::execute_slot()
//!         |
//!         v
//!   execute_semantic(op, ctx)  <-- handles ops with SemanticOp
//!         |
//!         | returns false (no semantic, or delegated)
//!         v
//!   ScalarAlu::execute(op, ctx)  <-- scalar-only + defensive fallback
//!         |
//!         v
//!   VectorAlu, MemoryUnit, etc.
//! ```
//!
//! ## CPU Flag Behavior (AIE2)
//!
//! AIE2 has only ONE hardware flag -- the **Carry flag (C)** in SR bit 0.
//! Zero (Z), Negative (N), and Overflow (V) are computed on-demand by branch logic.
//!
//! **Operations that SET the Carry flag:**
//! - `ADD`, `SUB` -- arithmetic carry/borrow
//! - `ADC`, `SBC` -- add/sub with carry (also read C)
//! - `ABS` -- carry set if input was negative
//!
//! **Operations that do NOT affect flags (preserve C):**
//! - `MUL`, `DIV`, `MOD` -- no flag effects
//! - `AND`, `OR`, `XOR`, `NOT` -- no flag effects
//! - `SHL`, `SHR`, `SRA` -- no flag effects (unlike x86!)
//! - Extensions, moves, comparisons -- no flag effects
//!
//! ## Scalar-Only Operations
//!
//! - **Movi**: immediate-value move (Operation-level encoding, no SemanticOp)
//! - **Cmp**: flag-setting-only subtraction (no destination write)
//! - **Clb**: count leading sign-extension bits (different from Ctlz)
//! - **Adc/Sbc**: add/subtract with carry (read and write the carry flag)

use crate::interpreter::bundle::{Operation, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::Flags;

/// Scalar ALU execution unit.
///
/// Handles scalar-only operations (Movi, Cmp, Clb, Adc, Sbc) and provides
/// a defensive fallback for operations that semantic dispatch also covers.
/// See module docs for the full dispatch architecture.
pub struct ScalarAlu;

impl ScalarAlu {
    /// Execute a scalar operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a scalar op.
    ///
    /// Called AFTER `execute_semantic()`. For operations that have a SemanticOp,
    /// semantic dispatch handles them first. This function handles:
    /// 1. Scalar-only operations (no SemanticOp equivalent)
    /// 2. Defensive fallback if semantic dispatch did not handle the op
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext) -> bool {
        match &op.op {
            // ═══════════════════════════════════════════════════════════════
            // SCALAR-ONLY OPERATIONS
            // These have no SemanticOp equivalent. Only handled here.
            // ═══════════════════════════════════════════════════════════════

            // Immediate-value move: the value is encoded in the Operation
            // variant, not as a source operand. No SemanticOp needed.
            Operation::ScalarMovi { value } => {
                Self::write_dest(op, ctx, *value as u32);
                true
            }

            // Flag-setting-only subtraction. No destination register write.
            // Different from Sub (which writes a result). No SemanticOp
            // because there's no dest to set -- it only updates flags.
            Operation::ScalarCmp => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = a.wrapping_sub(b);
                ctx.set_flags(Flags::from_sub(a, b, result));
                true
            }

            // Count leading sign-extension bits. Different from Ctlz:
            // - CLZ counts leading zeros
            // - CLB counts leading bits that match the sign bit, minus 1
            Operation::ScalarClb => {
                let src = Self::get_source(op, ctx, 0);
                let result = if (src as i32) >= 0 {
                    src.leading_zeros().saturating_sub(1)
                } else {
                    src.leading_ones().saturating_sub(1)
                };
                Self::write_dest(op, ctx, result);
                true
            }

            // Add with carry: dst = src1 + src2 + carry_flag.
            // Reads and writes the carry flag, making it inherently stateful.
            // Cannot be a pure SemanticOp.
            Operation::ScalarAdc => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let carry_in = if ctx.flags().c { 1u32 } else { 0u32 };
                let (result1, carry1) = a.overflowing_add(b);
                let (result, carry2) = result1.overflowing_add(carry_in);
                Self::write_dest(op, ctx, result);
                let mut flags = Flags::from_result(result);
                flags.c = carry1 || carry2;
                let a_sign = (a as i32) < 0;
                let b_sign = (b as i32) < 0;
                let r_sign = (result as i32) < 0;
                flags.v = (a_sign == b_sign) && (a_sign != r_sign);
                ctx.set_flags(flags);
                true
            }

            // Subtract with borrow: dst = src1 - src2 - !carry_flag.
            // ARM-style: C=1 means no borrow, C=0 means borrow.
            Operation::ScalarSbc => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let borrow_in = if ctx.flags().c { 0u32 } else { 1u32 };
                let (result1, borrow1) = a.overflowing_sub(b);
                let (result, borrow2) = result1.overflowing_sub(borrow_in);
                Self::write_dest(op, ctx, result);
                let mut flags = Flags::from_result(result);
                flags.c = !(borrow1 || borrow2);
                let a_sign = (a as i32) < 0;
                let b_sign = (b as i32) < 0;
                let r_sign = (result as i32) < 0;
                flags.v = (a_sign != b_sign) && (a_sign != r_sign);
                ctx.set_flags(flags);
                true
            }

            // ═══════════════════════════════════════════════════════════════
            // DEFENSIVE FALLBACK
            // All operations below have SemanticOp equivalents in semantic.rs.
            // When op.semantic is set (normal for decoded instructions),
            // execute_semantic() handles them and this code is not reached.
            // These exist as a safety net.
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
                Self::write_dest(op, ctx, a.wrapping_mul(b));
                true
            }
            Operation::ScalarAnd => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a & b);
                true
            }
            Operation::ScalarOr => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a | b);
                true
            }
            Operation::ScalarXor => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a ^ b);
                true
            }
            Operation::ScalarShl => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a << (b & 0x1F));
                true
            }
            Operation::ScalarShr => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, a >> (b & 0x1F));
                true
            }
            Operation::ScalarSra => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, ((a as i32) >> (b & 0x1F)) as u32);
                true
            }
            Operation::ScalarMov => {
                Self::write_dest(op, ctx, Self::get_source(op, ctx, 0));
                true
            }
            // Comparisons (produce 0 or 1)
            Operation::ScalarLt => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if (a as i32) < (b as i32) { 1 } else { 0 });
                true
            }
            Operation::ScalarLtu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if a < b { 1 } else { 0 });
                true
            }
            Operation::ScalarLe => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if (a as i32) <= (b as i32) { 1 } else { 0 });
                true
            }
            Operation::ScalarLeu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if a <= b { 1 } else { 0 });
                true
            }
            Operation::ScalarGt => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if (a as i32) > (b as i32) { 1 } else { 0 });
                true
            }
            Operation::ScalarGtu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if a > b { 1 } else { 0 });
                true
            }
            Operation::ScalarGe => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if (a as i32) >= (b as i32) { 1 } else { 0 });
                true
            }
            Operation::ScalarGeu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if a >= b { 1 } else { 0 });
                true
            }
            Operation::ScalarEq => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if a == b { 1 } else { 0 });
                true
            }
            Operation::ScalarNe => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if a != b { 1 } else { 0 });
                true
            }
            // Select operations (r27-implicit test)
            Operation::ScalarSel => {
                let cond = Self::get_source(op, ctx, 0);
                let true_val = Self::get_source(op, ctx, 1);
                let false_val = Self::get_source(op, ctx, 2);
                Self::write_dest(op, ctx, if cond != 0 { true_val } else { false_val });
                true
            }
            Operation::ScalarSelEqz => {
                let src_true = Self::get_source(op, ctx, 0);
                let src_false = Self::get_source(op, ctx, 1);
                let test = ctx.scalar_read(27);
                Self::write_dest(op, ctx, if test == 0 { src_true } else { src_false });
                true
            }
            Operation::ScalarSelNez => {
                let src_true = Self::get_source(op, ctx, 0);
                let src_false = Self::get_source(op, ctx, 1);
                let test = ctx.scalar_read(27);
                Self::write_dest(op, ctx, if test != 0 { src_true } else { src_false });
                true
            }
            // Unary operations
            Operation::ScalarAbs => {
                let src = Self::get_source(op, ctx, 0);
                let result = (src as i32).wrapping_abs() as u32;
                Self::write_dest(op, ctx, result);
                let was_negative = (src as i32) < 0;
                let mut flags = ctx.flags();
                flags.c = was_negative;
                ctx.set_flags(flags);
                true
            }
            Operation::ScalarClz => {
                Self::write_dest(op, ctx, Self::get_source(op, ctx, 0).leading_zeros());
                true
            }
            // Extensions
            Operation::ScalarExtendS8 => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, ((src as i8) as i32) as u32);
                true
            }
            Operation::ScalarExtendS16 => {
                let src = Self::get_source(op, ctx, 0);
                Self::write_dest(op, ctx, ((src as i16) as i32) as u32);
                true
            }
            Operation::ScalarExtendU8 => {
                Self::write_dest(op, ctx, Self::get_source(op, ctx, 0) & 0xFF);
                true
            }
            Operation::ScalarExtendU16 => {
                Self::write_dest(op, ctx, Self::get_source(op, ctx, 0) & 0xFFFF);
                true
            }
            // Division (division by zero returns saturated values)
            Operation::ScalarDiv => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if b == 0 { i32::MIN as u32 } else { ((a as i32).wrapping_div(b as i32)) as u32 };
                Self::write_dest(op, ctx, result);
                true
            }
            Operation::ScalarDivu => {
                let (a, b) = Self::get_two_sources(op, ctx);
                Self::write_dest(op, ctx, if b == 0 { u32::MAX } else { a / b });
                true
            }
            Operation::ScalarMod => {
                let (a, b) = Self::get_two_sources(op, ctx);
                let result = if b == 0 { a } else { ((a as i32).wrapping_rem(b as i32)) as u32 };
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
            Operand::PointerReg(r) => ctx.pointer_read(*r),
            Operand::ModifierReg(r) => ctx.modifier_read(*r),
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
                Operand::ControlReg(id) => {
                    // Control register write: update SRS/UPS config.
                    // Hardware register IDs from AIE2GenRegisterInfo.td.
                    match id {
                        9 => { // crSat
                            ctx.srs_config.saturation_mode = (value & 0x3) as u8;
                            log::trace!("crSat = {} (raw 0x{:X})", value & 0x3, value);
                        }
                        6 => { // crRnd
                            ctx.srs_config.rounding_mode = (value & 0xF) as u8;
                            log::trace!("crRnd = {} (raw 0x{:X})", value & 0xF, value);
                        }
                        8 => { // crSRSSign
                            ctx.srs_config.srs_sign = (value & 1) != 0;
                            log::trace!("crSRSSign = {} (raw 0x{:X})", value & 1, value);
                        }
                        _ => {
                            log::trace!("control register write: id={}, value=0x{:X}", id, value);
                        }
                    }
                }
                _ => {} // Other operand types not valid as scalar destinations
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

    #[test]
    fn test_control_reg_write_crsat() {
        let mut ctx = make_ctx();
        // Write value 3 (symmetric saturate) to crSat (id=9)
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(3))
            .with_dest(Operand::ControlReg(9));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.srs_config.saturation_mode, 3);
    }

    #[test]
    fn test_control_reg_write_crrnd() {
        let mut ctx = make_ctx();
        // Write rounding mode 9 (PosInf) to crRnd (id=6)
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(9))
            .with_dest(Operand::ControlReg(6));
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.srs_config.rounding_mode, 9);
    }

    #[test]
    fn test_control_reg_write_srssign() {
        let mut ctx = make_ctx();
        // Write 1 (signed) to crSRSSign (id=8)
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(1))
            .with_dest(Operand::ControlReg(8));
        ScalarAlu::execute(&op, &mut ctx);
        assert!(ctx.srs_config.srs_sign);
    }

    #[test]
    fn test_control_reg_write_masks_value() {
        let mut ctx = make_ctx();
        // crSat only uses lower 2 bits, crRnd only lower 4 bits.
        // Writing 0xFF should mask to the valid field width.
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(0xFF))
            .with_dest(Operand::ControlReg(9)); // crSat
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.srs_config.saturation_mode, 3); // 0xFF & 0x3

        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarMov)
            .with_source(Operand::Immediate(0xFF))
            .with_dest(Operand::ControlReg(6)); // crRnd
        ScalarAlu::execute(&op, &mut ctx);
        assert_eq!(ctx.srs_config.rounding_mode, 15); // 0xFF & 0xF
    }
}
