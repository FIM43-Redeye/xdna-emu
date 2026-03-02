//! Scalar ALU execution unit.
//!
//! Handles scalar operations that require special treatment beyond the
//! standard SemanticOp dispatch path (e.g., immediate moves, flag-based
//! select operations).
//!
//! ## CPU Flag Behavior (AIE2)
//!
//! AIE2 has only ONE hardware flag -- the **Carry flag (C)** in SR bit 0.
//! Zero (Z), Negative (N), and Overflow (V) are computed on-demand by branch logic.

use crate::interpreter::bundle::{Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;

/// Scalar ALU execution unit.
///
/// Handles scalar operations not covered by the primary semantic dispatch.
pub struct ScalarAlu;

impl ScalarAlu {
    /// Execute a scalar operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not.
    ///
    /// Called AFTER `execute_semantic()`. All scalar operations are now
    /// handled by semantic dispatch -- this is a stub for the dispatch chain.
    pub fn execute(_op: &SlotOp, _ctx: &mut ExecutionContext) -> bool {
        false
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
    use crate::interpreter::execute::semantic::execute_semantic;
    use crate::tablegen::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    // Tests for operations now handled by execute_semantic().
    // ScalarAlu no longer handles these, so we test through the
    // semantic dispatch path (which is the real execution path).

    #[test]
    fn test_scalar_add() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar.read(2), 30);
    }

    #[test]
    fn test_scalar_add_overflow() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 0xFFFF_FFFF);
        ctx.scalar.write(1, 1);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0);
        assert!(ctx.flags().c); // Carry set
        assert!(ctx.flags().z); // Zero set
    }

    #[test]
    fn test_scalar_sub() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 30);
        ctx.scalar.write(1, 10);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Sub)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 20);
    }

    #[test]
    fn test_scalar_mul() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 6);
        ctx.scalar.write(1, 7);

        let op = SlotOp::from_semantic(SlotIndex::Scalar1, SemanticOp::Mul)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 42);
    }

    #[test]
    fn test_scalar_logic() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 0xFF00);
        ctx.scalar.write(1, 0x0FF0);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::And)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0x0F00);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Or)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0xFFF0);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Xor)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0xF0F0);
    }

    #[test]
    fn test_scalar_shifts() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 0x0000_00FF);
        ctx.scalar.write(1, 4);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Shl)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0x0000_0FF0);

        ctx.scalar.write(0, 0x0000_0FF0);
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Srl)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0x0000_00FF);

        ctx.scalar.write(0, 0x8000_0000u32);
        ctx.scalar.write(1, 4);
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Sra)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(2), 0xF800_0000u32);
    }

    #[test]
    fn test_scalar_mov() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 42);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(0));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(1), 42);
    }

    // ScalarMovi now handled as Copy with Immediate source
    #[test]
    fn test_scalar_movi() {
        let mut ctx = make_ctx();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_dest(Operand::ScalarReg(5))
            .with_source(Operand::Immediate(12345));
        assert!(execute_semantic(&op, &mut ctx));
        assert_eq!(ctx.scalar.read(5), 12345);
    }

    // Cmp now handled by execute_semantic (SemanticOp::Cmp)
    #[test]
    fn test_scalar_cmp() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 10);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Cmp)
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        execute_semantic(&op, &mut ctx);
        assert!(ctx.flags().z);

        ctx.scalar.write(1, 20);
        execute_semantic(&op, &mut ctx);
        assert!(!ctx.flags().z);
        assert!(ctx.flags().n);
    }

    #[test]
    fn test_immediate_operand() {
        let mut ctx = make_ctx();
        ctx.scalar.write(0, 10);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::Immediate(5));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(1), 15);
    }

    #[test]
    fn test_pointer_reg_operand() {
        let mut ctx = make_ctx();
        ctx.pointer.write(0, 0x1000);
        ctx.scalar.write(0, 0x100);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::PointerReg(0))
            .with_source(Operand::ScalarReg(0));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.scalar.read(1), 0x1100);
    }

    // Control register writes go through semantic dispatch (Copy path)
    #[test]
    fn test_control_reg_write_crsat() {
        let mut ctx = make_ctx();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_source(Operand::Immediate(3))
            .with_dest(Operand::ControlReg(9));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.srs_config.saturation_mode, 3);
    }

    #[test]
    fn test_control_reg_write_crrnd() {
        let mut ctx = make_ctx();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_source(Operand::Immediate(9))
            .with_dest(Operand::ControlReg(6));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.srs_config.rounding_mode, 9);
    }

    #[test]
    fn test_control_reg_write_srssign() {
        let mut ctx = make_ctx();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_source(Operand::Immediate(1))
            .with_dest(Operand::ControlReg(8));
        execute_semantic(&op, &mut ctx);
        assert!(ctx.srs_config.srs_sign);
    }

    #[test]
    fn test_control_reg_write_masks_value() {
        let mut ctx = make_ctx();
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_source(Operand::Immediate(0xFF))
            .with_dest(Operand::ControlReg(9));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.srs_config.saturation_mode, 3); // 0xFF & 0x3

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Copy)
            .with_source(Operand::Immediate(0xFF))
            .with_dest(Operand::ControlReg(6));
        execute_semantic(&op, &mut ctx);
        assert_eq!(ctx.srs_config.rounding_mode, 15); // 0xFF & 0xF
    }
}
