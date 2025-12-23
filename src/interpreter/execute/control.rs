//! Control unit execution.
//!
//! Handles control flow and synchronization operations:
//!
//! - **Branch**: Conditional and unconditional branches
//! - **Call/Return**: Subroutine calls with link register
//! - **Lock**: Acquire/release synchronization primitives
//! - **DMA**: Start/wait for DMA transfers
//! - **Halt**: Stop core execution

use crate::device::tile::Tile;
use crate::interpreter::bundle::{BranchCondition, Operation, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{ExecuteResult, Flags};

/// Control unit for branches, calls, and synchronization.
pub struct ControlUnit;

impl ControlUnit {
    /// Execute a control operation.
    ///
    /// Returns `Some(result)` if handled, `None` if not a control op.
    pub fn execute(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> Option<ExecuteResult> {
        match &op.op {
            Operation::Branch { condition } => {
                let target = Self::get_branch_target(op, ctx);
                if Self::evaluate_condition(*condition, ctx.flags()) {
                    Some(ExecuteResult::Branch { target })
                } else {
                    Some(ExecuteResult::Continue)
                }
            }

            Operation::Call => {
                let target = Self::get_branch_target(op, ctx);
                // Save return address (PC + instruction size handled by caller)
                // Link register will be set by the executor after this returns
                Some(ExecuteResult::Branch { target })
            }

            Operation::Return => {
                let target = ctx.lr();
                Some(ExecuteResult::Branch { target })
            }

            Operation::LockAcquire => {
                let lock_id = Self::get_lock_id(op);
                if tile.locks[lock_id as usize].acquire() {
                    Some(ExecuteResult::Continue)
                } else {
                    Some(ExecuteResult::WaitLock { lock_id })
                }
            }

            Operation::LockRelease => {
                let lock_id = Self::get_lock_id(op);
                tile.locks[lock_id as usize].release();
                Some(ExecuteResult::Continue)
            }

            Operation::DmaStart => {
                let channel = Self::get_dma_channel(op);
                Self::start_dma(tile, channel);
                Some(ExecuteResult::Continue)
            }

            Operation::DmaWait => {
                let channel = Self::get_dma_channel(op);
                if Self::is_dma_complete(tile, channel) {
                    Some(ExecuteResult::Continue)
                } else {
                    Some(ExecuteResult::WaitDma { channel })
                }
            }

            Operation::Halt => Some(ExecuteResult::Halt),

            Operation::Nop => Some(ExecuteResult::Continue),

            _ => None, // Not a control operation
        }
    }

    /// Get branch target address from operand.
    fn get_branch_target(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        op.sources.first().map_or(0, |src| match src {
            Operand::Immediate(v) => *v as u32,
            Operand::ScalarReg(r) => ctx.scalar.read(*r),
            _ => 0,
        })
    }

    /// Evaluate a branch condition against flags.
    pub fn evaluate_condition(condition: BranchCondition, flags: Flags) -> bool {
        match condition {
            BranchCondition::Always => true,
            BranchCondition::Equal => flags.z,
            BranchCondition::NotEqual => !flags.z,
            BranchCondition::Less => flags.n != flags.v, // Signed less
            BranchCondition::GreaterEqual => flags.n == flags.v, // Signed >=
            BranchCondition::LessEqual => flags.z || (flags.n != flags.v), // Signed <=
            BranchCondition::Greater => !flags.z && (flags.n == flags.v), // Signed >
            BranchCondition::Negative => flags.n,
            BranchCondition::PositiveOrZero => !flags.n,
            BranchCondition::CarrySet => flags.c,
            BranchCondition::CarryClear => !flags.c,
            BranchCondition::OverflowSet => flags.v,
            BranchCondition::OverflowClear => !flags.v,
        }
    }

    /// Get lock ID from operand.
    fn get_lock_id(op: &SlotOp) -> u8 {
        op.sources.first().map_or(0, |src| match src {
            Operand::Lock(id) => *id,
            Operand::Immediate(v) => *v as u8,
            _ => 0,
        })
    }

    /// Get DMA channel from operand.
    fn get_dma_channel(op: &SlotOp) -> u8 {
        op.sources.first().map_or(0, |src| match src {
            Operand::DmaChannel(ch) => *ch,
            Operand::Immediate(v) => *v as u8,
            _ => 0,
        })
    }

    /// Start a DMA transfer (simplified - instant completion for now).
    fn start_dma(tile: &mut Tile, channel: u8) {
        let ch = &mut tile.dma_channels[channel as usize & 0x03];
        ch.running = true;
        // In a real implementation, this would schedule the transfer
        // For now, mark as complete immediately
        ch.running = false;
    }

    /// Check if DMA transfer is complete.
    fn is_dma_complete(tile: &Tile, channel: u8) -> bool {
        !tile.dma_channels[channel as usize & 0x03].running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_tile() -> Tile {
        Tile::compute(0, 2)
    }

    #[test]
    fn test_unconditional_branch() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::new(
            SlotIndex::Control,
            Operation::Branch {
                condition: BranchCondition::Always,
            },
        )
        .with_source(Operand::Immediate(0x1000));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x1000 })));
    }

    #[test]
    fn test_conditional_branch_taken() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set zero flag
        ctx.set_flags(Flags {
            z: true,
            n: false,
            c: false,
            v: false,
        });

        let op = SlotOp::new(
            SlotIndex::Control,
            Operation::Branch {
                condition: BranchCondition::Equal,
            },
        )
        .with_source(Operand::Immediate(0x2000));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x2000 })));
    }

    #[test]
    fn test_conditional_branch_not_taken() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Zero flag not set
        ctx.set_flags(Flags {
            z: false,
            n: false,
            c: false,
            v: false,
        });

        let op = SlotOp::new(
            SlotIndex::Control,
            Operation::Branch {
                condition: BranchCondition::Equal,
            },
        )
        .with_source(Operand::Immediate(0x2000));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }

    #[test]
    fn test_return() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.set_lr(0x5000);

        let op = SlotOp::new(SlotIndex::Control, Operation::Return);

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Branch { target: 0x5000 })));
    }

    #[test]
    fn test_lock_acquire_success() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Initialize lock with value 1 (available)
        tile.locks[5].value = 1;

        let op = SlotOp::new(SlotIndex::Control, Operation::LockAcquire)
            .with_source(Operand::Lock(5));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[5].value, 0); // Lock acquired
    }

    #[test]
    fn test_lock_acquire_blocked() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Lock value 0 (unavailable)
        tile.locks[5].value = 0;

        let op = SlotOp::new(SlotIndex::Control, Operation::LockAcquire)
            .with_source(Operand::Lock(5));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::WaitLock { lock_id: 5 })));
    }

    #[test]
    fn test_lock_release() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[3].value = 0;

        let op = SlotOp::new(SlotIndex::Control, Operation::LockRelease)
            .with_source(Operand::Lock(3));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[3].value, 1); // Lock released
    }

    #[test]
    fn test_halt() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::new(SlotIndex::Control, Operation::Halt);

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Halt)));
    }

    #[test]
    fn test_nop() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::nop(SlotIndex::Control);

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }

    #[test]
    fn test_condition_evaluation() {
        // Equal
        assert!(ControlUnit::evaluate_condition(
            BranchCondition::Equal,
            Flags { z: true, n: false, c: false, v: false }
        ));
        assert!(!ControlUnit::evaluate_condition(
            BranchCondition::Equal,
            Flags { z: false, n: false, c: false, v: false }
        ));

        // Less (signed: n != v)
        assert!(ControlUnit::evaluate_condition(
            BranchCondition::Less,
            Flags { z: false, n: true, c: false, v: false }
        ));
        assert!(!ControlUnit::evaluate_condition(
            BranchCondition::Less,
            Flags { z: false, n: true, c: false, v: true }
        ));

        // Greater (signed: !z && n == v)
        assert!(ControlUnit::evaluate_condition(
            BranchCondition::Greater,
            Flags { z: false, n: false, c: false, v: false }
        ));
        assert!(!ControlUnit::evaluate_condition(
            BranchCondition::Greater,
            Flags { z: true, n: false, c: false, v: false }
        ));
    }

    #[test]
    fn test_dma_operations() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Start DMA
        let op = SlotOp::new(SlotIndex::Control, Operation::DmaStart)
            .with_source(Operand::DmaChannel(0));
        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));

        // Wait DMA (should complete immediately in current implementation)
        let op = SlotOp::new(SlotIndex::Control, Operation::DmaWait)
            .with_source(Operand::DmaChannel(0));
        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }
}
