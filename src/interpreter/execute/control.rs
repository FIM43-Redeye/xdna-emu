//! Control unit execution.
//!
//! Handles control flow and synchronization operations:
//!
//! - **Branch**: Conditional and unconditional branches
//! - **Call/Return**: Subroutine calls with link register
//! - **Lock**: Acquire/release synchronization primitives
//! - **DMA**: Start/wait for DMA transfers
//! - **Halt**: Stop core execution
//!
//! # Lock Operations (AIE2 Semaphore Model)
//!
//! AIE2 uses semaphore locks with 6-bit unsigned values (0-63).
//! Lock operations can specify expected values and deltas:
//!
//! - **Acquire**: Waits until `value >= expected`, then applies delta
//!   - Operands: lock_id, [expected_value], [delta]
//!   - Default: expected=1, delta=-1 (simple binary semaphore)
//!
//! - **Release**: Applies delta (non-blocking)
//!   - Operands: lock_id, [delta]
//!   - Default: delta=+1 (simple binary semaphore)
//!
//! See AM020 Ch2 and AM025 Lock_Request register for details.

use crate::device::tile::{Tile, LockResult};
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
                let (lock_id, expected, delta) = Self::get_lock_acquire_params(op);
                let lock = &mut tile.locks[lock_id as usize];

                match lock.acquire_with_value(expected, delta) {
                    LockResult::Success => Some(ExecuteResult::Continue),
                    LockResult::WouldUnderflow => Some(ExecuteResult::WaitLock { lock_id }),
                    LockResult::WouldOverflow => {
                        // This shouldn't happen for acquire, but handle it
                        Some(ExecuteResult::Continue)
                    }
                }
            }

            Operation::LockRelease => {
                let (lock_id, delta) = Self::get_lock_release_params(op);
                let lock = &mut tile.locks[lock_id as usize];

                // Release is non-blocking; overflow just saturates
                lock.release_with_value(delta);
                Some(ExecuteResult::Continue)
            }

            Operation::DmaStart => {
                let channel = Self::get_dma_channel(op);
                let bd_index = Self::get_dma_bd(op);
                Self::start_dma(tile, channel, bd_index);
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

    /// Get lock acquire parameters from operands.
    ///
    /// Returns (lock_id, expected_value, delta).
    /// Default: expected=1, delta=-1 (simple binary semaphore).
    fn get_lock_acquire_params(op: &SlotOp) -> (u8, u8, i8) {
        // Lock ID from first operand
        let lock_id = op.sources.first().map_or(0, |src| match src {
            Operand::Lock(id) => *id,
            Operand::Immediate(v) => *v as u8,
            _ => 0,
        });

        // Expected value from second operand (default: 1)
        let expected = op.sources.get(1).map_or(1, |src| match src {
            Operand::Immediate(v) => *v as u8,
            _ => 1,
        });

        // Delta from third operand (default: -1)
        let delta = op.sources.get(2).map_or(-1, |src| match src {
            Operand::Immediate(v) => *v as i8,
            _ => -1,
        });

        (lock_id, expected, delta)
    }

    /// Get lock release parameters from operands.
    ///
    /// Returns (lock_id, delta).
    /// Default: delta=+1 (simple binary semaphore).
    fn get_lock_release_params(op: &SlotOp) -> (u8, i8) {
        // Lock ID from first operand
        let lock_id = op.sources.first().map_or(0, |src| match src {
            Operand::Lock(id) => *id,
            Operand::Immediate(v) => *v as u8,
            _ => 0,
        });

        // Delta from second operand (default: +1)
        let delta = op.sources.get(1).map_or(1, |src| match src {
            Operand::Immediate(v) => *v as i8,
            _ => 1,
        });

        (lock_id, delta)
    }

    /// Get DMA channel from operand.
    fn get_dma_channel(op: &SlotOp) -> u8 {
        op.sources.first().map_or(0, |src| match src {
            Operand::DmaChannel(ch) => *ch,
            Operand::Immediate(v) => *v as u8,
            _ => 0,
        })
    }

    /// Start a DMA transfer.
    ///
    /// This sets up the start request in tile.dma_channels[]. The interpreter
    /// is responsible for syncing this to the actual DmaEngine and stepping it.
    ///
    /// The start_queue field holds the BD index (from immediate operand).
    /// When the interpreter sees running=true and start_queue is set, it calls
    /// DmaEngine.start_channel(channel, start_queue).
    fn start_dma(tile: &mut Tile, channel: u8, bd_index: u8) {
        let ch = &mut tile.dma_channels[channel as usize & 0x03];
        ch.running = true;
        ch.start_queue = bd_index as u32;
    }

    /// Get the BD index for a DMA start operation.
    fn get_dma_bd(op: &SlotOp) -> u8 {
        // BD index is typically in second operand or immediate
        op.sources.get(1).map_or(0, |src| match src {
            Operand::Immediate(v) => *v as u8,
            _ => 0,
        })
    }

    /// Check if DMA transfer is complete.
    ///
    /// Returns true if channel is not marked as running.
    /// The interpreter is responsible for clearing running=false when
    /// the DmaEngine reports the channel as complete.
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
    fn test_lock_acquire_with_value() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Initialize lock with value 5
        tile.locks[2].value = 5;

        // Acquire with expected=3, delta=-2 (should succeed)
        let op = SlotOp::new(SlotIndex::Control, Operation::LockAcquire)
            .with_source(Operand::Lock(2))
            .with_source(Operand::Immediate(3)) // expected value
            .with_source(Operand::Immediate(-2)); // delta

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[2].value, 3); // 5 - 2 = 3
    }

    #[test]
    fn test_lock_acquire_with_value_blocked() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Initialize lock with value 2
        tile.locks[7].value = 2;

        // Acquire with expected=5 (should fail - only have 2)
        let op = SlotOp::new(SlotIndex::Control, Operation::LockAcquire)
            .with_source(Operand::Lock(7))
            .with_source(Operand::Immediate(5)) // expected value
            .with_source(Operand::Immediate(-3)); // delta

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::WaitLock { lock_id: 7 })));
        assert_eq!(tile.locks[7].value, 2); // Unchanged
    }

    #[test]
    fn test_lock_release_with_delta() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[4].value = 5;

        // Release with delta=3
        let op = SlotOp::new(SlotIndex::Control, Operation::LockRelease)
            .with_source(Operand::Lock(4))
            .with_source(Operand::Immediate(3)); // delta

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[4].value, 8); // 5 + 3 = 8
    }

    #[test]
    fn test_lock_release_saturates() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.locks[6].value = 60;

        // Release with delta=10 (should saturate at 63)
        let op = SlotOp::new(SlotIndex::Control, Operation::LockRelease)
            .with_source(Operand::Lock(6))
            .with_source(Operand::Immediate(10)); // delta

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
        assert_eq!(tile.locks[6].value, 63); // Saturated at MAX
        assert!(tile.locks[6].overflow); // Overflow flag set
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
    fn test_dma_start() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Start DMA on channel 0 with BD 5
        let op = SlotOp::new(SlotIndex::Control, Operation::DmaStart)
            .with_source(Operand::DmaChannel(0))
            .with_source(Operand::Immediate(5)); // BD index

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));

        // Verify channel state was set up for interpreter to sync
        assert!(tile.dma_channels[0].running);
        assert_eq!(tile.dma_channels[0].start_queue, 5);
    }

    #[test]
    fn test_dma_wait_blocks() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set channel as running (simulates DMA in progress)
        tile.dma_channels[1].running = true;

        // DmaWait should return WaitDma since channel is busy
        let op = SlotOp::new(SlotIndex::Control, Operation::DmaWait)
            .with_source(Operand::DmaChannel(1));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::WaitDma { channel: 1 })));
    }

    #[test]
    fn test_dma_wait_completes() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Channel not running (transfer complete)
        tile.dma_channels[2].running = false;

        // DmaWait should succeed
        let op = SlotOp::new(SlotIndex::Control, Operation::DmaWait)
            .with_source(Operand::DmaChannel(2));

        let result = ControlUnit::execute(&op, &mut ctx, &mut tile);
        assert!(matches!(result, Some(ExecuteResult::Continue)));
    }
}
