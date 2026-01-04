//! Fast executor implementation.
//!
//! The `FastExecutor` executes all operations within a VLIW bundle in a single
//! cycle, without modeling pipeline stages or hazards. This provides good
//! performance for functional verification while sacrificing cycle accuracy.
//!
//! # Execution Model
//!
//! 1. All slot operations within a bundle execute "simultaneously"
//! 2. Register reads happen before any writes (read-then-write ordering)
//! 3. Memory operations are atomic within a bundle
//! 4. Control flow changes take effect at bundle boundaries

use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operation, VliwBundle};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{ExecuteResult, Executor};

use super::control::ControlUnit;
use super::memory::MemoryUnit;
use super::scalar::ScalarAlu;
use super::stream::{StreamOps, StreamResult};
use super::vector::VectorAlu;

/// Fast executor that executes bundles in a single cycle.
///
/// This executor does not model:
/// - Pipeline stages
/// - Register hazards
/// - Memory bank conflicts
/// - Timing delays
///
/// It is suitable for:
/// - Functional verification
/// - Fast simulation
/// - Debug stepping
pub struct FastExecutor {
    /// Track whether we're in a call (to set link register).
    pending_call_return_addr: Option<u32>,
}

impl FastExecutor {
    /// Create a new fast executor.
    pub fn new() -> Self {
        Self {
            pending_call_return_addr: None,
        }
    }

    /// Execute a single slot operation.
    fn execute_slot(
        &mut self,
        op: &crate::interpreter::bundle::SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> Option<ExecuteResult> {
        // Check for call - need to save return address before branching
        if matches!(op.op, Operation::Call) {
            // The return address will be set by execute() after all ops
            self.pending_call_return_addr = Some(ctx.pc());
        }

        // Try each execution unit in order
        if ScalarAlu::execute(op, ctx) {
            return None; // Scalar op handled, continue with next slot
        }

        if VectorAlu::execute(op, ctx) {
            return None; // Vector op handled
        }

        if MemoryUnit::execute(op, ctx, tile) {
            return None; // Memory op handled
        }

        match StreamOps::execute(op, ctx, tile) {
            StreamResult::Completed => return None, // Stream op handled
            StreamResult::Stall { port } => {
                // Blocking stream read with no data - return WaitStream
                return Some(ExecuteResult::WaitStream { port });
            }
            StreamResult::NotStreamOp => {} // Fall through to next unit
        }

        if let Some(result) = ControlUnit::execute(op, ctx, tile) {
            return Some(result);
        }

        // Unknown operation - fail loudly to prevent silent incorrect behavior
        if let Operation::Unknown { opcode } = &op.op {
            return Some(ExecuteResult::Error {
                message: format!(
                    "Unknown instruction opcode 0x{:08X} at slot {:?}",
                    opcode, op.slot
                ),
            });
        }

        None
    }
}

impl Default for FastExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for FastExecutor {
    fn execute(
        &mut self,
        bundle: &VliwBundle,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
    ) -> ExecuteResult {
        self.pending_call_return_addr = None;

        let mut final_result = ExecuteResult::Continue;

        // Snapshot scalar registers for VLIW parallel read semantics.
        // All reads within the bundle will see pre-execution values.
        ctx.begin_bundle();

        // Execute all slot operations
        for op in bundle.active_slots() {
            #[cfg(test)]
            eprintln!("[EXEC] Slot {:?}: op={:?}", op.slot, op.op);

            // Debug log for PointerMov execution
            if matches!(op.op, Operation::PointerMov) {
                log::debug!("[EXEC PMOV] slot={:?} dest={:?} sources={:?}", op.slot, op.dest, op.sources);
            }

            if let Some(result) = self.execute_slot(op, ctx, tile) {
                // Control flow result - remember it
                match &result {
                    ExecuteResult::Branch { .. }
                    | ExecuteResult::Halt
                    | ExecuteResult::WaitLock { .. }
                    | ExecuteResult::WaitDma { .. }
                    | ExecuteResult::WaitStream { .. } => {
                        final_result = result;
                    }
                    ExecuteResult::Continue => {}
                    ExecuteResult::Error { .. } => {
                        return result; // Immediate return on error
                    }
                }
            }
        }

        // End VLIW bundle - clear the snapshot
        ctx.end_bundle();

        // Handle call return address
        if let Some(return_addr) = self.pending_call_return_addr {
            // Set LR to the address after this bundle
            ctx.set_lr(return_addr.wrapping_add(bundle.size() as u32));
        }

        // Update statistics
        ctx.record_instruction(1);

        final_result
    }

    fn is_cycle_accurate(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::{
        BranchCondition, MemWidth, Operand, Operation, PostModify, SlotIndex, SlotOp,
    };
    use crate::interpreter::traits::Flags;

    fn make_bundle(ops: Vec<SlotOp>) -> VliwBundle {
        let mut bundle = VliwBundle::empty();
        for op in ops {
            bundle.set_slot(op);
        }
        bundle
    }

    #[test]
    fn test_empty_bundle() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![]);
        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Continue));
        assert_eq!(ctx.instructions, 1);
    }

    #[test]
    fn test_multiple_scalar_ops() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);
        ctx.scalar.write(2, 5);

        // r3 = r0 + r1, r4 = r2 * r2 (in parallel)
        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
                .with_dest(Operand::ScalarReg(3))
                .with_source(Operand::ScalarReg(0))
                .with_source(Operand::ScalarReg(1)),
            SlotOp::new(SlotIndex::Scalar1, Operation::ScalarMul)
                .with_dest(Operand::ScalarReg(4))
                .with_source(Operand::ScalarReg(2))
                .with_source(Operand::ScalarReg(2)),
        ]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Continue));
        assert_eq!(ctx.scalar.read(3), 30); // 10 + 20
        assert_eq!(ctx.scalar.read(4), 25); // 5 * 5
    }

    #[test]
    fn test_load_and_compute() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Setup: memory[0x100] = 42
        tile.write_data_u32(0x100, 42);
        ctx.pointer.write(0, 0x100);
        ctx.scalar.write(0, 10);

        // r1 = [p0], r2 = r0 + 5 (in parallel)
        let bundle = make_bundle(vec![
            SlotOp::new(
                SlotIndex::Load,
                Operation::Load {
                    width: MemWidth::Word,
                    post_modify: PostModify::None,
                },
            )
            .with_dest(Operand::ScalarReg(1))
            .with_source(Operand::PointerReg(0)),
            SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
                .with_dest(Operand::ScalarReg(2))
                .with_source(Operand::ScalarReg(0))
                .with_source(Operand::Immediate(5)),
        ]);

        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.scalar.read(1), 42);
        assert_eq!(ctx.scalar.read(2), 15);
    }

    #[test]
    fn test_branch_taken() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.set_flags(Flags {
            z: true,
            n: false,
            c: false,
            v: false,
        });

        let bundle = make_bundle(vec![SlotOp::new(
            SlotIndex::Control,
            Operation::Branch {
                condition: BranchCondition::Equal,
            },
        )
        .with_source(Operand::Immediate(0x2000))]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Branch { target: 0x2000 }));
    }

    #[test]
    fn test_call_sets_link_register() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        ctx.set_pc(0x1000);

        let bundle = make_bundle(vec![
            SlotOp::new(SlotIndex::Control, Operation::Call)
                .with_source(Operand::Immediate(0x3000))
        ]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Branch { target: 0x3000 }));
        assert_eq!(ctx.lr(), 0x1004); // Return address = PC + size
    }

    #[test]
    fn test_halt() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![SlotOp::new(SlotIndex::Control, Operation::Halt)]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::Halt));
    }

    #[test]
    fn test_lock_wait() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // Lock 3 is unavailable
        tile.locks[3].value = 0;

        let bundle =
            make_bundle(vec![SlotOp::new(SlotIndex::Control, Operation::LockAcquire)
                .with_source(Operand::Lock(3))]);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);

        assert!(matches!(result, ExecuteResult::WaitLock { lock_id: 3 }));
    }

    #[test]
    fn test_instruction_count() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle(vec![SlotOp::nop(SlotIndex::Scalar0)]);

        executor.execute(&bundle, &mut ctx, &mut tile);
        executor.execute(&bundle, &mut ctx, &mut tile);
        executor.execute(&bundle, &mut ctx, &mut tile);

        assert_eq!(ctx.instructions, 3);
        assert_eq!(ctx.cycles, 3);
    }

    #[test]
    fn test_is_not_cycle_accurate() {
        let executor = FastExecutor::new();
        assert!(!executor.is_cycle_accurate());
    }
}
