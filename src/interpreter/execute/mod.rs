//! Execution units for AIE2 operations.
//!
//! This module provides the execution logic for decoded operations.
//! Each unit handles a specific category of operations:
//!
//! | Unit | Operations |
//! |------|------------|
//! | Scalar ALU | Integer arithmetic, logic, shifts |
//! | Vector ALU | SIMD arithmetic, shuffle |
//! | Memory | Load/store operations |
//! | Control | Branch, call, return |
//! | Special | Lock, DMA operations |
//!
//! # Architecture
//!
//! The `FastExecutor` executes all operations in a single cycle (no pipeline).
//! A future `CycleAccurateExecutor` would model the real pipeline stages.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::execute::FastExecutor;
//! use xdna_emu::interpreter::{VliwBundle, ExecutionContext};
//!
//! let mut executor = FastExecutor::new();
//! let result = executor.execute(&bundle, &mut ctx, &mut tile);
//! ```

mod scalar;
mod vector;
mod memory;
mod control;
mod fast_executor;

pub use scalar::ScalarAlu;
pub use vector::VectorAlu;
pub use memory::MemoryUnit;
pub use control::ControlUnit;
pub use fast_executor::FastExecutor;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tile::Tile;
    use crate::interpreter::state::ExecutionContext;
    use crate::interpreter::traits::Executor;
    use crate::interpreter::bundle::{VliwBundle, SlotOp, SlotIndex, Operation, Operand};

    fn make_bundle_with_op(op: SlotOp) -> VliwBundle {
        let mut bundle = VliwBundle::empty();
        bundle.set_slot(op);
        bundle
    }

    #[test]
    fn test_fast_executor_nop() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle_with_op(SlotOp::nop(SlotIndex::Scalar0));

        let result = executor.execute(&bundle, &mut ctx, &mut tile);
        assert!(matches!(result, crate::interpreter::traits::ExecuteResult::Continue));
    }

    #[test]
    fn test_fast_executor_scalar_add() {
        let mut executor = FastExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        // r0 = 10, r1 = 20
        ctx.scalar.write(0, 10);
        ctx.scalar.write(1, 20);

        // r2 = r0 + r1
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));

        let bundle = make_bundle_with_op(op);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);
        assert!(matches!(result, crate::interpreter::traits::ExecuteResult::Continue));
        assert_eq!(ctx.scalar.read(2), 30);
    }
}
