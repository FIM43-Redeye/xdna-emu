//! Execution units for AIE2 operations.
//!
//! This module provides the execution logic for decoded operations.
//!
//! # Execution Dispatch Architecture
//!
//! Operations are dispatched through a chain of execution units. The
//! **semantic dispatcher** is the preferred path for pure computations:
//!
//! ```text
//! CycleAccurateExecutor::execute_slot(op)
//!         |
//!         v
//!   ┌─────────────────────────┐
//!   │  execute_semantic(op)   │ <-- TableGen-driven, ~40 handlers
//!   │  (pure register ops)    │     Covers: Add, Sub, Mul, And, Or, ...
//!   └────────────┬────────────┘
//!                | false (no semantic or unhandled)
//!                v
//!   ┌─────────────────────────┐
//!   │   ScalarAlu::execute()  │ <-- Legacy fallback, sets CPU flags
//!   └────────────┬────────────┘
//!                | false
//!                v
//!   ┌─────────────────────────┐
//!   │   VectorAlu::execute()  │ <-- SIMD operations
//!   └────────────┬────────────┘
//!                | false
//!                v
//!   ┌─────────────────────────┐
//!   │   MemoryUnit::execute() │ <-- Load/Store (needs tile access)
//!   └────────────┬────────────┘
//!                | false
//!                v
//!   ┌─────────────────────────┐
//!   │   StreamOps::execute()  │ <-- Stream I/O (needs stream switch)
//!   └────────────┬────────────┘
//!                | NotStreamOp
//!                v
//!   ┌─────────────────────────┐
//!   │  ControlUnit::execute() │ <-- Branch, Lock, Halt (returns ExecuteResult)
//!   └─────────────────────────┘
//! ```
//!
//! # Unit Responsibilities
//!
//! | Unit | Purpose | Semantic Dispatch? |
//! |------|---------|-------------------|
//! | [`semantic`] | TableGen-driven pure ops | **Primary** |
//! | [`ScalarAlu`] | Legacy scalar + flag-setting | Fallback |
//! | [`VectorAlu`] | SIMD operations | Fallback |
//! | [`MemoryUnit`] | Load/Store (tile memory) | No (needs tile) |
//! | [`StreamOps`] | Stream I/O | No (needs switch) |
//! | [`ControlUnit`] | Branch/Lock/Halt | No (control flow) |
//!
//! # Migration Path
//!
//! The goal is to migrate pure computational operations to `execute_semantic()`,
//! which uses TableGen-derived SemanticOp patterns. This reduces the 133+
//! Operation variants to ~40 semantic handlers and ensures operand ordering
//! matches the ISA specification.
//!
//! See the plan at `~/.claude/plans/memoized-soaring-teapot.md` for details.
//!
//! # Executor
//!
//! The [`CycleAccurateExecutor`] models AM020-based instruction latencies,
//! register hazards (RAW/WAW), and memory bank conflicts. This is the only
//! execution mode - cycle accuracy is always enabled for reliable behavior.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::execute::CycleAccurateExecutor;
//! use xdna_emu::interpreter::{VliwBundle, ExecutionContext};
//!
//! let mut executor = CycleAccurateExecutor::new();
//! let result = executor.execute(&bundle, &mut ctx, &mut tile);
//! ```

mod scalar;
mod vector;
mod memory;
mod control;
mod stream;
mod cycle_accurate;
mod semantic;

pub use scalar::ScalarAlu;
pub use vector::VectorAlu;
pub use memory::MemoryUnit;
pub use control::ControlUnit;
pub use stream::{StreamOps, StreamResult};
pub use cycle_accurate::{CycleAccurateExecutor, CycleAccurateStats};
pub use semantic::execute_semantic;

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
    fn test_executor_nop() {
        let mut executor = CycleAccurateExecutor::new();
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(0, 2);

        let bundle = make_bundle_with_op(SlotOp::nop(SlotIndex::Scalar0));

        let result = executor.execute(&bundle, &mut ctx, &mut tile);
        assert!(matches!(result, crate::interpreter::traits::ExecuteResult::Continue));
    }

    #[test]
    fn test_executor_scalar_add() {
        let mut executor = CycleAccurateExecutor::new();
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
