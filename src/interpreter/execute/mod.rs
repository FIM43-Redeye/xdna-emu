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
//!   │  execute_semantic(op)   │ <-- TableGen-driven, ~85 handlers
//!   │  (pure register ops)    │     Covers: Add, Sub, Mul, And, Or, ...
//!   └────────────┬────────────┘
//!                | false (not a pure register op)
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
//!   │   CascadeOps::execute() │ <-- 384-bit cascade link
//!   └────────────┬────────────┘
//!                | NotCascadeOp
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
//! | [`VectorAlu`] | SIMD operations | Fallback |
//! | [`MemoryUnit`] | Load/Store (tile memory) | No (needs tile) |
//! | [`CascadeOps`] | 384-bit cascade link | No (needs tile) |
//! | [`StreamOps`] | Stream I/O | No (needs switch) |
//! | [`ControlUnit`] | Branch/Lock/Halt | No (control flow) |
//!
//! # Migration Path
//!
//! All execution is driven by `SemanticOp` from TableGen, with ~85 semantic
//! handlers covering the full instruction set. Metadata (element type, memory
//! width, etc.) comes from SlotOp fields rather than enum variants.
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

mod vector;
pub mod vector_semantic;
pub mod vector_pack;
pub mod vector_permute;
pub mod vector_srs;
pub mod vector_ups;
pub mod vector_config;
pub mod vector_float;
pub mod vector_matmul;
pub mod vector_matmul_sparse;
pub mod vmac_hw;
mod memory;
mod control;
mod stream;
mod cascade;
mod cycle_accurate;
mod semantic;

pub use vector::VectorAlu;
pub use memory::{MemoryUnit, NeighborMemory};
pub use control::ControlUnit;
pub use stream::{StreamOps, StreamResult};
pub use cascade::{CascadeOps, CascadeResult};
pub use cycle_accurate::{CycleAccurateExecutor, CycleAccurateStats};
pub use semantic::execute_semantic;

#[cfg(test)]
mod vector_validate;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tile::Tile;
    use crate::interpreter::state::ExecutionContext;
    use crate::interpreter::traits::Executor;
    use crate::interpreter::bundle::{VliwBundle, SlotOp, SlotIndex, Operand};
    use crate::tablegen::SemanticOp;

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
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));

        let bundle = make_bundle_with_op(op);

        let result = executor.execute(&bundle, &mut ctx, &mut tile);
        assert!(matches!(result, crate::interpreter::traits::ExecuteResult::Continue));
        assert_eq!(ctx.scalar.read(2), 30);
    }
}
