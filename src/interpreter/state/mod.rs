//! Processor state for AIE2 cores.
//!
//! This module provides the complete register state for an AIE2 compute core:
//!
//! | Register File | Count | Width | Purpose |
//! |---------------|-------|-------|---------|
//! | Scalar GPR | 32 | 32-bit | General purpose integers |
//! | Pointer | 8 | 20-bit | Memory addressing |
//! | Modifier | 8 | 20-bit | Post-modify addressing |
//! | Vector | 32 | 256-bit | SIMD operations |
//! | Accumulator | 8 | 512-bit | MAC operations |
//!
//! # Architecture
//!
//! The state is split into separate register file structs for cache efficiency.
//! Hot registers (scalar GPRs, PC, flags) are together; vector/accumulator
//! registers are accessed less frequently.
//!
//! # Example
//!
//! ```
//! use xdna_emu::interpreter::state::ExecutionContext;
//!
//! let mut ctx = ExecutionContext::new();
//! ctx.scalar.write(0, 42);  // r0 = 42
//! ctx.set_pc(0x1000);       // PC = 0x1000
//! ```

mod registers;
mod context;

pub use registers::{
    ScalarRegisterFile,
    VectorRegisterFile,
    AccumulatorRegisterFile,
    PointerRegisterFile,
    ModifierRegisterFile,
};
pub use context::{
    ExecutionContext, SpRegister, TimingContext,
    EventLog, EventType, TimestampedEvent,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_context_basic() {
        let mut ctx = ExecutionContext::new();

        // Test PC
        ctx.set_pc(0x1000);
        assert_eq!(ctx.pc(), 0x1000);

        // Test scalar register
        ctx.scalar.write(5, 42);
        assert_eq!(ctx.scalar.read(5), 42);

        // Test cycle counting
        assert_eq!(ctx.cycles, 0);
        ctx.cycles += 1;
        assert_eq!(ctx.cycles, 1);
    }
}
