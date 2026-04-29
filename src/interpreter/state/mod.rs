//! Processor state for AIE2 cores.
//!
//! This module provides the complete register state for an AIE2 compute core:
//!
//! | Register File | Count | Width | Purpose |
//! |---------------|-------|-------|---------|
//! | Scalar GPR | 32 | 32-bit | General purpose integers |
//! | Pointer | 8 | 20-bit | Memory addressing |
//! | Modifier | 32 | 20-bit | Post-modify + AGU (m/dn/dj/dc) |
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
mod event_trace;
mod timing_context;
mod context;

pub use registers::{
    ScalarRegisterFile,
    VectorRegisterFile,
    AccumulatorRegisterFile,
    PointerRegisterFile,
    MaskRegisterFile,
    ModifierRegisterFile,
    // Wide register type aliases.
    Vec512,
    Acc1024,
    // Register file dimensions and special register indices.
    NUM_SCALAR_REGS,
    NUM_SCALAR_GPRS,
    LR_REG_INDEX,
    LS_REG_INDEX,
    LE_REG_INDEX,
    LC_REG_INDEX,
    DP_REG_INDEX,
    CORE_ID_REG_INDEX,
    SP_PTR_INDEX,
    // Modifier register sub-class base indices.
    MOD_BASE_M,
    MOD_BASE_DN,
    MOD_BASE_DJ,
    MOD_BASE_DC,
    // TableGen validation.
    validate_register_model,
};
pub use context::{
    ExecutionContext, SpRegister, TimingContext, EventLog, EventType, TimestampedEvent, PendingBranch,
    PendingWrite, SrsConfig,
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
