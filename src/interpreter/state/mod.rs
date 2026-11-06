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
pub mod register_widths;
pub mod scrub;
pub mod write_collision;
pub mod scalar_collision;
mod event_trace;
mod timing_context;
mod context;
mod sparse_load;
pub mod load_ring;
pub mod carried;
#[cfg(test)]
mod acc_timing_golden;

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
    SPARSE_OF_REG_INDEX,
    COMPR_UF_REG_INDEX,
    SS_STATUS_REG_INDEX,
    SHIFT_REG_BASE,
    NUM_SHIFT_REGS,
    MS_STATUS_REG_INDEX,
    CARRY_REG_INDEX,
    SRS_OF_REG_INDEX,
    UPS_OF_REG_INDEX,
    FP_FLAGS_REG_INDEX,
    F2I_FLAGS_REG_INDEX,
    F2F_FLAGS_REG_INDEX,
    SP_PTR_INDEX,
    // q-register views in the ControlReg index space.
    CTRL_BASE_Q,
    CTRL_BASE_QWL,
    CTRL_BASE_QWH,
    CTRL_BASE_QL,
    CTRL_BASE_QH,
    CTRL_Q_COUNT,
    // Modifier register sub-class base indices.
    MOD_BASE_M,
    MOD_BASE_DN,
    MOD_BASE_DJ,
    MOD_BASE_DC,
    is_descriptor_operand,
    MOD_BASE_D2D,
    MOD_BASE_D3D,
    // Bypass classes (re-exported so execute helpers can use crate::interpreter::state::Bypass).
    Bypass,
    // TableGen validation.
    validate_register_model,
};
pub use sparse_load::{SparseLoadUnit, SparseRead};
pub use load_ring::{LoadPort, LoadPortRings};
pub use context::{
    DeferredPcKind, EventLog, EventType, ExecutionContext, PendingBranch, PendingDeferredEvent, PendingWrite,
    SpRegister, SrsConfig, TimestampedEvent, TimingContext, UnpackSource, TRACE_PC_PIPELINE_DEPTH,
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
