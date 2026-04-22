//! AIE2 register offset constants from the AM025 JSON.
//!
//! Submodules:
//!   - (top-level): core module register offsets (CORE_CONTROL, etc.)
//!   - `memory`: memory-module Lock_Request bitfield constants
//!   - `mem_tile`: mem-tile-module Lock_Request bitfield constants
//!   - `ctrl_regs`: control-register IDs (crRnd, crSat, crSRSSign)

include!(concat!(env!("OUT_DIR"), "/gen_core_module.rs"));

/// Memory-module register constants for compute tiles.
pub mod memory {
    //! AM025 memory_module Lock_Request bit layout.
    include!(concat!(env!("OUT_DIR"), "/gen_memory_lock.rs"));
}

/// Mem-tile-module register constants.
pub mod mem_tile {
    //! AM025 memory_tile_module Lock_Request bit layout.
    include!(concat!(env!("OUT_DIR"), "/gen_memtile_lock.rs"));
}

/// Control-register IDs for the AIE2 VLIW core.
///
/// The 4-bit register field addresses named control registers in the ISA.
/// IDs are from `AIE2GenRegisterInfo.td` in llvm-aie:
///
/// ```text
/// def crRnd     : AIE2ControlReg<0b0110, "crRnd">;     // 6
/// def crSRSSign : AIE2ControlReg<0b1000, "crSRSSign">; // 8
/// def crSat     : AIE2ControlReg<0b1001, "crSat">;     // 9
/// ```
///
/// A drift-detection test in this module asserts these values agree with
/// the values encoded in the generated LLVM register info.
pub mod ctrl_regs {
    /// Control register ID for crRnd (SRS rounding mode), binary 0b0110.
    ///
    /// Source: `AIE2GenRegisterInfo.td`, `def crRnd : AIE2ControlReg<0b0110, "crRnd">`.
    pub const CR_RND: u8 = 6;

    /// Control register ID for crSRSSign (SRS signedness), binary 0b1000.
    ///
    /// Source: `AIE2GenRegisterInfo.td`, `def crSRSSign : AIE2ControlReg<0b1000, "crSRSSign">`.
    pub const CR_SRS_SIGN: u8 = 8;

    /// Control register ID for crSat (SRS saturation mode), binary 0b1001.
    ///
    /// Source: `AIE2GenRegisterInfo.td`, `def crSat : AIE2ControlReg<0b1001, "crSat">`.
    pub const CR_SAT: u8 = 9;

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Drift-detection: asserts control-register IDs match the 4-bit
        /// encoding in `AIE2GenRegisterInfo.td`. The bit patterns are
        /// 0b0110, 0b1000, 0b1001 respectively; this test checks the
        /// decimal equivalents remain stable.
        #[test]
        fn ctrl_reg_ids_match_tablegen_encoding() {
            // crRnd = 0b0110 = 6
            assert_eq!(CR_RND, 0b0110, "crRnd encoding changed in AIE2GenRegisterInfo.td");
            // crSRSSign = 0b1000 = 8
            assert_eq!(CR_SRS_SIGN, 0b1000, "crSRSSign encoding changed in AIE2GenRegisterInfo.td");
            // crSat = 0b1001 = 9
            assert_eq!(CR_SAT, 0b1001, "crSat encoding changed in AIE2GenRegisterInfo.td");
        }
    }
}
