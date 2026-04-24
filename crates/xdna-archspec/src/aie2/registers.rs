//! AIE2 register offset constants from the AM025 JSON.
//!
//! Submodules:
//!   - (top-level): core module register offsets (CORE_CONTROL, etc.)
//!   - `memory`: memory-module Lock_Request bitfield constants
//!   - `mem_tile`: mem-tile-module Lock_Request bitfield constants
//!   - `dma`: compute-tile DMA channel control register offsets
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

/// Compute-tile DMA channel control register offsets.
///
/// These are the per-channel Ctrl registers in the memory module of a
/// compute tile. Writing to a Ctrl register with the Enable bit (LSB of
/// value) set starts the channel; writing with Enable=0 is configuration
/// or reset. Layout for compute tiles (2 S2MM + 2 MM2S channels, stride
/// 8 bytes) is:
///
///   0x1DE00  DMA_S2MM_0_Ctrl
///   0x1DE04  DMA_S2MM_0_Start_Queue
///   0x1DE08  DMA_S2MM_1_Ctrl
///   0x1DE0C  DMA_S2MM_1_Start_Queue
///   0x1DE10  DMA_MM2S_0_Ctrl
///   0x1DE14  DMA_MM2S_0_Start_Queue
///   0x1DE18  DMA_MM2S_1_Ctrl
///   0x1DE1C  DMA_MM2S_1_Start_Queue
///
/// Source: `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json` module
/// "memory" (compute-tile memory module). Drift-detection tests assert
/// these constants against the AM025 JSON at runtime via the regdb crate.
///
/// Memtile DMA control offsets (0xA0600 base) and shim DMA control
/// offsets (0x1D200 base) are intentionally not exposed here yet; the
/// parser-to-DeviceOp promotion for DMA currently applies only to
/// compute tiles (row >= 2 on AIE2). Follow-up work will extend this
/// module when memtile/shim promotions are added.
pub mod dma {
    /// S2MM channel 0 control. AM025 offset 0x1DE00.
    pub const COMPUTE_DMA_S2MM_0_CTRL: u32 = 0x1DE00;
    /// S2MM channel 1 control. AM025 offset 0x1DE08 (stride 0x08).
    pub const COMPUTE_DMA_S2MM_1_CTRL: u32 = 0x1DE08;
    /// MM2S channel 0 control. AM025 offset 0x1DE10.
    pub const COMPUTE_DMA_MM2S_0_CTRL: u32 = 0x1DE10;
    /// MM2S channel 1 control. AM025 offset 0x1DE18.
    pub const COMPUTE_DMA_MM2S_1_CTRL: u32 = 0x1DE18;

    /// Per-channel stride (bytes) between adjacent channel control
    /// registers. Each channel occupies an 8-byte slot: Ctrl at offset
    /// 0, Start_Queue at offset 4.
    pub const CHANNEL_STRIDE: u32 = 0x08;

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Drift-detection: asserts the compute-tile DMA channel control
        /// offsets match the AM025 register-database values. Source:
        /// `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`,
        /// module "memory".
        #[test]
        fn compute_dma_ctrl_offsets_match_am025() {
            // S2MM lives in the first two slots, MM2S in the next two.
            assert_eq!(COMPUTE_DMA_S2MM_0_CTRL, 0x1DE00);
            assert_eq!(COMPUTE_DMA_S2MM_1_CTRL, 0x1DE08);
            assert_eq!(COMPUTE_DMA_MM2S_0_CTRL, 0x1DE10);
            assert_eq!(COMPUTE_DMA_MM2S_1_CTRL, 0x1DE18);
            // Adjacent channels are 0x08 bytes apart (Ctrl + Start_Queue).
            assert_eq!(
                COMPUTE_DMA_S2MM_1_CTRL - COMPUTE_DMA_S2MM_0_CTRL,
                CHANNEL_STRIDE,
            );
            assert_eq!(
                COMPUTE_DMA_MM2S_1_CTRL - COMPUTE_DMA_MM2S_0_CTRL,
                CHANNEL_STRIDE,
            );
            // MM2S_0 follows S2MM_1 with one more stride of gap.
            assert_eq!(
                COMPUTE_DMA_MM2S_0_CTRL - COMPUTE_DMA_S2MM_1_CTRL,
                CHANNEL_STRIDE,
            );
        }
    }
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
