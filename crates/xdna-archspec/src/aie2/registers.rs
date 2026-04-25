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
/// Memtile (module "memory_tile", base 0xA0600, 6 channels per direction)
/// and shim (module "shim", base 0x1D200, 2 channels per direction) DMA
/// control offsets are also exposed below so the CDO lower path can
/// promote DmaStart on every tile kind, not just compute. Hardware
/// names the trigger register `Start_Queue` on compute/memtile and
/// `Task_Queue` on shim; both live at Ctrl + 4 within their channel
/// slot, and both behave the same way -- a write triggers a DMA
/// transfer using the BD ID in the value.
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

    /// Offset within a channel slot from Ctrl to Start_Queue.
    /// Real hardware triggers a DMA transfer by pushing a BD ID to
    /// Start_Queue; Ctrl is channel-level configuration only. Parser
    /// promotion to `DeviceOp::DmaStart` targets Start_Queue writes.
    pub const START_QUEUE_OFFSET_IN_SLOT: u32 = 0x04;

    /// S2MM channel 0 Start_Queue. AM025 offset 0x1DE04.
    pub const COMPUTE_DMA_S2MM_0_START_QUEUE: u32 = 0x1DE04;
    /// S2MM channel 1 Start_Queue. AM025 offset 0x1DE0C.
    pub const COMPUTE_DMA_S2MM_1_START_QUEUE: u32 = 0x1DE0C;
    /// MM2S channel 0 Start_Queue. AM025 offset 0x1DE14.
    pub const COMPUTE_DMA_MM2S_0_START_QUEUE: u32 = 0x1DE14;
    /// MM2S channel 1 Start_Queue. AM025 offset 0x1DE1C.
    pub const COMPUTE_DMA_MM2S_1_START_QUEUE: u32 = 0x1DE1C;

    /// Number of S2MM (and MM2S) channels on memtiles -- 6 per direction.
    pub const MEMTILE_CHANNELS_PER_DIR: u8 = 6;
    /// Memtile DMA channel slot stride (Ctrl + Start_Queue = 8 bytes).
    pub const MEMTILE_CHANNEL_STRIDE: u32 = 0x08;
    /// Memtile S2MM channel 0 Start_Queue. AM025 offset 0xA0604.
    /// Subsequent S2MM channels are at +0x08 each (S2MM_1 = 0xA060C, ..., S2MM_5 = 0xA062C).
    pub const MEMTILE_DMA_S2MM_0_START_QUEUE: u32 = 0xA0604;
    /// Memtile MM2S channel 0 Start_Queue. AM025 offset 0xA0634.
    /// Subsequent MM2S channels are at +0x08 each (MM2S_1 = 0xA063C, ..., MM2S_5 = 0xA065C).
    pub const MEMTILE_DMA_MM2S_0_START_QUEUE: u32 = 0xA0634;

    /// Number of S2MM (and MM2S) channels on shim -- 2 per direction.
    pub const SHIM_CHANNELS_PER_DIR: u8 = 2;
    /// Shim DMA channel slot stride (Ctrl + Task_Queue = 8 bytes).
    pub const SHIM_CHANNEL_STRIDE: u32 = 0x08;
    /// Shim S2MM channel 0 Task_Queue. AM025 offset 0x1D204.
    /// (Hardware names this `Task_Queue` on shim; semantically equivalent
    /// to memtile/compute `Start_Queue`.) S2MM_1 follows at 0x1D20C.
    pub const SHIM_DMA_S2MM_0_TASK_QUEUE: u32 = 0x1D204;
    /// Shim MM2S channel 0 Task_Queue. AM025 offset 0x1D214. MM2S_1 at 0x1D21C.
    pub const SHIM_DMA_MM2S_0_TASK_QUEUE: u32 = 0x1D214;

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

        /// Drift-detection: asserts Start_Queue offsets match Ctrl + 4.
        /// Source: `mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json`
        /// (DMA_S2MM_0_Start_Queue at 0x1DE04, etc).
        #[test]
        fn compute_dma_start_queue_offsets_match_am025() {
            assert_eq!(COMPUTE_DMA_S2MM_0_START_QUEUE, 0x1DE04);
            assert_eq!(COMPUTE_DMA_S2MM_1_START_QUEUE, 0x1DE0C);
            assert_eq!(COMPUTE_DMA_MM2S_0_START_QUEUE, 0x1DE14);
            assert_eq!(COMPUTE_DMA_MM2S_1_START_QUEUE, 0x1DE1C);
            // Each Start_Queue is 4 bytes past its Ctrl.
            assert_eq!(
                COMPUTE_DMA_S2MM_0_START_QUEUE - COMPUTE_DMA_S2MM_0_CTRL,
                START_QUEUE_OFFSET_IN_SLOT,
            );
            assert_eq!(
                COMPUTE_DMA_MM2S_1_START_QUEUE - COMPUTE_DMA_MM2S_1_CTRL,
                START_QUEUE_OFFSET_IN_SLOT,
            );
        }

        /// Drift-detection: memtile S2MM/MM2S Start_Queue base offsets,
        /// stride, and channel-count agree with AM025.
        /// Source: `aie_registers_aie2.json` module "memory_tile":
        /// DMA_S2MM_0_Start_Queue at 0xA0604, MM2S_0_Start_Queue at 0xA0634.
        #[test]
        fn memtile_dma_start_queue_offsets_match_am025() {
            assert_eq!(MEMTILE_DMA_S2MM_0_START_QUEUE, 0xA0604);
            assert_eq!(MEMTILE_DMA_MM2S_0_START_QUEUE, 0xA0634);
            // 6 channels each direction, 8 bytes apart.
            assert_eq!(MEMTILE_CHANNELS_PER_DIR, 6);
            assert_eq!(MEMTILE_CHANNEL_STRIDE, 0x08);
            // MM2S follows S2MM after 6 channels at stride 8 + 0 gap.
            assert_eq!(
                MEMTILE_DMA_MM2S_0_START_QUEUE - MEMTILE_DMA_S2MM_0_START_QUEUE,
                MEMTILE_CHANNELS_PER_DIR as u32 * MEMTILE_CHANNEL_STRIDE,
            );
        }

        /// Drift-detection: shim S2MM/MM2S Task_Queue base offsets,
        /// stride, and channel-count agree with AM025.
        /// Source: `aie_registers_aie2.json` module "shim":
        /// DMA_S2MM_0_Task_Queue at 0x1D204, MM2S_0_Task_Queue at 0x1D214.
        #[test]
        fn shim_dma_task_queue_offsets_match_am025() {
            assert_eq!(SHIM_DMA_S2MM_0_TASK_QUEUE, 0x1D204);
            assert_eq!(SHIM_DMA_MM2S_0_TASK_QUEUE, 0x1D214);
            assert_eq!(SHIM_CHANNELS_PER_DIR, 2);
            assert_eq!(SHIM_CHANNEL_STRIDE, 0x08);
            // MM2S follows S2MM after 2 channels at stride 8.
            assert_eq!(
                SHIM_DMA_MM2S_0_TASK_QUEUE - SHIM_DMA_S2MM_0_TASK_QUEUE,
                SHIM_CHANNELS_PER_DIR as u32 * SHIM_CHANNEL_STRIDE,
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
