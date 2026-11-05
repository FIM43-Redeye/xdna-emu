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
    //! Generated from AM025 by `build.rs` (`gen_dma_queue_offsets`): the compute-tile channel
    //! Ctrl and Start_Queue offsets, the memtile and shim channel-0 queue offsets, strides and
    //! channel counts.
    include!(concat!(env!("OUT_DIR"), "/gen_dma_queue.rs"));

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::regdb::RegisterDb;
        use crate::toolchain_paths::ToolchainPaths;

        fn am025() -> RegisterDb {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let mlir_aie = ToolchainPaths::require_mlir_aie(&root).expect("mlir-aie checkout");
            RegisterDb::from_file(&mlir_aie.join("lib/Dialect/AIE/Util/aie_registers_aie2.json"))
                .expect("AM025 register database")
        }

        /// Drift detection: every generated offset, stride and count equals what the AM025
        /// register database says, read here independently of the build script.
        #[test]
        fn dma_queue_constants_match_am025() {
            let db = am025();
            let at = |module: &str, name: &str| {
                db.module(module)
                    .and_then(|m| m.register(name))
                    .unwrap_or_else(|| panic!("{module}.{name}"))
                    .offset
            };
            for (n, (ctrl, queue)) in [
                (COMPUTE_DMA_S2MM_0_CTRL, COMPUTE_DMA_S2MM_0_START_QUEUE),
                (COMPUTE_DMA_S2MM_1_CTRL, COMPUTE_DMA_S2MM_1_START_QUEUE),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(ctrl, at("memory", &format!("DMA_S2MM_{n}_Ctrl")));
                assert_eq!(queue, at("memory", &format!("DMA_S2MM_{n}_Start_Queue")));
            }
            for (n, (ctrl, queue)) in [
                (COMPUTE_DMA_MM2S_0_CTRL, COMPUTE_DMA_MM2S_0_START_QUEUE),
                (COMPUTE_DMA_MM2S_1_CTRL, COMPUTE_DMA_MM2S_1_START_QUEUE),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(ctrl, at("memory", &format!("DMA_MM2S_{n}_Ctrl")));
                assert_eq!(queue, at("memory", &format!("DMA_MM2S_{n}_Start_Queue")));
            }
            assert_eq!(MEMTILE_DMA_S2MM_0_START_QUEUE, at("memory_tile", "DMA_S2MM_0_Start_Queue"));
            assert_eq!(MEMTILE_DMA_MM2S_0_START_QUEUE, at("memory_tile", "DMA_MM2S_0_Start_Queue"));
            assert_eq!(SHIM_DMA_S2MM_0_TASK_QUEUE, at("shim", "DMA_S2MM_0_Task_Queue"));
            assert_eq!(SHIM_DMA_MM2S_0_TASK_QUEUE, at("shim", "DMA_MM2S_0_Task_Queue"));
            // Last channel of each memtile direction sits on the generated stride and count.
            let last = u32::from(MEMTILE_CHANNELS_PER_DIR) - 1;
            assert_eq!(
                MEMTILE_DMA_S2MM_0_START_QUEUE + last * MEMTILE_CHANNEL_STRIDE,
                at("memory_tile", &format!("DMA_S2MM_{last}_Start_Queue"))
            );
            assert_eq!(
                MEMTILE_DMA_MM2S_0_START_QUEUE + last * MEMTILE_CHANNEL_STRIDE,
                at("memory_tile", &format!("DMA_MM2S_{last}_Start_Queue"))
            );
            assert!(db
                .module("memory_tile")
                .unwrap()
                .register(&format!("DMA_S2MM_{}_Ctrl", last + 1))
                .is_none());
            let last = u32::from(SHIM_CHANNELS_PER_DIR) - 1;
            assert_eq!(
                SHIM_DMA_S2MM_0_TASK_QUEUE + last * SHIM_CHANNEL_STRIDE,
                at("shim", &format!("DMA_S2MM_{last}_Task_Queue"))
            );
            assert!(db
                .module("shim")
                .unwrap()
                .register(&format!("DMA_S2MM_{}_Ctrl", last + 1))
                .is_none());
            // The compute, memtile and shim slots are all Ctrl + queue, 8 bytes.
            assert_eq!(CHANNEL_STRIDE, 0x08);
            assert_eq!(MEMTILE_CHANNEL_STRIDE, CHANNEL_STRIDE);
            assert_eq!(SHIM_CHANNEL_STRIDE, CHANNEL_STRIDE);
            assert_eq!(COMPUTE_DMA_S2MM_0_START_QUEUE - COMPUTE_DMA_S2MM_0_CTRL, START_QUEUE_OFFSET_IN_SLOT);
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

    /// Control register ID for crF2FMask (fp32 -> bf16 flag mask), binary 0b0001.
    ///
    /// Source: `AIE2GenRegisterInfo.td`, `def crF2FMask : AIE2ControlReg<0b0001, "crF2FMask">`.
    pub const CR_F2F_MASK: u8 = 1;

    /// Control register ID for crF2IMask (float -> int flag mask), binary 0b0010.
    ///
    /// Source: `AIE2GenRegisterInfo.td`, `def crF2IMask : AIE2ControlReg<0b0010, "crF2IMask">`.
    pub const CR_F2I_MASK: u8 = 2;

    /// Control register ID for crFPMask (float MAC / adder flag mask), binary 0b0011.
    ///
    /// Source: `AIE2GenRegisterInfo.td`, `def crFPMask : AIE2ControlReg<0b0011, "crFPMask">`.
    pub const CR_FP_MASK: u8 = 3;

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Drift detection: the IDs equal the `HWEncoding` TableGen gives the registers.
        #[test]
        fn ctrl_reg_ids_match_tablegen_encoding() {
            let model = crate::aie2::isa::load_from_generated().register_model;
            let encoding =
                |name: &str| model.registers.get(name).unwrap_or_else(|| panic!("no {name}")).hw_encoding;
            assert_eq!(u16::from(CR_RND), encoding("crRnd"));
            assert_eq!(u16::from(CR_SRS_SIGN), encoding("crSRSSign"));
            assert_eq!(u16::from(CR_SAT), encoding("crSat"));
            assert_eq!(u16::from(CR_F2F_MASK), encoding("crF2FMask"));
            assert_eq!(u16::from(CR_F2I_MASK), encoding("crF2IMask"));
            assert_eq!(u16::from(CR_FP_MASK), encoding("crFPMask"));
        }

        /// Control and status register widths come from AM025 (`Core_CR` / `Core_SR` fields,
        /// `Core_S0..3`); the plain registers agree with the widths TableGen gives.
        #[test]
        fn core_register_bits_follow_am025() {
            use super::super::core_register_bits as bits;
            assert_eq!(bits("s0"), Some(6));
            assert_eq!(bits("s3"), Some(6));
            assert_eq!(bits("crSat"), Some(2));
            assert_eq!(bits("crRnd"), Some(4));
            for run in ["crFPMask", "crF2IMask", "crF2FMask", "srFPFlags", "srF2IFlags", "srF2FFlags"] {
                assert_eq!(bits(run), Some(5), "{run}");
            }
            assert_eq!(bits("srSS0"), Some(2));
            for flag in [
                "crSRSSign",
                "crUPSSign",
                "crPackSign",
                "crUnpackSign",
                "crVaddSign",
                "crSCDEn",
                "crMCDEn",
                "srCarry",
                "srMS0",
                "srSRS_of",
                "srUPS_of",
                "srCompr_uf",
                "srSparse_of",
            ] {
                assert_eq!(bits(flag), Some(1), "{flag}");
            }
            assert_eq!(bits("r9"), Some(32));
            assert_eq!(bits("p3"), Some(20));
            assert_eq!(bits("lr"), Some(20));
            assert_eq!(bits("nonesuch"), None);
            // TableGen's class widths and AM025's register widths agree wherever both speak.
            let model = crate::aie2::isa::load_from_generated().register_model;
            for name in model.registers.keys() {
                if let (Some(am025), Some(tablegen)) = (bits(name), model.register_width_bits(name)) {
                    if name.starts_with("cr")
                        || name.starts_with("sr")
                        || name.len() == 2 && name.starts_with('s')
                    {
                        continue; // narrower than their 32-bit scalar class by design
                    }
                    assert_eq!(u16::from(am025), tablegen, "{name}");
                }
            }
        }

        /// The watchpoint layouts equal the AM025 field positions.
        #[test]
        fn watchpoint_layouts_follow_am025() {
            use super::super::watchpoint::{COMPUTE, MEMTILE};
            assert_eq!(COMPUTE.slot_offsets, &[0x14100, 0x14104]);
            assert_eq!(COMPUTE.addr_mask, 0xFFF0);
            assert_eq!(
                (COMPUTE.read_bit, COMPUTE.write_bit, COMPUTE.axi_bit, COMPUTE.dma_bit),
                (31, 30, 29, 28)
            );
            assert_eq!(
                (COMPUTE.east_bit, COMPUTE.north_bit, COMPUTE.west_bit, COMPUTE.south_bit),
                (27, Some(26), 25, Some(24))
            );
            assert_eq!((COMPUTE.strobes_lsb, COMPUTE.strobes_width), (20, 4));
            assert_eq!(MEMTILE.slot_offsets, &[0x94100, 0x94104, 0x94108, 0x9410C]);
            assert_eq!(MEMTILE.addr_mask, 0x7FFF0);
            assert_eq!(
                (MEMTILE.read_bit, MEMTILE.write_bit, MEMTILE.axi_bit, MEMTILE.dma_bit),
                (29, 28, 27, 26)
            );
            assert_eq!(
                (MEMTILE.east_bit, MEMTILE.north_bit, MEMTILE.west_bit, MEMTILE.south_bit),
                (25, None, 24, None)
            );
        }
    }
}
