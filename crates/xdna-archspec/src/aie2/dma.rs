//! AIE2 DMA model implementation.
//!
//! Covers NPU1 (Phoenix), NPU4 / NPU5 / NPU6 (Strix / Strix Halo /
//! Krackan). All AIE2-family devices share the same DMA feature set:
//!
//! - 8-deep task queue per channel.
//! - Out-of-order S2MM packet completion.
//! - Sparsity compression (MM2S) / decompression (S2MM).
//! - Per-BD iteration (stepsize + wrap + current counter).
//! - Independent acquire / release lock IDs.
//! - No interleave mode, no LockDesc_2 double-buffer.
//! - 128-bit memory bus (4 words/cycle).
//! - 3D addressing for compute / shim tiles; 4D for memtiles.
//!
//! Timing constants come from the build-time-generated
//! `xdna_archspec::aie2::timing` submodule.

use crate::dma::{DmaModel, DmaTimingConfig};
use crate::types::TileKind;

/// AIE2 DMA model.
///
/// Zero-sized: a single `AIE2_DMA_MODEL` static instance serves every
/// tile in every NPU device. `ArchModel::dma_model()` returns a
/// `&'static dyn DmaModel` pointing at this singleton.
#[derive(Debug, Clone, Copy)]
pub struct Aie2DmaModel;

/// The single `Aie2DmaModel` instance used across every AIE2-family
/// `DmaEngine`. Reference via `ArchModel::dma_model()`.
pub static AIE2_DMA_MODEL: Aie2DmaModel = Aie2DmaModel;

impl DmaModel for Aie2DmaModel {
    fn supports_task_queue(&self) -> bool {
        true
    }
    fn supports_ooo_mode(&self) -> bool {
        true
    }
    fn supports_compression(&self) -> bool {
        true
    }
    fn supports_bd_iteration(&self) -> bool {
        true
    }
    fn supports_independent_lock_ids(&self) -> bool {
        true
    }
    fn supports_interleave_mode(&self) -> bool {
        false
    }
    fn supports_double_buffer(&self) -> bool {
        false
    }

    fn max_tensor_dims(&self, tile: TileKind) -> u8 {
        match tile {
            TileKind::Compute | TileKind::ShimNoc | TileKind::ShimPl => 3,
            TileKind::Mem => 4,
        }
    }

    fn timing_config(&self) -> DmaTimingConfig {
        use crate::aie2::timing;
        DmaTimingConfig {
            bd_setup_cycles: timing::DMA_BD_SETUP_CYCLES,
            channel_start_cycles: timing::DMA_CHANNEL_START_CYCLES,
            words_per_cycle: timing::DMA_WORDS_PER_CYCLE,
            memory_latency_cycles: timing::DMA_MEMORY_LATENCY_CYCLES,
            lock_acquire_cycles: timing::DMA_LOCK_ACQUIRE_CYCLES,
            lock_release_cycles: timing::DMA_LOCK_RELEASE_CYCLES,
            bd_chain_cycles: timing::DMA_BD_CHAIN_CYCLES,
            host_memory_latency_cycles: timing::DMA_HOST_MEMORY_LATENCY_CYCLES,
            shim_ddr_cold_start_cycles: timing::DMA_SHIM_DDR_COLD_START_CYCLES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_task_queue_true() {
        assert!(AIE2_DMA_MODEL.supports_task_queue());
    }

    #[test]
    fn supports_ooo_mode_true() {
        assert!(AIE2_DMA_MODEL.supports_ooo_mode());
    }

    #[test]
    fn supports_compression_true() {
        assert!(AIE2_DMA_MODEL.supports_compression());
    }

    #[test]
    fn supports_bd_iteration_true() {
        assert!(AIE2_DMA_MODEL.supports_bd_iteration());
    }

    #[test]
    fn supports_independent_lock_ids_true() {
        assert!(AIE2_DMA_MODEL.supports_independent_lock_ids());
    }

    #[test]
    fn supports_interleave_mode_false() {
        assert!(!AIE2_DMA_MODEL.supports_interleave_mode());
    }

    #[test]
    fn supports_double_buffer_false() {
        assert!(!AIE2_DMA_MODEL.supports_double_buffer());
    }

    #[test]
    fn max_tensor_dims_compute_is_3() {
        assert_eq!(AIE2_DMA_MODEL.max_tensor_dims(TileKind::Compute), 3);
    }

    #[test]
    fn max_tensor_dims_shim_is_3() {
        assert_eq!(AIE2_DMA_MODEL.max_tensor_dims(TileKind::ShimNoc), 3);
        assert_eq!(AIE2_DMA_MODEL.max_tensor_dims(TileKind::ShimPl), 3);
    }

    #[test]
    fn max_tensor_dims_mem_is_4() {
        assert_eq!(AIE2_DMA_MODEL.max_tensor_dims(TileKind::Mem), 4);
    }

    #[test]
    fn timing_config_matches_aie2_constants() {
        use crate::aie2::timing;
        let cfg = AIE2_DMA_MODEL.timing_config();
        assert_eq!(cfg.bd_setup_cycles, timing::DMA_BD_SETUP_CYCLES);
        assert_eq!(cfg.channel_start_cycles, timing::DMA_CHANNEL_START_CYCLES);
        assert_eq!(cfg.words_per_cycle, timing::DMA_WORDS_PER_CYCLE);
        assert_eq!(cfg.memory_latency_cycles, timing::DMA_MEMORY_LATENCY_CYCLES);
        assert_eq!(cfg.lock_acquire_cycles, timing::DMA_LOCK_ACQUIRE_CYCLES);
        assert_eq!(cfg.lock_release_cycles, timing::DMA_LOCK_RELEASE_CYCLES);
        assert_eq!(cfg.bd_chain_cycles, timing::DMA_BD_CHAIN_CYCLES);
        assert_eq!(cfg.host_memory_latency_cycles, timing::DMA_HOST_MEMORY_LATENCY_CYCLES);
        assert_eq!(cfg.shim_ddr_cold_start_cycles, timing::DMA_SHIM_DDR_COLD_START_CYCLES);
    }

    #[test]
    fn static_model_is_usable_as_dyn() {
        // Verify the static can be referenced as &'static dyn DmaModel.
        let m: &'static dyn DmaModel = &AIE2_DMA_MODEL;
        assert!(m.supports_task_queue());
    }
}
