//! Constants and construction parameters for AIE tiles.

/// Size of program memory (16 KB = 1024 x 128-bit instructions).
pub const PROGRAM_MEMORY_SIZE: usize = crate::arch::compute::PROGRAM_MEMORY_SIZE as usize;

/// Parameters for constructing a Tile with correct per-tile-type sizing.
///
/// Production code derives these from `ArchConfig` (which reads mlir-aie
/// device models). Convenience constructors also load from the device model
/// so there is a single source of truth.
#[derive(Debug, Clone)]
pub struct TileParams {
    /// Data memory size in bytes (0 for shim, 64K for compute, 512K for mem tile).
    pub data_memory_size: usize,
    /// Number of locks (16 for shim/compute, 64 for mem tile).
    pub num_locks: usize,
    /// Number of DMA buffer descriptors (16 for compute, 48 for mem tile).
    pub num_bds: usize,
    /// Total DMA channels (4 for compute, 12 for mem tile).
    pub num_channels: usize,
    /// S2MM (write) DMA channels (2 for compute/shim, 6 for mem tile).
    pub dma_s2mm_channels: usize,
    /// MM2S (read) DMA channels (2 for compute/shim, 6 for mem tile).
    pub dma_mm2s_channels: usize,
}

impl TileParams {
    /// NPU1/AIE2 compute tile params, from compile-time arch constants.
    pub fn compute() -> Self {
        use crate::arch;
        let ch = arch::compute::NUM_DMA_CHANNELS as usize;
        Self {
            data_memory_size: arch::compute::MEMORY_SIZE as usize,
            num_locks: arch::compute::NUM_LOCKS as usize,
            num_bds: arch::compute::NUM_BDS as usize,
            num_channels: ch * 2,
            dma_s2mm_channels: ch,
            dma_mm2s_channels: ch,
        }
    }

    /// NPU1/AIE2 memory tile params, from compile-time arch constants.
    pub fn mem_tile() -> Self {
        use crate::arch;
        let ch = arch::memtile::NUM_DMA_CHANNELS as usize;
        Self {
            data_memory_size: arch::memtile::MEMORY_SIZE as usize,
            num_locks: arch::memtile::NUM_LOCKS as usize,
            num_bds: arch::memtile::NUM_BDS as usize,
            num_channels: ch * 2,
            dma_s2mm_channels: ch,
            dma_mm2s_channels: ch,
        }
    }

    /// NPU1/AIE2 shim tile params, from compile-time arch constants.
    pub fn shim() -> Self {
        use crate::arch;
        let ch = arch::shim::NUM_DMA_CHANNELS as usize;
        Self {
            data_memory_size: 0,
            num_locks: arch::shim::NUM_LOCKS as usize,
            num_bds: arch::shim::NUM_BDS as usize,
            num_channels: ch * 2,
            dma_s2mm_channels: ch,
            dma_mm2s_channels: ch,
        }
    }
}
