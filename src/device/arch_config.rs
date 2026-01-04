//! Architecture configuration trait and implementations.
//!
//! This module provides a trait-based abstraction for NPU architecture parameters,
//! enabling support for multiple NPU variants (NPU1, NPU2, NPU3, NPU4) without
//! hardcoded values scattered throughout the codebase.
//!
//! # Design Philosophy
//!
//! Instead of checking `row == 0` for shim tiles everywhere, code should call
//! `arch.is_shim_tile(col, row)`. This allows different architectures to have
//! different tile layouts without code changes.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::device::arch_config::{ArchConfig, Aie2Config};
//! use std::sync::Arc;
//!
//! let arch: Arc<dyn ArchConfig> = Arc::new(Aie2Config);
//! assert_eq!(arch.columns(), 5);
//! assert_eq!(arch.rows(), 6);
//! assert!(arch.is_shim_tile(0, 0));
//! assert!(arch.is_mem_tile(0, 1));
//! assert!(arch.is_compute_tile(0, 2));
//! ```

use super::aie2_spec;
use super::tile::TileType;

/// Architecture configuration trait for NPU variants.
///
/// Implementations of this trait encapsulate all architecture-specific parameters:
/// - Array dimensions (columns, rows)
/// - Tile type classification (which coordinates are shim/memtile/compute)
/// - Memory sizes per tile type
/// - DMA configuration (channels, buffer descriptors)
/// - Stream switch port layouts
///
/// This enables the emulator to support multiple NPU architectures without
/// hardcoded checks like `row == 0` scattered throughout the code.
pub trait ArchConfig: Send + Sync + std::fmt::Debug {
    // ========================================================================
    // Array Dimensions
    // ========================================================================

    /// Get the number of columns in the tile array.
    fn columns(&self) -> u8;

    /// Get the number of rows in the tile array.
    fn rows(&self) -> u8;

    // ========================================================================
    // Tile Type Classification
    // ========================================================================

    /// Get the tile type at the given coordinates.
    fn tile_type(&self, col: u8, row: u8) -> TileType;

    /// Check if a tile position is valid.
    fn is_valid_tile(&self, col: u8, row: u8) -> bool {
        col < self.columns() && row < self.rows()
    }

    /// Check if a tile is a shim tile (DDR interface).
    fn is_shim_tile(&self, col: u8, row: u8) -> bool {
        self.tile_type(col, row) == TileType::Shim
    }

    /// Check if a tile is a memory tile (large shared memory).
    fn is_mem_tile(&self, col: u8, row: u8) -> bool {
        self.tile_type(col, row) == TileType::MemTile
    }

    /// Check if a tile is a compute tile (has AIE core).
    fn is_compute_tile(&self, col: u8, row: u8) -> bool {
        self.tile_type(col, row) == TileType::Compute
    }

    // ========================================================================
    // Memory Configuration
    // ========================================================================

    /// Get the data memory size for a tile type.
    fn data_memory_size(&self, tile_type: TileType) -> usize;

    /// Get the program memory size for a tile type.
    /// Only compute tiles have program memory; returns 0 for others.
    fn program_memory_size(&self, tile_type: TileType) -> usize;

    /// Get the number of locks for a tile type.
    fn lock_count(&self, tile_type: TileType) -> usize;

    // ========================================================================
    // DMA Configuration
    // ========================================================================

    /// Get the number of S2MM (stream-to-memory) DMA channels.
    fn dma_s2mm_channels(&self, tile_type: TileType) -> usize;

    /// Get the number of MM2S (memory-to-stream) DMA channels.
    fn dma_mm2s_channels(&self, tile_type: TileType) -> usize;

    /// Get the total number of DMA channels.
    fn dma_total_channels(&self, tile_type: TileType) -> usize {
        self.dma_s2mm_channels(tile_type) + self.dma_mm2s_channels(tile_type)
    }

    /// Get the number of DMA buffer descriptors.
    fn dma_bd_count(&self, tile_type: TileType) -> usize;

    // ========================================================================
    // Stream Switch Port Layouts
    // ========================================================================

    /// Get the master port layout for a tile type.
    /// Returns an array of port type identifiers.
    fn master_ports(&self, tile_type: TileType) -> &'static [u8];

    /// Get the slave port layout for a tile type.
    fn slave_ports(&self, tile_type: TileType) -> &'static [u8];

    /// Get the range of north-facing master ports (start, end inclusive).
    fn north_master_range(&self, tile_type: TileType) -> (u8, u8);

    /// Get the range of south-facing master ports (start, end inclusive).
    fn south_master_range(&self, tile_type: TileType) -> (u8, u8);

    /// Get the range of north-facing slave ports (start, end inclusive).
    fn north_slave_range(&self, tile_type: TileType) -> (u8, u8);

    /// Get the range of south-facing slave ports (start, end inclusive).
    fn south_slave_range(&self, tile_type: TileType) -> (u8, u8);

    // ========================================================================
    // Architecture Name
    // ========================================================================

    /// Get the architecture name for display.
    fn name(&self) -> &'static str;
}

// ============================================================================
// AIE2 Configuration (NPU1 - Phoenix/HawkPoint)
// ============================================================================

/// AIE2 architecture configuration.
///
/// This is the primary architecture for Phoenix and HawkPoint NPUs (NPU1).
/// The tile array is 5 columns x 6 rows:
/// - Row 0: Shim tiles (DDR interface via NoC)
/// - Row 1: Memory tiles (512KB each)
/// - Rows 2-5: Compute tiles (64KB data + 16KB program memory each)
#[derive(Debug, Clone, Copy, Default)]
pub struct Aie2Config;

impl ArchConfig for Aie2Config {
    fn columns(&self) -> u8 {
        5
    }

    fn rows(&self) -> u8 {
        6
    }

    fn tile_type(&self, _col: u8, row: u8) -> TileType {
        match row {
            0 => TileType::Shim,
            1 => TileType::MemTile,
            _ => TileType::Compute,
        }
    }

    fn data_memory_size(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_DATA_MEMORY_SIZE,
            TileType::Compute => aie2_spec::COMPUTE_TILE_DATA_MEMORY_SIZE,
        }
    }

    fn program_memory_size(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Compute => aie2_spec::PROGRAM_MEMORY_SIZE,
            _ => 0,
        }
    }

    fn lock_count(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_NUM_LOCKS,
            TileType::Compute => aie2_spec::COMPUTE_TILE_NUM_LOCKS,
        }
    }

    fn dma_s2mm_channels(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0, // Shim DMA handled via MemTile
            TileType::MemTile => aie2_spec::MEM_TILE_S2MM_CHANNELS,
            TileType::Compute => aie2_spec::COMPUTE_TILE_S2MM_CHANNELS,
        }
    }

    fn dma_mm2s_channels(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_MM2S_CHANNELS,
            TileType::Compute => aie2_spec::COMPUTE_TILE_MM2S_CHANNELS,
        }
    }

    fn dma_bd_count(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEMTILE_NUM_DMA_BUFFER_DESCRIPTORS,
            TileType::Compute => aie2_spec::NUM_DMA_BUFFER_DESCRIPTORS,
        }
    }

    fn master_ports(&self, tile_type: TileType) -> &'static [u8] {
        match tile_type {
            TileType::Shim => aie2_spec::SHIM_MASTER_PORTS,
            TileType::MemTile => aie2_spec::MEMTILE_MASTER_PORTS,
            TileType::Compute => aie2_spec::COMPUTE_MASTER_PORTS,
        }
    }

    fn slave_ports(&self, tile_type: TileType) -> &'static [u8] {
        match tile_type {
            TileType::Shim => aie2_spec::SHIM_SLAVE_PORTS,
            TileType::MemTile => aie2_spec::MEMTILE_SLAVE_PORTS,
            TileType::Compute => aie2_spec::COMPUTE_SLAVE_PORTS,
        }
    }

    fn north_master_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            TileType::Shim => (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END),
            TileType::MemTile => (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END),
            TileType::Compute => (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END),
        }
    }

    fn south_master_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            // Shim tiles don't have south masters (south is NoC/DDR)
            TileType::Shim => (0, 0),
            TileType::MemTile => (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END),
            TileType::Compute => (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END),
        }
    }

    fn north_slave_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            TileType::Shim => (shim::NORTH_SLAVE_START, shim::NORTH_SLAVE_END),
            TileType::MemTile => (mem_tile::NORTH_SLAVE_START, mem_tile::NORTH_SLAVE_END),
            TileType::Compute => (compute::NORTH_SLAVE_START, compute::NORTH_SLAVE_END),
        }
    }

    fn south_slave_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            // Shim tiles don't have south slaves (south is NoC/DDR input)
            TileType::Shim => (0, 0),
            TileType::MemTile => (mem_tile::SOUTH_SLAVE_START, mem_tile::SOUTH_SLAVE_END),
            TileType::Compute => (compute::SOUTH_SLAVE_START, compute::SOUTH_SLAVE_END),
        }
    }

    fn name(&self) -> &'static str {
        "AIE2 (NPU1 - Phoenix/HawkPoint)"
    }
}

// ============================================================================
// AIE2P Configuration (NPU2+ - Strix/Krackan) - Placeholder
// ============================================================================

/// AIE2P architecture configuration (placeholder).
///
/// This will be the architecture for Strix and Krackan NPUs (NPU2, NPU3, NPU4).
/// For now, it uses the same parameters as AIE2 until we have detailed specs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Aie2pConfig;

impl ArchConfig for Aie2pConfig {
    fn columns(&self) -> u8 {
        5 // Same as AIE2 for now; NPU3 (Strix Halo) may have 9 columns
    }

    fn rows(&self) -> u8 {
        6
    }

    fn tile_type(&self, _col: u8, row: u8) -> TileType {
        match row {
            0 => TileType::Shim,
            1 => TileType::MemTile,
            _ => TileType::Compute,
        }
    }

    fn data_memory_size(&self, tile_type: TileType) -> usize {
        // AIE2P may have different memory sizes; using AIE2 values as placeholder
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_DATA_MEMORY_SIZE,
            TileType::Compute => aie2_spec::COMPUTE_TILE_DATA_MEMORY_SIZE,
        }
    }

    fn program_memory_size(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Compute => aie2_spec::PROGRAM_MEMORY_SIZE,
            _ => 0,
        }
    }

    fn lock_count(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_NUM_LOCKS,
            TileType::Compute => aie2_spec::COMPUTE_TILE_NUM_LOCKS,
        }
    }

    fn dma_s2mm_channels(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_S2MM_CHANNELS,
            TileType::Compute => aie2_spec::COMPUTE_TILE_S2MM_CHANNELS,
        }
    }

    fn dma_mm2s_channels(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEM_TILE_MM2S_CHANNELS,
            TileType::Compute => aie2_spec::COMPUTE_TILE_MM2S_CHANNELS,
        }
    }

    fn dma_bd_count(&self, tile_type: TileType) -> usize {
        match tile_type {
            TileType::Shim => 0,
            TileType::MemTile => aie2_spec::MEMTILE_NUM_DMA_BUFFER_DESCRIPTORS,
            TileType::Compute => aie2_spec::NUM_DMA_BUFFER_DESCRIPTORS,
        }
    }

    fn master_ports(&self, tile_type: TileType) -> &'static [u8] {
        // Using AIE2 port layouts as placeholder
        match tile_type {
            TileType::Shim => aie2_spec::SHIM_MASTER_PORTS,
            TileType::MemTile => aie2_spec::MEMTILE_MASTER_PORTS,
            TileType::Compute => aie2_spec::COMPUTE_MASTER_PORTS,
        }
    }

    fn slave_ports(&self, tile_type: TileType) -> &'static [u8] {
        match tile_type {
            TileType::Shim => aie2_spec::SHIM_SLAVE_PORTS,
            TileType::MemTile => aie2_spec::MEMTILE_SLAVE_PORTS,
            TileType::Compute => aie2_spec::COMPUTE_SLAVE_PORTS,
        }
    }

    fn north_master_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            TileType::Shim => (shim::NORTH_MASTER_START, shim::NORTH_MASTER_END),
            TileType::MemTile => (mem_tile::NORTH_MASTER_START, mem_tile::NORTH_MASTER_END),
            TileType::Compute => (compute::NORTH_MASTER_START, compute::NORTH_MASTER_END),
        }
    }

    fn south_master_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            TileType::Shim => (0, 0),
            TileType::MemTile => (mem_tile::SOUTH_MASTER_START, mem_tile::SOUTH_MASTER_END),
            TileType::Compute => (compute::SOUTH_MASTER_START, compute::SOUTH_MASTER_END),
        }
    }

    fn north_slave_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            TileType::Shim => (shim::NORTH_SLAVE_START, shim::NORTH_SLAVE_END),
            TileType::MemTile => (mem_tile::NORTH_SLAVE_START, mem_tile::NORTH_SLAVE_END),
            TileType::Compute => (compute::NORTH_SLAVE_START, compute::NORTH_SLAVE_END),
        }
    }

    fn south_slave_range(&self, tile_type: TileType) -> (u8, u8) {
        use aie2_spec::stream_switch::*;
        match tile_type {
            TileType::Shim => (0, 0),
            TileType::MemTile => (mem_tile::SOUTH_SLAVE_START, mem_tile::SOUTH_SLAVE_END),
            TileType::Compute => (compute::SOUTH_SLAVE_START, compute::SOUTH_SLAVE_END),
        }
    }

    fn name(&self) -> &'static str {
        "AIE2P (NPU2+ - Strix/Krackan)"
    }
}

// ============================================================================
// Default Architecture
// ============================================================================

/// Get the default architecture configuration (AIE2/NPU1).
pub fn default_arch() -> std::sync::Arc<dyn ArchConfig> {
    std::sync::Arc::new(Aie2Config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aie2_dimensions() {
        let arch = Aie2Config;
        assert_eq!(arch.columns(), 5);
        assert_eq!(arch.rows(), 6);
    }

    #[test]
    fn test_aie2_tile_classification() {
        let arch = Aie2Config;

        // Shim tiles (row 0)
        for col in 0..5 {
            assert!(arch.is_shim_tile(col, 0), "({}, 0) should be shim", col);
            assert_eq!(arch.tile_type(col, 0), TileType::Shim);
        }

        // Mem tiles (row 1)
        for col in 0..5 {
            assert!(arch.is_mem_tile(col, 1), "({}, 1) should be memtile", col);
            assert_eq!(arch.tile_type(col, 1), TileType::MemTile);
        }

        // Compute tiles (rows 2-5)
        for col in 0..5 {
            for row in 2..6 {
                assert!(
                    arch.is_compute_tile(col, row),
                    "({}, {}) should be compute",
                    col,
                    row
                );
                assert_eq!(arch.tile_type(col, row), TileType::Compute);
            }
        }
    }

    #[test]
    fn test_aie2_memory_sizes() {
        let arch = Aie2Config;

        assert_eq!(arch.data_memory_size(TileType::Shim), 0);
        assert_eq!(arch.data_memory_size(TileType::MemTile), 512 * 1024);
        assert_eq!(arch.data_memory_size(TileType::Compute), 64 * 1024);

        assert_eq!(arch.program_memory_size(TileType::Shim), 0);
        assert_eq!(arch.program_memory_size(TileType::MemTile), 0);
        assert_eq!(arch.program_memory_size(TileType::Compute), 16 * 1024);
    }

    #[test]
    fn test_aie2_lock_counts() {
        let arch = Aie2Config;

        assert_eq!(arch.lock_count(TileType::Shim), 0);
        assert_eq!(arch.lock_count(TileType::MemTile), 64);
        assert_eq!(arch.lock_count(TileType::Compute), 16);
    }

    #[test]
    fn test_aie2_dma_config() {
        let arch = Aie2Config;

        // Compute tile: 2 S2MM + 2 MM2S = 4 channels
        assert_eq!(arch.dma_s2mm_channels(TileType::Compute), 2);
        assert_eq!(arch.dma_mm2s_channels(TileType::Compute), 2);
        assert_eq!(arch.dma_total_channels(TileType::Compute), 4);
        assert_eq!(arch.dma_bd_count(TileType::Compute), 16);

        // Mem tile: 6 S2MM + 6 MM2S = 12 channels
        assert_eq!(arch.dma_s2mm_channels(TileType::MemTile), 6);
        assert_eq!(arch.dma_mm2s_channels(TileType::MemTile), 6);
        assert_eq!(arch.dma_total_channels(TileType::MemTile), 12);
        assert_eq!(arch.dma_bd_count(TileType::MemTile), 48);
    }

    #[test]
    fn test_aie2_port_layouts() {
        let arch = Aie2Config;

        // Verify port counts match spec
        assert_eq!(arch.master_ports(TileType::Shim).len(), 22);
        assert_eq!(arch.slave_ports(TileType::Shim).len(), 23);

        assert_eq!(arch.master_ports(TileType::MemTile).len(), 17);
        assert_eq!(arch.slave_ports(TileType::MemTile).len(), 18);

        assert_eq!(arch.master_ports(TileType::Compute).len(), 23);
        assert_eq!(arch.slave_ports(TileType::Compute).len(), 25);
    }

    #[test]
    fn test_aie2_port_ranges() {
        let arch = Aie2Config;

        // Shim north masters: 12-17 (6 ports)
        let (start, end) = arch.north_master_range(TileType::Shim);
        assert_eq!((start, end), (12, 17));
        assert_eq!(end - start + 1, 6);

        // MemTile south masters: 7-10 (4 ports)
        let (start, end) = arch.south_master_range(TileType::MemTile);
        assert_eq!((start, end), (7, 10));

        // MemTile north masters: 11-16 (6 ports)
        let (start, end) = arch.north_master_range(TileType::MemTile);
        assert_eq!((start, end), (11, 16));
    }

    #[test]
    fn test_valid_tile_check() {
        let arch = Aie2Config;

        // Valid positions
        assert!(arch.is_valid_tile(0, 0));
        assert!(arch.is_valid_tile(4, 5));

        // Invalid positions
        assert!(!arch.is_valid_tile(5, 0)); // col out of range
        assert!(!arch.is_valid_tile(0, 6)); // row out of range
        assert!(!arch.is_valid_tile(10, 10));
    }

    #[test]
    fn test_default_arch() {
        let arch = default_arch();
        assert_eq!(arch.columns(), 5);
        assert_eq!(arch.rows(), 6);
        assert_eq!(arch.name(), "AIE2 (NPU1 - Phoenix/HawkPoint)");
    }

    #[test]
    fn test_arch_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Aie2Config>();
        assert_send_sync::<Aie2pConfig>();
    }
}
