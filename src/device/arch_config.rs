//! Architecture configuration trait and data-driven implementations.
//!
//! This module provides a trait-based abstraction for NPU architecture parameters.
//! The primary implementation, `ModelConfig`, derives its values from mlir-aie's
//! `AIETargetModel` (parsed from JSON by `DeviceModel`). This replaces the previous
//! hardcoded `Aie2Config`/`Aie2pConfig` structs.
//!
//! Stream switch port layouts and ranges (AM025 register-level data not modeled
//! by mlir-aie) are still sourced from `aie2_spec`.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::device::arch_config::{ArchConfig, ModelConfig};
//! use std::sync::Arc;
//!
//! let arch: Arc<dyn ArchConfig> = ModelConfig::npu1();
//! assert_eq!(arch.columns(), 5);
//! assert_eq!(arch.rows(), 6);
//! assert!(arch.is_shim_tile(0, 0));
//! assert!(arch.is_mem_tile(0, 1));
//! assert!(arch.is_compute_tile(0, 2));
//! ```

use super::aie2_spec;
use super::model::DeviceModel;
use super::tile::TileType;
use std::sync::{Arc, LazyLock};

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

    /// Get the maximum value storable in a lock register.
    /// For AIE2 this is 63 (6-bit unsigned state).
    fn max_lock_value(&self) -> u32;

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
// Data-driven configuration from mlir-aie DeviceModel
// ============================================================================

/// Cached device model data, pre-extracted into per-tile-type lookup tables
/// for fast access during emulation. This avoids HashMap lookups on every
/// ArchConfig method call.
#[derive(Debug, Clone)]
struct TileTypeParams {
    /// Data memory size in bytes.
    data_memory_size: usize,
    /// Number of locks.
    num_locks: usize,
    /// Number of DMA buffer descriptors.
    num_bds: usize,
    /// Number of DMA S2MM (slave/inbound) channels.
    dma_s2mm: usize,
    /// Number of DMA MM2S (master/outbound) channels.
    dma_mm2s: usize,
}

/// Architecture configuration backed by mlir-aie's `DeviceModel`.
///
/// All device-configuration values (dimensions, memory sizes, lock/BD counts,
/// DMA channels) come from the parsed model. Stream switch port layouts and
/// ranges (AM025 register-level data) come from static arrays in `aie2_spec`
/// since mlir-aie does not model per-port-index assignments.
///
/// `ModelConfig` replaces the previous `Aie2Config` and `Aie2pConfig` structs.
/// Use `ModelConfig::npu1()` or `ModelConfig::npu2()` for convenience.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of tile columns (from model, includes +1 for the emulator's
    /// column 0 convention if needed).
    columns: u8,
    /// Number of tile rows.
    rows: u8,
    /// Number of memory tile rows (determines which rows are mem tiles).
    num_mem_tile_rows: u32,
    /// Maximum lock value (from model, e.g. 63 for AIE2).
    max_lock_value: u32,
    /// Per-tile-type parameters, pre-extracted from the model.
    core_params: TileTypeParams,
    mem_tile_params: TileTypeParams,
    shim_params: TileTypeParams,
    /// Architecture name for display (leaked &'static str).
    arch_name: &'static str,
}

/// Pre-loaded device model set from the embedded JSON.
///
/// Loaded once via LazyLock to avoid repeated file I/O or JSON parsing.
/// The JSON file is generated by `tools/aie-device-dump.py` and stored
/// at `tools/aie-device-models.json` in the repository.
static DEVICE_MODELS: LazyLock<super::model::DeviceModelSet> = LazyLock::new(|| {
    let json_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools/aie-device-models.json");
    super::model::DeviceModelSet::load_from_json(&json_path)
        .unwrap_or_else(|e| panic!(
            "Failed to load device model from {}: {}. \
             Run: python3 tools/aie-device-dump.py > tools/aie-device-models.json",
            json_path.display(), e
        ))
});

impl ModelConfig {
    /// Create a ModelConfig from a parsed DeviceModel.
    ///
    /// Adds 1 to the model's column count to match the emulator's convention
    /// where column 0 is the leftmost shim column. mlir-aie counts only
    /// "compute columns" (e.g. 4 for NPU1), while the emulator uses 5
    /// (columns 0-4).
    pub fn from_model(model: &DeviceModel, arch_name: &'static str) -> Self {
        // Extract per-tile-type params from the model's tile_types map.
        let core_params = model.core_config().map(|c| {
            let (s2mm, mm2s) = c.switchbox_ports.get("DMA")
                .map(|dma| (dma.slave as usize, dma.master as usize))
                .unwrap_or((2, 2));
            TileTypeParams {
                data_memory_size: model.local_memory_size as usize,
                num_locks: c.num_locks as usize,
                num_bds: c.num_bds as usize,
                dma_s2mm: s2mm,
                dma_mm2s: mm2s,
            }
        }).expect("core tile type missing from device model JSON; regenerate with aie-device-dump.py");

        let mem_tile_params = model.mem_tile_config().map(|c| {
            let (s2mm, mm2s) = c.switchbox_ports.get("DMA")
                .map(|dma| (dma.slave as usize, dma.master as usize))
                .unwrap_or((6, 6));
            TileTypeParams {
                data_memory_size: model.mem_tile_size as usize,
                num_locks: c.num_locks as usize,
                num_bds: c.num_bds as usize,
                dma_s2mm: s2mm,
                dma_mm2s: mm2s,
            }
        }).expect("mem_tile tile type missing from device model JSON; regenerate with aie-device-dump.py");

        let shim_params = model.shim_noc_config().map(|c| {
            // Shim DMA ports are in the shim mux (between the NoC and AXI),
            // not the regular switchbox. Fall back to switchbox_ports if the
            // shim_mux_ports key is absent.
            let (s2mm, mm2s) = c.shim_mux_ports.get("DMA")
                .or_else(|| c.switchbox_ports.get("DMA"))
                .map(|dma| (dma.slave as usize, dma.master as usize))
                .unwrap_or((2, 2));
            TileTypeParams {
                data_memory_size: 0,
                num_locks: c.num_locks as usize,
                num_bds: c.num_bds as usize,
                dma_s2mm: s2mm,
                dma_mm2s: mm2s,
            }
        }).expect("shim_noc tile type missing from device model JSON; regenerate with aie-device-dump.py");

        Self {
            // The emulator adds 1 column to the model's count for the
            // column 0 convention used in CDO addressing.
            columns: (model.columns + 1) as u8,
            rows: model.rows as u8,
            num_mem_tile_rows: model.num_mem_tile_rows,
            max_lock_value: model.max_lock_value,
            core_params,
            mem_tile_params,
            shim_params,
            arch_name,
        }
    }

    /// Get the parameters for a given tile type.
    fn params(&self, tile_type: TileType) -> &TileTypeParams {
        match tile_type {
            TileType::Shim => &self.shim_params,
            TileType::MemTile => &self.mem_tile_params,
            TileType::Compute => &self.core_params,
        }
    }

    /// Create the NPU1 (Phoenix/HawkPoint, AIE2) configuration.
    ///
    /// Loads from the pre-generated device model JSON.
    pub fn npu1() -> Arc<dyn ArchConfig> {
        let model = DEVICE_MODELS.npu1()
            .expect("npu1 device not found in device model JSON");
        Arc::new(Self::from_model(model, "AIE2 (NPU1 - Phoenix/HawkPoint)"))
    }

    /// Create the NPU2 (Strix Point, AIE2P) configuration.
    ///
    /// Loads from the pre-generated device model JSON.
    pub fn npu2() -> Arc<dyn ArchConfig> {
        let model = DEVICE_MODELS.npu2()
            .expect("npu2 device not found in device model JSON");
        Arc::new(Self::from_model(model, "AIE2P (NPU2+ - Strix/Krackan)"))
    }
}

impl ArchConfig for ModelConfig {
    fn columns(&self) -> u8 {
        self.columns
    }

    fn rows(&self) -> u8 {
        self.rows
    }

    fn tile_type(&self, _col: u8, row: u8) -> TileType {
        match row {
            0 => TileType::Shim,
            r if r <= self.num_mem_tile_rows as u8 => TileType::MemTile,
            _ => TileType::Compute,
        }
    }

    fn data_memory_size(&self, tile_type: TileType) -> usize {
        self.params(tile_type).data_memory_size
    }

    fn program_memory_size(&self, tile_type: TileType) -> usize {
        match tile_type {
            // Program memory size is not in the mlir-aie model; it's a
            // micro-architectural constant from AM020.
            TileType::Compute => aie2_spec::PROGRAM_MEMORY_SIZE,
            _ => 0,
        }
    }

    fn lock_count(&self, tile_type: TileType) -> usize {
        self.params(tile_type).num_locks
    }

    fn max_lock_value(&self) -> u32 {
        self.max_lock_value
    }

    fn dma_s2mm_channels(&self, tile_type: TileType) -> usize {
        self.params(tile_type).dma_s2mm
    }

    fn dma_mm2s_channels(&self, tile_type: TileType) -> usize {
        self.params(tile_type).dma_mm2s
    }

    fn dma_bd_count(&self, tile_type: TileType) -> usize {
        self.params(tile_type).num_bds
    }

    // Stream switch port layouts are AM025 register-level data. mlir-aie
    // only provides port *counts* per bundle, not per-port-index type
    // assignments. These static arrays remain in aie2_spec.
    //
    // When AIE2P diverges in port layout, we'll add AIE2P-specific arrays
    // and select based on architecture.

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
        self.arch_name
    }
}

// ============================================================================
// Default Architecture
// ============================================================================

/// Get the default architecture configuration (NPU1 from mlir-aie model).
pub fn default_arch() -> Arc<dyn ArchConfig> {
    ModelConfig::npu1()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn npu1_arch() -> Arc<dyn ArchConfig> {
        ModelConfig::npu1()
    }

    #[test]
    fn test_npu1_dimensions() {
        let arch = npu1_arch();
        assert_eq!(arch.columns(), 5);
        assert_eq!(arch.rows(), 6);
    }

    #[test]
    fn test_npu1_tile_classification() {
        let arch = npu1_arch();

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
    fn test_npu1_memory_sizes() {
        let arch = npu1_arch();

        assert_eq!(arch.data_memory_size(TileType::Shim), 0);
        assert_eq!(arch.data_memory_size(TileType::MemTile), 512 * 1024);
        assert_eq!(arch.data_memory_size(TileType::Compute), 64 * 1024);

        assert_eq!(arch.program_memory_size(TileType::Shim), 0);
        assert_eq!(arch.program_memory_size(TileType::MemTile), 0);
        assert_eq!(arch.program_memory_size(TileType::Compute), 16 * 1024);
    }

    #[test]
    fn test_npu1_lock_counts() {
        let arch = npu1_arch();

        assert_eq!(arch.lock_count(TileType::Shim), 16);
        assert_eq!(arch.lock_count(TileType::MemTile), 64);
        assert_eq!(arch.lock_count(TileType::Compute), 16);
    }

    #[test]
    fn test_npu1_dma_config() {
        let arch = npu1_arch();

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
    fn test_npu1_port_layouts() {
        let arch = npu1_arch();

        // Verify port counts match spec
        assert_eq!(arch.master_ports(TileType::Shim).len(), 22);
        assert_eq!(arch.slave_ports(TileType::Shim).len(), 23);

        assert_eq!(arch.master_ports(TileType::MemTile).len(), 17);
        assert_eq!(arch.slave_ports(TileType::MemTile).len(), 18);

        assert_eq!(arch.master_ports(TileType::Compute).len(), 23);
        assert_eq!(arch.slave_ports(TileType::Compute).len(), 25);
    }

    #[test]
    fn test_npu1_port_ranges() {
        use crate::device::aie2_spec::stream_switch::{compute, mem_tile, shim};

        let arch = npu1_arch();

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

        // Validate E/W generated constants match AM025 port arrays
        // Compute tile
        assert_eq!(compute::EAST_MASTER_START, 19);
        assert_eq!(compute::EAST_MASTER_END, 22);
        assert_eq!(compute::EAST_SLAVE_START, 19);
        assert_eq!(compute::EAST_SLAVE_END, 22);
        assert_eq!(compute::WEST_MASTER_START, 9);
        assert_eq!(compute::WEST_MASTER_END, 12);
        assert_eq!(compute::WEST_SLAVE_START, 11);
        assert_eq!(compute::WEST_SLAVE_END, 14);

        // Shim tile
        assert_eq!(shim::EAST_MASTER_START, 18);
        assert_eq!(shim::EAST_MASTER_END, 21);
        assert_eq!(shim::EAST_SLAVE_START, 18);
        assert_eq!(shim::EAST_SLAVE_END, 21);
        assert_eq!(shim::WEST_MASTER_START, 8);
        assert_eq!(shim::WEST_MASTER_END, 11);
        assert_eq!(shim::WEST_SLAVE_START, 10);
        assert_eq!(shim::WEST_SLAVE_END, 13);
        assert_eq!(shim::SOUTH_MASTER_START, 2);
        assert_eq!(shim::SOUTH_MASTER_END, 7);
        assert_eq!(shim::SOUTH_SLAVE_START, 2);
        assert_eq!(shim::SOUTH_SLAVE_END, 9);

        // Verify port counts are consistent (END - START + 1)
        assert_eq!(compute::EAST_MASTER_END - compute::EAST_MASTER_START + 1, 4);
        assert_eq!(compute::WEST_MASTER_END - compute::WEST_MASTER_START + 1, 4);
        assert_eq!(shim::EAST_MASTER_END - shim::EAST_MASTER_START + 1, 4);
        assert_eq!(shim::WEST_MASTER_END - shim::WEST_MASTER_START + 1, 4);
        assert_eq!(shim::SOUTH_MASTER_END - shim::SOUTH_MASTER_START + 1, 6);

        // Verify MemTile has no E/W ports (constants should not exist)
        // This is verified structurally: mem_tile module has no EAST_*/WEST_* constants
        assert_eq!(mem_tile::SOUTH_MASTER_START, 7);
        assert_eq!(mem_tile::NORTH_MASTER_START, 11);
    }

    #[test]
    fn test_valid_tile_check() {
        let arch = npu1_arch();

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
    fn test_model_config_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ModelConfig>();
    }

    #[test]
    fn test_npu2_dimensions() {
        let arch = ModelConfig::npu2();
        // NPU2 base has 8 compute columns + 1 = 9 total
        assert_eq!(arch.columns(), 9);
        assert_eq!(arch.rows(), 6);
    }
}
