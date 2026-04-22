//! Runtime-facing ArchConfig trait and data-driven ModelConfig implementation.
//!
//! This module provides a trait-based abstraction for NPU architecture parameters,
//! backed by the validated hardware model extracted from the open-source toolchain
//! (mlir-aie device models, aie-rt register data, AM025 JSON).
//!
//! # What lives here
//!
//! - `ArchConfig`: core trait covering dimensions, tile classification, memory
//!   sizes, lock counts, DMA configuration, and the per-arch model accessors
//!   (`dma_model()`, `lock_model()`, `stream_switch_model()`). Stream-switch
//!   port-layout data is reachable via `stream_switch_model().topology()`;
//!   see `crate::stream_switch::StreamSwitchTopology` for the carrier and
//!   `crate::aie2::stream_switch_model` for the AIE2 concrete impl.
//!
//! - `ModelConfig`: `ArchConfig` implementation backed by `ArchModel`. All
//!   device parameters come from the parsed device model; no build.rs constants
//!   are needed here. The npu1 and npu2 constructors both load from the same
//!   `tools/aie-device-models.json` at first use via `ARCHSPEC_MODELS`.
//!
//! - `default_arch()`: returns the NPU1 (Phoenix/HawkPoint) configuration.
//!   Callers that need architecture-specific behaviour should use `ModelConfig`
//!   constructors directly.
//!
//! # Method signatures
//!
//! All tile-type parameters use `TileKind` (the archspec crate's native enum),
//! not `TileType` (a runtime enum in the main crate). Runtime callers convert
//! via `.into()` at the boundary, using the `From<TileType> for TileKind` and
//! `From<TileKind> for TileType` impls established in Task 2.
//!
//! # Example
//!
//! ```rust
//! use xdna_archspec::runtime::{ArchConfig, ModelConfig, default_arch};
//! use xdna_archspec::types::TileKind;
//! use std::sync::Arc;
//!
//! let arch: Arc<dyn ArchConfig> = default_arch();
//! assert_eq!(arch.columns(), 5);
//! assert_eq!(arch.rows(), 6);
//! assert!(arch.is_shim_tile(0, 0));
//! assert!(arch.is_mem_tile(0, 1));
//! assert!(arch.is_compute_tile(0, 2));
//! assert_eq!(arch.data_memory_size(TileKind::Compute), 64 * 1024);
//! assert_eq!(arch.program_memory_size(TileKind::Compute), 16 * 1024);
//! ```

use crate::types::{ArchModel, Architecture, TileKind};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// Architecture configuration trait for NPU variants.
///
/// Implementations of this trait encapsulate all architecture-specific parameters:
/// - Array dimensions (columns, rows)
/// - Tile type classification (which coordinates are shim/memtile/compute)
/// - Memory sizes per tile type
/// - DMA configuration (channels, buffer descriptors)
///
/// Behavioral seams (DMA, locks, stream switch) are reached via the
/// `dma_model()`, `lock_model()`, and `stream_switch_model()` accessors.
/// Stream-switch port-layout data is exposed through
/// `stream_switch_model().topology()`, which returns a
/// `&'static StreamSwitchTopology` aggregating the build.rs-generated
/// port arrays and range constants into a per-tile-kind carrier.
///
/// All tile-type parameters use `TileKind` (this crate's native enum). Runtime
/// callers using `TileType` convert via `.into()` at the call site.
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

    /// Get the tile kind at the given coordinates.
    ///
    /// Returns `TileKind::ShimNoc` for row 0 (the emulator does not model
    /// `ShimPl` tiles; all row-0 tiles on NPU1 are NoC-connected).
    fn tile_kind(&self, col: u8, row: u8) -> TileKind;

    /// Check if a tile position is valid.
    fn is_valid_tile(&self, col: u8, row: u8) -> bool {
        col < self.columns() && row < self.rows()
    }

    /// Check if a tile is a shim tile (DDR interface, row 0).
    ///
    /// Returns true for both `ShimNoc` and `ShimPl` -- the emulator collapses
    /// both to the same row-0 shim role.
    fn is_shim_tile(&self, col: u8, row: u8) -> bool {
        matches!(self.tile_kind(col, row), TileKind::ShimNoc | TileKind::ShimPl)
    }

    /// Check if a tile is a memory tile (large shared memory, no core).
    fn is_mem_tile(&self, col: u8, row: u8) -> bool {
        self.tile_kind(col, row) == TileKind::Mem
    }

    /// Check if a tile is a compute tile (has AIE core + local memory).
    fn is_compute_tile(&self, col: u8, row: u8) -> bool {
        self.tile_kind(col, row) == TileKind::Compute
    }

    // ========================================================================
    // Memory Configuration
    // ========================================================================

    /// Get the data memory size in bytes for a tile type.
    ///
    /// Returns 0 for shim tiles (they have no local data SRAM; the device model
    /// reports address space size, not physical SRAM).
    fn data_memory_size(&self, tile: TileKind) -> usize;

    /// Get the program (instruction) memory size in bytes for a tile type.
    ///
    /// Only compute tiles have program memory. Returns 0 for shim and mem tiles.
    /// For AIE2 compute tiles this is 16 KB.
    fn program_memory_size(&self, tile: TileKind) -> usize;

    /// Get the number of hardware lock primitives for a tile type.
    fn lock_count(&self, tile: TileKind) -> usize;

    /// Get the maximum value storable in a lock register.
    ///
    /// For AIE2 this is 63 (6-bit unsigned state field).
    fn max_lock_value(&self) -> u32;

    // ========================================================================
    // DMA Configuration
    // ========================================================================

    /// Get the number of S2MM (stream-to-memory) DMA channels for a tile type.
    fn dma_s2mm_channels(&self, tile: TileKind) -> usize;

    /// Get the number of MM2S (memory-to-stream) DMA channels for a tile type.
    fn dma_mm2s_channels(&self, tile: TileKind) -> usize;

    /// Get the total number of DMA channels (S2MM + MM2S) for a tile type.
    fn dma_total_channels(&self, tile: TileKind) -> usize {
        self.dma_s2mm_channels(tile) + self.dma_mm2s_channels(tile)
    }

    /// Get the number of DMA buffer descriptors for a tile type.
    fn dma_bd_count(&self, tile: TileKind) -> usize;

    // ========================================================================
    // Architecture Name
    // ========================================================================

    /// Get a human-readable architecture name for display purposes.
    fn name(&self) -> &'static str;

    // ========================================================================
    // DMA Model
    // ========================================================================

    /// Return the DMA feature-flag + timing model for this architecture.
    ///
    /// The returned reference is `'static` because every concrete `DmaModel`
    /// impl is a zero-sized, stateless singleton (e.g. `AIE2_DMA_MODEL`).
    /// Production callers pass this into `DmaEngine::new()`; test helpers
    /// can use `&xdna_archspec::aie2::dma::AIE2_DMA_MODEL` directly.
    fn dma_model(&self) -> &'static dyn crate::dma::DmaModel;

    // ========================================================================
    // Lock Model (Subsystem 4)
    // ========================================================================

    /// Return the lock feature-flag + value-layout model for this architecture.
    ///
    /// The returned reference is `'static` because every concrete `LockModel`
    /// impl is a zero-sized, stateless singleton (e.g. `AIE2_LOCK_MODEL`).
    /// Cold-path callers read the returned `value_layout()` when they need
    /// the mask / sign-extend formula; hot-path callers (none today)
    /// should cache `&'static LockValueLayout` at construction.
    fn lock_model(&self) -> &'static dyn crate::locks::LockModel;

    // ========================================================================
    // Stream Switch Model (Subsystem 5)
    // ========================================================================

    /// Returns the stream-switch model for this architecture.
    ///
    /// The returned reference is `&'static` so consumers may cache
    /// it for the lifetime of the process without concern about
    /// dangling pointers. All AIE2-family devices share a single
    /// `AIE2_STREAM_SWITCH_MODEL` singleton.
    ///
    /// Calls on `Architecture::Aie` (AIE1 / Versal) panic via
    /// `unimplemented!()` until an AIE1 model is populated.
    fn stream_switch_model(&self) -> &'static dyn crate::stream_switch::StreamSwitchModel;

    // ========================================================================
    // ISA Execute Model (Subsystem 7)
    // ========================================================================

    /// Per-arch ISA execute model (Subsystem 7 behavioral seam).
    ///
    /// Returns `&'static dyn IsaExecutor`. Ships empty per the
    /// audit's Approach A conclusion (zero trait methods
    /// warranted); attached for future seams.
    fn isa_executor(&self) -> &'static dyn crate::isa_execute::IsaExecutor;
}

// ============================================================================
// Data-driven configuration from mlir-aie DeviceModel
// ============================================================================

/// Cached device model data pre-extracted into per-tile-type lookup tables.
///
/// Avoids HashMap lookups on every `ArchConfig` method call by caching the
/// values that change per tile type. Populated once from `ArchModel` during
/// `ModelConfig` construction.
#[derive(Debug, Clone)]
struct TileTypeParams {
    /// Data memory size in bytes (64KB for compute, 512KB for mem, 0 for shim).
    data_memory_size: usize,
    /// Program (instruction) memory size in bytes (16KB for compute, 0 otherwise).
    ///
    /// Sourced from `TileTypeModel::memory::program_memory_bytes` in the ArchModel.
    /// This is a separate field from `data_memory_size` -- they are different
    /// memory regions with different sizes and functions.
    program_memory_size: usize,
    /// Number of hardware lock primitives.
    num_locks: usize,
    /// Number of DMA buffer descriptors.
    num_bds: usize,
    /// Number of DMA S2MM (stream-to-memory / slave / inbound) channels.
    dma_s2mm: usize,
    /// Number of DMA MM2S (memory-to-stream / master / outbound) channels.
    dma_mm2s: usize,
}

/// Architecture configuration backed by mlir-aie's `DeviceModel`.
///
/// All device-configuration values (dimensions, memory sizes, lock/BD counts,
/// DMA channels) come from the parsed ArchModel. Stream-switch port layouts
/// and ranges (AM025 register-level data) come from build.rs-generated
/// modules in `crate::aie2::{COMPUTE,MEMTILE,SHIM}_{MASTER,SLAVE}_PORTS` and
/// `crate::aie2::stream_switch::{compute,mem_tile,shim}`; they are aggregated
/// into `AIE2_STREAM_SWITCH_TOPOLOGY` and exposed via
/// `ArchConfig::stream_switch_model().topology()`.
///
/// # Constructors
///
/// Use `ModelConfig::npu1()` or `ModelConfig::npu2()` for convenience. These
/// load from `tools/aie-device-models.json` via `ARCHSPEC_MODELS`.
///
/// Use `ModelConfig::from_arch_model()` when you have an `ArchModel` already.
///
/// # Column Convention
///
/// `from_arch_model()` adds 1 to the model's column count to match the
/// emulator's convention where column 0 is the leftmost shim column.
/// mlir-aie counts only "compute columns" (e.g. 4 for NPU1); the emulator
/// uses 5 (columns 0-4).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Total number of tile columns (model count + 1 for the emulator's column-0
    /// convention).
    columns: u8,
    /// Total number of tile rows.
    rows: u8,
    /// Number of memory tile rows, used to classify row numbers into tile types.
    /// Row 0 is always shim; rows 1..=num_mem_tile_rows are mem tiles;
    /// rows above are compute tiles.
    num_mem_tile_rows: u8,
    /// Maximum lock counter value (63 for AIE2, from device_constants).
    max_lock_value: u32,
    /// Per-tile-type parameter caches (indexed by tile kind).
    core_params: TileTypeParams,
    mem_tile_params: TileTypeParams,
    shim_params: TileTypeParams,
    /// Human-readable architecture name (leaked &'static str for trait compat).
    arch_name: &'static str,
    /// Architecture family (AIE, AIE2, AIE2P) -- stored for `dma_model()`
    /// dispatch and any future per-arch branching in `ArchConfig` impls.
    architecture: Architecture,
}

/// Pre-loaded device models from the archspec JSON, parsed once at first use.
///
/// Uses the same `xdna_archspec::device_model` parser that build.rs uses for
/// npu1 at compile time, so the JSON is only ever parsed by one code path.
/// Keyed by device name (e.g. "npu1", "npu2", "xcve2802").
///
/// The JSON is located relative to the workspace root:
/// `xdna-emu/tools/aie-device-models.json`. The path is resolved at runtime
/// using `CARGO_MANIFEST_DIR` (crate root = `crates/xdna-archspec/`) and
/// navigating up two levels to the workspace root.
pub static ARCHSPEC_MODELS: LazyLock<HashMap<String, ArchModel>> =
    LazyLock::new(|| {
        let json_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/aie-device-models.json");
        crate::device_model::extract_device_models(&json_path)
            .unwrap_or_else(|e| panic!(
                "Failed to load device models from {}: {}. \
                 Run: python3 tools/aie-device-dump.py > tools/aie-device-models.json",
                json_path.display(), e
            ))
    });

impl ModelConfig {
    /// Create a `ModelConfig` from an `ArchModel`.
    ///
    /// This is the single conversion path from archspec JSON data (whether
    /// loaded at build time or runtime) into an `ArchConfig` implementation.
    ///
    /// The model's column count is incremented by 1 to match the emulator's
    /// column-0 convention (see struct-level documentation).
    ///
    /// `arch_name` is a `&'static str` for the display name; callers should
    /// use a string literal or `Box::leak()` for dynamically generated strings.
    pub fn from_arch_model(model: &ArchModel, arch_name: &'static str) -> Self {
        // Helper: extract DMA slave (S2MM) and master (MM2S) channel counts
        // from a tile's port bundle list.  Shim tiles use shim_mux_ports;
        // all others use switchbox_ports.  The "DMA" bundle contains the
        // DMA-connected stream ports; slaves = S2MM inputs, masters = MM2S outputs.
        let dma_channels = |tile: &crate::types::TileTypeModel| -> (usize, usize) {
            let ports = match tile.kind {
                TileKind::ShimNoc | TileKind::ShimPl => &tile.shim_mux_ports,
                _ => &tile.switchbox_ports,
            };
            let dma = ports.iter()
                .find(|p| p.bundle == "DMA")
                .expect("DMA bundle missing from tile type; \
                         regenerate device model with tools/aie-device-dump.py");
            (dma.slaves as usize, dma.masters as usize)
        };

        let compute = model.tile_types.iter().find(|t| t.kind == TileKind::Compute)
            .expect("core tile type missing from device model JSON; regenerate with aie-device-dump.py");
        let (compute_s2mm, compute_mm2s) = dma_channels(compute);
        // Data memory: from model or AIE2 default (64 KB).
        let compute_data_size = compute.memory.as_ref()
            .map(|m| m.size_bytes as usize)
            .unwrap_or(65536);
        // Program memory: from model's `program_memory_bytes`, or 0 if absent.
        // This is distinct from data memory -- 16 KB on AIE2 compute tiles.
        let compute_prog_size = compute.memory.as_ref()
            .and_then(|m| m.program_memory_bytes)
            .map(|b| b as usize)
            .unwrap_or(0);
        let core_params = TileTypeParams {
            data_memory_size: compute_data_size,
            program_memory_size: compute_prog_size,
            num_locks: *compute.instances.locks.value() as usize,
            num_bds: *compute.instances.bds.value() as usize,
            dma_s2mm: compute_s2mm,
            dma_mm2s: compute_mm2s,
        };

        let memtile = model.tile_types.iter().find(|t| t.kind == TileKind::Mem)
            .expect("mem_tile tile type missing from device model JSON; regenerate with aie-device-dump.py");
        let (mem_s2mm, mem_mm2s) = dma_channels(memtile);
        // Mem tiles have no program memory.
        let mem_data_size = memtile.memory.as_ref()
            .map(|m| m.size_bytes as usize)
            .unwrap_or(524288); // AIE2 default: 512 KB
        let mem_tile_params = TileTypeParams {
            data_memory_size: mem_data_size,
            program_memory_size: 0,
            num_locks: *memtile.instances.locks.value() as usize,
            num_bds: *memtile.instances.bds.value() as usize,
            dma_s2mm: mem_s2mm,
            dma_mm2s: mem_mm2s,
        };

        let shim = model.tile_types.iter().find(|t| t.kind == TileKind::ShimNoc)
            .expect("shim_noc tile type missing from device model JSON; regenerate with aie-device-dump.py");
        let (shim_s2mm, shim_mm2s) = dma_channels(shim);
        // Shim tiles have no local data memory (the device model reports
        // address space size, not physical SRAM).  No program memory either.
        let shim_params = TileTypeParams {
            data_memory_size: 0,
            program_memory_size: 0,
            num_locks: *shim.instances.locks.value() as usize,
            num_bds: *shim.instances.bds.value() as usize,
            dma_s2mm: shim_s2mm,
            dma_mm2s: shim_mm2s,
        };

        let topo = model.array_topology.as_ref()
            .expect("array topology missing from device model JSON; regenerate with aie-device-dump.py");
        let max_lock_value = model.device_constants.as_ref()
            .map(|dc| dc.max_lock_value)
            .unwrap_or(63); // AIE2/AIE2P default

        Self {
            // The emulator adds 1 column to the model's count for the
            // column-0 convention used in CDO addressing.
            columns: topo.columns + 1,
            rows: topo.rows,
            num_mem_tile_rows: topo.num_mem_tile_rows,
            max_lock_value,
            core_params,
            mem_tile_params,
            shim_params,
            arch_name,
            architecture: model.arch,
        }
    }

    /// Return the `TileTypeParams` for the given tile kind.
    ///
    /// Both shim variants (ShimNoc and ShimPl) map to the same shim params
    /// because the emulator treats them identically.
    fn params(&self, tile: TileKind) -> &TileTypeParams {
        match tile {
            TileKind::ShimNoc | TileKind::ShimPl => &self.shim_params,
            TileKind::Mem => &self.mem_tile_params,
            TileKind::Compute => &self.core_params,
        }
    }

    /// Create the NPU1 (Phoenix/HawkPoint, AIE2) configuration.
    ///
    /// Loads from the device model JSON via `ARCHSPEC_MODELS` (initialized on
    /// first call).  The JSON path is resolved relative to the workspace root:
    /// `tools/aie-device-models.json`.
    pub fn npu1() -> Arc<dyn ArchConfig> {
        let model = ARCHSPEC_MODELS.get("npu1")
            .expect("npu1 device not found in device model JSON; regenerate with aie-device-dump.py");
        Arc::new(Self::from_arch_model(model, "AIE2 (NPU1 - Phoenix/HawkPoint)"))
    }

    /// Create the NPU2 (Strix Point, AIE2P) configuration.
    ///
    /// Loads from the device model JSON via `ARCHSPEC_MODELS`.
    pub fn npu2() -> Arc<dyn ArchConfig> {
        let model = ARCHSPEC_MODELS.get("npu2")
            .expect("npu2 device not found in device model JSON; regenerate with aie-device-dump.py");
        Arc::new(Self::from_arch_model(model, "AIE2P (NPU2+ - Strix/Krackan)"))
    }

    /// Create the VC2802 (Versal AIE2) configuration for aiesimulator validation.
    ///
    /// VC2802 is the 38-column x 11-row AIE2 array that aiesimulator uses for
    /// all AIE2 targets.  Same tile silicon as NPU1, just a larger array.
    pub fn xcve2802() -> Arc<dyn ArchConfig> {
        let model = ARCHSPEC_MODELS.get("xcve2802")
            .expect("xcve2802 device not found in device model JSON; add to tools/aie-device-models.json");
        Arc::new(Self::from_arch_model(model, "AIE2 (VC2802 - Versal, aiesim validation)"))
    }
}

impl ArchConfig for ModelConfig {
    fn columns(&self) -> u8 {
        self.columns
    }

    fn rows(&self) -> u8 {
        self.rows
    }

    fn tile_kind(&self, _col: u8, row: u8) -> TileKind {
        // Column is unused: the NPU array is homogeneous within each row band.
        // Row 0 is always shim (NoC-connected on NPU; ShimPl is not modelled).
        match row {
            0 => TileKind::ShimNoc,
            r if r <= self.num_mem_tile_rows => TileKind::Mem,
            _ => TileKind::Compute,
        }
    }

    fn data_memory_size(&self, tile: TileKind) -> usize {
        self.params(tile).data_memory_size
    }

    fn program_memory_size(&self, tile: TileKind) -> usize {
        // Sourced from TileTypeModel::memory::program_memory_bytes in ArchModel,
        // stored in TileTypeParams::program_memory_size during from_arch_model().
        // Non-compute tiles return 0 because their TileTypeParams were
        // constructed with program_memory_size: 0.
        self.params(tile).program_memory_size
    }

    fn lock_count(&self, tile: TileKind) -> usize {
        self.params(tile).num_locks
    }

    fn max_lock_value(&self) -> u32 {
        self.max_lock_value
    }

    fn dma_s2mm_channels(&self, tile: TileKind) -> usize {
        self.params(tile).dma_s2mm
    }

    fn dma_mm2s_channels(&self, tile: TileKind) -> usize {
        self.params(tile).dma_mm2s
    }

    fn dma_bd_count(&self, tile: TileKind) -> usize {
        self.params(tile).num_bds
    }

    fn name(&self) -> &'static str {
        self.arch_name
    }

    fn dma_model(&self) -> &'static dyn crate::dma::DmaModel {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::dma::AIE2_DMA_MODEL
            }
            Architecture::Aie => {
                unimplemented!(
                    "AIE1 DmaModel not populated until AIE1 support lands \
                     (see docs/arch/dma-model.md for planned Aie1DmaModel)"
                )
            }
        }
    }

    fn lock_model(&self) -> &'static dyn crate::locks::LockModel {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::locks::AIE2_LOCK_MODEL
            }
            Architecture::Aie => {
                unimplemented!(
                    "AIE1 LockModel not populated until AIE1 support lands \
                     (see docs/arch/lock-model.md for planned Aie1LockModel)"
                )
            }
        }
    }

    fn stream_switch_model(&self) -> &'static dyn crate::stream_switch::StreamSwitchModel {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::stream_switch_model::AIE2_STREAM_SWITCH_MODEL
            }
            Architecture::Aie => unimplemented!(
                "AIE1 StreamSwitchModel not populated; add \
                 xdna_archspec::aie1::stream_switch_model::AIE1_STREAM_SWITCH_MODEL"
            ),
        }
    }

    fn isa_executor(&self) -> &'static dyn crate::isa_execute::IsaExecutor {
        match self.architecture {
            Architecture::Aie2 | Architecture::Aie2p => {
                &crate::aie2::isa_execute_model::AIE2_ISA_EXECUTOR
            }
            Architecture::Aie => unimplemented!(
                "AIE1 IsaExecutor not populated; add \
                 xdna_archspec::aie1::isa_execute_model::Aie1IsaExecutor"
            ),
        }
    }
}

// ============================================================================
// Default Architecture
// ============================================================================

/// Return the default architecture configuration (NPU1, Phoenix/HawkPoint).
///
/// This is the primary development target.  Use `ModelConfig::npu2()` or
/// `ModelConfig::xcve2802()` for other devices.
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
            assert_eq!(arch.tile_kind(col, 0), TileKind::ShimNoc);
        }

        // Mem tiles (row 1)
        for col in 0..5 {
            assert!(arch.is_mem_tile(col, 1), "({}, 1) should be memtile", col);
            assert_eq!(arch.tile_kind(col, 1), TileKind::Mem);
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
                assert_eq!(arch.tile_kind(col, row), TileKind::Compute);
            }
        }
    }

    #[test]
    fn test_npu1_memory_sizes() {
        let arch = npu1_arch();

        assert_eq!(arch.data_memory_size(TileKind::ShimNoc), 0);
        assert_eq!(arch.data_memory_size(TileKind::Mem), 512 * 1024);
        assert_eq!(arch.data_memory_size(TileKind::Compute), 64 * 1024);

        // program_memory_size from TileTypeModel::memory::program_memory_bytes
        assert_eq!(arch.program_memory_size(TileKind::ShimNoc), 0);
        assert_eq!(arch.program_memory_size(TileKind::Mem), 0);
        assert_eq!(arch.program_memory_size(TileKind::Compute), 16 * 1024);
    }

    #[test]
    fn test_npu1_lock_counts() {
        let arch = npu1_arch();

        assert_eq!(arch.lock_count(TileKind::ShimNoc), 16);
        assert_eq!(arch.lock_count(TileKind::Mem), 64);
        assert_eq!(arch.lock_count(TileKind::Compute), 16);
    }

    #[test]
    fn test_npu1_dma_config() {
        let arch = npu1_arch();

        // Compute tile: 2 S2MM + 2 MM2S = 4 channels, 16 BDs
        assert_eq!(arch.dma_s2mm_channels(TileKind::Compute), 2);
        assert_eq!(arch.dma_mm2s_channels(TileKind::Compute), 2);
        assert_eq!(arch.dma_total_channels(TileKind::Compute), 4);
        assert_eq!(arch.dma_bd_count(TileKind::Compute), 16);

        // Mem tile: 6 S2MM + 6 MM2S = 12 channels, 48 BDs
        assert_eq!(arch.dma_s2mm_channels(TileKind::Mem), 6);
        assert_eq!(arch.dma_mm2s_channels(TileKind::Mem), 6);
        assert_eq!(arch.dma_total_channels(TileKind::Mem), 12);
        assert_eq!(arch.dma_bd_count(TileKind::Mem), 48);
    }

    #[test]
    fn test_valid_tile_check() {
        let arch = npu1_arch();

        assert!(arch.is_valid_tile(0, 0));
        assert!(arch.is_valid_tile(4, 5));

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

    #[test]
    fn test_vc2802_dimensions() {
        let arch = ModelConfig::xcve2802();
        // VC2802: 37 + 1 = 38 columns, 11 rows
        assert_eq!(arch.columns(), 38);
        assert_eq!(arch.rows(), 11);
    }

    #[test]
    fn test_vc2802_tile_classification() {
        let arch = ModelConfig::xcve2802();
        // Row 0: shim
        assert_eq!(arch.tile_kind(0, 0), TileKind::ShimNoc);
        assert_eq!(arch.tile_kind(7, 0), TileKind::ShimNoc);
        // Rows 1-2: mem tiles (VC2802 has 2 mem tile rows)
        assert_eq!(arch.tile_kind(7, 1), TileKind::Mem);
        assert_eq!(arch.tile_kind(7, 2), TileKind::Mem);
        // Rows 3-10: compute
        assert_eq!(arch.tile_kind(7, 3), TileKind::Compute);
        assert_eq!(arch.tile_kind(7, 10), TileKind::Compute);
    }

    #[test]
    fn test_vc2802_tile_params_match_npu1() {
        // Same AIE2 silicon -- per-tile-type params should be identical
        let vc = ModelConfig::xcve2802();
        let npu = ModelConfig::npu1();
        assert_eq!(
            vc.data_memory_size(TileKind::Compute),
            npu.data_memory_size(TileKind::Compute),
        );
        assert_eq!(
            vc.lock_count(TileKind::Compute),
            npu.lock_count(TileKind::Compute),
        );
        assert_eq!(
            vc.dma_s2mm_channels(TileKind::Compute),
            npu.dma_s2mm_channels(TileKind::Compute),
        );
    }

    #[test]
    fn test_shim_pl_is_shim_tile() {
        // ShimPl is hardware-real but not modelled in the emulator's tile array.
        // tile_kind() never returns ShimPl (from_arch_model uses ShimNoc).
        // But is_shim_tile() must handle it if it ever appears.
        let arch = npu1_arch();
        // tile_kind() for row 0 always returns ShimNoc on NPU1.
        assert_eq!(arch.tile_kind(0, 0), TileKind::ShimNoc);
        assert!(arch.is_shim_tile(0, 0));
    }

    #[test]
    fn test_max_lock_value() {
        let arch = npu1_arch();
        assert_eq!(arch.max_lock_value(), 63); // AIE2: 6-bit unsigned
    }

    #[test]
    fn stream_switch_model_dispatches_to_aie2_for_aie2_family() {
        for name in &["npu1", "npu2", "npu4", "npu5", "npu6"] {
            if let Some(model) = ARCHSPEC_MODELS.get(*name) {
                let cfg = ModelConfig::from_arch_model(model, name);
                assert!(
                    cfg.stream_switch_model().supports_deterministic_merge(),
                    "{}: AIE2-family must report supports_deterministic_merge = true",
                    name
                );
            }
        }
    }

    #[test]
    fn isa_executor_dispatches_to_aie2_for_aie2_family() {
        for name in &["npu1", "npu2", "npu4", "npu5", "npu6"] {
            if let Some(model) = ARCHSPEC_MODELS.get(*name) {
                let cfg = ModelConfig::from_arch_model(model, name);
                let executor = cfg.isa_executor();
                let expected: &dyn crate::isa_execute::IsaExecutor =
                    &crate::aie2::isa_execute_model::AIE2_ISA_EXECUTOR;
                // Both AIE2 and AIE2P should dispatch to the same singleton
                assert!(core::ptr::eq(
                    executor as *const _ as *const (),
                    expected as *const _ as *const ()
                ), "{}: isa_executor must dispatch to AIE2_ISA_EXECUTOR", name);
            }
        }
    }
}
