//! AIE tile array representation.
//!
//! The tile array is the top-level device state. It holds all tiles
//! and provides methods to access them efficiently.
//!
//! # NPU Array Layouts
//!
//! | Device | Columns | Rows | Layout |
//! |--------|---------|------|--------|
//! | NPU1 | 5 | 6 | Col 0 shim, Rows 0 shim, 1 mem, 2-5 compute |
//! | NPU2 | 5 | 6 | Same as NPU1 |
//! | NPU3 | 9 | 6 | 8 columns + shim |
//!
//! # Performance
//!
//! Tiles are stored in a flat Vec for cache efficiency. Access is O(1)
//! via `col * rows + row` indexing.
//!
//! # DMA Integration
//!
//! Each tile has an associated `DmaEngine` stored in a parallel array.
//! DMA engines are accessed via `dma_engine(col, row)` and stepped via
//! `step_dma(col, row, host_memory)`.

mod ctrl;
mod dma_ops;
mod routing;

#[cfg(test)]
mod tests;

use xdna_archspec::runtime::{ArchConfig, ModelConfig};
use super::dma::{self, DmaEngine, DmaResult};
use super::host_memory::HostMemory;
use super::tile::{Tile, TileParams, TileKind};
use crate::interpreter::state::EventType;
use std::sync::Arc;

/// Maximum supported columns (VC2802 has 38 columns)
pub const MAX_COLS: usize = 38;

/// Maximum supported rows (VC2802 has 11 rows)
pub const MAX_ROWS: usize = 11;

/// Get mutable references to up to 3 tiles at disjoint indices for cross-tile lock access.
///
/// Returns (west_neighbor, own_tile, east_neighbor) where neighbors are None
/// if they don't exist or aren't MemTiles. Uses `split_at_mut` for safe
/// disjoint mutable borrows -- no unsafe code needed.
///
/// Indices: west = own_idx - rows, east = own_idx + rows (separated by one
/// column's worth of rows, always distinct).
pub(super) fn get_three_mut(
    tiles: &mut [Tile],
    own_idx: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> (Option<&mut Tile>, &mut Tile, Option<&mut Tile>) {
    let west_idx = if col > 0 {
        let idx = own_idx - rows;
        // Only provide neighbor if it's also a MemTile
        if tiles[idx].is_mem() { Some(idx) } else { None }
    } else {
        None
    };
    let east_idx = if col + 1 < cols {
        let idx = own_idx + rows;
        if tiles[idx].is_mem() { Some(idx) } else { None }
    } else {
        None
    };

    match (west_idx, east_idx) {
        (None, None) => {
            (None, &mut tiles[own_idx], None)
        }
        (Some(w), None) => {
            // w < own_idx guaranteed (west is lower column)
            let (left, right) = tiles.split_at_mut(own_idx);
            (Some(&mut left[w]), &mut right[0], None)
        }
        (None, Some(e)) => {
            // own_idx < e guaranteed (east is higher column)
            let (left, right) = tiles.split_at_mut(e);
            (None, &mut left[own_idx], Some(&mut right[0]))
        }
        (Some(w), Some(e)) => {
            // w < own_idx < e guaranteed
            let (left, rest) = tiles.split_at_mut(own_idx);
            let (mid, right) = rest.split_at_mut(e - own_idx);
            (Some(&mut left[w]), &mut mid[0], Some(&mut right[0]))
        }
    }
}

/// AIE tile array.
///
/// Stores tiles in row-major order for cache-friendly column iteration.
pub struct TileArray {
    /// Architecture configuration (determines tile types, port layouts, etc.)
    pub(super) arch: Arc<dyn ArchConfig>,

    /// Number of columns (including shim column)
    pub(super) cols: u8,

    /// Number of rows
    pub(super) rows: u8,

    /// Tiles stored in flat array: tiles[col * rows + row]
    /// Using Vec because tile count varies by device
    pub(crate) tiles: Vec<Tile>,

    /// DMA engines stored in parallel with tiles.
    /// Each tile has exactly one DMA engine.
    pub(crate) dma_engines: Vec<DmaEngine>,

    /// Fatal errors accumulated during data movement.
    ///
    /// These represent conditions that are impossible on real hardware
    /// (e.g., packet with no route, stream push failure). The coordinator
    /// checks this after each step and aborts if non-empty.
    pub(crate) fatal_errors: Vec<String>,

    /// Control packet actions produced during stream routing.
    ///
    /// Control packet writes from tiles are collected here instead of being
    /// executed immediately. The caller (DeviceState or coordinator) drains
    /// these and routes them through `DeviceState::write_tile_register()` for
    /// full module dispatch.
    pub(crate) pending_ctrl_actions: Vec<crate::device::tile::CtrlPacketAction>,

    /// Current cycle, set by the coordinator before each step.
    ///
    /// Used for timestamping memory-module trace events (lock acquire/release)
    /// that are generated during lock arbiter resolution in `step_data_movement()`.
    pub(super) current_cycle: u64,

    /// Per-tile control packet reassemblers.
    ///
    /// Handles word-by-word reassembly of control packets arriving at
    /// TileCtrl master ports. Moved here from Tile to keep tiles stateless
    /// for control packet protocol handling.
    pub(super) ctrl_reassemblers: Vec<crate::device::control_packets::StreamReassembler>,

    /// In-flight words traversing inter-tile links.
    ///
    /// On real AIE2 hardware, data takes ROUTE_LATENCY_PER_HOP cycles to
    /// traverse each inter-tile link (physical wire + switch pipeline). This
    /// buffer models that latency: words enter with a countdown timer and
    /// are delivered to the destination slave port when the countdown reaches 0.
    ///
    /// Without this, data would teleport instantly between tiles, making stream
    /// delivery ~12x faster than hardware (observed: EMU=645cy vs HW=8185cy
    /// starvation on tile_dmas_writebd).
    pub(super) inter_tile_pipeline: Vec<InFlightWord>,
}

/// A word in flight between adjacent tiles.
///
/// Models the propagation delay through inter-tile stream switch links.
/// The word was popped from the source master port and will be pushed to
/// the destination slave port after `cycles_remaining` reaches 0.
#[derive(Debug, Clone)]
pub(super) struct InFlightWord {
    pub(super) dst_tile_idx: usize,
    pub(super) dst_slave_idx: usize,
    pub(super) data: u32,
    pub(super) tlast: bool,
    /// Cycles until delivery. Decremented each coordinator cycle.
    /// Word is delivered when this reaches 0.
    pub(super) cycles_remaining: u8,
}

impl TileArray {
    /// Create a new tile array for the given architecture configuration.
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        let cols = arch.columns();
        let rows = arch.rows();
        let capacity = (cols as usize) * (rows as usize);

        let mut tiles = Vec::with_capacity(capacity);
        let mut dma_engines = Vec::with_capacity(capacity);

        // Create tiles and DMA engines in column-major order.
        // Per-tile-type params come from ArchConfig (data-driven from mlir-aie).
        for col in 0..cols {
            for row in 0..rows {
                let tile_kind = arch.tile_kind(col, row);
                let params = TileParams {
                    data_memory_size: arch.data_memory_size(tile_kind),
                    num_locks: arch.lock_count(tile_kind),
                    num_bds: arch.dma_bd_count(tile_kind),
                    num_channels: arch.dma_total_channels(tile_kind),
                    dma_s2mm_channels: arch.dma_s2mm_channels(tile_kind),
                    dma_mm2s_channels: arch.dma_mm2s_channels(tile_kind),
                };
                tiles.push(Tile::new(tile_kind, col, row, &params));

                // Create DMA engine with ArchConfig-derived channel/BD/lock counts
                dma_engines.push(DmaEngine::new(
                    col, row, tile_kind,
                    params.dma_s2mm_channels, params.dma_mm2s_channels,
                    params.num_bds, params.num_locks as u8,
                ));
            }
        }

        // Create per-tile control packet reassemblers.
        let ctrl_reassemblers: Vec<_> = (0..capacity)
            .map(|i| {
                let col = (i / rows as usize) as u8;
                let row = (i % rows as usize) as u8;
                crate::device::control_packets::StreamReassembler::new(col, row)
            })
            .collect();

        Self {
            arch, cols, rows, tiles, dma_engines,
            fatal_errors: Vec::new(),
            pending_ctrl_actions: Vec::new(),
            current_cycle: 0,
            ctrl_reassemblers,
            inter_tile_pipeline: Vec::new(),
        }
    }

    /// Create an NPU1 (Phoenix/HawkPoint) array.
    #[inline]
    pub fn npu1() -> Self {
        Self::new(ModelConfig::npu1())
    }

    /// Create an NPU2 (Strix) array.
    #[inline]
    pub fn npu2() -> Self {
        Self::new(ModelConfig::npu2())
    }

    /// Drain accumulated fatal errors from data movement.
    ///
    /// Returns the errors and clears the internal list. The coordinator
    /// calls this after each step to detect impossible-on-hardware
    /// conditions and abort immediately.
    pub fn drain_fatal_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.fatal_errors)
    }

    /// Get the architecture configuration.
    #[inline]
    pub fn arch(&self) -> &dyn ArchConfig {
        self.arch.as_ref()
    }

    /// Get a shared reference to the architecture configuration.
    #[inline]
    pub fn arch_arc(&self) -> Arc<dyn ArchConfig> {
        Arc::clone(&self.arch)
    }

    /// Get number of columns.
    #[inline]
    pub fn cols(&self) -> u8 {
        self.cols
    }

    /// Get number of rows.
    #[inline]
    pub fn rows(&self) -> u8 {
        self.rows
    }

    /// Get tile index from coordinates.
    #[inline]
    pub fn tile_index(&self, col: u8, row: u8) -> usize {
        (col as usize) * (self.rows as usize) + (row as usize)
    }

    /// Get a tile by coordinates.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn get(&self, col: u8, row: u8) -> Option<&Tile> {
        if col < self.cols && row < self.rows {
            Some(&self.tiles[self.tile_index(col, row)])
        } else {
            None
        }
    }

    /// Get a mutable tile by coordinates.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn get_mut(&mut self, col: u8, row: u8) -> Option<&mut Tile> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            Some(&mut self.tiles[idx])
        } else {
            None
        }
    }

    /// Get a tile by coordinates (panics if out of bounds).
    ///
    /// Use this in hot paths where bounds are known valid.
    #[inline]
    pub fn tile(&self, col: u8, row: u8) -> &Tile {
        debug_assert!(col < self.cols && row < self.rows);
        &self.tiles[self.tile_index(col, row)]
    }

    /// Get a mutable tile by coordinates (panics if out of bounds).
    #[inline]
    pub fn tile_mut(&mut self, col: u8, row: u8) -> &mut Tile {
        debug_assert!(col < self.cols && row < self.rows);
        let idx = self.tile_index(col, row);
        &mut self.tiles[idx]
    }

    /// Iterate over all tiles.
    pub fn iter(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter()
    }

    /// Iterate over all tiles mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tile> {
        self.tiles.iter_mut()
    }

    /// Iterate over compute tiles only.
    pub fn compute_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter().filter(|t| t.is_compute())
    }

    /// Iterate over compute tiles mutably.
    pub fn compute_tiles_mut(&mut self) -> impl Iterator<Item = &mut Tile> {
        self.tiles.iter_mut().filter(|t| t.is_compute())
    }

    /// Get all shim tiles.
    pub fn shim_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter().filter(|t| t.is_shim())
    }

    /// Get all memory tiles.
    pub fn mem_tiles(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.iter().filter(|t| t.is_mem())
    }

    /// Count tiles by type.
    pub fn count_by_type(&self) -> (usize, usize, usize) {
        let mut shim = 0;
        let mut mem = 0;
        let mut compute = 0;
        for tile in &self.tiles {
            match tile.tile_kind {
                TileKind::ShimNoc | TileKind::ShimPl => shim += 1,
                TileKind::Mem => mem += 1,
                TileKind::Compute => compute += 1,
            }
        }
        (shim, mem, compute)
    }

    /// Reset all tiles to initial state.
    pub fn reset(&mut self) {
        for tile in &mut self.tiles {
            tile.core.reset();
            for lock in &mut tile.locks {
                lock.value = 0;
            }
            for bd in &mut tile.dma_bds {
                *bd = Default::default();
            }
            for ch in &mut tile.dma_channels {
                *ch = Default::default();
            }
            // Recreate stream switch based on tile type (they have different port configurations)
            tile.stream_switch = match tile.tile_kind {
                TileKind::ShimNoc | TileKind::ShimPl => crate::device::stream_switch::StreamSwitch::new_shim_tile(tile.col),
                TileKind::Mem => crate::device::stream_switch::StreamSwitch::new_mem_tile(tile.col, tile.row),
                TileKind::Compute => crate::device::stream_switch::StreamSwitch::new_compute_tile(tile.col, tile.row),
            };
            // Note: We don't zero memory here for performance
            // Call zero_memory() explicitly if needed
        }

        // Reset all DMA engines
        for engine in &mut self.dma_engines {
            engine.reset();
        }

        // Clear inter-tile pipeline
        self.inter_tile_pipeline.clear();
    }

    /// Zero all tile memory (slow, use only during initialization).
    pub fn zero_memory(&mut self) {
        for tile in &mut self.tiles {
            tile.data_memory_mut().fill(0);
            if let Some(pm) = tile.program_memory_mut() {
                pm.fill(0);
            }
        }
    }

    /// Print array summary.
    pub fn print_summary(&self) {
        println!("Tile Array: {} ({} cols x {} rows)", self.arch.name(), self.cols, self.rows);
        let (shim, mem, compute) = self.count_by_type();
        println!("  Shim tiles: {}", shim);
        println!("  Memory tiles: {}", mem);
        println!("  Compute tiles: {}", compute);
        println!("  Total: {} tiles", self.tiles.len());
    }
}

impl std::fmt::Debug for TileArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileArray")
            .field("arch", &self.arch)
            .field("cols", &self.cols)
            .field("rows", &self.rows)
            .field("tiles", &self.tiles.len())
            .finish()
    }
}
