//! Device state and CDO application.
//!
//! This module connects the parser (CDO commands) to the device model
//! (tile array). It applies configuration commands to build up the
//! device state that will be emulated.
//!
//! # CDO Application Process
//!
//! 1. Create a `DeviceState` with a `TileArray`
//! 2. Call `apply_cdo()` to process all CDO commands
//! 3. Device state is now configured and ready for emulation
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::device::DeviceState;
//! use xdna_emu::parser::Cdo;
//!
//! let mut state = DeviceState::new_npu1();
//! state.apply_cdo(&cdo)?;
//! state.print_summary();
//! ```

mod cdo;
mod compute;
mod ctrl_access;
mod dispatch;
mod effects;
mod memtile;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use xdna_archspec::runtime::{ArchConfig, ModelConfig};
use super::array::TileArray;
use super::async_errors::AsyncErrorSink;
use super::context::{Context, ContextId, DEFAULT_CONTEXT};
use super::registers::TileAddress;
use super::registers::{subsystem_from_offset, tile_kind_from_row};
use super::tile::Tile;
use super::tdr::TdrDetector;
use super::regdb;
use xdna_archspec::types::{SubsystemKind, TileKind};
use crate::parser::cdo::{Cdo, CdoRaw};

/// Statistics about CDO application.
#[derive(Debug, Default)]
pub struct CdoStats {
    /// Total commands processed
    pub commands: usize,
    /// WRITE commands
    pub writes: usize,
    /// MASK_WRITE commands
    pub mask_writes: usize,
    /// DMA_WRITE commands (memory/program loads)
    pub dma_writes: usize,
    /// NOP commands (skipped)
    pub nops: usize,
    /// Unknown/unhandled commands
    pub unknown: usize,
    /// Bytes written to data memory
    pub data_bytes: usize,
    /// Bytes written to program memory
    pub program_bytes: usize,
}

/// Device state wrapping a tile array.
///
/// Provides methods to apply CDO commands and query device configuration.
pub struct DeviceState {
    /// The tile array
    pub array: TileArray,
    /// Statistics from last CDO application
    pub stats: CdoStats,
    /// Pending core enable/disable events from Core_Control register writes.
    ///
    /// When `write_core_register` or `mask_write_core_register` changes the
    /// enable bit of Core_Control (offset 0x32000), it pushes (col, row, enabled)
    /// here. The coordinator drains this each cycle to sync the engine's internal
    /// core state, matching how real hardware immediately reacts to the register
    /// write regardless of source (CDO, NPU instruction, or control packet).
    pub(crate) pending_core_enables: Vec<(u8, u8, bool)>,
    /// Column offset applied to all CDO operations.
    ///
    /// CDO streams encode logical (partition-relative) tile columns
    /// starting at 0. The real xdna-driver allocates a physical
    /// `start_col` from `aie_partition.start_columns` (typically 1, since
    /// col 0 is reserved for the shim DMA host channels) and rebases
    /// every CDO write to that column. Setting this field replicates
    /// that rebase: every `DeviceOp` that carries a `tile.col` is shifted
    /// by `start_col` before address encoding, so the emulator's
    /// trace-packet headers and per-tile state line up with the real
    /// hardware's physical placement.
    pub start_col: u8,
    /// Tier B async-error subsystem: cache + per-column rings + drain queue.
    /// Populated from `state::effects::apply_tile_local_effects` when an
    /// error-category event is generated.
    pub async_errors: AsyncErrorSink,
    /// Per-context state. Single context (DEFAULT_CONTEXT) today; multi-context
    /// expansion is storage-only -- all APIs already key by ContextId.
    pub contexts: Vec<Context>,
    /// Per-context TDR classifier. Parallel index to `contexts`.
    pub tdr_detectors: Vec<TdrDetector>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetContextError {
    InvalidContextId,
}

impl DeviceState {
    /// Create a new device state for the given architecture configuration.
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        let array = TileArray::new(arch);
        let num_cols = array.cols() as usize;
        let contexts = vec![Context::new(DEFAULT_CONTEXT)];
        let tdr_detectors = vec![TdrDetector::new(DEFAULT_CONTEXT)];
        Self {
            array,
            stats: CdoStats::default(),
            pending_core_enables: Vec::new(),
            start_col: 0,
            async_errors: AsyncErrorSink::new(num_cols),
            contexts,
            tdr_detectors,
        }
    }

    /// Set the partition's physical start column.
    ///
    /// Called once before applying CDOs, with the value the xdna-driver
    /// would have chosen for the partition (typically `start_columns[0]`).
    /// Subsequent `apply_device_op` calls shift every operation's
    /// `tile.col` by this amount.
    pub fn set_start_col(&mut self, start_col: u8) {
        self.start_col = start_col;
    }

    /// Reset the given context to Connected and clear its Tier B sink slot.
    ///
    /// Idempotent on an already-Connected context. Returns an error if the
    /// context_id is out of range.
    pub fn reset_context(&mut self, context_id: ContextId) -> Result<(), ResetContextError> {
        let idx = context_id.0 as usize;
        let ctx = self.contexts.get_mut(idx).ok_or(ResetContextError::InvalidContextId)?;
        ctx.mark_connected();
        // Tier B: clear async errors for this context. Today AsyncErrorSink
        // is global; multi-context spec will give it per-context slots.
        self.async_errors.clear();
        // The engine.reset_for_new_context() call happens at the FFI layer
        // (xdna_emu_reset_context); this method only touches device-side state.
        Ok(())
    }

    /// Create an NPU1 device state.
    #[inline]
    pub fn new_npu1() -> Self {
        Self::new(ModelConfig::npu1())
    }

    /// Create an NPU2 device state.
    #[inline]
    pub fn new_npu2() -> Self {
        Self::new(ModelConfig::npu2())
    }

    /// Print a summary of the device state.
    pub fn print_summary(&self) {
        println!("Device State Summary");
        println!("====================");
        self.array.print_summary();

        println!();
        println!("CDO Application Stats:");
        println!("  Commands: {}", self.stats.commands);
        println!("  Writes: {}", self.stats.writes);
        println!("  MaskWrites: {}", self.stats.mask_writes);
        println!("  DmaWrites: {}", self.stats.dma_writes);
        println!("  NOPs: {}", self.stats.nops);
        println!("  Unknown: {}", self.stats.unknown);
        println!("  Data bytes: {}", self.stats.data_bytes);
        println!("  Program bytes: {}", self.stats.program_bytes);

        // Show configured tiles
        println!();
        println!("Configured Tiles:");
        for tile in self.array.iter() {
            let has_code = tile.program_memory().map(|pm| pm.iter().any(|&b| b != 0)).unwrap_or(false);
            let has_locks = tile.locks.iter().any(|l| l.value != 0);
            let has_bds = tile.dma_bds.iter().any(|bd| bd.is_valid());

            if has_code || has_locks || has_bds || tile.core.enabled {
                print!("  tile({},{}):", tile.col, tile.row);
                if tile.core.enabled {
                    print!(" core-enabled");
                }
                if has_code {
                    print!(" has-code");
                }
                if has_locks {
                    let active = tile.locks.iter().filter(|l| l.value != 0).count();
                    print!(" locks:{}", active);
                }
                if has_bds {
                    let active = tile.dma_bds.iter().filter(|bd| bd.is_valid()).count();
                    print!(" bds:{}", active);
                }
                println!();
            }
        }
    }

    /// Get count of enabled cores.
    pub fn enabled_cores(&self) -> usize {
        self.array.compute_tiles().filter(|t| t.core.enabled).count()
    }

    /// Get count of tiles with program code.
    pub fn tiles_with_code(&self) -> usize {
        self.array
            .compute_tiles()
            .filter(|t| t.program_memory().map(|pm| pm.iter().any(|&b| b != 0)).unwrap_or(false))
            .count()
    }

    /// Get number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.array.cols() as usize
    }

    /// Get number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.array.rows() as usize
    }

    /// Get the architecture name (e.g. "AIE2", "AIE2P").
    #[inline]
    pub fn arch_name(&self) -> &str {
        self.array.arch().name()
    }

    /// Get a tile by coordinates, or None if out of bounds.
    #[inline]
    pub fn tile(&self, col: usize, row: usize) -> Option<&Tile> {
        if col < self.cols() && row < self.rows() {
            Some(self.array.tile(col as u8, row as u8))
        } else {
            None
        }
    }

    /// Get a mutable tile by coordinates, or None if out of bounds.
    #[inline]
    pub fn tile_mut(&mut self, col: usize, row: usize) -> Option<&mut Tile> {
        if col < self.cols() && row < self.rows() {
            Some(self.array.tile_mut(col as u8, row as u8))
        } else {
            None
        }
    }

    /// Split the tile array so the executing tile can be borrowed mutably
    /// while the neighbors stay reachable through a `NeighborView`.
    ///
    /// Uses `slice::split_at_mut` + `split_first_mut` -- no unsafe.
    /// Returns `None` for out-of-bounds coordinates.
    ///
    /// # Why this exists
    ///
    /// The interpreter's per-step body needs `&mut Tile` for the executing
    /// tile (to step the core, drain trace events, etc.) AND read access to
    /// neighbor tiles' data memory (to refresh the cross-tile snapshot
    /// cache). Holding `&mut Tile` and `&DeviceState` simultaneously is a
    /// borrow conflict; this split disentangles them at one well-defined
    /// boundary so the rest of the interpreter doesn't have to.
    pub fn split_tile_mut(&mut self, col: usize, row: usize) -> Option<(&mut Tile, NeighborView<'_>)> {
        let cols = self.cols();
        let rows = self.rows();
        if col >= cols || row >= rows {
            return None;
        }
        let idx = col * rows + row;
        let tiles: &mut [Tile] = &mut self.array.tiles;
        let (left, rest) = tiles.split_at_mut(idx);
        let (own, right) = rest.split_first_mut()?;
        Some((own, NeighborView { left, right, own_idx: idx, cols, rows }))
    }
}

/// Read-only access to a tile by coordinates.
///
/// Implemented for both `DeviceState` (full array) and `NeighborView`
/// (everything except the executing tile). `NeighborMemory::ensure_snapshot`
/// is generic over this trait so the same body works in both contexts:
/// pre-step refresh against the whole device, and lazy refresh at a read
/// site that already holds `&mut Tile` for the executing tile.
pub trait TileLookup {
    fn tile(&self, col: usize, row: usize) -> Option<&Tile>;
}

impl TileLookup for DeviceState {
    #[inline]
    fn tile(&self, col: usize, row: usize) -> Option<&Tile> {
        DeviceState::tile(self, col, row)
    }
}

/// Read-through view of every tile EXCEPT the one at `own_idx`.
///
/// Produced by `DeviceState::split_tile_mut`. The `own_idx` slot is a hole
/// (`tile()` returns `None` there) because the caller has `&mut Tile` for
/// that position from the same split. All other in-bounds coordinates
/// resolve normally.
pub struct NeighborView<'a> {
    left: &'a [Tile],
    right: &'a [Tile],
    own_idx: usize,
    cols: usize,
    rows: usize,
}

impl<'a> NeighborView<'a> {
    /// Look up a tile by coordinates.
    ///
    /// Returns `None` for out-of-bounds coordinates and for the hole at
    /// `own_idx` (the executing tile is reachable via the `&mut Tile`
    /// returned alongside this view, not through the view itself).
    #[inline]
    pub fn tile(&self, col: usize, row: usize) -> Option<&Tile> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        let idx = col * self.rows + row;
        if idx < self.own_idx {
            Some(&self.left[idx])
        } else if idx == self.own_idx {
            None
        } else {
            Some(&self.right[idx - self.own_idx - 1])
        }
    }
}

impl<'a> TileLookup for NeighborView<'a> {
    #[inline]
    fn tile(&self, col: usize, row: usize) -> Option<&Tile> {
        NeighborView::tile(self, col, row)
    }
}

#[cfg(test)]
mod async_errors_integration_tests {
    use super::*;
    use xdna_archspec::aie2::async_errors::AieErrorOrigin;

    #[test]
    fn device_state_exposes_async_error_sink() {
        let dev = DeviceState::new_npu1();
        assert!(dev.async_errors.last_cache().is_none());
    }

    #[test]
    fn record_error_through_sink_reaches_cache() {
        let mut dev = DeviceState::new_npu1();
        dev.async_errors.record_error(1, 2, AieErrorOrigin::Core, 69, 10_000);
        assert!(dev.async_errors.last_cache().is_some());
    }
}
