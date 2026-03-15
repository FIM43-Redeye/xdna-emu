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
mod dispatch;
mod effects;
mod memtile;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use super::arch_config::{ArchConfig, ModelConfig};
use super::array::TileArray;
use super::registers::TileAddress;
use super::registers::{subsystem_from_offset, tile_kind_from_row};
use super::tile::{Tile, TileType};
use super::regdb;
use crate::archspec::types::{SubsystemKind, TileKind};
use crate::parser::cdo::{Cdo, CdoCommand};

/// Sign-extend a lock value from a register u32 to i8.
///
/// Delegates to DeviceRegLayout::sign_extend_lock_value() which derives the
/// field width from the AM025 register database (6 bits for AIE2).
fn sign_extend_lock_value(reg_layout: &regdb::DeviceRegLayout, raw: u32) -> i8 {
    reg_layout.sign_extend_lock_value(raw)
}

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
}

impl DeviceState {
    /// Create a new device state for the given architecture configuration.
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        Self {
            array: TileArray::new(arch),
            stats: CdoStats::default(),
            pending_core_enables: Vec::new(),
        }
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
            let has_code = tile.program_memory()
                .map(|pm| pm.iter().any(|&b| b != 0))
                .unwrap_or(false);
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
        self.array.compute_tiles()
            .filter(|t| {
                t.program_memory()
                    .map(|pm| pm.iter().any(|&b| b != 0))
                    .unwrap_or(false)
            })
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
}
