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

use super::dma::{DmaEngine, DmaResult};
use super::host_memory::HostMemory;
use super::tile::{Tile, TileType};
use super::AieArch;

/// Maximum supported columns
pub const MAX_COLS: usize = 9;

/// Maximum supported rows
pub const MAX_ROWS: usize = 6;

/// AIE tile array.
///
/// Stores tiles in row-major order for cache-friendly column iteration.
pub struct TileArray {
    /// Architecture variant
    arch: AieArch,

    /// Number of columns (including shim column)
    cols: u8,

    /// Number of rows
    rows: u8,

    /// Tiles stored in flat array: tiles[col * rows + row]
    /// Using Vec because tile count varies by device
    tiles: Vec<Tile>,

    /// DMA engines stored in parallel with tiles.
    /// Each tile has exactly one DMA engine.
    dma_engines: Vec<DmaEngine>,
}

impl TileArray {
    /// Create a new tile array for the given architecture.
    pub fn new(arch: AieArch) -> Self {
        let cols = arch.columns();
        let rows = arch.rows();
        let capacity = (cols as usize) * (rows as usize);

        let mut tiles = Vec::with_capacity(capacity);
        let mut dma_engines = Vec::with_capacity(capacity);

        // Create tiles and DMA engines in column-major order
        for col in 0..cols {
            for row in 0..rows {
                let tile_type = if row == 0 {
                    TileType::Shim
                } else if row == 1 {
                    TileType::MemTile
                } else {
                    TileType::Compute
                };
                tiles.push(Tile::new(tile_type, col, row));

                // Create appropriate DMA engine for tile type
                let engine = match tile_type {
                    TileType::MemTile => DmaEngine::new_mem_tile(col, row),
                    _ => DmaEngine::new_compute_tile(col, row),
                };
                dma_engines.push(engine);
            }
        }

        Self { arch, cols, rows, tiles, dma_engines }
    }

    /// Create an NPU1 (Phoenix/HawkPoint) array.
    #[inline]
    pub fn npu1() -> Self {
        Self::new(AieArch::Aie2)
    }

    /// Create an NPU2 (Strix) array.
    #[inline]
    pub fn npu2() -> Self {
        Self::new(AieArch::Aie2P)
    }

    /// Get the architecture.
    #[inline]
    pub fn arch(&self) -> AieArch {
        self.arch
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
    fn tile_index(&self, col: u8, row: u8) -> usize {
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
        self.tiles.iter().filter(|t| t.is_mem_tile())
    }

    // === DMA Engine Access ===

    /// Get the DMA engine for a tile.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn dma_engine(&self, col: u8, row: u8) -> Option<&DmaEngine> {
        if col < self.cols && row < self.rows {
            Some(&self.dma_engines[self.tile_index(col, row)])
        } else {
            None
        }
    }

    /// Get the mutable DMA engine for a tile.
    ///
    /// Returns None if coordinates are out of bounds.
    #[inline]
    pub fn dma_engine_mut(&mut self, col: u8, row: u8) -> Option<&mut DmaEngine> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            Some(&mut self.dma_engines[idx])
        } else {
            None
        }
    }

    /// Get tile and DMA engine together (for operations that need both).
    ///
    /// Returns separate references to allow independent mutation.
    pub fn tile_and_dma(&mut self, col: u8, row: u8) -> Option<(&mut Tile, &mut DmaEngine)> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            // Safety: We're returning references to different arrays
            Some((&mut self.tiles[idx], &mut self.dma_engines[idx]))
        } else {
            None
        }
    }

    /// Step the DMA engine for a specific tile.
    ///
    /// This advances the DMA transfer state by one cycle.
    pub fn step_dma(&mut self, col: u8, row: u8, host_memory: &mut HostMemory) -> Option<DmaResult> {
        if col < self.cols && row < self.rows {
            let idx = self.tile_index(col, row);
            let (tile, engine) = (&mut self.tiles[idx], &mut self.dma_engines[idx]);
            Some(engine.step(tile, host_memory))
        } else {
            None
        }
    }

    /// Step all DMA engines.
    ///
    /// Returns true if any DMA engine is still active.
    pub fn step_all_dma(&mut self, host_memory: &mut HostMemory) -> bool {
        let mut any_active = false;
        for i in 0..self.tiles.len() {
            let result = self.dma_engines[i].step(&mut self.tiles[i], host_memory);
            if matches!(result, DmaResult::InProgress | DmaResult::WaitingForLock(_)) {
                any_active = true;
            }
        }
        any_active
    }

    /// Check if any DMA engine has active transfers.
    pub fn any_dma_active(&self) -> bool {
        self.dma_engines.iter().any(|e| e.any_channel_active())
    }

    /// Count tiles by type.
    pub fn count_by_type(&self) -> (usize, usize, usize) {
        let mut shim = 0;
        let mut mem = 0;
        let mut compute = 0;
        for tile in &self.tiles {
            match tile.tile_type {
                TileType::Shim => shim += 1,
                TileType::MemTile => mem += 1,
                TileType::Compute => compute += 1,
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
            tile.stream_switch = Default::default();
            // Note: We don't zero memory here for performance
            // Call zero_memory() explicitly if needed
        }

        // Reset all DMA engines
        for engine in &mut self.dma_engines {
            engine.reset();
        }
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
        println!("Tile Array: {} ({} cols × {} rows)", self.arch, self.cols, self.rows);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation_npu1() {
        let array = TileArray::npu1();
        assert_eq!(array.cols(), 5);
        assert_eq!(array.rows(), 6);
        assert_eq!(array.tiles.len(), 30);

        let (shim, mem, compute) = array.count_by_type();
        assert_eq!(shim, 5);  // Row 0, all columns
        assert_eq!(mem, 5);   // Row 1, all columns
        assert_eq!(compute, 20); // Rows 2-5, all columns
    }

    #[test]
    fn test_tile_access() {
        let mut array = TileArray::npu1();

        // Access by coordinates
        let tile = array.get(1, 2).unwrap();
        assert_eq!(tile.col, 1);
        assert_eq!(tile.row, 2);
        assert!(tile.is_compute());

        // Modify tile
        let tile = array.get_mut(1, 2).unwrap();
        tile.core.pc = 0x100;
        assert_eq!(array.tile(1, 2).core.pc, 0x100);
    }

    #[test]
    fn test_out_of_bounds() {
        let array = TileArray::npu1();
        assert!(array.get(10, 0).is_none());
        assert!(array.get(0, 10).is_none());
    }

    #[test]
    fn test_tile_types() {
        let array = TileArray::npu1();

        // Shim tiles at row 0
        assert!(array.tile(0, 0).is_shim());
        assert!(array.tile(4, 0).is_shim());

        // Mem tiles at row 1
        assert!(array.tile(0, 1).is_mem_tile());
        assert!(array.tile(4, 1).is_mem_tile());

        // Compute tiles at rows 2-5
        assert!(array.tile(0, 2).is_compute());
        assert!(array.tile(4, 5).is_compute());
    }

    #[test]
    fn test_compute_tile_iteration() {
        let array = TileArray::npu1();
        let compute_count = array.compute_tiles().count();
        assert_eq!(compute_count, 20); // 5 cols × 4 rows (2-5)
    }

    #[test]
    fn test_reset() {
        let mut array = TileArray::npu1();

        // Modify some state
        array.tile_mut(1, 2).core.pc = 0x1000;
        array.tile_mut(1, 2).locks[0].value = 5;

        // Reset
        array.reset();

        // Verify reset
        assert_eq!(array.tile(1, 2).core.pc, 0);
        assert_eq!(array.tile(1, 2).locks[0].value, 0);
    }

    // === DMA Engine Integration Tests ===

    #[test]
    fn test_dma_engine_creation() {
        let array = TileArray::npu1();

        // Each tile should have a DMA engine
        assert_eq!(array.dma_engines.len(), 30);

        // Compute tile (row 2+) should have 4 channels
        let engine = array.dma_engine(1, 2).unwrap();
        assert_eq!(engine.num_channels(), 4);

        // Memory tile (row 1) should have 12 channels
        let engine = array.dma_engine(1, 1).unwrap();
        assert_eq!(engine.num_channels(), 12);
    }

    #[test]
    fn test_dma_engine_access() {
        let mut array = TileArray::npu1();

        // Get mutable engine and configure it
        let engine = array.dma_engine_mut(2, 3).unwrap();
        assert_eq!(engine.col, 2);
        assert_eq!(engine.row, 3);
    }

    #[test]
    fn test_tile_and_dma() {
        let mut array = TileArray::npu1();

        // Get both tile and DMA engine
        let (tile, engine) = array.tile_and_dma(3, 4).unwrap();
        assert_eq!(tile.col, 3);
        assert_eq!(tile.row, 4);
        assert_eq!(engine.col, 3);
        assert_eq!(engine.row, 4);
    }

    #[test]
    fn test_no_active_dma_initially() {
        let array = TileArray::npu1();
        assert!(!array.any_dma_active());
    }

    #[test]
    fn test_dma_reset() {
        use crate::device::dma::BdConfig;

        let mut array = TileArray::npu1();

        // Configure and start a DMA transfer
        let engine = array.dma_engine_mut(1, 2).unwrap();
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();
        engine.start_channel(0, 0).unwrap();
        assert!(engine.channel_active(0));

        // Reset should clear it
        array.reset();
        let engine = array.dma_engine(1, 2).unwrap();
        assert!(!engine.any_channel_active());
    }
}
