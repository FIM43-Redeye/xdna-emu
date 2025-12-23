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
}

impl TileArray {
    /// Create a new tile array for the given architecture.
    pub fn new(arch: AieArch) -> Self {
        let cols = arch.columns();
        let rows = arch.rows();

        let mut tiles = Vec::with_capacity((cols as usize) * (rows as usize));

        // Create tiles in column-major order
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
            }
        }

        Self { arch, cols, rows, tiles }
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
}
