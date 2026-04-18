//! Cross-tile memory access via lazy neighbor snapshots.
//!
//! NeighborMemory provides read-only snapshots of adjacent tiles' data memory,
//! built lazily on first access. Cross-tile writes are buffered and applied
//! after the core step completes, matching hardware behavior.

use crate::interpreter::timing::MemoryQuadrant;
use xdna_archspec::aie2::SHIM_ROW;

/// Read-only snapshots of neighbor tile data memory for cross-tile access.
///
/// Built lazily before each core step. Neighbor memory is only cloned on
/// the first cross-tile access, so cores that only touch local memory
/// pay zero allocation cost.
///
/// Cross-tile writes are buffered and applied after the core step completes,
/// matching hardware behavior where cross-tile writes have higher latency
/// and become visible on the next cycle.
///
/// # Cardinal Direction Mapping
///
/// | CardDir | Direction | Neighbor offset        | AIE2 behavior |
/// |---------|-----------|------------------------|---------------|
/// | 4       | South     | (col, row-1)           | Cross-tile    |
/// | 5       | West      | (col-1, row)           | Cross-tile    |
/// | 6       | North     | (col, row+1)           | Cross-tile    |
/// | 7       | East      | (col+1, row) or LOCAL  | Local (AIE2)  |
///
/// On AIE2 (IsCheckerBoard=0), East is always local -- `decode_data_address`
/// maps CardDir 7 to `MemoryQuadrant::Local`, so the East slot is unused.
/// On checkerboard architectures (AIE1), East may be a real cross-tile
/// neighbor depending on row parity.
pub struct NeighborMemory {
    /// Source tile coordinates (needed for resolving neighbors).
    col: usize,
    row: usize,

    /// Neighbor data memory snapshots indexed by cardinal direction:
    /// 0=South, 1=West, 2=North, 3=East.
    /// None until first access (lazy clone). Inner None if neighbor doesn't exist.
    snapshots: [Option<Option<Vec<u8>>>; 4],

    /// Buffered cross-tile writes: (direction, offset_within_tile, data).
    /// Applied after core step completes.
    pub pending_writes: Vec<(MemoryQuadrant, usize, Vec<u8>)>,
}

/// Map a cardinal direction to its snapshot array index.
///
/// Returns None for `Local` (no snapshot needed -- use own tile memory).
fn dir_index(dir: MemoryQuadrant) -> Option<usize> {
    match dir {
        MemoryQuadrant::South => Some(0),
        MemoryQuadrant::West => Some(1),
        MemoryQuadrant::North => Some(2),
        MemoryQuadrant::East => Some(3),
        MemoryQuadrant::Local => None,
    }
}

impl NeighborMemory {
    /// Create a new NeighborMemory for the core at (col, row).
    ///
    /// Does NOT clone any neighbor memory yet -- snapshots are lazy.
    pub fn new(col: usize, row: usize) -> Self {
        Self {
            col,
            row,
            snapshots: [None, None, None, None],
            pending_writes: Vec::new(),
        }
    }

    /// Resolve a cardinal direction to the neighbor tile coordinates.
    ///
    /// Returns None if the neighbor is outside the array bounds or if
    /// the direction is `Local`.
    fn neighbor_coords(&self, dir: MemoryQuadrant) -> Option<(usize, usize)> {
        match dir {
            MemoryQuadrant::South => {
                // Shim row (row == SHIM_ROW) has no south neighbor.
                if self.row > SHIM_ROW as usize {
                    Some((self.col, self.row - 1))
                } else {
                    None
                }
            }
            MemoryQuadrant::West => {
                if self.col > 0 { Some((self.col - 1, self.row)) } else { None }
            }
            MemoryQuadrant::North => Some((self.col, self.row + 1)),
            MemoryQuadrant::East => Some((self.col + 1, self.row)),
            MemoryQuadrant::Local => None,
        }
    }

    /// Lazily snapshot a neighbor tile's data memory.
    ///
    /// Called before execution for each direction that might be accessed.
    /// The `device` reference is only needed for this initial clone.
    pub fn ensure_snapshot(&mut self, dir: MemoryQuadrant, device: &crate::device::DeviceState) {
        let idx = match dir_index(dir) {
            Some(i) => i,
            None => return, // Local -- no snapshot needed
        };
        if self.snapshots[idx].is_some() {
            return; // Already loaded
        }

        let snapshot = self.neighbor_coords(dir)
            .and_then(|(c, r)| device.tile(c, r))
            .map(|tile| tile.data_memory().to_vec());

        self.snapshots[idx] = Some(snapshot);
    }

    /// Get a reference to a neighbor's data memory snapshot.
    ///
    /// Returns None if the neighbor doesn't exist or hasn't been snapshotted.
    pub fn get_memory(&self, dir: MemoryQuadrant) -> Option<&[u8]> {
        let idx = dir_index(dir)?;
        self.snapshots[idx]
            .as_ref()
            .and_then(|opt| opt.as_deref())
    }

    /// Buffer a cross-tile write for deferred application.
    pub fn buffer_write(&mut self, dir: MemoryQuadrant, offset: usize, data: &[u8]) {
        self.pending_writes.push((dir, offset, data.to_vec()));
    }

    /// Apply all buffered cross-tile writes to the device.
    ///
    /// Resolves each direction to the target tile and writes the data.
    /// Called after the core step completes.
    pub fn apply_writes(self, device: &mut crate::device::DeviceState) {
        let col = self.col;
        let row = self.row;
        for (dir, offset, data) in self.pending_writes {
            let coords = match dir {
                MemoryQuadrant::South => if row > SHIM_ROW as usize {
                    Some((col, row - 1))
                } else {
                    None
                },
                MemoryQuadrant::West => if col > 0 { Some((col - 1, row)) } else { None },
                MemoryQuadrant::North => Some((col, row + 1)),
                MemoryQuadrant::East => Some((col + 1, row)),
                MemoryQuadrant::Local => None,
            };
            if let Some((c, r)) = coords {
                if let Some(tile) = device.tile_mut(c, r) {
                    let mem = tile.data_memory_mut();
                    if offset + data.len() <= mem.len() {
                        mem[offset..offset + data.len()].copy_from_slice(&data);
                    }
                }
            }
        }
    }

    /// Check if there are any pending cross-tile writes.
    pub fn has_pending_writes(&self) -> bool {
        !self.pending_writes.is_empty()
    }
}
