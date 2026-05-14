//! Cross-tile memory access via lazy neighbor snapshots.
//!
//! NeighborMemory provides read-only snapshots of adjacent tiles' data memory,
//! built lazily on first access. Cross-tile writes are buffered and applied
//! after the core step completes, matching hardware behavior.

use crate::device::state::TileLookup;
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

    /// `data_memory_gen` of the neighbor at the time `snapshots[idx]` was taken.
    /// `ensure_snapshot` compares this against the neighbor's current gen and
    /// only re-snapshots on mismatch. None means "not yet snapshotted" -- which
    /// is also implied by `snapshots[idx].is_none()`, but tracking it
    /// separately keeps the gen check next to the gen value.
    snapshot_gens: [Option<u64>; 4],

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
            snapshot_gens: [None, None, None, None],
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
                if self.col > 0 {
                    Some((self.col - 1, self.row))
                } else {
                    None
                }
            }
            MemoryQuadrant::North => Some((self.col, self.row + 1)),
            MemoryQuadrant::East => Some((self.col + 1, self.row)),
            MemoryQuadrant::Local => None,
        }
    }

    /// Snapshot a neighbor tile's data memory if the cached copy is stale.
    ///
    /// Cache hit when the neighbor's `data_memory_gen` matches the gen
    /// recorded at the time of the previous snapshot. Cache miss (re-snapshot)
    /// when no snapshot exists yet, or when the neighbor's data memory has
    /// been mutated since (any `data_memory_mut()` call bumps the gen).
    ///
    /// Cheap to call repeatedly across steps -- this is what makes hoisting
    /// `NeighborMemory` across step boundaries safe.
    ///
    /// `source` is anything that resolves coordinates to tiles -- typically
    /// `&DeviceState` for pre-step refresh and `&NeighborView` for lazy
    /// refresh at a read site that already holds `&mut Tile` for the
    /// executing tile.
    pub fn ensure_snapshot<T: TileLookup>(&mut self, dir: MemoryQuadrant, source: &T) {
        let idx = match dir_index(dir) {
            Some(i) => i,
            None => return, // Local -- no snapshot needed
        };

        let neighbor = self.neighbor_coords(dir).and_then(|(c, r)| source.tile(c, r));
        let current_gen = neighbor.map(|t| t.data_memory_gen());

        // Cache hit: snapshot exists and the neighbor's gen hasn't changed
        // since (None == None handles "neighbor doesn't exist" idempotently).
        if self.snapshots[idx].is_some() && self.snapshot_gens[idx] == current_gen {
            return;
        }

        self.snapshots[idx] = Some(neighbor.map(|tile| tile.data_memory().to_vec()));
        self.snapshot_gens[idx] = current_gen;
    }

    /// Get a reference to a neighbor's data memory snapshot.
    ///
    /// Returns None if the neighbor doesn't exist or hasn't been snapshotted.
    pub fn get_memory(&self, dir: MemoryQuadrant) -> Option<&[u8]> {
        let idx = dir_index(dir)?;
        self.snapshots[idx].as_ref().and_then(|opt| opt.as_deref())
    }

    /// Buffer a cross-tile write for deferred application.
    pub fn buffer_write(&mut self, dir: MemoryQuadrant, offset: usize, data: &[u8]) {
        self.pending_writes.push((dir, offset, data.to_vec()));
    }

    /// Drain buffered cross-tile writes to the device.
    ///
    /// Resolves each direction to the target tile and writes the data.
    /// Called after the core step completes. Drains the `pending_writes`
    /// vec but leaves cached snapshots intact -- they remain valid across
    /// steps as long as their gen-counters match (see `ensure_snapshot`).
    pub fn drain_writes(&mut self, device: &mut crate::device::DeviceState) {
        let col = self.col;
        let row = self.row;
        for (dir, offset, data) in self.pending_writes.drain(..) {
            let coords = match dir {
                MemoryQuadrant::South => {
                    if row > SHIM_ROW as usize {
                        Some((col, row - 1))
                    } else {
                        None
                    }
                }
                MemoryQuadrant::West => {
                    if col > 0 {
                        Some((col - 1, row))
                    } else {
                        None
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// When the neighbor's data memory hasn't changed since the last snapshot,
    /// `ensure_snapshot` must be a no-op (no re-allocation). We detect this by
    /// poking a sentinel into the cached snapshot; if the snapshot survives a
    /// second `ensure_snapshot` call, no reclone happened.
    #[test]
    fn ensure_snapshot_is_noop_when_neighbor_unchanged() {
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);

        // Sentinel into the cached snapshot (test-only, in-module access).
        nbr.snapshots[0].as_mut().unwrap().as_mut().unwrap()[0x100] = 0xCD;

        // No underlying mutation -> must hit cache and preserve the sentinel.
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);
        assert_eq!(
            nbr.get_memory(MemoryQuadrant::South).unwrap()[0x100],
            0xCD,
            "ensure_snapshot must not re-clone when neighbor's data_memory_gen is unchanged"
        );
    }

    /// After taking a snapshot, if the neighbor's data memory is mutated,
    /// the next ensure_snapshot must re-snapshot so reads see the new data.
    /// Currently FAILS: the is_some() guard returns the stale snapshot.
    #[test]
    fn ensure_snapshot_picks_up_neighbor_write() {
        let mut device = crate::device::DeviceState::new_npu1();

        // Initial state: south neighbor (1, 2) has zero at offset 0x100.
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);
        assert_eq!(nbr.get_memory(MemoryQuadrant::South).unwrap()[0x100], 0x00);

        // Mutate the neighbor's data memory directly.
        device.tile_mut(1, 2).unwrap().data_memory_mut()[0x100] = 0xAB;

        // Re-snapshot must pick up the new value.
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);
        assert_eq!(
            nbr.get_memory(MemoryQuadrant::South).unwrap()[0x100],
            0xAB,
            "snapshot must reflect post-write neighbor data"
        );
    }
}
