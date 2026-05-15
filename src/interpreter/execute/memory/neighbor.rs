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

    /// Executing tile's `Tile_Control` isolation bits, snapshotted from
    /// the tile before each core step (see
    /// [`crate::device::tile::isolation`] for the bit layout). A set bit
    /// blocks cross-tile reads and buffered writes in that direction --
    /// `ensure_snapshot` returns early without touching the snapshot,
    /// `get_memory` returns None, and `buffer_write` drops the write.
    /// Mirrors aie-rt's gating where the AIE fabric refuses transit on
    /// an isolated boundary.
    isolation: u8,

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

    /// Test-only counter for `ensure_snapshot` invocations (cache hits AND
    /// misses both increment). Used to assert lazy refresh actually skips
    /// directions the kernel never touches.
    #[cfg(test)]
    pub ensure_snapshot_calls: u32,
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
            isolation: 0,
            snapshots: [None, None, None, None],
            snapshot_gens: [None, None, None, None],
            pending_writes: Vec::new(),
            #[cfg(test)]
            ensure_snapshot_calls: 0,
        }
    }

    /// Refresh the cached `Tile_Control` isolation byte from the executing
    /// tile. Called once per core step before the executor runs so the
    /// gating below picks up software's most recent write to Tile_Control.
    /// See [`crate::device::tile::isolation`] for the bit layout.
    #[inline]
    pub fn set_isolation(&mut self, isolation: u8) {
        self.isolation = isolation;
    }

    /// True if cross-tile transit in `dir` is currently blocked by a
    /// `Tile_Control` isolation bit on the executing tile. `Local` is
    /// never isolated -- it's not a cross-tile direction.
    #[inline]
    fn is_blocked(&self, dir: MemoryQuadrant) -> bool {
        use crate::device::tile::isolation::*;
        let bit = match dir {
            MemoryQuadrant::South => SOUTH,
            MemoryQuadrant::West => WEST,
            MemoryQuadrant::North => NORTH,
            MemoryQuadrant::East => EAST,
            MemoryQuadrant::Local => return false,
        };
        self.isolation & bit != 0
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
        #[cfg(test)]
        {
            self.ensure_snapshot_calls += 1;
        }

        // Tile_Control isolation: short-circuit before touching the
        // snapshot. `get_memory(dir)` returns None for blocked directions
        // regardless, but skipping the clone here avoids burning the
        // allocation when an isolated kernel still walks every quadrant.
        if self.is_blocked(dir) {
            return;
        }

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
    /// Returns None if the neighbor doesn't exist, hasn't been snapshotted,
    /// or transit in `dir` is gated by a `Tile_Control` isolation bit on
    /// the executing tile.
    pub fn get_memory(&self, dir: MemoryQuadrant) -> Option<&[u8]> {
        if self.is_blocked(dir) {
            return None;
        }
        let idx = dir_index(dir)?;
        self.snapshots[idx].as_ref().and_then(|opt| opt.as_deref())
    }

    /// Buffer a cross-tile write for deferred application.
    ///
    /// Drops the write silently when transit in `dir` is gated by a
    /// `Tile_Control` isolation bit. Real HW absorbs the store at the
    /// boundary -- the data never reaches the neighbor.
    pub fn buffer_write(&mut self, dir: MemoryQuadrant, offset: usize, data: &[u8]) {
        if self.is_blocked(dir) {
            return;
        }
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

    // ------------------------------------------------------------------
    // Tile_Control isolation gating
    // ------------------------------------------------------------------

    /// With the SOUTH isolation bit set, ensure_snapshot must NOT touch the
    /// south snapshot slot, and get_memory(South) must return None even if
    /// a stale snapshot is somehow present. Other directions stay open.
    #[test]
    fn isolation_blocks_south_snapshot_and_read() {
        use crate::device::tile::isolation as iso;
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.set_isolation(iso::SOUTH);

        nbr.ensure_snapshot(MemoryQuadrant::South, &device);
        // Snapshot slot must remain untouched -- no clone happened.
        assert!(nbr.snapshots[0].is_none(), "isolated south must not allocate a snapshot");
        assert!(nbr.get_memory(MemoryQuadrant::South).is_none(), "isolated south must not read");

        // Other directions still work -- isolation is per-direction.
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);
        assert!(nbr.get_memory(MemoryQuadrant::West).is_some(), "west must still work");
    }

    /// Buffered writes are silently dropped when the direction is isolated;
    /// pending_writes never grows, so drain_writes is a no-op for that
    /// direction. Mirrors HW where the AIE absorbs the store at the
    /// isolation boundary instead of forwarding it.
    #[test]
    fn isolation_drops_buffered_writes() {
        use crate::device::tile::isolation as iso;
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.set_isolation(iso::WEST);
        nbr.buffer_write(MemoryQuadrant::West, 0x100, &[0xAA, 0xBB]);
        assert!(nbr.pending_writes.is_empty(), "isolated west write must be dropped, not buffered");
        // South still permitted.
        nbr.buffer_write(MemoryQuadrant::South, 0x100, &[0xAA]);
        assert_eq!(nbr.pending_writes.len(), 1);
    }

    /// Stale snapshot cached BEFORE isolation was set must also stop being
    /// readable once the bit goes hot. set_isolation is the only state
    /// flip we need -- get_memory consults the live isolation byte.
    #[test]
    fn isolation_hides_previously_cached_snapshot() {
        use crate::device::tile::isolation as iso;
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 3);
        // Cache a snapshot under "no isolation".
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);
        assert!(nbr.get_memory(MemoryQuadrant::South).is_some());
        // Now the executing tile sets ISOLATE_FROM_SOUTH.
        nbr.set_isolation(iso::SOUTH);
        assert!(
            nbr.get_memory(MemoryQuadrant::South).is_none(),
            "isolation must hide a previously cached snapshot"
        );
    }

    /// All four bits set blocks every cross-tile direction simultaneously.
    /// Local stays accessible (it's not a cross-tile path).
    #[test]
    fn isolation_all_directions_blocks_every_cross_tile_dir() {
        use crate::device::tile::isolation as iso;
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.set_isolation(iso::ALL_DIRECTIONS);
        for dir in [MemoryQuadrant::South, MemoryQuadrant::West, MemoryQuadrant::North, MemoryQuadrant::East]
        {
            nbr.ensure_snapshot(dir, &device);
            assert!(nbr.get_memory(dir).is_none(), "{:?} must be blocked when ALL_DIRECTIONS is set", dir);
        }
    }

    /// Every cardinal bit individually gates ONLY its own direction. Sweeps
    /// the matrix to lock in that the bit -> quadrant mapping in `is_blocked`
    /// is one-to-one and doesn't accidentally collapse two dirs onto the
    /// same bit.
    #[test]
    fn isolation_each_bit_gates_only_its_own_direction() {
        use crate::device::tile::isolation as iso;
        let device = crate::device::DeviceState::new_npu1();
        let cases = [
            (iso::SOUTH, MemoryQuadrant::South),
            (iso::WEST, MemoryQuadrant::West),
            (iso::NORTH, MemoryQuadrant::North),
            (iso::EAST, MemoryQuadrant::East),
        ];
        for (bit, blocked_dir) in cases {
            let mut nbr = NeighborMemory::new(1, 3);
            nbr.set_isolation(bit);
            // Blocked direction: ensure_snapshot is a no-op, get_memory is None.
            nbr.ensure_snapshot(blocked_dir, &device);
            assert!(nbr.get_memory(blocked_dir).is_none(), "bit 0x{bit:X} must block {blocked_dir:?}");
            // All OTHER directions must still snapshot + read.
            for other in
                [MemoryQuadrant::South, MemoryQuadrant::West, MemoryQuadrant::North, MemoryQuadrant::East]
            {
                if other == blocked_dir {
                    continue;
                }
                nbr.ensure_snapshot(other, &device);
                assert!(nbr.get_memory(other).is_some(), "bit 0x{bit:X} must NOT block {other:?}");
            }
        }
    }

    /// Local memory access path is not gated by any isolation bit. AM025's
    /// Tile_Control isolation only governs cross-tile transit; the executing
    /// tile can always read/write its own memory.
    #[test]
    fn isolation_local_memory_unaffected_by_all_directions() {
        use crate::device::tile::isolation as iso;
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.set_isolation(iso::ALL_DIRECTIONS);
        assert!(!nbr.is_blocked(MemoryQuadrant::Local), "Local must never be blocked");
    }

    /// Buffered writes survive end-to-end when their direction is not
    /// isolated. Locks in that the gate is one-shot at buffer_write time
    /// and does not poison the queue for unrelated directions.
    #[test]
    fn isolation_permitted_direction_writes_survive_partial_isolation() {
        use crate::device::tile::isolation as iso;
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.set_isolation(iso::SOUTH | iso::EAST);
        // Two blocked dirs.
        nbr.buffer_write(MemoryQuadrant::South, 0x100, &[0x11]);
        nbr.buffer_write(MemoryQuadrant::East, 0x200, &[0x22]);
        assert!(nbr.pending_writes.is_empty(), "blocked-dir writes must not buffer");

        // Two permitted dirs.
        nbr.buffer_write(MemoryQuadrant::West, 0x300, &[0x33]);
        nbr.buffer_write(MemoryQuadrant::North, 0x400, &[0x44]);
        assert_eq!(nbr.pending_writes.len(), 2, "permitted-dir writes must buffer");
    }
}
