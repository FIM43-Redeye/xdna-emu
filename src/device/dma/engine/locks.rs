//! Lock resolution and submission for DMA engine.

use super::*;

impl DmaEngine {
    /// Map a channel index to a LockRequestor for arbiter submission.
    pub fn channel_requestor(&self, ch_idx: u8) -> crate::device::tile::LockRequestor {
        use crate::device::tile::LockRequestor;
        let ct = self.channel_type(ch_idx);
        match ct {
            ChannelType::S2MM => LockRequestor::DmaS2mm(ch_idx),
            ChannelType::MM2S => LockRequestor::DmaMm2s(ch_idx - self.s2mm_count as u8),
        }
    }

    /// Pre-step pass: submit all pending lock requests to tile arbiters.
    ///
    /// Scans channels for those needing lock operations:
    /// - `AcquiringLock { acquired: false }` -> submit acquire request
    /// - `ReleasingLock { cycles_remaining: 1 }` -> submit release request
    ///
    /// Called before arbiter resolution. After resolution, `step()` checks
    /// arbiter results to determine which operations succeeded.
    pub fn submit_lock_requests(&self, tile: &mut Tile, neighbors: &mut NeighborTiles<'_>) {
        for ch_idx in 0..self.channels.len() {
            match &self.channels[ch_idx].fsm {
                ChannelFsm::AcquiringLock { lock_id, acquired: false, transfer, .. } => {
                    self.submit_acquire_request(*lock_id, transfer, tile, neighbors, ch_idx as u8);
                }
                ChannelFsm::ReleasingLock { lock_id, release_value, cycles_remaining, .. }
                    if *cycles_remaining <= 1 =>
                {
                    self.submit_release_request(*lock_id, *release_value, tile, neighbors, ch_idx as u8);
                }
                _ => {}
            }
        }
    }

    /// Submit an acquire request to the appropriate tile's arbiter.
    fn submit_acquire_request(
        &self,
        lock_id: u8,
        transfer: &Transfer,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        ch_idx: u8,
    ) {
        use crate::device::tile::LockRequest;

        let lock_target = match self.resolve_lock_id(lock_id) {
            Some(target) => target,
            None => return,
        };

        let (target_tile, local_id): (&mut Tile, u8) = match lock_target {
            LockTarget::Own(id) => (tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref_mut() {
                Some(west) => (west, id),
                None => return,
            },
            LockTarget::East(id) => match neighbors.east.as_deref_mut() {
                Some(east) => (east, id),
                None => return,
            },
        };

        let acquire_value = transfer.acquire_value;
        let (expected, delta, equal_mode) = if acquire_value < 0 {
            ((-acquire_value) as i8, acquire_value, false)
        } else if acquire_value > 0 {
            (acquire_value as i8, -acquire_value, true)
        } else {
            (1i8, -1i8, false)
        };

        target_tile.submit_lock_request(LockRequest {
            requestor: self.channel_requestor(ch_idx),
            lock_id: local_id as usize,
            is_acquire: true,
            expected,
            delta,
            equal_mode,
        });
    }

    /// Apply a release directly to the target lock, bypassing the arbiter.
    ///
    /// Used by `begin_completion` to pipeline the BD's release with its
    /// final data cycle: releases are non-blocking and never have a
    /// precondition, so the arbiter only added a cycle of latency without
    /// changing the outcome. Applying inline matches HW behavior (release
    /// completes in the cycle the BD's last data word transfers).
    ///
    /// Same-cycle acquires that were submitted earlier this cycle won't see
    /// this release until next cycle's resolve -- a small ordering shift vs.
    /// the arbiter path, but consistent with HW (release happens late in
    /// the cycle, after arbiter resolution).
    pub(super) fn apply_lock_release_direct(
        &mut self,
        lock_id: u8,
        release_value: i8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) {
        let lock_target = match self.resolve_lock_id(lock_id) {
            Some(target) => target,
            None => return,
        };
        let (target_tile, local_id): (&mut Tile, u8) = match lock_target {
            LockTarget::Own(id) => (tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref_mut() {
                Some(west) => (west, id),
                None => return,
            },
            LockTarget::East(id) => match neighbors.east.as_deref_mut() {
                Some(east) => (east, id),
                None => return,
            },
        };
        if let Some(lock) = target_tile.locks.get_mut(local_id as usize) {
            let _ = lock.release_with_value(release_value);
        }
        log::info!(
            "DMA tile({},{}) lock release bd_lock={} delta={} (inline, pipelined with last data cycle)",
            self.col,
            self.row,
            lock_id,
            release_value
        );
    }

    /// Submit a release request to the appropriate tile's arbiter.
    fn submit_release_request(
        &self,
        lock_id: u8,
        release_value: i8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        ch_idx: u8,
    ) {
        use crate::device::tile::LockRequest;

        let lock_target =
            match Self::resolve_lock_id_static(self.tile_kind, self.col, self.row, self.num_locks, lock_id) {
                Some(target) => target,
                None => return,
            };

        let (target_tile, local_id): (&mut Tile, u8) = match lock_target {
            LockTarget::Own(id) => (tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref_mut() {
                Some(west) => (west, id),
                None => return,
            },
            LockTarget::East(id) => match neighbors.east.as_deref_mut() {
                Some(east) => (east, id),
                None => return,
            },
        };

        target_tile.submit_lock_request(LockRequest {
            requestor: self.channel_requestor(ch_idx),
            lock_id: local_id as usize,
            is_acquire: false,
            expected: 0,
            delta: release_value,
            equal_mode: false,
        });
    }

    /// Resolve a BD lock ID to a local lock index.
    ///
    /// For MemTile, maps the 8-bit cross-tile lock address space to local
    /// lock indices. For compute/shim, passes through directly.
    pub(super) fn resolve_lock_id(&self, lock_id: u8) -> Option<LockTarget> {
        Self::resolve_lock_id_static(self.tile_kind, self.col, self.row, self.num_locks, lock_id)
    }

    /// Static version of resolve_lock_id for use when &self is partially borrowed.
    ///
    /// MemTile: 8-bit lock ID field addressing 192 entries across three tiles
    /// (per mlir-aie getLockLocalBaseIndex, aie-rt NumLocks=192):
    ///   - IDs   0 .. num_locks-1:         West neighbor (col-1) locks
    ///   - IDs   num_locks .. 2*num_locks-1: Own tile locks
    ///   - IDs 2*num_locks .. 3*num_locks-1: East neighbor (col+1) locks
    ///
    /// Compute/shim: 4-bit field, always Own tile.
    pub fn resolve_lock_id_static(
        tile_kind: TileKind,
        col: u8,
        row: u8,
        num_locks: u8,
        lock_id: u8,
    ) -> Option<LockTarget> {
        if !tile_kind.is_mem() {
            // Compute/shim: 4-bit field, always local
            return Some(LockTarget::Own(lock_id));
        }

        // MemTile: 8-bit field, cross-tile address space (3 * num_locks entries)
        if lock_id < num_locks {
            Some(LockTarget::West(lock_id))
        } else if lock_id < num_locks * 2 {
            Some(LockTarget::Own(lock_id - num_locks))
        } else if lock_id < num_locks * 3 {
            Some(LockTarget::East(lock_id - num_locks * 2))
        } else {
            let msg = format!(
                "DMA tile({},{}) lock_id={} out of {}-entry address space",
                col,
                row,
                lock_id,
                num_locks as u16 * 3,
            );
            log::error!("{}", msg);
            // Cannot push to fatal_errors from static method; caller must handle None.
            None
        }
    }
}
