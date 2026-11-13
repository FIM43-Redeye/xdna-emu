//! Lock resolution and submission for DMA engine.

use super::*;

impl DmaEngine {
    /// A compute MM2S lock port's presentation of this step (made in `submit_lock_requests`)
    /// was granted by the arbiter: record it. Runs at the start of the channel's step.
    pub(super) fn absorb_lock_port_grant(&mut self, ch_idx: usize, tile: &Tile) {
        let now = self.read_clock;
        let Some((lock_id, _)) = self.channels[ch_idx].lock_port.presentation(now) else {
            return;
        };
        let Some(LockTarget::Own(local)) = self.resolve_lock_id(lock_id) else {
            return;
        };
        if tile.lock_was_granted(self.channel_requestor(ch_idx as u8), local as usize) {
            let ch = &mut self.channels[ch_idx];
            ch.lock_port.granted(&mut ch.word_clock, now);
        }
    }

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
            // A compute MM2S's acquires come from its lock port, whatever BD the FSM is on.
            if self.channels[ch_idx].word_clock.is_running() {
                if let Some((lock_id, value)) =
                    self.channels[ch_idx].lock_port.presentation(self.read_clock + 1)
                {
                    self.submit_acquire_request(lock_id, value, tile, neighbors, ch_idx as u8);
                }
                if matches!(self.channels[ch_idx].fsm, ChannelFsm::AcquiringLock { .. }) {
                    continue;
                }
            }
            match &self.channels[ch_idx].fsm {
                ChannelFsm::AcquiringLock { lock_id, acquired: false, transfer, .. } => {
                    if self.acquire_blocked_by_parked_releases(ch_idx) {
                        continue;
                    }
                    self.submit_acquire_request(
                        *lock_id,
                        transfer.acquire_value,
                        tile,
                        neighbors,
                        ch_idx as u8,
                    );
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

    /// Memtile: a channel with two releases parked performs no further acquire. Silicon
    /// (lock-leftovers mem_*_rel_c3..c7): with one release parked the next BD's acquire is
    /// still performed (the memtile issues it while the previous BD is transferring), with two
    /// parked the third BD's is not. Compute and shim channels are not limited (compute
    /// dma_rel_c6, shim *_rel_c7: every later gate taken).
    fn acquire_blocked_by_parked_releases(&self, ch_idx: usize) -> bool {
        self.tile_kind.is_mem()
            && self.channels[ch_idx].parked_releases.len() >= Self::MEMTILE_PARKED_RELEASE_LIMIT
    }

    const MEMTILE_PARKED_RELEASE_LIMIT: usize = 2;

    /// Submit an acquire request to the appropriate tile's arbiter.
    fn submit_acquire_request(
        &self,
        lock_id: u8,
        acquire_value: i8,
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

        // Lock::acquire_change's rule (silicon, gen_dma_lock_probe.py): a change >= 0
        // is an equal acquire that leaves the count alone.
        let (expected, delta, equal_mode) = if acquire_value < 0 {
            (acquire_value.unsigned_abs(), acquire_value, false)
        } else {
            (acquire_value as u8, 0, true)
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
        ch_idx: usize,
    ) {
        self.release_lock_value(lock_id, release_value, tile, neighbors, ch_idx);
        self.emit_lock_release_trace_at(lock_id, self.current_cycle, tile, neighbors);
    }

    /// Whether a release now has to wait: its lock holds an overflow / underflow flag, or an
    /// earlier release of the channel is still parked (they are performed in order).
    pub(super) fn release_must_park(
        &self,
        ch_idx: usize,
        lock_id: u8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> bool {
        !self.channels[ch_idx].parked_releases.is_empty()
            || self.release_target_frozen(lock_id, tile, neighbors)
    }

    /// Perform the channel's parked releases, oldest first, up to the first whose lock still
    /// holds a flag.
    pub(super) fn drain_parked_releases(
        &mut self,
        ch_idx: usize,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) {
        while let Some(&(lock_id, value)) = self.channels[ch_idx].parked_releases.front() {
            if self.release_target_frozen(lock_id, tile, neighbors) {
                break;
            }
            self.channels[ch_idx].parked_releases.pop_front();
            self.apply_lock_release_direct(lock_id, value, tile, neighbors, ch_idx);
        }
    }

    /// Whether the lock a BD releases holds an overflow / underflow flag (and so refuses).
    pub(super) fn release_target_frozen(
        &self,
        lock_id: u8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) -> bool {
        self.resolve_lock_tile(lock_id, tile, neighbors)
            .and_then(|(t, local)| t.locks.get(local as usize).map(|l| l.is_frozen()))
            .unwrap_or(false)
    }

    /// Resolve a (possibly-neighbor) lock id to its tile and local index.
    fn resolve_lock_tile<'t>(
        &self,
        lock_id: u8,
        tile: &'t mut Tile,
        neighbors: &'t mut NeighborTiles<'_>,
    ) -> Option<(&'t mut Tile, u8)> {
        let lock_target = self.resolve_lock_id(lock_id)?;
        match lock_target {
            LockTarget::Own(id) => Some((tile, id)),
            LockTarget::West(id) => neighbors.west.as_deref_mut().map(|w| (w, id)),
            LockTarget::East(id) => neighbors.east.as_deref_mut().map(|e| (e, id)),
        }
    }

    /// Apply the FUNCTIONAL semaphore release -- the lock value a waiting
    /// channel acquires. Carries no trace event; the observable LockRelease is
    /// emitted separately (and, on the memtile, later) via
    /// `emit_lock_release_trace_at`. Splitting the two lets the functional
    /// release stay prompt (at the buffer swap) while the trace event trails by
    /// the memtile pipeline latency -- see `PendingRelease`.
    ///
    /// `ch_idx` is the flat channel index of the releasing channel (S2MM <
    /// s2mm_count, MM2S >= s2mm_count).  It is passed explicitly so the gated
    /// lock recorder can tag the event without an additional borrow of `self`.
    pub(super) fn release_lock_value(
        &mut self,
        lock_id: u8,
        release_value: i8,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
        ch_idx: usize,
    ) {
        // Resolve the lock target before mutably borrowing the tile/neighbors.
        let lock_target = self.resolve_lock_id(lock_id);

        if let Some((target_tile, local_id)) = self.resolve_lock_tile(lock_id, tile, neighbors) {
            if let Some(lock) = target_tile.locks.get_mut(local_id as usize) {
                let _ = lock.release_with_value(release_value);
            }
        }

        // Record the functional release in the gated lock recorder (zero-cost
        // when `lock_recorder` is `None`).  Only `LockTarget::Own` handoffs
        // are in scope for E4 -- cross-tile releases are skipped.
        if let Some(LockTarget::Own(local_id)) = lock_target {
            if let Some(rec) = &mut self.lock_recorder {
                rec.push(LockEvent {
                    cycle: self.current_cycle,
                    channel_flat: ch_idx as u8,
                    lock_local_id: local_id,
                    op: LockOp::Release,
                });
            }
        }

        log::info!(
            "DMA tile({},{}) lock release bd_lock={} delta={} (inline, pipelined with last data cycle)",
            self.col,
            self.row,
            lock_id,
            release_value
        );
    }

    /// Emit the lock-release TRACE event at `emit_cycle` so the memory-module
    /// trace unit sees it, matching the arbiter path
    /// (`Tile::resolve_lock_requests`). The trace unit monitors all lock state
    /// changes regardless of whether the release went through the arbiter or
    /// this pipelined inline path. Without this, DMA BD-completion releases were
    /// invisible to the trace while acquires (arbiter path) were not -- an
    /// asymmetry vs real silicon (tenant-4 grant-order spike: NPU1 emits
    /// LOCK_SEL*_REL on memtile BD completion; this path dropped them).
    ///
    /// `emit_cycle` is passed explicitly because on the memtile the trace event
    /// trails the functional release by the lock-release pipeline latency.
    pub(super) fn emit_lock_release_trace_at(
        &mut self,
        lock_id: u8,
        emit_cycle: u64,
        tile: &mut Tile,
        neighbors: &mut NeighborTiles<'_>,
    ) {
        if let Some((target_tile, local_id)) = self.resolve_lock_tile(lock_id, tile, neighbors) {
            target_tile
                .mem_trace_pending
                .push((emit_cycle, EventType::LockRelease { lock_id: local_id }));
        }
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
    pub fn resolve_lock_id(&self, lock_id: u8) -> Option<LockTarget> {
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
