//! DMA engine access and stepping for the tile array.

use super::*;

impl TileArray {
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

    /// Set the current cycle on all DMA engines for trace event timestamps.
    /// Called by the coordinator before each step.
    pub fn set_dma_cycle(&mut self, cycle: u64) {
        self.current_cycle = cycle;
        for engine in &mut self.dma_engines {
            engine.set_current_cycle(cycle);
        }
    }

    /// Drain memory-module trace events from all sources (DMA engines, locks).
    ///
    /// On real hardware, the memory trace unit doesn't distinguish DMA from
    /// lock events -- it monitors all hardware event IDs from the memory
    /// module. This method unifies both sources into a single event stream.
    ///
    /// Returns (col, row, cycle, event) tuples for each buffered event.
    pub fn drain_mem_trace_events(&mut self) -> Vec<(u8, u8, u64, EventType)> {
        let mut all_events = Vec::new();
        // Drain DMA engine events (per-engine internal buffer).
        for engine in &mut self.dma_engines {
            let events = engine.drain_trace_events();
            if !events.is_empty() {
                let col = engine.col;
                let row = engine.row;
                for (cycle, event) in events {
                    all_events.push((col, row, cycle, event));
                }
            }
        }
        // Drain tile-level events (locks, etc.).
        for tile in &mut self.tiles {
            if !tile.mem_trace_pending.is_empty() {
                let col = tile.col;
                let row = tile.row;
                for (cycle, event) in tile.mem_trace_pending.drain(..) {
                    all_events.push((col, row, cycle, event));
                }
            }
        }
        all_events
    }

    /// Step the DMA engine for a specific tile.
    ///
    /// This advances the DMA transfer state by one cycle.
    /// For MemTiles, constructs neighbor lock access from adjacent columns.
    ///
    /// Returns `None` when the tile is out of bounds OR when the DMA module is
    /// clock-gated (column gate, module-level MCC gate, or adaptive DMA gate
    /// engaged).  Wake-on-event coverage (register-bus access, stream beat,
    /// lock change) resets the adaptive counter at the emit site, so an
    /// engaged tile resumes on the next cycle after the wake event lands.
    pub fn step_dma(&mut self, col: u8, row: u8, host_memory: &mut HostMemory) -> Option<DmaResult> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        // Module gate check: skip if column is gated, the DMA module is gated,
        // or the adaptive DMA gate has engaged due to sustained idleness.
        // Wake events (Wake 1/2/3 in cycle-accuracy-mission.md item #8)
        // reset the adaptive counter; the gate releases on the cycle
        // following the wake.
        {
            use crate::device::clock_control::ModuleKind;
            if !self.clock.is_column_active(col)
                || !self.clock.is_module_active(col, row, ModuleKind::Dma)
                || self.clock.is_adaptive_dma_engaged(col, row)
            {
                return None;
            }
        }
        let idx = self.tile_index(col, row);
        let is_mem = self.tiles[idx].is_mem();

        let result = if is_mem {
            let rows = self.rows as usize;
            let cols = self.cols as usize;
            let (west_ref, own_ref, east_ref) = get_three_mut(&mut self.tiles, idx, col as usize, rows, cols);
            let mut neighbors = dma::NeighborTiles { west: west_ref, east: east_ref };
            self.dma_engines[idx].step(own_ref, &mut neighbors, host_memory)
        } else {
            self.dma_engines[idx].step(&mut self.tiles[idx], &mut dma::NeighborTiles::empty(), host_memory)
        };
        Some(result)
    }

    /// Step all DMA engines.
    ///
    /// Returns true if any DMA engine is still active.
    /// MemTile engines receive neighbor lock access for cross-tile lock operations.
    /// Pre-step pass: submit all DMA lock requests to tile arbiters.
    ///
    /// Scans all DMA engine channels and submits pending lock acquire/release
    /// requests to the appropriate tile arbiters. Called before arbiter
    /// resolution so that all requests are collected before round-robin
    /// arbitration runs.
    pub fn submit_all_dma_lock_requests(&mut self, _host_memory: &mut HostMemory) {
        let rows = self.rows as usize;
        let cols = self.cols as usize;
        let tiles = &mut self.tiles;
        let engines = &self.dma_engines;
        let clock = &self.clock;

        for i in 0..tiles.len() {
            // Column gate check: skip tiles in gated columns.
            // Silicon does not clock DMA engines in ungated columns, so the
            // emulator skips lock request submission for them too.
            let col = i / rows;
            if !clock.is_column_active(col as u8) {
                continue;
            }

            let is_mem = engines[i].tile_kind.is_mem();
            if is_mem {
                let (west_ref, own_ref, east_ref) = get_three_mut(tiles, i, col, rows, cols);
                let mut neighbors = dma::NeighborTiles { west: west_ref, east: east_ref };
                engines[i].submit_lock_requests(own_ref, &mut neighbors);
            } else {
                engines[i].submit_lock_requests(&mut tiles[i], &mut dma::NeighborTiles::empty());
            }
        }
    }

    /// Step all DMA engines.
    ///
    /// Returns true if any DMA engine made progress this cycle (InProgress or
    /// WaitingForLock). Module gate check: tiles whose DMA module is
    /// clock-gated (column gate, module-level MCC, or adaptive DMA gate
    /// engaged) are skipped. Wake-on-event paths release the adaptive gate
    /// on the cycle after the wake event lands.
    pub fn step_all_dma(&mut self, host_memory: &mut HostMemory) -> bool {
        self.step_all_dma_with_denied(host_memory, &[])
    }

    /// `step_all_dma`, holding the channels that lost this cycle's memory-bank
    /// arbitration. `denied` is indexed by tile index; each entry names that
    /// tile's losing DMA channels (see `DmaEngine::step_with_denied`, which
    /// skips a held channel's FSM step entirely so it re-presents the
    /// identical demand next cycle -- the AM020 ch.2:166 retry contract).
    /// Borrowed slices, not `Vec` -- the caller passes a view over its own
    /// per-tile arbitration results with no per-tile clone (Minor-4).
    pub fn step_all_dma_with_denied(
        &mut self,
        host_memory: &mut HostMemory,
        denied: &[&[crate::device::bank_arbiter::Requester]],
    ) -> bool {
        use crate::device::clock_control::ModuleKind;
        const NO_DENIALS: &[crate::device::bank_arbiter::Requester] = &[];

        let mut any_active = false;
        let rows = self.rows as usize;
        let cols = self.cols as usize;

        // Destructure for disjoint field borrows (tiles vs engines)
        let tiles = &mut self.tiles;
        let engines = &mut self.dma_engines;
        let clock = &self.clock;

        for i in 0..tiles.len() {
            let col = (i / rows) as u8;
            let row = (i % rows) as u8;

            // Column gate check (top tier): skip tiles in gated columns.
            // Silicon does not clock DMA engines in gated columns, so the
            // emulator skips all DMA stepping for them.  This is the top-tier
            // perf win -- typical programs gate 3 of 5 columns.
            if !clock.is_column_active(col) {
                continue;
            }

            // Module gate check (mid tier): skip if the DMA module is gated.
            // On compute/memtile, DMA shares a clock bit with data memory (MCC
            // bit 1).  On shim, DMA (NoC module) is MCC_1 bit 0.
            if !clock.is_module_active(col, row, ModuleKind::Dma) {
                continue;
            }

            // Adaptive gate check (bottom tier): skip if the DMA module has
            // been idle for 2^abort_period cycles.  Wake-on-event coverage
            // (register access in dispatch.rs, stream beat via Phase 5
            // cycle_active chain, lock change via the dispatch path's Lock
            // subsystem arm) resets the counter on emission so the gate
            // releases without manual intervention.
            if clock.is_adaptive_dma_engaged(col, row) {
                continue;
            }

            // Reset bank tracking for this cycle
            tiles[i].reset_bank_tracking();
            engines[i].cycle_dma_banks = 0;

            let is_mem = engines[i].tile_kind.is_mem();
            let held = denied.get(i).copied().unwrap_or(NO_DENIALS);

            let result = if is_mem {
                let (west_ref, own_ref, east_ref) = get_three_mut(tiles, i, col as usize, rows, cols);
                let mut neighbors = dma::NeighborTiles { west: west_ref, east: east_ref };
                engines[i].step_with_denied(held, own_ref, &mut neighbors, host_memory)
            } else {
                engines[i].step_with_denied(
                    held,
                    &mut tiles[i],
                    &mut dma::NeighborTiles::empty(),
                    host_memory,
                )
            };

            if matches!(result, DmaResult::InProgress | DmaResult::WaitingForLock(_)) {
                any_active = true;
            }

            // Merge DMA engine bank accesses into the tile.
            // Static transfer methods record directly on tile.cycle_dma_banks;
            // MM2S/S2MM record on engine.cycle_dma_banks. Merge both.
            tiles[i].cycle_dma_banks |= engines[i].cycle_dma_banks;
        }
        // Drain fatal errors from all DMA engines into the array-level collector.
        // Use index loop to avoid conflicting borrows on self.
        for i in 0..self.dma_engines.len() {
            if !self.dma_engines[i].fatal_errors.is_empty() {
                let mut errs = std::mem::take(&mut self.dma_engines[i].fatal_errors);
                self.fatal_errors.append(&mut errs);
            }
        }

        any_active
    }

    /// Check if any DMA engine has anything to do this cycle.
    ///
    /// Returns true if any channel on any engine is FSM-active, has a queued
    /// BD, or has a non-empty task queue. The TDR uses this as the
    /// "DMA still has reason to run" signal feeding Quiescent / completion
    /// classification, and the FFI exports it as EngineSignals.any_dma_active.
    /// Using FSM-only here was a blindspot: a channel with a queued task and
    /// FSM=Idle (the moment after enqueue, before the next step promotes it
    /// to BdSetup) reported inactive, letting TDR classify the array as
    /// quiescent while work was actually pending.
    pub fn any_dma_active(&self) -> bool {
        self.dma_engines.iter().any(|e| e.any_channel_has_pending_work())
    }

    /// Check if any DMA is actually making progress (not just waiting for locks).
    ///
    /// Returns true if at least one DMA channel is in `Active` state (actively
    /// transferring data). Channels that are only `WaitingForLock` are not
    /// considered to be making progress.
    ///
    /// This is used by the coordinator to detect when all DMAs are stalled
    /// waiting for locks that will never be released (because all cores have
    /// halted). In that case, the engine should halt rather than spinning
    /// forever.
    pub fn any_dma_transferring(&self) -> bool {
        use crate::device::dma::engine::ChannelState;
        self.dma_engines.iter().any(|engine| {
            for ch in 0..engine.num_channels() {
                if matches!(engine.channel_state(ch as u8), ChannelState::Active) {
                    return true;
                }
            }
            false
        })
    }

    /// Check if any DMA is stalled waiting for locks.
    ///
    /// Returns true if at least one DMA channel is waiting for a lock to be
    /// released. When combined with `all_cores_halted`, this indicates the
    /// engine is deadlocked and should halt.
    pub fn any_dma_waiting_for_lock(&self) -> bool {
        use crate::device::dma::engine::ChannelState;
        self.dma_engines.iter().any(|engine| {
            for ch in 0..engine.num_channels() {
                if matches!(engine.channel_state(ch as u8), ChannelState::WaitingForLock(_)) {
                    return true;
                }
            }
            false
        })
    }

    /// True when all DMA channels across all tiles are in terminal states
    /// (Idle, Complete, Error, or WaitingForLock -- not Active).
    pub fn all_dma_terminal(&self) -> bool {
        !self.any_dma_transferring()
    }

    /// Check if any stream switch or cascade FIFO has pending data.
    ///
    /// Returns true if at least one stream switch has buffered words or
    /// at least one cascade FIFO (input or output) is non-empty.
    pub fn any_data_in_flight(&self) -> bool {
        for tile in &self.tiles {
            if tile.stream_switch.has_pending_data() {
                return true;
            }
            if !tile.cascade_input.is_empty() || !tile.cascade_output.is_empty() {
                return true;
            }
        }
        false
    }

    /// Sum of bytes_transferred across all DMA channels in the array.
    ///
    /// This serves as a progress counter for no-progress detection (TDR),
    /// mirroring the real xdna-driver's approach in `aie2_tdr.c`. If this
    /// value stops increasing, the workload is stalled.
    pub fn total_dma_bytes_transferred(&self) -> u64 {
        let mut total = 0u64;
        for engine in &self.dma_engines {
            for ch in 0..engine.num_channels() {
                if let Some(stats) = engine.channel_stats(ch as u8) {
                    total += stats.bytes_transferred;
                }
            }
        }
        total
    }

    /// Total lock releases granted across all tiles.
    ///
    /// Monotonically increasing counter used by stall detection.
    /// Counts both core and DMA lock releases.
    pub fn total_lock_releases(&self) -> u64 {
        self.tiles.iter().map(|t| t.lock_release_count()).sum()
    }
}
