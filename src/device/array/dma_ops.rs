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
    pub fn step_dma(&mut self, col: u8, row: u8, host_memory: &mut HostMemory) -> Option<DmaResult> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        let idx = self.tile_index(col, row);
        let is_mem_tile = self.tiles[idx].is_mem_tile();

        let result = if is_mem_tile {
            let rows = self.rows as usize;
            let cols = self.cols as usize;
            let (west_ref, own_ref, east_ref) = get_three_mut(
                &mut self.tiles, idx, col as usize, rows, cols,
            );
            let mut neighbors = dma::NeighborLocks { west: west_ref, east: east_ref };
            self.dma_engines[idx].step(own_ref, &mut neighbors, host_memory)
        } else {
            self.dma_engines[idx].step(&mut self.tiles[idx], &mut dma::NeighborLocks::empty(), host_memory)
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

        for i in 0..tiles.len() {
            let is_mem_tile = engines[i].tile_type.is_mem_tile();
            if is_mem_tile {
                let col = i / rows;
                let (west_ref, own_ref, east_ref) = get_three_mut(
                    tiles, i, col, rows, cols,
                );
                let mut neighbors = dma::NeighborLocks { west: west_ref, east: east_ref };
                engines[i].submit_lock_requests(own_ref, &mut neighbors);
            } else {
                engines[i].submit_lock_requests(&mut tiles[i], &mut dma::NeighborLocks::empty());
            }
        }
    }

    pub fn step_all_dma(&mut self, host_memory: &mut HostMemory) -> bool {
        let mut any_active = false;
        let rows = self.rows as usize;
        let cols = self.cols as usize;

        // Destructure for disjoint field borrows (tiles vs engines)
        let tiles = &mut self.tiles;
        let engines = &mut self.dma_engines;

        for i in 0..tiles.len() {
            // Reset bank tracking for this cycle
            tiles[i].reset_bank_tracking();
            engines[i].cycle_dma_banks = 0;

            let is_mem_tile = engines[i].tile_type.is_mem_tile();

            let result = if is_mem_tile {
                let col = i / rows;
                let (west_ref, own_ref, east_ref) = get_three_mut(
                    tiles, i, col, rows, cols,
                );
                let mut neighbors = dma::NeighborLocks { west: west_ref, east: east_ref };
                engines[i].step(own_ref, &mut neighbors, host_memory)
            } else {
                engines[i].step(&mut tiles[i], &mut dma::NeighborLocks::empty(), host_memory)
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

    /// Check if any DMA engine has active transfers.
    pub fn any_dma_active(&self) -> bool {
        self.dma_engines.iter().any(|e| e.any_channel_active())
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
}
