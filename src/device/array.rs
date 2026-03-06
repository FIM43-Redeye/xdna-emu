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

use super::aie2_spec::stream_switch::{compute, mem_tile, shim};
use super::arch_config::{ArchConfig, ModelConfig};
use super::dma::{self, DmaEngine, DmaResult};
use super::host_memory::HostMemory;
use super::tile::{Tile, TileParams, TileType};
use crate::interpreter::state::EventType;
use std::sync::Arc;

/// Maximum supported columns
pub const MAX_COLS: usize = 9;

/// Maximum supported rows
pub const MAX_ROWS: usize = 6;

/// Get mutable references to up to 3 tiles at disjoint indices for cross-tile lock access.
///
/// Returns (west_neighbor, own_tile, east_neighbor) where neighbors are None
/// if they don't exist or aren't MemTiles. Uses `split_at_mut` for safe
/// disjoint mutable borrows -- no unsafe code needed.
///
/// Indices: west = own_idx - rows, east = own_idx + rows (separated by one
/// column's worth of rows, always distinct).
fn get_three_mut(
    tiles: &mut [Tile],
    own_idx: usize,
    col: usize,
    rows: usize,
    cols: usize,
) -> (Option<&mut Tile>, &mut Tile, Option<&mut Tile>) {
    let west_idx = if col > 0 {
        let idx = own_idx - rows;
        // Only provide neighbor if it's also a MemTile
        if tiles[idx].is_mem_tile() { Some(idx) } else { None }
    } else {
        None
    };
    let east_idx = if col + 1 < cols {
        let idx = own_idx + rows;
        if tiles[idx].is_mem_tile() { Some(idx) } else { None }
    } else {
        None
    };

    match (west_idx, east_idx) {
        (None, None) => {
            (None, &mut tiles[own_idx], None)
        }
        (Some(w), None) => {
            // w < own_idx guaranteed (west is lower column)
            let (left, right) = tiles.split_at_mut(own_idx);
            (Some(&mut left[w]), &mut right[0], None)
        }
        (None, Some(e)) => {
            // own_idx < e guaranteed (east is higher column)
            let (left, right) = tiles.split_at_mut(e);
            (None, &mut left[own_idx], Some(&mut right[0]))
        }
        (Some(w), Some(e)) => {
            // w < own_idx < e guaranteed
            let (left, rest) = tiles.split_at_mut(own_idx);
            let (mid, right) = rest.split_at_mut(e - own_idx);
            (Some(&mut left[w]), &mut mid[0], Some(&mut right[0]))
        }
    }
}

/// AIE tile array.
///
/// Stores tiles in row-major order for cache-friendly column iteration.
pub struct TileArray {
    /// Architecture configuration (determines tile types, port layouts, etc.)
    arch: Arc<dyn ArchConfig>,

    /// Number of columns (including shim column)
    cols: u8,

    /// Number of rows
    rows: u8,

    /// Tiles stored in flat array: tiles[col * rows + row]
    /// Using Vec because tile count varies by device
    pub(crate) tiles: Vec<Tile>,

    /// DMA engines stored in parallel with tiles.
    /// Each tile has exactly one DMA engine.
    pub(crate) dma_engines: Vec<DmaEngine>,

    /// Fatal errors accumulated during data movement.
    ///
    /// These represent conditions that are impossible on real hardware
    /// (e.g., packet with no route, stream push failure). The coordinator
    /// checks this after each step and aborts if non-empty.
    pub(crate) fatal_errors: Vec<String>,

    /// Control packet actions produced during stream routing.
    ///
    /// Control packet writes from tiles are collected here instead of being
    /// executed immediately. The caller (DeviceState or coordinator) drains
    /// these and routes them through `DeviceState::write_tile_register()` for
    /// full module dispatch.
    pub(crate) pending_ctrl_actions: Vec<super::tile::CtrlPacketAction>,

    /// In-flight words traversing inter-tile links.
    ///
    /// On real AIE2 hardware, data takes ROUTE_LATENCY_PER_HOP cycles to
    /// traverse each inter-tile link (physical wire + switch pipeline). This
    /// buffer models that latency: words enter with a countdown timer and
    /// are delivered to the destination slave port when the countdown reaches 0.
    ///
    /// Without this, data would teleport instantly between tiles, making stream
    /// delivery ~12x faster than hardware (observed: EMU=645cy vs HW=8185cy
    /// starvation on tile_dmas_writebd).
    inter_tile_pipeline: Vec<InFlightWord>,
}

/// A word in flight between adjacent tiles.
///
/// Models the propagation delay through inter-tile stream switch links.
/// The word was popped from the source master port and will be pushed to
/// the destination slave port after `cycles_remaining` reaches 0.
#[derive(Debug, Clone)]
struct InFlightWord {
    dst_tile_idx: usize,
    dst_slave_idx: usize,
    data: u32,
    tlast: bool,
    /// Cycles until delivery. Decremented each coordinator cycle.
    /// Word is delivered when this reaches 0.
    cycles_remaining: u8,
}

impl TileArray {
    /// Create a new tile array for the given architecture configuration.
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        let cols = arch.columns();
        let rows = arch.rows();
        let capacity = (cols as usize) * (rows as usize);

        let mut tiles = Vec::with_capacity(capacity);
        let mut dma_engines = Vec::with_capacity(capacity);

        // Create tiles and DMA engines in column-major order.
        // Per-tile-type params come from ArchConfig (data-driven from mlir-aie).
        for col in 0..cols {
            for row in 0..rows {
                let tile_type = arch.tile_type(col, row);
                let params = TileParams {
                    data_memory_size: arch.data_memory_size(tile_type),
                    num_locks: arch.lock_count(tile_type),
                    num_bds: arch.dma_bd_count(tile_type),
                    num_channels: arch.dma_total_channels(tile_type),
                    dma_s2mm_channels: arch.dma_s2mm_channels(tile_type),
                    dma_mm2s_channels: arch.dma_mm2s_channels(tile_type),
                };
                tiles.push(Tile::new(tile_type, col, row, &params));

                // Create DMA engine with ArchConfig-derived channel/BD/lock counts
                dma_engines.push(DmaEngine::new(
                    col, row, tile_type,
                    params.dma_s2mm_channels, params.dma_mm2s_channels,
                    params.num_bds, params.num_locks as u8,
                ));
            }
        }

        Self {
            arch, cols, rows, tiles, dma_engines,
            fatal_errors: Vec::new(),
            pending_ctrl_actions: Vec::new(),
            inter_tile_pipeline: Vec::new(),
        }
    }

    /// Create an NPU1 (Phoenix/HawkPoint) array.
    #[inline]
    pub fn npu1() -> Self {
        Self::new(ModelConfig::npu1())
    }

    /// Create an NPU2 (Strix) array.
    #[inline]
    pub fn npu2() -> Self {
        Self::new(ModelConfig::npu2())
    }

    /// Drain accumulated fatal errors from data movement.
    ///
    /// Returns the errors and clears the internal list. The coordinator
    /// calls this after each step to detect impossible-on-hardware
    /// conditions and abort immediately.
    pub fn drain_fatal_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.fatal_errors)
    }

    /// Drain pending control packet actions produced during stream routing.
    ///
    /// Control packets arrive via the stream switch network at individual tiles.
    /// Rather than writing registers directly (which misses the full module
    /// dispatch in DeviceState), the tile returns actions. The caller drains
    /// these and routes them through `DeviceState::write_tile_register()`.
    pub fn drain_ctrl_packet_actions(&mut self) -> Vec<super::tile::CtrlPacketAction> {
        std::mem::take(&mut self.pending_ctrl_actions)
    }

    /// Handle a control packet OP_READ by reading registers and queuing
    /// a response packet for injection into the tile's TileCtrl slave port.
    ///
    /// The response consists of a stream packet header (with pkt_id =
    /// response_id, packet_type = Data) followed by `count` data words,
    /// with TLAST set on the final word.
    ///
    /// Response words are buffered in `tile.pending_ctrl_response` and
    /// drained into the TileCtrl slave port each routing cycle as FIFO
    /// space permits (same backpressure-aware pattern as trace injection).
    ///
    /// Returns true if the response was successfully queued.
    pub fn handle_read_registers(
        &mut self,
        col: u8,
        row: u8,
        offset: u32,
        count: u8,
        response_id: u8,
    ) -> bool {
        use super::stream_switch::{PacketHeader, PacketType};

        let tile = match self.get_mut(col, row) {
            Some(t) => t,
            None => {
                log::error!(
                    "handle_read_registers: tile({},{}) not found", col, row
                );
                return false;
            }
        };

        // Verify the TileCtrl slave port exists.
        if tile.stream_switch.tile_ctrl_slave_port().is_none() {
            log::error!(
                "handle_read_registers: tile({},{}) has no TileCtrl slave port",
                col, row,
            );
            return false;
        }

        // Read the register values (pure reads, no side effects).
        let mut values = Vec::with_capacity(count as usize);
        for i in 0..count as u32 {
            values.push(tile.read_register_pure(offset + i * 4));
        }

        // Build stream packet header: pkt_id = response_id, type = Data,
        // source = this tile's (col, row).
        let header = PacketHeader::new(response_id & 0x1F, col, row)
            .with_type(PacketType::Data);
        let header_word = header.encode();

        // Queue header + data words into pending buffer.
        // TLAST is set on the last data word (or on the header if count=0).
        tile.pending_ctrl_response.push_back((header_word, count == 0));
        for (i, &value) in values.iter().enumerate() {
            let is_last = i == values.len() - 1;
            tile.pending_ctrl_response.push_back((value, is_last));
        }

        log::info!(
            "handle_read_registers: tile({},{}) read {} regs from 0x{:05X}, \
             {} response words queued (resp_id={})",
            col, row, count, offset, count as usize + 1, response_id,
        );

        true
    }

    /// Drain pending control packet read responses into TileCtrl slave ports.
    ///
    /// Called during each routing cycle. Pushes as many queued response words
    /// as the TileCtrl slave FIFO can accept, respecting backpressure.
    /// Returns the number of words injected.
    pub fn drain_ctrl_responses(&mut self) -> usize {
        let mut words_injected = 0;

        for i in 0..self.tiles.len() {
            if self.tiles[i].pending_ctrl_response.is_empty() {
                continue;
            }

            let slave_idx = match self.tiles[i].stream_switch.tile_ctrl_slave_port() {
                Some(idx) => idx,
                None => continue,
            };

            while !self.tiles[i].pending_ctrl_response.is_empty()
                && self.tiles[i].stream_switch.slaves[slave_idx].can_accept()
            {
                let (word, tlast) = self.tiles[i].pending_ctrl_response.pop_front().unwrap();
                self.tiles[i].stream_switch.slaves[slave_idx].push_with_tlast(word, tlast);
                words_injected += 1;
            }
        }

        words_injected
    }

    /// Get the architecture configuration.
    #[inline]
    pub fn arch(&self) -> &dyn ArchConfig {
        self.arch.as_ref()
    }

    /// Get a shared reference to the architecture configuration.
    #[inline]
    pub fn arch_arc(&self) -> Arc<dyn ArchConfig> {
        Arc::clone(&self.arch)
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
    pub fn tile_index(&self, col: u8, row: u8) -> usize {
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

    /// Set the current cycle on all DMA engines for trace event timestamps.
    /// Called by the coordinator before each step.
    pub fn set_dma_cycle(&mut self, cycle: u64) {
        for engine in &mut self.dma_engines {
            engine.set_current_cycle(cycle);
        }
    }

    /// Drain trace events from all DMA engines.
    /// Returns (col, row, cycle, event) tuples for each buffered event.
    pub fn drain_dma_trace_events(&mut self) -> Vec<(u8, u8, u64, EventType)> {
        let mut all_events = Vec::new();
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
        use super::dma::engine::ChannelState;
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
        use super::dma::engine::ChannelState;
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

    /// Step all per-tile stream switches.
    ///
    /// This handles intra-tile routing: forwarding data from input (slave) ports
    /// to output (master) ports within each tile based on configured local routes.
    /// This is necessary for pass-through routing where data enters a tile from
    /// one direction and exits to another.
    ///
    /// Returns the total number of words forwarded across all tiles.
    pub fn step_tile_switches(&mut self) -> usize {
        let mut total_forwarded = 0;
        for tile in &mut self.tiles {
            total_forwarded += tile.stream_switch.step();
            // Drain any fatal errors from this tile's stream switch
            self.fatal_errors.append(&mut tile.stream_switch.fatal_errors);
        }
        total_forwarded
    }

    /// Convert S2MM DMA channel index to stream switch slave port index.
    ///
    /// Port indices derived from AM025 gen_stream_ranges.rs constants.
    #[allow(dead_code)] // Documented port mapping, may be used in future
    fn s2mm_channel_to_slave_port(tile_type: TileType, channel: u8) -> u8 {
        match tile_type {
            TileType::Compute => compute::DMA_SLAVE_START + channel,
            TileType::MemTile => mem_tile::DMA_SLAVE_START + channel,
            TileType::Shim => shim::NORTH_SLAVE_START + channel,
        }
    }

    /// Convert MM2S DMA channel index to stream switch master port index.
    ///
    /// The `channel` argument is the absolute DMA channel index (S2MM channels
    /// come first, MM2S channels follow). `s2mm_count` is the number of S2MM
    /// channels for this tile (from DmaEngine). Port indices derived from AM025
    /// gen_stream_ranges.rs.
    #[allow(dead_code)] // Documented port mapping, may be used in future
    fn mm2s_channel_to_master_port(tile_type: TileType, channel: u8, s2mm_count: u8) -> u8 {
        let ch_offset = channel.saturating_sub(s2mm_count);
        match tile_type {
            TileType::Compute => compute::DMA_MASTER_START + ch_offset,
            TileType::MemTile => mem_tile::DMA_MASTER_START + ch_offset,
            TileType::Shim => shim::NORTH_MASTER_START + ch_offset,
        }
    }

    /// Step the complete data movement system.
    ///
    /// This follows the NPU hardware architecture with 4 clean phases:
    ///
    /// 1. **Lock Snapshot**: Freeze lock values for consistent visibility
    /// 2. **DMA Step**: All DMA engines produce/consume stream data
    /// 3. **Stream Routing**: Route data through stream switches and inter-tile wires
    /// 4. **Lock Commit**: Apply accumulated lock changes for next cycle
    ///
    /// The stream routing phase handles:
    /// - DMA MM2S → StreamSwitch slave ports
    /// - Core stream output → StreamSwitch slave ports
    /// - StreamSwitch local routing (slave → master via configured routes)
    /// - Inter-tile propagation (physical wires: North master → South slave, etc.)
    /// - StreamSwitch master ports → DMA S2MM
    ///
    /// This matches real NPU behavior where each tile has a StreamSwitch and
    /// tiles connect via physical inter-tile wires. There is no global router.
    ///
    /// Returns (dma_active, streams_moved, words_routed)
    pub fn step_data_movement(&mut self, host_memory: &mut HostMemory) -> (bool, bool, usize) {
        // Phase 0: Port activity tracking reset.
        // Seed cycle_active flags from pre-existing FIFO state before routing.
        // Ports that receive data during routing will also be marked active.
        // The coordinator reads these flags after routing to generate
        // PORT_RUNNING trace events.
        for tile in &mut self.tiles {
            tile.stream_switch.begin_routing_cycle();
        }

        // Phase 1: DMA Lock Request Submission
        // Each DMA engine submits lock acquire/release requests to tile arbiters.
        // Core lock releases from the coordinator's Phase 2 are already pending.
        self.submit_all_dma_lock_requests(host_memory);

        // Phase 2: Lock Arbiter Resolution
        // Resolve all tile arbiters using round-robin. Applies granted requests
        // directly to lock values. DMA channels check results in Phase 3.
        for tile in &mut self.tiles {
            tile.resolve_lock_requests();
        }

        // Phase 3: DMA Step
        // All DMA engines advance channel FSMs, checking arbiter results.
        // MM2S channels produce stream words, S2MM channels consume them.
        let dma_active = self.step_all_dma(host_memory);

        // Phase 4: Stream Routing
        // Route all stream data through the stream switch network.
        let words_routed = self.route_streams();

        // Phase 4.5: Cascade Propagation
        // Dedicated point-to-point 384-bit links between compute tiles.
        // Entirely separate from the stream switch fabric.
        self.route_cascade();

        let switch_pipelines_active = self.tiles.iter()
            .any(|t| t.stream_switch.has_pipeline_data());
        let streams_active = words_routed > 0
            || !self.inter_tile_pipeline.is_empty()
            || switch_pipelines_active;
        (dma_active, streams_active, words_routed)
    }

    /// Route cascade data between adjacent compute tiles.
    ///
    /// Cascade is a dedicated 384-bit point-to-point link entirely separate
    /// from the stream switch fabric. Each compute tile can send cascade data
    /// to one neighbor (South or East) and receive from one neighbor (North or
    /// West), as configured by the accumulator control register at 0x36060.
    ///
    /// Uses a two-phase collect-then-apply approach to avoid borrow issues.
    fn route_cascade(&mut self) {
        let rows = self.rows;
        let cols = self.cols;

        // Phase 1: Collect pending transfers.
        // Each entry: (src_idx, dst_col, dst_row)
        let mut transfers: Vec<(usize, u8, u8)> = Vec::new();

        for col in 0..cols {
            for row in 0..rows {
                let idx = (col as usize) * (rows as usize) + (row as usize);
                let tile = &self.tiles[idx];

                if !tile.is_compute() || !tile.has_cascade_output() {
                    continue;
                }

                // Determine destination based on output direction
                let (dst_col, dst_row) = match tile.cascade_output_dir {
                    0 => {
                        // South: (col, row - 1)
                        if row == 0 { continue; }
                        (col, row - 1)
                    }
                    1 => {
                        // East: (col + 1, row)
                        if col + 1 >= cols { continue; }
                        (col + 1, row)
                    }
                    _ => continue,
                };

                // Verify destination exists and is a compute tile
                let dst_idx = (dst_col as usize) * (rows as usize) + (dst_row as usize);
                if dst_idx >= self.tiles.len() || !self.tiles[dst_idx].is_compute() {
                    continue;
                }

                // Verify destination's input direction matches
                let dst_input_dir = self.tiles[dst_idx].cascade_input_dir;
                let expected_dir = match tile.cascade_output_dir {
                    0 => 0, // South output -> North input (dir=0)
                    1 => 1, // East output -> West input (dir=1)
                    _ => continue,
                };
                if dst_input_dir != expected_dir {
                    continue;
                }

                // Check backpressure: destination must have room
                if self.tiles[dst_idx].cascade_input.is_empty() {
                    transfers.push((idx, dst_col, dst_row));
                }
            }
        }

        // Phase 2: Apply transfers
        for (src_idx, dst_col, dst_row) in transfers {
            let dst_idx = (dst_col as usize) * (rows as usize) + (dst_row as usize);

            if let Some(data) = self.tiles[src_idx].pop_cascade_output() {
                log::debug!(
                    "[CASCADE] Route ({},{}) -> ({},{})",
                    self.tiles[src_idx].col, self.tiles[src_idx].row,
                    dst_col, dst_row
                );
                self.tiles[dst_idx].push_cascade_input(data);
            }
        }
    }

    /// Route all stream data through the NPU stream network.
    ///
    /// This implements the per-tile StreamSwitch model from AM020/AM025:
    /// - Each tile has a StreamSwitch with slave (input) and master (output) ports
    /// - DMA MM2S outputs to slave ports, DMA S2MM receives from master ports
    /// - Configured routes forward data from slaves to masters within each tile
    /// - Inter-tile wires connect adjacent tiles (North master → South slave, etc.)
    ///
    /// Data flows: DMA MM2S → slave → [route] → master → [wire] → slave → [route] → master → DMA S2MM
    ///
    /// For multi-hop paths (e.g., Shim → MemTile → Compute), data takes multiple
    /// cycles to traverse. Inter-tile links add ROUTE_LATENCY_PER_HOP cycles per
    /// hop, matching the AM020 stream switch pipeline specification.
    fn route_streams(&mut self) -> usize {
        let mut words_routed = 0;

        // Step 1: DMA MM2S → StreamSwitch slave ports
        // Data from DMA output buffers enters the stream switch network
        words_routed += self.route_dma_to_tile_switches();

        // Step 2: Core stream → StreamSwitch slave ports
        // Data from core stream writes enters via the Core slave port (port 0)
        words_routed += self.route_core_to_tile_switches();

        // Step 2b: Trace unit → StreamSwitch trace slave ports
        // Binary trace packets enter on dedicated trace slave ports
        // (indices from AM025 gen_stream_ranges.rs)
        words_routed += self.route_trace_to_tile_switches();

        // Step 2c: Control packet read responses → TileCtrl slave ports
        // Queued OP_READ response words drain into TileCtrl slave ports
        // as FIFO space permits (backpressure-aware, same as trace).
        words_routed += self.drain_ctrl_responses();

        // Step 3: StreamSwitch local routing (first pass)
        // Apply configured routes within each tile: slave → master
        words_routed += self.step_tile_switches();

        // Step 4: Inter-tile propagation
        // Physical wires carry data between adjacent tiles:
        // - North masters → South slaves of tile above
        // - South masters → North slaves of tile below
        words_routed += self.propagate_inter_tile();

        // Step 5: StreamSwitch local routing (second pass)
        // Forward newly arrived data through local routes
        words_routed += self.step_tile_switches();

        // Step 6: StreamSwitch master ports → DMA S2MM
        // Data reaching DMA master ports enters DMA input buffers
        words_routed += self.route_tile_switches_to_dma();

        // Step 7: StreamSwitch Core master → tile stream_input
        // Data reaching the Core master port goes to tile.stream_input for core reads
        words_routed += self.route_tile_switches_to_core();

        // Step 8: StreamSwitch TileCtrl master → tile control packet handler
        // Data reaching the TileCtrl master port gets processed as register writes
        words_routed += self.route_tile_switches_to_ctrl();

        words_routed
    }

    /// Route data from StreamSwitch Core master port to tile stream input.
    ///
    /// When data arrives at the Core master port (port 0) on a compute tile,
    /// it should be delivered to the tile's stream_input buffer for core reads.
    fn route_tile_switches_to_core(&mut self) -> usize {
        use super::stream_switch::PortType;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Only compute tiles have core stream I/O
            if !self.tiles[i].is_compute() {
                continue;
            }

            // Check Core master port (port 0) for data
            if self.tiles[i].stream_switch.masters[0].fifo.is_empty() {
                continue;
            }

            // Verify it's a Core port
            if !matches!(self.tiles[i].stream_switch.masters[0].port_type, PortType::Core) {
                continue;
            }

            // Pop from Core master port and push to tile stream_input
            if let Some(data) = self.tiles[i].stream_switch.masters[0].pop() {
                // Push to tile's stream input buffer (port 0 for core)
                self.tiles[i].push_stream_input(0, data);
                words_routed += 1;
                log::debug!(
                    "TileSwitch->Core: tile ({},{}) master[0] -> stream_input = 0x{:08X}",
                    self.tiles[i].col, self.tiles[i].row, data
                );
            }
        }

        words_routed
    }

    /// Route data from StreamSwitch TileCtrl master port to tile control packet handler.
    ///
    /// When data arrives at the TileCtrl master port (port 3 on compute, port 6 on
    /// memtile, port 0 on shim), it is fed to the tile's control packet state machine
    /// which interprets it as register writes.
    fn route_tile_switches_to_ctrl(&mut self) -> usize {
        use super::stream_switch::PortType;
        use super::tile::ControlPacketState;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Find TileCtrl master port index
            let ctrl_master = self.tiles[i].stream_switch.masters.iter()
                .position(|p| matches!(p.port_type, PortType::TileCtrl));

            let master_idx = match ctrl_master {
                Some(idx) => idx,
                None => continue,
            };

            // Configure the tile's ctrl packet state from the master port config
            // (once only). If the master port is in packet mode with
            // Drop_Header=false, the stream header is forwarded and must be
            // consumed by the handler before the actual control packet header.
            if self.tiles[i].ctrl_pkt_drop_header {
                let pkt_cfg = self.tiles[i].stream_switch.master_packet_cfg(master_idx);
                if pkt_cfg.map_or(false, |c| c.packet_enable && !c.drop_header) {
                    self.tiles[i].ctrl_pkt_drop_header = false;
                    self.tiles[i].ctrl_pkt_state = ControlPacketState::WaitingForStreamHeader;
                }
            }

            // Drain all pending words from the TileCtrl master port
            while self.tiles[i].stream_switch.masters[master_idx].has_data() {
                if let Some((data, tlast)) = self.tiles[i].stream_switch.masters[master_idx].pop_with_tlast() {
                    let actions = self.tiles[i].process_ctrl_packet_word(data, tlast);
                    if !actions.is_empty() {
                        self.pending_ctrl_actions.extend(actions);
                    }
                    words_routed += 1;
                    log::debug!(
                        "TileSwitch->Ctrl: tile ({},{}) master[{}] -> ctrl_pkt = 0x{:08X}{}",
                        self.tiles[i].col, self.tiles[i].row, master_idx, data,
                        if tlast { " TLAST" } else { "" }
                    );
                }
            }
        }

        words_routed
    }

    /// Route core stream output to per-tile StreamSwitch slave ports.
    ///
    /// When a core executes StreamWriteScalar or StreamWritePacketHeader,
    /// data is pushed to tile.stream_output. This routes that data to
    /// the Core slave port (port 0) on the tile's StreamSwitch.
    fn route_core_to_tile_switches(&mut self) -> usize {
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Core stream output goes to slave port 0 (Core port)
            // The stream switch then routes it based on configured local routes
            for port in 0..8u8 {
                while let Some(data) = self.tiles[i].pop_stream_output(port) {
                    // For core-initiated streams, use the Core slave port (port 0)
                    // The actual destination is determined by local routes
                    if port == 0 {
                        // Direct core output goes to Core slave port
                        if self.tiles[i].stream_switch.slaves[0].push(data) {
                            words_routed += 1;
                            log::debug!(
                                "Core->TileSwitch: tile({},{}) slave[0] <- 0x{:08X}",
                                self.tiles[i].col, self.tiles[i].row, data
                            );
                        }
                    } else {
                        // Other ports: push to corresponding slave if it exists
                        // This handles multi-port core stream writes
                        let slave_port = port as usize;
                        if slave_port < self.tiles[i].stream_switch.slaves.len() {
                            if self.tiles[i].stream_switch.slaves[slave_port].push(data) {
                                words_routed += 1;
                                log::debug!(
                                    "Core->TileSwitch: tile({},{}) slave[{}] <- 0x{:08X}",
                                    self.tiles[i].col, self.tiles[i].row, slave_port, data
                                );
                            }
                        }
                    }
                }
            }
        }

        words_routed
    }

    /// Route trace unit packets to per-tile stream switch trace slave ports.
    ///
    /// Each tile has trace units that produce 8-word (32-byte) binary trace
    /// packets. These packets enter the stream switch via dedicated trace
    /// slave ports (indices from AM025 via gen_stream_ranges.rs):
    ///   - Compute tile: TRACE_SLAVE_START (core trace), TRACE_SLAVE_END (memory trace)
    ///   - MemTile: TRACE_SLAVE_START (memory trace)
    ///   - Shim: TRACE_SLAVE_START (trace, rarely used)
    ///
    /// Once on the slave port, the existing packet routing infrastructure
    /// handles forwarding to shim DMA and ultimately to host DDR.
    fn route_trace_to_tile_switches(&mut self) -> usize {
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            let tile_type = self.tiles[i].tile_type;

            match tile_type {
                TileType::Compute => {
                    let aie_trace = compute::TRACE_SLAVE_START as usize;
                    let mem_trace = compute::TRACE_SLAVE_END as usize;

                    // Core trace -> AIE_TRACE slave port
                    while self.tiles[i].core_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[aie_trace].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].core_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[aie_trace].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                    // Memory trace -> MEM_TRACE slave port
                    while self.tiles[i].mem_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[mem_trace].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].mem_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[mem_trace].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                }
                TileType::MemTile => {
                    let trace_port = mem_tile::TRACE_SLAVE_START as usize;

                    while self.tiles[i].mem_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[trace_port].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].mem_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[trace_port].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                }
                TileType::Shim => {
                    let trace_port = shim::TRACE_SLAVE_START as usize;

                    while self.tiles[i].mem_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[trace_port].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].mem_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[trace_port].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                }
            }
        }

        words_routed
    }

    /// Route DMA MM2S output to per-tile stream switch slave ports (as arbiter sources).
    ///
    /// In AIE stream switches, DMA MM2S output is presented as a "slave source" that
    /// the arbiter can route to external master ports. Local routes like
    /// `slave[1] -> master[5]` configure master[5] to receive from DMA0 MM2S output.
    ///
    /// DMA slave port ranges come from gen_stream_ranges.rs (AM025-derived).
    fn route_dma_to_tile_switches(&mut self) -> usize {
        use super::stream_switch::PortType;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Route DMA MM2S stream_out to tile switch slave ports with backpressure.
            // Peek first, determine target slave, check if it can accept. Only pop
            // when the push will succeed. This prevents silent data loss when the
            // downstream FIFO is full -- the data stays in stream_out until the
            // next cycle, matching real hardware credit-based flow control.
            loop {
                let data = match self.dma_engines[i].peek_stream_out() {
                    Some(d) => d.clone(),
                    None => break,
                };

                let tile_type = self.tiles[i].tile_type;
                let col = self.tiles[i].col;
                let row = self.tiles[i].row;

                // Determine the target slave port for this DMA MM2S word.
                let target_slave = if tile_type == TileType::Shim {
                    let s2mm_count = self.dma_engines[i].s2mm_channel_count();
                    let mm2s_ch = data.channel.saturating_sub(s2mm_count as u8) as usize;
                    let from_mux = self.tiles[i].shim_mux_mm2s_slaves
                        .get(mm2s_ch)
                        .copied()
                        .flatten();

                    if let Some(slave_idx) = from_mux {
                        Some((slave_idx, format!("Shim Mux MM2S ch{}", mm2s_ch)))
                    } else {
                        // Fallback: find South slave with circuit route to North
                        let mut fallback_slave = None;
                        for route in &self.tiles[i].stream_switch.local_routes {
                            let s = route.slave_idx as usize;
                            let m = route.master_idx as usize;
                            if s < self.tiles[i].stream_switch.slaves.len() &&
                               m < self.tiles[i].stream_switch.masters.len() &&
                               matches!(self.tiles[i].stream_switch.slaves[s].port_type, PortType::South) &&
                               matches!(self.tiles[i].stream_switch.masters[m].port_type, PortType::North) {
                                fallback_slave = Some(s);
                                break;
                            }
                        }
                        if let Some(slave_idx) = fallback_slave {
                            Some((slave_idx, "fallback South->North".to_string()))
                        } else {
                            let msg = format!(
                                "DMA_MM2S->TileSwitch: Shim ({},{}) no route for MM2S ch{} -- \
                                 no slave port or fallback available",
                                col, row, mm2s_ch,
                            );
                            log::error!("{}", msg);
                            self.fatal_errors.push(msg);
                            // Pop to avoid infinite loop on permanently unroutable data
                            self.dma_engines[i].pop_stream_out();
                            continue;
                        }
                    }
                } else {
                    let s2mm_count = self.dma_engines[i].s2mm_channel_count() as u8;
                    let slave_port = match tile_type {
                        TileType::MemTile => {
                            let ch_offset = data.channel.saturating_sub(s2mm_count);
                            (mem_tile::DMA_SLAVE_START + ch_offset) as usize
                        }
                        TileType::Compute => {
                            let ch_offset = data.channel.saturating_sub(s2mm_count);
                            (compute::DMA_SLAVE_START + ch_offset) as usize
                        }
                        TileType::Shim => unreachable!(),
                    };

                    if slave_port < self.tiles[i].stream_switch.slaves.len() {
                        let port_type = self.tiles[i].stream_switch.slaves[slave_port].port_type;
                        if matches!(port_type, PortType::Dma(_)) {
                            Some((slave_port, format!("ch {}, type={:?}", data.channel, port_type)))
                        } else {
                            log::debug!("DMA_MM2S->TileSwitch: tile ({},{}) slave[{}] rejected - wrong type {:?}",
                                col, row, slave_port, port_type);
                            // Pop to avoid infinite loop on misconfigured port
                            self.dma_engines[i].pop_stream_out();
                            continue;
                        }
                    } else {
                        // Pop to avoid infinite loop on out-of-range port
                        self.dma_engines[i].pop_stream_out();
                        continue;
                    }
                };

                // Backpressure: only pop and push if the target can accept.
                if let Some((slave_idx, desc)) = target_slave {
                    if !self.tiles[i].stream_switch.slaves[slave_idx].can_accept() {
                        // Target FIFO full -- leave data in stream_out (backpressure).
                        break;
                    }

                    // Safe to pop and push -- downstream has space.
                    let data = self.dma_engines[i].pop_stream_out().unwrap();
                    self.tiles[i].stream_switch.slaves[slave_idx].push_with_tlast(data.data, data.tlast);
                    words_routed += 1;

                    let prefix = if tile_type == TileType::Shim { "Shim" } else { "tile" };
                    log::info!("DMA_MM2S->TileSwitch: {} ({},{}) slave[{}] <- 0x{:08X}{} ({})",
                        prefix, col, row, slave_idx, data.data, if data.tlast { " TLAST" } else { "" }, desc);
                }
            }
        }

        words_routed
    }

    /// Route per-tile stream switch DMA master output to DMA S2MM input.
    ///
    /// For tiles with local routes, DMA master port output goes to DMA S2MM stream_in.
    /// For Shim tiles, South master ports (2-7) represent the DDR interface and
    /// route to S2MM channels (South0→ch0, South1→ch1, etc.).
    fn route_tile_switches_to_dma(&mut self) -> usize {
        use super::stream_switch::PortType;
        use super::dma::StreamData;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Process ALL tiles - DMA master ports can receive data from any slave source
            // via local routes or direct connections. Removing the local_route_count guard
            // ensures compute tiles can receive stream data from MemTile.
            let is_shim = self.tiles[i].row == 0;

            // Check DMA master ports for outgoing data
            let num_masters = self.tiles[i].stream_switch.masters.len();
            for master_port in 0..num_masters {
                // Determine the DMA channel for this master port
                // - For Compute/MemTile: only PortType::Dma ports route to DMA
                // - For Shim: South ports (2-7) represent DDR interface → S2MM
                let dma_channel = match &self.tiles[i].stream_switch.masters[master_port].port_type {
                    PortType::Dma(ch) => Some(*ch),
                    PortType::South if is_shim => {
                        // Shim South masters represent DDR interface.
                        // The Demux config (0x1F004) determines which South master feeds
                        // each DMA S2MM channel. Match the master port index against
                        // the configured S2MM mappings.
                        let mut matched_ch = None;
                        for (ch, mapped_master) in self.tiles[i].shim_mux_s2mm_masters.iter().enumerate() {
                            if *mapped_master == Some(master_port) {
                                matched_ch = Some(ch as u8);
                                break;
                            }
                        }
                        matched_ch.or(Some(0)) // Fallback to ch0 if no Demux config
                    }
                    _ => None,
                };

                if let Some(ch) = dma_channel {
                    // Debug: check if master has data
                    let fifo_len = self.tiles[i].stream_switch.masters[master_port].fifo.len();
                    if fifo_len > 0 {
                        log::debug!("TileSwitch->DMA check: tile ({},{}) master[{}] ch {} fifo_len={}",
                            self.tiles[i].col, self.tiles[i].row, master_port, ch, fifo_len);
                    }

                    // Per-channel backpressure: only pop from master if the
                    // target channel's FIFO has space. Each S2MM channel has its
                    // own buffer, so one channel can't block another.
                    if !self.dma_engines[i].can_accept_stream_in_for_channel(ch) {
                        continue;
                    }

                    if let Some((data, tlast)) = self.tiles[i].stream_switch.masters[master_port].pop_with_tlast() {
                        // Push to DMA S2MM stream_in
                        let stream_data = StreamData {
                            data,
                            tlast,
                            channel: ch,
                        };
                        let push_result = self.dma_engines[i].push_stream_in(stream_data);
                        let new_len = self.dma_engines[i].stream_in_len();
                        if push_result {
                            words_routed += 1;
                            log::info!("TileSwitch->DMA: tile ({},{}) master[{}] -> DMA ch {} = 0x{:08X} (stream_in_len={})",
                                self.tiles[i].col, self.tiles[i].row, master_port, ch, data, new_len);
                        } else {
                            let msg = format!(
                                "TileSwitch->DMA: push_stream_in FAILED for tile ({},{}) ch {} data=0x{:08X} -- \
                                 DMA input buffer overflow (impossible with hardware backpressure)",
                                self.tiles[i].col, self.tiles[i].row, ch, data,
                            );
                            log::error!("{}", msg);
                            self.fatal_errors.push(msg);
                        }
                    }
                }
            }
        }

        words_routed
    }

    /// Propagate data between adjacent tiles via directional ports.
    ///
    /// This is the critical inter-tile data movement phase. When a tile's North
    /// master port has data, it flows to the South slave port of the tile above.
    /// Similarly for South→North (downward) and East/West (horizontal) connections.
    ///
    /// Per AM025, the North/South port mappings are:
    /// - **Shim→MemTile**: Shim North masters 12-17 → MemTile South slaves 7-12
    /// - **MemTile→Compute**: MemTile North masters 11-16 → Compute South slaves 5-10
    /// - **Compute→MemTile**: Compute South masters 5-10 → MemTile North slaves 13-16 (only 4)
    /// - **MemTile→Shim**: MemTile South masters 7-10 → Shim North slaves 14-17
    /// - **Compute→Compute**: Compute South masters 5-10 → (below) Compute South slaves 5-10
    ///
    /// East/West port mappings (same-type adjacency only, MemTiles have no E/W):
    /// - **Compute East→West**: East masters 19-22 → West slaves 11-14
    /// - **Compute West→East**: West masters 9-12 → East slaves 19-22
    /// - **Shim East→West**: East masters 18-21 → West slaves 10-13
    /// - **Shim West→East**: West masters 8-11 → East slaves 18-21
    ///
    /// Returns the number of words transferred between tiles.
    fn propagate_inter_tile(&mut self) -> usize {
        // Phase 1: Advance the pipeline -- deliver words that have completed
        // their inter-tile traversal and decrement countdown timers.
        let words_transferred = self.advance_inter_tile_pipeline();

        // Phase 2: Accept new words from source masters into the pipeline.
        // Each transfer: (src_col, src_row, src_master_idx, dst_col, dst_row, dst_slave_idx, data, tlast)
        let mut transfers: Vec<(u8, u8, usize, u8, u8, usize, u32, bool)> = Vec::new();

        // Iterate through all tiles and check North/South master ports
        for col in 0..self.cols {
            for row in 0..self.rows {
                let idx = self.tile_index(col, row);
                let tile_type = self.tiles[idx].tile_type;

                // Check North masters - data flows to tile above (row + 1)
                if row + 1 < self.rows {
                    let above_idx = self.tile_index(col, row + 1);
                    let above_type = self.tiles[above_idx].tile_type;

                    // Determine port mappings based on tile types (AM025-derived constants)
                    let (north_master_start, north_master_count, south_slave_start) = match (tile_type, above_type) {
                        (TileType::Shim, TileType::MemTile) => (
                            shim::NORTH_MASTER_START as usize,
                            (shim::NORTH_MASTER_END - shim::NORTH_MASTER_START + 1) as usize,
                            mem_tile::SOUTH_SLAVE_START as usize,
                        ),
                        (TileType::MemTile, TileType::Compute) => (
                            mem_tile::NORTH_MASTER_START as usize,
                            (mem_tile::NORTH_MASTER_END - mem_tile::NORTH_MASTER_START + 1) as usize,
                            compute::SOUTH_SLAVE_START as usize,
                        ),
                        (TileType::Compute, TileType::Compute) => (
                            compute::NORTH_MASTER_START as usize,
                            (compute::NORTH_MASTER_END - compute::NORTH_MASTER_START + 1) as usize,
                            compute::SOUTH_SLAVE_START as usize,
                        ),
                        _ => continue,
                    };

                    // Transfer from each North master to corresponding South slave
                    for i in 0..north_master_count {
                        let master_idx = north_master_start + i;
                        let slave_idx = south_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.first(), port.peek_tlast()) {
                                // Check if destination slave can accept
                                if slave_idx < self.tiles[above_idx].stream_switch.slaves.len()
                                    && self.tiles[above_idx].stream_switch.slaves[slave_idx].can_accept()
                                {
                                    transfers.push((col, row, master_idx, col, row + 1, slave_idx, data, tlast));
                                }
                            }
                        }
                    }
                }

                // Check South masters - data flows to tile below (row - 1)
                if row > 0 {
                    let below_idx = self.tile_index(col, row - 1);
                    let below_type = self.tiles[below_idx].tile_type;

                    // Determine port mappings based on tile types (AM025-derived constants)
                    let (south_master_start, south_master_count, north_slave_start) = match (tile_type, below_type) {
                        (TileType::MemTile, TileType::Shim) => (
                            mem_tile::SOUTH_MASTER_START as usize,
                            (mem_tile::SOUTH_MASTER_END - mem_tile::SOUTH_MASTER_START + 1) as usize,
                            shim::NORTH_SLAVE_START as usize,
                        ),
                        (TileType::Compute, TileType::MemTile) => (
                            compute::SOUTH_MASTER_START as usize,
                            (compute::SOUTH_MASTER_END - compute::SOUTH_MASTER_START + 1) as usize,
                            mem_tile::NORTH_SLAVE_START as usize,
                        ),
                        (TileType::Compute, TileType::Compute) => (
                            compute::SOUTH_MASTER_START as usize,
                            (compute::SOUTH_MASTER_END - compute::SOUTH_MASTER_START + 1) as usize,
                            compute::NORTH_SLAVE_START as usize,
                        ),
                        _ => continue,
                    };

                    // Transfer from each South master to corresponding North slave
                    for i in 0..south_master_count {
                        let master_idx = south_master_start + i;
                        let slave_idx = north_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.first(), port.peek_tlast()) {
                                // Check if destination slave can accept
                                if slave_idx < self.tiles[below_idx].stream_switch.slaves.len()
                                    && self.tiles[below_idx].stream_switch.slaves[slave_idx].can_accept()
                                {
                                    transfers.push((col, row, master_idx, col, row - 1, slave_idx, data, tlast));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check East masters - data flows to tile to the right (col + 1)
        for col in 0..self.cols {
            for row in 0..self.rows {
                let idx = self.tile_index(col, row);
                let tile_type = self.tiles[idx].tile_type;

                if col + 1 < self.cols {
                    let right_idx = self.tile_index(col + 1, row);
                    let right_type = self.tiles[right_idx].tile_type;

                    // East masters on source -> West slaves on destination (AM025-derived)
                    let (east_master_start, east_count, west_slave_start) = match (tile_type, right_type) {
                        (TileType::Compute, TileType::Compute) => (
                            compute::EAST_MASTER_START as usize,
                            (compute::EAST_MASTER_END - compute::EAST_MASTER_START + 1) as usize,
                            compute::WEST_SLAVE_START as usize,
                        ),
                        (TileType::Shim, TileType::Shim) => (
                            shim::EAST_MASTER_START as usize,
                            (shim::EAST_MASTER_END - shim::EAST_MASTER_START + 1) as usize,
                            shim::WEST_SLAVE_START as usize,
                        ),
                        _ => continue,
                    };

                    for i in 0..east_count {
                        let master_idx = east_master_start + i;
                        let slave_idx = west_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.first(), port.peek_tlast()) {
                                if slave_idx < self.tiles[right_idx].stream_switch.slaves.len()
                                    && self.tiles[right_idx].stream_switch.slaves[slave_idx].can_accept()
                                {
                                    transfers.push((col, row, master_idx, col + 1, row, slave_idx, data, tlast));
                                }
                            }
                        }
                    }
                }

                // West masters on source -> East slaves on destination (col - 1)
                if col > 0 {
                    let left_idx = self.tile_index(col - 1, row);
                    let left_type = self.tiles[left_idx].tile_type;

                    // West masters on source -> East slaves on destination (AM025-derived)
                    let (west_master_start, west_count, east_slave_start) = match (tile_type, left_type) {
                        (TileType::Compute, TileType::Compute) => (
                            compute::WEST_MASTER_START as usize,
                            (compute::WEST_MASTER_END - compute::WEST_MASTER_START + 1) as usize,
                            compute::EAST_SLAVE_START as usize,
                        ),
                        (TileType::Shim, TileType::Shim) => (
                            shim::WEST_MASTER_START as usize,
                            (shim::WEST_MASTER_END - shim::WEST_MASTER_START + 1) as usize,
                            shim::EAST_SLAVE_START as usize,
                        ),
                        _ => continue,
                    };

                    for i in 0..west_count {
                        let master_idx = west_master_start + i;
                        let slave_idx = east_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.first(), port.peek_tlast()) {
                                if slave_idx < self.tiles[left_idx].stream_switch.slaves.len()
                                    && self.tiles[left_idx].stream_switch.slaves[slave_idx].can_accept()
                                {
                                    transfers.push((col, row, master_idx, col - 1, row, slave_idx, data, tlast));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pop words from source masters into the inter-tile pipeline.
        // Words will be delivered after ROUTE_LATENCY_PER_HOP cycles.
        use super::aie2_spec::ROUTE_LATENCY_PER_HOP;

        for (src_col, src_row, src_master, dst_col, dst_row, dst_slave, _data, _tlast) in transfers {
            let src_idx = self.tile_index(src_col, src_row);
            let dst_idx = self.tile_index(dst_col, dst_row);

            // Pop from source master (with TLAST)
            if let Some((data, tlast)) = self.tiles[src_idx].stream_switch.masters[src_master].pop_with_tlast() {
                self.inter_tile_pipeline.push(InFlightWord {
                    dst_tile_idx: dst_idx,
                    dst_slave_idx: dst_slave,
                    data,
                    tlast,
                    cycles_remaining: ROUTE_LATENCY_PER_HOP,
                });
                log::debug!(
                    "InterTile: ({},{}) master[{}] -> pipeline({}) -> ({},{}) slave[{}] = 0x{:08X}{} delay={}cy",
                    src_col, src_row, src_master, ROUTE_LATENCY_PER_HOP,
                    dst_col, dst_row, dst_slave, data,
                    if tlast { " TLAST" } else { "" },
                    ROUTE_LATENCY_PER_HOP,
                );
            }
        }

        words_transferred
    }

    /// Advance the inter-tile pipeline by one cycle.
    ///
    /// Called at the start of `propagate_inter_tile()`. Decrements countdown
    /// timers and delivers words that have completed their traversal.
    /// Returns the number of words delivered.
    fn advance_inter_tile_pipeline(&mut self) -> usize {
        let mut delivered = 0;
        let mut i = 0;

        while i < self.inter_tile_pipeline.len() {
            let word = &mut self.inter_tile_pipeline[i];
            if word.cycles_remaining > 0 {
                word.cycles_remaining -= 1;
            }

            if word.cycles_remaining == 0 {
                // Try to deliver to destination slave
                let dst_idx = word.dst_tile_idx;
                let dst_slave = word.dst_slave_idx;

                if dst_slave < self.tiles[dst_idx].stream_switch.slaves.len()
                    && self.tiles[dst_idx].stream_switch.slaves[dst_slave].can_accept()
                {
                    let data = word.data;
                    let tlast = word.tlast;
                    self.tiles[dst_idx].stream_switch.slaves[dst_slave]
                        .push_with_tlast(data, tlast);
                    self.inter_tile_pipeline.swap_remove(i);
                    delivered += 1;
                    // Don't increment i -- swap_remove moved the last element here
                } else {
                    // Destination can't accept -- word stays in pipeline (backpressure).
                    // It will be retried next cycle with cycles_remaining still at 0.
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        delivered
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
            // Recreate stream switch based on tile type (they have different port configurations)
            tile.stream_switch = match tile.tile_type {
                TileType::Shim => super::stream_switch::StreamSwitch::new_shim_tile(tile.col),
                TileType::MemTile => super::stream_switch::StreamSwitch::new_mem_tile(tile.col, tile.row),
                TileType::Compute => super::stream_switch::StreamSwitch::new_compute_tile(tile.col, tile.row),
            };
            // Note: We don't zero memory here for performance
            // Call zero_memory() explicitly if needed
        }

        // Reset all DMA engines
        for engine in &mut self.dma_engines {
            engine.reset();
        }

        // Clear inter-tile pipeline
        self.inter_tile_pipeline.clear();
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
        println!("Tile Array: {} ({} cols × {} rows)", self.arch.name(), self.cols, self.rows);
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

    // === Cascade Routing Tests ===

    #[test]
    fn test_cascade_route_south() {
        // NPU1: 5 cols x 6 rows. Compute tiles at rows 2-5.
        let mut array = TileArray::npu1();

        // Configure tile (1,3): output direction = South (0)
        // Configure tile (1,2): input direction = North (0)
        array.tile_mut(1, 3).write_register(0x36060, 0b00); // in=North, out=South
        array.tile_mut(1, 2).write_register(0x36060, 0b00); // in=North, out=South

        // Push cascade data to tile (1,3) output
        let data: [u64; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        array.tile_mut(1, 3).push_cascade_output(data);

        // Route cascade
        array.route_cascade();

        // Data should arrive at tile (1,2) input
        assert!(array.tile(1, 2).has_cascade_input());
        assert!(!array.tile(1, 3).has_cascade_output());

        let received = array.tile_mut(1, 2).pop_cascade_input().unwrap();
        assert_eq!(received, data);
    }

    #[test]
    fn test_cascade_route_east() {
        let mut array = TileArray::npu1();

        // Configure tile (1,3): output direction = East (1)
        array.tile_mut(1, 3).write_register(0x36060, 0b10); // in=North, out=East
        // Configure tile (2,3): input direction = West (1)
        array.tile_mut(2, 3).write_register(0x36060, 0b01); // in=West, out=South

        let data: [u64; 6] = [1, 2, 3, 4, 5, 6];
        array.tile_mut(1, 3).push_cascade_output(data);

        array.route_cascade();

        assert!(array.tile(2, 3).has_cascade_input());
        assert!(!array.tile(1, 3).has_cascade_output());

        let received = array.tile_mut(2, 3).pop_cascade_input().unwrap();
        assert_eq!(received, data);
    }

    #[test]
    fn test_cascade_backpressure() {
        let mut array = TileArray::npu1();

        // Configure South cascade path: (1,3) -> (1,2)
        array.tile_mut(1, 3).write_register(0x36060, 0b00);
        array.tile_mut(1, 2).write_register(0x36060, 0b00);

        // Fill destination's input FIFO
        array.tile_mut(1, 2).push_cascade_input([0; 6]);

        // Push data to source
        let data: [u64; 6] = [42; 6];
        array.tile_mut(1, 3).push_cascade_output(data);

        // Route should NOT transfer (backpressure)
        array.route_cascade();

        // Source still has data, destination still has old data
        assert!(array.tile(1, 3).has_cascade_output());
        assert!(array.tile(1, 2).has_cascade_input());
    }

    #[test]
    fn test_cascade_direction_mismatch_no_route() {
        let mut array = TileArray::npu1();

        // Source outputs South, but destination expects West (mismatch)
        array.tile_mut(1, 3).write_register(0x36060, 0b00); // out=South
        array.tile_mut(1, 2).write_register(0x36060, 0b01); // in=West (wrong!)

        let data: [u64; 6] = [99; 6];
        array.tile_mut(1, 3).push_cascade_output(data);

        array.route_cascade();

        // No transfer because directions don't match
        assert!(array.tile(1, 3).has_cascade_output());
        assert!(!array.tile(1, 2).has_cascade_input());
    }

    /// OP_READ response injection: handle_read_registers queues response
    /// words, drain_ctrl_responses pushes them into the TileCtrl slave
    /// port's FIFO across multiple drain cycles.
    #[test]
    fn test_handle_read_registers_injects_response() {
        use crate::device::stream_switch::{PacketHeader, PortType};

        let mut array = TileArray::npu1();

        // Use compute tile at (2,3)
        let col: u8 = 2;
        let row: u8 = 3;

        // Pre-populate 4 consecutive registers with known values
        let base_offset: u32 = 0x440;
        let values = [0xAAAA_0001u32, 0xBBBB_0002, 0xCCCC_0003, 0xDDDD_0004];
        for (i, &val) in values.iter().enumerate() {
            array.tile_mut(col, row).write_register(base_offset + (i as u32) * 4, val);
        }

        // Verify the TileCtrl slave port exists (port 3 on compute tiles)
        let ctrl_slave_idx = array.tile(col, row).stream_switch
            .tile_ctrl_slave_port()
            .expect("compute tile should have TileCtrl slave port");
        assert!(
            matches!(
                array.tile(col, row).stream_switch.slave(ctrl_slave_idx).unwrap().port_type,
                PortType::TileCtrl,
            ),
            "slave port at ctrl index should be TileCtrl"
        );

        // Call handle_read_registers with response_id=5, count=4
        let response_id: u8 = 5;
        let count: u8 = 4;
        let ok = array.handle_read_registers(col, row, base_offset, count, response_id);
        assert!(ok, "handle_read_registers should succeed");

        // Response words are queued, not yet in the FIFO
        assert_eq!(
            array.tile(col, row).pending_ctrl_response.len(), 5,
            "pending buffer should have header + 4 data words"
        );

        // Drain cycle 1: fills slave FIFO up to capacity (4 words)
        let injected1 = array.drain_ctrl_responses();
        assert_eq!(injected1, 4, "first drain should inject 4 words (FIFO capacity)");
        assert_eq!(
            array.tile(col, row).pending_ctrl_response.len(), 1,
            "1 word should remain in pending buffer"
        );

        // Pop and verify the stream header
        let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_slave_idx).unwrap();
        let (header_word, header_tlast) = slave.pop_with_tlast().unwrap();
        assert!(!header_tlast, "header should not have TLAST");

        let (decoded, parity_ok) = PacketHeader::decode(header_word);
        assert!(parity_ok, "packet header parity should be valid");
        assert_eq!(decoded.stream_id, response_id, "stream_id should equal response_id");
        assert_eq!(decoded.src_col, col, "src_col should match tile column");
        assert_eq!(decoded.src_row, row, "src_row should match tile row");

        // Pop first 3 data words (popping frees FIFO space)
        for (i, &expected) in values[..3].iter().enumerate() {
            let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_slave_idx).unwrap();
            let (data, tlast) = slave.pop_with_tlast().unwrap();
            assert_eq!(data, expected, "data word {} should match register value", i);
            assert!(!tlast, "non-last data word {} should not have TLAST", i);
        }

        // Drain cycle 2: pushes the remaining word
        let injected2 = array.drain_ctrl_responses();
        assert_eq!(injected2, 1, "second drain should inject remaining 1 word");
        assert!(
            array.tile(col, row).pending_ctrl_response.is_empty(),
            "pending buffer should be empty after second drain"
        );

        // Pop and verify the last data word
        let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_slave_idx).unwrap();
        let (data, tlast) = slave.pop_with_tlast().unwrap();
        assert_eq!(data, values[3], "last data word should match register value");
        assert!(tlast, "last data word should have TLAST");

        // FIFO should be empty now
        let slave = array.tile(col, row).stream_switch.slave(ctrl_slave_idx).unwrap();
        assert_eq!(slave.fifo_level(), 0, "FIFO should be empty after draining");
    }

    /// OP_READ with count=1: queue + drain produces header + 1 data word,
    /// where the data word has TLAST set.
    #[test]
    fn test_handle_read_registers_single_word() {
        let mut array = TileArray::npu1();

        let col: u8 = 1;
        let row: u8 = 2;
        let offset: u32 = 0x500;
        let expected: u32 = 0x1234_5678;
        array.tile_mut(col, row).write_register(offset, expected);

        let ok = array.handle_read_registers(col, row, offset, 1, 0);
        assert!(ok);

        // Drain response into FIFO
        let injected = array.drain_ctrl_responses();
        assert_eq!(injected, 2, "header + 1 data word should fit in one drain");

        let ctrl_idx = array.tile(col, row).stream_switch.tile_ctrl_slave_port().unwrap();
        let slave = array.tile(col, row).stream_switch.slave(ctrl_idx).unwrap();
        assert_eq!(slave.fifo_level(), 2, "header + 1 data word");

        // Pop header (no TLAST)
        let slave = array.tile_mut(col, row).stream_switch.slave_mut(ctrl_idx).unwrap();
        let (_, h_tlast) = slave.pop_with_tlast().unwrap();
        assert!(!h_tlast, "header should not have TLAST when count > 0");

        // Pop data word (with TLAST)
        let (data, d_tlast) = slave.pop_with_tlast().unwrap();
        assert_eq!(data, expected);
        assert!(d_tlast, "single data word should have TLAST");
    }

    /// OP_READ for a non-existent tile should fail gracefully.
    #[test]
    fn test_handle_read_registers_bad_tile() {
        let mut array = TileArray::npu1();
        let ok = array.handle_read_registers(99, 99, 0x440, 4, 0);
        assert!(!ok, "should fail for out-of-bounds tile");
    }
}
