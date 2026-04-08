//! DMA engine implementation.
//!
//! The DMA engine manages buffer descriptors and channels for a single tile.
//! It coordinates transfers, handles BD chaining, and tracks channel state.
//!
//! # Channel Architecture
//!
//! Compute tiles have 4 channels:
//! - Channels 0-1: S2MM (Stream to Memory) - receive from stream, write to memory
//! - Channels 2-3: MM2S (Memory to Stream) - read from memory, send to stream
//!
//! Memory tiles have 12 channels:
//! - Channels 0-5: S2MM
//! - Channels 6-11: MM2S
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::device::dma::{DmaEngine, BdConfig};
//!
//! let mut engine = DmaEngine::new_compute_tile(1, 2);
//!
//! // Configure a BD
//! engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256))?;
//!
//! // Start channel 0 using BD 0
//! engine.start_channel(0, 0)?;
//!
//! // Step the engine each cycle
//! while engine.any_channel_active() {
//!     engine.step(&mut tile_memory, &mut NeighborLocks::empty(), &mut host_memory)?;
//! }
//! ```

mod types;
mod stepping;
mod locks;
mod stream_io;
mod task_queue_ops;
mod status;

#[cfg(test)]
mod tests;

pub use types::*;

use std::collections::VecDeque;

use super::{BdConfig, ChannelType, DmaError, DmaResult};
use super::compression;
use super::transfer::{Transfer, TransferDirection, TransferEndpoint, PadAction};
use super::channel::{ChannelFsm, ChannelContext, CompletionInfo};
use super::timing::DmaTimingConfig;
use crate::device::host_memory::HostMemory;
use crate::device::tile::{Tile, TileType};
use crate::interpreter::state::EventType;
use crate::interpreter::timing::sync::LockTimingState;

/// DMA engine for a single tile.
#[derive(Debug)]
pub struct DmaEngine {
    /// Tile column
    pub col: u8,

    /// Tile row
    pub row: u8,

    /// Tile type (affects channel count, BD count, and transfer endpoints)
    pub tile_type: TileType,

    /// Buffer descriptor configurations
    pub(super) bd_configs: Vec<BdConfig>,

    /// Dirty flags for BDs written word-by-word (e.g., control packets).
    /// When true, the BD must be re-parsed from tile raw storage before use.
    bd_dirty: Vec<bool>,

    /// Per-channel state (FSM, task queue, stats, BD tracking).
    /// Replaces 12 parallel Vec<T> fields with one struct per channel.
    pub(super) channels: Vec<ChannelContext>,

    /// Timing configuration (controls cycle-accuracy vs fast mode)
    pub(super) timing_config: DmaTimingConfig,

    /// Stream output buffer (MM2S channels produce data here).
    /// Data is read from tile memory and queued for the stream router.
    pub(super) stream_out: VecDeque<StreamData>,

    /// Per-channel stream input buffers (S2MM channels consume data from here).
    /// Each S2MM channel has its own FIFO, matching real hardware where each
    /// channel connects to a dedicated stream switch master port. This prevents
    /// one channel's traffic (e.g., trace) from blocking another (e.g., output).
    pub(super) stream_in: Vec<VecDeque<StreamData>>,

    /// Task complete token output buffer.
    /// Tokens are emitted when tasks complete with Enable_Token_Issue set.
    pub(super) task_tokens: TokenState,

    /// Lock timing state for contention tracking (optional).
    /// When enabled, tracks detailed lock acquire/release timing per lock.
    pub(super) lock_timing: Option<LockTimingState>,

    /// Current cycle, set by coordinator before each step.
    /// Used for timestamping trace events.
    pub(super) current_cycle: u64,

    /// Trace events generated during this step, drained by coordinator.
    /// Events are buffered here because DmaEngine doesn't own an EventLog
    /// directly -- the coordinator collects them after each step.
    pub(super) trace_events: Vec<(u64, EventType)>,

    /// Number of locks per tile (16 for compute, 64 for MemTile, 0 for shim).
    /// Used by resolve_lock_id() for cross-tile lock addressing.
    pub(super) num_locks: u8,

    /// Number of S2MM channels (stream-to-memory) for this tile.
    /// Derived from architecture configuration at construction.
    pub(super) s2mm_count: usize,

    /// Number of MM2S channels (memory-to-stream) for this tile.
    /// Derived from architecture configuration at construction.
    pub(super) mm2s_count: usize,

    /// Number of memory banks for this tile type (8 for compute, 16 for MemTile).
    /// Used to compute bank indices for conflict detection.
    pub(super) num_banks: usize,

    /// Bitmask of memory banks accessed by DMA during this cycle.
    /// Transferred to the tile at the end of each DMA step for comparison
    /// with core bank accesses.
    pub cycle_dma_banks: u16,

    /// Fatal errors accumulated during DMA operations.
    ///
    /// Conditions that are impossible on real hardware (memory bounds
    /// violations, lock address overflow, missing neighbor tiles, buffer
    /// overflow). The owning TileArray drains these after each step.
    pub fatal_errors: Vec<String>,
}

impl DmaEngine {
    /// Create a new DMA engine with the given channel and BD counts.
    ///
    /// `num_channels` and `num_bds` come from the architecture configuration
    /// (ArchConfig) rather than compile-time constants. The caller is
    /// responsible for providing the correct values for the tile type.
    pub fn new(col: u8, row: u8, tile_type: TileType, s2mm_channels: usize, mm2s_channels: usize, num_bds: usize, num_locks: u8) -> Self {
        let num_channels = s2mm_channels + mm2s_channels;
        log::debug!("DmaEngine::new col={} row={} tile_type={:?} num_channels={} (s2mm={}, mm2s={}) num_bds={} num_locks={}",
            col, row, tile_type, num_channels, s2mm_channels, mm2s_channels, num_bds, num_locks);

        let channels = (0..num_channels)
            .map(|i| ChannelContext::new(i as u8))
            .collect();

        Self {
            col,
            row,
            tile_type,
            bd_configs: vec![BdConfig::default(); num_bds],
            bd_dirty: vec![false; num_bds],
            channels,
            timing_config: DmaTimingConfig::from_arch(),
            stream_out: VecDeque::with_capacity(16),
            stream_in: (0..s2mm_channels).map(|_| VecDeque::with_capacity(16)).collect(),
            task_tokens: TokenState::new(),
            lock_timing: None,
            current_cycle: 0,
            trace_events: Vec::new(),
            s2mm_count: s2mm_channels,
            mm2s_count: mm2s_channels,
            num_locks,
            num_banks: match tile_type {
                TileType::Compute => crate::arch::compute::PHYSICAL_BANKS as usize,
                TileType::MemTile => crate::arch::memtile::PHYSICAL_BANKS as usize,
                TileType::Shim => 0,
            },
            cycle_dma_banks: 0,
            fatal_errors: Vec::new(),
        }
    }

    /// Create a compute tile DMA engine with AIE2 defaults (2+2 channels, 16 BDs).
    ///
    /// Test convenience constructor. Production code uses `new()` with
    /// ArchConfig-derived values (see `DeviceArray::new()`).
    #[cfg(test)]
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::Compute, 2, 2, 16, 16)
    }

    /// Create a memory tile DMA engine with AIE2 defaults (6+6 channels, 48 BDs).
    #[cfg(test)]
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::MemTile, 6, 6, 48, 64)
    }

    /// Create a shim tile DMA engine with AIE2 defaults (2+2 channels, 16 BDs).
    #[cfg(test)]
    pub fn new_shim_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::Shim, 2, 2, 16, 0)
    }

    /// Configure custom timing parameters.
    ///
    /// The default is already cycle-accurate (`DmaTimingConfig::from_arch()`).
    /// This method allows tuning specific timing values if needed.
    pub fn with_timing(mut self, config: DmaTimingConfig) -> Self {
        self.timing_config = config;
        self
    }

    /// Enable lock timing and contention tracking.
    ///
    /// When enabled, tracks detailed statistics per lock:
    /// - Acquire/release counts
    /// - Contention cycles (time spent waiting for lock)
    /// - Maximum contention observed
    ///
    /// # Arguments
    ///
    /// * `num_locks` - Number of locks to track (typically 16 for compute, 64 for mem tile)
    pub fn with_lock_timing(mut self, num_locks: usize) -> Self {
        self.lock_timing = Some(LockTimingState::new(num_locks));
        self
    }

    /// Get the lock timing state (if enabled).
    pub fn lock_timing(&self) -> Option<&LockTimingState> {
        self.lock_timing.as_ref()
    }

    /// Get mutable lock timing state (if enabled).
    pub fn lock_timing_mut(&mut self) -> Option<&mut LockTimingState> {
        self.lock_timing.as_mut()
    }

    /// Set the current cycle for trace event timestamps.
    /// Called by the coordinator before each step.
    pub fn set_current_cycle(&mut self, cycle: u64) {
        self.current_cycle = cycle;
    }

    /// Record a trace event at the current cycle.
    #[inline]
    pub(super) fn trace(&mut self, event: EventType) {
        self.trace_events.push((self.current_cycle, event));
    }

    /// Drain all buffered trace events. Called by the coordinator after
    /// each step to collect events into the global trace log.
    pub fn drain_trace_events(&mut self) -> Vec<(u64, EventType)> {
        std::mem::take(&mut self.trace_events)
    }

    /// Get the timing configuration.
    pub fn timing_config(&self) -> &DmaTimingConfig {
        &self.timing_config
    }

    /// Get the number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Get the DMA channel status field layout for this engine's tile type.
    ///
    /// Returns the compute tile layout for compute/shim tiles, and the
    /// memtile layout for memory tiles. The key difference is the `cur_bd`
    /// field width: 4 bits for compute (16 BDs), 6 bits for memtile (48 BDs).
    pub(super) fn status_layout(&self) -> &'static crate::device::regdb::StatusFieldLayout {
        let layout = crate::device::regdb::device_reg_layout();
        if self.tile_type.is_mem_tile() {
            &layout.memtile_status
        } else {
            &layout.memory_status
        }
    }

    /// Get the channel type (S2MM or MM2S).
    pub fn channel_type(&self, channel: ChannelId) -> ChannelType {
        ChannelType::from_channel_index(channel as usize, self.s2mm_count)
    }

    /// Convert a flat channel index to its per-direction channel number.
    ///
    /// In the emulator, channels are numbered as a flat array: S2MM channels
    /// first, then MM2S. For a MemTile with 6 S2MM + 6 MM2S:
    ///   - Flat 0..5  -> S2MM per-direction 0..5
    ///   - Flat 6..11 -> MM2S per-direction 0..5
    ///
    /// This matches aie-rt's convention where ChNum is always per-direction.
    pub(super) fn per_direction_channel(&self, channel: ChannelId) -> u8 {
        let ch = channel as usize;
        if ch < self.s2mm_count {
            channel
        } else {
            (ch - self.s2mm_count) as u8
        }
    }

    /// Check MemTile BD-channel validity per aie-rt
    /// `_XAieMl_MemTileDmaCheckBdChValidity()`.
    ///
    /// MemTile BDs 0-23 are valid only for even per-direction channels (0, 2, 4)
    /// and BDs 24-47 are valid only for odd per-direction channels (1, 3, 5).
    /// This constraint reflects hardware wiring of BD register banks to DMA
    /// channel pairs.
    ///
    /// Returns true if the combination is valid (or the tile is not a MemTile).
    /// Returns false for invalid MemTile BD-channel combinations.
    pub(super) fn check_memtile_bd_channel_validity(&self, bd_index: u8, channel: ChannelId) -> bool {
        if !self.tile_type.is_mem_tile() {
            return true;
        }

        let dir_ch = self.per_direction_channel(channel);

        if bd_index < 24 && dir_ch % 2 == 0 {
            return true;
        }
        if bd_index >= 24 && dir_ch % 2 == 1 {
            return true;
        }

        false
    }

    /// Configure a buffer descriptor.
    pub fn configure_bd(&mut self, bd_index: u8, config: BdConfig) -> Result<(), DmaError> {
        if bd_index as usize >= self.bd_configs.len() {
            return Err(DmaError::InvalidBd(bd_index));
        }

        self.bd_configs[bd_index as usize] = config;
        self.clear_bd_dirty(bd_index);
        Ok(())
    }

    /// Mark a BD as needing re-parse from raw words before use.
    ///
    /// Called when single-word BD register writes modify the BD storage
    /// without providing the full word set needed for parsing.
    pub fn mark_bd_dirty(&mut self, bd_index: u8) {
        if let Some(d) = self.bd_dirty.get_mut(bd_index as usize) {
            *d = true;
        }
    }

    /// Clear the dirty flag for a BD (called after configure_bd or re-parse).
    pub fn clear_bd_dirty(&mut self, bd_index: u8) {
        if let Some(d) = self.bd_dirty.get_mut(bd_index as usize) {
            *d = false;
        }
    }

    /// Check whether a BD needs re-parsing from raw tile storage.
    pub fn is_bd_dirty(&self, bd_index: u8) -> bool {
        self.bd_dirty.get(bd_index as usize).copied().unwrap_or(false)
    }

    /// Number of buffer descriptors available on this DMA engine.
    pub fn num_bds(&self) -> usize {
        self.bd_configs.len()
    }

    /// Get a buffer descriptor configuration.
    pub fn get_bd(&self, bd_index: u8) -> Option<&BdConfig> {
        self.bd_configs.get(bd_index as usize)
    }

    /// Get a mutable buffer descriptor configuration.
    pub fn get_bd_mut(&mut self, bd_index: u8) -> Option<&mut BdConfig> {
        self.bd_configs.get_mut(bd_index as usize)
    }

    /// Start a channel with the specified BD.
    pub fn start_channel(&mut self, channel: ChannelId, bd_index: u8) -> Result<(), DmaError> {
        self.start_channel_with_repeat(channel, bd_index, 0)?;
        self.channels[channel as usize].chain_start_bd = Some(bd_index);
        Ok(())
    }

    /// Start a channel with the specified BD and repeat count.
    ///
    /// The repeat_count indicates how many additional times to run the BD after
    /// the first execution. So repeat_count=0 means run once, repeat_count=3
    /// means run 4 times total.
    pub fn start_channel_with_repeat(
        &mut self,
        channel: ChannelId,
        bd_index: u8,
        repeat_count: u8,
    ) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        if bd_index as usize >= self.bd_configs.len() {
            return Err(DmaError::InvalidBd(bd_index));
        }

        // MemTile BD-channel validity check (per aie-rt
        // _XAieMl_MemTileDmaCheckBdChValidity): BDs 0-23 are valid only
        // for even per-direction channels (0, 2, 4) and BDs 24-47 are
        // valid only for odd per-direction channels (1, 3, 5).
        if !self.check_memtile_bd_channel_validity(bd_index, channel) {
            let dir_ch = self.per_direction_channel(channel);
            log::warn!(
                "DMA tile({},{}) invalid MemTile BD-channel combination: \
                 BD {} is only valid for {} channels, but per-direction channel {} is {}",
                self.col, self.row, bd_index,
                if bd_index < 24 { "even" } else { "odd" },
                dir_ch,
                if dir_ch % 2 == 0 { "even" } else { "odd" },
            );
        }

        // Check if channel is already busy
        if self.channels[ch_idx].is_active() {
            return Err(DmaError::ChannelBusy(channel));
        }

        let bd_config = &self.bd_configs[bd_index as usize];

        // Determine transfer direction
        let direction = match self.channel_type(channel) {
            ChannelType::S2MM => TransferDirection::S2MM,
            ChannelType::MM2S => TransferDirection::MM2S,
        };

        // Create transfer
        let transfer = Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row, self.tile_type)?;

        log::info!("DMA tile({},{}) ch{} BD{} start: total_bytes={} base_addr=0x{:X} next_bd={:?} acq_lock={:?}(val={}) rel_lock={:?}(val={}) pkt={}(id={}) dir={:?}",
            self.col, self.row, channel, bd_index,
            transfer.total_bytes, bd_config.base_addr,
            bd_config.next_bd,
            bd_config.acquire_lock, bd_config.acquire_value,
            bd_config.release_lock, bd_config.release_value,
            transfer.enable_packet, transfer.packet_id, direction);

        // Packet header insertion is deferred until after lock acquisition.
        // On real hardware, the DMA only emits the packet header when it
        // starts the actual data transfer, not before the lock is acquired.
        // Emitting early would lock the stream switch packet arbiter while
        // the DMA waits for buffer locks, potentially deadlocking other
        // packets sharing the same arbiter.

        // Determine initial FSM state based on whether lock acquisition is needed
        let ch = &mut self.channels[ch_idx];
        ch.current_bd = Some(bd_index);
        ch.repeat_count = repeat_count as u32;

        if transfer.acquire_lock.is_some() {
            let lock_id = transfer.acquire_lock.unwrap();
            ch.fsm = ChannelFsm::AcquiringLock {
                lock_id,
                cycles_remaining: self.timing_config.lock_acquire_cycles as u16,
                acquired: false,
                transfer: Box::new(transfer),
            };
        } else {
            ch.fsm = ChannelFsm::BdSetup {
                cycles_remaining: self.timing_config.bd_setup_cycles as u16,
                transfer: Box::new(transfer),
            };
        }

        if repeat_count > 0 {
            log::info!("DMA tile({},{}) ch{} started BD {} with repeat_count={}",
                self.col, self.row, channel, bd_index, repeat_count);
        }

        self.trace(EventType::DmaStartTask { channel });

        Ok(())
    }

    /// Stop a channel.
    pub fn stop_channel(&mut self, channel: ChannelId) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        let ch = &mut self.channels[ch_idx];
        ch.fsm = ChannelFsm::Idle;
        ch.queued_bd = None;
        ch.chain_start_bd = None;

        Ok(())
    }

    /// Pause a channel.
    pub fn pause_channel(&mut self, channel: ChannelId) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        let ch = &mut self.channels[ch_idx];
        if ch.fsm.is_active() {
            let saved = std::mem::take(&mut ch.fsm);
            ch.fsm = ChannelFsm::Paused { saved: Box::new(saved) };
        }

        Ok(())
    }

    /// Resume a paused channel.
    pub fn resume_channel(&mut self, channel: ChannelId) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        let ch = &mut self.channels[ch_idx];
        if let ChannelFsm::Paused { saved } = std::mem::take(&mut ch.fsm) {
            ch.fsm = *saved;
        }

        Ok(())
    }

    /// Enable a channel by type and relative channel number.
    ///
    /// This is used by the NPU executor when a DMA control register is written.
    /// The channel is marked as active, ready to be started with a BD.
    ///
    /// # Arguments
    /// * `is_mm2s` - true for MM2S (memory-to-stream), false for S2MM
    /// * `relative_channel` - channel number within the type (0 or 1 for compute tiles)
    pub fn enable_channel(&mut self, is_mm2s: bool, relative_channel: u8) {
        // Convert relative channel to absolute channel index
        // For compute tiles: S2MM = 0-1, MM2S = 2-3
        // For mem tiles: S2MM = 0-5, MM2S = 6-11
        let ch_idx = if is_mm2s {
            self.s2mm_count + relative_channel as usize
        } else {
            relative_channel as usize
        };

        if ch_idx >= self.num_channels() {
            let msg = format!(
                "DmaEngine({},{})::enable_channel: invalid channel {} (is_mm2s={}, rel={})",
                self.col, self.row, ch_idx, is_mm2s, relative_channel,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return;
        }

        // Mark channel as ready/active
        // The actual transfer will start when start_channel is called with a BD
        log::debug!(
            "DmaEngine ({}, {}): enabled {} channel {}",
            self.col, self.row,
            if is_mm2s { "MM2S" } else { "S2MM" },
            relative_channel
        );

        // If the channel was idle, we might need to start it with a queued BD
        // For now, just mark it as being enabled (the queue mechanism handles this)
    }

    /// Check if a channel is active.
    pub fn channel_active(&self, channel: ChannelId) -> bool {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return false;
        }
        self.channels[ch_idx].is_active()
    }

    /// Get a human-readable description of a channel's FSM state.
    pub fn channel_state_name(&self, channel: ChannelId) -> String {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return "invalid".to_string();
        }
        self.channels[ch_idx].fsm_description()
    }

    /// Check if a channel has pending work (active, waiting, or has queued work).
    pub fn channel_has_pending_work(&self, channel: ChannelId) -> bool {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return false;
        }
        self.channels[ch_idx].has_pending_work()
    }

    /// Check if any channel is active.
    pub fn any_channel_active(&self) -> bool {
        self.channels.iter().any(|ch| ch.is_active())
    }

    /// Get channel state.
    pub fn channel_state(&self, channel: ChannelId) -> ChannelState {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return ChannelState::Idle;
        }
        self.channels[ch_idx].state()
    }

    /// Get detailed FSM description for diagnostics.
    pub fn channel_fsm_description(&self, channel: ChannelId) -> String {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return "Invalid".to_string();
        }
        self.channels[ch_idx].fsm_description()
    }

    /// Get channel statistics.
    pub fn channel_stats(&self, channel: ChannelId) -> Option<&ChannelStats> {
        self.channels.get(channel as usize).map(|ch| &ch.stats)
    }

    /// Get active transfer for a channel.
    pub fn get_transfer(&self, channel: ChannelId) -> Option<&Transfer> {
        self.channels.get(channel as usize).and_then(|ch| ch.transfer())
    }

    // NOTE: try_acquire_lock() has been replaced by try_acquire_lock_fsm().

    /// Execute a simple 1D transfer immediately (no cycling).
    ///
    /// This is a convenience method for testing that runs the transfer
    /// to completion in one call.
    pub fn execute_1d_transfer(
        &mut self,
        channel: ChannelId,
        bd_index: u8,
        tile: &mut Tile,
        neighbors: &mut NeighborLocks<'_>,
        host_memory: &mut HostMemory,
    ) -> Result<u64, DmaError> {
        self.start_channel(channel, bd_index)?;

        let mut cycles = 0u64;
        while self.channel_active(channel) {
            self.step(tile, neighbors, host_memory);
            cycles += 1;

            // Safety limit
            if cycles > 1_000_000 {
                return Err(DmaError::ChannelBusy(channel));
            }
        }

        Ok(cycles)
    }

    /// Reset all channels and clear state.
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
        self.stream_out.clear();
        self.stream_in.clear();
        self.task_tokens.reset();
        // Don't clear BD configs - those are persistent configuration
    }
}
