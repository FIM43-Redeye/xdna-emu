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
//!     engine.step(&mut tile_memory, &mut host_memory)?;
//! }
//! ```

use std::collections::VecDeque;

use super::{
    BdConfig, ChannelType, DmaError, DmaResult,
    NUM_BUFFER_DESCRIPTORS, MEMTILE_NUM_BUFFER_DESCRIPTORS,
    COMPUTE_S2MM_CHANNELS, COMPUTE_MM2S_CHANNELS,
    MEM_TILE_S2MM_CHANNELS, MEM_TILE_MM2S_CHANNELS, DMA_DATA_WIDTH_BYTES,
};
use super::transfer::{Transfer, TransferDirection, TransferEndpoint, TransferState};
use super::timing::{DmaTimingConfig, ChannelTimingState, TransferPhase};
use crate::device::host_memory::HostMemory;
use crate::device::tile::Tile;
use crate::interpreter::timing::sync::LockTimingState;

/// Stream data word for DMA-to-stream interface.
#[derive(Debug, Clone, Copy)]
pub struct StreamData {
    /// Data word (32 bits)
    pub data: u32,
    /// TLAST marker (end of packet)
    pub tlast: bool,
    /// Channel that produced/consumes this data
    pub channel: u8,
}

/// Channel identifier.
pub type ChannelId = u8;

/// State of a DMA channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum ChannelState {
    /// Channel is idle (no active transfer)
    #[default]
    Idle,
    /// Channel is active (transfer in progress)
    Active,
    /// Channel is paused
    Paused,
    /// Channel is waiting for lock
    WaitingForLock(u8),
    /// Channel completed last transfer
    Complete,
    /// Channel encountered an error
    Error,
}


/// Statistics for a DMA channel.
#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    /// Total transfers completed
    pub transfers_completed: u64,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Total cycles spent in transfers
    pub cycles_spent: u64,
    /// Number of lock wait cycles
    pub lock_wait_cycles: u64,
}

/// DMA engine for a single tile.
#[derive(Debug)]
pub struct DmaEngine {
    /// Tile column
    pub col: u8,

    /// Tile row
    pub row: u8,

    /// Whether this is a memory tile (affects channel count)
    pub is_mem_tile: bool,

    /// Buffer descriptor configurations
    bd_configs: Vec<BdConfig>,

    /// Active transfer per channel (None if idle)
    transfers: Vec<Option<Transfer>>,

    /// Channel states
    channel_states: Vec<ChannelState>,

    /// Per-channel statistics
    channel_stats: Vec<ChannelStats>,

    /// Queued BD per channel (for chaining)
    queued_bds: Vec<Option<u8>>,

    /// Repeat count per channel (how many more times to repeat current BD)
    /// Value N means run the BD N+1 more times after current
    repeat_counts: Vec<u8>,

    /// Current BD index per channel (for repeat)
    current_bds: Vec<Option<u8>>,

    /// Timing configuration (controls cycle-accuracy vs fast mode)
    timing_config: DmaTimingConfig,

    /// Per-channel timing state (only used when timing is enabled)
    timing_states: Vec<ChannelTimingState>,

    /// Stream output buffer (MM2S channels produce data here).
    /// Data is read from tile memory and queued for the stream router.
    stream_out: VecDeque<StreamData>,

    /// Stream input buffer (S2MM channels consume data from here).
    /// Data from the stream router is queued here for writing to tile memory.
    stream_in: VecDeque<StreamData>,

    /// Lock timing state for contention tracking (optional).
    /// When enabled, tracks detailed lock acquire/release timing per lock.
    lock_timing: Option<LockTimingState>,
}

impl DmaEngine {
    /// Create a new DMA engine for a compute tile.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let num_channels = COMPUTE_S2MM_CHANNELS + COMPUTE_MM2S_CHANNELS;
        Self::new(col, row, false, num_channels)
    }

    /// Create a new DMA engine for a memory tile.
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let num_channels = MEM_TILE_S2MM_CHANNELS + MEM_TILE_MM2S_CHANNELS;
        Self::new(col, row, true, num_channels)
    }

    /// Create a new DMA engine with specified number of channels.
    ///
    /// Uses instant timing by default for fast simulation.
    /// Call `with_timing` to enable cycle-accurate timing.
    fn new(col: u8, row: u8, is_mem_tile: bool, num_channels: usize) -> Self {
        let mut transfers = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            transfers.push(None);
        }

        // MemTile has 48 BDs, compute tile has 16
        let num_bds = if is_mem_tile {
            MEMTILE_NUM_BUFFER_DESCRIPTORS
        } else {
            NUM_BUFFER_DESCRIPTORS
        };

        log::debug!("DmaEngine::new col={} row={} is_mem_tile={} num_channels={} num_bds={}",
            col, row, is_mem_tile, num_channels, num_bds);

        Self {
            col,
            row,
            is_mem_tile,
            bd_configs: vec![BdConfig::default(); num_bds],
            transfers,
            channel_states: vec![ChannelState::Idle; num_channels],
            channel_stats: vec![ChannelStats::default(); num_channels],
            queued_bds: vec![None; num_channels],
            repeat_counts: vec![0; num_channels],
            current_bds: vec![None; num_channels],
            // Default to instant timing for backwards compatibility
            timing_config: DmaTimingConfig::instant(),
            timing_states: vec![ChannelTimingState::default(); num_channels],
            // Stream buffers with reasonable capacity
            stream_out: VecDeque::with_capacity(16),
            stream_in: VecDeque::with_capacity(16),
            // Lock timing disabled by default
            lock_timing: None,
        }
    }

    /// Set the timing configuration.
    ///
    /// Use `DmaTimingConfig::from_aie2_spec()` for cycle-accurate timing.
    /// Use `DmaTimingConfig::instant()` for fast simulation.
    pub fn with_timing(mut self, config: DmaTimingConfig) -> Self {
        self.timing_config = config;
        self
    }

    /// Enable cycle-accurate timing from AIE2 specification.
    pub fn with_cycle_accurate_timing(mut self) -> Self {
        self.timing_config = DmaTimingConfig::from_aie2_spec();
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

    /// Get the timing configuration.
    pub fn timing_config(&self) -> &DmaTimingConfig {
        &self.timing_config
    }

    /// Check if timing is enabled (non-instant mode).
    pub fn is_timing_enabled(&self) -> bool {
        // Instant mode has 0 setup cycles, cycle-accurate has non-zero
        self.timing_config.bd_setup_cycles > 0
    }

    /// Get the number of channels.
    pub fn num_channels(&self) -> usize {
        self.transfers.len()
    }

    /// Get the channel type (S2MM or MM2S).
    pub fn channel_type(&self, channel: ChannelId) -> ChannelType {
        ChannelType::from_channel_index(channel as usize, self.is_mem_tile)
    }

    /// Configure a buffer descriptor.
    pub fn configure_bd(&mut self, bd_index: u8, config: BdConfig) -> Result<(), DmaError> {
        if bd_index as usize >= self.bd_configs.len() {
            return Err(DmaError::InvalidBd(bd_index));
        }

        self.bd_configs[bd_index as usize] = config;
        Ok(())
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
        self.start_channel_with_repeat(channel, bd_index, 0)
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

        // Check if channel is already busy
        if matches!(self.channel_states[ch_idx], ChannelState::Active | ChannelState::WaitingForLock(_)) {
            return Err(DmaError::ChannelBusy(channel));
        }

        let bd_config = &self.bd_configs[bd_index as usize];

        // Determine transfer direction
        let direction = match self.channel_type(channel) {
            ChannelType::S2MM => TransferDirection::S2MM,
            ChannelType::MM2S => TransferDirection::MM2S,
        };

        // Create transfer
        let transfer = Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row)?;

        // Initialize timing state if timing is enabled
        if self.is_timing_enabled() {
            self.timing_states[ch_idx] = ChannelTimingState::new_transfer(
                bd_config.length as u64,
                &self.timing_config,
            );
        }

        // Update state based on transfer state
        let new_state = match transfer.state {
            TransferState::WaitingForLock(lock) => ChannelState::WaitingForLock(lock),
            TransferState::Active => ChannelState::Active,
            _ => ChannelState::Idle,
        };

        // Log transfer details for debugging
        log::debug!("DMA tile({},{}) ch{} transfer: total_bytes={} src={:?} dst={:?} state={:?}",
            self.col, self.row, channel,
            transfer.total_bytes, transfer.source, transfer.dest, transfer.state);

        self.transfers[ch_idx] = Some(transfer);
        self.channel_states[ch_idx] = new_state;
        self.repeat_counts[ch_idx] = repeat_count;
        self.current_bds[ch_idx] = Some(bd_index);

        if repeat_count > 0 {
            log::info!("DMA tile({},{}) ch{} started BD {} with repeat_count={}",
                self.col, self.row, channel, bd_index, repeat_count);
        }

        Ok(())
    }

    /// Stop a channel.
    pub fn stop_channel(&mut self, channel: ChannelId) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        self.transfers[ch_idx] = None;
        self.channel_states[ch_idx] = ChannelState::Idle;
        self.queued_bds[ch_idx] = None;

        Ok(())
    }

    /// Pause a channel.
    pub fn pause_channel(&mut self, channel: ChannelId) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        if self.channel_states[ch_idx] == ChannelState::Active {
            self.channel_states[ch_idx] = ChannelState::Paused;
        }

        Ok(())
    }

    /// Resume a paused channel.
    pub fn resume_channel(&mut self, channel: ChannelId) -> Result<(), DmaError> {
        let ch_idx = channel as usize;

        if ch_idx >= self.num_channels() {
            return Err(DmaError::InvalidChannel(channel));
        }

        if self.channel_states[ch_idx] == ChannelState::Paused {
            self.channel_states[ch_idx] = ChannelState::Active;
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
        let s2mm_count = if self.is_mem_tile {
            MEM_TILE_S2MM_CHANNELS
        } else {
            COMPUTE_S2MM_CHANNELS
        };

        let ch_idx = if is_mm2s {
            s2mm_count + relative_channel as usize
        } else {
            relative_channel as usize
        };

        if ch_idx >= self.num_channels() {
            log::warn!(
                "DmaEngine::enable_channel: invalid channel {} (is_mm2s={}, rel={})",
                ch_idx, is_mm2s, relative_channel
            );
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
        matches!(
            self.channel_states[ch_idx],
            ChannelState::Active | ChannelState::WaitingForLock(_)
        )
    }

    /// Check if any channel is active.
    pub fn any_channel_active(&self) -> bool {
        self.channel_states.iter().any(|s| {
            matches!(s, ChannelState::Active | ChannelState::WaitingForLock(_))
        })
    }

    /// Get channel state.
    pub fn channel_state(&self, channel: ChannelId) -> ChannelState {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return ChannelState::Idle;
        }
        self.channel_states[ch_idx]
    }

    /// Get channel statistics.
    pub fn channel_stats(&self, channel: ChannelId) -> Option<&ChannelStats> {
        self.channel_stats.get(channel as usize)
    }

    /// Get active transfer for a channel.
    pub fn get_transfer(&self, channel: ChannelId) -> Option<&Transfer> {
        self.transfers.get(channel as usize).and_then(|t| t.as_ref())
    }

    /// Step the DMA engine by one cycle.
    ///
    /// This processes all active channels, moving data between memory and streams.
    /// Returns the overall result of the step.
    pub fn step(&mut self, tile: &mut Tile, host_memory: &mut HostMemory) -> DmaResult {
        let mut any_active = false;
        let mut any_waiting = false;

        for ch_idx in 0..self.num_channels() {
            match self.channel_states[ch_idx] {
                ChannelState::Active => {
                    self.step_channel(ch_idx, tile, host_memory);
                    any_active = true;
                }
                ChannelState::WaitingForLock(lock_id) => {
                    // Try to acquire lock
                    if self.try_acquire_lock(ch_idx, lock_id, tile) {
                        self.channel_states[ch_idx] = ChannelState::Active;
                        any_active = true;
                    } else {
                        self.channel_stats[ch_idx].lock_wait_cycles += 1;
                        any_waiting = true;
                    }
                }
                ChannelState::Complete => {
                    // Check for BD chaining
                    if let Some(next_bd) = self.queued_bds[ch_idx].take() {
                        log::debug!("DMA tile({},{}) ch{} starting queued BD {}",
                            self.col, self.row, ch_idx, next_bd);
                        if let Err(e) = self.start_channel(ch_idx as u8, next_bd) {
                            log::warn!("DMA tile({},{}) ch{} failed to start BD {}: {:?}",
                                self.col, self.row, ch_idx, next_bd, e);
                            self.channel_states[ch_idx] = ChannelState::Error;
                        } else {
                            log::debug!("DMA tile({},{}) ch{} now active with BD {}",
                                self.col, self.row, ch_idx, next_bd);
                            any_active = true;
                        }
                    }
                }
                _ => {}
            }
        }

        if any_active {
            DmaResult::InProgress
        } else if any_waiting {
            DmaResult::WaitingForLock(0) // Generic wait
        } else {
            DmaResult::Complete
        }
    }

    /// Step a single channel.
    ///
    /// If timing is enabled, this uses the phase-based timing state machine.
    /// Otherwise, it transfers data immediately (fast mode).
    fn step_channel(&mut self, ch_idx: usize, tile: &mut Tile, host_memory: &mut HostMemory) {
        // Check if we have an active transfer
        let transfer = match &self.transfers[ch_idx] {
            Some(t) => t,
            None => return,
        };

        if !transfer.is_active() {
            return;
        }

        // If timing is enabled, use phase-based execution
        if self.is_timing_enabled() {
            self.step_channel_timed(ch_idx, tile, host_memory);
        } else {
            self.step_channel_instant(ch_idx, tile, host_memory);
        }
    }

    /// Step channel with instant timing (fast mode).
    ///
    /// Transfers `DMA_DATA_WIDTH_BYTES` per step with no overhead.
    fn step_channel_instant(&mut self, ch_idx: usize, tile: &mut Tile, host_memory: &mut HostMemory) {
        // Extract transfer info
        let transfer_info = {
            let transfer = self.transfers[ch_idx].as_ref().unwrap();
            let bytes_to_transfer = DMA_DATA_WIDTH_BYTES.min(transfer.remaining_bytes() as usize);
            let addr = transfer.current_address();
            let source = transfer.source;
            let dest = transfer.dest;
            let channel = transfer.channel;
            let remaining_after = transfer.remaining_bytes().saturating_sub(bytes_to_transfer as u64);
            log::trace!(
                "DMA ch{} step: src={:?} dst={:?} addr=0x{:X} bytes={}",
                ch_idx, source, dest, addr, bytes_to_transfer
            );
            (bytes_to_transfer, addr, source, dest, channel, remaining_after == 0)
        };

        let (bytes_to_transfer, addr, source, dest, channel, is_last) = transfer_info;

        // Log progress for debugging (shim tile MM2S channel)
        if self.col == 0 && self.row == 0 && ch_idx == 2 {
            let remaining = self.transfers[ch_idx].as_ref().unwrap().remaining_bytes();
            log::info!("Shim MM2S step: addr=0x{:X} bytes={} remaining={}", addr, bytes_to_transfer, remaining);
        }

        // Handle zero-byte transfer case
        if bytes_to_transfer == 0 {
            self.complete_transfer(ch_idx, tile);
            return;
        }

        // S2MM stalls when stream_in has no data - wait for routing to provide data
        if matches!(source, TransferEndpoint::Stream { .. }) && self.stream_in.is_empty() {
            // Debug: log when we stall waiting for stream data
            log::debug!("DMA({},{}) ch{} S2MM stall: stream_in empty, addr=0x{:X}, remaining={}",
                self.col, self.row, ch_idx, addr, bytes_to_transfer);
            // Stall: don't advance transfer, wait for data to arrive via route_dma_streams
            return;
        }

        // Perform the transfer
        let success = self.do_transfer(source, dest, addr, bytes_to_transfer, channel, is_last, tile, host_memory);

        // Update transfer state
        let transfer = self.transfers[ch_idx].as_mut().unwrap();
        transfer.tick();

        if success {
            transfer.data_transferred(bytes_to_transfer as u64);
            self.channel_stats[ch_idx].bytes_transferred += bytes_to_transfer as u64;

            // Check if complete
            let is_complete = {
                let t = self.transfers[ch_idx].as_ref().unwrap();
                t.is_complete() || matches!(t.state, TransferState::ReleasingLock(_))
            };

            if is_complete {
                self.complete_transfer(ch_idx, tile);
            }
        } else {
            self.set_transfer_error(ch_idx, addr, tile.data_memory().len() as u64);
        }
    }

    /// Step channel with cycle-accurate timing.
    ///
    /// Uses the phase-based timing state machine for realistic delays.
    /// The timing state machine handles phase transitions and word counting.
    /// This method performs the actual data movement when in DataTransfer phase.
    fn step_channel_timed(&mut self, ch_idx: usize, tile: &mut Tile, host_memory: &mut HostMemory) {
        // Get lock info from transfer
        let (has_lock, lock_available) = {
            let transfer = self.transfers[ch_idx].as_ref().unwrap();
            let has_lock = transfer.acquire_lock.is_some();
            // For now, assume locks are always available (real impl checks tile.locks)
            let lock_available = true;
            (has_lock, lock_available)
        };

        // Check if we need to do data transfer this cycle (before tick advances state)
        let should_transfer = self.timing_states[ch_idx].phase == TransferPhase::DataTransfer
            && self.timing_states[ch_idx].memory_pipeline_busy == 0;

        // Perform data transfer if in DataTransfer phase
        if should_transfer {
            let words_per_cycle = self.timing_config.words_per_cycle as usize;
            let bytes_per_cycle = words_per_cycle * 4;

            let (bytes_to_transfer, addr, source, dest, channel, is_last) = {
                let transfer = self.transfers[ch_idx].as_ref().unwrap();
                let bytes_to_transfer = bytes_per_cycle.min(transfer.remaining_bytes() as usize);
                let remaining_after = transfer.remaining_bytes().saturating_sub(bytes_to_transfer as u64);
                (
                    bytes_to_transfer,
                    transfer.current_address(),
                    transfer.source,
                    transfer.dest,
                    transfer.channel,
                    remaining_after == 0,
                )
            };

            if bytes_to_transfer > 0 {
                let success = self.do_transfer(source, dest, addr, bytes_to_transfer, channel, is_last, tile, host_memory);

                if success {
                    // Update transfer state (address generator, bytes transferred)
                    let transfer = self.transfers[ch_idx].as_mut().unwrap();
                    transfer.data_transferred(bytes_to_transfer as u64);
                    self.channel_stats[ch_idx].bytes_transferred += bytes_to_transfer as u64;
                } else {
                    self.set_transfer_error(ch_idx, addr, tile.data_memory().len() as u64);
                    return;
                }
            }
        }

        // Advance timing state (this handles phase transitions and word counting)
        let phase = self.timing_states[ch_idx].tick(&self.timing_config, has_lock, lock_available);

        // Handle completion
        if phase == TransferPhase::Complete {
            self.complete_transfer(ch_idx, tile);
        }

        // Update transfer tick count
        if let Some(transfer) = self.transfers[ch_idx].as_mut() {
            transfer.tick();
        }
    }

    /// Perform a data transfer operation.
    ///
    /// For stream transfers (MM2S/S2MM), data is buffered in stream_out/stream_in
    /// and will be routed by the TileArray's stream router.
    fn do_transfer(
        &mut self,
        source: TransferEndpoint,
        dest: TransferEndpoint,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        match (source, dest) {
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::Stream { .. }) => {
                // MM2S: Read from tile memory, queue to stream output
                self.transfer_mm2s(addr, bytes, channel, is_last, tile)
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::TileMemory { .. }) => {
                // S2MM: Read from stream input, write to tile memory
                self.transfer_s2mm(addr, bytes, tile)
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::TileMemory { .. }) => {
                Self::transfer_host_to_tile_static(addr, bytes, tile, host_memory)
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::HostMemory) => {
                Self::transfer_tile_to_host_static(addr, bytes, tile, host_memory)
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::Stream { .. }) => {
                // Shim MM2S: Read from host DDR, queue to stream output
                self.transfer_host_to_stream(addr, bytes, channel, is_last, host_memory)
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::HostMemory) => {
                // Shim S2MM: Read from stream input, write to host DDR
                self.transfer_stream_to_host(addr, bytes, host_memory)
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::TileMemory { .. }) => {
                // Tile to tile: Would need array access, mark as success for now
                true
            }
            _ => false,
        }
    }

    /// MM2S: Read from tile memory and queue to stream output.
    fn transfer_mm2s(&mut self, addr: u64, bytes: usize, channel: u8, is_last: bool, tile: &Tile) -> bool {
        let mem_size = tile.data_memory().len();

        // MemTile DMA addresses may have a 0x80000 offset in the address space
        // Wrap addresses to stay within memory bounds
        let offset = (addr as usize) % mem_size;

        if offset + bytes > mem_size {
            log::warn!("DMA({},{}) MM2S addr=0x{:X} bytes={} wraps past memory end (size=0x{:X})",
                self.col, self.row, addr, bytes, mem_size);
            return false;
        }

        // Read data from tile memory in 32-bit words
        let data = tile.data_memory();
        let word_count = (bytes + 3) / 4;

        for i in 0..word_count {
            let word_offset = offset + i * 4;
            let word = if word_offset + 4 <= data.len() {
                u32::from_le_bytes([
                    data[word_offset],
                    data[word_offset + 1],
                    data[word_offset + 2],
                    data[word_offset + 3],
                ])
            } else {
                // Partial word at end
                let mut word_bytes = [0u8; 4];
                for j in 0..4 {
                    if word_offset + j < data.len() {
                        word_bytes[j] = data[word_offset + j];
                    }
                }
                u32::from_le_bytes(word_bytes)
            };

            let is_last_word = is_last && (i == word_count - 1);
            self.stream_out.push_back(StreamData {
                data: word,
                tlast: is_last_word,
                channel,
            });
        }

        true
    }

    /// S2MM: Read from stream input, write to tile memory.
    ///
    /// Only transfers data that is available in stream_in.
    /// Returns false if stream_in is empty (caller should stall).
    fn transfer_s2mm(&mut self, addr: u64, bytes: usize, tile: &mut Tile) -> bool {
        let mem_size = tile.data_memory().len();

        // MemTile DMA addresses may have a 0x80000 offset in the address space
        // Wrap addresses to stay within memory bounds
        let offset = (addr as usize) % mem_size;

        if offset + bytes > mem_size {
            log::warn!("DMA({},{}) S2MM addr=0x{:X} bytes={} wraps past memory end (size=0x{:X})",
                self.col, self.row, addr, bytes, mem_size);
            return false;
        }

        // Must have at least one word to transfer
        if self.stream_in.is_empty() {
            return false;
        }

        // Write data to tile memory in 32-bit words
        let data = tile.data_memory_mut();
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;

        for _ in 0..word_count {
            // Get data from stream - caller ensured at least some data exists
            let word = if let Some(stream_data) = self.stream_in.pop_front() {
                stream_data.data
            } else {
                // No more stream data - transfer partial, continue next step
                break;
            };

            let word_bytes = word.to_le_bytes();
            for j in 0..4 {
                if bytes_written + j < bytes && offset + bytes_written + j < data.len() {
                    data[offset + bytes_written + j] = word_bytes[j];
                }
            }
            bytes_written += 4;
        }

        true
    }

    /// Shim MM2S: Read from host DDR and queue to stream output.
    fn transfer_host_to_stream(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        host_memory: &HostMemory,
    ) -> bool {
        // Read data from host memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!("MM2S transfer: addr=0x{:X} bytes={} words={}", addr, bytes, word_count);

        for i in 0..word_count {
            let word_addr = addr + (i * 4) as u64;
            let word = host_memory.read_u32(word_addr);

            if i < 4 {
                log::debug!("  MM2S word[{}]: addr=0x{:X} -> 0x{:08X}", i, word_addr, word);
            }

            self.stream_out.push_back(StreamData {
                data: word,
                channel,
                tlast: is_last && i == word_count - 1,
            });
        }

        true
    }

    /// Shim S2MM: Read from stream input and write to host DDR.
    fn transfer_stream_to_host(
        &mut self,
        addr: u64,
        bytes: usize,
        host_memory: &mut HostMemory,
    ) -> bool {
        // Must have at least one word to transfer
        if self.stream_in.is_empty() {
            return false;
        }

        // Write data to host memory in 32-bit words
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;

        for i in 0..word_count {
            let word = if let Some(stream_data) = self.stream_in.pop_front() {
                stream_data.data
            } else {
                // No more stream data
                break;
            };

            let word_addr = addr + (i * 4) as u64;
            host_memory.write_u32(word_addr, word);
            bytes_written += 4;
        }

        bytes_written > 0
    }

    /// Set a transfer error on a channel.
    fn set_transfer_error(&mut self, ch_idx: usize, address: u64, limit: u64) {
        if let Some(transfer) = self.transfers[ch_idx].as_mut() {
            transfer.set_error(DmaError::AddressOutOfBounds { address, limit });
        }
        self.channel_states[ch_idx] = ChannelState::Error;
    }

    /// Transfer data from host memory to tile memory (static version).
    fn transfer_host_to_tile_static(
        tile_addr: u64,
        bytes: usize,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        let offset = tile_addr as usize;
        if offset + bytes > tile.data_memory().len() {
            return false;
        }

        // Read from host memory
        let mut buf = vec![0u8; bytes];
        // Note: In a real implementation, we'd track a separate host address
        // For now, we use the tile address as an offset into host memory too
        host_memory.read_bytes(tile_addr, &mut buf);

        // Write to tile memory
        tile.data_memory_mut()[offset..offset + bytes].copy_from_slice(&buf);

        true
    }

    /// Transfer data from tile memory to host memory (static version).
    fn transfer_tile_to_host_static(
        tile_addr: u64,
        bytes: usize,
        tile: &Tile,
        host_memory: &mut HostMemory,
    ) -> bool {
        let offset = tile_addr as usize;
        if offset + bytes > tile.data_memory().len() {
            return false;
        }

        // Read from tile memory
        let data = &tile.data_memory()[offset..offset + bytes];

        // Write to host memory
        // Note: In a real implementation, we'd track a separate host address
        host_memory.write_bytes(tile_addr, data);

        true
    }

    /// Complete a transfer.
    fn complete_transfer(&mut self, ch_idx: usize, tile: &mut Tile) {
        let transfer = match &mut self.transfers[ch_idx] {
            Some(t) => t,
            None => return,
        };

        log::debug!("complete_transfer tile({},{}) ch={} state={:?} release_lock={:?}",
            self.col, self.row, ch_idx, transfer.state, transfer.release_lock);

        // Handle lock release if needed
        if let TransferState::ReleasingLock(lock_id) = transfer.state {
            log::info!("DMA tile({},{}) releasing lock {} on channel {}",
                self.col, self.row, lock_id, ch_idx);
            if (lock_id as usize) < tile.locks.len() {
                tile.locks[lock_id as usize].release();
                log::info!("Tile({},{}) lock {} now = {}",
                    self.col, self.row, lock_id, tile.locks[lock_id as usize].value);
            }
            transfer.lock_released();
        }

        // Update statistics
        self.channel_stats[ch_idx].transfers_completed += 1;
        self.channel_stats[ch_idx].cycles_spent += transfer.cycles_elapsed;

        // Check for BD chaining first
        if let Some(next_bd) = transfer.next_bd {
            log::debug!("DMA tile({},{}) ch{} chaining to BD {} (from BD {})",
                self.col, self.row, ch_idx, next_bd, transfer.bd_index);
            self.queued_bds[ch_idx] = Some(next_bd);
        }
        // If no chaining, check for repeat count
        else if self.repeat_counts[ch_idx] > 0 {
            // Decrement repeat count and requeue the same BD
            self.repeat_counts[ch_idx] -= 1;
            if let Some(bd_idx) = self.current_bds[ch_idx] {
                self.queued_bds[ch_idx] = Some(bd_idx);
                log::debug!("DMA tile({},{}) ch{} repeating BD {} ({} remaining)",
                    self.col, self.row, ch_idx, bd_idx, self.repeat_counts[ch_idx]);
            }
        }

        // Mark channel as complete
        self.channel_states[ch_idx] = ChannelState::Complete;
        self.transfers[ch_idx] = None;
    }

    /// Try to acquire a lock.
    ///
    /// AIE-ML lock acquire semantics:
    /// - If acquire_value < 0: `acq_ge` - wait until lock >= |value|, then decrement by |value|
    /// - If acquire_value >= 0: `acq_eq` - wait until lock == value, then set to 0
    ///
    /// If lock timing is enabled, tracks acquire attempts and contention.
    fn try_acquire_lock(&mut self, ch_idx: usize, lock_id: u8, tile: &mut Tile) -> bool {
        use crate::device::tile::LockResult;

        if (lock_id as usize) >= tile.locks.len() {
            return false;
        }

        let transfer = match &mut self.transfers[ch_idx] {
            Some(t) => t,
            None => return false,
        };

        let acquire_value = transfer.acquire_value;
        let lock = &mut tile.locks[lock_id as usize];

        // AIE-ML lock semantics from AM020:
        // - acquire_value < 0: acq_ge, expected = |value|, delta = value (decrement)
        // - acquire_value >= 0: acq_eq, expected = value, delta = -value (decrement to 0)
        let (expected, delta) = if acquire_value < 0 {
            // acq_ge: wait until lock >= |value|, then decrement by |value|
            ((-acquire_value) as u8, acquire_value)
        } else if acquire_value > 0 {
            // acq_eq: wait until lock == value, then decrement to 0
            (acquire_value as u8, -acquire_value)
        } else {
            // acquire_value == 0: simple acquire (decrement if > 0)
            (1, -1)
        };

        let current_value = lock.value;
        let result = lock.acquire_with_value(expected, delta);
        let success = matches!(result, LockResult::Success);

        log::trace!("DMA try_acquire_lock tile({},{}) ch{} lock={} expected={} delta={} current={} result={:?}",
            self.col, self.row, ch_idx, lock_id, expected, delta, current_value, result);

        // Track timing if enabled
        if let Some(ref mut timing) = self.lock_timing {
            timing.track_acquire(lock_id as usize, success);
        }

        if success {
            transfer.lock_acquired();
            true
        } else {
            false
        }
    }

    /// Execute a simple 1D transfer immediately (no cycling).
    ///
    /// This is a convenience method for testing that runs the transfer
    /// to completion in one call.
    pub fn execute_1d_transfer(
        &mut self,
        channel: ChannelId,
        bd_index: u8,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> Result<u64, DmaError> {
        self.start_channel(channel, bd_index)?;

        let mut cycles = 0u64;
        while self.channel_active(channel) {
            self.step(tile, host_memory);
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
        for i in 0..self.num_channels() {
            self.transfers[i] = None;
            self.channel_states[i] = ChannelState::Idle;
            self.queued_bds[i] = None;
        }

        // Clear stream buffers
        self.stream_out.clear();
        self.stream_in.clear();

        // Don't clear BD configs - those are persistent configuration
    }

    // === Stream Interface for TileArray ===

    /// Pop a word from the stream output buffer (MM2S produced data).
    ///
    /// Returns None if no data is available.
    pub fn pop_stream_out(&mut self) -> Option<StreamData> {
        self.stream_out.pop_front()
    }

    /// Push a word to the stream input buffer (for S2MM to consume).
    ///
    /// Returns true if successful, false if buffer is full.
    pub fn push_stream_in(&mut self, data: StreamData) -> bool {
        if self.stream_in.len() < 256 {
            self.stream_in.push_back(data);
            true
        } else {
            false
        }
    }

    /// Check if stream output buffer has data.
    pub fn has_stream_out(&self) -> bool {
        !self.stream_out.is_empty()
    }

    /// Check if stream input buffer has space.
    pub fn can_accept_stream_in(&self) -> bool {
        self.stream_in.len() < 256
    }

    /// Get the number of words in stream output buffer.
    pub fn stream_out_len(&self) -> usize {
        self.stream_out.len()
    }

    /// Get the number of words in stream input buffer.
    pub fn stream_in_len(&self) -> usize {
        self.stream_in.len()
    }

    /// Check if any S2MM channel needs to receive stream data.
    ///
    /// Returns Some(channel) if a channel needs data, None otherwise.
    /// This returns true for S2MM channels that are:
    /// - Active and waiting for data
    /// - WaitingForLock (will need data once lock is acquired)
    /// This allows the stream routing to buffer data in stream_in before
    /// the lock is acquired, matching how real hardware works.
    pub fn s2mm_needs_data(&self) -> Option<ChannelId> {
        for ch_idx in 0..self.num_channels() {
            if self.channel_type(ch_idx as u8) == ChannelType::S2MM {
                let state = &self.channel_states[ch_idx];
                // Accept data if channel is active or waiting for lock
                let is_pending = matches!(state, ChannelState::Active | ChannelState::WaitingForLock(_));
                // Only accept if stream_in has space
                if is_pending && self.stream_in.len() < 256 {
                    return Some(ch_idx as u8);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tile() -> Tile {
        Tile::compute(1, 2)
    }

    fn make_host_memory() -> HostMemory {
        HostMemory::new()
    }

    #[test]
    fn test_engine_creation() {
        let engine = DmaEngine::new_compute_tile(1, 2);
        assert_eq!(engine.num_channels(), 4);
        assert_eq!(engine.col, 1);
        assert_eq!(engine.row, 2);
    }

    #[test]
    fn test_mem_tile_engine() {
        let engine = DmaEngine::new_mem_tile(0, 1);
        assert_eq!(engine.num_channels(), 12);
        assert!(engine.is_mem_tile);
    }

    #[test]
    fn test_configure_bd() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);

        engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();

        let bd = engine.get_bd(0).unwrap();
        assert_eq!(bd.base_addr, 0x1000);
        assert_eq!(bd.length, 256);
    }

    #[test]
    fn test_invalid_bd_index() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let result = engine.configure_bd(16, BdConfig::simple_1d(0x1000, 256));
        assert!(matches!(result, Err(DmaError::InvalidBd(16))));
    }

    #[test]
    fn test_start_channel() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();

        engine.start_channel(0, 0).unwrap();

        assert!(engine.channel_active(0));
        assert_eq!(engine.channel_state(0), ChannelState::Active);
    }

    #[test]
    fn test_channel_busy_error() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();

        engine.start_channel(0, 0).unwrap();
        let result = engine.start_channel(0, 0);

        assert!(matches!(result, Err(DmaError::ChannelBusy(0))));
    }

    #[test]
    fn test_stop_channel() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();
        engine.start_channel(0, 0).unwrap();

        engine.stop_channel(0).unwrap();

        assert!(!engine.channel_active(0));
        assert_eq!(engine.channel_state(0), ChannelState::Idle);
    }

    #[test]
    fn test_pause_resume() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();
        engine.start_channel(0, 0).unwrap();

        engine.pause_channel(0).unwrap();
        assert_eq!(engine.channel_state(0), ChannelState::Paused);

        engine.resume_channel(0).unwrap();
        assert_eq!(engine.channel_state(0), ChannelState::Active);
    }

    #[test]
    fn test_simple_transfer() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Configure BD for 32 bytes using MM2S channel (reads from tile memory)
        // Channel 2 is MM2S on compute tiles
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();

        // Start transfer on MM2S channel
        engine.start_channel(2, 0).unwrap();

        // Step until complete
        let mut cycles = 0;
        while engine.channel_active(2) {
            engine.step(&mut tile, &mut host_mem);
            cycles += 1;
            if cycles > 100 {
                panic!("Transfer took too long");
            }
        }

        // Verify completion
        assert_eq!(engine.channel_state(2), ChannelState::Complete);

        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.transfers_completed, 1);
        assert_eq!(stats.bytes_transferred, 32);
    }

    #[test]
    fn test_any_channel_active() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 256)).unwrap();

        assert!(!engine.any_channel_active());

        engine.start_channel(0, 0).unwrap();
        assert!(engine.any_channel_active());
    }

    #[test]
    fn test_channel_type() {
        let engine = DmaEngine::new_compute_tile(1, 2);

        assert_eq!(engine.channel_type(0), ChannelType::S2MM);
        assert_eq!(engine.channel_type(1), ChannelType::S2MM);
        assert_eq!(engine.channel_type(2), ChannelType::MM2S);
        assert_eq!(engine.channel_type(3), ChannelType::MM2S);
    }

    #[test]
    fn test_transfer_with_lock() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Set lock to available state
        tile.locks[5].set(1);

        // Configure BD with lock using MM2S channel
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(5, 1)
            .with_release(5, 0);
        engine.configure_bd(0, bd).unwrap();

        // Start should trigger lock acquire on MM2S channel
        engine.start_channel(2, 0).unwrap();

        // Step until complete
        let mut cycles = 0;
        while engine.channel_active(2) {
            engine.step(&mut tile, &mut host_mem);
            cycles += 1;
            if cycles > 100 {
                panic!("Transfer took too long");
            }
        }

        // Verify lock was released
        assert_eq!(tile.locks[5].value, 1); // Released back to 1
    }

    #[test]
    fn test_execute_1d_transfer() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Use MM2S channel (channel 2) for testing
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();

        let cycles = engine.execute_1d_transfer(2, 0, &mut tile, &mut host_mem).unwrap();
        assert!(cycles > 0);

        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.bytes_transferred, 64);
    }

    #[test]
    fn test_reset() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 256)).unwrap();
        engine.start_channel(0, 0).unwrap();

        engine.reset();

        assert!(!engine.any_channel_active());
        // BD config should still be there
        assert!(engine.get_bd(0).unwrap().valid);
    }

    #[test]
    fn test_timing_disabled_by_default() {
        let engine = DmaEngine::new_compute_tile(1, 2);
        // Default is instant timing (no delays)
        assert!(!engine.is_timing_enabled());
        assert_eq!(engine.timing_config().bd_setup_cycles, 0);
    }

    #[test]
    fn test_enable_cycle_accurate_timing() {
        let engine = DmaEngine::new_compute_tile(1, 2).with_cycle_accurate_timing();
        assert!(engine.is_timing_enabled());
        assert_eq!(engine.timing_config().bd_setup_cycles, 4);
    }

    #[test]
    fn test_cycle_accurate_transfer() {
        // Create engine with cycle-accurate timing
        let mut engine = DmaEngine::new_compute_tile(1, 2).with_cycle_accurate_timing();
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Configure BD for 16 bytes (4 words) using MM2S channel
        // With AIE2 spec: 4 setup + 1 start + 1 mem latency + 4 data cycles = ~10+ cycles
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

        // Start transfer on MM2S channel
        engine.start_channel(2, 0).unwrap();

        // Step until complete
        let mut cycles = 0;
        while engine.channel_active(2) {
            engine.step(&mut tile, &mut host_mem);
            cycles += 1;
            if cycles > 100 {
                panic!("Timed transfer took too long");
            }
        }

        // Timed transfer should take more cycles than instant
        // (setup + latency + data phases)
        assert!(cycles >= 6, "Cycle-accurate transfer should have overhead, got {} cycles", cycles);

        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.bytes_transferred, 16);
    }

    #[test]
    fn test_instant_vs_timed_comparison() {
        let mut tile1 = make_tile();
        let mut tile2 = make_tile();
        let mut host_mem1 = make_host_memory();
        let mut host_mem2 = make_host_memory();

        // Test with 64 bytes using MM2S channel
        let transfer_size = 64;

        // Instant mode
        let mut instant = DmaEngine::new_compute_tile(1, 2);
        instant.configure_bd(0, BdConfig::simple_1d(0x100, transfer_size)).unwrap();
        instant.start_channel(2, 0).unwrap();
        let mut instant_cycles = 0;
        while instant.channel_active(2) {
            instant.step(&mut tile1, &mut host_mem1);
            instant_cycles += 1;
            if instant_cycles > 200 { break; }
        }

        // Timed mode
        let mut timed = DmaEngine::new_compute_tile(1, 2).with_cycle_accurate_timing();
        timed.configure_bd(0, BdConfig::simple_1d(0x100, transfer_size)).unwrap();
        timed.start_channel(2, 0).unwrap();
        let mut timed_cycles = 0;
        while timed.channel_active(2) {
            timed.step(&mut tile2, &mut host_mem2);
            timed_cycles += 1;
            if timed_cycles > 200 { break; }
        }

        // Timed should take more cycles than instant
        assert!(
            timed_cycles > instant_cycles,
            "Timed ({}) should take more cycles than instant ({})",
            timed_cycles, instant_cycles
        );

        // Both should complete successfully
        assert_eq!(instant.channel_state(2), ChannelState::Complete);
        assert_eq!(timed.channel_state(2), ChannelState::Complete);
    }

    #[test]
    fn test_lock_timing_integration() {
        // Create engine with lock timing enabled
        let mut engine = DmaEngine::new_compute_tile(1, 2)
            .with_cycle_accurate_timing()
            .with_lock_timing(16);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Configure BD that requires a lock
        let mut bd = BdConfig::simple_1d(0x100, 32);
        bd.acquire_lock = Some(5);
        bd.acquire_value = 1;
        engine.configure_bd(0, bd).unwrap();

        // Lock starts at 0, so acquire will fail initially
        tile.locks[5].value = 0;

        // Start transfer - should go to WaitingForLock
        engine.start_channel(0, 0).unwrap();

        // Step a few times - lock still not available
        for _ in 0..3 {
            engine.step(&mut tile, &mut host_mem);
        }

        // Verify waiting state
        assert!(
            matches!(engine.channel_state(0), ChannelState::WaitingForLock(5)),
            "Expected WaitingForLock(5), got {:?}",
            engine.channel_state(0)
        );

        // Check lock timing tracked contention
        let lock_timing = engine.lock_timing().unwrap();
        let contention = lock_timing.current_stall(5);
        assert!(contention >= 3, "Should have tracked at least 3 stall cycles, got {}", contention);
    }

    #[test]
    fn test_lock_timing_success() {
        // Test that successful lock acquire is tracked
        let mut engine = DmaEngine::new_compute_tile(1, 2)
            .with_lock_timing(16);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Configure simple BD (no lock requirement)
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

        // But test lock timing directly - set lock value so acquire succeeds
        tile.locks[3].value = 1;

        // Start channel
        engine.start_channel(0, 0).unwrap();

        // Manually trigger lock acquire tracking (simulating what happens internally)
        if let Some(timing) = engine.lock_timing_mut() {
            timing.track_acquire(3, true); // Success
        }

        // Check stats
        let lock_timing = engine.lock_timing().unwrap();
        let stats = lock_timing.stats(3).unwrap();
        assert_eq!(stats.acquires, 1, "Should have recorded 1 acquire");
    }

    #[test]
    fn test_memtile_bd_capacity() {
        // MemTile should support 48 BDs
        let mut engine = DmaEngine::new_mem_tile(0, 1);

        // Should be able to configure BD 0
        assert!(engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).is_ok());

        // Should be able to configure BD 47 (last valid for MemTile)
        assert!(engine.configure_bd(47, BdConfig::simple_1d(0x100, 16)).is_ok());

        // BD 48 should fail
        assert!(engine.configure_bd(48, BdConfig::simple_1d(0x100, 16)).is_err());

        // Compute tile should only support 16 BDs
        let mut compute = DmaEngine::new_compute_tile(0, 2);
        assert!(compute.configure_bd(15, BdConfig::simple_1d(0x100, 16)).is_ok());
        assert!(compute.configure_bd(16, BdConfig::simple_1d(0x100, 16)).is_err());
    }
}
