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
    COMPUTE_S2MM_CHANNELS, COMPUTE_MM2S_CHANNELS,
    MEM_TILE_S2MM_CHANNELS, MEM_TILE_MM2S_CHANNELS,
};
use super::compression;
use super::transfer::{Transfer, TransferDirection, TransferEndpoint, TransferState};
use super::timing::{DmaTimingConfig, ChannelTimingState, TransferPhase};
use crate::device::host_memory::HostMemory;
use crate::device::tile::{Tile, TileType};
use crate::interpreter::state::EventType;
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

/// Task complete token emitted when a DMA task finishes.
///
/// When Enable_Token_Issue is set in Start_Queue and the task completes,
/// the DMA engine emits a token with the channel's Controller_ID.
/// This allows software to track task completion without polling.
#[derive(Debug, Clone, Copy)]
pub struct TaskCompleteToken {
    /// Source tile column
    pub col: u8,
    /// Source tile row
    pub row: u8,
    /// Channel that completed the task
    pub channel: u8,
    /// Controller ID from channel control register (bits 15:8)
    pub controller_id: u8,
}

/// Per-channel task configuration (set when Start_Queue is written).
#[derive(Debug, Clone, Copy, Default)]
pub struct ChannelTaskConfig {
    /// Enable token issue when task completes
    pub enable_token_issue: bool,
    /// Controller ID for task tokens (from channel control register)
    pub controller_id: u8,
    /// FoT mode for S2MM channels
    pub fot_mode: u8,
    /// Compression enable (channel-level, MM2S only)
    pub compression_enable: bool,
    /// Decompression enable (channel-level, S2MM only)
    pub decompression_enable: bool,
    /// Out-of-order mode enable (S2MM only)
    pub out_of_order_enable: bool,
}

/// Task queue entry (enqueued via Start_Queue register).
///
/// Per AM025, each channel has an 8-deep task queue. Tasks are enqueued
/// by writing to the Start_Queue register and processed in FIFO order.
#[derive(Debug, Clone)]
pub struct TaskQueueEntry {
    /// Starting BD index for this task
    pub start_bd: u8,
    /// Repeat count (actual - 1, so 0 = run once, 255 = run 256 times)
    pub repeat_count: u8,
    /// Enable token issue when this task completes
    pub enable_token_issue: bool,
    /// Snapshot of the BD config at enqueue time.
    /// NPU instructions may overwrite the BD between enqueue and execution,
    /// so we capture the config when the task enters the queue.
    pub bd_snapshot: Option<super::BdConfig>,
}

/// Maximum task queue depth per channel (AM025: 3-bit Task_Queue_Size = 0-7)
pub const MAX_TASK_QUEUE_DEPTH: usize = 8;

/// Transfer operation result (internal).
#[derive(Debug, Clone, Copy)]
struct TransferResult {
    /// Transfer succeeded (or stalled waiting for resource)
    success: bool,
    /// Transfer stalled waiting for stream data (S2MM only)
    /// When stalled, timing should NOT advance and no error should be raised.
    stall: bool,
    /// TLAST received on S2MM - finish early if FoT enabled
    fot_finish: bool,
}

impl TransferResult {
    fn success() -> Self {
        Self { success: true, stall: false, fot_finish: false }
    }
    fn stalled() -> Self {
        Self { success: true, stall: true, fot_finish: false }
    }
    fn failure() -> Self {
        Self { success: false, stall: false, fot_finish: false }
    }
}

/// S2MM transfer result (internal).
#[derive(Debug, Clone, Copy)]
struct S2mmResult {
    /// Transfer was successful (data written) or stalled (no data available)
    success: bool,
    /// Stalled waiting for stream input data
    stall: bool,
    /// TLAST was received (for FoT mode)
    tlast_received: bool,
    /// Number of bytes actually written
    bytes_written: usize,
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

    /// Tile type (affects channel count, BD count, and transfer endpoints)
    pub tile_type: TileType,

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

    /// Task complete token output buffer.
    /// Tokens are emitted when tasks complete with Enable_Token_Issue set.
    task_tokens: VecDeque<TaskCompleteToken>,

    /// Per-channel task configuration (set from Start_Queue and channel control).
    channel_task_configs: Vec<ChannelTaskConfig>,

    /// Task queues per channel (8-deep FIFO per AM025).
    /// Tasks are enqueued via Start_Queue and processed in order.
    task_queues: Vec<VecDeque<TaskQueueEntry>>,

    /// Task queue overflow flags per channel (sticky bit, write-to-clear).
    /// Set when attempting to enqueue to a full queue.
    task_queue_overflow: Vec<bool>,

    /// Error: BD unavailable flags per channel (sticky bit, write-to-clear).
    /// Set in OOO mode when S2MM tries to load an invalid BD ID from packet header.
    error_bd_unavailable: Vec<bool>,

    /// Lock timing state for contention tracking (optional).
    /// When enabled, tracks detailed lock acquire/release timing per lock.
    lock_timing: Option<LockTimingState>,

    /// Current cycle, set by coordinator before each step.
    /// Used for timestamping trace events.
    current_cycle: u64,

    /// Trace events generated during this step, drained by coordinator.
    /// Events are buffered here because DmaEngine doesn't own an EventLog
    /// directly -- the coordinator collects them after each step.
    trace_events: Vec<(u64, EventType)>,

    /// Number of memory banks for this tile type (8 for compute, 16 for MemTile).
    /// Used to compute bank indices for conflict detection.
    num_banks: usize,

    /// Bitmask of memory banks accessed by DMA during this cycle.
    /// Transferred to the tile at the end of each DMA step for comparison
    /// with core bank accesses.
    pub cycle_dma_banks: u16,
}

impl DmaEngine {
    /// Create a new DMA engine with the given channel and BD counts.
    ///
    /// `num_channels` and `num_bds` come from the architecture configuration
    /// (ArchConfig) rather than compile-time constants. The caller is
    /// responsible for providing the correct values for the tile type.
    pub fn new(col: u8, row: u8, tile_type: TileType, num_channels: usize, num_bds: usize) -> Self {
        let mut transfers = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            transfers.push(None);
        }

        log::debug!("DmaEngine::new col={} row={} tile_type={:?} num_channels={} num_bds={}",
            col, row, tile_type, num_channels, num_bds);

        Self {
            col,
            row,
            tile_type,
            bd_configs: vec![BdConfig::default(); num_bds],
            transfers,
            channel_states: vec![ChannelState::Idle; num_channels],
            channel_stats: vec![ChannelStats::default(); num_channels],
            queued_bds: vec![None; num_channels],
            repeat_counts: vec![0; num_channels],
            current_bds: vec![None; num_channels],
            // Cycle-accurate timing is the default and only mode
            timing_config: DmaTimingConfig::from_aie2_spec(),
            timing_states: vec![ChannelTimingState::default(); num_channels],
            // Stream buffers with reasonable capacity
            stream_out: VecDeque::with_capacity(16),
            stream_in: VecDeque::with_capacity(16),
            // Task tokens and per-channel task config
            task_tokens: VecDeque::with_capacity(4),
            channel_task_configs: vec![ChannelTaskConfig::default(); num_channels],
            // Task queues (8-deep per channel)
            task_queues: (0..num_channels).map(|_| VecDeque::with_capacity(MAX_TASK_QUEUE_DEPTH)).collect(),
            task_queue_overflow: vec![false; num_channels],
            // Error tracking per channel
            error_bd_unavailable: vec![false; num_channels],
            // Lock timing disabled by default
            lock_timing: None,
            // Trace event buffer
            current_cycle: 0,
            trace_events: Vec::new(),
            // Bank conflict tracking
            num_banks: match tile_type {
                TileType::Compute => crate::device::aie2_spec::COMPUTE_TILE_MEMORY_BANKS,
                TileType::MemTile => crate::device::aie2_spec::MEMTILE_MEMORY_BANKS,
                TileType::Shim => 0,
            },
            cycle_dma_banks: 0,
        }
    }

    /// Create a new DMA engine for a compute tile (NPU1/AIE2 defaults).
    ///
    /// Uses hardcoded channel/BD counts matching Phoenix/HawkPoint.
    /// Production code should use `new()` with ArchConfig-derived values.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::Compute,
            COMPUTE_S2MM_CHANNELS + COMPUTE_MM2S_CHANNELS, 16)
    }

    /// Create a new DMA engine for a memory tile (NPU1/AIE2 defaults).
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::MemTile,
            MEM_TILE_S2MM_CHANNELS + MEM_TILE_MM2S_CHANNELS, 48)
    }

    /// Create a new DMA engine for a shim tile (NPU1/AIE2 defaults).
    pub fn new_shim_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::Shim,
            COMPUTE_S2MM_CHANNELS + COMPUTE_MM2S_CHANNELS, 16)
    }

    /// Configure custom timing parameters.
    ///
    /// The default is already cycle-accurate (`DmaTimingConfig::from_aie2_spec()`).
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
    fn trace(&mut self, event: EventType) {
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
        self.transfers.len()
    }

    /// Get the DMA channel status field layout for this engine's tile type.
    ///
    /// Returns the compute tile layout for compute/shim tiles, and the
    /// memtile layout for memory tiles. The key difference is the `cur_bd`
    /// field width: 4 bits for compute (16 BDs), 6 bits for memtile (48 BDs).
    fn status_layout(&self) -> &'static crate::device::regdb::StatusFieldLayout {
        let layout = crate::device::regdb::device_reg_layout();
        if self.tile_type.is_mem_tile() {
            &layout.memtile_status
        } else {
            &layout.memory_status
        }
    }

    /// Get the channel type (S2MM or MM2S).
    pub fn channel_type(&self, channel: ChannelId) -> ChannelType {
        ChannelType::from_channel_index(channel as usize, self.tile_type)
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
        let transfer = Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row, self.tile_type)?;

        // Initialize timing state for cycle-accurate execution
        self.timing_states[ch_idx] = ChannelTimingState::new_transfer(
            bd_config.length as u64,
            &self.timing_config,
        );

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

        // Emit DMA_START_TASK trace event
        self.trace(EventType::DmaStartTask { channel });

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
        let s2mm_count = if self.tile_type.is_mem_tile() {
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

    /// Check if a channel has pending work (active, waiting, or has queued work).
    ///
    /// This is useful for test loops that need to run until all work is complete,
    /// including BD chaining, repeats, and queued tasks.
    pub fn channel_has_pending_work(&self, channel: ChannelId) -> bool {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return false;
        }

        // Channel is actively working
        if matches!(
            self.channel_states[ch_idx],
            ChannelState::Active | ChannelState::WaitingForLock(_)
        ) {
            return true;
        }

        // Channel has queued BD (from chaining or repeat)
        if self.queued_bds[ch_idx].is_some() {
            return true;
        }

        // Channel has pending repeat count
        if self.repeat_counts[ch_idx] > 0 {
            return true;
        }

        // Channel has queued tasks
        if !self.task_queues[ch_idx].is_empty() {
            return true;
        }

        // Complete state needs one more step to process queued work
        if self.channel_states[ch_idx] == ChannelState::Complete {
            return true;
        }

        false
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

        // Debug: trace shim DMA stepping (only first 10 cycles to avoid spam)
        static SHIM_TRACE_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if self.col == 0 && self.row == 0 {
            let count = SHIM_TRACE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 20 {
                for ch_idx in 0..self.num_channels() {
                    let phase = self.timing_states[ch_idx].phase;
                    log::info!("SHIM_STEP[{}] ch{} state={:?} phase={:?}",
                        count, ch_idx, self.channel_states[ch_idx], phase);
                }
            }
        }

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
                        self.trace(EventType::DmaStalledLock { channel: ch_idx as u8 });
                        any_waiting = true;
                    }
                }
                ChannelState::Complete => {
                    // Check for BD chaining or repeat
                    if let Some(next_bd) = self.queued_bds[ch_idx].take() {
                        // Preserve existing repeat count (set by complete_transfer for repeats)
                        let repeat_count = self.repeat_counts[ch_idx];
                        log::debug!("DMA tile({},{}) ch{} starting queued BD {} (repeat={})",
                            self.col, self.row, ch_idx, next_bd, repeat_count);
                        if let Err(e) = self.start_channel_with_repeat(ch_idx as u8, next_bd, repeat_count) {
                            log::warn!("DMA tile({},{}) ch{} failed to start BD {}: {:?}",
                                self.col, self.row, ch_idx, next_bd, e);
                            self.channel_states[ch_idx] = ChannelState::Error;
                        } else {
                            log::debug!("DMA tile({},{}) ch{} now active with BD {}",
                                self.col, self.row, ch_idx, next_bd);
                            any_active = true;
                        }
                    } else {
                        // No more work to do, transition to Idle
                        self.channel_states[ch_idx] = ChannelState::Idle;
                        log::debug!("DMA tile({},{}) ch{} completed, now idle",
                            self.col, self.row, ch_idx);
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

    /// Step a single channel with cycle-accurate timing.
    ///
    /// Uses the phase-based timing state machine for realistic DMA behavior.
    fn step_channel(&mut self, ch_idx: usize, tile: &mut Tile, host_memory: &mut HostMemory) {
        // Check if we have a transfer that needs processing
        let transfer = match &self.transfers[ch_idx] {
            Some(t) => t,
            None => {
                if self.col == 0 && self.row == 0 {
                    log::trace!("SHIM step_channel ch{}: no transfer", ch_idx);
                }
                return;
            }
        };

        // Need to process both Active (data transfer) and ReleasingLock (lock release)
        // states - both require the timing state machine to tick through phases
        if !transfer.needs_processing() {
            if self.col == 0 && self.row == 0 {
                log::trace!("SHIM step_channel ch{}: transfer not needs_processing, state={:?}",
                    ch_idx, transfer.state);
            }
            return;
        }

        // Always use cycle-accurate timing
        self.step_channel_timed(ch_idx, tile, host_memory);
    }

    /// Step channel with cycle-accurate timing.
    ///
    /// Uses the phase-based timing state machine for realistic delays.
    /// The timing state machine handles phase transitions and word counting.
    /// This method performs the actual data movement when in DataTransfer phase.
    fn step_channel_timed(&mut self, ch_idx: usize, tile: &mut Tile, host_memory: &mut HostMemory) {
        // Insert packet header if needed (MM2S only, at start of transfer)
        self.maybe_insert_packet_header(ch_idx);

        // Get lock info from transfer and check if lock has been acquired.
        // has_acquire_lock: whether the BD requires lock acquisition before transfer
        // has_release_lock: whether the BD requires lock release after transfer
        // These are separate because a BD can have release-only (no acquire).
        let (has_acquire_lock, has_release_lock, lock_available) = {
            let transfer = self.transfers[ch_idx].as_ref()
                .expect("BUG: step_channel_timed called without active transfer");
            let has_acquire_lock = transfer.acquire_lock.is_some();
            let has_release_lock = transfer.release_lock.is_some();

            // Lock is "available" to the timing FSM if:
            // - No lock is required, OR
            // - The Transfer state is Active (meaning lock was already acquired via try_acquire_lock)
            //
            // Note: We don't check the raw lock value because try_acquire_lock() already
            // modified it when it succeeded. The Transfer.state tracks whether acquisition happened.
            let lock_available = !has_acquire_lock
                || transfer.state == TransferState::Active;
            (has_acquire_lock, has_release_lock, lock_available)
        };

        // Check if we need to do data transfer this cycle (before tick advances state)
        let should_transfer = self.timing_states[ch_idx].phase == TransferPhase::DataTransfer
            && self.timing_states[ch_idx].memory_pipeline_busy == 0;

        // Perform data transfer if in DataTransfer phase
        if should_transfer {
            let words_per_cycle = self.timing_config.words_per_cycle as usize;
            let bytes_per_cycle = words_per_cycle * 4;

            let (bytes_to_transfer, addr, source, dest, channel, is_last, tlast_suppress) = {
                let transfer = self.transfers[ch_idx].as_ref()
                    .expect("BUG: transfer missing in timed extraction block");
                let bytes_to_transfer = bytes_per_cycle.min(transfer.remaining_bytes() as usize);
                let remaining_after = transfer.remaining_bytes().saturating_sub(bytes_to_transfer as u64);
                (
                    bytes_to_transfer,
                    transfer.current_address(),
                    transfer.source,
                    transfer.dest,
                    transfer.channel,
                    remaining_after == 0,
                    transfer.tlast_suppress,
                )
            };

            if bytes_to_transfer > 0 {
                let result = self.do_transfer(source, dest, addr, bytes_to_transfer, channel, is_last, tlast_suppress, tile, host_memory);

                if result.stall {
                    // S2MM waiting for stream data - don't advance timing, just return
                    // The transfer will try again next cycle when data may be available
                    self.trace(EventType::DmaStreamStarvation { channel });
                    return;
                }

                if result.success {
                    // Update transfer state (address generator, bytes transferred)
                    let transfer = self.transfers[ch_idx].as_mut()
                        .expect("BUG: transfer missing after timed do_transfer");
                    transfer.data_transferred(bytes_to_transfer as u64);
                    self.channel_stats[ch_idx].bytes_transferred += bytes_to_transfer as u64;

                    // Handle FoT completion (early finish on TLAST)
                    if result.fot_finish {
                        self.complete_transfer(ch_idx, tile);
                        return;
                    }
                } else {
                    self.set_transfer_error(ch_idx, addr, tile.data_memory().len() as u64);
                    return;
                }
            }
        }

        // Advance timing state (this handles phase transitions and word counting)
        let old_phase = self.timing_states[ch_idx].phase;
        let phase = self.timing_states[ch_idx].tick(
            &self.timing_config, has_acquire_lock, has_release_lock, lock_available,
        );

        if self.col == 0 && self.row == 0 {
            log::trace!("SHIM tick ch{}: {:?} -> {:?} (remaining={}, words={}/{})",
                ch_idx, old_phase, phase,
                self.timing_states[ch_idx].cycles_remaining,
                self.timing_states[ch_idx].words_transferred,
                self.timing_states[ch_idx].total_words);
        }

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
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn do_transfer(
        &mut self,
        source: TransferEndpoint,
        dest: TransferEndpoint,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> TransferResult {
        match (source, dest) {
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::Stream { .. }) => {
                // MM2S: Read from tile memory, queue to stream output
                if self.transfer_mm2s(addr, bytes, channel, is_last, tlast_suppress, tile) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::TileMemory { .. }) => {
                // S2MM: Read from stream input, write to tile memory
                let result = self.transfer_s2mm(addr, bytes, channel, tile);
                if result.stall {
                    // No stream data available - stall (not an error)
                    return TransferResult::stalled();
                }
                if result.success {
                    // Check FoT mode: if enabled and TLAST received, signal early finish
                    let fot_mode = self.get_channel_fot_mode(channel);
                    let fot_finish = result.tlast_received && fot_mode != 0;

                    if fot_finish {
                        log::debug!(
                            "DMA({},{}) ch{} FoT mode {} triggered by TLAST ({} bytes written)",
                            self.col, self.row, channel, fot_mode, result.bytes_written
                        );
                    }

                    TransferResult { success: true, stall: false, fot_finish }
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::TileMemory { .. }) => {
                if Self::transfer_host_to_tile_static(addr, bytes, tile, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::HostMemory) => {
                // Record bank access for conflict detection (tile read by DMA)
                tile.record_dma_bank_access(addr as u32, bytes);
                if Self::transfer_tile_to_host_static(addr, bytes, tile, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::HostMemory, TransferEndpoint::Stream { .. }) => {
                // Shim MM2S: Read from host DDR, queue to stream output
                if self.transfer_host_to_stream(addr, bytes, channel, is_last, tlast_suppress, host_memory) {
                    TransferResult::success()
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::Stream { .. }, TransferEndpoint::HostMemory) => {
                // Shim S2MM: Read from stream input, write to host DDR
                let result = self.transfer_stream_to_host(addr, bytes, channel, host_memory);
                if result.stall {
                    // No stream data available - stall (not an error)
                    return TransferResult::stalled();
                }
                if result.success {
                    // Check FoT mode: if enabled and TLAST received, signal early finish
                    let fot_mode = self.get_channel_fot_mode(channel);
                    let fot_finish = result.tlast_received && fot_mode != 0;

                    if fot_finish {
                        log::debug!(
                            "DMA({},{}) Shim S2MM ch{} FoT mode {} triggered by TLAST ({} bytes written)",
                            self.col, self.row, channel, fot_mode, result.bytes_written
                        );
                    }

                    TransferResult { success: true, stall: false, fot_finish }
                } else {
                    TransferResult::failure()
                }
            }
            (TransferEndpoint::TileMemory { .. }, TransferEndpoint::TileMemory { .. }) => {
                // Tile to tile: Would need array access, mark as success for now
                TransferResult::success()
            }
            _ => TransferResult::failure(),
        }
    }

    /// MM2S: Read from tile memory and queue to stream output.
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn transfer_mm2s(&mut self, addr: u64, bytes: usize, channel: u8, is_last: bool, tlast_suppress: bool, tile: &Tile) -> bool {
        let mem_size = tile.data_memory().len();

        // MemTile DMA addresses may have a 0x80000 offset in the address space
        // Wrap addresses to stay within memory bounds
        let offset = (addr as usize) % mem_size;

        // Record bank access for conflict detection
        self.cycle_dma_banks |= crate::device::aie2_spec::banks_for_access(
            offset as u32, bytes, self.num_banks,
        );

        if offset + bytes > mem_size {
            log::warn!("DMA({},{}) MM2S addr=0x{:X} bytes={} wraps past memory end (size=0x{:X})",
                self.col, self.row, addr, bytes, mem_size);
            return false;
        }

        let data = tile.data_memory();

        // When compression is enabled, read 32-byte blocks and compress each one.
        // Compressed output (mask + packed non-zero bytes) is pushed as 32-bit stream words.
        if self.is_compression_enabled(channel) {
            return self.transfer_mm2s_compressed(data, offset, bytes, channel, is_last, tlast_suppress);
        }

        // Uncompressed path: read data from tile memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!("DMA({},{}) MM2S ch{}: addr=0x{:X} offset=0x{:X} bytes={} words={}",
            self.col, self.row, channel, addr, offset, bytes, word_count);

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

            if i < 2 || i == word_count - 1 {
                log::debug!("DMA({},{}) MM2S ch{} word[{}]: offset=0x{:X} value=0x{:08X}",
                    self.col, self.row, channel, i, word_offset, word);
            }

            // TLAST is asserted on last word unless suppressed
            // AM025: TLAST_Suppress (Word 5, bit 31) prevents TLAST assertion
            let is_last_word = is_last && (i == word_count - 1);
            let should_assert_tlast = is_last_word && !tlast_suppress;
            self.stream_out.push_back(StreamData {
                data: word,
                tlast: should_assert_tlast,
                channel,
            });
        }

        true
    }

    /// MM2S compressed path: read 32-byte blocks, compress, push to stream.
    ///
    /// Sparsity compression (AM020 Ch1) operates on 256-bit (32-byte) blocks.
    /// Each block produces a 32-bit mask followed by packed non-zero bytes,
    /// padded to a 4-byte boundary. The compressed output is pushed as 32-bit
    /// stream words.
    fn transfer_mm2s_compressed(
        &mut self,
        data: &[u8],
        offset: usize,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
    ) -> bool {
        const BLOCK_SIZE: usize = 32;

        log::debug!("DMA({},{}) MM2S ch{} COMPRESSED: offset=0x{:X} bytes={}",
            self.col, self.row, channel, offset, bytes);

        let num_blocks = (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for block_idx in 0..num_blocks {
            let block_start = offset + block_idx * BLOCK_SIZE;
            let block_end = (block_start + BLOCK_SIZE).min(offset + bytes).min(data.len());
            let block_len = block_end - block_start;

            // Pad to 32 bytes if this is a partial final block
            let mut block = [0u8; BLOCK_SIZE];
            block[..block_len].copy_from_slice(&data[block_start..block_end]);

            let compressed = match compression::compress(&block) {
                Some(c) => c,
                None => {
                    log::error!("DMA({},{}) MM2S ch{} compression failed at block {}",
                        self.col, self.row, channel, block_idx);
                    return false;
                }
            };

            log::debug!("DMA({},{}) MM2S ch{} block {}: {} bytes -> {} compressed bytes",
                self.col, self.row, channel, block_idx, BLOCK_SIZE, compressed.len());

            // Push compressed bytes as 32-bit stream words
            let compressed_words = compressed.len() / 4;
            let is_last_block = is_last && block_idx == num_blocks - 1;

            for w in 0..compressed_words {
                let wi = w * 4;
                let word = u32::from_le_bytes([
                    compressed[wi],
                    compressed[wi + 1],
                    compressed[wi + 2],
                    compressed[wi + 3],
                ]);

                let is_last_word = is_last_block && w == compressed_words - 1;
                let should_assert_tlast = is_last_word && !tlast_suppress;
                self.stream_out.push_back(StreamData {
                    data: word,
                    tlast: should_assert_tlast,
                    channel,
                });
            }
        }

        true
    }

    /// S2MM: Read from stream input, write to tile memory.
    ///
    /// Only transfers data that is available in stream_in for the specified channel.
    /// Returns S2mmResult indicating success, TLAST reception, and bytes written.
    fn transfer_s2mm(&mut self, addr: u64, bytes: usize, channel: u8, tile: &mut Tile) -> S2mmResult {
        let mem_size = tile.data_memory().len();

        // MemTile DMA addresses may have a 0x80000 offset in the address space
        // Wrap addresses to stay within memory bounds
        let offset = (addr as usize) % mem_size;

        // Record bank access for conflict detection
        self.cycle_dma_banks |= crate::device::aie2_spec::banks_for_access(
            offset as u32, bytes, self.num_banks,
        );

        if offset + bytes > mem_size {
            log::warn!("DMA({},{}) S2MM addr=0x{:X} bytes={} wraps past memory end (size=0x{:X})",
                self.col, self.row, addr, bytes, mem_size);
            return S2mmResult { success: false, stall: false, tlast_received: false, bytes_written: 0 };
        }

        // Must have at least one word for this channel to transfer
        // If no data is available, we stall (not an error - just waiting for producer)
        if !self.has_stream_in_for_channel(channel) {
            return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
        }

        // When decompression is enabled, consume compressed blocks from stream
        // and write decompressed 32-byte blocks to memory.
        if self.is_decompression_enabled(channel) {
            return self.transfer_s2mm_decompressed(offset, bytes, channel, tile);
        }

        // Uncompressed path: write data to tile memory in 32-bit words
        let data = tile.data_memory_mut();
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;
        let mut tlast_received = false;

        log::debug!("DMA({},{}) S2MM ch{}: addr=0x{:X} offset=0x{:X} bytes={} words={}",
            self.col, self.row, channel, addr, offset, bytes, word_count);

        for word_idx in 0..word_count {
            // Get data from stream for this specific channel
            if let Some(stream_data) = self.pop_stream_in_for_channel(channel) {
                let word = stream_data.data;
                let word_bytes = word.to_le_bytes();

                if word_idx < 2 || word_idx == word_count - 1 {
                    log::debug!("DMA({},{}) S2MM ch{} word[{}]: offset=0x{:X} value=0x{:08X}",
                        self.col, self.row, channel, word_idx, offset + bytes_written, word);
                }
                for j in 0..4 {
                    if bytes_written + j < bytes && offset + bytes_written + j < data.len() {
                        data[offset + bytes_written + j] = word_bytes[j];
                    }
                }
                bytes_written += 4;

                // Track TLAST for FoT mode
                if stream_data.tlast {
                    tlast_received = true;
                    break; // Stop receiving after TLAST
                }
            } else {
                // No more stream data for this channel - transfer partial, continue next step
                break;
            };
        }

        S2mmResult { success: true, stall: false, tlast_received, bytes_written }
    }

    /// S2MM decompressed path: read compressed blocks from stream, decompress, write to memory.
    ///
    /// Sparsity decompression (AM020 Ch1) consumes compressed blocks from the stream.
    /// Each block starts with a 32-bit mask word, followed by ceil(popcount(mask)/4)
    /// data words containing packed non-zero bytes. The decompressed 32-byte output
    /// is written to tile memory.
    fn transfer_s2mm_decompressed(
        &mut self,
        offset: usize,
        bytes: usize,
        channel: u8,
        tile: &mut Tile,
    ) -> S2mmResult {
        const BLOCK_SIZE: usize = 32;

        log::debug!("DMA({},{}) S2MM ch{} DECOMPRESSED: offset=0x{:X} bytes={}",
            self.col, self.row, channel, offset, bytes);

        let mut mem_bytes_written: usize = 0;
        let mut tlast_received = false;

        // Process compressed blocks until we have written enough decompressed bytes
        while mem_bytes_written < bytes {
            // Read the mask word (first word of compressed block)
            let mask_data = match self.pop_stream_in_for_channel(channel) {
                Some(sd) => sd,
                None => break, // No more stream data; partial transfer
            };

            if mask_data.tlast {
                // TLAST on the mask word itself -- unusual, but handle gracefully.
                // Treat as an empty block (mask only, no data bytes).
                tlast_received = true;
                break;
            }

            let mask = mask_data.data;
            let non_zero_count = mask.count_ones() as usize;
            let data_bytes_needed = non_zero_count;
            let data_words_needed = (data_bytes_needed + 3) / 4;

            // Collect the data words for this compressed block
            let mut compressed_buf = Vec::with_capacity(4 + data_words_needed * 4);
            compressed_buf.extend_from_slice(&mask.to_le_bytes());

            let mut got_tlast = false;
            for _ in 0..data_words_needed {
                match self.pop_stream_in_for_channel(channel) {
                    Some(sd) => {
                        compressed_buf.extend_from_slice(&sd.data.to_le_bytes());
                        if sd.tlast {
                            got_tlast = true;
                            break;
                        }
                    }
                    None => break, // Stream starved mid-block
                }
            }

            // Decompress (tolerates short input: missing bytes decompress as zero)
            match compression::decompress(&compressed_buf) {
                Some(decompressed) => {
                    let data = tile.data_memory_mut();
                    let write_len = BLOCK_SIZE.min(bytes - mem_bytes_written);
                    let dest_start = offset + mem_bytes_written;
                    let dest_end = (dest_start + write_len).min(data.len());
                    let actual_write = dest_end - dest_start;

                    data[dest_start..dest_end].copy_from_slice(&decompressed[..actual_write]);
                    mem_bytes_written += actual_write;

                    log::debug!("DMA({},{}) S2MM ch{} decompressed block: mask=0x{:08X} {} non-zero -> {} bytes to mem",
                        self.col, self.row, channel, mask, non_zero_count, actual_write);
                }
                None => {
                    log::error!("DMA({},{}) S2MM ch{} decompression failed (mask=0x{:08X}, buf_len={})",
                        self.col, self.row, channel, mask, compressed_buf.len());
                    return S2mmResult {
                        success: false, stall: false, tlast_received: false,
                        bytes_written: mem_bytes_written,
                    };
                }
            }

            if got_tlast {
                tlast_received = true;
                break;
            }
        }

        S2mmResult {
            success: mem_bytes_written > 0,
            stall: false,
            tlast_received,
            bytes_written: mem_bytes_written,
        }
    }

    /// Shim MM2S: Read from host DDR and queue to stream output.
    ///
    /// # Arguments
    /// * `tlast_suppress` - If true, TLAST is not asserted even on the last word
    fn transfer_host_to_stream(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        host_memory: &HostMemory,
    ) -> bool {
        // When compression is enabled, read 32-byte blocks and compress
        if self.is_compression_enabled(channel) {
            return self.transfer_host_to_stream_compressed(
                addr, bytes, channel, is_last, tlast_suppress, host_memory,
            );
        }

        // Uncompressed path: read data from host memory in 32-bit words
        let word_count = (bytes + 3) / 4;

        log::debug!("MM2S transfer: addr=0x{:X} bytes={} words={}", addr, bytes, word_count);

        for i in 0..word_count {
            let word_addr = addr + (i * 4) as u64;
            let word = host_memory.read_u32(word_addr);

            if i < 4 {
                log::debug!("  MM2S word[{}]: addr=0x{:X} -> 0x{:08X}", i, word_addr, word);
            }

            // TLAST is asserted on last word unless suppressed
            // AM025: TLAST_Suppress (Word 5, bit 31) prevents TLAST assertion
            let is_last_word = is_last && i == word_count - 1;
            let should_assert_tlast = is_last_word && !tlast_suppress;
            self.stream_out.push_back(StreamData {
                data: word,
                channel,
                tlast: should_assert_tlast,
            });
        }

        true
    }

    /// Shim MM2S compressed path: read 32-byte blocks from host DDR, compress, push to stream.
    fn transfer_host_to_stream_compressed(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        is_last: bool,
        tlast_suppress: bool,
        host_memory: &HostMemory,
    ) -> bool {
        const BLOCK_SIZE: usize = 32;

        log::debug!("Shim MM2S COMPRESSED: addr=0x{:X} bytes={}", addr, bytes);

        let num_blocks = (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for block_idx in 0..num_blocks {
            let block_start = addr + (block_idx * BLOCK_SIZE) as u64;
            let block_remaining = bytes - block_idx * BLOCK_SIZE;
            let block_len = BLOCK_SIZE.min(block_remaining);

            // Read 32-byte block from host memory
            let mut block = [0u8; BLOCK_SIZE];
            for i in 0..block_len {
                let byte_addr = block_start + i as u64;
                // Read individual bytes using word-aligned reads
                let word_addr = byte_addr & !3;
                let byte_offset = (byte_addr & 3) as usize;
                let word = host_memory.read_u32(word_addr);
                block[i] = word.to_le_bytes()[byte_offset];
            }

            let compressed = match compression::compress(&block) {
                Some(c) => c,
                None => {
                    log::error!("Shim MM2S ch{} compression failed at block {}", channel, block_idx);
                    return false;
                }
            };

            let compressed_words = compressed.len() / 4;
            let is_last_block = is_last && block_idx == num_blocks - 1;

            for w in 0..compressed_words {
                let wi = w * 4;
                let word = u32::from_le_bytes([
                    compressed[wi],
                    compressed[wi + 1],
                    compressed[wi + 2],
                    compressed[wi + 3],
                ]);

                let is_last_word = is_last_block && w == compressed_words - 1;
                let should_assert_tlast = is_last_word && !tlast_suppress;
                self.stream_out.push_back(StreamData {
                    data: word,
                    channel,
                    tlast: should_assert_tlast,
                });
            }
        }

        true
    }

    /// Shim S2MM: Read from stream input and write to host DDR.
    ///
    /// Returns S2mmResult to properly handle stalls when no stream data is available.
    fn transfer_stream_to_host(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        host_memory: &mut HostMemory,
    ) -> S2mmResult {
        // Must have at least one word for this channel to transfer
        // If no data is available, we stall (not an error - just waiting for producer)
        if !self.has_stream_in_for_channel(channel) {
            return S2mmResult { success: true, stall: true, tlast_received: false, bytes_written: 0 };
        }

        // When decompression is enabled, consume compressed blocks from stream
        if self.is_decompression_enabled(channel) {
            return self.transfer_stream_to_host_decompressed(addr, bytes, channel, host_memory);
        }

        // Uncompressed path: write data to host memory in 32-bit words
        let mut bytes_written = 0;
        let word_count = (bytes + 3) / 4;
        let mut tlast_received = false;

        for i in 0..word_count {
            let stream_data = if let Some(sd) = self.pop_stream_in_for_channel(channel) {
                sd
            } else {
                // No more stream data for this channel - transfer partial, continue next step
                break;
            };

            let word = stream_data.data;
            let word_addr = addr + (i * 4) as u64;
            log::info!("Shim S2MM write: addr=0x{:X} word=0x{:08X}", word_addr, word);
            host_memory.write_u32(word_addr, word);
            bytes_written += 4;

            // Track TLAST for FoT mode
            if stream_data.tlast {
                tlast_received = true;
                break; // Stop receiving after TLAST
            }
        }

        S2mmResult { success: bytes_written > 0, stall: false, tlast_received, bytes_written }
    }

    /// Shim S2MM decompressed path: read compressed blocks from stream, decompress, write to host DDR.
    fn transfer_stream_to_host_decompressed(
        &mut self,
        addr: u64,
        bytes: usize,
        channel: u8,
        host_memory: &mut HostMemory,
    ) -> S2mmResult {
        const BLOCK_SIZE: usize = 32;

        log::debug!("Shim S2MM DECOMPRESSED: addr=0x{:X} bytes={}", addr, bytes);

        let mut mem_bytes_written: usize = 0;
        let mut tlast_received = false;

        while mem_bytes_written < bytes {
            // Read the mask word
            let mask_data = match self.pop_stream_in_for_channel(channel) {
                Some(sd) => sd,
                None => break,
            };

            if mask_data.tlast {
                tlast_received = true;
                break;
            }

            let mask = mask_data.data;
            let non_zero_count = mask.count_ones() as usize;
            let data_words_needed = (non_zero_count + 3) / 4;

            let mut compressed_buf = Vec::with_capacity(4 + data_words_needed * 4);
            compressed_buf.extend_from_slice(&mask.to_le_bytes());

            let mut got_tlast = false;
            for _ in 0..data_words_needed {
                match self.pop_stream_in_for_channel(channel) {
                    Some(sd) => {
                        compressed_buf.extend_from_slice(&sd.data.to_le_bytes());
                        if sd.tlast {
                            got_tlast = true;
                            break;
                        }
                    }
                    None => break,
                }
            }

            match compression::decompress(&compressed_buf) {
                Some(decompressed) => {
                    let write_len = BLOCK_SIZE.min(bytes - mem_bytes_written);
                    let dest_addr = addr + mem_bytes_written as u64;

                    // Write decompressed bytes to host memory via word-aligned writes
                    let full_words = write_len / 4;
                    for w in 0..full_words {
                        let wi = w * 4;
                        let word = u32::from_le_bytes([
                            decompressed[wi],
                            decompressed[wi + 1],
                            decompressed[wi + 2],
                            decompressed[wi + 3],
                        ]);
                        host_memory.write_u32(dest_addr + wi as u64, word);
                    }
                    // Handle trailing bytes (partial last word)
                    let trailing = write_len % 4;
                    if trailing > 0 {
                        let wi = full_words * 4;
                        let mut word_bytes = [0u8; 4];
                        word_bytes[..trailing].copy_from_slice(&decompressed[wi..wi + trailing]);
                        host_memory.write_u32(dest_addr + wi as u64, u32::from_le_bytes(word_bytes));
                    }

                    mem_bytes_written += write_len;
                }
                None => {
                    log::error!("Shim S2MM ch{} decompression failed (mask=0x{:08X})", channel, mask);
                    return S2mmResult {
                        success: false, stall: false, tlast_received: false,
                        bytes_written: mem_bytes_written,
                    };
                }
            }

            if got_tlast {
                tlast_received = true;
                break;
            }
        }

        S2mmResult {
            success: mem_bytes_written > 0,
            stall: false,
            tlast_received,
            bytes_written: mem_bytes_written,
        }
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

        // Record bank access for conflict detection
        tile.record_dma_bank_access(offset as u32, bytes);

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
        // Use snapshot-based release for cycle-accurate behavior
        use crate::device::tile::Lock;
        if let TransferState::ReleasingLock(lock_id) = transfer.state {
            let release_delta = transfer.release_value;
            log::info!("DMA tile({},{}) releasing lock {} with delta {} on channel {} (snapshot-based)",
                self.col, self.row, lock_id, release_delta, ch_idx);
            if (lock_id as usize) < tile.locks.len() {
                // Record release as pending delta, applied at cycle end
                tile.release_snapshot(lock_id as usize, release_delta);
                let current = tile.locks[lock_id as usize].value;
                let new_val = (current as i16 + release_delta as i16).clamp(Lock::MIN_VALUE as i16, Lock::MAX_VALUE as i16) as i8;
                log::info!("Tile({},{}) lock {} pending delta {} (current={}, will be {} at cycle end)",
                    self.col, self.row, lock_id, release_delta, current, new_val);
            }
            transfer.lock_released();
        }

        // Update statistics
        self.channel_stats[ch_idx].transfers_completed += 1;
        self.channel_stats[ch_idx].cycles_spent += transfer.cycles_elapsed;

        // Extract transfer fields before releasing the borrow
        let next_bd = transfer.next_bd;
        let bd_index = transfer.bd_index;

        // Emit DMA_FINISHED_BD -- one BD within a task completed.
        // Push directly to avoid borrow conflict with `transfer`.
        self.trace_events.push((self.current_cycle, EventType::DmaFinishedBd { channel: ch_idx as u8 }));

        // Check for BD chaining first
        if let Some(next_bd) = next_bd {
            log::debug!("DMA tile({},{}) ch{} chaining to BD {} (from BD {})",
                self.col, self.row, ch_idx, next_bd, bd_index);
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
        // Task is complete (no chaining, no repeats) - emit token if enabled
        else {
            // Emit DMA_FINISHED_TASK -- entire task (all BDs and repeats) complete
            self.trace(EventType::DmaFinishedTask { channel: ch_idx as u8 });
            self.maybe_emit_task_token(ch_idx);

            // Check for more tasks in the queue
            if !self.task_queues[ch_idx].is_empty() {
                log::debug!(
                    "DMA tile({},{}) ch{} task complete, {} tasks remaining in queue",
                    self.col, self.row, ch_idx, self.task_queues[ch_idx].len()
                );
                // Mark as complete (will trigger next task in step())
                self.channel_states[ch_idx] = ChannelState::Complete;
                self.transfers[ch_idx] = None;
                // Start next queued task immediately
                self.start_next_queued_task(ch_idx as u8);
                return;
            }
        }

        // Mark channel as complete
        self.channel_states[ch_idx] = ChannelState::Complete;
        self.transfers[ch_idx] = None;
    }

    /// Emit a task complete token if Enable_Token_Issue is set for this channel.
    ///
    /// Called when a task completes (no more BD chaining, no more repeats).
    fn maybe_emit_task_token(&mut self, ch_idx: usize) {
        let config = &self.channel_task_configs[ch_idx];

        if config.enable_token_issue {
            let token = TaskCompleteToken {
                col: self.col,
                row: self.row,
                channel: ch_idx as u8,
                controller_id: config.controller_id,
            };

            log::debug!(
                "DMA tile({},{}) ch{} emitting task complete token (controller_id={})",
                self.col, self.row, ch_idx, config.controller_id
            );

            self.task_tokens.push_back(token);

            // Clear enable_token_issue after issuing (it's set per-task via Start_Queue)
            self.channel_task_configs[ch_idx].enable_token_issue = false;
        }
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

        // Capture pointers before mutable borrow for debugging
        let tile_ptr = tile as *const _ as usize;
        let lock_ptr = &tile.locks[lock_id as usize] as *const _ as usize;

        // Get snapshot value for logging (before any modifications)
        let snapshot_value = tile.lock_snapshot_value(lock_id as usize);

        // AIE-ML lock semantics from AM020:
        // - acquire_value < 0: acq_ge, wait until lock >= |value|, then decrement by |value|
        // - acquire_value > 0: acq_eq, wait until lock == value, then decrement to 0
        // - acquire_value == 0: simple acquire (decrement if > 0)
        //
        // Use snapshot-based arbitration for cycle-accurate behavior:
        // All lock operations within a cycle check against the snapshot taken at cycle start.
        let (expected, delta, equal_mode) = if acquire_value < 0 {
            // acq_ge: wait until lock >= |value|, then decrement by |value|
            ((-acquire_value) as i8, acquire_value, false)
        } else if acquire_value > 0 {
            // acq_eq: wait until lock == value, then decrement to 0
            (acquire_value as i8, -acquire_value, true)
        } else {
            // acquire_value == 0: simple acquire (decrement if > 0)
            (1i8, -1i8, false)
        };

        let result = tile.try_acquire_snapshot(lock_id as usize, expected, delta, equal_mode);
        let success = matches!(result, LockResult::Success);

        log::info!("DMA try_acquire_lock tile({},{}) ch{} lock={} expected={} delta={} snapshot={} result={:?} (tile_ptr=0x{:x} lock_ptr=0x{:x})",
            self.col, self.row, ch_idx, lock_id, expected, delta, snapshot_value, result,
            tile_ptr, lock_ptr);

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
            self.task_queues[i].clear();
            self.task_queue_overflow[i] = false;
            self.error_bd_unavailable[i] = false;
            self.repeat_counts[i] = 0;
            self.current_bds[i] = None;
        }

        // Clear stream buffers
        self.stream_out.clear();
        self.stream_in.clear();

        // Clear task tokens
        self.task_tokens.clear();

        // Don't clear BD configs - those are persistent configuration
    }

    // === Packet Header Insertion (MM2S only) ===

    /// Insert packet header if needed for MM2S transfers.
    ///
    /// Per AM025, when Enable_Packet=1 in BD Word 1, the MM2S channel inserts
    /// a 32-bit packet header before the data. This header contains:
    /// - Source tile location (col, row)
    /// - Packet type and ID from the BD
    /// - Odd parity bit
    ///
    /// This is called at the start of each step for active MM2S channels.
    fn maybe_insert_packet_header(&mut self, ch_idx: usize) {
        // Get transfer info
        let needs_header = self.transfers[ch_idx]
            .as_ref()
            .map(|t| t.needs_packet_header() && t.direction == TransferDirection::MM2S)
            .unwrap_or(false);

        if !needs_header {
            return;
        }

        // Generate and insert the packet header
        let (header, channel) = {
            let transfer = self.transfers[ch_idx].as_ref()
                .expect("BUG: transfer missing in packet header generation");
            let header = transfer.generate_packet_header();
            (header, transfer.channel)
        };

        if let Some(header_word) = header {
            // Insert packet header into stream output (before any data)
            // Packet header never has TLAST set
            self.stream_out.push_back(StreamData {
                data: header_word,
                tlast: false,
                channel,
            });

            log::debug!(
                "DMA({},{}) ch{} inserted packet header: 0x{:08X}",
                self.col, self.row, ch_idx, header_word
            );

            // Mark header as sent
            if let Some(transfer) = self.transfers[ch_idx].as_mut() {
                transfer.mark_packet_header_sent();
            }
        }
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
            log::warn!(
                "DMA({},{}) stream_in buffer full (256), dropping ch{} data: 0x{:08X}",
                self.col, self.row, data.channel, data.data
            );
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

    /// Check if stream input buffer has data for a specific channel.
    pub fn has_stream_in_for_channel(&self, channel: u8) -> bool {
        self.stream_in.iter().any(|d| d.channel == channel)
    }

    /// Pop data from stream input buffer for a specific channel.
    ///
    /// Scans the buffer for the first entry matching the channel and removes it.
    /// Returns None if no data for this channel is available.
    fn pop_stream_in_for_channel(&mut self, channel: u8) -> Option<StreamData> {
        // Find index of first matching entry
        let idx = self.stream_in.iter().position(|d| d.channel == channel)?;
        self.stream_in.remove(idx)
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

    // === Task Queue Interface ===

    /// Enqueue a task to the channel's task queue.
    ///
    /// Per AM025, each channel has an 8-deep task queue. Writing to Start_Queue
    /// enqueues a new task. If the queue is full, sets Task_Queue_Overflow.
    ///
    /// Returns true if the task was enqueued, false if queue was full.
    pub fn enqueue_task(
        &mut self,
        channel: u8,
        start_bd: u8,
        repeat_count: u8,
        enable_token_issue: bool,
    ) -> bool {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return false;
        }

        let queue = &mut self.task_queues[ch_idx];

        // Check for overflow
        if queue.len() >= MAX_TASK_QUEUE_DEPTH {
            self.task_queue_overflow[ch_idx] = true;
            log::warn!(
                "DMA tile({},{}) ch{} task queue overflow (queue full, BD {} rejected)",
                self.col, self.row, channel, start_bd
            );
            return false;
        }

        // Snapshot the BD config at enqueue time so later overwrites don't affect this task
        let bd_snapshot = self.bd_configs.get(start_bd as usize).cloned();

        // Enqueue the task
        queue.push_back(TaskQueueEntry {
            start_bd,
            repeat_count,
            enable_token_issue,
            bd_snapshot,
        });

        log::debug!(
            "DMA tile({},{}) ch{} enqueued task: BD={} repeat={} token={} (queue_size={})",
            self.col, self.row, channel, start_bd, repeat_count, enable_token_issue, queue.len()
        );

        // If channel is idle, start processing the queue
        if self.channel_states[ch_idx] == ChannelState::Idle
            || self.channel_states[ch_idx] == ChannelState::Complete
        {
            self.start_next_queued_task(channel);
        }

        true
    }

    /// Start the next task from the channel's queue.
    ///
    /// Dequeues the front task and starts the channel with it.
    fn start_next_queued_task(&mut self, channel: u8) {
        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return;
        }

        // Dequeue next task
        let task = match self.task_queues[ch_idx].pop_front() {
            Some(t) => t,
            None => return, // No tasks queued
        };

        // Set token config from the task
        self.channel_task_configs[ch_idx].enable_token_issue = task.enable_token_issue;

        // Validate BD index before attempting to start
        if task.start_bd as usize >= self.bd_configs.len() {
            log::error!(
                "DMA tile({},{}) ch{} queued task has invalid BD {} (max={})",
                self.col, self.row, channel, task.start_bd, self.bd_configs.len()
            );
            self.channel_states[ch_idx] = ChannelState::Error;
            return;
        }

        log::debug!(
            "DMA tile({},{}) ch{} starting queued task: BD={} repeat={} (remaining={})",
            self.col, self.row, channel, task.start_bd, task.repeat_count,
            self.task_queues[ch_idx].len()
        );

        // Restore BD snapshot: NPU instructions may have overwritten the BD
        // between enqueue time and now. The snapshot preserves the config that
        // was active when the task was enqueued (matching real hardware behavior).
        if let Some(snapshot) = task.bd_snapshot {
            let bd_idx = task.start_bd as usize;
            if bd_idx < self.bd_configs.len() {
                self.bd_configs[bd_idx] = snapshot;
            }
        }

        // Start the channel with the task
        if let Err(e) = self.start_channel_with_repeat(channel, task.start_bd, task.repeat_count) {
            log::error!(
                "DMA tile({},{}) ch{} failed to start queued task BD {}: {}",
                self.col, self.row, channel, task.start_bd, e
            );
            self.channel_states[ch_idx] = ChannelState::Error;
        }
    }

    /// Get the current task queue size for a channel.
    pub fn task_queue_size(&self, channel: u8) -> usize {
        self.task_queues
            .get(channel as usize)
            .map(|q| q.len())
            .unwrap_or(0)
    }

    /// Check if the task queue overflow flag is set for a channel.
    pub fn task_queue_overflow(&self, channel: u8) -> bool {
        self.task_queue_overflow
            .get(channel as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Clear the task queue overflow flag (write-to-clear per AM025).
    pub fn clear_task_queue_overflow(&mut self, channel: u8) {
        if let Some(flag) = self.task_queue_overflow.get_mut(channel as usize) {
            *flag = false;
        }
    }

    /// Check if the BD unavailable error flag is set for a channel.
    ///
    /// This is set in OOO mode when S2MM tries to load an invalid BD from packet header.
    pub fn error_bd_unavailable(&self, channel: u8) -> bool {
        self.error_bd_unavailable
            .get(channel as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Set the BD unavailable error flag (S2MM OOO mode).
    pub fn set_error_bd_unavailable(&mut self, channel: u8) {
        if let Some(flag) = self.error_bd_unavailable.get_mut(channel as usize) {
            *flag = true;
            log::warn!(
                "DMA tile({},{}) S2MM ch{} Error_BD_Unavailable: invalid BD in OOO packet header",
                self.col, self.row, channel
            );
        }
    }

    /// Clear the BD unavailable error flag (write-to-clear per AM025).
    pub fn clear_error_bd_unavailable(&mut self, channel: u8) {
        if let Some(flag) = self.error_bd_unavailable.get_mut(channel as usize) {
            *flag = false;
        }
    }

    // === Task Token Interface ===

    /// Set channel task configuration (called when channel control is written).
    ///
    /// This sets the persistent channel configuration (controller_id, fot_mode, etc.)
    /// that applies to all tasks on this channel.
    pub fn set_channel_task_config(
        &mut self,
        ch_idx: u8,
        enable_token_issue: bool,
        controller_id: u8,
        fot_mode: u8,
    ) {
        if (ch_idx as usize) < self.channel_task_configs.len() {
            let config = &mut self.channel_task_configs[ch_idx as usize];
            config.enable_token_issue = enable_token_issue;
            config.controller_id = controller_id;
            config.fot_mode = fot_mode;

            log::trace!(
                "DMA tile({},{}) ch{} set task config: token_issue={} controller_id={} fot_mode={}",
                self.col, self.row, ch_idx, enable_token_issue, controller_id, fot_mode
            );
        }
    }

    /// Set channel compression/decompression and out-of-order configuration.
    ///
    /// Called when channel control register is written.
    pub fn set_channel_compression_config(
        &mut self,
        ch_idx: u8,
        compression_enable: bool,
        decompression_enable: bool,
        out_of_order_enable: bool,
    ) {
        if (ch_idx as usize) < self.channel_task_configs.len() {
            let config = &mut self.channel_task_configs[ch_idx as usize];
            config.compression_enable = compression_enable;
            config.decompression_enable = decompression_enable;
            config.out_of_order_enable = out_of_order_enable;

            log::trace!(
                "DMA tile({},{}) ch{} set compression config: compress={} decompress={} ooo={}",
                self.col, self.row, ch_idx, compression_enable, decompression_enable, out_of_order_enable
            );
        }
    }

    /// Pop a task complete token from the output buffer.
    ///
    /// Returns None if no tokens are pending.
    pub fn pop_task_token(&mut self) -> Option<TaskCompleteToken> {
        self.task_tokens.pop_front()
    }

    /// Check if any task complete tokens are pending.
    pub fn has_task_token(&self) -> bool {
        !self.task_tokens.is_empty()
    }

    /// Get the number of pending task complete tokens.
    pub fn task_token_count(&self) -> usize {
        self.task_tokens.len()
    }

    /// Get the FoT mode for a channel (S2MM only).
    pub fn get_channel_fot_mode(&self, ch_idx: u8) -> u8 {
        self.channel_task_configs
            .get(ch_idx as usize)
            .map(|c| c.fot_mode)
            .unwrap_or(0)
    }

    // === Status Register Interface ===

    /// Get the current status register value for a channel.
    ///
    /// The status register format (AM025):
    /// - Bits 27:24: Cur_BD (current BD being processed)
    /// - Bits 22:20: Task_Queue_Size (current number of tasks in queue)
    /// - Bit 19: Channel_Running
    /// - Bit 18: Task_Queue_Overflow (sticky, write-to-clear)
    /// - Bits 1:0: Status state (00=IDLE, 01=STARTING, 10=RUNNING)
    ///
    /// Additional bits for stall conditions are set based on channel state.
    pub fn get_channel_status(&self, channel: u8) -> u32 {
        let layout = self.status_layout();

        let ch_idx = channel as usize;
        if ch_idx >= self.num_channels() {
            return 0;
        }

        let mut status: u32 = 0;

        // Cur_BD: 4 bits [27:24] for compute, 6 bits [29:24] for memtile
        if let Some(bd_idx) = self.current_bds[ch_idx] {
            status = layout.cur_bd.insert(status, bd_idx as u32);
        }

        // Task_Queue_Size (bits 22:20)
        let queue_size = self.task_queues[ch_idx].len() as u32;
        status = layout.task_queue_size.insert(status, queue_size);

        // Task_Queue_Overflow (bit 18)
        if self.task_queue_overflow[ch_idx] {
            status = layout.task_queue_overflow.set_bit(status);
        }

        // Error_BD_Unavailable (bit 10) - S2MM out-of-order mode only
        if self.error_bd_unavailable[ch_idx] {
            status = layout.error_bd_unavailable.set_bit(status);
        }

        // Channel state (bits 1:0 and bit 19)
        match self.channel_states[ch_idx] {
            ChannelState::Idle => {
                // State = 00 (IDLE), not running
            }
            ChannelState::Active => {
                // State = 10 (RUNNING), running = 1
                status = layout.status.insert(status, 0b10);
                status = layout.channel_running.set_bit(status);
            }
            ChannelState::Paused => {
                // Paused: running but not actively transferring
                status = layout.channel_running.set_bit(status);
            }
            ChannelState::WaitingForLock(_lock_id) => {
                // State = 10 (RUNNING), stalled on lock acquire
                status = layout.status.insert(status, 0b10);
                status = layout.channel_running.set_bit(status);
                status = layout.stalled_lock_acq.set_bit(status);
            }
            ChannelState::Complete => {
                // Transition state: will go to IDLE on next step
                status = layout.status.insert(status, 0b01);
            }
            ChannelState::Error => {
                // Error: set invalid BD error
                status = layout.error_bd_invalid.set_bit(status);
            }
        }

        // Check for stream stall (S2MM waiting for data)
        if let Some(transfer) = &self.transfers[ch_idx] {
            if matches!(transfer.direction, TransferDirection::S2MM)
                && !self.has_stream_in_for_channel(channel)
            {
                status = layout.stalled_stream.set_bit(status);
            }
        }

        status
    }

    /// Get whether compression is enabled for a channel (MM2S).
    pub fn is_compression_enabled(&self, channel: u8) -> bool {
        self.channel_task_configs
            .get(channel as usize)
            .map(|c| c.compression_enable)
            .unwrap_or(false)
    }

    /// Get whether decompression is enabled for a channel (S2MM).
    pub fn is_decompression_enabled(&self, channel: u8) -> bool {
        self.channel_task_configs
            .get(channel as usize)
            .map(|c| c.decompression_enable)
            .unwrap_or(false)
    }

    /// Get whether out-of-order mode is enabled for a channel (S2MM).
    pub fn is_out_of_order_enabled(&self, channel: u8) -> bool {
        self.channel_task_configs
            .get(channel as usize)
            .map(|c| c.out_of_order_enable)
            .unwrap_or(false)
    }

    // === Stream Port Mapping Integration ===
    //
    // These methods integrate with the stream_io module's port mappings,
    // providing a unified interface for determining which stream switch
    // ports DMA channels connect to.

    /// Get the slave port that MM2S channel `ch` sends data TO.
    ///
    /// For compute tiles: ch0 → slave port 1, ch1 → slave port 2
    /// For memtiles: ch0-5 → slave ports 0-5
    /// For shim tiles: ch0 → slave port 2, ch1 → slave port 3 (South ports)
    pub fn mm2s_slave_port(&self, ch: u8) -> u8 {
        match self.tile_type {
            TileType::Compute => super::stream_io::compute::mm2s_slave_port(ch),
            TileType::MemTile => super::stream_io::memtile::mm2s_slave_port(ch),
            TileType::Shim => {
                // Shim MM2S typically goes to South ports for NoC access
                // This mapping may need adjustment based on CDO configuration
                super::stream_io::shim::slave::SOUTH_0 + ch
            }
        }
    }

    /// Get the master port that S2MM channel `ch` receives data FROM.
    ///
    /// For compute tiles: ch0 ← master port 1, ch1 ← master port 2
    /// For memtiles: ch0-5 ← master ports 0-5
    /// For shim tiles: ch0 ← master port 2, ch1 ← master port 3 (South ports)
    pub fn s2mm_master_port(&self, ch: u8) -> u8 {
        match self.tile_type {
            TileType::Compute => super::stream_io::compute::s2mm_master_port(ch),
            TileType::MemTile => super::stream_io::memtile::s2mm_master_port(ch),
            TileType::Shim => {
                // Shim S2MM typically receives from South ports
                super::stream_io::shim::master::SOUTH_0 + ch
            }
        }
    }

    /// Get the number of S2MM channels for this tile type.
    pub fn s2mm_channel_count(&self) -> usize {
        match self.tile_type {
            TileType::Compute | TileType::Shim => COMPUTE_S2MM_CHANNELS,
            TileType::MemTile => MEM_TILE_S2MM_CHANNELS,
        }
    }

    /// Get the number of MM2S channels for this tile type.
    pub fn mm2s_channel_count(&self) -> usize {
        match self.tile_type {
            TileType::Compute | TileType::Shim => COMPUTE_MM2S_CHANNELS,
            TileType::MemTile => MEM_TILE_MM2S_CHANNELS,
        }
    }

    /// Convert a StreamWord to StreamData for a given channel.
    ///
    /// This bridges the stream_io module's `StreamWord` (used by stream switches)
    /// with the engine's `StreamData` (which tracks channel ownership).
    pub fn stream_word_to_data(word: super::stream_io::StreamWord, channel: u8) -> StreamData {
        StreamData {
            data: word.data,
            tlast: word.tlast,
            channel,
        }
    }

    /// Convert a StreamData to StreamWord.
    ///
    /// Drops the channel information (stream switches don't track channel).
    /// Parity is computed from the data.
    pub fn stream_data_to_word(data: &StreamData) -> super::stream_io::StreamWord {
        super::stream_io::StreamWord {
            data: data.data,
            tlast: data.tlast,
            parity: super::stream_io::StreamWord::compute_parity(data.data),
        }
    }

    /// Pop stream output as StreamWord (for stream switch integration).
    ///
    /// Returns the word and the channel it came from.
    pub fn pop_stream_out_as_word(&mut self) -> Option<(super::stream_io::StreamWord, u8)> {
        self.stream_out.pop_front().map(|data| {
            (Self::stream_data_to_word(&data), data.channel)
        })
    }

    /// Push stream input from StreamWord (for stream switch integration).
    ///
    /// Requires specifying which S2MM channel should receive this data.
    pub fn push_stream_in_from_word(&mut self, word: super::stream_io::StreamWord, channel: u8) -> bool {
        self.push_stream_in(Self::stream_word_to_data(word, channel))
    }
}

// ============================================================================
// StreamData <-> StreamWord Conversions
// ============================================================================

impl From<StreamData> for super::stream_io::StreamWord {
    fn from(data: StreamData) -> Self {
        Self {
            data: data.data,
            tlast: data.tlast,
            parity: Self::compute_parity(data.data),
        }
    }
}

impl StreamData {
    /// Create StreamData from a StreamWord with specified channel.
    pub fn from_stream_word(word: super::stream_io::StreamWord, channel: u8) -> Self {
        Self {
            data: word.data,
            tlast: word.tlast,
            channel,
        }
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
        assert!(engine.tile_type.is_mem_tile());
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

        // Write a known pattern to tile data memory at the source address
        let source_data: Vec<u8> = (0..32u8).collect();
        tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&source_data);

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

        // Verify the source data is still intact (DMA read shouldn't modify it)
        assert_eq!(&tile.data_memory()[0x100..0x100 + 32], &source_data[..],
            "Source data should remain intact after MM2S read");
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

        // Set lock to available state (value=1 for acq_eq mode)
        tile.locks[5].set(1);

        // Configure BD with lock using MM2S channel
        // Per AMD spec: acquire_value=1 means acq_eq (wait for lock==1, then decrement)
        // Per AMD spec: release_value=1 means add +1 to lock after transfer
        // release_value=0 would mean NO release per AM025
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(5, 1)   // Wait for lock==1, decrement by 1 (1->0)
            .with_release(5, 1);  // After transfer, add +1 to lock (0->1)
        engine.configure_bd(0, bd).unwrap();

        // Start should trigger lock acquire on MM2S channel
        engine.start_channel(2, 0).unwrap();

        // Step until complete (cycle-accurate timing needs more cycles)
        // Snapshot-based lock arbitration requires begin_lock_cycle() each cycle
        let mut cycles = 0;
        while engine.channel_active(2) {
            tile.begin_lock_cycle();  // Snapshot lock values for this cycle
            engine.step(&mut tile, &mut host_mem);
            tile.end_lock_cycle();    // Apply accumulated deltas at end of cycle
            cycles += 1;
            if cycles > 500 {
                panic!("Transfer took too long: {} cycles", cycles);
            }
        }

        // Verify lock was released: started at 1, acquired (->0), released +1 (->1)
        assert_eq!(tile.locks[5].value, 1);
    }

    #[test]
    fn test_execute_1d_transfer() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Write a recognizable pattern to source memory
        let source_data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
        let dm = tile.data_memory_mut();
        dm[0x100..0x100 + 64].copy_from_slice(&source_data);

        // Use MM2S channel (channel 2) for testing
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();

        let cycles = engine.execute_1d_transfer(2, 0, &mut tile, &mut host_mem).unwrap();
        assert!(cycles > 0, "Transfer should take at least one cycle");

        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.bytes_transferred, 64);
        assert_eq!(stats.transfers_completed, 1);

        // Source data should still be intact
        assert_eq!(&tile.data_memory()[0x100..0x100 + 64], &source_data[..],
            "Source data should remain intact after 1D transfer");
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
    fn test_default_cycle_accurate_timing() {
        // Cycle-accurate timing is the default and only mode
        let engine = DmaEngine::new_compute_tile(1, 2);
        assert_eq!(engine.timing_config().bd_setup_cycles, 4);
        assert_eq!(engine.timing_config().words_per_cycle, 1);
    }

    #[test]
    fn test_cycle_accurate_transfer() {
        // Cycle-accurate timing is the default
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // Write recognizable data to source address
        let source_data: [u8; 16] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
                                      0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        let dm = tile.data_memory_mut();
        dm[0x100..0x100 + 16].copy_from_slice(&source_data);

        // Configure BD for 16 bytes (4 words) using MM2S channel
        // With AIE2 spec: 4 setup + 1 start + 5 mem latency + 4 data cycles = 14+ cycles
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

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

        // Cycle-accurate transfer should have timing overhead
        assert!(cycles >= 6, "Cycle-accurate transfer should have overhead, got {} cycles", cycles);

        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.bytes_transferred, 16);
        assert_eq!(stats.transfers_completed, 1);

        // Source data should be unchanged
        assert_eq!(&tile.data_memory()[0x100..0x100 + 16], &source_data[..],
            "Source data integrity after cycle-accurate transfer");
    }

    #[test]
    fn test_lock_timing_integration() {
        // Create engine with lock timing enabled (cycle-accurate is default)
        let mut engine = DmaEngine::new_compute_tile(1, 2)
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
        let _host_mem = make_host_memory();

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

    // === Task Queue Tests ===

    #[test]
    fn test_task_queue_enqueue() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();
        engine.configure_bd(1, BdConfig::simple_1d(0x200, 32)).unwrap();

        // Enqueue first task - should start immediately (channel was idle)
        assert!(engine.enqueue_task(2, 0, 0, false));
        assert_eq!(engine.task_queue_size(2), 0); // Task started, not queued

        // Enqueue second task - should be queued (channel is busy)
        assert!(engine.enqueue_task(2, 1, 0, true));
        assert_eq!(engine.task_queue_size(2), 1);
    }

    #[test]
    fn test_task_queue_overflow() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();

        // First task starts immediately
        assert!(engine.enqueue_task(2, 0, 0, false));

        // Queue 8 more tasks (fills the queue)
        for i in 0..MAX_TASK_QUEUE_DEPTH {
            assert!(engine.enqueue_task(2, 0, 0, false), "Task {} should enqueue", i);
        }

        // 9th task should fail (queue full)
        assert!(!engine.enqueue_task(2, 0, 0, false));

        // Overflow flag should be set
        assert!(engine.task_queue_overflow(2));

        // Clear the flag
        engine.clear_task_queue_overflow(2);
        assert!(!engine.task_queue_overflow(2));
    }

    #[test]
    fn test_task_queue_status_register() {
        let layout = &crate::device::regdb::device_reg_layout().memory_status;

        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();

        // Enqueue some tasks (first starts, rest are queued)
        engine.enqueue_task(2, 0, 0, false); // Starts immediately
        engine.enqueue_task(2, 0, 0, false); // Queued
        engine.enqueue_task(2, 0, 0, false); // Queued

        let status = engine.get_channel_status(2);

        // Task_Queue_Size should be 2
        assert_eq!(layout.task_queue_size.extract(status), 2);

        // Channel should be running
        assert!(layout.channel_running.extract_bool(status));
    }

    #[test]
    fn test_task_queue_multiple_tasks_complete() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();
        engine.configure_bd(1, BdConfig::simple_1d(0x200, 16)).unwrap();

        // Enqueue two tasks with token issue on the second
        engine.enqueue_task(2, 0, 0, false);
        engine.enqueue_task(2, 1, 0, true);

        // Run until all work is complete (including queued tasks)
        let mut cycles = 0;
        while engine.channel_has_pending_work(2) {
            engine.step(&mut tile, &mut host_mem);
            cycles += 1;
            if cycles > 100 {
                panic!("Tasks took too long");
            }
        }

        // Both transfers should have completed
        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.transfers_completed, 2);
        assert_eq!(stats.bytes_transferred, 32); // 16 + 16

        // Second task had enable_token_issue, so should have emitted a token
        assert!(engine.has_task_token());
        let token = engine.pop_task_token().unwrap();
        assert_eq!(token.channel, 2);
    }

    #[test]
    fn test_task_queue_with_repeat() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

        // Enqueue a task with repeat_count=2 (runs 3 times total)
        engine.enqueue_task(2, 0, 2, true);

        // Run until all work is complete (including repeats)
        let mut cycles = 0;
        while engine.channel_has_pending_work(2) {
            engine.step(&mut tile, &mut host_mem);
            cycles += 1;
            if cycles > 200 {
                panic!("Repeated task took too long");
            }
        }

        // Should have transferred 3 * 16 = 48 bytes
        let stats = engine.channel_stats(2).unwrap();
        assert_eq!(stats.bytes_transferred, 48);
        assert_eq!(stats.transfers_completed, 3);
    }

    // === Stream Port Integration Tests ===

    #[test]
    fn test_compute_tile_port_mappings() {
        let engine = DmaEngine::new_compute_tile(1, 2);

        // MM2S channels send to slave ports 1, 2
        assert_eq!(engine.mm2s_slave_port(0), 1);
        assert_eq!(engine.mm2s_slave_port(1), 2);

        // S2MM channels receive from master ports 1, 2
        assert_eq!(engine.s2mm_master_port(0), 1);
        assert_eq!(engine.s2mm_master_port(1), 2);

        // Channel counts
        assert_eq!(engine.s2mm_channel_count(), 2);
        assert_eq!(engine.mm2s_channel_count(), 2);
    }

    #[test]
    fn test_memtile_port_mappings() {
        let engine = DmaEngine::new_mem_tile(0, 1);

        // MemTile: DMA channels map directly to ports 0-5
        for ch in 0..6 {
            assert_eq!(engine.mm2s_slave_port(ch), ch);
            assert_eq!(engine.s2mm_master_port(ch), ch);
        }

        // Channel counts
        assert_eq!(engine.s2mm_channel_count(), 6);
        assert_eq!(engine.mm2s_channel_count(), 6);
    }

    #[test]
    fn test_stream_data_to_word_conversion() {
        use crate::device::dma::stream_io::StreamWord;

        let data = StreamData {
            data: 0x12345678,
            tlast: true,
            channel: 2,
        };

        // Convert to StreamWord
        let word: StreamWord = data.into();
        assert_eq!(word.data, 0x12345678);
        assert!(word.tlast);
        // Parity should be computed
        assert_eq!(word.parity, StreamWord::compute_parity(0x12345678));
    }

    #[test]
    fn test_stream_word_to_data_conversion() {
        use crate::device::dma::stream_io::StreamWord;

        let word = StreamWord {
            data: 0xDEADBEEF,
            tlast: false,
            parity: true,
        };

        // Convert to StreamData with channel
        let data = StreamData::from_stream_word(word, 3);
        assert_eq!(data.data, 0xDEADBEEF);
        assert!(!data.tlast);
        assert_eq!(data.channel, 3);
    }

    #[test]
    fn test_engine_stream_word_interface() {
        use crate::device::dma::stream_io::StreamWord;

        let mut engine = DmaEngine::new_compute_tile(1, 2);

        // Push via StreamWord interface
        let word = StreamWord::with_tlast(0xCAFEBABE);
        assert!(engine.push_stream_in_from_word(word, 0));

        // Verify it's in the buffer
        assert!(engine.has_stream_in_for_channel(0));
        assert_eq!(engine.stream_in_len(), 1);

        // Add stream output data directly
        engine.stream_out.push_back(StreamData {
            data: 0x11111111,
            tlast: false,
            channel: 2,
        });

        // Pop as StreamWord
        let (out_word, channel) = engine.pop_stream_out_as_word().unwrap();
        assert_eq!(out_word.data, 0x11111111);
        assert!(!out_word.tlast);
        assert_eq!(channel, 2);
    }

    // === Compression / Decompression Integration Tests ===

    #[test]
    fn test_mm2s_compression_sparse_data() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();

        // Write sparse data: only bytes 0, 3, 8 are non-zero
        tile.data_memory_mut()[0] = 5;
        tile.data_memory_mut()[3] = 3;
        tile.data_memory_mut()[8] = 7;

        // Enable compression on MM2S channel 2
        engine.set_channel_compression_config(2, true, false, false);

        let result = engine.transfer_mm2s(0, 32, 2, true, false, &tile);
        assert!(result);

        // mask + 1 data word (3 bytes + padding) = 2 stream words
        assert_eq!(engine.stream_out_len(), 2);

        let mask_word = engine.stream_out.pop_front().unwrap();
        assert_eq!(mask_word.data, (1 << 0) | (1 << 3) | (1 << 8));
        assert!(!mask_word.tlast);

        let data_word = engine.stream_out.pop_front().unwrap();
        assert_eq!(data_word.data, u32::from_le_bytes([5, 3, 7, 0]));
        assert!(data_word.tlast);
    }

    #[test]
    fn test_mm2s_compression_all_zeros() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();

        engine.set_channel_compression_config(2, true, false, false);

        let result = engine.transfer_mm2s(0, 32, 2, true, false, &tile);
        assert!(result);

        // All zeros: just mask word, no data
        assert_eq!(engine.stream_out_len(), 1);
        let mask_word = engine.stream_out.pop_front().unwrap();
        assert_eq!(mask_word.data, 0);
        assert!(mask_word.tlast);
    }

    #[test]
    fn test_s2mm_decompression_round_trip() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();

        tile.data_memory_mut()[0] = 42;
        tile.data_memory_mut()[15] = 128;
        tile.data_memory_mut()[31] = 255;

        // MM2S compress from offset 0
        engine.set_channel_compression_config(2, true, false, false);
        let result = engine.transfer_mm2s(0, 32, 2, true, false, &tile);
        assert!(result);

        // Route compressed data from stream_out to stream_in (channel 0)
        while let Some(sd) = engine.stream_out.pop_front() {
            engine.push_stream_in(StreamData {
                data: sd.data,
                tlast: sd.tlast,
                channel: 0,
            });
        }

        // S2MM decompress to offset 256
        engine.set_channel_compression_config(0, false, true, false);
        let result = engine.transfer_s2mm(256, 32, 0, &mut tile);
        assert!(result.success);
        assert_eq!(result.bytes_written, 32);

        let data = tile.data_memory();
        assert_eq!(data[256], 42);
        assert_eq!(data[256 + 15], 128);
        assert_eq!(data[256 + 31], 255);
        for i in 0..32 {
            if i != 0 && i != 15 && i != 31 {
                assert_eq!(data[256 + i], 0, "byte {} should be zero", i);
            }
        }
    }

    #[test]
    fn test_mm2s_no_compression_when_disabled() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();

        tile.data_memory_mut()[0] = 0xAA;
        tile.data_memory_mut()[1] = 0xBB;
        tile.data_memory_mut()[2] = 0xCC;
        tile.data_memory_mut()[3] = 0xDD;

        assert!(!engine.is_compression_enabled(2));

        let result = engine.transfer_mm2s(0, 4, 2, true, false, &tile);
        assert!(result);

        assert_eq!(engine.stream_out_len(), 1);
        let word = engine.stream_out.pop_front().unwrap();
        assert_eq!(word.data, u32::from_le_bytes([0xAA, 0xBB, 0xCC, 0xDD]));
        assert!(word.tlast);
    }

    #[test]
    fn test_s2mm_no_decompression_when_disabled() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();

        engine.push_stream_in(StreamData {
            data: 0xDEADBEEF,
            tlast: true,
            channel: 0,
        });

        assert!(!engine.is_decompression_enabled(0));

        let result = engine.transfer_s2mm(0, 4, 0, &mut tile);
        assert!(result.success);
        assert_eq!(result.bytes_written, 4);

        let data = tile.data_memory();
        assert_eq!(
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
            0xDEADBEEF,
        );
    }

    #[test]
    fn test_compression_multiple_blocks() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();

        tile.data_memory_mut()[0] = 0x11;
        tile.data_memory_mut()[32] = 0x22;

        engine.set_channel_compression_config(2, true, false, false);

        let result = engine.transfer_mm2s(0, 64, 2, true, false, &tile);
        assert!(result);

        // Block 0: mask + data = 2 words; Block 1: mask + data = 2 words
        assert_eq!(engine.stream_out_len(), 4);

        let w0 = engine.stream_out.pop_front().unwrap();
        assert_eq!(w0.data, 1u32);
        assert!(!w0.tlast);

        let w1 = engine.stream_out.pop_front().unwrap();
        assert_eq!(w1.data, u32::from_le_bytes([0x11, 0, 0, 0]));
        assert!(!w1.tlast);

        let w2 = engine.stream_out.pop_front().unwrap();
        assert_eq!(w2.data, 1u32);
        assert!(!w2.tlast);

        let w3 = engine.stream_out.pop_front().unwrap();
        assert_eq!(w3.data, u32::from_le_bytes([0x22, 0, 0, 0]));
        assert!(w3.tlast);
    }
}
