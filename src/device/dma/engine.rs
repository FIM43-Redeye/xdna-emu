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

/// Identifies which tile's locks a DMA lock operation targets.
///
/// MemTile DMA BDs use an 8-bit lock ID field addressing 192 entries
/// across three tiles (per mlir-aie getLockLocalBaseIndex):
///   - IDs   0- 63: West column MemTile (col-1) locks
///   - IDs  64-127: Own MemTile locks (local_id = lock_id - 64)
///   - IDs 128-191: East column MemTile (col+1) locks
///
/// Compute/shim tiles use a 4-bit field (0-15), always Own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockTarget {
    /// Lock on own tile (local lock index).
    Own(u8),
    /// Lock on west neighbor MemTile, col-1 (local lock index on that tile).
    West(u8),
    /// Lock on east neighbor MemTile, col+1 (local lock index on that tile).
    East(u8),
}

/// Provides mutable access to neighbor tiles for cross-tile lock operations.
///
/// Models the NPU interconnect that routes MemTile DMA lock accesses to
/// neighbor columns. Constructed by the array level using disjoint borrows.
/// Non-MemTile tiles pass `NeighborLocks::empty()`.
pub struct NeighborLocks<'a> {
    /// West neighbor MemTile (col-1), if it exists.
    pub west: Option<&'a mut Tile>,
    /// East neighbor MemTile (col+1), if it exists.
    pub east: Option<&'a mut Tile>,
}

impl NeighborLocks<'_> {
    /// Create an empty neighbor context (no cross-tile access).
    ///
    /// Used for compute/shim tiles where cross-tile lock access
    /// is not applicable.
    pub fn empty() -> NeighborLocks<'static> {
        NeighborLocks { west: None, east: None }
    }
}

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

// Task queue entry and token types are now in super::token.
// Re-export for backward compatibility.
pub use super::token::{TaskQueueEntry, TaskQueue, Token, TokenState, MAX_TASK_QUEUE_DEPTH};

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

/// Result of one cycle of data transfer within the Transferring FSM state.
enum TransferCycleResult {
    /// Data moved successfully (or nothing to do this cycle)
    Continue,
    /// S2MM stalled waiting for stream data
    Stalled,
    /// FoT TLAST received, finish early
    FotFinish,
    /// Transfer error (bad address, etc.)
    Error,
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

    /// Dirty flags for BDs written word-by-word (e.g., control packets).
    /// When true, the BD must be re-parsed from tile raw storage before use.
    bd_dirty: Vec<bool>,

    /// Per-channel state (FSM, task queue, stats, BD tracking).
    /// Replaces 12 parallel Vec<T> fields with one struct per channel.
    channels: Vec<ChannelContext>,

    /// Timing configuration (controls cycle-accuracy vs fast mode)
    timing_config: DmaTimingConfig,

    /// Stream output buffer (MM2S channels produce data here).
    /// Data is read from tile memory and queued for the stream router.
    stream_out: VecDeque<StreamData>,

    /// Per-channel stream input buffers (S2MM channels consume data from here).
    /// Each S2MM channel has its own FIFO, matching real hardware where each
    /// channel connects to a dedicated stream switch master port. This prevents
    /// one channel's traffic (e.g., trace) from blocking another (e.g., output).
    stream_in: Vec<VecDeque<StreamData>>,

    /// Task complete token output buffer.
    /// Tokens are emitted when tasks complete with Enable_Token_Issue set.
    task_tokens: TokenState,

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

    /// Number of locks per tile (16 for compute, 64 for MemTile, 0 for shim).
    /// Used by resolve_lock_id() for cross-tile lock addressing.
    num_locks: u8,

    /// Number of S2MM channels (stream-to-memory) for this tile.
    /// Derived from architecture configuration at construction.
    s2mm_count: usize,

    /// Number of MM2S channels (memory-to-stream) for this tile.
    /// Derived from architecture configuration at construction.
    mm2s_count: usize,

    /// Number of memory banks for this tile type (8 for compute, 16 for MemTile).
    /// Used to compute bank indices for conflict detection.
    num_banks: usize,

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

    /// Create a new DMA engine for a compute tile (NPU1/AIE2 defaults).
    ///
    /// Uses hardcoded channel/BD counts matching Phoenix/HawkPoint.
    /// Production code should use `new()` with ArchConfig-derived values.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::Compute, 2, 2, 16, 16)
    }

    /// Create a new DMA engine for a memory tile (NPU1/AIE2 defaults).
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        Self::new(col, row, TileType::MemTile, 6, 6, 48, 64)
    }

    /// Create a new DMA engine for a shim tile (NPU1/AIE2 defaults).
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
        self.channels.len()
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
    fn per_direction_channel(&self, channel: ChannelId) -> u8 {
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
    fn check_memtile_bd_channel_validity(&self, bd_index: u8, channel: ChannelId) -> bool {
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
    pub fn submit_lock_requests(&self, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>) {
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
        neighbors: &mut NeighborLocks<'_>,
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

    /// Submit a release request to the appropriate tile's arbiter.
    fn submit_release_request(
        &self,
        lock_id: u8,
        release_value: i8,
        tile: &mut Tile,
        neighbors: &mut NeighborLocks<'_>,
        ch_idx: u8,
    ) {
        use crate::device::tile::LockRequest;

        let lock_target = match Self::resolve_lock_id_static(
            self.tile_type, self.col, self.row, self.num_locks, lock_id,
        ) {
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

    /// Resolve a BD lock ID to a local lock index on the own tile.
    ///
    /// MemTile DMA BDs use an 8-bit lock ID field that addresses 192 locks
    /// across three tiles (per mlir-aie getLockLocalBaseIndex):
    ///   - IDs   0- 63: West column MemTile (col-1) locks
    ///   - IDs  64-127: Own MemTile locks (local_id = lock_id - 64)
    ///   - IDs 128-191: East column MemTile (col+1) locks
    ///
    /// Compute and shim tile DMA BDs use a 4-bit field (0-15), always local.
    ///
    /// Returns `Some(local_lock_index)` for own-tile locks, `None` for
    /// cross-tile locks (which require neighbor tile access).
    /// Resolve a BD lock ID to a local lock index.
    ///
    /// For MemTile, maps the 8-bit cross-tile lock address space to local
    /// lock indices. For compute/shim, passes through directly.
    fn resolve_lock_id(&self, lock_id: u8) -> Option<LockTarget> {
        Self::resolve_lock_id_static(self.tile_type, self.col, self.row, self.num_locks, lock_id)
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
    pub fn resolve_lock_id_static(tile_type: TileType, col: u8, row: u8, num_locks: u8, lock_id: u8) -> Option<LockTarget> {
        if !tile_type.is_mem_tile() {
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
                col, row, lock_id, num_locks as u16 * 3,
            );
            log::error!("{}", msg);
            // Cannot push to fatal_errors from static method; caller must handle None.
            None
        }
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
        let mut transfer = Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row, self.tile_type)?;

        log::info!("DMA tile({},{}) ch{} BD{} start: total_bytes={} base_addr=0x{:X} next_bd={:?} acq_lock={:?}(val={}) rel_lock={:?}(val={}) pkt={}(id={}) dir={:?}",
            self.col, self.row, channel, bd_index,
            transfer.total_bytes, bd_config.base_addr,
            bd_config.next_bd,
            bd_config.acquire_lock, bd_config.acquire_value,
            bd_config.release_lock, bd_config.release_value,
            transfer.enable_packet, transfer.packet_id, direction);

        // Insert packet header before any FSM state transitions. On real
        // hardware the DMA prepends the header when the MM2S transfer begins,
        // regardless of whether lock acquisition is needed first.
        self.maybe_insert_packet_header_from_transfer(&mut transfer);

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

    /// Step the DMA engine by one cycle.
    ///
    /// This processes all active channels, moving data between memory and streams.
    /// Returns the overall result of the step.
    pub fn step(&mut self, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>, host_memory: &mut HostMemory) -> DmaResult {
        let mut any_active = false;
        let mut any_waiting = false;

        for ch_idx in 0..self.channels.len() {
            let phase_before = self.channels[ch_idx].fsm.phase_name();

            match &self.channels[ch_idx].fsm {
                ChannelFsm::Idle => {
                    // Check for queued BD from chaining or task queue
                    if let Some(next_bd) = self.channels[ch_idx].queued_bd.take() {
                        let repeat_count = self.channels[ch_idx].repeat_count as u8;
                        log::debug!("DMA tile({},{}) ch{} starting queued BD {} (repeat={})",
                            self.col, self.row, ch_idx, next_bd, repeat_count);
                        if let Err(e) = self.start_channel_with_repeat(ch_idx as u8, next_bd, repeat_count) {
                            log::warn!("DMA tile({},{}) ch{} failed to start BD {}: {:?}",
                                self.col, self.row, ch_idx, next_bd, e);
                            self.channels[ch_idx].fsm = ChannelFsm::Error;
                        } else {
                            any_active = true;
                        }
                    }
                }
                ChannelFsm::Paused { .. } | ChannelFsm::Error => {}
                _ => {
                    // Active channel -- run one FSM cycle
                    self.step_channel_fsm(ch_idx, tile, neighbors, host_memory);
                    if matches!(self.channels[ch_idx].fsm, ChannelFsm::AcquiringLock { acquired: false, .. }) {
                        any_waiting = true;
                    } else {
                        any_active = true;
                    }
                }
            }

            // Log transitions
            let phase_after = self.channels[ch_idx].fsm.phase_name();
            if phase_before != phase_after {
                log::info!("DMA({},{}) ch{}: {} -> {}",
                    self.col, self.row, ch_idx, phase_before, phase_after);
            }
        }

        if any_active {
            DmaResult::InProgress
        } else if any_waiting {
            DmaResult::WaitingForLock(0)
        } else {
            DmaResult::Complete
        }
    }

    /// Step a single channel through one cycle of its unified FSM.
    ///
    /// Each match arm does ONE cycle of work and optionally transitions to
    /// a new state. This replaces the old step_channel() + step_channel_timed() +
    /// complete_transfer() + finish_complete_transfer() chain.
    fn step_channel_fsm(&mut self, ch_idx: usize, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>, host_memory: &mut HostMemory) {
        // Take the FSM out temporarily so we can match on it while
        // mutating self (for stream buffers, do_transfer, etc.)
        let fsm = std::mem::take(&mut self.channels[ch_idx].fsm);

        let new_fsm = match fsm {
            ChannelFsm::BdSetup { cycles_remaining, mut transfer } => {
                if cycles_remaining <= 1 {
                    // BD setup done. Insert packet header if needed.
                    self.maybe_insert_packet_header_from_transfer(&mut transfer);
                    // Check if lock acquisition is needed
                    if let Some(lock_id) = transfer.acquire_lock {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining: self.timing_config.lock_acquire_cycles as u16,
                            acquired: false,
                            transfer,
                        }
                    } else {
                        ChannelFsm::MemoryLatency {
                            cycles_remaining: self.timing_config.memory_latency_cycles as u16,
                            transfer,
                        }
                    }
                } else {
                    ChannelFsm::BdSetup {
                        cycles_remaining: cycles_remaining - 1,
                        transfer,
                    }
                }
            }

            ChannelFsm::AcquiringLock { lock_id, cycles_remaining, acquired, transfer } => {
                if acquired {
                    // Lock already acquired, counting down latency
                    if cycles_remaining <= 1 {
                        ChannelFsm::MemoryLatency {
                            cycles_remaining: self.timing_config.memory_latency_cycles as u16,
                            transfer,
                        }
                    } else {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining: cycles_remaining - 1,
                            acquired: true,
                            transfer,
                        }
                    }
                } else {
                    // Check if arbiter granted the acquire (submitted in pre-step pass)
                    if self.check_acquire_granted(lock_id, tile, neighbors, ch_idx) {
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining,
                            acquired: true,
                            transfer,
                        }
                    } else {
                        self.channels[ch_idx].stats.lock_wait_cycles += 1;
                        self.trace(EventType::DmaStalledLock { channel: ch_idx as u8 });
                        ChannelFsm::AcquiringLock {
                            lock_id,
                            cycles_remaining,
                            acquired: false,
                            transfer,
                        }
                    }
                }
            }

            ChannelFsm::MemoryLatency { cycles_remaining, transfer } => {
                if cycles_remaining <= 1 {
                    // For shim tiles accessing host DDR, add NoC+DDR pipeline
                    // latency. This is the extra time for the first word to
                    // traverse the NoC to DDR and back. Once the pipeline
                    // fills, throughput is 1 word/cycle (same as tile memory).
                    let host_lat = self.timing_config.host_memory_latency_cycles;
                    if host_lat > 0 && transfer.involves_host_memory() {
                        ChannelFsm::HostPipelineLatency {
                            cycles_remaining: host_lat,
                            transfer,
                        }
                    } else {
                        ChannelFsm::Transferring { transfer }
                    }
                } else {
                    ChannelFsm::MemoryLatency {
                        cycles_remaining: cycles_remaining - 1,
                        transfer,
                    }
                }
            }

            ChannelFsm::HostPipelineLatency { cycles_remaining, transfer } => {
                if cycles_remaining <= 1 {
                    ChannelFsm::Transferring { transfer }
                } else {
                    ChannelFsm::HostPipelineLatency {
                        cycles_remaining: cycles_remaining - 1,
                        transfer,
                    }
                }
            }

            ChannelFsm::Transferring { mut transfer } => {
                // Move one cycle of data
                let result = self.do_transfer_cycle(ch_idx, &mut transfer, tile, host_memory);

                match result {
                    TransferCycleResult::Continue => {
                        // Tick the transfer's cycle counter
                        transfer.tick();
                        // Check if data movement is complete
                        if transfer.remaining_bytes() == 0 {
                            self.begin_completion(ch_idx, transfer)
                        } else {
                            ChannelFsm::Transferring { transfer }
                        }
                    }
                    TransferCycleResult::Stalled => {
                        // S2MM stall: stay in Transferring, don't advance
                        self.trace(EventType::DmaStreamStarvation { channel: ch_idx as u8 });
                        ChannelFsm::Transferring { transfer }
                    }
                    TransferCycleResult::FotFinish => {
                        // Early finish on TLAST (FoT mode)
                        self.begin_completion(ch_idx, transfer)
                    }
                    TransferCycleResult::Error => {
                        ChannelFsm::Error
                    }
                }
            }

            ChannelFsm::ReleasingLock { lock_id, release_value, cycles_remaining, completion } => {
                if cycles_remaining <= 1 {
                    // Execute the lock release
                    self.execute_lock_release(lock_id, release_value, tile, neighbors);
                    // Update stats and handle chaining/repeat
                    self.after_transfer_done(ch_idx, completion)
                } else {
                    ChannelFsm::ReleasingLock {
                        lock_id,
                        release_value,
                        cycles_remaining: cycles_remaining - 1,
                        completion,
                    }
                }
            }

            ChannelFsm::BdChaining { cycles_remaining, next_bd } => {
                if cycles_remaining <= 1 {
                    // Load next BD and start transfer
                    let bd_addr = self.bd_configs.get(next_bd as usize).map(|c| c.base_addr).unwrap_or(0);
                    log::info!("DMA tile({},{}) ch{} BD chain -> BD{} (base_addr=0x{:X})", self.col, self.row, ch_idx, next_bd, bd_addr);
                    match self.create_transfer_from_bd(next_bd, ch_idx as u8) {
                        Ok(transfer) => {
                            self.channels[ch_idx].current_bd = Some(next_bd);
                            ChannelFsm::BdSetup {
                                cycles_remaining: self.timing_config.bd_setup_cycles as u16,
                                transfer: Box::new(transfer),
                            }
                        }
                        Err(e) => {
                            log::warn!("DMA tile({},{}) ch{} BD chain to {} failed: {:?}",
                                self.col, self.row, ch_idx, next_bd, e);
                            ChannelFsm::Error
                        }
                    }
                } else {
                    ChannelFsm::BdChaining {
                        cycles_remaining: cycles_remaining - 1,
                        next_bd,
                    }
                }
            }

            // Idle/Paused/Error handled in step(), not here
            other => other,
        };

        self.channels[ch_idx].fsm = new_fsm;
    }

    /// Create a Transfer from a BD index without starting a channel.
    /// Used by BdChaining to load the next BD in a chain.
    fn create_transfer_from_bd(&self, bd_index: u8, channel: u8) -> Result<Transfer, DmaError> {
        if bd_index as usize >= self.bd_configs.len() {
            return Err(DmaError::InvalidBd(bd_index));
        }
        let bd_config = &self.bd_configs[bd_index as usize];
        let direction = match self.channel_type(channel) {
            ChannelType::S2MM => TransferDirection::S2MM,
            ChannelType::MM2S => TransferDirection::MM2S,
        };
        Transfer::new(bd_config, bd_index, channel, direction, self.col, self.row, self.tile_type)
    }

    /// Begin the completion sequence after data movement is done.
    ///
    /// If the BD has a release lock, transitions to ReleasingLock.
    /// Otherwise, goes directly to chaining/repeat/idle via after_transfer_done.
    fn begin_completion(&mut self, ch_idx: usize, transfer: Box<Transfer>) -> ChannelFsm {
        let completion = CompletionInfo {
            bd_index: transfer.bd_index,
            next_bd: transfer.next_bd,
            cycles_elapsed: transfer.cycles_elapsed,
            channel: ch_idx as u8,
        };

        if let Some(lock_id) = transfer.release_lock {
            let release_value = transfer.release_value;
            ChannelFsm::ReleasingLock {
                lock_id,
                release_value,
                cycles_remaining: self.timing_config.lock_release_cycles as u16,
                completion,
            }
        } else {
            self.after_transfer_done(ch_idx, completion)
        }
    }

    /// Handle post-transfer completion: stats, chaining, repeat, task queue.
    ///
    /// Returns the next FSM state (BdChaining, Idle, or starts next task).
    fn after_transfer_done(&mut self, ch_idx: usize, completion: CompletionInfo) -> ChannelFsm {
        self.channels[ch_idx].stats.transfers_completed += 1;
        self.channels[ch_idx].stats.cycles_spent += completion.cycles_elapsed;

        // Emit DMA_FINISHED_BD
        self.trace_events.push((self.current_cycle, EventType::DmaFinishedBd { channel: ch_idx as u8 }));

        // Check for BD chaining
        if let Some(next_bd) = completion.next_bd {
            log::debug!("DMA tile({},{}) ch{} chaining to BD {} (from BD {})",
                self.col, self.row, ch_idx, next_bd, completion.bd_index);
            return ChannelFsm::BdChaining {
                cycles_remaining: self.timing_config.bd_chain_cycles as u16,
                next_bd,
            };
        }

        // Check for repeat count
        if self.channels[ch_idx].repeat_count > 0 {
            self.channels[ch_idx].repeat_count -= 1;
            if let Some(start_bd) = self.channels[ch_idx].chain_start_bd {
                log::debug!("DMA tile({},{}) ch{} repeating chain from BD {} ({} remaining)",
                    self.col, self.row, ch_idx, start_bd, self.channels[ch_idx].repeat_count);
                return ChannelFsm::BdChaining {
                    cycles_remaining: self.timing_config.bd_chain_cycles as u16,
                    next_bd: start_bd,
                };
            }
        }

        // Task complete (no chaining, no repeats)
        self.trace(EventType::DmaFinishedTask { channel: ch_idx as u8 });
        self.maybe_emit_task_token(ch_idx);

        // Check for more tasks in the queue
        if !self.channels[ch_idx].task_queue.is_empty() {
            log::debug!("DMA tile({},{}) ch{} task complete, {} tasks remaining in queue",
                self.col, self.row, ch_idx, self.channels[ch_idx].task_queue.len());
            self.start_next_queued_task(ch_idx as u8);
            // start_next_queued_task sets the FSM, return what it set
            return std::mem::take(&mut self.channels[ch_idx].fsm);
        }

        ChannelFsm::Idle
    }

    /// Perform one cycle of data transfer for a channel in the Transferring state.
    ///
    /// Extracts transfer parameters, calls do_transfer, and advances the transfer.
    /// The Transfer is borrowed from the FSM via the caller (not from self.channels).
    fn do_transfer_cycle(
        &mut self,
        ch_idx: usize,
        transfer: &mut Transfer,
        tile: &mut Tile,
        host_memory: &mut HostMemory,
    ) -> TransferCycleResult {
        let words_per_cycle = self.timing_config.words_per_cycle as usize;
        let bytes_per_cycle = words_per_cycle * 4;

        if transfer.has_zero_padding() {
            // Padding-aware path: one word at a time
            let remaining = transfer.remaining_bytes();
            if remaining == 0 {
                return TransferCycleResult::Continue;
            }

            let action = transfer.next_output_action();
            let is_last_word = remaining <= 4;
            let channel = transfer.channel;
            let tlast_suppress = transfer.tlast_suppress;

            match action {
                PadAction::Zero => {
                    let should_assert_tlast = is_last_word && !tlast_suppress;
                    self.stream_out.push_back(StreamData {
                        data: 0,
                        tlast: should_assert_tlast,
                        channel,
                    });
                    transfer.advance(4);
                    self.channels[ch_idx].stats.bytes_transferred += 4;
                    TransferCycleResult::Continue
                }
                PadAction::Data(addr) => {
                    let source = transfer.source;
                    let dest = transfer.dest;
                    let result = self.do_transfer(
                        source, dest, addr, 4, channel,
                        is_last_word, tlast_suppress, tile, host_memory,
                    );

                    if result.stall {
                        return TransferCycleResult::Stalled;
                    }
                    if result.success {
                        transfer.advance(4);
                        self.channels[ch_idx].stats.bytes_transferred += 4;
                        if result.fot_finish {
                            TransferCycleResult::FotFinish
                        } else {
                            TransferCycleResult::Continue
                        }
                    } else {
                        TransferCycleResult::Error
                    }
                }
            }
        } else {
            // Standard path: transfer bytes_per_cycle at once
            let bytes_to_transfer = bytes_per_cycle.min(transfer.remaining_bytes() as usize);
            if bytes_to_transfer == 0 {
                return TransferCycleResult::Continue;
            }

            let remaining_after = transfer.remaining_bytes().saturating_sub(bytes_to_transfer as u64);
            let addr = transfer.current_address();
            let source = transfer.source;
            let dest = transfer.dest;
            let channel = transfer.channel;
            let is_last = remaining_after == 0;
            let tlast_suppress = transfer.tlast_suppress;

            let result = self.do_transfer(
                source, dest, addr, bytes_to_transfer, channel,
                is_last, tlast_suppress, tile, host_memory,
            );

            if result.stall {
                return TransferCycleResult::Stalled;
            }
            if result.success {
                transfer.advance(bytes_to_transfer as u64);
                self.channels[ch_idx].stats.bytes_transferred += bytes_to_transfer as u64;
                if result.fot_finish {
                    TransferCycleResult::FotFinish
                } else {
                    TransferCycleResult::Continue
                }
            } else {
                TransferCycleResult::Error
            }
        }
    }

    /// Check if a lock acquire was granted by the arbiter.
    ///
    /// The request was submitted in `submit_lock_requests()` before stepping.
    /// This checks whether the arbiter granted it during resolution.
    fn check_acquire_granted(
        &mut self,
        lock_id: u8,
        tile: &Tile,
        neighbors: &NeighborLocks<'_>,
        ch_idx: usize,
    ) -> bool {
        let lock_target = match self.resolve_lock_id(lock_id) {
            Some(target) => target,
            None => return false,
        };

        let (target_tile, local_id): (&Tile, u8) = match lock_target {
            LockTarget::Own(id) => (tile, id),
            LockTarget::West(id) => match neighbors.west.as_deref() {
                Some(west) => (west, id),
                None => return false,
            },
            LockTarget::East(id) => match neighbors.east.as_deref() {
                Some(east) => (east, id),
                None => return false,
            },
        };

        let requestor = self.channel_requestor(ch_idx as u8);
        let granted = target_tile.lock_was_granted(requestor, local_id as usize);

        log::info!("DMA check_acquire_granted tile({},{}) ch{} bd_lock={} target={:?} local_lock={} granted={}",
            self.col, self.row, ch_idx, lock_id, lock_target, local_id, granted);

        if let Some(ref mut timing) = self.lock_timing {
            timing.track_acquire(local_id as usize, granted);
        }

        granted
    }

    /// Execute a lock release operation (post-arbiter).
    ///
    /// The release was already submitted to the arbiter in `submit_lock_requests()`
    /// and resolved before `step()`. This method just logs for debugging.
    /// The lock value was already updated by the arbiter's `resolve()`.
    fn execute_lock_release(
        &mut self,
        lock_id: u8,
        release_value: i8,
        _tile: &mut Tile,
        _neighbors: &mut NeighborLocks<'_>,
    ) {
        log::info!("DMA tile({},{}) lock release bd_lock={} delta={} (applied by arbiter)",
            self.col, self.row, lock_id, release_value);
    }

    /// Insert a packet header from a Transfer reference (used during BdSetup).
    ///
    /// Unlike the old maybe_insert_packet_header which accessed self.transfers[],
    /// this takes a Transfer reference directly from the FSM.
    ///
    /// Takes `&mut Transfer` so it can call `mark_packet_header_sent()` after
    /// successful insertion. This prevents double-insertion when the same
    /// Transfer passes through both `start_channel_with_repeat` and `BdSetup`
    /// completion (the no-lock path).
    fn maybe_insert_packet_header_from_transfer(&mut self, transfer: &mut Transfer) {
        if !transfer.needs_packet_header() || transfer.direction != TransferDirection::MM2S {
            if transfer.enable_packet && transfer.direction == TransferDirection::MM2S {
                log::warn!("DMA({},{}) ch{} BD{}: enable_packet=true but needs_packet_header()={} header_sent={}",
                    self.col, self.row, transfer.channel, transfer.bd_index,
                    transfer.needs_packet_header(), transfer.packet_header_sent);
            }
            return;
        }

        if let Some(header_word) = transfer.generate_packet_header() {
            self.stream_out.push_back(StreamData {
                data: header_word,
                tlast: false,
                channel: transfer.channel,
            });
            transfer.mark_packet_header_sent();
            let (hdr, _) = crate::device::stream_switch::PacketHeader::decode(header_word);
            log::info!("DMA({},{}) ch{} BD{} packet header: 0x{:08X} (pkt_id={}, type={:?})",
                self.col, self.row, transfer.channel, transfer.bd_index,
                header_word, hdr.stream_id, hdr.packet_type);
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
        self.cycle_dma_banks |= crate::device::banking::banks_for_access(
            offset as u32, bytes, self.num_banks,
        );

        if offset + bytes > mem_size {
            let msg = format!(
                "DMA({},{}) MM2S addr=0x{:X} bytes={} wraps past memory end (size=0x{:X}) -- bus error",
                self.col, self.row, addr, bytes, mem_size,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
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
                    let msg = format!(
                        "DMA({},{}) MM2S ch{} compression failed at block {} -- data corruption",
                        self.col, self.row, channel, block_idx,
                    );
                    log::error!("{}", msg);
                    self.fatal_errors.push(msg);
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
        self.cycle_dma_banks |= crate::device::banking::banks_for_access(
            offset as u32, bytes, self.num_banks,
        );

        if offset + bytes > mem_size {
            let msg = format!(
                "DMA({},{}) S2MM addr=0x{:X} bytes={} wraps past memory end (size=0x{:X}) -- bus error",
                self.col, self.row, addr, bytes, mem_size,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
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
                    let msg = format!(
                        "DMA({},{}) S2MM ch{} decompression failed (mask=0x{:08X}, buf_len={}) -- data corruption",
                        self.col, self.row, channel, mask, compressed_buf.len(),
                    );
                    log::error!("{}", msg);
                    self.fatal_errors.push(msg);
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

    // NOTE: complete_transfer() and finish_complete_transfer() have been
    // replaced by begin_completion() and after_transfer_done() in the FSM.

    /// Emit a task complete token if Enable_Token_Issue is set for this channel.
    fn maybe_emit_task_token(&mut self, ch_idx: usize) {
        let config = &self.channels[ch_idx].task_config;

        if config.enable_token_issue {
            log::debug!(
                "DMA tile({},{}) ch{} emitting task complete token (controller_id={})",
                self.col, self.row, ch_idx, config.controller_id
            );

            self.task_tokens.issue(ch_idx as u8, config.controller_id);

            // Clear enable_token_issue after issuing (it's set per-task via Start_Queue)
            self.channels[ch_idx].task_config.enable_token_issue = false;
        }
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

    // === Packet Header Insertion (MM2S only) ===

    // NOTE: maybe_insert_packet_header() replaced by maybe_insert_packet_header_from_transfer()
    // which takes a Transfer reference directly from the FSM.

    // === Stream Interface for TileArray ===

    /// Pop a word from the stream output buffer (MM2S produced data).
    ///
    /// Returns None if no data is available.
    pub fn pop_stream_out(&mut self) -> Option<StreamData> {
        self.stream_out.pop_front()
    }

    /// Peek at the next word in stream output buffer without removing it.
    pub fn peek_stream_out(&self) -> Option<&StreamData> {
        self.stream_out.front()
    }

    /// Push a word to the per-channel stream input buffer (for S2MM to consume).
    ///
    /// Each S2MM channel has its own FIFO with independent capacity, matching
    /// real hardware where each channel connects to a dedicated stream switch
    /// master port. Returns true if successful, false if channel buffer is full.
    pub fn push_stream_in(&mut self, data: StreamData) -> bool {
        let ch = data.channel as usize;
        if ch >= self.stream_in.len() {
            let msg = format!(
                "DMA({},{}) push_stream_in: channel {} out of range (have {} S2MM channels)",
                self.col, self.row, ch, self.stream_in.len()
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            return false;
        }
        if self.stream_in[ch].len() < 256 {
            self.stream_in[ch].push_back(data);
            true
        } else {
            let msg = format!(
                "DMA({},{}) stream_in buffer full (256), dropping ch{} data: 0x{:08X} -- \
                 backpressure violation",
                self.col, self.row, data.channel, data.data,
            );
            log::error!("{}", msg);
            self.fatal_errors.push(msg);
            false
        }
    }

    /// Check if stream output buffer has data.
    pub fn has_stream_out(&self) -> bool {
        !self.stream_out.is_empty()
    }

    /// Check if any S2MM channel's stream input buffer has space.
    ///
    /// Returns true if at least one channel can accept data. Callers that
    /// know the target channel should use `can_accept_stream_in_for_channel`
    /// for precise per-channel backpressure.
    pub fn can_accept_stream_in(&self) -> bool {
        self.stream_in.iter().any(|q| q.len() < 256)
    }

    /// Check if a specific S2MM channel's stream input buffer has space.
    pub fn can_accept_stream_in_for_channel(&self, channel: u8) -> bool {
        self.stream_in.get(channel as usize).map_or(false, |q| q.len() < 256)
    }

    /// Get the number of words in stream output buffer.
    pub fn stream_out_len(&self) -> usize {
        self.stream_out.len()
    }

    /// Get the total number of words across all stream input channel buffers.
    pub fn stream_in_len(&self) -> usize {
        self.stream_in.iter().map(|q| q.len()).sum()
    }

    /// Check if stream input buffer has data for a specific channel.
    pub fn has_stream_in_for_channel(&self, channel: u8) -> bool {
        self.stream_in.get(channel as usize).map_or(false, |q| !q.is_empty())
    }

    /// Pop data from a specific channel's stream input buffer.
    ///
    /// Each channel has its own FIFO, so this is O(1) (front pop).
    fn pop_stream_in_for_channel(&mut self, channel: u8) -> Option<StreamData> {
        self.stream_in.get_mut(channel as usize)?.pop_front()
    }

    /// Check if any S2MM channel needs to receive stream data.
    ///
    /// Returns Some(channel) if a channel needs data, None otherwise.
    pub fn s2mm_needs_data(&self) -> Option<ChannelId> {
        for ch_idx in 0..self.channels.len() {
            if self.channel_type(ch_idx as u8) == ChannelType::S2MM {
                let is_pending = self.channels[ch_idx].is_active();
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
        if ch_idx >= self.channels.len() {
            return false;
        }

        let ch = &mut self.channels[ch_idx];

        // Push to task queue (handles overflow flag internally)
        let entry = TaskQueueEntry::new(start_bd, repeat_count, enable_token_issue);
        if ch.task_queue.push(entry).is_err() {
            log::trace!(
                "DMA tile({},{}) ch{} task queue full (BD {} rejected, queue_len={})",
                self.col, self.row, channel, start_bd, ch.task_queue.len()
            );
            return false;
        }

        log::debug!(
            "DMA tile({},{}) ch{} enqueued task: BD={} repeat={} token={} (queue_size={})",
            self.col, self.row, channel, start_bd, repeat_count, enable_token_issue,
            ch.task_queue.len()
        );

        // If channel is idle, start processing the queue
        if matches!(ch.fsm, ChannelFsm::Idle) {
            self.start_next_queued_task(channel);
        }

        true
    }

    /// Start the next task from the channel's queue.
    fn start_next_queued_task(&mut self, channel: u8) {
        let ch_idx = channel as usize;
        if ch_idx >= self.channels.len() {
            return;
        }

        let task = match self.channels[ch_idx].task_queue.pop() {
            Some(t) => t,
            None => return,
        };

        self.channels[ch_idx].task_config.enable_token_issue = task.enable_token_issue;

        if task.start_bd as usize >= self.bd_configs.len() {
            log::error!(
                "DMA tile({},{}) ch{} queued task has invalid BD {} (max={})",
                self.col, self.row, channel, task.start_bd, self.bd_configs.len()
            );
            self.channels[ch_idx].fsm = ChannelFsm::Error;
            return;
        }

        log::debug!(
            "DMA tile({},{}) ch{} starting queued task: BD={} repeat={} (remaining={})",
            self.col, self.row, channel, task.start_bd, task.repeat_count,
            self.channels[ch_idx].task_queue.len()
        );

        // BD is read from registers at execution time (not snapshotted at enqueue).
        // Per AM025: Start_Queue only stores BD_ID; the hardware fetches the BD
        // during the STARTING state transition.

        let start_bd = task.start_bd;
        if let Err(e) = self.start_channel_with_repeat(channel, start_bd, task.repeat_count) {
            log::error!(
                "DMA tile({},{}) ch{} failed to start queued task BD {}: {}",
                self.col, self.row, channel, start_bd, e
            );
            self.channels[ch_idx].fsm = ChannelFsm::Error;
        } else {
            self.channels[ch_idx].chain_start_bd = Some(start_bd);
        }
    }

    /// Get the current task queue size for a channel.
    pub fn task_queue_size(&self, channel: u8) -> usize {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_queue.len())
            .unwrap_or(0)
    }

    /// Check if the task queue overflow flag is set for a channel.
    pub fn task_queue_overflow(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_queue.has_overflow())
            .unwrap_or(false)
    }

    /// Clear the task queue overflow flag (write-to-clear per AM025).
    pub fn clear_task_queue_overflow(&mut self, channel: u8) {
        if let Some(ch) = self.channels.get_mut(channel as usize) {
            ch.task_queue.clear_overflow();
        }
    }

    /// Check if the BD unavailable error flag is set for a channel.
    pub fn error_bd_unavailable(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.error_bd_unavailable)
            .unwrap_or(false)
    }

    /// Set the BD unavailable error flag (S2MM OOO mode).
    pub fn set_error_bd_unavailable(&mut self, channel: u8) {
        if let Some(ch) = self.channels.get_mut(channel as usize) {
            ch.error_bd_unavailable = true;
            log::warn!(
                "DMA tile({},{}) S2MM ch{} Error_BD_Unavailable: invalid BD in OOO packet header",
                self.col, self.row, channel
            );
        }
    }

    /// Clear the BD unavailable error flag (write-to-clear per AM025).
    pub fn clear_error_bd_unavailable(&mut self, channel: u8) {
        if let Some(ch) = self.channels.get_mut(channel as usize) {
            ch.error_bd_unavailable = false;
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
        if let Some(ch) = self.channels.get_mut(ch_idx as usize) {
            ch.task_config.enable_token_issue = enable_token_issue;
            ch.task_config.controller_id = controller_id;
            ch.task_config.fot_mode = fot_mode;

            log::trace!(
                "DMA tile({},{}) ch{} set task config: token_issue={} controller_id={} fot_mode={}",
                self.col, self.row, ch_idx, enable_token_issue, controller_id, fot_mode
            );
        }
    }

    /// Set channel compression/decompression and out-of-order configuration.
    pub fn set_channel_compression_config(
        &mut self,
        ch_idx: u8,
        compression_enable: bool,
        decompression_enable: bool,
        out_of_order_enable: bool,
    ) {
        if let Some(ch) = self.channels.get_mut(ch_idx as usize) {
            ch.task_config.compression_enable = compression_enable;
            ch.task_config.decompression_enable = decompression_enable;
            ch.task_config.out_of_order_enable = out_of_order_enable;

            log::trace!(
                "DMA tile({},{}) ch{} set compression config: compress={} decompress={} ooo={}",
                self.col, self.row, ch_idx, compression_enable, decompression_enable, out_of_order_enable
            );
        }
    }

    /// Pop a task complete token from the output buffer.
    ///
    /// Returns None if no tokens are pending.
    pub fn pop_task_token(&mut self) -> Option<Token> {
        self.task_tokens.consume()
    }

    /// Check if any task complete tokens are pending.
    pub fn has_task_token(&self) -> bool {
        self.task_tokens.has_pending()
    }

    /// Get the number of pending task complete tokens.
    pub fn task_token_count(&self) -> usize {
        self.task_tokens.pending_count()
    }

    /// Get the FoT mode for a channel (S2MM only).
    pub fn get_channel_fot_mode(&self, ch_idx: u8) -> u8 {
        self.channels
            .get(ch_idx as usize)
            .map(|ch| ch.task_config.fot_mode)
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
        if ch_idx >= self.channels.len() {
            return 0;
        }

        let ch = &self.channels[ch_idx];
        let mut status: u32 = 0;

        // Cur_BD
        if let Some(bd_idx) = ch.current_bd {
            status = layout.cur_bd.insert(status, bd_idx as u32);
        }

        // Task_Queue_Size
        let queue_size = ch.task_queue.len() as u32;
        status = layout.task_queue_size.insert(status, queue_size);

        // Task_Queue_Overflow
        if ch.task_queue.has_overflow() {
            status = layout.task_queue_overflow.set_bit(status);
        }

        // Error_BD_Unavailable
        if ch.error_bd_unavailable {
            status = layout.error_bd_unavailable.set_bit(status);
        }

        // Derive external state from FSM
        match &ch.fsm {
            ChannelFsm::Idle => {}
            ChannelFsm::AcquiringLock { acquired: false, .. } => {
                // Stalled on lock acquire
                status = layout.status.insert(status, 0b10);
                status = layout.channel_running.set_bit(status);
                status = layout.stalled_lock_acq.set_bit(status);
            }
            ChannelFsm::Paused { .. } => {
                status = layout.channel_running.set_bit(status);
            }
            ChannelFsm::Error => {
                status = layout.error_bd_invalid.set_bit(status);
            }
            _ => {
                // All other active states
                status = layout.status.insert(status, 0b10);
                status = layout.channel_running.set_bit(status);
            }
        }

        // Check for stream stall (S2MM waiting for data in Transferring state)
        if let Some(transfer) = ch.fsm.transfer() {
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
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_config.compression_enable)
            .unwrap_or(false)
    }

    /// Get whether decompression is enabled for a channel (S2MM).
    pub fn is_decompression_enabled(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_config.decompression_enable)
            .unwrap_or(false)
    }

    /// Get whether out-of-order mode is enabled for a channel (S2MM).
    pub fn is_out_of_order_enabled(&self, channel: u8) -> bool {
        self.channels
            .get(channel as usize)
            .map(|ch| ch.task_config.out_of_order_enable)
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
            TileType::Shim => super::stream_io::shim::mm2s_slave_port(ch),
        }
    }

    /// Get the master port that S2MM channel `ch` receives data FROM.
    ///
    /// For compute tiles: ch0 <- master port 1, ch1 <- master port 2
    /// For memtiles: ch0-5 <- master ports 0-5
    /// For shim tiles: ch0 <- master port 2, ch1 <- master port 3 (South ports)
    pub fn s2mm_master_port(&self, ch: u8) -> u8 {
        match self.tile_type {
            TileType::Compute => super::stream_io::compute::s2mm_master_port(ch),
            TileType::MemTile => super::stream_io::memtile::s2mm_master_port(ch),
            TileType::Shim => super::stream_io::shim::s2mm_master_port(ch),
        }
    }

    /// Get the number of S2MM channels for this tile.
    pub fn s2mm_channel_count(&self) -> usize {
        self.s2mm_count
    }

    /// Get the number of MM2S channels for this tile.
    pub fn mm2s_channel_count(&self) -> usize {
        self.mm2s_count
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
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
            cycles += 1;
            if cycles > 100 {
                panic!("Transfer took too long");
            }
        }

        // Verify completion
        assert_eq!(engine.channel_state(2), ChannelState::Idle);

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
        // Arbiter-based lock arbitration: submit -> resolve -> step
        let mut cycles = 0;
        while engine.channel_active(2) {
            engine.submit_lock_requests(&mut tile, &mut NeighborLocks::empty());
            tile.resolve_lock_requests(0);
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
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

        let cycles = engine.execute_1d_transfer(2, 0, &mut tile, &mut NeighborLocks::empty(), &mut host_mem).unwrap();
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
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
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
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
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
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
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
        assert_eq!(token.channel_id, 2);
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
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
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

    /// Regression test: BD chain + repeat must restart from the FIRST BD,
    /// not the last one. Previously, `current_bds` was overwritten to the
    /// last chained BD, so repeat only re-executed the tail of the chain.
    #[test]
    fn test_bd_chain_with_repeat_restarts_from_start() {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        let mut tile = make_tile();
        let mut host_mem = make_host_memory();

        // BD0 (16 bytes at 0x100) chains to BD1 (16 bytes at 0x200)
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16).with_next(1)).unwrap();
        engine.configure_bd(1, BdConfig::simple_1d(0x200, 16)).unwrap();

        // Enqueue task: start at BD0, repeat_count=1 (run chain twice total)
        // Expected: BD0->BD1, BD0->BD1 = 4 transfers, 64 bytes
        engine.enqueue_task(2, 0, 1, false);

        let mut cycles = 0;
        while engine.channel_has_pending_work(2) {
            engine.step(&mut tile, &mut NeighborLocks::empty(), &mut host_mem);
            cycles += 1;
            if cycles > 500 {
                panic!("BD chain with repeat took too long (>500 cycles)");
            }
        }

        let stats = engine.channel_stats(2).unwrap();
        // 2 iterations x 2 BDs per chain = 4 transfers
        assert_eq!(stats.transfers_completed, 4,
            "Expected 4 transfers (2 chain iterations x 2 BDs), got {}",
            stats.transfers_completed);
        // 4 transfers x 16 bytes = 64 bytes total
        assert_eq!(stats.bytes_transferred, 64,
            "Expected 64 bytes (4 x 16), got {}",
            stats.bytes_transferred);
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
        let tile = make_tile();

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

    #[test]
    fn test_resolve_lock_id_memtile() {
        // MemTile: 64 locks, 192-entry address space
        let tile_type = TileType::MemTile;
        let num_locks = 64;

        // West neighbor: IDs 0-63
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 0),
            Some(LockTarget::West(0))
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 63),
            Some(LockTarget::West(63))
        );

        // Own tile: IDs 64-127
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 64),
            Some(LockTarget::Own(0))
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 127),
            Some(LockTarget::Own(63))
        );

        // East neighbor: IDs 128-191
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 128),
            Some(LockTarget::East(0))
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 191),
            Some(LockTarget::East(63))
        );

        // Out of range
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 1, num_locks, 192),
            None
        );
    }

    #[test]
    fn test_resolve_lock_id_compute() {
        // Compute tiles: 4-bit field, always Own
        let tile_type = TileType::Compute;
        let num_locks = 16;
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 1, 2, num_locks, 5),
            Some(LockTarget::Own(5))
        );
    }

    #[test]
    fn test_cross_tile_lock_acquire_west() {
        // Create MemTile DMA engine at col 1, row 1
        let mut engine = DmaEngine::new_mem_tile(1, 1);

        // Create own tile and west neighbor tile
        let mut own_tile = Tile::mem_tile(1, 1);
        let mut west_tile = Tile::mem_tile(0, 1);

        // Set west tile's lock 5 to value 1 (will be acquired via acq_eq)
        west_tile.locks[5].set(1);

        // Configure BD with acquire on west neighbor lock 5.
        // West locks are IDs 0-63, so lock_id=5 means west lock 5.
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(5, 1);  // acq_eq: wait for value == 1
        engine.configure_bd(0, bd).unwrap();

        // Write data to own tile memory (MM2S reads from here)
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xAA; 32]);

        // Start MM2S channel (channel 6 for MemTile)
        engine.start_channel(6, 0).unwrap();
        assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(5)));

        // Submit lock requests, resolve arbiters, then step
        {
            let mut neighbors = NeighborLocks {
                west: Some(&mut west_tile),
                east: None,
            };
            engine.submit_lock_requests(&mut own_tile, &mut neighbors);
        }
        own_tile.resolve_lock_requests(0);
        west_tile.resolve_lock_requests(0);

        let mut neighbors = NeighborLocks {
            west: Some(&mut west_tile),
            east: None,
        };
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

        // Channel should now be active (lock acquired from west neighbor)
        assert_eq!(engine.channel_state(6), ChannelState::Active,
            "Channel should be active after acquiring west neighbor lock");
    }

    #[test]
    fn test_cross_tile_lock_acquire_east() {
        // Create MemTile DMA engine at col 1, row 1
        let mut engine = DmaEngine::new_mem_tile(1, 1);

        let mut own_tile = Tile::mem_tile(1, 1);
        let mut east_tile = Tile::mem_tile(2, 1);

        // Set east tile's lock 10 to value 1
        east_tile.locks[10].set(1);

        // East locks are IDs 128-191, so lock_id=138 means east lock 10.
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(138, 1);  // acq_eq on east lock 10
        engine.configure_bd(0, bd).unwrap();
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xBB; 32]);

        engine.start_channel(6, 0).unwrap();
        assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(138)));

        {
            let mut neighbors = NeighborLocks {
                west: None,
                east: Some(&mut east_tile),
            };
            engine.submit_lock_requests(&mut own_tile, &mut neighbors);
        }
        own_tile.resolve_lock_requests(0);
        east_tile.resolve_lock_requests(0);

        let mut neighbors = NeighborLocks {
            west: None,
            east: Some(&mut east_tile),
        };
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

        assert_eq!(engine.channel_state(6), ChannelState::Active,
            "Channel should be active after acquiring east neighbor lock");
    }

    #[test]
    fn test_cross_tile_lock_acquire_fails_without_neighbor() {
        // MemTile at col 0 has no west neighbor
        let mut engine = DmaEngine::new_mem_tile(0, 1);
        let mut own_tile = Tile::mem_tile(0, 1);
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(5, 1);  // West lock -- but no west neighbor at col 0
        engine.configure_bd(0, bd).unwrap();
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xCC; 32]);

        engine.start_channel(6, 0).unwrap();

        let mut neighbors = NeighborLocks::empty();
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

        // Should remain waiting -- no neighbor to satisfy lock
        assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(5)),
            "Should stay waiting when neighbor tile is absent");
    }

    #[test]
    fn test_cross_tile_lock_release_west() {
        // Verify that after a transfer, the release lock targets the west neighbor
        let mut engine = DmaEngine::new_mem_tile(1, 1);
        let mut own_tile = Tile::mem_tile(1, 1);
        let mut west_tile = Tile::mem_tile(0, 1);

        // BD: acquire own lock 0 (ID 64), release west lock 3 (ID 3)
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(64, 1)   // own lock 0, acq_eq value=1
            .with_release(3, 1);   // west lock 3, release delta +1
        engine.configure_bd(0, bd).unwrap();
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xDD; 32]);

        // Set own lock 0 to 1 so acquire succeeds
        own_tile.locks[0].set(1);

        engine.start_channel(6, 0).unwrap();

        // Run to completion
        let mut cycles = 0;
        while engine.channel_active(6) {
            {
                let mut neighbors = NeighborLocks {
                    west: Some(&mut west_tile),
                    east: None,
                };
                engine.submit_lock_requests(&mut own_tile, &mut neighbors);
            }
            own_tile.resolve_lock_requests(0);
            west_tile.resolve_lock_requests(0);
            let mut neighbors = NeighborLocks {
                west: Some(&mut west_tile),
                east: None,
            };
            let mut host_mem = make_host_memory();
            engine.step(&mut own_tile, &mut neighbors, &mut host_mem);
            cycles += 1;
            if cycles > 500 {
                panic!("Transfer took too long: {} cycles, state={:?}", cycles, engine.channel_state(6));
            }
        }

        // West tile lock 3 should have been incremented by +1 (release delta)
        assert_eq!(west_tile.locks[3].value, 1,
            "West neighbor lock 3 should be 1 after release with delta +1");
    }

    #[test]
    fn test_cross_tile_lock_release_east() {
        // Symmetric to test_cross_tile_lock_release_west: verify release
        // targets east neighbor lock via the 128-191 ID range.
        let mut engine = DmaEngine::new_mem_tile(1, 1);
        let mut own_tile = Tile::mem_tile(1, 1);
        let mut east_tile = Tile::mem_tile(2, 1);

        // BD: acquire own lock 0 (ID 64), release east lock 7 (ID 128+7=135)
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(64, 1)   // own lock 0, acq_eq value=1
            .with_release(135, 1); // east lock 7, release delta +1
        engine.configure_bd(0, bd).unwrap();
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xEE; 32]);

        // Set own lock 0 to 1 so acquire succeeds
        own_tile.locks[0].set(1);

        engine.start_channel(6, 0).unwrap();

        // Run to completion
        let mut cycles = 0;
        while engine.channel_active(6) {
            {
                let mut neighbors = NeighborLocks {
                    west: None,
                    east: Some(&mut east_tile),
                };
                engine.submit_lock_requests(&mut own_tile, &mut neighbors);
            }
            own_tile.resolve_lock_requests(0);
            east_tile.resolve_lock_requests(0);
            let mut neighbors = NeighborLocks {
                west: None,
                east: Some(&mut east_tile),
            };
            let mut host_mem = make_host_memory();
            engine.step(&mut own_tile, &mut neighbors, &mut host_mem);
            cycles += 1;
            if cycles > 500 {
                panic!("Transfer took too long: {} cycles, state={:?}", cycles, engine.channel_state(6));
            }
        }

        // East tile lock 7 should have been incremented by +1 (release delta)
        assert_eq!(east_tile.locks[7].value, 1,
            "East neighbor lock 7 should be 1 after release with delta +1");
    }

    #[test]
    fn test_cross_tile_lock_own_acquire_memtile() {
        // Verify that lock_id in the 64-127 range (Own) works for MemTile DMA.
        // This exercises the second region of the 192-entry address space.
        let mut engine = DmaEngine::new_mem_tile(1, 1);
        let mut own_tile = Tile::mem_tile(1, 1);

        // Set own lock 10 to value 1 (lock_id = 64 + 10 = 74)
        own_tile.locks[10].set(1);

        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(74, 1);  // own lock 10, acq_eq value=1
        engine.configure_bd(0, bd).unwrap();
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xCC; 32]);

        engine.start_channel(6, 0).unwrap();
        assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(74)));

        // Submit and resolve -- no neighbors needed for own-tile lock
        {
            let mut neighbors = NeighborLocks::empty();
            engine.submit_lock_requests(&mut own_tile, &mut neighbors);
        }
        own_tile.resolve_lock_requests(0);

        let mut neighbors = NeighborLocks::empty();
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

        assert_eq!(engine.channel_state(6), ChannelState::Active,
            "Channel should be active after acquiring own lock via memtile ID 74");
    }

    #[test]
    fn test_cross_tile_lock_acquire_no_east_neighbor() {
        // MemTile at col 3 (rightmost in 4-column array) has no east neighbor.
        // East lock access should remain waiting.
        let mut engine = DmaEngine::new_mem_tile(3, 1);
        let mut own_tile = Tile::mem_tile(3, 1);

        // East lock 0 = lock_id 128
        let bd = BdConfig::simple_1d(0x100, 32)
            .with_acquire(128, 1);
        engine.configure_bd(0, bd).unwrap();
        own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xDD; 32]);

        engine.start_channel(6, 0).unwrap();

        let mut neighbors = NeighborLocks::empty();
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

        // Should remain waiting -- no east neighbor to satisfy lock
        assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(128)),
            "Should stay waiting when east neighbor tile is absent");
    }

    #[test]
    fn test_resolve_lock_id_memtile_boundary_values() {
        // Exhaustive boundary test for all three regions of the 192-entry space.
        let tile_type = TileType::MemTile;
        let num_locks: u8 = 64;

        // Region boundaries: 0, 63, 64, 127, 128, 191, 192
        // West region: [0, 64)
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 0),
            Some(LockTarget::West(0)),
            "lock_id=0 -> West(0)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 32),
            Some(LockTarget::West(32)),
            "lock_id=32 -> West(32) (mid-range)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 63),
            Some(LockTarget::West(63)),
            "lock_id=63 -> West(63) (last in West region)"
        );

        // Own region: [64, 128)
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 64),
            Some(LockTarget::Own(0)),
            "lock_id=64 -> Own(0) (first in Own region)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 96),
            Some(LockTarget::Own(32)),
            "lock_id=96 -> Own(32) (mid-range)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 127),
            Some(LockTarget::Own(63)),
            "lock_id=127 -> Own(63) (last in Own region)"
        );

        // East region: [128, 192)
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 128),
            Some(LockTarget::East(0)),
            "lock_id=128 -> East(0) (first in East region)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 160),
            Some(LockTarget::East(32)),
            "lock_id=160 -> East(32) (mid-range)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 191),
            Some(LockTarget::East(63)),
            "lock_id=191 -> East(63) (last in East region)"
        );

        // Out of range
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 192),
            None,
            "lock_id=192 -> None (out of range)"
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 2, 1, num_locks, 255),
            None,
            "lock_id=255 -> None (out of range, max u8)"
        );
    }

    #[test]
    fn test_resolve_lock_id_shim_passthrough() {
        // Shim tiles use a small lock ID field, always maps to Own.
        let tile_type = TileType::Shim;
        let num_locks = 16;
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 0, 0, num_locks, 0),
            Some(LockTarget::Own(0))
        );
        assert_eq!(
            DmaEngine::resolve_lock_id_static(tile_type, 0, 0, num_locks, 15),
            Some(LockTarget::Own(15))
        );
    }

    #[test]
    fn test_stream_in_per_channel_isolation() {
        // Bug: shared stream_in buffer lets one S2MM channel's data block another.
        // In real hardware, each S2MM channel has its own input FIFO connected to
        // a dedicated stream switch master port. One channel flooding its FIFO
        // must not prevent another channel from receiving data.
        let mut engine = DmaEngine::new_shim_tile(0, 0);

        // Fill stream_in with channel 1 (trace) data up to capacity.
        // With a shared buffer of 256 entries, this would block channel 0.
        for i in 0..256 {
            let pushed = engine.push_stream_in(StreamData {
                data: 0xFEED_0000 | i as u32,
                tlast: false,
                channel: 1,
            });
            assert!(pushed, "channel 1 push {} should succeed", i);
        }

        // Channel 0 (output) must still be able to receive data.
        // On real hardware, channel 0's FIFO is independent of channel 1's.
        let pushed = engine.push_stream_in(StreamData {
            data: 0x0010_0001,
            tlast: false,
            channel: 0,
        });
        assert!(pushed, "channel 0 push must succeed even when channel 1 is full");

        // And channel 0 data must be readable
        assert!(engine.has_stream_in_for_channel(0));
    }

    #[test]
    fn test_memtile_mm2s_packet_header_insertion() {
        // MemTile MM2S BD with enable_packet=true should create a Transfer
        // that can generate a packet header.
        //
        // This verifies the data path from BD config to Transfer to packet
        // header, which is the path that the packet_flow test depends on.
        use crate::device::dma::transfer::{Transfer, TransferDirection};
        use crate::device::TileType;

        let mut bd = BdConfig::default();
        bd.valid = true;
        bd.base_addr = 0x80000;
        bd.length = 16;
        bd.enable_packet = true;
        bd.packet_id = 5;
        bd.packet_type = 0;
        bd.d0.size = 4;
        bd.d0.stride = 4;

        // Create a Transfer from this BD (as MemTile MM2S)
        let transfer = Transfer::new(
            &bd, 1, 6, TransferDirection::MM2S, 0, 1, TileType::MemTile
        ).expect("should create transfer");

        // The transfer must carry the packet config from the BD
        assert!(transfer.enable_packet,
            "Transfer.enable_packet must be true when BD has it");
        assert_eq!(transfer.packet_id, 5,
            "Transfer.packet_id must match BD");
        assert!(transfer.needs_packet_header(),
            "needs_packet_header() should be true before sending");

        // Generate the header
        let header_word = transfer.generate_packet_header()
            .expect("should generate packet header");
        let (hdr, _) = crate::device::stream_switch::PacketHeader::decode(header_word);
        assert_eq!(hdr.stream_id, 5,
            "header stream_id should match BD packet_id");
    }

    #[test]
    fn test_memtile_mm2s_engine_inserts_header() {
        // End-to-end: MemTile DMA engine with enable_packet BD should
        // insert a packet header into stream_out during BdSetup.
        let mut engine = DmaEngine::new_mem_tile(0, 1);

        let mut bd = BdConfig::default();
        bd.valid = true;
        bd.base_addr = 0;
        bd.length = 16;
        bd.enable_packet = true;
        bd.packet_id = 5;
        bd.d0.size = 4;
        bd.d0.stride = 4;

        engine.configure_bd(1, bd.clone()).unwrap();

        // Verify the BD is stored with enable_packet
        let stored = engine.get_bd(1).unwrap();
        assert!(stored.enable_packet, "stored BD must have enable_packet=true");
        assert_eq!(stored.packet_id, 5, "stored BD must have packet_id=5");

        // Start MM2S channel 0 (index 6 for MemTile)
        let mm2s_ch = engine.s2mm_channel_count() as u8;
        assert_eq!(mm2s_ch, 6, "MemTile S2MM count should be 6");
        engine.start_channel(mm2s_ch, 1).unwrap();

        // After start_channel, the FSM should be in BdSetup or AcquiringLock.
        // BdSetup inserts the packet header on completion.
        // Since our BD has no acquire_lock, the path is:
        //   start_channel -> BdSetup{transfer} -> (step) -> skip AcquiringLock -> MemoryLatency
        // The header is inserted at the BdSetup->next transition.

        // Peek at the FSM to verify the transfer has enable_packet
        let ch = &engine.channels[mm2s_ch as usize];
        match &ch.fsm {
            ChannelFsm::BdSetup { transfer, .. } => {
                assert!(transfer.enable_packet,
                    "Transfer in BdSetup must have enable_packet=true, got false \
                     (BD enable_packet={}, packet_id={})",
                    bd.enable_packet, bd.packet_id);
            }
            other => panic!("Expected BdSetup, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---------------------------------------------------------------
    // MemTile BD-channel validity tests
    // ---------------------------------------------------------------

    #[test]
    fn test_memtile_bd_channel_validity_even_channels_low_bds() {
        // BDs 0-23 are valid only for even per-direction channels (0, 2, 4).
        let engine = DmaEngine::new_mem_tile(0, 1);

        // Even S2MM channels (flat 0, 2, 4) with low BDs -> valid
        assert!(engine.check_memtile_bd_channel_validity(0, 0));
        assert!(engine.check_memtile_bd_channel_validity(10, 2));
        assert!(engine.check_memtile_bd_channel_validity(23, 4));

        // Even MM2S channels (flat 6, 8, 10 -> per-dir 0, 2, 4) with low BDs -> valid
        assert!(engine.check_memtile_bd_channel_validity(0, 6));
        assert!(engine.check_memtile_bd_channel_validity(15, 8));
        assert!(engine.check_memtile_bd_channel_validity(23, 10));
    }

    #[test]
    fn test_memtile_bd_channel_validity_odd_channels_high_bds() {
        // BDs 24-47 are valid only for odd per-direction channels (1, 3, 5).
        let engine = DmaEngine::new_mem_tile(0, 1);

        // Odd S2MM channels (flat 1, 3, 5) with high BDs -> valid
        assert!(engine.check_memtile_bd_channel_validity(24, 1));
        assert!(engine.check_memtile_bd_channel_validity(35, 3));
        assert!(engine.check_memtile_bd_channel_validity(47, 5));

        // Odd MM2S channels (flat 7, 9, 11 -> per-dir 1, 3, 5) with high BDs -> valid
        assert!(engine.check_memtile_bd_channel_validity(24, 7));
        assert!(engine.check_memtile_bd_channel_validity(36, 9));
        assert!(engine.check_memtile_bd_channel_validity(47, 11));
    }

    #[test]
    fn test_memtile_bd_channel_validity_invalid_combinations() {
        // Even channel + high BD -> invalid
        // Odd channel + low BD -> invalid
        let engine = DmaEngine::new_mem_tile(0, 1);

        // Even S2MM channel with high BD -> invalid
        assert!(!engine.check_memtile_bd_channel_validity(24, 0));
        assert!(!engine.check_memtile_bd_channel_validity(47, 2));

        // Odd S2MM channel with low BD -> invalid
        assert!(!engine.check_memtile_bd_channel_validity(0, 1));
        assert!(!engine.check_memtile_bd_channel_validity(23, 3));

        // Even MM2S channel with high BD -> invalid
        assert!(!engine.check_memtile_bd_channel_validity(30, 6));
        assert!(!engine.check_memtile_bd_channel_validity(47, 8));

        // Odd MM2S channel with low BD -> invalid
        assert!(!engine.check_memtile_bd_channel_validity(0, 7));
        assert!(!engine.check_memtile_bd_channel_validity(10, 9));
    }

    #[test]
    fn test_memtile_bd_channel_validity_non_memtile_always_valid() {
        // Compute and shim tiles should always pass (no BD-channel constraint).
        let compute = DmaEngine::new_compute_tile(1, 2);
        assert!(compute.check_memtile_bd_channel_validity(0, 0));
        assert!(compute.check_memtile_bd_channel_validity(15, 1));

        let shim = DmaEngine::new_shim_tile(0, 0);
        assert!(shim.check_memtile_bd_channel_validity(0, 0));
        assert!(shim.check_memtile_bd_channel_validity(15, 1));
    }

    #[test]
    fn test_memtile_bd_channel_validity_boundary_bd23_bd24() {
        // BD 23 is the last "low" BD, BD 24 is the first "high" BD.
        let engine = DmaEngine::new_mem_tile(0, 1);

        // BD 23, even channel -> valid
        assert!(engine.check_memtile_bd_channel_validity(23, 0));
        // BD 23, odd channel -> invalid
        assert!(!engine.check_memtile_bd_channel_validity(23, 1));

        // BD 24, odd channel -> valid
        assert!(engine.check_memtile_bd_channel_validity(24, 1));
        // BD 24, even channel -> invalid
        assert!(!engine.check_memtile_bd_channel_validity(24, 0));
    }

    #[test]
    fn test_per_direction_channel() {
        let engine = DmaEngine::new_mem_tile(0, 1);

        // S2MM channels: flat index IS the per-direction index
        assert_eq!(engine.per_direction_channel(0), 0);
        assert_eq!(engine.per_direction_channel(3), 3);
        assert_eq!(engine.per_direction_channel(5), 5);

        // MM2S channels: flat index 6..11 -> per-direction 0..5
        assert_eq!(engine.per_direction_channel(6), 0);
        assert_eq!(engine.per_direction_channel(7), 1);
        assert_eq!(engine.per_direction_channel(11), 5);
    }
}
