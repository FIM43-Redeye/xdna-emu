//! Type definitions used by the DMA engine.

use crate::device::tile::Tile;

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
pub use super::super::token::{TaskQueueEntry, TaskQueue, Token, TokenState, MAX_TASK_QUEUE_DEPTH};

/// Transfer operation result (internal).
#[derive(Debug, Clone, Copy)]
pub(super) struct TransferResult {
    /// Transfer succeeded (or stalled waiting for resource)
    pub success: bool,
    /// Transfer stalled waiting for stream data (S2MM only)
    /// When stalled, timing should NOT advance and no error should be raised.
    pub stall: bool,
    /// TLAST received on S2MM - finish early if FoT enabled
    pub fot_finish: bool,
}

impl TransferResult {
    pub fn success() -> Self {
        Self { success: true, stall: false, fot_finish: false }
    }
    pub fn stalled() -> Self {
        Self { success: true, stall: true, fot_finish: false }
    }
    pub fn failure() -> Self {
        Self { success: false, stall: false, fot_finish: false }
    }
}

/// S2MM transfer result (internal).
#[derive(Debug, Clone, Copy)]
pub(super) struct S2mmResult {
    /// Transfer was successful (data written) or stalled (no data available)
    pub success: bool,
    /// Stalled waiting for stream input data
    pub stall: bool,
    /// TLAST was received (for FoT mode)
    pub tlast_received: bool,
    /// Number of bytes actually written
    pub bytes_written: usize,
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
pub(super) enum TransferCycleResult {
    /// Data moved successfully (or nothing to do this cycle)
    Continue,
    /// S2MM stalled waiting for stream data
    Stalled,
    /// FoT TLAST received, finish early
    FotFinish,
    /// Transfer error (bad address, etc.)
    Error,
}
