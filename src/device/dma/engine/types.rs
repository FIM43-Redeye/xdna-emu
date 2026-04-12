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

/// Identifies which MemTile's data memory a DMA byte address targets.
///
/// MemTile DMA BDs encode a byte address into a windowed space that maps to
/// the local MemTile and its two neighbours via the dedicated MemTile-to-
/// MemTile shared-memory bus (NOT through the stream switch). With each
/// window equal to one MemTile's data memory size (typically 0x80000 bytes
/// = 512 KB), the convention per AM025 / mlir-aie is:
///
///   - `[0x00000, 0x80000)` -> West neighbor (col-1) data memory
///   - `[0x80000, 0x100000)` -> Own data memory
///   - `[0x100000, 0x180000)` -> East neighbor (col+1) data memory
///
/// Compute/shim DMAs do not use this windowing; their addresses always
/// refer to local memory and are wrapped at `mem_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemTileTarget {
    /// Access targets the local MemTile's data memory.
    Own,
    /// Access targets the west neighbor MemTile (col-1).
    West,
    /// Access targets the east neighbor MemTile (col+1).
    East,
}

/// Address out of the legal three-window MemTile address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemTileAddrOutOfRange {
    /// Original byte address from the BD.
    pub byte_addr: u64,
    /// Window size that was used for decoding (= MemTile data memory bytes).
    pub mem_size: usize,
}

impl MemTileTarget {
    /// Resolve a MemTile DMA byte address into a target tile + offset.
    ///
    /// `mem_size` is the byte size of one MemTile's data memory; it sets the
    /// window size. Returns the target window (West/Own/East) and the offset
    /// within that window's memory.
    ///
    /// Returns `Err` if the address falls outside the three-window space.
    /// On real hardware, such an address would cause a bus error; the caller
    /// should treat it as a fatal kernel bug.
    pub fn resolve(byte_addr: u64, mem_size: usize) -> Result<(Self, usize), MemTileAddrOutOfRange> {
        debug_assert!(mem_size > 0, "mem_size must be > 0 for MemTile address decode");
        let addr = byte_addr as usize;
        let window = addr / mem_size;
        let offset = addr % mem_size;
        let target = match window {
            0 => MemTileTarget::West,
            1 => MemTileTarget::Own,
            2 => MemTileTarget::East,
            _ => return Err(MemTileAddrOutOfRange { byte_addr, mem_size }),
        };
        Ok((target, offset))
    }
}

/// Provides mutable access to neighbor tiles for cross-tile DMA operations.
///
/// Models the MemTile-to-MemTile interconnect (shared-memory bus + lock
/// routing) that lets a MemTile DMA reach its `+/-1` column neighbours
/// without going through the stream switch. Constructed by the array level
/// using disjoint borrows via `get_three_mut`.
///
/// Used for both lock arbitration (`LockTarget::{West,East}`) and data
/// transfers (`MemTileTarget::{West,East}`). Non-MemTile tiles pass
/// `NeighborTiles::empty()`.
pub struct NeighborTiles<'a> {
    /// West neighbor MemTile (col-1), if it exists.
    pub west: Option<&'a mut Tile>,
    /// East neighbor MemTile (col+1), if it exists.
    pub east: Option<&'a mut Tile>,
}

impl NeighborTiles<'_> {
    /// Create an empty neighbor context (no cross-tile access).
    ///
    /// Used for compute/shim tiles where cross-tile MemTile access
    /// is not applicable.
    pub fn empty() -> NeighborTiles<'static> {
        NeighborTiles { west: None, east: None }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// MemTile data memory size used for the standard NPU1 layout (512 KB).
    /// Matches `tile.data_memory().len()` for a real MemTile.
    const MEMTILE_BYTES: usize = 0x80000;

    #[test]
    fn memtile_target_resolve_west_window_at_zero() {
        // Window 0 = West neighbour
        let (target, offset) = MemTileTarget::resolve(0, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::West);
        assert_eq!(offset, 0);
    }

    #[test]
    fn memtile_target_resolve_west_window_offset() {
        let (target, offset) = MemTileTarget::resolve(0x1234, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::West);
        assert_eq!(offset, 0x1234);
    }

    #[test]
    fn memtile_target_resolve_own_window_at_boundary() {
        // 0x80000 = exact start of Own window
        let (target, offset) = MemTileTarget::resolve(0x80000, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::Own);
        assert_eq!(offset, 0);
    }

    #[test]
    fn memtile_target_resolve_own_window_offset() {
        let (target, offset) = MemTileTarget::resolve(0x80000 + 0xABC, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::Own);
        assert_eq!(offset, 0xABC);
    }

    #[test]
    fn memtile_target_resolve_east_window_at_boundary() {
        let (target, offset) = MemTileTarget::resolve(0x100000, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::East);
        assert_eq!(offset, 0);
    }

    #[test]
    fn memtile_target_resolve_east_window_offset() {
        let (target, offset) = MemTileTarget::resolve(0x100000 + 0x40, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::East);
        assert_eq!(offset, 0x40);
    }

    #[test]
    fn memtile_target_resolve_top_of_east_window() {
        // 0x17FFFC = last 4-byte word of East window
        let (target, offset) = MemTileTarget::resolve(0x17FFFC, MEMTILE_BYTES).unwrap();
        assert_eq!(target, MemTileTarget::East);
        assert_eq!(offset, 0x7FFFC);
    }

    #[test]
    fn memtile_target_resolve_out_of_range_returns_err() {
        // 0x180000 = first byte beyond East window, no fourth window exists.
        let err = MemTileTarget::resolve(0x180000, MEMTILE_BYTES).unwrap_err();
        assert_eq!(err.byte_addr, 0x180000);
        assert_eq!(err.mem_size, MEMTILE_BYTES);
    }

    #[test]
    fn memtile_target_resolve_far_out_of_range_returns_err() {
        let err = MemTileTarget::resolve(0xFFFF_FFFF, MEMTILE_BYTES).unwrap_err();
        assert_eq!(err.byte_addr, 0xFFFF_FFFF);
    }
}
