//! DMA (Direct Memory Access) engine subsystem.
//!
//! This module implements the DMA functionality for AIE2 tiles:
//! - Buffer descriptor (BD) interpretation
//! - Multi-dimensional addressing (1D, 2D, 3D, 4D)
//! - Transfer state machine
//! - Shim DMA (DDR interface)
//! - Tile DMA (local memory interface)
//!
//! # Architecture
//!
//! Each tile has a DMA engine with:
//! - 16 buffer descriptors (BDs)
//! - 4 channels (2 S2MM + 2 MM2S for compute tiles)
//! - Multi-dimensional addressing with wrap/stride
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DMA Controller                            │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │              Buffer Descriptors (16)                 │    │
//! │  │  BD0: addr, len, stride, wrap, next_bd              │    │
//! │  │  BD1: ...                                           │    │
//! │  │  ...                                                │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │                                                              │
//! │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
//! │  │  S2MM_0   │ │  S2MM_1   │ │  MM2S_0   │ │  MM2S_1   │   │
//! │  │ (stream   │ │ (stream   │ │ (memory   │ │ (memory   │   │
//! │  │  to mem)  │ │  to mem)  │ │  to strm) │ │  to strm) │   │
//! │  └───────────┘ └───────────┘ └───────────┘ └───────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Transfer Types
//!
//! - **S2MM (Stream to Memory)**: Receives data from stream switch, writes to memory
//! - **MM2S (Memory to Stream)**: Reads from memory, sends to stream switch
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::device::dma::{DmaEngine, TransferRequest};
//!
//! let mut engine = DmaEngine::new();
//!
//! // Configure a BD
//! engine.configure_bd(0, BdConfig {
//!     base_addr: 0x1000,
//!     length: 1024,
//!     ..Default::default()
//! });
//!
//! // Start a transfer
//! engine.start_channel(0, 0); // Channel 0, start from BD 0
//!
//! // Step the engine (call each cycle)
//! while engine.channel_busy(0) {
//!     engine.step(tile_memory, host_memory);
//! }
//! ```

pub mod addressing;
pub mod transfer;
pub mod engine;
pub mod timing;
pub mod compression;

pub use addressing::{AddressGenerator, DimensionConfig, AddressIterator, IterationConfig};
pub use transfer::{Transfer, TransferState, TransferDirection, TransferEndpoint, parse_ooo_bd_id_from_header, parse_source_tile_from_header};
pub use engine::{DmaEngine, ChannelState, ChannelId, StreamData, TaskCompleteToken, ChannelTaskConfig, TaskQueueEntry, MAX_TASK_QUEUE_DEPTH};
pub use timing::{DmaTimingConfig, ChannelTimingState, TransferPhase, ChannelArbiter};

use super::aie2_spec;
use super::tile::TileType;

/// Number of buffer descriptors per compute tile DMA controller.
pub const NUM_BUFFER_DESCRIPTORS: usize = aie2_spec::NUM_DMA_BUFFER_DESCRIPTORS;

/// Number of buffer descriptors per memory tile DMA controller.
pub const MEMTILE_NUM_BUFFER_DESCRIPTORS: usize = aie2_spec::MEMTILE_NUM_DMA_BUFFER_DESCRIPTORS;

/// Number of S2MM channels for compute tiles.
pub const COMPUTE_S2MM_CHANNELS: usize = aie2_spec::COMPUTE_TILE_S2MM_CHANNELS;

/// Number of MM2S channels for compute tiles.
pub const COMPUTE_MM2S_CHANNELS: usize = aie2_spec::COMPUTE_TILE_MM2S_CHANNELS;

/// Number of S2MM channels for memory tiles.
pub const MEM_TILE_S2MM_CHANNELS: usize = aie2_spec::MEM_TILE_S2MM_CHANNELS;

/// Number of MM2S channels for memory tiles.
pub const MEM_TILE_MM2S_CHANNELS: usize = aie2_spec::MEM_TILE_MM2S_CHANNELS;

/// DMA data width in bits.
pub const DMA_DATA_WIDTH_BITS: usize = aie2_spec::DMA_DATA_WIDTH_BITS;

/// DMA data width in bytes.
pub const DMA_DATA_WIDTH_BYTES: usize = DMA_DATA_WIDTH_BITS / 8;

/// Buffer descriptor configuration.
///
/// This is a user-friendly representation of a BD. The actual hardware
/// BD format is more compact and uses packed fields.
///
/// # Unit Conventions
///
/// This struct uses **byte units** throughout for programmer convenience.
/// The hardware BD format (AM025) uses **word units** (32-bit words) for
/// addresses and strides. Conversion happens at BD parsing/serialization:
///
/// | Field | BdConfig Unit | Hardware Unit | Conversion |
/// |-------|---------------|---------------|------------|
/// | `base_addr` | bytes | words | ÷4 to hardware |
/// | `length` | bytes | words | ÷4 to hardware |
/// | `d0-d3.stride` | bytes | words | ÷4 to hardware |
/// | `iteration.stepsize` | words (raw) | words | no conversion |
///
/// See `state.rs` for BD parsing which performs word→byte conversion.
#[derive(Debug, Clone, Default)]
pub struct BdConfig {
    /// Base address in bytes (32-bit for tile memory, 64-bit for host/DDR).
    /// Hardware stores this as a 32-bit word address.
    pub base_addr: u64,

    /// Transfer length in bytes.
    /// Hardware stores this as word count.
    pub length: u32,

    /// Dimension 0 configuration (innermost loop)
    pub d0: DimensionConfig,

    /// Dimension 1 configuration
    pub d1: DimensionConfig,

    /// Dimension 2 configuration
    pub d2: DimensionConfig,

    /// Dimension 3 configuration (outermost loop) - AIE2P only
    pub d3: DimensionConfig,

    /// Iteration configuration (repeat with offset)
    pub iteration: IterationConfig,

    /// Enable compression (MM2S only)
    pub compression_enable: bool,

    /// Enable packet header insertion (MM2S only)
    pub enable_packet: bool,

    /// Packet ID for header (5 bits, MM2S only)
    pub packet_id: u8,

    /// Packet type for header (3 bits, MM2S only)
    pub packet_type: u8,

    /// Out-of-order BD ID (6 bits, MM2S only)
    pub out_of_order_bd_id: u8,

    /// Enable out-of-order execution (S2MM only)
    pub out_of_order: bool,

    /// Suppress TLAST at end of transfer (MM2S only)
    /// When true, TLAST is not asserted at the end of this BD's transfer.
    /// Used for multi-BD transfers that should appear as a single stream.
    pub tlast_suppress: bool,

    /// Lock to acquire before transfer
    pub acquire_lock: Option<u8>,

    /// Lock value to wait for
    pub acquire_value: i8,

    /// Lock to release after transfer
    pub release_lock: Option<u8>,

    /// Lock value after release
    pub release_value: i8,

    /// Next BD to chain to (None = stop after this BD)
    pub next_bd: Option<u8>,

    /// Enable this BD (must be true for transfers)
    pub valid: bool,
}

impl BdConfig {
    /// Create a simple 1D contiguous transfer BD.
    ///
    /// Configures d0 for a linear transfer: `length` bytes starting at `base_addr`.
    /// AIE-ML DMA works in 32-bit word units, so the address generator advances
    /// once per word (4 bytes) with a stride of 4 bytes.
    pub fn simple_1d(base_addr: u64, length: u32) -> Self {
        Self {
            base_addr,
            length,
            // For 1D contiguous: d0.size = words, d0.stride = 4 bytes per word
            // This matches how real BD configuration works (word-level addressing)
            d0: DimensionConfig::new(length / 4, 4),
            valid: true,
            ..Default::default()
        }
    }

    /// Create a 2D transfer BD.
    ///
    /// Width and stride should account for the 4-byte word granularity.
    pub fn transfer_2d(
        base_addr: u64,
        width: u32,      // Bytes per row (must be multiple of 4)
        height: u32,     // Number of rows
        stride: i32,     // Bytes between row starts
    ) -> Self {
        Self {
            base_addr,
            length: width * height,
            // D0: words per row, 4-byte stride for contiguous elements
            d0: DimensionConfig {
                size: width / 4,  // Convert bytes to words
                stride: 4,        // 4 bytes per word
            },
            d1: DimensionConfig {
                size: height,
                stride,
            },
            valid: true,
            ..Default::default()
        }
    }

    /// Set lock acquisition.
    pub fn with_acquire(mut self, lock_id: u8, value: i8) -> Self {
        self.acquire_lock = Some(lock_id);
        self.acquire_value = value;
        self
    }

    /// Set lock release.
    pub fn with_release(mut self, lock_id: u8, value: i8) -> Self {
        self.release_lock = Some(lock_id);
        self.release_value = value;
        self
    }

    /// Set next BD for chaining.
    pub fn with_next(mut self, next_bd: u8) -> Self {
        self.next_bd = Some(next_bd);
        self
    }
}

/// Channel type (direction of transfer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Stream to Memory (receive from stream, write to memory)
    S2MM,
    /// Memory to Stream (read from memory, send to stream)
    MM2S,
}

impl ChannelType {
    /// Get the channel type for a channel index.
    ///
    /// For compute/shim tiles: channels 0-1 are S2MM, 2-3 are MM2S.
    /// For memory tiles: channels 0-5 are S2MM, 6-11 are MM2S.
    pub fn from_channel_index(idx: usize, tile_type: TileType) -> Self {
        let s2mm_count = if tile_type.is_mem_tile() {
            MEM_TILE_S2MM_CHANNELS
        } else {
            COMPUTE_S2MM_CHANNELS
        };

        if idx < s2mm_count {
            ChannelType::S2MM
        } else {
            ChannelType::MM2S
        }
    }
}

/// Result of a DMA operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmaResult {
    /// Operation completed successfully
    Complete,
    /// Transfer in progress (more cycles needed)
    InProgress,
    /// Waiting for lock
    WaitingForLock(u8),
    /// Waiting for stream data (S2MM)
    WaitingForStream,
    /// Error occurred
    Error(DmaError),
}

/// DMA error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DmaError {
    /// Invalid BD index
    InvalidBd(u8),
    /// Invalid channel index
    InvalidChannel(u8),
    /// BD not valid (not configured)
    BdNotValid(u8),
    /// Address out of bounds
    AddressOutOfBounds { address: u64, limit: u64 },
    /// Channel already active
    ChannelBusy(u8),
    /// Memory transfer address wraps past end of memory
    AddressWrap { offset: u64, bytes: usize, memory_size: usize },
    /// Stream input buffer full, data was dropped
    StreamBufferFull { channel: u8 },
    /// S2MM channel stalled waiting for stream data
    StreamStall { channel: u8, address: u64 },
}

/// Result of a single DMA transfer step (internal use).
///
/// This provides more detail than [`DmaResult`] for tracking step-level
/// success/failure and error conditions.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Whether the step succeeded
    pub success: bool,
    /// Number of bytes moved in this step
    pub bytes_moved: usize,
    /// Error details if step failed
    pub error: Option<DmaError>,
    /// First-of-transfer finish signal (TLAST received in FoT mode)
    pub fot_finish: bool,
}

impl StepResult {
    /// Create a successful step result.
    pub fn success(bytes: usize) -> Self {
        Self { success: true, bytes_moved: bytes, error: None, fot_finish: false }
    }

    /// Mark this result as triggering FoT finish.
    pub fn with_fot(mut self) -> Self {
        self.fot_finish = true;
        self
    }

    /// Create an error step result.
    pub fn error(e: DmaError) -> Self {
        Self { success: false, bytes_moved: 0, error: Some(e), fot_finish: false }
    }
}

impl std::fmt::Display for DmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBd(id) => write!(f, "Invalid BD index: {}", id),
            Self::InvalidChannel(ch) => write!(f, "Invalid channel index: {}", ch),
            Self::BdNotValid(id) => write!(f, "BD {} is not valid/configured", id),
            Self::AddressOutOfBounds { address, limit } => {
                write!(f, "Address 0x{:08x} out of bounds (limit: 0x{:08x})", address, limit)
            }
            Self::ChannelBusy(ch) => write!(f, "Channel {} is already active", ch),
            Self::AddressWrap { offset, bytes, memory_size } => {
                write!(f, "Address 0x{:08x} + {} bytes wraps past memory end (size: 0x{:x})",
                    offset, bytes, memory_size)
            }
            Self::StreamBufferFull { channel } => {
                write!(f, "Stream input buffer full, ch{} data dropped", channel)
            }
            Self::StreamStall { channel, address } => {
                write!(f, "S2MM ch{} stalled at 0x{:08x} waiting for stream data", channel, address)
            }
        }
    }
}

impl std::error::Error for DmaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bd_config_simple() {
        let bd = BdConfig::simple_1d(0x1000, 256);
        assert_eq!(bd.base_addr, 0x1000);
        assert_eq!(bd.length, 256);
        assert!(bd.valid);
        assert!(bd.next_bd.is_none());
    }

    #[test]
    fn test_bd_config_2d() {
        let bd = BdConfig::transfer_2d(0x2000, 64, 8, 128);
        assert_eq!(bd.base_addr, 0x2000);
        assert_eq!(bd.length, 512); // 64 * 8
        // d0.size is in words (64 bytes / 4 = 16 words per row)
        assert_eq!(bd.d0.size, 16);
        assert_eq!(bd.d0.stride, 4); // 4 bytes per word
        assert_eq!(bd.d1.size, 8);
        assert_eq!(bd.d1.stride, 128);
    }

    #[test]
    fn test_bd_config_with_locks() {
        let bd = BdConfig::simple_1d(0x3000, 128)
            .with_acquire(5, 1)
            .with_release(5, -1);

        assert_eq!(bd.acquire_lock, Some(5));
        assert_eq!(bd.acquire_value, 1);
        assert_eq!(bd.release_lock, Some(5));
        assert_eq!(bd.release_value, -1);
    }

    #[test]
    fn test_channel_type_compute() {
        assert_eq!(ChannelType::from_channel_index(0, TileType::Compute), ChannelType::S2MM);
        assert_eq!(ChannelType::from_channel_index(1, TileType::Compute), ChannelType::S2MM);
        assert_eq!(ChannelType::from_channel_index(2, TileType::Compute), ChannelType::MM2S);
        assert_eq!(ChannelType::from_channel_index(3, TileType::Compute), ChannelType::MM2S);
    }

    #[test]
    fn test_channel_type_mem_tile() {
        // Mem tile has 6 S2MM and 6 MM2S
        assert_eq!(ChannelType::from_channel_index(0, TileType::MemTile), ChannelType::S2MM);
        assert_eq!(ChannelType::from_channel_index(5, TileType::MemTile), ChannelType::S2MM);
        assert_eq!(ChannelType::from_channel_index(6, TileType::MemTile), ChannelType::MM2S);
        assert_eq!(ChannelType::from_channel_index(11, TileType::MemTile), ChannelType::MM2S);
    }
}
