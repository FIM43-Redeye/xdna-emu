//! DMA transfer state machine.
//!
//! A transfer represents an active DMA operation from a single buffer
//! descriptor. It tracks the current position, handles multi-dimensional
//! addressing, and manages the transfer lifecycle.
//!
//! # Transfer Lifecycle
//!
//! ```text
//! ┌─────────┐   lock    ┌───────────────┐   lock    ┌──────────┐
//! │ Created ├──acquire──► WaitingForLock├──acquired─► Active   │
//! └─────────┘           └───────────────┘           └────┬─────┘
//!                                                        │
//!                        ┌───────────────┐   all data   │
//!                        │   Complete    ◄──transferred─┘
//!                        └───────┬───────┘
//!                                │ lock release
//!                                ▼
//!                        ┌───────────────┐
//!                        │   Finished    │
//!                        └───────────────┘
//! ```
//!
//! # Data Flow
//!
//! For MM2S (Memory to Stream):
//! 1. Read data from tile memory at current address
//! 2. Send to stream switch
//! 3. Advance address generator
//!
//! For S2MM (Stream to Memory):
//! 1. Receive data from stream switch
//! 2. Write to tile memory at current address
//! 3. Advance address generator

use super::addressing::AddressGenerator;
use super::{BdConfig, DmaError};
use crate::device::tile::TileType;

/// Lock acquisition mode for DMA transfers.
///
/// AIE-ML supports different lock acquisition semantics. The BD encodes
/// these using a signed integer convention where negative values indicate
/// "greater-than-or-equal" mode.
///
/// # BD Encoding Convention
///
/// - `acquire_value > 0`: Equal mode - wait until lock == value, then set to 0
/// - `acquire_value < 0`: GE mode - wait until lock >= |value|, then decrement
/// - `acquire_value == 0`: Simple mode - decrement if lock > 0
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockAcquireMode {
    /// Wait until lock value >= threshold, then decrement by 1.
    /// Corresponds to negative acquire_value in BD.
    GreaterEqual(u8),
    /// Wait until lock value == threshold, then set to 0.
    /// Corresponds to positive acquire_value in BD.
    Equal(u8),
    /// Simple acquire: decrement if lock > 0.
    /// Corresponds to acquire_value == 0 in BD.
    Simple,
}

impl LockAcquireMode {
    /// Convert from the BD's signed integer convention.
    ///
    /// - Negative values -> GreaterEqual mode
    /// - Positive values -> Equal mode
    /// - Zero -> Simple mode
    pub fn from_bd_value(value: i8) -> Self {
        if value < 0 {
            LockAcquireMode::GreaterEqual((-value) as u8)
        } else if value > 0 {
            LockAcquireMode::Equal(value as u8)
        } else {
            LockAcquireMode::Simple
        }
    }

    /// Convert back to BD's signed integer convention.
    pub fn to_bd_value(&self) -> i8 {
        match self {
            LockAcquireMode::GreaterEqual(v) => -(*v as i8),
            LockAcquireMode::Equal(v) => *v as i8,
            LockAcquireMode::Simple => 0,
        }
    }

    /// Get the threshold value for this mode.
    pub fn threshold(&self) -> u8 {
        match self {
            LockAcquireMode::GreaterEqual(v) | LockAcquireMode::Equal(v) => *v,
            LockAcquireMode::Simple => 1,
        }
    }
}

/// Transfer state in the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferState {
    /// Transfer created but not yet started
    Created,
    /// Waiting to acquire lock before transfer
    WaitingForLock(u8),
    /// Transfer is actively moving data
    Active,
    /// All data transferred, waiting to release lock
    ReleasingLock(u8),
    /// Transfer complete
    Complete,
    /// Transfer failed with error
    Error,
}

/// Direction of a DMA transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Stream to Memory: receive from stream, write to tile memory
    S2MM,
    /// Memory to Stream: read from tile memory, send to stream
    MM2S,
}

/// Endpoint for a transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferEndpoint {
    /// Tile local memory (compute tile: 64KB, mem tile: 512KB)
    TileMemory {
        col: u8,
        row: u8,
    },
    /// Host/DDR memory (via shim tile)
    HostMemory,
    /// Stream switch connection
    Stream {
        port: u8,
    },
}

/// An active DMA transfer.
#[derive(Debug)]
pub struct Transfer {
    /// Source of data
    pub source: TransferEndpoint,

    /// Destination of data
    pub dest: TransferEndpoint,

    /// Transfer direction
    pub direction: TransferDirection,

    /// Current state
    pub state: TransferState,

    /// Address generator for multi-dimensional addressing
    pub address_gen: AddressGenerator,

    /// BD index this transfer was created from
    pub bd_index: u8,

    /// Channel performing this transfer
    pub channel: u8,

    /// Lock to acquire before transfer (if any)
    pub acquire_lock: Option<u8>,

    /// Lock value to acquire
    pub acquire_value: i8,

    /// Lock to release after transfer (if any)
    pub release_lock: Option<u8>,

    /// Lock value after release
    pub release_value: i8,

    /// Next BD to chain to (if any)
    pub next_bd: Option<u8>,

    /// Suppress TLAST at end of transfer (MM2S only)
    /// When true, TLAST is not asserted at the end of this transfer.
    /// Used for multi-BD transfers that should appear as a single stream.
    pub tlast_suppress: bool,

    /// Enable packet header insertion (MM2S only)
    /// When true, a 32-bit packet header is inserted before the data.
    pub enable_packet: bool,

    /// Packet ID for header (5 bits, MM2S only)
    /// Maps to Stream_ID field in packet header.
    pub packet_id: u8,

    /// Packet type for header (3 bits, MM2S only)
    pub packet_type: u8,

    /// Out-of-order BD ID for header (6 bits, MM2S only)
    pub out_of_order_bd_id: u8,

    /// Source tile column (for packet header)
    pub tile_col: u8,

    /// Source tile row (for packet header)
    pub tile_row: u8,

    /// Whether packet header has been sent (tracking for current transfer)
    pub packet_header_sent: bool,

    /// Bytes transferred so far
    pub bytes_transferred: u64,

    /// Total bytes to transfer
    pub total_bytes: u64,

    /// Cycle count for this transfer (for timing)
    pub cycles_elapsed: u64,

    /// Last error (if any)
    pub error: Option<DmaError>,
}

impl Transfer {
    /// Create a new transfer from a BD configuration.
    ///
    /// The `tile_type` parameter determines how endpoints are configured:
    /// - Shim tiles transfer to/from host DDR memory
    /// - Compute and MemTiles transfer to/from local tile memory
    pub fn new(
        bd_config: &BdConfig,
        bd_index: u8,
        channel: u8,
        direction: TransferDirection,
        tile_col: u8,
        tile_row: u8,
        tile_type: TileType,
    ) -> Result<Self, DmaError> {
        if !bd_config.valid {
            return Err(DmaError::BdNotValid(bd_index));
        }

        // Create address generator based on BD dimensions and iteration config
        let address_gen = AddressGenerator::with_iteration(
            bd_config.base_addr,
            [
                bd_config.d0,
                bd_config.d1,
                bd_config.d2,
                bd_config.d3,
            ],
            bd_config.iteration,
        );

        // Determine endpoints based on direction and tile type
        let is_shim = tile_type == TileType::Shim;

        let (source, dest) = match (direction, is_shim) {
            (TransferDirection::S2MM, true) => (
                // Shim S2MM: Stream (from tiles) -> Host DDR
                TransferEndpoint::Stream { port: channel },
                TransferEndpoint::HostMemory,
            ),
            (TransferDirection::MM2S, true) => (
                // Shim MM2S: Host DDR -> Stream (to tiles)
                TransferEndpoint::HostMemory,
                TransferEndpoint::Stream { port: channel },
            ),
            (TransferDirection::S2MM, false) => (
                // Compute S2MM: Stream -> Tile Memory
                TransferEndpoint::Stream { port: channel },
                TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
            ),
            (TransferDirection::MM2S, false) => (
                // Compute MM2S: Tile Memory -> Stream
                TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
                TransferEndpoint::Stream { port: channel },
            ),
        };

        let initial_state = if bd_config.acquire_lock.is_some() {
            TransferState::WaitingForLock(bd_config.acquire_lock.unwrap())
        } else {
            TransferState::Active
        };

        Ok(Self {
            source,
            dest,
            direction,
            state: initial_state,
            address_gen,
            bd_index,
            channel,
            acquire_lock: bd_config.acquire_lock,
            acquire_value: bd_config.acquire_value,
            release_lock: bd_config.release_lock,
            release_value: bd_config.release_value,
            next_bd: bd_config.next_bd,
            tlast_suppress: bd_config.tlast_suppress,
            enable_packet: bd_config.enable_packet,
            packet_id: bd_config.packet_id,
            packet_type: bd_config.packet_type,
            out_of_order_bd_id: bd_config.out_of_order_bd_id,
            tile_col,
            tile_row,
            packet_header_sent: false,
            bytes_transferred: 0,
            total_bytes: bd_config.length as u64,
            cycles_elapsed: 0,
            error: None,
        })
    }

    /// Create a simple memory-to-memory transfer (for testing).
    pub fn new_mem_copy(
        source_col: u8,
        source_row: u8,
        dest_col: u8,
        dest_row: u8,
        base_addr: u64,
        length: u32,
    ) -> Self {
        let address_gen = AddressGenerator::new_1d(base_addr, length, 1);

        Self {
            source: TransferEndpoint::TileMemory { col: source_col, row: source_row },
            dest: TransferEndpoint::TileMemory { col: dest_col, row: dest_row },
            direction: TransferDirection::MM2S, // Arbitrary for mem-to-mem
            state: TransferState::Active,
            address_gen,
            bd_index: 0,
            channel: 0,
            acquire_lock: None,
            acquire_value: 0,
            release_lock: None,
            release_value: 0,
            next_bd: None,
            tlast_suppress: false,
            enable_packet: false,
            packet_id: 0,
            packet_type: 0,
            out_of_order_bd_id: 0,
            tile_col: source_col,
            tile_row: source_row,
            packet_header_sent: false,
            bytes_transferred: 0,
            total_bytes: length as u64,
            cycles_elapsed: 0,
            error: None,
        }
    }

    /// Create a host-to-tile transfer (for loading data).
    pub fn new_host_to_tile(
        tile_col: u8,
        tile_row: u8,
        _host_addr: u64,
        tile_addr: u64,
        length: u32,
    ) -> Self {
        // Address gen uses tile address since that's where we write
        let address_gen = AddressGenerator::new_1d(tile_addr, length, 1);

        Self {
            source: TransferEndpoint::HostMemory,
            dest: TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
            direction: TransferDirection::S2MM,
            state: TransferState::Active,
            address_gen,
            bd_index: 0,
            channel: 0,
            acquire_lock: None,
            acquire_value: 0,
            release_lock: None,
            release_value: 0,
            next_bd: None,
            tlast_suppress: false,
            enable_packet: false,
            packet_id: 0,
            packet_type: 0,
            out_of_order_bd_id: 0,
            tile_col,
            tile_row,
            packet_header_sent: false,
            bytes_transferred: 0,
            total_bytes: length as u64,
            cycles_elapsed: 0,
            error: None,
        }
    }

    /// Create a tile-to-host transfer (for reading results).
    pub fn new_tile_to_host(
        tile_col: u8,
        tile_row: u8,
        tile_addr: u64,
        _host_addr: u64,
        length: u32,
    ) -> Self {
        // Address gen uses tile address since that's where we read
        let address_gen = AddressGenerator::new_1d(tile_addr, length, 1);

        Self {
            source: TransferEndpoint::TileMemory { col: tile_col, row: tile_row },
            dest: TransferEndpoint::HostMemory,
            direction: TransferDirection::MM2S,
            state: TransferState::Active,
            address_gen,
            bd_index: 0,
            channel: 0,
            acquire_lock: None,
            acquire_value: 0,
            release_lock: None,
            release_value: 0,
            next_bd: None,
            tlast_suppress: false,
            enable_packet: false,
            packet_id: 0,
            packet_type: 0,
            out_of_order_bd_id: 0,
            tile_col,
            tile_row,
            packet_header_sent: false,
            bytes_transferred: 0,
            total_bytes: length as u64,
            cycles_elapsed: 0,
            error: None,
        }
    }

    /// Check if transfer is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, TransferState::Complete)
    }

    /// Check if transfer is actively moving data.
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self.state, TransferState::Active)
    }

    /// Check if transfer needs processing (timing ticks).
    ///
    /// Returns true for states that require the timing state machine to advance:
    /// - `Active`: Data transfer phase
    /// - `ReleasingLock`: Lock release phase
    #[inline]
    pub fn needs_processing(&self) -> bool {
        matches!(self.state, TransferState::Active | TransferState::ReleasingLock(_))
    }

    /// Check if transfer is waiting for a lock.
    #[inline]
    pub fn is_waiting_for_lock(&self) -> bool {
        matches!(self.state, TransferState::WaitingForLock(_))
    }

    /// Get the lock acquisition mode, if a lock is configured.
    ///
    /// Returns `None` if no acquire lock is configured, otherwise returns
    /// the [`LockAcquireMode`] based on the BD's `acquire_value` convention.
    pub fn acquire_mode(&self) -> Option<LockAcquireMode> {
        self.acquire_lock.map(|_| LockAcquireMode::from_bd_value(self.acquire_value))
    }

    /// Check if transfer has an error.
    #[inline]
    pub fn has_error(&self) -> bool {
        matches!(self.state, TransferState::Error)
    }

    /// Get current address for the transfer.
    #[inline]
    pub fn current_address(&self) -> u64 {
        self.address_gen.current()
    }

    /// Get progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_bytes == 0 {
            1.0
        } else {
            self.bytes_transferred as f32 / self.total_bytes as f32
        }
    }

    /// Get remaining bytes to transfer.
    #[inline]
    pub fn remaining_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.bytes_transferred)
    }

    /// Notify that lock was acquired.
    pub fn lock_acquired(&mut self) {
        if let TransferState::WaitingForLock(_) = self.state {
            self.state = TransferState::Active;
        }
    }

    /// Notify that data was transferred.
    pub fn data_transferred(&mut self, bytes: u64) {
        self.bytes_transferred += bytes;

        // Advance address generator once per 32-bit word transferred
        // AIE-ML DMA works in word units: the address generator has total_elements
        // equal to length_words, not length_bytes. Each advance produces the next
        // word address based on the BD's stride configuration.
        let words = (bytes / 4) as usize;
        for _ in 0..words {
            if self.address_gen.next().is_none() {
                break;
            }
        }

        // Check if transfer is complete
        if self.bytes_transferred >= self.total_bytes {
            if let Some(lock) = self.release_lock {
                self.state = TransferState::ReleasingLock(lock);
            } else {
                self.state = TransferState::Complete;
            }
        }
    }

    /// Notify that lock was released.
    pub fn lock_released(&mut self) {
        if let TransferState::ReleasingLock(_) = self.state {
            self.state = TransferState::Complete;
        }
    }

    /// Mark transfer as failed.
    pub fn set_error(&mut self, error: DmaError) {
        self.error = Some(error);
        self.state = TransferState::Error;
    }

    /// Increment cycle counter.
    #[inline]
    pub fn tick(&mut self) {
        self.cycles_elapsed += 1;
    }

    /// Generate packet header word for MM2S packet-switched streams.
    ///
    /// Packet header format (AM025):
    /// - Bit 31: Odd parity over bits 30:0
    /// - Bits 30:28: 3'b000
    /// - Bits 27:25: Source Column (3 bits)
    /// - Bits 24:22: Source Row (3 bits)
    /// - Bit 21: 1'b0
    /// - Bits 20:18: Packet_Type (from BD, 3 bits)
    /// - Bits 17:11: 7'b0000000
    /// - Bits 10:5: Stream_ID/Packet_ID (from BD, 6 bits - note: BD has 5 bits, MSB is 0)
    /// - Bits 4:0: Reserved 5'b00000
    ///
    /// Returns None if packet mode is disabled.
    pub fn generate_packet_header(&self) -> Option<u32> {
        if !self.enable_packet {
            return None;
        }

        let mut header: u32 = 0;

        // Bits 27:25: Source Column (3 bits)
        header |= ((self.tile_col as u32) & 0x7) << 25;

        // Bits 24:22: Source Row (3 bits)
        header |= ((self.tile_row as u32) & 0x7) << 22;

        // Bit 21: 0

        // Bits 20:18: Packet_Type (3 bits)
        header |= ((self.packet_type as u32) & 0x7) << 18;

        // Bits 17:12: Out_Of_Order_BD_ID (6 bits, for S2MM to select BD in OOO mode)
        // This is placed in the upper reserved bits for OOO identification
        header |= ((self.out_of_order_bd_id as u32) & 0x3F) << 12;

        // Bits 10:5: Stream_ID (6 bits, from packet_id which is 5 bits)
        header |= ((self.packet_id as u32) & 0x3F) << 5;

        // Bits 4:0: Reserved

        // Bit 31: Odd parity over bits 30:0
        let ones = (header & 0x7FFF_FFFF).count_ones();
        if ones % 2 == 0 {
            // Even number of ones - set parity bit to make it odd
            header |= 1 << 31;
        }

        Some(header)
    }

    /// Mark packet header as sent.
    pub fn mark_packet_header_sent(&mut self) {
        self.packet_header_sent = true;
    }

    /// Check if packet header needs to be sent.
    pub fn needs_packet_header(&self) -> bool {
        self.enable_packet && !self.packet_header_sent
    }
}

/// Parse an out-of-order BD ID from a packet header.
///
/// The OOO BD ID is stored in bits 17:12 of the packet header.
/// Returns the 6-bit BD ID that S2MM should use in out-of-order mode.
pub fn parse_ooo_bd_id_from_header(header: u32) -> u8 {
    ((header >> 12) & 0x3F) as u8
}

/// Parse source tile coordinates from a packet header.
///
/// Returns (col, row) extracted from header bits 27:25 and 24:22.
pub fn parse_source_tile_from_header(header: u32) -> (u8, u8) {
    let col = ((header >> 25) & 0x7) as u8;
    let row = ((header >> 22) & 0x7) as u8;
    (col, row)
}

/// Parse packet type from a packet header.
///
/// Returns the 3-bit packet type from bits 20:18.
pub fn parse_packet_type_from_header(header: u32) -> u8 {
    ((header >> 18) & 0x7) as u8
}

/// Parse stream/packet ID from a packet header.
///
/// Returns the 6-bit stream ID from bits 10:5.
pub fn parse_stream_id_from_header(header: u32) -> u8 {
    ((header >> 5) & 0x3F) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::addressing::DimensionConfig;

    fn simple_bd() -> BdConfig {
        BdConfig {
            base_addr: 0x1000,
            length: 256,
            valid: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_transfer_creation() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert_eq!(transfer.bd_index, 0);
        assert_eq!(transfer.channel, 0);
        assert_eq!(transfer.total_bytes, 256);
        assert!(transfer.is_active());
    }

    #[test]
    fn test_transfer_invalid_bd() {
        let mut bd = simple_bd();
        bd.valid = false;

        let result = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute);
        assert!(matches!(result, Err(DmaError::BdNotValid(0))));
    }

    #[test]
    fn test_transfer_with_acquire_lock() {
        let mut bd = simple_bd();
        bd.acquire_lock = Some(5);
        bd.acquire_value = 1;

        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();
        assert!(transfer.is_waiting_for_lock());
        assert!(matches!(transfer.state, TransferState::WaitingForLock(5)));
    }

    #[test]
    fn test_transfer_lifecycle() {
        let mut bd = simple_bd();
        bd.acquire_lock = Some(5);
        bd.release_lock = Some(5);

        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        // Initially waiting for lock
        assert!(transfer.is_waiting_for_lock());

        // Acquire lock
        transfer.lock_acquired();
        assert!(transfer.is_active());

        // Transfer data
        transfer.data_transferred(256);
        assert!(matches!(transfer.state, TransferState::ReleasingLock(5)));

        // Release lock
        transfer.lock_released();
        assert!(transfer.is_complete());
    }

    #[test]
    fn test_transfer_progress() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert_eq!(transfer.progress(), 0.0);

        transfer.data_transferred(128);
        assert_eq!(transfer.progress(), 0.5);

        transfer.data_transferred(128);
        assert_eq!(transfer.progress(), 1.0);
    }

    #[test]
    fn test_transfer_remaining() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert_eq!(transfer.remaining_bytes(), 256);

        transfer.data_transferred(100);
        assert_eq!(transfer.remaining_bytes(), 156);
    }

    #[test]
    fn test_transfer_direction_s2mm() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::S2MM, 1, 2, TileType::Compute).unwrap();

        assert!(matches!(transfer.source, TransferEndpoint::Stream { .. }));
        assert!(matches!(transfer.dest, TransferEndpoint::TileMemory { .. }));
    }

    #[test]
    fn test_transfer_direction_mm2s() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert!(matches!(transfer.source, TransferEndpoint::TileMemory { .. }));
        assert!(matches!(transfer.dest, TransferEndpoint::Stream { .. }));
    }

    #[test]
    fn test_host_to_tile_transfer() {
        let transfer = Transfer::new_host_to_tile(1, 2, 0x8000_0000, 0x1000, 512);

        assert!(matches!(transfer.source, TransferEndpoint::HostMemory));
        assert!(matches!(transfer.dest, TransferEndpoint::TileMemory { col: 1, row: 2 }));
        assert_eq!(transfer.total_bytes, 512);
    }

    #[test]
    fn test_tile_to_host_transfer() {
        let transfer = Transfer::new_tile_to_host(1, 2, 0x1000, 0x8000_0000, 512);

        assert!(matches!(transfer.source, TransferEndpoint::TileMemory { col: 1, row: 2 }));
        assert!(matches!(transfer.dest, TransferEndpoint::HostMemory));
        assert_eq!(transfer.total_bytes, 512);
    }

    #[test]
    fn test_transfer_error() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        transfer.set_error(DmaError::AddressOutOfBounds { address: 0xFFFF, limit: 0x1000 });

        assert!(transfer.has_error());
        assert!(matches!(transfer.state, TransferState::Error));
    }

    #[test]
    fn test_next_bd_chaining() {
        let bd = BdConfig::simple_1d(0x1000, 256).with_next(3);
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert_eq!(transfer.next_bd, Some(3));
    }

    #[test]
    fn test_packet_header_disabled_by_default() {
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert!(!transfer.enable_packet);
        assert!(!transfer.needs_packet_header());
        assert!(transfer.generate_packet_header().is_none());
    }

    #[test]
    fn test_packet_header_generation() {
        let mut bd = simple_bd();
        bd.enable_packet = true;
        bd.packet_id = 0x1F;     // 5-bit value, max
        bd.packet_type = 0x5;    // 3-bit value

        // Create transfer at tile (3, 5)
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 3, 5, TileType::Compute).unwrap();

        assert!(transfer.needs_packet_header());

        let header = transfer.generate_packet_header().unwrap();

        // Verify header format per AM025:
        // Bits 27:25 = Source Column (3) = 0b011
        // Bits 24:22 = Source Row (5)    = 0b101
        // Bits 20:18 = Packet_Type (5)   = 0b101
        // Bits 10:5  = Stream_ID (0x1F)  = 0b011111

        // Check source column (bits 27:25)
        let col = (header >> 25) & 0x7;
        assert_eq!(col, 3, "Source column should be 3");

        // Check source row (bits 24:22)
        let row = (header >> 22) & 0x7;
        assert_eq!(row, 5, "Source row should be 5");

        // Check packet type (bits 20:18)
        let pkt_type = (header >> 18) & 0x7;
        assert_eq!(pkt_type, 5, "Packet type should be 5");

        // Check stream ID (bits 10:5)
        let stream_id = (header >> 5) & 0x3F;
        assert_eq!(stream_id, 0x1F, "Stream ID should be 0x1F");

        // Verify odd parity
        let ones = header.count_ones();
        assert_eq!(ones % 2, 1, "Parity should be odd, got {} ones", ones);
    }

    #[test]
    fn test_packet_header_sent_tracking() {
        let mut bd = simple_bd();
        bd.enable_packet = true;
        bd.packet_id = 0x10;

        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        // Initially needs header
        assert!(transfer.needs_packet_header());
        assert!(!transfer.packet_header_sent);

        // Mark as sent
        transfer.mark_packet_header_sent();

        // No longer needs header
        assert!(!transfer.needs_packet_header());
        assert!(transfer.packet_header_sent);
    }

    #[test]
    fn test_lock_acquire_mode_conversion() {
        // Positive values -> Equal mode
        assert_eq!(LockAcquireMode::from_bd_value(1), LockAcquireMode::Equal(1));
        assert_eq!(LockAcquireMode::from_bd_value(5), LockAcquireMode::Equal(5));

        // Negative values -> GreaterEqual mode
        assert_eq!(LockAcquireMode::from_bd_value(-1), LockAcquireMode::GreaterEqual(1));
        assert_eq!(LockAcquireMode::from_bd_value(-5), LockAcquireMode::GreaterEqual(5));

        // Zero -> Simple mode
        assert_eq!(LockAcquireMode::from_bd_value(0), LockAcquireMode::Simple);

        // Round-trip conversion
        assert_eq!(LockAcquireMode::Equal(3).to_bd_value(), 3);
        assert_eq!(LockAcquireMode::GreaterEqual(3).to_bd_value(), -3);
        assert_eq!(LockAcquireMode::Simple.to_bd_value(), 0);

        // Threshold values
        assert_eq!(LockAcquireMode::Equal(5).threshold(), 5);
        assert_eq!(LockAcquireMode::GreaterEqual(7).threshold(), 7);
        assert_eq!(LockAcquireMode::Simple.threshold(), 1);
    }

    #[test]
    fn test_transfer_acquire_mode() {
        // Transfer without lock
        let bd = simple_bd();
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();
        assert!(transfer.acquire_mode().is_none());

        // Transfer with lock in Equal mode
        let mut bd_locked = simple_bd();
        bd_locked.acquire_lock = Some(5);
        bd_locked.acquire_value = 1;
        let transfer_locked = Transfer::new(&bd_locked, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();
        assert_eq!(transfer_locked.acquire_mode(), Some(LockAcquireMode::Equal(1)));

        // Transfer with lock in GE mode
        let mut bd_ge = simple_bd();
        bd_ge.acquire_lock = Some(5);
        bd_ge.acquire_value = -2;
        let transfer_ge = Transfer::new(&bd_ge, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();
        assert_eq!(transfer_ge.acquire_mode(), Some(LockAcquireMode::GreaterEqual(2)));
    }
}
