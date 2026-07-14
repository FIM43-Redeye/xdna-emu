//! DMA transfer data carrier.
//!
//! A `Transfer` represents an active DMA operation from a single buffer
//! descriptor. It tracks the current position, handles multi-dimensional
//! addressing, and provides data queries for the channel FSM.
//!
//! State transitions (lock acquire/release, completion) are managed by the
//! `ChannelFsm` in `channel.rs` -- Transfer is a pure data carrier that
//! only advances its internal counters and address generator.
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

use super::super::addressing::AddressGenerator;
use super::super::{BdConfig, DmaError};
use super::padding::{PadAction, ZeroPadState};
use xdna_archspec::types::TileKind;

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
    TileMemory { col: u8, row: u8 },
    /// Host/DDR memory (via shim tile)
    HostMemory,
    /// Stream switch connection
    Stream { port: u8 },
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

    /// Zero-padding state machine (MemTile MM2S only).
    /// When Some, the transfer inserts zero words at dimension boundaries.
    pub zero_pad_state: Option<ZeroPadState>,

    /// Byte address of the last word this transfer moved to/from tile memory,
    /// or None before the first word.
    ///
    /// The DMA's memory-side access granule is one bank width (128 bits), not
    /// one word: a word only costs a bank access if it opens a granule the
    /// previous word did not already fetch/stage. That predicate needs the
    /// PREVIOUS address, which the address generator does not retain -- so the
    /// transfer carries it. Padding words never reach memory and never update
    /// it. Advanced only in `advance()`, i.e. only when a word actually moved,
    /// so a bank-arbitration-denied channel (whose whole FSM step is skipped)
    /// leaves it untouched and re-presents the identical demand next cycle.
    pub last_access_addr: Option<u64>,

    /// MM2S only: words fetched from memory into the channel's egress staging
    /// FIFO but not yet handed to the stream port.
    ///
    /// The memory side reads a whole 16-byte granule in one bank slot; the
    /// stream side drains one 32-bit word per cycle. This is the occupancy
    /// between them, and it is what lets the DMA lose a bank arbitration
    /// without bubbling the stream. Nothing is buffered here: the words are
    /// still read from tile memory at drain time, which is identical data
    /// because the BD's lock holds the buffer exclusive for the whole
    /// transfer -- this counts the staging, it does not duplicate it.
    ///
    /// The fetch position is always `address_gen.current()` advanced by
    /// `staged_words`, so draining a word (address_gen += 1, staged -= 1)
    /// leaves the next granule fetch's address -- and therefore its bank
    /// demand -- unchanged. That is what keeps a bank-denied channel
    /// re-presenting the identical demand next cycle.
    pub staged_words: usize,

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
        tile_kind: TileKind,
    ) -> Result<Self, DmaError> {
        if !bd_config.valid {
            return Err(DmaError::BdNotValid(bd_index));
        }

        // Create address generator based on BD dimensions and iteration config
        let address_gen = AddressGenerator::with_iteration(
            bd_config.base_addr,
            [bd_config.d0, bd_config.d1, bd_config.d2, bd_config.d3],
            bd_config.iteration,
        );

        // Total transfer size: BD length (per-iteration) * number of iterations.
        // Buffer_Length sets the per-iteration data volume; iteration repeats
        // the entire dimensional pattern with an address offset.
        let data_bytes = bd_config.length as u64 * bd_config.iteration.total_iterations() as u64;

        // Zero-padding adds extra output words for MemTile MM2S only.
        // The padding state machine tracks where to insert zeros.
        let pad = &bd_config.zero_padding;
        let zero_pad_state = if pad.is_enabled() && direction == TransferDirection::MM2S && tile_kind.is_mem()
        {
            Some(ZeroPadState::new(
                *pad,
                bd_config.d0.effective_size(),
                bd_config.d1.effective_size(),
                bd_config.d2.effective_size(),
            ))
        } else {
            None
        };

        // For padded MemTile MM2S: Buffer_Length already includes both data
        // and padding words (the CDO lowering sets length = data + padding).
        // Don't add padding on top or it gets double-counted, causing extra
        // zero words to bleed into subsequent BD firings.
        let total_bytes = data_bytes;

        // Determine endpoints based on direction and tile type
        let is_shim = tile_kind.is_shim();

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

        Ok(Self {
            source,
            dest,
            direction,
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
            total_bytes,
            cycles_elapsed: 0,
            last_access_addr: None,
            staged_words: 0,
            zero_pad_state,
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
            last_access_addr: None,
            staged_words: 0,
            zero_pad_state: None,
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
            last_access_addr: None,
            staged_words: 0,
            zero_pad_state: None,
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
            last_access_addr: None,
            staged_words: 0,
            zero_pad_state: None,
            error: None,
        }
    }

    /// Get the lock acquisition mode, if a lock is configured.
    ///
    /// Returns `None` if no acquire lock is configured, otherwise returns
    /// the [`LockAcquireMode`] based on the BD's `acquire_value` convention.
    pub fn acquire_mode(&self) -> Option<LockAcquireMode> {
        self.acquire_lock.map(|_| LockAcquireMode::from_bd_value(self.acquire_value))
    }

    /// Get current address for the transfer.
    #[inline]
    pub fn current_address(&self) -> u64 {
        self.address_gen.current()
    }

    /// Get the next output action for this transfer.
    ///
    /// When zero-padding is active, returns `PadAction::Zero` for padding
    /// positions and `PadAction::Data(addr)` for real data positions.
    /// Without padding, always returns `PadAction::Data(addr)`.
    pub fn next_output_action(&self) -> PadAction {
        if let Some(ref pad) = self.zero_pad_state {
            pad.current_action(&self.address_gen)
        } else {
            PadAction::Data(self.address_gen.current())
        }
    }

    /// Whether TLAST should be suppressed at the end of this transfer.
    ///
    /// In packet mode (`enable_packet = true`), TLAST is ALWAYS asserted at
    /// BD boundaries regardless of the BD's TLAST_Suppress field. The hardware
    /// requires TLAST to delineate packets for stream switch arbiter release.
    /// Without TLAST, packet arbiters lock permanently, blocking other packets
    /// that share the same arbiter.
    ///
    /// TLAST_Suppress only takes effect for circuit-switched (non-packet) MM2S
    /// transfers, where it allows multiple chained BDs to appear as a single
    /// continuous stream without TLAST gaps.
    #[inline]
    pub fn effective_tlast_suppress(&self) -> bool {
        self.tlast_suppress && !self.enable_packet
    }

    /// Whether zero-padding is active for this transfer.
    #[inline]
    pub fn has_zero_padding(&self) -> bool {
        self.zero_pad_state.is_some()
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

    /// Whether this transfer touches host DDR memory (via shim tile NoC).
    /// Used to determine if extra pipeline latency should be applied.
    pub fn involves_host_memory(&self) -> bool {
        matches!(self.source, TransferEndpoint::HostMemory)
            || matches!(self.dest, TransferEndpoint::HostMemory)
    }

    /// Whether this transfer crosses a stream-switch port (MM2S egress or
    /// S2MM ingress).  Such transfers are rate-limited by the 32-bit
    /// AXI4-Stream beat width (`stream_words_per_cycle`), not the wider tile
    /// data-memory bus.
    pub fn involves_stream(&self) -> bool {
        matches!(self.source, TransferEndpoint::Stream { .. })
            || matches!(self.dest, TransferEndpoint::Stream { .. })
    }

    /// Advance the transfer by the given number of bytes.
    ///
    /// Updates `bytes_transferred`, advances the address generator, and
    /// steps the zero-padding state machine if active. Does NOT perform
    /// any state transitions -- that responsibility belongs to the
    /// channel FSM.
    ///
    /// The caller should use `next_output_action()` to determine what
    /// each word should be before calling this.
    pub fn advance(&mut self, bytes: u64) {
        self.bytes_transferred += bytes;

        let words = (bytes / 4) as usize;

        if let Some(ref mut pad_state) = self.zero_pad_state {
            // With zero-padding: advance state machine per word.
            // The state machine tells us whether to advance the address
            // generator (data word) or skip it (padding zero).
            for _ in 0..words {
                let advance_addr = pad_state.advance();
                if advance_addr {
                    self.last_access_addr = Some(self.address_gen.current());
                    self.address_gen.next();
                }
            }
        } else {
            // Without padding: advance address generator for every word.
            // AIE-ML DMA works in word units: the address generator has
            // total_elements equal to length_words, not length_bytes.
            for _ in 0..words {
                self.last_access_addr = Some(self.address_gen.current());
                if self.address_gen.next().is_none() {
                    break;
                }
            }
        }
    }

    /// Record an error on this transfer.
    pub fn set_error(&mut self, error: DmaError) {
        self.error = Some(error);
    }

    /// Increment cycle counter.
    #[inline]
    pub fn tick(&mut self) {
        self.cycles_elapsed += 1;
    }

    /// Generate packet header word for MM2S packet-switched streams.
    ///
    /// Uses the AM020 Ch2 Table 2 format via `PacketHeader::encode()`:
    /// | 31    | 30-28 | 27-21      | 20-16     | 15  | 14-12       | 11-5    | 4-0       |
    /// | Parity| Rsvd  | Src Column | Src Row   | Rsvd| Packet Type | Rsvd    | Stream ID |
    ///
    /// Returns None if packet mode is disabled.
    pub fn generate_packet_header(&self) -> Option<u32> {
        if !self.enable_packet {
            return None;
        }

        use crate::device::stream_switch::{PacketHeader, PacketType};
        let header = PacketHeader::new(self.packet_id, self.tile_col, self.tile_row)
            .with_type(PacketType::from_u8(self.packet_type));
        Some(header.encode())
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

/// Parse source tile coordinates from a packet header (AM020 Table 2).
///
/// Returns (col, row) extracted from header bits 27:21 and 20:16.
pub fn parse_source_tile_from_header(header: u32) -> (u8, u8) {
    use xdna_archspec::aie2::packet;
    let col = ((header >> packet::SRC_COL_SHIFT) & packet::SRC_COL_MASK) as u8;
    let row = ((header >> packet::SRC_ROW_SHIFT) & packet::SRC_ROW_MASK) as u8;
    (col, row)
}

/// Parse packet type from a packet header (AM020 Table 2).
///
/// Returns the 3-bit packet type from bits 14:12.
pub fn parse_packet_type_from_header(header: u32) -> u8 {
    use xdna_archspec::aie2::packet;
    ((header >> packet::TYPE_SHIFT) & packet::TYPE_MASK) as u8
}

/// Parse stream/packet ID from a packet header (AM020 Table 2).
///
/// Returns the 5-bit stream ID from bits 4:0.
pub fn parse_stream_id_from_header(header: u32) -> u8 {
    use xdna_archspec::aie2::packet;
    (header & packet::STREAM_ID_MASK) as u8
}
