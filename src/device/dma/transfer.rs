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

use super::addressing::{AddressGenerator, ZeroPadConfig};
use super::{BdConfig, DmaError};
use crate::device::tile::TileType;

/// What a padded transfer should output on the next cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadAction {
    /// Read a data word from memory at the given address.
    Data(u64),
    /// Emit a zero word (padding, no memory read).
    Zero,
}

/// Phase within the zero-padding state machine.
///
/// The output pattern for a padded transfer follows nested dimension loops:
///
/// ```text
/// for each D2 iteration:
///   [d2_before zeros]
///   for each D1 iteration:
///     [d1_before zeros]
///     [d0_before zeros] [d0_size data words] [d0_after zeros]
///     [d1_after zeros]
///   [d2_after zeros]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PadPhase {
    D2Before,
    D1Before,
    D0Before,
    D0Data,
    D0After,
    D1After,
    D2After,
    Done,
}

/// Tracks zero-padding state for a MemTile MM2S transfer.
///
/// Wraps around the address generator: on each `next()` call, returns either
/// `PadAction::Data(addr)` (read from memory) or `PadAction::Zero` (emit zero).
/// The address generator only advances for data words.
///
/// Per AM025, the padding dimensions have different units:
/// - D0 before/after: individual 32-bit zero words
/// - D1 before/after: "wraps of dim0" (complete D0 output rows of zeros)
/// - D2 before/after: "wraps of dim0dim1" (complete D1 output blocks of zeros)
#[derive(Debug, Clone)]
pub struct ZeroPadState {
    config: ZeroPadConfig,
    phase: PadPhase,
    /// Remaining words in the current phase
    phase_remaining: u32,
    /// D0 data size (from dimension config)
    d0_size: u32,
    /// Size of one complete D0 output row (d0_before + d0_size + d0_after)
    d0_wrap_size: u32,
    /// D1 data iteration count
    d1_size: u32,
    /// Total D1 iterations including padding (d1_before + d1_size + d1_after)
    d1_total: u32,
    /// D2 iteration count
    d2_size: u32,
    /// Current D1 counter (0..d1_size, data iterations only)
    d1_counter: u32,
    /// Current D2 counter (0..d2_size)
    d2_counter: u32,
    /// Total output words (data + padding) for the whole transfer
    total_output_words: u64,
    /// Output words emitted so far
    words_emitted: u64,
}

impl ZeroPadState {
    /// Create a new padding state from BD dimensions and padding config.
    pub fn new(
        config: ZeroPadConfig,
        d0_size: u32,
        d1_size: u32,
        d2_size: u32,
    ) -> Self {
        let d0_eff = if d0_size == 0 { 1 } else { d0_size };
        let d1_eff = if d1_size == 0 { 1 } else { d1_size };
        let d2_eff = if d2_size == 0 { 1 } else { d2_size };

        // D0 wrap = one complete D0 output row including D0 padding
        let d0_wrap = config.d0_before as u32 + d0_eff + config.d0_after as u32;
        // D1 total = data iterations + D1 padding iterations
        let d1_total = config.d1_before as u32 + d1_eff + config.d1_after as u32;

        let data_words = d0_eff as u64 * d1_eff as u64 * d2_eff as u64;
        let pad_words = config.total_pad_words(d0_eff, d1_eff, d2_eff);
        let total = data_words + pad_words;

        // Start at D2Before if there are D2-level before zeros, otherwise
        // cascade down through the phase hierarchy
        let (phase, phase_remaining) = if config.d2_before > 0 {
            // D2 before: emit d2_before complete D1 blocks of zeros
            (PadPhase::D2Before, config.d2_before as u32 * d1_total * d0_wrap)
        } else {
            Self::enter_d1_iteration(&config, d0_eff, d0_wrap)
        };

        Self {
            config,
            phase,
            phase_remaining,
            d0_size: d0_eff,
            d0_wrap_size: d0_wrap,
            d1_size: d1_eff,
            d1_total,
            d2_size: d2_eff,
            d1_counter: 0,
            d2_counter: 0,
            total_output_words: total,
            words_emitted: 0,
        }
    }

    /// Total output words (data + padding).
    pub fn total_output_words(&self) -> u64 {
        self.total_output_words
    }

    /// Whether all output words have been emitted.
    pub fn is_finished(&self) -> bool {
        self.phase == PadPhase::Done || self.words_emitted >= self.total_output_words
    }

    /// Remaining output words.
    pub fn remaining(&self) -> u64 {
        self.total_output_words.saturating_sub(self.words_emitted)
    }

    /// Get the next output action without advancing state.
    pub fn current_action(&self, addr_gen: &AddressGenerator) -> PadAction {
        match self.phase {
            PadPhase::D0Data => PadAction::Data(addr_gen.current()),
            PadPhase::Done => PadAction::Zero, // shouldn't be called, but safe
            _ => PadAction::Zero,
        }
    }

    /// Advance the state machine by one output word.
    ///
    /// Returns true if the address generator should also be advanced
    /// (i.e., the word was data, not padding).
    pub fn advance(&mut self) -> bool {
        if self.phase == PadPhase::Done {
            return false;
        }

        self.words_emitted += 1;
        let is_data = self.phase == PadPhase::D0Data;

        self.phase_remaining = self.phase_remaining.saturating_sub(1);
        if self.phase_remaining == 0 {
            self.transition();
        }

        is_data
    }

    /// Transition to the next phase when the current phase is exhausted.
    fn transition(&mut self) {
        match self.phase {
            PadPhase::D2Before => {
                // D2 before padding done, start first D1 iteration
                let (phase, remaining) = Self::enter_d1_iteration(
                    &self.config, self.d0_size, self.d0_wrap_size,
                );
                self.phase = phase;
                self.phase_remaining = remaining;
            }
            PadPhase::D1Before => {
                // D1 before padding done, start D0 data pattern
                self.phase = if self.config.d0_before > 0 {
                    PadPhase::D0Before
                } else {
                    PadPhase::D0Data
                };
                self.phase_remaining = if self.config.d0_before > 0 {
                    self.config.d0_before as u32
                } else {
                    self.d0_size
                };
            }
            PadPhase::D0Before => {
                self.phase = PadPhase::D0Data;
                self.phase_remaining = self.d0_size;
            }
            PadPhase::D0Data => {
                if self.config.d0_after > 0 {
                    self.phase = PadPhase::D0After;
                    self.phase_remaining = self.config.d0_after as u32;
                } else {
                    self.advance_d1_data_iter();
                }
            }
            PadPhase::D0After => {
                // D0 complete for this D1 data iteration.
                // Advance to next D1 data iteration, or emit D1 after.
                self.advance_d1_data_iter();
            }
            PadPhase::D1After => {
                // D1 after padding done (all D1 data iterations complete).
                // Advance to the next D2 iteration.
                self.finish_d2_iteration();
            }
            PadPhase::D2After => {
                self.d2_counter += 1;
                if self.d2_counter < self.d2_size {
                    // Start next D2 iteration
                    if self.config.d2_before > 0 {
                        self.phase = PadPhase::D2Before;
                        self.phase_remaining = self.config.d2_before as u32
                            * self.d1_total * self.d0_wrap_size;
                    } else {
                        let (phase, remaining) = Self::enter_d1_iteration(
                            &self.config, self.d0_size, self.d0_wrap_size,
                        );
                        self.phase = phase;
                        self.phase_remaining = remaining;
                    }
                    self.d1_counter = 0;
                } else {
                    self.phase = PadPhase::Done;
                    self.phase_remaining = 0;
                }
            }
            PadPhase::Done => {}
        }
    }

    /// Advance after completing one D1 data iteration's D0 pattern.
    ///
    /// If more D1 data iterations remain, start the next D0 pattern.
    /// If all D1 data iterations are done, emit D1 after padding (or
    /// advance to the next D2 iteration if no D1 after padding).
    fn advance_d1_data_iter(&mut self) {
        self.d1_counter += 1;
        if self.d1_counter < self.d1_size {
            // More D1 data iterations -- start next D0 pattern directly
            // (D1 before was already emitted once at the start of this D1 block)
            self.phase = if self.config.d0_before > 0 {
                PadPhase::D0Before
            } else {
                PadPhase::D0Data
            };
            self.phase_remaining = if self.config.d0_before > 0 {
                self.config.d0_before as u32
            } else {
                self.d0_size
            };
        } else {
            // All D1 data iterations done -- emit D1 after padding
            if self.config.d1_after > 0 {
                self.phase = PadPhase::D1After;
                // Each D1 after unit = one complete D0 output row of zeros
                self.phase_remaining = self.config.d1_after as u32 * self.d0_wrap_size;
            } else {
                self.finish_d2_iteration();
            }
        }
    }

    /// Called when all data D1 iterations within a D2 iteration are done.
    fn finish_d2_iteration(&mut self) {
        if self.config.d2_after > 0 {
            self.phase = PadPhase::D2After;
            // Each D2 after unit = one complete D1 block of zeros
            self.phase_remaining = self.config.d2_after as u32
                * self.d1_total * self.d0_wrap_size;
        } else {
            self.d2_counter += 1;
            if self.d2_counter < self.d2_size {
                self.d1_counter = 0;
                if self.config.d2_before > 0 {
                    self.phase = PadPhase::D2Before;
                    self.phase_remaining = self.config.d2_before as u32
                        * self.d1_total * self.d0_wrap_size;
                } else {
                    let (phase, remaining) = Self::enter_d1_iteration(
                        &self.config, self.d0_size, self.d0_wrap_size,
                    );
                    self.phase = phase;
                    self.phase_remaining = remaining;
                }
            } else {
                self.phase = PadPhase::Done;
                self.phase_remaining = 0;
            }
        }
    }

    /// Determine initial phase when entering a new D1 iteration.
    ///
    /// D1 before padding = d1_before complete D0 wraps of zeros (per AM025:
    /// "wraps of dim0 before dim1").
    fn enter_d1_iteration(
        config: &ZeroPadConfig, d0_size: u32, d0_wrap_size: u32,
    ) -> (PadPhase, u32) {
        if config.d1_before > 0 {
            // Each D1 before unit = one complete D0 output row of zeros
            (PadPhase::D1Before, config.d1_before as u32 * d0_wrap_size)
        } else if config.d0_before > 0 {
            (PadPhase::D0Before, config.d0_before as u32)
        } else {
            (PadPhase::D0Data, d0_size)
        }
    }
}

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

        // Total transfer size: BD length (per-iteration) * number of iterations.
        // Buffer_Length sets the per-iteration data volume; iteration repeats
        // the entire dimensional pattern with an address offset.
        let data_bytes = bd_config.length as u64
            * bd_config.iteration.total_iterations() as u64;

        // Zero-padding adds extra output words for MemTile MM2S only.
        // The padding state machine tracks where to insert zeros.
        let pad = &bd_config.zero_padding;
        let zero_pad_state = if pad.is_enabled()
            && direction == TransferDirection::MM2S
            && tile_type == TileType::MemTile
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

        // Total output includes padding words (each is 4 bytes)
        let pad_bytes = zero_pad_state.as_ref()
            .map(|s| (s.total_output_words() - address_gen.total_elements()) * 4)
            .unwrap_or(0);
        let total_bytes = data_bytes + pad_bytes;

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
                    self.address_gen.next();
                }
            }
        } else {
            // Without padding: advance address generator for every word.
            // AIE-ML DMA works in word units: the address generator has
            // total_elements equal to length_words, not length_bytes.
            for _ in 0..words {
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
    use crate::arch::packet;
    let col = ((header >> packet::SRC_COL_SHIFT) & packet::SRC_COL_MASK) as u8;
    let row = ((header >> packet::SRC_ROW_SHIFT) & packet::SRC_ROW_MASK) as u8;
    (col, row)
}

/// Parse packet type from a packet header (AM020 Table 2).
///
/// Returns the 3-bit packet type from bits 14:12.
pub fn parse_packet_type_from_header(header: u32) -> u8 {
    use crate::arch::packet;
    ((header >> packet::TYPE_SHIFT) & packet::TYPE_MASK) as u8
}

/// Parse stream/packet ID from a packet header (AM020 Table 2).
///
/// Returns the 5-bit stream ID from bits 4:0.
pub fn parse_stream_id_from_header(header: u32) -> u8 {
    use crate::arch::packet;
    (header & packet::STREAM_ID_MASK) as u8
}

#[cfg(test)]
mod tests {
    use super::*;


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
        assert_eq!(transfer.remaining_bytes(), 256);
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
        assert_eq!(transfer.acquire_lock, Some(5));
        assert_eq!(transfer.acquire_value, 1);
        assert_eq!(transfer.acquire_mode(), Some(LockAcquireMode::Equal(1)));
    }

    #[test]
    fn test_transfer_advance_to_completion() {
        let mut bd = simple_bd();
        bd.acquire_lock = Some(5);
        bd.release_lock = Some(5);

        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        // Lock config is stored on the transfer
        assert_eq!(transfer.acquire_lock, Some(5));
        assert_eq!(transfer.release_lock, Some(5));

        // Advance all data
        transfer.advance(256);
        assert_eq!(transfer.remaining_bytes(), 0);
        assert_eq!(transfer.bytes_transferred, 256);
    }

    #[test]
    fn test_transfer_progress() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert_eq!(transfer.progress(), 0.0);

        transfer.advance(128);
        assert_eq!(transfer.progress(), 0.5);

        transfer.advance(128);
        assert_eq!(transfer.progress(), 1.0);
    }

    #[test]
    fn test_transfer_remaining() {
        let bd = simple_bd();
        let mut transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute).unwrap();

        assert_eq!(transfer.remaining_bytes(), 256);

        transfer.advance(100);
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

        assert!(transfer.error.is_some());
        assert!(matches!(transfer.error, Some(DmaError::AddressOutOfBounds { .. })));
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
        bd.packet_type = 0x3;    // 3-bit value (Trace)

        // Create transfer at tile (3, 5)
        let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 3, 5, TileType::Compute).unwrap();

        assert!(transfer.needs_packet_header());

        let header = transfer.generate_packet_header().unwrap();

        // Verify header format per AM020 Ch2, Table 2:
        // | 31    | 30-28 | 27-21      | 20-16     | 15  | 14-12       | 11-5    | 4-0       |
        // | Parity| Rsvd  | Src Column | Src Row   | Rsvd| Packet Type | Rsvd    | Stream ID |

        // Check source column (bits 27:21, 7-bit field)
        let col = (header >> 21) & 0x7F;
        assert_eq!(col, 3, "Source column should be 3");

        // Check source row (bits 20:16, 5-bit field)
        let row = (header >> 16) & 0x1F;
        assert_eq!(row, 5, "Source row should be 5");

        // Check packet type (bits 14:12, 3-bit field)
        let pkt_type = (header >> 12) & 0x7;
        assert_eq!(pkt_type, 3, "Packet type should be 3 (Trace)");

        // Check stream ID (bits 4:0, 5-bit field)
        let stream_id = header & 0x1F;
        assert_eq!(stream_id, 0x1F, "Stream ID should be 0x1F");

        // Verify odd parity
        let ones = header.count_ones();
        assert_eq!(ones % 2, 1, "Parity should be odd, got {} ones", ones);

        // Verify round-trip via PacketHeader::decode
        let (decoded, parity_ok) = crate::device::stream_switch::PacketHeader::decode(header);
        assert!(parity_ok, "Parity check should pass");
        assert_eq!(decoded.stream_id, 0x1F);
        assert_eq!(decoded.src_col, 3);
        assert_eq!(decoded.src_row, 5);

        // Verify parse helper functions
        assert_eq!(super::parse_stream_id_from_header(header), 0x1F);
        assert_eq!(super::parse_packet_type_from_header(header), 3);
        assert_eq!(super::parse_source_tile_from_header(header), (3, 5));
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

    // --- ZeroPadState tests ---

    #[test]
    fn test_pad_state_no_padding() {
        // No padding configured: all output should be data
        let config = ZeroPadConfig::default();
        let mut state = ZeroPadState::new(config, 4, 1, 1);

        // With no padding, total output = data words only
        assert_eq!(state.total_output_words(), 4);

        let gen = AddressGenerator::new_1d(0x1000, 4, 4);
        for _ in 0..4 {
            assert!(matches!(state.current_action(&gen), PadAction::Data(_)));
            state.advance();
        }
        assert!(state.is_finished());
    }

    #[test]
    fn test_pad_state_d0_only() {
        // D0 before=2, after=1, d0_size=3, d1=1, d2=1
        // Expected output: [0,0, D,D,D, 0] = 6 words total
        let config = ZeroPadConfig {
            d0_before: 2,
            d0_after: 1,
            ..Default::default()
        };
        let mut state = ZeroPadState::new(config, 3, 1, 1);
        let gen = AddressGenerator::new_1d(0x1000, 3, 4);

        assert_eq!(state.total_output_words(), 6);

        // Collect output sequence
        let mut actions = Vec::new();
        while !state.is_finished() {
            actions.push(state.current_action(&gen));
            state.advance();
        }

        assert_eq!(actions.len(), 6);
        assert_eq!(actions[0], PadAction::Zero);  // d0_before
        assert_eq!(actions[1], PadAction::Zero);  // d0_before
        assert!(matches!(actions[2], PadAction::Data(_)));  // data
        assert!(matches!(actions[3], PadAction::Data(_)));  // data
        assert!(matches!(actions[4], PadAction::Data(_)));  // data
        assert_eq!(actions[5], PadAction::Zero);  // d0_after
    }

    #[test]
    fn test_pad_state_d0_with_d1_iterations() {
        // D0 before=1, after=1, d0_size=2, d1_size=3, d2=1
        // Per D1 iteration: [0, D,D, 0] = 4 words
        // Total: 4 * 3 = 12 words
        let config = ZeroPadConfig {
            d0_before: 1,
            d0_after: 1,
            ..Default::default()
        };
        let mut state = ZeroPadState::new(config, 2, 3, 1);
        let gen = AddressGenerator::new_2d(0x1000, 2, 4, 3, 8);

        assert_eq!(state.total_output_words(), 12);

        let mut actions = Vec::new();
        while !state.is_finished() {
            actions.push(state.current_action(&gen));
            state.advance();
        }

        // Verify structure: 3 repetitions of [Zero, Data, Data, Zero]
        for iter in 0..3 {
            let base = iter * 4;
            assert_eq!(actions[base], PadAction::Zero, "d0_before at D1 iter {}", iter);
            assert!(matches!(actions[base + 1], PadAction::Data(_)), "data at D1 iter {}", iter);
            assert!(matches!(actions[base + 2], PadAction::Data(_)), "data at D1 iter {}", iter);
            assert_eq!(actions[base + 3], PadAction::Zero, "d0_after at D1 iter {}", iter);
        }
    }

    #[test]
    fn test_pad_state_all_dimensions() {
        // D0 before=1, d0_size=2, D1 before=1, after=1, d1=2, D2 before=1, after=1, d2=1
        //
        // Per AM025:
        // - D0 padding: individual words
        // - D1 padding: "wraps of dim0" (complete D0 rows of zeros)
        // - D2 padding: "wraps of dim0dim1" (complete D1 blocks of zeros)
        //
        // d0_wrap = 1+2+0 = 3 words per D0 output row
        // d1_total = 1+2+1 = 4 D0 wraps per D1 block (including D1 padding)
        //
        // D2 before = 1 D1 block = 4 * 3 = 12 zero words
        // D1 iteration 0:
        //   D1 before = 1 D0 wrap = 3 zero words
        //   D0 data row: [0, D, D] = 3 words (1 d0_before zero + 2 data)
        //   D1 after = 1 D0 wrap = 3 zero words
        // D1 iteration 1: same as iter 0
        // D2 after = 1 D1 block = 12 zero words
        //
        // Total = 12 + (3+3+3)*2 + 12 = 12 + 18 + 12 = 42 words
        let config = ZeroPadConfig {
            d0_before: 1,
            d0_after: 0,
            d1_before: 1,
            d1_after: 1,
            d2_before: 1,
            d2_after: 1,
        };
        let mut state = ZeroPadState::new(config, 2, 2, 1);
        let gen = AddressGenerator::new_2d(0x1000, 2, 4, 2, 8);

        // total = (1+1+1)*(1+2+1)*(1+2+0) = 3*4*3 = 36; data = 2*2*1 = 4; pad = 36-4 = 32
        // Hmm, total output should be d2_total * d1_total * d0_wrap = 3 * 4 * 3 = 36
        assert_eq!(state.total_output_words(), 36);

        let mut sequence = Vec::new();
        while !state.is_finished() {
            let is_data = matches!(state.current_action(&gen), PadAction::Data(_));
            sequence.push(if is_data { 'D' } else { '0' });
            state.advance();
        }

        let pattern: String = sequence.into_iter().collect();
        // D2 before: 12 zeros
        // D1 iter 0: d1_before(3 zeros) + d0_before(1)+data(2) + d1_after(3 zeros) = 9 words
        // D1 iter 1: same = 9 words
        // D2 after: 12 zeros
        // Total: 12 + 9 + 9 + 12 = 42... but total_output says 36?
        //
        // Wait: total_output = d2_total * d1_total * d0_wrap = 3 * 4 * 3 = 36
        // D2 before: 1 D1 block = d1_total * d0_wrap = 4 * 3 = 12 zeros
        // Data D1 iterations: 2 * (d1_before_wraps + d0_row + d1_after_wraps)
        //   = 2 * (1*3 + 3 + 1*3) = 2 * 9 = 18 words (but only 4 are data)
        // D2 after: 12 zeros
        // Total: 12 + 18 + 12 = 42 -- but formula says 36!
        //
        // The discrepancy is because d1_total already includes d1_before+d1_after,
        // so the data D1 iterations should NOT add their own d1_before/d1_after again.
        // Actually d1_total = d1_before + d1_size + d1_after = 1+2+1 = 4
        // And d2_total = d2_before + d2_size + d2_after = 1+1+1 = 3
        // total = 3 * 4 * 3 = 36.
        // This means: 3 D2 blocks, each containing 4 D0 wraps, each of 3 words.
        // D2 block 0 (d2_before): all zeros = 4*3 = 12 zeros
        // D2 block 1 (data):
        //   D1 wrap 0 (d1_before): 3 zeros
        //   D1 wrap 1 (data iter 0): 1 zero + 2 data = 0DD
        //   D1 wrap 2 (data iter 1): 1 zero + 2 data = 0DD
        //   D1 wrap 3 (d1_after): 3 zeros
        // D2 block 2 (d2_after): all zeros = 12 zeros
        // Total: 12 + 3 + 3 + 3 + 3 + 12 = 36
        assert_eq!(pattern, "000000000000" // D2 before (12 zeros)
            .to_owned()
            + "000" // D1 before wrap (3 zeros)
            + "0DD" // data D1 iter 0
            + "0DD" // data D1 iter 1
            + "000" // D1 after wrap (3 zeros)
            + "000000000000"); // D2 after (12 zeros)
    }

    #[test]
    fn test_pad_state_advance_returns_is_data() {
        // Verify advance() returns true for data words, false for padding
        let config = ZeroPadConfig {
            d0_before: 1,
            d0_after: 1,
            ..Default::default()
        };
        let mut state = ZeroPadState::new(config, 2, 1, 1);

        // Expected: [Zero, Data, Data, Zero]
        assert!(!state.advance()); // d0_before zero
        assert!(state.advance());  // data
        assert!(state.advance());  // data
        assert!(!state.advance()); // d0_after zero
        assert!(state.is_finished());
    }

    #[test]
    fn test_transfer_with_padding_total_bytes() {
        // Verify total_bytes includes padding words
        use crate::device::dma::DimensionConfig;

        let bd = BdConfig {
            base_addr: 0x80000,
            length: 40,  // 10 data words * 4 bytes (d0=5 * d1=2)
            d0: DimensionConfig::new(5, 4),
            d1: DimensionConfig::new(2, 20),
            valid: true,
            zero_padding: ZeroPadConfig {
                d0_before: 1,
                d0_after: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        // On a MemTile MM2S transfer, padding should increase total_bytes
        let transfer = Transfer::new(
            &bd, 0, 0, TransferDirection::MM2S, 1, 1, TileType::MemTile,
        ).unwrap();

        // Data: d0=5, d1=2 -> 10 words = 40 bytes
        // D0 wrap = 1+5+1 = 7 words per row
        // D1 total = 0+2+0 = 2 (no D1 padding)
        // Total output = 2 * 7 = 14 words = 56 bytes
        assert_eq!(transfer.total_bytes, 56);
        assert!(transfer.has_zero_padding());

        // Same BD on compute tile (not MemTile) -- no padding applied
        let transfer_compute = Transfer::new(
            &bd, 0, 0, TransferDirection::MM2S, 1, 2, TileType::Compute,
        ).unwrap();
        assert_eq!(transfer_compute.total_bytes, 40);
        assert!(!transfer_compute.has_zero_padding());

        // Same BD on MemTile S2MM -- no padding (S2MM ignores padding)
        let transfer_s2mm = Transfer::new(
            &bd, 0, 0, TransferDirection::S2MM, 1, 1, TileType::MemTile,
        ).unwrap();
        assert!(!transfer_s2mm.has_zero_padding());
    }

    /// Validates the CDO-path i8 padding scenario from add_21_i8 test.
    ///
    /// In the CDO path, mlir-aie converts D0 padding from element counts to
    /// 32-bit word counts before writing to BD registers. For i8 data with
    /// pad_before=4 elements and pad_after=4 elements:
    /// - d0_size = 8 elements * (8/32) = 2 words
    /// - d0_zero_before = 4 elements * (8/32) = 1 word
    /// - d0_zero_after = 4 elements * (8/32) = 1 word
    /// - buffer_length = 16 elements * (8/32) = 4 words
    ///
    /// Total output should be 1 + 2 + 1 = 4 words = 16 bytes.
    #[test]
    fn test_pad_state_i8_cdo_path() {
        // Simulates the BD register values from CDO-compiled
        // add_21_i8_using_dma_op_with_padding test:
        // memref<16xi8>, size=8, stride=1, pad_before=4, pad_after=4
        let config = ZeroPadConfig {
            d0_before: 1,  // 4 i8 elements -> 1 word (CDO converts)
            d0_after: 1,   // 4 i8 elements -> 1 word (CDO converts)
            ..Default::default()
        };
        // d0_size = 2 words (8 i8 elements)
        let mut state = ZeroPadState::new(config, 2, 1, 1);
        let gen = AddressGenerator::new_1d(0x0, 2, 4);

        // Total: 1 pad + 2 data + 1 pad = 4 words
        assert_eq!(state.total_output_words(), 4);

        let mut actions = Vec::new();
        while !state.is_finished() {
            actions.push(state.current_action(&gen));
            state.advance();
        }

        assert_eq!(actions.len(), 4);
        assert_eq!(actions[0], PadAction::Zero);  // 1 word of zeros (4 i8 zeros)
        assert!(matches!(actions[1], PadAction::Data(_)));
        assert!(matches!(actions[2], PadAction::Data(_)));
        assert_eq!(actions[3], PadAction::Zero);  // 1 word of zeros (4 i8 zeros)
    }
}
