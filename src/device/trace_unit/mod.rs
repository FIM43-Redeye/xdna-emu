//! Hardware trace unit emulation.
//!
//! Each AIE tile has trace units (Core module trace, Memory module trace)
//! that monitor hardware events and emit compressed binary trace packets.
//! These packets flow through the stream switch to shim DMA and into host
//! DDR, where tools like mlir-aie's `parse.py` decode them.
//!
//! # Register Interface (from AM025)
//!
//! | Register       | Offset   | Fields                                           |
//! |----------------|----------|--------------------------------------------------|
//! | Trace_Control0 | +0x00    | [31:24] stop_event, [23:16] start_event, [1:0] mode |
//! | Trace_Control1 | +0x04    | [14:12] packet_type, [4:0] packet_id             |
//! | Trace_Event0   | +0x10    | [31:24] evt3, [23:16] evt2, [15:8] evt1, [7:0] evt0 |
//! | Trace_Event1   | +0x14    | [31:24] evt7, [23:16] evt6, [15:8] evt5, [7:0] evt4 |
//!
//! # Byte Encoding (from mlir-aie utils/trace/utils.py)
//!
//! Events are compressed into variable-length byte sequences:
//!
//! | Type    | Pattern                          | Content                         |
//! |---------|----------------------------------|---------------------------------|
//! | Single0 | `0b0EEETTTT`                     | event(3), delta(4): 0-15 cycles |
//! | Single1 | `0b100EEETT TTTTTTTT`            | event(3), delta(10): 0-1023     |
//! | Single2 | `0b101EEETT TTTTTTTT TTTTTTTT`   | event(3), delta(18): 0-262143   |
//! | Start   | `0b11110000 + 7 bytes timer`     | 56-bit timer sync               |
//! | Pad     | `0xFE`                           | Fills remainder of packet       |
//!
//! # Packet Format
//!
//! Each packet is 8 x 32-bit words (32 bytes):
//! - Word 0: packet header (col, row, packet_type, packet_id)
//! - Words 1-7: compressed byte stream (28 bytes of encoded events)

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

/// Trace operating mode (Trace_Control0 bits [1:0]).
///
/// Per AM025: 00=event-time, 01=event-PC, 10=execution.
/// There is no explicit "off" mode -- the trace unit is idle until the
/// configured start event fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraceMode {
    /// Event-time mode (00): records event slot + cycle delta.
    /// This is the default mode and what trace-inject.py uses.
    #[default]
    EventTime = 0,
    /// Event-PC mode (01): records event slot + program counter.
    EventPc = 1,
    /// Execution mode (10): records detailed execution trace.
    Execution = 2,
    /// Reserved (11).
    Reserved = 3,
}

impl TraceMode {
    pub(super) fn from_u32(val: u32) -> Self {
        match val & 0x3 {
            0 => TraceMode::EventTime,
            1 => TraceMode::EventPc,
            2 => TraceMode::Execution,
            3 => TraceMode::Reserved,
            _ => unreachable!(),
        }
    }
}

/// Trace unit running state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum TraceState {
    /// Waiting for start event.
    #[default]
    Idle,
    /// Actively recording events.
    Running,
    /// Stop event received, no longer recording.
    Stopped,
}

/// Hardware trace unit for one tile module (Core or Memory).
///
/// Configured by register writes from CDO/control packets. When active,
/// it maps hardware events through 8 configurable slots, compresses
/// them into byte-encoded packets, and emits 8-word trace packets that
/// flow through the stream switch.
#[derive(Debug)]
pub struct TraceUnit {
    // -- Configuration (from register writes) --

    /// Operating mode.
    pub(super) mode: TraceMode,
    /// Hardware event ID that starts tracing.
    pub(super) start_event: u8,
    /// Hardware event ID that stops tracing.
    pub(super) stop_event: u8,
    /// Slot-to-hardware-event-ID mapping (8 slots).
    pub(super) event_slots: [u8; 8],
    /// Packet type field for the header (Trace_Control1 bits [14:12]).
    pub(super) packet_type: u8,
    /// Packet ID field for the header (Trace_Control1 bits [4:0]).
    pub(super) packet_id: u8,

    // -- Runtime state --

    /// Current state (Idle/Running/Stopped).
    pub(super) state: TraceState,
    /// Cycle counter (starts when tracing begins).
    timer: u64,
    /// Cycle of the last encoded event (for delta computation).
    last_event_cycle: u64,
    /// Accumulated encoded bytes waiting to be packed into packets.
    pub(super) byte_buffer: Vec<u8>,
    /// Individual words ready for emission to the stream switch.
    /// Each entry is (word, tlast). Words are pushed in order: header first,
    /// then 7 data words, with TLAST=true on the last word. This allows the
    /// routing layer to push one word at a time, respecting FIFO backpressure.
    pending_words: VecDeque<(u32, bool)>,

    // -- Tile identity (for packet headers) --

    /// Column of the owning tile.
    col: u8,
    /// Row of the owning tile.
    row: u8,

    /// Whether Control0 has been written (trace unit is configured).
    /// The trace unit only processes events after configuration.
    configured: bool,
}

impl TraceUnit {
    /// Create a new trace unit for the given tile location.
    pub fn new(col: u8, row: u8) -> Self {
        Self {
            mode: TraceMode::EventTime,
            start_event: 0,
            stop_event: 0,
            event_slots: [0; 8],
            packet_type: 0,
            packet_id: 0,
            state: TraceState::Idle,
            timer: 0,
            last_event_cycle: 0,
            byte_buffer: Vec::with_capacity(64),
            pending_words: VecDeque::new(),
            col,
            row,
            configured: false,
        }
    }

    /// Read a trace register (offset relative to the trace register block).
    ///
    /// Offsets:
    /// - 0x00: Trace_Control0
    /// - 0x04: Trace_Control1
    /// - 0x08: Trace_Status (state[9:8], mode[2:0])
    /// - 0x10: Trace_Event0
    /// - 0x14: Trace_Event1
    pub fn read_register(&self, offset: u32) -> u32 {
        match offset {
            0x00 => {
                // Trace_Control0: mode[1:0], start_event[23:16], stop_event[31:24]
                (self.mode as u32)
                    | ((self.start_event as u32) << 16)
                    | ((self.stop_event as u32) << 24)
            }
            0x04 => {
                // Trace_Control1: packet_type[14:12], packet_id[4:0]
                ((self.packet_type as u32) << 12) | (self.packet_id as u32)
            }
            0x08 => {
                // Trace_Status: state[9:8], mode[2:0]
                // Per aie-rt xaie_trace.c: XAie_TraceGetState reads STATE,
                // XAie_TraceGetMode reads MODE from this register.
                let state_bits: u32 = match self.state {
                    TraceState::Idle => 0,
                    TraceState::Running => 1,
                    TraceState::Stopped => 2,
                };
                (state_bits << 8) | (self.mode as u32)
            }
            0x10 => {
                // Trace_Event0: evt3[31:24], evt2[23:16], evt1[15:8], evt0[7:0]
                (self.event_slots[0] as u32)
                    | ((self.event_slots[1] as u32) << 8)
                    | ((self.event_slots[2] as u32) << 16)
                    | ((self.event_slots[3] as u32) << 24)
            }
            0x14 => {
                // Trace_Event1: evt7[31:24], evt6[23:16], evt5[15:8], evt4[7:0]
                (self.event_slots[4] as u32)
                    | ((self.event_slots[5] as u32) << 8)
                    | ((self.event_slots[6] as u32) << 16)
                    | ((self.event_slots[7] as u32) << 24)
            }
            _ => 0,
        }
    }

    /// Write a trace register (offset relative to the trace register block).
    ///
    /// Offsets:
    /// - 0x00: Trace_Control0
    /// - 0x04: Trace_Control1
    /// - 0x10: Trace_Event0
    /// - 0x14: Trace_Event1
    pub fn write_register(&mut self, offset: u32, value: u32) {
        match offset {
            0x00 => {
                // Trace_Control0: mode[1:0], start_event[23:16], stop_event[31:24]
                // AM025 claims 7-bit fields [22:16] and [30:24], but mlir-aie
                // uses the full byte (MemTile events go up to 160). Use 8 bits
                // to match real hardware behavior.
                self.mode = TraceMode::from_u32(value);
                self.start_event = ((value >> 16) & 0xFF) as u8;
                self.stop_event = ((value >> 24) & 0xFF) as u8;
                self.configured = true;

                // Trace unit waits in Idle state until its start_event fires.
                // The CDO sequence configures broadcast channels, then writes
                // Event_Generate to fire USER_EVENT -> BROADCAST_N -> trace
                // start. The tile's write_register() handles Event_Generate
                // and propagates broadcast events to all tiles in the column,
                // which triggers notify_event() on each trace unit.

                log::debug!(
                    "TraceUnit ({},{}) Control0: mode={:?} start={} stop={} -> {:?}",
                    self.col, self.row, self.mode, self.start_event, self.stop_event,
                    self.state
                );
            }
            0x04 => {
                // Trace_Control1: packet_type[14:12], packet_id[4:0]
                self.packet_type = ((value >> 12) & 0x7) as u8;
                self.packet_id = (value & 0x1F) as u8;
                log::debug!(
                    "TraceUnit ({},{}) Control1: pkt_type={} pkt_id={}",
                    self.col, self.row, self.packet_type, self.packet_id
                );
            }
            0x10 => {
                // Trace_Event0: evt3[31:24], evt2[23:16], evt1[15:8], evt0[7:0]
                self.event_slots[0] = (value & 0xFF) as u8;
                self.event_slots[1] = ((value >> 8) & 0xFF) as u8;
                self.event_slots[2] = ((value >> 16) & 0xFF) as u8;
                self.event_slots[3] = ((value >> 24) & 0xFF) as u8;
                log::debug!(
                    "TraceUnit ({},{}) Event0: slots[0-3]={:?}",
                    self.col, self.row, &self.event_slots[0..4]
                );
            }
            0x14 => {
                // Trace_Event1: evt7[31:24], evt6[23:16], evt5[15:8], evt4[7:0]
                self.event_slots[4] = (value & 0xFF) as u8;
                self.event_slots[5] = ((value >> 8) & 0xFF) as u8;
                self.event_slots[6] = ((value >> 16) & 0xFF) as u8;
                self.event_slots[7] = ((value >> 24) & 0xFF) as u8;
                log::debug!(
                    "TraceUnit ({},{}) Event1: slots[4-7]={:?}",
                    self.col, self.row, &self.event_slots[4..8]
                );
            }
            _ => {
                log::debug!(
                    "TraceUnit ({},{}) unknown register offset 0x{:X}",
                    self.col, self.row, offset
                );
            }
        }
    }

    /// Notify the trace unit of a hardware event at the given cycle.
    ///
    /// If the event matches a configured slot, it is encoded and buffered.
    /// Start/stop events control the tracing state machine.
    pub fn notify_event(&mut self, hw_event_id: u8, cycle: u64) {
        if !self.configured {
            return;
        }

        // Check for start/stop events
        if self.state == TraceState::Idle && hw_event_id == self.start_event {
            self.state = TraceState::Running;
            self.timer = cycle;
            self.last_event_cycle = cycle;
            // Emit a Start marker with the current timer value
            self.encode_start(cycle);
            log::debug!(
                "TraceUnit ({},{}) started at cycle {}",
                self.col, self.row, cycle
            );
            return;
        }

        if self.state == TraceState::Running && hw_event_id == self.stop_event {
            self.state = TraceState::Stopped;
            log::debug!(
                "TraceUnit ({},{}) stopped at cycle {}",
                self.col, self.row, cycle
            );
            // Flush remaining data as a final packet
            self.flush();
            return;
        }

        // Only record events while running
        if self.state != TraceState::Running {
            return;
        }

        // Find matching slot (0-7) for this hardware event ID
        let slot = match self.event_slots.iter().position(|&s| s == hw_event_id) {
            Some(idx) => idx as u8,
            None => return, // Event not in any slot
        };

        // Encode the event with cycle delta
        let delta = cycle.saturating_sub(self.last_event_cycle);
        self.last_event_cycle = cycle;
        self.encode_single(slot, delta);

        // Check if we have enough bytes for a complete packet
        self.try_emit_packet();
    }

    /// Advance the timer. Called each cycle to keep the internal clock in sync.
    pub fn tick(&mut self, cycle: u64) {
        if self.state == TraceState::Running {
            self.timer = cycle;
        }
    }

    /// Pop one complete trace packet (8 x u32 words).
    ///
    /// Returns `None` if no packets are ready. Consumes 8 words from the
    /// pending queue. Callers that need backpressure support should use
    /// `pop_word()` instead.
    pub fn pop_packet(&mut self) -> Option<[u32; 8]> {
        if self.pending_words.len() < 8 {
            return None;
        }
        let mut packet = [0u32; 8];
        for p in &mut packet {
            *p = self.pending_words.pop_front().unwrap().0;
        }
        Some(packet)
    }

    /// Pop a single word with its TLAST flag.
    ///
    /// Returns `None` if no words are pending. The caller should push the
    /// word to the stream switch slave port, checking `can_accept()` first
    /// to respect FIFO backpressure.
    pub fn pop_word(&mut self) -> Option<(u32, bool)> {
        self.pending_words.pop_front()
    }

    /// Check if there are pending words to emit.
    pub fn has_pending_words(&self) -> bool {
        !self.pending_words.is_empty()
    }

    /// Check if there are pending packets to emit (8+ words queued).
    pub fn has_pending_packets(&self) -> bool {
        self.pending_words.len() >= 8
    }

    /// Check if the trace unit is configured (mode != Off).
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Flush any remaining bytes as a padded final packet.
    pub fn flush(&mut self) {
        if self.byte_buffer.is_empty() {
            return;
        }

        log::trace!(
            "TraceUnit ({},{}) flush: {} bytes -> padded packet (packets so far: {})",
            self.col, self.row, self.byte_buffer.len(), self.pending_words.len() / 8
        );

        // Pad to 28 bytes with 0xFE
        while self.byte_buffer.len() < 28 {
            self.byte_buffer.push(0xFE);
        }

        self.emit_packet_from_buffer();
    }

    // -- Internal encoding methods --

    /// Encode a single event with a cycle delta.
    ///
    /// Chooses Single0 (1 byte), Single1 (2 bytes), or Single2 (3 bytes)
    /// based on the magnitude of the delta.
    ///
    /// Bit layouts (from mlir-aie utils/trace/utils.py decode logic):
    ///
    /// Single0: `0b0EEETTTT` (1 byte)
    ///   - bits [6:4] = slot (event), bits [3:0] = delta (0-15)
    ///
    /// Single1: `0b100EEETT TTTTTTTT` (2 bytes)
    ///   - byte0 bits [7:5] = 100, [4:2] = slot, [1:0] = delta_hi
    ///   - byte1 = delta_lo. Total delta = 10 bits (0-1023).
    ///
    /// Single2: `0b101EEETT TTTTTTTT TTTTTTTT` (3 bytes)
    ///   - byte0 bits [7:5] = 101, [4:2] = slot, [1:0] = delta_hi
    ///   - byte1-2 = delta_mid_lo. Total delta = 18 bits (0-262143).
    fn encode_single(&mut self, slot: u8, delta: u64) {
        debug_assert!(slot < 8);

        if delta <= 15 {
            // Single0: 0b0EEETTTT (1 byte)
            let byte = (slot << 4) | (delta as u8 & 0x0F);
            self.byte_buffer.push(byte);
        } else if delta <= 1023 {
            // Single1: 0b100EEETT TTTTTTTT (2 bytes)
            let d = delta as u16;
            let byte0 = 0x80 | (slot << 2) | ((d >> 8) as u8 & 0x03);
            let byte1 = (d & 0xFF) as u8;
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
        } else {
            // Single2: 0b101EEETT TTTTTTTT TTTTTTTT (3 bytes)
            // Clamp to 18 bits (262143)
            let d = delta.min(0x3FFFF) as u32;
            let byte0 = 0xA0 | (slot << 2) | ((d >> 16) as u8 & 0x03);
            let byte1 = ((d >> 8) & 0xFF) as u8;
            let byte2 = (d & 0xFF) as u8;
            self.byte_buffer.push(byte0);
            self.byte_buffer.push(byte1);
            self.byte_buffer.push(byte2);
        }
    }

    /// Encode a Start marker with 56-bit timer value.
    ///
    /// Format: 0xF0 prefix byte + 7 bytes of timer (big-endian).
    fn encode_start(&mut self, timer: u64) {
        self.byte_buffer.push(0xF0);
        // 7 bytes of timer, big-endian (56 bits)
        for i in (0..7).rev() {
            self.byte_buffer.push(((timer >> (i * 8)) & 0xFF) as u8);
        }
    }

    /// If the byte buffer has >= 28 bytes, pack them into an 8-word packet.
    fn try_emit_packet(&mut self) {
        while self.byte_buffer.len() >= 28 {
            self.emit_packet_from_buffer();
        }
    }

    /// Consume 28 bytes from the buffer and emit one packet as individual words.
    fn emit_packet_from_buffer(&mut self) {
        // Word 0: packet header
        // Format from mlir-aie utils/trace/utils.py extract_tile():
        //   [4:0]   = packet_id
        //   [11:5]  = reserved (must be 0)
        //   [13:12] = packet_type
        //   [15:14] = reserved
        //   [20:16] = row
        //   [27:21] = col
        //   [30:28] = reserved (must be 0)
        //   [31]    = parity (odd parity)
        let mut header: u32 = 0;
        header |= (self.packet_id as u32) & 0x1F;
        header |= ((self.packet_type as u32) & 0x3) << 12;
        header |= ((self.row as u32) & 0x1F) << 16;
        header |= ((self.col as u32) & 0x7F) << 21;
        // Compute odd parity over bits 30:0
        let ones = header.count_ones();
        if ones % 2 == 0 {
            header |= 1 << 31; // Set parity bit to make total odd
        }
        self.pending_words.push_back((header, false));

        // Words 1-7: 28 bytes of trace data, packed big-endian into u32s
        for word_idx in 0..7 {
            let base = word_idx * 4;
            let mut word: u32 = 0;
            for byte_idx in 0..4 {
                let byte = if base + byte_idx < self.byte_buffer.len() {
                    self.byte_buffer[base + byte_idx]
                } else {
                    0xFE // Pad byte
                };
                word = (word << 8) | (byte as u32);
            }
            let tlast = word_idx == 6; // Last data word (word 7 of 8)
            self.pending_words.push_back((word, tlast));
        }

        // Remove consumed bytes (up to 28)
        let consumed = self.byte_buffer.len().min(28);
        self.byte_buffer.drain(..consumed);
    }
}
