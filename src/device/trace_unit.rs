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
    fn from_u32(val: u32) -> Self {
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
enum TraceState {
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
    mode: TraceMode,
    /// Hardware event ID that starts tracing.
    start_event: u8,
    /// Hardware event ID that stops tracing.
    stop_event: u8,
    /// Slot-to-hardware-event-ID mapping (8 slots).
    event_slots: [u8; 8],
    /// Packet type field for the header (Trace_Control1 bits [14:12]).
    packet_type: u8,
    /// Packet ID field for the header (Trace_Control1 bits [4:0]).
    packet_id: u8,

    // -- Runtime state --

    /// Current state (Idle/Running/Stopped).
    state: TraceState,
    /// Cycle counter (starts when tracing begins).
    timer: u64,
    /// Cycle of the last encoded event (for delta computation).
    last_event_cycle: u64,
    /// Accumulated encoded bytes waiting to be packed into packets.
    byte_buffer: Vec<u8>,
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
        // Format from mlir-aie parse.py:
        //   [4:0]   = packet_id
        //   [12:11] = packet_type
        //   [20:16] = row
        //   [28:21] = col
        //   [31]    = parity (odd parity)
        let mut header: u32 = 0;
        header |= (self.packet_id as u32) & 0x1F;
        header |= ((self.packet_type as u32) & 0x3) << 11;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_unit_default_unconfigured() {
        let tu = TraceUnit::new(0, 2);
        assert_eq!(tu.mode, TraceMode::EventTime); // Default mode per AM025
        assert!(!tu.is_configured()); // Not configured until Control0 written
        assert!(!tu.has_pending_packets());
    }

    #[test]
    fn test_register_configuration() {
        let mut tu = TraceUnit::new(0, 2);

        // Control0: mode=EventTime(0), start_event=28 (ACTIVE), stop_event=29 (DISABLED)
        // Per AM025: mode 00=event-time, which is what trace-inject.py uses.
        let ctrl0 = 0 | (28 << 16) | (29 << 24);
        tu.write_register(0x00, ctrl0);
        assert_eq!(tu.mode, TraceMode::EventTime);
        assert_eq!(tu.start_event, 28);
        assert_eq!(tu.stop_event, 29);
        assert!(tu.is_configured());

        // Control1: packet_type=0 (Core), packet_id=1
        let ctrl1 = (0 << 12) | 1;
        tu.write_register(0x04, ctrl1);
        assert_eq!(tu.packet_type, 0);
        assert_eq!(tu.packet_id, 1);

        // Event0: slots 0-3 = [37, 38, 39, 26] (INSTR_VECTOR, INSTR_LOAD, INSTR_STORE, LOCK_STALL)
        let evt0 = 37 | (38 << 8) | (39 << 16) | (26 << 24);
        tu.write_register(0x10, evt0);
        assert_eq!(tu.event_slots[0], 37);
        assert_eq!(tu.event_slots[1], 38);
        assert_eq!(tu.event_slots[2], 39);
        assert_eq!(tu.event_slots[3], 26);

        // Event1: slots 4-7 = [23, 24, 35, 36]
        let evt1 = 23 | (24 << 8) | (35 << 16) | (36 << 24);
        tu.write_register(0x14, evt1);
        assert_eq!(tu.event_slots[4], 23);
        assert_eq!(tu.event_slots[5], 24);
        assert_eq!(tu.event_slots[6], 35);
        assert_eq!(tu.event_slots[7], 36);
    }

    #[test]
    fn test_start_stop_state_machine() {
        let mut tu = TraceUnit::new(0, 2);

        // Configure: mode=EventTime, start=28, stop=29
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37); // slot 0 = INSTR_VECTOR (37)

        // Configured but idle -- waits for start event (like real hardware).
        assert_eq!(tu.state, TraceState::Idle);
        assert!(tu.is_configured());

        // Start event triggers Running state
        tu.notify_event(28, 0);
        assert_eq!(tu.state, TraceState::Running);

        // Matched event is encoded while running
        let before = tu.byte_buffer.len();
        tu.notify_event(37, 100);
        assert!(tu.byte_buffer.len() > before);

        // Stop event transitions to Stopped
        tu.notify_event(29, 300);
        assert_eq!(tu.state, TraceState::Stopped);

        // Events after stop are ignored
        tu.notify_event(37, 400);
        // No crash, no data
    }

    #[test]
    fn test_single0_encoding() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37); // slot 0 = event 37

        // Start tracing
        tu.notify_event(28, 0);
        let start_len = tu.byte_buffer.len(); // Start marker = 8 bytes

        // Event with delta=5 (fits in Single0: 4-bit delta)
        tu.notify_event(37, 5);
        assert_eq!(tu.byte_buffer.len(), start_len + 1); // Single0 = 1 byte

        // Verify encoding: slot=0, delta=5 -> 0b00000101 = 0x05
        assert_eq!(tu.byte_buffer[start_len], 0x05);
    }

    #[test]
    fn test_single1_encoding() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37 | (38 << 8)); // slot 0=37, slot 1=38

        tu.notify_event(28, 0);
        let start_len = tu.byte_buffer.len();

        // Event with delta=500 (fits in Single1: 10-bit delta)
        tu.notify_event(37, 500);
        assert_eq!(tu.byte_buffer.len(), start_len + 2); // Single1 = 2 bytes

        // Verify encoding: slot=0, delta=500
        // Format: 0b100EEETT TTTTTTTT
        // byte0 = 0x80 | (0 << 2) | (500 >> 8 = 1) = 0x81
        // byte1 = 500 & 0xFF = 0xF4
        assert_eq!(tu.byte_buffer[start_len], 0x81);
        assert_eq!(tu.byte_buffer[start_len + 1], 0xF4);
    }

    #[test]
    fn test_single1_encoding_nonzero_slot() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37 | (38 << 8) | (39 << 16)); // slot 0=37, slot 1=38, slot 2=39

        tu.notify_event(28, 0);
        let start_len = tu.byte_buffer.len();

        // Slot 1, delta=500: format 0b100EEETT
        // byte0 = 0x80 | (1 << 2) | (500 >> 8 = 1) = 0x80 | 0x04 | 0x01 = 0x85
        // byte1 = 500 & 0xFF = 0xF4
        tu.notify_event(38, 500);
        assert_eq!(tu.byte_buffer[start_len], 0x85);
        assert_eq!(tu.byte_buffer[start_len + 1], 0xF4);

        // Verify mlir-aie decode: event = (0x85 >> 2) & 7 = 0x21 & 7 = 1, cycles = (0x85 & 3)*256 + 0xF4 = 500
    }

    #[test]
    fn test_single2_encoding() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37); // slot 0 = event 37

        tu.notify_event(28, 0);
        let start_len = tu.byte_buffer.len();

        // Event with delta=100000 (fits in Single2: 18-bit delta)
        tu.notify_event(37, 100000);
        assert_eq!(tu.byte_buffer.len(), start_len + 3); // Single2 = 3 bytes

        // Verify encoding: slot=0, delta=100000 = 0x186A0
        // Format: 0b101EEETT TTTTTTTT TTTTTTTT
        // byte0 = 0xA0 | (0 << 2) | (0x186A0 >> 16 = 1) = 0xA1
        // byte1 = (0x186A0 >> 8) & 0xFF = 0x86
        // byte2 = 0x186A0 & 0xFF = 0xA0
        assert_eq!(tu.byte_buffer[start_len], 0xA1);
        assert_eq!(tu.byte_buffer[start_len + 1], 0x86);
        assert_eq!(tu.byte_buffer[start_len + 2], 0xA0);
    }

    #[test]
    fn test_single2_encoding_nonzero_slot() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (40 << 24));
        tu.write_register(0x14, 23 | (24 << 8) | (35 << 16) | (36 << 24));

        tu.notify_event(28, 0);
        let start_len = tu.byte_buffer.len();

        // Slot 3, delta=100000 = 0x186A0
        // Format: 0b101EEETT TTTTTTTT TTTTTTTT
        // byte0 = 0xA0 | (3 << 2) | (0x186A0 >> 16 = 1) = 0xA0 | 0x0C | 0x01 = 0xAD
        // byte1 = 0x86, byte2 = 0xA0
        tu.notify_event(40, 100000);
        assert_eq!(tu.byte_buffer[start_len], 0xAD);
        assert_eq!(tu.byte_buffer[start_len + 1], 0x86);
        assert_eq!(tu.byte_buffer[start_len + 2], 0xA0);

        // Verify mlir-aie decode: event = (0xAD >> 2) & 7 = 0x2B & 7 = 3
        // cycles = (0xAD & 3)*65536 + 0x86*256 + 0xA0 = 1*65536 + 34304 + 160 = 100000. WRONG.
        // Actually: (0xAD & 3) = 1, 1*65536 = 65536, + 0x86*256 = 34304, + 0xA0 = 160
        // = 65536 + 34304 + 160 = 100000. Correct!
    }

    #[test]
    fn test_packet_formation() {
        let mut tu = TraceUnit::new(1, 3);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x04, (0 << 12) | 5); // pkt_type=0, pkt_id=5
        // Fill all 8 slots so we can generate many events
        tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (40 << 24));
        tu.write_register(0x14, 23 | (24 << 8) | (35 << 16) | (36 << 24));

        // Start tracing (emits 8-byte Start marker)
        tu.notify_event(28, 0);

        // Generate enough Single0 events to fill 28 bytes
        // Start marker is 8 bytes, so we need 20 more Single0 events (1 byte each)
        for i in 1..=20 {
            tu.notify_event(37, i as u64); // delta=1 each time
        }

        // Should have exactly one packet now (28 bytes consumed)
        assert!(tu.has_pending_packets());
        let packet = tu.pop_packet().unwrap();

        // Verify header: col=1, row=3, pkt_type=0, pkt_id=5
        let header = packet[0];
        assert_eq!(header & 0x1F, 5); // packet_id
        assert_eq!((header >> 11) & 0x3, 0); // packet_type
        assert_eq!((header >> 16) & 0x1F, 3); // row
        assert_eq!((header >> 21) & 0x7F, 1); // col
    }

    #[test]
    fn test_flush_pads_partial_packet() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x04, (0 << 12) | 1);
        tu.write_register(0x10, 37);

        // Start tracing and emit a few events
        tu.notify_event(28, 0);
        tu.notify_event(37, 5);
        tu.notify_event(37, 10);

        // Not enough for a full packet yet
        assert!(!tu.has_pending_packets());

        // Flush should pad and emit
        tu.flush();
        assert!(tu.has_pending_packets());

        let packet = tu.pop_packet().unwrap();
        // Words 1-7 should contain data + 0xFE padding
        // The last bytes of the data words should be 0xFE
        let last_word = packet[7];
        // At least some padding bytes present (0xFE = 254)
        let last_byte = last_word & 0xFF;
        assert_eq!(last_byte, 0xFE);
    }

    #[test]
    fn test_unconfigured_ignores_events() {
        let mut tu = TraceUnit::new(0, 2);
        // Don't write Control0 (unconfigured)
        tu.notify_event(28, 100);
        tu.notify_event(37, 200);
        assert!(!tu.has_pending_packets());
        assert!(tu.byte_buffer.is_empty());
    }

    #[test]
    fn test_unmatched_event_ignored() {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37); // Only slot 0 = 37

        tu.notify_event(28, 0); // Start
        let len_after_start = tu.byte_buffer.len();

        // Event 99 is not in any slot -- should be ignored
        tu.notify_event(99, 50);
        assert_eq!(tu.byte_buffer.len(), len_after_start);
    }

    #[test]
    fn test_start_marker_encoding() {
        let mut tu = TraceUnit::new(0, 2);
        // Configure trace unit, then fire start event
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));

        // After configuration, unit is idle (no auto-start)
        assert!(tu.byte_buffer.is_empty());

        // Fire start event to begin tracing
        tu.notify_event(28, 0);

        // Start marker: 0xF0 prefix + 7 bytes timer (big-endian, timer=0)
        assert_eq!(tu.byte_buffer.len(), 8);
        assert_eq!(tu.byte_buffer[0], 0xF0);
        // Timer = 0, big-endian in 7 bytes
        for i in 1..8 {
            assert_eq!(tu.byte_buffer[i], 0x00, "byte {} should be 0", i);
        }
    }

    #[test]
    fn test_packet_header_parity() {
        let mut tu = TraceUnit::new(2, 4);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x04, (3 << 12) | 7); // pkt_type=3, pkt_id=7
        tu.write_register(0x10, 37);

        tu.notify_event(28, 0);
        // Generate 20 events to fill a packet
        for i in 1..=20 {
            tu.notify_event(37, i as u64);
        }

        let packet = tu.pop_packet().unwrap();
        let header = packet[0];
        // Odd parity: total number of set bits should be odd
        assert_eq!(header.count_ones() % 2, 1);
    }

    #[test]
    fn test_slot_index_in_encoding() {
        // Verify that different slots produce different byte patterns
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        // Slots 0-3 = events 37,38,39,40
        tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (40 << 24));

        tu.notify_event(28, 0);
        let start_len = tu.byte_buffer.len();

        // Slot 0, delta=1: 0b00000001 = 0x01
        tu.notify_event(37, 1);
        assert_eq!(tu.byte_buffer[start_len], 0x01);

        // Slot 1, delta=1: 0b00010001 = 0x11
        tu.notify_event(38, 2);
        assert_eq!(tu.byte_buffer[start_len + 1], 0x11);

        // Slot 2, delta=1: 0b00100001 = 0x21
        tu.notify_event(39, 3);
        assert_eq!(tu.byte_buffer[start_len + 2], 0x21);

        // Slot 3, delta=1: 0b00110001 = 0x31
        tu.notify_event(40, 4);
        assert_eq!(tu.byte_buffer[start_len + 3], 0x31);
    }

    /// Verify our encoder matches mlir-aie's decoder (utils/trace/utils.py).
    ///
    /// This test implements the mlir-aie decode logic in Rust and verifies
    /// round-trip correctness for all slots and representative deltas.
    /// Each (slot, delta) pair is tested in isolation to avoid buffer drain
    /// from packet emission.
    #[test]
    fn test_roundtrip_all_slots_all_formats() {
        /// Decode one event from a byte buffer, returning (slot, delta, bytes_consumed).
        /// Implements the same logic as mlir-aie convert_to_commands().
        fn decode_single(buf: &[u8]) -> (u8, u64, usize) {
            let b0 = buf[0];
            if (b0 & 0x80) == 0 {
                // Single0: 0b0EEETTTT
                let event = (b0 >> 4) & 0x07;
                let cycles = (b0 & 0x0F) as u64;
                (event, cycles, 1)
            } else if (b0 & 0xE0) == 0x80 {
                // Single1: 0b100EEETT TTTTTTTT
                let event = (b0 >> 2) & 0x07;
                let cycles = ((b0 & 0x03) as u64) * 256 + buf[1] as u64;
                (event, cycles, 2)
            } else if (b0 & 0xE0) == 0xA0 {
                // Single2: 0b101EEETT TTTTTTTT TTTTTTTT
                let event = (b0 >> 2) & 0x07;
                let cycles = ((b0 & 0x03) as u64) * 65536
                    + (buf[1] as u64) * 256
                    + buf[2] as u64;
                (event, cycles, 3)
            } else {
                panic!("unexpected byte 0x{:02X}", b0);
            }
        }

        // Test all 8 slots with representative deltas for each format.
        // Each combination gets a fresh TraceUnit to avoid buffer drain
        // from packet emission (try_emit_packet drains at 28 bytes).
        let all_deltas: &[(u64, usize)] = &[
            // Single0 (1 byte)
            (0, 1), (1, 1), (7, 1), (15, 1),
            // Single1 (2 bytes)
            (16, 2), (100, 2), (500, 2), (1023, 2),
            // Single2 (3 bytes)
            (1024, 3), (10000, 3), (100000, 3), (262143, 3),
        ];

        for slot in 0u8..8 {
            for &(d, expected_size) in all_deltas {
                let mut tu = TraceUnit::new(0, 2);
                tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
                let evt0 = 37u32 | (38 << 8) | (39 << 16) | (40 << 24);
                let evt1 = 41u32 | (42 << 8) | (43 << 16) | (44 << 24);
                tu.write_register(0x10, evt0);
                tu.write_register(0x14, evt1);
                tu.notify_event(28, 0); // start
                let start_len = tu.byte_buffer.len(); // 8 bytes start marker

                let event_id = 37 + slot;
                tu.notify_event(event_id, d); // delta = d (from start cycle 0)
                let base = start_len;
                assert!(
                    tu.byte_buffer.len() >= base + expected_size,
                    "buffer too short for slot={} delta={}: have {} bytes, need {}",
                    slot, d, tu.byte_buffer.len() - base, expected_size
                );
                let (dec_slot, dec_delta, consumed) = decode_single(&tu.byte_buffer[base..]);
                assert_eq!(
                    dec_slot, slot,
                    "slot mismatch: slot={} delta={} byte0=0x{:02X}",
                    slot, d, tu.byte_buffer[base]
                );
                assert_eq!(
                    dec_delta, d,
                    "delta mismatch: slot={} delta={}", slot, d
                );
                assert_eq!(consumed, expected_size, "size mismatch: slot={} delta={}", slot, d);
            }
        }
    }

    #[test]
    fn test_read_register_roundtrip() {
        let mut tu = TraceUnit::new(3, 5);

        // Write Control0: mode=0, start_event=28, stop_event=29
        let ctrl0 = 0 | (28 << 16) | (29 << 24);
        tu.write_register(0x00, ctrl0);
        assert_eq!(tu.read_register(0x00), ctrl0);

        // Write Control1: packet_type=3, packet_id=7
        let ctrl1 = (3 << 12) | 7;
        tu.write_register(0x04, ctrl1);
        assert_eq!(tu.read_register(0x04), ctrl1);

        // Write Event0/Event1
        let evt0 = 37 | (38 << 8) | (39 << 16) | (40 << 24);
        let evt1 = 41 | (42 << 8) | (43 << 16) | (44 << 24);
        tu.write_register(0x10, evt0);
        tu.write_register(0x14, evt1);
        assert_eq!(tu.read_register(0x10), evt0);
        assert_eq!(tu.read_register(0x14), evt1);

        // Read Status: should be Idle (0) + mode EventTime (0)
        assert_eq!(tu.read_register(0x08), 0);

        // Start tracing -> state becomes Running
        tu.notify_event(28, 0);
        assert_eq!(tu.read_register(0x08), 1 << 8); // Running=1 at bits [9:8]

        // Stop tracing -> state becomes Stopped
        tu.notify_event(29, 100);
        assert_eq!(tu.read_register(0x08), 2 << 8); // Stopped=2 at bits [9:8]
    }
}
