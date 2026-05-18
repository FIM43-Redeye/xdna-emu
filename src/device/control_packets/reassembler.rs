//! Word-by-word control packet reassembly from stream switch data.
//!
//! The stream switch delivers control packets one 32-bit word at a time
//! to the TileCtrl master port. This module reassembles those words into
//! complete [`ControlPacket`] instances that the [`ControlPacketProcessor`]
//! can execute.
//!
//! # Stream Protocol
//!
//! When a master port has `Drop_Header=false`, the stream routing header
//! is forwarded before the actual control packet header. The reassembler
//! handles this by starting in `WaitingForStreamHeader` state when configured.
//!
//! # Lifecycle
//!
//! ```text
//! [WaitingForStreamHeader] --stream_header--> [Idle]
//! [Idle]                   --ctrl_header-->   [Collecting] or [Complete(Read)]
//! [Collecting]             --data_beats-->    [Complete]
//! [Complete]               --take_packet-->   [Idle] or [WaitingForStreamHeader]
//! ```

use super::parser::{ControlPacket, CtrlOpCode, HeaderFields, parse_header};
use super::status::PktHandlerError;
// odd_parity_ok/PacketHeader live in stream_switch (parity formula and
// the stream routing header are shared there; packet_types is private).
use crate::device::stream_switch::{odd_parity_ok, PacketHeader};

/// Reassembly state machine for control packet words.
#[derive(Debug)]
pub struct StreamReassembler {
    state: ReassemblerState,
    /// Whether stream headers are forwarded (Drop_Header=false on master port).
    drop_header: bool,
    /// Tile location for logging/error context.
    col: u8,
    row: u8,
}

#[derive(Debug, Default)]
enum ReassemblerState {
    /// Waiting for stream routing header (when drop_header=false).
    WaitingForStreamHeader,
    /// Ready for control packet header.
    #[default]
    Idle,
    /// Collecting data beats after header.
    Collecting {
        header: HeaderFields,
        beats_collected: u8,
        data: [u32; 4],
    },
}

/// Result of feeding a word to the reassembler.
#[derive(Debug)]
pub enum ReassembleResult {
    /// Still collecting -- no complete packet yet.
    Pending,
    /// A complete control packet is ready.
    Complete(ControlPacket),
    /// A protocol violation with a faithful Control_Packet_Handler_Status
    /// detecting path (parity / TLAST). Latched as a sticky bit.
    HandlerError(super::status::PktHandlerError),
    /// A structural rejection (logged only, no status bit).
    Error(String),
}

impl StreamReassembler {
    /// Create a new reassembler for a tile.
    ///
    /// Initially in `Idle` state (assumes Drop_Header=true). Call
    /// `set_drop_header(false)` to enable stream header consumption.
    pub fn new(col: u8, row: u8) -> Self {
        Self { state: ReassemblerState::Idle, drop_header: true, col, row }
    }

    /// Configure whether stream headers are forwarded to this port.
    ///
    /// When `drop_header=false`, the reassembler expects a stream routing
    /// header before each control packet header. Call this when the
    /// TileCtrl master port's packet config is detected.
    pub fn set_drop_header(&mut self, drop: bool) {
        self.drop_header = drop;
        if !drop {
            // If we're now expecting headers, transition to waiting state
            if matches!(self.state, ReassemblerState::Idle) {
                self.state = ReassemblerState::WaitingForStreamHeader;
            }
        }
    }

    /// Whether stream headers are being dropped (not forwarded).
    pub fn drop_header(&self) -> bool {
        self.drop_header
    }

    /// Feed one word from the TileCtrl master port.
    ///
    /// Returns `Complete(packet)` when a full control packet has been
    /// reassembled, `Pending` when more words are needed,
    /// `HandlerError(e)` on a protocol violation with a faithful
    /// Control_Packet_Handler_Status bit, or `Error(msg)` on a structural
    /// rejection (logged only, no status bit).
    pub fn feed_word(&mut self, word: u32, tlast: bool) -> ReassembleResult {
        match std::mem::take(&mut self.state) {
            ReassemblerState::WaitingForStreamHeader => {
                // The stream routing header is exactly what
                // PacketHeader::decode parses. Validate its odd parity
                // (First_Header_Parity). Reachable only when
                // drop_header=false; when the switch dropped the header
                // the handler never sees it and this cannot fire.
                let (hdr, parity_ok) = PacketHeader::decode(word);
                if !parity_ok {
                    // Stay waiting for a valid header.
                    self.state = ReassemblerState::WaitingForStreamHeader;
                    return ReassembleResult::HandlerError(PktHandlerError::FirstHeaderParity);
                }
                log::debug!(
                    "Tile ({},{}) ctrl_pkt: consuming stream header 0x{:08X} (stream_id={}, type={:?})",
                    self.col,
                    self.row,
                    word,
                    hdr.stream_id,
                    hdr.packet_type
                );
                self.state = ReassemblerState::Idle;
                ReassembleResult::Pending
            }
            ReassemblerState::Idle => {
                // Second_Header_Parity: validate opcode-header odd parity
                // BEFORE structural decode -- hardware checks header parity
                // at ingress, before acting on opcode/length. Structural
                // rejections (below) are logged only and do NOT set this
                // bit (spec 3.2 tightening).
                if !odd_parity_ok(word) {
                    self.state = ReassemblerState::Idle;
                    return ReassembleResult::HandlerError(PktHandlerError::SecondHeaderParity);
                }

                // Parse control packet header.
                let header = match parse_header(word) {
                    Ok(h) => h,
                    Err(e) => {
                        self.state = ReassemblerState::Idle;
                        return ReassembleResult::Error(format!(
                            "Tile ({},{}) ctrl_pkt: header parse error: {}",
                            self.col, self.row, e
                        ));
                    }
                };

                log::info!(
                    "Tile ({},{}) ctrl_pkt: header 0x{:08X} addr=0x{:05X} op={:?} beats={} resp_id={}",
                    self.col,
                    self.row,
                    word,
                    header.address,
                    header.opcode,
                    header.beats,
                    header.response_id
                );

                // OP_READ has no data payload -- complete immediately.
                if header.opcode == CtrlOpCode::Read {
                    let packet = ControlPacket::read(header.address, header.beats, header.response_id);
                    self.transition_after_complete(tlast);
                    return ReassembleResult::Complete(packet);
                }

                // For write operations, start collecting data beats.
                if header.beats == 0 {
                    // Shouldn't happen (beats = length+1, minimum 1), but handle gracefully.
                    self.transition_after_complete(tlast);
                    return ReassembleResult::Error(format!(
                        "Tile ({},{}) ctrl_pkt: zero beats in write operation",
                        self.col, self.row
                    ));
                }

                self.state = ReassemblerState::Collecting { header, beats_collected: 0, data: [0; 4] };
                ReassembleResult::Pending
            }
            ReassemblerState::Collecting { header, mut beats_collected, mut data } => {
                data[beats_collected as usize] = word;
                beats_collected += 1;

                log::debug!(
                    "Tile ({},{}) ctrl_pkt: data[{}] = 0x{:08X} ({}/{}){}",
                    self.col,
                    self.row,
                    beats_collected - 1,
                    word,
                    beats_collected,
                    header.beats,
                    if tlast { " TLAST" } else { "" }
                );

                let is_final_beat = beats_collected >= header.beats;

                // TLAST must land exactly on the final declared beat for
                // write-class packets. Read packets never reach Collecting
                // (they complete in the Idle arm), so this check covers all
                // reachable opcodes.
                if tlast && !is_final_beat {
                    // Unexpected early TLAST -- hardware sets Tlast_Error (bit 3).
                    self.transition_after_complete(true);
                    return ReassembleResult::HandlerError(PktHandlerError::Tlast);
                }
                if is_final_beat && !tlast {
                    // Missing TLAST on the final beat -- hardware sets Tlast_Error (bit 3).
                    self.transition_after_complete(false);
                    return ReassembleResult::HandlerError(PktHandlerError::Tlast);
                }

                if is_final_beat {
                    // All beats received with correct TLAST -- build the complete packet.
                    let payload = &data[..beats_collected as usize];
                    let packet = match header.opcode {
                        CtrlOpCode::Write | CtrlOpCode::BlockWrite => {
                            ControlPacket::block_write(header.address, payload.to_vec())
                        }
                        CtrlOpCode::WriteIncr => ControlPacket::write_incr(header.address, payload.to_vec()),
                        CtrlOpCode::Read => unreachable!("Read handled above"),
                    };
                    self.transition_after_complete(tlast);
                    ReassembleResult::Complete(packet)
                } else {
                    // Still collecting.
                    self.state = ReassemblerState::Collecting { header, beats_collected, data };
                    ReassembleResult::Pending
                }
            }
        }
    }

    /// Reset the reassembler to its initial state.
    pub fn reset(&mut self) {
        self.state = if self.drop_header {
            ReassemblerState::Idle
        } else {
            ReassemblerState::WaitingForStreamHeader
        };
    }

    /// Transition after a complete packet, considering TLAST and drop_header.
    fn transition_after_complete(&mut self, tlast: bool) {
        self.state = if tlast && !self.drop_header {
            ReassemblerState::WaitingForStreamHeader
        } else {
            ReassemblerState::Idle
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_reassembler_starts_idle() {
        let r = StreamReassembler::new(0, 2);
        assert!(r.drop_header());
    }

    #[test]
    fn single_word_write() {
        let mut r = StreamReassembler::new(0, 2);
        // Build header: addr=0x100, op=Write(0), beats=1(encoded as 0), resp_id=0
        let header = build_test_header(0x100, 0, 0, 0);
        // Feed header -- should go to Collecting
        assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
        // Feed data beat
        match r.feed_word(0xDEADBEEF, true) {
            ReassembleResult::Complete(pkt) => {
                assert_eq!(pkt.address, 0x100);
                assert_eq!(pkt.data.len(), 1);
                assert_eq!(pkt.data[0], 0xDEADBEEF);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn multi_word_write() {
        let mut r = StreamReassembler::new(0, 2);
        // 3 beats: length field = 2 (beats = length+1 = 3).
        // addr=0x201 (2 set bits) + length=2 (1 bit) = 3 bits (odd, valid parity).
        // Old: 0x200 -> 2 bits (even, invalid); fixed to 0x201 -> 3 bits (odd).
        let header = build_test_header(0x201, 2, 0, 0);
        assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
        assert!(matches!(r.feed_word(0x11, false), ReassembleResult::Pending));
        assert!(matches!(r.feed_word(0x22, false), ReassembleResult::Pending));
        match r.feed_word(0x33, true) {
            ReassembleResult::Complete(pkt) => {
                assert_eq!(pkt.data, vec![0x11, 0x22, 0x33]);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn read_completes_immediately() {
        let mut r = StreamReassembler::new(0, 2);
        // OP_READ = 1. resp_id=43 (0b101011, 4 set bits) + op=1 (1 bit) + addr=0x300 (2 bits) = 7 bits (odd, valid).
        // Old: resp_id=42 (0b101010, 3 bits) -> total 6 bits (even, invalid); fixed resp_id to 43.
        let header = build_test_header(0x300, 0, 1, 43);
        match r.feed_word(header, true) {
            ReassembleResult::Complete(pkt) => {
                assert_eq!(pkt.address, 0x300);
                assert!(pkt.data.is_empty());
                assert_eq!(pkt.opcode, CtrlOpCode::Read);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn stream_header_consumed_when_drop_false() {
        let mut r = StreamReassembler::new(0, 2);
        r.set_drop_header(false);

        // First word should be consumed as stream header
        assert!(matches!(r.feed_word(0x0000_0007, false), ReassembleResult::Pending));

        // Now the actual control packet header
        let header = build_test_header(0x100, 0, 0, 0);
        assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));

        // Data beat with TLAST
        match r.feed_word(0x42, true) {
            ReassembleResult::Complete(pkt) => {
                assert_eq!(pkt.data[0], 0x42);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn tlast_with_no_drop_transitions_to_waiting() {
        let mut r = StreamReassembler::new(0, 2);
        r.set_drop_header(false);

        // Stream header
        r.feed_word(0x0000_0007, false);

        // OP_READ with TLAST.
        // addr=0x101 (2 bits) + op=1 (1 bit) = 3 bits (odd, valid parity).
        // Old: addr=0x100 -> 2 bits (even, invalid); fixed to 0x101 -> 3 bits.
        let header = build_test_header(0x101, 0, 1, 0);
        r.feed_word(header, true);

        // Next word should be treated as stream header again
        // (Feed a ctrl header without consuming stream header first -- should go Pending)
        assert!(matches!(r.feed_word(0x0000_0007, false), ReassembleResult::Pending));
    }

    #[test]
    fn consecutive_packets_back_to_back() {
        // Two write packets in the same stream. Each must assert TLAST on its
        // own final beat; the test checks that after the first packet completes
        // the reassembler is ready for the second header immediately.
        let mut r = StreamReassembler::new(0, 2);

        // First packet: write 1 word, TLAST on final beat.
        // (Old test used tlast=false on the final beat, which is a protocol
        // violation. Corrected to true -- hardware requires TLAST on the final
        // declared beat of every write-class packet.)
        let h1 = build_test_header(0x100, 0, 0, 0);
        r.feed_word(h1, false);
        match r.feed_word(0xAA, true) {
            ReassembleResult::Complete(pkt) => assert_eq!(pkt.data[0], 0xAA),
            other => panic!("Expected Complete, got {:?}", other),
        }

        // Second packet in same stream: another write.
        // build_test_header(0x200,0,0,0) = 0x00000200 = 1 set bit (odd, valid).
        let h2 = build_test_header(0x200, 0, 0, 0);
        r.feed_word(h2, false);
        match r.feed_word(0xBB, true) {
            ReassembleResult::Complete(pkt) => assert_eq!(pkt.data[0], 0xBB),
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn reset_returns_to_initial_state() {
        let mut r = StreamReassembler::new(0, 2);
        // Start collecting.
        // addr=0x101 (2 bits) + length=1 (1 bit) = 3 bits (odd, valid parity).
        // Old: addr=0x100 -> 2 bits (even, invalid); fixed to 0x101 -> 3 bits.
        let h = build_test_header(0x101, 1, 0, 0);
        r.feed_word(h, false);
        r.feed_word(0x11, false);
        // Reset mid-collection
        r.reset();
        // Should be back to Idle, ready for a new header.
        // addr=0x201 (2 bits) + op=1 (1 bit) = 3 bits (odd, valid parity).
        // Old: addr=0x200 -> 2 bits (even, invalid); fixed to 0x201 -> 3 bits.
        let h2 = build_test_header(0x201, 0, 1, 0);
        match r.feed_word(h2, true) {
            ReassembleResult::Complete(pkt) => assert_eq!(pkt.opcode, CtrlOpCode::Read),
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn write_incr_opcode() {
        let mut r = StreamReassembler::new(0, 2);
        // OP_WRITE_INCR = 2. addr=0x401 (2 bits) + op=2 (1 bit) = 3 bits (odd, valid parity).
        // Old: addr=0x400 -> 2 bits (even, invalid); fixed to 0x401 -> 3 bits.
        let header = build_test_header(0x401, 0, 2, 0);
        r.feed_word(header, false);
        match r.feed_word(0x55, true) {
            ReassembleResult::Complete(pkt) => {
                assert_eq!(pkt.opcode, CtrlOpCode::WriteIncr);
                assert_eq!(pkt.data[0], 0x55);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn handler_error_variant_carries_pkt_handler_error() {
        use crate::device::control_packets::status::PktHandlerError;
        let r = ReassembleResult::HandlerError(PktHandlerError::SecondHeaderParity);
        match r {
            ReassembleResult::HandlerError(e) => assert_eq!(e.bit(), 0x2),
            other => panic!("expected HandlerError, got {:?}", other),
        }
    }

    #[test]
    fn bad_parity_stream_header_sets_first_header_parity() {
        use crate::device::control_packets::status::PktHandlerError;
        let mut r = StreamReassembler::new(0, 2);
        r.set_drop_header(false); // stream header expected

        // Stream header with EVEN ones -> odd-parity invalid.
        // 0b11 has two set bits (even) -> odd_parity_ok == false.
        match r.feed_word(0b11, false) {
            ReassembleResult::HandlerError(PktHandlerError::FirstHeaderParity) => {}
            other => panic!("expected FirstHeaderParity, got {:?}", other),
        }

        // The arm's core contract: stay waiting -- a valid header fed
        // immediately after the bad one is still consumed as a header.
        assert!(matches!(r.feed_word(0b1, false), ReassembleResult::Pending));
    }

    #[test]
    fn good_parity_stream_header_is_consumed() {
        let mut r = StreamReassembler::new(0, 2);
        r.set_drop_header(false);
        // 0b1 has one set bit (odd) -> odd_parity_ok == true -> consumed.
        assert!(matches!(r.feed_word(0b1, false), ReassembleResult::Pending));
    }

    #[test]
    fn bad_parity_ctrl_header_sets_second_header_parity() {
        use crate::device::control_packets::status::PktHandlerError;
        let mut r = StreamReassembler::new(0, 2); // drop_header=true: starts Idle

        // build_test_header sets no parity bit; word = 0x00000101, count_ones=2 (even) -> invalid.
        let header = build_test_header(0x101, 0, 0, 0);
        match r.feed_word(header, false) {
            ReassembleResult::HandlerError(PktHandlerError::SecondHeaderParity) => {}
            other => panic!("expected SecondHeaderParity, got {:?}", other),
        }
    }

    #[test]
    fn good_parity_ctrl_header_proceeds() {
        let mut r = StreamReassembler::new(0, 2);
        // build_test_header sets no parity bit; word = 0x00000100, count_ones=1 (odd) -> valid.
        let header = build_test_header(0x100, 0, 0, 0);
        assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
    }

    #[test]
    fn missing_final_tlast_on_write_sets_tlast_error() {
        use crate::device::control_packets::status::PktHandlerError;
        let mut r = StreamReassembler::new(0, 2);
        // 1-beat write. build_test_header(0x100,0,0,0) = 0x00000100 = 1 set bit (odd, valid).
        let header = build_test_header(0x100, 0, 0, 0);
        assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
        // Final beat WITHOUT tlast -- hardware would set Tlast_Error (bit 3).
        match r.feed_word(0xDEAD_BEEF, false) {
            ReassembleResult::HandlerError(PktHandlerError::Tlast) => {}
            other => panic!("expected Tlast, got {:?}", other),
        }

        // Recovery: missing-TLAST -> transition_after_complete(false) -> Idle.
        // A fresh valid packet must reassemble cleanly afterward.
        let header2 = build_test_header(0x100, 0, 0, 0); // 0x100, 1 bit, odd parity OK
        assert!(matches!(r.feed_word(header2, false), ReassembleResult::Pending));
        match r.feed_word(0x55, true) {
            ReassembleResult::Complete(pkt) => assert_eq!(pkt.data[0], 0x55),
            other => panic!("expected Complete after recovery, got {:?}", other),
        }
    }

    #[test]
    fn early_tlast_on_multibeat_write_sets_tlast_error() {
        use crate::device::control_packets::status::PktHandlerError;
        let mut r = StreamReassembler::new(0, 2);
        // 3-beat write (length_field=2 -> beats=3).
        // build_test_header(0x101,2,0,0) = 0x00200101 = 3 set bits (odd, valid).
        let header = build_test_header(0x101, 2, 0, 0);
        assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
        // TLAST on beat 1 of 3 -- unexpected early TLAST.
        match r.feed_word(0x11, true) {
            ReassembleResult::HandlerError(PktHandlerError::Tlast) => {}
            other => panic!("expected Tlast, got {:?}", other),
        }
    }

    #[test]
    fn correct_final_tlast_completes_normally() {
        let mut r = StreamReassembler::new(0, 2);
        // build_test_header(0x100,0,0,0) = 0x00000100 = 1 set bit (odd, valid).
        let header = build_test_header(0x100, 0, 0, 0);
        r.feed_word(header, false);
        // Final beat WITH tlast -- correct.
        match r.feed_word(0x42, true) {
            ReassembleResult::Complete(pkt) => assert_eq!(pkt.data[0], 0x42),
            other => panic!("expected Complete, got {:?}", other),
        }
    }

    #[test]
    fn read_packet_needs_no_tlast() {
        // OP_READ completes at the header (Idle arm); TLAST on the header beat
        // is irrelevant to Collecting-arm TLAST validation -- no Tlast error.
        let mut r = StreamReassembler::new(0, 2);
        // build_test_header(0x300,0,1,0): 0x300=2 bits + op=1 bit = 3 bits (odd, valid).
        let header = build_test_header(0x300, 0, 1, 0);
        match r.feed_word(header, false) {
            ReassembleResult::Complete(pkt) => assert_eq!(pkt.opcode, CtrlOpCode::Read),
            other => panic!("expected Complete, got {:?}", other),
        }
    }

    // Helper: build a control packet header word.
    // length_field is the raw 2-bit value (beats = length_field + 1).
    fn build_test_header(address: u32, length_field: u32, operation: u32, response_id: u32) -> u32 {
        use xdna_archspec::aie2::ctrl_packet::*;
        (address & ADDRESS_MASK)
            | ((length_field & LENGTH_MASK) << LENGTH_SHIFT)
            | ((operation & OPERATION_MASK) << OPERATION_SHIFT)
            | ((response_id & RESPONSE_ID_MASK) << RESPONSE_ID_SHIFT)
    }
}
