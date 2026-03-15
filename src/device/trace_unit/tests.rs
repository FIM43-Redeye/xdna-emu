//! Tests for the hardware trace unit emulation.

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
    // Per mlir-aie utils/trace/utils.py extract_tile():
    //   pkt_id  = (data >> 0)  & 0x1F  -- bits [4:0]
    //   pkt_type = (data >> 12) & 0x3  -- bits [13:12]
    //   row     = (data >> 16) & 0x1F  -- bits [20:16]
    //   col     = (data >> 21) & 0x7F  -- bits [27:21]
    //   parity  = bit 31 (odd parity)
    let header = packet[0];
    assert_eq!(header & 0x1F, 5); // packet_id
    assert_eq!((header >> 12) & 0x3, 0); // packet_type
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

/// Validate packet header against mlir-aie decode logic for all packet types.
///
/// The mlir-aie utils/trace/utils.py `parse_pkt_hdr_in_stream()` function
/// checks that bits [11:5] and [30:28] are zero and bit 19 is zero,
/// rejecting the header as invalid otherwise. This test exercises all
/// PacketType values (0=Core, 1=Mem, 2=ShimTile, 3=MemTile) to ensure
/// the emulator's header format passes the decoder's validity checks.
#[test]
fn test_packet_header_matches_mlir_aie_decoder() {
    /// Reimplements mlir-aie `parse_pkt_hdr_in_stream` validity check.
    fn mlir_aie_valid(w: u32) -> bool {
        // Odd parity check
        if w.count_ones() % 2 != 1 {
            return false;
        }
        // Reserved fields must be zero
        if ((w >> 5) & 0x7F) != 0 {
            return false;
        }
        if ((w >> 19) & 0x1) != 0 {
            return false;
        }
        if ((w >> 28) & 0x7) != 0 {
            return false;
        }
        true
    }

    /// Reimplements mlir-aie `extract_tile`.
    fn extract_tile(w: u32) -> (u32, u32, u32, u32) {
        let col = (w >> 21) & 0x7F;
        let row = (w >> 16) & 0x1F;
        let pkt_type = (w >> 12) & 0x3;
        let pkt_id = w & 0x1F;
        (col, row, pkt_type, pkt_id)
    }

    for pkt_type in 0u8..4 {
        let col = 2u8;
        let row = 3u8;
        let pkt_id = 5u8;

        let mut tu = TraceUnit::new(col, row);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x04, ((pkt_type as u32) << 12) | (pkt_id as u32));
        tu.write_register(0x10, 37);

        tu.notify_event(28, 0);
        for i in 1..=20 {
            tu.notify_event(37, i as u64);
        }

        let packet = tu.pop_packet().unwrap();
        let header = packet[0];

        assert!(
            mlir_aie_valid(header),
            "pkt_type={}: header 0x{:08X} rejected by mlir-aie decoder",
            pkt_type, header
        );

        let (dec_col, dec_row, dec_type, dec_id) = extract_tile(header);
        assert_eq!(dec_col, col as u32, "col mismatch for pkt_type={}", pkt_type);
        assert_eq!(dec_row, row as u32, "row mismatch for pkt_type={}", pkt_type);
        assert_eq!(dec_type, pkt_type as u32, "pkt_type mismatch for pkt_type={}", pkt_type);
        assert_eq!(dec_id, pkt_id as u32, "pkt_id mismatch for pkt_type={}", pkt_type);
    }
}
