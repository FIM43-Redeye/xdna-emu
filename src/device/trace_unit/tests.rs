//! Tests for the hardware trace unit emulation.

use super::*;

/// Notify a single event and immediately commit the frame for its cycle.
///
/// Matches the coordinator's per-cycle commit pass so tests exercising the
/// Single0/1/2 encoding paths see byte-buffer growth right after the call
/// without needing to chain an extra notify on a later cycle.
fn notify_commit(tu: &mut TraceUnit, hw_id: u8, cycle: u64) {
    tu.notify_event(hw_id, cycle, None);
    tu.commit_cycle(cycle);
}

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
    tu.notify_event(28, 0, None);
    assert_eq!(tu.state, TraceState::Running);

    // Matched event is encoded while running (per-cycle commit)
    let before = tu.byte_buffer.len();
    notify_commit(&mut tu, 37, 100);
    assert!(tu.byte_buffer.len() > before);

    // Stop event transitions to Stopped
    tu.notify_event(29, 300, None);
    assert_eq!(tu.state, TraceState::Stopped);

    // Events after stop are ignored
    tu.notify_event(37, 400, None);
    // No crash, no data
}

#[test]
fn test_single0_encoding() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37); // slot 0 = event 37

    // Start tracing
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len(); // Start marker = 8 bytes

    // Event with delta=5 (fits in Single0: 4-bit delta)
    notify_commit(&mut tu, 37, 5);
    assert_eq!(tu.byte_buffer.len(), start_len + 1); // Single0 = 1 byte

    // Verify encoding: slot=0, delta=5 -> 0b00000101 = 0x05
    assert_eq!(tu.byte_buffer[start_len], 0x05);
}

#[test]
fn test_single1_encoding() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8)); // slot 0=37, slot 1=38

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Event with delta=500 (fits in Single1: 10-bit delta)
    notify_commit(&mut tu, 37, 500);
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

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Slot 1, delta=500: format 0b100EEETT
    // byte0 = 0x80 | (1 << 2) | (500 >> 8 = 1) = 0x80 | 0x04 | 0x01 = 0x85
    // byte1 = 500 & 0xFF = 0xF4
    notify_commit(&mut tu, 38, 500);
    assert_eq!(tu.byte_buffer[start_len], 0x85);
    assert_eq!(tu.byte_buffer[start_len + 1], 0xF4);

    // Verify mlir-aie decode: event = (0x85 >> 2) & 7 = 0x21 & 7 = 1, cycles = (0x85 & 3)*256 + 0xF4 = 500
}

#[test]
fn test_single2_encoding() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37); // slot 0 = event 37

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Event with delta=100000 (fits in Single2: 18-bit delta)
    notify_commit(&mut tu, 37, 100000);
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

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Slot 3, delta=100000 = 0x186A0
    // Format: 0b101EEETT TTTTTTTT TTTTTTTT
    // byte0 = 0xA0 | (3 << 2) | (0x186A0 >> 16 = 1) = 0xA0 | 0x0C | 0x01 = 0xAD
    // byte1 = 0x86, byte2 = 0xA0
    notify_commit(&mut tu, 40, 100000);
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
    tu.notify_event(28, 0, None);

    // Generate enough Single0 events to fill 28 bytes. Each event fires on
    // its own cycle and the per-cycle commit encodes it as 1 byte.
    // Start marker is 8 bytes, so we need 20 more Single0 events.
    for i in 1..=20 {
        notify_commit(&mut tu, 37, i as u64); // delta=1 each time
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
    tu.notify_event(28, 0, None);
    notify_commit(&mut tu, 37, 5);
    notify_commit(&mut tu, 37, 10);

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
    tu.notify_event(28, 100, None);
    tu.notify_event(37, 200, None);
    assert!(!tu.has_pending_packets());
    assert!(tu.byte_buffer.is_empty());
}

#[test]
fn test_unmatched_event_ignored() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37); // Only slot 0 = 37

    tu.notify_event(28, 0, None); // Start
    let len_after_start = tu.byte_buffer.len();

    // Event 99 is not in any slot -- should be ignored
    notify_commit(&mut tu, 99, 50);
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
    tu.notify_event(28, 0, None);

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

    tu.notify_event(28, 0, None);
    // Generate 20 events (each on its own cycle) to fill a packet
    for i in 1..=20 {
        notify_commit(&mut tu, 37, i as u64);
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

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Slot 0, delta=1: 0b00000001 = 0x01
    notify_commit(&mut tu, 37, 1);
    assert_eq!(tu.byte_buffer[start_len], 0x01);

    // Slot 1, delta=1: 0b00010001 = 0x11
    notify_commit(&mut tu, 38, 2);
    assert_eq!(tu.byte_buffer[start_len + 1], 0x11);

    // Slot 2, delta=1: 0b00100001 = 0x21
    notify_commit(&mut tu, 39, 3);
    assert_eq!(tu.byte_buffer[start_len + 2], 0x21);

    // Slot 3, delta=1: 0b00110001 = 0x31
    notify_commit(&mut tu, 40, 4);
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
            let cycles = ((b0 & 0x03) as u64) * 65536 + (buf[1] as u64) * 256 + buf[2] as u64;
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
        (0, 1),
        (1, 1),
        (7, 1),
        (15, 1),
        // Single1 (2 bytes)
        (16, 2),
        (100, 2),
        (500, 2),
        (1023, 2),
        // Single2 (3 bytes)
        (1024, 3),
        (10000, 3),
        (100000, 3),
        (262143, 3),
    ];

    for slot in 0u8..8 {
        for &(d, expected_size) in all_deltas {
            let mut tu = TraceUnit::new(0, 2);
            tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
            let evt0 = 37u32 | (38 << 8) | (39 << 16) | (40 << 24);
            let evt1 = 41u32 | (42 << 8) | (43 << 16) | (44 << 24);
            tu.write_register(0x10, evt0);
            tu.write_register(0x14, evt1);
            tu.notify_event(28, 0, None); // start
            let start_len = tu.byte_buffer.len(); // 8 bytes start marker

            let event_id = 37 + slot;
            notify_commit(&mut tu, event_id, d); // delta = d (from start cycle 0)
            let base = start_len;
            assert!(
                tu.byte_buffer.len() >= base + expected_size,
                "buffer too short for slot={} delta={}: have {} bytes, need {}",
                slot,
                d,
                tu.byte_buffer.len() - base,
                expected_size
            );
            let (dec_slot, dec_delta, consumed) = decode_single(&tu.byte_buffer[base..]);
            assert_eq!(
                dec_slot, slot,
                "slot mismatch: slot={} delta={} byte0=0x{:02X}",
                slot, d, tu.byte_buffer[base]
            );
            assert_eq!(dec_delta, d, "delta mismatch: slot={} delta={}", slot, d);
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
    tu.notify_event(28, 0, None);
    assert_eq!(tu.read_register(0x08), 1 << 8); // Running=1 at bits [9:8]

    // Stop tracing -> state becomes Stopped
    tu.notify_event(29, 100, None);
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

        tu.notify_event(28, 0, None);
        for i in 1..=20 {
            notify_commit(&mut tu, 37, i as u64);
        }

        let packet = tu.pop_packet().unwrap();
        let header = packet[0];

        assert!(
            mlir_aie_valid(header),
            "pkt_type={}: header 0x{:08X} rejected by mlir-aie decoder",
            pkt_type,
            header
        );

        let (dec_col, dec_row, dec_type, dec_id) = extract_tile(header);
        assert_eq!(dec_col, col as u32, "col mismatch for pkt_type={}", pkt_type);
        assert_eq!(dec_row, row as u32, "row mismatch for pkt_type={}", pkt_type);
        assert_eq!(dec_type, pkt_type as u32, "pkt_type mismatch for pkt_type={}", pkt_type);
        assert_eq!(dec_id, pkt_id as u32, "pkt_id mismatch for pkt_type={}", pkt_type);
    }
}

// ===== Per-cycle coalescing / Multiple encoding (#138 regression tests) =====

/// Two simultaneous events in the same cycle must coalesce into one
/// Multiple0 frame (2 bytes), not two Single0 frames (1 byte each). The
/// slot bits should appear in the mask portion and the cycle delta should
/// appear once in the delta portion. This matches AM020 event-time mode
/// and mlir-aie's `convert_to_commands` decoder for Multiple0.
#[test]
fn test_multiple0_two_slots_same_cycle() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    // slot 0 = 37, slot 3 = 40
    tu.write_register(0x10, 37 | (40 << 24));

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Two events fire on cycle 5 (delta=5 from last emitted frame at 0).
    tu.notify_event(37, 5, None);
    tu.notify_event(40, 5, None);
    tu.commit_cycle(5);

    // Multiple0 = 2 bytes. Expected mask = 0b00001001 (slots 0 and 3).
    assert_eq!(
        tu.byte_buffer.len(),
        start_len + 2,
        "popcount=2 events should coalesce into Multiple0 (2 bytes)"
    );

    // byte0 bits [7:4] = 0b1100, bits [3:0] = mask[7:4] = 0b0000
    // byte1 bits [7:4] = mask[3:0] = 0b1001, bits [3:0] = delta = 5
    assert_eq!(tu.byte_buffer[start_len], 0xC0);
    assert_eq!(tu.byte_buffer[start_len + 1], (0b1001 << 4) | 5);
}

/// Three simultaneous events with delta too large for Multiple0 must use
/// Multiple1 (3 bytes, 10-bit delta).
#[test]
fn test_multiple1_three_slots_delta_500() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (40 << 24)); // slots 0,1,3

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    tu.notify_event(37, 500, None);
    tu.notify_event(38, 500, None);
    tu.notify_event(40, 500, None);
    tu.commit_cycle(500);

    // Multiple1 = 3 bytes, mask=0b00001011, delta=500=0x1F4
    assert_eq!(tu.byte_buffer.len(), start_len + 3);
    let mask: u8 = 0b0000_1011;
    let delta: u16 = 500;
    // byte0 = 0b110100EE where E is mask[7:6]
    assert_eq!(tu.byte_buffer[start_len], 0xD0 | (mask >> 6));
    // byte1 = mask[5:0] in [7:2], delta[9:8] in [1:0]
    assert_eq!(tu.byte_buffer[start_len + 1], ((mask & 0x3F) << 2) | ((delta >> 8) as u8 & 0x03));
    assert_eq!(tu.byte_buffer[start_len + 2], (delta & 0xFF) as u8);
}

/// All 8 slots firing in the same cycle with a large delta must use
/// Multiple2 (4 bytes, 18-bit delta). Regression test for #138: without
/// per-cycle coalescing this would emit 8 Single-format frames inflating
/// the byte stream by ~4x and overloading routing.
#[test]
fn test_multiple2_all_slots_delta_100000() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    // Fill all 8 slots with distinct event IDs.
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (40 << 24));
    tu.write_register(0x14, 41 | (42 << 8) | (43 << 16) | (44 << 24));

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    for id in 37..=44 {
        tu.notify_event(id, 100000, None);
    }
    tu.commit_cycle(100000);

    // Multiple2 = 4 bytes, mask=0xFF, delta=100000=0x186A0 (18 bits)
    assert_eq!(tu.byte_buffer.len(), start_len + 4);
    let mask: u8 = 0xFF;
    let delta: u32 = 100000;
    assert_eq!(tu.byte_buffer[start_len], 0xD4 | (mask >> 6));
    assert_eq!(tu.byte_buffer[start_len + 1], ((mask & 0x3F) << 2) | ((delta >> 16) as u8 & 0x03));
    assert_eq!(tu.byte_buffer[start_len + 2], ((delta >> 8) & 0xFF) as u8);
    assert_eq!(tu.byte_buffer[start_len + 3], (delta & 0xFF) as u8);
}

/// Mask decodes correctly via mlir-aie's `convert_to_commands` discriminator
/// logic for each Multiple format. This is an end-to-end round-trip check
/// that guarantees downstream tooling will read the bitmask exactly right.
#[test]
fn test_multiple_roundtrip_matches_mlir_aie() {
    fn decode_multiple(buf: &[u8]) -> (u8, u64, usize) {
        let b0 = buf[0];
        if (b0 & 0xF0) == 0xC0 {
            // Multiple0
            let cycles = (buf[1] & 0x0F) as u64;
            let mask = ((b0 & 0x0F) << 4) | (buf[1] >> 4);
            (mask, cycles, 2)
        } else if (b0 & 0xFC) == 0xD0 {
            // Multiple1
            let mask = ((b0 & 0x03) << 6) | (buf[1] >> 2);
            let cycles = (((buf[1] & 0x03) as u64) << 8) | buf[2] as u64;
            (mask, cycles, 3)
        } else if (b0 & 0xFC) == 0xD4 {
            // Multiple2
            let mask = ((b0 & 0x03) << 6) | (buf[1] >> 2);
            let cycles = (((buf[1] & 0x03) as u64) << 16) | ((buf[2] as u64) << 8) | buf[3] as u64;
            (mask, cycles, 4)
        } else {
            panic!("not a Multiple byte: 0x{:02X}", b0);
        }
    }

    let cases: &[(u8, u64, usize)] = &[
        (0b0000_0011, 0, 2),      // Multiple0, delta=0
        (0b1100_0011, 15, 2),     // Multiple0, max delta
        (0b0101_0101, 16, 3),     // Multiple1 threshold
        (0b1111_1111, 1023, 3),   // Multiple1 max
        (0b1111_0000, 1024, 4),   // Multiple2 threshold
        (0b1100_0011, 262143, 4), // Multiple2 max
    ];

    for &(mask, delta, expected_size) in cases {
        let mut tu = TraceUnit::new(0, 2);
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (40 << 24));
        tu.write_register(0x14, 41 | (42 << 8) | (43 << 16) | (44 << 24));

        tu.notify_event(28, 0, None);
        let base = tu.byte_buffer.len();

        for i in 0u8..8 {
            if mask & (1 << i) != 0 {
                tu.notify_event(37 + i, delta, None);
            }
        }
        tu.commit_cycle(delta);

        assert_eq!(
            tu.byte_buffer.len(),
            base + expected_size,
            "mask=0b{:08b} delta={}: encoded size",
            mask,
            delta
        );
        let (dec_mask, dec_delta, consumed) = decode_multiple(&tu.byte_buffer[base..]);
        assert_eq!(dec_mask, mask, "mask roundtrip for 0b{:08b} delta={}", mask, delta);
        assert_eq!(dec_delta, delta, "delta roundtrip for 0b{:08b} delta={}", mask, delta);
        assert_eq!(consumed, expected_size, "size roundtrip for 0b{:08b} delta={}", mask, delta);
    }
}

/// Events that arrive for different cycles must each get their own frame.
/// Lazy commit on cycle change: notifying a new cycle while a prior cycle
/// has pending slots flushes the prior frame first.
#[test]
fn test_lazy_commit_on_cycle_change() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8)); // slot 0 = 37, slot 1 = 38

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // Cycle 5: slot 0 fires. No commit yet.
    tu.notify_event(37, 5, None);
    assert_eq!(tu.byte_buffer.len(), start_len);
    assert_eq!(tu.pending_slot_mask, 0b0000_0001);

    // Cycle 10: slot 1 fires. Previous cycle's frame commits first.
    tu.notify_event(38, 10, None);
    // Single0 for slot 0 delta=5 = 0x05 (1 byte committed)
    assert_eq!(tu.byte_buffer.len(), start_len + 1);
    assert_eq!(tu.byte_buffer[start_len], 0x05);
    assert_eq!(tu.pending_slot_mask, 0b0000_0010);

    // commit_cycle(10) flushes the cycle-10 frame.
    tu.commit_cycle(10);
    // Single0 for slot 1 delta=5 = 0b00010101 = 0x15
    assert_eq!(tu.byte_buffer.len(), start_len + 2);
    assert_eq!(tu.byte_buffer[start_len + 1], 0x15);
}

/// With 8 slots firing every cycle for many cycles, the byte rate must stay
/// bounded (at most 4 bytes/cycle for Multiple2 worst case) instead of
/// scaling linearly with event count (which would be 8 Single0s/cycle = 8
/// bytes/cycle). This is the core property that fixes #138's packet pile-up.
#[test]
fn test_per_cycle_coalesce_bounds_byte_rate() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (40 << 24));
    tu.write_register(0x14, 41 | (42 << 8) | (43 << 16) | (44 << 24));

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    const CYCLES: u64 = 100;
    for cycle in 1..=CYCLES {
        for id in 37..=44 {
            tu.notify_event(id, cycle, None);
        }
        tu.commit_cycle(cycle);
    }

    let bytes_emitted = tu.byte_buffer.len() + tu.pending_words.len() * 4 - start_len;
    // Worst-case with delta=1: each cycle emits one Multiple0 = 2 bytes.
    // Upper bound: CYCLES * 2 (plus some rounding from packetization).
    // Legacy Single0-per-event encoding would emit CYCLES * 8 = 800 bytes;
    // Multiple-coalesced encoding emits CYCLES * 2 = 200 bytes.
    assert!(
        bytes_emitted as u64 <= CYCLES * 3,
        "8 slots over {} cycles should coalesce to <= {} bytes, got {}",
        CYCLES,
        CYCLES * 3,
        bytes_emitted
    );
    assert!(
        bytes_emitted as u64 >= CYCLES * 2,
        "expected at least {} bytes for {} cycles of Multiple0, got {}",
        CYCLES * 2,
        CYCLES,
        bytes_emitted
    );
}

// ===== Mode-1 (EventPC) encoder tests =====

/// Decoded frame from a mode-1 byte stream.
/// Used by `decode_mode1_for_test` to verify encoder round-trips.
#[derive(Debug, PartialEq, Eq)]
enum DecodedFrame {
    Start { timer_value: u64 },
    EventPc { mask: u8, pc: u16 },
}

/// Minimal mode-1 byte-stream decoder, ported from
/// tools/trace_decoder/modes/mode1.py. Handles only Start and EventPC;
/// Skip (0xFE) is ignored. Enough to validate encoder round-trips.
fn decode_mode1_for_test(bytes: &[u8]) -> Vec<DecodedFrame> {
    let mut frames = Vec::new();
    let n = bytes.len();
    let mut cursor = 0usize;

    while cursor < n {
        let b = bytes[cursor];

        // Skip (idle filler)
        if b == 0xFE {
            cursor += 1;
            continue;
        }

        // Start: 1111 0XX1 + 7 timer bytes
        // Matches mode-1 Start (0xF1 / 0xF5). bit 0 = 1 is the mode discriminator.
        if (b & 0b1111_0011) == 0b1111_0001 {
            if cursor + 7 >= n {
                break;
            }
            let mut timer_value: u64 = 0;
            for i in 0..7 {
                timer_value = (timer_value << 8) | (bytes[cursor + 1 + i] as u64);
            }
            frames.push(DecodedFrame::Start { timer_value });
            cursor += 8;
            continue;
        }

        // EventPC: 1100 01EE EEEEEERR RRPPPPPP PPPPPPPP  (4 bytes)
        // Discriminator: (b & 0b11111100) == 0b11000100
        if (b & 0b1111_1100) == 0b1100_0100 {
            if cursor + 3 >= n {
                break;
            }
            let b1 = bytes[cursor + 1];
            let b2 = bytes[cursor + 2];
            let b3 = bytes[cursor + 3];
            let mask = ((b & 0b11) << 6) | (b1 >> 2);
            let pc = (((b2 & 0b0011_1111) as u16) << 8) | (b3 as u16);
            frames.push(DecodedFrame::EventPc { mask, pc });
            cursor += 4;
            continue;
        }

        // Unknown byte: skip
        cursor += 1;
    }

    frames
}

/// Verify encoder output round-trips through the mode-1 decoder with the
/// expected byte layout documented in tools/trace_decoder/modes/mode1.py.
///
/// Slot 3 carries LOCK_STALL (hw_id 26). Two consecutive EventPC frames
/// with the same PC are expected because the events fire on cycles 0 and 1.
#[test]
fn mode1_encoder_byte_equivalent_to_decoder_fixture() {
    let mut tu = TraceUnit::new(0, 2);
    tu.set_packet_type(0); // core
    tu.set_mode(TraceMode::EventPc);
    tu.set_event_slot(3, 26); // LOCK_STALL = hw_id 26 in slot 3
    tu.set_start_event(1);

    tu.notify_event(1, 0, None); // start (fires state machine)
    tu.notify_event(26, 0, Some(816)); // EventPC mask=0b1000 (slot 3), pc=816
    tu.commit_cycle(0);
    tu.notify_event(26, 1, Some(816)); // next cycle, same PC, same slot
    tu.commit_cycle(1);

    let bytes = tu.encoded_bytes();
    let decoded = decode_mode1_for_test(bytes);

    // Expect Start + 2x EventPC
    assert_eq!(decoded.len(), 3, "expected Start + 2 EventPC frames, got: {:?}", decoded);
    assert!(
        matches!(decoded[0], DecodedFrame::Start { .. }),
        "first frame must be Start, got {:?}",
        decoded[0]
    );
    assert_eq!(decoded[1], DecodedFrame::EventPc { mask: 0b1000, pc: 816 }, "first EventPC frame mismatch");
    assert_eq!(decoded[2], DecodedFrame::EventPc { mask: 0b1000, pc: 816 }, "second EventPC frame mismatch");
}

/// Non-core trace units (packet_type != 0) must be clamped to EventTime when
/// EventPc mode is requested. Per regdb (aie_registers_aie2.json), only the
/// core-module Trace_Control0 has a Mode bitfield; setting EventPc on
/// memtile/memmod/shim trace units would be a HW-impossible state.
#[test]
fn mode1_on_non_core_trace_unit_clamps_to_event_time() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(3); // memtile
    tu.set_mode(TraceMode::EventPc); // attempt mode 1 on non-core
    assert_eq!(
        tu.mode(),
        TraceMode::EventTime,
        "memtile trace unit must reject EventPc mode (regdb: no Mode bitfield on memtile Trace_Control0)"
    );
}

/// When mode-1 receives an event with no PC (pc=None), it encodes a sentinel
/// pc=0 frame rather than crashing or dropping the event. This allows
/// no-PC callers (like the current coordinator path) to still produce a
/// valid trace stream.
#[test]
fn mode1_no_pc_emits_sentinel_zero() {
    let mut tu = TraceUnit::new(0, 2);
    tu.set_packet_type(0); // core
    tu.set_mode(TraceMode::EventPc);
    tu.set_event_slot(0, 23); // MEMORY_STALL
    tu.set_start_event(1);

    tu.notify_event(1, 0, None); // start
    tu.notify_event(23, 0, None); // memory stall, no PC -> sentinel pc=0
    tu.commit_cycle(0);

    let frames = decode_mode1_for_test(tu.encoded_bytes());
    let event_frames: Vec<&DecodedFrame> =
        frames.iter().filter(|f| matches!(f, DecodedFrame::EventPc { .. })).collect();
    assert_eq!(event_frames.len(), 1, "expected exactly one EventPC frame");
    assert_eq!(
        *event_frames[0],
        DecodedFrame::EventPc { mask: 0b0000_0001, pc: 0 },
        "no-PC event should encode sentinel pc=0 with mask=0b1 (slot 0)"
    );
}

// ===== Mode-2 (Execution) bit accumulator scaffolding =====

#[test]
fn bit_accumulator_packs_msb_first() {
    let mut tu = TraceUnit::new(0, 1);
    tu.push_bits(0b1010, 4);
    tu.push_bits(0b1100, 4);
    // Now pending_word_bits == 8, pending_word == 0b1010_1100 << 24
    // Continue to a full word:
    tu.push_bits(0, 24);
    // Should have flushed one word: 0xAC000000 -> bytes AC 00 00 00
    let bytes = tu.encoded_bytes();
    assert_eq!(bytes, &[0xAC, 0x00, 0x00, 0x00]);
}

#[test]
fn bit_accumulator_flushes_at_32_bits() {
    let mut tu = TraceUnit::new(0, 1);
    tu.push_bits(0xDEADBEEF, 32);
    let bytes = tu.encoded_bytes();
    assert_eq!(bytes, &[0xDE, 0xAD, 0xBE, 0xEF]);
}

#[test]
fn align_pads_partial_word_with_filler0() {
    let mut tu = TraceUnit::new(0, 1);
    // Push 12 bits (3 nibbles); align should pad 5 nibbles of Filler0
    // (each Filler0 = 4 bits 0010), then flush.
    tu.push_bits(0xABC, 12);
    tu.align_to_word_via_filler0();
    // Expect: 0xABC followed by 5 nibbles of 0010 = 0x22222
    // Total: 0xABC22222 -> bytes [0xAB, 0xC2, 0x22, 0x22]
    assert_eq!(tu.encoded_bytes(), &[0xAB, 0xC2, 0x22, 0x22]);
}

#[test]
fn align_is_noop_when_word_boundary() {
    let mut tu = TraceUnit::new(0, 1);
    tu.push_bits(0xDEADBEEF, 32);
    let before = tu.encoded_bytes().len();
    tu.align_to_word_via_filler0();
    assert_eq!(tu.encoded_bytes().len(), before);
}

#[test]
fn emit_long_frame_aligns_then_pushes_word() {
    let mut tu = TraceUnit::new(0, 1);
    tu.push_bits(0b1110_0011, 8); // partial word (Repeat0 example)
    tu.emit_long_frame(0xCAFEBABE);
    // Expect: partial word padded with 6 nibbles Filler0 = 0xE3222222
    // Then long frame as next word: 0xCAFEBABE
    assert_eq!(tu.encoded_bytes(), &[0xE3, 0x22, 0x22, 0x22, 0xCA, 0xFE, 0xBA, 0xBE]);
}

#[test]
fn round_trip_atoms() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_atom(true); // E_atom
    tu.encode_atom(false); // N_atom
    tu.encode_atom(true); // E_atom
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    // Expect ENE then 5 Filler0 nibbles to reach word boundary.
    assert!(summary.starts_with("ENE"), "got: {}", summary);
}

#[test]
fn round_trip_new_pc() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_new_pc(0x1234);
    tu.encode_new_pc(0x0042);
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(frames
        .iter()
        .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::NewPc { pc: 0x1234 })));
    assert!(frames
        .iter()
        .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::NewPc { pc: 0x0042 })));
}

#[test]
fn round_trip_lc() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_lc(1, 0xABCDEF);
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(frames
        .iter()
        .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::Lc { flag: 1, count: 0xABCDEF })));
}

#[test]
fn round_trip_lc_after_partial_word() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_atom(true); // 4 bits
    tu.encode_atom(false); // 4 bits
    tu.encode_lc(0, 8); // long frame: aligns first
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    // Expect E + N + 6 Filler0s + LC(0,8)
    assert!(summary.contains("EN"));
    assert!(summary.contains("LC(0,8)"));
}

#[test]
fn round_trip_repeat0() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_repeat0(5);
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(frames
        .iter()
        .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::Repeat0 { count: 5 })));
}

#[test]
fn round_trip_repeat1() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_repeat1(0x1FF);
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(frames
        .iter()
        .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::Repeat1 { count: 0x1FF })));
}

#[test]
fn round_trip_mode2_start() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_mode2_start(0x100);
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(frames
        .iter()
        .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::Start { anchor_pc: 0x100 })));
}

#[test]
fn round_trip_mode2_stop() {
    let mut tu = TraceUnit::new(0, 1);
    tu.encode_mode2_stop();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(frames.iter().any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::Stop)));
}

// -- RLE state machine (Task 1.8) --
//
// `append_atom_to_run` accumulates same-polarity atoms into a run; the
// run is flushed by `flush_atoms_run` as one base atom plus zero or more
// Repeat0/Repeat1 frames.  Polarity flips force a flush before starting
// a new run.

#[test]
fn rle_single_atom_no_repeat() {
    let mut tu = TraceUnit::new(0, 1);
    tu.append_atom_to_run(true);
    tu.flush_atoms_run();
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    assert!(summary.starts_with("E"), "got: {}", summary);
    assert!(!summary.contains("R0") && !summary.contains("R1"), "single atom should not RLE: {}", summary);
}

#[test]
fn rle_uses_repeat0_for_short_runs() {
    let mut tu = TraceUnit::new(0, 1);
    for _ in 0..16 {
        tu.append_atom_to_run(true);
    }
    tu.flush_atoms_run();
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    // Expect: E + R0(15)
    assert!(summary.starts_with("ER0(15)"), "got: {}", summary);
}

#[test]
fn rle_uses_repeat1_for_medium_runs() {
    let mut tu = TraceUnit::new(0, 1);
    for _ in 0..1024 {
        tu.append_atom_to_run(false);
    }
    tu.flush_atoms_run();
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    // Expect: N + R1(1023)
    assert!(summary.starts_with("NR1(1023)"), "got: {}", summary);
}

#[test]
fn rle_chains_repeat1_for_long_runs() {
    let mut tu = TraceUnit::new(0, 1);
    for _ in 0..1500 {
        tu.append_atom_to_run(true);
    }
    tu.flush_atoms_run();
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    // 1500 atoms = atom(E) + R1(1023) + R1(476): each R1(N) represents
    // N additional same-polarity atoms beyond the base. 1 + 1023 + 476 = 1500.
    // (The plan's draft expected R1(475) but that totals 1499; corrected here.)
    assert!(summary.starts_with("ER1(1023)R1(476)"), "got: {}", summary);
}

#[test]
fn rle_polarity_flip_breaks_run() {
    let mut tu = TraceUnit::new(0, 1);
    for _ in 0..5 {
        tu.append_atom_to_run(true);
    }
    for _ in 0..3 {
        tu.append_atom_to_run(false);
    }
    tu.flush_atoms_run();
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    // 5 E + 3 N => E + R0(4) + N + R0(2)
    assert!(summary.starts_with("ER0(4)NR0(2)"), "got: {}", summary);
}

#[test]
fn mode2_notify_atom_taken_emits_e_atom_on_commit() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0); // core
    tu.set_mode(TraceMode::Execution);
    tu.set_event_slot(0, 1); // TRUE = slot 0
    tu.set_start_event(1);
    tu.notify_event(1, 0, None); // start event fires at cycle 0
    tu.notify_atom(true);
    tu.commit_cycle(0);
    // Align pending bits out to bytes so the decoder can read them.
    // (We don't call flush() here because flush() drains byte_buffer
    // into the packet pipeline, leaving encoded_bytes() empty.)
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    assert!(summary.contains("Start("), "got: {}", summary);
    assert!(summary.contains("E"), "got: {}", summary);
}

#[test]
fn mode2_notify_branch_taken_queues_new_pc() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::Execution);
    tu.set_event_slot(0, 1);
    tu.set_start_event(1);
    tu.notify_event(1, 0, None);
    tu.notify_atom(true);
    tu.notify_branch_taken(0, 0x300);
    tu.commit_cycle(0);
    tu.align_to_word_via_filler0();
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    let summary = crate::trace::mode2_decode::frame_summary(&frames);
    assert!(summary.contains("PC(0x300)"), "got: {}", summary);
}

#[test]
fn mode2_notify_in_other_mode_is_noop() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_mode(TraceMode::EventTime);
    tu.notify_atom(true);
    tu.notify_branch_taken(0, 0x300);
    tu.notify_loop_boundary(0, 8, 7);
    // No mode-2 buffering happened.
    assert_eq!(tu.encoded_bytes_len(), 0);
    assert_eq!(tu.pending_word_bits, 0);
    assert!(tu.pending_atoms_run.is_none());
    assert!(tu.pending_mode2_frames.is_empty());
}

#[test]
fn mode2_notify_when_idle_is_noop() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::Execution);
    // No start event fired -- state == Idle
    tu.notify_atom(true);
    tu.notify_branch_taken(0, 0x300);
    assert_eq!(tu.encoded_bytes_len(), 0);
    assert!(tu.pending_atoms_run.is_none());
    assert!(tu.pending_mode2_frames.is_empty());
}

#[test]
fn mode2_notify_loop_boundary_emits_once_per_zol() {
    // Phase 0 finding: HW emits exactly ONE LC frame per ZOL invocation,
    // at loop start, with count = trip count and flag = 0. The interpreter
    // calls notify_loop_boundary at every iteration's LE_PC; the trace_unit
    // must dedupe and only emit on the first call.
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::Execution);
    tu.set_event_slot(0, 1);
    tu.set_start_event(1);
    tu.notify_event(1, 0, None);

    // Simulate an 8-iteration ZOL: lc_before walks 8,7,6,5,4,3,2,1; lc_after = lc_before-1.
    for lc in (1..=8u32).rev() {
        tu.notify_loop_boundary(0, lc, lc - 1);
    }

    let lc_frames: Vec<_> = tu
        .pending_mode2_frames
        .iter()
        .filter(|f| matches!(f, super::PendingMode2Frame::Lc { .. }))
        .collect();
    assert_eq!(lc_frames.len(), 1, "expected exactly 1 LC frame, got {:?}", lc_frames);
    match lc_frames[0] {
        super::PendingMode2Frame::Lc { flag, count } => {
            assert_eq!(*flag, 0, "flag should be 0 (Phase 0 finding)");
            assert_eq!(*count, 8, "count should equal trip count, got {}", count);
        }
        _ => unreachable!(),
    }
}

#[test]
fn mode2_notify_loop_boundary_back_to_back_zols() {
    // Two ZOL invocations should produce two LC frames: state must reset
    // when LC reaches 0 so the next ZOL's first iteration emits afresh.
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::Execution);
    tu.set_event_slot(0, 1);
    tu.set_start_event(1);
    tu.notify_event(1, 0, None);

    // First ZOL: 4 iterations.
    for lc in (1..=4u32).rev() {
        tu.notify_loop_boundary(0, lc, lc - 1);
    }
    // Second ZOL: 6 iterations.
    for lc in (1..=6u32).rev() {
        tu.notify_loop_boundary(0, lc, lc - 1);
    }

    let counts: Vec<u32> = tu
        .pending_mode2_frames
        .iter()
        .filter_map(|f| match f {
            super::PendingMode2Frame::Lc { count, .. } => Some(*count),
            _ => None,
        })
        .collect();
    assert_eq!(counts, vec![4, 6], "expected one LC per ZOL with the trip count");
}

#[test]
fn rewriting_control0_resets_state_for_rerun() {
    // Models the bridge-runner batch-stdin pattern: each batch's insts.bin
    // re-issues the Trace_Control0 write as part of its trace setup. A new
    // configuration must yield a clean Idle state -- otherwise after the
    // first batch's stop_event fires, the unit latches in Stopped and no
    // subsequent start_event can re-arm it.
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    // Arm: mode=EventTime(0), start_event=28, stop_event=29
    tu.write_register(0x00, (29 << 24) | (28 << 16));
    tu.set_event_slot(0, 28);
    tu.notify_event(28, 0, None); // start
    tu.notify_event(29, 100, None); // stop -- transitions to Stopped + flush
    assert!(matches!(tu.state, super::TraceState::Stopped));
    let after_first = tu.encoded_bytes_len();

    // Second "batch": rewrite Control0 with the same config. State should
    // reset to Idle and buffers should be cleared so the next start fires.
    tu.write_register(0x00, (29 << 24) | (28 << 16));
    assert!(matches!(tu.state, super::TraceState::Idle), "state after re-config");
    assert_eq!(tu.encoded_bytes_len(), 0, "byte_buffer should be cleared on reconfigure");

    // Re-arm and verify start works again.
    tu.notify_event(28, 200, None);
    assert!(matches!(tu.state, super::TraceState::Running), "rerun start should fire");
    let _ = after_first; // silence unused-var lint
}

#[test]
fn mode2_start_uses_real_pc_anchor() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::Execution);
    tu.set_event_slot(0, 1);
    tu.set_start_event(1);
    tu.notify_event(1, 0, Some(0x1234));
    // Don't flush() -- it drains byte_buffer into pending_words.
    // Start is a 32-bit long frame so it's already byte-aligned.
    let frames = crate::trace::mode2_decode::decode(tu.encoded_bytes());
    assert!(
        frames
            .iter()
            .any(|f| matches!(f, crate::trace::mode2_decode::Mode2Frame::Start { anchor_pc: 0x1234 })),
        "expected Start with anchor 0x1234, got {:?}",
        frames
    );
}
