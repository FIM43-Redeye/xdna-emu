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

/// Fire `start_event` at cycle 0 AND immediately promote the trace unit to
/// Running, so subsequent test events at cycle 0 are captured.
///
/// HW pipelines start by 1 cycle: `start_event` arms the controller, but
/// events arriving in the same cycle as the arming event are NOT recorded.
/// Production code preserves that pipelining; most unit tests, however,
/// don't care about the pipeline gap — they just want a Running unit
/// from a clean baseline so they can exercise encoding logic. This helper
/// performs an unmatched-event notify on a +1 cycle to trip the promotion
/// path, then leaves `last_event_cycle` / `pending_cycle` at 0 so deltas
/// computed against later events match the pre-pipelining behavior.
fn force_start(tu: &mut TraceUnit, start_event: u8) {
    tu.notify_event(start_event, 0, None);
    // Sentinel hw_event_id 0xFF is never in any slot; this notify just
    // bumps the cycle counter past the arm cycle and triggers the
    // arm-promotion check at the top of notify_event without writing
    // any pending-frame state.
    tu.notify_event(0xFF, 1, None);
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

    // Start event arms the trace unit. HW pipelines the Idle→Running
    // transition by 1 cycle; until cycle advances past the arm cycle,
    // state remains Idle. (The arm-cycle latch is private; we can only
    // observe the side effect: state stays Idle, the Start marker bytes
    // are emitted into the buffer.)
    tu.notify_event(28, 0, None);
    assert_eq!(tu.state, TraceState::Idle);
    assert!(tu.byte_buffer.len() >= 8); // Start marker = 1 prefix + 7 timer bytes

    // First event in a later cycle promotes to Running and is encoded.
    let before = tu.byte_buffer.len();
    notify_commit(&mut tu, 37, 100);
    assert_eq!(tu.state, TraceState::Running);
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

    // 5 idle cycles (fits in Single0: 4-bit field): the event is in cycle 6
    notify_commit(&mut tu, 37, 6);
    assert_eq!(tu.byte_buffer.len(), start_len + 1); // Single0 = 1 byte

    // Verify encoding: slot=0, cycles field 5 -> 0b00000101 = 0x05
    assert_eq!(tu.byte_buffer[start_len], 0x05);
}

/// A frame's cycles field is the number of idle cycles since the previous frame (silicon:
/// markers 2..1500 bundles apart decode 2..1502 apart, so the frame's own cycle is the
/// decoder's implicit +1). The start cycle counts as the previous frame.
#[test]
fn cycles_field_is_the_idle_cycles_between_frames() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8));
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    notify_commit(&mut tu, 37, 1); // the very next cycle: nothing idle
    notify_commit(&mut tu, 38, 2); // and again
    notify_commit(&mut tu, 37, 7); // 4 idle cycles (3..6)
    tu.flush_repeats();
    assert_eq!(&tu.byte_buffer[start_len..], &[0x00, 0x10, 0x04]);
}

/// Identical consecutive frames are one frame plus Repeat tokens (silicon: a periodic
/// PERF_CNT_2 is `Single2(cyc 1024)` plus a Repeat, never one frame per period).
#[test]
fn identical_consecutive_frames_become_a_repeat() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37);
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    for n in 1..=4u64 {
        notify_commit(&mut tu, 37, 100 * n); // frames 99 idle cycles apart; the first is 99 too
    }
    tu.flush_repeats();
    // Single0 holds 4 bits only: 99 idle cycles is a Single1: 0x80 | slot 0 | hi 0, lo 99
    assert_eq!(&tu.byte_buffer[start_len..], &[0x80, 99, 0xE3]);
}

/// A held level writes a frame in every cycle it is asserted, run-length compressed: the
/// opening frame carries the idle cycles before it, the rest of the hold is a second frame
/// (cycles 0) plus a Repeat of the remaining cycles. An 8-cycle PORT_RUNNING hold after a
/// gap decodes on silicon as `Single0(gap) Single0(0) Repeat0(6)`.
#[test]
fn a_lone_hold_is_an_opening_frame_a_cycles_0_frame_and_a_repeat() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    force_start(&mut tu, 28);
    let start_len = tu.byte_buffer.len();

    // slot1 asserts at cycle 10 (9 idle cycles after the start at 0), is asserted in 10..=17
    // and drops at 18 with nothing coincident
    tu.set_event_level(38, 10, true);
    tu.commit_cycle(10);
    tu.set_event_level(38, 18, false);
    tu.commit_cycle(18);
    tu.flush_repeats();

    assert_eq!(&tu.byte_buffer[start_len..], &[0x19, 0x10, 0xE6]);
}

/// A hold opened in the very cycle after the previous frame has no distinct opening frame:
/// the frames are all `(mask, 0)` and compress into one frame plus a Repeat. A hold of
/// 6353 cycles is six Repeat1(1023) and a Repeat1(215) (silicon's first LOCK_STALL hold).
#[test]
fn a_long_hold_chunks_its_repeats_like_silicon() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    tu.set_event_level(26, 1, true); // frames in 1..=6353
    tu.commit_cycle(1);
    tu.set_event_level(26, 6354, false);
    tu.commit_cycle(6354);
    tu.flush_repeats();

    // one frame plus 6352 repeats: 6 x 1023 + 214 (the fills keep each opcode in a word)
    let mut expected = vec![0x30];
    for _ in 0..6 {
        expected.extend_from_slice(&[0xDB, 0xFF]);
    }
    expected.extend_from_slice(&[0xD8, 0xD6]);
    let body: Vec<u8> = tu.byte_buffer[start_len..].iter().copied().filter(|&b| b != 0xFE).collect();
    assert_eq!(body, expected);
}

/// Repeat counts up to 15 are a Repeat0, up to 1023 a Repeat1 (silicon corpus: no Repeat1 of
/// 15 or fewer, none of 0), longer runs are chunks of 1023.
#[test]
fn repeat_tokens_use_the_smallest_opcode() {
    for (owed, expected) in [
        (0u64, vec![]),
        (6, vec![0xE6]),
        (15, vec![0xEF]),
        (16, vec![0xD8, 0x10]),
        (1023, vec![0xDB, 0xFF]),
        (1024, vec![0xDB, 0xFF, 0xE1]),
    ] {
        let mut tu = TraceUnit::new(0, 2);
        tu.repeat_pending = owed;
        tu.flush_repeats();
        assert_eq!(tu.byte_buffer, expected, "{owed} repeats");
    }
}

/// A one-cycle pulse inside a hold is its own frame with the held mask; the cycle after it
/// is a held-only frame again (silicon: `Multiple(held|pulse) ; Multiple(held)`).
#[test]
fn pulse_during_hold_is_followed_by_a_held_only_frame() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    // slot0=37 (pulse), slot3=26 (LOCK_STALL, a level)
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    tu.set_event_level(26, 1, true);
    tu.commit_cycle(1);
    tu.notify_event(37, 40, None);
    tu.commit_cycle(40);
    tu.set_event_level(26, 50, false);
    tu.commit_cycle(50);
    tu.flush_repeats();

    // 1: (slot3, 0); 2..=39: the same, 38 frames; 40: (slot0|slot3, 0); 41..=49: (slot3, 0)
    // again, 9 frames; nothing at 50
    assert_eq!(
        &tu.byte_buffer[start_len..],
        &[0x30, 0xD8, 38, 0xFE, 0xC0, 0x90, 0x30, 0xE8],
        "got {:02x?}",
        &tu.byte_buffer[start_len..]
    );
    assert_eq!(tu.held_mask, 0);
}

/// A level that drops and re-asserts a cycle later is two holds: the frame in the re-asserted
/// cycle follows an idle cycle (silicon's PORT_RUNNING re-pulses are `Event(cyc=0) Repeat(14)
/// Event(cyc=1) Event(cyc=0) Repeat(14)`).
#[test]
fn a_dropped_and_reasserted_level_is_two_holds() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    tu.set_event_level(26, 1, true); // frames 1..=15
    tu.commit_cycle(1);
    tu.set_event_level(26, 16, false); // cycle 16 idle
    tu.commit_cycle(16);
    tu.set_event_level(26, 17, true); // frames 17..=31
    tu.commit_cycle(17);
    tu.set_event_level(26, 32, false);
    tu.commit_cycle(32);
    tu.flush_repeats();

    assert_eq!(
        &tu.byte_buffer[start_len..],
        &[0x30, 0xEE, 0x31, 0x30, 0xED],
        "got {:02x?}",
        &tu.byte_buffer[start_len..]
    );
}

/// With no levels asserted the encoder is a plain pulse encoder.
#[test]
fn pulse_events_with_no_held_levels() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8)); // slot0=37, slot1=38
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    notify_commit(&mut tu, 37, 5); // 4 idle cycles: Single0 0x04
    notify_commit(&mut tu, 38, 9); // 3 idle cycles: Single0 (1<<4)|3 = 0x13
    assert_eq!(tu.byte_buffer.len(), start_len + 2);
    assert_eq!(tu.byte_buffer[start_len], 0x04);
    assert_eq!(tu.byte_buffer[start_len + 1], 0x13);
    assert_eq!(tu.held_mask, 0, "no levels asserted");
}

/// Silicon never lets a multi-byte opcode straddle a 32-bit word of the payload: the rest of
/// the word is Skip (0xFE) and the opcode starts in the next one.
#[test]
fn multi_byte_opcodes_do_not_straddle_a_word() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8));
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();
    assert_eq!(start_len % 4, 0);

    notify_commit(&mut tu, 37, 2); // Single0 at offset 0
    notify_commit(&mut tu, 38, 4); // Single0 at offset 1
    notify_commit(&mut tu, 37, 7); // Single0 at offset 2
    notify_commit(&mut tu, 38, 200); // Single1 (2 bytes) would start at offset 3
    tu.flush_repeats();
    let body = &tu.byte_buffer[start_len..];
    for op_at in 0..body.len() {
        if body[op_at] & 0xE0 == 0x80 {
            assert!(op_at % 4 < 3, "a Single1 at word offset {} straddles: {:02x?}", op_at % 4, body);
        }
    }
    assert!(body.contains(&0xFE), "the straddling opcode needs a fill: {:02x?}", body);
}

/// Several level edges on the SAME cycle coalesce into ONE frame (AM020's one-frame-per-cycle
/// rule): `set_event_level` accumulates same-cycle held-mask changes and `commit_cycle`
/// writes the single snapshot.
#[test]
fn same_cycle_level_edges_coalesce_into_one_frame() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24)); // start=28, stop=29
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24)); // slot1=38,slot2=39,slot3=26
    force_start(&mut tu, 28);
    let start_len = tu.byte_buffer.len();

    // slot1 asserts @10; @20 both slot2 and slot3 assert; slot1 deasserts @30
    tu.set_event_level(38, 10, true);
    tu.commit_cycle(10);
    tu.set_event_level(39, 20, true);
    tu.set_event_level(26, 20, true);
    tu.commit_cycle(20);
    tu.set_event_level(38, 30, false);
    tu.commit_cycle(30);
    tu.flush_repeats();

    // 10: slot1 after 9 idle cycles; 11..=19 slot1 alone (a cycles-0 frame, 8 repeats);
    // 20..=29: slots 1-3 in ONE frame, 9 repeats; 30: slots 2-3 (the fills keep each
    // two-byte opcode inside a word)
    let body = &tu.byte_buffer[start_len..];
    assert_eq!(
        body,
        &[0x19, 0x10, 0xE8, 0xFE, 0xC0, 0xE0, 0xE9, 0xFE, 0xC0, 0xC0],
        "same-cycle level edges must coalesce into one frame, got {:02x?}",
        body
    );
}

#[test]
fn test_single1_encoding() {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8)); // slot 0=37, slot 1=38

    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();

    // 500 idle cycles (fits in Single1: 10-bit field)
    notify_commit(&mut tu, 37, 501);
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
    notify_commit(&mut tu, 38, 501);
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

    // 100000 idle cycles (fits in Single2: 18-bit field)
    notify_commit(&mut tu, 37, 100001);
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
    notify_commit(&mut tu, 40, 100001);
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
    // Alternate two slots: identical consecutive frames would be one frame plus a Repeat.
    for i in 1..=20 {
        notify_commit(&mut tu, if i % 2 == 0 { 37 } else { 38 }, i as u64);
    }
    // the output stage hands the packet over once its emit latency has passed
    commit_each(&mut tu, 21..=40);

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
    tu.write_register(0x10, 37 | (38 << 8));

    tu.notify_event(28, 0, None);
    // Generate 20 events (each on its own cycle, alternating slots) to fill a packet
    for i in 1..=20 {
        notify_commit(&mut tu, if i % 2 == 0 { 37 } else { 38 }, i as u64);
    }
    // the output stage hands the packet over once its emit latency has passed
    commit_each(&mut tu, 21..=40);

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

    // Slot 0, cycles field 0 (the cycle after the start): 0b00000000 = 0x00
    notify_commit(&mut tu, 37, 1);
    assert_eq!(tu.byte_buffer[start_len], 0x00);

    // Slot 1: 0b00010000 = 0x10
    notify_commit(&mut tu, 38, 2);
    assert_eq!(tu.byte_buffer[start_len + 1], 0x10);

    // Slot 2: 0b00100000 = 0x20
    notify_commit(&mut tu, 39, 3);
    assert_eq!(tu.byte_buffer[start_len + 2], 0x20);

    // Slot 3: 0b00110000 = 0x30
    notify_commit(&mut tu, 40, 4);
    assert_eq!(tu.byte_buffer[start_len + 3], 0x30);
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
            // HW pipelines start by 1 cycle; force_start arms the unit
            // and trips the promotion path so the test can use cycle 0
            // as its baseline (last_event_cycle = 0).
            force_start(&mut tu, 28);
            let start_len = tu.byte_buffer.len(); // 8 bytes start marker

            let event_id = 37 + slot;
            notify_commit(&mut tu, event_id, d + 1); // d idle cycles after the start cycle 0
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

    // Start tracing arms the unit; HW pipelines Idle→Running by 1 cycle,
    // so the status register stays Idle until cycle advances.
    tu.notify_event(28, 0, None);
    assert_eq!(tu.read_register(0x08), 0); // Idle (still arming)

    // First event in a later cycle promotes to Running.
    tu.notify_event(0xFF, 1, None); // sentinel cycle bump (unmatched event)
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
        tu.write_register(0x10, 37 | (38 << 8));

        tu.notify_event(28, 0, None);
        for i in 1..=20 {
            notify_commit(&mut tu, if i % 2 == 0 { 37 } else { 38 }, i as u64);
        }
        commit_each(&mut tu, 21..=40);

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
    // byte1 bits [7:4] = mask[3:0] = 0b1001, bits [3:0] = idle cycles = 4
    assert_eq!(tu.byte_buffer[start_len], 0xC0);
    assert_eq!(tu.byte_buffer[start_len + 1], (0b1001 << 4) | 4); // cycle 5: 4 idle cycles
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

    tu.notify_event(37, 501, None);
    tu.notify_event(38, 501, None);
    tu.notify_event(40, 501, None);
    tu.commit_cycle(501); // 500 idle cycles

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
        tu.notify_event(id, 100001, None);
    }
    tu.commit_cycle(100001); // 100000 idle cycles

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

        force_start(&mut tu, 28);
        let base = tu.byte_buffer.len();

        for i in 0u8..8 {
            if mask & (1 << i) != 0 {
                tu.notify_event(37 + i, delta + 1, None); // delta idle cycles
            }
        }
        tu.commit_cycle(delta + 1);

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
    // Single0 for slot 0, 4 idle cycles = 0x04 (1 byte committed)
    assert_eq!(tu.byte_buffer.len(), start_len + 1);
    assert_eq!(tu.byte_buffer[start_len], 0x04);
    assert_eq!(tu.pending_slot_mask, 0b0000_0010);

    // commit_cycle(10) flushes the cycle-10 frame.
    tu.commit_cycle(10);
    // Single0 for slot 1, 4 idle cycles = 0b00010100 = 0x14
    assert_eq!(tu.byte_buffer.len(), start_len + 2);
    assert_eq!(tu.byte_buffer[start_len + 1], 0x14);
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
        // alternate 8 and 7 slots: identical consecutive frames would be one frame and a Repeat
        let ids = if cycle % 2 == 0 { 37..=44 } else { 37..=43 };
        for id in ids {
            tu.notify_event(id, cycle, None);
        }
        tu.commit_cycle(cycle);
    }

    let staged_words: usize = tu.staged.iter().map(|(_, w)| w.len()).sum();
    let bytes_emitted = tu.byte_buffer.len() + (tu.pending_words.len() + staged_words) * 4 - start_len;
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

    force_start(&mut tu, 1);
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

    force_start(&mut tu, 1);
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

// ===== Mode 2 (INST_EXEC) is not modeled =====

#[test]
fn mode2_emits_no_frames() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::Execution);
    tu.set_event_slot(0, 2);
    tu.write_register(0x00, 2 | (1 << 16) | (3 << 24));
    tu.notify_event(1, 0, Some(0x1234)); // start
    tu.notify_event(2, 1, Some(0x1236)); // slot event
    tu.commit_cycle(1);
    tu.notify_event(3, 5, Some(0x1240)); // stop
    tu.flush();
    assert_eq!(tu.encoded_bytes_len(), 0, "mode 2 must not fabricate frames");
    assert!(!tu.has_pending_words(), "mode 2 must not emit packets");
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

    // Re-arm and verify start works again. HW pipelines start by 1 cycle:
    // notify_event(28, 200) arms; state transitions to Running on the next
    // notify with cycle > 200.
    tu.notify_event(28, 200, None);
    assert!(matches!(tu.state, super::TraceState::Idle), "armed but not yet Running");
    tu.notify_event(0xFF, 201, None); // unmatched event bumps cycle past arm
    assert!(matches!(tu.state, super::TraceState::Running), "rerun start should fire");
    let _ = after_first; // silence unused-var lint
}

#[test]
fn origin_offset_shifts_only_the_start_absolute() {
    // Build the same stream twice, differing only in origin_offset. The Start
    // frame's 7 absolute bytes must shift by exactly the offset; every other
    // byte (deltas, event frames) must be identical -- the within-domain
    // byte-identity invariant.
    fn stream(offset: u64) -> Vec<u8> {
        let mut tu = TraceUnit::new(0, 2);
        // mode=EventTime, start=28, stop=29 (same idiom as test_register_configuration).
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37); // slot 0 = event 37
        tu.set_origin_offset(offset);
        force_start(&mut tu, 28); // arm at cycle 0 -> Start absolute = 0 + offset
        notify_commit(&mut tu, 37, 5); // one event at cycle 5 (small: whole stream stays in byte_buffer)
        tu.encoded_bytes().to_vec()
    }
    let base = stream(0);
    let shifted = stream(64);
    assert_eq!(base.len(), shifted.len(), "offset must not change stream length");
    assert_eq!(base[0] & 0xF0, 0xF0, "Start marker prefix present");
    for i in 0..base.len() {
        if (1..=7).contains(&i) {
            continue; // the 7 Start-absolute bytes are allowed to differ
        }
        assert_eq!(base[i], shifted[i], "byte {i} changed outside the Start absolute");
    }
    let decode_start = |b: &[u8]| (0..7).fold(0u64, |v, i| (v << 8) | b[1 + i] as u64);
    assert_eq!(decode_start(&base), 0, "zero-offset Start absolute == raw arm cycle (0)");
    assert_eq!(
        decode_start(&shifted),
        decode_start(&base) + 64,
        "Start absolute must shift by exactly the offset"
    );
}

#[test]
fn origin_offset_persists_across_control0_rewrite_and_clears_on_reset() {
    let mut tu = TraceUnit::new(0, 2);
    tu.set_origin_offset(42);
    assert_eq!(tu.origin_offset(), 42);
    // Rewriting Trace_Control0 (per-batch reconfig) clears trace bookkeeping
    // but NOT the broadcast-network origin offset.
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    assert_eq!(tu.origin_offset(), 42, "Control0 rewrite must preserve origin offset");
    // A full reset returns to the power-on default.
    tu.reset();
    assert_eq!(tu.origin_offset(), 0, "reset must clear origin offset");
}

/// Drive `n` isolated one-cycle MEMORY_STALL rising/falling pairs through a
/// fresh `TraceUnit` (mode=EventPc, matching real MEMORY_STALL captures),
/// decode the resulting binary trace through `tools/trace_decoder` (the
/// SAME decoder Task 8's HW-vs-EMU comparison uses for mode 1 -- upstream
/// mlir-aie's `aie.utils.trace.parse_trace` is mode-0 only), and return the
/// number of decoded events named MEMORY_STALL.
fn decode_n_isolated_memory_stalls(n: u64) -> u64 {
    use std::io::Write;

    const MEMORY_STALL_HW_ID: u8 = 23; // core_events::MEMORY_STALL, xaie_events_aieml.h

    // Core trace unit, mode=EventPc(1) (matches how MEMORY_STALL is
    // actually captured on real cores -- see event_trace.rs's PC-carrying
    // stall variants), start=ACTIVE_CORE(28), stop=DISABLED_CORE(29),
    // slot 0 = MEMORY_STALL. packet_type=0 (Core) is also the default, so
    // EventPc mode is legal per `mode_supports_pc`.
    let mut tu = TraceUnit::new(1, 2);
    tu.write_register(0x00, 1 | (28 << 16) | (29 << 24));
    tu.write_register(0x04, 1); // packet_type=0 (Core), packet_id=1
    tu.write_register(0x10, MEMORY_STALL_HW_ID as u32); // slot 0

    force_start(&mut tu, 28); // uses cycles 0 and 1

    // n isolated one-cycle stalls: rising at T, falling at T+1, then an
    // idle gap before the next one -- exactly what `mem_stall_edge`
    // produces for n non-adjacent single-cycle arbitration losses.
    let mut cycle = 2u64;
    for _ in 0..n {
        tu.set_event_level(MEMORY_STALL_HW_ID, cycle, true);
        tu.commit_cycle(cycle);
        cycle += 1;
        tu.set_event_level(MEMORY_STALL_HW_ID, cycle, false);
        tu.commit_cycle(cycle);
        cycle += 2; // idle gap: the next stall is fully isolated
    }
    tu.flush();

    let mut words: Vec<u32> = Vec::new();
    while let Some(packet) = tu.pop_packet() {
        words.extend_from_slice(&packet);
    }
    assert!(!words.is_empty(), "encoder produced no packets for n={n}");

    let dir = tempfile::tempdir().expect("create temp dir");
    let bin_path = dir.path().join("memory_stall.bin");
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in &words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    std::fs::File::create(&bin_path)
        .expect("create trace-bin file")
        .write_all(&bytes)
        .expect("write trace-bin");

    // Decode through tools/trace_decoder -- the SAME decoder Task 8 will
    // use for the real HW-vs-EMU MEMORY_STALL comparison (mode 1 has no
    // upstream mlir-aie oracle; this in-tree decoder is authoritative for
    // it, per tools/parse-trace.py's own docstring).
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = format!(
        r#"
import sys
sys.path.insert(0, "tools")
import numpy as np
from trace_decoder import parse_trace, TraceMode

raw = np.fromfile({bin_path:?}, dtype=np.uint32).tolist()
slot_names = {{0: {{"2,1": ["MEMORY_STALL", "", "", "", "", "", "", ""]}}}}
events = parse_trace(raw, slot_names=slot_names, mode=TraceMode.EVENT_PC)
count = sum(1 for e in events if e.name == "MEMORY_STALL")
print(count)
"#,
        bin_path = bin_path.display().to_string(),
    );
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .current_dir(&manifest)
        .output()
        .expect("run python3 tools/trace_decoder check (is python3 on PATH?)");
    assert!(
        output.status.success(),
        "trace_decoder script failed for n={n}:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("decoder printed non-numeric output for n={n}: {e} (stdout={stdout:?})"))
}

/// Task 7 BLOCKER check (core-memory-stall-model, #140): MEMORY_STALL is
/// emitted as a held LEVEL via `mem_stall_edge` (coordinator.rs), not a
/// pulse -- the claim is that N ISOLATED one-cycle stalls each produce
/// their own rising edge and therefore decode as N discrete events, not
/// one merged span. This must be verified through the REAL encoder
/// (`TraceUnit`) and the REAL decoder (`tools/trace_decoder`), not a
/// hand-rolled re-implementation of the bit format that could share the
/// same misunderstanding as the encoder.
///
/// Sweeps several N (including small and non-packet-aligned values, so a
/// pass isn't a coincidence of one particular byte-buffer/packet boundary)
/// to confirm N always decodes as exactly N.
///
/// Ignored (not part of the default `cargo test --lib` gate) because it
/// shells out to python3 + numpy; run explicitly to confirm the encoding
/// assumption end to end.
#[test]
#[ignore = "shells out to python3 (tools/trace_decoder) for an independent decode of the real encoder output"]
fn n_isolated_one_cycle_memory_stalls_decode_as_n_events() {
    for n in [1u64, 2, 7, 13, 50] {
        let decoded = decode_n_isolated_memory_stalls(n);
        eprintln!(
            "[n_isolated_one_cycle_memory_stalls_decode_as_n_events] n={n} -> tools/trace_decoder \
             decoded {decoded} MEMORY_STALL events"
        );
        assert_eq!(
            decoded, n,
            "n={n} isolated one-cycle MEMORY_STALL pulses must decode as n discrete events, got \
             {decoded} -- if this is LESS than n, isolated stalls are coalescing into fewer spans \
             and the Task 8 count comparison against real hardware is meaningless (BLOCKER)."
        );
    }
}

/// Silicon (build/experiments/lock-stall-pulse, a core held on an acquire since before the
/// trace started): the first frame after the Start already reports the LOCK_STALL level,
/// one cycle after the start event, and the later falling edge closes it as usual. The
/// level rose before the unit ran (here even before it was configured), so the unit has to
/// know what is asserted when it starts.
#[test]
fn a_level_asserted_before_the_start_is_in_the_first_frame() {
    let mut tu = TraceUnit::new(0, 2);
    tu.set_event_level(26, 3, true); // the core stalls before any trace register is written
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24)); // slot3 = LOCK_STALL
    tu.notify_event(28, 100, None); // start event: arms, Start marker
    let start_len = tu.byte_buffer.len();
    tu.notify_event(1, 101, None); // the TRUE metronome promotes the armed start
    tu.commit_cycle(101);
    assert!(tu.byte_buffer.len() > start_len, "the held level must open in the first frame");
    assert_eq!(tu.held_mask, 1 << 3);
    // a level that fell before the start is not reported
    let mut tu = TraceUnit::new(0, 2);
    tu.set_event_level(26, 3, true);
    tu.set_event_level(26, 4, false);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    tu.notify_event(28, 100, None);
    let start_len = tu.byte_buffer.len();
    tu.notify_event(1, 101, None);
    tu.commit_cycle(101);
    assert_eq!(tu.byte_buffer.len(), start_len);
    assert_eq!(tu.held_mask, 0);
}

/// A gap too long for the 18-bit cycles field is Sync tokens, one per wrap of the field,
/// then the frame with the remainder. Silicon (quiet traces, idle 262142..1000000 cycles
/// between PERF_CNT_2 pulses): a wrap that lands in the frame's own cycle is lost.
#[test]
fn idle_longer_than_the_cycles_field_is_sync_tokens() {
    for (idle, syncs, field) in [
        (262_143u64, 0u64, 262_143u64),
        (262_144, 0, 0),
        (262_145, 1, 1),
        (524_287, 1, 262_143),
        (524_288, 1, 0),
        (524_289, 2, 1),
        (600_000, 2, 75_712),
        (1_000_000, 3, 213_568),
    ] {
        assert_eq!(split_idle(idle), (syncs, field), "idle {idle}");
    }

    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37);
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();
    notify_commit(&mut tu, 37, 600_001); // 600_000 idle cycles: Sync + Repeat(1), then 75712
    notify_commit(&mut tu, 37, 1_200_002); // the same again; not merged across the Syncs
    tu.flush_repeats();
    // Single2 slot 0, 75712 = 0x127C0
    let frame = [0xA1, 0x27, 0xC0];
    let mut expected = vec![0xFF, 0xE1];
    expected.extend_from_slice(&frame);
    expected.extend_from_slice(&[0xFF, 0xE1]);
    expected.extend_from_slice(&frame);
    let body: Vec<u8> = tu.byte_buffer[start_len..].iter().copied().filter(|&b| b != 0xFE).collect();
    assert_eq!(body, expected);
}

// -- Saturated Repeat1 tokens are written when they complete -------------------------------
//
// Silicon writes each Repeat1(1023) in the cycle its 1023rd repeat happens, not when the run
// ends: in the quiet stretches of objectfifo_repeat_init_values_repeat every shim trace packet
// is the same 28 bytes (`Rep0(10) Sng0(64,0) Rep0(7) Skip Mul0(192,0)` and 11 x Rep1(1023))
// and reaches the S2MM 11 cycles after the 11th chunk's last repeat. A chunk is written in
// the cycle of the next identical frame; the byte stream is the same as when the run is
// written at its end, only the cycle bytes enter the buffer moves.

/// A started unit with slot 3 (hw event 26, a level) and slots 5..=7 (24, 35, 36) mapped.
/// Returns the unit and the length of the Start marker already in the buffer.
fn held_run_unit() -> (TraceUnit, usize) {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    tu.write_register(0x14, 23 | (24 << 8) | (35 << 16) | (36 << 24));
    tu.notify_event(28, 0, None);
    let start_len = tu.byte_buffer.len();
    (tu, start_len)
}

/// Drive the coordinator's per-cycle commit pass over `cycles` (inclusive).
fn commit_each(tu: &mut TraceUnit, cycles: std::ops::RangeInclusive<u64>) {
    for c in cycles {
        tu.commit_cycle(c);
    }
}

/// Test 1: a level held alone writes its first Repeat1(1023) in the cycle after the 1023rd
/// repeat: the frame of that cycle cannot extend the run.
#[test]
fn a_lone_held_level_writes_its_saturated_repeat_one_cycle_after_the_1023rd_repeat() {
    let (mut tu, start_len) = held_run_unit();
    tu.set_event_level(26, 1, true); // the frame in cycle 1, repeats in 2..=1024
    commit_each(&mut tu, 1..=1024);
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30], "1023 repeats: still counting");

    commit_each(&mut tu, 1025..=1025);
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF]);
    assert_ne!(tu.held_mask, 0, "the level is still held");
}

/// Test 2: three saturations at their own cycles (with the word-straddle Skip fill between
/// them), the remainder at the drop. The literal bytes are what the end-flushed code wrote
/// for this run before the change; `the_same_run_written_at_its_end_is_the_same_bytes`
/// feeds the run with no intermediate commits.
#[test]
fn a_long_hold_writes_each_saturation_at_its_cycle() {
    let (mut tu, start_len) = held_run_unit();
    tu.set_event_level(26, 1, true);
    commit_each(&mut tu, 1..=1024);
    commit_each(&mut tu, 1025..=1025);
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF]);
    commit_each(&mut tu, 1026..=2047);
    assert_eq!(tu.byte_buffer.len(), start_len + 3);
    commit_each(&mut tu, 2048..=2048);
    // the second chunk would straddle the word: Skip fill first
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF, 0xFE, 0xDB, 0xFF]);
    commit_each(&mut tu, 2049..=3070);
    assert_eq!(tu.byte_buffer.len(), start_len + 6);
    commit_each(&mut tu, 3071..=3071);
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF, 0xFE, 0xDB, 0xFF, 0xDB, 0xFF]);
    commit_each(&mut tu, 3072..=3075); // four more held cycles, five repeats with 3071
    assert_eq!(tu.byte_buffer.len(), start_len + 8);

    tu.set_event_level(26, 3076, false);
    commit_each(&mut tu, 3076..=3076);
    tu.flush_repeats();
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF, 0xFE, 0xDB, 0xFF, 0xDB, 0xFF, 0xE5]);
}

/// The same hold with no commit in between (the whole run written when it ends, as before the
/// change) gives the same bytes: pre-change literal, recorded by running this test unchanged
/// against the old code.
#[test]
fn the_same_run_written_at_its_end_is_the_same_bytes() {
    let (mut tu, start_len) = held_run_unit();
    tu.set_event_level(26, 1, true);
    tu.commit_cycle(1);
    tu.set_event_level(26, 3076, false);
    tu.commit_cycle(3076);
    tu.flush_repeats();
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF, 0xFE, 0xDB, 0xFF, 0xDB, 0xFF, 0xE5]);
}

/// Test 3: the silicon packet. A 28-byte payload of `Rep0(10) Sng0 Rep0(7) Mul0(192,0)` and
/// 11 x Rep1(1023): the 11th chunk is written one cycle after its 1023rd repeat and the packet
/// is handed over `EMIT` cycles later, level still held.
/// (The lead is built from events: a 11-cycle hold of slot 5 handing over to slot 6 held for
/// 8 cycles, then slot 7 joining; that is 6 lead bytes as on silicon, but with no Skip fill
/// since the buffer is aligned differently.)
#[test]
fn the_silicon_packet_is_handed_over_emit_cycles_after_the_11th_saturation_is_written() {
    let (mut tu, _) = held_run_unit();
    tu.set_event_level(24, 1, true); // slot 5: cycles 1..=11 (Single + Repeat0(10))
    tu.commit_cycle(1);
    tu.set_event_level(24, 12, false); // hands over to slot 6 in cycle 12
    tu.set_event_level(35, 12, true); // slot 6: 12..=19 (Single0(64,0) + Repeat0(7))
    commit_each(&mut tu, 2..=19);
    tu.set_event_level(36, 20, true); // slot 7 joins: Multiple0(192,0) from cycle 20
    commit_each(&mut tu, 20..=20);
    // the Start marker is the 8 bytes before; drop them to start a fresh 28-byte packet
    tu.byte_buffer.drain(..8);
    assert_eq!(tu.byte_buffer, vec![0x50, 0xEA, 0x60, 0xE7, 0xCC, 0x00]);

    let last = 20 + 11 * 1023; // cycle of the 11th chunk's 1023rd repeat
    let written = last + 1; // the cycle of the next identical frame
    commit_each(&mut tu, 21..=written + EMIT - 1);
    assert!(!tu.has_pending_packets(), "packet not handed over before written + EMIT");
    commit_each(&mut tu, written + EMIT..=written + EMIT);
    assert!(tu.has_pending_packets(), "packet handed over EMIT cycles after the 11th chunk is written");
    assert_eq!(tu.held_mask & 0xC0, 0xC0, "both levels still held");

    let packet = tu.pop_packet().unwrap();
    let payload: Vec<u8> = packet[1..].iter().flat_map(|w| w.to_be_bytes()).collect();
    let mut expected = vec![0x50, 0xEA, 0x60, 0xE7, 0xCC, 0x00];
    for _ in 0..11 {
        expected.extend_from_slice(&[0xDB, 0xFF]);
    }
    assert_eq!(payload, expected);
}

/// Test 4: a foreign pulse in a long hold below the saturation count behaves as before:
/// repeats flush before the pulse frame, the held-only frame follows it.
#[test]
fn a_pulse_in_a_hold_below_saturation_is_unchanged() {
    let (mut tu, start_len) = held_run_unit();
    tu.set_event_level(26, 1, true);
    commit_each(&mut tu, 1..=499);
    tu.notify_event(37, 500, None);
    commit_each(&mut tu, 500..=599);
    tu.set_event_level(26, 600, false);
    commit_each(&mut tu, 600..=600);
    tu.flush_repeats();
    // 1: frame; 2..=499: 498 repeats; 500: Multiple0(slot0|slot3); 501..=599: frame + 98
    let body: Vec<u8> = tu.byte_buffer[start_len..].iter().copied().filter(|&b| b != 0xFE).collect();
    assert_eq!(body, [0x30, 0xD9, 0xF2, 0xC0, 0x90, 0x30, 0xD8, 98]);
}

/// Test 5a: a Sync right after a run that saturated exactly: the Repeat1 is written with the
/// Sync (the next frame's turn, INFERRED), then the Sync (idle >= 2^18 before the next frame), then the frame.
#[test]
fn saturation_then_a_sync_is_the_same_bytes() {
    let (mut tu, start_len) = held_run_unit();
    tu.set_event_level(26, 1, true);
    commit_each(&mut tu, 1..=1024);
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30], "a run of exactly 1023 repeats waits for a frame");
    tu.set_event_level(26, 1025, false);
    tu.commit_cycle(1025);
    let again = 1025 + (1 << 18) + 5; // idle = 2^18 + 5: one Sync, field 5
    tu.set_event_level(26, again, true);
    tu.commit_cycle(again);
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF, 0xFF, 0x35]);
}

/// Test 5b: saturation landing where the chunk would straddle a word writes the Skip fill in
/// that cycle, and the whole stream equals the run written at its end.
#[test]
fn saturation_with_a_word_straddle_fill_is_the_same_bytes() {
    let (mut eager, start_len) = held_run_unit();
    eager.set_event_level(26, 1, true);
    commit_each(&mut eager, 1..=2048);
    assert_eq!(eager.byte_buffer.len(), start_len + 6, "second chunk and its fill are in");
    commit_each(&mut eager, 2049..=2100);
    eager.set_event_level(26, 2101, false);
    commit_each(&mut eager, 2101..=2101);
    eager.flush_repeats();

    let (mut late, _) = held_run_unit();
    late.set_event_level(26, 1, true);
    late.commit_cycle(1);
    late.set_event_level(26, 2101, false);
    late.commit_cycle(2101);
    late.flush_repeats();
    assert_eq!(eager.byte_buffer, late.byte_buffer);
}

/// Test 6: a run that ends exactly at 1023 repeats is one Repeat1(1023) and no empty Repeat.
#[test]
fn a_run_ending_at_exactly_1023_repeats_has_no_trailing_repeat() {
    let (mut tu, start_len) = held_run_unit();
    tu.set_event_level(26, 1, true);
    commit_each(&mut tu, 1..=1024);
    tu.set_event_level(26, 1025, false);
    commit_each(&mut tu, 1025..=1025);
    tu.flush_repeats();
    assert_eq!(&tu.byte_buffer[start_len..], &[0x30, 0xDB, 0xFF]);
}

// -- Hand-off timing of completed packets ---------------------------------------------------
//
// Silicon (shim trace packets, three captures of six tests): a packet's first word reaches the
// shim S2MM 11 cycles after the last cycle its tokens cover when its last token is a single or
// multiple frame, a saturated Repeat1(1023), or a Repeat followed at once by a frame; g + 10
// when a Repeat is followed by g - 1 idle cycles. That is a Repeat written in the cycle a frame
// that cannot extend the run is sampled (R1), a frame's own token one cycle after its cycle
// (R2), and a packet handed to the stream `STREAM_TRACE_PACKET_EMIT_CYCLES` after its last byte
// is written (R3), on top of the 2 cycles the stream path takes. Hand-off cycles below are the
// cycles at whose `commit_cycle` the packet first shows in `has_pending_packets`.

const EMIT: u64 = xdna_archspec::aie2::timing::STREAM_TRACE_PACKET_EMIT_CYCLES as u64;

/// The first cycle in `cycles` at whose commit the unit has a packet to hand over, with the
/// byte stream untouched by the choice.
fn first_handoff(tu: &mut TraceUnit, cycles: std::ops::RangeInclusive<u64>) -> Option<u64> {
    cycles.into_iter().find(|&c| {
        tu.commit_cycle(c);
        tu.has_pending_packets()
    })
}

/// Test: a packet completed by a single frame sampled in cycle `t_f` is handed over at
/// `t_f + 1 + EMIT`, not at `t_f + EMIT` (R2, R3).
#[test]
fn a_packet_completed_by_a_single_frame_is_handed_over_at_t_f_plus_1_plus_emit() {
    let (mut tu, _) = held_run_unit();
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(27, 0xFE);
    tu.notify_event(37, 5, None); // slot 0, one pulse: Single0
    assert_eq!(first_handoff(&mut tu, 5..=200), Some(5 + 1 + EMIT));
    assert_eq!(tu.byte_buffer.len(), 0, "the 28 bytes went into the packet");
}

/// Test: the same for a frame of several events (Multiple0).
#[test]
fn a_packet_completed_by_a_multiple_frame_is_handed_over_at_t_f_plus_1_plus_emit() {
    let (mut tu, _) = held_run_unit();
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(26, 0xFE);
    tu.notify_event(37, 5, None);
    tu.notify_event(38, 5, None); // slots 0 and 1: Multiple0 (2 bytes)
    assert_eq!(first_handoff(&mut tu, 5..=200), Some(5 + 1 + EMIT));
}

/// Test: a saturated Repeat1(1023) is written in the cycle of the next identical frame, one
/// after the 1023rd repeat, and the packet it completes is handed over `EMIT` later (R1, R3).
#[test]
fn a_packet_completed_by_a_saturated_repeat_is_handed_over_one_cycle_after_the_1023rd_repeat_plus_emit() {
    let (mut tu, _) = held_run_unit();
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(25, 0xFE);
    tu.set_event_level(26, 1, true); // frame in cycle 1, repeats in 2..=1024
    let t_s = 1024;
    assert_eq!(first_handoff(&mut tu, 1..=t_s + 100), Some(t_s + 1 + EMIT));
    assert_ne!(tu.held_mask, 0, "the level is still held");
}

/// Test: a Repeat0 ending a run is written when the next frame arrives. With that frame in the
/// very next cycle (gap 1) and 8 idle cycles later (gap 9) the hand-off is `EMIT` after the
/// next frame's cycle either way, so the delay from the last covered cycle is gap + 10.
#[test]
fn a_run_ending_repeat_is_handed_over_emit_cycles_after_the_next_frame() {
    for gap in [1u64, 9] {
        let (mut tu, _) = held_run_unit();
        tu.byte_buffer.clear();
        tu.byte_buffer.resize(26, 0xFE);
        tu.set_event_level(26, 1, true); // frame in 1, Repeat0(10) for 2..=11
        let last_covered = 11;
        commit_each(&mut tu, 1..=last_covered);
        tu.set_event_level(26, last_covered + 1, false);
        let t_n = last_covered + gap;
        tu.notify_event(37, t_n, None); // the different frame
        assert_eq!(
            first_handoff(&mut tu, last_covered + 1..=t_n + 100),
            Some(t_n + EMIT),
            "gap {gap}: the Repeat is written in the cycle of the next frame"
        );
        assert_eq!(tu.byte_buffer.len(), 1, "the frame's own token is the next packet's");
    }
}

/// Test: exactly 1023 repeats and then nothing (INFERRED corner: no silicon sample) writes no
/// token until the next frame arrives, like any run-ending Repeat.
#[test]
fn a_run_of_exactly_1023_repeats_then_idle_waits_for_the_next_frame() {
    let (mut tu, start_len) = held_run_unit();
    tu.byte_buffer.truncate(start_len);
    tu.byte_buffer.resize(25, 0xFE);
    tu.set_event_level(26, 1, true);
    commit_each(&mut tu, 1..=1024);
    tu.set_event_level(26, 1025, false);
    commit_each(&mut tu, 1025..=1999);
    assert!(!tu.has_pending_packets());
    assert_eq!(tu.byte_buffer.len(), 26, "frame written, the Repeat1(1023) still pending");
    tu.notify_event(37, 2000, None);
    assert_eq!(first_handoff(&mut tu, 2000..=2100), Some(2000 + EMIT));
}

/// Test: two packets completing within `EMIT` cycles of each other leave in order, each no
/// earlier than its own completion + `EMIT`, the second only after the first.
#[test]
fn two_packets_completing_close_together_leave_in_order() {
    let (mut tu, _) = held_run_unit();
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(27, 0xFE);
    tu.notify_event(37, 5, None);
    commit_each(&mut tu, 5..=7);
    tu.byte_buffer.resize(27, 0xFE); // a second packet completes with the frame of cycle 8
    tu.notify_event(38, 8, None);
    let mut words_at = Vec::new();
    for c in 8..=40 {
        tu.commit_cycle(c);
        words_at.push((c, tu.pending_words.len()));
    }
    let first = words_at.iter().find(|(_, n)| *n == 8).map(|(c, _)| *c);
    let second = words_at.iter().find(|(_, n)| *n == 16).map(|(c, _)| *c);
    assert_eq!(first, Some(5 + 1 + EMIT));
    assert_eq!(second, Some(8 + 1 + EMIT));
    let a = tu.pop_packet().unwrap();
    let b = tu.pop_packet().unwrap();
    assert_eq!(a[7] & 0xFF, 0x04, "first packet ends in slot 0's Single0 (4 idle cycles)");
    assert_eq!(b[7] & 0xFF, 0x12, "second packet ends in slot 1's Single0 (2 idle cycles)");
}

/// Test: a packet is handed over at exactly its ready cycle whether that cycle is quiet or
/// carries a frame of its own.
#[test]
fn hand_off_does_not_depend_on_what_else_happens_in_the_ready_cycle() {
    for dirty in [false, true] {
        let (mut tu, _) = held_run_unit();
        tu.byte_buffer.clear();
        tu.byte_buffer.resize(27, 0xFE);
        tu.notify_event(37, 5, None);
        let ready = 5 + 1 + EMIT;
        if dirty {
            commit_each(&mut tu, 5..=ready - 1);
            tu.notify_event(38, ready, None);
        }
        assert_eq!(first_handoff(&mut tu, 5..=ready + 20), Some(ready), "dirty = {dirty}");
    }
}

// -- Packets finished by Skip padding ---------------------------------------------------------
//
// A multi-byte token that would straddle a payload word is preceded by Skip bytes; when those
// bytes are the last of a packet, the packet is finished by the padding and the token starts the
// next one. Silicon (shim trace packets, three captures, every traced bridge test): such a packet
// reaches the shim S2MM 10 cycles after the cycle the token that did not fit covers up to, for a
// Multiple0 frame (16 + 52, 12 + 37, 12 + 37 packets), a saturated Repeat1(1023) (36, 45, 29) and
// a Repeat1 ending a run with the next frame in the following cycle (49, 57, 60), and 18 for one
// with 8 idle cycles before the next frame (12, 13, 14): one cycle sooner than a packet whose last
// byte is that token's own (11, and gap + 10). The padding is written one cycle before the token.

/// Test: a Multiple0 frame of cycle `t_f` that does not fit: the padded packet is handed over at
/// `t_f + EMIT`, one cycle before a packet the frame itself would have completed.
#[test]
fn a_packet_finished_by_padding_before_a_frame_is_handed_over_at_t_f_plus_emit() {
    let (mut tu, _) = held_run_unit();
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(27, 0xFE); // one byte left in the last word: a 2-byte frame straddles it
    tu.notify_event(37, 5, None);
    tu.notify_event(38, 5, None); // slots 0 and 1: Multiple0 (2 bytes)
    assert_eq!(first_handoff(&mut tu, 5..=200), Some(5 + EMIT));
    assert_eq!(tu.byte_buffer.len(), 2, "the frame starts the next packet");
}

/// Test: a saturated Repeat1(1023) that does not fit: written one cycle after the 1023rd repeat,
/// the padded packet before it handed over at the 1023rd repeat + `EMIT`.
#[test]
fn a_packet_finished_by_padding_before_a_saturated_repeat_leaves_at_the_1023rd_repeat_plus_emit() {
    let (mut tu, _) = held_run_unit();
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(26, 0xFE); // the level's Single0 makes 27: the Repeat1 straddles
    tu.set_event_level(26, 1, true); // frame in cycle 1, repeats in 2..=1024
    let t_s = 1024;
    assert_eq!(first_handoff(&mut tu, 1..=t_s + 100), Some(t_s + EMIT));
    assert_eq!(tu.byte_buffer, vec![0xDB, 0xFF], "the Repeat1(1023) starts the next packet");
}

/// Test: a Repeat1 ending a run, written when the next different frame is sampled in `t_n`, that
/// does not fit: the padded packet is handed over at `t_n - 1 + EMIT`, for the next frame in the
/// cycle right after the run and 8 idle cycles later.
#[test]
fn a_packet_finished_by_padding_before_a_run_ending_repeat_leaves_a_cycle_before_the_next_frame() {
    for gap in [1u64, 9] {
        let (mut tu, _) = held_run_unit();
        tu.byte_buffer.clear();
        tu.byte_buffer.resize(26, 0xFE);
        tu.set_event_level(26, 1, true); // frame in 1 (27 bytes), Repeat1(20) for 2..=21
        let last_covered = 21;
        commit_each(&mut tu, 1..=last_covered);
        tu.set_event_level(26, last_covered + 1, false);
        let t_n = last_covered + gap;
        commit_each(&mut tu, last_covered + 1..=t_n - 1);
        tu.notify_event(37, t_n, None);
        assert_eq!(
            first_handoff(&mut tu, t_n..=t_n + 100),
            Some(t_n - 1 + EMIT),
            "gap {gap}: the padding goes in one cycle before the Repeat"
        );
    }
}

// -- When a Sync is written ---------------------------------------------------------------------
//
// Silicon (no-data trace probe: a core packet filled with 27 one-byte frames, last frame in cycle
// L, then 2^18 + 3000 idle cycles before the next frame, or none until the trace stops): the
// packet whose 28th byte is the Sync reaches the shim S2MM at L + 2^18 + 14 in both runs, about
// 3000 cycles before the next frame (a run with 3000 idle cycles and no Sync puts the padded
// packet at the closing frame + 12). With the 10-cycle hand-off and the 2-cycle wake of the
// route that is the Sync written in cycle L + 2^18 + 2: the first cycle a frame would need it
// (`split_idle`), whether or not a frame comes then. Each further wrap is one more Sync, run-
// length coded into the Repeat written with the next token, so the bytes are unchanged.

const WRAP: u64 = CYCLES_FIELD_RANGE;

/// A unit with one slot (hw event 37), its first frame in cycle 5, the buffer at 27 bytes.
fn sync_unit() -> TraceUnit {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37);
    tu.notify_event(28, 0, None);
    tu.byte_buffer.clear();
    tu.byte_buffer.resize(26, 0xFE);
    notify_commit(&mut tu, 37, 5);
    assert_eq!(tu.byte_buffer.len(), 27);
    tu
}

#[test]
fn a_sync_is_written_in_the_cycle_the_idle_count_wraps() {
    let mut tu = sync_unit();
    let due = 5 + WRAP + 2;
    commit_each(&mut tu, 6..=due - 1);
    assert_eq!(tu.byte_buffer.len(), 27, "no Sync before the wrap");
    assert_eq!(first_handoff(&mut tu, due..=due + 100), Some(due + EMIT));
    let p = tu.pop_packet().unwrap();
    assert_eq!(p[7] & 0xFF, 0xFF, "the Sync is the packet's last byte");
}

#[test]
fn a_sync_is_written_at_the_wrap_with_no_frame_after_it() {
    // The same stretch closed by the trace stop instead of a frame: the Sync still goes in at
    // the wrap (silicon: the packet arrives at L + 2^18 + 14, long before the stop).
    let mut tu = sync_unit();
    let due = 5 + WRAP + 2;
    assert_eq!(first_handoff(&mut tu, 6..=due + 4 * WRAP), Some(due + EMIT));
}

#[test]
fn syncs_written_as_time_passes_are_the_bytes_of_syncs_written_with_the_frame() {
    for idle in [WRAP + 1, 2 * WRAP + 1, 3 * WRAP + 17, 1_000_000] {
        let t = 5 + 1 + idle;
        let mut per_cycle = sync_unit();
        commit_each(&mut per_cycle, 6..=t - 1);
        notify_commit(&mut per_cycle, 37, t);
        let mut at_once = sync_unit();
        notify_commit(&mut at_once, 37, t);
        let bytes = |tu: &mut TraceUnit| {
            tu.flush_repeats();
            let mut b: Vec<u8> = Vec::new();
            while let Some(p) = tu.pop_packet() {
                b.extend(p[1..].iter().flat_map(|w| w.to_be_bytes()));
            }
            b.extend(
                tu.staged
                    .iter()
                    .flat_map(|(_, w)| w[1..].iter().flat_map(|(x, _)| x.to_be_bytes())),
            );
            b.extend(tu.byte_buffer.iter());
            b
        };
        assert_eq!(bytes(&mut per_cycle), bytes(&mut at_once), "idle {idle}");
    }
}

// -- End-of-stream token ---------------------------------------------------------------------
//
// Silicon ends each tile's stream with a 4-byte token, 0xDC and an 18-bit count of the idle
// cycles from the last frame (Repeats included) to the stop, on a 4-byte boundary, the rest of
// the packet 0xFE. A quiet tile's stream is `Start DC`; a long idle tail writes Syncs first;
// when the last data fills its packet the token would need a packet of its own and is left out.
// The tails below are copied from three silicon captures (impl notes: held level to the stop
// ends `Rep1 DC(0)`, a periodic counter tail `Sng2(1,1024) Rep0(5) DC(422)`).

/// Payload bytes of every packet queued so far, one `Vec` per packet (header dropped).
fn payloads(tu: &mut TraceUnit) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(words) = tu.pop_packet() {
        out.push(words[1..].iter().flat_map(|w| w.to_be_bytes()).collect());
    }
    out
}

/// A running unit with slots 0..=3 on hw events 37..=39 and 26 (a level), started at cycle 0.
fn running_unit() -> TraceUnit {
    let mut tu = TraceUnit::new(0, 2);
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    tu.write_register(0x10, 37 | (38 << 8) | (39 << 16) | (26 << 24));
    force_start(&mut tu, 28);
    tu
}

#[test]
fn a_level_held_to_the_stop_ends_with_its_repeat_and_a_zero_count() {
    let mut tu = running_unit();
    tu.set_event_level(26, 1, true);
    commit_each(&mut tu, 1..=9);
    tu.notify_event(29, 10, None); // stop: the level was asserted through cycle 9
    let p = payloads(&mut tu);
    assert_eq!(p.len(), 1);
    // Start (8), Single0 slot 3 cycles 0, Repeat0(8), two Skip to the word boundary, DC(0)
    assert_eq!(&p[0][8..16], &[0x30, 0xE8, 0xFE, 0xFE, 0xDC, 0x00, 0x00, 0x00]);
    assert!(p[0][16..].iter().all(|&b| b == 0xFE));
}

#[test]
fn an_idle_tail_after_a_periodic_run_counts_the_idle_cycles() {
    // Silicon core tail: `a0 04 00 | e5 | dc 00 01 a6`: Single2(slot 0, 1024), Repeat0(5), count 422.
    let mut tu = running_unit();
    for k in 1..=6u64 {
        notify_commit(&mut tu, 37, 1025 * k);
    }
    tu.notify_event(29, 6150 + 423, None);
    let p = payloads(&mut tu);
    assert_eq!(p.len(), 1);
    assert_eq!(&p[0][8..16], &[0xA0, 0x04, 0x00, 0xE5, 0xDC, 0x00, 0x01, 0xA6]);
    assert!(p[0][16..].iter().all(|&b| b == 0xFE));
}

#[test]
fn a_quiet_tile_is_a_start_and_the_token() {
    let mut tu = running_unit();
    tu.notify_event(29, 100, None);
    let p = payloads(&mut tu);
    assert_eq!(p.len(), 1);
    assert_eq!(p[0][0], 0xF0);
    assert_eq!(&p[0][8..12], &[0xDC, 0x00, 0x00, 99]);
    assert!(p[0][12..].iter().all(|&b| b == 0xFE));
}

#[test]
fn a_long_quiet_tail_writes_syncs_before_the_count() {
    // Silicon `Start Sync Rep0(1) DC`: two wraps of the field, the rest in the count.
    let mut tu = running_unit();
    let idle = 2 * (1u64 << 18) + 5;
    tu.notify_event(29, idle + 1, None);
    let p = payloads(&mut tu);
    assert_eq!(&p[0][8..16], &[0xFF, 0xE1, 0xFE, 0xFE, 0xDC, 0x00, 0x00, 0x05]);
}

#[test]
fn the_count_field_is_eighteen_bits() {
    let mut tu = running_unit();
    tu.notify_event(29, 0x2_ABCD + 1, None);
    let p = payloads(&mut tu);
    assert_eq!(&p[0][8..12], &[0xDC, 0x02, 0xAB, 0xCD]);
}

/// Alternating slots in consecutive cycles: one byte per frame, no Repeat merging.
fn fill_with_frames(tu: &mut TraceUnit, frames: u64) {
    for c in 1..=frames {
        notify_commit(tu, if c % 2 == 1 { 37 } else { 38 }, c);
    }
}

#[test]
fn the_token_may_end_exactly_at_the_end_of_the_packet() {
    let mut tu = running_unit();
    fill_with_frames(&mut tu, 16); // 8 + 16 = 24 bytes
    tu.notify_event(29, 20, None); // count 3
    let p = payloads(&mut tu);
    assert_eq!(p.len(), 1);
    assert_eq!(&p[0][24..], &[0xDC, 0x00, 0x00, 0x03]);
}

#[test]
fn data_that_fills_its_packet_gets_no_token_packet() {
    // Measured as an absence in 75 of 226 streams that need a packet of their own.
    for frames in [20u64, 18, 19] {
        // 28 bytes exactly (packet already emitted), 26 and 27 used (token does not fit)
        let mut tu = running_unit();
        fill_with_frames(&mut tu, frames);
        tu.notify_event(29, 100, None);
        let p = payloads(&mut tu);
        assert_eq!(p.len(), 1, "{frames} frames");
        assert!(!p[0].contains(&0xDC), "{frames} frames: no token");
    }
}

#[test]
fn a_pending_repeat_is_written_before_the_token() {
    let mut tu = running_unit();
    for c in 5..=7 {
        notify_commit(&mut tu, 37, c);
    }
    tu.notify_event(29, 20, None);
    let p = payloads(&mut tu);
    // Single0 slot 0 cycles 4, Single0 slot 0 cycles 0, Repeat0(1), count 20 - 1 - 7 = 12
    assert_eq!(&p[0][8..16], &[0x04, 0x00, 0xE1, 0xFE, 0xDC, 0x00, 0x00, 0x0C]);
}

/// The end of a stream with an idle tail longer than the cycles field: the Sync of the wrap
/// goes in at the wrap and completes its packet, which leaves long before the stop. Silicon
/// (no-data trace probe, the core stopped 27 bytes into a packet): that packet reaches the
/// shim S2MM at L + 2^18 + 14 and is the stream's last; no end-token packet follows, as for any
/// stream whose data fills its last packet exactly.
#[test]
fn an_idle_tail_writes_its_sync_at_the_wrap_long_before_the_stop() {
    let mut tu = sync_unit(); // last frame in cycle 5, 27 bytes
    let due = 5 + WRAP + 2;
    let stop = 5 + WRAP + 3000;
    assert_eq!(
        first_handoff(&mut tu, 6..=stop - 1),
        Some(due + EMIT),
        "the Sync packet leaves before the stop"
    );
    let sync_packet = tu.pop_packet().unwrap();
    assert_eq!(sync_packet[7] & 0xFF, 0xFF);
    tu.notify_event(29, stop, None);
    assert!(payloads(&mut tu).is_empty(), "no end-token packet after a full one");
}

/// A Sync whose wrap has not come by the stop is written at the stop, before the end token,
/// which counts the rest of the idle tail.
#[test]
fn a_sync_not_yet_due_at_the_stop_goes_in_before_the_end_token() {
    let mut tu = sync_unit();
    tu.byte_buffer.truncate(21); // room for the Sync and the 4-byte token
    let stop = 5 + WRAP + 2; // idle to the stop 2^18 + 1: one Sync, count 1
    commit_each(&mut tu, 6..=stop - 1);
    assert_eq!(tu.byte_buffer.len(), 21, "not due before the stop");
    tu.notify_event(29, stop, None);
    let p = payloads(&mut tu);
    assert_eq!(p.len(), 1);
    assert_eq!(&p[0][21..], &[0xFF, 0xFE, 0xFE, 0xDC, 0x00, 0x00, 0x01]);
}
