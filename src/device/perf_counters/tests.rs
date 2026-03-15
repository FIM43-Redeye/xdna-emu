use super::*;

// -- Construction and defaults --

#[test]
fn new_core_module_has_4_counters() {
    let bank = PerfCounterBank::new(4);
    assert_eq!(bank.num_counters(), 4);
}

#[test]
fn new_memory_module_has_2_counters() {
    let bank = PerfCounterBank::new(2);
    assert_eq!(bank.num_counters(), 2);
}

#[test]
fn new_counters_default_to_zero_and_idle() {
    let bank = PerfCounterBank::new(4);
    for i in 0..4 {
        assert_eq!(bank.read_counter(i), 0);
        assert_eq!(bank.read_event_value(i), 0);
        assert_eq!(bank.start_event(i), 0);
        assert_eq!(bank.stop_event(i), 0);
        assert_eq!(bank.reset_event(i), 0);
        assert!(!bank.is_active(i));
    }
}

#[test]
#[should_panic(expected = "num_counters (5) exceeds MAX_PERF_COUNTERS (4)")]
fn new_panics_on_too_many_counters() {
    PerfCounterBank::new(5);
}

#[test]
fn new_zero_counters_is_valid() {
    let bank = PerfCounterBank::new(0);
    assert_eq!(bank.num_counters(), 0);
}

// -- Direct counter value read/write --

#[test]
fn write_counter_then_read_back() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_counter(0, 0xDEAD_BEEF);
    bank.write_counter(3, 0xCAFE_BABE);
    assert_eq!(bank.read_counter(0), 0xDEAD_BEEF);
    assert_eq!(bank.read_counter(3), 0xCAFE_BABE);
    // Unwritten counters remain 0
    assert_eq!(bank.read_counter(1), 0);
    assert_eq!(bank.read_counter(2), 0);
}

#[test]
fn write_counter_out_of_range_is_noop() {
    let mut bank = PerfCounterBank::new(2);
    bank.write_counter(2, 42);
    assert_eq!(bank.read_counter(2), 0);
}

#[test]
fn read_counter_out_of_range_returns_zero() {
    let bank = PerfCounterBank::new(2);
    assert_eq!(bank.read_counter(5), 0);
}

// -- Event value (threshold) read/write --

#[test]
fn write_event_value_then_read_back() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_event_value(0, 1000);
    bank.write_event_value(3, 5000);
    assert_eq!(bank.read_event_value(0), 1000);
    assert_eq!(bank.read_event_value(3), 5000);
}

#[test]
fn write_event_value_out_of_range_is_noop() {
    let mut bank = PerfCounterBank::new(2);
    bank.write_event_value(3, 42);
    assert_eq!(bank.read_event_value(3), 0);
}

// -- Control register write/read round-trip --

#[test]
fn control_start_stop_round_trip_7bit() {
    let mut bank = PerfCounterBank::new(4);
    // Pack: start0=42, stop0=43, start1=44, stop1=45
    let value = 42 | (43 << 8) | (44 << 16) | (45 << 24);
    bank.write_control_start_stop(value, 0, 1, 7);

    assert_eq!(bank.start_event(0), 42);
    assert_eq!(bank.stop_event(0), 43);
    assert_eq!(bank.start_event(1), 44);
    assert_eq!(bank.stop_event(1), 45);

    // Read back should reconstruct the same value
    let readback = bank.read_control_start_stop(0, 1, 7);
    assert_eq!(readback, value);
}

#[test]
fn control_start_stop_round_trip_8bit_memtile() {
    let mut bank = PerfCounterBank::new(4);
    // MemTile uses 8-bit event fields, can have values > 127
    let value = 200u32 | (201 << 8) | (202 << 16) | (203 << 24);
    bank.write_control_start_stop(value, 0, 1, 8);

    assert_eq!(bank.start_event(0), 200);
    assert_eq!(bank.stop_event(0), 201);
    assert_eq!(bank.start_event(1), 202);
    assert_eq!(bank.stop_event(1), 203);
}

#[test]
fn control_start_stop_7bit_masks_high_bits() {
    let mut bank = PerfCounterBank::new(4);
    // Value 0xFF should be masked to 0x7F for 7-bit events
    let value = 0xFF | (0xFF << 8);
    bank.write_control_start_stop(value, 0, 1, 7);
    assert_eq!(bank.start_event(0), 0x7F);
    assert_eq!(bank.stop_event(0), 0x7F);
}

#[test]
fn control_reset_round_trip() {
    let mut bank = PerfCounterBank::new(4);
    let value = 5 | (6 << 8) | (7 << 16) | (8 << 24);
    bank.write_control_reset(value, 7);

    assert_eq!(bank.reset_event(0), 5);
    assert_eq!(bank.reset_event(1), 6);
    assert_eq!(bank.reset_event(2), 7);
    assert_eq!(bank.reset_event(3), 8);

    let readback = bank.read_control_reset();
    assert_eq!(readback, value);
}

#[test]
fn control_reset_2_counter_module() {
    let mut bank = PerfCounterBank::new(2);
    let value = 10 | (20 << 8) | (30 << 16) | (40 << 24);
    bank.write_control_reset(value, 7);

    // Only first 2 counters should be set
    assert_eq!(bank.reset_event(0), 10);
    assert_eq!(bank.reset_event(1), 20);
    // Out of range
    assert_eq!(bank.reset_event(2), 0);
    assert_eq!(bank.reset_event(3), 0);
}

// -- Start/Stop/Reset Event Handling --

#[test]
fn counter_starts_on_start_event() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(42 | (0 << 8), 0, 1, 7); // start=42, stop=0

    assert!(!bank.is_active(0));
    bank.handle_event(42);
    assert!(bank.is_active(0));
}

#[test]
fn counter_stops_on_stop_event() {
    let mut bank = PerfCounterBank::new(4);
    // start=42, stop=43
    bank.write_control_start_stop(42 | (43 << 8), 0, 1, 7);

    bank.handle_event(42); // start
    assert!(bank.is_active(0));

    bank.handle_event(43); // stop
    assert!(!bank.is_active(0));
}

#[test]
fn counter_increments_each_tick_while_active() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(42 | (43 << 8), 0, 1, 7);

    // Not active yet -- tick should not increment
    bank.tick();
    assert_eq!(bank.read_counter(0), 0);

    // Start counting
    bank.handle_event(42);
    bank.tick();
    assert_eq!(bank.read_counter(0), 1);
    bank.tick();
    assert_eq!(bank.read_counter(0), 2);
    bank.tick();
    assert_eq!(bank.read_counter(0), 3);

    // Stop counting
    bank.handle_event(43);
    bank.tick();
    assert_eq!(bank.read_counter(0), 3); // Value preserved
}

#[test]
fn counter_does_not_increment_when_idle() {
    let mut bank = PerfCounterBank::new(2);
    // No start event configured
    for _ in 0..100 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 0);
    assert_eq!(bank.read_counter(1), 0);
}

#[test]
fn reset_event_zeroes_counter_and_goes_idle() {
    let mut bank = PerfCounterBank::new(4);
    // start=10, stop=0
    bank.write_control_start_stop(10, 0, 1, 7);
    // reset=20
    bank.write_control_reset(20, 7);

    bank.handle_event(10); // start
    for _ in 0..50 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 50);
    assert!(bank.is_active(0));

    bank.handle_event(20); // reset
    assert_eq!(bank.read_counter(0), 0);
    assert!(!bank.is_active(0));
}

#[test]
fn reset_event_takes_priority_over_start() {
    let mut bank = PerfCounterBank::new(4);
    // Both start and reset are the same event
    bank.write_control_start_stop(10, 0, 1, 7); // start=10
    bank.write_control_reset(10, 7); // reset=10

    bank.write_counter(0, 100);
    bank.handle_event(10);
    // Reset should take priority: counter zeroed, state is Idle
    assert_eq!(bank.read_counter(0), 0);
    assert!(!bank.is_active(0));
}

#[test]
fn multiple_counters_independent() {
    let mut bank = PerfCounterBank::new(4);
    // Counter 0: start=10, stop=11
    bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);
    // Counter 2: start=20, stop=21
    bank.write_control_start_stop(20 | (21 << 8), 2, 3, 7);

    // Start counter 0 only
    bank.handle_event(10);
    for _ in 0..5 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 5);
    assert_eq!(bank.read_counter(2), 0);

    // Now start counter 2
    bank.handle_event(20);
    for _ in 0..3 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 8); // 5 + 3
    assert_eq!(bank.read_counter(2), 3);

    // Stop counter 0
    bank.handle_event(11);
    for _ in 0..2 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 8); // Stopped
    assert_eq!(bank.read_counter(2), 5); // Still counting
}

#[test]
fn stopped_counter_can_restart() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);

    bank.handle_event(10); // start
    for _ in 0..5 {
        bank.tick();
    }
    bank.handle_event(11); // stop
    assert_eq!(bank.read_counter(0), 5);

    bank.handle_event(10); // restart from current value
    for _ in 0..3 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 8); // 5 + 3, continues from where stopped
}

#[test]
fn event_0_never_matches() {
    let mut bank = PerfCounterBank::new(4);
    // start_event is 0 (default)
    bank.handle_event(0);
    assert!(!bank.is_active(0)); // Should not start
}

#[test]
fn direct_write_does_not_affect_state() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);

    // Directly write a counter value while idle
    bank.write_counter(0, 1000);
    assert_eq!(bank.read_counter(0), 1000);
    assert!(!bank.is_active(0)); // Still idle

    // Start and verify it counts from the written value
    bank.handle_event(10);
    bank.tick();
    assert_eq!(bank.read_counter(0), 1001);
}

// -- Threshold Events --

#[test]
fn tick_fires_threshold_event_when_counter_reaches_event_value() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10, 0, 1, 7);
    bank.write_event_value(0, 5);

    bank.handle_event(10); // start

    // Tick 4 times -- no threshold yet
    for _ in 0..4 {
        let events = bank.tick();
        assert!(events.is_empty());
    }
    assert_eq!(bank.read_counter(0), 4);

    // Tick once more -- counter hits 5 = event_value
    let events = bank.tick();
    assert_eq!(events, vec![0]);
    assert_eq!(bank.read_counter(0), 5);
}

#[test]
fn threshold_event_fires_only_once_at_exact_value() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10, 0, 1, 7);
    bank.write_event_value(0, 3);

    bank.handle_event(10);

    // Tick to threshold
    bank.tick(); // 1
    bank.tick(); // 2
    let events = bank.tick(); // 3 -- fires
    assert_eq!(events, vec![0]);

    // Subsequent ticks should not re-fire
    let events = bank.tick(); // 4
    assert!(events.is_empty());
}

#[test]
fn event_value_zero_means_no_threshold() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10, 0, 1, 7);
    // event_value defaults to 0

    bank.handle_event(10);
    for _ in 0..100 {
        let events = bank.tick();
        assert!(events.is_empty());
    }
}

#[test]
fn multiple_counters_can_fire_threshold_same_tick() {
    let mut bank = PerfCounterBank::new(4);
    // Both counter 0 and counter 1 start on event 10
    bank.write_control_start_stop(10 | (10 << 16), 0, 1, 7);
    bank.write_event_value(0, 3);
    bank.write_event_value(1, 3);

    bank.handle_event(10);
    bank.tick(); // 1
    bank.tick(); // 2
    let events = bank.tick(); // 3
    assert_eq!(events, vec![0, 1]);
}

#[test]
fn counter_wraps_at_u32_max() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10, 0, 1, 7);
    bank.write_counter(0, u32::MAX);

    bank.handle_event(10);
    bank.tick();
    assert_eq!(bank.read_counter(0), 0); // Wrapped
}

// -- Register Interface --

#[test]
fn register_read_write_round_trip_control0() {
    let mut bank = PerfCounterBank::new(4);
    let value = 42 | (43 << 8) | (44 << 16) | (45 << 24);
    assert!(bank.write_register(0x00, value, 7));
    assert_eq!(bank.read_register(0x00, 7), Some(value));
}

#[test]
fn register_read_write_round_trip_control1_4counter() {
    let mut bank = PerfCounterBank::new(4);
    let value = 10 | (11 << 8) | (12 << 16) | (13 << 24);
    assert!(bank.write_register(0x04, value, 7));
    assert_eq!(bank.read_register(0x04, 7), Some(value));
}

#[test]
fn register_read_write_round_trip_reset() {
    let mut bank = PerfCounterBank::new(4);
    let value = 5 | (6 << 8) | (7 << 16) | (8 << 24);
    assert!(bank.write_register(0x08, value, 7));
    assert_eq!(bank.read_register(0x08, 7), Some(value));
}

#[test]
fn register_read_write_counter_values() {
    let mut bank = PerfCounterBank::new(4);
    assert!(bank.write_register(0x20, 0xAAAA, 7));
    assert!(bank.write_register(0x24, 0xBBBB, 7));
    assert!(bank.write_register(0x28, 0xCCCC, 7));
    assert!(bank.write_register(0x2C, 0xDDDD, 7));

    assert_eq!(bank.read_register(0x20, 7), Some(0xAAAA));
    assert_eq!(bank.read_register(0x24, 7), Some(0xBBBB));
    assert_eq!(bank.read_register(0x28, 7), Some(0xCCCC));
    assert_eq!(bank.read_register(0x2C, 7), Some(0xDDDD));
}

#[test]
fn register_read_write_event_values() {
    let mut bank = PerfCounterBank::new(4);
    assert!(bank.write_register(0x80, 1000, 7));
    assert!(bank.write_register(0x84, 2000, 7));
    assert!(bank.write_register(0x88, 3000, 7));
    assert!(bank.write_register(0x8C, 4000, 7));

    assert_eq!(bank.read_register(0x80, 7), Some(1000));
    assert_eq!(bank.read_register(0x84, 7), Some(2000));
    assert_eq!(bank.read_register(0x88, 7), Some(3000));
    assert_eq!(bank.read_register(0x8C, 7), Some(4000));
}

#[test]
fn register_unknown_offset_returns_none() {
    let bank = PerfCounterBank::new(4);
    assert_eq!(bank.read_register(0x10, 7), None);
    assert_eq!(bank.read_register(0x30, 7), None);
    assert_eq!(bank.read_register(0x90, 7), None);
}

#[test]
fn register_unknown_offset_write_returns_false() {
    let mut bank = PerfCounterBank::new(4);
    assert!(!bank.write_register(0x10, 42, 7));
    assert!(!bank.write_register(0xFF, 42, 7));
}

// -- Integration-style tests --

#[test]
fn full_lifecycle_start_count_threshold_stop_reset() {
    let mut bank = PerfCounterBank::new(4);

    // Configure counter 0: start=10, stop=11, reset=12, threshold=5
    bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);
    bank.write_control_reset(12, 7);
    bank.write_event_value(0, 5);

    // Idle: should not count
    bank.tick();
    assert_eq!(bank.read_counter(0), 0);

    // Start
    bank.handle_event(10);
    assert!(bank.is_active(0));

    // Count to threshold
    for _ in 0..4 {
        let ev = bank.tick();
        assert!(ev.is_empty());
    }
    assert_eq!(bank.read_counter(0), 4);

    let ev = bank.tick(); // counter hits 5
    assert_eq!(ev, vec![0]);
    assert_eq!(bank.read_counter(0), 5);

    // Counter keeps going past threshold
    bank.tick();
    assert_eq!(bank.read_counter(0), 6);

    // Stop
    bank.handle_event(11);
    assert!(!bank.is_active(0));
    bank.tick();
    assert_eq!(bank.read_counter(0), 6); // Preserved

    // Reset
    bank.handle_event(12);
    assert_eq!(bank.read_counter(0), 0);
    assert!(!bank.is_active(0));
}

#[test]
fn two_counter_module_ignores_counters_2_and_3() {
    let mut bank = PerfCounterBank::new(2);

    // Configure via register: counter 2/3 should be ignored
    let value = 10 | (11 << 8) | (20 << 16) | (21 << 24);
    bank.write_control_start_stop(value, 0, 1, 7);

    assert_eq!(bank.start_event(0), 10);
    assert_eq!(bank.stop_event(0), 11);
    assert_eq!(bank.start_event(1), 20);
    assert_eq!(bank.stop_event(1), 21);

    // Write to counter index 2 should be silently ignored
    bank.write_counter(2, 42);
    assert_eq!(bank.read_counter(2), 0);

    // But counter 0 and 1 should work fine
    bank.handle_event(10);
    bank.tick();
    assert_eq!(bank.read_counter(0), 1);
}

#[test]
fn stop_event_preserves_value_across_restart() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);

    // First run: count to 10
    bank.handle_event(10);
    for _ in 0..10 {
        bank.tick();
    }
    bank.handle_event(11);
    assert_eq!(bank.read_counter(0), 10);

    // Second run: count continues from 10
    bank.handle_event(10);
    for _ in 0..5 {
        bank.tick();
    }
    assert_eq!(bank.read_counter(0), 15);
}

#[test]
fn reset_while_stopped_zeroes_value() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10 | (11 << 8), 0, 1, 7);
    bank.write_control_reset(12, 7);

    bank.handle_event(10); // start
    for _ in 0..5 {
        bank.tick();
    }
    bank.handle_event(11); // stop
    assert_eq!(bank.read_counter(0), 5);

    bank.handle_event(12); // reset while stopped
    assert_eq!(bank.read_counter(0), 0);
    assert!(!bank.is_active(0));
}

#[test]
fn threshold_wraps_around() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10, 0, 1, 7);
    bank.write_counter(0, u32::MAX - 1);
    bank.write_event_value(0, 1); // threshold at value 1

    bank.handle_event(10);
    // u32::MAX - 1 -> u32::MAX (no threshold, event_value=1 != u32::MAX)
    let ev = bank.tick();
    assert!(ev.is_empty());
    // u32::MAX -> 0 (wraps, 0 != 1)
    let ev = bank.tick();
    assert!(ev.is_empty());
    // 0 -> 1 (matches threshold!)
    let ev = bank.tick();
    assert_eq!(ev, vec![0]);
}

#[test]
fn handle_event_with_no_matching_counter_is_noop() {
    let mut bank = PerfCounterBank::new(4);
    bank.write_control_start_stop(10, 0, 1, 7);

    // Event 99 does not match any configured event
    bank.handle_event(99);
    assert!(!bank.is_active(0));
    assert_eq!(bank.read_counter(0), 0);
}

#[test]
fn same_event_for_multiple_counters() {
    let mut bank = PerfCounterBank::new(4);
    // Counter 0 and counter 1 both start on event 10
    bank.write_control_start_stop(10 | (10 << 16), 0, 1, 7);

    bank.handle_event(10);
    assert!(bank.is_active(0));
    assert!(bank.is_active(1));

    bank.tick();
    assert_eq!(bank.read_counter(0), 1);
    assert_eq!(bank.read_counter(1), 1);
}
