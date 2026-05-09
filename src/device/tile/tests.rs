//! Tests for the tile module.

use super::*;

#[test]
fn test_tile_creation() {
    let tile = Tile::compute(1, 2);
    assert_eq!(tile.col, 1);
    assert_eq!(tile.row, 2);
    assert!(tile.is_compute());
    assert!(tile.program_memory().is_some());
    assert_eq!(tile.data_memory().len(), 64 * 1024);
    assert_eq!(tile.locks.len(), 16);
    assert_eq!(tile.dma_bds.len(), 16);
    assert_eq!(tile.dma_channels.len(), 4);
}

#[test]
fn test_mem_tile_creation() {
    let tile = Tile::mem_tile(0, 1);
    assert!(tile.is_mem());
    assert!(tile.program_memory().is_none());
    assert_eq!(tile.data_memory().len(), 512 * 1024);
    assert_eq!(tile.locks.len(), 64);
    assert_eq!(tile.dma_bds.len(), 48);
    assert_eq!(tile.dma_channels.len(), 12);
}

#[test]
fn test_shim_tile_creation() {
    let tile = Tile::shim(0, 0);
    assert!(tile.is_shim());
    assert!(tile.program_memory().is_none());
    assert_eq!(tile.data_memory().len(), 0);
    assert_eq!(tile.locks.len(), 16);
    assert_eq!(tile.dma_bds.len(), 16);
    assert_eq!(tile.dma_channels.len(), 4);
}

#[test]
fn test_data_memory_write() {
    let mut tile = Tile::compute(0, 2);
    let data = [0xDE, 0xAD, 0xBE, 0xEF];
    assert!(tile.write_data(0x100, &data));
    assert_eq!(&tile.data_memory()[0x100..0x104], &data);
}

#[test]
fn test_program_memory_write() {
    let mut tile = Tile::compute(0, 2);
    let code = [0x15, 0x01, 0x00, 0x40]; // Sample AIE instruction
    assert!(tile.write_program(0, &code));
    assert_eq!(&tile.program_memory().unwrap()[0..4], &code);
}

#[test]
fn test_data_u32_operations() {
    let mut tile = Tile::compute(0, 2);
    assert!(tile.write_data_u32(0x200, 0xCAFEBABE));
    assert_eq!(tile.read_data_u32(0x200), Some(0xCAFEBABE));
}

#[test]
fn test_lock_operations() {
    let mut lock = Lock::new(2);
    assert!(lock.acquire()); // 2 -> 1
    assert!(lock.acquire()); // 1 -> 0
    assert!(!lock.acquire()); // 0 -> can't acquire
    lock.release(); // 0 -> 1
    assert!(lock.acquire()); // 1 -> 0
}

#[test]
fn test_lock_max_value() {
    // Test clamping at creation (positive overflow)
    let lock = Lock::new(100);
    assert_eq!(lock.value, Lock::MAX_VALUE); // Clamped to 63

    // Test clamping at creation (negative overflow)
    let lock = Lock::new(-100);
    assert_eq!(lock.value, Lock::MIN_VALUE); // Clamped to -64

    // Test saturation on release
    let mut lock = Lock::new(63);
    lock.release();
    assert_eq!(lock.value, 63); // Saturated at max

    // Test set
    let mut lock = Lock::new(0);
    lock.set(50);
    assert_eq!(lock.value, 50);
    lock.set(Lock::MAX_VALUE + 1); // would be 64, but i8 can't hold it; test boundary
                                   // i8 max is 127, so test with explicit value
    lock.set(100); // > 63
    assert_eq!(lock.value, 63); // Clamped
    lock.set(-100); // < -64
    assert_eq!(lock.value, -64); // Clamped
}

#[test]
fn test_dma_bd_valid() {
    let mut bd = DmaBufferDescriptor::default();
    assert!(!bd.is_valid());
    bd.control = 1;
    assert!(bd.is_valid());
}

#[test]
fn test_core_state_reset() {
    let mut core = CoreState {
        pc: 0x1000,
        sp: 0x8000,
        lr: 0x500,
        status: 0xFF,
        control: 0x3,
        enabled: true,
        running: true,
        _pad: [0; 2],
    };
    core.reset();
    assert_eq!(core.pc, 0);
    assert_eq!(core.sp, 0x7_0000);
    assert!(!core.enabled);
}

#[test]
fn test_struct_sizes() {
    // Ensure structs are reasonably sized
    assert_eq!(std::mem::size_of::<Lock>(), 3); // u8 + 2 bools
    assert_eq!(std::mem::size_of::<DmaBufferDescriptor>(), 24);
    // DmaChannel: control u32 + start_queue u32 + 4x u8 + bool (3 padding) + status u32 = 20 bytes
    assert_eq!(std::mem::size_of::<DmaChannel>(), 20);
    assert_eq!(std::mem::size_of::<CoreState>(), 24);
}

#[test]
fn test_program_memory_size() {
    // Verify program memory is 16KB per AM020
    assert_eq!(PROGRAM_MEMORY_SIZE, 16 * 1024);
}

#[test]
fn test_lock_counts() {
    // Verify lock counts per AM020 (via TileParams defaults)
    assert_eq!(TileParams::compute().num_locks, 16);
    assert_eq!(TileParams::mem_tile().num_locks, 64);
}

#[test]
fn test_lock_acquire_with_value() {
    let mut lock = Lock::new(5);

    // Acquire with value >= 3, decrement by 2
    assert_eq!(lock.acquire_with_value(3, -2), LockResult::Success);
    assert_eq!(lock.value, 3);

    // Acquire with value >= 2, decrement by 1
    assert_eq!(lock.acquire_with_value(2, -1), LockResult::Success);
    assert_eq!(lock.value, 2);

    // Try to acquire with value >= 5 - should fail (only have 2)
    assert_eq!(lock.acquire_with_value(5, -3), LockResult::PreconditionNotMet);
    assert_eq!(lock.value, 2); // Value unchanged

    // Acquire all remaining
    assert_eq!(lock.acquire_with_value(2, -2), LockResult::Success);
    assert_eq!(lock.value, 0);

    // Can't acquire when value is 0
    assert_eq!(lock.acquire_with_value(1, -1), LockResult::PreconditionNotMet);
    assert_eq!(lock.value, 0);
}

#[test]
fn test_lock_release_with_value() {
    let mut lock = Lock::new(0);

    // Release by 3
    assert_eq!(lock.release_with_value(3), LockResult::Success);
    assert_eq!(lock.value, 3);

    // Release by 10
    assert_eq!(lock.release_with_value(10), LockResult::Success);
    assert_eq!(lock.value, 13);

    // Release to max (60 + 13 = 73, saturates to 63)
    assert_eq!(lock.release_with_value(60), LockResult::WouldOverflow);
    assert_eq!(lock.value, 63);
    assert!(lock.overflow);
}

#[test]
fn test_lock_release_negative_delta() {
    // Release with negative delta (unusual but supported)
    let mut lock = Lock::new(10);

    // "Release" with -3 is like an acquire
    assert_eq!(lock.release_with_value(-3), LockResult::Success);
    assert_eq!(lock.value, 7);

    // Large negative delta: goes into negative range (7 - 10 = -3, valid)
    assert_eq!(lock.release_with_value(-10), LockResult::Success);
    assert_eq!(lock.value, -3);

    // Push to underflow past MIN_VALUE (-3 - 62 = -65, beyond -64)
    assert_eq!(lock.release_with_value(-62), LockResult::PreconditionNotMet);
    assert_eq!(lock.value, Lock::MIN_VALUE); // Clamped to -64
    assert!(lock.underflow);
}

#[test]
fn test_lock_flags_clear() {
    let mut lock = Lock::new(63);

    // Cause overflow
    lock.release_with_value(10);
    assert!(lock.overflow);
    assert!(!lock.underflow);
    assert!(lock.has_error());

    // Clear flags
    lock.clear_flags();
    assert!(!lock.overflow);
    assert!(!lock.has_error());
}

#[test]
fn test_lock_acquire_equal() {
    // Test acquire_eq semantics (wait for exact match)
    let mut lock = Lock::new(2);

    // acquire_equal: wait for value == 1, should fail (value is 2)
    assert_eq!(lock.acquire_equal(1, -1), LockResult::PreconditionNotMet);
    assert_eq!(lock.value, 2); // Unchanged

    // acquire_equal: wait for value == 2, should succeed
    assert_eq!(lock.acquire_equal(2, -2), LockResult::Success);
    assert_eq!(lock.value, 0); // Decremented to 0

    // Reset and test acquire_ge vs acquire_eq difference
    lock.set(3);

    // acquire_ge (acquire_with_value): wait for value >= 2, succeeds with 3
    assert_eq!(lock.acquire_with_value(2, -1), LockResult::Success);
    assert_eq!(lock.value, 2); // 3 - 1 = 2

    // acquire_eq: wait for value == 2, succeeds
    assert_eq!(lock.acquire_equal(2, -2), LockResult::Success);
    assert_eq!(lock.value, 0);

    // Reset to test exact-match requirement
    lock.set(5);

    // acquire_eq for value == 3 should fail (we have 5)
    assert_eq!(lock.acquire_equal(3, -3), LockResult::PreconditionNotMet);
    assert_eq!(lock.value, 5); // Unchanged

    // acquire_ge for value >= 3 should succeed (we have 5)
    assert_eq!(lock.acquire_with_value(3, -2), LockResult::Success);
    assert_eq!(lock.value, 3); // 5 - 2 = 3
}

// === Edge Detection Tests ===

#[test]
fn test_edge_detector_default() {
    let det = EdgeDetector::default();
    assert_eq!(det.input_event, 0);
    assert!(!det.trigger_rising);
    assert!(!det.trigger_falling);
    assert!(!det.prev_active);
    assert!(!det.curr_active);
}

#[test]
fn test_configure_edge_detectors_compute() {
    let mut dets = [EdgeDetector::default(); 2];
    // Event 0: event=37 (0x25), rising=1, falling=0
    // Event 1: event=29 (0x1D), rising=0, falling=1
    // value = (1<<26) | (29<<16) | (1<<9) | 37
    let value = (1 << 26) | (29 << 16) | (1 << 9) | 37;
    Tile::configure_edge_detectors(&mut dets, value, false);

    assert_eq!(dets[0].input_event, 37);
    assert!(dets[0].trigger_rising);
    assert!(!dets[0].trigger_falling);

    assert_eq!(dets[1].input_event, 29);
    assert!(!dets[1].trigger_rising);
    assert!(dets[1].trigger_falling);
}

#[test]
fn test_configure_edge_detectors_memtile_8bit() {
    let mut dets = [EdgeDetector::default(); 2];
    // MemTile uses 8-bit event fields: [7:0] and [23:16]
    // Event 0: event=200 (> 127, needs 8 bits), rising+falling
    // Event 1: event=150, rising only
    let value = (1 << 25) | (150 << 16) | (1 << 10) | (1 << 9) | 200;
    Tile::configure_edge_detectors(&mut dets, value, true);

    assert_eq!(dets[0].input_event, 200);
    assert!(dets[0].trigger_rising);
    assert!(dets[0].trigger_falling);

    assert_eq!(dets[1].input_event, 150);
    assert!(dets[1].trigger_rising);
    assert!(!dets[1].trigger_falling);
}

#[test]
fn test_edge_detector_rising_edge() {
    let mut tile = Tile::compute(0, 2);
    // Configure core edge detector 0: monitor event 37, rising edge
    tile.core_edge_detectors[0].input_event = 37;
    tile.core_edge_detectors[0].trigger_rising = true;

    // Configure core trace to accept edge events (need start event)
    tile.core_trace.write_register(0x00, 0x01); // mode=EventTime
    tile.core_trace.write_register(0x10, 37); // event slot 0 = event 37
                                              // Also configure slot for edge detection event (ID 13)
    tile.core_trace.write_register(0x10, 37 | (13 << 8)); // slot 0=37, slot 1=13

    // Cycle 1: event 37 fires (0->1 = rising edge)
    tile.notify_core_trace_event(37, 100, None);
    tile.evaluate_edge_detectors(100);
    // The edge detector should have detected rising edge and fired event 13

    // Cycle 2: event 37 does not fire (1->0 = falling, not configured)
    tile.evaluate_edge_detectors(200);
    // No event should fire (falling not configured)

    // Cycle 3: event 37 fires again (0->1 = rising edge again)
    tile.notify_core_trace_event(37, 300, None);
    tile.evaluate_edge_detectors(300);
    // Rising edge detected again
}

#[test]
fn test_edge_detector_falling_edge() {
    let mut tile = Tile::compute(0, 2);
    // Configure mem edge detector 1: monitor event 77, falling edge
    tile.mem_edge_detectors[1].input_event = 77;
    tile.mem_edge_detectors[1].trigger_falling = true;

    // Cycle 1: event fires (0->1), no trigger (falling only)
    tile.notify_mem_trace_event(77, 100, None);
    tile.evaluate_edge_detectors(100);

    // Cycle 2: event does NOT fire (1->0 = falling edge)
    tile.evaluate_edge_detectors(200);
    // Falling edge should fire EDGE_DETECTION_EVENT_1 (mem ID 12)
}

#[test]
fn test_edge_detector_register_write() {
    let mut tile = Tile::compute(0, 2);
    // Core module edge detection register (0x34408)
    // Event 0: event=42, rising=1; Event 1: event=50, falling=1
    let value = (1u32 << 26) | (50 << 16) | (1 << 9) | 42;
    Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);

    assert_eq!(tile.core_edge_detectors[0].input_event, 42);
    assert!(tile.core_edge_detectors[0].trigger_rising);
    assert!(!tile.core_edge_detectors[0].trigger_falling);

    assert_eq!(tile.core_edge_detectors[1].input_event, 50);
    assert!(!tile.core_edge_detectors[1].trigger_rising);
    assert!(tile.core_edge_detectors[1].trigger_falling);
}

#[test]
fn test_edge_detector_mem_module_register() {
    let mut tile = Tile::compute(0, 2);
    // Memory module edge detection register (0x14408)
    let value = (1u32 << 25) | (30 << 16) | (1 << 10) | (1 << 9) | 20;
    Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, false);

    assert_eq!(tile.mem_edge_detectors[0].input_event, 20);
    assert!(tile.mem_edge_detectors[0].trigger_rising);
    assert!(tile.mem_edge_detectors[0].trigger_falling);

    assert_eq!(tile.mem_edge_detectors[1].input_event, 30);
    assert!(tile.mem_edge_detectors[1].trigger_rising);
    assert!(!tile.mem_edge_detectors[1].trigger_falling);
}

#[test]
fn test_edge_detector_memtile_register() {
    let mut tile = Tile::mem_tile(0, 1);
    // MemTile edge detection register (0x94408)
    // Use event > 127 to verify 8-bit field (is_memtile=true)
    let value = (1u32 << 25) | (200 << 16) | (1 << 9) | 180;
    Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, true);

    assert_eq!(tile.mem_edge_detectors[0].input_event, 180);
    assert!(tile.mem_edge_detectors[0].trigger_rising);

    assert_eq!(tile.mem_edge_detectors[1].input_event, 200);
    assert!(tile.mem_edge_detectors[1].trigger_rising);
}

#[test]
fn test_edge_detector_no_trigger_when_unconfigured() {
    let mut tile = Tile::compute(0, 2);
    // Default: no edge detection configured (input_event=0, no triggers)
    // Notify event 37
    tile.notify_core_trace_event(37, 100, None);
    tile.evaluate_edge_detectors(100);
    // No edge events should fire (detectors not configured)
    // Just verify it doesn't panic
}

// === Shim Tile Tracing Tests ===

#[test]
fn test_shim_trace_register_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    // Write Trace_Control0 at 0x340D0 (same offset as core module)
    // start_event=1 (TRUE), stop_event=0 (NONE), mode=0 (event-time)
    let ctrl0 = (0u32 << 24) | (1 << 16) | 0;
    device.write_tile_register(0, 0, 0x340D0, ctrl0);
    // Trace unit should now be configured
    let tile = device.array.get(0, 0).unwrap();
    assert!(tile.core_trace.is_configured());
}

#[test]
fn test_shim_edge_detection_register() {
    let mut tile = Tile::shim(0, 0);
    // Edge detection register at 0x34408 for PL module
    // Shim uses core_edge_detectors for its PL module
    let value = (1u32 << 25) | (14 << 16) | (1 << 9) | 22;
    Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);

    assert_eq!(tile.core_edge_detectors[0].input_event, 22);
    assert!(tile.core_edge_detectors[0].trigger_rising);

    assert_eq!(tile.core_edge_detectors[1].input_event, 14);
    assert!(tile.core_edge_detectors[1].trigger_rising);
}

#[test]
fn test_shim_dma_event_notification() {
    let mut device = super::super::state::DeviceState::new_npu1();
    // Configure trace unit with start=TRUE(1)
    device.write_tile_register(0, 0, 0x340D0, (1 << 16) | 0); // start=1, mode=0

    // Shim DMA events go through core_trace (PL module)
    // DMA_S2MM_0_START_TASK = PL event 14
    let tile = device.array.get_mut(0, 0).unwrap();
    tile.notify_core_trace_event(14, 100, None);
    // Should not panic, trace unit accepts it
}

/// End-to-end shim trace lowering scenario, matching mlir-aie's #372 stage 1
/// output for `add_one_using_dma --shim-sweep-events all`:
///
/// - Trace_Control1 = 0x2001 (packet_type = 2 / ShimTile, packet_id = 1)
/// - Trace_Control0 = 0x7E7F0000 (start = 127 = PL_USER_EVENT_1,
///                                stop  = 126 = PL_USER_EVENT_0,
///                                mode  = 0 / EventTime)
/// - Trace_Event0 = 0x16100F0E (slots 0..3 = 14, 15, 16, 22)
/// - Trace_Event1 = 0x1F1E1817 (slots 4..7 = 23, 24, 30, 31)
///
/// The runtime sequence then writes Event_Generate(127) to fire the start
/// event locally and propagate BROADCAST_15 column-wise. This test exercises
/// only the local fire path, which is what the shim trace unit listens for.
///
/// Verifies:
///   1. The trace unit accepts the lowered configuration.
///   2. Event_Generate(127) on the shim takes the trace unit out of Idle.
///   3. After the cycle advances, a DMA_S2MM_0_START_TASK (id 14) is
///      recorded into the pending slot mask, producing emitted bytes.
#[test]
fn test_shim_trace_unit_records_dma_start_task_for_lowered_config() {
    let mut device = super::super::state::DeviceState::new_npu1();
    device.array.set_dma_cycle(0);

    // Lower-bit-faithful Control1 / Control0 / Event0 / Event1 writes.
    device.write_tile_register(0, 0, 0x340D4, 0x2001);
    device.write_tile_register(0, 0, 0x340D0, 0x7E7F_0000);
    device.write_tile_register(0, 0, 0x340E0, 0x1610_0F0E);
    device.write_tile_register(0, 0, 0x340E4, 0x1F1E_1817);

    {
        let tile = device.array.get(0, 0).unwrap();
        let tu = &tile.core_trace;
        assert!(tu.is_configured());
        assert_eq!(tu.start_event, 127);
        assert_eq!(tu.stop_event, 126);
        assert_eq!(tu.packet_type, 2);
        assert_eq!(tu.packet_id, 1);
        assert_eq!(tu.event_slots, [14, 15, 16, 22, 23, 24, 30, 31]);
    }

    // Runtime sequence: Event_Generate on shim with event_id = 127.
    // 0x34008 is the PL module Event_Generate register; the state-effects
    // path is supposed to fire the event on local trace units AND queue a
    // broadcast if a channel is configured to listen for that event.
    device.array.set_dma_cycle(10);
    device.write_tile_register(0, 0, 0x34008, 127);

    // Advance one cycle and fire DMA_S2MM_0_START_TASK (PL event id 14).
    // HW's Idle -> Running transition is pipelined by 1 cycle, so the
    // event must arrive at cycle > armed_cycle to be recorded.
    let cycle_dma = 11u64;
    device.array.set_dma_cycle(cycle_dma);
    let tile = device.array.get_mut(0, 0).unwrap();
    tile.notify_core_trace_event(14, cycle_dma, None);
    tile.core_trace.commit_cycle(cycle_dma);

    assert!(
        tile.core_trace.is_running(),
        "shim trace unit should be Running after the start_event's cycle elapses"
    );
    tile.core_trace.flush();
    assert!(
        tile.core_trace.has_pending_packets(),
        "shim trace unit should have a pending packet recording the DMA event"
    );
}

// === Cascade Stream Tests ===

#[test]
fn test_cascade_init_state() {
    let tile = Tile::compute(1, 2);
    assert!(tile.cascade_input.is_empty());
    assert!(tile.cascade_output.is_empty());
    assert_eq!(tile.cascade_input_dir, 0);
    assert_eq!(tile.cascade_output_dir, 0);
}

#[test]
fn test_cascade_register_write() {
    let mut device = super::super::state::DeviceState::new_npu1();

    // Input=North(0), Output=South(0)
    device.write_tile_register(1, 2, 0x36060, 0b00);
    let tile = device.array.get(1, 2).unwrap();
    assert_eq!(tile.cascade_input_dir, 0);
    assert_eq!(tile.cascade_output_dir, 0);

    // Input=West(1), Output=East(1)
    device.write_tile_register(1, 2, 0x36060, 0b11);
    let tile = device.array.get(1, 2).unwrap();
    assert_eq!(tile.cascade_input_dir, 1);
    assert_eq!(tile.cascade_output_dir, 1);

    // Input=West(1), Output=South(0)
    device.write_tile_register(1, 2, 0x36060, 0b01);
    let tile = device.array.get(1, 2).unwrap();
    assert_eq!(tile.cascade_input_dir, 1);
    assert_eq!(tile.cascade_output_dir, 0);

    // Input=North(0), Output=East(1)
    device.write_tile_register(1, 2, 0x36060, 0b10);
    let tile = device.array.get(1, 2).unwrap();
    assert_eq!(tile.cascade_input_dir, 0);
    assert_eq!(tile.cascade_output_dir, 1);
}

#[test]
fn test_cascade_register_ignored_for_non_compute() {
    let mut device = super::super::state::DeviceState::new_npu1();
    device.write_tile_register(1, 1, 0x36060, 0b11);
    // MemTile should not have cascade direction changed
    let tile = device.array.get(1, 1).unwrap();
    assert_eq!(tile.cascade_input_dir, 0);
    assert_eq!(tile.cascade_output_dir, 0);
}

#[test]
fn test_cascade_fifo_push_pop() {
    let mut tile = Tile::compute(1, 2);
    let data: [u64; 6] = [1, 2, 3, 4, 5, 6];

    assert!(!tile.has_cascade_input());
    tile.push_cascade_input(data);
    assert!(tile.has_cascade_input());

    let result = tile.pop_cascade_input().unwrap();
    assert_eq!(result, data);
    assert!(!tile.has_cascade_input());
    assert!(tile.pop_cascade_input().is_none());
}

#[test]
fn test_cascade_output_fifo() {
    let mut tile = Tile::compute(1, 2);
    let data: [u64; 6] = [10, 20, 30, 40, 50, 60];

    assert!(!tile.has_cascade_output());
    tile.push_cascade_output(data);
    assert!(tile.has_cascade_output());

    let result = tile.pop_cascade_output().unwrap();
    assert_eq!(result, data);
    assert!(!tile.has_cascade_output());
}

/// Proves that DeviceState::write_tile_register() correctly dispatches
/// MemTile BD writes through the full module dispatch path. This is the
/// unified path used by all register write sources (CDO, NPU executor,
/// control packets).
#[test]
fn test_write_tile_register_updates_memtile_bd() {
    let reg_layout = super::super::regdb::device_reg_layout();
    let bd0_word2_offset = reg_layout.memtile_bd_base + 2 * 4; // BD0, word 2 (length)

    let mut device = super::super::state::DeviceState::new_npu1();

    // Verify BD starts zeroed on the MemTile (col=1, row=1)
    let tile = device.array.get(1, 1).expect("tile(1,1) should exist");
    assert!(tile.is_mem(), "tile(1,1) should be a MemTile");
    assert_eq!(tile.dma_bds[0].length, 0, "BD0 length should start at 0");

    // Write via write_tile_register -- the unified register bus
    let test_length: u32 = 0x0000_1000;
    device.write_tile_register(1, 1, bd0_word2_offset, test_length);

    // Both the register HashMap AND the structured BD should be updated
    let tile = device.array.get(1, 1).unwrap();
    assert_eq!(
        *tile.registers_ref().get(&bd0_word2_offset).unwrap_or(&0),
        test_length,
        "Register HashMap should have the value"
    );
    assert_eq!(
        tile.dma_bds[0].length, test_length,
        "MemTile BD0 length should be updated via write_tile_register dispatch"
    );
}

/// OP_READ (operation=1) produces a ReadRegisters action immediately upon
/// receiving the header, with no data payload. The action carries the
/// offset, count (beats+1), and response_id from the header.
#[test]
fn test_ctrl_packet_op_read_via_reassembler() {
    use xdna_archspec::aie2::ctrl_packet::*;
    use super::super::control_packets::{StreamReassembler, ReassembleResult, CtrlOpCode};

    let mut reassembler = StreamReassembler::new(2, 3);

    // Build OP_READ header: addr=0x440, beats=4 (raw=3), op=READ(1), resp_id=2
    let header = 0x440u32
        | (3u32 << LENGTH_SHIFT)
        | ((OP_READ as u32) << OPERATION_SHIFT)
        | (2u32 << RESPONSE_ID_SHIFT);

    // OP_READ completes immediately (no data payload)
    match reassembler.feed_word(header, true) {
        ReassembleResult::Complete(pkt) => {
            assert_eq!(pkt.opcode, CtrlOpCode::Read);
            assert_eq!(pkt.address, 0x440);
            assert_eq!(pkt.beats, 4);
            assert_eq!(pkt.response_id, 2);
            assert!(pkt.data.is_empty());
        }
        other => panic!("Expected Complete, got {:?}", other),
    }
}

// === Lock Trace Event Pipeline Tests ===

#[test]
fn test_lock_event_reaches_trace_unit() {
    // Verify end-to-end: lock acquire -> mem_trace_pending -> trace unit capture.
    //
    // This tests the full pipeline that sweep batch 3 exercises:
    // 1. Lock acquire resolves -> pushes EventType::LockAcquire{lock_id:0}
    // 2. mem_event_to_hw_id maps to 45 (LOCK_SEL0_ACQ_GE)
    // 3. Trace unit with slot configured to 45 captures the event
    let mut tile = Tile::compute(0, 2);

    // Initialize lock 0 to value 1 so an acquire(>=1) will succeed.
    tile.locks[0] = Lock { value: 1, ..Default::default() };

    // Configure mem trace unit:
    //   Control0: mode=EventTime(0), start=1 (TRUE), stop=0
    //   Event0: slot0=1(TRUE), slot1=45(LOCK_SEL0_ACQ_GE), slot2=46(LOCK_0_REL)
    tile.mem_trace.write_register(0x00, 0 | (1 << 16) | (0 << 24)); // start=TRUE(1)
    tile.mem_trace.write_register(0x04, (1 << 12) | 1); // pkt_type=1, pkt_id=1
    tile.mem_trace.write_register(0x10, 1 | (45 << 8) | (46 << 16)); // slots 0-2

    // Start the trace unit by firing TRUE event
    tile.mem_trace.notify_event(1, 0, None); // TRUE at cycle 0
    assert!(tile.mem_trace.is_configured());

    // Submit and resolve a lock acquire on lock 0
    tile.submit_lock_request(LockRequest {
        requestor: LockRequestor::DmaS2mm(0),
        lock_id: 0,
        is_acquire: true,
        expected: 1,
        delta: -1,
        equal_mode: false,
    });
    let results = tile.resolve_lock_requests(100);

    // Lock should be granted (value was 1, needed >=1)
    assert_eq!(results.len(), 1, "Expected one lock result");
    assert!(results[0].2, "Lock acquire should be granted");
    assert!(results[0].3, "Should be marked as acquire");

    // mem_trace_pending should have the lock event
    assert_eq!(tile.mem_trace_pending.len(), 1, "Expected one pending trace event");
    let (cycle, ref event) = tile.mem_trace_pending[0];
    assert_eq!(cycle, 100);
    assert!(
        matches!(event, crate::interpreter::state::EventType::LockAcquire { lock_id: 0 }),
        "Expected LockAcquire{{lock_id:0}}, got {:?}",
        event
    );

    // Map through mem_event_to_hw_id -- should return 45
    let hw_id = crate::trace::mem_event_to_hw_id(event);
    assert_eq!(hw_id, Some(45), "LOCK_SEL0_ACQ_GE should be event ID 45");

    // Notify the trace unit and flush to check capture.
    // Flush drains any pending per-cycle mask before padding the final packet.
    tile.mem_trace.notify_event(45, 100, None);
    tile.mem_trace.flush();
    assert!(
        tile.mem_trace.has_pending_packets(),
        "Trace unit should have recorded the lock event (packet pending after flush)"
    );
}

#[test]
fn test_lock_release_event_reaches_trace_unit() {
    let mut tile = Tile::compute(0, 2);

    // Configure mem trace with LOCK_0_REL (46) in slot 1
    tile.mem_trace.write_register(0x00, 0 | (1 << 16)); // start=TRUE(1)
    tile.mem_trace.write_register(0x04, (1 << 12) | 1);
    tile.mem_trace.write_register(0x10, 1 | (46 << 8)); // slot0=TRUE, slot1=LOCK_0_REL
    tile.mem_trace.notify_event(1, 0, None); // start

    // Submit a lock release on lock 0
    tile.submit_lock_request(LockRequest {
        requestor: LockRequestor::Core,
        lock_id: 0,
        is_acquire: false,
        expected: 0,
        delta: 1,
        equal_mode: false,
    });
    let results = tile.resolve_lock_requests(50);

    assert_eq!(results.len(), 1);
    assert!(results[0].2, "Release should be granted");
    assert!(!results[0].3, "Should be marked as release");

    // Check pending event
    assert_eq!(tile.mem_trace_pending.len(), 1);
    let hw_id = crate::trace::mem_event_to_hw_id(&tile.mem_trace_pending[0].1);
    assert_eq!(hw_id, Some(46), "LOCK_0_REL should be event ID 46");

    // Notify, flush, and verify capture
    tile.mem_trace.notify_event(46, 50, None);
    tile.mem_trace.flush();
    assert!(tile.mem_trace.has_pending_packets(), "Trace unit should have recorded the lock release event");
}

// === Performance Counter Integration Tests ===
// Unit tests for PerfCounterBank are in src/device/perf_counters/mod.rs.
// These tests verify the full CDO register routing path.

#[test]
fn test_perf_counters_init_state() {
    let tile = Tile::compute(0, 2);
    assert_eq!(tile.core_perf_counters.num_counters(), 4);
    assert_eq!(tile.mem_perf_counters.num_counters(), 2);
    for i in 0..4 {
        assert_eq!(tile.core_perf_counters.start_event(i), 0);
        assert_eq!(tile.core_perf_counters.stop_event(i), 0);
        assert_eq!(tile.core_perf_counters.reset_event(i), 0);
        assert_eq!(tile.core_perf_counters.read_counter(i), 0);
        assert_eq!(tile.core_perf_counters.read_event_value(i), 0);
    }
}

#[test]
fn test_perf_counters_memtile_init() {
    let tile = Tile::mem_tile(0, 1);
    assert_eq!(tile.core_perf_counters.num_counters(), 0);
    assert_eq!(tile.mem_perf_counters.num_counters(), 4);
}

#[test]
fn test_perf_counters_shim_init() {
    let tile = Tile::shim(0, 0);
    assert_eq!(tile.core_perf_counters.num_counters(), 2);
    assert_eq!(tile.mem_perf_counters.num_counters(), 0);
}

#[test]
fn test_perf_counter_core_control0_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    let value = (45u32 << 24) | (44 << 16) | (43 << 8) | 42;
    device.write_tile_register(0, 2, 0x31500, value);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.core_perf_counters.start_event(0), 42);
    assert_eq!(tile.core_perf_counters.stop_event(0), 43);
    assert_eq!(tile.core_perf_counters.start_event(1), 44);
    assert_eq!(tile.core_perf_counters.stop_event(1), 45);
}

#[test]
fn test_perf_counter_core_control1_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    let value = (13u32 << 24) | (12 << 16) | (11 << 8) | 10;
    device.write_tile_register(0, 2, 0x31504, value);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.core_perf_counters.start_event(2), 10);
    assert_eq!(tile.core_perf_counters.stop_event(2), 11);
    assert_eq!(tile.core_perf_counters.start_event(3), 12);
    assert_eq!(tile.core_perf_counters.stop_event(3), 13);
}

#[test]
fn test_perf_counter_core_control2_reset() {
    let mut device = super::super::state::DeviceState::new_npu1();
    let value = (8u32 << 24) | (7 << 16) | (6 << 8) | 5;
    device.write_tile_register(0, 2, 0x31508, value);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.core_perf_counters.reset_event(0), 5);
    assert_eq!(tile.core_perf_counters.reset_event(1), 6);
    assert_eq!(tile.core_perf_counters.reset_event(2), 7);
    assert_eq!(tile.core_perf_counters.reset_event(3), 8);
}

#[test]
fn test_perf_counter_core_counter_value_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    device.write_tile_register(0, 2, 0x31520, 0xDEAD_BEEF);
    device.write_tile_register(0, 2, 0x31528, 0xCAFE_BABE);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.core_perf_counters.read_counter(0), 0xDEAD_BEEF);
    assert_eq!(tile.core_perf_counters.read_counter(2), 0xCAFE_BABE);
}

#[test]
fn test_perf_counter_core_event_value_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    device.write_tile_register(0, 2, 0x31580, 1000);
    device.write_tile_register(0, 2, 0x3158C, 5000);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.core_perf_counters.read_event_value(0), 1000);
    assert_eq!(tile.core_perf_counters.read_event_value(3), 5000);
}

#[test]
fn test_perf_counter_memory_module_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    let value = (23u32 << 24) | (22 << 16) | (21 << 8) | 20;
    device.write_tile_register(0, 2, 0x11000, value);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.mem_perf_counters.start_event(0), 20);
    assert_eq!(tile.mem_perf_counters.stop_event(0), 21);
    assert_eq!(tile.mem_perf_counters.start_event(1), 22);
    assert_eq!(tile.mem_perf_counters.stop_event(1), 23);

    device.write_tile_register(0, 2, 0x11024, 42);
    let tile = device.array.get(0, 2).unwrap();
    assert_eq!(tile.mem_perf_counters.read_counter(1), 42);
}

#[test]
fn test_perf_counter_memtile_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    let value = (103u32 << 24) | (102 << 16) | (101 << 8) | 100;
    device.write_tile_register(0, 1, 0x91000, value);
    let tile = device.array.get(0, 1).unwrap();
    assert_eq!(tile.mem_perf_counters.start_event(0), 100);
    assert_eq!(tile.mem_perf_counters.stop_event(0), 101);
    assert_eq!(tile.mem_perf_counters.start_event(1), 102);
    assert_eq!(tile.mem_perf_counters.stop_event(1), 103);

    let value2 = (203u32 << 24) | (202 << 16) | (201 << 8) | 200;
    device.write_tile_register(0, 1, 0x91004, value2);
    let tile = device.array.get(0, 1).unwrap();
    assert_eq!(tile.mem_perf_counters.start_event(2), 200);
    assert_eq!(tile.mem_perf_counters.stop_event(2), 201);
    assert_eq!(tile.mem_perf_counters.start_event(3), 202);
    assert_eq!(tile.mem_perf_counters.stop_event(3), 203);

    device.write_tile_register(0, 1, 0x91020, 0x1234_5678);
    let tile = device.array.get(0, 1).unwrap();
    assert_eq!(tile.mem_perf_counters.read_counter(0), 0x1234_5678);
}

#[test]
fn test_perf_counter_shim_write() {
    let mut device = super::super::state::DeviceState::new_npu1();
    let value = (33u32 << 24) | (32 << 16) | (31 << 8) | 30;
    device.write_tile_register(0, 0, 0x31000, value);
    let tile = device.array.get(0, 0).unwrap();
    assert_eq!(tile.core_perf_counters.start_event(0), 30);
    assert_eq!(tile.core_perf_counters.stop_event(0), 31);
    assert_eq!(tile.core_perf_counters.start_event(1), 32);
    assert_eq!(tile.core_perf_counters.stop_event(1), 33);

    device.write_tile_register(0, 0, 0x31020, 99);
    let tile = device.array.get(0, 0).unwrap();
    assert_eq!(tile.core_perf_counters.read_counter(0), 99);

    device.write_tile_register(0, 0, 0x31080, 500);
    let tile = device.array.get(0, 0).unwrap();
    assert_eq!(tile.core_perf_counters.read_event_value(0), 500);
}

#[test]
fn test_perf_counter_read_out_of_range() {
    let counters = super::super::perf_counters::PerfCounterBank::new(2);
    assert_eq!(counters.read_counter(3), 0);
    assert_eq!(counters.read_event_value(3), 0);
}

#[test]
fn test_perf_counter_write_out_of_range() {
    let mut counters = super::super::perf_counters::PerfCounterBank::new(2);
    counters.write_counter(3, 0xFFFF);
    assert_eq!(counters.read_counter(3), 0);
    counters.write_event_value(3, 0xFFFF);
    assert_eq!(counters.read_event_value(3), 0);
}

#[test]
fn test_lock_release_counter() {
    let mut tile = Tile::compute(0, 2);
    assert_eq!(tile.lock_release_count(), 0);

    // Set lock 0 to value 1 so a release (decrement) can succeed.
    tile.locks[0].value = 1;

    // Submit a release request and resolve.
    tile.defer_core_lock_release(0, 1);
    tile.resolve_lock_requests(0);

    assert_eq!(tile.lock_release_count(), 1);

    // Second release: set lock back to 1 and release again.
    tile.locks[0].value = 1;
    tile.defer_core_lock_release(0, 1);
    tile.resolve_lock_requests(0);

    assert_eq!(tile.lock_release_count(), 2);
}
