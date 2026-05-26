//! Tests for DMA engine.

use super::*;

fn make_tile() -> Tile {
    Tile::compute(1, 2)
}

fn make_host_memory() -> HostMemory {
    HostMemory::new()
}

#[test]
fn test_engine_creation() {
    let engine = DmaEngine::new_compute_tile(1, 2);
    assert_eq!(engine.num_channels(), 4);
    assert_eq!(engine.col, 1);
    assert_eq!(engine.row, 2);
}

#[test]
fn test_mem_tile_engine() {
    let engine = DmaEngine::new_mem_tile(0, 1);
    assert_eq!(engine.num_channels(), 12);
    assert!(engine.tile_kind.is_mem());
}

#[test]
fn test_configure_bd() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);

    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();

    let bd = engine.get_bd(0).unwrap();
    assert_eq!(bd.base_addr, 0x1000);
    assert_eq!(bd.length, 256);
}

#[test]
fn test_invalid_bd_index() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let result = engine.configure_bd(16, BdConfig::simple_1d(0x1000, 256));
    assert!(matches!(result, Err(DmaError::InvalidBd(16))));
}

#[test]
fn test_start_channel() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();

    engine.start_channel(0, 0).unwrap();

    assert!(engine.channel_active(0));
    assert_eq!(engine.channel_state(0), ChannelState::Active);
}

#[test]
fn test_channel_busy_error() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();

    engine.start_channel(0, 0).unwrap();
    let result = engine.start_channel(0, 0);

    assert!(matches!(result, Err(DmaError::ChannelBusy(0))));
}

#[test]
fn test_stop_channel() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();
    engine.start_channel(0, 0).unwrap();

    engine.stop_channel(0).unwrap();

    assert!(!engine.channel_active(0));
    assert_eq!(engine.channel_state(0), ChannelState::Idle);
}

#[test]
fn test_pause_resume() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();
    engine.start_channel(0, 0).unwrap();

    engine.pause_channel(0).unwrap();
    assert_eq!(engine.channel_state(0), ChannelState::Paused);

    engine.resume_channel(0).unwrap();
    assert_eq!(engine.channel_state(0), ChannelState::Active);
}

#[test]
fn test_reset_clears_pending_trace_events() {
    // The coordinator drains trace_events once per cycle, but events
    // generated near end-of-run can linger past the final drain. If
    // reset() doesn't clear them, the next run picks them up on its
    // first drain and emits them in the new trace -- the j>1 parallel
    // sweep saw this as a 3-event count divergence in batches that
    // followed a high-activity batch on the same runner session.
    let mut engine = DmaEngine::new_compute_tile(1, 2);

    engine.set_current_cycle(42);
    engine.trace(EventType::DmaStartTask { channel: 0 });
    engine.trace(EventType::DmaFinishedBd { channel: 0 });

    engine.reset();

    let drained = engine.drain_trace_events();
    assert!(drained.is_empty(), "reset() must clear pending trace events, but found {drained:?}");
}

#[test]
fn test_simple_transfer() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // Write a known pattern to tile data memory at the source address
    let source_data: Vec<u8> = (0..32u8).collect();
    tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&source_data);

    // Configure BD for 32 bytes using MM2S channel (reads from tile memory)
    // Channel 2 is MM2S on compute tiles
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();

    // Start transfer on MM2S channel
    engine.start_channel(2, 0).unwrap();

    // Step until complete. Drain stream_out per cycle to simulate an
    // always-ready downstream consumer; without this, the MM2S backpressures
    // once the slave-port FIFO (4 words) fills and the test never finishes.
    let mut cycles = 0;
    while engine.channel_active(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 100 {
            panic!("Transfer took too long");
        }
    }

    // Verify completion
    assert_eq!(engine.channel_state(2), ChannelState::Idle);

    let stats = engine.channel_stats(2).unwrap();
    assert_eq!(stats.transfers_completed, 1);
    assert_eq!(stats.bytes_transferred, 32);

    // Verify the source data is still intact (DMA read shouldn't modify it)
    assert_eq!(
        &tile.data_memory()[0x100..0x100 + 32],
        &source_data[..],
        "Source data should remain intact after MM2S read"
    );
}

#[test]
fn test_any_channel_active() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 256)).unwrap();

    assert!(!engine.any_channel_active());

    engine.start_channel(0, 0).unwrap();
    assert!(engine.any_channel_active());
}

#[test]
fn test_channel_type() {
    let engine = DmaEngine::new_compute_tile(1, 2);

    assert_eq!(engine.channel_type(0), ChannelType::S2MM);
    assert_eq!(engine.channel_type(1), ChannelType::S2MM);
    assert_eq!(engine.channel_type(2), ChannelType::MM2S);
    assert_eq!(engine.channel_type(3), ChannelType::MM2S);
}

#[test]
fn test_transfer_with_lock() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // Set lock to available state (value=1 for acq_eq mode)
    tile.locks[5].set(1);

    // Configure BD with lock using MM2S channel
    // Per AMD spec: acquire_value=1 means acq_eq (wait for lock==1, then decrement)
    // Per AMD spec: release_value=1 means add +1 to lock after transfer
    // release_value=0 would mean NO release per AM025
    let bd = BdConfig::simple_1d(0x100, 32)
        .with_acquire(5, 1) // Wait for lock==1, decrement by 1 (1->0)
        .with_release(5, 1); // After transfer, add +1 to lock (0->1)
    engine.configure_bd(0, bd).unwrap();

    // Start should trigger lock acquire on MM2S channel
    engine.start_channel(2, 0).unwrap();

    // Step until complete (cycle-accurate timing needs more cycles).
    // Arbiter-based lock arbitration: submit -> resolve -> step.
    // Drain stream_out so MM2S doesn't stall on the 4-word slave-port FIFO.
    let mut cycles = 0;
    while engine.channel_active(2) {
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(0);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 500 {
            panic!("Transfer took too long: {} cycles", cycles);
        }
    }

    // Verify lock was released: started at 1, acquired (->0), released +1 (->1)
    assert_eq!(tile.locks[5].value, 1);
}

#[test]
fn test_execute_1d_transfer() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // Write a recognizable pattern to source memory
    let source_data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
    let dm = tile.data_memory_mut();
    dm[0x100..0x100 + 64].copy_from_slice(&source_data);

    // Use MM2S channel (channel 2) for testing
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();

    let cycles = engine
        .execute_1d_transfer(2, 0, &mut tile, &mut NeighborTiles::empty(), &mut host_mem)
        .unwrap();
    assert!(cycles > 0, "Transfer should take at least one cycle");

    let stats = engine.channel_stats(2).unwrap();
    assert_eq!(stats.bytes_transferred, 64);
    assert_eq!(stats.transfers_completed, 1);

    // Source data should still be intact
    assert_eq!(
        &tile.data_memory()[0x100..0x100 + 64],
        &source_data[..],
        "Source data should remain intact after 1D transfer"
    );
}

#[test]
fn test_reset() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 256)).unwrap();
    engine.start_channel(0, 0).unwrap();

    engine.reset();

    assert!(!engine.any_channel_active());
    // BD config should still be there
    assert!(engine.get_bd(0).unwrap().valid);
}

#[test]
fn test_default_cycle_accurate_timing() {
    // Cycle-accurate timing is the default and only mode
    let engine = DmaEngine::new_compute_tile(1, 2);
    assert_eq!(engine.timing_config().bd_setup_cycles, 4);
    // AIE2: 128-bit bus = 4 words/cycle (xaiemlgbl_params.h DATAMEMORY_WIDTH=128)
    assert_eq!(engine.timing_config().words_per_cycle, 4);
}

/// First-BD-of-task bonus: channel_start_cycles applies to every tile kind.
///
/// On a fresh compute MM2S channel the FSM transitions
/// Idle -> BdSetup -> MemoryLatency -> Transferring. Without the bonus,
/// MemoryLatency budget is `memory_latency_cycles` (5). With the bonus,
/// it is `memory_latency_cycles + channel_start_cycles` (5 + 2 = 7) on
/// the first BD only. Subsequent chained BDs don't pay it.
#[test]
fn test_first_bd_bonus_applies_channel_start_cycles() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

    assert!(engine.channels[2].is_first_bd, "fresh channel should start with is_first_bd=true");
    engine.start_channel(2, 0).unwrap();
    assert!(engine.channels[2].is_first_bd, "is_first_bd shouldn't clear before MemoryLatency entry");

    // Step through BdSetup (4 cycles: cycles_remaining counts down 4->1).
    for _ in 0..4 {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
    }

    // Now in MemoryLatency. Bonus should be consumed and budget should be
    // memory_latency + channel_start = 5 + 2 = 7 (not just 5).
    assert!(!engine.channels[2].is_first_bd, "is_first_bd cleared on MemoryLatency entry");
    match &engine.channels[2].fsm {
        crate::device::dma::channel::ChannelFsm::MemoryLatency { cycles_remaining, .. } => {
            assert_eq!(*cycles_remaining, 7, "first-BD bonus folds channel_start into MemoryLatency");
        }
        other => panic!("expected MemoryLatency after BdSetup, got {:?}", other.phase_name()),
    }
}

/// First-BD bonus re-arms after the channel returns to Idle.
#[test]
fn test_first_bd_bonus_rearms_after_idle() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

    engine.start_channel(2, 0).unwrap();
    let mut cycles = 0;
    while engine.channel_active(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        assert!(cycles < 100, "transfer hung");
    }

    // Channel returned to Idle; flag should be re-armed for the next task.
    assert!(engine.channels[2].is_first_bd, "is_first_bd re-armed when channel goes Idle");
}

/// Shim DDR cold-start fires only on shim+host-memory first BDs.
///
/// On compute the bonus is `channel_start_cycles` only (no DDR cold-start).
/// On shim with a host-memory transfer the bonus adds either
/// `shim_ddr_cold_start_mm2s_cycles` or `_s2mm_cycles` on top, depending
/// on the transfer direction.
#[test]
fn test_shim_ddr_cold_start_only_on_shim_with_host_memory() {
    let cfg = DmaTimingConfig::default();
    assert_eq!(cfg.channel_start_cycles, 2);
    // Calibrated 2026-05-25 from _diag_shim_throughput_sweep.
    assert_eq!(cfg.shim_ddr_cold_start_mm2s_cycles, 747);
    assert_eq!(cfg.shim_ddr_cold_start_s2mm_cycles, 171);
    assert_eq!(cfg.shim_words_per_cycle, 1);

    // stop_channel re-arms the flag (covers external interrupts of in-flight tasks).
    let mut engine = DmaEngine::new_shim_tile(0, 0);
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 16)).unwrap();
    engine.start_channel(0, 0).unwrap();
    engine.stop_channel(0).unwrap();
    assert!(engine.channels[0].is_first_bd, "stop_channel re-arms the bonus flag");
}

#[test]
fn test_cycle_accurate_transfer() {
    // Cycle-accurate timing is the default
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // Write recognizable data to source address
    let source_data: [u8; 16] =
        [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
    let dm = tile.data_memory_mut();
    dm[0x100..0x100 + 16].copy_from_slice(&source_data);

    // Configure BD for 16 bytes (4 words) using MM2S channel
    // With AIE2 spec: 4 setup + 1 start + 5 mem latency + 4 data cycles = 14+ cycles
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

    // Start transfer on MM2S channel
    engine.start_channel(2, 0).unwrap();

    // Step until complete
    let mut cycles = 0;
    while engine.channel_active(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        cycles += 1;
        if cycles > 100 {
            panic!("Transfer took too long");
        }
    }

    // Cycle-accurate transfer should have timing overhead
    assert!(cycles >= 6, "Cycle-accurate transfer should have overhead, got {} cycles", cycles);

    let stats = engine.channel_stats(2).unwrap();
    assert_eq!(stats.bytes_transferred, 16);
    assert_eq!(stats.transfers_completed, 1);

    // Source data should be unchanged
    assert_eq!(
        &tile.data_memory()[0x100..0x100 + 16],
        &source_data[..],
        "Source data integrity after cycle-accurate transfer"
    );
}

#[test]
fn test_lock_timing_integration() {
    // Create engine with lock timing enabled (cycle-accurate is default)
    let mut engine = DmaEngine::new_compute_tile(1, 2).with_lock_timing(16);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // Configure BD that requires a lock
    let mut bd = BdConfig::simple_1d(0x100, 32);
    bd.acquire_lock = Some(5);
    bd.acquire_value = 1;
    engine.configure_bd(0, bd).unwrap();

    // Lock starts at 0, so acquire will fail initially
    tile.locks[5].value = 0;

    // Start transfer - should go to WaitingForLock
    engine.start_channel(0, 0).unwrap();

    // Step a few times - lock still not available
    for _ in 0..3 {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
    }

    // Verify waiting state
    assert!(
        matches!(engine.channel_state(0), ChannelState::WaitingForLock(5)),
        "Expected WaitingForLock(5), got {:?}",
        engine.channel_state(0)
    );

    // Check lock timing tracked contention
    let lock_timing = engine.lock_timing().unwrap();
    let contention = lock_timing.current_stall(5);
    assert!(contention >= 3, "Should have tracked at least 3 stall cycles, got {}", contention);
}

#[test]
fn test_lock_timing_success() {
    // Test that successful lock acquire is tracked
    let mut engine = DmaEngine::new_compute_tile(1, 2).with_lock_timing(16);
    let mut tile = make_tile();
    let _host_mem = make_host_memory();

    // Configure simple BD (no lock requirement)
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

    // But test lock timing directly - set lock value so acquire succeeds
    tile.locks[3].value = 1;

    // Start channel
    engine.start_channel(0, 0).unwrap();

    // Manually trigger lock acquire tracking (simulating what happens internally)
    if let Some(timing) = engine.lock_timing_mut() {
        timing.track_acquire(3, true); // Success
    }

    // Check stats
    let lock_timing = engine.lock_timing().unwrap();
    let stats = lock_timing.stats(3).unwrap();
    assert_eq!(stats.acquires, 1, "Should have recorded 1 acquire");
}

#[test]
fn test_memtile_bd_capacity() {
    // MemTile should support 48 BDs
    let mut engine = DmaEngine::new_mem_tile(0, 1);

    // Should be able to configure BD 0
    assert!(engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).is_ok());

    // Should be able to configure BD 47 (last valid for MemTile)
    assert!(engine.configure_bd(47, BdConfig::simple_1d(0x100, 16)).is_ok());

    // BD 48 should fail
    assert!(engine.configure_bd(48, BdConfig::simple_1d(0x100, 16)).is_err());

    // Compute tile should only support 16 BDs
    let mut compute = DmaEngine::new_compute_tile(0, 2);
    assert!(compute.configure_bd(15, BdConfig::simple_1d(0x100, 16)).is_ok());
    assert!(compute.configure_bd(16, BdConfig::simple_1d(0x100, 16)).is_err());
}

// === Task Queue Tests ===

#[test]
fn test_task_queue_enqueue() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();
    engine.configure_bd(1, BdConfig::simple_1d(0x200, 32)).unwrap();

    // Enqueue first task - should start immediately (channel was idle)
    assert!(engine.enqueue_task(2, 0, 0, false));
    assert_eq!(engine.task_queue_size(2), 0); // Task started, not queued

    // Enqueue second task - should be queued (channel is busy)
    assert!(engine.enqueue_task(2, 1, 0, true));
    assert_eq!(engine.task_queue_size(2), 1);
}

#[test]
fn test_task_queue_overflow() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();

    // First task starts immediately
    assert!(engine.enqueue_task(2, 0, 0, false));

    // Queue 8 more tasks (fills the queue)
    for i in 0..MAX_TASK_QUEUE_DEPTH {
        assert!(engine.enqueue_task(2, 0, 0, false), "Task {} should enqueue", i);
    }

    // 9th task should fail (queue full)
    assert!(!engine.enqueue_task(2, 0, 0, false));

    // Overflow flag should be set
    assert!(engine.task_queue_overflow(2));

    // Clear the flag
    engine.clear_task_queue_overflow(2);
    assert!(!engine.task_queue_overflow(2));
}

#[test]
fn test_task_queue_status_register() {
    let layout = &crate::device::regdb::device_reg_layout().memory_status;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 32)).unwrap();

    // Enqueue some tasks (first starts, rest are queued)
    engine.enqueue_task(2, 0, 0, false); // Starts immediately
    engine.enqueue_task(2, 0, 0, false); // Queued
    engine.enqueue_task(2, 0, 0, false); // Queued

    let status = engine.get_channel_status(2);

    // Task_Queue_Size should be 2
    assert_eq!(layout.task_queue_size.extract(status), 2);

    // Channel should be running
    assert!(layout.channel_running.extract_bool(status));
}

#[test]
fn test_mm2s_stalled_stream_backpressure_bit() {
    // AM025 names the bit-4 stall in DMA_MM2S_Status_0 as
    // Stalled_Stream_Backpressure: channel stalled because the
    // downstream slave FIFO has no space. We exercise that by
    // configuring an MM2S transfer larger than the local-slave FIFO
    // depth and never draining stream_out -- after the first
    // output_fifo_capacity() words land, the channel must report
    // the bit set.
    let layout = &crate::device::regdb::device_reg_layout().memory_status;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    let cap = engine.output_fifo_capacity();
    let bd_bytes = (cap as u32 + 4) * 4;
    engine.configure_bd(0, BdConfig::simple_1d(0x100, bd_bytes)).unwrap();
    engine.enqueue_task(2, 0, 0, false);

    // Drive the engine until stream_out reaches capacity. No draining,
    // so MM2S must stall once the slave FIFO model is full.
    let mut steps = 0;
    while engine.stream_out_len() < cap && steps < 100 {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        steps += 1;
    }
    assert!(
        engine.stream_out_len() >= cap,
        "MM2S never reached stream_out capacity ({} words after {} cycles)",
        engine.stream_out_len(),
        steps,
    );

    // One more step in the stalled condition so the FSM stays in
    // Transferring with backpressure active.
    engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);

    let status = engine.get_channel_status(2);
    assert!(
        layout.stalled_stream_backpressure.extract_bool(status),
        "MM2S Stalled_Stream_Backpressure (bit 4) should be set when \
         stream_out is at capacity, got status=0x{:08X}",
        status,
    );
    assert!(
        layout.channel_running.extract_bool(status),
        "Channel_Running should remain set during a stream stall, got \
         status=0x{:08X}",
        status,
    );
}

#[test]
fn test_task_queue_multiple_tasks_complete() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();
    engine.configure_bd(1, BdConfig::simple_1d(0x200, 16)).unwrap();

    // Enqueue two tasks with token issue on the second
    engine.enqueue_task(2, 0, 0, false);
    engine.enqueue_task(2, 1, 0, true);

    // Run until all work is complete (including queued tasks). Drain
    // stream_out per cycle so MM2S doesn't stall on the 4-word slave FIFO.
    let mut cycles = 0;
    while engine.channel_has_pending_work(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 100 {
            panic!("Tasks took too long");
        }
    }

    // Both transfers should have completed
    let stats = engine.channel_stats(2).unwrap();
    assert_eq!(stats.transfers_completed, 2);
    assert_eq!(stats.bytes_transferred, 32); // 16 + 16

    // Second task had enable_token_issue, so should have emitted a token
    assert!(engine.has_task_token());
    let token = engine.pop_task_token().unwrap();
    assert_eq!(token.channel_id, 2);
}

#[test]
fn test_task_queue_with_repeat() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

    // Enqueue a task with repeat_count=2 (runs 3 times total)
    engine.enqueue_task(2, 0, 2, true);

    // Run until all work is complete (including repeats). MM2S transfers
    // need a downstream consumer or stream_out backpressure will stall;
    // drain the queue each cycle to simulate one.
    let mut cycles = 0;
    while engine.channel_has_pending_work(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 200 {
            panic!("Repeated task took too long");
        }
    }

    // Should have transferred 3 * 16 = 48 bytes
    let stats = engine.channel_stats(2).unwrap();
    assert_eq!(stats.bytes_transferred, 48);
    assert_eq!(stats.transfers_completed, 3);
}

/// Regression test: BD chain + repeat must restart from the FIRST BD,
/// not the last one. Previously, `current_bds` was overwritten to the
/// last chained BD, so repeat only re-executed the tail of the chain.
#[test]
fn test_bd_chain_with_repeat_restarts_from_start() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // BD0 (16 bytes at 0x100) chains to BD1 (16 bytes at 0x200)
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16).with_next(1)).unwrap();
    engine.configure_bd(1, BdConfig::simple_1d(0x200, 16)).unwrap();

    // Enqueue task: start at BD0, repeat_count=1 (run chain twice total)
    // Expected: BD0->BD1, BD0->BD1 = 4 transfers, 64 bytes
    engine.enqueue_task(2, 0, 1, false);

    let mut cycles = 0;
    while engine.channel_has_pending_work(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 500 {
            panic!("BD chain with repeat took too long (>500 cycles)");
        }
    }

    let stats = engine.channel_stats(2).unwrap();
    // 2 iterations x 2 BDs per chain = 4 transfers
    assert_eq!(
        stats.transfers_completed, 4,
        "Expected 4 transfers (2 chain iterations x 2 BDs), got {}",
        stats.transfers_completed
    );
    // 4 transfers x 16 bytes = 64 bytes total
    assert_eq!(stats.bytes_transferred, 64, "Expected 64 bytes (4 x 16), got {}", stats.bytes_transferred);
}

/// Self-chaining BD (next_bd == self) with Use_Next_BD=1 loops indefinitely
/// on real hardware -- the silicon does NOT detect chain cycles. The chain
/// terminates only when Use_Next_BD=0 (next_bd = None in our model).
///
/// This test verifies the hardware-correct behavior: a self-chaining BD
/// runs continuously until the cycle limit.
#[test]
fn test_self_chaining_bd_loops_indefinitely() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // BD0 chains to itself (self-chain). Hardware follows next_bd blindly.
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16).with_next(0)).unwrap();
    engine.enqueue_task(2, 0, 0, false);

    // Run for a fixed number of cycles -- channel should still be active.
    // Drain stream_out each cycle so MM2S backpressure doesn't stall the loop.
    for _ in 0..200 {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
    }

    assert!(
        engine.channel_has_pending_work(2),
        "Self-chaining BD should still be running (hardware loops indefinitely)"
    );

    let stats = engine.channel_stats(2).unwrap();
    assert!(
        stats.transfers_completed > 2,
        "Expected multiple transfers from self-chaining loop, got {}",
        stats.transfers_completed
    );
}

// === Stream Port Integration Tests ===

#[test]
fn test_compute_tile_port_mappings() {
    let engine = DmaEngine::new_compute_tile(1, 2);

    // MM2S channels send to slave ports 1, 2
    assert_eq!(engine.mm2s_slave_port(0), 1);
    assert_eq!(engine.mm2s_slave_port(1), 2);

    // S2MM channels receive from master ports 1, 2
    assert_eq!(engine.s2mm_master_port(0), 1);
    assert_eq!(engine.s2mm_master_port(1), 2);

    // Channel counts
    assert_eq!(engine.s2mm_channel_count(), 2);
    assert_eq!(engine.mm2s_channel_count(), 2);
}

#[test]
fn test_memtile_port_mappings() {
    let engine = DmaEngine::new_mem_tile(0, 1);

    // MemTile: DMA channels map directly to ports 0-5
    for ch in 0..6 {
        assert_eq!(engine.mm2s_slave_port(ch), ch);
        assert_eq!(engine.s2mm_master_port(ch), ch);
    }

    // Channel counts
    assert_eq!(engine.s2mm_channel_count(), 6);
    assert_eq!(engine.mm2s_channel_count(), 6);
}

#[test]
fn test_stream_data_to_word_conversion() {
    use crate::device::dma::stream_io::StreamWord;

    let data = StreamData { data: 0x12345678, tlast: true, channel: 2 };

    // Convert to StreamWord
    let word: StreamWord = data.into();
    assert_eq!(word.data, 0x12345678);
    assert!(word.tlast);
    // Parity should be computed
    assert_eq!(word.parity, StreamWord::compute_parity(0x12345678));
}

#[test]
fn test_stream_word_to_data_conversion() {
    use crate::device::dma::stream_io::StreamWord;

    let word = StreamWord { data: 0xDEADBEEF, tlast: false, parity: true };

    // Convert to StreamData with channel
    let data = StreamData::from_stream_word(word, 3);
    assert_eq!(data.data, 0xDEADBEEF);
    assert!(!data.tlast);
    assert_eq!(data.channel, 3);
}

#[test]
fn test_engine_stream_word_interface() {
    use crate::device::dma::stream_io::StreamWord;

    let mut engine = DmaEngine::new_compute_tile(1, 2);

    // Push via StreamWord interface
    let word = StreamWord::with_tlast(0xCAFEBABE);
    assert!(engine.push_stream_in_from_word(word, 0));

    // Verify it's in the buffer
    assert!(engine.has_stream_in_for_channel(0));
    assert_eq!(engine.stream_in_len(), 1);

    // Add stream output data directly
    engine.push_stream_out(StreamData { data: 0x11111111, tlast: false, channel: 2 });

    // Pop as StreamWord
    let (out_word, channel) = engine.pop_stream_out_as_word().unwrap();
    assert_eq!(out_word.data, 0x11111111);
    assert!(!out_word.tlast);
    assert_eq!(channel, 2);
}

// === Compression / Decompression Integration Tests ===

#[test]
fn test_mm2s_compression_sparse_data() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    // Write sparse data: only bytes 0, 3, 8 are non-zero
    tile.data_memory_mut()[0] = 5;
    tile.data_memory_mut()[3] = 3;
    tile.data_memory_mut()[8] = 7;

    // Enable compression on MM2S channel 2
    engine.set_channel_compression_config(2, true, false, false);

    let result = engine.transfer_mm2s(0, 32, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(result);

    // mask + 1 data word (3 bytes + padding) = 2 stream words
    assert_eq!(engine.stream_out_len(), 2);

    let mask_word = engine.pop_stream_out().unwrap();
    assert_eq!(mask_word.data, (1 << 0) | (1 << 3) | (1 << 8));
    assert!(!mask_word.tlast);

    let data_word = engine.pop_stream_out().unwrap();
    assert_eq!(data_word.data, u32::from_le_bytes([5, 3, 7, 0]));
    assert!(data_word.tlast);
}

#[test]
fn test_mm2s_compression_all_zeros() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    engine.set_channel_compression_config(2, true, false, false);

    let result = engine.transfer_mm2s(0, 32, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(result);

    // All zeros: just mask word, no data
    assert_eq!(engine.stream_out_len(), 1);
    let mask_word = engine.pop_stream_out().unwrap();
    assert_eq!(mask_word.data, 0);
    assert!(mask_word.tlast);
}

#[test]
fn test_s2mm_decompression_round_trip() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.data_memory_mut()[0] = 42;
    tile.data_memory_mut()[15] = 128;
    tile.data_memory_mut()[31] = 255;

    // MM2S compress from offset 0
    engine.set_channel_compression_config(2, true, false, false);
    let result = engine.transfer_mm2s(0, 32, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(result);

    // Route compressed data from stream_out to stream_in (channel 0)
    while let Some(sd) = engine.pop_stream_out() {
        engine.push_stream_in(StreamData { data: sd.data, tlast: sd.tlast, channel: 0 });
    }

    // S2MM decompress to offset 256
    engine.set_channel_compression_config(0, false, true, false);
    let result = engine.transfer_s2mm(256, 32, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);
    assert_eq!(result.bytes_written, 32);

    let data = tile.data_memory();
    assert_eq!(data[256], 42);
    assert_eq!(data[256 + 15], 128);
    assert_eq!(data[256 + 31], 255);
    for i in 0..32 {
        if i != 0 && i != 15 && i != 31 {
            assert_eq!(data[256 + i], 0, "byte {} should be zero", i);
        }
    }
}

#[test]
fn test_mm2s_no_compression_when_disabled() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.data_memory_mut()[0] = 0xAA;
    tile.data_memory_mut()[1] = 0xBB;
    tile.data_memory_mut()[2] = 0xCC;
    tile.data_memory_mut()[3] = 0xDD;

    assert!(!engine.is_compression_enabled(2));

    let result = engine.transfer_mm2s(0, 4, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(result);

    assert_eq!(engine.stream_out_len(), 1);
    let word = engine.pop_stream_out().unwrap();
    assert_eq!(word.data, u32::from_le_bytes([0xAA, 0xBB, 0xCC, 0xDD]));
    assert!(word.tlast);
}

#[test]
fn test_s2mm_no_decompression_when_disabled() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    engine.push_stream_in(StreamData { data: 0xDEADBEEF, tlast: true, channel: 0 });

    assert!(!engine.is_decompression_enabled(0));

    let result = engine.transfer_s2mm(0, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);
    assert_eq!(result.bytes_written, 4);

    let data = tile.data_memory();
    assert_eq!(u32::from_le_bytes([data[0], data[1], data[2], data[3]]), 0xDEADBEEF,);
}

#[test]
fn test_compression_multiple_blocks() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.data_memory_mut()[0] = 0x11;
    tile.data_memory_mut()[32] = 0x22;

    engine.set_channel_compression_config(2, true, false, false);

    let result = engine.transfer_mm2s(0, 64, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(result);

    // Block 0: mask + data = 2 words; Block 1: mask + data = 2 words
    assert_eq!(engine.stream_out_len(), 4);

    let w0 = engine.pop_stream_out().unwrap();
    assert_eq!(w0.data, 1u32);
    assert!(!w0.tlast);

    let w1 = engine.pop_stream_out().unwrap();
    assert_eq!(w1.data, u32::from_le_bytes([0x11, 0, 0, 0]));
    assert!(!w1.tlast);

    let w2 = engine.pop_stream_out().unwrap();
    assert_eq!(w2.data, 1u32);
    assert!(!w2.tlast);

    let w3 = engine.pop_stream_out().unwrap();
    assert_eq!(w3.data, u32::from_le_bytes([0x22, 0, 0, 0]));
    assert!(w3.tlast);
}

#[test]
fn test_resolve_lock_id_memtile() {
    // MemTile: 64 locks, 192-entry address space
    let tile_kind = TileKind::Mem;
    let num_locks = 64;

    // West neighbor: IDs 0-63
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 0), Some(LockTarget::West(0)));
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 63), Some(LockTarget::West(63)));

    // Own tile: IDs 64-127
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 64), Some(LockTarget::Own(0)));
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 127), Some(LockTarget::Own(63)));

    // East neighbor: IDs 128-191
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 128), Some(LockTarget::East(0)));
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 191),
        Some(LockTarget::East(63))
    );

    // Out of range
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 1, num_locks, 192), None);
}

#[test]
fn test_resolve_lock_id_compute() {
    // Compute tiles: 4-bit field, always Own
    let tile_kind = TileKind::Compute;
    let num_locks = 16;
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 1, 2, num_locks, 5), Some(LockTarget::Own(5)));
}

#[test]
fn test_cross_tile_lock_acquire_west() {
    // Create MemTile DMA engine at col 1, row 1
    let mut engine = DmaEngine::new_mem_tile(1, 1);

    // Create own tile and west neighbor tile
    let mut own_tile = Tile::mem_tile(1, 1);
    let mut west_tile = Tile::mem_tile(0, 1);

    // Set west tile's lock 5 to value 1 (will be acquired via acq_eq)
    west_tile.locks[5].set(1);

    // Configure BD with acquire on west neighbor lock 5.
    // West locks are IDs 0-63, so lock_id=5 means west lock 5.
    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32).with_acquire(5, 1); // acq_eq: wait for value == 1
    engine.configure_bd(0, bd).unwrap();

    // Write data to own tile memory (MM2S reads from here)
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xAA; 32]);

    // Start MM2S channel (channel 6 for MemTile)
    engine.start_channel(6, 0).unwrap();
    assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(5)));

    // Submit lock requests, resolve arbiters, then step
    {
        let mut neighbors = NeighborTiles { west: Some(&mut west_tile), east: None };
        engine.submit_lock_requests(&mut own_tile, &mut neighbors);
    }
    own_tile.resolve_lock_requests(0);
    west_tile.resolve_lock_requests(0);

    let mut neighbors = NeighborTiles { west: Some(&mut west_tile), east: None };
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    // Channel should now be active (lock acquired from west neighbor)
    assert_eq!(
        engine.channel_state(6),
        ChannelState::Active,
        "Channel should be active after acquiring west neighbor lock"
    );
}

#[test]
fn test_cross_tile_lock_acquire_east() {
    // Create MemTile DMA engine at col 1, row 1
    let mut engine = DmaEngine::new_mem_tile(1, 1);

    let mut own_tile = Tile::mem_tile(1, 1);
    let mut east_tile = Tile::mem_tile(2, 1);

    // Set east tile's lock 10 to value 1
    east_tile.locks[10].set(1);

    // East locks are IDs 128-191, so lock_id=138 means east lock 10.
    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32).with_acquire(138, 1); // acq_eq on east lock 10
    engine.configure_bd(0, bd).unwrap();
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xBB; 32]);

    engine.start_channel(6, 0).unwrap();
    assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(138)));

    {
        let mut neighbors = NeighborTiles { west: None, east: Some(&mut east_tile) };
        engine.submit_lock_requests(&mut own_tile, &mut neighbors);
    }
    own_tile.resolve_lock_requests(0);
    east_tile.resolve_lock_requests(0);

    let mut neighbors = NeighborTiles { west: None, east: Some(&mut east_tile) };
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    assert_eq!(
        engine.channel_state(6),
        ChannelState::Active,
        "Channel should be active after acquiring east neighbor lock"
    );
}

#[test]
fn test_cross_tile_lock_acquire_fails_without_neighbor() {
    // MemTile at col 0 has no west neighbor
    let mut engine = DmaEngine::new_mem_tile(0, 1);
    let mut own_tile = Tile::mem_tile(0, 1);
    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32).with_acquire(5, 1); // West lock -- but no west neighbor at col 0
    engine.configure_bd(0, bd).unwrap();
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xCC; 32]);

    engine.start_channel(6, 0).unwrap();

    let mut neighbors = NeighborTiles::empty();
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    // Should remain waiting -- no neighbor to satisfy lock
    assert!(
        matches!(engine.channel_state(6), ChannelState::WaitingForLock(5)),
        "Should stay waiting when neighbor tile is absent"
    );
}

#[test]
fn test_cross_tile_lock_release_west() {
    // Verify that after a transfer, the release lock targets the west neighbor
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);
    let mut west_tile = Tile::mem_tile(0, 1);

    // BD: acquire own lock 0 (ID 64), release west lock 3 (ID 3).
    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32)
        .with_acquire(64, 1) // own lock 0, acq_eq value=1
        .with_release(3, 1); // west lock 3, release delta +1
    engine.configure_bd(0, bd).unwrap();
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xDD; 32]);

    // Set own lock 0 to 1 so acquire succeeds
    own_tile.locks[0].set(1);

    engine.start_channel(6, 0).unwrap();

    // Run to completion
    let mut cycles = 0;
    while engine.channel_active(6) {
        {
            let mut neighbors = NeighborTiles { west: Some(&mut west_tile), east: None };
            engine.submit_lock_requests(&mut own_tile, &mut neighbors);
        }
        own_tile.resolve_lock_requests(0);
        west_tile.resolve_lock_requests(0);
        let mut neighbors = NeighborTiles { west: Some(&mut west_tile), east: None };
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);
        // Drain MM2S output so the 4-word slave FIFO never backpressures.
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 500 {
            panic!("Transfer took too long: {} cycles, state={:?}", cycles, engine.channel_state(6));
        }
    }

    // West tile lock 3 should have been incremented by +1 (release delta)
    assert_eq!(west_tile.locks[3].value, 1, "West neighbor lock 3 should be 1 after release with delta +1");
}

#[test]
fn test_cross_tile_lock_release_east() {
    // Symmetric to test_cross_tile_lock_release_west: verify release
    // targets east neighbor lock via the 128-191 ID range.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);
    let mut east_tile = Tile::mem_tile(2, 1);

    // BD: acquire own lock 0 (ID 64), release east lock 7 (ID 128+7=135)
    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32)
        .with_acquire(64, 1) // own lock 0, acq_eq value=1
        .with_release(135, 1); // east lock 7, release delta +1
    engine.configure_bd(0, bd).unwrap();
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xEE; 32]);

    // Set own lock 0 to 1 so acquire succeeds
    own_tile.locks[0].set(1);

    engine.start_channel(6, 0).unwrap();

    // Run to completion
    let mut cycles = 0;
    while engine.channel_active(6) {
        {
            let mut neighbors = NeighborTiles { west: None, east: Some(&mut east_tile) };
            engine.submit_lock_requests(&mut own_tile, &mut neighbors);
        }
        own_tile.resolve_lock_requests(0);
        east_tile.resolve_lock_requests(0);
        let mut neighbors = NeighborTiles { west: None, east: Some(&mut east_tile) };
        let mut host_mem = make_host_memory();
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);
        // Drain MM2S output so the 4-word slave FIFO never backpressures.
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        if cycles > 500 {
            panic!("Transfer took too long: {} cycles, state={:?}", cycles, engine.channel_state(6));
        }
    }

    // East tile lock 7 should have been incremented by +1 (release delta)
    assert_eq!(east_tile.locks[7].value, 1, "East neighbor lock 7 should be 1 after release with delta +1");
}

#[test]
fn test_cross_tile_lock_own_acquire_memtile() {
    // Verify that lock_id in the 64-127 range (Own) works for MemTile DMA.
    // This exercises the second region of the 192-entry address space.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);

    // Set own lock 10 to value 1 (lock_id = 64 + 10 = 74)
    own_tile.locks[10].set(1);

    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32).with_acquire(74, 1); // own lock 10, acq_eq value=1
    engine.configure_bd(0, bd).unwrap();
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xCC; 32]);

    engine.start_channel(6, 0).unwrap();
    assert!(matches!(engine.channel_state(6), ChannelState::WaitingForLock(74)));

    // Submit and resolve -- no neighbors needed for own-tile lock
    {
        let mut neighbors = NeighborTiles::empty();
        engine.submit_lock_requests(&mut own_tile, &mut neighbors);
    }
    own_tile.resolve_lock_requests(0);

    let mut neighbors = NeighborTiles::empty();
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    assert_eq!(
        engine.channel_state(6),
        ChannelState::Active,
        "Channel should be active after acquiring own lock via memtile ID 74"
    );
}

#[test]
fn test_cross_tile_lock_acquire_no_east_neighbor() {
    // MemTile at col 3 (rightmost in 4-column array) has no east neighbor.
    // East lock access should remain waiting.
    let mut engine = DmaEngine::new_mem_tile(3, 1);
    let mut own_tile = Tile::mem_tile(3, 1);

    // East lock 0 = lock_id 128.
    // Address 0x80100 = Own window (0x80000) + offset 0x100; see MemTileTarget.
    let bd = BdConfig::simple_1d(0x80100, 32).with_acquire(128, 1);
    engine.configure_bd(0, bd).unwrap();
    own_tile.data_memory_mut()[0x100..0x100 + 32].copy_from_slice(&[0xDD; 32]);

    engine.start_channel(6, 0).unwrap();

    let mut neighbors = NeighborTiles::empty();
    let mut host_mem = make_host_memory();
    engine.step(&mut own_tile, &mut neighbors, &mut host_mem);

    // Should remain waiting -- no east neighbor to satisfy lock
    assert!(
        matches!(engine.channel_state(6), ChannelState::WaitingForLock(128)),
        "Should stay waiting when east neighbor tile is absent"
    );
}

// === Cross-MemTile data access tests ===
//
// MemTile DMA BDs encode addresses in a windowed three-tile space:
//   West=[0, 0x80000), Own=[0x80000, 0x100000), East=[0x100000, 0x180000).
// These tests verify that data reads/writes are routed to the correct
// neighbour tile via the MemTile-to-MemTile shared-memory bus, not the
// stream switch. See `MemTileTarget` in `dma::engine::types`.

/// Drive an MM2S MemTile DMA channel to completion in one helper, asserting
/// it never goes to Error and never spins past `max_cycles`.
fn run_memtile_mm2s_to_completion(
    engine: &mut DmaEngine,
    channel: u8,
    own: &mut Tile,
    neighbors_west: Option<&mut Tile>,
    neighbors_east: Option<&mut Tile>,
    max_cycles: usize,
) {
    use std::cell::RefCell;
    // Stash neighbours in local RefCells so we can rebuild a NeighborTiles
    // each iteration without violating disjoint-borrow rules.
    let west_cell = neighbors_west.map(RefCell::new);
    let east_cell = neighbors_east.map(RefCell::new);

    let mut host_mem = make_host_memory();
    for cycle in 0..max_cycles {
        let mut west_borrow = west_cell.as_ref().map(|c| c.borrow_mut());
        let mut east_borrow = east_cell.as_ref().map(|c| c.borrow_mut());
        let mut neighbors = NeighborTiles {
            west: west_borrow.as_deref_mut().map(|b| &mut **b),
            east: east_borrow.as_deref_mut().map(|b| &mut **b),
        };
        engine.step(own, &mut neighbors, &mut host_mem);
        drop(west_borrow);
        drop(east_borrow);

        if !engine.channel_active(channel) {
            return;
        }
        if matches!(engine.channel_state(channel), ChannelState::Error) {
            panic!("DMA channel {} entered Error state at cycle {}", channel, cycle);
        }
    }
    panic!("DMA channel {} did not complete within {} cycles", channel, max_cycles);
}

#[test]
fn test_memtile_mm2s_reads_from_east_neighbor() {
    // Verify a MemTile MM2S BD with a base_addr in the East window reads
    // from the neighbour tile's memory, not the local memory.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);
    let mut east_tile = Tile::mem_tile(2, 1);

    // Seed both tiles distinctly so a wrong route would be obvious.
    own_tile.data_memory_mut()[0x100..0x110].copy_from_slice(&[0xAA; 16]);
    east_tile.data_memory_mut()[0x100..0x110].copy_from_slice(&[0xEE; 16]);

    // BD reads from East window at offset 0x100, length 16 bytes (4 words).
    // 0x100100 = East window (0x100000) + offset 0x100.
    let bd = BdConfig::simple_1d(0x100100, 16);
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(6, 0).unwrap();

    run_memtile_mm2s_to_completion(&mut engine, 6, &mut own_tile, None, Some(&mut east_tile), 500);

    // The four stream words must contain east_tile's 0xEE pattern.
    let expected_word = u32::from_le_bytes([0xEE; 4]);
    assert_eq!(engine.stream_out_len(), 4, "expected 4 stream words from east tile");
    while let Some(w) = engine.pop_stream_out() {
        assert_eq!(
            w.data, expected_word,
            "stream word should carry east-neighbour byte pattern (0xEE), got 0x{:08X}",
            w.data
        );
    }
}

#[test]
fn test_memtile_mm2s_reads_from_west_neighbor() {
    // Symmetric to the East test: BD with base_addr in the West window
    // should read from the west tile's memory.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);
    let mut west_tile = Tile::mem_tile(0, 1);

    own_tile.data_memory_mut()[0x200..0x210].copy_from_slice(&[0x11; 16]);
    west_tile.data_memory_mut()[0x200..0x210].copy_from_slice(&[0x77; 16]);

    // BD reads from West window at offset 0x200, length 16 bytes.
    // 0x000200 = West window (0x000000) + offset 0x200.
    let bd = BdConfig::simple_1d(0x000200, 16);
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(6, 0).unwrap();

    run_memtile_mm2s_to_completion(&mut engine, 6, &mut own_tile, Some(&mut west_tile), None, 500);

    let expected_word = u32::from_le_bytes([0x77; 4]);
    assert_eq!(engine.stream_out_len(), 4, "expected 4 stream words from west tile");
    while let Some(w) = engine.pop_stream_out() {
        assert_eq!(
            w.data, expected_word,
            "stream word should carry west-neighbour byte pattern (0x77), got 0x{:08X}",
            w.data
        );
    }
}

#[test]
fn test_memtile_s2mm_writes_to_east_neighbor() {
    // S2MM with a BD in the East window should land bytes in the east
    // neighbour's data memory, leaving the local tile untouched.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);
    let mut east_tile = Tile::mem_tile(2, 1);

    // Pre-fill both tiles with sentinels so a wrong write target is obvious.
    own_tile.data_memory_mut()[0x300..0x310].copy_from_slice(&[0x55; 16]);
    east_tile.data_memory_mut()[0x300..0x310].copy_from_slice(&[0x00; 16]);

    // Push 4 stream words for a 16-byte S2MM transfer on channel 0.
    let payload_word = u32::from_le_bytes([0xC3; 4]);
    for i in 0..4 {
        engine.push_stream_in(StreamData { data: payload_word, tlast: i == 3, channel: 0 });
    }

    // BD writes into the East window at offset 0x300, length 16.
    // 0x100300 = East window (0x100000) + offset 0x300.
    let bd = BdConfig::simple_1d(0x100300, 16);
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(0, 0).unwrap();

    run_memtile_mm2s_to_completion(&mut engine, 0, &mut own_tile, None, Some(&mut east_tile), 500);

    // East tile should now hold the payload at offset 0x300.
    let east_data = &east_tile.data_memory()[0x300..0x310];
    assert!(
        east_data.iter().all(|&b| b == 0xC3),
        "east neighbour memory should hold S2MM payload (0xC3), got {:?}",
        east_data
    );

    // Own tile must remain untouched at the same offset.
    let own_data = &own_tile.data_memory()[0x300..0x310];
    assert!(
        own_data.iter().all(|&b| b == 0x55),
        "own tile memory should be unchanged (0x55), got {:?}",
        own_data
    );
}

#[test]
fn test_memtile_mm2s_missing_neighbour_falls_back_to_own() {
    // mlir-aie's `dma_configure_task` lowering on AIE2 emits *flat* MemTile
    // addresses (no window offset added), so a BD with addr=0x100100 on a
    // last-column MemTile that has no east tile is intended as Own[0x100]
    // (the address falls in the "East" window only because the toolchain
    // never adds the windowing offset for that path -- see
    // mlir-aie's AIEDMATasksToNPU.cpp comment for AIE2 vs AIE2P).
    //
    // Real hardware silently aliases such accesses to Own when the
    // addressed neighbour does not exist. We mirror that: no fatal error,
    // and the read returns local data at the same byte offset.
    let mut engine = DmaEngine::new_mem_tile(3, 1); // last col, no east
    let mut own_tile = Tile::mem_tile(3, 1);
    own_tile.data_memory_mut()[0x100..0x110].copy_from_slice(&[0x42; 16]);

    let bd = BdConfig::simple_1d(0x100100, 16); // East-window addr, no east tile
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(6, 0).unwrap();

    run_memtile_mm2s_to_completion(&mut engine, 6, &mut own_tile, None, None, 500);

    assert!(
        engine.fatal_errors.is_empty(),
        "missing East neighbour should fall back to Own, not fatal-error: {:?}",
        engine.fatal_errors
    );
    let expected_word = u32::from_le_bytes([0x42; 4]);
    assert_eq!(engine.stream_out_len(), 4, "expected 4 stream words from own tile");
    while let Some(w) = engine.pop_stream_out() {
        assert_eq!(
            w.data, expected_word,
            "stream word should fall back to own-tile data (0x42), got 0x{:08X}",
            w.data
        );
    }
}

#[test]
fn test_memtile_mm2s_out_of_window_addr_records_fatal_error() {
    // Addresses beyond 3*mem_size have no defined window and remain a
    // fatal error -- this guards against actual programming bugs (vs the
    // benign "windowed addr without neighbour" case above).
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own_tile = Tile::mem_tile(1, 1);

    // 0x180000 = first byte beyond East window (3 * 0x80000).
    let bd = BdConfig::simple_1d(0x180000, 16);
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(6, 0).unwrap();

    let mut neighbors = NeighborTiles::empty();
    let mut host_mem = make_host_memory();
    for _ in 0..500 {
        engine.step(&mut own_tile, &mut neighbors, &mut host_mem);
        if matches!(engine.channel_state(6), ChannelState::Error) {
            break;
        }
        if !engine.channel_active(6) {
            break;
        }
    }

    assert!(!engine.fatal_errors.is_empty(), "addr beyond 3*mem_size should fatal-error, got none");
    assert!(
        engine.fatal_errors.iter().any(|e| e.contains("outside three-window")),
        "fatal_errors should mention out-of-window, got: {:?}",
        engine.fatal_errors,
    );
}

#[test]
fn test_resolve_lock_id_memtile_boundary_values() {
    // Exhaustive boundary test for all three regions of the 192-entry space.
    let tile_kind = TileKind::Mem;
    let num_locks: u8 = 64;

    // Region boundaries: 0, 63, 64, 127, 128, 191, 192
    // West region: [0, 64)
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 0),
        Some(LockTarget::West(0)),
        "lock_id=0 -> West(0)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 32),
        Some(LockTarget::West(32)),
        "lock_id=32 -> West(32) (mid-range)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 63),
        Some(LockTarget::West(63)),
        "lock_id=63 -> West(63) (last in West region)"
    );

    // Own region: [64, 128)
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 64),
        Some(LockTarget::Own(0)),
        "lock_id=64 -> Own(0) (first in Own region)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 96),
        Some(LockTarget::Own(32)),
        "lock_id=96 -> Own(32) (mid-range)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 127),
        Some(LockTarget::Own(63)),
        "lock_id=127 -> Own(63) (last in Own region)"
    );

    // East region: [128, 192)
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 128),
        Some(LockTarget::East(0)),
        "lock_id=128 -> East(0) (first in East region)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 160),
        Some(LockTarget::East(32)),
        "lock_id=160 -> East(32) (mid-range)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 191),
        Some(LockTarget::East(63)),
        "lock_id=191 -> East(63) (last in East region)"
    );

    // Out of range
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 192),
        None,
        "lock_id=192 -> None (out of range)"
    );
    assert_eq!(
        DmaEngine::resolve_lock_id_static(tile_kind, 2, 1, num_locks, 255),
        None,
        "lock_id=255 -> None (out of range, max u8)"
    );
}

#[test]
fn test_resolve_lock_id_shim_passthrough() {
    // Shim tiles use a small lock ID field, always maps to Own.
    let tile_kind = TileKind::ShimNoc;
    let num_locks = 16;
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 0, 0, num_locks, 0), Some(LockTarget::Own(0)));
    assert_eq!(DmaEngine::resolve_lock_id_static(tile_kind, 0, 0, num_locks, 15), Some(LockTarget::Own(15)));
}

#[test]
fn test_stream_in_per_channel_isolation() {
    // Bug: shared stream_in buffer lets one S2MM channel's data block another.
    // In real hardware, each S2MM channel has its own input FIFO connected to
    // a dedicated stream switch master port. One channel flooding its FIFO
    // must not prevent another channel from receiving data.
    let mut engine = DmaEngine::new_shim_tile(0, 0);

    // Fill stream_in with channel 1 (trace) data up to capacity.
    // With a shared buffer of 256 entries, this would block channel 0.
    for i in 0..256 {
        let pushed =
            engine.push_stream_in(StreamData { data: 0xFEED_0000 | i as u32, tlast: false, channel: 1 });
        assert!(pushed, "channel 1 push {} should succeed", i);
    }

    // Channel 0 (output) must still be able to receive data.
    // On real hardware, channel 0's FIFO is independent of channel 1's.
    let pushed = engine.push_stream_in(StreamData { data: 0x0010_0001, tlast: false, channel: 0 });
    assert!(pushed, "channel 0 push must succeed even when channel 1 is full");

    // And channel 0 data must be readable
    assert!(engine.has_stream_in_for_channel(0));
}

#[test]
fn test_memtile_mm2s_packet_header_insertion() {
    // MemTile MM2S BD with enable_packet=true should create a Transfer
    // that can generate a packet header.
    //
    // This verifies the data path from BD config to Transfer to packet
    // header, which is the path that the packet_flow test depends on.
    use crate::device::dma::transfer::{Transfer, TransferDirection};
    use xdna_archspec::types::TileKind;

    let mut bd = BdConfig::default();
    bd.valid = true;
    bd.base_addr = 0x80000;
    bd.length = 16;
    bd.enable_packet = true;
    bd.packet_id = 5;
    bd.packet_type = 0;
    bd.d0.size = 4;
    bd.d0.stride = 4;

    // Create a Transfer from this BD (as MemTile MM2S)
    let transfer = Transfer::new(&bd, 1, 6, TransferDirection::MM2S, 0, 1, TileKind::Mem)
        .expect("should create transfer");

    // The transfer must carry the packet config from the BD
    assert!(transfer.enable_packet, "Transfer.enable_packet must be true when BD has it");
    assert_eq!(transfer.packet_id, 5, "Transfer.packet_id must match BD");
    assert!(transfer.needs_packet_header(), "needs_packet_header() should be true before sending");

    // Generate the header
    let header_word = transfer.generate_packet_header().expect("should generate packet header");
    let (hdr, _) = crate::device::stream_switch::PacketHeader::decode(header_word);
    assert_eq!(hdr.stream_id, 5, "header stream_id should match BD packet_id");
}

#[test]
fn test_memtile_mm2s_engine_inserts_header() {
    // End-to-end: MemTile DMA engine with enable_packet BD should
    // insert a packet header into stream_out during BdSetup.
    let mut engine = DmaEngine::new_mem_tile(0, 1);

    let mut bd = BdConfig::default();
    bd.valid = true;
    bd.base_addr = 0;
    bd.length = 16;
    bd.enable_packet = true;
    bd.packet_id = 5;
    bd.d0.size = 4;
    bd.d0.stride = 4;

    engine.configure_bd(1, bd.clone()).unwrap();

    // Verify the BD is stored with enable_packet
    let stored = engine.get_bd(1).unwrap();
    assert!(stored.enable_packet, "stored BD must have enable_packet=true");
    assert_eq!(stored.packet_id, 5, "stored BD must have packet_id=5");

    // Start MM2S channel 0 (index 6 for MemTile)
    let mm2s_ch = engine.s2mm_channel_count() as u8;
    assert_eq!(mm2s_ch, 6, "MemTile S2MM count should be 6");
    engine.start_channel(mm2s_ch, 1).unwrap();

    // After start_channel, the FSM should be in BdSetup or AcquiringLock.
    // BdSetup inserts the packet header on completion.
    // Since our BD has no acquire_lock, the path is:
    //   start_channel -> BdSetup{transfer} -> (step) -> skip AcquiringLock -> MemoryLatency
    // The header is inserted at the BdSetup->next transition.

    // Peek at the FSM to verify the transfer has enable_packet
    let ch = &engine.channels[mm2s_ch as usize];
    match &ch.fsm {
        ChannelFsm::BdSetup { transfer, .. } => {
            assert!(
                transfer.enable_packet,
                "Transfer in BdSetup must have enable_packet=true, got false \
                 (BD enable_packet={}, packet_id={})",
                bd.enable_packet, bd.packet_id
            );
        }
        other => panic!("Expected BdSetup, got {:?}", std::mem::discriminant(other)),
    }
}

// ---------------------------------------------------------------
// MemTile BD-channel validity tests
// ---------------------------------------------------------------

#[test]
fn test_memtile_bd_channel_validity_even_channels_low_bds() {
    // BDs 0-23 are valid only for even per-direction channels (0, 2, 4).
    let engine = DmaEngine::new_mem_tile(0, 1);

    // Even S2MM channels (flat 0, 2, 4) with low BDs -> valid
    assert!(engine.check_memtile_bd_channel_validity(0, 0));
    assert!(engine.check_memtile_bd_channel_validity(10, 2));
    assert!(engine.check_memtile_bd_channel_validity(23, 4));

    // Even MM2S channels (flat 6, 8, 10 -> per-dir 0, 2, 4) with low BDs -> valid
    assert!(engine.check_memtile_bd_channel_validity(0, 6));
    assert!(engine.check_memtile_bd_channel_validity(15, 8));
    assert!(engine.check_memtile_bd_channel_validity(23, 10));
}

#[test]
fn test_memtile_bd_channel_validity_odd_channels_high_bds() {
    // BDs 24-47 are valid only for odd per-direction channels (1, 3, 5).
    let engine = DmaEngine::new_mem_tile(0, 1);

    // Odd S2MM channels (flat 1, 3, 5) with high BDs -> valid
    assert!(engine.check_memtile_bd_channel_validity(24, 1));
    assert!(engine.check_memtile_bd_channel_validity(35, 3));
    assert!(engine.check_memtile_bd_channel_validity(47, 5));

    // Odd MM2S channels (flat 7, 9, 11 -> per-dir 1, 3, 5) with high BDs -> valid
    assert!(engine.check_memtile_bd_channel_validity(24, 7));
    assert!(engine.check_memtile_bd_channel_validity(36, 9));
    assert!(engine.check_memtile_bd_channel_validity(47, 11));
}

#[test]
fn test_memtile_bd_channel_validity_invalid_combinations() {
    // Even channel + high BD -> invalid
    // Odd channel + low BD -> invalid
    let engine = DmaEngine::new_mem_tile(0, 1);

    // Even S2MM channel with high BD -> invalid
    assert!(!engine.check_memtile_bd_channel_validity(24, 0));
    assert!(!engine.check_memtile_bd_channel_validity(47, 2));

    // Odd S2MM channel with low BD -> invalid
    assert!(!engine.check_memtile_bd_channel_validity(0, 1));
    assert!(!engine.check_memtile_bd_channel_validity(23, 3));

    // Even MM2S channel with high BD -> invalid
    assert!(!engine.check_memtile_bd_channel_validity(30, 6));
    assert!(!engine.check_memtile_bd_channel_validity(47, 8));

    // Odd MM2S channel with low BD -> invalid
    assert!(!engine.check_memtile_bd_channel_validity(0, 7));
    assert!(!engine.check_memtile_bd_channel_validity(10, 9));
}

#[test]
fn test_memtile_bd_channel_validity_non_memtile_always_valid() {
    // Compute and shim tiles should always pass (no BD-channel constraint).
    let compute = DmaEngine::new_compute_tile(1, 2);
    assert!(compute.check_memtile_bd_channel_validity(0, 0));
    assert!(compute.check_memtile_bd_channel_validity(15, 1));

    let shim = DmaEngine::new_shim_tile(0, 0);
    assert!(shim.check_memtile_bd_channel_validity(0, 0));
    assert!(shim.check_memtile_bd_channel_validity(15, 1));
}

#[test]
fn test_memtile_bd_channel_validity_boundary_bd23_bd24() {
    // BD 23 is the last "low" BD, BD 24 is the first "high" BD.
    let engine = DmaEngine::new_mem_tile(0, 1);

    // BD 23, even channel -> valid
    assert!(engine.check_memtile_bd_channel_validity(23, 0));
    // BD 23, odd channel -> invalid
    assert!(!engine.check_memtile_bd_channel_validity(23, 1));

    // BD 24, odd channel -> valid
    assert!(engine.check_memtile_bd_channel_validity(24, 1));
    // BD 24, even channel -> invalid
    assert!(!engine.check_memtile_bd_channel_validity(24, 0));
}

#[test]
fn test_per_direction_channel() {
    let engine = DmaEngine::new_mem_tile(0, 1);

    // S2MM channels: flat index IS the per-direction index
    assert_eq!(engine.per_direction_channel(0), 0);
    assert_eq!(engine.per_direction_channel(3), 3);
    assert_eq!(engine.per_direction_channel(5), 5);

    // MM2S channels: flat index 6..11 -> per-direction 0..5
    assert_eq!(engine.per_direction_channel(6), 0);
    assert_eq!(engine.per_direction_channel(7), 1);
    assert_eq!(engine.per_direction_channel(11), 5);
}

/// Chained BDs with acquire+release locks on a private lock pair finish
/// back-to-back with `interval == data_cycles` (no dead cycles between
/// FINISHED_BD events). Matches HW behavior via three non-speculative
/// pipelining optimizations:
/// 1. Inline release on the last data cycle (bypass ReleasingLock).
/// 2. Inline grant transition (collapse AcquiringLock{acq=true,cr=0}).
/// 3. Inline first data cycle on the grant (run do_transfer_cycle in
///    the same step the arbiter grants).
///
/// For 2-word BDs (8 bytes, 1 Transferring cycle each) the FINISHED_BD
/// interval is 1 (data only). Earlier EMU paid 4 cycles: 1 data + 1
/// ReleasingLock + 1 grant + 1 acquired=true cooldown. See finding doc
/// `2026-05-11-emu-chained-bd-spec-acquire-attempt.md` for the failed
/// speculative attempts and the eventual non-speculative fix.
#[test]
fn chained_bd_lock_interval_baseline() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // 2-word BDs in a double-buffer ping-pong on locks 0/1.
    engine
        .configure_bd(0, BdConfig::simple_1d(0x100, 8).with_acquire(0, 1).with_release(1, 1).with_next(1))
        .unwrap();
    engine
        .configure_bd(1, BdConfig::simple_1d(0x200, 8).with_acquire(1, 1).with_release(0, 1))
        .unwrap();

    // BD#0 acquires lock 0 (must start =1); releases lock 1 to signal BD#1.
    // BD#1 acquires lock 1 (initially 0, set by BD#0's release).
    tile.locks[0].set(1);
    tile.locks[1].set(0);
    engine.start_channel(2, 0).unwrap();

    let mut cycle: u64 = 0;
    let mut all_events: Vec<(u64, EventType)> = Vec::new();
    while engine.channel_active(2) {
        engine.set_current_cycle(cycle);
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        all_events.extend(engine.drain_trace_events());
        cycle += 1;
        if cycle > 50 {
            panic!("chain stalled beyond expected cycle budget");
        }
    }

    let finished_bd_cycles: Vec<u64> = all_events
        .iter()
        .filter_map(|(c, e)| match e {
            EventType::DmaFinishedBd { channel: 2 } => Some(*c),
            _ => None,
        })
        .collect();
    assert_eq!(
        finished_bd_cycles.len(),
        2,
        "expected exactly 2 FINISHED_BD events, got cycles {:?}",
        finished_bd_cycles
    );
    let interval = finished_bd_cycles[1] - finished_bd_cycles[0];
    assert_eq!(
        interval, 1,
        "chained-BD FINISHED_BD interval == data cycles (no dead cycles); \
         matches HW. cycles={:?}",
        finished_bd_cycles
    );
}

/// 4-BD chain of 16-word locked BDs. With all of #26's pipelining
/// optimizations -- inline release on last data cycle, inline grant
/// transition, AND inline first data cycle on grant -- the chained
/// interval matches data cycles exactly (`16w / 4wpc = 4`).
#[test]
fn chained_bd_16w_lock_interval_diagnostic() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // 4 chained 16-word (64-byte) BDs ping-ponging on locks 0/1.
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(0x100, 64)
                .with_acquire(0, 1)
                .with_release(1, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(
            1,
            BdConfig::simple_1d(0x200, 64)
                .with_acquire(1, 1)
                .with_release(0, 1)
                .with_next(2),
        )
        .unwrap();
    engine
        .configure_bd(
            2,
            BdConfig::simple_1d(0x300, 64)
                .with_acquire(0, 1)
                .with_release(1, 1)
                .with_next(3),
        )
        .unwrap();
    engine
        .configure_bd(3, BdConfig::simple_1d(0x400, 64).with_acquire(1, 1).with_release(0, 1))
        .unwrap();

    // with_acquire(N, 1) is acq_eq: lock must == 1, then decrement.
    // Ping-pong: BD#0/#2 acquire lock 0 (which BD#1/#3 just released back).
    tile.locks[0].set(1);
    tile.locks[1].set(0);
    engine.start_channel(2, 0).unwrap();

    let mut cycle: u64 = 0;
    let mut all_events: Vec<(u64, EventType)> = Vec::new();
    while engine.channel_active(2) {
        engine.set_current_cycle(cycle);
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        all_events.extend(engine.drain_trace_events());
        cycle += 1;
        if cycle > 100 {
            panic!("chain stalled beyond expected cycle budget");
        }
    }

    let finished_bd_cycles: Vec<u64> = all_events
        .iter()
        .filter_map(|(c, e)| match e {
            EventType::DmaFinishedBd { channel: 2 } => Some(*c),
            _ => None,
        })
        .collect();
    let intervals: Vec<u64> = finished_bd_cycles.windows(2).map(|w| w[1] - w[0]).collect();
    assert_eq!(
        intervals,
        vec![4u64, 4, 4],
        "16w chained-BD intervals match data cycles; FINISHED_BD cycles={:?}",
        finished_bd_cycles
    );
}

/// Diagnostic: chained BDs *without* locks. Pure chain overhead, no
/// AcquiringLock state on the critical path. Reveals whether the +1
/// cyc residual on locked chains is from the lock dance specifically
/// or from a generic chain-transition cost.
#[test]
fn chained_bd_no_lock_interval_diagnostic() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // 4 chained 16-word BDs, no locks.
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 64).with_next(1)).unwrap();
    engine.configure_bd(1, BdConfig::simple_1d(0x200, 64).with_next(2)).unwrap();
    engine.configure_bd(2, BdConfig::simple_1d(0x300, 64).with_next(3)).unwrap();
    engine.configure_bd(3, BdConfig::simple_1d(0x400, 64)).unwrap();

    engine.start_channel(2, 0).unwrap();

    let mut cycle: u64 = 0;
    let mut all_events: Vec<(u64, EventType)> = Vec::new();
    while engine.channel_active(2) {
        engine.set_current_cycle(cycle);
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        all_events.extend(engine.drain_trace_events());
        cycle += 1;
        if cycle > 100 {
            panic!("chain stalled beyond expected cycle budget");
        }
    }

    let finished_bd_cycles: Vec<u64> = all_events
        .iter()
        .filter_map(|(c, e)| match e {
            EventType::DmaFinishedBd { channel: 2 } => Some(*c),
            _ => None,
        })
        .collect();
    let intervals: Vec<u64> = finished_bd_cycles.windows(2).map(|w| w[1] - w[0]).collect();
    // 16w BD with wpc=4 = 4 data cycles. Without locks, enter_chained_bd
    // returns Transferring directly (no AcquiringLock). The Transferring
    // state's match arm runs do_transfer_cycle. With no transition
    // overhead, interval = data cycles = 4.
    assert_eq!(
        intervals,
        vec![4u64, 4, 4],
        "no-lock chained-BD intervals; FINISHED_BD cycles={:?}",
        finished_bd_cycles
    );
}

#[test]
fn memtile_invalid_bd_channel_combination_returns_error() {
    // BD 24 is the first "odd channel" BD; per-direction channel 0 (S2MM ch 0)
    // is even, so BD 24 + ch 0 should be rejected.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    // Configure BD 24 to something valid otherwise.
    let bd = BdConfig::simple_1d(0x1000, 16);
    engine.configure_bd(24, bd).unwrap();

    let result = engine.start_channel_with_repeat(0, 24, 0);
    assert!(matches!(result, Err(DmaError::InvalidBd(24))), "expected InvalidBd(24), got {:?}", result);
}

// ---------------------------------------------------------------------------
// DMA watchpoint hookup (task #68)
//
// AM025 watchpoint slots include AXI_Access / DMA_Access / quadrant filter
// bits; #65 modelled the filter semantics, #68 wires the DMA engine to fire
// the slot with `AccessOrigin::Dma` so a configured slot actually catches
// DMA traffic. We probe firing via mem_perf_counters: a counter armed to
// start on WATCHPOINT_N transitions to active iff the event is delivered.
// ---------------------------------------------------------------------------

/// Compute-tile WatchPoint register layout (mirrors the helper in
/// cycle_accurate.rs tests). `dma` sets the DMA_Access filter bit (28).
fn wp_compute(read: bool, write: bool, addr: u32, dma: bool, axi: bool) -> u32 {
    let dir = ((read as u32) << 31) | ((write as u32) << 30);
    let strobes = 0xFu32 << 20;
    let mut v = dir | strobes | (addr & 0xFFF0);
    if dma {
        v |= 1 << 28;
    }
    if axi {
        v |= 1 << 29;
    }
    v
}

fn wp_memtile(read: bool, write: bool, addr: u32, dma: bool, axi: bool) -> u32 {
    let dir = ((read as u32) << 29) | ((write as u32) << 28);
    let strobes = 0xFu32 << 20;
    let mut v = dir | strobes | (addr & 0x7FFF0);
    if dma {
        v |= 1 << 26;
    }
    if axi {
        v |= 1 << 27;
    }
    v
}

#[test]
fn test_dma_s2mm_fires_watchpoint_on_own_tile() {
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    // Wildcard slot 0 watching writes at offset 0x100.
    tile.registers.insert(0x14100, wp_compute(false, true, 0x100, false, false));
    // Counter 0 starts on WATCHPOINT_0; we verify by checking is_active().
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);
    assert!(!tile.mem_perf_counters.is_active(0), "counter must start idle");

    // Push one stream word and drive a S2MM that writes 4 bytes at 0x100.
    engine.push_stream_in(StreamData { data: 0xCAFEBABE, tlast: true, channel: 0 });
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(tile.mem_perf_counters.is_active(0), "S2MM write at the watched address must fire WATCHPOINT_0");
}

#[test]
fn test_dma_mm2s_fires_watchpoint_on_own_tile() {
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.write_data_u32(0x200, 0xFEEDF00D);
    // Wildcard slot 0 watching reads at 0x200.
    tile.registers.insert(0x14100, wp_compute(true, false, 0x200, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    let ok = engine.transfer_mm2s(0x200, 4, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(ok);

    assert!(tile.mem_perf_counters.is_active(0), "MM2S read at the watched address must fire WATCHPOINT_0");
}

#[test]
fn test_dma_filter_excludes_axi_only() {
    // Slot configured with ONLY the AXI filter bit must NOT fire on a DMA
    // access -- the origin filter is non-zero, so the access origin must
    // match one of the enabled bits, and DMA isn't enabled.
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.registers
        .insert(0x14100, wp_compute(false, true, 0x100, /*dma=*/ false, /*axi=*/ true));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(
        !tile.mem_perf_counters.is_active(0),
        "AXI-only filter must exclude DMA accesses; counter should stay idle"
    );
}

#[test]
fn test_dma_filter_includes_dma_only() {
    // Slot configured with ONLY the DMA filter bit must fire on a DMA
    // access (and NOT on a Core access -- but the DMA engine only emits
    // Dma origins, so we just check that the firing happens).
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.registers
        .insert(0x14100, wp_compute(false, true, 0x100, /*dma=*/ true, /*axi=*/ false));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(tile.mem_perf_counters.is_active(0), "DMA-only filter must include DMA writes");
}

#[test]
fn test_dma_mm2s_multi_word_fires_per_word() {
    // A 16-byte MM2S read crosses 4 words within a single 16-byte comparator
    // block; the slot fires per word, so we'd expect at least one firing
    // (counter activation suffices to verify the wiring).
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    for i in 0..4usize {
        tile.write_data_u32(0x300 + i * 4, 0x1000_0000 | i as u32);
    }
    tile.registers.insert(0x14100, wp_compute(true, false, 0x300, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    let ok = engine.transfer_mm2s(0x300, 16, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(ok);

    assert!(tile.mem_perf_counters.is_active(0), "multi-word MM2S must fire WATCHPOINT_0 at least once");
}

#[test]
fn test_dma_wrong_direction_does_not_fire() {
    // Slot configured for Read only; a DMA write must not match.
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.registers
        .insert(0x14100, wp_compute(/*read=*/ true, /*write=*/ false, 0x100, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(!tile.mem_perf_counters.is_active(0), "read-only slot must ignore DMA writes");
}

#[test]
fn test_dma_address_mismatch_does_not_fire() {
    // Slot at 0x100; DMA writes 0x200 -- no match.
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.registers.insert(0x14100, wp_compute(false, true, 0x100, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    let result = engine.transfer_s2mm(0x200, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(!tile.mem_perf_counters.is_active(0), "address mismatch must leave WATCHPOINT_0 unfired");
}

#[test]
fn test_dma_compressed_mm2s_fires_watchpoint() {
    // Compressed MM2S reads tile memory through the same bank interface;
    // the comparator should still see the source address range.
    use xdna_archspec::aie2::trace_events::mem_events;

    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();

    tile.data_memory_mut()[0x100] = 0x42;
    tile.data_memory_mut()[0x108] = 0x84;
    tile.registers.insert(0x14100, wp_compute(true, false, 0x100, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(mem_events::WATCHPOINT_0 as u32, 0, 1, 7);

    engine.set_channel_compression_config(2, true, false, false);
    let ok = engine.transfer_mm2s(0x100, 32, 2, true, false, &mut tile, &mut NeighborTiles::empty());
    assert!(ok);

    assert!(
        tile.mem_perf_counters.is_active(0),
        "compressed MM2S must still fire WATCHPOINT_0 (comparator is independent of compression)"
    );
}

#[test]
fn test_memtile_dma_own_window_fires_watchpoint() {
    // MemTile WatchPoint 2 watching writes at offset 0x100; DMA writes to
    // address 0x80100 (Own window, offset 0x100).
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);

    tile.registers.insert(0x94108, wp_memtile(false, true, 0x100, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0xABAB, tlast: true, channel: 0 });
    let result = engine.transfer_s2mm(0x80100, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(tile.mem_perf_counters.is_active(0), "memtile Own-window S2MM must fire WATCHPOINT_2");
}

#[test]
fn test_memtile_dma_neighbor_window_does_not_fire_own_watchpoint() {
    // Cross-tile (West/East) DMA fires on the *target* tile, not the source.
    // A write through the West window addresses the col-1 neighbour, so the
    // SOURCE tile's WatchPoint must remain idle even when its slot would
    // otherwise match. Companion tests cover the target-side firing.
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut west = Tile::mem_tile(0, 1);

    // Watch writes at offset 0x100 on the SOURCE tile. The West neighbour is
    // the actual target, so the source must stay idle.
    tile.registers.insert(0x94108, wp_memtile(false, true, 0x100, false, false));
    tile.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0xABAB, tlast: true, channel: 0 });
    let mut neighbors = NeighborTiles { west: Some(&mut west), east: None };
    // Address 0x100 is in the West window (window 0 = West for MemTile).
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut neighbors);
    assert!(result.success);

    assert!(
        !tile.mem_perf_counters.is_active(0),
        "cross-tile DMA must not fire the SOURCE tile's watchpoint"
    );
}

// ---------------------------------------------------------------------------
// Cross-tile DMA quadrant detection (task #69)
//
// MemTile DMAs that address the West/East windows reach the col-1/col+1
// MemTile through the dedicated MemTile-to-MemTile shared bus. The HW
// comparator on the *target* tile sees those accesses with the AM025
// East/West quadrant filter bits ([25:24]) so software can distinguish
// "DMA from my own tile" from "DMA from my neighbour". Following the
// AccessOrigin scaffolding from #65 and the local-DMA hookup from #68,
// the four DMA paths now fire on the resolved target with
// `Neighbour(<direction-from-target>)` for cross-tile writes/reads.
// ---------------------------------------------------------------------------

#[test]
fn test_memtile_dma_west_window_fires_west_neighbor_watchpoint() {
    // S2MM through the West window (col-1 target) must fire the WEST
    // tile's WatchPoint with `Neighbour(East)` (we are east of the target).
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut west = Tile::mem_tile(0, 1);

    // Slot on the WEST tile watches writes at offset 0x100 with no origin
    // filter (wildcard) -- it should match the cross-tile DMA write.
    west.registers.insert(0x94108, wp_memtile(false, true, 0x100, false, false));
    west.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0xC0FFEE, tlast: true, channel: 0 });
    let mut neighbors = NeighborTiles { west: Some(&mut west), east: None };
    // Address 0x100 -> West window (window 0), offset 0x100 in west.data_memory.
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut neighbors);
    assert!(result.success);

    assert!(
        west.mem_perf_counters.is_active(0),
        "cross-tile S2MM through West window must fire the WEST neighbour's watchpoint"
    );
}

#[test]
fn test_memtile_dma_east_window_fires_east_neighbor_watchpoint() {
    // S2MM through the East window (col+1 target) must fire the EAST
    // tile's WatchPoint with `Neighbour(West)` (we are west of the target).
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut east = Tile::mem_tile(2, 1);

    east.registers.insert(0x94108, wp_memtile(false, true, 0x100, false, false));
    east.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0xDEC0DE, tlast: true, channel: 0 });
    let mut neighbors = NeighborTiles { west: None, east: Some(&mut east) };
    // Address 0x100100 -> East window (window 2 * 0x80000 = 0x100000), offset 0x100.
    let result = engine.transfer_s2mm(0x100100, 4, 0, &mut tile, &mut neighbors);
    assert!(result.success);

    assert!(
        east.mem_perf_counters.is_active(0),
        "cross-tile S2MM through East window must fire the EAST neighbour's watchpoint"
    );
}

#[test]
fn test_memtile_dma_west_window_mm2s_fires_west_neighbor_watchpoint() {
    // Symmetric MM2S check: read through the West window must fire the
    // WEST tile's WatchPoint with the same `Neighbour(East)` origin.
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut west = Tile::mem_tile(0, 1);

    // Seed source bytes on the West tile (the read target).
    west.write_data_u32(0x100, 0xBEEFCAFE);
    west.registers.insert(0x94108, wp_memtile(true, false, 0x100, false, false));
    west.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    let mut neighbors = NeighborTiles { west: Some(&mut west), east: None };
    let ok = engine.transfer_mm2s(0x100, 4, 0, true, false, &mut tile, &mut neighbors);
    assert!(ok);

    assert!(
        west.mem_perf_counters.is_active(0),
        "cross-tile MM2S through West window must fire the WEST neighbour's watchpoint"
    );
}

#[test]
fn test_memtile_dma_neighbor_filter_dma_only_does_not_fire() {
    // A target slot with ONLY the DMA filter bit must NOT fire on a
    // cross-tile access -- the origin is `Neighbour(East)`, not `Dma`,
    // so the AM025 filter mask excludes it.
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut west = Tile::mem_tile(0, 1);

    // West tile's slot 2 watches writes at 0x100 with DMA filter only.
    west.registers
        .insert(0x94108, wp_memtile(false, true, 0x100, /*dma=*/ true, /*axi=*/ false));
    west.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    let mut neighbors = NeighborTiles { west: Some(&mut west), east: None };
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut neighbors);
    assert!(result.success);

    assert!(
        !west.mem_perf_counters.is_active(0),
        "DMA-only filter on the target must reject cross-tile origins"
    );
}

#[test]
fn test_memtile_dma_neighbor_filter_east_only_fires() {
    // Target slot with ONLY the East quadrant filter bit (bit 25) must
    // fire on a cross-tile access through the West window: we're east of
    // that target, so the origin matches.
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut west = Tile::mem_tile(0, 1);

    // wp_memtile only exposes dma/axi convenience bits; build the East
    // quadrant filter (bit 25) directly.
    let east_only = wp_memtile(false, true, 0x100, false, false) | (1u32 << 25);
    west.registers.insert(0x94108, east_only);
    west.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    let mut neighbors = NeighborTiles { west: Some(&mut west), east: None };
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut neighbors);
    assert!(result.success);

    assert!(
        west.mem_perf_counters.is_active(0),
        "East-quadrant filter on the West target must accept the cross-tile origin"
    );
}

#[test]
fn test_memtile_dma_missing_neighbor_falls_back_to_own_dma_origin() {
    // When the addressed neighbour does not exist, resolve_s2mm_target
    // falls back to the local tile (silent aliasing per the resolver
    // doc). The fallback target is `Own`, so the firing must use the
    // local `Dma` origin -- not a Neighbour origin against a slot that
    // would expect a different filter.
    use xdna_archspec::aie2::trace_events::memtile_events;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);

    // OWN tile slot: DMA filter only. If the fallback used a Neighbour
    // origin, this would NOT fire; with the correct Dma-on-Own origin,
    // the slot matches and the counter activates.
    tile.registers
        .insert(0x94108, wp_memtile(false, true, 0x100, /*dma=*/ true, /*axi=*/ false));
    tile.mem_perf_counters
        .write_control_start_stop(memtile_events::WATCHPOINT_2 as u32, 0, 1, 7);

    engine.push_stream_in(StreamData { data: 0x1, tlast: true, channel: 0 });
    // Address 0x100 -> West window, but no west neighbour -- falls back to Own.
    let result = engine.transfer_s2mm(0x100, 4, 0, &mut tile, &mut NeighborTiles::empty());
    assert!(result.success);

    assert!(
        tile.mem_perf_counters.is_active(0),
        "missing neighbour falls back to Own with Dma origin (slot armed for Dma must fire)"
    );
}
