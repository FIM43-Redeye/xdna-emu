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
fn test_direct_lock_release_emits_trace_event() {
    // GAP A (tenant-4 grant-order spike): the pipelined BD-completion release
    // path (`apply_lock_release_direct`, used by `begin_completion`) must emit a
    // LockRelease trace event into the tile's mem_trace_pending, matching the
    // arbiter path (`Tile::resolve_lock_requests`). Before the fix the DMA
    // emitted lock ACQUIREs (arbiter path) but not RELEASEs (this inline path),
    // so memtile lock-release events were invisible to the trace while NPU1
    // silicon emits them -- breaking the emulator-vs-HW grant-order match.
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    tile.locks[5].set(1);
    let bd = BdConfig::simple_1d(0x100, 32)
        .with_acquire(5, 1) // lock==1 -> 0
        .with_release(5, 1); // +1 -> 1, via the inline completion path
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(2, 0).unwrap();

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

    // The inline release must have emitted a LockRelease trace event for lock 5.
    let released = tile
        .mem_trace_pending
        .iter()
        .any(|(_, e)| matches!(e, crate::interpreter::state::EventType::LockRelease { lock_id: 5 }));
    assert!(
        released,
        "direct BD-completion release must emit LockRelease{{lock_id:5}}; pending={:?}",
        tile.mem_trace_pending
    );
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

/// Shim DDR timing decomposes into three terms (chain-sweep calibration
/// 2026-05-25): channel_start_cycles (every task, all tiles),
/// shim_per_task_overhead_{mm2s,s2mm} (every task on shim+host), and
/// shim_ddr_cold_start_{mm2s,s2mm} (once per channel session, on shim+host).
///
/// On compute / non-host transfers the bonus is `channel_start_cycles` only.
/// On shim+host transfers the bonus includes the per-task overhead always,
/// plus the cold-start on the FIRST task of the session.
#[test]
fn test_shim_ddr_timing_decomposition() {
    let cfg = DmaTimingConfig::default();
    assert_eq!(cfg.channel_start_cycles, 2);
    // Recalibrated 2026-05-27 from N=50 multi-run K-sweep HW campaign
    // on _diag_shim_chain_sweep K={1,2,4,8}.  The previous 2026-05-25
    // K=8-only fit (498/249/0/168) under-modelled K=1 single-task by
    // ~50% in both directions; the multi-run data lets us calibrate
    // cold-start against K=1 totals and per-task against K=4+ steady-
    // state simultaneously.
    assert_eq!(cfg.shim_ddr_cold_start_mm2s_cycles, 1330);
    assert_eq!(cfg.shim_ddr_cold_start_s2mm_cycles, 341);
    assert_eq!(cfg.shim_per_task_overhead_mm2s_cycles, 325);
    assert_eq!(cfg.shim_per_task_overhead_s2mm_cycles, 179);
    assert_eq!(cfg.shim_words_per_cycle, 1);
    // Phase 2d warm-up transient decay ratios (per-mille).  MM2S decays the
    // cold-start across the chain (r~0.31); S2MM has no tail (one-shot).
    assert_eq!(cfg.shim_warmup_decay_mm2s_permille, 310);
    assert_eq!(cfg.shim_warmup_decay_s2mm_permille, 0);

    // stop_channel re-arms BOTH gates (channel reset == fresh boot).
    let mut engine = DmaEngine::new_shim_tile(0, 0);
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 16)).unwrap();
    engine.start_channel(0, 0).unwrap();
    engine.stop_channel(0).unwrap();
    assert!(engine.channels[0].is_first_bd, "stop_channel re-arms is_first_bd");
    assert_eq!(engine.channels[0].warm_task_index, 0, "stop_channel re-arms warm_task_index");
}

/// Phase 2d warm-up transient: on shim MM2S the cold-start cost is not a
/// one-shot -- HW per-task transfer durations decay geometrically across the
/// chain (1739/804/497/422/... at K=8).  The first-BD bonus on successive
/// tasks of a single channel session must follow
/// `channel_start + per_task + cold_start * (decay/1000)^i`.
///
/// Calibrated against the 2026-05-27 N=50 K=8 HW multi-run campaign.
#[test]
fn test_shim_mm2s_warmup_transient_decays_geometrically() {
    let mut engine = DmaEngine::new_shim_tile(0, 0);
    let bd = BdConfig::simple_1d(0x1000, 16);
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 0, 0, engine.tile_kind).unwrap();
    assert!(transfer.involves_host_memory(), "shim MM2S transfer touches host DDR");

    // Drive successive tasks within one channel session.  is_first_bd is
    // re-armed per task on Idle re-entry on real runs; emulate that here so
    // each call represents the first BD of a new task.
    let mut seq = Vec::new();
    for _ in 0..6 {
        engine.channels[0].is_first_bd = true;
        seq.push(engine.consume_first_bd_bonus(0, &transfer));
    }

    // channel_start(2) + per_task_mm2s(325) + cold_start(1330) * 0.310^i,
    // integer fixed-point (term = term * 310 / 1000 each task):
    //   cold terms: 1330, 412, 127, 39, 12, 3
    assert_eq!(seq, vec![1657, 739, 454, 366, 339, 330]);
}

/// S2MM warm-up decay is 0 on Phoenix: the cold-start fires once on task 0
/// and subsequent tasks pay only the per-task overhead.  Guards that the
/// Phase 2d geometric model leaves the S2MM direction's behavior unchanged.
#[test]
fn test_shim_s2mm_warmup_has_no_tail() {
    let mut engine = DmaEngine::new_shim_tile(0, 0);
    let bd = BdConfig::simple_1d(0x1000, 16);
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::S2MM, 0, 0, engine.tile_kind).unwrap();
    assert!(transfer.involves_host_memory(), "shim S2MM transfer touches host DDR");

    let mut seq = Vec::new();
    for _ in 0..3 {
        engine.channels[0].is_first_bd = true;
        seq.push(engine.consume_first_bd_bonus(0, &transfer));
    }

    // channel_start(2) + per_task_s2mm(179) + cold_start(341) only on task 0.
    assert_eq!(seq, vec![522, 181, 181]);
}

/// #140 layer-2: non-shim DMA channels (memtile / compute) carry a
/// one-time-per-channel-session pipeline-FILL cost on their first task -- the
/// "STARTING" phase the shim cold-start models for the shim, but which the
/// memtile/core channels need too (their deeper ObjectFifo pipeline fills over
/// ~1200cy on HW, which EMU otherwise collapses to ~55cy; #140).  Crucially it
/// is latched as a POST-transfer hold (`startup_hold_cycles` -> `StartupHold`)
/// rather than added to the pre-transfer MemoryLatency budget, so it delays
/// downstream first-output without backpressuring the shim MM2S.  The
/// MemoryLatency bonus stays channel_start-only.  Latched once per session
/// (`warm_task_index == 0`).
#[test]
fn test_memtile_first_bd_startup_latches_post_transfer_hold_once() {
    let mut engine = DmaEngine::new_mem_tile(0, 1);
    engine.timing_config.memtile_first_bd_startup_cycles = 100;
    let bd = BdConfig::simple_1d(0x1000, 16);
    // Memtile MM2S moves local memory <-> stream; it does NOT touch host DDR,
    // so it takes the non-shim startup path rather than the shim cold-start.
    let transfer = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 0, 1, engine.tile_kind).unwrap();
    assert!(!transfer.involves_host_memory(), "memtile transfer is not host DDR");

    // First task: MemoryLatency bonus is channel_start only; the startup is
    // latched as a post-transfer hold instead.
    engine.channels[0].is_first_bd = true;
    assert_eq!(engine.consume_first_bd_bonus(0, &transfer), 2, "bonus stays channel_start only");
    assert_eq!(engine.channels[0].startup_hold_cycles, 100, "startup latched as post-transfer hold");

    // The hold fires once (begin_completion clears it); emulate that, then a
    // second task -- warm_task_index>0, so no re-latch (one-time per session).
    engine.channels[0].startup_hold_cycles = 0;
    engine.channels[0].is_first_bd = true;
    assert_eq!(engine.consume_first_bd_bonus(0, &transfer), 2);
    assert_eq!(engine.channels[0].startup_hold_cycles, 0, "startup is one-time per session");
}

/// The non-shim startup knobs default to 0 (behavior-preserving), and each
/// applies only to its own tile kind -- the compute knob must not latch onto a
/// memtile channel.
#[test]
fn test_non_shim_first_bd_startup_defaults_zero_and_is_tile_scoped() {
    let cfg = DmaTimingConfig::default();
    assert_eq!(cfg.memtile_first_bd_startup_cycles, 0);
    assert_eq!(cfg.compute_first_bd_startup_cycles, 0);

    // Set the WRONG knob for a memtile: the compute knob must not latch.
    let mut mem = DmaEngine::new_mem_tile(0, 1);
    mem.timing_config.compute_first_bd_startup_cycles = 99;
    let bd = BdConfig::simple_1d(0x1000, 16);
    let t = Transfer::new(&bd, 0, 0, TransferDirection::MM2S, 0, 1, mem.tile_kind).unwrap();
    mem.channels[0].is_first_bd = true;
    assert_eq!(mem.consume_first_bd_bonus(0, &t), 2);
    assert_eq!(mem.channels[0].startup_hold_cycles, 0, "compute knob must not latch on a memtile");
}

/// The non-shim startup is a POST-transfer hold, not a pre-transfer stall: the
/// extra cycles are spent in `StartupHold` (after all data has moved), so input
/// is consumed on schedule and nothing backpressures upstream.  Drives a compute
/// MM2S transfer end-to-end with startup 0 vs 50 and checks the delta lands in
/// the hold.
///
/// The measured delta is `startup - 1`, not `startup`: an MM2S completion now
/// spends one `DrainingEgress` cycle waiting for its last beat to handshake off
/// the egress FIFO (#140 SP-4a, lock-release-on-handshake).  The startup=0
/// baseline pays that cycle visibly; the startup>0 run drains its egress DURING
/// the hold, so `DrainingEgress` is a no-op there and the cycle is absorbed.
/// The hold still adds exactly `startup` cycles -- the -1 is the baseline's
/// egress tail, not a smaller hold.
#[test]
fn test_non_shim_startup_is_post_transfer_hold() {
    fn run(startup: u16) -> (u64, bool) {
        let mut engine = DmaEngine::new_compute_tile(1, 2);
        engine.timing_config.compute_first_bd_startup_cycles = startup;
        let mut tile = make_tile();
        let mut host = make_host_memory();
        engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();
        engine.start_channel(2, 0).unwrap();
        let mut cycles = 0u64;
        let mut saw_hold = false;
        while engine.channel_active(2) {
            engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host);
            while engine.pop_stream_out().is_some() {}
            if engine.channels[2].fsm.phase_name() == "StartupHold" {
                saw_hold = true;
            }
            cycles += 1;
            assert!(cycles < 500, "transfer hung");
        }
        (cycles, saw_hold)
    }
    let (base, base_hold) = run(0);
    let (held, held_hold) = run(50);
    assert!(!base_hold, "no StartupHold when startup is 0");
    assert!(held_hold, "channel passes through StartupHold when startup > 0");
    // startup(50) - 1 egress-handshake cycle absorbed by the hold; see doc above.
    assert_eq!(
        held - base,
        49,
        "post-transfer hold adds the startup, net the baseline's 1-cycle egress tail"
    );
}

/// Phase 2d.2 Part 2: the controller dispatch index is monotonic per
/// channel session.  It counts task dispatches (Task_Queue writes) and
/// must NOT collapse when the channel drains back to Idle between tasks
/// -- that persistence is what lets the dispatch gate ramp to its
/// plateau instead of snapping back to the base rate every time a short
/// task finishes (instantaneous occupancy can't do this).  Reset only on
/// stop_channel (channel reset == fresh boot), mirroring warm_task_index.
#[test]
fn controller_dispatch_index_is_monotonic_across_drains() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 16)).unwrap();

    // Fresh channel: no dispatches yet.
    assert_eq!(engine.controller_dispatch_index(2), 0, "fresh channel -> index 0");

    // First dispatch -- enqueue auto-starts the idle MM2S channel.
    assert!(engine.enqueue_task(2, 0, 0, false), "first task enqueues");
    assert_eq!(engine.controller_dispatch_index(2), 1, "first dispatch -> index 1");

    // Drain the task fully; channel returns to Idle, queue empties.
    let mut cycles = 0;
    while engine.channel_active(2) {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        assert!(cycles < 1000, "transfer hung");
    }
    assert_eq!(engine.channel_state(2), ChannelState::Idle);
    assert_eq!(engine.task_queue_size(2), 0);

    // Instantaneous occupancy is now 0 -- but the dispatch index must
    // persist at 1, not collapse.  This is the whole point of the pivot.
    assert_eq!(
        engine.controller_dispatch_index(2),
        1,
        "dispatch index persists across a drain to Idle (occupancy would read 0 here)"
    );

    // Second dispatch climbs to 2 -- the ramp keeps rising.
    assert!(engine.enqueue_task(2, 0, 0, false), "second task enqueues");
    assert_eq!(engine.controller_dispatch_index(2), 2, "second dispatch -> index 2");

    // stop_channel re-arms to 0 (fresh boot), like warm_task_index.
    engine.stop_channel(2).unwrap();
    assert_eq!(engine.controller_dispatch_index(2), 0, "stop_channel re-arms dispatch index");
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
        // Mirror routing's per-cycle egress drain so the MM2S completion's
        // DrainingEgress (#140 SP-4a) clears instead of hanging with no consumer.
        while engine.pop_stream_out().is_some() {}
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

    // Fill the queue to its hardware depth
    for i in 0..MAX_TASK_QUEUE_DEPTH {
        assert!(engine.enqueue_task(2, 0, 0, false), "Task {} should enqueue", i);
    }

    // One past depth should fail (queue full)
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

/// BD-prefetch overlap (Phase 2d.2): when a task sits in the queue while the
/// channel transfers the current task, HW loads the next BD and fires its
/// START_TASK event *during* the current transfer -- not after FINISHED.  This
/// dual-state (Cur_BD executing + next BD loaded) is the mechanism behind a
/// negative MM2S gap[0] on a long first task: START[i+1] can precede
/// FINISHED[i].  The channel data path stays serial; only the event moves.
#[test]
fn test_queued_task_start_emitted_during_prior_transfer() {
    let mut engine = DmaEngine::new_shim_tile(0, 0);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();
    engine.configure_bd(1, BdConfig::simple_1d(0x2000, 256)).unwrap();

    let ch = (0..engine.num_channels() as u8)
        .find(|&c| matches!(engine.channel_type(c), ChannelType::MM2S))
        .expect("shim has an MM2S channel");

    // Both tasks queued up front, so task[1] is in the queue while task[0]
    // transfers and the channel can prefetch it.
    assert!(engine.enqueue_task(ch, 0, 0, false));
    assert!(engine.enqueue_task(ch, 1, 0, false));

    let mut events: Vec<(u64, EventType)> = Vec::new();
    let mut steps = 0;
    loop {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {} // drain so MM2S doesn't backpressure
        events.extend(engine.drain_trace_events());
        steps += 1;
        if !engine.channel_active(ch) || steps > 4000 {
            break;
        }
    }
    events.extend(engine.drain_trace_events());

    // Event kinds in emission order.  current_cycle is driven externally in
    // the real device (0 in this standalone test), so we assert on order, not
    // timestamps.  Prefetch overlap => both STARTs fire before either FINISH:
    // S[0], S[1] (prefetched during task[0]), F[0], F[1].  The pre-2d.2 serial
    // model emits S[0], F[0], S[1], F[1].
    let seq: Vec<char> = events
        .iter()
        .filter_map(|(_, e)| match e {
            EventType::DmaStartTask { .. } => Some('S'),
            EventType::DmaFinishedTask { .. } => Some('F'),
            _ => None,
        })
        .collect();

    assert_eq!(seq, vec!['S', 'S', 'F', 'F'], "prefetch: both STARTs must precede either FINISH");
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

/// At each `next_bd` boundary, hardware deasserts PORT_RUNNING for a minimum
/// 1-cycle bubble -- the BD-switch handshake (next_bd fetch + lock cycle) costs
/// a cycle even on the prefetch fast path. NPU1 add_one memtile slot0 traces a
/// clean `on16 off1 x4` for its four 16-word S2MM BD executions; slot4 (MM2S)
/// shows the same `off1` bubbles. The emulator's chained-BD prefetch went
/// straight to Transferring with NO inter-BD gap, so a chained transfer came
/// out as one unbroken beat-run instead of per-BD runs separated by bubbles.
///
/// This drives the consumer/producer-side BD-switch bubble: a looping MM2S
/// chain must show a gap in stream-beat production between consecutive BDs.
#[test]
fn chained_bd_inserts_port_running_bubble_at_each_boundary() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // BD0 -> BD1 -> BD0 ... each 64 bytes (16 words => multiple Transferring
    // cycles per BD at 4 words/cycle), no locks (so the only inter-BD cost is
    // the BD-switch bubble we are modeling).
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 64).with_next(1)).unwrap();
    engine.configure_bd(1, BdConfig::simple_1d(0x200, 64).with_next(0)).unwrap();
    engine.enqueue_task(2, 0, 0, false); // MM2S ch0, start BD0, self-looping chain

    // Record stream beats produced per cycle (drain fully so MM2S never
    // backpressures and adds gaps of its own).
    let mut per_cycle_beats = Vec::new();
    for _ in 0..160 {
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        let mut n = 0;
        while engine.pop_stream_out().is_some() {
            n += 1;
        }
        per_cycle_beats.push(n);
    }

    // Count "interior" zero-gaps: a run of zero-beat cycles bracketed by beats
    // on both sides. The leading cold-start zeros are excluded (no beats before
    // them); each remaining gap is a per-BD PORT_RUNNING bubble. With the
    // prefetch fast path going straight to Transferring there are zero interior
    // gaps (one unbroken run); the bubble must produce one per BD boundary.
    let mut gap_widths = Vec::new();
    let mut seen_beat = false;
    let mut i = 0;
    while i < per_cycle_beats.len() {
        if per_cycle_beats[i] > 0 {
            seen_beat = true;
            i += 1;
            continue;
        }
        let start = i;
        while i < per_cycle_beats.len() && per_cycle_beats[i] == 0 {
            i += 1;
        }
        let beat_after = i < per_cycle_beats.len() && per_cycle_beats[i] > 0;
        if seen_beat && beat_after {
            gap_widths.push(i - start);
        }
    }
    assert!(
        gap_widths.len() >= 3,
        "expected a PORT_RUNNING bubble at each BD boundary (>=3 interior gaps), got {}; per-cycle beats: {per_cycle_beats:?}",
        gap_widths.len()
    );
    // Each bubble is exactly the default 1-cycle off1 -- not 2 (no double-count
    // with the lock/grant cycle) and not wider.
    assert!(
        gap_widths.iter().all(|&w| w == 1),
        "expected every BD-switch bubble to be exactly 1 cycle (HW off1), got widths {gap_widths:?}"
    );
}

/// Symmetric recv-side counterpart to
/// `chained_bd_inserts_port_running_bubble_at_each_boundary`: on an S2MM
/// (receive) chain, the channel deasserts its *external* stream-accept (TREADY)
/// for the 1-cycle BD-switch reconfiguration, so the recv PORT_RUNNING (the
/// switch->DMA pop) gaps once per BD. HW: NPU1 add_one memtile slot0 traces a
/// clean `on16 off1 x4` for its four 16-word S2MM BD executions.
///
/// This drives the *locked* objfifo path that add_one actually uses: each BD
/// re-acquires its producer lock, and with the lock immediately available the
/// chained-grant collapses straight to Transferring (no `BdSwitchBubble`), so
/// the memory-side bubble that the no-lock path gets is skipped AND the
/// external accept is never gapped -- the channel front-loads through every
/// boundary. The accept must be gated during the BD-switch reconfiguration
/// regardless of whether the grant was immediate.
#[test]
fn s2mm_recv_deasserts_accept_at_each_bd_boundary() {
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    // Producer lock kept plentiful so every chained acquire grants immediately
    // (the immediate-grant collapse path -- add_one's `prod_lock` is freed by
    // the consumer fast enough that the recv never hard-waits, so its only
    // inter-BD cost is the 1-cycle reconfiguration bubble).
    own.locks[0].set(8);

    // Chained 2-BD ring, each 64 bytes (16 words), each re-acquiring lock 0 and
    // releasing it inline (same-lock self-chain). Self-loops BD0 -> BD1 -> BD0;
    // we feed 64 words = 4 BD executions = 3 interior boundaries.
    engine
        .configure_bd(0, BdConfig::simple_1d(0x100, 64).with_acquire(64, -1).with_next(1))
        .unwrap();
    engine
        .configure_bd(1, BdConfig::simple_1d(0x200, 64).with_acquire(64, -1).with_next(0))
        .unwrap();
    engine.enqueue_task(0, 0, 0, false); // S2MM ch0, start BD0, self-looping chain

    // Offer one word per cycle, accepting only when the channel will (mirrors
    // `route_tile_switches_to_dma`: it offers one beat/cycle and, when the
    // channel refuses, elapses one cycle of the BD-switch deassert via
    // `consume_bd_switch_accept_block`). Record per-cycle whether a word was
    // accepted. The accepted stream forms runs separated by gaps; HW gates the
    // recv port per-BD, so each run is one BD's worth (`on16 off1`), NOT a
    // front-loaded double-buffer.
    let total = 64usize;
    let bd_words = 16usize; // each BD is 64 bytes = 16 words
    let mut fed = 0usize;
    let mut per_cycle_accept = Vec::new();
    for _ in 0..400 {
        let mut accepted = 0u32;
        if !engine.can_accept_stream_in_for_channel(0) {
            // Refused: elapse one cycle of the BD-switch reconfiguration gap
            // (no-op if the refusal was FIFO-full rather than the deassert).
            engine.consume_bd_switch_accept_block(0);
        } else if fed < total {
            let tlast = fed == total - 1;
            let data = 0xA000_0000 | fed as u32;
            if engine.push_stream_in(StreamData { data, tlast, channel: 0 }) {
                fed += 1;
                accepted = 1;
            }
        }
        engine.submit_lock_requests(&mut own, &mut NeighborTiles::empty());
        own.resolve_lock_requests(0);
        engine.step(&mut own, &mut NeighborTiles::empty(), &mut host_mem);
        per_cycle_accept.push(accepted);
        if fed >= total && !engine.channel_active(0) {
            break;
        }
    }
    assert!(fed >= total, "channel did not consume all {total} words (fed {fed})");

    // Accepted-word runs (consecutive accept cycles) and the gaps between them.
    let mut runs = Vec::new();
    let mut run = 0usize;
    let mut gaps = 0usize;
    let mut seen_accept = false;
    for &a in &per_cycle_accept {
        if a > 0 {
            run += 1;
            seen_accept = true;
        } else {
            if run > 0 {
                runs.push(run);
                run = 0;
            }
            if seen_accept {
                gaps += 1; // interior-ish gap (after at least one accept)
            }
        }
    }
    if run > 0 {
        runs.push(run);
    }

    // The recv port gates per-BD: NO accepted run exceeds one BD's word count.
    // Before the fix the channel front-loads the prod_lock=2 double-buffer into
    // one ~32-word run (decoded slot0 `[34,16,14]`); after, each run is bounded
    // by the BD length (`on16 off1` -> `[16,16,16,16]`).
    let max_run = runs.iter().copied().max().unwrap_or(0);
    assert!(
        max_run <= bd_words,
        "recv accept must be gated per-BD ({bd_words} words/BD), but saw a run of {max_run} \
         (front-loading the double-buffer); runs={runs:?}"
    );
    // Sanity: the transfer really did span multiple BDs with gaps between them
    // (so the bound above isn't vacuously satisfied by a stalled channel).
    assert!(gaps >= 3, "expected >=3 inter-BD gaps across the 4 BD executions, got {gaps}; runs={runs:?}");
}

#[test]
fn memtile_s2mm_recv_stages_full_bd_while_buffer_lock_stalled() {
    // HW (#140 relay-fill, decisive co-capture 2026-06-27): the memtile S2MM
    // ingress absorbs a FULL 16-word BD ahead of a lock-stalled buffer write, so
    // PORT_RUNNING stays a clean [16,16,16,16]. The staging depth is the device-
    // model StreamSwitch.fifo_depth (16), shared by compute and mem tiles (both
    // also have s2mmChannel.buffer_depth=12); the clean [16,16,16,16] with 16-word
    // BDs already refutes a 12-deep model (depth 12 would split BD3 as 12+4) and
    // pins depth >= 16. EMU previously modeled
    // the DMA ingress as the 2-deep `STREAM_LOCAL_MASTER_FIFO_DEPTH`, so a
    // lock-stalled recv backpressured after just 2 words -- the [16,16,2,14,16]
    // relay-fill split. With the corrected depth, a recv whose buffer write is
    // blocked on its producer lock stages the whole BD before backpressuring.
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut own = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    // Lock 0 left at its default (0) so the BD's AcquireGreaterEqual(>=1) never
    // grants: the buffer write stalls, exactly the recv-BD3-reuse case where the
    // producer lock is not yet freed by the consumer MM2S.
    engine
        .configure_bd(0, BdConfig::simple_1d(0x100, 64).with_acquire(64, -1))
        .unwrap();
    engine.enqueue_task(0, 0, 0, false); // S2MM ch0, single 16-word BD

    let mut accepted_run = 0usize;
    for _ in 0..200 {
        if engine.can_accept_stream_in_for_channel(0) {
            let data = 0xB000_0000 | accepted_run as u32;
            if engine.push_stream_in(StreamData { data, tlast: false, channel: 0 }) {
                accepted_run += 1;
            }
        } else {
            engine.consume_bd_switch_accept_block(0);
            // The buffer write is permanently lock-stalled, so once the channel
            // refuses after accepting something, the FIFO is full -- the accept
            // run has reached the ingress depth.
            if accepted_run > 0 {
                break;
            }
        }
        engine.submit_lock_requests(&mut own, &mut NeighborTiles::empty());
        own.resolve_lock_requests(0);
        engine.step(&mut own, &mut NeighborTiles::empty(), &mut host_mem);
    }

    assert_eq!(
        accepted_run, 16,
        "a lock-stalled memtile S2MM recv must stage a full 16-word BD into the \
         ingress FIFO before backpressuring (HW absorbs the whole BD); got {accepted_run} \
         (pre-fix the 2-deep master-port model capped it at 2)"
    );
}

#[test]
fn compute_s2mm_recv_stages_full_bd_while_buffer_lock_stalled() {
    // HW (#140 device-model audit, co-capture 2026-06-27): the COMPUTE tile's
    // S2MM ingress also absorbs a full BD ahead of a lock-stalled buffer write,
    // identical to the memtile. add_one_using_dma's compute recv decodes to a
    // clean [8,8,8,8,8,8,8,8] on HW, while EMU (pre-fix) split every reused BD
    // as 2+6 -- the 2-deep `STREAM_LOCAL_MASTER_FIFO_DEPTH` backpressure, the
    // exact pre-fix memtile signature on the tile type we never fixed. Compute
    // and memtile share identical device-model DMA FIFO params (s2mmChannel.
    // buffer_depth=12, StreamSwitch.fifo_depth=16), so the deep S2MM ingress
    // depth (16) generalizes across both. This test uses a 16-word BD to
    // exercise the full depth (the kernel's 8-word BD only proves >=8).
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut own = Tile::compute(1, 2);
    let mut host_mem = make_host_memory();

    // Lock 0 left at its default (0) so AcquireGreaterEqual(>=1) never grants:
    // the buffer write stalls, exactly the recv-BD-reuse case.
    engine
        .configure_bd(0, BdConfig::simple_1d(0x100, 64).with_acquire(64, -1))
        .unwrap();
    engine.enqueue_task(0, 0, 0, false); // S2MM ch0, single 16-word BD

    let mut accepted_run = 0usize;
    for _ in 0..200 {
        if engine.can_accept_stream_in_for_channel(0) {
            let data = 0xC000_0000 | accepted_run as u32;
            if engine.push_stream_in(StreamData { data, tlast: false, channel: 0 }) {
                accepted_run += 1;
            }
        } else {
            engine.consume_bd_switch_accept_block(0);
            if accepted_run > 0 {
                break;
            }
        }
        engine.submit_lock_requests(&mut own, &mut NeighborTiles::empty());
        own.resolve_lock_requests(0);
        engine.step(&mut own, &mut NeighborTiles::empty(), &mut host_mem);
    }

    assert_eq!(
        accepted_run, 16,
        "a lock-stalled compute S2MM recv must stage a full 16-word BD into the \
         ingress FIFO before backpressuring (HW stages the whole BD, same as the \
         memtile); got {accepted_run} (pre-fix the 2-deep master-port model capped it at 2)"
    );
}

#[test]
fn s2mm_drains_staged_ingress_to_memory_at_memory_bus_rate() {
    // The S2MM data path has two DISTINCT interfaces with different widths:
    //   - the AXI4-Stream READ that FILLS the ingress FIFO: 1 word/cyc (the
    //     32-bit stream beat), modeled by the routing layer / `push_stream_in`.
    //   - the memory WRITE that DRAINS the ingress to tile data memory: the
    //     128-bit data-memory bus = 4 words/cyc (DATAMEMORY_WIDTH).
    // When the ingress already holds several words (staged during a lock-stall
    // or BD setup) and the buffer lock is free, the DMA must drain them at the
    // MEMORY-bus rate, not the stream rate. Conflating the two (draining at
    // 1 word/cyc) keeps the ingress artificially full-headroom between buffers
    // and inflates the producer's transient lead in the send cascade (#140).
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut own = Tile::compute(1, 2);
    let mut host_mem = make_host_memory();

    // 16-word S2MM BD, no acquire -> the buffer write never lock-stalls, so the
    // channel drains the ingress as fast as the memory bus allows.
    engine.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();
    engine.enqueue_task(0, 0, 0, false); // S2MM ch0

    // Pre-stage a full ingress (capacity admits 16) BEFORE stepping, mimicking
    // words that arrived while the channel was setting up. No drain has run yet,
    // so all 16 pushes succeed.
    for i in 0..16u32 {
        assert!(
            engine.push_stream_in(StreamData { data: 0xD000_0000 | i, tlast: i == 15, channel: 0 }),
            "ingress should admit word {i} (capacity 16, nothing drained yet)"
        );
    }

    // Step to completion WITHOUT feeding more words. The 16 staged words must
    // drain to memory; track the largest per-cycle byte advance. At the stream
    // rate the drain is capped at 4 bytes/cyc (one word); at the memory-bus rate
    // a cycle commits up to 4 words = 16 bytes.
    let mut max_delta = 0u64;
    let mut prev = 0u64;
    for _ in 0..200 {
        engine.submit_lock_requests(&mut own, &mut NeighborTiles::empty());
        own.resolve_lock_requests(0);
        engine.step(&mut own, &mut NeighborTiles::empty(), &mut host_mem);
        let now = engine.channel_stats(0).map_or(0, |s| s.bytes_transferred);
        max_delta = max_delta.max(now.saturating_sub(prev));
        prev = now;
        if !engine.channel_active(0) {
            break;
        }
    }

    assert_eq!(prev, 64, "all 16 staged words must reach memory");
    assert!(
        max_delta >= 8,
        "S2MM must drain a staged ingress at the memory-bus rate (>=2 words/cyc); \
         saw max {max_delta} bytes/cyc, i.e. the 1-word/cyc stream rate (the egress \
         meter wrongly applied to the memory-write side)"
    );
}

#[test]
fn s2mm_blocks_next_bd_accept_until_current_bd_drains() {
    // HW (#140 send-cadence co-trace, 2026-06-28): an S2MM ingress stages at
    // most ONE BD ahead of the memory write. A single BD absorbs fully (the
    // cold absorb-16 case), but the NEXT BD in a chain is NOT accepted until
    // the current BD's words have DRAINED from the ingress. aie-rt confirms the
    // stream-accept is occupancy-driven and fills regardless of lock (the memory
    // WRITE blocks on the lock, not the accept), so the bound is per-BD drain --
    // not lock-gated accept (which would break the cold absorb).
    //
    // Consequence on the send cascade: a memtile MM2S producer feeding a
    // lock-stalled compute S2MM consumer (8-word double-buffer) gets ONE 8-word
    // BD ahead, not two. Pre-fix the fixed 1-cycle bd_switch bubble let BD1 stage
    // on top of undrained BD0, so the ingress held 16 and the producer ran
    // 16-ahead -- the [16,16,16,...] send signature vs HW's [8,8,14,...].
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut own = Tile::compute(1, 2);
    let mut host_mem = make_host_memory();

    // Chained 2-BD ring, each 32 bytes (8 words), each acquiring lock 64 which is
    // never set -- the buffer WRITE never grants, so BD0's words stage into the
    // ingress but cannot drain. Self-loop BD0 -> BD1 -> BD0.
    engine
        .configure_bd(0, BdConfig::simple_1d(0x100, 32).with_acquire(64, -1).with_next(1))
        .unwrap();
    engine
        .configure_bd(1, BdConfig::simple_1d(0x200, 32).with_acquire(64, -1).with_next(0))
        .unwrap();
    engine.enqueue_task(0, 0, 0, false); // S2MM ch0, start BD0, self-looping chain

    // Feed one word per cycle for a fixed window, elapsing the per-BD bubble on
    // refusal (mirrors the routing pass). With the buffer write permanently
    // lock-stalled, the ingress can only ever hold ONE BD: BD0's 8 words stage,
    // then -- because BD0 never drains -- BD1 is held off. Total staged = 8.
    let mut fed = 0usize;
    for _ in 0..200 {
        if engine.can_accept_stream_in_for_channel(0) {
            let data = 0xE000_0000 | fed as u32;
            if engine.push_stream_in(StreamData { data, tlast: false, channel: 0 }) {
                fed += 1;
            }
        } else {
            engine.consume_bd_switch_accept_block(0);
        }
        engine.submit_lock_requests(&mut own, &mut NeighborTiles::empty());
        own.resolve_lock_requests(0);
        engine.step(&mut own, &mut NeighborTiles::empty(), &mut host_mem);
    }

    assert_eq!(
        fed, 8,
        "a lock-stalled S2MM with chained 8-word BDs must stage only ONE BD (8 words) \
         into the ingress -- the next BD cannot accept until the current BD drains; \
         got {fed} (pre-fix the fixed 1-cycle bubble let BD1 stage on top of undrained \
         BD0 -> 16 = two BDs, the producer-16-ahead send-cadence defect)"
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
/// it never goes to Error and never spins past `max_cycles`.  Mirrors routing
/// by draining `stream_out` every cycle (so the MM2S completion's
/// `DrainingEgress` clears, #140 SP-4a) and returns the drained words in order
/// -- after completion `stream_out` is empty, as it is in the real array, so
/// callers assert on the returned words rather than on post-run `stream_out`.
fn run_memtile_mm2s_to_completion(
    engine: &mut DmaEngine,
    channel: u8,
    own: &mut Tile,
    neighbors_west: Option<&mut Tile>,
    neighbors_east: Option<&mut Tile>,
    max_cycles: usize,
) -> Vec<StreamData> {
    use std::cell::RefCell;
    // Stash neighbours in local RefCells so we can rebuild a NeighborTiles
    // each iteration without violating disjoint-borrow rules.
    let west_cell = neighbors_west.map(RefCell::new);
    let east_cell = neighbors_east.map(RefCell::new);

    let mut host_mem = make_host_memory();
    let mut collected: Vec<StreamData> = Vec::new();
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
        // Mirror routing's per-cycle egress drain (the array pops each MM2S
        // stream_out every cycle), collecting the beats so the caller can
        // assert on them. Without this the isolated engine would hold in
        // DrainingEgress forever (no consumer). (#140 SP-4a.)
        while let Some(w) = engine.pop_stream_out() {
            collected.push(w);
        }

        if !engine.channel_active(channel) {
            return collected;
        }
        if matches!(engine.channel_state(channel), ChannelState::Error) {
            panic!("DMA channel {} entered Error state at cycle {}", channel, cycle);
        }
    }
    panic!("DMA channel {} did not complete within {} cycles", channel, max_cycles);
}

/// Drive an S2MM MemTile DMA channel to completion, streaming `feed` words
/// into `stream_in` as FIFO capacity allows each cycle.
///
/// Real silicon delivers S2MM input one word per cycle via the stream switch;
/// the input FIFO is shallow (`input_fifo_capacity`). Tests must not bulk-push
/// the whole payload up front -- that overflows the HW-faithful FIFO. This
/// helper mirrors routing by offering the next word each cycle only when the
/// channel's FIFO has room.
fn run_memtile_s2mm_to_completion(
    engine: &mut DmaEngine,
    channel: u8,
    own: &mut Tile,
    neighbors_west: Option<&mut Tile>,
    neighbors_east: Option<&mut Tile>,
    feed: &[(u32, bool)],
    max_cycles: usize,
) {
    use std::cell::RefCell;
    let west_cell = neighbors_west.map(RefCell::new);
    let east_cell = neighbors_east.map(RefCell::new);

    let mut host_mem = make_host_memory();
    let mut next = 0;
    for cycle in 0..max_cycles {
        // Offer the next word if the channel FIFO has room (mimics the stream
        // switch delivering one beat per cycle under backpressure).
        if next < feed.len() && engine.can_accept_stream_in_for_channel(channel) {
            let (data, tlast) = feed[next];
            if engine.push_stream_in(StreamData { data, tlast, channel }) {
                next += 1;
            }
        }

        let mut west_borrow = west_cell.as_ref().map(|c| c.borrow_mut());
        let mut east_borrow = east_cell.as_ref().map(|c| c.borrow_mut());
        let mut neighbors = NeighborTiles {
            west: west_borrow.as_deref_mut().map(|b| &mut **b),
            east: east_borrow.as_deref_mut().map(|b| &mut **b),
        };
        engine.step(own, &mut neighbors, &mut host_mem);
        drop(west_borrow);
        drop(east_borrow);

        if next >= feed.len() && !engine.channel_active(channel) {
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

    let out = run_memtile_mm2s_to_completion(&mut engine, 6, &mut own_tile, None, Some(&mut east_tile), 500);

    // The four stream words must contain east_tile's 0xEE pattern.
    let expected_word = u32::from_le_bytes([0xEE; 4]);
    assert_eq!(out.len(), 4, "expected 4 stream words from east tile");
    for w in out {
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

    let out = run_memtile_mm2s_to_completion(&mut engine, 6, &mut own_tile, Some(&mut west_tile), None, 500);

    let expected_word = u32::from_le_bytes([0x77; 4]);
    assert_eq!(out.len(), 4, "expected 4 stream words from west tile");
    for w in out {
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

    // 4 stream words for a 16-byte S2MM transfer on channel 0, streamed in as
    // the shallow input FIFO drains (not bulk-pushed -- that overflows it).
    let payload_word = u32::from_le_bytes([0xC3; 4]);
    let feed: Vec<(u32, bool)> = (0..4).map(|i| (payload_word, i == 3)).collect();

    // BD writes into the East window at offset 0x300, length 16.
    // 0x100300 = East window (0x100000) + offset 0x300.
    let bd = BdConfig::simple_1d(0x100300, 16);
    engine.configure_bd(0, bd).unwrap();
    engine.start_channel(0, 0).unwrap();

    run_memtile_s2mm_to_completion(&mut engine, 0, &mut own_tile, None, Some(&mut east_tile), &feed, 500);

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

    let out = run_memtile_mm2s_to_completion(&mut engine, 6, &mut own_tile, None, None, 500);

    assert!(
        engine.fatal_errors.is_empty(),
        "missing East neighbour should fall back to Own, not fatal-error: {:?}",
        engine.fatal_errors
    );
    let expected_word = u32::from_le_bytes([0x42; 4]);
    assert_eq!(out.len(), 4, "expected 4 stream words from own tile");
    for w in out {
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

    // Flood channel 1's (trace) FIFO to capacity. With a shared buffer this
    // would block channel 0; per-channel FIFOs keep them independent.
    let cap = engine.input_fifo_capacity();
    for i in 0..cap {
        let pushed =
            engine.push_stream_in(StreamData { data: 0xFEED_0000 | i as u32, tlast: false, channel: 1 });
        assert!(pushed, "channel 1 push {} should succeed", i);
    }
    // Channel 1 is now full -- the next push to it must be rejected.
    assert!(!engine.can_accept_stream_in_for_channel(1), "channel 1 FIFO should be full at capacity {}", cap);

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

/// Chained BDs with acquire+release locks on a private lock pair finish with
/// `interval == data_cycles + bd_switch_bubble_cycles`. The #26 pipelining
/// optimizations (inline release, inline grant, inline first-data) removed the
/// EMU-only dead cycles -- correct -- but the inline-first-data step also
/// hid the ONE real cycle HW keeps: the per-BD `next_bd`-fetch/lock-handshake
/// bubble. NPU1 add_one memtile slot0 traces `on16 off1` per 16-word BD, i.e.
/// a 17-cycle period; PORT_RUNNING cannot deassert (`off1`) without a no-beat
/// cycle, and a no-beat cycle is by definition additive to BD throughput. So
/// the HW interval is data+1, and `bd_switch_bubble_cycles` (default 1)
/// restores it. (Set the env to 0 to recover the old back-to-back behavior.)
///
/// For 2-word BDs (8 bytes) at the 1-word/cyc stream rate (these are MM2S
/// transfers -- memory -> stream), each BD is 2 Transferring cycles, so the
/// FINISHED_BD interval is 2 data + 1 bubble = 3. (Stream egress meters at the
/// 32-bit AXI4-Stream beat width, not the 4-word tile data bus -- #140.)
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
        interval, 3,
        "chained-BD FINISHED_BD interval == data cycles (2w @ 1 word/cyc stream rate) \
         + BD-switch bubble (1); matches HW off1. cycles={:?}",
        finished_bd_cycles
    );
}

/// 4-BD chain of 16-word locked MM2S (memory -> stream) BDs. Stream egress
/// meters at the 32-bit AXI4-Stream rate (1 word/cyc), so `16w / 1wpc = 16`
/// data cycles per BD, plus the 1-cycle per-BD-switch bubble HW keeps
/// => interval 17 per BD. This is exactly HW's add_one slot0 `on16 off1`
/// (16 cycles asserted per 16-word BD, 1 cycle bubble) -- before #140 the
/// model used the 4-word data-bus rate here and produced the wrong `on4`.
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
        vec![17u64, 17, 17],
        "16w chained-BD intervals = 16 data (1 word/cyc stream) + 1 BD-switch bubble \
         = HW on16/off1; FINISHED_BD cycles={:?}",
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
    // 16w MM2S (memory -> stream) BD at the 1-word/cyc stream rate = 16 data
    // cycles. Without locks, enter_chained_bd routes through the BD-switch
    // bubble (BdSwitchBubble for 1 cycle) before Transferring, so interval =
    // 16 data + 1 bubble = 17. The bubble is the same per-BD `off1` HW shows
    // whether or not the BD carries locks (#140 stream-rate metering).
    assert_eq!(
        intervals,
        vec![17u64, 17, 17],
        "no-lock chained-BD intervals = 16 data + 1 bubble; FINISHED_BD cycles={:?}",
        finished_bd_cycles
    );
}

/// Cross-lock handoff: a chained BD's release is DEFERRED to the next BD's
/// acquire-grant, not applied at BD completion.
///
/// Tenant-4 mechanism (HW-validated on NPU1): on a memtile shared-link
/// producer/consumer, a completing BD's lock RELEASE is held until the next
/// A cross-lock FUNCTIONAL release must fire INLINE at BD completion -- never
/// deferred to the next BD's acquire grant. This is the deadlock-prevention
/// invariant: the consumer of a cross-lock release may be the compute CORE,
/// which is not a DMA channel, so it cannot trigger a buffer swap. If the
/// functional release waited for the swap, a DMA->core handoff with no buffer
/// slack would wedge (the DMA defers its release awaiting its own next acquire,
/// which needs the core to free the buffer, which needs that release).
///
/// (Only the OBSERVABLE memtile-S2MM trace event defers, to BD-slot reuse --
/// the lock VALUE is always prompt. See PendingRelease and the memtile_*
/// trace-timing tests.)
///
/// This drives a single compute-tile channel whose BD0 releases lock 2 and
/// chains to BD1, which acquires lock 1. lock 1 starts unavailable, so BD1
/// blocks. While blocked, lock 2 must already read 1 -- the functional release
/// fired at BD0 completion, available immediately to a waiting consumer.
#[test]
fn cross_lock_release_is_prompt_not_deferred() {
    let mut engine = DmaEngine::new_compute_tile(1, 2);
    let mut tile = make_tile();
    let mut host_mem = make_host_memory();

    // BD0 acquires lock0, releases lock2 (cross-lock: next BD acquires lock1),
    // chains to BD1. BD1 acquires lock1, releases lock3, ends the task.
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(0x100, 16)
                .with_acquire(0, 1)
                .with_release(2, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(1, BdConfig::simple_1d(0x200, 16).with_acquire(1, 1).with_release(3, 1))
        .unwrap();

    tile.locks[0].set(1); // BD0 can acquire immediately
    tile.locks[1].set(0); // BD1 will block until we free it
    tile.locks[2].set(0);
    tile.locks[3].set(0);

    engine.start_channel(2, 0).unwrap();

    // Phase 1: run until BD1 is blocked on lock 1.
    let mut cycles = 0;
    loop {
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(0);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        assert!(cycles < 100, "BD1 never reached the blocked-on-lock1 state");
        if matches!(engine.channel_state(2), ChannelState::WaitingForLock(1)) {
            break;
        }
    }

    // BD0 has completed. Its cross-lock release of lock2 fired INLINE at
    // completion, so lock2 MUST read 1 here even though BD1 is still blocked --
    // a waiting consumer (e.g. the core) must be able to proceed immediately.
    assert_eq!(
        tile.locks[2].value, 1,
        "cross-lock functional release must be prompt: lock2 should be 1 at BD0 completion, got {}",
        tile.locks[2].value
    );

    // Phase 2: free lock1 and run to completion; BD1's own release also fires.
    tile.locks[1].set(1);
    cycles = 0;
    while engine.channel_active(2) {
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(0);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        while engine.pop_stream_out().is_some() {}
        cycles += 1;
        assert!(cycles < 100, "channel did not complete after lock1 freed");
    }

    assert_eq!(tile.locks[2].value, 1, "lock2 release stays applied");
    assert_eq!(tile.locks[3].value, 1, "BD1 task-end release (lock3) fires inline");
}

/// Two-channel memtile producer/consumer double-buffer: deferred TRACE events
/// must couple at the swap AND the pair must not deadlock.
///
/// This reproduces the tenant-4 shared-link in miniature on ONE memtile: an
/// S2MM producer (ch0) fills buf0/buf1, an MM2S consumer (ch6) drains them,
/// sharing own locks 0 (free-sem, init 2) and 1 (full-sem, init 0). The
/// consumer is throttled (downstream drained slowly) so it is the bottleneck.
/// Functional lock releases fire inline at completion, so the pair never wedges;
/// the deferred memtile-S2MM full-release TRACE events emit at BD-slot reuse, so
/// the backpressured reuse's trace couples to the consumer's free-release at the
/// swap (asserted below on the timestamp-sorted events).
///
/// Without the deferral this would either stagger the trace events (old
/// inline model) or, with deferral but no break, deadlock -- caught here as a
/// bounded cycle-budget panic, never a box hang.
#[test]
fn memtile_producer_consumer_releases_couple_at_swap_no_deadlock() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    // Own window = 0x80000. buf0 @ 0x81000, buf1 @ 0x82000, 16 bytes (4 words).
    // free-sem = own lock 0 (id 64), full-sem = own lock 1 (id 65).
    // Negative acquire value = AcquireGreaterEqual (free-sem starts at 2).
    const FREE: u8 = 64;
    const FULL: u8 = 65;
    let buf0 = 0x81000u64;
    let buf1 = 0x82000u64;

    // Producer S2MM ch0: fill buf0, buf1, buf0 (3 BDs, chained).
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FREE, -1)
                .with_release(FULL, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(
            1,
            BdConfig::simple_1d(buf1, 16)
                .with_acquire(FREE, -1)
                .with_release(FULL, 1)
                .with_next(2),
        )
        .unwrap();
    engine
        .configure_bd(2, BdConfig::simple_1d(buf0, 16).with_acquire(FREE, -1).with_release(FULL, 1))
        .unwrap();

    // Consumer MM2S ch6: drain buf0, buf1, buf0 (3 BDs, chained).
    engine
        .configure_bd(
            3,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FULL, -1)
                .with_release(FREE, 1)
                .with_next(4),
        )
        .unwrap();
    engine
        .configure_bd(
            4,
            BdConfig::simple_1d(buf1, 16)
                .with_acquire(FULL, -1)
                .with_release(FREE, 1)
                .with_next(5),
        )
        .unwrap();
    engine
        .configure_bd(5, BdConfig::simple_1d(buf0, 16).with_acquire(FULL, -1).with_release(FREE, 1))
        .unwrap();

    tile.locks[0].set(2); // free-sem: two empty buffers
    tile.locks[1].set(0); // full-sem: nothing filled yet

    engine.start_channel(0, 0).unwrap(); // producer S2MM
    engine.start_channel(6, 3).unwrap(); // consumer MM2S

    // 12 input words: buf0=[0x100..], buf1=[0x200..], buf0(2nd)=[0x300..].
    let feed: Vec<u32> = (0..12u32).map(|i| 0x1000_0000 + i).collect();
    let mut fed = 0usize;
    let mut drained: Vec<u32> = Vec::new();
    let mut events: Vec<(u64, EventType)> = Vec::new();

    let mut cycle: u64 = 0;
    let max_cycles: u64 = 3000;
    loop {
        engine.set_current_cycle(cycle);

        // Feed producer stream_in (ch0) greedily -- producer is the fast side.
        while fed < feed.len()
            && engine.push_stream_in(StreamData { data: feed[fed], tlast: false, channel: 0 })
        {
            fed += 1;
        }

        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        // Lock releases land in the tile's memory-module trace queue (the
        // pipelined inline path emits LockRelease there, matching the arbiter
        // path); drain it each cycle so we capture release timing.
        events.extend(tile.mem_trace_pending.drain(..));

        // Drain consumer stream_out SLOWLY: one word every 5 cycles. This is the
        // downstream backpressure that makes the consumer the bottleneck.
        if cycle % 5 == 0 {
            if let Some(w) = engine.pop_stream_out() {
                drained.push(w.data);
            }
        }

        let done = !engine.channel_has_pending_work(0) && !engine.channel_has_pending_work(6);
        if done && engine.stream_out_len() == 0 && drained.len() == feed.len() {
            break;
        }
        cycle += 1;
        assert!(
            cycle < max_cycles,
            "producer/consumer did not complete within {} cycles -- likely a release-coupling deadlock \
             (fed={}, drained={}, prod={:?}, cons={:?})",
            max_cycles,
            fed,
            drained.len(),
            engine.channel_fsm_description(0),
            engine.channel_fsm_description(6),
        );
    }

    // Data integrity: consumer output is the fed stream, in order.
    assert_eq!(drained, feed, "consumer output must equal the produced stream in order");

    // Flush the tail: the memtile release latency can leave the final deferred
    // releases pending after the data work is done. Step the (now idle) engine
    // until the trace queue stops producing, so we capture every release.
    for _ in 0..(engine.timing_config().memtile_lock_release_latency_cycles as u64 + 8) {
        cycle += 1;
        engine.set_current_cycle(cycle);
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        events.extend(tile.mem_trace_pending.drain(..));
    }

    // Release coupling: full-sem releases (producer, local lock 1) and free-sem
    // releases (consumer, local lock 0). Sort by cycle: the deferred trace
    // events emit when their BD slot recycles (or at channel-idle for slots that
    // never recycle), so they may be *pushed* out of timestamp order -- exactly
    // like the real trace pipeline, where the decoder sorts the buffer by
    // timestamp. Compare on the timestamps, not the emission order.
    let mut full_releases: Vec<u64> = events
        .iter()
        .filter_map(|(c, e)| matches!(e, EventType::LockRelease { lock_id: 1 }).then_some(*c))
        .collect();
    let mut free_releases: Vec<u64> = events
        .iter()
        .filter_map(|(c, e)| matches!(e, EventType::LockRelease { lock_id: 0 }).then_some(*c))
        .collect();
    full_releases.sort_unstable();
    free_releases.sort_unstable();

    assert_eq!(full_releases.len(), 3, "expected 3 full-sem releases, got {:?}", full_releases);
    assert_eq!(free_releases.len(), 3, "expected 3 free-sem releases, got {:?}", free_releases);

    // The producer's THIRD fill reuses buf0, so it cannot signal "full" until
    // the consumer has freed buf0. That backpressured full-release couples to
    // the consumer's FIRST free-release (buf0) at the swap: it is the earliest
    // full-release (its reuse completes only once the consumer frees buf0),
    // coincident with the consumer's earliest free.
    let prod_full_reuse = full_releases[0] as i64;
    let cons_free_1st = free_releases[0] as i64;
    assert!(
        (prod_full_reuse - cons_free_1st).abs() <= 5,
        "backpressured full-release (cycle {}) must couple to the consumer's free-release (cycle {}) at \
         the swap; full={:?} free={:?}",
        prod_full_reuse,
        cons_free_1st,
        full_releases,
        free_releases,
    );
}

/// Single-buffer (no double-buffering) producer/consumer must not deadlock.
///
/// With ONE shared buffer the producer and consumer strictly ping-pong: every
/// fill's full-release and every drain's free-release is cross-lock and DEFERRED
/// to the swap, and the swap for each can only fire after the *other* channel
/// acts. That makes every handoff a mutual block that the deadlock-break flush
/// must resolve -- there is no slack buffer to hide behind (unlike the depth-2
/// case, where the second buffer lets one side run ahead). This is the
/// `add_256_using_dma_op_no_double_buffering` topology distilled to two memtile
/// channels; a regression in the functional-release path wedges it. Bounded so a
/// wedge fails as a cycle-budget panic, never a box hang.
#[test]
fn memtile_single_buffer_producer_consumer_no_deadlock() {
    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    const FREE: u8 = 64;
    const FULL: u8 = 65;
    let buf0 = 0x81000u64;

    // Producer S2MM ch0: fill buf0 three times (single buffer, chained).
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FREE, -1)
                .with_release(FULL, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(
            1,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FREE, -1)
                .with_release(FULL, 1)
                .with_next(2),
        )
        .unwrap();
    engine
        .configure_bd(2, BdConfig::simple_1d(buf0, 16).with_acquire(FREE, -1).with_release(FULL, 1))
        .unwrap();

    // Consumer MM2S ch6: drain buf0 three times (single buffer, chained).
    engine
        .configure_bd(
            3,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FULL, -1)
                .with_release(FREE, 1)
                .with_next(4),
        )
        .unwrap();
    engine
        .configure_bd(
            4,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FULL, -1)
                .with_release(FREE, 1)
                .with_next(5),
        )
        .unwrap();
    engine
        .configure_bd(5, BdConfig::simple_1d(buf0, 16).with_acquire(FULL, -1).with_release(FREE, 1))
        .unwrap();

    tile.locks[0].set(1); // free-sem: ONE empty buffer
    tile.locks[1].set(0); // full-sem: nothing filled yet

    engine.start_channel(0, 0).unwrap(); // producer S2MM
    engine.start_channel(6, 3).unwrap(); // consumer MM2S

    let feed: Vec<u32> = (0..12u32).map(|i| 0x2000_0000 + i).collect();
    let mut fed = 0usize;
    let mut drained: Vec<u32> = Vec::new();

    let mut cycle: u64 = 0;
    let max_cycles: u64 = 3000;
    loop {
        engine.set_current_cycle(cycle);
        while fed < feed.len()
            && engine.push_stream_in(StreamData { data: feed[fed], tlast: false, channel: 0 })
        {
            fed += 1;
        }
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        let _ = tile.mem_trace_pending.drain(..);
        if let Some(w) = engine.pop_stream_out() {
            drained.push(w.data);
        }

        let done = !engine.channel_has_pending_work(0) && !engine.channel_has_pending_work(6);
        if done && engine.stream_out_len() == 0 && drained.len() == feed.len() {
            break;
        }
        cycle += 1;
        assert!(
            cycle < max_cycles,
            "single-buffer producer/consumer wedged at {} cyc -- functional-release/flush regression \
             (fed={}, drained={}, prod={:?}, cons={:?})",
            max_cycles,
            fed,
            drained.len(),
            engine.channel_fsm_description(0),
            engine.channel_fsm_description(6),
        );
    }

    assert_eq!(drained, feed, "single-buffer consumer output must equal the produced stream in order");
}

/// Memtile S2MM cross-lock release pipeline latency: even when the next BD's
/// acquire is immediately grantable (warmup, buffers free), the release does
/// not land at BD completion -- it lands `memtile_lock_release_latency_cycles`
/// later. HW (tenant-4 probe) shows the producer's full-sem release trailing
/// FINISHED_BD by ~63 cycles on the prompt/warmup releases; the emulator's old
/// inline model fired it at +0. The latency is an S2MM (write/fill) effect --
/// the write pipeline must drain before the buffer is observably full -- so it
/// is exercised on an S2MM channel here.
#[test]
fn memtile_cross_lock_release_lands_after_pipeline_latency() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    // Memtile S2MM ch0: BD0 fills Own buf, releases own lock 2 (cross-lock --
    // next BD acquires own lock 1, a different lock), chains to BD1 which ends
    // the task. Own locks are ids 64+local. Both acquire locks are available so
    // both BDs run prompt -- no buffer backpressure, so the only thing delaying
    // BD0's release is the pipeline latency.
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(0x80000, 16)
                .with_acquire(64, 1)
                .with_release(66, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(1, BdConfig::simple_1d(0x80100, 16).with_acquire(65, 1).with_release(67, 1))
        .unwrap();

    tile.locks[0].set(1); // own lock 0 (id 64): BD0 acquire available
    tile.locks[1].set(1); // own lock 1 (id 65): BD1 acquire available (prompt)

    engine.start_channel(0, 0).unwrap();

    let latency = engine.timing_config().memtile_lock_release_latency_cycles as u64;
    let mut finished_bd0: Option<u64> = None;
    let mut release_lock2: Option<u64> = None;

    // Run a fixed window past the release latency so the deferred release is
    // serviced even after the (short) transfers leave the channel idle. Feed
    // S2MM stream_in greedily so the fills aren't stream-starved.
    let feed: Vec<u32> = (0..8u32).collect();
    let mut fed = 0usize;
    for cycle in 0..(latency + 60) {
        engine.set_current_cycle(cycle);
        while fed < feed.len()
            && engine.push_stream_in(StreamData { data: feed[fed], tlast: false, channel: 0 })
        {
            fed += 1;
        }
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        for (c, e) in engine.drain_trace_events() {
            if matches!(e, EventType::DmaFinishedBd { channel: 0 }) && finished_bd0.is_none() {
                finished_bd0 = Some(c);
            }
        }
        for (c, e) in tile.mem_trace_pending.drain(..) {
            if matches!(e, EventType::LockRelease { lock_id: 2 }) && release_lock2.is_none() {
                release_lock2 = Some(c);
            }
        }
    }

    let finished = finished_bd0.expect("BD0 should emit FINISHED_BD");
    let released = release_lock2.expect("BD0's cross-lock release (own lock 2) should land");
    assert_eq!(
        released - finished,
        latency,
        "prompt cross-lock release must trail FINISHED_BD by the pipeline latency ({} cyc); \
         finished={} released={}",
        latency,
        finished,
        released,
    );
}

/// The memtile lock-release pipeline latency is an OBSERVABILITY effect, not a
/// functional DMA stall. The FUNCTIONAL semaphore (what a waiting consumer
/// acquires) must release PROMPTLY at the buffer swap; only the LockRelease
/// *trace event* lags by the latency.
///
/// HW evidence (tenant-4 producer-fill probe): the producer's warmup fills land
/// at exactly the shim's 65-cyc/buffer cadence -- back-to-back, no extra stall.
/// That is only possible if the consumer-free path is OFF the critical path,
/// i.e. the full-sem release is functionally prompt. If the latency gated the
/// semaphore, the consumer drain (and thus the producer's buffer reuse) would
/// stall ~63 cyc and the fills would not stay shim-paced. The +63 appears only
/// in the trace (LOCK_SEL1_REL trailing FINISHED_BD), so it must be applied to
/// the trace event alone, not the lock value.
///
/// This pins the two halves apart: the lock VALUE increments at the swap (a few
/// chain cycles after FINISHED_BD), while the trace event lands at +latency
/// (asserted by `memtile_cross_lock_release_lands_after_pipeline_latency`).
#[test]
fn memtile_cross_lock_release_frees_consumer_promptly_trace_lags() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    // Same shape as the latency test: S2MM ch0, BD0 fills + cross-lock releases
    // own lock 2 (id 66), chains to BD1 which acquires own lock 1 (id 65, a
    // DIFFERENT lock -> cross-lock defer) and ends. Both acquires are available
    // so the swap (BD1's acquire grant) is prompt -- nothing but the latency
    // could delay the release.
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(0x80000, 16)
                .with_acquire(64, 1)
                .with_release(66, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(1, BdConfig::simple_1d(0x80100, 16).with_acquire(65, 1).with_release(67, 1))
        .unwrap();

    tile.locks[0].set(1); // own lock 0 (id 64): BD0 acquire available
    tile.locks[1].set(1); // own lock 1 (id 65): BD1 acquire available (prompt swap)
                          // own lock 2 (id 66, the full-sem BD0 releases) starts at 0.
    assert_eq!(tile.locks[2].value, 0, "full-sem must start empty");

    engine.start_channel(0, 0).unwrap();

    let latency = engine.timing_config().memtile_lock_release_latency_cycles as u64;
    let mut finished_bd0: Option<u64> = None;
    let mut functional_release: Option<u64> = None; // first cycle lock 2 value > 0
    let mut trace_release: Option<u64> = None; // LockRelease{2} trace event cycle

    let feed: Vec<u32> = (0..8u32).collect();
    let mut fed = 0usize;
    for cycle in 0..(latency + 60) {
        engine.set_current_cycle(cycle);
        while fed < feed.len()
            && engine.push_stream_in(StreamData { data: feed[fed], tlast: false, channel: 0 })
        {
            fed += 1;
        }
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        for (c, e) in engine.drain_trace_events() {
            if matches!(e, EventType::DmaFinishedBd { channel: 0 }) && finished_bd0.is_none() {
                finished_bd0 = Some(c);
            }
        }
        for (c, e) in tile.mem_trace_pending.drain(..) {
            if matches!(e, EventType::LockRelease { lock_id: 2 }) && trace_release.is_none() {
                trace_release = Some(c);
            }
        }
        // Read the FUNCTIONAL lock value AFTER the step: the semaphore that a
        // waiting consumer would acquire.
        if functional_release.is_none() && tile.locks[2].value > 0 {
            functional_release = Some(cycle);
        }
    }

    let finished = finished_bd0.expect("BD0 should emit FINISHED_BD");
    let functional = functional_release.expect("BD0's cross-lock release must land on the lock value");
    let trace = trace_release.expect("BD0's cross-lock release must emit a trace event");

    // FUNCTIONAL release is prompt: the lock becomes available at the swap, a
    // few chain cycles after FINISHED_BD -- NOT delayed by the pipeline latency.
    // A generous grace (well under `latency`) cleanly separates "prompt at swap"
    // from "gated by the 63-cyc latency".
    let grace = (engine.timing_config().chain_cycles() + 8).min(latency / 2);
    assert!(
        functional.saturating_sub(finished) <= grace,
        "functional full-sem release must be PROMPT at the swap (<= {} cyc after FINISHED_BD), not \
         gated by the {}-cyc trace latency; finished={} functional_release={}",
        grace,
        latency,
        finished,
        functional,
    );

    // The TRACE event still lags by the full pipeline latency (HW observability).
    assert_eq!(
        trace - finished,
        latency,
        "LockRelease trace event must trail FINISHED_BD by the pipeline latency ({} cyc); \
         finished={} trace={}",
        latency,
        finished,
        trace,
    );
}

/// End-of-stream release tail: the producer's FINAL full-release trace fires on
/// the consumer-free cadence, even after the producer stream-stalls and never
/// re-acquires that slot.
///
/// HW evidence (tenant-4 producer probe): the 8th LOCK_SEL1_REL fires ~2 periods
/// after the 8th FINISHED_BD, right where the next consumer-free lands -- with NO
/// 9th LOCK_SEL0_ACQ after it (the producer is stalled on exhausted input). So
/// the deferred full-release is gated on the SWAP-enable (the next buffer
/// becoming FREE = a consumer-free event), not on the producer's actual
/// re-acquire grant. The emulator gates on the re-acquire grant, so a trailing
/// fill whose slot is never re-acquired at end-of-stream drops its release (the
/// probe's 7-of-8 tail).
///
/// This pins the gate: a producer that has stream-stalled with a trailing pending
/// release must emit it when its acquire lock (FREE) is incremented by the
/// consumer -- at the consumer-free cadence -- not drop it.
#[test]
fn memtile_stalled_producer_emits_trailing_release_on_consumer_free() {
    use crate::interpreter::state::EventType;

    let mut engine = DmaEngine::new_mem_tile(1, 1);
    let mut tile = Tile::mem_tile(1, 1);
    let mut host_mem = make_host_memory();

    const FREE: u8 = 64; // local lock 0
    const FULL: u8 = 65; // local lock 1 (the producer releases this -> LOCK_SEL1_REL)
    let buf0 = 0x80000u64;
    let buf1 = 0x80100u64;

    // Looping double buffer: bd0 -> bd1 -> bd0 -> ... each fill acquires FREE,
    // releases FULL (cross-lock -> deferred trace).
    engine
        .configure_bd(
            0,
            BdConfig::simple_1d(buf0, 16)
                .with_acquire(FREE, -1)
                .with_release(FULL, 1)
                .with_next(1),
        )
        .unwrap();
    engine
        .configure_bd(
            1,
            BdConfig::simple_1d(buf1, 16)
                .with_acquire(FREE, -1)
                .with_release(FULL, 1)
                .with_next(0),
        )
        .unwrap();

    // FREE high enough that fills are not acquire-backpressured: the producer
    // stream-stalls on exhausted input (not on the lock), matching the probe.
    tile.locks[0].set(6);
    tile.locks[1].set(0);

    engine.start_channel(0, 0).unwrap();

    // Feed exactly N buffers, then stop -> the (N+1)th fill stream-stalls.
    let n_bufs = 4usize;
    let feed: Vec<u32> = (0..(n_bufs as u32 * 4)).collect();
    let mut fed = 0usize;
    let mut events: Vec<(u64, EventType)> = Vec::new();

    // Run until the producer is stream-stalled (fed everything, no forward
    // progress for a while).
    let mut cycle = 0u64;
    let mut stall_since: Option<u64> = None;
    loop {
        engine.set_current_cycle(cycle);
        while fed < feed.len()
            && engine.push_stream_in(StreamData { data: feed[fed], tlast: false, channel: 0 })
        {
            fed += 1;
        }
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        events.extend(tile.mem_trace_pending.drain(..));

        let stalled = fed == feed.len() && !engine.channel_active(0)
            || (fed == feed.len() && engine.channel_fsm_description(0).starts_with("Transferring"));
        if stalled {
            stall_since.get_or_insert(cycle);
            if cycle - stall_since.unwrap() > 80 {
                break;
            }
        }
        cycle += 1;
        assert!(cycle < 2000, "producer never reached steady stream-stall");
    }

    let before = events
        .iter()
        .filter(|(_, e)| matches!(e, EventType::LockRelease { lock_id: 1 }))
        .count();
    assert!(
        before < n_bufs,
        "precondition: a trailing release should be undelivered while stalled (got {} of {})",
        before,
        n_bufs
    );

    // Simulate the consumer freeing the last buffer: increment FREE. On HW this
    // is what fires the trailing LOCK_SEL1_REL.
    let _ = tile.locks[0].release_with_value(1);
    for _ in 0..40 {
        cycle += 1;
        engine.set_current_cycle(cycle);
        engine.submit_lock_requests(&mut tile, &mut NeighborTiles::empty());
        tile.resolve_lock_requests(cycle);
        engine.step(&mut tile, &mut NeighborTiles::empty(), &mut host_mem);
        events.extend(tile.mem_trace_pending.drain(..));
    }

    let after = events
        .iter()
        .filter(|(_, e)| matches!(e, EventType::LockRelease { lock_id: 1 }))
        .count();
    assert!(
        after > before,
        "a consumer-free (FREE increment) while the producer is stream-stalled must emit the \
         trailing full-release; before={} after={} (of {})",
        before,
        after,
        n_bufs
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

// === Phase-ordering fix: registered-FIFO invariant (#140) ===

#[test]
fn phase_ordering_blocks_routing_when_ingress_was_full() {
    // Regression test for the produce-before-consume phase-ordering race (#140).
    //
    // On real HW, the S2MM ingress is a registered FIFO: the TREADY signal the
    // producer sees in cycle N is determined by the ingress state at the END of
    // cycle N-1. If the ingress was full at end of cycle N-1, TREADY=0 in cycle
    // N even if the consumer drains D words during cycle N itself.
    //
    // EMU's step_data_movement runs Phase 3 (step_all_dma, S2MM consumes) BEFORE
    // Phase 4 (route_streams, producer fills). Without the fix, Phase 4's
    // can_accept check sees post-drain occupancy (len < cap after drain freed D
    // slots) and allows the producer to push D extra words in the same cycle.
    // That is the "head-of-stream race" that causes EMU to emit three 16-word
    // BDs back-to-back where HW throttles after the first.
    //
    // Fix: can_accept_stream_in_for_routing tracks stream_in_drained_this_cycle
    // (reset each cycle before Phase 3, incremented in pop_stream_in_for_channel)
    // and checks current + drained < capacity. A slot freed in Phase 3 is NOT
    // available to the producer in Phase 4 of the same cycle.
    //
    // Test parameters (scaled from add_one_using_dma memtile-to-compute path):
    //   consumer_buffer = DMA_S2MM_INGRESS_FIFO_DEPTH = 16
    //   crossing_depth  = 0 (same-tile; inter-tile crossing is a separate bound)
    //   drain_per_step  = 1 (simulated Phase 3 drains 1 word)
    //   Expected head run before first stall: consumer_buffer + crossing_depth = 16
    use xdna_archspec::aie2::timing::DMA_S2MM_INGRESS_FIFO_DEPTH;
    let cap = DMA_S2MM_INGRESS_FIFO_DEPTH as usize;

    let mut engine = DmaEngine::new_compute_tile(1, 2);

    // Fill ingress to capacity using push_stream_in WITHOUT starting a task.
    // With no task running, accept_bd is None so advance_accept_cursor is a
    // no-op -- bd_switch_accept_block stays 0. This isolates the capacity-
    // based phase-ordering invariant from the per-BD-boundary deassert.
    for i in 0..(cap as u32) {
        let ok = engine.push_stream_in(StreamData { data: i, tlast: false, channel: 0 });
        assert!(ok, "pre-fill push {i} should succeed (ingress has room)");
    }
    assert_eq!(engine.stream_in_len(), cap, "ingress must be at capacity ({cap}) before the cycle starts");

    // Phase 2.5 (reset): mirrors step_data_movement resetting drain counters
    // before Phase 3.
    engine.reset_cycle_drain_counters();

    // Phase 3 (simulated): drain 1 word, as step_all_dma / transfer_s2mm does
    // via pop_stream_in_for_channel. This increments stream_in_drained_this_cycle.
    let drained = engine.pop_stream_in_for_channel(0);
    assert!(drained.is_some(), "drain must succeed from a full ingress");
    assert_eq!(engine.stream_in_len(), cap - 1, "ingress must have cap-1 words after one drain");

    // Phase 4 (simulated): can_accept_stream_in_for_routing must return false.
    // Effective start-of-cycle occupancy = (cap - 1) + 1 = cap, which is NOT
    // less than cap. Before the fix, route_tile_switches_to_dma called
    // can_accept_stream_in_for_channel instead, which returned true here
    // (physical len = cap-1 < cap), allowing the producer to race ahead.
    assert!(
        !engine.can_accept_stream_in_for_routing(0),
        "Phase 4 routing must NOT accept: effective occupancy = {} + 1 = {} >= cap {}",
        cap - 1,
        cap,
        cap,
    );

    // Sanity: the legacy per-channel check still returns true (physical room
    // exists); only the routing-phase method enforces start-of-cycle occupancy.
    // This ensures the fix does not regress the DMA engine's own internal checks.
    assert!(
        engine.can_accept_stream_in_for_channel(0),
        "can_accept_stream_in_for_channel must still return true (physical len < cap); \
         only the routing-phase method accounts for start-of-cycle occupancy"
    );
}

#[test]
fn unstarted_s2mm_channel_reads_as_unstarted_until_task_enqueued() {
    // #140 SP-4a terminal-send: an S2MM channel with no dispatched or queued
    // task (Idle, empty task queue) asserts no stream readiness -- the routing
    // accept path (route_tile_switches_to_dma) must not deliver stream data to
    // it. This models the shim S2MM (an objectfifo DDR drain), which on HW gets
    // no static CDO config and stays Idle until the runtime `npu_dma_memcpy_nd`
    // dispatches it; before the fix the emulator pre-filled the terminal
    // memtile->shim send into that idle channel's fabric.
    //
    // `channel_is_started` (has_pending_work) must be false for a fresh channel
    // and true once a task is enqueued (the runtime dispatch). It is distinct
    // from `channel_active` (FSM-only): a just-enqueued, not-yet-stepped task
    // counts as started.
    let mut engine = DmaEngine::new_mem_tile(0, 1);
    assert!(
        !engine.channel_is_started(0),
        "a fresh S2MM channel (Idle, empty task queue) must read as unstarted"
    );
    // Interior CDO-started channels are the safety case: once a BD is configured
    // and the channel started (as CDO Start_Queue / a runtime dispatch does), it
    // is live and routing may deliver to it.
    engine.configure_bd(0, BdConfig::simple_1d(0x1000, 256)).unwrap();
    engine.start_channel(0, 0).unwrap();
    assert!(engine.channel_is_started(0), "after configure + start (dispatch), the channel reads as started");
}
