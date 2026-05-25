//! Integration tests for clock-control execution gating.
//!
//! These tests drive the full path from `DeviceState::write_tile_register`
//! (the canonical register-bus entry) through `step_data_movement`,
//! exercising real gate behavior end-to-end.  They complement the
//! per-task unit tests above by driving the system at the DeviceState
//! level rather than poking the `ClockController` API directly.
//!
//! Setup pattern: configure a real DMA workload (BD + start_channel)
//! via the DMA engine API, set the column-gate state via
//! `write_tile_register` to Column_Clock_Control (0x000FFF20), then
//! step the array and observe whether the DMA channel advanced.
//!
//! Spec: docs/superpowers/specs/2026-05-24-clock-control-design.md
//! § Integration.

use crate::device::DeviceState;
use crate::device::dma::BdConfig;
use crate::device::host_memory::HostMemory;

/// AM025 offset for Column_Clock_Control (shim tiles only).
const COLUMN_CLOCK_CONTROL_OFFSET: u32 = 0x000F_FF20;

/// Configure a minimal MM2S DMA workload on the given compute tile.
///
/// Loads 64 bytes of source data into the tile's data memory at 0x100,
/// configures BD 0 for a 64-byte 1D MM2S transfer, and starts MM2S ch0
/// (absolute channel index 2 on a compute tile: S2MM_0, S2MM_1,
/// MM2S_0, MM2S_1).
fn arm_compute_mm2s_workload(state: &mut DeviceState, col: u8, row: u8) {
    // Seed the tile's data memory so MM2S has something to read.
    let tile = state.tile_mut(col as usize, row as usize).expect("compute tile exists");
    for i in 0..64usize {
        tile.data_memory_mut()[0x100 + i] = i as u8;
    }

    // Configure BD 0 for a 64-byte 1D transfer from local addr 0x100.
    let bd = BdConfig::simple_1d(0x100, 64);
    let dma = state.array.dma_engine_mut(col, row).expect("compute tile has DMA");
    dma.configure_bd(0, bd).unwrap();
    // Absolute channel index 2 = MM2S ch0 (after S2MM_0, S2MM_1).
    dma.start_channel(2, 0).unwrap();
}

/// Drain MM2S stream-out FIFO on (col, row) so backpressure does not
/// stall the channel during `step_data_movement` (which routes streams
/// but only between configured stream-switch endpoints; with no route
/// programmed the MM2S output piles up until it caps).  Mirrors the
/// drain loop in `npu::executor::tests::test_sync_requires_channel_started`.
fn drain_stream_out(state: &mut DeviceState, col: u8, row: u8) {
    if let Some(dma) = state.array.dma_engine_mut(col, row) {
        while dma.pop_stream_out().is_some() {}
    }
}

#[test]
fn gated_column_produces_no_dma_progress() {
    // Fresh DeviceState: every column gated (silicon-accurate boot).
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let col: u8 = 2;
    let row: u8 = 2; // compute tile

    arm_compute_mm2s_workload(&mut state, col, row);

    // Sanity: the gate is engaged through the canonical query path.
    assert!(!state.array.clock().is_column_active(col), "col {} should boot gated", col);

    // Step the array for a generous number of cycles.  step_data_movement
    // is the production entry point for DMA + stream advancement; a gated
    // column must produce no progress on any of its modules.
    for _ in 0..200 {
        let (dma_active, streams_active, words_routed) = state.array.step_data_movement(&mut host);
        assert!(!dma_active, "no DMA must advance while col {} is gated", col);
        assert!(!streams_active, "no stream activity while col {} is gated", col);
        assert_eq!(words_routed, 0, "no words should route while col {} is gated", col);
        drain_stream_out(&mut state, col, row);
    }

    // Direct DMA-engine inspection confirms no transfer completed.
    let stats = state.array.dma_engine(col, row).unwrap().channel_stats(2).unwrap();
    assert_eq!(stats.transfers_completed, 0, "no transfer should complete on a gated column");
    assert_eq!(stats.bytes_transferred, 0, "no bytes should move on a gated column");
}

#[test]
fn ungate_via_write_tile_register_enables_dma_progress() {
    // Same workload as the gated case, but route an ungate through the
    // canonical register-bus entry point first.
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let col: u8 = 2;
    let row: u8 = 2;

    arm_compute_mm2s_workload(&mut state, col, row);

    // Ungate column 2 by writing Column_Clock_Control bit 0 = 1 through
    // the dispatcher.  This is the same path the CDO uses; the gate is
    // observable via the canonical query.
    state.write_tile_register(col, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
    assert!(
        state.array.clock().is_column_active(col),
        "col {} must be active after write_tile_register",
        col
    );

    // Step until the channel finishes (transfers_completed > 0) or budget
    // runs out.  A 64-byte transfer completes in well under 200 cycles
    // when stream-out is drained continuously.
    let mut completed = false;
    for _ in 0..2_000 {
        state.array.step_data_movement(&mut host);
        drain_stream_out(&mut state, col, row);
        let stats = state.array.dma_engine(col, row).unwrap().channel_stats(2).unwrap();
        if stats.transfers_completed > 0 {
            completed = true;
            break;
        }
    }
    assert!(completed, "DMA transfer should complete once column {} is ungated", col);

    let stats = state.array.dma_engine(col, row).unwrap().channel_stats(2).unwrap();
    assert!(stats.bytes_transferred > 0, "ungated column should move bytes; got {}", stats.bytes_transferred);
}

#[test]
fn mixed_column_gating_is_per_column_selective() {
    // NPU1 has 5 columns (0-4).  Ungate cols 0-3 via the dispatcher;
    // leave col 4 gated.  Identical workloads on col 1 (ungated) and
    // col 4 (gated) must produce divergent results.
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let active_col: u8 = 1;
    let gated_col: u8 = 4;
    let row: u8 = 2; // compute row used on both columns

    arm_compute_mm2s_workload(&mut state, active_col, row);
    arm_compute_mm2s_workload(&mut state, gated_col, row);

    // Ungate cols 0..=3, leave col 4 gated.
    for col in 0..=3u8 {
        state.write_tile_register(col, 0, COLUMN_CLOCK_CONTROL_OFFSET, 0x1);
    }

    // Verify the dispatcher landed the gates exactly where intended.
    for col in 0..=3u8 {
        assert!(state.array.clock().is_column_active(col), "col {} should be active", col);
    }
    assert!(!state.array.clock().is_column_active(gated_col), "col {} should remain gated", gated_col);

    // Step the array; drain stream-out on both armed columns so neither
    // backpressure-stalls (the gated one wouldn't progress anyway, but
    // the drain is harmless and keeps the helper symmetric).
    for _ in 0..2_000 {
        state.array.step_data_movement(&mut host);
        drain_stream_out(&mut state, active_col, row);
        drain_stream_out(&mut state, gated_col, row);
        let stats = state.array.dma_engine(active_col, row).unwrap().channel_stats(2).unwrap();
        if stats.transfers_completed > 0 {
            break;
        }
    }

    let active_stats = state.array.dma_engine(active_col, row).unwrap().channel_stats(2).unwrap();
    let gated_stats = state.array.dma_engine(gated_col, row).unwrap().channel_stats(2).unwrap();

    assert!(active_stats.transfers_completed > 0, "ungated col {} should complete its DMA", active_col);
    assert_eq!(
        gated_stats.transfers_completed, 0,
        "gated col {} must not advance even when neighbors run",
        gated_col
    );
    assert_eq!(gated_stats.bytes_transferred, 0, "gated col {} must move zero bytes", gated_col);
}

#[test]
fn adaptive_dma_counter_advances_via_step_data_movement() {
    // Verify that tick_adaptive_dma is called from step_data_movement by
    // confirming the adaptive DMA gate engages after enough idle cycles.
    //
    // Setup: ungate all columns/modules, set abort_period = 3 (engages
    // after 2^3 = 8 idle cycles) on a compute tile, step 10 times with
    // no DMA traffic.  The adaptive gate must be engaged at the end.
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let col: u8 = 2;
    let row: u8 = 3; // compute tile (row >= 2)

    // Ungate all so adaptive counters can advance.
    state.array.clock_mut().ungate_all();

    // Set abort_period = 3 on our target tile.
    state.array.clock_mut().set_adaptive_abort_period(col, row, 3);

    // Step 10 times with no DMA traffic.  No BDs are configured, so every
    // DMA step is idle.  The adaptive counter must reach 2^3 = 8 and engage.
    for _ in 0..10 {
        state.array.step_data_movement(&mut host);
    }

    assert!(
        state.array.clock().is_adaptive_dma_engaged(col, row),
        "adaptive DMA gate must engage after 10 idle step_data_movement cycles \
         with abort_period=3 (threshold=8)"
    );
}

#[test]
fn adaptive_dma_gate_releases_when_task_queued_during_idle() {
    // Reproduces the ctrl_packet_reconfig deadlock: tile sits idle long enough
    // for the adaptive DMA gate to engage; control packet then enqueues a task
    // on an Idle-FSM channel. If the gate keeps engaging on subsequent ticks
    // (because the idle detector only watches FSM, not queued tasks),
    // step_all_dma skips the tile forever and the queued task never starts.
    //
    // Setup: same as adaptive_dma_counter_advances_via_step_data_movement
    // (ungate, abort_period=3, step until engaged), then enqueue a task
    // without taking another step.  One more step_data_movement must release
    // the gate -- the Phase-5 tick now sees pending work via
    // any_channel_has_pending_work() even with FSM=Idle.
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let col: u8 = 2;
    let row: u8 = 3;

    state.array.clock_mut().ungate_all();
    state.array.clock_mut().set_adaptive_abort_period(col, row, 3);

    // Idle until the adaptive gate engages.
    for _ in 0..10 {
        state.array.step_data_movement(&mut host);
    }
    assert!(
        state.array.clock().is_adaptive_dma_engaged(col, row),
        "pre-condition: gate must be engaged before we enqueue"
    );

    // Enqueue a task on MM2S ch0 (absolute index 2). FSM stays Idle until
    // the next step promotes Idle->BdSetup.
    {
        let dma = state.array.dma_engine_mut(col, row).expect("compute tile DMA");
        dma.configure_bd(0, BdConfig::simple_1d(0x100, 64)).unwrap();
        dma.enqueue_task(2, 0, 0, false);
    }

    // One step_data_movement: Phase 5 must observe pending work and reset
    // the counter, releasing the gate even though step_all_dma still
    // skipped this tile (gate was engaged at the time of step_all_dma).
    state.array.step_data_movement(&mut host);

    assert!(
        !state.array.clock().is_adaptive_dma_engaged(col, row),
        "gate must release after a queued task is observed by Phase 5; \
         otherwise step_all_dma deadlocks the task queue"
    );
}

// ---- Wake-on-event integration tests (Wake 1: register write) ----

/// Helper: ungate everything, set abort_period=3 on (col, row), idle until
/// both DMA and SS adaptive gates engage.  Caller then exercises a wake
/// path and asserts the corresponding gate has released.
fn idle_until_both_gates_engaged(state: &mut DeviceState, host: &mut HostMemory, col: u8, row: u8) {
    state.array.clock_mut().ungate_all();
    state.array.clock_mut().set_adaptive_abort_period(col, row, 3);
    for _ in 0..16 {
        state.array.step_data_movement(host);
    }
    assert!(state.array.clock().is_adaptive_dma_engaged(col, row), "precondition: DMA gate engaged");
    assert!(state.array.clock().is_adaptive_ss_engaged(col, row), "precondition: SS gate engaged");
}

#[test]
fn write_to_dma_bd_register_wakes_dma_adaptive_gate() {
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let (col, row) = (2u8, 3u8); // compute tile

    idle_until_both_gates_engaged(&mut state, &mut host, col, row);

    // 0x1D000 = compute DMA BD0 word 0.  Classified as SubsystemKind::Dma
    // by subsystem_from_offset.
    state.write_tile_register(col, row, 0x1D000, 0xDEAD_BEEF);

    assert!(
        !state.array.clock().is_adaptive_dma_engaged(col, row),
        "DMA register write must wake the DMA adaptive gate"
    );
    // SS counter is unrelated and must stay engaged.
    assert!(
        state.array.clock().is_adaptive_ss_engaged(col, row),
        "DMA register write must not affect SS adaptive gate"
    );
}

#[test]
fn write_to_lock_register_wakes_dma_adaptive_gate() {
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let (col, row) = (2u8, 3u8);

    idle_until_both_gates_engaged(&mut state, &mut host, col, row);

    // 0x1F000 = compute Lock0_value.  Classified as SubsystemKind::Lock.
    // Lock shares the Memory clock bit with DMA on compute, so a lock
    // write must wake the DMA adaptive counter.
    state.write_tile_register(col, row, 0x1F000, 0x0);

    assert!(
        !state.array.clock().is_adaptive_dma_engaged(col, row),
        "Lock register write must wake the DMA adaptive gate (shared Memory clock bit)"
    );
    assert!(
        state.array.clock().is_adaptive_ss_engaged(col, row),
        "Lock register write must not affect SS adaptive gate"
    );
}

#[test]
fn write_to_stream_switch_register_wakes_ss_adaptive_gate() {
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let (col, row) = (2u8, 3u8);

    idle_until_both_gates_engaged(&mut state, &mut host, col, row);

    // 0x3FF00 = compute stream switch master config base.
    state.write_tile_register(col, row, 0x3FF00, 0x0);

    assert!(
        !state.array.clock().is_adaptive_ss_engaged(col, row),
        "SS register write must wake the SS adaptive gate"
    );
    assert!(
        state.array.clock().is_adaptive_dma_engaged(col, row),
        "SS register write must not affect DMA adaptive gate"
    );
}

#[test]
fn write_to_clock_control_register_does_not_emit_wake() {
    // Clock-control writes are themselves the ungate mechanism and have
    // their own counter-reset logic in clock_control::write_register
    // (column-ungate / module-ungate transitions).  The wake-on-event
    // dispatcher path must NOT also wake them, because a clock-control
    // write to a *non-transitioning* offset (e.g., re-writing the same
    // value) should leave the counters alone.
    let mut state = DeviceState::new_npu1();
    let mut host = HostMemory::new();
    let (col, row) = (2u8, 3u8);

    // Ungate column 2 first so the rest of the test runs in the
    // ungated-counter regime.
    state.write_tile_register(col, 0, 0x000F_FF20, 0x1);
    // Now engage the gates.
    state.array.clock_mut().set_adaptive_abort_period(col, row, 3);
    for _ in 0..16 {
        state.array.step_data_movement(&mut host);
    }
    assert!(state.array.clock().is_adaptive_dma_engaged(col, row));
    assert!(state.array.clock().is_adaptive_ss_engaged(col, row));

    // Re-write Column_Clock_Control with the same value -- no transition.
    // Both counters must stay engaged: the dispatcher does NOT route this
    // through the wake path, and clock_control::write_register's reset
    // logic only fires on a 0->1 transition.
    state.write_tile_register(col, 0, 0x000F_FF20, 0x1);

    assert!(
        state.array.clock().is_adaptive_dma_engaged(col, row),
        "non-transitioning clock-control write must not wake gates"
    );
    assert!(
        state.array.clock().is_adaptive_ss_engaged(col, row),
        "non-transitioning clock-control write must not wake gates"
    );
}
