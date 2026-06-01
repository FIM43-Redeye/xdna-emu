//! NPU execution functions for the FFI interface.
//!
//! Instruction loading, cycle limits, and the main run loop.

use std::slice;

use xdna_emu_core::npu::NpuInstructionStream;

use super::{XdnaEmuHandle, XdnaEmuResult, XdnaEmuExecStatus, XdnaEmuHaltReason};

/// Execute NPU instructions.
///
/// This executes the instruction buffer which triggers DMA transfers
/// and configures the shim tiles.
///
/// # Safety
/// - `handle` must be valid
/// - `instr_data` must point to at least `instr_size` bytes
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_execute_npu_instructions(
    handle: *mut XdnaEmuHandle,
    instr_data: *const u8,
    instr_size: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if instr_data.is_null() && instr_size > 0 {
        return XdnaEmuResult::NullPointer;
    }

    let handle = &mut *handle;
    let instr_slice = slice::from_raw_parts(instr_data, instr_size as usize);

    // Parse instruction stream
    let stream = match NpuInstructionStream::parse(instr_slice) {
        Ok(s) => s,
        Err(e) => {
            log::error!("Failed to parse NPU instructions: {}", e);
            return XdnaEmuResult::ParseError;
        }
    };

    log::info!("Executing {} NPU instructions", stream.instructions().len());

    // Load instructions for interleaved execution in xdna_emu_run().
    handle.npu_executor.load(&stream);

    XdnaEmuResult::Success
}

/// Set maximum cycles for execution.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_set_max_cycles(
    handle: *mut XdnaEmuHandle,
    max_cycles: u64,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    handle.max_cycles = max_cycles;

    XdnaEmuResult::Success
}

/// Run the emulator until completion or max cycles.
///
/// Returns execution status including halt reason and cycle count.
///
/// When `max_cycles` is 0 (set via `xdna_emu_set_max_cycles`), execution is
/// unbounded — the emulator runs until cores halt naturally or syncs are
/// satisfied. A non-zero `max_cycles` caps execution; if the budget is
/// exhausted before natural completion, `halt_reason` is `Budget`.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_run(handle: *mut XdnaEmuHandle) -> XdnaEmuExecStatus {
    if handle.is_null() {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::InvalidHandle,
            cycles_executed: 0,
            halted: false,
            halt_reason: XdnaEmuHaltReason::Error,
        };
    }

    let handle = &mut *handle;

    if handle.backend.as_interpreter().is_none() {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::ExecutionError,
            cycles_executed: 0,
            halted: false,
            halt_reason: XdnaEmuHaltReason::Error,
        };
    }

    let mut cycles = 0u64;
    let max = handle.max_cycles;
    // max == 0 means unbounded: the loop runs until a natural exit point.
    let unbounded = max == 0;

    log::info!(
        "Running emulator (max {})",
        if unbounded {
            "unbounded".to_string()
        } else {
            format!("{} cycles", max)
        }
    );

    // Entry guard: context(0) must be Connected. Otherwise the caller
    // forgot to call xdna_emu_reset_context after a prior wedge.
    {
        use xdna_emu_core::device::context::DEFAULT_CONTEXT;
        let device = handle.backend.as_interpreter().expect("Plan A: interpreter backend").device();
        let ctx = &device.contexts[DEFAULT_CONTEXT.0 as usize];
        if !ctx.is_connected() {
            log::error!(
                "xdna_emu_run: context {:?} not Connected; \
                 caller must reset_context before re-submitting",
                DEFAULT_CONTEXT
            );
            return XdnaEmuExecStatus {
                result: XdnaEmuResult::ExecutionError,
                cycles_executed: 0,
                halted: false,
                halt_reason: XdnaEmuHaltReason::Error,
            };
        }
    }

    // Warm-up: let cores run to their first blocking point before
    // processing NPU instructions.  On real hardware, the core has been
    // running for thousands of cycles (CDO enables the core well before
    // NPU instructions arrive through firmware + NoC).  Without this,
    // maskwrite/blockwrite instructions modify tile memory before the
    // core's init loop has written its initial values.
    //
    // NOTE: warm-up cycles count against `max_cycles`. For small budgets
    // (e.g. `max_cycles=1` for smoke tests), the warm-up alone can exhaust
    // the budget and the run loop below won't execute a single cycle of
    // real work -- you'll see `halt_reason=Budget` with `cycles == warm-up
    // count`. This is acceptable for the cycle-budget use case (any real
    // test needs a budget much larger than 100k), but documented here so
    // a smoke-test-sized budget isn't mistaken for a "kernel did nothing"
    // signal.
    if handle
        .backend
        .as_interpreter()
        .expect("Plan A: interpreter backend")
        .enabled_cores()
        > 0
        && !handle.npu_executor.is_done()
    {
        const MAX_WARMUP: u64 = 100_000;
        while cycles < MAX_WARMUP {
            handle.backend.as_interpreter_mut().expect("Plan A: interpreter backend").step();
            cycles += 1;
            if handle
                .backend
                .as_interpreter()
                .expect("Plan A: interpreter backend")
                .all_cores_blocked()
            {
                break;
            }
        }
        log::info!("Core warm-up: {} cycles (all cores at first blocking point)", cycles);
    }

    // Per-cycle: build snapshots, classify, dispatch on verdict.
    use xdna_emu_core::device::tdr::{TdrVerdict, WedgeReason, TdrDiagnosis};
    use xdna_emu_core::device::context::DEFAULT_CONTEXT;

    let mut natural_halt = false;
    let mut maskpoll_unsatisfied = false;
    let mut wedged: Option<(WedgeReason, TdrDiagnosis)> = None;

    'run: while unbounded || cycles < max {
        // Publish the current simulation cycle to the tile array before the
        // NPU executor runs. Register-write side effects (trace unit
        // start/stop, broadcast propagation) driven by NPU instructions
        // read array.current_cycle to time-stamp events.
        handle
            .backend
            .as_interpreter_mut()
            .expect("Plan A: interpreter backend")
            .device_mut()
            .array
            .set_dma_cycle(cycles);

        // Advance NPU instruction execution (interleaved with engine step).
        let npu_progressed;
        {
            let (device, host_mem) = handle
                .backend
                .as_interpreter_mut()
                .expect("Plan A: interpreter backend")
                .device_and_host_memory();
            let result = handle.npu_executor.try_advance(device, host_mem);
            if let xdna_emu_core::npu::AdvanceResult::Error(msg) = result {
                log::error!("NPU executor fatal: {}", msg);
                // Flush trace before returning so partial data reaches DDR.
                handle
                    .backend
                    .as_interpreter_mut()
                    .expect("Plan A: interpreter backend")
                    .flush_trace_to_host();
                return XdnaEmuExecStatus {
                    result: XdnaEmuResult::ExecutionError,
                    cycles_executed: cycles,
                    halted: false,
                    halt_reason: XdnaEmuHaltReason::Error,
                };
            }
            npu_progressed = matches!(result, xdna_emu_core::npu::AdvanceResult::Progressed);
        }

        // Invariant from try_advance(): BlockedOnPoll => try_advance returned
        // Blocked, so npu_progressed must be false. Preserved from pre-refactor.
        debug_assert!(
            !(handle.npu_executor.is_blocked_on_poll() && npu_progressed),
            "executor reported BlockedOnPoll yet npu_progressed=true"
        );

        // When the NPU executor progressed (executed an instruction that may
        // configure DMA or write START_QUEUE), flush in-flight control packet
        // data through the stream switch.  On real hardware the stream switch
        // latency is invisible to firmware; in the emulator, routing is batched
        // per cycle so control packet register writes can lag behind.
        if npu_progressed {
            handle
                .backend
                .as_interpreter_mut()
                .expect("Plan A: interpreter backend")
                .flush_ctrl_packets();
        }

        // DMA-only path: when no cores are enabled the engine halts
        // immediately, but the NPU executor may still be configuring and
        // triggering DMA. force_running() lets step() keep advancing DMA
        // and stream switches. No-op when the engine is already Running.
        if !handle.npu_executor.is_done() {
            handle
                .backend
                .as_interpreter_mut()
                .expect("Plan A: interpreter backend")
                .force_running();
        }

        handle.backend.as_interpreter_mut().expect("Plan A: interpreter backend").step();
        // Tier B: drain newly-recorded async errors and fire the registered
        // callback (if any). Mirrors the flush_trace_to_host pattern -- FFI
        // layer observes between engine steps.
        crate::async_errors::fire_async_callbacks_for(handle);
        cycles += 1;

        // Build per-cycle snapshots and classify.
        let interp = handle.backend.as_interpreter().expect("Plan A: interpreter backend");
        let engine_signals = build_engine_signals(interp);
        let executor_signals = build_executor_signals(&mut handle.npu_executor, interp);

        let device = handle
            .backend
            .as_interpreter_mut()
            .expect("Plan A: interpreter backend")
            .device_mut();
        let detector = &mut device.tdr_detectors[DEFAULT_CONTEXT.0 as usize];
        let verdict = detector.classify(&engine_signals, Some(&executor_signals));

        match verdict {
            TdrVerdict::Progressing => continue,
            TdrVerdict::NaturalCompletion => {
                log::info!("Natural completion after {} cycles", cycles);
                natural_halt = true;
                break 'run;
            }
            TdrVerdict::MaskPollUnsatisfied => {
                log::info!("MASKPOLL unsatisfiable after {} cycles", cycles);
                maskpoll_unsatisfied = true;
                natural_halt = true;
                break 'run;
            }
            TdrVerdict::Wedged { reason, diagnosis } => {
                log::warn!("Tier C wedge after {} cycles: {:?} -- {}", cycles, reason, diagnosis);
                wedged = Some((reason, diagnosis));
                break 'run;
            }
        }
    }

    // Flush any pending trace packets through the stream switch to host DDR.
    // Trace units may have partial packets buffered; flush() pads and emits
    // them, then step_data_movement() routes them through the network.
    handle
        .backend
        .as_interpreter_mut()
        .expect("Plan A: interpreter backend")
        .flush_trace_to_host();

    // On wedge, transition context state.
    if let Some((reason, diagnosis)) = wedged {
        let device = handle
            .backend
            .as_interpreter_mut()
            .expect("Plan A: interpreter backend")
            .device_mut();
        device.contexts[DEFAULT_CONTEXT.0 as usize].mark_failed(reason, diagnosis);
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::Success,
            cycles_executed: cycles,
            // halted=true: the run terminated (not still running) -- the
            // plugin's last_run_complete_ flag should fire so run.wait() exits
            // its wait loop. The distinction between "completed cleanly" and
            // "aborted via TDR" is carried by halt_reason, not halted.
            halted: true,
            halt_reason: XdnaEmuHaltReason::WedgeRecovered,
        };
    }

    // On natural completion, advance the context's completed_counter.
    if natural_halt && !maskpoll_unsatisfied {
        let device = handle
            .backend
            .as_interpreter_mut()
            .expect("Plan A: interpreter backend")
            .device_mut();
        device.contexts[DEFAULT_CONTEXT.0 as usize].note_submission_complete();
    }

    // `halted` is the run-lifecycle / quiescence indicator -- "the run is
    // done" (cores halted OR all DMA syncs satisfied), which drives the C++
    // plugin's `last_run_complete_` and thus run.wait() returning
    // ERT_CMD_STATE_COMPLETED. It is NOT the semantic core-debug-halt: that
    // is Core_Status[16] (DEBUG_HALT), which is never touched/faked here (on
    // EMU it stays unset -- that is the whole point of MaskPollUnsatisfied).
    //
    // For the MaskPollUnsatisfied path, maskpoll_unsatisfied terminates the
    // run while the engine may still be non-Halted. Both MaskPollUnsatisfied
    // termination paths (fast-path via classifier and budget-burn) agree on
    // halted=true since natural_halt is set in both cases.
    let halted = maskpoll_unsatisfied || natural_halt;
    let halt_reason = if maskpoll_unsatisfied {
        XdnaEmuHaltReason::MaskPollUnsatisfied
    } else if natural_halt {
        XdnaEmuHaltReason::Completed
    } else {
        XdnaEmuHaltReason::Budget
    };

    XdnaEmuExecStatus { result: XdnaEmuResult::Success, cycles_executed: cycles, halted, halt_reason }
}

fn build_engine_signals(
    engine: &xdna_emu_core::interpreter::engine::InterpreterEngine,
) -> xdna_emu_core::device::tdr::EngineSignals {
    use xdna_emu_core::device::tdr::{EngineSignals, EngineStatusSnapshot};
    use xdna_emu_core::interpreter::engine::EngineStatus;

    let status = match engine.status() {
        EngineStatus::Running => EngineStatusSnapshot::Running,
        EngineStatus::Halted => EngineStatusSnapshot::Halted,
        EngineStatus::Stalled => EngineStatusSnapshot::Stalled,
        _ => EngineStatusSnapshot::Other,
    };
    let device = engine.device();

    // Core statuses: compute tiles only (rows 2+). Rows 0 (shim) and 1
    // (memtile) have no compute cores.
    let mut core_statuses = Vec::new();
    for col in 0..device.cols() {
        for row in 2..device.rows() {
            if engine.is_core_enabled(col, row) {
                if let Some(s) = engine.core_status(col, row) {
                    core_statuses.push((col as u8, row as u8, s));
                }
            }
        }
    }

    // DMA: present on all tile types (shim, memtile, compute).
    let mut dma_states = Vec::new();
    use xdna_emu_core::device::dma::ChannelState;
    for col in 0..device.cols() {
        for row in 0..device.rows() {
            if let Some(dma) = device.array.dma_engine(col as u8, row as u8) {
                for ch in 0..dma.num_channels() {
                    let state = dma.channel_state(ch as u8);
                    if !matches!(state, ChannelState::Idle) {
                        let desc = dma.channel_fsm_description(ch as u8);
                        dma_states.push((col as u8, row as u8, ch as u8, desc));
                    }
                }
            }
        }
    }

    EngineSignals {
        engine_status: status,
        any_dma_active: device.array.any_dma_active(),
        any_data_in_flight: device.array.any_data_in_flight(),
        total_dma_bytes_transferred: device.array.total_dma_bytes_transferred(),
        total_lock_releases: device.array.total_lock_releases(),
        total_instructions: engine.total_instructions(),
        core_statuses,
        dma_states,
    }
}

fn build_executor_signals(
    executor: &mut xdna_emu_core::npu::NpuExecutor,
    engine: &xdna_emu_core::interpreter::engine::InterpreterEngine,
) -> xdna_emu_core::device::tdr::ExecutorSignals {
    use xdna_emu_core::device::tdr::ExecutorSignals;
    ExecutorSignals {
        is_done: executor.is_done(),
        syncs_satisfied: executor.syncs_satisfied(engine.device()),
        is_blocked_on_poll: executor.is_blocked_on_poll(),
        pending_syncs: executor
            .pending_syncs()
            .iter()
            .map(|s| (s.column, s.row, s.channel, s.direction))
            .collect(),
    }
}
