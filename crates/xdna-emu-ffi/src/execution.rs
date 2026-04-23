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

    use xdna_emu_core::interpreter::engine::EngineStatus;

    let mut cycles = 0u64;
    let max = handle.max_cycles;
    // max == 0 means unbounded: the loop runs until a natural exit point.
    let unbounded = max == 0;

    log::info!(
        "Running emulator (max {})",
        if unbounded { "unbounded".to_string() } else { format!("{} cycles", max) }
    );

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
    if handle.engine.enabled_cores() > 0 && !handle.npu_executor.is_done() {
        const MAX_WARMUP: u64 = 100_000;
        while cycles < MAX_WARMUP {
            handle.engine.step();
            cycles += 1;
            if handle.engine.all_cores_blocked() {
                break;
            }
        }
        log::info!("Core warm-up: {} cycles (all cores at first blocking point)", cycles);
    }

    // Track whether we exited via a natural halt (Completed) or fell through
    // to budget exhaustion.
    let mut natural_halt = false;

    'run: while unbounded || cycles < max {
        // Publish the current simulation cycle to the tile array before the
        // NPU executor runs. Register-write side effects (trace unit
        // start/stop, broadcast propagation) driven by NPU instructions
        // read array.current_cycle to time-stamp events.
        handle.engine.device_mut().array.set_dma_cycle(cycles);

        // Advance NPU instruction execution (interleaved with engine step).
        let npu_progressed;
        {
            let (device, host_mem) = handle.engine.device_and_host_memory();
            let result = handle.npu_executor.try_advance(device, host_mem);
            if let xdna_emu_core::npu::AdvanceResult::Error(msg) = result {
                log::error!("NPU executor fatal: {}", msg);
                // Flush trace before returning so partial data reaches DDR.
                handle.engine.flush_trace_to_host();
                return XdnaEmuExecStatus {
                    result: XdnaEmuResult::ExecutionError,
                    cycles_executed: cycles,
                    halted: false,
                    halt_reason: XdnaEmuHaltReason::Error,
                };
            }
            npu_progressed = matches!(result, xdna_emu_core::npu::AdvanceResult::Progressed);
        }

        // When the NPU executor progressed (executed an instruction that may
        // configure DMA or write START_QUEUE), flush in-flight control packet
        // data through the stream switch.  On real hardware the stream switch
        // latency is invisible to firmware; in the emulator, routing is batched
        // per cycle so control packet register writes can lag behind.
        if npu_progressed {
            handle.engine.flush_ctrl_packets();
        }

        handle.engine.step();
        cycles += 1;

        if handle.engine.status() == EngineStatus::Halted {
            // For DMA-only tests (no cores loaded), the engine halts
            // immediately because no cores are enabled.  But the NPU
            // executor may still be issuing instructions that configure
            // and trigger DMA, or DMA channels may already be running.
            // Keep running while executor has pending work or syncs are
            // unsatisfied.
            let executor_pending = !handle.npu_executor.is_done()
                || !handle.npu_executor.syncs_satisfied(handle.engine.device());
            if executor_pending || handle.engine.device().array.any_dma_active() {
                handle.engine.force_running();
            } else {
                log::info!("Cores halted after {} cycles", cycles);
                natural_halt = true;
                break 'run;
            }
        }

        if handle.engine.status() == EngineStatus::Stalled {
            log::warn!("Stall detected after {} cycles: no monotonic progress", cycles);
            // A stall is treated as a natural (if unhappy) halt — the run
            // completed as far as the emulator can tell, not a budget cut.
            natural_halt = true;
            break 'run;
        }

        // Check if all NPU sync conditions are satisfied.  On real
        // hardware, firmware considers execution complete when the sync
        // fires -- DMA channels may still be running (trace channels,
        // BD chains with use_next_bd, etc.) but the host transfer is
        // done.  Do NOT require all DMA to be idle here.
        if handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device())
        {
            log::info!("All DMA syncs satisfied after {} cycles", cycles);
            natural_halt = true;
            break 'run;
        }
    }

    // Flush any pending trace packets through the stream switch to host DDR.
    // Trace units may have partial packets buffered; flush() pads and emits
    // them, then step_data_movement() routes them through the network.
    handle.engine.flush_trace_to_host();

    let halted = handle.engine.status() == EngineStatus::Halted
        || (handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device()));

    // If we didn't exit naturally (halted/stalled/syncs), the while-loop
    // condition `cycles >= max` ended the run — budget was exhausted.
    let halt_reason = if natural_halt || halted {
        XdnaEmuHaltReason::Completed
    } else {
        XdnaEmuHaltReason::Budget
    };

    XdnaEmuExecStatus {
        result: XdnaEmuResult::Success,
        cycles_executed: cycles,
        halted,
        halt_reason,
    }
}
