//! NPU execution functions for the FFI interface.
//!
//! Instruction loading, cycle limits, and the main run loop.

use std::slice;

use crate::npu::NpuInstructionStream;

use super::{XdnaEmuHandle, XdnaEmuResult, XdnaEmuExecStatus};

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
/// Returns execution status including whether the cores halted.
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_run(handle: *mut XdnaEmuHandle) -> XdnaEmuExecStatus {
    if handle.is_null() {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::InvalidHandle,
            cycles_executed: 0,
            halted: false,
        };
    }

    let handle = &mut *handle;

    use crate::interpreter::engine::EngineStatus;

    let mut cycles = 0u64;
    let max = handle.max_cycles;

    log::info!("Running emulator (max {} cycles)", max);

    // Warm-up: let cores run to their first blocking point before
    // processing NPU instructions.  On real hardware, the core has been
    // running for thousands of cycles (CDO enables the core well before
    // NPU instructions arrive through firmware + NoC).  Without this,
    // maskwrite/blockwrite instructions modify tile memory before the
    // core's init loop has written its initial values.
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

    while cycles < max {
        // Advance NPU instruction execution (interleaved with engine step)
        let npu_progressed;
        {
            let (device, host_mem) = handle.engine.device_and_host_memory();
            let result = handle.npu_executor.try_advance(device, host_mem);
            if let crate::npu::AdvanceResult::Error(msg) = result {
                log::error!("NPU executor fatal: {}", msg);
                break;
            }
            npu_progressed = matches!(result, crate::npu::AdvanceResult::Progressed);
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
            // Keep running while any of: executor pending, DMA active,
            // or sync conditions unsatisfied.
            let executor_pending = !handle.npu_executor.is_done()
                || !handle.npu_executor.syncs_satisfied(handle.engine.device());
            let dma_active = handle.engine.device().array.any_dma_active();
            if executor_pending || dma_active {
                handle.engine.force_running();
            } else {
                log::info!("Cores halted after {} cycles", cycles);
                break;
            }
        }

        // Check if DMA syncs are satisfied (execution complete).
        // Only check after all NPU instructions have been processed
        // and no DMA channels are still running.
        if handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device())
            && !handle.engine.device().array.any_dma_active()
        {
            log::info!("All DMA syncs satisfied after {} cycles", cycles);
            break;
        }
    }

    // Flush any pending trace packets through the stream switch to host DDR.
    // Trace units may have partial packets buffered; flush() pads and emits
    // them, then step_data_movement() routes them through the network.
    handle.engine.flush_trace_to_host();

    let halted = handle.engine.status() == EngineStatus::Halted
        || (handle.npu_executor.is_done()
            && handle.npu_executor.syncs_satisfied(handle.engine.device()));

    XdnaEmuExecStatus {
        result: XdnaEmuResult::Success,
        cycles_executed: cycles,
        halted,
    }
}
