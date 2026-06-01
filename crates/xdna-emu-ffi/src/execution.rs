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
    let max = handle.max_cycles;
    let mut observer = crate::async_errors::CallbackObserver { cb: handle.async_callback };

    // Disjoint field borrows: backend (engine) and npu_executor.
    let Some(engine) = handle.backend.as_interpreter_mut() else {
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::ExecutionError,
            cycles_executed: 0,
            halted: false,
            halt_reason: XdnaEmuHaltReason::Error,
        };
    };
    let outcome = crate::backend::run_interpreter(engine, &mut handle.npu_executor, max, &mut observer);
    let (result, halted, halt_reason) = map_halt(outcome.halt);
    XdnaEmuExecStatus { result, cycles_executed: outcome.cycles, halted, halt_reason }
}

fn map_halt(halt: crate::backend::HaltKind) -> (XdnaEmuResult, bool, XdnaEmuHaltReason) {
    use crate::backend::HaltKind;
    match halt {
        HaltKind::Completed => (XdnaEmuResult::Success, true, XdnaEmuHaltReason::Completed),
        // Budget = cycle cap hit before quiescence: the run is NOT done, so
        // halted=false (matches the pre-unification `maskpoll_unsatisfied ||
        // natural_halt` truth table -- both are false on budget exhaustion).
        HaltKind::Budget => (XdnaEmuResult::Success, false, XdnaEmuHaltReason::Budget),
        HaltKind::MaskPollUnsatisfied => {
            (XdnaEmuResult::Success, true, XdnaEmuHaltReason::MaskPollUnsatisfied)
        }
        HaltKind::WedgeRecovered => (XdnaEmuResult::Success, true, XdnaEmuHaltReason::WedgeRecovered),
        HaltKind::Error => (XdnaEmuResult::ExecutionError, false, XdnaEmuHaltReason::Error),
    }
}

#[cfg(test)]
mod tests {
    use super::map_halt;
    use crate::backend::HaltKind;
    use crate::{XdnaEmuHaltReason, XdnaEmuResult};

    /// Pin the full `HaltKind` -> (result, halted, reason) truth table. This
    /// mirrors the pre-unification `xdna_emu_run` final-return logic exactly;
    /// `halted` is the run-done / `run.wait()` signal, NOT a semantic core halt.
    /// In particular Budget (cycle cap hit) is `halted=false` -- the run did not
    /// finish, it was capped -- matching the old `maskpoll || natural_halt`.
    #[test]
    fn map_halt_truth_table_matches_pre_unification() {
        assert_eq!(
            map_halt(HaltKind::Completed),
            (XdnaEmuResult::Success, true, XdnaEmuHaltReason::Completed)
        );
        assert_eq!(
            map_halt(HaltKind::Budget),
            (XdnaEmuResult::Success, false, XdnaEmuHaltReason::Budget),
            "budget exhaustion is not a completed run: halted must be false"
        );
        assert_eq!(
            map_halt(HaltKind::MaskPollUnsatisfied),
            (XdnaEmuResult::Success, true, XdnaEmuHaltReason::MaskPollUnsatisfied)
        );
        assert_eq!(
            map_halt(HaltKind::WedgeRecovered),
            (XdnaEmuResult::Success, true, XdnaEmuHaltReason::WedgeRecovered)
        );
        assert_eq!(
            map_halt(HaltKind::Error),
            (XdnaEmuResult::ExecutionError, false, XdnaEmuHaltReason::Error)
        );
    }
}
