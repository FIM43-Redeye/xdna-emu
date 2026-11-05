//! NPU execution functions for the FFI interface.
//!
//! Instruction loading, cycle limits, and the main run loop.

use std::slice;

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

    // Load instructions for execution in xdna_emu_run().
    match handle.backend.load_instruction_bytes(instr_slice) {
        Ok(count) => {
            log::info!("Executing {} NPU instructions", count);
            XdnaEmuResult::Success
        }
        Err(crate::backend::ConfigError::Parse(e)) => {
            log::error!("Failed to parse NPU instructions: {}", e);
            XdnaEmuResult::ParseError
        }
        Err(crate::backend::ConfigError::Apply(e)) => {
            log::error!("execute_npu_instructions: {}", e);
            crate::set_last_error(e);
            XdnaEmuResult::ExecutionError
        }
    }
}

fn config_result(error: crate::backend::ConfigError, what: &str) -> XdnaEmuResult {
    match error {
        crate::backend::ConfigError::Parse(e) => {
            log::error!("{what}: {e}");
            crate::set_last_error(e);
            XdnaEmuResult::ParseError
        }
        crate::backend::ConfigError::Apply(e) => {
            log::error!("{what}: {e}");
            crate::set_last_error(e);
            XdnaEmuResult::ExecutionError
        }
    }
}

/// Hand the backend the compute units of host hardware context `ctx` before
/// a submission on it: `count` PDIs (`pdis[i]`, `pdi_sizes[i]` bytes) with
/// their CU functions. `*executes_commands` reports whether the backend runs
/// that context's commands itself; if so the caller queues each command with
/// `xdna_emu_queue_command` and runs them with one `xdna_emu_run`, otherwise
/// it loads each command's PDI and instruction bytes as before.
///
/// # Safety
/// - `handle` must be valid
/// - `pdis`, `pdi_sizes` and `functions` must each point to `count` entries,
///   and each `pdis[i]` to `pdi_sizes[i]` bytes
/// - `executes_commands` must be writable
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_configure_cus(
    handle: *mut XdnaEmuHandle,
    ctx: u32,
    count: u32,
    pdis: *const *const u8,
    pdi_sizes: *const u64,
    functions: *const u32,
    executes_commands: *mut u8,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if executes_commands.is_null()
        || (count > 0 && (pdis.is_null() || pdi_sizes.is_null() || functions.is_null()))
    {
        return XdnaEmuResult::NullPointer;
    }
    let count = count as usize;
    let mut cus = Vec::with_capacity(count);
    for index in 0..count {
        let (pdi, size) = (*pdis.add(index), *pdi_sizes.add(index));
        if pdi.is_null() && size > 0 {
            return XdnaEmuResult::NullPointer;
        }
        let pdi = if size == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(pdi, size as usize)
        };
        cus.push(crate::backend::CuConfig { pdi, function: *functions.add(index) });
    }
    match (*handle).backend.configure_cus(ctx, &cus) {
        Ok(owned) => {
            *executes_commands = u8::from(owned);
            XdnaEmuResult::Success
        }
        Err(e) => config_result(e, "configure_cus"),
    }
}

/// Queue one ERT command of the next submission on `ctx` (backends that
/// reported `executes_commands`): the packet opcode, the CU index its CU
/// mask selects, and the `count` payload words after the CU masks.
///
/// # Safety
/// - `handle` must be valid
/// - `words` must point to at least `count` u32 values
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_queue_command(
    handle: *mut XdnaEmuHandle,
    ctx: u32,
    opcode: u32,
    cu_index: u32,
    words: *const u32,
    count: u32,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    if words.is_null() && count > 0 {
        return XdnaEmuResult::NullPointer;
    }
    let payload = if count == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(words, count as usize)
    };
    let command = crate::backend::ErtCommand { opcode, cu_index, payload };
    match (*handle).backend.queue_command(ctx, &command) {
        Ok(()) => XdnaEmuResult::Success,
        Err(e) => config_result(e, "queue_command"),
    }
}

/// The host destroyed hardware context `ctx`.
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_destroy_hw_context(handle: *mut XdnaEmuHandle, ctx: u32) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }
    (*handle).backend.destroy_hw_context(ctx);
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
    let outcome = handle.backend.run(max, &mut observer);
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
