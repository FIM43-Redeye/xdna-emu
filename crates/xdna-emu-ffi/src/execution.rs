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
        if unbounded {
            "unbounded".to_string()
        } else {
            format!("{} cycles", max)
        }
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
    // Set when the run terminated because an injected MASKPOLL can never be
    // satisfied (the engine went quiescent with the poll still blocked). This
    // is the expected EMU outcome for the debug_halt_probe's halt-sync
    // MASKPOLL: Core_Status[16] (DEBUG_HALT) never sets on the emulator, so
    // the poll is unsatisfiable by design. We end deterministically here
    // rather than spinning until the stall detector trips ~100k cycles later.
    let mut maskpoll_unsatisfied = false;
    // Bounded poll-stall counter: consecutive cycles the executor has been
    // parked in BlockedOnPoll with no executor progress. A *satisfiable*
    // poll resolves within a few cycles of whatever writes the polled
    // register running; the debug_halt_probe MASKPOLL (Core_Status[16] /
    // DEBUG_HALT) is unsatisfiable on EMU because the breakpoint-arming
    // control-packet writes are dropped, so nothing the run loop steps can
    // ever change the polled value. This counter is the deterministic,
    // DMA-churn-independent "no monotonic progress" signal the spec §4.2
    // contract requires -- it does NOT depend on any_dma_active() (a
    // starved ctrl-out DMA re-routing the same packet is not progress) or
    // the ~100k generic stall threshold (too slow for the bridge
    // wall-clock). The window is generous enough that any legitimately
    // satisfiable poll resolves first, but far below the bridge timeout.
    //
    // Known edge case: this bound is safe for firmware-issued MASKPOLLs,
    // which resolve within a few cycles of the writing event (DMA-completion
    // status, a control-packet register write, etc.). A hypothetical future
    // MASKPOLL polling a register written only by a long (>20k-cycle)
    // compute-core computation would be terminated prematurely. No such user
    // exists today; revisit the bound if one is added.
    let mut poll_stall_cycles: u64 = 0;
    const POLL_STALL_LIMIT: u64 = 20_000;

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

        // Bounded poll-stall detection (spec §4.2 graceful poll-termination).
        // The counter advances purely because the executor is BlockedOnPoll
        // each cycle: while blocked on a poll, try_advance() returns Blocked
        // (never Progressed), so npu_progressed is necessarily false in this
        // branch -- there is no separate "no progress" condition to test
        // (asserted below). It is reset the moment the executor is no longer
        // blocked on a poll (a poll that does eventually satisfy never trips
        // this). Crossing the bound means the poll is unsatisfiable: end
        // deterministically with the distinct terminal reason. This is the
        // DMA-churn-independent signal -- any_dma_active() was rejected
        // because a starved ctrl-out DMA re-routing the same packet keeps it
        // true forever, and the ~100k generic stall threshold is too slow
        // for the bridge wall-clock. We do NOT fake the polled register,
        // pretend the core debug-halted, or skip the OP_READ -- the executor
        // stays BlockedOnPoll, no further instruction is issued, and the
        // run.wait() path returns cleanly.
        if handle.npu_executor.is_blocked_on_poll() {
            // Invariant: BlockedOnPoll => try_advance() returned Blocked, so
            // the executor cannot have progressed this cycle.
            debug_assert!(!npu_progressed, "executor reported BlockedOnPoll yet npu_progressed=true");
            poll_stall_cycles += 1;
            if poll_stall_cycles >= POLL_STALL_LIMIT {
                log::info!(
                    "MASKPOLL unsatisfiable: executor BlockedOnPoll with no progress \
                     for {} cycles (total {} cycles) -- terminating deterministically \
                     (MaskPollUnsatisfied)",
                    poll_stall_cycles,
                    cycles
                );
                maskpoll_unsatisfied = true;
                natural_halt = true;
                break 'run;
            }
        } else {
            poll_stall_cycles = 0;
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
        // Tier B: drain newly-recorded async errors and fire the registered
        // callback (if any). Mirrors the flush_trace_to_host pattern -- FFI
        // layer observes between engine steps.
        crate::async_errors::fire_async_callbacks_for(handle);
        cycles += 1;

        if handle.engine.status() == EngineStatus::Halted {
            // Graceful poll-termination contract (debug_halt_probe MASKPOLL,
            // spec §4.2). If the executor is parked in BlockedOnPoll and the
            // engine is otherwise quiescent (cores halted, no DMA in flight),
            // the poll can never become satisfied -- nothing left running can
            // change the polled register. End deterministically with a
            // distinct terminal reason instead of force-running until the
            // stall detector trips ~100k cycles later. We do NOT fake the
            // polled register, pretend the core halted, or skip to the next
            // instruction: the executor stays BlockedOnPoll and no further
            // instruction (the OP_READ push) is issued.
            if handle.npu_executor.is_blocked_on_poll() && !handle.engine.device().array.any_dma_active() {
                log::info!(
                    "MASKPOLL unsatisfiable: engine quiescent (cores halted, no DMA) \
                     with poll still blocked after {} cycles -- terminating deterministically",
                    cycles
                );
                maskpoll_unsatisfied = true;
                natural_halt = true;
                break 'run;
            }

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
            // Defensive: if the stall is because the executor is parked in an
            // unsatisfiable BlockedOnPoll (e.g. a path where the core never
            // reaches the Halted status), still surface the distinct
            // MASKPOLL-unsatisfied terminal reason rather than a generic
            // Completed -- the cause is the same unsatisfiable poll.
            if handle.npu_executor.is_blocked_on_poll() {
                log::info!("Stall is an unsatisfiable MASKPOLL -- terminating as MaskPollUnsatisfied");
                maskpoll_unsatisfied = true;
            }
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
        if handle.npu_executor.is_done() && handle.npu_executor.syncs_satisfied(handle.engine.device()) {
            log::info!("All DMA syncs satisfied after {} cycles", cycles);
            natural_halt = true;
            break 'run;
        }
    }

    // Flush any pending trace packets through the stream switch to host DDR.
    // Trace units may have partial packets buffered; flush() pads and emits
    // them, then step_data_movement() routes them through the network.
    handle.engine.flush_trace_to_host();

    // `halted` is the run-lifecycle / quiescence indicator -- "the run is
    // done" (cores halted OR all DMA syncs satisfied), which drives the C++
    // plugin's `last_run_complete_` and thus run.wait() returning
    // ERT_CMD_STATE_COMPLETED. It is NOT the semantic core-debug-halt: that
    // is Core_Status[16] (DEBUG_HALT), which is never touched/faked here (on
    // EMU it stays unset -- that is the whole point of MaskPollUnsatisfied).
    //
    // For the MaskPollUnsatisfied path, the poll-stall counter terminates
    // the run while engine.status() may still be non-Halted (the core is
    // gated on the objectfifo, never debug-halted). Without this the
    // poll-stall exit would report halted=false while the logically
    // identical Halted+!any_dma_active fast path reports halted=true -- the
    // two termination paths must agree. So set halted=true explicitly for
    // this case: it means "run terminated deterministically", not
    // "core debug-halted".
    let halted = maskpoll_unsatisfied
        || handle.engine.status() == EngineStatus::Halted
        || (handle.npu_executor.is_done() && handle.npu_executor.syncs_satisfied(handle.engine.device()));

    // Terminal-reason priority:
    //  1. MaskPollUnsatisfied: an injected MASKPOLL could never be satisfied
    //     (debug_halt_probe halt-sync on EMU). Distinct from a generic
    //     completion so the host harness can treat it as the expected EMU
    //     baseline rather than misreading absent OP_READ responses.
    //  2. Completed: natural halt / stalled / syncs satisfied.
    //  3. Budget: the while-loop condition `cycles >= max` ended the run.
    let halt_reason = if maskpoll_unsatisfied {
        XdnaEmuHaltReason::MaskPollUnsatisfied
    } else if natural_halt || halted {
        XdnaEmuHaltReason::Completed
    } else {
        XdnaEmuHaltReason::Budget
    };

    XdnaEmuExecStatus { result: XdnaEmuResult::Success, cycles_executed: cycles, halted, halt_reason }
}
