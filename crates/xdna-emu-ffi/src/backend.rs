//! Backend abstraction behind `XdnaEmuHandle`.
//!
//! The FFI dispatches over `dyn NpuBackend` so a second backend (aiesim,
//! Plan B) can replace the hand-rolled interpreter without touching call
//! sites. The trait is intentionally narrow: it carries the cross-backend
//! operations the FFI needs, while deep interpreter-only introspection
//! routes through the `as_interpreter()` downcast hatch.
//!
//! `run()` and `execute_npu_instructions()` are first-class trait methods: the
//! interpreter's runtime-sequence executor (`npu_executor`), TDR detection, and
//! per-cycle async-callback firing all live inside `InterpreterBackend`, so
//! `xdna_emu_run` is a thin dispatcher with no downcast in the run path. The
//! trait is the complete execution contract.

use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;
use xdna_emu_core::device::context::ContextId;
use xdna_emu_core::device::host_memory::HostMemory;
use xdna_emu_core::interpreter::engine::InterpreterEngine;
use xdna_emu_core::npu::NpuExecutor;
use xdna_emu_core::parser::Cdo; // same path config.rs imports (re-exported at parser root)

/// How a `run()` ended. Mirrors `XdnaEmuHaltReason`; `execution.rs::map_halt`
/// translates it to the FFI exec-status triple. Kept FFI-struct-free so
/// backend.rs has no dependency on the C ABI types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaltKind {
    /// Kernel ran to natural completion (cores halted, syncs satisfied).
    Completed,
    /// Cycle budget reached before natural completion.
    Budget,
    /// A MaskPoll could not be satisfied (engine quiescent, condition unmet).
    MaskPollUnsatisfied,
    /// The in-flight submission wedged; the context was marked Failed.
    WedgeRecovered,
    /// Error during execution (FFI fault, executor error, bad precondition).
    Error,
}

/// Result of a `run()`.
#[derive(Debug, Clone, Copy)]
pub struct RunOutcome {
    pub cycles: u64,
    pub halt: HaltKind,
}

/// Observer the FFI passes into `run()`. The backend reports newly-recorded
/// async errors at whatever granularity it supports (interpreter: every cycle;
/// aiesim: per run / at sync points). This replaces `fire_async_callbacks_for`,
/// which took the whole handle -- the borrow tangle a prior phase dodged by
/// deferring `run()`. The FFI's `CallbackObserver` (async_errors.rs) fires the
/// registered C callback for each record.
pub trait RunObserver {
    fn on_async_errors(&mut self, records: &[AmdxdnaAsyncError]);
}

/// The operations the FFI performs that are common to every backend.
pub trait NpuBackend {
    // --- configuration ---
    // Cdo is lifetime-generic (`Cdo<'a>`); the elided `<'_>` keeps the trait
    // object-safe and lets callers pass any borrowed CDO.
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String>;
    fn set_start_col(&mut self, start_col: u8);
    fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String>;

    // --- host memory (tier-2) ---
    fn host_memory_mut(&mut self) -> &mut HostMemory;

    // --- lifecycle ---
    fn sync_cores_from_device(&mut self);
    fn reset_for_new_context(&mut self);
    fn reset_context(&mut self, cid: ContextId) -> Result<(), ()>;

    // --- execution (the unified seam) ---
    /// Load the runtime-sequence (NPU instruction) stream for this submission.
    /// Interpreter: feed its executor. aiesim (later): encode + buffer for
    /// register-write replay.
    fn execute_npu_instructions(
        &mut self,
        stream: &xdna_emu_core::npu::NpuInstructionStream,
    ) -> Result<(), String>;
    /// Run the configured submission to quiescence or `max_cycles` (0 =
    /// unbounded). Reports async errors via `observer`.
    fn run(&mut self, max_cycles: u64, observer: &mut dyn RunObserver) -> RunOutcome;

    // --- host-buffer registration (address patching for NPU instructions) ---
    /// Register a host buffer for runtime-sequence address patching.
    fn add_host_buffer(&mut self, address: u64, size: usize);
    /// Clear the registered host-buffer list (before a new submission).
    fn clear_host_buffers(&mut self);

    // --- topology / identity (tier-2, cross-backend) ---
    fn cols(&self) -> usize;
    fn rows(&self) -> usize;
    fn arch_name(&self) -> String;

    // --- downcast hatch (tier-3: interpreter-only introspection) ---
    fn as_interpreter(&self) -> Option<&InterpreterEngine> {
        None
    }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> {
        None
    }
}

/// The interpreter backend: the pure-ISA `InterpreterEngine` plus (a later task)
/// its runtime-sequence executor. The engine itself stays a clean core type; this
/// FFI-side wrapper is where host/firmware-level driving lives.
pub(crate) struct InterpreterBackend {
    pub(crate) engine: InterpreterEngine,
    pub(crate) npu_executor: NpuExecutor,
}

impl InterpreterBackend {
    pub(crate) fn new(engine: InterpreterEngine) -> Self {
        Self { engine, npu_executor: NpuExecutor::new() }
    }
}

impl NpuBackend for InterpreterBackend {
    fn apply_cdo(&mut self, cdo: &Cdo<'_>) -> Result<(), String> {
        self.engine.device_mut().apply_cdo(cdo).map_err(|e| e.to_string())
    }
    fn set_start_col(&mut self, start_col: u8) {
        self.engine.device_mut().set_start_col(start_col);
    }
    fn load_elf_bytes(&mut self, col: usize, row: usize, data: &[u8]) -> Result<u32, String> {
        self.engine.load_elf_bytes(col, row, data)
    }
    fn host_memory_mut(&mut self) -> &mut HostMemory {
        self.engine.host_memory_mut()
    }
    fn sync_cores_from_device(&mut self) {
        self.engine.sync_cores_from_device();
    }
    fn reset_for_new_context(&mut self) {
        self.engine.reset_for_new_context();
    }
    fn reset_context(&mut self, cid: ContextId) -> Result<(), ()> {
        self.engine.device_mut().reset_context(cid).map_err(|_| ())
    }
    fn execute_npu_instructions(
        &mut self,
        stream: &xdna_emu_core::npu::NpuInstructionStream,
    ) -> Result<(), String> {
        self.npu_executor.load(stream);
        Ok(())
    }
    fn run(&mut self, max_cycles: u64, observer: &mut dyn RunObserver) -> RunOutcome {
        run_interpreter(&mut self.engine, &mut self.npu_executor, max_cycles, observer)
    }
    fn add_host_buffer(&mut self, address: u64, size: usize) {
        self.npu_executor.add_host_buffer(address, size);
    }
    fn clear_host_buffers(&mut self) {
        self.npu_executor.set_host_buffers(Vec::new());
    }
    fn cols(&self) -> usize {
        self.engine.device().cols()
    }
    fn rows(&self) -> usize {
        self.engine.device().rows()
    }
    fn arch_name(&self) -> String {
        self.engine.device().arch_name().to_string()
    }
    fn as_interpreter(&self) -> Option<&InterpreterEngine> {
        Some(&self.engine)
    }
    fn as_interpreter_mut(&mut self) -> Option<&mut InterpreterEngine> {
        Some(&mut self.engine)
    }
}

/// Run the interpreter engine to completion or budget. Lifted verbatim from
/// `xdna_emu_run` -- the loop logic is unchanged; only the handle-coupled
/// borrows are now explicit parameters (`engine`, `executor`, `observer`).
pub(crate) fn run_interpreter(
    engine: &mut InterpreterEngine,
    executor: &mut NpuExecutor,
    max_cycles: u64,
    observer: &mut dyn RunObserver,
) -> RunOutcome {
    let mut cycles = 0u64;
    // max_cycles == 0 means "unbounded": run until a natural exit point. But a
    // genuinely unbounded loop is dangerous -- a DMA/scheduler deadlock (e.g. a
    // buggy lock-release change) never reaches NaturalCompletion and the TDR
    // wedge detector, which keys off core progress, classifies a pure-DMA
    // lock-deadlock as "Progressing" forever. The loop then spins without ever
    // returning, hanging the caller (bridge-trace-runner) and the box until a
    // hard reboot. 2026-06-13: a symmetric memtile release-deferral did exactly
    // this and forced a REISUB. So even in unbounded mode we enforce a hard
    // RUNAWAY_CEILING -- far above any real NPU1 program (v2_core ~124k cyc; the
    // config default budget is 10M) but finite, so a deadlock terminates the run
    // (loud error + Budget halt) instead of wedging the host. Real long runs that
    // legitimately need more should pass an explicit max_cycles.
    const RUNAWAY_CEILING: u64 = 50_000_000;
    let unbounded = max_cycles == 0;
    let effective_max = if unbounded { RUNAWAY_CEILING } else { max_cycles };

    log::info!(
        "Running emulator (max {})",
        if unbounded {
            "unbounded".to_string()
        } else {
            format!("{} cycles", max_cycles)
        }
    );

    // Entry guard: context(0) must be Connected. Otherwise the caller
    // forgot to call xdna_emu_reset_context after a prior wedge.
    {
        use xdna_emu_core::device::context::DEFAULT_CONTEXT;
        let device = engine.device();
        let ctx = &device.contexts[DEFAULT_CONTEXT.0 as usize];
        if !ctx.is_connected() {
            log::error!(
                "xdna_emu_run: context {:?} not Connected; \
                 caller must reset_context before re-submitting",
                DEFAULT_CONTEXT
            );
            return RunOutcome { cycles: 0, halt: HaltKind::Error };
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
    if engine.enabled_cores() > 0 && !executor.is_done() {
        const MAX_WARMUP: u64 = 100_000;
        while cycles < MAX_WARMUP {
            engine.step();
            cycles += 1;
            if engine.all_cores_blocked() {
                break;
            }
        }
        log::info!("Core warm-up: {} cycles", cycles);
    }

    // Per-cycle: build snapshots, classify, dispatch on verdict.
    use xdna_emu_core::device::tdr::{TdrVerdict, WedgeReason, TdrDiagnosis};
    use xdna_emu_core::device::context::DEFAULT_CONTEXT;

    let mut natural_halt = false;
    let mut maskpoll_unsatisfied = false;
    let mut wedged: Option<(WedgeReason, TdrDiagnosis)> = None;

    'run: while cycles < effective_max {
        // Publish the current simulation cycle to the tile array before the
        // NPU executor runs. Register-write side effects (trace unit
        // start/stop, broadcast propagation) driven by NPU instructions
        // read array.current_cycle to time-stamp events.
        engine.device_mut().array.set_dma_cycle(cycles);

        // Advance NPU instruction execution (interleaved with engine step).
        let npu_progressed;
        {
            let (device, host_mem) = engine.device_and_host_memory();
            let result = executor.try_advance(device, host_mem);
            if let xdna_emu_core::npu::AdvanceResult::Error(msg) = result {
                log::error!("NPU executor fatal: {}", msg);
                // Flush trace before returning so partial data reaches DDR.
                engine.flush_trace_to_host();
                return RunOutcome { cycles, halt: HaltKind::Error };
            }
            npu_progressed = matches!(result, xdna_emu_core::npu::AdvanceResult::Progressed);
        }

        // Invariant from try_advance(): BlockedOnPoll => try_advance returned
        // Blocked, so npu_progressed must be false. Preserved from pre-refactor.
        debug_assert!(
            !(executor.is_blocked_on_poll() && npu_progressed),
            "executor reported BlockedOnPoll yet npu_progressed=true"
        );

        // When the NPU executor progressed, the former flush_ctrl_packets used
        // to fast-forward in-flight control packets here. That flush was
        // vestigial under the active firmware-latency model (executor config
        // writes are immediate; normal step() delivers control packets every
        // cycle) and its only side effect was the tenant-4 tail-collapse. It is
        // now a detector: it records a hazard if a packet-switched control
        // packet is still in flight at this instruction boundary (the condition
        // the flush masked). See InterpreterEngine::note_ctrl_packet_ordering_hazard.
        if npu_progressed {
            engine.note_ctrl_packet_ordering_hazard();
        }

        // DMA-only path: when no cores are enabled the engine halts
        // immediately, but the NPU executor may still be configuring and
        // triggering DMA. force_running() lets step() keep advancing DMA
        // and stream switches. No-op when the engine is already Running.
        if !executor.is_done() {
            engine.force_running();
        }

        engine.step();
        // Tier B: drain newly-recorded async errors and fire the registered
        // callback (if any). Mirrors the flush_trace_to_host pattern -- FFI
        // layer observes between engine steps.
        let recs = engine.device_mut().async_errors.drain_newly_recorded();
        observer.on_async_errors(&recs);
        cycles += 1;

        // Build per-cycle snapshots and classify.
        let engine_signals = build_engine_signals(engine);
        let executor_signals = build_executor_signals(executor, engine);

        let device = engine.device_mut();
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

    // Runaway guard: an unbounded run that hit the hard ceiling without a
    // natural halt or a wedge verdict is almost certainly a DMA/scheduler
    // deadlock. Surface it loudly -- it halts as Budget below, but this is NOT a
    // normal cycle-budget exhaustion; it means the kernel never reached
    // quiescence. (Prevents the silent infinite spin that forced a REISUB.)
    if unbounded && !natural_halt && wedged.is_none() && cycles >= effective_max {
        log::error!(
            "xdna_emu_run: RUNAWAY CEILING hit at {} cycles with no natural halt -- \
             likely a DMA/scheduler deadlock (no core/DMA forward progress). Aborting \
             to avoid hanging the host. Investigate the lock/BD-chain state.",
            cycles
        );
    }

    // Flush any pending trace packets through the stream switch to host DDR.
    // Trace units may have partial packets buffered; flush() pads and emits
    // them, then step_data_movement() routes them through the network.
    engine.flush_trace_to_host();

    // On wedge, transition context state.
    if let Some((reason, diagnosis)) = wedged {
        let device = engine.device_mut();
        device.contexts[DEFAULT_CONTEXT.0 as usize].mark_failed(reason, diagnosis);
        return RunOutcome { cycles, halt: HaltKind::WedgeRecovered };
    }

    // On natural completion, advance the context's completed_counter.
    if natural_halt && !maskpoll_unsatisfied {
        let device = engine.device_mut();
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
    let halt = if maskpoll_unsatisfied {
        HaltKind::MaskPollUnsatisfied
    } else if natural_halt {
        HaltKind::Completed
    } else {
        HaltKind::Budget
    };

    RunOutcome { cycles, halt }
}

fn build_engine_signals(engine: &InterpreterEngine) -> xdna_emu_core::device::tdr::EngineSignals {
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
    executor: &mut NpuExecutor,
    engine: &InterpreterEngine,
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

#[cfg(test)]
pub(crate) mod mock {
    use super::*;

    /// A do-nothing backend used to prove FFI dispatch and the graceful
    /// "not an interpreter" path without constructing a real engine.
    #[derive(Default)]
    pub(crate) struct MockBackend {
        pub apply_cdo_calls: u32,
        pub start_col: u8,
        pub reset_for_new_context_calls: u32,
    }

    impl NpuBackend for MockBackend {
        fn apply_cdo(&mut self, _cdo: &Cdo<'_>) -> Result<(), String> {
            self.apply_cdo_calls += 1;
            Ok(())
        }
        fn set_start_col(&mut self, start_col: u8) {
            self.start_col = start_col;
        }
        fn load_elf_bytes(&mut self, _c: usize, _r: usize, _d: &[u8]) -> Result<u32, String> {
            Ok(0)
        }
        fn host_memory_mut(&mut self) -> &mut HostMemory {
            unimplemented!("MockBackend has no host memory")
        }
        fn sync_cores_from_device(&mut self) {}
        fn reset_for_new_context(&mut self) {
            self.reset_for_new_context_calls += 1;
        }
        fn reset_context(&mut self, _cid: ContextId) -> Result<(), ()> {
            Ok(())
        }
        fn execute_npu_instructions(
            &mut self,
            _stream: &xdna_emu_core::npu::NpuInstructionStream,
        ) -> Result<(), String> {
            Ok(())
        }
        fn run(&mut self, _max_cycles: u64, _observer: &mut dyn RunObserver) -> RunOutcome {
            RunOutcome { cycles: 0, halt: HaltKind::Completed }
        }
        fn add_host_buffer(&mut self, _address: u64, _size: usize) {}
        fn clear_host_buffers(&mut self) {}
        fn cols(&self) -> usize {
            5
        }
        fn rows(&self) -> usize {
            6
        }
        fn arch_name(&self) -> String {
            "mock".to_string()
        }
        // as_interpreter / as_interpreter_mut use the default None impls.
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockBackend;
    use super::NpuBackend;

    #[test]
    fn run_outcome_and_observer_compose() {
        use super::{HaltKind, RunObserver, RunOutcome};
        use xdna_emu_core::device::async_errors::AmdxdnaAsyncError;

        struct Counting(u32);
        impl RunObserver for Counting {
            fn on_async_errors(&mut self, records: &[AmdxdnaAsyncError]) {
                self.0 += records.len() as u32;
            }
        }
        let mut o = Counting(0);
        o.on_async_errors(&[]);
        assert_eq!(o.0, 0);
        let out = RunOutcome { cycles: 7, halt: HaltKind::Completed };
        assert_eq!(out.cycles, 7);
    }

    #[test]
    fn mock_backend_is_not_an_interpreter() {
        let m = MockBackend::default();
        assert!(m.as_interpreter().is_none());
    }

    #[test]
    fn mock_backend_dispatches_trait_methods() {
        let mut m: Box<dyn NpuBackend> = Box::new(MockBackend::default());
        m.set_start_col(3);
        m.reset_for_new_context();
        assert_eq!(m.cols(), 5);
        assert_eq!(m.rows(), 6);
        assert_eq!(m.arch_name(), "mock");
    }

    #[test]
    fn dyn_dispatch_hits_the_concrete_backend() {
        use super::mock::MockBackend;

        // Own the concrete mock; dispatch through a `&mut dyn` (vtable) reference.
        let mut mock = MockBackend::default();
        {
            let backend: &mut dyn NpuBackend = &mut mock;
            backend.set_start_col(2);
            backend.reset_for_new_context();
            backend.reset_for_new_context();

            // The downcast hatch correctly reports "not an interpreter".
            assert!(backend.as_interpreter().is_none());
            assert!(backend.as_interpreter_mut().is_none());
        } // dyn borrow ends here

        // The dynamic calls landed on the concrete MockBackend (proves dispatch
        // reached it, not a default).
        assert_eq!(mock.start_col, 2);
        assert_eq!(mock.reset_for_new_context_calls, 2);
    }

    #[test]
    fn interpreter_implements_backend_and_downcasts() {
        use xdna_emu_core::interpreter::engine::InterpreterEngine;
        let mut be = super::InterpreterBackend::new(InterpreterEngine::new_npu1());

        // Trait methods reflect the real device.
        let b: &mut dyn NpuBackend = &mut be;
        assert_eq!(b.cols(), 5);
        assert_eq!(b.rows(), 6);
        assert!(b.arch_name().to_lowercase().contains("npu") || b.arch_name().to_lowercase().contains("aie"));

        // The downcast hatch returns the engine.
        assert!(b.as_interpreter().is_some());
        assert!(b.as_interpreter_mut().is_some());
    }
}
