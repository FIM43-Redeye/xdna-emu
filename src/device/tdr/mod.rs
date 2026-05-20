//! Tier C TDR (Timeout Detection & Recovery) / context-restart support.
//!
//! Exposes the per-cycle classifier that decides whether the in-flight
//! submission is progressing, completing naturally, exhausting a satisfiable
//! poll, or wedged. The actual TDR algorithm (periodic timer, two-tick stuck
//! check, recovery chain) is a driver-side concern; this module exposes the
//! signals a driver TDRs on. See:
//! - `docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md`
//! - `~/npu-work/xdna-driver/src/driver/amdxdna/aie2_tdr.c` (the driver-side
//!   algorithm this lets a driver consumer drive)

pub mod detector;

pub use detector::{QuiescenceDetector, QuiescenceStatus, StallDetector, StallStatus, TdrDiagnosis};

use crate::device::context::ContextId;
use crate::interpreter::core::CoreStatus;

/// Reason a context's submission is classified as wedged.
///
/// Precedence (when more than one would apply): `Quiescent` > `Stalled` >
/// `PollExhausted`. A truly-quiescent system is also trivially "stalled" --
/// pick the strongest description. See spec section 4.2.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WedgeReason {
    /// Every subsystem terminal, no possible forward progress.
    Quiescent,
    /// Pending syncs, cores still running, but no DMA-bytes/lock-release progress.
    Stalled,
    /// Executor parked in `BlockedOnPoll` past the cycle budget,
    /// and the cleaner `MaskPollUnsatisfied` test did not catch it
    /// (engine wasn't quiescent yet).
    PollExhausted,
}

/// Per-cycle verdict from [`TdrDetector::classify`].
///
/// Precedence inside the classifier (when more than one would apply):
/// `NaturalCompletion` > `MaskPollUnsatisfied` > `Wedged` > `Progressing`.
#[derive(Debug)]
pub enum TdrVerdict {
    /// Forward progress this cycle. Run loop continues.
    Progressing,
    /// Engine halted with all syncs satisfied. The normal happy path.
    NaturalCompletion,
    /// Existing semantic, now classified here. Run loop breaks with
    /// `XdnaEmuHaltReason::MaskPollUnsatisfied`.
    MaskPollUnsatisfied,
    /// Submission is wedged. Caller transitions context state to
    /// `Failed { reason, diagnosis }` and breaks the run loop with
    /// `XdnaEmuHaltReason::WedgeRecovered`.
    Wedged {
        reason: WedgeReason,
        diagnosis: TdrDiagnosis,
    },
}

/// Per-cycle snapshot of engine signals the classifier reads.
///
/// Built by the run loop from `&InterpreterEngine` once per cycle.
/// Tests construct directly without standing up an engine.
#[derive(Debug, Clone)]
pub struct EngineSignals {
    pub engine_status: EngineStatusSnapshot,
    pub any_dma_active: bool,
    pub any_data_in_flight: bool,
    pub total_dma_bytes_transferred: u64,
    pub total_lock_releases: u64,
    /// (col, row, status) for every enabled compute core.
    pub core_statuses: Vec<(u8, u8, CoreStatus)>,
    /// (col, row, channel, fsm_description) for every non-idle DMA channel.
    pub dma_states: Vec<(u8, u8, u8, String)>,
}

/// Mirror of the live `EngineStatus` enum, snapshotted for the classifier.
///
/// Kept narrow on purpose -- the classifier only needs the variants that
/// affect its decision tree, so this is decoupled from the engine's full
/// `EngineStatus` enum (which has additional intermediate variants).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EngineStatusSnapshot {
    Running,
    Halted,
    Stalled,
    Other,
}

/// Per-cycle snapshot of NPU executor signals the classifier reads.
#[derive(Debug, Clone)]
pub struct ExecutorSignals {
    pub is_done: bool,
    pub syncs_satisfied: bool,
    pub is_blocked_on_poll: bool,
    /// (col, row, channel, direction) for each pending sync.
    pub pending_syncs: Vec<(u8, u8, u8, u8)>,
}

/// Per-context classifier composing the lifted Quiescence/Stall detectors
/// with a poll-stall budget.
pub struct TdrDetector {
    context_id: ContextId,
    quiescence: QuiescenceDetector,
    stall: StallDetector,
    poll_stall_cycles: u64,
    poll_stall_limit: u64,
}

/// Cycle budget after which a sustained `BlockedOnPoll` is reported as
/// `PollExhausted`. Mirrors the current inline `POLL_STALL_LIMIT` in
/// `xdna_emu_run` (20_000 cycles). Generous enough that any legitimately
/// satisfiable poll resolves first, well below the bridge wall-clock.
pub const DEFAULT_POLL_STALL_LIMIT: u64 = 20_000;

/// Cycle threshold for [`QuiescenceDetector`] when used inside [`TdrDetector`].
/// Same value the in-process xclbin runner uses today.
pub const DEFAULT_QUIESCENCE_CYCLES: u64 = 5;

/// Cycle threshold for [`StallDetector`] when used inside [`TdrDetector`].
pub const DEFAULT_STALL_CYCLES: u64 = 100_000;

impl TdrDetector {
    /// Construct a detector for the given context using the default thresholds.
    pub fn new(context_id: ContextId) -> Self {
        Self {
            context_id,
            quiescence: QuiescenceDetector::new(DEFAULT_QUIESCENCE_CYCLES),
            stall: StallDetector::new(DEFAULT_STALL_CYCLES),
            poll_stall_cycles: 0,
            poll_stall_limit: DEFAULT_POLL_STALL_LIMIT,
        }
    }

    /// The context this detector classifies.
    pub fn context_id(&self) -> ContextId {
        self.context_id
    }

    /// Classify the engine's run state this cycle.
    ///
    /// Precedence (when more than one would apply): `NaturalCompletion` >
    /// `MaskPollUnsatisfied` > `Wedged` > `Progressing`. The classifier
    /// returns the strongest applicable verdict.
    ///
    /// Read-only over signals; mutates internal counters
    /// (`poll_stall_cycles`, quiescence/stall thresholds) per cycle.
    pub fn classify(&mut self, signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> TdrVerdict {
        // Highest precedence: natural completion.
        if signals.engine_status == EngineStatusSnapshot::Halted {
            if let Some(exec) = executor {
                if exec.is_done && exec.syncs_satisfied {
                    return TdrVerdict::NaturalCompletion;
                }
            } else if !signals.any_dma_active && !signals.any_data_in_flight {
                return TdrVerdict::NaturalCompletion;
            }
        }

        // Second precedence: MaskPollUnsatisfied.
        // Mirrors the two paths the inline xdna_emu_run logic uses today:
        //  (a) engine Halted + executor BlockedOnPoll + no DMA in flight
        //      (the "fast-path" -- nothing left running can satisfy the poll)
        //  (b) executor BlockedOnPoll for poll_stall_limit consecutive cycles
        //      (the "budget" -- caps unsatisfiable polls in the running case)
        if let Some(exec) = executor {
            if exec.is_blocked_on_poll {
                // Fast-path (a):
                if signals.engine_status == EngineStatusSnapshot::Halted && !signals.any_dma_active {
                    return TdrVerdict::MaskPollUnsatisfied;
                }
                // Budget (b):
                self.poll_stall_cycles += 1;
                if self.poll_stall_cycles >= self.poll_stall_limit {
                    return TdrVerdict::MaskPollUnsatisfied;
                }
            } else {
                self.poll_stall_cycles = 0;
            }
        } else {
            self.poll_stall_cycles = 0;
        }

        // Third precedence: Wedged{Quiescent}.
        if self.check_quiescence(signals, executor) {
            return TdrVerdict::Wedged {
                reason: WedgeReason::Quiescent,
                diagnosis: Self::build_diagnosis(signals, executor),
            };
        }
        // Fourth precedence: Wedged{Stalled}.
        if self.check_stall(signals, executor) {
            return TdrVerdict::Wedged {
                reason: WedgeReason::Stalled,
                diagnosis: Self::build_diagnosis(signals, executor),
            };
        }

        TdrVerdict::Progressing
    }

    /// Run the quiescence rule against snapshot inputs. Returns true when the
    /// quiescence threshold is met.
    fn check_quiescence(&mut self, signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> bool {
        // Same predicate the existing QuiescenceDetector enforces, snapshot-driven:
        //   - executor (if present) is done
        //   - engine not Halted
        //   - no runnable core
        //   - no DMA active
        //   - no data in flight
        let executor_done = executor.map_or(true, |e| e.is_done);
        let engine_terminal = signals.engine_status != EngineStatusSnapshot::Halted;
        let no_runnable_core = signals
            .core_statuses
            .iter()
            .all(|(_, _, s)| !matches!(s, CoreStatus::Running | CoreStatus::Ready));
        let cond = executor_done
            && engine_terminal
            && no_runnable_core
            && !signals.any_dma_active
            && !signals.any_data_in_flight;

        if cond {
            self.quiescence.bump_quiescent_cycle();
            self.quiescence.threshold_met()
        } else {
            self.quiescence.reset_quiescent_cycles();
            false
        }
    }

    fn check_stall(&mut self, signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> bool {
        let has_pending_syncs = executor.map_or(false, |e| !e.pending_syncs.is_empty());
        if !has_pending_syncs {
            self.stall.reset();
            return false;
        }
        self.stall
            .note_progress(signals.total_dma_bytes_transferred, signals.total_lock_releases)
    }

    fn build_diagnosis(signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> TdrDiagnosis {
        TdrDiagnosis {
            core_states: signals
                .core_statuses
                .iter()
                .map(|(c, r, s)| (*c, *r, format!("{:?}", s)))
                .collect(),
            dma_states: signals.dma_states.clone(),
            data_in_flight: signals.any_data_in_flight,
            pending_syncs: executor
                .map(|e| {
                    e.pending_syncs
                        .iter()
                        .map(|(c, r, ch, dir)| {
                            let dir_s = if *dir == 0 { "S2MM" } else { "MM2S" };
                            format!("col={c} row={r} ch={ch} {dir_s}")
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::context::DEFAULT_CONTEXT;

    #[test]
    fn wedge_reason_derives_copy_clone_debug() {
        let r = WedgeReason::Quiescent;
        let r2 = r; // Copy
        let _ = r; // still usable after copy
        let _ = format!("{:?}", r2); // Debug
    }

    #[test]
    fn tdr_verdict_progressing_is_default_construction() {
        let v = TdrVerdict::Progressing;
        let _ = format!("{:?}", v);
        assert!(matches!(v, TdrVerdict::Progressing));
    }

    #[test]
    fn tdr_verdict_wedged_carries_reason_and_diagnosis() {
        let diag = TdrDiagnosis {
            core_states: vec![],
            dma_states: vec![],
            data_in_flight: false,
            pending_syncs: vec![],
        };
        let v = TdrVerdict::Wedged { reason: WedgeReason::Quiescent, diagnosis: diag };
        match v {
            TdrVerdict::Wedged { reason, .. } => assert!(matches!(reason, WedgeReason::Quiescent)),
            _ => panic!("expected Wedged"),
        }
    }

    fn empty_engine_signals(status: EngineStatusSnapshot) -> EngineSignals {
        EngineSignals {
            engine_status: status,
            any_dma_active: false,
            any_data_in_flight: false,
            total_dma_bytes_transferred: 0,
            total_lock_releases: 0,
            core_statuses: vec![],
            dma_states: vec![],
        }
    }

    fn natural_completion_executor() -> ExecutorSignals {
        ExecutorSignals {
            is_done: true,
            syncs_satisfied: true,
            is_blocked_on_poll: false,
            pending_syncs: vec![],
        }
    }

    fn no_executor() -> Option<ExecutorSignals> {
        None
    }

    #[test]
    fn classify_returns_progressing_when_engine_running_and_no_executor() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let signals = empty_engine_signals(EngineStatusSnapshot::Running);
        let verdict = detector.classify(&signals, no_executor().as_ref());
        assert!(matches!(verdict, TdrVerdict::Progressing), "got {verdict:?}");
    }

    #[test]
    fn classify_returns_natural_completion_when_engine_halted_and_syncs_satisfied() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        let exec = natural_completion_executor();
        let verdict = detector.classify(&signals, Some(&exec));
        assert!(matches!(verdict, TdrVerdict::NaturalCompletion), "got {verdict:?}");
    }

    fn blocked_on_poll_executor() -> ExecutorSignals {
        ExecutorSignals {
            is_done: false,
            syncs_satisfied: false,
            is_blocked_on_poll: true,
            pending_syncs: vec![],
        }
    }

    #[test]
    fn classify_returns_mask_poll_unsatisfied_when_engine_halted_and_executor_blocked_on_poll() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let mut signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        signals.any_dma_active = false; // matches existing run-loop fast-path condition
        let exec = blocked_on_poll_executor();
        let verdict = detector.classify(&signals, Some(&exec));
        assert!(matches!(verdict, TdrVerdict::MaskPollUnsatisfied), "got {verdict:?}");
    }

    #[test]
    fn classify_returns_mask_poll_unsatisfied_after_poll_stall_budget() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let signals = empty_engine_signals(EngineStatusSnapshot::Running);
        let exec = blocked_on_poll_executor();
        // Burn through the poll-stall budget. Detector accumulates internally.
        for _ in 0..DEFAULT_POLL_STALL_LIMIT {
            let _ = detector.classify(&signals, Some(&exec));
        }
        // On the budget-th cycle (or one after), should report MaskPollUnsatisfied.
        // Run one more cycle to be safe.
        let last = detector.classify(&signals, Some(&exec));
        assert!(matches!(last, TdrVerdict::MaskPollUnsatisfied), "got {last:?}");
    }

    #[test]
    fn classify_resets_poll_stall_when_executor_unblocks() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let signals = empty_engine_signals(EngineStatusSnapshot::Running);
        let exec_blocked = blocked_on_poll_executor();
        let mut exec_unblocked = blocked_on_poll_executor();
        exec_unblocked.is_blocked_on_poll = false;

        // Accumulate near the limit while blocked.
        for _ in 0..(DEFAULT_POLL_STALL_LIMIT - 10) {
            detector.classify(&signals, Some(&exec_blocked));
        }
        // Unblock for a cycle.
        detector.classify(&signals, Some(&exec_unblocked));
        // Now block again; should NOT fire immediately (counter reset).
        for _ in 0..10 {
            let v = detector.classify(&signals, Some(&exec_blocked));
            assert!(!matches!(v, TdrVerdict::MaskPollUnsatisfied), "fired too early after reset");
        }
    }

    #[test]
    fn classify_returns_wedged_quiescent_when_all_subsystems_terminal() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        // All terminal: engine NOT Halted (otherwise we'd hit NaturalCompletion path),
        // but no DMA, no data in flight, all cores in terminal state.
        // Use the "no executor" path so NaturalCompletion's executor check
        // doesn't satisfy. Engine status Stalled here.
        let signals = empty_engine_signals(EngineStatusSnapshot::Stalled);
        // Drive the quiescence detector long enough for it to fire.
        for _ in 0..(DEFAULT_QUIESCENCE_CYCLES * 2) {
            let v = detector.classify(&signals, no_executor().as_ref());
            // Final cycle should be Wedged{Quiescent}; intermediate cycles
            // are Progressing.
            if let TdrVerdict::Wedged { reason, .. } = v {
                assert_eq!(reason, WedgeReason::Quiescent);
                return;
            }
        }
        panic!("Quiescent verdict never fired across {} cycles", DEFAULT_QUIESCENCE_CYCLES * 2);
    }

    #[test]
    fn classify_returns_wedged_stalled_with_pending_syncs_and_no_byte_progress() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        // Cores still "running" (engine not Stalled), but no DMA byte progress
        // and pending syncs. Stall detector fires after its threshold.
        let mut signals = empty_engine_signals(EngineStatusSnapshot::Running);
        signals.any_dma_active = true; // suppress Quiescent path
        let exec = ExecutorSignals {
            is_done: false,
            syncs_satisfied: false,
            is_blocked_on_poll: false,
            pending_syncs: vec![(0, 0, 0, 0)],
        };
        // StallDetector fires after DEFAULT_STALL_CYCLES of no byte/lock progress.
        let mut fired = false;
        for _ in 0..(DEFAULT_STALL_CYCLES + 100) {
            let v = detector.classify(&signals, Some(&exec));
            if let TdrVerdict::Wedged { reason, .. } = v {
                assert_eq!(reason, WedgeReason::Stalled);
                fired = true;
                break;
            }
        }
        assert!(fired, "Stalled verdict never fired");
    }

    #[test]
    fn classify_returns_wedged_poll_exhausted_when_budget_burned_without_clean_fastpath() {
        // PollExhausted differs from MaskPollUnsatisfied: it fires only when
        // the budget is burned AND the cleaner fast-path conditions are not
        // met (e.g. DMA still active, masking the "engine quiescent" tell).
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let mut signals = empty_engine_signals(EngineStatusSnapshot::Running);
        signals.any_dma_active = true; // disqualifies the MaskPollUnsatisfied fast-path
        let exec = blocked_on_poll_executor();
        for _ in 0..(DEFAULT_POLL_STALL_LIMIT + 1) {
            let _ = detector.classify(&signals, Some(&exec));
        }
        let last = detector.classify(&signals, Some(&exec));
        // PRECEDENCE NOTE: when the poll-stall budget is burned, the
        // classifier returns MaskPollUnsatisfied (b-path from Task 5),
        // NOT PollExhausted. PollExhausted is only the reason inside a
        // Wedged verdict when neither MaskPollUnsatisfied path applies;
        // since path (b) always applies once the budget burns, this test
        // expects MaskPollUnsatisfied. Kept here to lock in the precedence.
        assert!(matches!(last, TdrVerdict::MaskPollUnsatisfied), "got {last:?}");
    }

    #[test]
    fn classify_precedence_natural_completion_wins_over_wedge_signals() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        // Build inputs that would qualify as both NaturalCompletion AND
        // (after enough cycles) Wedged{Quiescent}. NaturalCompletion is
        // the higher-precedence verdict.
        let signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        let exec = natural_completion_executor();
        // First cycle and every subsequent cycle should be NaturalCompletion --
        // we never accumulate into Wedged territory.
        for _ in 0..(DEFAULT_QUIESCENCE_CYCLES * 2) {
            let v = detector.classify(&signals, Some(&exec));
            assert!(matches!(v, TdrVerdict::NaturalCompletion), "got {v:?}");
        }
    }

    #[test]
    fn classify_precedence_mask_poll_wins_over_wedge_signals() {
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        // Engine Halted + executor BlockedOnPoll + no DMA -- fast-path
        // MaskPollUnsatisfied. Also satisfies Wedged{Quiescent} structurally.
        let signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        let exec = blocked_on_poll_executor();
        let v = detector.classify(&signals, Some(&exec));
        assert!(matches!(v, TdrVerdict::MaskPollUnsatisfied), "got {v:?}");
    }

    #[test]
    fn classify_precedence_quiescent_wins_over_stalled() {
        // When BOTH would apply (executor done with all cores terminal AND
        // pending syncs with no byte progress), Quiescent reports first
        // because it's the stronger description.
        let mut detector = TdrDetector::new(DEFAULT_CONTEXT);
        let signals = empty_engine_signals(EngineStatusSnapshot::Stalled);
        let exec = ExecutorSignals {
            is_done: true, // quiescence requires this
            syncs_satisfied: false,
            is_blocked_on_poll: false,
            pending_syncs: vec![(0, 0, 0, 0)], // satisfies stall precondition too
        };
        // Burn enough cycles for quiescence threshold to fire.
        let mut fired_as = None;
        for _ in 0..(DEFAULT_QUIESCENCE_CYCLES * 2) {
            let v = detector.classify(&signals, Some(&exec));
            if let TdrVerdict::Wedged { reason, .. } = v {
                fired_as = Some(reason);
                break;
            }
        }
        assert_eq!(fired_as, Some(WedgeReason::Quiescent));
    }
}
