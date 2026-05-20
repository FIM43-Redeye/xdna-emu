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
    context_id: u32,
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
    pub fn new(context_id: u32) -> Self {
        Self {
            context_id,
            quiescence: QuiescenceDetector::new(DEFAULT_QUIESCENCE_CYCLES),
            stall: StallDetector::new(DEFAULT_STALL_CYCLES),
            poll_stall_cycles: 0,
            poll_stall_limit: DEFAULT_POLL_STALL_LIMIT,
        }
    }

    /// The context this detector classifies.
    pub fn context_id(&self) -> u32 {
        self.context_id
    }

    // classify(...) added in Tasks 4-7 (TDD per verdict variant).
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
