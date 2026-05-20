//! Device-state classifiers used by Tier C TDR.
//!
//! `QuiescenceDetector` catches deadlocks (every subsystem terminal,
//! no possible forward progress). `StallDetector` catches livelocks
//! (cores running but no DMA-bytes/lock-release progress with pending
//! syncs). Composed by [`super::TdrDetector`] which classifies engine
//! run state into a single per-cycle verdict.
//!
//! Lifted from `src/testing/quiescence.rs` (where they were misfiled
//! as test infrastructure) on 2026-05-19 as part of Tier C.

use std::fmt;

use crate::interpreter::core::CoreStatus;
use crate::interpreter::engine::{InterpreterEngine, EngineStatus};
use crate::npu::NpuExecutor;

/// Result of a quiescence check on a single cycle.
pub enum QuiescenceStatus {
    /// System is making progress or not yet fully configured.
    Running,
    /// System is quiescent (deadlocked). Contains diagnostic snapshot.
    Quiescent(TdrDiagnosis),
}

/// Diagnostic snapshot of system state at the moment deadlock is declared.
#[derive(Debug)]
pub struct TdrDiagnosis {
    /// Per-core status: (col, row, description).
    pub core_states: Vec<(u8, u8, String)>,
    /// Per-channel DMA status: (col, row, channel, state description).
    pub dma_states: Vec<(u8, u8, u8, String)>,
    /// Whether any data was in flight at the moment of diagnosis.
    pub data_in_flight: bool,
    /// Unsatisfied pending sync descriptions.
    pub pending_syncs: Vec<String>,
}

impl fmt::Display for TdrDiagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Core states (only non-halted or interesting ones)
        let interesting_cores: Vec<_> =
            self.core_states.iter().filter(|(_, _, desc)| desc != "Halted").collect();

        if interesting_cores.is_empty() {
            write!(f, "all cores halted")?;
        } else {
            for (i, (col, row, desc)) in interesting_cores.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "core({},{}) {}", col, row, desc)?;
            }
        }

        // DMA states (only non-idle ones)
        let active_dma: Vec<_> = self.dma_states.iter().filter(|(_, _, _, desc)| desc != "Idle").collect();

        if !active_dma.is_empty() {
            write!(f, "; DMA: ")?;
            for (i, (col, row, ch, desc)) in active_dma.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "({},{})ch{} {}", col, row, ch, desc)?;
            }
        }

        // Pending syncs
        if !self.pending_syncs.is_empty() {
            write!(f, "; pending syncs: ")?;
            for (i, sync) in self.pending_syncs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", sync)?;
            }
        }

        Ok(())
    }
}

/// Detects system quiescence (deadlock) by monitoring all subsystems.
pub struct QuiescenceDetector {
    /// Consecutive cycles where all quiescence conditions hold.
    quiescent_cycles: u64,
    /// Threshold before declaring deadlock.
    threshold: u64,
}

impl QuiescenceDetector {
    /// Create a new detector with the given cycle threshold.
    pub fn new(threshold: u64) -> Self {
        Self { quiescent_cycles: 0, threshold }
    }

    pub(crate) fn bump_quiescent_cycle(&mut self) {
        self.quiescent_cycles += 1;
    }

    pub(crate) fn reset_quiescent_cycles(&mut self) {
        self.quiescent_cycles = 0;
    }

    pub(crate) fn threshold_met(&self) -> bool {
        self.quiescent_cycles >= self.threshold
    }

    /// Check whether the system is quiescent this cycle.
    ///
    /// Must be called once per cycle in the run loop. Returns:
    /// - `Running` if the system is still making progress or waiting
    /// - `Quiescent(diagnosis)` if deadlock is confirmed (sustained for threshold cycles)
    pub fn check(
        &mut self,
        engine: &InterpreterEngine,
        npu_executor: Option<&NpuExecutor>,
    ) -> QuiescenceStatus {
        // Condition 1: NPU executor must be done.
        if let Some(executor) = npu_executor {
            if !executor.is_done() {
                self.quiescent_cycles = 0;
                return QuiescenceStatus::Running;
            }
        }

        // If the coordinator already declared Halted, let the normal completion
        // path in run_engine() handle it -- no need for quiescence detection.
        if matches!(engine.status(), EngineStatus::Halted) {
            self.quiescent_cycles = 0;
            return QuiescenceStatus::Running;
        }

        // Condition 2: No core is in a runnable state (Running or Ready).
        let device = engine.device();
        let cols = device.cols();
        let rows = device.rows();

        for col in 0..cols {
            for row in 2..rows {
                if let Some(status) = engine.core_status(col, row) {
                    if matches!(status, CoreStatus::Running | CoreStatus::Ready) {
                        self.quiescent_cycles = 0;
                        return QuiescenceStatus::Running;
                    }
                }
            }
        }

        // Condition 3: No DMA is actively transferring.
        if device.array.any_dma_transferring() {
            self.quiescent_cycles = 0;
            return QuiescenceStatus::Running;
        }

        // Condition 4: No data in flight (stream switches, cascade FIFOs).
        if device.array.any_data_in_flight() {
            self.quiescent_cycles = 0;
            return QuiescenceStatus::Running;
        }

        // All conditions met -- system is quiescent this cycle.
        self.quiescent_cycles += 1;

        if self.quiescent_cycles >= self.threshold {
            QuiescenceStatus::Quiescent(Self::diagnose(engine, npu_executor))
        } else {
            QuiescenceStatus::Running
        }
    }

    /// Build a diagnostic snapshot of the current system state.
    pub(crate) fn diagnose(engine: &InterpreterEngine, npu_executor: Option<&NpuExecutor>) -> TdrDiagnosis {
        let device = engine.device();
        let cols = device.cols();
        let rows = device.rows();

        // Collect core states for enabled compute tiles.
        let mut core_states = Vec::new();
        for col in 0..cols {
            for row in 2..rows {
                if engine.is_core_enabled(col, row) {
                    let desc = match engine.core_status(col, row) {
                        Some(CoreStatus::Ready) => "Ready".to_string(),
                        Some(CoreStatus::Running) => "Running".to_string(),
                        Some(CoreStatus::WaitingLock { raw_lock_id }) => {
                            format!("WaitingLock({})", raw_lock_id)
                        }
                        Some(CoreStatus::WaitingDma { channel }) => {
                            format!("WaitingDma({})", channel)
                        }
                        Some(CoreStatus::WaitingStream { port }) => {
                            format!("WaitingStream({})", port)
                        }
                        Some(CoreStatus::Halted) => "Halted".to_string(),
                        Some(CoreStatus::Error) => "Error".to_string(),
                        None => "Unknown".to_string(),
                    };
                    core_states.push((col as u8, row as u8, desc));
                }
            }
        }

        // Collect DMA channel states for tiles with non-idle channels.
        use crate::device::dma::engine::ChannelState;
        let mut dma_states = Vec::new();
        for col in 0..cols {
            for row in 0..rows {
                if let Some(dma) = device.array.dma_engine(col as u8, row as u8) {
                    for ch in 0..dma.num_channels() {
                        let state = dma.channel_state(ch as u8);
                        if !matches!(state, ChannelState::Idle) {
                            // Use detailed FSM description instead of the
                            // coarse ChannelState to distinguish Transferring
                            // from ReleasingLock, BdChaining, etc.
                            let desc = dma.channel_fsm_description(ch as u8);
                            dma_states.push((col as u8, row as u8, ch as u8, desc));
                        }
                    }
                }
            }
        }

        // Pending syncs from the NPU executor.
        let pending_syncs = if let Some(executor) = npu_executor {
            executor
                .pending_syncs()
                .iter()
                .map(|s| {
                    let dir = if s.direction == 0 { "S2MM" } else { "MM2S" };
                    format!("col={} row={} ch={} {}", s.column, s.row, s.channel, dir)
                })
                .collect()
        } else {
            Vec::new()
        };

        TdrDiagnosis {
            core_states,
            dma_states,
            data_in_flight: device.array.any_data_in_flight(),
            pending_syncs,
        }
    }
}

// ---------------------------------------------------------------------------
// DMA Stall Detection
// ---------------------------------------------------------------------------

/// Result of a stall check on a single cycle.
pub enum StallStatus {
    /// DMA is making byte-level progress (or no syncs pending yet).
    Progressing,
    /// DMA has not transferred any new bytes for `threshold` consecutive cycles
    /// while syncs remain unsatisfied. Contains a diagnostic snapshot.
    Stalled(TdrDiagnosis),
}

/// Detects DMA stalls by monitoring lock-release progress.
///
/// Complements [`QuiescenceDetector`] which catches true deadlocks (all
/// subsystems terminal). This detector catches **livelocks** where cores
/// are still running but neither DMA bytes are moving nor any lock is being
/// released, so the workload can never complete.
///
/// The stall counter resets whenever either DMA bytes or lock release count
/// changes. Lock releases are a robust forward-progress signal: any time a
/// core releases a lock it is making productive progress toward completion,
/// even if DMA bytes are not moving in that cycle.
///
/// The counter only fires when there are unsatisfied pending syncs --
/// otherwise the test may legitimately be doing pure computation.
pub struct StallDetector {
    /// Last observed total DMA bytes transferred.
    last_dma_bytes: u64,
    /// Last observed total lock releases across all tiles.
    last_lock_releases: u64,
    /// Consecutive cycles with no progress (DMA or lock releases).
    cycles_since_progress: u64,
    /// Threshold before declaring a stall.
    threshold: u64,
}

impl StallDetector {
    /// Create a new stall detector with the given cycle threshold.
    pub fn new(threshold: u64) -> Self {
        Self { last_dma_bytes: 0, last_lock_releases: 0, cycles_since_progress: 0, threshold }
    }

    pub(crate) fn reset(&mut self) {
        self.cycles_since_progress = 0;
    }

    /// Returns true when threshold met (stalled).
    pub(crate) fn note_progress(&mut self, dma_bytes: u64, lock_releases: u64) -> bool {
        if dma_bytes != self.last_dma_bytes || lock_releases != self.last_lock_releases {
            self.last_dma_bytes = dma_bytes;
            self.last_lock_releases = lock_releases;
            self.cycles_since_progress = 0;
            false
        } else {
            self.cycles_since_progress += 1;
            self.cycles_since_progress >= self.threshold
        }
    }

    /// Check whether the system has stalled this cycle.
    ///
    /// Must be called once per cycle. Returns:
    /// - `Progressing` if DMA bytes or lock releases are advancing
    /// - `Stalled(diagnosis)` if no progress for `threshold` cycles
    ///   with unsatisfied syncs
    pub fn check(&mut self, engine: &InterpreterEngine, npu_executor: Option<&NpuExecutor>) -> StallStatus {
        // Only check if there are pending syncs to satisfy. Without syncs,
        // there is no DMA completion target and the test may be purely
        // compute-based (or hasn't started DMA yet).
        let has_pending_syncs = if let Some(executor) = npu_executor {
            !executor.pending_syncs().is_empty()
        } else {
            false
        };

        if !has_pending_syncs {
            self.cycles_since_progress = 0;
            return StallStatus::Progressing;
        }

        let current_bytes = engine.device().array.total_dma_bytes_transferred();
        let current_lock_releases = engine.device().array.total_lock_releases();

        if current_bytes != self.last_dma_bytes || current_lock_releases != self.last_lock_releases {
            self.last_dma_bytes = current_bytes;
            self.last_lock_releases = current_lock_releases;
            self.cycles_since_progress = 0;
            StallStatus::Progressing
        } else {
            self.cycles_since_progress += 1;
            if self.cycles_since_progress >= self.threshold {
                StallStatus::Stalled(QuiescenceDetector::diagnose(engine, npu_executor))
            } else {
                StallStatus::Progressing
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiescence_threshold_counting() {
        // Verify the counter logic in isolation: once we start calling
        // check() and all conditions are met, it should take exactly
        // `threshold` cycles to declare quiescence.
        let mut detector = QuiescenceDetector::new(5);

        // Simulate 4 cycles of quiescence -- not enough yet.
        for _ in 0..4 {
            detector.quiescent_cycles += 1;
        }
        assert_eq!(detector.quiescent_cycles, 4);
        assert!(detector.quiescent_cycles < detector.threshold);

        // One more should cross the threshold.
        detector.quiescent_cycles += 1;
        assert!(detector.quiescent_cycles >= detector.threshold);
    }

    #[test]
    fn test_quiescence_reset_on_progress() {
        // Verify that making progress resets the counter.
        let mut detector = QuiescenceDetector::new(10);

        // Accumulate some quiescent cycles.
        detector.quiescent_cycles = 8;

        // Simulate progress detection (what check() does internally).
        detector.quiescent_cycles = 0;
        assert_eq!(detector.quiescent_cycles, 0);

        // Verify we need the full threshold again.
        for _ in 0..9 {
            detector.quiescent_cycles += 1;
        }
        assert!(detector.quiescent_cycles < detector.threshold);
    }

    #[test]
    fn test_diagnosis_display_all_halted() {
        let diag = TdrDiagnosis {
            core_states: vec![(0, 2, "Halted".to_string()), (0, 3, "Halted".to_string())],
            dma_states: vec![],
            data_in_flight: false,
            pending_syncs: vec!["col=0 row=0 ch=0 S2MM".to_string()],
        };
        let s = diag.to_string();
        assert!(s.contains("all cores halted"), "got: {}", s);
        assert!(s.contains("pending syncs"), "got: {}", s);
        assert!(s.contains("col=0 row=0 ch=0 S2MM"), "got: {}", s);
    }

    #[test]
    fn test_diagnosis_display_waiting_core() {
        let diag = TdrDiagnosis {
            core_states: vec![(0, 2, "WaitingLock(5)".to_string()), (0, 3, "Halted".to_string())],
            dma_states: vec![(0, 2, 0, "WaitingForLock(5)".to_string())],
            data_in_flight: false,
            pending_syncs: vec![],
        };
        let s = diag.to_string();
        assert!(s.contains("core(0,2) WaitingLock(5)"), "got: {}", s);
        // Halted core should NOT appear since we filter those.
        assert!(!s.contains("core(0,3)"), "got: {}", s);
        assert!(s.contains("DMA:"), "got: {}", s);
        assert!(s.contains("(0,2)ch0 WaitingForLock(5)"), "got: {}", s);
    }
}
