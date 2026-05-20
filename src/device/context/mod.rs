//! Per-context state model for Tier C.
//!
//! Driver cross-reference: this mirrors `amdxdna_ctx`'s observable state
//! (state enum + completion counter) at the level the emulator needs to
//! expose to a future driver consumer doing TDR. The driver's AIE2 vocabulary
//! (`CTX_STATE_*` in `aie2_pci.h`) maps as follows:
//!
//! | Emulator state | Driver state | Notes |
//! |---|---|---|
//! | `Connected` | `CTX_STATE_CONNECTED` (0x2) | Ready to accept submissions |
//! | `Stopped` | `CTX_STATE_DISCONNECTED` (0x0) | Reserved -- not entered by Tier C |
//! | `Failed { .. }` | `CTX_STATE_DEAD` (0xFF) | `errno` set on driver side; we carry full diagnosis |
//!
//! `CTX_STATE_DISPATCHED` and `CTX_STATE_DISCONNECTING` are transient scheduler
//! states with no emulator equivalent. `CTX_STATE_DEBUG` is a special
//! debug-mode state outside the TDR scope. Implementation is emulator-original;
//! behavior is constrained by the spec (see
//! `docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md`).

use crate::device::tdr::{TdrDiagnosis, WedgeReason};

/// Identifies a context. `Vec<Context>` is indexed by this value; today
/// there is always exactly one (`DEFAULT_CONTEXT`).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ContextId(pub u32);

pub const DEFAULT_CONTEXT: ContextId = ContextId(0);

/// Per-context state. Driver's hwctx vocabulary subset.
#[derive(Clone, Debug)]
pub enum ContextState {
    /// Ready to accept submissions.
    Connected,
    /// Idle but re-Connectable without firmware reload. Reserved for the
    /// multi-context spec; not entered by any Tier C path.
    #[allow(dead_code)]
    Stopped,
    /// Submission wedged; carries reason + diagnostic snapshot.
    Failed {
        reason: WedgeReason,
        diagnosis: TdrDiagnosis,
    },
    // Disconnected (firmware-reload required) reserved for multi-context.
}

pub struct Context {
    pub id: ContextId,
    pub state: ContextState,
    pub completed_counter: u64,
    pub pending_cmd_count: u32,
}

impl Context {
    pub fn new(id: ContextId) -> Self {
        Self { id, state: ContextState::Connected, completed_counter: 0, pending_cmd_count: 0 }
    }

    pub fn mark_failed(&mut self, reason: WedgeReason, diagnosis: TdrDiagnosis) {
        self.state = ContextState::Failed { reason, diagnosis };
    }

    pub fn mark_connected(&mut self) {
        self.state = ContextState::Connected;
    }

    pub fn note_submission_complete(&mut self) {
        self.completed_counter = self.completed_counter.saturating_add(1);
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state, ContextState::Connected)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.state, ContextState::Failed { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_diagnosis() -> TdrDiagnosis {
        TdrDiagnosis {
            core_states: vec![],
            dma_states: vec![],
            data_in_flight: false,
            pending_syncs: vec![],
        }
    }

    #[test]
    fn new_context_starts_connected_with_zero_counter() {
        let ctx = Context::new(DEFAULT_CONTEXT);
        assert!(ctx.is_connected());
        assert!(!ctx.is_failed());
        assert_eq!(ctx.completed_counter, 0);
        assert_eq!(ctx.pending_cmd_count, 0);
        assert_eq!(ctx.id, DEFAULT_CONTEXT);
    }

    #[test]
    fn mark_failed_transitions_state_and_preserves_counter() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.completed_counter = 7;
        ctx.mark_failed(WedgeReason::Quiescent, fake_diagnosis());
        assert!(ctx.is_failed());
        assert!(!ctx.is_connected());
        assert_eq!(ctx.completed_counter, 7, "counter should not reset on failure");
    }

    #[test]
    fn mark_connected_clears_failed_state() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.mark_failed(WedgeReason::Stalled, fake_diagnosis());
        ctx.mark_connected();
        assert!(ctx.is_connected());
    }

    #[test]
    fn note_submission_complete_advances_counter() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.note_submission_complete();
        ctx.note_submission_complete();
        ctx.note_submission_complete();
        assert_eq!(ctx.completed_counter, 3);
    }

    #[test]
    fn mark_connected_is_idempotent_on_connected_context() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.mark_connected();
        ctx.mark_connected();
        assert!(ctx.is_connected());
    }

    #[test]
    fn default_context_id_is_zero() {
        assert_eq!(DEFAULT_CONTEXT, ContextId(0));
    }
}
