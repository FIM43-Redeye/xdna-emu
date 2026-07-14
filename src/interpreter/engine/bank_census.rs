//! Per-cycle recorder for memory-bank arbitration outcomes.
//!
//! Validation instrument, not part of the execution model: when enabled, the
//! coordinator's arbitration pass (`arbitrate_memory_banks`) pushes one record
//! per compute tile per cycle in which *anything* happened at the bank arbiter
//! -- a contended bank, a core that lost, or a DMA channel that lost. Cycles
//! with no contention are not recorded.
//!
//! It exists so the model can be compared against the Phoenix HW capture at
//! `build/experiments/memory-stall-bankcap/` on the same axis the hardware
//! trace measures: CYCLES in which an event was asserted (interval area), plus
//! the run-length / gap SHAPE of the stall. Decoded trace *record* counts are
//! an encoding artifact and must never be compared (see the finding doc).
//!
//! Off by default, and scoped to the THREAD that enabled it. There is no handle
//! to hang it on -- the in-process xclbin runner owns its engine -- so the
//! recorder is ambient; but ambient plus process-global would mean a consumer
//! asserting over records produced by whatever other engines happened to be
//! stepping elsewhere in the process. `cargo test --lib` runs its tests in
//! parallel threads in one process and dozens of them build an
//! `InterpreterEngine` and step it, so that is not hypothetical -- it held only
//! because no concurrent test happened to produce a cycle that recorded. Thread
//! scope makes the single-consumer contract structural: a consumer sees exactly
//! the records of the engine it stepped, because it stepped it itself.
//!
//! (An engine stepped on a thread that did not `enable()` records nothing. That
//! is the correct answer for every consumer we have -- each drives its own
//! engine inline -- and a loud one if it ever stops being: the records come back
//! empty rather than mixed with a stranger's.)

use std::cell::{Cell, RefCell};

/// One compute tile's bank-arbitration outcome for one cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankCensusRecord {
    /// Simulation cycle.
    pub cycle: u64,
    pub col: u8,
    pub row: u8,
    /// The core lost a bank it needed this cycle (-> MEMORY_STALL).
    pub core_lost: bool,
    /// Banks with more than one requester this cycle (-> CONFLICT_DM_BANK_n).
    pub contended_banks: u16,
    /// S2MM channels that lost a bank arbitration this cycle. NOT an event:
    /// hardware proved a denied DMA channel raises nothing (it absorbs the loss
    /// in its staging FIFO). Recorded because the denial rate is what the
    /// bank-width finding measured the DMA against.
    pub denied_s2mm: u8,
    /// MM2S channels that lost a bank arbitration this cycle. See `denied_s2mm`.
    pub denied_mm2s: u8,
    /// S2MM channels asserting DMA_S2MM_n_MEMORY_BACKPRESSURE this cycle
    /// (ingress FIFO full with a beat on offer).
    pub s2mm_backpressure: u8,
    /// MM2S channels asserting DMA_MM2S_n_MEMORY_STARVATION this cycle (egress
    /// staging FIFO empty with the stream port ready).
    pub mm2s_starvation: u8,
    /// S2MM channels stalled on a lock acquire this cycle. Not an event this
    /// module emits -- recorded so the SHAPE of the backpressure windows can be
    /// checked against silicon's, where backpressure begins exactly 15 cycles
    /// into a lock stall and ends 1 cycle after it, with zero variance
    /// (2026-07-14-dma-memory-pressure-event-semantics.md). A matching total
    /// with the wrong shape is a failure.
    pub s2mm_lock_stalled: u8,
    /// Banks any DMA channel DEMANDED this cycle (won or lost).
    ///
    /// The loss fields above only name losers, which cannot distinguish a core
    /// that lost a bank to the DMA from a core that lost a bank to ITS OWN other
    /// memory port (two of LoadA/LoadB/Store landing in one physical bank). The
    /// two have completely different causes and completely different fixes, so
    /// the census records the DMA's demand, not just its denials.
    pub dma_demand: u16,
}

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static RECORDS: RefCell<Vec<BankCensusRecord>> = const { RefCell::new(Vec::new()) };
}

/// Start recording on THIS thread (clears any previous run's records).
pub fn enable() {
    RECORDS.with_borrow_mut(|r| r.clear());
    ENABLED.set(true);
}

/// Is recording on for this thread? The coordinator checks this once per cycle.
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.get()
}

/// Push one record (no-op unless this thread enabled recording).
pub fn record(rec: BankCensusRecord) {
    RECORDS.with_borrow_mut(|r| r.push(rec));
}

/// Stop recording and drain everything this thread collected.
pub fn take() -> Vec<BankCensusRecord> {
    ENABLED.set(false);
    RECORDS.with_borrow_mut(std::mem::take)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(cycle: u64) -> BankCensusRecord {
        BankCensusRecord {
            cycle,
            col: 0,
            row: 2,
            core_lost: true,
            contended_banks: 1,
            denied_s2mm: 0,
            denied_mm2s: 0,
            s2mm_backpressure: 0,
            mm2s_starvation: 0,
            s2mm_lock_stalled: 0,
            dma_demand: 0,
        }
    }

    #[test]
    fn a_consumer_only_ever_sees_its_own_engines_records() {
        // The census is ambient, so nothing stops another engine stepping
        // concurrently -- and `cargo test --lib` runs dozens of them in parallel
        // threads in this very process. A consumer must never be handed a
        // stranger's cycles: it asserts things like "backpressure == 0" over
        // them. The other thread here does exactly what the coordinator's Phase E
        // does (gate on `is_enabled`, then `record`). Process-global state
        // returned [1, 999, 1000] here.
        enable();
        record(rec(1));

        std::thread::spawn(|| {
            if is_enabled() {
                record(rec(999)); // a foreign engine's cycle
            }
            // ...and even if it recorded unconditionally, it must not reach us.
            record(rec(1000));
        })
        .join()
        .unwrap();

        let mine = take();
        assert_eq!(
            mine.iter().map(|r| r.cycle).collect::<Vec<_>>(),
            vec![1],
            "take() must return only the records of the engine this consumer stepped"
        );
    }
}
