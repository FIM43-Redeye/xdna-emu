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
//! Off by default and global (the in-process xclbin runner owns its engine, so
//! there is no handle to hang this on). Single-consumer: `enable()` then
//! `take()` from one test at a time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

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

static ENABLED: AtomicBool = AtomicBool::new(false);
static RECORDS: Mutex<Vec<BankCensusRecord>> = Mutex::new(Vec::new());

/// Start recording (clears any previous run's records).
pub fn enable() {
    RECORDS.lock().unwrap().clear();
    ENABLED.store(true, Ordering::Relaxed);
}

/// Is recording on? The coordinator checks this once per cycle.
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Push one record (no-op unless enabled).
pub fn record(rec: BankCensusRecord) {
    RECORDS.lock().unwrap().push(rec);
}

/// Stop recording and drain everything collected.
pub fn take() -> Vec<BankCensusRecord> {
    ENABLED.store(false, Ordering::Relaxed);
    std::mem::take(&mut RECORDS.lock().unwrap())
}
