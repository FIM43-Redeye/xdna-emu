//! Compute-tile MM2S egress queue: the two counted resources between a BD's bank accesses
//! and the MM2S port, both released a fixed time after a word leaves the port.
//!
//! - **Granule slots.** Every fetch (one bank access) takes a slot until its last word has left
//!   the port and `DMA_MM2S_EGRESS_RELEASE_AFTER_PORT_CYCLES` have passed; at most
//!   `DMA_MM2S_EGRESS_GRANULE_SLOTS` are taken. A partial granule takes a whole slot.
//! - **BD queue.** Every BD takes an entry from its first access until its last word has left
//!   the port, plus the same release time. The queue has three uses: (a) a BD's first access
//!   waits until at most `DMA_MM2S_BD_QUEUE_ACCESS_LIMIT` BDs, itself included, hold an entry;
//!   (b) the BD at chain position p starts sequencing no earlier than the release of the BD
//!   at p - `DMA_MM2S_BD_QUEUE_SEQUENCE_LOOKBACK`, plus
//!   `DMA_MM2S_BD_QUEUE_SEQUENCE_AFTER_RELEASE_CYCLES` (`WordClock::tick`); (c) a lock BD at
//!   p >= the sequencing look-back behind a lock-free BD is requested from the release of the
//!   BD at p - `DMA_MM2S_BD_QUEUE_REQUEST_LOOKBACK`, plus
//!   `DMA_MM2S_BD_QUEUE_REQUEST_AFTER_RELEASE_CYCLES` (`LockPort::update`).
//!
//! Silicon (NPU1, 2026-11, `tools/gen_dma_lock_probe.py` depth probes 8, 9 and 11, own
//! receiver): chains of six to eight short BDs behind 4- to 32-word heads, where the request
//! and grant law alone finishes the late BDs and grants the waiting BD up to 30 cycles early.
//! MEASURED: the FINISHED_BDs, STALLED_LOCK rises and grants these rules give, every own-receiver
//! row. INFERRED: that the two counts are what bounds them, and that the port time is the own
//! receiver's word time - 4. Chain position is the word clock's `seq` (the run's first BD is 0).
//!
//! Those rows were taken with the source switch quiet, so their port times carry the switch's
//! hold of the first word (one port cycle, `TileArray`'s `crossing_hold`): the release is 2
//! cycles after the port, and 3 after the unheld port time the rows were first fitted against.
//! The crossing-wake probes' awake-source cases (c2o, a3 / a4 / n at T 64) finish one cycle
//! earlier than the own rows, as 2 gives. Holds farther out on the route reach the port the same
//! way; none of them is modelled here.

use std::collections::VecDeque;

use xdna_archspec::aie2::timing::{
    DMA_MM2S_BD_QUEUE_ACCESS_LIMIT, DMA_MM2S_BD_QUEUE_REQUEST_AFTER_RELEASE_CYCLES,
    DMA_MM2S_BD_QUEUE_REQUEST_LOOKBACK, DMA_MM2S_BD_QUEUE_SEQUENCE_AFTER_RELEASE_CYCLES,
    DMA_MM2S_BD_QUEUE_SEQUENCE_LOOKBACK, DMA_MM2S_EGRESS_GRANULE_SLOTS,
    DMA_MM2S_EGRESS_RELEASE_AFTER_PORT_CYCLES,
};

/// A BD holding (or having held) a BD-queue entry.
#[derive(Debug, Clone, Copy)]
struct BdEntry {
    /// Its chain position in the word clock's current run; `None` once a new run started
    /// (the entry still counts, but no rule of the new run looks it up).
    seq: Option<u64>,
    /// Step from which the entry is free, once its last word has left the port.
    release: Option<u64>,
}

/// Released entries kept so that the look-backs of (b) and (c) can find them.
const HISTORY: usize = DMA_MM2S_BD_QUEUE_SEQUENCE_LOOKBACK as usize + 2;

/// The compute MM2S egress queue (module docs). Channel state beside `egress_staging`, whose
/// words close its slots and entries as they leave the port.
#[derive(Debug, Clone, Default)]
pub struct EgressQueue {
    /// Release step of every granule fetched and not yet released (`None` while words of it
    /// have not all left the port), in fetch order.
    granules: VecDeque<Option<u64>>,
    /// BDs in first-access order: the ones holding an entry, and the latest released ones.
    bds: VecDeque<BdEntry>,
}

fn held_at(release: Option<u64>, at: u64) -> bool {
    release.is_none_or(|r| r > at)
}

impl EgressQueue {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The word clock started a new run: chain positions restart at 0. Entries still held
    /// keep counting, under no position.
    pub fn new_run(&mut self) {
        for e in &mut self.bds {
            e.seq = None;
        }
    }

    /// Whether a fetch at `at` finds a free granule slot.
    pub fn granule_slot_free(&self, at: u64) -> bool {
        self.granules.iter().filter(|&&r| held_at(r, at)).count() < DMA_MM2S_EGRESS_GRANULE_SLOTS as usize
    }

    /// Whether a BD's first access at `at` is within the BD-queue access limit (use (a)).
    pub fn bd_may_open(&self, at: u64) -> bool {
        self.bds.iter().filter(|e| held_at(e.release, at)).count() < DMA_MM2S_BD_QUEUE_ACCESS_LIMIT as usize
    }

    /// A fetch at `at` of BD `seq`; `opens_bd` when it is the BD's first access.
    pub fn note_fetch(&mut self, seq: u64, opens_bd: bool) {
        self.granules.push_back(None);
        if opens_bd {
            self.bds.push_back(BdEntry { seq: Some(seq), release: None });
            let mut released = self.bds.iter().filter(|e| e.release.is_some()).count();
            while released > HISTORY && self.bds.front().is_some_and(|e| e.release.is_some()) {
                self.bds.pop_front();
                released -= 1;
            }
        }
    }

    /// A word left the port at `at`; `ends_granule` / `ends_bd` say whether it closes its
    /// granule's slot / its BD's entry.
    pub fn note_port_word(&mut self, at: u64, ends_granule: bool, ends_bd: bool) {
        let release = at + DMA_MM2S_EGRESS_RELEASE_AFTER_PORT_CYCLES as u64;
        while self.granules.front().is_some_and(|r| r.is_some_and(|r| r <= at)) {
            self.granules.pop_front();
        }
        if ends_granule {
            if let Some(r) = self.granules.iter_mut().find(|r| r.is_none()) {
                *r = Some(release);
            }
        }
        if ends_bd {
            if let Some(e) = self.bds.iter_mut().find(|e| e.release.is_none()) {
                e.release = Some(release);
            }
        }
    }

    /// The BD-queue release step of BD `seq` of the current run, once its last word has left
    /// the port.
    fn release_of(&self, seq: u64) -> Option<u64> {
        self.bds.iter().find(|e| e.seq == Some(seq)).and_then(|e| e.release)
    }

    /// Whether BD `seq` may start sequencing at `now` (use (b)).
    pub fn sequencing_allows(&self, seq: u64, now: u64) -> bool {
        let Some(back) = seq.checked_sub(DMA_MM2S_BD_QUEUE_SEQUENCE_LOOKBACK as u64) else {
            return true;
        };
        self.release_of(back)
            .is_some_and(|r| now >= r + DMA_MM2S_BD_QUEUE_SEQUENCE_AFTER_RELEASE_CYCLES as u64)
    }

    /// The BD-queue floor of lock BD `seq`'s request when its predecessor acquires no lock
    /// (use (c)): `None` when the rule does not apply, `Some(None)` while the BD it looks back
    /// to has not been released yet, `Some(Some(q))` once known.
    pub fn request_floor(&self, seq: u64) -> Option<Option<u64>> {
        if seq < DMA_MM2S_BD_QUEUE_SEQUENCE_LOOKBACK as u64 {
            return None;
        }
        let back = seq - DMA_MM2S_BD_QUEUE_REQUEST_LOOKBACK as u64;
        Some(
            self.release_of(back)
                .map(|r| r + DMA_MM2S_BD_QUEUE_REQUEST_AFTER_RELEASE_CYCLES as u64),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_granule_slot_frees_two_cycles_after_its_last_word_leaves_the_port() {
        let mut q = EgressQueue::default();
        for _ in 0..DMA_MM2S_EGRESS_GRANULE_SLOTS {
            assert!(q.granule_slot_free(0));
            q.note_fetch(0, false);
        }
        assert!(!q.granule_slot_free(0));
        q.note_port_word(10, false, false);
        assert!(!q.granule_slot_free(20), "a word that does not end its granule frees nothing");
        q.note_port_word(11, true, false);
        assert!(!q.granule_slot_free(12));
        assert!(q.granule_slot_free(13));
    }

    #[test]
    fn a_bd_entry_is_held_from_its_first_access_to_its_last_word_plus_two() {
        let mut q = EgressQueue::default();
        for seq in 0..DMA_MM2S_BD_QUEUE_ACCESS_LIMIT as u64 {
            assert!(q.bd_may_open(0));
            q.note_fetch(seq, true);
        }
        assert!(!q.bd_may_open(0));
        q.note_port_word(20, true, true);
        assert!(!q.bd_may_open(21));
        assert!(q.bd_may_open(22));
    }

    #[test]
    fn the_look_backs_find_the_releases_of_the_current_run_only() {
        let mut q = EgressQueue::default();
        for seq in 0..8 {
            q.note_fetch(seq, true);
            q.note_port_word(10 + seq, true, true);
        }
        // BD 0 released at 12: BD 7 sequences from 18, BD 7's request floor is BD 1's 13 + 4.
        assert!(!q.sequencing_allows(7, 17));
        assert!(q.sequencing_allows(7, 18));
        assert!(q.sequencing_allows(6, 0));
        assert_eq!(q.request_floor(6), None);
        assert_eq!(q.request_floor(7), Some(Some(17)));
        q.new_run();
        assert!(!q.sequencing_allows(7, 100));
        assert_eq!(q.request_floor(7), Some(None));
    }
}
