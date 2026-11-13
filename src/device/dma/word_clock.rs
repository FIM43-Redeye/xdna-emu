//! Compute-tile MM2S word clock and lock port: the channel-level state that sequences a BD
//! chain's words ahead of the bank accesses and requests each BD's lock.
//!
//! Both are channel state, not BD state: on silicon (NPU1, `tools/gen_dma_lock_probe.py`
//! sync and chain families) the sequencer runs across BD boundaries and through lock waits,
//! and a BD's lock is requested, granted and shown stalled while the channel is still
//! moving the BD before it. The FSM (`DmaEngine::step_channel_fsm`) only consumes what
//! these two produce: which words may be accessed, and from when a BD's lock is held.
//!
//! Every quantity is kept as an engine step (`DmaEngine::read_clock`) so that the order in
//! which the engine evaluates things within one step does not matter: a word sequenced at
//! step `t` is accessible from `t + 1`, an access presented at `t` frees a lead slot from
//! `t + 1`.

use std::collections::VecDeque;

use xdna_archspec::aie2::timing::{
    DMA_MM2S_BD_SWITCH_ACCESS_GAP_CYCLES, DMA_MM2S_LATE_GRANULE_ACCESS_GAP_CYCLES,
    DMA_MM2S_LOCK_BDS_IN_FLIGHT, DMA_MM2S_LOCK_RELEASE_AFTER_FINISH_CYCLES,
    DMA_MM2S_LOCK_REQUEST_TRIGGER_WORDS, DMA_MM2S_WORD_CLOCK_LEAD_GRANULES,
};

/// Words per sequencer granule (one 16-byte bank access of 32-bit words).
const GRANULE_WORDS: u32 = 4;

/// What the word clock needs of a BD in the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainBdSpec {
    pub bd_index: u8,
    /// Words the BD moves.
    pub words: u32,
    /// The BD's acquire, if it has one: (lock id, acquire value).
    pub acquire: Option<(u8, i8)>,
    /// The chained next BD, if `Use_Next_BD` is set.
    pub next_bd: Option<u8>,
}

/// A BD the word clock knows: the current BD (front) and the chained BDs behind it.
#[derive(Debug, Clone)]
pub struct ChainBd {
    /// Position in the chain since the clock started (unique per channel run).
    pub seq: u64,
    pub spec: ChainBdSpec,
    /// Words sequenced so far (granules started, counting a started granule's words whole).
    started_words: u32,
    /// Step the BD's first word was sequenced.
    pub start_at: Option<u64>,
    /// Step its lock-request trigger word (`DMA_MM2S_LOCK_REQUEST_TRIGGER_WORDS` before its
    /// end, its first word if it is shorter) was sequenced.
    pub trigger_at: Option<u64>,
    /// Words accessed so far, and the step of the latest access.
    accessed_words: u32,
    pub last_access: Option<u64>,
    /// Step the BD's first access was presented to the bank arbiter (won or lost).
    first_presented: Option<u64>,
    /// Step from which the channel holds the BD's lock, once granted (`LockPort`).
    pub granted_at: Option<u64>,
    /// The bank refused one of the BD's accesses (it was served later than first presented).
    pub refused: bool,
}

impl ChainBd {
    fn new(seq: u64, spec: ChainBdSpec) -> Self {
        Self {
            seq,
            spec,
            started_words: 0,
            start_at: None,
            accessed_words: 0,
            last_access: None,
            first_presented: None,
            trigger_at: None,
            granted_at: None,
            refused: false,
        }
    }

    /// Index of the word whose sequencing triggers the next BD's lock request.
    fn trigger_word(&self) -> u32 {
        self.spec.words.saturating_sub(DMA_MM2S_LOCK_REQUEST_TRIGGER_WORDS as u32)
    }

    /// Whether the BD holds the clock to its lead at `now`: it acquires a lock and has not
    /// presented an access before `now`.
    fn parks(&self, now: u64) -> bool {
        self.spec.acquire.is_some() && self.first_presented.is_none_or(|p| p >= now)
    }

    /// Step the BD's first granule has its last word sequenced, once it was started.
    pub fn first_granule_done_at(&self) -> Option<u64> {
        self.start_at.map(|s| s + GRANULE_WORDS.min(self.spec.words).max(1) as u64 - 1)
    }

    /// Whether every word has been accessed.
    pub fn fully_accessed(&self) -> bool {
        self.accessed_words >= self.spec.words
    }
}

/// A granule the sequencer has started and the channel has not fully accessed.
#[derive(Debug, Clone)]
struct SeqGranule {
    bd_seq: u64,
    words: u32,
    /// Step its first word was sequenced; its last is sequenced at `start_at + words - 1`.
    start_at: u64,
    /// Step its access was first presented to the bank arbiter (won or lost).
    presented_at: Option<u64>,
    accessed: u32,
}

impl SeqGranule {
    /// Words sequenced in a step before `at`.
    fn sequenced_before(&self, at: u64) -> u32 {
        (at.saturating_sub(self.start_at) as u32).min(self.words)
    }

    fn done_at(&self) -> u64 {
        self.start_at + self.words as u64 - 1
    }
}

/// Compute-tile MM2S word clock (silicon-fitted sequencer, see `DmaEngine::word_clock_mm2s_cycle`).
///
/// A BD's words split into granules of 4 (the last may be partial). The clock sequences one
/// word per step, granule by granule, in chain order across BD boundaries and during lock
/// waits. While the channel is parked on a lock -- the BD at the head of the accesses, or the
/// BD the clock is sequencing into, acquires a lock and has not presented an access yet -- a
/// granule may start only while fewer than `DMA_MM2S_WORD_CLOCK_LEAD_GRANULES` started
/// granules are still waiting for their access to be presented: a partial granule takes a
/// whole slot, and a slot frees when the access is presented, whether the bank arbiter serves
/// it then or the channel retries it. Unparked, the clock runs on (the accesses are bounded by
/// the channel's staging instead).
///
/// Silicon (NPU1, 2026-11): the parked lead is the fitted model M1g of the lock-timing probes
/// (`tools/gen_dma_lock_probe.py` sync and chain families). That the lead lets go once the
/// parked BD presents its first access is what the contended rows require: a core camping on
/// the bank (ring probe, `the_emulated_mm2s_against_a_camping_core_matches_the_ring_camp`)
/// finds the DMA ready after every lost run, and a lost first access leaves granule 4 at grant
/// + 7 (lock-fetch probe). M1g bounds the lead always; in the uncontended rows it was fitted to
/// the two readings agree.
///
/// A lock-acquiring BD starts sequencing no earlier than the lock release (FINISHED_BD +
/// `DMA_MM2S_LOCK_RELEASE_AFTER_FINISH_CYCLES`) of the run's lock BD
/// `DMA_MM2S_LOCK_BDS_IN_FLIGHT` lock BDs before it, and the clock, sequencing in order, waits
/// with it. Silicon (NPU1, 2026-11, chain families k4f and c1f..c4f, head frame): MEASURED, the
/// sequencing starts the rises and finishes imply (the third of a chain of 4-word free-lock BDs
/// starts at exactly 7, the 64-word BD behind two, three or five of them at 7, 11 or 19);
/// INFERRED, that those are the release cycles of the third lock BD before (three lock BDs in
/// flight between sequencing start and release). No chain with three lock BDs or fewer meets it.
#[derive(Debug, Clone, Default)]
pub struct WordClock {
    /// Step from which the clock sequences.
    from: u64,
    /// The current BD (front) and the chained BDs behind it that the clock has reached, plus
    /// one not yet started (the next BD of the last started one) so its lock can be requested.
    bds: VecDeque<ChainBd>,
    granules: VecDeque<SeqGranule>,
    next_seq: u64,
    /// The BD before the current one, once the current BD took over: its last access (the
    /// BD-switch gap) and its trigger step (the current BD's lock request).
    prev_last_access: Option<u64>,
    prev_trigger_at: Option<u64>,
    prev_seq: Option<u64>,
    /// Step from which the next access may come, after an access whose next granule was not
    /// sequenced yet (`late_gap_allows`).
    late_gap_until: u64,
    /// The run's lock-acquiring BDs in chain order (seq, FINISHED_BD step once finished), the
    /// latest few: the in-flight bound looks `DMA_MM2S_LOCK_BDS_IN_FLIGHT` back.
    lock_bds: VecDeque<(u64, Option<u64>)>,
}

impl WordClock {
    /// Start on `first` (the channel's current BD), sequencing from step `from`.
    pub fn start(&mut self, from: u64, first: ChainBdSpec) {
        *self = Self { from, ..Self::default() };
        self.push_bd(first);
    }

    /// Stop: nothing is sequenced until the next `start`.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn push_bd(&mut self, spec: ChainBdSpec) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.bds.push_back(ChainBd::new(seq, spec));
        if spec.acquire.is_some() {
            self.lock_bds.push_back((seq, None));
            while self.lock_bds.len() > 2 * DMA_MM2S_LOCK_BDS_IN_FLIGHT as usize + 2 {
                self.lock_bds.pop_front();
            }
        }
    }

    /// BD `seq` finished (its FINISHED_BD) at step `at`.
    pub fn note_finished(&mut self, seq: u64, at: u64) {
        if let Some(e) = self.lock_bds.iter_mut().find(|e| e.0 == seq) {
            e.1 = Some(at);
        }
    }

    /// Whether lock BD `seq` may start sequencing at `now`: the lock BD
    /// `DMA_MM2S_LOCK_BDS_IN_FLIGHT` lock BDs before it has released its lock by then.
    fn lock_bd_may_start(&self, seq: u64, now: u64) -> bool {
        let Some(i) = self.lock_bds.iter().position(|e| e.0 == seq) else {
            return true;
        };
        let Some(back) = i.checked_sub(DMA_MM2S_LOCK_BDS_IN_FLIGHT as usize) else {
            return true;
        };
        self.lock_bds[back]
            .1
            .is_some_and(|f| now >= f + DMA_MM2S_LOCK_RELEASE_AFTER_FINISH_CYCLES as u64)
    }

    /// Step from which the clock sequences (the run's first word).
    pub fn from(&self) -> u64 {
        self.from
    }

    /// Whether the clock has been started (a compute MM2S channel with work).
    pub fn is_running(&self) -> bool {
        !self.bds.is_empty()
    }

    pub fn current(&self) -> Option<&ChainBd> {
        self.bds.front()
    }

    pub fn bd(&self, seq: u64) -> Option<&ChainBd> {
        self.bds.iter().find(|b| b.seq == seq)
    }

    fn bd_mut(&mut self, seq: u64) -> Option<&mut ChainBd> {
        self.bds.iter_mut().find(|b| b.seq == seq)
    }

    /// The BD before `seq` in the chain: (its trigger step, its last access, whether it has
    /// been fully accessed). `None` if `seq` is the first BD of the run.
    pub fn predecessor(&self, seq: u64) -> Option<(Option<u64>, Option<u64>, bool)> {
        let prev = seq.checked_sub(1)?;
        if let Some(b) = self.bd(prev) {
            return Some((b.trigger_at, b.last_access, b.fully_accessed()));
        }
        (self.prev_seq == Some(prev)).then_some((self.prev_trigger_at, self.prev_last_access, true))
    }

    /// Record the grant of BD `seq` (held from step `at`).
    pub fn set_granted(&mut self, seq: u64, at: u64) {
        if let Some(b) = self.bd_mut(seq) {
            b.granted_at = Some(at);
        }
    }

    pub fn bds(&self) -> impl Iterator<Item = &ChainBd> {
        self.bds.iter()
    }

    /// One step of the sequencer at `now`, run before anything else of the channel's step.
    /// `next_of` looks up a BD by index (the chain's next BD).
    pub fn tick(&mut self, now: u64, next_of: impl Fn(u8) -> Option<ChainBdSpec>) {
        if self.bds.is_empty() || now < self.from {
            return;
        }
        // The granule being sequenced, if any, takes this step's word.
        if self.granules.back().is_some_and(|g| g.done_at() >= now) {
            return;
        }
        // The first BD with a word left to sequence (zero-word BDs are passed over).
        let Some(pos) = self.bds.iter().position(|b| b.started_words < b.spec.words) else {
            return;
        };
        if self.bds[pos].started_words == 0
            && self.bds[pos].spec.acquire.is_some()
            && !self.lock_bd_may_start(self.bds[pos].seq, now)
        {
            return;
        }
        let head = self.granules.front().and_then(|g| self.bd(g.bd_seq)).unwrap_or(&self.bds[pos]);
        if head.parks(now) || self.bds[pos].parks(now) {
            let waiting = self.granules.iter().filter(|g| g.presented_at.is_none_or(|p| p >= now)).count();
            if waiting >= DMA_MM2S_WORD_CLOCK_LEAD_GRANULES as usize {
                return;
            }
        }
        let bd = &mut self.bds[pos];
        let first_word = bd.started_words;
        let words = GRANULE_WORDS.min(bd.spec.words - first_word);
        bd.started_words += words;
        if bd.start_at.is_none() {
            bd.start_at = Some(now);
        }
        let trigger = bd.trigger_word();
        if (first_word..first_word + words).contains(&trigger) {
            bd.trigger_at = Some(now + (trigger - first_word) as u64);
        }
        let (seq, next_bd) = (bd.seq, bd.spec.next_bd);
        self.granules.push_back(SeqGranule {
            bd_seq: seq,
            words,
            start_at: now,
            presented_at: None,
            accessed: 0,
        });
        // Keep the next BD of the last started one in view, so its lock can be requested.
        if pos + 1 == self.bds.len() {
            if let Some(spec) = next_bd.and_then(&next_of) {
                self.push_bd(spec);
            }
        }
    }

    /// Words of the current BD sequenced before step `at` and not yet accessed.
    pub fn accessible_words(&self, at: u64) -> u32 {
        let Some(cur) = self.bds.front() else { return 0 };
        self.granules
            .iter()
            .filter(|g| g.bd_seq == cur.seq)
            .map(|g| g.sequenced_before(at).saturating_sub(g.accessed))
            .sum()
    }

    /// Whether the current BD's next access may happen at `at` as far as the BD switch goes:
    /// a BD's first access comes `DMA_MM2S_BD_SWITCH_ACCESS_GAP_CYCLES` after the previous
    /// BD's last.
    pub fn switch_gap_allows(&self, at: u64) -> bool {
        let first = self.bds.front().is_none_or(|b| b.accessed_words == 0);
        !first
            || self
                .prev_last_access
                .is_none_or(|p| at >= p + DMA_MM2S_BD_SWITCH_ACCESS_GAP_CYCLES as u64)
    }

    /// Whether the next access may happen at `at` as far as the access before it goes: after an
    /// access that the bank refused at least once, a granule that was not accessible before the
    /// cycle it was served comes `DMA_MM2S_LATE_GRANULE_ACCESS_GAP_CYCLES` after it; one that
    /// was follows the next cycle, and an access served when first presented leaves the next
    /// one to the word clock. Silicon (NPU1, 2026-11): in the producer probe a refused access is served when the
    /// core's store dwell leaves its bank, and the next granule, sequenced meanwhile, asks two
    /// cycles later (grant + 23 -> 25, then every 4 cycles in step with the dwell, the core
    /// winning both cycles of each contact); in the ring camps a refused access is followed the
    /// next cycle by a granule sequenced long before (served g + 6, next at g + 7).
    pub fn late_gap_allows(&self, at: u64) -> bool {
        at >= self.late_gap_until
    }

    /// The current BD's next access was presented to the bank arbiter at `at`.
    pub fn note_presented(&mut self, at: u64) {
        let Some(cur) = self.bds.front_mut() else {
            return;
        };
        cur.first_presented.get_or_insert(at);
        let cur = cur.seq;
        if let Some(g) = self.granules.iter_mut().find(|g| g.bd_seq == cur && g.accessed < g.words) {
            g.presented_at.get_or_insert(at);
        }
    }

    /// The current BD accessed `words` words at `at`.
    pub fn note_access(&mut self, at: u64, mut words: u32) {
        // Refused before: the access was first presented in an earlier step.
        let refused = self
            .bds
            .front()
            .and_then(|cur| self.granules.iter().find(|g| g.bd_seq == cur.seq && g.accessed < g.words))
            .and_then(|g| g.presented_at)
            .is_some_and(|p| p < at);
        self.note_presented(at);
        let Some(cur) = self.bds.front_mut() else { return };
        cur.refused |= refused;
        cur.accessed_words += words;
        cur.last_access = Some(at);
        let seq = cur.seq;
        while words > 0 {
            let Some(g) = self.granules.iter_mut().find(|g| g.bd_seq == seq && g.accessed < g.words) else {
                break;
            };
            let take = words.min(g.words - g.accessed);
            g.accessed += take;
            g.presented_at.get_or_insert(at);
            words -= take;
        }
        self.granules.retain(|g| g.accessed < g.words);
        // After a refused access, a next granule (of this BD or a later one) that was not
        // accessible before this cycle comes the gap later.
        let next_ready = self
            .granules
            .front()
            .is_some_and(|g| g.sequenced_before(at.saturating_sub(1)) >= g.words);
        if refused && !next_ready {
            self.late_gap_until = at + DMA_MM2S_LATE_GRANULE_ACCESS_GAP_CYCLES as u64;
        }
    }

    /// The current BD is done and the channel moves to the next one. Returns false if the
    /// clock's next BD is not `next_bd` (the chain changed under it): the caller restarts it.
    pub fn advance(&mut self, next_bd: u8) -> bool {
        let Some(done) = self.bds.pop_front() else {
            return false;
        };
        self.granules.retain(|g| g.bd_seq != done.seq);
        self.prev_last_access = done.last_access;
        self.prev_trigger_at = done.trigger_at;
        self.prev_seq = Some(done.seq);
        self.bds.front().is_some_and(|b| b.spec.bd_index == next_bd)
    }
}
