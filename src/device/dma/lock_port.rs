//! Compute-tile MM2S lock port: when a BD's lock is requested and granted.

use std::collections::VecDeque;

use super::egress_queue::EgressQueue;
use super::word_clock::WordClock;
use xdna_archspec::aie2::timing::{
    DMA_MM2S_LOCK_GRANT_BLOCKED_PHASE, DMA_MM2S_LOCK_GRANT_PERIOD_CYCLES, DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES,
    DMA_MM2S_LOCK_RELEASE_AFTER_FINISH_CYCLES, DMA_MM2S_LOCK_REQUEST_AFTER_GRANT_CYCLES,
    DMA_MM2S_LOCK_REQUEST_TRIGGER_LATENCY_CYCLES, DMA_MM2S_LOCK_STALL_BEFORE_FIRST_GRANULE_CYCLES,
    DMA_MM2S_LOCK_STALL_BEFORE_LAST_ACCESS_CYCLES, DMA_MM2S_LOCK_STALL_BEFORE_REQUEST_CYCLES,
};

/// The lock request of the next lock-acquiring BD, and when it may be granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Request {
    /// The run's first BD: no previous BD or lock grant to time a request from. Presented
    /// from `first_grant - 1` on the grid counted from `origin`, the run's first sequenced word
    /// (see `LockPort`).
    Ungated { first_grant: u64, origin: u64 },
    /// From the request step `q` on, grantable in 2 of every 3 steps.
    Gridded { q: u64 },
}

/// The BD whose lock the channel is requesting.
#[derive(Debug, Clone, Copy)]
struct Target {
    seq: u64,
    lock_id: u8,
    acquire_value: i8,
    request: Option<Request>,
    /// Granted (held from the BD's `granted_at`): nothing more is presented for it.
    granted: bool,
}

/// Compute-tile MM2S lock port: requests and holds BD locks on silicon's timing.
///
/// Silicon (NPU1, `tools/gen_dma_lock_probe.py` sync and chain families, every design with
/// at most four MM2S BDs): for a BD that acquires a lock, with g the grant of the channel's
/// previous lock-acquiring BD and T the step the previous BD's trigger word was sequenced,
/// the request is Q = max(g + 3, T + 3), moved off DEAD, the release slots (FINISHED_BD + 4)
/// of the run's earlier lock BDs. The grant G (STALLED_LOCK falling) is the first step >=
/// max(Q, R + 2) with (G - Q) mod 3 != 2 and G not in DEAD, R the release that made the lock
/// available; LOCK_SEL*_ACQ_GE fires at G - 1 (the arbiter grants then,
/// `DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES` before the channel holds it), and a release reaches
/// this requester one step after the arbiter applied it. STALLED_LOCK shows on [P, G - 1] with
/// P = max(D, Q, the previous BD's last access) - 1, D the step the BD's first granule has its
/// last word sequenced.
///
/// DEAD and D as stated come from the short-BD chains (chain families k4f, c1f..c4f): a chain
/// of 4-word free-lock BDs is never granted at the fourth BD's FINISHED_BD + 4 two BDs before
/// the waiting one (MEASURED; that it is that BD's own release slot is INFERRED), and a 3-word
/// BD whose first word is sequenced at 7 shows STALLED_LOCK from 8 (MEASURED). Designs with
/// at most four BDs have one earlier lock BD whose slot can matter and a 4-word first granule
/// (D - 1 = S + 2), where the two readings agree.
///
/// Outside the rule, chosen here: a run's first BD (no previous BD, no previous grant) is
/// requested every cycle from the cycle it becomes the target, with Q its earliest grant, so a
/// release is always followed by the grant at + 2. Silicon's first-BD lock-fetch rows give + 2
/// in 39 rows and + 3 in 12, a phase this rule does not reproduce (open: its anchor is not
/// known). A BD whose run has no previous lock grant but has a previous BD takes Q = T + 3
/// (not measured on its own). The predicted last-access term of P assumes the access is not
/// refused by the bank (not measured).
#[derive(Debug, Clone, Default)]
pub struct LockPort {
    /// Grant step of the run's latest granted lock BD, and its seq.
    last_grant: Option<u64>,
    last_seq: Option<u64>,
    /// DEAD: the release slots (FINISHED_BD + 4) of the run's latest finished lock BDs.
    release_slots: VecDeque<u64>,
    target: Option<Target>,
    /// Grid origin of the BD `seq` whose predecessor had refused accesses: that BD's
    /// FINISHED_BD - 1 (seq, origin).
    finish_origin: Option<(u64, u64)>,
}

/// Whether a grant at `grant` is open on the grid counted from `origin` (period
/// `DMA_MM2S_LOCK_GRANT_PERIOD_CYCLES`, one blocked phase).
fn open_on_grid(grant: u64, origin: u64) -> bool {
    let period = DMA_MM2S_LOCK_GRANT_PERIOD_CYCLES as i64;
    (grant as i64 - origin as i64).rem_euclid(period) != DMA_MM2S_LOCK_GRANT_BLOCKED_PHASE as i64
}

impl LockPort {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Whether `at` is in DEAD: an earlier lock BD's release slot.
    fn dead(&self, at: u64) -> bool {
        self.release_slots.contains(&at)
    }

    /// The request step, moved off DEAD.
    fn effective_q(&self, mut q: u64) -> u64 {
        while self.dead(q) {
            q += 1;
        }
        q
    }

    /// The lock the channel presents to the arbiter at step `at` (whose grant would be held
    /// from `at + 1`), if it presents one: (lock id, acquire value).
    pub fn presentation(&self, at: u64) -> Option<(u8, i8)> {
        let t = self.target.filter(|t| !t.granted)?;
        let grant = at + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES as u64;
        let ok = match t.request? {
            Request::Ungated { first_grant, origin } => grant >= first_grant && open_on_grid(grant, origin),
            Request::Gridded { q } => {
                let q = self.effective_q(q);
                let origin = self.finish_origin.filter(|&(seq, _)| seq == t.seq).map_or(q, |(_, o)| o);
                grant >= q && open_on_grid(grant, origin) && !self.dead(grant)
            }
        };
        ok.then_some((t.lock_id, t.acquire_value))
    }

    /// The arbiter granted this step's presentation (made at step `at`).
    pub fn granted(&mut self, wc: &mut WordClock, at: u64) {
        let Some(t) = self.target.as_mut() else { return };
        t.granted = true;
        let t = *t;
        let g = at + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES as u64;
        wc.set_granted(t.seq, g);
        self.last_grant = Some(g);
        self.last_seq = Some(t.seq);
    }

    /// The current BD `seq` finished at `at`; `acquires` says it held a lock (its release slot
    /// is FINISHED_BD + 4), `refused` that the bank refused one of its accesses: then the next
    /// BD's grid is counted from this FINISHED_BD - 1 (silicon, refused-access probe: the grant
    /// is release + 3 exactly when (release - FINISHED_BD) mod 3 = 2, 144 of 144 runs, also where
    /// late refusals moved the finish; for a BD served as presented that origin is the request's
    /// own residue, so only refused predecessors change).
    pub fn note_finished(&mut self, seq: u64, acquires: bool, refused: bool, at: u64) {
        if refused {
            self.finish_origin = Some((seq + 1, at.saturating_sub(1)));
        }
        if acquires {
            self.release_slots
                .push_back(at + DMA_MM2S_LOCK_RELEASE_AFTER_FINISH_CYCLES as u64);
            // Slots further back lie before any request still being timed.
            while self.release_slots.len() > 8 {
                self.release_slots.pop_front();
            }
        }
    }

    /// End of the channel's step at `now`: retire a target whose grant is held, pick the
    /// next lock-acquiring BD and time its request once its inputs are known.
    pub fn update(&mut self, wc: &WordClock, egress: &EgressQueue, now: u64) {
        if let Some(t) = self.target {
            match wc.bd(t.seq) {
                Some(b) if b.granted_at.is_some_and(|g| g > now) => return,
                Some(b) if b.granted_at.is_none() => {}
                _ => self.target = None,
            }
        }
        if self.target.is_none() {
            let after = self.last_seq;
            self.target = wc
                .bds()
                .find(|b| {
                    after.is_none_or(|s| b.seq > s) && b.spec.acquire.is_some() && b.granted_at.is_none()
                })
                .map(|b| {
                    let (lock_id, acquire_value) = b.spec.acquire.expect("filtered");
                    Target { seq: b.seq, lock_id, acquire_value, request: None, granted: false }
                });
        }
        let Some(mut t) = self.target else { return };
        if t.request.is_none() {
            let from_grant = self.last_grant.map(|g| g + DMA_MM2S_LOCK_REQUEST_AFTER_GRANT_CYCLES as u64);
            // Deep in a chain, behind a lock-free BD, the BD queue times the request in place
            // of the trigger word (`EgressQueue::request_floor`).
            let queued = egress
                .request_floor(t.seq)
                .filter(|_| wc.acquires_lock(t.seq.wrapping_sub(1)) == Some(false));
            t.request = if let Some(floor) = queued {
                floor.map(|q| Request::Gridded { q: from_grant.map_or(q, |g| g.max(q)) })
            } else {
                match wc.predecessor(t.seq) {
                    // Q needs the previous BD's trigger word sequenced.
                    Some((Some(trigger), _, _)) => {
                        let q = trigger + DMA_MM2S_LOCK_REQUEST_TRIGGER_LATENCY_CYCLES as u64;
                        Some(Request::Gridded { q: from_grant.map_or(q, |g| g.max(q)) })
                    }
                    Some((None, _, _)) => None,
                    None => Some(match from_grant {
                        Some(q) => Request::Gridded { q },
                        None => Request::Ungated {
                            first_grant: now + 1 + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES as u64,
                            origin: wc.from(),
                        },
                    }),
                }
            };
            self.target = Some(t);
        }
    }

    /// Whether STALLED_LOCK rises at (or stays up through) `now`: the target's lock is not held
    /// yet and `now` >= P. `prev_last_access_next` says the previous BD's last access will be
    /// presented at `now + 1` (its accesses still running).
    pub fn stall_level(&self, wc: &WordClock, now: u64, prev_last_access_next: bool) -> bool {
        let Some(t) = self.target else { return false };
        let Some(bd) = wc.bd(t.seq) else { return false };
        if bd.granted_at.is_some_and(|g| g <= now) {
            return false;
        }
        let Some(first_done) = bd.first_granule_done_at() else {
            return false;
        };
        if first_done > now + DMA_MM2S_LOCK_STALL_BEFORE_FIRST_GRANULE_CYCLES as u64 {
            return false;
        }
        let q = match t.request {
            None => return false,
            Some(Request::Ungated { first_grant, .. }) => first_grant,
            Some(Request::Gridded { q }) => self.effective_q(q),
        };
        if q > now + DMA_MM2S_LOCK_STALL_BEFORE_REQUEST_CYCLES as u64 {
            return false;
        }
        // P >= the previous BD's last access - 1: that access is done, or comes next step.
        debug_assert_eq!(DMA_MM2S_LOCK_STALL_BEFORE_LAST_ACCESS_CYCLES, 1, "the look-ahead is one step");
        match wc.predecessor(t.seq) {
            None | Some((_, _, true)) => true,
            Some(_) => prev_last_access_next,
        }
    }

    /// Whether the target's lock is held by `now` (or there is no target): the level falls.
    pub fn target_held(&self, wc: &WordClock, now: u64) -> bool {
        self.target
            .and_then(|t| wc.bd(t.seq))
            .is_none_or(|b| b.granted_at.is_some_and(|g| g <= now))
    }

    /// The BD before the target, when it is the channel's current BD with accesses left.
    pub fn predecessor_is_current(&self, wc: &WordClock) -> bool {
        let Some(t) = self.target else { return false };
        wc.current().is_some_and(|c| c.seq + 1 == t.seq && !c.fully_accessed())
    }

    /// The BD whose lock is being requested, if any.
    pub fn target_seq(&self) -> Option<u64> {
        self.target.map(|t| t.seq)
    }
}
