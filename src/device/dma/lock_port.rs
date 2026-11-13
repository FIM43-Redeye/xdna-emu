//! Compute-tile MM2S lock port: when a BD's lock is requested and granted.

use super::word_clock::WordClock;
use xdna_archspec::aie2::timing::{
    DMA_MM2S_LOCK_REQUEST_AFTER_GRANT_CYCLES, DMA_MM2S_LOCK_REQUEST_TRIGGER_LATENCY_CYCLES,
};

/// The lock request of the next lock-acquiring BD, and when it may be granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Request {
    /// The run's first BD: no previous BD or lock grant to time a request from. Presented
    /// every cycle from `first_grant` (not measured; see `LockPort`).
    Ungated { first_grant: u64 },
    /// Presented from the request step `q` on.
    Gridded { q: u64 },
}

/// The BD whose lock the channel is requesting.
#[derive(Debug, Clone, Copy)]
struct Target {
    seq: u64,
    lock_id: u8,
    acquire_value: i8,
    request: Option<Request>,
}

/// Compute-tile MM2S lock port: requests BD locks on silicon's timing.
///
/// Silicon (NPU1, `tools/gen_dma_lock_probe.py` sync and chain families, every design with
/// at most four MM2S BDs): for a BD that acquires a lock, with g the grant of the channel's
/// previous lock-acquiring BD and T the step the previous BD's trigger word was sequenced,
/// the request is Q = max(g + 3, T + 3). The request is made while the channel is still on
/// the BDs before (a free lock behind a 4-word lock BD and a 13-word lock-free BD is granted
/// at g + 3, 5 cycles before the 13-word BD finishes), so it is channel state, not BD state.
///
/// Not measured, chosen here: a run's first BD (no previous BD, no previous grant) is
/// requested every cycle from the cycle it becomes the target; a BD whose run has no previous
/// lock grant but has a previous BD takes Q = T + 3.
#[derive(Debug, Clone, Default)]
pub struct LockPort {
    /// Grant step of the run's latest granted lock BD and its seq.
    last_grant: Option<u64>,
    last_seq: Option<u64>,
    target: Option<Target>,
}

impl LockPort {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The lock the channel presents to the arbiter at step `at`, if it presents one: (lock
    /// id, acquire value).
    pub fn presentation(&self, at: u64) -> Option<(u8, i8)> {
        let t = self.target?;
        let ok = match t.request? {
            Request::Ungated { first_grant } => at >= first_grant,
            Request::Gridded { q } => at >= q,
        };
        ok.then_some((t.lock_id, t.acquire_value))
    }

    /// The arbiter granted this step's presentation (made at step `at`).
    pub fn granted(&mut self, wc: &mut WordClock, at: u64) {
        let Some(t) = self.target else { return };
        wc.set_granted(t.seq, at);
        self.last_grant = Some(at);
        self.last_seq = Some(t.seq);
    }

    /// End of the channel's step at `now`: retire a target whose grant is held, pick the
    /// next lock-acquiring BD and time its request once its inputs are known.
    pub fn update(&mut self, wc: &WordClock, now: u64) {
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
                    Target { seq: b.seq, lock_id, acquire_value, request: None }
                });
        }
        let Some(mut t) = self.target else { return };
        if t.request.is_none() {
            let from_grant = self.last_grant.map(|g| g + DMA_MM2S_LOCK_REQUEST_AFTER_GRANT_CYCLES as u64);
            t.request = match wc.predecessor(t.seq) {
                // Q needs the previous BD's trigger word sequenced.
                Some((Some(trigger), _, _)) => {
                    let q = trigger + DMA_MM2S_LOCK_REQUEST_TRIGGER_LATENCY_CYCLES as u64;
                    Some(Request::Gridded { q: from_grant.map_or(q, |g| g.max(q)) })
                }
                Some((None, _, _)) => None,
                None => Some(match from_grant {
                    Some(q) => Request::Gridded { q },
                    None => Request::Ungated { first_grant: now + 1 },
                }),
            };
            self.target = Some(t);
        }
    }

    /// The BD whose lock is being requested, if any.
    pub fn target_seq(&self) -> Option<u64> {
        self.target.map(|t| t.seq)
    }
}
