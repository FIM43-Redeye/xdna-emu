//! Compute-tile S2MM lock port: when a BD's lock is granted, and when its first write may follow.

use xdna_archspec::aie2::timing::{
    DMA_MM2S_LOCK_GRANT_BLOCKED_PHASE, DMA_MM2S_LOCK_GRANT_PERIOD_CYCLES, DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES,
    DMA_S2MM_LOCK_GRID_AFTER_GRANT_CYCLES, DMA_S2MM_LOCK_TO_WRITE_CYCLES,
};

/// Compute-tile S2MM lock port.
///
/// Silicon (NPU1, 2026-11, `tools/gen_dma_lock_probe.py` sl family and objectfifo_turns_walkers
/// re-traced with lock selectors): a compute S2MM BD parked on a lock that a core releases at R
/// (LOCK_n_REL) is granted (STALLED_LOCK falls) at R + 2 or R + 3, never earlier, first-BD and
/// chained alike: the lock unit's same grid as the compute MM2S lock port (`LockPort`), period
/// `DMA_MM2S_LOCK_GRANT_PERIOD_CYCLES` with one blocked cycle. After an earlier grant g of the
/// channel the grid is counted from g + `DMA_S2MM_LOCK_GRID_AFTER_GRANT_CYCLES` (walkers: R + 3
/// exactly when (R - g) mod 3 = 2, 119 of 119 iterations). The task's first lock BD has no grant
/// to count from; its grid is counted from the task's anchor (chosen: the point a compute MM2S's
/// word clock would start, whose first-BD grid silicon measured; the sl rows cannot show the
/// S2MM's, the channel being parked before the trace starts). The channel holds the grant
/// `DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES` after the arbiter gives it, and its first bank write
/// comes `DMA_S2MM_LOCK_TO_WRITE_CYCLES` after that.
#[derive(Debug, Clone, Default)]
pub struct S2mmPort {
    /// Grant step (held from) of the run's latest granted lock BD.
    last_grant: Option<u64>,
    /// Grid origin of the task's first lock BD.
    task_anchor: Option<u64>,
    /// Step from which the current BD holds its lock (STALLED_LOCK falls there).
    held_from: Option<u64>,
    /// Step from which the current lock-started BD may write memory.
    write_from: u64,
    /// The STALLED_LOCK fall is still to be traced.
    fall_pending: bool,
}

impl S2mmPort {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// A task starts: no grant yet, the grid counted from `anchor`.
    pub fn start_task(&mut self, anchor: u64) {
        *self = Self { task_anchor: Some(anchor), ..Self::default() };
    }

    fn origin(&self) -> Option<u64> {
        self.last_grant
            .map(|g| g + DMA_S2MM_LOCK_GRID_AFTER_GRANT_CYCLES as u64)
            .or(self.task_anchor)
    }

    /// Whether the channel presents its acquire at step `at` (the arbiter's grant step; the
    /// channel would hold the lock from `at + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES`).
    pub fn presents(&self, at: u64) -> bool {
        let grant = at + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES as u64;
        let Some(origin) = self.origin() else { return true };
        let period = DMA_MM2S_LOCK_GRANT_PERIOD_CYCLES as i64;
        (grant as i64 - origin as i64).rem_euclid(period) != DMA_MM2S_LOCK_GRANT_BLOCKED_PHASE as i64
    }

    /// The arbiter granted the acquire presented at step `at`.
    pub fn granted(&mut self, at: u64) {
        let held = at + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES as u64;
        self.last_grant = Some(held);
        self.held_from = Some(held);
        self.write_from = held + DMA_S2MM_LOCK_TO_WRITE_CYCLES as u64;
        self.fall_pending = true;
    }

    /// Whether STALLED_LOCK falls at `now` (once, when the grant is held).
    pub fn take_fall(&mut self, now: u64) -> bool {
        if self.fall_pending && self.held_from.is_some_and(|h| now >= h) {
            self.fall_pending = false;
            return true;
        }
        false
    }

    /// Step from which the current lock-started BD may write memory.
    pub fn write_from(&self) -> u64 {
        self.write_from
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grant after a release reaching the lock unit at `r` (seen by the requester from
    /// r + 1, as `LockPort`): the first arbiter step that presents, held one later.
    fn grant_after(port: &S2mmPort, r: u64) -> u64 {
        (r + 1..).find(|&at| port.presents(at)).unwrap() + DMA_MM2S_LOCK_GRANT_VISIBLE_CYCLES as u64
    }

    #[test]
    fn a_grant_follows_the_release_by_two_or_three_on_the_walkers_grid() {
        // objectfifo_turns_walkers, silicon: R + 3 exactly when (R - previous grant) mod 3 = 2.
        for g in [1000u64, 1001, 1002] {
            for d in 700u64..712 {
                let mut port = S2mmPort::default();
                port.start_task(0);
                port.granted(g - 1);
                let r = g + d;
                let want = if d % 3 == 2 { r + 3 } else { r + 2 };
                assert_eq!(grant_after(&port, r), want, "previous grant {g}, R - g = {d}");
            }
        }
    }

    #[test]
    fn the_first_write_comes_two_cycles_after_the_grant() {
        let mut port = S2mmPort::default();
        port.granted(99);
        assert_eq!(port.write_from(), 102);
        assert!(!port.take_fall(99));
        assert!(port.take_fall(100));
        assert!(!port.take_fall(101), "the fall is traced once");
    }

    #[test]
    fn the_first_lock_bd_of_a_task_counts_its_grid_from_the_task_anchor() {
        let mut port = S2mmPort::default();
        port.start_task(10);
        // blocked when (grant - 10) mod 3 = 2: grants 12, 15, ... are skipped.
        assert_eq!(grant_after(&port, 9), 11);
        assert_eq!(grant_after(&port, 10), 13);
        assert_eq!(grant_after(&port, 11), 13);
    }
}
