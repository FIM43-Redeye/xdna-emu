//! The core's instruction-fetch front end: the stream of 16-byte program-memory lines read
//! ahead of issue, and the zero-overhead-loop decisions taken on that stream.
//!
//! Silicon reads program memory in aligned lines ([`FETCH_LINE_BYTES`]) and keeps them
//! [`FETCH_AHEAD_LINES`] lines ahead of the line being issued. The loop registers are consulted
//! when a line is *fetched*, not when its bundles issue:
//!
//! - a line whose address equals LE (exactly: a LE that is not a multiple of 16 never matches)
//!   ends the loop body. If LC is not 1, LC is decremented (LC = 0 wraps and so loops on) and
//!   the next line fetched is the line holding LS; the redirect reaches issue once the last
//!   bundle of the LE line has issued, so every bundle of the LE line runs on every pass;
//! - a redirect disarms the comparator, and fetching a line whose address equals LS exactly
//!   re-arms it. An LS that is not a multiple of 16 therefore allows exactly one redirect, to
//!   the start of its line;
//! - a change of LE arms the comparator (it starts disarmed).
//!
//! The decision therefore uses the register values at fetch time. A write to LE, LC or LS is
//! seen by the fetch only for lines fetched after the writing bundle issued, and the lines up to
//! four beyond the line of the next PC are already fetched when that bundle's writes land: LE
//! written less than ~72 bytes before the loop end is not in time, and LC or LS written that late
//! is read stale. Evidence and the probe variants: `docs/fidelity-gaps/core-compute-timing.md`
//! "Zero-overhead loop set-up", `tools/gen_zol_probe.py`.
//!
//! A taken branch flushes the stream (undoing the LC decrements of the flushed lines).

use std::collections::VecDeque;

use super::timing_context::{fetch_line_start, FETCH_LINE_BYTES};

/// Lines fetched beyond the line of the next PC when a bundle issues. Silicon: an LE line at
/// least five lines beyond the line of the end of the last register-writing bundle is in time
/// (`tools/gen_zol_probe.py` `last_*`: the threshold sits at `LE - bundle_start >= 72` for a
/// 6-byte `movxm`, `>= 70` for the 4-byte `add.nc lc`, for every bundle size tried).
pub const FETCH_AHEAD_LINES: usize = 4;

/// LS, LE and LC as the fetch reads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopRegs {
    pub ls: u32,
    pub le: u32,
    pub lc: u32,
}

/// A loop-back decided when its LE line was fetched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopRedirect {
    /// Line the fetch continues at (the line holding LS).
    pub target: u32,
    pub lc_before: u32,
    pub lc_after: u32,
    pub le: u32,
    pub ls: u32,
}

#[derive(Debug, Clone, Copy)]
struct FetchedLine {
    addr: u32,
    redirect: Option<LoopRedirect>,
}

#[derive(Debug, Clone, Default)]
pub struct FetchStage {
    /// Fetched lines in stream order; the front is the line being issued.
    queue: VecDeque<FetchedLine>,
    /// The next line the fetch reads.
    next_line: u32,
    /// The loop-end comparator.
    armed: bool,
    /// The LE the comparator was last armed for.
    seen_le: u32,
}

impl FetchStage {
    /// Before the bundle at `pc` (ending at `end`) issues: read lines ahead until the stream
    /// reaches [`FETCH_AHEAD_LINES`] beyond the line `end` is in. Uses `regs` as the earlier
    /// bundles left them, so the bundle's own writes are seen from the next call on. A PC that is
    /// not the front line of the stream (a taken branch) restarts the stream there.
    pub fn fetch_ahead(&mut self, pc: u32, end: u32, regs: &mut LoopRegs) {
        let line = fetch_line_start(pc);
        if self.queue.front().map(|l| l.addr) != Some(line) {
            self.flush(regs);
            self.next_line = line;
        }
        if regs.le != self.seen_le {
            self.seen_le = regs.le;
            self.armed = true;
        }
        let issue_line = usize::from(end >= line.wrapping_add(FETCH_LINE_BYTES));
        while self.queue.len() < issue_line + 1 + FETCH_AHEAD_LINES {
            self.fetch_one(regs);
        }
    }

    fn fetch_one(&mut self, regs: &mut LoopRegs) {
        let addr = self.next_line;
        if addr == regs.ls {
            self.armed = true;
        }
        let mut redirect = None;
        if self.armed && addr == regs.le && regs.lc != 1 {
            let lc_before = regs.lc;
            regs.lc = lc_before.wrapping_sub(1);
            self.armed = false;
            redirect = Some(LoopRedirect {
                target: fetch_line_start(regs.ls),
                lc_before,
                lc_after: regs.lc,
                le: regs.le,
                ls: regs.ls,
            });
        }
        self.next_line = redirect.map_or(addr.wrapping_add(FETCH_LINE_BYTES), |r| r.target);
        self.queue.push_back(FetchedLine { addr, redirect });
    }

    /// After the bundle issued and the sequential PC is `next_pc`: when issue has left the front
    /// line, drop it and return the loop-back it carried, if any.
    pub fn retire(&mut self, next_pc: u32) -> Option<LoopRedirect> {
        let front = self.queue.front()?;
        if next_pc.wrapping_sub(front.addr) < FETCH_LINE_BYTES {
            return None;
        }
        self.queue.pop_front().and_then(|l| l.redirect)
    }

    /// Drop the stream, giving back the LC decrements of the flushed loop-backs.
    pub fn flush(&mut self, regs: &mut LoopRegs) {
        for line in self.queue.drain(..) {
            if line.redirect.is_some() {
                regs.lc = regs.lc.wrapping_add(1);
                self.armed = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: u32 = FETCH_LINE_BYTES;

    /// Run bundles of `sizes` bytes from `pc` through the stage, applying `writes` (bundle index
    /// -> register update) after the bundle's fetch step; returns the issued PCs.
    fn run(
        stage: &mut FetchStage,
        regs: &mut LoopRegs,
        mut pc: u32,
        sizes: &[u32],
        writes: &[(usize, fn(&mut LoopRegs))],
    ) -> Vec<u32> {
        let mut issued = vec![];
        for (i, &size) in sizes.iter().enumerate() {
            stage.fetch_ahead(pc, pc + size, regs);
            issued.push(pc);
            for (at, f) in writes {
                if *at == i {
                    f(regs);
                }
            }
            pc += size;
            if let Some(r) = stage.retire(pc) {
                pc = r.target;
            }
        }
        issued
    }

    #[test]
    fn a_line_equal_to_le_loops_back_and_the_whole_line_runs_each_pass() {
        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x100, le: 0x140, lc: 3 };
        // 2-byte bundles from the loop start: 0x100..0x150 is 5 lines (LE line 0x140 included)
        let sizes = vec![2; 8 * 5 * 3];
        let pcs = run(&mut stage, &mut regs, 0x100, &sizes, &[]);
        let at_le_line = pcs.iter().filter(|&&p| p == 0x140).count();
        let at_post_le = pcs.iter().filter(|&&p| p == 0x142).count();
        assert_eq!(
            (at_le_line, at_post_le),
            (3, 3),
            "LC=3 gives three passes of the LE line, bundles after LE included"
        );
        assert_eq!(regs.lc, 1, "the last pass leaves LC at 1");
    }

    #[test]
    fn lc_one_falls_through_without_decrement_and_lc_zero_loops_on_wrapping() {
        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x100, le: 0x120, lc: 1 };
        let pcs = run(&mut stage, &mut regs, 0x100, &vec![2; 40], &[]);
        assert_eq!(pcs.iter().filter(|&&p| p == 0x120).count(), 1);
        assert_eq!(regs.lc, 1);

        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x100, le: 0x120, lc: 0 };
        let pcs = run(&mut stage, &mut regs, 0x100, &vec![2; 8 * 3 * 4], &[]);
        assert!(pcs.iter().filter(|&&p| p == 0x120).count() >= 3, "LC=0 does not end the loop");
        assert!(regs.lc > 0xFFFF_FF00, "LC counted down through zero: {:#x}", regs.lc);
    }

    #[test]
    fn a_le_that_is_not_a_line_start_never_matches() {
        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x100, le: 0x124, lc: 4 };
        let pcs = run(&mut stage, &mut regs, 0x100, &vec![2; 60], &[]);
        assert_eq!(pcs.iter().filter(|&&p| p == 0x100).count(), 1, "one pass");
        assert_eq!(regs.lc, 4);
    }

    #[test]
    fn an_ls_off_the_line_start_allows_one_redirect_to_the_line_start() {
        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x104, le: 0x140, lc: 4 };
        let pcs = run(&mut stage, &mut regs, 0x100, &vec![2; 8 * 5 * 3], &[]);
        assert_eq!(
            pcs.iter().filter(|&&p| p == 0x100).count(),
            2,
            "the bundle before LS in its line re-runs once"
        );
        assert_eq!(regs.lc, 3, "and only one redirect is taken");
    }

    #[test]
    fn le_written_inside_the_fetch_window_is_not_in_time() {
        // `movxm le` (6 bytes) at 0x100 with the LE line at 0x100 + 6 + 64 = line 0x140 + ...:
        // LE 0x140 is line(end = 0x106) + 4 lines, already fetched; LE 0x150 is the first in time.
        for (le, expect_passes) in [(0x140u32, 1usize), (0x150, 3)] {
            let mut stage = FetchStage::default();
            let mut regs = LoopRegs { ls: 0x110, le: 0, lc: 3 };
            let mut sizes = vec![6];
            sizes.extend(vec![2; 5]); // up to the LS line at 0x110
            sizes.extend(vec![2; 8 * 12 * 3]);
            let write_le: fn(&mut LoopRegs) = if le == 0x140 {
                |r| r.le = 0x140
            } else {
                |r| r.le = 0x150
            };
            let pcs = run(&mut stage, &mut regs, 0x100, &sizes, &[(0, write_le)]);
            assert_eq!(pcs.iter().filter(|&&p| p == 0x110).count(), expect_passes, "LE=0x{le:x}");
        }
    }

    #[test]
    fn an_lc_written_inside_the_window_is_read_stale() {
        // LC=0 at the fetch of the LE line (written late): one redirect without a real count, then
        // the late write of 4 gives three more -- five passes, LC left at 1.
        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x110, le: 0x140, lc: 0 };
        // arm by an LE change before the LC write bundle
        stage.fetch_ahead(0x100, 0x102, &mut regs); // sees LE=0x140 armed, window to line 0x140: LE line fetched, stale LC=0
        assert_eq!(regs.lc, 0u32.wrapping_sub(1), "the stale LC=0 looped");
        regs.lc = 4; // the late `add.nc lc` lands (overwrites the decremented value)
        let mut pc = 0x102u32;
        let mut passes = 1; // the first pass through the loop start counts
        for _ in 0..8 * 5 * 6 {
            if pc == 0x110 {
                passes += 1;
            }
            stage.fetch_ahead(pc, pc + 2, &mut regs);
            pc += 2;
            if let Some(r) = stage.retire(pc) {
                pc = r.target;
            }
        }
        assert_eq!((passes - 1 /* entry at 0x110 counted once by the loop above */, regs.lc), (5, 1));
    }

    #[test]
    fn a_branch_flushes_the_stream_and_gives_back_the_decrements() {
        let mut stage = FetchStage::default();
        let mut regs = LoopRegs { ls: 0x100, le: 0x120, lc: 5 };
        stage.fetch_ahead(0x100, 0x102, &mut regs);
        assert!(regs.lc < 5, "the LE line was inside the window");
        // issue jumps elsewhere
        stage.fetch_ahead(0x400, 0x402, &mut regs);
        assert_eq!(regs.lc, 5);
        let _ = L;
    }
}
