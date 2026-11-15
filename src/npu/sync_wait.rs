//! What a runtime-sequence sync costs: the controller's wait for a
//! task-complete token.
//!
//! On silicon the time from a sync's issue to the controller's next operation
//! depends on how late the awaited token arrives after the sync is issued
//! (`x` = arrival - issue, array cycles), and not smoothly: it falls into
//! discrete levels over ranges of `x`. The structure the measurements show:
//!
//! 1. **Fast path.** The token is already there when the sync is issued: a
//!    fixed cost ([`FAST_PATH`]).
//! 2. **Short wait.** The token arrives soon after the issue: the controller
//!    returns a fixed cost after the arrival ([`AWAKE_RETURN`]), so the
//!    resume follows the token.
//! 3. **Gated wait.** Once the controller has waited longer than an engage
//!    threshold ([`ENGAGE_AFTER`]) its wait gate engages. From then on it
//!    does not follow the token: it resumes only at discrete points, which
//!    gives a plateau (a resume time fixed relative to the issue) over each
//!    range of `x` ([`GATED_STAGES`]). Each stage's resume time is a measured
//!    level; the measurements do not show what sets it.
//! 4. **Beyond the last stage edge** the resume follows the token again: the
//!    wake-up falls on the next tick of a grid anchored at the issue
//!    ([`TICK`]) and is followed by a fixed exit cost ([`GATED_EXIT`]).
//!
//! Every quantity is stated in controller cycles: the levels, the edges and
//! the tick scale with the controller/array clock ratio on silicon (at the
//! lower power level the plateau grows by the ratio change, x1.167, and the
//! tick is 256 array cycles = 128 controller cycles at a ratio of 2), while
//! `x` is set by the DMA and counts array cycles. Each constant names the
//! array-cycle value it reproduces at the default clocks (600 / 1028 MHz).
//!
//! The levels are those of the FIRST wait of a command that follows no other
//! command (every bridge test): the bridge corpus fit of three silicon sample
//! sets per test, cross-checked by dedicated runtime-sequence probes. Later
//! waits in long runs, and commands that follow another command, sit at other
//! levels on silicon; this model applies the first-wait structure to every
//! wait (an open row in `docs/fidelity-gaps/host-firmware-dispatch.md`).
//!
//! **Tokens and their service.** A token that arrives while the controller is
//! executing operations suspends the sequence for a service episode
//! (`cycle_cost::TOKEN_EPISODE`, about 2300 array cycles when isolated). A sync
//! whose token arrived before its issue has therefore had its episode already;
//! the bridge-corpus level for such a sync (about 2950 array cycles in total
//! from the operation before the sync, the episode included) is split as the episode
//! plus [`FAST_PATH`], so the token is not charged twice. A token that arrives
//! during the wait is covered by the wait's own cost.

use crate::timebase::ClockPair;

/// The token was already there at the sync's issue: about 650 array cycles
/// from the issue to the next operation. The silicon level is about 2950 in
/// total for a sync on a task that finished before it (bridge corpus fit,
/// about 100 syncs: 3000 with the earlier operation costs, 52.6 less now that
/// the shim masked write after the sync costs 231 rather than 178.5), of which
/// the token's service episode (2300) runs before the sync issues. 378
/// controller cycles = 647.6 array. OPEN: a dedicated probe
/// reads about 1200 for a sync whose token's episode ran 3..9k earlier (warm and
/// cold command alike), where episode + fast path would then give 3.5k, not the
/// corpus's 3.0k; the model keeps the corpus split.
pub const FAST_PATH: u64 = 378;

/// Short wait: the controller resumes about 3000 array cycles after the token
/// arrives (bridge corpus fit, finish-anchored low end: 3050 with the earlier
/// operation costs, 52.6 less with the measured shim masked-write cost after
/// the sync; dedicated probes on a warm command read 3740..3820). 1749
/// controller cycles = 2996.6 array.
pub const AWAKE_RETURN: u64 = 1749;

/// The wait gate engages once the controller has waited this long: a token
/// later than about 600 array cycles after the issue no longer gets the
/// short-wait return (bridge corpus fit; dedicated probe edge between 593 and
/// 630). 350 controller cycles = 599.7 array.
pub const ENGAGE_AFTER: u64 = 350;

/// One plateau of the gated wait: a token arriving before `until` (and after
/// the previous stage's edge, or [`ENGAGE_AFTER`]) resumes the controller
/// `level` after the sync's issue, whatever its exact arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedStage {
    /// Upper edge of the stage, controller cycles after the issue.
    pub until: u64,
    /// Resume time, controller cycles after the issue (a measured level).
    pub level: u64,
}

/// The gated wait's plateaus, in order of `x`.
pub const GATED_STAGES: [GatedStage; 3] = [
    // Level 5850 for x up to 950: dedicated probe level 5.94k (issue to resume,
    // less one marker write); bridge corpus 5.5..6.1k; edge 0.92..0.98k on the
    // corpus. 555 / 3414 controller cycles = 950.9 / 5849.3 array.
    GatedStage { until: 555, level: 3414 },
    // Cold plateau 9540 for x up to 5000: dedicated probe on a cold command
    // 9.60..9.66k less one marker write; bridge corpus single-sync tails
    // 9.70..9.79k = the level + two writes, in 60+ tests.
    // 2918 / 5568 controller cycles = 4999.4 / 9539.8 array.
    GatedStage { until: 2918, level: 5568 },
    // Level 14100 for x up to 7200: bridge corpus (5 tails at 14.0..14.6k);
    // the edge is the cold command's gated edge (dedicated probe 7.16..7.26k).
    // 4202 / 8230 controller cycles = 7199.4 / 14100.7 array.
    GatedStage { until: 4202, level: 8230 },
];

/// The gated wait's tick: wake-ups beyond the last stage edge fall on a grid
/// of this period anchored at the sync's issue. Silicon: 219.80 array cycles
/// at the default clocks and 256.0 at the lower power level, i.e. 128
/// controller cycles (dedicated probes, 131 periods). 128 controller cycles =
/// 219.3 array at the nominal clock pair.
pub const TICK: u64 = 128;

/// Exit cost after a gated wake-up: with the wake on the first tick after the
/// token, the resume lands [`GATED_EXIT`] .. [`GATED_EXIT`] + one tick after
/// the token, about 8750 on average (dedicated probes: 8.61..8.83k beyond the
/// last edge, out to x = 30k; bridge corpus 8.6..8.9k). 5043 controller
/// cycles = 8640.3 array (+ half a tick = 8750).
pub const GATED_EXIT: u64 = 5043;

/// When the awaited token arrived, relative to the sync's issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenWait {
    /// The token was there when the sync was issued.
    AlreadyArrived,
    /// The token arrived this many array cycles after the issue (`x`).
    ArrivedAfter(u64),
}

/// Which part of the wait structure a sync resolved in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitRegion {
    FastPath,
    ShortWait,
    /// Index into [`GATED_STAGES`].
    Stage(usize),
    GatedWake,
}

/// The controller's sync wait, in controller cycles, and the clock pair that
/// converts it to array cycles.
#[derive(Debug, Clone)]
pub struct SyncWaitModel {
    pub fast_path: u64,
    pub awake_return: u64,
    pub engage_after: u64,
    pub stages: [GatedStage; 3],
    pub tick: u64,
    pub gated_exit: u64,
    pub clocks: ClockPair,
}

impl SyncWaitModel {
    /// NPU1 (Phoenix), first wait of a cold command, at `clocks`.
    pub fn npu1(clocks: ClockPair) -> Self {
        Self {
            fast_path: FAST_PATH,
            awake_return: AWAKE_RETURN,
            engage_after: ENGAGE_AFTER,
            stages: GATED_STAGES,
            tick: TICK,
            gated_exit: GATED_EXIT,
            clocks,
        }
    }

    /// Whole array cycles of `controller` cycles (truncated).
    fn to_array(&self, controller: u64) -> u64 {
        (u128::from(controller) * u128::from(self.clocks.array_mhz) / u128::from(self.clocks.firmware_mhz))
            as u64
    }

    /// Whether `x` array cycles lie before `edge` controller cycles (exact).
    fn before(&self, x: u64, edge: u64) -> bool {
        u128::from(x) * u128::from(self.clocks.firmware_mhz)
            < u128::from(edge) * u128::from(self.clocks.array_mhz)
    }

    /// The part of the wait structure a token arriving at `wait` falls in.
    pub fn region(&self, wait: TokenWait) -> WaitRegion {
        let x = match wait {
            TokenWait::AlreadyArrived => return WaitRegion::FastPath,
            TokenWait::ArrivedAfter(x) => x,
        };
        if self.before(x, self.engage_after) {
            return WaitRegion::ShortWait;
        }
        match self.stages.iter().position(|s| self.before(x, s.until)) {
            Some(i) => WaitRegion::Stage(i),
            None => WaitRegion::GatedWake,
        }
    }

    /// Array cycles from the sync's issue to the controller's next operation.
    pub fn resume_after_issue(&self, wait: TokenWait) -> u64 {
        match (self.region(wait), wait) {
            (WaitRegion::FastPath, _) => self.to_array(self.fast_path),
            (WaitRegion::ShortWait, TokenWait::ArrivedAfter(x)) => x + self.to_array(self.awake_return),
            (WaitRegion::Stage(i), _) => self.to_array(self.stages[i].level),
            (WaitRegion::GatedWake, TokenWait::ArrivedAfter(x)) => {
                // First tick at or after the token, in controller cycles from the
                // issue: ceil(x * firmware / (tick * array)).
                let num = u128::from(x) * u128::from(self.clocks.firmware_mhz);
                let den = u128::from(self.tick) * u128::from(self.clocks.array_mhz);
                let ticks = num.div_ceil(den) as u64;
                self.to_array(ticks * self.tick + self.gated_exit)
            }
            (_, TokenWait::AlreadyArrived) => unreachable!("an arrived token is the fast path"),
        }
    }
}

impl Default for SyncWaitModel {
    fn default() -> Self {
        Self::npu1(super::CycleCostModel::default().clocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npu::cycle_cost::TOKEN_EPISODE;
    use crate::timebase::FirmwareToArray;

    fn model() -> SyncWaitModel {
        SyncWaitModel::npu1(ClockPair::DEFAULT)
    }

    fn after(x: u64) -> u64 {
        model().resume_after_issue(TokenWait::ArrivedAfter(x))
    }

    fn near(got: u64, want: f64, tol: f64, what: &str) {
        assert!((got as f64 - want).abs() <= tol, "{what}: {got} against silicon {want}");
    }

    /// Silicon levels at the default clocks (array cycles), first wait of a
    /// cold command: fast 2.95k with the token's episode | x + 3.0k to 0.6k |
    /// 5.85k to 0.95k | 9.54k to 5k | 14.1k to 7.2k | x + 8.75k beyond.
    #[test]
    fn the_fast_path_costs_the_measured_level_less_the_token_episode() {
        let fast = model().resume_after_issue(TokenWait::AlreadyArrived);
        near(fast, 648.0, 2.0, "fast path");
        let service = FirmwareToArray::new(ClockPair::DEFAULT).array_cycles(TOKEN_EPISODE);
        near(fast + service, 2947.0, 2.0, "episode + fast path (the corpus level)");
    }

    /// The double-count rule: the token is charged once, whichever side of the
    /// issue it lands on, so a token just before the issue and one just after
    /// cost nearly the same in total.
    #[test]
    fn a_token_just_before_the_issue_and_one_just_after_cost_alike() {
        let service = FirmwareToArray::new(ClockPair::DEFAULT).array_cycles(TOKEN_EPISODE);
        let before = service + model().resume_after_issue(TokenWait::AlreadyArrived);
        let just_after = after(1);
        assert!(just_after.abs_diff(before) <= 60, "before {before}, just after {just_after}");
    }

    #[test]
    fn a_short_wait_returns_a_fixed_cost_after_the_token() {
        for x in [1, 100, 300, 599] {
            assert_eq!(model().region(TokenWait::ArrivedAfter(x)), WaitRegion::ShortWait, "x={x}");
            near(after(x) - x, 2997.0, 2.0, &format!("short wait x={x}"));
        }
    }

    #[test]
    fn the_gate_engages_after_the_threshold() {
        assert_eq!(model().region(TokenWait::ArrivedAfter(599)), WaitRegion::ShortWait);
        assert_eq!(model().region(TokenWait::ArrivedAfter(600)), WaitRegion::Stage(0));
        near(after(599), 599.0 + 2997.0, 2.0, "x=599");
        near(after(600), 5850.0, 2.0, "x=600");
    }

    #[test]
    fn the_first_stage_holds_its_level_up_to_its_edge() {
        for x in [600, 800, 950] {
            near(after(x), 5850.0, 2.0, &format!("x={x}"));
        }
        assert_eq!(model().region(TokenWait::ArrivedAfter(950)), WaitRegion::Stage(0));
        assert_eq!(model().region(TokenWait::ArrivedAfter(951)), WaitRegion::Stage(1));
        near(after(951), 9540.0, 2.0, "x=951");
    }

    #[test]
    fn the_cold_plateau_holds_up_to_its_edge() {
        for x in [951, 3000, 4999] {
            near(after(x), 9540.0, 2.0, &format!("x={x}"));
        }
        assert_eq!(model().region(TokenWait::ArrivedAfter(4999)), WaitRegion::Stage(1));
        assert_eq!(model().region(TokenWait::ArrivedAfter(5000)), WaitRegion::Stage(2));
        near(after(5000), 14100.0, 2.0, "x=5000");
    }

    #[test]
    fn the_last_stage_holds_up_to_the_gated_edge() {
        for x in [5000, 6000, 7199] {
            near(after(x), 14100.0, 2.0, &format!("x={x}"));
        }
        assert_eq!(model().region(TokenWait::ArrivedAfter(7199)), WaitRegion::Stage(2));
        assert_eq!(model().region(TokenWait::ArrivedAfter(7200)), WaitRegion::GatedWake);
        let r = after(7200) - 7200;
        assert!((8640..8640 + 220).contains(&r), "x=7200: resume {r} after the token");
    }

    /// Beyond the last edge: wake on the first tick after the token, then the
    /// exit cost; the resume follows the token by 8.75k on average.
    #[test]
    fn gated_wakes_fall_on_the_tick_and_follow_the_token() {
        let m = model();
        let tick_array = 128.0 * 1028.0 / 600.0;
        let mut sum = 0.0;
        let xs: Vec<u64> = (7200..30_000).step_by(7).collect();
        for &x in &xs {
            let r = m.resume_after_issue(TokenWait::ArrivedAfter(x));
            // The wake (resume less the exit) sits on the tick grid from the issue.
            let wake_ctrl = (r as f64) * 600.0 / 1028.0 - GATED_EXIT as f64;
            let k = (wake_ctrl / TICK as f64).round();
            assert!(
                (wake_ctrl - k * TICK as f64).abs() < 1.0,
                "x={x}: wake at {wake_ctrl} controller cycles"
            );
            assert!(k * tick_array >= x as f64 - 1.0, "x={x}: wake before the token");
            let behind = (r - x) as f64;
            assert!((8640.0 - 1.0..8640.0 + tick_array + 1.0).contains(&behind), "x={x}: {behind}");
            sum += behind;
        }
        near((sum / xs.len() as f64) as u64, 8750.0, 5.0, "mean resume after the token");
        // Tokens within one tick of each other resume together.
        let t0 = m.resume_after_issue(TokenWait::ArrivedAfter(10_000));
        let base = (10_000.0 / tick_array).ceil() * tick_array;
        let same = m.resume_after_issue(TokenWait::ArrivedAfter((base - 1.0) as u64));
        assert_eq!(t0, same, "one tick, one wake");
    }

    #[test]
    fn the_wait_scales_with_the_clock_ratio() {
        // DPM 0: array at twice the controller clock, so every controller-clocked
        // level is twice its controller cycles; x stays in array cycles.
        let m = SyncWaitModel::npu1(ClockPair::DRIVER_DPM0);
        assert_eq!(m.resume_after_issue(TokenWait::ArrivedAfter(3000)), 2 * GATED_STAGES[1].level);
        assert_eq!(m.region(TokenWait::ArrivedAfter(2 * ENGAGE_AFTER - 1)), WaitRegion::ShortWait);
        assert_eq!(m.region(TokenWait::ArrivedAfter(2 * ENGAGE_AFTER)), WaitRegion::Stage(0));
    }
}
