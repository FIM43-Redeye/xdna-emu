//! Runtime-sequence operation costs on the management controller's clock.
//!
//! On silicon the controller executes a runtime sequence one operation at a
//! time, and each operation holds it for a cost set by the operation's class.
//! Nothing else paces the sequence: there is no gate on task-queue pushes (the
//! gap between two pushes is the sum of the costs of the operations between
//! them, linear in their number, and a second task pushed on a busy channel
//! does not wait for the first).
//!
//! Costs are stated in controller (firmware-clock) cycles and converted to
//! array cycles by the configured clock pair (`crate::timebase`), so they
//! scale with the firmware/array ratio instead of being array-cycle literals.
//! Each constant names the array-cycle value it was measured as and how it
//! was measured, and is the whole number of controller cycles closest to
//! that value at the default clocks (600 / 1028 MHz), so the quantisation is
//! under one array cycle per operation.
//!
//! **Task-complete tokens.** A token that arrives while the controller is
//! executing the sequence suspends it for one service episode. Tokens that
//! arrive within [`TOKEN_EPISODE_WINDOW`] of the episode's start are absorbed by the
//! same episode; a later one starts another episode, right after the current one
//! if it arrived during it. An episode costs [`TOKEN_EPISODE`] isolated and
//! [`TOKEN_EPISODE_AFTER_EPISODE`] when it follows another directly: the cost is
//! per episode, not per token. Applied in `NpuExecutor` (`ServiceEpisode`).
//!
//! A sync's cost is not an operation-class cost: it depends on how late its
//! token arrives after the sync is issued (`super::sync_wait`, applied in
//! `NpuExecutor::sync_charge`). A token that arrived before the sync was
//! issued has had its episode already; the sync's fast path does not charge it
//! again.

use super::parser::NpuInstruction;
use crate::timebase::ClockPair;

/// Register write to tile memory or tile registers: 100.47 array cycles on
/// silicon (dedicated probe: runs of 4, 16 and 64 single-word writes, slope
/// +-0.04). 59 controller cycles = 101.09 array cycles. OPEN: the first few
/// writes of a run cost more (about 72 array cycles in all), not modelled.
pub const WRITE_32_TILE: u64 = 59;

/// Register write to a shim register other than a task queue: 102.7 array
/// cycles on silicon (dedicated probe: runs of writes to one shim register,
/// slope over 0 / 8 / 32 writes in 36 runs). 60 controller cycles = 102.8.
pub const WRITE_32_SHIM: u64 = 60;

/// Task-queue push (a register write to a channel's task queue): 104.85
/// array cycles (free push gaps of the bridge corpus and the earlier
/// runtime-sequence probes). Not yet compared with the other write classes
/// by one probe. 61 controller cycles = 104.51 array cycles.
pub const PUSH: u64 = 61;

/// Masked write to a shim register, the controller-id write a shim push
/// carries: 231 array cycles on silicon (push gaps with and without it, two
/// dedicated probes, +-1). 135 controller cycles = 231.3 array cycles.
pub const MASK_WRITE_SHIM: u64 = 135;

/// Masked write to a memtile register: 216.9 array cycles each on silicon
/// (dedicated probe, one and four in a row, 11 runs +-1). 127 controller
/// cycles = 217.6 array cycles.
pub const MASK_WRITE_MEMTILE: u64 = 127;

/// Masked write to a compute-tile register: 225.9 array cycles each (same
/// probe). 132 controller cycles = 226.2 array cycles.
pub const MASK_WRITE_COMPUTE: u64 = 132;

/// Extra cost of the first tile masked write of a run (the operation before
/// it is not a tile masked write): 34.4 (memtile) and 30.1 (compute tile)
/// array cycles on silicon (first of a run 251.3 / 256.0). 19 controller
/// cycles = 32.6 array cycles.
pub const MASK_WRITE_FIRST_OF_RUN: u64 = 19;

/// Block write to any tile (BD registers or data memory alike, memtile and
/// compute tile alike), fixed part, in hundredths of a controller cycle: the
/// dedicated probes on a warm command are exactly linear from 8 words on,
/// 179.4 + 13.98 per word array cycles (290 at 8 words, 1074 at 64, residual
/// under 2.1). 104.71 controller cycles = 179.4 array cycles. OPEN: below 8
/// words silicon is cheaper than the line (1 word 166, 6 words 232, against
/// 193 and 263), with steps the line does not have; and a block write early
/// in a cold command was fitted on the bridge corpus at about 450 for 6 to 8
/// words, never probed. Neither is modelled.
pub const BLOCK_WRITE_FIXED_CENTI: u64 = 10_471;

/// Block write, cost per word in hundredths of a controller cycle: 13.98
/// array cycles per word on silicon (slope over 8..64 words, +-0.02).
/// 8.159 controller cycles = 13.98 array cycles. At the lower power level
/// silicon's per-word cost grows by 1.146 rather than by the clock-ratio
/// change of 1.167 (part of it is not on the controller clock); the model
/// scales it with the controller clock.
pub const BLOCK_WRITE_PER_WORD_CENTI: u64 = 816;

/// A shim BD block write together with its address patch: 584.15 array cycles
/// on silicon (block write + patch + push lead of 686..689 at 12 stream
/// positions, less the push). 341 controller cycles = 584.25 array cycles.
/// The split between the block write and the patch is not measured: the
/// patch takes what the 8-word block write leaves ([`DDR_PATCH`]).
pub const SHIM_BD_WITH_PATCH: u64 = 341;

/// Words of a shim BD (the block write a DDR patch follows).
const SHIM_BD_WORDS: u64 = 8;

/// Address patch of a shim BD: the part of [`SHIM_BD_WITH_PATCH`] the shim
/// BD's block write does not take, so the measured pair total holds.
pub const DDR_PATCH: u64 = SHIM_BD_WITH_PATCH - block_write_cost(SHIM_BD_WORDS);

/// PROVISIONAL. Mask-poll satisfied at its read: about 215 array cycles,
/// fitted on the bridge corpus only (a dedicated probe is queued).
/// 125 controller cycles = 214.17 array cycles.
pub const MASK_POLL: u64 = 125;

/// A token service episode with no episode right before it: about 2300 array
/// cycles on silicon (a single token in a push gap, FINISHED to the next
/// push, +-30). 1342 controller cycles = 2299.3 array cycles.
pub const TOKEN_EPISODE: u64 = 1342;

/// An episode that starts right after another, because its token arrived during
/// it beyond the window: about 1600 array cycles on silicon (pairs of tokens
/// 1076..2105 apart against pairs 102..987 apart, 18 and 15 runs). 934
/// controller cycles = 1600.2 array cycles. OPEN: an episode a little later than
/// that (3..4k after the previous one's start) read about 2.0k; how the cost
/// returns to the isolated value is not measured, and the model charges an
/// isolated episode to every episode that does not follow another directly.
pub const TOKEN_EPISODE_AFTER_EPISODE: u64 = 934;

/// Window of an episode: tokens arriving within this time of the episode's start
/// are absorbed by it at no measurable cost (silicon: one episode for two
/// tokens 987 apart, two episodes for 1076 apart). 602 controller cycles =
/// 1031.4 array cycles, the middle of that edge.
pub const TOKEN_EPISODE_WINDOW: u64 = 602;

/// Block write of `words` words, in whole controller cycles (nearest).
pub const fn block_write_cost(words: u64) -> u64 {
    (BLOCK_WRITE_FIXED_CENTI + BLOCK_WRITE_PER_WORD_CENTI * words + 50) / 100
}

/// What an operation writes to, as far as its cost depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpTarget {
    /// A shim tile register (`push` when it is a channel's task queue).
    Shim {
        push: bool,
    },
    MemTile,
    Compute,
}

/// Operation-class costs of the runtime-sequence controller, in controller
/// cycles, and the clock pair that converts them to array cycles.
#[derive(Debug, Clone)]
pub struct CycleCostModel {
    pub write_32_tile: u64,
    pub write_32_shim: u64,
    pub push: u64,
    pub mask_write_shim: u64,
    pub mask_write_memtile: u64,
    pub mask_write_compute: u64,
    pub mask_write_first_of_run: u64,
    pub ddr_patch: u64,
    pub mask_poll: u64,
    pub token_episode: u64,
    pub token_episode_after_episode: u64,
    pub token_episode_window: u64,
    pub clocks: ClockPair,
}

impl CycleCostModel {
    /// NPU1 (Phoenix) costs at `clocks`.
    pub fn npu1(clocks: ClockPair) -> Self {
        Self {
            write_32_tile: WRITE_32_TILE,
            write_32_shim: WRITE_32_SHIM,
            push: PUSH,
            mask_write_shim: MASK_WRITE_SHIM,
            mask_write_memtile: MASK_WRITE_MEMTILE,
            mask_write_compute: MASK_WRITE_COMPUTE,
            mask_write_first_of_run: MASK_WRITE_FIRST_OF_RUN,
            ddr_patch: DDR_PATCH,
            mask_poll: MASK_POLL,
            token_episode: TOKEN_EPISODE,
            token_episode_after_episode: TOKEN_EPISODE_AFTER_EPISODE,
            token_episode_window: TOKEN_EPISODE_WINDOW,
            clocks,
        }
    }

    /// Controller cycles an operation holds the controller for. `target` is
    /// what it writes to; `after_tile_mask_write` whether the operation
    /// before it was a masked write to a tile register.
    ///
    /// A sync returns 0 here: its cost is decided when it resolves
    /// (`NpuExecutor::sync_charge`). A mask-poll's cost is the satisfied
    /// read's, charged when the read sees its value.
    pub fn cost_of(&self, instr: &NpuInstruction, target: OpTarget, after_tile_mask_write: bool) -> u64 {
        match instr {
            NpuInstruction::Write32 { .. } => match target {
                OpTarget::Shim { push: true } => self.push,
                OpTarget::Shim { push: false } => self.write_32_shim,
                OpTarget::MemTile | OpTarget::Compute => self.write_32_tile,
            },
            NpuInstruction::MaskWrite { .. } => {
                let first = if after_tile_mask_write {
                    0
                } else {
                    self.mask_write_first_of_run
                };
                match target {
                    OpTarget::Shim { .. } => self.mask_write_shim,
                    OpTarget::MemTile => self.mask_write_memtile + first,
                    OpTarget::Compute => self.mask_write_compute + first,
                }
            }
            NpuInstruction::BlockWrite { values, .. } => block_write_cost(values.len() as u64),
            NpuInstruction::DdrPatch { .. } => self.ddr_patch,
            NpuInstruction::MaskPoll { .. } => self.mask_poll,
            NpuInstruction::Sync { .. } | NpuInstruction::Unknown { .. } => 0,
        }
    }
}

impl Default for CycleCostModel {
    /// NPU1 costs at the configured clocks (`XDNA_EMU_DPM`); an invalid
    /// setting is reported and the default pair used.
    fn default() -> Self {
        let clocks = ClockPair::configured().unwrap_or_else(|e| {
            log::warn!("{e}; using the default clocks");
            ClockPair::DEFAULT
        });
        Self::npu1(clocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timebase::FirmwareToArray;

    /// Array cycles of `n` back-to-back operations of controller cost `c`, per op.
    fn mean_array_cost(c: u64, n: u64) -> f64 {
        let mut conv = FirmwareToArray::new(ClockPair::DEFAULT);
        (0..n).map(|_| conv.array_cycles(c)).sum::<u64>() as f64 / n as f64
    }

    #[test]
    fn the_class_costs_reproduce_the_silicon_values_at_the_default_clocks() {
        for (name, c, silicon, tol) in [
            ("write32, tile", WRITE_32_TILE, 100.47, 0.7),
            ("write32, shim register", WRITE_32_SHIM, 102.7, 0.5),
            ("push", PUSH, 104.85, 0.5),
            ("maskwrite32, shim", MASK_WRITE_SHIM, 231.0, 0.5),
            ("maskwrite32, memtile", MASK_WRITE_MEMTILE, 216.9, 0.8),
            ("maskwrite32, compute tile", MASK_WRITE_COMPUTE, 225.9, 0.5),
            ("maskwrite32, first of a run (mean of the two tiles)", MASK_WRITE_FIRST_OF_RUN, 32.25, 0.5),
            ("shim BD + patch", SHIM_BD_WITH_PATCH, 584.15, 0.5),
            ("block write, 8 words", block_write_cost(8), 291.2, 0.5),
            ("block write, 64 words", block_write_cost(64), 1074.1, 0.5),
            ("mask-poll (provisional)", MASK_POLL, 215.0, 1.0),
            ("token episode", TOKEN_EPISODE, 2300.0, 1.0),
            ("token episode after an episode", TOKEN_EPISODE_AFTER_EPISODE, 1600.0, 1.0),
            ("token episode window", TOKEN_EPISODE_WINDOW, 1031.5, 1.0),
        ] {
            let a = mean_array_cost(c, 600);
            assert!((a - silicon).abs() < tol, "{name}: {a} array cycles against {silicon}");
        }
    }

    #[test]
    fn a_block_write_grows_by_the_measured_cost_per_word() {
        let per_word =
            (mean_array_cost(block_write_cost(64), 600) - mean_array_cost(block_write_cost(8), 600)) / 56.0;
        assert!((per_word - 13.98).abs() < 0.03, "per word {per_word}");
    }

    #[test]
    fn the_shim_bd_pair_holds_its_measured_total_whatever_the_split() {
        assert_eq!(block_write_cost(SHIM_BD_WORDS) + DDR_PATCH, SHIM_BD_WITH_PATCH);
    }

    #[test]
    fn costs_scale_with_the_clock_ratio() {
        let mut dpm0 = FirmwareToArray::new(ClockPair::DRIVER_DPM0);
        assert_eq!(dpm0.array_cycles(PUSH), 2 * PUSH, "DPM 0: array at twice the firmware clock");
    }

    #[test]
    fn each_instruction_kind_maps_to_its_class() {
        let m = CycleCostModel::npu1(ClockPair::DEFAULT);
        let w = NpuInstruction::Write32 { reg_off: 0, value: 0 };
        let mw = NpuInstruction::MaskWrite { reg_off: 0, value: 0, mask: 0 };
        let bw = NpuInstruction::BlockWrite { reg_off: 0x1D000, values: vec![0; 8] };
        let p = NpuInstruction::DdrPatch { reg_addr: 0x1D004, arg_idx: 0, arg_plus: 0 };
        let mp = NpuInstruction::MaskPoll { reg_off: 0, value: 0, mask: 0 };
        let sy =
            NpuInstruction::Sync { channel: 0, column: 0, direction: 0, column_num: 1, row: 0, row_num: 1 };
        let shim = OpTarget::Shim { push: false };
        assert_eq!(m.cost_of(&w, OpTarget::Shim { push: true }, false), PUSH);
        assert_eq!(m.cost_of(&w, shim, false), WRITE_32_SHIM);
        assert_eq!(m.cost_of(&w, OpTarget::Compute, false), WRITE_32_TILE);
        assert_eq!(m.cost_of(&w, OpTarget::MemTile, false), WRITE_32_TILE);
        assert_eq!(m.cost_of(&mw, shim, false), MASK_WRITE_SHIM);
        assert_eq!(m.cost_of(&mw, shim, true), MASK_WRITE_SHIM, "no run term on the shim");
        assert_eq!(m.cost_of(&mw, OpTarget::MemTile, false), MASK_WRITE_MEMTILE + MASK_WRITE_FIRST_OF_RUN);
        assert_eq!(m.cost_of(&mw, OpTarget::MemTile, true), MASK_WRITE_MEMTILE);
        assert_eq!(m.cost_of(&mw, OpTarget::Compute, true), MASK_WRITE_COMPUTE);
        assert_eq!(m.cost_of(&bw, shim, false), block_write_cost(8));
        assert_eq!(m.cost_of(&p, shim, false), DDR_PATCH);
        assert_eq!(m.cost_of(&mp, shim, false), MASK_POLL);
        assert_eq!(m.cost_of(&sy, shim, false), 0, "a sync's cost is decided when it resolves");
    }
}
