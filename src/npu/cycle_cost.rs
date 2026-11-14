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
//! Each constant names the array-cycle value it was measured as at the
//! default clocks (600 / 1028 MHz) and how it was measured.
//!
//! A sync's cost is not an operation-class cost: it is decided in one place,
//! `NpuExecutor::sync_charge`, from the sync's issue time and the arrival of
//! the token it awaits.

use super::parser::NpuInstruction;
use crate::timebase::ClockPair;

/// Register write, including every task-queue push: 104.85 array cycles on
/// silicon (free gaps of dedicated runtime-sequence probes; slope 102.5..106.0
/// over 0 / 4 / 16 writes). 61 controller cycles = 104.51 array cycles.
pub const WRITE_32: u64 = 61;

/// Masked write (read-modify-write): 178.5 array cycles on silicon
/// (dedicated probes). 104 controller cycles = 178.19 array cycles.
pub const MASK_WRITE: u64 = 104;

/// A shim BD block write together with its address patch: 584.15 array cycles
/// on silicon (block write + patch + push lead of 689, less the push's write).
/// 341 controller cycles = 584.25 array cycles. The split between the block
/// write and the patch is not measured: see [`BLOCK_WRITE`].
pub const SHIM_BD_WITH_PATCH: u64 = 341;

/// PROVISIONAL. Block write (a BD's 8 or 6 words, any tile): about 450 array
/// cycles, fitted on the bridge corpus only (memtile BD 455, core BD 445; one
/// core data-memory sample at 411), not measured by a dedicated probe (one is
/// queued). 263 controller cycles = 450.6 array cycles. A shim BD's block
/// write is taken to cost the same, which leaves the patch the rest of
/// [`SHIM_BD_WITH_PATCH`].
pub const BLOCK_WRITE: u64 = 263;

/// Address patch of a shim BD: the part of [`SHIM_BD_WITH_PATCH`] the block
/// write does not take, so the measured pair total holds whatever the split.
pub const DDR_PATCH: u64 = SHIM_BD_WITH_PATCH - BLOCK_WRITE;

/// PROVISIONAL. Mask-poll satisfied at its read: about 215 array cycles,
/// fitted on the bridge corpus only (a dedicated probe is queued).
/// 125 controller cycles = 214.17 array cycles.
pub const MASK_POLL: u64 = 125;

/// Operation-class costs of the runtime-sequence controller, in controller
/// cycles, and the clock pair that converts them to array cycles.
#[derive(Debug, Clone)]
pub struct CycleCostModel {
    pub write_32: u64,
    pub mask_write: u64,
    pub block_write: u64,
    pub ddr_patch: u64,
    pub mask_poll: u64,
    pub clocks: ClockPair,
}

impl CycleCostModel {
    /// NPU1 (Phoenix) costs at `clocks`.
    pub fn npu1(clocks: ClockPair) -> Self {
        Self {
            write_32: WRITE_32,
            mask_write: MASK_WRITE,
            block_write: BLOCK_WRITE,
            ddr_patch: DDR_PATCH,
            mask_poll: MASK_POLL,
            clocks,
        }
    }

    /// Controller cycles an operation holds the controller for.
    ///
    /// A sync returns 0 here: its cost is decided when it resolves
    /// (`NpuExecutor::sync_charge`). A mask-poll's cost is the satisfied
    /// read's, charged when the read sees its value.
    pub fn cost_of(&self, instr: &NpuInstruction) -> u64 {
        match instr {
            NpuInstruction::Write32 { .. } => self.write_32,
            NpuInstruction::MaskWrite { .. } => self.mask_write,
            NpuInstruction::BlockWrite { .. } => self.block_write,
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
            ("write32", WRITE_32, 104.85, 0.5),
            ("maskwrite32", MASK_WRITE, 178.5, 0.5),
            ("shim BD + patch", SHIM_BD_WITH_PATCH, 584.15, 0.5),
            ("block write (provisional)", BLOCK_WRITE, 450.0, 1.0),
            ("mask-poll (provisional)", MASK_POLL, 215.0, 1.0),
        ] {
            let a = mean_array_cost(c, 600);
            assert!((a - silicon).abs() < tol, "{name}: {a} array cycles against {silicon}");
        }
    }

    #[test]
    fn the_shim_bd_pair_holds_its_measured_total_whatever_the_split() {
        assert_eq!(BLOCK_WRITE + DDR_PATCH, SHIM_BD_WITH_PATCH);
    }

    #[test]
    fn costs_scale_with_the_clock_ratio() {
        let mut dpm0 = FirmwareToArray::new(ClockPair::DRIVER_DPM0);
        assert_eq!(dpm0.array_cycles(WRITE_32), 2 * WRITE_32, "DPM 0: array at twice the firmware clock");
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
        assert_eq!(m.cost_of(&w), WRITE_32);
        assert_eq!(m.cost_of(&mw), MASK_WRITE);
        assert_eq!(m.cost_of(&bw), BLOCK_WRITE);
        assert_eq!(m.cost_of(&p), DDR_PATCH);
        assert_eq!(m.cost_of(&mp), MASK_POLL);
        assert_eq!(m.cost_of(&sy), 0, "a sync's cost is decided when it resolves");
    }
}
