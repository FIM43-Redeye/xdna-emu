//! AIE2 clock-control subsystem.
//!
//! Owns all clock-gating state for the array (column / module / adaptive
//! tiers).  Boots with every tile gated, matching silicon behavior per
//! aie-rt's XAie_PmRequestTiles documentation.  Tests opt out via
//! `ungate_all()` which exercises the same register-write path the
//! real CDO uses.
//!
//! Spec: docs/superpowers/specs/2026-05-24-clock-control-design.md

use std::collections::HashMap;

/// Which module within a tile is being queried.  Mirrors the bit-field
/// breakdown of Module_Clock_Control in AM025.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    Core,
    Memory,
    Dma,
    StreamSwitch,
}

/// Per-tile module-gate state.  Stores the raw register value(s); the
/// per-module active queries decode bit fields on demand.
#[derive(Debug, Clone)]
struct TileGates {
    /// `Module_Clock_Control` (compute, memtile) or `Module_Clock_Control_0` (shim).
    raw_mcc_0: u32,
    /// `Module_Clock_Control_1` (shim only); None for other tile types.
    raw_mcc_1: Option<u32>,
}

/// Per-array clock-gating state.  Single source of truth for all
/// column / module / adaptive gates.
#[derive(Debug, Clone)]
pub struct ClockController {
    /// Per-column clock gate; index = col.  `false` = gated (inactive).
    columns: Vec<bool>,
    /// Number of rows in the array.  Stored so `ungate_all()` knows how
    /// far to iterate without needing to consult arch config.
    num_rows: u8,
    /// Per-tile module-gate state, keyed by (col, row).
    /// Tile entry is absent until first write to that tile's MCC register;
    /// is_module_active falls back to the AM025 reset values in that case
    /// (consistent with what a read of the register would return).
    tiles: HashMap<(u8, u8), TileGates>,
}

/// AM025 offset for Column_Clock_Control (shim tiles only).
const COLUMN_CLOCK_CONTROL_OFFSET: u32 = 0x000FFF20;

impl ClockController {
    /// Construct a controller for an array of `num_cols` columns and
    /// `num_rows` rows.  All columns boot gated (silicon-accurate).
    pub fn new(num_cols: u8, num_rows: u8) -> Self {
        Self { columns: vec![false; num_cols as usize], num_rows, tiles: HashMap::new() }
    }

    /// Returns true iff column `col` has its clock enabled.
    /// Returns false for out-of-range columns.
    pub fn is_column_active(&self, col: u8) -> bool {
        self.columns.get(col as usize).copied().unwrap_or(false)
    }

    /// Returns true iff the named module on tile (col, row) is currently
    /// clocked.  Column gate dominates: a gated column means every module
    /// reports inactive regardless of MCC.  An ungated column with no MCC
    /// writes yet uses the AM025 reset value.
    ///
    /// Task 4 stub: always returns false until bit-field decode is added in
    /// Task 5.
    pub fn is_module_active(&self, col: u8, row: u8, kind: ModuleKind) -> bool {
        // Column gate dominates.
        if !self.is_column_active(col) {
            return false;
        }
        // For now (filled in by Task 5), return false until bit-field
        // decode is added.
        let _ = (row, kind);
        false
    }

    /// Returns true iff any module on this tile is active.
    pub fn is_tile_active(&self, col: u8, row: u8) -> bool {
        use ModuleKind::*;
        self.is_module_active(col, row, Core)
            || self.is_module_active(col, row, Memory)
            || self.is_module_active(col, row, Dma)
            || self.is_module_active(col, row, StreamSwitch)
    }

    /// Handle a register write at the given tile / offset.  Silently
    /// ignores offsets that are not clock-control registers.
    pub fn write_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        match offset {
            COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
                // Bit 0 = column-clock-enable; per AM025.
                if let Some(slot) = self.columns.get_mut(col as usize) {
                    *slot = (value & 0x1) != 0;
                }
            }
            _ => {} // not a clock-control offset
        }
    }

    /// Read a clock-control register.  Returns the current value, or the
    /// AM025 reset value if the register has not been written yet.
    /// Returns None if the offset is not a known clock-control register.
    pub fn read_register(&self, col: u8, row: u8, offset: u32) -> Option<u32> {
        let _ = (col, row);
        match offset {
            COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
                // Reflect current state: bit 0 = column enabled.
                let enabled = self.is_column_active(col);
                Some(if enabled { 0x1 } else { 0x0 })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_clock_controller_boots_with_all_columns_gated() {
        let clock = ClockController::new(5, 6);
        for col in 0..5 {
            assert!(!clock.is_column_active(col), "col {} should be gated at boot", col);
        }
    }

    #[test]
    fn is_column_active_returns_false_for_out_of_range_col() {
        let clock = ClockController::new(5, 6);
        assert!(!clock.is_column_active(5));
        assert!(!clock.is_column_active(99));
    }

    #[test]
    fn write_column_clock_control_bit0_enables_column() {
        let mut clock = ClockController::new(5, 6);
        // Column 2's shim tile is at (col=2, row=0).
        // Column_Clock_Control offset is 0x000FFF20.
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        assert!(clock.is_column_active(2));
        assert!(!clock.is_column_active(0), "other cols unaffected");
    }

    #[test]
    fn write_column_clock_control_bit0_clear_disables_column() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        assert!(clock.is_column_active(2));
        clock.write_register(2, 0, 0x000FFF20, 0x0);
        assert!(!clock.is_column_active(2));
    }

    #[test]
    fn read_column_clock_control_returns_reset_before_any_write() {
        let clock = ClockController::new(5, 6);
        // AM025 reset is 0x00000000 for Column_Clock_Control.
        assert_eq!(clock.read_register(2, 0, 0x000FFF20), Some(0x00000000));
    }

    #[test]
    fn read_column_clock_control_returns_written_value() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(3, 0, 0x000FFF20, 0x1);
        assert_eq!(clock.read_register(3, 0, 0x000FFF20), Some(0x1));
    }

    #[test]
    fn module_kind_variants_exist() {
        let _ = ModuleKind::Core;
        let _ = ModuleKind::Memory;
        let _ = ModuleKind::Dma;
        let _ = ModuleKind::StreamSwitch;
    }

    #[test]
    fn is_module_active_default_false_for_all_kinds() {
        let clock = ClockController::new(5, 6);
        for kind in [ModuleKind::Core, ModuleKind::Memory, ModuleKind::Dma, ModuleKind::StreamSwitch] {
            assert!(!clock.is_module_active(2, 2, kind), "module {:?} should be gated at boot", kind);
        }
    }

    #[test]
    fn is_tile_active_default_false() {
        let clock = ClockController::new(5, 6);
        assert!(!clock.is_tile_active(2, 2));
    }
}
