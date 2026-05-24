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

/// AM025 offset for Module_Clock_Control on compute tiles.
const MCC_COMPUTE_OFFSET: u32 = 0x00060000;
/// AM025 offset for Module_Clock_Control on memtile and shim (MCC_0).
/// NOTE: MCC_MEMTILE_OFFSET == MCC_SHIM_0_OFFSET == 0x000FFF00.  They share
/// the same offset; the row number disambiguates (row 0 = shim, row 1 =
/// memtile).  They must NOT appear as separate match arms -- Rust errors on
/// duplicate patterns.
const MCC_MEMTILE_OFFSET: u32 = 0x000FFF00;
/// AM025 offset for Module_Clock_Control_1 (shim only, NoC enable).
const MCC_SHIM_1_OFFSET: u32 = 0x000FFF04;

/// Internal tile-kind discriminator for bit-layout selection.
/// Uses a clock-control-local enum rather than xdna_archspec::types::TileKind
/// to avoid coupling the controller to the broader archspec model -- this
/// only needs three buckets (Compute / Memtile / Shim) regardless of how
/// many shim variants exist elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockTileKind {
    Compute,
    Memtile,
    Shim,
}

fn clock_tile_kind_from_row(row: u8) -> ClockTileKind {
    if row == 0 {
        ClockTileKind::Shim
    } else if row == 1 {
        ClockTileKind::Memtile
    } else {
        ClockTileKind::Compute
    }
}

/// AM025 reset value for Module_Clock_Control (and MCC_0 for shim).
/// Confirmed from aie_registers_aie2.json on 2026-05-24.
fn reset_value_for_mcc(tile_kind: ClockTileKind) -> u32 {
    match tile_kind {
        ClockTileKind::Compute => 0x37,
        ClockTileKind::Memtile => 0x33,
        ClockTileKind::Shim => 0x3B,
    }
}

/// AM025 reset value for shim Module_Clock_Control_1 (NoC enable bit set).
fn reset_value_for_mcc_1() -> u32 {
    0x01
}

/// Decode `kind` from a Module_Clock_Control register value.
/// Bit positions confirmed from aie_registers_aie2.json on 2026-05-24.
///
/// | ModuleKind    | Compute      | Memtile      | Shim              |
/// |---------------|--------------|--------------|-------------------|
/// | Core          | MCC bit 2    | always false | always false      |
/// | Memory        | MCC bit 1    | MCC bit 1    | always false      |
/// | Dma           | MCC bit 1    | MCC bit 1    | MCC_1 bit 0 (NoC) |
/// | StreamSwitch  | MCC bit 0    | MCC bit 0    | MCC_0 bit 0       |
///
/// On compute/memtile Dma and Memory share the same physical bit -- the
/// silicon does not separately clock-gate DMA from data memory.
fn mcc_module_active(
    raw_mcc_0: u32,
    raw_mcc_1: Option<u32>,
    tile_kind: ClockTileKind,
    kind: ModuleKind,
) -> bool {
    use ClockTileKind::*;
    use ModuleKind::*;
    let (reg, bit): (u32, u8) = match (tile_kind, kind) {
        // Compute tile: MCC bit 2 = Core, bit 1 = Memory (= Dma), bit 0 = SS.
        (Compute, Core) => (raw_mcc_0, 2),
        (Compute, Memory) => (raw_mcc_0, 1),
        (Compute, Dma) => (raw_mcc_0, 1), // same bit as Memory
        (Compute, StreamSwitch) => (raw_mcc_0, 0),
        // Memtile: no Core; bit 1 = Memory (= Dma), bit 0 = SS.
        (Memtile, Core) => return false,
        (Memtile, Memory) => (raw_mcc_0, 1),
        (Memtile, Dma) => (raw_mcc_0, 1),
        (Memtile, StreamSwitch) => (raw_mcc_0, 0),
        // Shim: no Core / Memory.  Dma lives in MCC_1 bit 0 (NoC module).
        (Shim, Core) => return false,
        (Shim, Memory) => return false,
        (Shim, Dma) => match raw_mcc_1 {
            Some(r) => (r, 0),
            None => return false,
        },
        (Shim, StreamSwitch) => (raw_mcc_0, 0),
    };
    (reg >> bit) & 0x1 != 0
}

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
    pub fn is_module_active(&self, col: u8, row: u8, kind: ModuleKind) -> bool {
        if !self.is_column_active(col) {
            return false;
        }
        let tile_kind = clock_tile_kind_from_row(row);
        let (raw_mcc_0, raw_mcc_1) = match self.tiles.get(&(col, row)) {
            Some(gates) => (gates.raw_mcc_0, gates.raw_mcc_1),
            None => (
                reset_value_for_mcc(tile_kind),
                if matches!(tile_kind, ClockTileKind::Shim) {
                    Some(reset_value_for_mcc_1())
                } else {
                    None
                },
            ),
        };
        mcc_module_active(raw_mcc_0, raw_mcc_1, tile_kind, kind)
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
            // MCC_MEMTILE_OFFSET == MCC_SHIM_0_OFFSET == 0x000FFF00; row
            // disambiguates.  Both write into raw_mcc_0.
            MCC_COMPUTE_OFFSET | MCC_MEMTILE_OFFSET => {
                let entry = self
                    .tiles
                    .entry((col, row))
                    .or_insert_with(|| TileGates { raw_mcc_0: 0, raw_mcc_1: None });
                entry.raw_mcc_0 = value;
            }
            MCC_SHIM_1_OFFSET if row == 0 => {
                let entry = self
                    .tiles
                    .entry((col, row))
                    .or_insert_with(|| TileGates { raw_mcc_0: 0, raw_mcc_1: Some(0) });
                entry.raw_mcc_1 = Some(value);
            }
            _ => {} // not a clock-control offset
        }
    }

    /// Read a clock-control register.  Returns the current value, or the
    /// AM025 reset value if the register has not been written yet.
    /// Returns None if the offset is not a known clock-control register.
    pub fn read_register(&self, col: u8, row: u8, offset: u32) -> Option<u32> {
        let tile_kind = clock_tile_kind_from_row(row);
        match offset {
            COLUMN_CLOCK_CONTROL_OFFSET if row == 0 => {
                // Reflect current state: bit 0 = column enabled.
                Some(if self.is_column_active(col) { 0x1 } else { 0x0 })
            }
            // MCC_MEMTILE_OFFSET == MCC_SHIM_0_OFFSET == 0x000FFF00.
            MCC_COMPUTE_OFFSET | MCC_MEMTILE_OFFSET => self
                .tiles
                .get(&(col, row))
                .map(|g| g.raw_mcc_0)
                .or_else(|| Some(reset_value_for_mcc(tile_kind))),
            MCC_SHIM_1_OFFSET if row == 0 => self
                .tiles
                .get(&(col, row))
                .and_then(|g| g.raw_mcc_1)
                .or_else(|| Some(reset_value_for_mcc_1())),
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

    // ---- Task 5: Module_Clock_Control bit-field decode ----

    const MCC_COMPUTE_OFFSET: u32 = 0x00060000;
    const MCC_MEMTILE_OFFSET: u32 = 0x000FFF00;
    const MCC_SHIM_0_OFFSET: u32 = 0x000FFF00;
    const MCC_SHIM_1_OFFSET: u32 = 0x000FFF04;

    #[test]
    fn mcc_compute_decodes_core_bit() {
        // Bit positions per aie_registers_aie2.json (2026-05-24 lookup).
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1); // ungate column 2
                                                     // Compute MCC bit 2 = Core_Module_Clock_Enable.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 2);
        assert!(clock.is_module_active(2, 3, ModuleKind::Core));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Memory));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Dma));
        assert!(!clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
    }

    #[test]
    fn mcc_compute_decodes_memory_bit_as_both_memory_and_dma() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        // Compute MCC bit 1 = Memory_Module_Clock_Enable.  Same bit clocks DMA.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 1);
        assert!(clock.is_module_active(2, 3, ModuleKind::Memory));
        assert!(clock.is_module_active(2, 3, ModuleKind::Dma));
        assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
    }

    #[test]
    fn mcc_compute_decodes_ss_bit() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 0);
        assert!(clock.is_module_active(2, 3, ModuleKind::StreamSwitch));
    }

    #[test]
    fn mcc_memtile_no_core() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        // Set everything we can on memtile.
        clock.write_register(2, 1, MCC_MEMTILE_OFFSET, 0xFFFF_FFFF);
        // Memtile has no Core module -- always false regardless of writes.
        assert!(!clock.is_module_active(2, 1, ModuleKind::Core));
        // Memory and Dma both reflect bit 1.
        assert!(clock.is_module_active(2, 1, ModuleKind::Memory));
        assert!(clock.is_module_active(2, 1, ModuleKind::Dma));
        assert!(clock.is_module_active(2, 1, ModuleKind::StreamSwitch));
    }

    #[test]
    fn mcc_shim_dma_lives_in_mcc_1_not_mcc_0() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        // MCC_0 with everything set; MCC_1 explicitly 0.
        clock.write_register(2, 0, MCC_SHIM_0_OFFSET, 0xFFFF_FFFF);
        clock.write_register(2, 0, MCC_SHIM_1_OFFSET, 0x0);
        // SS comes from MCC_0 bit 0 -> active.
        assert!(clock.is_module_active(2, 0, ModuleKind::StreamSwitch));
        // DMA on shim comes from MCC_1 bit 0 (NoC) -> inactive.
        assert!(!clock.is_module_active(2, 0, ModuleKind::Dma));
        // Shim has no Core or Memory.
        assert!(!clock.is_module_active(2, 0, ModuleKind::Core));
        assert!(!clock.is_module_active(2, 0, ModuleKind::Memory));
    }

    #[test]
    fn mcc_shim_dma_lives_in_mcc_1_bit_0_when_set() {
        let mut clock = ClockController::new(5, 6);
        clock.write_register(2, 0, 0x000FFF20, 0x1);
        clock.write_register(2, 0, MCC_SHIM_1_OFFSET, 1 << 0);
        assert!(clock.is_module_active(2, 0, ModuleKind::Dma));
    }

    #[test]
    fn module_inactive_when_column_gated_even_with_mcc_set() {
        let mut clock = ClockController::new(5, 6);
        // Column 2 NOT ungated.
        clock.write_register(2, 3, MCC_COMPUTE_OFFSET, 1 << 2); // try to enable Core
                                                                // Column gate dominates -- everything inactive.
        assert!(!clock.is_module_active(2, 3, ModuleKind::Core));
    }

    #[test]
    fn read_mcc_returns_am025_reset_value_before_any_write() {
        let clock = ClockController::new(5, 6);
        // Compute tile reset is 0x37.
        assert_eq!(clock.read_register(2, 3, MCC_COMPUTE_OFFSET), Some(0x37));
        // Memtile reset is 0x33.
        assert_eq!(clock.read_register(2, 1, MCC_MEMTILE_OFFSET), Some(0x33));
        // Shim Module_Clock_Control_0 reset is 0x3B.
        assert_eq!(clock.read_register(2, 0, MCC_SHIM_0_OFFSET), Some(0x3B));
        // Shim Module_Clock_Control_1 reset is 0x01.
        assert_eq!(clock.read_register(2, 0, MCC_SHIM_1_OFFSET), Some(0x01));
    }
}
