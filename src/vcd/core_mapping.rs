//! Core/pipeline subsystem signal mapping.
//!
//! Compute tile only. VCD hierarchy under the `cm` (compute module) scope:
//!
//! ```text
//! tile_0_3.cm.pc_E1          (32-bit, pipeline stage 1)
//! tile_0_3.cm.pc_E2          (32-bit, pipeline stage 2)
//! tile_0_3.cm.pc_E3          (32-bit)
//! tile_0_3.cm.pc_E4          (32-bit)
//! tile_0_3.cm.pc_E5          (32-bit)
//! tile_0_3.cm.pc_E6          (32-bit)
//! tile_0_3.cm.pc_E7          (32-bit, pipeline stage 7)
//! tile_0_3.cm.pm_rd_in       (128-bit, instruction word from program memory)
//! tile_0_3.cm.pm_ad_out      (20-bit, program memory address)
//! tile_0_3.cm.tm_rd_in       (32-bit, tile memory read data)
//! tile_0_3.cm.tm_ad_out      (20-bit, tile memory address)
//! tile_0_3.cm.tm_wr_out      (32-bit, tile memory write data)
//! tile_0_3.cm.tm_ld_out      (1-bit, load active)
//! tile_0_3.cm.tm_st_out      (1-bit, store active)
//! tile_0_3.cm.reset          (1-bit, core reset)
//! tile_0_3.cm.pc_breakpoint_halted  (1-bit, breakpoint halted)
//! ```
//!
//! The pipeline stages are 1-indexed (E1..E7), matching the AIE2 VLIW
//! pipeline depth. The `iss.` scope prefix sometimes seen in older aietools
//! versions does NOT appear in the current AIE2 VCD output -- signals are
//! directly under `cm`.
//!
//! # Maps to
//!
//! - `pc_E{N}` -> [`StatePath::CorePc`] with `stage = N`
//! - `pm_rd_in` -> [`StatePath::CorePmData`]
//! - `pm_ad_out` -> [`StatePath::CorePmAddress`]
//! - `tm_rd_in` -> [`StatePath::CoreTmReadData`]
//! - `tm_ad_out` -> [`StatePath::CoreTmAddress`]
//! - `tm_wr_out` -> [`StatePath::CoreTmWriteData`]
//! - `tm_ld_out` -> [`StatePath::CoreTmLoad`]
//! - `tm_st_out` -> [`StatePath::CoreTmStore`]
//! - `reset` -> [`StatePath::CoreReset`]
//! - `pc_breakpoint_halted` -> [`StatePath::CoreBreakpointHalted`]

use crate::vcd::mapping::{NestedScopeMapping, SubsystemMapping, TileMapping};
use crate::vcd::state_path::{StatePath, Subsystem};

/// Build the core/pipeline subsystem mapping for a compute tile.
///
/// Returns a [`SubsystemMapping`] covering the VCD `cm` scope, mapping all
/// observable core state signals: pipeline stage PCs, program memory bus,
/// tile memory bus, load/store activity, reset, and breakpoint halt.
///
/// This mapping is only valid for compute tiles. Shim tiles and mem tiles
/// do not have a `cm` scope.
///
/// # Signal names (from aiesimulator VCD, AIE2)
///
/// The pipeline stage signals are `pc_E1` through `pc_E7` (1-indexed,
/// matching the AIE2 7-stage VLIW pipeline). These are registered as
/// individual fixed signals because the VCD uses 1-based indices while the
/// [`SubsystemMapping::indexed_signal`] API uses 0-based indices, and there
/// are only 7 stages.
pub fn core_mapping() -> SubsystemMapping {
    SubsystemMapping::new("cm", Subsystem::Core)
        // Pipeline stage program counters (E1 through E7, 1-indexed).
        // Using fixed_signal for each because there are only 7 and they are
        // 1-indexed, so looping with indexed_signal would require an index bias.
        .fixed_signal("pc_E1", 32, |col, row, _| StatePath::CorePc { col, row, stage: 1 })
        .fixed_signal("pc_E2", 32, |col, row, _| StatePath::CorePc { col, row, stage: 2 })
        .fixed_signal("pc_E3", 32, |col, row, _| StatePath::CorePc { col, row, stage: 3 })
        .fixed_signal("pc_E4", 32, |col, row, _| StatePath::CorePc { col, row, stage: 4 })
        .fixed_signal("pc_E5", 32, |col, row, _| StatePath::CorePc { col, row, stage: 5 })
        .fixed_signal("pc_E6", 32, |col, row, _| StatePath::CorePc { col, row, stage: 6 })
        .fixed_signal("pc_E7", 32, |col, row, _| StatePath::CorePc { col, row, stage: 7 })
        // Program memory bus: instruction word read and address output.
        .fixed_signal("pm_rd_in", 128, |col, row, _| StatePath::CorePmData { col, row })
        .fixed_signal("pm_ad_out", 20, |col, row, _| StatePath::CorePmAddress { col, row })
        // Tile (data) memory bus: address, read data, write data.
        .fixed_signal("tm_rd_in", 32, |col, row, _| StatePath::CoreTmReadData { col, row })
        .fixed_signal("tm_ad_out", 20, |col, row, _| StatePath::CoreTmAddress { col, row })
        .fixed_signal("tm_wr_out", 32, |col, row, _| StatePath::CoreTmWriteData { col, row })
        // Load/store activity signals (1-bit each).
        .fixed_signal("tm_ld_out", 1, |col, row, _| StatePath::CoreTmLoad { col, row })
        .fixed_signal("tm_st_out", 1, |col, row, _| StatePath::CoreTmStore { col, row })
        // Core control signals.
        .fixed_signal("reset", 1, |col, row, _| StatePath::CoreReset { col, row })
        .fixed_signal("pc_breakpoint_halted", 1, |col, row, _| StatePath::CoreBreakpointHalted { col, row })
}

/// Build the core mapping for VC2802 aiesimulator VCD format.
///
/// In aiesimulator VCD output, core signals are nested deeper than in the
/// idealized NPU1 mapping:
///
/// ```text
/// tile_7_3.cm.proc.pc_E1              (PC stages under cm.proc)
/// tile_7_3.cm.proc.iss.pm_rd_in       (ISS signals under cm.proc.iss)
/// tile_7_3.cm.proc.iss.reset          (reset under cm.proc.iss)
/// tile_7_3.cm.proc.iss.tm_rd_in       (tile memory under cm.proc.iss)
/// tile_7_3.cm.proc.performance_counter.counter_event_value_0
/// ```
///
/// This function returns a [`CompositeMapping`] with scope `"cm"` that
/// handles both nesting levels.
pub fn core_mapping_vc2802() -> CompositeMapping {
    // Combine: cm contains proc (with PCs) and proc.iss (with ISS signals).
    // The CompositeMapping tries each child in order.
    CompositeMapping::new(
        "cm",
        vec![
            Box::new(proc_pc_mapping()),
            Box::new(NestedScopeMapping::new("proc", Box::new(proc_iss_mapping()))),
        ],
    )
}

/// The pipeline-stage PC mapping under the `proc` scope (`cm.proc.pc_E1`..`pc_E7`).
///
/// Exposed so other device trees (e.g. the in-process NPU1 cluster, which uses
/// the same `cm.proc` / `cm.proc.iss` nesting) can reuse it without duplicating
/// the signal table. PC stages are 1-indexed, matching the AIE2 7-stage VLIW
/// pipeline.
pub fn proc_pc_mapping() -> SubsystemMapping {
    SubsystemMapping::new("proc", Subsystem::Core)
        .fixed_signal("pc_E1", 32, |col, row, _| StatePath::CorePc { col, row, stage: 1 })
        .fixed_signal("pc_E2", 32, |col, row, _| StatePath::CorePc { col, row, stage: 2 })
        .fixed_signal("pc_E3", 32, |col, row, _| StatePath::CorePc { col, row, stage: 3 })
        .fixed_signal("pc_E4", 32, |col, row, _| StatePath::CorePc { col, row, stage: 4 })
        .fixed_signal("pc_E5", 32, |col, row, _| StatePath::CorePc { col, row, stage: 5 })
        .fixed_signal("pc_E6", 32, |col, row, _| StatePath::CorePc { col, row, stage: 6 })
        .fixed_signal("pc_E7", 32, |col, row, _| StatePath::CorePc { col, row, stage: 7 })
}

/// The ISS bus mapping under the `iss` scope (`cm.proc.iss.pm_rd_in`, etc.).
///
/// Exposed for reuse by other device trees sharing the `cm.proc.iss` nesting.
/// Covers the program-memory and tile-memory buses, reset, and breakpoint halt
/// -- the architecturally-observable core state. The ISS model's internal bus
/// lanes (dme/dmo/lock adaptors) are not core state and fall to the raw tier.
pub fn proc_iss_mapping() -> SubsystemMapping {
    SubsystemMapping::new("iss", Subsystem::Core)
        .fixed_signal("pm_rd_in", 128, |col, row, _| StatePath::CorePmData { col, row })
        .fixed_signal("pm_ad_out", 20, |col, row, _| StatePath::CorePmAddress { col, row })
        .fixed_signal("tm_rd_in", 32, |col, row, _| StatePath::CoreTmReadData { col, row })
        .fixed_signal("tm_ad_out", 20, |col, row, _| StatePath::CoreTmAddress { col, row })
        .fixed_signal("tm_wr_out", 32, |col, row, _| StatePath::CoreTmWriteData { col, row })
        .fixed_signal("tm_ld_out", 1, |col, row, _| StatePath::CoreTmLoad { col, row })
        .fixed_signal("tm_st_out", 1, |col, row, _| StatePath::CoreTmStore { col, row })
        .fixed_signal("reset", 1, |col, row, _| StatePath::CoreReset { col, row })
        .fixed_signal("pc_breakpoint_halted", 1, |col, row, _| StatePath::CoreBreakpointHalted { col, row })
}

/// A composite mapping that tries multiple child mappings in order.
///
/// Used when a single VCD scope contains signals with different sub-scope
/// structures. For example, `cm` in VC2802 VCDs contains both `proc.pc_E1`
/// (handled by one SubsystemMapping) and `proc.iss.pm_rd_in` (handled by
/// a NestedScopeMapping wrapping another SubsystemMapping).
pub struct CompositeMapping {
    scope: String,
    children: Vec<Box<dyn TileMapping>>,
}

impl CompositeMapping {
    pub fn new(scope: &str, children: Vec<Box<dyn TileMapping>>) -> Self {
        CompositeMapping { scope: scope.to_string(), children }
    }
}

impl TileMapping for CompositeMapping {
    fn scope_name(&self) -> &str {
        &self.scope
    }

    fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        for child in &self.children {
            // Try each child's scope_name against the first segment.
            if !segments.is_empty() && segments[0] == child.scope_name() {
                if let Some(path) = child.resolve(&segments[1..], col, row) {
                    return Some(path);
                }
            }
        }
        None
    }

    fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        let mut paths = Vec::new();
        for child in &self.children {
            paths.extend(child.enumerate(col, row));
        }
        paths
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::StatePath;

    // -- PC stage resolution --

    #[test]
    fn core_resolves_pc_e1() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["pc_E1"], 0, 3);
        assert_eq!(result, Some(StatePath::CorePc { col: 0, row: 3, stage: 1 }));
    }

    #[test]
    fn core_resolves_pc_e3() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["pc_E3"], 0, 3);
        assert_eq!(result, Some(StatePath::CorePc { col: 0, row: 3, stage: 3 }));
    }

    #[test]
    fn core_resolves_pc_e7() {
        // Stage 7 is the last pipeline stage.
        let mapping = core_mapping();
        let result = mapping.resolve(&["pc_E7"], 2, 4);
        assert_eq!(result, Some(StatePath::CorePc { col: 2, row: 4, stage: 7 }));
    }

    #[test]
    fn core_resolves_all_pipeline_stages() {
        let mapping = core_mapping();
        for stage in 1u8..=7 {
            let name = format!("pc_E{}", stage);
            let result = mapping.resolve(&[name.as_str()], 1, 2);
            assert_eq!(
                result,
                Some(StatePath::CorePc { col: 1, row: 2, stage }),
                "Expected CorePc for stage {} at pc_E{}",
                stage,
                stage
            );
        }
    }

    // -- Program memory bus --

    #[test]
    fn core_resolves_pm_data() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["pm_rd_in"], 1, 2);
        assert_eq!(result, Some(StatePath::CorePmData { col: 1, row: 2 }));
    }

    #[test]
    fn core_resolves_pm_address() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["pm_ad_out"], 0, 3);
        assert_eq!(result, Some(StatePath::CorePmAddress { col: 0, row: 3 }));
    }

    // -- Tile (data) memory bus --

    #[test]
    fn core_resolves_tm_read_data() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["tm_rd_in"], 0, 3);
        assert_eq!(result, Some(StatePath::CoreTmReadData { col: 0, row: 3 }));
    }

    #[test]
    fn core_resolves_tm_address() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["tm_ad_out"], 1, 4);
        assert_eq!(result, Some(StatePath::CoreTmAddress { col: 1, row: 4 }));
    }

    #[test]
    fn core_resolves_tm_write_data() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["tm_wr_out"], 0, 2);
        assert_eq!(result, Some(StatePath::CoreTmWriteData { col: 0, row: 2 }));
    }

    #[test]
    fn core_resolves_tm_load() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["tm_ld_out"], 2, 3);
        assert_eq!(result, Some(StatePath::CoreTmLoad { col: 2, row: 3 }));
    }

    #[test]
    fn core_resolves_tm_store() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["tm_st_out"], 1, 5);
        assert_eq!(result, Some(StatePath::CoreTmStore { col: 1, row: 5 }));
    }

    // -- Control signals --

    #[test]
    fn core_resolves_reset() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["reset"], 0, 3);
        assert_eq!(result, Some(StatePath::CoreReset { col: 0, row: 3 }));
    }

    #[test]
    fn core_resolves_breakpoint_halted() {
        let mapping = core_mapping();
        let result = mapping.resolve(&["pc_breakpoint_halted"], 3, 4);
        assert_eq!(result, Some(StatePath::CoreBreakpointHalted { col: 3, row: 4 }));
    }

    // -- Boundary and rejection tests --

    #[test]
    fn core_rejects_unknown_signal() {
        let mapping = core_mapping();
        assert_eq!(mapping.resolve(&["nonexistent"], 0, 3), None);
        // pc_E0 does not exist (stages are 1-indexed).
        assert_eq!(mapping.resolve(&["pc_E0"], 0, 3), None);
        // pc_E8 does not exist (only 7 stages).
        assert_eq!(mapping.resolve(&["pc_E8"], 0, 3), None);
    }

    #[test]
    fn core_rejects_iss_prefixed_names() {
        // The current AIE2 VCD output does NOT use the "iss." scope prefix.
        // If this test fails, it means the VCD format has changed.
        let mapping = core_mapping();
        assert_eq!(mapping.resolve(&["iss.pc_E1"], 0, 3), None);
    }

    #[test]
    fn core_tile_coordinates_propagated() {
        let mapping = core_mapping();
        let a = mapping.resolve(&["pc_E1"], 0, 2).unwrap();
        let b = mapping.resolve(&["pc_E1"], 3, 5).unwrap();
        assert_ne!(a, b);
        assert_eq!(a, StatePath::CorePc { col: 0, row: 2, stage: 1 });
        assert_eq!(b, StatePath::CorePc { col: 3, row: 5, stage: 1 });
    }

    #[test]
    fn core_scope_name_is_cm() {
        let mapping = core_mapping();
        assert_eq!(mapping.scope_name(), "cm");
    }

    // -- Enumeration tests --

    #[test]
    fn core_enumerates_all_signals() {
        let mapping = core_mapping();
        let paths = mapping.enumerate(0, 3);
        // 7 PC stages + pm_rd_in + pm_ad_out + tm_rd_in + tm_ad_out + tm_wr_out
        // + tm_ld_out + tm_st_out + reset + pc_breakpoint_halted = 16 total
        assert_eq!(paths.len(), 16);
    }

    #[test]
    fn core_enumerate_contains_all_stages() {
        let mapping = core_mapping();
        let paths = mapping.enumerate(0, 3);
        for stage in 1u8..=7 {
            assert!(
                paths.contains(&StatePath::CorePc { col: 0, row: 3, stage }),
                "Missing CorePc stage {}",
                stage
            );
        }
    }

    #[test]
    fn core_enumerate_contains_all_signals() {
        let mapping = core_mapping();
        let paths = mapping.enumerate(1, 2);
        assert!(paths.contains(&StatePath::CorePmData { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CorePmAddress { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreTmReadData { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreTmAddress { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreTmWriteData { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreTmLoad { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreTmStore { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreReset { col: 1, row: 2 }));
        assert!(paths.contains(&StatePath::CoreBreakpointHalted { col: 1, row: 2 }));
    }

    #[test]
    fn core_enumerate_tile_coordinates_propagated() {
        let mapping = core_mapping();
        let paths = mapping.enumerate(2, 4);
        assert!(paths.iter().all(|p| p.tile() == (2, 4)));
    }

    #[test]
    fn core_enumerate_all_are_core_subsystem() {
        let mapping = core_mapping();
        let paths = mapping.enumerate(0, 3);
        use crate::vcd::state_path::Subsystem;
        assert!(
            paths.iter().all(|p| p.subsystem() == Subsystem::Core),
            "All core signals should belong to Core subsystem"
        );
    }
}
