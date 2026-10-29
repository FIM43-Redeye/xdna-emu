//! Coverage audit: compare VCD signals against the mapping tree.
//!
//! Loads a real aiesimulator VCD file via the `wellen` crate, walks every
//! variable in its signal hierarchy, and reports how many signals are mapped
//! (i.e., the [`MappingTree`] can resolve them to a [`StatePath`]) vs unmapped
//! (no emulator counterpart defined).
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::vcd::coverage::coverage_audit;
//!
//! let tree = build_my_mapping_tree();
//! let report = coverage_audit("/tmp/aiesim-test2/trace.vcd", &tree).unwrap();
//! println!("{}", report);
//! ```

use crate::vcd::mapping::MappingTree;
use crate::vcd::state_path::Subsystem;
use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// CoverageReport
// ---------------------------------------------------------------------------

/// Summary produced by [`coverage_audit`].
///
/// Counts how many VCD signals the mapping tree can resolve, broken down by
/// subsystem for mapped signals and by hierarchy prefix for unmapped signals.
pub struct CoverageReport {
    /// Total number of variables seen in the VCD.
    pub total_count: usize,
    /// Number of variables that resolve to a [`StatePath`] via the tree.
    pub mapped_count: usize,
    /// Number of variables that do not resolve (no emulator counterpart).
    pub unmapped_count: usize,
    /// Unmapped signals grouped by prefix (first 3 hierarchy levels joined
    /// by '.'), mapped to the count of signals with that prefix.
    pub unmapped_groups: BTreeMap<String, usize>,
    /// Mapped signal counts per subsystem.
    pub mapped_by_subsystem: BTreeMap<Subsystem, usize>,
}

// ---------------------------------------------------------------------------
// coverage_audit
// ---------------------------------------------------------------------------

/// Run a coverage audit on a VCD file against a mapping tree.
///
/// Reads the VCD header (which contains the full signal hierarchy) via
/// [`wellen::simple::read`], then walks every variable. For each variable:
/// 1. Splits its full dotted name into segments.
/// 2. Calls [`MappingTree::resolve`] on those segments.
/// 3. Counts the result as mapped or unmapped.
///
/// Returns a [`CoverageReport`] on success, or a descriptive error string on
/// failure (e.g., the file is not a valid VCD).
pub fn coverage_audit(vcd_path: &str, tree: &MappingTree) -> Result<CoverageReport, String> {
    // Guard against missing files before calling wellen. wellen panics rather
    // than returning Err when the path does not exist (it calls unwrap() on the
    // file open internally), so we must check upfront.
    if !std::path::Path::new(vcd_path).exists() {
        return Err(format!("VCD file not found: {}", vcd_path));
    }

    // Load the VCD hierarchy. wellen reads the full header (scopes and
    // variable declarations) during this call; the signal body data is also
    // loaded but we do not use it here.
    let waveform =
        wellen::simple::read(vcd_path).map_err(|e| format!("Failed to read VCD '{}': {:?}", vcd_path, e))?;
    let hierarchy = waveform.hierarchy();

    let mut total = 0usize;
    let mut mapped = 0usize;
    let mut unmapped_groups: BTreeMap<String, usize> = BTreeMap::new();
    let mut mapped_by_subsystem: BTreeMap<Subsystem, usize> = BTreeMap::new();

    // iter_vars() returns a flat iterator over every Var in the hierarchy,
    // regardless of nesting depth. Each Var knows its full dotted path via
    // full_name(&hierarchy).
    for var in hierarchy.iter_vars() {
        total += 1;
        let full_name = var.full_name(hierarchy);

        // Split the dotted name into segments for tree resolution.
        let segments: Vec<&str> = full_name.split('.').collect();

        if let Some(path) = tree.resolve(&segments) {
            mapped += 1;
            *mapped_by_subsystem.entry(path.subsystem()).or_insert(0) += 1;
        } else {
            // Group unmapped signals by their first 3 hierarchy levels.
            // This groups together all signals under e.g. "top.math_engine.shim"
            // so the report highlights which areas are entirely unmapped.
            let prefix = segments.iter().take(3).cloned().collect::<Vec<_>>().join(".");
            *unmapped_groups.entry(prefix).or_insert(0) += 1;
        }
    }

    Ok(CoverageReport {
        total_count: total,
        mapped_count: mapped,
        unmapped_count: total - mapped,
        unmapped_groups,
        mapped_by_subsystem,
    })
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for CoverageReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Signal Coverage Report:")?;
        writeln!(
            f,
            "  MAPPED:    {:>6} / {} ({:.1}%)",
            self.mapped_count,
            self.total_count,
            if self.total_count > 0 {
                self.mapped_count as f64 / self.total_count as f64 * 100.0
            } else {
                0.0
            }
        )?;
        writeln!(f, "  UNMAPPED:  {:>6} signals (no emulator counterpart)", self.unmapped_count)?;

        if !self.mapped_by_subsystem.is_empty() {
            writeln!(f, "\n  Mapped by subsystem:")?;
            for (sub, count) in &self.mapped_by_subsystem {
                writeln!(f, "    {:<15} {:>6}", sub.as_str(), count)?;
            }
        }

        if !self.unmapped_groups.is_empty() {
            writeln!(f, "\n  Unmapped by prefix (top 20):")?;
            let mut groups: Vec<_> = self.unmapped_groups.iter().collect();
            // Sort by descending count so the biggest gaps appear first.
            groups.sort_by(|a, b| b.1.cmp(a.1));
            for (prefix, count) in groups.iter().take(20) {
                writeln!(f, "    {:<40} {:>6}", prefix, count)?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::dma_mapping::dma_mapping;
    use crate::vcd::lock_mapping::lock_mapping;
    use crate::vcd::mapping::MappingTree;
    use crate::vcd::state_path::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a small mapping tree covering a representative NPU1 subset:
    /// - mem_row (tile 0,1): 64 locks + 6 s2mm/6 mm2s DMA channels
    /// - array (tiles 0..3, rows 2..5): 16 locks + 2 s2mm/2 mm2s DMA
    /// - shim (tiles 0..3, row 0): 16 locks
    ///
    /// This matches the AIE2 (xcve2802) device shape that aiesimulator uses
    /// for NPU1 programs.
    fn build_npu1_tree() -> MappingTree {
        // Compute column range 0-3, row range 2-5 (above mem row at 1)
        let array_tiles: Vec<(u8, u8)> = (0u8..4).flat_map(|c| (2u8..6).map(move |r| (c, r))).collect();

        // Shim tiles at row 0, columns 0-3
        let shim_tiles: Vec<(u8, u8)> = (0u8..4).map(|c| (c, 0u8)).collect();

        MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            // mem_row: memtile with 64 locks and 6+6 DMA channels
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(lock_mapping(64))
            .subsystem(dma_mapping(6, 6))
            .done_tile_group()
            // array: compute tiles with 16 locks and 2+2 DMA channels
            .tile_group("array", &array_tiles)
            .subsystem(lock_mapping(16))
            .subsystem(dma_mapping(2, 2))
            .done_tile_group()
            // shim row: shim tiles with 16 locks
            .tile_group("shim", &shim_tiles)
            .subsystem(lock_mapping(16))
            .done_tile_group()
            .build()
    }

    // -----------------------------------------------------------------------
    // Real VCD test (skips gracefully when file is absent)
    // -----------------------------------------------------------------------

    #[test]
    fn coverage_report_from_real_vcd() {
        let vcd_path = "/tmp/aiesim-test2/trace.vcd";
        if !std::path::Path::new(vcd_path).exists() {
            eprintln!("Skipping: {} not found (run aiesim to generate)", vcd_path);
            return;
        }

        let tree = build_npu1_tree();
        let report = coverage_audit(vcd_path, &tree).unwrap();

        // A real NPU1 aiesim VCD should have tens of thousands of signals.
        assert!(
            report.total_count > 1_000,
            "Expected at least 1,000 total signals, got {}",
            report.total_count
        );

        // Even a partial tree should map at least some lock signals.
        assert!(report.mapped_count > 0, "Expected some mapped signals; total={}", report.total_count);

        eprintln!("{}", report);
    }

    // -----------------------------------------------------------------------
    // Display tests (no VCD file required)
    // -----------------------------------------------------------------------

    #[test]
    fn coverage_report_display_percentage() {
        let report = CoverageReport {
            total_count: 1000,
            mapped_count: 800,
            unmapped_count: 200,
            unmapped_groups: BTreeMap::from([
                ("foo.bar.baz".to_string(), 150),
                ("foo.bar.qux".to_string(), 50),
            ]),
            mapped_by_subsystem: BTreeMap::from([(Subsystem::Lock, 500), (Subsystem::Dma, 300)]),
        };
        let text = format!("{}", report);
        assert!(text.contains("80.0%"), "Expected '80.0%' in:\n{}", text);
        assert!(text.contains("lock"), "Expected 'lock' in:\n{}", text);
        assert!(text.contains("dma"), "Expected 'dma' in:\n{}", text);
    }

    #[test]
    fn coverage_report_display_100_percent() {
        let report = CoverageReport {
            total_count: 50,
            mapped_count: 50,
            unmapped_count: 0,
            unmapped_groups: BTreeMap::new(),
            mapped_by_subsystem: BTreeMap::from([(Subsystem::Lock, 50)]),
        };
        let text = format!("{}", report);
        assert!(text.contains("100.0%"), "Expected '100.0%' in:\n{}", text);
        // No unmapped groups: the section should be absent.
        assert!(!text.contains("Unmapped by prefix"), "Expected no unmapped prefix section");
    }

    #[test]
    fn coverage_report_display_zero_total() {
        // Edge case: empty VCD (no variables).
        let report = CoverageReport {
            total_count: 0,
            mapped_count: 0,
            unmapped_count: 0,
            unmapped_groups: BTreeMap::new(),
            mapped_by_subsystem: BTreeMap::new(),
        };
        let text = format!("{}", report);
        // Should report 0.0% without panicking on division by zero.
        assert!(text.contains("0.0%"), "Expected '0.0%' in:\n{}", text);
    }

    #[test]
    fn coverage_report_display_shows_top_20_unmapped() {
        // Build a report with 25 unmapped prefix groups; only top 20 shown.
        let groups: BTreeMap<String, usize> =
            (0u32..25).map(|i| (format!("top.scope.group_{}", i), i as usize + 1)).collect();
        let report = CoverageReport {
            total_count: 300,
            mapped_count: 0,
            unmapped_count: 300,
            unmapped_groups: groups,
            mapped_by_subsystem: BTreeMap::new(),
        };
        let text = format!("{}", report);
        // Count how many "group_" entries appear. Should be exactly 20.
        let group_count = text.matches("group_").count();
        assert_eq!(group_count, 20, "Expected exactly 20 unmapped groups displayed, got {}", group_count);
    }

    // -----------------------------------------------------------------------
    // coverage_audit unit test using a synthetic in-memory VCD
    // -----------------------------------------------------------------------

    /// Build a minimal VCD file as a byte string for use in unit tests.
    ///
    /// The VCD declares three variables:
    /// - `top.math_engine.mem_row.tile_0_1.locks.value_0` -- maps to LockValue
    /// - `top.math_engine.mem_row.tile_0_1.locks.lock_op_0` -- maps to LockOp
    /// - `top.other_scope.some_signal` -- does not map
    fn make_synthetic_vcd() -> Vec<u8> {
        // Minimal syntactically-valid VCD. The signal body is empty (just $end
        // after the header) which is sufficient for hierarchy parsing.
        let content = "\
$timescale 1 ns $end\n\
$scope module top $end\n\
$scope module math_engine $end\n\
$scope module mem_row $end\n\
$scope module tile_0_1 $end\n\
$scope module locks $end\n\
$var wire 32 ! value_0 $end\n\
$var wire 32 \" lock_op_0 $end\n\
$upscope $end\n\
$upscope $end\n\
$upscope $end\n\
$upscope $end\n\
$scope module other_scope $end\n\
$var wire 1 # some_signal $end\n\
$upscope $end\n\
$upscope $end\n\
$enddefinitions $end\n\
";
        content.as_bytes().to_vec()
    }

    #[test]
    fn coverage_audit_synthetic_vcd() {
        // Write the synthetic VCD to a temp file.
        let dir = std::env::temp_dir();
        let vcd_file = dir.join("xdna_emu_coverage_test.vcd");
        std::fs::write(&vcd_file, make_synthetic_vcd()).expect("Failed to write synthetic VCD");

        // Build a tree that covers the two mapped signals.
        let tree = MappingTree::builder()
            .scope("top")
            .scope("math_engine")
            .tile_group("mem_row", &[(0, 1)])
            .subsystem(lock_mapping(64))
            .done_tile_group()
            .build();

        let path_str = vcd_file.to_str().expect("temp path is valid UTF-8");
        let report = coverage_audit(path_str, &tree).expect("coverage_audit should succeed");

        // 3 total signals: 2 mapped (lock value + op), 1 unmapped (other_scope)
        assert_eq!(report.total_count, 3, "total_count mismatch");
        assert_eq!(report.mapped_count, 2, "mapped_count mismatch");
        assert_eq!(report.unmapped_count, 1, "unmapped_count mismatch");

        // The mapped signals are both locks.
        assert_eq!(
            report.mapped_by_subsystem.get(&Subsystem::Lock),
            Some(&2),
            "expected 2 lock signals mapped"
        );
        assert_eq!(report.mapped_by_subsystem.get(&Subsystem::Dma), None, "no DMA signals expected");

        // The unmapped group prefix should be the 3-segment prefix of "top.other_scope.some_signal".
        assert!(
            report.unmapped_groups.contains_key("top.other_scope.some_signal"),
            "expected 'top.other_scope.some_signal' in unmapped groups, got: {:?}",
            report.unmapped_groups
        );

        std::fs::remove_file(&vcd_file).ok();
    }

    #[test]
    fn coverage_audit_missing_file_returns_error() {
        let tree = MappingTree::builder().build();
        let result = coverage_audit("/nonexistent/path/to/trace.vcd", &tree);
        assert!(result.is_err(), "Expected error for missing file, got Ok");
    }
}
