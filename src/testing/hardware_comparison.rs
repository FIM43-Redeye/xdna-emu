//! Three-way comparison logic for emulator-vs-hardware validation.
//!
//! Provides three validation layers:
//! 1. **Emu vs Manifest**: Does the emulator produce the expected output?
//!    (Already exists in xclbin_suite -- this module adds the other two.)
//! 2. **HW vs Manifest**: Does real hardware produce the expected output?
//! 3. **Emu vs HW**: Do the emulator and hardware agree?
//!
//! Layer 1 catches emulator bugs against spec. Layer 2 catches manifest
//! errors. Layer 3 catches emulator/hardware divergence regardless of
//! the "correct" answer.
//!
//! This module is pure logic -- no hardware dependency. It works on raw
//! byte buffers from either live capture or saved files.

use std::collections::HashMap;
use std::path::Path;

use super::manifest_runner::{TestManifest, ElementType, read_values};

/// Result of comparing two output buffers element-by-element.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Total number of elements compared.
    pub total: usize,
    /// Number of elements that match exactly.
    pub matching: usize,
    /// First mismatching element: (index, left_value, right_value).
    pub first_mismatch: Option<(usize, i64, i64)>,
}

impl ComparisonResult {
    /// True when every compared element matched.
    pub fn is_match(&self) -> bool {
        self.matching == self.total && self.total > 0
    }

    /// Match rate as a percentage (0.0 to 100.0).
    pub fn match_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.matching as f64 / self.total as f64) * 100.0
    }

    /// Human-readable summary, e.g. "64/64 (MATCH)" or "60/64 (DIVERGE)".
    pub fn summary(&self) -> String {
        if self.is_match() {
            format!("{}/{} (MATCH)", self.matching, self.total)
        } else {
            format!("{}/{} (DIVERGE)", self.matching, self.total)
        }
    }
}

/// Compare two byte buffers element-by-element using the given type.
///
/// This is the core comparison primitive used by all three validation layers.
pub fn compare_buffers(
    left: &[u8],
    right: &[u8],
    element_type: ElementType,
) -> ComparisonResult {
    let left_vals = read_values(left, element_type);
    let right_vals = read_values(right, element_type);
    compare_value_slices(&left_vals, &right_vals)
}

/// Compare two value slices element-by-element.
pub fn compare_value_slices(left: &[i64], right: &[i64]) -> ComparisonResult {
    let total = left.len().min(right.len());
    let mut matching = 0;
    let mut first_mismatch = None;

    for i in 0..total {
        if left[i] == right[i] {
            matching += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some((i, left[i], right[i]));
        }
    }

    ComparisonResult {
        total,
        matching,
        first_mismatch,
    }
}

/// Layer 2: Validate hardware output against manifest expected transform.
///
/// Generates expected values from the manifest's transform expression
/// and compares them against the raw hardware output bytes.
///
/// Returns `None` if the manifest lacks enough information to generate
/// expected values (missing output buffer def, unknown element type, etc.).
pub fn validate_hw_vs_manifest(
    hw_output: &[u8],
    manifest: &TestManifest,
    input_values: &HashMap<String, Vec<i64>>,
    reference_dir: Option<&Path>,
) -> Option<ComparisonResult> {
    let output_buf = manifest.get_output()?;
    let elem_type = ElementType::from_str(&output_buf.element_type)?;

    // Generate expected values from the manifest transform or reference file
    let expected = manifest.generate_expected(input_values, reference_dir)?;
    let actual = read_values(hw_output, elem_type);

    Some(compare_value_slices(&expected, &actual))
}

/// Layer 3: Compare emulator output against hardware output.
///
/// A direct byte-level comparison (interpreted through the element type)
/// that catches any divergence between the two execution environments.
pub fn compare_emu_vs_hw(
    emu_output: &[u8],
    hw_output: &[u8],
    element_type: ElementType,
) -> ComparisonResult {
    compare_buffers(emu_output, hw_output, element_type)
}

/// Complete three-way comparison result for a single test.
#[derive(Debug)]
pub struct CrossValidation {
    pub test_name: String,
    /// Layer 1: Emulator output vs manifest expected values.
    pub emu_vs_manifest: Option<ComparisonResult>,
    /// Layer 2: Hardware output vs manifest expected values.
    pub hw_vs_manifest: Option<ComparisonResult>,
    /// Layer 3: Emulator output vs hardware output.
    pub emu_vs_hw: Option<ComparisonResult>,
}

impl CrossValidation {
    /// Run all three comparison layers for a test.
    ///
    /// Any layer can be `None` if the required data is unavailable
    /// (e.g. no hardware capture, no manifest, emulator didn't produce output).
    pub fn compare(
        test_name: &str,
        manifest: &TestManifest,
        input_values: &HashMap<String, Vec<i64>>,
        emu_output: Option<&[u8]>,
        hw_output: Option<&[u8]>,
        reference_dir: Option<&Path>,
    ) -> Self {
        let output_buf = manifest.get_output();
        let elem_type = output_buf
            .and_then(|b| ElementType::from_str(&b.element_type));

        // Layer 1: Emu vs Manifest
        let emu_vs_manifest = emu_output.and_then(|emu| {
            validate_hw_vs_manifest(emu, manifest, input_values, reference_dir)
        });

        // Layer 2: HW vs Manifest
        let hw_vs_manifest = hw_output.and_then(|hw| {
            validate_hw_vs_manifest(hw, manifest, input_values, reference_dir)
        });

        // Layer 3: Emu vs HW
        let emu_vs_hw = match (emu_output, hw_output, elem_type) {
            (Some(emu), Some(hw), Some(et)) => Some(compare_emu_vs_hw(emu, hw, et)),
            _ => None,
        };

        CrossValidation {
            test_name: test_name.to_string(),
            emu_vs_manifest,
            hw_vs_manifest,
            emu_vs_hw,
        }
    }
}

/// Format a comparison result column for the report table.
fn format_column(result: &Option<ComparisonResult>) -> String {
    match result {
        Some(r) if r.is_match() => format!("PASS ({}/{})", r.matching, r.total),
        Some(r) => format!("FAIL ({}/{})", r.matching, r.total),
        None => "N/A".to_string(),
    }
}

/// Format the Emu-vs-HW column (uses MATCH/DIVERGE instead of PASS/FAIL).
fn format_emu_hw_column(result: &Option<ComparisonResult>) -> String {
    match result {
        Some(r) => r.summary(),
        None => "N/A".to_string(),
    }
}

/// Generate a three-column comparison report from cross-validation results.
///
/// Output format:
/// ```text
/// === Cross-Validation Report ===
///
/// Test                           | Emu vs Manifest | HW vs Manifest | Emu vs HW
/// -------------------------------|-----------------|----------------|----------
/// add_one_using_dma              | PASS (64/64)    | PASS (64/64)   | MATCH
/// ```
pub fn format_report(results: &[CrossValidation]) -> String {
    let mut report = String::new();

    report.push_str("=== Cross-Validation Report ===\n\n");

    // Header
    report.push_str(&format!(
        "{:<35} | {:<17} | {:<17} | {}\n",
        "Test", "Emu vs Manifest", "HW vs Manifest", "Emu vs HW"
    ));
    report.push_str(&format!(
        "{:-<35}-+-{:-<17}-+-{:-<17}-+-{:-<17}\n",
        "", "", "", ""
    ));

    // Rows
    for cv in results {
        let name = if cv.test_name.len() > 35 {
            &cv.test_name[..35]
        } else {
            &cv.test_name
        };

        report.push_str(&format!(
            "{:<35} | {:<17} | {:<17} | {}\n",
            name,
            format_column(&cv.emu_vs_manifest),
            format_column(&cv.hw_vs_manifest),
            format_emu_hw_column(&cv.emu_vs_hw),
        ));
    }

    // Summary statistics
    let total = results.len();
    let emu_manifest_pass = results.iter()
        .filter(|cv| cv.emu_vs_manifest.as_ref().map_or(false, |r| r.is_match()))
        .count();
    let hw_manifest_pass = results.iter()
        .filter(|cv| cv.hw_vs_manifest.as_ref().map_or(false, |r| r.is_match()))
        .count();
    let emu_hw_match = results.iter()
        .filter(|cv| cv.emu_vs_hw.as_ref().map_or(false, |r| r.is_match()))
        .count();

    let emu_manifest_total = results.iter()
        .filter(|cv| cv.emu_vs_manifest.is_some())
        .count();
    let hw_manifest_total = results.iter()
        .filter(|cv| cv.hw_vs_manifest.is_some())
        .count();
    let emu_hw_total = results.iter()
        .filter(|cv| cv.emu_vs_hw.is_some())
        .count();

    report.push_str(&format!("\n=== Summary ===\n"));
    if emu_manifest_total > 0 {
        report.push_str(&format!(
            "Emulator matches manifest: {}/{} ({:.1}%)\n",
            emu_manifest_pass, emu_manifest_total,
            100.0 * emu_manifest_pass as f64 / emu_manifest_total as f64
        ));
    }
    if hw_manifest_total > 0 {
        report.push_str(&format!(
            "Hardware matches manifest: {}/{} ({:.1}%)\n",
            hw_manifest_pass, hw_manifest_total,
            100.0 * hw_manifest_pass as f64 / hw_manifest_total as f64
        ));
    }
    if emu_hw_total > 0 {
        report.push_str(&format!(
            "Emulator matches hardware: {}/{} ({:.1}%)\n",
            emu_hw_match, emu_hw_total,
            100.0 * emu_hw_match as f64 / emu_hw_total as f64
        ));
    }
    report.push_str(&format!("Total tests: {}\n", total));

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_matching_buffers() {
        let data: Vec<u8> = (1..=4i32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = compare_buffers(&data, &data, ElementType::I32);
        assert!(result.is_match());
        assert_eq!(result.total, 4);
        assert_eq!(result.matching, 4);
        assert!(result.first_mismatch.is_none());
    }

    #[test]
    fn test_compare_diverging_buffers() {
        let left: Vec<u8> = (1..=4i32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let right: Vec<u8> = [1i32, 2, 99, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = compare_buffers(&left, &right, ElementType::I32);
        assert!(!result.is_match());
        assert_eq!(result.total, 4);
        assert_eq!(result.matching, 3);
        assert_eq!(result.first_mismatch, Some((2, 3, 99)));
    }

    #[test]
    fn test_compare_empty_buffers() {
        let result = compare_buffers(&[], &[], ElementType::I32);
        assert!(!result.is_match()); // empty is not a match
        assert_eq!(result.total, 0);
        assert_eq!(result.match_rate(), 0.0);
    }

    #[test]
    fn test_compare_different_lengths() {
        // Comparison uses the shorter length
        let short: Vec<u8> = (1..=2i32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let long: Vec<u8> = (1..=4i32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = compare_buffers(&short, &long, ElementType::I32);
        assert!(result.is_match());
        assert_eq!(result.total, 2);
    }

    #[test]
    fn test_comparison_result_summary() {
        let pass = ComparisonResult {
            total: 64,
            matching: 64,
            first_mismatch: None,
        };
        assert_eq!(pass.summary(), "64/64 (MATCH)");

        let fail = ComparisonResult {
            total: 64,
            matching: 60,
            first_mismatch: Some((3, 4, 99)),
        };
        assert_eq!(fail.summary(), "60/64 (DIVERGE)");
    }

    #[test]
    fn test_match_rate() {
        let result = ComparisonResult {
            total: 100,
            matching: 75,
            first_mismatch: Some((0, 0, 1)),
        };
        assert!((result.match_rate() - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_validate_hw_vs_manifest() {
        let toml_content = r#"
[test]
name = "test"
source_dir = "test"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 4
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 4
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 1"
"#;
        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input_a".to_string(), vec![1, 2, 3, 4]);

        // Hardware produced correct output: [2, 3, 4, 5]
        let hw_output: Vec<u8> = [2i32, 3, 4, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = validate_hw_vs_manifest(&hw_output, &manifest, &inputs, None).unwrap();
        assert!(result.is_match());
        assert_eq!(result.total, 4);
    }

    #[test]
    fn test_cross_validation() {
        let toml_content = r#"
[test]
name = "add_one"
source_dir = "test"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 4
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 4
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 1"
"#;
        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input_a".to_string(), vec![1, 2, 3, 4]);

        // Both produce correct output
        let correct: Vec<u8> = [2i32, 3, 4, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let cv = CrossValidation::compare(
            "add_one",
            &manifest,
            &inputs,
            Some(&correct),
            Some(&correct),
            None,
        );

        assert!(cv.emu_vs_manifest.as_ref().unwrap().is_match());
        assert!(cv.hw_vs_manifest.as_ref().unwrap().is_match());
        assert!(cv.emu_vs_hw.as_ref().unwrap().is_match());
    }

    #[test]
    fn test_cross_validation_emu_wrong() {
        let toml_content = r#"
[test]
name = "add_one"
source_dir = "test"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 4
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 4
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 1"
"#;
        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input_a".to_string(), vec![1, 2, 3, 4]);

        // Hardware correct, emulator wrong
        let hw_correct: Vec<u8> = [2i32, 3, 4, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let emu_wrong: Vec<u8> = [42i32, 43, 44, 45]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let cv = CrossValidation::compare(
            "add_one",
            &manifest,
            &inputs,
            Some(&emu_wrong),
            Some(&hw_correct),
            None,
        );

        // Emu vs manifest: FAIL (emulator produced wrong values)
        assert!(!cv.emu_vs_manifest.as_ref().unwrap().is_match());
        // HW vs manifest: PASS
        assert!(cv.hw_vs_manifest.as_ref().unwrap().is_match());
        // Emu vs HW: DIVERGE
        assert!(!cv.emu_vs_hw.as_ref().unwrap().is_match());
    }

    #[test]
    fn test_format_report() {
        let results = vec![
            CrossValidation {
                test_name: "add_one".to_string(),
                emu_vs_manifest: Some(ComparisonResult {
                    total: 64, matching: 64, first_mismatch: None,
                }),
                hw_vs_manifest: Some(ComparisonResult {
                    total: 64, matching: 64, first_mismatch: None,
                }),
                emu_vs_hw: Some(ComparisonResult {
                    total: 64, matching: 64, first_mismatch: None,
                }),
            },
        ];
        let report = format_report(&results);
        assert!(report.contains("Cross-Validation Report"));
        assert!(report.contains("add_one"));
        assert!(report.contains("PASS (64/64)"));
        assert!(report.contains("64/64 (MATCH)"));
    }
}
