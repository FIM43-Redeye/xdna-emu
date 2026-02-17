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

/// Diagnosis from three-way comparison: emulator vs hardware vs manifest.
///
/// Maps the truth table of three pairwise comparisons to a human-readable
/// root cause category:
///
/// | emu=manifest | hw=manifest | emu=hw | Diagnosis    |
/// |:-------------|:------------|:-------|:-------------|
/// | yes          | yes         | yes    | Correct      |
/// | no           | no          | yes    | CompilerBug  |
/// | no           | yes         | no     | EmulatorBug  |
/// | no           | no          | no     | BothBroken   |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnosis {
    /// All three agree: emulator = hardware = manifest.
    Correct,
    /// Emulator matches hardware, but both disagree with manifest.
    /// This typically indicates a compiler or toolchain bug --
    /// the binary is wrong, but the emulator faithfully reproduces
    /// the hardware behavior.
    CompilerBug,
    /// Hardware matches manifest, but emulator diverges.
    /// This is an emulator bug.
    EmulatorBug,
    /// All three disagree. Both the binary and the emulator are wrong
    /// in different ways.
    BothBroken,
    /// No hardware capture available for this test.
    NoReference,
}

impl Diagnosis {
    /// Short label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Diagnosis::Correct => "CORRECT",
            Diagnosis::CompilerBug => "COMPILER BUG",
            Diagnosis::EmulatorBug => "EMULATOR BUG",
            Diagnosis::BothBroken => "BOTH BROKEN",
            Diagnosis::NoReference => "NO HW REF",
        }
    }
}

impl std::fmt::Display for Diagnosis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Result of hardware cross-validation for a single test.
#[derive(Debug)]
pub struct HardwareValidation {
    pub diagnosis: Diagnosis,
    pub cross: CrossValidation,
}

impl HardwareValidation {
    /// Classify a CrossValidation result into a Diagnosis.
    ///
    /// Requires at least emu_vs_hw to be present; otherwise returns NoReference.
    pub fn classify(cross: CrossValidation) -> Self {
        let diagnosis = match (&cross.emu_vs_manifest, &cross.hw_vs_manifest, &cross.emu_vs_hw) {
            // No hardware data at all
            (_, _, None) => Diagnosis::NoReference,

            // All three layers available
            (Some(em), Some(hm), Some(eh)) => {
                match (em.is_match(), hm.is_match(), eh.is_match()) {
                    (true, true, true) => Diagnosis::Correct,
                    (false, false, true) => Diagnosis::CompilerBug,
                    (false, true, false) => Diagnosis::EmulatorBug,
                    // emu=manifest=true, hw!=manifest -- unusual but possible
                    // (hardware has a bug? Or manifest is wrong in a way that
                    // accidentally matches the emulator). Call it BothBroken
                    // since we cannot cleanly attribute it.
                    _ => Diagnosis::BothBroken,
                }
            }

            // Have emu_vs_hw but missing one of the manifest comparisons.
            // We can still check if emu matches hw.
            (_, _, Some(eh)) => {
                if eh.is_match() {
                    // Emu and HW agree, but we cannot check manifest.
                    // Conservative: report as NoReference since we lack
                    // the full truth table.
                    Diagnosis::NoReference
                } else {
                    Diagnosis::BothBroken
                }
            }
        };

        HardwareValidation { diagnosis, cross }
    }
}

/// Load a hardware reference output for a test from the npu-outputs directory.
///
/// Looks for `{reference_dir}/{test_name}/output.bin`.
pub fn load_hw_reference(reference_dir: &Path, test_name: &str) -> Option<Vec<u8>> {
    let hw_path = reference_dir.join(test_name).join("output.bin");
    if hw_path.exists() {
        std::fs::read(&hw_path).ok()
    } else {
        None
    }
}

/// Generate input values from a manifest's buffer definitions.
///
/// Returns a map of buffer name -> element values, suitable for passing
/// to `CrossValidation::compare()` or `TestManifest::generate_expected()`.
///
/// Shared between `compare_emu_hw.rs` and `xclbin_suite.rs`.
pub fn generate_input_values(manifest: &TestManifest) -> HashMap<String, Vec<i64>> {
    let mut inputs = HashMap::new();

    for (buf_name, buf_def) in &manifest.buffers {
        if buf_name == "output" {
            continue;
        }
        if let Some(elem_type) = ElementType::from_str(&buf_def.element_type) {
            if let Some(data) = manifest.generate_input(buf_name) {
                inputs.insert(buf_name.clone(), read_values(&data, elem_type));
            }
        }
    }

    inputs
}

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

/// Dual-compiler comparison result for a single test.
///
/// Captures emulator and hardware outputs from both the Peano and Chess
/// compilers, enabling fault isolation across the full matrix:
///
/// |               | Emulator      | Hardware      |
/// |:--------------|:--------------|:--------------|
/// | Peano binary  | peano_emu     | peano_hw      |
/// | Chess binary  | chess_emu     | chess_hw      |
///
/// When a test fails, comparing across this matrix reveals the root cause:
/// - Peano-emu fails, Chess-emu passes -> Peano compiler bug
/// - Both-emu fail, both-hw pass -> Emulator bug
/// - Both-emu fail, both-hw fail -> Both compilers produce wrong code
/// - Peano-emu passes, Chess-emu passes -> Correct (double-confirmed)
#[derive(Debug)]
pub struct CompilerComparison {
    pub test_name: String,
    /// Emulator output with Peano-compiled binary.
    pub peano_emu: Option<Vec<u8>>,
    /// Emulator output with Chess-compiled binary.
    pub chess_emu: Option<Vec<u8>>,
    /// Hardware output with Peano-compiled binary.
    pub peano_hw: Option<Vec<u8>>,
    /// Hardware output with Chess-compiled binary.
    pub chess_hw: Option<Vec<u8>>,
    /// Expected output from the manifest.
    pub manifest_expected: Option<Vec<i64>>,
}

/// Dual-compiler diagnosis.
///
/// Extends the single-compiler `Diagnosis` with compiler-specific attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerDiagnosis {
    /// Both compilers produce correct output on the emulator.
    Correct,
    /// Peano binary fails on emulator, Chess binary passes.
    PeanoCompilerBug,
    /// Chess binary fails on emulator, Peano binary passes.
    ChessCompilerBug,
    /// Both compilers produce same wrong output on emulator.
    /// If hardware also fails, it's a toolchain issue. If hardware passes,
    /// it's an emulator bug.
    EmulatorBug,
    /// Both compilers fail differently on the emulator.
    Inconclusive,
    /// Insufficient data to classify (missing Chess or Peano results).
    Incomplete,
}

impl CompilerDiagnosis {
    pub fn label(&self) -> &'static str {
        match self {
            CompilerDiagnosis::Correct => "CORRECT",
            CompilerDiagnosis::PeanoCompilerBug => "PEANO BUG",
            CompilerDiagnosis::ChessCompilerBug => "CHESS BUG",
            CompilerDiagnosis::EmulatorBug => "EMULATOR BUG",
            CompilerDiagnosis::Inconclusive => "INCONCLUSIVE",
            CompilerDiagnosis::Incomplete => "INCOMPLETE",
        }
    }
}

impl std::fmt::Display for CompilerDiagnosis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl CompilerComparison {
    /// Classify the comparison into a diagnosis.
    ///
    /// Requires at least peano_emu and chess_emu to be present.
    pub fn classify(&self, element_type: ElementType) -> CompilerDiagnosis {
        let (peano_emu, chess_emu) = match (&self.peano_emu, &self.chess_emu) {
            (Some(p), Some(c)) => (p, c),
            _ => return CompilerDiagnosis::Incomplete,
        };

        let expected = match &self.manifest_expected {
            Some(e) => e,
            None => return CompilerDiagnosis::Incomplete,
        };

        let peano_vals = read_values(peano_emu, element_type);
        let chess_vals = read_values(chess_emu, element_type);

        let peano_match = compare_value_slices(&peano_vals, expected).is_match();
        let chess_match = compare_value_slices(&chess_vals, expected).is_match();

        match (peano_match, chess_match) {
            (true, true) => CompilerDiagnosis::Correct,
            (false, true) => CompilerDiagnosis::PeanoCompilerBug,
            (true, false) => CompilerDiagnosis::ChessCompilerBug,
            (false, false) => {
                // Both wrong -- are they wrong in the same way?
                let emu_agree = compare_value_slices(&peano_vals, &chess_vals).is_match();
                if emu_agree {
                    CompilerDiagnosis::EmulatorBug
                } else {
                    CompilerDiagnosis::Inconclusive
                }
            }
        }
    }

    /// Classify with hardware data to refine the emulator-only diagnosis.
    ///
    /// Starts from `classify()` then uses `peano_hw` / `chess_hw` output
    /// to confirm or revise the diagnosis:
    ///
    /// - **EmulatorBug + hardware correct** -> confirmed EmulatorBug
    ///   (both compilers wrong on emu, but hardware produces correct output,
    ///   so the binaries are fine -- emulator is the problem).
    /// - **EmulatorBug + hardware also wrong** -> Inconclusive
    ///   (maybe the manifest expected values are wrong, not the emulator).
    /// - **PeanoCompilerBug + Peano HW correct** -> EmulatorBug
    ///   (Peano binary runs correctly on hardware; emulator misexecutes it).
    /// - **ChessCompilerBug + Chess HW correct** -> EmulatorBug
    ///   (same logic, Chess side).
    ///
    /// Falls back to `classify()` when no hardware data is available.
    pub fn classify_full(&self, element_type: ElementType) -> CompilerDiagnosis {
        let emu_diag = self.classify(element_type);

        let expected = match &self.manifest_expected {
            Some(e) => e,
            None => return emu_diag,
        };

        match emu_diag {
            CompilerDiagnosis::EmulatorBug => {
                // Both compilers wrong the same way on emulator.
                // Check if hardware produces correct output.
                let peano_hw_correct = self.peano_hw.as_ref()
                    .map(|hw| compare_value_slices(&read_values(hw, element_type), expected).is_match());
                let chess_hw_correct = self.chess_hw.as_ref()
                    .map(|hw| compare_value_slices(&read_values(hw, element_type), expected).is_match());

                match (peano_hw_correct, chess_hw_correct) {
                    // At least one hardware run confirms correct output
                    (Some(true), _) | (_, Some(true)) => CompilerDiagnosis::EmulatorBug,
                    // Hardware also wrong -- manifest might be the problem
                    (Some(false), _) | (_, Some(false)) => CompilerDiagnosis::Inconclusive,
                    // No hardware data -- keep emulator-only diagnosis
                    (None, None) => emu_diag,
                }
            }
            CompilerDiagnosis::PeanoCompilerBug => {
                // Chess-emu correct, Peano-emu wrong. Check Peano on hardware.
                if let Some(hw) = &self.peano_hw {
                    let hw_correct = compare_value_slices(
                        &read_values(hw, element_type), expected,
                    ).is_match();
                    if hw_correct {
                        // Peano binary is fine on hardware; emulator misexecutes it
                        return CompilerDiagnosis::EmulatorBug;
                    }
                }
                emu_diag
            }
            CompilerDiagnosis::ChessCompilerBug => {
                // Peano-emu correct, Chess-emu wrong. Check Chess on hardware.
                if let Some(hw) = &self.chess_hw {
                    let hw_correct = compare_value_slices(
                        &read_values(hw, element_type), expected,
                    ).is_match();
                    if hw_correct {
                        // Chess binary is fine on hardware; emulator misexecutes it
                        return CompilerDiagnosis::EmulatorBug;
                    }
                }
                emu_diag
            }
            // Correct, Inconclusive, Incomplete -- no refinement possible
            _ => emu_diag,
        }
    }
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

    /// Helper: build a match result.
    fn make_match(n: usize) -> ComparisonResult {
        ComparisonResult { total: n, matching: n, first_mismatch: None }
    }

    /// Helper: build a diverge result.
    fn make_diverge(matching: usize, total: usize) -> ComparisonResult {
        ComparisonResult {
            total,
            matching,
            first_mismatch: Some((matching, 0, 1)),
        }
    }

    #[test]
    fn test_diagnosis_correct() {
        let cv = CrossValidation {
            test_name: "t".to_string(),
            emu_vs_manifest: Some(make_match(4)),
            hw_vs_manifest: Some(make_match(4)),
            emu_vs_hw: Some(make_match(4)),
        };
        let hv = HardwareValidation::classify(cv);
        assert_eq!(hv.diagnosis, Diagnosis::Correct);
    }

    #[test]
    fn test_diagnosis_compiler_bug() {
        // emu and hw agree, but both differ from manifest
        let cv = CrossValidation {
            test_name: "t".to_string(),
            emu_vs_manifest: Some(make_diverge(0, 4)),
            hw_vs_manifest: Some(make_diverge(0, 4)),
            emu_vs_hw: Some(make_match(4)),
        };
        let hv = HardwareValidation::classify(cv);
        assert_eq!(hv.diagnosis, Diagnosis::CompilerBug);
    }

    #[test]
    fn test_diagnosis_emulator_bug() {
        // hw matches manifest, emu diverges
        let cv = CrossValidation {
            test_name: "t".to_string(),
            emu_vs_manifest: Some(make_diverge(0, 4)),
            hw_vs_manifest: Some(make_match(4)),
            emu_vs_hw: Some(make_diverge(0, 4)),
        };
        let hv = HardwareValidation::classify(cv);
        assert_eq!(hv.diagnosis, Diagnosis::EmulatorBug);
    }

    #[test]
    fn test_diagnosis_both_broken() {
        // All three disagree
        let cv = CrossValidation {
            test_name: "t".to_string(),
            emu_vs_manifest: Some(make_diverge(0, 4)),
            hw_vs_manifest: Some(make_diverge(1, 4)),
            emu_vs_hw: Some(make_diverge(2, 4)),
        };
        let hv = HardwareValidation::classify(cv);
        assert_eq!(hv.diagnosis, Diagnosis::BothBroken);
    }

    #[test]
    fn test_diagnosis_no_reference() {
        // No hardware data
        let cv = CrossValidation {
            test_name: "t".to_string(),
            emu_vs_manifest: Some(make_match(4)),
            hw_vs_manifest: None,
            emu_vs_hw: None,
        };
        let hv = HardwareValidation::classify(cv);
        assert_eq!(hv.diagnosis, Diagnosis::NoReference);
    }

    #[test]
    fn test_diagnosis_label() {
        assert_eq!(Diagnosis::Correct.label(), "CORRECT");
        assert_eq!(Diagnosis::CompilerBug.label(), "COMPILER BUG");
        assert_eq!(Diagnosis::EmulatorBug.label(), "EMULATOR BUG");
        assert_eq!(Diagnosis::BothBroken.label(), "BOTH BROKEN");
        assert_eq!(Diagnosis::NoReference.label(), "NO HW REF");
    }

    #[test]
    fn test_load_hw_reference_missing() {
        // Non-existent directory returns None
        let result = load_hw_reference(Path::new("/nonexistent"), "test");
        assert!(result.is_none());
    }

    // -- CompilerComparison tests --

    /// Helper: build i32 bytes from values.
    fn make_i32_bytes(vals: &[i32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_compiler_comparison_correct() {
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(correct.clone()),
            chess_emu: Some(correct),
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify(ElementType::I32), CompilerDiagnosis::Correct);
    }

    #[test]
    fn test_compiler_comparison_peano_bug() {
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong),
            chess_emu: Some(correct),
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify(ElementType::I32), CompilerDiagnosis::PeanoCompilerBug);
    }

    #[test]
    fn test_compiler_comparison_chess_bug() {
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(correct),
            chess_emu: Some(wrong),
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify(ElementType::I32), CompilerDiagnosis::ChessCompilerBug);
    }

    #[test]
    fn test_compiler_comparison_emulator_bug() {
        // Both compilers produce the same wrong output
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong.clone()),
            chess_emu: Some(wrong),
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify(ElementType::I32), CompilerDiagnosis::EmulatorBug);
    }

    #[test]
    fn test_compiler_comparison_inconclusive() {
        // Both wrong but in different ways
        let wrong_a = make_i32_bytes(&[10, 20, 30, 40]);
        let wrong_b = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong_a),
            chess_emu: Some(wrong_b),
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify(ElementType::I32), CompilerDiagnosis::Inconclusive);
    }

    #[test]
    fn test_compiler_comparison_incomplete() {
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(make_i32_bytes(&[2, 3, 4, 5])),
            chess_emu: None, // Missing Chess data
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify(ElementType::I32), CompilerDiagnosis::Incomplete);
    }

    #[test]
    fn test_compiler_diagnosis_labels() {
        assert_eq!(CompilerDiagnosis::Correct.label(), "CORRECT");
        assert_eq!(CompilerDiagnosis::PeanoCompilerBug.label(), "PEANO BUG");
        assert_eq!(CompilerDiagnosis::ChessCompilerBug.label(), "CHESS BUG");
        assert_eq!(CompilerDiagnosis::EmulatorBug.label(), "EMULATOR BUG");
        assert_eq!(CompilerDiagnosis::Inconclusive.label(), "INCONCLUSIVE");
        assert_eq!(CompilerDiagnosis::Incomplete.label(), "INCOMPLETE");
    }

    // -- classify_full() tests --

    #[test]
    fn test_classify_full_emulator_bug_confirmed_by_hw() {
        // Both compilers produce same wrong output on emulator,
        // but Peano hardware produces correct output -> confirmed EmulatorBug.
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong.clone()),
            chess_emu: Some(wrong),
            peano_hw: Some(correct),
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::EmulatorBug);
    }

    #[test]
    fn test_classify_full_emulator_bug_hw_also_wrong() {
        // Both compilers wrong on emulator, hardware also wrong ->
        // Inconclusive (manifest might be the problem).
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let also_wrong = make_i32_bytes(&[50, 50, 50, 50]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong.clone()),
            chess_emu: Some(wrong),
            peano_hw: Some(also_wrong),
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::Inconclusive);
    }

    #[test]
    fn test_classify_full_peano_bug_revised_to_emu_bug() {
        // Peano-emu wrong, Chess-emu correct, but Peano-hw correct ->
        // binary is fine, emulator misexecutes it.
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong),
            chess_emu: Some(correct.clone()),
            peano_hw: Some(correct),
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::EmulatorBug);
    }

    #[test]
    fn test_classify_full_chess_bug_revised_to_emu_bug() {
        // Chess-emu wrong, Peano-emu correct, but Chess-hw correct ->
        // binary is fine, emulator misexecutes it.
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(correct.clone()),
            chess_emu: Some(wrong),
            peano_hw: None,
            chess_hw: Some(correct),
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::EmulatorBug);
    }

    #[test]
    fn test_classify_full_no_hw_data_falls_back() {
        // No hardware data -> classify_full() returns same as classify().
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong.clone()),
            chess_emu: Some(wrong),
            peano_hw: None,
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::EmulatorBug);
        assert_eq!(cc.classify(ElementType::I32), cc.classify_full(ElementType::I32));
    }

    #[test]
    fn test_classify_full_correct_stays_correct() {
        // Both compilers correct on emulator -> Correct, regardless of hw data.
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(correct.clone()),
            chess_emu: Some(correct.clone()),
            peano_hw: Some(correct),
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::Correct);
    }

    #[test]
    fn test_classify_full_peano_bug_hw_also_wrong() {
        // Peano-emu wrong, Chess-emu correct, Peano-hw also wrong ->
        // stays PeanoCompilerBug (hardware confirms the binary is broken).
        let correct = make_i32_bytes(&[2, 3, 4, 5]);
        let wrong = make_i32_bytes(&[99, 99, 99, 99]);
        let cc = CompilerComparison {
            test_name: "t".to_string(),
            peano_emu: Some(wrong.clone()),
            chess_emu: Some(correct),
            peano_hw: Some(wrong),
            chess_hw: None,
            manifest_expected: Some(vec![2, 3, 4, 5]),
        };
        assert_eq!(cc.classify_full(ElementType::I32), CompilerDiagnosis::PeanoCompilerBug);
    }
}
