//! Display formatting for the test runner.
//!
//! Contains the `TestResult` struct, result formatting, verbose comparison
//! output, elfanalyzer integration, and aiesimulator comparison display.

use std::path::Path;

use crate::testing::xclbin_suite::{XclbinTest, TestOutcome};
use crate::testing::test_cpp_parser::{BufferDir, read_values, generate_input_data};
use crate::testing::hardware_comparison::{
    Diagnosis, HardwareValidation, load_hw_reference,
};
use crate::integration::aietools::AieTools;
use crate::integration::aiesimulator;
use crate::integration::elfanalyzer;

/// Result from running a single test, ready for display.
pub struct TestResult {
    pub idx: usize,
    pub name: String,
    pub elf_count: usize,
    pub embedded_count: usize,
    pub has_npu: bool,
    pub outcome: TestOutcome,
    pub raw_output: Option<Vec<u8>>,
    /// Hardware cross-validation result (if npu-outputs are available).
    pub hw_validation: Option<HardwareValidation>,
    /// Warnings collected during emulator execution (e.g., DMA queue full).
    pub warnings: Vec<String>,
}

/// Format a test result line for display.
pub fn format_result(r: &TestResult, total: usize) -> String {
    let mut out = String::new();

    // Header: index, name, code sources
    out.push_str(&format!("[{:2}/{}] {:55} ... ", r.idx + 1, total,
        &r.name[..r.name.len().min(55)]));

    if r.elf_count == 0 && r.embedded_count == 0 {
        out.push_str("(no code) ");
    } else if r.elf_count > 0 && r.embedded_count > 0 {
        out.push_str(&format!("({} ELFs, {} CDO) ", r.elf_count, r.embedded_count));
    } else if r.elf_count > 0 {
        out.push_str(&format!("({} ELFs) ", r.elf_count));
    } else {
        out.push_str(&format!("({} CDO) ", r.embedded_count));
    }

    if !r.has_npu {
        out.push_str("(no NPU) ");
    }

    // Outcome
    match &r.outcome {
        TestOutcome::Pass { cycles, correct, total } => {
            if let (Some(c), Some(t)) = (correct, total) {
                out.push_str(&format!("PASS ({} cycles, {}/{} validated)", cycles, c, t));
            } else {
                out.push_str(&format!("PASS ({} cycles, no validation)", cycles));
            }
        }
        TestOutcome::ValidationFail { cycles, correct, total, first_mismatch } => {
            out.push_str(&format!("VALIDATION FAIL ({} cycles, {}/{} correct)", cycles, correct, total));
            if let Some((idx, expected, actual)) = first_mismatch {
                out.push_str(&format!("\n      First mismatch at [{}]: expected {}, got {}", idx, expected, actual));
            }
        }
        TestOutcome::Fail { message, cycles } => {
            out.push_str(&format!("FAIL ({} cycles)\n      {}", cycles, message));
        }
        TestOutcome::UnknownOpcode { details, cycles } => {
            out.push_str(&format!("UNKNOWN ({} cycles)\n      {:?}", cycles, details));
        }
        TestOutcome::Timeout { cycles } => {
            out.push_str(&format!("TIMEOUT ({} cycles)", cycles));
        }
        TestOutcome::LoadError { message } => {
            out.push_str(&format!("LOAD ERROR\n      {}", message));
        }
        TestOutcome::ExpectedFail { cycles, reason, actual } => {
            out.push_str(&format!("EXPECTED FAIL ({} cycles)", cycles));
            if !reason.is_empty() {
                out.push_str(&format!("\n      reason: {}", reason));
            }
            out.push_str(&format!("\n      actual: {}", actual));
        }
        TestOutcome::UnexpectedPass { cycles, correct, total } => {
            out.push_str(&format!("UNEXPECTED PASS ({} cycles, {}/{} correct)", cycles, correct, total));
            out.push_str("\n      Test was expected to fail but passed -- update test_overrides.toml!");
        }
        TestOutcome::Skipped { reason } => {
            out.push_str(&format!("SKIP\n      {}", reason));
        }
        TestOutcome::Platform { required, reason } => {
            out.push_str(&format!("PLATFORM (requires {})", required));
            if !reason.is_empty() {
                out.push_str(&format!("\n      {}", reason));
            }
        }
    }

    // Append hardware cross-validation diagnosis if available
    if let Some(ref hv) = r.hw_validation {
        match hv.diagnosis {
            Diagnosis::NoReference => {} // Omit -- no useful info
            Diagnosis::Correct => {
                out.push_str("\n      hw: CORRECT (emulator matches hardware)");
            }
            Diagnosis::CompilerBug => {
                out.push_str("\n      hw: COMPILER BUG (emulator matches hardware, both wrong)");
            }
            Diagnosis::EmulatorBug => {
                out.push_str("\n      hw: EMULATOR BUG (hardware correct, emulator diverges)");
            }
            Diagnosis::BothBroken => {
                out.push_str("\n      hw: BOTH BROKEN (emulator and hardware both wrong)");
            }
        }
    }

    // Append per-test warnings (DMA queue full, parse errors, etc.)
    if !r.warnings.is_empty() {
        out.push_str(&format!("\n      warnings ({}):", r.warnings.len()));
        for w in &r.warnings {
            out.push_str(&format!("\n        - {}", w));
        }
    }

    out
}

/// Print full expected vs actual output arrays.
///
/// Loads the hardware reference from disk and compares against raw emulator
/// output. Shows element-by-element comparison for debugging.
pub fn print_verbose_comparison(
    raw_output: &[u8],
    test: &XclbinTest,
    reference_dir: &Path,
) {
    let spec = match test.buffer_spec.as_ref() {
        Some(s) => s,
        None => { println!("      (no buffer spec)"); return; }
    };

    let output_buf = match spec.buffers.iter().find(|b| b.direction == BufferDir::Output) {
        Some(b) => b,
        None => { println!("      (no output buffer in spec)"); return; }
    };

    let elem_type = output_buf.element_type;

    // Parse actual output from raw bytes
    let output_size = output_buf.size_elements * elem_type.byte_size();
    let actual_bytes = &raw_output[..output_size.min(raw_output.len())];
    let actual = read_values(actual_bytes, elem_type);

    // Load expected values from hardware reference
    let expected_bytes = match load_hw_reference(reference_dir, &test.name) {
        Some(b) => b,
        None => { println!("      (no hardware reference for '{}')", test.name); return; }
    };
    let expected = read_values(&expected_bytes[..output_size.min(expected_bytes.len())], elem_type);

    let total = expected.len().min(actual.len());
    let correct = (0..total).filter(|&i| actual[i] == expected[i]).count();
    let hex = elem_type.byte_size() >= 4;

    // Print header
    println!("      --- Output Detail ({}/{} correct, {} elements) ---", correct, total, total);

    // Show all elements, marking mismatches
    let show_max = total.min(128); // Cap at 128 for readability
    for i in 0..show_max {
        let marker = if actual[i] == expected[i] { " " } else { "X" };
        if hex {
            println!("      {} [{:4}] expected {:12} (0x{:08X})  got {:12} (0x{:08X})",
                marker, i, expected[i], expected[i] as u32, actual[i], actual[i] as u32);
        } else {
            println!("      {} [{:4}] expected {:6}  got {:6}", marker, i, expected[i], actual[i]);
        }
    }
    if show_max < total {
        let remaining_correct = ((show_max)..total).filter(|&i| actual[i] == expected[i]).count();
        println!("      ... ({} more elements, {} correct)", total - show_max, remaining_correct);
    }
    println!("      --- End Output Detail ---");
}

/// Run elfanalyzer on a test's ELF files and cross-validate against our parser.
pub fn run_elfanalyzer(
    tools: &AieTools,
    test: &XclbinTest,
    _mlir_aie_path: &Path,
) {
    let elf_files = test.find_elf_files();
    if elf_files.is_empty() {
        return;
    }

    for (_col, _row, elf_path) in &elf_files {
        let name = elf_path.file_name()
            .map(|n: &std::ffi::OsStr| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        match elfanalyzer::analyze(tools, elf_path) {
            Ok(analysis) => {
                print!("{}", elfanalyzer::format_analysis(&analysis, &name));

                // Cross-validate: compare elfanalyzer output against our parser
                match std::fs::read(elf_path) {
                    Ok(elf_data) => {
                        match elfanalyzer::cross_validate(&analysis, &elf_data) {
                            Ok(cv) => {
                                print!("{}", elfanalyzer::format_cross_validation(&cv, &name));
                            }
                            Err(e) => {
                                eprintln!("      cross-validation error for {}: {}", name, e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("      failed to read ELF for cross-validation: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("      elfanalyzer error for {}: {}", name, e);
            }
        }
    }
}

/// Run aiesimulator on a Chess-built .prj directory and compare output.
///
/// Returns a status label for display ("PASS (N/N)", "FAIL", "ERROR: ...", etc.).
pub fn run_aiesim_comparison(
    tools: &AieTools,
    test: &XclbinTest,
    prj_dir: &Path,
    reference_dir: &Path,
) -> String {
    let spec = match test.buffer_spec.as_ref() {
        Some(s) => s,
        None => return "SKIP (no buffer spec)".to_string(),
    };

    // Prepare input data from buffer spec
    let mut input_data = Vec::new();
    for buf in &spec.buffers {
        if buf.direction == BufferDir::Output {
            continue;
        }
        let data = generate_input_data(buf);
        input_data.push((format!("{}.bin", buf.name), data));
    }

    // Run simulation with generous timeout (aiesimulator is slow)
    let sim_result = match aiesimulator::run_simulation(
        tools, prj_dir, &input_data, super::runner_config::DEFAULT_MAX_CYCLES,
    ) {
        Ok(r) => r,
        Err(e) => return format!("ERROR: {}", e),
    };

    if sim_result.exit_code != 0 {
        // Truncate stderr for display
        let stderr_preview: String = sim_result.stderr
            .lines()
            .take(3)
            .collect::<Vec<_>>()
            .join(" | ");
        return format!("ERROR (exit {}): {}", sim_result.exit_code, stderr_preview);
    }

    // Read output and compare against hardware reference
    let sim_output = match aiesimulator::read_output_data(&sim_result) {
        Ok(data) => data,
        Err(e) => return format!("ERROR reading output: {}", e),
    };

    let output_buf = match spec.buffers.iter().find(|b| b.direction == BufferDir::Output) {
        Some(b) => b,
        None => return "SKIP (no output buffer)".to_string(),
    };
    let elem_type = output_buf.element_type;

    let expected_bytes = match load_hw_reference(reference_dir, &test.name) {
        Some(b) => b,
        None => return "SKIP (no hw reference)".to_string(),
    };
    let expected = read_values(&expected_bytes, elem_type);

    let actual = read_values(&sim_output, elem_type);
    let total = expected.len().min(actual.len());
    let correct = (0..total).filter(|&i| actual[i] == expected[i]).count();

    if correct == total && total > 0 {
        format!("PASS ({}/{})", correct, total)
    } else {
        format!("FAIL ({}/{})", correct, total)
    }
}
