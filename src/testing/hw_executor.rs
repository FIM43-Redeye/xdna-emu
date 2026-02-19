//! Hardware execution helpers for running xclbins on real NPU silicon.
//!
//! Wraps the npu_runner interface with validation against hardware reference
//! captures and display formatting for the test runner output.

use std::path::Path;
use std::time::Instant;

use crate::testing::test_cpp_parser::{BufferSpec, BufferDir, read_values};
use crate::testing::hardware_comparison::load_hw_reference;
use crate::testing::npu_runner;

use super::runner_stats::HwRunResult;

/// Default timeout for NPU hardware execution (seconds).
/// 30s is generous for any test; prevents infinite hangs from TDR failures.
pub const DEFAULT_HW_TIMEOUT_SEC: u32 = 30;

/// Run an xclbin on real NPU hardware and validate output against reference.
///
/// Returns `Ok((label, raw_output))` on successful execution (even if
/// validation fails), or `Err(message)` if the hardware run itself failed.
pub fn run_on_hw_and_validate(
    spec: &BufferSpec,
    test_name: &str,
    xclbin_path: &Path,
    insts_path: &Path,
    reference_dir: &Path,
) -> Result<(String, Vec<u8>), String> {
    let result = npu_runner::run_on_npu(spec, test_name, xclbin_path, insts_path, DEFAULT_HW_TIMEOUT_SEC)
        .map_err(|e| {
            // Truncate to first line -- stderr from GLIBCXX errors can be 10+ lines
            let msg = e.to_string();
            msg.lines().next().unwrap_or(&msg).to_string()
        })?;

    // Validate output against hardware reference
    let label = (|| -> Option<String> {
        let output_buf = spec.buffers.iter().find(|b| b.direction == BufferDir::Output)?;
        let elem_type = output_buf.element_type;
        let expected_bytes = load_hw_reference(reference_dir, test_name)?;
        let expected = read_values(&expected_bytes, elem_type);

        let actual = read_values(&result.output, elem_type);
        let total = expected.len().min(actual.len());
        let correct = (0..total).filter(|&i| actual[i] == expected[i]).count();

        if correct == total && total > 0 {
            Some(format!("PASS ({}/{})", correct, total))
        } else {
            Some(format!("FAIL ({}/{})", correct, total))
        }
    })().unwrap_or_else(|| "DONE (no validation)".to_string());

    Ok((label, result.output))
}

/// Run an xclbin on real NPU hardware, display the result, and return it.
///
/// Deduplicates the Peano HW / Chess HW run-and-display logic. In compact
/// mode (hw-only), uses `print!` for single-line output. In normal mode,
/// uses `println!` with indentation.
pub fn run_hw_and_print(
    spec: &BufferSpec,
    test_name: &str,
    xclbin: &Path,
    insts: &Path,
    prefix: &str,
    compact: bool,
    reference_dir: &Path,
) -> HwRunResult {
    let hw_start = Instant::now();
    let result = match run_on_hw_and_validate(spec, test_name, xclbin, insts, reference_dir) {
        Ok((label, output)) => {
            let elapsed = hw_start.elapsed().as_secs_f64();
            let passed = label.starts_with("PASS");
            if compact {
                print!("{}: {:18} ({:.1}s)", prefix, label, elapsed);
            } else {
                println!("      {}: {} ({:.1}s)", prefix, label, elapsed);
            }
            HwRunResult { label, output, passed, elapsed_secs: elapsed }
        }
        Err(e) => {
            let elapsed = hw_start.elapsed().as_secs_f64();
            // Extract the meaningful error: look for "ERROR:" lines from
            // npu-runner stderr, falling back to the NpuRunError message.
            let meaningful = e.lines()
                .filter(|l| l.contains("ERROR:") || l.starts_with("Kernel timed out"))
                .last()
                .unwrap_or_else(|| e.lines().next().unwrap_or(&e));
            let label = format!("ERROR ({})", &meaningful[..meaningful.len().min(60)]);
            if compact {
                print!("{}: {}", prefix, label);
            } else {
                println!("      {}: {}", prefix, label);
            }
            HwRunResult { label, output: Vec::new(), passed: false, elapsed_secs: elapsed }
        }
    };

    // Wait for the NPU device to become idle before the next test.
    // Polls sysfs runtime PM instead of fixed-duration sleeps.
    let is_error = result.label.starts_with("ERROR");
    npu_runner::wait_for_device_idle(is_error);

    result
}
