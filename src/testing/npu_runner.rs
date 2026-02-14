//! Rust orchestration for running tests on real NPU hardware.
//!
//! Drives the C++ `npu-runner` tool, translating manifest-defined tests
//! into CLI invocations.  All TOML parsing and data generation stays in
//! Rust; the C++ side is a thin XRT wrapper.
//!
//! # Architecture
//!
//! ```text
//! Manifest (TOML)
//!   |
//!   v
//! npu_runner.rs          -- generates input files, invokes npu-runner
//!   |
//!   v
//! tools/npu-runner       -- C++ binary, talks to XRT
//!   |
//!   v
//! /dev/accel/accel0      -- real NPU hardware
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::manifest_runner::{TestManifest, ElementType, read_values};

/// Well-known paths where the npu-runner binary might be found.
const RUNNER_SEARCH_PATHS: &[&str] = &[
    "tools/npu-runner/build/npu_runner",
    "target/npu-runner/npu_runner",
];

/// NPU accelerator device nodes to probe.
const ACCEL_DEVICE_NODES: &[&str] = &[
    "/dev/accel/accel0",
    "/dev/accel/accel1",
];

/// Check whether NPU hardware is available on this machine.
///
/// Looks for `/dev/accel/accel0` or `/dev/accel/accel1` which are
/// created by the xdna-driver when an AMD NPU is present.
pub fn npu_available() -> bool {
    ACCEL_DEVICE_NODES.iter().any(|p| Path::new(p).exists())
}

/// Find the npu-runner binary, searching known build locations.
///
/// Returns `None` if the binary has not been built yet.
pub fn runner_binary() -> Option<PathBuf> {
    // Try relative to CARGO_MANIFEST_DIR (works during cargo run/test)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        for rel_path in RUNNER_SEARCH_PATHS {
            let path = PathBuf::from(&manifest_dir).join(rel_path);
            if path.exists() {
                return Some(path);
            }
        }
    }

    // Try relative to current working directory
    for rel_path in RUNNER_SEARCH_PATHS {
        let path = PathBuf::from(rel_path);
        if path.exists() {
            return Some(std::fs::canonicalize(path).unwrap_or_default());
        }
    }

    // Try PATH
    if let Ok(output) = Command::new("which").arg("npu_runner").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
    }

    None
}

/// Error type for NPU runner operations.
#[derive(Debug)]
pub enum NpuRunError {
    /// NPU hardware not available on this machine.
    NoHardware,
    /// npu-runner binary not found (not built).
    NoBinary,
    /// Failed to generate input data from manifest.
    InputGeneration(String),
    /// npu-runner process failed.
    ExecutionFailed {
        exit_code: Option<i32>,
        stderr: String,
    },
    /// Kernel timed out on hardware.
    Timeout,
    /// I/O error (file read/write).
    Io(std::io::Error),
}

impl std::fmt::Display for NpuRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NpuRunError::NoHardware => write!(f, "NPU hardware not available"),
            NpuRunError::NoBinary => write!(f, "npu-runner binary not found (build with: cd tools/npu-runner && cmake -B build && cmake --build build)"),
            NpuRunError::InputGeneration(msg) => write!(f, "Failed to generate input: {}", msg),
            NpuRunError::ExecutionFailed { exit_code, stderr } => {
                write!(f, "npu-runner failed (exit code {:?}): {}", exit_code, stderr)
            }
            NpuRunError::Timeout => write!(f, "Kernel timed out on NPU hardware"),
            NpuRunError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl From<std::io::Error> for NpuRunError {
    fn from(e: std::io::Error) -> Self {
        NpuRunError::Io(e)
    }
}

/// Result of running a test on real NPU hardware.
pub struct NpuRunResult {
    /// Raw output bytes from the NPU.
    pub output: Vec<u8>,
    /// Input values that were generated and sent (for comparison).
    pub input_values: HashMap<String, Vec<i64>>,
}

/// Run a single test on real NPU hardware.
///
/// 1. Generates input data from the manifest pattern
/// 2. Writes input files to a temporary directory
/// 3. Invokes the npu-runner C++ binary
/// 4. Reads the output file and returns raw bytes
///
/// # Arguments
/// * `manifest` - Test manifest defining inputs, outputs, and transforms
/// * `xclbin_path` - Path to the compiled .xclbin file
/// * `insts_path` - Path to the NPU instruction binary (insts.bin)
/// * `timeout_sec` - Kernel execution timeout in seconds
pub fn run_on_npu(
    manifest: &TestManifest,
    xclbin_path: &Path,
    insts_path: &Path,
    timeout_sec: u32,
) -> Result<NpuRunResult, NpuRunError> {
    if !npu_available() {
        return Err(NpuRunError::NoHardware);
    }

    let runner = runner_binary().ok_or(NpuRunError::NoBinary)?;

    // Create a temporary directory for input/output files
    let tmp_dir = std::env::temp_dir().join(format!("npu_run_{}", manifest.test.name));
    std::fs::create_dir_all(&tmp_dir)?;

    // Build CLI arguments and generate input files
    let mut args: Vec<String> = vec![
        xclbin_path.to_string_lossy().to_string(),
        insts_path.to_string_lossy().to_string(),
        "--timeout".to_string(),
        timeout_sec.to_string(),
    ];

    let mut input_values = HashMap::new();

    // Generate and write input buffers
    for (buf_name, buf_def) in &manifest.buffers {
        if buf_name == "output" {
            continue;
        }

        let elem_type = ElementType::from_str(&buf_def.element_type)
            .ok_or_else(|| NpuRunError::InputGeneration(
                format!("Unknown element type: {}", buf_def.element_type),
            ))?;

        let input_data = manifest.generate_input(buf_name)
            .ok_or_else(|| NpuRunError::InputGeneration(
                format!("Failed to generate input for buffer '{}'", buf_name),
            ))?;

        // Save input values for later comparison
        input_values.insert(buf_name.clone(), read_values(&input_data, elem_type));

        // Write input data to temp file
        let input_file = tmp_dir.join(format!("{}.bin", buf_name));
        let mut f = std::fs::File::create(&input_file)?;
        f.write_all(&input_data)?;

        let size_bytes = buf_def.size * elem_type.byte_size();

        args.extend_from_slice(&[
            "--in".to_string(),
            buf_def.group_id.to_string(),
            input_file.to_string_lossy().to_string(),
            size_bytes.to_string(),
        ]);
    }

    // Set up output buffer
    let output_buf = manifest.get_output()
        .ok_or_else(|| NpuRunError::InputGeneration(
            "No output buffer defined in manifest".to_string(),
        ))?;
    let output_elem_type = ElementType::from_str(&output_buf.element_type)
        .ok_or_else(|| NpuRunError::InputGeneration(
            format!("Unknown output element type: {}", output_buf.element_type),
        ))?;

    let output_size = output_buf.size * output_elem_type.byte_size();
    let output_file = tmp_dir.join("output.bin");

    args.extend_from_slice(&[
        "--out".to_string(),
        output_buf.group_id.to_string(),
        output_file.to_string_lossy().to_string(),
        output_size.to_string(),
    ]);

    // Invoke npu-runner
    log::info!("Running on NPU: {} {}", runner.display(),
        args.iter().map(|a| {
            if a.contains(' ') { format!("\"{}\"", a) } else { a.clone() }
        }).collect::<Vec<_>>().join(" "));

    let output = Command::new(&runner)
        .args(&args)
        .output()?;

    // Check result
    match output.status.code() {
        Some(0) => {
            // Success -- read output file
            let output_data = std::fs::read(&output_file)?;

            // Clean up temp directory (best-effort)
            let _ = std::fs::remove_dir_all(&tmp_dir);

            Ok(NpuRunResult {
                output: output_data,
                input_values,
            })
        }
        Some(2) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(NpuRunError::Timeout)
        }
        code => {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(NpuRunError::ExecutionFailed {
                exit_code: code,
                stderr,
            })
        }
    }
}

/// Generate input values from a manifest (without running on hardware).
///
/// Useful for comparison when you already have hardware output captured
/// on disk and just need the input values to compute expected output.
pub fn generate_inputs(manifest: &TestManifest) -> Option<HashMap<String, Vec<i64>>> {
    let mut inputs = HashMap::new();

    for (buf_name, buf_def) in &manifest.buffers {
        if buf_name == "output" {
            continue;
        }
        let elem_type = ElementType::from_str(&buf_def.element_type)?;
        let data = manifest.generate_input(buf_name)?;
        inputs.insert(buf_name.clone(), read_values(&data, elem_type));
    }

    Some(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npu_device_detection() {
        // This test just verifies the function runs without panicking.
        // It will return true or false depending on the machine.
        let _available = npu_available();
    }

    #[test]
    fn test_runner_binary_search() {
        // This test verifies the search logic runs without panicking.
        // It may or may not find the binary depending on build state.
        let _binary = runner_binary();
    }

    #[test]
    fn test_generate_inputs_single() {
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
        let inputs = generate_inputs(&manifest).unwrap();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs["input_a"], vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_generate_inputs_two_buffers() {
        let toml_content = r#"
[test]
name = "vec_add"
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

[buffers.input_b]
size = 4
element_type = "i32"
group_id = 4

[buffers.input_b.pattern]
type = "sequential"
start = 10
step = 10

[buffers.output]
size = 4
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + input_b"
"#;
        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        let inputs = generate_inputs(&manifest).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs["input_a"], vec![1, 2, 3, 4]);
        assert_eq!(inputs["input_b"], vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_npu_run_error_display() {
        let err = NpuRunError::NoHardware;
        assert_eq!(err.to_string(), "NPU hardware not available");

        let err = NpuRunError::Timeout;
        assert_eq!(err.to_string(), "Kernel timed out on NPU hardware");
    }
}
