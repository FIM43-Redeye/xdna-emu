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

/// Run a test on real NPU hardware.
///
/// Dispatches to single-kernel or multi-kernel execution based on the
/// manifest's `multi_kernel` section.  For multi-kernel tests, generates
/// a run spec file and invokes npu-runner in `--spec` mode.
///
/// # Arguments
/// * `manifest` - Test manifest defining inputs, outputs, and transforms
/// * `xclbin_path` - Path to the compiled .xclbin file
/// * `insts_path` - Path to the NPU instruction binary (insts.bin for single-kernel)
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

    if manifest.is_multi_kernel() {
        run_multi_kernel(manifest, xclbin_path, &runner, timeout_sec)
    } else {
        run_single_kernel(manifest, xclbin_path, insts_path, &runner, timeout_sec)
    }
}

/// Run a single-kernel test using the legacy CLI mode.
fn run_single_kernel(
    manifest: &TestManifest,
    xclbin_path: &Path,
    insts_path: &Path,
    runner: &Path,
    timeout_sec: u32,
) -> Result<NpuRunResult, NpuRunError> {
    let tmp_dir = std::env::temp_dir().join(format!("npu_run_{}", manifest.test.name));
    std::fs::create_dir_all(&tmp_dir)?;

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

        input_values.insert(buf_name.clone(), read_values(&input_data, elem_type));

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

    log::info!("Running on NPU (single-kernel): {} {}", runner.display(),
        args.iter().map(|a| {
            if a.contains(' ') { format!("\"{}\"", a) } else { a.clone() }
        }).collect::<Vec<_>>().join(" "));

    let output = Command::new(runner).args(&args).output()?;
    handle_runner_result(output, &output_file, &tmp_dir, input_values)
}

/// Run a multi-kernel test using a spec file.
///
/// Generates a run specification file from the manifest's `multi_kernel`
/// section, then invokes npu-runner with `--spec`.  The spec file describes
/// the kernel invocation order, buffer allocations, and linkages.
fn run_multi_kernel(
    manifest: &TestManifest,
    xclbin_path: &Path,
    runner: &Path,
    timeout_sec: u32,
) -> Result<NpuRunResult, NpuRunError> {
    let mk = manifest.multi_kernel.as_ref().unwrap();
    let build_dir = xclbin_path.parent().unwrap_or(Path::new("."));

    let tmp_dir = std::env::temp_dir().join(format!("npu_run_{}", manifest.test.name));
    std::fs::create_dir_all(&tmp_dir)?;

    let mut input_values = HashMap::new();

    // Generate input files
    let mut input_files: HashMap<String, (PathBuf, usize)> = HashMap::new();
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

        input_values.insert(buf_name.clone(), read_values(&input_data, elem_type));

        let input_file = tmp_dir.join(format!("{}.bin", buf_name));
        let mut f = std::fs::File::create(&input_file)?;
        f.write_all(&input_data)?;

        let size_bytes = buf_def.size * elem_type.byte_size();
        input_files.insert(buf_name.clone(), (input_file, size_bytes));
    }

    // Get output buffer info
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

    // Build the linkage lookup: (to_run, to_gid) -> (from_run, from_gid)
    let mut link_map: HashMap<(usize, u32), (usize, u32)> = HashMap::new();
    for link in &mk.links {
        link_map.insert(
            (link.to_run, link.to_group_id),
            (link.from_run, link.from_group_id),
        );
    }

    // Generate run specification file
    let spec_file = tmp_dir.join("run_spec.txt");
    let mut spec = String::new();
    spec.push_str(&format!("# Multi-kernel spec for {}\n", manifest.test.name));
    spec.push_str(&format!("xclbin {}\n", xclbin_path.to_string_lossy()));
    spec.push_str(&format!("mode {}\n", mk.mode));
    spec.push_str(&format!("timeout {}\n\n", timeout_sec));

    let last_run = mk.runs.len() - 1;

    for (run_idx, kr) in mk.runs.iter().enumerate() {
        let insts_path = build_dir.join(&kr.insts);
        spec.push_str(&format!("run {} {}\n", kr.kernel,
            insts_path.to_string_lossy()));

        // For each buffer group_id referenced by input buffers, check
        // if this run needs it as input, linked, or output.
        //
        // Strategy: iterate all manifest buffers and assign them to the
        // appropriate run based on the linkage map.

        // Input buffers go to the first run (unless overridden by links)
        for (buf_name, buf_def) in &manifest.buffers {
            if buf_name == "output" {
                continue;
            }
            // Check if this buffer is linked to this run from a previous run
            if link_map.contains_key(&(run_idx, buf_def.group_id)) {
                continue; // Will be handled as a link below
            }
            // Input buffers go to run 0 (or whichever run consumes them)
            if run_idx == 0 {
                if let Some((file, size)) = input_files.get(buf_name) {
                    spec.push_str(&format!("in {} {} {}\n",
                        buf_def.group_id,
                        file.to_string_lossy(),
                        size));
                }
            }
        }

        // Links
        for link in &mk.links {
            if link.to_run == run_idx {
                spec.push_str(&format!("link {} {} {}\n",
                    link.to_group_id, link.from_run, link.from_group_id));
            }
        }

        // Output buffer: intermediate (no file) for all runs except the last
        if run_idx == last_run {
            spec.push_str(&format!("out {} {} {}\n",
                output_buf.group_id,
                output_file.to_string_lossy(),
                output_size));
        } else {
            // Check if any links reference this run's output
            for link in &mk.links {
                if link.from_run == run_idx {
                    spec.push_str(&format!("out {} {}\n",
                        link.from_group_id, output_size));
                }
            }
        }
        spec.push('\n');
    }

    std::fs::write(&spec_file, &spec)?;

    log::info!("Running on NPU (multi-kernel): {} --spec {}",
        runner.display(), spec_file.display());
    log::debug!("Spec file contents:\n{}", spec);

    let output = Command::new(runner)
        .args(["--spec", &spec_file.to_string_lossy()])
        .output()?;

    handle_runner_result(output, &output_file, &tmp_dir, input_values)
}

/// Process the result of a npu-runner invocation.
fn handle_runner_result(
    output: std::process::Output,
    output_file: &Path,
    tmp_dir: &Path,
    input_values: HashMap<String, Vec<i64>>,
) -> Result<NpuRunResult, NpuRunError> {
    match output.status.code() {
        Some(0) => {
            let output_data = std::fs::read(output_file)?;
            let _ = std::fs::remove_dir_all(tmp_dir);
            Ok(NpuRunResult {
                output: output_data,
                input_values,
            })
        }
        Some(2) => {
            let _ = std::fs::remove_dir_all(tmp_dir);
            Err(NpuRunError::Timeout)
        }
        code => {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let _ = std::fs::remove_dir_all(tmp_dir);
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

    #[test]
    fn test_multi_kernel_manifest_parsing() {
        let toml_content = r#"
[test]
name = "add_one_two"
source_dir = "test/npu-xrt/add_one_two"

[build]
mlir_file = "aie.mlir"
device = "npu1_1col"

[buffers.input_a]
size = 64
element_type = "i32"
group_id = 3

[buffers.input_a.pattern]
type = "sequential"
start = 1
step = 1

[buffers.output]
size = 64
element_type = "i32"
group_id = 5

[expected]
type = "transform"
transform = "input_a + 3"

[multi_kernel]
mode = "sequential"

[[multi_kernel.runs]]
kernel = "ADDONE"
insts = "insts.bin"

[[multi_kernel.runs]]
kernel = "ADDTWO"
insts = "insts.bin"

[[multi_kernel.links]]
from_run = 0
from_group_id = 5
to_run = 1
to_group_id = 3
"#;
        let manifest: TestManifest = toml::from_str(toml_content).unwrap();
        assert!(manifest.is_multi_kernel());

        let mk = manifest.multi_kernel.as_ref().unwrap();
        assert_eq!(mk.mode, "sequential");
        assert_eq!(mk.runs.len(), 2);
        assert_eq!(mk.runs[0].kernel, "ADDONE");
        assert_eq!(mk.runs[1].kernel, "ADDTWO");
        assert_eq!(mk.links.len(), 1);
        assert_eq!(mk.links[0].from_run, 0);
        assert_eq!(mk.links[0].from_group_id, 5);
        assert_eq!(mk.links[0].to_run, 1);
        assert_eq!(mk.links[0].to_group_id, 3);
    }

    #[test]
    fn test_single_kernel_has_no_multi_kernel() {
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
        assert!(!manifest.is_multi_kernel());
    }
}
