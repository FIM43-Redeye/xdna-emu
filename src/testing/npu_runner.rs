//! Rust orchestration for running tests on real NPU hardware.
//!
//! Drives the C++ `npu-runner` tool, translating BufferSpec-defined tests
//! into CLI invocations.  Buffer metadata is parsed from test.cpp; the
//! C++ side is a thin XRT wrapper.
//!
//! # Architecture
//!
//! ```text
//! test.cpp (parsed by test_cpp_parser)
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
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::process_control::{self, ProcessOutcome};
use super::test_cpp_parser::{BufferSpec, BufferDef, BufferDir, generate_input_data, read_values};

/// Well-known paths where the npu-runner binary might be found.
const RUNNER_SEARCH_PATHS: &[&str] = &["tools/npu-runner/build/npu_runner", "target/npu-runner/npu_runner"];

/// Build a sanitized LD_LIBRARY_PATH for npu-runner.
///
/// aietools ships an ancient libstdc++.so.6 that shadows the system one
/// when its lib directory is on LD_LIBRARY_PATH. npu-runner only needs
/// XRT, not aietools, so we strip aietools paths to avoid the conflict.
use super::sanitized_ld_library_path;

/// NPU accelerator device nodes to probe.
const ACCEL_DEVICE_NODES: &[&str] = &["/dev/accel/accel0", "/dev/accel/accel1"];

/// Check whether NPU hardware is available on this machine.
///
/// Looks for `/dev/accel/accel0` or `/dev/accel/accel1` which are
/// created by the xdna-driver when an AMD NPU is present.
pub fn npu_available() -> bool {
    ACCEL_DEVICE_NODES.iter().any(|p| Path::new(p).exists())
}

/// Probe whether the NPU device appears healthy.
///
/// Checks that the accel device node exists and is stat-able.
/// This is a lightweight check -- it does NOT open the device
/// or allocate any XRT resources. Useful for distinguishing
/// "device node gone" (driver crash) from "XRT can't open it"
/// (device wedged but driver alive).
pub fn probe_device_health() -> bool {
    use std::fs;
    ACCEL_DEVICE_NODES.iter().any(|p| fs::metadata(p).is_ok())
}

/// Check whether the NPU device is idle (suspended or zero active users).
///
/// Returns true if runtime PM shows the device is suspended or has no
/// active users, meaning TDR recovery completed and the device is ready
/// for the next test. Returns true (optimistic) if sysfs is unavailable.
pub fn device_is_idle() -> bool {
    let Some(power_dir) = npu_power_sysfs() else {
        return true; // no sysfs, assume OK
    };

    // Check runtime_status first (most reliable after TDR)
    let status_path = power_dir.join("runtime_status");
    if let Ok(status) = std::fs::read_to_string(&status_path) {
        if status.trim() == "suspended" {
            return true;
        }
    }

    // Fall back to runtime_usage (0 = no active fd holders)
    let usage_path = power_dir.join("runtime_usage");
    if let Ok(usage) = std::fs::read_to_string(&usage_path) {
        if usage.trim() == "0" {
            return true;
        }
    }

    false
}

// ── Adaptive device readiness polling ──────────────────────────────
//
// Instead of fixed-duration sleeps between test runs, we poll the
// Linux runtime PM sysfs entries that the kernel maintains for every
// PCI device. These reads are free (no device wake, no side effects):
//
//   runtime_usage   - refcount of active users (0 = nobody has the fd open)
//   runtime_status  - power state lifecycle (active / suspending / suspended)
//
// Path: /sys/class/accel/<node>/device/power/

/// Cached sysfs power directory for the NPU device.
/// Discovered once on first access, reused for all subsequent calls.
static NPU_POWER_SYSFS: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Maximum time to wait for device suspension after an error (seconds).
const DEVICE_IDLE_TIMEOUT_SECS: u64 = 10;

/// Poll interval when waiting for device suspension (ms).
const DEVICE_IDLE_POLL_MS: u64 = 100;

/// Fixed fallback cooldown after success when sysfs is unavailable (ms).
const FALLBACK_COOLDOWN_MS: u64 = 200;

/// Fixed fallback cooldown after error when sysfs is unavailable (ms).
const FALLBACK_ERROR_COOLDOWN_MS: u64 = 2000;

/// Discover the sysfs power directory for the NPU device.
///
/// Follows `/sys/class/accel/<node>/device/power/` which is a stable
/// path through the sysfs symlink chain to the PCI device's runtime PM.
fn npu_power_sysfs() -> Option<&'static Path> {
    NPU_POWER_SYSFS
        .get_or_init(|| {
            for node in ACCEL_DEVICE_NODES {
                if let Some(dev_name) = Path::new(node).file_name() {
                    let power_dir = PathBuf::from("/sys/class/accel").join(dev_name).join("device/power");
                    if power_dir.exists() {
                        return Some(power_dir);
                    }
                }
            }
            None
        })
        .as_deref()
}

/// Wait for the NPU device to become idle after a test run.
///
/// Uses Linux runtime PM sysfs to adaptively wait for device readiness
/// instead of fixed-duration sleeps:
///
/// - **After success**: reads `runtime_usage` once. If 0, the device has
///   no active users and we proceed immediately. Since npu-runner has
///   already exited, this is typically instant.
///
/// - **After error/timeout**: polls `runtime_status` for `"suspended"`,
///   meaning the device has fully powered down and recovered from TDR.
///   Polls every 100ms, gives up after 10s.
///
/// Falls back to fixed cooldowns if sysfs is unavailable (different
/// kernel, container, etc.).
pub fn wait_for_device_idle(after_error: bool) {
    let Some(power_dir) = npu_power_sysfs() else {
        // Sysfs not available -- fall back to fixed cooldown.
        let ms = if after_error {
            FALLBACK_ERROR_COOLDOWN_MS
        } else {
            FALLBACK_COOLDOWN_MS
        };
        std::thread::sleep(Duration::from_millis(ms));
        return;
    };

    if after_error {
        // After error/TDR: wait for full device suspension (recovery complete).
        // The driver resets the device during TDR, then runtime PM eventually
        // suspends it once all contexts are torn down.
        let status_path = power_dir.join("runtime_status");
        let deadline = Instant::now() + Duration::from_secs(DEVICE_IDLE_TIMEOUT_SECS);

        loop {
            if let Ok(status) = std::fs::read_to_string(&status_path) {
                if status.trim() == "suspended" {
                    return;
                }
            }
            if Instant::now() >= deadline {
                // Timed out -- proceed anyway, cascade detection will catch
                // persistent problems.
                return;
            }
            std::thread::sleep(Duration::from_millis(DEVICE_IDLE_POLL_MS));
        }
    } else {
        // After success: verify runtime_usage is 0 (no active users).
        // npu-runner has already exited so this should be immediate.
        let usage_path = power_dir.join("runtime_usage");
        if let Ok(usage) = std::fs::read_to_string(&usage_path) {
            if usage.trim() == "0" {
                return;
            }
        }
        // Usage not 0 or unreadable -- brief fallback.
        std::thread::sleep(Duration::from_millis(FALLBACK_COOLDOWN_MS));
    }
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

    // Try relative to the executable's location (works for release binaries
    // where CARGO_MANIFEST_DIR is not set). The binary is typically at
    // target/release/npu-test or target/debug/npu-test, so the project root
    // is two directories up.
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // exe_dir = target/release/ or target/debug/
            if let Some(project_root) = exe_dir.parent().and_then(|p| p.parent()) {
                for rel_path in RUNNER_SEARCH_PATHS {
                    let path = project_root.join(rel_path);
                    if path.exists() {
                        return Some(path);
                    }
                }
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
    /// Failed to generate input data from buffer spec.
    InputGeneration(String),
    /// npu-runner process failed.
    ExecutionFailed { exit_code: Option<i32>, stderr: String },
    /// Kernel timed out on hardware.
    Timeout,
    /// Process survived SIGKILL -- device is in D-state (uninterruptible sleep).
    /// The NPU is wedged and no further hardware tests should run.
    DeviceWedged { pid: u32 },
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
            NpuRunError::DeviceWedged { pid } => write!(f, "NPU device wedged: process {} survived SIGKILL (D-state)", pid),
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
    /// Raw output bytes from the primary output buffer.
    pub output: Vec<u8>,
    /// Additional output buffers beyond the primary (e.g. trace data).
    /// Keyed by buffer name from the BufferSpec.
    pub extra_outputs: HashMap<String, Vec<u8>>,
    /// Input values that were generated and sent (for comparison).
    pub input_values: HashMap<String, Vec<i64>>,
}

/// Run a test on real NPU hardware.
///
/// Uses buffer metadata from the parsed test.cpp (BufferSpec) to set up
/// input/output buffers and invoke npu-runner.  Multi-kernel tests are
/// not yet supported via BufferSpec; they return `InputGeneration` error.
///
/// # Arguments
/// * `spec` - Buffer specification parsed from test.cpp
/// * `test_name` - Test name (used for temp directory naming)
/// * `xclbin_path` - Path to the compiled .xclbin file
/// * `insts_path` - Path to the NPU instruction binary (insts.bin)
/// * `timeout_sec` - Kernel execution timeout in seconds
pub fn run_on_npu(
    spec: &BufferSpec,
    test_name: &str,
    xclbin_path: &Path,
    insts_path: &Path,
    timeout_sec: u32,
) -> Result<NpuRunResult, NpuRunError> {
    if !npu_available() {
        return Err(NpuRunError::NoHardware);
    }

    let runner = runner_binary().ok_or(NpuRunError::NoBinary)?;

    if spec.multi_kernel {
        return Err(NpuRunError::InputGeneration(
            "Multi-kernel tests not yet supported via BufferSpec".to_string(),
        ));
    }

    run_single_kernel(spec, test_name, xclbin_path, insts_path, &runner, timeout_sec)
}

/// Run a single-kernel test using the CLI mode.
fn run_single_kernel(
    spec: &BufferSpec,
    test_name: &str,
    xclbin_path: &Path,
    insts_path: &Path,
    runner: &Path,
    timeout_sec: u32,
) -> Result<NpuRunResult, NpuRunError> {
    let tmp_dir = std::env::temp_dir().join(format!("npu_run_{}", test_name));
    std::fs::create_dir_all(&tmp_dir)?;

    let mut args: Vec<String> = vec![
        xclbin_path.to_string_lossy().to_string(),
        insts_path.to_string_lossy().to_string(),
        "--timeout".to_string(),
        timeout_sec.to_string(),
    ];

    let mut input_values = HashMap::new();

    // Generate and write input buffers
    for buf in &spec.buffers {
        if buf.direction != BufferDir::Input {
            continue;
        }

        let input_data = generate_input_data(buf);
        input_values.insert(buf.name.clone(), read_values(&input_data, buf.element_type));

        let input_file = tmp_dir.join(format!("{}.bin", buf.name));
        let mut f = std::fs::File::create(&input_file)?;
        f.write_all(&input_data)?;

        let size_bytes = buf.size_elements * buf.element_type.byte_size();

        args.extend_from_slice(&[
            "--in".to_string(),
            buf.group_id.to_string(),
            input_file.to_string_lossy().to_string(),
            size_bytes.to_string(),
        ]);
    }

    // Set up all output buffers (first is primary, rest are extra).
    let output_bufs: Vec<&BufferDef> =
        spec.buffers.iter().filter(|b| b.direction == BufferDir::Output).collect();
    if output_bufs.is_empty() {
        return Err(NpuRunError::InputGeneration("No output buffer defined in buffer spec".to_string()));
    }

    let mut output_files: Vec<(String, PathBuf)> = Vec::new();
    for (i, buf) in output_bufs.iter().enumerate() {
        let size_bytes = buf.size_elements * buf.element_type.byte_size();
        let filename = if i == 0 {
            "output.bin".to_string()
        } else {
            format!("output_{}.bin", buf.name)
        };
        let file_path = tmp_dir.join(&filename);
        args.extend_from_slice(&[
            "--out".to_string(),
            buf.group_id.to_string(),
            file_path.to_string_lossy().to_string(),
            size_bytes.to_string(),
        ]);
        output_files.push((buf.name.clone(), file_path));
    }

    log::info!(
        "Running on NPU (single-kernel): {} {}",
        runner.display(),
        args.iter()
            .map(|a| {
                if a.contains(' ') {
                    format!("\"{}\"", a)
                } else {
                    a.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    );

    let mut cmd = Command::new(runner);
    cmd.args(&args).env("LD_LIBRARY_PATH", sanitized_ld_library_path());

    match process_control::spawn_with_timeout(&mut cmd, timeout_sec) {
        ProcessOutcome::Completed { exit_code, stderr, .. } => {
            handle_runner_exit(exit_code, &stderr, &output_files, &tmp_dir, input_values)
        }
        ProcessOutcome::Timeout { .. } => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(NpuRunError::Timeout)
        }
        ProcessOutcome::Wedged { pid, .. } => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(NpuRunError::DeviceWedged { pid })
        }
        ProcessOutcome::SpawnError(msg) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(NpuRunError::Io(std::io::Error::new(std::io::ErrorKind::Other, msg)))
        }
    }
}

/// Find the first output buffer in a buffer list.
#[cfg(test)]
fn find_output_buffer(buffers: &[BufferDef]) -> Option<&BufferDef> {
    buffers.iter().find(|b| b.direction == BufferDir::Output)
}

/// Process the exit status of a npu-runner invocation.
///
/// Exit codes: 0 = success, 2 = kernel timeout (C++ side), anything else = error.
/// `output_files` is a list of (buffer_name, file_path) pairs. The first is
/// the primary output buffer; any others are extra (e.g. trace data).
fn handle_runner_exit(
    exit_code: i32,
    stderr: &str,
    output_files: &[(String, PathBuf)],
    tmp_dir: &Path,
    input_values: HashMap<String, Vec<i64>>,
) -> Result<NpuRunResult, NpuRunError> {
    match exit_code {
        0 => {
            let output_data = std::fs::read(&output_files[0].1)?;
            let mut extra_outputs = HashMap::new();
            for (name, path) in &output_files[1..] {
                if let Ok(data) = std::fs::read(path) {
                    extra_outputs.insert(name.clone(), data);
                }
            }
            let _ = std::fs::remove_dir_all(tmp_dir);
            Ok(NpuRunResult { output: output_data, extra_outputs, input_values })
        }
        2 => {
            let _ = std::fs::remove_dir_all(tmp_dir);
            Err(NpuRunError::Timeout)
        }
        code => {
            let _ = std::fs::remove_dir_all(tmp_dir);
            Err(NpuRunError::ExecutionFailed { exit_code: Some(code), stderr: stderr.to_string() })
        }
    }
}

/// Generate input values from a BufferSpec (without running on hardware).
///
/// Useful for comparison when you already have hardware output captured
/// on disk and just need the input values to compute expected output.
pub fn generate_inputs(spec: &BufferSpec) -> HashMap<String, Vec<i64>> {
    let mut inputs = HashMap::new();

    for buf in &spec.buffers {
        if buf.direction != BufferDir::Input {
            continue;
        }
        let data = generate_input_data(buf);
        inputs.insert(buf.name.clone(), read_values(&data, buf.element_type));
    }

    inputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_cpp_parser::{InputPattern, ElementType};

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

    /// Helper to build a BufferDef for tests.
    fn make_input(name: &str, group_id: u32, size: usize, pattern: InputPattern) -> BufferDef {
        BufferDef {
            name: name.to_string(),
            group_id,
            size_elements: size,
            element_type: ElementType::I32,
            direction: BufferDir::Input,
            input_pattern: pattern,
        }
    }

    fn make_output(name: &str, group_id: u32, size: usize) -> BufferDef {
        BufferDef {
            name: name.to_string(),
            group_id,
            size_elements: size,
            element_type: ElementType::I32,
            direction: BufferDir::Output,
            input_pattern: InputPattern::Zeros,
        }
    }

    #[test]
    fn test_generate_inputs_single() {
        let spec = BufferSpec {
            buffers: vec![
                make_input("bo_inA", 3, 4, InputPattern::Sequential { start: 1, step: 1 }),
                make_output("bo_out", 5, 4),
            ],
            multi_kernel: false,
        };

        let inputs = generate_inputs(&spec);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs["bo_inA"], vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_generate_inputs_two_buffers() {
        let spec = BufferSpec {
            buffers: vec![
                make_input("bo_inA", 3, 4, InputPattern::Sequential { start: 1, step: 1 }),
                make_input("bo_inB", 4, 4, InputPattern::Sequential { start: 10, step: 10 }),
                make_output("bo_out", 5, 4),
            ],
            multi_kernel: false,
        };

        let inputs = generate_inputs(&spec);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs["bo_inA"], vec![1, 2, 3, 4]);
        assert_eq!(inputs["bo_inB"], vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_npu_run_error_display() {
        let err = NpuRunError::NoHardware;
        assert_eq!(err.to_string(), "NPU hardware not available");

        let err = NpuRunError::Timeout;
        assert_eq!(err.to_string(), "Kernel timed out on NPU hardware");
    }

    #[test]
    fn test_multi_kernel_spec_returns_error() {
        let spec = BufferSpec {
            buffers: vec![
                make_input("bo_inA", 3, 4, InputPattern::Sequential { start: 1, step: 1 }),
                make_output("bo_out", 5, 4),
            ],
            multi_kernel: true,
        };

        let result = run_on_npu(&spec, "test", Path::new("a.xclbin"), Path::new("insts.bin"), 30);
        // Will fail with NoHardware or NoBinary before reaching the multi-kernel check
        // in CI, but on a machine with hardware it would hit InputGeneration.
        assert!(result.is_err());
    }

    #[test]
    fn test_find_output_buffer() {
        let buffers = vec![
            make_input("bo_inA", 3, 4, InputPattern::Sequential { start: 1, step: 1 }),
            make_output("bo_out", 5, 4),
        ];
        let out = find_output_buffer(&buffers).unwrap();
        assert_eq!(out.name, "bo_out");
        assert_eq!(out.group_id, 5);
    }

    #[test]
    fn test_find_output_buffer_none() {
        let buffers = vec![make_input("bo_inA", 3, 4, InputPattern::Sequential { start: 1, step: 1 })];
        assert!(find_output_buffer(&buffers).is_none());
    }
}
