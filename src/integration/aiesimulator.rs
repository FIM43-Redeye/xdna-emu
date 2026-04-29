//! AMD aiesimulator invocation and output parsing.
//!
//! aiesimulator is AMD's cycle-accurate AIE simulator. It runs on a `.prj`
//! directory produced by aiecc.py (when `--no-aiesim` is NOT passed).
//!
//! This module provides:
//! - `run_simulation()`: invoke aiesimulator with input data and timeout
//! - `read_output_data()`: read raw binary output from simulation results
//!
//! # Prerequisites
//!
//! - aietools must be installed with aiesimulator binary
//! - A valid Xilinx license (aiesimulator requires one)
//! - A `.prj` directory with `sim/` subdirectory (from a Chess build)
//!
//! # Example
//!
//! ```ignore
//! let tools = AieTools::discover(&config)?;
//! let result = run_simulation(&tools, &prj_dir, &input_data, 100_000)?;
//! let output = read_output_data(&result)?;
//! ```

use std::path::{Path, PathBuf};

use super::aietools::AieTools;
use crate::trace::vcd;

/// Check whether the process can reach aiesimulator's prerequisites.
///
/// aiesimulator requires both:
/// 1. Read access to `~/.Xilinx/Xilinx.lic` (Xilinx feature license)
/// 2. Network access for FlexLM license verification
///
/// Sandboxed environments (e.g. Claude Code) block both, causing
/// misleading "license not found" errors. This function detects that
/// condition and returns a clear error message instead.
pub fn check_environment() -> Result<(), String> {
    // Check license file readability.
    let home = std::env::var("HOME").unwrap_or_default();
    let license_path = PathBuf::from(&home).join(".Xilinx/Xilinx.lic");
    if !license_path.exists() {
        return Err(format!(
            "Xilinx license file not found at {}. \
             aiesimulator requires a valid AIEMLsim license.",
            license_path.display()
        ));
    }
    // Try to actually read it (sandbox may block read even if stat succeeds).
    match std::fs::read_to_string(&license_path) {
        Ok(content) => {
            if !content.contains("AIEMLsim") {
                return Err(format!(
                    "License file {} does not contain AIEMLsim feature. \
                     aiesimulator requires this license.",
                    license_path.display()
                ));
            }
        }
        Err(e) => {
            return Err(format!(
                "Cannot read license file {}: {}. \
                 This typically means aiesimulator is being run in a \
                 sandboxed environment that blocks filesystem access. \
                 Run outside the sandbox (aiesimulator also needs network \
                 access for FlexLM license verification).",
                license_path.display(),
                e
            ));
        }
    }

    Ok(())
}

/// Result of an aiesimulator run.
#[derive(Debug)]
pub struct SimResult {
    /// Directory containing simulation output files.
    pub output_dir: PathBuf,
    /// Exit code from aiesimulator process.
    pub exit_code: i32,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
}

/// Result of a unit test simulation (self-contained ps.so test).
#[derive(Debug)]
pub struct UnitSimResult {
    /// Whether stdout contained "PASS!" (the standard success marker).
    pub passed: bool,
    /// Captured stdout from the simulation.
    pub stdout: String,
    /// Captured stderr from the simulation.
    pub stderr: String,
    /// Exit code from aiesimulator.
    pub exit_code: i32,
    /// Wall clock time for the simulation in seconds.
    pub wall_time_secs: f64,
}

/// Run aiesimulator on a .prj directory.
///
/// # Arguments
///
/// * `tools` - Discovered aietools installation (must have aiesimulator)
/// * `prj_dir` - Path to the .prj directory (must contain `sim/` subdirectory)
/// * `input_data` - Named input files: `(filename, raw_bytes)` pairs.
///   These are written into an input directory that aiesimulator reads from.
/// * `timeout_cycles` - Maximum simulation cycles before timeout
///
/// # Returns
///
/// `SimResult` with output directory path, exit code, and captured output.
/// The caller should use `read_output_data()` to extract results.
pub fn run_simulation(
    tools: &AieTools,
    prj_dir: &Path,
    input_data: &[(String, Vec<u8>)],
    timeout_cycles: u64,
) -> Result<SimResult, String> {
    // Validate simulator is available
    if tools.aiesimulator.is_none() {
        return Err("aiesimulator not available in aietools".to_string());
    }

    // Check license and sandbox conditions before wasting time
    check_environment()?;

    // Validate .prj/sim/ exists
    let sim_dir = prj_dir.join("sim");
    if !sim_dir.is_dir() {
        return Err(format!(
            "No sim/ directory in {} -- was the build run without --no-aiesim?",
            prj_dir.display()
        ));
    }

    // Create working directories alongside the .prj
    let work_dir = prj_dir.parent().unwrap_or(Path::new("."));
    let input_dir = work_dir.join("aiesim_input");
    let output_dir = work_dir.join("aiesim_output");

    // Clean and recreate output directory
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).map_err(|e| format!("Failed to clean output dir: {}", e))?;
    }
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("Failed to create output dir: {}", e))?;

    // Write input data files
    std::fs::create_dir_all(&input_dir).map_err(|e| format!("Failed to create input dir: {}", e))?;
    for (name, data) in input_data {
        let path = input_dir.join(name);
        std::fs::write(&path, data).map_err(|e| format!("Failed to write input file {}: {}", name, e))?;
    }

    // Ensure libxaienginecdo.so is available alongside ps.so
    ensure_xaiengine_symlink(&sim_dir);

    // aiesimulator is a complex bash wrapper that sources setupEnv.sh,
    // chess_env_LNa64.sh, and manages its own LD_LIBRARY_PATH/PATH.
    // We invoke via bash -c to ensure proper shell environment setup.
    let aiesim_path = tools.aiesimulator.as_ref().ok_or("aiesimulator not available in aietools")?;

    // Build the shell command string with LD_LIBRARY_PATH prepended
    // (the wrapper script prepends its own dirs, so ours go in front).
    let ld_prefix = xaiengine_ld_prefix();

    let shell_cmd = format!(
        "{}{} --pkg-dir={} --input-dir={} --output-dir={} --dump-vcd --simulation-cycle-timeout={}",
        ld_prefix,
        aiesim_path.display(),
        sim_dir.display(),
        input_dir.display(),
        output_dir.display(),
        timeout_cycles,
    );

    let mut cmd = std::process::Command::new("bash");
    cmd.arg("-c");
    cmd.arg(&shell_cmd);
    cmd.current_dir(work_dir);

    log::info!("Running aiesimulator: pkg-dir={}, timeout={}", sim_dir.display(), timeout_cycles);

    let output = cmd.output().map_err(|e| format!("Failed to execute aiesimulator: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    if !output.status.success() {
        log::warn!(
            "aiesimulator exited with code {}: {}",
            exit_code,
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        );
    }

    // Auto-convert VCD traces to Perfetto JSON
    convert_vcd_traces(work_dir);

    Ok(SimResult { output_dir, exit_code, stdout, stderr })
}

/// Run aiesimulator on a unit test .prj directory.
///
/// Unit tests (from chess_compiler_tests_aie2) are self-contained: the
/// compiled `ps.so` handles all buffer writes, lock management, result
/// checking, and prints "PASS!" to stdout on success. No separate input
/// or output directories are needed.
///
/// This function runs `aiesim.sh` from the .prj directory and checks
/// stdout for the "PASS!" marker.
///
/// # Arguments
///
/// * `tools` - Discovered aietools installation
/// * `prj_dir` - Path to the .prj directory (must contain `aiesim.sh`)
/// * `timeout_cycles` - Maximum simulation cycles before timeout
pub fn run_unit_simulation(
    tools: &AieTools,
    prj_dir: &Path,
    timeout_cycles: u64,
) -> Result<UnitSimResult, String> {
    if tools.aiesimulator.is_none() {
        return Err("aiesimulator not available in aietools".to_string());
    }

    // Check license and sandbox conditions before wasting time
    check_environment()?;

    // Unit tests use aiesim.sh which wraps aiesimulator with the right flags
    let aiesim_sh = prj_dir.join("aiesim.sh");
    if !aiesim_sh.exists() {
        return Err(format!("No aiesim.sh in {} -- was the build run with --aiesim?", prj_dir.display()));
    }

    // Ensure libxaienginecdo.so is available in the ps/ directory
    ensure_xaiengine_symlink(&prj_dir.join("sim"));

    // Build LD_LIBRARY_PATH prefix for the xaiengine library
    let ld_prefix = xaiengine_ld_prefix();

    // Run aiesim.sh with the timeout. The script internally calls
    // aiesimulator with --pkg-dir pointing to sim/.
    let shell_cmd =
        format!("{}bash {} --simulation-cycle-timeout={}", ld_prefix, aiesim_sh.display(), timeout_cycles,);

    let work_dir = prj_dir.parent().unwrap_or(Path::new("."));

    let mut cmd = std::process::Command::new("bash");
    cmd.arg("-c");
    cmd.arg(&shell_cmd);
    cmd.current_dir(work_dir);

    log::info!("Running unit sim: prj={}, timeout={}", prj_dir.display(), timeout_cycles);

    let start = std::time::Instant::now();
    let output = cmd.output().map_err(|e| format!("Failed to execute aiesim.sh: {}", e))?;
    let wall_time_secs = start.elapsed().as_secs_f64();

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);
    let passed = stdout.contains("PASS!");

    if !output.status.success() && !passed {
        log::warn!(
            "Unit sim exited with code {}: {}",
            exit_code,
            stderr.lines().take(5).collect::<Vec<_>>().join("\n")
        );
    }

    // Auto-convert VCD traces to Perfetto JSON
    convert_vcd_traces(work_dir);

    Ok(UnitSimResult { passed, stdout, stderr, exit_code, wall_time_secs })
}

/// Find VCD files in a directory and convert them to Perfetto JSON.
///
/// aiesimulator writes VCD files with `--dump-vcd`. This function finds
/// all `.vcd` files in `dir`, converts each to a `.perfetto.json` file
/// alongside it, and logs the results. Non-fatal: conversion errors are
/// logged but do not fail the simulation.
fn convert_vcd_traces(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "vcd") {
            let json_path = path.with_extension("perfetto.json");
            match convert_single_vcd(&path, &json_path) {
                Ok(result) => {
                    log::info!(
                        "VCD -> Perfetto: {} ({} signals, {} transitions, {} tiles) -> {}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        result.signal_count,
                        result.transition_count,
                        result.tile_count,
                        json_path.file_name().unwrap_or_default().to_string_lossy(),
                    );
                }
                Err(e) => {
                    log::warn!("Failed to convert VCD {}: {}", path.display(), e);
                }
            }
        }
    }
}

/// Convert a single VCD file to Perfetto JSON.
fn convert_single_vcd(vcd_path: &Path, json_path: &Path) -> Result<vcd::VcdConvertResult, String> {
    let vcd_file = std::fs::File::open(vcd_path).map_err(|e| format!("open VCD: {}", e))?;
    let reader = std::io::BufReader::new(vcd_file);

    let json_file = std::fs::File::create(json_path).map_err(|e| format!("create JSON: {}", e))?;
    let mut writer = std::io::BufWriter::new(json_file);

    vcd::vcd_to_perfetto(reader, &mut writer, None).map_err(|e| format!("conversion: {}", e))
}

/// Symlink libxaienginecdo.so into a sim/ps/ directory if not already present.
///
/// ps.so (the simulation host shim) links against libxaienginecdo.so from
/// the mlir-aie install, but the aiesimulator bash wrapper clobbers
/// LD_LIBRARY_PATH. The reliable fix: symlink the library into ps/ so
/// the dynamic linker finds it alongside ps.so.
fn ensure_xaiengine_symlink(sim_dir: &Path) {
    let ps_dir = sim_dir.join("ps");
    if !ps_dir.is_dir() {
        return;
    }

    let config = crate::config::Config::get();
    let mlir_aie = PathBuf::from(config.mlir_aie_path());
    let xaiengine_src_raw = mlir_aie.join("install/runtime_lib/x86_64/xaiengine/lib/libxaienginecdo.so");
    let xaiengine_src = xaiengine_src_raw.canonicalize().unwrap_or(xaiengine_src_raw);
    let xaiengine_link = ps_dir.join("libxaienginecdo.so");

    if xaiengine_src.exists() && !xaiengine_link.exists() {
        #[cfg(unix)]
        {
            if let Err(e) = std::os::unix::fs::symlink(&xaiengine_src, &xaiengine_link) {
                log::warn!("Failed to symlink libxaienginecdo.so: {}", e);
            }
        }
    }
}

/// Build an LD_LIBRARY_PATH prefix string for the xaiengine library.
///
/// Returns a string like `LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH `
/// or an empty string if the library directory does not exist.
fn xaiengine_ld_prefix() -> String {
    let config = crate::config::Config::get();
    let mlir_aie = PathBuf::from(config.mlir_aie_path());
    let xaiengine_lib_raw = mlir_aie.join("install/runtime_lib/x86_64/xaiengine/lib");
    let xaiengine_lib = xaiengine_lib_raw.canonicalize().unwrap_or(xaiengine_lib_raw);

    if xaiengine_lib.is_dir() {
        format!("LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH ", xaiengine_lib.display())
    } else {
        String::new()
    }
}

/// Read raw binary output data from aiesimulator output directory.
///
/// Searches for output files in `{output_dir}/data/` and concatenates
/// them into a single byte buffer. The exact output format depends on
/// the test's ps.so wrapper, so this is best-effort.
///
/// Returns the concatenated bytes, or an error if no output files are found.
pub fn read_output_data(sim_result: &SimResult) -> Result<Vec<u8>, String> {
    // aiesimulator writes output to {output_dir}/data/ or directly
    // to {output_dir}/. Check both locations.
    let data_dir = sim_result.output_dir.join("data");
    let search_dir = if data_dir.is_dir() {
        &data_dir
    } else {
        &sim_result.output_dir
    };

    let mut output = Vec::new();
    let mut found_files = Vec::new();

    let entries = std::fs::read_dir(search_dir)
        .map_err(|e| format!("Failed to read output dir {}: {}", search_dir.display(), e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // Read .bin files (raw binary) and .txt files (text format)
        let ext = path.extension().map(|e| e.to_string_lossy().to_string()).unwrap_or_default();

        match ext.as_str() {
            "bin" => {
                if let Ok(data) = std::fs::read(&path) {
                    found_files.push(path.clone());
                    output.extend_from_slice(&data);
                }
            }
            "txt" => {
                // Try parsing as whitespace-separated integers
                if let Ok(text) = std::fs::read_to_string(&path) {
                    let parsed = parse_text_output(&text);
                    if !parsed.is_empty() {
                        found_files.push(path.clone());
                        output.extend_from_slice(&parsed);
                    }
                }
            }
            _ => {
                // Try reading as raw binary
                if let Ok(data) = std::fs::read(&path) {
                    if !data.is_empty() {
                        found_files.push(path.clone());
                        output.extend_from_slice(&data);
                    }
                }
            }
        }
    }

    if found_files.is_empty() {
        return Err(format!("No output files found in {}", search_dir.display()));
    }

    log::info!(
        "Read {} bytes from {} output file(s): {:?}",
        output.len(),
        found_files.len(),
        found_files
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
            .collect::<Vec<_>>()
    );

    Ok(output)
}

/// Parse text output from aiesimulator (whitespace-separated integers).
///
/// Each integer is converted to a 4-byte little-endian representation.
/// Lines starting with `#` are treated as comments.
fn parse_text_output(text: &str) -> Vec<u8> {
    let mut result = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        for token in line.split_whitespace() {
            // Try parsing as integer (decimal or hex)
            let value = if let Some(hex) = token.strip_prefix("0x").or_else(|| token.strip_prefix("0X")) {
                i64::from_str_radix(hex, 16).ok()
            } else {
                token.parse::<i64>().ok()
            };
            if let Some(v) = value {
                result.extend_from_slice(&(v as i32).to_le_bytes());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_output_decimal() {
        let text = "1 2 3 4\n5 6 7 8\n";
        let bytes = parse_text_output(text);
        // 8 integers * 4 bytes each = 32 bytes
        assert_eq!(bytes.len(), 32);
        // First value should be 1 in little-endian
        assert_eq!(&bytes[0..4], &1i32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2i32.to_le_bytes());
    }

    #[test]
    fn test_parse_text_output_hex() {
        let text = "0x01 0x0A 0xFF\n";
        let bytes = parse_text_output(text);
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[0..4], &1i32.to_le_bytes());
        assert_eq!(&bytes[4..8], &10i32.to_le_bytes());
        assert_eq!(&bytes[8..12], &255i32.to_le_bytes());
    }

    #[test]
    fn test_parse_text_output_comments() {
        let text = "# This is a comment\n1 2\n# Another comment\n3 4\n";
        let bytes = parse_text_output(text);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_parse_text_output_empty() {
        assert!(parse_text_output("").is_empty());
        assert!(parse_text_output("# only comments\n").is_empty());
        assert!(parse_text_output("   \n\n  ").is_empty());
    }

    #[test]
    fn test_parse_text_output_mixed_garbage() {
        // Non-numeric tokens are silently skipped
        let text = "1 hello 3\n";
        let bytes = parse_text_output(text);
        assert_eq!(bytes.len(), 8); // Only 1 and 3 parsed
    }
}
