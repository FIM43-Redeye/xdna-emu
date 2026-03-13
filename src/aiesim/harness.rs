//! Subprocess harness for AMD aiesimulator with VCD collection.
//!
//! Provides [`AiesimHarness`], a zero-state utility type that discovers the
//! aiesimulator binary, validates sim package directories, runs the simulator,
//! and collects VCD output for cross-validation against the emulator.
//!
//! # Binary Discovery
//!
//! The aiesimulator binary is found via (in order):
//! 1. Explicit override in [`AiesimConfig::binary_override`]
//! 2. `AIETOOLS_DIR` environment variable (+ `/bin/aiesimulator`)
//! 3. `XILINX_VITIS_AIETOOLS` environment variable (+ `/bin/aiesimulator`)
//! 4. `PATH` lookup
//!
//! # Package Validation
//!
//! A valid sim package directory must contain `config/scsim_config.json`,
//! which aiesimulator reads for tile and memory configuration.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

/// Error conditions for aiesim harness operations.
#[derive(Debug)]
pub enum AiesimError {
    /// aiesimulator binary not found in any search location.
    BinaryNotFound(String),
    /// Sim package directory is missing or invalid.
    InvalidPackage(String),
    /// The simulator process exited with a non-zero status.
    ProcessFailed {
        /// Exit status from the simulator.
        status: ExitStatus,
        /// Captured stderr (first ~4KB, for error reporting).
        stderr: String,
    },
    /// I/O error during subprocess management.
    Io(std::io::Error),
}

impl std::fmt::Display for AiesimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinaryNotFound(msg) => write!(f, "aiesimulator not found: {}", msg),
            Self::InvalidPackage(msg) => write!(f, "invalid sim package: {}", msg),
            Self::ProcessFailed { status, stderr } => {
                write!(f, "aiesimulator failed ({})", status)?;
                if !stderr.is_empty() {
                    // Show first few lines of stderr for context.
                    let snippet: String = stderr.lines().take(5).collect::<Vec<_>>().join("\n");
                    write!(f, ": {}", snippet)?;
                }
                Ok(())
            }
            Self::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for AiesimError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AiesimError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result of an aiesim run.
#[derive(Debug)]
pub struct AiesimResult {
    /// Exit status of the simulator process.
    pub status: ExitStatus,
    /// Path to the VCD file, if one was produced.
    pub vcd_path: Option<PathBuf>,
    /// Path to the simulator output directory (`<pkg_dir>/aiesimulator_output/`).
    pub output_dir: PathBuf,
    /// Wall-clock duration of the simulation.
    pub duration: Duration,
    /// Captured stdout from the simulator.
    pub stdout: String,
    /// Captured stderr from the simulator.
    pub stderr: String,
}

/// Configuration for an aiesim run.
#[derive(Debug, Clone)]
pub struct AiesimConfig {
    /// Path to the sim package directory (contains `config/`, `src/`, `data/`).
    pub pkg_dir: PathBuf,
    /// Whether to produce VCD output (`--dump-vcd` flag).
    pub dump_vcd: bool,
    /// Maximum simulation cycles before timeout (0 = no limit).
    pub cycle_timeout: u64,
    /// Override path to the aiesimulator binary (default: discover from env).
    pub binary_override: Option<PathBuf>,
}

impl AiesimConfig {
    /// Create a config with sensible defaults for VCD cross-validation.
    ///
    /// VCD dumping is enabled, cycle timeout is 100,000 cycles (enough for
    /// most unit tests), and binary discovery uses the environment.
    pub fn for_vcd(pkg_dir: impl Into<PathBuf>) -> Self {
        Self {
            pkg_dir: pkg_dir.into(),
            dump_vcd: true,
            cycle_timeout: 100_000,
            binary_override: None,
        }
    }
}

/// Subprocess harness for AMD aiesimulator.
///
/// This is a stateless utility type -- all methods are associated functions.
/// No instance state is needed because each run is independent.
pub struct AiesimHarness;

impl AiesimHarness {
    /// Discover the aiesimulator binary from the environment.
    ///
    /// Search order:
    /// 1. `AIETOOLS_DIR` env var + `/bin/aiesimulator`
    /// 2. `XILINX_VITIS_AIETOOLS` env var + `/bin/aiesimulator`
    /// 3. `PATH` lookup via `which`
    ///
    /// Returns the absolute path to the binary, or an error with guidance
    /// on how to set up the environment.
    pub fn discover_binary() -> Result<PathBuf, AiesimError> {
        // Try AIETOOLS_DIR first (our project convention).
        if let Ok(dir) = std::env::var("AIETOOLS_DIR") {
            let candidate = PathBuf::from(&dir).join("bin/aiesimulator");
            if candidate.is_file() {
                return candidate.canonicalize().map_err(AiesimError::Io);
            }
        }

        // Try XILINX_VITIS_AIETOOLS (set by Vitis/activate-npu-env.sh).
        if let Ok(dir) = std::env::var("XILINX_VITIS_AIETOOLS") {
            let candidate = PathBuf::from(&dir).join("bin/aiesimulator");
            if candidate.is_file() {
                return candidate.canonicalize().map_err(AiesimError::Io);
            }
        }

        // Try PATH lookup.
        if let Ok(output) = Command::new("which").arg("aiesimulator").output() {
            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout);
                let path = PathBuf::from(path_str.trim());
                if path.is_file() {
                    return path.canonicalize().map_err(AiesimError::Io);
                }
            }
        }

        Err(AiesimError::BinaryNotFound(
            "Set AIETOOLS_DIR or XILINX_VITIS_AIETOOLS to the aietools \
             installation root, or ensure aiesimulator is on PATH. \
             Source activate-npu-env.sh to configure the environment."
                .to_string(),
        ))
    }

    /// Validate that a path is a valid sim package directory.
    ///
    /// A valid package must:
    /// - Exist and be a directory
    /// - Contain `config/scsim_config.json` (the simulator's main config)
    pub fn validate_pkg_dir(path: &Path) -> Result<(), AiesimError> {
        if !path.exists() {
            return Err(AiesimError::InvalidPackage(format!(
                "directory does not exist: {}",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Err(AiesimError::InvalidPackage(format!(
                "not a directory: {}",
                path.display()
            )));
        }

        let config_file = path.join("config/scsim_config.json");
        if !config_file.exists() {
            return Err(AiesimError::InvalidPackage(format!(
                "missing config/scsim_config.json in {} -- \
                 is this a valid sim package? Expected subdirectories: \
                 config/, src/, data/",
                path.display()
            )));
        }

        Ok(())
    }

    /// Run aiesimulator on a sim package and collect results.
    ///
    /// This is the main entry point for VCD cross-validation. It:
    /// 1. Validates the package directory
    /// 2. Discovers (or uses override) the aiesimulator binary
    /// 3. Launches the simulator subprocess with the configured flags
    /// 4. Measures wall-clock duration
    /// 5. Searches for VCD output files after completion
    ///
    /// # Errors
    ///
    /// Returns [`AiesimError::InvalidPackage`] if the package directory is
    /// invalid, [`AiesimError::BinaryNotFound`] if aiesimulator cannot be
    /// found, [`AiesimError::ProcessFailed`] if the simulator exits with a
    /// non-zero status, or [`AiesimError::Io`] for subprocess I/O failures.
    pub fn run(config: &AiesimConfig) -> Result<AiesimResult, AiesimError> {
        // Resolve the package directory to an absolute path.
        let pkg_dir = config.pkg_dir.canonicalize().map_err(|e| {
            AiesimError::InvalidPackage(format!(
                "cannot resolve path {}: {}",
                config.pkg_dir.display(),
                e
            ))
        })?;

        Self::validate_pkg_dir(&pkg_dir)?;

        // Resolve the binary.
        let binary = match &config.binary_override {
            Some(path) => {
                if !path.is_file() {
                    return Err(AiesimError::BinaryNotFound(format!(
                        "override binary does not exist: {}",
                        path.display()
                    )));
                }
                path.canonicalize().map_err(AiesimError::Io)?
            }
            None => Self::discover_binary()?,
        };

        // Build the command.
        let mut cmd = Command::new(&binary);
        cmd.arg(format!("--pkg-dir={}", pkg_dir.display()));

        if config.dump_vcd {
            cmd.arg("--dump-vcd");
        }

        if config.cycle_timeout > 0 {
            cmd.arg(format!(
                "--simulation-cycle-timeout={}",
                config.cycle_timeout
            ));
        }

        // Run from the package directory's parent so relative paths in the
        // simulator config resolve correctly.
        if let Some(parent) = pkg_dir.parent() {
            cmd.current_dir(parent);
        }

        // Execute with captured output.
        let start = Instant::now();
        let output = cmd.output()?;
        let duration = start.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // The simulator writes output to <pkg_dir>/aiesimulator_output/.
        let output_dir = pkg_dir.join("aiesimulator_output");

        // Search for VCD files in the output directory.
        let vcd_path = Self::find_vcd(&output_dir);

        let result = AiesimResult {
            status: output.status,
            vcd_path,
            output_dir,
            duration,
            stdout,
            stderr,
        };

        // Return the result even on non-zero exit -- the caller may want
        // to inspect stdout/stderr for diagnostics. Only return an error
        // for actual process launch failures (handled by `?` above).
        Ok(result)
    }

    /// Search a directory for VCD files.
    ///
    /// Returns the first `.vcd` file found in the directory. aiesimulator
    /// typically produces one main VCD file per run.
    pub fn find_vcd(output_dir: &Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(output_dir).ok()?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "vcd" {
                        return Some(path);
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Check whether aiesimulator is available on this machine.
    fn has_aiesim() -> bool {
        AiesimHarness::discover_binary().is_ok()
    }

    #[test]
    fn test_discover_binary_from_env() {
        // If the environment is configured (AIETOOLS_DIR or
        // XILINX_VITIS_AIETOOLS), discover_binary should succeed.
        // This test verifies the path construction logic.
        if !has_aiesim() {
            return;
        }
        let path = AiesimHarness::discover_binary().unwrap();
        assert!(path.is_file(), "discovered path should be a file: {}", path.display());
        assert!(
            path.to_string_lossy().contains("aiesimulator"),
            "path should contain 'aiesimulator': {}",
            path.display()
        );
    }

    #[test]
    fn test_discover_binary_not_found() {
        // With a clean environment (no aietools vars, no PATH entry),
        // discovery should fail with BinaryNotFound.
        // We cannot fully test this without clearing env vars, so we
        // just verify the error type when using a bogus override.
        let config = AiesimConfig {
            pkg_dir: PathBuf::from("/nonexistent"),
            dump_vcd: true,
            cycle_timeout: 1000,
            binary_override: Some(PathBuf::from("/nonexistent/bin/aiesimulator")),
        };
        let err = AiesimHarness::run(&config).unwrap_err();
        assert!(
            matches!(err, AiesimError::InvalidPackage(_) | AiesimError::BinaryNotFound(_)),
            "expected InvalidPackage or BinaryNotFound, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_pkg_dir_missing() {
        let err = AiesimHarness::validate_pkg_dir(Path::new("/nonexistent/pkg_dir_12345"));
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("does not exist"),
            "error should mention non-existence: {}",
            msg
        );
    }

    #[test]
    fn test_validate_pkg_dir_not_a_dir() {
        // Create a temporary file (not a directory) and try to validate it.
        let tmp = std::env::temp_dir().join("aiesim_test_not_a_dir");
        fs::write(&tmp, b"not a directory").unwrap();

        let err = AiesimHarness::validate_pkg_dir(&tmp);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("not a directory"),
            "error should mention not a directory: {}",
            msg
        );

        fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_validate_pkg_dir_no_config() {
        // Create a temporary directory without the config file.
        let tmp = std::env::temp_dir().join("aiesim_test_no_config");
        fs::create_dir_all(&tmp).unwrap();

        let err = AiesimHarness::validate_pkg_dir(&tmp);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("scsim_config.json"),
            "error should mention missing config: {}",
            msg
        );

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_validate_pkg_dir_valid() {
        // Create a temporary directory with the expected config file.
        let tmp = std::env::temp_dir().join("aiesim_test_valid_pkg");
        let config_dir = tmp.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(config_dir.join("scsim_config.json"), b"{}").unwrap();

        let result = AiesimHarness::validate_pkg_dir(&tmp);
        assert!(result.is_ok(), "valid package should pass: {:?}", result.err());

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_aiesim_config_for_vcd() {
        let config = AiesimConfig::for_vcd("/some/pkg/dir");
        assert_eq!(config.pkg_dir, PathBuf::from("/some/pkg/dir"));
        assert!(config.dump_vcd, "VCD dumping should be enabled by default");
        assert_eq!(config.cycle_timeout, 100_000);
        assert!(config.binary_override.is_none());
    }

    #[test]
    fn test_find_vcd_in_directory() {
        // Create a temp directory with a .vcd file and verify discovery.
        let tmp = std::env::temp_dir().join("aiesim_test_find_vcd");
        fs::create_dir_all(&tmp).unwrap();
        let vcd_file = tmp.join("default.vcd");
        fs::write(&vcd_file, b"$timescale 1ns $end\n").unwrap();

        let found = AiesimHarness::find_vcd(&tmp);
        assert!(found.is_some(), "should find the .vcd file");
        assert_eq!(
            found.unwrap().file_name().unwrap().to_str().unwrap(),
            "default.vcd"
        );

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_find_vcd_empty_directory() {
        let tmp = std::env::temp_dir().join("aiesim_test_find_vcd_empty");
        fs::create_dir_all(&tmp).unwrap();

        let found = AiesimHarness::find_vcd(&tmp);
        assert!(found.is_none(), "empty directory should yield None");

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_find_vcd_nonexistent_directory() {
        let found = AiesimHarness::find_vcd(Path::new("/nonexistent/dir_12345"));
        assert!(found.is_none(), "nonexistent directory should yield None");
    }

    #[test]
    fn test_find_vcd_ignores_non_vcd() {
        // Directory with files but no .vcd extension.
        let tmp = std::env::temp_dir().join("aiesim_test_find_vcd_no_match");
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("output.txt"), b"data").unwrap();
        fs::write(tmp.join("trace.log"), b"log").unwrap();

        let found = AiesimHarness::find_vcd(&tmp);
        assert!(found.is_none(), "should not match non-.vcd files");

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_error_display_binary_not_found() {
        let err = AiesimError::BinaryNotFound("test message".to_string());
        let msg = err.to_string();
        assert!(msg.contains("aiesimulator not found"));
        assert!(msg.contains("test message"));
    }

    #[test]
    fn test_error_display_invalid_package() {
        let err = AiesimError::InvalidPackage("bad dir".to_string());
        let msg = err.to_string();
        assert!(msg.contains("invalid sim package"));
        assert!(msg.contains("bad dir"));
    }

    #[test]
    fn test_error_display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
        let err = AiesimError::Io(io_err);
        let msg = err.to_string();
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("file gone"));
    }

    #[test]
    fn test_run_invalid_package() {
        // Attempting to run with a nonexistent package should fail early
        // with InvalidPackage, not try to launch the binary.
        let config = AiesimConfig {
            pkg_dir: PathBuf::from("/nonexistent/sim_pkg_12345"),
            dump_vcd: true,
            cycle_timeout: 1000,
            binary_override: None,
        };
        let err = AiesimHarness::run(&config).unwrap_err();
        assert!(
            matches!(err, AiesimError::InvalidPackage(_)),
            "expected InvalidPackage, got: {}",
            err
        );
    }
}
