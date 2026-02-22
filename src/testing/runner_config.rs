//! Runner configuration and CLI argument parsing.
//!
//! Configuration sources (in priority order):
//! 1. CLI flags (highest priority, override everything)
//! 2. `runner.toml` at the project root
//! 3. Built-in defaults
//!
//! The `Options` struct merges all three into a single decision object
//! that the test runner uses throughout its lifecycle.

use std::path::PathBuf;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Runner configuration (loaded from runner.toml)
// ---------------------------------------------------------------------------

/// Persistent configuration for the test runner.
///
/// Loaded from `runner.toml` at the project root. Missing keys fall back
/// to built-in defaults. CLI flags override config values.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct RunnerConfig {
    pub execution: ExecutionConfig,
    pub build: BuildConfig,
    pub chess: ChessConfig,
    pub aiesim: AiesimConfig,
    pub unit_tests: UnitTestsConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ExecutionConfig {
    /// Maximum emulator cycles before TDR-based timeout. 0 = TDR only.
    pub max_cycles: u64,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct BuildConfig {
    /// Nice level for compilation subprocesses (0-19).
    pub nice_level: i32,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ChessConfig {
    /// Build tests with Chess compiler during batch build phase.
    pub build: bool,
    /// Run Chess-compiled binaries through the emulator.
    pub run_emulator: bool,
    /// Run Chess-compiled binaries on real NPU hardware.
    pub run_hardware: bool,
    /// Use Chess as the primary compiler for ALL tests (skip Peano entirely).
    /// Temporary option while the IRON/Peano stack matures.
    pub chess_only: bool,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct AiesimConfig {
    /// Run aiesimulator on Chess-built .prj directories.
    pub enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct UnitTestsConfig {
    /// Enable unit test discovery and execution.
    pub enabled: bool,
    /// Run discovered unit tests through aiesimulator.
    pub aiesim: bool,
    /// Maximum simulation cycles for unit test aiesimulator runs.
    pub aiesim_timeout: u64,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            execution: ExecutionConfig::default(),
            build: BuildConfig::default(),
            chess: ChessConfig::default(),
            aiesim: AiesimConfig::default(),
            unit_tests: UnitTestsConfig::default(),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self { max_cycles: 1_000_000 }
    }
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self { nice_level: 19 }
    }
}

impl Default for ChessConfig {
    fn default() -> Self {
        Self {
            build: false,
            run_emulator: false,
            run_hardware: true,
            chess_only: false,
        }
    }
}

impl Default for AiesimConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

impl Default for UnitTestsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            aiesim: true,
            aiesim_timeout: 1_000_000,
        }
    }
}

impl RunnerConfig {
    /// Load config from `runner.toml` in the given manifest directory.
    ///
    /// Returns the built-in defaults if the file is missing or unparseable.
    pub fn load(manifest_dir: &str) -> Self {
        let config_path = PathBuf::from(manifest_dir).join("runner.toml");
        match std::fs::read_to_string(&config_path) {
            Ok(content) => {
                match toml::from_str(&content) {
                    Ok(config) => {
                        log::debug!("Loaded runner config from {}", config_path.display());
                        config
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to parse {}: {}", config_path.display(), e);
                        Self::default()
                    }
                }
            }
            Err(_) => Self::default(),
        }
    }

    /// Print a summary of non-default settings.
    pub fn print_active_overrides(&self) {
        let d = Self::default();
        let mut overrides = Vec::new();
        if self.chess.chess_only != d.chess.chess_only {
            overrides.push(format!("chess.chess_only={}", self.chess.chess_only));
        }
        if self.chess.run_emulator != d.chess.run_emulator {
            overrides.push(format!("chess.run_emulator={}", self.chess.run_emulator));
        }
        if self.chess.run_hardware != d.chess.run_hardware {
            overrides.push(format!("chess.run_hardware={}", self.chess.run_hardware));
        }
        if self.aiesim.enabled != d.aiesim.enabled {
            overrides.push(format!("aiesim.enabled={}", self.aiesim.enabled));
        }
        if self.unit_tests.enabled != d.unit_tests.enabled {
            overrides.push(format!("unit_tests.enabled={}", self.unit_tests.enabled));
        }
        if self.build.nice_level != d.build.nice_level {
            overrides.push(format!("build.nice_level={}", self.build.nice_level));
        }
        if self.execution.max_cycles != d.execution.max_cycles {
            overrides.push(format!("execution.max_cycles={}", self.execution.max_cycles));
        }
        if !overrides.is_empty() {
            println!("config: {}", overrides.join(", "));
        }
    }
}

// ---------------------------------------------------------------------------
// CLI options (merged with config)
// ---------------------------------------------------------------------------

/// Parsed CLI options, merged with runner config.
pub struct Options {
    pub verbose: bool,
    pub jobs: usize,
    pub filters: Vec<String>,
    /// Run elfanalyzer on each test's ELF files (requires aietools).
    pub elfanalyze: bool,
    /// Build with Chess compiler (batch build phase).
    pub chess_build: bool,
    /// Run Chess-compiled binaries through the emulator.
    pub chess_emulator: bool,
    /// Run Chess-compiled binaries on real NPU hardware.
    pub chess_hardware: bool,
    /// Run tests on real NPU hardware (requires /dev/accel/accel0 + npu-runner).
    pub hw: bool,
    /// Run aiesimulator on Chess-built .prj directories.
    pub aiesim: bool,
    /// Nice level for build subprocesses.
    pub build_nice: i32,
    /// Maximum emulator cycles before timeout.
    pub max_cycles: u64,
    /// Run mlir-aie unit tests (chess_compiler_tests_aie2).
    pub unit_tests: bool,
    /// Run unit tests through aiesimulator.
    pub unit_tests_aiesim: bool,
    /// Maximum simulation cycles for unit test aiesimulator runs.
    pub unit_tests_aiesim_timeout: u64,
    /// Skip build phase entirely -- discover from build output directories.
    /// Useful for quick re-runs of already-built tests.
    pub no_build: bool,
    /// Hardware-only mode: skip emulator, run only on real NPU hardware.
    /// Useful for validating known-good tests quickly on real silicon.
    pub hw_only: bool,
    /// Use Chess as the primary compiler for ALL tests (skip Peano entirely).
    pub chess_only: bool,
}

/// Parse CLI arguments and merge with runner config.
pub fn parse_args(config: &RunnerConfig) -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut verbose = false;
    let mut jobs: usize = 0; // 0 = auto-detect
    let mut filters = Vec::new();
    let mut elfanalyze = false;
    let mut hw = true; // auto-detected, disabled by --no-hw
    let mut hw_only = false;
    let mut aiesim = false;
    let mut full = false;
    let mut unit_tests = false;
    let mut no_build = false;
    let mut no_chess = false;
    let mut chess_only = false;
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--verbose" | "-v" => verbose = true,
            "--elfanalyze" => elfanalyze = true,
            // --chess is accepted for backward compat but is now the default
            "--chess" => {}
            "--no-chess" => no_chess = true,
            "--chess-only" => chess_only = true,
            // --hw is accepted for backward compat but is now the default
            "--hw" => hw = true,
            "--no-hw" => hw = false,
            "--hw-only" => { hw_only = true; hw = true; },
            "--aiesim" => aiesim = true,
            "--full" => full = true,
            "--unit-tests" => unit_tests = true,
            "--no-build" => no_build = true,
            "-j" => {
                if let Some(n) = iter.next() {
                    jobs = n.parse().unwrap_or_else(|_| {
                        eprintln!("Invalid -j value: {}", n);
                        std::process::exit(1);
                    });
                    // -j 0 is nonsensical; treat as 1
                    if jobs == 0 { jobs = 1; }
                } else {
                    eprintln!("-j requires a number");
                    std::process::exit(1);
                }
            }
            _ if !arg.starts_with('-') => filters.push(arg.clone()),
            other => {
                eprintln!("Unknown option: {}", other);
                eprintln!("Usage: run_mlir_aie_tests [--verbose|-v] [-j N] [--elfanalyze] [--no-chess] [--chess-only] [--no-hw] [--hw-only] [--aiesim] [--unit-tests] [--full] [--no-build] [FILTER...]");
                std::process::exit(1);
            }
        }
    }

    // --full enables everything
    if full {
        elfanalyze = true;
        no_chess = false;
        hw = true;
        aiesim = true;
        unit_tests = true;
    }

    // Auto-detect parallelism: use available CPU count, capped at 8 to avoid
    // overwhelming the system during builds. User can override with explicit -j N.
    if jobs == 0 {
        jobs = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
    }

    // Merge chess_only from CLI and config.
    let chess_only = chess_only || config.chess.chess_only;

    // Chess builds are ON by default (auto-detected from aietools).
    // --no-chess explicitly disables. --chess-only implies chess_build.
    let chess_build = if chess_only { true } else { !no_chess };
    let chess_emulator = if chess_build { config.chess.run_emulator } else { false };
    let chess_hardware = if chess_build { config.chess.run_hardware } else { false };
    let aiesim = if aiesim { true } else if chess_build { config.aiesim.enabled } else { false };

    // Unit tests: CLI --unit-tests overrides config.unit_tests.enabled
    let unit_tests = unit_tests || config.unit_tests.enabled;

    Options {
        verbose,
        jobs,
        filters,
        elfanalyze,
        chess_build,
        chess_emulator,
        chess_hardware,
        hw,
        aiesim,
        build_nice: config.build.nice_level,
        max_cycles: config.execution.max_cycles,
        unit_tests,
        unit_tests_aiesim: config.unit_tests.aiesim,
        unit_tests_aiesim_timeout: config.unit_tests.aiesim_timeout,
        no_build,
        hw_only,
        chess_only,
    }
}

/// Check if a test name matches any of the given filters.
pub fn matches_filter(name: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    filters.iter().any(|f| name.contains(f.as_str()))
}
