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

/// Default emulator cycle limit (1M cycles). Used as the default for both
/// `execution.max_cycles` and `unit_tests.aiesim_timeout` in runner config.
pub const DEFAULT_MAX_CYCLES: u64 = 1_000_000;

/// Default timeout for NPU hardware execution (seconds).
/// 30s is generous for any test; prevents infinite hangs from TDR failures.
/// Used by both `hw_executor` (npu-runner path) and `native_hw` (test.exe path).
pub const DEFAULT_HW_TIMEOUT_SECS: u32 = 30;

// ---------------------------------------------------------------------------
// Run mode
// ---------------------------------------------------------------------------

/// Which test runner mode to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    /// Emulator + hardware validation (default).
    EmuHw,
    /// Wrap LLVM's `lit` for standard test execution.
    Lit,
    /// Trace collection: inject tracing, compile, execute on NPU.
    Trace,
    /// Triple trace comparison: HW + emulator (+ optional aiesimulator).
    TraceAll,
}

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
        Self { max_cycles: DEFAULT_MAX_CYCLES }
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
            aiesim_timeout: DEFAULT_MAX_CYCLES,
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
    // --- Common (all modes) ---
    /// Which runner mode was selected.
    pub mode: RunMode,
    pub verbose: bool,
    pub jobs: usize,
    pub filters: Vec<String>,
    /// Use Chess as the primary compiler for ALL tests (skip Peano entirely).
    pub chess_only: bool,

    // --- Emu+HW mode ---
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
    pub no_build: bool,
    /// Hardware-only mode: skip emulator, run only on real NPU hardware.
    pub hw_only: bool,

    // --- Lit mode ---
    /// Per-test timeout in seconds (forwarded to lit / trace pipeline).
    pub timeout_secs: Option<u32>,
    /// Explicit mlir-aie build directory (auto-detected if None).
    pub build_dir: Option<PathBuf>,
    /// Stop after this many failures.
    pub max_failures: Option<usize>,
    /// Extra arguments passed through to lit verbatim.
    pub lit_args: Vec<String>,
    /// Outer watchdog timeout for entire lit process (seconds).
    pub watchdog_secs: u32,

    // --- Trace modes ---
    /// Trace buffer size in bytes (default: 1MB).
    pub trace_size: usize,
    /// Also collect aiesimulator VCD traces (with --trace-all).
    pub aiesim_trace: bool,
}

/// Parse CLI arguments and merge with runner config.
///
/// Handles all modes: default (emu+hw), --lit, --trace, --trace-all.
/// Mode-specific flags are accepted in any mode but only take effect in
/// their respective mode.
pub fn parse_args(config: &RunnerConfig) -> Options {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Common
    let mut verbose = false;
    let mut jobs: usize = 0; // 0 = auto-detect
    let mut filters = Vec::new();
    let mut chess_only = false;

    // Mode selection
    let mut lit_mode = false;
    let mut trace_mode = false;
    let mut trace_all_mode = false;

    // Emu+HW mode
    let mut elfanalyze = false;
    let mut hw = true;
    let mut hw_only = false;
    let mut aiesim = false;
    let mut full = false;
    let mut unit_tests = false;
    let mut no_build = false;
    let mut no_chess = false;

    // Lit mode
    let mut timeout_secs: Option<u32> = None;
    let mut build_dir: Option<PathBuf> = None;
    let mut max_failures: Option<usize> = None;
    let mut lit_args: Vec<String> = Vec::new();
    let mut watchdog_secs: u32 = 3600;

    // Trace modes
    let mut trace_size: usize = 1_048_576; // 1MB default
    let mut aiesim_trace = false;

    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            // --- Mode selection ---
            "--lit" => lit_mode = true,
            "--trace" => trace_mode = true,
            "--trace-all" => { trace_all_mode = true; chess_only = true; }

            // --- Common ---
            "--verbose" | "-v" => verbose = true,
            "--chess-only" => chess_only = true,
            "-j" => {
                let n = next_value(&mut iter, "-j");
                jobs = n.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid -j value: {}", n);
                    std::process::exit(1);
                });
                if jobs == 0 { jobs = 1; }
            }

            // --- Emu+HW mode ---
            "--elfanalyze" => elfanalyze = true,
            "--chess" => {} // backward compat, now the default
            "--no-chess" => no_chess = true,
            "--hw" => hw = true, // backward compat
            "--no-hw" => hw = false,
            "--hw-only" => { hw_only = true; hw = true; }
            "--aiesim" => aiesim = true,
            "--full" => full = true,
            "--unit-tests" => unit_tests = true,
            "--no-build" => no_build = true,

            // --- Lit mode ---
            "--timeout" => {
                let v = next_value(&mut iter, "--timeout");
                timeout_secs = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --timeout value: {}", v);
                    std::process::exit(1);
                }));
            }
            "--build-dir" => {
                let v = next_value(&mut iter, "--build-dir");
                build_dir = Some(PathBuf::from(v));
            }
            "--max-failures" => {
                let v = next_value(&mut iter, "--max-failures");
                max_failures = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --max-failures value: {}", v);
                    std::process::exit(1);
                }));
            }
            "--lit-args" => {
                let v = next_value(&mut iter, "--lit-args");
                lit_args.extend(v.split_whitespace().map(String::from));
            }
            "--watchdog" => {
                let v = next_value(&mut iter, "--watchdog");
                watchdog_secs = v.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --watchdog value: {}", v);
                    std::process::exit(1);
                });
            }

            // --- Trace modes ---
            "--trace-size" => {
                let v = next_value(&mut iter, "--trace-size");
                trace_size = v.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --trace-size value: {}", v);
                    std::process::exit(1);
                });
            }
            "--aiesim-trace" => aiesim_trace = true,

            // --- Help ---
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }

            // --- Positional / unknown ---
            _ if !arg.starts_with('-') => filters.push(arg.clone()),
            other => {
                eprintln!("Unknown option: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
    }

    // Determine mode (mutually exclusive)
    let mode = if trace_all_mode {
        RunMode::TraceAll
    } else if trace_mode {
        RunMode::Trace
    } else if lit_mode {
        RunMode::Lit
    } else {
        RunMode::EmuHw
    };

    // --full enables everything (emu+hw mode)
    if full {
        elfanalyze = true;
        no_chess = false;
        hw = true;
        aiesim = true;
        unit_tests = true;
    }

    // Auto-detect parallelism: use available CPU count, capped at 8.
    if jobs == 0 {
        jobs = match mode {
            // Lit mode defaults to 1 (lit handles parallelism internally)
            RunMode::Lit | RunMode::Trace | RunMode::TraceAll => 1,
            RunMode::EmuHw => {
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8))
                    .unwrap_or(4)
            }
        };
    }

    // Merge chess_only from CLI and config.
    let chess_only = chess_only || config.chess.chess_only;

    // Chess builds are ON by default (auto-detected from aietools).
    let chess_build = if chess_only { true } else { !no_chess };
    let chess_emulator = if chess_build { config.chess.run_emulator } else { false };
    let chess_hardware = if chess_build { config.chess.run_hardware } else { false };
    let aiesim = if aiesim { true } else if chess_build { config.aiesim.enabled } else { false };

    // Unit tests: CLI overrides config
    let unit_tests = unit_tests || config.unit_tests.enabled;

    Options {
        mode,
        verbose,
        jobs,
        filters,
        chess_only,

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

        timeout_secs,
        build_dir,
        max_failures,
        lit_args,
        watchdog_secs,

        trace_size,
        aiesim_trace,
    }
}

/// Consume the next argument as a value for the given flag, or exit.
fn next_value<'a>(iter: &mut impl Iterator<Item = &'a String>, flag: &str) -> &'a String {
    iter.next().unwrap_or_else(|| {
        eprintln!("{} requires a value", flag);
        std::process::exit(1);
    })
}

/// Print unified usage for all modes.
fn print_usage() {
    eprintln!("npu-test: Unified test runner for mlir-aie npu-xrt tests");
    eprintln!();
    eprintln!("Usage: npu-test [MODE] [OPTIONS] [FILTER...]");
    eprintln!();
    eprintln!("Modes (default: emulator + hardware):");
    eprintln!("  (none)              Emulator + hardware validation");
    eprintln!("  --lit               Run tests via LLVM's lit framework");
    eprintln!("  --trace             Trace collection on real NPU hardware");
    eprintln!("  --trace-all         Triple trace: HW + emulator (+ aiesimulator)");
    eprintln!();
    eprintln!("Common options:");
    eprintln!("  -v, --verbose       Show detailed output");
    eprintln!("  -j N                Parallelism (default: auto for emu, 1 for lit/trace)");
    eprintln!("  --chess-only        Use Chess compiler (skip Peano)");
    eprintln!("  -h, --help          Show this help");
    eprintln!();
    eprintln!("Emulator mode options:");
    eprintln!("  --elfanalyze        Run elfanalyzer on each test's ELFs");
    eprintln!("  --no-chess          Disable Chess compiler builds");
    eprintln!("  --no-hw             Skip NPU hardware validation");
    eprintln!("  --hw-only           Skip emulator, hardware only");
    eprintln!("  --aiesim            Run aiesimulator on Chess builds");
    eprintln!("  --unit-tests        Run mlir-aie unit tests");
    eprintln!("  --full              Enable all validation modes");
    eprintln!("  --no-build          Skip build phase, use pre-built tests");
    eprintln!();
    eprintln!("Lit mode options:");
    eprintln!("  --timeout SECS      Per-test timeout (forwarded to lit)");
    eprintln!("  --build-dir PATH    mlir-aie build directory (auto-detected)");
    eprintln!("  --max-failures N    Stop after N failures");
    eprintln!("  --lit-args \"...\"    Extra arguments passed to lit verbatim");
    eprintln!("  --watchdog SECS     Outer watchdog for entire lit process (default: 3600)");
    eprintln!();
    eprintln!("Trace mode options:");
    eprintln!("  --trace-size BYTES  Trace buffer size (default: 1048576 = 1MB)");
    eprintln!("  --aiesim-trace      Also collect aiesimulator VCD traces (--trace-all)");
    eprintln!();
    eprintln!("Filters are substring matches on test names, OR-ed together.");
}

/// Check if a test name matches any of the given filters.
pub fn matches_filter(name: &str, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    filters.iter().any(|f| name.contains(f.as_str()))
}
