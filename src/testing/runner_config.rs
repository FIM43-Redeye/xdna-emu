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
    /// Differential fuzzing: generate random kernels, compare emulator vs NPU.
    Fuzz,
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
    pub examples: ExamplesConfig,
    pub defaults: DefaultsConfig,
}

/// Default selections from config file (runtime, compiler, suite axes).
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct DefaultsConfig {
    /// Comma-separated runtime spec (execution targets).
    /// Valid tokens: emu, hw, aiesim, all. Negation: -emu, -hw, etc.
    pub runtime: String,
    /// Comma-separated compiler spec.
    /// Valid tokens: peano, chess, all. Negation: -peano, -chess.
    pub compiler: String,
    /// Comma-separated suite spec (test sources).
    /// Valid tokens: npu-xrt, examples, unit-tests, all. Negation: -npu-xrt, etc.
    pub suite: String,
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

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ExamplesConfig {
    /// Include programming_examples/ tests in test discovery.
    pub enabled: bool,
}

impl Default for ExamplesConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            runtime: "emu".to_string(),
            compiler: "peano".to_string(),
            suite: "npu-xrt".to_string(),
        }
    }
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            execution: ExecutionConfig::default(),
            build: BuildConfig::default(),
            chess: ChessConfig::default(),
            aiesim: AiesimConfig::default(),
            unit_tests: UnitTestsConfig::default(),
            examples: ExamplesConfig::default(),
            defaults: DefaultsConfig::default(),
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
            run_emulator: true,
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

// ---------------------------------------------------------------------------
// Generic flag set parser
// ---------------------------------------------------------------------------

/// Parse a comma-separated flag spec against a list of valid tokens.
///
/// Supports `all` (enables everything) and negation via `-` prefix.
/// Negations are applied after all positive tokens, so `all,-foo` enables
/// everything except `foo`.
///
/// Returns a `HashSet<String>` of enabled tokens, or an error on empty
/// input, unknown tokens, or negation-only specs.
///
/// `axis_name` is used in error messages (e.g. "runtime", "compiler").
/// `migration_hints` maps old tokens to helpful error messages.
fn parse_flag_set(
    spec: &str,
    valid_tokens: &[&str],
    axis_name: &str,
    migration_hints: &[(&str, &str)],
) -> Result<std::collections::HashSet<String>, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(format!("{} spec is empty", axis_name));
    }

    let mut enabled = std::collections::HashSet::new();
    let mut negations = Vec::new();
    let mut has_positive = false;

    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() { continue; }

        let (negate, name) = if let Some(rest) = token.strip_prefix('-') {
            (true, rest)
        } else {
            (false, token)
        };

        // Check for migrated tokens first (helpful error on old usage).
        if let Some((_, hint)) = migration_hints.iter().find(|(old, _)| *old == name) {
            return Err(format!(
                "unknown {}: '{}' ({})", axis_name, name, hint
            ));
        }

        if name == "all" {
            if negate {
                negations.extend(valid_tokens.iter().map(|s| s.to_string()));
            } else {
                for t in valid_tokens {
                    enabled.insert(t.to_string());
                }
                has_positive = true;
            }
        } else if valid_tokens.contains(&name) {
            if negate {
                negations.push(name.to_string());
            } else {
                enabled.insert(name.to_string());
                has_positive = true;
            }
        } else {
            return Err(format!("unknown {}: '{}'", axis_name, name));
        }
    }

    if !has_positive {
        return Err(format!(
            "{} spec has only negations, no positive tokens", axis_name
        ));
    }

    for neg in negations {
        enabled.remove(&neg);
    }

    Ok(enabled)
}

// ---------------------------------------------------------------------------
// Three-axis flag sets
// ---------------------------------------------------------------------------

/// Execution targets: where test binaries run.
///
/// Parsed from `--runtime=` or `[defaults] runtime` in runner.toml.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSet {
    pub emu: bool,
    pub hw: bool,
    pub aiesim: bool,
}

/// Tokens that migrated out of --runtime= into other axes.
const RUNTIME_MIGRATION_HINTS: &[(&str, &str)] = &[
    ("chess", "use --compiler=chess instead"),
    ("unit-tests", "use --suite=unit-tests instead"),
    ("examples", "use --suite=examples instead"),
];

impl RuntimeSet {
    /// Valid tokens for the runtime axis.
    const TOKENS: &[&str] = &["emu", "hw", "aiesim"];

    /// Parse a comma-separated runtime specification.
    ///
    /// Valid tokens: `emu`, `hw`, `aiesim`, `all`.
    /// Negation: prefix with `-` (e.g. `-aiesim`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let set = parse_flag_set(
            spec, Self::TOKENS, "runtime", RUNTIME_MIGRATION_HINTS,
        )?;
        Ok(Self {
            emu: set.contains("emu"),
            hw: set.contains("hw"),
            aiesim: set.contains("aiesim"),
        })
    }

    /// Reconstruct a display string from enabled flags.
    pub fn to_spec(&self) -> String {
        let mut parts = Vec::new();
        if self.emu { parts.push("emu"); }
        if self.hw { parts.push("hw"); }
        if self.aiesim { parts.push("aiesim"); }
        if parts.is_empty() { "emu".to_string() } else { parts.join(",") }
    }
}

/// Compiler selection: what builds the test binaries.
///
/// Parsed from `--compiler=` or `[defaults] compiler` in runner.toml.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerSet {
    pub peano: bool,
    pub chess: bool,
}

impl CompilerSet {
    /// Valid tokens for the compiler axis.
    const TOKENS: &[&str] = &["peano", "chess"];

    /// Parse a comma-separated compiler specification.
    ///
    /// Valid tokens: `peano`, `chess`, `all`.
    /// Negation: prefix with `-` (e.g. `-chess`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let set = parse_flag_set(spec, Self::TOKENS, "compiler", &[])?;
        Ok(Self {
            peano: set.contains("peano"),
            chess: set.contains("chess"),
        })
    }

    /// Reconstruct a display string from enabled flags.
    pub fn to_spec(&self) -> String {
        let mut parts = Vec::new();
        if self.peano { parts.push("peano"); }
        if self.chess { parts.push("chess"); }
        if parts.is_empty() { "peano".to_string() } else { parts.join(",") }
    }
}

/// Test suite selection: which test sources to include.
///
/// Parsed from `--suite=` or `[defaults] suite` in runner.toml.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteSet {
    /// The base npu-xrt test suite (always available).
    pub npu_xrt: bool,
    /// Include programming_examples/ tests.
    pub examples: bool,
    /// Include chess_compiler_tests_aie2 unit tests.
    pub unit_tests: bool,
}

impl SuiteSet {
    /// Valid tokens for the suite axis.
    const TOKENS: &[&str] = &["npu-xrt", "examples", "unit-tests"];

    /// Parse a comma-separated suite specification.
    ///
    /// Valid tokens: `npu-xrt`, `examples`, `unit-tests`, `all`.
    /// Negation: prefix with `-` (e.g. `-examples`).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let set = parse_flag_set(spec, Self::TOKENS, "suite", &[])?;
        Ok(Self {
            npu_xrt: set.contains("npu-xrt"),
            examples: set.contains("examples"),
            unit_tests: set.contains("unit-tests"),
        })
    }

    /// Reconstruct a display string from enabled flags.
    pub fn to_spec(&self) -> String {
        let mut parts = Vec::new();
        if self.npu_xrt { parts.push("npu-xrt"); }
        if self.examples { parts.push("examples"); }
        if self.unit_tests { parts.push("unit-tests"); }
        if parts.is_empty() { "npu-xrt".to_string() } else { parts.join(",") }
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
        if self.examples.enabled != d.examples.enabled {
            overrides.push(format!("examples.enabled={}", self.examples.enabled));
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
    /// Include programming_examples/ as an additional test source.
    pub examples: bool,
    /// Force rebuild even if cached artifacts are fresh.
    pub rebuild: bool,

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

    /// List-only mode: discover and print tests, then exit.
    pub list_only: bool,

    // --- JSON output ---
    /// Path for machine-readable JSON results (default: build/results.json).
    pub output_path: PathBuf,
    /// When true, print JSON to stdout instead of human-readable output.
    pub format_json: bool,

    // --- Fuzz mode ---
    /// Number of fuzz iterations to run.
    pub fuzz_iterations: usize,
    /// Base seed for fuzz generation (None = use wall clock).
    pub fuzz_seed: Option<u64>,
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
    let mut fuzz_mode = false;

    // Fuzz mode
    let mut fuzz_iterations: usize = 100;
    let mut fuzz_seed: Option<u64> = None;

    // Emu+HW mode
    let mut elfanalyze = false;
    let mut runtime_spec: Option<String> = None;
    let mut compiler_spec: Option<String> = None;
    let mut suite_spec: Option<String> = None;
    let mut no_build = false;
    let mut rebuild = false;

    // Lit mode
    let mut timeout_secs: Option<u32> = None;
    let mut build_dir: Option<PathBuf> = None;
    let mut max_failures: Option<usize> = None;
    let mut lit_args: Vec<String> = Vec::new();
    let mut watchdog_secs: u32 = 3600;

    // Trace modes
    let mut trace_size: usize = 1_048_576; // 1MB default
    let mut aiesim_trace = false;
    let mut list_only = false;

    // JSON output
    let mut output_path: Option<PathBuf> = None;
    let mut format_json = false;

    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        // Handle --runtime=value (= form)
        if let Some(spec) = arg.strip_prefix("--runtime=") {
            runtime_spec = Some(spec.to_string());
            continue;
        }
        // Handle --compiler=value (= form)
        if let Some(spec) = arg.strip_prefix("--compiler=") {
            compiler_spec = Some(spec.to_string());
            continue;
        }
        // Handle --suite=value (= form)
        if let Some(spec) = arg.strip_prefix("--suite=") {
            suite_spec = Some(spec.to_string());
            continue;
        }
        // Handle --output=value (= form)
        if let Some(path) = arg.strip_prefix("--output=") {
            output_path = Some(PathBuf::from(path));
            continue;
        }
        // Handle --format=value (= form)
        if let Some(fmt) = arg.strip_prefix("--format=") {
            match fmt {
                "json" => format_json = true,
                other => {
                    eprintln!("Unknown format: '{}' (valid: json)", other);
                    std::process::exit(1);
                }
            }
            continue;
        }

        match arg.as_str() {
            // --- Mode selection ---
            "--lit" => lit_mode = true,
            "--trace" => trace_mode = true,
            "--trace-all" => { trace_all_mode = true; chess_only = true; }
            "--fuzz" => fuzz_mode = true,

            // --- Fuzz mode ---
            "--iterations" => {
                let v = next_value(&mut iter, "--iterations");
                fuzz_iterations = v.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --iterations value: {}", v);
                    std::process::exit(1);
                });
            }
            "--seed" => {
                let v = next_value(&mut iter, "--seed");
                fuzz_seed = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --seed value: {}", v);
                    std::process::exit(1);
                }));
            }

            // --- Common ---
            "--verbose" | "-v" => verbose = true,
            "--chess-only" => { chess_only = true; compiler_spec = Some("chess".to_string()); }
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
            "--no-build" => no_build = true,
            "--rebuild" => rebuild = true,

            // --- Selection (space-separated forms) ---
            "--runtime" => {
                let v = next_value(&mut iter, "--runtime");
                runtime_spec = Some(v.clone());
            }
            "--compiler" => {
                let v = next_value(&mut iter, "--compiler");
                compiler_spec = Some(v.clone());
            }
            "--suite" => {
                let v = next_value(&mut iter, "--suite");
                suite_spec = Some(v.clone());
            }

            // --- JSON output (space-separated form) ---
            "--output" => {
                let v = next_value(&mut iter, "--output");
                output_path = Some(PathBuf::from(v.as_str()));
            }
            "--format" => {
                let v = next_value(&mut iter, "--format");
                match v.as_str() {
                    "json" => format_json = true,
                    other => {
                        eprintln!("Unknown format: '{}' (valid: json)", other);
                        std::process::exit(1);
                    }
                }
            }

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

            // --- Discovery ---
            "--list" => list_only = true,

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
    let mode = if fuzz_mode {
        RunMode::Fuzz
    } else if trace_all_mode {
        RunMode::TraceAll
    } else if trace_mode {
        RunMode::Trace
    } else if lit_mode {
        RunMode::Lit
    } else {
        RunMode::EmuHw
    };

    // Parse all three axes from CLI or config defaults.
    let rt_spec = runtime_spec.unwrap_or_else(|| config.defaults.runtime.clone());
    let runtime = RuntimeSet::parse(&rt_spec).unwrap_or_else(|e| {
        eprintln!("Invalid runtime spec '{}': {}", rt_spec, e);
        std::process::exit(1);
    });

    let comp_spec = compiler_spec.unwrap_or_else(|| config.defaults.compiler.clone());
    let mut compiler = CompilerSet::parse(&comp_spec).unwrap_or_else(|e| {
        eprintln!("Invalid compiler spec '{}': {}", comp_spec, e);
        std::process::exit(1);
    });

    let suite_spec = suite_spec.unwrap_or_else(|| config.defaults.suite.clone());
    let suite = SuiteSet::parse(&suite_spec).unwrap_or_else(|e| {
        eprintln!("Invalid suite spec '{}': {}", suite_spec, e);
        std::process::exit(1);
    });

    // Auto-inference: aiesim requires Chess builds.
    if runtime.aiesim && !compiler.chess {
        eprintln!("note: aiesim requires Chess builds, enabling --compiler=chess");
        compiler.chess = true;
    }

    // Auto-detect parallelism: use all available CPUs.
    if jobs == 0 {
        jobs = match mode {
            // Lit/trace/fuzz modes default to 1 (sequential)
            RunMode::Lit | RunMode::Trace | RunMode::TraceAll | RunMode::Fuzz => 1,
            RunMode::EmuHw => {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            }
        };
    }

    // Derive Options fields from the three axes.
    let chess_build = compiler.chess;
    let chess_only = chess_only || config.chess.chess_only || (chess_build && !compiler.peano);
    let hw = runtime.hw;
    let hw_only = runtime.hw && !runtime.emu;
    let chess_emulator = if chess_build { config.chess.run_emulator && runtime.emu } else { false };
    let chess_hardware = if chess_build { config.chess.run_hardware && runtime.hw } else { false };
    let aiesim = runtime.aiesim;
    let unit_tests = suite.unit_tests;
    let examples = suite.examples;
    let elfanalyze = elfanalyze || (runtime.emu && runtime.hw && runtime.aiesim);

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
        examples,
        rebuild,

        timeout_secs,
        build_dir,
        max_failures,
        lit_args,
        watchdog_secs,

        trace_size,
        aiesim_trace,
        list_only,

        output_path: output_path.unwrap_or_else(|| PathBuf::from("build/results.json")),
        format_json,

        fuzz_iterations,
        fuzz_seed,
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
    eprintln!("  --fuzz              Differential fuzzing (generate + emulate + compare)");
    eprintln!();
    eprintln!("Common options:");
    eprintln!("  -v, --verbose       Show detailed output");
    eprintln!("  -j N                Parallelism (default: auto for emu, 1 for lit/trace)");
    eprintln!("  --chess-only        Shorthand for --compiler=chess");
    eprintln!("  -h, --help          Show this help");
    eprintln!();
    eprintln!("Selection:");
    eprintln!("  --runtime=TARGETS   Execution targets (default: from runner.toml)");
    eprintln!("                      Valid: emu, hw, aiesim, all");
    eprintln!("  --compiler=COMPILERS  Compiler selection (default: from runner.toml)");
    eprintln!("                      Valid: peano, chess, all");
    eprintln!("  --suite=SUITES      Test suites to include (default: from runner.toml)");
    eprintln!("                      Valid: npu-xrt, examples, unit-tests, all");
    eprintln!("                      Negation: -token on any axis. Example: all,-aiesim");
    eprintln!();
    eprintln!("Emulator mode options:");
    eprintln!("  --elfanalyze        Run elfanalyzer on each test's ELFs");
    eprintln!("  --no-build          Skip build phase, use pre-built tests");
    eprintln!("  --rebuild           Force rebuild even if cached artifacts are fresh");
    eprintln!("  --list              Discover tests and print list, then exit");
    eprintln!();
    eprintln!("Output options:");
    eprintln!("  --output=PATH       JSON results file (default: build/results.json)");
    eprintln!("  --format=json       Write JSON to stdout instead of human-readable");
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
    eprintln!("Fuzz mode options:");
    eprintln!("  --iterations N      Number of fuzz cases to generate (default: 100)");
    eprintln!("  --seed N            Base RNG seed (default: wall clock)");
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- RuntimeSet tests ---

    #[test]
    fn runtime_set_emu_only() {
        let r = RuntimeSet::parse("emu").unwrap();
        assert!(r.emu);
        assert!(!r.hw);
        assert!(!r.aiesim);
    }

    #[test]
    fn runtime_set_all() {
        let r = RuntimeSet::parse("all").unwrap();
        assert!(r.emu);
        assert!(r.hw);
        assert!(r.aiesim);
    }

    #[test]
    fn runtime_set_all_minus_aiesim() {
        let r = RuntimeSet::parse("all,-aiesim").unwrap();
        assert!(r.emu);
        assert!(r.hw);
        assert!(!r.aiesim);
    }

    #[test]
    fn runtime_set_multi() {
        let r = RuntimeSet::parse("emu,hw").unwrap();
        assert!(r.emu);
        assert!(r.hw);
        assert!(!r.aiesim);
    }

    #[test]
    fn runtime_set_empty_is_error() {
        assert!(RuntimeSet::parse("").is_err());
    }

    #[test]
    fn runtime_set_unknown_is_error() {
        assert!(RuntimeSet::parse("bogus").is_err());
        assert!(RuntimeSet::parse("emu,nope").is_err());
    }

    #[test]
    fn runtime_set_negation_only_is_error() {
        assert!(RuntimeSet::parse("-emu").is_err());
        assert!(RuntimeSet::parse("-hw,-aiesim").is_err());
    }

    #[test]
    fn runtime_set_whitespace_tolerance() {
        let r = RuntimeSet::parse(" emu , hw ").unwrap();
        assert!(r.emu);
        assert!(r.hw);
    }

    #[test]
    fn runtime_set_migrated_token_chess() {
        let err = RuntimeSet::parse("chess").unwrap_err();
        assert!(err.contains("--compiler=chess"), "got: {}", err);
    }

    #[test]
    fn runtime_set_migrated_token_unit_tests() {
        let err = RuntimeSet::parse("emu,unit-tests").unwrap_err();
        assert!(err.contains("--suite=unit-tests"), "got: {}", err);
    }

    #[test]
    fn runtime_set_migrated_token_examples() {
        let err = RuntimeSet::parse("emu,examples").unwrap_err();
        assert!(err.contains("--suite=examples"), "got: {}", err);
    }

    #[test]
    fn runtime_set_to_spec() {
        let r = RuntimeSet { emu: true, hw: true, aiesim: false };
        assert_eq!(r.to_spec(), "emu,hw");
    }

    // --- CompilerSet tests ---

    #[test]
    fn compiler_set_peano_only() {
        let c = CompilerSet::parse("peano").unwrap();
        assert!(c.peano);
        assert!(!c.chess);
    }

    #[test]
    fn compiler_set_chess_only() {
        let c = CompilerSet::parse("chess").unwrap();
        assert!(!c.peano);
        assert!(c.chess);
    }

    #[test]
    fn compiler_set_all() {
        let c = CompilerSet::parse("all").unwrap();
        assert!(c.peano);
        assert!(c.chess);
    }

    #[test]
    fn compiler_set_all_minus_peano() {
        let c = CompilerSet::parse("all,-peano").unwrap();
        assert!(!c.peano);
        assert!(c.chess);
    }

    #[test]
    fn compiler_set_empty_is_error() {
        assert!(CompilerSet::parse("").is_err());
    }

    #[test]
    fn compiler_set_unknown_is_error() {
        assert!(CompilerSet::parse("gcc").is_err());
    }

    #[test]
    fn compiler_set_to_spec() {
        let c = CompilerSet { peano: true, chess: true };
        assert_eq!(c.to_spec(), "peano,chess");
    }

    // --- SuiteSet tests ---

    #[test]
    fn suite_set_npu_xrt_only() {
        let s = SuiteSet::parse("npu-xrt").unwrap();
        assert!(s.npu_xrt);
        assert!(!s.examples);
        assert!(!s.unit_tests);
    }

    #[test]
    fn suite_set_all() {
        let s = SuiteSet::parse("all").unwrap();
        assert!(s.npu_xrt);
        assert!(s.examples);
        assert!(s.unit_tests);
    }

    #[test]
    fn suite_set_all_minus_examples() {
        let s = SuiteSet::parse("all,-examples").unwrap();
        assert!(s.npu_xrt);
        assert!(!s.examples);
        assert!(s.unit_tests);
    }

    #[test]
    fn suite_set_multi() {
        let s = SuiteSet::parse("npu-xrt,examples").unwrap();
        assert!(s.npu_xrt);
        assert!(s.examples);
        assert!(!s.unit_tests);
    }

    #[test]
    fn suite_set_empty_is_error() {
        assert!(SuiteSet::parse("").is_err());
    }

    #[test]
    fn suite_set_unknown_is_error() {
        assert!(SuiteSet::parse("integration").is_err());
    }

    #[test]
    fn suite_set_to_spec() {
        let s = SuiteSet { npu_xrt: true, examples: true, unit_tests: false };
        assert_eq!(s.to_spec(), "npu-xrt,examples");
    }

    // --- DefaultsConfig tests ---

    #[test]
    fn defaults_config_default_parses() {
        let cfg = DefaultsConfig::default();
        let r = RuntimeSet::parse(&cfg.runtime).unwrap();
        assert!(r.emu);
        assert!(!r.hw);
        let c = CompilerSet::parse(&cfg.compiler).unwrap();
        assert!(c.peano);
        assert!(!c.chess);
        let s = SuiteSet::parse(&cfg.suite).unwrap();
        assert!(s.npu_xrt);
        assert!(!s.examples);
    }
}
