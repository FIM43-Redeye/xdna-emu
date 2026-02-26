//! Triple trace comparison pipeline: NPU hardware + emulator + aiesimulator.
//!
//! Orchestrates parallel trace collection from three independent sources for
//! the same AIE2 test, enabling cycle-accurate comparison:
//!
//! 1. **NPU Hardware** -- real silicon via trace-injected MLIR (ground truth)
//! 2. **xdna-emu Emulator** -- reuses the traced xclbin (trace injection is
//!    passive packet-sniffing that doesn't affect kernel behavior)
//! 3. **aiesimulator** -- AMD's functional sim via VCD (opt-in, requires
//!    separate Chess build with `--aiesim` for the `.prj/sim/` directory)
//!
//! All three produce Perfetto JSON for visualization at ui.perfetto.dev.
//!
//! # Output structure
//!
//! ```text
//! build/traces/<test>/
//!   hw-trace.json        # NPU hardware packet traces
//!   emu-trace.json       # Emulator event traces
//!   sim-trace.json       # aiesimulator VCD traces (opt-in)
//!   combined-trace.json  # All sources merged, PID-separated
//!
//! build/normal-chess/<test>/         # Only when --aiesim-trace
//!   aie.xclbin           # Chess build with --aiesim (for aiesimulator)
//!   insts.bin
//!   .source-hash         # Staleness check
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::Config;
use crate::integration::aietools::AieTools;
use crate::testing::lit_trace::{
    self, SubprocessEnv, TraceConfig, TraceOutcome,
    copy_kernel_sources, compile_kernels, is_up_to_date,
    write_hashes, read_extra_aiecc_flags,
};
use crate::testing::process_control::{spawn_with_timeout, ProcessOutcome};
use crate::testing::test_cpp_parser;
use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};
use crate::trace;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the triple trace comparison pipeline.
pub struct TraceCompareConfig {
    /// Base trace config (reused for the HW trace pipeline).
    pub trace: TraceConfig,
    /// Enable aiesimulator VCD trace collection.
    pub aiesim_trace: bool,
    /// Emulator cycle limit (0 = TDR-only).
    pub max_cycles: u64,
    /// Build directory for normal (non-traced) Chess builds.
    pub normal_build_dir: PathBuf,
}

/// Outcome of a single test's trace comparison run.
pub struct TraceCompareOutcome {
    pub test_name: String,
    /// HW trace result (None if HW trace was skipped/failed).
    pub hw_trace: Option<PathBuf>,
    /// Emulator trace result (None if emulator failed).
    pub emu_trace: Option<PathBuf>,
    /// aiesimulator trace result (None if disabled or failed).
    pub sim_trace: Option<PathBuf>,
    /// Combined merged trace (None if merge failed).
    pub combined: Option<PathBuf>,
    /// Per-source error messages.
    pub errors: Vec<String>,
}

/// Summary of a full trace comparison run.
pub struct TraceCompareSummary {
    pub total: usize,
    pub hw_success: usize,
    pub emu_success: usize,
    pub sim_success: usize,
    pub merge_success: usize,
    pub skipped: usize,
    pub hw_fail: usize,
    pub emu_fail: usize,
    pub sim_fail: usize,
    pub wedged: bool,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run the triple trace pipeline across a set of tests.
///
/// For each test:
/// 1. Run the existing HW trace pipeline (inject + compile + execute on NPU)
/// 2. Run the emulator on the traced xclbin and export Perfetto JSON
/// 3. Optionally build a separate Chess xclbin with `--aiesim` and run
///    aiesimulator to collect VCD traces (converted to Perfetto JSON)
/// 4. Merge available traces into a combined Perfetto file
pub fn run_trace_compare(
    tests: &[(String, PathBuf)],
    config: &TraceCompareConfig,
) -> TraceCompareSummary {
    let env = SubprocessEnv::from_config(&config.trace);

    let total = tests.len();
    let mut hw_success = 0usize;
    let mut emu_success = 0usize;
    let mut sim_success = 0usize;
    let mut merge_success = 0usize;
    let mut skipped = 0usize;
    let mut hw_fail = 0usize;
    let mut emu_fail = 0usize;
    let mut sim_fail = 0usize;
    let mut wedged = false;

    for (i, (name, source_dir)) in tests.iter().enumerate() {
        if wedged {
            println!(
                "[{:>3}/{}] {:<40}  SKIPPED (device wedged)",
                i + 1, total, name,
            );
            skipped += 1;
            continue;
        }

        let start = std::time::Instant::now();
        let mut errors: Vec<String> = Vec::new();
        let trace_output_dir = config.trace.traces_dir.join(name);
        fs::create_dir_all(&trace_output_dir).ok();

        // -- Stage 1: HW trace (existing pipeline) --
        let hw_trace = match lit_trace::run_trace_pipeline(name, source_dir, &config.trace) {
            TraceOutcome::Success { trace_json, .. } => {
                // Rename to hw-trace.json in the traces dir
                let hw_dest = trace_output_dir.join("hw-trace.json");
                if trace_json != hw_dest {
                    fs::copy(&trace_json, &hw_dest).ok();
                }
                hw_success += 1;
                Some(hw_dest)
            }
            TraceOutcome::Skipped { reason } => {
                errors.push(format!("HW: skipped ({})", reason));
                skipped += 1;
                // Skip the whole test if it is blocklisted
                let elapsed = start.elapsed().as_secs_f64();
                println!(
                    "[{:>3}/{}] {:<40}  SKIPPED ({}) ({:.1}s)",
                    i + 1, total, name, reason, elapsed,
                );
                continue;
            }
            TraceOutcome::Wedged => {
                wedged = true;
                errors.push("HW: DEVICE WEDGED".to_string());
                eprintln!("DEVICE WEDGED -- stopping all hardware tests");
                hw_fail += 1;
                None
            }
            other => {
                let msg = match other {
                    TraceOutcome::InjectFailed { stderr } => format!("HW inject: {}", first_lines(&stderr, 2)),
                    TraceOutcome::CompileFailed { stderr } => format!("HW compile: {}", first_lines(&stderr, 2)),
                    TraceOutcome::RunFailed { stderr } => format!("HW run: {}", first_lines(&stderr, 2)),
                    TraceOutcome::Timeout { stage } => format!("HW timeout: {}", stage),
                    _ => "HW: unknown error".to_string(),
                };
                errors.push(msg);
                hw_fail += 1;
                None
            }
        };

        // -- Stage 2: Emulator trace (reuses traced xclbin) --
        // The traced xclbin is functionally identical for emulation --
        // trace injection only adds passive packet-sniffing infrastructure
        // (monitor DMA channels, stream routes for trace data) that the
        // emulator either executes harmlessly or ignores.
        let traced_xclbin = config.trace.build_traced_dir.join(name).join("aie.xclbin");
        let traced_insts = config.trace.build_traced_dir.join(name).join("insts.bin");

        let emu_trace = if traced_xclbin.exists() {
            match run_emulator_trace(
                name,
                &traced_xclbin,
                &traced_insts,
                source_dir,
                &trace_output_dir,
                config.max_cycles,
            ) {
                Ok(path) => {
                    emu_success += 1;
                    Some(path)
                }
                Err(msg) => {
                    errors.push(format!("Emulator: {}", msg));
                    emu_fail += 1;
                    None
                }
            }
        } else {
            errors.push("Emulator: no xclbin (HW trace build failed)".into());
            emu_fail += 1;
            None
        };

        // -- Stage 3: aiesimulator trace (opt-in, needs separate --aiesim build) --
        // aiesimulator requires a .prj/sim/ directory which only --aiesim
        // produces, so we need a separate Chess build for this path only.
        let aiesim_build_ok = if config.aiesim_trace {
            let normal_build = config.normal_build_dir.join(name);
            match compile_normal(
                name, source_dir, &normal_build, config, &env,
            ) {
                Ok(()) => true,
                Err(msg) => {
                    errors.push(format!("aiesim build: {}", msg));
                    false
                }
            }
        } else {
            false
        };

        let sim_trace = if config.aiesim_trace && aiesim_build_ok {
            let app_config = Config::get();
            match AieTools::discover(&app_config) {
                Some(tools) => {
                    // Find the .prj directory in the aiesim build
                    let aiesim_build = config.normal_build_dir.join(name);
                    let prj_dir = find_prj_dir(&aiesim_build);
                    match prj_dir {
                        Some(prj) => {
                            match run_aiesim_trace(
                                name, &prj, &trace_output_dir, &tools,
                            ) {
                                Ok(path) => {
                                    sim_success += 1;
                                    Some(path)
                                }
                                Err(msg) => {
                                    errors.push(format!("aiesim: {}", msg));
                                    sim_fail += 1;
                                    None
                                }
                            }
                        }
                        None => {
                            errors.push("aiesim: no .prj directory found (was --aiesim passed to aiecc.py?)".to_string());
                            sim_fail += 1;
                            None
                        }
                    }
                }
                None => {
                    errors.push("aiesim: aietools not found".to_string());
                    sim_fail += 1;
                    None
                }
            }
        } else {
            None
        };

        // -- Stage 4: Merge available traces --
        let combined = match merge_traces(
            &trace_output_dir,
            hw_trace.as_deref(),
            emu_trace.as_deref(),
            sim_trace.as_deref(),
        ) {
            Ok(path) => {
                merge_success += 1;
                Some(path)
            }
            Err(msg) => {
                errors.push(format!("Merge: {}", msg));
                None
            }
        };

        // -- Summary line --
        let elapsed = start.elapsed().as_secs_f64();
        let hw_label = if hw_trace.is_some() { "HW" } else { "--" };
        let emu_label = if emu_trace.is_some() { "EMU" } else { "---" };
        let sim_label = if sim_trace.is_some() { "SIM" } else { "---" };
        let merge_label = if combined.is_some() { "MERGED" } else { "------" };

        let display_name = if name.len() > 35 {
            format!("{}...", &name[..32])
        } else {
            format!("{:<35}", name)
        };

        println!(
            "[{:>3}/{}] {}  {} {} {} {}  ({:.1}s)",
            i + 1, total, display_name,
            hw_label, emu_label, sim_label, merge_label, elapsed,
        );

        if !errors.is_empty() {
            for err in &errors {
                println!("         {}", err);
            }
        }
    }

    TraceCompareSummary {
        total,
        hw_success,
        emu_success,
        sim_success,
        merge_success,
        skipped,
        hw_fail,
        emu_fail,
        sim_fail,
        wedged,
    }
}

impl TraceCompareSummary {
    /// Print a summary of the trace comparison run.
    pub fn print(&self) {
        println!();
        println!("Triple trace comparison summary:");
        println!("  Total tests: {}", self.total);
        println!("  HW traces:   {} success, {} failed", self.hw_success, self.hw_fail);
        println!("  Emu traces:  {} success, {} failed", self.emu_success, self.emu_fail);
        if self.sim_success > 0 || self.sim_fail > 0 {
            println!("  Sim traces:  {} success, {} failed", self.sim_success, self.sim_fail);
        }
        println!("  Merged:      {}", self.merge_success);
        if self.skipped > 0 {
            println!("  Skipped:     {}", self.skipped);
        }
        if self.wedged {
            println!("  DEVICE WEDGED");
        }
    }

    /// True if any non-skip failures occurred.
    pub fn has_failures(&self) -> bool {
        self.hw_fail > 0 || self.emu_fail > 0 || self.sim_fail > 0 || self.wedged
    }
}

// ---------------------------------------------------------------------------
// aiesimulator Chess build (separate from traced build)
// ---------------------------------------------------------------------------

/// Compile a test with Chess and `--aiesim` for aiesimulator.
///
/// Only needed when `--aiesim-trace` is requested, because aiesimulator
/// requires the `.prj/sim/` directory that only `--aiesim` produces.
/// The emulator reuses the traced xclbin directly and does not need this.
///
/// Produces `aie.xclbin` and `insts.bin` in `build_dir`. Staleness-checked
/// via the same hashing mechanism as the trace pipeline, including a config
/// fingerprint to detect compiler or setting changes.
fn compile_normal(
    test_name: &str,
    upstream_dir: &Path,
    build_dir: &Path,
    config: &TraceCompareConfig,
    env: &SubprocessEnv,
) -> Result<(), String> {
    // Use the source directory hash as the staleness sentinel.
    // We re-use the trace-inject.py path as the "tool hash" since we do not
    // have a separate tool here; the aiecc.py version is captured implicitly
    // via the source hash changing when the toolchain is rebuilt.
    let sentinel = config.trace.xdna_emu_root.join("tools/trace-inject.py");
    let config_fp = format!(
        "compiler=chess;aiesim={}",
        config.aiesim_trace,
    );

    if is_up_to_date(build_dir, upstream_dir, &sentinel, &config_fp) {
        log::debug!("Normal Chess build up-to-date for {}", test_name);
        return Ok(());
    }

    fs::create_dir_all(build_dir)
        .map_err(|e| format!("create build dir: {}", e))?;

    // Copy kernel sources (.cc/.cpp/.h) from upstream
    copy_kernel_sources(upstream_dir, build_dir);

    // Compile kernel .cc files to .o with xchesscc_wrapper
    compile_kernels(build_dir, &config.trace, env)?;

    // Find the MLIR source file (aie.mlir or aie2.py -> generated .mlir)
    let mlir_source = find_mlir_source(upstream_dir, build_dir, &config.trace, env)?;

    // Build aiecc.py command
    let mut cmd = Command::new(&config.trace.python);
    cmd.arg(&config.trace.aiecc)
        .arg("--no-aiesim")
        .arg("--aie-generate-xclbin")
        .arg("--aie-generate-npu-insts")
        .arg("--no-compile-host")
        .arg("--xclbin-name=aie.xclbin")
        .arg("--npu-insts-name=insts.bin")
        .arg("--xchesscc")
        .arg("--xbridge");

    // If aiesim traces requested, pass --aiesim to generate .prj/sim/
    if config.aiesim_trace {
        // Remove --no-aiesim and add --aiesim instead
        // (We need to rebuild the command without --no-aiesim)
        let mut cmd2 = Command::new(&config.trace.python);
        cmd2.arg(&config.trace.aiecc)
            .arg("--aiesim")
            .arg("--aie-generate-xclbin")
            .arg("--aie-generate-npu-insts")
            .arg("--no-compile-host")
            .arg("--xclbin-name=aie.xclbin")
            .arg("--npu-insts-name=insts.bin")
            .arg("--xchesscc")
            .arg("--xbridge");

        // Apply extra flags from any .aiecc-extra-flags file
        for flag in read_extra_aiecc_flags(build_dir) {
            cmd2.arg(&flag);
        }

        cmd2.arg(&mlir_source)
            .current_dir(build_dir);
        env.apply(&mut cmd2);

        return run_aiecc_command(test_name, &mut cmd2, &config.trace, build_dir, upstream_dir, &sentinel, &config_fp);
    }

    // Apply extra flags
    for flag in read_extra_aiecc_flags(build_dir) {
        cmd.arg(&flag);
    }

    cmd.arg(&mlir_source)
        .current_dir(build_dir);
    env.apply(&mut cmd);

    run_aiecc_command(test_name, &mut cmd, &config.trace, build_dir, upstream_dir, &sentinel, &config_fp)
}

/// Execute an aiecc.py command and handle the result.
fn run_aiecc_command(
    test_name: &str,
    cmd: &mut Command,
    trace_config: &TraceConfig,
    build_dir: &Path,
    upstream_dir: &Path,
    sentinel: &Path,
    config_fp: &str,
) -> Result<(), String> {
    match spawn_with_timeout(cmd, trace_config.compile_timeout) {
        ProcessOutcome::Completed { exit_code, stdout, stderr } => {
            if exit_code == 0 {
                write_hashes(build_dir, upstream_dir, sentinel, config_fp);
                Ok(())
            } else {
                // aiecc.py may write errors to stdout or stderr
                let combined = if stderr.trim().is_empty() {
                    first_lines(&stdout, 3)
                } else {
                    first_lines(&stderr, 3)
                };
                Err(format!("aiecc.py failed for {}: {}", test_name, combined))
            }
        }
        ProcessOutcome::Timeout { .. } => {
            Err(format!("compile timed out: {}", test_name))
        }
        ProcessOutcome::Wedged { .. } => {
            Err("device wedged during compile".to_string())
        }
        ProcessOutcome::SpawnError(msg) => {
            Err(format!("spawn error: {}", msg))
        }
    }
}

/// Prepare the MLIR source file in the build directory.
///
/// For aie.mlir: copies and substitutes the `NPUDEVICE` placeholder with the
/// actual device target (e.g., `npu1`). This placeholder is used by lit's
/// test infrastructure and must be resolved before aiecc.py can parse the IR.
///
/// For aie2.py: runs the Python generator to produce MLIR output, then writes
/// the result to `aie2.mlir` in the build directory. aiecc.py expects MLIR
/// input, not Python -- the lit RUN lines show a two-step process:
///   `%python aie2.py > ./aie2.mlir` then `aiecc.py ... ./aie2.mlir`
fn find_mlir_source(
    upstream_dir: &Path,
    build_dir: &Path,
    trace_config: &TraceConfig,
    env: &SubprocessEnv,
) -> Result<PathBuf, String> {
    let aie_mlir = upstream_dir.join("aie.mlir");
    if aie_mlir.exists() {
        let dest = build_dir.join("aie.mlir");
        if !dest.exists() {
            // Read, substitute NPUDEVICE, and write (not symlink) so
            // aiecc.py sees valid MLIR without the lit placeholder.
            let content = fs::read_to_string(&aie_mlir)
                .map_err(|e| format!("read aie.mlir: {}", e))?;
            let resolved = resolve_npudevice(&content);
            fs::write(&dest, &resolved)
                .map_err(|e| format!("write aie.mlir: {}", e))?;
        }
        return Ok(dest);
    }

    let aie2_py = upstream_dir.join("aie2.py");
    if aie2_py.exists() {
        let dest = build_dir.join("aie2.mlir");
        if !dest.exists() {
            // Run the Python generator to produce MLIR on stdout.
            let mut cmd = Command::new(&trace_config.python);
            cmd.arg(&aie2_py)
                .current_dir(build_dir);
            env.apply(&mut cmd);

            let output = cmd.output()
                .map_err(|e| format!("run aie2.py: {}", e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "aie2.py failed: {}",
                    first_lines(&stderr, 3),
                ));
            }

            let mlir_output = String::from_utf8_lossy(&output.stdout);
            if mlir_output.trim().is_empty() {
                return Err("aie2.py produced empty output".to_string());
            }
            fs::write(&dest, mlir_output.as_bytes())
                .map_err(|e| format!("write aie2.mlir: {}", e))?;
        }
        return Ok(dest);
    }

    Err(format!(
        "No aie.mlir or aie2.py found in {}",
        upstream_dir.display()
    ))
}

/// Replace the `NPUDEVICE` lit placeholder with an auto-detected device target.
///
/// Mirrors the logic in trace-inject.py: scan tile coordinates to determine
/// the minimum column count, then pick the most specific npu1 variant.
fn resolve_npudevice(mlir_text: &str) -> String {
    if !mlir_text.contains("NPUDEVICE") {
        return mlir_text.to_string();
    }

    // Auto-detect: find max column used in aie.tile(col, row) calls
    let tile_re = regex::Regex::new(r"aie\.tile\(\s*(\d+)\s*,").unwrap();
    let max_col = tile_re.captures_iter(mlir_text)
        .filter_map(|cap| cap[1].parse::<u32>().ok())
        .max()
        .unwrap_or(0);

    let device = match max_col {
        0 => "npu1_1col",
        1 => "npu1_2col",
        2 => "npu1_3col",
        3 => "npu1_4col",
        _ => "npu1",
    };

    mlir_text.replace("NPUDEVICE", device)
}

// ---------------------------------------------------------------------------
// Stage 2: Emulator trace
// ---------------------------------------------------------------------------

/// Run the emulator on a normal xclbin and export Perfetto trace JSON.
///
/// Parses `test.cpp` for buffer layout, constructs an XclbinTest, runs
/// through XclbinSuite, and writes `emu-trace.json` from the trace events.
fn run_emulator_trace(
    test_name: &str,
    xclbin_path: &Path,
    insts_path: &Path,
    upstream_dir: &Path,
    output_dir: &Path,
    max_cycles: u64,
) -> Result<PathBuf, String> {
    if !xclbin_path.exists() {
        return Err(format!("xclbin not found: {}", xclbin_path.display()));
    }

    // Parse test.cpp for buffer spec
    let buffer_spec = test_cpp_parser::parse_test_cpp(upstream_dir);

    // Build test object
    let mut test = XclbinTest::from_path(xclbin_path);
    test.name = test_name.to_string();
    if insts_path.exists() {
        test.insts_path = Some(insts_path.to_path_buf());
    }
    if let Some(spec) = buffer_spec {
        test.buffer_spec = Some(spec);
    }

    // Create a minimal suite and run the test
    let suite = XclbinSuite::new().with_max_cycles(max_cycles);
    let (outcome, _raw_output, trace_events, binary_trace) = suite.run_single_with_trace(&test);

    // Check if the test at least started (we want traces even on validation fail)
    match &outcome {
        crate::testing::xclbin_suite::TestOutcome::LoadError { message } => {
            return Err(format!("load error: {}", message));
        }
        crate::testing::xclbin_suite::TestOutcome::Skipped { reason } => {
            return Err(format!("skipped: {}", reason));
        }
        _ => {
            // Pass, ValidationFail, Timeout, etc. are all fine --
            // we want traces regardless of correctness.
        }
    }

    if trace_events.is_empty() {
        return Err("no trace events collected".to_string());
    }

    // Write binary trace buffer if available (raw packets from trace units)
    if let Some(ref trace_data) = binary_trace {
        let bin_path = output_dir.join("emu-trace_raw.bin");
        fs::write(&bin_path, trace_data)
            .map_err(|e| format!("write emu-trace_raw.bin: {}", e))?;
        // Count non-zero bytes to report meaningful data size
        let non_zero = trace_data.iter().filter(|&&b| b != 0).count();
        log::info!(
            "{}: wrote {} bytes binary trace ({} non-zero) -> {}",
            test_name, trace_data.len(), non_zero, bin_path.display()
        );
    }

    // Export to Perfetto JSON
    let emu_trace_path = output_dir.join("emu-trace.json");
    let mut file = fs::File::create(&emu_trace_path)
        .map_err(|e| format!("create emu-trace.json: {}", e))?;
    trace::export_perfetto(&trace_events, &mut file)
        .map_err(|e| format!("write emu-trace.json: {}", e))?;

    log::info!(
        "{}: emulator produced {} trace events -> {}",
        test_name,
        trace_events.len(),
        emu_trace_path.display()
    );

    Ok(emu_trace_path)
}

// ---------------------------------------------------------------------------
// Stage 3: aiesimulator trace
// ---------------------------------------------------------------------------

/// Run aiesimulator on the normal Chess build and convert VCD to Perfetto.
///
/// The normal build must have been compiled with `--aiesim` to produce
/// a `.prj/sim/` directory. aiesimulator writes VCD files that we convert
/// to Perfetto JSON.
fn run_aiesim_trace(
    test_name: &str,
    prj_dir: &Path,
    output_dir: &Path,
    tools: &AieTools,
) -> Result<PathBuf, String> {
    use crate::integration::aiesimulator;

    // Run aiesimulator (already passes --dump-vcd in run_simulation)
    let sim_result = aiesimulator::run_simulation(
        tools,
        prj_dir,
        &[],         // No input data (test.cpp handles IO via ps.so)
        super::runner_config::DEFAULT_MAX_CYCLES,
    )?;

    if sim_result.exit_code != 0 {
        return Err(format!(
            "aiesimulator exited with code {}: {}",
            sim_result.exit_code,
            first_lines(&sim_result.stderr, 3),
        ));
    }

    // Find the Perfetto JSON that convert_vcd_traces() already created
    // alongside the VCD file in the simulation working directory.
    let work_dir = prj_dir.parent().unwrap_or(Path::new("."));
    let perfetto_json = find_perfetto_json(work_dir);

    match perfetto_json {
        Some(src) => {
            let dest = output_dir.join("sim-trace.json");
            fs::copy(&src, &dest)
                .map_err(|e| format!("copy sim trace: {}", e))?;
            log::info!("{}: aiesimulator trace -> {}", test_name, dest.display());
            Ok(dest)
        }
        None => Err("no VCD-derived Perfetto JSON found after aiesimulator run".to_string()),
    }
}

/// Search for a .perfetto.json file in a directory (produced by VCD conversion).
fn find_perfetto_json(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name()?.to_string_lossy();
            if name.contains("perfetto") {
                return Some(path);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Stage 4: Merge traces
// ---------------------------------------------------------------------------

/// Merge available trace files into a combined Perfetto JSON.
///
/// PID ranges:
/// - 0-99: NPU hardware (unchanged)
/// - 100-199: Emulator (+100)
/// - 200-299: aiesimulator (+200)
fn merge_traces(
    output_dir: &Path,
    hw_trace: Option<&Path>,
    emu_trace: Option<&Path>,
    sim_trace: Option<&Path>,
) -> Result<PathBuf, String> {
    let mut sources_available = 0u32;
    if hw_trace.is_some() { sources_available += 1; }
    if emu_trace.is_some() { sources_available += 1; }
    if sim_trace.is_some() { sources_available += 1; }

    if sources_available == 0 {
        return Err("no trace sources available to merge".to_string());
    }

    // If only one source, just copy it as "combined"
    if sources_available == 1 {
        let src = hw_trace.or(emu_trace).or(sim_trace).unwrap();
        let dest = output_dir.join("combined-trace.json");
        fs::copy(src, &dest)
            .map_err(|e| format!("copy single trace: {}", e))?;
        return Ok(dest);
    }

    // Read each available source, offset PIDs, and concatenate into one array
    let mut all_events = String::from("[\n");
    let mut first_source = true;

    if let Some(path) = hw_trace {
        if let Ok(json) = fs::read_to_string(path) {
            let stripped = strip_json_array_brackets(&json);
            if !stripped.trim().is_empty() {
                if !first_source { all_events.push_str(",\n"); }
                // HW traces keep PID 0-based, just prefix the names
                let adjusted = trace::offset_perfetto_pids(&stripped, 0, "NPU: ");
                all_events.push_str(adjusted.trim());
                first_source = false;
            }
        }
    }

    if let Some(path) = emu_trace {
        if let Ok(json) = fs::read_to_string(path) {
            let stripped = strip_json_array_brackets(&json);
            if !stripped.trim().is_empty() {
                if !first_source { all_events.push_str(",\n"); }
                let adjusted = trace::offset_perfetto_pids(&stripped, 100, "Emulator: ");
                all_events.push_str(adjusted.trim());
                first_source = false;
            }
        }
    }

    if let Some(path) = sim_trace {
        if let Ok(json) = fs::read_to_string(path) {
            let stripped = strip_json_array_brackets(&json);
            if !stripped.trim().is_empty() {
                if !first_source { all_events.push_str(",\n"); }
                let adjusted = trace::offset_perfetto_pids(&stripped, 200, "aiesimulator: ");
                all_events.push_str(adjusted.trim());
            }
        }
    }

    all_events.push_str("\n]\n");

    let combined_path = output_dir.join("combined-trace.json");
    fs::write(&combined_path, &all_events)
        .map_err(|e| format!("write combined trace: {}", e))?;

    Ok(combined_path)
}

/// Strip the outer `[` and `]` brackets from a JSON array string.
///
/// Returns the inner content (the comma-separated event objects) so
/// multiple arrays can be concatenated into a single merged array.
fn strip_json_array_brackets(json: &str) -> &str {
    let trimmed = json.trim();
    let start = trimmed.find('[').map(|i| i + 1).unwrap_or(0);
    let end = trimmed.rfind(']').unwrap_or(trimmed.len());
    if start < end {
        &trimmed[start..end]
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find a .prj directory in a build output directory.
fn find_prj_dir(build_dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(build_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().ends_with(".prj") {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Take the first N lines from a string for error message previews.
fn first_lines(s: &str, n: usize) -> String {
    s.lines().take(n).collect::<Vec<_>>().join(" | ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_json_array_brackets() {
        let json = r#"[
{"name":"foo"},
{"name":"bar"}
]"#;
        let stripped = strip_json_array_brackets(json);
        assert!(stripped.contains(r#"{"name":"foo"}"#));
        assert!(stripped.contains(r#"{"name":"bar"}"#));
        assert!(!stripped.contains('['));
        assert!(!stripped.contains(']'));
    }

    #[test]
    fn test_strip_json_array_brackets_empty() {
        assert_eq!(strip_json_array_brackets("[]").trim(), "");
        assert_eq!(strip_json_array_brackets("[\n]").trim(), "");
    }

    #[test]
    fn test_first_lines() {
        assert_eq!(first_lines("a\nb\nc\nd", 2), "a | b");
        assert_eq!(first_lines("single", 5), "single");
        assert_eq!(first_lines("", 3), "");
    }

    #[test]
    fn test_merge_traces_single_source() {
        let tmp = std::env::temp_dir().join("test_merge_single");
        fs::create_dir_all(&tmp).ok();

        let hw = tmp.join("hw-trace.json");
        fs::write(&hw, r#"[{"name":"test","pid":0}]"#).unwrap();

        let result = merge_traces(&tmp, Some(&hw), None, None);
        assert!(result.is_ok());
        let combined = result.unwrap();
        assert!(combined.exists());

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_merge_traces_no_sources() {
        let tmp = std::env::temp_dir().join("test_merge_none");
        fs::create_dir_all(&tmp).ok();

        let result = merge_traces(&tmp, None, None, None);
        assert!(result.is_err());

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_resolve_npudevice_auto() {
        let mlir = r#"module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
  }
}"#;
        let resolved = resolve_npudevice(mlir);
        assert!(resolved.contains("npu1_1col"), "Single column should map to npu1_1col");
        assert!(!resolved.contains("NPUDEVICE"));
    }

    #[test]
    fn test_resolve_npudevice_multi_col() {
        let mlir = r#"aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 2)
    %t10 = aie.tile(1, 2)
    %t20 = aie.tile(2, 2)
}"#;
        let resolved = resolve_npudevice(mlir);
        assert!(resolved.contains("npu1_3col"));
    }

    #[test]
    fn test_resolve_npudevice_passthrough() {
        let mlir = r#"aie.device(npu1) { }"#;
        let resolved = resolve_npudevice(mlir);
        assert_eq!(resolved, mlir, "Should not modify MLIR without NPUDEVICE");
    }

    #[test]
    fn test_merge_traces_two_sources() {
        let tmp = std::env::temp_dir().join("test_merge_two");
        fs::create_dir_all(&tmp).ok();

        let hw = tmp.join("hw-trace.json");
        let emu = tmp.join("emu-trace.json");
        fs::write(&hw, r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"INSTR_LOAD","ph":"B","pid":0,"tid":1,"ts":10}
]"#).unwrap();
        fs::write(&emu, r#"[
{"name":"process_name","ph":"M","pid":0,"args":{"name":"core_trace for tile2,0"}},
{"name":"INSTR_LOAD","ph":"B","pid":0,"tid":1,"ts":12}
]"#).unwrap();

        let result = merge_traces(&tmp, Some(&hw), Some(&emu), None);
        assert!(result.is_ok());

        let combined = fs::read_to_string(result.unwrap()).unwrap();
        // HW: pid stays 0 (with "NPU: " prefix on process_name)
        assert!(combined.contains("NPU: "));
        // Emulator: pid shifted to 100
        assert!(combined.contains(r#""pid":100"#));
        assert!(combined.contains("Emulator: "));

        fs::remove_dir_all(&tmp).ok();
    }
}
