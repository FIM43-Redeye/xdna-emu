//! Parallel build orchestrator with ncurses-style grid display.
//!
//! **Status: integrated into `npu-test` (default emu+hw mode).**
//! Called from `emu_runner::run()` for parallel test builds.
//!
//! Replaces the sequential `batch_build_primary()` and `batch_build_chess_comparison()`
//! with a parallel build system that shows progress as a compact, color-coded grid
//! using crossterm for terminal control.
//!
//! Each test is one cell in the grid. Cells show their build state:
//! - `X` dim: not started
//! - `.` yellow: building
//! - `P` green: Peano passed
//! - `C` green: Chess passed
//! - `O` green: both passed
//! - `X` red: failed
//! - Flickering cells: one track done, other still building

use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crossterm::{cursor, execute, style, terminal};

use crate::integration::chess_build::{BuildEnv, BuildOpts, BuildResult};
use crate::testing::xclbin_suite::{Compiler, XclbinTest};

/// Extract a short, human-readable diagnostic from a build error string.
///
/// Scans through the log looking for recognizable error patterns and returns
/// the first meaningful one. Falls back to the first line of the error.
fn extract_build_diagnostic(error: &str) -> String {
    // Scan all lines for recognizable error patterns, in priority order.
    for line in error.lines() {
        let trimmed = line.trim();

        // LLVM fatal error (Peano backend crash / legalization failure)
        if let Some(msg) = trimmed.strip_prefix("LLVM ERROR: ") {
            // Truncate to keep the one-liner readable.
            let short = if msg.len() > 80 { &msg[..80] } else { msg };
            return format!("LLVM: {}", short);
        }

        // MLIR diagnostic (aie-opt / aie-translate errors)
        if trimmed.contains("error:") && trimmed.contains(".mlir") {
            // e.g. "aie_arch.mlir:42:5: error: op violates constraint"
            // Extract just the error portion after the file:line:col prefix.
            if let Some(pos) = trimmed.find("error:") {
                let msg = trimmed[pos..].trim();
                let short = if msg.len() > 80 { &msg[..80] } else { msg };
                return short.to_string();
            }
        }

        // Python exception (aiecc.py or build script failure)
        // Match common Python error classes at start of line.
        if trimmed.starts_with("ModuleNotFoundError:")
            || trimmed.starts_with("ImportError:")
            || trimmed.starts_with("FileNotFoundError:")
            || trimmed.starts_with("RuntimeError:")
            || trimmed.starts_with("ValueError:")
            || trimmed.starts_with("TypeError:")
            || trimmed.starts_with("KeyError:")
            || trimmed.starts_with("subprocess.CalledProcessError:")
        {
            let short = if trimmed.len() > 80 { &trimmed[..80] } else { trimmed };
            return format!("Python: {}", short);
        }

        // make error
        if trimmed.starts_with("make:") && trimmed.contains("***") {
            let short = if trimmed.len() > 80 { &trimmed[..80] } else { trimmed };
            return short.to_string();
        }
    }

    // Fallback: first line of the error string (the "step N/M failed" header).
    error.lines().next().unwrap_or(error).to_string()
}

/// Save a build log to `build/logs/<name>-<track>.log`, overwriting any
/// previous log for the same test+track.
///
/// Name is sanitized: slashes become dashes. The logs directory is created
/// on first call.
fn save_build_log(name: &str, track: &str, content: &str) {
    let log_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("build/logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let sanitized = name.replace('/', "-");
    let log_path = log_dir.join(format!("{}-{}.log", sanitized, track.to_lowercase()));
    if let Err(e) = std::fs::write(&log_path, content) {
        log::warn!("Failed to write build log {}: {}", log_path.display(), e);
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Which compiler tracks a test will be built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildTracks {
    /// Peano-primary test (no Chess comparison build).
    PeanoOnly,
    /// Chess-only test (REQUIRES: chess).
    ChessOnly,
    /// Peano primary + Chess comparison build.
    Both,
}

/// Result state for a single compiler track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackResult {
    NotStarted,
    Building,
    Passed,
    Failed,
}

/// Per-cell state in the build grid.
#[derive(Debug, Clone)]
pub struct CellState {
    pub name: String,
    pub tracks: BuildTracks,
    pub peano: TrackResult,
    pub chess: TrackResult,
}

impl CellState {
    /// Determine the display character and color for this cell.
    ///
    /// `flicker_phase` alternates every 250ms to create a visual pulse
    /// when one track is done but the other is still building.
    pub fn display_char(&self, flicker_phase: bool) -> (char, style::Color) {
        use BuildTracks::*;
        use TrackResult::*;

        match self.tracks {
            PeanoOnly => match self.peano {
                NotStarted => ('X', style::Color::DarkGrey),
                Building => ('.', style::Color::Yellow),
                Passed => ('P', style::Color::Green),
                Failed => ('X', style::Color::Red),
            },
            ChessOnly => match self.chess {
                NotStarted => ('X', style::Color::DarkGrey),
                Building => ('.', style::Color::Yellow),
                Passed => ('C', style::Color::Green),
                Failed => ('X', style::Color::Red),
            },
            Both => match (self.peano, self.chess) {
                (NotStarted, NotStarted) => ('X', style::Color::DarkGrey),
                // One or both building, neither done
                (Building, NotStarted) | (NotStarted, Building) |
                (Building, Building) => ('.', style::Color::Yellow),
                // Peano done, Chess building -- flicker
                (Passed, Building) => if flicker_phase {
                    ('P', style::Color::Green)
                } else {
                    ('.', style::Color::Yellow)
                },
                (Failed, Building) => if flicker_phase {
                    ('P', style::Color::Red)
                } else {
                    ('.', style::Color::Yellow)
                },
                // Chess done, Peano building -- flicker
                (Building, Passed) => if flicker_phase {
                    ('.', style::Color::Yellow)
                } else {
                    ('C', style::Color::Green)
                },
                (Building, Failed) => if flicker_phase {
                    ('.', style::Color::Yellow)
                } else {
                    ('C', style::Color::Red)
                },
                // Both done
                (Passed, Passed) => ('O', style::Color::Green),
                (Passed, NotStarted) => ('P', style::Color::Green),
                (NotStarted, Passed) => ('C', style::Color::Green),
                (Failed, Failed) => ('X', style::Color::Red),
                (Passed, Failed) => ('P', style::Color::Yellow),
                (Failed, Passed) => ('C', style::Color::Yellow),
                // Edge cases: one failed, other not started
                (Failed, NotStarted) => ('X', style::Color::Red),
                (NotStarted, Failed) => ('X', style::Color::Red),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Buildable trait
// ---------------------------------------------------------------------------

/// Trait for anything the parallel build system can build.
///
/// Implementations provide the build-specific logic (npu-xrt RUN lines
/// vs programming_example Makefiles vs future sources), while the
/// orchestrator handles parallelism, progress display, and result collection.
pub trait Buildable: Send + Sync {
    /// Human-readable test name (used in display and result keys).
    fn name(&self) -> &str;

    /// Whether this test requires Chess (and cannot use Peano).
    fn requires_chess(&self) -> bool;

    /// Whether this test should be skipped entirely (e.g. requires NPU2).
    fn skip_reason(&self) -> Option<&str>;

    /// Whether this test has anything to build (false = no build steps).
    fn can_build(&self) -> bool;

    /// Execute the build for a given compiler.
    ///
    /// `output_dir` is determined by the orchestrator (compiler-specific).
    /// `build_env` provides environment, cache checks, and tool paths.
    fn build(
        &self,
        build_env: &BuildEnv,
        output_dir: &Path,
        opts: &BuildOpts,
    ) -> Result<BuildResult, String>;

    /// Determine the output directory for this test and compiler.
    fn output_dir(&self, use_chess: bool) -> PathBuf;

    /// Collect result artifacts from a successful build.
    fn collect_artifacts(
        &self,
        output_dir: &Path,
        result: &BuildResult,
    ) -> Vec<BuiltArtifact>;

    /// Enrich an XclbinTest with source-level metadata (buffer_spec, etc.).
    fn enrich_test(&self, test: &mut XclbinTest);
}

// ---------------------------------------------------------------------------
// Grid layout
// ---------------------------------------------------------------------------

/// Grid dimensions for the progress display.
#[derive(Debug, Clone, Copy)]
pub struct GridLayout {
    pub cols: usize,
    pub rows: usize,
    pub total: usize,
}

impl GridLayout {
    /// Compute a roughly square grid layout for `n` cells.
    ///
    /// Uses ceil(sqrt(n)) columns and ceil(n/cols) rows.
    pub fn new(n: usize) -> Self {
        if n == 0 {
            return Self { cols: 0, rows: 0, total: 0 };
        }
        let cols = (n as f64).sqrt().ceil() as usize;
        let rows = (n + cols - 1) / cols;
        Self { cols, rows, total: n }
    }

    /// Total character width needed: "[ " + cells ("X " each) + "]".
    pub fn display_width(&self) -> usize {
        2 + self.cols * 2 + 1
    }

    /// Total rows needed for the grid alone.
    pub fn display_height(&self) -> usize {
        self.rows
    }
}

// ---------------------------------------------------------------------------
// Shared progress state
// ---------------------------------------------------------------------------

/// Thread-safe container for build progress, shared between workers and renderer.
pub struct BuildProgress {
    cells: Vec<Mutex<CellState>>,
    pub layout: GridLayout,
    pub completed: AtomicUsize,
    pub total_builds: usize,
    pub start_time: Instant,
    pub done: AtomicBool,
}

impl BuildProgress {
    /// Create progress tracker from the test/track plan.
    fn new(cells: Vec<CellState>, total_builds: usize) -> Self {
        let layout = GridLayout::new(cells.len());
        let cells = cells.into_iter().map(Mutex::new).collect();
        Self {
            cells,
            layout,
            completed: AtomicUsize::new(0),
            total_builds,
            start_time: Instant::now(),
            done: AtomicBool::new(false),
        }
    }

    /// Read a snapshot of cell state (brief lock, struct copy).
    fn get_cell(&self, idx: usize) -> CellState {
        self.cells[idx].lock().unwrap().clone()
    }

    /// Update a cell's track result. Called by workers.
    fn update_cell(&self, idx: usize, track: Track, result: TrackResult) {
        let mut cell = self.cells[idx].lock().unwrap();
        match track {
            Track::Peano => cell.peano = result,
            Track::Chess => cell.chess = result,
        }
    }
}

/// Which track a build task targets.
#[derive(Debug, Clone, Copy)]
enum Track {
    Peano,
    Chess,
}

// ---------------------------------------------------------------------------
// Work queue
// ---------------------------------------------------------------------------

/// A single unit of work in the build queue.
struct BuildTask {
    cell_idx: usize,
    track: Track,
    test_idx: usize,
    use_chess: bool,
    gen_sim: bool,
}

// ---------------------------------------------------------------------------
// Terminal rendering
// ---------------------------------------------------------------------------

/// RAII guard that restores cursor visibility on drop.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = execute!(stdout, cursor::Show);
        // Move below the grid area so subsequent output doesn't overwrite
        let _ = execute!(stdout, cursor::MoveToNextLine(1));
    }
}

/// Total lines output per frame: grid rows + blank line + status line.
fn frame_height(layout: &GridLayout) -> u16 {
    layout.rows as u16 + 2
}

/// Render one frame of the grid to stdout.
///
/// Uses relative cursor movement (`MoveUp`) to overwrite the previous frame.
/// This avoids `cursor::position()` / `cursor::MoveTo()` which rely on DSR
/// queries and absolute row numbers -- both fragile across terminal emulators
/// (known issue with Ghostty) and invalidated by scrolling.
fn render_frame(
    progress: &BuildProgress,
    is_first_frame: bool,
    flicker_phase: bool,
) {
    let mut stdout = io::stdout();
    let layout = &progress.layout;

    // Move back up to overwrite the previous frame (skip on first frame).
    if !is_first_frame {
        let lines_up = frame_height(layout);
        let _ = execute!(stdout, cursor::MoveUp(lines_up), cursor::MoveToColumn(0));
    }

    for row in 0..layout.rows {
        let _ = execute!(stdout, style::ResetColor, style::Print("[ "));
        for col in 0..layout.cols {
            let idx = row * layout.cols + col;
            if idx < layout.total {
                let cell = progress.get_cell(idx);
                let (ch, color) = cell.display_char(flicker_phase);
                let _ = execute!(
                    stdout,
                    style::SetForegroundColor(color),
                    style::Print(ch),
                    style::Print(' '),
                );
            } else {
                // Empty cell in last row
                let _ = execute!(stdout, style::Print("  "));
            }
        }
        let _ = execute!(stdout, style::ResetColor, style::Print("]\n"));
    }
    // Blank line between grid and status line
    let _ = execute!(stdout, style::Print('\n'));

    // Status line below grid (clear to end of line in case previous was longer)
    let completed = progress.completed.load(Ordering::Relaxed);
    let elapsed = progress.start_time.elapsed().as_secs_f64();
    let _ = execute!(
        stdout,
        style::ResetColor,
        style::Print(format!(
            "\rBuilding: {}/{} ({:.1}s)    ",
            completed, progress.total_builds, elapsed
        )),
    );
    let _ = stdout.flush();
}

/// Print the legend (once, below the grid area).
fn print_legend() {
    let mut stdout = io::stdout();
    let _ = execute!(stdout, style::ResetColor);
    println!();
    println!(
        "Legend: \
         X=pending  .=building  P=Peano  C=Chess  O=both  \
         green=pass  red=fail  yellow=mixed"
    );
    println!();
}

// ---------------------------------------------------------------------------
// Build result types
// ---------------------------------------------------------------------------

/// Artifacts from a successful build of a single test+compiler pair.
pub struct BuiltArtifact {
    pub test_name: String,
    pub xclbin: PathBuf,
    pub insts: Option<PathBuf>,
    pub prj_dir: Option<PathBuf>,
    pub build_log: String,
}

/// Result of a single build task (success or failure).
enum BuildTaskResult {
    /// Build succeeded, possibly with multiple xclbin variants.
    Success {
        track: Track,
        artifacts: Vec<BuiltArtifact>,
    },
    /// Build failed.
    Failure,
}

/// Aggregated results from the parallel build phase.
pub struct ParallelBuildResult {
    /// XclbinTest entries ready for the emulator (primary builds only).
    pub primary_tests: Vec<XclbinTest>,
    /// Chess comparison artifacts, keyed by test name.
    pub chess_artifacts: HashMap<String, ChessArtifacts>,
}

/// Chess build artifacts for comparison.
pub struct ChessArtifacts {
    pub xclbin: PathBuf,
    pub insts: Option<PathBuf>,
    pub prj_dir: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the parallel build system.
pub struct ParallelBuildConfig {
    /// Number of worker threads (0 = auto-detect).
    pub thread_count: usize,
    /// Nice level for build subprocesses.
    pub nice_level: i32,
    /// Verbose mode: fall back to sequential line output.
    pub verbose: bool,
    /// Generate simulation artifacts (--aiesim).
    pub gen_sim: bool,
    /// Use Chess as the primary compiler for ALL tests (skip Peano).
    pub chess_only: bool,
    /// Force rebuild even if cached artifacts are fresh.
    pub force_rebuild: bool,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Run all builds in parallel with a grid progress display.
///
/// Replaces `batch_build_primary()` + `batch_build_chess_comparison()` with
/// a single parallel invocation. All builds (Peano primary, Chess-only primary,
/// Chess comparison) run concurrently.
///
/// Falls back to sequential line output when:
/// - `config.verbose` is true
/// - stdout is not a terminal (piped)
/// - terminal is too small for the grid
pub fn run_parallel_builds<T: Buildable>(
    build_env: &BuildEnv,
    tests: &[&T],
    chess_available: bool,
    config: &ParallelBuildConfig,
) -> ParallelBuildResult {
    let start_time = Instant::now();

    // Categorize tests and count skips
    let mut skipped_reason = 0usize;
    let mut skipped_no_chess = 0usize;
    let mut skipped_no_steps = 0usize;

    // Build the cell list and work queue
    let mut cells: Vec<CellState> = Vec::new();
    let mut work_queue: Vec<BuildTask> = Vec::new();

    for (test_idx, test) in tests.iter().enumerate() {
        if test.skip_reason().is_some() {
            skipped_reason += 1;
            continue;
        }
        if !test.can_build() {
            skipped_no_steps += 1;
            continue;
        }

        let is_chess_only = config.chess_only || test.requires_chess();
        if is_chess_only && !chess_available {
            skipped_no_chess += 1;
            continue;
        }

        let cell_idx = cells.len();
        let tracks = if is_chess_only {
            BuildTracks::ChessOnly
        } else if chess_available {
            BuildTracks::Both
        } else {
            BuildTracks::PeanoOnly
        };

        cells.push(CellState {
            name: test.name().to_string(),
            tracks,
            peano: TrackResult::NotStarted,
            chess: TrackResult::NotStarted,
        });

        // Queue primary build
        if is_chess_only {
            work_queue.push(BuildTask {
                cell_idx,
                track: Track::Chess,
                test_idx,
                use_chess: true,
                gen_sim: false,
            });
        } else {
            work_queue.push(BuildTask {
                cell_idx,
                track: Track::Peano,
                test_idx,
                use_chess: false,
                gen_sim: false,
            });
        }

        // Queue Chess comparison build for Peano-primary tests
        if !is_chess_only && chess_available {
            work_queue.push(BuildTask {
                cell_idx,
                track: Track::Chess,
                test_idx,
                use_chess: true,
                gen_sim: config.gen_sim,
            });
        }
    }

    let total_buildable = cells.len();
    let total_builds = work_queue.len();
    let total_skipped = skipped_reason + skipped_no_chess + skipped_no_steps;

    println!("\n=== BUILD PHASE ===");
    if config.chess_only {
        println!("Mode: --chess-only (all tests use Chess compiler)");
    }
    if config.force_rebuild {
        println!("Mode: --rebuild (forcing all rebuilds)");
    }
    println!(
        "{} tests ({} builds: {} primary, {} comparison), {} skipped \
         (skip-reason: {}, no chess: {}, no steps: {})",
        total_buildable,
        total_builds,
        total_buildable,
        total_builds - total_buildable,
        total_skipped,
        skipped_reason,
        skipped_no_chess,
        skipped_no_steps,
    );

    if total_buildable == 0 {
        return ParallelBuildResult {
            primary_tests: Vec::new(),
            chess_artifacts: HashMap::new(),
        };
    }

    // Decide display mode: grid vs verbose
    let use_grid = !config.verbose
        && io::stdout().is_terminal()
        && grid_fits_terminal(&GridLayout::new(total_buildable));

    let thread_count = if config.thread_count > 0 {
        config.thread_count
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    };

    let progress = BuildProgress::new(cells, total_builds);
    let work_counter = AtomicUsize::new(0);
    let results: Mutex<Vec<BuildTaskResult>> = Mutex::new(Vec::with_capacity(total_builds));
    let nice = if config.nice_level > 0 { Some(config.nice_level) } else { None };
    let force_rebuild = config.force_rebuild;

    if use_grid {
        run_with_grid(
            build_env, tests, &progress, &work_queue, &work_counter,
            &results, thread_count, nice, force_rebuild,
        );
    } else {
        run_verbose(
            build_env, tests, &progress, &work_queue, &work_counter,
            &results, thread_count, nice, force_rebuild,
        );
    }

    // Collect results
    let task_results = results.into_inner().unwrap();
    collect_results(&task_results, tests, total_builds, start_time)
}

/// Check if the grid fits in the current terminal.
fn grid_fits_terminal(layout: &GridLayout) -> bool {
    match terminal::size() {
        Ok((w, h)) => {
            // Grid needs: width for cells (with brackets) + blank line + status + legend
            layout.display_width() <= w as usize
                && (layout.display_height() + 5) <= h as usize
        }
        Err(_) => false,
    }
}

/// Run builds with the grid display.
fn run_with_grid<T: Buildable>(
    build_env: &BuildEnv,
    tests: &[&T],
    progress: &BuildProgress,
    work_queue: &[BuildTask],
    work_counter: &AtomicUsize,
    results: &Mutex<Vec<BuildTaskResult>>,
    thread_count: usize,
    nice: Option<i32>,
    force_rebuild: bool,
) {
    let mut stdout = io::stdout();

    // Setup: hide cursor, print legend
    let _ = execute!(stdout, cursor::Hide);
    let _guard = TerminalGuard;

    print_legend();

    // Pre-render initial grid (all X dim) -- first frame, no MoveUp
    render_frame(progress, true, false);

    std::thread::scope(|s| {
        // Render thread -- loops until done flag is set.
        // All subsequent frames use MoveUp to overwrite the previous.
        s.spawn(|| {
            let mut flicker_phase = false;
            while !progress.done.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(250));
                flicker_phase = !flicker_phase;
                render_frame(progress, false, flicker_phase);
            }
            // Final render with all cells in terminal state
            render_frame(progress, false, false);
        });

        // Worker threads -- collect handles so we can join them explicitly
        let workers: Vec<_> = (0..thread_count)
            .map(|_| {
                s.spawn(|| {
                    run_worker(
                        build_env, tests, progress, work_queue,
                        work_counter, results, nice, force_rebuild,
                    );
                })
            })
            .collect();

        // Wait for all workers to finish, THEN signal the render thread.
        // This must happen inside the scope -- if we set done after the
        // scope, the scope would block waiting for the render thread,
        // which would block waiting for done. Deadlock.
        for w in workers {
            w.join().unwrap();
        }
        progress.done.store(true, Ordering::Relaxed);
    });

    // Move to a fresh line below the status line
    let _ = execute!(stdout, style::ResetColor, style::Print("\n\n"), cursor::Show);
    let _ = stdout.flush();
}

/// Run builds in parallel with per-completion line output (non-TTY or -v).
fn run_verbose<T: Buildable>(
    build_env: &BuildEnv,
    tests: &[&T],
    progress: &BuildProgress,
    work_queue: &[BuildTask],
    work_counter: &AtomicUsize,
    results: &Mutex<Vec<BuildTaskResult>>,
    thread_count: usize,
    nice: Option<i32>,
    force_rebuild: bool,
) {
    // Parallel with atomic per-build output: each build prints its
    // status line when it completes, so output order reflects completion
    // order rather than submission order.
    eprintln!(
        "Building {} tasks with {} threads...",
        work_queue.len(),
        thread_count,
    );
    std::thread::scope(|s| {
        for _ in 0..thread_count {
            s.spawn(|| {
                run_worker(build_env, tests, progress, work_queue, work_counter, results, nice, force_rebuild);
            });
        }
    });
    eprintln!();
}

/// Worker thread body: pull tasks from atomic counter, execute builds.
fn run_worker<T: Buildable>(
    build_env: &BuildEnv,
    tests: &[&T],
    progress: &BuildProgress,
    work_queue: &[BuildTask],
    work_counter: &AtomicUsize,
    results: &Mutex<Vec<BuildTaskResult>>,
    nice: Option<i32>,
    force_rebuild: bool,
) {
    loop {
        let task_idx = work_counter.fetch_add(1, Ordering::SeqCst);
        if task_idx >= work_queue.len() {
            break;
        }

        let task = &work_queue[task_idx];
        let test = &tests[task.test_idx];
        let name = test.name().to_string();
        let track_name = match task.track {
            Track::Peano => "Peano",
            Track::Chess => "Chess",
        };
        let task_start = Instant::now();

        progress.update_cell(task.cell_idx, task.track, TrackResult::Building);

        let output_dir = test.output_dir(task.use_chess);
        match test.build(
            build_env,
            &output_dir,
            &BuildOpts {
                use_chess: task.use_chess,
                gen_sim: task.gen_sim,
                device: String::new(),
                nice,
                force_rebuild,
            },
        ) {
            Ok(result) => {
                let cached = result.build_log == "(cached)";
                let elapsed = task_start.elapsed();
                let label = if cached { "cached" } else { "built" };
                if !cached {
                    save_build_log(&name, track_name, &result.build_log);
                }
                let completed = progress.completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[{:2}/{}] {:40} {} {} ({:.1}s)",
                    completed, progress.total_builds,
                    &name[..name.len().min(40)],
                    track_name, label, elapsed.as_secs_f64(),
                );

                progress.update_cell(task.cell_idx, task.track, TrackResult::Passed);

                let artifacts = test.collect_artifacts(&output_dir, &result);
                results.lock().unwrap().push(BuildTaskResult::Success {
                    track: task.track,
                    artifacts,
                });
            }
            Err(e) => {
                let elapsed = task_start.elapsed();
                save_build_log(&name, track_name, &e);
                let diag = extract_build_diagnostic(&e);
                let completed = progress.completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[{:2}/{}] {:40} {} FAILED ({:.1}s): {}",
                    completed, progress.total_builds,
                    &name[..name.len().min(40)],
                    track_name, elapsed.as_secs_f64(), diag,
                );
                eprintln!(
                    "         log: build/logs/{}-{}.log",
                    name.replace('/', "-"), track_name.to_lowercase(),
                );

                progress.update_cell(task.cell_idx, task.track, TrackResult::Failed);

                results.lock().unwrap().push(BuildTaskResult::Failure);
            }
        }
    }
}

/// Collect parallel build results into the structures the test runner expects.
fn collect_results<T: Buildable>(
    task_results: &[BuildTaskResult],
    tests: &[&T],
    total_builds: usize,
    start_time: Instant,
) -> ParallelBuildResult {
    let mut primary_tests = Vec::new();
    let mut chess_artifacts = HashMap::new();
    let mut peano_built = 0usize;
    let mut chess_built = 0usize;

    for result in task_results {
        match result {
            BuildTaskResult::Success { track, artifacts } => {
                match track {
                    Track::Peano => {
                        peano_built += 1;
                        for artifact in artifacts {
                            let mut t = XclbinTest::from_path(&artifact.xclbin);
                            t.name = artifact.test_name.clone();
                            t.insts_path = artifact.insts.clone();
                            t.compiler = Some(Compiler::Peano);
                            // Find the original test to enrich with metadata
                            if let Some(src) = tests.iter().find(|s| {
                                artifact.test_name == s.name()
                                    || artifact.test_name.starts_with(&format!("{}/", s.name()))
                            }) {
                                src.enrich_test(&mut t);
                            }
                            primary_tests.push(t);
                        }
                    }
                    Track::Chess => {
                        chess_built += 1;
                        for artifact in artifacts {
                            // For chess-only tests that are primaries, they go
                            // into primary_tests. For chess comparison builds
                            // (Both track), they go into chess_artifacts.
                            let is_primary = tests.iter().any(|s| {
                                (artifact.test_name == s.name()
                                    || artifact.test_name.starts_with(&format!("{}/", s.name())))
                                    && s.requires_chess()
                            });

                            if is_primary {
                                let mut t = XclbinTest::from_path(&artifact.xclbin);
                                t.name = artifact.test_name.clone();
                                t.insts_path = artifact.insts.clone();
                                t.compiler = Some(Compiler::Chess);
                                if let Some(src) = tests.iter().find(|s| {
                                    artifact.test_name == s.name()
                                        || artifact.test_name.starts_with(&format!("{}/", s.name()))
                                }) {
                                    src.enrich_test(&mut t);
                                }
                                primary_tests.push(t);
                            } else {
                                // Chess comparison artifact -- extract the base
                                // test name (strip variant suffix if present)
                                let base_name = if let Some(src) = tests.iter().find(|s| {
                                    artifact.test_name == s.name()
                                        || artifact.test_name.starts_with(&format!("{}/", s.name()))
                                }) {
                                    src.name().to_string()
                                } else {
                                    artifact.test_name.clone()
                                };

                                chess_artifacts.insert(base_name, ChessArtifacts {
                                    xclbin: artifact.xclbin.clone(),
                                    insts: artifact.insts.clone(),
                                    prj_dir: artifact.prj_dir.clone(),
                                });
                            }
                        }
                    }
                }
            }
            BuildTaskResult::Failure { .. } => {}
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let succeeded = peano_built + chess_built;
    println!(
        "Builds: {}/{} succeeded ({} Peano, {} Chess) ({:.1}s)\n",
        succeeded, total_builds, peano_built, chess_built, elapsed,
    );

    // Sort primary tests by name for consistent ordering
    primary_tests.sort_by(|a, b| a.name.cmp(&b.name));

    ParallelBuildResult {
        primary_tests,
        chess_artifacts,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_layout_edge_cases() {
        let g0 = GridLayout::new(0);
        assert_eq!(g0.cols, 0);
        assert_eq!(g0.rows, 0);

        let g1 = GridLayout::new(1);
        assert_eq!(g1.cols, 1);
        assert_eq!(g1.rows, 1);

        let g4 = GridLayout::new(4);
        assert_eq!(g4.cols, 2);
        assert_eq!(g4.rows, 2);

        let g9 = GridLayout::new(9);
        assert_eq!(g9.cols, 3);
        assert_eq!(g9.rows, 3);

        let g10 = GridLayout::new(10);
        assert_eq!(g10.cols, 4); // ceil(sqrt(10)) = 4
        assert_eq!(g10.rows, 3); // ceil(10/4) = 3

        let g67 = GridLayout::new(67);
        // ceil(sqrt(67)) = 9 (since 8^2=64 < 67)
        assert_eq!(g67.cols, 9);
        assert_eq!(g67.rows, 8); // ceil(67/9) = 8 (72 cells, 5 empty)

        let g64 = GridLayout::new(64);
        assert_eq!(g64.cols, 8);
        assert_eq!(g64.rows, 8);

        let g100 = GridLayout::new(100);
        assert_eq!(g100.cols, 10);
        assert_eq!(g100.rows, 10);
    }

    #[test]
    fn display_char_peano_only() {
        let mut cell = CellState {
            name: "test".to_string(),
            tracks: BuildTracks::PeanoOnly,
            peano: TrackResult::NotStarted,
            chess: TrackResult::NotStarted,
        };
        assert_eq!(cell.display_char(false), ('X', style::Color::DarkGrey));

        cell.peano = TrackResult::Building;
        assert_eq!(cell.display_char(false), ('.', style::Color::Yellow));

        cell.peano = TrackResult::Passed;
        assert_eq!(cell.display_char(false), ('P', style::Color::Green));
        // Flicker phase doesn't affect single-track cells
        assert_eq!(cell.display_char(true), ('P', style::Color::Green));

        cell.peano = TrackResult::Failed;
        assert_eq!(cell.display_char(false), ('X', style::Color::Red));
    }

    #[test]
    fn display_char_chess_only() {
        let mut cell = CellState {
            name: "test".to_string(),
            tracks: BuildTracks::ChessOnly,
            peano: TrackResult::NotStarted,
            chess: TrackResult::NotStarted,
        };
        assert_eq!(cell.display_char(false), ('X', style::Color::DarkGrey));

        cell.chess = TrackResult::Building;
        assert_eq!(cell.display_char(false), ('.', style::Color::Yellow));

        cell.chess = TrackResult::Passed;
        assert_eq!(cell.display_char(false), ('C', style::Color::Green));

        cell.chess = TrackResult::Failed;
        assert_eq!(cell.display_char(false), ('X', style::Color::Red));
    }

    #[test]
    fn display_char_both_tracks() {
        let mut cell = CellState {
            name: "test".to_string(),
            tracks: BuildTracks::Both,
            peano: TrackResult::NotStarted,
            chess: TrackResult::NotStarted,
        };

        // Both not started
        assert_eq!(cell.display_char(false), ('X', style::Color::DarkGrey));

        // Both building
        cell.peano = TrackResult::Building;
        cell.chess = TrackResult::Building;
        assert_eq!(cell.display_char(false), ('.', style::Color::Yellow));

        // Peano passed, Chess building -- flickers
        cell.peano = TrackResult::Passed;
        cell.chess = TrackResult::Building;
        assert_eq!(cell.display_char(true), ('P', style::Color::Green));
        assert_eq!(cell.display_char(false), ('.', style::Color::Yellow));

        // Peano failed, Chess building -- flickers
        cell.peano = TrackResult::Failed;
        cell.chess = TrackResult::Building;
        assert_eq!(cell.display_char(true), ('P', style::Color::Red));
        assert_eq!(cell.display_char(false), ('.', style::Color::Yellow));

        // Peano building, Chess passed -- flickers
        cell.peano = TrackResult::Building;
        cell.chess = TrackResult::Passed;
        assert_eq!(cell.display_char(true), ('.', style::Color::Yellow));
        assert_eq!(cell.display_char(false), ('C', style::Color::Green));

        // Both passed
        cell.peano = TrackResult::Passed;
        cell.chess = TrackResult::Passed;
        assert_eq!(cell.display_char(false), ('O', style::Color::Green));

        // Both failed
        cell.peano = TrackResult::Failed;
        cell.chess = TrackResult::Failed;
        assert_eq!(cell.display_char(false), ('X', style::Color::Red));

        // Peano passed, Chess failed -- mixed
        cell.peano = TrackResult::Passed;
        cell.chess = TrackResult::Failed;
        assert_eq!(cell.display_char(false), ('P', style::Color::Yellow));

        // Peano failed, Chess passed -- mixed
        cell.peano = TrackResult::Failed;
        cell.chess = TrackResult::Passed;
        assert_eq!(cell.display_char(false), ('C', style::Color::Yellow));
    }

    #[test]
    fn grid_layout_display_dimensions() {
        let g = GridLayout::new(67);
        assert_eq!(g.display_width(), 21); // "[ " + 9 cols * 2 chars + "]"
        assert_eq!(g.display_height(), 8); // 8 rows
    }

    #[test]
    fn diagnostic_llvm_error() {
        let log = "\
Peano build step 3/3 failed (exit 250):
--- step 3/3 ---
aiecc.py ...
--- stderr ---
LLVM ERROR: unable to legalize instruction: %89:_(<4 x s8>) = G_ADD %88:_, %13:_
PLEASE submit a bug report";
        let diag = extract_build_diagnostic(log);
        assert!(diag.starts_with("LLVM: "), "got: {}", diag);
        assert!(diag.contains("unable to legalize"), "got: {}", diag);
    }

    #[test]
    fn diagnostic_mlir_error() {
        let log = "\
Build step 2/3 failed (exit 1):
--- stderr ---
aie_arch.mlir:42:5: error: 'aie.tile' op column index out of range";
        let diag = extract_build_diagnostic(log);
        assert!(diag.starts_with("error:"), "got: {}", diag);
        assert!(diag.contains("column index"), "got: {}", diag);
    }

    #[test]
    fn diagnostic_python_error() {
        let log = "\
Build step 1/1 failed (exit 1):
--- stderr ---
Traceback (most recent call last):
  File \"aiecc.py\", line 5
ModuleNotFoundError: No module named 'aie'";
        let diag = extract_build_diagnostic(log);
        assert!(diag.starts_with("Python:"), "got: {}", diag);
        assert!(diag.contains("ModuleNotFoundError"), "got: {}", diag);
    }

    #[test]
    fn diagnostic_make_error() {
        let log = "\
Peano example build failed (exit 2):
--- stderr ---
make: *** [Makefile:12: build/final.xclbin] Error 2";
        let diag = extract_build_diagnostic(log);
        assert!(diag.starts_with("make:"), "got: {}", diag);
    }

    #[test]
    fn diagnostic_fallback() {
        let log = "Build step 1/1 failed (exit 42):\nsome unknown output";
        let diag = extract_build_diagnostic(log);
        assert_eq!(diag, "Build step 1/1 failed (exit 42):");
    }
}
