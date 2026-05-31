# Differential Fuzzer Revival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-host the orphaned `src/fuzzer/` pipeline behind a permanent `cargo run -- fuzz` subcommand, flush any runtime drift, and prove the EMU+HW differential loop on the Phoenix NPU.

**Architecture:** The fuzzer pipeline (generate -> Peano-compile -> EMU + NPU -> diff) already exists and compiles; it just has no caller. Add a testable arg-parser in the library (`src/fuzzer/cli.rs`), wire a `fuzz` subcommand in `src/main.rs` that calls the existing `fuzzer::runner::run_fuzz`, then validate the pipeline with a smoke seed (EMU then HW) before scaling to a batch.

**Tech Stack:** Rust (the emulator + fuzzer), Peano (open-source AIE compiler), `tools/fuzz_template.py` (IRON/MLIR template), the Phoenix NPU via `testing::npu_runner`.

**Spec:** `docs/superpowers/specs/2026-05-30-fuzzer-revival-design.md`

---

## File Structure

- **Create `src/fuzzer/cli.rs`** -- pure `parse_fuzz_args(&[String]) -> Result<FuzzOptions, String>`; unit-tested. One responsibility: map CLI args to the existing `FuzzOptions` struct.
- **Modify `src/fuzzer/mod.rs`** -- add `pub mod cli;`.
- **Modify `src/main.rs`** -- add the `fuzz` subcommand dispatch + a thin `run_fuzz_command` wrapper + help text.
- **Modify `tools/fuzz_template.py`** -- only if the M0 smoke surfaces IRON/aiecc drift.
- **Modify `docs/roadmap/phase4-validation-testing.md`** -- de-stale the "Differential Kernel Fuzzer | Not started" line.
- **Create `docs/fuzzer-usage.md`** -- short usage note (M5).

`FuzzOptions` (already defined in `src/fuzzer/runner.rs:23`, all fields `pub`):

```rust
pub struct FuzzOptions {
    pub verbose: bool,
    pub jobs: usize,
    pub hw: bool,
    pub max_cycles: u64,
    pub fuzz_iterations: usize,
    pub fuzz_seed: Option<u64>,
    pub trace_sweep: bool,
    pub trace_sweep_reps: usize,
}
```

---

## Task 1: Testable fuzz arg-parser (`parse_fuzz_args`)

**Files:**
- Create: `src/fuzzer/cli.rs`
- Modify: `src/fuzzer/mod.rs` (add `pub mod cli;`)
- Test: in `src/fuzzer/cli.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Add the module declaration**

In `src/fuzzer/mod.rs`, add after the existing `pub mod ast;` line:

```rust
pub mod cli;
```

- [ ] **Step 2: Write the failing tests**

Create `src/fuzzer/cli.rs` with ONLY the tests first (the function comes in Step 4):

```rust
//! CLI argument parsing for the `fuzz` subcommand.
//!
//! Pure mapping from process args to `FuzzOptions`, kept in the library so it
//! is unit-testable (binaries are not). `main.rs` is a thin caller.

use crate::fuzzer::runner::FuzzOptions;

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["xdna-emu".to_string(), "fuzz".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn defaults_when_only_subcommand() {
        let o = parse_fuzz_args(&argv(&[])).unwrap();
        assert_eq!(o.fuzz_iterations, 0);
        assert_eq!(o.fuzz_seed, None);
        assert!(!o.hw);
        assert!(!o.trace_sweep);
        assert_eq!(o.trace_sweep_reps, 5);
        assert!(!o.verbose);
    }

    #[test]
    fn parses_iterations_and_seed() {
        let o = parse_fuzz_args(&argv(&["--iterations", "100", "--seed", "42"])).unwrap();
        assert_eq!(o.fuzz_iterations, 100);
        assert_eq!(o.fuzz_seed, Some(42));
    }

    #[test]
    fn hw_flag_sets_hw_and_no_hw_clears_it_last_wins() {
        assert!(parse_fuzz_args(&argv(&["--hw"])).unwrap().hw);
        assert!(!parse_fuzz_args(&argv(&["--hw", "--no-hw"])).unwrap().hw);
    }

    #[test]
    fn parses_jobs_max_cycles_trace_sweep() {
        let o = parse_fuzz_args(&argv(&[
            "--jobs", "4", "--max-cycles", "500000", "--trace-sweep", "--trace-sweep-reps", "3",
        ]))
        .unwrap();
        assert_eq!(o.jobs, 4);
        assert_eq!(o.max_cycles, 500_000);
        assert!(o.trace_sweep);
        assert_eq!(o.trace_sweep_reps, 3);
    }

    #[test]
    fn unknown_flag_is_an_error() {
        assert!(parse_fuzz_args(&argv(&["--bogus"])).is_err());
    }

    #[test]
    fn missing_value_is_an_error() {
        assert!(parse_fuzz_args(&argv(&["--iterations"])).is_err());
    }

    #[test]
    fn non_numeric_value_is_an_error() {
        assert!(parse_fuzz_args(&argv(&["--iterations", "lots"])).is_err());
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib fuzzer::cli`
Expected: FAIL to compile -- `parse_fuzz_args` not found.

- [ ] **Step 4: Write the minimal implementation**

Add ABOVE the `#[cfg(test)]` block in `src/fuzzer/cli.rs`:

```rust
/// Runaway-guard ceiling for emulator cycles per fuzz case. Fuzz kernels are
/// tiny (buffers 16-256 elems, 2-8 ops), so this only bounds pathological runs.
const DEFAULT_MAX_CYCLES: u64 = 10_000_000;

fn default_jobs() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn parse_next<'a, I, T>(iter: &mut I, flag: &str) -> Result<T, String>
where
    I: Iterator<Item = &'a String>,
    T: std::str::FromStr,
{
    let raw = iter.next().ok_or_else(|| format!("{} requires a value", flag))?;
    raw.parse::<T>().map_err(|_| format!("invalid value for {}: {}", flag, raw))
}

/// Parse `fuzz` subcommand args (full argv, including argv[0] and the `fuzz`
/// token) into `FuzzOptions`. Unknown flags are errors so typos fail loudly.
pub fn parse_fuzz_args(args: &[String]) -> Result<FuzzOptions, String> {
    let mut opts = FuzzOptions {
        verbose: false,
        jobs: default_jobs(),
        hw: false,
        max_cycles: DEFAULT_MAX_CYCLES,
        fuzz_iterations: 0,
        fuzz_seed: None,
        trace_sweep: false,
        trace_sweep_reps: 5,
    };

    // Skip argv[0] (binary name); the `fuzz` token is consumed by its match arm.
    // Items borrow from `args`, not from `iter`, so calling parse_next(&mut iter)
    // inside the loop body after iter.next() is borrow-clean.
    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "fuzz" => {}
            "--iterations" | "-n" => opts.fuzz_iterations = parse_next(&mut iter, "--iterations")?,
            "--seed" => opts.fuzz_seed = Some(parse_next(&mut iter, "--seed")?),
            "--jobs" | "-j" => opts.jobs = parse_next(&mut iter, "--jobs")?,
            "--max-cycles" => opts.max_cycles = parse_next(&mut iter, "--max-cycles")?,
            "--hw" => opts.hw = true,
            "--no-hw" => opts.hw = false,
            "--trace-sweep" => opts.trace_sweep = true,
            "--trace-sweep-reps" => opts.trace_sweep_reps = parse_next(&mut iter, "--trace-sweep-reps")?,
            "--verbose" | "-v" => opts.verbose = true,
            other => return Err(format!("unknown fuzz argument: {}", other)),
        }
    }
    Ok(opts)
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib fuzzer::cli`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add src/fuzzer/cli.rs src/fuzzer/mod.rs
git commit -m "fuzzer: add testable parse_fuzz_args CLI parser

Generated using Claude Code."
```

---

## Task 2: Wire the `fuzz` subcommand into main.rs

**Files:**
- Modify: `src/main.rs` (dispatch block after the `test-suite` block at line ~51; new `run_fuzz_command` fn; `print_help`)

- [ ] **Step 1: Add the dispatch block**

In `src/main.rs`, immediately AFTER the `test-suite` block (after its closing `}` at line ~51, before `// Check for GUI mode`), insert:

```rust
    // Check for fuzz command
    if args.len() >= 2 && args[1] == "fuzz" {
        #[cfg(feature = "tooling")]
        {
            return run_fuzz_command(&args);
        }
        #[cfg(not(feature = "tooling"))]
        {
            eprintln!("fuzz command requires --features tooling");
            std::process::exit(1);
        }
    }
```

- [ ] **Step 2: Add the `run_fuzz_command` function**

In `src/main.rs`, add near `run_test_suite` (around line 348):

```rust
#[cfg(feature = "tooling")]
fn run_fuzz_command(args: &[String]) -> anyhow::Result<()> {
    let opts = xdna_emu::fuzzer::cli::parse_fuzz_args(args)
        .map_err(|e| anyhow::anyhow!("fuzz: {}", e))?;
    xdna_emu::fuzzer::runner::run_fuzz(&opts);
    Ok(())
}
```

- [ ] **Step 3: Update `print_help`**

In `src/main.rs` `print_help()`, add to the `COMMANDS:` section (after the `test-suite` line):

```rust
    println!("    fuzz [OPTIONS]      Differential logic fuzzer (EMU vs NPU)");
```

and to the `EXAMPLES:` section:

```rust
    println!("    xdna-emu fuzz --iterations 100              # EMU-only fuzz batch");
    println!("    xdna-emu fuzz --iterations 1000 --hw        # EMU+HW differential");
```

- [ ] **Step 4: Build and verify the wiring with a no-toolchain smoke**

`--iterations 0` returns before any tool discovery (`run_fuzz` early-returns at `iterations == 0`), so this checks the wiring without needing Peano.

Run: `TMPDIR=/tmp/claude-1000 cargo run -- fuzz --iterations 0 --seed 1`
Expected output (exactly):
```
Fuzzing 0 iterations, base seed 1
Fuzz complete: 0 pass, 0 fail, 0 error
```

- [ ] **Step 5: Verify the help text renders**

Run: `cargo run -- --help`
Expected: output now includes the `fuzz [OPTIONS]` command line and the two fuzz examples.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "fuzzer: wire 'fuzz' subcommand into the CLI

Generated using Claude Code."
```

---

## Task 3: M0 -- EMU-only single-seed smoke (flush compile/template drift)

This is an investigation+repair task: run one seed through the real pipeline
and fix whatever runtime drift the orphaning left. The fix code cannot be
pre-written (it depends on what breaks), so this task gives the command, the
success criteria, and the concrete diagnostic procedure for the known-likely
failure points.

**Files (potentially):**
- Modify: `tools/fuzz_template.py` (if IRON/aiecc API drifted)
- Modify: `src/fuzzer/runner.rs` (if a toolchain flag/path drifted)

- [ ] **Step 1: Ensure the NPU env is OFF (EMU path uses in-process xclbin_suite, not the plugin)**

Confirm `XDNA_EMU` is unset in this shell (the fuzzer's EMU side is in-process and must not route through the XRT emu plugin):

Run: `echo "XDNA_EMU=[$XDNA_EMU]"`
Expected: `XDNA_EMU=[]`

- [ ] **Step 2: Run one EMU-only seed**

Run: `TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 1 --seed 1`
Expected (success): tool-discovery lines (`peano:`, `aiecc:`, `template:`), then a compile line, then an EMU run, then a `Fuzz complete: 1 pass, 0 fail, 0 error` (or a `fail`/`error` if the kernel legitimately diverges or the pipeline breaks). Artifacts under `build/fuzz/`.

- [ ] **Step 3: If it fails, diagnose by stage**

Work the failure to its stage and fix minimally:

- **Tool discovery error** (`failed to discover build tools`): check `src/fuzzer/runner.rs:62` `ToolPaths::discover` against `crate::integration::chess_build::BuildEnv` -- a method name (`peano_clang`, `aiecc`, `pythonpath`, ...) may have drifted. Fix the call to the current `BuildEnv` API.
- **Peano kernel compile error**: inspect `build/fuzz/<seed>/fuzz_kernel.cc.o` step. Compare the `peano_clang` invocation in `runner.rs:939` (`--target=aie2-none-unknown-elf -O2 -c`) against a known-good Peano command from `scripts/emu-bridge-test.sh`.
- **Template / MLIR error** (most likely): run the template command shown in the log by hand:
  `python3 tools/fuzz_template.py --kernel fuzz_kernel.cc --size 64 --dtype i32 --outdir <case_dir>` and read the Python traceback. Fix `tools/fuzz_template.py` to the current IRON API (compare against a current `aie2.py` under `${MLIR_AIE}/test/npu-xrt/`). Common drifts: `aie.dialects` import paths, `object_fifo` signature, `aiecc.py` flag names.
- **xclbin parse / EMU error**: the in-process side is `src/testing/xclbin_suite.rs`; a parse failure points at the generated xclbin, an EMU panic at a real emulator bug (capture the seed -- that is a genuine finding, see Task 7 triage).

- [ ] **Step 4: Re-run until green**

Run: `TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 1 --seed 1`
Expected: `Fuzz complete: 1 pass, 0 fail, 0 error` (a clean EMU run of seed 1).

- [ ] **Step 5: Confirm no Rust unit-test regressions**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib fuzzer`
Expected: PASS (existing fuzzer unit tests plus Task 1's).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "fuzzer: revive EMU-only pipeline (M0 smoke green)

<one line per drift fix actually made; if none, say 'no drift -- pipeline ran clean'>

Generated using Claude Code."
```

---

## Task 4: M1 -- HW single-seed smoke (the risk gate)

Proves `npu_runner` can drive the Phoenix NPU from the sandbox. This is the one
genuinely unknown step; everything else is mechanical.

**Files (potentially):**
- Modify: `src/fuzzer/runner.rs` or `src/testing/npu_runner.rs` (only if an in-sandbox HW path issue is found and is a small, legitimate fix)

- [ ] **Step 1: Run one seed with HW differential**

Ensure `XDNA_EMU` is unset (HW must use the real driver), then:

Run: `TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 1 --seed 1 --hw`
Expected (success): the EMU run AND an NPU run, then a byte-for-byte comparison resulting in `1 pass` (outputs match) or a recorded mismatch under `build/fuzz/<seed>/` (`npu_output.bin`, mismatch detail).

- [ ] **Step 2: If `NPU hardware not available`**

`run_on_npu_raw` returns this when `npu_runner::npu_available()` is false. Diagnose:
- Confirm the NPU device is present: `ls -l /dev/accel/ 2>/dev/null || ls -l /dev/dri/ 2>/dev/null`.
- Check `npu_runner::npu_available()` (`src/testing/npu_runner.rs`) for what it probes; if it relies on an env var or path that the sandbox doesn't expose, that is the finding to report back (per the spec's risk note, the fallback is driving M4 outside the sandbox -- STOP and report rather than papering over it).

- [ ] **Step 3: If the NPU run errors or wedges**

- A clean `Err(...)` from `run_on_npu` (compile/load/timeout): capture the message; if it is a licensing/filesystem error, that confirms the spec's risk hypothesis -- report it.
- If the NPU wedges (D-state, mailbox timeout): recover per CLAUDE.md -- `pkexec sh -c 'modprobe -r amdxdna && modprobe amdxdna'` -- then reduce to a single seed and retry once before escalating.

- [ ] **Step 4: Confirm a clean HW smoke**

Run: `TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 1 --seed 1 --hw`
Expected: `Fuzz complete: 1 pass, 0 fail, 0 error` (EMU and NPU agree on seed 1), OR a deterministic recorded mismatch (a real finding for Task 7).

- [ ] **Step 5: Commit (only if a fix was needed)**

```bash
git add -A
git commit -m "fuzzer: in-sandbox HW differential path verified (M1 smoke)

Generated using Claude Code."
```

If no code change was needed (the path just worked), skip the commit and note the result in the handoff.

---

## Task 5: M3 -- EMU-only batch + determinism

**Files:** none (validation task; only touch `src/fuzzer/runner.rs` if the report is genuinely unreadable).

- [ ] **Step 1: Run a few-dozen-seed EMU-only batch**

Run: `TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 50 --seed 1000`
Expected: 50 cases compile + run on EMU; a summary `Fuzz complete: <p> pass, <f> fail, <e> error`. Any `fail`/`error` seeds are recorded under `build/fuzz/`.

- [ ] **Step 2: Verify determinism (same base seed -> same result)**

Run the identical command again:
`TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 50 --seed 1000`
Expected: identical pass/fail/error counts and identical per-seed outcomes (the generator and EMU are deterministic).

- [ ] **Step 3: Triage any EMU-only failures**

For each `fail`/`error` seed, reproduce in isolation:
`TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 1 --seed <SEED>`
Classify: compile/template defect (fix in pipeline) vs. a real emulator bug (record the seed; fix is a separate TDD cycle -- start from the failing assertion per the debugging guidelines). Do not bulk-fix here; record and continue.

- [ ] **Step 4: Commit findings**

```bash
git add -A
git commit -m "fuzzer: EMU-only batch green + deterministic (M3)

<summary: N seeds, pass/fail/error counts, any triaged seeds noted>

Generated using Claude Code."
```

---

## Task 6: M4 -- EMU+HW differential batch (the deliverable)

**Files:** none (validation task).

- [ ] **Step 1: Run a large differential batch, trace-sweep OFF**

HW is fast, so scale up; keep trace-sweep off (it is the slow path).

Run: `TMPDIR=/tmp/claude-1000 cargo run --release -- fuzz --iterations 2000 --seed 1 --hw`
Expected: 2000 seeds compiled, run on EMU + NPU, byte-compared. Summary counts; mismatches recorded under `build/fuzz/`.

- [ ] **Step 2: Triage every divergence**

For each mismatch seed, reproduce single-seed (`--iterations 1 --seed <SEED> --hw`) and classify:
- **Real emulator bug**: the EMU disagrees with silicon. Record the seed, the first-mismatch element (from the recorded detail), and the kernel params. Each fix is its own TDD cycle (failing test reproducing the divergence -> fix -> green) -- out of scope for this validation task; collect them.
- **Clean**: outputs agree -> the scalar surface holds for that seed.

- [ ] **Step 3: Record the batch result**

Write a short findings note (counts, list of real-divergence seeds with their first-mismatch detail) to `docs/superpowers/findings/2026-05-30-fuzzer-revival-first-batch.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/findings/2026-05-30-fuzzer-revival-first-batch.md
git commit -m "fuzzer: first EMU+HW differential batch (M4) -- findings

<N seeds, pass/divergence counts, real-bug seed list>

Generated using Claude Code."
```

---

## Task 7: M5 -- standing capability (docs + de-stale roadmap)

**Files:**
- Create: `docs/fuzzer-usage.md`
- Modify: `docs/roadmap/phase4-validation-testing.md`

- [ ] **Step 1: Write the usage doc**

Create `docs/fuzzer-usage.md`:

```markdown
# Differential Fuzzer

`cargo run --release -- fuzz [OPTIONS]` generates random single-tile scalar
kernels, compiles them with Peano, runs each on the emulator and (with `--hw`)
the real NPU, and diffs the outputs. Mismatches indicate emulator bugs.

## Options

| Flag | Meaning | Default |
|------|---------|---------|
| `--iterations N` | number of seeds to run | 0 |
| `--seed S` | base seed (deterministic) | wall clock |
| `--hw` | also run on the real NPU and diff | off (EMU-only) |
| `--jobs J` | parallel compile/EMU jobs | nproc |
| `--max-cycles C` | per-case EMU runaway guard | 10_000_000 |
| `--trace-sweep` | multi-group trace capture + compare (SLOW) | off |
| `--trace-sweep-reps R` | NPU reps for trace determinism | 5 |
| `--verbose` | extra logging | off |

## Examples

    cargo run --release -- fuzz --iterations 100              # EMU-only batch
    cargo run --release -- fuzz --iterations 2000 --hw        # EMU+HW differential

A divergence's reproducer is its seed: `... fuzz --iterations 1 --seed <SEED> --hw`.

## Scope

Single-tile scalar ops only (arith/bitwise/shift/branch/hw-loop), Peano-compiled.
Vector ops and Chess are planned next phases. Artifacts land under `build/fuzz/`.
```

- [ ] **Step 2: De-stale the roadmap**

In `docs/roadmap/phase4-validation-testing.md`, change the `Differential kernel fuzzer | Not started` row to reflect the revived state, e.g.:

```markdown
| Differential kernel fuzzer | Revived (scalar) | `cargo run -- fuzz`; EMU+HW differential, Peano, single-tile scalar. Vector/Chess deferred. See docs/fuzzer-usage.md |
```

- [ ] **Step 3: Commit**

```bash
git add docs/fuzzer-usage.md docs/roadmap/phase4-validation-testing.md
git commit -m "fuzzer: usage doc + de-stale roadmap (M5)

Generated using Claude Code."
```

---

## Notes for the implementer

- **No `.so` rebuild needed.** The fuzzer's EMU side is in-process (`xclbin_suite`); the HW side is the real driver via `npu_runner`. Neither uses the XRT emu plugin.
- **Debug for wiring, release for runs.** Use plain `cargo run` for Task 2's `--iterations 0` wiring check; use `cargo run --release` for any real compile/EMU/HW work (the EMU is far faster in release).
- **Tasks 3 and 4 are repair tasks** -- their commit bodies should record exactly what drifted (or "no drift"). If M1 (Task 4) reveals the NPU is not driveable in-sandbox, STOP and report; do not force it.
- **Real divergences are collected, not fixed inline** (Tasks 5-6). Each becomes its own TDD bug-fix cycle afterward, starting from a failing test that reproduces the seed.
