# Differential Fuzzer Revival -- Design

Status: approved (brainstorm), pending implementation plan
Date: 2026-05-30

## Context

The AIE2 implementation-gaps queue is empty -- every capability-spine domain
is Full/closed. The frontier has shifted from *modeling* to *verification*:
the Vector and SideEffect semantic categories still carry `Unverified`
verdicts (Vector ops were reimplemented from aietools SystemC models and have
never been cross-checked against silicon). The hardware that can verify them
is the Phoenix NPU we currently own, and it disappears at the Strix swap --
so HW-grounded verification is time-sensitive.

A differential logic fuzzer is the highest-leverage way to find
emulator/silicon divergence: generate valid kernels, run them on both the
emulator and the real NPU, diff the outputs. It tests the emulator against the
actual silicon rather than against our reading of the architecture docs.

### The fuzzer already exists -- and is orphaned

`src/fuzzer/` already contains ~4330 lines built over 15+ commits: an AST
(`ast.rs`), a deterministic seed-driven generator (`gen.rs`), C++/MLIR lowering
(`lower_cpp.rs`, `tools/fuzz_template.py`), a full pipeline runner (`runner.rs`:
generate -> Peano-compile -> run EMU + run NPU -> diff), and a trace-sweep
comparator (`trace_sweep.rs`) with the complete AIE2 event catalogue and
determinism reporting.

It compiles today (it is `#[cfg(feature = "tooling")]`-gated and `tooling` is a
default feature). But **nothing calls it**: `grep "fuzzer::"` outside the module
returns nothing, and there is no `fuzz` subcommand. Its former entry point was
`npu-test --fuzz`, and `npu-test` was deleted in commit 24fdf72 ("cleanup:
delete npu-test runner and old test infrastructure"). The `FuzzOptions`
doc-comment records the intent at the time -- "the fuzzer will be invoked
through the bridge test infrastructure going forward" -- so the orphaning
happened mid-migration: lifted out of `npu-test`'s config, never re-hosted.

The roadmap (`docs/roadmap/phase4-validation-testing.md`) still lists
"Differential Kernel Fuzzer | Not started", which is stale.

## Goal

Revive the existing scalar fuzzer to a standing, repeatable capability and use
it for a real EMU+HW differential batch against the Phoenix NPU. Concretely:

1. A permanent `cargo run -- fuzz` subcommand that drives the existing
   `run_fuzz()` pipeline.
2. The full loop proven end-to-end on real hardware (generate -> Peano-compile
   -> EMU + NPU -> diff), with any divergences triaged.

This is **re-hosting and de-bitrotting an intact system**, not building a new
one. The design stays ruthlessly inside that boundary.

## Non-goals

**Deferred -- planned next phases (explicitly coming, just not in this revive):**

- **Vector ops.** The current AST generates only single-tile scalar arithmetic,
  stores, branches, and hardware loops. Extending the generator to vector ops
  (MAC/matmul/SRS/UPS) is how the fuzzer will eventually attack the high-risk
  `Unverified` Vector surface. It is the intended *next* phase after this revive
  lands -- not part of it.
- **Chess.** The revive is Peano-only (open-source, no license, runs in-sandbox
  cleanly). Chess is ground truth and will be added as an opt-in compiler path
  later, mirroring the bridge suite's dual-compiler model.

**Out of scope -- may never be needed:**

- **Shrinking / test-case minimization.** The generator is deterministic, so a
  divergence's minimal reproducer is its seed. That is sufficient for triage.
  Proptest-style shrinking is not planned.
- **Bridge-script integration.** The `fuzz` subcommand is self-contained; the
  bridge harness can shell out to it later if we want HW-batch orchestration to
  match bridge ergonomics, but the revive does not depend on it.

## Architecture

### Entry point

Add a `fuzz` subcommand to `src/main.rs`, mirroring the existing `test-suite`
subcommand pattern: manual `env::args` parsing, gated on
`#[cfg(feature = "tooling")]` (the fuzzer module is already behind this
feature). The subcommand maps CLI flags to the existing `FuzzOptions` struct and
calls `fuzzer::runner::run_fuzz(&opts)`.

Flag surface (one-to-one with `FuzzOptions` fields):

| Flag | FuzzOptions field | Default |
|------|-------------------|---------|
| `--iterations N` | `fuzz_iterations` | required |
| `--seed S` | `fuzz_seed` (`Option<u64>`) | wall-clock |
| `--jobs J` | `jobs` | nproc |
| `--hw` | `hw` | `false` (EMU-only) |
| `--max-cycles C` | `max_cycles` | existing default |
| `--trace-sweep` | `trace_sweep` | `false` |
| `--trace-sweep-reps R` | `trace_sweep_reps` | 5 |
| `--verbose` | `verbose` | `false` |

Plus a help entry. EMU-only is the default so the common in-sandbox case needs
no HW flag.

### Components touched

- **`src/main.rs`** -- new subcommand + arg parse + help text. The only
  substantial new Rust.
- **`src/fuzzer/runner.rs`** -- touched only where runtime drift demands
  (toolchain flags, paths); divergence-report tightening if warranted. No
  structural change.
- **`tools/fuzz_template.py`** -- the most likely drift site, since the
  IRON/aiecc Python API moves faster than the Rust. Fix to current mlir-aie.
- **`gen.rs` / `lower_cpp.rs` / `ast.rs`** -- untouched unless a smoke run
  proves a real defect.
- **Docs** -- short usage note; de-stale `docs/roadmap/phase4-validation-testing.md`.

### Data flow (unchanged from the existing pipeline)

```
seed
  -> FuzzParams              (gen::generate)
  -> C++ kernel + MLIR        (lower_cpp + tools/fuzz_template.py)
  -> Peano/aiecc compile      (runner::compile_fuzz_case)
  -> aie.xclbin
  -> { EMU: xclbin_suite (in-process)
       NPU: testing::npu_runner::run_on_npu }
  -> byte-compare outputs + trace compare
  -> divergence/determinism report under build/fuzz/
```

The HW side is gated by `npu_runner::npu_available()`, which returns a clean
error when no NPU is present -- EMU-only degrades automatically.

## Execution plan: smoke before scale, EMU before HW

The milestones front-load the one genuinely unknown risk (driving the NPU from
the sandbox) so we hit any wall on seed #1, not seed #400.

- **M0 -- EMU-only smoke (in-sandbox).** Minimal `fuzz` subcommand; 1 seed,
  `--no-hw` (default). Proves gen -> template -> Peano -> xclbin -> EMU
  end-to-end.
- **M1 -- HW smoke (in-sandbox).** Same seed, `--hw`. Proves `npu_runner` runs
  on the NPU from the sandbox -- the license/filesystem reconciliation. This is
  the risk gate.
- **M2 -- fix drift.** Repair whatever M0/M1 surface (template API, aiecc flags,
  paths). TDD any Rust logic; iterate the template against real compile output.
- **M3 -- EMU-only batch.** A few dozen seeds in-sandbox; confirm determinism
  (same seed -> same result) and report quality at scale before involving HW.
- **M4 -- EMU+HW differential batch.** As many seeds as practical -- HW
  execution is very fast, so thousands per session is reasonable and preferred
  (broad seed coverage is the whole point). Run with trace-sweep OFF: it is the
  slow path and would throttle the batch. Full output diff; triage every
  divergence (real emulator bug vs. clean scalar surface). The deliverable that
  spends Phoenix time while we have it.
- **M5 -- standing capability.** Finalize subcommand UX + help + a short usage
  doc; update the roadmap.

## Testing

- TDD the new Rust: arg-string -> `FuzzOptions` mapping, and any report logic.
- The pipeline itself is integration-validated by M0/M1/M3 -- toolchain paths
  and the compile flow cannot be meaningfully unit-tested.
- Existing fuzzer unit tests (`ast`, `gen`, `params`, `lower_cpp` determinism)
  must stay green.

## Error handling

- `npu_available()` gates HW use; EMU-only fallback is automatic.
- Per-seed compile failures are reported, not fatal to the batch.
- The trace path carries a determinism check (repeat the NPU run, compare
  decoded event sequences) before trusting a trace comparison.

## Success criteria

1. `cargo run -- fuzz` is a permanent, documented subcommand that runs the
   pipeline EMU-only in-sandbox and EMU+HW with `--hw`.
2. A large EMU+HW differential batch (thousands of seeds -- HW is fast, so
   throughput is not the constraint) has run against the Phoenix NPU, with all
   divergences triaged (each classified as a real emulator bug to fix, or the
   scalar surface confirmed clean).
3. The existing fuzzer unit tests stay green; new Rust is TDD-covered.

## Risks

- **M1 (in-sandbox NPU execution) is the real risk.** Everything else is
  mechanical. CLAUDE.md historically warns against in-sandbox NPU captures
  citing license checks and filesystem isolation; the working hypothesis is
  that the license gate is Chess-specific (Peano is license-free) and that raw
  NPU execution via `npu_runner` is reachable from the sandbox. M1 confirms or
  refutes this on the first seed. If it refutes, the HW batch (M4) is driven
  outside the sandbox instead, and the rest of the design is unaffected.
- **Template drift (`fuzz_template.py`).** IRON/aiecc API churn is the most
  likely de-bitrotting cost. Bounded; surfaced by M0.
