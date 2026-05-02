# Phase E: Trace-Diff-Based Cycle Budget Design

**Date:** 2026-04-22
**Status:** Brainstormed and approved; awaiting implementation plan
**Supersedes:** Phase D.3 (HW spot-check) and Phase C (cycle budget enforcement)
  from `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`

## Why this supersedes prior plans

The original `2026-04-22-cycle-budget-testing.md` Phase C tasks were
specced before Phase B settled on its trace-based capture approach.
They assumed Phase B would produce per-tile cycle counter snapshots
(via the abandoned `CycleCounterHelper` path) and built per-tile
budget enforcement on top. Phase B as actually shipped produces a
single integer per `(test, compiler)` derived from `max(ts) - min(ts)`
across event timestamps in the trace buffer — a fundamentally
different data shape.

Phase D.3 ("HW integration spot-check") was specced as "compare EMU
per-tile counter values to HW per-tile cycle file" — a comparison
that's not directly possible against Phase B's actual output shape.

Rather than mechanically port the old tasks to the new data shape,
this design takes a different approach: **use the existing EMU
trace-emission infrastructure (`src/trace/`, `src/device/trace_unit/`,
`src/bin/trace_compare.rs`) to produce HW-binary-compatible trace
output from EMU, then diff it against HW's trace using the existing
`trace_compare` binary.** The cycle budget falls out as one slice of
that comparison; debuggability of EMU/HW divergences is the bigger
win.

## Goal

Turn today's opaque `TIMEOUT` results from EMU bridge tests into
actionable signals:

- `MATCH`: EMU and HW execute the same workload and produce
  binary-comparable trace output within tolerances.
- `BUDGET` (formerly `DRIFT`): EMU completes but its trace diverges
  from HW's beyond tolerance — a real cycle-modeling regression.
- `EMU_TRACE_BUG` / `HW_TRACE_BUG`: same xclbin, same trace setup,
  exactly one pipeline emitted events. **Hard fail with the bug
  attribution.** The project principle is "fix bugs where they are";
  silent acceptance of trace asymmetry is the wrong shape.
- `TIMEOUT`: actual wall-clock timeout (preserved).
- `EMPTY`: both sides legitimately emitted no trace events (Phase B
  Limitation 1 — scalar kernels with the default event set). Counted
  in summary so the gap stays visible.

## Existing infrastructure this design builds on

Substantial work already exists. Discovered via Explore agent survey
on 2026-04-22:

| Component | Location | Status |
|-----------|----------|--------|
| Event ID mapping (EMU → HW IDs) | `src/trace/mod.rs` | Built; comprehensive AIE2 event coverage; 200+ test points |
| Per-tile HW trace unit emulation | `src/device/trace_unit/` | Built; state machine + binary packet encoding + backpressure handling |
| Per-tile event log (the source of trace events) | `src/interpreter/state/event_trace.rs` | Built; circular buffer; 24 EventType variants |
| Coordinator wiring (events → trace units → stream switch → host) | `src/interpreter/engine/coordinator.rs` Phase 3a/3b/3c | Built; wires core, DMA, lock events to trace units; flushes on completion |
| Binary trace comparison tool | `src/bin/trace_compare.rs` | Built; modes: `--hw/--emu`, `--sweep`; flags: `--stalls`, `--cross-tile`, `--iterations`; `--stalls` already does stall→resolution attribution with gap deltas |
| Event sweep harness (future use) | `src/fuzzer/trace_sweep.rs` | Built; 13 event groups covering 100+ AIE2 core events |

The trace-emission pipeline switched to HW-binary-compatible output
in commit `28711df` (2026-03-01, "event broadcast system and binary
trace pipeline"). Last meaningful work was around then. The Explore
agent's verdict: **scaffolding-complete, end-to-end, awaiting HW
validation**. This design does that validation, plus the bridge
integration, plus the classification logic — not the construction
work.

## Scope

In scope:

1. Validate that EMU's `trace_unit` output is binary-compatible with
   `aie.utils.trace.parse_trace()` on at least one representative
   test (`vector_scalar_using_dma`).
2. Capture EMU's trace output during a bridge run and write it to a
   known location alongside HW's trace.
3. Reuse `tools/trace-to-cycles.py` (Phase B) on EMU's trace to
   produce a symmetric `cycles.EMU.<test>.<compiler>.txt`.
4. Run `trace_compare` per (test, compiler) pair where both traces
   exist; capture report; classify result.
5. Add a `--with-cycle-diff` flag to `scripts/emu-bridge-test.sh`
   that drives the EMU-side capture + comparison. Implies
   `--with-hw-cycles`. The HW pipeline (Phase B) is the upstream
   dependency.
6. Add dual-bound EMU timing: cycle budget via `XDNA_EMU_MAX_CYCLES`
   (already plumbed in Phase A), wall-clock timeout scaled from HW
   cycle data with the existing 600 s as a floor.
7. Surface trace-related compile failures distinctly (e.g.,
   `COMPILE-FAIL(traced)` vs generic `COMPILE-FAIL`) so users know
   when injection broke the build.
8. Add a `scripts/show-cycle-drift.sh` reporting tool that sorts
   tests by drift severity.
9. Phase D.3 spot-check folds into the validation task: the first
   end-to-end run on `vector_scalar_using_dma` is the spot-check.

Out of scope:

- Building new trace-emission machinery (already exists).
- Expanding the default event set to make scalar kernels traceable
  (Phase B Limitation 1; tracked as future work).
- Resurrecting trace event sweeping (`tools/trace-sweep.py` style).
  Future work, mentioned briefly below.
- Hooking trace into `.py`-only MLIR sources (Phase B Limitation 3;
  future work).
- Auto-fixing trace+ctrlpkt incompatibility (Phase B Limitation 4;
  upstream mlir-aie issue).

## Architecture

### Data flow

```
HW path (Phase B, already done):
  [aie.mlir] → mlir-trace-inject → [traced.mlir] → aiecc → [traced.xclbin]
            → bridge-trace-runner → [trace_hw.<compiler>.bin]
            → trace-to-cycles → [<test>.<compiler>.cycles.HW.txt]

EMU path (this design, all pieces exist except glue):
  [traced.xclbin]                                      ← same xclbin as HW
  → EMU run (XDNA_EMU=debug ./test.exe) emits trace via existing trace_unit
  → host DDR memory contains the trace bytes
  → bridge captures the bytes → [trace_emu.<compiler>.bin]
  → trace-to-cycles                                    ← reuse, no changes
  → [<test>.<compiler>.cycles.EMU.txt]

Comparison:
  trace_compare --hw <hw.bin> --emu <emu.bin> --stalls
  → [<test>.<compiler>.cycles.compare.txt]

Classification + budget enforcement:
  bridge parses trace_compare report
  bridge sets XDNA_EMU_MAX_CYCLES = ceil(HW_cycles * 2.0) for the run
  bridge sets wall-clock timeout = max(600, ceil(HW_cycles * 2.0 * SECONDS_PER_CYCLE))
  bridge classifies: MATCH / BUDGET / EMU_TRACE_BUG / HW_TRACE_BUG / EMPTY / TIMEOUT
```

### Components

Five components. All but #1 are bridge-script changes; #1 is the
validation gate before any of the rest can land.

1. **EMU trace output validation**

   Goal: confirm EMU's `trace_unit` produces bytes that
   `parse_trace()` can read on `vector_scalar_using_dma`.

   Method: run the test under EMU, capture the trace BO contents
   from the xrt-plugin, run `trace-to-cycles.py` on the output.
   Success = a non-zero integer cycle count emerges; bonus = the
   EMU cycle count is comparable in magnitude to HW's
   (vector_scalar_using_dma reference: HW = 41,181 cycles).

   If output is empty or unparseable, the validation task opens a
   focused debugging sub-task (likely in `src/device/trace_unit/`
   or `src/interpreter/engine/coordinator.rs`'s flush logic).
   This must succeed before subsequent components can be tested
   meaningfully.

2. **EMU trace capture in the bridge**

   The xrt-plugin already gets the trace BO via the same xclbin
   the HW path uses. The plugin needs to either (a) sync and dump
   the trace BO contents to a known path after EMU completes, or
   (b) expose the trace BO contents via a new FFI or a path the
   bridge reads directly.

   Most likely (a): plugin writes
   `${XDNA_TRACE_DIR}/trace_emu.<compiler>.bin` after each run,
   mirroring how HW results land via `XDNA_TRACE_DIR`. The
   bridge already exports `XDNA_TRACE_DIR` for HW (line 1149 of
   `scripts/emu-bridge-test.sh`); reuse the same env var for EMU.

3. **EMU cycle extraction**

   Pure reuse: same `tools/trace-to-cycles.py` Phase B uses, same
   args, different input path. Produces
   `cycles.EMU.<test>.<compiler>.txt`. No code changes.

4. **Comparison via `trace_compare`**

   For each `(test, compiler)` pair where both `trace_hw.<compiler>.bin`
   and `trace_emu.<compiler>.bin` exist, run:

   ```
   ./target/release/trace-compare \
     --hw  $RESULTS_DIR/<test>.hw-cycles/trace_hw.<compiler>.bin \
     --emu $RESULTS_DIR/<test>.hw-cycles/trace_emu.<compiler>.bin \
     --stalls \
     --extended \
     > $RESULTS_DIR/<test>.<compiler>.cycles.compare.txt
   ```

   The bridge invokes this after each EMU run completes. Exit
   code + report content drive classification (next component).

   Note on stall handling: `trace_compare` already implements
   `StallRule`-based attribution that maps stall events (DMA,
   lock, port) to their resolution events with per-pair gap
   deltas (`hw_gap - emu_gap`). The user's prior concern about
   "DMA stall time differences derailing comparison" is partially
   addressed by this existing logic; the validation step will
   surface whether it's complete enough for our cases.

5. **Bridge classification + drift surfacing**

   Bridge reads the compare report and classifies the run per the
   rules in the next section. Adds a new column to the results
   table; `--show-cycle-drift` (a separate small script) reports
   tests by drift severity for manual triage.

## Classification rules

### Pre-check: was trace injection attempted?

Bridge already knows this via the `HW_CYCLES_TRACED_MLIR` env var
set during `compile_one` in Phase B. If injection wasn't attempted
(e.g., test has no `aie.mlir` source — Phase B Limitation 3), skip
cycle comparison entirely. The cycle column shows `—`. Test result
unchanged.

### With injection attempted, four outcomes

| Outcome | Meaning | Action |
|---------|---------|--------|
| `MATCH` | Both sides traced. `cycles.EMU / cycles.HW` ratio in [0.5, 2.0] AND no per-event divergence over `trace_compare`'s threshold. | Cycle column: `MATCH(<ratio>)`. Test result unchanged. |
| `BUDGET` | Both sides traced. Ratio out of bounds OR per-event divergence over threshold. | Cycle column: `DRIFT(ratio=<r>, max_gap=<n>c)`. Test result classified as `BUDGET`. |
| `EMPTY` | Both sides traced but neither emitted events (Phase B Limitation 1: scalar kernels with default event set). | Cycle column: `EMPTY`. Test result unchanged. **Summary surfaces a count** of EMPTY tests. |
| `EMU_TRACE_BUG` or `HW_TRACE_BUG` | Same xclbin, same trace setup, exactly one pipeline emitted events. | Test result = `FAIL` with `reason=<which>_TRACE_BUG`. Compare report path logged. **No silent acceptance.** |

### Tolerance defaults and overrides

- Ratio bounds: `[0.5, 2.0]` (chosen with the user; intentionally
  generous; tighten as data accumulates)
- Per-event divergence threshold: `trace_compare`'s existing
  `DIVERGE_THRESHOLD = 10 cycles` (no change)
- Per-test overrides: `scripts/cycle-drift-overrides.txt` with
  format `<test_name> <ratio_lower> <ratio_upper> [max_gap_cycles]`.
  Loaded similarly to `hw-quarantine.txt`. Bridge logs which tests
  used overrides in summary.

### Surfacing and counts

Summary section additions:

- Per-compiler line: `Cycle drift: N MATCH, N BUDGET, N EMPTY, N skipped (no inject)`
- Standalone block listing test+compiler pairs classified `BUDGET`
  and `EMU_TRACE_BUG`/`HW_TRACE_BUG`, with their cycle column
  contents and report paths.
- Standalone block listing `EMPTY` tests as a reminder that the
  default event set isn't enough for them — implicit nudge toward
  expanding the event set.

## Bridge flag surface

| Flag | Meaning | Implies |
|------|---------|---------|
| `--with-hw-cycles` (Phase B) | Run HW trace pipeline; emit `cycles.HW.*.txt` | (none) |
| `--with-cycle-diff` (this design) | Run EMU trace pipeline + comparison; emit `cycles.EMU.*.txt` and `cycles.compare.*.txt` | `--with-hw-cycles` |
| `--no-timeout` (existing) | Disable EMU wall-clock timeout entirely | (overrides dual-bound logic) |

`--with-cycle-diff` is the integrated "do everything" flag; it
implies `--with-hw-cycles` so users don't need both.

## Naming conventions

Reuses Phase B's `.hw-cycles` subdirectory naming for symmetry
(renaming would create churn; the directory now stores both HW and
EMU trace bytes, despite the directory name).

| Artifact | Path | Status |
|----------|------|--------|
| HW trace bytes | `RESULTS_DIR/<test>.hw-cycles/trace_hw.<compiler>.bin` | Phase B currently writes `trace.<compiler>.bin`; rename for symmetry as part of this plan |
| HW cycles | `RESULTS_DIR/<test>.<compiler>.cycles.HW.txt` | Already shipped (Phase B) |
| EMU trace bytes | `RESULTS_DIR/<test>.hw-cycles/trace_emu.<compiler>.bin` | New |
| EMU cycles | `RESULTS_DIR/<test>.<compiler>.cycles.EMU.txt` | New |
| Compare report | `RESULTS_DIR/<test>.<compiler>.cycles.compare.txt` | New |
| Drift overrides | `scripts/cycle-drift-overrides.txt` | New |
| Drift surface tool | `scripts/show-cycle-drift.sh` | New |
| Trace-incompat list | `scripts/trace-incompat-tests.txt` | New |

## Dual-bound EMU timing

Two distinct bounds. Cycle budget is the principled measurement;
wall-clock timeout is the safety net.

```
XDNA_EMU_MAX_CYCLES = ceil(HW_cycles * 2.0)
EMU_timeout_seconds = max(600, ceil(HW_cycles * 2.0 * SECONDS_PER_CYCLE))

where SECONDS_PER_CYCLE = 1e-3  (≈ 1000 sim cycles/sec, conservative;
                                 to be tuned empirically against
                                 vector_scalar_using_dma EMU runs)
```

For `vector_scalar_using_dma` (HW = 41,181 cycles): budget = 82,362
cycles, timeout = `max(600, 82.4) = 600 s`. For a hypothetical
1M-HW-cycle test: budget = 2M cycles, timeout = `max(600, 2000) =
2000 s`. Floor protects small tests; scales out for big ones.

For tests with no HW cycles file:
- No cycle budget set (EMU runs unconstrained, may legitimately
  loop until host kills it via the wall-clock timeout)
- Wall-clock timeout = 600 s (existing behavior)

`--no-timeout` continues to override both — passes through unchanged.

The `SECONDS_PER_CYCLE` constant is a tunable in the bridge script.
First implementation hardcodes 1e-3; a follow-up commit (after
empirical measurement on a few real EMU runs) calibrates it. We
should observe the actual wall-clock duration of
`vector_scalar_using_dma` and several others to set this honestly.

## Trace-related compile failures

The Phase B validation found that some tests' aiecc compile fails
specifically when injection is on (e.g., `ctrl_packet_reconfig`
hits "Trace lowering pipeline failed" because `AIEInsertTraceFlows`
clashes with `aie.packet_flow`). Today, the bridge reports a
generic `COMPILE-FAIL` with no indication that the trace path is
the cause.

Fix: when `WITH_HW_CYCLES=true` and the compile of the injected
MLIR fails, the bridge appends `(traced)` to the failure reason,
producing `COMPILE-FAIL(traced)`. Detection is local — the bridge
knows the compile it just attempted was on the injected MLIR. No
re-compile cost, no extra plumbing.

A `scripts/trace-incompat-tests.txt` file (similar to
`hw-quarantine.txt`) lists known trace-incompatible tests so the
bridge can skip injection for them up front and avoid the noisy
compile failure on every run. Bridge logs which tests it skipped
and why in summary.

## Error handling

Each step has a clean failure mode that doesn't tank the underlying
test PASS/FAIL:

| Failure | Result column | Test result |
|---------|---------------|-------------|
| EMU trace capture fails (file not produced) | `EMU_TRACE_BUG` | `FAIL`, reason set |
| `trace_compare` non-zero exit | `COMPARE-ERR` | unchanged; report path logged |
| `trace-to-cycles` failure on EMU side | `EMU_PARSE-ERR` | unchanged; same handling as Phase B |
| `trace-to-cycles` failure on HW side | `HW_PARSE-ERR` | unchanged; same handling as Phase B |
| Compile of injected MLIR fails | `COMPILE-FAIL(traced)` | `FAIL`; user can retry without `--with-hw-cycles` |
| EMU exceeds `XDNA_EMU_MAX_CYCLES` | `BUDGET` | unchanged from Phase A semantics; halt_reason from `XDNA_EMU_STATUS:` line drives this |

The asymmetry-as-bug rule is the only case where cycle-comparison
results actively change the test PASS/FAIL. All others are
informational columns.

## Testing approach

This is an integration-heavy effort with little new code, so
testing is mostly empirical validation rather than unit tests.

1. **Validation gate** (Component #1 above): the Phase E plan
   doesn't proceed past Task 1 unless EMU produces `parse_trace`-readable
   output for `vector_scalar_using_dma`. If it doesn't, we have a
   real bug to fix before integration.
2. **Per-task smoke tests**: each component, when wired in,
   gets validated against `vector_scalar_using_dma` end-to-end.
3. **Batch validation**: rerun the Phase B 7-test batch under
   `--with-cycle-diff`. Expected:
   - `vector_scalar_using_dma` (chess): MATCH or modest DRIFT
   - `cascade_flows` (chess): EMPTY or single-event DRIFT (Phase
     B Limitation 2)
   - `add_blockwrite`, `add_one_objFifo`, `add_one_using_dma`
     (both compilers): EMPTY (no INSTR_VECTOR events)
   - `column_specific`: skipped (no `aie.mlir`, Phase B Lim 3)
   - `ctrl_packet_reconfig`: `COMPILE-FAIL(traced)` (Phase B Lim 4)
4. **Negative tests**: a synthetic test where EMU has a
   deliberately-injected cycle drift (e.g., add a constant offset
   to one event type's emission cycle) should be classified as
   `BUDGET`, proving the detector works.

Unit tests where they make sense:
- Bridge classification logic: a small bash test harness that
  feeds canned compare reports and checks the resulting column
  text. Lives next to `scripts/cycle-drift-overrides.txt`.
- `trace-to-cycles.py` already has unit tests from Phase B.
- `trace_compare` already has Rust unit tests (`src/trace/compare.rs`).

## Future work (not in Phase E scope)

These are real follow-ons but explicitly excluded from this
design's task list. Captured here so the next person knows the
hooks exist.

1. **Expand default event set in `mlir-trace-inject.py`** to
   include `INSTR_LOAD`, `INSTR_STORE`, DMA port events. Drives
   `EMPTY` count toward zero. Each kernel that touches memory
   would then emit some events.
2. **Trace event sweeping**: rebuild `trace-sweep.py`-style
   functionality on top of `mlir-trace-inject.py`. The injector
   already accepts a custom event list (parameterizable);
   `src/fuzzer/trace_sweep.rs` defines 13 event groups covering
   100+ AIE2 core events. Loop = "for each group, inject with
   that group's events, run, capture trace, merge results."
3. **Hook trace into `.py`-only MLIR sources** (Phase B
   Limitation 3): inject from inside the IRON Python flow rather
   than at MLIR text level — call `configure_trace`/`start_trace`
   from inside the test's design when `WITH_HW_CYCLES` is on.
4. **Investigate trace+ctrlpkt incompatibility** (Phase B
   Limitation 4): upstream mlir-aie pass-ordering issue between
   `AIEInsertTraceFlows` and `aie.packet_flow`. May require
   coordination with mlir-aie maintainers.
5. **Empirical `SECONDS_PER_CYCLE` calibration**: measure EMU
   wall-clock vs simulated cycles across a representative test
   set; replace the `1e-3` constant with a measured value and a
   note on its derivation.
6. **Tighten ratio bounds** as data accumulates. `[0.5, 2.0]` is
   a starting point; if real-world drift is mostly within
   `[0.8, 1.25]`, tighten and reclassify the outliers as
   `BUDGET` to surface them.

## Risks called out

1. **EMU trace emission may not be HW-binary-compatible despite
   the infrastructure existing.** The Explore agent's verdict
   ("scaffolding-complete, awaiting validation") is encouraging
   but unverified. Component #1 explicitly gates the rest of the
   plan on this validation.
2. **`trace_compare`'s stall attribution may not handle our
   actual divergence cases.** Existing logic targets specific
   stall types (lock, port, DMA-stalled-lock); real EMU/HW
   divergences may have causes outside that catalog. Mitigation:
   start with `--stalls` enabled, examine compare reports
   manually for the first batch, file follow-up tickets for any
   uncovered stall types.
3. **Per-tile capture and routing under contention.** The Explore
   agent flagged "multi-tile trace routing through stream switch
   under contention" as untested. `cascade_flows` exercises this;
   if it surfaces problems, fix them in `coordinator.rs` Phase
   3a/3b/3c logic.
4. **Wall-clock timeout extension may exceed CI patience.** A
   1B-cycle test would get a 1000 s+ timeout. CI environments may
   not tolerate that. Mitigation: `--no-timeout` overrides;
   long-running tests can be marked in
   `cycle-drift-overrides.txt` with a custom upper-bound flag
   later if needed.

## Relationship to existing plans

- **Supersedes**: Phase D.3 (Task 10) and Phase C (Tasks 11-15)
  in `docs/superpowers/plans/2026-04-22-cycle-budget-testing.md`.
  The parent plan's Phase B Pivot Note should be updated to
  reference this design once approved.
- **Builds on**: Phase B (`docs/superpowers/plans/2026-04-22-phase-b-trace-cycle-capture.md`),
  shipped 2026-04-22. Phase E reuses Phase B's
  `tools/trace-to-cycles.py`, `bridge-runner/bridge-trace-runner`,
  `tools/mlir-trace-inject.py`, the `--with-hw-cycles` flag, and
  the `RESULTS_DIR/.hw-cycles/` directory layout.
- **Coexists with**: the older `--trace` path
  (`tools/trace-prepare.py`). No interaction — different
  codepaths, different artifacts. Phase B's policy of coexistence
  carries forward.

## Open questions deliberately deferred

These don't block Phase E but should be revisited later:

- Should the bridge cache `trace_compare` reports across runs to
  detect *new* drift vs preexisting drift? Probably yes, but
  needs a story for what counts as "the same compare invocation"
  (xclbin hash? source MLIR hash?).
- How does Phase E interact with `--sweep`? Originally `--sweep`
  drove the deprecated `tools/trace-sweep.py`. With trace-sweep
  resurrected as Future Work #2, `--sweep --with-cycle-diff`
  could mean "run the comparison across all event groups." Out
  of Phase E scope.
- Should `EMPTY` runs nudge the user toward enabling
  `--events INSTR_LOAD,INSTR_STORE` (or whatever the future
  enhanced default is)? A small UX nudge in summary, maybe.
