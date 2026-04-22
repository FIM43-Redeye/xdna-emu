# Cycle-Budget Testing — Design Spec

**Date:** 2026-04-22
**Status:** Design approved, ready for implementation plan
**Context:** Subsystem 7 (ISA Execute) landed and produced a bridge-test
regression: 10 Chess + 10 Peano tests went PASS → TIMEOUT at the wall-clock
`timeout 120` boundary. Log-volume evidence (72MB → 35MB) suggests the
slowdown is host-side (wall-clock), not cycle-modeled (EMU_cycles > HW_cycles).
Today's bridge harness can't tell these apart — both surface as a timeout.

## Goal

Separate "cycle-count correctness" from "wall-clock performance" in the
bridge test suite. Give the harness two independent signals, so a slowdown
is classified as one or the other and we can treat them accordingly.

## Non-Goals

- Fix the actual Subsystem 7 perf regression. That's the next workstream;
  this one just gives us the instrumentation to distinguish signal types.
- Cycle-accurate DMA / NoC / memory-bank modeling in EMU. Out of scope;
  EMU's cycle model will tighten over time and the tolerance here will
  tighten with it.
- Coverage of HW perf counter modes beyond `ACTIVE_CORE`. Future extension
  if we want stall-type breakdowns.

## Phasing: B → A → D → C

Four phases, landed in order. Each is independently useful:

| Phase | What | Value on landing |
|-------|------|------------------|
| **B** | HW cycle capture via `XAie_PerfCounter` | Every HW bridge test produces `{t}.hw.cycles.txt` |
| **A** | EMU cycle budget plumbing | `XDNA_EMU_MAX_CYCLES` env var makes EMU stoppable and observable |
| **D** | EMU ACTIVE_CORE counter semantic fix | Makes the comparison in C trustworthy |
| **C** | Bridge comparison gate | Actual gate: fails tests where `EMU_cycles > 3 × HW_cycles` |

Order rationale: B measures HW before we know what EMU "should" do. A makes
EMU's cycle budget observable without gating on anything yet. D tightens EMU's
counter so the comparison in C reflects reality. C ties it all together.

## Framing: EMU cycles should be ≤ HW cycles

EMU's DMA is near-instantaneous today; NoC latency is not modeled; stream
switch crossings are free. Result: for the same kernel, EMU runs fewer
cycles than HW (it skips stalls HW actually incurs).

This matters for tolerance interpretation. A 3× upper bound is a **runaway
cap**, not a slack allowance. The expected steady state is `EMU_cycles ≤
HW_cycles`. If EMU_cycles >> HW_cycles, that's a cycle-modeling regression
in EMU — a real signal worth investigating.

As EMU's cycle modeling tightens, tolerance tightens too. Today's 3× is
deliberately loose because we know EMU underestimates.

---

## Phase B — HW Cycle Capture

### What we build

A small C++ helper linked into every mlir-aie bridge test.exe:

```cpp
// Called after XRT context / AIE partition is live, before kernel launch.
void xdna_cycle_counter_setup(xrt::hw_context& ctx, const PartitionLayout& layout);

// Called after kernel completion, before context teardown.
void xdna_cycle_counter_readout(const std::string& out_path);
```

`setup` iterates every compute tile in the active partition and configures
perf counter 0 on each core with `XAIE_EVENT_ACTIVE_CORE` as both start and
reset event (free-running).

`readout` iterates the same tiles, reads each counter, and writes a
plain-text file:

```
# col row cycles
0 2 8421
0 3 8387
0 4 8396
1 2 8410
```

One line per compute tile. Header line for human readability. Comment-only
lines ignored by the parser. Path: `{test_build_dir}/{compiler}/{t}.hw.cycles.txt`.

### Where it lands

Under mlir-aie's `test/npu-xrt/` scaffolding:

- `test/npu-xrt/cycle_counter.{h,cc}` — helper implementation.
- CMake change that links the helper into every test.exe built under
  `test/npu-xrt/`.
- One-line additions to the shared test-main template: setup after context
  creation, readout before exit.

Start with a local patch; upstream to mlir-aie once stable. Design does not
depend on which path we take.

### Why `XAIE_EVENT_ACTIVE_CORE`

Counts cycles the core is actually executing an instruction. Doesn't tick
during stall, disable, or reset. For cycle-budget comparison this is the
right signal — "how many instructions' worth of time did this core spend on
the kernel." Raw clock-cycle count would include stall cycles EMU doesn't
model today, polluting the comparison.

### Coverage notes

- Shim tiles and memtiles have no cores; no counter there. File contains
  compute-tile entries only.
- Tests that never launch a kernel on a given core produce `0` for that
  tile. Bridge takes `max()` across tiles for the budget, so zeros don't
  distort.
- Counter width is 32 bits on AIE2. Bridge-test kernels are well under
  4G cycles per core; overflow not a practical concern.
- If a test crashes mid-kernel, no file is written, and bridge skips the
  cycle-budget check for that run.

### Testing Phase B

After landing: run bridge once, verify `*.hw.cycles.txt` files appear for
every HW-path test that completes. Spot-check a handful against intuition
(a no-op should be tiny; a dense matmul should be large). No EMU-side
change yet, so no regression surface.

---

## Phase A — EMU Cycle Budget

### What's already in place

- `xdna_emu_ffi::xdna_emu_set_max_cycles(handle, u64)` — FFI entry point.
- `xdna_emu_ffi::xdna_emu_run() -> XdnaEmuExecStatus { cycles_executed, halted, ... }`.
- `xrt-plugin/src/transport_inprocess.cpp:100` — plugin resolves the symbol
  via dlsym.
- **Missing:** plugin reads no env var, never calls `set_max_cycles`, never
  logs status.

### What we build

**1. Plugin reads env var on context creation.**

In `transport_inprocess.cpp`, when the emulator handle is constructed, read
`XDNA_EMU_MAX_CYCLES`:

- Unset → unbounded (don't call `set_max_cycles`; preserve current behavior).
- Set to `0` → explicit unbounded (same as unset). Useful when scripts
  want to override-off one test without unsetting the env globally.
- Set to any other u64 → call `sym_set_max_cycles_(handle, value)`.
- Unparseable → log a warning, treat as unset.

**2. Plugin emits status after run.**

Wherever the plugin receives `XdnaEmuExecStatus` from `xdna_emu_run()`, emit
a line to stderr before returning:

```
XDNA_EMU_STATUS: halted=<bool> cycles=<u64> budget_exceeded=<bool>
```

- `budget_exceeded = halted && max_cycles_was_set && cycles_executed >= max_cycles`.
- Keyed prefix so bridge can grep cleanly.

**3. Bridge script parses status.**

In `scripts/emu-bridge-test.sh`, after each EMU run, grep the captured log
for `XDNA_EMU_STATUS:`. If `budget_exceeded=true`, mark the test `BUDGET`
— a distinct result from `TIMEOUT` (wall-clock) and `FAIL` (functional).

**4. Wall-clock safety net relaxed.**

In `scripts/emu-bridge-test.sh`, the `timeout 120` wrapper around each EMU
invocation becomes `timeout 600`. Still
present as the escape hatch for genuinely hung processes (emulator
livelock, FFI deadlock). Looser so it doesn't fire ahead of the cycle
budget. Parallel-running nature of the EMU suite mitigates concerns
about long test stacking.

**5. Bridge `--no-timeout` flag.**

Add a CLI option to `scripts/emu-bridge-test.sh`. When passed, replaces
the `timeout 600 ...` invocation with a bare invocation. For genuinely
long runs (heavy debug builds, large suites, `-j1` serial mode). Docs
note it's unsafe — a truly hung process hangs the run until Ctrl-C —
but when the tests are known-long, you need it.

### FFI cycle-budget semantics

- `set_max_cycles(0)` → unbounded (no cap).
- When cap is reached, emulator voluntarily halts **at bundle boundary**
  (not mid-bundle); `run()` returns with `halted=true`, `cycles_executed
  == max_cycles` (modulo partial bundle completion).
- Emulator state stays consistent on halt — bundle either completes or
  doesn't start.

### Result classification changes

Today: `| TEST | COMPILER | HW | EMU | TRACE |` with EMU values
`PASS / FAIL / TIMEOUT / ERROR`.

After Phase A, EMU column distinguishes:

- `PASS` — ran to completion, test passed.
- `FAIL` — ran to completion, test failed.
- `BUDGET` — voluntarily halted on cycle budget.
- `TIMEOUT` — wall-clock timeout fired (escape hatch; should be rare).
- `ERROR` — emulator crashed or FFI error.

Phase A alone does not *set* a budget (no env var exported yet). It just
makes the budget settable and observable. Phase C sets it per-test.

### Testing Phase A

After landing:
- `XDNA_EMU_MAX_CYCLES=1` on a single test → expect `BUDGET`, `XDNA_EMU_STATUS`
  line in log.
- No env var → baseline behavior unchanged (regression check).
- `XDNA_EMU_MAX_CYCLES=999999999999` → expect `PASS` / `FAIL` as before.
- `XDNA_EMU_MAX_CYCLES=0` → same as unset.
- `--no-timeout` → runs to completion with no wall-clock cap.

---

## Phase D — EMU ACTIVE_CORE Counter Semantic Alignment

### The drift

EMU's current `PerfCounterBank` model:

- `handle_event(start_event)` transitions the counter to `Active`.
- Coordinator calls `tick()` every cycle → increments all `Active` counters.
- Result: once started, the counter ticks every cycle until stopped,
  regardless of whether the core is actually executing.

HW `XAIE_EVENT_ACTIVE_CORE`:

- Level signal — high only on cycles the core is in Execute state.
- Counter ticks only during cycles the signal is high. Stalls, disables,
  idle waits don't tick.

Fine when the core is continuously active (short, well-behaved kernels).
Diverges when the core stalls on lock wait, cascade wait, or explicit idle
— all reachable from EMU's existing state model.

### What HW actually counts (verified in D.1)

First step is confirmation: does HW `ACTIVE_CORE` tick during DMA stalls,
lock-wait stalls, cascade stalls, or none? Expected answer (from AM025 +
aie-rt): only during Execute state, not during any of the stall states.

If confirmed, the DMA-delay story is actually fine: EMU doesn't model DMA
delays, HW doesn't count them in `ACTIVE_CORE` either — both sides report
"cycles the core ran instructions." Lock-wait and cascade-wait, however,
are modeled in EMU — and the counter currently ticks during those (wrong).

### The fix

Make counter ticking conditional on the configured start_event's
level-assertion state. Two implementation shapes (finalize in plan):

**Option 1 — Level-predicate on tick.**
`PerfCounterBank::tick()` accepts the current level-assertion state of each
relevant level event from the calling context. Ticks only counters whose
start_event is asserted this cycle.

**Option 2 — Move tick site.**
For counters whose start_event is `ACTIVE_CORE`, move the tick invocation
into the core's execute path: tick once per successfully-executed
instruction. Narrower surface; doesn't change `tick()`'s contract for
pulse-event counters.

Plan task will pick based on minimum diff + clearest invariant. Both
achieve the same behavior for the workload at hand.

### Audit scope

`ACTIVE_CORE` is the primary case. Other level-valued events reachable
from EMU state:

- `ACTIVE_MEMORY_STALL` (core stalled on memory contention)
- `ACTIVE_LOCK_STALL` (core stalled on lock acquire)
- `ACTIVE_CASCADE_STALL` (core stalled on cascade full/empty)

If fixing these falls out of the same refactor (same plumbing, same
predicate), include them. If any needs significant separate plumbing
(more than a few lines), **defer to a follow-up** — not chasing 100%
perf-counter coverage here, just making ACTIVE_CORE trustworthy for
Phase C.

### Test coverage

New unit tests in `src/device/perf_counters/tests.rs`:

- `active_core_ticks_only_while_core_active` — mock tile where core runs
  10 cycles, stalls 5, runs 10 more. Counter reports 20, not 25.
- Mirror test for any other level events fixed in the same pass.
- Regression: existing ~30 pulse-event tests pass unchanged.

**Integration spot-check.** After Phase B lands (so HW cycles are
available), run a couple of simple tests through EMU with the same
counter config. EMU number should land close to HW (same order of
magnitude, small constant factor max). If wildly off, that's
cycle-modeling drift worth flagging separately.

### Landing

1. **D.1:** Audit — confirm what HW `ACTIVE_CORE` counts; identify which
   level events fit the refactor. One commit, documentation + list.
2. **D.2:** Refactor with unit tests. One commit.
3. **D.3:** Integration spot-check against Phase B HW references. If any
   systematic issue surfaces, a follow-up commit addresses.

### What this does NOT fix

- DMA delay modeling in EMU (EMU's DMA is still near-instant).
- NoC / stream-switch latency.

These are downstream cycle-modeling work. Phase D just makes the counter
itself faithful. Once DMA/NoC modeling lands, the tolerance in Phase C
tightens automatically.

---

## Phase C — Bridge Comparison Gate

### Per-test flow

For each test `t` under compiler `c`:

1. After HW run, Phase B helper has written `{build_dir}/{c}/{t}.hw.cycles.txt`.
2. Before EMU run, bridge reads that file:
   - Skip comment lines.
   - Parse `col row cycles` rows.
   - `hw_max = max(cycles)` across rows.
3. Check `scripts/cycle-budget-overrides.txt` for a per-test override:
   ```
   # test-name                multiplier   reason
   dense_matmul_large         5            high cross-crate PRMX traffic; 3x too tight as of 2026-04
   ```
4. `budget = hw_max * (override_multiplier || 3)`.
5. Export `XDNA_EMU_MAX_CYCLES=$budget`, run EMU, capture status.
6. Record in results table: merged `PASS 8421/8400` format
   (EMU_status + EMU_cycles/HW_cycles). Ratio-column separate only if
   readability suffers.

### Skip cases

- **No `{t}.hw.cycles.txt`** — HW run crashed, was skipped, or test is
  EMU-only. Export `XDNA_EMU_MAX_CYCLES=0` (unbounded). Cycle ratio shown
  as `—`.
- **HW run failed functionally (FAIL, not slow)** — still compute budget
  if cycles file exists. A test can fail functionally on HW but the cycle
  count remains a reference point for EMU modeling.

### Override file format

Plain text at `scripts/cycle-budget-overrides.txt`:

- Fields whitespace-separated: `test-name multiplier reason`. Reason may
  contain spaces — parser reads the first two fields and treats the rest of
  the line as the reason string.
- Comments start with `#`.
- Empty lines ignored.
- Test name: exact string match (no globbing for now).
- Every override must include a reason string. Code review expectation:
  an override without explanation gets rejected. Overrides are where
  perf knowledge lives; they can't be silent.

### Results table

Today: `| TEST | COMPILER | HW | EMU | TRACE |`

After Phase C: merge cycle counts into status columns.
- HW column shows `PASS 8400` (status + cycles).
- EMU column shows `PASS 8421/8400` (status + EMU/HW ratio embedded).
- If width gets unwieldy, fall back to a separate `CYCLE_RATIO` column.

### Diagnostic helper

`scripts/show-cycle-drift.sh` — runs after a bridge test, prints tests
sorted by `EMU_cycles / HW_cycles` descending. Fast way to see which tests
have the largest drift regardless of whether they tripped the budget.
Directly useful for the perf hunt that follows this workstream.

### Testing Phase C

- Rerun today's 20-test regression set. **Expected:** perf issues show
  as `BUDGET` with `EMU_cycles > 3 × HW_cycles`. Wall-clock timeouts
  become rare.
- Pick a known-clean test. Drop its override multiplier to 0.5× actual
  ratio → expect `BUDGET`. Remove override → expect `PASS`. Exercises
  override path.
- Compare bridge results pre-Phase-C vs post-Phase-C on clean tree:
  cycle-ratio column should be ≤ 1.0 for most tests, matching the
  "EMU under-counts DMA stalls that HW doesn't count either" framing.

### Order within Phase C

1. **C.1:** Parser for `{t}.hw.cycles.txt` → `hw_max`.
2. **C.2:** Override file loader (parse + validate).
3. **C.3:** Budget calculation + env var export around EMU invocation.
4. **C.4:** Status parsing + `BUDGET` classification (already produced
   by Phase A; consumed here).
5. **C.5:** Results table column merge + `show-cycle-drift.sh`.

---

## Landing Sequence (Cross-Phase Summary)

| # | Phase | Step | Commit |
|---|-------|------|--------|
| 1 | B.1 | Helper C++ source + CMake glue in mlir-aie | `[mlir-aie] test: add xdna_cycle_counter helper for HW cycle capture` |
| 2 | B.2 | Rebuild test.exe binaries, verify artifacts | `chore: rebuild bridge test.exe binaries with cycle helper` |
| 3 | A.1 | Plugin env-var reader + XDNA_EMU_STATUS line + bridge parser | `feat(plugin,bridge): wire XDNA_EMU_MAX_CYCLES; emit status line` |
| 4 | A.2 | Wall-clock timeout relaxed to 600s + `--no-timeout` flag | `feat(bridge): relax wall-clock timeout; add --no-timeout` |
| 5 | D.1 | Audit: confirm HW ACTIVE_CORE semantics + level-event list | `docs: cycle-budget D.1 audit of level-valued perf events` |
| 6 | D.2 | Refactor PerfCounterBank for level-event semantics + unit tests | `fix(perf_counters): ACTIVE_CORE ticks only during core execute` |
| 7 | D.3 | Integration spot-check vs Phase B HW references | (no commit unless fix needed) |
| 8 | C.1 | `*.hw.cycles.txt` parser | `feat(bridge): parse hw cycles artifacts for budget` |
| 9 | C.2 | Override file loader | `feat(bridge): cycle-budget override file loader` |
| 10 | C.3 | Budget calculation + env var export | `feat(bridge): apply per-test XDNA_EMU_MAX_CYCLES budget` |
| 11 | C.4 | Status → BUDGET result classification | (rolled into earlier commit if trivial) |
| 12 | C.5 | Results column merge + `show-cycle-drift.sh` | `feat(bridge): merge cycle counts into results; add drift helper` |

Each commit independently runnable. Rollback at any step is a straight
revert.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Perf-counter semantic drift (EMU vs HW) | Addressed head-on in Phase D rather than deferred |
| mlir-aie patch rebasing | Keep helper surgical; upstream ASAP |
| Multi-phase kernels with idle gaps | Bridge tests are single-phase; revisit if multi-phase tests appear |
| Shell-script complexity | Factor budget logic into one function (~30 lines). If it grows, move to a small Python helper |
| `XDNA_EMU_STATUS:` parser fragility (stderr buffering, TTY quirks, output interleaving) | Distinctive prefix; anchored grep; test against real bridge logs before gating on it |
| Counter width overflow (32-bit) | Bridge tests stay well under 4G cycles per core; not a practical concern for this workload |
| `EMU_cycles << HW_cycles` always true → comparison meaningless | Tolerance is upper-bound only; the point is catching regressions, not gating on ratio parity. As EMU modeling tightens, tolerance tightens |

## Success Criteria

After all four phases land:

- Every HW-path bridge test produces a `{t}.hw.cycles.txt` with sensible numbers.
- Every EMU-path run produces an `XDNA_EMU_STATUS:` line.
- Bridge results show cycle counts merged into status columns.
- Today's 20 regressions produce `BUDGET` results (cycle-modeled slowdown),
  not `TIMEOUT` (wall-clock) — confirming the slowdown is in cycle count,
  not just host-side overhead. Or vice versa — whichever it actually is.
  Either way, we've separated the signals.
- `scripts/show-cycle-drift.sh` gives an at-a-glance ranked drift view,
  ready for the perf-hunt workstream to follow.

## Open Questions (Deferred to Plan Phase)

- Exact location of the `setup` / `readout` call sites in mlir-aie's test
  main template (requires reading the current template).
- Option 1 vs Option 2 for the D.2 refactor (level-predicate on tick vs
  move tick site). Pick based on diff size after reading the current tick
  call sites.
- Whether override multiplier should be an integer or a float. Lean integer
  for simplicity unless a real test needs fractional.
