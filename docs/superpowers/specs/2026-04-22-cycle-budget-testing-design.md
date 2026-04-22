# Cycle-Budget Testing — Design Spec

**Date:** 2026-04-22
**Status:** Design approved + load-bearing open questions resolved, ready for implementation plan
**Context:** Subsystem 7 (ISA Execute) landed and produced a bridge-test
regression: 10 Chess + 10 Peano tests went PASS → TIMEOUT at the wall-clock
`timeout 120` boundary. The cause could be host-side slowdown (EMU taking
longer wall-clock time to emulate the same kernel) or cycle-modeled slowdown
(EMU spending more emulated cycles to complete the same kernel work) — and
log-volume alone can't distinguish them (both produce smaller logs for
different reasons). Today's bridge harness can't tell these apart; both
surface as a timeout.

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

## Framing: what `ACTIVE_CORE` actually measures

`XAIE_EVENT_ACTIVE_CORE` ticks only during cycles the core is in Execute
state. It does *not* tick during DMA stalls, lock waits, cascade stalls, or
idle. On HW and (after Phase D) on EMU alike.

So the comparison in Phase C isn't "wall-clock work" — it's
"instruction-executing cycles for this kernel." For identical instruction
streams (we run the same `.xclbin` on both), the expected relationship is
**`EMU_cycles ≈ HW_cycles`**.

Directional caveats:
- `EMU_cycles < HW_cycles` — EMU's pipeline/hazard/scheduling model is
  *simpler* than HW's silicon behavior. EMU might compress instructions
  into fewer cycles if it misses a hazard HW enforces.
- `EMU_cycles > HW_cycles` — EMU over-counting execute cycles. Either a
  regression in EMU's cycle model, or Phase D didn't eliminate all
  level-event drift (stall cycles still being counted as execute cycles).

The 3× tolerance in Phase C is a **runaway cap**, not a slack allowance.
It catches pathological regressions (EMU taking many multiples of HW
cycles). As EMU's modeling tightens, tolerance tightens. The drift
helper (`show-cycle-drift.sh`) exposes bi-directional drift for
investigation regardless of gating.

This framing is distinct from wall-clock: EMU spending more *host* time
per emulated cycle is a Phase A / bridge concern (host-side perf), not
reflected in `ACTIVE_CORE` at all.

---

## Phase B — HW Cycle Capture

### Mechanism: XRT register API (not aie-rt)

Bridge test.exe binaries talk to XRT via `xrt::hw_context`, not aie-rt
directly. XRT exposes tile register access through `xrt::device` via
`xrt_aie.h`:

```cpp
uint32_t device.read_aie_reg(col, row, offset);
void     device.write_aie_reg(col, row, offset, value);
```

`xrt::device` is reachable from `hw_context.get_device()`. Backed by DRM
ioctls `DRM_AMDXDNA_READ_AIE_REG` / `DRM_AMDXDNA_WRITE_AIE_REG` in the
xdna kernel driver, coexists cleanly with the shim's own `XAie_DevInst`.
This lets us program perf counters without linking aie-rt or standing up
a second `XAie_DevInst`.

Register offsets and bit layouts come from the AM025 database that
`regdb.rs` already loads:

- Perf counter 0 control register (start-event field, stop-event, reset).
- Perf counter 0 value register (32-bit counter readout).

`XAIE_EVENT_ACTIVE_CORE` event ID also lives in the register DB /
aie-rt's event enumeration; helper encodes the event ID constant
directly (constant is stable across the AIE2 lifetime).

### What we build

A small C++ helper linked into every mlir-aie bridge test.exe:

```cpp
class CycleCounterHelper {
public:
  // Constructed once after hw_context is live.
  CycleCounterHelper(xrt::hw_context& ctx,
                     const std::vector<std::pair<uint16_t, uint16_t>>& compute_tiles);

  // Write control reg + zero the counter on each tile.
  void start();

  // Read each counter, write `{col row cycles}` lines to out_path.
  void readout(const std::string& out_path);

private:
  xrt::device device_;
  std::vector<std::pair<uint16_t, uint16_t>> tiles_;
};
```

(Class vs free-function shape is a plan-phase decision; what matters
here is that both operations need `xrt::device` and the tile list, so
a helper object is the natural shape.)

`start` iterates every compute tile and, via `device.write_aie_reg`,
performs two register accesses per tile:

1. **Masked write to the control register** at
   `PerfCtrlBaseAddr + (counter/2) * PerfCtrlOffsetAdd`, setting the
   Start event field to `ACTIVE_CORE` and the Stop event field to 0
   (no auto-stop). (Since `read_aie_reg` + `write_aie_reg` don't have
   a native masked-write, the helper reads the current value,
   clears the Start/Stop fields, and writes back OR'd with the
   new values.)
2. **Plain write of 0 to the counter value register** at
   `PerfCounterBaseAddr + counter * PerfCounterOffsetAdd`, zeroing the
   count.

This matches aie-rt's
`XAie_PerfCounterControlSet` + `XAie_PerfCounterReset` sequence
(`aie-rt/driver/src/perfcnt/xaie_perfcnt.c:206,550`): the control
register write is a masked write that preserves other bits, and the
counter value register is a separate plain write. Counter runs free
from first core-active cycle through readout.

(Counter index 0 is a convention. If upstreamed, other users may want to
use counter 0 for their own purposes; we'd need to coordinate. For our
own tests today there's no collision.)

`readout` iterates the same tiles via `device.read_aie_reg`, and writes a
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
depend on which path we take. `../mlir-aie/` is a git checkout, so the
practical approach is to commit on a local branch (e.g. `xdna-emu-cycle-budget`)
and track via the script's existing mlir-aie path assumptions. Upstream once
the helper stabilizes.

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
XDNA_EMU_STATUS: halt_reason=<completed|budget|error> cycles=<u64> max_cycles=<u64>
```

- `halt_reason` distinguishes *why* the emulator stopped. `completed` = core
  hit a HALT instruction (kernel done). `budget` = cycle budget reached.
  `error` = crash / FFI fault.
- `max_cycles=0` when unset / unbounded.
- Keyed prefix so bridge can grep cleanly.

This requires the FFI's `XdnaEmuExecStatus` to surface `halt_reason`
explicitly. If it doesn't today, add it — the `halted` boolean alone is
ambiguous (can't distinguish "kernel finished right at budget" from
"budget fired exactly when kernel would have completed"). Checking the
FFI struct's current shape is a plan-phase concrete task.

**3. Bridge script parses status.**

In `scripts/emu-bridge-test.sh`, after each EMU run, grep the captured log
for `XDNA_EMU_STATUS:`. If `halt_reason=budget`, mark the test `BUDGET` — a
distinct result from `TIMEOUT` (wall-clock) and `FAIL` (functional).

Bridge script already captures stderr (`emu-bridge-test.sh:1255` uses
`> "$log_file" 2>&1`), so the parser can grep the log directly — no
capture-redirection change needed.

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
  (not mid-bundle); `run()` returns with `halt_reason=budget`,
  `cycles_executed == max_cycles` (modulo partial bundle completion).
- When kernel hits HALT instruction before cap, `run()` returns with
  `halt_reason=completed`, `cycles_executed <= max_cycles`.
- Emulator state stays consistent on halt — bundle either completes or
  doesn't start.

### FFI changes required in Phase A

Verified against `crates/xdna-emu-ffi/src/lib.rs:95-99` and
`crates/xdna-emu-ffi/src/execution.rs:70-182`:

- `XdnaEmuExecStatus` has `{ result, cycles_executed, halted }` today;
  **add** a `halt_reason` field (C-ABI enum: `Completed | Budget | Error`).
  The existing `halted: bool` is ambiguous and stays collapsed behind the
  new enum (or gets removed; plan decides).
- `execution.rs:108` uses `while cycles < max` — with `max = 0` the loop
  doesn't run at all. Change to `while max == 0 || cycles < max` so the
  "0 = unbounded" semantic holds at the run loop, not just in the
  plugin wrapper.
- Populate `halt_reason` at the three exit points in `xdna_emu_run`:
  natural halt (EngineStatus::Halted + syncs done) → `Completed`; loop
  condition falling out with `cycles >= max` and no halt → `Budget`;
  any error branch → `Error`.

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

**Option 3 — Event-emission-per-cycle.**
Emit `ACTIVE_CORE` as an event every cycle the core executes, and have
`handle_event()` itself increment the counter on matching fires. Most
faithful to aie-rt's documented event-driven model (counters tick on
event occurrence, not on a separate per-cycle callback). Largest
surface — changes the counter increment path globally.

Plan task picks based on minimum diff + clearest invariant. All three
achieve the same observable behavior; Option 2 probably has the
smallest diff.

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
| 2 | A.1 | Plugin env-var reader + XDNA_EMU_STATUS line + bridge parser | `feat(plugin,bridge): wire XDNA_EMU_MAX_CYCLES; emit status line` |
| 3 | A.2 | Wall-clock timeout relaxed to 600s + `--no-timeout` flag | `feat(bridge): relax wall-clock timeout; add --no-timeout` |
| 4 | D.1 | Audit: confirm HW ACTIVE_CORE semantics + level-event list | `docs: cycle-budget D.1 audit of level-valued perf events` |
| 5 | D.2 | Refactor PerfCounterBank for level-event semantics + unit tests | `fix(perf_counters): ACTIVE_CORE ticks only during core execute` |
| 6 | D.3 | Integration spot-check vs Phase B HW references | (no commit unless fix needed) |
| 7 | C.1 | `*.hw.cycles.txt` parser | `feat(bridge): parse hw cycles artifacts for budget` |
| 8 | C.2 | Override file loader | `feat(bridge): cycle-budget override file loader` |
| 9 | C.3 | Budget calculation + env var export | `feat(bridge): apply per-test XDNA_EMU_MAX_CYCLES budget` |
| 10 | C.4 | Status → BUDGET result classification | (rolled into earlier commit if trivial) |
| 11 | C.5 | Results column merge + `show-cycle-drift.sh` | `feat(bridge): merge cycle counts into results; add drift helper` |

Each commit independently runnable. Rollback at any step is a straight
revert.

Note: rebuilding bridge test.exe binaries after B.1 is a procedural step
(part of running `scripts/emu-bridge-test.sh --compile`), not a separate
git commit.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Perf-counter semantic drift (EMU vs HW) | Addressed head-on in Phase D rather than deferred |
| mlir-aie patch rebasing | Keep helper surgical; upstream ASAP |
| Multi-phase kernels with idle gaps | Bridge tests are single-phase; revisit if multi-phase tests appear |
| Shell-script complexity | Factor budget logic into one function (~30 lines). If it grows, move to a small Python helper |
| `XDNA_EMU_STATUS:` parser fragility (stderr buffering, TTY quirks, output interleaving) | Distinctive prefix; anchored grep; test against real bridge logs before gating on it |
| Counter width overflow (32-bit) | Bridge tests stay well under 4G cycles per core; not a practical concern for this workload |
| Bridge gate fires only on `EMU > HW` drift; under-drift invisible | `show-cycle-drift.sh` surfaces drift in either direction for investigation. Gate is upper-bound only (catching regressions); drift helper catches under-counting that would indicate incomplete modeling |

## Success Criteria

After all four phases land:

- Every HW-path bridge test produces a `{t}.hw.cycles.txt` with sensible numbers.
- Every EMU-path run produces an `XDNA_EMU_STATUS:` line.
- Bridge results show cycle counts merged into status columns.
- Today's 20 regressions produce either `BUDGET` (cycle-modeled slowdown)
  or `TIMEOUT` (host-side wall-clock slowdown). Whichever class it is, we
  know, and the perf-hunt workstream that follows attacks the right target.
- `scripts/show-cycle-drift.sh` gives an at-a-glance ranked drift view,
  ready for the perf-hunt workstream to follow.

## Open Questions (Deferred to Plan Phase)

**Resolved during spec review** (see sections above for how each is now
handled):

- **Phase B tile-register access path.** Resolved: use
  `xrt::device::read_aie_reg` / `write_aie_reg` from `xrt_aie.h` against
  AM025 register offsets. No aie-rt linkage needed in test.exe.
- **Bridge stderr capture.** Resolved: `emu-bridge-test.sh:1255` already
  uses `> "$log_file" 2>&1`. Parser reads the existing log.
- **FFI `halt_reason` field.** Resolved: struct currently has only
  `halted: bool` (`crates/xdna-emu-ffi/src/lib.rs:95-99`); Phase A adds
  a C-ABI enum `halt_reason` field. Loop condition at
  `execution.rs:108` also changes so `max_cycles=0` reads as unbounded.
- **mlir-aie patch deployment.** Resolved to a practical default: commit
  on a local branch in the existing `../mlir-aie/` checkout. Upstream
  when the helper stabilizes.

**Non-blocking — choose during planning:**

- **Exact location of `setup` / `readout` call sites** in mlir-aie's test
  main template (requires reading the current template).
- **Concrete source for partition tile layout.** The helper needs the list
  of `(col, row)` compute tile pairs in the active partition. Sources to
  investigate: xclbin metadata, test-harness-provided config, MLIR-generated
  constants.
- **Option 1 / 2 / 3 for the D.2 refactor.** Pick based on diff size after
  reading current tick call sites. Option 2 is the default presumption.
- **Override multiplier — integer or float.** Lean integer for simplicity
  unless a real test needs fractional.
- **Helper shape: class vs two free functions.** Class is cleaner for
  shared state (device, tile list). Either works; plan picks whichever
  is smaller diff.
