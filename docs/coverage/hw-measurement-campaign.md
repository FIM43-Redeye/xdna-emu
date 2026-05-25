# HW measurement campaign -- shim throughput + perf-counter emission

**Status**: PLAN (2026-05-25). Not yet executed.

Concrete spec for the HW-side measurement work that unblocks
items #9 and #10 in [`cycle-accuracy-mission.md`](cycle-accuracy-mission.md).
This doc is the design; a future session executes against it.

The plan is shaped by two findings landed earlier today:

- [`docs/superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md`](../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md)
- [`docs/superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md`](../superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md)

Read both before executing. The ts/soc gotcha in particular is a
prerequisite measurement-discipline rule; every cycle delta in this
campaign uses `soc`, never `ts`.

## Purpose and scope

Two structural cycle-accuracy items are blocked on the same kind of
data: HW trace captures, decoded under the correct ts/soc semantics,
across a small set of calibration tests. This campaign produces that
data set once, then both items get modeled off of it.

**In scope:**

- Item #10 -- shim streaming throughput modeling. Need shim_dispatch
  -> shim_done and shim_dispatch -> memtile_s2mm_done deltas as a
  function of (BD payload size, access pattern). Solve for cold-start
  intercept and per-word streaming rate.

- Item #9 -- perf-counter-driven trace event emission (LOCK_STALL et
  al.). Need HW LOCK_STALL event counts and inter-event cycle deltas
  on a set of tests that exercise short-stall and long-stall regimes,
  correlated against each test's PERF_CTRL CDO writes.

**Explicitly out of scope (deferred to model-build phase):**

- Building the CDO PERF_CTRL parser (item #9b). Comes after the
  campaign; the data tells us what the model needs to compute.
- Implementing throughput-bound streaming in the DMA stepping
  state machine (item #10). Comes after the rate is known.
- MEMORY_STALL / STREAM_STALL / PORT_STALL cadence -- LOCK_STALL is
  the only state event we have a clean signal on. The other three
  ride along if their cadence happens to fall out of the same
  PERF_CTRL config, but we don't design experiments specifically for
  them yet.

## Calibration corpus

One corpus serves both items. Each test contributes anchors for at
least one of the two items.

| Test | Anchors usable for #10 | Anchors usable for #9 | Notes |
|------|:---:|:---:|------|
| `_diag_phase_b_add_one_instrumented` | yes | yes | 64-word BD baseline; Phase C reference test; ~44 LOCK_STALL events historic |
| `add_one_using_dma` | yes | yes | Short stalls (~22 LOCK_STALL); covers short-stall regime |
| `add_one_objFifo` | yes | maybe | Different BD shape; check stall profile |
| `add_one_objFifo_elf` | yes | maybe | ELF-driven variant; cross-check vs Python-driven |
| `objectfifo_repeat` (variant TBD) | yes | maybe | Multi-iteration; tests sustained streaming |
| `dynamic_object_fifo` | maybe | yes | Object FIFO with dynamic alloc; stall-heavy candidate |

Test selection rule: each test must (a) successfully complete on HW
under the current bridge harness with trace injection, (b) have a
clean shim DMA -> memtile S2MM data path so item-#10 anchors exist,
and (c) be on the trace-compatible list (not in
`scripts/trace-quarantine.txt` or `scripts/trace-incompat-tests.txt`).

**Open: do we need a new parameterized kernel for item #10?**

First-pass approach: mine the existing test corpus for BD-size
variation. The existing tests span 32 -- 1024 word BDs across a
handful of access patterns. If the regression on existing data has
small residual (within ~10% of fit), we have our rate.

If the existing-data regression is too noisy, build a parameterized
calibration kernel:

- One column, one shim S2MM channel, one memtile S2MM channel.
- BD size parameterized in the C++ launcher: {8, 16, 32, 64, 128,
  256, 512, 1024} words. (Lower end exercises cold-start dominance;
  upper end exercises steady-state throughput.)
- Access pattern variants: linear, stride-2, stride-4, single-bank,
  cross-bank.
- Cross-product gives ~40 data points per shim DMA channel kind.

This kernel becomes a permanent calibration asset, lives at
`mlir-aie/test/npu-xrt/_diag_shim_throughput_sweep/` if built.

## Item #10 -- shim throughput design

### Anchors

Per BD-size data point, capture:

- `shim_dispatch_t` -- INSTR_VECTOR or START_TASK from the shim tile
  at the point the DMA descriptor is committed.
- `shim_done_t` -- FINISHED_TASK from the shim tile after the BD
  completes.
- `memtile_s2mm_done_t` -- FINISHED_BD from the receiving memtile
  S2MM channel.

All three are `soc` fields. Two deltas per test:

```
shim_task_duration   = shim_done_t - shim_dispatch_t
memtile_tail         = memtile_s2mm_done_t - shim_done_t
end_to_end           = memtile_s2mm_done_t - shim_dispatch_t
```

### Sweep dimensions

| Dimension | Values |
|-----------|--------|
| BD payload size (words) | 8, 16, 32, 64, 128, 256, 512, 1024 |
| Access pattern | linear, stride-2, stride-4 |
| Bank selection | single-bank, cross-bank |
| Shim DMA channel | S2MM 0, S2MM 1 (cross-check arbitration) |

### Analysis

For each (access pattern, bank, channel) combination, plot
`shim_task_duration` vs BD size:

- Linear fit: `shim_task_duration = cold_start + words / rate`
- Intercept -> true cold-start (compare against current 1500 cyc
  EMU value)
- Slope -> 1/rate, gives shim egress rate in words/cyc

For `memtile_tail`, expect either:

- Constant tail (small, ~10-30 cyc) -> memtile receives in parallel,
  finishes a fixed tail after shim done (current EMU model's intent).
- Tail proportional to BD size -> memtile is the throughput bottleneck,
  not shim. Less likely on AIE2 but worth verifying.

The decision between these two model shapes is the primary outcome
of the shim throughput campaign.

### Output artifacts

- `data/hw-shim-throughput-2026-05.csv` -- per-test rows with
  (test_name, bd_words, access_pattern, bank, channel,
  shim_task_duration, memtile_tail).
- `data/hw-shim-throughput-2026-05-fit.json` -- per-combination
  regression results (cold_start, rate, R^2).
- A plot or two for the campaign close-out finding.

### Validation gate before declaring success

The model derived from this campaign is validated when:

1. EMU stage 1a residual on `_diag_phase_b_add_one_instrumented`
   drops within stage 3 noise floor (~10 cyc).
2. EMU stage 1b residual on same drops within ~10 cyc (currently
   wrong direction at -27 vs HW +1017).
3. No other test that previously passed the trace-comparison sweep
   regresses by more than 20 cyc total in stage 1+2.

## Item #9 -- perf-counter LOCK_STALL emission design

### Anchors

Per test in the corpus, capture:

- All `LOCK_STALL` events with their `soc` timestamps and tile IDs.
- Stall-entry and stall-exit edges (derive from kernel state, not
  trace -- the trace controller doesn't emit "stall start/end"
  directly; LOCK_STALL itself is the signal).
- Per tile, the PERF_CTRL0 / PERF_CTRL1 register values written by
  the test's CDO. Today this requires either: (a) parsing the
  loaded xclbin's CDO section, or (b) reading the registers via
  EMU's MMIO trace at xclbin-load time. Either path needs new
  scaffolding (see Tooling gaps).

### Sweep dimensions

| Dimension | Values |
|-----------|--------|
| Stall window length | short (~10s of cyc), medium (~100s), long (>1000) |
| PERF_CTRL config | as-found per test (no parametric variation in v1) |
| Tile location (memtile vs compute vs shim) | as-found |

We are not varying PERF_CTRL programmatically; we are correlating
across tests that happen to have different configs. This is the
v1 budget. v2 (which would need a custom kernel that varies
PERF_CTRL programmatically) is deferred.

### Analysis

For each test, compute:

```
lock_stall_count_per_tile     # observed in HW trace
lock_stall_intervals_per_tile # consecutive soc deltas
stall_window_duration         # total stall time per tile (from kernel state)
```

Two hypotheses to test:

1. **Perf-counter-driven**: `lock_stall_count = floor(stall_window_duration / PERF_THRESHOLD)`
   where PERF_THRESHOLD is read from PERF_CTRL0 of the tile, plus 1
   for the initial entry edge. Intervals should be approximately
   constant at PERF_THRESHOLD cyc.

2. **Edge-only with periodic re-emit**: counts independent of
   stall_window_duration. Intervals random or near-constant at a
   different value.

Phase C historic data (44 events on `_diag_phase_b_add_one_instrumented`,
~1024-cyc spacing, configured threshold also 1024) is consistent with
H1; this campaign verifies it across the corpus and gives us the
right PERF_THRESHOLD lookup.

### Output artifacts

- `data/hw-lock-stall-2026-05.csv` -- per-tile rows with
  (test_name, tile, lock_stall_count, mean_interval, stall_window).
- `data/hw-perf-ctrl-config-2026-05.json` -- per-test perf-ctrl
  configs (which event, which threshold) extracted from CDO.
- Decision: H1 vs H2, plus the formula to feed into the EMU
  model-builder.

### Validation gate before declaring success

The model derived from this campaign is validated when:

1. EMU LOCK_STALL event counts match HW within +/- 5% across the
   corpus, with current LOCK_STALL_TRACE_PERIOD removed (the
   constant becomes per-tile, sourced from PERF_CTRL).
2. The ts/soc gotcha stops biting: `ts` field across tests becomes
   directly comparable for in-stream event ordering (which is what
   it was designed for).
3. MEMORY_STALL / STREAM_STALL / PORT_STALL emission, if the same
   PERF_CTRL drives them, falls out for free. Document that in the
   close-out finding even if it's beyond the v1 scope.

## Tooling gaps

The campaign requires three pieces of scaffolding that don't exist
or aren't usable today. Address before kicking off the HW run.

### G1. Trace artifact preservation in bridge runs

Today's bridge sweep keeps PASS/FAIL/DRIFT verdicts in
`build/bridge-test-results/<date>/`, but the raw `trace.bin` and the
decoded events JSON are not preserved across runs. The campaign
needs both.

**Proposal**: add `--retain-traces` to `scripts/emu-bridge-test.sh`
that, when set, copies each test's `trace_buffer.bin` (HW side) and
`events.json` (post-parse-trace.py) to
`build/bridge-test-results/<date>/<test_name>/{hw,emu}/`. Default
off so the disk footprint of normal sweep runs doesn't grow.

**Estimate**: ~30 min of shell-script work.

### G2. CDO PERF_CTRL extraction utility

Item #9 analysis requires knowing each test's PERF_CTRL0 /
PERF_CTRL1 configuration per tile. Today we have no direct readout.

**Proposal**: small Python tool at `tools/extract-perf-ctrl.py`
that reads an xclbin, walks the CDO writes, and emits per-tile
`{tile_id: {perf_ctrl0: ..., perf_ctrl1: ..., event_thresholds: [...]}}`.
Doesn't need to be complete -- only needs to handle the writes
that hit PERF_CTRL register offsets. Use the existing aie-rt
register offset definitions and our regdb JSON to identify them.

**Estimate**: ~1 hr of Python.

### G3. soc-aware trace comparator (the persistent version)

Today's ad-hoc Python comparator (used to recover from this
afternoon's misdiagnosis) is not checked in. If we're going to
do this campaign systematically, we want a persistent comparator
that reads `events.json`, identifies the stage anchors, computes
sub-stage deltas using `soc`, and reports per-test rows.

**Proposal**: refactor or extend `src/bin/trace_compare.rs` so its
output mode includes stage-decomposition (anchors configurable per
test or by convention). The existing comparator already has event-
sequence alignment; we need it to also report stage cycle deltas.

**Estimate**: ~2-3 hr of Rust.

## Operational considerations

### TDR / wedge handling

Long sweep runs on HW historically wedge the NPU on certain
failure modes. The campaign should:

- Run tests one at a time (no parallel HW execution).
- After every 5 tests, smoke-test with `xrt-smi validate` (per
  `feedback_smoke_test_with_xrt_smi.md`).
- On wedge, follow the recovery escalation chain in `CLAUDE.md`
  Operational Notes -- modprobe -r/install first; reboot if that
  hangs on `synchronize_srcu`.

### Run cadence

The campaign is ~6 tests in the corpus, plus the parameterized
kernel if needed. Each test runs in ~30-60s. Total HW time:
roughly 30 min for the existing-corpus pass, plus another hour
if the parameterized sweep runs.

This is small enough to do in a single session. Don't try to
spread across days -- the day-to-day HW state varies enough
(thermal, autosuspend, prior wedge recovery) that a single
contiguous session keeps the measurements comparable.

### Sandbox vs poisoned shell

The HW runs need real XRT, not the EMU plugin. From a poisoned
shell, run each test as:

```
env -u XDNA_EMU -u XDNA_EMU_RUNTIME ./test.exe
```

per `CLAUDE.md` XRT plugin env contract.

## Execution checklist

Ordered phases. Each phase produces a checkpoint that should be
committed before moving on.

1. **G1 + G3 land.** Trace artifact preservation in bridge harness,
   plus the persistent soc-aware comparator. Without these, the
   campaign data is non-reproducible.
2. **G2 lands.** PERF_CTRL extraction tool. Required for item #9
   analysis; not blocking item #10.
3. **Existing-corpus baseline run.** Six tests, HW side only, with
   `--retain-traces`. Outputs per-test events.json + trace_buffer.bin.
4. **Item #10 first-pass regression.** Mine the existing-corpus
   data for BD size variation. Linear fit per (access pattern,
   bank, channel). If R^2 > 0.95 across the corpus and residual on
   `_diag_phase_b` validation gate looks tractable, skip the
   parameterized kernel.
5. **Item #9 first-pass analysis.** PERF_CTRL config from CDO,
   LOCK_STALL counts and intervals from events.json, hypothesis
   test (H1 perf-counter-driven vs H2 edge-only).
6. **Decision point.** Do we need the parameterized calibration
   kernel for item #10? Do we need a PERF_CTRL-varying kernel for
   item #9? Either, both, or neither.
7. **(Conditional) Build calibration kernels.** Only if step 6
   determined we need them.
8. **(Conditional) Second-pass HW run.** Sweep the new calibration
   kernels.
9. **Close-out findings.** Write two findings:
   `findings/<date>-shim-throughput-model.md` and
   `findings/<date>-lock-stall-cadence-model.md`. Each lands the
   model parameters with the data backing them.
10. **EMU modeling work** (downstream, not part of this campaign).
    Implement the throughput-bound streaming in DMA stepping;
    implement the CDO PERF_CTRL parser and per-tile perf-counter
    state. Re-run the validation gates from each item above.

Steps 1-9 are the campaign proper. Step 10 is the cycle-accuracy
mission's continuation, gated on the campaign's outputs.

## See also

- [`cycle-accuracy-mission.md`](cycle-accuracy-mission.md) items #9, #10
- [`../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md`](../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md)
- [`../superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md`](../superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md)
- Phase C archive: `docs/archive/findings/2026-05-10-phase-c-stage-attribution.md`
- Bridge harness: `scripts/emu-bridge-test.sh`
- Trace decoder: `tools/parse-trace.py`
- Existing comparator: `src/bin/trace_compare.rs`
