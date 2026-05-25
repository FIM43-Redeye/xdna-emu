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

- ~~Item #9 -- perf-counter-driven trace event emission (LOCK_STALL et
  al.)~~ **Closed via different fix 2026-05-25**. Hypothesis H1
  (perf-counter-driven cadence) was refuted by direct measurement: HW
  emits ~2233-2766 LOCK_STALL events per kernel run on the diag test,
  matching EMU's current behavior. The real gap is trace-unit
  wire-format compression (HW uses skip-tokens; EMU emits one packet
  per cycle). Item #9 retargets to trace-unit work in
  `src/device/trace_unit/`, no HW campaign needed. Detail:
  [`../superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md`](../superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md).

**Explicitly out of scope (deferred to model-build phase):**

- Implementing throughput-bound streaming in the DMA stepping
  state machine (item #10). Comes after the rate is known.
- Trace-unit wire-format compression (item #9 retargeted scope) --
  separate work, no HW campaign needed; lands in
  `src/device/trace_unit/`.

## Calibration corpus (for item #10)

Item #9 closed via different fix; the corpus below now serves item
#10 (shim throughput) only.

| Test | BD shape | Notes |
|------|----------|------|
| `_diag_phase_b_add_one_instrumented` | 64-word linear | Phase C reference test |
| `add_one_using_dma` | small linear | Cross-check vs IRON-style |
| `add_one_objFifo` | objectfifo | Different BD shape |
| `add_one_objFifo_elf` | objectfifo | ELF-driven variant |
| `objectfifo_repeat` (variant TBD) | multi-iteration | Sustained streaming |
| `dynamic_object_fifo` | dynamic alloc | Different driver path |

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

## Item #9 -- CLOSED (refuted 2026-05-25)

Previously slated for a perf-counter-driven LOCK_STALL emission
model. Direct HW measurement against preserved captures
(2026-05-20/21/24) refuted the hypothesis: HW emits ~2233-2766
LOCK_STALL events per kernel run on `_diag_phase_b_add_one_instrumented`,
matching EMU's per-cycle behavior. The original "44 events at
~1024-cyc spacing" claim from Phase C cannot be reproduced and was
likely a measurement error.

The real gap is trace-unit wire-format compression. EMU emits one
trace packet per cycle; HW packs runs of repeated events into
skip-tokens. Item #9 retargets to trace-unit work in
`src/device/trace_unit/` -- no HW measurement campaign needed.

Detail: [`../superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md`](../superpowers/findings/2026-05-25-lock-stall-cadence-h1-refuted.md).

The G2 tooling (PERF_CTRL extraction, MLIR + CDO paths) stays useful
as a ground-truth readout for perf-counter trace events themselves,
just not for LOCK_STALL cadence modeling.

## Tooling gaps

The campaign requires three pieces of scaffolding that don't exist
or aren't usable today. Address before kicking off the HW run.

### G1. Pre-decoded events.json next to trace_raw.bin

Today's bridge sweep already preserves `trace_raw.bin` and
`trace_config.json` per test, in
`build/bridge-test-results/<date>/<test>.<compiler>.{hw,emu}/`.
What's missing is the **decoded** events JSON -- the existing
`_bin_to_events_json` helper writes to `tmp/` and deletes after
trace-compare runs.

**Proposal**: after each side's `trace_raw.bin` is trimmed, decode
once via `_bin_to_events_json` and write `events.json` next to the
trace_raw.bin. Skipped when the side didn't run (e.g., `--no-hw`).
Best-effort -- if parse-trace.py fails, log and continue.

Self-contained per-test directories let the campaign analysis tools
consume `events.json` directly without re-decoding.

**Estimate**: ~15 min of shell-script work, two insertion sites
(`run_one_hardware`, `run_one_bridge`).

### G2. PERF_CTRL extraction utilities (retained, scope reduced)

Originally planned for item #9 cadence modeling; that item is closed
via a different fix (see top of doc). The tools still ship since
they're useful for any future work that needs ground-truth PERF_CTRL
readout (e.g., perf counter trace events on tests that DO use them).

**Finding (2026-05-25)**: IRON-style tests configure perf counters
at runtime via the NPU instruction stream, NOT via CDO at xclbin
load. The diag corpus reference test
`_diag_phase_b_add_one_instrumented` has zero PERF_CTRL writes in
its xclbin CDO; the configuration lives in the post-injection
MLIR's `aie.trace.config @perf_*` blocks (which `aiecc` then
lowers to instruction-stream control packets).

Two tools, complementary:

1. `tools/extract-perf-ctrl.py` -- **primary path**. Parses the
   post-injection MLIR (e.g.,
   `build/test/npu-xrt/<test>/traced/aie_traced.mlir`) for
   `aie.trace.reg register = "Performance_..." value = N` writes
   inside `aie.trace.config` blocks. On the diag test yields:
   `(0,2)/core: {Performance_Counter2_Event_Value: 1024, ...}`,
   matching the configured threshold.

2. `src/bin/extract_perf_ctrl.rs` -- **fallback** for tests that
   DO configure perf via CDO at xclbin load. Reads via our
   existing parser, filters on the AM025 PERF_CTRL register offset
   ranges (core 0x031500-0x03158C, memory 0x011000-0x011084,
   memory_tile 0x091000-0x09108C, shim 0x031000-0x031084), emits
   per-tile JSON. Returns empty for IRON tests by design.

**Actual time**: ~2 hr (1.5 hr Rust + 30 min Python pivot after
finding the CDO doesn't carry the config for IRON tests).

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
   plus the persistent soc-aware comparator. Done 2026-05-25
   (commits `eaff65e`, `85bab82`).
2. **G2 lands.** PERF_CTRL extraction tool. Done 2026-05-25
   (commit `b25f6e2`); scope reduced after item #9 closed.
3. **Item #9 closed via different fix.** Done 2026-05-25
   (see top of doc + finding link).
4. **Existing-corpus baseline run.** Six tests, HW side only, with
   `--retain-traces`. Outputs per-test events.json + trace_buffer.bin.
   **Done 2026-05-25** (chess on 5 tests, plus a top-up peano run on
   `objectfifo_repeat/simple_repeat` which is peano-only by REQUIRES).
   - Usable HW captures (events.json + trace_raw.bin): 4 tests --
     `_diag_phase_b_add_one_instrumented`, `add_one_using_dma`,
     `add_one_objFifo`, `add_one_objFifo_elf`.
   - Empty trace_raw.bin (all zeros) on `dynamic_object_fifo/ping_pong`
     and `objectfifo_repeat/simple_repeat`. Functional PASS, trace
     prep reports OK, but BO never receives data on HW.  Both tests
     compile against `aie2.mlir` (not `aie_arch.mlir`) and have
     `placement.origin_col=0`; suspected trace-BO address patch
     mismatch in the C++ launcher.  Tracked as a Phase B gap, separate
     from this campaign.
   - G1 helper fix landed in the same session: the predecode helper
     hardcoded `aie_arch.mlir.prj/input_with_addresses.mlir`, so the
     four corpus tests with that layout got events.json automatically
     and the two with `aie2.mlir.prj` did not. Helper now discovers
     `input_with_addresses.mlir` under `<test>/<compiler>/*.prj/`.
5. **Item #10 first-pass regression.** Mine the existing-corpus
   data for BD size variation. Linear fit per (access pattern,
   bank, channel). If R^2 > 0.95 across the corpus and residual on
   `_diag_phase_b` validation gate looks tractable, skip the
   parameterized kernel.  **Open**: 4 usable tests may be too narrow
   a BD-size span; revisit step 6 early if so.
6. **Decision point.** Do we need the parameterized calibration
   kernel for item #10?
7. **(Conditional) Build calibration kernels.** Only if step 6
   says yes.
8. **(Conditional) Second-pass HW run.** Sweep the new calibration
   kernels.
9. **Close-out finding.** Write `findings/<date>-shim-throughput-model.md`
   landing model parameters with the data backing them.
10. **EMU modeling work** (downstream, not part of this campaign).
    Implement the throughput-bound streaming in DMA stepping.
    Re-run the validation gate from item #10.

Steps 1-3 are done. Steps 4-9 are the campaign proper. Step 10 is
the cycle-accuracy mission's continuation, gated on the campaign's
outputs.

## See also

- [`cycle-accuracy-mission.md`](cycle-accuracy-mission.md) items #9, #10
- [`../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md`](../superpowers/findings/2026-05-25-trace-ts-vs-soc-measurement-gotcha.md)
- [`../superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md`](../superpowers/findings/2026-05-25-shim-stage1a-1b-structural-limit.md)
- Phase C archive: `docs/archive/findings/2026-05-10-phase-c-stage-attribution.md`
- Bridge harness: `scripts/emu-bridge-test.sh`
- Trace decoder: `tools/parse-trace.py`
- Existing comparator: `src/bin/trace_compare.rs`
