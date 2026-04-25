# Phase E Validation Results (2026-04-23)

End-to-end validation of the trace-diff-based cycle budget pipeline
(`--with-cycle-diff`) on the Phase B 7-test batch.

## Setup

- Branch: `dev`
- Phase E commits (`git log --oneline master..dev`, most recent first):
  - `330f89a` bridge-test: re-classify cycle diffs after Phase 3+4 (race fix)
  - `2cb95e3` scripts: show-cycle-drift.sh
  - `d6f3004` bridge-test: CYCLES column + drift summary
  - `2d849db` bridge-test: per-test cycle-drift ratio overrides
  - `1e41dcd` bridge-test: skip trace injection for incompat tests
  - `e0e1352` bridge-test: distinguish COMPILE-FAIL(traced)
  - `cb0dba7` bridge-test: dual-bound EMU timing from HW cycle data
  - `a785bd2` bridge-test: classify cycle diff
  - `e2b741e` docs: decoder migration + Task 7 framing
  - `1adcf0b` trace-compare: consume events JSON
  - `8aa12fa` trace: parse-trace.py single-source decoder
- Invocation:
  ```
  ./scripts/emu-bridge-test.sh --with-cycle-diff --no-timeout \
    -v '^(vector_scalar_using_dma|add_one_using_dma|add_one_objFifo|cascade_flows|add_blockwrite|column_specific|ctrl_packet_reconfig)$'
  ```
- Classifier defaults: EMU/HW ratio bounds `[0.5, 2.0]`; trace-compare
  `DIVERGE_THRESHOLD = 10` cycles.
- `EMU_SECONDS_PER_CYCLE = 1e-3` (un-calibrated starting constant).
- Both compilers (chess + peano) attempted per test where supported.
- Classifications below reflect the post-fix state (see §Race surfaced):
  the initial batch-run produced several stale `HW_TRACE_BUG` /
  `EMU_TRACE_BUG` labels due to an ordering race; the corrected
  classifier was applied to the same artifacts to get the terminal
  state shown here.

## Results

| Test | Compiler | HW cycles | EMU cycles | Compare diverge | Classification | Notes |
|------|----------|-----------|------------|-----------------|----------------|-------|
| `vector_scalar_using_dma` | chess | 41181 | 41176 | 0 | **MATCH(1.00)** | Reference success — the one test whose default event set produces matched, clean, non-degenerate traces on both sides. |
| `vector_scalar_using_dma` | peano | — | — | — | (SKIP compiler) | Chess-only test. |
| `add_blockwrite` | chess | 0 | 0 | — | EMPTY | Scalar kernel, no INSTR_VECTOR fired on either side. Phase B Limitation 1. |
| `add_blockwrite` | peano | 0 | 0 | — | EMPTY | Same. |
| `add_one_objFifo` | chess | 0 | 0 | — | EMPTY | Same. |
| `add_one_objFifo` | peano | 0 | 0 | — | EMPTY | Same. |
| `add_one_using_dma` | chess | 0 | 0 | — | EMPTY | Same. |
| `add_one_using_dma` | peano | 0 | 0 | — | EMPTY | Same. |
| `cascade_flows` | chess | 9 | 0 | — | **EMU_TRACE_BUG** | HW captures 9-cycle span (degenerate per Phase B Limitation 2: only 2 timestamped events total, across 2 tiles); EMU captures *nothing*. EMU-side trace-unit behavior on cascade configurations is a real gap. |
| `cascade_flows` | peano | — | — | — | (SKIP compiler) | Chess-only test. |
| `column_specific` | chess | — | — | — | NO_DATA | No `aie.mlir` (uses `aie2.py` generator); trace inject silently skipped. Phase B Limitation 3. **Update 2026-04-25:** classifier now reports `NO_CORE` correctly -- traced MLIR is generated, but the test is DMA-only (no `aie.core`), so trace events cannot fire by design. Not a bug; just expected silence. |
| `column_specific` | peano | — | — | — | NO_DATA | Same. |
| `ctrl_packet_reconfig` | chess | — | — | — | NO_DATA | `insts.bin` not present (test uses `aie_run_seq.bin` + `ctrlpkt.bin` convention); cycle pipeline needs a code-path for this. **Update 2026-04-25:** discovery is now correct (`_discover_instr_binary` returns `aie_run_seq.bin`, `_discover_ctrlpkt_binary` returns `ctrlpkt.bin`), but `bridge-trace-runner` doesn't apply the ctrlpkt before submitting the run sequence. test.exe directly under EMU completes in 46828 cycles; bridge-runner fails on both sides. Tracked as a runner bug -- see `docs/superpowers/findings/2026-04-25-ctrl-packet-reconfig-bridge-runner.md`. |
| `ctrl_packet_reconfig` | peano | — | — | — | NO_DATA | Same. |

## Aggregate

Per-compiler `CYCLE DRIFT` summary:

- **chess**: 1 MATCH, 0 DRIFT, 3 EMPTY, 1 EMU_TRACE_BUG, 0 HW_TRACE_BUG, 0 COMPARE-ERR, 2 skipped
- **peano**: 0 MATCH, 0 DRIFT, 3 EMPTY, 0 EMU_TRACE_BUG, 0 HW_TRACE_BUG, 0 COMPARE-ERR, 4 skipped

## Surfacing sanity check

`./scripts/show-cycle-drift.sh` output (post-fix severity order):

```
|log|     TEST                                              RESULT
--------  ------------------------------------------------  ------
10.0000   cascade_flows.chess                               EMU_TRACE_BUG
 1.0000   (12 EMPTY / NO_DATA entries)
 0.0000   vector_scalar_using_dma.chess                     MATCH(1.00)
```

The bug floats to the top; the baseline match sits at the bottom. As
DRIFT cases start appearing with real non-unit ratios they will sort
between 10 and 0.

## Observations

### Race surfaced (fixed)

The batch-run revealed an ordering race in the Task 7 classifier.
`_classify_cycle_diff` was invoked inside `run_one_bridge`, i.e. on the
EMU job's timeline. Phase 3+4 runs HW (serial, `-j1`) and EMU
(parallel, `-j$JOBS`) concurrently; for any given test, the EMU job can
finish before its HW counterpart has written `cycles.HW.txt`. Result:
the early classification sees `emu_cycles > 0 && hw_cycles == 0` and
records `HW_TRACE_BUG` even when HW subsequently completes cleanly.
`cascade_flows` was the most visible example in this batch —
early-classified `HW_TRACE_BUG`, terminal state `EMU_TRACE_BUG`.

The fix (commit `330f89a`) adds a re-classification pass in `main()`
between Phase 5 and Phase 6: after all HW+EMU jobs have joined, iterate
all reported rows and re-invoke `_classify_cycle_diff` with the
complete on-disk state. Cycle.result writes are idempotent, so this
cleanly overwrites early-race values. Verified against this batch's
artifacts without re-running the tests themselves.

### Phase B Limitation 1 is real on both sides

Six of seven tests (excluding the single baseline) produce EMPTY on
*at least* the chess side, and six of seven fail on peano too. The
cause is known: the default event set
(`INSTR_VECTOR` / `INSTR_EVENT_0` / `INSTR_EVENT_1`) doesn't fire on
scalar kernels. Both HW and EMU emit nothing; the classifier
correctly reports EMPTY rather than inventing a bug.

To turn these into usable data points, the trace-inject default event
set needs broader coverage (e.g., `INSTR_LOAD` / `INSTR_STORE` or a
DMA port event). Out of Phase E scope; tracked against
`tools/mlir-trace-inject.py`.

### `cascade_flows`: real EMU-side gap

`cascade_flows.chess` is the one substantive finding. HW captures 2
timestamped events (9-cycle span — degenerate per Phase B Limitation 2,
but non-zero). EMU captures nothing. That's an emulator trace-unit
behavior difference specific to cascade topology, and it's exactly the
kind of asymmetry Phase E was designed to surface. Follow-up
investigation — not a Phase E deliverable, but clearly flagged now.

### `ctrl_packet_reconfig`: cycle pipeline convention gap

This test's build produces `aie_run_seq.bin` + `ctrlpkt.bin` rather
than the standard `insts.bin`. The cycle pipeline's `insts.bin` lookup
misses, and the runner is skipped, producing NO_DATA. Not a classifier
bug; an unsupported test shape. Fixable in
`_run_trace_cycles_pipeline` by adding an `aie_run_seq.bin` fallback,
if control-packet tests become interesting cycle targets.

## Tuning opportunities surfaced

- **`EMU_SECONDS_PER_CYCLE` calibration: closed (2026-04-25).** The
  Phase E doc was looking at the stale `1e-3` value; the current value
  in `scripts/emu-bridge-test.sh` is `2e-9` (500 MHz sim rate), which
  is reasonable. Per A.5 findings
  (`docs/superpowers/findings/2026-04-25-cycle-accumulator-status.md`),
  this constant is a wallclock-timeout parameter, not a cycle-diff
  parameter, and the 600s floor dominates for any test under ~300 G
  cycles, so further calibration is not warranted.
- **Default ratio bounds `[0.5, 2.0]` are not exercised by this batch.**
  The only matched test gave ratio 1.00 (EMU=41176, HW=41181 — 0.012%
  drift). We have no data on how tight these bounds should be for other
  kernels until Limitation 1 (empty traces) is addressed.
- **No per-test overrides are warranted yet.**
  `scripts/cycle-drift-overrides.txt` remains empty; add entries as
  calibrated drifts emerge.

## Verdict

**MATCH — Phase E pipeline is ready for normal bridge-test use.**

- The end-to-end pipeline works: compile → HW/EMU → cycle extraction →
  comparison → classification → report column & summary.
- The one baseline-clean test (`vector_scalar_using_dma`) correctly
  classifies as `MATCH(1.00)` with zero divergence and
  sub-0.02 % cycle drift.
- The one substantive EMU-side issue (`cascade_flows`) is correctly
  surfaced as `EMU_TRACE_BUG` and ordered at the top of the drift
  report.
- All other batch results (`EMPTY` / `NO_DATA`) are correct
  classifications of known pre-Phase-E limitations (scalar kernels,
  `.py`-generated MLIR, non-standard `insts.bin` conventions).
- One race bug discovered and fixed during this validation
  (`330f89a`); no other defects surfaced.

Next steps beyond Phase E:
1. Broaden the trace-inject default event set to cover scalar kernels.
2. Investigate `cascade_flows` EMU trace-unit behavior.
3. Add `aie_run_seq.bin` fallback in `_run_trace_cycles_pipeline`.
4. Calibrate `EMU_SECONDS_PER_CYCLE` empirically.
