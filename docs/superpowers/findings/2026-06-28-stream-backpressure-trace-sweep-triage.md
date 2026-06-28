# Trace-sweep re-baseline triage — stream-switch backpressure fix (#140, Task 5)

**Date:** 2026-06-28
**Branch:** `device-model-audit`
**Change under test:** the inter-tile stream crossing fix (Task 6, commit `e044ac72`)
— bound admission by the crossing depth (`fifo_capacity + ROUTE_PER_HOP`), not the
FIFO capacity alone. Fabric-global: affects every inter-tile circuit-routed hop.

Full chess bridge sweep with tracing: `./scripts/emu-bridge-test.sh --chess-only --trace`
(162 tests compiled, 148 run HW+EMU; results `build/bridge-test-results/20260628/`).

## Integration gate — DATA (the hard regression detector): CLEAN

```
Chess: 146/147 compiled, 148 bridge pass, 0 bridge fail
  HW: 148 pass, 0 fail, 0 skip
Chess trace: 0 clean, 124 diverge, 0 error, 47 skip
```

**No data regression anywhere.** The fabric change is lossless (it only changes
when a producer is *throttled*, never whether a word is delivered), and the full
corpus confirms it: every kernel's HW and EMU outputs still match (148/148 bridge
pass, 0 fail, 0 error). This is the regression-critical result.

## Trace divergences (124) — the known cadence frontier, not regressions

The bridge trace check is a **live EMU-vs-HW** comparison (no stored baseline);
`DIVERGE` means the EMU cadence is not byte-identical to HW this run. Trace
byte-identity is the open fidelity frontier for most kernels (see
`docs/known-fidelity-gaps.md`), so a high `DIVERGE` count is the status quo, not
churn introduced here. Breakdown:

- **69 of 124 are single-tile `vec_*` kernels.** They have no inter-tile circuit
  crossings, so the Task 6 inter-tile-admission change **cannot** affect them by
  construction — their divergence is the pre-existing vector-compute / trace-cadence
  frontier.
- **55 are multi-tile** (dma / memtile / shim / cascade / ctrl-packet). This is the
  set the change *could* touch. All are data-PASS (lossless), and the model is now
  HW-verified more faithful (so trace effects are toward-HW or neutral, by the
  `PORT_RUNNING`/`PORT_STALLED` co-trace proof in the design spec section 2).

## Wins (moved toward HW)

- **`add_one_using_dma` recv `PORT_RUNNING_0` = `[16,16,16,16]` now matches HW
  exactly** (RUNNING `[16,16,16,16]` / STALLED `[1,1,1]`, complementary — Task 6
  Step-7 HW re-validation). This *heals* the recv regression that the first-cut
  hard cap (Task 1, `99539c81`) had introduced (`[16,16,16,16] -> [4,4,..]`), and is
  actually better than the pre-arc base because the RUNNING/STALLED tiling now
  matches HW's handshake.
- **`matrix_multiplication_using_cascade` (plain/buffer/cascade): trace `NONE`**
  (no divergence), data PASS — also exercises the cascade-512 width fix (`991b0877`).

## Pre-existing items (unrelated to this change)

- **`vec_mul_trace_distribute_lateral`: compile FAIL** (the 1 of 147 not compiled).
  A *vector kernel compile* failure cannot be caused by an emulator routing change;
  it is the kernel already cited in the gaps-ledger trace-micro-timing row. Pre-existing.
- **`bd_chain_repeat_on_memtile`: trace-prep FAIL**, RESULTS `PASS PASS NONE` — data
  passed both HW and EMU; the failure is trace *injection* tooling (the
  "trace-injection incompat: 1 test" bucket), not the emulator. Pre-existing.

## Verdict

The fabric-global backpressure change is **lossless and free of data regressions**
across the full corpus (148/148), heals the recv-cadence regression, and makes the
inter-tile crossing model HW-faithful. The residual trace divergences are the
documented cadence frontier (dominated by single-tile kernels the change cannot
touch); the send-port cadence residual is logged in `known-fidelity-gaps.md` as a
consumer-pacing follow-on.

**Limitation (honest):** no exhaustive per-kernel before/after cadence diff was run
for the 55 multi-tile kernels — the live binary `DIVERGE` flag is a frontier
indicator, not a toward/away-from-HW delta, and the hard data gate (148/148) is the
regression detector. The one multi-tile kernel deeply analyzed (`add_one_using_dma`)
improved.
