# Stochastic-aware trace comparison (task #140)

**Date:** 2026-06-14
**Status:** DESIGN for review, no code yet.
**Motivation:** [DDR stochasticity finding](../findings/2026-06-14-ddr-delivery-stochasticity.md)
proved DDR-boundary trace events (STREAM_STARVATION, LOCK_STALL) are genuinely
stochastic (CV 6-25%, memoryless), while compute-structure events
(PORT_RUNNING) are deterministic. Comparing a deterministic EMU run against a
*single* HW capture manufactures phantom divergence on the stochastic events --
which is much of why the corpus reads 125/140 DIVERGE. The fix: compare EMU
against the HW *distribution* (multiple runs), with a per-event tolerance band
derived from the HW variance itself.

## Core principle: derive the band from HW variance

No hardcoded "these events are stochastic" table. Feed the comparator N HW
captures of the same binary; per event, compute the count distribution across
runs. The band width *emerges* from the data:

- An event with HW std ~= 0 (e.g. PORT_RUNNING) -> **deterministic** -> exact
  band -> EMU must match exactly (a mismatch is a real bug).
- An event with HW std > 0 (e.g. STREAM_STARVATION) -> **stochastic** -> a
  tolerance band -> EMU in-band is CLEAN.

This is the toolchain-derivation principle applied to HW observation: the
hardware's own run-to-run variance defines the tolerance. Add more HW runs ->
sharper bands; the classification is never hand-maintained.

## Scope (MVP)

**In:** per-event *count* band comparison. Count-divergence is where the 125
DIVERGE live, and counts are what the finding quantified (mean/std per event).

**Out (follow-ups):** per-event *timing*-band (dt tolerance widened for
stochastic edge events); bridge-harness integration (auto-capturing N HW runs
per kernel); multi-kernel band library. MVP delivers the comparison primitive;
wiring it into the bridge verdict is a separate step.

## Design

New module `src/trace/stochastic.rs`, additive -- does not touch
`compare_batch_with_opts` or the existing single-HW path.

```
pub enum Regime { Deterministic, Stochastic }

pub struct EventBand {
    key: TileKey, slot: u8, name: String,
    hw_counts: Vec<usize>,            // one per HW run
    mean: f64, std: f64, min: usize, max: usize,
    emu_count: usize,
    band_lo: f64, band_hi: f64,
    regime: Regime, in_band: bool,
}

pub struct StochasticReport { bands: Vec<EventBand>, n_runs: usize, sigma_k: f64 }

pub fn compare_stochastic(
    hw_paths: &[PathBuf], emu_path: &Path,
    config: &EventsConfig, sigma_k: f64, remap_columns: bool,
) -> Result<StochasticReport, String>
```

Algorithm:
1. Load each HW run (`load_events_json`, reusing the just-fixed `names_for_pkt`
   labeling) -> per `(TileKey, slot)` count.
2. Load EMU -> per `(TileKey, slot)` count.
3. Per `(TileKey, slot)` present on HW: gather the N counts, compute
   mean/std/min/max; `band = [mean - k*std, mean + k*std]`; `regime =
   if std == 0 { Deterministic } else { Stochastic }`; `in_band = band_lo <=
   emu_count <= band_hi` (deterministic collapses to exact equality).
4. EMU-only events (not in any HW run) are reported out-of-band.
5. Verdict: `STOCHASTIC_VERDICT: CLEAN` iff every event in-band, else `DIVERGE
   (<n> of <m> events out of band)`.

CLI (additive to `trace-compare`):
```
trace-compare --hw-dir <dir-of-run_*.json> --emu <emu.json> \
              [--band-sigma <k=2.0>] [--remap-columns]
```
`--hw-dir` loads all `run_*.json` (or a `--hw-runs a,b,c` comma list).

## The one genuine design choice: band definition

Two defensible band definitions; I recommend (A):

- **(A) Parametric: mean +/- k*std** (k default 2.0, configurable). Tunable,
  standard, and degrades cleanly to exact for deterministic events (std 0). With
  k=2 and the measured S2MM_0 (mean 28.1, std 2.7) the band is [22.7, 33.5] --
  EMU centered ~28 lands well inside.
- **(B) Non-parametric: observed [min, max]**. No normality assumption; "EMU is
  fine if it falls in the range HW actually exhibited." But sensitive to a single
  outlier run (S2MM_0 min was 20, a lone low) and to under-sampling the tails.

Recommendation: **(A) mean +/- k*std**, and *also report* the observed [min,max]
and EMU's sigma-distance when out-of-band, so a reviewer sees both. k is a flag
so we can tighten/loosen without recompiling.

## Validation (TDD, all EMU-side / offline -- no HW needed)

The 20 `add_one_using_dma` HW captures in
`build/experiments/ddr-stochasticity/` are the test fixture:
- Unit: synthetic distributions -> band math (deterministic std0 -> exact;
  stochastic -> mean+/-k*std; EMU in/out classification; EMU-only event).
- Integration: load the 20 real captures + the EMU baseline; assert PORT_RUNNING
  is classified Deterministic and STREAM_STARVATION Stochastic; assert the
  EMU-off baseline (S2MM 20/29) lands the count *below* band (proving the model
  gap is real, not a sampling artifact), and that a mean-calibrated EMU would be
  in-band.
- Full `cargo test --lib` green; the new path is additive so the existing
  single-HW comparison and corpus verdict are untouched.

## Decision (2026-06-14)

Band = **(A) mean +/- k*std, k=2 default (flag-configurable)**, also reporting
observed [min,max] and EMU sigma-distance. HW-derived (the silicon's run-to-run
variance is ground truth).

## North star: a stochastic emulator DDR model (Maya, 2026-06-14)

The longer arc for the emulator's own DMA simulation is **real DDR specs
(JEDEC: burst length, tRC, bandwidth, refresh interval) + a calibrated random
perturbation** -- replacing today's fixed `BurstParams` knobs with a
physically-grounded, intentionally-stochastic model. That makes the emulator
itself emit a *distribution* (seeded RNG), not a point.

Two design guardrails that keep this clean:

1. **The comparison band stays HW-derived, not model-derived.** Deriving the
   tolerance from the emulator's own DDR specs would test the model against
   itself (circular). The HW's observed variance is the ground-truth band; the
   emulator's perturbation model *earns its keep by reproducing that variance*.
2. **End state = distribution-vs-distribution.** When both sides are stochastic,
   the comparison becomes a two-sample test (do EMU-over-seeds and HW-over-runs
   come from the same distribution?). The mean+/-k*std machinery built here is
   the natural primitive for that -- this MVP is forward-compatible, not throwaway.

So the evolution is: (this task) EMU-point vs HW-band -> (DDR-model task)
EMU-distribution vs HW-band -> (maturity) EMU-distribution vs HW-distribution.
`BurstParams` -> spec-driven DDR model is the #129/#131 + DDR-model follow-up,
not this task.
