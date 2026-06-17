# DMA nondeterminism characterization harness (task #140)

**Date:** 2026-06-16
**Status:** DESIGN for review, no code yet.
**Base kernel:** `add_one_using_dma` (canonical DMA path: shim MM2S -> memtile ->
compute core -> memtile -> shim S2MM).

## Why this exists

The contention control pass (2026-06-16) ran two contention-free HW passes of the
full bridge corpus and diffed them against each other:

```
solo-vs-solo  : 171 DIVERGE / 212   (two HW runs of the same kernel, vs each other)
solo-vs-packed: 169 DIVERGE / 212
diverge ONLY under packing: 0
```

Two conclusions fall straight out:

1. **Contention is timing-neutral.** Zero kernels diverge *only* under packing, so
   the column-packer is safe everywhere and the run-to-run jitter is intrinsic, not
   a packing artifact.
2. **The single-run oracle is dead for ~81% of the corpus.** 171/212 kernels
   diverge *against themselves* across two contention-free runs. Comparing a
   deterministic EMU against a *single* HW capture manufactures phantom divergence
   wherever the silicon itself wandered between runs.

The 171 is therefore **not an EMU-fidelity number** -- it is the comparator
flunking kernels whose genuinely-stochastic events differed HW-vs-HW. We do not yet
know how much of the corpus EMU already gets right, because that signal is buried
under un-masked nondeterminism. This harness measures, provably, *how much* of a
kernel's trace timing is genuinely nondeterministic, *where* it lives, and *how
wide* the bands are -- the data the masking comparator needs, and the decomposition
of the 171 into "maskable noise" vs "real residual bug."

## The load-bearing distinction: event classes

Prior-art findings (below) plus the HW-vs-HW de-confound pin down *two event classes
with opposite expected behavior*. The harness must not blur them:

- **Level / span events** (`PORT_RUNNING`, `PORT_STALLED`, ...). Deterministic by a
  hardware law: `sum(PORT_RUNNING spans) == words` per port, 1 cycle/word
  (AXI4-Stream). Measured span-based, HW `PORT_RUNNING` is std 0 across 15-20 runs.
  A HW-vs-HW divergence here would be a **real bug** (or a metric artifact), never
  noise -- so these must **never** be masked.
- **Milestone / edge events** (DMA `START`/`FINISHED`, lock acquire/release
  timestamps). Their absolute timing depends on DDR-delivery latency, which the SoC
  genuinely varies run-to-run (add_one shim `S2MM_FINISHED` swung ~1311 cycles
  HW-vs-HW). A divergence here is **stochastic noise** -- the thing the masking
  comparator must tolerate.

**Why the 171 has to be milestone-edge jitter, mechanically:** the control pass was
HW-vs-HW (two silicon runs, decoded identically). Any *deterministic* processing
artifact -- the trace-unit encoder, the decoder, frame-record inflation -- is applied
to both sides equally and **cancels**. And level spans are deterministic-by-law. So
whatever drives the 171 must be genuine silicon run-to-run variance on events that
truly jitter on the metal: the milestone edges. (This is consistent with the
encoder-inflation bug being real but *orthogonal* -- it shows up EMU-vs-HW, not
HW-vs-HW.)

**Crucially, the classification is *emergent*, not hardcoded.** The harness measures
per-event variance across the repeats; the two classes *fall out of the data* (level
spans collapse to std~=0, milestone edges smear wide). Prior art gives us a
*prediction* the harness confirms or refutes -- we never bake in a "these events are
stochastic" table. This is the same HW-derivation principle the count-band
comparator already uses (derive the band from observed variance).

## The one hard constraint: span-based metric, never frame-records

On 2026-06-16 a stochastic DDR model was **removed** because the "jitter" it
reproduced was a **metric artifact**: the cadence tool counted trace *frame-records*
(a held level is re-checkpointed in a `cycles==0` frame every time a *concurrent*
signal toggles -- HW does this too) instead of *spans* (continuous runs of the
signal asserted). Frame counts are inflated and cross-contaminated by other signals'
edge timing, manufacturing run-to-run variance that does not exist on silicon.

**This harness measures spans / true edges, never frame-records.** It reuses the
corrected `be_spans.py` / `tools/port-span-cadence.py` lineage (pair B/E by
pid=tile / tid=slot, idle-gap > 2 grouping), *not* the superseded frame-counting
`port-cadence-baseline.py`. Getting this wrong re-walks the exact trap that cost a
model today.

## Relationship to prior art (do NOT rebuild)

| Artifact | What it is | Relationship |
|----------|-----------|--------------|
| `src/trace/stochastic.rs` + `trace-compare --hw-dir/--hw-runs` | Per-event **count**-band comparator (mean +/- k*std, classification emergent). Built, wired. | This harness is the **timing-band sibling** the `2026-06-14` spec explicitly deferred ("per-event *timing*-band ... follow-up"). Same band machinery, applied to span/edge *timing* instead of *counts*. |
| `2026-06-14-stochastic-aware-trace-comparison.md` | Count-band design + the "derive band from HW variance" principle. | We inherit the principle and the `EventBand`/`mean+/-k*std` vocabulary. |
| `2026-06-15-ddr-stochastic-delivery-jitter.md` | **SUPERSEDED / model removed.** Reproduced the frame-record artifact. | Cautionary: do not re-derive a stochastic *emulator* model from this characterization. The harness characterizes HW; it does not (here) change EMU. |
| `2026-06-16-port-cadence-metric-was-frame-records.md` | The metric-artifact finding + `sum(spans)==words` law + encoder-inflation localization. | Source of the span-metric constraint and the level/milestone class split. |
| `build/experiments/ddr-stochasticity/run_01..20.json` | 20 single-config HW captures of `add_one_using_dma`. | Analysis-tool **fixtures** (real multi-run data to validate the variance math offline). Not the full event coverage. |
| `tools/trace-sweep.py` + `--sweep` | Full event coverage: discovers tiles from MLIR, 8-slot batches per tile, patches `insts.bin` per batch (no recompile), decodes to `events.json`. | The coverage engine. We add a **repeat loop** on top; we do not reimplement coverage. |

## Scope

**In (this spec):**
- HW-only characterization. EMU is the expensive thing under test, not part of
  measuring HW's own spread; it does not run here.
- Base kernel `add_one_using_dma`; full event coverage across all 3 tiles (shim
  0:0, memtile 0:1, compute 0:2) via the existing batch sweep.
- N=20 repeats per config (flag-configurable).
- Output: per-event span/edge variance, emergent deterministic/stochastic
  classification, and the **decomposition of divergence** into the two classes.

**Out (explicitly deferred):**
- Wiring timing bands into the comparator (`stochastic.rs` masking path / a
  `TRACE_VERDICT` change). Characterize first; *then* decide what to mask.
- Generalizing to other kernels / per-event-type transfer rules.
- Any EMU model change or EMU-vs-HW fidelity scoring.
- The encoder-inflation fix and the relay-fill residual (separate, deterministic
  work tracked in the `2026-06-16` finding).

## Architecture

Three units, each independently testable.

### 1. Repeat-sweep driver

Extends the existing sweep path with a repeat count. For each of the ~16 lockstep
batches that cover every event on every tile, run the **HW** config **N=20** times,
dumping each repeat's decoded `events.json`.

- **HW side only.** The EMU half of `--sweep` is skipped entirely.
- **Layout:** `build/experiments/gap140/nondeterminism/<kernel>/batch_<b>/run_<r>.json`.
- **Parallelism:** repeats may run `-j5` (contention proven neutral today by the
  control pass) for ~5x wall-clock; `--hw-jobs` already exists on the bridge.
  Default parameterizable.
- **Reuse:** coverage, `insts.bin` patching, and decode are all existing
  `trace-sweep.py` / `parse-trace.py` machinery. The new surface is a `--repeat N`
  loop and the run-indexed output layout.

### 2. Span / edge extraction + re-anchoring

Per run, per tile, turn `events.json` into comparable per-event measurements.

- **Re-anchor per tile.** Subtract the tile's grounding event so two runs' differing
  trace-clock starts do not masquerade as variance. **Within-tile only** -- never
  cross-tile (cross-tile carries per-tile skew; this is the same anchoring the
  bridge already does).
- **Level/span events:** group B/E into spans (idle-gap > 2), emit per-port
  `{span_count, span_durations, span_sum}`. `span_sum` is checked against the
  `==words` law.
- **Milestone/edge events:** emit per-event re-anchored timestamps and the
  within-tile intervals between consecutive same-tile edges, keyed by
  `(tile, event, occurrence)`.
- Reuses the corrected `be_spans.py` span logic; this module is its productized form.

### 3. Variance analysis + report

Per measurement key, aggregate the N samples and classify.

- **Aggregate:** `{n, mean, std, min, max, range}` per key.
- **Classify (emergent):** `deterministic` if `std <= eps` (a few cycles), else
  `stochastic`. Sort by std so the fat milestone-edge smears float to the top and
  the level-span skeleton sits at the bottom near std 0.
- **Confound check:** does a measurement depend on *which batch config* traced it?
  For events that recur across overlapping batches, compare cross-config -- if
  trace-config placement perturbs timing, flag it (it should not, but the metric
  finding earned us this paranoia).
- **Law check:** assert `span_sum == words` for every level port; a violation is a
  real bug surfaced by the harness, not noise.
- **Decomposition (the headline output):** report, for `add_one_using_dma`, the
  cycle budget split -- how many intervals / how many total cycles are
  deterministic skeleton vs stochastic, and *which* events are the stochastic ones
  (prediction: the DMA milestone edges). This is the "how much of the 171 is real"
  number in microcosm.
- **Artifact:** a markdown + JSON report with the per-key table, the
  classification summary, the law-check result, and ASCII histograms for the
  top-variance keys.

## Data flow

```
sweep batches (full event coverage, 3 tiles)
  -> N=20 HW runs each (-j5)  [unit 1]
  -> per-run events.json
  -> re-anchor per tile, extract spans + edge intervals  [unit 2]
  -> aggregate per key across 20 runs -> {mean,std,min,max}  [unit 3]
  -> emergent classification + law check + decomposition
  -> report (md + json + histograms)
```

## Success criteria

For `add_one_using_dma`, the harness lets us state with evidence:

1. The level-span skeleton (`PORT_RUNNING`/`PORT_STALLED`) collapses to std~=0, and
   every port satisfies `sum(spans) == words` -- confirming the deterministic
   backbone is genuinely deterministic (not a sampling accident).
2. The stochastic events are *specifically* the DMA milestone edges (shim/memtile
   `START`/`FINISHED`), with their measured band widths -- confirming Maya's "it's
   DMA stuff" hypothesis *and* localizing it to milestone-edge timing, not level
   events.
3. A cycle-budget decomposition: X% of the timeline deterministic, Y% stochastic,
   with the stochastic share attributable to named DMA events. This is the template
   for decomposing the whole 171.

If instead a *level* event shows variance, or a milestone edge proves deterministic,
the harness has refuted the prediction -- which is itself a finding, and exactly why
the classification is emergent rather than assumed.

## Validation (TDD)

- **Unit (offline, no HW):** synthetic event streams -> span grouping (idle-gap > 2),
  re-anchoring (constant offset cancels), interval extraction, variance math
  (std 0 -> deterministic; spread -> stochastic + band), `span_sum` law check.
- **Integration (offline, real data):** the 20 `ddr-stochasticity/run_*.json`
  captures are the fixture -- assert the analysis tool runs end-to-end on real
  multi-run data and produces a sane classification (level events near std 0,
  milestone edges with spread). No HW needed for the analysis path.
- **HW (the actual characterization run):** the N=20 full sweep on
  `add_one_using_dma`, producing the report. Gated, run once.
- `cargo test --lib` green; the harness is additive (new tool + analysis module,
  no change to the comparator or EMU).

## Open implementation questions (for the plan, not this spec)

- Exact interval-key scheme when a milestone event fires multiple times per run
  (occurrence indexing vs BD-id keying).
- `eps` threshold for the deterministic/stochastic boundary (start a few cycles;
  the histograms will show whether the gap between the two classes is clean).
- Where the new code lives: a `--repeat N` flag on `trace-sweep.py` (or a thin
  wrapper) + a new `tools/trace-variance.py` analyzer, vs folding analysis into an
  existing tool. Lean toward a standalone analyzer so the math is unit-testable
  without the sweep.
