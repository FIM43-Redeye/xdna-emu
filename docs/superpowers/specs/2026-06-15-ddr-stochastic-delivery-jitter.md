# Stochastic DDR delivery jitter (task #140 close-out)

**Date:** 2026-06-15
**Status:** SUPERSEDED / model REMOVED (2026-06-16). This entire approach rested
on a **metric artifact**: the cadence tool counted trace frame-records (the
re-checkpoint frames emitted at every concurrent edge), not spans, manufacturing
the run-to-run "DDR jitter" this model was built to reproduce. Measured
span-based, HW PORT_RUNNING is *deterministic* (std 0), and the residual gap is
on the **compute-core-gated** ports (slot1/slot4), not the shim-DDR ports
(slot0/slot5, which match exactly). The `AIE2_DDR_PHOENIX` model + env vars are
deleted (`burst.rs` gone); it modeled a ghost and made slot0 worse (2 vs 1). See
`docs/superpowers/findings/2026-06-16-port-cadence-metric-was-frame-records.md`
and known-fidelity-gaps row "Held-level count under-emission". Kept below only as
a record of the calibration that was chasing the artifact.

---
**(Original IN PROGRESS status, retained for the record):** steps 1-4 landed
(PRNG range draws + seed/env wiring + fresh 100-run HW capture + calibrated
`AIE2_DDR_PHOENIX` profile). Step 5 (bridge no-regress) is trivially satisfied by
default-off; full lib suite green (3532). Design decisions resolved (§7).

## Calibration outcome (2026-06-15)

Fresh HW capture (step 3): 100 runs of `add_one_using_dma` on NPU1 (FW 1.5.5.391),
`build/experiments/port-running-baseline/highN-2026-06-15/`. Distribution target:
slot0 `{5:35%, 6:65%}` mean 5.65±0.48, slot4 `{7:72%, 8:27%}` mean 7.26±0.46,
slot1=8 / slot5=4 deterministic.

Calibrated (step 4) via `--emu` multirun sweep (the decoded-trace oracle — same
decoder as HW; the BP `cycle_beat` probe is NOT a valid oracle, it disagrees by
~2 bursts). Winner **`AIE2_DDR_PHOENIX` = burst `[36,46]`, gap `[8,14]`**
(N=50 confirm):

| slot | EMU | HW | |
|------|-----|----|----|
| slot0 | 5.34±0.47 `[5,6]` | 5.65±0.48 `[5,6]` | distinct exact, means <1σ |
| slot4 | 7.62±0.49 `[7,8]` | 7.26±0.46 `[6,7,8]` | means <1σ, misses rare 6 |
| slot1 | 8.00 | 8.00 | exact ✓ |
| slot5 | 4.00 | 4.00 | exact ✓ |

**Irreducible coupling:** slot0 and slot4 are both driven by the *same* burst
range with *opposite* gradients (slot0 wants smaller bursts → more breaks; slot4
wants larger), so no single range optimizes both. `[36,46]` sits on that Pareto
frontier — both stochastic slots within 1σ, deterministic slots exact, and no
out-of-HW-range outliers (a wider range gives slot4 its `[6,7,8]` spread but
introduces slot0=3, which HW never produces — a worse artifact). Good enough for
a ballpark, opt-in, per-DRAM profile (the spec's intent), not a claim of exact
distribution match.
**Supersedes:** the `ddr_burst_jitter_permille` ("deterministic jitter") knob
sketched in `2026-06-14-ddr-burst-delivery-model.md` §4.1 — replaced here by a
seeded-PRNG draw (Maya: "reproduce it by literal RNG").

---

## 1. Why this exists (the #140 conclusion)

The deep-dive in `2026-06-15-recv-port-cadence-root-cause.md` proved that the
**entire** memtile input-path PORT_RUNNING cadence gap — *both* slot0 and slot4 —
collapses to **one** physical cause: **shim host-DDR burst-delivery jitter.**

- slot0 (S2MM ← shim) undercounts because EMU delivers shim input too smoothly.
- slot4 (MM2S → compute) undercounts its *early* sub-bursts for the same reason:
  the memtile MM2S stalls waiting for the bursty shim-fed producer to refill the
  next buffer. (slot4's *late* bursts — compute-processing backpressure — EMU
  already reproduces faithfully. The consumer-coupling / objfifo / fabric model
  needs **no change**; verified by env-gated A/B.)
- Turning the existing (deterministic, fixed-param) `BurstGate` on lifts slot4
  6→7 with the correct gap *shape* and slot0 1→8 — proving the mechanism, and
  showing fixed params **overshoot** because the real effect is **stochastic**.

HW is itself non-deterministic here (other SoC/DDR traffic): slot0 = **5.73 ±
0.44** bursts, slot4 = **7.07 ± 0.25** (baseline `2026-06-15-port-cadence-hw-
baseline.md`). A single fixed cadence cannot match a *distribution*. The fix is a
**stochastic** delivery model whose output distribution band-matches HW.

This also reframes the project north star: **byte-identical trace applies to the
deterministic backbone; DDR-jitter-driven events (STREAM_STARVATION,
PORT_RUNNING sub-bursts, DMA-milestone timing) are band-matched, not
byte-matched** — because the silicon itself does not reproduce them run-to-run.

## 2. What already exists (do NOT rebuild)

- **`src/device/dma/burst.rs` `BurstGate` / `BurstParams`** — per-channel
  deliver-`burst_words`-then-idle-`inter_burst_cycles` state machine, with
  `first_access_latency`. Deterministic, resume-safe, default-off
  (`burst_words == 0` → `words_allowed` returns `u16::MAX`). Plumbed into
  `do_transfer_cycle` (`stepping.rs:1007`) for shim MM2S host reads, drawn down
  per delivered word via `consume`.
- **`DmaTimingConfig` + `from_env()` overlay** — `XDNA_EMU_DDR_BURST_WORDS`,
  `XDNA_EMU_DDR_INTER_BURST_CYCLES`, parsed once at device construction.
- The gate drives the existing S2MM starvation-edge logic *naturally* (we never
  emit starvation directly): an idle DDR cycle drains the consumer FIFO → stall
  edge fires. Physically faithful.

**This spec adds exactly one thing: per-burst random draws of `burst_words` and
`inter_burst_cycles` from configured distributions, with a seeded PRNG.**

## 3. The determinism contract (the load-bearing decision)

The emulator is deliberately deterministic and resume-safe. Randomness must not
erode that for normal use — resolved by making jitter **opt-in** and keeping a
**reproducibility escape hatch**:

- **Default-off ⇒ unchanged.** With DDR jitter disabled (the default — §7.3),
  the emulator is byte-identical/resume-safe exactly as today. Randomness only
  exists when a user *opts in* to the DDR model, and there it is the **intended
  behavior**: real DRAM delivery varies run-to-run with SoC/DDR traffic, so the
  model should too (Maya: "as random as possible").
- **Seed from system entropy by default.** When jitter is on, draw the seed once
  at device construction from the OS (`std::collections::hash_map::RandomState`'s
  process-random hasher, or `SystemTime` nanos — no `rand` crate). Each fresh run
  differs, like silicon. PRNG: tiny inline `splitmix64` (one `u64`, ~5 lines).
- **Escape hatch: log + override.** The drawn seed is **logged at startup**
  (`log::info!`) and overridable via `XDNA_EMU_DDR_SEED`. So any specific run can
  be replayed for debugging, and tests/CI pin it explicitly. This is what keeps
  "as random as possible" from costing us debuggability.
- **Resume-safe within a run.** PRNG state lives per-channel in `BurstGate`
  (already `Clone` → serialized/resumed for free); a paused-and-resumed run
  replays the same sequence. Non-determinism is *run-to-run* (fresh seed at fresh
  construction), never *within* a run.
- **Sweep needs no seed wiring.** Because each fresh EMU invocation self-seeds,
  the 8-batch sweep draws N independent samples for free; we just log each run's
  seed for reproducibility. (Simpler than per-batch seed injection.)
- **off1 / chained-BD / tile-local paths are untouched** — jitter is gated to
  shim host-DDR MM2S only (`is_shim() && involves_host_memory()`), exactly the
  current `BurstGate` gate. Bounds blast radius; protects the #26 off1 bubble
  and the k8 chain-sweep that parked the fixed-param default.

## 4. The stochastic model

### 4.1 What is randomized

Per *burst* (i.e. each time `BurstGate` reloads after a gap), draw:
- `burst_words` ~ a distribution centered on the AXI/DDR burst size,
- `inter_burst_cycles` ~ a distribution centered on the DDR refill gap.

`first_access_latency` stays deterministic (cold-start is already modeled by
`shim_ddr_cold_start_*`; do not double-count or randomize it).

### 4.2 Distribution shape

**Resolved:** discrete uniform over `[min, max]` per knob to start. HW slot0
modal gaps `[16, 16, 7, 9, 13]` → gap range ~`[7, 16]`; burst sizes ~`[8, 16]`
words (AXI ARLEN split). Uniform captures the observed range with one obvious
parameterization and no distribution-fitting rabbit hole.

**Refine from data, not theory:** the fresh HW capture (§5.3) is high-N
specifically so we can *observe the real delivery distribution shape* — if the
empirical burst/gap histogram is clearly non-uniform (bimodal, long-tailed), swap
the draw for an empirical-CDF sampler or a clamped Gaussian. The draw site is one
function (`BurstGate::next_in`); changing the shape is local. Start uniform, let
lots of HW runs tell us if a better shape is warranted.

### 4.3 Config surface (extend `BurstParams`)

```
// replaces the fixed burst_words / inter_burst_cycles with ranges:
ddr_burst_words_min:        u16
ddr_burst_words_max:        u16
ddr_inter_burst_cycles_min: u16
ddr_inter_burst_cycles_max: u16
ddr_seed:                   Option<u64>   // None = draw from system entropy at
                                          // construction; Some(s) = pin (tests/CI)
```

- `min == max` ⇒ degenerate (the current fixed-param behavior — full back-compat,
  the landed unit tests keep passing if expressed as equal bounds).
- `*_min == *_max == 0` on `burst_words` ⇒ disabled (uniform delivery), as today.
- `ddr_seed`: default `None` → system entropy, logged at startup (§3). Each
  channel mixes `(col,row,ch)` into the master seed so channels decorrelate.
- Env overlay: `XDNA_EMU_DDR_BURST_WORDS_{MIN,MAX}`,
  `XDNA_EMU_DDR_INTER_BURST_CYCLES_{MIN,MAX}`, `XDNA_EMU_DDR_SEED` (pins the
  otherwise-random seed). Keep the existing single-value `XDNA_EMU_DDR_BURST_WORDS`
  as an alias setting min=max (don't break the investigation scaffolding).

### 4.4 Phoenix-ballpark defaults

Derived from the HW baseline (documented OBSERVED, user-overridable). Calibrated
in step 3 below; seed the sweep from the investigation's working point
(`burst≈8, inter≈8`) which already put slot4 in-band.

## 5. Implementation steps (TDD)

1. **PRNG + range draws in `BurstGate`** (unit-tested in isolation):
   - inline `splitmix64`; `next_in(min,max)` inclusive.
   - `words_allowed` reloads `burst_left`/`gap_left` from *draws* not constants.
   - Tests: fixed seed → fixed sequence (reproducible); `min==max` reproduces the
     current deterministic cadence (the landed `burst.rs` tests, re-expressed);
     draws stay within `[min,max]`; distinct seeds decorrelate.
2. **Wire config + env overlay** (`BurstParams`, `DmaTimingConfig::from_env`).
   Back-compat alias for the single-value env vars. Per-channel seed derivation.
3. **Fresh high-N HW capture** (calibration target): re-capture
   `add_one_using_dma` PORT_RUNNING cadence on NPU1 across **many** runs (target
   ~50-100, not the baseline's 15) so we get the real burst/gap *distribution
   shape*, not just mean ± sigma. This both sets the calibration target and tells
   us whether uniform suffices or an empirical/Gaussian shape is warranted (§4.2).
4. **Calibrate defaults against that capture**: EMU param sweep × N self-seeded
   runs on `add_one_using_dma` in-process (reuse the beat-accurate
   `XDNA_EMU_BP_PROBE` harness), fit `burst`/`inter` ranges so slot0 → 5.73 ±
   0.44 and slot4 → 7.07 ± 0.25 *distributions*. Ballpark, not exact — and these
   are **Phoenix-LPDDR5 ballpark defaults**, not a universal truth (§7.3).
5. **Validate** (§6). Default stays **off** (opt-in framework — §7.3).

## 6. Validation strategy

Per the calibration guidance (ballpark + deterministic-per-seed, band-matched
across seeds — *never a single capture*):

- **Primary gate — band-sigma over the 8-batch sweep:** EMU runs self-seed, so N
  fresh runs give N independent samples with **no new harness wiring** (just log
  each run's seed for reproducibility). Build the slot0/slot4 burst-count
  distribution, compare mean ± sigma to the fresh HW band.
- **Unit (TDD):** §5.1 PRNG/draw tests; uniform-path unchanged for tile-local.
- **Regression:** full `cargo test --lib`; bridge corpus `TRACE_VERDICT` re-tally
  vs the 15-CLEAN / 125-DIVERGE baseline — must not regress a CLEAN kernel, and
  must move the STREAM_STARVATION / PORT_RUNNING / DMA-milestone family toward
  CLEAN (default-off run is byte-identical, so this is the *opt-in* tally).
- **Guard the parked regression:** the k8 `_diag_shim_chain_sweep` must not smear
  — jitter is shim-host-only and off by default, so chained tile-local BDs are
  untouched; assert it explicitly in the sweep.
- **Cross-check:** total-cycle ratio + mode-0 DMA anchors stay within tolerance
  (gaps add starvations, must not blow up end-to-end cycle counts past HW).

## 7. Decisions (resolved 2026-06-15)

1. **Distribution shape** → **uniform `[min,max]` to start**, refine to an
   empirical/Gaussian shape only if the fresh high-N capture shows uniform misses
   the real histogram. The draw is one function, cheap to swap (§4.2).
2. **Seed** → **as random as possible: system entropy at construction**
   (RandomState/SystemTime, no `rand` crate), per-channel mixed with
   `(col,row,ch)`. Logged at startup + `XDNA_EMU_DDR_SEED` override as the
   reproducibility escape hatch (§3). Default-off keeps the rest of the emulator
   deterministic.
3. **Default-on flip** → **stays OFF; opt-in framework.** There is no single
   correct calibration — burst/gap cadence depends on the user's actual DRAM
   (Phoenix LPDDR5 vs Strix DDR5 vs a hypothetical part). We ship Phoenix-ballpark
   defaults and let users dial their own DRAM/profile. No corpus-wide default-on.
4. **Calibration target** → **fresh high-N HW capture** (§5.3), ~50-100 runs, to
   get the real distribution *shape* (not just the ~15-run baseline's mean±sigma)
   — which also informs decision 1.

## 8. Scope boundaries

**In:** seeded-PRNG stochastic `burst_words` / `inter_burst_cycles` on the shim
host-DDR `BurstGate`; the range/seed config + env surface; calibration to the
slot0/slot4 HW bands; band-sigma validation wiring (per-batch seed).

**Out:** the consumer-coupling / objfifo / fabric model (verified faithful — no
change); tile-local stream delivery; `CONFLICT_DM_BANK` (separate mechanism);
exact per-event cycle matching; #129 iter (follow-on, reuses this loop); flipping
default-on (separate gated decision, §7.3).
