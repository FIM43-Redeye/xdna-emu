# Design: Phoenix-Survival Capture (output corpus + inventory)

**Date:** 2026-05-31
**Status:** **WIP CHECKPOINT** — design sections 1–2 approved and all scoping
decisions locked; sections 3–5 stubbed (paused mid-brainstorm, resumable cold).
Do **not** start implementation: the brainstorming flow has not reached the
written-spec review gate. Pick up by completing sections 3–5, then the spec
self-review + user-review gates, then `writing-plans`.

---

## Why this exists (read this first if resuming cold)

The Strix-Point (NPU2 / AIE2P) upgrade **replaces** the current Phoenix (NPU1 /
AIE2) devbox — it is a **one-way door** that deletes our only real NPU1. Today
the emulator's validation oracle *is* the live NPU1 (fuzzer, bridge tests, trace
diffs all phone the silicon). The moment Phoenix is gone, every currently-passing
AIE2 behavior becomes **unverifiable forever**, and a future refactor (especially
the AIE2→AIE2P generalization) could silently regress it with nothing to catch
the break.

**The concrete gap that motivated this** (`runner.rs:506-545`): the fuzzer saves
NPU ground truth **only on mismatch**. Of ~1625 saved goldens, every one is a
case the emulator got *wrong*. We have **zero frozen ground truth for the ~98% of
cases the emulator currently gets right** — exactly the cases a regression would
silently break.

**Goal:** freeze enough of NPU1, in a replayable-without-hardware form, that the
emulator's AIE2 fidelity stays regression-testable forever. This spec designs the
**output corpus** track in full and inventories the other Phoenix-gated tracks.

### Operating principles (the framing that de-risks the swap)

- **Maya controls the door.** There is no swap date and no clock; the upgrade
  happens only when Maya is *genuinely confident*, never on a deadline. This plan
  is the machine that produces that confidence. (See memory
  `project_strix_swap_replaces_phoenix`: the constraint is the one-way door, not
  a ticking clock.)
- **The swap is expansion, not reset.** The emulator is built "derive from the
  toolchain" — TableGen ISA, the AM025 register DB, and device models are
  already architecture-*parameterized*, not Phoenix-hardcoded. AIE2P is layered
  *on top of* a saved AIE2 checkpoint; the NPU1 work transfers. The corpus is
  what makes that checkpoint permanent.
- **Unknown unknowns can't be zeroed — make them cheap to absorb.** The corpus is
  *targeted* insurance (freezes behaviors we know to ask about). The hedge
  against what we *haven't* thought of is **breadth + rawness**: capture more raw
  Phoenix than we know we need, in a form future-us can re-interrogate for
  questions not yet formulated. This is an explicit goal of the inventory
  (§ Track D below), not an afterthought.
- **Methodical pace; the time pressure is illusory** (memory
  `feedback_pace_methodical_over_fast`). Per the swap memory: capture
  *exhaustively*, don't optimize for Phoenix-minutes.

---

## Locked decisions (from the brainstorm)

1. **Scope:** design the **output corpus** track fully now; **inventory** the
   trace-corpus and HW-gated-verification tracks (plus the raw-breadth hedge) as
   named follow-on specs. One door-closing checklist, one buildable spec.
2. **Storage:** **hybrid** — a curated, stratified subset committed to git as the
   backed-up regression gate (~few hundred–1000 cases, ~10–15 MB); the full
   capture archived to `~/npu-work/experiments/phoenix-aie2-golden-<date>/`.
   (Per-case replay footprint ≈ 15 KB: `aie.xclbin` 9 KB + `aie.mlir` 5.5 KB +
   `npu_output.bin` 64 B. The frozen xclbin pins the binary↔golden pairing, so
   the corpus is immune to Peano toolchain drift.)
3. **Composition:** **stratified grid generation** on the controllable axes +
   **classify-and-tag** for emergent regimes (approach C — see § Section 2).
4. **Done = swap-safe:** an **automated regression gate** (`cargo test` over the
   curated subset + a script) replaying with **zero HW**, **plus** a per-case
   **golden-output diff report** (which elements diverged + signatures, so a
   future regression is diagnosable, not just red/green), plus a committed
   coverage report and the archived full capture + manifest.
5. **Capture both classes:** the campaign produces `agree/` (EMU==HW) **and**
   `diverge/` (EMU≠HW) cases; **both are frozen as first-class data.** The agree
   set gates against regressions; the diverge set is frozen golden-divergence
   data so a *fix* can later be proven against real silicon we no longer have.

---

## Section 1 — Architecture & data flow  *(APPROVED)*

Four stages, left to right:

```
  [1] CAPTURE                [2] CORPUS              [3] CURATE            [4] GATE
  grid-driven campaign  -->  full capture       -->  stratified subset --> zero-HW replay
  (fuzzer --capture,         (~/npu-work archive)     + provenance          + diff report
   live Phoenix)             agree/ + diverge/        -> git corpus         (cargo test + script)
```

- **[1] Capture** runs only while Phoenix lives. A new fuzzer capture mode drives
  generation from the coverage grid (approach C), runs each case on EMU **and**
  real NPU1, and persists **every non-vacuous case** — `aie.xclbin` +
  `npu_output.bin` + `meta.json`, and on divergence also the frozen
  `emu_output.bin`. Cases split into `agree/` (EMU==HW) and `diverge/` (EMU≠HW).
- **[2] Corpus** is the full self-contained capture, archived to
  `~/npu-work/experiments/phoenix-aie2-golden-<date>/`. Self-contained = the
  frozen xclbin pins the binary↔golden pairing (toolchain-drift-proof).
- **[3] Curate** selects a stratified, deduplicated subset (guaranteed grid
  coverage + all interesting-regime tags + all divergences) and copies it into a
  committed git corpus dir, with a coverage report and manifest.
- **[4] Gate** replays the committed corpus through the in-process `XclbinSuite`
  (the `validate_seeds` mechanism, `examples/validate_seeds.rs`) with **zero HW**:
  asserts agree-cases stay EMU==frozen-HW (regression tripwire) and diverge-cases
  stay EMU==frozen-EMU (known-divergence tripwire), emitting a per-case diff
  report.

**Key property:** once [4] exists, NPU1 fidelity is permanently regression-testable
without NPU1. An AIE2→AIE2P refactor that silently breaks an AIE2 behavior turns
the gate red.

---

## Section 2 — Coverage grid + tagging (approach C)  *(APPROVED)*

**Controlled axes — constrained generation (guaranteed grid).** A new `gen.rs`
entry fixes the three top-level `FuzzParams` fields and randomizes only the op
body (seed still drives the body, so determinism holds):

| axis | values | n |
|------|--------|---|
| dtype | i8, i16, i32 | 3 |
| buffer size | 16, 32, 64, 128, 256 | 5 |
| loop style | Simple, HardwareLoop | 2 |
| op-count band | 1–4, 5–9, 10–16 | 3 |

Grid = **90 cells**; capture **N per cell** (default 12, tunable) → ~1,080
controlled cases; every cell guaranteed non-empty.

**Emergent regimes — classify-and-tag during capture (honest, not forced).** The
fidelity-critical behaviors that only *emerge* from random bodies get detected
(from disasm/params) and tagged in `meta.json`, never forced through generation:

- `store_at_le` (the BUG-B back-edge flush regime)
- `has_load`, `has_branch_taken`, `has_hwloop`, `multi_store`
- `vacuous` (both-sides-zero — tagged and **excluded from the gate**)

Curation guarantees each tag has ≥ M representatives (default 8), oversampling the
campaign if a tag is thin. Concretely: run the 90-cell grid at N each for
controlled coverage, **plus** a large random overflow run whose job is to harvest
enough tagged emergent cases to satisfy the M-per-tag floor.

**Named risk (small, bounded):** `store_at_le` is a *rare* emergent regime (low
single-digit % of random seeds), so its M-floor may need a big overflow run.
Lever: the `XDNA_FUZZ_RECENCY1` harvest mode (already in `gen.rs`) boosts that
regime ~5×, and ~1600 already-labeled cases exist to draw from. Known risk, known
lever — not a hole. Grid dims, N, and M are spec parameters (defaults given).

---

## Section 3 — Corpus schema, storage & durability  *(TODO — resume here)*

Sketch to flesh out:
- Committed corpus dir (proposed `tests/corpus/phoenix-aie2/` — confirm location)
  with `agree/` and `diverge/` subtrees; per-case dir
  `{cell-id}_{seed}/{aie.xclbin, npu_output.bin, meta.json[, emu_output.bin]}`.
- `meta.json` schema: seed, dtype, size, loop_style, op-count band, cell_id,
  class (agree/diverge), tags[], op-signature, vacuous flag, capture_date,
  **toolchain provenance** (Peano commit/version, driver/firmware version),
  `npu_output_sha`, `xclbin_sha`.
- `manifest.json` (whole-corpus) + `coverage.md` (cells × counts, tag counts).
- Archive layout under `~/npu-work/experiments/phoenix-aie2-golden-<date>/`.
- Decide: how much goes in git vs archive (the curated subset size / selection).

## Section 4 — Regression gate + diff report  *(TODO)*

Sketch to flesh out:
- `cargo test` (e.g. `tests/corpus_replay.rs`) + `scripts/replay-corpus.sh`,
  replaying via in-process `XclbinSuite` (generalize `validate_seeds`).
- Assertions: agree-set EMU==frozen-HW; diverge-set EMU==frozen-EMU (HW side is
  immutable reference). When a divergence gets *fixed*, its case "fails" the
  diverge tripwire → deliberate signal to **re-classify diverge→agree** (a corpus
  update, not a bug). Document this lifecycle.
- Diff-report artifact: per failing case, which elements diverged + signatures.

## Section 5 — Error/edge handling, testing & the inventory  *(TODO)*

- Capture-time edge handling: compile failures, HW panics/timeouts (already an
  `error` category in `runner.rs`), vacuous exclusion, determinism re-check.
- Testing of the capture/curation tooling itself (not just the corpus).
- **Inventory of the remaining Phoenix-gated tracks** (named follow-on specs):
  - **Track B — Trace corpus:** golden NPU1 Perfetto traces for the
    trace-fidelity regression. Different pipeline (trace tooling/strategy layer;
    decode partly upstream-owned). See `docs/trace/`.
  - **Track C — HW-gated verifications:** the Vector / SideEffect "Unverified"
    semantics (coverage memory flags these; Perf_Counter cycle oracle broken for
    NPU1). **First sub-task: confirm the recording isn't stale** — a grep for
    Unverified markers in `docs/coverage/aie2/implementation-gaps.md` found
    nothing, so the open-item list itself may need a pass.
  - **Track D — Raw-breadth reservoir (the unknown-unknown hedge):** broad raw
    Phoenix artifacts *not* tied to current understanding — raw traces across
    diverse kernels, capability/register surveys, firmware+driver+toolchain
    version pinning — stored raw for future re-interrogation. Explicit goal:
    make unknown unknowns cheap to absorb.

---

## Resume checklist

1. Flesh out Sections 3–5 (above), getting per-section approval.
2. Spec self-review (placeholders / consistency / scope / ambiguity).
3. User review gate (Maya reviews this file).
4. Invoke `writing-plans` for the implementation plan.

Related: memory `project_strix_swap_replaces_phoenix`,
`project_coverage_plan2_done_regroup_before_plan3`, `feedback_pace_methodical_over_fast`;
the BUG-B findings doc (`docs/superpowers/findings/2026-05-31-bugb-zol-store-flush-investigation.md`)
for the `store_at_le` regime and the recency harvest mode.
