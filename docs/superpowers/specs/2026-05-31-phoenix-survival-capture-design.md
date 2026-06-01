# Design: Phoenix-Survival Capture (output corpus + inventory)

**Date:** 2026-05-31
**Status:** **APPROVED — implementation plan being written.** All five design
sections approved by Maya (2026-06-01); self-review done. `writing-plans`
produces the implementation plan next. Do **not** start implementation before the
plan exists. Scope note: this completes the spec *as designed* — a larger aiesim
role (Section 5) is intentionally deferred to a follow-up, not folded in here.

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
   (Per-case *committed* footprint ≈ 9–10 KB: `aie.xclbin` 9 KB +
   `npu_output.bin` 64 B + `meta.json`. The source `aie.mlir` lives in the
   archive only, not the committed corpus — see Section 3, so the gate never
   re-parses MLIR. The frozen xclbin pins the binary↔golden pairing, so the
   corpus is immune to Peano toolchain drift.)
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

## Section 3 — Corpus schema, storage & durability  *(APPROVED)*

**Committed corpus location.** `tests/corpus/phoenix-aie2/` — it is replayed by a
`cargo test` (Section 4), so it lives under `tests/`. Two subtrees by class:
`agree/` (EMU==HW at capture) and `diverge/` (EMU!=HW at capture). Each case is a
directory `{cell_id}_{seed}/`:

```
tests/corpus/phoenix-aie2/agree/i8-128-Simple-5to9_000412/
  aie.xclbin       # frozen binary (~9 KB) -- pins the binary<->golden pairing
  npu_output.bin   # real NPU1 output (the golden)
  meta.json        # everything the gate needs to replay + classify
diverge/<...>/     # same three, PLUS:
  emu_output.bin   # frozen EMU output (the known-divergence reference)
```

**No `aie.mlir` in the committed corpus.** The gate must not depend on re-parsing
MLIR; `meta.json` carries dtype/size/buffer-spec directly. (This is a deliberate
change from `examples/validate_seeds.rs`, which parses `aie.mlir` for dtype/size
— the corpus replayer reads `meta.json` instead.) The `aie.mlir` source is kept
in the archive for provenance, not in the gate path.

**Schema-first.** `meta.json` and `manifest.json` are defined as serde Rust types
and serialization is derived — types first, JSON second:

```rust
struct CaseMeta {
    schema_version: u32,
    seed: u64,
    cell_id: String,              // "i8/128/Simple/5-9"
    dtype: ElementType,           // i8 | i16 | i32
    size_elements: usize,
    loop_style: LoopStyle,        // Simple | HardwareLoop
    op_count: u32,
    op_count_band: OpBand,        // 1-4 | 5-9 | 10-16
    class: Class,                 // Agree | Diverge
    tags: Vec<Tag>,               // store_at_le, has_load, has_branch_taken, has_hwloop, multi_store
    op_signature: String,         // stable hash of the op sequence (dedup key)
    vacuous: bool,
    buffer_spec: BufferSpecMeta,  // group_ids, dirs, input patterns -> reconstruct the run
    capture: Provenance,          // date, peano_commit, driver_version, fw_version, determinism_ok
    hashes: Hashes,               // xclbin_sha256, npu_output_sha256, emu_output_sha256?
}
```

`buffer_spec` is load-bearing: `validate_seeds.rs` hardcodes a single
`fuzz_spec`, but a general corpus must record each case's actual buffer layout so
the replayer reconstructs inputs deterministically.

**Whole-corpus index.** `manifest.json` (a `Manifest` struct): schema_version,
campaign id + date, counts (total / agree / diverge / vacuous-excluded / error),
per-cell coverage map, per-tag counts, a toolchain-provenance summary, and a flat
case index (path, class, tags) for fast loading. `coverage.md` is the
human-readable companion: the 90-cell grid as a counts table, tag counts, and a
note on any thin or oversampled cells.

**Archive (full capture).** `~/npu-work/experiments/phoenix-aie2-golden-<date>/`
holds the *entire* campaign: every non-vacuous case (agree + diverge), the
`aie.mlir` sources, any captured `emu_trace.bin`/`npu_trace.bin`, the `errors/`
bucket (Section 5), and the manifest. Self-contained and toolchain-drift-proof.

**git vs archive — the curated subset.** The committed corpus is a stratified
selection from the archive, ~few hundred–1000 cases (~10–15 MB), chosen by a
deterministic (seed-ordered) algorithm:
- **grid floor:** for each of the 90 cells, the first min(K, available) cases,
  guaranteeing >=1 per non-empty cell;
- **tag floor:** every emergent tag gets >=M representatives (oversample the
  campaign if thin — Section 2's `store_at_le` risk + the RECENCY1 lever);
- **all divergences:** *every* `diverge/` case is committed — they are rare and
  unrecapturable once Phoenix is gone;
- **dedup:** within a cell, drop cases sharing an `op_signature` so
  near-identical kernels do not eat the budget.
The selection is recorded in the manifest, so the committed subset is
reproducible from the archive.

**Durability.** The frozen xclbin pins the binary<->golden pairing, so the corpus
is immune to Peano/toolchain drift (we replay the binary, never recompile).
`capture.*` records what produced each golden; `hashes.*` detect corruption. The
gate needs no hardware, license, or network — it is the tripwire that outlives
Phoenix.

## Section 4 — Regression gate + diff report  *(APPROVED)*

**The gate.** `tests/corpus_replay.rs` (a `cargo test`) plus
`scripts/replay-corpus.sh` for ad-hoc runs. It generalizes
`examples/validate_seeds.rs`: per case it reads `meta.json`, reconstructs the
`BufferSpec` from `buffer_spec`, runs the xclbin through the in-process
`XclbinSuite` (zero HW), and compares bytes. The hardcoded `fuzz_spec` and
`aie.mlir` parsing are replaced by `meta.json`-driven reconstruction.

**Assertions.**
- **agree case:** `EMU == npu_output.bin` (frozen real HW). A mismatch means the
  emulator regressed against silicon — the core tripwire.
- **diverge case:** `EMU == emu_output.bin` (frozen EMU at capture). The HW side
  stays the immutable reference; this catches *any* change in EMU behavior on a
  known-divergent case.
- **vacuous:** excluded (both-sides-zero tests nothing).

**Diverge->agree lifecycle.** When someone *fixes* a known divergence, that
case's EMU output stops matching the frozen EMU and starts matching the frozen
HW — so the diverge tripwire goes red **by design**. That red is the signal to
**re-classify** the case `diverge -> agree` (move it, flip `class`; it now gates
against HW forever). A helper (`scripts/reclassify-corpus-case.sh <case>`) does
the move + meta update. This is a corpus update, not a bug — documented so a
fixer knows the red is expected and what to do.

**Diff report.** On any failing case the gate writes, to
`build/corpus-replay-report/`, a per-case record: first-diff element index, total
diverging elements, the `(expected, got)` values at the first few diffs, and a
signature hash; a human summary table lists failures with signatures. A future
regression is thus *diagnosable* (which lanes, what pattern) rather than a bare
red/green — the difference between "something broke" and "element 5 is off by the
store-flush amount."

**Cost & CI.** In-process `XclbinSuite` replay is fast and HW-free; the curated
subset (hundreds of cases) fits a normal `cargo test`/nextest budget. The gate
runs in ordinary CI with no Phoenix, no license — which is the entire point: it
is the regression guard that still works after the silicon is gone.

## Section 5 — Error/edge handling, testing & the inventory  *(APPROVED)*

**Capture-time edge handling.**
- **Compile failures** (Peano can't build a generated kernel): logged, skipped,
  never a corpus case (the fuzzer already does this).
- **HW errors** (panic / timeout / TDR — the existing `error` category in
  `runner.rs`): the HW output is untrustworthy, so these are **not** frozen as
  agree/diverge. They go to an `errors/` archive bucket for triage, classified
  catastrophic (dmesg NOAVAIL) vs recoverable (TDR) per the existing discipline.
  Excluded from the corpus.
- **Vacuous** (both-sides-zero): tagged and gate-excluded.
- **Determinism gate:** before a case is frozen, its EMU run is repeated and must
  be byte-identical; non-deterministic cases are flagged and excluded (we only
  freeze deterministic goldens). HW-side flakiness is caught by the campaign's
  repeat captures; determinism status is recorded in `meta.capture`.

**Testing the tooling itself** (not just the corpus). Schema-first pays off here:
`CaseMeta`/`Manifest` round-trip (de)serialization tests; a unit test for the
curation selection algorithm (grid floor / tag floor / all-divergences / dedup)
over a synthetic case set; and a tiny synthetic fixture corpus (a couple of
hand-made agree + diverge cases) that exercises the gate's pass / regress /
diverge-tripwire / diff-report paths without the real corpus. The
capture/curate/replay code is covered like any subsystem (the finish-to-100%
rule).

**Complementary (non-Phoenix) oracle: aiesim.** *Newly proven this session*
(`docs/aiesimulator.md`, "Running Peano-compiled cores"): a Peano core runs
correctly through AMD's cycle-approximate `aiesimulator` (scalar + vector),
giving a **functional oracle that needs no Phoenix** — it can answer *new* AIE2
questions after the swap, where the frozen corpus only answers the ones we
thought to capture. It is *complementary*, not a substitute: license-gated and
not silicon-exact, so it does not replace the distributable, silicon-exact
corpus; it does reduce how much purely-functional breadth the corpus must carry.
**A larger role for aiesim is intended (to be defined after this spec
consolidates) and is deliberately out of scope here** — noted so the spec
records the asset without re-architecting around it.

**Inventory of remaining Phoenix-gated tracks** (named follow-on specs):
- **Track B — Trace corpus:** golden NPU1 Perfetto traces for the trace-fidelity
  regression. Different pipeline (the trace tooling/strategy layer; decode partly
  upstream-owned). See `docs/trace/`.
- **Track C — HW-gated verifications:** the Vector / SideEffect "Unverified"
  semantics (coverage memory flags these; the Perf_Counter cycle oracle is broken
  for NPU1). First sub-task: confirm the recording isn't stale — a grep for
  Unverified markers in `docs/coverage/aie2/implementation-gaps.md` found nothing,
  so the open-item list itself may need a pass. (The aiesim oracle above may now
  cover part of this functionally — weigh that when Track C is specced.)
- **Track D — Raw-breadth reservoir (the unknown-unknown hedge):** broad raw
  Phoenix artifacts *not* tied to current understanding — raw traces across
  diverse kernels, capability/register surveys, firmware+driver+toolchain version
  pinning — stored raw for future re-interrogation. Explicit goal: make unknown
  unknowns cheap to absorb.

---

## Next steps

1. ~~Flesh out Sections 3–5.~~ **Done** — all five sections written.
2. ~~Spec self-review (placeholders / consistency / scope / ambiguity).~~ **Done.**
3. ~~User review gate (Maya reviews this file).~~ **Done — approved 2026-06-01.**
4. **Invoke `writing-plans` for the implementation plan.** <- we are here.

Related: memory `project_strix_swap_replaces_phoenix`,
`project_coverage_plan2_done_regroup_before_plan3`, `feedback_pace_methodical_over_fast`;
the BUG-B findings doc (`docs/superpowers/findings/2026-05-31-bugb-zol-store-flush-investigation.md`)
for the `store_at_le` regime and the recency harvest mode.
