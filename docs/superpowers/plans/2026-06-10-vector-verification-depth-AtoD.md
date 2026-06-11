# Vector-Compute Verification Depth (A -> B -> C -> D)

**Goal:** Deepen AIE2 vector-compute *silicon* verification beyond one-mode-per-class,
then flip `clean_release(Aie2)` green with an airtight `Verified{evidence}` claim.

**Why now (the motivating finding):** the pack-saturation gap (2026-06-10) proved a
class can be **model-exact yet silicon-divergent in an unexercised mode**. Half A
showed emulator == aietools model across the full mode space (bit-exact); Half B
showed emulator == silicon for *one representative mode per class*. The bridge
between them is the assumption "model == silicon" -- and pack violated it (model
truncated, silicon saturated; only running the saturate mode on HW caught it).
Deepening closes exactly that assumption.

## Current state (pre-deepening) -- the resume anchor

- Branch `vector-halfb-silicon-verify` @ `f47dfc4` (pack `crSat` fix committed,
  HW-validated). Parent line is `master` (the dev->master swap landed earlier).
- **All 9 representative vector kernels HW==EMU green**, trace CLEAN, 0 TDR
  (`build/bridge-test-results/20260609/`, durable). Classes covered on silicon:
  SRS (`vec_srs_i32`), UPS (`vec_ups_i32`), Pack truncate+saturate
  (`vec_pack_i16` + `vec_pack_i16_sat`), Convert/bf16 (`vec_conv_bf16`),
  MAC int8/int16/bf16 (`vec_mac_i8/i16/bf16`), element-wise (`vec_eltwise_add`).
- `clean_release(Aie2)` **deliberately NOT yet flipped** -- deepening first, by
  Maya's call (greening it == the machine-checked "safe to retire NPU1" signal;
  the Strix swap is a one-way door gated by our confidence, no external clock).
- Half A complete (#103): emulator == genuine aietools model, full discrete mode
  space, bit-exact (modulo HW-correct bf16-NaN payload).

**Order is A -> B -> C -> D (Maya's call: nice and easy, in sequence).**

---

## A. Mode-exhaustive on silicon  [DONE 2026-06-10 -- 45/45 HW==EMU green]

**Result:** generator extended with `SweepSpec` (commit `1008cde`); 45 mode-point
kernels generated (SRS 10rnd x 3sat = 30, Convert 10rnd, Pack 3sat incl. the
never-before-HW-tested symmetric point, UPS 2sat). Single full-bridge artifact
`build/bridge-test-results/20260610/` (run log
`build/experiments/vector-sweep-A/bridge-run-confirm.log`): **45/45 PASS on both
Chess/HW and Chess/EMU**, 0 fail.

**Bug the sweep flushed out (commit `26874db`):** the mode sweep recompiled the
SRS kernel and Chess happened to emit a fused **post-increment** narrowing store
`VST.SRS.s16.s64 cm0, s0, [p1], #32`. `get_store_address` assumed the pointer sat
at `sources[1]`, but fused stores carry sources `[acc, shift, pointer]`, so it
read the **shift register (=4) as the base address** and scattered the first 16
outputs to ~0x4 -> output buffer zero -> all 30 SRS mode-points failed EMU while
passing on silicon. Fixed by searching `sources` for the `PointerReg` (mirroring
the already-correct fused-*load* path `get_address`); `post_modify` carries the
increment. Two regression tests; `cargo test --lib` 3334 pass. This is a
codegen-lottery latent bug -- the anchor `vec_srs_i32` passed earlier only
because that compile emitted separate srs+store / indexed (`Memory{}`) stores. It
validates the whole mode-exhaustive premise: a fresh compile surfaced a path no
passing kernel had exercised.

**Original plan text (retained for reference):** Run each class's **full mode
space** on HW, not one representative -- the same move that caught pack.

**Mode spaces to sweep:**
- SRS: rounding {Floor, Ceil, Even/ConvEven, PosInf, NegInf, ...} x saturation
  {none, saturate, symmetric}. One point HW-checked so far.
- UPS: shift x sign x saturation. One point so far.
- Pack: saturation {none, saturate, symmetric}. truncate + saturate done;
  **symmetric (crSat mode 3) remains**.
- Convert (f32->bf16): ~10 rounding modes. Only ConvEven (normal-finite) so far.
- Element-wise: op variants beyond add (sub/mul/min/max/and/or/xor/shift) -- this
  edge bleeds into C.

**Generator extension (the build):** parametrize `KernelSpec` over the mode axes
(`tools/vector_kernel_specs.py` currently bakes a single `set_rounding` /
`set_saturation` into the body + selects one Half-A golden slice). Emit one kernel
per mode-point; bake the matching Half-A golden slice (corpus
`tools/golden/vector_ops.json` already has every mode -- no re-oracle needed).
Suggested CLI: `gen_vector_kernel.py <class> --sweep` or a sweep registry.

**Run:** stage all -> `../mlir-aie/test/npu-xrt/` -> bridge HW sweep
(`./scripts/emu-bridge-test.sh --chess-only <filter>`). Every model-vs-silicon
divergence found = a real fidelity fix, derived from the toolchain (like pack:
`crSat` -> `PackMode::from_sat_flags`).

**Files:** `tools/gen_vector_kernel.py`, `tools/vector_kernel_specs.py`,
`tests/vector-verify/`. **Acceptance:** every reachable mode per class HW==EMU;
divergences fixed.

## B. Edge inputs on silicon  [DONE 2026-06-10 -- 11/11 EMU==silicon green]

**Result:** bf16 edge frontier silicon-verified. Built a HW-observed golden tier
(`tools/golden/silicon_edge/`, capture tool `tools/capture_silicon_edge.py`):
each edge kernel runs once on real NPU1 Phoenix, its output becomes `EXP`, and the
model-vs-silicon divergences are recorded. Corpus regrown with rich bf16 edge inputs
(dense denormals straddling the FTZ boundary, more NaN payloads, 54 overflow tiles)
without perturbing phase-A baked slices (safety-gated byte-identical). 11 kernels:
`vec_conv_bf16_edge_r{0,1,2,3,8,9,10,11,12,13}` (denormal/NaN/Inf x 10 rounding
modes) + `vec_mac_bf16_ovf` (overflow tiles). **All 11 EMU==silicon**; full vector
regression 134/134 green, `cargo test --lib` 3335.

**What silicon said (the findings):**
- **Convert denormals: silicon ROUNDS them, does NOT flush-to-zero.** 0 divergences
  vs the aietools model across all 10 modes; 46 denormal lanes produce nonzero bf16
  (e.g. `0x00008000`->`0x0001`). The FTZ hypothesis was wrong for this datapath --
  the `VST.CONV` fused store-convert path rounds (no input FTZ), matching silicon.
- **MAC overflow: 638/640 divergent lanes are silicon canonical NaN mantissa=1**
  (model uses `0x7F`); the emulator already matched silicon there. Inf results agree.

**Bug the silicon flushed out (commit `430d841`):** the real divergence was NOT
denormal handling -- it was **accumulator-write visibility**. `VMOV bml,x` (wide
vector->accumulator move) wrote the accumulator immediately, but AIE2 gives it a
2-cycle def latency. Chess packed the convert-stores into a `RET`'s delay slots, so
each delay-slot `VST.CONV` read the freshly-overwritten accumulator -> Inf/NaN
garbage shifted one group forward (store before the branch correct; the 5 in delay
slots corrupted). Fix: defer the vmov-to-accum write through `queue_matmul_accum_write`
(is_half). This advances the deferred `FIXME(bypass-model)` accumulator-visibility
gap (functionally, for the vmov-to-accum case).

**Open, migrated to C (op breadth):** the *other* convert FTZ paths in
`vector_convert.rs` -- the **`VCONV` f32->bf16**, **bf16->f32 expand** (line 272),
and **`vfloor` bf16->s32** (line 342) -- still apply `fp32_flush_to_zero` and are
**untested on silicon** (the phase-B kernel uses `VST.CONV`, a different instruction,
which does not FTZ). Since silicon proved no-FTZ for the narrow-convert datapath,
these blanket-FTZ paths (added in `ef77756` as an unvalidated accuracy-era
generalization) are suspect. Author VCONV/vfloor/expand denormal kernels in C and
confirm on silicon while Phoenix is available.

## C. Op breadth on silicon  [DONE 2026-06-10 -- convert FTZ audit; bug fixed]

**Scope (Maya's call):** Phase C reduced to the **convert FTZ-path audit only**;
op breadth (shuffle routing, vsel/vcmp/vshift/min-max/reductions) folds into the
phase-D fuzzer (#112) rather than hand-authored kernels.

**Result -- the audit caught a real bug.** Two silicon kernels on NPU1 Phoenix
(exhaustive bf16 denormal sweep: all 254 denormals + +/-0), via the phase-B
silicon-golden tier + a new generator direct-input mode (`DirectIO`):
- **`vec_vfloor_bf16_denorm`** (standalone `VFLOOR.s32.bf16`): silicon floors a
  tiny negative bf16 denormal to **-1**; the emulator FTZ'd it to **0** -> EMU
  FAIL. **Silicon does NOT flush-to-zero.**
- **`vec_convexp_bf16_denorm`** (bf16->f32 expand): Chess fuses this into
  `VLDA.CONV.fp32.bf16` (no standalone `VCONV` for a load->store kernel). Silicon
  widens denormals (`0x8001`->`0x80010000`, sign preserved); the emulator's
  fused-load path (`memory/mod.rs:891-900`) already did no-FTZ -> EMU PASS,
  silicon-validated. (The bf16->f32 *expand* direction is new coverage; phase B
  only did f32->bf16 narrowing.)

**The bug + fix (`506e7cb`).** Root cause was NOT the `vector_convert` branch this
plan originally named. The compiled VFLOOR routes through a **separate wide-path
implementation, `vector_floor_bf16_to_s32` (`vector_arith.rs:1209`)**, which had
its own `fp32_flush_to_zero`. The first attempt patched `vector_convert` (unit
test passed) but the bridge EMU re-verify still FAILed -- end-to-end verification
caught the wrong-function fix before it shipped. Removed the FTZ from the real
path **and** the parallel `vector_convert` bf16->f32 / f32->bf16 / bf16->int32
branches (silicon-grounded; f32->bf16 also grounded in phase B's fused-store
finding). Both kernels now **EMU == silicon**; `cargo test --lib` 3340 pass. The
f32->int FTZ is kept as a **documented provable no-op** (`97bf60b`): an f32
denormal truncates to 0 either way.

**Deferred (hardware instability 2026-06-10 -- mainboard/GPU faulty, board swap
imminent):** the broader bf16-family EMU regression sweep and the two-stage code
review of the fix. The fix is already verified (unit 3340 + both kernels
EMU==silicon); these are extra-confidence steps for when the box is stable.
Commit chain: `dfc0490`..`97bf60b` (direct-input mode, audit specs, silicon
goldens `3526869`, fix `506e7cb`, f32->int doc `97bf60b`).

## D. Vector differential fuzzer  [gold standard, biggest build]

- Generate random valid vector kernels (random modes/inputs/ops), run HW vs EMU,
  diff. The roadmap's planned differential fuzzer. Today's fuzzer
  (`src/fuzzer/ast.rs`) is **scalar-only** -- extend the AST to vector ops.
- The "as deep as it gets" verification and the strongest pre-Strix confidence
  signal. Catches what any hand-picked suite misses.

---

## The gate flip (after A, or A-C, per confidence)

Once mode-exhaustive (A) [+ B/C] is green:
1. Add `override_registry` `Verified{evidence}` entries for the perishable vector
   `SemanticOp`s (`crates/xdna-archspec/src/coverage/units.rs`, mirror the `#104`
   Intrinsic-Accept entry). `is_perishable` = `AietoolsModeled|DocSpecified`
   provenance + `Unverified`; setting `Verified` removes them from the queue.
2. Regen coverage artifacts: `cargo run -p xdna-archspec --example gen_coverage_artifacts`
   (`docs/coverage/aie2/perishable-queue.md` empties).
3. **Update the tests that assert the gate is red** --
   `crates/xdna-archspec/src/coverage/mod.rs` `clean_release_is_false_via_perishable_not_comprehension`
   (~:269) and siblings encode "vector ops perishable until silicon verification";
   now satisfied.
4. Confirm `clean_release(Aie2) == true`; `cargo test --lib`; commit.
5. Evidence string per class: "Half-A model-exact full mode space + Half-B silicon
   mode-exhaustive HW==EMU <date> (kernels ...)".

Original Half-B plan: `docs/superpowers/plans/2026-06-08-vector-compute-halfB-silicon.md`
(Task 5 = the flip). This doc supersedes its "one representative mode" bar with
the mode-exhaustive bar the pack finding justified.
