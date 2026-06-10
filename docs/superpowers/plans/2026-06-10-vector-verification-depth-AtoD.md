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

## B. Edge inputs on silicon

- bf16 denormals (FTZ), NaN/Inf canonicalization (the conv kernel explicitly
  **deferred** these; MAC-bf16 used an all-finite slice), overflow edges, full range.
- Extend input selection (`select_records` predicate) to edge slices; HW-capture.
- Known: bf16 NaN payload -- emulator HW-correct, model outlier (Half-A). Confirm on
  silicon and record.

## C. Op breadth on silicon

- Ops not yet HW-captured: shuffle **routing** (Half-A enum-verified, routing
  HW-gated), vsel, vcmp, vshift, vmin/vmax, reductions.
- Author kernels per op; HW-capture; fix divergences. Breadth, not depth.

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
