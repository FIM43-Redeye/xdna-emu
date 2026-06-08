# Vector-compute differential audit (Half A) -- roll-up

**Date:** 2026-06-08. **Task:** #103. **Status:** Half A complete.
**Plan:** `docs/superpowers/plans/2026-06-08-vector-compute-differential-audit.md`.

## What Half A did

Bit-exact differential of the emulator's AIE2 vector semantics against the
**genuine** aietools Python reference models (not the prior hand-port), driven
out-of-repo as an oracle; only derived golden JSON is committed. The generator
(`tools/gen_vector_golden.py`) was first **de-circularized** -- it now drives
the real model functions (`srs.srs_lane`, `ups.ups_lane`, `pack.pack_lane`,
`srs.srs_bf_lane`, `bfloat16.bf16_mac_hw`) instead of re-implementing them, so
a misread of the reference can no longer corrupt the emulator and the golden
identically.

## Results by class

| Class | Oracle | Cases | Verdict |
|-------|--------|-------|---------|
| SRS | genuine `srs.srs_lane` | 32400 | **clean** (also proved the retired hand-port was faithful: byte-identical regen) |
| UPS | genuine `ups.ups_lane` | 2840 | **clean** |
| Pack | genuine `pack.pack_lane` | 1890 | **clean** (5 width pairs x signed x {Trunc,Sat,SymSat}) |
| BF16 conversion | genuine `srs.srs_bf_lane` | 5020 | **clean** (f32->bf16, 10 rounding modes, NaN/denorm/ties/overflow; 396/502 inputs mode-dependent) |
| MatMul (integer) | independent integer arithmetic | part of 1238 | **clean** (i8xi8, i16xi16, i32xi16; sign combos, subtract, Acc32 overflow-wrap) |
| MatMul (bf16) | genuine `bfloat16.bf16_mac_hw` | part of 1238 | **clean on finite+inf**; one documented NaN divergence (below) |
| Element-wise | plain arithmetic (pre-existing) | 720 | clean |

Discrimination was verified for each class (not vacuous): Pack via a deliberate
saturate->truncate collapse caught; BF16 via 396/502 mode-dependent inputs;
MatMul by independent arithmetic across 32k+ lanes and by surfacing the one real
divergence below.

## The one finding: bf16 NaN canonicalization (emulator is HW-correct)

The MatMul bf16 differential surfaced exactly one emulator-vs-model difference:
NaN payload. The aietools model (`bfloat16.py:fp32_make_nan`) emits NaN mantissa
`0x7F`; the emulator emits mantissa `1`, because **real NPU1 silicon produces
mantissa 1** (HW-verified, per the note in
`src/interpreter/execute/vector_float.rs:fp32_make_nan`). So the emulator is
HW-correct and the *model* is the outlier. The differential treats NaN-vs-NaN as
a match; all finite + inf results are bit-exact. This is not a fidelity gap (the
emulator matches hardware), so it is not added to `known-fidelity-gaps.md`.

## Classes with no python-model oracle

- **Convert** (`SemanticOp::Convert`): composite -- int-width pieces are covered
  by Pack/SRS/UPS; the bf16<->f32 piece by the BF16 class. No standalone oracle
  needed.
- **Shuffle** (`SemanticOp::Shuffle`): the python model has no shuffle function.
  `me_enums.h` provides the 48 mode names/indices (cross-checked: exact match to
  our `ShuffleMode` enum), but the per-byte routing exists nowhere in aietools --
  it is hardware-probed. Routing fidelity is therefore a Half-B silicon item, not
  a no-HW gap. See `2026-06-08-shuffle-oracle-enum-verified-routing-hw-gated.md`.

## Path-3 conclusion (intrinsic -> SemanticOp correctness)

The substantive question behind the (phantom) intrinsics comprehension gap was
not "is the `Intrinsic` catch-all handled" (it is never constructed) but "does
every intrinsic-backed instruction resolve to a *behaviourally correct*
SemanticOp?" Half A answers yes for the vector-compute semantics that the
intrinsics resolve to: `Srs`, `Ups`, `Pack`/`Unpack`, the bf16 conversion path,
and `Mac`/`MatMul` (integer + bf16) are all differentially clean against the
genuine model (modulo the HW-correct NaN payload). This substantiates closing
the comprehension gap via `Accept{rationale}` -- the remaining step (#104).

## What remains

- **Half B (HW-gated):** the silicon `Verified{evidence}` flip of the
  perishable-queue vector item, batched with the Phoenix-survival capture.
  Model-vs-reference agreement is fidelity-to-the-modeled-source, not silicon
  verification. Shuffle routing values also land here.
- **#104:** close the intrinsics comprehension gap with `Accept{rationale}`
  written against these confirmed-correct mappings, then retire the dead
  `SemanticOp::from_intrinsic()`.
