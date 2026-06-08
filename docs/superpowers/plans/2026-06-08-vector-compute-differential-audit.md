# Vector-Compute Differential Audit (Half A) — scoping plan

- Date: 2026-06-08. Status: plan for review (no code yet).
- Task: #103. Background: `docs/superpowers/findings/2026-06-08-intrinsic-comprehension-gap-is-phantom-variant.md`.

## Goal

Bit-exact differential of the emulator's AIE2 vector semantics against the
**genuine** aietools Python reference models (not our hand-port), across the full
discrete mode space, extended to every vector class. This audits both the
vector-op semantics **and** the intrinsic→SemanticOp mapping correctness (path 3
from the intrinsics examination). No HW, no sandbox gating.

This is **Half A** only. The silicon `Verified{evidence}` flip of the
perishable-queue vector item is **Half B** — HW-gated, batches with the
Phoenix-survival capture, out of scope here. A model-vs-reference match proves
fidelity-to-the-source we modeled from; it is *not* silicon verification (the
coverage spec is explicit), but it is the necessary first half and the only half
that finds *our* implementation bugs.

## Why now / what it buys

- Closes path-3: confirms every intrinsic resolves to a SemanticOp+mode that
  computes the *right* thing — the substantive question behind the (phantom)
  comprehension gap.
- **De-circularizes the existing golden harness.** Today `gen_vector_golden.py`
  *re-implements* `srs_round`/`srs_lane` in Python ("from srs.py:…"). A misread
  of the reference corrupts the emulator *and* the golden identically — the bug
  hides. Driving the real model as oracle removes that blind spot.

## Existing assets to build on (don't start cold)

- **Harness pattern (proven):** `src/interpreter/execute/vector_validate.rs` +
  `tools/golden/vector_ops.json` + `tools/gen_vector_golden.py`. Python emits
  golden `(input, mode, expected)` → Rust test asserts bit-exact. Currently
  covers SRS, UPS, element-wise (5 ops × 6 types).
- **Clean lane seams for isolated invocation:** `srs_lane`, `ups_lane_signed`,
  `pack_lane`, `shuffle_vectors`, `execute_matmul`, `run_binary_vec_op`.
- **The upgrade:** (a) swap the hand-port oracle for the real aietools model fns;
  (b) extend to the uncovered classes; (c) enumerate the full mode space.

## Oracle runnability — the linchpin (resolved, bounded)

- The models are **Python 2** (`long()`, `print` statements, py2 relative
  imports; confirmed they do not import under py3.13). Deps: **numpy only**
  (avoid the `pdg_wrapper` path — its `.so` is not in the tree; use the pure
  `model/*.py`).
- **Plan:** `2to3` a working **copy** of `model/` into an out-of-repo dir
  (`build/experiments/vector-oracle/`, gitignored). Drive the real
  `instruction_srs` / `instruction_mul` / `instruction_mac` / `instruction_ups`
  / `instruction_pack` etc. as the oracle.
- **Licensing hygiene:** aietools code stays **out of the repo** (transient
  oracle artifact, same posture as aiesim). Only derived **golden JSON** is
  committed — matching the existing `tools/golden/vector_ops.json` precedent.

## Op-class × mode-axis inventory (the scope)

Oracle fns in `aietools/.../python_model/model/`; mode axes from `constants.py`.

| Class | Oracle fn | Discrete mode axes | Emu seam | Status |
|-------|-----------|--------------------|----------|--------|
| SRS | `instruction_srs` | 10 rnd × {half,full}×{acc32,acc64} (4) × sat × symsat × signed × shift | `srs_lane` | partial → full |
| UPS | `instruction_ups` | 4 size/acc tuples × shift × sat × signed | `ups_lane_signed` | partial → full |
| Pack/Unpack | `instruction_pack`/`_unpack` | width pairs (32→16/8, 16→8) × {Trunc,Sat,SymSat} × signed | `pack_lane` | **new** |
| MatMul/MAC | `instruction_mul`/`_mac`/`vmac` | 8 mult modes / dense+sparse geom × acc32/64 × x/y sign × subtract × sparse × bf16 | `execute_matmul` | **new** |
| Permute/Shuffle | (perm modes) | 48 shuffle modes + 26 MAC permute modes | `shuffle_vectors` | **new** |
| Convert | (in srs/helpers) | from/to type × shift × wide/narrow × accum source | `VectorAlu::execute` | **new** |
| Element-wise (vadd) | `vadd` model | 31 modes × element types | `run_binary_vec_op` | partial → broaden |
| BF16 / float | `instruction_srs_bf`, bf16 utils | rnd modes × denorm/NaN/inf edge classes | `vector_float` seams | **new** (prior analysis scripts in `tests/bf16_*`) |

## Harness shape

1. **Generator** (python, out-of-repo oracle on `PYTHONPATH`): for each class,
   enumerate *all* discrete mode combos; per combo sample K input vectors
   including edge cases (0, min, max, overflow-triggering, denorm/NaN/inf for
   float); call the real model fn; emit golden JSON.
2. **Rust assertions:** extend `vector_validate.rs` to load + assert bit-exact
   for every class via the lane seams.
3. **Mismatch → emulator bug → fix TDD** (the audit's payload).

## Mode-space strategy (combinatorics)

Discrete axes are **small per class** (SRS ≈ 10×4×2×2×2; matmul ≈ 8 modes ×
acc × sign² × subtract × sparse × bf16). Strategy: **enumerate all discrete mode
combos, sample K inputs each** — tractable, not a cartesian explosion of inputs.
Edge inputs are explicit, not just random.

**Intrinsic cross-walk:** map each AIE2 intrinsic → its `(SemanticOp, mode)` so
the differential's mode coverage *demonstrably spans the intrinsic set* — this is
what lets us state path-3 closed with evidence, not assertion.

## Phasing within Half A

0. **Spike (de-risk the linchpin):** 2to3 a `model/` copy; run `instruction_srs`
   under py3; reproduce the *existing* SRS golden from the **real** model. Should
   agree (the hand-port was faithful) — any disagreement is itself finding #1.
1. Extend class-by-class: SRS/UPS full → Pack → Convert → Permute → MatMul →
   BF16. TDD each; fix mismatches as they surface.
2. Build the intrinsic→(SemanticOp,mode) cross-walk; confirm coverage spans it.
3. Roll-up: which classes are differentially-clean (model-vs-reference). Feeds
   Half B (silicon) and the path-3 audit conclusion; then unblocks #104.

## Risks

- **Oracle py2→py3** — bounded (2to3 mechanical, numpy-only); Step 0 spike proves
  it before any class work.
- **Combinatorial blow-up** — mitigated: enumerate discrete modes, *sample* inputs.
- **`pdg_wrapper` .so absent** — use pure `model/*.py` only.
- **Licensing** — oracle stays out-of-repo; only golden data committed.
