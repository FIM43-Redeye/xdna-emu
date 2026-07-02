# SP-5b R3b-PC (Enriched Geometry) Implementation Plan

> **RESOLVED (2026-07-01): Option 1 — this enriched-geometry plan is superseded.**
> During execution, Task 3 hit a real identifiability limit: the R3b two-flood
> interval cannot separate within-axis directions (`d_hE` vs `d_hW`, `d_vN` vs
> `d_vS`) at any two-source placement (cross-axis `d_h` vs `d_v` IS identifiable).
> The 5-param enrichment (Task 1) is abandoned; R3b fits `{d_h, d_v}`, and vertical
> within-axis anisotropy is reallocated to a two-sided R1 spine. The realized work
> lives in the rollback (commit `77180706`) — see the repo-root `NEXT-STEPS.md`
> Sec.3 and the finding
> `docs/superpowers/findings/2026-07-01-r3b-two-source-identifiability-limit.md`.
> Tasks 3-6 (kernel/host/gate) still apply for the `{d_h, d_v}` instrument; only
> the enriched-geometry solver premise is dropped. This plan is retained for
> history; do not execute Task 1 as written.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the R3b-PC skew instrument (SP-5b Phase 2) with the *enriched
geometry* the soundness audit requires, so it can falsify all four structural
assumptions of the 4-knob broadcast-timing model instead of just one.

**Architecture:** Software-first. The load-bearing change is the extractor +
observation bridge: fit *per-direction* hop coefficients (`d_hE, d_hW, d_vN,
d_vS`) plus a turn/interaction term, so direction anisotropy and cross-axis
non-additivity surface as identifiable parameters (or fail-loud rank deficiency)
rather than averaging invisibly into two scalars. The kernel/host/gate are then
authored to *produce* geometry that populates those columns (two sources at
opposite corners, two-sided tile spans per axis, an off-axis diagonal tile, and a
channel-uniformity control pair). Everything through Task 3 is executable and
verifiable now with zero hardware; Tasks 4-6 are authored now but their silicon
gate is deferred to SP-5c.

**Tech Stack:** Python 3.13 + numpy (extractor/bridge, `pytest`), hand-authored
MLIR (`aiecc.py` via mlir-aie), bash (gate), the `bridge-trace-runner` C++ tool.

## Global Constraints

- **Derive from the toolchain; hardcode nothing extractable.** Register offsets,
  event IDs, and target names come from the AM025 DB / mlir-aie, cited per use.
- **SP-5b produces NO number and flips NO flag.** Every gate asserts shape,
  reproducibility (range-0), and non-degeneracy — **never a value**. Numbers are
  SP-5c.
- **Governing spec:** `docs/superpowers/specs/2026-06-30-sp5b-kernel-hw-bringup-design.md`
  (rev3) — Sec.5 is R3b-PC, Sec.13 is the audit corrections. Parent:
  `docs/superpowers/specs/2026-06-30-sp5b-measurement-apparatus-design.md` (rev3).
  Audit: `docs/superpowers/findings/2026-07-01-sp5b-soundness-audit.md`.
- **Verified anchors (use verbatim):** target `npu1_3col` (NOT `npu1_4col`,
  which does not exist); `Performance_Control0` @ `0x31500` (`Cnt0_Start_Event`
  bits 6:0, `Cnt0_Stop_Event` bits 14:8); `Performance_Counter0` @ `0x31520`;
  `Timer_Control` @ `0x34000` (NOT written on measured tiles in R3b-PC);
  `BROADCAST_15` = 122 (core/mem); `USER_EVENT_2` = 126 (core/mem), shim
  `USER_EVENT_0` = 126 / `USER_EVENT_1` = 127 (no shim `USER_EVENT_2`).
- **HW safety (Tasks 4-6, when they eventually run on Phoenix in SP-5c):** `env -u
  XDNA_EMU`; never two HW suites at once; no `xrt-smi` during a HW run; `pkexec`
  not `sudo`; reboot handed to the user; rebuild the FFI `.so` before plugin use;
  never pipe long-running commands through `tail`/`grep`.
- **Python import root:** tests import `from calibration.skew.<mod>` (see existing
  `tools/test_skew_*.py`); run with `cd tools && python3 -m pytest`.
- **Commit trailer** ends with `Generated using Claude Code.` +
  `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`. No emoji.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tools/calibration/skew/r3b_extract.py` | **Modified.** Fit per-direction hop coeffs + turn; expose anisotropy diagnostics; fail loud when geometry can't identify a requested column | 1 |
| `tools/test_skew_r3b_extract.py` | **Rewritten** for the enriched schema (two-sided recovery, anisotropy/turn falsification, rank-deficient fail-loud) | 1 |
| `tools/calibration/skew/r3b_observe.py` | **New.** Read control-packet readback buffer → per-tile design-row coefficients via `geometry.json`; fail loud on malformed/short buffer | 2 |
| `tools/test_skew_r3b_observe.py` | **New.** Frozen readback-buffer + geometry fixture → expected coefficient rows; fail-loud paths | 2 |
| `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/geometry.json` | **New.** Enriched geometry: two sources (opposite corners), two-sided per axis, diagonal tile, channel-control pair | 3 |
| `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/aie.mlir` | **New.** Hand-authored kernel: two floods on distinct channels, perf-counter config per measured tile, runtime-seq ordering | 4 |
| `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/run.cpp` (or host per repo convention) | **New.** Control-packet register-read of `Performance_Counter0`; bind/dump readback BO | 5 |
| `build/experiments/sp5-skew/r3b_pc_gate.sh` | **New.** N serial runs + arrival-jitter pre-check + channel-uniformity residual + range-0 b-vector + non-inversion. HW gate = SP-5c | 6 |

**Executability boundary:** Tasks 1-3 are fully executable and verifiable now
(pure software/data). Tasks 4-6 are authored now against exact references; their
*silicon* verification is SP-5c. Task 4/5 get an emulator smoke-run
(compiles + runs, structure only) as their now-verification.

---

## The enriched model (read before Task 1)

Each measured tile X yields the two-flood interval `r_X = D(s2,X) - D(s1,X) +
(T0_2 - T0_1)`, where `D(s,X)` is the broadcast path cost from source `s` to X.
Decomposing each path into per-direction hop counts, the interval is linear:

```
r_X = c + a_hE(X)*d_hE + a_hW(X)*d_hW + a_vN(X)*d_vN + a_vS(X)*d_vS + a_turn(X)*d_turn
```

- `d_hE, d_hW` = per-hop cost East / West; `d_vN, d_vS` = North / South.
- `a_*(X)` = the *signed* net coefficient the interval accumulates in that
  direction (e.g. if `D(s1,X)` travels East and `D(s2,X)` travels West, the East
  hops enter with a minus sign). These are **precomputed by the observation
  bridge** (Task 2) from the two source positions and X's position in
  `geometry.json`; the extractor is geometry-agnostic and just fits.
- `d_turn` with coefficient `a_turn = (net east hops)*(net north hops)` (or the
  agreed interaction count) is **zero on every axis-collinear tile** and non-zero
  only for an off-axis diagonal tile — so it is identifiable only when the
  geometry includes a diagonal.
- Isotropy is the hypothesis `d_hE == d_hW` and `d_vN == d_vS`; additivity is
  `d_turn == 0`. The enriched extractor *fits these as free parameters*, so SP-5c
  can test them; SP-5b only proves the apparatus recovers whatever is injected and
  fails loud when the geometry cannot identify a requested column.

Consequence: a **one-sided** geometry (all tiles same side of the sole source)
leaves `d_hW`/`d_vS` coefficients all-zero → rank-deficient on those columns →
**fail loud**. That is correct: one-sided geometry genuinely cannot see
anisotropy. This is why the existing one-sided tests are rewritten, not reformatted.

---

## Task 1: Enriched R3b extractor

**Files:**
- Modify: `tools/calibration/skew/r3b_extract.py`
- Test: `tools/test_skew_r3b_extract.py` (rewrite)

**Interfaces:**
- Consumes: `solve_design_matrix(A, b, min_rank)` and `RankDeficientError` from
  `tools/calibration/skew/_solve.py` (unchanged).
- Produces: `extract_r3b(observations, reference=0, params=("d_hE","d_hW","d_vN","d_vS","d_turn"))
  -> {"d_hE","d_hW","d_vN","d_vS","d_turn","d_h","d_v","aniso_h","aniso_v","fit_residual"}`
  where each observation is
  `{"a_hE","a_hW","a_vN","a_vS","a_turn": float, "r": float}`. Convenience
  outputs: `d_h = (d_hE + d_hW)/2`, `d_v = (d_vN + d_vS)/2`,
  `aniso_h = d_hE - d_hW`, `aniso_v = d_vN - d_vS`. `params` lets a caller
  request a reduced column set (e.g. drop `d_turn` when no diagonal tile);
  columns not in `params` are omitted from the design matrix and the returned
  dict. `min_rank = len(params)`; rank deficiency raises `RankDeficientError`.

- [ ] **Step 1: Write the failing tests** (`tools/test_skew_r3b_extract.py`, full rewrite)

```python
"""R3b enriched interval-difference skew extraction tests (#140 SP-5b, rev3).

Enriched model (per kernel spec rev3 Sec.5.1/5.2): the interval decomposes into
signed per-direction hop coefficients so anisotropy (d_hE!=d_hW, d_vN!=d_vS) and
a turn term (d_turn) are identifiable — the falsification apparatus the soundness
audit requires.
"""
import math
import pytest
from calibration.skew.r3b_extract import extract_r3b
from calibration.skew._solve import RankDeficientError

PARAMS5 = ("d_hE", "d_hW", "d_vN", "d_vS", "d_turn")
PARAMS4 = ("d_hE", "d_hW", "d_vN", "d_vS")


def _obs(a_hE, a_hW, a_vN, a_vS, a_turn, truth, const):
    """Synthesize a reading from coefficients and a truth param dict."""
    r = const + (a_hE * truth["d_hE"] + a_hW * truth["d_hW"]
                 + a_vN * truth["d_vN"] + a_vS * truth["d_vS"]
                 + a_turn * truth["d_turn"])
    return {"a_hE": a_hE, "a_hW": a_hW, "a_vN": a_vN, "a_vS": a_vS,
            "a_turn": a_turn, "r": r}


def _two_sided_isotropic_geometry(truth, const):
    """Ref + 2 East + 2 West + 2 North + 2 South + 1 diagonal. Populates all
    five columns with rank 5 after ref-differencing."""
    return [
        _obs(0, 0, 0, 0, 0, truth, const),   # reference
        _obs(1, 0, 0, 0, 0, truth, const),   # East 1
        _obs(2, 0, 0, 0, 0, truth, const),   # East 2
        _obs(0, 1, 0, 0, 0, truth, const),   # West 1
        _obs(0, 2, 0, 0, 0, truth, const),   # West 2
        _obs(0, 0, 1, 0, 0, truth, const),   # North 1
        _obs(0, 0, 2, 0, 0, truth, const),   # North 2
        _obs(0, 0, 0, 1, 0, truth, const),   # South 1
        _obs(0, 0, 0, 2, 0, truth, const),   # South 2
        _obs(1, 0, 1, 0, 1, truth, const),   # diagonal (E1,N1, turn=1)
    ]


def test_recovers_all_five_params():
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = _two_sided_isotropic_geometry(truth, const=100.0)
    r = extract_r3b(obs, params=PARAMS5)
    for k in PARAMS5:
        assert math.isclose(r[k], truth[k], abs_tol=1e-6), k
    assert math.isclose(r["d_h"], 2.0, abs_tol=1e-6)
    assert math.isclose(r["d_v"], 3.0, abs_tol=1e-6)
    assert math.isclose(r["aniso_h"], 0.0, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_const_cancels():
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    ra = extract_r3b(_two_sided_isotropic_geometry(truth, const=100.0), params=PARAMS5)
    rb = extract_r3b(_two_sided_isotropic_geometry(truth, const=999.0), params=PARAMS5)
    for k in PARAMS5:
        assert math.isclose(ra[k], rb[k], abs_tol=1e-6), k


def test_exposes_horizontal_anisotropy():
    # Truth has d_hE != d_hW. The enriched fit RECOVERS the split (aniso_h != 0)
    # at zero residual — the apparatus can SEE anisotropy the old scalar hid.
    truth = {"d_hE": 2.0, "d_hW": 5.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    r = extract_r3b(_two_sided_isotropic_geometry(truth, const=0.0), params=PARAMS5)
    assert math.isclose(r["aniso_h"], -3.0, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_reduced_isotropic_fit_residual_grows_under_anisotropy():
    # If a caller fits the ASSUMED isotropic-additive shape (collapse E/W and N/S,
    # drop turn) against an anisotropic truth, the residual fires — the
    # falsification the audit requires.
    truth = {"d_hE": 2.0, "d_hW": 5.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = _two_sided_isotropic_geometry(truth, const=0.0)
    # Collapse to an isotropic design: a_h = a_hE - a_hW, a_v = a_vN - a_vS.
    iso = [{"a_hE": o["a_hE"] - o["a_hW"], "a_hW": 0.0,
            "a_vN": o["a_vN"] - o["a_vS"], "a_vS": 0.0, "a_turn": 0.0,
            "r": o["r"]} for o in obs]
    r = extract_r3b(iso, params=("d_hE", "d_vN"))
    assert r["fit_residual"] > 1.0


def test_exposes_turn_term():
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 7.0}
    r = extract_r3b(_two_sided_isotropic_geometry(truth, const=0.0), params=PARAMS5)
    assert math.isclose(r["d_turn"], 7.0, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_one_sided_geometry_fails_loud():
    # All tiles East/North of the sole source -> d_hW, d_vS columns all-zero ->
    # rank-deficient -> fail loud. One-sided geometry cannot see anisotropy.
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = [_obs(0, 0, 0, 0, 0, truth, 0.0),
           _obs(1, 0, 0, 0, 0, truth, 0.0),
           _obs(2, 0, 0, 0, 0, truth, 0.0),
           _obs(0, 0, 1, 0, 0, truth, 0.0),
           _obs(0, 0, 2, 0, 0, truth, 0.0)]
    with pytest.raises(RankDeficientError):
        extract_r3b(obs, params=PARAMS5)


def test_no_diagonal_cannot_request_turn():
    # Axis-collinear only -> a_turn all-zero -> requesting d_turn fails loud.
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    obs = _two_sided_isotropic_geometry(truth, const=0.0)[:-1]  # drop diagonal
    with pytest.raises(RankDeficientError):
        extract_r3b(obs, params=PARAMS5)
    # But the 4-param fit (no turn) succeeds on the same geometry.
    r = extract_r3b(obs, params=PARAMS4)
    assert r["fit_residual"] < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_skew_r3b_extract.py -v`
Expected: FAIL (extract_r3b does not accept `params=` / the coefficient schema).

- [ ] **Step 3: Rewrite the implementation** (`tools/calibration/skew/r3b_extract.py`)

```python
"""R3b enriched interval-difference skew extraction (#140 SP-5b, rev3).

Fits per-direction hop coefficients (d_hE, d_hW, d_vN, d_vS) plus a turn term
(d_turn) from per-tile two-flood interval readings, so direction anisotropy and
cross-axis non-additivity are identifiable — the falsification apparatus the
soundness audit (2026-07-01) requires. See kernel spec rev3 Sec.5.1/5.2.

The const (= T0_2 - T0_1) is removed by differencing against a reference tile.
Each observation carries the SIGNED design-row coefficients the interval
accumulates per direction; the observation bridge (r3b_observe) computes them
from geometry.json, so this extractor is geometry-agnostic.
"""
from ._solve import solve_design_matrix

ALL_PARAMS = ("d_hE", "d_hW", "d_vN", "d_vS", "d_turn")
_COEF = {"d_hE": "a_hE", "d_hW": "a_hW", "d_vN": "a_vN", "d_vS": "a_vS",
         "d_turn": "a_turn"}


def extract_r3b(observations, reference=0, params=ALL_PARAMS):
    """observations: list of {a_hE, a_hW, a_vN, a_vS, a_turn, r}.
    reference: index differenced against to drop const.
    params: which per-direction columns to identify (subset of ALL_PARAMS,
        in ALL_PARAMS order). min_rank = len(params); a column the geometry
        leaves all-zero makes the matrix rank-deficient -> RankDeficientError.
    Returns {each requested param, plus d_h, d_v, aniso_h, aniso_v, fit_residual}."""
    for p in params:
        if p not in ALL_PARAMS:
            raise ValueError(f"unknown param {p!r}")
    cols = [_COEF[p] for p in params]
    ref = observations[reference]
    A, b = [], []
    for i, o in enumerate(observations):
        if i == reference:
            continue
        A.append([float(o[c] - ref[c]) for c in cols])
        b.append(float(o["r"] - ref["r"]))
    x, resid = solve_design_matrix(A, b, min_rank=len(params))
    out = {p: float(xi) for p, xi in zip(params, x)}
    out["fit_residual"] = resid
    # Convenience aggregates (present only when both halves were fit).
    if "d_hE" in out and "d_hW" in out:
        out["d_h"] = 0.5 * (out["d_hE"] + out["d_hW"])
        out["aniso_h"] = out["d_hE"] - out["d_hW"]
    if "d_vN" in out and "d_vS" in out:
        out["d_v"] = 0.5 * (out["d_vN"] + out["d_vS"])
        out["aniso_v"] = out["d_vN"] - out["d_vS"]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_skew_r3b_extract.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/r3b_extract.py tools/test_skew_r3b_extract.py
git commit -m "feat(#140): enriched R3b extractor -- per-direction hop coeffs + turn

Fits d_hE/d_hW/d_vN/d_vS + d_turn so anisotropy and cross-axis non-additivity
are identifiable (or fail loud on one-sided/no-diagonal geometry), per the
SP-5b soundness audit. Replaces the single-scalar d_h/d_v fit that hid them.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 2: R3b observation bridge (`r3b_observe.py`)

**Files:**
- Create: `tools/calibration/skew/r3b_observe.py`
- Test: `tools/test_skew_r3b_observe.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime; produces observations shaped
  for `extract_r3b` (Task 1): dicts with `a_hE, a_hW, a_vN, a_vS, a_turn, r`.
- Produces: `observe_r3b(readback_bytes: bytes, geometry: dict) -> list[dict]`.
  `geometry` is the parsed `geometry.json` (Task 3 schema): `{"sources":
  {"s1": {col,row}, "s2": {col,row}}, "tiles": [{col, row, counter_index}...]}`.
  For each tile, read its `Performance_Counter0` value `r` from
  `readback_bytes` at `counter_index*4` (little-endian u32), compute the signed
  per-direction coefficients of `D(s2,X) - D(s1,X)`, and emit one observation.
  Fail loud (`ValueError`) on short buffer or a tile whose broadcast path is
  ambiguous (non-monotone). Coefficient computation, `hops(src, tile)` returning
  `(east, west, north, south)` monotone hop counts assuming rectilinear
  min-cost broadcast:
  - `east = max(tile.col - src.col, 0)`, `west = max(src.col - tile.col, 0)`,
    `north = max(tile.row - src.row, 0)`, `south = max(src.row - tile.row, 0)`.
  - `a_hE = e2 - e1`, `a_hW = w2 - w1`, `a_vN = n2 - n1`, `a_vS = s2 - s1`
    where `(e1,w1,n1,s1) = hops(s1, X)` and `(e2,w2,n2,s2) = hops(s2, X)`.
  - `a_turn = (e2+w2)*(n2+s2) - (e1+w1)*(n1+s1)` (net Manhattan turn interaction
    per source path, differenced).

- [ ] **Step 1: Write the failing test** (`tools/test_skew_r3b_observe.py`)

```python
"""R3b observation-bridge tests (#140 SP-5b, rev3): readback buffer + geometry
-> design-row coefficients, against a frozen fixture."""
import struct
import pytest
from calibration.skew.r3b_observe import observe_r3b

# s1 at (0,0), s2 at (2,4) [opposite corner of a 3-col x 5-row virtual frame].
GEOM = {
    "sources": {"s1": {"col": 0, "row": 0}, "s2": {"col": 2, "row": 4}},
    "tiles": [
        {"col": 1, "row": 2, "counter_index": 0},  # interior
        {"col": 2, "row": 2, "counter_index": 1},  # east edge
        {"col": 0, "row": 2, "counter_index": 2},  # west edge
    ],
}


def _buf(values):
    return b"".join(struct.pack("<I", v) for v in values)


def test_coefficients_match_hand_computation():
    obs = observe_r3b(_buf([1000, 1010, 1020]), GEOM)
    # tile (1,2): hops(s1)= (e1,w1,n1,s1)=(1,0,2,0); hops(s2)=(e2,w2,n2,s2)=(0,1,0,2)
    #   a_hE = 0-1 = -1, a_hW = 1-0 = 1, a_vN = 0-2 = -2, a_vS = 2-0 = 2
    #   a_turn = (0+1)*(0+2) - (1+0)*(2+0) = 2 - 2 = 0
    o0 = obs[0]
    assert (o0["a_hE"], o0["a_hW"], o0["a_vN"], o0["a_vS"], o0["a_turn"]) == (-1, 1, -2, 2, 0)
    assert o0["r"] == 1000.0


def test_short_buffer_fails_loud():
    with pytest.raises(ValueError):
        observe_r3b(_buf([1000, 1010]), GEOM)  # 3 tiles, 2 words


def test_reads_counter_by_index():
    obs = observe_r3b(_buf([7, 8, 9]), GEOM)
    assert [o["r"] for o in obs] == [7.0, 8.0, 9.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m pytest test_skew_r3b_observe.py -v`
Expected: FAIL with "No module named ... r3b_observe".

- [ ] **Step 3: Write the implementation** (`tools/calibration/skew/r3b_observe.py`)

```python
"""R3b observation bridge (#140 SP-5b, rev3): control-packet readback buffer +
geometry.json -> per-tile design-row coefficients for extract_r3b. Shared by the
PC and TM R3b kernels (same {a_*, r} contract). See kernel spec rev3 Sec.5.2."""
import struct


def _hops(src, tile):
    """Monotone rectilinear hop counts (east, west, north, south) from src to tile."""
    return (max(tile["col"] - src["col"], 0), max(src["col"] - tile["col"], 0),
            max(tile["row"] - src["row"], 0), max(src["row"] - tile["row"], 0))


def observe_r3b(readback_bytes, geometry):
    """readback_bytes: little-endian u32 Performance_Counter0 values, one per
    tile at counter_index*4. geometry: parsed geometry.json. Returns a list of
    {a_hE, a_hW, a_vN, a_vS, a_turn, r} for extract_r3b. Fails loud on short buffer."""
    s1 = geometry["sources"]["s1"]
    s2 = geometry["sources"]["s2"]
    tiles = geometry["tiles"]
    need = max(t["counter_index"] for t in tiles) + 1
    if len(readback_bytes) < need * 4:
        raise ValueError(f"readback buffer too short: {len(readback_bytes)} bytes, "
                         f"need >= {need * 4}")
    out = []
    for t in tiles:
        e1, w1, n1, s1h = _hops(s1, t)
        e2, w2, n2, s2h = _hops(s2, t)
        (r,) = struct.unpack_from("<I", readback_bytes, t["counter_index"] * 4)
        out.append({
            "a_hE": float(e2 - e1), "a_hW": float(w2 - w1),
            "a_vN": float(n2 - n1), "a_vS": float(s2h - s1h),
            "a_turn": float((e2 + w2) * (n2 + s2h) - (e1 + w1) * (n1 + s1h)),
            "r": float(r),
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m pytest test_skew_r3b_observe.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/r3b_observe.py tools/test_skew_r3b_observe.py
git commit -m "feat(#140): R3b observation bridge -- readback -> design-row coeffs

observe_r3b maps each measured tile's Performance_Counter0 readback + its
position relative to both flood sources into the signed per-direction
coefficients extract_r3b consumes. Fail-loud on short buffer.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 3: Enriched kernel geometry (`geometry.json`)

**Files:**
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/geometry.json`
- Test: fold into Task 2's bridge test by loading this file (add one assertion).

**Interfaces:**
- Consumes: the `observe_r3b` geometry schema (Task 2).
- Produces: the concrete geometry the kernel (Task 4) must realize. Two sources
  at opposite corners of an `npu1_3col` partition (cols 0..2, rows 0..5); measured
  tiles giving two-sided spans on both axes plus a diagonal and a channel-control
  tile. `counter_index` is the tile's slot in the readback buffer.

- [ ] **Step 1: Write the geometry file**

```json
{
  "target": "npu1_3col",
  "sources": {
    "s1": {"col": 0, "row": 0, "channel": 0, "generate_event": "USER_EVENT_0"},
    "s2": {"col": 2, "row": 5, "channel": 1, "generate_event": "USER_EVENT_1"}
  },
  "tiles": [
    {"col": 1, "row": 2, "kind": "core", "counter_index": 0, "role": "interior"},
    {"col": 0, "row": 2, "kind": "core", "counter_index": 1, "role": "west_edge"},
    {"col": 2, "row": 2, "kind": "core", "counter_index": 2, "role": "east_edge"},
    {"col": 1, "row": 1, "kind": "core", "counter_index": 3, "role": "south"},
    {"col": 1, "row": 3, "kind": "core", "counter_index": 4, "role": "north"},
    {"col": 1, "row": 4, "kind": "core", "counter_index": 5, "role": "north2"},
    {"col": 2, "row": 4, "kind": "core", "counter_index": 6, "role": "diagonal"},
    {"col": 1, "row": 2, "kind": "core", "counter_index": 7, "role": "channel_control",
     "note": "same tile as counter 0 but its counter is armed START=s1/STOP=s1 on the OTHER channel; the (index0 - index7) delta exposes d_h^{s1}!=d_h^{s2}. See Task 4/6."}
  ]
}
```

> **Rank note for the executor:** with `s1=(0,0)`, `s2=(2,5)`, the West/South
> coefficients are populated because interior/edge tiles sit between the two
> corners (e.g. tile `(0,2)` has `a_hE` from `s2`'s westward path). Before wiring
> the kernel, dry-run `observe_r3b(<zeros>, json.load(...))` then
> `extract_r3b(obs, params=("d_hE","d_hW","d_vN","d_vS","d_turn"))` on a synthetic
> reading built from a known truth — it must **not** raise `RankDeficientError`.
> If it does, the geometry is under-spanned; add a tile, do not weaken the solver.

- [ ] **Step 2: Verify the geometry is rank-sufficient** (add to `test_skew_r3b_observe.py`)

```python
import json, os
from calibration.skew.r3b_extract import extract_r3b

def test_shipped_geometry_is_rank_sufficient():
    path = os.path.join(os.path.dirname(__file__), os.pardir,
                        "..", "mlir-aie", "test", "npu-xrt", "sp5_skew_r3b_pc",
                        "geometry.json")
    geom = json.load(open(os.path.abspath(path)))
    # Drop the channel_control duplicate for the identifiability check.
    geom = {**geom, "tiles": [t for t in geom["tiles"] if t["role"] != "channel_control"]}
    truth = {"d_hE": 2.0, "d_hW": 2.0, "d_vN": 3.0, "d_vS": 3.0, "d_turn": 0.0}
    import struct
    # Build a synthetic readback from the truth via the bridge's own coefficients.
    zero = struct.pack("<%dI" % (max(t["counter_index"] for t in geom["tiles"]) + 1),
                       *([0] * (max(t["counter_index"] for t in geom["tiles"]) + 1)))
    from calibration.skew.r3b_observe import observe_r3b
    obs = observe_r3b(zero, geom)
    for o in obs:  # synthesize r from truth
        o["r"] = (o["a_hE"]*truth["d_hE"] + o["a_hW"]*truth["d_hW"]
                  + o["a_vN"]*truth["d_vN"] + o["a_vS"]*truth["d_vS"]
                  + o["a_turn"]*truth["d_turn"])
    r = extract_r3b(obs, params=("d_hE", "d_hW", "d_vN", "d_vS", "d_turn"))
    assert r["fit_residual"] < 1e-6
```

- [ ] **Step 3: Run the check**

Run: `cd tools && python3 -m pytest test_skew_r3b_observe.py::test_shipped_geometry_is_rank_sufficient -v`
Expected: PASS. If `RankDeficientError`, add a spanning tile to `geometry.json`
(the plan's geometry is designed to pass; this guards against edits).

- [ ] **Step 4: Commit**

```bash
git add mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/geometry.json tools/test_skew_r3b_observe.py
git commit -m "feat(#140): enriched R3b-PC kernel geometry + rank-sufficiency guard

Two sources at opposite corners; two-sided spans per axis + diagonal +
channel-control tile, so the extractor can identify all five params. A test
asserts the shipped geometry is rank-sufficient.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 4: R3b-PC kernel (hand-authored MLIR)

> **Authoring-against-reference task, not fill-in-code.** Register-level MLIR must
> be validated by the compiler, not pre-guessed. The plan fixes every value and
> the ordering; the executor writes the MLIR against the cited templates and
> verifies it **compiles + emulator-smoke-runs** now. The silicon gate is SP-5c.

**Files:**
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/aie.mlir`

**Interfaces:**
- Consumes: `geometry.json` (Task 3) — realizes exactly its sources/tiles.
- Produces: an xclbin + `insts.bin` that `bridge-trace-runner` can run; a
  readback BO the host (Task 5) dumps into the byte layout `observe_r3b` expects
  (counter_index order).

**Exact build content (from kernel spec rev3 Sec.5.1 + Global Constraints):**

- [ ] **Step 1: Two floods on distinct channels.** `s1` from corner `(0,0)`,
  channel 0, generate `USER_EVENT_0` (=126, shim space); `s2` from corner
  `(2,5)`, channel 1, generate `USER_EVENT_1` (=127, shim space). Each flood =
  `Event_Broadcast{N}` + `Event_Generate`, hand-authored `aiex.npu.write32` at
  the AM025-DB offsets. Template: `AIEInsertTraceFlows.cpp:672-723`. **No
  `Timer_Control.Reset_Event` on measured tiles** (perf counter is a separate HW
  unit; `0x31500` vs timer `0x34000`).

- [ ] **Step 2: Perf-counter config per measured tile.** For each tile in
  `geometry.json` (counter_index 0..6), `write32` `Performance_Control0` @
  `0x31500`: `Cnt0_Start_Event` (bits 6:0) = `s1`'s broadcast event
  (`BROADCAST_15`=122), `Cnt0_Stop_Event` (bits 14:8) = `s2`'s broadcast event.
  The channel-control tile (counter_index 7, same physical tile as 0) configures a
  **second** counter armed START=`s1`/STOP=`s1` on channel 1's broadcast event, so
  `r[0]-r[7]` isolates `d_h^{s1}` vs the blend (Task 6 reads it).

- [ ] **Step 3: Runtime-sequence ordering (load-bearing).** Emit in exactly this
  order: configure counters on **all** measured tiles → `Event_Generate(s1)` →
  `Event_Generate(s2)` → readback packets (Task 5). Counter config MUST precede
  `generate(s1)` or the start event is missed. Document the order in a comment.

- [ ] **Step 4: Compile.** Run (bare, not piped):
  `cd mlir-aie/test/npu-xrt/sp5_skew_r3b_pc && aiecc.py --aie-generate-xclbin
  --xclbin-name=aie.xclbin --aie-generate-npu-insts --npu-insts-name=insts.bin
  aie.mlir` (mirror the flags the existing `sp5_skew_r1` build uses; adjust to the
  repo's `Makefile`/`run.lit` convention).
  Expected: `aie.xclbin` + `insts.bin` produced, no error.

- [ ] **Step 5: Emulator smoke-run (structure only, no HW).** Rebuild the FFI:
  `cargo build -p xdna-emu-ffi`. Then run the compiled kernel through the emulator
  path and confirm it executes without panic and the two floods + counter writes
  appear (this checks structure, not values). Use the in-process runner or
  `XDNA_EMU=1` bridge-runner. Expected: clean run, non-empty readback region.

- [ ] **Step 6: Commit**

```bash
git add mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/aie.mlir
git commit -m "feat(#140): R3b-PC kernel -- two-flood + perf-counter (enriched geometry)

Hand-authored MLIR realizing geometry.json: floods s1/s2 on distinct channels
from opposite corners, Performance_Control0 START=s1/STOP=s2 per measured tile,
channel-control second counter, load-bearing config-before-generate ordering.
Compiles + emulator-smoke-runs; silicon gate is SP-5c.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 5: Control-packet readback host (the critical path)

> **Authoring-against-reference task.** `write32` is write-only; the counter is
> read via a control-packet register-read. This is the highest-effort R3b task
> (kernel spec rev3 Sec.5.1, Risk Sec.10).

**Files:**
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/run.cpp` (or the repo's host
  convention for this test dir — match `sp5_skew_r1`'s host).

**Interfaces:**
- Consumes: the compiled kernel (Task 4).
- Produces: a readback BO dumped to `--output` in `counter_index` order (u32 LE),
  exactly the byte layout `observe_r3b` (Task 2) parses.

- [ ] **Step 1: Emit a control-packet register-read** of `Performance_Counter0`
  (@ `0x31520`) on each measured tile (counter_index 0..7), using
  `aiex.npu.control_packet` read opcode (`AIEX.td:944`, lowered by
  `AIECtrlPacketToDma.cpp` — generic `$opcode` I32 attr, no named READ enum).
  Order the reads by `counter_index` so the dumped BO matches Task 2's layout.

- [ ] **Step 2: Host binds + dumps the readback BO.** Model the host on
  `sp5_skew_r1`'s host and the `bridge-trace-runner --output` path; the readback BO
  is an output-only BO — **size it from the compiled BD length** (the runner's
  `discover_arg_sizes_from_insts` fix, `65c3f852`, handles this; verify the BO is
  `8 * n_tiles` bytes, not 8).

- [ ] **Step 3: Compile + emulator smoke-run.** Same commands as Task 4 Steps 4-5.
  Expected: readback BO dumped, length == `4 * 8` bytes (8 counters), parseable by
  `observe_r3b` without `ValueError`.

- [ ] **Step 4: Fallback note (do not build unless Step 3 fails).** If the
  control-packet read proves unworkable, fall back to post-run core-`LDA 0x31520`
  → store → DMA out (heavier, core-program shape; kernel spec rev3 Sec.5.1). Record
  which path was used in a comment for the Phase-3 go/no-go.

- [ ] **Step 5: Commit**

```bash
git add mlir-aie/test/npu-xrt/sp5_skew_r3b_pc/run.cpp
git commit -m "feat(#140): R3b-PC control-packet readback host

Reads Performance_Counter0 per measured tile via control-packet register-read,
dumps a counter_index-ordered u32 BO matching observe_r3b's layout. Output-only
BO sized from the compiled BD length. Compiles + emulator-smoke-runs.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 6: HW runnability gate (`r3b_pc_gate.sh`)

> **Authored now; runs green only on Phoenix (SP-5c).** Mirror
> `build/experiments/sp5-skew/r1_gate.sh` structure and its dmesg TDR/IOMMU
> discipline (already backported, `2f193667`).

**Files:**
- Create: `build/experiments/sp5-skew/r3b_pc_gate.sh`

**Interfaces:**
- Consumes: the kernel (Task 4), host (Task 5), `observe_r3b` + `extract_r3b`.
- Produces: a per-run pass/fail on shape + reproducibility. **No value assertions.**

- [ ] **Step 1: Write the gate** (bash; N=20 serial runs). Include, mirroring
  `r1_gate.sh`: `tdr_count()` + `iommu_fault_count()` with per-run deltas (a
  non-zero delta = NOT clean); rc-0 + non-empty readback check; per-run
  `observe_r3b` → `extract_r3b(params=("d_hE","d_hW","d_vN","d_vS","d_turn"))`
  (rank-sufficient or the gate fails loud). Then the rev3-added checks:
  - **Range-0 b-vector:** the per-tile readback counter vector must be range-0
    (or within a stated tolerance) across the N runs.
  - **`s1`-before-`s2` non-inversion:** any zero/garbage counter = inversion, flagged.
  - **Cross-column arrival-jitter pre-check:** report the run-to-run range of the
    START=s1/STOP=s1 channel-control counter (counter_index 7); a range comparable
    to single-digit `d_h` is a red flag surfaced to the Phase-3 go/no-go.
  - **Channel-uniformity residual:** report `r[0] - r[7]`; non-zero = the `d_h`
    blend is real (not the ch15 hop cost).

- [ ] **Step 2: Dry-run the gate's Python glue offline** (no HW) by feeding a
  synthetic readback fixture through the `observe_r3b`/`extract_r3b` invocation the
  gate uses, confirming it parses and reports without a HW device. Expected: prints
  the five params + the two rev3 diagnostics, no exception.

- [ ] **Step 3: Commit** (force-add — `build/` is gitignored; r1_gate.sh precedent)

```bash
git add -f build/experiments/sp5-skew/r3b_pc_gate.sh
git commit -m "feat(#140): R3b-PC HW gate -- range-0 + arrival-jitter + channel-uniformity

N serial Phoenix runs (SP-5c): dmesg TDR/IOMMU deltas, rc-0, rank-sufficient
extract, range-0 b-vector, s1-before-s2 non-inversion, cross-column arrival-
jitter pre-check, channel-uniformity residual. No value assertions.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Self-Review

**Spec coverage (kernel spec rev3 Sec.5 + Sec.13 corrections):**
- Enriched geometry (two-sided + diagonal + channel-control) → Tasks 3, 4. ✓
- Signed N/S + E/W + interaction solver columns → Task 1. ✓
- `r3b_observe` bridge + frozen fixture → Task 2. ✓
- Immunity downgraded / arrival-jitter pre-check → Task 6. ✓
- Channel-uniformity control + residual → Tasks 3, 4, 6. ✓
- Perf-counter config + control-packet readback + ordering → Tasks 4, 5. ✓
- `npu1_3col` (not 4col) → Task 3 (`target`), Task 4. ✓
- No value assertions / range-0 → Task 6. ✓
- **Deferred to SP-5c (correctly out of this plan):** the `calibrated` flip and
  its pre-flip gates (held-out kernel, joint sign anchors, `dn_v`-Δwall,
  hard b-vector gate — kernel spec rev3 Sec.11); R3b-`LDA_TM` Phase-3 go/no-go.

**Placeholder scan:** Tasks 1-3 carry complete code. Tasks 4-6 are explicitly
authoring-against-reference (register-level MLIR/host must be compiler-validated,
not pre-guessed); every value, offset, event ID, and ordering step is fixed, and
each has a concrete compile + emulator-smoke verification. This is deliberate
fidelity, not a placeholder.

**Type consistency:** `extract_r3b(observations, reference, params)` returns keys
`{d_hE,d_hW,d_vN,d_vS,d_turn,d_h,d_v,aniso_h,aniso_v,fit_residual}`; `observe_r3b`
emits `{a_hE,a_hW,a_vN,a_vS,a_turn,r}` — the coefficient keys match the `_COEF`
map in Task 1. `geometry.json` keys (`sources.s1/s2`, `tiles[].counter_index`)
match `observe_r3b`'s reads. Consistent across Tasks 1-3.

**Known coupling for the executor:** Task 4's physical source-corner placement
must match Task 3's `geometry.json` sources exactly, because `observe_r3b`
computes coefficients from those positions. If HW constraints force a different
corner in SP-5c, update `geometry.json` and re-run Task 3's rank check — do not
patch the solver.
