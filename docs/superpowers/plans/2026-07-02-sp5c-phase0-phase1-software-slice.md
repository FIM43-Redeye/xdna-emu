# SP-5c Phase 0 + Phase 1 (Pure-Software Slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-software foundation of SP-5c -- the identifiability guards that encode the decoupled `d_h`/`d_v` capture split (so no future HW phase re-introduces the `d_v`-collapse defect), and the inert-but-complete `skew_constants.json -> emulator` ingestion wire plus the two §9a plumbing prerequisites -- all while the model stays behavior-neutral (`calibrated = false`).

**Architecture:** Phase 0 (Python, `tools/calibration/skew/` + `tools/`) adds a routing-aware coefficient model and executable identifiability proofs for the two decoupled R3b captures, plus a consolidating findings doc. Phase 1 (Rust, `crates/xdna-archspec/` + `src/device/state/`, and one Python glue point in `tools/inference/`) builds the build-time `skew_constants.json` ingestion with fail-loud guards, a §9a fixpoint-ch15 single-source test, and wires the sweep sidecar consumption. Nothing here flips `calibrated`; nothing here needs Phoenix silicon.

**Tech Stack:** Python 3.13 + numpy + pytest (calibration tooling); Rust + `serde_json` (archspec crate, build-time codegen); the emulator's existing broadcast-flood and inference-engine machinery.

## Global Constraints

- **Model stays inert:** every change in this plan keeps `calibrated = false` and all `BROADCAST_*` constants at 0. No behavior change to emulation. (Design Sec. 0, 5 Phase-1 gate.)
- **Derive from the toolchain; no hardcoding what can be extracted.** (`xdna-emu/CLAUDE.md`.)
- **No emoji anywhere.** Commit messages end with the two trailer lines:
  `Generated using Claude Code.` then
  `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`
- **Python calibration tests run from `tools/`:** `cd tools && python3 -m pytest test_skew_*.py -q` (config: `tools/conftest.py`; no repo-level pytest.ini).
- **Rust tests:** `cargo test --lib` (from repo root). Do NOT bare-`cargo build` expecting the FFI `.so` to relink -- irrelevant here (no `.so` use), but `cargo test --lib` is the gate.
- **Reference the design spec** at each task: `docs/superpowers/specs/2026-07-01-sp5c-skew-characterization-design.md` (rev4). The `d_v`-collapse algebra is Sec. 1 pt 2 / Sec. 2.
- **Commit often**, one commit per task minimum.

---

## File Structure

**Phase 0 (Python):**
- Modify: `tools/calibration/skew/r3b_observe.py` -- add reset-routed coefficient functions alongside the existing free-flood `_hops`.
- Modify: `tools/test_skew_r3b_identifiability.py` -- extend with decoupled-capture guards (the `d_v`-collapse proof, free-flood `d_v` identifiability, block-routed `d_h` identifiability).
- Create: `docs/superpowers/findings/2026-07-02-sp5c-phase0-identifiability.md` -- the consolidating Phase-0 proof record (references the guard tests; states IF-1/2/3 decisions, the determinism basis pointer, the magnitude/tolerance estimate).

**Phase 1 (Rust + one Python glue):**
- Modify: `tools/calibration/skew/schema.py` -- extend the handoff schema with the honest-provenance fields.
- Create: `crates/xdna-archspec/src/skew_ingest.rs` -- pure, unit-testable `BroadcastTiming`-from-JSON mapping + fail-loud validation.
- Modify: `crates/xdna-archspec/src/lib.rs` -- declare the new module.
- Modify: `crates/xdna-archspec/src/model_builder.rs:270-280` -- source `BroadcastTiming` from `skew_ingest` (committed JSON) instead of the hardcoded literal.
- Modify: `crates/xdna-archspec/build.rs` -- `cargo:rerun-if-changed` for the constants file.
- Create: `crates/xdna-archspec/data/skew_constants.json` -- committed uncalibrated placeholder (`calibrated: false`, null constants).
- Modify: `src/device/state/effects.rs` (test module `mod flood_source_capture_tests`, ~1263) -- add the §9a fixpoint-ch15 single-source test.
- Modify: `tools/inference/run_experiment.py:101-102` -- thread `model_path` into `run_engine`.
- Modify: `tools/trace-sweep.py` -- pass the produced `origin_d.json` path into the experiment run.

---

# PHASE 0 -- Theory lock-down (executable proofs)

### Task 0.1: Reset-routed coefficient model + the `d_v`-collapse guard

Encode, as code + a failing-then-passing test, WHY a single block-replicated R3b
capture cannot fit `d_v`: under the reset's shim-row-then-climb routing the
interval's vertical term is `(s2.row - s1.row)*d_v`, constant for every tile.

**Files:**
- Modify: `tools/calibration/skew/r3b_observe.py`
- Test: `tools/test_skew_r3b_identifiability.py`

**Interfaces:**
- Produces: `reset_routed_coeffs(s1, s2, tile) -> (dn_h, dn_v)` in `r3b_observe.py`,
  where `s1`/`s2`/`tile` are dicts with `"col"`/`"row"` keys. `dn_h =
  abs(s2["col"] - tile["col"]) - abs(s1["col"] - tile["col"])`; `dn_v = s2["row"] -
  s1["row"]` (tile-independent, by construction).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Add to `tools/test_skew_r3b_identifiability.py` (imports: add
`from calibration.skew.r3b_observe import reset_routed_coeffs` at top; the existing
file imports only numpy, so also confirm `import numpy as np` is present):

```python
def test_reset_routed_vertical_term_is_constant_dv_collapses():
    """Under reset routing (shim-row horizontal, then column climb), both floods
    share each tile's vertical climb, so the interval's vertical coefficient is
    (s2.row - s1.row) for EVERY tile -- a constant. d_v is therefore unidentifiable
    from a block-replicated capture (design Sec.1 pt2)."""
    s1 = {"col": 0, "row": 0}
    s2 = {"col": 2, "row": 5}
    tiles = [{"col": 1, "row": r} for r in (2, 3, 4, 5)]  # the bring-up vertical spine
    dn_v = [reset_routed_coeffs(s1, s2, t)[1] for t in tiles]
    assert dn_v == [5, 5, 5, 5], dn_v  # constant -> zero signal after referencing
    # and horizontal DOES vary across columns, so d_h is still identifiable
    dn_h_row3 = [reset_routed_coeffs(s1, s2, {"col": c, "row": 3})[0] for c in (0, 1, 2)]
    assert dn_h_row3 == [2, 0, -2], dn_h_row3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m pytest test_skew_r3b_identifiability.py::test_reset_routed_vertical_term_is_constant_dv_collapses -v`
Expected: FAIL with `ImportError` / `cannot import name 'reset_routed_coeffs'`.

- [ ] **Step 3: Write minimal implementation**

Add to `tools/calibration/skew/r3b_observe.py` (below the existing `_hops`, keeping
`_hops` untouched):

```python
def reset_routed_coeffs(s1, s2, tile):
    """Interval coefficients (dn_h, dn_v) under the AIE2 timer-reset routing:
    horizontal only on the shim row, then a vertical climb up the tile's column
    (design Sec.1). Both floods share the tile's climb, so the vertical term is
    the source-row difference -- constant across tiles, i.e. d_v is NOT identifiable
    here. Horizontal is the shim-row distance difference, so d_h IS identifiable.
    s1/s2/tile are dicts with 'col'/'row'."""
    dn_h = abs(s2["col"] - tile["col"]) - abs(s1["col"] - tile["col"])
    dn_v = s2["row"] - s1["row"]
    return (float(dn_h), float(dn_v))
```

Note: the test compares against ints (`[5,5,5,5]`, `[2,0,-2]`); `float(5) == 5` and
`float(2) == 2` are True in Python, so the assertions pass with float returns.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m pytest test_skew_r3b_identifiability.py::test_reset_routed_vertical_term_is_constant_dv_collapses -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/r3b_observe.py tools/test_skew_r3b_identifiability.py
git commit -m "feat(#140): SP-5c Phase 0 -- reset-routed coeffs encode the d_v-collapse

Under the timer-reset's shim-row-then-climb routing, the two-flood interval's
vertical coefficient is (s2.row-s1.row), constant for every tile -- so a
block-replicated capture cannot identify d_v (design rev4 Sec.1 pt2). Guard test
locks this in.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 0.2: Decoupled-capture identifiability guards

Prove, executably, that the fix works: the block-routed capture is rank-deficient
for a `{d_h,d_v}` fit (must be `d_h`-only), and a free-flood straddling capture
recovers `d_v` with spine leverage.

**Files:**
- Test: `tools/test_skew_r3b_identifiability.py`

**Interfaces:**
- Consumes: `reset_routed_coeffs` (Task 0.1); the existing module-level `_hops(src,
  tile)` (tuple form, lines 13-16) for the free-flood rows.
- Produces: nothing (test-only).

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_skew_r3b_identifiability.py`:

```python
def test_block_routed_capture_is_rank_deficient_for_dh_dv():
    """A single block-replicated capture: the [dn_h, dn_v] design matrix has a
    constant dn_v column, so after referencing the dn_v column is all-zero and the
    rank is 1 -- a {d_h,d_v} fit is rank-deficient. This is WHY the captures are
    decoupled (design Sec.2)."""
    s1 = {"col": 0, "row": 0}
    s2 = {"col": 2, "row": 5}
    tiles = ([{"col": c, "row": 3} for c in range(3)] +
             [{"col": 1, "row": r} for r in (2, 4, 5)])
    A = np.array([reset_routed_coeffs(s1, s2, t) for t in tiles], dtype=float)
    D = A[1:] - A[0]  # reference-difference, same convention as the rank-2 guard
    assert np.linalg.matrix_rank(D) == 1, np.linalg.matrix_rank(D)

def test_free_flood_straddling_capture_identifies_dv():
    """A free-flood capture with sources straddling the measured tiles vertically
    (s1 below, s2 above) recovers d_v with >=3 collinear-spine leverage: the
    [dn_h, dn_v] design matrix is rank 2 AND the dn_v column varies."""
    s1, s2 = (1, 0), (1, 5)  # same column, straddling -> pure vertical signal
    tiles = [(1, r) for r in (1, 2, 3, 4)]
    rows = []
    for c, r in tiles:
        e1, w1, n1, s1h = _hops(s1, (c, r))
        e2, w2, n2, s2h = _hops(s2, (c, r))
        rows.append([(e2 + w2) - (e1 + w1), (n2 + s2h) - (n1 + s1h)])
    A = np.array(rows, dtype=float)
    D = A[1:] - A[0]
    # dn_v varies across the spine (not constant) -> d_v is identifiable
    assert not np.allclose(A[:, 1], A[0, 1]), A[:, 1]
    # rank of the referenced vertical column is 1 (single axis exercised, full signal)
    assert np.linalg.matrix_rank(D) == 1 and not np.allclose(D[:, 1], 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_skew_r3b_identifiability.py -k "block_routed or free_flood_straddling" -v`
Expected: the block-routed test may already import-resolve (uses Task 0.1's
function); if Task 0.1 is complete it will PASS immediately -- that is acceptable
(these are guards, not red-green of new production code). If `reset_routed_coeffs`
is missing, FAIL with ImportError.

- [ ] **Step 3: (No production code)**

These are pure guard tests over existing helpers. No implementation step.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_skew_r3b_identifiability.py -v`
Expected: all tests PASS (the three new ones plus the three pre-existing rank-2
guards).

- [ ] **Step 5: Commit**

```bash
git add tools/test_skew_r3b_identifiability.py
git commit -m "test(#140): SP-5c Phase 0 -- decoupled-capture identifiability guards

Block-routed capture is rank-deficient for {d_h,d_v} (dn_v column constant);
free-flood straddling capture identifies d_v with spine leverage. Encodes the
rev4 decoupling so no HW phase can re-introduce the one-capture d_v-collapse.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 0.3: Phase-0 identifiability findings doc

Consolidate the Phase-0 "written proof" gate: the decoupling decision, IF-1/2/3
resolutions, the determinism-basis pointer, and the magnitude/tolerance estimate.
This is the doc the design's Phase-0 gate ("rank proof per column + per capture;
determinism basis + IF decisions recorded") points at.

**Files:**
- Create: `docs/superpowers/findings/2026-07-02-sp5c-phase0-identifiability.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Write the findings doc**

Create `docs/superpowers/findings/2026-07-02-sp5c-phase0-identifiability.md` with:

```markdown
# SP-5c Phase 0: Identifiability Proofs + IF Decisions (#140)

**Date:** 2026-07-02
**Status:** Phase-0 theory lock-down for SP-5c. Pure software; no `calibrated` flip.

## Decoupled captures (the load-bearing decision)
- `d_h` <- block-replicated / shim-row capture, scoped `d_h`-ONLY.
- `d_v` <- free-flood R3b (straddling sources) + R1.
- Proof (executable): `tools/test_skew_r3b_identifiability.py`
  - `test_reset_routed_vertical_term_is_constant_dv_collapses`
  - `test_block_routed_capture_is_rank_deficient_for_dh_dv`
  - `test_free_flood_straddling_capture_identifies_dv`
- Algebra: reset routing gives interval vertical coefficient `(s2.row - s1.row)`,
  constant across tiles -> `d_v` unidentifiable from a block capture. Horizontal
  coefficient `|s2.col - t.col| - |s1.col - t.col|` varies -> `d_h` identifiable.

## IF decisions
- IF-1 (path fidelity): `d_h` MUST be measured on the shim-row path (block
  replication or shim-row tiles); bring-up `d_h~=4` was the AIE-row path.
- IF-2 (linearity): tested by enriched geometry residuals in Phase 2 (accessible
  cols only; power ceiling acknowledged).
- IF-3 (arrival->latch uniformity): uniform -> cancels; per-kind -> folds into the
  gauge intra-offset. Disclosed assumption.

## Determinism basis
XDNA is globally clocked (single clock domain); broadcast transport deterministic;
async only at NoC egress (`grounding.py:is_async_cdc`). Range-0 is predicted.
Full argument: design Sec.1.5.

## Magnitude / tolerance estimate (not a kill gate)
Per-hop skew is single-digit cycles (bring-up `d_h~=4`, `d_v~=2`; mlir-aie
benchmark corroborates ~4/col-hop horizontal via 2cy x 2 modules, ~2/tile
vertical). Array-span skew is tens of cycles; `Delta_wall` is hundreds. Tolerances
for the Phase-5 held-out gate must sit strictly above the held-out kernel's known
`Delta_wall` residual (design Sec.5).

## Ceiling components (disclosed, not measured)
E/W anisotropy; absolute intra-offset (gauge); `d_h^{ch15}` in isolation;
per-module horizontal split; clock-tree phase skew; structured OR-tree asymmetry.
Toolchain sweep dispositions: design Sec.4.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/findings/2026-07-02-sp5c-phase0-identifiability.md
git commit -m "doc(#140): SP-5c Phase 0 identifiability + IF decisions findings

Consolidates the Phase-0 written-proof gate: decoupled-capture decision (with
pointers to the executable guards), IF-1/2/3 resolutions, determinism basis, and
the magnitude/tolerance estimate.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

# PHASE 1 -- Ingestion wire + inert §9a plumbing

### Task 1.1: Extend the `skew_constants.json` handoff schema

Add the honest-provenance fields the flip will need, before wiring anything.

**Files:**
- Modify: `tools/calibration/skew/schema.py`
- Test: `tools/test_skew_schema.py` (create)

**Interfaces:**
- Produces: `empty_constants()` and `write_constants(...)` gain keys
  `per_channel` (dict|None), `b_vector_range` (float|None), `jitter_range`
  (float|None), `assumptions` (dict), `d_h_path` (str|None). `write_constants`
  gains matching keyword-only params (all defaulting to `None`, `assumptions`
  defaulting to `None` -> written as `{}`).

- [ ] **Step 1: Write the failing test**

Create `tools/test_skew_schema.py`:

```python
import json
from calibration.skew.schema import empty_constants, write_constants, read_constants


def test_empty_constants_has_provenance_fields():
    c = empty_constants()
    for k in ("d_h", "d_v", "intra", "fit_residual", "source_route", "provenance",
              "per_channel", "b_vector_range", "jitter_range", "assumptions", "d_h_path"):
        assert k in c, k
    assert c["assumptions"] == {}


def test_write_read_roundtrip_with_provenance(tmp_path):
    p = tmp_path / "skew_constants.json"
    write_constants(
        p, d_h=4.0, d_v=2.0, intra_core=0.0, intra_mem=0.0, fit_residual=1e-9,
        source_route="shim_row", provenance="phase2 N=20 spaced",
        per_channel={"ch14": 4.0, "ch13": 4.0}, b_vector_range=0.0,
        jitter_range=0.0, assumptions={"horizontal_direction_isotropy": "assumed"},
        d_h_path="shim_row",
    )
    got = read_constants(p)
    assert got["d_h_path"] == "shim_row"
    assert got["per_channel"]["ch14"] == 4.0
    assert got["assumptions"]["horizontal_direction_isotropy"] == "assumed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m pytest test_skew_schema.py -v`
Expected: FAIL -- `empty_constants` missing `per_channel` etc.

- [ ] **Step 3: Write minimal implementation**

Replace the body of `tools/calibration/skew/schema.py` with (keeping the existing
`read_constants` passthrough):

```python
"""Read/write of skew_constants.json -- the SP-5b -> SP-5c handoff schema."""
import json


def empty_constants():
    return {
        "d_h": None, "d_v": None,
        "intra": {"core": None, "mem": None},
        "fit_residual": None, "source_route": None, "provenance": None,
        # SP-5c honest-provenance fields (design Sec.7):
        "per_channel": None, "b_vector_range": None, "jitter_range": None,
        "assumptions": {}, "d_h_path": None,
    }


def write_constants(path, *, d_h=None, d_v=None, intra_core=None, intra_mem=None,
                    fit_residual=None, source_route=None, provenance=None,
                    per_channel=None, b_vector_range=None, jitter_range=None,
                    assumptions=None, d_h_path=None):
    c = empty_constants()
    c.update({
        "d_h": d_h, "d_v": d_v,
        "intra": {"core": intra_core, "mem": intra_mem},
        "fit_residual": fit_residual, "source_route": source_route,
        "provenance": provenance, "per_channel": per_channel,
        "b_vector_range": b_vector_range, "jitter_range": jitter_range,
        "assumptions": assumptions if assumptions is not None else {},
        "d_h_path": d_h_path,
    })
    with open(path, "w") as f:
        json.dump(c, f, indent=2)
    return c


def read_constants(path):
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python3 -m pytest test_skew_schema.py -v`
Expected: PASS. Also run the existing suite to confirm no regression:
`cd tools && python3 -m pytest test_skew_*.py -q`

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/schema.py tools/test_skew_schema.py
git commit -m "feat(#140): SP-5c Phase 1 -- extend skew_constants.json handoff schema

Adds honest-provenance fields (per_channel, b_vector_range, jitter_range,
assumptions, d_h_path) the calibrated flip will disclose. Schema-first, before the
Rust ingestion wire reads it.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 1.2: `skew_ingest` -- JSON -> BroadcastTiming with fail-loud validation

The pure, unit-testable mapping + the `calibrated => all constants non-null`
assertion. Kept in the crate `src/` (not `build.rs`) so it is unit-tested.

**Files:**
- Create: `crates/xdna-archspec/src/skew_ingest.rs`
- Modify: `crates/xdna-archspec/src/lib.rs` (add `pub mod skew_ingest;`)

**Interfaces:**
- Consumes: `crate::types::BroadcastTiming` (fields `per_hop_horizontal: u8`,
  `per_hop_vertical: u8`, `intra_tile_core_offset: u8`, `intra_tile_mem_offset: u8`,
  `calibrated: bool`).
- Produces:
  `pub fn broadcast_timing_from_json(v: &serde_json::Value) -> Result<BroadcastTiming, String>`
  and `pub fn uncalibrated() -> BroadcastTiming` (all-zero, `calibrated: false`).

- [ ] **Step 1: Write the failing test**

Create `crates/xdna-archspec/src/skew_ingest.rs`:

```rust
//! Build-time ingestion of the SP-5c `skew_constants.json` handoff into a
//! `BroadcastTiming`. Fail-loud: a calibrated file with any null constant is an
//! error (never a silent fallback to zeros). Design Sec.7.

use crate::types::BroadcastTiming;

// ... (implementation added in Step 3) ...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncalibrated_is_all_zero() {
        let t = uncalibrated();
        assert_eq!(t.per_hop_horizontal, 0);
        assert_eq!(t.per_hop_vertical, 0);
        assert_eq!(t.intra_tile_core_offset, 0);
        assert_eq!(t.intra_tile_mem_offset, 0);
        assert!(!t.calibrated);
    }

    #[test]
    fn uncalibrated_json_maps_to_zeros() {
        let v = serde_json::json!({
            "calibrated": false, "d_h": null, "d_v": null,
            "intra": {"core": null, "mem": null}
        });
        let t = broadcast_timing_from_json(&v).unwrap();
        assert!(!t.calibrated);
        assert_eq!(t.per_hop_horizontal, 0);
    }

    #[test]
    fn calibrated_json_maps_all_fields() {
        let v = serde_json::json!({
            "calibrated": true, "d_h": 4, "d_v": 2,
            "intra": {"core": 0, "mem": 0}
        });
        let t = broadcast_timing_from_json(&v).unwrap();
        assert!(t.calibrated);
        assert_eq!(t.per_hop_horizontal, 4);
        assert_eq!(t.per_hop_vertical, 2);
    }

    #[test]
    fn calibrated_with_null_constant_is_error() {
        let v = serde_json::json!({
            "calibrated": true, "d_h": null, "d_v": 2,
            "intra": {"core": 0, "mem": 0}
        });
        let err = broadcast_timing_from_json(&v).unwrap_err();
        assert!(err.contains("calibrated"), "{err}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-archspec skew_ingest`
Expected: FAIL to compile -- `broadcast_timing_from_json` / `uncalibrated` not found
(and `lib.rs` does not yet declare the module).

- [ ] **Step 3: Write minimal implementation**

Add `pub mod skew_ingest;` to `crates/xdna-archspec/src/lib.rs` (alongside the
other `pub mod` declarations). Then insert the implementation into
`skew_ingest.rs` between the `use` and the `#[cfg(test)]`:

```rust
pub fn uncalibrated() -> BroadcastTiming {
    BroadcastTiming {
        per_hop_horizontal: 0,
        per_hop_vertical: 0,
        intra_tile_core_offset: 0,
        intra_tile_mem_offset: 0,
        calibrated: false,
    }
}

fn field_u8(v: &serde_json::Value, path: &[&str]) -> Option<u8> {
    let mut cur = v;
    for k in path {
        cur = cur.get(k)?;
    }
    cur.as_u64().map(|n| n as u8)
}

pub fn broadcast_timing_from_json(v: &serde_json::Value) -> Result<BroadcastTiming, String> {
    let calibrated = v.get("calibrated").and_then(|c| c.as_bool()).unwrap_or(false);
    if !calibrated {
        return Ok(uncalibrated());
    }
    let d_h = field_u8(v, &["d_h"]);
    let d_v = field_u8(v, &["d_v"]);
    let core = field_u8(v, &["intra", "core"]);
    let mem = field_u8(v, &["intra", "mem"]);
    match (d_h, d_v, core, mem) {
        (Some(h), Some(vv), Some(c), Some(m)) => Ok(BroadcastTiming {
            per_hop_horizontal: h,
            per_hop_vertical: vv,
            intra_tile_core_offset: c,
            intra_tile_mem_offset: m,
            calibrated: true,
        }),
        _ => Err(format!(
            "calibrated skew_constants.json has a null/missing constant \
             (d_h={d_h:?} d_v={d_v:?} core={core:?} mem={mem:?}); refusing to \
             build a model that claims calibrated with incomplete data"
        )),
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p xdna-archspec skew_ingest`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-archspec/src/skew_ingest.rs crates/xdna-archspec/src/lib.rs
git commit -m "feat(#140): SP-5c Phase 1 -- skew_constants.json -> BroadcastTiming ingest

Pure, unit-tested mapping with fail-loud validation: calibrated=true with any null
constant is an error, never a silent fallback to zeros (design Sec.7). Uncalibrated
maps to all-zero behavior-neutral timing.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 1.3: Wire the ingest into the model build (inert placeholder file)

Source `BroadcastTiming` from a committed `skew_constants.json` via `skew_ingest`,
replacing the hardcoded literal. Commit an uncalibrated placeholder so the build is
deterministic and the wire is exercised while staying inert.

**Files:**
- Create: `crates/xdna-archspec/data/skew_constants.json`
- Modify: `crates/xdna-archspec/src/model_builder.rs:270-280`
- Modify: `crates/xdna-archspec/build.rs` (add `cargo:rerun-if-changed`)

**Interfaces:**
- Consumes: `skew_ingest::broadcast_timing_from_json` / `uncalibrated` (Task 1.2).
- Produces: the model's `broadcast` field, now sourced from the JSON. Downstream
  `build.rs:556-580` codegen and `effects.rs:552-559` consumers are unchanged.

- [ ] **Step 1: Create the committed placeholder**

Create `crates/xdna-archspec/data/skew_constants.json`:

```json
{
  "d_h": null,
  "d_v": null,
  "intra": { "core": null, "mem": null },
  "fit_residual": null,
  "source_route": null,
  "provenance": "uncalibrated placeholder (SP-5c Phase 1); flip happens in Phase 6",
  "per_channel": null,
  "b_vector_range": null,
  "jitter_range": null,
  "assumptions": {},
  "d_h_path": null,
  "calibrated": false
}
```

- [ ] **Step 2: Write the failing test**

Add to `crates/xdna-archspec/src/model_builder.rs` (in its `#[cfg(test)] mod`, or
create one if absent) a test that the built model's broadcast timing is the
ingested (uncalibrated) value. First find how the module builds the model in tests;
if there is a `build_model()`-style entry, assert on its `.timing`. Minimal test:

```rust
#[test]
fn broadcast_timing_sourced_from_skew_constants_json() {
    // The committed placeholder is uncalibrated -> all zero, calibrated=false.
    let t = crate::skew_ingest::broadcast_timing_from_json(
        &serde_json::from_str::<serde_json::Value>(
            include_str!("../data/skew_constants.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(!t.calibrated);
    assert_eq!(t.per_hop_horizontal, 0);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p xdna-archspec broadcast_timing_sourced`
Expected: FAIL to compile if `include_str!` path is wrong, or PASS trivially if the
file resolves -- in which case proceed (this test is a wire-check, and the real
change is Step 4's model_builder edit).

- [ ] **Step 4: Replace the hardcoded literal**

In `crates/xdna-archspec/src/model_builder.rs:270-280`, replace the
`broadcast: BroadcastTiming { ... }` literal with:

```rust
broadcast: crate::skew_ingest::broadcast_timing_from_json(
    &serde_json::from_str::<serde_json::Value>(include_str!(
        "../data/skew_constants.json"
    ))
    .expect("skew_constants.json is not valid JSON"),
)
.expect("skew_constants.json failed validation"),
```

`include_str!` embeds the committed file at compile time (deterministic; no runtime
file IO). The `.expect` on validation is the **build-fails-loud** guard: a future
calibrated-with-null file panics the build here.

In `crates/xdna-archspec/build.rs`, near the top of `main` (or wherever
`rerun-if-changed` directives live), add:

```rust
println!("cargo:rerun-if-changed=data/skew_constants.json");
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p xdna-archspec` then `cargo test --lib`
Expected: PASS, including the existing `broadcast_timing_defaults_uncalibrated`
(`runtime.rs:804`) -- the model is still uncalibrated, so nothing regresses.

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-archspec/data/skew_constants.json crates/xdna-archspec/src/model_builder.rs crates/xdna-archspec/build.rs
git commit -m "feat(#140): SP-5c Phase 1 -- source BroadcastTiming from skew_constants.json

model_builder now ingests the committed data/skew_constants.json via skew_ingest
(include_str!, build-time, fail-loud on calibrated-with-null). Placeholder is
uncalibrated so the model stays behavior-neutral; the Phase-6 flip becomes a
reviewable JSON+regen diff. rerun-if-changed wired in build.rs.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

**Note -- deferred Sec.7 guards (content-hash + compiled-in provenance string).**
Design Sec.7 lists three ingestion guards. This slice implements guard (1) the
`calibrated => all-constants-non-null` fail-loud build assertion (Tasks 1.2/1.3).
Guards (2) a provenance/content hash of the *measured* JSON and (3) a compiled-in,
runtime-queryable provenance string are **deferred to Phase 6 (the flip)**, because
against the uncalibrated placeholder they are inert: `include_str!` +
`cargo:rerun-if-changed=data/skew_constants.json` (Task 1.3) already closes the
stale-artifact rebuild mode for Phase 1, and a content hash / provenance readout is
only meaningful once a real measured JSON exists. Recorded here so the Phase-6 plan
picks them up rather than assuming Sec.7 was fully discharged.

---

### Task 1.4: §9a(a) -- fixpoint-ch15 single-source test

Confirm one logical channel-15 flood driven through the fixpoint loop records
exactly one source (guards against a re-queued ch15 inserting a spurious second
source). If it FAILS, that is a real bug to fix before the flip -- surface it.

**Files:**
- Modify: `src/device/state/effects.rs` (test module `mod flood_source_capture_tests`, ~1263)

**Interfaces:**
- Consumes: `DeviceState::new_npu1()`, `propagate_broadcasts_fixpoint(col, row)`
  (effects.rs:714), `flood_sources() -> &HashSet<(u8,u8)>` (mod.rs:273),
  `PendingBroadcast::originated(channel)`, `array.get_mut(c,r)`.
- Produces: nothing (test-only, a gate).

- [ ] **Step 1: Write the test**

Add to `mod flood_source_capture_tests` in `src/device/state/effects.rs`, modeled
on `channel_15_flood_records_its_source` (1267-1280):

```rust
#[test]
fn fixpoint_channel_15_flood_records_exactly_one_source() {
    // §9a(a): one logical ch15 flood through the fixpoint loop must yield exactly
    // one recorded source, even though the fixpoint re-scans/re-propagates
    // pending broadcasts. A re-queued ch15 must NOT register a second source.
    let mut dev = DeviceState::new_npu1();
    let src = (0u8, 0u8);
    dev.array
        .get_mut(src.0, src.1)
        .unwrap()
        .pending_broadcasts
        .push(PendingBroadcast::originated(15));
    dev.propagate_broadcasts_fixpoint(src.0, src.1);
    assert_eq!(
        dev.flood_sources().len(),
        1,
        "one logical ch15 flood must record exactly one source, got {:?}",
        dev.flood_sources()
    );
    assert!(dev.flood_sources().contains(&src));
}
```

(Match the exact `PendingBroadcast::originated` / `pending_broadcasts` access
pattern used by the neighboring `channel_15_flood_records_its_source` test; adjust
field/method names to that test's actual form if they differ.)

- [ ] **Step 2: Run the test**

Run: `cargo test --lib fixpoint_channel_15_flood_records_exactly_one_source`
Expected: PASS (single-source contract holds through the fixpoint). If it FAILS
with `len() > 1`, STOP -- a real §9a bug is present; fix the re-queue path
(`propagate_broadcasts_fixpoint`, effects.rs:714+) so a relayed ch15 is not
recorded as `Originated`, then re-run.

- [ ] **Step 3: Commit**

```bash
git add src/device/state/effects.rs
git commit -m "test(#140): SP-5c Phase 1 -- §9a(a) fixpoint ch15 single-source gate

Confirms one logical channel-15 flood through propagate_broadcasts_fixpoint records
exactly one source (no spurious second from a re-queued/relayed ch15). Pre-flip
prerequisite from SP-4b design 9a.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 1.5: §9a(b) -- wire the sweep sidecar consumption

The consumption path (`run_engine(model_path=...)`) already exists and is
test-covered; wire the sweep's produced `origin_d.json` into it. Inert while
uncalibrated (skew=0, no causal fact) -- correct pre-flip.

**Files:**
- Modify: `tools/inference/run_experiment.py:101-102`
- Modify: `tools/trace-sweep.py` (pass the sidecar path into the experiment run)
- Test: `tools/test_inference_run_experiment_wires_model.py` (create)

**Interfaces:**
- Consumes: `run_engine(run_dirs, ledger_path, candidate_pairs, dump=, start_col=,
  model_path=)` (engine.py:28); the sweep already sets `XDNA_EMU_ORIGIN_D_OUT` to
  `work_dir/origin_d.json` (trace-sweep.py:591,1089).
- Produces: `run_experiment(...)` gains an optional `model_path=None` param threaded
  into its `run_engine` call.

- [ ] **Step 1: Write the failing test**

Create `tools/test_inference_run_experiment_wires_model.py`. Inspect
`run_experiment.py` first for the exact function signature; this test asserts the
`model_path` kwarg reaches `run_engine`:

```python
import inspect
from inference import run_experiment as re_mod


def test_run_experiment_forwards_model_path(monkeypatch):
    captured = {}

    def fake_run_engine(*args, **kwargs):
        captured.update(kwargs)
        captured["_args"] = args
        return {"ok": True}

    monkeypatch.setattr(re_mod, "run_engine", fake_run_engine)
    # run_experiment must accept model_path and forward it to run_engine.
    sig = inspect.signature(re_mod.run_experiment)
    assert "model_path" in sig.parameters, "run_experiment must accept model_path"
```

(If `run_experiment`'s real invocation needs fixtures to call end-to-end, keep the
test at the signature+forwarding level as above; a full-run test belongs to the
sweep integration, not this unit.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python3 -m pytest test_inference_run_experiment_wires_model.py -v`
Expected: FAIL -- `run_experiment` has no `model_path` parameter.

- [ ] **Step 3: Thread `model_path`**

In `tools/inference/run_experiment.py`, add `model_path=None` to the
`run_experiment(...)` signature and forward it at lines 101-102:

```python
    rep = run_engine(res["run_dirs"], str(ledger_path), candidate_pairs,
                     dump=engine_dump, start_col=start_col, model_path=model_path)
```

In `tools/trace-sweep.py`, where the per-work-dir experiment is invoked, pass the
sidecar path when it exists (the sweep already writes it via
`XDNA_EMU_ORIGIN_D_OUT`):

```python
    origin_d = work_dir / "origin_d.json"
    model_path = str(origin_d) if origin_d.exists() else None
    # ... in the run_experiment(...) call, add: model_path=model_path
```

(Locate the `run_experiment(` call site in trace-sweep.py and add the kwarg. If the
sweep calls the engine through a different entry, thread `model_path` along that
path to the same `run_engine` call.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_inference_run_experiment_wires_model.py -v`
Then the inference suite for no regression:
`cd tools && python3 -m pytest test_inference_*.py -q`
Expected: PASS. (Uncalibrated sidecar -> `causal_offset` returns None -> no-op, so
existing inference tests are unaffected.)

- [ ] **Step 5: Commit**

```bash
git add tools/inference/run_experiment.py tools/trace-sweep.py tools/test_inference_run_experiment_wires_model.py
git commit -m "feat(#140): SP-5c Phase 1 -- §9a(b) wire sweep origin_d sidecar consumption

Thread model_path from trace-sweep through run_experiment into run_engine so the
sweep's produced origin_d.json is consumed. Inert while uncalibrated (skew=0, no
causal fact); the plumbing is ready for the Phase-6 flip. SP-4b design 9a.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Final Verification

- [ ] **Full Python calibration suite:** `cd tools && python3 -m pytest test_skew_*.py test_inference_*.py -q` -- all green.
- [ ] **Full Rust lib suite:** `cargo test --lib` -- all green, model still uncalibrated (`broadcast_timing_defaults_uncalibrated` passes).
- [ ] **Inert confirmation:** `grep BROADCAST_CALIBRATED` in generated `gen_arch.rs` reads `false`; no emulation behavior changed.

## Out of scope (HW phases 2-6, planned after this slice)

`d_h` shim-row capture, free-flood `d_v` capture, the two-sided R1 spine, the
b-vector spaced-jitter gate, the held-out validation kernel
(`matrix_multiplication_using_cascade` candidate + trace-prep dependency), and the
one-way `calibrated` flip. These need Phoenix silicon and Phase-0's per-capture
proofs (this slice) in hand first. See design Sec. 5.
```
