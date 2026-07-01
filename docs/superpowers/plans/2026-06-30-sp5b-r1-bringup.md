# SP-5b R1 Bring-up Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up the R1 within-column skew instrument -- the differencing extractor, the trace->observation bridge, a taller real-spine kernel, the heavy emu inject-and-recover loop, and the HW runnability gate -- so it *runs* on Phoenix and satisfies the cross-language contract, producing no skew number.

**Architecture:** Python differencing solver (`r1_diff_extract`) + observation bridge (`r1_observe`, event-pair minus emulator `Delta_wall`) feed off decoded traces. A taller-spine objectfifo kernel (`sp5_skew_r1`, seeded from `of_q0_lean.py`) supplies real compute-event pairs down a single column. A Rust emu loop runs the compiled xclbin in-process twice (injected + zero constants) through a new runner override-seam and asserts the differencing recovers the injected `{d_v, intra_contrast}` exactly. A shell HW gate runs the kernel on Phoenix N times, subtracts an emulator `Delta_wall`, and asserts range-0 reproducibility -- never a value.

**Tech Stack:** Python 3.13 (numpy via existing `_solve`), Rust (`xclbin_suite.rs` in-process runner, `xdna-archspec`), mlir-aie IRON objectfifo kernels + `trace-prepare.py`/`mlir-trace-inject.py`, `bridge-trace-runner`, `tools/trace_decoder`.

This is Phase 1 of the SP-5b kernel/HW bring-up (spec:
`docs/superpowers/specs/2026-06-30-sp5b-kernel-hw-bringup-design.md`). Phases 2
(R3b-PC) and 3 (R3b-TM) are separate later plans.

## Global Constraints

- **Derive from the toolchain; never hardcode what can be extracted** (register offsets, event ids, geometry). Data-driven via `geometry.json`; no magic bit positions.
- **`cargo test --lib` stays green throughout.** The runtime-override seam is unset by default -> byte-identical flood/trace -> the 3 existing neutrality guards stay green.
- **Only `{d_v, intra_contrast}` are observable** (`intra_contrast = core_off - mem_off`); the absolute intras are gauge-unobservable (spec Sec.4.2). `r1_diff_extract` returns exactly these two + `fit_residual`; `min_rank=2`.
- **Kind grouping (verbatim from `effects.rs:527-532`):** `{core, shim}` -> core group, `{mem, memtile}` -> mem group.
- **Sign is pinned only for emu self-consistency** (inject convention X, recover X). The emu<->silicon sign correspondence is an open SP-5c question -- do not claim silicon validation.
- **HW gates assert shape + range-0 reproducibility across runs, NEVER a skew value.** Range-0 = the measured `b`-vector reproduces across N runs (mirrors SP-3's 20-run evidence), not that it equals any number.
- **>=3 collinear same-kind (`core`) points at >=3 distinct `dn_v`** for the `d_v` falsification (do not borrow the shim as the 3rd core-group point -- that assumes the grouping SP-5c validates).
- **Intra contrast is best-effort (Q2-A):** the emu loop recovers it exactly (Delta_wall cancels by construction); on HW it is contingent on emulator core<->mem Delta_wall fidelity and is emitted provisional if it fails range-0. `d_v` is the solid deliverable.
- **Virtualized column frame** for all kernel geometry (relative cols 0..N-1, rows 0 shim / 1 memtile / 2-5 core). Hop distances are relocation-invariant.
- **Commit trailer on every commit:**
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```

---

### Task 1: `r1_diff_extract` -- differencing solver

**Files:**
- Create: `tools/calibration/skew/r1_diff_extract.py`
- Test: `tools/test_skew_r1_diff_extract.py`

**Interfaces:**
- Consumes: `_solve.solve_design_matrix(A, b, min_rank) -> (x, fit_residual)`, raises `RankDeficientError` (already merged, `tools/calibration/skew/_solve.py`).
- Produces: `extract_r1_diff(pairs) -> {"d_v": float, "intra_contrast": float, "fit_residual": float}`. `pairs`: list of `{"a": {"dn_v": int, "kind": str}, "b": {"dn_v": int, "kind": str}, "skew": float}` where `skew = module_delay(b) - module_delay(a)`. `intra_contrast = core_off - mem_off`.

Model: `module_delay(M) = dn_v(M)*d_v + mem_off + (core_off - mem_off)*core_ind(M)`, `core_ind = 1` for `{core,shim}` else `0`. Differencing cancels `mem_off` (the gauge level), leaving `skew = d_v*(Ddn_v) + (core_off-mem_off)*(Dcore_ind)`.

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_skew_r1_diff_extract.py
import math
import pytest
from calibration.skew.r1_diff_extract import extract_r1_diff
from calibration.skew._solve import RankDeficientError


def _pair(dn_a, kind_a, dn_b, kind_b, d_v, contrast):
    # contrast = core_off - mem_off; core_ind = 1 for core/shim else 0.
    ci = {"core": 1, "shim": 1, "mem": 0, "memtile": 0}
    md = lambda dn, k: dn * d_v + (contrast if ci[k] else 0)  # mem_off gauge = 0
    return {"a": {"dn_v": dn_a, "kind": kind_a},
            "b": {"dn_v": dn_b, "kind": kind_b},
            "skew": float(md(dn_b, kind_b) - md(dn_a, kind_a))}


def test_recovers_injected_dv_and_contrast():
    d_v, contrast = 3.0, -2.0
    pairs = [
        _pair(2, "core", 3, "core", d_v, contrast),   # d_v hop
        _pair(3, "core", 4, "core", d_v, contrast),   # d_v hop
        _pair(2, "core", 2, "mem", d_v, contrast),    # contrast (same tile)
    ]
    r = extract_r1_diff(pairs)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-9)
    assert math.isclose(r["intra_contrast"], contrast, abs_tol=1e-9)
    assert r["fit_residual"] < 1e-9


def test_fit_residual_grows_on_nonuniform_hops():
    # 3 collinear core points, but the 2->3 hop differs from 3->4 (non-uniform d_v).
    pairs = [
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 3, "kind": "core"}, "skew": 3.0},
        {"a": {"dn_v": 3, "kind": "core"}, "b": {"dn_v": 4, "kind": "core"}, "skew": 5.0},
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 4, "kind": "core"}, "skew": 8.0},
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 2, "kind": "mem"}, "skew": -2.0},
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] > 1.0  # linear model cannot fit non-uniform hops


def test_two_points_per_axis_cannot_falsify():
    # Only 2 distinct dn_v (one hop) + one contrast pair: fits with ~0 residual.
    pairs = [
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 3, "kind": "core"}, "skew": 3.0},
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 2, "kind": "mem"}, "skew": -2.0},
    ]
    r = extract_r1_diff(pairs)
    assert r["fit_residual"] < 1e-9  # 2 points fit any line -- no falsification power


def test_rank_deficient_all_core_raises():
    # All-core pairs: contrast column is all-zero -> rank 1 < min_rank 2.
    pairs = [
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 3, "kind": "core"}, "skew": 3.0},
        {"a": {"dn_v": 3, "kind": "core"}, "b": {"dn_v": 4, "kind": "core"}, "skew": 3.0},
    ]
    with pytest.raises(RankDeficientError):
        extract_r1_diff(pairs)


def test_unknown_kind_raises():
    pairs = [{"a": {"dn_v": 2, "kind": "core"},
              "b": {"dn_v": 3, "kind": "bogus"}, "skew": 1.0}]
    with pytest.raises(ValueError):
        extract_r1_diff(pairs)


def test_sign_pin_contrast_negative():
    # core is EARLIER than mem (core_off < mem_off) -> contrast negative.
    # same tile: skew = md(mem) - md(core) = mem_off - core_off = -contrast.
    pairs = [
        _pair(2, "core", 3, "core", 3.0, -2.0),
        _pair(3, "core", 4, "core", 3.0, -2.0),
        {"a": {"dn_v": 2, "kind": "core"}, "b": {"dn_v": 2, "kind": "mem"}, "skew": 2.0},
    ]
    r = extract_r1_diff(pairs)
    assert r["intra_contrast"] < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest test_skew_r1_diff_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibration.skew.r1_diff_extract'`

- [ ] **Step 3: Write the implementation**

```python
# tools/calibration/skew/r1_diff_extract.py
"""R1 differencing skew extraction: solve {d_v, intra_contrast} from within-column
cross-domain pair-difference observations (#140 SP-5b).

Model: module_delay(M) = dn_v(M)*d_v + intra_off(kind_M), intra_off = core_off for
{core,shim}, mem_off for {mem,memtile}. Only the gauge-invariant contrast
(core_off - mem_off) is observable -- adding a constant to both offsets leaves every
reset target (max_delay - module_delay) unchanged -- so this recovers exactly
{d_v, intra_contrast} from pair differences (spec Sec.4.2). Structurally identical
to r3b_extract's reference-differencing. Falsifying per-hop uniformity needs >=3
collinear observations per axis (2 points fit any line with zero residual).
"""
from ._solve import solve_design_matrix

_CORE_GROUP = {"core", "shim"}
_MEM_GROUP = {"mem", "memtile"}


def _core_ind(kind):
    if kind in _CORE_GROUP:
        return 1.0
    if kind in _MEM_GROUP:
        return 0.0
    raise ValueError(f"unknown module kind: {kind!r}")


def extract_r1_diff(pairs):
    """pairs: list of {"a": {"dn_v": int, "kind": str},
                       "b": {"dn_v": int, "kind": str},
                       "skew": float}  where skew = module_delay(b) - module_delay(a).
    Returns {"d_v", "intra_contrast", "fit_residual"} where
    intra_contrast = (core_off - mem_off)."""
    A, bvec = [], []
    for p in pairs:
        a, b = p["a"], p["b"]
        A.append([float(b["dn_v"] - a["dn_v"]),
                  _core_ind(b["kind"]) - _core_ind(a["kind"])])
        bvec.append(float(p["skew"]))
    x, resid = solve_design_matrix(A, bvec, min_rank=2)
    return {"d_v": float(x[0]), "intra_contrast": float(x[1]),
            "fit_residual": resid}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest test_skew_r1_diff_extract.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/r1_diff_extract.py tools/test_skew_r1_diff_extract.py
git commit -m "feat(#140): SP-5b R1 differencing extractor -> {d_v, intra_contrast}

<trailer>"
```

---

### Task 2: `r1_observe` -- decoded trace(s) -> pair-difference observations

**Files:**
- Create: `tools/calibration/skew/r1_observe.py`
- Test: `tools/test_skew_r1_observe.py`

**Interfaces:**
- Consumes: two lists of flat event dicts (`trace.events.json` schema: `{"col","row","pkt_type","name","soc", ...}`; `pkt_type` 0=core/1=mem/2=shim/3=memtile) and a parsed `geometry.json`.
- Produces: `observe_r1(measured_events, dwall_events, geometry) -> list[pair]` in the exact shape Task 1 consumes: `{"a": {"dn_v","kind"}, "b": {"dn_v","kind"}, "skew": float}`.

Per geometry pair `(a, b)`, using the first (min-`soc`) occurrence of each anchor event (mirrors `tools/trace_join.py:41-54`):
`skew = (soc_meas(a) - soc_meas(b)) - (soc_dwall(a) - soc_dwall(b))` -- the deterministic `Delta_wall = soc_dwall(a) - soc_dwall(b)` cancels, leaving `module_delay(b) - module_delay(a)`. On emu, `dwall_events` is a zero-constants run; on silicon, an emulator run at zero constants (spec Sec.4.2/4.4).

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_skew_r1_observe.py
import pytest
from calibration.skew.r1_observe import observe_r1


def _ev(col, row, pkt, name, soc):
    return {"col": col, "row": row, "pkt_type": pkt, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


_GEOM = {
    "pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
         "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 3}},
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "PORT_RUNNING_0", "dn_v": 2}},
    ]
}


def test_delta_wall_subtraction_isolates_skew():
    # dwall run: pure execution timing (skew=0). measured: dwall + injected skew.
    dwall = [_ev(0, 2, 0, "LOCK_STALL", 100), _ev(0, 3, 0, "LOCK_STALL", 90),
             _ev(0, 2, 1, "PORT_RUNNING_0", 200)]
    # inject: core(0,3) later by 3 vs core(0,2); mem(0,2) later by 2 vs core(0,2).
    measured = [_ev(0, 2, 0, "LOCK_STALL", 100), _ev(0, 3, 0, "LOCK_STALL", 93),
                _ev(0, 2, 1, "PORT_RUNNING_0", 202)]
    obs = observe_r1(measured, dwall, _GEOM)
    # pair 0: skew = (100-93) - (100-90) = -3  (md(b=core@3) - md(a=core@2))
    assert obs[0]["skew"] == pytest.approx(-3.0)
    assert obs[0]["a"]["kind"] == "core" and obs[0]["b"]["kind"] == "core"
    # pair 1: skew = (100-202) - (100-200) = -2  (md(mem@2) - md(core@2))
    assert obs[1]["skew"] == pytest.approx(-2.0)
    assert obs[1]["b"]["kind"] == "mem"


def test_uses_first_occurrence():
    dwall = [_ev(0, 2, 0, "LOCK_STALL", 100), _ev(0, 3, 0, "LOCK_STALL", 100),
             _ev(0, 2, 1, "PORT_RUNNING_0", 100)]
    measured = [_ev(0, 2, 0, "LOCK_STALL", 50), _ev(0, 2, 0, "LOCK_STALL", 999),
                _ev(0, 3, 0, "LOCK_STALL", 50), _ev(0, 2, 1, "PORT_RUNNING_0", 50)]
    obs = observe_r1(measured, dwall, _GEOM)
    assert obs[0]["skew"] == pytest.approx(0.0)  # first-occurrence 50, not 999


def test_missing_anchor_raises():
    dwall = [_ev(0, 2, 0, "LOCK_STALL", 100)]
    measured = [_ev(0, 2, 0, "LOCK_STALL", 100)]
    with pytest.raises(KeyError):
        observe_r1(measured, dwall, _GEOM)  # core(0,3) anchor absent


def test_unknown_pkt_type_raises():
    geom = {"pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "E", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 9, "name": "E", "dn_v": 2}}]}
    ev = [_ev(0, 2, 0, "E", 10), _ev(0, 2, 9, "E", 10)]
    with pytest.raises(ValueError):
        observe_r1(ev, ev, geom)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest test_skew_r1_observe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibration.skew.r1_observe'`

- [ ] **Step 3: Write the implementation**

```python
# tools/calibration/skew/r1_observe.py
"""R1 observation bridge: decoded trace(s) -> pair-difference observations for
r1_diff_extract (#140 SP-5b).

For each deterministic cross-domain pair (a, b) named in geometry.json:
  skew = (soc_meas(a) - soc_meas(b)) - (soc_dwall(a) - soc_dwall(b))
       = module_delay(b) - module_delay(a)
The Delta_wall = soc_dwall(a) - soc_dwall(b) cancels the (deterministic) wall-clock
separation of the two events, leaving the skew. Works identically on emu and
silicon traces (spec Sec.4.2): on emu, dwall = a zero-constants run; on silicon,
an emulator run at zero constants. Anchor = first (min-soc) occurrence of the named
event on that (col,row,pkt_type), mirroring tools/trace_join.anchored_firsts.

Pairs need not literally co-fire (any deterministic pair works), but small
|Delta_wall| keeps emulator Delta_wall-prediction error from swamping the
single-digit skew -- so the kernel picks near-co-firing anchors.
"""

_PKT_KIND = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}


def _kind(pkt_type):
    if pkt_type not in _PKT_KIND:
        raise ValueError(f"unknown pkt_type: {pkt_type!r}")
    return _PKT_KIND[pkt_type]


def _first_soc(events, col, row, pkt_type, name):
    best = None
    for e in events:
        if (e["col"] == col and e["row"] == row
                and e["pkt_type"] == pkt_type and e["name"] == name):
            if best is None or e["soc"] < best:
                best = e["soc"]
    if best is None:
        raise KeyError(f"anchor event not found: {col}|{row}|{pkt_type}|{name}")
    return best


def observe_r1(measured_events, dwall_events, geometry):
    """measured_events, dwall_events: list of flat event dicts
    (trace.events.json schema: col,row,pkt_type,name,soc,...).
    geometry: parsed geometry.json with 'pairs', each = {"a": {...}, "b": {...}}
    where each endpoint carries col,row,pkt_type,name,dn_v.
    Returns list of {"a": {dn_v,kind}, "b": {dn_v,kind}, "skew": float}."""
    out = []
    for p in geometry["pairs"]:
        a, b = p["a"], p["b"]
        sa_m = _first_soc(measured_events, a["col"], a["row"], a["pkt_type"], a["name"])
        sb_m = _first_soc(measured_events, b["col"], b["row"], b["pkt_type"], b["name"])
        sa_d = _first_soc(dwall_events, a["col"], a["row"], a["pkt_type"], a["name"])
        sb_d = _first_soc(dwall_events, b["col"], b["row"], b["pkt_type"], b["name"])
        skew = (sa_m - sb_m) - (sa_d - sb_d)
        out.append({"a": {"dn_v": a["dn_v"], "kind": _kind(a["pkt_type"])},
                    "b": {"dn_v": b["dn_v"], "kind": _kind(b["pkt_type"])},
                    "skew": float(skew)})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest test_skew_r1_observe.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/r1_observe.py tools/test_skew_r1_observe.py
git commit -m "feat(#140): SP-5b r1_observe bridge -- event-pair minus Delta_wall

<trailer>"
```

---

### Task 3: Runner override seam on `XclbinSuite`/`XclbinTest`

**Files:**
- Modify: `src/testing/xclbin_suite.rs` (add builder field + apply in `run_single_inner` after `apply_cdo`)
- Test: `src/testing/xclbin_suite.rs` `#[cfg(test)] mod tests` (a neutrality test)

**Interfaces:**
- Consumes: `DeviceState::set_broadcast_timing_override(Option<BroadcastTiming>)` (`effects.rs:564`); `InterpreterEngine::device_mut()` (`coordinator.rs:458`).
- Produces: `XclbinTest::with_broadcast_timing_override(self, Option<BroadcastTiming>) -> Self`, applied inside `run_single_inner` immediately after `engine.device_mut().apply_cdo(&cdo)` and before ELF load. When `None` (default), behavior is byte-identical to today.

The override is a `DeviceState` field read fresh by `propagate_broadcasts` on every flood, so setting it before the run keeps it in force for every flood including the trace-start flood -- no clobber hazard (spec Sec.4.3).

- [ ] **Step 1: Write the failing test**

```rust
// in src/testing/xclbin_suite.rs #[cfg(test)] mod tests
#[test]
fn broadcast_timing_override_builder_defaults_none() {
    use crate::testing::xclbin_suite::XclbinTest;
    let t = XclbinTest::from_path("/nonexistent/aie.xclbin");
    assert!(t.broadcast_timing_override.is_none());
    let t2 = t.with_broadcast_timing_override(Some(
        xdna_archspec::types::BroadcastTiming {
            per_hop_horizontal: 0, per_hop_vertical: 3,
            intra_tile_core_offset: 0, intra_tile_mem_offset: 0, calibrated: true,
        }));
    assert_eq!(t2.broadcast_timing_override.as_ref().unwrap().per_hop_vertical, 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib broadcast_timing_override_builder_defaults_none`
Expected: FAIL to compile -- no field `broadcast_timing_override`, no method `with_broadcast_timing_override`.

- [ ] **Step 3: Add the field, builder, and application point**

In the `XclbinTest` struct (around `src/testing/xclbin_suite.rs:125-154`), add field:
```rust
    /// SP-5b: runtime broadcast-timing override applied to DeviceState before the
    /// run (after apply_cdo). None (default) = byte-identical to production.
    pub broadcast_timing_override: Option<xdna_archspec::types::BroadcastTiming>,
```
Initialize it to `None` in `XclbinTest::from_path` (`:158`) and any other constructor that builds the struct literally.

Add the builder near `with_buffer_spec` (`:245`):
```rust
    /// SP-5b (#140): inject a broadcast-timing override for skew inject-and-recover.
    pub fn with_broadcast_timing_override(
        mut self,
        override_timing: Option<xdna_archspec::types::BroadcastTiming>,
    ) -> Self {
        self.broadcast_timing_override = override_timing;
        self
    }
```

In `run_single_inner` (`:646`), immediately after the existing
`engine.device_mut().apply_cdo(&cdo)` call (`:746`), add:
```rust
        if let Some(bt) = self.broadcast_timing_override.clone() {
            engine.device_mut().set_broadcast_timing_override(Some(bt));
        }
```

- [ ] **Step 4: Run the test + neutrality suite**

Run: `cargo test --lib broadcast_timing_override`
Expected: PASS.
Run: `cargo test --lib` (full library) -- Expected: no regressions; the 3 broadcast-timing neutrality guards stay green (default `None` path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/testing/xclbin_suite.rs
git commit -m "feat(#140): SP-5b xclbin-runner broadcast-timing override seam

<trailer>"
```

---

### Task 4: R1 kernel -- taller real spine + trace config + geometry

**Files:**
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r1/sp5_skew_r1.py` (kernel, seeded from `of_q0_lean.py`)
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r1/trace_config.json` (trace core+mem of rows 2/3/4 + anchors)
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r1/geometry.json` (pairs + dn_v/kind for `r1_observe`)
- Create: `mlir-aie/test/npu-xrt/sp5_skew_r1/README.md` (build/run recipe, `REQUIRES: ryzen_ai_npu1`)

This task is kernel authoring: iterate against the compiler; the "test" is compile + emu-run + no-TDR reproduce, not a unit test.

**Interfaces:**
- Produces: a compiled traced xclbin under `mlir-aie/build/test/npu-xrt/sp5_skew_r1/` (path consumed by Tasks 5 and 6), and `geometry.json` in the shape Task 2 consumes.

- [ ] **Step 1: Extend the spine to 3 collinear cores**

Copy `mlir-aie/test/npu-xrt/spike_bringup/of_q0_lean.py` to `sp5_skew_r1.py` and add a third vertical consumer core at row 4, keeping pure lock/DMA handshakes (no buffer compute) and Q=0:
```python
dev = AIEDevice.npu1_2col   # single-column geometry; col 0 relative frame
REPS = 32
OBJ = 64

# ... inside device_body(), after ConsA:
ProdCore = tile(0, 2)   # dn_v 2
ConsA    = tile(0, 3)   # dn_v 3
ConsB    = tile(0, 4)   # dn_v 4  (NEW third collinear core)

of_src = object_fifo("src", ProdCore, MemTile, 2, tObj)
of_d   = object_fifo("d", MemTile, ConsA, 2, tObj)
object_fifo_link(of_src, [of_d], [], [0])

of_e   = object_fifo("e", ConsA, MemTile, 2, tObj)   # ConsA -> memtile
of_f   = object_fifo("f", MemTile, ConsB, 2, tObj)   # memtile -> ConsB
object_fifo_link(of_e, [of_f], [], [0])

of_j   = object_fifo("j", ConsB, MemTile, 2, tObj)   # ConsB -> memtile -> shim drain
of_out = object_fifo("out", MemTile, ShimTile, 2, tObj)
object_fifo_link([of_j], of_out, [0], [])

@core(ProdCore)
def prod_body():
    for _ in range_(REPS):
        of_src.acquire(ObjectFifoPort.Produce, 1); of_src.release(ObjectFifoPort.Produce, 1)

@core(ConsA)
def consa_body():
    for _ in range_(REPS):
        of_d.acquire(ObjectFifoPort.Consume, 1); of_e.acquire(ObjectFifoPort.Produce, 1)
        of_d.release(ObjectFifoPort.Consume, 1); of_e.release(ObjectFifoPort.Produce, 1)

@core(ConsB)
def consb_body():
    for _ in range_(REPS):
        of_f.acquire(ObjectFifoPort.Consume, 1); of_j.acquire(ObjectFifoPort.Produce, 1)
        of_f.release(ObjectFifoPort.Consume, 1); of_j.release(ObjectFifoPort.Produce, 1)
```
Keep the `runtime_sequence` draining `of_out` (as in `of_q0_lean.py`).

- [ ] **Step 2: Author the trace config (core + mem of rows 2/3/4 + anchors)**

`trace_config.json` mirrors `spike_bringup/build_q0_lean_trace/trace_config.json`, extended to trace, for each of tiles (0,2),(0,3),(0,4): the **core** module (pkt_type 0) with the lock-event set incl. `LOCK_STALL`, and the **mem** module (pkt_type 1) with a reproducible mem event (e.g. `PORT_RUNNING_0`). Also keep memtile (0,1) and shim (0,0) anchors. Routing/buffer block identical to the lean config (`shim_col:0, shim_dma_channel:1, shim_bd_id:15, trace_done:{broadcast:14,user_event:USER_EVENT_2}, size_bytes:16384, kernel_arg_slot:4`).

- [ ] **Step 3: Write `geometry.json`**

```json
{
  "source": {"col": 0, "row": 0},
  "pairs": [
    {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
     "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 3}},
    {"a": {"col": 0, "row": 3, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 3},
     "b": {"col": 0, "row": 4, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 4}},
    {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
     "b": {"col": 0, "row": 4, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 4}},
    {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "LOCK_STALL", "dn_v": 2},
     "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "PORT_RUNNING_0", "dn_v": 2}}
  ]
}
```
The first three are the `d_v` pairs (>=3 collinear core points at dn_v 2/3/4); the fourth is the best-effort intra-contrast pair (same-tile core<->mem, Q2-A).

- [ ] **Step 4: Build the traced xclbin (both compilers) and emu-run**

Build via the trace-prepare path (as `spike_bringup` does):
```bash
cd /home/triple/npu-work/xdna-emu
# Generate MLIR, inject trace, compile (Chess ground truth + Peano informational),
# following spike_bringup/README.md recipe with this dir's trace_config.json.
python tools/trace-prepare.py --kernel mlir-aie/test/npu-xrt/sp5_skew_r1/sp5_skew_r1.py \
  --trace-config mlir-aie/test/npu-xrt/sp5_skew_r1/trace_config.json \
  --out mlir-aie/build/test/npu-xrt/sp5_skew_r1/
```
(Exact invocation per `spike_bringup/README.md:52-82`; reproduce its aiecc + trace steps for this dir.)

Acceptance: the xclbin compiles under Chess; an emulator run (in-process runner, or `XDNA_EMU=1 ./test.exe`) produces a decodable trace whose event set includes `LOCK_STALL` on core (0,2),(0,3),(0,4) and `PORT_RUNNING_0` on mem (0,2), with no emulator panic.

- [ ] **Step 5: Prove Q=0 / no-TDR / reproducible on emu**

Run the emu 3x; confirm identical tile/event set and no TDR/panic. (HW no-TDR proof is Task 6.)

- [ ] **Step 6: Commit**

```bash
git add mlir-aie/test/npu-xrt/sp5_skew_r1/
git commit -m "feat(#140): SP-5b R1 within-column kernel (3-core spine + trace + geometry)

<trailer>"
```

---

### Task 5: Emu inject-and-recover loop (heavy, real xclbin in-process)

**Files:**
- Create: `tools/calibration/skew/r1_emu_recover.py` (Python harness: decode two trace bins -> observe -> extract -> assert)
- Test: `src/testing/skew_r1_emu_loop.rs` (Rust integration test) + register in `src/testing/mod.rs`
- Test: `tools/test_skew_r1_emu_recover.py` (unit-test the Python harness on synthetic bins-as-events)

**Interfaces:**
- Consumes: Task 3 seam (`with_broadcast_timing_override`), `run_single_with_trace` (`xclbin_suite.rs:594`), Task 4 xclbin, `observe_r1` (Task 2), `extract_r1_diff` (Task 1), `tools/trace_decoder.parse_trace`.
- Produces: a `cargo test --lib` (or `--test`) gate asserting recovered `{d_v, intra_contrast} == injected` exactly.

Design: the Rust test runs the xclbin twice via the runner -- once with `with_broadcast_timing_override(Some(injected))`, once with `None` (zero) -- captures each `trace_bytes` (3rd tuple element), writes both to temp `.bin` files, then invokes the Python harness which decodes both, runs `observe_r1(measured=injected, dwall=zero, geometry)` -> `extract_r1_diff`, and asserts against the injected constants passed on argv. Rust checks the harness exit code.

- [ ] **Step 1: Write the Python harness + its failing unit test**

```python
# tools/test_skew_r1_emu_recover.py
import json, math
from calibration.skew.r1_emu_recover import recover_and_check


def _ev(col, row, pkt, name, soc):
    return {"col": col, "row": row, "pkt_type": pkt, "name": name, "soc": soc}


def test_recover_matches_injected(tmp_path):
    geom = {"pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3}},
        {"a": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3},
         "b": {"col": 0, "row": 4, "pkt_type": 0, "name": "L", "dn_v": 4}},
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "P", "dn_v": 2}}]}
    dwall = [_ev(0, 2, 0, "L", 100), _ev(0, 3, 0, "L", 100),
             _ev(0, 4, 0, "L", 100), _ev(0, 2, 1, "P", 100)]
    d_v, contrast = 3, -2
    # Build measured so observe skew = md(b)-md(a): core pairs -> d_v per hop,
    # the same-tile core->mem pair -> -contrast (= mem_off - core_off).
    # core@r soc = 100 - (r-2)*d_v ; mem@2 soc = 100 + contrast.
    meas = [_ev(0, 2, 0, "L", 100), _ev(0, 3, 0, "L", 100 - 1 * d_v),
            _ev(0, 4, 0, "L", 100 - 2 * d_v), _ev(0, 2, 1, "P", 100 + contrast)]
    ok, got = recover_and_check(meas, dwall, geom,
                                expect_d_v=d_v, expect_contrast=contrast)
    assert ok, got


def test_mismatch_flags(tmp_path):
    geom = {"pairs": [
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3}},
        {"a": {"col": 0, "row": 3, "pkt_type": 0, "name": "L", "dn_v": 3},
         "b": {"col": 0, "row": 4, "pkt_type": 0, "name": "L", "dn_v": 4}},
        {"a": {"col": 0, "row": 2, "pkt_type": 0, "name": "L", "dn_v": 2},
         "b": {"col": 0, "row": 2, "pkt_type": 1, "name": "P", "dn_v": 2}}]}
    ev = [_ev(0, 2, 0, "L", 100), _ev(0, 3, 0, "L", 100),
          _ev(0, 4, 0, "L", 100), _ev(0, 2, 1, "P", 100)]
    ok, _ = recover_and_check(ev, ev, geom, expect_d_v=3, expect_contrast=-2)
    assert not ok  # recovered d_v=0,contrast=0 != injected
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_skew_r1_emu_recover.py -v`
Expected: FAIL -- `No module named 'calibration.skew.r1_emu_recover'`

- [ ] **Step 3: Implement the harness**

```python
# tools/calibration/skew/r1_emu_recover.py
"""Emu inject-and-recover check for R1 (#140 SP-5b): decode injected + zero
trace bins, run observe -> differencing extract, assert recovered {d_v,
intra_contrast} == injected. Plumbing/regression -- validates the seam ->
timer-reset -> in-process-run -> decode -> observe -> extract pipeline, NOT
silicon correctness (spec Sec.4.3)."""
import json
import math
import sys

from .r1_observe import observe_r1
from .r1_diff_extract import extract_r1_diff


def recover_and_check(measured_events, dwall_events, geometry,
                      *, expect_d_v, expect_contrast, abs_tol=1e-6):
    obs = observe_r1(measured_events, dwall_events, geometry)
    r = extract_r1_diff(obs)
    ok = (math.isclose(r["d_v"], expect_d_v, abs_tol=abs_tol)
          and math.isclose(r["intra_contrast"], expect_contrast, abs_tol=abs_tol))
    return ok, r


def _events_from_bin(path):
    # Decode a raw trace.bin (as produced by the in-process runner) to flat dicts.
    from trace_decoder import parse_trace  # tools/ on sys.path
    with open(path, "rb") as f:
        raw = f.read()
    return [{"col": e.col, "row": e.row, "pkt_type": e.pkt_type,
             "name": e.name, "soc": e.soc} for e in parse_trace(raw)]


def main(argv):
    # argv: injected.bin zero.bin geometry.json expect_d_v expect_contrast
    inj, zero, geom_path, d_v, contrast = argv[1:6]
    with open(geom_path) as f:
        geom = json.load(f)
    ok, r = recover_and_check(_events_from_bin(inj), _events_from_bin(zero),
                              geom, expect_d_v=float(d_v),
                              expect_contrast=float(contrast))
    print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Run the Python unit test (passes)**

Run: `cd tools && python -m pytest test_skew_r1_emu_recover.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the Rust integration test**

```rust
// src/testing/skew_r1_emu_loop.rs
//! SP-5b R1 emu inject-and-recover loop (#140): run the R1 xclbin in-process
//! twice (injected + zero constants) and assert the Python differencing bridge
//! recovers the injected {d_v, intra_contrast} exactly. Plumbing, not physics.
#[cfg(test)]
mod tests {
    use crate::testing::xclbin_suite::{XclbinTest, XclbinSuite};
    use xdna_archspec::types::BroadcastTiming;
    use std::io::Write;
    use std::process::Command;

    #[test]
    #[ignore] // requires the Task-4 xclbin built under mlir-aie/build/...
    fn r1_emu_recover_matches_injected() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let xclbin = manifest.join("../mlir-aie/build/test/npu-xrt/sp5_skew_r1/chess/aie.xclbin");
        let geom = manifest.join("../mlir-aie/test/npu-xrt/sp5_skew_r1/geometry.json");
        let (d_v, core_off, mem_off) = (3u8, 4u8, 2u8); // contrast = core-mem = +2
        let injected = BroadcastTiming {
            per_hop_horizontal: 0, per_hop_vertical: d_v,
            intra_tile_core_offset: core_off, intra_tile_mem_offset: mem_off,
            calibrated: false,
        };
        let suite = XclbinSuite::new();
        let dir = tempfile::tempdir().unwrap();

        let run = |bt: Option<BroadcastTiming>, name: &str| -> std::path::PathBuf {
            let test = XclbinTest::from_path(&xclbin).with_broadcast_timing_override(bt);
            let (_outcome, _out, trace) = suite.run_single_with_trace(&test);
            let bytes = trace.expect("in-process run produced no trace");
            let p = dir.path().join(name);
            std::fs::File::create(&p).unwrap().write_all(&bytes).unwrap();
            p
        };
        let inj = run(Some(injected.clone()), "injected.bin");
        let zero = run(None, "zero.bin");

        let status = Command::new("python")
            .arg(manifest.join("../tools/calibration/skew/r1_emu_recover.py"))
            .arg(&inj).arg(&zero).arg(&geom)
            .arg(d_v.to_string())
            .arg(((core_off as i32) - (mem_off as i32)).to_string())
            .env("PYTHONPATH", manifest.join("../tools"))
            .status().expect("run r1_emu_recover.py");
        assert!(status.success(), "recovered != injected (see harness JSON)");
    }
}
```
Register the module in `src/testing/mod.rs` (`mod skew_r1_emu_loop;`). The test is `#[ignore]` because it needs the compiled xclbin; it is run explicitly (Step 6), not in the default `cargo test --lib` sweep, so `cargo test --lib` stays green without the build artifact.

- [ ] **Step 6: Build the xclbin (Task 4) then run the loop**

Run: `cargo build -p xdna-emu-ffi` (ensure runner deps current), then
`cargo test --lib r1_emu_recover_matches_injected -- --ignored --nocapture`
Expected: PASS -- recovered `{d_v: 3.0, intra_contrast: 2.0}` matches injected.

- [ ] **Step 7: Commit**

```bash
git add tools/calibration/skew/r1_emu_recover.py tools/test_skew_r1_emu_recover.py \
        src/testing/skew_r1_emu_loop.rs src/testing/mod.rs
git commit -m "feat(#140): SP-5b R1 emu inject-and-recover loop (real xclbin in-process)

<trailer>"
```

---

### Task 6: R1 HW runnability gate

**Files:**
- Create: `build/experiments/sp5-skew/r1_gate.sh` (templated off `build/experiments/sp3-spike-trace/task3_gate.sh`)
- Create: `build/experiments/sp5-skew/r1_tally.py` (range-0 + non-degeneracy check over N runs)

HW-gated: needs Phoenix. Not classic TDD; the "test" is N clean runs + range-0.

**Interfaces:**
- Consumes: Task 4 xclbin, `observe_r1` (Task 2), `parse_trace`, an emulator zero-constants run for `Delta_wall`.
- Produces: a pass/fail gate that asserts runnability + reproducibility, never a value.

- [ ] **Step 1: Author `r1_gate.sh`**

Clone `task3_gate.sh` structure. For N serial runs (default 20), each: `env -u XDNA_EMU XDNA_EMU_RUNTIME= ./test.exe` (real HW), capture `trace.bin`, rc, and dmesg delta (TDR/reset scan). Once, run the same xclbin through the emulator at zero constants to produce `dwall.events.json`. Decode each HW `trace.bin` to `run_$i.events.json` via `parse-trace.py`.

- [ ] **Step 2: Author `r1_tally.py`**

```python
# build/experiments/sp5-skew/r1_tally.py
"""R1 HW gate tally (#140 SP-5b): assert runnability + range-0 reproducibility of
the per-pair skew across N HW runs. NO value assertion -- range-0 is a
reproducibility bound, mirroring SP-3's 20-run evidence."""
import json, sys, glob
sys.path.insert(0, "tools")
from calibration.skew.r1_observe import observe_r1
from calibration.skew.r1_diff_extract import extract_r1_diff


def _load(p):
    return json.load(open(p))["events"]


def main(run_glob, dwall_path, geom_path, max_range=0):
    geom = json.load(open(geom_path))
    dwall = _load(dwall_path)
    per_pair = None
    for run in sorted(glob.glob(run_glob)):
        obs = observe_r1(_load(run), dwall, geom)
        skews = [o["skew"] for o in obs]
        if per_pair is None:
            per_pair = [[] for _ in skews]
        for i, s in enumerate(skews):
            per_pair[i].append(s)
    # Non-degeneracy: >=3 distinct dn_v of kind core among the pairs' endpoints.
    dn_core = {ep["dn_v"] for p in geom["pairs"] for ep in (p["a"], p["b"])
               if ep["pkt_type"] == 0}
    assert len(dn_core) >= 3, f"degenerate: only {len(dn_core)} core dn_v"
    # Range-0 (or within max_range) reproducibility, per pair.
    ranges = [max(v) - min(v) for v in per_pair]
    print(json.dumps({"n_runs": len(per_pair[0]), "ranges": ranges,
                      "dn_core": sorted(dn_core)}))
    # The intra-contrast pair (last) is best-effort (Q2-A): report but do not fail.
    dv_ranges = ranges[:-1]
    assert all(r <= max_range for r in dv_ranges), f"d_v not range-0: {dv_ranges}"
    contrast_ok = ranges[-1] <= max_range
    print("contrast: " + ("range-0" if contrast_ok else "PROVISIONAL (Q2-A)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
```

- [ ] **Step 3: Run the gate on Phoenix (single HW capture is cheap)**

Run: `bash build/experiments/sp5-skew/r1_gate.sh` (drives 20 HW runs + 1 emu dwall + tally).
Expected: rc-0 all runs, no TDR, `>=3` distinct core `dn_v`, `d_v` pair ranges all 0 (or `<= max_range`); contrast pair reported range-0 or PROVISIONAL. **No skew value is asserted.**

- [ ] **Step 4: Commit**

```bash
git add build/experiments/sp5-skew/r1_gate.sh build/experiments/sp5-skew/r1_tally.py
git commit -m "feat(#140): SP-5b R1 HW runnability gate (range-0, no value assertion)

<trailer>"
```

---

## Phase-1 done criteria

- `r1_diff_extract` + `r1_observe` unit-green (Tasks 1-2).
- Runner override seam landed, `cargo test --lib` green (Task 3).
- R1 kernel compiles (Chess), emu-runs, no-TDR reproducible on emu (Task 4).
- Emu loop recovers injected `{d_v, intra_contrast}` exactly (Task 5).
- HW gate: N clean Phoenix runs, `d_v` range-0, contrast range-0-or-provisional, no value asserted (Task 6).

Handoff: Phase 2 (R3b-PC) is a separate plan. `intra_contrast` reliability on silicon, sign correspondence, and any value measurement are SP-5c.
