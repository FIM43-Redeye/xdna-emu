# SP-5b Software Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-hardware software core of the SP-5b skew-measurement apparatus -- the emulator runtime-override seam, both Python extraction solvers, and the emu trace-origin plumbing check -- so the kernel/HW bring-up (a separate later plan) has a proven data contract to emit against.

**Architecture:** Four TDD tasks. (1) A runtime-override seam on `DeviceState` lets a test/tool set the four broadcast-timing constants + `calibrated` non-zero at run time (they are compile-time-frozen today), consumed by both the flood and the sidecar export. (2) A shared least-squares core plus the R1 within-column extractor (`d_v` + intra, reflected-sign reconciliation). (3) The R3b interval-difference extractor (`d_h`, `d_v`). (4) A Rust plumbing test proving a seam-injected constant surfaces in the *decoded* trace. Kernel authoring, HW runnability gates, and the cross-language "extractor on a real multi-module kernel trace" contract are OUT -- they go in the kernel/HW bring-up plan.

**Tech Stack:** Rust (emulator, `cargo test --lib`), Python 3.13 + numpy (extraction, pytest via `tools/conftest.py`).

## Global Constraints

- Derive hardware behavior from the toolchain; never hardcode what can be extracted. This is alpha hardware-characterization tooling, not production -- pragmatic engineering, but the honesty hooks stay.
- The 4-knob model is exactly `BROADCAST_PER_HOP_HORIZONTAL` (`d_h`), `BROADCAST_PER_HOP_VERTICAL` (`d_v`), `BROADCAST_INTRA_TILE_CORE_OFFSET`, `BROADCAST_INTRA_TILE_MEM_OFFSET`. The runtime override reuses `xdna_archspec::types::BroadcastTiming` (fields `per_hop_horizontal`, `per_hop_vertical`, `intra_tile_core_offset`, `intra_tile_mem_offset`, `calibrated`, all `u8` except `calibrated: bool`).
- Default (override unset) MUST be byte-identical to today: `calibrated` stays `false`, all four constants `0`. The three neutrality guards (`crates/xdna-archspec/src/runtime.rs:807`, `src/interpreter/engine/coordinator.rs:4078`, `src/device/state/effects.rs:1355`) stay green.
- **Falsification requires >=3 collinear observations per axis.** Two points fit any line with zero residual; the fit-residual can only detect per-hop non-uniformity with >=3. Every extractor's tests must include a >=3-point non-uniformity case AND a 2-point case documenting that it cannot fire.
- **Sign convention:** the decoded trace carries the *reflected* origin offset (`max_delay - module_delay`); `origin_D`/`BroadcastTiming` is `module_delay` directly. Cross-domain differences cancel `max_delay` but invert the sign. The R1 extractor reconciles this and a unit test pins it.
- `cargo test --lib` stays green after every Rust task. No emoji. Commit messages end with the two-line trailer:
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```
- Spec: `docs/superpowers/specs/2026-06-30-sp5b-measurement-apparatus-design.md`.

---

## File Structure

**Rust (emulator):**
- `src/device/state/mod.rs` -- `DeviceState` struct (line 72) + `new()` (lines 131-146): add the override field.
- `src/device/state/effects.rs` -- `propagate_broadcasts` (563-569): add `effective_broadcast_timing` + `set_broadcast_timing_override`; rewire the flood. Task-1 + Task-4 tests append to its `#[cfg(test)]` modules.
- `src/interpreter/engine/coordinator.rs` -- `export_origin_d_sidecar` (378-408): consult the override.

**Python (extraction, all new under `tools/calibration/skew/`):**
- `__init__.py` -- makes `calibration.skew` importable.
- `_solve.py` -- shared design-matrix least-squares core (Task 2).
- `schema.py` -- `skew_constants.json` read/write (Task 2).
- `r1_extract.py` -- R1 within-column extractor (Task 2).
- `r3b_extract.py` -- R3b interval-difference extractor (Task 3).
- `tools/test_skew_r1_extract.py`, `tools/test_skew_r3b_extract.py` -- pytest (Task 2/3).

---

## Task 1: Runtime-override seam

**Files:**
- Modify: `src/device/state/mod.rs:72` (struct field), `src/device/state/mod.rs:136-145` (`new()` init)
- Modify: `src/device/state/effects.rs:563-569` (flood) + new methods in `impl DeviceState`
- Modify: `src/interpreter/engine/coordinator.rs:378-408` (`export_origin_d_sidecar`)
- Test: `src/device/state/effects.rs` (new `#[cfg(test)]` test)

**Interfaces:**
- Consumes: existing `propagate_broadcasts_with_timing(&mut self, col, source_row, d_h, d_v, core_off, mem_off)` (effects.rs:571).
- Produces:
  - `DeviceState.broadcast_timing_override: Option<xdna_archspec::types::BroadcastTiming>`
  - `pub fn set_broadcast_timing_override(&mut self, timing: Option<xdna_archspec::types::BroadcastTiming>)`
  - `pub(crate) fn effective_broadcast_timing(&self) -> (u32, u32, u32, u32, bool)` returning `(d_h, d_v, core_off, mem_off, calibrated)`.

- [ ] **Step 1: Write the failing test** (append to the `broadcast_origin_offset_tests` module in `src/device/state/effects.rs`, which already has `use super::*;`):

```rust
    #[test]
    fn override_drives_flood_timing() {
        use xdna_archspec::types::BroadcastTiming;
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let bcast_id = EventModuleType::Core.broadcast_event_base() + channel; // 112 + 5
        let src = (0u8, 2u8);
        let hop = (0u8, 3u8); // one vertical hop north
        for &(c, r) in &[src, hop] {
            dev.array.get_mut(c, r).unwrap()
                .core_timer.write_register(0x000, (bcast_id as u32) << 8);
        }
        dev.array.get_mut(src.0, src.1).unwrap()
            .pending_broadcasts.push(PendingBroadcast::originated(channel));
        // Inject d_v = 4 via the runtime override, then flood through the
        // CONST-reading path (propagate_broadcasts, NOT _with_timing).
        dev.set_broadcast_timing_override(Some(BroadcastTiming {
            per_hop_horizontal: 0, per_hop_vertical: 4,
            intra_tile_core_offset: 0, intra_tile_mem_offset: 0,
            calibrated: false,
        }));
        dev.propagate_broadcasts(src.0, src.1);
        dev.array.get_mut(src.0, src.1).unwrap().core_timer.tick();
        dev.array.get_mut(hop.0, hop.1).unwrap().core_timer.tick();
        let v_src = dev.array.get(src.0, src.1).unwrap().core_timer.value();
        let v_hop = dev.array.get(hop.0, hop.1).unwrap().core_timer.value();
        assert_eq!(v_src - v_hop, 4, "override d_v must drive the flood");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib override_drives_flood_timing`
Expected: FAIL -- `set_broadcast_timing_override` does not exist (compile error), or once stubbed, `v_src - v_hop == 0` (flood used the zero consts).

- [ ] **Step 3: Add the override field to `DeviceState`**

In `src/device/state/mod.rs`, add to the struct (after line 121, `channel15_flood_sources`):
```rust
    /// Runtime override for the broadcast-timing constants (SP-5b, #140). When
    /// `None` (default), the flood and sidecar read the compile-time archspec
    /// consts (all zero, uncalibrated). When `Some`, it supersedes them for a
    /// run -- the measurement apparatus injects known constants this way.
    pub broadcast_timing_override: Option<xdna_archspec::types::BroadcastTiming>,
```
And in `new()` (the struct literal at lines 136-145), add:
```rust
            broadcast_timing_override: None,
```

- [ ] **Step 4: Add the seam methods and rewire the flood**

In `src/device/state/effects.rs`, replace `propagate_broadcasts` (563-569) with:
```rust
    /// Effective broadcast-timing 4-tuple + calibrated: the runtime override if
    /// set, else the compile-time archspec constants. (SP-5b, #140.)
    pub(crate) fn effective_broadcast_timing(&self) -> (u32, u32, u32, u32, bool) {
        if let Some(o) = &self.broadcast_timing_override {
            (o.per_hop_horizontal as u32, o.per_hop_vertical as u32,
             o.intra_tile_core_offset as u32, o.intra_tile_mem_offset as u32, o.calibrated)
        } else {
            use xdna_archspec::aie2::timing as bt;
            (bt::BROADCAST_PER_HOP_HORIZONTAL as u32, bt::BROADCAST_PER_HOP_VERTICAL as u32,
             bt::BROADCAST_INTRA_TILE_CORE_OFFSET as u32, bt::BROADCAST_INTRA_TILE_MEM_OFFSET as u32,
             bt::BROADCAST_CALIBRATED)
        }
    }

    /// Set (or clear) the runtime broadcast-timing override. (SP-5b, #140.)
    pub fn set_broadcast_timing_override(&mut self, timing: Option<xdna_archspec::types::BroadcastTiming>) {
        self.broadcast_timing_override = timing;
    }

    pub(crate) fn propagate_broadcasts(&mut self, col: u8, source_row: u8) {
        let (d_h, d_v, core_off, mem_off, _cal) = self.effective_broadcast_timing();
        self.propagate_broadcasts_with_timing(col, source_row, d_h, d_v, core_off, mem_off);
    }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test --lib override_drives_flood_timing`
Expected: PASS.

- [ ] **Step 6: Write the sidecar-override test and default-neutrality test**

Append to the same test module:
```rust
    #[test]
    fn effective_timing_defaults_to_zero_consts() {
        let dev = DeviceState::new_npu1();
        assert_eq!(dev.effective_broadcast_timing(), (0, 0, 0, 0, false));
    }
```
And in `src/interpreter/engine/coordinator.rs`'s sidecar test module (near `export_origin_d_sidecar_matches_contract`, ~4064), add:
```rust
    #[test]
    fn export_origin_d_sidecar_reflects_override() {
        use xdna_archspec::types::BroadcastTiming;
        let mut engine = InterpreterEngine::new_npu1();
        engine.device_mut().set_broadcast_timing_override(Some(BroadcastTiming {
            per_hop_horizontal: 0, per_hop_vertical: 3,
            intra_tile_core_offset: 0, intra_tile_mem_offset: 0,
            calibrated: true,
        }));
        {
            let tile = engine.device_mut().array.get_mut(0, 0).expect("shim (0,0)");
            tile.pending_broadcasts.push(PendingBroadcast::originated(15));
        }
        engine.device_mut().propagate_broadcasts(0, 0);
        let v = engine.export_origin_d_sidecar();
        assert_eq!(v["calibrated"], serde_json::json!(true), "override calibrated must surface");
        // A tile one vertical hop from the (0,0) shim source has origin_D d_v = 3.
        assert_eq!(v["modules"]["0|1|memtile"], serde_json::json!(3),
                   "override d_v=3 must drive the sidecar origin_D: {v}");
    }
```

- [ ] **Step 7: Run to verify the sidecar test fails, then wire the sidecar**

Run: `cargo test --lib export_origin_d_sidecar_reflects_override`
Expected: FAIL (`calibrated` is `false`, origin_D is `0` -- the sidecar still reads consts).

In `src/interpreter/engine/coordinator.rs`, `export_origin_d_sidecar` (378-408): replace the four `bt::BROADCAST_*` args to `origin_d_table` and the `"calibrated": bt::BROADCAST_CALIBRATED` line with values from the override. Insert at the top of the method body:
```rust
        let (d_h, d_v, core_off, mem_off, calibrated) = self.device.effective_broadcast_timing();
```
then use `d_h, d_v, core_off, mem_off` as the `origin_d_table` args (dropping the `bt::` casts) and `"calibrated": calibrated`. The `use xdna_archspec::aie2::timing as bt;` line may be removed if no longer referenced.

- [ ] **Step 8: Run the full suite and commit**

Run: `cargo test --lib`
Expected: PASS (all prior tests green, including the three neutrality guards, plus the four new tests).

```bash
git add src/device/state/mod.rs src/device/state/effects.rs src/interpreter/engine/coordinator.rs
git commit -m "feat(#140): SP-5b Task 1 -- runtime broadcast-timing override seam

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 2: Shared solver + R1 extractor + schema

**Files:**
- Create: `tools/calibration/skew/__init__.py`, `tools/calibration/skew/_solve.py`, `tools/calibration/skew/schema.py`, `tools/calibration/skew/r1_extract.py`
- Test: `tools/test_skew_r1_extract.py`

**Interfaces:**
- Produces:
  - `_solve.solve_design_matrix(A, b, min_rank) -> (x: np.ndarray, fit_residual: float)`; raises `_solve.RankDeficientError`.
  - `r1_extract.extract_r1(observations, reflected=True) -> {"d_v", "intra_core", "intra_mem", "fit_residual"}` where `observations` is a list of `{"dn_v": int, "kind": str, "origin": float}` (`origin` = module origin relative to the source, source = 0).
  - `schema.write_constants(path, **kw)`, `schema.read_constants(path)`, `schema.empty_constants()`.

- [ ] **Step 1: Write the failing tests** (`tools/test_skew_r1_extract.py`):

```python
"""R1 within-column skew extraction tests (#140 SP-5b)."""
import math
import pytest
from calibration.skew.r1_extract import extract_r1
from calibration.skew._solve import RankDeficientError


# Emulator model (effects.rs:527-532): the intra offset applied to a module's
# origin_D is core_off for {core, shim} and mem_off for {mem, memtile}.
def _obs(dn_v, kind, d_v, intra_core, intra_mem, reflected=True, extra=0.0):
    off = intra_core if kind in ("core", "shim") else intra_mem  # {mem, memtile}
    origin_d = dn_v * d_v + off + extra
    return {"dn_v": dn_v, "kind": kind, "origin": (-origin_d if reflected else origin_d)}


def test_recovers_dv_and_intra():
    d_v, ic, im = 3.0, 2.0, 4.0
    obs = [_obs(n, "core", d_v, ic, im) for n in (2, 3, 4, 5)] + [_obs(1, "mem", d_v, ic, im)]
    r = extract_r1(obs)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-6)
    assert math.isclose(r["intra_core"], ic, abs_tol=1e-6)
    assert math.isclose(r["intra_mem"], im, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_fit_residual_grows_on_nonuniform_with_three_points():
    d_v, ic, im = 3.0, 2.0, 4.0
    obs = [_obs(2, "core", d_v, ic, im), _obs(3, "core", d_v, ic, im),
           _obs(4, "core", d_v, ic, im, extra=5.0), _obs(1, "mem", d_v, ic, im)]
    r = extract_r1(obs)
    assert r["fit_residual"] > 1.0, "non-uniform per-hop must produce a large residual"


def test_two_points_cannot_falsify():
    # Only 2 core points + 1 mem point: core sub-system is exactly determined,
    # so a non-uniform input still fits with ~0 residual. Documents why >=3.
    d_v, ic, im = 3.0, 2.0, 4.0
    obs = [_obs(2, "core", d_v, ic, im), _obs(4, "core", d_v, ic, im, extra=5.0),
           _obs(1, "mem", d_v, ic, im)]
    r = extract_r1(obs)
    assert r["fit_residual"] < 1e-6, "2 collinear points cannot detect non-uniformity"


def test_sign_convention_reflected_gives_positive_dv():
    # Reflected origins are negative for positive origin_D; extractor must flip.
    obs = [{"dn_v": n, "kind": "core", "origin": -(n * 3.0)} for n in (1, 2, 3)]
    r = extract_r1(obs, reflected=True)
    assert math.isclose(r["d_v"], 3.0, abs_tol=1e-6)


def test_rank_deficient_fails_loud():
    obs = [{"dn_v": 0, "kind": "core", "origin": 0.0} for _ in range(3)]
    with pytest.raises(RankDeficientError):
        extract_r1(obs)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd tools && python -m pytest test_skew_r1_extract.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'calibration.skew'`.

- [ ] **Step 3: Create the shared solver** (`tools/calibration/skew/__init__.py` empty; `tools/calibration/skew/_solve.py`):

```python
"""Shared design-matrix least-squares core for skew extraction (#140 SP-5b)."""
import numpy as np


class RankDeficientError(ValueError):
    """Raised when the design matrix cannot identify the requested parameters."""


def solve_design_matrix(A, b, min_rank):
    """Least-squares solve of A @ x = b.

    A: (n_obs, n_params) array-like; b: (n_obs,) array-like.
    Returns (x, fit_residual) where fit_residual = ||A @ x - b||_2.
    Raises RankDeficientError if rank(A) < min_rank (geometry cannot identify).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError(f"shape mismatch: A={A.shape}, b={b.shape}")
    if np.linalg.matrix_rank(A) < min_rank:
        raise RankDeficientError(f"design-matrix rank {np.linalg.matrix_rank(A)} < required {min_rank}")
    x, _res, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    fit_residual = float(np.linalg.norm(A @ x - b))
    return x, fit_residual
```

- [ ] **Step 4: Create the R1 extractor** (`tools/calibration/skew/r1_extract.py`):

```python
"""R1 within-column skew extraction: solve {d_v, intra_core, intra_mem} from
within-column cross-domain trace residuals (#140 SP-5b).

Sign convention: the decoded trace carries the REFLECTED origin offset
(max_delay - module_delay); origin_D (BroadcastTiming) is module_delay directly.
Cross-domain differences cancel max_delay but INVERT the sign, so with
reflected=True we negate each origin before solving to report in origin_D form.
Falsification of per-hop uniformity requires >=3 collinear observations per axis
(2 points fit any line with zero residual).

Kind grouping mirrors the emulator's origin_d_table (effects.rs:527-532): the
intra offset is core_off for {core, shim} and mem_off for {mem, memtile}, so the
extractor fits real emu/silicon data directly. (Provisional emulator model; SP-5c
may revise the grouping.)
"""
from ._solve import solve_design_matrix

# params order: [d_v, intra_core, intra_mem].
_CORE_GROUP = {"core", "shim"}
_MEM_GROUP = {"mem", "memtile"}


def _design_row(dn_v, kind):
    return [float(dn_v),
            1.0 if kind in _CORE_GROUP else 0.0,
            1.0 if kind in _MEM_GROUP else 0.0]


def extract_r1(observations, reflected=True):
    """observations: list of {"dn_v": int, "kind": str, "origin": float}
    (origin = module origin relative to the source, source = 0).
    Returns {"d_v", "intra_core", "intra_mem", "fit_residual"}."""
    A, b = [], []
    for o in observations:
        A.append(_design_row(o["dn_v"], o["kind"]))
        b.append(-o["origin"] if reflected else o["origin"])
    x, resid = solve_design_matrix(A, b, min_rank=2)
    return {"d_v": float(x[0]), "intra_core": float(x[1]),
            "intra_mem": float(x[2]), "fit_residual": resid}
```

- [ ] **Step 5: Run to verify the R1 tests pass**

Run: `cd tools && python -m pytest test_skew_r1_extract.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Add the schema module and its round-trip test**

Create `tools/calibration/skew/schema.py`:
```python
"""skew_constants.json read/write -- the SP-5b -> SP-5c handoff schema (#140)."""
import json


def empty_constants():
    return {"d_h": None, "d_v": None,
            "intra": {"core": None, "mem": None},
            "fit_residual": None, "source_route": None, "provenance": None}


def write_constants(path, *, d_h=None, d_v=None, intra_core=None, intra_mem=None,
                    fit_residual=None, source_route=None, provenance=None):
    obj = {"d_h": d_h, "d_v": d_v,
           "intra": {"core": intra_core, "mem": intra_mem},
           "fit_residual": fit_residual,
           "source_route": source_route, "provenance": provenance}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return obj


def read_constants(path):
    with open(path) as f:
        return json.load(f)
```
Add to `tools/test_skew_r1_extract.py`:
```python
def test_schema_round_trip(tmp_path):
    from calibration.skew.schema import write_constants, read_constants, empty_constants
    assert empty_constants()["intra"] == {"core": None, "mem": None}
    p = tmp_path / "skew_constants.json"
    written = write_constants(p, d_h=1.0, d_v=3.0, intra_core=2.0, intra_mem=4.0,
                              fit_residual=0.0, source_route="r1", provenance="measured-silicon")
    assert read_constants(p) == written
    assert written["intra"] == {"core": 2.0, "mem": 4.0}
```

- [ ] **Step 7: Run to verify, then commit**

Run: `cd tools && python -m pytest test_skew_r1_extract.py -v`
Expected: PASS (6 tests).

```bash
git add tools/calibration/skew/__init__.py tools/calibration/skew/_solve.py \
        tools/calibration/skew/r1_extract.py tools/calibration/skew/schema.py \
        tools/test_skew_r1_extract.py
git commit -m "feat(#140): SP-5b Task 2 -- shared solver + R1 extractor + handoff schema

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 3: R3b interval-difference extractor

**Files:**
- Create: `tools/calibration/skew/r3b_extract.py`
- Test: `tools/test_skew_r3b_extract.py`

**Interfaces:**
- Consumes: `_solve.solve_design_matrix` (Task 2).
- Produces: `r3b_extract.extract_r3b(observations, reference=0) -> {"d_h", "d_v", "fit_residual"}` where `observations` is a list of `{"dn_h": int, "dn_v": int, "r": float}` (`r` = per-tile two-flood interval reading).

- [ ] **Step 1: Write the failing tests** (`tools/test_skew_r3b_extract.py`):

```python
"""R3b interval-difference skew extraction tests (#140 SP-5b)."""
import math
import pytest
from calibration.skew.r3b_extract import extract_r3b
from calibration.skew._solve import RankDeficientError


def _r(dn_h, dn_v, d_h, d_v, const, extra=0.0):
    return {"dn_h": dn_h, "dn_v": dn_v, "r": const + dn_h * d_h + dn_v * d_v + extra}


def test_recovers_dh_dv():
    d_h, d_v, const = 2.0, 3.0, 100.0
    obs = [_r(0, 0, d_h, d_v, const), _r(1, 0, d_h, d_v, const), _r(2, 0, d_h, d_v, const),
           _r(0, 1, d_h, d_v, const), _r(0, 2, d_h, d_v, const)]
    r = extract_r3b(obs)
    assert math.isclose(r["d_h"], d_h, abs_tol=1e-6)
    assert math.isclose(r["d_v"], d_v, abs_tol=1e-6)
    assert r["fit_residual"] < 1e-6


def test_const_cancels():
    d_h, d_v = 2.0, 3.0
    obs_a = [_r(0, 0, d_h, d_v, 100.0), _r(1, 0, d_h, d_v, 100.0), _r(0, 1, d_h, d_v, 100.0)]
    obs_b = [_r(0, 0, d_h, d_v, 999.0), _r(1, 0, d_h, d_v, 999.0), _r(0, 1, d_h, d_v, 999.0)]
    ra, rb = extract_r3b(obs_a), extract_r3b(obs_b)
    assert math.isclose(ra["d_h"], rb["d_h"], abs_tol=1e-6)
    assert math.isclose(ra["d_v"], rb["d_v"], abs_tol=1e-6)


def test_fit_residual_grows_on_nonuniform_with_three_points():
    d_h, d_v, const = 2.0, 3.0, 0.0
    obs = [_r(0, 0, d_h, d_v, const), _r(1, 0, d_h, d_v, const), _r(2, 0, d_h, d_v, const, extra=5.0),
           _r(0, 1, d_h, d_v, const), _r(0, 2, d_h, d_v, const)]
    r = extract_r3b(obs)
    assert r["fit_residual"] > 1.0


def test_rank_deficient_fails_loud():
    # All tiles on the vertical axis only -> d_h unidentifiable.
    obs = [{"dn_h": 0, "dn_v": n, "r": float(n)} for n in range(3)]
    with pytest.raises(RankDeficientError):
        extract_r3b(obs)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd tools && python -m pytest test_skew_r3b_extract.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'calibration.skew.r3b_extract'`.

- [ ] **Step 3: Create the R3b extractor** (`tools/calibration/skew/r3b_extract.py`):

```python
"""R3b interval-difference skew extraction: solve {d_h, d_v} from per-tile
two-flood interval readings r_X = const + dn_h*d_h + dn_v*d_v (#140 SP-5b).

const (= T0_2 - T0_1) is removed by differencing against a reference tile.
Because differencing removes one degree of freedom, >=3 collinear tiles per axis
are needed to detect per-hop non-uniformity via the fit residual.
"""
from ._solve import solve_design_matrix


def extract_r3b(observations, reference=0):
    """observations: list of {"dn_h": int, "dn_v": int, "r": float}.
    reference: index differenced against to drop const.
    Returns {"d_h", "d_v", "fit_residual"}."""
    ref = observations[reference]
    A, b = [], []
    for i, o in enumerate(observations):
        if i == reference:
            continue
        A.append([float(o["dn_h"] - ref["dn_h"]), float(o["dn_v"] - ref["dn_v"])])
        b.append(float(o["r"] - ref["r"]))
    x, resid = solve_design_matrix(A, b, min_rank=2)
    return {"d_h": float(x[0]), "d_v": float(x[1]), "fit_residual": resid}
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `cd tools && python -m pytest test_skew_r3b_extract.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/calibration/skew/r3b_extract.py tools/test_skew_r3b_extract.py
git commit -m "feat(#140): SP-5b Task 3 -- R3b interval-difference extractor

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Task 4: Emu trace-origin plumbing check

**Goal:** prove a seam-injected constant (Task 1) surfaces in the *decoded* trace -- the "SP-2 origin surfaces in the decoded trace" half of the emu inject-and-recover loop, scoped to what is testable without a kernel. (The full "extractor on a real multi-module kernel trace" contract belongs to the kernel/HW bring-up plan.)

**Files:**
- Test: `src/device/state/effects.rs` (new `#[cfg(test)]` test), reusing the mode-1 decode helper.

**Interfaces:**
- Consumes: `set_broadcast_timing_override` (Task 1); the flood + Start-byte emit pattern in `flood_sets_origin_offset_before_arming_reached_trace` (`src/device/state/effects.rs:1406-1435`) -- which arms the source `core_trace` on the broadcast event, floods, and reads `core_trace.origin_offset()` + `core_trace.encoded_bytes()` (the Start's 7-byte big-endian absolute is decoded by the same fold `tools/trace_decoder` uses).

This is a regression/plumbing test: it needs NO new production code (Task 1 built the seam; the trace path already works). It proves the seam drives the trace origin offset end to end. Model it directly on the reference test at `effects.rs:1406-1435`, differing only in that it floods through the seam (`propagate_broadcasts` + override) instead of `propagate_broadcasts_with_timing` literals, and it decodes the Start absolute rather than only bit-checking it.

- [ ] **Step 1: Write the test** (append to the `broadcast_origin_offset_tests` module in `src/device/state/effects.rs`, `use super::*;`):

```rust
    #[test]
    fn override_origin_offset_surfaces_in_encoded_start() {
        use xdna_archspec::types::BroadcastTiming;
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let bcast_id = EventModuleType::Core.broadcast_event_base() + channel;
        let src = (0u8, 2u8);
        // Arm the source core trace to START on the broadcast event (as in the
        // reference test), so the flood's own notify emits the Start frame.
        dev.array.get_mut(src.0, src.1).unwrap()
            .core_trace.write_register(0x00, (bcast_id as u32) << 16);
        dev.array.get_mut(src.0, src.1).unwrap()
            .pending_broadcasts.push(PendingBroadcast::originated(channel));
        // Inject a nonzero intra-tile offset via the SEAM (consts are all zero,
        // so any nonzero decoded offset proves the override drove the flood).
        // NOTE the inversion: core_target = max_delay - (origin_d + core_off), so
        // the module with the LARGER offset gets the SMALLER target. With
        // d_h=d_v=0, giving CORE the offset would make it the max-delay module and
        // zero out the core_trace's own offset (the module this test arms/reads).
        // So mem carries the nonzero offset -> core is the faster module -> its
        // origin_offset = max_delay > 0. (Same inversion the reference test relies
        // on: it uses mem_off=4 > core_off=2.)
        dev.set_broadcast_timing_override(Some(BroadcastTiming {
            per_hop_horizontal: 0, per_hop_vertical: 0,
            intra_tile_core_offset: 0, intra_tile_mem_offset: 2, calibrated: false,
        }));
        dev.propagate_broadcasts(src.0, src.1); // seam path (const-reading entry point)

        let tile = dev.array.get(src.0, src.1).unwrap();
        let off = tile.core_trace.origin_offset();
        assert!(off > 0, "seam override must drive a nonzero trace origin offset");
        let bytes = tile.core_trace.encoded_bytes();
        assert_eq!(bytes[0] & 0xF0, 0xF0, "flood's arming notify emits a Start frame");
        // Decode the Start's 56-bit absolute exactly as tools/trace_decoder does.
        let decoded_start = (0..7).fold(0u64, |v, i| (v << 8) | bytes[1 + i] as u64);
        assert_eq!(decoded_start, off,
                   "injected offset must surface in the decoded Start absolute");
    }
```

- [ ] **Step 2: Confirm the test is meaningful (temporary red)**

To prove the assertion detects the seam's effect (not a tautology), temporarily comment out the `set_broadcast_timing_override(...)` call and run:

Run: `cargo test --lib override_origin_offset_surfaces_in_encoded_start`
Expected: FAIL -- `off` is `0` (zero consts), so `assert!(off > 0)` fails. Restore the override call.

- [ ] **Step 3: Run to verify it passes**

Run: `cargo test --lib override_origin_offset_surfaces_in_encoded_start`
Expected: PASS.

- [ ] **Step 4: Run the full suite and commit**

Run: `cargo test --lib`
Expected: PASS.

```bash
git add src/device/state/effects.rs
git commit -m "test(#140): SP-5b Task 4 -- injected constant surfaces in decoded trace

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Notes for the kernel/HW bring-up plan (out of scope here)

Recorded so the boundary is explicit, not forgotten:
- The R1 within-column kernel, R3b PerfCounter kernel, R3b `LDA_TM` kernel (hand-authored `write32` register programming -- second flood + perf-counter config, net-new, not SP-3 reuse).
- HW runnability gates (SP-3 Task-3 style) for all three kernels.
- The cross-language contract: the extractors (`r1_extract`/`r3b_extract`) recovering injected constants from a *real emu/HW multi-module trace* -- deferred here because it needs the R1 kernel to produce a realistic multi-tile trace. The data contract the kernel must emit is defined by the `observations` shapes in Tasks 2/3.
- `LDA_TM` kernel MUST never run inside the R1 differential loop (it reads its own timer -> breaks `Delta_wall` cancellation).
- R3b routing must guarantee `s1` arrives before `s2` at every measured tile.
