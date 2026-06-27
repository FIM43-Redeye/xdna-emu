# Multi-Column Trace Capture (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trace-capture pipeline capture kernels spanning more than one
hardware column, validated end-to-end on `two_col`, plus the mechanical
kernel-agnostic bits (insts discovery, `start_col` from the dump, dump fallback).

**Architecture:** Approach A — the batch is expressed in ABSOLUTE column space;
`configure_batch` is the single place that reconciles relative↔absolute (emitting
`patch_spec` in relative col and `label_map` keyed by absolute col);
`label_events` becomes a pure absolute-col lookup with no `traced_col` and no
foreign-column guard.

**Tech Stack:** Python 3.13 stdlib + pytest (run from `tools/`); Rust (the config
extractor `examples/dump_config_json.rs`, tested via `cargo test`).

**Spec:** `docs/superpowers/specs/2026-06-26-multi-column-trace-capture-design.md`

## Global Constraints

- **Approach A:** batch in ABSOLUTE col; `configure_batch(batch, anchor, mode, start_col)` emits `patch_spec` col = `abs - start_col` (RELATIVE, for the patcher) and `label_map` keyed `(pkt, row, abs_col, slot)`; `label_events(raw, label_map)` looks up by `(pkt, row, col, slot)` with the decoder's absolute col.
- **The `probe_slot_capacity` gate stays relative-col** in `HwInstrument` (it reads `insts.bin`, which is relative-col space). Only the patcher-facing conversion moves into `configure_batch`. Never feed absolute col to `probe_slot_capacity`.
- **Unconfigured-slot hard error is the single invariant:** every decoded event must resolve in `label_map` or `label_events` raises `CaptureError`. No tolerant drop path (YAGNI).
- **Connectivity = disconnected-but-honest:** SP1 does not require the two columns to causally connect; every cross-column oracle coupling must be either grounded as a `CrossTrackEdge` or surfaced as a `connectivity_defect` flag — never silently dropped.
- **`start_col` in the dump is sourced from xclbin partition metadata (`AiePartition::start_columns()[0]`), set AFTER the CDO/insts config-write application** (which shifts by `start_col`) so it is pure reporting metadata and does not corrupt tile data. `run_experiment` uses `dump.start_col` only when present-and-non-zero, else the config value (default 1).
- **Box-agnostic:** no quiet-host dependency; SP1 validates structure, not determinism fidelity.
- **TDD throughout; run Python tests from `tools/`; commit per task.**

---

## File Structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `tools/trace_capture.py` | label layer (`configure_batch`, `label_events`, `capture`), insts discovery, (remove dead `run_loop`/`SEED_ACTIVE_PLAN`) | 1,2,4 |
| `tools/inference/hw_instrument.py` | drive capture; abs batches to plan, rel col to probe gate | 2 |
| `tools/inference/run_experiment.py` | `KernelConfig`, CLI, dump fallback, `start_col` preference | 2,3,5,6 |
| `tools/canary_witness.py` | drop vestigial `traced_col` | 3 |
| `tools/config_extract/dump_model.py` | read `start_col` from dump JSON | 6 |
| `examples/dump_config_json.rs` | emit `start_col` into `ConfigDump` | 6 |
| `tools/config_extract/fixtures/two_col.config.json` | regenerated with `start_col` | 6 |
| `tools/capture_infer_smoke.py` | remove (uses dead `run_loop`) | 1 |
| Test files | per-task (see tasks) | all |

---

### Task 1: Remove the dead relative-col `run_loop` path

Removes `run_loop`, `SEED_ACTIVE_PLAN`, and `capture_infer_smoke.py`. These are
dead on the live `run_experiment` path (only `capture_infer_smoke` uses them) and
`run_loop` calls `capture(..., traced_col=...)`, which Task 2 will break. Removing
them first keeps every later task green.

**Files:**
- Modify: `tools/trace_capture.py` (delete `run_loop`, `SEED_ACTIVE_PLAN`)
- Delete: `tools/capture_infer_smoke.py`
- Test: `tools/test_trace_capture.py` (no test covers `run_loop`; verify suite green)

**Interfaces:**
- Consumes: nothing.
- Produces: a `trace_capture.py` whose only `capture()` caller is `HwInstrument`.

- [ ] **Step 1: Confirm nothing live depends on `run_loop`/`SEED_ACTIVE_PLAN`**

Run: `cd tools && grep -rn "run_loop\|SEED_ACTIVE_PLAN" . --include=*.py`
Expected: references only in `trace_capture.py` (definitions) and
`capture_infer_smoke.py` (the only caller). If anything else appears, STOP and
report.

- [ ] **Step 2: Delete `capture_infer_smoke.py`**

Run: `git rm tools/capture_infer_smoke.py`

- [ ] **Step 3: Delete `run_loop` and `SEED_ACTIVE_PLAN` from `trace_capture.py`**

Remove the `SEED_ACTIVE_PLAN = {...}` constant (the `0|0|2`/`0|1|3`/`0|2|0`/`0|2|1`
add_one block) and the entire `def run_loop(...)` function. Leave `HwRunner`,
`build_active_plan`, `capture`, `configure_batch`, `label_events`,
`_discover_xclbin_insts` intact.

- [ ] **Step 4: Run the full offline suite**

Run: `cd tools && python -m pytest -q -k "trace_capture or hw_instrument or experiment or timeline or trace_join or inference or canary"`
Expected: PASS (no new failures vs the pre-task baseline; the ~44 pre-existing
ISA/sweep failures are unrelated and out of scope).

- [ ] **Step 5: Commit**

```bash
git add -A tools/
git commit -m "refactor(#140): remove dead relative-col run_loop capture path"
```

---

### Task 2: Multi-column label layer + capture path

The core change. `configure_batch` becomes the single rel↔abs reconcile point;
`label_events` drops the foreign-column guard and `traced_col`; `capture` and
`HwInstrument` thread `start_col`; the `probe_slot_capacity` gate stays relative.

**Files:**
- Modify: `tools/trace_capture.py` (`configure_batch`, `label_events`, `capture`)
- Modify: `tools/inference/hw_instrument.py` (`capture`, `__init__`)
- Modify: `tools/inference/run_experiment.py` (stop passing `traced_col` to `HwInstrument`)
- Test: `tools/test_trace_capture.py`, `tools/test_hw_instrument.py`

**Interfaces:**
- Consumes: Task 1's cleaned `trace_capture.py`.
- Produces:
  - `configure_batch(batch, anchor="PERF_CNT_2", mode=0, start_col=0) -> (patch_spec, label_map)` — `label_map` keyed `(pkt, row, abs_col, slot)`; `patch_spec` col = `abs - start_col`.
  - `label_events(raw_events, label_map) -> List[dict]` — lookup `(pkt, row, col, slot)`; unconfigured → `CaptureError`.
  - `capture(plan, runner, *, test, out_dir, start_col=1, trace_size=TRACE_SIZE_DEFAULT, instr=None, inputs=(), outputs=()) -> List[label_map]`.
  - `HwInstrument.__init__(...)` without `traced_col`.

- [ ] **Step 1: Write failing test — multi-column collision + absolute lookup**

Add to `tools/test_trace_capture.py`:

```python
def test_configure_batch_multicolumn_no_collision_and_relative_patch():
    # two_col's collision: cores at absolute 1|2|0 and 2|2|0 both write slot 0.
    batch = {"1|2|0": ["PERF_CNT_2"], "2|2|0": ["PERF_CNT_2"]}
    spec, lmap = tc.configure_batch(batch, anchor="PERF_CNT_2", start_col=1)
    # label_map is keyed by ABSOLUTE col -> no collision, both survive
    assert lmap[(0, 2, 1, 0)] == "PERF_CNT_2"
    assert lmap[(0, 2, 2, 0)] == "PERF_CNT_2"
    # patch_spec is in RELATIVE col (abs - start_col): cols 0 and 1
    patch_cols = sorted(s["col"] for s in spec)
    assert patch_cols == [0, 1]


def test_label_events_absolute_col_lookup_two_columns():
    lmap = {(0, 2, 1, 0): "PERF_CNT_2", (0, 2, 2, 0): "PERF_CNT_2"}
    raw = [_raw(1, 2, 0, 0, 100, 100), _raw(2, 2, 0, 0, 200, 200)]
    out = tc.label_events(raw, lmap)
    assert {(e["col"], e["name"]) for e in out} == {(1, "PERF_CNT_2"), (2, "PERF_CNT_2")}


def test_label_events_unconfigured_is_hard_error_no_col_guard():
    lmap = {(0, 2, 1, 0): "PERF_CNT_2"}
    import pytest
    # column 2 is NOT a "foreign column" error anymore -- it's unconfigured (not in map)
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(2, 2, 0, 0, 100, 100)], lmap)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python -m pytest test_trace_capture.py -q -k "multicolumn or absolute_col_lookup or unconfigured_is_hard_error_no_col"`
Expected: FAIL (`configure_batch` has no `start_col` kwarg / `label_map` keyed by 3-tuple; `label_events` requires `traced_col`).

- [ ] **Step 3: Implement `configure_batch` (absolute-keyed map, relative patch_spec)**

In `tools/trace_capture.py`, change the signature and the two keyed lines:

```python
def configure_batch(batch: Dict[str, List[str]], anchor: str = "PERF_CNT_2",
                    mode: int = 0, start_col: int = 0):
    # ... docstring: batch keys are ABSOLUTE col; patch_spec is RELATIVE col
    #     (abs - start_col) for the patcher; label_map is keyed by ABSOLUTE col.
    patch_spec = []
    label_map: Dict[tuple, str] = {}
    for tile_key, names in batch.items():
        col, row, pkt = (int(x) for x in tile_key.split("|"))   # col is ABSOLUTE
        tile_type = PKT_TO_TILE_TYPE[pkt]
        ordered = ([anchor] if anchor in names else []) + [n for n in names if n != anchor]
        if len(ordered) > 8:
            raise ValueError(f"tile {tile_key} has {len(ordered)} events > 8 slots")
        ids = load_event_ids(tile_type)
        event_ids = []
        for slot, name in enumerate(ordered):
            if name not in ids:
                raise ValueError(f"event {name!r} not in {tile_type} table")
            event_ids.append(ids[name])
            label_map[(pkt, row, col, slot)] = name          # ABSOLUTE-col key
        event_ids += [0] * (8 - len(event_ids))
        patch_spec.append({"col": col - start_col, "row": row,  # RELATIVE col
                           "tile_type": tile_type, "events": event_ids, "mode": mode})
    return patch_spec, label_map
```

- [ ] **Step 4: Implement `label_events` (drop guard + traced_col, absolute lookup)**

```python
def label_events(raw_events, label_map) -> List[dict]:
    """Apply label_map to raw decoded events. Each decoded event carries its
    ABSOLUTE col (the decoder reports absolute col). Every event must resolve in
    label_map (keyed (pkt,row,col,slot)) or it is a hard error -- the single
    uniform invariant that also catches mis-decoded streams."""
    out = []
    for ev in raw_events:
        col = _get(ev, "col"); row = _get(ev, "row"); pkt = _get(ev, "pkt_type")
        slot = _get(ev, "slot")
        key = (pkt, row, col, slot)
        if key not in label_map:
            raise CaptureError(f"event at unconfigured (pkt,row,col,slot)={key}")
        out.append({"col": col, "row": row, "pkt_type": pkt,
                    "name": label_map[key], "slot": slot,
                    "ts": _get(ev, "ts"), "soc": _get(ev, "soc"),
                    "mode": _get(ev, "mode")})
    return out
```

- [ ] **Step 5: Implement `capture` (traced_col → start_col)**

In `capture`, change the signature `traced_col=1` → `start_col=1`, pass
`start_col` to `configure_batch`, and call `label_events` without it:

```python
def capture(plan, runner, *, test, out_dir, start_col=1,
            trace_size=TRACE_SIZE_DEFAULT, instr=None, inputs=(), outputs=()):
    # ...
    for i, batch in enumerate(plan["batches"]):
        spec, lmap = configure_batch(batch, start_col=start_col)
        # ... (patch/run/decode unchanged) ...
        events = label_events(raw, lmap)
        # ... write unchanged ...
```

- [ ] **Step 6: Update the existing label-layer tests to the new shapes**

In `tools/test_trace_capture.py`:
- `test_configure_batch_anchor_first_and_label_map`: call `tc.configure_batch(batch, anchor="PERF_CNT_2", start_col=1)`; assert `lmap[(0, 2, 1, 0)] == "PERF_CNT_2"` and `lmap[(0, 2, 1, 1)] == "LOCK_STALL"` and `lmap[(2, 0, 1, 0)] == "DMA_S2MM_0_START_TASK"`; the core patch entry now has `col == 0` (relative).
- `test_configure_batch_pads_patch_spec_to_8_slots_with_none`: call with `start_col=1`; assert `set(lmap) == {(0, 2, 1, 0), (0, 2, 1, 1)}`.
- `test_label_events_applies_map`: `lmap = {(0, 2, 1, 0): "PERF_CNT_2", (0, 2, 1, 1): "LOCK_STALL"}`; call `tc.label_events(raw, lmap)` (no `traced_col`).
- `test_label_events_unconfigured_slot_is_hard_error`: `lmap = {(0, 2, 1, 0): "PERF_CNT_2"}`; call `tc.label_events([_raw(1, 2, 0, 5, 100, 100)], lmap)`.
- **Delete** `test_label_events_foreign_column_is_hard_error` (the guard is gone; replaced by `test_label_events_unconfigured_is_hard_error_no_col_guard` from Step 1).
- `test_capture_writes_labeled_events_per_batch`: call `tc.capture(plan, FakeRunner(), test="add_one_using_dma", out_dir=tmp_path, start_col=1)`; the `fake_parse` returns a col-1 event, so the labeled name still resolves.

- [ ] **Step 7: Implement `HwInstrument` — absolute batches to plan, relative col to probe**

In `tools/inference/hw_instrument.py`:
- Drop `traced_col` from `__init__` (remove the param and `self._traced_col`).
- Rewrite `capture()` to keep batches ABSOLUTE for `build_active_plan` while
  computing relative col only for the probe gate:

```python
def capture(self, batch: Batch) -> List[str]:
    xclbin, insts = _discover_xclbin_insts(self._test, self._compiler)
    insts_bytes = Path(insts).read_bytes()

    # Drop tiles the xclbin compiled WITHOUT trace. probe_slot_capacity reads
    # insts.bin, which is in RELATIVE col space -> probe with abs - start_col.
    # The plan/anchor stay in ABSOLUTE col (configure_batch does the abs->rel
    # for the patcher).
    traceable_abs: Dict[str, set] = {}
    for tile_abs, names in batch.tiles.items():
        col_a, row_s, pkt_s = tile_abs.split("|")
        col_rel = int(col_a) - self._start_col
        tile_type = PKT_TO_TILE_TYPE[int(pkt_s)]
        if probe_slot_capacity(insts_bytes, col_rel, int(row_s), tile_type) > 0:
            traceable_abs.setdefault(tile_abs, set()).update(names)

    plan = build_active_plan(traceable_abs, anchor=self._anchor_event,
                             anchor_tile=self._anchor_tile_abs)   # ABS anchor tile
    run_dirs: List[str] = []
    base = self._out_root / f"capture_{self._iter:02d}"
    for i in range(self._n_runs):
        rd = base / f"run_{i:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        runner = HwRunner(xclbin, stderr_log=rd / "hw.runner.log")
        try:
            capture(plan, runner, test=self._test, out_dir=rd,
                    start_col=self._start_col, instr=insts)
        finally:
            runner.close()
        run_dirs.append(str(rd))
    self._iter += 1
    return run_dirs
```

Also remove the now-unused `_abs_to_rel` method.

- [ ] **Step 8: Stop passing `traced_col` to `HwInstrument` in `run_experiment`**

In `tools/inference/run_experiment.py`, the `HwInstrument(...)` construction
drops `traced_col=cfg.traced_col`. (Leave `KernelConfig.traced_col` and the CLI
flag for now — Task 3 removes them.)

- [ ] **Step 9: Update `test_hw_instrument.py` fakes**

`tools/test_hw_instrument.py`'s `fake_capture` signature changes from
`(plan, runner, *, test, out_dir, traced_col, instr)` to
`(plan, runner, *, test, out_dir, start_col, instr)`, and any `HwInstrument(...)`
construction drops `traced_col=`. Assertions that batches reach the plan in
absolute col now hold without the abs→rel conversion (the planner already emits
absolute).

- [ ] **Step 10: Run to verify passing + suite**

Run: `cd tools && python -m pytest test_trace_capture.py test_hw_instrument.py -q`
Then: `cd tools && python -m pytest -q -k "experiment or timeline or trace_join or inference or canary"`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add tools/trace_capture.py tools/inference/hw_instrument.py tools/inference/run_experiment.py tools/test_trace_capture.py tools/test_hw_instrument.py
git commit -m "feat(#140): multi-column label layer (Approach A) + capture path"
```

---

### Task 3: Remove the vestigial `traced_col` config surface

After Task 2 `traced_col` is unused. Remove it from `KernelConfig`, the CLI, and
`canary_witness`.

**Files:**
- Modify: `tools/inference/run_experiment.py` (`KernelConfig.traced_col`, `--traced-col`, the `KernelConfig(...)` construction in `main`)
- Modify: `tools/canary_witness.py` (`KernelConfig(..., traced_col=1, ...)`)
- Test: `tools/test_experiment_report.py`, `tools/test_experiment_loop_hw.py`, `tools/test_canary_witness.py` (any `KernelConfig(..., traced_col=...)`)

**Interfaces:**
- Consumes: Task 2's `HwInstrument` (no `traced_col`).
- Produces: `KernelConfig` without `traced_col`.

- [ ] **Step 1: Grep the full `traced_col` surface**

Run: `cd tools && grep -rn "traced_col\|traced-col" . --include=*.py`
Expected after this task: zero matches. Use this list to drive the edits.

- [ ] **Step 2: Remove the field, CLI flag, and construction usages**

- `run_experiment.py`: delete `traced_col: int` from `KernelConfig`; delete
  `ap.add_argument("--traced-col", ...)`; delete `traced_col=a.traced_col` from
  the `KernelConfig(...)` build in `main`.
- `canary_witness.py`: delete `traced_col=1,` from the `KernelConfig(...)` in
  `_capture_sentinel_runs`.

- [ ] **Step 3: Update any test constructing `KernelConfig` with `traced_col`**

Remove `traced_col=...` from `KernelConfig(...)` in
`tools/test_experiment_report.py`, `tools/test_experiment_loop_hw.py`, and any
other file the Step-1 grep surfaced.

- [ ] **Step 4: Run to verify**

Run: `cd tools && grep -rn "traced_col\|traced-col" . --include=*.py` (expect none)
Then: `cd tools && python -m pytest -q -k "experiment or canary or timeline or hw_instrument or trace_capture"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/run_experiment.py tools/canary_witness.py tools/test_experiment_report.py tools/test_experiment_loop_hw.py
git commit -m "refactor(#140): remove vestigial traced_col config surface"
```

---

### Task 4: Generalize insts discovery

`_discover_xclbin_insts` hardcodes `insts.bin`. Generalize: prefer `insts.bin`,
else parse the test's `run.lit` for `--npu-insts-name=`, else a single `*.bin`.

**Files:**
- Modify: `tools/trace_capture.py` (`_discover_xclbin_insts`)
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Produces: `_discover_xclbin_insts(test, compiler="chess") -> (Path, Path)` resolving the insts file by the rule above; raises `CaptureError` on none/ambiguous.

- [ ] **Step 1: Write failing tests over synthetic build dirs**

Add to `tools/test_trace_capture.py`. These need to control the build-dir root;
add a `build_root` parameter to the helper (Step 3 introduces it) so tests don't
touch the real mlir-aie tree:

```python
def _mk_build(tmp_path, name, files, runlit=None):
    d = tmp_path / "build" / "test" / "npu-xrt" / name / "chess"
    d.mkdir(parents=True)
    (d / "aie.xclbin").write_bytes(b"x")
    for f in files:
        (d / f).write_bytes(b"i")
    if runlit is not None:
        (tmp_path / "build" / "test" / "npu-xrt" / name).joinpath("run.lit").write_text(runlit)
    return tmp_path / "build"


def test_discover_insts_prefers_insts_bin(tmp_path):
    root = _mk_build(tmp_path, "k", ["insts.bin", "other.bin"])
    _x, insts = tc._discover_xclbin_insts("k", build_root=root)
    assert insts.name == "insts.bin"


def test_discover_insts_parses_runlit(tmp_path):
    root = _mk_build(tmp_path, "k", ["k_insts.bin"],
                     runlit="// RUN: ... --npu-insts-name=k_insts.bin ...")
    _x, insts = tc._discover_xclbin_insts("k", build_root=root)
    assert insts.name == "k_insts.bin"


def test_discover_insts_single_bin_fallback(tmp_path):
    root = _mk_build(tmp_path, "k", ["aie_run_seq.bin"])
    _x, insts = tc._discover_xclbin_insts("k", build_root=root)
    assert insts.name == "aie_run_seq.bin"


def test_discover_insts_ambiguous_is_error(tmp_path):
    import pytest
    root = _mk_build(tmp_path, "k", ["a.bin", "b.bin"])   # no insts.bin, no run.lit
    with pytest.raises(tc.CaptureError):
        tc._discover_xclbin_insts("k", build_root=root)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python -m pytest test_trace_capture.py -q -k "discover_insts"`
Expected: FAIL (`_discover_xclbin_insts` has no `build_root` param; hardcodes `insts.bin`).

- [ ] **Step 3: Implement the generalized discovery**

```python
def _discover_xclbin_insts(test, compiler="chess", build_root=None):
    """Locate (xclbin, insts) for a test. insts resolution order:
       1) insts.bin if present;
       2) --npu-insts-name=<name> parsed from the test's run.lit;
       3) the single *.bin in the build dir.
    Raises CaptureError on missing xclbin or none/ambiguous insts."""
    import re
    root = Path(build_root) if build_root else (_MLIR_AIE / "build" / "test" / "npu-xrt")
    build_dir = root / test / compiler
    xclbin = build_dir / "aie.xclbin"
    if not xclbin.is_file():
        raise CaptureError(f"xclbin not found: {xclbin} (is the kernel built?)")
    cand = build_dir / "insts.bin"
    if cand.is_file():
        return xclbin, cand
    runlit = root / test / "run.lit"
    if runlit.is_file():
        m = re.search(r"--npu-insts-name=(\S+)", runlit.read_text())
        if m and (build_dir / m.group(1)).is_file():
            return xclbin, build_dir / m.group(1)
    bins = sorted(build_dir.glob("*.bin"))
    if len(bins) == 1:
        return xclbin, bins[0]
    raise CaptureError(
        f"cannot resolve insts for {test}: no insts.bin, no run.lit hit, "
        f"and {len(bins)} *.bin candidates in {build_dir}")
```

- [ ] **Step 4: Run to verify passing + suite**

Run: `cd tools && python -m pytest test_trace_capture.py test_hw_instrument.py -q`
Expected: PASS (`HwInstrument` calls `_discover_xclbin_insts(self._test, self._compiler)` — the new `build_root=None` default preserves the real path).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): generalize insts discovery (insts.bin / run.lit / glob)"
```

---

### Task 5: Dump fallback by test name

When `dump_path` is None, `run_experiment` should try
`config_extract/fixtures/{test}.config.json` before degrading to `dump=None`.

**Files:**
- Modify: `tools/inference/run_experiment.py`
- Test: `tools/test_experiment_report.py`

**Interfaces:**
- Produces: a helper `_resolve_dump_path(cfg) -> Optional[str]` used by `run_experiment`.

- [ ] **Step 1: Write failing test**

Add to `tools/test_experiment_report.py`:

```python
def test_resolve_dump_path_falls_back_to_fixture_by_test_name(tmp_path, monkeypatch):
    from inference import run_experiment as re
    fixtures = tmp_path / "config_extract" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "mykernel.config.json").write_text("{}")
    monkeypatch.setattr(re, "_FIXTURES_DIR", fixtures)
    cfg = re.KernelConfig(test="mykernel", compiler="chess", dump_path=None,
                          start_col=1, anchor_tile_abs="1|2|0",
                          anchor_event="PERF_CNT_2", n_runs=1, out_root=str(tmp_path))
    assert re._resolve_dump_path(cfg) == str(fixtures / "mykernel.config.json")


def test_resolve_dump_path_prefers_explicit(tmp_path):
    from inference import run_experiment as re
    cfg = re.KernelConfig(test="k", compiler="chess", dump_path="/explicit.json",
                          start_col=1, anchor_tile_abs="1|2|0",
                          anchor_event="PERF_CNT_2", n_runs=1, out_root=str(tmp_path))
    assert re._resolve_dump_path(cfg) == "/explicit.json"
```

(Note: `KernelConfig` no longer has `traced_col` after Task 3.)

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python -m pytest test_experiment_report.py -q -k "resolve_dump_path"`
Expected: FAIL (`_resolve_dump_path` / `_FIXTURES_DIR` undefined).

- [ ] **Step 3: Implement**

In `tools/inference/run_experiment.py`:

```python
_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "config_extract" / "fixtures"


def _resolve_dump_path(cfg) -> Optional[str]:
    if cfg.dump_path:
        return cfg.dump_path
    cand = _FIXTURES_DIR / f"{cfg.test}.config.json"
    return str(cand) if cand.is_file() else None
```

Then in `run_experiment`, replace `dump = load_dump(cfg.dump_path) if cfg.dump_path else None`
with `dp = _resolve_dump_path(cfg); dump = load_dump(dp) if dp else None`, and use
`dp` in place of `cfg.dump_path` for the engine dump load too.

- [ ] **Step 4: Run to verify passing**

Run: `cd tools && python -m pytest test_experiment_report.py -q -k "resolve_dump_path"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/run_experiment.py tools/test_experiment_report.py
git commit -m "feat(#140): dump fallback to fixtures/<test>.config.json"
```

---

### Task 6: `start_col` in the dump (Rust + Python)

The Rust extractor emits the partition `start_col` into `ConfigDump`; Python reads
it; `run_experiment` prefers it when present-and-non-zero.

**CRITICAL:** `state.start_col` shifts the CDO/insts config-write application
(`src/device/state/cdo.rs:164`, `apply_config_writes_from_insts`). Set it ONLY
after both `apply_cdo` and `apply_config_writes_from_insts` have run, so it is
pure reporting metadata and does not corrupt tile data. `resolve_route_graph` /
`build_dump` do not read `start_col`, so setting it at the end of
`load_state_from_xclbin` is safe and needs no caller changes.

**Files:**
- Modify: `examples/dump_config_json.rs` (`ConfigDump` struct, `build_dump`, `load_state_from_xclbin`)
- Modify: `tools/config_extract/dump_model.py` (`ConfigDump`, `load_dump`)
- Modify: `tools/inference/run_experiment.py` (prefer `dump.start_col`)
- Regenerate: `tools/config_extract/fixtures/two_col.config.json`
- Test: a Rust `#[cfg(test)]` test in `examples/dump_config_json.rs`; `tools/test_config_extract.py` (or the dump_model test module) for the Python read; `tools/test_experiment_report.py` for the preference.

**Interfaces:**
- Produces: `ConfigDump.start_col: Optional[int]` (Python); `dump.start_col` consumed by `run_experiment`.

- [ ] **Step 1 (Rust): Write failing test — dump carries partition start_col, tiles unshifted**

In `examples/dump_config_json.rs` tests, add (mirror an existing
`load_state_from_xclbin` + `build_dump` test for the two_col xclbin path; reuse
that test's xclbin/insts path resolution):

```rust
#[test]
fn dump_reports_partition_start_col_without_shifting_tiles() {
    // two_col partition start_columns()[0] == 1.
    let state = load_state_from_xclbin(TWO_COL_XCLBIN, Some(TWO_COL_INSTS)).expect("load");
    let dump = build_dump(&state);
    assert_eq!(dump.start_col, Some(1));
    // tiles remain at logical/relative positions (col 0 present, not shifted to 1)
    assert!(dump.tiles.iter().any(|t| t.col == 0));
}
```

(Use the same `TWO_COL_XCLBIN`/`TWO_COL_INSTS` path constants the existing tests
use; if only add_one constants exist, add two_col ones pointing at
`mlir-aie/build/test/npu-xrt/two_col/chess/{aie.xclbin,insts.bin}`, and `#[ignore]`
the test if the build is absent, matching the existing skip pattern.)

- [ ] **Step 2 (Rust): Run to verify failure**

Run: `cargo test --example dump_config_json dump_reports_partition_start_col`
Expected: FAIL (`ConfigDump` has no `start_col` field).

- [ ] **Step 3 (Rust): Implement**

- Add to `ConfigDump`: `pub start_col: Option<u8>,` with
  `#[serde(skip_serializing_if = "Option::is_none")]`.
- In `build_dump`, set `start_col: if state.start_col == 0 { None } else { Some(state.start_col) }`.
- In `load_state_from_xclbin`, right after parsing the partition
  (`let partition = AiePartition::parse(...)?;`), capture
  `let start_col_meta = partition.start_columns().first().copied().unwrap_or(0) as u8;`
  and, just before `Ok(state)`, call `state.set_start_col(start_col_meta);`
  (after `apply_cdo` and `apply_config_writes_from_insts`).

- [ ] **Step 4 (Rust): Run to verify passing + lib suite**

Run: `cargo test --example dump_config_json`
Then: `cargo test --lib`
Expected: PASS. (If a pre-existing example test asserted full-dump equality, add
the `start_col` field to its expected value.)

- [ ] **Step 5 (Rust): Commit**

```bash
git add examples/dump_config_json.rs
git commit -m "feat(#140): emit partition start_col into config dump (reporting-only)"
```

- [ ] **Step 6 (Python): Write failing test — dump_model reads start_col, absent → None**

In the dump_model test module (`tools/test_config_extract.py` or wherever
`load_dump` is tested):

```python
def test_load_dump_reads_start_col(tmp_path):
    from config_extract.dump_model import load_dump
    p = tmp_path / "d.json"
    p.write_text('{"device":"npu1","start_col":1,"route_graph":{"edges":[]},"tiles":[]}')
    assert load_dump(p).start_col == 1


def test_load_dump_start_col_absent_is_none(tmp_path):
    from config_extract.dump_model import load_dump
    p = tmp_path / "d.json"
    p.write_text('{"device":"npu1","route_graph":{"edges":[]},"tiles":[]}')
    assert load_dump(p).start_col is None
```

- [ ] **Step 7 (Python): Run to verify failure**

Run: `cd tools && python -m pytest -q -k "load_dump_reads_start_col or start_col_absent"`
Expected: FAIL (`ConfigDump` has no `start_col`).

- [ ] **Step 8 (Python): Implement**

In `tools/config_extract/dump_model.py`:
- Add to `ConfigDump`: `start_col: Optional[int] = None`.
- In `load_dump`, pass `start_col=raw.get("start_col")` to the `ConfigDump(...)` build.

- [ ] **Step 9 (Python): `run_experiment` prefers dump.start_col when present-and-non-zero**

Add a failing test to `tools/test_experiment_report.py`:

```python
def test_start_col_prefers_dump_when_present(tmp_path):
    from inference import run_experiment as re
    from config_extract.dump_model import ConfigDump, RouteGraph
    dump = ConfigDump(device="npu1", route_graph=RouteGraph(edges=()), tiles=(), start_col=2)
    assert re._effective_start_col(dump, cfg_start_col=1) == 2


def test_start_col_falls_back_when_absent_or_zero(tmp_path):
    from inference import run_experiment as re
    from config_extract.dump_model import ConfigDump, RouteGraph
    none_dump = ConfigDump(device="npu1", route_graph=RouteGraph(edges=()), tiles=(), start_col=None)
    zero_dump = ConfigDump(device="npu1", route_graph=RouteGraph(edges=()), tiles=(), start_col=0)
    assert re._effective_start_col(none_dump, cfg_start_col=1) == 1
    assert re._effective_start_col(zero_dump, cfg_start_col=1) == 1
```

Implement in `run_experiment.py`:

```python
def _effective_start_col(dump, cfg_start_col: int) -> int:
    sc = getattr(dump, "start_col", None) if dump is not None else None
    return sc if sc else cfg_start_col   # None or 0 -> fall back to config
```

Then in `run_experiment`, after loading `dump`, compute
`start_col = _effective_start_col(dump, cfg.start_col)` and use it everywhere the
function currently uses `cfg.start_col` (enumerate, candidate_pairs, HwInstrument,
engine, anchor_key is unaffected — it is built from `anchor_tile_abs`).

- [ ] **Step 10 (Python): Run to verify + regenerate two_col fixture**

Run: `cd tools && python -m pytest -q -k "start_col or load_dump or resolve_dump"`
Expected: PASS.

Regenerate the two_col fixture so the preference path is exercised on real data:
```bash
cargo run --release --example dump_config_json -- \
  ../mlir-aie/build/test/npu-xrt/two_col/chess/aie.xclbin \
  ../mlir-aie/build/test/npu-xrt/two_col/chess/insts.bin \
  > tools/config_extract/fixtures/two_col.config.json
```
Verify it now contains `"start_col": 1` and the tile cols are unchanged
(relative). The other 3 fixtures are left as-is (absent `start_col` → None → config
fallback, correct for all).

- [ ] **Step 11 (Python): Commit**

```bash
git add tools/config_extract/dump_model.py tools/inference/run_experiment.py tools/test_config_extract.py tools/test_experiment_report.py tools/config_extract/fixtures/two_col.config.json
git commit -m "feat(#140): read+prefer dump start_col; regenerate two_col fixture"
```

---

### Task 7: Synthetic two-column offline timeline test (the DoD lock)

Prove the multi-column claim WITHOUT HW: hand-built two-column run dirs through
`assemble_timeline`, asserting the discriminating predicate and the
disconnected-but-honest connectivity policy.

**Files:**
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: `assemble_timeline(run_dirs, configured, derives_pairs, cross_domain_pairs, dump=None, start_col=1, anchor_key=ANCHOR, capture=None)` (existing), `coupling_oracle`, the `Track.domain` shape `"col|row|pkt"`.

- [ ] **Step 1: Write the failing multi-column timeline test**

Add to `tools/test_timeline.py` (reuse the existing `_write_run` / `_ev` helpers;
note `_ev` defaults col=1, pass `col=2` for the second column). Build two columns
each with the anchor (so column-2 events anchor against the same-batch anchor) and
a fired event, plus a cross-column candidate pair:

```python
def test_assemble_timeline_multicolumn_tracks_cover_both_columns(tmp_path):
    runs = []
    for i in range(3):
        runs.append(_write_run(tmp_path, f"run_{i}", [
            _ev("PERF_CNT_2", 0, col=1, row=2, pkt_type=0),       # anchor (col 1)
            _ev("INSTR_VECTOR", 10 + i, col=1, row=2, pkt_type=0),
            _ev("INSTR_VECTOR", 20 + i, col=2, row=2, pkt_type=0),  # column 2 fires
        ]))
    configured = ["1|2|0|PERF_CNT_2", "1|2|0|INSTR_VECTOR", "2|2|0|INSTR_VECTOR"]
    tl = T.assemble_timeline(runs, configured, derives_pairs=set(),
                             cross_domain_pairs=[], dump=None, start_col=1)
    cols = {tr.domain.split("|")[0] for tr in tl.tracks}
    assert "1" in cols and "2" in cols          # discriminating: both columns present
```

- [ ] **Step 2: Run to verify failure first, then confirm it is the predicate that matters**

Run: `cd tools && python -m pytest test_timeline.py -q -k "multicolumn_tracks_cover_both"`
Expected: with the engine already multi-column-ready, this likely PASSES on the
first run. That is acceptable for a characterization/lock test ONLY IF you first
confirm it would FAIL on a single-column input: temporarily change the second
`_ev(... col=2 ...)` to `col=1` and verify the `"2" in cols` assertion fails, then
revert. Record this check in the task report.

- [ ] **Step 3: Add the connectivity-honesty test**

```python
def test_assemble_timeline_crosscolumn_coupling_grounded_or_flagged(tmp_path):
    # A cross-column oracle coupling must be EITHER a CrossTrackEdge OR a
    # connectivity_defect flag -- never silently dropped (disconnected-but-honest).
    from config_extract.dump_model import ConfigDump, RouteGraph, RouteEdge, PortRef
    def pr(col, row): return PortRef(col=col, row=row, port=0, dir="out", kind="x")
    # relative-col dump: cols 0 and 1 -> absolute 1 and 2 at start_col=1
    rg = RouteGraph(edges=(RouteEdge(pr(0, 2), pr(1, 2), "inter_tile"),))
    dump = ConfigDump(device="npu1", route_graph=rg, tiles=(), start_col=1)
    runs = []
    for i in range(3):
        runs.append(_write_run(tmp_path, f"run_{i}", [
            _ev("PERF_CNT_2", 0, col=1, row=2, pkt_type=0),
            _ev("INSTR_VECTOR", 10 + i, col=1, row=2, pkt_type=0),
            _ev("INSTR_VECTOR", 20 + i, col=2, row=2, pkt_type=0),
        ]))
    configured = ["1|2|0|PERF_CNT_2", "1|2|0|INSTR_VECTOR", "2|2|0|INSTR_VECTOR"]
    tl = T.assemble_timeline(runs, configured, derives_pairs=set(),
                             cross_domain_pairs=[], dump=dump, start_col=1)
    oracle = T.coupling_oracle(dump, start_col=1)   # contains ("1|2","2|2")
    edge_pairs = {tuple(sorted((e.child.rsplit("|", 1)[0], e.parent.rsplit("|", 1)[0])))
                  for e in tl.cross_track_edges}
    for a, b in oracle:
        grounded = tuple(sorted((a, b))) in edge_pairs
        flagged = f"connectivity_defect:{a}~{b}" in tl.flags
        assert grounded or flagged, f"coupling {a}~{b} silently dropped"
```

- [ ] **Step 4: Run both tests**

Run: `cd tools && python -m pytest test_timeline.py -q -k "multicolumn or crosscolumn"`
Expected: PASS. If the connectivity test fails because the defect flag format
differs, align the assertion to the actual `connectivity_defects` flag string
(`flags` entries are `"connectivity_defect:<a>~<b>"` per `assemble_timeline`).

- [ ] **Step 5: Commit**

```bash
git add tools/test_timeline.py
git commit -m "test(#140): offline multi-column timeline + connectivity-honesty lock"
```

---

### Task 8: HW integration — `two_col` capture through `run_experiment`

The HW DoD (box-agnostic, structural). Not pure TDD — a validation procedure on
the real NPU. Touches HW: run ONLY when no other HW suite is active.

**Files:**
- Create: `build/experiments/two_col_capture/driver.py` (a thin reuse of the existing shakedown driver pattern, `--test two_col`)
- (No production code; this task validates Tasks 2/4/6 end-to-end.)

**Interfaces:**
- Consumes: `run_experiment(KernelConfig(test="two_col", ...))` and the resulting `report["timeline"]`.

- [ ] **Step 1: Pre-flight HW safety**

Run: `pgrep -af 'emu-bridge-test|isa-test|test.exe|run_experiment'` (expect none),
and confirm `/dev/accel/accel0` exists. If a HW suite is running, STOP.

- [ ] **Step 2: Capture two_col through run_experiment**

Run (real HW — `env -u XDNA_EMU`):
```bash
cd tools && env -u XDNA_EMU -u XDNA_EMU_RUNTIME python -c "
from inference.run_experiment import KernelConfig, run_experiment
cfg = KernelConfig(test='two_col', compiler='chess',
                   dump_path='config_extract/fixtures/two_col.config.json',
                   start_col=1, anchor_tile_abs='1|2|0', anchor_event='PERF_CNT_2',
                   n_runs=8, out_root='../build/experiments/two_col_capture/cap')
rep = run_experiment(cfg)
tl = rep['timeline']
cols = {tr.domain.split('|')[0] for tr in tl.tracks}
print('engine_ok', rep['engine_ok'], 'cols', sorted(cols))
print('tracks', [tr.domain for tr in tl.tracks])
print('edges', len(tl.cross_track_edges), 'flags', tl.flags)
print('census', tl.census.events)
assert rep['engine_ok'], 'engine did not complete'
assert '1' in cols and '2' in cols, f'timeline not multi-column: {cols}'
print('PASS: multi-column two_col timeline assembled')
"
```

Expected: completes with no `foreign column` / unconfigured-slot `CaptureError`;
prints `PASS`; tracks cover both columns 1 and 2.

- [ ] **Step 3: Handle the expected-possible first-run failure**

If Step 2 raises an unconfigured-slot `CaptureError` naming a `(col,row,pkt,slot)`,
that is a `two_col` compile-time-trace tile not in the route-graph menu (the
spec's documented residual). Add that tile to the seed coverage by confirming it
is route-graph-active (it should already be enumerated); if it is genuinely
outside the menu, NONE-disable it by ensuring it appears in `build_active_plan`'s
input (it will, if `enumerate_configured_events` lists it). Document the tile and
the resolution in the task report. Do NOT add a tolerant drop path.

- [ ] **Step 4: Record the connectivity outcome**

Note in the task report whether any cross-column edge grounded or whether the
columns are disconnected with `connectivity_defect` flags (the expected
disconnected-but-honest outcome). Either is an SP1 PASS; this records the input
to the immediate follow-up sub-project (cross-column edge grounding).

- [ ] **Step 5: Commit the driver**

```bash
git add build/experiments/two_col_capture/driver.py 2>/dev/null || true
git commit -m "test(#140): two_col HW integration validation (multi-column capture)" --allow-empty
```

(`build/` is gitignored; the `--allow-empty` records the validation milestone in
history even though the capture artifacts and driver are not tracked.)

---

## Notes for the executor

- Run Python tests from `tools/` (bare imports `inference.*`, `config_extract.*`,
  `trace_capture`).
- The ~44 pre-existing failures in `test_isa_test_gen.py` /
  `test_isa_multi_tile_gen.py` / `test_trace_sweep.py` are unrelated; the bar is
  no NEW failures.
- After the Rust change (Task 6), `cargo test --lib` and
  `cargo test --example dump_config_json` must stay green.
- Tasks 1→2→3 are ordered (each unblocks the next); Tasks 4, 5, 6, 7 are mutually
  independent; Task 8 depends on 2, 4, 6.
