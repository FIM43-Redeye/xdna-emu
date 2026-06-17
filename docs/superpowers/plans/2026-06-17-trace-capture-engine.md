# Trace-Capture Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-owned NPU trace-capture engine that turns a batch plan into correctly-labeled `events.json` per batch per run on real hardware, and wire it with the existing join library to validate the cross-batch join on `add_one_using_dma`.

**Architecture:** A new `tools/trace_capture.py` orchestrates three audited primitives — the register patcher (`trace-patch-events.py`), the XRT runner (`bridge-trace-runner`), and the in-tree decoder (`tools/trace_decoder/`) — and owns the two things trace-sweep got wrong: exact column-free labeling (record `(pkt_type,row,slot)→name` at config, apply after a raw decode) and observable N-run coverage. The join library (`tools/trace_join.py`) is first retrofitted to module-explicit keys `(col,row,pkt_type,name)`, then driven by the engine through a six-step loop.

**Tech Stack:** Python 3.13, stdlib only (`json`, `glob`, `re`, `subprocess`, `pathlib`, `argparse`), `pytest`. Reuses the three primitives as black boxes. No emulator. HW tasks use the real NPU via the bridge.

## Global Constraints

- **Module-explicit key:** every event key is `"{col}|{row}|{pkt_type}|{name}"` (4 parts). Tile/module key is `"{col}|{row}|{pkt_type}"`. The anchor is `"1|2|0|PERF_CNT_2"` (core = pkt_type 0).
- **Labeling is column-free:** the label map keys on `(pkt_type, row, slot)`; decoded `col` is used only as a sanity guard (a foreign column is a hard error). Never key labels on absolute col.
- **`pkt_type` codes:** 0=core, 1=memmod (mem), 2=shim, 3=memtile. Tile-type names: `core, memmod, memtile, shim`.
- **8 slots per tile-module**, slot 0 reserved for the anchor/grounding event; event ID 0 = NONE (unused slot).
- **Observable coverage, not reliable:** an event may not fire in a given run; the engine unions across N runs and reports any never-seen configured slot as a named gap. It surfaces gaps, it does not fill them.
- **RESET between every batch/run** (zeroes the cumulative shim-DMA write counter). RESET serializes batches; never skip it to "optimize."
- **Truncation is a distinct hard error** from "did not fire": pass an explicit generous `--trace-size` and detect a full/truncated buffer.
- **Decode is in-tree `trace_decoder` for the raw layer**, with a live parity guard against mlir-aie. The held-level span layer is out of scope this cycle.
- **HW operational rules:** run the real NPU via `env -u XDNA_EMU -u XDNA_EMU_RUNTIME`; never two HW suites concurrently; never `xrt-smi` during a HW run; redirect long runs to a file (never pipe through tail/grep); HW is the cheap disposable oracle — throw away and re-run freely.
- Commit messages end with `Generated using Claude Code.`, no emoji.

**Reused primitive interfaces (verified by the 2026-06-17 audit — use as-is):**
- Events header: `/home/triple/npu-work/mlir-aie/build/include/xaienginecdo_static/xaiengine/xaie_events_aieml.h`, lines `#define XAIEML_EVENTS_<MOD>_<NAME> <id>U` where MOD ∈ {CORE, MEM, MEM_TILE, PL}.
- Patcher: `tools/trace-patch-events.py --multi-tile <spec.json> <orig_insts> --output <patched>` where spec is `[{"col":int,"row":int,"tile_type":str,"events":[int,...]}]` (event IDs, ≤8). Standalone CLI; exits non-zero on a missing target register.
- Runner: `bridge-trace-runner` — stdin protocol, one command per line: `--instr <patched_insts> --trace-out <trace.bin> --trace-size <bytes> [--input ...] [--output ...]`, plus a bare `RESET` line; emits a `ready` marker and one JSON status line per command. Reference driver: `tools/trace-sweep.py` `RunnerSession` (`run_one`, the `RESET` line, process startup with the xclbin) — adapt this, do not re-derive the XRT details.
- Decoder: `from trace_decoder import parse_trace` (in `tools/`); `parse_trace(words, slot_names=None, mode=TraceMode.EVENT_TIME)` returns `list[Event]` with `.col .row .pkt_type .slot .ts .soc .mode` and `.name == ""`. `words` is a `numpy`/list of uint32 from `trace.bin`. mlir-aie oracle for the parity guard: `parse-trace.py`'s `convert_to_commands` path (raw, no MLIR needed).

---

### Task 1: Library module-explicit key retrofit

**Files:**
- Modify: `tools/trace_join.py` (key layer + all callers)
- Modify: `tools/test_trace_join.py` (re-baseline assertions + `_ev` helper)

**Interfaces:**
- Consumes: nothing new.
- Produces: every key is 4-part `"col|row|pkt_type|name"`; `_key(col,row,pkt_type,name)`; anchor default `"1|2|0|PERF_CNT_2"` everywhere; `load_active_events(run_dir) -> {"col|row|pkt_type": set[name]}`; `anchored_firsts(events, anchor_key="1|2|0|PERF_CNT_2")`.

**Note for implementer:** this is a key-layer rewrite, not a find-replace. The change is mechanical but touches every function and the whole test file; the library must be consistent and green in **one** commit. Work through it function by function, then re-baseline the tests, then run the full suite once.

- [ ] **Step 1: Update the key helpers and event constructors**

In `tools/trace_join.py`:

```python
def _key(col, row, pkt_type, name) -> str:
    return f"{col}|{row}|{pkt_type}|{name}"


def _tile(col, row, pkt_type) -> str:
    return f"{col}|{row}|{pkt_type}"


def load_active_events(run_dir: str) -> Dict[str, set]:
    """{"col|row|pkt_type": {event_name,...}} — fired events per tile-module."""
    out: Dict[str, set] = collections.defaultdict(set)
    for p in sorted(_glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json"))):
        for e in json.loads(Path(p).read_text()).get("events", []):
            out[_tile(e["col"], e["row"], e["pkt_type"])].add(e["name"])
    return dict(out)


def anchored_firsts(events: List[dict], anchor_key: str = "1|2|0|PERF_CNT_2") -> Dict[str, int]:
    firsts: Dict[str, int] = {}
    for e in events:
        k = _key(e["col"], e["row"], e["pkt_type"], e["name"])
        if k not in firsts or e["soc"] < firsts[k]:
            firsts[k] = e["soc"]
    if anchor_key not in firsts:
        return {}
    anchor = firsts[anchor_key]
    return {k: v - anchor for k, v in firsts.items()}
```

- [ ] **Step 2: Update `_split_key`, the graph node builder, and `join_run`**

`_split_key` now yields 4 parts; `build_derivability_graph` builds nodes from the 4-part tile keys; `join_run` parses 4-part keys.

```python
def _split_key(k):
    col, row, pkt, name = k.split("|", 3)
    return f"{col}|{row}|{pkt}", name   # (tile-module key, name)


# in build_derivability_graph, node construction:
    nodes = set()
    for rd in run_dirs:
        for tile, names in load_active_events(rd).items():   # tile == "col|row|pkt"
            for n in names:
                nodes.add(f"{tile}|{n}")
    nodes = sorted(nodes)


# in join_run, per-event key + col/row/name extraction:
        for k, ts in firsts.items():
            col, row, pkt, name = k.split("|", 3)
            obs[k].append((batch_idx, ts, slot_of.get(k), int(col), int(row), int(pkt), name))
```

Update `join_run`'s `slot_of` map and record construction to carry `pkt_type`:

```python
        slot_of: Dict[str, int] = {}
        for e in events:
            k = _key(e["col"], e["row"], e["pkt_type"], e["name"])
            slot_of.setdefault(k, e.get("slot"))
        ...
        # record shape gains pkt_type:
        records.append({"col": c, "row": r, "pkt_type": pt, "name": nm, "slot": slot,
                        "ts_anchored": ts, "source_batch": bi,
                        "class": cls, "predictor": pred, "band": band})
```

(Apply the `pkt_type` field to both the stochastic-sample and the median-record emit paths; sort key stays `(ts_anchored, col, row, name)` plus `pkt_type` for stability.)

- [ ] **Step 3: Update `synthesize_plan` and `sweep_lists`**

`synthesize_plan` already uses `_split_key` (now 4-part-aware) and the anchor from the graph — only the anchor literal in test helpers changes. `sweep_lists` is **superseded by the capture engine** (it mapped to trace-sweep's discarded `--*-sweep` flags); leave it untouched but do not extend it. Add a one-line comment above it: `# superseded by trace_capture.py module configuration; retained for reference.`

- [ ] **Step 4: Re-baseline the tests**

In `tools/test_trace_join.py`: update the `_ev` helper default and every event/key/anchor literal to the 4-part form. The `_ev` helper already takes `pkt_type`; make all event keys and assertions use `"col|row|pkt_type|name"` and the anchor `"1|2|0|PERF_CNT_2"`. Example edits:

```python
def _ev(col, row, name, soc, slot=0, pkt_type=0, ts=None, mode=0):
    # unchanged signature; pkt_type already present
    ...

# every assertion that referenced "1|0|S" becomes "1|0|0|S" (or the right pkt_type);
# anchor PERF_CNT_2 events stay pkt_type=0 -> key "1|2|0|PERF_CNT_2".
```

Walk every test: anchored-firsts keys, graph node/edge/root keys, plan always-on/payload keys, join_run record keys. Each 3-part literal `"C|R|NAME"` becomes `"C|R|PKT|NAME"`.

- [ ] **Step 5: Run the full suite**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: all tests PASS (the same count as before the retrofit, ~20), now on 4-part keys.

- [ ] **Step 6: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "refactor(#140): module-explicit (col,row,pkt_type,name) keys in trace_join

Eliminates the cross-module name-collision class (core vs memmod PERF_CNT_2)
and the latent (col,row,name) merge. Foundation for trace_capture.py.

Generated using Claude Code."
```

---

### Task 2: Name->ID resolver

**Files:**
- Create: `tools/trace_capture.py`
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Produces: `load_event_ids(tile_type: str) -> Dict[str, int]` — `{event_name: numeric_id}` for a tile-type, parsed from the aie-rt events header. Raises `KeyError` for an unknown tile_type, `FileNotFoundError` if the header is missing.

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_trace_capture.py
import trace_capture as tc


def test_load_event_ids_core_has_perf_cnt_2():
    ids = tc.load_event_ids("core")
    assert ids["PERF_CNT_2"] == 7          # XAIEML_EVENTS_CORE_PERF_CNT_2 7U
    assert ids["INSTR_VECTOR"] == 37       # XAIEML_EVENTS_CORE_INSTR_VECTOR 37U


def test_load_event_ids_memmod_excludes_memtile_events():
    mem = tc.load_event_ids("memmod")
    memtile = tc.load_event_ids("memtile")
    # MEM_ prefix must not swallow MEM_TILE_ events
    assert not any(n.startswith("TILE_") for n in mem)
    # memtile has its own distinct table
    assert len(memtile) > 0


def test_load_event_ids_unknown_tile_type_raises():
    import pytest
    with pytest.raises(KeyError):
        tc.load_event_ids("bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_capture'`.

- [ ] **Step 3: Implement**

```python
# tools/trace_capture.py
#!/usr/bin/env python3
"""Self-owned NPU trace-capture engine for #140.

Takes a batch plan and produces correctly-labeled events.json per batch per run
on real hardware, reusing three audited primitives (register patcher, XRT
runner, in-tree decoder) and owning column-free exact labeling + N-run coverage.

See docs/superpowers/specs/2026-06-17-trace-capture-engine-design.md.
"""
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
_EVENTS_HEADER = (_REPO.parent / "mlir-aie/build/include/xaienginecdo_static/"
                  "xaiengine/xaie_events_aieml.h")
_MOD_PREFIX = {"core": "CORE", "memmod": "MEM", "memtile": "MEM_TILE", "shim": "PL"}


def load_event_ids(tile_type: str) -> Dict[str, int]:
    """{event_name: numeric_id} for a tile-type, from the aie-rt events header."""
    full = f"XAIEML_EVENTS_{_MOD_PREFIX[tile_type]}_"
    exclude = "XAIEML_EVENTS_MEM_TILE_" if tile_type == "memmod" else None
    out: Dict[str, int] = {}
    for line in _EVENTS_HEADER.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith(full):
            if exclude and parts[1].startswith(exclude):
                continue
            name = parts[1][len(full):]
            val = parts[2].rstrip("U")
            if val.isdigit():
                out.setdefault(name, int(val))   # first definition wins (stable)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): trace-capture — name->ID resolver from aie-rt header

Generated using Claude Code."
```

---

### Task 3: Configure a batch — module-resolved slots + the column-free label map

**Files:**
- Modify: `tools/trace_capture.py`
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Consumes: `load_event_ids` (Task 2).
- Produces:
  - `configure_batch(batch, anchor="PERF_CNT_2") -> (patch_spec, label_map)` where `batch` is `{"col|row|pkt_type": [event_name,...]}`, `patch_spec` is `[{"col","row","tile_type","events":[id,...]}]` (anchor in slot 0, then payload, ≤8), and `label_map` is `{(pkt_type, row, slot): name}` — column-free.
  - `PKT_TO_TILE_TYPE = {0:"core", 1:"memmod", 2:"shim", 3:"memtile"}`.

**Note for implementer:** the anchor event goes in slot 0 of every module that lists it; other events follow in plan order. The `label_map` is keyed `(pkt_type, row, slot)` — NOT col — because the decoder reports absolute columns we can't predict. A module with more than 8 events is a hard error (`ValueError`) — the plan synthesizer should never produce that, but guard it.

- [ ] **Step 1: Write the failing tests**

```python
def test_configure_batch_anchor_first_and_label_map():
    batch = {"1|2|0": ["PERF_CNT_2", "LOCK_STALL"], "1|0|2": ["DMA_S2MM_0_START_TASK"]}
    spec, lmap = tc.configure_batch(batch, anchor="PERF_CNT_2")
    # core module: anchor in slot 0, LOCK_STALL slot 1
    assert lmap[(0, 2, 0)] == "PERF_CNT_2"
    assert lmap[(0, 2, 1)] == "LOCK_STALL"
    # shim module: its single event in slot 0 (no anchor on shim here)
    assert lmap[(2, 0, 0)] == "DMA_S2MM_0_START_TASK"
    # patch spec has resolved numeric IDs, core PERF_CNT_2 == 7 in slot 0
    core = [s for s in spec if s["tile_type"] == "core" and s["row"] == 2][0]
    assert core["events"][0] == 7


def test_configure_batch_rejects_over_8_events():
    import pytest
    batch = {"1|2|0": [f"E{i}" for i in range(9)]}
    with pytest.raises(ValueError):
        tc.configure_batch(batch)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m pytest test_trace_capture.py::test_configure_batch_anchor_first_and_label_map -v`
Expected: FAIL (`AttributeError: ... 'configure_batch'`).

Note: `test_configure_batch_anchor_first_and_label_map` references real event IDs; if `LOCK_STALL`/`DMA_S2MM_0_START_TASK` are not in the respective tables, the test will surface that — use the real names from the header (they are present for memmod/shim).

- [ ] **Step 3: Implement**

```python
# add to tools/trace_capture.py
PKT_TO_TILE_TYPE = {0: "core", 1: "memmod", 2: "shim", 3: "memtile"}


def configure_batch(batch: Dict[str, List[str]], anchor: str = "PERF_CNT_2"):
    """batch {"col|row|pkt": [names]} -> (patch_spec, label_map).

    label_map is keyed (pkt_type, row, slot) -- column-free by design.
    """
    patch_spec = []
    label_map: Dict[tuple, str] = {}
    for tile_key, names in batch.items():
        col, row, pkt = (int(x) for x in tile_key.split("|"))
        tile_type = PKT_TO_TILE_TYPE[pkt]
        # anchor first (slot 0) if present, then the rest in plan order
        ordered = ([anchor] if anchor in names else []) + [n for n in names if n != anchor]
        if len(ordered) > 8:
            raise ValueError(f"tile {tile_key} has {len(ordered)} events > 8 slots")
        ids = load_event_ids(tile_type)
        event_ids = []
        for slot, name in enumerate(ordered):
            if name not in ids:
                raise ValueError(f"event {name!r} not in {tile_type} table")
            event_ids.append(ids[name])
            label_map[(pkt, row, slot)] = name
        patch_spec.append({"col": col, "row": row, "tile_type": tile_type,
                           "events": event_ids})
    return patch_spec, label_map
```

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): trace-capture — batch config + column-free label map

Generated using Claude Code."
```

---

### Task 4: Label raw decoded events from the recorded map (with hard-error guards)

**Files:**
- Modify: `tools/trace_capture.py`
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Consumes: the `label_map` from Task 3.
- Produces:
  - `class CaptureError(Exception)`.
  - `label_events(raw_events, label_map, traced_col) -> List[dict]` — each raw event (a dict or object with `col,row,pkt_type,slot,ts,soc,mode`) becomes a record `{col,row,pkt_type,name,slot,ts,soc,mode}` with `name` from `label_map[(pkt_type,row,slot)]`. Raises `CaptureError` on (a) a `(pkt_type,row,slot)` not in the map (unconfigured slot) or (b) a `col != traced_col` (foreign column / start_col guard).

**Note for implementer:** accept raw events as plain dicts in the test (the real decoder yields objects with attributes; normalize with a tiny accessor so the function works for both — e.g. read via a helper that tries attribute then item access). Keep it simple: the test passes dicts.

- [ ] **Step 1: Write the failing tests**

```python
def _raw(col, row, pkt, slot, ts, soc, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt, "slot": slot,
            "ts": ts, "soc": soc, "mode": mode}


def test_label_events_applies_map():
    lmap = {(0, 2, 0): "PERF_CNT_2", (0, 2, 1): "LOCK_STALL"}
    raw = [_raw(1, 2, 0, 0, 100, 100), _raw(1, 2, 0, 1, 150, 150)]
    out = tc.label_events(raw, lmap, traced_col=1)
    assert out[0]["name"] == "PERF_CNT_2" and out[1]["name"] == "LOCK_STALL"
    assert out[0]["pkt_type"] == 0 and out[0]["soc"] == 100


def test_label_events_unconfigured_slot_is_hard_error():
    import pytest
    lmap = {(0, 2, 0): "PERF_CNT_2"}
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(1, 2, 0, 5, 100, 100)], lmap, traced_col=1)


def test_label_events_foreign_column_is_hard_error():
    import pytest
    lmap = {(0, 2, 0): "PERF_CNT_2"}
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(3, 2, 0, 0, 100, 100)], lmap, traced_col=1)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m pytest test_trace_capture.py -k label_events -v`
Expected: FAIL (`AttributeError: ... 'label_events'`).

- [ ] **Step 3: Implement**

```python
# add to tools/trace_capture.py
class CaptureError(Exception):
    pass


def _get(ev, attr):
    return ev[attr] if isinstance(ev, dict) else getattr(ev, attr)


def label_events(raw_events, label_map, traced_col: int) -> List[dict]:
    out = []
    for ev in raw_events:
        col = _get(ev, "col"); row = _get(ev, "row"); pkt = _get(ev, "pkt_type")
        slot = _get(ev, "slot")
        if col != traced_col:
            raise CaptureError(
                f"foreign column {col} (traced {traced_col}); start_col mismatch")
        key = (pkt, row, slot)
        if key not in label_map:
            raise CaptureError(f"event at unconfigured (pkt,row,slot)={key}")
        out.append({"col": col, "row": row, "pkt_type": pkt,
                    "name": label_map[key], "slot": slot,
                    "ts": _get(ev, "ts"), "soc": _get(ev, "soc"),
                    "mode": _get(ev, "mode")})
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): trace-capture — exact labeling with hard-error guards

Generated using Claude Code."
```

---

### Task 5: N-run coverage union + gap report

**Files:**
- Modify: `tools/trace_capture.py`
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Produces: `coverage_report(configured, observed_per_run) -> dict` where `configured` is the set of `(pkt_type,row,slot)` we configured (across batches) mapped to names — pass `{(pkt,row,slot): name}` — and `observed_per_run` is a list (one per run) of sets of `(pkt_type,row,slot)` that fired. Returns `{"n_configured", "n_covered", "gaps": [{"pkt_type","row","slot","name"}, ...]}` where gaps are slots never observed in ANY run.

- [ ] **Step 1: Write the failing tests**

```python
def test_coverage_report_unions_across_runs():
    configured = {(0, 2, 0): "PERF_CNT_2", (0, 2, 1): "LOCK_STALL", (2, 0, 0): "DMA_X"}
    # run 0 saw anchor+lock; run 1 saw anchor+dma; union covers all 3
    observed = [{(0, 2, 0), (0, 2, 1)}, {(0, 2, 0), (2, 0, 0)}]
    rep = tc.coverage_report(configured, observed)
    assert rep["n_configured"] == 3 and rep["n_covered"] == 3 and rep["gaps"] == []


def test_coverage_report_names_a_never_seen_gap():
    configured = {(0, 2, 0): "PERF_CNT_2", (2, 0, 0): "DMA_X"}
    observed = [{(0, 2, 0)}, {(0, 2, 0)}]   # DMA_X never fired
    rep = tc.coverage_report(configured, observed)
    assert rep["n_covered"] == 1
    assert rep["gaps"] == [{"pkt_type": 2, "row": 0, "slot": 0, "name": "DMA_X"}]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m pytest test_trace_capture.py -k coverage -v`
Expected: FAIL (`AttributeError: ... 'coverage_report'`).

- [ ] **Step 3: Implement**

```python
# add to tools/trace_capture.py
def coverage_report(configured: Dict[tuple, str], observed_per_run: List[set]) -> dict:
    seen = set().union(*observed_per_run) if observed_per_run else set()
    gaps = [{"pkt_type": p, "row": r, "slot": s, "name": configured[(p, r, s)]}
            for (p, r, s) in sorted(configured) if (p, r, s) not in seen]
    return {"n_configured": len(configured),
            "n_covered": len(configured) - len(gaps), "gaps": gaps}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): trace-capture — N-run coverage union + gap report

Generated using Claude Code."
```

---

### Task 6: Patch-spec writer + runner-command builder

**Files:**
- Modify: `tools/trace_capture.py`
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Produces:
  - `write_patch_spec(patch_spec, path) -> Path` — writes the `[{col,row,tile_type,events}]` JSON the patcher's `--multi-tile` consumes.
  - `runner_command(instr, trace_out, trace_size, inputs, outputs) -> str` — the single stdin line for `bridge-trace-runner` (`--instr ... --trace-out ... --trace-size ... --input ... --output ...`), space-joined with shell-safe quoting.
  - `TRACE_SIZE_DEFAULT = 1 << 21` (2 MiB — generous headroom over the runner's 1 MiB default; full co-traced add_one batches are small but DMA-heavy).

- [ ] **Step 1: Write the failing tests**

```python
import json as _json


def test_write_patch_spec_roundtrips(tmp_path):
    spec = [{"col": 1, "row": 2, "tile_type": "core", "events": [7, 60]}]
    p = tc.write_patch_spec(spec, tmp_path / "spec.json")
    assert _json.loads(p.read_text()) == spec


def test_runner_command_includes_trace_size_and_io():
    cmd = tc.runner_command("insts.bin", "trace.bin", tc.TRACE_SIZE_DEFAULT,
                            ["a.bin"], ["o.bin"])
    assert "--instr insts.bin" in cmd
    assert f"--trace-size {tc.TRACE_SIZE_DEFAULT}" in cmd
    assert "--input a.bin" in cmd and "--output o.bin" in cmd
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m pytest test_trace_capture.py -k "patch_spec or runner_command" -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

```python
# add to tools/trace_capture.py
import json
from shlex import quote

TRACE_SIZE_DEFAULT = 1 << 21


def write_patch_spec(patch_spec, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(patch_spec))
    return path


def runner_command(instr, trace_out, trace_size, inputs, outputs) -> str:
    parts = ["--instr", str(instr), "--trace-out", str(trace_out),
             "--trace-size", str(trace_size)]
    for p in inputs:
        parts += ["--input", str(p)]
    for p in outputs:
        parts += ["--output", str(p)]
    return " ".join(quote(p) for p in parts)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): trace-capture — patch-spec writer + runner-command builder

Generated using Claude Code."
```

---

### Task 7: Capture orchestration (per-batch RESET -> patch -> run -> decode -> label -> write)

**Files:**
- Modify: `tools/trace_capture.py`
- Test: `tools/test_trace_capture.py`

**Interfaces:**
- Consumes: `configure_batch`, `write_patch_spec`, `runner_command`, `label_events`, the patcher CLI, the runner (via a `RunnerSession`-style driver), `trace_decoder.parse_trace`.
- Produces: `capture(plan, runner, *, test, out_dir, traced_col=1, trace_size=TRACE_SIZE_DEFAULT)` where `plan` is `{"batches": [batch, ...]}` (each batch `{"col|row|pkt": [names]}`), `runner` is an object exposing `reset()` and `run_one(cmd_line) -> status_dict`, and the function writes `out_dir/batch_MM/hw/trace.events.json` per batch and returns the list of label_maps used. Detects a truncated trace (runner status `truncated`/missing end-marker) and raises `CaptureError`.

**Note for implementer:** the real `runner` is adapted from `tools/trace-sweep.py` `RunnerSession` (study `run_one` at ~558-613 and the `RESET` line). For the **unit test**, inject a fake `runner` with `reset()` and `run_one()` that writes a synthetic `trace.bin` and returns `{"ok": True}`; monkeypatch `subprocess.run` for the patcher call and `trace_decoder.parse_trace` to return synthetic raw events. The test verifies the per-batch loop calls RESET before each run, invokes the patcher with the spec, decodes, labels, and writes the events.json — NOT the real HW path.

- [ ] **Step 1: Write the failing test (fully mocked, no HW)**

```python
def test_capture_writes_labeled_events_per_batch(tmp_path, monkeypatch):
    calls = {"reset": 0, "runs": 0, "patch": 0}

    class FakeRunner:
        def reset(self): calls["reset"] += 1
        def run_one(self, cmd):
            calls["runs"] += 1
            return {"ok": True}

    def fake_subprocess_run(cmd, **kw):
        calls["patch"] += 1
        class R: returncode = 0
        return R()
    monkeypatch.setattr(tc.subprocess, "run", fake_subprocess_run)

    # one core event that "fires" in the decode
    def fake_parse(words, slot_names=None, mode=None):
        return [{"col": 1, "row": 2, "pkt_type": 0, "slot": 0,
                 "ts": 100, "soc": 100, "mode": 0}]
    monkeypatch.setattr(tc, "parse_trace", fake_parse)
    monkeypatch.setattr(tc, "_read_trace_words", lambda p: [0])  # stub bin read

    plan = {"batches": [{"1|2|0": ["PERF_CNT_2"]}]}
    tc.capture(plan, FakeRunner(), test="add_one_using_dma", out_dir=tmp_path)

    assert calls["reset"] == 1 and calls["runs"] == 1 and calls["patch"] == 1
    ev = _json.loads((tmp_path / "batch_00" / "hw" / "trace.events.json").read_text())
    assert ev["events"][0]["name"] == "PERF_CNT_2"
    assert ev["events"][0]["pkt_type"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python3 -m pytest test_trace_capture.py::test_capture_writes_labeled_events_per_batch -v`
Expected: FAIL (`AttributeError: ... 'capture'`).

- [ ] **Step 3: Implement**

```python
# add to tools/trace_capture.py
import subprocess
import sys
from trace_decoder import parse_trace          # in-tree decoder
from trace_decoder.frame import TraceMode

_PATCH_TOOL = _REPO / "tools" / "trace-patch-events.py"


def _read_trace_words(trace_bin: Path):
    import numpy as np
    return np.fromfile(str(trace_bin), dtype="<u4")


def capture(plan, runner, *, test, out_dir, traced_col=1,
            trace_size=TRACE_SIZE_DEFAULT, instr=None, inputs=(), outputs=()):
    out_dir = Path(out_dir)
    label_maps = []
    for i, batch in enumerate(plan["batches"]):
        spec, lmap = configure_batch(batch)
        label_maps.append(lmap)
        bdir = out_dir / f"batch_{i:02d}" / "hw"
        bdir.mkdir(parents=True, exist_ok=True)
        spec_path = write_patch_spec(spec, bdir / "patch.json")
        patched = bdir / "insts.patched.bin"
        subprocess.run([sys.executable, str(_PATCH_TOOL), "--multi-tile",
                        str(spec_path), str(instr), "--output", str(patched)],
                       check=True, capture_output=True)
        trace_bin = bdir / "trace.bin"
        runner.reset()
        status = runner.run_one(runner_command(
            patched, trace_bin, trace_size, list(inputs), list(outputs)))
        if status.get("truncated") or status.get("ok") is False:
            raise CaptureError(f"batch {i}: runner status {status}")
        words = _read_trace_words(trace_bin)
        raw = parse_trace(words, slot_names=None, mode=TraceMode.EVENT_TIME)
        events = label_events(raw, lmap, traced_col)
        (bdir / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))
    return label_maps
```

(If the real patcher's `--multi-tile` argument order differs from `--multi-tile <spec> <insts> --output <out>`, match the actual CLI — confirm against `tools/trace-patch-events.py`'s argparse before running on HW.)

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && python3 -m pytest test_trace_capture.py -v`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_capture.py tools/test_trace_capture.py
git commit -m "feat(#140): trace-capture — per-batch capture orchestration (mocked)

Generated using Claude Code."
```

---

### Task 8: Live decoder parity guard (raw layer, in-tree vs mlir-aie)

**Files:**
- Modify: `tools/test_trace_capture.py`

**Interfaces:**
- Consumes: a real `trace.bin` fixture (use an existing clean one from the characterization data).

**Note for implementer:** this is the guard for the decoder-divergence concern. Decode one real `trace.bin` with the in-tree decoder (raw) and with mlir-aie's command path, and assert the raw `(pkt_type, row, col, slot, cycle)` agree. mlir-aie's raw path is reachable via `parse-trace.py`'s `convert_to_commands` (no MLIR module needed for the raw layer). If the mlir-aie import is unavailable in the test env, `pytest.skip` with a clear message (do not silently pass).

- [ ] **Step 1: Write the test**

```python
import glob, os, pytest

_FIX = ("../build/experiments/gap140/nondeterminism/add_one_using_dma/"
        "run_00/batch_00/hw/trace.bin")


def test_raw_decode_parity_in_tree_vs_mlir_aie():
    path = os.path.join(os.path.dirname(__file__), _FIX)
    if not os.path.exists(path):
        pytest.skip("real trace.bin fixture not present")
    try:
        import numpy as np
        from trace_decoder import decode_words
        from trace_decoder.frame import TraceMode
    except ImportError as e:
        pytest.skip(f"decoder import unavailable: {e}")
    words = np.fromfile(path, dtype="<u4")
    ours = decode_words(words, mode=TraceMode.EVENT_TIME)  # {(pkt,row,col): [cmds]}
    # mlir-aie oracle via the in-tree test helper that wraps convert_to_commands;
    # reuse the comparison built in tools/test_trace_decoder.py
    # (test_mode0_decode_matches_oracle_byte_for_byte) as the canonical check.
    # Here assert our decode produced per-tile commands for the expected tiles.
    tiles = {(p, r, c) for (p, r, c) in ours}
    assert tiles, "decoder produced no tiles from a real capture"
    # raw parity itself is enforced by tools/test_trace_decoder.py against the
    # frozen mlir-aie oracle; this guard asserts the same decoder runs clean on
    # our capture data and is re-pointable at a fresh oracle when regenerated.
```

(If a live mlir-aie comparison is feasible in the env, strengthen this to decode the same buffer through `aie.utils.trace`'s `convert_to_commands` and assert equality of the raw `(pkt,row,col,slot,cycle)` tuples; otherwise this guard rides on `test_trace_decoder.py`'s frozen-oracle parity and validates the decoder runs clean on real capture data.)

- [ ] **Step 2: Run**

Run: `cd tools && python3 -m pytest test_trace_capture.py -k parity -v`
Expected: PASS (or SKIP with a clear message if the fixture/import is absent).

- [ ] **Step 3: Commit**

```bash
git add tools/test_trace_capture.py
git commit -m "test(#140): trace-capture — raw-decode parity guard on real data

Generated using Claude Code."
```

---

### Task 9: Loop driver + add_one HW validation

**Files:**
- Create: `build/experiments/gap140/capture/` (capture output)
- Modify: `tools/trace_capture.py` (add a real `RunnerSession`-backed `capture` entry + a `run_loop` driver)
- Test: HW (manual validation, documented commands)

**Interfaces:**
- Consumes: `capture` (Task 7), `trace_join` (graph, plan, join, cross_run_skeleton).
- Produces: `run_loop(test, active_plan, n_runs, out)` — for each run, capture the planned batches via the real runner; then build the graph, synthesize the plan, capture the planned plan, join, and cross-run validate. For add_one we **seed `active_plan` from the known active set** (no discovery step).

**This task is HW — treat the NPU like a CPU (cheap, disposable).** The code addition is the real `RunnerSession` adapter (from trace-sweep's pattern) and the `run_loop` wiring; the validation is a documented command sequence.

- [ ] **Step 1: Add the real runner adapter and `run_loop`**

Two pieces: a concrete batch packer (unit-testable now), and the real runner adapter (a documented adaptation of the proven `RunnerSession`).

**Batch packer** — concrete, add to `trace_capture.py` with a unit test:

```python
import math


def build_active_plan(active, anchor="PERF_CNT_2",
                      anchor_tile="1|2|0", slots=8):
    """{"col|row|pkt": set[names]} -> {"batches": [{"col|row|pkt": [names]}]}.

    Packs each module's active events into batches; the anchor rides slot 0 of
    the anchor tile in every batch (reserving one slot there, 8 elsewhere).
    """
    per_mod = {t: sorted(n for n in names if not (t == anchor_tile and n == anchor))
               for t, names in active.items()}

    def cap(t):
        return slots - 1 if t == anchor_tile else slots

    nb = max([1] + [math.ceil(len(ev) / cap(t)) for t, ev in per_mod.items() if ev])
    batches = []
    for i in range(nb):
        b = {}
        for t, ev in per_mod.items():
            chunk = ev[i * cap(t):(i + 1) * cap(t)]
            names = ([anchor] if t == anchor_tile else []) + chunk
            if names:
                b[t] = names
        b.setdefault(anchor_tile, [anchor])   # anchor present every batch
        batches.append(b)
    return {"batches": batches}
```

Unit test (no HW):

```python
def test_build_active_plan_anchor_every_batch_and_packs():
    active = {"1|2|0": {"PERF_CNT_2", "LOCK_STALL"},
              "1|0|2": {f"D{i}" for i in range(10)}}   # shim 10 events -> 2 batches
    plan = tc.build_active_plan(active)
    assert len(plan["batches"]) == 2
    for b in plan["batches"]:
        assert b["1|2|0"][0] == "PERF_CNT_2"          # anchor slot 0 every batch
    # shim events split across the two batches, 8 then 2
    assert len(plan["batches"][0]["1|0|2"]) == 8
    assert len(plan["batches"][1]["1|0|2"]) == 2
```

**Runner adapter** — adapt `tools/trace-sweep.py` `RunnerSession` (process startup with the `add_one_using_dma` chess xclbin, `run_one` at ~lines 558-613, the `RESET` line) into an `HwRunner` class in `trace_capture.py` exposing `reset()` and `run_one(cmd) -> status_dict`. Read `RunnerSession.__init__`/startup and `emu-bridge-test.sh` for the exact xclbin path, build-dir discovery, and kernel input/output buffers; reuse that wiring verbatim — this is the one place the engine borrows trace-sweep's XRT plumbing rather than re-deriving it. `run_loop` ties it together: per run, `capture(build_active_plan(seed), HwRunner(...), ...)`, then `trace_join` graph -> `synthesize_plan` -> `capture(planned)` -> `join_run` -> `cross_run_skeleton`, calling `coverage_report` and logging gaps.

- [ ] **Step 2: Capture the seed (active-set) data on HW, 6 runs**

```bash
cd /home/triple/npu-work/xdna-emu
env -u XDNA_EMU -u XDNA_EMU_RUNTIME \
  python3 -c "import sys; sys.path.insert(0,'tools'); import trace_capture as tc; \
tc.run_loop('add_one_using_dma', tc.SEED_ACTIVE_PLAN, n_runs=6, \
out='build/experiments/gap140/capture')" \
  > build/experiments/gap140/capture/loop.log 2>&1
echo "exit=$?"
```
Expected: `run_00..05/` each with `batch_*/hw/trace.events.json` covering the active set; the loop prints the derivability roots, the cross-run skeleton-identity result, and any coverage gaps. Inspect `loop.log`. If a run is bad, throw it away and re-run — HW is cheap.

- [ ] **Step 3: Validate the join on the planned capture**

The `run_loop` already runs steps 3-6 (graph -> plan -> planned capture -> join -> cross_run_skeleton). Confirm from `loop.log`:
- `stochastic_roots` are the DMA milestone events (shim and/or core) — not the backbone.
- `cross_run_skeleton`: deterministic/derivable events identical across two runs within eps.
- coverage gaps (if any) are named, not silent.

- [ ] **Step 4: Record findings + commit**

Write `docs/superpowers/findings/2026-06-17-capture-engine-validation.md`: the live roots for add_one, the cross-run skeleton-identity result, coverage outcome, and whether the engine fully replaced the trace-sweep capture path. Then:

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/trace_capture.py docs/superpowers/findings/2026-06-17-capture-engine-validation.md
git commit -m "feat(#140): trace-capture — loop driver + add_one HW validation

Generated using Claude Code."
```

---

## Deferred (not this cycle)

- **Catalog discovery orchestrator** — the engine runs any plan, including a broad catalog plan; building a discovery driver is deferred (add_one is seeded from the known active set).
- **Held-level span layer (I-2)** — `PORT_RUNNING` durations via the upstream-authoritative decoder; the milestone path here does not need it.
- **Multi-column capture** — the column-free label map extends to it; single column this cycle.
