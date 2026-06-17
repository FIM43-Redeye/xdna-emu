# Cross-Batch Trace Join Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the analysis + merge library that joins a kernel sweep's per-batch HW traces into one complete every-event trace, anchoring on deterministic events and placing stochastic DMA milestones as real samples with a measured uncertainty band.

**Architecture:** A new standalone library `tools/trace_join.py` (importable + CLI, same pattern as `tools/trace_variance.py`) implements four phases as pure functions over existing sweep output: (0) load the active event set, (1) build a derivability graph whose roots are the minimal independent stochastic set, (2) synthesize a batch plan that panics if the always-on set overflows the slot budget, (3) anchor each batch on core `PERF_CNT_2` and merge into `merged.events.json` + `merged.perfetto.json`. It reuses `trace_variance.py`'s `Stats`/`aggregate`/`classify` primitives and the `col|row|name` key convention.

**Tech Stack:** Python 3.13, stdlib only (`json`, `glob`, `statistics`, `collections`, `argparse`), `pytest`. No new dependencies. No emulator, no hardware — operates on decoded `events.json` already on disk.

## Global Constraints

- **HW-only.** The library reads decoded `events.json`/perfetto artifacts; it never runs the emulator or hardware. (Spec: Scope.)
- **Anchor is core-tile `PERF_CNT_2` first-fire**, never `min(soc)`. The anchor tile/row is `(col=1, row=2)` for FFI-col-1 sweeps; the anchor is parameterized, defaulting to `("1|2|PERF_CNT_2")`. (Spec: finding I-1; empirical anchor noise ≤179.)
- **Key convention:** every event key is the string `"{col}|{row}|{name}"`, matching `trace_variance.report_json`. Tile keys are `"{col}|{row}"`.
- **`soc` is the timestamp field** used for anchoring and derivability (consistent with `trace_variance.load_milestone_events`). Multi-fire events use first occurrence = `min(soc)`.
- **Derivability threshold `eps` defaults to 2.0** (same default as `trace_variance.classify`), parameterized everywhere.
- **No silent caps.** When the always-on set cannot fit the per-tile slot budget, `synthesize_plan` raises `PlannerError` with a diagnostic naming the tile, its always-on events, and the overage. (Spec: Phase 2 hard constraint.)
- **Slot capacity defaults to 8** per tile, parameterized.
- **Out of scope for this plan (next increment):** the `trace-sweep.py --plan` mode that runs a *new* planned sweep on hardware, and full windowed-placement validation against such a capture. This plan builds and validates the library against existing sweep data.
- Commit messages end with `Generated using Claude Code.` and contain no emoji.

**Shared data shapes (used across tasks):**

- Event record (from `events.json`): `{"col":int,"row":int,"pkt_type":int,"slot":int,"name":str,"ts":int,"soc":int,"mode":int}`.
- `anchored_firsts(...) -> Dict[str, int]`: maps `"col|row|name"` → first-occurrence `soc - anchor_soc` for one batch's event list.
- Derivability graph (Task 3 output):
  ```json
  {"anchor":"1|2|PERF_CNT_2","eps":2.0,
   "nodes":["1|0|DMA_S2MM_0_START_TASK", ...],
   "edges":[{"from":"1|2|PERF_CNT_2","to":"1|0|DMA_S2MM_0_START_TASK","offset":6667,"std":1.0}],
   "roots":["1|2|PERF_CNT_2", ...],
   "stochastic_roots":["1|0|DMA_S2MM_0_START_TASK", ...],
   "bands":{"1|0|DMA_S2MM_0_START_TASK":{"n":20,"mean":6667.0,"std":1966.0,"min":3829,"max":11438,"range":7609}}}
  ```
- Batch plan (Task 4 output):
  ```json
  {"slot_capacity":8,"anchor":"1|2|PERF_CNT_2",
   "always_on":{"1|2":["PERF_CNT_2"],"1|0":["DMA_S2MM_0_START_TASK"]},
   "batches":[{"1|0":["DMA_S2MM_0_START_TASK","DMA_MM2S_0_START_TASK"],"1|2":["PERF_CNT_2","LOCK_STALL"]}],
   "n_batches":3}
  ```
- Merged record (Task 5 output): `{"col":int,"row":int,"name":str,"slot":int,"ts_anchored":int,"source_batch":int,"class":str,"predictor":dict|None,"band":dict|None}`.

---

### Task 1: Phase 0 active-event loader + anchored-firsts helper

**Files:**
- Create: `tools/trace_join.py`
- Test: `tools/test_trace_join.py`

**Interfaces:**
- Consumes: nothing (foundation).
- Produces:
  - `load_active_events(run_dir: str) -> Dict[str, set]` — `{"col|row": {name,...}}`, union of fired event names per tile across all `batch_*/hw/trace.events.json` under `run_dir`.
  - `anchored_firsts(events: List[dict], anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, int]` — first-occurrence `soc - anchor_soc` per `"col|row|name"`; `{}` if the anchor never fired.

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_trace_join.py
import json
import trace_join as tj


def _ev(col, row, name, soc, slot=0, pkt_type=0, ts=None, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc if ts is None else ts, "soc": soc, "mode": mode}


def _write_batch(d, events):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.events.json").write_text(
        json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))


def test_anchored_firsts_subtracts_anchor_first_fire():
    events = [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 2, "PERF_CNT_2", 2024),
              _ev(1, 0, "DMA_S2MM_0_START_TASK", 1500),
              _ev(1, 2, "LOCK_STALL", 1200)]
    out = tj.anchored_firsts(events)
    assert out["1|0|DMA_S2MM_0_START_TASK"] == 500   # 1500 - 1000
    assert out["1|2|LOCK_STALL"] == 200              # 1200 - 1000
    assert out["1|2|PERF_CNT_2"] == 0                # first fire is the anchor


def test_anchored_firsts_uses_first_occurrence():
    events = [_ev(1, 2, "PERF_CNT_2", 1000),
              _ev(1, 0, "X", 1800), _ev(1, 0, "X", 1500)]
    out = tj.anchored_firsts(events)
    assert out["1|0|X"] == 500   # min soc (1500) - anchor (1000)


def test_anchored_firsts_empty_when_no_anchor():
    out = tj.anchored_firsts([_ev(1, 0, "X", 1500)])
    assert out == {}


def test_load_active_events_unions_across_batches(tmp_path):
    _write_batch(tmp_path / "batch_00" / "hw",
                 [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "A", 1100)])
    _write_batch(tmp_path / "batch_01" / "hw",
                 [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "B", 1100),
                  _ev(1, 1, "C", 1200)])
    out = tj.load_active_events(str(tmp_path))
    assert out["1|0"] == {"A", "B"}
    assert out["1|1"] == {"C"}
    assert out["1|2"] == {"PERF_CNT_2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trace_join'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/trace_join.py
#!/usr/bin/env python3
"""Derivability-driven cross-batch trace join for #140.

Merges a kernel sweep's per-batch HW traces into one complete every-event
trace. Deterministic/derivable events are placed exactly; stochastic DMA
milestones are placed as a real observed sample carrying a measured band.
HW-only: reads decoded events.json artifacts, never runs the emulator.

See docs/superpowers/specs/2026-06-16-cross-batch-trace-join-design.md.
"""
import collections
import glob as _glob
import json
from pathlib import Path
from typing import Dict, List


def _key(col, row, name) -> str:
    return f"{col}|{row}|{name}"


def _tile(col, row) -> str:
    return f"{col}|{row}"


def load_active_events(run_dir: str) -> Dict[str, set]:
    """{"col|row": {event_name,...}} — fired events per tile, unioned over batches."""
    out: Dict[str, set] = collections.defaultdict(set)
    for p in sorted(_glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json"))):
        for e in json.loads(Path(p).read_text()).get("events", []):
            out[_tile(e["col"], e["row"])].add(e["name"])
    return dict(out)


def anchored_firsts(events: List[dict], anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, int]:
    """First-occurrence (soc - anchor_soc) per "col|row|name" for one batch.

    Returns {} if the anchor event never fired in this batch.
    """
    firsts: Dict[str, int] = {}
    for e in events:
        k = _key(e["col"], e["row"], e["name"])
        if k not in firsts or e["soc"] < firsts[k]:
            firsts[k] = e["soc"]
    if anchor_key not in firsts:
        return {}
    anchor = firsts[anchor_key]
    return {k: v - anchor for k, v in firsts.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "feat(#140): trace-join phase 0 — active events + anchored firsts

Generated using Claude Code."
```

---

### Task 2: Phase 1 within-execution pair derivability

**Files:**
- Modify: `tools/trace_join.py`
- Test: `tools/test_trace_join.py`

**Interfaces:**
- Consumes: `anchored_firsts` (Task 1); `trace_variance.Stats`, `trace_variance.aggregate`.
- Produces:
  - `batch_firsts(run_dir, batch_name) -> Dict[str,int]` — `anchored_firsts` of one batch in one run (or `{}` if missing/no anchor).
  - `pair_derivability(run_dirs, key_x, key_s, anchor_key="1|2|PERF_CNT_2") -> Optional[Stats]` — Stats of `(X − S)` measured within each run's first batch that co-traces both X and S; `None` if never co-traced in any run.

**Note for implementer:** `(X − S)` within one execution cancels the anchor, so it equals `firsts_X − firsts_S` regardless of anchor value — but both must come from the *same batch* (same execution). Pairs that never co-occur in a batch are unmeasurable and return `None` (they cannot be claimed derivable). This is intentional: it is why the planned sweep must co-trace candidate sources.

- [ ] **Step 1: Write the failing tests**

```python
import trace_variance as tv  # add to imports at top of test file


def _make_run(tmp_path, run_name, batches):
    # batches: {batch_name: [event,...]}
    rd = tmp_path / run_name
    for bn, evs in batches.items():
        _write_batch(rd / bn / "hw", evs)
    return str(rd)


def test_pair_derivability_constant_offset_is_low_std(tmp_path):
    # X = S + 50 in every run -> derivable (std ~ 0)
    runs = []
    for i, base in enumerate([1000, 1200, 900]):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", base),
            _ev(1, 2, "S", base + 100),
            _ev(1, 2, "X", base + 150)]}))
    s = tj.pair_derivability(runs, "1|2|X", "1|2|S")
    assert s is not None
    assert s.n == 3
    assert s.mean == 50
    assert s.std == 0.0


def test_pair_derivability_varying_offset_is_high_std(tmp_path):
    offsets = [50, 900, 1700]
    runs = []
    for i, off in enumerate(offsets):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000),
            _ev(1, 0, "S", 1100),
            _ev(1, 2, "X", 1100 + off)]}))
    s = tj.pair_derivability(runs, "1|2|X", "1|0|S")
    assert s is not None
    assert s.std > 100   # clearly stochastic difference


def test_pair_derivability_none_when_never_cotraced(tmp_path):
    # X and S live in different batches -> never co-traced in one execution
    runs = [_make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1100)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 2, "X", 1300)]})]
    assert tj.pair_derivability(runs, "1|2|X", "1|0|S") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: FAIL with `AttributeError: module 'trace_join' has no attribute 'pair_derivability'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/trace_join.py
import functools
from typing import Optional
import trace_variance as tv


@functools.lru_cache(maxsize=None)
def batch_firsts(run_dir: str, batch_name: str,
                 anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, int]:
    # Memoized: the O(nodes^2) graph build calls this repeatedly for the same
    # batch. Callers treat the returned dict as read-only. Files do not change
    # mid-run, so caching is sound.
    p = Path(run_dir) / batch_name / "hw" / "trace.events.json"
    if not p.exists():
        return {}
    return anchored_firsts(json.loads(p.read_text()).get("events", []), anchor_key)


def _batch_names(run_dir: str) -> List[str]:
    return sorted(Path(p).parent.parent.name
                  for p in _glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json")))


def pair_derivability(run_dirs: List[str], key_x: str, key_s: str,
                      anchor_key: str = "1|2|PERF_CNT_2") -> Optional[tv.Stats]:
    """Stats of (X - S) within-execution across runs; None if never co-traced."""
    diffs: List[Dict[str, int]] = []
    for rd in run_dirs:
        for bn in _batch_names(rd):
            f = batch_firsts(rd, bn, anchor_key)
            if key_x in f and key_s in f:
                diffs.append({"d": f[key_x] - f[key_s]})
                break  # first co-tracing batch in this run
    if not diffs:
        return None
    return tv.aggregate(diffs)["d"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "feat(#140): trace-join phase 1 — within-execution pair derivability

Generated using Claude Code."
```

---

### Task 3: Phase 1 derivability graph builder

**Files:**
- Modify: `tools/trace_join.py`
- Test: `tools/test_trace_join.py`

**Interfaces:**
- Consumes: `load_active_events`, `batch_firsts`, `pair_derivability` (Tasks 1–2); `trace_variance.aggregate`/`classify`.
- Produces:
  - `event_bands(run_dirs, keys, anchor_key="1|2|PERF_CNT_2") -> Dict[str,dict]` — per key, the Stats (as dict) of its anchored first-occurrence across runs.
  - `build_derivability_graph(run_dirs, anchor_key="1|2|PERF_CNT_2", eps=2.0) -> dict` — the graph shape in Global Constraints. Edges where `pair_derivability(...).std <= eps`. `roots` = nodes with no incoming edge. `stochastic_roots` = roots whose own band std `> eps` (excluding the anchor).

**Note for implementer:** candidate nodes = the union of active events across all runs (use `load_active_events` on each run dir, union the per-tile sets). Build edges over all ordered pairs `(S, X)`, `S != X`; skip pairs where `pair_derivability` is `None` (never co-traced). A node with multiple incoming edges keeps them all; "has any incoming edge" is what excludes it from `roots`. The anchor is always a root (it is its own zero-reference) but is excluded from `stochastic_roots`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_graph_finds_edge_and_roots(tmp_path):
    # Two runs. X = S + 50 (derivable). S floats vs anchor (stochastic root).
    runs = []
    for i, sbase in enumerate([1100, 1900]):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000),
            _ev(1, 0, "S", sbase),
            _ev(1, 0, "X", sbase + 50)]}))
    g = tj.build_derivability_graph(runs, eps=2.0)
    assert set(g["nodes"]) == {"1|2|PERF_CNT_2", "1|0|S", "1|0|X"}
    # S -> X edge with offset 50
    edge = [e for e in g["edges"] if e["to"] == "1|0|X" and e["from"] == "1|0|S"]
    assert len(edge) == 1 and edge[0]["offset"] == 50
    # X has an incoming edge -> not a root; S and anchor are roots
    assert "1|0|X" not in g["roots"]
    assert "1|0|S" in g["roots"] and "1|2|PERF_CNT_2" in g["roots"]
    # S floats vs anchor -> stochastic root; anchor never a stochastic root
    assert "1|0|S" in g["stochastic_roots"]
    assert "1|2|PERF_CNT_2" not in g["stochastic_roots"]


def test_build_graph_deterministic_event_not_stochastic_root(tmp_path):
    # D fires at a fixed offset from the anchor in every run -> root but deterministic
    runs = []
    for i in range(3):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000 + 10 * i),
            _ev(1, 1, "D", 1300 + 10 * i)]}))   # always anchor+300
    g = tj.build_derivability_graph(runs, eps=2.0)
    assert "1|1|D" in g["roots"]
    assert "1|1|D" not in g["stochastic_roots"]   # std of (D-anchor) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: FAIL with `AttributeError: ... 'build_derivability_graph'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/trace_join.py
def event_bands(run_dirs: List[str], keys, anchor_key: str = "1|2|PERF_CNT_2") -> Dict[str, dict]:
    per_run: List[Dict[str, int]] = []
    for rd in run_dirs:
        merged: Dict[str, int] = {}
        for bn in _batch_names(rd):
            for k, v in batch_firsts(rd, bn, anchor_key).items():
                merged.setdefault(k, v)   # first batch that observed k
        per_run.append({k: merged[k] for k in keys if k in merged})
    stats = tv.aggregate(per_run)
    return {k: s._asdict() for k, s in stats.items()}


def build_derivability_graph(run_dirs: List[str],
                             anchor_key: str = "1|2|PERF_CNT_2",
                             eps: float = 2.0) -> dict:
    nodes = set()
    for rd in run_dirs:
        for tile, names in load_active_events(rd).items():
            col, row = tile.split("|")
            for n in names:
                nodes.add(f"{col}|{row}|{n}")
    nodes = sorted(nodes)
    edges = []
    has_incoming = set()
    for s_key in nodes:
        for x_key in nodes:
            if x_key == s_key:
                continue
            st = pair_derivability(run_dirs, x_key, s_key, anchor_key)
            if st is not None and st.std <= eps:
                edges.append({"from": s_key, "to": x_key,
                              "offset": int(round(st.mean)), "std": st.std})
                has_incoming.add(x_key)
    roots = [n for n in nodes if n not in has_incoming]
    bands = event_bands(run_dirs, nodes, anchor_key)
    stochastic_roots = [r for r in roots
                        if r != anchor_key and bands.get(r, {}).get("std", 0.0) > eps]
    return {"anchor": anchor_key, "eps": eps, "nodes": nodes, "edges": edges,
            "roots": roots, "stochastic_roots": stochastic_roots, "bands": bands}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "feat(#140): trace-join phase 1 — derivability graph + roots

Generated using Claude Code."
```

---

### Task 4: Phase 2 batch-plan synthesis with panic

**Files:**
- Modify: `tools/trace_join.py`
- Test: `tools/test_trace_join.py`

**Interfaces:**
- Consumes: the graph dict (Task 3).
- Produces:
  - `class PlannerError(Exception)`.
  - `synthesize_plan(graph, slot_capacity=8) -> dict` — the plan shape in Global Constraints. Always-on per tile = the anchor (on its tile) plus every `stochastic_root` (on its tile). Swept payload = all other nodes, packed per tile into `slot_capacity - len(always_on_for_that_tile)` slots per batch. `n_batches` = max over tiles of `ceil(payload_count / free_slots)` (at least 1). Raises `PlannerError` if any tile's always-on count exceeds `slot_capacity`.

**Note for implementer:** group nodes by tile (`"col|row"` prefix of the key). A tile's always-on names are the event-name suffixes of its anchor/stochastic-root keys. `batches[i]` maps each tile to `always_on_names + payload_slice_for_batch_i`; tiles whose payload is exhausted in batch `i` still carry their always-on names. Keep names as bare event names (strip the `col|row|` prefix) so the slot assignment is per tile.

- [ ] **Step 1: Write the failing tests**

```python
def _graph(stochastic_roots, nodes, anchor="1|2|PERF_CNT_2"):
    return {"anchor": anchor, "eps": 2.0, "nodes": nodes, "edges": [],
            "roots": [anchor] + stochastic_roots,
            "stochastic_roots": stochastic_roots, "bands": {}}


def test_synthesize_plan_reserves_always_on_every_batch():
    nodes = ["1|2|PERF_CNT_2", "1|0|DMA_S2MM_0_START_TASK",
             "1|0|A", "1|0|B", "1|2|C"]
    g = _graph(["1|0|DMA_S2MM_0_START_TASK"], nodes)
    plan = tj.synthesize_plan(g, slot_capacity=8)
    assert plan["always_on"]["1|2"] == ["PERF_CNT_2"]
    assert plan["always_on"]["1|0"] == ["DMA_S2MM_0_START_TASK"]
    # every batch carries the always-on names on each tile
    for b in plan["batches"]:
        assert "PERF_CNT_2" in b["1|2"]
        assert "DMA_S2MM_0_START_TASK" in b["1|0"]


def test_synthesize_plan_batch_count_from_busiest_tile():
    # tile 1|0: 1 always-on + 14 payload, 7 free slots -> ceil(14/7)=2 batches
    nodes = ["1|2|PERF_CNT_2", "1|0|DMA_S2MM_0_START_TASK"] + \
            [f"1|0|E{i}" for i in range(14)]
    g = _graph(["1|0|DMA_S2MM_0_START_TASK"], nodes)
    plan = tj.synthesize_plan(g, slot_capacity=8)
    assert plan["n_batches"] == 2


def test_synthesize_plan_panics_when_always_on_overflows():
    # 9 stochastic roots on one tile, capacity 8 -> cannot fit anchor+roots
    roots = [f"1|0|R{i}" for i in range(9)]
    nodes = ["1|2|PERF_CNT_2"] + roots
    g = _graph(roots, nodes)
    import pytest
    with pytest.raises(tj.PlannerError) as exc:
        tj.synthesize_plan(g, slot_capacity=8)
    assert "1|0" in str(exc.value)   # diagnostic names the offending tile
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: FAIL with `AttributeError: ... 'synthesize_plan'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/trace_join.py
import math


class PlannerError(Exception):
    pass


def _split_key(k):
    col, row, name = k.split("|", 2)
    return f"{col}|{row}", name


def synthesize_plan(graph: dict, slot_capacity: int = 8) -> dict:
    always_keys = [graph["anchor"]] + list(graph["stochastic_roots"])
    always_on: Dict[str, List[str]] = collections.defaultdict(list)
    for k in always_keys:
        tile, name = _split_key(k)
        if name not in always_on[tile]:
            always_on[tile].append(name)

    payload: Dict[str, List[str]] = collections.defaultdict(list)
    always_set = set(always_keys)
    for k in graph["nodes"]:
        if k in always_set:
            continue
        tile, name = _split_key(k)
        payload[tile].append(name)

    n_batches = 1
    for tile, on in always_on.items():
        if len(on) > slot_capacity:
            raise PlannerError(
                f"always-on set for tile {tile} needs {len(on)} slots "
                f"({on}) but capacity is {slot_capacity}: overage "
                f"{len(on) - slot_capacity}")
    for tile in set(list(always_on) + list(payload)):
        free = slot_capacity - len(always_on.get(tile, []))
        if payload.get(tile) and free <= 0:
            raise PlannerError(
                f"tile {tile} has no free slots for payload after always-on "
                f"({always_on.get(tile)}); capacity {slot_capacity}")
        if payload.get(tile):
            n_batches = max(n_batches, math.ceil(len(payload[tile]) / free))

    batches = []
    for i in range(n_batches):
        batch: Dict[str, List[str]] = {}
        for tile in set(list(always_on) + list(payload)):
            free = slot_capacity - len(always_on.get(tile, []))
            sl = payload.get(tile, [])[i * free:(i + 1) * free] if free > 0 else []
            batch[tile] = list(always_on.get(tile, [])) + sl
        batches.append(batch)

    return {"slot_capacity": slot_capacity, "anchor": graph["anchor"],
            "always_on": dict(always_on), "batches": batches, "n_batches": n_batches}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "feat(#140): trace-join phase 2 — batch-plan synthesis with panic

Generated using Claude Code."
```

---

### Task 5: Phase 3 join/merge with placement gates

**Files:**
- Modify: `tools/trace_join.py`
- Test: `tools/test_trace_join.py`

**Interfaces:**
- Consumes: `batch_firsts`, the graph dict (Tasks 1–3).
- Produces:
  - `class JoinError(Exception)`.
  - `join_run(run_dir, graph, eps=2.0) -> List[dict]` — merged records (shape in Global Constraints), sorted by `ts_anchored`. For each batch under `run_dir`, anchor every event; classify each `col|row|name` as `"stochastic"` (a `stochastic_root`), `"derivable"` (has an incoming edge in the graph), else `"deterministic"`. Attach `band` for stochastic, `predictor` (`{"name":from_key,"offset":...}`) for derivable. Reconcile multi-batch observations: deterministic/derivable keys must agree within `eps` across batches (`JoinError` on spread `> eps`); stochastic keys keep each batch's sample as a separate record tagged `source_batch`.

**Note for implementer:** iterate batches in sorted order; track `source_batch` as the integer parsed from `batch_NN`. For non-stochastic keys, collect all `(batch_idx, ts_anchored)` then assert `max - min <= eps`, emit one record at the median; for stochastic keys emit one record per batch occurrence. `slot` comes from the event record (first occurrence). Build a quick lookup of incoming edges from `graph["edges"]` keyed by `to`.

- [ ] **Step 1: Write the failing tests**

```python
import statistics


def _band_graph():
    return {
        "anchor": "1|2|PERF_CNT_2", "eps": 2.0,
        "nodes": ["1|2|PERF_CNT_2", "1|0|S", "1|0|X", "1|1|D"],
        "edges": [{"from": "1|0|S", "to": "1|0|X", "offset": 50, "std": 0.0}],
        "roots": ["1|2|PERF_CNT_2", "1|0|S", "1|1|D"],
        "stochastic_roots": ["1|0|S"],
        "bands": {"1|0|S": {"n": 2, "mean": 600.0, "std": 400.0,
                            "min": 200, "max": 1000, "range": 800}},
    }


def test_join_run_classifies_and_attaches_band(tmp_path):
    rd = _make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1200, slot=3),
                     _ev(1, 0, "X", 1250), _ev(1, 1, "D", 1400)],
    })
    recs = tj.join_run(rd, _band_graph(), eps=2.0)
    by = {(r["col"], r["row"], r["name"]): r for r in recs}
    assert by[(1, 0, "S")]["class"] == "stochastic"
    assert by[(1, 0, "S")]["band"]["std"] == 400.0
    assert by[(1, 0, "S")]["slot"] == 3
    assert by[(1, 0, "X")]["class"] == "derivable"
    assert by[(1, 0, "X")]["predictor"] == {"name": "1|0|S", "offset": 50}
    assert by[(1, 1, "D")]["class"] == "deterministic"
    # sorted by ts_anchored
    assert [r["ts_anchored"] for r in recs] == sorted(r["ts_anchored"] for r in recs)


def test_join_run_keeps_stochastic_samples_per_batch(tmp_path):
    rd = _make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1200)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1700)],
    })
    recs = [r for r in tj.join_run(rd, _band_graph()) if r["name"] == "S"]
    assert sorted(r["ts_anchored"] for r in recs) == [200, 700]
    assert {r["source_batch"] for r in recs} == {0, 1}


def test_join_run_raises_on_deterministic_spread(tmp_path):
    # D is deterministic in the graph but observed at divergent anchored ts
    rd = _make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 1, "D", 1400)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 1, "D", 1900)],
    })
    import pytest
    with pytest.raises(tj.JoinError):
        tj.join_run(rd, _band_graph(), eps=2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: FAIL with `AttributeError: ... 'join_run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/trace_join.py
import statistics as _st


class JoinError(Exception):
    pass


def join_run(run_dir: str, graph: dict, eps: float = 2.0) -> List[dict]:
    anchor_key = graph["anchor"]
    stochastic = set(graph["stochastic_roots"])
    incoming = {e["to"]: e for e in graph["edges"]}
    bands = graph.get("bands", {})

    # gather per-key observations: key -> list of (batch_idx, ts_anchored, slot, col,row,name)
    obs: Dict[str, List[tuple]] = collections.defaultdict(list)
    for p in sorted(_glob.glob(str(Path(run_dir) / "batch_*" / "hw" / "trace.events.json"))):
        batch_idx = int(Path(p).parent.parent.name.split("_")[1])
        events = json.loads(Path(p).read_text()).get("events", [])
        firsts = anchored_firsts(events, anchor_key)
        if not firsts:
            continue
        slot_of: Dict[str, int] = {}
        for e in events:
            k = _key(e["col"], e["row"], e["name"])
            slot_of.setdefault(k, e.get("slot"))
        for k, ts in firsts.items():
            col, row, name = k.split("|", 2)
            obs[k].append((batch_idx, ts, slot_of.get(k), int(col), int(row), name))

    records: List[dict] = []
    for k, samples in obs.items():
        col, row, name = samples[0][3], samples[0][4], samples[0][5]
        if k in stochastic:
            cls, pred, band = "stochastic", None, bands.get(k)
        elif k in incoming:
            cls = "derivable"
            pred = {"name": incoming[k]["from"], "offset": incoming[k]["offset"]}
            band = None
        else:
            cls, pred, band = "deterministic", None, None

        if cls == "stochastic":
            for (bi, ts, slot, c, r, nm) in samples:
                records.append({"col": c, "row": r, "name": nm, "slot": slot,
                                "ts_anchored": ts, "source_batch": bi,
                                "class": cls, "predictor": pred, "band": band})
        else:
            ts_vals = [s[1] for s in samples]
            if max(ts_vals) - min(ts_vals) > eps:
                raise JoinError(
                    f"{cls} event {k} spread {max(ts_vals) - min(ts_vals)} > eps "
                    f"{eps} across batches {[s[0] for s in samples]}")
            bi, _, slot = samples[0][0], None, samples[0][2]
            records.append({"col": col, "row": row, "name": name, "slot": slot,
                            "ts_anchored": int(_st.median(ts_vals)),
                            "source_batch": bi, "class": cls,
                            "predictor": pred, "band": band})

    records.sort(key=lambda r: (r["ts_anchored"], r["col"], r["row"], r["name"]))
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: PASS (15 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "feat(#140): trace-join phase 3 — merge with placement gates

Generated using Claude Code."
```

---

### Task 6: Perfetto emit + CLI + real-data smoke

**Files:**
- Modify: `tools/trace_join.py`
- Test: `tools/test_trace_join.py`

**Interfaces:**
- Consumes: `join_run`, `build_derivability_graph`, `synthesize_plan` (Tasks 3–5).
- Produces:
  - `to_perfetto(records) -> dict` — `{"traceEvents":[...]}`; one instant event (`"ph":"i"`) per record at `ts_anchored`, `pid = col*100+row`, `name`, and `args` carrying `class`/`source_batch`/`band`/`predictor`.
  - `main(argv=None) -> int` — CLI: `--runs-glob` (required, glob of run dirs), `--eps` (default 2.0), `--slot-capacity` (default 8), `--join-run` (one run dir to merge; default = first matched), `--out` (required dir). Writes `derivability-graph.json`, `batch-plan.json`, `merged.events.json`, `merged.perfetto.json`. Returns 1 with a stderr message if `--runs-glob` matches nothing.

**Note for implementer:** the CLI builds the graph from all matched runs, synthesizes the plan (a `PlannerError` should be caught and reported as a returned-1 error, not a traceback), joins `--join-run`, and writes all four artifacts. The real-data smoke test is structural (artifacts exist, are valid JSON, merged trace is non-empty and sorted) — not exact values, because the existing sweep data is catalog-scattered, not plan-captured.

- [ ] **Step 1: Write the failing tests**

```python
import os

REAL_RUNS = "../build/experiments/gap140/nondeterminism/add_one_using_dma/run_*"


def test_to_perfetto_shape():
    recs = [{"col": 1, "row": 0, "name": "S", "slot": 3, "ts_anchored": 200,
             "source_batch": 0, "class": "stochastic", "predictor": None,
             "band": {"std": 400.0}}]
    doc = tj.to_perfetto(recs)
    ev = doc["traceEvents"][0]
    assert ev["ph"] == "i" and ev["ts"] == 200 and ev["name"] == "S"
    assert ev["pid"] == 100 and ev["args"]["class"] == "stochastic"


def test_cli_empty_glob_returns_1(tmp_path, capsys):
    rc = tj.main(["--runs-glob", str(tmp_path / "nope_*"),
                  "--out", str(tmp_path / "out")])
    assert rc == 1


def test_cli_real_data_smoke(tmp_path):
    import glob
    if not glob.glob(os.path.join(os.path.dirname(__file__), REAL_RUNS)):
        import pytest
        pytest.skip("real sweep data not present")
    out = tmp_path / "out"
    rc = tj.main(["--runs-glob", os.path.join(os.path.dirname(__file__), REAL_RUNS),
                  "--out", str(out)])
    assert rc == 0
    for fn in ["derivability-graph.json", "batch-plan.json",
               "merged.events.json", "merged.perfetto.json"]:
        doc = json.loads((out / fn).read_text())
        assert doc
    merged = json.loads((out / "merged.events.json").read_text())
    assert len(merged) > 0
    assert [r["ts_anchored"] for r in merged] == \
        sorted(r["ts_anchored"] for r in merged)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: FAIL with `AttributeError: ... 'to_perfetto'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/trace_join.py
import argparse
import sys


def to_perfetto(records: List[dict]) -> dict:
    evs = []
    for r in records:
        evs.append({"ph": "i", "ts": r["ts_anchored"], "name": r["name"],
                    "pid": r["col"] * 100 + r["row"], "tid": r.get("slot") or 0,
                    "s": "t",
                    "args": {"class": r["class"], "source_batch": r["source_batch"],
                             "band": r.get("band"), "predictor": r.get("predictor")}})
    return {"traceEvents": evs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-batch trace join (#140)")
    ap.add_argument("--runs-glob", required=True, help="glob of sweep run dirs")
    ap.add_argument("--join-run", default=None, help="run dir to merge (default: first match)")
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--slot-capacity", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    runs = sorted(d for d in _glob.glob(args.runs_glob) if Path(d).is_dir())
    if not runs:
        print(f"no run dirs matched {args.runs_glob}", file=sys.stderr)
        return 1

    graph = build_derivability_graph(runs, eps=args.eps)
    try:
        plan = synthesize_plan(graph, slot_capacity=args.slot_capacity)
    except PlannerError as e:
        print(f"planner panic: {e}", file=sys.stderr)
        return 1

    join_run_dir = args.join_run or runs[0]
    records = join_run(join_run_dir, graph, eps=args.eps)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "derivability-graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (args.out / "batch-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (args.out / "merged.events.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.out / "merged.perfetto.json").write_text(json.dumps(to_perfetto(records)) + "\n")
    print(f"wrote {args.out}: {len(records)} events, {graph['stochastic_roots']} stochastic roots, "
          f"plan n_batches={plan['n_batches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python3 -m pytest test_trace_join.py -v`
Expected: PASS (18 passed; the real-data smoke runs if the sweep dirs exist, else skips).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_join.py tools/test_trace_join.py
git commit -m "feat(#140): trace-join phase 3 — perfetto emit + CLI + real-data smoke

Generated using Claude Code."
```

---

## Deferred to the next increment (not this plan)

- **`trace-sweep.py --plan <batch-plan.json>` mode** — run a *new* planned sweep on hardware that honors the explicit per-batch slot assignment (always-on roots in every batch). Needed to capture data where post-gap windows are placeable, and to fully exercise the windowed model end-to-end.
- **Cross-run skeleton-identity validation on plan-captured data** — `join(run_A)` backbone == `join(run_B)` backbone byte-for-byte; stochastic roots within band. On the current catalog-scattered data the deterministic gate (Task 5) is the available check.
- **Span/level-event lane fix (I-2) integration** — fold corrected `(col,row,slot)` span naming from `trace_variance.load_spans` into the merged output so held-level `PORT_RUNNING` durations join correctly. The merge currently treats every event by first-occurrence `soc`; level spans ride along but their duration law is not yet asserted in the join.
```
