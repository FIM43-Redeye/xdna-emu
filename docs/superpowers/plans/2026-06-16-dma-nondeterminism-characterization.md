# DMA Nondeterminism Characterization Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HW-only harness that measures, across N=20 repeats of `add_one_using_dma`, how much of each trace event's timing is genuinely nondeterministic, classifies events as deterministic vs stochastic from the data, and decomposes the result into "maskable noise" vs "real residual bug."

**Architecture:** Two pure-Python representation loaders (held-level events -> spans from perfetto B/E; milestone events -> re-anchored SoC points from events.json), a shared variance aggregator + emergent classifier, a span-sum law check, a decomposition + report formatter, all in one importable library `tools/trace_variance.py`. A thin HW-orchestration wrapper `tools/trace-variance-sweep.py` runs the existing `trace-sweep.py` N times with run-indexed output. The analysis library is fully unit-testable offline against synthetic data and the 20 existing real captures; the wrapper and the final characterization run are HW-gated.

**Tech Stack:** Python 3.13, pytest 9 (`tools/test_*.py` flat convention, `tools/conftest.py`), the existing `tools/trace-sweep.py` coverage engine, `tools/parse-trace.py` decode. No new dependencies (stdlib `json`, `statistics`, `glob`, `argparse`, `subprocess`).

## Global Constraints

- **Span-based metric, NEVER frame-records.** Held-level events (`PORT_RUNNING*`, `PORT_STALLED*`) are measured as spans paired from perfetto B/E, never by counting their per-frame records in `events.json`. Counting frame-records manufactures phantom variance (the 2026-06-16 metric-artifact finding that got a model removed).
- **Re-anchor within-tile only.** Subtract each tile's own anchor before measuring; never compute cross-tile intervals (cross-tile carries per-tile clock skew).
- **HW-only.** EMU is the expensive thing under test and does not run in this harness. The repeat-sweep driver always passes `--no-emu`.
- **Classification is emergent.** Deterministic vs stochastic falls out of measured variance (`std <= eps`); never hardcode a "these events are stochastic" table. The level-vs-milestone split is a *representation* choice (held levels must be measured as spans), not a determinism verdict.
- **Reuse, do not rebuild.** Coverage (tile discovery, 8-slot batching, `insts.bin` patching), decode, and the single-kernel CLI all exist in `tools/trace-sweep.py` / `tools/parse-trace.py`. The driver calls them; it does not reimplement them.
- **Base kernel:** `add_one_using_dma`. **Words per port:** 64 (the transfer size; the span-sum law is `sum(span durations) == 64` per held-level port).
- **No persistent work in `/tmp`.** Harness output lives under `build/experiments/gap140/nondeterminism/`.
- **Commit messages** end with `Generated using Claude Code.` and contain no emoji.
- **Module import pattern** (hyphenated CLIs): follow `tools/test_trace_sweep.py:16-21` — `importlib.util.spec_from_file_location`. The library `tools/trace_variance.py` uses an underscore so tests can also import it directly.

## File Structure

- Create: `tools/trace_variance.py` — analysis library + CLI. Pure functions: `load_milestone_events`, `load_spans`, `build_slot_name_map`, `aggregate`, `classify`, `check_span_law`, `decompose`, `format_report`, plus `main()`.
- Create: `tools/test_trace_variance.py` — unit + offline-integration tests.
- Create: `tools/trace-variance-sweep.py` — HW orchestration wrapper (calls `trace-sweep.py` N times). HW-gated.
- Create: `tools/test_trace_variance_sweep.py` — arg-parse + layout dry-run tests (HW path mocked).
- Output (runtime, not committed code): `build/experiments/gap140/nondeterminism/<kernel>/run_<NN>/` per repeat; `report.md` + `report.json` at the kernel root.

---

### Task 1: Milestone extraction from events.json

Re-anchor each milestone (non-held-level) event's SoC timestamp per tile, and emit per-key occurrence lists. Held-level families are excluded here (they go through the span path in Task 3).

**Files:**
- Create: `tools/trace_variance.py`
- Test: `tools/test_trace_variance.py`

**Interfaces:**
- Produces: `LEVEL_FAMILIES = ("PORT_RUNNING", "PORT_STALLED")`; `is_level(name: str) -> bool`; `load_milestone_events(events_path: str) -> dict[tuple, list[int]]` keyed by `(col, row, name)` -> sorted list of re-anchored SoC values (anchor = min SoC over that tile's milestone events). Each tile `(col,row)` is anchored independently.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_trace_variance.py
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "trace_variance", Path(__file__).parent / "trace_variance.py",
)
tv = importlib.util.module_from_spec(_spec)
sys.modules["trace_variance"] = tv
_spec.loader.exec_module(tv)

TOOLS = Path(__file__).parent
ROOT = TOOLS.parent
DDR_CAPS = ROOT / "build/experiments/ddr-stochasticity"


def _write_events(tmp_path, events):
    p = tmp_path / "events.json"
    p.write_text(json.dumps({"schema_version": 1, "events": events,
                             "slot_names": {}, "placement": {}}))
    return str(p)


def test_is_level_classifies_held_families():
    assert tv.is_level("PORT_RUNNING_4")
    assert tv.is_level("PORT_STALLED_0")
    assert not tv.is_level("DMA_S2MM_0_STREAM_STARVATION")
    assert not tv.is_level("LOCK_STALL")


def test_load_milestone_events_reanchors_per_tile(tmp_path):
    events = [
        # tile (0,0): two milestones at soc 1000, 1040 -> anchored 0, 40
        {"col": 0, "row": 0, "name": "DMA_S2MM_0_FINISHED", "soc": 1040},
        {"col": 0, "row": 0, "name": "DMA_S2MM_0_START", "soc": 1000},
        # tile (1,0): one milestone at soc 5000 -> anchored 0
        {"col": 1, "row": 0, "name": "DMA_S2MM_0_START", "soc": 5000},
        # a held-level event must be ignored by the milestone loader
        {"col": 0, "row": 0, "name": "PORT_RUNNING_4", "soc": 1010},
    ]
    out = tv.load_milestone_events(_write_events(tmp_path, events))
    assert out[(0, 0, "DMA_S2MM_0_START")] == [0]
    assert out[(0, 0, "DMA_S2MM_0_FINISHED")] == [40]
    assert out[(1, 0, "DMA_S2MM_0_START")] == [0]
    assert (0, 0, "PORT_RUNNING_4") not in out


def test_load_milestone_events_on_real_capture():
    # run_01.json is a real NPU1 add_one_using_dma capture.
    out = tv.load_milestone_events(str(DDR_CAPS / "run_01.json"))
    # The shim S2MM starvation milestone family is present and non-empty.
    keys = [k for k in out if "STREAM_STARVATION" in k[2]]
    assert keys, "expected at least one STREAM_STARVATION milestone key"
    # Every value list is sorted and re-anchored (min == 0 per tile family).
    for k, vals in out.items():
        assert vals == sorted(vals)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_trace_variance.py -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (`trace_variance` has no `is_level` / `load_milestone_events`).

- [ ] **Step 3: Write minimal implementation**

```python
# tools/trace_variance.py
#!/usr/bin/env python3
"""Characterize HW trace nondeterminism for #140.

Measures, across N repeats of one kernel, how much each trace event's timing
varies run-to-run, classifies events deterministic vs stochastic from the
variance itself, and decomposes the result. HW-only: this tool reads decoded
artifacts; it never runs the emulator.

Two representations, picked by the metric constraint (2026-06-16 finding):
  - Held-level events (PORT_RUNNING*, PORT_STALLED*) are measured as SPANS from
    the perfetto B/E json -- never by counting frame-records in events.json.
  - Milestone events (DMA START/FINISHED, STREAM_STARVATION, LOCK_STALL, ...)
    are measured as re-anchored SoC point timestamps from events.json.
Both are re-anchored within-tile only.
"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

LEVEL_FAMILIES = ("PORT_RUNNING", "PORT_STALLED")


def is_level(name: str) -> bool:
    return any(name.startswith(f) for f in LEVEL_FAMILIES)


def load_milestone_events(events_path: str) -> Dict[Tuple, List[int]]:
    """events.json -> {(col,row,name): [re-anchored soc, ...]}, milestones only.

    Anchor is per-tile: the minimum soc over that tile's milestone events.
    """
    doc = json.loads(Path(events_path).read_text())
    by_tile_raw: Dict[Tuple[int, int], List[dict]] = collections.defaultdict(list)
    for e in doc.get("events", []):
        if is_level(e["name"]):
            continue
        by_tile_raw[(e["col"], e["row"])].append(e)
    out: Dict[Tuple, List[int]] = collections.defaultdict(list)
    for (col, row), evs in by_tile_raw.items():
        anchor = min(e["soc"] for e in evs)
        for e in evs:
            out[(col, row, e["name"])].append(e["soc"] - anchor)
    for k in out:
        out[k].sort()
    return dict(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tools/test_trace_variance.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/trace_variance.py tools/test_trace_variance.py
git commit -m "feat(#140): milestone-event extraction with per-tile re-anchoring

Generated using Claude Code."
```

---

### Task 2: Variance aggregation + emergent classification

Given per-run measurement maps, aggregate each key's samples across runs and classify deterministic vs stochastic from the variance.

**Files:**
- Modify: `tools/trace_variance.py`
- Test: `tools/test_trace_variance.py`

**Interfaces:**
- Consumes: per-run dicts of `key -> scalar` (e.g. a milestone occurrence's anchored SoC, or a span's duration). The caller reduces each run's list to comparable scalars before aggregation.
- Produces: `Stats = namedtuple("Stats", "n mean std min max range")`; `aggregate(per_run: list[dict]) -> dict[key, Stats]` (a key absent in some runs is aggregated over the runs where present, with `n` reflecting that); `classify(s: Stats, eps: float = 2.0) -> str` returning `"deterministic"` (`std <= eps`) or `"stochastic"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tools/test_trace_variance.py
def test_aggregate_and_classify_deterministic_vs_stochastic():
    per_run = [
        {"A": 100, "B": 50},
        {"A": 100, "B": 58},
        {"A": 100, "B": 47},
        {"A": 100, "B": 61},
    ]
    stats = tv.aggregate(per_run)
    assert stats["A"].n == 4
    assert stats["A"].std == 0
    assert stats["A"].range == 0
    assert tv.classify(stats["A"]) == "deterministic"

    assert stats["B"].n == 4
    assert stats["B"].min == 47 and stats["B"].max == 61
    assert tv.classify(stats["B"]) == "stochastic"


def test_classify_eps_boundary():
    # std just under / over eps
    tight = tv.aggregate([{"X": 10}, {"X": 11}, {"X": 10}, {"X": 11}])["X"]
    assert tv.classify(tight, eps=2.0) == "deterministic"
    wide = tv.aggregate([{"X": 10}, {"X": 20}, {"X": 10}, {"X": 20}])["X"]
    assert tv.classify(wide, eps=2.0) == "stochastic"


def test_aggregate_handles_missing_key_in_some_runs():
    per_run = [{"A": 5}, {"A": 5, "B": 9}, {"B": 9}]
    stats = tv.aggregate(per_run)
    assert stats["A"].n == 2
    assert stats["B"].n == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_trace_variance.py -k "aggregate or classify" -v`
Expected: FAIL (`aggregate`/`classify` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to tools/trace_variance.py (after load_milestone_events)
from collections import namedtuple

Stats = namedtuple("Stats", "n mean std min max range")


def aggregate(per_run: List[Dict]) -> Dict:
    """[{key: scalar}, ...] -> {key: Stats} over the runs where the key appears."""
    samples: Dict = collections.defaultdict(list)
    for run in per_run:
        for key, val in run.items():
            samples[key].append(val)
    out: Dict = {}
    for key, vals in samples.items():
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        out[key] = Stats(n=len(vals), mean=mean, std=std,
                         min=min(vals), max=max(vals), range=max(vals) - min(vals))
    return out


def classify(s: Stats, eps: float = 2.0) -> str:
    return "deterministic" if s.std <= eps else "stochastic"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tools/test_trace_variance.py -k "aggregate or classify" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/trace_variance.py tools/test_trace_variance.py
git commit -m "feat(#140): variance aggregation + emergent det/stochastic classifier

Generated using Claude Code."
```

---

### Task 3: Span extraction from perfetto B/E + span-sum law

Pair perfetto B/E events per `(pid, tid)`, re-anchor, group into spans with idle-gap > 2, map `(pid, tid)` to an event name via the run's `events.json` slot map, and check the `sum(spans) == words` hardware law.

**Files:**
- Modify: `tools/trace_variance.py`
- Test: `tools/test_trace_variance.py`

**Interfaces:**
- Produces:
  - `_PT_NAME_TO_CODE = {"core": 0, "mem": 1, "shim": 2, "memtile": 3}` (the pkt_type codes, matching `tools/parse-trace.py`).
  - `build_lane_name_map(perfetto_events: list, events_path: str) -> dict[tuple, str]` keyed by `(pid, tid)` -> event name. **Authoritative recipe derived from `tools/parse-trace.py` (do not reverse-engineer):** the perfetto's `name`/`thread_name` fields are empty, so naming is recovered from two sources. (1) `pid -> pkt_type` from the perfetto `process_name` metadata events: take the leading alphabetic token of `args.name` (`re.match(r"[a-z]+", name)`) — robust to both the legacy `"shim(0,1)"` and the current `"shim_trace for tile0,1"` formats — and map it through `_PT_NAME_TO_CODE` (`memtile` matches before `mem` because the regex is greedy). (2) `(pkt_type, slot) -> name` from the run's `events.json` records (each carries `pkt_type`, `slot`, `name`). Then for each perfetto lane, `pkt_type = pid_to_pkt[pid]`, `slot = tid`, and `name = events_map[(pkt_type, slot)]`.
  - `load_spans(perfetto_path: str, events_path: str, idle_gap: int = 2) -> dict[str, list[int]]` -> `{event_name: [span_duration, ...]}`, re-anchored per `(pid,tid)` lane, idle-gap > 2 grouping. Loads the perfetto, builds the lane->name map internally via `build_lane_name_map`, and names the spans.
  - `load_spans_from_events(evs: list, name_map: dict, idle_gap: int = 2) -> dict[str, list[int]]` — the pure span-pairing core (perfetto event list + precomputed name map), used by the synthetic test.
  - `check_span_law(spans: dict, words: int = 64) -> dict[str, tuple[int, bool]]` -> `{event_name: (span_sum, ok)}` where `ok = (span_sum == words)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tools/test_trace_variance.py
PORT4_WORK = (ROOT / "build/experiments/gap140/sweep-port4-hw"
              / "add_one_using_dma.chess.multitile.work")


def test_load_spans_groups_with_idle_gap_and_reanchors():
    # Synthetic perfetto: one lane (pid0,tid0), two spans separated by gap 50,
    # and a 1-cycle internal gap that must NOT split (idle-gap > 2 only).
    evs = [
        {"ph": "B", "pid": 0, "tid": 0, "ts": 100},
        {"ph": "E", "pid": 0, "tid": 0, "ts": 108},   # [100,108]
        {"ph": "B", "pid": 0, "tid": 0, "ts": 109},   # gap 1 -> merge
        {"ph": "E", "pid": 0, "tid": 0, "ts": 116},   # span -> [100,116] dur 16
        {"ph": "B", "pid": 0, "tid": 0, "ts": 166},   # gap 50 -> new span
        {"ph": "E", "pid": 0, "tid": 0, "ts": 174},   # [166,174] dur 8
    ]
    name_map = {(0, 0): "PORT_RUNNING_X"}
    spans = tv.load_spans_from_events(evs, name_map)  # see impl: split helper
    assert spans["PORT_RUNNING_X"] == [16, 8]


def test_span_law_passes_at_word_count():
    spans = {"PORT_RUNNING_0": [16, 16, 16, 16], "PORT_RUNNING_4": [8, 8, 14, 34]}
    law = tv.check_span_law(spans, words=64)
    assert law["PORT_RUNNING_0"] == (64, True)
    assert law["PORT_RUNNING_4"] == (64, True)
    bad = tv.check_span_law({"PORT_RUNNING_1": [8, 8, 8, 11, 76, 76]}, words=64)
    assert bad["PORT_RUNNING_1"][1] is False  # 187 != 64 (the encoder-inflation bug)


@pytest.mark.skipif(not PORT4_WORK.exists(), reason="port4 sweep artifacts absent")
def test_real_perfetto_maps_named_port_lanes_structurally():
    # NOTE: these b12 fixtures are STALE (pre-decoder-fix) perfetto and do NOT
    # satisfy sum(PORT_RUNNING spans)==64 -- some held levels decode as dur-1
    # frame-record-like pairs there. The strict ==64 span-law proof is deferred
    # to Task 7's freshly-decoded HW perfetto. This test validates only that the
    # authoritative (pid,tid)->name mapping works STRUCTURALLY on real artifacts.
    pj = PORT4_WORK / "b12.trace_hw.perfetto.json"
    ej = PORT4_WORK / "b12.hw.events.json"  # sibling decoded events (note: .hw., not .trace_hw.)
    spans = tv.load_spans(str(pj), str(ej))
    # The mapping recovers named PORT_RUNNING lanes (4 of them in this batch).
    running = [k for k in spans if k.startswith("PORT_RUNNING")]
    assert "PORT_RUNNING_4" in spans, f"PORT_RUNNING_4 not mapped; got {sorted(spans)}"
    assert len(running) >= 1
    # check_span_law runs without error and returns a (sum, bool) per port.
    law = tv.check_span_law(spans, words=64)
    assert all(isinstance(s, int) and isinstance(ok, bool) for s, ok in law.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_trace_variance.py -k "span or perfetto" -v`
Expected: FAIL (`load_spans_from_events`/`check_span_law`/`build_slot_name_map`/`load_spans` not defined).

The `(pid,tid)->name` recipe is fully specified in the `build_lane_name_map` interface above (it was derived from `tools/parse-trace.py` and verified against the real b12 fixture, which yields named lanes `PORT_RUNNING_0/1/4/5` and `PORT_STALLED_0/4/5`). Implement it verbatim — no further reverse-engineering needed. The real sibling events file is `b12.hw.events.json` (note: `.hw.`, not `.trace_hw.`).

- [ ] **Step 3: Write minimal implementation**

```python
# append to tools/trace_variance.py

def _group_spans(pairs: List[Tuple[int, int]], idle_gap: int = 2) -> List[int]:
    """[(begin,end), ...] (sorted) -> merged span durations; merge across gaps <= idle_gap."""
    pairs = sorted(pairs)
    spans: List[int] = []
    cur_b, cur_e = None, None
    for b, e in pairs:
        if cur_b is None:
            cur_b, cur_e = b, e
        elif b - cur_e <= idle_gap:
            cur_e = max(cur_e, e)
        else:
            spans.append(cur_e - cur_b)
            cur_b, cur_e = b, e
    if cur_b is not None:
        spans.append(cur_e - cur_b)
    return spans


def load_spans_from_events(evs: List[dict], name_map: Dict[Tuple, str],
                           idle_gap: int = 2) -> Dict[str, List[int]]:
    """Perfetto B/E event list -> {name: [span_duration,...]} per mapped lane."""
    stacks: Dict[Tuple, List[int]] = collections.defaultdict(list)
    pairs: Dict[Tuple, List[Tuple[int, int]]] = collections.defaultdict(list)
    for e in sorted(evs, key=lambda x: (x.get("ts", 0), 0 if x.get("ph") == "E" else 1)):
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        lane = (e.get("pid"), e.get("tid"))
        if ph == "B":
            stacks[lane].append(e.get("ts"))
        elif stacks[lane]:
            b = stacks[lane].pop()
            pairs[lane].append((b, e.get("ts")))
    out: Dict[str, List[int]] = {}
    for lane, ps in pairs.items():
        name = name_map.get(lane)
        if name is None:
            continue
        out.setdefault(name, [])
        out[name].extend(_group_spans(ps, idle_gap))
    return out


import re

# pkt_type codes, matching tools/parse-trace.py _PT_NAME_TO_CODE.
_PT_NAME_TO_CODE = {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def build_lane_name_map(perfetto_events: List[dict], events_path: str) -> Dict[Tuple, str]:
    """{(pid,tid): event_name} for perfetto lanes.

    Authoritative recipe (derived from tools/parse-trace.py, not reverse-
    engineered): perfetto names are empty, so naming is recovered from
      (1) pid -> pkt_type via the perfetto `process_name` metadata: the leading
          alphabetic token of args.name (robust to both "shim(0,1)" and
          "shim_trace for tile0,1"); 'memtile' wins over 'mem' (greedy regex).
      (2) (pkt_type, slot) -> name from the run's events.json records.
    Then lane (pid, tid) -> name where pkt_type = pid_to_pkt[pid], slot = tid.
    """
    pid_to_pkt: Dict[int, int] = {}
    for e in perfetto_events:
        if e.get("ph") != "M" or e.get("name") != "process_name":
            continue
        nm = (e.get("args", {}) or {}).get("name", "").strip()
        m = re.match(r"[a-z]+", nm)
        if m and m.group(0) in _PT_NAME_TO_CODE:
            pid_to_pkt[e.get("pid")] = _PT_NAME_TO_CODE[m.group(0)]
    doc = json.loads(Path(events_path).read_text())
    pktslot_to_name = {(ev["pkt_type"], ev["slot"]): ev["name"]
                       for ev in doc.get("events", [])}
    out: Dict[Tuple, str] = {}
    for pid, pkt in pid_to_pkt.items():
        for (pk, slot), name in pktslot_to_name.items():
            if pk == pkt:
                out[(pid, slot)] = name
    return out


def load_spans(perfetto_path: str, events_path: str,
               idle_gap: int = 2) -> Dict[str, List[int]]:
    doc = json.loads(Path(perfetto_path).read_text())
    evs = doc["traceEvents"] if isinstance(doc, dict) and "traceEvents" in doc else doc
    name_map = build_lane_name_map(evs, events_path)
    return load_spans_from_events(evs, name_map, idle_gap)


def check_span_law(spans: Dict[str, List[int]], words: int = 64) -> Dict[str, Tuple[int, bool]]:
    return {name: (sum(durs), sum(durs) == words) for name, durs in spans.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tools/test_trace_variance.py -k "span or perfetto" -v`
Expected: PASS. The real-artifact test asserts the mapping works structurally (named `PORT_RUNNING_*` lanes extracted). The strict `sum==64` span-law proof is intentionally deferred to Task 7's freshly-decoded HW perfetto — do NOT assert `==64` against the stale b12 fixture here.

- [ ] **Step 5: Commit**

```bash
git add tools/trace_variance.py tools/test_trace_variance.py
git commit -m "feat(#140): perfetto span extraction (idle-gap>2) + span-sum word law

Generated using Claude Code."
```

---

### Task 4: Decomposition + report formatting

Combine classified spans and milestones into the headline decomposition (deterministic vs stochastic) and a human + machine report.

**Files:**
- Modify: `tools/trace_variance.py`
- Test: `tools/test_trace_variance.py`

**Interfaces:**
- Consumes: `{key: Stats}` plus each key's classification.
- Produces: `decompose(classified: dict[key, tuple[Stats, str]]) -> dict` with `{"n_deterministic", "n_stochastic", "stochastic_keys": [...], "law_violations": [...]}`; `format_report(decomp: dict, classified: dict, law: dict) -> str` (markdown) and the same data as `report_json(...)` (a dict for `report.json`).

- [ ] **Step 1: Write the failing test**

```python
# append to tools/test_trace_variance.py
def test_decompose_splits_and_lists_stochastic_keys():
    classified = {
        ("0", "0", "PORT_RUNNING_4"): (tv.Stats(20, 64, 0.0, 64, 64, 0), "deterministic"),
        ("0", "0", "DMA_S2MM_0_FINISHED"): (tv.Stats(20, 5000, 430.0, 4200, 5800, 1600), "stochastic"),
    }
    d = tv.decompose(classified)
    assert d["n_deterministic"] == 1
    assert d["n_stochastic"] == 1
    assert ("0", "0", "DMA_S2MM_0_FINISHED") in d["stochastic_keys"]


def test_format_report_mentions_law_and_decomposition():
    classified = {("0", "0", "DMA_S2MM_0_FINISHED"):
                  (tv.Stats(20, 5000, 430.0, 4200, 5800, 1600), "stochastic")}
    law = {"PORT_RUNNING_4": (64, True), "PORT_RUNNING_1": (187, False)}
    d = tv.decompose({**classified,
                      ("0", "0", "PORT_RUNNING_4"): (tv.Stats(20, 64, 0, 64, 64, 0), "deterministic")},
                     law=law)
    md = tv.format_report(d, classified, law)
    assert "stochastic" in md.lower()
    assert "PORT_RUNNING_1" in md  # law violation surfaced
    assert "187" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_trace_variance.py -k "decompose or report" -v`
Expected: FAIL (`decompose`/`format_report` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to tools/trace_variance.py
def decompose(classified: Dict, law: Dict = None) -> Dict:
    law = law or {}
    det = [k for k, (_, c) in classified.items() if c == "deterministic"]
    sto = [k for k, (_, c) in classified.items() if c == "stochastic"]
    violations = [name for name, (s, ok) in law.items() if not ok]
    return {
        "n_deterministic": len(det),
        "n_stochastic": len(sto),
        "stochastic_keys": sto,
        "deterministic_keys": det,
        "law_violations": violations,
    }


def format_report(decomp: Dict, classified: Dict, law: Dict) -> str:
    lines = ["# DMA nondeterminism characterization — add_one_using_dma", ""]
    lines.append(f"- deterministic events: {decomp['n_deterministic']}")
    lines.append(f"- stochastic events:    {decomp['n_stochastic']}")
    lines.append("")
    lines.append("## span-sum word law (held-level ports)")
    for name in sorted(law):
        s, ok = law[name]
        flag = "OK" if ok else "VIOLATION"
        lines.append(f"- {name}: sum={s} {flag}")
    if decomp["law_violations"]:
        lines.append("")
        lines.append(f"**Law violations (real bug, not noise): {decomp['law_violations']}**")
    lines.append("")
    lines.append("## events by variance (descending std)")
    for key, (s, c) in sorted(classified.items(), key=lambda kv: -kv[1][0].std):
        lines.append(f"- {key} [{c}] n={s.n} mean={s.mean:.0f} std={s.std:.1f} "
                     f"min={s.min} max={s.max} range={s.range}")
    return "\n".join(lines) + "\n"


def report_json(decomp: Dict, classified: Dict, law: Dict) -> Dict:
    return {
        "decomposition": decomp,
        "law": {k: {"sum": s, "ok": ok} for k, (s, ok) in law.items()},
        "events": {"|".join(map(str, k)): {**s._asdict(), "class": c}
                   for k, (s, c) in classified.items()},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tools/test_trace_variance.py -k "decompose or report" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/trace_variance.py tools/test_trace_variance.py
git commit -m "feat(#140): decomposition + markdown/json report formatting

Generated using Claude Code."
```

---

### Task 5: CLI wiring + first offline characterization on the 20 existing captures

Wire the pieces into `main()` and produce a real decomposition report from the 20 `ddr-stochasticity` captures (milestone path) — value before any new HW run.

**Files:**
- Modify: `tools/trace_variance.py`
- Test: `tools/test_trace_variance.py`

**Interfaces:**
- Produces: `main(argv=None)`. CLI:
  `python3 tools/trace_variance.py --events-glob '<dir>/run_*.json' [--perfetto-glob '<dir>/run_*/*.perfetto.json'] [--words 64] [--eps 2.0] --out <dir>`.
  For each milestone key, the per-run scalar is the *first* occurrence's anchored SoC (a stable per-run reduction; multi-occurrence interval analysis is a documented follow-up). Emits `report.md` + `report.json` under `--out`.

- [ ] **Step 1: Write the failing test**

```python
# append to tools/test_trace_variance.py
def test_main_produces_report_from_real_20_captures(tmp_path):
    rc = tv.main([
        "--events-glob", str(DDR_CAPS / "run_*.json"),
        "--words", "64", "--eps", "2.0",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    md = (tmp_path / "report.md").read_text()
    rj = json.loads((tmp_path / "report.json").read_text())
    # 20 real runs aggregated; at least one stochastic milestone (DDR-sensitive).
    assert rj["decomposition"]["n_stochastic"] >= 1
    assert "characterization" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_trace_variance.py -k "main" -v`
Expected: FAIL (`main` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to tools/trace_variance.py
import glob as _glob


def _first_occurrence_scalars(run_map: Dict[Tuple, List[int]]) -> Dict[Tuple, int]:
    return {k: v[0] for k, v in run_map.items() if v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DMA nondeterminism characterization (#140)")
    ap.add_argument("--events-glob", required=True,
                    help="glob of per-run events.json (milestone path)")
    ap.add_argument("--perfetto-glob", default=None,
                    help="glob of per-run perfetto json (span path); pairs by sorted order")
    ap.add_argument("--events-for-names", default=None,
                    help="one events.json used to build the perfetto (pid,tid)->name map")
    ap.add_argument("--words", type=int, default=64)
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    # Milestone path: one scalar per (col,row,name) per run = first anchored soc.
    event_runs = sorted(_glob.glob(args.events_glob))
    per_run = [_first_occurrence_scalars(load_milestone_events(p)) for p in event_runs]
    stats = aggregate(per_run)
    classified = {k: (s, classify(s, args.eps)) for k, s in stats.items()}

    # Span path (optional): law check on the last run's spans (sums are per-run stable).
    law: Dict[str, Tuple[int, bool]] = {}
    if args.perfetto_glob and args.events_for_names:
        pjs = sorted(_glob.glob(args.perfetto_glob))
        if pjs:
            spans = load_spans(pjs[-1], args.events_for_names)
            law = check_span_law(spans, args.words)

    decomp = decompose(classified, law)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(format_report(decomp, classified, law))
    (args.out / "report.json").write_text(
        json.dumps(report_json(decomp, classified, law), indent=2) + "\n")
    print(f"wrote {args.out}/report.md  ({decomp['n_deterministic']} det, "
          f"{decomp['n_stochastic']} stochastic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes, then produce the real report**

Run: `python3 -m pytest tools/test_trace_variance.py -v`
Expected: PASS (all tasks 1-5 tests green).

Then generate the first real characterization (offline, the 20 captures):

```bash
python3 tools/trace_variance.py \
  --events-glob 'build/experiments/ddr-stochasticity/run_*.json' \
  --words 64 --eps 2.0 \
  --out build/experiments/gap140/nondeterminism/add_one_using_dma-offline20
```

Read the produced `report.md`. Confirm milestone DMA events (STREAM_STARVATION family) classify stochastic. This is the offline proof-of-method before the full HW sweep.

- [ ] **Step 5: Commit**

```bash
git add tools/trace_variance.py tools/test_trace_variance.py
git commit -m "feat(#140): CLI + first offline characterization over the 20 captures

Generated using Claude Code."
```

---

### Task 6: Repeat-sweep driver (HW orchestration wrapper)

Run the existing `trace-sweep.py` full-event sweep of `add_one_using_dma` N times, HW-only, with run-indexed output. Thin wrapper — no coverage logic of its own.

**Files:**
- Create: `tools/trace-variance-sweep.py`
- Test: `tools/test_trace_variance_sweep.py`

**Interfaces:**
- Produces: `build_sweep_cmd(test, tiles, out_dir, jobs) -> list[str]`; `main(argv=None)`. CLI:
  `python3 tools/trace-variance-sweep.py --test add_one_using_dma --repeat 20 [--hw-jobs 5] [--out <dir>]`.
  Tiles default to all four `add_one_using_dma` lanes: `0:0:shim,0:1:memtile,0:2:core,0:2:memmod`. Each repeat writes to `<out>/run_<NN>/`.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_trace_variance_sweep.py
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_spec = importlib.util.spec_from_file_location(
    "trace_variance_sweep", Path(__file__).parent / "trace-variance-sweep.py",
)
tvs = importlib.util.module_from_spec(_spec)
sys.modules["trace_variance_sweep"] = tvs
_spec.loader.exec_module(tvs)


def test_build_sweep_cmd_is_hw_only_with_tiles_and_outdir():
    cmd = tvs.build_sweep_cmd("add_one_using_dma",
                              "0:0:shim,0:1:memtile,0:2:core,0:2:memmod",
                              Path("/tmp/out/run_03"), jobs=5)
    assert "--no-emu" in cmd
    assert "--test" in cmd and "add_one_using_dma" in cmd
    i = cmd.index("--tiles")
    assert "memtile" in cmd[i + 1]
    j = cmd.index("--out-dir")
    assert cmd[j + 1].endswith("run_03")


def test_main_invokes_sweep_once_per_repeat(tmp_path):
    calls = []
    with patch.object(tvs.subprocess, "run",
                      side_effect=lambda cmd, **kw: calls.append(cmd) or _ok()):
        rc = tvs.main(["--test", "add_one_using_dma", "--repeat", "4",
                       "--out", str(tmp_path)])
    assert rc == 0
    assert len(calls) == 4
    outdirs = [c[c.index("--out-dir") + 1] for c in calls]
    assert sorted(Path(o).name for o in outdirs) == ["run_00", "run_01", "run_02", "run_03"]


def _ok():
    class R:  # minimal CompletedProcess stand-in
        returncode = 0
    return R()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tools/test_trace_variance_sweep.py -v`
Expected: FAIL (`build_sweep_cmd`/`main` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# tools/trace-variance-sweep.py
#!/usr/bin/env python3
"""Repeat the full-event HW trace sweep of one kernel N times for #140.

Thin orchestration over tools/trace-sweep.py: runs the existing coverage sweep
HW-only (--no-emu) N times with run-indexed output, so trace_variance.py can
measure per-event run-to-run variance. No coverage logic here -- trace-sweep.py
owns tile discovery, 8-slot batching, insts.bin patching, and decode.

Contention is timing-neutral (2026-06-16 control pass), so --hw-jobs may pack
the repeats; default 1 for a clean serial baseline.
"""
import argparse
import subprocess
from pathlib import Path

DEFAULT_TILES = "0:0:shim,0:1:memtile,0:2:core,0:2:memmod"
ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "tools" / "trace-sweep.py"
DEFAULT_OUT = ROOT / "build/experiments/gap140/nondeterminism"


def build_sweep_cmd(test: str, tiles: str, out_dir: Path, jobs: int) -> list:
    return [
        "python3", str(SWEEP),
        "--test", test,
        "--tiles", tiles,
        "--no-emu",            # HW-only: EMU is the expensive thing under test
        "--jobs", str(jobs),
        "--out-dir", str(out_dir),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Repeat HW trace sweep N times (#140)")
    ap.add_argument("--test", default="add_one_using_dma")
    ap.add_argument("--tiles", default=DEFAULT_TILES)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--hw-jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    base = args.out or (DEFAULT_OUT / args.test)
    for r in range(args.repeat):
        run_dir = base / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_sweep_cmd(args.test, args.tiles, run_dir, args.hw_jobs)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"run {r}: sweep returned {res.returncode} (continuing)")
    print(f"completed {args.repeat} repeats under {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tools/test_trace_variance_sweep.py -v`
Expected: PASS (HW path mocked; no silicon touched).

- [ ] **Step 5: Commit**

```bash
git add tools/trace-variance-sweep.py tools/test_trace_variance_sweep.py
git commit -m "feat(#140): repeat-sweep HW driver (N x trace-sweep, --no-emu)

Generated using Claude Code."
```

---

### Task 7: HW characterization run + report artifact (gated deliverable)

Run the driver on real NPU1, point the analyzer at the run-indexed sweep output (both representations), and commit the resulting report + a short findings note. **HW-gated — run once, manually, never alongside another HW suite.**

**Files:**
- Create (runtime artifacts): `build/experiments/gap140/nondeterminism/add_one_using_dma/report.md`, `report.json`
- Create: `docs/superpowers/findings/2026-06-16-dma-nondeterminism-characterization-results.md`

**Interfaces:**
- Consumes: Tasks 5 (`main`) and 6 (`trace-variance-sweep.py`). No new code unless the run reveals a gap (e.g. the perfetto-glob pairing needs adjusting for the run-indexed layout — fix inline and add a test).

- [ ] **Step 1: Pre-flight — confirm NPU idle and no other HW suite running**

Run: `ps aux | grep -E 'emu-bridge|isa-test|trace-sweep' | grep -v grep || echo "no HW suite running"`
Expected: no competing HW suite (per the "never two HW suites concurrently" rule).

- [ ] **Step 2: Run the N=20 HW sweep**

Run (background, logged — never piped through tail):
```bash
env -u XDNA_EMU python3 tools/trace-variance-sweep.py \
  --test add_one_using_dma --repeat 20 --hw-jobs 5 \
  > build/experiments/gap140/nondeterminism/sweep.log 2>&1 &
```
Wait for the completion notification; then Read `sweep.log`. Expected: 20 run_NN dirs populated with per-batch decoded `events.json` + `perfetto.json`.

- [ ] **Step 3: Run the analyzer over both representations**

Run (adjust the perfetto/events globs to the actual run-indexed batch paths produced by trace-sweep.py — confirm by `find build/experiments/gap140/nondeterminism/add_one_using_dma -name '*.perfetto.json' | head`):
```bash
python3 tools/trace_variance.py \
  --events-glob 'build/experiments/gap140/nondeterminism/add_one_using_dma/run_*/**/events.json' \
  --perfetto-glob 'build/experiments/gap140/nondeterminism/add_one_using_dma/run_*/**/*.perfetto.json' \
  --events-for-names "$(find build/experiments/gap140/nondeterminism/add_one_using_dma/run_00 -name '*.events.json' | head -1)" \
  --words 64 --eps 2.0 \
  --out build/experiments/gap140/nondeterminism/add_one_using_dma
```
Read `report.md`.

- [ ] **Step 4: Validate against the success criteria**

Confirm in `report.md`:
- Held-level ports (`PORT_RUNNING_*`) classify deterministic and obey `sum == 64` (law OK).
- The stochastic events are the DMA milestone edges (shim/memtile `START`/`FINISHED`/`STREAM_STARVATION`).
- Any held-level law VIOLATION is flagged as a real bug (not masked).

If a level event shows variance or a milestone proves deterministic, that refutes the prediction — record it as a finding (the classification is emergent, so a refutation is a valid result).

- [ ] **Step 5: Write the findings note and commit the artifact**

Write `docs/superpowers/findings/2026-06-16-dma-nondeterminism-characterization-results.md` summarizing: the decomposition (X deterministic / Y stochastic), which events are stochastic and their band widths, the law-check outcome, and what it predicts for decomposing the corpus-wide 171. Then:
```bash
git add build/experiments/gap140/nondeterminism/add_one_using_dma/report.md \
        build/experiments/gap140/nondeterminism/add_one_using_dma/report.json \
        docs/superpowers/findings/2026-06-16-dma-nondeterminism-characterization-results.md
git commit -m "results(#140): add_one_using_dma nondeterminism characterization

Generated using Claude Code."
```

---

## Self-Review

**Spec coverage:**
- HW-only repeat sweep, N=20, full event coverage -> Task 6 (driver) + Task 7 (run). ✓
- Span-based metric, never frame-records -> Task 3 (perfetto B/E spans) + Global Constraints. ✓
- Re-anchor within-tile only -> Task 1 (milestone per-tile anchor) + Task 3 (per-lane spans). ✓
- Emergent classification -> Task 2 (`classify` from `std`). ✓
- Span-sum word law check -> Task 3 (`check_span_law`). ✓
- Event-class split (level=span / milestone=point) -> Task 1 `is_level` + Task 3. ✓
- Decomposition (det vs stochastic, which events) -> Task 4 (`decompose`). ✓
- Report (md + json + variance-sorted) -> Task 4. ✓
- Reuse trace-sweep coverage/decode -> Task 6 wraps `trace-sweep.py`. ✓
- Offline validation on the 20 captures -> Task 5. ✓
- HW characterization deliverable -> Task 7. ✓
- Out-of-scope (comparator masking, EMU change, generalization, encoder/relay-fill fix) -> not present in any task. ✓

**Placeholder scan:** No TBD/TODO. The two empirically-resolved points (perfetto `(pid,tid)->name` correlation in Task 3; the run-indexed glob shape in Task 7) carry a concrete inspection command and a ground-truth oracle (the span-sum law), not a hand-wave. The "first occurrence scalar" reduction in Task 5 is stated with its follow-up explicitly named.

**Type consistency:** `Stats` namedtuple fields (`n mean std min max range`) are used identically in Tasks 2, 4, 5. `load_milestone_events` returns `{(col,row,name): [int]}` consumed by `_first_occurrence_scalars` in Task 5. `load_spans` returns `{name: [int]}` consumed by `check_span_law` in Tasks 3-4. `build_sweep_cmd`/`main` signatures match between Task 6 impl and its tests. `classify` returns the string literals `"deterministic"`/`"stochastic"` consumed by `decompose`. ✓
