# Explicit Jitter-Robust Grounding Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inference engine's statistical grounding rule (`std <= eps` over cross-run offsets) with an explicit structural rule that grounds within-timer-domain edge segments to exact cycle offsets (cross-run equality) and reports everything else as named gaps.

**Architecture:** Edge existence + orientation already come from the static config_path/program_path facts (unchanged). This plan rewrites only the *measurement* layer: new primitives `offset_exact` (range <= Q) and `anchor_rigid` replace `correlates`/`deterministic`; a new `grounding.py` classifies each edge as a `Segment` (same per-module timer domain AND exact cross-run offset) or a `Gap` (cross-domain, or within-domain but non-exact = bundles a wait); a falsifier triad (ordering / lock-handoff / additivity) rejects rules that violate structural invariants; `derives` facts gain a `segment|gap` kind. No statistics anywhere: exact agreement is equality, not a tolerance.

**Tech Stack:** Python 3 (offline `pytest`, run under the ironenv); the inference module at `tools/inference/`; trace data accessed via `trace_join` (`tj`) and `trace_variance` (`tv`).

## Global Constraints

- **NO statistical inference.** No median, MAD, outlier tolerance, or tuned epsilon. The deterministic/jitter discriminator is exact cross-run agreement (range `<= Q`), which is equality. `Q = 0` for edge events, empirically confirmed (spike findings, range 0 over 20 HW runs). `Q` is a documented measurement-floor constant, never a value chosen to pass a test.
- **DERIVE FROM THE TOOLCHAIN.** Nothing hardcoded to a kernel. Domain identity is read from the event key; offsets are read from captured trace data.
- **Timer domain key is per-MODULE: `(col, row, pkt_type)`, NOT `(col, row)`** (C1, findings doc). Two events on the same tile but different modules are CROSS-domain.
- **Orientation is unchanged** — it stays a config_path/program_path structural fact; this plan never touches how orientation is derived, only how offsets are measured/classified.
- **`kind="program"` is reused, not extended** — the instruction-event layer (merged 2026-06-23) already produces the oriented core ACQUIRE→RELEASE candidate pair. This plan CONSUMES it; it does not re-expose it.
- **Run tests bare**, from `tools/`, with `env -u XDNA_EMU PYTHONPATH=tools <ironenv>/python -m pytest ...`. Offline tests use `tmp_path` fixtures (no NPU). HW tests are gated on `XDNA_HW_SMOKE=1`.
- **Offset convention:** `offset_exact(a, b)` and `pair_derivability(a, b)` measure `a - b` (child − parent). A `Segment(parent, child, offset)` has `offset = child_ts - parent_ts >= 0`.

**Spec:** `docs/superpowers/specs/2026-06-23-explicit-jitter-robust-grounding-design.md` (reconciled 2026-06-23).
**Evidence:** `docs/superpowers/findings/2026-06-23-jitter-grounding-spikes.md`.

---

### Task 1: Exact-agreement measurement primitives (`offset_exact`, `anchor_rigid`, `Q`)

Additive change to `verifier.py` — adds the two exact primitives alongside the existing `correlates`/`deterministic` (which stay until their consumers migrate in later tasks). `pair_derivability` already returns a `tv.Stats` namedtuple carrying a `range` field, so these are thin range-gated wrappers.

**Files:**
- Modify: `tools/inference/verifier.py` (add `Q`, `offset_exact`, `anchor_rigid` after the existing primitives)
- Test: `tools/test_inference_verifier.py` (add tests; existing tests untouched in this task)

**Interfaces:**
- Consumes: `tj.pair_derivability(run_dirs, a, b, anchor_key) -> Optional[tv.Stats]` where `Stats` has fields `n mean std min max range`; existing `_anchored_per_run(run_dirs, event_key, anchor_key) -> List[Dict[str,int]]`; `tv.aggregate(per_run) -> Dict[str, Stats]`.
- Produces: `Q: int = 0`; `offset_exact(run_dirs, a, b, anchor_key=ANCHOR) -> Optional[int]`; `anchor_rigid(run_dirs, e, anchor_key=ANCHOR) -> bool`.

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_inference_verifier.py` (reuse the existing `_make_runs` helper already in that file):

```python
from inference.verifier import offset_exact, anchor_rigid, Q


def test_offset_exact_returns_offset_when_range_zero(tmp_path):
    # child = parent + 22 in EVERY run (range 0) -> exact offset 22
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 122}, {"S": 200, "A": 222},
                                 {"S": 350, "A": 372}])
    assert offset_exact(dirs, "1|0|0|A", "1|0|0|S") == 22


def test_offset_exact_none_when_offset_varies_by_one(tmp_path):
    # range 1 (not exact) -> None under Q=0, no tolerance
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 122}, {"S": 200, "A": 223},
                                 {"S": 350, "A": 372}])
    assert offset_exact(dirs, "1|0|0|A", "1|0|0|S") is None


def test_offset_exact_none_when_never_cotraced(tmp_path):
    dirs = _make_runs(tmp_path, [{"S": 100}, {"S": 200}])
    assert offset_exact(dirs, "1|0|0|A", "1|0|0|S") is None


def test_anchor_rigid_true_when_anchored_ts_identical(tmp_path):
    dirs = _make_runs(tmp_path, [{"D": 40}, {"D": 40}, {"D": 40}])
    assert anchor_rigid(dirs, "1|0|0|D") is True


def test_anchor_rigid_false_when_anchored_ts_varies(tmp_path):
    dirs = _make_runs(tmp_path, [{"D": 40}, {"D": 41}, {"D": 40}])  # range 1
    assert anchor_rigid(dirs, "1|0|0|D") is False


def test_q_is_zero():
    assert Q == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_verifier.py -k "offset_exact or anchor_rigid or q_is_zero" -v`
Expected: FAIL with `ImportError: cannot import name 'offset_exact'`.

- [ ] **Step 3: Implement the primitives**

In `tools/inference/verifier.py`, after the `EPS = 2.0` line add:

```python
# Q -- the measurement floor for exact-agreement grounding. Within-domain
# edge-event offsets agree EXACTLY across runs (range 0 over 20 HW runs, spike
# findings). Q is a measured toolchain property, NOT a tuned tolerance: if a
# future kernel exposes a genuine discrete trace-frame quantum it is documented
# as that quantum, never a value chosen to pass a test.
Q = 0
```

After the `deterministic(...)` function add:

```python
def offset_exact(run_dirs: List[str], a: str, b: str,
                 anchor_key: str = ANCHOR) -> Optional[int]:
    """The exact within-execution offset (a - b) iff it agrees across all
    co-traced runs (cross-run range <= Q). None if never co-traced or non-exact.
    Replaces `correlates`: equality, not std <= eps."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is None or st.range > Q:
        return None
    return int(st.mean)  # range <= Q == 0 -> min == max == mean, exact


def anchor_rigid(run_dirs: List[str], event_key: str,
                 anchor_key: str = ANCHOR) -> bool:
    """e's anchored first-occurrence agrees exactly across runs (range <= Q).
    Replaces std-based `deterministic`."""
    per_run = _anchored_per_run(run_dirs, event_key, anchor_key)
    stats = tv.aggregate(per_run).get(event_key)
    return stats is not None and stats.range <= Q
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_verifier.py -v`
Expected: PASS (new tests pass; pre-existing `correlates`/`deterministic` tests still pass — untouched).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/verifier.py tools/test_inference_verifier.py
git commit -m "feat(#140): exact-agreement primitives offset_exact/anchor_rigid (Q=0)"
```

---

### Task 2: Falsifier triad (`check_ordering`, `check_lock_handoff`, `check_additivity`)

Additive change to `verifier.py`. Each falsifier returns `None` when the invariant holds and a `RejectedRule` on the first violation. Vacuous-true (e.g. a single-segment additivity chain) returns `None`.

**Files:**
- Modify: `tools/inference/verifier.py` (add three falsifiers after `anchor_rigid`)
- Test: `tools/test_inference_verifier.py` (add tests)

**Interfaces:**
- Consumes: `tj.batch_firsts`, `tj._batch_names`, the `RejectedRule` dataclass (already defined in `verifier.py`), `offset_exact` (Task 1).
- Produces:
  - `check_ordering(run_dirs, edges, anchor_key=ANCHOR) -> Optional[RejectedRule]` — `edges: List[Tuple[parent, child]]`.
  - `check_lock_handoff(run_dirs, lock_pairs, anchor_key=ANCHOR) -> Optional[RejectedRule]` — `lock_pairs: List[Tuple[release, acquire]]`.
  - `check_additivity(run_dirs, chain, anchor_key=ANCHOR) -> Optional[RejectedRule]` — `chain: List[str]` (ordered within-domain keys, parent-first).

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_inference_verifier.py`:

```python
from inference.verifier import (check_ordering, check_lock_handoff,
                                check_additivity)


def test_check_ordering_passes_when_parent_precedes_child(tmp_path):
    dirs = _make_runs(tmp_path, [{"P": 10, "C": 30}, {"P": 20, "C": 50}])
    assert check_ordering(dirs, [("1|0|0|P", "1|0|0|C")]) is None


def test_check_ordering_rejects_when_child_precedes_parent(tmp_path):
    dirs = _make_runs(tmp_path, [{"P": 30, "C": 10}, {"P": 20, "C": 50}])
    rej = check_ordering(dirs, [("1|0|0|P", "1|0|0|C")])
    assert rej is not None and rej.name == "ordering"
    assert rej.evidence["edge"] == ("1|0|0|P", "1|0|0|C")


def test_check_lock_handoff_passes_when_release_precedes_acquire(tmp_path):
    dirs = _make_runs(tmp_path, [{"REL": 10, "ACQ": 12}, {"REL": 20, "ACQ": 22}])
    assert check_lock_handoff(dirs, [("1|0|0|REL", "1|0|0|ACQ")]) is None


def test_check_lock_handoff_rejects_when_acquire_precedes_release(tmp_path):
    dirs = _make_runs(tmp_path, [{"REL": 30, "ACQ": 12}, {"REL": 20, "ACQ": 22}])
    rej = check_lock_handoff(dirs, [("1|0|0|REL", "1|0|0|ACQ")])
    assert rej is not None and rej.name == "lock_handoff"


def test_check_additivity_vacuous_for_single_segment(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10}, {"A": 0, "B": 10}])
    assert check_additivity(dirs, ["1|0|0|A", "1|0|0|B"]) is None  # < 3 keys


def test_check_additivity_passes_when_offsets_sum(tmp_path):
    # B = A+10, C = A+30 every run -> offset(A,C)=30 == 10 + 20
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10, "C": 30},
                                 {"A": 5, "B": 15, "C": 35}])
    assert check_additivity(dirs, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) is None


def test_check_additivity_rejects_when_offsets_do_not_sum(tmp_path):
    # exact segments A->B (10) and B->C (20) but A->C measured 999 -> contradiction
    dirs = _make_runs(tmp_path, [{"A": 0, "B": 10, "C": 999},
                                 {"A": 0, "B": 10, "C": 999}])
    rej = check_additivity(dirs, ["1|0|0|A", "1|0|0|B", "1|0|0|C"])
    assert rej is not None and rej.name == "additivity"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_verifier.py -k "ordering or lock_handoff or additivity" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the falsifiers**

In `tools/inference/verifier.py`, after `anchor_rigid`:

```python
def check_ordering(run_dirs: List[str], edges: List[Tuple[str, str]],
                   anchor_key: str = ANCHOR) -> Optional[RejectedRule]:
    """Every (parent, child) edge: parent fires no later than child, every
    co-traced batch. RejectedRule on the first violation."""
    for parent, child in edges:
        for rd in run_dirs:
            for bn in tj._batch_names(rd):
                f = tj.batch_firsts(rd, bn, anchor_key)
                if parent in f and child in f:
                    if f[parent] > f[child]:
                        return RejectedRule(
                            "ordering",
                            f"{parent} ({f[parent]}) > {child} ({f[child]})",
                            {"edge": (parent, child), "run": rd, "batch": bn})
                    break  # first co-tracing batch per run
    return None


def check_lock_handoff(run_dirs: List[str], lock_pairs: List[Tuple[str, str]],
                       anchor_key: str = ANCHOR) -> Optional[RejectedRule]:
    """Every (release, acquire) lock pair: release fires no later than the
    matching acquire, every co-traced batch."""
    for rel, acq in lock_pairs:
        for rd in run_dirs:
            for bn in tj._batch_names(rd):
                f = tj.batch_firsts(rd, bn, anchor_key)
                if rel in f and acq in f:
                    if f[rel] > f[acq]:
                        return RejectedRule(
                            "lock_handoff",
                            f"release {rel} ({f[rel]}) > acquire {acq} ({f[acq]})",
                            {"pair": (rel, acq), "run": rd, "batch": bn})
                    break
    return None


def check_additivity(run_dirs: List[str], chain: List[str],
                     anchor_key: str = ANCHOR) -> Optional[RejectedRule]:
    """offset(chain[0] -> chain[-1]) must equal the sum of consecutive exact
    offsets. Vacuous (None) for a chain of < 3 keys (single segment). A
    non-exact consecutive offset is a gap, not an additivity violation -> None."""
    if len(chain) < 3:
        return None
    parts = []
    for i in range(len(chain) - 1):
        o = offset_exact(run_dirs, chain[i + 1], chain[i], anchor_key)
        if o is None:
            return None  # a gap in the chain: additivity does not apply
        parts.append(o)
    end_to_end = offset_exact(run_dirs, chain[-1], chain[0], anchor_key)
    if end_to_end is None:
        return None
    if end_to_end != sum(parts):
        return RejectedRule(
            "additivity",
            f"offset({chain[0]} -> {chain[-1]}) = {end_to_end} != sum {sum(parts)}",
            {"chain": chain, "parts": parts})
    return None
```

Confirm `Tuple` is imported in `verifier.py` (it is, in the existing `from typing import ...` line).

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_verifier.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/verifier.py tools/test_inference_verifier.py
git commit -m "feat(#140): falsifier triad (ordering/lock-handoff/additivity)"
```

---

### Task 3: `grounding.py` — `same_domain`, `Segment`/`Gap`, `ground_edge`, `Timeline`, `assemble`

New module. Classifies a static edge into a `Segment` (exact, same per-module domain) or a `Gap`, and assembles a chain of edges into an ordered timeline.

**Files:**
- Create: `tools/inference/grounding.py`
- Test: `tools/test_inference_grounding.py`

**Interfaces:**
- Consumes: `offset_exact`, `ANCHOR`, `Q` from `inference.verifier`.
- Produces:
  - `same_domain(a, b) -> bool`
  - `Segment(parent, child, offset)` (frozen dataclass), `Gap(parent, child)` (frozen dataclass), `Grounding = Union[Segment, Gap]`
  - `ground_edge(run_dirs, child, parent, anchor_key=ANCHOR) -> Grounding`
  - `Timeline(items: List[Grounding])` (dataclass), `assemble(run_dirs, edges, anchor_key=ANCHOR) -> Timeline` where `edges: List[Tuple[parent, child]]`.

- [ ] **Step 1: Write the failing tests**

Create `tools/test_inference_grounding.py`:

```python
import json
from inference.grounding import (same_domain, ground_edge, assemble,
                                 Segment, Gap, Timeline)


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    """rows: list over runs of {key: anchored_offset}. Keys are full
    'col|row|pkt|name' strings so a test can place events in any domain."""
    dirs = []
    for i, row in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_same_domain_true_for_identical_col_row_pkt():
    assert same_domain("1|2|0|ACQUIRE", "1|2|0|RELEASE") is True


def test_same_domain_false_for_different_pkt_type():
    # same tile (1,2) but different module (pkt 0 vs 3) -> CROSS domain (C1)
    assert same_domain("1|2|0|A", "1|2|3|B") is False


def test_same_domain_false_for_different_tile():
    assert same_domain("1|2|0|A", "1|0|0|B") is False


def test_ground_edge_segment_when_same_domain_and_exact(tmp_path):
    dirs = _runs(tmp_path, [{"1|2|0|ACQ": 0, "1|2|0|REL": 22},
                            {"1|2|0|ACQ": 50, "1|2|0|REL": 72}])
    g = ground_edge(dirs, "1|2|0|REL", "1|2|0|ACQ")
    assert g == Segment(parent="1|2|0|ACQ", child="1|2|0|REL", offset=22)


def test_ground_edge_gap_when_same_domain_but_nonexact(tmp_path):
    dirs = _runs(tmp_path, [{"1|2|0|ACQ": 0, "1|2|0|REL": 22},
                            {"1|2|0|ACQ": 50, "1|2|0|REL": 73}])  # range 1
    g = ground_edge(dirs, "1|2|0|REL", "1|2|0|ACQ")
    assert g == Gap(parent="1|2|0|ACQ", child="1|2|0|REL")


def test_ground_edge_gap_when_cross_domain_even_if_exact(tmp_path):
    # exact offset (30 every run) but different modules (shim pkt 2 vs core pkt 0)
    # -> still a gap (cross-domain timer skew is not groundable as a segment)
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 35}])
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE")


def test_assemble_interleaves_segments_and_gaps(tmp_path):
    # chain: shim MM2S (parent) -> core ACQ -> core REL.
    # MM2S->ACQ is cross-domain (gap); ACQ->REL is within-domain exact (segment).
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|ACQ": 30, "1|2|0|REL": 52},
                            {"1|0|2|MM2S": 7, "1|2|0|ACQ": 40, "1|2|0|REL": 62}])
    tl = assemble(dirs, [("1|0|2|MM2S", "1|2|0|ACQ"), ("1|2|0|ACQ", "1|2|0|REL")])
    assert isinstance(tl, Timeline)
    assert tl.items[0] == Gap(parent="1|0|2|MM2S", child="1|2|0|ACQ")
    assert tl.items[1] == Segment(parent="1|2|0|ACQ", child="1|2|0|REL", offset=22)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_grounding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.grounding'`.

- [ ] **Step 3: Implement `grounding.py`**

Create `tools/inference/grounding.py`:

```python
"""Explicit grounding: classify a static causal edge as a cycle-exact within-
domain Segment or a named Gap, and assemble a chain into a timeline.

The deterministic, cycle-accurate unit is a segment bounded by milestone events
WITHIN one per-module timer domain whose per-run offset agrees EXACTLY (range
<= Q == 0). Everything else -- cross-domain offsets, and within-domain offsets
that bundle a delivery wait (non-exact) -- is a Gap: existence + orientation
only, no cycle count. A through-core span is therefore reported as
gap + (exact segment) + gap, never as one deterministic number.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union
from inference.verifier import offset_exact, ANCHOR, Q  # noqa: F401 (Q documents the floor)


def same_domain(a: str, b: str) -> bool:
    """a and b share a per-module timer domain iff their col|row|pkt_type prefix
    matches. The trace timer resets per (pkt_type, row, col) (C1), so two events
    on the same tile but different modules are CROSS-domain."""
    return a.rsplit("|", 1)[0] == b.rsplit("|", 1)[0]


@dataclass(frozen=True)
class Segment:
    parent: str
    child: str
    offset: int


@dataclass(frozen=True)
class Gap:
    parent: str
    child: str


Grounding = Union[Segment, Gap]


def ground_edge(run_dirs: List[str], child: str, parent: str,
                anchor_key: str = ANCHOR) -> Grounding:
    """Segment iff parent and child share a timer domain AND their cross-run
    offset is exact (range <= Q); otherwise a named Gap."""
    if same_domain(child, parent):
        off = offset_exact(run_dirs, child, parent, anchor_key)
        if off is not None:
            return Segment(parent=parent, child=child, offset=off)
    return Gap(parent=parent, child=child)


@dataclass
class Timeline:
    items: List[Grounding]


def assemble(run_dirs: List[str], edges: List[Tuple[str, str]],
             anchor_key: str = ANCHOR) -> Timeline:
    """edges: ordered [(parent, child)] forming a static causal chain. Returns a
    Timeline of per-edge groundings (exact segments interleaved with named
    gaps), in chain order."""
    return Timeline([ground_edge(run_dirs, child, parent, anchor_key)
                     for parent, child in edges])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_grounding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/grounding.py tools/test_inference_grounding.py
git commit -m "feat(#140): grounding.py -- same_domain/ground_edge/assemble (segment vs gap)"
```

---

### Task 4: `facts.py` — `derives` kind discriminator + accessors

The `derives` fact's args grow from `(child, parent, offset)` to `(child, parent, offset, kind)` where `kind in {"segment", "gap"}` and `offset` is an `int` for segments, `None` for gaps. Backward-compatible: existing readers of `args[0..2]` keep working; legacy 3-arg facts read as segments.

**Files:**
- Modify: `tools/inference/facts.py` (add two module-level accessor functions)
- Test: `tools/test_inference_facts.py` (add tests)

**Interfaces:**
- Consumes: the `Fact` dataclass (already defined).
- Produces: `derive_kind(f: Fact) -> str`; `derive_offset(f: Fact) -> Optional[int]`.

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_inference_facts.py`:

```python
from inference.facts import Fact, Derived, Measured, derive_kind, derive_offset


def _leaf(pred, args):
    return Fact(pred, args, Measured())


def test_derive_kind_and_offset_for_segment():
    prem = _leaf("fired", ("1|2|0|REL",))
    f = Fact("derives", ("1|2|0|REL", "1|2|0|ACQ", 22, "segment"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_kind(f) == "segment"
    assert derive_offset(f) == 22


def test_derive_kind_and_offset_for_gap():
    prem = _leaf("fired", ("1|0|2|S2MM",))
    f = Fact("derives", ("1|0|2|S2MM", "1|0|2|MM2S", None, "gap"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_kind(f) == "gap"
    assert derive_offset(f) is None


def test_derive_kind_legacy_three_arg_reads_as_segment():
    prem = _leaf("fired", ("1|0|0|C",))
    f = Fact("derives", ("1|0|0|C", "1|0|0|S", 30),
             Derived("derives_rule_placement", (prem,)))
    assert derive_kind(f) == "segment"
    assert derive_offset(f) == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_facts.py -k derive -v`
Expected: FAIL with `ImportError: cannot import name 'derive_kind'`.

- [ ] **Step 3: Implement the accessors**

In `tools/inference/facts.py`, after the `leaves(...)` function add:

```python
def derive_kind(fact: Fact) -> str:
    """segment | gap for a `derives` fact. Legacy 3-arg facts (offset only)
    read as segments -- backward-compatible."""
    return fact.args[3] if len(fact.args) >= 4 else "segment"


def derive_offset(fact: Fact) -> Optional[int]:
    """The exact cycle offset for a segment derive; None for a gap."""
    return fact.args[2]
```

`Optional` is already imported in `facts.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_facts.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/facts.py tools/test_inference_facts.py
git commit -m "feat(#140): derives kind discriminator (segment|gap) + accessors"
```

---

### Task 5: `rules.py` + `chainer.py` — `try_derives` rewrite (segment/gap) and `anchor_rigid` determinism

Rewrite the grounding decision: orientation stays a config/program_path lookup; the stochastic-parent gate moves to `anchor_rigid`; ordering is checked as a falsifier (rejection recorded, returns `None`); grounding becomes a `Segment` (kind `segment`, exact offset) or a `Gap` (kind `gap`, offset `None`) derive. `mark_determinism` switches to `anchor_rigid`. `chainer` deduplicates derives by (child, parent) regardless of offset/kind, and drops the now-unused `eps` threading.

**Files:**
- Modify: `tools/inference/rules.py`
- Modify: `tools/inference/chainer.py`
- Test: `tools/test_inference_rules.py` (rewrite affected tests)
- Test: `tools/test_inference_chainer.py` (verify still green; adjust fixtures if std-tolerant)

**Interfaces:**
- Consumes: `anchor_rigid`, `check_ordering`, `RejectedRule`, `ANCHOR` from `inference.verifier`; `ground_edge`, `Segment` from `inference.grounding`; `Fact`, `Derived`, `KB` from `inference.facts`.
- Produces (changed signatures — `eps` removed):
  - `mark_determinism(run_dirs, kb, event_keys, anchor_key=ANCHOR) -> None`
  - `try_derives(run_dirs, kb, child, parent, anchor_key=ANCHOR) -> Optional[Fact]` (derives fact now 4-arg)
  - unchanged: `try_same_source`, `is_stochastic_root`, `_measured_premises`
  - `chain(run_dirs, kb, candidate_pairs, anchor_key=ANCHOR) -> KB`; new helper `_has_derive(kb, child, parent) -> bool`; `_existing_offset` removed.

- [ ] **Step 1: Rewrite the failing tests**

Replace the body of `tools/test_inference_rules.py` (keep `_ev`, `_runs`, `_kb_with_ledger` helpers; replace the test functions below). The key change: exact fixtures (range 0) for segment cases, and a NEW gap case.

```python
def test_derives_segment_with_stochastic_parent_within_domain(tmp_path):
    # parent S jitters; child C = S + 30 EVERY run (range 0); same domain (1|0|0);
    # config routes S -> C -> a SEGMENT derive with exact offset 30.
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 380}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    from inference.facts import derive_kind, derive_offset, provenance_ok
    assert f is not None and f.predicate == "derives"
    assert f.args[:2] == ("1|0|0|C", "1|0|0|S")
    assert derive_kind(f) == "segment" and derive_offset(f) == 30
    kb.add(f)
    assert provenance_ok(kb) is True


def test_derives_gap_when_within_domain_offset_nonexact(tmp_path):
    # offset C-S varies (30, 30, 31) -> not exact -> GAP derive (placed, no offset)
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 381}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    from inference.facts import derive_kind, derive_offset
    assert f is not None and derive_kind(f) == "gap" and derive_offset(f) is None


def test_derives_rejected_when_parent_rigid(tmp_path):
    # parent D is rigid (range 0 anchored) -> not a stochastic source -> None
    dirs = _runs(tmp_path, [{"D": 40, "C": 70}, {"D": 40, "C": 70},
                            {"D": 40, "C": 70}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|D", "b": "1|0|0|C", "kind": "route"}])
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|D") is None


def test_derives_rejected_without_orientation(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230}])
    kb = KB.empty()  # empty ledger -> no config_path/program_path
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|S") is None


def test_derives_ordering_violation_is_rejected_and_recorded(tmp_path):
    # config routes S -> C but C fires BEFORE S every run -> ordering falsifier
    dirs = _runs(tmp_path, [{"S": 130, "C": 100}, {"S": 230, "C": 200}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|S") is None
    assert any(r.name == "ordering" for r in kb.rejected_rules)


def test_derives_places_backpressure_event_without_causal_claim(tmp_path):
    dirs = _runs(tmp_path, [{"P": 100, "SS": 130}, {"P": 220, "SS": 250},
                            {"P": 300, "SS": 330}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "route#9", "a": "1|0|0|P", "b": "1|0|0|SS", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|SS", "1|0|0|P")
    assert f is not None and f.predicate == "derives"
    assert "caus" not in f.support.rule.lower()


def test_try_derives_consumes_program_path(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 380}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "program:1|0|0|S--via-core-->1|0|0|C", "a": "1|0|0|S",
         "b": "1|0|0|C", "kind": "program"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    from inference.facts import derive_kind
    assert f is not None and derive_kind(f) == "segment"
    assert kb.by_predicate("program_path") != []
    assert kb.by_predicate("config_path") == []


def test_same_source_requires_identity_and_coincidence(tmp_path):
    dirs = _runs(tmp_path, [{"A": 40, "A2": 40}, {"A": 41, "A2": 41}])
    kb = _kb_with_ledger(tmp_path, dirs, [
        {"cite": "id#1", "a": "1|0|0|A", "b": "1|0|0|A2", "kind": "identity"}])
    f = try_same_source(dirs, kb, "1|0|0|A", "1|0|0|A2")
    assert f is not None and f.predicate == "same_source"


def test_stochastic_root_when_jittery_and_underived(tmp_path):
    dirs = _runs(tmp_path, [{"R": 40}, {"R": 90}, {"R": 140}])
    kb = KB.empty()
    for f in load_fired(dirs):
        kb.add(f)
    mark_determinism(dirs, kb, ["1|0|0|R"])
    assert is_stochastic_root(kb, "1|0|0|R") is True
```

Note: `test_same_source_requires_identity_and_coincidence` still uses `coincident` (std-based), which is untouched until Task 9. It stays green here because `coincident` is unchanged in this task.

- [ ] **Step 2: Run tests to verify they fail**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_rules.py -v`
Expected: FAIL (segment/gap/kind assertions and `kb.rejected_rules` not yet produced).

- [ ] **Step 3: Rewrite `try_derives` and `mark_determinism` in `rules.py`**

Replace the imports and the `mark_determinism` / `try_derives` functions in `tools/inference/rules.py`:

```python
from inference.facts import Fact, Derived, KB
from inference.verifier import anchor_rigid, check_ordering, ANCHOR
from inference.grounding import ground_edge, Segment
```

```python
def mark_determinism(run_dirs: List[str], kb: KB, event_keys: List[str],
                     anchor_key: str = ANCHOR) -> None:
    for ek in event_keys:
        is_det = anchor_rigid(run_dirs, ek, anchor_key)
        pred = "deterministic" if is_det else "stochastic"
        premises = tuple(f for f in kb.by_predicate("fired") if f.args[0] == ek)
        kb.add(Fact(pred, (ek,), Derived("determinism_rule", premises)))


def try_derives(run_dirs: List[str], kb: KB, child: str, parent: str,
                anchor_key: str = ANCHOR) -> Optional[Fact]:
    # (1) orientation by verified rule: config_path OR program_path
    cp = next((f for f in (kb.by_predicate("config_path") + kb.by_predicate("program_path"))
               if f.args[0] == parent and f.args[1] == child), None)
    if cp is None:
        return None
    # (2) parent must be a stochastic source -- a rigid parent transmits no jitter
    if anchor_rigid(run_dirs, parent, anchor_key):
        return None
    # (3) falsifier: ordering must hold on this edge in every run
    rej = check_ordering(run_dirs, [(parent, child)], anchor_key)
    if rej is not None:
        kb.rejected_rules.append(rej)
        return None
    # (4) grounding: exact within-domain segment, else a named gap. Both are
    # PLACED (existence + orientation); only the segment carries a cycle offset.
    g = ground_edge(run_dirs, child, parent, anchor_key)
    if isinstance(g, Segment):
        grd = Fact("segment", (child, parent, g.offset),
                   Derived("grounding_rule", _measured_premises(kb, child, parent)))
        return Fact("derives", (child, parent, g.offset, "segment"),
                    Derived("derives_rule_placement", (cp, grd)))
    grd = Fact("gap", (child, parent),
               Derived("grounding_rule", _measured_premises(kb, child, parent)))
    return Fact("derives", (child, parent, None, "gap"),
                Derived("derives_rule_placement", (cp, grd)))
```

Update `try_same_source` and `is_stochastic_root` signatures only to drop the now-unused `eps` parameter from `try_same_source` (it still calls `coincident(run_dirs, a, b, anchor_key)` — drop the `eps` arg). Leave `coincident` itself for Task 9. `is_stochastic_root` is unchanged.

- [ ] **Step 4: Update `chainer.py`**

In `tools/inference/chainer.py`: drop the `EPS` import and the `eps` parameter; replace `_existing_offset` dedup with a `(child, parent)` existence check.

Replace the import line `from inference.verifier import ANCHOR, EPS` with `from inference.verifier import ANCHOR`.

Replace `chain(...)` signature and body:

```python
def chain(run_dirs: List[str], kb: KB,
          candidate_pairs: Iterable[Tuple[str, str]],
          anchor_key: str = ANCHOR) -> KB:
    pairs = list(candidate_pairs)
    keys = _fired_event_keys(kb)
    undetermined = [k for k in keys
                    if not (kb.has("deterministic", (k,)) or kb.has("stochastic", (k,)))]
    if undetermined:
        mark_determinism(run_dirs, kb, undetermined, anchor_key)

    changed = True
    while changed:
        changed = False
        for a, b in pairs:
            if not _has_derive(kb, a, b):
                d = try_derives(run_dirs, kb, a, b, anchor_key)
                if d is not None and not kb.has(d.predicate, d.args):
                    kb.add(d); changed = True
            if not _has_same_source(kb, a, b):
                s = try_same_source(run_dirs, kb, a, b, anchor_key)
                if s is not None and not kb.has(s.predicate, s.args):
                    kb.add(s); changed = True
    return kb


def _has_derive(kb: KB, child: str, parent: str) -> bool:
    return any(f.args[0] == child and f.args[1] == parent
               for f in kb.by_predicate("derives"))
```

Delete the `_existing_offset` function.

- [ ] **Step 5: Run tests to verify they pass**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_rules.py test_inference_chainer.py -v`
Expected: PASS. If a pre-existing `test_inference_chainer.py` fixture used a range-1 "deterministic" case, tighten it to range 0 (anchored ts identical across runs) — exact-agreement has no tolerance.

- [ ] **Step 6: Commit**

```bash
git add tools/inference/rules.py tools/inference/chainer.py tools/test_inference_rules.py tools/test_inference_chainer.py
git commit -m "feat(#140): try_derives -> segment/gap grounding + anchor_rigid + ordering falsifier"
```

---

### Task 6: `engine.py` + `run_experiment.py` — segment/gap/rejected reporting

Surface the new structure in reports: each derived edge as a `segment` (with exact offset) or a `gap`, plus the `rejected_rules` recorded by the falsifiers. The legacy `derives` list stays (now 4-tuples) for backward compatibility.

**Files:**
- Modify: `tools/inference/engine.py`
- Modify: `tools/inference/run_experiment.py`
- Test: `tools/test_inference_engine.py`
- Test: `tools/test_experiment_report.py`

**Interfaces:**
- Consumes: `derive_kind`, `derive_offset` from `inference.facts`; `kb.rejected_rules`.
- Produces: `run_engine(...)` return dict gains `"segments": List[Tuple[child, parent, offset]]`, `"gaps": List[Tuple[child, parent]]`, `"rejected_rules": List[dict]`. `run_experiment(...)` report gains the same three keys (sourced from the engine report).

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_inference_engine.py` (mirror the existing engine test's run-dir + ledger construction; this test asserts the new keys exist and split correctly). Use the existing helpers in that file if present; otherwise this self-contained test:

```python
import json
from inference.engine import run_engine


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, row in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_engine_reports_segment_and_gap(tmp_path):
    # within-domain exact (1|0|0 S->C offset 30) -> segment;
    # cross-domain (shim 1|0|2 MM2S -> core 1|2|0 CORE) -> gap.
    dirs = _runs(tmp_path, [
        {"1|0|0|S": 100, "1|0|0|C": 130, "1|0|2|MM2S": 0, "1|2|0|CORE": 40},
        {"1|0|0|S": 200, "1|0|0|C": 230, "1|0|2|MM2S": 9, "1|2|0|CORE": 55}])
    led = tmp_path / "led.json"
    led.write_text(json.dumps({"entries": [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"},
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}]}))
    rep = run_engine(dirs, str(led),
                     [("1|0|0|C", "1|0|0|S"), ("1|2|0|CORE", "1|0|2|MM2S")])
    assert ("1|0|0|C", "1|0|0|S", 30) in rep["segments"]
    assert ("1|2|0|CORE", "1|0|2|MM2S") in rep["gaps"]
    assert isinstance(rep["rejected_rules"], list)
    assert rep["provenance_ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_engine.py -k segment_and_gap -v`
Expected: FAIL with `KeyError: 'segments'`.

- [ ] **Step 3: Add segment/gap/rejected to `engine.py`**

In `tools/inference/engine.py`, add the import and build the three lists in `run_engine` before the `return`:

```python
from inference.facts import KB, provenance_ok, derive_kind, derive_offset
```

```python
    derives_facts = kb.by_predicate("derives")
    segments = [(f.args[0], f.args[1], derive_offset(f))
                for f in derives_facts if derive_kind(f) == "segment"]
    gaps = [(f.args[0], f.args[1])
            for f in derives_facts if derive_kind(f) == "gap"]
    rejected = [{"name": r.name, "reason": r.reason, "evidence": r.evidence}
                for r in kb.rejected_rules]
```

Add to the returned dict:

```python
            "segments": segments,
            "gaps": gaps,
            "rejected_rules": rejected,
```

(Keep the existing `"derives": derives` line.)

- [ ] **Step 4: Thread the keys through `run_experiment.py`**

In `tools/inference/run_experiment.py`, in `run_experiment(...)`, after `derives = rep.get("derives", [])` add:

```python
        segments = rep.get("segments", [])
        gaps = rep.get("gaps", [])
        rejected_rules = rep.get("rejected_rules", [])
```

Initialize them in the pre-`try` defaults line — change:

```python
    derives, roots, provenance_ok, engine_ok = [], [], None, False
```
to:
```python
    derives, roots, provenance_ok, engine_ok = [], [], None, False
    segments, gaps, rejected_rules = [], [], []
```

And add to the returned report dict (next to `"derives": derives,`):

```python
        "segments": segments,
        "gaps": gaps,
        "rejected_rules": rejected_rules,
```

- [ ] **Step 5: Add a report-shape assertion to `test_experiment_report.py`**

Append to `test_run_experiment_with_mock_writes_report` (the mock loop test), after the existing assertions:

```python
    assert "segments" in loaded and "gaps" in loaded
    assert "rejected_rules" in loaded
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_engine.py test_experiment_report.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/inference/engine.py tools/inference/run_experiment.py tools/test_inference_engine.py tools/test_experiment_report.py
git commit -m "feat(#140): report segments/gaps/rejected_rules from engine + run_experiment"
```

---

### Task 7: `loop.py` + `planner.py` — exact-agreement wiring

Migrate the loop's empirical-limit ranking and the planner's co-trace gain check off `std > EPS` onto the exact equivalent (`range > Q` / `offset_exact`). The MockInstrument-based loop tests must stay green (a jitter=1 ground truth → range 1 > Q=0 → still recorded as `cannot_correlate`).

**Files:**
- Modify: `tools/inference/loop.py`
- Modify: `tools/inference/planner.py`
- Test: `tools/test_inference_loop.py`, `tools/test_inference_planner.py` (verify green; adjust only if a fixture relied on std tolerance)

**Interfaces:**
- Consumes: `tj.pair_derivability(...).range`; `offset_exact`, `Q` from `inference.verifier`.
- Produces (changed): `loop` no longer imports/uses `EPS`; `planner._co_traced(run_dirs, a, b, anchor_key) -> bool` (drops `eps`); `planner.propose_next(..., anchor_key=ANCHOR)` (drops `eps`).

- [ ] **Step 1: Update `loop.py`**

Replace the `from inference.verifier import ANCHOR, EPS` import with `from inference.verifier import ANCHOR, Q`.

At both empirical-limit sites (the seed-time check near line 124 and the act-time check near line 171), replace:

```python
            st = trace_join.pair_derivability(all_run_dirs, a, b, anchor_key)
            if st is not None and st.std > EPS:
```
with:
```python
            st = trace_join.pair_derivability(all_run_dirs, a, b, anchor_key)
            if st is not None and st.range > Q:
```

Update the two adjacent comments from "co-traced but std>eps" / "std>eps ->" to "co-traced but offset not exact (range > Q)".

- [ ] **Step 2: Update `planner.py`**

Replace `from inference.verifier import correlates, ANCHOR, EPS` with `from inference.verifier import offset_exact, ANCHOR`.

Replace `_co_traced` and `propose_next`:

```python
def _co_traced(run_dirs: List[str], a: str, b: str, anchor_key: str) -> bool:
    return offset_exact(run_dirs, a, b, anchor_key) is not None


def propose_next(kb: KB, run_dirs: List[str], pair: Tuple[str, str],
                 model: ReachabilityModel, anchor_key: str = ANCHOR):
    a, b = pair
    if not _co_traced(run_dirs, a, b, anchor_key):
        if model.can_separate(a, b) is False:
            return NO_GAIN
        return plan_cotrace(a, b)
    return NO_GAIN
```

Note: under exact agreement, `_co_traced` now means "co-traced AND exact". A co-traced-but-jittery pair reads as not-co-traced here, so the planner may propose another co-trace batch; the loop's empirical-limit check (Step 1) records `cannot_correlate` for it, so the loop still terminates. This is the intended behavior — verify the loop tests confirm it (Step 3).

- [ ] **Step 3: Run the loop + planner tests**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_loop.py test_inference_planner.py -v`
Expected: PASS. `test_uncorrelated_pair_halts_falsifiably_not_spins` (jitter=1) must still halt with a `cannot_correlate` constraint and `iterations < 12`. If `test_inference_planner.py` passes `eps=` to `propose_next`/`_co_traced`, drop that argument in the test.

- [ ] **Step 4: Commit**

```bash
git add tools/inference/loop.py tools/inference/planner.py tools/test_inference_loop.py tools/test_inference_planner.py
git commit -m "feat(#140): loop/planner exact-agreement wiring (range>Q, offset_exact)"
```

---

### Task 8: HW validation — remove the retry stopgap, ground the exact segment, fix falsifiability (HW-GATED)

Rewrite the Phoenix-gated tests in `test_experiment_loop_hw.py` for the new model. Three changes: (a) the through-core span is now a deterministically-derivable **gap** (no retry needed); (b) a new test asserts the core lock edge grounds to an exact **segment**; (c) falsifiability now targets **segment downgrade** (perturbing an exact segment's offset turns it into a gap), because perturbing a gap's offset cannot remove a gap.

This task requires a real NPU (`XDNA_HW_SMOKE=1`) and a built `add_one_using_dma` chess kernel. Run it once, after the offline suite is green.

**Files:**
- Modify: `tools/test_experiment_loop_hw.py`

**Interfaces:**
- Consumes: `run_experiment(cfg)` report's new `"segments"`, `"gaps"` keys (Task 6); `candidate_pairs_from_dump` already surfaces the core lock pair (instruction-event layer). The core lock event keys are `1|2|0|INSTR_LOCK_ACQUIRE_REQ` (parent) and `1|2|0|INSTR_LOCK_RELEASE_REQ` (child) for `add_one_using_dma` at start_col 1.

- [ ] **Step 1: Replace the through-core stopgap test with a single-run gap assertion**

Replace `test_loop_places_a_through_core_event_hw` (lines ~39-69) with:

```python
def test_through_core_event_is_placed_as_gap_hw(tmp_path):
    # The through-core (program_path) pair S2MM_0_START <- MM2S_0_START is
    # orientable ONLY via the core_lock_relay edge. It spans shim -> core -> shim
    # (cross timer-domain), so under explicit grounding it is a NAMED GAP:
    # existence + orientation, deterministically derivable every run -- NO retry.
    from inference.run_experiment import run_experiment
    target = "1|0|2|DMA_S2MM_0_START_TASK"
    rep = run_experiment(_cfg(tmp_path))
    assert rep["engine_ok"] is True
    children = {d[0] for d in rep["derives"]}
    assert target in children, f"through-core {target} not placed; derives={rep['derives']}"
    gap_children = {g[0] for g in rep["gaps"]}
    assert target in gap_children, f"{target} should be a gap (cross-domain), not a segment"
```

- [ ] **Step 2: Add the exact-segment grounding test**

Add after it:

```python
def test_core_lock_segment_grounds_exact_hw(tmp_path):
    # The core compute segment INSTR_LOCK_ACQUIRE_REQ -> INSTR_LOCK_RELEASE_REQ
    # is within ONE timer domain (core module) and exact across the run set ->
    # a SEGMENT with a cycle-accurate offset. This is the cycle-exact deliverable
    # the instruction-event layer made producible.
    from inference.run_experiment import run_experiment
    child = "1|2|0|INSTR_LOCK_RELEASE_REQ"
    parent = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
    rep = run_experiment(_cfg(tmp_path))
    seg = next((s for s in rep["segments"] if s[0] == child and s[1] == parent), None)
    assert seg is not None, f"core lock segment not grounded; segments={rep['segments']}"
    # Exact, positive offset (release after acquire). Value is kernel-specific
    # (~22 on add_one); assert it is a concrete int that agreed across the runs.
    assert isinstance(seg[2], int) and seg[2] > 0
```

- [ ] **Step 3: Rewrite the falsifiability test for segment downgrade**

Replace `test_forced_wrong_batch_changes_outcome_hw` (perturbing a level-event offset no longer removes a derive — it was already a gap) with a test that perturbs the exact core-lock segment so it downgrades to a gap:

```python
def test_perturbed_segment_downgrades_to_gap_hw(tmp_path):
    # Falsifiability: corrupt the EXACT core-lock segment's offset per-run so the
    # cross-run range != 0. The engine must DOWNGRADE it from a segment to a gap
    # (it can no longer claim a cycle-exact offset) -- existence/orientation
    # survive, the cycle count does not.
    import json, shutil
    from pathlib import Path
    from inference.run_experiment import run_experiment, KernelConfig
    from inference.engine import run_engine
    from inference.selfmodel import (enumerate_configured_events,
                                     candidate_pairs_from_dump)
    from config_extract.dump_model import load_dump

    cfg = _cfg(tmp_path)
    rep = run_experiment(cfg)
    child, parent = "1|2|0|INSTR_LOCK_RELEASE_REQ", "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
    assert any(s[0] == child and s[1] == parent for s in rep["segments"]), \
        "baseline must ground the core-lock segment to perturb it"

    # Perturb RELEASE's ts by a per-run-varying amount so the offset range != 0.
    pert = Path(cfg.out_root) / "perturbed"
    run_dirs = []
    for idx, rd in enumerate(sorted(p for p in Path(cfg.out_root).glob("capture_*/run_*"))):
        dst = pert / rd.relative_to(cfg.out_root)
        shutil.copytree(rd, dst)
        bump = (idx + 1) * 7  # per-run-varying -> offset range explodes
        for ev_path in dst.glob("batch_*/hw/trace.events.json"):
            doc = json.loads(ev_path.read_text())
            for e in doc["events"]:
                if e["name"] == "INSTR_LOCK_RELEASE_REQ" and e["col"] == 1 and e["row"] == 2:
                    e["ts"] += bump; e["soc"] += bump
            ev_path.write_text(json.dumps(doc))
        run_dirs.append(str(dst))

    dump = load_dump(cfg.dump_path)
    configured = enumerate_configured_events(dump, cfg.start_col)
    pairs = candidate_pairs_from_dump(dump, configured, cfg.start_col)
    led = Path(cfg.out_root) / "ledger.json"   # written by run_experiment
    perturbed = run_engine(run_dirs, str(led), pairs)
    seg_children = {s[0] for s in perturbed["segments"]}
    gap_children = {g[0] for g in perturbed["gaps"]}
    assert child not in seg_children, "perturbed segment must no longer be exact"
    assert child in gap_children, "perturbed edge must survive as a gap (placed)"
```

- [ ] **Step 4: Run the HW tests once (real NPU)**

Ensure the `.so` and kernel are built and the NPU is healthy (`xrt-smi validate` first if unsure). Then:

Run: `cd tools && XDNA_HW_SMOKE=1 env -u XDNA_EMU XDNA_EMU_RUNTIME=debug python -m pytest test_experiment_loop_hw.py -v`
Expected: PASS — `test_through_core_event_is_placed_as_gap_hw`, `test_core_lock_segment_grounds_exact_hw`, `test_perturbed_segment_downgrades_to_gap_hw`, `test_loop_converges_on_add_one_hw`, and the suite (`add_one_objFifo`, `vector_scalar_using_dma`) all reach a defined terminal state. (Do not run any other HW suite concurrently.)

- [ ] **Step 5: Commit**

```bash
git add tools/test_experiment_loop_hw.py
git commit -m "test(#140): HW grounding -- through-core gap, exact core-lock segment, segment-downgrade falsifiability"
```

---

### Task 9 [SCOPE FORK — Maya decides include/defer]: full EPS purge + suite re-verify

**This task is the optional tail.** Tasks 1-8 deliver the grounding rule and leave `coincident`/`same_source` (identity) and the dead `correlates`/`deterministic`/`verify_offset_stable` on the old `EPS=2.0`. Leaving `EPS` anywhere means a statistical tolerance still lives in the module, which conflicts with the "no statistics anywhere" line. This task removes it entirely. Defer ONLY if we want the smallest possible diff this round.

**Files:**
- Modify: `tools/inference/verifier.py` (convert `coincident` to exact; remove `EPS`, `correlates`, `deterministic`, `verify_offset_stable` if unused; keep `RejectedRule`)
- Modify: `tools/inference/rules.py` (the `coincident` call drops `eps`)
- Modify: `tools/inference/engine.py` (the `coincident` call in degeneracy drops `eps`)
- Test: `tools/test_inference_verifier.py` (drop/replace `correlates`/`deterministic`/`verify_offset_stable` tests; add exact `coincident` tests)

**Interfaces:**
- Produces (changed): `coincident(run_dirs, a, b, anchor_key=ANCHOR) -> bool` (exact: offset is exactly 0); `EPS` removed.

- [ ] **Step 1: Confirm `correlates`/`deterministic`/`verify_offset_stable` are dead**

Run: `cd tools && grep -rn "correlates\|deterministic(\| EPS\|verify_offset_stable\|, eps" inference/`
Expected after Tasks 1-8: the only remaining references are `correlates`/`deterministic`/`verify_offset_stable` definitions in `verifier.py` and their tests. If any non-test caller remains, migrate it before deleting.

- [ ] **Step 2: Convert `coincident` to exact + remove `EPS` and dead functions**

In `tools/inference/verifier.py`: replace `coincident` with an exact version and delete `EPS`, `correlates`, `deterministic`, `verify_offset_stable`:

```python
def coincident(run_dirs: List[str], a: str, b: str,
               anchor_key: str = ANCHOR) -> bool:
    """a and b fire at the SAME anchored ts in every co-traced run -- the exact
    offset between them is 0. Identity-coincidence, equality not statistics."""
    return offset_exact(run_dirs, a, b, anchor_key) == 0
```

- [ ] **Step 3: Drop the `eps` argument at the two `coincident` call sites**

In `rules.py` `try_same_source`: `coincident(run_dirs, a, b, anchor_key)` (already done in Task 5).
In `engine.py` degeneracy loop: change `coincident(run_dirs, a, b, anchor_key)` (it currently passes only positional args — confirm no `eps`).

- [ ] **Step 4: Update verifier tests**

In `tools/test_inference_verifier.py`, delete the `correlates`/`deterministic`/`verify_offset_stable` tests and their imports. Replace the `coincident` test with an exact pair (range 0) and a negative case:

```python
from inference.verifier import coincident


def test_coincident_true_when_offset_exactly_zero(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 40, "B": 40}, {"A": 90, "B": 90}])
    assert coincident(dirs, "1|0|0|A", "1|0|0|B") is True


def test_coincident_false_when_offset_nonzero_or_jittery(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 40, "B": 41}, {"A": 90, "B": 90}])
    assert coincident(dirs, "1|0|0|A", "1|0|0|B") is False
```

- [ ] **Step 5: Run the full offline inference suite**

Run: `env -u XDNA_EMU PYTHONPATH=tools python -m pytest test_inference_*.py test_experiment_report.py test_selfmodel.py -v`
Expected: PASS, zero references to `EPS` remaining.

- [ ] **Step 6: Re-verify the HW suite (real NPU, once)**

Run: `cd tools && XDNA_HW_SMOKE=1 env -u XDNA_EMU XDNA_EMU_RUNTIME=debug python -m pytest test_experiment_loop_hw.py -v`
Expected: unchanged from Task 8 — all defined terminal states, segment + gap as before.

- [ ] **Step 7: Commit**

```bash
git add tools/inference/verifier.py tools/inference/rules.py tools/inference/engine.py tools/test_inference_verifier.py
git commit -m "refactor(#140): purge EPS -- coincident exact, remove dead std primitives"
```

---

## Self-Review Notes

- **Spec coverage:** grounding.py `same_domain`/`ground_edge`/`assemble` (Task 3); verifier `offset_exact`/`anchor_rigid` + falsifier triad (Tasks 1-2); rules `try_derives` segment/gap with config/program orientation unchanged (Task 5); facts kind discriminator + accessor (Task 4); report segments/gaps/RejectedRules + determinism-partition rewiring to `anchor_rigid` (Tasks 5-7); Q=0 (Task 1); through-core HW = gap + exact segment + gap, stopgap removed, falsifiability (Task 8). The spec's "out of scope" (cross-domain timer-sync, active event selection, full facts-schema migration, new event types) is honored — none appear here.
- **Determinism-partition scope:** the spec mandates this rewiring "because removing std breaks them" — `try_derives`'s stochastic-parent gate uses the determinism notion, so `mark_determinism`/`classify_events`/`is_stochastic_root` must move to `anchor_rigid` (Task 5). Confirmed forced, not optional.
- **EPS tail (Task 9):** the only genuinely-optional scope. Identity-coincidence (`coincident`/`same_source`) is not grounding, so it can stay on `EPS` if we want a smaller round — but then a statistical tolerance survives in the module. Recommendation: include it.
- **Type consistency:** `Segment(parent, child, offset)` and `Gap(parent, child)` field order is fixed across grounding.py, rules.py, and the HW tests. `derives` args are `(child, parent, offset, kind)` everywhere; readers use `derive_kind`/`derive_offset`. `offset_exact(a, b) == a - b`; `Segment.offset == child - parent >= 0`.
- **Behavior change to flag at review:** under Q=0 the memtile PORT_RUNNING relays (genuine ±1..4 HW jitter) become **gaps**, not segments — the cycle-exact segment now comes from the core lock edge. Engine baseline output for add_one shifts accordingly; terminal state stays `placed` (gaps still count as derived children in `classify_events`).
