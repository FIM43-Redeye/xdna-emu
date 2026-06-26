# Integrated Timeline Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `assemble_timeline`, which turns N witness-clean HW trace captures of one kernel into an `IntegratedTimeline` — per-timer-domain tracks of cycle-by-cycle deterministic frames and windowed nondeterministic periods, woven by typed cross-domain edges, with every event and gap accounted for.

**Architecture:** A new `tools/inference/timeline.py` holds the data model and the segmentation algorithm. It consumes the existing measurement primitives (`trace_join.batch_firsts`/`anchored_firsts`, `verifier.offset_exact`/`anchor_rigid`, `grounding.ground_edge`/`same_domain`) and the existing causal graph (`derives` facts + `selfmodel.candidate_pairs_from_dump`). Determinism is detected by **jitter-vector equivalence** (events whose per-run anchored-ts vectors differ by a constant are rigidly linked, exact integer, Q=0). Cross-domain offsets are **never** composed into cycles — tracks relate only through typed edges. The result is wired into `engine.run_engine` and `run_experiment`.

**Tech Stack:** Python 3.13, stdlib only, run from `tools/` so `inference.*` and `config_extract.*` import bare. Tests use `pytest` with `tmp_path`. No Rust; `cargo test --lib` not required.

## Global Constraints

- **Q=0, never tuned.** A quantity is deterministic only if its cross-run range is exactly 0 (`verifier.Q == 0`). No statistical tolerance is ever introduced to make a test pass. Thresholds (`MIN_N_FLOATING`, `FALSE_CLUSTER_BOUND`, `CENSUS_CONTENT_FLOOR`) are provisional constants pending corpus calibration; tests must exercise gate *behavior* by construction (control N and corroboration directly), never by tuning a threshold to a real dataset.
- **No cross-domain cycle.** A raw cross-domain anchored offset is `Δwall + skew` (per `docs/trace/cross-domain-skew-limit.md`) and must never be emitted as a cycle. Cross-domain positioning is a typed edge only (`reproduction_offset` when range-0, else existence-only). Within-domain offsets (same `col|row|pkt_type` prefix, tested by `grounding.same_domain`) are cycles.
- **Anchor constant.** The global anchor is `"1|2|0|PERF_CNT_2"` (= `verifier.ANCHOR`); reuse it, do not redefine.
- **Single source for anchored values.** All per-run anchored-ts reads go through the existing `trace_join.batch_firsts`/`anchored_firsts` path (and `verifier._anchored_per_run` where a single-event vector is needed), so the timeline and the edge grounder agree on every measured number.
- **Provisional, re-verifiable verdicts.** Every determinism verdict is provisional: confidence scales with N and is reconfirmed by witness re-capture. Below the documented N-floor, a verdict is flagged `provisional_low_n`, never silently stamped rigid.
- **Pure addition, no regressions.** Existing functions keep their signatures and behavior. `check_additivity` stays as-is (test-only caller); the tri-state cross-check is a NEW sibling function. Run `cd tools && python -m pytest -q` green before and after each task.
- **Commit convention.** Commit messages carry no emoji and end with the repo trailer (`Generated using Claude Code.` + the `Claude-Session:` line). Steps below show the short subject only.

---

## File Structure

- `tools/inference/timeline.py` (NEW) — data model (`Pulse`, `Span`, `RigidRun`, `JitterPoint`, `EventRecord`, `DeterministicPeriod`, `NondeterministicPeriod`, `PresenceClass`, `CrossTrackEdge`, `Census`, `IntegratedTimeline`), the algorithm (`characterize_event`, eligibility gates, `rigid_clusters`, `internal_cycles`, `build_track`, `order_nondeterministic`, `weave`, `coupling_oracle`, `census_of`, `assemble_timeline`), and `render_timeline`. Flag constants.
- `tools/trace_join.py` (MODIFY) — add `batch_occurrences` (all firings per event in a batch, anchored, ordered).
- `tools/inference/verifier.py` (MODIFY) — add `anchored_occurrences_per_run`, `offset_window`, and `additivity_state` (tri-state sibling of `check_additivity`).
- `tools/inference/engine.py` (MODIFY) — call `assemble_timeline`, add `timeline` to the report.
- `tools/inference/run_experiment.py` (MODIFY) — thread `timeline` through the returned report.
- `tools/inference/grounding.py` (MODIFY) — remove unused `Timeline`/`assemble` (superseded).
- Tests: `tools/test_timeline.py` (NEW, the bulk), plus additions to `tools/test_trace_join.py` and `tools/test_inference_verifier.py`.

---

## Task 1: Occurrence capture — `batch_occurrences` + verifier accessor

**Files:**
- Modify: `tools/trace_join.py` (add `batch_occurrences` near `batch_firsts`)
- Modify: `tools/inference/verifier.py` (add `anchored_occurrences_per_run`)
- Test: `tools/test_trace_join.py`, `tools/test_inference_verifier.py`

**Interfaces:**
- Consumes: `trace_join.anchored_firsts(events, anchor_key) -> Dict[str,int]`, `trace_join._batch_names(run_dir) -> List[str]`, `trace_join._key`.
- Produces:
  - `trace_join.batch_occurrences(run_dir: str, batch_name: str, anchor_key: str = "1|2|0|PERF_CNT_2") -> Dict[str, List[int]]` — per event key, the sorted list of `(soc - anchor_soc)` for EVERY firing in that batch; `{}` if the anchor never fired in the batch.
  - `verifier.anchored_occurrences_per_run(run_dirs: List[str], event_key: str, pinned_batch: str, anchor_key: str = ANCHOR) -> List[List[int]]` — for each run dir, the occurrence list of `event_key` read from `pinned_batch` (empty list for a run where the batch/anchor/event is absent).

- [ ] **Step 1: Write the failing test for `batch_occurrences`**

In `tools/test_trace_join.py` add (NOTE: this file already defines module-level
`_ev` and `_write_batch` with different signatures — do NOT redefine them; use the
distinct names `_occ_ev` / `_occ_write_batch` below):

```python
import json
from pathlib import Path
import trace_join as tj

def _occ_write_batch(rd: Path, batch: str, events):
    p = rd / batch / "hw"
    p.mkdir(parents=True)
    (p / "trace.events.json").write_text(json.dumps({"events": events}))

def _occ_ev(col, row, pkt, name, soc, slot=0):
    return {"col": col, "row": row, "pkt_type": pkt, "name": name, "soc": soc, "slot": slot}

def test_batch_occurrences_returns_all_firings_anchored(tmp_path):
    rd = tmp_path / "run_00"
    _occ_write_batch(rd, "batch_00", [
        _occ_ev(1, 2, 0, "PERF_CNT_2", 1000),
        _occ_ev(1, 1, 3, "PORT_RUNNING_0", 1010),
        _occ_ev(1, 1, 3, "PORT_RUNNING_0", 1026),
        _occ_ev(1, 1, 3, "PORT_RUNNING_0", 1042),
    ])
    occ = tj.batch_occurrences(str(rd), "batch_00")
    assert occ["1|1|3|PORT_RUNNING_0"] == [10, 26, 42]   # anchored, sorted
    assert occ["1|2|0|PERF_CNT_2"] == [0]

def test_batch_occurrences_empty_without_anchor(tmp_path):
    rd = tmp_path / "run_00"
    _occ_write_batch(rd, "batch_00", [_occ_ev(1, 1, 3, "PORT_RUNNING_0", 1010)])
    assert tj.batch_occurrences(str(rd), "batch_00") == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd tools && python -m pytest test_trace_join.py -k batch_occurrences -q`
Expected: FAIL with `AttributeError: module 'trace_join' has no attribute 'batch_occurrences'`.

- [ ] **Step 3: Implement `batch_occurrences`**

In `tools/trace_join.py`, after `batch_firsts` (around line 67), add:

```python
def batch_occurrences(run_dir: str, batch_name: str,
                      anchor_key: str = "1|2|0|PERF_CNT_2") -> Dict[str, List[int]]:
    """All firings per "col|row|pkt_type|name" for one batch, anchored to
    anchor_key and sorted ascending. {} if the anchor never fired in this batch."""
    p = Path(run_dir) / batch_name / "hw" / "trace.events.json"
    if not p.exists():
        return {}
    events = json.loads(p.read_text()).get("events", [])
    anchor_soc = None
    for e in events:
        if _key(e["col"], e["row"], e["pkt_type"], e["name"]) == anchor_key:
            anchor_soc = e["soc"] if anchor_soc is None else min(anchor_soc, e["soc"])
    if anchor_soc is None:
        return {}
    out: Dict[str, List[int]] = collections.defaultdict(list)
    for e in events:
        out[_key(e["col"], e["row"], e["pkt_type"], e["name"])].append(e["soc"] - anchor_soc)
    return {k: sorted(v) for k, v in out.items()}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd tools && python -m pytest test_trace_join.py -k batch_occurrences -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the failing test for `anchored_occurrences_per_run`**

In `tools/test_inference_verifier.py` add (reuse its existing `tmp_path` event-writing helpers if present; otherwise inline like Task 1's `_write_batch`/`_ev`):

```python
from inference.verifier import anchored_occurrences_per_run

def test_anchored_occurrences_per_run_reads_pinned_batch(tmp_path):
    for i, base in enumerate((1000, 2000)):
        rd = tmp_path / f"run_{i:02d}"
        p = rd / "batch_00" / "hw"; p.mkdir(parents=True)
        (p / "trace.events.json").write_text(__import__("json").dumps({"events": [
            {"col":1,"row":2,"pkt_type":0,"name":"PERF_CNT_2","soc":base,"slot":0},
            {"col":1,"row":1,"pkt_type":3,"name":"PORT_RUNNING_0","soc":base+10,"slot":0},
            {"col":1,"row":1,"pkt_type":3,"name":"PORT_RUNNING_0","soc":base+26,"slot":0},
        ]}))
    runs = [str(tmp_path / "run_00"), str(tmp_path / "run_01")]
    got = anchored_occurrences_per_run(runs, "1|1|3|PORT_RUNNING_0", "batch_00")
    assert got == [[10, 26], [10, 26]]
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd tools && python -m pytest test_inference_verifier.py -k anchored_occurrences -q`
Expected: FAIL with ImportError on `anchored_occurrences_per_run`.

- [ ] **Step 7: Implement `anchored_occurrences_per_run`**

In `tools/inference/verifier.py`, after `_anchored_per_run` (around line 45), add:

```python
def anchored_occurrences_per_run(run_dirs: List[str], event_key: str,
                                 pinned_batch: str,
                                 anchor_key: str = ANCHOR) -> List[List[int]]:
    """Per run dir, the occurrence list of event_key from pinned_batch (anchored,
    sorted). Empty list for a run where the batch/anchor/event is absent."""
    return [tj.batch_occurrences(rd, pinned_batch, anchor_key).get(event_key, [])
            for rd in run_dirs]
```

- [ ] **Step 8: Run both new tests; full suite stays green**

Run: `cd tools && python -m pytest test_trace_join.py test_inference_verifier.py -q`
Expected: PASS (all, including pre-existing).

- [ ] **Step 9: Commit**

```bash
git add tools/trace_join.py tools/inference/verifier.py tools/test_trace_join.py tools/test_inference_verifier.py
git commit -m "feat(#140): batch_occurrences + per-run occurrence accessor"
```

---

## Task 2: `offset_window` + tri-state `additivity_state`

**Files:**
- Modify: `tools/inference/verifier.py`
- Test: `tools/test_inference_verifier.py`

**Interfaces:**
- Consumes: `verifier.pair_via tj.pair_derivability`, `verifier.offset_exact`, `verifier.ANCHOR`, `verifier.Q`.
- Produces:
  - `verifier.offset_window(run_dirs, a, b, anchor_key=ANCHOR) -> Optional[Tuple[int,int]]` — `(min, max)` of `(a - b)` across the first co-tracing batch per run; `None` if never co-traced. (When range 0 this equals `(off, off)`; `offset_exact` is the range-0 special case.)
  - `verifier.additivity_state(run_dirs, chain, anchor_key=ANCHOR) -> str` — one of `"pass"` / `"violation"` / `"vacuous"` / `"unverifiable"`. `"vacuous"` for a chain of < 3 keys; `"unverifiable"` when any consecutive or end-to-end offset is not co-traced (`None`); `"violation"` when end-to-end != sum of parts; else `"pass"`. (Distinct from `check_additivity`, which stays untouched.)
  - `verifier.cross_batch_range(run_dirs, event_key, anchor_key=ANCHOR) -> int` — the max over runs of `(max - min)` of `event_key`'s anchored first-occurrence across the batches that trace it within that run; `0` means the event is batch-invariant (the real-data check (i) precondition for cross-batch frame membership). Reads `trace_join.batch_firsts` over `_batch_names`.

- [ ] **Step 1: Write the failing tests**

In `tools/test_inference_verifier.py` add (follow the file's existing pattern for writing `run_NN/batch_00/hw/trace.events.json`; a local `_runs(tmp_path, rows)` like in `test_canary_witness.py` is fine):

```python
from inference.verifier import offset_window, additivity_state

def test_offset_window_min_max(tmp_path):
    # A-B = 10 in run0, 13 in run1 -> window (10, 13).
    dirs = _runs(tmp_path, [{"1|0|0|A": 0, "1|0|0|B": 10},
                            {"1|0|0|A": 0, "1|0|0|B": 13}])
    assert offset_window(dirs, "1|0|0|B", "1|0|0|A") == (10, 13)

def test_offset_window_none_when_not_co_traced(tmp_path):
    dirs = _runs(tmp_path, [{"1|0|0|A": 0}])  # B never present
    assert offset_window(dirs, "1|0|0|B", "1|0|0|A") is None

def test_additivity_state_pass_vacuous_unverifiable_violation(tmp_path):
    ok = _runs(tmp_path, [{"1|0|0|A": 0, "1|0|0|B": 5, "1|0|0|C": 12}])
    assert additivity_state(ok, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) == "pass"
    assert additivity_state(ok, ["1|0|0|A", "1|0|0|B"]) == "vacuous"
    miss = _runs(tmp_path, [{"1|0|0|A": 0, "1|0|0|C": 12}])  # B missing -> a gap
    assert additivity_state(miss, ["1|0|0|A", "1|0|0|B", "1|0|0|C"]) == "unverifiable"
```

(For a `violation` case, the simple single-batch `_runs` helper CANNOT work — with fixed per-run deltas `C−A == (C−B)+(B−A)` is an arithmetic identity, never a violation. Reuse the existing `test_check_additivity_rejects_when_offsets_do_not_sum` setup, which uses the multibatch helper (`_make_multibatch_run`), and assert `additivity_state(...) == "violation"`.)

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python -m pytest test_inference_verifier.py -k "offset_window or additivity_state" -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement both**

In `tools/inference/verifier.py` add after `offset_exact`:

```python
def offset_window(run_dirs: List[str], a: str, b: str,
                  anchor_key: str = ANCHOR) -> Optional[Tuple[int, int]]:
    """(min, max) of (a - b) over the first co-tracing batch per run; None if
    never co-traced. offset_exact is the range-0 special case (min == max)."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is None:
        return None
    return (int(st.min), int(st.max))
```

and after `check_additivity`:

```python
def additivity_state(run_dirs: List[str], chain: List[str],
                     anchor_key: str = ANCHOR) -> str:
    """Tri-state additivity cross-check (does NOT replace check_additivity).
    "vacuous" (<3 keys), "unverifiable" (a consecutive/end offset not co-traced),
    "violation" (end != sum), else "pass"."""
    if len(chain) < 3:
        return "vacuous"
    parts = []
    for i in range(len(chain) - 1):
        o = offset_exact(run_dirs, chain[i + 1], chain[i], anchor_key)
        if o is None:
            return "unverifiable"
        parts.append(o)
    end = offset_exact(run_dirs, chain[-1], chain[0], anchor_key)
    if end is None:
        return "unverifiable"
    return "pass" if end == sum(parts) else "violation"


def cross_batch_range(run_dirs: List[str], event_key: str,
                      anchor_key: str = ANCHOR) -> int:
    """Max over runs of (max-min) of event_key's anchored value across the batches
    that trace it in that run. 0 => batch-invariant (cross-batch membership safe)."""
    worst = 0
    for rd in run_dirs:
        vals = [tj.batch_firsts(rd, bn, anchor_key)[event_key]
                for bn in tj._batch_names(rd)
                if event_key in tj.batch_firsts(rd, bn, anchor_key)]
        if len(vals) > 1:
            worst = max(worst, max(vals) - min(vals))
    return worst
```

Add a test: write a run where `event_key` appears in two batches with equal anchored value → `cross_batch_range == 0`; and one where it differs by 2 → `cross_batch_range == 2`.

(`tv.Stats` is `namedtuple("Stats", "n mean std min max range")` — `.min`/`.max` exist, so `offset_window` reads them directly; no change to `trace_variance.py` needed.)

- [ ] **Step 4: Run to verify passing + full verifier suite green**

Run: `cd tools && python -m pytest test_inference_verifier.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/verifier.py tools/test_inference_verifier.py
git commit -m "feat(#140): offset_window + tri-state additivity_state"
```

---

## Task 3: Timeline data model

**Files:**
- Create: `tools/inference/timeline.py`
- Test: `tools/test_timeline.py`

**Interfaces:**
- Produces (consumed by all later tasks):

```python
# tools/inference/timeline.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from inference.verifier import ANCHOR, Q  # noqa: F401

# Provisional thresholds (corpus-calibrated later; OQ1). NEVER tuned to pass a test.
# MIN_N_FLOATING and (P_C, FALSE_CLUSTER_BOUND) are co-calibrated: the estimated
# false-cluster probability pairs*P_C**(N-1) must drop below FALSE_CLUSTER_BOUND at
# N == MIN_N_FLOATING. The values below are provisional placeholders pending the
# corpus jitter-entropy measurement; they are never adjusted to make a test pass.
MIN_N_FLOATING = 12         # uncorroborated floating cluster needs >= this many runs
P_C = 0.4                   # provisional per-component jitter collision rate (low-entropy)
FALSE_CLUSTER_BOUND = 1e-3  # estimated coincidence prob must be below this
CENSUS_CONTENT_FLOOR = 0.5  # >= this fraction of events in a deterministic frame

# Honesty flags
F_COUNT_WINDOW = "count_window"
F_COUNT_TRUNCATED = "count_truncated"
F_REORDERABLE = "occurrences_reorderable"
F_PROVISIONAL_LOW_N = "provisional_low_n"
F_UNGROUNDED_TAIL = "ungrounded_tail"
F_RESUMPTION_UNATTESTED = "resumption_unattested"
F_OVERLAPS_FRAME = "overlaps_frame"
F_ANCHOR_DROPOUT = "anchor_dropout"
F_BATCH_FLIP = "batch_flip"
F_CROSS_BATCH_FRAME = "cross_batch_frame"   # frame members span pinned batches w/o batch-invariance

@dataclass(frozen=True)
class Pulse:
    ts: int                       # anchored_ts of one firing

@dataclass(frozen=True)
class Span:
    begin: int                    # anchored begin ts
    length: int                   # end - begin

Occurrence = Union[Pulse, Span]

@dataclass
class RigidRun:
    start_index: int              # index of first occurrence in this run
    cycles: List[int]             # per-occurrence exact cycle (Pulse) or begin-cycle (Span)
    lengths: Optional[List[int]] = None  # per-occurrence exact span length, if Span

@dataclass
class JitterPoint:
    index: int
    window: Tuple[int, int]       # [min,max] of this occurrence across runs
    length_window: Optional[Tuple[int, int]] = None  # for spans

@dataclass
class EventRecord:
    key: str
    domain: str                   # "col|row|pkt_type"
    pinned_batch: Optional[str]
    n_eff: int                    # per-event effective N (post dropout/presence)
    rigid_runs: List[RigidRun] = field(default_factory=list)
    jitter_points: List[JitterPoint] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

@dataclass
class DeterministicPeriod:
    events: List[str]             # event keys, in local-cycle order
    cycles: Dict[str, int]        # event key -> exact local cycle (zero at grounding event)
    grounding_event: str          # the frame's local zero
    floating: bool                # True if not anchor-rigid
    offset_to_prior_frame: Optional[Tuple[int, int]] = None  # within-track; (x,x) exact or window

@dataclass
class NondeterministicPeriod:
    events: List[str]
    windows: Dict[str, Tuple[int, int]]      # event key -> [min,max] vs upstream frame ref
    reasons: Dict[str, str]                  # event key -> gap reason
    order_edges: List[Tuple[str, str, str]]  # (a, b, "causal"|"stable_position") meaning a<b
    grounding_event: Optional[str]           # None -> ungrounded_tail
    flags: List[str] = field(default_factory=list)

@dataclass
class Track:
    domain: str
    periods: List[Union[DeterministicPeriod, NondeterministicPeriod]]

@dataclass
class CrossTrackEdge:
    child: str
    parent: str
    reason: str                   # cross_domain | async_cdc
    reproduction_offset: Optional[int]  # None -> existence-only

@dataclass
class PresenceClass:
    appearance: Tuple[int, ...]   # sorted run indices the members fired in
    events: List[str]
    complement_of: Optional[Tuple[int, ...]] = None  # candidate mutual-exclusion

@dataclass
class Census:
    events: Dict[str, int]        # bucket -> count (anchored/floating/nondeterministic/intermittent/excluded)
    edges: Dict[str, int]         # bucket -> count (reproduction/existence_only)
    content_ok: bool              # meets CENSUS_CONTENT_FLOOR

@dataclass
class IntegratedTimeline:
    tracks: List[Track]
    cross_track_edges: List[CrossTrackEdge]
    intermittent: List[PresenceClass]
    flags: List[str]
    census: Census
    capture: Dict[str, object]    # {witness, n_runs, input_id}
```

- [ ] **Step 1: Write the failing test (construct + round-trip the model)**

Create `tools/test_timeline.py`:

```python
from inference import timeline as T

def test_data_model_constructs():
    er = T.EventRecord(key="1|2|0|A", domain="1|2|0", pinned_batch="batch_00", n_eff=8,
                       rigid_runs=[T.RigidRun(start_index=0, cycles=[0, 16])])
    dp = T.DeterministicPeriod(events=["1|2|0|A"], cycles={"1|2|0|A": 0},
                               grounding_event="1|2|0|A", floating=False)
    ndp = T.NondeterministicPeriod(events=["1|2|0|B"], windows={"1|2|0|B": (5, 9)},
                                   reasons={"1|2|0|B": "within_domain_nonexact"},
                                   order_edges=[], grounding_event=None,
                                   flags=[T.F_UNGROUNDED_TAIL])
    tl = T.IntegratedTimeline(tracks=[T.Track(domain="1|2|0", periods=[dp, ndp])],
                              cross_track_edges=[], intermittent=[], flags=[],
                              census=T.Census(events={}, edges={}, content_ok=True),
                              capture={"n_runs": 8})
    assert tl.tracks[0].periods[0].grounding_event == "1|2|0|A"
    assert T.MIN_N_FLOATING == 8 and T.CENSUS_CONTENT_FLOOR == 0.5
    assert er.flags == [] and ndp.flags == [T.F_UNGROUNDED_TAIL]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python -m pytest test_timeline.py -k data_model -q`
Expected: FAIL (no module `inference.timeline`).

- [ ] **Step 3: Create `tools/inference/timeline.py`** with the dataclasses + constants from the Interfaces block above (and a module docstring stating the no-cross-domain-cycle invariant and Q=0 provisionality).

- [ ] **Step 4: Run to verify passing**

Run: `cd tools && python -m pytest test_timeline.py -k data_model -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): timeline data model"
```

---

## Task 4: Occurrence characterization — rigid-run segmentation (pulse + span)

**Files:**
- Modify: `tools/inference/timeline.py` (add `characterize_event`, `LEVEL_EVENTS`, `_is_level`, `_pair_spans`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: `verifier.anchored_occurrences_per_run`, the Task 3 types.
- Produces: `timeline.characterize_event(run_dirs, key, pinned_batch, *, is_span=None, buffer_ceiling=None, anchor_key=ANCHOR) -> EventRecord`. When `is_span is None` the kind is auto-detected via `_is_level(key)` (a held-level event: `PORT_RUNNING_*` by prefix, or `LOCK_STALL`/`MEMORY_STALL`/`STREAM_STALL`). **Pulse** events: occurrence `k` is matched across runs and the list is partitioned into maximal `RigidRun`s (consecutive indices whose anchored value is range-0 across the runs where index `k` exists) and `JitterPoint`s. **Span** events: each run's firings are paired into `(begin, length)` spans (`_pair_spans`: consecutive firings alternate begin/end; a dangling begin is dropped); span `k` is rigid iff BOTH `begin` and `length` are range-0 across runs — a rigid run carries `cycles` (begin) AND `lengths`; a jitter span becomes a `JitterPoint` with `window` (begin) and `length_window`. Sets `F_COUNT_WINDOW` if per-run counts (of pulses, or of spans) differ, `F_REORDERABLE` if any run's raw occurrence list is not strictly increasing, `F_COUNT_TRUNCATED` if a per-run raw count equals `buffer_ceiling`. `n_eff` = number of runs where the event fired at least once.

- [ ] **Step 1: Write failing tests**

In `tools/test_timeline.py`:

```python
from inference import timeline as T

def test_characterize_event_rigid_then_jittery(tmp_path, monkeypatch):
    # 3 runs; occ index 0 rigid (cycle 10), index 1 jitters (26/26/28).
    occ = {"1|1|3|P": [[10, 26], [10, 26], [10, 28]]}
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1", "r2"], "1|1|3|P", "batch_00")
    assert er.n_eff == 3
    assert er.rigid_runs == [T.RigidRun(start_index=0, cycles=[10])]
    assert er.jitter_points == [T.JitterPoint(index=1, window=(26, 28))]
    assert T.F_COUNT_WINDOW not in er.flags  # counts equal (all length 2)

def test_characterize_event_jittery_first_then_steady(tmp_path, monkeypatch):
    # index 0 jitters (5/7), indices 1..2 rigid -> recovered, not discarded.
    occ = {"1|1|3|P": [[5, 20, 40], [7, 20, 40]]}
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1"], "1|1|3|P", "batch_00")
    assert er.jitter_points == [T.JitterPoint(index=0, window=(5, 7))]
    assert er.rigid_runs == [T.RigidRun(start_index=1, cycles=[20, 40])]

def test_characterize_event_count_window_and_reorderable(tmp_path, monkeypatch):
    occ = {"1|1|3|P": [[10, 26], [10], [26, 10]]}  # counts differ; run2 not increasing
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1", "r2"], "1|1|3|P", "batch_00")
    assert T.F_COUNT_WINDOW in er.flags
    assert T.F_REORDERABLE in er.flags

def test_characterize_event_span_begin_and_length(tmp_path, monkeypatch):
    # PORT_RUNNING_0 is a level event -> firings pair into (begin,length) spans.
    # run0 [10,26, 50,70] -> (10,16),(50,20); run1 [10,26, 50,72] -> (10,16),(50,22)
    occ = {"1|1|3|PORT_RUNNING_0": [[10, 26, 50, 70], [10, 26, 50, 72]]}
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1"], "1|1|3|PORT_RUNNING_0", "batch_00")
    # span 0 fully rigid (begin 10, length 16); span 1 begin-rigid but length jitters.
    assert er.rigid_runs == [T.RigidRun(start_index=0, cycles=[10], lengths=[16])]
    assert er.jitter_points == [T.JitterPoint(index=1, window=(50, 50), length_window=(20, 22))]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && python -m pytest test_timeline.py -k characterize -q`
Expected: FAIL (`characterize_event` undefined).

- [ ] **Step 3: Implement `characterize_event`**

Add to `timeline.py` (import the accessor at module top: `from inference.verifier import anchored_occurrences_per_run` and reference it as `anchored_occurrences_per_run` so the monkeypatch on `T.anchored_occurrences_per_run` works):

```python
LEVEL_EVENTS = {"LOCK_STALL", "MEMORY_STALL", "STREAM_STALL"}  # + PORT_RUNNING_* by prefix

def _domain_of(key: str) -> str:
    return key.rsplit("|", 1)[0]

def _is_level(key: str) -> bool:
    name = key.rsplit("|", 1)[1]
    return name.startswith("PORT_RUNNING") or name in LEVEL_EVENTS

def _pair_spans(occ):
    # consecutive firings alternate begin/end -> (begin, length); drop a dangling begin
    return [(occ[i], occ[i + 1] - occ[i]) for i in range(0, len(occ) - 1, 2)]

def characterize_event(run_dirs, key, pinned_batch, *, is_span=None,
                       buffer_ceiling=None, anchor_key=ANCHOR) -> EventRecord:
    if is_span is None:
        is_span = _is_level(key)
    per_run = anchored_occurrences_per_run(run_dirs, key, pinned_batch, anchor_key)
    present = [r for r in per_run if r]
    n_eff = len(present)
    flags = []
    if any(r != sorted(r) for r in present):                 # raw firings out of order
        flags.append(F_REORDERABLE)
    if buffer_ceiling is not None and any(len(r) == buffer_ceiling for r in present):
        flags.append(F_COUNT_TRUNCATED)
    er = EventRecord(key=key, domain=_domain_of(key), pinned_batch=pinned_batch,
                     n_eff=n_eff, flags=flags)
    runs = [_pair_spans(r) for r in present] if is_span else present
    if len({len(r) for r in runs}) > 1:
        flags.append(F_COUNT_WINDOW)
    max_k = max((len(r) for r in runs), default=0)
    run = None
    for k in range(max_k):
        items = [r[k] for r in runs if len(r) > k]
        if is_span:
            begins = [it[0] for it in items]; lens = [it[1] for it in items]
            blo, bhi = min(begins), max(begins); llo, lhi = min(lens), max(lens)
            rigid = (blo == bhi and llo == lhi)            # begin AND length range-0
        else:
            blo, bhi = min(items), max(items); llo = lhi = None
            rigid = (blo == bhi)
        if rigid:
            if run is None:
                run = RigidRun(start_index=k, cycles=[], lengths=([] if is_span else None))
            run.cycles.append(blo)
            if is_span:
                run.lengths.append(llo)
        else:
            if run is not None:
                er.rigid_runs.append(run); run = None
            jp = JitterPoint(index=k, window=(blo, bhi),
                             length_window=((llo, lhi) if is_span else None))
            er.jitter_points.append(jp)
    if run is not None:
        er.rigid_runs.append(run)
    return er
```

- [ ] **Step 4: Run to verify passing**

Run: `cd tools && python -m pytest test_timeline.py -k characterize -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): rigid-run occurrence characterization"
```

---

## Task 5: Eligibility gates + presence classes

**Files:**
- Modify: `tools/inference/timeline.py` (add `eligibility`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: `trace_join.batch_firsts`, `trace_join._batch_names`, Task 3 types.
- Produces: `timeline.eligibility(run_dirs, configured, anchor_key=ANCHOR) -> EligibilityResult` where
  `EligibilityResult = dataclass(clusterable: List[str], pinned: Dict[str,str], intermittent: List[PresenceClass], excluded: Dict[str,str], dropout_runs: List[int], dropout_batches: List[Tuple[int,str]])`.
  Order: (1) **anchor dropout (per batch)** — each `(run, batch)` with no anchor firing is recorded in `dropout_batches` (capture-health) and does NOT count as event absence; a run with every batch dropped is also in `dropout_runs`. Presence is computed only over anchor-present batches, so a single anchorless batch never fabricates absence; (2) **presence** — an event absent in some non-dropout run goes to an intermittent `PresenceClass` (grouped by identical appearance tuple; complementary appearance sets cross-linked via `complement_of`); (3) **batch-stability** — among present-in-all events, one whose lowest co-tracing batch index differs across runs is excluded with reason `F_BATCH_FLIP`; survivors are `clusterable` with their stable `pinned` batch.

- [ ] **Step 1: Write failing tests** (monkeypatch `T.batch_firsts`/`T._batch_names` with small dicts):

```python
def test_eligibility_partitions(tmp_path, monkeypatch):
    # run0,run1 fire A in batch_00; B fires only in run0 (intermittent);
    # C fires in both but in batch_00 (run0) vs batch_01 (run1) -> batch_flip.
    per = {
        "r0": {"batch_00": {"1|2|0|A": 0, "1|2|0|B": 5, "1|2|0|C": 9}},
        "r1": {"batch_00": {"1|2|0|A": 0}, "batch_01": {"1|2|0|C": 9}},
    }
    monkeypatch.setattr(T, "_batch_names", lambda rd: sorted(per[rd]))
    monkeypatch.setattr(T, "batch_firsts", lambda rd, bn, anchor_key=T.ANCHOR: per[rd].get(bn, {}))
    res = T.eligibility(["r0", "r1"], ["1|2|0|A", "1|2|0|B", "1|2|0|C"])
    assert res.clusterable == ["1|2|0|A"]
    assert res.pinned["1|2|0|A"] == "batch_00"
    assert any("1|2|0|B" in pc.events for pc in res.intermittent)
    assert res.excluded.get("1|2|0|C") == T.F_BATCH_FLIP
```

(Add a second test for anchor dropout: a run where `_batch_names` returns a batch but `batch_firsts` returns `{}` for all → the run index is in `dropout_runs` and does not turn a present event into intermittent.)

- [ ] **Step 2: Run to verify failure** — `cd tools && python -m pytest test_timeline.py -k eligibility -q` → FAIL.

- [ ] **Step 3: Implement `eligibility`** in `timeline.py`:

```python
# trace_join is a TOP-LEVEL module in tools/ (NOT inference.trace_join). Bind the
# names at module level so the tests' monkeypatch.setattr(T, "batch_firsts", ...)
# / setattr(T, "_batch_names", ...) intercept the bare calls below.
from trace_join import batch_firsts, _batch_names

@dataclass
class EligibilityResult:
    clusterable: List[str]
    pinned: Dict[str, str]
    intermittent: List[PresenceClass]
    excluded: Dict[str, str]
    dropout_runs: List[int]                      # runs where EVERY batch lost the anchor
    dropout_batches: List[Tuple[int, str]]       # per-batch anchor dropout (capture-health)

def _pinned_batch_index(run_dir, key, anchor_key):
    # Only anchor-present batches are considered (batch_firsts returns {} when the
    # anchor didn't fire) -- so a single anchorless batch never fabricates absence.
    for bn in _batch_names(run_dir):
        if key in batch_firsts(run_dir, bn, anchor_key):
            return bn
    return None

def eligibility(run_dirs, configured, anchor_key=ANCHOR) -> EligibilityResult:
    # Per-batch anchor dropout (spec gate 1): a batch with no anchor firing does not
    # count as event absence. Record each for capture-health; a run is fully-dropout
    # only if ALL its batches lost the anchor.
    dropout_batches = [(i, bn) for i, rd in enumerate(run_dirs)
                       for bn in _batch_names(rd)
                       if not batch_firsts(rd, bn, anchor_key)]
    dropout_runs = [i for i, rd in enumerate(run_dirs)
                    if all(not batch_firsts(rd, bn, anchor_key) for bn in _batch_names(rd))]
    live = [(i, rd) for i, rd in enumerate(run_dirs) if i not in dropout_runs]
    appear, pin = {}, {}
    for key in configured:
        runs_present = []
        batches = set()
        for i, rd in live:
            bn = _pinned_batch_index(rd, key, anchor_key)
            if bn is not None:
                runs_present.append(i); batches.add(bn)
        appear[key] = tuple(sorted(runs_present))
        pin[key] = batches
    all_live = tuple(sorted(i for i, _ in live))
    clusterable, pinned, excluded = [], {}, {}
    by_appearance: Dict[tuple, List[str]] = {}
    for key in configured:
        if not appear[key]:
            continue  # never fired
        if appear[key] != all_live:
            by_appearance.setdefault(appear[key], []).append(key)
            continue
        if len(pin[key]) != 1:
            excluded[key] = F_BATCH_FLIP
            continue
        clusterable.append(key)
        pinned[key] = next(iter(pin[key]))
    intermittent = [PresenceClass(appearance=a, events=evs)
                    for a, evs in sorted(by_appearance.items())]
    # cross-link complementary appearance sets (candidate mutual exclusion)
    aset = {pc.appearance: pc for pc in intermittent}
    for pc in intermittent:
        comp = tuple(i for i in all_live if i not in pc.appearance)
        if comp in aset:
            pc.complement_of = comp
    return EligibilityResult(sorted(clusterable), pinned, intermittent, excluded,
                             dropout_runs, dropout_batches)
```

- [ ] **Step 4: Run to verify passing + full suite green** — `cd tools && python -m pytest test_timeline.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): eligibility gates + presence classes"
```

---

## Task 6: Jitter-vector rigid clustering + corroboration gate

**Files:**
- Modify: `tools/inference/timeline.py` (add `rigid_clusters`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: `EventRecord` (for the per-event anchored vector at occurrence 0), the `derives` graph as a set of `(child, parent)` pairs, Task 3 constants.
- Produces: `timeline.rigid_clusters(jitter_vectors: Dict[str, Tuple[int,...]], n_eff: Dict[str,int], derives_pairs: set) -> ClusterResult` where `ClusterResult = dataclass(frames: List[ClusterFrame], nondeterministic: List[str])` and `ClusterFrame = dataclass(members: List[str], floating: bool, corroborated: bool, flags: List[str])`. Grouping: events with identical jitter-vector are one group. The all-zero group is the anchored frame (`floating=False`) — all-zero IS anchor-rigidity by definition, so no domain-key set is needed. A shared non-zero group is a floating frame; emitted only if **corroborated** (a direct `derives` edge between two members, OR a common parent P with `(a,P)` and `(b,P)` both in `derives_pairs`) OR the statistical gate passes (`min(n_eff over members) >= MIN_N_FLOATING` AND the estimated false-cluster bound `pairs*P_C**(N-1) < FALSE_CLUSTER_BOUND`); else its members go to `nondeterministic`. A single non-zero jitter-vector (unique) is a `nondeterministic` event. Members admitted by corroboration whose `min(n_eff) < MIN_N_FLOATING` carry `F_PROVISIONAL_LOW_N`.

- [ ] **Step 1: Write failing tests**

```python
def test_rigid_clusters_anchored_group(tmp_path):
    jv = {"1|2|0|A": (0, 0, 0), "1|2|0|B": (0, 0, 0), "1|2|0|C": (0, 3, 1)}
    n = {"1|2|0|A": 3, "1|2|0|B": 3, "1|2|0|C": 3}
    res = T.rigid_clusters(jv, n, set())
    anchored = [f for f in res.frames if not f.floating][0]
    assert set(anchored.members) == {"1|2|0|A", "1|2|0|B"}
    assert res.nondeterministic == ["1|2|0|C"]   # unique non-zero jv

def test_rigid_clusters_floating_needs_corroboration(tmp_path):
    jv = {"1|2|0|X": (0, 4, 1), "1|2|0|Y": (0, 4, 1)}   # shared non-zero
    n = {"1|2|0|X": 3, "1|2|0|Y": 3}                     # below MIN_N_FLOATING
    # No corroboration AND below the N-floor -> demoted to nondeterministic.
    res = T.rigid_clusters(jv, n, set())
    assert sorted(res.nondeterministic) == ["1|2|0|X", "1|2|0|Y"]
    assert not [f for f in res.frames if f.floating]
    # Common-parent corroboration -> emitted as a floating frame (low-N flagged).
    res2 = T.rigid_clusters(jv, n, {("1|2|0|X", "1|1|1|P"), ("1|2|0|Y", "1|1|1|P")})
    fr = [f for f in res2.frames if f.floating][0]
    assert set(fr.members) == {"1|2|0|X", "1|2|0|Y"} and fr.corroborated
    assert T.F_PROVISIONAL_LOW_N in fr.flags
```

- [ ] **Step 2: Run to verify failure** — `... -k rigid_clusters` → FAIL.

- [ ] **Step 3: Implement `rigid_clusters`**:

```python
import collections as _c

@dataclass
class ClusterFrame:
    members: List[str]
    floating: bool
    corroborated: bool
    flags: List[str] = field(default_factory=list)

@dataclass
class ClusterResult:
    frames: List[ClusterFrame]
    nondeterministic: List[str]

def _corroborated(members, derives_pairs) -> bool:
    ms = set(members)
    # direct chain edge between two members
    if any((a, b) in derives_pairs for a in ms for b in ms if a != b):
        return True
    # common parent: P with (m, P) for >=2 members
    parents = _c.Counter(p for (c, p) in derives_pairs if c in ms)
    return any(v >= 2 for v in parents.values())

def _false_cluster_ok(members, n_eff) -> bool:
    # BOTH gates (spec step 4): N-floor AND the estimated false-cluster bound.
    n = min(n_eff[m] for m in members)
    if n < MIN_N_FLOATING:
        return False
    pairs = len(members) * (len(members) - 1) // 2
    return pairs * (P_C ** (n - 1)) < FALSE_CLUSTER_BOUND

def rigid_clusters(jitter_vectors, n_eff, derives_pairs) -> ClusterResult:
    groups: Dict[tuple, List[str]] = {}
    for k, jv in jitter_vectors.items():
        groups.setdefault(jv, []).append(k)
    frames, nondet = [], []
    for jv, members in groups.items():
        members = sorted(members)
        if all(v == 0 for v in jv):
            frames.append(ClusterFrame(members, floating=False, corroborated=True))
            continue
        if len(members) < 2:
            nondet.extend(members); continue
        corr = _corroborated(members, derives_pairs)
        if corr or _false_cluster_ok(members, n_eff):
            flags = [] if corr and min(n_eff[m] for m in members) >= MIN_N_FLOATING \
                    else ([F_PROVISIONAL_LOW_N] if corr else [])
            frames.append(ClusterFrame(members, floating=True, corroborated=corr, flags=flags))
        else:
            nondet.extend(members)
    return ClusterResult(frames, sorted(nondet))
```

- [ ] **Step 4: Run to verify passing** — `... -k rigid_clusters` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): jitter-vector clustering + corroboration gate"
```

---

## Task 7: Internal cycles (anchor-composition) + additivity cross-check

**Files:**
- Modify: `tools/inference/timeline.py` (add `internal_cycles`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: per-event anchored occurrence-0 vector, `verifier.additivity_state`, Task 6 `ClusterFrame`.
- Produces: `timeline.internal_cycles(frame, anchored0: Dict[str,int], run_dirs=None, anchor_key=ANCHOR) -> Tuple[str, Dict[str,int]]` returning `(grounding_event, {member: local_cycle})`. Local zero = the member with the smallest anchored occurrence-0 value; each member's cycle = its `anchored0` minus the zero's (constant by cluster definition, same-domain ⇒ skew-free, no co-tracing needed). When `run_dirs` is given and a member pair is co-traced, call `additivity_state`; a `"violation"` raises `ClusterViolation` (caller demotes the frame); `"unverifiable"`/`"vacuous"`/`"pass"` do not.

- [ ] **Step 1: Write failing tests**

```python
def test_internal_cycles_from_anchored(tmp_path):
    frame = T.ClusterFrame(members=["1|2|0|A", "1|2|0|B"], floating=False, corroborated=True)
    g, cyc = T.internal_cycles(frame, {"1|2|0|A": 100, "1|2|0|B": 116})
    assert g == "1|2|0|A" and cyc == {"1|2|0|A": 0, "1|2|0|B": 16}

def test_internal_cycles_violation_raises(tmp_path, monkeypatch):
    frame = T.ClusterFrame(members=["1|0|0|A", "1|0|0|B", "1|0|0|C"], floating=False, corroborated=True)
    monkeypatch.setattr(T, "additivity_state", lambda runs, chain, anchor_key=T.ANCHOR: "violation")
    import pytest
    with pytest.raises(T.ClusterViolation):
        T.internal_cycles(frame, {"1|0|0|A": 0, "1|0|0|B": 5, "1|0|0|C": 99},
                          run_dirs=["r0"])
```

- [ ] **Step 2: Run to verify failure** — `... -k internal_cycles` → FAIL.

- [ ] **Step 3: Implement** (import `from inference.verifier import additivity_state` at top so the monkeypatch on `T.additivity_state` binds):

```python
class ClusterViolation(Exception):
    pass

def internal_cycles(frame, anchored0, run_dirs=None, anchor_key=ANCHOR):
    # Invariant guard: a frame is single-domain by construction; a cross-domain
    # "cycle" would be Delta_wall + skew (fatal-A). Fail loud on a wiring slip.
    assert len({_domain_of(m) for m in frame.members}) == 1, \
        f"cross-domain frame members: {frame.members}"
    members = sorted(frame.members, key=lambda m: anchored0[m])
    zero = members[0]
    cycles = {m: anchored0[m] - anchored0[zero] for m in members}
    if run_dirs is not None and len(members) >= 3:
        if additivity_state(run_dirs, members, anchor_key) == "violation":
            raise ClusterViolation(f"additivity violation in frame {members}")
    return zero, cycles
```

- [ ] **Step 4: Run to verify passing** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): internal cycles via anchor-composition + additivity check"
```

---

## Task 8: Per-track period builder

**Files:**
- Modify: `tools/inference/timeline.py` (add `build_track`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: a track's `ClusterFrame`s (with their `(grounding_event, cycles)` from Task 7), nondeterministic event keys with their anchored windows, the `derives` pairs (for `resumption_unattested`), Task 3 types.
- Produces: `timeline.build_track(domain, frames, nondet_windows, mean_pos, derives_pairs) -> Track` (5 args; cycles ride inside `frames`, there is NO separate `frame_cycles` param). `frames`: list of `(grounding_event, {member:cycle}, floating, anchor_pos)` where `anchor_pos` is `int` for an anchored frame (its grounding event's exact anchored value) or `(min,max)` for a floating frame. `mean_pos: Dict[str,float]` sequencing key (mean anchored occurrence-0; sequencing only, never reported). Algorithm: order frames and nondet events by `mean_pos`; emit a `DeterministicPeriod` per frame (cluster-identity grouping — a frame is never fragmented); a maximal run of nondet events between two frames becomes a `NondeterministicPeriod` whose `windows` are vs the upstream frame's grounding event, `grounding_event` = next frame's grounding event (None ⇒ `F_UNGROUNDED_TAIL`), `F_RESUMPTION_UNATTESTED` when the closing frame has no `derives` edge to any period member. Each non-first frame's `offset_to_prior_frame` is computed by interval subtraction of the two frames' `anchor_pos` (within-domain, skew-free): `(this_lo - prior_hi, this_hi - prior_lo)` — degenerate `(x,x)` exact when both frames anchored, a window when either floats.

- [ ] **Step 1: Write failing tests** — (a) grounded case: two anchored frames (A with grounding cycle data + `anchor_pos=100`, then frame C `anchor_pos=200`) with a nondet event B between them; assert period order `[Det(A), Nondet(B), Det(C)]`, `B`'s window present, `C`'s grounding event closes the nondet period, and `Det(C).offset_to_prior_frame == (100, 100)` (exact, both anchored). (b) ungrounded-tail case: a trailing nondet event with no following frame yields a `NondeterministicPeriod` with `grounding_event is None` and `F_UNGROUNDED_TAIL`. (c) floating case: a frame with `anchor_pos=(180,210)` after an anchored frame `anchor_pos=100` yields `offset_to_prior_frame == (80, 110)` (a window).

- [ ] **Step 2: Run to verify failure** — `... -k build_track` → FAIL.

- [ ] **Step 3: Implement `build_track`** producing the `Track` per the algorithm above:

```python
def _as_window(x):
    return x if isinstance(x, tuple) else (x, x)

def build_track(domain, frames, nondet_windows, mean_pos, derives_pairs) -> Track:
    # frames: List[(grounding_event, {member: cycle}, floating, anchor_pos)]
    #   anchor_pos = int (anchored frame, exact) | (min,max) (floating frame)
    items = []  # (mean_pos_key, "frame"|"nondet", payload)
    for (g, cyc, floating, apos) in frames:
        items.append((min(mean_pos[m] for m in cyc), "frame", (g, cyc, floating, apos)))
    for k in nondet_windows:
        items.append((mean_pos[k], "nondet", k))
    items.sort(key=lambda t: t[0])
    periods, pending, prior_apos = [], [], None
    def flush(closing_g):
        nonlocal pending
        if not pending:
            return
        evs = list(pending)
        attested = closing_g is not None and any(
            (k, closing_g) in derives_pairs or (closing_g, k) in derives_pairs for k in evs)
        flags = []
        if closing_g is None:
            flags.append(F_UNGROUNDED_TAIL)
        elif not attested:
            flags.append(F_RESUMPTION_UNATTESTED)
        periods.append(NondeterministicPeriod(
            events=evs, windows={k: nondet_windows[k] for k in evs},
            reasons={k: "within_domain_nonexact" for k in evs},
            order_edges=[], grounding_event=closing_g, flags=flags))
        pending = []
    for (_, kind, payload) in items:
        if kind == "nondet":
            pending.append(payload); continue
        g, cyc, floating, apos = payload
        flush(g)
        otp = None
        if prior_apos is not None:
            lo1, hi1 = _as_window(apos); lo0, hi0 = _as_window(prior_apos)
            otp = (lo1 - hi0, hi1 - lo0)   # interval subtraction; (x,x) exact iff both anchored
        periods.append(DeterministicPeriod(
            events=sorted(cyc, key=cyc.get), cycles=cyc, grounding_event=g,
            floating=floating, offset_to_prior_frame=otp))
        prior_apos = apos
    flush(None)
    return Track(domain=domain, periods=periods)
```

- [ ] **Step 4: Run to verify passing** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): per-track period builder"
```

---

## Task 9: Intra-period ordering

**Files:**
- Modify: `tools/inference/timeline.py` (add `order_nondeterministic`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: a `NondeterministicPeriod`'s event keys, the `derives` pairs (causal), and per-pair stable-position info, Task 3 types.
- Produces: `timeline.order_nondeterministic(events, derives_pairs, stable_before) -> List[Tuple[str,str,str]]` — `order_edges` `(a, b, tag)` meaning a-before-b. `tag="causal"` when `(b,a) in derives_pairs` (parent a -> child b) or a common chain; `tag="stable_position"` when `stable_before[(a,b)]` is True (a < b in every run) and no causal edge exists. Pairs with neither are omitted (concurrent). `stable_before: Dict[Tuple[str,str], bool]`.

- [ ] **Step 1: Write failing test** — 3 events; one causal edge, one stable-position pair, one concurrent pair; assert exactly the two edges with correct tags and the concurrent pair absent.

- [ ] **Step 2: Run to verify failure** — `... -k order_nondeterministic` → FAIL.

- [ ] **Step 3: Implement**:

```python
def order_nondeterministic(events, derives_pairs, stable_before):
    edges = []
    for a in events:
        for b in events:
            if a == b:
                continue
            if (b, a) in derives_pairs:           # parent a -> child b
                edges.append((a, b, "causal"))
            elif stable_before.get((a, b)):
                edges.append((a, b, "stable_position"))
    return edges
```

- [ ] **Step 4: Run to verify passing** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): intra-period honest partial order"
```

---

## Task 10: Cross-track weave + connectivity oracle

**Files:**
- Modify: `tools/inference/timeline.py` (add `coupling_oracle`, `weave`, `connectivity_defects`)
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: `grounding.ground_edge` (returns `Segment`|`Gap`), `grounding.same_domain`, `dump_model.ConfigDump` (`dump.route_graph.edges` of `RouteEdge(src,dst,kind)`, `src/dst` are `PortRef(col,row,port,dir,kind)`), cross-domain candidate pairs `[(child,parent)]`, Task 3 types.
- Produces:
  - `timeline.coupling_oracle(dump, start_col) -> set[Tuple[str,str]]` — the set of coupled **tile pairs**, sorted `(tileA, tileB)` where `tileA = "col|row"`, derived from `dump.route_graph.edges` (ALL kinds incl. `dma_buffer_relay`/`lock_pair`/`core_lock_relay`) with absolute cols (`col + start_col`). NOT via `generate_ledger`/`candidate_pairs_from_dump` (that would re-circularize the check). DESIGN NOTE: the spec calls for domain-pair (`col|row|pkt_type`) adjacency, but route-graph edges key on physical ports, not `pkt_type`, so domain-pair derivation is underspecified; **tile granularity is a conscious, documented relaxation** — it still catches the headline F1 failure (a fully-dropped tile-to-tile handoff) and never false-alarms. Domain-pair tightening (once port→pkt_type is established) is noted follow-up.
  - `timeline.weave(run_dirs, cross_domain_pairs, anchor_key=ANCHOR) -> List[CrossTrackEdge]` — for each `(child,parent)` with `not same_domain(child,parent)`, call `ground_edge(run_dirs, child, parent, anchor_key)` and map a `Gap` to a `CrossTrackEdge(child, parent, reason, reproduction_offset)`.
  - `timeline.connectivity_defects(oracle, edges) -> List[Tuple[str,str]]` — coupled tile pairs from `oracle` with NO `CrossTrackEdge` connecting their tiles. Non-empty = defect.

- [ ] **Step 1: Write failing tests**

```python
from config_extract.dump_model import ConfigDump, RouteGraph, RouteEdge, PortRef

def _pr(col, row): return PortRef(col=col, row=row, port=0, dir="out", kind="x")

def test_coupling_oracle_from_route_graph(tmp_path):
    rg = RouteGraph(edges=(
        RouteEdge(_pr(0, 0), _pr(0, 1), "dma_buffer_relay"),
        RouteEdge(_pr(0, 1), _pr(0, 2), "circuit"),
    ))
    dump = ConfigDump(device="npu1", route_graph=rg, tiles=())
    cpl = T.coupling_oracle(dump, start_col=1)   # abs col = 0+1
    assert ("1|0", "1|1") in cpl and ("1|1", "1|2") in cpl

def test_weave_and_connectivity(tmp_path, monkeypatch):
    from inference.grounding import Gap, GAP_CROSS_DOMAIN
    monkeypatch.setattr(T, "ground_edge",
        lambda runs, c, p, anchor=T.ANCHOR: Gap(parent=p, child=c, reason=GAP_CROSS_DOMAIN, reproduction_offset=7))
    edges = T.weave(["r0"], [("1|1|3|X", "1|2|0|Y")])
    assert edges[0].reproduction_offset == 7 and edges[0].reason == GAP_CROSS_DOMAIN
    # oracle says 1|2 and 1|1 coupled and the weave connects them -> no defect
    assert T.connectivity_defects({("1|1", "1|2")}, edges) == []
    # oracle says 1|0 and 1|2 coupled but nothing connects them -> defect
    assert T.connectivity_defects({("1|0", "1|2")}, edges) == [("1|0", "1|2")]
```

- [ ] **Step 2: Run to verify failure** — `... -k "coupling_oracle or weave"` → FAIL.

- [ ] **Step 3: Implement** (import `from inference.grounding import ground_edge, same_domain` at top; for `weave`, skip same-domain pairs and any `Segment` return — within-domain pairs are not cross-track):

```python
def _tile_of(key_or_domain: str) -> str:
    parts = key_or_domain.split("|")
    return f"{parts[0]}|{parts[1]}"

def coupling_oracle(dump, start_col) -> set:
    out = set()
    for e in dump.route_graph.edges:
        a = f"{e.src.col + start_col}|{e.src.row}"
        b = f"{e.dst.col + start_col}|{e.dst.row}"
        if a != b:
            out.add(tuple(sorted((a, b))))
    return out

def weave(run_dirs, cross_domain_pairs, anchor_key=ANCHOR) -> List[CrossTrackEdge]:
    from inference.grounding import Gap
    edges = []
    for (child, parent) in cross_domain_pairs:
        if same_domain(child, parent):
            continue
        g = ground_edge(run_dirs, child, parent, anchor_key)
        if isinstance(g, Gap):
            edges.append(CrossTrackEdge(child=child, parent=parent, reason=g.reason,
                                        reproduction_offset=g.reproduction_offset))
    return edges

def connectivity_defects(oracle, edges) -> List[Tuple[str, str]]:
    connected = {tuple(sorted((_tile_of(e.child), _tile_of(e.parent)))) for e in edges}
    return sorted(p for p in oracle if p not in connected)
```

- [ ] **Step 4: Run to verify passing** — PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): cross-track weave + independent connectivity oracle"
```

---

## Task 11: `census_of` + `assemble_timeline` + `render_timeline`

**Files:**
- Modify: `tools/inference/timeline.py`
- Test: `tools/test_timeline.py`

**Interfaces:**
- Consumes: every prior function. Produces:
  - `timeline.census_of(tracks, intermittent, excluded, edges) -> Census`.
  - `timeline.assemble_timeline(run_dirs, configured, derives_pairs, cross_domain_pairs, dump=None, start_col=1, anchor_key=ANCHOR, capture=None) -> IntegratedTimeline` — orchestrates, in order:
    1. `eligibility(run_dirs, configured, anchor_key)` → clusterable + pinned + intermittent + excluded + dropout.
    2. **count_truncated ceiling (G4):** derive `buffer_ceiling` from `dump` (trace buffer bytes / bytes-per-event) when available; pass it to `characterize_event`. When `dump`/capacity is unavailable, leave `buffer_ceiling=None` AND record a timeline flag `"count_ceiling_unknown"` (a declared best-effort limit, NOT a silently-absent flag).
    3. `characterize_event` per clusterable event (auto pulse/span); jitter-vector = its occurrence-0 anchored value across runs minus the run-0 value.
    4. **cross-batch frame gate (G1):** for each clusterable event compute `verifier.cross_batch_range`; an event with `cross_batch_range > 0` whose frame would span >1 pinned batch is held OUT of multi-member clustering (treated as its own nondeterministic singleton) and the timeline records `F_CROSS_BATCH_FRAME`. (In practice check (i) shows range 0, so this rarely triggers — but it is gated, not assumed.)
    5. per-track `rigid_clusters` → `internal_cycles` (demote the frame to nondeterministic on `ClusterViolation`).
    6. `build_track` per track (frames carry `anchor_pos` = the frame grounding event's anchored value: exact int for an anchored frame, `offset_window` of the grounding event for a floating frame).
    7. **intra-period order (G7):** per `NondeterministicPeriod`, build `stable_before[(a,b)] = (offset_window(run_dirs,b,a) is not None and its min > 0)` (a strictly before b in every run), then call `order_nondeterministic(events, derives_pairs, stable_before)` and set `.order_edges`.
    8. **overlaps_frame (G2):** a nondeterministic event whose anchored window overlaps the anchored extent of an adjacent frame is flagged `F_OVERLAPS_FRAME` on its period; its earned stable-position edges vs individual frame events are retained (it is NOT blanket-marked concurrent).
    9. `weave(run_dirs, cross_domain_pairs, anchor_key)` + `connectivity_defects(coupling_oracle(dump, start_col), edges)` (recorded in `flags` as `"connectivity_defect:<a>~<b>"`, only when `dump` is provided).
    10. `census_of(...)`.
  - `timeline.render_timeline(tl) -> str` — plain-text per-track A→B→C view with local cycles, windows, concurrency, cross-track edges, flags, and the census line.

- [ ] **Step 1: Write a failing end-to-end test on synthetic run dirs** — one track (core domain) with two anchored events forming a frame plus one nondet event, no cross-domain pairs; assert `assemble_timeline` returns one `Track` with the expected periods and a `Census` whose `content_ok` is True. Plus a `render_timeline` smoke test asserting the output contains the domain id and "DET"/"NONDET" markers.

- [ ] **Step 2: Run to verify failure** — `... -k "assemble or census or render"` → FAIL.

- [ ] **Step 3: Implement** `census_of`, `assemble_timeline` (wiring per the Interfaces description; group `clusterable` by domain, build jitter-vectors from each event's occurrence-0 anchored value across runs, route `nondeterministic` events to per-track windows via `verifier.offset_window` vs the track's anchored frame reference), and `render_timeline`. Record any `connectivity_defects` as `flags`. Keep functions small and delegate to Tasks 4-10.

- [ ] **Step 4: Run to verify passing + full suite** — `cd tools && python -m pytest test_timeline.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py
git commit -m "feat(#140): assemble_timeline + census + renderer"
```

---

## Task 12: Wire into engine + run_experiment; retire dead grounding code

**Files:**
- Modify: `tools/inference/engine.py` (around lines 55-80, the report dict)
- Modify: `tools/inference/run_experiment.py` (around lines 57-97, the engine block + return dict)
- Modify: `tools/inference/grounding.py` (remove `Timeline` + `assemble`, lines ~110-122)
- Test: `tools/test_inference_engine.py` (or the existing engine test file), `tools/test_timeline.py`

**Interfaces:**
- Consumes: `timeline.assemble_timeline`. The engine already computes `candidate_pairs` and `derives` facts; pass `derives_pairs = {(f.args[0], f.args[1]) for f in kb.by_predicate("derives")}` and `cross_domain_pairs = [(c,p) for (c,p) in candidate_pairs if not same_domain(c,p)]`. NOTE: `engine.py` currently imports only `from inference.grounding import gap_accounted` — extend it to `from inference.grounding import gap_accounted, same_domain` (calling a bare `grounding.same_domain` without `import grounding` is a `NameError`).
- Produces: `run_engine(...)` report dict gains `"timeline"` (an `IntegratedTimeline`, or its `render_timeline`/`asdict` form for JSON); `run_experiment(...)` return dict gains `"timeline"`.

- [ ] **Step 1: Write a failing test** asserting `run_engine(...)`'s report contains a `"timeline"` key that is an `IntegratedTimeline` (use a tiny synthetic run-dir set + an empty ledger + the candidate pairs for those events). For `run_experiment`, add a monkeypatched test asserting `report["timeline"]` is present (mirror the existing best-effort engine block test).

- [ ] **Step 2: Run to verify failure** — FAIL (no `timeline` key).

- [ ] **Step 3: Implement** — extend the engine import to `from inference.grounding import gap_accounted, same_domain`. In `engine.run_engine`, after the gaps/warnings block, build `derives_pairs`/`cross_domain_pairs` (using `same_domain`) and call `assemble_timeline(run_dirs, fired_keys, derives_pairs, cross_domain_pairs, anchor_key=anchor_key)`; add `"timeline"` to the returned dict. In `run_experiment.run_experiment`, inside the existing best-effort engine `try` block, read `rep.get("timeline")` and add it to the returned report. Delete `Timeline` and `assemble` from `grounding.py`; grep first to confirm no live importers (`grep -rn "from inference.grounding import.*assemble\|grounding.assemble\|grounding.Timeline" tools/`); fix any (there should be none).

- [ ] **Step 4: Run to verify passing + FULL suite** — `cd tools && python -m pytest -q` → PASS (everything).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/engine.py tools/inference/run_experiment.py tools/inference/grounding.py tools/test_inference_engine.py tools/test_timeline.py
git commit -m "feat(#140): wire timeline into engine + run_experiment; retire dead grounding code"
```

---

## Task 13: two_col fixture + multi-track E2E + real-data A/B checks

**Files:**
- Create: `tools/config_extract/fixtures/two_col.config.json` (generated; see Step 1)
- Test: `tools/test_timeline.py` (multi-track synthetic E2E + real-data A/B)

**Interfaces:**
- Consumes: `timeline.assemble_timeline`, `timeline.coupling_oracle`, the persisted real captures under `build/experiments/lock-jitter-{clean,loaded}` (from the canary work), `dump_model.load_dump`.

- [ ] **Step 1: Generate the `two_col` config fixture (controller/HW step, documented here).** Build a two-column kernel and extract its config JSON with the Rust extractor:
  `cargo run --release --example dump_config_json -- <two_col.xclbin> > tools/config_extract/fixtures/two_col.config.json`
  (mirror how `add_one_using_dma.config.json` was produced; a `two_col` build exists under `mlir-aie/.../npu-xrt/`). If no two_col build is available in this environment, SKIP the on-disk fixture and rely on the synthetic multi-track test below; note the skip in the task report. This step does not block the offline suite.

- [ ] **Step 2: Write the multi-track synthetic E2E test** — construct synthetic run dirs (via the Task-1 batch-writing helper) spanning TWO domains (e.g. `1|2|0` core and `1|1|3` memtile), with a cross-domain candidate pair between them and a route-graph dump (constructed `ConfigDump`) whose oracle says the two tiles are coupled. Assert: `assemble_timeline` returns two `Track`s; a `CrossTrackEdge` connects them; `connectivity_defects` is empty; NO `DeterministicPeriod` contains members from two domains (no cross-domain frame); the census has both tracks represented.

- [ ] **Step 3: Run to verify it fails, then passes** as you implement any missing wiring (most should already exist from Tasks 5-11). Run: `cd tools && python -m pytest test_timeline.py -k multi_track -q`.

- [ ] **Step 4: Write the real-data A/B checks (guarded `skipif` on fixture presence, like `test_canary_witness.py`'s real-fixture tests):**

```python
import glob, os
_EXP = "/home/triple/npu-work/xdna-emu/build/experiments"

@pytest.mark.skipif(not os.path.isdir(f"{_EXP}/lock-jitter-clean"),
                    reason="persisted HW fixtures absent")
def test_real_clean_clusters_stable():
    runs = sorted(glob.glob(f"{_EXP}/lock-jitter-clean/capture_00/run_*"))
    # core-lock ACQUIRE/RELEASE must land in one rigid frame (range-0 offset 24).
    # Build configured set from the run dirs, assemble, assert the two lock events
    # share a DeterministicPeriod with exact cycle delta 24.
    ...

@pytest.mark.skipif(not os.path.isdir(f"{_EXP}/lock-jitter-loaded"),
                    reason="persisted HW fixtures absent")
def test_real_loaded_clusters_fragment():
    runs = sorted(glob.glob(f"{_EXP}/lock-jitter-loaded/capture_00/run_*"))
    # Under load the lock pair's offset flickers -> NOT one rigid frame
    # (the canary working: the timeline reports them nondeterministic, not a frame).
    ...

@pytest.mark.skipif(not os.path.isdir(f"{_EXP}/lock-jitter-clean"),
                    reason="persisted HW fixtures absent")
def test_real_event_batch_invariant_check_i():
    # Spec real-data check (i): an event co-traced in >=2 batches has range-0
    # batch-to-batch anchored_ts, else cross-batch frame membership is unsound.
    from inference.verifier import cross_batch_range
    import trace_join as tj
    runs = sorted(glob.glob(f"{_EXP}/lock-jitter-clean/capture_00/run_*"))
    # Find an event traced in >=2 batches of run 0, then assert it is batch-invariant.
    rd0 = runs[0]
    counts = {}
    for bn in tj._batch_names(rd0):
        for k in tj.batch_firsts(rd0, bn):
            counts[k] = counts.get(k, 0) + 1
    multi = [k for k, c in counts.items() if c >= 2]
    assert multi, "expected at least one multi-batch event in the clean fixture"
    for k in multi:
        assert cross_batch_range(runs, k) == 0, f"{k} not batch-invariant"
```

Fill the `...` to assemble over the real run dirs and assert clean→one frame (offset 24) / loaded→fragmented, mirroring the witness's clean/dirty validation.

- [ ] **Step 5: Run the full suite** — `cd tools && python -m pytest -q`. Expected: PASS (real-data tests run if fixtures present, else skipped).

- [ ] **Step 6: Commit**

```bash
git add tools/test_timeline.py tools/config_extract/fixtures/two_col.config.json
git commit -m "feat(#140): two_col fixture + multi-track E2E + real-data A/B checks"
```

---

## HW Acceptance (controller-run, after offline tasks are green)

Not a coding task. After Tasks 1-13 land and the offline suite is green, the controller runs (on a witness-certified quiet box, NPU turbo, no second HW suite): capture `two_col` via `run_experiment` for N runs and assert a well-formed multi-track `IntegratedTimeline` — no cross-domain cycles, no swallowed events, connectivity holds against the oracle, census meets `CENSUS_CONTENT_FLOOR`. Bracket with the canary witness (`tools/canary_witness.py`). This is the HW-gated DoD; the offline suite is the regression gate.

---

## Self-Review (updated after plan-vs-spec review)

- **Spec coverage:** per-track tracks (T8), jitter-vector clustering (T6), no-cross-domain-cycle via typed edges + same-domain assertion (T7,T10), occurrence sequences/rigid runs incl. **span/held-level (begin,length)** (T1,T4 — span path implemented, pairing heuristic validated on real level events in T13), eligibility gates + per-batch dropout + presence classes (T5), internal cycles by anchor-composition + tri-state additivity (T2,T7), corroboration gate + N-floor + **live false-cluster bound** + provisional_low_n (T3,T6), **cross-batch frame gate + real-data check (i)** (T2 `cross_batch_range`, T11 gate, T13 test), independent connectivity oracle (T10), **stable_before partial order** (T11→T9), **overlaps_frame emission** (T11), **count_truncated ceiling derivation** (T11), census + content floor (T11), honesty flags (ungrounded_tail/resumption_unattested/overlaps_frame/count_*/batch_flip/anchor_dropout/provisional_low_n/cross_batch_frame), wiring + dead-code removal (T12), two_col + real-data A/B (i)+(ii) (T13).
- **Review fixes folded in:** F1 (renamed `_occ_*` test helpers), F2 (`from trace_join import`), F3 (5-arg `build_track`), F4/M1 (`offset_to_prior_frame` computed via interval subtraction of `anchor_pos`), F5 (engine imports `same_domain`), F6 (violation test uses multibatch helper), F7/G5 (live bound, dropped dead `anchor_domain_keys`), M2 (same-domain assert), M3 (per-batch dropout), G1/G2/G4/G6/G7 as above. Remaining conscious relaxation: tile-granular connectivity oracle (documented; domain-pair tightening is follow-up).
- **Placeholder scan:** no "TBD"/"handle errors" — each step has concrete code or concrete commands. The two `...` blocks in T13 Step 4 are explicitly marked "fill the `...`" with the exact assertion described; acceptable as the only intentionally-sketched test bodies (they depend on real fixture key names discovered at implementation time).
- **Type consistency:** `EventRecord`, `ClusterFrame`/`ClusterResult`, `DeterministicPeriod`/`NondeterministicPeriod`, `CrossTrackEdge`, `Census`, `IntegratedTimeline` names and fields are used identically across T3-T13; flag constants (`F_*`) defined once in T3 and referenced thereafter; `assemble_timeline`/`weave`/`coupling_oracle`/`internal_cycles`/`rigid_clusters` signatures match between producer and consumer tasks.
