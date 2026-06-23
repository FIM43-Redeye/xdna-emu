# Cross-Domain Reproduction-Target Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the exact raw cross-domain trace offset as a reproduction-target annotation on each cross-domain `Gap`, and stop shim NoC-egress (async-CDC) DMA-completion events from ever grounding as a (spurious) Segment.

**Architecture:** Pure-Python change to the trace inference engine (`tools/inference/`). Cross-domain edges already fall to `Gap` today (correct, unchanged). We (1) add an optional `reproduction_offset` to `Gap`, set to the exact raw offset when it agrees across runs; (2) add a semantics-derived `is_async_cdc` guard so shim NoC-egress `DMA_*_FINISHED_TASK` events are gap-only (fixing an existing same-domain false-ground bug); (3) thread `reproduction_offset` through the `derives` fact and the engine report projection. No new cycle "kind", no skew model, no statistical tolerance.

**Tech Stack:** Python 3.13, pytest. Run offline tests from `tools/` (its `conftest.py` and the `inference/` package live there): `cd /home/triple/npu-work/xdna-emu/tools`.

**Spec:** `docs/superpowers/specs/2026-06-23-cross-domain-grounding-shared-timebase-design.md`
**Boundary doc (the why):** `docs/trace/cross-domain-skew-limit.md`

## Global Constraints

- **No statistical inference.** Exact agreement only: `range <= Q`, `Q = 0` (a measured floor in `verifier.py`, never a tuned tolerance). No median/MAD/epsilon.
- **Reuse the existing `kind` strings** (`"segment"` | `"gap"`). Do NOT add a new kind. `reproduction_offset` is a separate optional field, NOT a smuggled structured-facts migration.
- **`reproduction_offset` is never a causal offset.** It must NOT be placed in the `derives` args[2] (`derive_offset`) slot. It is a distinct field (args[4]); a gap's args[2] stays `None`.
- **DERIVE FROM THE TOOLCHAIN.** The async-CDC set is derived from event semantics (shim row + `DMA_*_FINISHED_TASK` name), never hardcoded per kernel. The offset comes from measurement.
- **Event key format** is `"col|row|pkt|name"` (e.g. `"1|0|2|DMA_S2MM_0_FINISHED_TASK"`). The shim is row 0 (AIE2 topology).
- **Commits:** no emoji; end every commit message with the two trailer lines:
  `Generated using Claude Code.` and `Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh`
- **Branch is pure Python** (`tools/inference`). No Rust changes; `cargo test --lib` is not required.

## File Structure

- `tools/inference/grounding.py` — `Gap` dataclass gains `reproduction_offset`; new `is_async_cdc`; `ground_edge` rewritten. (Task 1)
- `tools/inference/facts.py` — new `derive_reproduction_offset` accessor. (Task 2)
- `tools/inference/rules.py` — `try_derives` gap branch emits a 5-arg `derives` fact carrying the reproduction offset. (Task 3)
- `tools/inference/engine.py` — `gaps` projection becomes `(child, parent, reproduction_offset)`. (Task 3)
- `tools/test_inference_grounding.py`, `tools/test_inference_facts.py`, `tools/test_inference_engine.py` — units (Tasks 1–3).
- `tools/test_experiment_loop_hw.py` — HW acceptance gate (Task 4).

---

### Task 1: `Gap.reproduction_offset`, `is_async_cdc`, and `ground_edge` rewrite

**Files:**
- Modify: `tools/inference/grounding.py`
- Test: `tools/test_inference_grounding.py`

**Interfaces:**
- Consumes: `offset_exact(run_dirs, a, b, anchor_key) -> Optional[int]`, `ANCHOR`, `Q` (from `inference.verifier`); `same_domain(a, b) -> bool` (same file).
- Produces:
  - `Gap(parent: str, child: str, reproduction_offset: Optional[int] = None)` (frozen dataclass).
  - `is_async_cdc(event_key: str) -> bool`.
  - `ground_edge(run_dirs, child, parent, anchor_key=ANCHOR) -> Segment | Gap` with new semantics.

- [ ] **Step 1: Write the failing tests**

Append to `tools/test_inference_grounding.py` (and add `is_async_cdc` to the existing import on line 2):

```python
def test_async_cdc_classifies_shim_dma_finished_only():
    assert is_async_cdc("1|0|2|DMA_S2MM_0_FINISHED_TASK") is True
    assert is_async_cdc("1|0|2|DMA_MM2S_0_FINISHED_TASK") is True
    assert is_async_cdc("1|0|2|DMA_S2MM_0_START_TASK") is False   # start, not egress
    assert is_async_cdc("1|2|1|DMA_S2MM_0_FINISHED_TASK") is False  # memtile/core, not shim
    assert is_async_cdc("1|2|0|INSTR_VECTOR") is False


def test_ground_edge_cross_domain_exact_carries_reproduction_offset(tmp_path):
    # exact raw offset (30 every run), cross-domain (shim pkt 2 vs core pkt 0):
    # a Gap, but annotated with the reproduction target.
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 35}])
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE", reproduction_offset=30)


def test_ground_edge_cross_domain_nonexact_no_reproduction_offset(tmp_path):
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 36}])  # range 1
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE", reproduction_offset=None)


def test_ground_edge_async_cdc_same_domain_is_gap_not_segment(tmp_path):
    # Both shim NoC-egress completions, SAME domain (1|0|2), exact offset 0 ->
    # WOULD be a spurious Segment(0); the async-CDC guard makes it a Gap.
    dirs = _runs(tmp_path, [
        {"1|0|2|DMA_MM2S_0_FINISHED_TASK": 0, "1|0|2|DMA_S2MM_0_FINISHED_TASK": 0},
        {"1|0|2|DMA_MM2S_0_FINISHED_TASK": 9, "1|0|2|DMA_S2MM_0_FINISHED_TASK": 9}])
    g = ground_edge(dirs, "1|0|2|DMA_S2MM_0_FINISHED_TASK",
                    "1|0|2|DMA_MM2S_0_FINISHED_TASK")
    assert g == Gap(parent="1|0|2|DMA_MM2S_0_FINISHED_TASK",
                    child="1|0|2|DMA_S2MM_0_FINISHED_TASK")
```

Also UPDATE the existing test `test_ground_edge_gap_when_cross_domain_even_if_exact` (currently lines ~55–61): its cross-domain offset is now annotated, so change its assertion to:

```python
def test_ground_edge_gap_when_cross_domain_even_if_exact(tmp_path):
    # exact offset (30 every run) but different modules (shim pkt 2 vs core pkt 0)
    # -> still a Gap (not a Segment), now carrying the reproduction offset.
    dirs = _runs(tmp_path, [{"1|0|2|MM2S": 0, "1|2|0|CORE": 30},
                            {"1|0|2|MM2S": 5, "1|2|0|CORE": 35}])
    g = ground_edge(dirs, "1|2|0|CORE", "1|0|2|MM2S")
    assert g == Gap(parent="1|0|2|MM2S", child="1|2|0|CORE", reproduction_offset=30)
```

The import line at the top becomes:

```python
from inference.grounding import (same_domain, ground_edge, assemble,
                                 Segment, Gap, Timeline, is_async_cdc)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_grounding.py -v`
Expected: the new tests FAIL (`is_async_cdc` not defined / `Gap` has no `reproduction_offset`); the updated `..._even_if_exact` test FAILS on the new assertion.

- [ ] **Step 3: Implement in `tools/inference/grounding.py`**

Change the typing import (line 13) to add `Optional`:

```python
from typing import List, Optional, Tuple, Union
```

Replace the `Gap` dataclass (lines 31–34) with:

```python
@dataclass(frozen=True)
class Gap:
    parent: str
    child: str
    reproduction_offset: Optional[int] = None
```

Add, immediately after `same_domain` (i.e. before the `Segment` dataclass):

```python
def is_async_cdc(event_key: str) -> bool:
    """True for shim NoC-egress DMA completion events. Their timing crosses the
    async 1 GHz<->960 MHz NoC FIFO to DDR (AM020 CDC) and is non-deterministic --
    never a cycle-deterministic causal fact. Derived from event semantics: a
    shim-row (row 0, AIE2 topology) DMA_*_FINISHED_TASK event. Gap-only: never a
    Segment, never a reproduction target."""
    parts = event_key.split("|")
    if len(parts) != 4:
        return False
    _col, row, _pkt, name = parts
    return row == "0" and name.startswith("DMA_") and name.endswith("_FINISHED_TASK")
```

Replace `ground_edge` (lines 40–48) with:

```python
def ground_edge(run_dirs: List[str], child: str, parent: str,
                anchor_key: str = ANCHOR) -> Grounding:
    """Within-domain exact offset -> Segment (cycle-accurate causal latency).
    Otherwise a named Gap. A cross-domain Gap carries the exact raw offset as a
    `reproduction_offset` when it agrees across runs (range <= Q), else None.
    Async-CDC events (shim NoC-egress DMA completion) are gap-only by semantics:
    never a Segment, never a reproduction target."""
    if is_async_cdc(child) or is_async_cdc(parent):
        return Gap(parent=parent, child=child)
    if same_domain(child, parent):
        off = offset_exact(run_dirs, child, parent, anchor_key)
        if off is not None:
            return Segment(parent=parent, child=child, offset=off)
        return Gap(parent=parent, child=child)
    raw = offset_exact(run_dirs, child, parent, anchor_key)
    return Gap(parent=parent, child=child, reproduction_offset=raw)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_grounding.py -v`
Expected: PASS (all, including the unchanged `test_ground_edge_segment_when_same_domain_and_exact`, `test_ground_edge_gap_when_same_domain_but_nonexact`, and `test_assemble_interleaves_segments_and_gaps` — the assemble test's cross-domain gap has a non-exact offset, so its `Gap(parent=..., child=...)` assertion still matches the `reproduction_offset=None` default).

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/inference/grounding.py tools/test_inference_grounding.py
git commit -m "feat(#140): cross-domain Gap reproduction_offset + async-CDC gap-only guard"
```
(Append the standard two-line trailer to the commit message.)

---

### Task 2: `derive_reproduction_offset` fact accessor

**Files:**
- Modify: `tools/inference/facts.py`
- Test: `tools/test_inference_facts.py`

**Interfaces:**
- Produces: `derive_reproduction_offset(fact: Fact) -> Optional[int]` — reads `fact.args[4]` for a 5-arg `derives` fact; `None` for shorter (legacy) facts, gaps with no target, or segments.
- The `derives` gap fact shape it reads is `(child, parent, None, "gap", reproduction_offset)` (produced in Task 3). Segments stay `(child, parent, offset, "segment")` (4-arg) and read as `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tools/test_inference_facts.py` (and add `derive_reproduction_offset` to the import on lines 2–3):

```python
def test_derive_reproduction_offset_for_cross_domain_gap():
    prem = _leaf("fired", ("1|2|0|CORE",))
    f = Fact("derives", ("1|2|0|CORE", "1|0|2|MM2S", None, "gap", 30),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) == 30


def test_derive_reproduction_offset_none_for_plain_gap():
    prem = _leaf("fired", ("1|0|2|S2MM",))
    f = Fact("derives", ("1|0|2|S2MM", "1|0|2|MM2S", None, "gap", None),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) is None


def test_derive_reproduction_offset_none_for_segment():
    prem = _leaf("fired", ("1|2|0|REL",))
    f = Fact("derives", ("1|2|0|REL", "1|2|0|ACQ", 22, "segment"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) is None


def test_derive_reproduction_offset_legacy_four_arg_gap_is_none():
    prem = _leaf("fired", ("1|0|2|S2MM",))
    f = Fact("derives", ("1|0|2|S2MM", "1|0|2|MM2S", None, "gap"),
             Derived("derives_rule_placement", (prem,)))
    assert derive_reproduction_offset(f) is None
```

The import becomes:

```python
from inference.facts import (Measured, Structural, Derived, Fact, KB,
                             leaves, provenance_ok, derive_kind, derive_offset,
                             derive_reproduction_offset)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_facts.py -v`
Expected: FAIL with `ImportError` / `cannot import name 'derive_reproduction_offset'`.

- [ ] **Step 3: Implement in `tools/inference/facts.py`**

Add immediately after `derive_offset` (after line 64):

```python
def derive_reproduction_offset(fact: Fact) -> Optional[int]:
    """The exact raw cross-domain reproduction-target offset for a gap derive.
    None for a segment, a gap with no deterministic target, an async-CDC gap, or a
    legacy <5-arg derives fact -- backward-compatible. This is NOT a causal offset;
    `derive_offset` (args[2]) stays None for gaps."""
    return fact.args[4] if len(fact.args) >= 5 else None
```

(`Optional` is already imported on line 11.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_facts.py -v`
Expected: PASS (all, including the existing `test_derive_kind_*` tests, which are unaffected).

- [ ] **Step 5: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/inference/facts.py tools/test_inference_facts.py
git commit -m "feat(#140): derive_reproduction_offset accessor for gap derives facts"
```
(Append the standard two-line trailer.)

---

### Task 3: Thread `reproduction_offset` through `try_derives` and the engine report

**Files:**
- Modify: `tools/inference/rules.py` (gap branch of `try_derives`)
- Modify: `tools/inference/engine.py` (`gaps` projection + import)
- Test: `tools/test_inference_engine.py`

**Interfaces:**
- Consumes: `Gap.reproduction_offset` (Task 1); `derive_reproduction_offset` (Task 2); `derive_kind` (existing).
- Produces: a gap `derives` fact `(child, parent, None, "gap", reproduction_offset)`; an engine report `gaps` list of `(child, parent, reproduction_offset)` 3-tuples.

- [ ] **Step 1: Write the failing test + update the breaking one**

In `tools/test_inference_engine.py`, UPDATE the assertion on line 89 (in `test_engine_reports_segment_and_gap`) from the 2-tuple to the 3-tuple form (this cross-domain pair is non-exact, so the target is `None`):

```python
    assert ("1|2|0|CORE", "1|0|2|MM2S", None) in rep["gaps"]
```

Append a new test asserting an EXACT cross-domain gap carries its reproduction offset end-to-end through the engine:

```python
def test_engine_gap_carries_reproduction_offset(tmp_path):
    # cross-domain pair (shim 1|0|2 MM2S -> core 1|2|0 CORE) with an EXACT raw
    # offset (40 every run) -> gap annotated with reproduction_offset=40.
    dirs = []
    for i, row in enumerate([
        {"1|0|2|MM2S": 0, "1|2|0|CORE": 40},
        {"1|0|2|MM2S": 9, "1|2|0|CORE": 49}
    ]):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for key, delta in row.items():
            col, r, pkt, name = key.split("|")
            evs.append(_ev(int(col), int(r), name, 1000 + delta, pkt_type=int(pkt)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    led = tmp_path / "led.json"
    led.write_text(json.dumps({"entries": [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}]}))
    rep = run_engine(dirs, str(led), [("1|2|0|CORE", "1|0|2|MM2S")])
    assert ("1|2|0|CORE", "1|0|2|MM2S", 40) in rep["gaps"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_engine.py -v`
Expected: `test_engine_gap_carries_reproduction_offset` FAILS (gaps are 2-tuples, no offset); the updated `test_engine_reports_segment_and_gap` FAILS (still emitting 2-tuples).

- [ ] **Step 3: Implement the rules.py gap branch**

In `tools/inference/rules.py`, replace the gap branch of `try_derives` (current lines 54–57) with:

```python
    grd = Fact("gap", (child, parent),
               Derived("grounding_rule", _measured_premises(kb, child, parent)))
    return Fact("derives", (child, parent, None, "gap", g.reproduction_offset),
                Derived("derives_rule_placement", (cp, grd)))
```

(`g` is the `Gap` returned by `ground_edge` on line 48; the segment branch above is unchanged.)

- [ ] **Step 4: Implement the engine.py projection**

In `tools/inference/engine.py`, extend the facts import (line 14):

```python
from inference.facts import (KB, provenance_ok, derive_kind, derive_offset,
                             derive_reproduction_offset)
```

Replace the `gaps` projection (current lines 56–57) with:

```python
    gaps = [(f.args[0], f.args[1], derive_reproduction_offset(f))
            for f in derives_facts if derive_kind(f) == "gap"]
```

(The `segments` projection above is unchanged. `run_experiment.py` passes `gaps` straight through to the report, so the reproduction target now appears in the JSON report's `gaps` list as the third element — no change needed there.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_engine.py test_inference_grounding.py test_inference_facts.py -v`
Expected: PASS (all three files). Confirms the cross-task wiring: `Gap.reproduction_offset` -> `derives` 5-tuple -> `derive_reproduction_offset` -> report `gaps` 3-tuple.

- [ ] **Step 6: Run the full offline inference suite for regressions**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_inference_engine.py test_inference_grounding.py test_inference_facts.py test_inference_loop.py -v`
Expected: PASS, zero regressions. (If `test_inference_loop.py` is absent, run the three inference test files plus any other `test_inference_*.py` present: `python -m pytest -k inference -v`.)

- [ ] **Step 7: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/inference/rules.py tools/inference/engine.py tools/test_inference_engine.py
git commit -m "feat(#140): surface cross-domain reproduction_offset through engine report gaps"
```
(Append the standard two-line trailer.)

---

### Task 4: HW acceptance gate

**Files:**
- Modify: `tools/test_experiment_loop_hw.py`

**Interfaces:**
- Consumes: `run_experiment(cfg) -> report` with `report["gaps"]` now `(child, parent, reproduction_offset)` 3-tuples and `report["segments"]` unchanged; `is_async_cdc` (Task 1).
- This task is **CODE-ONLY** in the subagent run (write + commit). The controller runs the HW suite once, batched, against the finished branch (real NPU1; `XDNA_HW_SMOKE=1`, `env -u XDNA_EMU`). Do NOT invoke the NPU from the implementer subagent.

- [ ] **Step 1: Write the HW assertions**

Append to `tools/test_experiment_loop_hw.py`:

```python
def test_cross_domain_gap_carries_reproduction_offset_hw(tmp_path):
    # A cross-domain edge that agrees exactly across runs (broadcast-skew-locked,
    # outside DMA jitter) is a GAP carrying the exact raw offset as a reproduction
    # target -- the byte value the emulator broadcast model will be validated
    # against. (limit doc: docs/trace/cross-domain-skew-limit.md)
    from inference.run_experiment import run_experiment
    rep = run_experiment(_cfg(tmp_path))
    assert rep["engine_ok"] is True
    annotated = [g for g in rep["gaps"] if len(g) >= 3 and isinstance(g[2], int)]
    assert annotated, (
        "expected at least one cross-domain gap with a reproduction offset; "
        f"gaps={rep['gaps']}")


def test_async_cdc_finished_stays_gap_with_no_reproduction_offset_hw(tmp_path):
    # Shim NoC-egress DMA completion is async-CDC: gap-only and never a
    # reproduction target (reproduction_offset is None), never a segment.
    from inference.run_experiment import run_experiment
    from inference.grounding import is_async_cdc
    rep = run_experiment(_cfg(tmp_path))
    seg_children = {s[0] for s in rep["segments"]}
    async_gaps = [g for g in rep["gaps"] if is_async_cdc(g[0])]
    for g in async_gaps:
        assert g[2] is None, f"async-CDC {g[0]} must carry no reproduction offset; {g}"
        assert g[0] not in seg_children, f"async-CDC {g[0]} must not be a segment"
```

- [ ] **Step 2: Verify the tests collect and skip cleanly (NO HW run)**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_experiment_loop_hw.py --collect-only -q`
Expected: both new tests are collected (the module is `skipif XDNA_HW_SMOKE != 1`, so they SKIP when run without the env var — that is correct; do not set the env var here).

- [ ] **Step 3: Commit**

```bash
cd /home/triple/npu-work/xdna-emu
git add tools/test_experiment_loop_hw.py
git commit -m "test(#140): HW gate for cross-domain reproduction offset + async-CDC gap-only"
```
(Append the standard two-line trailer.)

- [ ] **Step 4 (CONTROLLER, batched HW run): run the HW suite on real NPU1**

After all tasks are merged-ready, the controller runs once (NPU health checked first; never two HW suites concurrently):

```bash
cd /home/triple/npu-work/xdna-emu/tools
XDNA_HW_SMOKE=1 env -u XDNA_EMU XDNA_EMU_RUNTIME=release \
  python -m pytest test_experiment_loop_hw.py -v 2>&1 | tee /tmp/cross-domain-hw.log
```
Expected: all HW tests PASS, including the two new ones and the unchanged `test_core_lock_segment_grounds_exact_hw` (within-domain segment unchanged) and `test_through_core_event_is_placed_as_gap_hw`.

**HW-validated items to confirm at this run** (flag if any fails — each is a real finding, not a code bug to silently patch):
1. At least one cross-domain candidate edge has a range-0 raw offset over the run set (so `reproduction_offset` is populated). If empty: investigate whether add_one's deterministic cross-domain offsets (the −2/+2/+4 class) reach the candidate set.
2. `test_async_cdc_finished_stays_gap_with_no_reproduction_offset_hw` is only non-vacuous if a shim `DMA_*_FINISHED_TASK` appears as a derives child. If add_one's config does not trace/candidate those events, the test passes vacuously — the deterministic guarantee is fully covered by the Task 1 unit test `test_ground_edge_async_cdc_same_domain_is_gap_not_segment`. Note vacuity in the run report; do not treat it as a gap in coverage.

---

## Self-Review

**1. Spec coverage:**
- Cross-domain `Gap` reproduction-target annotation (range≤Q else None) → Task 1 (`ground_edge`) + Task 3 (through facts/report). ✓
- Async-CDC gap-only guard (incl. the same-domain false-ground fix) → Task 1 (`is_async_cdc` + `ground_edge`). ✓
- Facts/report plumbing → Task 2 (accessor) + Task 3 (rules + engine projection). ✓
- `is_async_cdc` classifies shim NoC-egress completions and nothing else → Task 1 unit. ✓
- Within-domain byte-identical regression → covered by the unchanged grounding/engine segment tests (Task 1 Step 4, Task 3 Step 5) and the HW `test_core_lock_segment_grounds_exact_hw`. ✓
- Cross-domain→Gap-with-target and →None, async-CDC→Gap unit tests → Task 1. ✓
- HW acceptance gate → Task 4. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code and exact commands. ✓

**3. Type consistency:**
- `Gap(parent, child, reproduction_offset: Optional[int] = None)` — used identically in Tasks 1, 3. ✓
- `derives` gap fact `(child, parent, None, "gap", reproduction_offset)` — produced in Task 3 (rules), read by `derive_reproduction_offset` (args[4]) from Task 2, projected in Task 3 (engine). ✓
- `gaps` report tuple `(child, parent, reproduction_offset)` — produced in Task 3 (engine), asserted in Task 3 + Task 4 tests; existing HW tests read only `g[0]` (child), unaffected. ✓
- `derive_offset` (args[2]) stays `None` for gaps — `reproduction_offset` is args[4], never conflated. ✓

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-23-cross-domain-reproduction-targets.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
