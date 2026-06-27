# Connectivity: Cross-Column Edge Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trace-inference engine model inter-tile/cross-column communication as logical dataflow conversations and report each honestly (grounded / observed-but-ungrounded / unobserved), replacing the physical-hop connectivity oracle that flags transit wiring as defects; then diagnose the compute-DMA trace gap that blocks the hardware proof.

**Architecture:** P1 (Tasks 1-4, offline): a new pure `connectivity.classify_connectivity` audits the cross-domain candidate pairs (route-graph reachability the ledger already computes) against the trace and weave result, emitting a three-way classification; `assemble_timeline` wires it in and the old physical-hop `coupling_oracle`/`connectivity_defects` are removed. P2 (Task 5, HW-gated): a timeboxed diagnosis spike isolating why compute memmod (`pkt_type 1`) trace never reaches the buffer, ending in a findings doc and a decision gate.

**Tech Stack:** Python 3.13, pytest. Tools live under `tools/`; tests run with `cd tools && python -m pytest <file> -q`. The inference package is `tools/inference/`.

## Global Constraints

Every task inherits these (copied from the spec's Global Constraints):

- **No statistical inference. Q=0.** Connectivity is derived from route reachability + observed firing; never inferred from timing correlation, never tuned to make a test pass.
- **Derive from the toolchain.** The route graph (and the ledger reachability over it) is the authoritative source for which tiles talk. No hardcoded topology.
- **HW is the cheap fast ground-truth oracle; EMU is the thing under test.**
- **HW discipline (Task 5 only):** never two HW suites concurrently; no `xrt-smi` during HW runs; `dmesg` never via pkexec; privileged ops via `pkexec` (one combined call), never `sudo`; for HW from a poisoned shell use `env -u XDNA_EMU XDNA_EMU_RUNTIME=release`; never self-reboot (hand to Maya via `!`); reboot-first when the kernel is wedged.
- **Build/test discipline:** never pipe long builds/tests through tail/head/grep (redirect to file or run in background; tee OK); run bare; `cargo test --lib` after any Rust change.
- **Never persistent work in `/tmp`.** Experiment output under `build/experiments/`.
- **Commits:** no emoji; end messages with the standard Claude Code trailer. Internal project — no commit-message pre-approval; show EXTERNAL posts before posting. Push to origin only on Maya's explicit say-so.
- **Branch:** `connectivity-cross-column-grounding` (already created; spec committed there).

## Key Facts (verified during planning)

- `assemble_timeline(run_dirs, configured, derives_pairs, cross_domain_pairs, dump=None, start_col=1, anchor_key=ANCHOR, capture=None)` already receives `cross_domain_pairs` — the route-implied conversations at event-key granularity (`(child_key, parent_key)`, already cross-domain-filtered upstream in `engine.py`).
- An event key is `"col|row|pkt|name"`; its tile is `"col|row"`; its domain is `"col|row|pkt"`.
- `weave(run_dirs, cross_domain_pairs, anchor_key)` returns `List[CrossTrackEdge]`; `CrossTrackEdge` has `.child`, `.parent`, `.reason`, `.reproduction_offset`. It only grounds pairs where **both** endpoints fired; for cross-domain pairs `ground_edge` always returns a `Gap`, so a both-fired cross-domain pair always grounds (hence `observed_but_ungrounded` is unreachable for the live engine today — but the classifier defines it and it is unit-tested directly).
- `load_fired(run_dirs, anchor_key)` returns `Fact`s; `f.args[0]` is the event key. `from inference.loader import load_fired`.
- Verified two_col offline result with the new classifier: **1 grounded** (`1|0~1|1`), **8 unobserved** (every coupling whose other end is a compute DMA endpoint — the P2 gap, including all 4 cross-column couplings), **0 defects**. Old behavior emitted ~11 physical-hop `connectivity_defect` flags.
- Test helpers in `tools/test_timeline.py`: `_write_run(base, run_name, events)` writes `run_name/batch_00/hw/trace.events.json` and returns the path; `_ev(name, soc, col=1, row=2, pkt_type=0, slot=0)` builds one event dict. `_EXP = "/home/triple/npu-work/xdna-emu/build/experiments"`.
- The experiment captures under `build/experiments/` are git-ignored; real-data tests use `@pytest.mark.skipif(not os.path.isdir(...))`.

---

### Task 1: `connectivity.classify_connectivity` (pure classifier + unit tests)

**Files:**
- Create: `tools/inference/connectivity.py`
- Test: `tools/test_connectivity.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads only event-key strings, a `fired` set, and edge objects with `.child`/`.parent`.
- Produces: `classify_connectivity(cross_domain_pairs: List[Tuple[str,str]], fired: Set[str], edges) -> Dict[Tuple[str,str], str]` returning `{sorted (tileA,tileB): status}`; the status string constants `GROUNDED = "grounded"`, `OBSERVED_UNGROUNDED = "observed_but_ungrounded"`, `UNOBSERVED = "unobserved"`; and `_tile(key) -> "col|row"`.

- [ ] **Step 1: Write the failing tests**

Create `tools/test_connectivity.py`:

```python
"""Unit tests for the pure logical-connectivity classifier."""
from types import SimpleNamespace

from inference.connectivity import (
    classify_connectivity, GROUNDED, OBSERVED_UNGROUNDED, UNOBSERVED, _tile)


def _edge(child, parent):
    return SimpleNamespace(child=child, parent=parent)


def test_tile_extracts_col_row():
    assert _tile("2|4|1|DMA_MM2S_0_START_TASK") == "2|4"
    assert _tile("1|1|3") == "1|1"


def test_grounded_when_both_fired_and_edge_present():
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    fired = {"2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"}
    edges = [_edge("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    out = classify_connectivity(pairs, fired, edges)
    assert out == {("1|1", "2|4"): GROUNDED}


def test_observed_but_ungrounded_when_both_fired_no_edge():
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    fired = {"2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"}
    out = classify_connectivity(pairs, fired, edges=[])
    assert out == {("1|1", "2|4"): OBSERVED_UNGROUNDED}


def test_unobserved_when_an_endpoint_did_not_fire():
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    fired = {"1|1|3|PORT_RUNNING_6"}          # col-2 DMA endpoint never fired
    out = classify_connectivity(pairs, fired, edges=[])
    assert out == {("1|1", "2|4"): UNOBSERVED}


def test_same_tile_pairs_are_skipped():
    # Two events on the same tile (different module) are NOT a cross-tile
    # conversation -> not in the connectivity report.
    pairs = [("1|2|1|DMA_MM2S_0_START_TASK", "1|2|0|INSTR_VECTOR")]
    out = classify_connectivity(pairs, fired={"1|2|1|DMA_MM2S_0_START_TASK",
                                              "1|2|0|INSTR_VECTOR"}, edges=[])
    assert out == {}


def test_grounded_wins_when_pair_seen_both_ways():
    # One candidate pair for a coupling grounds; another for the same tile pair
    # has an unfired endpoint. The coupling is grounded (at least one grounded).
    pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"),
             ("2|4|1|DMA_S2MM_0_START_TASK", "1|1|3|PORT_RUNNING_7")]
    fired = {"2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6"}
    edges = [_edge("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    out = classify_connectivity(pairs, fired, edges)
    assert out == {("1|1", "2|4"): GROUNDED}


def test_same_tile_edge_does_not_leak_as_grounded():
    # A weave edge between two events on the SAME tile (cross-module, e.g. core
    # pkt0 -> memmod pkt1) is cross-domain, so weave can emit an edge for it. Its
    # tile projection is ("1|2","1|2") -- a same-tile pair the module must NOT
    # report. Guards the `all_pairs |= grounded` fold against same-tile leakage.
    edges = [_edge("1|2|1|DMA_MM2S_0_START_TASK", "1|2|0|INSTR_VECTOR")]
    out = classify_connectivity([], fired=set(), edges=edges)
    assert out == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_connectivity.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.connectivity'`.

- [ ] **Step 3: Write the implementation**

Create `tools/inference/connectivity.py`:

```python
"""Logical connectivity classification for the integrated timeline.

The connectivity oracle is the set of cross-tile logical CONVERSATIONS the
config implies -- the tile-pair projection of the cross-domain candidate pairs
(route-graph reachability that generate_ledger computes over configured events).
It is NOT the physical adjacent-hop wiring the old coupling_oracle enumerated.

Each conversation is classified honestly against the trace:

  grounded                -- some candidate pair for the tile coupling had both
                             endpoints fire AND weave produced a CrossTrackEdge
                             connecting the two tiles.  Healthy: no flag.
  observed_but_ungrounded -- both endpoints fired for some candidate pair, but
                             weave grounded no edge between the tiles -> the
                             genuine connectivity defect.
  unobserved              -- no candidate pair for this coupling had both
                             endpoints fire -> honest gap (the trace does not
                             watch both ends), NOT a defect.

NOTE: with the present grounding a both-fired cross-domain pair always resolves
to a Gap and therefore always grounds, so the live engine does not currently
produce observed_but_ungrounded; that bucket is reserved for a future grounding
that can fail on observed pairs.  The classifier defines it regardless and it is
unit-tested directly.

Weave couples; this module audits.  Q=0: derived from route reachability +
observed firing, never inferred from timing correlation, never tuned to pass.
"""
from typing import Dict, List, Set, Tuple

GROUNDED = "grounded"
OBSERVED_UNGROUNDED = "observed_but_ungrounded"
UNOBSERVED = "unobserved"


def _tile(key: str) -> str:
    """'col|row|pkt|name' or 'col|row|pkt' -> 'col|row'."""
    parts = key.split("|")
    return f"{parts[0]}|{parts[1]}"


def classify_connectivity(cross_domain_pairs: List[Tuple[str, str]],
                          fired: Set[str],
                          edges) -> Dict[Tuple[str, str], str]:
    """Classify each cross-tile logical conversation into a connectivity status.

    cross_domain_pairs: (child_key, parent_key) candidate pairs from the ledger
        reachability.  Same-tile pairs are ignored -- intra-tile handoffs are not
        cross-track conversations.
    fired: event keys observed to fire across the runs.
    edges: iterable of objects with .child and .parent event-key attributes
        (weave's CrossTrackEdge list).

    Returns {sorted (tileA, tileB): status} for every cross-tile coupling seen
    in the candidate pairs or grounded by weave.
    """
    # A weave edge can be cross-MODULE but same-TILE (core pkt0 -> memmod pkt1);
    # such an edge is not a cross-tile conversation, so drop it here -- otherwise
    # the `all_pairs |= grounded` fold below would reintroduce the same-tile pair.
    grounded: Set[Tuple[str, str]] = set()
    for e in edges:
        ta_e, tb_e = _tile(e.child), _tile(e.parent)
        if ta_e != tb_e:
            grounded.add(tuple(sorted((ta_e, tb_e))))
    all_pairs: Set[Tuple[str, str]] = set()
    observed: Set[Tuple[str, str]] = set()
    for child, parent in cross_domain_pairs:
        ta, tb = _tile(child), _tile(parent)
        if ta == tb:
            continue
        pr = tuple(sorted((ta, tb)))
        all_pairs.add(pr)
        if child in fired and parent in fired:
            observed.add(pr)
    all_pairs |= grounded  # a grounded coupling is reported even if only edges saw it
    out: Dict[Tuple[str, str], str] = {}
    for pr in sorted(all_pairs):
        if pr in grounded:
            out[pr] = GROUNDED
        elif pr in observed:
            out[pr] = OBSERVED_UNGROUNDED
        else:
            out[pr] = UNOBSERVED
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_connectivity.py -q`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/connectivity.py tools/test_connectivity.py
git commit -m "feat(#140): logical connectivity classifier (three-way honest status)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 2: Wire the classifier into `assemble_timeline`; remove the physical-hop oracle

**Files:**
- Modify: `tools/inference/timeline.py` (remove `_tile_of`, `coupling_oracle`, `connectivity_defects`; rewrite the connectivity block in `assemble_timeline`)
- Modify: `tools/test_timeline.py` (drop/replace tests of the removed functions)
- Modify: `tools/test_inference_engine.py` (replace the dump-oracle defect test with an unobserved-flag test; refresh comments)

**Interfaces:**
- Consumes: `classify_connectivity`, `GROUNDED`, `OBSERVED_UNGROUNDED`, `UNOBSERVED` from Task 1; `load_fired` from `inference.loader`.
- Produces: `assemble_timeline` emits timeline flags `connectivity_defect:<a>~<b>` (for `observed_but_ungrounded`) and `connectivity_unobserved:<a>~<b>` (for `unobserved`); grounded couplings emit no flag. `coupling_oracle`, `connectivity_defects`, and `_tile_of` no longer exist in `timeline.py`.

- [ ] **Step 1: Write/adjust the failing tests**

In `tools/test_timeline.py`:

(a) **Delete** `test_coupling_oracle_from_route_graph` entirely (the physical oracle is gone).

(b) In `test_weave_and_connectivity`, **delete** the two `connectivity_defects` assertion lines (the classifier is tested in `test_connectivity.py`); keep the weave assertions. The function body becomes:

```python
def test_weave_and_connectivity(tmp_path, monkeypatch):
    from inference.grounding import Gap, GAP_CROSS_DOMAIN
    monkeypatch.setattr(T, "ground_edge",
        lambda runs, c, p, anchor=T.ANCHOR: Gap(parent=p, child=c, reason=GAP_CROSS_DOMAIN, reproduction_offset=7))
    # Real run dirs where both endpoints fire, so the weave firing-gate admits the
    # pair (the monkeypatched ground_edge then controls the offset). X=1|1|3, Y=1|2|0.
    runs = [_write_run(tmp_path, f"run_{i}", [
        _ev("PERF_CNT_2", 0), _ev("Y", 50),
        _ev("X", 30, col=1, row=1, pkt_type=3)]) for i in range(2)]
    edges = T.weave(runs, [("1|1|3|X", "1|2|0|Y")])
    assert edges[0].reproduction_offset == 7 and edges[0].reason == GAP_CROSS_DOMAIN
```

(c) In `test_assemble_timeline_multi_track_two_domains`, **replace** step (3) (the `coupling_oracle`/`connectivity_defects` lines) with classification assertions. The coupling `1|1~1|2` grounds (both A and Y fire), so it is `grounded` and emits no flag:

```python
    # (3) the coupled tile pair grounds -> classified grounded, no connectivity flag.
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)
    assert not any(str(f).startswith("connectivity_unobserved") for f in tl.flags)
```

(d) **Replace** `test_assemble_timeline_crosscolumn_coupling_grounded_or_flagged` with a version driven by `cross_domain_pairs` (the new model). A cross-column conversation whose col-2 endpoint never fires must surface as `connectivity_unobserved` — never silently dropped:

```python
def test_assemble_timeline_crosscolumn_coupling_unobserved_when_one_end_silent(tmp_path):
    # A cross-column conversation (memtile col1 -> compute col2 DMA) whose col-2
    # endpoint never fires must surface as connectivity_unobserved -- never
    # silently dropped (disconnected-but-honest).
    runs = []
    for i in range(3):
        runs.append(_write_run(tmp_path, f"run_{i}", [
            _ev("PERF_CNT_2", 0, col=1, row=2, pkt_type=0),       # anchor (col 1)
            _ev("PORT_RUNNING_6", 10 + i, col=1, row=1, pkt_type=3),  # memtile fires
            # col-2 compute DMA endpoint (2|4|1|DMA_MM2S_0_START_TASK) NEVER written
        ]))
    configured = ["1|2|0|PERF_CNT_2", "1|1|3|PORT_RUNNING_6", "2|4|1|DMA_MM2S_0_START_TASK"]
    cross_domain_pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    tl = T.assemble_timeline(runs, configured, derives_pairs=set(),
                             cross_domain_pairs=cross_domain_pairs,
                             dump=None, start_col=1)
    assert "connectivity_unobserved:1|1~2|4" in tl.flags
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)
```

In `tools/test_inference_engine.py`:

(e) In `test_engine_dump_connectivity_oracle_no_defect`, update the comment to reflect that the oracle is now the candidate pairs (the dump's route edge is no longer the connectivity source); the assertion (no `connectivity_defect`) stands because the pair `(1|2|0|CORE, 1|0|2|MM2S)` grounds. Rename to `test_engine_connectivity_no_defect_when_grounded`. Body:

```python
def test_engine_connectivity_no_defect_when_grounded(tmp_path):
    # The candidate pair (CORE<-MM2S) couples tiles 1|2 and 1|0; both fire and
    # the weave grounds it -> grounded -> no connectivity_defect / unobserved flag.
    dirs = _cross_domain_dirs(tmp_path)
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|2|0|CORE", "kind": "program"}])
    rep = run_engine(dirs, led, [("1|2|0|CORE", "1|0|2|MM2S")], start_col=1)
    tl = rep["timeline"]
    assert isinstance(tl, IntegratedTimeline)
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)
    assert not any(str(f).startswith("connectivity_unobserved") for f in tl.flags)
```

(f) **Replace** `test_engine_dump_connectivity_oracle_flags_defect` with an unobserved-flag test (the live engine cannot produce `observed_but_ungrounded`, but it produces `unobserved` when an endpoint never fires). Rename to `test_engine_connectivity_unobserved_flag`:

```python
def test_engine_connectivity_unobserved_flag(tmp_path):
    # A candidate pair whose child endpoint never fires -> the conversation is
    # unobserved (honest gap), surfaced as connectivity_unobserved, NOT a defect.
    dirs = _cross_domain_dirs(tmp_path)   # fires MM2S (1|0|2) and CORE (1|2|0)
    led = _ledger(tmp_path, [
        {"cite": "program:x", "a": "1|0|2|MM2S", "b": "1|3|1|DMA", "kind": "program"}])
    rep = run_engine(dirs, led, [("1|3|1|DMA", "1|0|2|MM2S")], start_col=1)
    tl = rep["timeline"]
    assert "connectivity_unobserved:1|0~1|3" in tl.flags
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)
```

(g) In `test_engine_backward_compat_no_dump`, update the comment (connectivity now runs without a dump); the assertions stand (the grounded pair emits no defect). No code change beyond the comment.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_timeline.py test_inference_engine.py -q`
Expected: FAIL — e.g. `AttributeError: module 'inference.timeline' has no attribute 'coupling_oracle'` is NOT yet the error (functions still exist); instead the new tests fail because `assemble_timeline` does not yet emit `connectivity_unobserved` flags (`assert "connectivity_unobserved:1|1~2|4" in tl.flags` fails) and the renamed engine test asserts a flag not yet produced.

- [ ] **Step 3: Edit `timeline.py`**

(i) **Remove** the `_tile_of` function (the `def _tile_of(...)` block, currently around lines 530-534 — it is used only by `connectivity_defects`).

(ii) **Remove** the `coupling_oracle` function (the whole `def coupling_oracle(dump, start_col) -> set:` block).

(iii) **Remove** the `connectivity_defects` function (the whole `def connectivity_defects(oracle, edges) -> List[...]:` block).

(iv) In `assemble_timeline`, **replace** the connectivity block:

```python
    # (9) cross-track weave + independent connectivity oracle (defects need dump).
    edges = weave(run_dirs, cross_domain_pairs, anchor_key)
    if dump is not None:
        defects = connectivity_defects(coupling_oracle(dump, start_col), edges)
        flags.extend(f"connectivity_defect:{a}~{b}" for (a, b) in defects)
```

with:

```python
    # (9) cross-track weave + logical connectivity classification. The conversation
    # set is the cross-domain candidate pairs (route reachability the ledger
    # computes); classify each cross-tile coupling honestly against the trace.
    # weave couples; classify_connectivity audits. No dump needed.
    edges = weave(run_dirs, cross_domain_pairs, anchor_key)
    from inference.connectivity import (classify_connectivity, OBSERVED_UNGROUNDED,
                                        UNOBSERVED)
    from inference.loader import load_fired
    fired = {f.args[0] for f in load_fired(run_dirs, anchor_key)}
    for (a, b), status in sorted(classify_connectivity(cross_domain_pairs, fired, edges).items()):
        if status == OBSERVED_UNGROUNDED:
            flags.append(f"connectivity_defect:{a}~{b}")
        elif status == UNOBSERVED:
            flags.append(f"connectivity_unobserved:{a}~{b}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_timeline.py test_inference_engine.py test_connectivity.py -q`
Expected: PASS — all green (deletions removed the obsolete tests; the new flag tests pass).

- [ ] **Step 5: Run the broader Python suite to catch fallout**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest -q 2>&1 | tail -20`
Expected: PASS (no other module referenced the removed functions; any pre-existing skips remain skips). If a test fails because it referenced `coupling_oracle`/`connectivity_defects`, fix it to the new model.

- [ ] **Step 5b: Remove now-dead imports/helpers left by the deletions**

The deleted tests were the only users of some imports/helpers. Grep and remove anything now unused (the plan's "removed, not left dead" rule applies to test scaffolding too):

```bash
cd /home/triple/npu-work/xdna-emu/tools
# confirm the removed names are gone everywhere
grep -rn "coupling_oracle\|connectivity_defects\|_tile_of" --include=*.py inference/ test_timeline.py test_inference_engine.py
# find now-unused test scaffolding to prune (verify each before deleting):
grep -n "from config_extract.dump_model import\|def _pr\|RouteGraph\|RouteEdge\|PortRef\|ConfigDump" test_timeline.py test_inference_engine.py
```
- In `test_timeline.py`: the import at line ~281 (`from config_extract.dump_model import ConfigDump, RouteGraph, RouteEdge, PortRef`) and the `_pr` helper were used by the deleted `test_coupling_oracle_from_route_graph` and the replaced crosscolumn test. If no remaining test in the file uses them (the `_RouteGraph`/`_RouteEdge`/`_PortRef`/`_ConfigDump` aliases at ~594-597 are a *separate* import still used by `test_assemble_timeline_multi_track_two_domains`), remove the now-unused line-281 import and `_pr`. Keep whatever the surviving tests still reference.
- In `test_inference_engine.py`: after replacing the two dump-oracle tests, the `RouteGraph`/`RouteEdge`/`ConfigDump`/`_pr` imports/helpers are unused (the new tests pass no dump). Remove them. Refresh the stale comments referencing the dump-based oracle.
- The duplicate `load_fired` pass (weave loads it internally; `assemble_timeline` loads it again for connectivity) is harmless and intentional — leave it.

Re-run the suite after pruning: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest -q 2>&1 | tail -5` — expected still green.

- [ ] **Step 6: Commit**

```bash
git add tools/inference/timeline.py tools/test_timeline.py tools/test_inference_engine.py
git commit -m "feat(#140): logical connectivity classification in assemble_timeline

Replace the physical-hop coupling_oracle/connectivity_defects with the
logical classify_connectivity over the cross-domain candidate pairs. Emit
connectivity_defect only for genuine observed-but-ungrounded couplings and
connectivity_unobserved for honest gaps; grounded couplings emit no flag.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 3: Real-capture characterization on two_col

**Files:**
- Modify: `tools/test_timeline.py` (add one skipif-gated real-data test near the Task-13 real-data block)

**Interfaces:**
- Consumes: `assemble_timeline` with the new classifier (Task 2); the on-disk capture `build/experiments/two_col_capture/` (run dirs + `ledger.json`) and `tools/config_extract/fixtures/two_col.config.json`.
- Produces: a regression test asserting the honest two_col connectivity picture.

- [ ] **Step 1: Write the failing test**

Add to `tools/test_timeline.py` (near the other real-data tests; reuses the module-level `import glob, os, pytest`):

```python
_TWO_COL_CAP = f"{_EXP}/two_col_capture/cap"


@pytest.mark.skipif(not os.path.isdir(_TWO_COL_CAP),
                    reason="two_col HW capture not present (git-ignored experiment dir)")
def test_two_col_connectivity_is_logical_and_honest(tmp_path):
    # On the real two_col capture, the logical connectivity report must:
    #  - contain the cross-column conversation 1|1~2|4 (memtile -> col-2 compute),
    #  - contain NO physical transit-hop pair (e.g. 1|2~2|2),
    #  - classify the cross-column couplings UNOBSERVED (col-2 DMA never fired),
    #  - classify 1|0~1|1 (shim~memtile, both ends fired) GROUNDED,
    #  - emit ZERO connectivity_defect flags (the old behavior emitted ~11).
    from config_extract.dump_model import load_dump
    from inference.selfmodel import (enumerate_configured_events,
                                     candidate_pairs_from_dump)
    from inference.engine import run_engine

    dump = load_dump("config_extract/fixtures/two_col.config.json")
    sc = 1
    configured = enumerate_configured_events(dump, sc)
    cps = candidate_pairs_from_dump(dump, configured, sc)
    run_dirs = sorted(glob.glob(f"{_TWO_COL_CAP}/capture_00/run_*"))
    assert run_dirs, "no run dirs under the two_col capture"
    rep = run_engine(run_dirs, f"{_TWO_COL_CAP}/ledger.json", cps,
                     dump=dump, start_col=sc)
    flags = rep["timeline"].flags

    # logical (not physical): the conversation exists, the transit hop does not.
    assert "connectivity_unobserved:1|1~2|4" in flags
    assert not any("1|2~2|2" in str(f) for f in flags)        # physical transit hop gone
    # all four cross-column couplings are unobserved (col-2 DMA endpoints silent).
    for pr in ("1|0~2|4", "1|0~2|5", "1|1~2|4", "1|1~2|5"):
        assert f"connectivity_unobserved:{pr}" in flags, pr
    # shim~memtile grounds -> NOT flagged either way.
    assert not any(f"1|0~1|1" in str(f) for f in flags)
    # zero genuine defects.
    assert not any(str(f).startswith("connectivity_defect") for f in flags)
```

- [ ] **Step 2: Run the test to verify it passes (and prove it would have failed)**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_timeline.py -q -k two_col_connectivity`
Expected: PASS (the capture is present on this box).

Prove the RED per SP1 discipline: temporarily `git stash` the Task-2 change to `timeline.py` (or revert the connectivity block to the physical oracle), re-run, and confirm the test FAILS (old behavior emits `connectivity_defect:1|2~2|2` and no `connectivity_unobserved`). Restore the change. Record the observed failure in the task report. (Do not commit the revert.)

- [ ] **Step 3: Commit**

```bash
git add tools/test_timeline.py
git commit -m "test(#140): two_col real-capture connectivity is logical and honest

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 4: Synthetic both-sides-observed cross-column grounded test

**Files:**
- Modify: `tools/test_timeline.py` (add one offline test)

**Interfaces:**
- Consumes: `assemble_timeline` with the new classifier (Task 2).
- Produces: the offline stand-in proof that a cross-column conversation grounds when both ends are observed (what the P2 fix will make true on hardware).

- [ ] **Step 1: Write the failing test**

Add to `tools/test_timeline.py` (near the Task-7 multicolumn tests):

```python
def test_assemble_timeline_crosscolumn_grounds_when_both_ends_observed(tmp_path):
    # The offline stand-in for the P2 hardware proof: when BOTH a col-1 and a
    # col-2 endpoint fire, the cross-column conversation grounds (CrossTrackEdge)
    # and is classified grounded -> no connectivity flag.
    runs = []
    for i in range(3):
        runs.append(_write_run(tmp_path, f"run_{i}", [
            _ev("PERF_CNT_2", 0, col=1, row=2, pkt_type=0),                 # anchor (col 1)
            _ev("PORT_RUNNING_6", 10 + i, col=1, row=1, pkt_type=3),        # memtile (col 1) fires
            _ev("DMA_MM2S_0_START_TASK", 40 + i, col=2, row=4, pkt_type=1), # compute (col 2) DMA fires
        ]))
    configured = ["1|2|0|PERF_CNT_2", "1|1|3|PORT_RUNNING_6",
                  "2|4|1|DMA_MM2S_0_START_TASK"]
    # child = col-2 compute DMA, parent = col-1 memtile port (a cross-column conversation).
    cross_domain_pairs = [("2|4|1|DMA_MM2S_0_START_TASK", "1|1|3|PORT_RUNNING_6")]
    tl = T.assemble_timeline(runs, configured, derives_pairs=set(),
                             cross_domain_pairs=cross_domain_pairs,
                             dump=None, start_col=1)
    # a cross-track edge connects the two columns.
    tile_pairs = {tuple(sorted(("|".join(e.child.split("|")[:2]),
                                "|".join(e.parent.split("|")[:2]))))
                  for e in tl.cross_track_edges}
    assert ("1|1", "2|4") in tile_pairs
    # classified grounded -> NOT flagged as defect or unobserved.
    assert not any("1|1~2|4" in str(f) for f in tl.flags)
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu/tools && python -m pytest test_timeline.py -q -k crosscolumn_grounds`
Expected: PASS. (If it fails because the real `ground_edge` declines this synthetic cross-domain pair, inspect the failure: a cross-domain pair with both endpoints fired should yield a `Gap` edge. Do not weaken the assertion — fix the test inputs so both endpoints genuinely fire under the anchor.)

- [ ] **Step 3: Commit**

```bash
git add tools/test_timeline.py
git commit -m "test(#140): cross-column conversation grounds when both ends observed

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 5: P2 diagnosis spike — why compute memmod (pkt_type 1) trace never reaches the buffer [HW-GATED]

**Files:**
- Create: `build/experiments/two_col_p2_spike/P2-FINDINGS.md` (git-ignored; the deliverable)
- Create: `build/experiments/two_col_p2_spike/probe.py` (git-ignored; the capture+inspect driver) if a fresh capture is needed

**Interfaces:**
- Consumes: the real NPU1 and the two_col trace pipeline (`tools/trace_capture.py`, `trace-patch-events.py`, the decoder).
- Produces: a root-cause finding (one of H1 routing / H2 control / H3 decode) with evidence, and a recommended fix + size estimate, then a decision gate with Maya.

> This task is an investigation, not TDD code. It runs FIRST in execution order if the HW is free, but it is independent of Tasks 1-4. It is controller-run (not a subagent) for HW-safety sequencing, per the SP1 Task-8 precedent.

- [ ] **Step 1: HW safety pre-flight**

Confirm no competing HW suite is running and the device is present:

```bash
ps -eo pid,cmd | grep -E 'emu-bridge|isa-test|test\.exe|run_experiment' | grep -v grep
ls -l /dev/accel/accel0
```
Expected: no competing suite; `/dev/accel/accel0` present. (Per the standing rule: never two HW suites concurrently; do not run `xrt-smi` while a HW test is active.)

- [ ] **Step 2: Confirm the symptom (configured + patched, yet absent)**

Verify on the existing capture that `pkt_type 1` events are configured/patched but absent from the decoded output:

```bash
cd /home/triple/npu-work/xdna-emu
python3 - <<'PY'
import json, glob
# patched? -- patch.json should carry memmod (pkt_type 1) tiles
pj = sorted(glob.glob("build/experiments/two_col_capture/cap/capture_00/run_00/batch_00/hw/patch.json"))
print("patch.json memmod entries:",
      [s for s in json.load(open(pj[0])).get("patch_spec", [])
       if s.get("tile_type") == "memmod"] if pj else "ABSENT")
# decoded? -- any pkt_type 1 in trace.events.json
pts = set()
for f in glob.glob("build/experiments/two_col_capture/cap/**/trace.events.json", recursive=True):
    for e in json.load(open(f)).get("events", []):
        pts.add(e.get("pkt_type"))
print("decoded pkt_types present:", sorted(pts))
PY
```
Expected: memmod entries present in patch.json; `1` absent from decoded pkt_types. This confirms the gap is downstream of configuration.

- [ ] **Step 3: The decisive cut — raw `trace.bin` vs decoded output**

Inspect the raw `trace.bin` (pre-decode) for `pkt_type 1` packets, against the working `pkt_type 3` (memtile) reference. The trace packet header carries the packet type; locate the decoder's packet-type extraction (`tools/trace_capture.py` / the upstream `parse_trace` path) to read the header bits, then scan a `trace.bin` from `build/experiments/two_col_capture/cap/capture_00/run_00/batch_00/hw/trace.bin`. Write the scan into `probe.py` under the spike dir.

- If raw `trace.bin` **contains** pkt-1 packets but decoded output drops them -> **H3 decode-layer** bug. Record which decode step discards them.
- If raw `trace.bin` **contains no** pkt-1 packets -> emission/routing. Distinguish:
  - **H1 routing:** inspect `mlir-aie/build/test/npu-xrt/two_col/traced/aie_traced.mlir` for the compute mem-module trace packet route to the shim collector, versus the memtile's (which works). A missing/again mis-routed compute-memmod trace stream is H1.
  - **H2 control:** check whether the compute memmod `Trace_Control` (start) register is armed in `insts.bin` (the patcher only sets `Trace_Event0/1`; if `Trace_Control` for memmod is never written, the module never traces). Compare against the memtile.

- [ ] **Step 4: Write the findings doc**

Create `build/experiments/two_col_p2_spike/P2-FINDINGS.md` with: the confirmed symptom, the raw-vs-decoded result, the isolated root cause (H1/H2/H3) with file:line evidence, and a recommended fix with a size estimate (e.g. "decode: ~1 fn in parse_trace path"; "routing: recompile two_col traced MLIR with compute-memmod trace route — touches the trace-injection layer, HW-gated"; "control: add Trace_Control patch in trace-patch-events.py").

- [ ] **Step 5: Decision gate with Maya**

Present the root cause and recommended fix. Decide together whether the fix lands in this sub-project or splits to a follow-on. Do not begin an open-ended fix without that decision. Record the decision at the end of `P2-FINDINGS.md`.

- [ ] **Step 6: Commit the spike record (the findings doc is git-ignored; commit a pointer)**

Since `build/experiments/` is git-ignored, record the spike outcome in the ledger and (if the team wants it tracked) copy the key conclusion into `docs/known-fidelity-gaps.md` as a row. Commit only the tracked doc:

```bash
# only if a tracked note is added:
git add docs/known-fidelity-gaps.md
git commit -m "doc(#140): record compute-memmod trace gap (P2 spike finding)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Self-Review

**1. Spec coverage:**
- P1 logical oracle (candidate-pair projection) → Task 1 (classifier) + Task 2 (wiring). ✓
- Three-way classification + flags → Task 1 (all buckets unit-tested) + Task 2 (flag emission). ✓
- Remove physical-hop oracle (`coupling_oracle`/`connectivity_defects`/`_tile_of`) → Task 2. ✓
- New module `connectivity.py` → Task 1. ✓
- Classifier unit tests (all three buckets) → Task 1. ✓
- Real-capture characterization (logical, not physical; cross-column unobserved; 1|0~1|1 grounded; zero defects) → Task 3. ✓
- Synthetic both-sides-observed cross-column grounded → Task 4. ✓
- P2 diagnosis spike + findings doc + decision gate → Task 5. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Every code step shows complete code; every command shows expected output. Task 5 is an investigation (no production code), with concrete commands and a concrete deliverable. ✓

**3. Type consistency:** `classify_connectivity(cross_domain_pairs, fired, edges) -> Dict[Tuple[str,str], str]` and the constants `GROUNDED/OBSERVED_UNGROUNDED/UNOBSERVED` are defined in Task 1 and consumed identically in Task 2. `_tile` (connectivity.py) is distinct from the removed `_tile_of` (timeline.py). Flag strings `connectivity_defect:<a>~<b>` and `connectivity_unobserved:<a>~<b>` are used consistently across Tasks 2-4. ✓
