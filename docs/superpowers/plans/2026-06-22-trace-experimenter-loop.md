# Trace-Experimenter Loop: Active HW Convergence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the verified inference loop (`tools/inference/loop.py`) against real Phoenix (NPU1/AIE2) hardware and demonstrate it converges on a suite of three kernels, replacing the `MockInstrument` with a real-HW adapter and populating the planner's empty reachability self-model.

**Architecture:** A thin `HwInstrument` conforms to the loop's existing duck-typed instrument interface (`ledger_entries() -> List[dict]`, `capture(batch) -> List[str]`) and drives the already-HW-validated capture plumbing (`trace_capture.capture()` / `HwRunner` / `build_active_plan`). A new self-model module enumerates the static configured-event set + orientable candidate pairs from the config dump (reusing `config_extract.generator`), and the loop is extended to fold empirical limits (never-fired events, co-traced-but-uncorrelated pairs) back into the `ReachabilityModel` with measured provenance — which also fixes a latent non-termination on uncorrelated pairs. An entry-point CLI runs a kernel to a terminal state and writes a provenance-complete convergence report.

**Tech Stack:** Python 3.13 (stdlib + numpy, run from `tools/` so `inference.*` / `config_extract.*` import bare); the in-tree decoder (`trace_decoder`), `trace_join`, `trace_variance`; the Rust extractor `examples/dump_config_json.rs` (offline xclbin→JSON); real-NPU capture via `bridge-trace-runner` + `trace_runner.RunnerSession`.

## Global Constraints

- **Run tests from `tools/`** — the inference/config_extract packages import bare (`from inference.X import Y`, `import trace_join`). Invoke pytest as `cd tools && python -m pytest <file> -v`.
- **Column spaces (the start_col reconcile).** Decoder/engine event keys are **absolute** col (`"1|2|0|PERF_CNT_2"`); the patcher consumes **relative** col (add_one's single column is relative col 0, placed at absolute col 1). `build_active_plan` + the patcher take relative col; decoded events land at `traced_col` (absolute). The `label_map` is column-free precisely so it bridges. Any absolute→relative conversion is `col -= start_col`.
- **Anchor.** Cross-run/cross-batch anchoring requires one anchor event present in **every** batch on its tile. `build_active_plan` injects `anchor` (default `PERF_CNT_2`) into `anchor_tile` slot 0 every batch. The loader/verifier anchor key is absolute (`load_fired`'s `ANCHOR = "1|2|0|PERF_CNT_2"`).
- **HW invocation hygiene.** Real-NPU runs must not target the emulator: run capture under `env -u XDNA_EMU` (the bridge then targets the real NPU). Never run two hardware test suites concurrently. Don't run `xrt-smi` while a HW capture is active.
- **No emoji** anywhere. Commit messages end with the two trailer lines (see any recent commit), no pre-approval needed (internal project).
- **HW batches are cheap** (long-lived `RunnerSession`, batch-stdin) — many small batches are expected and fine. Do not artificially minimize batch count.
- **Folds are out of scope** (trace-GROUPS+Z3, per-tile EVENT_PC mode) — they are the immediately-following plans. A kernel that needs them yields an honest **blocked-needs-fold** terminal state, which is a *success* of this plan, not a failure.

---

## File Structure

**Create:**
- `tools/inference/hw_instrument.py` — `HwInstrument` (the real-HW adapter; ledger + capture).
- `tools/inference/selfmodel.py` — static self-model: configured-event enumeration, candidate-pair derivation, batch legality.
- `tools/inference/run_experiment.py` — entry point: kernel → loop → convergence report (+ CLI `__main__`).
- `tools/test_hw_instrument.py` — offline unit tests for the adapter (fake runner; no NPU).
- `tools/test_selfmodel.py` — offline unit tests for enumeration/candidate-pairs/legality (add_one fixture).
- `tools/test_experiment_report.py` — offline tests for the loop's empirical limits + report writer (MockInstrument).
- `tools/test_experiment_loop_hw.py` — Phoenix-gated integration tests (add_one + suite), skipped unless `XDNA_HW_SMOKE=1`.
- `tools/config_extract/fixtures/memtile_dmas.config.json` — generated dump (Task 6).
- `tools/config_extract/fixtures/two_col.config.json` — generated dump (Task 6).

**Modify:**
- `tools/inference/reachability.py` — add `never_fired` single-event constraint query (`unfirable_events()`).
- `tools/inference/loop.py` — fold empirical limits into the model; thread `anchor_key`; return final `run_dirs` + `model` in the result dict.

---

## Task 1: HwInstrument adapter

The loop drives any object exposing `ledger_entries() -> List[dict]` and `capture(batch: Batch) -> List[str]` (see `MockInstrument` in `loop.py:26-78`). Build the real-HW version over the existing capture plumbing. All HW coupling is lazy (imported inside methods) so the module imports clean offline and unit tests can monkeypatch the capture call.

**Files:**
- Create: `tools/inference/hw_instrument.py`
- Test: `tools/test_hw_instrument.py`

**Interfaces:**
- Consumes: `trace_capture.build_active_plan(active, anchor, anchor_tile, slots)`, `trace_capture.capture(plan, runner, *, test, out_dir, traced_col, instr)`, `trace_capture.HwRunner(xclbin, stderr_log)`, `trace_capture._discover_xclbin_insts(test, compiler)`; `config_extract.dump_model.load_dump(path) -> ConfigDump`; `config_extract.generator.generate_ledger(dump, fired_event_keys, start_col) -> dict`; `inference.planner.Batch` (`.tiles: Dict[str,List[str]]`).
- Produces: `HwInstrument(test, dump, configured_events, *, start_col, anchor_tile_abs, anchor_event, traced_col, n_runs, out_root, compiler)` with methods `ledger_entries() -> List[dict]` and `capture(batch) -> List[str]`. Used by `run_experiment` (Task 4) and the HW tests (Task 5/6).

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_hw_instrument.py
"""Offline tests for HwInstrument: Batch->plan translation, col reconcile,
ledger generation. No NPU -- the capture() call is monkeypatched."""
from pathlib import Path
import json
import pytest
from inference.hw_instrument import HwInstrument
from inference.planner import Batch

_FIXTURE = (Path(__file__).resolve().parent
            / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _load_dump():
    from config_extract.dump_model import load_dump
    return load_dump(str(_FIXTURE))


def test_ledger_entries_nonempty_and_oriented():
    dump = _load_dump()
    # configured events in ABSOLUTE col-1 space (the add_one active set).
    configured = ["1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4",
                  "1|2|0|PERF_CNT_2"]
    inst = HwInstrument("add_one_using_dma", dump, configured,
                        start_col=1, anchor_tile_abs="1|2|0",
                        anchor_event="PERF_CNT_2", traced_col=1,
                        n_runs=3, out_root="/tmp/unused", compiler="chess")
    entries = inst.ledger_entries()
    # Every entry is a parent->child route/program fact (a=parent, b=child).
    assert all(set(("a", "b", "kind", "cite")) <= set(e) for e in entries)
    # The memtile buffer relay PR0 -> PR4 must be present (config_path).
    assert any(e["a"] == "1|1|3|PORT_RUNNING_0" and e["b"] == "1|1|3|PORT_RUNNING_4"
               for e in entries)


def test_capture_converts_abs_to_rel_col_and_runs_n_runs(monkeypatch, tmp_path):
    dump = _load_dump()
    inst = HwInstrument("add_one_using_dma", dump,
                        ["1|2|0|PERF_CNT_2"], start_col=1,
                        anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
                        traced_col=1, n_runs=3, out_root=str(tmp_path),
                        compiler="chess")

    seen = {"plans": [], "out_dirs": []}

    # Stub the HW boundary: record the plan + out_dir, write a minimal
    # trace.events.json so load_fired can read it back.
    def fake_capture(plan, runner, *, test, out_dir, traced_col, instr):
        seen["plans"].append(plan)
        seen["out_dirs"].append(str(out_dir))
        bdir = Path(out_dir) / "batch_00" / "hw"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "trace.events.json").write_text(json.dumps(
            {"schema_version": 1, "events": [], "slot_names": {}}))
        return [{}]

    class FakeRunner:
        def __init__(self, *a, **k): pass
        def close(self): pass

    monkeypatch.setattr("inference.hw_instrument.capture", fake_capture)
    monkeypatch.setattr("inference.hw_instrument.HwRunner", FakeRunner)
    # Don't require a built kernel on disk for this offline test.
    monkeypatch.setattr("inference.hw_instrument._discover_xclbin_insts",
                        lambda test, compiler: ("aie.xclbin", "insts.bin"))

    batch = Batch(tiles={"1|2|0": ["INSTR_VECTOR"]})
    run_dirs = inst.capture(batch)

    assert len(run_dirs) == 3                     # n_runs run dirs
    # plan tiles were converted ABS col 1 -> REL col 0 for the patcher.
    plan = seen["plans"][0]
    tiles = {t for b in plan["batches"] for t in b}
    assert all(t.split("|")[0] == "0" for t in tiles), tiles
    # anchor injected on the anchor tile (rel "0|2|0") every batch.
    assert all("0|2|0" in b for b in plan["batches"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest test_hw_instrument.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.hw_instrument'`

- [ ] **Step 3: Implement HwInstrument**

```python
# tools/inference/hw_instrument.py
"""The real-HW adapter: drive the verified loop against Phoenix.

Conforms to the loop's instrument interface (ledger_entries + capture) and
reuses the HW-validated capture plumbing (build_active_plan / capture /
HwRunner). The planner emits batches in ABSOLUTE (decoder) col; the patcher
consumes RELATIVE col -- we subtract start_col on the way in. HW imports are
lazy so this module loads clean offline (tests monkeypatch `capture`/`HwRunner`).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from config_extract.generator import generate_ledger
from inference.planner import Batch
# Imported at module scope so tests can monkeypatch these names directly.
from trace_capture import build_active_plan, capture, HwRunner, _discover_xclbin_insts


class HwInstrument:
    def __init__(self, test, dump, configured_events: List[str], *,
                 start_col: int, anchor_tile_abs: str, anchor_event: str,
                 traced_col: int, n_runs: int, out_root: str,
                 compiler: str = "chess"):
        self._test = test
        self._dump = dump
        self._configured = list(configured_events)
        self._start_col = start_col
        self._anchor_tile_abs = anchor_tile_abs
        self._anchor_event = anchor_event
        self._traced_col = traced_col
        self._n_runs = n_runs
        self._out_root = Path(out_root)
        self._compiler = compiler
        self._iter = 0

    def ledger_entries(self) -> List[dict]:
        # Generate over the full configured set so every orientable pair is
        # present regardless of which events a given batch traced.
        return generate_ledger(self._dump, self._configured,
                               start_col=self._start_col)["entries"]

    def _abs_to_rel(self, tile_key: str) -> str:
        col, row, pkt = tile_key.split("|")
        return f"{int(col) - self._start_col}|{row}|{pkt}"

    def capture(self, batch: Batch) -> List[str]:
        # Convert the planner's ABS-col batch tiles to REL col for the patcher.
        active: Dict[str, set] = {}
        for tile_abs, names in batch.tiles.items():
            active.setdefault(self._abs_to_rel(tile_abs), set()).update(names)
        anchor_tile_rel = self._abs_to_rel(self._anchor_tile_abs)
        # build_active_plan splits to <=8 slots and rides the anchor in slot 0
        # of the anchor tile in every batch.
        plan = build_active_plan(active, anchor=self._anchor_event,
                                 anchor_tile=anchor_tile_rel)
        xclbin, insts = _discover_xclbin_insts(self._test, self._compiler)
        run_dirs: List[str] = []
        base = self._out_root / f"capture_{self._iter:02d}"
        for i in range(self._n_runs):
            rd = base / f"run_{i:02d}"
            rd.mkdir(parents=True, exist_ok=True)
            runner = HwRunner(xclbin, stderr_log=rd / "hw.runner.log")
            try:
                capture(plan, runner, test=self._test, out_dir=rd,
                        traced_col=self._traced_col, instr=insts)
            finally:
                runner.close()
            run_dirs.append(str(rd))
        self._iter += 1
        return run_dirs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest test_hw_instrument.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/inference/hw_instrument.py tools/test_hw_instrument.py
git commit -m "feat(#140): HwInstrument adapter -- drive the verified loop on real HW

$(printf 'Conforms to the loop instrument interface (ledger_entries + capture)\nover the HW-validated capture plumbing; converts planner ABS-col batches\nto patcher REL col; ledger from generate_ledger over the configured set.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 2: Static self-model (configured events, candidate pairs, legality)

The loop needs two static inputs derived from the config dump: the **configured-event set** (termination domain) and the **candidate pairs** (orientable parent→child relationships). Plus a **legality** predicate the planner can use to prove a batch is physically traceable before running. Enumeration is a deliberate over-approximation (a generous per-tile-type dataflow-event menu on active tiles); never-firing entries are pruned empirically in Task 3. This is the documented starter superset — full per-device event tables are follow-up.

**Files:**
- Create: `tools/inference/selfmodel.py`
- Test: `tools/test_selfmodel.py`

**Interfaces:**
- Consumes: `config_extract.dump_model.ConfigDump` (`.route_graph.edges`, each edge with `.src`/`.dst` PortRef carrying `.col`/`.row`); `config_extract.generator.generate_ledger`; `trace_capture.configure_batch` (raises `ValueError` on >8 slots or unknown event).
- Produces:
  - `enumerate_configured_events(dump, start_col) -> List[str]` — absolute-col event keys on active tiles.
  - `candidate_pairs_from_dump(dump, configured_events, start_col) -> List[Tuple[str,str]]` — deduped `(child, parent)` pairs (the engine's candidate order).
  - `legal_batch(batch) -> bool` — True iff `configure_batch` accepts it (≤8 slots/tile, valid event names).

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_selfmodel.py
"""Offline tests for the static self-model against the add_one fixture."""
from pathlib import Path
from inference.selfmodel import (enumerate_configured_events,
                                 candidate_pairs_from_dump, legal_batch)
from inference.planner import Batch

_FIXTURE = (Path(__file__).resolve().parent
            / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _dump():
    from config_extract.dump_model import load_dump
    return load_dump(str(_FIXTURE))


def test_enumeration_superset_of_known_add_one_events():
    evs = set(enumerate_configured_events(_dump(), start_col=1))
    # The known-firing add_one dataflow events must all be enumerated.
    for k in ("1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4",
              "1|0|2|DMA_S2MM_0_START_TASK", "1|0|2|DMA_MM2S_0_START_TASK"):
        assert k in evs, k


def test_candidate_pairs_include_memtile_relay_child_parent_order():
    dump = _dump()
    configured = enumerate_configured_events(dump, start_col=1)
    pairs = candidate_pairs_from_dump(dump, configured, start_col=1)
    # (child, parent): PR4 derives from PR0 (the buffer relay).
    assert ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0") in pairs
    # No duplicates.
    assert len(pairs) == len(set(pairs))


def test_legal_batch_accepts_small_rejects_oversize():
    assert legal_batch(Batch(tiles={"1|2|0": ["PERF_CNT_2", "INSTR_VECTOR"]}))
    # 9 events on one tile -> illegal (>8 slots).
    nine = [f"PORT_RUNNING_{i}" for i in range(8)] + ["PERF_CNT_2"]
    assert not legal_batch(Batch(tiles={"1|1|3": nine}))
    # Bogus event name -> illegal.
    assert not legal_batch(Batch(tiles={"1|2|0": ["NOT_A_REAL_EVENT"]}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest test_selfmodel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.selfmodel'`

- [ ] **Step 3: Implement the self-model**

Note on tile-type by row (NPU1/AIE2 topology, per CLAUDE.md device map): row 0 = shim (pkt 2), row 1 = memtile (pkt 3), rows ≥2 = compute core (pkt 0) + its memmod (pkt 1). The event menu generalizes the hand-derived `SEED_ACTIVE_PLAN` families; over-enumeration is pruned by Task 3.

```python
# tools/inference/selfmodel.py
"""The static reachability self-model: legality + gain inputs from the config.

enumerate_configured_events() is the termination domain (a generous per-tile-type
dataflow-event menu on active tiles -- a deliberate over-approximation pruned
empirically by the loop's never-fired constraints). candidate_pairs_from_dump()
reuses the generator's route-graph reachability as the orientation oracle, so
"gain" is proven from config (never emit-then-discover). legal_batch() defers to
configure_batch's real <=8-slot / valid-name checks.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from config_extract.generator import generate_ledger
from inference.planner import Batch

# Per-tile-type trace-event menu (pkt_type -> event names). Starter superset of
# the dataflow events; row decides tile-type on NPU1 (row0 shim, row1 memtile,
# row>=2 core+memmod). Full per-device event tables are follow-up.
_MENU: Dict[int, List[str]] = {
    2: ["DMA_MM2S_0_START_TASK", "DMA_MM2S_0_FINISHED_TASK",       # shim
        "DMA_S2MM_0_START_TASK", "DMA_S2MM_0_FINISHED_TASK",
        "DMA_S2MM_0_STREAM_STARVATION", "DMA_MM2S_1_START_TASK",
        "DMA_MM2S_1_FINISHED_TASK", "DMA_S2MM_1_START_TASK",
        "DMA_S2MM_1_FINISHED_TASK"],
    3: [f"PORT_RUNNING_{i}" for i in range(8)],                    # memtile
    0: ["PERF_CNT_2", "INSTR_VECTOR", "LOCK_STALL",               # core
        "MEMORY_STALL", "STREAM_STALL"],
    1: ["DMA_MM2S_0_START_TASK", "DMA_S2MM_0_START_TASK",         # memmod
        "EDGE_DETECTION_EVENT_0"],
}


def _pkts_for_row(row: int) -> List[int]:
    if row == 0:
        return [2]
    if row == 1:
        return [3]
    return [0, 1]


def _active_tiles(dump) -> set:
    """(rel_col, row) tiles referenced by the route graph (dump is relative-col)."""
    tiles = set()
    for e in dump.route_graph.edges:
        for node in (e.src, e.dst):
            tiles.add((node.col, node.row))
    return tiles


def enumerate_configured_events(dump, start_col: int) -> List[str]:
    # The dump is RELATIVE-col (generate_ledger subtracts start_col for tile
    # lookup); engine/decoder event keys are ABSOLUTE -- so add start_col here.
    out: List[str] = []
    for (col, row) in sorted(_active_tiles(dump)):
        for pkt in _pkts_for_row(row):
            for name in _MENU.get(pkt, []):
                out.append(f"{col + start_col}|{row}|{pkt}|{name}")
    # Deduplicate preserving order.
    seen, deduped = set(), []
    for k in out:
        if k not in seen:
            seen.add(k); deduped.append(k)
    return deduped


def candidate_pairs_from_dump(dump, configured_events: List[str],
                              start_col: int) -> List[Tuple[str, str]]:
    led = generate_ledger(dump, configured_events, start_col=start_col)
    # Ledger stores a=parent, b=child; the engine's candidate order is (child, parent).
    pairs = [(e["b"], e["a"]) for e in led["entries"]]
    seen, deduped = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p); deduped.append(p)
    return deduped


def legal_batch(batch: Batch) -> bool:
    from trace_capture import configure_batch
    try:
        configure_batch(batch.tiles)
    except (ValueError, KeyError):
        return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest test_selfmodel.py -v`
Expected: PASS (3 passed). If `dump.route_graph.edges` node attribute names differ (`.src`/`.dst` vs `.a`/`.b`), check `config_extract/dump_model.py`'s `RouteEdge` and `Reachability` (it does the same traversal) and match them.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/selfmodel.py tools/test_selfmodel.py
git commit -m "feat(#140): static self-model -- configured events, candidate pairs, legality

$(printf 'enumerate_configured_events (over-approx termination domain on active\ntiles), candidate_pairs_from_dump (reuses generator reachability as the\norientation oracle), legal_batch (defers to configure_batch). The static\nhalf of the hybrid self-model; empirical pruning is Task 3.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 3: Empirical limits in the loop (and fix uncorrelated non-termination)

Today the loop never calls `model.add_constraint`, and `correlates()` returns `None` for *both* "never co-traced" and "co-traced but std>eps" — so the planner re-proposes an uncorrelated pair every iteration and the loop spins to `max_iters` (`converged=False` with no explanation). This task folds two empirical limits into the `ReachabilityModel` with measured provenance, which both makes the self-model honest and lets the loop terminate cleanly:

1. **never-fired:** after the Phase-0 seed (which sweeps the whole configured set), any configured event that did not fire is constrained `never_fired` and excluded from the unfired set thereafter.
2. **uncorrelated:** after a co-trace batch, if the pair is now co-traced (`pair_derivability` returns Stats) but `std > eps`, constrain it `cannot_correlate` so `propose_next` returns `NO_GAIN` next time.

Also thread `anchor_key` through the loop (today hardwired to the add_one default via `load_fired`/`propose_next`) and return the final `run_dirs` + `model` so the entry point can build a complete report.

**Files:**
- Modify: `tools/inference/reachability.py`
- Modify: `tools/inference/loop.py`
- Test: `tools/test_experiment_report.py` (the loop half; report writer added in Task 4)

**Interfaces:**
- Consumes: `trace_join.pair_derivability(run_dirs, a, b, anchor_key) -> Stats|None` (Stats has `.std`); `inference.verifier.EPS`, `inference.reachability.Constraint`.
- Produces: `ReachabilityModel.unfirable_events() -> set[str]`; `run_loop_until_converged(instrument, configured_events, candidate_pairs, *, anchor_key=ANCHOR, max_iters=50)` now returns a dict additionally containing `"run_dirs": List[str]`, `"model": ReachabilityModel`, and `"terminal_state": "placed"|"halted_falsifiable"|"halted_unexplained"`.

- [ ] **Step 1: Write the failing tests**

```python
# tools/test_experiment_report.py
"""The loop's empirical-limit folding, exercised with MockInstrument ground
truth (no NPU). An uncorrelated pair must HALT honestly (not spin to max_iters);
a never-fired configured event must be constrained and not block convergence."""
from inference.loop import MockInstrument, run_loop_until_converged
from inference.verifier import ANCHOR


def _gt(workdir, *, uncorrelated=False, with_unfired=False):
    # Two memtile ports + the anchor on the core tile. PR0 -> PR4 is a route.
    events = {
        ANCHOR: {"base": 0, "jitter": 0},
        "1|1|3|PORT_RUNNING_0": {"base": 10, "jitter": 0},
        "1|1|3|PORT_RUNNING_4": {"base": 40, "jitter": 1 if uncorrelated else 0},
    }
    gt = {"events": events, "routes": [("1|1|3|PORT_RUNNING_0",
                                        "1|1|3|PORT_RUNNING_4")],
          "workdir": str(workdir)}
    return gt


def test_correlated_pair_converges(tmp_path):
    inst = MockInstrument(_gt(tmp_path), n_runs=6)
    configured = [ANCHOR, "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4"]
    pairs = [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")]
    res = run_loop_until_converged(inst, configured, pairs)
    assert res["converged"] is True
    assert res["terminal_state"] == "placed"


def test_uncorrelated_pair_halts_falsifiably_not_spins(tmp_path):
    inst = MockInstrument(_gt(tmp_path, uncorrelated=True), n_runs=6)
    configured = [ANCHOR, "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4"]
    pairs = [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")]
    res = run_loop_until_converged(inst, configured, pairs, max_iters=12)
    # It must NOT exhaust max_iters spinning; a cannot_correlate constraint
    # with measured provenance must be recorded.
    assert res["iterations"] < 12
    assert res["terminal_state"] in ("placed", "halted_falsifiable")
    constraints = res["model"]._constraints
    assert any(c.predicate == "cannot_correlate"
               and c.provenance_batch is not None for c in constraints)


def test_never_fired_event_is_constrained_and_excluded(tmp_path):
    gt = _gt(tmp_path)
    # A configured event that is NOT in the ground-truth event set -> never fires.
    configured = [ANCHOR, "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4",
                  "1|1|3|PORT_RUNNING_7"]
    pairs = [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")]
    inst = MockInstrument(gt, n_runs=6)
    res = run_loop_until_converged(inst, configured, pairs, max_iters=12)
    # PR7 never fires; it must be constrained never_fired (with provenance) and
    # not prevent convergence on the rest.
    assert "1|1|3|PORT_RUNNING_7" in res["model"].unfirable_events()
    assert res["converged"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools && python -m pytest test_experiment_report.py -v`
Expected: FAIL — `test_uncorrelated_pair_halts_falsifiably_not_spins` (no `cannot_correlate` constraint; likely `iterations == 12`) and `test_never_fired_event_is_constrained_and_excluded` (`unfirable_events` AttributeError / `KeyError: 'terminal_state'`).

- [ ] **Step 3a: Add the never_fired query to ReachabilityModel**

In `tools/inference/reachability.py`, add this method to `ReachabilityModel` (after `blocking_constraints`):

```python
    def unfirable_events(self) -> set:
        """Single events with a discharged never_fired constraint (args=(event,))."""
        return {c.args[0] for c in self._constraints
                if c.predicate == "never_fired" and c.provenance_batch is not None
                and len(c.args) == 1}
```

- [ ] **Step 3b: Fold empirical limits into the loop**

Rewrite `tools/inference/loop.py`'s `run_loop_until_converged` (keep `MockInstrument` and `ranking` unchanged). The diff: import `pair_derivability` + `Constraint` + `EPS`; thread `anchor_key`; after the seed, constrain never-fired events; after each co-trace capture, constrain uncorrelated pairs; exclude `unfirable_events()` from the unfired set; return `run_dirs`/`model`/`terminal_state`.

```python
# tools/inference/loop.py -- replace the imports block and run_loop_until_converged.

# add to the existing imports near the top:
from inference.reachability import ReachabilityModel, Constraint
from inference.verifier import ANCHOR, EPS
import trace_join


def run_loop_until_converged(instrument, configured_events: List[str],
                             candidate_pairs: List[Tuple[str, str]],
                             *, anchor_key: str = ANCHOR,
                             max_iters: int = 50) -> dict:
    kb = KB.empty()
    install_ledger(kb, {e["cite"]: e for e in instrument.ledger_entries()})
    model = ReachabilityModel()
    all_run_dirs: List[str] = []
    rankings: List[Tuple[int, int, int]] = []

    # Phase 0: seed sweep over the static configured-event set.
    seed = seed_plan(configured_events)
    seed_dirs = instrument.capture(seed)
    all_run_dirs += seed_dirs

    # Empirical limit (never-fired): the seed traced everything once; anything
    # that did not fire is unfirable -- constrain it with the seed as provenance.
    seeded_fired = {f.args[0] for f in load_fired(all_run_dirs, anchor_key)}
    for ev in configured_events:
        if ev not in seeded_fired and ev not in model.unfirable_events():
            model.add_constraint(Constraint(
                name=f"never_fired:{ev}", predicate="never_fired",
                args=(ev,), provenance_batch=seed_dirs[0]))

    def live_unfired(fired_set):
        unfirable = model.unfirable_events()
        return [e for e in configured_events
                if e not in fired_set and e not in unfirable]

    prev = None
    for _ in range(max_iters):
        kb_iter = KB.empty()
        install_ledger(kb_iter, {e["cite"]: e for e in instrument.ledger_entries()})
        for f in load_fired(all_run_dirs, anchor_key):
            kb_iter.add(f)
        kb_iter = chain(all_run_dirs, kb_iter, candidate_pairs)
        kb = kb_iter

        r = ranking(kb, configured_events, candidate_pairs)
        rankings.append(r)
        if prev is not None and r > prev:
            raise RuntimeError(f"ranking increased {prev} -> {r} (livelock)")
        prev = r

        fired = {f.args[0] for f in kb.by_predicate("fired")}
        cls = classify_events(kb, sorted(fired))
        unresolved = [e for e, c in cls.items() if c == "unresolved"]
        unfired = live_unfired(fired)
        if not unresolved and not unfired:
            return {"converged": True, "iterations": len(rankings),
                    "rankings": rankings, "classification": cls,
                    "run_dirs": all_run_dirs, "model": model,
                    "terminal_state": "placed"}

        # Act on the stall: propose the next measurement (proven gain only).
        progressed = False
        for pair in candidate_pairs:
            batch = propose_next(kb, all_run_dirs, pair, model, anchor_key)
            if batch is not NO_GAIN:
                new_dirs = instrument.capture(batch)
                all_run_dirs += new_dirs
                # Empirical limit (uncorrelated): co-traced now but std>eps ->
                # no stable offset, no derivation. Constrain so we don't re-propose.
                a, b = pair
                st = trace_join.pair_derivability(all_run_dirs, a, b, anchor_key)
                if st is not None and st.std > EPS:
                    model.add_constraint(Constraint(
                        name=f"uncorrelated:{a}:{b}", predicate="cannot_correlate",
                        args=(a, b), provenance_batch=new_dirs[0]))
                progressed = True
                break
        if not progressed and unfired:
            all_run_dirs += instrument.capture(seed_plan(unfired))
            progressed = True
        if not progressed:
            # Halt: distinguish a falsifiable halt (we recorded WHY) from an
            # unexplained one (a bug signal -- unresolved with no constraint).
            state = "halted_falsifiable" if model._constraints else "halted_unexplained"
            return {"converged": False, "iterations": len(rankings),
                    "rankings": rankings, "classification": cls,
                    "run_dirs": all_run_dirs, "model": model,
                    "terminal_state": state}

    return {"converged": False, "iterations": len(rankings),
            "rankings": rankings, "classification": cls,
            "run_dirs": all_run_dirs, "model": model,
            "terminal_state": "halted_unexplained"}
```

Note: `propose_next(kb, run_dirs, pair, model, anchor_key)` is called positionally — its signature is `propose_next(kb, run_dirs, pair, model, anchor_key=ANCHOR, eps=EPS)`, so this matches.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools && python -m pytest test_experiment_report.py -v`
Expected: PASS (3 passed). Then run the existing loop tests to confirm no regression: `cd tools && python -m pytest -k "loop or planner or reachab" -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/inference/reachability.py tools/inference/loop.py tools/test_experiment_report.py
git commit -m "feat(#140): fold empirical limits into the loop (never-fired, uncorrelated)

$(printf 'After the seed, constrain configured events that never fired; after a\nco-trace, constrain co-traced-but-uncorrelated pairs (std>eps) -- both with\nmeasured provenance. Fixes a latent non-termination where an uncorrelated\npair was re-proposed every iter. Thread anchor_key; return run_dirs/model/\nterminal_state for the convergence report.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 4: Entry point + convergence report

Tie it together: a `run_experiment(kernel_cfg, instrument=None)` that loads the dump, derives the configured-event set + candidate pairs + anchor, builds the `HwInstrument` (unless one is injected for tests), runs the loop, and writes a provenance-complete convergence report. Plus a `__main__` CLI.

**Files:**
- Create: `tools/inference/run_experiment.py`
- Test: `tools/test_experiment_report.py` (extend with report-writer tests; MockInstrument injected)

**Interfaces:**
- Consumes: `inference.selfmodel.enumerate_configured_events`, `candidate_pairs_from_dump`; `inference.loop.run_loop_until_converged`; `inference.engine.run_engine(run_dirs, ledger_path, candidate_pairs) -> report` (for the rich placement backbone); `config_extract.dump_model.load_dump`.
- Produces: `KernelConfig` dataclass (`test, compiler, dump_path, start_col, anchor_tile_abs, anchor_event, traced_col, n_runs, out_root`); `run_experiment(cfg, instrument=None) -> dict`; `write_report(report, path)`.

- [ ] **Step 1: Write the failing tests** (append to `tools/test_experiment_report.py`)

```python
def test_run_experiment_with_mock_writes_report(tmp_path):
    from inference.run_experiment import KernelConfig, run_experiment, write_report
    from inference.loop import MockInstrument
    from inference.verifier import ANCHOR
    import json

    gt = _gt(tmp_path)
    inst = MockInstrument(gt, n_runs=6)
    cfg = KernelConfig(test="add_one_using_dma", compiler="chess",
                       dump_path=None, start_col=1, anchor_tile_abs="1|2|0",
                       anchor_event="PERF_CNT_2", traced_col=1, n_runs=6,
                       out_root=str(tmp_path / "out"))
    # Inject configured/pairs directly via the mock-test override hook.
    report = run_experiment(cfg, instrument=inst,
                            configured=[ANCHOR, "1|1|3|PORT_RUNNING_0",
                                        "1|1|3|PORT_RUNNING_4"],
                            candidate_pairs=[("1|1|3|PORT_RUNNING_4",
                                              "1|1|3|PORT_RUNNING_0")])
    assert report["terminal_state"] == "placed"
    assert "classification" in report and "constraints" in report
    out = tmp_path / "report.json"
    write_report(report, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["kernel"] == "add_one_using_dma"
    # Every recorded constraint carries its provenance batch (falsifiability).
    assert all(c["provenance_batch"] for c in loaded["constraints"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tools && python -m pytest test_experiment_report.py::test_run_experiment_with_mock_writes_report -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.run_experiment'`

- [ ] **Step 3: Implement the entry point**

```python
# tools/inference/run_experiment.py
"""Entry point: run a kernel through the active loop to a terminal state and
write a provenance-complete convergence report.

CLI:
    cd tools && env -u XDNA_EMU python -m inference.run_experiment \\
        --test add_one_using_dma --dump config_extract/fixtures/add_one_using_dma.config.json \\
        --start-col 1 --out ../build/experiments/exp-loop/add_one
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class KernelConfig:
    test: str
    compiler: str
    dump_path: Optional[str]
    start_col: int
    anchor_tile_abs: str
    anchor_event: str
    traced_col: int
    n_runs: int
    out_root: str


def run_experiment(cfg: KernelConfig, instrument=None,
                   configured: Optional[List[str]] = None,
                   candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> dict:
    from config_extract.dump_model import load_dump
    from inference.selfmodel import (enumerate_configured_events,
                                     candidate_pairs_from_dump)
    from inference.loop import run_loop_until_converged
    from inference.hw_instrument import HwInstrument

    anchor_key = f"{cfg.anchor_tile_abs}|{cfg.anchor_event}"

    dump = load_dump(cfg.dump_path) if cfg.dump_path else None
    if configured is None:
        configured = enumerate_configured_events(dump, cfg.start_col)
    if candidate_pairs is None:
        candidate_pairs = candidate_pairs_from_dump(dump, configured, cfg.start_col)

    if instrument is None:
        instrument = HwInstrument(
            cfg.test, dump, configured, start_col=cfg.start_col,
            anchor_tile_abs=cfg.anchor_tile_abs, anchor_event=cfg.anchor_event,
            traced_col=cfg.traced_col, n_runs=cfg.n_runs,
            out_root=cfg.out_root, compiler=cfg.compiler)

    res = run_loop_until_converged(instrument, configured, candidate_pairs,
                                   anchor_key=anchor_key)

    # Rich placement backbone from the engine over the final run dirs.
    derives, roots, provenance_ok = [], [], None
    try:
        from inference.engine import run_engine
        led = {"entries": instrument.ledger_entries()}
        ledger_path = Path(cfg.out_root) / "ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(led))
        rep = run_engine(res["run_dirs"], str(ledger_path), candidate_pairs)
        derives = rep.get("derives", [])
        roots = rep.get("stochastic_roots", [])
        provenance_ok = rep.get("provenance_ok")
    except Exception as exc:  # engine report is best-effort; loop result stands.
        provenance_ok = f"engine_report_error: {exc}"

    return {
        "kernel": cfg.test,
        "converged": res["converged"],
        "terminal_state": res["terminal_state"],
        "iterations": res["iterations"],
        "classification": res["classification"],
        "derives": derives,
        "stochastic_roots": roots,
        "provenance_ok": provenance_ok,
        "constraints": [
            {"name": c.name, "predicate": c.predicate, "args": list(c.args),
             "provenance_batch": c.provenance_batch}
            for c in res["model"]._constraints],
        "config": asdict(cfg),
    }


def write_report(report: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--compiler", default="chess")
    ap.add_argument("--start-col", type=int, default=1)
    ap.add_argument("--anchor-tile", default="1|2|0")
    ap.add_argument("--anchor-event", default="PERF_CNT_2")
    ap.add_argument("--traced-col", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = KernelConfig(test=a.test, compiler=a.compiler, dump_path=a.dump,
                       start_col=a.start_col, anchor_tile_abs=a.anchor_tile,
                       anchor_event=a.anchor_event, traced_col=a.traced_col,
                       n_runs=a.n_runs, out_root=a.out)
    report = run_experiment(cfg)
    out_path = str(Path(a.out) / "convergence_report.json")
    write_report(report, out_path)
    print(f"[run_experiment] {a.test}: {report['terminal_state']} "
          f"({report['iterations']} iters); report -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd tools && python -m pytest test_experiment_report.py -v`
Expected: PASS (all, including the new report-writer test).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/run_experiment.py tools/test_experiment_report.py
git commit -m "feat(#140): experiment entry point + provenance-complete convergence report

$(printf 'run_experiment ties dump -> self-model -> loop -> report; builds the\nHwInstrument (or takes an injected one for offline tests); report carries\nthe placement backbone, terminal_state, and every empirical constraint\nwith its provenance batch. CLI under -m inference.run_experiment.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 5: HW convergence on add_one (Phoenix-gated, falsifiable)

The first real-hardware close of the loop. Gated behind `XDNA_HW_SMOKE=1` (like `test_inference_hw_smoke.py`). Three assertions enforce the "loads ≠ derives" discipline at the loop level: it converges, it places at least one event that config alone could not orient (a `program_path` through-core derive — proof it consumed HW timing evidence), and a forced-wrong batch changes the outcome (the oracle is genuine).

**Files:**
- Create: `tools/test_experiment_loop_hw.py`

**Interfaces:**
- Consumes: `inference.run_experiment.{KernelConfig, run_experiment}`; the add_one fixture dump; real NPU via `HwInstrument`.

- [ ] **Step 1: Write the gated integration test**

```python
# tools/test_experiment_loop_hw.py
"""Phoenix-gated: the active loop closes on real NPU1 and converges.

    cd tools && XDNA_HW_SMOKE=1 env -u XDNA_EMU \\
      python -m pytest test_experiment_loop_hw.py -v -k add_one

Requires a built kernel under mlir-aie/build/test/npu-xrt/<test>/chess/.
"""
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XDNA_HW_SMOKE") != "1",
    reason="HW loop test requires a real NPU; set XDNA_HW_SMOKE=1")

_FIX = (Path(__file__).resolve().parent
        / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _cfg(tmp_path):
    from inference.run_experiment import KernelConfig
    return KernelConfig(test="add_one_using_dma", compiler="chess",
                        dump_path=str(_FIX), start_col=1,
                        anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
                        traced_col=1, n_runs=6,
                        out_root=str(tmp_path / "add_one"))


def test_loop_converges_on_add_one_hw(tmp_path):
    from inference.run_experiment import run_experiment
    rep = run_experiment(_cfg(tmp_path))
    assert rep["terminal_state"] in ("placed", "halted_falsifiable")
    # Every recorded constraint is falsifiable (carries its provenance batch).
    assert all(c["provenance_batch"] for c in rep["constraints"])


def test_loop_places_a_through_core_event_hw(tmp_path):
    # The through-core (program_path) pair S2MM_0_START <- MM2S_0_START is
    # orientable ONLY via the core_lock_relay edge -- config alone cannot place
    # it. Its presence in derives proves the loop used HW timing evidence.
    from inference.run_experiment import run_experiment
    rep = run_experiment(_cfg(tmp_path))
    derived_children = {d[0] for d in rep["derives"]}
    assert "1|0|2|DMA_S2MM_0_START_TASK" in derived_children, rep["derives"]


def test_forced_wrong_batch_changes_outcome_hw(tmp_path):
    # Falsifiability: if we corrupt the measured offset (shuffle one event's
    # timestamps across runs so std>>eps), the engine must REJECT that derive.
    # We run normally, then re-run the engine on a perturbed copy of the run
    # dirs and assert the perturbed pair is no longer derived.
    import json, shutil
    from inference.run_experiment import run_experiment, KernelConfig
    from inference.engine import run_engine

    cfg = _cfg(tmp_path)
    rep = run_experiment(cfg)
    base_children = {d[0] for d in rep["derives"]}
    assert base_children, "nothing derived; cannot test falsifiability"

    # Perturb: copy run dirs, scramble PORT_RUNNING_4's ts in one run so the
    # PR4<-PR0 offset becomes unstable.
    target = "PORT_RUNNING_4"
    pert = Path(cfg.out_root) / "perturbed"
    run_dirs = []
    for rd in sorted(p for p in Path(cfg.out_root).glob("capture_*/run_*")):
        dst = pert / rd.relative_to(cfg.out_root)
        shutil.copytree(rd, dst)
        for ev_path in dst.glob("batch_*/hw/trace.events.json"):
            doc = json.loads(ev_path.read_text())
            for e in doc["events"]:
                if e["name"] == target:
                    e["ts"] += 9999; e["soc"] += 9999  # break the offset
            ev_path.write_text(json.dumps(doc))
        run_dirs.append(str(dst))

    led = Path(cfg.out_root) / "ledger.json"   # written by run_experiment
    perturbed = run_engine(run_dirs, str(led),
                           [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")])
    perturbed_children = {d[0] for d in perturbed["derives"]}
    assert "1|1|3|PORT_RUNNING_4" not in perturbed_children
```

- [ ] **Step 2: Build the kernel + verify the gate**

Confirm the chess build exists (or build it), then confirm the test is collected and skips without the env:

Run: `ls /home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie.xclbin`
Run: `cd tools && python -m pytest test_experiment_loop_hw.py -v`
Expected: 3 skipped (gate off).

- [ ] **Step 3: Run on hardware**

Run (NPU; not concurrent with any other HW suite): `cd tools && XDNA_HW_SMOKE=1 env -u XDNA_EMU python -m pytest test_experiment_loop_hw.py -v -k add_one 2>&1 | tee /tmp/exp-loop-add_one.log`
Expected: 3 passed. (Redirect to a file per the no-pipe-to-tail rule for long runs; this `tee` is acceptable for a backgrounded/logged run. If it wedges, recover the NPU per the operations chain before retrying.)

- [ ] **Step 4: Commit**

```bash
git add tools/test_experiment_loop_hw.py
git commit -m "test(#140): HW convergence on add_one -- loop closes on real NPU1

$(printf 'Phoenix-gated falsifiable integration test: the active loop converges;\nplaces the through-core (program_path) S2MM<-MM2S pair that config alone\ncannot orient (proof it used HW timing); a forced-unstable offset is\nrejected by the engine (genuine oracle).\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Task 6: Suite convergence (memtile_dmas + two_col)

Generate config dumps for the two additional kernels (offline — `dump_config_json` is pure xclbin/CDO parsing, no NPU) and prove the loop reaches a defined terminal state on each. `memtile_dmas` exercises config_path-only convergence (no through-core); `two_col` exercises cross-column reachability. A kernel that needs a fold yields `halted_falsifiable` with a `blocked-needs-fold`-style constraint — that is a pass for this plan.

**Files:**
- Create: `tools/config_extract/fixtures/memtile_dmas.config.json` (generated)
- Create: `tools/config_extract/fixtures/two_col.config.json` (generated)
- Modify: `tools/test_experiment_loop_hw.py` (add the two suite tests)

**Interfaces:**
- Consumes: `examples/dump_config_json.rs` (Rust); `inference.run_experiment`.

- [ ] **Step 1: Generate the two config dumps (offline)**

Build the kernels if needed, then dump. `dump_config_json` takes `<aie.xclbin> [insts.bin]` and prints pretty JSON to stdout:

Run (bare; no NPU needed):
```bash
cd /home/triple/npu-work/xdna-emu
BD=/home/triple/npu-work/mlir-aie/build/test/npu-xrt
cargo run --release --example dump_config_json -- \
  $BD/memtile_dmas/chess/aie.xclbin $BD/memtile_dmas/chess/insts.bin \
  > tools/config_extract/fixtures/memtile_dmas.config.json
cargo run --release --example dump_config_json -- \
  $BD/two_col/chess/aie.xclbin $BD/two_col/chess/insts.bin \
  > tools/config_extract/fixtures/two_col.config.json
```
Expected: two non-empty JSON files. Verify each parses + has edges:
Run: `cd tools && python -c "from config_extract.dump_model import load_dump; d=load_dump('config_extract/fixtures/two_col.config.json'); print(len(d.route_graph.edges),'edges')"`
Expected: a positive edge count for each. (If a kernel's build dir or insts.bin name differs, locate it: `ls $BD/<kernel>/chess/`.)

- [ ] **Step 2: Determine each kernel's start_col / anchor (offline)**

The anchor must be `PERF_CNT_2` on a tile traced in every batch; `start_col` is the absolute placement of the kernel's relative col 0. Inspect the dump's tiles to choose an anchor tile (a core tile, pkt 0, that the kernel uses) and confirm the absolute column:

Run: `cd tools && python -c "from config_extract.dump_model import load_dump; d=load_dump('config_extract/fixtures/two_col.config.json'); print(sorted({(n.col,n.row) for e in d.route_graph.edges for n in (e.src,e.dst)}))"`
Expected: a list of (col,row). Record, for each kernel: `start_col` (min col), `anchor_tile_abs` (a `col|row|0` core tile present), `traced_col`. Note these in the test (Step 3). For a multi-column kernel like `two_col`, `traced_col` and the anchor must be consistent — pick the column the anchor tile lives in; cross-column events on the *other* column will surface as `never_fired` under single-column tracing and be constrained honestly (a correct blocked-style limit, not a bug). Document that in the test docstring.

- [ ] **Step 3: Add the suite tests** (append to `tools/test_experiment_loop_hw.py`)

```python
# Fill start_col / anchor / traced_col from Step 2's inspection before running.
_SUITE = {
    "memtile_dmas": dict(start_col=1, anchor_tile_abs="1|2|0", traced_col=1),
    "two_col":      dict(start_col=0, anchor_tile_abs="0|2|0", traced_col=0),
}


@pytest.mark.parametrize("kernel", sorted(_SUITE))
def test_suite_reaches_terminal_state_hw(kernel, tmp_path):
    from inference.run_experiment import KernelConfig, run_experiment
    p = _SUITE[kernel]
    fix = (Path(__file__).resolve().parent / "config_extract" / "fixtures"
           / f"{kernel}.config.json")
    cfg = KernelConfig(test=kernel, compiler="chess", dump_path=str(fix),
                       start_col=p["start_col"],
                       anchor_tile_abs=p["anchor_tile_abs"],
                       anchor_event="PERF_CNT_2", traced_col=p["traced_col"],
                       n_runs=6, out_root=str(tmp_path / kernel))
    rep = run_experiment(cfg)
    # A defined terminal state (placed, or an honest falsifiable halt) -- never
    # the unexplained-halt bug signal.
    assert rep["terminal_state"] in ("placed", "halted_falsifiable"), rep
    # Report is provenance-complete: every constraint cites the batch that set it.
    assert all(c["provenance_batch"] for c in rep["constraints"])
```

- [ ] **Step 4: Run the suite on hardware**

Run (NPU; serial, not concurrent with other HW suites): `cd tools && XDNA_HW_SMOKE=1 env -u XDNA_EMU python -m pytest test_experiment_loop_hw.py -v -k suite 2>&1 | tee /tmp/exp-loop-suite.log`
Expected: 2 passed (one per kernel). Inspect each `convergence_report.json` terminal_state; if a kernel is `halted_falsifiable`, confirm the constraints explain why (never_fired / uncorrelated), which is the expected honest outcome.

- [ ] **Step 5: Commit**

```bash
git add tools/config_extract/fixtures/memtile_dmas.config.json \
        tools/config_extract/fixtures/two_col.config.json \
        tools/test_experiment_loop_hw.py
git commit -m "test(#140): suite convergence -- memtile_dmas + two_col on real NPU1

$(printf 'Generated config dumps (offline CDO parse) for two topologically-distinct\nkernels; the active loop reaches a defined terminal state on each (placed\nor honest falsifiable halt), report provenance-complete. Closes the\nexperimenter-loop suite-to-convergence bar.\n\nGenerated using Claude Code.\nClaude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh')"
```

---

## Final regression

After all tasks, confirm nothing offline regressed (run from `tools/`):

Run: `cd tools && python -m pytest -q 2>&1 | tee /tmp/exp-loop-final.log`
Expected: all non-HW tests pass; HW tests skipped without `XDNA_HW_SMOKE=1`.

Then the Rust suite (the dump example shares the crate): `cargo test --lib` — Expected: pass (unchanged; this plan adds no Rust).

---

## Notes for the executor

- **Deferred, immediately-following plans** (do NOT build here): trace-GROUPS + Z3 groups phase; per-tile EVENT_PC mode threading. If a suite kernel can only progress with one of these, the loop's honest terminal state (`halted_falsifiable` with a never_fired/uncorrelated constraint, or an explicit blocked finding) is the correct result and the driver for the next plan.
- **Column-space bugs are the most likely failure.** When an HW capture returns zero events or "foreign column" errors, re-check the abs/rel reconcile (Task 1 `_abs_to_rel`, the `traced_col`, and the anchor key's absolute col) before suspecting anything else.
- **`pair_derivability` return shape:** confirm `.std` is the attribute name in `tools/trace_join.py` (the Stats namedtuple) when wiring Task 3; the verifier already consumes it, so match the verifier's usage.
- **dump_model attribute names:** Task 2 assumes `RouteEdge.src`/`.dst` with `.col`/`.row`. If they differ, mirror whatever `config_extract/reachability.py` uses for the same traversal.
