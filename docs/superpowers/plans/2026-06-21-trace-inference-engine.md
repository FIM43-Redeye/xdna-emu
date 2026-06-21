# Trace Inference Engine — Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 trace inference engine — a self-verifying forward chainer that
reconstructs the *placement* of every configured trace event relative to the anchor,
drives the next measurement when the chain stalls, and proves degeneracy when it
cannot — fully test-driven against synthetic ground truth (Axis 1), with
`config_path` consumed as an input structural ledger.

**Architecture:** A Python package `tools/inference/` layered as the spec's stack:
a measured-leaf loader and empirical verifier (seeded by the existing
`trace_variance`/`trace_join` cross-run machinery), a structural-ledger loader, a set
of verified derivation rules, a forward chainer to fixpoint, a union-find degeneracy
reasoner with SCC condensation behind a two-method interface, a verified reachability
self-model, a proven-gain planner, and a closed loop with three-component
lexicographic termination. A prerequisite refactor extracts `RunnerSession`/
`ParseSession` out of `trace-sweep.py` into a shared `tools/trace_runner.py` so the
actuator can be driven without the sweep orchestration.

**Tech Stack:** Python 3.13, pytest (flat colocated `tools/test_*.py` convention,
`tools/conftest.py` already on path), `trace_variance.py`/`trace_join.py` (cross-run
std machinery), `trace_capture.py` (capture engine / actuator), no third-party deps
beyond what `tools/` already uses. **Z3 is NOT used in Plan 1** (union-find suffices
for the pure-equality v1 model; Z3 arrives in Plan 3 with groups).

## Global Constraints

- **Spec is authoritative:** `docs/superpowers/specs/2026-06-21-trace-inference-engine-design.md`.
  Every task implements a part of it; the review archive
  (`...-review.md`) explains why each decision was made.
- **Placement, not causation.** `derives(child, parent, offset)` means
  *dataflow-upstream + stable offset = placement*; timing-causation is explicitly
  disclaimed and must never be asserted (a `STREAM_STARVATION` event fires from
  downstream backpressure — its config orientation opposes its timing cause, yet it
  must still be *placed* correctly).
- **No unaudited axioms.** Every fact has support `measured` | `structural(cite)` |
  `derived(rule, premises)`. Every rule is a hypothesis verified against measured data
  before it may derive truth; a rejected rule is a recorded finding. Orientation and
  identity are **derived** from `structural` config premises by stated, verified rules
  — never primitive assertions.
- **eps default = `2.0`** (matches `trace_variance.classify` and
  `trace_join.build_derivability_graph`). v1 uses a hard global eps; the per-pair
  statistical test is a documented open question, not Plan-1 scope.
- **anchor_key = `"1|2|0|PERF_CNT_2"`** (`col|row|pkt|name`). Anchored timestamps come
  from `trace_join.anchored_firsts` (first-fire minus anchor first-fire, over the
  `soc` field). **event_key is the 4-part string `col|row|pkt|name`** everywhere.
- **Reuse, don't reinvent:** the `correlates` verifier is seeded by
  `trace_join.pair_derivability` (computes `std(a.ts − b.ts)`); the `deterministic`
  verifier by `trace_variance.aggregate` + `classify`; the `fired` loader by
  `trace_join.batch_firsts`/`_batch_names`. Do **not** copy `trace_variance.check_span_law`
  (a hardcoded `==64` span-sum check — unrelated to placement).
- **The derivability graph is a reference algorithm, not a seed.** Its roots are
  greedy/order-dependent/unverified and it discards per-run `fired` atoms — but those
  atoms survive on disk in `batch_NN/hw/trace.events.json`. Re-emit `fired` and
  re-derive edges under the verified rule; keep the algorithm as reference only.
- **Tests live at `tools/test_inference_<module>.py`** (flat, colocated). Modules
  import as `from inference.X import Y` (requires `tools/inference/__init__.py`;
  `tools/` is on the pytest path via existing rootdir + conftest).
- **Commits:** conventional-commit subject lines shown per task; the executing agent
  appends this session's standard commit trailer. No emoji. Commit after every task.
- **Run tests bare**, never piped through `grep`/`head`/`tail`. No HW is required for
  Tasks 1–13 (synthetic Axis-1). Task 14 is HW-gated and explicitly marked.
- **Out of Plan 1 (do not build here):** automated `config_path` extraction from real
  binaries (Plan 2: a Rust `examples/dump_config_json.rs` + the derivation rule over
  it, then full Axis-2 HW validation); the trace GROUPS actuator phase and Z3
  (Plan 3); retiring `trace-sweep.py` by the parity gate (post-Plan-1).

---

## File Structure

**Prerequisite refactor:**
- Create `tools/trace_runner.py` — extracted `RunnerSession`, `ParseSession`,
  `RunResult`, `_run_one_side`, patch/parse helpers, and the constants they need.
- Modify `tools/trace-sweep.py` — import the moved symbols from `trace_runner`.
- Modify `tools/trace_capture.py` — `HwRunner` imports `RunnerSession` from
  `trace_runner` instead of dynamically loading `trace-sweep.py`.

**New package `tools/inference/`** (one responsibility per module):
- `__init__.py` — package marker + version constant.
- `facts.py` — `Support` (`Measured`/`Structural`/`Derived`), `Fact`, `KB`,
  provenance-leaf walk, the no-unaudited-axioms property.
- `loader.py` — measured leaves: load `fired(event_key, run, anchored_ts)` from
  `batch_NN/hw/trace.events.json`; cross-batch replication check.
- `verifier.py` — empirical verifiers (`correlates`, `deterministic`, `coincident`)
  + the `Rule`/`RejectedRule` types and the admit/reject ledger.
- `ledger.py` — structural ledger: load `config_path`/identity entries from JSON,
  emit `structural`-supported facts, resolve citations.
- `rules.py` — the derivation rules: `derives`, `same_source`, `stochastic_root`.
- `chainer.py` — forward chain to fixpoint; chaining-soundness property.
- `degeneracy.py` — union-find `same_class`/`classes` over `same_source` only; SCC
  condensation of the `derives` placement graph into irreducible groups.
- `reachability.py` — the verified reachability self-model (constraints carry
  `measured` provenance; observational verdicts blocked until discharged).
- `classify.py` — the degeneracy trichotomy (structural / observational / separable)
  with the falsifiable-non-separation gate and `unconfirmable-structural` state.
- `planner.py` — proven-gain MEASURE-NEXT batches (co-trace / distinguish), per-tile
  mode, Phase-0 seed sweep.
- `loop.py` — the closed loop + three-component lexicographic termination measure;
  a `MockInstrument` for synthetic convergence tests.
- `engine.py` — top-level orchestration tying loader+ledger+chainer+loop together;
  CLI entry point; report emission.

**Tests:** `tools/test_inference_facts.py`, `..._loader.py`, `..._verifier.py`,
`..._ledger.py`, `..._rules.py`, `..._chainer.py`, `..._degeneracy.py`,
`..._reachability.py`, `..._classify.py`, `..._planner.py`, `..._loop.py`,
`..._engine.py`, `..._hw_smoke.py`. Plus a refactor characterization test
`tools/test_trace_runner.py`.

---

## Task 1: Extract `RunnerSession`/`ParseSession` into `tools/trace_runner.py`

**Files:**
- Create: `tools/trace_runner.py`
- Modify: `tools/trace-sweep.py` (remove the moved defs; import them back)
- Modify: `tools/trace_capture.py:339-368` (`HwRunner.__init__` imports from `trace_runner`)
- Test: `tools/test_trace_runner.py`

**Interfaces:**
- Produces: module `trace_runner` exposing `RunnerSession`, `ParseSession`,
  `RunResult`, `_run_one_side`, `_run_patch`, `_run_patch_multi`, `_relabel_events`,
  `_parse_trace_bin`, and constants `RUNNER`, `PATCH_TOOL`, `PARSE_TOOL`, `REPO_ROOT`,
  `MLIR_AIE_ROOT`, `_MOD_TO_TILE_TYPE`, `_TILE_TYPE_TO_MOD`, `_MODE_INT`. Signatures
  preserved verbatim (see spec Section 3 / the existing `trace-sweep.py`):
  - `RunnerSession.__init__(self, xclbin, runner_env, side, stderr_log, verbose=False, cdo_preambles=None, trace_buf_idx=None, reuse_ctx=False)`
  - `RunnerSession.run_one(self, instr, trace_out, inputs=None, outputs=None, ctrlpkts=None, trace_size=1<<20) -> dict`; `.reset()`; `.close()`; context manager.
  - `ParseSession.__init__(self, side, stderr_log, env_for_parse=None)`; `.parse_one(...)`; `.close()`; context manager.
  - `_run_one_side(side, session, runner_env, instr, trace_bin, mlir, events_out, cycles_out, parse_log, ctrlpkt, parser_session=None, trace_mode="event_time") -> RunResult`
- Consumes: nothing new (pure move).

**This is a characterization refactor:** no behavior change. The test pins the public
surface and that both downstream importers still resolve.

- [ ] **Step 1: Write the characterization test (failing)**

```python
# tools/test_trace_runner.py
"""Pins the extracted runner module's public surface and downstream imports."""
import importlib
import inspect


def test_trace_runner_exports_runner_session():
    tr = importlib.import_module("trace_runner")
    for name in ("RunnerSession", "ParseSession", "RunResult", "_run_one_side",
                 "_run_patch", "_run_patch_multi", "_relabel_events",
                 "_parse_trace_bin"):
        assert hasattr(tr, name), f"trace_runner missing {name}"
    for const in ("RUNNER", "PATCH_TOOL", "PARSE_TOOL", "REPO_ROOT",
                  "MLIR_AIE_ROOT", "_MOD_TO_TILE_TYPE", "_MODE_INT"):
        assert hasattr(tr, const), f"trace_runner missing {const}"


def test_runner_session_signature_preserved():
    tr = importlib.import_module("trace_runner")
    sig = inspect.signature(tr.RunnerSession.__init__)
    params = list(sig.parameters)
    assert params[:5] == ["self", "xclbin", "runner_env", "side", "stderr_log"]
    assert "reuse_ctx" in params


def test_sweep_imports_from_runner():
    # trace-sweep.py is hyphenated; load it by path and confirm it re-exports
    # RunnerSession that IS trace_runner.RunnerSession (same object, not a copy).
    import importlib.util
    from pathlib import Path
    tr = importlib.import_module("trace_runner")
    sweep_path = Path(__file__).resolve().parent / "trace-sweep.py"
    spec = importlib.util.spec_from_file_location("_sweep_mod", str(sweep_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.RunnerSession is tr.RunnerSession


def test_capture_hwrunner_uses_runner_module():
    import trace_capture
    r = trace_capture.HwRunner.__init__
    src = inspect.getsource(r)
    assert "trace_runner" in src, "HwRunner must import RunnerSession from trace_runner"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_trace_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trace_runner'`.

- [ ] **Step 3: Create `trace_runner.py` by moving the run-core out of `trace-sweep.py`**

Cut these definitions from `trace-sweep.py` and paste them, verbatim, into a new
`tools/trace_runner.py` (keep their bodies unchanged): the constants block
(`REPO_ROOT`, `MLIR_AIE_ROOT`, `EVENTS_HEADER`, `RUNNER`, `PATCH_TOOL`, `PARSE_TOOL`,
`_MOD_TO_TILE_TYPE`, `_TILE_TYPE_TO_MOD`, `_MODE_INT`, `_GROUNDING_BY_TILE_TYPE`); the
dataclass `RunResult`; classes `RunnerSession` and `ParseSession`; functions
`_run_patch`, `_run_patch_multi`, `_relabel_events`, `_parse_trace_bin`,
`_run_one_side`, `_discover_trace_buf_idx`, `_find_cdo_preambles`,
`_find_post_lowering_mlir`. Add the module docstring and the imports those bodies
need (`json`, `math`, `os`, `subprocess`, `sys`, `Path`, `shlex.quote`, `typing`,
`dataclass`). Leave `EventDef`/`TileSpec`/`TestConfig`/`load_events` and the
`sweep_*`/`_build_lockstep_patch_spec` orchestration in `trace-sweep.py`.

```python
# tools/trace_runner.py
"""Shared bridge-trace run+parse core, extracted from trace-sweep.py.

RunnerSession (long-lived bridge-trace-runner in --batch-stdin mode), ParseSession
(long-lived parse-trace decode server), and the single-batch run+parse cycle
_run_one_side. The sweep orchestration (sweep_multi/sweep_lockstep) stays in
trace-sweep.py and imports these. trace_capture.HwRunner adapts RunnerSession here.
"""
# ... (moved constants, RunResult, RunnerSession, ParseSession, helpers,
#      _run_one_side — bodies unchanged from trace-sweep.py)
```

- [ ] **Step 4: Re-import the moved symbols back into `trace-sweep.py`**

At the top of `trace-sweep.py` (after its own imports), add:

```python
from trace_runner import (
    REPO_ROOT, MLIR_AIE_ROOT, RUNNER, PATCH_TOOL, PARSE_TOOL,
    _MOD_TO_TILE_TYPE, _TILE_TYPE_TO_MOD, _MODE_INT, _GROUNDING_BY_TILE_TYPE,
    RunResult, RunnerSession, ParseSession,
    _run_patch, _run_patch_multi, _relabel_events, _parse_trace_bin, _run_one_side,
    _discover_trace_buf_idx, _find_cdo_preambles, _find_post_lowering_mlir,
)
```

Delete the now-duplicated definitions from `trace-sweep.py`. Keep every call site
unchanged — names resolve to the imports.

- [ ] **Step 5: Point `trace_capture.HwRunner` at `trace_runner`**

Replace the dynamic `trace-sweep.py` load in `trace_capture.py` `HwRunner.__init__`
(currently lines ~352–362) with a direct import:

```python
    def __init__(self, xclbin, stderr_log, side="HW"):
        import trace_runner
        self._RunnerSession = trace_runner.RunnerSession
        self._session = self._RunnerSession(
            xclbin=xclbin, runner_env={}, side=side, stderr_log=stderr_log)
```

(Preserve whatever `runner_env`/kwargs the existing body passed; only the source of
`RunnerSession` changes.)

- [ ] **Step 6: Run the new test and the existing sweep/capture tests**

Run: `cd tools && python -m pytest test_trace_runner.py test_trace_sweep.py test_trace_capture.py -v`
Expected: PASS for `test_trace_runner.py`; `test_trace_sweep.py` and
`test_trace_capture.py` still PASS (no behavior change). If a sweep test imported a
moved private symbol directly, fix its import to `from trace_runner import ...`.

- [ ] **Step 7: Commit**

```bash
git add tools/trace_runner.py tools/trace-sweep.py tools/trace_capture.py tools/test_trace_runner.py
git commit -m "refactor(#140): extract RunnerSession/ParseSession into trace_runner module"
```

---

## Task 2: Fact / Support / KB data model + no-axiom property

**Files:**
- Create: `tools/inference/__init__.py`
- Create: `tools/inference/facts.py`
- Test: `tools/test_inference_facts.py`

**Interfaces:**
- Produces:
  - `Measured()`, `Structural(cite: str)`, `Derived(rule: str, premises: tuple[Fact, ...])` — frozen dataclasses; `Support = Measured | Structural | Derived`.
  - `Fact(predicate: str, args: tuple, support: Support)` — frozen, hashable; method `.key() -> tuple[str, tuple]` returns `(predicate, args)`.
  - `leaves(fact: Fact) -> frozenset[Fact]` — the measured/structural leaves of a fact's provenance DAG.
  - `KB` with `.facts: dict[tuple, Fact]`, `.admitted_rules: list`, `.rejected_rules: list`, `.ledger: dict[str, dict]`; methods `.add(fact) -> Fact`, `.get(predicate, args) -> Fact | None`, `.by_predicate(predicate) -> list[Fact]`, `.has(predicate, args) -> bool`.
  - `provenance_ok(kb: KB) -> bool` — every leaf of every fact is `Measured`, or `Structural` whose `cite` is in `kb.ledger`; a `Derived` leaf is impossible (returns False).
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_facts.py
from inference.facts import (Measured, Structural, Derived, Fact, KB,
                             leaves, provenance_ok)


def test_fact_is_hashable_and_keyed():
    f = Fact("fired", ("1|0|2|X", 0, 5), Measured())
    assert f.key() == ("fired", ("1|0|2|X", 0, 5))
    assert f in {f}  # hashable


def test_leaves_of_measured_is_itself():
    f = Fact("fired", ("X", 0, 5), Measured())
    assert leaves(f) == frozenset({f})


def test_leaves_walk_to_measured_and_structural():
    m = Fact("correlates", ("A", "B", 3), Measured())
    s = Fact("config_path", ("B", "A", "cite:route#7"), Structural("cite:route#7"))
    d = Fact("derives", ("A", "B", 3), Derived("derives_rule", (m, s)))
    assert leaves(d) == frozenset({m, s})


def test_provenance_ok_requires_structural_leaf_in_ledger():
    kb = KB.empty()
    kb.ledger["cite:route#7"] = {"a": "B", "b": "A", "kind": "route"}
    m = Fact("correlates", ("A", "B", 3), Measured())
    s = Fact("config_path", ("B", "A", "cite:route#7"), Structural("cite:route#7"))
    d = kb.add(Fact("derives", ("A", "B", 3), Derived("derives_rule", (m, s))))
    kb.add(m); kb.add(s)
    assert provenance_ok(kb) is True


def test_provenance_ok_fails_when_citation_missing_from_ledger():
    kb = KB.empty()  # ledger empty -> the cite is unaudited
    s = Fact("config_path", ("B", "A", "cite:ghost"), Structural("cite:ghost"))
    kb.add(s)
    assert provenance_ok(kb) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_facts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/__init__.py
"""Trace inference engine (v1). See docs/superpowers/specs/2026-06-21-*."""
__version__ = "0.1.0"
```

```python
# tools/inference/facts.py
"""Facts, support types, and the knowledge base.

A fact is an immutable record (predicate, args, support). Support is exactly one of
measured (straight from capture data), structural(cite) (a quote of the loaded
configuration, ledgered to its location), or derived(rule, premises). The keystone
property `provenance_ok` enforces that every fact bottoms out only in measured or
ledgered-structural leaves -- no unaudited axioms.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Union, FrozenSet, Optional, List, Dict


@dataclass(frozen=True)
class Measured:
    """Support: read straight from replicated capture data."""


@dataclass(frozen=True)
class Structural:
    """Support: a quote of the loaded configuration, cited to its location."""
    cite: str


@dataclass(frozen=True)
class Derived:
    """Support: produced by an admitted rule from existing facts."""
    rule: str
    premises: Tuple["Fact", ...]


Support = Union[Measured, Structural, Derived]


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: Tuple
    support: Support

    def key(self) -> Tuple[str, Tuple]:
        return (self.predicate, self.args)


def leaves(fact: Fact) -> FrozenSet[Fact]:
    """The measured/structural leaves of a fact's provenance DAG."""
    s = fact.support
    if isinstance(s, (Measured, Structural)):
        return frozenset({fact})
    out: set = set()
    for p in s.premises:
        out |= leaves(p)
    return frozenset(out)


@dataclass
class KB:
    facts: Dict[Tuple, Fact]
    admitted_rules: List = field(default_factory=list)
    rejected_rules: List = field(default_factory=list)
    ledger: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "KB":
        return cls(facts={}, admitted_rules=[], rejected_rules=[], ledger={})

    def add(self, fact: Fact) -> Fact:
        self.facts[fact.key()] = fact
        return fact

    def get(self, predicate: str, args: Tuple) -> Optional[Fact]:
        return self.facts.get((predicate, args))

    def has(self, predicate: str, args: Tuple) -> bool:
        return (predicate, args) in self.facts

    def by_predicate(self, predicate: str) -> List[Fact]:
        return [f for (p, _), f in self.facts.items() if p == predicate]


def provenance_ok(kb: KB) -> bool:
    """Keystone: every leaf is measured, or structural with a ledgered citation."""
    for f in kb.facts.values():
        for leaf in leaves(f):
            s = leaf.support
            if isinstance(s, Measured):
                continue
            if isinstance(s, Structural):
                if s.cite not in kb.ledger:
                    return False
            else:  # a Derived fact can never be a leaf
                return False
    return True
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_facts.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/__init__.py tools/inference/facts.py tools/test_inference_facts.py
git commit -m "feat(#140): inference KB -- facts, support types, no-axiom property"
```

---

## Task 3: Measured leaves — load `fired` from disk + replication check

**Files:**
- Create: `tools/inference/loader.py`
- Test: `tools/test_inference_loader.py`

**Interfaces:**
- Consumes: `Fact`, `Measured` (Task 2); `trace_join.batch_firsts`,
  `trace_join._batch_names`, `trace_join.anchored_firsts` (existing).
- Produces:
  - `load_fired(run_dirs: list[str], anchor_key: str = "1|2|0|PERF_CNT_2") -> list[Fact]` — one `Fact("fired", (event_key, run_idx, anchored_ts), Measured())` per (event, run), taking each event's first co-traced batch per run (matching `pair_derivability`'s "first co-tracing batch" rule).
  - `replication_violations(run_dirs, anchor_key="1|2|0|PERF_CNT_2", eps=2.0) -> list[dict]` — for each (event_key, run) appearing in more than one batch, the disagreements where `|ts_i - ts_j| > eps`. An empty list means measured leaves replicate (leaf-validity sub-case 1; would have caught the precursor's mode/start_col bugs).

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_loader.py
import json
from inference.loader import load_fired, replication_violations


def _ev(col, row, name, soc, slot=0, pkt_type=0, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc, "soc": soc, "mode": mode}


def _write_batch(d, events):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.events.json").write_text(
        json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))


def _make_run(tmp_path, run_name, batches):
    rd = tmp_path / run_name
    for bn, evs in batches.items():
        _write_batch(rd / bn / "hw", evs)
    return str(rd)


def test_load_fired_emits_anchored_measured_facts(tmp_path):
    r0 = _make_run(tmp_path, "run0", {"batch_00": [
        _ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)]})
    facts = load_fired([r0])
    keyed = {f.args[0]: f for f in facts}
    assert keyed["1|0|0|DMA"].args == ("1|0|0|DMA", 0, 300)   # 1300 - 1000
    assert type(keyed["1|0|0|DMA"].support).__name__ == "Measured"
    assert keyed["1|2|0|PERF_CNT_2"].args == ("1|2|0|PERF_CNT_2", 0, 0)


def test_load_fired_indexes_runs(tmp_path):
    r0 = _make_run(tmp_path, "run0", {"batch_00": [
        _ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)]})
    r1 = _make_run(tmp_path, "run1", {"batch_00": [
        _ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1305)]})
    facts = load_fired([r0, r1])
    runs = sorted(f.args[1] for f in facts if f.args[0] == "1|0|0|DMA")
    assert runs == [0, 1]


def test_replication_clean_when_batches_agree(tmp_path):
    r0 = _make_run(tmp_path, "run0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1301)]})
    assert replication_violations([r0]) == []


def test_replication_flags_disagreement(tmp_path):
    # same (event, run) but the two batches disagree by 50 >> eps -> a planted bug
    r0 = _make_run(tmp_path, "run0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1350)]})
    viols = replication_violations([r0])
    assert any(v["event_key"] == "1|0|0|DMA" for v in viols)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.loader'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/loader.py
"""Measured leaves: load `fired` facts from captured trace.events.json on disk.

Reuses trace_join's batch loading + anchoring (the per-run `fired` atoms survive on
disk in batch_NN/hw/trace.events.json even though build_derivability_graph discards
them). One `fired(event_key, run_idx, anchored_ts)` per (event, run), from each
event's first co-traced batch -- matching pair_derivability's selection so the
verifier (Task 4) sees a consistent value per (event, run).
"""
from __future__ import annotations
from typing import List, Dict
import trace_join as tj
from inference.facts import Fact, Measured

ANCHOR = "1|2|0|PERF_CNT_2"


def _first_firsts(run_dir: str, anchor_key: str) -> Dict[str, int]:
    """event_key -> anchored_ts, from the first batch that traces it in this run."""
    seen: Dict[str, int] = {}
    for bn in tj._batch_names(run_dir):
        f = tj.batch_firsts(run_dir, bn, anchor_key)
        for ekey, ats in f.items():
            seen.setdefault(ekey, ats)
    return seen


def load_fired(run_dirs: List[str], anchor_key: str = ANCHOR) -> List[Fact]:
    facts: List[Fact] = []
    for run_idx, rd in enumerate(run_dirs):
        for ekey, ats in _first_firsts(rd, anchor_key).items():
            facts.append(Fact("fired", (ekey, run_idx, ats), Measured()))
    return facts


def replication_violations(run_dirs: List[str], anchor_key: str = ANCHOR,
                           eps: float = 2.0) -> List[dict]:
    """Same (event_key, run) across multiple batches must agree within eps."""
    out: List[dict] = []
    for run_idx, rd in enumerate(run_dirs):
        per_batch: Dict[str, List[int]] = {}
        for bn in tj._batch_names(rd):
            f = tj.batch_firsts(rd, bn, anchor_key)
            for ekey, ats in f.items():
                per_batch.setdefault(ekey, []).append(ats)
        for ekey, vals in per_batch.items():
            if len(vals) > 1 and (max(vals) - min(vals)) > eps:
                out.append({"event_key": ekey, "run": run_idx,
                            "values": sorted(vals),
                            "spread": max(vals) - min(vals)})
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_loader.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/loader.py tools/test_inference_loader.py
git commit -m "feat(#140): load fired measured-leaf facts + cross-batch replication check"
```

---

## Task 4: Empirical verifier — `correlates` / `deterministic` / `coincident`

**Files:**
- Create: `tools/inference/verifier.py`
- Test: `tools/test_inference_verifier.py`

**Interfaces:**
- Consumes: `trace_join.pair_derivability`, `trace_variance.aggregate` (existing);
  `trace_join.batch_firsts`/`_batch_names`.
- Produces:
  - `EPS = 2.0`
  - `Rule(name: str, verify)` and `RejectedRule(name: str, reason: str, evidence: dict)` — dataclasses.
  - `correlates(run_dirs, a, b, anchor_key=ANCHOR, eps=EPS) -> int | None` — `int(round(mean(a-b)))` offset if `std(a-b) <= eps` across co-traced runs, else `None`. Symmetric in existence (`std` is sign-invariant); the offset carries the `a-b` sign.
  - `deterministic(run_dirs, event_key, anchor_key=ANCHOR, eps=EPS) -> bool` — `std(anchored_ts) <= eps` across runs.
  - `coincident(run_dirs, a, b, anchor_key=ANCHOR, eps=EPS) -> bool` — both fire at identical anchored ts within eps in *every* co-traced run (a degeneracy candidate; offset ~ 0 with low std).
  - `verify_offset_stable(run_dirs, a, b, anchor_key=ANCHOR, eps=EPS) -> tuple[bool, RejectedRule | None]` — the rule-as-hypothesis wrapper: returns `(True, None)` when admitted, `(False, RejectedRule(...))` when the offset is not stable (a recorded finding).

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_verifier.py
import json
from inference.verifier import (correlates, deterministic, coincident,
                                verify_offset_stable, RejectedRule)


def _ev(col, row, name, soc, slot=0, pkt_type=0, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc, "soc": soc, "mode": mode}


def _make_runs(tmp_path, per_run_events):
    """per_run_events: list over runs of {event_name(at 1|0|0): anchored_offset}."""
    dirs = []
    for i, off in enumerate(per_run_events):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for name, delta in off.items():
            evs.append(_ev(1, 0, name, 1000 + delta))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_correlates_constant_offset(tmp_path):
    # A = S + 50 in every run -> std(A-S) ~ 0 -> correlates, offset 50
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 150}, {"S": 200, "A": 250},
                                 {"S": 300, "A": 350}])
    off = correlates(dirs, "1|0|0|A", "1|0|0|S")
    assert off == 50


def test_correlates_none_when_offset_unstable(tmp_path):
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 150}, {"S": 200, "A": 400},
                                 {"S": 300, "A": 360}])
    assert correlates(dirs, "1|0|0|A", "1|0|0|S") is None


def test_deterministic_true_for_fixed_anchored_ts(tmp_path):
    dirs = _make_runs(tmp_path, [{"D": 40}, {"D": 41}, {"D": 40}])
    assert deterministic(dirs, "1|0|0|D") is True


def test_deterministic_false_for_jittery_event(tmp_path):
    dirs = _make_runs(tmp_path, [{"J": 40}, {"J": 90}, {"J": 140}])
    assert deterministic(dirs, "1|0|0|J") is False


def test_coincident_when_two_events_share_anchored_ts(tmp_path):
    dirs = _make_runs(tmp_path, [{"A": 40, "B": 40}, {"A": 41, "B": 41}])
    assert coincident(dirs, "1|0|0|A", "1|0|0|B") is True


def test_verify_offset_stable_rejects_eps_boundary(tmp_path):
    # stable for 2 runs, breaks on the 3rd -> rejected finding
    dirs = _make_runs(tmp_path, [{"S": 100, "A": 150}, {"S": 200, "A": 250},
                                 {"S": 300, "A": 999}])
    ok, finding = verify_offset_stable(dirs, "1|0|0|A", "1|0|0|S")
    assert ok is False
    assert isinstance(finding, RejectedRule)
    assert "1|0|0|A" in finding.evidence["pair"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.verifier'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/verifier.py
"""Empirical verifiers: correlates / deterministic / coincident.

Seeded by the existing cross-run std machinery -- correlates wraps
trace_join.pair_derivability (std of a.ts - b.ts), deterministic wraps
trace_variance.aggregate on the anchored ts. A rule is a hypothesis paired with a
verifier; for these numeric rules the rule body and the verifier coincide, so
self-verification falls out. A failed rule is never used and is itself a finding
(RejectedRule).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
import trace_join as tj
import trace_variance as tv

ANCHOR = "1|2|0|PERF_CNT_2"
EPS = 2.0


@dataclass
class Rule:
    name: str
    verify: Callable


@dataclass
class RejectedRule:
    name: str
    reason: str
    evidence: dict


def correlates(run_dirs: List[str], a: str, b: str,
               anchor_key: str = ANCHOR, eps: float = EPS) -> Optional[int]:
    """Offset int(round(mean(a-b))) if std(a-b) <= eps across co-traced runs."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is None or st.std > eps:
        return None
    return int(round(st.mean))


def _anchored_per_run(run_dirs: List[str], event_key: str,
                      anchor_key: str) -> List[Dict[str, int]]:
    per_run: List[Dict[str, int]] = []
    for rd in run_dirs:
        for bn in tj._batch_names(rd):
            f = tj.batch_firsts(rd, bn, anchor_key)
            if event_key in f:
                per_run.append({event_key: f[event_key]})
                break
    return per_run


def deterministic(run_dirs: List[str], event_key: str,
                  anchor_key: str = ANCHOR, eps: float = EPS) -> bool:
    per_run = _anchored_per_run(run_dirs, event_key, anchor_key)
    stats = tv.aggregate(per_run).get(event_key)
    return stats is not None and stats.std <= eps


def coincident(run_dirs: List[str], a: str, b: str,
               anchor_key: str = ANCHOR, eps: float = EPS) -> bool:
    """a and b fire at identical anchored ts (within eps) in every co-traced run."""
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    return st is not None and st.std <= eps and abs(st.mean) <= eps


def verify_offset_stable(run_dirs: List[str], a: str, b: str,
                         anchor_key: str = ANCHOR,
                         eps: float = EPS) -> Tuple[bool, Optional[RejectedRule]]:
    st = tj.pair_derivability(run_dirs, a, b, anchor_key)
    if st is not None and st.std <= eps:
        return True, None
    reason = "never co-traced" if st is None else f"std {st.std:.2f} > eps {eps}"
    return False, RejectedRule(
        name="offset_stable",
        reason=reason,
        evidence={"pair": (a, b), "stats": None if st is None else st._asdict()})
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_verifier.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/verifier.py tools/test_inference_verifier.py
git commit -m "feat(#140): empirical verifier -- correlates/deterministic/coincident + rejected-rule findings"
```

---

## Task 5: Structural ledger — load `config_path` / identity entries from JSON

**Files:**
- Create: `tools/inference/ledger.py`
- Test: `tools/test_inference_ledger.py`

**Interfaces:**
- Consumes: `Fact`, `Structural`, `KB` (Task 2).
- Produces:
  - Ledger JSON schema (documented in module docstring): `{"entries": [{"cite": str, "a": event_key, "b": event_key, "kind": "route"|"bd"|"lock"|"identity"}]}`. For `route`/`bd`/`lock`, the entry asserts the config routes `a`'s producer to `b`'s consumer (orientation premise for `derives`). For `identity`, it asserts `a` and `b` are the *same physical event* at two trace units (premise for `same_source`).
  - `load_ledger(path: str) -> dict[str, dict]` — `cite -> entry`; raises `ValueError` on a duplicate cite or a malformed entry.
  - `ledger_facts(ledger: dict) -> list[Fact]` — `config_path(a, b, cite)` (`Structural(cite)`) for kind in {route, bd, lock}; `identity(a, b, cite)` (`Structural(cite)`) for kind == identity.
  - `install_ledger(kb: KB, ledger: dict) -> None` — sets `kb.ledger = ledger` and adds all `ledger_facts` to `kb`.
  - `citation_resolves(ledger: dict, cite: str) -> bool` — v1: the cite is present in the ledger. (Plan 2 replaces this with resolution against the real loaded binary.)

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_ledger.py
import json
import pytest
from inference.facts import KB
from inference.ledger import (load_ledger, ledger_facts, install_ledger,
                              citation_resolves)


def _write(tmp_path, entries):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"entries": entries}))
    return str(p)


def test_load_ledger_keys_by_cite(tmp_path):
    p = _write(tmp_path, [
        {"cite": "route#7", "a": "1|0|0|DMA", "b": "1|1|3|PORT", "kind": "route"}])
    led = load_ledger(p)
    assert led["route#7"]["kind"] == "route"


def test_load_ledger_rejects_duplicate_cite(tmp_path):
    p = _write(tmp_path, [
        {"cite": "x", "a": "A", "b": "B", "kind": "route"},
        {"cite": "x", "a": "C", "b": "D", "kind": "lock"}])
    with pytest.raises(ValueError):
        load_ledger(p)


def test_ledger_facts_emit_config_path_and_identity(tmp_path):
    p = _write(tmp_path, [
        {"cite": "route#7", "a": "A", "b": "B", "kind": "route"},
        {"cite": "id#1", "a": "A", "b": "A2", "kind": "identity"}])
    led = load_ledger(p)
    facts = ledger_facts(led)
    preds = {(f.predicate, f.args) for f in facts}
    assert ("config_path", ("A", "B", "route#7")) in preds
    assert ("identity", ("A", "A2", "id#1")) in preds
    for f in facts:
        assert type(f.support).__name__ == "Structural"


def test_install_ledger_makes_provenance_ok_hold(tmp_path):
    from inference.facts import provenance_ok
    p = _write(tmp_path, [
        {"cite": "route#7", "a": "A", "b": "B", "kind": "route"}])
    led = load_ledger(p)
    kb = KB.empty()
    install_ledger(kb, led)
    assert citation_resolves(led, "route#7") is True
    assert citation_resolves(led, "ghost") is False
    assert provenance_ok(kb) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.ledger'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/ledger.py
"""Structural ledger: config_path / identity facts read from an input JSON file.

In Plan 1 the ledger is an INPUT -- a quote of the loaded configuration provided as
JSON (hand-authored for the add_one_using_dma HW smoke test). Plan 2 replaces the
loader with an automated extractor (a Rust examples/dump_config_json.rs over the
parsed CDO: stream-switch routes, BD chains, lock pairings) and citation_resolves
with resolution against the real binary.

Schema:
  {"entries": [
    {"cite": str, "a": event_key, "b": event_key,
     "kind": "route"|"bd"|"lock"|"identity"}, ...]}

route/bd/lock -> config_path(a, b, cite): the config routes a's producer to b's
consumer (orientation premise for `derives`).
identity      -> identity(a, b, cite): a and b are the same physical event at two
                 trace units (premise for `same_source`).
"""
from __future__ import annotations
import json
from typing import Dict, List
from inference.facts import Fact, Structural, KB

_KINDS = {"route", "bd", "lock", "identity"}


def load_ledger(path: str) -> Dict[str, dict]:
    raw = json.loads(open(path).read())
    out: Dict[str, dict] = {}
    for e in raw.get("entries", []):
        for k in ("cite", "a", "b", "kind"):
            if k not in e:
                raise ValueError(f"ledger entry missing {k!r}: {e}")
        if e["kind"] not in _KINDS:
            raise ValueError(f"unknown kind {e['kind']!r} in {e}")
        if e["cite"] in out:
            raise ValueError(f"duplicate cite {e['cite']!r}")
        out[e["cite"]] = e
    return out


def ledger_facts(ledger: Dict[str, dict]) -> List[Fact]:
    facts: List[Fact] = []
    for cite, e in ledger.items():
        pred = "identity" if e["kind"] == "identity" else "config_path"
        facts.append(Fact(pred, (e["a"], e["b"], cite), Structural(cite)))
    return facts


def install_ledger(kb: KB, ledger: Dict[str, dict]) -> None:
    kb.ledger = ledger
    for f in ledger_facts(ledger):
        kb.add(f)


def citation_resolves(ledger: Dict[str, dict], cite: str) -> bool:
    return cite in ledger
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_ledger.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/ledger.py tools/test_inference_ledger.py
git commit -m "feat(#140): structural ledger loader -- config_path/identity facts from JSON"
```

---

## Task 6: Derivation rules — `derives`, `same_source`, `stochastic_root`

**Files:**
- Create: `tools/inference/rules.py`
- Test: `tools/test_inference_rules.py`

**Interfaces:**
- Consumes: `Fact`, `Derived`, `KB`, `provenance` types (Task 2);
  `correlates`/`deterministic`/`coincident` (Task 4); the `config_path`/`identity`
  facts installed by the ledger (Task 5).
- Produces, each `(run_dirs, kb, a, b) -> Fact | None` returning a `Derived`-supported
  fact when admitted, else `None`:
  - `try_derives(run_dirs, kb, child, parent, anchor_key=ANCHOR, eps=EPS) -> Fact | None`
    — admits `derives(child, parent, offset)` iff **(1)** `correlates(child, parent)`
    holds, **(2)** the parent is stochastic (`not deterministic(parent)`), **(3)**
    `config_path(parent, child, cite)` is in `kb`. Premises = the `config_path` fact +
    a `correlates` `Derived` fact. **Placement only** — never inspects timing
    direction, so a backpressure event whose timing-cause opposes its dataflow
    orientation is still placed and never labeled causal.
  - `try_same_source(run_dirs, kb, a, b, anchor_key=ANCHOR, eps=EPS) -> Fact | None`
    — admits `same_source(a, b)` iff an `identity(a, b, cite)` fact is in `kb` **and**
    `coincident(a, b)` holds.
  - `is_stochastic_root(kb, event_key) -> bool` — true iff `event_key` is not
    `deterministic` (carries a `stochastic` marker, see below) and no `derives(child=event_key, ...)` fact exists in `kb`.
  - A helper `mark_determinism(run_dirs, kb, event_keys, anchor_key=ANCHOR, eps=EPS) -> None`
    that adds `deterministic(event_key)` / `stochastic(event_key)` `Derived` facts
    (premised on the relevant `fired` facts) for each key, so the chainer can reason
    over determinism as facts rather than re-measuring.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_rules.py
import json
from inference.facts import KB
from inference.ledger import install_ledger
from inference.rules import (try_derives, try_same_source, is_stochastic_root,
                             mark_determinism)
from inference.loader import load_fired


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    """rows: list over runs of {name: anchored_offset at 1|0|0}."""
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for name, delta in off.items():
            evs.append(_ev(1, 0, name, 1000 + delta))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def _kb_with_ledger(tmp_path, entries):
    p = tmp_path / "led.json"
    p.write_text(json.dumps({"entries": entries}))
    from inference.ledger import load_ledger
    kb = KB.empty()
    install_ledger(kb, load_ledger(str(p)))
    return kb


def test_derives_admitted_with_stochastic_parent_and_config_path(tmp_path):
    # parent S jitters; child C = S + 30 every run; config routes S -> C
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 380}])
    kb = _kb_with_ledger(tmp_path, [
        {"cite": "route#1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|C", "1|0|0|S")
    assert f is not None
    assert f.predicate == "derives" and f.args == ("1|0|0|C", "1|0|0|S", 30)


def test_derives_rejected_when_parent_deterministic(tmp_path):
    # parent D is fixed -> constant offset is NOT placement-derivation
    dirs = _runs(tmp_path, [{"D": 40, "C": 70}, {"D": 41, "C": 71},
                            {"D": 40, "C": 70}])
    kb = _kb_with_ledger(tmp_path, [
        {"cite": "route#1", "a": "1|0|0|D", "b": "1|0|0|C", "kind": "route"}])
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|D") is None


def test_derives_rejected_without_config_path(tmp_path):
    # co-varying pair but NO config_path -> stays correlates, no derives
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 200, "C": 230},
                            {"S": 350, "C": 380}])
    kb = KB.empty()  # empty ledger
    assert try_derives(dirs, kb, "1|0|0|C", "1|0|0|S") is None


def test_derives_places_backpressure_event_without_causal_claim(tmp_path):
    # STREAM_STARVATION fires from downstream backpressure; config still routes
    # producer P -> starvation observer SS. It must be PLACED, never causal.
    dirs = _runs(tmp_path, [{"P": 100, "SS": 130}, {"P": 220, "SS": 250},
                            {"P": 300, "SS": 330}])
    kb = _kb_with_ledger(tmp_path, [
        {"cite": "route#9", "a": "1|0|0|P", "b": "1|0|0|SS", "kind": "route"}])
    f = try_derives(dirs, kb, "1|0|0|SS", "1|0|0|P")
    assert f is not None and f.predicate == "derives"  # placed
    assert "caus" not in f.support.rule.lower()         # never labeled causal


def test_same_source_requires_identity_and_coincidence(tmp_path):
    dirs = _runs(tmp_path, [{"A": 40, "A2": 40}, {"A": 41, "A2": 41}])
    kb = _kb_with_ledger(tmp_path, [
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

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.rules'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/rules.py
"""Verified derivation rules: derives, same_source, stochastic_root.

Orientation and identity are DERIVED from structural config premises by these stated,
verified rules -- not primitive assertions. `derives` is placement (dataflow-upstream
+ stable offset), explicitly NOT timing-causation: the rule never inspects timing
direction, so a backpressure event (timing-cause opposite to dataflow) is still
placed correctly and is never labeled causal.
"""
from __future__ import annotations
from typing import List, Optional
from inference.facts import Fact, Derived, KB
from inference.verifier import correlates, deterministic, coincident, ANCHOR, EPS


def mark_determinism(run_dirs: List[str], kb: KB, event_keys: List[str],
                     anchor_key: str = ANCHOR, eps: float = EPS) -> None:
    for ek in event_keys:
        is_det = deterministic(run_dirs, ek, anchor_key, eps)
        pred = "deterministic" if is_det else "stochastic"
        premises = tuple(f for f in kb.by_predicate("fired") if f.args[0] == ek)
        kb.add(Fact(pred, (ek,), Derived("determinism_rule", premises)))


def try_derives(run_dirs: List[str], kb: KB, child: str, parent: str,
                anchor_key: str = ANCHOR, eps: float = EPS) -> Optional[Fact]:
    # (1) measured stable offset
    offset = correlates(run_dirs, child, parent, anchor_key, eps)
    if offset is None:
        return None
    # (2) parent is stochastic -- a deterministic parent transmits no jitter
    if deterministic(run_dirs, parent, anchor_key, eps):
        return None
    # (3) config_path(parent, child) gives orientation by verified rule
    cp = next((f for f in kb.by_predicate("config_path")
               if f.args[0] == parent and f.args[1] == child), None)
    if cp is None:
        return None
    corr = Fact("correlates", (child, parent, offset), Derived("correlates_rule", ()))
    return Fact("derives", (child, parent, offset),
                Derived("derives_rule_placement", (cp, corr)))


def try_same_source(run_dirs: List[str], kb: KB, a: str, b: str,
                    anchor_key: str = ANCHOR, eps: float = EPS) -> Optional[Fact]:
    ident = next((f for f in kb.by_predicate("identity")
                  if {f.args[0], f.args[1]} == {a, b}), None)
    if ident is None:
        return None
    if not coincident(run_dirs, a, b, anchor_key, eps):
        return None
    coin = Fact("coincident", (a, b), Derived("coincident_rule", ()))
    return Fact("same_source", (a, b), Derived("same_source_rule", (ident, coin)))


def is_stochastic_root(kb: KB, event_key: str) -> bool:
    if kb.has("deterministic", (event_key,)):
        return False
    if not kb.has("stochastic", (event_key,)):
        return False
    has_parent = any(f.args[0] == event_key for f in kb.by_predicate("derives"))
    return not has_parent
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_rules.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/rules.py tools/test_inference_rules.py
git commit -m "feat(#140): verified derivation rules -- derives (placement), same_source, stochastic_root"
```

---

## Task 7: Forward chainer to fixpoint + chaining-soundness property

**Files:**
- Create: `tools/inference/chainer.py`
- Test: `tools/test_inference_chainer.py`

**Interfaces:**
- Consumes: `KB`, `leaves`, `provenance_ok`, `Measured`, `Structural` (Task 2);
  `load_fired` (Task 3); `mark_determinism`, `try_derives`, `try_same_source`,
  `is_stochastic_root` (Task 6); ledger install (Task 5).
- Produces:
  - `chain(run_dirs, kb, candidate_pairs, anchor_key=ANCHOR, eps=EPS) -> KB` — apply
    `mark_determinism` (over all `fired` event keys), then repeatedly attempt
    `try_derives`/`try_same_source` over `candidate_pairs` until no new fact is added
    (fixpoint). `candidate_pairs` is an iterable of `(child, parent)` /
    `(a, b)` event-key pairs to test (in v1 supplied by the caller / planner; Task 11).
  - `classify_events(kb, event_keys) -> dict[str, str]` — each event ends `"derived"`
    (a `derives(child=ek,...)` exists), `"stochastic_root"` (`is_stochastic_root`),
    `"deterministic"`, or `"unresolved"`.
  - `chaining_sound(kb) -> bool` — alias asserting `provenance_ok(kb)` after a fixpoint
    (every derived fact bottoms out only in measured/ledgered-structural leaves).

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_chainer.py
import json
from inference.facts import KB, provenance_ok
from inference.ledger import load_ledger, install_ledger
from inference.loader import load_fired
from inference.chainer import chain, classify_events, chaining_sound


def _ev(col, row, name, soc):
    return {"col": col, "row": row, "pkt_type": 0, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for name, delta in off.items():
            evs.append(_ev(1, 0, name, 1000 + delta))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def _kb(tmp_path, dirs, entries):
    p = tmp_path / "led.json"
    p.write_text(json.dumps({"entries": entries}))
    kb = KB.empty()
    install_ledger(kb, load_ledger(str(p)))
    for f in load_fired(dirs):
        kb.add(f)
    return kb


def test_chain_places_child_and_marks_root(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 230, "C": 260},
                            {"S": 300, "C": 330}])
    kb = _kb(tmp_path, dirs, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    cls = classify_events(kb, ["1|0|0|S", "1|0|0|C", "1|2|0|PERF_CNT_2"])
    assert cls["1|0|0|C"] == "derived"
    assert cls["1|0|0|S"] == "stochastic_root"
    assert cls["1|2|0|PERF_CNT_2"] == "deterministic"


def test_chain_reaches_fixpoint_idempotent(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 230, "C": 260}])
    kb = _kb(tmp_path, dirs, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    n1 = len(kb.facts)
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    assert len(kb.facts) == n1


def test_chaining_soundness_property_holds(tmp_path):
    dirs = _runs(tmp_path, [{"S": 100, "C": 130}, {"S": 230, "C": 260}])
    kb = _kb(tmp_path, dirs, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    kb = chain(dirs, kb, [("1|0|0|C", "1|0|0|S")])
    assert chaining_sound(kb) is True
    assert provenance_ok(kb) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_chainer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.chainer'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/chainer.py
"""Forward chainer: apply verified rules to fixpoint.

Marks determinism for every fired event, then repeatedly attempts derives/same_source
over the candidate pairs until no new fact is added. Every fact it adds carries a
Derived provenance whose leaves are measured or ledgered-structural -- chaining_sound
(== provenance_ok at fixpoint) pins the no-unaudited-axiom keystone.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
from inference.facts import KB, provenance_ok
from inference.rules import (mark_determinism, try_derives, try_same_source,
                             is_stochastic_root)
from inference.verifier import ANCHOR, EPS


def _fired_event_keys(kb: KB) -> List[str]:
    return sorted({f.args[0] for f in kb.by_predicate("fired")})


def chain(run_dirs: List[str], kb: KB,
          candidate_pairs: Iterable[Tuple[str, str]],
          anchor_key: str = ANCHOR, eps: float = EPS) -> KB:
    pairs = list(candidate_pairs)
    keys = _fired_event_keys(kb)
    undetermined = [k for k in keys
                    if not (kb.has("deterministic", (k,)) or kb.has("stochastic", (k,)))]
    if undetermined:
        mark_determinism(run_dirs, kb, undetermined, anchor_key, eps)

    changed = True
    while changed:
        changed = False
        for a, b in pairs:
            if not kb.has("derives", (a, b, _existing_offset(kb, a, b))):
                d = try_derives(run_dirs, kb, a, b, anchor_key, eps)
                if d is not None and not kb.has(d.predicate, d.args):
                    kb.add(d); changed = True
            if not _has_same_source(kb, a, b):
                s = try_same_source(run_dirs, kb, a, b, anchor_key, eps)
                if s is not None and not kb.has(s.predicate, s.args):
                    kb.add(s); changed = True
    return kb


def _existing_offset(kb: KB, child: str, parent: str):
    for f in kb.by_predicate("derives"):
        if f.args[0] == child and f.args[1] == parent:
            return f.args[2]
    return object()  # sentinel: no existing derives -> has(...) is False


def _has_same_source(kb: KB, a: str, b: str) -> bool:
    return any({f.args[0], f.args[1]} == {a, b}
               for f in kb.by_predicate("same_source"))


def classify_events(kb: KB, event_keys: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    derived_children = {f.args[0] for f in kb.by_predicate("derives")}
    for ek in event_keys:
        if ek in derived_children:
            out[ek] = "derived"
        elif is_stochastic_root(kb, ek):
            out[ek] = "stochastic_root"
        elif kb.has("deterministic", (ek,)):
            out[ek] = "deterministic"
        else:
            out[ek] = "unresolved"
    return out


def chaining_sound(kb: KB) -> bool:
    return provenance_ok(kb)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_chainer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/chainer.py tools/test_inference_chainer.py
git commit -m "feat(#140): forward chainer to fixpoint + chaining-soundness property"
```

---

## Task 8: Union-find degeneracy substrate + SCC condensation

**Files:**
- Create: `tools/inference/degeneracy.py`
- Test: `tools/test_inference_degeneracy.py`

**Interfaces:**
- Consumes: `KB` (Task 2); the `same_source` / `derives` facts (Tasks 6–7).
- Produces:
  - `IdentityClasses` with the two-method interface from the spec:
    `same_class(a, b) -> bool` and `classes() -> list[frozenset[str]]` (partition).
    Built from `same_source` edges **only** (a non-zero `derives` offset is proof two
    events are not identical, so `derives` edges never enter the identity closure).
    `IdentityClasses.from_kb(kb) -> IdentityClasses`.
  - `condense(kb) -> tuple[dict[str, int], list[frozenset[str]]]` — over the `derives`
    placement edges (directed `parent -> child`), detect SCCs (Tarjan); return
    `(component_id_by_event, [members_of_each_irreducible_group])`. A multi-event SCC
    (circular BD chain / lock round-trip / ping-pong) is one **irreducible group**.
    Singletons are their own component.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_degeneracy.py
from inference.facts import KB, Fact, Derived
from inference.degeneracy import IdentityClasses, condense


def _same_source(a, b):
    return Fact("same_source", (a, b), Derived("same_source_rule", ()))


def _derives(child, parent, off):
    return Fact("derives", (child, parent, off), Derived("derives_rule_placement", ()))


def test_identity_closure_unions_same_source_chain():
    kb = KB.empty()
    kb.add(_same_source("A", "B"))
    kb.add(_same_source("B", "C"))
    ic = IdentityClasses.from_kb(kb)
    assert ic.same_class("A", "C") is True
    assert ic.same_class("A", "D") is False
    parts = {frozenset(c) for c in ic.classes()}
    assert frozenset({"A", "B", "C"}) in parts


def test_derives_edges_excluded_from_identity():
    # nonzero offset proves NOT identical -> never same_class
    kb = KB.empty()
    kb.add(_derives("C", "S", 30))
    ic = IdentityClasses.from_kb(kb)
    assert ic.same_class("C", "S") is False


def test_condense_collapses_cycle_to_one_group():
    # lock round-trip: A -> B -> A is one irreducible group
    kb = KB.empty()
    kb.add(_derives("B", "A", 10))
    kb.add(_derives("A", "B", -10))
    kb.add(_derives("D", "C", 5))   # acyclic edge stays singletons
    comp, groups = condense(kb)
    multi = [g for g in groups if len(g) > 1]
    assert multi == [frozenset({"A", "B"})]
    assert comp["A"] == comp["B"] and comp["C"] != comp["D"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_degeneracy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.degeneracy'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/degeneracy.py
"""Degeneracy substrate: union-find identity closure + SCC condensation.

v1 model is pure equalities, so union-find (congruence closure over same_source) is
"obviously correct by reading it". The two-method interface (same_class / classes) is
the swap point for Z3, which arrives in Plan 3 with group disjunctions. The placement
graph (derives edges) may contain cycles (circular BD chains, lock round-trips,
ping-pong); condense() collapses each strongly-connected component into one
irreducible group so the reduction runs over the acyclic condensation.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, FrozenSet
from inference.facts import KB


class IdentityClasses:
    """Union-find over same_source edges only. derives edges are excluded:
    a non-zero offset is proof two events are not the same physical event."""

    def __init__(self):
        self._parent: Dict[str, str] = {}

    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def same_class(self, a: str, b: str) -> bool:
        if a not in self._parent or b not in self._parent:
            return a == b
        return self._find(a) == self._find(b)

    def classes(self) -> List[FrozenSet[str]]:
        buckets: Dict[str, set] = {}
        for x in self._parent:
            buckets.setdefault(self._find(x), set()).add(x)
        return [frozenset(s) for s in buckets.values()]

    @classmethod
    def from_kb(cls, kb: KB) -> "IdentityClasses":
        ic = cls()
        for f in kb.by_predicate("same_source"):
            ic.union(f.args[0], f.args[1])
        return ic


def condense(kb: KB) -> Tuple[Dict[str, int], List[FrozenSet[str]]]:
    """Tarjan SCC over directed derives edges (parent -> child)."""
    adj: Dict[str, List[str]] = {}
    nodes: set = set()
    for f in kb.by_predicate("derives"):
        child, parent = f.args[0], f.args[1]
        nodes.add(child); nodes.add(parent)
        adj.setdefault(parent, []).append(child)
        adj.setdefault(child, adj.get(child, []))

    index = {}
    low = {}
    on_stack = {}
    stack: List[str] = []
    counter = [0]
    comp_of: Dict[str, int] = {}
    groups: List[FrozenSet[str]] = []

    def strongconnect(v: str):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v); on_stack[v] = True
        for w in adj.get(v, []):
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            members = set()
            while True:
                w = stack.pop(); on_stack[w] = False
                members.add(w)
                if w == v:
                    break
            cid = len(groups)
            for m in members:
                comp_of[m] = cid
            groups.append(frozenset(members))

    for v in sorted(nodes):
        if v not in index:
            strongconnect(v)
    return comp_of, groups
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_degeneracy.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/degeneracy.py tools/test_inference_degeneracy.py
git commit -m "feat(#140): degeneracy substrate -- union-find identity + SCC condensation of cycles"
```

---

## Task 9: Verified reachability self-model

**Files:**
- Create: `tools/inference/reachability.py`
- Test: `tools/test_inference_reachability.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (a standalone artifact consumed by Tasks 10–11).
- Produces:
  - `Constraint(name: str, predicate: str, args: tuple, provenance_batch: str | None)`
    — a reachability limit; `provenance_batch` is the batch dir that *demonstrated* the
    limit (its `measured` provenance). `None` means undischarged.
  - `ReachabilityModel` with:
    - `.add_constraint(c: Constraint)`,
    - `.is_discharged(name) -> bool` (constraint exists and has a non-`None` provenance_batch),
    - `.can_separate(a, b) -> bool | None` — whether a batch can co-trace+distinguish the pair; `None` if unknown (no constraint either way),
    - `.blocking_constraints(a, b) -> list[Constraint]` — undischarged constraints an observational verdict on `(a,b)` would rest on.
  - `observational_blocked(model, a, b) -> bool` — True iff any constraint the
    `irreducible-by-instrument` verdict would rely on is undischarged (the self-sealing
    error guard: an observational verdict is blocked until every constraint it relies
    on carries measured provenance). The memmod row-2 confound is the first such
    constraint and ships as a fixture, not a hardcoded limit.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_reachability.py
from inference.reachability import (Constraint, ReachabilityModel,
                                    observational_blocked)


def test_discharged_constraint_has_provenance():
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod_row2", "cannot_cotrace",
                                ("1|2|0|X", "1|2|1|Y"), provenance_batch="batch_07"))
    assert m.is_discharged("memmod_row2") is True


def test_undischarged_constraint_blocks_observational_verdict():
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod_row2", "cannot_cotrace",
                                ("1|2|0|X", "1|2|1|Y"), provenance_batch=None))
    assert m.is_discharged("memmod_row2") is False
    assert observational_blocked(m, "1|2|0|X", "1|2|1|Y") is True


def test_discharged_constraint_does_not_block():
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod_row2", "cannot_cotrace",
                                ("1|2|0|X", "1|2|1|Y"), provenance_batch="batch_07"))
    assert observational_blocked(m, "1|2|0|X", "1|2|1|Y") is False


def test_can_separate_unknown_when_no_constraint():
    m = ReachabilityModel()
    assert m.can_separate("a", "b") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_reachability.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.reachability'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/reachability.py
"""The reachability self-model -- the one place the keystone could leak.

The observational degeneracy branch stops measuring based on "no reachable batch
separates them". If the model were incomplete, a separable pair would be misclassified
as irreducible and the error would be self-sealing. So every constraint is a
first-class verified artifact: it carries the batch that DEMONSTRATED the limit
(measured provenance), and an observational verdict is BLOCKED until every constraint
it relies on is discharged. The memmod row-2 confound is the first such constraint.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Constraint:
    name: str
    predicate: str            # e.g. "cannot_cotrace"
    args: Tuple
    provenance_batch: Optional[str]   # the batch dir that demonstrated the limit


class ReachabilityModel:
    def __init__(self):
        self._constraints: List[Constraint] = []

    def add_constraint(self, c: Constraint) -> None:
        self._constraints.append(c)

    def is_discharged(self, name: str) -> bool:
        return any(c.name == name and c.provenance_batch is not None
                   for c in self._constraints)

    def _relevant(self, a: str, b: str) -> List[Constraint]:
        pair = {a, b}
        return [c for c in self._constraints
                if pair & set(x for x in c.args)]

    def can_separate(self, a: str, b: str) -> Optional[bool]:
        rel = self._relevant(a, b)
        if not rel:
            return None
        # a discharged "cannot_cotrace"/"cannot_separate" constraint -> cannot separate
        for c in rel:
            if c.predicate.startswith("cannot") and c.provenance_batch is not None:
                return False
        return True

    def blocking_constraints(self, a: str, b: str) -> List[Constraint]:
        return [c for c in self._relevant(a, b)
                if c.predicate.startswith("cannot") and c.provenance_batch is None]


def observational_blocked(model: ReachabilityModel, a: str, b: str) -> bool:
    return len(model.blocking_constraints(a, b)) > 0
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_reachability.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/reachability.py tools/test_inference_reachability.py
git commit -m "feat(#140): verified reachability self-model -- constraints carry measured provenance"
```

---

## Task 10: Degeneracy trichotomy classifier + falsifiable gate

**Files:**
- Create: `tools/inference/classify.py`
- Test: `tools/test_inference_classify.py`

**Interfaces:**
- Consumes: `IdentityClasses` (Task 8); `ReachabilityModel`, `observational_blocked`
  (Task 9).
- Produces:
  - `Verdict` dataclass: `kind: str` in
    `{"structural-candidate", "unconfirmable-structural", "irreducible-by-instrument", "separable", "blocked"}`,
    `pair: tuple[str, str]`, `detail: dict`.
  - `classify_pair(a, b, identity: IdentityClasses, model: ReachabilityModel) -> Verdict`:
    1. If `identity.same_class(a, b)` → `structural-candidate` *unless* the falsifiable
       non-separation prediction is unrunnable: if `model.can_separate(a, b) is None`
       and there is no discharged constraint establishing non-separability, downgrade
       to `unconfirmable-structural` (we cite a structure but cannot run the
       experiment that would falsify it).
    2. Else if `observational_blocked(model, a, b)` → `blocked` (an undischarged
       reachability constraint; must not emit an observational verdict).
    3. Else if `model.can_separate(a, b)` is `False` → `irreducible-by-instrument`
       (finite enumeration found no reachable separating batch).
    4. Else (`can_separate` is `True` or `None` with no blocking) → `separable`
       (a reachable batch reaches a distinguishing edge → MEASURE-NEXT).

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_classify.py
from inference.facts import KB, Fact, Derived
from inference.degeneracy import IdentityClasses
from inference.reachability import Constraint, ReachabilityModel
from inference.classify import classify_pair, Verdict


def _ic_with(*edges):
    kb = KB.empty()
    for a, b in edges:
        kb.add(Fact("same_source", (a, b), Derived("same_source_rule", ())))
    return IdentityClasses.from_kb(kb)


def test_structural_candidate_when_same_class_and_confirmable():
    ic = _ic_with(("A", "B"))
    m = ReachabilityModel()
    m.add_constraint(Constraint("sep_AB", "can_separate", ("A", "B"),
                                provenance_batch="batch_03"))
    v = classify_pair("A", "B", ic, m)
    assert v.kind == "structural-candidate"


def test_unconfirmable_structural_when_no_experiment():
    ic = _ic_with(("A", "B"))
    m = ReachabilityModel()  # nothing known about separability of A,B
    v = classify_pair("A", "B", ic, m)
    assert v.kind == "unconfirmable-structural"


def test_blocked_when_undischarged_constraint():
    ic = _ic_with()  # not same class
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod", "cannot_cotrace", ("A", "B"),
                                provenance_batch=None))
    assert classify_pair("A", "B", ic, m).kind == "blocked"


def test_irreducible_when_discharged_cannot_separate():
    ic = _ic_with()
    m = ReachabilityModel()
    m.add_constraint(Constraint("memmod", "cannot_cotrace", ("A", "B"),
                                provenance_batch="batch_07"))
    assert classify_pair("A", "B", ic, m).kind == "irreducible-by-instrument"


def test_separable_when_reachable():
    ic = _ic_with()
    m = ReachabilityModel()
    m.add_constraint(Constraint("sep", "can_separate", ("A", "B"),
                                provenance_batch="batch_02"))
    assert classify_pair("A", "B", ic, m).kind == "separable"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.classify'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/classify.py
"""The degeneracy trichotomy: structural / observational / separable.

structurally degenerate -> structural-candidate (provisional until its falsifiable
non-separation prediction is run and confirmed) or unconfirmable-structural (the
confirmation batch is itself unreachable -- a distinct honest finding, never a
silently-trusted collapse). observationally degenerate -> irreducible-by-instrument
(finite enumeration over the verified self-model found no reachable separating batch).
separable -> MEASURE-NEXT. An observational verdict is blocked until its reachability
constraints are discharged.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from inference.degeneracy import IdentityClasses
from inference.reachability import ReachabilityModel, observational_blocked


@dataclass
class Verdict:
    kind: str
    pair: Tuple[str, str]
    detail: dict


def classify_pair(a: str, b: str, identity: IdentityClasses,
                  model: ReachabilityModel) -> Verdict:
    pair = (a, b)
    if identity.same_class(a, b):
        sep = model.can_separate(a, b)
        if sep is None and not _has_discharged_nonsep(model, a, b):
            return Verdict("unconfirmable-structural", pair,
                           {"why": "non-separation prediction unrunnable"})
        return Verdict("structural-candidate", pair,
                       {"why": "identity collapse over audited same_source edges",
                        "gate": "run non-separation prediction to confirm"})
    if observational_blocked(model, a, b):
        return Verdict("blocked", pair,
                       {"why": "undischarged reachability constraint",
                        "constraints": [c.name for c in
                                        model.blocking_constraints(a, b)]})
    if model.can_separate(a, b) is False:
        return Verdict("irreducible-by-instrument", pair,
                       {"why": "no reachable batch separates them"})
    return Verdict("separable", pair, {"why": "reachable distinguishing batch exists"})


def _has_discharged_nonsep(model: ReachabilityModel, a: str, b: str) -> bool:
    return model.can_separate(a, b) is False
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_classify.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/classify.py tools/test_inference_classify.py
git commit -m "feat(#140): degeneracy trichotomy classifier with falsifiable structural gate"
```

---

## Task 11: Planner — proven-gain MEASURE-NEXT + per-tile mode + Phase-0 seed

**Files:**
- Create: `tools/inference/planner.py`
- Test: `tools/test_inference_planner.py`

**Interfaces:**
- Consumes: `KB` (Task 2); `correlates` (Task 4); `ReachabilityModel` (Task 9).
- Produces:
  - `Batch` dataclass: `tiles: dict[str, list[str]]` (`"col|row|pkt" -> [event_names]`,
    the `configure_batch` shape) and `mode_by_tile: dict[str, int]` (per-tile trace
    mode: cores can use EVENT_PC mode 1, memmod/memtile/shim are always mode 0).
  - `MEASURE_NEXT` / `NO_GAIN` sentinels.
  - `plan_cotrace(a, b, mode_by_tile=None) -> Batch` — a batch that co-traces a pair
    never before in one batch (the genuine-gain case). Per-tile mode defaults to 0
    unless the tile is a core (row != 0 and pkt == 0) and `mode_by_tile` overrides.
  - `propose_next(kb, run_dirs, pair, model, anchor_key=ANCHOR, eps=EPS) -> Batch | object`
    — proven-gain gate: returns a `Batch` only if it *first proves* a reachable batch
    adds a separating/co-tracing gain; a **fully-measured** tight `correlates` pair
    with a stable offset and no orientation returns `NO_GAIN` (straight to
    observational degeneracy without burning a batch). Never emit-then-discover.
  - `seed_plan(configured_events: list[str]) -> Batch` — Phase 0: the initial coverage
    sweep batch over the static configured-event set (the ranking function's top
    component tracks its progress against this set).

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_planner.py
import json
from inference.facts import KB
from inference.reachability import ReachabilityModel, Constraint
from inference.planner import (plan_cotrace, propose_next, seed_plan, Batch,
                               NO_GAIN)


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for (tile, name), delta in off.items():
            c, r, p = tile.split("|")
            evs.append(_ev(int(c), int(r), name, 1000 + delta, pkt_type=int(p)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_plan_cotrace_groups_events_by_tile():
    b = plan_cotrace("1|0|0|DMA", "1|1|3|PORT")
    assert b.tiles["1|0|0"] == ["DMA"]
    assert b.tiles["1|1|3"] == ["PORT"]
    assert b.mode_by_tile["1|0|0"] == 0


def test_plan_cotrace_core_tile_mode_override():
    b = plan_cotrace("1|2|0|INSTR", "1|0|0|DMA",
                     mode_by_tile={"1|2|0": 1})   # core can use EVENT_PC
    assert b.mode_by_tile["1|2|0"] == 1
    assert b.mode_by_tile["1|0|0"] == 0


def test_propose_next_no_gain_for_measured_stable_unoriented_pair(tmp_path):
    # C = S + 30 every run, fully co-traced, no config_path -> NO_GAIN
    dirs = _runs(tmp_path, [{("1|0|0", "S"): 100, ("1|0|0", "C"): 130},
                            {("1|0|0", "S"): 220, ("1|0|0", "C"): 250}])
    kb = KB.empty()
    m = ReachabilityModel()
    assert propose_next(kb, dirs, ("1|0|0|C", "1|0|0|S"), m) is NO_GAIN


def test_propose_next_emits_cotrace_for_never_cotraced_pair(tmp_path):
    # X and Y were never in one batch -> co-trace gain
    dirs = _runs(tmp_path, [{("1|0|0", "X"): 100}])  # Y absent
    kb = KB.empty()
    m = ReachabilityModel()
    m.add_constraint(Constraint("sep", "can_separate", ("1|0|0|X", "1|1|3|Y"),
                                provenance_batch="b0"))
    b = propose_next(kb, dirs, ("1|0|0|X", "1|1|3|Y"), m)
    assert isinstance(b, Batch)
    assert "1|1|3" in b.tiles


def test_seed_plan_covers_configured_events():
    b = seed_plan(["1|0|0|DMA", "1|2|0|INSTR", "1|1|3|PORT"])
    assert b.tiles["1|0|0"] == ["DMA"]
    assert b.tiles["1|2|0"] == ["INSTR"]
    assert b.tiles["1|1|3"] == ["PORT"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_planner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.planner'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/planner.py
"""The planner: proven-gain MEASURE-NEXT batches.

Turns an ambiguity into a batch ONLY after first proving the batch adds a
separating/co-tracing gain -- never emit-then-discover. A fully-measured tight
correlates pair with a stable offset and no orientation returns NO_GAIN and goes
straight to observational degeneracy without burning a batch. Carries per-tile mode
on the write side (cores can use EVENT_PC mode 1; memmod/memtile/shim are always mode
0) and a Phase-0 seed sweep over the static configured-event set.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from inference.facts import KB
from inference.verifier import correlates, ANCHOR, EPS
from inference.reachability import ReachabilityModel

MEASURE_NEXT = object()
NO_GAIN = object()


@dataclass
class Batch:
    tiles: Dict[str, List[str]] = field(default_factory=dict)
    mode_by_tile: Dict[str, int] = field(default_factory=dict)


def _tile_of(event_key: str) -> Tuple[str, str]:
    col, row, pkt, name = event_key.split("|")
    return f"{col}|{row}|{pkt}", name


def _default_mode(tile_key: str) -> int:
    col, row, pkt = tile_key.split("|")
    # Only cores (compute tiles, pkt-type 0, row != 0) support EVENT_PC; default 0.
    return 0


def plan_cotrace(a: str, b: str,
                 mode_by_tile: Optional[Dict[str, int]] = None) -> Batch:
    mode_by_tile = mode_by_tile or {}
    batch = Batch()
    for ek in (a, b):
        tile, name = _tile_of(ek)
        batch.tiles.setdefault(tile, [])
        if name not in batch.tiles[tile]:
            batch.tiles[tile].append(name)
        batch.mode_by_tile[tile] = mode_by_tile.get(tile, _default_mode(tile))
    return batch


def _co_traced(run_dirs: List[str], a: str, b: str,
               anchor_key: str, eps: float) -> bool:
    return correlates(run_dirs, a, b, anchor_key, eps) is not None


def propose_next(kb: KB, run_dirs: List[str], pair: Tuple[str, str],
                 model: ReachabilityModel, anchor_key: str = ANCHOR,
                 eps: float = EPS):
    a, b = pair
    if not _co_traced(run_dirs, a, b, anchor_key, eps):
        # never co-traced: genuine co-trace gain, but only if reachable
        if model.can_separate(a, b) is False:
            return NO_GAIN
        return plan_cotrace(a, b)
    # fully measured, stable offset, but unoriented -> no batch can add orientation
    # (orientation comes from config, not measurement) -> observational degeneracy
    return NO_GAIN


def seed_plan(configured_events: List[str]) -> Batch:
    batch = Batch()
    for ek in configured_events:
        tile, name = _tile_of(ek)
        batch.tiles.setdefault(tile, [])
        if name not in batch.tiles[tile]:
            batch.tiles[tile].append(name)
        batch.mode_by_tile[tile] = _default_mode(tile)
    return batch
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_planner.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/planner.py tools/test_inference_planner.py
git commit -m "feat(#140): proven-gain planner -- MEASURE-NEXT, per-tile mode, Phase-0 seed"
```

---

## Task 12: Closed loop + three-component termination + mock instrument

**Files:**
- Create: `tools/inference/loop.py`
- Test: `tools/test_inference_loop.py`

**Interfaces:**
- Consumes: `chain`, `classify_events` (Task 7); `propose_next`, `NO_GAIN`, `Batch`,
  `seed_plan` (Task 11).
- Produces:
  - `Instrument` protocol (informal): `.capture(batch: Batch) -> str` returns a new
    run-dir path (or extends batches under existing run-dirs) with the requested events
    traced. Provided concretely by `engine.py` (Task 13) over the real actuator.
  - `MockInstrument(ground_truth)` — a synthetic instrument: holds a known
    ground-truth model (per-event anchored offsets and a config-route map) and writes
    `batch_NN/hw/trace.events.json` for requested events on `.capture(batch)`.
  - `ranking(kb, configured_events, candidate_pairs) -> tuple[int, int, int]` — the
    three-component lexicographic measure `(# configured-but-unfired, # unresolved
    fired, # untested candidate edges)`; the top component's domain is the static
    `configured_events` set.
  - `run_loop_until_converged(instrument, configured_events, candidate_pairs, max_iters=50) -> dict`
    — chain → inspect stall → act (`MEASURE-NEXT` via instrument, or stop). Returns a
    report dict `{"converged": bool, "iterations": int, "rankings": [tuple,...],
    "classification": {event: label}}`. Asserts (internally) the ranking strictly
    decreases lexicographically each iteration; raises `RuntimeError` on non-decrease
    (livelock guard).

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_loop.py
from inference.loop import (MockInstrument, ranking, run_loop_until_converged)


def test_ranking_top_component_counts_unfired():
    # 3 configured, only 1 fired -> top = 2
    fired = {"1|0|0|A"}
    configured = ["1|0|0|A", "1|0|0|B", "1|1|3|C"]
    r = ranking_fixture(fired, configured)
    assert r[0] == 2


def ranking_fixture(fired_set, configured):
    from inference.facts import KB, Fact, Measured
    kb = KB.empty()
    for ek in fired_set:
        kb.add(Fact("fired", (ek, 0, 5), Measured()))
    return ranking(kb, configured, candidate_pairs=[])


def test_loop_converges_on_mock_ground_truth(tmp_path):
    # ground truth: S stochastic root, C = S + 30 derived (config routes S->C)
    gt = {
        "events": {"1|2|0|PERF_CNT_2": {"base": 0, "jitter": 0},
                   "1|0|0|S": {"base": 100, "jitter": 50},
                   "1|0|0|C": {"base": 130, "jitter": 50}},  # C tracks S (=S+30)
        "routes": [("1|0|0|S", "1|0|0|C")],
        "workdir": str(tmp_path)}
    inst = MockInstrument(gt)
    report = run_loop_until_converged(
        inst,
        configured_events=["1|2|0|PERF_CNT_2", "1|0|0|S", "1|0|0|C"],
        candidate_pairs=[("1|0|0|C", "1|0|0|S")])
    assert report["converged"] is True
    assert report["classification"]["1|0|0|C"] == "derived"
    assert report["classification"]["1|0|0|S"] == "stochastic_root"


def test_ranking_strictly_decreases_with_discovery(tmp_path):
    # a discovery step surfaces a never-before-fired event: middle count may rise,
    # but the TOP component strictly decreases (unfired -> fired).
    gt = {
        "events": {"1|2|0|PERF_CNT_2": {"base": 0, "jitter": 0},
                   "1|0|0|S": {"base": 100, "jitter": 50},
                   "1|0|0|C": {"base": 130, "jitter": 50}},
        "routes": [("1|0|0|S", "1|0|0|C")],
        "reveal_on_iter": {1: "1|0|0|C"},   # C only appears after a discovery batch
        "workdir": str(tmp_path)}
    inst = MockInstrument(gt)
    report = run_loop_until_converged(
        inst,
        configured_events=["1|2|0|PERF_CNT_2", "1|0|0|S", "1|0|0|C"],
        candidate_pairs=[("1|0|0|C", "1|0|0|S")])
    tops = [r[0] for r in report["rankings"]]
    assert tops == sorted(tops, reverse=True)   # top never increases
    assert tops[0] > tops[-1]                    # and strictly decreased overall
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.loop'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/loop.py
"""The closed loop: chain to fixpoint, act on the stall, converge or halt.

Termination rests on a three-component lexicographic measure
(# configured-but-unfired events, # unresolved fired events, # untested candidate
edges); the top component's domain is the static configured-event set from the
xclbin. A discovery batch that surfaces a never-before-fired event raises the middle
count but strictly DECREASES the top (an event moves unfired->fired). No livelock
branch exists -- "ambiguous and no batch separates" is degeneracy (it halts).

MockInstrument provides synthetic ground truth for Axis-1 convergence tests; the real
instrument (engine.py) drives the actuator.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
from inference.facts import KB
from inference.loader import load_fired
from inference.ledger import install_ledger
from inference.chainer import chain, classify_events
from inference.planner import propose_next, seed_plan, NO_GAIN, Batch
from inference.reachability import ReachabilityModel, Constraint
from inference.verifier import ANCHOR


class MockInstrument:
    """Synthetic instrument over a known ground-truth model.

    gt = {"events": {event_key: {"base": int, "jitter": int}},
          "routes": [(producer_key, consumer_key), ...],
          "reveal_on_iter": {iter_index: event_key},   # optional discovery
          "workdir": str}
    Each .capture(batch) writes a new run-dir set (n_runs) tracing the requested
    events, sampling jitter deterministically per run index.
    """

    def __init__(self, gt: dict, n_runs: int = 6):
        self.gt = gt
        self.n_runs = n_runs
        self._iter = 0
        self._revealed = set()

    def ledger_entries(self) -> List[dict]:
        return [{"cite": f"route#{i}", "a": a, "b": b, "kind": "route"}
                for i, (a, b) in enumerate(self.gt.get("routes", []))]

    def capture(self, batch: Batch) -> List[str]:
        reveal = self.gt.get("reveal_on_iter", {})
        if self._iter in reveal:
            self._revealed.add(reveal[self._iter])
        requested = {f"{tile}|{name}"
                     for tile, names in batch.tiles.items() for name in names}
        # Always include the anchor so anchoring works.
        requested.add(ANCHOR)
        # Withhold not-yet-revealed events.
        hidden = {k for k in self.gt.get("reveal_on_iter", {}).values()
                  if k not in self._revealed}
        visible = [k for k in requested
                   if k in self.gt["events"] and k not in hidden]
        run_dirs = []
        base_dir = Path(self.gt["workdir"]) / f"capture_{self._iter:02d}"
        for run in range(self.n_runs):
            evs = []
            for ek in visible:
                col, row, pkt, name = ek.split("|")
                spec = self.gt["events"][ek]
                # Deterministic per-run jitter; co-derived events share the run's draw.
                draw = (run * 37) % 100 if spec["jitter"] else 0
                soc = 1000 + spec["base"] + (draw if spec["jitter"] else 0)
                evs.append({"col": int(col), "row": int(row), "pkt_type": int(pkt),
                            "slot": 0, "name": name, "ts": soc, "soc": soc, "mode": 0})
            rd = base_dir / f"run{run}"
            (rd / "batch_00" / "hw").mkdir(parents=True, exist_ok=True)
            (rd / "batch_00" / "hw" / "trace.events.json").write_text(
                json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
            run_dirs.append(str(rd))
        self._iter += 1
        return run_dirs


def ranking(kb: KB, configured_events: List[str],
            candidate_pairs: List[Tuple[str, str]]) -> Tuple[int, int, int]:
    fired = {f.args[0] for f in kb.by_predicate("fired")}
    unfired = sum(1 for e in configured_events if e not in fired)
    cls = classify_events(kb, sorted(fired))
    unresolved = sum(1 for e, c in cls.items() if c == "unresolved")
    derived_pairs = {(f.args[0], f.args[1]) for f in kb.by_predicate("derives")}
    untested = sum(1 for p in candidate_pairs if p not in derived_pairs)
    return (unfired, unresolved, untested)


def run_loop_until_converged(instrument, configured_events: List[str],
                             candidate_pairs: List[Tuple[str, str]],
                             max_iters: int = 50) -> dict:
    kb = KB.empty()
    install_ledger(kb, {e["cite"]: e for e in instrument.ledger_entries()})
    model = ReachabilityModel()
    all_run_dirs: List[str] = []
    rankings: List[Tuple[int, int, int]] = []

    # Phase 0: seed sweep over the static configured-event set.
    seed = seed_plan(configured_events)
    all_run_dirs += instrument.capture(seed)

    prev = None
    for _ in range(max_iters):
        kb_iter = KB.empty()
        install_ledger(kb_iter, {e["cite"]: e for e in instrument.ledger_entries()})
        for f in load_fired(all_run_dirs):
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
        unfired = [e for e in configured_events if e not in fired]
        if not unresolved and not unfired:
            return {"converged": True, "iterations": len(rankings),
                    "rankings": rankings, "classification": cls}

        # Act on the stall: propose the next measurement (proven gain only).
        progressed = False
        for pair in candidate_pairs:
            batch = propose_next(kb, all_run_dirs, pair, model)
            if batch is not NO_GAIN:
                all_run_dirs += instrument.capture(batch)
                progressed = True
                break
        if not progressed and unfired:
            # discovery: re-seed to surface not-yet-fired configured events
            all_run_dirs += instrument.capture(seed_plan(unfired))
            progressed = True
        if not progressed:
            return {"converged": False, "iterations": len(rankings),
                    "rankings": rankings, "classification": cls}

    return {"converged": False, "iterations": len(rankings),
            "rankings": rankings, "classification": cls}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_loop.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/inference/loop.py tools/test_inference_loop.py
git commit -m "feat(#140): closed loop + 3-component termination + mock instrument"
```

---

## Task 13: Engine orchestration + CLI + end-to-end synthetic

**Files:**
- Create: `tools/inference/engine.py`
- Test: `tools/test_inference_engine.py`

**Interfaces:**
- Consumes: everything — `KB`, `install_ledger`/`load_ledger`, `load_fired`,
  `replication_violations`, `chain`, `classify_events`, `IdentityClasses`, `condense`,
  `classify_pair`, `ReachabilityModel`.
- Produces:
  - `run_engine(run_dirs, ledger_path, candidate_pairs, anchor_key=ANCHOR) -> dict` —
    static (no actuator) reconstruction over already-captured data + a ledger:
    loads fired + ledger, checks replication, chains to fixpoint, condenses cycles,
    classifies degeneracy for coincident root pairs, returns a **placement report**:
    `{"replication_violations": [...], "classification": {event: label},
      "derives": [(child, parent, offset)...], "stochastic_roots": [...],
      "irreducible_groups": [frozenset...], "degeneracy": [Verdict...],
      "provenance_ok": bool}`.
  - `main(argv=None)` — CLI: `python -m inference.engine --runs DIR [DIR...]
    --ledger FILE --pairs A:B [A:B...]`; prints the report as JSON.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_engine.py
import json
from inference.engine import run_engine


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for (tile, name), delta in off.items():
            c, r, p = tile.split("|")
            evs.append(_ev(int(c), int(r), name, 1000 + delta, pkt_type=int(p)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def _ledger(tmp_path, entries):
    p = tmp_path / "led.json"
    p.write_text(json.dumps({"entries": entries}))
    return str(p)


def test_engine_reconstructs_placement(tmp_path):
    dirs = _runs(tmp_path, [
        {("1|0|0", "S"): 100, ("1|0|0", "C"): 130},
        {("1|0|0", "S"): 240, ("1|0|0", "C"): 270},
        {("1|0|0", "S"): 310, ("1|0|0", "C"): 340}])
    led = _ledger(tmp_path, [
        {"cite": "r1", "a": "1|0|0|S", "b": "1|0|0|C", "kind": "route"}])
    rep = run_engine(dirs, led, [("1|0|0|C", "1|0|0|S")])
    assert rep["provenance_ok"] is True
    assert rep["replication_violations"] == []
    assert ("1|0|0|C", "1|0|0|S", 30) in rep["derives"]
    assert "1|0|0|S" in rep["stochastic_roots"]
    assert rep["classification"]["1|0|0|C"] == "derived"


def test_engine_reports_irreducible_cycle(tmp_path):
    # A and B each derive the other (lock round-trip) -> one irreducible group
    dirs = _runs(tmp_path, [
        {("1|0|0", "A"): 100, ("1|0|0", "B"): 110},
        {("1|0|0", "A"): 200, ("1|0|0", "B"): 210}])
    led = _ledger(tmp_path, [
        {"cite": "r1", "a": "1|0|0|A", "b": "1|0|0|B", "kind": "route"},
        {"cite": "r2", "a": "1|0|0|B", "b": "1|0|0|A", "kind": "route"}])
    rep = run_engine(dirs, led, [("1|0|0|B", "1|0|0|A"), ("1|0|0|A", "1|0|0|B")])
    multi = [g for g in rep["irreducible_groups"] if len(g) > 1]
    assert frozenset({"1|0|0|A", "1|0|0|B"}) in multi
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd tools && python -m pytest test_inference_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.engine'`.

- [ ] **Step 3: Write the implementation**

```python
# tools/inference/engine.py
"""Top-level engine: static placement reconstruction over captured data + a ledger.

Loads fired (measured) + ledger (structural), checks replication, chains to fixpoint
under the verified rules, condenses placement cycles into irreducible groups, and
classifies degeneracy for coincident root pairs. Returns a placement report. The CLI
runs it over already-captured batch dirs; the closed loop (loop.py) drives the
actuator for MEASURE-NEXT.
"""
from __future__ import annotations
import argparse
import json
import sys
from typing import Dict, List, Tuple
from inference.facts import KB, provenance_ok
from inference.ledger import load_ledger, install_ledger
from inference.loader import load_fired, replication_violations
from inference.chainer import chain, classify_events
from inference.degeneracy import IdentityClasses, condense
from inference.classify import classify_pair
from inference.reachability import ReachabilityModel
from inference.verifier import ANCHOR, coincident


def run_engine(run_dirs: List[str], ledger_path: str,
               candidate_pairs: List[Tuple[str, str]],
               anchor_key: str = ANCHOR) -> dict:
    kb = KB.empty()
    install_ledger(kb, load_ledger(ledger_path))
    for f in load_fired(run_dirs, anchor_key):
        kb.add(f)

    reps = replication_violations(run_dirs, anchor_key)
    kb = chain(run_dirs, kb, candidate_pairs, anchor_key)

    fired = sorted({f.args[0] for f in kb.by_predicate("fired")})
    cls = classify_events(kb, fired)
    derives = [f.args for f in kb.by_predicate("derives")]
    roots = [e for e in fired if cls.get(e) == "stochastic_root"]

    _comp, groups = condense(kb)
    irreducible = [g for g in groups]

    # Degeneracy classification for coincident root pairs.
    identity = IdentityClasses.from_kb(kb)
    model = ReachabilityModel()
    verdicts = []
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            a, b = roots[i], roots[j]
            if coincident(run_dirs, a, b, anchor_key):
                verdicts.append(classify_pair(a, b, identity, model).__dict__)

    return {"replication_violations": reps,
            "classification": cls,
            "derives": derives,
            "stochastic_roots": roots,
            "irreducible_groups": irreducible,
            "degeneracy": verdicts,
            "provenance_ok": provenance_ok(kb)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Trace inference engine (static report).")
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs")
    ap.add_argument("--ledger", required=True, help="structural ledger JSON")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="candidate child:parent pairs")
    ap.add_argument("--anchor", default=ANCHOR)
    args = ap.parse_args(argv)
    pairs = [tuple(p.split(":", 1)) for p in args.pairs]
    rep = run_engine(args.runs, args.ledger, pairs, args.anchor)
    # frozensets are not JSON-serializable -> list them
    rep["irreducible_groups"] = [sorted(g) for g in rep["irreducible_groups"]]
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd tools && python -m pytest test_inference_engine.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the whole inference suite + CLI smoke**

Run: `cd tools && python -m pytest test_inference_*.py -v`
Expected: PASS (all inference tests). Then a CLI smoke over a fixture:
Run: `cd tools && python -m inference.engine --help`
Expected: prints usage without error.

- [ ] **Step 6: Commit**

```bash
git add tools/inference/engine.py tools/test_inference_engine.py
git commit -m "feat(#140): engine orchestration + CLI -- placement report over captured data"
```

---

## Task 14: Axis-2 HW smoke on `add_one_using_dma` (hand-authored ledger)

> **HW-GATED.** This task runs on the real NPU. Follow the operational rules: never
> two HW suites concurrently; rebuild the `.so` if the plugin path is exercised (not
> needed here — this drives the bridge-trace-runner directly via the actuator); HW
> invocations use `env -u XDNA_EMU -u XDNA_EMU_RUNTIME`; if the NPU wedges, hand the
> recovery to Maya. **Capture is expensive — run the capture once, then iterate the
> engine offline against the saved batch dirs.**

**Files:**
- Create: `tools/inference/fixtures/add_one_using_dma.ledger.json` (hand-authored)
- Test: `tools/test_inference_hw_smoke.py` (marked `hw`, skipped without NPU)

**Interfaces:**
- Consumes: `run_engine` (Task 13); the capture engine (`trace_capture.run_loop`) over
  the extracted `trace_runner` (Task 1).

The goal (spec Axis 2): on `add_one_using_dma`, independently **re-derive the
validation's 5 stochastic roots and the deterministic backbone**, and take the first
`same_source` candidate (shim-DMA-seen-at-memtile) through the **full gate** — emit
the non-separation prediction, run it, confirm before trusting the structural verdict.
In Plan 1 the structural ledger is **hand-authored** (Plan 2 automates it).

- [ ] **Step 1: Hand-author the structural ledger fixture**

Read the precursor validation
(`docs/superpowers/findings/2026-06-17-capture-engine-validation.md`) for the 5
stochastic roots, the deterministic backbone, and the shim-DMA→memtile identity
candidate. Encode only the routes/identity that `add_one_using_dma` actually
configures, each cited to the config location named in the finding:

```json
{
  "entries": [
    {"cite": "ss:shim0->memtile-route",
     "a": "0|0|2|DMA_MM2S_0_START_TASK",
     "b": "0|1|3|PORT_RUNNING_0",
     "kind": "route"},
    {"cite": "identity:shimdma-at-memtile",
     "a": "0|0|2|DMA_MM2S_0_FINISHED_TASK",
     "b": "0|1|3|PORT_RUNNING_0",
     "kind": "identity"}
  ]
}
```

(Use the exact `event_key`s the capture emits for this kernel — verify against a
captured `batch_NN/hw/trace.events.json` before finalizing; correct the placeholders
above to the real producer/consumer keys.)

- [ ] **Step 2: Write the HW-gated smoke test**

```python
# tools/test_inference_hw_smoke.py
"""Axis-2: engine output matches silicon on add_one_using_dma. HW-gated."""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XDNA_HW_SMOKE") != "1",
    reason="HW smoke requires a real NPU; set XDNA_HW_SMOKE=1 to run")


def test_engine_rederives_roots_on_add_one_using_dma():
    # Captured batch dirs produced by Step 3 (saved under build/experiments/).
    from pathlib import Path
    from inference.engine import run_engine
    cap = Path(os.environ["XDNA_SMOKE_RUNS"])          # dir holding run0..runN
    run_dirs = sorted(str(p) for p in cap.glob("run*"))
    ledger = str(Path(__file__).resolve().parent
                 / "inference" / "fixtures" / "add_one_using_dma.ledger.json")
    rep = run_engine(run_dirs, ledger, candidate_pairs=[])
    # Re-derive the validation's count of stochastic roots (5).
    assert len(rep["stochastic_roots"]) == 5
    assert rep["provenance_ok"] is True
    assert rep["replication_violations"] == []
```

- [ ] **Step 3: Capture real data once (HW), then run the smoke offline**

Capture `add_one_using_dma` trace batches via the capture engine (saved under
`build/experiments/`, never `/tmp`):

```bash
cd /home/triple/npu-work/xdna-emu
env -u XDNA_EMU -u XDNA_EMU_RUNTIME \
  python tools/trace_capture.py --test add_one_using_dma \
  --out build/experiments/infer-smoke --runs 6 \
  > build/experiments/infer-smoke/capture.log 2>&1
```

(Adjust to `trace_capture.py`'s real CLI; if it has no `__main__`, write a 10-line
driver under `tools/` that calls `run_loop(...)`. Confirm `batch_NN/hw/trace.events.json`
appear under each `run*`.) Then run the engine offline against the saved dirs:

```bash
cd tools && XDNA_HW_SMOKE=1 \
  XDNA_SMOKE_RUNS=../build/experiments/infer-smoke \
  python -m pytest test_inference_hw_smoke.py -v
```

Expected: PASS — 5 stochastic roots re-derived, provenance sound, replication clean.
If the count differs, **do not edit the assertion to match** — investigate per
systematic-debugging (the finding is the oracle; a mismatch is either a ledger error
or a genuine engine bug).

- [ ] **Step 4: Take the `same_source` candidate through the full gate (HW)**

Run the engine with the identity pair as a candidate; when it returns a
`structural-candidate` verdict, emit its non-separation prediction batch, capture it,
and confirm non-separation before trusting the collapse (spec Section 2b gate). Record
the outcome in the finding doc (Step 5). This step exercises the
`structural-candidate` → confirmed path end-to-end against silicon.

- [ ] **Step 5: Write the validation finding + commit**

Create `docs/superpowers/findings/2026-06-21-inference-engine-axis2-smoke.md`
recording: the captured run set, the 5 re-derived roots vs the precursor's, the
identity-gate outcome, and any ledger corrections. Then:

```bash
git add tools/inference/fixtures/add_one_using_dma.ledger.json \
        tools/test_inference_hw_smoke.py \
        docs/superpowers/findings/2026-06-21-inference-engine-axis2-smoke.md
git commit -m "test(#140): Axis-2 HW smoke -- re-derive add_one_using_dma roots, gate identity candidate"
```

---

## Self-Review (completed by plan author)

**Spec coverage** — every spec section maps to a task:
- Placement-not-causation / observational regime → Task 6 (`derives` admission, no
  causal label; backpressure test) + Global Constraints.
- Keystone (three support types, no unaudited axioms) → Task 2 (`Support`,
  `provenance_ok`) + Task 7 (`chaining_sound`).
- Build-not-adopt (union-find v1, two-method interface) → Task 8.
- Section 1 predicate table → `fired` (T3), `config_path`/`identity` (T5),
  `deterministic`/`stochastic` (T6), `correlates`/`coincident` (T4),
  `derives`/`same_source`/`stochastic_root` (T6). *Groups predicates
  (`group_fired`/`group_member`) are Plan 3 — explicitly deferred.*
- Section 2 loop + 3-component ranking → Task 12.
- Section 2b degeneracy (union-find, SCC condensation, identity-vs-placement
  separation, falsifiable gate, unconfirmable-structural) → Tasks 8, 10.
- Section 3 stack + folds: read layer (existing decoder, untouched), actuator
  (Task 1 extraction), verifier (T4), inference (T6–T8), planner (T11); folds —
  mode-threading per-tile (T11), seed/Phase-0 (T11/T12), memmod row-2 as the first
  reachability constraint (T9). *Groups fold = Plan 3.*
- Section 4 testing two axes → Axis 1 across T2–T13 (chaining-soundness T7,
  leaf-validity replication T3 + structural-citation T5, rule-soundness T4/T6,
  degeneracy oracle T8/T10, closed-loop convergence+termination T12); Axis 2 → T14.
- Reachability self-model (keystone leak guard) → Task 9 + `observational_blocked`.
- Open questions (eps calibration, config_path derivation, ledger format, groups
  specifics) → noted as deferred; eps is a hard global 2.0 in v1 per Global
  Constraints; config_path *extraction* is Plan 2; groups are Plan 3.

**Placeholder scan** — one intentional placeholder remains: Task 14 Step 1's ledger
`event_key`s are marked "correct against a real captured batch" because the exact
producer/consumer keys for `add_one_using_dma` must be read from real capture output,
not guessed. Every code step for Tasks 1–13 contains complete, runnable code.

**Type consistency** — `event_key` is the 4-part `col|row|pkt|name` string everywhere;
`anchor_key`/`ANCHOR = "1|2|0|PERF_CNT_2"`; `EPS = 2.0`; `Fact(predicate, args,
support)` with `args` a tuple throughout; `Batch.tiles` is `{"col|row|pkt":[names]}`
matching `configure_batch`; `correlates` returns `int|None`; the degeneracy interface
is `same_class`/`classes` in both Task 8 and its consumers (Tasks 10, 13). `derives`
args are `(child, parent, offset)` consistently across Tasks 6, 7, 8, 13.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-21-trace-inference-engine.md`.**

This is **Plan 1 of 3**: the v1 engine machinery (RunnerSession extraction + verifier +
chainer + union-find degeneracy + planner + closed loop), fully Axis-1 TDD'd, with
`config_path` consumed as an input ledger and a hand-authored Axis-2 HW smoke. Plan 2
(automated config extraction + full Axis-2) and Plan 3 (groups + Z3) follow.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review
   between tasks, fast iteration. Tasks 1–13 are HW-free and parallel-safe to review;
   Task 14 is HW-gated and runs last under the operational rules.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch
   execution with checkpoints for review.

Which approach?
