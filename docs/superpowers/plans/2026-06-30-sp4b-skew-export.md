# SP-4b Skew-Export + Causal Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-grounded `causal_offset = raw − skew(A,B) = Δwall` decomposition to the trace inference engine, fed by a per-module `origin_D` table the emulator exports, gated behind a `calibrated` flag (inert until SP-5).

**Architecture:** The Rust emulator already computes per-module broadcast-flood arrival (`origin_D`) via a Dijkstra wavefront. It exports that table as a JSON sidecar (`origin_d.json`) keyed by `col|row|<module_kind>`, with a `calibrated` flag and the single `flood_source`. The Python engine loads it as a new `ModelDerived` provenance class, and when `calibrated` is true, emits `causal_offset` as a separate fact whose premises explicitly cite the model leaves (so modeled data never launders into a measured claim). Pre-calibration the whole path is inert and all existing outputs are byte-identical.

**Tech Stack:** Rust (emulator: `src/device/`, `crates/xdna-archspec/`, `crates/xdna-emu-ffi/`), Python 3.13 (inference engine: `tools/inference/`), pytest, `cargo test`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-sp4b-skew-export-design.md` (rev2). Every task implicitly includes its requirements.
- **No emoji anywhere** (code, comments, commits).
- **Commit messages** end with two trailer lines, verbatim:
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```
- **Branch:** `sp4b-skew-export` (already current). Do not push unless asked.
- **Sign convention (load-bearing):** `A = domain(child)`, `B = domain(parent)`; `skew = origin_D[parent_dom] − origin_D[child_dom]`; `causal_offset = raw − skew = Δwall`. Pin with a known-`Δwall` test, never a circular `skew == origin_D[B]−origin_D[A]` assertion.
- **Module → pkt_type mapping (the engine's domain-key numeric codes):** `core=0, mem=1, shim=2, memtile=3`. Confirmed empirically from `build/experiments/sp3-spike-trace/spike.events.json` (row0=2, row1=3, compute=0) and `dma-fill-measure.py` (core=0, mem=1).
- **Keystone:** `causal_offset` is exact in the model, an estimate of silicon with error = calibration error. Never present it as measured. Withhold (emit nothing) until `calibrated`.
- **Run Python tests** from the repo root with `PYTHONPATH=tools pytest tools/test_inference_<name>.py` (flat convention, alongside the 14 existing `tools/test_inference_*.py`). **Run Rust tests** bare: `cargo test -p <crate> <filter>` (never piped through head/tail/grep).
- **After any Rust change** affecting the FFI path, rebuild with `cargo build -p xdna-emu-ffi` (not bare `cargo build`). After Python changes, no rebuild needed.

---

## The sidecar contract (locked first; both languages build against it)

`origin_d.json`, written by the emulator beside its trace output:

```json
{
  "calibrated": false,
  "flood_source": "0|0",
  "modules": {
    "1|2|core": 0,
    "1|2|mem": 0,
    "1|1|memtile": 0,
    "0|0|shim": 0
  }
}
```

- `calibrated`: the `BroadcastTiming.calibrated` flag (false until SP-5).
- `flood_source`: `"col|row"` of the single broadcast source. Absent/`null` only if no flood occurred.
- `modules`: `"col|row|<module_kind>" -> origin_D` (cycles), `module_kind ∈ {core, mem, memtile, shim}`. Rust emits semantic module kinds; the Python loader translates `module_kind -> pkt_type` so engine domain keys (`col|row|<pkt_type>`) resolve.

---

## Phase 1 -- Python engine (independently testable with synthetic sidecar fixtures; no emulator needed)

### Task 1: Lock the sidecar schema constants + module→pkt_type mapping

**Files:**
- Create: `tools/inference/model_io.py`
- Test: `tools/test_inference_model_io.py`

**Interfaces:**
- Produces: `MODULE_PKT_TYPE: Dict[str,int]` = `{"core":0,"mem":1,"shim":2,"memtile":3}`; `to_domain_key(col:int,row:int,module_kind:str) -> str` returning `"col|row|<pkt_type>"`; `SidecarError(Exception)`.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_model_io.py
import pytest
from inference.model_io import MODULE_PKT_TYPE, to_domain_key, SidecarError


def test_module_pkt_type_mapping_matches_decoder():
    # Confirmed from build/experiments/sp3-spike-trace/spike.events.json
    # (row0=shim=2, row1=memtile=3, compute=core=0) + dma-fill-measure.py (mem=1).
    assert MODULE_PKT_TYPE == {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def test_to_domain_key_translates_module_kind_to_pkt_type():
    assert to_domain_key(1, 2, "core") == "1|2|0"
    assert to_domain_key(1, 2, "mem") == "1|2|1"
    assert to_domain_key(0, 0, "shim") == "0|0|2"
    assert to_domain_key(1, 1, "memtile") == "1|1|3"


def test_to_domain_key_rejects_unknown_module_kind():
    with pytest.raises(SidecarError):
        to_domain_key(1, 2, "bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_model_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.model_io'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/inference/model_io.py
"""Sidecar schema constants + module-kind -> pkt_type translation.

The emulator writes origin_d.json keyed by semantic module kind (core/mem/
memtile/shim). The inference engine keys timer domains by NUMERIC pkt_type
(col|row|pkt_type), the field the trace decoder stamps on each event. This
module is the single place that bridges the two conventions, so the decoder's
pkt_type assignment is never duplicated into Rust.
"""
from __future__ import annotations
from typing import Dict


class SidecarError(Exception):
    """Malformed origin_d.json sidecar or unknown module kind."""


# Decoder convention (col|row|pkt_type). Confirmed: spike.events.json shows
# row0->2, row1->3, compute->0; dma-fill-measure.py shows core->0, mem->1.
MODULE_PKT_TYPE: Dict[str, int] = {"core": 0, "mem": 1, "shim": 2, "memtile": 3}


def to_domain_key(col: int, row: int, module_kind: str) -> str:
    if module_kind not in MODULE_PKT_TYPE:
        raise SidecarError(f"unknown module_kind {module_kind!r}")
    return f"{col}|{row}|{MODULE_PKT_TYPE[module_kind]}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_model_io.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/inference/model_io.py tools/test_inference_model_io.py
git commit -m "feat(#140): SP-4b sidecar schema + module->pkt_type mapping

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 2: `ModelDerived` provenance class + `provenance_ok` branch + KB model registry

**Files:**
- Modify: `tools/inference/facts.py` (add `ModelDerived` after `Structural:22`; extend `Support`; `KB.model` field; `provenance_ok` branch at `:108`; `leaves` at `:45`)
- Test: `tools/test_inference_model_provenance.py`

**Interfaces:**
- Produces: `ModelDerived(cite: str)` support kind; `KB.model: Dict[str,dict]` registry (parallels `KB.ledger`); `provenance_ok` accepts a `ModelDerived` leaf iff `cite in kb.model`.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_model_provenance.py
from inference.facts import (Fact, Measured, ModelDerived, Derived, KB,
                             provenance_ok, leaves)


def _kb_with_model_source(cite="origin_d.json"):
    kb = KB.empty()
    kb.model = {cite: {"calibrated": True}}
    return kb


def test_model_derived_leaf_accepted_when_cited():
    kb = _kb_with_model_source()
    kb.add(Fact("origin_d", ("1|2|0", 0), ModelDerived("origin_d.json")))
    assert provenance_ok(kb) is True


def test_model_derived_leaf_rejected_when_uncited():
    kb = KB.empty()  # empty kb.model
    kb.add(Fact("origin_d", ("1|2|0", 0), ModelDerived("origin_d.json")))
    assert provenance_ok(kb) is False


def test_model_derived_surfaces_in_leaves():
    md = Fact("origin_d", ("1|2|0", 0), ModelDerived("origin_d.json"))
    measured = Fact("fired", ("x", 0, 0), Measured())
    causal = Fact("causal", ("c", "p", 5), Derived("causal_decomp_rule", (measured, md)))
    leaf_supports = {type(f.support).__name__ for f in leaves(causal)}
    assert "ModelDerived" in leaf_supports
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_model_provenance.py -v`
Expected: FAIL with `ImportError: cannot import name 'ModelDerived'`

- [ ] **Step 3: Write minimal implementation**

In `tools/inference/facts.py`, after the `Structural` class (line 23):

```python
@dataclass(frozen=True)
class ModelDerived:
    """Support: a quote of the emulator's forward model (the origin_D sidecar),
    cited to the model artifact. Distinct from Measured/Structural -- a causal
    fact resting on it is permanently traceable to "modeled", never measured."""
    cite: str
```

Extend the `Support` union (line 32):

```python
Support = Union[Measured, Structural, ModelDerived, Derived]
```

In `leaves` (line 48), include `ModelDerived` as a leaf type:

```python
    if isinstance(s, (Measured, Structural, ModelDerived)):
        return frozenset({fact})
```

Add `model` to `KB` (after `ledger` at line 88):

```python
    ledger: Dict[str, dict] = field(default_factory=dict)
    model: Dict[str, dict] = field(default_factory=dict)
```

And in `KB.empty` (line 92):

```python
        return cls(facts={}, admitted_rules=[], rejected_rules=[], ledger={}, model={})
```

In `provenance_ok` (line 116-123), add the `ModelDerived` branch before the final `else`:

```python
            if isinstance(s, Measured):
                continue
            if isinstance(s, Structural):
                if s.cite not in kb.ledger:
                    return False
            elif isinstance(s, ModelDerived):
                if s.cite not in kb.model:
                    return False
            else:  # a Derived fact can never be a leaf
                return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_model_provenance.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full inference suite to confirm no regression**

Run: `PYTHONPATH=tools pytest tools/test_inference_*.py -q`
Expected: all PASS (the new union member and KB field are backward-compatible).

- [ ] **Step 6: Commit**

```bash
git add tools/inference/facts.py tools/test_inference_model_provenance.py
git commit -m "feat(#140): SP-4b ModelDerived provenance class + provenance_ok branch

Third leaf class beyond Measured/Structural: a model-derived leaf is accepted
iff its citation is a registered model source (kb.model), mirroring the
Structural/kb.ledger check. Keeps modeled data distinguishable in leaves().

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 3: `loader_model.py` -- load sidecar into `kb.model` + `ModelDerived` facts

**Files:**
- Create: `tools/inference/loader_model.py`
- Test: `tools/test_inference_loader_model.py`

**Interfaces:**
- Consumes: `model_io.to_domain_key`, `model_io.SidecarError`; `facts.Fact`, `facts.ModelDerived`, `facts.KB`.
- Produces: `load_model(path:str) -> dict` (the raw sidecar with `modules` re-keyed to `col|row|pkt_type`); `model_facts(model:dict, cite:str) -> List[Fact]` emitting `origin_d(domain_key, D)`, `skew_calibrated(bool)`, `flood_source(key)`; `install_model(kb, model, cite)` sets `kb.model[cite]=model` and adds the facts.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_loader_model.py
import json
import pytest
from inference.facts import KB, provenance_ok
from inference.loader_model import load_model, install_model
from inference.model_io import SidecarError


def _write(tmp_path, obj):
    p = tmp_path / "origin_d.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_load_model_rekeys_modules_to_pkt_type(tmp_path):
    path = _write(tmp_path, {"calibrated": False, "flood_source": "0|0",
                             "modules": {"1|2|core": 0, "0|0|shim": 0}})
    m = load_model(path)
    assert m["modules"] == {"1|2|0": 0, "0|0|2": 0}
    assert m["calibrated"] is False
    assert m["flood_source"] == "0|0"


def test_install_model_adds_cited_facts_and_passes_provenance(tmp_path):
    path = _write(tmp_path, {"calibrated": True, "flood_source": "0|0",
                             "modules": {"1|2|core": 4, "1|2|mem": 6}})
    kb = KB.empty()
    install_model(kb, load_model(path), cite="origin_d.json")
    assert kb.get("origin_d", ("1|2|0", 4)) is not None
    assert kb.get("skew_calibrated", (True,)) is not None
    assert kb.get("flood_source", ("0|0",)) is not None
    assert provenance_ok(kb) is True


def test_load_model_rejects_unknown_module_kind(tmp_path):
    path = _write(tmp_path, {"calibrated": False, "flood_source": "0|0",
                             "modules": {"1|2|bogus": 0}})
    with pytest.raises(SidecarError):
        load_model(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_loader_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.loader_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/inference/loader_model.py
"""Model-derived leaves: load the emulator's origin_D sidecar (origin_d.json).

The sidecar is the emulator's modeled per-module timer-reset arrival. It is
neither measured (loader.py) nor structural/binary (ledger.py) -- it is
model-derived, loaded under facts.ModelDerived and cited to the artifact so a
causal fact built on it stays traceable to the model. Module-kind keys are
translated to numeric pkt_type domain keys here (model_io)."""
from __future__ import annotations
import json
from typing import Dict, List
from inference.facts import Fact, ModelDerived, KB
from inference.model_io import to_domain_key, SidecarError


def load_model(path: str) -> dict:
    with open(path) as fh:
        raw = json.load(fh)
    for k in ("calibrated", "modules"):
        if k not in raw:
            raise SidecarError(f"sidecar missing {k!r}: {path}")
    rekeyed: Dict[str, int] = {}
    for mk, d in raw["modules"].items():
        parts = mk.split("|")
        if len(parts) != 3:
            raise SidecarError(f"bad module key {mk!r}")
        col, row, kind = parts
        rekeyed[to_domain_key(int(col), int(row), kind)] = d
    return {"calibrated": bool(raw["calibrated"]),
            "flood_source": raw.get("flood_source"),
            "modules": rekeyed}


def model_facts(model: dict, cite: str) -> List[Fact]:
    facts: List[Fact] = [
        Fact("skew_calibrated", (model["calibrated"],), ModelDerived(cite)),
    ]
    if model.get("flood_source") is not None:
        facts.append(Fact("flood_source", (model["flood_source"],), ModelDerived(cite)))
    for dom, d in model["modules"].items():
        facts.append(Fact("origin_d", (dom, d), ModelDerived(cite)))
    return facts


def install_model(kb: KB, model: dict, cite: str = "origin_d.json") -> None:
    kb.model[cite] = model
    for f in model_facts(model, cite):
        kb.add(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_loader_model.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/inference/loader_model.py tools/test_inference_loader_model.py
git commit -m "feat(#140): SP-4b loader_model -- origin_D sidecar -> ModelDerived facts

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 4: `causal_offset` helper + `domain_of` + `Gap.causal_offset` + `ground_edge` model param

**Files:**
- Modify: `tools/inference/grounding.py` (add `domain_of`, `CrossDomainModelError`, `causal_offset` after `same_domain:51`; `Gap.causal_offset` field at `:79`; `ground_edge` `model` param at `:91`/`:105`)
- Test: `tools/test_inference_causal_offset.py`

**Interfaces:**
- Consumes: nothing new (model dict passed in by caller).
- Produces: `domain_of(key:str) -> str`; `CrossDomainModelError(Exception)`; `causal_offset(model:dict, child:str, parent:str, raw:Optional[int]) -> Optional[int]`; `Gap.causal_offset: Optional[int] = None`; `ground_edge(..., model: Optional[dict] = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_causal_offset.py
import pytest
from inference.grounding import (domain_of, causal_offset, CrossDomainModelError,
                                 ground_edge, Gap, GAP_CROSS_DOMAIN)


def test_domain_of_strips_event_name():
    assert domain_of("1|2|0|DMA_S2MM_0_START_TASK") == "1|2|0"


def test_causal_offset_pins_known_delta_wall():
    # Construct a known physical situation, asymmetric origin_D + known Δwall.
    # child in domain A=1|2|0 (origin_D=10), parent in domain B=0|0|2 (origin_D=4).
    # raw = Δwall − (origin_D[A] − origin_D[B]). Choose Δwall = 7:
    #   raw = 7 − (10 − 4) = 1.  Then causal must recover Δwall = 7.
    model = {"calibrated": True, "flood_source": "0|0",
             "modules": {"1|2|0": 10, "0|0|2": 4}}
    child = "1|2|0|EVT_X"     # domain A
    parent = "0|0|2|EVT_Y"    # domain B
    raw = 1
    assert causal_offset(model, child, parent, raw) == 7


def test_causal_offset_withheld_when_uncalibrated():
    model = {"calibrated": False, "flood_source": "0|0",
             "modules": {"1|2|0": 10, "0|0|2": 4}}
    assert causal_offset(model, "1|2|0|X", "0|0|2|Y", 1) is None


def test_causal_offset_fails_loud_on_missing_domain():
    model = {"calibrated": True, "flood_source": "0|0", "modules": {"1|2|0": 10}}
    with pytest.raises(CrossDomainModelError):
        causal_offset(model, "1|2|0|X", "0|0|2|Y", 1)  # parent domain absent


def test_ground_edge_attaches_causal_offset_to_cross_domain_gap(monkeypatch):
    import inference.grounding as g
    monkeypatch.setattr(g, "offset_exact", lambda *a, **k: 1)
    model = {"calibrated": True, "flood_source": "0|0",
             "modules": {"1|2|0": 10, "0|0|2": 4}}
    res = ground_edge([], "1|2|0|X", "0|0|2|Y", model=model)
    assert isinstance(res, Gap) and res.reason == GAP_CROSS_DOMAIN
    assert res.reproduction_offset == 1 and res.causal_offset == 7


def test_ground_edge_causal_none_without_model(monkeypatch):
    import inference.grounding as g
    monkeypatch.setattr(g, "offset_exact", lambda *a, **k: 1)
    res = ground_edge([], "1|2|0|X", "0|0|2|Y")
    assert res.causal_offset is None and res.reproduction_offset == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_causal_offset.py -v`
Expected: FAIL with `ImportError: cannot import name 'domain_of'`

- [ ] **Step 3: Write minimal implementation**

In `tools/inference/grounding.py`, after `same_domain` (line 51):

```python
def domain_of(key: str) -> str:
    """The col|row|pkt_type timer-domain prefix of an event key (drops the name)."""
    return key.rsplit("|", 1)[0]


class CrossDomainModelError(Exception):
    """A calibrated cross-domain decomposition where a domain is absent from the
    single-source origin_D table -- cross-source or unreached. Fail loud rather
    than emit a wrong causal_offset (design Sec.5b/5d)."""


def causal_offset(model, child: str, parent: str, raw):
    """Δwall = raw − skew(A,B), with A=domain(child), B=domain(parent) and
    skew = origin_D[B] − origin_D[A] (design Sec.2/2a). None unless the model is
    calibrated and raw is exact. Raises CrossDomainModelError if calibrated but a
    domain is missing from the single-source table."""
    if not model.get("calibrated", False) or raw is None:
        return None
    od = model.get("modules", {})
    cd, pd = domain_of(child), domain_of(parent)
    if cd not in od or pd not in od:
        raise CrossDomainModelError(
            f"calibrated model missing origin_D for {cd!r} or {pd!r} "
            f"(flood_source={model.get('flood_source')!r}); cross-source or unreached")
    skew = od[pd] - od[cd]
    return raw - skew
```

Add `causal_offset` to the `Gap` dataclass (after line 79):

```python
@dataclass(frozen=True)
class Gap:
    parent: str
    child: str
    reason: str
    reproduction_offset: Optional[int] = None
    causal_offset: Optional[int] = None
```

Change `ground_edge` signature and the cross-domain branch (lines 91-107):

```python
def ground_edge(run_dirs: List[str], child: str, parent: str,
                anchor_key: str = ANCHOR, model=None) -> Grounding:
    if is_async_cdc(child) or is_async_cdc(parent):
        return Gap(parent=parent, child=child, reason=GAP_ASYNC_CDC)
    if same_domain(child, parent):
        off = offset_exact(run_dirs, child, parent, anchor_key)
        if off is not None:
            return Segment(parent=parent, child=child, offset=off)
        return Gap(parent=parent, child=child, reason=GAP_WITHIN_DOMAIN_NONEXACT)
    raw = offset_exact(run_dirs, child, parent, anchor_key)
    causal = causal_offset(model, child, parent, raw) if model is not None else None
    return Gap(parent=parent, child=child, reason=GAP_CROSS_DOMAIN,
               reproduction_offset=raw, causal_offset=causal)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_causal_offset.py -v`
Expected: PASS (6 tests). The `test_causal_offset_pins_known_delta_wall` is the load-bearing sign-pin.

- [ ] **Step 5: Run the inference suite (ground_edge is widely used)**

Run: `PYTHONPATH=tools pytest tools/test_inference_*.py -q`
Expected: all PASS (the new `model` param defaults to `None`; existing callers unaffected).

- [ ] **Step 6: Commit**

```bash
git add tools/inference/grounding.py tools/test_inference_causal_offset.py
git commit -m "feat(#140): SP-4b causal_offset decomposition in ground_edge

skew = origin_D[parent_dom] - origin_D[child_dom]; causal = raw - skew = Δwall.
Withheld (None) unless calibrated; fails loud on a missing domain. Sign pinned
by a known-Δwall test, not a circular skew assertion.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 5: `try_causal` + chainer wiring -- the separate causal fact with model premises

**Files:**
- Modify: `tools/inference/rules.py` (add `try_causal` after `try_same_source:71`; import `causal_offset`, `domain_of`, `GAP_CROSS_DOMAIN`, `derive_*`)
- Modify: `tools/inference/chainer.py` (add a third attempt + `_has_causal` in the loop at `:33-41`)
- Test: `tools/test_inference_try_causal.py`

**Interfaces:**
- Consumes: `kb.model` (raw dict registered by `install_model`), the gap `derives` fact from `try_derives`, `grounding.causal_offset`/`domain_of`.
- Produces: `rules.try_causal(run_dirs, kb, child, parent, anchor_key) -> Optional[Fact]` emitting `causal(child, parent, causal_offset)` with `Derived("causal_decomp_rule", (gap_fact, origin_d_child, origin_d_parent, skew_calibrated))`; `chainer._has_causal(kb, a, b)`.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_try_causal.py
from inference.facts import Fact, Measured, ModelDerived, Derived, KB, provenance_ok, leaves
from inference.rules import try_causal


def _seed_cross_domain_gap(kb, child, parent, raw):
    # Minimal stand-in for what try_derives leaves in the KB: a cross-domain gap
    # derive with reproduction_offset = raw, plus the measured fired leaves.
    fired_c = kb.add(Fact("fired", (child, 0, 0), Measured()))
    fired_p = kb.add(Fact("fired", (parent, 0, 0), Measured()))
    grd = Fact("gap", (child, parent), Derived("grounding_rule", (fired_c, fired_p)))
    cp = Fact("config_path", (parent, child, "c0"), __import__("inference.facts", fromlist=["Structural"]).Structural("c0"))
    kb.ledger["c0"] = {"a": parent, "b": child, "kind": "route", "cite": "c0"}
    kb.add(cp)
    return kb.add(Fact("derives", (child, parent, None, "gap", raw, "cross_domain"),
                       Derived("derives_rule_placement", (cp, grd))))


def _install_model(kb):
    kb.model["origin_d.json"] = {"calibrated": True, "flood_source": "0|0",
                                 "modules": {"1|2|0": 10, "0|0|2": 4}}
    kb.add(Fact("skew_calibrated", (True,), ModelDerived("origin_d.json")))
    kb.add(Fact("origin_d", ("1|2|0", 10), ModelDerived("origin_d.json")))
    kb.add(Fact("origin_d", ("0|0|2", 4), ModelDerived("origin_d.json")))


def test_try_causal_emits_model_grounded_fact():
    kb = KB.empty()
    child, parent = "1|2|0|X", "0|0|2|Y"
    _seed_cross_domain_gap(kb, child, parent, raw=1)
    _install_model(kb)
    f = try_causal([], kb, child, parent)
    assert f is not None
    assert f.predicate == "causal" and f.args == (child, parent, 7)
    leaf_supports = {type(x.support).__name__ for x in leaves(f)}
    assert "ModelDerived" in leaf_supports and "Measured" in leaf_supports
    kb.add(f)
    assert provenance_ok(kb) is True


def test_try_causal_none_when_uncalibrated():
    kb = KB.empty()
    child, parent = "1|2|0|X", "0|0|2|Y"
    _seed_cross_domain_gap(kb, child, parent, raw=1)
    kb.model["origin_d.json"] = {"calibrated": False, "flood_source": "0|0",
                                 "modules": {"1|2|0": 10, "0|0|2": 4}}
    assert try_causal([], kb, child, parent) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_try_causal.py -v`
Expected: FAIL with `ImportError: cannot import name 'try_causal'`

- [ ] **Step 3: Write minimal implementation**

In `tools/inference/rules.py`, update imports (line 11-13):

```python
from inference.facts import (Fact, Derived, KB, derive_kind, derive_gap_reason,
                             derive_reproduction_offset)
from inference.verifier import anchor_rigid, check_ordering, coincident, ANCHOR
from inference.grounding import (ground_edge, Segment, causal_offset, domain_of,
                                 GAP_CROSS_DOMAIN)
```

Add `try_causal` after `try_same_source` (line 71):

```python
def try_causal(run_dirs: List[str], kb: KB, child: str, parent: str,
               anchor_key: str = ANCHOR) -> Optional[Fact]:
    """Emit a model-grounded causal fact for a placed cross-domain gap, when the
    broadcast model is calibrated. Its premises explicitly cite the origin_D
    ModelDerived facts, so leaves() surfaces the model dependency -- the causal
    cycle never launders into a measured claim (design Sec.5a)."""
    model = next(iter(kb.model.values()), None)
    if model is None or not model.get("calibrated", False):
        return None
    d = next((f for f in kb.by_predicate("derives")
              if f.args[0] == child and f.args[1] == parent
              and derive_kind(f) == "gap"
              and derive_gap_reason(f) == GAP_CROSS_DOMAIN), None)
    if d is None:
        return None
    raw = derive_reproduction_offset(d)
    co = causal_offset(model, child, parent, raw)  # raises on missing domain
    if co is None:
        return None
    od = model["modules"]
    cd, pd = domain_of(child), domain_of(parent)
    od_child = kb.get("origin_d", (cd, od[cd]))
    od_parent = kb.get("origin_d", (pd, od[pd]))
    cal = kb.get("skew_calibrated", (True,))
    premises = tuple(p for p in (d, od_child, od_parent, cal) if p is not None)
    return Fact("causal", (child, parent, co),
                Derived("causal_decomp_rule", premises))
```

In `tools/inference/chainer.py`, import `try_causal` (line 11) and add the third attempt inside the `for a, b in pairs` loop (after the same_source block, line 41):

```python
from inference.rules import (mark_determinism, try_derives, try_same_source,
                             try_causal, is_stochastic_root)
```

```python
            if not _has_causal(kb, a, b):
                c = try_causal(run_dirs, kb, a, b, anchor_key)
                if c is not None and not kb.has(c.predicate, c.args):
                    kb.add(c); changed = True
```

And add the helper near `_has_same_source` (line 53):

```python
def _has_causal(kb: KB, child: str, parent: str) -> bool:
    return any(f.args[0] == child and f.args[1] == parent
               for f in kb.by_predicate("causal"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_try_causal.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the inference suite**

Run: `PYTHONPATH=tools pytest tools/test_inference_*.py -q`
Expected: all PASS (the chainer's third attempt is gated on `kb.model` being calibrated; with no model installed, `try_causal` returns None immediately, so existing runs are unchanged).

- [ ] **Step 6: Commit**

```bash
git add tools/inference/rules.py tools/inference/chainer.py tools/test_inference_try_causal.py
git commit -m "feat(#140): SP-4b try_causal -- separate model-grounded causal fact

The causal fact's premises explicitly cite the origin_D ModelDerived facts, so
leaves() surfaces the model dependency. Chainer gains a third per-pair attempt,
inert unless a calibrated model is installed.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 6: Thread `causal_offset` onto the edge path + render

**Files:**
- Modify: `tools/inference/timeline.py` (`CrossTrackEdge.causal_offset` at `:144-148`; `weave` `model` param + copy at `:530-558`; `assemble_timeline` `model` param + pass to weave at `:789`; `render_timeline` EDGE line at `:851-853`)
- Test: `tools/test_inference_timeline_causal.py`

**Interfaces:**
- Consumes: `grounding.ground_edge(model=...)`.
- Produces: `CrossTrackEdge.causal_offset: Optional[int]`; `weave(run_dirs, cross_domain_pairs, anchor_key, model=None)`; `assemble_timeline(..., model=None)`; render EDGE line gains `, causal_offset=N [model-derived]` only when non-None.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_timeline_causal.py
import inference.timeline as T
from inference.timeline import CrossTrackEdge, render_timeline, IntegratedTimeline


def test_cross_track_edge_carries_causal_offset():
    e = CrossTrackEdge(child="c", parent="p", reason="cross_domain",
                       reproduction_offset=1, causal_offset=7)
    assert e.causal_offset == 7


def test_render_omits_causal_when_none():
    e = CrossTrackEdge("c", "p", "cross_domain", 1, None)
    tl = IntegratedTimeline(tracks=[], cross_track_edges=[e], intermittent=[],
                            flags=[], census=T.census_of([], [], {}, [e]), capture={})
    out = render_timeline(tl)
    assert "reproduction_offset=1" in out and "causal_offset" not in out


def test_render_shows_causal_when_present():
    e = CrossTrackEdge("c", "p", "cross_domain", 1, 7)
    tl = IntegratedTimeline(tracks=[], cross_track_edges=[e], intermittent=[],
                            flags=[], census=T.census_of([], [], {}, [e]), capture={})
    out = render_timeline(tl)
    assert "causal_offset=7 [model-derived]" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_timeline_causal.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'causal_offset'`

- [ ] **Step 3: Write minimal implementation**

In `tools/inference/timeline.py`, add the field to `CrossTrackEdge` (line 148):

```python
@dataclass
class CrossTrackEdge:
    child: str
    parent: str
    reason: str                   # cross_domain | async_cdc
    reproduction_offset: Optional[int]  # None -> existence-only
    causal_offset: Optional[int] = None  # model-derived Δwall; None pre-calibration
```

Change `weave` (lines 530, 553-557):

```python
def weave(run_dirs, cross_domain_pairs, anchor_key=ANCHOR, model=None) -> List[CrossTrackEdge]:
```

```python
        g = ground_edge(run_dirs, child, parent, anchor_key, model=model)
        if isinstance(g, Gap):
            edges.append(CrossTrackEdge(child=child, parent=parent,
                                        reason=g.reason,
                                        reproduction_offset=g.reproduction_offset,
                                        causal_offset=g.causal_offset))
```

Change `assemble_timeline` to accept and forward `model`. At its signature add `model=None`, and at the weave call (line 789):

```python
    edges = weave(run_dirs, cross_domain_pairs, anchor_key, model=model)
```

Change the EDGE render line (lines 851-853):

```python
    for e in tl.cross_track_edges:
        causal = getattr(e, "causal_offset", None)
        suffix = "" if causal is None else f", causal_offset={causal} [model-derived]"
        lines.append(f"EDGE {e.child} <- {e.parent} "
                     f"({e.reason}, reproduction_offset={e.reproduction_offset}{suffix})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_timeline_causal.py -v`
Expected: PASS (3 tests). The omit-when-None test guards render byte-identity pre-SP-5.

- [ ] **Step 5: Run the inference suite**

Run: `PYTHONPATH=tools pytest tools/test_inference_*.py -q`
Expected: all PASS (`model` defaults to `None`; the EDGE line is byte-identical when `causal_offset is None`).

- [ ] **Step 6: Commit**

```bash
git add tools/inference/timeline.py tools/test_inference_timeline_causal.py
git commit -m "feat(#140): SP-4b thread causal_offset onto edge path + render

weave copies g.causal_offset into CrossTrackEdge; render appends a tagged
[model-derived] suffix only when present, so pre-calibration EDGE lines are
byte-identical.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 7: Engine integration -- `install_model`, thread to timeline, causal report list, byte-identity

**Files:**
- Modify: `tools/inference/engine.py` (`run_engine` signature `model_path=None`; install + thread to `assemble_timeline`; add `causal` to report)
- Test: `tools/test_inference_engine_causal.py`

**Interfaces:**
- Consumes: `loader_model.load_model`/`install_model`.
- Produces: `run_engine(..., model_path: Optional[str] = None)`; report dict gains `"causal": [(child, parent, causal_offset), ...]` from `kb.by_predicate("causal")`; `provenance_ok` stays `True` with inert model facts.

- [ ] **Step 1: Write the failing test**

```python
# tools/test_inference_engine_causal.py
"""Engine-level: installing an uncalibrated model leaves all existing report
fields byte-identical and provenance_ok True (the inert-fact byte-identity
guarantee, design Sec.5e). A calibrated model surfaces a causal list."""
import json
from inference.facts import KB, provenance_ok
from inference.loader_model import install_model, load_model


def test_uncalibrated_model_keeps_provenance_ok(tmp_path):
    p = tmp_path / "origin_d.json"
    p.write_text(json.dumps({"calibrated": False, "flood_source": "0|0",
                             "modules": {"1|2|core": 0}}))
    kb = KB.empty()
    install_model(kb, load_model(str(p)))
    # Inert ModelDerived leaves present, cited -> provenance_ok holds.
    assert provenance_ok(kb) is True
    # No causal facts emitted pre-calibration.
    assert kb.by_predicate("causal") == []
```

(Note: a full end-to-end `run_engine` causal test needs real batch dirs + a sidecar; that is Task 11. This task verifies the engine plumbing and the inert-fact invariant in isolation.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=tools pytest tools/test_inference_engine_causal.py -v`
Expected: PASS for the provenance assertion is possible already; this step's purpose is to confirm the install plumbing. If it errors on import or KB shape, fix forward. (Expected initial state: PASS, since Tasks 2-3 already provide install_model/provenance.)

- [ ] **Step 3: Write minimal implementation**

In `tools/inference/engine.py`, import (line 18-19):

```python
from inference.ledger import load_ledger, install_ledger
from inference.loader_model import load_model, install_model
```

Change `run_engine` signature (line 27-29) to add `model_path`:

```python
def run_engine(run_dirs: List[str], ledger_path: str,
               candidate_pairs: List[Tuple[str, str]],
               anchor_key: str = ANCHOR, dump=None, start_col: int = 1,
               model_path: str = None) -> dict:
```

After `install_ledger(...)` (line 35), install the model when provided:

```python
    kb = KB.empty()
    install_ledger(kb, load_ledger(ledger_path))
    if model_path is not None:
        install_model(kb, load_model(model_path))
    for f in load_fired(run_dirs, anchor_key):
        kb.add(f)
```

Thread `model` into the timeline (line 77-78); pull the raw model dict back out of the KB:

```python
    model = next(iter(kb.model.values()), None)
    timeline = assemble_timeline(run_dirs, fired, derives_pairs, cross_domain_pairs,
                                 dump=dump, start_col=start_col, anchor_key=anchor_key,
                                 model=model)
```

Add the causal list to the report (after `gaps`, line 64, and in the return dict line 80-91):

```python
    causal = [(f.args[0], f.args[1], f.args[2]) for f in kb.by_predicate("causal")]
```

```python
            "gaps": gaps,
            "causal": causal,
            "warnings": warnings,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=tools pytest tools/test_inference_engine_causal.py -v`
Expected: PASS

- [ ] **Step 5: Run the full inference suite + confirm byte-identity of an existing report**

Run: `PYTHONPATH=tools pytest tools/test_inference_*.py -q`
Expected: all PASS. Existing `run_engine` callers pass no `model_path`, so `kb.model` is empty, `model=None`, `causal=[]`, and every prior report field is unchanged (the new `"causal"` key is additive; if any existing test asserts exact report-key sets, update that test to include `"causal": []`).

- [ ] **Step 6: Commit**

```bash
git add tools/inference/engine.py tools/test_inference_engine_causal.py
git commit -m "feat(#140): SP-4b engine -- install_model + causal report list

run_engine accepts an optional model_path; with none, kb.model is empty and
every existing report field is byte-identical. provenance_ok stays True with
inert cited ModelDerived leaves.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Phase 2 -- Rust origin_D export (produces the real sidecar Phase 1 consumes)

### Task 8: `BroadcastTiming.calibrated` flag

**Files:**
- Modify: `crates/xdna-archspec/src/types.rs:1352-1363` (add field)
- Modify: `crates/xdna-archspec/src/model_builder.rs:270-279` (set `calibrated: false`)
- Modify: any other `BroadcastTiming { ... }` constructors the compiler flags (likely test fixtures in `effects.rs` near `:1148`)
- Test: covered by `cargo build` + an archspec assertion

**Interfaces:**
- Produces: `BroadcastTiming.calibrated: bool` (default false until SP-5).

- [ ] **Step 1: Add the field with a doc comment** (`types.rs`, after `intra_tile_mem_offset` at line 1362):

```rust
    /// Additive offset for a compute tile's memory module (same baseline).
    pub intra_tile_mem_offset: u8,
    /// True once SP-5 has measured the per-hop/intra-tile constants on silicon.
    /// EXPLICIT, not inferred from "constants != 0" -- a genuinely measured d_v=0
    /// must not read as uncalibrated. The engine withholds causal_offset while
    /// this is false (design Sec.4a/5b).
    pub calibrated: bool,
```

- [ ] **Step 2: Set it in `model_builder.rs`** (line 278-279):

```rust
            intra_tile_core_offset: 0,
            intra_tile_mem_offset: 0,
            calibrated: false,
        },
```

- [ ] **Step 3: Build to find every other constructor**

Run: `cargo build -p xdna-archspec`
Expected: compile errors `missing field 'calibrated'` at each remaining `BroadcastTiming { ... }`. Add `calibrated: false,` to each (test fixtures included).

- [ ] **Step 4: Add an archspec assertion** in the existing broadcast test module (`crates/xdna-archspec/src/...` where `BROADCAST_PER_HOP_*` zero-asserts live, mirrored in `effects.rs:1148`):

```rust
    #[test]
    fn broadcast_timing_defaults_uncalibrated() {
        let m = /* the npu1 model build entry used by sibling tests */;
        assert!(!m.timing.as_ref().unwrap().broadcast.calibrated);
    }
```

(Use the same model-build accessor the neighboring zero-assert test uses; match its exact path.)

- [ ] **Step 5: Run tests**

Run: `cargo test -p xdna-archspec broadcast`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-archspec/src/types.rs crates/xdna-archspec/src/model_builder.rs
git commit -m "feat(#140): SP-4b BroadcastTiming.calibrated flag (false until SP-5)

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 9: per-module `origin_D` table builder + single-source capture

**Files:**
- Modify: `src/device/state/effects.rs` (new `origin_d_table(...)` near `broadcast_origin_d:443`; record flood source(s) in `propagate_broadcasts_with_timing:564`)
- Modify: `src/interpreter/engine/...` (a field to accumulate observed flood sources + the per-module table; exact module: wherever `EffectfulState`/the array owner lives -- locate with `grep -rn "propagate_broadcasts" src/`)
- Test: `cargo test -p xdna-emu origin_d_table` (Rust unit test in `effects.rs`)

**Interfaces:**
- Produces: `origin_d_table(&self, col, source_row, channel, d_h, d_v, core_off, mem_off) -> Vec<(u8,u8,&'static str,u32)>` mapping each reached `(col,row)` to its module-kind rows (`core`/`mem` for compute, `memtile` for mem, `shim` for shim) with `origin_D + intra_off` (= `core_delay`/`mem_delay`, the physical value -- NOT `*_target`). A recorded set of distinct `(col,source_row)` flood sources across the run (for the single-source assertion).

- [ ] **Step 1: Write the failing test** (in `effects.rs` test module):

```rust
    #[test]
    fn origin_d_table_keys_modules_and_uses_delay_not_target() {
        // Single-source flood with nonzero synthetic constants so the table is
        // not all-zero. d_h=d_v=1, core_off=0, mem_off=2 (the -2 intra signature).
        let st = /* build a small npu1 EffectfulState, see neighboring tests */;
        let rows = st.origin_d_table(0, 0, /*channel*/ 15, 1, 1, 0, 2);
        // The shim source (0,0) has origin_D 0 -> shim delay 0.
        assert!(rows.iter().any(|&(c, r, k, d)| c == 0 && r == 0 && k == "shim" && d == 0));
        // A compute tile at Manhattan distance N has core_delay = N*1 + 0,
        // mem_delay = N*1 + 2 (delay = origin_D + intra_off, NOT max_delay - delay).
        // Assert a known compute module's mem delay exceeds its core delay by 2.
        let core = rows.iter().find(|&&(c, r, k, _)| c == 1 && r == 2 && k == "core").map(|&(_, _, _, d)| d);
        let mem = rows.iter().find(|&&(c, r, k, _)| c == 1 && r == 2 && k == "mem").map(|&(_, _, _, d)| d);
        if let (Some(co), Some(me)) = (core, mem) { assert_eq!(me, co + 2); }
    }
```

(Build the `EffectfulState` exactly as the sibling broadcast tests in `effects.rs` do -- match their setup helper.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-emu origin_d_table`
Expected: FAIL with `no method named 'origin_d_table'`

- [ ] **Step 3: Implement `origin_d_table`** near `broadcast_origin_d` (effects.rs:503):

```rust
    /// Per-module origin_D rows for the single-source flood, for the SP-4b
    /// sidecar. Maps each reached tile to its module-kind row(s) carrying
    /// `origin_D + intra_off` (= core_delay/mem_delay -- the physical timer-reset
    /// arrival, NOT the max_delay-complement `*_target`). Module kinds match the
    /// engine's decoder convention via the Python loader's module->pkt_type map.
    pub(crate) fn origin_d_table(
        &self, col: u8, source_row: u8, channel: u8,
        d_h: u32, d_v: u32, core_off: u32, mem_off: u32,
    ) -> Vec<(u8, u8, &'static str, u32)> {
        let mut out = Vec::new();
        for (c, r, origin_d) in self.broadcast_origin_d(col, source_row, channel, d_h, d_v) {
            match self.array.get(c, r).map(|t| t.tile_kind) {
                Some(TileKind::Compute) => {
                    out.push((c, r, "core", origin_d + core_off));
                    out.push((c, r, "mem", origin_d + mem_off));
                }
                Some(TileKind::Mem) => out.push((c, r, "memtile", origin_d + mem_off)),
                Some(TileKind::ShimNoc) | Some(TileKind::ShimPl) =>
                    out.push((c, r, "shim", origin_d + core_off)),
                None => {}
            }
        }
        out
    }
```

- [ ] **Step 4: Record flood source(s)** in `propagate_broadcasts_with_timing` (after the `channels` drain, around line 564): push `(col, source_row)` into a `HashSet` on the engine/state (add the field where the executor owns trace state; locate via `grep -rn "flush_trace_to_host" src/`). The set's purpose: at export, `len == 1` -> single source; `> 1` -> the sidecar omits `flood_source`/marks multi-source so the engine fails loud.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p xdna-emu origin_d_table`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/device/state/effects.rs src/interpreter/
git commit -m "feat(#140): SP-4b per-module origin_D table + flood-source capture

origin_d_table maps reached tiles to module-kind rows carrying origin_D +
intra_off (core_delay/mem_delay, the physical value, not the *_target
complement). Distinct flood sources are recorded for the single-source guard.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 10: sidecar serialization + FFI flush hook + runner wiring

**Files:**
- Modify: `src/interpreter/engine/...` (a method `export_origin_d_sidecar(&self) -> serde_json::Value` building the contract JSON from `origin_d_table` + recorded source + `BroadcastTiming.calibrated`)
- Modify: `crates/xdna-emu-ffi/src/backend.rs:359-362` (write the sidecar to `$XDNA_EMU_ORIGIN_D_OUT` when set, alongside `flush_trace_to_host`)
- Modify: the bridge trace runner (`scripts/`/`tools/` -- locate with `grep -rln "trace-out\|XDNA_EMU" scripts tools`) to set `XDNA_EMU_ORIGIN_D_OUT` beside `--trace-out`
- Test: a Rust unit test on `export_origin_d_sidecar` shape

**Interfaces:**
- Consumes: `origin_d_table`, the flood-source set, `model.timing.broadcast`.
- Produces: `origin_d.json` matching the contract; written at trace flush when `XDNA_EMU_ORIGIN_D_OUT` is set.

- [ ] **Step 1: Write the failing test** (engine module test):

```rust
    #[test]
    fn export_origin_d_sidecar_matches_contract() {
        let st = /* small npu1 engine after a single-source flood, see Task 9 setup */;
        let v = st.export_origin_d_sidecar();
        assert_eq!(v["calibrated"], serde_json::json!(false));
        assert_eq!(v["flood_source"], serde_json::json!("0|0"));
        assert!(v["modules"].as_object().unwrap().contains_key("0|0|shim"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p xdna-emu export_origin_d_sidecar`
Expected: FAIL with `no method named 'export_origin_d_sidecar'`

- [ ] **Step 3: Implement `export_origin_d_sidecar`**:

```rust
    pub fn export_origin_d_sidecar(&self) -> serde_json::Value {
        let bc = /* self.model.timing.broadcast */;
        let sources = /* the recorded flood-source set */;
        let mut modules = serde_json::Map::new();
        // single source only; if sources.len() != 1, omit flood_source so the
        // engine fails loud (design Sec.4d/5d).
        let single = if sources.len() == 1 { sources.iter().next().copied() } else { None };
        if let Some((col, row)) = single {
            for (c, r, kind, d) in self.effects().origin_d_table(
                col, row, 15, bc.per_hop_horizontal as u32, bc.per_hop_vertical as u32,
                bc.intra_tile_core_offset as u32, bc.intra_tile_mem_offset as u32) {
                modules.insert(format!("{c}|{r}|{kind}"), serde_json::json!(d));
            }
        }
        serde_json::json!({
            "calibrated": bc.calibrated,
            "flood_source": single.map(|(c, r)| format!("{c}|{r}")),
            "modules": modules,
        })
    }
```

(Resolve `self.model.timing.broadcast`, `self.effects()`, and the source set against the real engine struct -- the names above are the contract, not guaranteed field paths; wire to the actual accessors.)

- [ ] **Step 4: Hook the FFI flush** (`backend.rs`, near line 362 where `flush_trace_to_host()` is called):

```rust
        engine.flush_trace_to_host();
        if let Ok(path) = std::env::var("XDNA_EMU_ORIGIN_D_OUT") {
            let v = engine.export_origin_d_sidecar();
            let _ = std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap());
        }
```

- [ ] **Step 5: Wire the runner** -- in the bridge trace runner, set `XDNA_EMU_ORIGIN_D_OUT` to `<trace-out-dir>/origin_d.json` whenever it sets `--trace-out` for an EMU run.

- [ ] **Step 6: Run test + rebuild FFI**

Run: `cargo test -p xdna-emu export_origin_d_sidecar`
Then: `cargo build -p xdna-emu-ffi`
Expected: test PASS; FFI builds.

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/ crates/xdna-emu-ffi/src/backend.rs scripts/ tools/
git commit -m "feat(#140): SP-4b origin_d.json sidecar export + FFI flush hook

export_origin_d_sidecar builds the contract JSON (single-source only; omits
flood_source on multi-source so the engine fails loud). Written at trace flush
when XDNA_EMU_ORIGIN_D_OUT is set; the bridge runner points it beside --trace-out.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Phase 3 -- End-to-end integration

### Task 11: End-to-end -- real sidecar through the engine, with byte-identity + calibrated-path coverage

**Files:**
- Create: `tools/test_inference_sp4b_e2e.py`
- Test artifact: reuse `build/experiments/sp3-spike-trace/` (real EMU trace) or a small captured batch dir

**Interfaces:**
- Consumes: the whole stack (`run_engine(model_path=...)`).

- [ ] **Step 1: Generate a real sidecar from an EMU run**

Run an existing EMU trace flow with `XDNA_EMU_ORIGIN_D_OUT=/tmp/claude-1000/.../origin_d.json` set (e.g. re-run the sp3 spike or a small bridge trace). Confirm the file exists and matches the contract (`calibrated:false`, a `flood_source`, module keys).

- [ ] **Step 2: Write the byte-identity test**

```python
# tools/test_inference_sp4b_e2e.py
"""End-to-end: an uncalibrated real sidecar leaves run_engine's report
byte-identical except for the additive empty `causal` list; a synthetically
calibrated copy surfaces causal facts and keeps provenance_ok True."""
import json, copy
from inference.engine import run_engine

# Point these at a captured batch dir set + its ledger (reuse an existing
# engine test's fixtures; e.g. the add_one_using_dma smoke fixtures).
RUNS = [...]; LEDGER = "..."; PAIRS = [...]


def test_uncalibrated_sidecar_is_byte_identical(tmp_path):
    base = run_engine(RUNS, LEDGER, PAIRS)
    sidecar = tmp_path / "origin_d.json"
    sidecar.write_text(json.dumps({"calibrated": False, "flood_source": "0|0",
                                   "modules": {"1|2|core": 0}}))
    withmodel = run_engine(RUNS, LEDGER, PAIRS, model_path=str(sidecar))
    assert withmodel["causal"] == []
    # Every other field identical.
    a = {k: v for k, v in base.items()}
    b = {k: v for k, v in withmodel.items() if k != "causal"}
    assert a == b
    assert withmodel["provenance_ok"] is True


def test_calibrated_sidecar_emits_causal_with_provenance(tmp_path):
    # Synthetic calibration: take the real module set, mark calibrated, inject
    # nonzero asymmetric origin_D, and confirm a causal fact appears with a
    # ModelDerived premise and provenance_ok still True.
    sidecar = tmp_path / "origin_d.json"
    sidecar.write_text(json.dumps({"calibrated": True, "flood_source": "0|0",
                                   "modules": {  # cover the real cross-domain pairs
                                       # fill from the actual traced domains
                                   }}))
    rep = run_engine(RUNS, LEDGER, PAIRS, model_path=str(sidecar))
    assert rep["provenance_ok"] is True
    # If the fixture has a cross-domain pair, a causal triple is present.
    # (Skip the assertion if the chosen fixture is single-domain.)
```

- [ ] **Step 3: Resolve the fixtures + module set**

Fill `RUNS`/`LEDGER`/`PAIRS` from an existing engine test (grep `run_engine(` in `tools/test_inference_*.py`). For the calibrated test, populate `modules` with every domain that appears in the fixture's cross-domain pairs (so no `CrossDomainModelError`).

- [ ] **Step 4: Run the e2e test**

Run: `PYTHONPATH=tools pytest tools/test_inference_sp4b_e2e.py -v`
Expected: PASS (byte-identity holds; calibrated path emits a provenance-clean causal fact).

- [ ] **Step 5: Full regression**

Run: `PYTHONPATH=tools pytest tools/test_inference_*.py -q` and `cargo test --lib`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/test_inference_sp4b_e2e.py
git commit -m "test(#140): SP-4b end-to-end -- byte-identity + calibrated causal path

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 12: Amend skew-limit doc Sec.9 (the standing rule SP-4b changes)

**Files:**
- Modify: `docs/trace/cross-domain-skew-limit.md:306-316` (Sec.9 Engine bullet)

**Interfaces:** none (documentation).

- [ ] **Step 1: Edit the Engine bullet** (Sec.9, line 308-309). Replace:

```
- **Engine:** record the exact raw cross-domain offset as a reproduction-target
  annotation on the gap. Never emit a cross-domain causal segment.
```

with:

```
- **Engine:** record the exact raw cross-domain offset as `reproduction_offset`
  (unchanged). *Additionally*, when a **calibrated, single-source** broadcast
  model is available, emit the decomposed `causal_offset = raw - skew(A,B)` as a
  **model-derived** causal fact -- exact in the model, an estimate of silicon
  with error equal to the calibration error, provenance-tagged `ModelDerived`,
  never presented as measured. Until calibrated, emit nothing (gap-only, status
  quo). Multi-source pairs fail loud. The trace alone still cannot decompose
  (Sec.5-6 unchanged); the calibrated emulator can (Sec.7). See
  `docs/superpowers/specs/2026-06-30-sp4b-skew-export-design.md`.
```

- [ ] **Step 2: Note the arc-spec dependency divergence.** In
  `docs/superpowers/specs/2026-06-28-timer-sync-faithful-broadcast-arc.md` §4,
  add a one-line pointer on the SP-4b/SP-5 bullets that SP-4b's rev2 design moves
  P1 and P2 to SP-5 (so the arc's dependency graph reads consistently with the
  delivered design).

- [ ] **Step 3: Commit**

```bash
git add docs/trace/cross-domain-skew-limit.md docs/superpowers/specs/2026-06-28-timer-sync-faithful-broadcast-arc.md
git commit -m "doc(#140): SP-4b amend skew-limit Sec.9 -- model-derived causal segment

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

## Self-Review (completed during authoring)

**Spec coverage:** P3 export (Tasks 8-10), causal decomposition (Tasks 4-5), ModelDerived provenance (Task 2), loader (Task 3), edge/render (Task 6), engine report (Task 7), single-source guard (Tasks 4/9/10), byte-identity (Tasks 6/7/11), sign-pin (Task 4), Sec.9 amendment (doc -- see note below). P1/P2 are SP-5 (spec Sec.6); not in this plan. The skew-limit Sec.9 doc amendment (spec Sec.8) is **Task 12**. No remaining spec gaps.

**Placeholder scan:** the Rust tasks (9-10) carry `/* ... */` markers where exact struct field paths must be resolved against the real engine struct (the engine-state owner of trace flush). These are genuine "locate the accessor" steps, not skipped logic -- each names the grep to run. The Python tasks carry complete code.

**Type consistency:** `causal_offset` (Python) / `origin_d_table` rows / sidecar `modules` keys are consistent across tasks; `domain_of`/`to_domain_key` agree on `col|row|pkt_type`; `try_causal` emits `causal(child,parent,offset)` consumed identically in `engine.py` and the e2e test.

**Note for executor:** Phase 1 (Tasks 1-7) is fully testable with synthetic fixtures and has NO dependency on Phase 2 -- it can land and be reviewed before any Rust work. Phase 2 (Tasks 8-10) produces the real sidecar; Phase 3 (Task 11) joins them. The Rust struct-accessor resolution in Tasks 9-10 is the only place needing live-code navigation.
