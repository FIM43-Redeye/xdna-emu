# SP-3 Phase 1: Engine Event-Menu Completeness + On-Chip Q=0 Spike

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inference engine's swept event universe complete (derived from
the toolchain, not hand-curated), then cheaply de-risk the SP-3 gate-carrier by
proving on real NPU1 that an *entirely on-chip* kernel yields Q=0 within-domain
segments and a non-None cross-domain reproduction offset.

**Architecture:** Two independent deliverables. (A) Replace `selfmodel._MENU`'s
27-event hand-list with a per-module event set derived from
`trace_capture.load_event_ids` (the same toolchain table `configure_batch`
already validates against), minus a documented sentinel-exclusion policy. (B) A
minimal hand-written raw-MLIR kernel whose data originates *on-chip* (no DDR in
the measured path) and whose couplings ride non-circular DMA-task events, captured
~20x on HW and gated on cross-run range 0. (A) unblocks the sweep from reaching
memtile/memmod task events; (B) settles the empirical question that two_col's
thousand-cycle jitter raised, before any large MLIR investment.

**Tech Stack:** Python 3.13 + pytest (engine, under `tools/`); raw MLIR + AIE2 DMA
task API (kernel); XRT bridge-test harness for HW capture on NPU1 (Phoenix).

## REVISION PENDING -- fold in before execution (adversarial pass 2026-06-29)

**Status:** plan DRAFTED; two adversarial passes complete (spec + plan); NOT yet
executed. The body below is the pre-revision draft -- apply the decisions and
fixes here first, then re-scan and execute via subagent-driven-development.

### Decisions (Maya, 2026-06-29)

- **D1 -- "Complete" = maximal literal completeness, include-and-deprioritize (NOT
  exclude).** Keep trace-framework infrastructure (`BROADCAST_*`, `USER_EVENT_*`,
  `TIMER_SYNC`) and always-on level signals (`ACTIVE`/`DISABLED`/`DEBUG_HALTED`/
  `PORT_IDLE_*`) IN the menu, marked low-priority / separable; the engine must not
  drop events. Only `NONE` (the slot-pad/disable sentinel) is hard-excluded.
  **Open sub-question to resolve in the revision:** the always-on *flooders*
  (`TRUE`/`ACTIVE`/`PORT_IDLE_*`) truncate the 8 KB trace buffer and kill
  `capture()` (`CaptureError`) if co-traced in the seed -- "deprioritize" fixes
  graph-pollution but NOT buffer-truncation. Design a flood-safe sweep (e.g.
  flooders stay in the universe but are traced in isolation / gated by the HW
  seed-smoke result), so completeness does not break capture. Rewrite Task 1 from
  "exclude four categories" to "include all but `NONE`, tier by priority, handle
  flooders safely."

- **D2 -- Spike enriched to 2-hop + cross-column.** The kernel gains a second hop
  and one cross-column coupling so a PASS de-risks the diagonal / memtile-in-path /
  multi-tile trace contention (the full-kernel risks, spec Sec.11), not merely
  single-hop on-chip determinism. Revise Task 2's topology accordingly; downscope
  nothing -- this is the richer spike.

### Task 1 fixes (engine completeness)

- Reframe per **D1** (include + tier + flood-safe), not exclude.
- Run ONE cheap HW seed-sweep smoke on `add_one` under the complete menu; the
  events that truncate the buffer define the flood-handling set (empirical, not
  guessed). HW is the cheap oracle.
- State the honest seed cost: ~19 batches / ~10x HW runs on EVERY kernel
  (module-bounded, not size-bounded). Re-scope the `n_runs=1` seed-prune as a
  near-term follow-up, not an indefinite "future optimization."
- Widen the Step-5 regression run to include `test_timeline.py` and `config_extract/`.
- MINOR (verified): the `RSVD`/`RESERVED` clause is dead code for aieml (no such
  header entries) -- keep it defensive, soften the comment. There is NO real
  circular import (`trace_capture` does not import `selfmodel`); the lazy import is
  harmless but its stated rationale is wrong. Add a pytest autouse fixture resetting
  `_MENU_CACHE` (latent stale-cache footgun under future monkeypatching).

### Task 2/3 fixes (spike)

- **`PERF_CNT_2` anchor is not free.** `vec_mul_event_trace` does NOT configure a
  perf counter. Configure it explicitly (`Performance_Control1` Cnt2_Start=ACTIVE,
  `Performance_Control2`=0x00070000 self-reset, `Counter2_Event_Value`=period; cf.
  `tools/mlir-trace-inject.py:506-587`, `tools/perfcnt_defaults.py`) and size the
  core body / period so it actually FIRES (it fires every ~period *active* cycles,
  not once per run). DROP the "no-core / pure-DMA-passthrough" fallback -- it kills
  the anchor (no active cycles -> no `PERF_CNT_2`).
- **Task 2 is from-scratch MLIR.** No template combines core self-generation + the
  task API + core-to-core on-chip flow (`core_dmas` is DDR-fed, no core body; the
  only self-gen example uses circular BDs). Budget real bring-up. Specify concretely:
  core writes a local buffer behind a lock; a terminating (`aie.end`) memory-module
  MM2S task streams it to the neighbor; the neighbor's S2MM task receives behind an
  interlocking lock.
- **Custom trace-validating host.** Every sibling `test.cpp` memcmps a DDR output
  the on-chip kernel won't produce -> "copy a sibling host" yields a false FAIL.
  Either write a minimal host that validates trace presence (no data check) or skip
  the `emu-bridge-test.sh` PASS gate for the smoke run and decode via
  `trace_capture.capture` directly (the gate path needs no DDR I/O -- confirmed).
- **Within-domain pair choice.** Do NOT anchor Q=0 on `MM2S_START->FINISHED` (the
  most backpressure-coupled pair -> risks `GAP_WITHIN_DOMAIN_NONEXACT`). Measure
  SEVERAL candidate within-domain pairs and report which reach range 0 (diagnostic);
  spec Sec.9 used the lower-coupling `S2MM_FINISHED->MM2S_START`.
- **Gate hardening.** Add `same_domain(within_pair)` / `not same_domain(cross_pair)`
  defensive asserts in `evaluate`. Drop the redundant `ANCHOR` key from the synthetic
  test row dicts (the `_runs` helper auto-injects `PERF_CNT_2@1000`). Do not read a
  passing cross leg as "causal coupling validated" -- it reproduces because both
  endpoints are deterministic vs the anchor, not from a tight producer->consumer link.

### After revision

Quick re-scan of the revised plan, then execute via
superpowers:subagent-driven-development.

---

## Global Constraints

- **Derive from the toolchain.** The event menu is derived from
  `trace_capture.load_event_ids(tile_type)`, never hand-listed. The same source
  `configure_batch` validates against, so menu names validate by construction.
- **Sentinel-exclusion policy (the only curation), each toolchain-grounded:**
  exclude `NONE` (disable/pad sentinel, id 0), `TRUE` (fires every cycle, floods
  the 8 KB trace buffer), `GROUP_*` (aggregate group-enable config events), and
  `rsvd_*`/`RSVD_*` (reserved). Everything else is in; non-firing events (errors,
  combos) are pruned by the existing never-fired logic after the seed sweep.
- **No menu regression.** Every event the legacy hand-list enumerated that is a
  valid toolchain event stays enumerated.
- **Spike kernel is entirely on-chip.** No DDR feed or drain in the measured
  path; the shim exists only to generate the BROADCAST_15 sync and drain the
  trace buffer (post-hoc, timing-irrelevant). Data originates on-chip.
- **Non-circular BD.** Spike DMAs use the task API
  (`aiex.dma_configure_task`/`dma_start_task`/`dma_await_task`/`dma_free_task`),
  terminating chains (`aie.end`), so `DMA_*_TASK` events fire once, in-window.
- **Uniform broadcast=15 reset** on every traced *module* (per-module, not
  per-tile -- DMA task events live in the memory module).
- **Q=0 is measured, not tuned:** a within-domain offset qualifies iff its
  cross-run range is 0 over the run set (`verifier.Q == 0`).
- **Python:** run tests with `cd tools && python -m pytest <file> -v`. No emoji
  anywhere. Commit after each green task.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `tools/inference/selfmodel.py` | event menu + enumeration | replace `_MENU` literal with derived `complete_menu()`; update the one consumer | 
| `tools/test_selfmodel.py` | selfmodel unit tests | add menu-completeness tests |
| `mlir-aie/test/npu-xrt/onchip_spike/aie.mlir` | the spike kernel | create (raw MLIR) |
| `mlir-aie/test/npu-xrt/onchip_spike/` (harness files) | bridge-test integration | create (CMake/run glue per sibling kernels) |
| `tools/inference/spike_gate.py` | the Q=0 / non-None gate script | create |
| `tools/test_spike_gate.py` | gate-script unit test (synthetic dirs) | create |

---

## Task 1: Derive the complete event menu from the toolchain

**Files:**
- Modify: `tools/inference/selfmodel.py:19-66` (replace the `_MENU` dict literal and
  update `enumerate_configured_events`'s single menu read)
- Test: `tools/test_selfmodel.py`

**Interfaces:**
- Consumes: `trace_capture.load_event_ids(tile_type: str) -> Dict[str, int]`
  (parses the aie-rt event header per module) and
  `trace_capture.PKT_TO_TILE_TYPE = {0:"core",1:"memmod",2:"shim",3:"memtile"}`.
- Produces: `selfmodel.complete_menu() -> Dict[int, List[str]]` (pkt_type ->
  ordered event names), cached. `enumerate_configured_events` unchanged in
  signature; now reads `complete_menu()` instead of the literal.

**Background the implementer needs:** `selfmodel.py` currently hardcodes `_MENU`
(a 27-event hand-list) and `enumerate_configured_events` reads `_MENU.get(pkt, [])`
at line 59. The hand-list omits memtile DMA-task events and memmod
`_FINISHED_TASK`, so the sweep can never place them -- permanent timeline blind
spots. `load_event_ids` already returns the *complete* `{name: id}` for a module
from the toolchain header; we derive the menu from it. Note `selfmodel` already
imports `trace_capture` lazily inside `legal_batch` (line ~85) to avoid a circular
import -- do the same here (import inside the function, not at module top).

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_selfmodel.py` (it already imports from `inference.selfmodel`):

```python
from inference.selfmodel import complete_menu, enumerate_configured_events
from inference.selfmodel import legal_batch
from inference.planner import Batch
from trace_capture import load_event_ids, PKT_TO_TILE_TYPE, build_active_plan


def test_complete_menu_names_all_validate():
    # Every menu name must exist in the toolchain table for its module, so
    # configure_batch never raises on a menu-sourced event.
    menu = complete_menu()
    for pkt, names in menu.items():
        ids = load_event_ids(PKT_TO_TILE_TYPE[pkt])
        for n in names:
            assert n in ids, (pkt, n)


def test_complete_menu_excludes_sentinels():
    for pkt, names in complete_menu().items():
        assert "NONE" not in names
        assert "TRUE" not in names
        assert not any(n.startswith("GROUP_") for n in names)
        assert not any(n.upper().startswith("RSVD") for n in names)


def test_complete_menu_fills_the_reviewer_gaps():
    menu = complete_menu()
    # memtile (pkt 3) must now expose DMA task boundaries (named *_SEL0/SEL1_*).
    assert any("DMA" in n and "START_TASK" in n for n in menu[3]), menu[3]
    # memmod (pkt 1) must now expose FINISHED_TASK, absent from the hand-list.
    assert any("FINISHED_TASK" in n for n in menu[1]), menu[1]


def test_complete_menu_preserves_known_firing_events():
    # No regression: the add_one known-firing set stays enumerable.
    menu = complete_menu()
    assert "PORT_RUNNING_0" in menu[3] and "PORT_RUNNING_4" in menu[3]
    assert "DMA_S2MM_0_START_TASK" in menu[2] and "DMA_MM2S_0_START_TASK" in menu[2]
    assert "PERF_CNT_2" in menu[0]


def test_complete_menu_enumeration_builds_legal_batches():
    # The (larger) complete menu must still pack into <=8-slot legal batches.
    evs = enumerate_configured_events(_dump(), start_col=1)
    active = {}
    for k in evs:
        col, row, pkt, name = k.split("|")
        active.setdefault(f"{col}|{row}|{pkt}", set()).add(name)
    plan = build_active_plan(active)
    for b in plan["batches"]:
        assert legal_batch(Batch(tiles=b)), b
```

`_dump()` already exists in `test_selfmodel.py` (used by the existing
`test_enumeration_superset_of_known_add_one_events`); reuse it. If `legal_batch`
expects a `Batch`, confirm the import path (`inference.planner.Batch`) against the
existing tests before running.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools && python -m pytest test_selfmodel.py -v -k "complete_menu"`
Expected: FAIL with `ImportError: cannot import name 'complete_menu'`.

- [ ] **Step 3: Implement `complete_menu` and the exclusion policy**

In `tools/inference/selfmodel.py`, delete the `_MENU = {...}` literal (lines 19-33)
and replace with:

```python
# The swept event menu is the COMPLETE per-module traceable event set, derived
# from the toolchain table (trace_capture.load_event_ids) so the sweep can reach
# every event the hardware can trace. The only curation is dropping four
# non-measurement categories, each toolchain-grounded:
#   NONE    -- the disable sentinel (id 0; also the slot-pad value)
#   TRUE    -- fires unconditionally every cycle; floods the 8 KB trace buffer
#   GROUP_* -- aggregate group-enable config events, not individual measurements
#   rsvd_*  -- reserved / undefined event ids
def _is_swept_event(name: str) -> bool:
    if name in ("NONE", "TRUE"):
        return False
    if name.startswith("GROUP_"):
        return False
    if name.upper().startswith("RSVD") or name.upper().startswith("RESERVED"):
        return False
    return True


_MENU_CACHE: Optional[Dict[int, List[str]]] = None


def complete_menu() -> Dict[int, List[str]]:
    """Complete per-packet-type swept event menu, derived from the toolchain
    event table. Names are ordered by event id and validate against
    configure_batch by construction. Cached after first call."""
    global _MENU_CACHE
    if _MENU_CACHE is not None:
        return _MENU_CACHE
    from trace_capture import load_event_ids, PKT_TO_TILE_TYPE  # lazy: avoid cycle
    menu: Dict[int, List[str]] = {}
    for pkt, tile_type in PKT_TO_TILE_TYPE.items():
        ids = load_event_ids(tile_type)
        menu[pkt] = sorted((n for n in ids if _is_swept_event(n)), key=lambda n: ids[n])
    _MENU_CACHE = menu
    return menu
```

Then change `enumerate_configured_events` (the `for name in _MENU.get(pkt, []):`
line, ~59) to `for name in complete_menu().get(pkt, []):`. Ensure `Optional` is
imported (`from typing import Optional` -- it likely already imports `Dict, List`).
Grep the file for any other `_MENU` reference and update it: `grep -n _MENU
tools/inference/selfmodel.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools && python -m pytest test_selfmodel.py -v`
Expected: PASS (the new tests and all pre-existing selfmodel tests). If
`test_complete_menu_fills_the_reviewer_gaps` fails, print
`load_event_ids("memtile")` keys to confirm the exact memtile task-event names and
adjust the substring assertion -- do NOT hand-add names to the menu.

- [ ] **Step 5: Run the broader engine test suite for regressions**

Run: `cd tools && python -m pytest test_selfmodel.py test_trace_capture.py test_hw_instrument.py -v`
Expected: PASS. The complete menu is larger, so `enumerate_configured_events`
returns more events and `build_active_plan` produces more batches; any test that
asserted an exact enumerated-event *count* (not membership) must be updated to
assert membership/superset, which is the correct invariant.

- [ ] **Step 6: Commit**

```bash
git add tools/inference/selfmodel.py tools/test_selfmodel.py
git commit -m "feat(#140): derive complete event menu from toolchain (engine completeness)"
```

---

## Task 2: The on-chip spike kernel

**Files:**
- Create: `mlir-aie/test/npu-xrt/onchip_spike/aie.mlir`
- Create: harness glue mirroring a sibling 2-tile npu-xrt kernel
  (`CMakeLists.txt` / `run.lit` / host `test.cpp` as the discover step expects)

**Interfaces:**
- Produces: a compiled xclbin + insts the bridge-test harness can run on NPU1,
  emitting a labeled trace with the spike's events firing in-window.

**Kernel design (from the SP-3 spec, minimized for the spike):**

```
            col 0
  row 3   core(0,3)   consumer (receives on-chip stream; trivial body)
  row 2   core(0,2)   producer (fills local mem from a constant pattern, streams out)
  row 1   --          (unused)
  row 0   shim(0,0)   BROADCAST_15 sync generator + trace drain ONLY
```

- **On-chip origination:** `core(0,2)`'s core body writes a fixed pattern (e.g.
  `for i: buf[i] = i`) into its local memory -- no input DMA. Its memory-module
  MM2S task streams `buf` to `core(0,3)`; `core(0,3)`'s S2MM task receives into
  local memory; its core body is trivial (passthrough or `+const`). **No DDR
  touches the measured path.** (Reference for core-internal data + task-API DMA:
  `mlir-aie/test/npu-xrt/core_dmas/dma_configure_task_lock/aie.mlir`; it runs on
  NPU1 and proves host-issued terminating DMA tasks on a compute tile. If the
  core-body + lock + task-DMA interlock proves fiddly, the fallback is a pure
  local-to-local DMA passthrough with no `aie.core` -- `core_dmas` shows this
  works -- since the timing gate rides the DMA task events, not the arithmetic.)
- **Couplings the spike measures (both on-chip, deterministic):**
  - *Within-domain anchor (Q=0 candidate):* a same-module pair on `core(0,2)`'s
    memory module, e.g. `DMA_MM2S_0_START_TASK -> DMA_MM2S_0_FINISHED_TASK`
    (both pkt_type 1, same `col|row|pkt` -> same domain).
  - *Cross-domain coupling (non-None target):* `core(0,2)` memmod
    `DMA_MM2S_0_FINISHED_TASK` -> `core(0,3)` memmod `DMA_S2MM_0_START_TASK`
    (rows 2 vs 3 -> different domain). Δn `(0,1)`, one on-chip hop.
- **Anchor event:** configure a performance counter on `core(0,2)` so it emits a
  stable, once-per-run anchor (the engine pins all offsets to one anchor;
  default key `1|2|0|PERF_CNT_2`). Trace `PERF_CNT_2` on `core(0,2)`'s core
  module. (Confirm the perf-counter config against `vec_mul_event_trace`.)
- **Trace config (template: `mlir-aie/test/npu-xrt/vec_mul_event_trace/aie.mlir`):**
  per-module `aie.trace` blocks with distinct packet ids; trace the events above;
  `aie.trace.start broadcast=15` / `stop broadcast=14` on **every traced module**
  (core module of `core(0,2)` for the anchor; memory module of `core(0,2)` and
  `core(0,3)` for the DMA-task carriers). The shim generates the sync.

- [ ] **Step 1: Author `aie.mlir`** against the templates above (device
  `npu1_1col`; tiles shim(0,0), core(0,2), core(0,3)). On-chip producer/consumer,
  task-API DMAs, per-module trace blocks, uniform `broadcast=15`.

- [ ] **Step 2: Add the harness glue** so the bridge-test discover step finds it
  (copy the structure of a sibling 2-tile npu-xrt kernel's `CMakeLists.txt`/`run.lit`/host).

- [ ] **Step 3: Compile (Chess is ground truth; Peano informational)**

Run: `./scripts/emu-bridge-test.sh --compile -v onchip_spike`
Expected: a Chess build under `mlir-aie/build/test/npu-xrt/onchip_spike/chess/`
with an xclbin + insts. Fix MLIR errors until it compiles.

- [ ] **Step 4: Single HW smoke run, confirm the target events fire in-window**

Run (real HW; from a clean shell):
`env -u XDNA_EMU ./scripts/emu-bridge-test.sh --no-hw=false --serial-hw -v onchip_spike`
(or the harness's single-kernel HW invocation). Then decode one trace and confirm
the four target events (`core(0,2)` MM2S START+FINISHED, `core(0,3)` S2MM START,
`core(0,2)` PERF_CNT_2) each appear exactly once per run, none looping.
Expected: all four present, single-shot. If a `*_TASK` event is missing, the BD
chain is circular or the task didn't run -- fix before Task 3.

- [ ] **Step 5: Commit**

```bash
git add mlir-aie/test/npu-xrt/onchip_spike/
git commit -m "feat(#140): on-chip spike kernel (no-DDR, task-API, traced) for SP-3 Q=0 de-risk"
```

---

## Task 3: The Q=0 / non-None gate, and the decision

**Files:**
- Create: `tools/inference/spike_gate.py`
- Test: `tools/test_spike_gate.py`

**Interfaces:**
- Consumes: `verifier`/`timeline` primitives -- `inference.verifier.offset_exact(run_dirs, child, parent, anchor_key) -> Optional[int]`
  (non-None iff cross-run range <= Q == 0) and
  `inference.grounding.ground_edge(run_dirs, child, parent, anchor_key) -> Grounding`
  (a cross-domain `Gap` carrying `reproduction_offset`, non-None iff the raw
  offset agrees across runs).
- Produces: `spike_gate.evaluate(run_dirs, *, anchor_key, within_pair, cross_pair)
  -> dict` with `{"within_range0": bool, "cross_reproduction": Optional[int],
  "pass": bool}`, and a `__main__` CLI that prints the verdict over the captured
  run dirs.

**Background:** the gate uses the engine's own primitives so "Q=0" and "non-None
reproduction" mean exactly what the engine means. A pair is `(child_key,
parent_key)` with keys `col|row|pkt|NAME`. `within_pair` is the same-domain anchor
pair on `core(0,2)`; `cross_pair` is the cross-domain `core(0,2)->core(0,3)` pair.

- [ ] **Step 1: Write the failing test** (synthetic run dirs, no HW)

Create `tools/test_spike_gate.py`. Mirror the run-dir fixture style of
`tools/test_inference_grounding.py` (its `_runs(tmp_path, [...])` helper builds
fake co-traced run dirs from dicts of `event_key -> soc`); import and reuse that
helper if exported, else copy its construction.

```python
from inference.spike_gate import evaluate

W_PARENT = "1|2|1|DMA_MM2S_0_START_TASK"
W_CHILD  = "1|2|1|DMA_MM2S_0_FINISHED_TASK"   # same domain (1|2|1) -> within
X_PARENT = "1|2|1|DMA_MM2S_0_FINISHED_TASK"
X_CHILD  = "1|3|1|DMA_S2MM_0_START_TASK"      # rows 2 vs 3 -> cross-domain
ANCHOR   = "1|2|0|PERF_CNT_2"

def test_gate_passes_when_within_range0_and_cross_reproduces(tmp_path):
    # Two runs; within-offset identical (range 0), cross-offset identical (range 0).
    runs = _runs(tmp_path, [
        {ANCHOR: 0, W_PARENT: 10, W_CHILD: 30, X_CHILD: 35},
        {ANCHOR: 0, W_PARENT: 50, W_CHILD: 70, X_CHILD: 75},  # both offsets stable
    ])
    res = evaluate(runs, anchor_key=ANCHOR,
                   within_pair=(W_CHILD, W_PARENT), cross_pair=(X_CHILD, X_PARENT))
    assert res["within_range0"] is True
    assert res["cross_reproduction"] == 5
    assert res["pass"] is True

def test_gate_fails_when_within_jitters(tmp_path):
    runs = _runs(tmp_path, [
        {ANCHOR: 0, W_PARENT: 10, W_CHILD: 30, X_CHILD: 35},
        {ANCHOR: 0, W_PARENT: 50, W_CHILD: 71, X_CHILD: 75},  # within range 1
    ])
    res = evaluate(runs, anchor_key=ANCHOR,
                   within_pair=(W_CHILD, W_PARENT), cross_pair=(X_CHILD, X_PARENT))
    assert res["within_range0"] is False
    assert res["pass"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd tools && python -m pytest test_spike_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.spike_gate'`.

- [ ] **Step 3: Implement `spike_gate.evaluate`**

```python
"""SP-3 on-chip spike gate: does an entirely-on-chip kernel give Q=0 within-domain
and a non-None cross-domain reproduction offset on real NPU1?"""
from typing import List, Optional, Tuple
from inference.verifier import offset_exact
from inference.grounding import ground_edge, Gap


def evaluate(run_dirs: List[str], *, anchor_key: str,
             within_pair: Tuple[str, str],
             cross_pair: Tuple[str, str]) -> dict:
    w_child, w_parent = within_pair
    x_child, x_parent = cross_pair
    within = offset_exact(run_dirs, w_child, w_parent, anchor_key)  # None unless range 0
    g = ground_edge(run_dirs, x_child, x_parent, anchor_key)
    cross = g.reproduction_offset if isinstance(g, Gap) else None
    res = {
        "within_range0": within is not None,
        "within_offset": within,
        "cross_reproduction": cross,
        "pass": within is not None and cross is not None,
    }
    return res
```

Add a `__main__` that takes `--runs <dir>...`, `--anchor`, `--within child parent`,
`--cross child parent`, calls `evaluate`, prints the dict, and exits nonzero on
fail. (Confirm `Gap` and `reproduction_offset` against `grounding.py:105-107`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd tools && python -m pytest test_spike_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Capture ~20 HW runs of the spike and run the gate**

Capture the four target events over 20 runs on NPU1 (a single targeted batch fits
one <=8-slot batch with the anchor; use the engine's `HwInstrument`/`capture` or a
direct `trace_capture.capture` with a hand-built one-batch plan -- the events fit
8 slots so no multi-batch split is needed). Then:

Run: `cd tools && python -m inference.spike_gate --runs <20 run dirs> \
  --anchor '1|2|0|PERF_CNT_2' \
  --within '1|2|1|DMA_MM2S_0_FINISHED_TASK' '1|2|1|DMA_MM2S_0_START_TASK' \
  --cross  '1|3|1|DMA_S2MM_0_START_TASK' '1|2|1|DMA_MM2S_0_FINISHED_TASK'`
Expected: prints the verdict. (Event names/keys must match the kernel's actual
traced events and absolute columns from Task 2's smoke decode -- adjust if the
harness places the kernel at a different start column.)

- [ ] **Step 6: Record the decision in the progress ledger**

This is the plan's terminal gate, not a code step:
- **PASS** (within-domain range 0 AND cross-domain reproduction non-None): the
  on-chip approach delivers Q=0. Proceed to the full rank-2 SP-3 kernel (separate
  spec/plan), confident in the topology.
- **FAIL:** the cheap HW oracle has told us the on-chip premise is insufficient
  *before* the large MLIR investment. Stop and reconsider -- do not proceed to the
  full kernel. Capture which leg failed (within-domain jitter vs cross-domain
  non-reproduction) and the observed ranges for the redesign.

Record the verdict, the observed within-domain range, and the cross-domain
reproduction value in `.superpowers/sdd/progress.md`.

---

## Self-Review

**Spec coverage.** This plan implements the two pieces Maya scoped: engine
completeness (Task 1) and the on-chip Q=0 spike (Tasks 2-3). The full rank-2
kernel, the SP-4a per-module reset invariant in production form, batch packing for
the full event set, and the `cross_track_edges` acceptance assertion are
deliberately *deferred* to the post-spike full-kernel plan -- in scope only if the
spike passes.

**Placeholder scan.** No TBD/TODO. The MLIR in Task 2 is specified by design +
template reference rather than verbatim source, because a novel HW kernel is
brought up by iteration against the cited templates; the *requirements* (on-chip
origination, task API, per-module broadcast=15, the four traced events) are
concrete and checkable, and Task 2's smoke step is the gate.

**Type consistency.** `complete_menu() -> Dict[int, List[str]]` consumed by
`enumerate_configured_events`; `evaluate(...) -> dict` with the keys the CLI and
tests read; pair tuples are `(child, parent)` consistently (matching
`offset_exact`/`ground_edge` argument order). Event keys are `col|row|pkt|NAME`
throughout.

**Known tradeoff flagged for the adversarial pass:** a complete menu enlarges the
seed sweep (more events traced once before never-fired pruning). For the small
spike kernel this is negligible (targeted capture, not a full sweep); for large
kernels it is a real cost. A future optimization (seed-prune at n_runs=1, then
n_runs on survivors) is out of scope here and noted, not built.
