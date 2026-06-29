# SP-3 Phase 1: Engine Event-Menu Completeness + On-Chip Q=0 Spike

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inference engine's swept event universe *complete* (derived from
the toolchain, not hand-curated) yet *flood-safe* (the complete universe never
breaks capture), then cheaply de-risk the SP-3 gate-carrier by proving on real
NPU1 that an *entirely on-chip*, 2-hop, cross-column kernel yields Q=0
within-domain segments and non-None cross-domain reproduction offsets across a
rank-2 coupling set.

**Architecture:** Two independent deliverables.

- **(A) Engine completeness, flood-safe.** Replace `selfmodel._MENU`'s 27-event
  hand-list with two toolchain-derived views: `complete_menu()` (the *universe* --
  every traceable event per module from `trace_capture.load_event_ids`, minus only
  the `NONE` sentinel) and `swept_menu()` (what the sweep actually enumerates --
  the universe minus a flood-risk *stateful* tier, with infra events
  deprioritized). The stateful tier (always-on level signals that truncate the
  trace buffer) stays *in the universe* (reachable, documented) but *out of the
  default co-traced batches*, so completeness never breaks `capture()`. A single
  cheap HW seed-smoke on `add_one` validates the tiering empirically (the swept
  set captures clean and reaches the previously-blind memtile/memmod task events;
  the stateful prior is the events that actually truncate).

- **(B) On-chip Q=0 spike, enriched.** A hand-written raw-MLIR kernel whose data
  originates *on-chip* (no DDR in the measured path), routed core->core across two
  hops and two columns, so its couplings span a *rank-2* direction-diverse set
  (vertical Delta_n (0,1) + diagonal Delta_n (1,1), hop counts {1,2}). Captured
  ~20x on HW and gated on within-domain cross-run range 0 + non-None cross-domain
  reproduction. (A) unblocks the sweep from reaching memtile/memmod task events;
  (B) settles -- before any large MLIR investment -- whether the on-chip premise
  actually delivers Q=0 across the *structure the full kernel needs* (diagonal,
  cross-column, multi-tile contention), not merely single-hop determinism.

**Tech Stack:** Python 3.13 + pytest (engine, under `tools/`); raw MLIR + AIE2 DMA
task API (kernel); XRT bridge-test harness for HW capture on NPU1 (Phoenix).

---

## Global Constraints

- **Derive from the toolchain.** The event universe is derived from
  `trace_capture.load_event_ids(tile_type)`, never hand-listed -- the same source
  `configure_batch` validates against, so menu names validate by construction.
- **Completeness policy (D1): include all but `NONE`; tier, don't drop.**
  - `complete_menu()` -- the *universe*: every event in the toolchain table for a
    module, minus only `NONE` (the disable/slot-pad sentinel, id 0). `TRUE`,
    `GROUP_*`, `BROADCAST_*`, level signals -- all present. The engine drops nothing.
  - **Tiers** (`_event_tier(name)`), toolchain-category-grounded:
    - `measurement` -- discrete dataflow/compute events: `DMA_*_TASK`,
      `PORT_RUNNING_*`/`PORT_STALLED_*`/`PORT_TLAST_*`, `PERF_CNT_*`, `LOCK_*`,
      `INSTR_*` (non-`INSTR_EVENT`), `STREAM_*`, `MEMORY_STALL`, `EDGE_DETECTION_*`,
      errors, etc. The high-priority bulk.
    - `infra` -- trace-framework carriers / aggregates, swept but *low-priority*
      (graph pollution, not buffer pollution -- safe to co-trace): `BROADCAST_*`,
      `USER_EVENT_*`, `TIMER_SYNC`, `INSTR_EVENT_*`, `GROUP_*`.
    - `stateful` -- always-on level signals that flood the buffer (conservative
      prior, HW-confirmed by the seed-smoke): `TRUE`, `ACTIVE`, `DISABLED`,
      `DEBUG_HALTED`, `PORT_IDLE_*`. **Kept in the universe, excluded from the
      default sweep** -- "separable" per D1. `PORT_RUNNING_*` is NOT stateful
      (it's a proven-swept measurement event, in `add_one`'s firing set).
  - `swept_menu()` -- what `enumerate_configured_events` draws from: `measurement`
    names (ordered by event id) followed by `infra` names (ordered by id),
    *excluding* `stateful`. Flood-safe by construction: no every-cycle event ever
    enters a co-traced batch, so `capture()` never truncates on a swept event.
- **Empirical flood set, not guessed.** The `stateful` predicate is a conservative
  prior. The HW seed-smoke (Task 1, Step 6) is the arbiter: run `swept_menu` on
  `add_one`; any batch that truncates names an event to promote into `stateful`
  (recorded). HW is the cheap oracle -- one capture settles it.
- **No menu regression.** Every event the legacy hand-list enumerated stays
  enumerated by `swept_menu` (all 27 are `measurement`-tier: DMA tasks,
  `PORT_RUNNING_*`, perf, lock/stall/instr, `EDGE_DETECTION`).
- **Spike kernel is entirely on-chip.** No DDR feed or drain in the measured
  path; the shim exists only to generate the BROADCAST_15 sync and drain the
  trace buffer (post-hoc, timing-irrelevant). Data originates on-chip in a core.
- **Non-circular BD.** Spike DMAs use the task API
  (`aiex.dma_configure_task`/`dma_start_task`/`dma_await_task`/`dma_free_task`),
  terminating chains (`aie.end`), so `DMA_*_TASK` events fire once, in-window.
- **Uniform broadcast=15 reset** on every traced *module* (per-module, not
  per-tile -- DMA task events live in the memory module).
- **Q=0 is measured, not tuned:** a within-domain offset qualifies iff its
  cross-run range is 0 over the run set (`verifier.offset_exact != None`, i.e.
  `range <= Q == 0`).
- **Python:** run tests with `cd tools && python -m pytest <file> -v`. No emoji
  anywhere. Commit after each green task.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `tools/inference/selfmodel.py` | event universe + tiers + enumeration | replace `_MENU` literal with `complete_menu()` / `_event_tier()` / `swept_menu()`; point the one consumer at `swept_menu()` |
| `tools/test_selfmodel.py` | selfmodel unit tests | add completeness + flood-safety + tier tests; autouse cache-reset fixture |
| `mlir-aie/test/npu-xrt/onchip_spike/aie.mlir` | the spike kernel | create (raw MLIR, from scratch) |
| `mlir-aie/test/npu-xrt/onchip_spike/` (harness files) | bridge-test integration | create (CMake/run glue per sibling kernels) |
| `tools/inference/spike_gate.py` | the Q=0 / non-None gate script | create |
| `tools/test_spike_gate.py` | gate-script unit test (synthetic dirs) | create |

---

## Task 1: Derive the complete, flood-safe event menu from the toolchain

**Files:**
- Modify: `tools/inference/selfmodel.py:16-66` (replace the `_MENU` dict literal and
  point `enumerate_configured_events`'s single menu read at `swept_menu()`)
- Test: `tools/test_selfmodel.py`

**Interfaces:**
- Consumes: `trace_capture.load_event_ids(tile_type: str) -> Dict[str, int]`
  (parses the aie-rt event header per module) and
  `trace_capture.PKT_TO_TILE_TYPE = {0:"core",1:"memmod",2:"shim",3:"memtile"}`.
- Produces:
  - `selfmodel.complete_menu() -> Dict[int, List[str]]` -- pkt_type -> every
    traceable event name (minus `NONE`), ordered by event id. Cached.
  - `selfmodel._event_tier(name: str) -> str` -- `"measurement"|"infra"|"stateful"`.
  - `selfmodel.swept_menu() -> Dict[int, List[str]]` -- pkt_type -> measurement
    names then infra names (stateful excluded), each ordered by id. Cached.
  - `enumerate_configured_events` unchanged in signature; now reads `swept_menu()`.

**Background the implementer needs:** `selfmodel.py` currently hardcodes `_MENU`
(a 27-event hand-list, lines 19-33) and `enumerate_configured_events` reads
`_MENU.get(pkt, [])` at line 59. The hand-list omits memtile DMA-task events
(`DMA_*_SEL0/SEL1_*_TASK`) and memmod `*_FINISHED_TASK`, so the sweep can never
place them -- permanent timeline blind spots. `load_event_ids` already returns the
*complete* `{name: id}` for a module from the toolchain header; we derive both
views from it.

**Verified facts (grounding done 2026-06-29):**
- Universe sizes per module: core 127, memmod ~122 (MEM minus MEM_TILE), memtile
  161, shim 128 -- vs the old 27. The complete sweep is ~5x larger; see the honest
  seed-cost note below.
- `RSVD`/`RESERVED`: **no such entries exist** for aieml. The legacy
  `rsvd_*`/`RSVD_*` exclusion clause is dead code -- keep one defensive `RSVD`
  guard in `_event_tier` (harmless, future-proofs other devices) but soften the
  comment to say so; do not claim it filters anything today.
- **No circular import.** `trace_capture` does not import `selfmodel`. The existing
  lazy import inside `legal_batch` is harmless but its "avoid a cycle" rationale is
  wrong; keep the lazy import in `complete_menu` (so module import stays clean and
  the header is read on first use, not at import), and fix the comment to say
  "lazy: header read on demand," not "avoid cycle."
- `TRUE`, `GROUP_*`, `BROADCAST_0..15`, `USER_EVENT_0..3`, `TIMER_SYNC`,
  `INSTR_EVENT_0..1`, and level signals `ACTIVE`/`DISABLED`/`DEBUG_HALTED`/
  `PORT_IDLE_*`/`PORT_STALLED_*`/`PORT_TLAST_*`/`PORT_RUNNING_*` all confirmed
  present in the header (names after stripping the `XAIEML_EVENTS_<MOD>_` prefix).

**Honest seed cost (record, don't hide):** the swept menu is module-bounded, not
size-bounded -- `build_active_plan` packs ceil(N/cap) batches where N is the
largest active module's swept count (~110-150). On a small kernel like `add_one`
that is ~19 seed batches vs the old ~2: roughly **10x the HW runs on every
kernel**, because every module's full swept set is traced once before the
never-fired pruning. This is a real cost, accepted for completeness. **Near-term
follow-up (noted, not built here):** a two-pass seed-prune (`n_runs=1` to discover
which events ever fire, then `n_runs` only on survivors) cuts the steady-state
cost; scope it as the next engine task, not an indefinite "future optimization."

- [ ] **Step 1: Write the failing tests**

Add to `tools/test_selfmodel.py` (it already imports from `inference.selfmodel`
and has a `_dump()` helper used by `test_enumeration_superset_of_known_add_one_events`).

```python
import pytest
from inference.selfmodel import (
    complete_menu, swept_menu, _event_tier, enumerate_configured_events, legal_batch,
)
from inference.planner import Batch
from trace_capture import load_event_ids, PKT_TO_TILE_TYPE, build_active_plan


@pytest.fixture(autouse=True)
def _reset_menu_caches():
    # complete_menu / swept_menu memoize at module scope; reset around each test
    # so monkeypatching load_event_ids in a future test can't leak a stale cache.
    import inference.selfmodel as sm
    sm._COMPLETE_CACHE = None
    sm._SWEPT_CACHE = None
    yield
    sm._COMPLETE_CACHE = None
    sm._SWEPT_CACHE = None


def test_complete_menu_names_all_validate():
    # Every universe name must exist in the toolchain table for its module.
    menu = complete_menu()
    for pkt, names in menu.items():
        ids = load_event_ids(PKT_TO_TILE_TYPE[pkt])
        for n in names:
            assert n in ids, (pkt, n)


def test_complete_menu_excludes_only_none():
    # D1: the universe drops NOTHING but NONE. TRUE / GROUP_* / BROADCAST_* stay.
    for pkt, names in complete_menu().items():
        assert "NONE" not in names
        assert "TRUE" in names
        assert any(n.startswith("GROUP_") for n in names)
        assert any(n.startswith("BROADCAST_") for n in names)


def test_swept_menu_is_flood_safe():
    # The default sweep excludes the stateful (always-on) flood-risk tier.
    for pkt, names in swept_menu().items():
        assert "TRUE" not in names
        assert "ACTIVE" not in names
        assert "DISABLED" not in names
        assert "DEBUG_HALTED" not in names
        assert not any(n.startswith("PORT_IDLE_") for n in names)


def test_swept_menu_subset_of_complete():
    comp, swp = complete_menu(), swept_menu()
    for pkt in comp:
        assert set(swp[pkt]) <= set(comp[pkt])


def test_swept_menu_keeps_measurement_port_running():
    # PORT_RUNNING is measurement, NOT stateful -- it must survive into the sweep.
    assert any(n.startswith("PORT_RUNNING_") for n in swept_menu()[3])  # memtile


def test_swept_menu_measurement_before_infra():
    # measurement-tier events sort ahead of infra-tier in the swept order, so
    # build_active_plan packs the useful events into the earliest batches.
    for pkt, names in swept_menu().items():
        tiers = [_event_tier(n) for n in names]
        last_meas = max((i for i, t in enumerate(tiers) if t == "measurement"),
                        default=-1)
        first_infra = next((i for i, t in enumerate(tiers) if t == "infra"),
                           len(names))
        assert last_meas < first_infra, (pkt, list(zip(names, tiers)))


def test_event_tiers():
    assert _event_tier("DMA_MM2S_0_START_TASK") == "measurement"
    assert _event_tier("PORT_RUNNING_3") == "measurement"
    assert _event_tier("PERF_CNT_2") == "measurement"
    assert _event_tier("BROADCAST_15") == "infra"
    assert _event_tier("GROUP_STALL") == "infra"
    assert _event_tier("USER_EVENT_0") == "infra"
    assert _event_tier("TRUE") == "stateful"
    assert _event_tier("PORT_IDLE_0") == "stateful"
    assert _event_tier("ACTIVE") == "stateful"


def test_swept_menu_fills_the_reviewer_gaps():
    menu = swept_menu()
    # memtile (pkt 3): DMA task boundaries (named *_SEL0/SEL1_*_TASK) now reachable.
    assert any("DMA" in n and "START_TASK" in n for n in menu[3]), menu[3]
    # memmod (pkt 1): FINISHED_TASK, absent from the hand-list, now reachable.
    assert any("FINISHED_TASK" in n for n in menu[1]), menu[1]


def test_swept_menu_no_regression_vs_handlist():
    # Every event the legacy hand-list enumerated stays enumerable.
    menu = swept_menu()
    assert "PORT_RUNNING_0" in menu[3] and "PORT_RUNNING_4" in menu[3]
    assert "DMA_S2MM_0_START_TASK" in menu[2] and "DMA_MM2S_0_START_TASK" in menu[2]
    assert "PERF_CNT_2" in menu[0]
    assert "DMA_MM2S_0_START_TASK" in menu[1] and "EDGE_DETECTION_EVENT_0" in menu[1]


def test_enumeration_builds_legal_batches():
    # The (larger) swept menu must still pack into <=8-slot legal batches.
    evs = enumerate_configured_events(_dump(), start_col=1)
    active = {}
    for k in evs:
        col, row, pkt, name = k.split("|")
        active.setdefault(f"{col}|{row}|{pkt}", set()).add(name)
    plan = build_active_plan(active)
    for b in plan["batches"]:
        assert legal_batch(Batch(tiles=b)), b
```

Reuse the existing `_dump()` helper. Confirm the `Batch` import path
(`inference.planner.Batch`) against the existing tests before running.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd tools && python -m pytest test_selfmodel.py -v -k "menu or tier"`
Expected: FAIL with `ImportError: cannot import name 'complete_menu'`.

- [ ] **Step 3: Implement the universe, tiers, and the flood-safe swept view**

In `tools/inference/selfmodel.py`, delete the `_MENU = {...}` literal (lines 19-33)
and replace with:

```python
# The trace-event menu is derived COMPLETE from the toolchain table
# (trace_capture.load_event_ids) -- the engine knows every event the hardware can
# trace. Two views:
#   complete_menu() -- the universe: all events minus NONE (the disable/pad sentinel).
#   swept_menu()    -- what the sweep enumerates: measurement events first, infra
#                      events last, the always-on "stateful" flood-risk tier excluded
#                      (still in the universe, just kept out of co-traced batches so
#                      capture() never truncates).
# Tiers are toolchain-category classifications; see docs and _event_tier below.

_INFRA_PREFIXES = ("BROADCAST_", "USER_EVENT_", "INSTR_EVENT_", "GROUP_")
_STATEFUL_EXACT = {"TRUE", "ACTIVE", "DISABLED", "DEBUG_HALTED"}


def _event_tier(name: str) -> str:
    """Classify a (prefix-stripped) event name: measurement | infra | stateful.

    stateful = always-on level signals that flood the 2 MB trace buffer (TRUE,
    core enable/halt state, PORT_IDLE_*). Conservative prior; the HW seed-smoke
    promotes any further event that actually truncates. PORT_RUNNING/STALLED/TLAST
    are discrete measurement events, NOT stateful.
    """
    if name in _STATEFUL_EXACT or name.startswith("PORT_IDLE_"):
        return "stateful"
    if name == "TIMER_SYNC" or name.startswith(_INFRA_PREFIXES):
        return "infra"
    # Defensive: no RSVD/RESERVED entries exist for aieml today, but guard other
    # devices -- treat reserved ids as infra (low priority), never measurement.
    if name.upper().startswith("RSVD") or name.upper().startswith("RESERVED"):
        return "infra"
    return "measurement"


_COMPLETE_CACHE: "Dict[int, List[str]] | None" = None
_SWEPT_CACHE: "Dict[int, List[str]] | None" = None


def complete_menu() -> Dict[int, List[str]]:
    """The complete per-packet-type traceable event universe, toolchain-derived.
    All events minus NONE, ordered by event id. The engine's completeness claim."""
    global _COMPLETE_CACHE
    if _COMPLETE_CACHE is not None:
        return _COMPLETE_CACHE
    from trace_capture import load_event_ids, PKT_TO_TILE_TYPE  # lazy: header on demand
    menu: Dict[int, List[str]] = {}
    for pkt, tile_type in PKT_TO_TILE_TYPE.items():
        ids = load_event_ids(tile_type)
        menu[pkt] = sorted((n for n in ids if n != "NONE"), key=lambda n: ids[n])
    _COMPLETE_CACHE = menu
    return menu


def swept_menu() -> Dict[int, List[str]]:
    """The flood-safe default sweep menu: per packet type, measurement-tier names
    (ordered by id) then infra-tier names (ordered by id); the stateful tier is
    excluded (reachable via complete_menu, kept out of co-traced batches). This is
    what enumerate_configured_events draws from."""
    global _SWEPT_CACHE
    if _SWEPT_CACHE is not None:
        return _SWEPT_CACHE
    out: Dict[int, List[str]] = {}
    for pkt, names in complete_menu().items():
        meas = [n for n in names if _event_tier(n) == "measurement"]
        infra = [n for n in names if _event_tier(n) == "infra"]
        out[pkt] = meas + infra   # within each tier already id-ordered from complete_menu
    _SWEPT_CACHE = out
    return out
```

Then change `enumerate_configured_events` (the `for name in _MENU.get(pkt, []):`
line, ~59) to `for name in swept_menu().get(pkt, []):`. Ensure the typing imports
cover `Dict, List` (already imported). Grep for any other `_MENU` reference and
update: `grep -n "_MENU" tools/inference/selfmodel.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools && python -m pytest test_selfmodel.py -v`
Expected: PASS (the new tests and all pre-existing selfmodel tests). If
`test_swept_menu_fills_the_reviewer_gaps` fails, print `load_event_ids("memtile")`
and `load_event_ids("memmod")` keys to confirm the exact task-event names and adjust
the *substring* assertion -- do NOT hand-add names to the menu.

- [ ] **Step 5: Run the broader engine test suite for regressions**

Run: `cd tools && python -m pytest test_selfmodel.py test_trace_capture.py test_hw_instrument.py test_timeline.py test_inference_ledger.py test_config_extract_generator.py test_config_extract_reachability.py -v`
Expected: PASS. The swept menu is larger, so `enumerate_configured_events` returns
more events and `build_active_plan` produces more batches; any test that asserted
an exact enumerated-event *count* (not membership) must be updated to assert
membership/superset, which is the correct invariant. (The ledger/generator/
reachability tests are in scope because `generate_ledger` consumes the enumerated
set; `test_timeline.py` because the gate primitives read its anchored offsets.)

- [ ] **Step 6: HW seed-smoke -- empirically validate the tiering (the cheap oracle)**

Once the unit suite is green, run ONE seed sweep of the swept menu on `add_one`
on real NPU1 (from a clean shell, `env -u XDNA_EMU`), using the engine's
`HwInstrument`/`capture` path over `enumerate_configured_events` -> `build_active_plan`.
This is the empirical arbiter, not a guess:
- **Expected:** the full swept sweep completes with NO `CaptureError` truncation,
  and the decoded events include at least one memtile `DMA_*_TASK` and one memmod
  `*_FINISHED_TASK` that the old hand-list could never reach (the gap this task closes).
- **If a batch truncates:** the offending event(s) are real flooders the prior
  missed -- add them to `_STATEFUL_EXACT`/the stateful predicate (re-deriving from
  the truncation, toolchain-grounded), re-run, and record the promotion in the
  commit message. Convergence is fast (the prior is conservative; one or two
  iterations at most).

This step is HW-gated; the menu code lands green on the unit suite first (Step 4),
so Task 1's deliverable is not blocked on hardware availability -- the smoke is the
empirical confirmation, run when the NPU is free.

- [ ] **Step 7: Commit**

```bash
git add tools/inference/selfmodel.py tools/test_selfmodel.py
git commit -m "feat(#140): complete + flood-safe event menu derived from toolchain (engine completeness)"
```

---

## Task 2: The on-chip spike kernel (enriched: 2-hop, cross-column, rank-2)

**Files:**
- Create: `mlir-aie/test/npu-xrt/onchip_spike/aie.mlir` (raw MLIR, from scratch)
- Create: harness glue mirroring a sibling 2-tile npu-xrt kernel
  (`CMakeLists.txt` / `run.lit` / host as the discover step expects)

**Interfaces:**
- Produces: a compiled xclbin + insts the bridge-test harness can run on NPU1,
  emitting a labeled trace with the spike's events firing in-window.

**This is from-scratch MLIR -- budget real bring-up.** No existing template
combines core self-generation + the task API + core->core on-chip flow.
`core_dmas/dma_configure_task_lock/aie.mlir` proves host-issued *terminating* DMA
tasks on a compute tile but is DDR-fed with no core body; the only self-gen
example uses circular BDs. So the producer core body, the lock interlock, and the
core->core routing are genuinely new and will need iteration against the cited
templates. The *requirements* (below) are concrete and checkable; the smoke step
(Step 4) is the gate.

**Enriched topology (D2 -- 2-hop + cross-column so a PASS de-risks the full
kernel's diagonal / cross-column / multi-tile-contention risks, not merely
single-hop determinism):**

```
            col 0          col 1
  row 3   core(0,3) ---->  core(1,3)   relay (vertical in) then horizontal out; consumer
  row 2   core(0,2)        --          producer: core body fills local buf on-chip
  row 1   --               --
  row 0   shim(0,0)        --          BROADCAST_15 sync + trace drain ONLY
```

Data path (all on-chip, task-API DMAs, terminating chains):
- `core(0,2)` core body writes a fixed pattern into local memory behind a lock
  (e.g. `for i: buf[i] = i`) -- **no input DMA, no DDR**.
- `core(0,2)` memmod **MM2S** task streams `buf` to `core(0,3)`.  *(hop 1:
  vertical, Delta_n (0,1))*
- `core(0,3)` memmod **S2MM** task receives behind an interlocking lock; its
  memmod **MM2S** task re-streams to `core(1,3)`. (Pure DMA relay -- no core body
  needed on (0,3).)  *(hop 2: horizontal, Delta_n (1,0))*
- `core(1,3)` memmod **S2MM** task receives behind a lock; trivial/no core body.

**Couplings the spike measures (rank-2, the structure the full kernel needs):**
- *Vertical (1 hop):* `core(0,2)` memmod `DMA_MM2S_0_FINISHED_TASK` ->
  `core(0,3)` memmod `DMA_S2MM_0_START_TASK`. Delta_n (0,1).
- *Diagonal (2 hop, endpoints):* `core(0,2)` memmod `DMA_MM2S_0_FINISHED_TASK` ->
  `core(1,3)` memmod `DMA_S2MM_0_START_TASK`. Delta_n (1,1), cross-column.
- Together: linearly-independent Delta_n {(0,1),(1,1)} -> **rank-2**, hop counts
  {1,2}. This rehearses Option A's coupling set core->core (the full kernel uses
  memtile(0,1) as the shared parent; the **memtile-in-path timing is the one
  residual the spike does not cover** -- flagged deliberately, deferred to the
  full-kernel plan, since memtile relay timing is already partly characterized
  from add_one and adding a memtile relay roughly doubles the spike's MLIR).
- *Within-domain Q=0 candidates (measure SEVERAL, report which reach range 0 --
  do NOT bet Q=0 on the most backpressure-coupled `MM2S_START->FINISHED` pair):*
  - `core(0,3)` memmod `DMA_S2MM_0_FINISHED_TASK -> DMA_MM2S_0_START_TASK`
    (relay receive->resend, lower coupling -- spec Sec.9's choice).
  - `core(0,2)` memmod `DMA_MM2S_0_START_TASK -> DMA_MM2S_0_FINISHED_TASK`
    (high-coupling, included as a diagnostic contrast).
  - `core(1,3)` memmod `DMA_S2MM_0_START_TASK -> DMA_S2MM_0_FINISHED_TASK`.

**Anchor:** a performance counter on `core(0,2)` emitting a stable per-run anchor
(`PERF_CNT_2`, default engine anchor key `1|2|0|PERF_CNT_2`). **It is not free --
configure it explicitly** (do not rely on a template): `Performance_Control1`
Cnt2_Start = `ACTIVE`, `Performance_Control2` = `0x00070000` (self-reset on
counter event), `Counter2_Event_Value` = period; cf.
`tools/mlir-trace-inject.py:506-587` and `tools/perfcnt_defaults.py`. Size the
producer core body / period so the counter actually FIRES (it fires every
~`period` *active* cycles, not once per run). `core(0,2)`'s producer loop supplies
the active cycles -- so the **no-core / pure-DMA fallback is dropped** (it would
kill the anchor: no active cycles -> no `PERF_CNT_2`).

**Trace config (template: `mlir-aie/test/npu-xrt/vec_mul_event_trace/aie.mlir`):**
per-module `aie.trace` blocks with distinct packet ids; trace the events above;
`aie.trace.start broadcast=15` / `stop broadcast=14` on **every traced module**:
`core(0,2)` core (anchor), `core(0,2)` memmod, `core(0,3)` memmod, `core(1,3)`
memmod. Slot budget per module is well within 8 (anchor 1; (0,2) memmod 2; (0,3)
memmod 3; (1,3) memmod 2). The shim generates the sync.

- [ ] **Step 1: Author `aie.mlir`** (device `npu1` / `npu1_2col` -- needs 2
  columns; confirm against a sibling 2-col kernel). Tiles shim(0,0), core(0,2),
  core(0,3), core(1,3). On-chip producer core body + lock; task-API MM2S/S2MM
  relay chain; per-module trace blocks; explicit `PERF_CNT_2` config; uniform
  `broadcast=15`.

- [ ] **Step 2: Add the harness glue** so the bridge-test discover step finds it
  (copy the structure of a sibling 2-col npu-xrt kernel's `CMakeLists.txt`/`run.lit`).
  **Host:** every sibling `test.cpp` memcmps a DDR output the on-chip kernel won't
  produce -> copying one yields a false FAIL. Either write a minimal host that
  validates *trace presence only* (no data check), OR skip the `emu-bridge-test.sh`
  PASS gate for the smoke and decode via `trace_capture.capture` directly (the
  gate path needs no DDR I/O -- confirmed). Pick whichever is less glue; the trace,
  not a data memcmp, is the deliverable.

- [ ] **Step 3: Compile (Chess is ground truth; Peano informational)**

Run: `./scripts/emu-bridge-test.sh --compile -v onchip_spike`
Expected: a Chess build under `mlir-aie/build/test/npu-xrt/onchip_spike/chess/`
with an xclbin + insts. Fix MLIR errors until it compiles.

- [ ] **Step 4: Single HW smoke run, confirm the target events fire in-window**

Run (real HW; clean shell): the harness's single-kernel HW invocation under
`env -u XDNA_EMU` (or `trace_capture.capture` directly with a hand-built one-batch
plan). Decode one trace and confirm each target event appears **exactly once per
run, none looping**: `core(0,2)` MM2S START+FINISHED, `core(0,3)` S2MM START+
FINISHED + MM2S START, `core(1,3)` S2MM START+FINISHED, `core(0,2)` PERF_CNT_2.
Expected: all present, single-shot. A missing `*_TASK` event means the BD chain is
circular or the task didn't run -- fix before Task 3. A missing `PERF_CNT_2` means
the counter never fired -- enlarge the body or lower the period.

- [ ] **Step 5: Commit**

```bash
git add mlir-aie/test/npu-xrt/onchip_spike/
git commit -m "feat(#140): on-chip 2-hop cross-column spike kernel (rank-2, no-DDR, task-API) for SP-3 Q=0 de-risk"
```

---

## Task 3: The Q=0 / non-None gate, and the decision

**Files:**
- Create: `tools/inference/spike_gate.py`
- Test: `tools/test_spike_gate.py`

**Interfaces:**
- Consumes: `inference.verifier.offset_exact(run_dirs, a, b, anchor_key) -> Optional[int]`
  (returns `a - b`, non-None iff cross-run range <= Q == 0),
  `inference.grounding.ground_edge(run_dirs, child, parent, anchor_key) -> Gap`
  (`Gap.reproduction_offset` non-None iff the raw cross-domain offset agrees across
  runs), and `inference.grounding.same_domain(a, b) -> bool` (the `col|row|pkt`
  prefix test) for defensive asserts.
- Produces: `spike_gate.evaluate(run_dirs, *, anchor_key, within_pairs, cross_pairs)
  -> dict` with per-pair results and an overall `pass`, plus a `__main__` CLI.

**Background:** the gate uses the engine's own primitives so "Q=0" and "non-None
reproduction" mean exactly what the engine means. A pair is `(child_key,
parent_key)` with keys `col|row|pkt|NAME`. `within_pairs` are several same-domain
candidates (Task 2 measures multiple; the gate reports which reach range 0).
`cross_pairs` are the cross-domain vertical + diagonal couplings.

**Gate-hardening (from the adversarial pass):**
- Assert `same_domain(*pair)` for every within pair and `not same_domain(*pair)`
  for every cross pair -- a mis-specified key (wrong row/pkt) is a test bug, not a
  silent pass.
- PASS = **at least one** within pair reaches range 0 **and every** cross pair
  reproduces (non-None). Reporting all within candidates is diagnostic; requiring
  *all* of them to be range-0 would over-constrain on the deliberately-included
  high-coupling contrast pair.
- Do **not** read a passing cross leg as "causal coupling validated." A cross pair
  reproduces because *both endpoints are deterministic vs the anchor* (which is
  exactly what SP-4b's `raw = Delta_wall + skew` decomposition needs), not because
  a tight producer->consumer link was proven. Keep the verdict claim precise.

- [ ] **Step 1: Write the failing test** (synthetic run dirs, no HW)

Create `tools/test_spike_gate.py`. Reuse `test_inference_grounding.py`'s `_runs(tmp_path, rows)`
helper (import it, or copy its construction) -- it builds fake co-traced run dirs
from dicts of `event_key -> soc`.

```python
from inference.spike_gate import evaluate

ANCHOR  = "1|2|0|PERF_CNT_2"
# within: same domain (1|3|1) on the relay tile (low-coupling receive->resend)
W_PAR   = "1|3|1|DMA_S2MM_0_FINISHED_TASK"
W_CHILD = "1|3|1|DMA_MM2S_0_START_TASK"
# cross vertical: 1|2|1 -> 1|3|1 ; cross diagonal: 1|2|1 -> 2|3|1
XV_PAR   = "1|2|1|DMA_MM2S_0_FINISHED_TASK"
XV_CHILD = "1|3|1|DMA_S2MM_0_START_TASK"
XD_PAR   = "1|2|1|DMA_MM2S_0_FINISHED_TASK"
XD_CHILD = "2|3|1|DMA_S2MM_0_START_TASK"


def test_gate_passes_when_a_within_reaches_range0_and_all_cross_reproduce(tmp_path):
    runs = _runs(tmp_path, [
        {ANCHOR: 0, W_PAR: 10, W_CHILD: 30, XV_PAR: 12, XV_CHILD: 17, XD_PAR: 12, XD_CHILD: 25},
        {ANCHOR: 0, W_PAR: 50, W_CHILD: 70, XV_PAR: 52, XV_CHILD: 57, XD_PAR: 52, XD_CHILD: 65},
    ])  # within offset 20, vertical 5, diagonal 13 -- all stable
    res = evaluate(runs, anchor_key=ANCHOR,
                   within_pairs=[(W_CHILD, W_PAR)],
                   cross_pairs=[(XV_CHILD, XV_PAR), (XD_CHILD, XD_PAR)])
    assert res["pass"] is True
    assert res["any_within_range0"] is True
    assert res["cross"][0]["reproduction"] == 5
    assert res["cross"][1]["reproduction"] == 13


def test_gate_fails_when_within_jitters_and_no_other_within(tmp_path):
    runs = _runs(tmp_path, [
        {ANCHOR: 0, W_PAR: 10, W_CHILD: 30, XV_PAR: 12, XV_CHILD: 17, XD_PAR: 12, XD_CHILD: 25},
        {ANCHOR: 0, W_PAR: 50, W_CHILD: 71, XV_PAR: 52, XV_CHILD: 57, XD_PAR: 52, XD_CHILD: 65},
    ])  # within offset 20 vs 21 -> range 1
    res = evaluate(runs, anchor_key=ANCHOR,
                   within_pairs=[(W_CHILD, W_PAR)],
                   cross_pairs=[(XV_CHILD, XV_PAR), (XD_CHILD, XD_PAR)])
    assert res["any_within_range0"] is False
    assert res["pass"] is False


def test_gate_rejects_misspecified_domains(tmp_path):
    import pytest
    runs = _runs(tmp_path, [{ANCHOR: 0, XV_PAR: 1, XV_CHILD: 2}])
    with pytest.raises(AssertionError):
        evaluate(runs, anchor_key=ANCHOR,
                 within_pairs=[(XV_CHILD, XV_PAR)],          # cross keys as a "within" pair
                 cross_pairs=[(XV_CHILD, XV_PAR)])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd tools && python -m pytest test_spike_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference.spike_gate'`.

- [ ] **Step 3: Implement `spike_gate.evaluate`**

```python
"""SP-3 on-chip spike gate: does an entirely-on-chip, 2-hop, cross-column kernel
give Q=0 within-domain and non-None cross-domain reproduction on real NPU1?"""
from typing import List, Optional, Tuple
from inference.verifier import offset_exact
from inference.grounding import ground_edge, same_domain, Gap


def _cross_reproduction(run_dirs, child, parent, anchor_key) -> Optional[int]:
    g = ground_edge(run_dirs, child, parent, anchor_key)
    return g.reproduction_offset if isinstance(g, Gap) else None


def evaluate(run_dirs: List[str], *, anchor_key: str,
             within_pairs: List[Tuple[str, str]],
             cross_pairs: List[Tuple[str, str]]) -> dict:
    for child, parent in within_pairs:
        assert same_domain(child, parent), f"within pair not same-domain: {child} {parent}"
    for child, parent in cross_pairs:
        assert not same_domain(child, parent), f"cross pair not cross-domain: {child} {parent}"

    within = [{"pair": (c, p),
               "offset": offset_exact(run_dirs, c, p, anchor_key)}  # None unless range 0
              for c, p in within_pairs]
    cross = [{"pair": (c, p),
              "reproduction": _cross_reproduction(run_dirs, c, p, anchor_key)}
             for c, p in cross_pairs]

    any_within = any(w["offset"] is not None for w in within)
    all_cross = bool(cross) and all(x["reproduction"] is not None for x in cross)
    return {"within": within, "cross": cross,
            "any_within_range0": any_within, "all_cross_reproduce": all_cross,
            "pass": any_within and all_cross}
```

Add a `__main__` taking `--runs <dir>...`, `--anchor`, repeatable
`--within child parent` and `--cross child parent`, calling `evaluate`, printing
the dict, exiting nonzero on fail. (Confirm `Gap`/`reproduction_offset` against
`grounding.py:75-107` and `same_domain` against `grounding.py:47`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd tools && python -m pytest test_spike_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Capture ~20 HW runs of the spike and run the gate**

Capture the target events over ~20 runs on NPU1. The events fit a single <=8-slot
batch per module, so no multi-batch split is needed -- use the engine's
`HwInstrument`/`capture` or a direct `trace_capture.capture` with a hand-built
one-batch plan. Then (event names/keys must match Task 2's smoke decode -- adjust
absolute columns if the harness places the kernel elsewhere):

```bash
cd tools && python -m inference.spike_gate --runs <20 run dirs> \
  --anchor '1|2|0|PERF_CNT_2' \
  --within '1|3|1|DMA_MM2S_0_START_TASK'  '1|3|1|DMA_S2MM_0_FINISHED_TASK' \
  --within '1|2|1|DMA_MM2S_0_FINISHED_TASK' '1|2|1|DMA_MM2S_0_START_TASK' \
  --within '2|3|1|DMA_S2MM_0_FINISHED_TASK' '2|3|1|DMA_S2MM_0_START_TASK' \
  --cross  '1|3|1|DMA_S2MM_0_START_TASK'  '1|2|1|DMA_MM2S_0_FINISHED_TASK' \
  --cross  '2|3|1|DMA_S2MM_0_START_TASK'  '1|2|1|DMA_MM2S_0_FINISHED_TASK'
```

- [ ] **Step 6: Record the decision in the progress ledger**

This is the plan's terminal gate, not a code step:
- **PASS** (>=1 within pair range 0 AND both cross pairs reproduce): the on-chip
  approach delivers Q=0 across the rank-2 structure. Proceed to the full SP-3
  kernel (separate spec/plan), confident in the on-chip + diagonal + cross-column
  topology.
- **FAIL:** the cheap HW oracle has told us the on-chip premise is insufficient
  *before* the large MLIR investment. Stop and reconsider -- do not proceed to the
  full kernel. Capture which leg failed (within jitter vs which cross pair failed
  to reproduce -- vertical vs diagonal distinguishes hop-count from cross-column as
  the culprit) and the observed ranges for the redesign.

Record the verdict, the per-within-pair ranges, and the two cross reproduction
values in `.superpowers/sdd/progress.md`.

---

## Self-Review

**Decisions folded in (adversarial pass 2026-06-29).** D1: completeness is
include-and-tier (`complete_menu` universe minus only `NONE`; `swept_menu` excludes
the flood-risk stateful tier; HW seed-smoke is the empirical flood arbiter). D2:
the spike is enriched to 2-hop + cross-column so a PASS de-risks the rank-2
diagonal / cross-column / multi-tile structure, not just single-hop determinism.
Reviewer fixes folded in: explicit `PERF_CNT_2` config (no-core fallback dropped),
from-scratch MLIR budgeted with a concrete producer/relay/consumer spec,
trace-validating host (no DDR memcmp), multi-candidate within-domain pairs,
`same_domain` gate asserts, honest ~10x seed cost + the `n_runs=1` prune follow-up,
widened regression run, corrected dead-code/circular-import comments.

**Spec coverage.** Implements the two pieces Maya scoped: engine completeness
(Task 1) and the enriched on-chip Q=0 spike (Tasks 2-3). The full SP-3 kernel, the
SP-4a per-module reset invariant in production form, full-event-set batch packing,
the `cross_track_edges` acceptance assertion, and the seed-prune optimization are
deliberately deferred to follow-on plans -- in scope only if the spike passes.

**Residual flagged for the full-kernel plan:** the spike routes core->core and so
does NOT exercise **memtile-in-path** timing (Option A uses memtile(0,1) as the
shared parent). Deliberate: memtile relay timing is partly characterized from
add_one, and a memtile relay roughly doubles the spike MLIR. The full kernel must
still validate it.

**Type consistency.** `complete_menu()/swept_menu() -> Dict[int, List[str]]`
consumed by `enumerate_configured_events`; `_event_tier(str) -> str`;
`evaluate(...) -> dict` with the keys the CLI and tests read; pair tuples are
`(child, parent)` consistently (matching `offset_exact`/`ground_edge` arg order).
Event keys are `col|row|pkt|NAME` throughout.
