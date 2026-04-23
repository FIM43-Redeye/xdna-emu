"""Tests for trace-sweep.py pure helpers.

Covers the grounding-event batch construction and anchor/merge logic
introduced in v2. Pure-function tests only; full sweep integration is
exercised by scripts/trace-sweep-all.sh on real artifacts.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "trace_sweep", Path(__file__).parent / "trace-sweep.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["trace_sweep"] = _mod
_spec.loader.exec_module(_mod)

EventDef = _mod.EventDef
_build_batches = _mod._build_batches
_anchor_events = _mod._anchor_events
_merge_anchored = _mod._merge_anchored


def _mk_event(id_: int, name: str) -> EventDef:
    return EventDef(name=name, id=id_)


# ---------------------------------------------------------------------------
# _build_batches
# ---------------------------------------------------------------------------

def test_build_batches_no_grounding_splits_by_slots():
    evs = [_mk_event(i, f"E{i}") for i in range(18)]
    batches = _build_batches(evs, slots=8)
    assert len(batches) == 3
    assert [e.id for e in batches[0]] == list(range(8))
    assert [e.id for e in batches[1]] == list(range(8, 16))
    assert [e.id for e in batches[2]] == [16, 17]


def test_build_batches_with_grounding_reserves_slot_zero():
    ground = _mk_event(125, "USER_EVENT_1")
    evs = [_mk_event(i, f"E{i}") for i in range(14)]
    batches = _build_batches(evs, slots=8, ground_event=ground)
    # 14 sweep events / 7 per batch = 2 batches
    assert len(batches) == 2
    for b in batches:
        assert b[0] is ground  # slot 0 is always grounding
        assert len(b) <= 8
    # All 14 non-ground events covered across batches, no dupes
    swept_ids = [e.id for b in batches for e in b[1:]]
    assert sorted(swept_ids) == list(range(14))


def test_build_batches_drops_grounding_from_sweep_list():
    ground = _mk_event(125, "USER_EVENT_1")
    evs = [_mk_event(125, "USER_EVENT_1"), _mk_event(5, "INSTR_VECTOR")]
    batches = _build_batches(evs, slots=8, ground_event=ground)
    # Grounding event should not appear in sweep slots even if caller included it
    sweep_slots = [e for b in batches for e in b[1:]]
    assert all(e.id != 125 for e in sweep_slots)
    # Single batch with grounding + the one real event
    assert len(batches) == 1
    assert [e.id for e in batches[0]] == [125, 5]


def test_build_batches_with_grounding_but_empty_sweep_still_runs_once():
    ground = _mk_event(125, "USER_EVENT_1")
    batches = _build_batches([], slots=8, ground_event=ground)
    assert len(batches) == 1
    assert [e.id for e in batches[0]] == [125]


# ---------------------------------------------------------------------------
# _anchor_events
# ---------------------------------------------------------------------------

def test_anchor_events_subtracts_ground_ts():
    events = [
        {"slot": 0, "name": "USER_EVENT_1", "ts": 100, "row": 2},
        {"slot": 1, "name": "INSTR_VECTOR", "ts": 120, "row": 2},
        {"slot": 2, "name": "LOCK_STALL",   "ts": 250, "row": 2},
    ]
    anchor, out = _anchor_events(events, ground_slot=0)
    assert anchor == 100
    # Grounding event itself dropped from output
    assert [e["name"] for e in out] == ["INSTR_VECTOR", "LOCK_STALL"]
    assert [e["ts_anchored"] for e in out] == [20, 150]
    # Raw ts preserved
    assert [e["ts"] for e in out] == [120, 250]


def test_anchor_events_uses_first_ground_firing():
    events = [
        {"slot": 1, "name": "INSTR_VECTOR", "ts": 50},
        {"slot": 0, "name": "USER_EVENT_1", "ts": 100},
        {"slot": 0, "name": "USER_EVENT_1", "ts": 500},  # re-firing; ignored
        {"slot": 2, "name": "LOCK_STALL",   "ts": 200},
    ]
    anchor, out = _anchor_events(events, ground_slot=0)
    assert anchor == 100
    # INSTR_VECTOR fired before the anchor -- anchored ts goes negative,
    # which is informational. We keep it rather than filtering so users
    # can see the ordering.
    assert [e["ts_anchored"] for e in out] == [-50, 100]


def test_anchor_events_no_ground_fire_returns_none_anchor():
    events = [
        {"slot": 1, "name": "INSTR_VECTOR", "ts": 50},
        {"slot": 2, "name": "LOCK_STALL",   "ts": 200},
    ]
    anchor, out = _anchor_events(events, ground_slot=0)
    assert anchor is None
    assert [e["name"] for e in out] == ["INSTR_VECTOR", "LOCK_STALL"]
    assert all(e["ts_anchored"] is None for e in out)


def test_anchor_events_empty_input():
    anchor, out = _anchor_events([], ground_slot=0)
    assert anchor is None
    assert out == []


# ---------------------------------------------------------------------------
# _merge_anchored
# ---------------------------------------------------------------------------

def test_merge_anchored_sorts_by_anchor_and_tags_batch():
    per_batch = [
        (0, [{"name": "A", "ts_anchored": 50},
             {"name": "B", "ts_anchored": 200}]),
        (1, [{"name": "C", "ts_anchored": 100},
             {"name": "D", "ts_anchored": 20}]),
    ]
    merged = _merge_anchored(per_batch)
    assert [e["name"] for e in merged] == ["D", "A", "C", "B"]
    assert [e["source_batch"] for e in merged] == [1, 0, 1, 0]


def test_merge_anchored_sinks_unanchored_events_to_end():
    # Events with anchor=None (grounding never fired in their batch) still
    # appear in the merged stream, but after all anchored events so the
    # readable prefix is the deterministic part.
    per_batch = [
        (0, [{"name": "A", "ts_anchored": 50}]),
        (1, [{"name": "B", "ts_anchored": None}]),
        (2, [{"name": "C", "ts_anchored": 10}]),
    ]
    merged = _merge_anchored(per_batch)
    assert [e["name"] for e in merged] == ["C", "A", "B"]
