"""Tests for trace-sweep.py pure helpers.

Covers the grounding-event batch construction and anchor/merge logic
introduced in v2. Pure-function tests only; full sweep integration is
exercised by scripts/trace-sweep-all.sh on real artifacts.
"""

import importlib.util
import json
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
_build_lockstep_batches = _mod._build_lockstep_batches
_check_grounding_pc_invariance = _mod._check_grounding_pc_invariance


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


# ---------------------------------------------------------------------------
# _build_lockstep_batches
# ---------------------------------------------------------------------------

def test_lockstep_batches_one_per_tile_per_batch():
    """Verify _build_lockstep_batches generates per-batch event assignments
    across all tile cursors, with cursor exhaustion handled gracefully."""
    cursors = {
        "core_0_2":   {"sweep": ["INSTR_VECTOR", "MEMORY_STALL", "STREAM_STALL"], "remaining_slots": 5},
        "core_0_3":   {"sweep": ["INSTR_VECTOR", "MEMORY_STALL"],                  "remaining_slots": 5},
        "memmod_0_2": {"sweep": ["DMA_S2MM_0_FINISHED_BD"],                        "remaining_slots": 7},
    }
    batches = _build_lockstep_batches(cursors)
    # max(ceil(3/5), ceil(2/5), ceil(1/7)) = max(1,1,1) = 1 batch
    assert len(batches) == 1
    b0 = batches[0]
    assert set(b0["core_0_2"]) == {"INSTR_VECTOR", "MEMORY_STALL", "STREAM_STALL"}
    assert set(b0["core_0_3"]) == {"INSTR_VECTOR", "MEMORY_STALL"}
    assert set(b0["memmod_0_2"]) == {"DMA_S2MM_0_FINISHED_BD"}


def test_lockstep_batches_partitions_long_sweep():
    """A sweep longer than remaining_slots produces multiple batches."""
    cursors = {
        "core_0_2": {"sweep": list(range(12)), "remaining_slots": 5},
    }
    batches = _build_lockstep_batches(cursors)
    assert len(batches) == 3  # ceil(12/5)
    assert sum(len(b["core_0_2"]) for b in batches) == 12


def test_lockstep_batches_handles_cursor_exhaustion():
    """When one cursor exhausts before others, later batches emit
    grounding-only ([]) for that cursor."""
    cursors = {
        "core_0_2": {"sweep": ["A", "B"],         "remaining_slots": 1},
        "core_0_3": {"sweep": ["A", "B", "C", "D"], "remaining_slots": 1},
    }
    batches = _build_lockstep_batches(cursors)
    assert len(batches) == 4
    assert batches[2]["core_0_2"] == []  # exhausted
    assert batches[3]["core_0_2"] == []  # still exhausted
    assert batches[3]["core_0_3"] == ["D"]


def test_lockstep_batches_empty_cursors():
    """Empty cursor dict returns empty batch list."""
    assert _build_lockstep_batches({}) == []


def test_lockstep_batches_empty_sweep_produces_one_batch():
    """A cursor with an empty sweep still produces one batch (grounding-only)."""
    cursors = {
        "core_0_2": {"sweep": [], "remaining_slots": 5},
    }
    batches = _build_lockstep_batches(cursors)
    assert len(batches) == 1
    assert batches[0]["core_0_2"] == []


# ---------------------------------------------------------------------------
# _check_grounding_pc_invariance
# ---------------------------------------------------------------------------

def test_check_grounding_pc_invariance_passes_when_pcs_consistent(tmp_path):
    """All batches show the same INSTR_EVENT_0 PC -> unsafe_for_pc_join=False."""
    for bi in range(3):
        bd = tmp_path / f"batch_{bi:02d}" / "hw"
        bd.mkdir(parents=True)
        (bd / "trace.events.json").write_text(json.dumps({
            "events": [
                {"name": "INSTR_EVENT_0", "ts": 100, "col": 0, "row": 2, "slot": 0, "pkt_type": 0},
            ],
        }))
    result = _check_grounding_pc_invariance(tmp_path, ["INSTR_EVENT_0"])
    assert result["unsafe_for_pc_join"] is False
    assert result["reason"] is None


def test_check_grounding_pc_invariance_flags_drift(tmp_path):
    """Different batches show different INSTR_EVENT_0 PCs -> unsafe_for_pc_join=True."""
    for bi, pc in enumerate([100, 100, 200]):
        bd = tmp_path / f"batch_{bi:02d}" / "hw"
        bd.mkdir(parents=True)
        (bd / "trace.events.json").write_text(json.dumps({
            "events": [
                {"name": "INSTR_EVENT_0", "ts": pc, "col": 0, "row": 2, "slot": 0, "pkt_type": 0},
            ],
        }))
    result = _check_grounding_pc_invariance(tmp_path, ["INSTR_EVENT_0"])
    assert result["unsafe_for_pc_join"] is True
    assert "INSTR_EVENT_0" in result["reason"]


def test_check_grounding_pc_invariance_no_batches(tmp_path):
    """An empty batches dir is safe (nothing to drift)."""
    result = _check_grounding_pc_invariance(tmp_path, ["INSTR_EVENT_0"])
    assert result["unsafe_for_pc_join"] is False
    assert result["reason"] is None


def test_check_grounding_pc_invariance_missing_events_file(tmp_path):
    """Batches with no trace.events.json are skipped without error."""
    bd = tmp_path / "batch_00" / "hw"
    bd.mkdir(parents=True)
    # no trace.events.json written
    result = _check_grounding_pc_invariance(tmp_path, ["INSTR_EVENT_0"])
    assert result["unsafe_for_pc_join"] is False


def test_check_grounding_pc_invariance_multiple_grounding_events(tmp_path):
    """Invariance check is per-event: one drifting event flags the whole result."""
    for bi in range(2):
        bd = tmp_path / f"batch_{bi:02d}" / "hw"
        bd.mkdir(parents=True)
        pc0 = 100          # INSTR_EVENT_0 stable
        pc1 = 200 + bi * 50  # INSTR_EVENT_1 drifts on batch 1
        (bd / "trace.events.json").write_text(json.dumps({
            "events": [
                {"name": "INSTR_EVENT_0", "ts": pc0, "col": 0, "row": 2, "slot": 0, "pkt_type": 0},
                {"name": "INSTR_EVENT_1", "ts": pc1, "col": 0, "row": 2, "slot": 1, "pkt_type": 0},
            ],
        }))
    result = _check_grounding_pc_invariance(tmp_path, ["INSTR_EVENT_0", "INSTR_EVENT_1"])
    assert result["unsafe_for_pc_join"] is True
    assert "INSTR_EVENT_1" in result["reason"]
