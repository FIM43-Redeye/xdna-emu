"""Tests for trace-sweep.py pure helpers.

Covers the grounding-event batch construction and anchor/merge logic
introduced in v2. Pure-function tests only; full sweep integration is
exercised by scripts/trace-sweep-all.sh on real artifacts.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# trace_mode threading: ParseSession.parse_one builds the right request,
# _run_one_side forwards trace_mode through to it.
# ---------------------------------------------------------------------------

def _fake_parse_session(canned_response: dict | None = None):
    """Build a fake ParseSession-like object.

    Only what ParseSession.parse_one writes to / reads from is mocked:
    a stdin-like object with .write/.flush, a stdout-like that yields
    a canned line, and a .proc.poll() returning None. Captures every
    request line so tests can assert on the JSON shape.
    """
    sess = _mod.ParseSession.__new__(_mod.ParseSession)
    sess.side = "TEST"
    sent: list[str] = []

    class _StdinLike:
        def write(self, s):
            sent.append(s)
        def flush(self):
            pass

    response_line = json.dumps(canned_response or {
        "ok": True, "events_count": 0, "cycles": 0, "empty": True,
    }) + "\n"

    class _StdoutLike:
        def __init__(self):
            self._lines = [response_line]
        def readline(self):
            return self._lines.pop(0) if self._lines else ""

    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = _StdinLike()
    proc.stdout = _StdoutLike()
    sess.proc = proc
    sess.stderr_log = Path("/dev/null")
    sess._stderr_fh = None
    return sess, sent


def test_parse_session_request_includes_trace_mode_event_pc(tmp_path):
    sess, sent = _fake_parse_session()
    sess.parse_one(
        trace_bin=tmp_path / "trace.bin",
        xclbin_mlir=tmp_path / "x.mlir",
        trace_mode="event_pc",
    )
    assert len(sent) == 1
    req = json.loads(sent[0])
    assert req["trace_mode"] == "event_pc"


def test_parse_session_request_includes_trace_mode_inst_exec(tmp_path):
    """The mode-2 finishing batch's load-bearing case: server must see
    trace_mode='inst_exec' so it routes through the right decoder."""
    sess, sent = _fake_parse_session()
    sess.parse_one(
        trace_bin=tmp_path / "trace.bin",
        xclbin_mlir=tmp_path / "x.mlir",
        trace_mode="inst_exec",
    )
    req = json.loads(sent[0])
    assert req["trace_mode"] == "inst_exec"


def test_parse_session_default_trace_mode_event_time(tmp_path):
    """Legacy callers (sweep_multi) get event_time when they don't pass
    trace_mode -- preserves backwards-compatible behavior."""
    sess, sent = _fake_parse_session()
    sess.parse_one(
        trace_bin=tmp_path / "trace.bin",
        xclbin_mlir=tmp_path / "x.mlir",
    )
    req = json.loads(sent[0])
    assert req["trace_mode"] == "event_time"


# ---------------------------------------------------------------------------
# sweep_lockstep: mode-2 batch decodes with trace_mode="inst_exec";
# manifest written even on patch-subprocess failure.
# ---------------------------------------------------------------------------

def _make_min_lockstep_env(tmp_path: Path):
    """Set up the minimum filesystem state sweep_lockstep expects so the
    setup phase passes and we reach the patch-and-run loop. Returns the
    patches we need to install on the trace_sweep module."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "aie.xclbin").write_bytes(b"\x00" * 64)
    (build_dir / "insts.bin").write_bytes(b"\x00" * 64)
    prj = build_dir / "x.mlir.prj"
    prj.mkdir()
    mlir = prj / "input_with_addresses.mlir"
    mlir.write_text("// fake mlir\n")
    out_dir = tmp_path / "out"
    return build_dir, out_dir, mlir


def test_sweep_lockstep_mode2_calls_parse_with_inst_exec(tmp_path, monkeypatch):
    """The mode-2 finishing batch must dispatch a parse with
    trace_mode='inst_exec'. We mock everything below sweep_lockstep so
    the test runs purely in-process and verifies the call site arguments."""
    TileSpec = _mod.TileSpec
    EventDef = _mod.EventDef
    build_dir, out_dir, _ = _make_min_lockstep_env(tmp_path)

    # Stub out load_events to return a deterministic small list.
    monkeypatch.setattr(_mod, "load_events", lambda tt: [
        EventDef(name="PERF_CNT_0", id=1),
        EventDef(name="INSTR_EVENT_0", id=2),
        EventDef(name="INSTR_EVENT_1", id=3),
        EventDef(name="INSTR_VECTOR", id=4),
    ])
    # Stub run_lit discovery to use insts.bin.
    monkeypatch.setattr(_mod, "discover_test_config",
                        lambda name: _mod.TestConfig(insts_name="insts.bin", ctrlpkt_name=None))

    # Stub subprocess.run for the patch tool to no-op.
    fake_completed = MagicMock()
    fake_completed.returncode = 0
    monkeypatch.setattr(_mod.subprocess, "run", MagicMock(return_value=fake_completed))

    # Replace RunnerSession + ParseSession with stubs that record calls.
    runner_calls: list[dict] = []

    class StubRunner:
        def __init__(self, *a, **kw):
            self.side = kw.get("side", "?")
        def run_one(self, **kw):
            runner_calls.append(kw)
            # Side-effect: create the trace_out file so downstream code
            # is happy if it inspects the path.
            Path(kw["trace_out"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kw["trace_out"]).write_bytes(b"\x00")
            return {"ok": True}
        def close(self): pass

    parse_calls: list[dict] = []

    class StubParser:
        def __init__(self, *a, **kw): pass
        def parse_one(self, **kw):
            parse_calls.append(kw)
            # Write a minimal events.json so the invariance check has
            # something to read (and stays happy).
            if "out_events" in kw:
                Path(kw["out_events"]).parent.mkdir(parents=True, exist_ok=True)
                Path(kw["out_events"]).write_text(json.dumps({
                    "events": [{"name": "INSTR_EVENT_0", "ts": 100, "slot": 0,
                                "row": 2, "col": 0, "pkt_type": 0}],
                }))
            return {"ok": True, "events_count": 1, "cycles": 100, "empty": False}
        def close(self): pass

    monkeypatch.setattr(_mod, "RunnerSession", StubRunner)
    monkeypatch.setattr(_mod, "ParseSession", StubParser)

    tiles = [TileSpec(col=0, row=2, tile_type="core")]
    _mod.sweep_lockstep(
        test_name="fake",
        compiler="chess",
        tiles=tiles,
        build_dir=build_dir,
        out_dir=out_dir,
        run_hw=True,
        run_emu=False,  # keep mocking surface small
        mode="event_pc",
        with_mode2_baseline=True,
    )

    # Expect parse_one called for: each regular batch + the mode-2 batch.
    # Mode-2 must have trace_mode="inst_exec"; regular ones have "event_pc".
    modes_seen = [c.get("trace_mode") for c in parse_calls]
    assert "inst_exec" in modes_seen, (
        f"mode-2 batch never decoded with inst_exec; saw: {modes_seen}"
    )
    # All non-mode-2 calls should be event_pc.
    non_inst_exec = [m for m in modes_seen if m != "inst_exec"]
    assert all(m == "event_pc" for m in non_inst_exec), (
        f"regular batches should decode with event_pc; saw: {non_inst_exec}"
    )

    # Manifest should also exist and report mode2_baseline_captured=True.
    manifest = json.loads((out_dir / "sweep-manifest.json").read_text())
    assert manifest["mode2_baseline_captured"] is True
    assert manifest["mode"] == "event_pc"


def test_sweep_lockstep_writes_manifest_on_patch_failure(tmp_path, monkeypatch):
    """If the patch subprocess raises CalledProcessError partway through,
    the manifest still lands and reports completed_batches < n_batches."""
    import subprocess as _sp

    TileSpec = _mod.TileSpec
    EventDef = _mod.EventDef
    build_dir, out_dir, _ = _make_min_lockstep_env(tmp_path)

    # Force enough events to produce 3 batches. Per cursor: 4 events,
    # 5 remaining slots (8 - 3 grounding). 4/5 = 1 batch. To get 3
    # batches, give 13 events (ceil(13/5) = 3).
    monkeypatch.setattr(_mod, "load_events", lambda tt: [
        EventDef(name="PERF_CNT_0", id=1),
        EventDef(name="INSTR_EVENT_0", id=2),
        EventDef(name="INSTR_EVENT_1", id=3),
    ] + [EventDef(name=f"E{i}", id=10 + i) for i in range(13)])
    monkeypatch.setattr(_mod, "discover_test_config",
                        lambda name: _mod.TestConfig(insts_name="insts.bin", ctrlpkt_name=None))

    # Make the patch subprocess fail on the 2nd batch (calls 0 and 1
    # are the regular patch calls; we fail call index 1).
    call_count = {"n": 0}

    def fake_run(cmd, **kwargs):
        # Only count calls that look like the patcher; let other
        # subprocess.run uses (none currently inside sweep_lockstep,
        # but just in case) pass through unscathed.
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise _sp.CalledProcessError(returncode=2, cmd=cmd, stderr=b"boom")
        m = MagicMock(); m.returncode = 0; return m

    monkeypatch.setattr(_mod.subprocess, "run", fake_run)

    class StubRunner:
        def __init__(self, *a, **kw): pass
        def run_one(self, **kw):
            Path(kw["trace_out"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kw["trace_out"]).write_bytes(b"\x00")
            return {"ok": True}
        def close(self): pass

    class StubParser:
        def __init__(self, *a, **kw): pass
        def parse_one(self, **kw):
            if "out_events" in kw:
                Path(kw["out_events"]).parent.mkdir(parents=True, exist_ok=True)
                Path(kw["out_events"]).write_text(
                    json.dumps({"events": []}))
            return {"ok": True, "events_count": 0, "cycles": 0, "empty": True}
        def close(self): pass

    monkeypatch.setattr(_mod, "RunnerSession", StubRunner)
    monkeypatch.setattr(_mod, "ParseSession", StubParser)

    tiles = [TileSpec(col=0, row=2, tile_type="core")]
    with pytest.raises(RuntimeError, match="sweep_lockstep failed"):
        _mod.sweep_lockstep(
            test_name="fake",
            compiler="chess",
            tiles=tiles,
            build_dir=build_dir,
            out_dir=out_dir,
            run_hw=True,
            run_emu=False,
            mode="event_pc",
            with_mode2_baseline=False,  # focus on the sweep failure path
        )

    # Despite the raise, the manifest must exist and report partial
    # completion + the error.
    manifest_path = out_dir / "sweep-manifest.json"
    assert manifest_path.exists(), "manifest must land even on patch failure"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_batches_completed"] < manifest["n_batches"]
    assert manifest["sweep_error"] is not None
    assert "CalledProcessError" in manifest["sweep_error"]
    # mode2_baseline_captured must be False because the sweep didn't
    # reach the mode-2 stage.
    assert manifest["mode2_baseline_captured"] is False
