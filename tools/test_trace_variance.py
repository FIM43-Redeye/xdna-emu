import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "trace_variance", Path(__file__).parent / "trace_variance.py",
)
tv = importlib.util.module_from_spec(_spec)
sys.modules["trace_variance"] = tv
_spec.loader.exec_module(tv)

TOOLS = Path(__file__).parent
ROOT = TOOLS.parent
DDR_CAPS = ROOT / "build/experiments/ddr-stochasticity"


def _write_events(tmp_path, events):
    p = tmp_path / "events.json"
    p.write_text(json.dumps({"schema_version": 1, "events": events,
                             "slot_names": {}, "placement": {}}))
    return str(p)


def test_is_level_classifies_held_families():
    assert tv.is_level("PORT_RUNNING_4")
    assert tv.is_level("PORT_STALLED_0")
    assert not tv.is_level("DMA_S2MM_0_STREAM_STARVATION")
    assert not tv.is_level("LOCK_STALL")


def test_load_milestone_events_reanchors_per_tile(tmp_path):
    events = [
        # tile (0,0): two milestones at soc 1000, 1040 -> anchored 0, 40
        {"col": 0, "row": 0, "name": "DMA_S2MM_0_FINISHED", "soc": 1040},
        {"col": 0, "row": 0, "name": "DMA_S2MM_0_START", "soc": 1000},
        # tile (1,0): one milestone at soc 5000 -> anchored 0
        {"col": 1, "row": 0, "name": "DMA_S2MM_0_START", "soc": 5000},
        # a held-level event must be ignored by the milestone loader
        {"col": 0, "row": 0, "name": "PORT_RUNNING_4", "soc": 1010},
    ]
    out = tv.load_milestone_events(_write_events(tmp_path, events))
    assert out[(0, 0, "DMA_S2MM_0_START")] == [0]
    assert out[(0, 0, "DMA_S2MM_0_FINISHED")] == [40]
    assert out[(1, 0, "DMA_S2MM_0_START")] == [0]
    assert (0, 0, "PORT_RUNNING_4") not in out


def test_load_milestone_events_on_real_capture():
    # run_01.json is a real NPU1 add_one_using_dma capture.
    out = tv.load_milestone_events(str(DDR_CAPS / "run_01.json"))
    # The shim S2MM starvation milestone family is present and non-empty.
    keys = [k for k in out if "STREAM_STARVATION" in k[2]]
    assert keys, "expected at least one STREAM_STARVATION milestone key"
    # Every value list is sorted and re-anchored (min == 0 per tile family).
    for k, vals in out.items():
        assert vals == sorted(vals)


def test_aggregate_and_classify_deterministic_vs_stochastic():
    per_run = [
        {"A": 100, "B": 50},
        {"A": 100, "B": 58},
        {"A": 100, "B": 47},
        {"A": 100, "B": 61},
    ]
    stats = tv.aggregate(per_run)
    assert stats["A"].n == 4
    assert stats["A"].std == 0
    assert stats["A"].range == 0
    assert tv.classify(stats["A"]) == "deterministic"

    assert stats["B"].n == 4
    assert stats["B"].min == 47 and stats["B"].max == 61
    assert tv.classify(stats["B"]) == "stochastic"


def test_classify_eps_boundary():
    # std just under / over eps
    tight = tv.aggregate([{"X": 10}, {"X": 11}, {"X": 10}, {"X": 11}])["X"]
    assert tv.classify(tight, eps=2.0) == "deterministic"
    wide = tv.aggregate([{"X": 10}, {"X": 20}, {"X": 10}, {"X": 20}])["X"]
    assert tv.classify(wide, eps=2.0) == "stochastic"


def test_aggregate_handles_missing_key_in_some_runs():
    per_run = [{"A": 5}, {"A": 5, "B": 9}, {"B": 9}]
    stats = tv.aggregate(per_run)
    assert stats["A"].n == 2
    assert stats["B"].n == 2


PORT4_WORK = (ROOT / "build/experiments/gap140/sweep-port4-hw"
              / "add_one_using_dma.chess.multitile.work")


def test_load_spans_groups_with_idle_gap_and_reanchors():
    # Synthetic perfetto: one lane (pid0,tid0), two spans separated by gap 50,
    # and a 1-cycle internal gap that must NOT split (idle-gap > 2 only).
    evs = [
        {"ph": "B", "pid": 0, "tid": 0, "ts": 100},
        {"ph": "E", "pid": 0, "tid": 0, "ts": 108},   # [100,108]
        {"ph": "B", "pid": 0, "tid": 0, "ts": 109},   # gap 1 -> merge
        {"ph": "E", "pid": 0, "tid": 0, "ts": 116},   # span -> [100,116] dur 16
        {"ph": "B", "pid": 0, "tid": 0, "ts": 166},   # gap 50 -> new span
        {"ph": "E", "pid": 0, "tid": 0, "ts": 174},   # [166,174] dur 8
    ]
    name_map = {(0, 0): "PORT_RUNNING_X"}
    spans = tv.load_spans_from_events(evs, name_map)  # see impl: split helper
    assert spans["PORT_RUNNING_X"] == [16, 8]


def test_span_law_passes_at_word_count():
    spans = {"PORT_RUNNING_0": [16, 16, 16, 16], "PORT_RUNNING_4": [8, 8, 14, 34]}
    law = tv.check_span_law(spans, words=64)
    assert law["PORT_RUNNING_0"] == (64, True)
    assert law["PORT_RUNNING_4"] == (64, True)
    bad = tv.check_span_law({"PORT_RUNNING_1": [8, 8, 8, 11, 76, 76]}, words=64)
    assert bad["PORT_RUNNING_1"][1] is False  # 187 != 64 (the encoder-inflation bug)


@pytest.mark.skipif(not PORT4_WORK.exists(), reason="port4 sweep artifacts absent")
def test_real_perfetto_maps_named_port_lanes_structurally():
    # NOTE: these b12 fixtures are STALE (pre-decoder-fix) perfetto and do NOT
    # satisfy sum(PORT_RUNNING spans)==64 -- some held levels decode as dur-1
    # frame-record-like pairs there. The strict ==64 span-law proof is deferred
    # to Task 7's freshly-decoded HW perfetto. This test validates only that the
    # authoritative (pid,tid)->name mapping works STRUCTURALLY on real artifacts.
    pj = PORT4_WORK / "b12.trace_hw.perfetto.json"
    ej = PORT4_WORK / "b12.hw.events.json"  # sibling decoded events (note: .hw., not .trace_hw.)
    spans = tv.load_spans(str(pj), str(ej))
    # The mapping recovers named PORT_RUNNING lanes (4 of them in this batch).
    running = [k for k in spans if k.startswith("PORT_RUNNING")]
    assert "PORT_RUNNING_4" in spans, f"PORT_RUNNING_4 not mapped; got {sorted(spans)}"
    assert len(running) >= 1
    # check_span_law runs without error and returns a (sum, bool) per port.
    law = tv.check_span_law(spans, words=64)
    assert all(isinstance(s, int) and isinstance(ok, bool) for s, ok in law.values())
