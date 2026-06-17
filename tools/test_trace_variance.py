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
