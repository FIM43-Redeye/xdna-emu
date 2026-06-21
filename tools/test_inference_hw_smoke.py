"""Axis-2: the inference engine's output matches real silicon on add_one_using_dma.

HW-gated. Capture once with `tools/capture_infer_smoke.py` (drives
trace_capture.run_loop on the real NPU1, chess), then point this test at the
saved run dirs:

    cd tools && XDNA_HW_SMOKE=1 \
      XDNA_SMOKE_RUNS=../build/experiments/infer-smoke \
      python -m pytest test_inference_hw_smoke.py -v

The engine reads the captured run_NN/batch_00/hw/trace.events.json offline and,
given the hand-authored structural ledger (inference/fixtures/), independently
re-derives the capture-engine validation's 5 stochastic roots and the
deterministic backbone -- soundly: every `derives` is admitted only because
config_path orients it AND `correlates` confirms a stable offset on the measured
HW data, and provenance bottoms out in measured `fired` + ledgered `config_path`
leaves.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XDNA_HW_SMOKE") != "1",
    reason="HW smoke requires a real NPU capture; set XDNA_HW_SMOKE=1 and "
           "XDNA_SMOKE_RUNS=<capture dir> to run")

# Absolute-col-1 event keys (decoder space), verified against the captured
# trace.events.json. The five config-oriented candidate (child, parent) pairs
# that the sound engine must derive, leaving exactly the validation's 5 roots.
_CANDIDATE_PAIRS = [
    ("1|0|2|DMA_S2MM_0_START_TASK", "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|0|2|DMA_S2MM_0_STREAM_STARVATION", "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|1|3|PORT_RUNNING_1", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_0"),
]

# The capture-engine validation's 5 stochastic roots (the DMA-delivery degrees
# of freedom), in absolute-col-1 space.
_EXPECTED_ROOTS = {
    "1|0|2|DMA_MM2S_0_FINISHED_TASK",
    "1|0|2|DMA_MM2S_0_START_TASK",
    "1|0|2|DMA_S2MM_0_FINISHED_TASK",
    "1|1|3|PORT_RUNNING_0",
    "1|2|0|LOCK_STALL",
}

_LEDGER = (Path(__file__).resolve().parent
           / "inference" / "fixtures" / "add_one_using_dma.ledger.json")


def _run_dirs():
    cap = Path(os.environ["XDNA_SMOKE_RUNS"])
    dirs = sorted(str(p) for p in cap.glob("run_*") if p.is_dir())
    assert dirs, f"no run_* dirs under {cap}"
    return dirs


def _report():
    from inference.engine import run_engine
    return run_engine(_run_dirs(), str(_LEDGER), _CANDIDATE_PAIRS)


def test_engine_rederives_five_stochastic_roots():
    rep = _report()
    assert set(rep["stochastic_roots"]) == _EXPECTED_ROOTS


def test_engine_provenance_sound_and_replicates_on_silicon():
    rep = _report()
    assert rep["provenance_ok"] is True
    assert rep["replication_violations"] == []


def test_engine_places_all_five_derivable_events():
    rep = _report()
    derived_children = {d[0] for d in rep["derives"]}
    assert derived_children == {c for c, _ in _CANDIDATE_PAIRS}


def test_stream_starvation_is_placed_not_causal():
    # The spec's placement-not-causation case, on real silicon: STREAM_STARVATION
    # fires from downstream backpressure, yet is correctly PLACED relative to the
    # upstream MM2S start -- classified `derived`, never labeled causal.
    rep = _report()
    assert rep["classification"]["1|0|2|DMA_S2MM_0_STREAM_STARVATION"] == "derived"
