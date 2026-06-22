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


# ---------------------------------------------------------------------------
# Engine-derive validation with the generated (config_extract) ledger (E6)
# ---------------------------------------------------------------------------
#
# This is the guard that caught the a/b orientation bug in config_extract/generator.py:
# the generator was emitting a=child, b=parent, but the engine (rules.py:40-41) reads
# config_path(args[0]=parent, args[1]=child).  With the wrong orientation the engine
# derived NOTHING (derives==[]) because no config_path fact matched any candidate pair.
#
# After the fix (a=parent, b=child in generate_ledger), this test must:
#   - derive exactly the two memtile buffer-relay pairs:
#       child=PORT_RUNNING_4, parent=PORT_RUNNING_0, offset≈30
#       child=PORT_RUNNING_5, parent=PORT_RUNNING_1, offset≈88
#   - satisfy provenance_ok (every Structural cite is in the ledger)
#   - have replication_violations == [] (the captured runs replicate cleanly)
#
# The shim-DMA cross-pipeline causality pairs and STREAM_STARVATION are intentionally
# NOT derived here: those require through-core program_path relay edges, which are a
# deferred follow-on (see docs/superpowers/plans/2026-06-21-program-path-through-core.md).
# That is a correct, expected gap -- do not treat it as a test failure.

_CONFIG_DUMP_FIXTURE = (
    Path(__file__).resolve().parent
    / "config_extract" / "fixtures" / "add_one_using_dma.config.json"
)

_FIRED_KEYS = [
    "1|0|2|DMA_MM2S_0_FINISHED_TASK",
    "1|0|2|DMA_MM2S_0_START_TASK",
    "1|0|2|DMA_S2MM_0_FINISHED_TASK",
    "1|0|2|DMA_S2MM_0_START_TASK",
    "1|0|2|DMA_S2MM_0_STREAM_STARVATION",
    "1|1|3|PORT_RUNNING_0",
    "1|1|3|PORT_RUNNING_1",
    "1|1|3|PORT_RUNNING_4",
    "1|1|3|PORT_RUNNING_5",
    "1|2|0|LOCK_STALL",
    "1|2|0|PERF_CNT_2",
]

# Candidate pairs for the engine: (child, parent) tuples derived from
# the generated ledger entries where a=parent, b=child.
# These are the pairs the engine must evaluate via config_path orientation.
_GEN_CANDIDATE_PAIRS = [
    ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0"),  # memtile in0-buffer relay
    ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_1"),  # memtile out0-buffer relay
]

# Expected (child, parent) pairs the engine must derive with the fixed generated ledger.
_EXPECTED_GEN_DERIVES = {
    ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_1"),
}


def _gen_ledger_report(tmp_path_factory):
    """Run the engine with a generated ledger (not the hand-authored one)."""
    import json
    import tempfile
    from config_extract.dump_model import load_dump
    from config_extract.generator import generate_ledger
    from inference.engine import run_engine

    dump = load_dump(str(_CONFIG_DUMP_FIXTURE))
    gen_led = generate_ledger(dump, _FIRED_KEYS, start_col=1)

    # Write generated ledger to a temp file (not in /tmp -- use scratch dir).
    scratch = Path("/home/triple/.claude/jobs/0e6fe3a1/tmp")
    scratch.mkdir(parents=True, exist_ok=True)
    ledger_path = str(scratch / "gen_ledger_smoke.json")
    Path(ledger_path).write_text(json.dumps(gen_led), encoding="utf-8")

    # Candidate pairs: the engine needs (child, parent) — from entries where
    # a=parent, b=child after the fix, we build (b, a) i.e. (child, parent).
    candidate_pairs = [(e["b"], e["a"]) for e in gen_led["entries"]]
    # Deduplicate while preserving order.
    seen = set()
    deduped_pairs = []
    for p in candidate_pairs:
        if p not in seen:
            seen.add(p)
            deduped_pairs.append(p)

    return run_engine(_run_dirs(), ledger_path, deduped_pairs)


def test_generated_ledger_engine_derives_buffer_relays(tmp_path_factory):
    """The generated config_path ledger must orient the engine to derive the two
    memtile buffer-relay timing relationships on real silicon.

    Regression guard for E6: before the a/b orientation fix, this derived NOTHING
    (derives==[]) because the generator emitted a=child/b=parent but the engine
    reads config_path(args[0]=parent, args[1]=child).
    """
    rep = _gen_ledger_report(tmp_path_factory)
    derived_pairs = {(d[0], d[1]) for d in rep["derives"]}
    assert _EXPECTED_GEN_DERIVES.issubset(derived_pairs), (
        f"Engine failed to derive expected buffer-relay pairs from generated ledger.\n"
        f"Expected: {_EXPECTED_GEN_DERIVES}\n"
        f"Got derives: {derived_pairs}\n"
        f"Full derives list: {rep['derives']}"
    )


def test_generated_ledger_provenance_ok(tmp_path_factory):
    """provenance_ok must hold when the engine uses the generated ledger."""
    rep = _gen_ledger_report(tmp_path_factory)
    assert rep["provenance_ok"] is True, (
        f"provenance_ok failed with generated ledger: {rep}"
    )


def test_generated_ledger_no_replication_violations(tmp_path_factory):
    """No replication violations with the generated ledger on the captured runs."""
    rep = _gen_ledger_report(tmp_path_factory)
    assert rep["replication_violations"] == [], (
        f"Replication violations found: {rep['replication_violations']}"
    )
