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

# Expected (child, parent) pairs the engine must derive with the fixed generated ledger.
_EXPECTED_GEN_DERIVES = {
    ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0"),
    ("1|1|3|PORT_RUNNING_5", "1|1|3|PORT_RUNNING_1"),
}


def _gen_ledger_report(tmp_path_factory):
    """Run the engine with a generated ledger (not the hand-authored one)."""
    import json
    from config_extract.dump_model import load_dump
    from config_extract.generator import generate_ledger
    from inference.engine import run_engine

    dump = load_dump(str(_CONFIG_DUMP_FIXTURE))
    gen_led = generate_ledger(dump, _FIRED_KEYS, start_col=1)

    # Write the generated ledger to a pytest-managed temp dir (portable; never
    # a hardcoded path).
    ledger_path = str(tmp_path_factory.mktemp("gen_ledger") / "gen_ledger_smoke.json")
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


# ---------------------------------------------------------------------------
# P7: E2E through-core derive (the C4 lesson applied to program_path)
# ---------------------------------------------------------------------------
#
# A ledger that loads is not a ledger that derives (C4 lesson).
# This test closes the loop on the program_path (through-core relay) stack:
#
#   P0.5-P2  static analysis emits core_lock_relay edges
#   P3       dump fixture carries the core_lock_relay edge
#   P4       engine rules.py unions config_path + program_path for orientation
#   P5       generator classifies through-core-only pairs as kind="program"
#   P6       runtime validation proves the static edges sound
#   P7       (HERE) the regenerated fixture + captured HW runs produce a
#            program entry AND the engine soundly DERIVES the through-core
#            timing relationship on real captured data.
#
# The two through-core pairs are backed by kind="program" ledger entries --
# the core_lock_relay edge in the config dump makes them reachable in the
# full graph but NOT in the config-only graph.  Their derivation on real
# silicon data is what distinguishes "program_path is wired in" from
# "program_path only loads."
#
# Derived from probe run (see task-P7-report.md):
#   child="1|0|2|DMA_S2MM_0_START_TASK",        parent="1|0|2|DMA_MM2S_0_START_TASK",   offset=934
#   child="1|0|2|DMA_S2MM_0_STREAM_STARVATION", parent="1|0|2|DMA_MM2S_0_START_TASK",   offset=939
# Both backed by program-kind entries (parent=MM2S_START -> child=S2MM_* via core).

# The two (child, parent) program_path-backed pairs the engine must derive.
# Pinned from the probe run: these are the only derives that require the
# core_lock_relay edge (program_path fact) -- config_path alone cannot orient them.
_EXPECTED_PROGRAM_DERIVES = {
    ("1|0|2|DMA_S2MM_0_START_TASK",        "1|0|2|DMA_MM2S_0_START_TASK"),
    ("1|0|2|DMA_S2MM_0_STREAM_STARVATION", "1|0|2|DMA_MM2S_0_START_TASK"),
}


def _gen_ledger_report_with_program(tmp_path_factory):
    """Run the engine with a generated ledger that includes program_path entries.

    Identical to _gen_ledger_report but returns the report from the same
    generated ledger so callers can assert on the program-kind content.
    Defined separately to keep the E6 fixture and P7 fixture independent
    (the tmp_path_factory isolation is the key invariant).
    """
    import json
    from config_extract.dump_model import load_dump
    from config_extract.generator import generate_ledger
    from inference.engine import run_engine

    dump = load_dump(str(_CONFIG_DUMP_FIXTURE))
    gen_led = generate_ledger(dump, _FIRED_KEYS, start_col=1)

    ledger_path = str(
        tmp_path_factory.mktemp("gen_ledger_prog") / "gen_ledger_program.json"
    )
    Path(ledger_path).write_text(json.dumps(gen_led), encoding="utf-8")

    candidate_pairs = [(e["b"], e["a"]) for e in gen_led["entries"]]
    seen: set = set()
    deduped_pairs = []
    for p in candidate_pairs:
        if p not in seen:
            seen.add(p)
            deduped_pairs.append(p)

    return gen_led, run_engine(_run_dirs(), ledger_path, deduped_pairs)


def test_generated_ledger_has_program_kind_entry(tmp_path_factory):
    """The regenerated fixture must yield at least one kind='program' ledger entry.

    If this fails, the fixture/generator chain is broken (P3 core_lock_relay
    edge missing or P5 classification broken) -- investigate before forcing.
    """
    gen_led, _rep = _gen_ledger_report_with_program(tmp_path_factory)
    program_entries = [e for e in gen_led["entries"] if e["kind"] == "program"]
    assert program_entries, (
        "generate_ledger produced NO kind='program' entries from the regenerated "
        "fixture.  The core_lock_relay edge is missing from the config dump or "
        "the generator's through-core classification is broken (P3/P5).\n"
        f"All entry kinds: {[e['kind'] for e in gen_led['entries']]}"
    )


def test_engine_derives_through_core_relay_from_generated_ledger(tmp_path_factory):
    """P7 capstone: the engine derives the through-core timing relationship on
    real captured NPU1 data, using only the generated (program_path) ledger.

    This is the C4/E6 lesson applied to program_path: a ledger that loads is
    not a ledger that derives.  The two expected pairs are backed exclusively
    by kind='program' (core_lock_relay) entries -- config_path alone cannot
    orient them, so their presence in rep['derives'] proves that program_path
    is wired into the rules, not merely serialized.
    """
    gen_led, rep = _gen_ledger_report_with_program(tmp_path_factory)
    derived_pairs = {(d[0], d[1]) for d in rep["derives"]}
    assert _EXPECTED_PROGRAM_DERIVES.issubset(derived_pairs), (
        f"Engine failed to derive expected through-core (program_path) pairs.\n"
        f"Expected: {_EXPECTED_PROGRAM_DERIVES}\n"
        f"Got derives: {derived_pairs}\n"
        f"Full derives list: {rep['derives']}\n"
        f"Hint: check that kind='program' entries produce program_path facts "
        f"in ledger.py and that rules.py unions config_path + program_path."
    )


def test_engine_derives_through_core_relay_provenance_ok(tmp_path_factory):
    """provenance_ok must hold for the program_path-extended generated ledger."""
    _gen_led, rep = _gen_ledger_report_with_program(tmp_path_factory)
    assert rep["provenance_ok"] is True, (
        f"provenance_ok failed for program_path-extended generated ledger: {rep}"
    )


def test_engine_derives_through_core_relay_no_replication_violations(tmp_path_factory):
    """No replication violations with the program_path-extended generated ledger."""
    _gen_led, rep = _gen_ledger_report_with_program(tmp_path_factory)
    assert rep["replication_violations"] == [], (
        f"Replication violations found: {rep['replication_violations']}"
    )
