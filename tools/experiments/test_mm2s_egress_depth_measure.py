#!/usr/bin/env python3
"""HW-free fixture tests for mm2s_egress_depth.py + mm2s_egress_depth_measure.py.

Plants a synthetic decoded-events capture (the same perfetto B/E +
trace_config.json shape bankdisc_measure.load_intervals consumes) with FOUR
STALLED_LOCK windows, each paired with a STARVATION onset a HAND-CHOSEN,
DISTINCT number of beats later: 7, 12, 3, 20. Distinct on purpose -- an
implementation that returns the first window's delay, the min, the last
window's delay, or an un-normalized duration would all disagree with the
correct answer (MAX over VALID windows = 12). The fourth window additionally
overlaps a STREAM_BACKPRESSURE interval and carries the LARGEST delay (20),
specifically so that a measure() which fails to exclude it from the max would
report 20 instead of 12 -- proving the invalidity check actually gates the
depth reading, not just sets a flag nothing reads.

Expected values below are derived BY HAND from the planted timestamps in
_EVENTS (see the comment above it), never by running the code under test.

Run: python3 tools/experiments/test_mm2s_egress_depth_measure.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mm2s_egress_depth import (  # noqa: E402
    MARCH_ADDR, SRC_ADDR, SINGLE_TILE_VARIANTS, TWO_TILE_K, VARIANTS, emit,
)
from mm2s_egress_depth_measure import measure  # noqa: E402

_SLOTS = [
    "DMA_MM2S_0_STALLED_LOCK",         # tid 0
    "DMA_MM2S_0_MEMORY_STARVATION",    # tid 1
    "DMA_MM2S_0_STREAM_BACKPRESSURE",  # tid 2
]

# Four STALLED_LOCK windows, hand-chosen distinct onset delays:
#   A: stall [ 100,  500) -> STARVATION @ 107  delay  7   (valid)
#   B: stall [ 600,  900) -> STARVATION @ 612  delay 12   (valid, the ceiling)
#   C: stall [1000, 1400) -> STARVATION @1003  delay  3   (valid)
#   D: stall [2000, 2500) -> STARVATION @2020  delay 20   BUT STREAM_
#      BACKPRESSURE [2010, 2400) overlaps [2000, 2020) -- INVALID; if wrongly
#      included the max would read 20, not 12.
_EVENTS = [
    ("B", 0, 100),  ("E", 0, 500),     # STALLED_LOCK A
    ("B", 1, 107),  ("E", 1, 120),     # STARVATION A (delay 7)
    ("B", 0, 600),  ("E", 0, 900),     # STALLED_LOCK B
    ("B", 1, 612),  ("E", 1, 630),     # STARVATION B (delay 12)
    ("B", 0, 1000), ("E", 0, 1400),    # STALLED_LOCK C
    ("B", 1, 1003), ("E", 1, 1010),    # STARVATION C (delay 3)
    ("B", 0, 2000), ("E", 0, 2500),    # STALLED_LOCK D
    ("B", 1, 2020), ("E", 1, 2100),    # STARVATION D (delay 20, invalid)
    ("B", 2, 2010), ("E", 2, 2400),    # STREAM_BACKPRESSURE overlapping D
]
_EXPECTED_DELAYS = [7, 12, 3, 20]
_EXPECTED_VALID = [True, True, True, False]
_EXPECTED_DEPTH = 12


def _write_synthetic_capture(build_dir: Path):
    build_dir.mkdir(parents=True, exist_ok=True)
    perfetto = [{"ph": "M", "name": "process_name", "pid": 0,
                 "args": {"name": "mem(0,2)"}}]
    for ph, tid, ts in _EVENTS:
        perfetto.append({"ph": ph, "pid": 0, "tid": tid, "ts": ts})
    (build_dir / "perfetto_r1.json").write_text(json.dumps(perfetto))
    config = {"tiles_traced": [{"module": "mem", "events": _SLOTS}]}
    (build_dir / "trace_config.json").write_text(json.dumps(config))


def _measure_fixture() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic_capture(Path(tmp))
        return measure(Path(tmp), 1)


def test_measure_recovers_distinct_planted_delays():
    result = _measure_fixture()
    assert result["n_windows"] == 4, result["n_windows"]
    delays = [w["onset_delay"] for w in result["windows"]]
    assert delays == _EXPECTED_DELAYS, delays


def test_depth_is_max_of_valid_windows_only():
    result = _measure_fixture()
    assert result["depth_estimate"] == _EXPECTED_DEPTH, result["depth_estimate"]


def test_backpressure_overlap_flags_window_invalid():
    result = _measure_fixture()
    valids = [w["valid"] for w in result["windows"]]
    assert valids == _EXPECTED_VALID, valids
    window_d = result["windows"][3]
    assert window_d["backpressure_overlap"] is True
    assert window_d["valid"] is False


def test_variants_cover_the_escalation_ladder():
    assert set(SINGLE_TILE_VARIANTS) == {"fill_stall", "never_stall", "cold", "fetch_starve"}
    assert "stream_backpressure" in TWO_TILE_K
    assert any(v.startswith("dwell_sweep_") for v in TWO_TILE_K)
    assert set(VARIANTS) == set(SINGLE_TILE_VARIANTS) | set(TWO_TILE_K)


def test_emit_does_not_crash_for_every_variant():
    for v in VARIANTS:
        text = emit(v)
        assert "aie.device(npu1_2col)" in text, v
        assert "aiex.npu.dma_wait" in text, v


def test_never_stall_ratchet_preloads_reps_credit_no_handshake():
    from mm2s_egress_depth import REPS
    text = emit("never_stall")
    assert f"init = {REPS} : i32" in text
    assert "lk_empty" not in text  # no producer/consumer handshake needed


def test_fetch_starve_hammers_same_bank_as_source():
    text = emit("fetch_starve")
    assert "march_buf" in text
    assert (MARCH_ADDR >> 14) & 3 == (SRC_ADDR >> 14) & 3


def test_stream_backpressure_and_dwell_sweep_share_two_tile_shape():
    for v in ("stream_backpressure", "dwell_sweep_16"):
        text = emit(v)
        assert "core_0_3" in text, v          # sink tile present
        assert "lk_sink_go" in text, v
        assert "lk_sink_ready" in text, v


def test_cold_is_a_single_rep_of_the_naive_skeleton():
    text = emit("cold")
    assert "%cREPS  = arith.constant 1 : index" in text


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
