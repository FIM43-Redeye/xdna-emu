# SPDX-License-Identifier: MIT
"""Tests for tools/trace_count_regress.py (bridge-suite regression gate).

Synthetic result dirs are built in tmp_path: one ``events.json`` per
``<test>.<compiler>.<hw|emu>`` directory, the same layout emu-bridge-test.sh
writes.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import trace_count_regress as tcr


def level(name, starts, dur=2, col=1, row=2, pkt=0):
    """A level event: one event per cycle while high, so dur events per interval."""
    return [
        dict(col=col, row=row, pkt_type=pkt, name=name, ts=s + k, slot=0, soc=0, mode=0)
        for s in starts
        for k in range(dur)
    ]


def edge(name, stamps, col=1, row=2, pkt=0):
    return [
        dict(col=col, row=row, pkt_type=pkt, name=name, ts=t, slot=0, soc=0, mode=0)
        for t in stamps
    ]


def periodic(n, gap, t0=1000):
    return [t0 + i * gap for i in range(n)]


def write_run(root: Path, day: str, tests: dict) -> Path:
    """tests: {(test, compiler): {"hw": [events] | None, "emu": [events] | None}}"""
    d = root / day
    d.mkdir(parents=True)
    for (test, comp), sides in tests.items():
        for side, events in sides.items():
            if events is None:
                continue
            sd = d / f"{test}.{comp}.{side}"
            sd.mkdir()
            (sd / "events.json").write_text(
                json.dumps({"schema_version": 1, "events": events,
                            "placement": {"origin_col": 1, "origin_row": 0}})
            )
    return d


def run(tmp_path, base, cand, **kw):
    b = write_run(tmp_path, "base", base)
    c = write_run(tmp_path, "cand", cand)
    return tcr.compare_runs(b, c, **kw)


def only_row(report, event):
    rows = [r for r in report.rows if r.key.event == event]
    assert len(rows) == 1, [r.key for r in report.rows]
    return rows[0]


K = ("t", "peano")


# --- primitives -----------------------------------------------------------

def test_wasserstein_identical_is_zero():
    assert tcr.wasserstein1([5, 7, 9], [9, 5, 7]) == 0


def test_wasserstein_shift():
    assert tcr.wasserstein1([128] * 10, [140] * 10) == 12


def test_wasserstein_unequal_lengths_quantile_matched():
    # same distribution sampled at different resolution: distance 0
    assert tcr.wasserstein1([10, 20], [10, 10, 20, 20]) == 0
    # constant gaps, any length
    assert tcr.wasserstein1([100] * 3, [130] * 50) == 30


def test_range_distance():
    assert tcr.range_distance(5, [5, 7]) == 0
    assert tcr.range_distance(9, [5, 7]) == 2
    assert tcr.range_distance(1, [5, 7]) == 4
    assert tcr.range_distance(0, [0]) == 0


def test_silicon_band_widens_a_jittery_range_by_its_spread():
    # Two samples underestimate silicon's spread: [39, 41] (spread 2) becomes [37, 43].
    assert tcr.silicon_band([39, 41]) == (37, 43)
    assert tcr.band_distance(38, [39, 41]) == 0
    assert tcr.band_distance(45, [39, 41]) == 2


def test_silicon_band_keeps_a_repeatable_count_exact():
    assert tcr.silicon_band([36, 36]) == (36, 36)
    assert tcr.band_distance(35, [36, 36]) == 1
    assert tcr.band_distance(0, [0, 0]) == 0
    assert tcr.silicon_band([17]) == (17, 17)


def test_one_event_wiggle_inside_silicon_jitter_does_not_gate(tmp_path):
    # objectfifo_channel_turns shape: silicon 39 and 41, EMU 41 -> 38, cadence a hair farther.
    b = write_run(tmp_path, "base", {K: {"hw": edge("X", periodic(39, 50)),
                                         "emu": edge("X", periodic(41, 50))}})
    c = write_run(tmp_path, "cand", {K: {"hw": edge("X", periodic(41, 50)),
                                         "emu": edge("X", periodic(38, 51))}})
    rep = tcr.compare_runs(b, c)
    assert not any(r.gate_fail for r in rep.rows)
    assert rep.exit_code == 0


def test_level_series_collapses_runs_to_interval_starts():
    ev = level("LOCK_STALL", [100, 200, 300], dur=3)
    got = tcr.series_from_events(ev, "LOCK_STALL")
    assert got == [100, 200, 300]


def test_edge_series_keeps_every_event():
    assert tcr.series_from_events(edge("INSTR_VECTOR", [1, 2, 3]), "INSTR_VECTOR") == [1, 2, 3]


@pytest.mark.parametrize("name", [
    "LOCK_STALL", "MEMORY_STALL", "STREAM_STALL", "DMA_S2MM_0_STREAM_STARVATION",
    "PORT_RUNNING_3", "CONFLICT_DM_BANK_2", "DMA_S2MM_0_STALLED_LOCK",
    "DMA_MM2S_SEL1_STREAM_BACKPRESSURE"])
def test_level_names(name):
    assert tcr.is_level_name(name)


@pytest.mark.parametrize("name", [
    "INSTR_VECTOR", "PERF_CNT_2", "DMA_S2MM_0_START_TASK", "INSTR_LOCK_ACQUIRE_REQ"])
def test_edge_names(name):
    assert not tcr.is_level_name(name)


# --- classification -------------------------------------------------------

def test_count_inside_hw_range_unchanged_is_not_reported(tmp_path):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    emu = edge("INSTR_VECTOR", periodic(10, 50))
    t = {K: {"hw": hw, "emu": emu}}
    rep = run(tmp_path, t, t)
    assert rep.changed == []
    assert rep.exit_code == 0


def test_count_moves_into_hw_range_is_improvement(tmp_path):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    base = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(14, 50))}}
    cand = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(10, 50))}}
    r = only_row(run(tmp_path, base, cand), "INSTR_VECTOR")
    assert (r.count_before, r.count_after) == (4, 0)
    assert r.verdict == "improved"


def test_count_outside_range_regresses_and_gates(tmp_path):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    base = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(10, 50))}}
    cand = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(25, 50))}}
    rep = run(tmp_path, base, cand)
    r = only_row(rep, "INSTR_VECTOR")
    assert r.verdict == "regressed"
    assert r.gate_fail
    assert rep.exit_code == 1


def test_two_hw_samples_form_a_range(tmp_path):
    # HW disagrees with itself (10 vs 12): EMU going 11 -> 12 stays inside.
    def mk(day_emu_n):
        return None
    b = write_run(tmp_path, "base", {K: {"hw": edge("X", periodic(10, 50)),
                                         "emu": edge("X", periodic(11, 50))}})
    c = write_run(tmp_path, "cand", {K: {"hw": edge("X", periodic(12, 50)),
                                         "emu": edge("X", periodic(12, 50))}})
    rep = tcr.compare_runs(b, c)
    # EMU count changed but both inside [10,12]; pattern identical -> neutral
    for r in rep.rows:
        assert r.count_before == r.count_after == 0
        assert not r.gate_fail
    assert all(not r.hw_repeatable for r in rep.rows)


def test_repeatable_exact_match_broken_gates(tmp_path):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    base = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(10, 50))}}
    cand = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(11, 50))}}
    r = only_row(run(tmp_path, base, cand), "INSTR_VECTOR")
    assert r.hw_repeatable
    assert r.exact_match_broken
    assert r.gate_fail


def test_phase_shift_is_not_a_regression(tmp_path):
    # objectfifo_turns_walkers: silicon gap 140, EMU 128 -> 131; the count
    # was equal only by a lucky phase alignment.
    hw = level("LOCK_STALL", periodic(20, 140))
    base = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 128))}}
    cand = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(21, 131))}}
    rep = run(tmp_path, base, cand)
    r = only_row(rep, "LOCK_STALL")
    assert r.exact_match_broken
    assert r.pattern_before == 12 and r.pattern_after == 9
    assert r.verdict == "neutral"
    assert r.label == "phase shift (pattern closer)"
    assert not r.gate_fail
    assert rep.exit_code == 0


def test_count_drift_with_unimproved_pattern_is_regression(tmp_path):
    hw = level("LOCK_STALL", periodic(20, 140))
    base = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 128))}}
    cand = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(21, 128))}}
    r = only_row(run(tmp_path, base, cand), "LOCK_STALL")
    assert r.verdict == "regressed" and r.gate_fail


def test_count_worse_pattern_unchanged_inexact_is_soft(tmp_path):
    # HW varies (30 vs 36, band [24, 42]): EMU 44 -> 45 is outside both times, no exact
    # match to break, cadence identical -> regressed but does not fail the gate.
    b = write_run(tmp_path, "base", {K: {"hw": level("LOCK_STALL", periodic(30, 140)),
                                         "emu": level("LOCK_STALL", periodic(44, 140))}})
    c = write_run(tmp_path, "cand", {K: {"hw": level("LOCK_STALL", periodic(36, 140)),
                                         "emu": level("LOCK_STALL", periodic(45, 140))}})
    rep = tcr.compare_runs(b, c)
    r = only_row(rep, "LOCK_STALL")
    assert r.verdict == "regressed" and not r.gate_fail and rep.exit_code == 0


def test_large_count_jump_not_excused_by_closer_pattern(tmp_path):
    hw = level("LOCK_STALL", periodic(20, 140))
    base = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 128))}}
    cand = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(60, 139))}}
    r = only_row(run(tmp_path, base, cand), "LOCK_STALL")
    assert r.verdict == "regressed" and r.gate_fail


def test_pattern_closer_with_same_count_is_improvement(tmp_path):
    hw = level("LOCK_STALL", periodic(20, 140))
    base = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 128))}}
    cand = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 138))}}
    r = only_row(run(tmp_path, base, cand), "LOCK_STALL")
    assert r.verdict == "improved"


def test_pattern_farther_with_same_count_regresses_softly(tmp_path):
    hw = level("LOCK_STALL", periodic(20, 140))
    base = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 138))}}
    cand = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 120))}}
    rep = run(tmp_path, base, cand)
    r = only_row(rep, "LOCK_STALL")
    assert r.verdict == "regressed" and not r.gate_fail
    assert rep.exit_code == 0


def test_count_closer_pattern_farther_is_mixed_neutral(tmp_path):
    hw = level("LOCK_STALL", periodic(20, 140))
    base = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(26, 139))}}
    cand = {K: {"hw": hw, "emu": level("LOCK_STALL", periodic(20, 100))}}
    r = only_row(run(tmp_path, base, cand), "LOCK_STALL")
    assert r.verdict == "neutral" and r.label.startswith("mixed")


def test_phantom_row_hw_has_none(tmp_path):
    # shim S2MM_1 unused on silicon; EMU grows phantom starvation intervals.
    used = level("DMA_S2MM_0_STREAM_STARVATION", periodic(32, 5000), row=0)
    base = {K: {"hw": used, "emu": used}}
    phantom = level("DMA_S2MM_1_STREAM_STARVATION", periodic(59, 700), row=0)
    cand = {K: {"hw": used, "emu": used + phantom}}
    rep = run(tmp_path, base, cand)
    r = only_row(rep, "DMA_S2MM_1_STREAM_STARVATION")
    assert (r.emu_count_before, r.emu_count_after) == (0, 59)
    assert r.hw_counts == [0, 0]
    assert r.pattern_after is None          # HW has no gaps to compare against
    assert r.verdict == "regressed" and r.gate_fail
    assert rep.exit_code == 1


def test_row_only_in_baseline_emu_counts_as_zero_in_candidate(tmp_path):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    base = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(10, 50))
                + edge("EXTRA_EVT", periodic(4, 30))}}
    cand = {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(10, 50))}}
    r = only_row(run(tmp_path, base, cand), "EXTRA_EVT")
    assert (r.emu_count_before, r.emu_count_after) == (4, 0)
    assert r.verdict == "improved"


def test_tests_present_in_only_one_run_are_listed_not_scored(tmp_path):
    hw = edge("X", periodic(5, 10))
    base = {K: {"hw": hw, "emu": hw}, ("old", "chess"): {"hw": hw, "emu": hw}}
    cand = {K: {"hw": hw, "emu": hw}, ("new", "chess"): {"hw": hw, "emu": hw}}
    rep = run(tmp_path, base, cand)
    assert rep.only_in_baseline == [("old", "chess")]
    assert rep.only_in_candidate == [("new", "chess")]
    assert rep.rows == [] or all(r.key.test == "t" for r in rep.rows)


def test_missing_emu_side_is_skipped(tmp_path):
    hw = edge("X", periodic(5, 10))
    t = {K: {"hw": hw, "emu": None}}
    rep = run(tmp_path, t, t)
    assert rep.rows == [] and rep.skipped


def test_tile_columns_are_shifted_by_placement(tmp_path):
    # EMU placed at origin col 2, HW at col 1: same logical tile.
    hw_ev = edge("INSTR_VECTOR", periodic(5, 50), col=1)
    emu_ev = edge("INSTR_VECTOR", periodic(5, 50), col=2)
    d = tmp_path / "r"
    for side, ev, oc in (("hw", hw_ev, 1), ("emu", emu_ev, 2)):
        sd = d / f"t.peano.{side}"
        sd.mkdir(parents=True)
        (sd / "events.json").write_text(json.dumps(
            {"events": ev, "placement": {"origin_col": oc, "origin_row": 0}}))
    run_data = tcr.load_run(d)
    assert set(run_data[K]["hw"]) == set(run_data[K]["emu"])


def test_only_filter(tmp_path):
    hw = edge("X", periodic(10, 10))
    base = {K: {"hw": hw, "emu": hw}, ("u", "peano"): {"hw": hw, "emu": hw}}
    cand = {K: {"hw": hw, "emu": edge("X", periodic(30, 10))},
            ("u", "peano"): {"hw": hw, "emu": edge("X", periodic(30, 10))}}
    rep = run(tmp_path, base, cand, only=["u"])
    assert {r.key.test for r in rep.rows} == {"u"}


# --- CLI ------------------------------------------------------------------

def test_cli_json_and_exit_code(tmp_path, capsys):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": hw}})
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(25, 50))}})
    rc = tcr.main([str(b), str(c), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["summary"]["regressed"] == 1
    assert out["exit_code"] == 1


def test_cli_text_summary(tmp_path, capsys):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": hw}})
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": hw}})
    rc = tcr.main([str(b), str(c)])
    assert rc == 0
    assert "rows changed: 0" in capsys.readouterr().out


def test_cli_missing_dir_is_error(tmp_path):
    assert tcr.main([str(tmp_path / "nope"), str(tmp_path / "nope2")]) == 2
