# SPDX-License-Identifier: MIT
"""Tests for tools/trace_count_regress.py (bridge-suite regression gate).

Synthetic result dirs are built in tmp_path: one ``events.json`` per
``<test>.<compiler>.<hw|emu>`` directory, the same layout emu-bridge-test.sh
writes.
"""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import trace_count_regress as tcr


# The traced event set, the same on every side of every test unless a test says otherwise.
_EVENT_SET = ["INSTR_VECTOR", "X", "LOCK_STALL", "DMA_S2MM_0_STREAM_STARVATION",
              "DMA_S2MM_1_STREAM_STARVATION", "EXTRA_EVT", "", ""]
_KINDS = {0: "core", 1: "mem", 2: "shim", 3: "memtile"}


def _slot(name):
    return _EVENT_SET.index(name)


def level(name, starts, dur=2, col=1, row=2, pkt=0):
    """A level event: one event per cycle while high, so dur events per interval."""
    return [
        dict(col=col, row=row, pkt_type=pkt, name=name, ts=s + k, slot=_slot(name), soc=0, mode=0)
        for s in starts
        for k in range(dur)
    ]


def edge(name, stamps, col=1, row=2, pkt=0):
    return [
        dict(col=col, row=row, pkt_type=pkt, name=name, ts=t, slot=_slot(name), soc=0, mode=0)
        for t in stamps
    ]


def periodic(n, gap, t0=1000):
    return [t0 + i * gap for i in range(n)]


# --- a mode-0 trace_raw.bin encoder for the tests ---------------------------

def _frame_commands(events):
    """Per-cycle events of one tile -> the frames silicon writes: an open frame carrying the idle
    cycles, then for a run of identical masks on consecutive cycles one (mask, 0) frame and a
    Repeat for the rest."""
    masks: dict[int, int] = {}
    for e in events:
        masks[e["ts"]] = masks.get(e["ts"], 0) | (1 << e["slot"])
    out, prev, ts_list = [], 0, sorted(masks)
    i = 0
    while i < len(ts_list):
        ts, m = ts_list[i], masks[ts_list[i]]
        run = 1
        while (i + run < len(ts_list) and ts_list[i + run] == ts + run
               and masks[ts_list[i + run]] == m):
            run += 1
        out.append((m, ts - prev - 1, 0))
        if run >= 2:
            out.append((m, 0, 0))
        if run >= 3:
            out.append((None, None, run - 2))
        prev = ts + run - 1
        i += run
    return out


def quiet_unit(col=1, row=2, pkt=0):
    """A trace unit that was captured (Start frame, Stop token) and recorded no event."""
    return dict(col=col, row=row, pkt_type=pkt, unit_only=True, ts=0, slot=0, name="", soc=0, mode=0)


def cut_tail(events):
    """The same events, but their unit's stream ends without the Stop token (last packet lost)."""
    return [dict(e, no_stop=True) for e in events]


def _tile_bytes(events):
    stop = not any(e.get("no_stop") for e in events)
    events = [e for e in events if not e.get("unit_only")]
    b = bytearray([0xF0]) + bytearray(7)          # Start, timer 0
    for m, cycles, rep in _frame_commands(events):
        if m is None:
            n = rep
            while n:
                c = min(n, 1023)
                b += bytes([0xD8 | (c >> 8), c & 0xFF])
                n -= c
        else:
            b += bytes([0xD4 | (m >> 6), ((m & 0x3F) << 2) | (cycles >> 16),
                        (cycles >> 8) & 0xFF, cycles & 0xFF])
    if stop:
        b += bytes([0xDC, 0, 0, 0])               # End token, zero idle cycles
    return b


def _raw_trace(events):
    """events (per-cycle, absolute cols) -> trace_raw.bin bytes, one packet stream per tile."""
    words = []
    tiles = sorted({(e["col"], e["row"], e["pkt_type"]) for e in events})
    for col, row, pt in tiles:
        b = _tile_bytes([e for e in events if (e["col"], e["row"], e["pkt_type"]) == (col, row, pt)])
        b += bytes([0xFE]) * (-len(b) % 28)
        hdr = (col << 21) | (row << 16) | (pt << 12) | 1
        if bin(hdr).count("1") % 2 == 0:
            hdr |= 1 << 31
        for off in range(0, len(b), 28):
            words.append(hdr)
            words += [int.from_bytes(b[off + j:off + j + 4], "big") for j in range(0, 28, 4)]
    return struct.pack(f"<{len(words)}I", *words)


def _events_json(events):
    """What the in-tree rebuild writes for the raw trace: no events for held cycles."""
    import trace_decoder as td
    words = list(struct.unpack(f"<{len(_raw_trace(events)) // 4}I", _raw_trace(events)))
    cmds = td.decode_words(words, mode=None)
    plain = td.rebuild_timeline_mode0(cmds, {})
    return [dict(col=e.col, row=e.row, pkt_type=e.pkt_type, slot=e.slot,
                 name=_EVENT_SET[e.slot], ts=e.ts, soc=e.soc, mode=0)
            for e in plain]


def write_run(root: Path, day: str, tests: dict, raw=True, stamp=True, observed_placement=False) -> Path:
    """tests: {(test, compiler): {"hw": [events] | None, "emu": [events] | None}}

    Level events are given one per asserted cycle.  With ``raw`` each side gets the trace_raw.bin
    that encodes them and an events.json as the decoder writes it (held cycles absent); without,
    only an events.json holding the given events.  ``stamp`` marks each capture with its day so
    sides given the same events are still distinct captures (a copied-forward dir is the
    ``stamp=False`` case).  ``observed_placement`` writes the placement the trace parser writes: the
    smallest column and row that produced an event, so it moves with the set of traced tiles.
    """
    d = root / day
    d.mkdir(parents=True)
    for (test, comp), sides in tests.items():
        for side, events in sides.items():
            if events is None:
                continue
            sd = d / f"{test}.{comp}.{side}"
            sd.mkdir()
            if stamp:
                shift = {"base": 0, "cand": 1}.get(day, 2 + len(day))
                events = [dict(e, ts=e["ts"] + shift) for e in events]
            slot_names = {k: list(_EVENT_SET) for k in _KINDS.values()}
            doc = {"schema_version": 1, "events": events, "slot_names": slot_names,
                   "placement": {"origin_col": 1, "origin_row": 0}}
            if observed_placement and events:
                doc["placement"] = {"origin_col": min(e["col"] for e in events if not e.get("unit_only")),
                                    "origin_row": min(e["row"] for e in events if not e.get("unit_only"))}
            if raw and events:
                (sd / "trace_raw.bin").write_bytes(_raw_trace(events))
                doc["events"] = _events_json(events)
            else:
                doc["events"] = [e for e in events if not e.get("unit_only")]
            (sd / "events.json").write_text(json.dumps(doc))
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
    assert set(run_data[K]["hw"].series) == set(run_data[K]["emu"].series)


# A shim row-0 unit that one capture carries and the other lacks.  The parser's observed origin is
# the lowest traced row, so it differs between the two captures of one test; rows are physical
# (0 shim, 1 memtile, 2+ compute) and must not follow it.
def _core_and_shim():
    core = edge("INSTR_VECTOR", periodic(5, 50), col=1, row=2) + level("LOCK_STALL", [1000], col=1, row=2)
    shim = edge("DMA_S2MM_0_STREAM_STARVATION", periodic(3, 70), col=1, row=0, pkt=2)
    return core, shim


def test_rows_are_physical_and_do_not_follow_the_lowest_traced_row(tmp_path):
    core, shim = _core_and_shim()
    d = write_run(tmp_path, "r", {K: {"hw": core + shim, "emu": core}}, observed_placement=True)
    docs = {s: json.loads((d / f"t.peano.{s}" / "events.json").read_text())["placement"] for s in ("hw", "emu")}
    assert (docs["hw"]["origin_row"], docs["emu"]["origin_row"]) == (0, 2)   # the artefact's precondition
    sides = tcr.load_run(d)[K]
    assert (0, 2, 0, "INSTR_VECTOR") in sides["hw"].series
    assert (0, 2, 0, "INSTR_VECTOR") in sides["emu"].series
    assert (0, 0, 2, "DMA_S2MM_0_STREAM_STARVATION") in sides["hw"].series
    assert set(sides["emu"].units) == {(0, 2, 0)} and (0, 0, 2) in sides["hw"].units


def test_an_extra_silicon_shim_unit_leaves_the_core_rows_measured(tmp_path):
    core, shim = _core_and_shim()
    b = write_run(tmp_path, "base", {K: {"hw": core + shim, "emu": core}}, observed_placement=True)
    moved = edge("INSTR_VECTOR", periodic(9, 50), col=1, row=2) + level("LOCK_STALL", [1000], col=1, row=2)
    c = write_run(tmp_path, "cand", {K: {"emu": moved}}, observed_placement=True)
    rep = tcr.compare_runs(b, c)
    assert rep.not_measured == []
    # the core row lands on its own coordinate on both sides, so the move is scored against silicon
    r = only_row(rep, "INSTR_VECTOR")
    assert (r.key.col, r.key.row, r.key.module) == (0, 2, 0) and r.hw_counts == [5]
    # the one thing missing is the emulator's shim unit, at its own coordinate
    (u,) = rep.emu_absent
    assert (u.col, u.row, u.module) == (0, 0, 2)
    assert rep.exit_code == 0


def test_the_offset_section_places_tiles_on_physical_rows_too(tmp_path):
    import mem_core_offset
    core, shim = _core_and_shim()
    mem = edge("X", periodic(3, 70), col=1, row=2, pkt=1)
    d = write_run(tmp_path, "r", {K: {"hw": core + mem + shim, "emu": core + mem}}, observed_placement=True)
    tiles = {s: mem_core_offset.load_tiles(d / f"t.peano.{s}") for s in ("hw", "emu")}
    assert set(tiles["hw"]) == set(tiles["emu"]) == {(0, 2)}


def test_only_filter(tmp_path):
    hw = edge("X", periodic(10, 10))
    base = {K: {"hw": hw, "emu": hw}, ("u", "peano"): {"hw": hw, "emu": hw}}
    cand = {K: {"hw": hw, "emu": edge("X", periodic(30, 10))},
            ("u", "peano"): {"hw": hw, "emu": edge("X", periodic(30, 10))}}
    rep = run(tmp_path, base, cand, only=["u"])
    assert {r.key.test for r in rep.rows} == {"u"}


# --- level rows are scored on true assertion intervals ---------------------------

def test_a_held_level_with_foreign_pulses_inside_is_one_interval(tmp_path):
    # events.json shows the level only at frames where the tile's mask changes (the pulses);
    # the true series is one interval starting at its first asserted cycle.
    held = level("LOCK_STALL", [1000], dur=100)
    pulses = edge("INSTR_VECTOR", [1010, 1030, 1050])
    d = write_run(tmp_path, "r", {K: {"hw": held + pulses}}, stamp=False)
    doc = json.loads((d / "t.peano.hw" / "events.json").read_text())
    shown = [e for e in doc["events"] if e["name"] == "LOCK_STALL"]
    assert 1 < len(shown) < 100                       # events.json: mask changes only
    side = tcr.load_run(d)[K]["hw"]
    assert side.series[(0, 2, 0, "LOCK_STALL")] == [1000]


def test_a_level_dropped_for_one_cycle_is_two_intervals(tmp_path):
    ev = level("LOCK_STALL", [1000, 1011], dur=10)    # asserted 1000..1009, then 1011..1020
    d = write_run(tmp_path, "r", {K: {"hw": ev}}, stamp=False)
    assert tcr.load_run(d)[K]["hw"].series[(0, 2, 0, "LOCK_STALL")] == [1000, 1011]


def test_level_rows_without_a_raw_trace_are_unscored_not_counted(tmp_path):
    hw = level("LOCK_STALL", periodic(20, 140)) + edge("INSTR_VECTOR", periodic(10, 50))
    emu_b = level("LOCK_STALL", periodic(20, 140)) + edge("INSTR_VECTOR", periodic(10, 50))
    emu_c = level("LOCK_STALL", periodic(30, 140)) + edge("INSTR_VECTOR", periodic(25, 50))
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": emu_b}}, raw=False)
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": emu_c}}, raw=False)
    rep = tcr.compare_runs(b, c)
    assert [r.key.event for r in rep.rows] == ["INSTR_VECTOR"]       # edge rows still scored
    assert [(k.event, "no trace_raw.bin" in why) for k, why in rep.unscored] == [("LOCK_STALL", True)]
    text = tcr.format_text(rep)
    assert "unscored rows: 1" in text and "no trace_raw.bin" in text


def test_a_raw_trace_that_does_not_reproduce_events_json_leaves_levels_unscored(tmp_path):
    ev = level("LOCK_STALL", periodic(5, 100))
    d = write_run(tmp_path, "r", {K: {"hw": ev}})
    other = write_run(tmp_path, "o", {K: {"hw": level("LOCK_STALL", periodic(9, 70))}})
    (d / "t.peano.hw" / "trace_raw.bin").write_bytes((other / "t.peano.hw" / "trace_raw.bin").read_bytes())
    side = tcr.load_run(d)[K]["hw"]
    assert "does not reproduce" in side.level_problem


# --- silicon samples --------------------------------------------------------

def test_extra_hw_samples_widen_the_silicon_band(tmp_path):
    # Two samples (10, 12) leave EMU 11 -> 12 inside; one sample (10) would call it a regression.
    emu = lambda n: edge("X", periodic(n, 50))
    b = write_run(tmp_path, "base", {K: {"hw": edge("X", periodic(10, 50)), "emu": emu(10)}})
    c = write_run(tmp_path, "cand", {K: {"hw": edge("X", periodic(10, 50)), "emu": emu(12)}})
    extra = write_run(tmp_path, "extra", {K: {"hw": edge("X", periodic(12, 50))}})
    one = only_row(tcr.compare_runs(b, c), "X")
    assert one.hw_counts == [10, 10] and one.count_after == 2     # base and cand differ by shift only
    rep = tcr.compare_runs(b, c, hw_samples=[extra])
    assert rep.rows == [] or only_row(rep, "X").count_after == 0
    assert all(sorted(r.hw_counts) == [10, 10, 12] for r in rep.rows)


def test_identical_hw_captures_count_once(tmp_path):
    hw = edge("X", periodic(10, 50))
    t = {K: {"hw": hw, "emu": edge("X", periodic(14, 50))}}
    b = write_run(tmp_path, "base", t, stamp=False)
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": edge("X", periodic(10, 50))}}, stamp=False)
    r = only_row(tcr.compare_runs(b, c), "X")
    assert r.hw_counts == [10] and not r.hw_repeatable
    rep = tcr.compare_runs(b, c, hw_samples=[b])
    assert only_row(rep, "X").hw_counts == [10]
    assert any("identical" in why for *_, why in rep.dropped_samples)


def test_a_sample_with_another_event_set_is_not_used(tmp_path):
    t = {K: {"hw": edge("X", periodic(10, 50)), "emu": edge("X", periodic(14, 50))}}
    b = write_run(tmp_path, "base", t)
    c = write_run(tmp_path, "cand", {K: {"hw": edge("X", periodic(10, 50)),
                                         "emu": edge("X", periodic(10, 50))}})
    extra = write_run(tmp_path, "extra", {K: {"hw": edge("X", periodic(40, 50))}})
    doc_path = extra / "t.peano.hw" / "events.json"
    doc = json.loads(doc_path.read_text())
    doc["slot_names"]["core"] = ["Y"] + doc["slot_names"]["core"][1:]
    doc_path.write_text(json.dumps(doc))
    rep = tcr.compare_runs(b, c, hw_samples=[extra])
    assert only_row(rep, "X").hw_counts == [10, 10]
    assert any("event set" in why for *_, why in rep.dropped_samples)


# --- absent units and cut tails -----------------------------------------------
#
# A unit whose whole stream fits in the one stop-flushed packet can be lost with it: no Start
# frame, no events.  That is "absent", not "recorded zero events".  A unit with a Start but no
# End token lost the tail of its stream ("tail not delivered").

def absent_setup(tmp_path, hw_sides, emu_base, emu_cand, hw_extra=()):
    """The baseline dir holds the silicon sample hw_sides and emu_base, the candidate dir emu_cand;
    each of hw_extra is one more silicon day."""
    b = write_run(tmp_path, "base", {K: {"hw": hw_sides, "emu": emu_base}})
    c = write_run(tmp_path, "cand", {K: {"emu": emu_cand}})
    extra = [write_run(tmp_path, f"extra{i}", {K: {"hw": e}}) for i, e in enumerate(hw_extra)]
    return b, c, extra


def test_a_sample_without_the_unit_does_not_widen_the_band(tmp_path):
    mem = lambda n: edge("X", periodic(n, 50), pkt=1)
    only_core = edge("INSTR_VECTOR", periodic(5, 50))           # the memory unit never arrived
    b, c, x = absent_setup(tmp_path, mem(10), mem(10), mem(14), hw_extra=[mem(10), only_core])
    rep = tcr.compare_runs(b, c, hw_samples=x)
    r = only_row(rep, "X")
    # the zero of the absent sample used to open the band to [-10, 20] and hide 14
    assert r.hw_counts == [10, 10] and (r.hw_carrying, r.hw_total) == (2, 3)
    assert r.gate_fail and rep.exit_code == 1
    assert "carried by 2 of 3 HW samples" in tcr.format_text(rep)


def test_a_unit_that_recorded_zero_events_is_a_measured_zero(tmp_path):
    hw = [quiet_unit(pkt=1)]
    b, c, x = absent_setup(tmp_path, hw, [quiet_unit(pkt=1)], edge("X", periodic(3, 50), pkt=1))
    r = only_row(tcr.compare_runs(b, c), "X")
    assert r.hw_counts == [0] and (r.hw_carrying, r.hw_total) == (1, 1) and r.gate_fail


def test_a_row_no_sample_measured_is_listed_not_scored(tmp_path):
    only_core = edge("INSTR_VECTOR", periodic(5, 50))
    mem = lambda n: edge("X", periodic(n, 50), pkt=1)
    b, c, x = absent_setup(tmp_path, only_core, mem(10) + only_core, mem(14) + only_core,
                           hw_extra=[only_core])
    rep = tcr.compare_runs(b, c, hw_samples=x)
    assert not [r for r in rep.rows if r.key.event == "X"]
    (nm,) = rep.not_measured
    assert (nm.key.event, nm.key.module, nm.hw_total, nm.emu_before, nm.emu_after) == ("X", 1, 2, 10, 14)
    assert rep.exit_code == 0
    assert "not measured on silicon (1)" in tcr.format_text(rep)
    assert tcr.to_json(rep)["not_measured"][0]["emu_after"] == 14


def test_an_absent_emulator_unit_that_silicon_carries_fails(tmp_path):
    core = edge("INSTR_VECTOR", periodic(5, 50))
    b, c, x = absent_setup(tmp_path, core + [quiet_unit(pkt=1)], core + [quiet_unit(pkt=1)], core)
    rep = tcr.compare_runs(b, c)
    (u,) = rep.emu_absent
    assert (u.module, u.hw_carrying) == (1, 1) and rep.exit_code == 1
    assert "absent from the candidate emulator trace" in tcr.format_text(rep)


def test_an_emulator_unit_absent_where_silicon_lacks_it_too_is_not_a_failure(tmp_path):
    core = edge("INSTR_VECTOR", periodic(5, 50))
    b, c, x = absent_setup(tmp_path, core, core + [quiet_unit(pkt=1)], core)
    rep = tcr.compare_runs(b, c)
    assert rep.emu_absent == [] and rep.exit_code == 0


def test_a_unit_both_emulator_traces_lack_is_listed_as_standing_not_gated(tmp_path):
    core = edge("INSTR_VECTOR", periodic(5, 50))
    b, c, x = absent_setup(tmp_path, core + [quiet_unit(pkt=1)], core, core)
    rep = tcr.compare_runs(b, c)
    assert len(rep.emu_absent) == 1 and rep.emu_lost == [] and rep.exit_code == 0
    assert "standing, not gated" in tcr.format_text(rep)


def test_a_stream_without_its_end_token_is_reported_as_cut(tmp_path):
    mem = lambda n: edge("X", periodic(n, 50), pkt=1)
    b, c, x = absent_setup(tmp_path, cut_tail(mem(10)), mem(10), mem(14), hw_extra=[mem(10)])
    rep = tcr.compare_runs(b, c, hw_samples=x)
    r = only_row(rep, "X")
    assert (r.hw_carrying, r.hw_total, r.hw_tail_cut) == (2, 2, 1)
    assert "tail not delivered in 1" in tcr.format_text(rep)
    (u,) = [u for u in rep.units if u.module == 1]
    assert (u.hw_total, u.hw_carrying, u.hw_tail_cut) == (2, 2, 1)


def test_cut_tail_samples_stay_in_the_band_unless_excluded(tmp_path):
    mem = lambda n: edge("X", periodic(n, 50), pkt=1)
    b, c, x = absent_setup(tmp_path, mem(10), mem(10), mem(14), hw_extra=[cut_tail(mem(6)), mem(10)])
    rep = tcr.compare_runs(b, c, hw_samples=x)
    assert only_row(rep, "X").hw_counts == [10, 6, 10]
    rep = tcr.compare_runs(b, c, hw_samples=x, exclude_cut_tails=True)
    r = only_row(rep, "X")
    assert r.hw_counts == [10, 10] and r.hw_tail_cut == 1 and r.hw_carrying == 3
    assert "1 of them left out of the band" in tcr.format_text(rep)


def test_excluding_cut_tails_with_none_left_is_not_measured(tmp_path):
    mem = lambda n: edge("X", periodic(n, 50), pkt=1)
    b, c, x = absent_setup(tmp_path, cut_tail(mem(6)), mem(10), mem(14))
    rep = tcr.compare_runs(b, c, exclude_cut_tails=True)
    assert not rep.rows and len(rep.not_measured) == 1 and rep.exit_code == 0
    assert "tail not delivered in every sample" in tcr.format_text(rep)
    assert only_row(tcr.compare_runs(b, c), "X").hw_counts == [6]


def test_a_capture_without_a_raw_trace_cannot_show_absence(tmp_path):
    hw = edge("X", periodic(10, 50), pkt=1)
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": hw}}, raw=False)
    c = write_run(tmp_path, "cand", {K: {"emu": edge("X", periodic(14, 50), pkt=1)}}, raw=False)
    r = only_row(tcr.compare_runs(b, c), "X")
    assert (r.hw_carrying, r.hw_total, r.hw_tail_cut) == (1, 1, 0)


def test_the_offset_section_skips_a_sample_whose_unit_is_absent(tmp_path):
    b, c, x = offset_run(tmp_path, 1, 1)
    only_core = edge("INSTR_VECTOR", periodic(3, 20))
    extra = write_run(tmp_path, "gone", {K: {"hw": only_core}})
    rep = tcr.compare_runs(b, c, hw_samples=x + [extra])
    row = offset_row(rep)
    assert row.spread == 0 and row.baseline == 1      # the unit-less sample added no offset


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


def test_cli_hw_samples_option(tmp_path, capsys):
    hw = edge("INSTR_VECTOR", periodic(10, 50))
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": hw}})
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": edge("INSTR_VECTOR", periodic(25, 50))}})
    x = write_run(tmp_path, "extra", {K: {"hw": edge("INSTR_VECTOR", periodic(11, 50))}})
    assert tcr.main([str(b), str(c), "--hw-samples", str(x), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["regressed"][0]["hw_counts"]) == 3
    assert tcr.main([str(b), str(c), "--hw-samples", str(tmp_path / "nope")]) == 2


def test_cli_missing_dir_is_error(tmp_path):
    assert tcr.main([str(tmp_path / "nope"), str(tmp_path / "nope2")]) == 2


# --- cross-module (memory minus core) offset ---------------------------------

def unit_pair(mem_shift, core_shift=0, tile=(1, 2), n=3):
    """One core edge row and one memory edge row, each seen n times; the memory unit shifted."""
    col, row = tile
    return (edge("INSTR_VECTOR", periodic(n, 20, 1000 + core_shift), col=col, row=row, pkt=0)
            + edge("X", periodic(n, 20, 1005 + mem_shift), col=col, row=row, pkt=1))


def offset_run(tmp_path, base_mem, cand_mem, hw_mems=(0,), k=K, hw_core=0):
    """Baseline and candidate dirs whose emulator memory unit is shifted by base_mem / cand_mem
    cycles against silicon; one extra silicon sample per entry of hw_mems."""
    b = write_run(tmp_path, "base", {k: {"hw": unit_pair(hw_mems[0], hw_core), "emu": unit_pair(base_mem)}}, raw=False, stamp=False)
    c = write_run(tmp_path, "cand", {k: {"hw": unit_pair(hw_mems[0], hw_core), "emu": unit_pair(cand_mem)}}, raw=False, stamp=False)
    extra = [write_run(tmp_path, f"hw{i}", {k: {"hw": unit_pair(m)}}, raw=False, stamp=False)
             for i, m in enumerate(hw_mems[1:])]
    return b, c, extra


def offset_row(rep, k=K, tile=(0, 2)):
    rows = [r for r in rep.offsets if (r.test, r.compiler, r.tile) == (*k, tile)]
    assert len(rows) == 1
    return rows[0]


def test_a_uniform_memory_shift_in_the_candidate_is_flagged(tmp_path):
    b, c, x = offset_run(tmp_path, 0, 1)
    rep = tcr.compare_runs(b, c, hw_samples=x)
    row = offset_row(rep)
    assert (row.baseline, row.candidate) == (0, 1)
    assert row.gate_fail and rep.exit_code == 1
    assert "[GATE]" in tcr.format_text(rep)


def test_a_shift_toward_silicon_is_better_not_flagged(tmp_path):
    b, c, x = offset_run(tmp_path, 2, 0)
    rep = tcr.compare_runs(b, c, hw_samples=x)
    row = offset_row(rep)
    assert row.verdict == "better" and not row.gate_fail and rep.exit_code == 0


def test_a_shift_inside_silicons_own_spread_is_not_flagged(tmp_path):
    # silicon's memory unit itself sits at 0, 1 or 2 across samples: emulator at 1 or 2 is inside
    b, c, x = offset_run(tmp_path, 1, 2, hw_mems=(0, 1, 2))
    row = offset_row(tcr.compare_runs(b, c, hw_samples=x))
    assert row.spread == 2 and row.verdict == "unchanged" and not row.gate_fail


def test_a_shift_leaving_silicons_spread_is_flagged(tmp_path):
    b, c, x = offset_run(tmp_path, 1, 3, hw_mems=(0, 1, 2))
    row = offset_row(tcr.compare_runs(b, c, hw_samples=x))
    assert row.gate_fail


def test_an_unmeasurable_tile_is_reported_and_not_flagged(tmp_path):
    only_core = edge("INSTR_VECTOR", periodic(3, 20))
    b = write_run(tmp_path, "base", {K: {"hw": only_core, "emu": only_core}}, raw=False, stamp=False)
    c = write_run(tmp_path, "cand", {K: {"hw": only_core, "emu": only_core}}, raw=False, stamp=False)
    rep = tcr.compare_runs(b, c)
    assert rep.offsets == [] and rep.exit_code == 0
    # measurable on the baseline only: unmeasured, never failed
    b2 = write_run(tmp_path, "base2", {K: {"hw": unit_pair(0), "emu": unit_pair(0)}}, raw=False, stamp=False)
    c2 = write_run(tmp_path, "cand2", {K: {"hw": unit_pair(0), "emu": only_core}}, raw=False, stamp=False)
    rep = tcr.compare_runs(b2, c2)
    row = offset_row(rep)
    assert row.verdict == "unmeasured" and row.candidate is None and not row.gate_fail
    assert not any(o.gate_fail for o in rep.offsets) and "unmeasured" in tcr.format_text(rep)


def test_offset_summary_line_and_switch(tmp_path, capsys):
    b, c, x = offset_run(tmp_path, 0, 1)
    assert tcr.main([str(b), str(c)]) == 1
    out = capsys.readouterr().out
    assert "cross-module offset: 1 tiles measured; median candidate offset 1; " \
           "within 0.5 of silicon: baseline 1, candidate 0" in out
    assert out.rstrip().endswith("RESULT: FAIL")
    assert tcr.main([str(b), str(c), "--no-offset"]) == 0
    assert "cross-module" not in capsys.readouterr().out


def test_a_move_on_a_workload_dominated_tile_is_reported_not_flagged(tmp_path):
    # the memory unit sits 500 cycles from silicon on both sides: the pairing is workload, not timer
    b, c, x = offset_run(tmp_path, 500, 501)
    rep = tcr.compare_runs(b, c, hw_samples=x)
    row = offset_row(rep)
    assert row.verdict == "workload-dominated" and not row.gate_fail
    assert not any(o.gate_fail for o in rep.offsets)


# --- standing misses --------------------------------------------------------

def standing_setup(tmp_path, hw, emu, hw_extra=()):
    """The same emulator capture in baseline and candidate, so no row changes."""
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": emu}})
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": emu}})
    extra = [write_run(tmp_path, f"extra{i}", {K: {"hw": e}}) for i, e in enumerate(hw_extra)]
    return b, c, extra


def test_a_standing_miss_is_counted_and_listed_in_json(tmp_path):
    hw = edge("X", periodic(10, 50))
    b, c, x = standing_setup(tmp_path, hw, edge("X", periodic(3, 50)), hw_extra=[hw])
    rep = tcr.compare_runs(b, c, hw_samples=x)
    assert rep.rows == [] and rep.exit_code == 0
    (m,) = rep.standing
    assert (m.key.event, m.emu_count, m.hw_counts) == ("X", 3, [10, 10, 10])
    doc = tcr.to_json(rep)
    assert doc["summary"]["rows_standing_miss"] == 1
    (j,) = doc["standing_misses"]
    assert j["key"]["test"] == "t" and j["key"]["event"] == "X" and j["emu_count"] == 3
    assert j["hw_counts"] == [10, 10, 10]
    text = tcr.format_text(rep)
    assert "Standing misses (1)" in text and "RESULT: ok" in text


def test_a_row_inside_the_band_is_not_a_standing_miss(tmp_path):
    hw = edge("X", periodic(10, 50))
    b, c, x = standing_setup(tmp_path, hw, edge("X", periodic(10, 50)), hw_extra=[hw])
    rep = tcr.compare_runs(b, c, hw_samples=x)
    assert rep.standing == []
    assert "Standing misses" not in tcr.format_text(rep)


def test_the_standing_miss_uses_the_widened_band(tmp_path):
    # silicon 10 and 12: band [8, 14]; an emulator at 13 sits inside, at 15 outside
    a, b2 = edge("X", periodic(10, 50)), edge("X", periodic(12, 50))
    for n, expect in ((13, 0), (15, 1)):
        d = tmp_path / str(n)
        d.mkdir()
        b = write_run(d, "base", {K: {"hw": a, "emu": edge("X", periodic(n, 50))}})
        c = write_run(d, "cand", {K: {"hw": a, "emu": edge("X", periodic(n, 50))}})
        x = write_run(d, "extra", {K: {"hw": b2}})
        assert len(tcr.compare_runs(b, c, hw_samples=[x]).standing) == expect


def test_a_row_not_measured_on_silicon_is_not_a_standing_miss(tmp_path):
    only_core = edge("INSTR_VECTOR", periodic(5, 50))
    mem = edge("X", periodic(7, 50), pkt=1)
    b, c, x = standing_setup(tmp_path, only_core, mem + only_core)
    rep = tcr.compare_runs(b, c)
    assert len(rep.not_measured) == 1 and rep.standing == []


def test_a_changed_row_is_not_double_reported(tmp_path):
    hw = edge("X", periodic(10, 50))
    b = write_run(tmp_path, "base", {K: {"hw": hw, "emu": edge("X", periodic(3, 50))}})
    c = write_run(tmp_path, "cand", {K: {"hw": hw, "emu": edge("X", periodic(4, 50))}})
    rep = tcr.compare_runs(b, c)
    assert len(rep.rows) == 1 and rep.standing == []


def test_a_zero_emulator_row_the_silicon_unit_carries_is_a_standing_miss(tmp_path):
    hw = edge("X", periodic(1, 50)) + edge("INSTR_VECTOR", periodic(5, 50))
    b, c, x = standing_setup(tmp_path, hw, edge("INSTR_VECTOR", periodic(5, 50)))
    rep = tcr.compare_runs(b, c)
    assert [(m.key.event, m.emu_count) for m in rep.standing] == [("X", 0)]


def test_standing_misses_never_change_the_exit_status(tmp_path):
    hw = edge("X", periodic(10, 50))
    b, c, x = standing_setup(tmp_path, hw, edge("X", periodic(3, 50)))
    assert tcr.compare_runs(b, c).exit_code == 0
    assert tcr.main([str(b), str(c)]) == 0
