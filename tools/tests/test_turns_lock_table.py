"""Tests for tools/turns_lock_table.py: per-iteration lock releases, channel grants and finishes."""
import pytest

import trace_decoder as td

import turns_lock_table as t


def ev(name, ts, pkt=1, row=2, col=1):
    return td.Event(col=col, row=row, pkt_type=pkt, slot=0, name=name, ts=ts, soc=ts)


def held(name, first, last, **kw):
    return [ev(name, x, **kw) for x in range(first, last + 1)]


def iteration(base, s2mm_grant=0, mm2s_grant=3, s2mm_fin=40, mm2s_fin=52, out_off=15, conflicts=()):
    """One loop turn: a released at `base`, out at base + out_off. The S2MM waited until its grant; the MM2S
    waited until its grant; each finished its BD later; conflict stamps are relative to the out release."""
    e = [ev("LOCK_1_REL", base), ev("LOCK_2_REL", base + out_off)]
    e += held("DMA_S2MM_0_STALLED_LOCK", base - 200, base + s2mm_grant - 1)
    e += held("DMA_MM2S_0_STALLED_LOCK", base - 100, base + out_off + mm2s_grant - 1)
    e += [ev("DMA_S2MM_0_FINISHED_BD", base + s2mm_grant + s2mm_fin),
          ev("DMA_MM2S_0_FINISHED_BD", base + out_off + mm2s_grant + mm2s_fin)]
    for bank, at in conflicts:
        e.append(ev(f"CONFLICT_DM_BANK_{bank}", base + out_off + at))
    return e


def test_rows_pair_each_grant_with_the_release_it_waited_for():
    e = iteration(1000, s2mm_grant=0, mm2s_grant=3) + iteration(2000, s2mm_grant=2, mm2s_grant=4)
    rows = t.turns_rows(e)
    assert [r["a_rel"] for r in rows] == [1000, 2000]
    assert [r["out_rel"] - r["a_rel"] for r in rows] == [15, 15]
    assert rows[0]["s2mm_grants"] == [0] and rows[1]["s2mm_grants"] == [2]
    assert rows[0]["mm2s_grants"] == [3] and rows[1]["mm2s_grants"] == [4]


def test_rows_finish_is_counted_from_its_own_grant():
    rows = t.turns_rows(iteration(1000, s2mm_grant=2, s2mm_fin=40, mm2s_grant=3, mm2s_fin=52))
    assert rows[0]["s2mm_fins"] == [40] and rows[0]["mm2s_fins"] == [52]


def test_rows_conflicts_are_relative_to_the_out_release_per_bank():
    rows = t.turns_rows(iteration(1000, conflicts=[(2, 4), (3, 8), (2, 12)]))
    assert rows[0]["conflicts"] == {2: [4, 12], 3: [8]}


def test_rows_before_the_first_a_release_are_left_out_and_a_missing_grant_is_none():
    e = [x for x in iteration(1000) if x.name != "DMA_S2MM_0_STALLED_LOCK"]
    e += [ev("DMA_S2MM_0_FINISHED_BD", 500)]                       # boot-time finish: before iteration 0
    rows = t.turns_rows(e)
    assert len(rows) == 1 and rows[0]["s2mm_grants"] == []


def test_rows_ignore_other_tiles_and_the_core_module():
    e = iteration(1000) + [ev("LOCK_1_REL", 1010, row=1), ev("LOCK_1_REL", 1011, pkt=0)]
    assert [r["a_rel"] for r in t.turns_rows(e)] == [1000]
    assert [r["a_rel"] for r in t.turns_rows(e + iteration(5000, ), col=1, row=2)] == [1000, 5000]


def test_format_table_one_line_per_iteration_and_a_summary_of_the_deltas():
    e = iteration(1000, s2mm_grant=0, mm2s_grant=3) + iteration(2000, s2mm_grant=2, mm2s_grant=3)
    text = t.format_turns_table(t.turns_rows(e), "e1.emu")
    lines = text.splitlines()
    assert any(ln.startswith("   0 ") and "a 1000" in ln and "S2MM grant +0" in ln for ln in lines)
    assert any(ln.startswith("   1 ") and "S2MM grant +2" in ln and "MM2S grant +3" in ln for ln in lines)
    summary = text.split("summary")[1]
    assert "S2MM grant after REL(a)" in summary and "+0 x1" in summary and "+2 x1" in summary
    assert "MM2S grant after REL(out)" in summary and "+3 x2" in summary


def test_main_rejects_a_missing_run_directory(tmp_path, capsys):
    assert t.main([str(tmp_path / "nope")]) == 2
