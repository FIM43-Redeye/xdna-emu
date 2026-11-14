"""Tests for tools/accept_probe_table.py: event lists, run dating, INVALID marking."""
from collections import namedtuple

import accept_probe_table as t

Ev = namedtuple("Ev", "col row pkt_type name ts")


def core(name, *stamps):
    return [Ev(1, 2, t.PT_CORE, name, s) for s in stamps]


def mem(name, *stamps):
    return [Ev(1, 2, t.PT_MEM, name, s) for s in stamps]


def valid_events():
    # stall 100..109 and 200..209; port runs 110..113 and 211..212
    return (core("LOCK_STALL", *range(100, 110), *range(200, 210))
            + core("PORT_RUNNING_0", *range(110, 114), 211, 212)
            + mem("DMA_S2MM_0_START_TASK", 90) + mem("DMA_S2MM_0_FINISHED_BD", 114, 213)
            + mem("CONFLICT_DM_BANK_3", 120, 121, 300))


def test_event_lists_fit_the_trace_units_and_name_the_questions_events():
    assert len(t.CORE_EVENTS) <= 5           # three grounding events take the other core slots
    assert len(t.MEMMOD_EVENTS) <= 8
    assert {"LOCK_STALL", "PORT_RUNNING_0"} <= set(t.CORE_EVENTS)
    assert set(t.REQUIRED_MEM) <= set(t.MEMMOD_EVENTS)


def test_env_lines_are_exports_of_the_event_lists():
    lines = dict(l.removeprefix("export ").split("=", 1) for l in t.env_lines().splitlines())
    assert lines["XDNA_TRACE_CORE_EVENTS"].split(",") == list(t.CORE_EVENTS)
    assert lines["XDNA_TRACE_MEMMOD_EVENTS"].split(",") == list(t.MEMMOD_EVENTS)


def test_level_runs_group_consecutive_cycles():
    assert t.level_runs([5, 3, 4, 9, 9, 11]) == [(3, 5), (9, 9), (11, 11)]


def test_port_runs_are_dated_against_the_nearest_stall_edges():
    info = t.analyse(valid_events())
    assert not info["missing"]
    first, second = info["port_runs"]
    # stall 100..109 falls at 110: the first run starts on the fall; the second starts 2 after the fall at 210
    assert (first["start"], first["end"], first["len"]) == (110, 114, 4)
    assert first["start_vs_fall"] == 0 and first["start_vs_rise"] == 10
    assert second["start_vs_fall"] == 1 and second["end_vs_fall"] == 3
    assert [len(info["conflicts"][b]) for b in t.BANKS] == [0, 0, 0, 3]


def test_nearest_picks_the_earlier_edge_on_a_tie():
    assert t.nearest([10, 20], 15) == 10
    assert t.nearest([], 15) is None


def test_missing_port_event_marks_the_run_invalid_and_hides_the_dating():
    events = [e for e in valid_events() if e.name != "PORT_RUNNING_0"]
    info = t.analyse(events)
    assert info["missing"] == ["core:PORT_RUNNING_0"]
    text = t.format_run(info, "run")
    assert "INVALID" in text and "start-vs-stall" not in text


def test_absent_memory_unit_is_invalid_not_zero_conflicts():
    events = [e for e in valid_events() if e.pkt_type != t.PT_MEM]
    info = t.analyse(events)
    assert "mem:DMA_S2MM_0_START_TASK" in info["missing"]
    assert "INVALID" in t.format_run(info, "run")


def test_other_tiles_are_ignored():
    other = [Ev(1, 3, t.PT_CORE, "LOCK_STALL", 5)]
    assert t.analyse(valid_events() + other)["stall_runs"] == 2
    assert t.analyse(valid_events(), col=2)["missing"]


def test_valid_run_prints_the_table():
    text = t.format_run(t.analyse(valid_events()), "run")
    assert "INVALID" not in text
    assert "B3 3" in text and "run start vs stall fall" in text
