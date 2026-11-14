"""Pure-logic tests for tools/gen_shim_cold_probe.py: the case list and its priority subset, the layout and
buffer arithmetic that the host-memory bounds rest on, the core programs' release order and delays, the
generated design text, the predictions and the table logic on synthetic events. No compiler, runner or
device."""
import re
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_shim_cold_probe as g  # noqa: E402


# -- cases ------------------------------------------------------------------------------------------------

def test_sixty_eight_cases_in_six_families():
    cs = g.default_cases()
    assert len(cs) == len({c.name for c in cs}) == 68
    fam = {f: sum(c.family == f for c in cs) for f in "ABCDEF"}
    assert fam == {"A": 12, "B": 20, "C": 12, "D": 16, "E": 4, "F": 4}


def test_priority_subset_has_the_specified_members():
    pri = {f: [c.name for c in g.priority_cases() if c.family == f] for f in "ABCDEF"}
    assert {f: len(v) for f, v in pri.items()} == {"A": 3, "B": 6, "C": 5, "D": 6, "E": 2, "F": 4}
    assert pri["A"] == ["a16d0", "a16d20000", "a4096d20000"]
    assert pri["B"] == ["b1g100", "b1g10000", "b3g100", "b3g10000", "b5g100", "b5g10000"]
    assert pri["C"] == ["c0i10000", "c0i100000", "c0i300000", "c1i100000", "c2i100000"]
    assert pri["D"] == ["d0i30000", "d0i100000", "d0i300000", "d1i100000", "d2i100000", "d3i100000"]


def test_priority_and_rest_partition_the_cases():
    assert len(g.priority_cases()) + len(g.rest_cases()) == 68
    assert not {c.name for c in g.priority_cases()} & {c.name for c in g.rest_cases()}


def test_case_from_name_round_trips_and_rejects_unknown():
    for c in g.default_cases():
        assert g.case_from_name(c.name) == c
    with pytest.raises(ValueError):
        g.case_from_name("a16d7")


def test_session_runs_each_set_three_rounds_with_distinct_tags():
    for which, n in (("priority", 26), ("rest", 42)):
        order = g.session_order(which)
        assert len(order) == 3 * n
        assert [t for _, t in order[:n]] == ["h1"] * n and [t for _, t in order[-n:]] == ["h3"] * n
        assert len(set(order)) == len(order)
    both = {name for name, _ in g.session_order("priority")} | {name for name, _ in g.session_order("rest")}
    assert both == {c.name for c in g.default_cases()}


def test_family_f_repeats_an_a_and_a_c_case_under_both_trace_configurations():
    f = {c.name: c for c in g.default_cases() if c.family == "F"}
    assert {c.trace for c in f.values()} == {"Q", "N"}
    assert f["fa16d1000q"].transfers == g.case_from_name("a16d1000").transfers
    assert f["fc0i100000q"].transfers == g.case_from_name("c0i100000").transfers


# -- buffers and bounds ------------------------------------------------------------------------------------------

def test_every_destination_lies_inside_the_words_the_case_declares():
    for c in g.default_cases():
        need = g.arg_words(c)
        for t in c.transfers:
            assert t.offset + t.span_words <= need[t.buf], c.name


def test_offset_variants_of_family_b():
    t2 = {v: g.case_from_name(f"b{v}g0").transfers[1] for v in "12345"}
    assert t2["1"].offset * g.WORD == 64 and t2["2"].offset * g.WORD == 4096
    assert t2["3"].offset * g.WORD == 64 * 1024 and t2["4"].offset * g.WORD == 2 * 1024 * 1024
    assert t2["5"].buf == 1 and t2["5"].offset == 0
    assert g.arg_words(g.case_from_name("b4g0"))[0] * g.WORD == 2 * 1024 * 1024 + 64
    assert g.arg_words(g.case_from_name("b5g0")) == {0: 16, 1: 16}


def test_prewarm_touches_one_word_in_each_of_the_four_burst_pages():
    case = g.case_from_name("d3i100000")
    pre, burst = case.transfers[1], case.transfers[2]
    pages = {(pre.offset + k * pre.stride) * g.WORD // g.PAGE for k in range(pre.words)}
    burst_pages = set(range(burst.offset * g.WORD // g.PAGE, (burst.offset + burst.words) * g.WORD // g.PAGE))
    assert pages == burst_pages and len(pages) == 4
    assert pre.span_words == 1 + 3 * 1024
    assert pre.offset + pre.span_words <= burst.offset + burst.words


def test_burst_pages_are_not_the_warm_up_page():
    for fill in "0123":
        case = g.case_from_name(f"d{fill}i30000")
        w, burst = case.transfers[0], case.transfers[-1]
        assert w.offset * g.WORD // g.PAGE not in range(burst.offset * g.WORD // g.PAGE,
                                                        (burst.offset + burst.words) * g.WORD // g.PAGE)


def test_source_buffers_are_packed_aligned_and_do_not_overlap():
    for c in g.default_cases():
        lay = g.source_layout(c)
        for (a, w), (b, _) in zip(lay, lay[1:]):
            assert b >= a + w * g.WORD and b % g.SRC_ALIGN == 0


def test_background_buffer_and_rounds():
    c1 = g.case_from_name("c1i100000")
    assert g.background_rounds(c1) == 100 and g.arg_words(c1)[1] == 100 * g.BG_WORDS
    long = g.case_from_name("c2i300000")
    assert g.background_rounds(long) == g.BG_MAX_ROUNDS
    assert g.background_rounds(long) * g.background_period(long) >= long.measured.gap - g.background_period(long)
    assert g.background_rounds(g.case_from_name("c0i1000")) == 0
    e1 = g.case_from_name("e1s20000")
    assert g.background_rounds(e1) == 16 and g.arg_words(e1)[1] == g.BG_WORDS


def test_expected_runner_sizes_include_the_trace_buffer_last():
    exp = g.expected_arg_bytes(g.case_from_name("b5g100"))
    assert exp == {0: 64, 1: 64, g.TRACE_ARG: g.TRACE_BUFFER_BYTES}
    assert g.expected_arg_bytes(g.case_from_name("a16d0")) == {0: 64, g.TRACE_ARG: g.TRACE_BUFFER_BYTES}


def test_the_one_word_tasks_of_e_step_by_four_bytes():
    ts = g.case_from_name("e0s80000").transfers
    assert len(ts) == 16 and [t.offset for t in ts] == list(range(16)) and {t.words for t in ts} == {1}


# -- timing of the core programs ------------------------------------------------------------------------------------

def test_release_spacings_add_the_previous_stream_time():
    c = g.case_from_name("b3g1000")
    assert g.release_spacings(c) == [0, 1000 + 16 + g.CHAIN_DELTA]
    a = g.case_from_name("a256d1000")
    assert g.release_spacings(a) == [1000]
    d = g.case_from_name("d3i30000")
    assert g.release_spacings(d) == [0, 0 + 16 + g.CHAIN_DELTA, 30000 + 4 + g.CHAIN_DELTA]


def test_delay_is_nops_when_short_and_a_counted_loop_when_long():
    assert g.delay(0, ".L") == [] and len(g.delay(5, ".L")) == 5
    long = g.delay(300000, ".L")
    movxm = next(x for x in long if "movxm" in x)
    iters = int(movxm.split("#")[1])
    nops_after = sum(1 for x in long[long.index(movxm):] if x.strip() == "nop")
    # loop body: LOOP_ITER cycles per iteration, then the remainder as nops; the fixed part is LOOP_BASE
    assert iters * g.LOOP_ITER + g.LOOP_BASE <= 300000 < (iters + 1) * g.LOOP_ITER + g.LOOP_BASE
    assert any(x.endswith(":") for x in long)
    with pytest.raises(ValueError):
        g.delay(-1, ".L")


def test_core_waits_for_go_then_releases_once_per_transfer():
    asm = g.core_asm(g.case_from_name("b1g100"))
    assert asm.count(f"acq #{g.OWN + g.LOCK_GO}") == 1
    assert asm.count(f"rel #{g.SOUTH + g.LOCK_SRC}") == 2
    assert asm.index("acq") < asm.index("rel")
    asm = g.core_asm(g.case_from_name("e0s20000"))
    assert asm.count("acq #") == 16 and asm.count("rel #") == 16


def test_noisy_trace_keeps_the_core_running_after_its_last_release():
    q, n = g.core_asm(g.case_from_name("fa16d1000q")), g.core_asm(g.case_from_name("fa16d1000n"))
    assert ".Ltail" in n and ".Ltail" not in q


def test_background_core_exists_only_for_core_paced_background_cases():
    assert g.background_asm(g.case_from_name("c0i1000")) is None
    assert g.background_asm(g.case_from_name("e1s20000")) is None
    asm = g.background_asm(g.case_from_name("c1i100000"))
    assert f"movxm r1, #{g.background_rounds(g.case_from_name('c1i100000'))}" in asm
    assert asm.count(f"rel #{g.SOUTH + g.LOCK_SRC}") == 1       # inside the loop


def test_padding_covers_the_core_schedule_with_slack():
    for c in g.default_cases():
        if c.paced_by == "core":
            assert g.sequence_pads(c) * g.OPS_CYCLES >= g.core_cycles(c) + g.SEQ_SLACK_CYCLES
    e = g.case_from_name("e0s80000")
    assert g.sequence_pads(e) == 0 and g.round_ops(e) * g.OPS_CYCLES == 80000


# -- the design ---------------------------------------------------------------------------------------------------------

def _mlir(name):
    return g.design_mlir(g.case_from_name(name))


def test_design_has_one_s2mm_task_and_one_source_bd_per_transfer():
    for name in ("a16d0", "b2g100", "d3i30000", "e0s20000"):
        case, text = g.case_from_name(name), _mlir(name)
        assert text.count("aiex.dma_configure_task(%t00, S2MM, 0)") == case.tasks
        assert len(re.findall(r"^    \^bd\d+:", text, re.M)) == case.tasks
        assert text.count("aie.use_lock(%lSrc, AcquireGreaterEqual") == case.tasks


def test_every_shim_bd_names_a_runtime_argument_and_a_length_equal_to_the_source():
    text = _mlir("b5g100")
    shim = re.findall(r"aie\.dma_bd\(%(dst|alt) : memref<(\d+)xi32> offset = (\d+) len = (\d+)\)", text)
    assert [(b[0], b[2], b[3]) for b in shim] == [("dst", "0", "16"), ("alt", "0", "16")]
    srcs = re.findall(r"aie\.dma_bd\(%src\d : memref<(\d+)xi32>\)", text)
    assert srcs == ["16", "16"]


def test_the_scatter_bd_writes_single_words_a_page_apart():
    text = _mlir("d3i30000")
    assert "sizes = [1, 1, 4, 1] strides = [0, 0, 1024, 1]" in text and "len = 4" in text


def test_all_dma_tasks_are_freed_without_a_wait():
    for c in g.default_cases():
        text = g.design_mlir(c)
        assert "dma_await_task" not in text and "wait" not in text.lower().replace("nothing waits", "")


def test_background_cases_declare_the_second_column_tiles():
    assert "%t12 = aie.tile(1, 2)" in _mlir("c1i1000") and "%t11 = aie.tile(1, 1)" in _mlir("e1s20000")
    assert "%t12" not in _mlir("e1s20000") and "%t11" not in _mlir("a16d0")


def test_flush_pulses_every_traced_unit_with_irregular_spacing():
    text = g.flush_block(g.case_from_name("a16d0"))
    per_unit = {}
    for m in re.finditer(r"column = (\d) : i32, row = (\d) : i32", text):
        per_unit[(m.group(1), m.group(2))] = per_unit.get((m.group(1), m.group(2)), 0) + 1
    assert per_unit == {("0", "0"): g.FLUSH_PULSES, ("1", "0"): g.FLUSH_PULSES, ("0", "1"): g.FLUSH_PULSES}
    assert ("1", "1") in {(m.group(1), m.group(2)) for m in
                          re.finditer(r"column = (\d) : i32, row = (\d) : i32", g.flush_block(g.case_from_name("c1i1000")))}
    # no two consecutive rounds have the same number of padding writes between them
    gaps = [len(x.splitlines()) for x in re.split(r"column = \d : i32, row = \d : i32} : i32, i32\n", text)]
    assert len(set(zip(gaps, gaps[1:]))) > 1


def test_the_flush_event_never_collides_with_the_carrier_units_start_and_stop():
    # the carrier shim's trace unit starts on USER_EVENT_1 (127) and stops on USER_EVENT_0 (126)
    assert g.SW_EVENT_SHIM[1] not in (126, 127)
    assert g.SW_EVENT_SHIM[0] in g.SHIM_EVENTS and g.SW_EVENT_MEMTILE[0] in g.MEMTILE_EVENTS
    assert len(g.SHIM_EVENTS) <= 8 and len(g.MEMTILE_EVENTS) <= 8


def test_trace_arguments_per_configuration():
    q, n = g.inject_args(g.case_from_name("a16d0")), g.inject_args(g.case_from_name("fa16d1000n"))
    assert q[q.index("--trace-tiles") + 1] == "" and "--core-grounding" not in q
    assert n[n.index("--trace-tiles") + 1] == "0:2" and n[n.index("--core-grounding") + 1] == "PERF_CNT_2"


# -- predictions ------------------------------------------------------------------------------------------------------

def test_family_a_is_cold_under_every_hypothesis():
    for c in g.default_cases():
        if c.family == "A":
            assert set(g.predict(c).values()) == {"cold"}


def test_family_b_follows_the_page_granule_reading():
    assert g.predict(g.case_from_name("b1g100"))["HP"] == "warm"
    for v in "2345":
        assert g.predict(g.case_from_name(f"b{v}g100"))["HP"] == "cold"
    p = g.predict(g.case_from_name("b3g10000"))
    assert p["HR"] == p["HC"] == "warm" and p["HI"] == "cold"
    assert g.predict(g.case_from_name("b3g100"))["HI"] == "warm"
    assert g.predict(g.case_from_name("b3g1000"))["HI"] == "?"


def test_family_c_and_d_predictions():
    c = g.predict(g.case_from_name("c0i300000"))
    assert (c["HP"], c["HR"], c["HC"], c["HI"], c["HE"]) == ("warm", "warm", "warm", "cold", "warm")
    assert g.predict(g.case_from_name("c1i100000"))["HI"] == "warm"
    assert g.predict(g.case_from_name("c1i100000"))["HE"] == "cold"
    d0, d3 = g.predict(g.case_from_name("d0i100000")), g.predict(g.case_from_name("d3i100000"))
    assert d0["HI"] == "slow" and d0["HP"] == "slow" and d3["HP"] == "normal"
    assert g.predict(g.case_from_name("d0i10000"))["HI"] == "normal"
    assert g.predict(g.case_from_name("d1i100000"))["HI"] == "normal"


def test_consistency_marks_contradictions_only_where_a_claim_exists():
    assert g.consistent("cold", "cold") == "ok" and g.consistent("cold", "warm") == "CONTRADICTS"
    assert g.consistent("?", "warm") == "-" and g.consistent("n/a", "slow") == "-"
    assert g.consistent("cold", "-") == "-" and g.consistent("mixed:none-cold", "cold 3 of 15") == "-"


def test_every_case_has_a_termination_line_and_a_prediction_for_each_hypothesis():
    for c in g.default_cases():
        assert g.termination(c).endswith("waits on nothing (padding writes, then the flush pulses).")
        assert set(g.predict(c)) == set(g.HYPOTHESES)


# -- the table ----------------------------------------------------------------------------------------------------------------

Ev = namedtuple("Ev", "ts row col pkt_type name")


def level(name, start, end, pkt=2, row=0, col=1):
    return [Ev(t, row, col, pkt, name) for t in range(start, end + 1)]


def pulse(name, t, pkt=2, row=0, col=1):
    return [Ev(t, row, col, pkt, name)]


def unit(events, origin=1):
    return g.Unit(events, origin)


def one_task_events(start=100, onset=300, fin=1000, words=16):
    ev = pulse("DMA_S2MM_0_START_TASK", start) + level("DMA_S2MM_0_STREAM_STARVATION", start + 6, onset - 1)
    ev += level("PORT_RUNNING_4", onset - 10, onset - 10 + words - 1, pkt=3, row=1, col=1)
    ev += pulse("DMA_S2MM_0_FINISHED_TASK", fin)
    ev += level("DMA_S2MM_1_STREAM_STARVATION", 0, 49, col=2) + level("DMA_S2MM_1_STREAM_STARVATION", 60, 199, col=2)
    return ev


def test_first_word_is_the_starvation_fall_and_f_and_x_follow():
    tr = g.task_rows(unit(one_task_events()), g.case_from_name("a16d0"))
    r = tr["rows"][0]
    assert (r["start"], r["fw"], r["fin"], r["waiting"]) == (100, 300, 1000, False)
    assert r["F"] == 700 and r["X"] == 700 - 15 and r["class"] == "cold"
    assert tr["notes"] == []
    assert r["window"] == 16 and r["excess"] == 0
    assert r["carrier_since"] == 300 - 200       # last carrier fall at 200, before the first word


def test_a_waiting_word_makes_the_first_word_start_plus_six():
    ev = [e for e in one_task_events() if "STARVATION" not in e.name or e.col == 2]
    r = g.task_rows(unit(ev), g.case_from_name("a16d0"))["rows"][0]
    assert r["waiting"] and r["fw"] == 106


def test_warm_task_is_classified_warm():
    r = g.task_rows(unit(one_task_events(fin=300 + 165)), g.case_from_name("a16d0"))["rows"][0]
    assert r["class"] == "warm" and r["X"] == 150


def test_one_word_task_uses_the_f_threshold():
    case = g.case_from_name("e0s20000")
    ev = pulse("DMA_S2MM_0_START_TASK", 100) + level("DMA_S2MM_0_STREAM_STARVATION", 106, 199)
    ev += pulse("DMA_S2MM_0_FINISHED_TASK", 200 + 165) + level("PORT_RUNNING_4", 190, 190, pkt=3, row=1, col=1)
    r = g.task_rows(unit(ev), case)["rows"][0]
    assert r["F"] == 165 and r["class"] == "warm"
    ev2 = [e for e in ev if e.name != "DMA_S2MM_0_FINISHED_TASK"] + pulse("DMA_S2MM_0_FINISHED_TASK", 200 + 600)
    assert g.task_rows(unit(ev2), case)["rows"][0]["class"] == "cold"


def test_missing_events_are_reported_not_hidden():
    tr = g.task_rows(unit(one_task_events()[:3]), g.case_from_name("b1g100"))
    assert any("START_TASK" in n or "FINISHED_TASK" in n for n in tr["notes"])
    assert any("source port" in n for n in tr["notes"])


def test_observed_slow_drain_is_read_from_the_window_excess():
    case = g.case_from_name("d0i100000")
    assert g.observed(case, [{"k": 0, "class": "warm", "excess": 0}]) == "-"    # the burst task never ran
    rows = [{"k": 0}, {"k": 1, "class": "plain", "excess": 3000}]
    assert g.observed(case, rows) == "slow"
    rows[-1]["excess"] = 20
    assert g.observed(case, rows) == "normal"


def test_table_prints_one_line_per_task_with_the_verdict_on_the_last():
    case = g.case_from_name("a16d0")
    tr = g.task_rows(unit(one_task_events()), case)
    text = g.format_table([(case, "h1.hw", tr)])
    line = [ln for ln in text.splitlines() if ln.startswith("a16d0")][0]
    assert "cold" in line and "HP:ok" in line and "HI:ok" in line
