import re
import gen_path_onset_probe as g


def test_a_length_suffix_sets_the_bd_length_and_keeps_the_route():
    assert g.case_spec("mem_own_n23") == ("mem", "mem", 23)
    assert g.case_spec("prod_mem") == ("prod", "mem", g.N_WORDS)
    text = g.design_mlir("prod", "prod", 23)
    assert "len = 23" in text and f"len = {g.N_WORDS}" not in text


def test_port_cases_parse_like_the_others():
    assert g.case_spec("mem_own_ports_n40") == ("mem", "mem", 40)
    assert g.case_spec("prod_own_ports") == ("prod", "prod", g.N_WORDS)
    assert g.case_spec("core_own_ports_n20") == ("core", "prod", 20)
    assert g.is_ports_case("core_own_ports_n20") and not g.is_ports_case("mem_own_n20")


def test_a_core_source_is_n_straight_line_puts_gated_after_the_trace_starts():
    text = g.design_mlir("core", "prod", 12)
    assert text.count("aie.put_stream") == 12
    assert "aie.flow(%t_prod, Core : 0, %t_prod, DMA : 0)" in text
    assert "aie.core(%t_prod)" in text
    # the core waits for a lock the runtime sequence sets after the trace is armed
    assert "AcquireGreaterEqual" in text.split("aie.core(%t_prod)")[1].split("aie.end")[0]
    assert "MM2S" not in text.split("aie.runtime_sequence")[1]
    assert "len = 12" in text
    # distinct words
    assert len({int(x) for x in re.findall(r"arith.constant (\d+) : i32\n\s+aie.put_stream", text)}) == 12


def test_dma_sources_have_no_core():
    assert "aie.core" not in g.design_mlir("prod", "prod", 12)


def test_inject_args_watch_both_dma_ports_on_a_memtile():
    a = g.inject_args("mem_own_ports_n40")
    ev = a[a.index("--memtile-sweep-events") + 1].split(",")
    assert ev == ["DMA_MM2S_SEL0_START_TASK", "DMA_MM2S_SEL0_STREAM_BACKPRESSURE",
                  "DMA_S2MM_SEL0_STALLED_LOCK", "PORT_RUNNING_0", "PORT_STALLED_0",
                  "PORT_RUNNING_4", "PORT_STALLED_4"]
    assert "--port-slots" not in a


def test_inject_args_prod_keeps_the_memory_module_trace_and_adds_the_core_ports():
    a = g.inject_args("prod_own_ports_n40")
    assert a[a.index("--memmod-sweep-events") + 1] == g.MEMMOD_EVENTS
    assert a[a.index("--core-sweep-events") + 1] == "PORT_RUNNING_0,PORT_STALLED_0,PORT_RUNNING_1,PORT_STALLED_1"
    assert a[a.index("--core-grounding") + 1] == ""


def test_inject_args_core_source_watches_the_core_port_in_slot_1():
    a = g.inject_args("core_own_ports_n40")
    assert a[a.index("--port-slots") + 1] == "1=Core:0:slave"
    assert "--memmod-sweep-events" not in a


def test_existing_cases_keep_their_injection():
    a = g.inject_args("prod_own_n23")
    assert "--core-sweep-events" not in a and a[a.index("--memmod-sweep-events") + 1] == g.MEMMOD_EVENTS


def _frames(running_runs, stalled_runs, end):
    by = {}
    for name, runs in (("PORT_RUNNING_0", running_runs), ("PORT_STALLED_0", stalled_runs)):
        for s, e in runs:
            for t in range(s, e):
                by.setdefault(t, set()).add(name)
    by.setdefault(end, set())
    return [(t, frozenset(n)) for t, n in sorted(by.items())]


def test_port_report_counts_running_cycles_before_the_sustained_stall():
    # 3 words, a one-cycle stall blip, 2 words, then the sustained stall to the end
    frames = _frames([(10, 13), (14, 16)], [(13, 14), (16, 90)], 90)
    r = g.port_report(frames, "PORT_RUNNING_0", "PORT_STALLED_0")
    assert r["stall_onset"] == 16
    assert r["running_before_stall"] == 5
    assert r["running"] == [(10, 13), (14, 16)]
    assert r["stalled"] == [(13, 14), (16, 90)]


def test_port_report_without_a_stall_counts_everything():
    frames = _frames([(10, 30)], [], 30)
    r = g.port_report(frames, "PORT_RUNNING_0", "PORT_STALLED_0")
    assert r["stall_onset"] is None and r["running_before_stall"] == 20


def test_port_report_of_an_event_that_never_fires():
    r = g.port_report(_frames([], [], 5), "PORT_RUNNING_0", "PORT_STALLED_0")
    assert r["running_before_stall"] == 0 and r["running"] == []
