import pytest
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


def test_a_chain_suffix_is_k_chained_bds_of_l_words_in_one_task():
    assert g.case_shape("prod_own_c4x3") == g.Shape("prod", "prod", 4, 3, "chain")
    assert g.case_spec("prod_own_c4x3") == ("prod", "prod", 12)
    text = g.design_mlir(*g.case_shape("prod_own_c4x3").design_args())
    assert text.count("MM2S, 0)") == 1
    assert text.count("aie.next_bd") == 2
    for j in range(3):
        assert f"offset = {4 * j} len = 4" in text
    assert "len = 12" in text  # the S2MM takes the whole stream in one BD


def test_a_task_suffix_is_k_back_to_back_tasks_of_l_words():
    s = g.case_shape("mem_own_t8x2")
    assert s == g.Shape("mem", "mem", 8, 2, "tasks")
    text = g.design_mlir(*s.design_args())
    assert text.count("MM2S, 0)") == 2 and "aie.next_bd" not in text
    assert text.count("aiex.dma_start_task(%ts") == 2
    assert "offset = 8 len = 8" in text and "len = 16" in text


def test_a_plain_or_length_name_is_one_bd_one_task():
    assert g.case_shape("prod_own_n23") == g.Shape("prod", "prod", 23, 1, "single")
    assert g.case_shape("prod_own") == g.Shape("prod", "prod", g.N_WORDS, 1, "single")
    # unchanged text for single-BD cases
    assert g.design_mlir(*g.case_shape("prod_own_n23").design_args()) == g.design_mlir("prod", "prod", 23)


@pytest.mark.parametrize("name", ["prod_own", "prod_own_n23", "prod_own_c4x3", "mem_own_t8x2",
                                  "core_own_ports_n20"])
def test_every_design_has_balanced_braces(name):
    text = g.design_mlir(*g.case_shape(name).design_args())
    assert text.count("{") == text.count("}")


def _clock_ops(text):
    return [l.strip() for l in text.splitlines() if "maskwrite32" in l]


def test_k0_adds_nothing_to_a_design():
    assert g.clock_writes("K0", "prod") == [] and g.clock_writes("K0", "mem") == []
    assert _clock_ops(g.design_mlir("prod", "prod", 16)) == []
    assert g.design_mlir("prod", "prod", 16, clock="K0") == g.design_mlir("prod", "prod", 16)


@pytest.mark.skipif(g.toolchain_paths.try_resolve_mlir_aie(g.REPO) is None, reason="no mlir-aie checkout")
def test_clock_configs_clear_the_adaptive_enables_from_the_register_database():
    # Module_Clock_Control: compute tile 0x60000, memtile 0xFFF00; bit 4 stream switch, bit 5 DMA
    assert g.clock_writes("K1", "prod") == [(0x60000, 0, 0x10)]
    assert g.clock_writes("K2", "mem") == [(0xFFF00, 0, 0x20)]
    assert g.clock_writes("K3", "prod") == [(0x60000, 0, 0x30)]
    assert g.clock_writes("K3", "mem") == [(0xFFF00, 0, 0x30)]


@pytest.mark.skipif(g.toolchain_paths.try_resolve_mlir_aie(g.REPO) is None, reason="no mlir-aie checkout")
def test_k4_sets_the_stream_switch_abort_period_to_the_database_minimum():
    # Abort_Period[3:0], supported range [3-12]: compute 0x3FF38, memtile 0xB0F38
    assert g.clock_writes("K4", "prod") == [(0x3FF38, 3, 0xF)]
    assert g.clock_writes("K4", "mem") == [(0xB0F38, 3, 0xF)]


@pytest.mark.skipif(g.toolchain_paths.try_resolve_mlir_aie(g.REPO) is None, reason="no mlir-aie checkout")
def test_clock_writes_target_the_data_tiles_and_precede_the_dmas():
    text = g.design_mlir("prod", "prod", 16, clock="K1")
    ops = _clock_ops(text)
    assert len(ops) == 1 and "column = 0 : i32, row = 2 : i32" in ops[0]
    assert "arith.constant 393216 : i32" in text           # 0x60000
    assert text.index("maskwrite32") < text.index("aiex.dma_start_task")
    # a memtile-sourced case into the memtile touches only the memtile
    mem = _clock_ops(g.design_mlir("mem", "mem", 16, clock="K3"))
    assert len(mem) == 1 and "row = 1 : i32" in mem[0]
    # mem -> prod touches both tiles, one write each
    both = _clock_ops(g.design_mlir("mem", "prod", 16, clock="K2"))
    assert len(both) == 2 and any("row = 1" in o for o in both) and any("row = 2" in o for o in both)


def test_an_unknown_clock_config_is_refused():
    with pytest.raises(KeyError):
        g.design_mlir("prod", "prod", 16, clock="K9")


needs_mlir_aie = pytest.mark.skipif(g.toolchain_paths.try_resolve_mlir_aie(g.REPO) is None,
                                    reason="no mlir-aie checkout")


@needs_mlir_aie
def test_proof_steps_order_control_reads_around_the_clock_reads():
    import ctrlpkt_readback as cp
    steps = g.clock_proof_steps("K3", "prod")
    reads = [s for s in steps if isinstance(s, cp.Read)]
    clock = list(g.PROOF_READS)
    assert [r.label for r in reads] == (
        ["pre control Lock0_value"] + [f"pre {n}" for n in clock] + [f"post {n}" for n in clock]
        + ["post control Lock0_value", "handler Tile_Control_Packet_Handler_Status"])
    assert [r.addr for r in reads] == [0x1F000, 0x60000, 0x3FF38, 0x3FF34, 0x60000, 0x3FF38, 0x3FF34,
                                       0x1F000, 0x3FF30]
    raw = [i for i, s in enumerate(steps) if isinstance(s, cp.Raw)]
    assert raw == [4]                                       # after the 4 pre reads, before the post reads
    assert any("maskwrite32" in line and "row = 2" in line for line in steps[4].lines)


@needs_mlir_aie
def test_proof_memtile_reads_its_own_registers():
    import ctrlpkt_readback as cp
    reads = [s for s in g.clock_proof_steps("K1", "mem") if isinstance(s, cp.Read)]
    assert reads[0].addr == 0xC0000 and reads[1].addr == 0xFFF00 and reads[-1].addr == 0xB0F30
    assert all(r.tile == "mem" for r in reads)


@needs_mlir_aie
def test_proof_k0_has_no_write_and_k4_writes_the_abort_period():
    import ctrlpkt_readback as cp
    assert not any(isinstance(s, cp.Raw) for s in g.clock_proof_steps("K0", "prod"))
    raw = next(s for s in g.clock_proof_steps("K4", "mem") if isinstance(s, cp.Raw))
    assert f"arith.constant {0xB0F38} : i32" in raw.lines[0]
    assert any("row = 1" in line for line in raw.lines)


@needs_mlir_aie
def test_proof_design_does_not_wait_on_the_responses():
    text, steps, layout = g.clock_proof_design("K1", "prod")
    body = text.split("aie.runtime_sequence")[1]
    assert "aiex.npu.dma_wait {symbol = @ctrl0}" not in body       # the response channel is never awaited
    resp = next(l for l in body.splitlines() if "metadata = @ctrl0" in l)
    assert "issue_token = true" not in resp
    assert body.count("aiex.npu.dma_wait {symbol = @ctrlin0}") == len(layout.labels)   # packets leave the shim
    assert f"memref<{len(layout.labels)}xi32>" in text


@needs_mlir_aie
def test_proof_design_routes_control_packets_and_seeds_the_control_lock():
    text, _, _ = g.clock_proof_design("K1", "prod")
    assert "aie.packet_dest<%t_core, TileControl : 0>" in text and "pkt_id = 5" in text
    assert f"aie.lock(%t_core, 0) {{init = {g.PROOF_LOCK_VALUE}" in text
    mem, _, _ = g.clock_proof_design("K2", "mem")
    assert "aie.packet_dest<%t_mem, TileControl : 0>" in mem and "pkt_id = 6" in mem
    assert f"aie.lock(%t_mem, 0) {{init = {g.PROOF_LOCK_VALUE}" in mem


@needs_mlir_aie
def test_proof_write_comes_between_the_read_packets():
    text, _, _ = g.clock_proof_design("K1", "prod")
    body = text.split("aie.runtime_sequence")[1]
    assert body.index("dma_memcpy_nd(%ctrlin[0, 0, 0, 0]") < body.index("maskwrite32") < body.rindex("dma_memcpy_nd(%ctrlin")


@needs_mlir_aie
def test_proof_expectation_names_the_written_values():
    want = g.proof_expectation("K3", "prod")
    assert want["pre Module_Clock_Control"] == 0x37 and want["post Module_Clock_Control"] == 0x07
    assert want["pre Stream_Switch_Adaptive_Clock_Gate_Abort_Period"] == 7
    assert g.proof_expectation("K4", "mem")["post Stream_Switch_Adaptive_Clock_Gate_Abort_Period"] == 3


def _words(config="K3", tile="prod", **over):
    want = g.proof_expectation(config, tile)
    w = {"pre control Lock0_value": g.PROOF_LOCK_VALUE, "post control Lock0_value": g.PROOF_LOCK_VALUE,
         "pre Stream_Switch_Adaptive_Clock_Gate_Status": 0, "post Stream_Switch_Adaptive_Clock_Gate_Status": 0,
         "handler Tile_Control_Packet_Handler_Status": 0, **want}
    w.update(over)
    return w


@needs_mlir_aie
def test_verdicts():
    assert g.proof_verdict("K3", "prod", _words()) == "LANDED"
    assert g.proof_verdict("K3", "prod", _words(**{"post Module_Clock_Control": 0x37})) == "DROPPED"
    nr = _words(**{"pre Module_Clock_Control": None, "post Module_Clock_Control": None})
    assert g.proof_verdict("K3", "prod", nr) == "NO-REPLY"
    assert g.proof_verdict("K3", "prod", _words(**{"post control Lock0_value": None})) == "BROKEN"
    assert g.proof_verdict("K3", "prod", _words(**{"pre control Lock0_value": None})) == "BROKEN"
    assert g.proof_verdict("K3", "prod", _words(**{"post Module_Clock_Control": 0x11})) == "UNEXPECTED"
    assert g.proof_verdict("K0", "prod", _words("K0")) == "NO-WRITE"


@needs_mlir_aie
def test_decode_assigns_replies_in_order_and_marks_missing_ones():
    import ctrlpkt_readback as cp
    labels = list(cp.encode(g.clock_proof_steps("K3", "prod")).labels)
    EE, L = 0xEEEEEEEE, g.PROOF_LOCK_VALUE
    # every reply present
    full = [L, 0x37, 7, 0, 0x07, 7, 0, L, 0]
    assert list(g.proof_decode(labels, full).values()) == full
    # the clock registers never answer: the later control reply must not slide into their slots
    silent = [L, L, 0, EE, EE, EE, EE, EE, EE]
    d = g.proof_decode(labels, silent)
    assert d["pre control Lock0_value"] == L and d["post control Lock0_value"] == L
    assert d["pre Module_Clock_Control"] is None and d["post Module_Clock_Control"] is None
    assert d["handler Tile_Control_Packet_Handler_Status"] == 0
    # nothing at all answers
    assert all(v is None for v in g.proof_decode(labels, [EE] * 9).values())


# -- prewake probe: does a word crossing the switch shortly before a transfer hide the gate's blip? --

from collections import namedtuple

Ev = namedtuple("Ev", "name ts row pkt_type")


def bundles(text):
    lines = [ln.strip() for ln in text.splitlines()]
    start = max(i for i, ln in enumerate(lines) if ln.startswith("st r31,")) + 1
    end = min(i for i, ln in enumerate(lines) if ln.startswith("lda r16,"))
    return [ln for ln in lines[start:end] if ln and not ln.startswith((".", "#")) and not ln.endswith(":")]


def test_prewake_case_names_round_trip():
    assert g.prewake_case_name(None) == "pw_ctl" and g.prewake_case_name(126) == "pw_d126"
    assert g.prewake_case_d("pw_ctl") is None and g.prewake_case_d("pw_d020") == 20
    assert g.PREWAKE_DS[:12] == (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 64)   # the short cases first
    assert g.PREWAKE_DS[12:] == (20, 50, 100, 120, 126, 127, 128, 129, 130, 140, 200)


@pytest.mark.parametrize("d", [0, 20, 128, 200])
def test_prewake_core_spaces_the_two_releases_by_d_nops(d):
    body = bundles(g.prewake_core_asm(d))
    rels = [i for i, b in enumerate(body) if b.startswith("rel ")]
    assert body[rels[0]] == f"rel #{g.OWN_LOCKS + g.PW_LOCK_DUMMY}, r2"
    assert body[rels[1]] == f"rel #{g.OWN_LOCKS + g.PW_LOCK_MEAS}, r2"
    assert body[rels[0] + 1:rels[1]] == ["nop"] * d
    # the filler releases come after the measured release, on the dummy lock only
    assert len(rels) == 2 + g.PW_FILLERS
    assert all(body[i] == f"rel #{g.OWN_LOCKS + g.PW_LOCK_DUMMY}, r2" for i in rels[2:])
    assert rels[2] - rels[1] > g.PW_AFTER_MEASURED_NOPS - 1


def test_prewake_control_core_has_no_dummy_release_before_the_measured_one():
    body = bundles(g.prewake_core_asm(None))
    rels = [b for b in body if b.startswith("rel ")]
    assert rels[0] == f"rel #{g.OWN_LOCKS + g.PW_LOCK_MEAS}, r2"
    assert len(rels) == 1 + g.PW_FILLERS


def test_prewake_core_waits_only_on_the_go_lock():
    body = bundles(g.prewake_core_asm(50))
    acqs = [b for b in body if b.startswith("acq ")]
    assert acqs == [f"acq #{g.OWN_LOCKS + g.PW_LOCK_GO}, r3"]
    assert body.index(acqs[0]) < body.index(next(b for b in body if b.startswith("rel ")))


def test_prewake_design_is_two_loopbacks_and_a_blind_sequence():
    t = g.prewake_design_mlir()
    assert "aie.flow(%t_prod, DMA : 0, %t_prod, DMA : 0)" in t
    assert "aie.flow(%t_prod, DMA : 1, %t_prod, DMA : 1)" in t
    assert f"aie.dma_bd(%src : memref<{g.PW_WORDS}xi32>)" in t and g.PW_WORDS == 16
    send0 = t.split("^send0:")[1].split("^start1:")[0]
    assert "use_lock(%lM, AcquireGreaterEqual" in send0 and "next_bd" in send0
    send1 = t.split("^send1:")[1].split("^start2:")[0]
    assert "use_lock(%lDm, AcquireGreaterEqual" in send1 and "aie.next_bd ^send1" in send1   # a repeating 1-word BD
    for recv, nxt in (("^recv0:", "^start3:"), ("^recv1:", "^end:")):
        assert "use_lock" not in t.split(recv)[1].split(nxt)[0]          # the S2MMs wait on nothing
    seq = t.split("aie.runtime_sequence")[1]
    assert "aiex.set_lock(%lGo" in seq
    for forbidden in ("dma_await", "dma_wait", "dma_memcpy_nd", "dma_start_task", "npu.sync",
                      "blockwrite"):
        assert forbidden not in seq.lower()
    assert "aie.shim_dma_allocation" not in t


def test_prewake_traces_only_the_memory_module_and_few_events():
    a = g.prewake_inject_args()
    assert a[a.index("--memmod-sweep-events") + 1].split(",") == g.PW_EVENTS
    assert len(g.PW_EVENTS) <= 6
    assert a[a.index("--core-grounding") + 1] == "INSTR_EVENT_1"   # an event the core never raises
    assert "event #1" not in g.prewake_core_asm(50)              # so no core packet ever leaves
    assert a[a.index("--core-sweep-events") + 1] == ""


def ev(name, *ts):
    return [Ev(name, t, 2, 1) for t in ts]


def window(blip_at=None, d=50, latency=19, fillers=0):
    """Events of one run: MM2S 1 start/finish at 1000, MM2S 0 start at 1000 + d."""
    s0 = 1000 + d
    out = ev("DMA_MM2S_1_STALLED_LOCK", *range(900, 1000)) + ev("DMA_MM2S_1_FINISHED_BD", 1006)
    out += ev("DMA_MM2S_0_STALLED_LOCK", *range(900, s0))
    out += ev("DMA_S2MM_0_STREAM_STARVATION", *range(900, s0 + latency))
    if blip_at is not None:
        out += ev("DMA_MM2S_0_STREAM_BACKPRESSURE", s0 + blip_at)
    return out


def test_prewake_row_reads_d_blip_and_latency():
    r = g.prewake_row(window(blip_at=15, d=50, latency=19))
    assert (r["d_start"], r["d_fin"], r["blip"], r["blip_at"], r["latency"]) == (50, 44, True, 15, 19)
    r = g.prewake_row(window(blip_at=None, d=20, latency=18))
    assert (r["d_start"], r["blip"], r["blip_at"], r["latency"]) == (20, False, None, 18)


def test_prewake_row_ignores_backpressure_outside_the_onset_window():
    r = g.prewake_row(window(blip_at=g.PW_BLIP_WINDOW + 5))
    assert r["blip"] is False


def test_prewake_row_without_a_dummy_is_the_control():
    evs = [e for e in window(blip_at=15) if "MM2S_1" not in e.name]
    r = g.prewake_row(evs)
    assert r["d_start"] is None and r["blip"] is True


def test_prewake_row_flags_a_possible_trace_packet_in_the_window():
    quiet = g.prewake_row(window(blip_at=15))
    assert quiet["frames"] < g.PW_FRAME_GUARD and not quiet["packet_risk"]
    noisy = window(blip_at=15) + ev("DMA_MM2S_0_STREAM_BACKPRESSURE", *range(1060, 1060 + 40, 2))
    assert g.prewake_row(noisy)["packet_risk"]


def test_prewake_row_with_no_events_is_empty():
    assert g.prewake_row([])["start0"] is None
    assert g.prewake_row(ev("DMA_MM2S_0_STALLED_LOCK", *range(5)))["start0"] == 5


def test_prewake_threshold_brackets_the_flip_and_rejects_a_non_monotone_set():
    rows = [{"d_start": d, "blip": d >= 128} for d in (20, 100, 126, 128, 130, 200)]
    assert g.prewake_threshold(rows) == (126, 128)
    rows[1]["blip"] = True
    assert g.prewake_threshold(rows) is None
    assert g.prewake_threshold([{"d_start": 20, "blip": False}]) == (20, None)


def test_prewake_table_prints_one_line_per_case_and_the_bracket():
    rows = [("pw_ctl", "e1.emu", g.prewake_row(window(blip_at=15))),
            ("pw_d020", "e1.emu", g.prewake_row(window(blip_at=None, d=20, latency=18)))]
    text = g.format_prewake_table(rows)
    assert text.count("\n") >= 3 and "pw_ctl" in text and "pw_d020" in text and "blip" in text


def test_prewake_row_flags_any_core_module_event_as_a_packet_risk():
    evs = window(blip_at=15) + [Ev("LOCK_STALL", 1010, 2, 0)]
    r = g.prewake_row(evs)
    assert r["core_events"] == 1 and r["packet_risk"]


def test_prewake_control_ignores_the_fillers_that_start_mm2s_1_later():
    evs = [e for e in window(blip_at=15) if "MM2S_1" not in e.name]
    evs += ev("DMA_MM2S_1_STALLED_LOCK", *range(900, 1400)) + ev("DMA_MM2S_1_FINISHED_BD", 1406)
    r = g.prewake_row(evs)
    assert r["d_start"] is None and r["d_fin"] is None


def test_prewake_threshold_works_on_the_late_first_word_too():
    rows = [{"d_start": d, "late": d >= 130} for d in (20, 128, 130, 200)]
    assert g.prewake_threshold(rows, "late") == (128, 130)
    text = g.format_prewake_table([("pw_d020", "r", g.prewake_row(window(None, 20, 19))),
                                   ("pw_d200", "r", g.prewake_row(window(None, 200, 20)))])
    assert "first word one cycle late flips between D(start) = 20 (absent) and 200 (present)" in text


def _rows(run, spec):
    """(case, run, row) per (d, blip): a window whose MM2S 0 starts d after MM2S 1."""
    return [(f"pw_d{d:03d}", run, g.prewake_row(window(blip_at=15 if blip else None, d=d, latency=19)))
            for d, blip in spec]


def test_a_case_a_run_did_not_capture_is_listed_once_not_as_a_row():
    rows = _rows("h1.hw", [(20, False), (200, True)])
    rows += [("pw_d050", "h1.hw", {"start0": None}), ("pw_d100", "h1.hw", {"start0": None})]
    text = g.format_prewake_table(rows)
    assert "no MM2S 0 grant" not in text.split("not in this run")[0]
    assert text.count("not in this run") == 1
    assert "pw_d050" in text and "pw_d100" in text
    assert text.count("pw_d050") == 1 and text.count("pw_d100") == 1


def test_the_flip_is_pooled_over_the_runs_of_one_platform_so_a_partial_run_is_not_non_monotone():
    # h1 ran only the low and high D, h2 the middle ones: each alone is fine, pooled is still monotone
    rows = (_rows("h1.hw", [(20, False), (200, True)]) + _rows("h2.hw", [(100, False), (126, False), (128, True)])
            + _rows("e3.emu", [(20, False), (140, True)]))
    text = g.format_prewake_table(rows)
    assert "NOT MONOTONE" not in text
    assert "hw: the BACKPRESSURE blip flips between D(start) = 126 (absent) and 128 (present)" in text
    assert "emu: the BACKPRESSURE blip flips between D(start) = 20 (absent) and 140 (present)" in text


def test_interleaved_runs_of_one_platform_say_overlap_and_name_the_rows():
    rows = (_rows("h1.hw", [(20, False), (128, True), (200, True)])
            + _rows("h2.hw", [(128, False), (130, True)]))
    text = g.format_prewake_table(rows)
    line = next(ln for ln in text.splitlines() if ln.startswith("hw: the BACKPRESSURE"))
    assert "overlap" in line and "NOT MONOTONE" not in line
    assert "h1.hw pw_d128" in line and "h2.hw pw_d128" in line


def test_flip_per_platform_uses_a_late_threshold_from_that_platforms_own_fastest_first_word():
    rows = (_rows("h1.hw", [(20, False)]) + _rows("h2.hw", [(200, False)])
            + [("pw_d020", "e3.emu", g.prewake_row(window(None, 20, 30))),
               ("pw_d200", "e3.emu", g.prewake_row(window(None, 200, 30)))])
    text = g.format_prewake_table(rows)
    # the emulator's 30-cycle first words are its own minimum, so nothing there is "late"
    line = next(ln for ln in text.splitlines() if ln.startswith("emu: a first word"))
    assert "200 (absent), none present" in line
