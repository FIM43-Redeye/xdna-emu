"""Pure-logic tests for tools/gen_packet_stage_probe.py: the case list and hardware order, the core program's
release order and delays, the generated design, the route checks on canned switchbox text and the table and
prediction logic on synthetic events. No compiler, runner or device."""
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_packet_stage_probe as g  # noqa: E402

Ev = namedtuple("Ev", "ts row pkt_type name")


# -- cases -----------------------------------------------------------------------------------------

def test_eighteen_cases_with_the_expected_names():
    names = [c.name for c in g.default_cases()]
    assert len(names) == len(set(names)) == 18
    assert names == ["t008", "t064", "t112", "t124", "t132", "t144", "t300", "t1000", "hdrop", "hkeep",
                     "wp016", "wp120", "wp136", "wp1000", "wc016", "wc136", "wc1000", "c"]


def test_case_from_name_round_trips_and_rejects_unknown():
    for c in g.default_cases():
        assert g.case_from_name(c.name) == c
    with pytest.raises(ValueError):
        g.case_from_name("t999")


def test_session_runs_controls_first_and_each_repeat_twice_in_a_row():
    order = g.session_order()
    assert len(order) == 22
    assert order[:6] == ["hdrop", "c", "t112", "t112", "t144", "t144"]
    for name in g.REPEATED:
        i = order.index(name)
        assert order[i + 1] == name and order.count(name) == 2
    assert sorted(set(order)) == sorted(c.name for c in g.default_cases())


def test_destination_words_match_what_reaches_each_sink():
    words = {c.name: c.dst_words for c in g.default_cases()}
    assert words["t112"] == 32 and words["hdrop"] == 16 and words["hkeep"] == 17
    assert words["wp016"] == words["wc016"] == words["c"] == 16


def test_release_order_per_family():
    assert g.case_from_name("t112").order == ["A", "B"]
    assert g.case_from_name("wp136").order == ["W", "A"]
    assert g.case_from_name("hdrop").order == ["A"]


def test_circuit_streams_have_no_packet_id():
    assert g.case_from_name("c").stream("A").pkt_id is None
    assert g.case_from_name("wc136").stream("W").pkt_id is None
    assert g.case_from_name("wp136").stream("W").pkt_id == g.PKT_W
    assert g.case_from_name("wc136").stream("A").pkt_id == g.PKT_A


# -- the core program -----------------------------------------------------------------------------------

def test_release_gap_uses_nops_when_short():
    gap = g.release_gap(28)
    assert gap == ["  nop"] * 27


def test_release_gap_loop_cycles_add_up_to_the_spacing():
    for spacing in (41, 132, 164, 320, 1020):
        gap = g.release_gap(spacing)
        loop = next(ln for ln in gap if ln.startswith("  movxm"))
        iters = int(loop.split("#")[1])
        remainder = (spacing - 1 - g.LOOP_BASE) % g.LOOP_ITER
        assert iters == (spacing - 1 - g.LOOP_BASE) // g.LOOP_ITER
        assert iters * g.LOOP_ITER + g.LOOP_BASE + remainder + 1 == spacing
        assert gap[-remainder:] == ["  nop"] * remainder if remainder else True


def test_release_gap_rejects_a_zero_spacing():
    with pytest.raises(ValueError):
        g.release_gap(0)


def _rels(asm: str) -> list[int]:
    return [int(ln.split("#")[1].split(",")[0]) for ln in asm.splitlines() if ln.startswith("  rel ")]


def test_core_releases_the_first_stream_then_the_second():
    assert _rels(g.core_asm(g.case_from_name("t112")))[:2] == [g.OWN + g.LOCK_A, g.OWN + g.LOCK_X1]
    assert _rels(g.core_asm(g.case_from_name("wp136")))[:2] == [g.OWN + g.LOCK_X1, g.OWN + g.LOCK_A]
    assert _rels(g.core_asm(g.case_from_name("hdrop")))[:2][0] == g.OWN + g.LOCK_A
    assert len([r for r in _rels(g.core_asm(g.case_from_name("hdrop")))]) == 2     # the A release and the flush loop's


def test_core_waits_for_go_first_and_aligns_before_the_first_release():
    lines = g.core_asm(g.case_from_name("t112")).splitlines()
    assert lines.index(f"  acq #{g.OWN + g.LOCK_GO}, r3") < lines.index("  event #0")
    first_rel = next(i for i, ln in enumerate(lines) if ln.startswith("  rel "))
    assert lines[first_rel - 1] == ".p2align 4"


def test_the_flush_loop_follows_the_streams_and_never_acquires():
    asm = g.core_asm(g.case_from_name("t112"))
    assert asm.index(".Lflush:") > asm.index(f"  rel #{g.OWN + g.LOCK_X1}, r2")
    assert "acq #" in asm and asm.count("  acq #") == 1          # only the go lock
    assert g.FLUSH_PULSES + 1 < 63                                 # the lock never overflows


# -- the design ---------------------------------------------------------------------------------------------

def test_t_design_has_two_packet_flows_into_one_destination_channel():
    mlir = g.design_mlir(g.case_from_name("t112"))
    assert mlir.count("aie.packet_flow(") == 2
    assert "aie.packet_source<%t02, DMA : 0>" in mlir and "aie.packet_source<%t02, DMA : 1>" in mlir
    assert mlir.count("aie.packet_dest<%t03, DMA : 0>") == 2
    assert "keep_pkt_header" not in mlir
    assert "memref<32xi32>" in mlir and "memtile_dma" not in mlir


def test_only_hkeep_keeps_the_header_and_receives_seventeen_words():
    keep, drop = g.design_mlir(g.case_from_name("hkeep")), g.design_mlir(g.case_from_name("hdrop"))
    assert "{keep_pkt_header = true}" in keep and "keep_pkt_header" not in drop
    assert "%dst = aie.buffer(%t03) {address = 32768 : i32, sym_name = \"dst\"} : memref<17xi32>" in keep
    assert "memref<16xi32>" in drop


def test_circuit_control_has_a_flow_and_no_packet_info():
    mlir = g.design_mlir(g.case_from_name("c"))
    assert "aie.flow(%t02, DMA : 0, %t03, DMA : 0)" in mlir
    assert "packet_flow" not in mlir and "packet_info" not in mlir


def test_w_packet_goes_to_the_memtile_and_w_circuit_is_a_flow():
    pkt, circ = g.design_mlir(g.case_from_name("wp016")), g.design_mlir(g.case_from_name("wc016"))
    assert "aie.packet_dest<%t01, DMA : 0>" in pkt and "aie.memtile_dma(%t01)" in pkt
    assert "aie.flow(%t02, DMA : 1, %t01, DMA : 0)" in circ and "pkt_id = 3" not in circ
    assert "pkt_id = 1" in circ


def test_every_stream_lock_starts_at_zero_and_has_no_waiter_beyond_its_dma():
    for c in g.default_cases():
        mlir = g.design_mlir(c)
        assert mlir.count("init = 0 : i32") == mlir.count("aie.lock(")
        assert mlir.count("AcquireGreaterEqual") == len(c.streams)         # one acquire per started stream
        assert "aiex.set_lock(%lGo" in mlir and "dma_wait" not in mlir     # nothing waits on the array


def test_port_slots_and_events():
    t, w = g.case_from_name("t112"), g.case_from_name("wp016")
    assert g.port_slots(t, ("North", 5), None) == "0=North:5:master,1=DMA:0:slave,2=DMA:1:slave"
    assert g.port_slots(w, ("North", 5), ("South", 0)).endswith(",3=South:0:master")
    assert "PORT_RUNNING_3" in g.core_events(w) and "PORT_RUNNING_3" not in g.core_events(t)
    assert "--trace-tiles" in g.inject_args(t, ("North", 5), None)
    assert len(g.MEMMOD_EVENTS) <= 8 and len(g.core_events(w)) + 2 <= 8


# -- route checks on canned switchboxes ----------------------------------------------------------------------

def _box(tile: str, body: str) -> str:
    return f"    %sb = aie.switchbox(%{tile}) {{\n{body}\n    }}\n"


SRC_T = _box("tile_0_2", """      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 3, %1)
      %3 = aie.masterset(North : 5, %0)
      aie.packet_rules(Trace : 1) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }""")
DST = _box("tile_0_3", """      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 5) {
        aie.rule(28, 0, %0)
      }""")
DST_KEEP = DST.replace("%1 = aie.masterset(DMA : 0, %0)", "%1 = aie.masterset(DMA : 0, %0) {keep_pkt_header = true}")
SRC_H = _box("tile_0_2", """      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 3, %1)
      %3 = aie.masterset(North : 5, %0)
      aie.packet_rules(Trace : 1) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }""")
SRC_WP = _box("tile_0_2", """      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(South : 0, %1)
      %4 = aie.masterset(South : 2, %2)
      %5 = aie.masterset(North : 5, %0)
      aie.packet_rules(Trace : 1) {
        aie.rule(31, 2, %2)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 1, %2)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 3, %1)
      }""")
MEM_WP = _box("mem_tile_0_1", """      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(DMA : 0, %0)
      %3 = aie.masterset(South : 2, %1)
      aie.packet_rules(North : 2) {
        aie.rule(28, 0, %1)
      }
      aie.packet_rules(North : 0) {
        aie.rule(31, 3, %0)
      }""")
SRC_WC = _box("tile_0_2", """      aie.connect<DMA : 1, South : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 3, %1)
      %3 = aie.masterset(North : 5, %0)
      aie.packet_rules(Trace : 1) {
        aie.rule(31, 2, %1)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 1, %1)
      }
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 1, %0)
      }""")
MEM_WC = _box("mem_tile_0_1", """      aie.connect<North : 0, DMA : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(28, 0, %0)
      }""")
SRC_C = _box("tile_0_2", """      aie.connect<DMA : 0, North : 5>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 3, %0)
      aie.packet_rules(Trace : 1) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 1, %0)
      }""")
DST_C = _box("tile_0_3", "      aie.connect<South : 5, DMA : 0>")


def test_switchbox_parser_reads_masters_rules_and_keep():
    box = g.switchbox(SRC_T + DST_KEEP, (0, 2))
    assert box.route(("DMA", 0), 1) == [(("North", 5), 0, False)]
    assert box.route(("DMA", 1), 2) == [(("North", 5), 0, False)]
    assert box.route(("DMA", 0), 2) == []
    dst = g.switchbox(SRC_T + DST_KEEP, (0, 3))
    assert dst.route(("South", 5), 1) == [(("DMA", 0), 0, True)]
    assert dst.route(("South", 5), 2) == [(("DMA", 0), 0, True)]         # mask 28 covers ids 0-3
    assert dst.route(("South", 5), 4) == []


def test_trace_ports_are_the_trace_rules_masters_and_arbiters():
    masters, arbs = g.trace_ports(g.switchbox(SRC_T, (0, 2)))
    assert masters == {("South", 3)} and arbs == {1}


def test_t_routes_pass_and_name_the_shared_master():
    problems, routes = g.check_routes(SRC_T + DST, g.case_from_name("t112"))
    assert problems == [] and routes == g.Routes(("North", 5), None)


def test_t_fails_when_the_two_flows_use_two_arbiters():
    split = SRC_T.replace("aie.rule(31, 2, %0)", "aie.rule(31, 2, %1)")
    problems, routes = g.check_routes(split + DST, g.case_from_name("t112"))
    assert routes is None and any("trace rule" in p or "arbiter" in p for p in problems)


def test_a_flow_the_router_made_a_circuit_is_refused():
    problems, _ = g.check_routes(SRC_C + DST, g.case_from_name("hdrop"))
    assert any("no packet rule" in p for p in problems)


def test_hkeep_needs_the_kept_flag_and_hdrop_the_absence_of_it():
    assert g.check_routes(SRC_H + DST_KEEP, g.case_from_name("hkeep"))[0] == []
    assert any("keep_pkt_header" in p for p in g.check_routes(SRC_H + DST, g.case_from_name("hkeep"))[0])
    assert any("keep_pkt_header" in p for p in g.check_routes(SRC_H + DST_KEEP, g.case_from_name("hdrop"))[0])
    assert g.check_routes(SRC_H + DST, g.case_from_name("hdrop"))[0] == []


def test_w_packet_routes_pass_with_its_own_master_and_arbiter():
    problems, routes = g.check_routes(SRC_WP + DST + MEM_WP, g.case_from_name("wp016"))
    assert problems == [] and routes == g.Routes(("North", 5), ("South", 0))


def test_w_packet_fails_when_it_shares_the_trace_arbiter_or_the_master():
    shared = SRC_WP.replace("aie.rule(31, 3, %1)", "aie.rule(31, 3, %2)")
    problems, _ = g.check_routes(shared + DST + MEM_WP, g.case_from_name("wp016"))
    assert any("trace" in p for p in problems)


def test_w_packet_fails_on_a_trace_id_collision_at_the_memtile():
    clash = MEM_WP.replace("aie.rule(31, 3, %0)", "aie.rule(31, 3, %0)\n        aie.rule(28, 0, %1)")
    problems, _ = g.check_routes(SRC_WP + DST + clash, g.case_from_name("wp016"))
    assert any("(0,1)" in p for p in problems)


def test_w_circuit_and_circuit_control_routes_pass():
    problems, routes = g.check_routes(SRC_WC + DST + MEM_WC, g.case_from_name("wc016"))
    assert problems == [] and routes == g.Routes(("North", 5), ("South", 0))
    problems, routes = g.check_routes(SRC_C + DST_C, g.case_from_name("c"))
    assert problems == [] and routes.shared == ("North", 5)


# -- sequences and the table ----------------------------------------------------------------------------------

def test_seq_text_and_stall_slots():
    stamps = [10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29]
    assert g.seq_text(stamps) == "r4 g1 r4 g1 r4 g1 r5"
    assert g.stall_slots(stamps) == [(4, 1), (8, 1), (12, 1)]
    assert g.seq_text([]) == "-"


def test_pulse_slots_count_the_words_before_each_pulse():
    words = [10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29]
    assert g.pulse_slots([9, 14, 19, 24], words) == [-1, 4, 8, 12]


def test_predictions_follow_the_stage_model():
    t = g.case_from_name("t112")
    assert g.predict(t, "A", None).stalls == (4, 8, 12) and g.predict(t, "A", None).bp == (-1, 4, 8, 12)
    low = g.predict(t, "B", 100)
    assert low.stalls == (4, 12) and low.bp == (4, 12) and not low.s0 and not low.edge
    high = g.predict(t, "B", 300)
    assert high.stalls == (4, 8, 12) and high.bp == (-1, 4, 8, 12) and high.s0
    assert g.predict(t, "B", g.THRESHOLD + 3).edge
    w = g.case_from_name("wp136")
    assert g.predict(w, "A", 16).stalls == (4, 8, 12) and g.predict(w, "A", 16).bp == (4, 8, 12)
    assert g.predict(w, "A", 1000).bp == (-1, 4, 8, 12)
    c = g.predict(g.case_from_name("c"), "A", None)
    assert c.stalls == (8,) and c.bp == (-1, 8)
    assert g.predict(g.case_from_name("wc136"), "W", None).stalls == (8,)


def _stream_events(first: int, stalls=(4, 8, 12), words=17, port=1):
    cycles, t, slot = [], first, 0
    while slot < words:
        if slot in stalls:
            t += 1
        cycles.append(t)
        t += 1
        slot += 1
    return [Ev(c, 2, 0, f"PORT_RUNNING_{port}") for c in cycles]


def test_stream_row_reads_stalls_and_backpressure_slots():
    events = _stream_events(100)
    events += [Ev(c, 2, 1, "DMA_MM2S_0_STREAM_BACKPRESSURE") for c in (99, 104, 109, 114)]
    events += [Ev(90, 2, 1, "DMA_MM2S_0_STALLED_LOCK"), Ev(91, 2, 1, "DMA_MM2S_0_STALLED_LOCK"),
               Ev(95, 2, 1, "DMA_MM2S_0_FINISHED_BD")]
    row = g.stream_row(events, g.case_from_name("hdrop"), "A", backend="emu")
    assert row["seq"] == "r4 g1 r4 g1 r4 g1 r5"
    assert row["stalls"] == [(4, 1), (8, 1), (12, 1)]
    assert row["bp"] == [-1, 4, 8, 12]
    assert row["bp_seen"] and row["lock_to_word"] == 100 - 92
    pred, verdict = g.judge(g.case_from_name("hdrop"), "A", row, None)
    assert verdict == "ok"


def test_hardware_backend_shifts_the_memory_stamps_by_one():
    events = _stream_events(100) + [Ev(98, 2, 1, "DMA_MM2S_0_STREAM_BACKPRESSURE"),
                                    Ev(95, 2, 1, "DMA_MM2S_0_FINISHED_BD")]
    assert g.stream_row(events, g.case_from_name("hdrop"), "A", backend="hw")["bp"] == [-1]
    assert g.stream_row(events, g.case_from_name("hdrop"), "A", backend="emu")["bp"] == [-1]
    events = _stream_events(100) + [Ev(99, 2, 1, "DMA_MM2S_0_STREAM_BACKPRESSURE")]
    assert g.stream_row(events, g.case_from_name("hdrop"), "A", backend="hw")["bp"] == [0]


def test_a_stream_with_the_wrong_word_count_names_the_gap():
    events = _stream_events(100, words=5)
    row = g.stream_row(events, g.case_from_name("hdrop"), "A")
    assert row["missing"] and "expected 17" in row["missing"][0]
    assert g.judge(g.case_from_name("hdrop"), "A", row, None) == (None, "-")


def test_case_row_reads_n_from_the_shared_master_for_t():
    a = _stream_events(100, stalls=(), port=1)
    b = _stream_events(240, stalls=(), port=2)
    master = [Ev(e.ts, 2, 0, "PORT_RUNNING_0") for e in a + b]
    events = a + b + master + [Ev(116, 2, 0, "PORT_TLAST_0")]
    row = g.case_row(events, g.case_from_name("t112"), "emu")
    assert row["n"] == 240 - 116 - 1 and row["tlast_ok"] is True


def test_case_row_reads_n_for_w_from_w_master_to_a_slave():
    w = _stream_events(100, stalls=(), port=3)
    a = _stream_events(300, stalls=(), port=1)
    events = w + a
    assert g.case_row(events, g.case_from_name("wp136"), "emu")["n"] == 300 - 116 - 1
    wc = [Ev(e.ts, 2, 0, "PORT_RUNNING_3") for e in _stream_events(100, stalls=(), words=16, port=3)]
    assert g.case_row(wc + a, g.case_from_name("wc136"), "emu")["n"] == 300 - 115 - 1


def test_format_table_flags_a_missing_stream_and_edge_matches():
    case = g.case_from_name("t132")
    events = _stream_events(100) + _stream_events(250, stalls=(4, 12), port=2)
    events += [Ev(e.ts, 2, 0, "PORT_RUNNING_0") for e in events] + [Ev(119, 2, 0, "PORT_TLAST_0")]
    events += [Ev(c, 2, 1, "DMA_MM2S_0_STREAM_BACKPRESSURE") for c in (99, 104, 109, 114)]
    events += [Ev(c, 2, 1, "DMA_MM2S_1_STREAM_BACKPRESSURE") for c in (254, 263)]
    events += [Ev(95, 2, 1, "DMA_MM2S_0_FINISHED_BD"), Ev(245, 2, 1, "DMA_MM2S_1_FINISHED_BD")]
    text = g.format_table([(case, "x.emu", g.case_row(events, case, "emu"))])
    lines = [ln for ln in text.splitlines() if ln.startswith("t132")]
    assert len(lines) == 2 and lines[0].rstrip().endswith("ok")
    assert lines[1].rstrip().endswith("DIFF~")        # N = 130 is inside the threshold's edge band
    empty = g.format_table([(case, "x.emu", g.case_row([], case, "emu"))])
    assert "missing" in empty
