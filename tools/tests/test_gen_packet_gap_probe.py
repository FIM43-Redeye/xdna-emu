"""Tests for tools/gen_packet_gap_probe.py: design text, safety shape, trace-sharing check, table."""
import re
import subprocess
from collections import namedtuple

import pytest

import gen_packet_gap_probe as g
from harness.build import _peano_tool

Ev = namedtuple("Ev", "row pkt_type name ts")
SRC = g.ROW, g.PT_MEM


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


# -- cases and design text -------------------------------------------------------------

def test_case_names_round_trip():
    for case in g.default_cases():
        assert g.case_from_name(case.name) == case
    with pytest.raises(ValueError):
        g.case_from_name("nonsense")


def test_default_cases_cover_the_window_and_controls():
    names = [c.name for c in g.default_cases()]
    assert "solo" in names and "far" in names
    assert len(names) == len(set(names))
    assert [c.k for c in g.default_cases() if c.mode == "ab"] == list(g.AB_KS)
    assert len(g.AB_KS) >= 13        # at least the -6..+6 window, one cycle apart


def test_ab_design_has_two_sources_into_one_destination():
    text = g.design_mlir(g.Case("ab", 10))
    assert text.count("aie.packet_flow(") == 2
    assert "aie.packet_source<%t02, DMA : 0>" in text and "aie.packet_source<%t02, DMA : 1>" in text
    assert text.count("aie.packet_dest<%t03, DMA : 0>") == 2
    assert f"pkt_id = {g.PKT_A}>" in text and f"pkt_id = {g.PKT_B}>" in text
    assert "memref<20xi32>" in text            # room for every word sent


def test_same_design_chains_two_bds_with_one_packet_id():
    text = g.design_mlir(g.Case("same", 10))
    assert text.count("aie.packet_flow(") == 1
    assert text.count(f"pkt_id = {g.PKT_A}>") == 2 and f"pkt_id = {g.PKT_B}>" not in text
    assert "aie.next_bd ^b" in text and text.count("aie.dma_start(MM2S") == 1


def test_solo_design_sends_only_b_and_sizes_the_destination_for_it():
    text = g.design_mlir(g.Case("solo"))
    assert text.count("aie.dma_start(MM2S") == 1 and "DMA : 1>" in text
    assert re.search(r'sym_name = "dst"\} : memref<4xi32>', text)


@pytest.mark.parametrize("case", g.default_cases()[:3] + [g.Case("same", 8), g.Case("solo"), g.Case("far")])
def test_design_waits_on_nothing_host_side(case):
    text = g.design_mlir(case)
    seq = text[text.index("aie.runtime_sequence"):]
    ops = set(re.findall(r"aiex?\.[\w.]+", seq))
    assert ops <= {"aie.runtime_sequence", "aiex.set_lock", "aiex.npu.write32"}
    for banned in ("dma_wait", "npu.sync", "dma_memcpy", "issue_token"):
        assert banned not in text
    # the only lock a DMA acquires is released by the core, the go lock by the sequence
    assert "%lGo" in seq and text.count("AcquireGreaterEqual") == (1 if case.mode == "solo" else 2)


# -- core program ------------------------------------------------------------------------

def bundles(text):
    lines = [ln.strip() for ln in text.splitlines()]
    start = max(i for i, ln in enumerate(lines) if ln.startswith("st r31,")) + 1
    end = min(i for i, ln in enumerate(lines) if ln.startswith("lda r16,"))
    return [ln for ln in lines[start:end] if ln and not ln.startswith((".", "#")) and not ln.endswith(":")]


@pytest.mark.parametrize("k", [0, 1, 9, 29])
def test_core_releases_a_then_k_nops_then_b(k):
    body = bundles(g.core_asm(g.Case("ab", k)))
    i = body.index(f"rel #{g.OWN + g.LOCK_A}, r2")
    assert body[i - 1] == "event #0"
    assert body[i + 1:i + 1 + k] == ["nop"] * k
    assert body[i + 1 + k] == f"rel #{g.OWN + g.LOCK_B}, r2"


def test_core_waits_for_go_first_and_never_releases_a_in_solo():
    body = bundles(g.core_asm(g.Case("solo")))
    assert any(b == f"acq #{g.OWN + g.LOCK_GO}, r3" for b in body)
    assert f"rel #{g.OWN + g.LOCK_A}, r2" not in body and f"rel #{g.OWN + g.LOCK_B}, r2" in body


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("case", [g.Case("ab", 0), g.Case("ab", 29), g.Case("solo"), g.Case("far")])
def test_core_asm_assembles(case, tmp_path):
    s = tmp_path / "core.s"
    s.write_text(g.core_asm(case))
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "c.o"), str(s)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_inject_args_fit_the_trace_slots():
    assert len(g.MEMMOD_EVENTS) <= 8
    assert len(g.CORE_GROUNDING.split(",")) + len(g.CORE_EVENTS) <= 8
    assert "0=North:5:master" in g.PORT_SLOTS


# -- the trace-sharing check ---------------------------------------------------------------

def box(trace_amsel):
    return f"""    %switchbox_0_2 = aie.switchbox(%tile_0_2) {{
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.masterset(South : 1, %1)
      %4 = aie.masterset(South : 3, %2)
      %5 = aie.masterset(North : 5, %0)
      aie.packet_rules(Trace : 0) {{
        aie.rule(31, 1, {trace_amsel})
      }}
      aie.packet_rules(DMA : 1) {{
        aie.rule(31, 2, %0)
      }}
      aie.packet_rules(DMA : 0) {{
        aie.rule(31, 1, %0)
      }}
    }}
"""


def test_data_flows_share_one_master():
    assert g.data_masters(box("%2")) == {("North", 5)}


def test_trace_on_its_own_arbiter_is_clean():
    assert g.trace_shares_master(box("%2")) == []


def test_trace_on_the_data_arbiter_is_flagged():
    bad = g.trace_shares_master(box("%0"))
    assert bad and "North" in bad[0]


def test_trace_on_the_data_arbiter_number_is_flagged_even_on_another_master():
    text = box("%1").replace("%1 = aie.amsel<1> (0)", "%1 = aie.amsel<0> (1)")
    assert any("arbiter 0" in b for b in g.trace_shares_master(text))


# -- the table -------------------------------------------------------------------------------

def levels(row, pkt, name, spans):
    return [Ev(row, pkt, name, t) for a, b in spans for t in range(a, b + 1)]


def run_events(gap, grant, with_tlast=True):
    """A's 16 words from cycle 100, then `gap` starved cycles, then B's 4 words; the destination's
    starvation level is high before the first word and during the gap."""
    a_end = 115
    starv = [(90, 99)] + ([(a_end + 1, a_end + gap)] if gap else [])
    starv = [(90, 99)] + ([(a_end + 1, a_end + gap + 1)])   # B's header word is one more starved cycle
    ev = levels(g.DST_ROW, g.PT_MEM, "DMA_S2MM_0_STREAM_STARVATION", starv)
    ev += levels(g.ROW, g.PT_MEM, "DMA_MM2S_1_STALLED_LOCK", [(80, grant - 1)])
    ev += levels(g.ROW, g.PT_MEM, "DMA_MM2S_1_STREAM_BACKPRESSURE", [(110, 111)])
    if with_tlast:
        ev += levels(g.ROW, g.PT_CORE, "PORT_RUNNING_0", [(40, 40), (50, 66), (67 + gap, 71 + gap)])
        ev += [Ev(g.ROW, g.PT_CORE, "PORT_TLAST_0", 65)]
    return ev


@pytest.mark.parametrize("gap", [0, 1, 2])
def test_gap_row_reads_the_gap_and_port_gap(gap):
    row = g.gap_row(run_events(gap, 95), g.Case("ab", 12))
    assert row["first_word"] == 100
    assert row["gap"] == gap and row["raw_gap"] == gap + 1
    assert row["port_gap"] == gap
    assert row["a_last"] == 15
    assert row["b_grant"] == -5
    assert row["backpressure"] == {0: 0, 1: 2}


def test_gap_row_without_any_starvation_event_has_no_first_word():
    assert g.gap_row([], g.Case("ab", 3)) == {"first_word": None}


def test_same_stream_case_reads_the_last_stalled_lock_of_channel_zero():
    ev = levels(g.DST_ROW, g.PT_MEM, "DMA_S2MM_0_STREAM_STARVATION", [(90, 99)])
    ev += levels(g.ROW, g.PT_MEM, "DMA_MM2S_0_STALLED_LOCK", [(80, 94), (120, 130)])
    assert g.gap_row(ev, g.Case("same", 5))["b_grant"] == 131 - 100


def test_ready_offset_uses_the_solo_latency():
    solo = g.gap_row(run_events(0, 85), g.Case("solo"))
    lat = g.solo_latency(solo)
    assert lat == 15
    ab = g.gap_row(run_events(1, 95), g.Case("ab", 12))
    assert g.b_ready(ab, lat) == -5 + 15 - 15
    assert g.b_ready(ab, None) is None


def test_format_table_has_a_row_per_case_and_marks_missing_runs():
    solo = g.gap_row(run_events(0, 85), g.Case("solo"))
    text = g.format_table([(g.Case("ab", 12), "e1.emu", g.gap_row(run_events(2, 95), g.Case("ab", 12))),
                           (g.Case("solo"), "e1.emu", solo), (g.Case("far"), "e1.emu", {"first_word": None})],
                          g.solo_latency(solo))
    lines = text.splitlines()
    assert len(lines) == 4 and "no S2MM starvation event" in lines[3]
    assert lines[1].split()[:5] == ["ab12", "e1.emu", "-5", "2", "2"]
