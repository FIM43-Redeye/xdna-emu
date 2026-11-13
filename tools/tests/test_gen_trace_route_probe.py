"""Tests for tools/gen_trace_route_probe.py: cases, design text, the hand decoder."""
import pytest

import gen_trace_route_probe as g


def test_every_case_names_its_carrier_and_device():
    assert set(g.CASES) == {"w0", "w1", "w2", "w3", "sc", "si", "se", "d1q", "d2q"}
    for c in g.CASES.values():
        assert c.device in ("npu1_1col", "npu1_2col")
        assert c.carrier_col < (2 if c.device == "npu1_2col" else 1)


@pytest.mark.parametrize("name", ["w1", "w3"])
def test_next_column_cases_send_the_trace_to_that_shim(name):
    text = g.design_mlir(g.CASES[name])
    assert "egress_shim_col = 1 : i32" in text
    assert "aie.trace @trace_carrier(%t10)" in text


@pytest.mark.parametrize("name,events", [("w2", ["BROADCAST_1", "BROADCAST_2"]),
                                         ("w3", ["BROADCAST_A_1", "BROADCAST_A_2"])])
def test_broadcast_sources_trace_the_core_events_as_broadcasts(name, events):
    text = g.design_mlir(g.CASES[name])
    for ev in events:
        assert f'aie.trace.event <"{ev}">' in text
    # the core module raises INSTR_EVENT_0 / _1 (33 / 34) on channels 1 / 2
    assert f"arith.constant {0x34014} : i32" in text and "arith.constant 33 : i32" in text
    assert f"arith.constant {0x34018} : i32" in text and "arith.constant 34 : i32" in text
    assert "INSTR_EVENT_0" not in text, "the core itself is not traced"


def test_core_sources_write_no_broadcast_config():
    assert "column = 0 : i32, row = 2 : i32" not in g.design_mlir(g.CASES["w0"])


def test_sync_cases_outlast_the_wrap_and_differ_only_in_the_wait():
    si, sc, se = g.CASES["si"], g.CASES["sc"], g.CASES["se"]
    assert si.sync_wait > g.SYNC_FIELD > sc.sync_wait
    assert se.sync_end and se.sync_wait is None
    # about 103 cycles per pad write: the trace must still run 2^18 cycles after the program's start
    assert si.pad_writes * 100 > si.sync_wait


def _packet(order, payload, col=0, row=2, pkt_type=0):
    return {"order": order, "col": col, "row": row, "pkt_type": pkt_type, "bytes": payload}


def test_timeline_decodes_frames_syncs_and_repeats():
    start = [0xF0, 0, 0, 0, 0, 0, 0x10, 0x00]           # Start, timer 0x1000
    pkts = [_packet(0, start + [0x03, 0x00, 0xFF, 0xE1, 0xA0, 0x00, 0x05]
                    + [0xFE] * (28 - 15))]
    tl = g.stream_timeline(pkts, (0, 2, 0))[0]
    # Single0 delta 3 -> 0x1004; Single0 delta 0 -> 0x1005; Sync + Repeat(1) -> +2 wraps; Single2 delta 5
    assert tl["frames"] == [0x1004, 0x1005, 0x1005 + 2 * g.SYNC_FIELD + 6]
    assert tl["toks"][:5] == ["Start", "Single0", "Single0", "Sync", "Rep(1)"]


def test_timeline_repeats_a_frame_after_a_frame():
    start = [0xF0, 0, 0, 0, 0, 0, 0, 0]
    pkts = [_packet(0, start + [0x00, 0xE3] + [0xFE] * 18)]
    assert g.stream_timeline(pkts, (0, 2, 0))[0]["frames"] == [1, 2, 3, 4]


@pytest.mark.parametrize("name", ["d1q", "d2q"])
def test_data_cases_finish_every_task_inside_the_buffer(name):
    text = g.design_mlir(g.CASES[name]) if g.CASES[name].data_dest is None else g.data_design_mlir(g.CASES[name])
    # every BD starts at offset 0 and moves DATA_WORDS words
    assert text.count(f"[%c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c{g.DATA_WORDS}]") == text.count("dma_memcpy_nd")
    # the sequence ends waiting for the shim task it can wait on
    assert text.rstrip().splitlines()[-4].strip().startswith("aiex.npu.dma_wait")
    if name == "d2q":
        assert "aie.next_bd ^bb99" in text and "len = 16" in text   # one memtile BD, no successor
