"""Tests for tools/gen_port_latch_probe.py: the self-describing input, the uniqueness of every
access a case makes, the suite it writes, and the analysis of a synthetic output."""
import struct

import gen_port_latch_probe as g


def test_every_window_halfword_names_its_region_and_offset():
    data = g.in_bytes()
    assert data[:g.MARK_BYTES] == bytes([g.MARKER]) * g.MARK_BYTES
    for r in g.REGIONS:
        for off in (0, 2, g.WIN_BYTES - 2):
            h = struct.unpack_from("<H", data, g.WIN_OFF[r] + off)[0]
            assert g.halfword_source(h) == (r, off)


def test_marker_and_zero_decode_as_such():
    assert g.halfword_source(0xEEEE) == "marker"
    assert g.halfword_source(0) == "zero"


def test_addresses_follow_the_quadrant_xor():
    acc = g.Access("vldb", "S", "SC", 3, 4)
    assert acc.address(0x70400) == ((0x70400 + g.WIN_OFF["SC"] + 3 * 32 + 4) ^ 0x30000)
    assert g.Access("vlda", "W", "WC", 0).address(0x70400) >> 16 == 5
    assert g.Access("vlda", "N", "NC", 0).address(0x70400) >> 16 == 6


def test_record_cases_never_reuse_bytes():
    for name, (tile, recs, setup) in g.build_cases().items():
        g.check_blocks(recs)


def test_ring_flush_lines_stay_clear_of_record_blocks():
    assert g.OWN_LIMIT == g.FLUSH_A[0]
    assert g.FLUSH_B[-1] < g.REC_BLOCKS
    assert len(g.FLUSH_A) == len(g.FLUSH_B) == g.RING_DEPTH


def test_ring_cases_plan_within_their_windows():
    for name, blocks in g.ring_cases().items():
        plans = g.ring_plans(blocks)
        assert len(plans) == len(blocks)
        for fills, ops, probes in plans:
            assert len(fills) == g.RING_DEPTH and len(probes) == g.RING_PROBES


def test_generate_writes_every_case(tmp_path):
    names = g.generate(tmp_path)
    assert {"unit", "bytes", "age", "ctrl", "sweep", "ring_a", "ring_top", "ring_f"} <= set(names)
    asm = (tmp_path / "cases" / "ring_c" / "kernel.s").read_text()
    assert "vldb.4x64.lo wl6, wl4" in asm
    topo = (tmp_path / "cases" / "ring_top" / "topology.json").read_text()
    assert '"row": 5' in topo


def _ring_model_blob(blocks):
    """The output the silicon rule predicts for a port-B ring case with no ops: each block's
    probes return its fill lines oldest first, then the first again."""
    blob = bytearray(struct.pack("<2I", 0x70400, 0x74000) + bytes(g.HDR_BYTES - 8))
    data = g.in_bytes()
    for fills, ops, probes in g.ring_plans(blocks):
        blob += bytes([g.MARKER]) * 32
        for k in range(g.RING_PROBES):
            f = fills[k % g.RING_DEPTH]
            blob += data[f.offset():f.offset() + 32]
    return bytes(blob)


def test_analysis_names_fill_lines_in_order():
    blocks = [g.RingBlock("none")]
    lines = g.analyze_ring(_ring_model_blob(blocks), blocks, g.SETUP_ROW2)
    probes = [ln for ln in lines if ln.startswith("  p")]
    assert [p.split(": ")[1] for p in probes] == ["[0:32]=F0", "[0:32]=F1", "[0:32]=F2",
                                                  "[0:32]=F3", "[0:32]=F4", "[0:32]=F5",
                                                  "[0:32]=F0"]
