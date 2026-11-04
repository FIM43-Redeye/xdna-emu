"""Tests for tools/gen_lut_addr_probe.py: the window pattern, the record catalogue, the suite
it writes, and the analysis on a synthetic output that follows the silicon rule."""
import struct

import gen_lut_addr_probe as g


def test_every_window_halfword_names_its_window_and_position():
    data = g.in_bytes()
    assert data[:32] == bytes([g.MARKER]) * 32
    for w in (1, 2, 3):
        for j in (0, 1, 255):
            off = g.WIN_OFF[w] + 2 * j
            assert struct.unpack_from("<H", data, off)[0] == (w << 12) | j


def test_lane_offsets_stay_inside_the_window_after_the_lane_rule():
    recs = [r for rs in g.build_records().values() for r in rs if isinstance(r, g.Record)]
    for r in recs:
        offs = g.LANE_OFFS[4:] if r.half == "hi" else g.LANE_OFFS[:4]
        for k, off in enumerate(offs):
            eff = g.lut_lane_address(0x74000 + 64 + off + r.shift, k, r.width) - (0x74000 + 64)
            assert 0 <= eff < g.WIN_BYTES, r.name


def test_cases_and_record_names_are_unique():
    recs = g.build_records()
    assert list(recs) == ["lut_alias", "ptr_alias", "lut_unwritten"]
    for rs in recs.values():
        names = [r.name for r in rs]
        assert len(set(names)) == len(names)


def test_generate_writes_one_case_per_group(tmp_path):
    names = g.generate(tmp_path)
    assert names == ["lut_alias", "ptr_alias", "lut_unwritten"]
    asm = (tmp_path / "cases" / "lut_alias" / "kernel.s").read_text()
    assert "vldb.4x64.lo wl2, wl4" in asm and "vldb.4x16.hi wl2, wl4" in asm


def _synthetic(names, recs, inp=0x74000, far=0x7FE00, north=0x64000):
    """An output blob where every gather follows the hypothesis: bits 16..17 pick the region."""
    h = g.Header(inp, 0x70400, far, north)
    blob = bytearray(struct.pack("<4I", inp, 0x70400, far, north) + bytes(g.HDR_BYTES - 16))
    for r in recs:
        base = {"w1": inp + g.WIN_OFF[1], "far": far, "north": north}[r.target]
        lanes = [((base + o + r.shift) ^ r.xor) & 0xFFFFFFFF for o in g.LANE_OFFS]
        first = 4 if r.half == "hi" else 0
        res = b""
        for k in range(4):
            addr = g.model_address(g.lut_lane_address(lanes[first + k], k, r.width))
            res += g.expected_bytes(h, addr)
        blob += struct.pack("<8I", *lanes) + res
    return bytes(blob)


def test_analysis_decodes_aliased_lanes_to_the_region_they_read():
    recs = g.build_records()["lut_alias"]
    names = [r.name for r in recs]
    lines = g.report(_synthetic(names, recs), names)
    lanes = [ln for ln in lines if ln.startswith("  lane")]
    assert len(lanes) == 4 * len(recs)
    assert all(ln.endswith(" model") for ln in lanes), [ln for ln in lanes if "model" not in ln][:3]


def test_model_ignores_bits_above_seventeen():
    assert g.model_address(0xFFF74040) == 0x74040
    assert g.model_address(0x000E4000) == 0x64000
    assert g.model_address(0x00034000) == 0x74000
