"""Tests for tools/gen_ldst_align_probe.py: catalogue, layout, packing, manifest guard, analysis
on synthetic outputs that follow the silicon rule (floor to the access width)."""
import json
from pathlib import Path
import struct
import subprocess

import pytest

import gen_ldst_align_probe as g
from harness.build import _peano_tool


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


def case(name):
    return next(c for c in g.build_cases() if c.name == name)


def synth_output(cases, rule=g.floor_to):
    """An output blob a machine following `rule(addr, width)` would produce (raw families only)."""
    blob = bytearray([g.MARKER]) * g.out_bytes(cases)
    blob[0:8] = struct.pack("<II", 0x74000, 0x70400)
    for c, off in zip(cases, g.layout(cases)):
        w = c.fam.width
        for i in range(c.n):
            addr = c.ptr_start + i * c.ptr_step + (c.mod_start + i * c.mod_step if "dj" in c.form else 0)
            addr += c.sp_imm
            at = rule(addr, w)
            if c.fam.kind == "st":
                base = off + i * g.SLOT
                blob[base + at:base + at + w] = bytes(range(1, w + 1))
            else:
                rec = bytes((at + k) & 0xFF for k in range(w))
                blob[off + i * c.stride:off + i * c.stride + w] = rec
        tab = off + c.data_bytes
        step = c.fam.imm if c.form.startswith("pstm") else 0
        blob[tab:tab + 4 * c.n] = struct.pack(f"<{c.n}i", *([step] * c.n))
    return bytes(blob)


def test_layout_is_contiguous_and_aligned():
    cases = g.build_cases()[:10]
    offs = g.layout(cases)
    assert offs[0] == g.HDR_BYTES
    for c, o, nxt in zip(cases, offs, offs[1:] + [g.out_bytes(cases)]):
        assert o % 64 == 0 and o + c.total_bytes == nxt


def test_packing_respects_the_budget_and_keeps_every_case():
    cases = g.build_cases()
    groups = g.pack(cases)
    assert [c.name for grp in groups for c in grp] == [c.name for c in cases]
    assert all(g.out_bytes(grp) <= g.OUT_BUDGET for grp in groups)


def test_case_names_are_unique():
    names = [c.name for c in g.build_cases()]
    assert len(names) == len(set(names))


def test_every_family_width_matches_its_register_class():
    widths = {f.key: f.width for f in g.families()}
    assert widths["st_w"] == 4 and widths["st_s16"] == 2 and widths["st_s8"] == 1
    assert widths["st_q"] == 16 and widths["vst_128"] == 16 and widths["vst_w"] == 32
    assert widths["vlda_am"] == 32


def test_natural_rule_verifies_on_a_synthetic_machine_that_follows_it():
    for name in ("st_w.imm0", "st_s16.imm0", "vst_w.imm0", "vst_128.imm0", "lda_w.imm0",
                 "vlda_w.imm0", "vst_w.dj_mix", "st_w.spill", "vlda_w.pstm_m"):
        c = case(name)
        _, res = g.parse_output([c], synth_output([c]))
        assert g.verify_natural(res[0]) == [], name


def test_a_machine_that_does_not_mask_fails_verification():
    c = case("vst_w.imm0")
    _, res = g.parse_output([c], synth_output([c], rule=lambda a, w: a))
    assert g.verify_natural(res[0])


def test_a_machine_that_masks_the_wrong_width_fails_verification():
    c = case("vst_w.imm0")
    _, res = g.parse_output([c], synth_output([c], rule=lambda a, w: g.floor_to(a, 16)))
    assert g.verify_natural(res[0])


def test_store_landing_reports_first_and_count():
    c = case("st_w.imm0")
    _, res = g.parse_output([c], synth_output([c]))
    rows = g.store_landing(res[0])
    assert rows[3]["addr"] == g.STORE_BASE + 3 and rows[3]["first"] == g.STORE_BASE
    assert rows[3]["count"] == 4


def test_pointer_update_check_flags_a_masked_pointer():
    c = case("vst_w.pstm_imm")
    blob = bytearray(synth_output([c]))
    tab = g.layout([c])[0] + c.data_bytes
    blob[tab:tab + 4] = struct.pack("<i", 0)
    _, res = g.parse_output([c], bytes(blob))
    assert any("pointer delta" in p for p in g.verify_natural(res[0]))


def test_unaligned_buffers_are_rejected():
    c = case("st_w.imm0")
    blob = bytearray(synth_output([c]))
    blob[0:8] = struct.pack("<II", 0x74020, 0x70400)
    with pytest.raises(ValueError):
        g.parse_output([c], bytes(blob))


def test_short_output_is_rejected():
    c = case("st_w.imm0")
    with pytest.raises(ValueError):
        g.parse_output([c], b"\0" * 16)


def test_lut_lane_model_matches_recorded_silicon_values():
    # gen6lut lut_4x16_lo: offsets from a 32-byte aligned base, lane 0 and lane 1
    land = [g.lut_expected_lane(0x4200 + k, 0, 16)[0] for k in range(32)]
    assert land[:4] == [0] * 4 and land[4:8] == [2] * 4 and land[28:32] == [14] * 4
    assert g.lut_expected_lane(0x4200, 1, 16)[0] == 0x10
    # lutmix_4x16_lo i=14 lane 2: bytes ca..cf then zeros (the 16-byte row ends)
    assert g.lut_expected_lane(0x40c0 + 20, 0, 16) == bytes([0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0, 0])
    # 4x64 groups of 16 map to 0 and 8
    assert g.lut_expected_lane(0x4200 + 17, 0, 64)[0] == 8


def write_suite(tmp_path, meta):
    from harness import suite
    from harness.topology import Core, TileRef, Topology
    topo = Topology("ldst_00", (Core(g.CORE, TileRef(0, 2), Path("kernel.s")),))
    suite.write_case(tmp_path, "ldst_00", topo, files={"kernel.s": ""},
                     meta=dict(name="ldst_00", cases=[], out_bytes=64))
    suite.write_suite_json(tmp_path, "ldst-align", "test", ["ldst_00"], meta=meta, generator_rev="x")


def test_manifest_with_the_current_fingerprint_loads(tmp_path):
    write_suite(tmp_path, dict(layout_fingerprint=g.layout_fingerprint()))
    assert g.load_manifest(tmp_path)[0]["name"] == "ldst_00"


def test_manifest_from_another_layout_is_refused(tmp_path):
    write_suite(tmp_path, dict(layout_fingerprint="0" * 16))
    with pytest.raises(g.ManifestMismatch):
        g.load_manifest(tmp_path)


def test_manifest_without_a_fingerprint_is_refused(tmp_path):
    write_suite(tmp_path, {})
    with pytest.raises(g.ManifestMismatch):
        g.load_manifest(tmp_path)


def test_a_packed_kernel_becomes_a_one_core_topology():
    topo, asm = g.case_topology("ldst_03", g.pack(g.build_cases())[0])
    (core,) = topo.cores
    assert (core.name, str(core.source), topo.device) == ("k", "kernel.s", "auto")
    assert core.args[1].size_bytes == g.out_bytes(g.pack(g.build_cases())[0]) and "test_kernel" in asm


def test_fingerprint_changes_when_a_case_changes(monkeypatch):
    before = g.layout_fingerprint()
    monkeypatch.setattr(g, "SECONDARY_STEP", g.SECONDARY_STEP + 1)
    assert g.layout_fingerprint() != before


def test_input_buffer_has_distinct_window_bytes_and_a_marker_block():
    data = g.in_bytes()
    assert len(data) == g.IN_BYTES
    assert data[:64] == bytes(range(1, 65))
    assert set(data[64:128]) == {g.MARKER}
    assert data[g.WIN_OFF:g.WIN_OFF + 256] == bytes(range(256))


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
def test_every_kernel_assembles(tmp_path):
    mc = peano_mc()
    for gi, grp in enumerate(g.pack(g.build_cases())):
        src = tmp_path / f"k{gi}.s"
        src.write_text(g.kernel_asm(grp, tag=f"k{gi}"))
        r = subprocess.run([mc, "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "o"), str(src)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:500]
