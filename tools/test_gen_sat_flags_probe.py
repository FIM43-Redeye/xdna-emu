"""The saturation-flag probe generator: forms, records, kernel text, the analyzer's model."""
import random
import re
import struct

import gen_sat_flags_probe as g


def test_forms_cover_every_flag_defining_family():
    names = {f.name for f in g.all_forms()}
    for stem in ("vsrs_d16_s32", "vsrsm_d32_s64", "vst_srs_d8_s32", "vst_2d_srs_d16_s64",
                 "vst_3d_srs_d32_s64", "vups_s64_d32", "vups_x2c_s32_d16", "vlda_ups_s32_d8",
                 "vlda_ups_s64_d32_pstm", "vlda_ups_s64_s32_pstm", "vlda_2d_ups_s32_d16_2d",
                 "vlda_3d_ups_s64_d16_3d"):
        assert stem in names, stem
    assert len(names) == len(g.all_forms())


def test_records_sweep_crsat_and_both_signs_of_a_d_form():
    for form in g.all_forms():
        recs = g.records_for(form)
        assert {r.sat for r in recs} >= {0, 1, 3}, form.name
        assert {r.sign for r in recs} == ({0, 1} if form.dform else {0}), form.name
        assert {r.init for r in recs} == {0, 1}, form.name


def test_record_bytes_carry_the_parameters_at_their_offset():
    rng = random.Random(1)
    form = g.all_forms()[0]
    rec = g.Record("all", "none", 5, 1, 2, 1, 9)
    b = g.record_bytes(form, rec, rng)
    assert len(b) == g.REC_IN
    assert struct.unpack_from("<5I", b, g.PARAM_OFF) == (5, 1, 2, 1, 9)


def test_kernels_stay_inside_the_budget_and_load_params_through_p4():
    rng = random.Random(1)
    for form in g.all_forms()[:6]:
        for name, text, data, meta in g.cases_for(form, rng):
            n = len(meta["records"])
            assert len(data) == n * g.REC_IN
            assert n * (g.REC_IN + g.REC_OUT) <= g.CASE_BYTES
            # A scalar load's immediate is 8 bits signed.
            for imm in re.findall(r"lda r\d+, \[p\d, #(-?\d+)\]", text):
                assert -128 <= int(imm) <= 127
            # Read-back, final read, and the first read after each of A and B.
            assert text.count(f"mov r0, {form.flag}") == 4
            assert ".p2align 4\n.Lrec:" in text


def test_the_model_flags_any_lane_out_of_range_and_the_symmetric_minimum():
    meta = {"family": "srs", "dform": True, "lanes": 16, "acc_bits": 32, "vec_bits": 16}
    lanes = [0] * 16
    lanes[3] = -32768
    opnd = b"".join(struct.pack("<i", v) for v in lanes)
    assert not any(g.lanes_overflow(meta, opnd, ["smin", "none", 0, 1, 1, 0, 0]))
    assert any(g.lanes_overflow(meta, opnd, ["smin", "none", 0, 1, 2, 0, 0]))
    assert any(g.lanes_overflow(meta, opnd, ["smin", "none", 0, 0, 0, 0, 0])), "unsigned"


def test_predicted_words_are_sticky_and_late():
    meta = {"family": "ups", "kind": "load", "dform": False, "lanes": 8, "acc_bits": 64,
            "vec_bits": 32}
    a = struct.pack("<8I", *([0x7FFF_FFFF] * 8)).ljust(g.OPND_BYTES, b"\0")
    b = bytes(g.OPND_BYTES)
    w = g.predicted_words(meta, a + b, ["all", "none", 33, 0, 1, 0, 0])
    assert w[g.F_A:g.F_A + 8] == [0] * 8 and w[g.F_A + 8] == 1     # visible at offset 9
    assert w[g.F_B:g.F_B + g.READS] == [1] * g.READS and w[g.F_FINAL] == 1
