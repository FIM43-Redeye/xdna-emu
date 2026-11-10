"""The dynamic-sign probe generator: lane edges, record layouts, kernel text, case budget."""
import random
import struct

import gen_dynamic_sign_probe as g


def test_lane_edges_cover_min_max_and_minus_one():
    for bits in (8, 16, 32, 64):
        e = g.lane_edges(bits)
        m = (1 << bits) - 1
        assert {0, 1, m, 1 << (bits - 1), (1 << (bits - 1)) - 1} <= set(e)
        assert all(0 <= v <= m for v in e)


def test_every_record_matches_its_layout_and_carries_its_sign():
    rng = random.Random(1)
    for form in g.all_forms():
        for k, rec in enumerate(form.records[:7]):
            b = form.build(rng, *rec, k)
            assert len(b) == form.rec_in, form.name
            assert struct.pack("<I", rec[0]) in b, form.name
        assert form.rec_in % 64 == 0 and form.rec_out % 64 == 0, form.name


def test_every_form_sweeps_both_sign_values():
    for form in g.all_forms():
        signs = {r[0] & 1 for r in form.records}
        assert signs == {0, 1}, form.name


def test_kernels_name_the_builtin_and_stay_inside_the_budget():
    rng = random.Random(1)
    for form in g.all_forms():
        for k in g.kernels_for(form, rng):
            assert k.in_size + k.out_size <= g.CASE_BYTES, k.name
            assert k.in_size == form.rec_in * len(k.meta["records"])
            if form.compiler == "peano":
                assert "__builtin_aiev2_" in k.source


def test_register_srs_routes_through_a_runtime_zero_shift():
    reg = [f for f in g.all_forms() if f.name.startswith("srs_") and f.name.endswith("_reg")]
    assert len(reg) == len(g.SRS_PEANO)
    rng = random.Random(1)
    for f in reg:
        assert "vshift_I512_I512" in f.body
        _, _, acc_bits, lanes, *_ = g.SRS_PEANO[f.name.removesuffix("_reg")]
        off = acc_bits * lanes // 8
        sign, shift, sat = f.records[-1]
        b = f.build(rng, sign, shift, sat, 0)
        assert struct.unpack("<4I", b[off:off + 16]) == (shift, sign, sat, 0)


def test_register_ups_routes_through_a_runtime_zero_shift():
    reg = [f for f in g.all_forms() if f.name.startswith("ups_") and f.name.endswith("_reg")]
    assert len(reg) == len(g.UPS_PEANO)
    assert all("vshift_I512_I512" in f.body for f in reg)


def test_rnd_forms_cover_every_srs_form_mode_and_sign():
    forms = g.rnd_forms()
    assert len(forms) == 2 * len(g.SRS_PEANO)
    for f in forms:
        assert {r[2] for r in f.records} == set(range(16)), f.name
        assert {r[0] for r in f.records} == {0, 1}, f.name
        assert "__builtin_aiev2_set_ctrl_reg(6, " in f.body, f.name
        assert f.epilogue.strip() == "__builtin_aiev2_set_ctrl_reg(6, 0);", f.name


def test_rnd_records_carry_mode_as_fifth_scalar():
    rng = random.Random(1)
    for f in g.rnd_forms():
        key = f.name.removeprefix("rnd_").rsplit("_", 1)[0]
        _, _, acc_bits, lanes, *_ = g.SRS_PEANO[key]
        off = acc_bits * lanes // 8
        sign, shift, mode = f.records[-1]
        b = f.build(rng, sign, shift, mode, 0)
        assert len(b) == f.rec_in
        assert struct.unpack("<5I", b[off:off + 20]) == (shift, sign, 1, 0, mode)


def test_rnd_lanes_include_ties_and_inexact_values():
    rng = random.Random(1)
    for shift in g.RND_SHIFTS:
        raw = g.rnd_acc_lanes_bytes(32, 16, 64, shift, rng, 0)
        vals = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
        dropped = {v & ((1 << shift) - 1) for v in vals}
        assert {0, 1 << (shift - 1), (1 << shift) - 1} <= dropped


def test_dynsign_kernels_have_no_epilogue():
    assert all(f.epilogue == "" for f in g.all_forms())
