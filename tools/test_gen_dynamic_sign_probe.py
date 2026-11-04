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
