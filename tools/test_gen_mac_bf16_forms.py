"""Tests for gen_mac_bf16_forms.py: the design is reproducible, well-formed, and
the baked kernel agrees with the committed silicon golden."""

import json

import gen_mac_bf16_forms as g
import vv_data


def test_design_is_deterministic_and_whole_chunks():
    t1, t2 = g.design(), g.design()
    assert t1 == t2
    assert len(t1) % g.T == 0 and len(t1) // g.T <= g.MAX_CHUNKS


def test_forms_are_unique_and_described():
    names = [f.name for f in g.FORMS]
    assert len(set(names)) == len(names)
    for f in g.FORMS:
        assert f.geometry in ("mm", "el")
        assert f.instr in ("mul", "negmul", "mac", "msc", "addmac", "addmsc")


def test_prod_exponent_split_is_exact():
    for exp in (-127, -3, 0, 5, 100, 127):
        a, b = g.prod(exp)
        ea, eb = (a >> 7) & 0xFF, (b >> 7) & 0xFF
        assert ea + eb - 254 == exp


def test_golden_matches_design_and_staged_expected():
    golden = json.load(open(g.GOLDEN))
    tiles = g.design()
    assert golden["inputs"] == [list(x) for x in g.flat_inputs(tiles)]
    assert golden["n"] == len(tiles) * g.NF * g.LANES == len(golden["silicon"])
    assert [f["name"] for f in golden["forms"]] == [f.name for f in g.FORMS]
    data = g.kernel_data()
    assert vv_data.unpack(data.expected, "uint32_t") == golden["silicon"]


def test_input_buffer_is_chunked_u16_with_split_accumulator_words():
    tiles = g.design()
    a, b, c0, c1 = g.flat_inputs(tiles)
    words = g.pack_input(tiles)
    chunks = len(tiles) // g.T
    per_chunk = len(words) // chunks
    assert per_chunk == g.T * 32 * 2 + g.T * 16 * 2 * 2
    first = words[:per_chunk]
    assert first[:g.T * 32] == a[:g.T * 32]
    assert first[g.T * 32:g.T * 64] == b[:g.T * 32]
    c0_halves = first[g.T * 64:g.T * 64 + g.T * 32]
    assert c0_halves[0] == c0[0] & 0xFFFF and c0_halves[1] == c0[0] >> 16
    assert vv_data.unpack(g.kernel_data().inputs[0], "uint16_t") == words


def test_run_lit_names_the_shared_host_knobs():
    lit = g.render_run_lit(123)
    assert "-DVV_NIN=1 -DVV_OUT_T=uint32_t -DVV_NOUT=123" in lit


def test_generate_writes_no_host_and_no_data(tmp_path):
    g.generate(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["aie.mlir", "forms.cc", "run.lit"]
