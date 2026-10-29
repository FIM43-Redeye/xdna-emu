"""Tests for gen_mac_bf16_forms.py: the design is reproducible, well-formed, and
the baked kernel agrees with the committed silicon golden."""

import json
import re

import gen_mac_bf16_forms as g


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


def test_golden_matches_design_and_baked_expected():
    golden = json.load(open(g.GOLDEN))
    tiles = g.design()
    assert golden["inputs"] == [list(x) for x in g.flat_inputs(tiles)]
    assert golden["n"] == len(tiles) * g.NF * g.LANES == len(golden["silicon"])
    assert [f["name"] for f in golden["forms"]] == [f.name for f in g.FORMS]
    text = open(g.OUT_DIR + "/test.cpp").read()
    m = re.search(r"EXP\[\d+\] = \{([^}]*)\}", text, re.S)
    assert [int(x) for x in re.findall(r"\d+", m.group(1))] == golden["silicon"]
