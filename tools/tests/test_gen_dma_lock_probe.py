"""Tests for tools/gen_dma_lock_probe.py: predictions, programs, design text."""
import struct
import subprocess

import pytest

import gen_dma_lock_probe as g
from harness.build import _peano_tool


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


def test_every_case_tells_the_rules_apart():
    for case in g.CASES.values():
        assert (g.grant_step(case, "eq"), g.taken(case, "eq")) != \
            (g.grant_step(case, "eqdec"), g.taken(case, "eqdec"))


def test_grant_steps():
    assert g.grant_step(g.CASES["c2v1"], "eq") == 1
    assert g.grant_step(g.CASES["c0v0"], "eq") == 0
    assert g.grant_step(g.CASES["c0v0"], "eqdec") == 1


def test_design_uses_the_equal_form_on_bd0():
    text = g.design_mlir(g.CASES["c2v1"])
    assert "%a0 = arith.constant 2 : i32\n      aie.use_lock(%lA, Acquire, %a0)" in text
    assert "aie.flow(%t02, DMA : 0, %t02, DMA : 0)" in text


def test_analyze_out_matches_a_rule():
    meta = {"steps": 3, "grant_step": {"eq": 1, "eqdec": 1}, "taken": {"eq": 0, "eqdec": 2}}
    w = [100, 3700, 7300, 10900, 10905, 10910] + [0] * 9 + [1]
    r = g.analyze_out(meta, struct.pack("<16I", *w), 4000.0)
    assert r["match"] == ["eq"] and r["marker"] == 1


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("name", list(g.CASES))
def test_asm_assembles(name, tmp_path):
    for k, text in (("core", g.core_asm(g.CASES[name], 50)), ("rescuer", g.rescuer_asm(g.CASES[name], 50))):
        s = tmp_path / f"{k}.s"
        s.write_text(text)
        r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / f"{k}.o"),
                            str(s)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
