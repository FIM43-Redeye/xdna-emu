"""Tests for tools/gen_cascade_probe.py: asm text, topologies, depth analysis."""
import json
import shutil
import struct
import subprocess
import os
from pathlib import Path

import pytest

import gen_cascade_probe as g
from harness.build import _peano_tool
from harness.topology import Cascade, load_topology
from harness.validate import validate_topology


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


# -- asm text ----------------------------------------------------------------

def count(text, needle):
    return sum(1 for l in text.splitlines() if needle in l.split("//")[0])


@pytest.mark.parametrize("form", ["x", "bm"])
def test_smoke_asm_counts(form):
    prod = g.producer_smoke_asm(5, form)
    cons = g.consumer_smoke_asm(5, form)
    assert count(prod, "vmov MCD") == 5
    assert count(cons, ", SCD") == 5
    assert "crMCDEn" in prod and "crSCDEn" not in prod
    assert "crSCDEn" in cons and "crMCDEn" not in cons


def test_asm_frames_a_returning_function():
    for text in (g.producer_smoke_asm(2, "x"), g.consumer_smoke_asm(2, "x"),
                 g.producer_depth_asm(4), g.consumer_depth_asm(4, 10)):
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("//")]
        assert ".globl test_kernel" in lines
        i = lines.index("ret lr")
        assert lines[i + 1:i + 6] == ["nop"] * 5 and len(lines) == i + 6


def test_depth_producer_cycles_distinct_registers():
    text = g.producer_depth_asm(14)
    srcs = [l.split("MCD,")[1].strip() for l in text.splitlines() if "vmov MCD" in l]
    assert len(srcs) == 14
    assert srcs[:g.DEPTH_WORD_REGS] == [f"x{j}" for j in range(g.DEPTH_WORD_REGS)]
    assert srcs[g.DEPTH_WORD_REGS] == "x0"
    assert count(text, "cntr") == 14 + 1       # one timestamp per write plus the start


def test_depth_consumer_reads_and_delay():
    text = g.consumer_depth_asm(6, 123)
    assert count(text, ", SCD") == 6
    assert "mov r0, #123" in text
    assert count(text, "jnz") == 1


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("text", [
    g.producer_smoke_asm(3, "x"), g.consumer_smoke_asm(3, "x"),
    g.producer_smoke_asm(3, "bm"), g.consumer_smoke_asm(3, "bm"),
    g.producer_depth_asm(16), g.consumer_depth_asm(16, 500)])
def test_asm_assembles(text, tmp_path):
    s = tmp_path / "k.s"
    s.write_text(text)
    r = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o",
                        str(tmp_path / "k.o"), str(s)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- data --------------------------------------------------------------------

def test_smoke_data_words_are_distinct_and_lane_distinct():
    data = g.smoke_data(8)
    assert len(data) == 8 * 64
    words = [data[i * 64:(i + 1) * 64] for i in range(8)]
    assert len(set(words)) == 8
    lanes = struct.unpack("<16I", words[0])
    assert len(set(lanes)) == 16


def test_depth_word_values_repeat_with_register_cycle():
    assert g.depth_word_value(0) == 1
    assert g.depth_word_value(g.DEPTH_WORD_REGS) == 1
    assert g.depth_word_value(g.DEPTH_WORD_REGS + 2) == 3


# -- topologies --------------------------------------------------------------

@pytest.mark.parametrize("direction, prod, cons", [
    ("south", (0, 3), (0, 2)),
    ("east", (0, 2), (1, 2)),
])
def test_smoke_topology_placement_and_validity(direction, prod, cons, tmp_path):
    t = g.smoke_topology(direction, 4, tmp_path)
    p = next(c for c in t.cores if c.name == "producer")
    c = next(c for c in t.cores if c.name == "consumer")
    assert (p.tile.col, p.tile.row) == prod and (c.tile.col, c.tile.row) == cons
    assert t.links == (Cascade("producer", "consumer"),)
    assert validate_topology(t) == []
    assert p.args[0].direction == "in" and p.args[0].data == g.smoke_data(4)
    assert c.args[0].direction == "out" and c.args[0].size_bytes == 4 * 64


def test_depth_topology_sizes(tmp_path):
    t = g.depth_topology("south", 20, 777, tmp_path)
    p, c = t.cores
    assert [a.direction for a in p.args] == ["out"]       # no inputs: no DMA start skew
    assert c.args[0].size_bytes == g.depth_consumer_out_bytes(20)
    assert p.args[0].size_bytes == g.depth_producer_out_bytes(20)
    assert validate_topology(t) == []
    assert Path(p.source).is_file() and Path(c.source).is_file()


def test_unknown_direction_rejected(tmp_path):
    with pytest.raises(ValueError):
        g.smoke_topology("north", 2, tmp_path)


# -- analysis ----------------------------------------------------------------

def producer_out(deltas, start=1000):
    ts = [start]
    for d in deltas:
        ts.append(ts[-1] + d)
    return struct.pack(f"<{len(ts)}I", *ts)


def test_analyze_depth_finds_jump_and_count():
    # 3 fast writes, then a 5000-cycle stall, then paced writes.
    out = producer_out([6, 6, 6, 5000, 40, 40, 40])
    r = g.analyze_depth(out, 7)
    assert r.pitch == 6
    assert r.buffered == 3
    assert r.jump_cycles == 5000 - 6


def test_analyze_depth_no_stall_reports_none():
    out = producer_out([6] * 6)
    r = g.analyze_depth(out, 6)
    assert r.buffered is None and r.jump_cycles == 0


def test_analyze_depth_tolerates_pitch_jitter():
    out = producer_out([6, 7, 6, 6, 4000, 30])
    assert g.analyze_depth(out, 6).buffered == 4


def test_cli_generate_writes_loadable_topologies(tmp_path):
    rc = g.main(["generate", str(tmp_path), "--delays", "300", "--depth-k", "6"])
    assert rc == 0
    jsons = sorted(tmp_path.glob("*.json"))
    names = {j.stem for j in jsons}
    assert {"smoke_south", "smoke_east"} <= names
    assert any(n.startswith("depth_south") for n in names)
    for j in jsons:
        t = load_topology(j)
        assert validate_topology(t) == []
        for c in t.cores:
            assert Path(c.source).is_file()
