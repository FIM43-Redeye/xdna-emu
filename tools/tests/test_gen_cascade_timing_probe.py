"""Tests for tools/gen_cascade_timing_probe.py: spec expansion, program shapes, analysis, assembly."""
import struct
import subprocess

import pytest

import gen_cascade_timing_probe as t
from harness.build import _peano_tool


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


def test_expand_specs_ranges():
    assert t.expand_specs(["rel_n0..2_m0"]) == ["rel_n0_m0", "rel_n1_m0", "rel_n2_m0"]
    assert t.expand_specs(["rt_w60_g0_r4"]) == ["rt_w60_g0_r4"]
    assert len(t.expand_specs(["wrel_n0..1_m0..2"])) == 6


def test_expand_stores_stamps_after_every_counter_read():
    asm, n = t.expand(["@ts", "  vmov x0, SCD", "@ts"], ["x0"])
    assert n == 2 and "@ts" not in asm
    last_read = max(i for i, l in enumerate(asm) if "cntr" in l)
    assert asm.index("  st r16, [p0, #0]") > last_read
    assert "  vst wl0, [p1, #0]" in asm


def test_too_many_stamps_rejected():
    with pytest.raises(ValueError):
        t.expand(["@ts"] * (len(t.TS_PAIRS) + 1), [])


def test_rel_places_the_data_read_hold_on_a_stamp():
    p = t.rel_programs(n=7, m=3)
    cons = p.consumer
    read = [i for i, l in enumerate(cons) if "SCD" in l][1]
    assert cons[read + t.HOLD_AT] == "@ts"
    assert sum("MCD" in l for l in p.producer) == 2 and p.words == 2


def test_rt_stamps_each_hold_bundle():
    p = t.rt_programs(w=60, g=1, r=4)
    sched = p.consumer[p.consumer.index("@ts", 1) + 1 + 60:]
    reads = [i for i, l in enumerate(sched) if "SCD" in l]
    assert reads == [0, 2, 4, 6]
    for i in reads:
        hold = i + t.HOLD_AT
        assert hold >= len(sched) or sched[hold] == "@ts"
    assert sum("MCD" in l for l in p.producer) == 5


def test_mem_programs_pick_the_windows():
    p = t.mem_programs("mf1", 4, delay=10)
    assert f"  movxm p7, #{t.SOUTH + t.MEM_CONS}" in p.pre[0]
    assert f"  movxm p6, #{t.LOCAL + t.MEM_CONS}" in p.pre[1]
    q = t.mem_programs("mr2", 2)
    assert f"  movxm p7, #{t.NORTH + t.MEM_PROD}" in q.pre[1]


def test_mem_probe_needs_the_south_placement():
    with pytest.raises(ValueError):
        t.build_case("mf1_k0", "east")


def _buf(stamps, words=()):
    b = bytearray(t.TS_BYTES + len(words) * t.WORD)
    struct.pack_into(f"<{len(stamps)}I", b, 0, *stamps)
    for i, w in enumerate(words):
        struct.pack_into("<16I", b, t.TS_BYTES + i * t.WORD, *([w] * 16))
    return bytes(b)


def test_analyze_rel_reads_the_consumer_stall():
    # silicon shape for n = 6, m = 0: two cycles of stall
    meta = {"family": "rel", "params": {"n": 6, "m": 0}, "words": 2, "producer_ts": 3, "consumer_ts": 3}
    prod = _buf([1000, 1002, 1010])
    cons = _buf([500, 1003, 1003 + 6 + 2], [0x100, 0x101])
    r = t.analyze(meta, prod, cons)
    assert r["data_ok"] and r["cons_stall"] == 2 and r["fwd_sync"] == 3


def test_analyze_wrel_reads_the_producer_stall():
    meta = {"family": "wrel", "params": {"n": 1, "m": 0}, "words": 4, "producer_ts": 5, "consumer_ts": 3}
    prod = _buf([1000, 1002, 1005, 1005 + 3 + 2, 1011])
    cons = _buf([500, 1003, 1009], [0x100, 0x101, 0x102, 0x103])
    r = t.analyze(meta, prod, cons)
    assert r["prod_stall"] == 2 and r["bwd"] == 1


def test_sem_case_shapes():
    (topo, files), meta = t.sem_case("semc", 3)
    assert [c.name for c in topo.cores] == ["a", "b", "c"]
    b = files["b.s"]
    assert b.index("vmov x0, SCD") < b.index("vmov MCD, x5")
    (topo, files), _ = t.sem_case("semsr", 2)
    assert files["b.s"].index("mov ms, r5") < files["b.s"].index("vmov x0, SCD")
    with pytest.raises(ValueError):
        t.sem_case("semc", 0)


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("spec", ["rel_n5_m0", "wrel_n2_m0", "rt_w0_g0_r6", "mf2_s10_k3", "mr1_k2"])
def test_programs_assemble(tmp_path, spec):
    (topo, files), _ = t.build_case(spec, "south")
    for name, text in files.items():
        src = tmp_path / name
        src.write_text(text)
        r = subprocess.run([str(peano_mc()), "--triple=aie2", "--filetype=obj", "-o",
                            str(tmp_path / (name + ".o")), str(src)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
