"""Tests for tools/gen_lock_stall_probe.py: programs, the acquire-rule simulator, assembly."""
import struct
import subprocess

import pytest

import gen_lock_stall_probe as g
from harness.build import _peano_tool
from harness.device_model import load_device_model
from harness.validate import validate


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


@pytest.mark.parametrize("d", range(1, 9))
def test_lk_program_puts_the_acquire_d_bundles_after_the_load(d):
    prog = g.lk_program(d)
    load = next(i for i, l in enumerate(prog) if "lda r6" in l)
    acq = next(i for i, l in enumerate(prog) if "acq" in l)
    assert acq - load == d
    assert sum("mov r" in l and ", r6" in l for l in prog) == 7


def test_lk_d_out_of_range():
    with pytest.raises(ValueError):
        g.lk_program(9)


def test_cases_validate_and_share_one_lock():
    dev = load_device_model("npu1")
    for topo, _ in (g.lk_case(3, "stall", 100), g.same_case("free", 100), g.eq_case("c0v1", 50)):
        assert validate(topo, dev) == []
        (lock,) = topo.links
        assert (lock.src, lock.dst, lock.lock_id) == ("consumer", "producer", g.LOCK_ID)


def test_stall_mode_delays_the_producer_and_free_mode_the_consumer():
    _, stall = g.lk_case(2, "stall", 100)
    _, free = g.lk_case(2, "free", 100)
    assert ".Lprod" in stall["producer.s"] and ".Lcons" not in stall["consumer.s"]
    assert ".Lcons" in free["consumer.s"] and ".Lprod" not in free["producer.s"]


@pytest.mark.parametrize("name", list(g.EQ_CASES))
def test_every_eq_case_completes_under_every_rule(name):
    for rule in g.RULES:
        steps = g.simulate(g.EQ_CASES[name], rule)
        assert len(steps) == len(g.EQ_CASES[name].consumer)


@pytest.mark.parametrize("name", list(g.EQ_CASES))
def test_every_eq_case_tells_the_silicon_rule_from_the_emu_core_rule(name):
    case = g.EQ_CASES[name]
    assert g.simulate(case, "eq") != g.simulate(case, "ge")


def test_some_eq_case_tells_the_dma_rule_from_the_core_rule():
    assert any(g.simulate(c, "ge") != g.simulate(c, "eqdec") for c in g.EQ_CASES.values())


def test_rule_grants_matches_the_host_path_measurements():
    # build/experiments/objectfifo-wedges NOTES finding 3: (2,+2) yes, (1,+1) yes,
    # (0,0) yes, (10,+2) no, (5,0) no, all leaving the count unchanged
    assert [g.rule_grants("eq", c, ch) for c, ch in ((2, 2), (1, 1), (0, 0), (10, 2), (5, 0))] == \
        [2, 1, 0, None, None]


def test_simulate_detects_a_deadlock():
    with pytest.raises(g.Deadlock):
        g.simulate(g.EqCase(0, (("acq", -1),), (("acq", -1),)), "eq")


def test_parse_sampler():
    words = [111] * 5 + [222] * 3 + [1000, 1040, 4] + [0] * 4 + [222]
    r = g.parse_sampler(struct.pack("<16I", *words))
    assert (r["new_from"], r["window"], r["ptr_delta"]) == (6, 40, 4)


def test_parse_eq_quantizes_waits_to_producer_steps():
    log = struct.pack("<3I", 100, 4100 + 90, 4100 + 95) + bytes(52)
    r = g.parse_eq("c2v1", log, bytes(32), 4000.0)
    assert r["steps"] == [1, 1] and r["match"] == ["eq"]


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("made", [
    lambda: g.lk_case(1, "stall", 100), lambda: g.lk_case(8, "free", 100),
    lambda: g.same_case("stall", 100), lambda: g.pos_case("pos", "free", 100),
    lambda: g.pos_case("relpos", "stall", 0), *[(lambda n=n: g.eq_case(n, 50)) for n in g.EQ_CASES]])
def test_asm_assembles(made, tmp_path):
    _, files = made()
    for fname, text in files.items():
        s = tmp_path / fname
        s.write_text(text)
        res = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o",
                              str(tmp_path / (fname + ".o")), str(s)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_cli_generate_writes_a_loadable_suite(tmp_path):
    from harness import suite
    assert g.main(["generate", str(tmp_path), "--cases", "lk_2", "same", "eq_c1v1", "--modes", "stall"]) == 0
    s = suite.load_suite(tmp_path)
    assert s.cases == ("lk_2_stall", "same_stall", "eq_c1v1")
    assert s.case_meta("eq_c1v1")["predictions"]["eq"] == [0, 0]


def test_pos_reads_the_counter_in_every_bundle_after_the_request():
    _, files = g.pos_case("pos", "stall", 100)
    lines = files["consumer.s"].splitlines()
    i = next(k for k, l in enumerate(lines) if "acq #" in l)
    assert all("cntr" in l for l in lines[i - 1:i + 8] if l != lines[i])


def test_parse_pos():
    log = struct.pack("<8I", 10, 11, 12, 13, 900, 901, 902, 903) + bytes(32)
    assert g.parse_pos(log) == [1, 1, 1, 887, 1, 1, 1]


@pytest.mark.parametrize("d", range(1, g.B2B_MAX_D + 1))
@pytest.mark.parametrize("window", list(g.B2B_WINDOWS))
def test_b2b_program_puts_request_2_d_bundles_after_request_1(d, window):
    prog = g.b2b_program("relacq", "gg", d, window)
    reqs = [i for i, l in enumerate(prog) if "acq #" in l or "rel #" in l]
    assert len(reqs) == 2 and reqs[1] - reqs[0] == d
    assert [i - reqs[0] for i, l in enumerate(prog) if "cntr" in l] == g.b2b_layout(window)
    assert len(prog) == g.B2B_LAST + 2


def test_b2b_windows_overlap_in_one_read():
    a, b = (set(g.b2b_layout(w)[1:]) for w in ("a", "b"))
    assert a & b == {6} and a | b == set(range(0, 13))


def test_b2b_cases_validate_and_release_only_the_refused_lock():
    dev = load_device_model("npu1")
    for pair, variants in g.B2B_VARIANTS.items():
        for v in variants:
            topo, files = g.b2b_case(pair, v, 3, "a", 100)
            assert validate(topo, dev) == []
            inits = {l.lock_id: l.init for l in topo.links}
            assert (inits[g.B2B_FIRST], inits[g.LOCK_ID]) == g.b2b_inits(pair, v)
            lock = g.b2b_refused_lock(v)
            rels = [l.split()[1].strip("#,") for l in files["producer.s"].splitlines() if "rel #" in l]
            fallbacks = [str(l) for l in g.B2B_COUNT_LOCKS for _ in range(2)]
            assert rels == ([] if lock is None else [str(lock)]) + fallbacks
            if lock is not None:
                assert inits[lock] == 0


def test_parse_b2b_reports_lost_cycles_and_the_resume():
    # window a: reads at -1, 0..6; one cycle lost in front of +4 and a stall of 100 there
    ts = [0, 1, 2, 3, 105, 106, 107, 108]
    r = g.parse_b2b("a", struct.pack("<8I", *ts) + bytes(32), struct.pack("<I", 98) + bytes(28))
    assert r["layout"] == [-1, 0, 1, 2, 3, 4, 5, 6]
    assert r["lost"] == [0, 0, 0, 0, 101, 101, 101, 101]
    assert r["resume"] == 7


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("pv", [(p, v) for p, vs in g.B2B_VARIANTS.items() for v in vs])
def test_b2b_asm_assembles(pv, tmp_path):
    for d, w in ((1, "a"), (g.B2B_MAX_D, "b")):
        _, files = g.b2b_case(*pv, d, w, 100)
        for fname, text in files.items():
            s = tmp_path / fname
            s.write_text(text)
            res = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o",
                                  str(tmp_path / (fname + ".o")), str(s)], capture_output=True, text=True)
            assert res.returncode == 0, res.stderr


def test_parse_b2b_counts():
    # producer: window release at 100, fallbacks for lock 14 at 1000, 2000, lock 15 at 3000,
    # 4000; lock 14 tests at 500 (before its first fallback), 1010; lock 15 at 3010, 4010
    log = bytes(32) + struct.pack("<4I", 500, 1010, 3010, 4010) + bytes(16)
    prod = struct.pack("<5I", 100, 1000, 2000, 3000, 4000) + bytes(12)
    assert g.parse_b2b_counts(log, prod) == {g.B2B_FIRST: 1, g.LOCK_ID: 0}
