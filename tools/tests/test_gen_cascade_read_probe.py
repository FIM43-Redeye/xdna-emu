"""Tests for tools/gen_cascade_read_probe.py: program expansion, layout, assembly."""
import struct
import subprocess

import pytest

import gen_cascade_read_probe as r
from harness.build import _peano_tool


def peano_mc():
    try:
        return _peano_tool("llvm-mc")
    except Exception:
        return None


def test_expand_program_replaces_markers_and_stores_timestamps_last():
    asm, n_ts, n_st = r.expand_program(["@ts", "  vmov x0, SCD", "@ts", "@store", "@ts"])
    assert (n_ts, n_st) == (3, 1)
    text = "\n".join(asm)
    assert text.count("cntr") == 3 and "@" not in text
    # the timestamp stores come after every measured bundle
    assert asm.index("  st r16, [p0, #0]") > max(i for i, l in enumerate(asm) if "cntr" in l)


def test_too_many_timestamps_rejected():
    with pytest.raises(ValueError):
        r.expand_program(["@ts"] * (r.TS_SLOTS + 1))


@pytest.mark.parametrize("case", ["use_3", "nouse_4", "indep_2", "bm_5", "waw_4", "two_1"])
def test_every_case_has_a_known_layout(case):
    prog, n_ts, n_st = r.expand_program(r.case_program(case))
    assert n_ts <= r.TS_SLOTS
    assert r.consumer_out_bytes(case) == r.TS_BYTES + r.reads_per_case(case) * r.WORD
    assert sum("SCD" in l for l in r.case_program(case)) == r.reads_per_case(case)


def test_unknown_case_rejected():
    with pytest.raises(ValueError):
        r.case_program("nope_1")


def test_parse_result_deltas_and_words():
    case = "use_2"
    _, n_ts, _ = r.expand_program(r.case_program(case))
    out = struct.pack(f"<{n_ts}I", 100, 104, 9000) + bytes(r.TS_BYTES - 4 * n_ts)
    out += struct.pack("<16I", *range(16)) * r.reads_per_case(case)
    res = r.parse_result(case, out)
    assert res.deltas == [4, 8896] and res.words[0][3] == 3


@pytest.mark.skipif(peano_mc() is None, reason="Peano llvm-mc not available")
@pytest.mark.parametrize("case", ["use_3", "bm_2", "two_1", "indep_2", "waw_4", "nouse_4"])
@pytest.mark.parametrize("fn", ["consumer", "producer"])
def test_asm_assembles(case, fn, tmp_path):
    text = r.consumer_asm(case, 100) if fn == "consumer" else r.producer_asm(2, 100)
    s = tmp_path / "k.s"
    s.write_text(text)
    res = subprocess.run([peano_mc(), "--triple=aie2", "--filetype=obj", "-o", str(tmp_path / "k.o"), str(s)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_cli_generate_writes_a_loadable_suite(tmp_path):
    from harness import suite
    assert r.main([str(tmp_path), "--cases", "use_3", "two_2", "--modes", "present"]) == 0
    s = suite.load_suite(tmp_path)
    assert s.cases == ("rd_use_3_present_south", "rd_two_2_present_south")
    assert s.case_meta("rd_two_2_present_south")["mode"] == "present"
    assert s.topology("rd_use_3_present_south").cores[1].source.read_text()
