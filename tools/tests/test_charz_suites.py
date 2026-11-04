"""The characterizer suite generator and the analyzers that read its runner layout."""
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import charz_suites as C
from harness import suite as S
from harness.build import BuildError, _peano_tool
from harness.run import prng_bytes

TESTS = C.TESTS
bf16_elem = C.load_test_module("bf16_elemwise_analyze.py")
subprec_an = C.load_test_module("bf16_sub_precision_analyze.py")
subprec_gen = C.load_test_module("bf16_sub_precision_gen.py")
vldb = C.load_test_module("vldb_4x_analyze.py")
smask = C.load_test_module("sparse-mask-characterize.py")
smac = C.load_test_module("sparse-mac-analyze.py")


# -- generator -----------------------------------------------------------------

def test_generate_writes_one_case_per_characterizer(tmp_path):
    C.generate(tmp_path)
    s = S.load_suite(tmp_path)
    assert s.name == "charz" and s.cases == tuple(c.case for c in C.CHARACTERIZERS)
    assert s.hw_checks == ("aligned_labels",)
    for c in C.CHARACTERIZERS:
        (core,) = s.topology(c.case).cores
        assert core.source.read_text() == (TESTS / c.asm).read_text()
        assert [(b.name, b.direction, b.size_bytes) for b in core.args] == [
            ("in", "in", c.in_size), ("out", "out", c.out_size)]
        assert core.args[0].data == c.input()
        assert (TESTS / c.analyzer).is_file()
        assert s.case_meta(c.case)["analyzer"] == f"tests/{c.analyzer}"
        assert s.output("r1", c.case).name == "core.out.bin"


def test_inputs_are_what_the_retired_scripts_fed():
    # run-sparse-*.sh and run_vldb_4x_characterizer.sh: `test_host --seed 42`
    assert C.CHARACTERIZERS[0].input() == prng_bytes(32, 42)
    assert C.CHARACTERIZERS[4].input() == vldb.fill_prng(256, 42)
    # run_bf16_elemwise_characterizer.sh: [Sidon][ones][ones][Sidon] as bf16
    data = C.CHARACTERIZERS[2].input()
    assert len(data) == 256 and data[:8] == struct.pack("<4H", 0x3F80, 0x4000, 0x4080, 0x4100)
    assert data[64:128] == data[128:192] == struct.pack("<H", 0x3F80) * 32 and data[192:] == data[:64]


def test_subprec_input_is_the_old_gen_script_output(tmp_path):
    out = tmp_path / "in.bin"
    r = subprocess.run([sys.executable, str(TESTS / "bf16_sub_precision_gen.py"), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and out.read_bytes() == subprec_gen.build_input() and len(out.read_bytes()) == 384


def test_no_kernel_touches_the_reserved_r8_as_scratch():
    # r8 is the saved sp of the ISA-test convention; sparse-mac-probe once used it as a zero register
    import re
    for c in C.CHARACTERIZERS:
        text = (TESTS / c.asm).read_text()
        if c.case == "bf16-subprec":        # saves sp in r8 itself, the ISA-test convention
            continue
        uses = [l for l in text.splitlines() if re.search(r"\br8\b", l)]
        assert not uses, (c.case, uses[:3])


@pytest.mark.parametrize("char", C.CHARACTERIZERS, ids=lambda c: c.case)
def test_kernel_assembles(char, tmp_path):
    try:
        mc = _peano_tool("llvm-mc")
    except BuildError:
        pytest.skip("llvm-mc not available")
    r = subprocess.run([mc, "-triple=aie2", "-filetype=obj", str(TESTS / char.asm), "-o", str(tmp_path / "k.o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


# -- reading results --------------------------------------------------------------

def test_read_outputs_takes_suite_tags_or_plain_files(tmp_path):
    C.generate(tmp_path)
    for tag, byte in (("r1", b"\x01"), ("e1", b"\x02")):
        p = S.output_path(tmp_path, tag, "vldb-4x")
        p.parent.mkdir(parents=True)
        p.write_bytes(byte * 4)
    assert C.read_outputs(["--suite", str(tmp_path), "r1", "e1"], "vldb-4x") == [b"\x01" * 4, b"\x02" * 4]
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert C.read_outputs([str(f)], "vldb-4x") == [b"abc"]
    assert C.suite_input(["--suite", str(tmp_path), "r1"], "vldb-4x") == prng_bytes(256, 42)
    assert C.suite_input([str(f)], "vldb-4x") is None
    with pytest.raises(SystemExit):
        C.read_outputs(["--suite", str(tmp_path)], "vldb-4x")


# -- analyzers ------------------------------------------------------------------------

def test_sparse_mask_counts_active_inner_positions():
    data = b"".join(struct.pack("<32h", *([16 * (p % 5)] * 32)) for p in range(16))
    counts = smask.active_inner_positions(data)
    assert counts[7] == [2] * 8 and counts[4] == [4] * 8 and counts[5] == [0] * 8


def fp32_halves(values):
    """Pack fp32 lanes the way the kernels store them: 16 hi halves then 16 lo halves."""
    bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]
    return (struct.pack("<16H", *[(b >> 16) & 0xFFFF for b in bits])
            + struct.pack("<16H", *[b & 0xFFFF for b in bits]))


def test_bf16_elemwise_recovers_the_index_pairs_and_verifies_the_cross_check(capsys, monkeypatch):
    S_ = bf16_elem.SIDON
    a = [(2 * lane, 2 * lane + 1) for lane in range(16)]
    b = [(lane, 16 + lane) for lane in range(16)]
    t1 = [float(S_[i] + S_[j]) for i, j in a]
    t2 = [float(S_[i] + S_[j]) for i, j in b]
    t3 = [float(S_[a[l][0]] * S_[b[l][0]] + S_[a[l][1]] * S_[b[l][1]]) for l in range(16)]
    out = fp32_halves(t1) + fp32_halves(t2) + fp32_halves(t3)
    assert len(out) == 192
    monkeypatch.setattr(sys, "argv", ["x", "--suite", "unused"])
    monkeypatch.setattr(bf16_elem, "read_outputs", lambda argv, case: [out, out])
    bf16_elem.main()
    text = capsys.readouterr().out
    assert "ALL LANES VERIFIED" in text and "MISMATCH" not in text
    assert "Lane  3: A[ 6], A[ 7]" in text


def test_subprec_reads_each_test_at_its_128_byte_slot(capsys):
    inp = subprec_gen.build_input()
    hw = bytearray(512)
    for t in range(4):
        hw[t * 128:t * 128 + 64] = bytes([t + 1]) * 64
    emu = bytearray(hw)
    emu[3 * 128 + 32 + 2 * 5] ^= 0x04          # test 4, lane 5, low half
    emu[1 * 128 + 2 * 15] ^= 0x01              # test 2, lane 15, high half: needs all 16 lanes read
    assert subprec_an.analyze_test(4, inp[128:256], bytes(hw[384:448]), bytes(emu[384:448]), "t") == 1
    capsys.readouterr()
    import io
    old = sys.argv
    try:
        d = Path(__import__("tempfile").mkdtemp())
        (d / "i").write_bytes(inp); (d / "h").write_bytes(bytes(hw)); (d / "e").write_bytes(bytes(emu))
        sys.argv = ["x", str(d / "i"), str(d / "h"), str(d / "e")]
        assert subprec_an.main() == 2
    finally:
        sys.argv = old
    assert "TOTAL: 2 differing lanes" in capsys.readouterr().out


def test_subprec_input_offsets_share_the_first_vector():
    assert [subprec_an.input_offset(t) for t in range(4)] == [0, 0, 128, 256]


def test_vldb_lut_rule_matches_the_emulator_rule():
    # lane value 0x...20 + odd lane: bit 4 set; low bits reduced to the granule then halved
    assert vldb.lut_lane_address(0x40, 0, 0xFFFFFFF8) == 0x40
    assert vldb.lut_lane_address(0x40, 1, 0xFFFFFFF8) == 0x50
    assert vldb.lut_lane_address(0x4B, 0, 0xFFFFFFF8 & 0x1F) == 0x40 | ((0x0B & 0x18) >> 1)


def test_vldb_lut_prediction_cuts_the_row_and_zero_fills():
    data = bytes(range(256))
    # base 0x100, lane 1 of value 0x100 -> eff 0x110; row 0x110..0x120; 8 bytes
    assert vldb.predict_gather_lut(data, 0x100, [0x100, 0x100], 0xFFFFFFF8) == [
        *struct.unpack("<2I", data[0:8]), *struct.unpack("<2I", data[0x10:0x18])]
    # value 0x11C, even lane, 4-byte granule: eff = 0x100 | (0x1C >> 1) = 0x10E reads two bytes of
    # row 0x100..0x10F, the other six are past the row and read as zero
    out = vldb.predict_gather_lut(data, 0x100, [0x11C], 0xFFFFFFFC)
    assert out == list(struct.unpack("<2I", data[0x0E:0x10] + bytes(6)))


def test_sparse_mac_expectations_and_compare(capsys):
    ident = struct.pack("<32h", *[8] * 32)
    count = struct.pack("<32h", *([36] + [0] * 31))
    good = ident + count
    assert smac.header_expectations(smac.decode(good)) == {
        "kernel header expects identity: every lane == 8": True, "kernel header expects counting: lane 0 == 36": True}
    assert smac.decode(good + b"pad")[1][0] == 36
    with pytest.raises(ValueError):
        smac.decode(good[:100])
    bad = ident + struct.pack("<32h", *([35] + [0] * 31))
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "a").write_bytes(good); (d / "b").write_bytes(bad)
    assert smac.main([str(d / "a"), str(d / "b")]) == 1
    assert "lane  0: second=     35  first=     36" in capsys.readouterr().out
    assert smac.main([str(d / "a")]) == 0
