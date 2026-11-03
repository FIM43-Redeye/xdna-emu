"""tools/probe_to_suite.py: a flat recorded probe dir becomes a suite of the same kernels."""
import json
from pathlib import Path

import probe_to_suite as P
from harness import suite as S


def legacy(tmp_path):
    d = tmp_path / "old"
    d.mkdir()
    (d / "input.bin").write_bytes(bytes(range(16)))
    kernels = [{"name": f"k{i:03d}", "points": [i], "in_size": 16, "out_size": 8 * (i + 1)} for i in range(2)]
    (d / "manifest.json").write_text(json.dumps({"in_size": 16, "record_bytes": 8, "kernels": kernels}))
    for k in kernels:
        (d / f"{k['name']}.s").write_text(f"; {k['name']}\n")
        (d / "r1").mkdir(exist_ok=True)
        (d / "r1" / f"{k['name']}.bin").write_bytes(b"\x05" * k["out_size"])
    return d, kernels


def test_convert_keeps_kernels_input_sizes_and_metadata(tmp_path):
    d, kernels = legacy(tmp_path)
    assert P.convert(d, tmp_path / "new") == 2
    s = S.load_suite(tmp_path / "new")
    assert s.cases == ("k000", "k001") and s.meta == {"in_size": 16, "record_bytes": 8}
    assert s.case_entries()[1] == kernels[1]
    core = s.topology("k001").cores[0]
    assert core.source.read_text() == "; k001\n"
    assert core.args[0].data == bytes(range(16)) and core.args[1].size_bytes == 16


def test_legacy_bin_matches_reports_differences(tmp_path):
    d, _ = legacy(tmp_path)
    P.convert(d, tmp_path / "new")
    for c, fill in (("k000", 5), ("k001", 6)):
        out = tmp_path / "new" / "runs" / "r1" / c
        out.mkdir(parents=True)
        (out / "core.out.bin").write_bytes(bytes([fill]) * (8 if c == "k000" else 16))
    assert P.legacy_bin_matches(d, "r1", tmp_path / "new", "r1") == (2, ["k001"])
