"""tools/diff_probe_points.py reads the suite result layout."""
import subprocess
import sys
from pathlib import Path

from harness import suite as S

TOOL = Path(__file__).resolve().parents[1] / "diff_probe_points.py"


def test_counts_differing_records_per_kind(tmp_path):
    pts = [{"kind": "pair", "id": 0}, {"kind": "control", "id": 1}, {"kind": "pair", "id": 2}]
    entry = {"name": "k000", "points": pts, "in_size": 16, "out_size": 24}
    S.write_single_core_case(tmp_path, "k000", "; a\n", input_bytes=bytes(16), out_bytes=24, meta=entry)
    S.write_suite_json(tmp_path, "d", "t", ["k000"], meta={"record_bytes": 8})
    for tag, data in (("r1", bytes(24)), ("e1", bytes(8) + b"\x01" + bytes(7) + bytes(8))):
        d = tmp_path / "runs" / tag / "k000"
        d.mkdir(parents=True)
        (d / "core.out.bin").write_bytes(data)
    out = subprocess.run([sys.executable, str(TOOL), str(tmp_path), "r1", "e1"],
                         capture_output=True, text=True, check=True).stdout
    assert "DIFFER 1/3 (control 1/1, pair 0/2)" in out
