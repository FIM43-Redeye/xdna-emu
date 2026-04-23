"""Tests for tools/trace-to-cycles.py."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXTRACTOR = REPO / "tools" / "trace-to-cycles.py"
FIXTURES = REPO / "tools" / "tests" / "fixtures"
TRACE_JSON = FIXTURES / "sample_traced_trace.json"


def _run(args, check=True):
    return subprocess.run(
        [sys.executable, str(EXTRACTOR), *args],
        capture_output=True, text=True, check=check,
    )


def test_extractor_help():
    r = _run(["--help"])
    assert "usage" in r.stdout.lower() or "usage" in r.stderr.lower()


def test_extractor_reads_json_and_emits_cycles(tmp_path):
    """Given a canned Perfetto JSON, extractor should emit one integer line."""
    out = tmp_path / "cycles.txt"
    r = _run(["--trace-json", str(TRACE_JSON), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    content = out.read_text().strip()
    assert content == "410", f"expected '410', got {content!r}"


def test_extractor_requires_xclbin_mlir_with_trace_bin(tmp_path):
    """--trace-bin requires --xclbin-mlir to tell parse_trace the layout."""
    fake_bin = tmp_path / "trace.bin"
    fake_bin.write_bytes(b"\x00" * 16)
    out = tmp_path / "cycles.txt"
    r = _run(
        ["--trace-bin", str(fake_bin), "--out", str(out)],
        check=False,
    )
    assert r.returncode != 0, f"expected failure, got success: stdout={r.stdout}"
    assert "xclbin-mlir" in r.stderr.lower() or "required" in r.stderr.lower()


def test_extractor_rejects_empty_json(tmp_path):
    """Trace JSON with no timestamped events should fail cleanly."""
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    out = tmp_path / "cycles.txt"
    r = _run(
        ["--trace-json", str(empty), "--out", str(out)],
        check=False,
    )
    assert r.returncode != 0
