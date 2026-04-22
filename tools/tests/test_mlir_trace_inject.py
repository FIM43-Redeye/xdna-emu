"""Tests for tools/mlir-trace-inject.py."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INJECTOR = REPO / "tools" / "mlir-trace-inject.py"
FIXTURES = REPO / "tools" / "tests" / "fixtures"
UNTRACED = FIXTURES / "sample_untraced.mlir"


def _run(args, check=True):
    # Use sys.executable so the subprocess inherits the same Python that is
    # running pytest.  The injector imports aie.ir which requires the ironenv
    # Python (not the system python3) -- using sys.executable ensures the test
    # works whether pytest was invoked via the activated venv or via a direct
    # path to the ironenv interpreter.
    return subprocess.run(
        [sys.executable, str(INJECTOR), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def test_injector_exists_and_prints_help():
    r = _run(["--help"])
    assert "usage" in r.stdout.lower() or "usage" in r.stderr.lower()


def test_injector_no_op_mode_round_trips(tmp_path):
    """With --no-op, injector should read and write the MLIR unchanged."""
    out = tmp_path / "out.mlir"
    r = _run(["--no-op", "--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"injector failed: stderr={r.stderr}"
    original = UNTRACED.read_text()
    result = out.read_text()
    # The mlir-aie parser may normalize whitespace; compare parsed structure,
    # not raw text. At minimum, tile count and op kinds should match.
    assert result.count("aie.tile") == original.count("aie.tile")
    assert result.count("aie.device") == original.count("aie.device")
