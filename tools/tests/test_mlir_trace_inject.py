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


ALREADY_TRACED = FIXTURES / "sample_already_traced.mlir"


def test_injector_bails_on_already_traced(tmp_path):
    """If input already has aie.trace ops, injector should refuse (exit 2)."""
    out = tmp_path / "out.mlir"
    r = _run(
        ["--input", str(ALREADY_TRACED), "--out", str(out)],
        check=False,
    )
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}; stderr={r.stderr}"
    assert "already contains" in r.stderr.lower(), \
        f"stderr should cite 'already contains'; got: {r.stderr}"
    assert not out.exists(), "output file should not be written when injector refuses"


def test_no_op_round_trips_already_traced(tmp_path):
    """--no-op must not trigger the idempotency check (identity pass is always safe)."""
    out = tmp_path / "out.mlir"
    r = _run(
        ["--no-op", "--input", str(ALREADY_TRACED), "--out", str(out)],
        check=False,
    )
    assert r.returncode == 0, f"--no-op on already-traced input failed: stderr={r.stderr}"
    assert out.exists(), "output file should be written in --no-op mode"
    result = out.read_text()
    assert result.count("aie.trace") >= 1, "aie.trace op should survive round-trip"


def test_injector_adds_trace_decl_per_compute_tile(tmp_path):
    """Each non-shim tile in the input should get one aie.trace decl with
    the mandatory body (mode, packet, events, start/stop broadcasts)."""
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    # Fixture has one compute tile (0, 2). Shim tile (0, 0) is not compute.
    # Count aie.trace at the op-start level (not aie.trace.event/etc. sub-ops):
    # the decl form "aie.trace @" is uniquely the outer TraceOp (its body has
    # aie.trace.event but those are indented/nested).
    trace_count = result.count("aie.trace @")
    assert trace_count == 1, f"expected 1 aie.trace decl, got {trace_count}\n---\n{result}"
    # Symbol name should follow trace_t{col}_{row} convention.
    assert "@trace_t0_2" in result, f"sym_name missing; got:\n{result}"
    # Body must contain all the spec-mandated fields. Regressions that drop
    # the mode op, event list, or broadcast channels should fail this test.
    assert "Event-Time" in result, "aie.trace.mode 'Event-Time' missing"
    assert "INSTR_VECTOR" in result, "INSTR_VECTOR event missing"
    assert "INSTR_EVENT_0" in result, "INSTR_EVENT_0 event missing"
    assert "INSTR_EVENT_1" in result, "INSTR_EVENT_1 event missing"
    # start broadcast=15, stop broadcast=14 (mlir-aie defaults).
    assert "15" in result and "14" in result, "broadcast channels 15/14 missing"


def test_injector_adds_runtime_sequence_trace_config(tmp_path):
    """The aie.runtime_sequence body should start with trace.host_config +
    one trace.start_config per trace decl, before existing runtime ops."""
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(UNTRACED),
        "--out", str(out),
        "--buffer-size", "16384",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "aie.trace.host_config" in result, f"host_config missing in output:\n{result}"
    assert "aie.trace.start_config" in result, f"start_config missing in output:\n{result}"
    # Start_config should reference the trace symbol by name.
    assert "@trace_t0_2" in result, "start_config should reference @trace_t0_2"
    # Custom buffer size should flow through.
    assert "16384" in result, "custom --buffer-size did not reach the output"


def test_injector_default_buffer_size_used_when_not_specified(tmp_path):
    """If --buffer-size is omitted, the default (8192) should appear."""
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "aie.trace.host_config" in result
    assert "8192" in result, f"default buffer size missing:\n{result}"
