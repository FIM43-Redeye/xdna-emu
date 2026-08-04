"""Pins the extracted runner module's public surface and downstream imports."""
import importlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest


def test_trace_runner_exports_runner_session():
    tr = importlib.import_module("trace_runner")
    for name in ("RunnerSession", "ParseSession", "RunResult", "_run_one_side",
                 "_run_patch", "_run_patch_multi", "_relabel_events",
                 "_parse_trace_bin"):
        assert hasattr(tr, name), f"trace_runner missing {name}"
    for const in ("RUNNER", "PATCH_TOOL", "PARSE_TOOL", "REPO_ROOT",
                  "MLIR_AIE_ROOT", "_MOD_TO_TILE_TYPE", "_MODE_INT"):
        assert hasattr(tr, const), f"trace_runner missing {const}"


def test_runner_session_signature_preserved():
    tr = importlib.import_module("trace_runner")
    sig = inspect.signature(tr.RunnerSession.__init__)
    params = list(sig.parameters)
    assert params[:5] == ["self", "xclbin", "runner_env", "side", "stderr_log"]
    assert "reuse_ctx" in params


def test_runner_session_passes_qos_to_outer_cli(tmp_path, monkeypatch):
    tr = importlib.import_module("trace_runner")
    argv_path = tmp_path / "argv.json"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['ARGV_PATH'], 'w').write(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'event': 'ready', 'pid': os.getpid()}), flush=True)\n"
        "for _ in sys.stdin: pass\n"
    )
    runner.chmod(0o755)
    monkeypatch.setattr(tr, "RUNNER", runner)
    monkeypatch.setenv("ARGV_PATH", str(argv_path))

    with tr.RunnerSession(
        xclbin=Path("fixture.xclbin"), runner_env={}, side="HW",
        stderr_log=tmp_path / "runner.log", qos_gops=1, qos_fps=1800,
    ):
        pass

    assert json.loads(argv_path.read_text()) == [
        "--batch-stdin", "--xclbin", "fixture.xclbin",
        "--qos-gops", "1", "--qos-fps", "1800",
    ]


@pytest.mark.parametrize("qos_gops,qos_fps", [
    (1, None), (None, 1800), (0, 1800), (1, 0), (1 << 32, 1800),
])
def test_runner_session_rejects_incomplete_or_invalid_qos(
    tmp_path, monkeypatch, qos_gops, qos_fps,
):
    tr = importlib.import_module("trace_runner")
    monkeypatch.setattr(tr, "RUNNER", tmp_path / "missing-runner")

    with pytest.raises(ValueError, match="qos_gops and qos_fps"):
        tr.RunnerSession(
            xclbin=Path("fixture.xclbin"), runner_env={}, side="HW",
            stderr_log=tmp_path / "runner.log",
            qos_gops=qos_gops, qos_fps=qos_fps,
        )


@pytest.mark.parametrize("qos_args,message", [
    (["--qos-gops", "1"],
     "--qos-gops and --qos-fps must be provided together"),
    (["--qos-gops", "0", "--qos-fps", "1"],
     "--qos-gops must be a positive 32-bit integer"),
    (["--qos-gops", str(1 << 32), "--qos-fps", "1"],
     "--qos-gops must be a positive 32-bit integer"),
])
def test_bridge_runner_rejects_invalid_qos_before_device_open(qos_args, message):
    tr = importlib.import_module("trace_runner")
    if not tr.RUNNER.is_file():
        pytest.skip("bridge-trace-runner is not built")

    result = subprocess.run(
        [str(tr.RUNNER), "--batch-stdin", *qos_args],
        text=True, capture_output=True,
    )

    assert result.returncode == 1
    assert message in result.stderr


def test_sweep_imports_from_runner():
    # trace-sweep.py is hyphenated; load it by path and confirm it re-exports
    # RunnerSession that IS trace_runner.RunnerSession (same object, not a copy).
    import importlib.util
    import sys
    from pathlib import Path
    tr = importlib.import_module("trace_runner")
    sweep_path = Path(__file__).resolve().parent / "trace-sweep.py"
    spec = importlib.util.spec_from_file_location("_sweep_mod", str(sweep_path))
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module: Python 3.13 dataclass processing calls
    # sys.modules.get(cls.__module__) which returns None for unregistered
    # synthetic modules, causing AttributeError on the frozen EventDef class.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.RunnerSession is tr.RunnerSession


def test_capture_hwrunner_uses_runner_module():
    import trace_capture
    r = trace_capture.HwRunner.__init__
    src = inspect.getsource(r)
    assert "trace_runner" in src, "HwRunner must import RunnerSession from trace_runner"
