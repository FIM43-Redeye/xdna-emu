"""Pins the extracted runner module's public surface and downstream imports."""
import importlib
import inspect


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
