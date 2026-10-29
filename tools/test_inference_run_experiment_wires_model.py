"""SP-5c Phase 1, §9a(b): run_experiment must accept and forward model_path
to run_engine, so the sweep's produced origin_d.json sidecar can eventually
be threaded through. Inert while uncalibrated -- this test asserts the
plumbing (signature + live forwarding), not any behavior change.

The forwarding test is modeled on
test_timeline.py::test_run_experiment_threads_dump_into_engine: build a
KernelConfig + a fake instrument, monkeypatch run_engine to capture kwargs,
call run_experiment, and assert the kwarg arrived."""
import inspect

from inference import run_experiment as re_mod


def test_run_experiment_accepts_model_path():
    sig = inspect.signature(re_mod.run_experiment)
    assert "model_path" in sig.parameters, "run_experiment must accept model_path"


def test_run_experiment_forwards_model_path(tmp_path, monkeypatch):
    import inference.loop as loop_mod
    import inference.engine as eng_mod
    from inference.run_experiment import run_experiment, KernelConfig

    loop_result = {
        "converged": True, "terminal_state": "segment", "iterations": 1,
        "classification": {}, "run_dirs": [str(tmp_path / "run0")],
        "model": type("_Model", (), {"constraints": lambda self: []})(),
    }
    monkeypatch.setattr(loop_mod, "run_loop_until_converged",
                        lambda *a, **kw: loop_result)

    captured = {}

    def fake_run_engine(*args, **kwargs):
        captured.update(kwargs)
        captured["_args"] = args
        return {"derives": [], "segments": [], "gaps": [], "warnings": [],
                "rejected_rules": [], "stochastic_roots": [], "provenance_ok": True,
                "timeline": None}

    # The delegate in run_experiment.py re-imports inference.engine.run_engine
    # fresh per call, so an attribute-patch on the engine module intercepts it.
    monkeypatch.setattr(eng_mod, "run_engine", fake_run_engine)

    class _FakeInstrument:
        def ledger_entries(self): return []

    cfg = KernelConfig(
        test="dummy", compiler="chess", dump_path=None,
        start_col=1, anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
        n_runs=1, out_root=str(tmp_path / "out"),
    )
    run_experiment(cfg, instrument=_FakeInstrument(),
                   configured=["1|2|0|A"], candidate_pairs=[],
                   model_path="fake/origin_d.json")
    assert captured.get("model_path") == "fake/origin_d.json"
