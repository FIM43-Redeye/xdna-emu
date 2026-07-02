"""SP-5c Phase 1, §9a(b): run_experiment must accept and forward model_path
to run_engine, so the sweep's produced origin_d.json sidecar can eventually
be threaded through. Inert while uncalibrated -- this test only asserts the
plumbing (signature + forwarding), not any behavior change."""
import inspect

from inference import run_experiment as re_mod


def test_run_experiment_forwards_model_path(monkeypatch):
    captured = {}

    def fake_run_engine(*args, **kwargs):
        captured.update(kwargs)
        captured["_args"] = args
        return {"ok": True}

    monkeypatch.setattr(re_mod, "run_engine", fake_run_engine)
    # run_experiment must accept model_path and forward it to run_engine.
    sig = inspect.signature(re_mod.run_experiment)
    assert "model_path" in sig.parameters, "run_experiment must accept model_path"
