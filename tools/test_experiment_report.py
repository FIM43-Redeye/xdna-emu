"""The loop's empirical-limit folding, exercised with MockInstrument ground
truth (no NPU). An uncorrelated pair must HALT honestly (not spin to max_iters);
a never-fired configured event must be constrained and not block convergence."""
from inference.loop import MockInstrument, run_loop_until_converged
from inference.verifier import ANCHOR


def _gt(workdir, *, uncorrelated=False, with_unfired=False):
    # Two memtile ports + the anchor on the core tile. PR0 -> PR4 is a route.
    events = {
        ANCHOR: {"base": 0, "jitter": 0},
        "1|1|3|PORT_RUNNING_0": {"base": 10, "jitter": 0},
        "1|1|3|PORT_RUNNING_4": {"base": 40, "jitter": 1 if uncorrelated else 0},
    }
    gt = {"events": events, "routes": [("1|1|3|PORT_RUNNING_0",
                                        "1|1|3|PORT_RUNNING_4")],
          "workdir": str(workdir)}
    return gt


def test_correlated_pair_converges(tmp_path):
    inst = MockInstrument(_gt(tmp_path), n_runs=6)
    configured = [ANCHOR, "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4"]
    pairs = [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")]
    res = run_loop_until_converged(inst, configured, pairs)
    assert res["converged"] is True
    assert res["terminal_state"] == "placed"


def test_uncorrelated_pair_halts_falsifiably_not_spins(tmp_path):
    inst = MockInstrument(_gt(tmp_path, uncorrelated=True), n_runs=6)
    configured = [ANCHOR, "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4"]
    pairs = [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")]
    res = run_loop_until_converged(inst, configured, pairs, max_iters=12)
    # It must NOT exhaust max_iters spinning; a cannot_correlate constraint
    # with measured provenance must be recorded.
    assert res["iterations"] < 12
    assert res["terminal_state"] in ("placed", "halted_falsifiable")
    constraints = res["model"].constraints()
    assert any(c.predicate == "cannot_correlate"
               and c.provenance_batch is not None for c in constraints)


def test_never_fired_event_is_constrained_and_excluded(tmp_path):
    gt = _gt(tmp_path)
    # A configured event that is NOT in the ground-truth event set -> never fires.
    configured = [ANCHOR, "1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4",
                  "1|1|3|PORT_RUNNING_7"]
    pairs = [("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0")]
    inst = MockInstrument(gt, n_runs=6)
    res = run_loop_until_converged(inst, configured, pairs, max_iters=12)
    # PR7 never fires; it must be constrained never_fired (with provenance) and
    # not prevent convergence on the rest.
    assert "1|1|3|PORT_RUNNING_7" in res["model"].unfirable_events()
    assert res["converged"] is True


def test_run_experiment_with_mock_writes_report(tmp_path):
    from inference.run_experiment import KernelConfig, run_experiment, write_report
    from inference.loop import MockInstrument
    from inference.verifier import ANCHOR
    import json

    gt = _gt(tmp_path)
    inst = MockInstrument(gt, n_runs=6)
    cfg = KernelConfig(test="add_one_using_dma", compiler="chess",
                       dump_path=None, start_col=1, anchor_tile_abs="1|2|0",
                       anchor_event="PERF_CNT_2", traced_col=1, n_runs=6,
                       out_root=str(tmp_path / "out"))
    # Inject configured/pairs directly via the mock-test override hook.
    report = run_experiment(cfg, instrument=inst,
                            configured=[ANCHOR, "1|1|3|PORT_RUNNING_0",
                                        "1|1|3|PORT_RUNNING_4"],
                            candidate_pairs=[("1|1|3|PORT_RUNNING_4",
                                              "1|1|3|PORT_RUNNING_0")])
    assert report["terminal_state"] == "placed"
    assert report["engine_ok"] is True
    assert "classification" in report and "constraints" in report
    out = tmp_path / "report.json"
    write_report(report, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["kernel"] == "add_one_using_dma"
    # Every recorded constraint carries its provenance batch (falsifiability).
    assert all(c["provenance_batch"] for c in loaded["constraints"])
    assert "segments" in loaded and "gaps" in loaded
    assert "rejected_rules" in loaded
    # The canary surfaces through run_experiment: a clean correlated pair (PR0->PR4,
    # exact offset, a segment) raises no unaccounted-gap warning.
    assert loaded["warnings"] == []
