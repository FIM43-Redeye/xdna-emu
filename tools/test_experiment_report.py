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
    constraints = res["model"]._constraints
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
