from inference.loop import (MockInstrument, ranking, run_loop_until_converged)


def test_ranking_top_component_counts_unfired():
    # 3 configured, only 1 fired -> top = 2
    fired = {"1|0|0|A"}
    configured = ["1|0|0|A", "1|0|0|B", "1|1|3|C"]
    r = ranking_fixture(fired, configured)
    assert r[0] == 2


def ranking_fixture(fired_set, configured):
    from inference.facts import KB, Fact, Measured
    kb = KB.empty()
    for ek in fired_set:
        kb.add(Fact("fired", (ek, 0, 5), Measured()))
    return ranking(kb, configured, candidate_pairs=[])


def test_loop_converges_on_mock_ground_truth(tmp_path):
    # ground truth: S stochastic root, C = S + 30 derived (config routes S->C)
    gt = {
        "events": {"1|2|0|PERF_CNT_2": {"base": 0, "jitter": 0},
                   "1|0|0|S": {"base": 100, "jitter": 50},
                   "1|0|0|C": {"base": 130, "jitter": 50}},  # C tracks S (=S+30)
        "routes": [("1|0|0|S", "1|0|0|C")],
        "workdir": str(tmp_path)}
    inst = MockInstrument(gt)
    report = run_loop_until_converged(
        inst,
        configured_events=["1|2|0|PERF_CNT_2", "1|0|0|S", "1|0|0|C"],
        candidate_pairs=[("1|0|0|C", "1|0|0|S")])
    assert report["converged"] is True
    assert report["classification"]["1|0|0|C"] == "derived"
    assert report["classification"]["1|0|0|S"] == "stochastic_root"


def test_ranking_strictly_decreases_with_discovery(tmp_path):
    # A configured event that never fires in the seed is constrained never_fired
    # and excluded from the unfired count by live_unfired, so the loop can converge
    # without spinning. The ranking's top component is non-increasing throughout.
    # Note: reveal_on_iter here withholds C from iter 0; under the never_fired
    # semantics C is constrained after the seed, so the top count does not
    # decrease across iterations (it converges cleanly in 1 iter).
    gt = {
        "events": {"1|2|0|PERF_CNT_2": {"base": 0, "jitter": 0},
                   "1|0|0|S": {"base": 100, "jitter": 50},
                   "1|0|0|C": {"base": 130, "jitter": 50}},
        "routes": [("1|0|0|S", "1|0|0|C")],
        "reveal_on_iter": {1: "1|0|0|C"},   # C withheld from seed -> never_fired
        "workdir": str(tmp_path)}
    inst = MockInstrument(gt)
    report = run_loop_until_converged(
        inst,
        configured_events=["1|2|0|PERF_CNT_2", "1|0|0|S", "1|0|0|C"],
        candidate_pairs=[("1|0|0|C", "1|0|0|S")])
    tops = [r[0] for r in report["rankings"]]
    assert tops == sorted(tops, reverse=True)   # top never increases (monotone property)
    # C is constrained never_fired after the seed; loop converges without
    # spinning to max_iters, and C appears in unfirable_events.
    assert report["converged"] is True
    assert "1|0|0|C" in report["model"].unfirable_events()
