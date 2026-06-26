from inference import timeline as T


def test_characterize_event_rigid_then_jittery(tmp_path, monkeypatch):
    # 3 runs; occ index 0 rigid (cycle 10), index 1 jitters (26/26/28).
    occ = {"1|1|3|P": [[10, 26], [10, 26], [10, 28]]}
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1", "r2"], "1|1|3|P", "batch_00")
    assert er.n_eff == 3
    assert er.rigid_runs == [T.RigidRun(start_index=0, cycles=[10])]
    assert er.jitter_points == [T.JitterPoint(index=1, window=(26, 28))]
    assert T.F_COUNT_WINDOW not in er.flags  # counts equal (all length 2)


def test_characterize_event_jittery_first_then_steady(tmp_path, monkeypatch):
    # index 0 jitters (5/7), indices 1..2 rigid -> recovered, not discarded.
    occ = {"1|1|3|P": [[5, 20, 40], [7, 20, 40]]}
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1"], "1|1|3|P", "batch_00")
    assert er.jitter_points == [T.JitterPoint(index=0, window=(5, 7))]
    assert er.rigid_runs == [T.RigidRun(start_index=1, cycles=[20, 40])]


def test_characterize_event_count_window_and_reorderable(tmp_path, monkeypatch):
    occ = {"1|1|3|P": [[10, 26], [10], [26, 10]]}  # counts differ; run2 not increasing
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1", "r2"], "1|1|3|P", "batch_00")
    assert T.F_COUNT_WINDOW in er.flags
    assert T.F_REORDERABLE in er.flags


def test_characterize_event_span_begin_and_length(tmp_path, monkeypatch):
    # PORT_RUNNING_0 is a level event -> firings pair into (begin,length) spans.
    # run0 [10,26, 50,70] -> (10,16),(50,20); run1 [10,26, 50,72] -> (10,16),(50,22)
    occ = {"1|1|3|PORT_RUNNING_0": [[10, 26, 50, 70], [10, 26, 50, 72]]}
    monkeypatch.setattr(T, "anchored_occurrences_per_run",
                        lambda runs, key, pb, anchor_key=T.ANCHOR: occ[key])
    er = T.characterize_event(["r0", "r1"], "1|1|3|PORT_RUNNING_0", "batch_00")
    # span 0 fully rigid (begin 10, length 16); span 1 begin-rigid but length jitters.
    assert er.rigid_runs == [T.RigidRun(start_index=0, cycles=[10], lengths=[16])]
    assert er.jitter_points == [T.JitterPoint(index=1, window=(50, 50), length_window=(20, 22))]


def test_data_model_constructs():
    er = T.EventRecord(key="1|2|0|A", domain="1|2|0", pinned_batch="batch_00", n_eff=8,
                       rigid_runs=[T.RigidRun(start_index=0, cycles=[0, 16])])
    dp = T.DeterministicPeriod(events=["1|2|0|A"], cycles={"1|2|0|A": 0},
                               grounding_event="1|2|0|A", floating=False)
    ndp = T.NondeterministicPeriod(events=["1|2|0|B"], windows={"1|2|0|B": (5, 9)},
                                   reasons={"1|2|0|B": "within_domain_nonexact"},
                                   order_edges=[], grounding_event=None,
                                   flags=[T.F_UNGROUNDED_TAIL])
    tl = T.IntegratedTimeline(tracks=[T.Track(domain="1|2|0", periods=[dp, ndp])],
                              cross_track_edges=[], intermittent=[], flags=[],
                              census=T.Census(events={}, edges={}, content_ok=True),
                              capture={"n_runs": 8})
    assert tl.tracks[0].periods[0].grounding_event == "1|2|0|A"
    assert T.MIN_N_FLOATING == 12 and T.CENSUS_CONTENT_FLOOR == 0.5
    assert er.flags == [] and ndp.flags == [T.F_UNGROUNDED_TAIL]
