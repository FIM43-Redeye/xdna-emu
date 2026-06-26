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


def test_eligibility_partitions(tmp_path, monkeypatch):
    # run0,run1 fire A in batch_00; B fires only in run0 (intermittent);
    # C fires in both but in batch_00 (run0) vs batch_01 (run1) -> batch_flip.
    per = {
        "r0": {"batch_00": {"1|2|0|A": 0, "1|2|0|B": 5, "1|2|0|C": 9}},
        "r1": {"batch_00": {"1|2|0|A": 0}, "batch_01": {"1|2|0|C": 9}},
    }
    monkeypatch.setattr(T, "_batch_names", lambda rd: sorted(per[rd]))
    monkeypatch.setattr(T, "batch_firsts", lambda rd, bn, anchor_key=T.ANCHOR: per[rd].get(bn, {}))
    res = T.eligibility(["r0", "r1"], ["1|2|0|A", "1|2|0|B", "1|2|0|C"])
    assert res.clusterable == ["1|2|0|A"]
    assert res.pinned["1|2|0|A"] == "batch_00"
    assert any("1|2|0|B" in pc.events for pc in res.intermittent)
    assert res.excluded.get("1|2|0|C") == T.F_BATCH_FLIP


def test_eligibility_anchor_dropout(tmp_path, monkeypatch):
    # run0 has a batch where anchor fired (event present); run1 has a batch where
    # batch_firsts returns {} (anchor did not fire) -> run1 is a dropout_run and
    # the event present in run0 must NOT be turned intermittent (it stays clusterable).
    per = {
        "r0": {"batch_00": {"1|2|0|A": 0, "1|2|0|PERF_CNT_2": 0}},
        "r1": {"batch_00": {}},  # anchor absent -> dropout batch
    }
    monkeypatch.setattr(T, "_batch_names", lambda rd: sorted(per[rd]))
    monkeypatch.setattr(T, "batch_firsts", lambda rd, bn, anchor_key=T.ANCHOR: per[rd].get(bn, {}))
    res = T.eligibility(["r0", "r1"], ["1|2|0|A"])
    assert 1 in res.dropout_runs
    assert (1, "batch_00") in res.dropout_batches
    # A is present in run0 (the only live run); it must be clusterable, not intermittent
    assert res.clusterable == ["1|2|0|A"]
    assert not any("1|2|0|A" in pc.events for pc in res.intermittent)


def test_rigid_clusters_anchored_group(tmp_path):
    jv = {"1|2|0|A": (0, 0, 0), "1|2|0|B": (0, 0, 0), "1|2|0|C": (0, 3, 1)}
    n = {"1|2|0|A": 3, "1|2|0|B": 3, "1|2|0|C": 3}
    res = T.rigid_clusters(jv, n, set())
    anchored = [f for f in res.frames if not f.floating][0]
    assert set(anchored.members) == {"1|2|0|A", "1|2|0|B"}
    assert res.nondeterministic == ["1|2|0|C"]   # unique non-zero jv


def test_rigid_clusters_floating_needs_corroboration(tmp_path):
    jv = {"1|2|0|X": (0, 4, 1), "1|2|0|Y": (0, 4, 1)}   # shared non-zero
    n = {"1|2|0|X": 3, "1|2|0|Y": 3}                     # below MIN_N_FLOATING
    # No corroboration AND below the N-floor -> demoted to nondeterministic.
    res = T.rigid_clusters(jv, n, set())
    assert sorted(res.nondeterministic) == ["1|2|0|X", "1|2|0|Y"]
    assert not [f for f in res.frames if f.floating]
    # Common-parent corroboration -> emitted as a floating frame (low-N flagged).
    res2 = T.rigid_clusters(jv, n, {("1|2|0|X", "1|1|1|P"), ("1|2|0|Y", "1|1|1|P")})
    fr = [f for f in res2.frames if f.floating][0]
    assert set(fr.members) == {"1|2|0|X", "1|2|0|Y"} and fr.corroborated
    assert T.F_PROVISIONAL_LOW_N in fr.flags
