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


def test_internal_cycles_from_anchored(tmp_path):
    frame = T.ClusterFrame(members=["1|2|0|A", "1|2|0|B"], floating=False, corroborated=True)
    g, cyc = T.internal_cycles(frame, {"1|2|0|A": 100, "1|2|0|B": 116})
    assert g == "1|2|0|A" and cyc == {"1|2|0|A": 0, "1|2|0|B": 16}


def test_internal_cycles_violation_raises(tmp_path, monkeypatch):
    frame = T.ClusterFrame(members=["1|0|0|A", "1|0|0|B", "1|0|0|C"], floating=False, corroborated=True)
    monkeypatch.setattr(T, "additivity_state", lambda runs, chain, anchor_key=T.ANCHOR: "violation")
    import pytest
    with pytest.raises(T.ClusterViolation):
        T.internal_cycles(frame, {"1|0|0|A": 0, "1|0|0|B": 5, "1|0|0|C": 99},
                          run_dirs=["r0"])


# ---------------------------------------------------------------------------
# Task 8: build_track
# ---------------------------------------------------------------------------

def test_build_track_grounded():
    # (a) Two anchored frames A (anchor_pos=100) and C (anchor_pos=200), with a
    # nondeterministic event X between them (mean_pos=150).
    # derives_pairs attests X -> G_C so F_RESUMPTION_UNATTESTED must NOT appear.
    G_A = "1|2|0|G_A"
    G_C = "1|2|0|G_C"
    X   = "1|2|0|X"

    frames = [
        (G_A, {G_A: 0}, False, 100),
        (G_C, {G_C: 0}, False, 200),
    ]
    nondet_windows = {X: (140, 160)}
    mean_pos       = {G_A: 100.0, G_C: 200.0, X: 150.0}
    derives_pairs  = {(X, G_C)}   # attests the nondet period is not ungrounded

    track = T.build_track("1|2|0", frames, nondet_windows, mean_pos, derives_pairs)

    assert len(track.periods) == 3
    det_a, nondet_x, det_c = track.periods

    # Period 0: deterministic for frame A
    assert isinstance(det_a, T.DeterministicPeriod)
    assert det_a.grounding_event == G_A
    assert det_a.offset_to_prior_frame is None          # first frame -> no prior

    # Period 1: nondeterministic for X, closed by C's grounding event
    assert isinstance(nondet_x, T.NondeterministicPeriod)
    assert nondet_x.events == [X]
    assert nondet_x.windows[X] == (140, 160)
    assert nondet_x.grounding_event == G_C
    assert T.F_UNGROUNDED_TAIL not in nondet_x.flags
    assert T.F_RESUMPTION_UNATTESTED not in nondet_x.flags   # attested via derives_pairs

    # Period 2: deterministic for frame C
    assert isinstance(det_c, T.DeterministicPeriod)
    assert det_c.grounding_event == G_C
    # Both frames anchored -> exact (x,x) interval subtraction: 200-100, 200-100
    assert det_c.offset_to_prior_frame == (100, 100)


def test_build_track_ungrounded_tail():
    # (b) One anchored frame, then a trailing nondeterministic event with no
    # following frame.  The NondeterministicPeriod must carry F_UNGROUNDED_TAIL
    # and have grounding_event is None.
    G = "1|2|0|G"
    Y = "1|2|0|Y"

    frames         = [(G, {G: 0}, False, 50)]
    nondet_windows = {Y: (20, 40)}
    mean_pos       = {G: 50.0, Y: 80.0}
    derives_pairs  = set()

    track = T.build_track("1|2|0", frames, nondet_windows, mean_pos, derives_pairs)

    assert len(track.periods) == 2
    det_g, nondet_y = track.periods

    assert isinstance(det_g, T.DeterministicPeriod)
    assert det_g.grounding_event == G

    assert isinstance(nondet_y, T.NondeterministicPeriod)
    assert nondet_y.grounding_event is None
    assert T.F_UNGROUNDED_TAIL in nondet_y.flags


def test_build_track_floating_frame():
    # (c) Anchored frame A (anchor_pos=100) followed by floating frame B
    # (anchor_pos=(180,210)).  offset_to_prior_frame must be a window (80,110):
    #   lo = 180 - 100 = 80, hi = 210 - 100 = 110  (interval subtraction).
    G_A = "1|2|0|G_A"
    G_B = "1|2|0|G_B"

    frames = [
        (G_A, {G_A: 0}, False, 100),
        (G_B, {G_B: 0}, True,  (180, 210)),
    ]
    nondet_windows = {}
    mean_pos       = {G_A: 100.0, G_B: 195.0}
    derives_pairs  = set()

    track = T.build_track("1|2|0", frames, nondet_windows, mean_pos, derives_pairs)

    assert len(track.periods) == 2
    det_a, det_b = track.periods

    assert isinstance(det_a, T.DeterministicPeriod)
    assert det_a.offset_to_prior_frame is None

    assert isinstance(det_b, T.DeterministicPeriod)
    assert det_b.floating is True
    # interval subtraction: (180-100, 210-100) = (80, 110)
    assert det_b.offset_to_prior_frame == (80, 110)


# ---------------------------------------------------------------------------
# Task 9: order_nondeterministic
# ---------------------------------------------------------------------------

def test_order_nondeterministic():
    # Three events: A, B, C.
    # Causal edge: derives_pairs = {("B", "A")} means A is parent of B ->
    #   emitted as ("A", "B", "causal").
    # Stable-position edge: stable_before[("B", "C")] = True ->
    #   emitted as ("B", "C", "stable_position") (no causal edge for this pair).
    # Concurrent pair: (A, C) -- no derives edge, no stable_before entry -> omitted.

    events = ["A", "B", "C"]
    derives_pairs = {("B", "A")}                # (child, parent)
    stable_before = {("B", "C"): True}

    edges = T.order_nondeterministic(events, derives_pairs, stable_before)

    # Both expected edges must be present with correct tags.
    assert ("A", "B", "causal") in edges
    assert ("B", "C", "stable_position") in edges

    # Concurrent pair (A, C) must be absent in both directions.
    assert ("A", "C", "causal") not in edges
    assert ("A", "C", "stable_position") not in edges
    assert ("C", "A", "causal") not in edges
    assert ("C", "A", "stable_position") not in edges

    # Exactly two edges total (the causal edge prevents stable_position from
    # appearing for the same pair, and the concurrent pair is omitted entirely).
    assert len(edges) == 2
