from inference import timeline as T

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
