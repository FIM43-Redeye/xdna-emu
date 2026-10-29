import inference.timeline as T
from inference.timeline import CrossTrackEdge, render_timeline, IntegratedTimeline


def test_cross_track_edge_carries_causal_offset():
    e = CrossTrackEdge(child="c", parent="p", reason="cross_domain",
                       reproduction_offset=1, causal_offset=7)
    assert e.causal_offset == 7


def test_render_omits_causal_when_none():
    e = CrossTrackEdge("c", "p", "cross_domain", 1, None)
    tl = IntegratedTimeline(tracks=[], cross_track_edges=[e], intermittent=[],
                            flags=[], census=T.census_of([], [], {}, [e]), capture={})
    out = render_timeline(tl)
    assert "reproduction_offset=1" in out and "causal_offset" not in out


def test_render_shows_causal_when_present():
    e = CrossTrackEdge("c", "p", "cross_domain", 1, 7)
    tl = IntegratedTimeline(tracks=[], cross_track_edges=[e], intermittent=[],
                            flags=[], census=T.census_of([], [], {}, [e]), capture={})
    out = render_timeline(tl)
    assert "causal_offset=7 [model-derived]" in out
