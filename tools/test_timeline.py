import glob
import json
import os

import pytest

import trace_join as tj
from inference import timeline as T
from inference import verifier as V


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


# ---------------------------------------------------------------------------
# Task 10: coupling_oracle, weave, connectivity_defects
# ---------------------------------------------------------------------------

from config_extract.dump_model import ConfigDump, RouteGraph, RouteEdge, PortRef


def _pr(col, row): return PortRef(col=col, row=row, port=0, dir="out", kind="x")


def test_coupling_oracle_from_route_graph(tmp_path):
    rg = RouteGraph(edges=(
        RouteEdge(_pr(0, 0), _pr(0, 1), "dma_buffer_relay"),
        RouteEdge(_pr(0, 1), _pr(0, 2), "circuit"),
    ))
    dump = ConfigDump(device="npu1", route_graph=rg, tiles=())
    cpl = T.coupling_oracle(dump, start_col=1)   # abs col = 0+1
    assert ("1|0", "1|1") in cpl and ("1|1", "1|2") in cpl


def test_weave_and_connectivity(tmp_path, monkeypatch):
    from inference.grounding import Gap, GAP_CROSS_DOMAIN
    monkeypatch.setattr(T, "ground_edge",
        lambda runs, c, p, anchor=T.ANCHOR: Gap(parent=p, child=c, reason=GAP_CROSS_DOMAIN, reproduction_offset=7))
    edges = T.weave(["r0"], [("1|1|3|X", "1|2|0|Y")])
    assert edges[0].reproduction_offset == 7 and edges[0].reason == GAP_CROSS_DOMAIN
    # oracle says 1|2 and 1|1 coupled and the weave connects them -> no defect
    assert T.connectivity_defects({("1|1", "1|2")}, edges) == []
    # oracle says 1|0 and 1|2 coupled but nothing connects them -> defect
    assert T.connectivity_defects({("1|0", "1|2")}, edges) == [("1|0", "1|2")]


# ---------------------------------------------------------------------------
# Task 11: census_of, assemble_timeline, render_timeline (end-to-end)
# ---------------------------------------------------------------------------

def _write_run(base, run_name, events):
    """Write one synthetic run dir in the Task-1 layout
    (run_NN/batch_00/hw/trace.events.json) and return its path string."""
    d = base / run_name / "batch_00" / "hw"
    d.mkdir(parents=True)
    (d / "trace.events.json").write_text(json.dumps({"events": events}))
    return str(base / run_name)


def _ev(name, soc, col=1, row=2, pkt_type=0, slot=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "name": name,
            "soc": soc, "slot": slot}


def _synthetic_one_core_track(base):
    """One core domain 1|2|0 over three runs.

    Anchor PERF_CNT_2 at soc 0 every run (anchored ts == soc).
      A: anchored 10 every run  -> jitter (0,0,0)  } shared all-zero jitter
      B: anchored 26 every run  -> jitter (0,0,0)  } -> one anchored frame {A,B}
      C: anchored 40 / 42 / 45  -> jitter (0,2,5)  -> nondeterministic singleton
    """
    socs_c = [40, 42, 45]
    run_dirs = []
    for i, c_soc in enumerate(socs_c):
        events = [
            _ev("PERF_CNT_2", 0),
            _ev("A", 10),
            _ev("B", 26),
            _ev("C", c_soc),
        ]
        run_dirs.append(_write_run(base, f"run_{i:02d}", events))
    return run_dirs


def test_assemble_timeline_one_core_track(tmp_path):
    run_dirs = _synthetic_one_core_track(tmp_path)
    configured = ["1|2|0|A", "1|2|0|B", "1|2|0|C"]

    tl = T.assemble_timeline(run_dirs, configured, derives_pairs=set(),
                             cross_domain_pairs=[], dump=None)

    # exactly one track, for the single core domain
    assert len(tl.tracks) == 1
    track = tl.tracks[0]
    assert track.domain == "1|2|0"

    # the frame {A,B} forms a DeterministicPeriod; C is a NondeterministicPeriod
    dets = [p for p in track.periods if isinstance(p, T.DeterministicPeriod)]
    nondets = [p for p in track.periods if isinstance(p, T.NondeterministicPeriod)]
    assert len(dets) == 1 and len(nondets) == 1

    det = dets[0]
    assert set(det.events) == {"1|2|0|A", "1|2|0|B"}
    assert det.floating is False
    assert det.grounding_event == "1|2|0|A"
    assert det.cycles == {"1|2|0|A": 0, "1|2|0|B": 16}

    nondet = nondets[0]
    assert nondet.events == ["1|2|0|C"]
    assert nondet.windows["1|2|0|C"] == (30, 35)   # (C - A) across runs

    # no cross-domain pairs -> no cross-track edges
    assert tl.cross_track_edges == []
    # dump=None -> count-ceiling honesty flag declared
    assert "count_ceiling_unknown" in tl.flags

    # census: 2 of 3 events live in a deterministic frame -> content_ok True
    assert tl.census.events["anchored"] == 2
    assert tl.census.events["nondeterministic"] == 1
    assert tl.census.content_ok is True


def test_render_timeline_smoke(tmp_path):
    run_dirs = _synthetic_one_core_track(tmp_path)
    configured = ["1|2|0|A", "1|2|0|B", "1|2|0|C"]
    tl = T.assemble_timeline(run_dirs, configured, derives_pairs=set(),
                             cross_domain_pairs=[], dump=None)
    out = T.render_timeline(tl)
    assert "1|2|0" in out          # the domain id
    assert "DET" in out            # deterministic marker
    assert "NONDET" in out         # nondeterministic marker
    assert "census" in out.lower() # the census line


def test_census_of_buckets():
    det = T.DeterministicPeriod(events=["d|A", "d|B"], cycles={"d|A": 0, "d|B": 4},
                                grounding_event="d|A", floating=False)
    flt = T.DeterministicPeriod(events=["d|F"], cycles={"d|F": 0},
                                grounding_event="d|F", floating=True)
    nd = T.NondeterministicPeriod(events=["d|N"], windows={"d|N": (1, 3)},
                                  reasons={"d|N": "within_domain_nonexact"},
                                  order_edges=[], grounding_event=None)
    track = T.Track(domain="d", periods=[det, flt, nd])
    repro = T.CrossTrackEdge(child="x", parent="y", reason="cross_domain",
                             reproduction_offset=7)
    exist = T.CrossTrackEdge(child="p", parent="q", reason="async_cdc",
                             reproduction_offset=None)
    pc = T.PresenceClass(appearance=(0,), events=["d|I"])
    census = T.census_of([track], intermittent=[pc],
                         excluded={"d|E": T.F_BATCH_FLIP}, edges=[repro, exist])
    assert census.events == {"anchored": 2, "floating": 1, "nondeterministic": 1,
                             "intermittent": 1, "excluded": 1}
    assert census.edges == {"reproduction": 1, "existence_only": 1}
    # 3 of 6 events in deterministic frames (anchored+floating) == floor 0.5 -> True
    assert census.content_ok is True


def test_run_experiment_report_includes_timeline(tmp_path, monkeypatch):
    """run_experiment's returned report must include a 'timeline' key surfaced from
    the best-effort engine block. run_loop_until_converged and run_engine are
    monkeypatched to avoid hardware and keep the test self-contained."""
    import inference.loop as loop_mod
    import inference.engine as eng_mod
    from inference.run_experiment import run_experiment, KernelConfig

    fake_tl = T.IntegratedTimeline(
        tracks=[], cross_track_edges=[], intermittent=[],
        flags=["count_ceiling_unknown"],
        census=T.Census(events={}, edges={}, content_ok=True),
        capture={})

    loop_result = {
        "converged": True,
        "terminal_state": "segment",
        "iterations": 1,
        "classification": {},
        "run_dirs": [str(tmp_path / "run0")],
        "model": type("_Model", (), {"constraints": lambda self: []})(),
    }

    monkeypatch.setattr(loop_mod, "run_loop_until_converged",
                        lambda *a, **kw: loop_result)
    monkeypatch.setattr(eng_mod, "run_engine", lambda *a, **kw: {
        "derives": [], "segments": [], "gaps": [], "warnings": [],
        "rejected_rules": [], "stochastic_roots": [], "provenance_ok": True,
        "classification": {}, "replication_violations": [],
        "irreducible_groups": [], "degeneracy": [],
        "timeline": fake_tl,
    })

    class _FakeInstrument:
        def ledger_entries(self): return []

    cfg = KernelConfig(
        test="dummy", compiler="chess", dump_path=None,
        start_col=1, anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
        traced_col=1, n_runs=1, out_root=str(tmp_path / "out"),
    )
    report = run_experiment(cfg, instrument=_FakeInstrument(),
                            configured=["1|2|0|A"], candidate_pairs=[])
    assert "timeline" in report
    assert report["timeline"] is fake_tl


# ---------------------------------------------------------------------------
# Task 13: multi-track synthetic end-to-end (two domains, one cross-track edge)
# ---------------------------------------------------------------------------

from config_extract.dump_model import ConfigDump as _ConfigDump
from config_extract.dump_model import PortRef as _PortRef
from config_extract.dump_model import RouteEdge as _RouteEdge
from config_extract.dump_model import RouteGraph as _RouteGraph


def _ev2(name, soc, col, row, pkt_type):
    return {"col": col, "row": row, "pkt_type": pkt_type, "name": name,
            "soc": soc, "slot": 0}


def _synthetic_two_domain(base):
    """Two domains over three runs, anchored to PERF_CNT_2 at 1|2|0:

      core domain    1|2|0 : A@10  B@26     -> one anchored frame {A,B}
      memtile domain 1|1|2 : Y@100 Z@140    -> one anchored frame {Y,Z}

    All events anchor-rigid (same soc every run) so each domain yields one
    DeterministicPeriod; the cross-domain candidate (A <- Y) grounds as a Gap.
    """
    run_dirs = []
    for i in range(3):
        events = [
            _ev2("PERF_CNT_2", 0, 1, 2, 0),     # anchor
            _ev2("A", 10, 1, 2, 0),
            _ev2("B", 26, 1, 2, 0),
            _ev2("Y", 100, 1, 1, 2),
            _ev2("Z", 140, 1, 1, 2),
        ]
        run_dirs.append(_write_run(base, f"run_{i:02d}", events))
    return run_dirs


def test_assemble_timeline_multi_track_two_domains(tmp_path):
    run_dirs = _synthetic_two_domain(tmp_path)
    configured = ["1|2|0|A", "1|2|0|B", "1|1|2|Y", "1|1|2|Z"]
    # A (core) is the child; Y (memtile) is the parent -> a genuine cross-domain
    # candidate that grounding.ground_edge resolves to a cross_domain Gap.
    cross_domain_pairs = [("1|2|0|A", "1|1|2|Y")]

    # Route graph (internal cols are 0; start_col=1 lifts them to absolute col 1):
    # an inter-tile edge between tile 1|1 and tile 1|2 -> coupling_oracle yields the
    # ("1|1","1|2") coupled pair the weave must cover.
    rg = _RouteGraph(edges=(
        _RouteEdge(_PortRef(col=0, row=1, port=0, dir="out", kind="x"),
                   _PortRef(col=0, row=2, port=0, dir="in", kind="x"),
                   "inter_tile"),
    ))
    dump = _ConfigDump(device="npu1", route_graph=rg, tiles=())

    tl = T.assemble_timeline(run_dirs, configured, derives_pairs=set(),
                             cross_domain_pairs=cross_domain_pairs,
                             dump=dump, start_col=1)

    # (1) exactly TWO tracks, one per domain.
    assert len(tl.tracks) == 2
    assert {tr.domain for tr in tl.tracks} == {"1|1|2", "1|2|0"}

    # (2) a single cross-track edge connects the two domains.
    assert len(tl.cross_track_edges) == 1
    edge = tl.cross_track_edges[0]
    assert edge.child == "1|2|0|A" and edge.parent == "1|1|2|Y"
    assert edge.reason == "cross_domain"

    # (3) the weave covers the coupled tile pair -> no connectivity defect.
    oracle = T.coupling_oracle(dump, start_col=1)
    assert oracle == {("1|1", "1|2")}
    assert T.connectivity_defects(oracle, tl.cross_track_edges) == []
    assert not any(str(f).startswith("connectivity_defect") for f in tl.flags)

    # (4) NO cross-domain DeterministicPeriod: every frame's members share one
    # domain prefix (the fatal-A invariant -- a cross-domain "cycle" would be
    # Delta_wall + skew, never a tile-cycle).
    for tr in tl.tracks:
        for p in tr.periods:
            if isinstance(p, T.DeterministicPeriod):
                domains = {m.rsplit("|", 1)[0] for m in p.events}
                assert len(domains) == 1, f"cross-domain frame: {p.events}"

    # (5) the census reflects both tracks: all four anchored events are bucketed.
    assert tl.census.events["anchored"] == 4
    assert tl.census.events["nondeterministic"] == 0
    assert tl.census.content_ok is True


# ---------------------------------------------------------------------------
# Task 13: real-data A/B checks (clean cluster-stable vs loaded fragmentation).
# Fixtures are persisted HW captures (20 runs each); these tests RUN here.
# ---------------------------------------------------------------------------

_EXP = "/home/triple/npu-work/xdna-emu/build/experiments"
_CLEAN = f"{_EXP}/lock-jitter-clean"
_LOADED = f"{_EXP}/lock-jitter-loaded"

# Reuse the canary's sentinel core-lock keys (col 1, row 2, core pkt 0).
_ACQ = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
_REL = "1|2|0|INSTR_LOCK_RELEASE_REQ"


def _det_periods_with(tl, *keys):
    """All DeterministicPeriods that contain every key in keys."""
    out = []
    for tr in tl.tracks:
        for p in tr.periods:
            if isinstance(p, T.DeterministicPeriod) and all(k in p.cycles for k in keys):
                out.append(p)
    return out


@pytest.mark.skipif(not os.path.isdir(_CLEAN),
                    reason="persisted HW clean fixture absent on this machine")
def test_real_clean_clusters_stable():
    """Over the 20 clean runs the core-lock pair lands in ONE DeterministicPeriod
    with an exact local-cycle delta of 24. Empirically the pair is a FLOATING
    frame (ACQ and REL share an identical non-zero jitter vector -> their
    difference is range-0), admitted by the N>=MIN_N_FLOATING statistical gate
    even with derives_pairs=set(). The delta is genuine (Q=0), never tuned."""
    run_dirs = sorted(glob.glob(f"{_CLEAN}/capture_00/run_*"))
    assert len(run_dirs) >= T.MIN_N_FLOATING, "need >= MIN_N_FLOATING runs"

    tl = T.assemble_timeline(run_dirs, [_ACQ, _REL], derives_pairs=set(),
                             cross_domain_pairs=[], dump=None)

    periods = _det_periods_with(tl, _ACQ, _REL)
    assert len(periods) == 1, "the lock pair must share exactly one DeterministicPeriod"
    det = periods[0]
    assert det.cycles[_REL] - det.cycles[_ACQ] == 24
    assert det.floating is True   # empirical: floating frame, not anchor-rigid


@pytest.mark.skipif(not os.path.isdir(_LOADED),
                    reason="persisted HW loaded fixture absent on this machine")
def test_real_loaded_clusters_fragment():
    """Over the 20 loaded runs the core-lock span flickers (host-load capture
    contamination): ACQ and REL no longer share an identical jitter vector, so
    they do NOT form one rigid DeterministicPeriod with delta 24. Assert the
    negative precisely -- no deterministic frame holds both with that delta."""
    run_dirs = sorted(glob.glob(f"{_LOADED}/capture_00/run_*"))
    assert run_dirs, "loaded fixture must have runs"

    tl = T.assemble_timeline(run_dirs, [_ACQ, _REL], derives_pairs=set(),
                             cross_domain_pairs=[], dump=None)

    bad = [p for p in _det_periods_with(tl, _ACQ, _REL)
           if p.cycles[_REL] - p.cycles[_ACQ] == 24]
    assert bad == [], "loaded capture must NOT yield a rigid delta-24 lock frame"


@pytest.mark.skipif(not os.path.isdir(_CLEAN),
                    reason="persisted HW clean fixture absent on this machine")
def test_real_event_batch_invariant_check_i():
    """Check (i): every event traced in >= 2 batches of run 0 must be
    batch-invariant (cross_batch_range == 0) -- the precondition for cross-batch
    frame membership. Read all anchored values through verifier.cross_batch_range
    (single source), never by re-parsing the JSON."""
    run_dirs = sorted(glob.glob(f"{_CLEAN}/capture_00/run_*"))
    r0 = run_dirs[0]

    seen = {}
    for bn in tj._batch_names(r0):
        for k in tj.batch_firsts(r0, bn):
            seen[k] = seen.get(k, 0) + 1
    multi = [k for k, c in seen.items() if c >= 2]
    assert multi, "expected at least one event traced in >= 2 batches of run 0"

    for k in multi:
        assert V.cross_batch_range(run_dirs, k) == 0, \
            f"{k} is not batch-invariant across the batches that trace it"
