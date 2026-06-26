import json
from pathlib import Path
import trace_join as tj
import trace_variance as tv


def _ev(col, row, name, soc, slot=0, pkt_type=0, ts=None, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc if ts is None else ts, "soc": soc, "mode": mode}


def _write_batch(d, events):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.events.json").write_text(
        json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))


def _make_run(tmp_path, run_name, batches):
    # batches: {batch_name: [event,...]}
    rd = tmp_path / run_name
    for bn, evs in batches.items():
        _write_batch(rd / bn / "hw", evs)
    return str(rd)


def test_anchored_firsts_subtracts_anchor_first_fire():
    events = [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 2, "PERF_CNT_2", 2024),
              _ev(1, 0, "DMA_S2MM_0_START_TASK", 1500),
              _ev(1, 2, "LOCK_STALL", 1200)]
    out = tj.anchored_firsts(events)
    assert out["1|0|0|DMA_S2MM_0_START_TASK"] == 500   # 1500 - 1000
    assert out["1|2|0|LOCK_STALL"] == 200              # 1200 - 1000
    assert out["1|2|0|PERF_CNT_2"] == 0                # first fire is the anchor


def test_anchored_firsts_uses_first_occurrence():
    events = [_ev(1, 2, "PERF_CNT_2", 1000),
              _ev(1, 0, "X", 1800), _ev(1, 0, "X", 1500)]
    out = tj.anchored_firsts(events)
    assert out["1|0|0|X"] == 500   # min soc (1500) - anchor (1000)


def test_anchored_firsts_empty_when_no_anchor():
    out = tj.anchored_firsts([_ev(1, 0, "X", 1500)])
    assert out == {}


def test_load_active_events_unions_across_batches(tmp_path):
    _write_batch(tmp_path / "batch_00" / "hw",
                 [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "A", 1100)])
    _write_batch(tmp_path / "batch_01" / "hw",
                 [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "B", 1100),
                  _ev(1, 1, "C", 1200)])
    out = tj.load_active_events(str(tmp_path))
    assert out["1|0|0"] == {"A", "B"}
    assert out["1|1|0"] == {"C"}
    assert out["1|2|0"] == {"PERF_CNT_2"}


def test_pair_derivability_constant_offset_is_low_std(tmp_path):
    # X = S + 50 in every run -> derivable (std ~ 0)
    runs = []
    for i, base in enumerate([1000, 1200, 900]):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", base),
            _ev(1, 2, "S", base + 100),
            _ev(1, 2, "X", base + 150)]}))
    s = tj.pair_derivability(runs, "1|2|0|X", "1|2|0|S")
    assert s is not None
    assert s.n == 3
    assert s.mean == 50
    assert s.std == 0.0


def test_pair_derivability_varying_offset_is_high_std(tmp_path):
    offsets = [50, 900, 1700]
    runs = []
    for i, off in enumerate(offsets):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000),
            _ev(1, 0, "S", 1100),
            _ev(1, 2, "X", 1100 + off)]}))
    s = tj.pair_derivability(runs, "1|2|0|X", "1|0|0|S")
    assert s is not None
    assert s.std > 100   # clearly stochastic difference


def test_pair_derivability_none_when_never_cotraced(tmp_path):
    # X and S live in different batches -> never co-traced in one execution
    runs = [_make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1100)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 2, "X", 1300)]})]
    assert tj.pair_derivability(runs, "1|2|0|X", "1|0|0|S") is None


def test_build_graph_finds_edge_and_roots(tmp_path):
    # Two runs. X = S + 50 (derivable). S floats vs anchor (stochastic root).
    runs = []
    for i, sbase in enumerate([1100, 1900]):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000),
            _ev(1, 0, "S", sbase),
            _ev(1, 0, "X", sbase + 50)]}))
    g = tj.build_derivability_graph(runs, eps=2.0)
    assert set(g["nodes"]) == {"1|2|0|PERF_CNT_2", "1|0|0|S", "1|0|0|X"}
    # S -> X edge with offset 50
    edge = [e for e in g["edges"] if e["to"] == "1|0|0|X" and e["from"] == "1|0|0|S"]
    assert len(edge) == 1 and edge[0]["offset"] == 50
    # X has an incoming edge -> not a root; S and anchor are roots
    assert "1|0|0|X" not in g["roots"]
    assert "1|0|0|S" in g["roots"] and "1|2|0|PERF_CNT_2" in g["roots"]
    # S floats vs anchor -> stochastic root; anchor never a stochastic root
    assert "1|0|0|S" in g["stochastic_roots"]
    assert "1|2|0|PERF_CNT_2" not in g["stochastic_roots"]


def test_build_graph_deterministic_event_not_stochastic_root(tmp_path):
    # D fires at a fixed offset from the anchor in every run -> root but deterministic
    runs = []
    for i in range(3):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000 + 10 * i),
            _ev(1, 1, "D", 1300 + 10 * i)]}))   # always anchor+300
    g = tj.build_derivability_graph(runs, eps=2.0)
    assert "1|1|0|D" in g["roots"]
    assert "1|1|0|D" not in g["stochastic_roots"]   # std of (D-anchor) == 0


def _graph(stochastic_roots, nodes, anchor="1|2|0|PERF_CNT_2"):
    return {"anchor": anchor, "eps": 2.0, "nodes": nodes, "edges": [],
            "roots": [anchor] + stochastic_roots,
            "stochastic_roots": stochastic_roots, "bands": {}}


def _band_graph():
    return {
        "anchor": "1|2|0|PERF_CNT_2", "eps": 2.0,
        "nodes": ["1|2|0|PERF_CNT_2", "1|0|0|S", "1|0|0|X", "1|1|0|D"],
        "edges": [{"from": "1|0|0|S", "to": "1|0|0|X", "offset": 50, "std": 0.0}],
        "roots": ["1|2|0|PERF_CNT_2", "1|0|0|S", "1|1|0|D"],
        "stochastic_roots": ["1|0|0|S"],
        "bands": {"1|0|0|S": {"n": 2, "mean": 600.0, "std": 400.0,
                              "min": 200, "max": 1000, "range": 800}},
    }


def test_synthesize_plan_reserves_always_on_every_batch():
    nodes = ["1|2|0|PERF_CNT_2", "1|0|0|DMA_S2MM_0_START_TASK",
             "1|0|0|A", "1|0|0|B", "1|2|0|C"]
    g = _graph(["1|0|0|DMA_S2MM_0_START_TASK"], nodes)
    plan = tj.synthesize_plan(g, slot_capacity=8)
    assert plan["always_on"]["1|2|0"] == ["PERF_CNT_2"]
    assert plan["always_on"]["1|0|0"] == ["DMA_S2MM_0_START_TASK"]
    # every batch carries the always-on names on each tile
    for b in plan["batches"]:
        assert "PERF_CNT_2" in b["1|2|0"]
        assert "DMA_S2MM_0_START_TASK" in b["1|0|0"]


def test_synthesize_plan_batch_count_from_busiest_tile():
    # tile 1|0|0: 1 always-on + 14 payload, 7 free slots -> ceil(14/7)=2 batches
    nodes = ["1|2|0|PERF_CNT_2", "1|0|0|DMA_S2MM_0_START_TASK"] + \
            [f"1|0|0|E{i}" for i in range(14)]
    g = _graph(["1|0|0|DMA_S2MM_0_START_TASK"], nodes)
    plan = tj.synthesize_plan(g, slot_capacity=8)
    assert plan["n_batches"] == 2


def test_synthesize_plan_panics_when_always_on_overflows():
    # 9 stochastic roots on one tile, capacity 8 -> cannot fit anchor+roots
    roots = [f"1|0|0|R{i}" for i in range(9)]
    nodes = ["1|2|0|PERF_CNT_2"] + roots
    g = _graph(roots, nodes)
    import pytest
    with pytest.raises(tj.PlannerError) as exc:
        tj.synthesize_plan(g, slot_capacity=8)
    assert "1|0" in str(exc.value)   # diagnostic names the offending tile


def test_join_run_classifies_and_attaches_band(tmp_path):
    rd = _make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1200, slot=3),
                     _ev(1, 0, "X", 1250), _ev(1, 1, "D", 1400)],
    })
    recs = tj.join_run(rd, _band_graph(), eps=2.0)
    by = {(r["col"], r["row"], r["name"]): r for r in recs}
    assert by[(1, 0, "S")]["class"] == "stochastic"
    assert by[(1, 0, "S")]["band"]["std"] == 400.0
    assert by[(1, 0, "S")]["slot"] == 3
    assert by[(1, 0, "X")]["class"] == "derivable"
    assert by[(1, 0, "X")]["predictor"] == {"name": "1|0|0|S", "offset": 50}
    assert by[(1, 1, "D")]["class"] == "deterministic"
    # sorted by ts_anchored
    assert [r["ts_anchored"] for r in recs] == sorted(r["ts_anchored"] for r in recs)


def test_join_run_keeps_stochastic_samples_per_batch(tmp_path):
    rd = _make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1200)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1700)],
    })
    recs = [r for r in tj.join_run(rd, _band_graph()) if r["name"] == "S"]
    assert sorted(r["ts_anchored"] for r in recs) == [200, 700]
    assert {r["source_batch"] for r in recs} == {0, 1}


def test_join_run_raises_on_deterministic_spread(tmp_path):
    # D is deterministic in the graph but observed at divergent anchored ts
    rd = _make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 1, "D", 1400)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 1, "D", 1900)],
    })
    import pytest
    with pytest.raises(tj.JoinError):
        tj.join_run(rd, _band_graph(), eps=2.0)


def test_to_perfetto_shape():
    recs = [{"col": 1, "row": 0, "name": "S", "slot": 3, "ts_anchored": 200,
             "source_batch": 0, "class": "stochastic", "predictor": None,
             "band": {"std": 400.0}}]
    doc = tj.to_perfetto(recs)
    ev = doc["traceEvents"][0]
    assert ev["ph"] == "i" and ev["ts"] == 200 and ev["name"] == "S"
    assert ev["pid"] == 100 and ev["args"]["class"] == "stochastic"


def test_cli_empty_glob_returns_1(tmp_path, capsys):
    rc = tj.main(["--runs-glob", str(tmp_path / "nope_*"),
                  "--out", str(tmp_path / "out")])
    assert rc == 1


def _occ_write_batch(rd: Path, batch: str, events):
    p = rd / batch / "hw"
    p.mkdir(parents=True)
    (p / "trace.events.json").write_text(json.dumps({"events": events}))

def _occ_ev(col, row, pkt, name, soc, slot=0):
    return {"col": col, "row": row, "pkt_type": pkt, "name": name, "soc": soc, "slot": slot}

def test_batch_occurrences_returns_all_firings_anchored(tmp_path):
    rd = tmp_path / "run_00"
    _occ_write_batch(rd, "batch_00", [
        _occ_ev(1, 2, 0, "PERF_CNT_2", 1000),
        _occ_ev(1, 1, 3, "PORT_RUNNING_0", 1010),
        _occ_ev(1, 1, 3, "PORT_RUNNING_0", 1026),
        _occ_ev(1, 1, 3, "PORT_RUNNING_0", 1042),
    ])
    occ = tj.batch_occurrences(str(rd), "batch_00")
    assert occ["1|1|3|PORT_RUNNING_0"] == [10, 26, 42]   # anchored, sorted
    assert occ["1|2|0|PERF_CNT_2"] == [0]

def test_batch_occurrences_empty_without_anchor(tmp_path):
    rd = tmp_path / "run_00"
    _occ_write_batch(rd, "batch_00", [_occ_ev(1, 1, 3, "PORT_RUNNING_0", 1010)])
    assert tj.batch_occurrences(str(rd), "batch_00") == {}


def test_cli_end_to_end_synthetic_planned(tmp_path):
    # Two runs, one batch each, all candidates co-traced (planned shape):
    #   anchor PERF_CNT_2; S floats vs anchor (stochastic root); X = S + 50.
    for i, sbase in enumerate([1200, 1900]):
        _make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", 1000),
            _ev(1, 0, "S", sbase, slot=3),
            _ev(1, 0, "X", sbase + 50, slot=4)]})
    out = tmp_path / "out"
    rc = tj.main(["--runs-glob", str(tmp_path / "run_*"),
                  "--join-run", str(tmp_path / "run_0"), "--out", str(out)])
    assert rc == 0
    graph = json.loads((out / "derivability-graph.json").read_text())
    assert "1|0|0|S" in graph["stochastic_roots"]
    assert any(e["to"] == "1|0|0|X" for e in graph["edges"])
    merged = json.loads((out / "merged.events.json").read_text())
    by = {r["name"]: r for r in merged}
    assert by["S"]["class"] == "stochastic" and by["S"]["band"] is not None
    assert by["X"]["class"] == "derivable"
    assert [r["ts_anchored"] for r in merged] == \
        sorted(r["ts_anchored"] for r in merged)
    # plan + perfetto are valid JSON and non-empty
    assert json.loads((out / "batch-plan.json").read_text())["n_batches"] >= 1
    assert json.loads((out / "merged.perfetto.json").read_text())["traceEvents"]


