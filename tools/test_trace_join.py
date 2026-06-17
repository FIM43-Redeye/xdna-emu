import json
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
    assert out["1|0|DMA_S2MM_0_START_TASK"] == 500   # 1500 - 1000
    assert out["1|2|LOCK_STALL"] == 200              # 1200 - 1000
    assert out["1|2|PERF_CNT_2"] == 0                # first fire is the anchor


def test_anchored_firsts_uses_first_occurrence():
    events = [_ev(1, 2, "PERF_CNT_2", 1000),
              _ev(1, 0, "X", 1800), _ev(1, 0, "X", 1500)]
    out = tj.anchored_firsts(events)
    assert out["1|0|X"] == 500   # min soc (1500) - anchor (1000)


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
    assert out["1|0"] == {"A", "B"}
    assert out["1|1"] == {"C"}
    assert out["1|2"] == {"PERF_CNT_2"}


def test_pair_derivability_constant_offset_is_low_std(tmp_path):
    # X = S + 50 in every run -> derivable (std ~ 0)
    runs = []
    for i, base in enumerate([1000, 1200, 900]):
        runs.append(_make_run(tmp_path, f"run_{i}", {"batch_00": [
            _ev(1, 2, "PERF_CNT_2", base),
            _ev(1, 2, "S", base + 100),
            _ev(1, 2, "X", base + 150)]}))
    s = tj.pair_derivability(runs, "1|2|X", "1|2|S")
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
    s = tj.pair_derivability(runs, "1|2|X", "1|0|S")
    assert s is not None
    assert s.std > 100   # clearly stochastic difference


def test_pair_derivability_none_when_never_cotraced(tmp_path):
    # X and S live in different batches -> never co-traced in one execution
    runs = [_make_run(tmp_path, "run_0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "S", 1100)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 2, "X", 1300)]})]
    assert tj.pair_derivability(runs, "1|2|X", "1|0|S") is None
