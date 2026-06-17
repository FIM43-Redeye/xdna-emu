import json
import trace_join as tj


def _ev(col, row, name, soc, slot=0, pkt_type=0, ts=None, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc if ts is None else ts, "soc": soc, "mode": mode}


def _write_batch(d, events):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.events.json").write_text(
        json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))


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
