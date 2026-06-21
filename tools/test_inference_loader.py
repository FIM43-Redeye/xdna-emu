import json
from inference.loader import load_fired, replication_violations


def _ev(col, row, name, soc, slot=0, pkt_type=0, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": slot,
            "name": name, "ts": soc, "soc": soc, "mode": mode}


def _write_batch(d, events):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.events.json").write_text(
        json.dumps({"schema_version": 1, "events": events, "slot_names": {}}))


def _make_run(tmp_path, run_name, batches):
    rd = tmp_path / run_name
    for bn, evs in batches.items():
        _write_batch(rd / bn / "hw", evs)
    return str(rd)


def test_load_fired_emits_anchored_measured_facts(tmp_path):
    r0 = _make_run(tmp_path, "run0", {"batch_00": [
        _ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)]})
    facts = load_fired([r0])
    keyed = {f.args[0]: f for f in facts}
    assert keyed["1|0|0|DMA"].args == ("1|0|0|DMA", 0, 300)   # 1300 - 1000
    assert type(keyed["1|0|0|DMA"].support).__name__ == "Measured"
    assert keyed["1|2|0|PERF_CNT_2"].args == ("1|2|0|PERF_CNT_2", 0, 0)


def test_load_fired_indexes_runs(tmp_path):
    r0 = _make_run(tmp_path, "run0", {"batch_00": [
        _ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)]})
    r1 = _make_run(tmp_path, "run1", {"batch_00": [
        _ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1305)]})
    facts = load_fired([r0, r1])
    runs = sorted(f.args[1] for f in facts if f.args[0] == "1|0|0|DMA")
    assert runs == [0, 1]


def test_replication_clean_when_batches_agree(tmp_path):
    r0 = _make_run(tmp_path, "run0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1301)]})
    assert replication_violations([r0]) == []


def test_replication_flags_disagreement(tmp_path):
    # same (event, run) but the two batches disagree by 50 >> eps -> a planted bug
    r0 = _make_run(tmp_path, "run0", {
        "batch_00": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1300)],
        "batch_01": [_ev(1, 2, "PERF_CNT_2", 1000), _ev(1, 0, "DMA", 1350)]})
    viols = replication_violations([r0])
    assert any(v["event_key"] == "1|0|0|DMA" for v in viols)
