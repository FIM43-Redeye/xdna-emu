import json
from inference.facts import KB
from inference.reachability import ReachabilityModel, Constraint
from inference.planner import (plan_cotrace, propose_next, seed_plan, Batch,
                               NO_GAIN)


def _ev(col, row, name, soc, pkt_type=0):
    return {"col": col, "row": row, "pkt_type": pkt_type, "slot": 0,
            "name": name, "ts": soc, "soc": soc, "mode": 0}


def _runs(tmp_path, rows):
    dirs = []
    for i, off in enumerate(rows):
        rd = tmp_path / f"run{i}"
        evs = [_ev(1, 2, "PERF_CNT_2", 1000)]
        for (tile, name), delta in off.items():
            c, r, p = tile.split("|")
            evs.append(_ev(int(c), int(r), name, 1000 + delta, pkt_type=int(p)))
        (rd / "batch_00" / "hw").mkdir(parents=True)
        (rd / "batch_00" / "hw" / "trace.events.json").write_text(
            json.dumps({"schema_version": 1, "events": evs, "slot_names": {}}))
        dirs.append(str(rd))
    return dirs


def test_plan_cotrace_groups_events_by_tile():
    b = plan_cotrace("1|0|0|DMA", "1|1|3|PORT")
    assert b.tiles["1|0|0"] == ["DMA"]
    assert b.tiles["1|1|3"] == ["PORT"]
    assert b.mode_by_tile["1|0|0"] == 0


def test_plan_cotrace_core_tile_mode_override():
    b = plan_cotrace("1|2|0|INSTR", "1|0|0|DMA",
                     mode_by_tile={"1|2|0": 1})   # core can use EVENT_PC
    assert b.mode_by_tile["1|2|0"] == 1
    assert b.mode_by_tile["1|0|0"] == 0


def test_propose_next_no_gain_for_measured_stable_unoriented_pair(tmp_path):
    # C = S + 30 every run, fully co-traced, no config_path -> NO_GAIN
    dirs = _runs(tmp_path, [{("1|0|0", "S"): 100, ("1|0|0", "C"): 130},
                            {("1|0|0", "S"): 220, ("1|0|0", "C"): 250}])
    kb = KB.empty()
    m = ReachabilityModel()
    assert propose_next(kb, dirs, ("1|0|0|C", "1|0|0|S"), m) is NO_GAIN


def test_propose_next_emits_cotrace_for_never_cotraced_pair(tmp_path):
    # X and Y were never in one batch -> co-trace gain
    dirs = _runs(tmp_path, [{("1|0|0", "X"): 100}])  # Y absent
    kb = KB.empty()
    m = ReachabilityModel()
    m.add_constraint(Constraint("sep", "can_separate", ("1|0|0|X", "1|1|3|Y"),
                                provenance_batch="b0"))
    b = propose_next(kb, dirs, ("1|0|0|X", "1|1|3|Y"), m)
    assert isinstance(b, Batch)
    assert "1|1|3" in b.tiles


def test_seed_plan_covers_configured_events():
    b = seed_plan(["1|0|0|DMA", "1|2|0|INSTR", "1|1|3|PORT"])
    assert b.tiles["1|0|0"] == ["DMA"]
    assert b.tiles["1|2|0"] == ["INSTR"]
    assert b.tiles["1|1|3"] == ["PORT"]
