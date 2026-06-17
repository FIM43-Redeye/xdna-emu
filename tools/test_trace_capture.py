import trace_capture as tc


def test_load_event_ids_core_has_perf_cnt_2():
    ids = tc.load_event_ids("core")
    assert ids["PERF_CNT_2"] == 7          # XAIEML_EVENTS_CORE_PERF_CNT_2 7U
    assert ids["INSTR_VECTOR"] == 37       # XAIEML_EVENTS_CORE_INSTR_VECTOR 37U


def test_load_event_ids_memmod_excludes_memtile_events():
    mem = tc.load_event_ids("memmod")
    memtile = tc.load_event_ids("memtile")
    # MEM_ prefix must not swallow MEM_TILE_ events
    assert not any(n.startswith("TILE_") for n in mem)
    # memtile has its own distinct table
    assert len(memtile) > 0


def test_load_event_ids_unknown_tile_type_raises():
    import pytest
    with pytest.raises(KeyError):
        tc.load_event_ids("bogus")


def test_configure_batch_anchor_first_and_label_map():
    batch = {"1|2|0": ["PERF_CNT_2", "LOCK_STALL"], "1|0|2": ["DMA_S2MM_0_START_TASK"]}
    spec, lmap = tc.configure_batch(batch, anchor="PERF_CNT_2")
    # core module: anchor in slot 0, LOCK_STALL slot 1
    assert lmap[(0, 2, 0)] == "PERF_CNT_2"
    assert lmap[(0, 2, 1)] == "LOCK_STALL"
    # shim module: its single event in slot 0 (no anchor on shim here)
    assert lmap[(2, 0, 0)] == "DMA_S2MM_0_START_TASK"
    # patch spec has resolved numeric IDs, core PERF_CNT_2 == 7 in slot 0
    core = [s for s in spec if s["tile_type"] == "core" and s["row"] == 2][0]
    assert core["events"][0] == 7


def test_configure_batch_rejects_over_8_events():
    import pytest
    batch = {"1|2|0": [f"E{i}" for i in range(9)]}
    with pytest.raises(ValueError):
        tc.configure_batch(batch)


def _raw(col, row, pkt, slot, ts, soc, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt, "slot": slot,
            "ts": ts, "soc": soc, "mode": mode}


def test_label_events_applies_map():
    lmap = {(0, 2, 0): "PERF_CNT_2", (0, 2, 1): "LOCK_STALL"}
    raw = [_raw(1, 2, 0, 0, 100, 100), _raw(1, 2, 0, 1, 150, 150)]
    out = tc.label_events(raw, lmap, traced_col=1)
    assert out[0]["name"] == "PERF_CNT_2" and out[1]["name"] == "LOCK_STALL"
    assert out[0]["pkt_type"] == 0 and out[0]["soc"] == 100


def test_label_events_unconfigured_slot_is_hard_error():
    import pytest
    lmap = {(0, 2, 0): "PERF_CNT_2"}
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(1, 2, 0, 5, 100, 100)], lmap, traced_col=1)


def test_label_events_foreign_column_is_hard_error():
    import pytest
    lmap = {(0, 2, 0): "PERF_CNT_2"}
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(3, 2, 0, 0, 100, 100)], lmap, traced_col=1)
