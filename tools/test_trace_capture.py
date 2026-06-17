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
