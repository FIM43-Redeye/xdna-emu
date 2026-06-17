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
