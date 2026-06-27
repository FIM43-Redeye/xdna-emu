import trace_capture as tc
from pathlib import Path


def _mk_build(tmp_path, name, files, runlit=None):
    d = tmp_path / "build" / "test" / "npu-xrt" / name / "chess"
    d.mkdir(parents=True)
    (d / "aie.xclbin").write_bytes(b"x")
    for f in files:
        (d / f).write_bytes(b"i")
    if runlit is not None:
        (tmp_path / "build" / "test" / "npu-xrt" / name).joinpath("run.lit").write_text(runlit)
    return tmp_path / "build" / "test" / "npu-xrt"


def test_discover_insts_prefers_insts_bin(tmp_path):
    root = _mk_build(tmp_path, "k", ["insts.bin", "other.bin"])
    _x, insts = tc._discover_xclbin_insts("k", build_root=root)
    assert insts.name == "insts.bin"


def test_discover_insts_parses_runlit(tmp_path):
    root = _mk_build(tmp_path, "k", ["k_insts.bin"],
                     runlit="// RUN: ... --npu-insts-name=k_insts.bin ...")
    _x, insts = tc._discover_xclbin_insts("k", build_root=root)
    assert insts.name == "k_insts.bin"


def test_discover_insts_single_bin_fallback(tmp_path):
    root = _mk_build(tmp_path, "k", ["aie_run_seq.bin"])
    _x, insts = tc._discover_xclbin_insts("k", build_root=root)
    assert insts.name == "aie_run_seq.bin"


def test_discover_insts_ambiguous_is_error(tmp_path):
    import pytest
    root = _mk_build(tmp_path, "k", ["a.bin", "b.bin"])   # no insts.bin, no run.lit
    with pytest.raises(tc.CaptureError):
        tc._discover_xclbin_insts("k", build_root=root)


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
    spec, lmap = tc.configure_batch(batch, anchor="PERF_CNT_2", start_col=1)
    # core module: anchor in slot 0, LOCK_STALL slot 1 (keyed by ABSOLUTE col 1)
    assert lmap[(0, 2, 1, 0)] == "PERF_CNT_2"
    assert lmap[(0, 2, 1, 1)] == "LOCK_STALL"
    # shim module: its single event in slot 0 (ABSOLUTE col 1)
    assert lmap[(2, 0, 1, 0)] == "DMA_S2MM_0_START_TASK"
    # patch spec has resolved numeric IDs, core PERF_CNT_2 == 7 in slot 0
    core = [s for s in spec if s["tile_type"] == "core" and s["row"] == 2][0]
    assert core["events"][0] == 7
    assert core["col"] == 0   # RELATIVE (abs 1 - start_col 1 = 0)


def test_configure_batch_rejects_over_8_events():
    import pytest
    batch = {"1|2|0": [f"E{i}" for i in range(9)]}
    with pytest.raises(ValueError):
        tc.configure_batch(batch)


def test_configure_batch_pads_patch_spec_to_8_slots_with_none():
    # The patcher overwrites only the slots we supply; short lists leave the
    # kernel's compile-time trace events live in the trailing slots, which fire
    # on HW. configure_batch must pad each tile's patch events to 8 with NONE(0)
    # while label_map maps only the real (configured) slots.
    batch = {"1|2|0": ["PERF_CNT_2", "LOCK_STALL"]}   # 2 real core events
    spec, lmap = tc.configure_batch(batch, anchor="PERF_CNT_2", start_col=1)
    core = [s for s in spec if s["tile_type"] == "core"][0]
    assert len(core["events"]) == 8                   # padded to 8 slots
    assert core["events"][0] == 7                     # PERF_CNT_2 still slot 0
    assert core["events"][2:] == [0, 0, 0, 0, 0, 0]   # slots 2-7 = NONE
    # label_map only carries the two real slots (0 and 1), never the padding
    # keyed by ABSOLUTE col (1), not relative
    assert set(lmap) == {(0, 2, 1, 0), (0, 2, 1, 1)}
    # every entry forces the trace mode (Trace_Control0) so HW and decoder agree
    assert core["mode"] == 0


def _raw(col, row, pkt, slot, ts, soc, mode=0):
    return {"col": col, "row": row, "pkt_type": pkt, "slot": slot,
            "ts": ts, "soc": soc, "mode": mode}


def test_label_events_applies_map():
    lmap = {(0, 2, 1, 0): "PERF_CNT_2", (0, 2, 1, 1): "LOCK_STALL"}
    raw = [_raw(1, 2, 0, 0, 100, 100), _raw(1, 2, 0, 1, 150, 150)]
    out = tc.label_events(raw, lmap)
    assert out[0]["name"] == "PERF_CNT_2" and out[1]["name"] == "LOCK_STALL"
    assert out[0]["pkt_type"] == 0 and out[0]["soc"] == 100


def test_label_events_unconfigured_slot_is_hard_error():
    import pytest
    lmap = {(0, 2, 1, 0): "PERF_CNT_2"}
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(1, 2, 0, 5, 100, 100)], lmap)


def test_configure_batch_multicolumn_no_collision_and_relative_patch():
    # two_col's collision: cores at absolute 1|2|0 and 2|2|0 both write slot 0.
    batch = {"1|2|0": ["PERF_CNT_2"], "2|2|0": ["PERF_CNT_2"]}
    spec, lmap = tc.configure_batch(batch, anchor="PERF_CNT_2", start_col=1)
    # label_map is keyed by ABSOLUTE col -> no collision, both survive
    assert lmap[(0, 2, 1, 0)] == "PERF_CNT_2"
    assert lmap[(0, 2, 2, 0)] == "PERF_CNT_2"
    # patch_spec is in RELATIVE col (abs - start_col): cols 0 and 1
    patch_cols = sorted(s["col"] for s in spec)
    assert patch_cols == [0, 1]


def test_label_events_absolute_col_lookup_two_columns():
    lmap = {(0, 2, 1, 0): "PERF_CNT_2", (0, 2, 2, 0): "PERF_CNT_2"}
    raw = [_raw(1, 2, 0, 0, 100, 100), _raw(2, 2, 0, 0, 200, 200)]
    out = tc.label_events(raw, lmap)
    assert {(e["col"], e["name"]) for e in out} == {(1, "PERF_CNT_2"), (2, "PERF_CNT_2")}


def test_label_events_unconfigured_is_hard_error_no_col_guard():
    lmap = {(0, 2, 1, 0): "PERF_CNT_2"}
    import pytest
    # column 2 is NOT a "foreign column" error anymore -- it's unconfigured (not in map)
    with pytest.raises(tc.CaptureError):
        tc.label_events([_raw(2, 2, 0, 0, 100, 100)], lmap)


def test_coverage_report_unions_across_runs():
    configured = {(0, 2, 0): "PERF_CNT_2", (0, 2, 1): "LOCK_STALL", (2, 0, 0): "DMA_X"}
    # run 0 saw anchor+lock; run 1 saw anchor+dma; union covers all 3
    observed = [{(0, 2, 0), (0, 2, 1)}, {(0, 2, 0), (2, 0, 0)}]
    rep = tc.coverage_report(configured, observed)
    assert rep["n_configured"] == 3 and rep["n_covered"] == 3 and rep["gaps"] == []


def test_coverage_report_names_a_never_seen_gap():
    configured = {(0, 2, 0): "PERF_CNT_2", (2, 0, 0): "DMA_X"}
    observed = [{(0, 2, 0)}, {(0, 2, 0)}]   # DMA_X never fired
    rep = tc.coverage_report(configured, observed)
    assert rep["n_covered"] == 1
    assert rep["gaps"] == [{"pkt_type": 2, "row": 0, "slot": 0, "name": "DMA_X"}]


import json as _json


def test_write_patch_spec_roundtrips(tmp_path):
    spec = [{"col": 1, "row": 2, "tile_type": "core", "events": [7, 60]}]
    p = tc.write_patch_spec(spec, tmp_path / "spec.json")
    assert _json.loads(p.read_text()) == spec


def test_runner_command_includes_trace_size_and_io():
    cmd = tc.runner_command("insts.bin", "trace.bin", tc.TRACE_SIZE_DEFAULT,
                            ["a.bin"], ["o.bin"])
    assert "--instr insts.bin" in cmd
    assert f"--trace-size {tc.TRACE_SIZE_DEFAULT}" in cmd
    assert "--input a.bin" in cmd and "--output o.bin" in cmd


def test_capture_writes_labeled_events_per_batch(tmp_path, monkeypatch):
    calls = {"reset": 0, "runs": 0, "patch": 0}

    class FakeRunner:
        def reset(self): calls["reset"] += 1
        def run_one(self, cmd):
            calls["runs"] += 1
            return {"ok": True}

    def fake_subprocess_run(cmd, **kw):
        calls["patch"] += 1
        class R: returncode = 0
        return R()
    monkeypatch.setattr(tc.subprocess, "run", fake_subprocess_run)

    # one core event that "fires" in the decode
    def fake_parse(words, slot_names=None, mode=None):
        return [{"col": 1, "row": 2, "pkt_type": 0, "slot": 0,
                 "ts": 100, "soc": 100, "mode": 0}]
    monkeypatch.setattr(tc, "parse_trace", fake_parse)
    monkeypatch.setattr(tc, "_read_trace_words", lambda p: [0])  # stub bin read

    plan = {"batches": [{"1|2|0": ["PERF_CNT_2"]}]}
    tc.capture(plan, FakeRunner(), test="add_one_using_dma", out_dir=tmp_path, start_col=1)

    assert calls["reset"] == 1 and calls["runs"] == 1 and calls["patch"] == 1
    ev = _json.loads((tmp_path / "batch_00" / "hw" / "trace.events.json").read_text())
    assert ev["events"][0]["name"] == "PERF_CNT_2"
    assert ev["events"][0]["pkt_type"] == 0


def test_decoder_smoke_on_real_capture():
    """Smoke test: in-tree decoder runs clean and produces per-tile commands on a real HW capture.

    Loads a real trace.bin captured from the NPU (add_one_using_dma,
    gap140/nondeterminism run) and verifies that the in-tree
    trace_decoder.decode_words does not raise and returns a non-empty per-tile
    command dict.

    This is a smoke test only — it does NOT compare two decoders.  The
    authoritative byte-for-byte parity against mlir-aie's decoder lives in
    tools/test_trace_decoder.py::test_mode0_decode_matches_oracle_byte_for_byte.

    Fixture: real trace.bin from add_one_using_dma gap140/nondeterminism capture.
    """
    from pathlib import Path
    import numpy as np
    from trace_decoder import decode_words
    from trace_decoder.frame import TraceMode

    # Fixture: real captured trace from NPU
    fixture_path = Path(__file__).parent.parent / "build" / "experiments" / "gap140" / \
                   "nondeterminism" / "add_one_using_dma" / "run_00" / "batch_00" / \
                   "hw" / "trace.bin"

    # Skip gracefully if fixture is missing (build not run yet)
    if not fixture_path.exists():
        import pytest
        pytest.skip(f"fixture not found at {fixture_path}")

    # Load the raw trace binary
    raw_words = np.fromfile(str(fixture_path), dtype="<u4")
    assert len(raw_words) > 0, "trace.bin is empty"

    # Decode with the in-tree decoder (EVENT_TIME mode)
    try:
        per_tile_commands = decode_words(raw_words.tolist(), mode=TraceMode.EVENT_TIME)
    except Exception as e:
        raise AssertionError(f"decoder failed on real capture: {e}") from e

    # Verify decoder produced output
    assert isinstance(per_tile_commands, dict), "decode_words must return a dict"
    assert len(per_tile_commands) > 0, "decode produced no per-tile commands"

    # Verify structure: each key is (pkt_type, row, col) and value is a list of commands
    for (pkt_type, row, col), commands in per_tile_commands.items():
        assert isinstance(pkt_type, int), f"pkt_type must be int, got {type(pkt_type)}"
        assert isinstance(row, int), f"row must be int, got {type(row)}"
        assert isinstance(col, int), f"col must be int, got {type(col)}"
        assert isinstance(commands, list), f"commands must be list, got {type(commands)}"
        # Verify we have at least some commands decoded
        assert len(commands) > 0, f"tile ({pkt_type},{row},{col}) has no commands"


def test_build_active_plan_anchor_every_batch_and_packs():
    active = {"1|2|0": {"PERF_CNT_2", "LOCK_STALL"},
              "1|0|2": {f"D{i}" for i in range(10)}}   # shim 10 events -> 2 batches
    plan = tc.build_active_plan(active)
    assert len(plan["batches"]) == 2
    for b in plan["batches"]:
        assert b["1|2|0"][0] == "PERF_CNT_2"          # anchor slot 0 every batch
    # shim events split across the two batches, 8 then 2
    assert len(plan["batches"][0]["1|0|2"]) == 8
    assert len(plan["batches"][1]["1|0|2"]) == 2


def test_build_active_plan_includes_every_tile_in_every_batch():
    # Regression for the missing-tile bug: when one module's events exhaust in
    # fewer batches than another, the later batches must still include the short
    # module (with an empty chunk -> configure_batch writes 8 NONEs to disable
    # its compile-time trace events, preventing unconfigured-slot CaptureErrors).
    #
    # shim: 9 events -> ceil(9/8) = 2 batches (cap=8, no anchor slot reserved)
    # memtile: 3 events -> ceil(3/8) = 1 batch (fits without a second)
    # core (anchor tile): PERF_CNT_2 + INSTR_VECTOR -> cap=7, fits in 1 batch
    # nb = max(2, 1, 1) = 2; second batch must include ALL THREE tiles.
    active = {
        "0|0|2": {f"DMA_S2MM_{i}_START_TASK" for i in range(2)} | {
                  f"DMA_MM2S_{i}_START_TASK" for i in range(2)} | {
                  f"DMA_S2MM_{i}_FINISHED_TASK" for i in range(2)} | {
                  f"DMA_MM2S_{i}_FINISHED_TASK" for i in range(2)} | {
                  "DMA_S2MM_0_STREAM_STARVATION"},   # 9 distinct events
        "0|1|3": {"PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_RUNNING_2"},  # 3 events
        "0|2|0": {"PERF_CNT_2", "INSTR_VECTOR"},   # anchor tile
    }
    assert len(active["0|0|2"]) == 9, "shim set must have exactly 9 to force 2 batches"
    plan = tc.build_active_plan(active, anchor="PERF_CNT_2", anchor_tile="0|2|0")
    assert len(plan["batches"]) >= 2, "expected at least 2 batches (shim overflows cap=8)"
    tiles = set(active)
    for idx, b in enumerate(plan["batches"]):
        missing = tiles - set(b)
        assert not missing, (
            f"batch {idx} is missing tile(s) {missing}; absent tiles keep their "
            "compile-time trace config and cause unconfigured-slot CaptureErrors on HW"
        )
