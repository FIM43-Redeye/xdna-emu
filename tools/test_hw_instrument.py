# tools/test_hw_instrument.py
"""Offline tests for HwInstrument: Batch->plan translation, col reconcile,
ledger generation. No NPU -- the capture() call is monkeypatched."""
from pathlib import Path
import json
import pytest
from inference.hw_instrument import HwInstrument
from inference.planner import Batch

_FIXTURE = (Path(__file__).resolve().parent
            / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _load_dump():
    from config_extract.dump_model import load_dump
    return load_dump(str(_FIXTURE))


def test_ledger_entries_nonempty_and_oriented():
    dump = _load_dump()
    # configured events in ABSOLUTE col-1 space (the add_one active set).
    configured = ["1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4",
                  "1|2|0|PERF_CNT_2"]
    inst = HwInstrument("add_one_using_dma", dump, configured,
                        start_col=1, anchor_tile_abs="1|2|0",
                        anchor_event="PERF_CNT_2", traced_col=1,
                        n_runs=3, out_root="/tmp/unused", compiler="chess")
    entries = inst.ledger_entries()
    # Every entry is a parent->child route/program fact (a=parent, b=child).
    assert all(set(("a", "b", "kind", "cite")) <= set(e) for e in entries)
    # The memtile buffer relay PR0 -> PR4 must be present (config_path).
    assert any(e["a"] == "1|1|3|PORT_RUNNING_0" and e["b"] == "1|1|3|PORT_RUNNING_4"
               for e in entries)


def test_capture_converts_abs_to_rel_col_and_runs_n_runs(monkeypatch, tmp_path):
    dump = _load_dump()
    inst = HwInstrument("add_one_using_dma", dump,
                        ["1|2|0|PERF_CNT_2"], start_col=1,
                        anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
                        traced_col=1, n_runs=3, out_root=str(tmp_path),
                        compiler="chess")

    seen = {"plans": [], "out_dirs": []}

    # Stub the HW boundary: record the plan + out_dir, write a minimal
    # trace.events.json so load_fired can read it back.
    def fake_capture(plan, runner, *, test, out_dir, traced_col, instr):
        seen["plans"].append(plan)
        seen["out_dirs"].append(str(out_dir))
        bdir = Path(out_dir) / "batch_00" / "hw"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "trace.events.json").write_text(json.dumps(
            {"schema_version": 1, "events": [], "slot_names": {}}))
        return [{}]

    class FakeRunner:
        def __init__(self, *a, **k): pass
        def close(self): pass

    monkeypatch.setattr("inference.hw_instrument.capture", fake_capture)
    monkeypatch.setattr("inference.hw_instrument.HwRunner", FakeRunner)
    # Don't require a built kernel on disk for this offline test.
    # _discover_xclbin_insts returns a fake insts path; stub probe_slot_capacity
    # so capture() doesn't try to read bytes from that non-existent path.
    fake_insts = tmp_path / "insts.bin"
    fake_insts.write_bytes(b"\x00" * 16)  # minimal 16-byte header, no instructions
    monkeypatch.setattr("inference.hw_instrument._discover_xclbin_insts",
                        lambda test, compiler: ("aie.xclbin", str(fake_insts)))
    # All tiles are traceable in this test -- stub returns 8 for every tile.
    monkeypatch.setattr("inference.hw_instrument.probe_slot_capacity",
                        lambda data, col, row, tile_type: 8)

    batch = Batch(tiles={"1|2|0": ["INSTR_VECTOR"]})
    run_dirs = inst.capture(batch)

    assert len(run_dirs) == 3                     # n_runs run dirs
    # plan tiles were converted ABS col 1 -> REL col 0 for the patcher.
    plan = seen["plans"][0]
    tiles = {t for b in plan["batches"] for t in b}
    assert all(t.split("|")[0] == "0" for t in tiles), tiles
    # anchor injected on the anchor tile (rel "0|2|0") every batch.
    assert all("0|2|0" in b for b in plan["batches"])


def test_capture_drops_untraceable_tiles(monkeypatch, tmp_path):
    """Tiles with probe_slot_capacity==0 must be silently dropped from the plan.

    Reproduces the vector_scalar_using_dma failure where the memtile (0,1)
    has no compile-time trace and the patcher exits non-zero.
    """
    dump = _load_dump()
    # Batch that spans a shim tile (abs 1|2|0, pkt=0=core) and a memtile
    # (abs 1|1|3, pkt=3=memtile).  anchor lives on the shim/core tile.
    inst = HwInstrument("vector_scalar_using_dma", dump,
                        ["1|2|0|PERF_CNT_2", "1|1|3|PORT_RUNNING_0"],
                        start_col=1,
                        anchor_tile_abs="1|2|0", anchor_event="PERF_CNT_2",
                        traced_col=1, n_runs=1, out_root=str(tmp_path),
                        compiler="chess")

    seen = {"plans": []}

    def fake_capture(plan, runner, *, test, out_dir, traced_col, instr):
        seen["plans"].append(plan)
        bdir = Path(out_dir) / "batch_00" / "hw"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "trace.events.json").write_text(json.dumps(
            {"schema_version": 1, "events": [], "slot_names": {}}))
        return [{}]

    class FakeRunner:
        def __init__(self, *a, **k): pass
        def close(self): pass

    monkeypatch.setattr("inference.hw_instrument.capture", fake_capture)
    monkeypatch.setattr("inference.hw_instrument.HwRunner", FakeRunner)
    fake_insts = tmp_path / "insts.bin"
    fake_insts.write_bytes(b"\x00" * 16)
    monkeypatch.setattr("inference.hw_instrument._discover_xclbin_insts",
                        lambda test, compiler: ("aie.xclbin", str(fake_insts)))

    # memtile pkt=3 has capacity 0 (not compiled with trace); core pkt=0 has 8.
    def fake_probe(data, col, row, tile_type):
        return 0 if tile_type == "memtile" else 8

    monkeypatch.setattr("inference.hw_instrument.probe_slot_capacity", fake_probe)

    batch = Batch(tiles={"1|2|0": ["PERF_CNT_2"], "1|1|3": ["PORT_RUNNING_0"]})
    run_dirs = inst.capture(batch)

    assert len(run_dirs) == 1
    plan = seen["plans"][0]
    all_tiles = {t for b in plan["batches"] for t in b}
    # The memtile (rel "0|1|3") must NOT appear in the plan.
    assert "0|1|3" not in all_tiles, f"untraceable memtile leaked into plan: {all_tiles}"
    # The core tile (rel "0|2|0") must be present (traceable).
    assert "0|2|0" in all_tiles, f"core tile missing from plan: {all_tiles}"
