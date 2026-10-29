"""Tests for tools/dma-fill-measure.py.

These exercise the pure metric-extraction layer with hand-built event lists.
The CLI entrypoint is covered indirectly: the dataclasses + extractor cover
the only nontrivial logic in the tool; everything else is plumbing.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "dma-fill-measure.py"


def _load_tool():
    """Import the hyphenated tool as a module so tests can call into it."""
    spec = importlib.util.spec_from_file_location("dma_fill_measure", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dma_fill_measure"] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


def _ev(col, row, pkt, name, ts, slot=0):
    return {"col": col, "row": row, "pkt_type": pkt, "slot": slot, "name": name, "ts": ts}


def test_total_cycles_uses_min_max_span():
    events = [
        _ev(0, 2, 0, "INSTR_VECTOR", 100),
        _ev(0, 2, 0, "INSTR_VECTOR", 250),
        _ev(0, 2, 0, "MEMORY_STALL", 175),
    ]
    by_tile = tool.extract_tile_metrics(events)
    m = by_tile[(0, 2, 0)]
    assert m.event_count == 3
    assert m.total_cycles == 150  # 250 - 100


def test_dma_first_start_and_finished_tracked_independently():
    events = [
        _ev(0, 2, 1, "DMA_S2MM_0_FINISHED_TASK", 500),  # earliest finished
        _ev(0, 2, 1, "DMA_S2MM_0_START_TASK", 200),     # earliest start
        _ev(0, 2, 1, "DMA_S2MM_1_START_TASK", 300),
        _ev(0, 2, 1, "DMA_MM2S_0_FINISHED_TASK", 800),
    ]
    by_tile = tool.extract_tile_metrics(events)
    m = by_tile[(0, 2, 1)]
    assert m.t_first_dma_start == 200
    assert m.t_first_dma_finished == 500
    assert m.dma_roundtrip == 300


def test_dma_roundtrip_is_none_when_only_one_side_present():
    events = [_ev(0, 2, 1, "DMA_S2MM_0_START_TASK", 200)]
    by_tile = tool.extract_tile_metrics(events)
    m = by_tile[(0, 2, 1)]
    assert m.t_first_dma_start == 200
    assert m.t_first_dma_finished is None
    assert m.dma_roundtrip is None


def test_dma_roundtrip_clamps_negative_to_none():
    # FINISHED predates START -- stale BD; calibration should ignore.
    events = [
        _ev(0, 2, 1, "DMA_S2MM_0_FINISHED_TASK", 100),
        _ev(0, 2, 1, "DMA_S2MM_0_START_TASK", 500),
    ]
    by_tile = tool.extract_tile_metrics(events)
    assert by_tile[(0, 2, 1)].dma_roundtrip is None


def test_acq_req_and_lock_stall_tracked_on_core():
    events = [
        _ev(0, 2, 0, "INSTR_LOCK_ACQUIRE_REQ", 50),
        _ev(0, 2, 0, "INSTR_LOCK_ACQUIRE_REQ", 700),  # later occurrence ignored
        _ev(0, 2, 0, "LOCK_STALL", 60),
        _ev(0, 2, 0, "LOCK_STALL", 75),
    ]
    by_tile = tool.extract_tile_metrics(events)
    m = by_tile[(0, 2, 0)]
    assert m.t_first_acq_req == 50
    assert m.t_first_lock_stall == 60


def test_pair_metrics_combine_core_and_memmod():
    events = [
        _ev(0, 2, 0, "INSTR_LOCK_ACQUIRE_REQ", 100),
        _ev(0, 2, 1, "DMA_S2MM_0_START_TASK", 80),
        _ev(0, 2, 1, "DMA_S2MM_0_FINISHED_TASK", 900),
    ]
    by_tile = tool.extract_tile_metrics(events)
    pairs = tool.pair_core_with_memmod(by_tile)
    assert (0, 2) in pairs
    p = pairs[(0, 2)]
    assert p.core is not None and p.mem is not None
    # acq_to_finish = first DMA finished - first acq req = 900 - 100 = 800
    assert p.acq_to_finish == 800


def test_pair_metrics_none_when_only_one_side_present():
    events = [_ev(0, 2, 0, "INSTR_LOCK_ACQUIRE_REQ", 100)]
    by_tile = tool.extract_tile_metrics(events)
    pairs = tool.pair_core_with_memmod(by_tile)
    p = pairs[(0, 2)]
    assert p.core is not None
    assert p.mem is None
    assert p.acq_to_finish is None


def test_separate_tiles_do_not_cross_contaminate():
    events = [
        _ev(0, 2, 0, "INSTR_LOCK_ACQUIRE_REQ", 50),
        _ev(0, 3, 0, "INSTR_LOCK_ACQUIRE_REQ", 200),  # different tile
    ]
    by_tile = tool.extract_tile_metrics(events)
    assert by_tile[(0, 2, 0)].t_first_acq_req == 50
    assert by_tile[(0, 3, 0)].t_first_acq_req == 200


def test_metrics_to_rows_attaches_acq_to_finish_to_core_row():
    events = [
        _ev(0, 2, 0, "INSTR_LOCK_ACQUIRE_REQ", 100),
        _ev(0, 2, 1, "DMA_S2MM_0_FINISHED_TASK", 700),
        _ev(0, 2, 1, "DMA_S2MM_0_START_TASK", 80),
    ]
    by_tile = tool.extract_tile_metrics(events)
    pairs = tool.pair_core_with_memmod(by_tile)
    rows = tool.metrics_to_rows("test1", "chess", "hw", by_tile, pairs)
    core_rows = [r for r in rows if r["pkt_type"] == 0]
    mem_rows = [r for r in rows if r["pkt_type"] == 1]
    assert len(core_rows) == 1 and len(mem_rows) == 1
    assert core_rows[0]["acq_to_finish"] == 600  # 700 - 100
    assert mem_rows[0]["acq_to_finish"] == ""    # only on core row


def test_discover_pairs_groups_hw_emu_by_test_compiler(tmp_path):
    # Synthesize the directory layout: <name>.<compiler>.<side>/trace_raw.bin.
    layout = [
        "add_one_using_dma.chess.hw",
        "add_one_using_dma.chess.emu",
        "add_one_using_dma.peano.emu",  # peano EMU only
        "passthrough.chess.hw",         # chess HW only
        "not_a_test_dir",               # ignored
    ]
    for d in layout:
        (tmp_path / d).mkdir()
        if "." in d:
            (tmp_path / d / "trace_raw.bin").write_bytes(b"\x00" * 64)

    pairs = tool.discover_pairs(tmp_path, tmp_path / "no-build-base-here")
    by_key = {(p.name, p.compiler): p for p in pairs}

    assert ("add_one_using_dma", "chess") in by_key
    assert ("add_one_using_dma", "peano") in by_key
    assert ("passthrough", "chess") in by_key

    p_full = by_key[("add_one_using_dma", "chess")]
    assert p_full.hw_bin and p_full.emu_bin

    p_emu_only = by_key[("add_one_using_dma", "peano")]
    assert p_emu_only.hw_bin is None and p_emu_only.emu_bin

    p_hw_only = by_key[("passthrough", "chess")]
    assert p_hw_only.hw_bin and p_hw_only.emu_bin is None


def test_discover_pairs_resolves_mlir_when_present(tmp_path):
    # results dir
    results = tmp_path / "results"
    results.mkdir()
    (results / "kernA.chess.emu").mkdir()
    (results / "kernA.chess.emu" / "trace_raw.bin").write_bytes(b"\x00")

    # build base with the expected MLIR location
    build_base = tmp_path / "build"
    mlir = build_base / "kernA" / "chess" / "aie_arch.mlir.prj" / "input_with_addresses.mlir"
    mlir.parent.mkdir(parents=True)
    mlir.write_text("dummy")

    pairs = tool.discover_pairs(results, build_base)
    assert len(pairs) == 1
    assert pairs[0].mlir == mlir
