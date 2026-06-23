"""Offline tests for the static self-model against the add_one fixture."""
from pathlib import Path
from inference.selfmodel import (enumerate_configured_events,
                                 candidate_pairs_from_dump, legal_batch)
from inference.planner import Batch

_FIXTURE = (Path(__file__).resolve().parent
            / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _dump():
    from config_extract.dump_model import load_dump
    return load_dump(str(_FIXTURE))


def test_enumeration_superset_of_known_add_one_events():
    evs = set(enumerate_configured_events(_dump(), start_col=1))
    # The known-firing add_one dataflow events must all be enumerated.
    for k in ("1|1|3|PORT_RUNNING_0", "1|1|3|PORT_RUNNING_4",
              "1|0|2|DMA_S2MM_0_START_TASK", "1|0|2|DMA_MM2S_0_START_TASK"):
        assert k in evs, k


def test_candidate_pairs_include_memtile_relay_child_parent_order():
    dump = _dump()
    configured = enumerate_configured_events(dump, start_col=1)
    pairs = candidate_pairs_from_dump(dump, configured, start_col=1)
    # (child, parent): PR4 derives from PR0 (the buffer relay).
    assert ("1|1|3|PORT_RUNNING_4", "1|1|3|PORT_RUNNING_0") in pairs
    # No duplicates.
    assert len(pairs) == len(set(pairs))


def test_legal_batch_accepts_small_rejects_oversize():
    assert legal_batch(Batch(tiles={"1|2|0": ["PERF_CNT_2", "INSTR_VECTOR"]}))
    # 9 events on one tile -> illegal (>8 slots).
    nine = [f"PORT_RUNNING_{i}" for i in range(8)] + ["PERF_CNT_2"]
    assert not legal_batch(Batch(tiles={"1|1|3": nine}))
    # Bogus event name -> illegal.
    assert not legal_batch(Batch(tiles={"1|2|0": ["NOT_A_REAL_EVENT"]}))
