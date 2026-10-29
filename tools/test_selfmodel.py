"""Offline tests for the static self-model against the add_one fixture."""
import pytest
from pathlib import Path
from inference.selfmodel import (
    complete_menu, swept_menu, _event_tier,
    enumerate_configured_events, candidate_pairs_from_dump, legal_batch,
)
from inference.planner import Batch
from trace_capture import load_event_ids, PKT_TO_TILE_TYPE, build_active_plan

_FIXTURE = (Path(__file__).resolve().parent
            / "config_extract" / "fixtures" / "add_one_using_dma.config.json")


def _dump():
    from config_extract.dump_model import load_dump
    return load_dump(str(_FIXTURE))


@pytest.fixture(autouse=True)
def _reset_menu_caches():
    # complete_menu / swept_menu memoize at module scope; reset around each test
    # so monkeypatching load_event_ids in a future test can't leak a stale cache.
    import inference.selfmodel as sm
    sm._COMPLETE_CACHE = None
    sm._SWEPT_CACHE = None
    yield
    sm._COMPLETE_CACHE = None
    sm._SWEPT_CACHE = None


def test_complete_menu_names_all_validate():
    # Every universe name must exist in the toolchain table for its module.
    menu = complete_menu()
    for pkt, names in menu.items():
        ids = load_event_ids(PKT_TO_TILE_TYPE[pkt])
        for n in names:
            assert n in ids, (pkt, n)


def test_complete_menu_excludes_only_none():
    # D1: the universe drops NOTHING but NONE. TRUE / GROUP_* / BROADCAST_* stay.
    for pkt, names in complete_menu().items():
        assert "NONE" not in names
        assert "TRUE" in names
        assert any(n.startswith("GROUP_") for n in names)
        assert any(n.startswith("BROADCAST_") for n in names)


def test_swept_menu_is_flood_safe():
    # The default sweep excludes the stateful (always-on) flood-risk tier.
    for pkt, names in swept_menu().items():
        assert "TRUE" not in names
        assert "ACTIVE" not in names
        assert "DISABLED" not in names
        assert "DEBUG_HALTED" not in names
        assert not any(n.startswith("PORT_IDLE_") for n in names)


def test_swept_menu_subset_of_complete():
    comp, swp = complete_menu(), swept_menu()
    for pkt in comp:
        assert set(swp[pkt]) <= set(comp[pkt])


def test_swept_menu_keeps_measurement_port_running():
    # PORT_RUNNING is measurement, NOT stateful -- it must survive into the sweep.
    assert any(n.startswith("PORT_RUNNING_") for n in swept_menu()[3])  # memtile


def test_swept_menu_measurement_before_infra():
    # measurement-tier events sort ahead of infra-tier in the swept order, so
    # build_active_plan packs the useful events into the earliest batches.
    for pkt, names in swept_menu().items():
        tiers = [_event_tier(n) for n in names]
        last_meas = max((i for i, t in enumerate(tiers) if t == "measurement"),
                        default=-1)
        first_infra = next((i for i, t in enumerate(tiers) if t == "infra"),
                           len(names))
        assert last_meas < first_infra, (pkt, list(zip(names, tiers)))


def test_event_tiers():
    assert _event_tier("DMA_MM2S_0_START_TASK") == "measurement"
    assert _event_tier("PORT_RUNNING_3") == "measurement"
    assert _event_tier("PERF_CNT_2") == "measurement"
    assert _event_tier("BROADCAST_15") == "infra"
    assert _event_tier("GROUP_STALL") == "infra"
    assert _event_tier("USER_EVENT_0") == "infra"
    assert _event_tier("TRUE") == "stateful"
    assert _event_tier("PORT_IDLE_0") == "stateful"
    assert _event_tier("ACTIVE") == "stateful"


def test_swept_menu_fills_the_reviewer_gaps():
    menu = swept_menu()
    # memtile (pkt 3): DMA task boundaries (named *_SEL0/SEL1_*_TASK) now reachable.
    assert any("DMA" in n and "START_TASK" in n for n in menu[3]), menu[3]
    # memmod (pkt 1): FINISHED_TASK, absent from the hand-list, now reachable.
    assert any("FINISHED_TASK" in n for n in menu[1]), menu[1]


def test_swept_menu_no_regression_vs_handlist():
    # Every event the legacy hand-list enumerated stays enumerable.
    menu = swept_menu()
    assert "PORT_RUNNING_0" in menu[3] and "PORT_RUNNING_4" in menu[3]
    assert "DMA_S2MM_0_START_TASK" in menu[2] and "DMA_MM2S_0_START_TASK" in menu[2]
    assert "PERF_CNT_2" in menu[0]
    assert "DMA_MM2S_0_START_TASK" in menu[1] and "EDGE_DETECTION_EVENT_0" in menu[1]


def test_enumeration_builds_legal_batches():
    # The (larger) swept menu must still pack into <=8-slot legal batches.
    evs = enumerate_configured_events(_dump(), start_col=1)
    active = {}
    for k in evs:
        col, row, pkt, name = k.split("|")
        active.setdefault(f"{col}|{row}|{pkt}", set()).add(name)
    plan = build_active_plan(active)
    for b in plan["batches"]:
        assert legal_batch(Batch(tiles=b)), b


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
