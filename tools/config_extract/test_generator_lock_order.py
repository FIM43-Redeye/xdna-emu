"""Task 4 (#140): Tests for oriented core lock-order candidate pair emission.

Verifies:
  1. generate_ledger emits a lock-order edge (kind="program",
     cite="program_order:...") for a tile with lock_order + both events fired.
  2. enumerate_configured_events includes the two lock instruction events.
  3. candidate_pairs_from_dump returns the (rel_key, acq_key) pair.
"""

from pathlib import Path

from config_extract.dump_model import load_dump
from config_extract.generator import generate_ledger, audit_ledger
from inference.selfmodel import (
    enumerate_configured_events, candidate_pairs_from_dump)

_FIX = (Path(__file__).resolve().parent / "fixtures"
        / "add_one_using_dma.config.json")
_ACQ = "1|2|0|INSTR_LOCK_ACQUIRE_REQ"
_REL = "1|2|0|INSTR_LOCK_RELEASE_REQ"


def test_lock_order_edge_emitted_and_oriented():
    dump = load_dump(str(_FIX))
    led = generate_ledger(dump, [_ACQ, _REL], start_col=1)
    # Filter to the lock pair -- the generator emits many program entries for
    # add_one from the existing through-core DMA fan-out; assert on the lock one.
    lock = [e for e in led["entries"]
            if e["a"].endswith("INSTR_LOCK_ACQUIRE_REQ")
            and e["b"].endswith("INSTR_LOCK_RELEASE_REQ")]
    assert len(lock) == 1
    e = lock[0]
    assert e["a"] == _ACQ and e["b"] == _REL and e["kind"] == "program"
    assert e["cite"].startswith("program_order:")
    # The portless lock entry must pass audit.
    assert audit_ledger(led, dump, start_col=1) == []


def test_menu_enumerates_lock_events():
    dump = load_dump(str(_FIX))
    keys = enumerate_configured_events(dump, start_col=1)
    assert _ACQ in keys and _REL in keys


def test_candidate_pairs_includes_lock_order():
    dump = load_dump(str(_FIX))
    configured = enumerate_configured_events(dump, start_col=1)
    pairs = candidate_pairs_from_dump(dump, configured, start_col=1)
    assert (_REL, _ACQ) in pairs
