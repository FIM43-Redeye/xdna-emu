from pathlib import Path
from config_extract.dump_model import load_dump, LockOrder

_FIX = (Path(__file__).resolve().parent / "fixtures"
        / "add_one_using_dma.config.json")


def test_compute_tile_lock_order_loaded():
    dump = load_dump(str(_FIX))
    compute = [t for t in dump.tiles if t.kind == "compute"]
    assert compute, "fixture must have a compute tile"
    los = [t.lock_order for t in compute if t.lock_order is not None]
    assert los, "compute tile must carry a lock_order fact"
    lo = los[0]
    assert isinstance(lo, LockOrder)
    assert lo.acq_pc < lo.rel_pc


def test_lock_order_absent_is_none():
    # A tile dict without lock_order loads with lock_order=None (backward-compat).
    from config_extract.dump_model import _load_tile
    minimal = {"col": 0, "row": 0, "kind": "shim", "ports": [],
               "event_port_selection": [None] * 8, "dma_channels": [],
               "bds": [], "locks": []}
    t = _load_tile(minimal)
    assert t.lock_order is None
