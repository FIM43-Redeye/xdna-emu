"""Tests for tools/harness/topology.py: schema, JSON round trip, loader."""
import dataclasses
import json
from pathlib import Path

import pytest

from harness.topology import (
    Buffer, Cascade, Core, CoreStream, TileRef, Topology, TopologyFormatError,
    load_topology, topology_from_dict, to_dict,
)


def _topology() -> Topology:
    return Topology(
        name="cascade_pair",
        cores=(
            Core("prod", TileRef(0, 3), Path("prod.s"),
                 args=(Buffer("a", "in", 8, data=bytes(range(8))),
                       Buffer("o", "out", 4))),
            Core("cons", TileRef(0, 2), Path("cons.o"), symbol="entry"),
        ),
        links=(Cascade("prod", "cons"), CoreStream("prod", "cons", 0, 0)),
        reserve_shim_s2mm=1,
        seed=7,
    )


def test_dataclasses_are_frozen():
    t = _topology()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.name = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.cores[0].tile.col = 5


def test_defaults():
    t = Topology("t", cores=())
    assert (t.device, t.links, t.reserve_shim_s2mm, t.seed) == ("npu1", (), 0, 1)
    c = Core("c", TileRef(0, 2), Path("c.s"))
    assert (c.symbol, c.args) == ("test_kernel", ())
    assert Buffer("b", "in", 4).data is None
    s = CoreStream("a", "b")
    assert (s.src_port, s.dst_port) == (0, 0)


def test_dict_round_trip_is_identity():
    t = _topology()
    assert topology_from_dict(to_dict(t)) == t


def test_to_dict_is_json_serializable_and_stable():
    d = to_dict(_topology())
    text = json.dumps(d)
    assert json.loads(text) == d
    assert d["cores"][0]["args"][0]["data"] == "0001020304050607"
    assert d["links"][0]["kind"] == "cascade"
    assert d["links"][1]["kind"] == "stream"


def test_load_topology_from_file(tmp_path):
    t = _topology()
    p = tmp_path / "topology.json"
    p.write_text(json.dumps(to_dict(t)))
    assert load_topology(p) == t


def test_unknown_key_is_rejected():
    d = to_dict(_topology())
    d["cores"][0]["colour"] = "red"
    with pytest.raises(TopologyFormatError, match="colour"):
        topology_from_dict(d)


def test_unknown_link_kind_is_rejected():
    d = to_dict(_topology())
    d["links"][0]["kind"] = "teleport"
    with pytest.raises(TopologyFormatError, match="teleport"):
        topology_from_dict(d)


def test_missing_required_key_is_rejected():
    d = to_dict(_topology())
    del d["cores"][0]["tile"]
    with pytest.raises(TopologyFormatError, match="tile"):
        topology_from_dict(d)


def test_bad_hex_data_is_rejected():
    d = to_dict(_topology())
    d["cores"][0]["args"][0]["data"] = "zz"
    with pytest.raises(TopologyFormatError, match="data"):
        topology_from_dict(d)
