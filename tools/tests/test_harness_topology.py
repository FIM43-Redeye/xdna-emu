"""Tests for tools/harness/topology.py: schema, JSON round trip, loader."""
import dataclasses
import json
from pathlib import Path

import pytest

from harness.topology import (
    Buffer, Cascade, Core, CoreStream, SharedLock, TileRef, Topology, TopologyFormatError,
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
    assert (s.src_port, s.dst_port, s.via_fifo) == (0, 0, False)


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
    resolved = dataclasses.replace(t, cores=tuple(
        dataclasses.replace(c, source=tmp_path / c.source) for c in t.cores))
    assert load_topology(p) == resolved


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


# -- compiler is a package-level property ------------------------------------

def test_compiler_defaults_to_peano_and_round_trips():
    t = Topology("t", cores=())
    assert t.compiler == "peano"
    assert to_dict(t)["compiler"] == "peano"
    t2 = dataclasses.replace(_topology(), compiler="peano")
    assert topology_from_dict(to_dict(t2)) == t2


def test_dict_without_compiler_key_means_peano():
    d = to_dict(Topology("t", cores=()))
    del d["compiler"]
    assert topology_from_dict(d).compiler == "peano"


def test_validator_rejects_unknown_compilers():
    from harness.validate import validate_topology
    t = Topology("t", (Core("c", TileRef(0, 2), Path("c.s")),), device="auto", compiler="gcc")
    assert "unsupported_compiler" in {e.code for e in validate_topology(t)}


def test_validator_accepts_chess_on_aie2():
    from harness.validate import validate_topology
    t = Topology("t", (Core("c", TileRef(0, 2), Path("c.cc")),), device="npu1", compiler="chess")
    assert validate_topology(t) == []


def test_validator_rejects_chess_beyond_aie2():
    from harness.validate import validate_topology
    t = Topology("t", (Core("c", TileRef(0, 2), Path("c.cc")),), device="npu2", compiler="chess")
    assert "chess_aie2_only" in {e.code for e in validate_topology(t)}


def test_a_core_cannot_pick_its_own_compiler():
    """aiecc links per package, so mixed compilers are unrepresentable, not just rejected."""
    d = to_dict(Topology("t", (Core("c", TileRef(0, 2), Path("c.cc")),)))
    d["cores"][0]["compiler"] = "chess"
    with pytest.raises(TopologyFormatError, match="unknown key"):
        topology_from_dict(d)


# -- relative sources resolve against the topology file ----------------------

def test_load_topology_resolves_relative_sources(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    t = Topology("t", (Core("c", TileRef(0, 2), Path("kernel.s")),
                       Core("e", TileRef(0, 3), Path("/abs/e.s"))))
    (d / "topology.json").write_text(json.dumps(to_dict(t)))
    got = load_topology(d / "topology.json")
    assert got.cores[0].source == d / "kernel.s"
    assert got.cores[1].source == Path("/abs/e.s")


def test_via_fifo_round_trips_and_is_absent_from_plain_streams():
    t = Topology("t", cores=(), links=(CoreStream("a", "b", via_fifo=True), CoreStream("a", "b")))
    d = to_dict(t)
    assert d["links"][0]["via_fifo"] is True
    assert "via_fifo" not in d["links"][1], "plain streams serialize as before (case hashes stay)"
    assert topology_from_dict(d) == t


def test_shared_lock_round_trips_and_defaults_init_to_zero():
    t = Topology("t", cores=(), links=(SharedLock("a", "b", 15, init=3), SharedLock("a", "a", 4)))
    assert SharedLock("a", "b", 1).init == 0
    d = to_dict(t)
    assert d["links"][0] == {"kind": "lock", "src": "a", "dst": "b", "lock_id": 15, "init": 3}
    assert topology_from_dict(d) == t


def test_shared_lock_needs_an_id():
    d = {"name": "t", "cores": [], "links": [{"kind": "lock", "src": "a", "dst": "b"}]}
    with pytest.raises(TopologyFormatError, match="lock_id"):
        topology_from_dict(d)


def test_sequence_writes_round_trip_and_default_to_none():
    from harness.topology import SequenceWrite
    assert Topology("t", cores=()).sequence_writes == ()
    t = dataclasses.replace(_topology(), sequence_writes=(
        SequenceWrite(0, 2, 0x34000, 0x80000000), SequenceWrite(0, 0, 0x3452C, 0)))
    d = to_dict(t)
    assert d["sequence_writes"][0] == {"col": 0, "row": 2, "address": 0x34000, "value": 0x80000000}
    assert topology_from_dict(json.loads(json.dumps(d))) == t
    assert "sequence_writes" not in to_dict(_topology())


def test_sequence_write_value_must_fit_32_bits():
    d = to_dict(_topology())
    d["sequence_writes"] = [{"col": 0, "row": 2, "address": 0x34000, "value": 1 << 32}]
    with pytest.raises(TopologyFormatError):
        topology_from_dict(d)
