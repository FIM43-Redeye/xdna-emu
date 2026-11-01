"""Tests for tools/harness/validate.py. Each check has a pass and a fail case."""
from pathlib import Path

import pytest

from harness.device_model import load_device_model
from harness.topology import (
    Buffer, Cascade, Core, CoreStream, TileRef, Topology,
)
from harness.validate import ValidationError, validate, validate_topology


@pytest.fixture(scope="module")
def dev():
    return load_device_model("npu1")


def core(name="c", col=0, row=2, args=()):
    return Core(name, TileRef(col, row), Path(f"{name}.s"), args=tuple(args))


def topo(*cores, links=(), **kw):
    return Topology("t", cores=tuple(cores), links=tuple(links), **kw)


def codes(errors):
    return [e.code for e in errors]


def test_valid_topology_has_no_errors(dev):
    t = topo(core("a", 0, 3, [Buffer("i", "in", 64), Buffer("o", "out", 64)]),
             core("b", 0, 2), links=[Cascade("a", "b"), CoreStream("a", "b")])
    assert validate(t, dev) == []


def test_errors_are_structured(dev):
    errs = validate(topo(core("a", 9, 2)), dev)
    assert isinstance(errs[0], ValidationError)
    assert errs[0].code and errs[0].message and errs[0].names == ("a",)


# -- device / tiles ----------------------------------------------------------

def test_device_mismatch(dev):
    assert "device_mismatch" in codes(validate(topo(core(), device="npu2"), dev))


def test_validate_topology_unknown_device(dev):
    errs = validate_topology(topo(core(), device="nope"))
    assert codes(errs) == ["unknown_device"]


def test_validate_topology_known_device():
    assert validate_topology(topo(core())) == []


def test_column_out_of_range(dev):
    assert codes(validate(topo(core(col=4)), dev)) == ["column_out_of_range"]
    assert validate(topo(core(col=3)), dev) == []


def test_tile_missing_row(dev):
    assert codes(validate(topo(core(row=6)), dev)) == ["tile_missing"]


def test_tile_not_core(dev):
    assert codes(validate(topo(core(row=0)), dev)) == ["tile_not_core"]  # shim
    assert codes(validate(topo(core(row=1)), dev)) == ["tile_not_core"]  # memtile


def test_one_core_per_tile(dev):
    errs = validate(topo(core("a"), core("b")), dev)
    assert codes(errs) == ["tile_shared"]
    assert set(errs[0].names) == {"a", "b"}


def test_core_names_unique(dev):
    errs = validate(topo(core("a", 0, 2), core("a", 0, 3)), dev)
    assert codes(errs) == ["core_name_duplicate"]


def test_buffer_names_unique_per_core_only(dev):
    dup = core("a", 0, 2, [Buffer("x", "in", 4), Buffer("x", "out", 4)])
    assert codes(validate(topo(dup), dev)) == ["buffer_name_duplicate"]
    same_in_two_cores = topo(core("a", 0, 2, [Buffer("x", "in", 4)]),
                             core("b", 0, 3, [Buffer("x", "in", 4)]))
    assert validate(same_in_two_cores, dev) == []


# -- buffers -----------------------------------------------------------------

@pytest.mark.parametrize("size", [0, -4, 6, 3])
def test_buffer_size_invalid(dev, size):
    t = topo(core(args=[Buffer("b", "in", size)]))
    assert "buffer_size_invalid" in codes(validate(t, dev))


def test_buffer_size_valid(dev):
    assert validate(topo(core(args=[Buffer("b", "in", 4)])), dev) == []


def test_buffer_direction_invalid(dev):
    t = topo(core(args=[Buffer("b", "inout", 4)]))  # type: ignore[arg-type]
    assert codes(validate(t, dev)) == ["buffer_direction_invalid"]


def test_data_on_out_buffer(dev):
    t = topo(core(args=[Buffer("b", "out", 4, data=bytes(4))]))
    assert codes(validate(t, dev)) == ["buffer_data_on_out"]


def test_data_length_must_match(dev):
    bad = topo(core(args=[Buffer("b", "in", 8, data=bytes(4))]))
    assert codes(validate(bad, dev)) == ["buffer_data_length"]
    ok = topo(core(args=[Buffer("b", "in", 8, data=bytes(8))]))
    assert validate(ok, dev) == []


# -- memory ------------------------------------------------------------------

def test_memory_budget_excludes_default_stack(dev):
    budget = dev.local_memory_size - dev.default_core_stack_size
    # Two channels per direction: spread budget over two in buffers.
    half = budget // 2
    fits = topo(core(args=[Buffer("a", "in", half), Buffer("b", "in", half)]))
    assert validate(fits, dev) == []
    over = topo(core(args=[Buffer("a", "in", half), Buffer("b", "in", half + 4)]))
    assert codes(validate(over, dev)) == ["core_memory_overflow"]


# -- DMA channels ------------------------------------------------------------

def test_at_most_n_in_and_n_out_buffers(dev):
    ch = dev.dma_channels("core")
    ok = [Buffer(f"i{k}", "in", 4) for k in range(ch.s2mm)] + \
         [Buffer(f"o{k}", "out", 4) for k in range(ch.mm2s)]
    assert validate(topo(core(args=ok)), dev) == []
    too_in = [Buffer(f"i{k}", "in", 4) for k in range(ch.s2mm + 1)]
    assert codes(validate(topo(core(args=too_in)), dev)) == ["too_many_in_buffers"]
    too_out = [Buffer(f"o{k}", "out", 4) for k in range(ch.mm2s + 1)]
    assert codes(validate(topo(core(args=too_out)), dev)) == ["too_many_out_buffers"]


# -- cascade -----------------------------------------------------------------

def two(src_tile, dst_tile, link):
    return topo(core("a", *src_tile), core("b", *dst_tile), links=[link])


def test_cascade_south_flow_legal(dev):
    assert validate(two((0, 3), (0, 2), Cascade("a", "b")), dev) == []


def test_cascade_east_flow_legal(dev):
    assert validate(two((0, 2), (1, 2), Cascade("a", "b")), dev) == []


@pytest.mark.parametrize("src,dst", [
    ((0, 2), (0, 3)),   # flow would go North
    ((1, 2), (0, 2)),   # flow would go West
    ((0, 3), (1, 2)),   # diagonal
    ((0, 4), (0, 2)),   # not adjacent
])
def test_cascade_adjacency_illegal(dev, src, dst):
    assert codes(validate(two(src, dst, Cascade("a", "b")), dev)) == ["cascade_adjacency"]


def test_cascade_unknown_endpoint(dev):
    t = topo(core("a"), links=[Cascade("a", "ghost")])
    errs = validate(t, dev)
    assert codes(errs) == ["link_endpoint_unknown"] and errs[0].names == ("ghost",)


def test_cascade_self_loop_is_illegal(dev):
    t = topo(core("a"), links=[Cascade("a", "a")])
    assert codes(validate(t, dev)) == ["cascade_adjacency"]


def test_at_most_one_cascade_in_and_out(dev):
    # Fan-out and fan-in are both illegal; a straight chain is fine.
    chain = topo(core("a", 0, 4), core("b", 0, 3), core("c", 0, 2),
                 links=[Cascade("a", "b"), Cascade("b", "c")])
    assert validate(chain, dev) == []
    # a feeds b (east) and a also feeds c (south): two outputs from a.
    two_out = topo(core("a", 1, 3), core("b", 2, 3), core("c", 1, 2),
                   links=[Cascade("a", "b"), Cascade("a", "c")])
    assert codes(validate(two_out, dev)) == ["cascade_multiple_out"]
    # b (east of a) and c (north of b) both feed b: two inputs into b.
    two_in = topo(core("a", 0, 3), core("b", 1, 3), core("c", 1, 4),
                  links=[Cascade("a", "b"), Cascade("c", "b")])
    assert codes(validate(two_in, dev)) == ["cascade_multiple_in"]


# -- core streams ------------------------------------------------------------

def test_core_stream_legal(dev):
    assert validate(two((0, 2), (3, 5), CoreStream("a", "b")), dev) == []


def test_core_stream_endpoints_must_exist_and_differ(dev):
    t = topo(core("a"), links=[CoreStream("a", "ghost")])
    assert codes(validate(t, dev)) == ["link_endpoint_unknown"]
    t = topo(core("a"), links=[CoreStream("a", "a")])
    assert codes(validate(t, dev)) == ["stream_endpoints_same"]


def test_core_stream_ports_within_bundle(dev):
    n = dev.stream_ports("core", "Core")
    ok = two((0, 2), (0, 3), CoreStream("a", "b", n.slave - 1, n.master - 1))
    assert validate(ok, dev) == []
    bad_src = two((0, 2), (0, 3), CoreStream("a", "b", src_port=n.slave))
    assert codes(validate(bad_src, dev)) == ["stream_port_range"]
    bad_dst = two((0, 2), (0, 3), CoreStream("a", "b", dst_port=n.master))
    assert codes(validate(bad_dst, dev)) == ["stream_port_range"]
    neg = two((0, 2), (0, 3), CoreStream("a", "b", src_port=-1))
    assert codes(validate(neg, dev)) == ["stream_port_range"]


# -- shim reserve ------------------------------------------------------------

def test_reserve_shim_s2mm(dev):
    n = dev.dma_channels("shim_noc").s2mm
    assert validate(topo(core(), reserve_shim_s2mm=n), dev) == []
    assert codes(validate(topo(core(), reserve_shim_s2mm=n + 1), dev)) == ["shim_reserve_range"]
    assert codes(validate(topo(core(), reserve_shim_s2mm=-1), dev)) == ["shim_reserve_range"]


def test_validate_reports_all_errors_without_raising(dev):
    t = topo(core("a", 9, 2, [Buffer("b", "in", 3)]), core("a", 0, 0))
    assert len(validate(t, dev)) >= 3
