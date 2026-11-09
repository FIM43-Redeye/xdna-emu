"""Tests for tools/harness/emit.py: the MLIR text, no toolchain needed."""
import dataclasses
from pathlib import Path

import pytest

from harness.device_model import load_device_model
from harness.emit import EmitError, emit_mlir, kernel_object, kernel_symbol
from harness.plan import plan_channels
from harness.topology import Buffer, Cascade, Core, CoreStream, SharedLock, TileRef, Topology


@pytest.fixture(scope="module")
def dev():
    return load_device_model("npu1")


def core(name, col, row, ins=(), outs=()):
    args = [Buffer(f"i{k}", "in", n) for k, n in enumerate(ins)]
    args += [Buffer(f"o{k}", "out", n) for k, n in enumerate(outs)]
    return Core(name, TileRef(col, row), Path(f"{name}.s"), args=tuple(args))


def topo(*cores, links=(), **kw):
    return Topology("t", cores=tuple(cores), links=tuple(links), **kw)


def emit(t, dev, phase=0):
    plan = plan_channels(t, dev)
    return emit_mlir(t, plan, phase, dev)


def test_single_core_direct(dev):
    text = emit(topo(core("a", 0, 2, ins=(64,), outs=(128,))), dev)
    assert "aie.device(npu1)" in text
    assert "aie.objectfifo @of_a_i0(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<16xi32>>" in text
    assert "aie.objectfifo @of_a_o0(%tile_0_2, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<32xi32>>" in text
    assert 'func.func private @k_a(memref<16xi32>, memref<32xi32>) attributes {link_with = "a.o"}' in text
    assert "aie.objectfifo.acquire @of_a_i0(Consume, 1)" in text
    assert "aie.objectfifo.acquire @of_a_o0(Produce, 1)" in text
    assert "func.call @k_a(%a0, %a1)" in text
    assert "aie.runtime_sequence(%in : memref<16xi32>, %buf : memref<16xi32>, %out : memref<32xi32>)" in text
    assert "metadata = @of_a_o0, id = 1 : i64, issue_token = true" in text
    assert "metadata = @of_a_i0, id = 0 : i64}" in text
    assert "aiex.npu.dma_wait {symbol = @of_a_o0}" in text
    assert "objectfifo.link" not in text


def test_output_is_deterministic(dev):
    t = topo(*[core(f"c{r}", 0, r, ins=(64,), outs=(64,)) for r in range(2, 6)])
    assert emit(t, dev) == emit(t, dev)


def test_empty_args_core_and_minimum_one_element(dev):
    a = core("a", 0, 3)                       # no args: cascade producer
    b = core("b", 0, 2, outs=(64,))
    text = emit(topo(a, b, links=[Cascade("a", "b")]), dev)
    assert 'func.func private @k_a() attributes {link_with = "a.o"}' in text
    assert "func.call @k_a() : () -> ()" in text
    assert "aie.cascade_flow(%tile_0_3, %tile_0_2)" in text
    assert "%buf : memref<1xi32>" in text     # no inputs: one element minimum


def test_core_stream_flow(dev):
    a, b = core("a", 0, 3), core("b", 0, 2, outs=(64,))
    text = emit(topo(a, b, links=[CoreStream("a", "b", 1, 0)]), dev)
    assert "aie.flow(%tile_0_3, Core : 1, %tile_0_2, Core : 0)" in text


def test_core_stream_through_the_switch_fifo(dev):
    """via_fifo routes the flow through the source tile's stream-switch FIFO 0 (AM020 ch.2:74)."""
    a, b = core("a", 0, 3), core("b", 0, 2, outs=(64,))
    text = emit(topo(a, b, links=[CoreStream("a", "b", via_fifo=True)]), dev)
    assert "aie.flow(%tile_0_3, Core : 0, %tile_0_3, FIFO : 0)" in text
    assert "aie.flow(%tile_0_3, FIFO : 0, %tile_0_2, Core : 0)" in text
    assert "aie.flow(%tile_0_3, Core : 0, %tile_0_2, Core : 0)" not in text


def test_staged_join_and_distribute_use_element_offsets(dev):
    cores = [core(f"c{r}", 0, r, ins=(64,), outs=(64,)) for r in range(2, 6)]
    text = emit(topo(*cores), dev)
    assert ("aie.objectfifo.link [@of_grp0_0_in] -> "
            "[@of_c2_i0, @of_c3_i0, @of_c4_i0, @of_c5_i0]([] [0, 16, 32, 48])") in text
    assert ("aie.objectfifo.link [@of_c2_o0, @of_c3_o0, @of_c4_o0, @of_c5_o0] -> "
            "[@of_grp0_1_out]([0, 16, 32, 48] [])") in text
    assert "aie.objectfifo @of_grp0_0_in(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<64xi32>>" in text
    assert "aie.objectfifo @of_c2_i0(%tile_0_1, {%tile_0_2}" in text
    # one host transfer per group, naming the shim-side fifo
    assert text.count("aiex.npu.dma_memcpy_nd") == 2
    assert "metadata = @of_grp0_1_out" in text and "metadata = @of_grp0_0_in" in text


def test_borrowed_route_uses_other_columns_shim(dev):
    # Borrowing never happens at real limits (the stream-switch capacity binds
    # first), so plan against a memtile too small to stage anything.
    small = dataclasses.replace(dev, mem_tile_size=4)
    cores = [core(f"c{r}", 1, r, ins=(64,)) for r in range(2, 5)]
    plan = plan_channels(topo(*cores), small)
    assert any(r.borrowed for r in plan.phases[0].routes)
    text = emit_mlir(topo(*cores), plan, 0, dev)
    assert "%tile_1_0 = aie.tile(1, 0)" in text


def test_phases_emit_only_their_cores(dev):
    cores = [core(f"c{col}_{r}", col, 2 + r, ins=(64, 64), outs=(64, 64))
             for col in range(4) for r in range(4)]
    t = topo(*cores)
    plan = plan_channels(t, dev)
    assert len(plan.phases) == 2
    first, second = (emit_mlir(t, plan, i, dev) for i in (0, 1))
    for name in plan.phases[0].cores:
        assert f"@k_{name}(" in first and f"@k_{name}(" not in second
    for name in plan.phases[1].cores:
        assert f"@k_{name}(" in second and f"@k_{name}(" not in first
    assert "of_grp0_" in first and "of_grp1_" in second


def test_phase_out_of_range(dev):
    t = topo(core("a", 0, 2, ins=(64,)))
    plan = plan_channels(t, dev)
    with pytest.raises(EmitError):
        emit_mlir(t, plan, 1, dev)


def test_identifier_collision_is_an_error(dev):
    t = topo(core("a-b", 0, 2, ins=(64,)), core("a_b", 0, 3, ins=(64,)))
    plan = plan_channels(t, dev)
    with pytest.raises(EmitError, match="both reduce"):
        emit_mlir(t, plan, 0, dev)


def test_names_are_reduced_to_identifier_characters():
    c = core("my core.1", 0, 2)
    assert kernel_symbol(c) == "k_my_core_1" and kernel_object(c) == "my_core_1.o"


def test_shim_bd_budget_is_checked(dev):
    small = dataclasses.replace(
        dev, _type_configs={**dev._type_configs,
                            "shim_noc": {**dev._type_configs["shim_noc"], "num_bds": 1}})
    t = topo(core("a", 0, 2, ins=(64,), outs=(64,)))
    plan = plan_channels(t, dev)
    with pytest.raises(EmitError, match="BDs"):
        emit_mlir(t, plan, 0, small)


def test_shared_lock_is_declared_on_the_owner_tile_with_its_init(dev):
    """The lock gets a fixed ID (objectfifo locks are assigned around it) and an init, so
    every load of the design starts it from the same count."""
    a, b = core("a", 0, 2, outs=(64,)), core("b", 0, 3)
    text = emit(topo(a, b, links=[SharedLock("a", "b", 15, init=2)]), dev)
    assert ('%lock_0_2_15 = aie.lock(%tile_0_2, 15) {init = 2 : i32, sym_name = "lock_0_2_15"}'
            in text)
    assert text.index("aie.lock(") > text.index("%tile_0_3 = aie.tile(0, 3)")


def test_sequence_writes_precede_the_transfers_in_order(dev):
    """Host register writes run first, in the order given: probes place them relative to the
    first transfer (e.g. a timer reset, then padding)."""
    from harness.topology import SequenceWrite
    t = dataclasses.replace(topo(core("a", 0, 2, outs=(64,))), sequence_writes=(
        SequenceWrite(0, 2, 0x34000, 0x80000000), SequenceWrite(0, 2, 0x3152C, 7)))
    text = emit(t, dev)
    first = text.index("aiex.npu.write32(%w0_addr, %w0_val) {column = 0 : i32, row = 2 : i32} : i32, i32")
    second = text.index("aiex.npu.write32(%w1_addr, %w1_val) {column = 0 : i32, row = 2 : i32} : i32, i32")
    assert first < second < text.index("aiex.npu.dma_memcpy_nd")
    assert "%w0_addr = arith.constant 213000 : i32" not in text
    assert "%w0_addr = arith.constant 212992 : i32" in text
    assert "%w0_val = arith.constant -2147483648 : i32" in text
    assert "%w1_val = arith.constant 7 : i32" in text
    assert "aiex.npu.write32" not in emit(topo(core("a", 0, 2, outs=(64,))), dev)
