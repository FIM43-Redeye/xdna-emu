"""Tests for tools/harness/plan.py: one test group per planner rule.

Shrinking the memtile (dataclasses.replace on the loaded model) is how the
borrow and phase-split rules are reached without 17+ real buffers: it makes
staging infeasible, which is exactly the condition those rules react to.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from harness.device_model import load_device_model
from harness.plan import ChannelPlan, PlanError, plan_channels, resolve_device
from harness.topology import Buffer, Cascade, Core, CoreStream, TileRef, Topology


@pytest.fixture(scope="module")
def dev():
    return load_device_model("npu1")


def core(name, col, row, ins=(), outs=()):
    args = [Buffer(f"i{k}", "in", n) for k, n in enumerate(ins)]
    args += [Buffer(f"o{k}", "out", n) for k, n in enumerate(outs)]
    return Core(name, TileRef(col, row), Path(f"{name}.s"), args=tuple(args))


def topo(*cores, links=(), device="npu1", **kw):
    return Topology("t", cores=tuple(cores), links=tuple(links), device=device, **kw)


def column_cores(col, n=4, ins=(64,), outs=(64,), prefix="c"):
    return [core(f"{prefix}{col}_{r}", col, 2 + r, ins, outs) for r in range(n)]


def routes(plan, phase=0):
    return {(r.core, r.buffer): r for r in plan.phases[phase].routes}


def usage(plan, col, row, phase=0):
    return next(u for u in plan.phases[phase].usage if (u.col, u.row) == (col, row))


def tiny_memtile(dev, size=4):
    return dataclasses.replace(dev, mem_tile_size=size)


# -- rule 5 / basics: single core, direct -----------------------------------

def test_single_core_is_direct(dev):
    plan = plan_channels(topo(core("a", 0, 2, ins=(64,), outs=(128,))), dev)
    assert isinstance(plan, ChannelPlan)
    assert plan.device == "npu1" and len(plan.phases) == 1
    r = routes(plan)
    assert r[("a", "i0")].kind == "direct" and r[("a", "i0")].shim_col == 0
    assert r[("a", "o0")].kind == "direct" and not r[("a", "o0")].borrowed
    assert r[("a", "i0")].memtile_col is None and r[("a", "i0")].group is None
    assert (r[("a", "i0")].host_offset, r[("a", "o0")].host_offset) == (0, 0)
    assert (plan.phases[0].host_in_bytes, plan.phases[0].host_out_bytes) == (64, 128)
    assert plan.phases[0].cores == ("a",)


def test_two_buffers_per_direction_still_direct(dev):
    plan = plan_channels(topo(core("a", 0, 2, ins=(64, 64), outs=(32, 32))), dev)
    assert {r.kind for r in plan.phases[0].routes} == {"direct"}


def test_no_cores_is_an_error(dev):
    err = plan_channels(topo(), dev)
    assert isinstance(err, PlanError) and err.code == "no_cores"


# -- rule 2: memtile join / distribute --------------------------------------

def test_four_cores_in_a_column_stage_through_the_memtile(dev):
    plan = plan_channels(topo(*column_cores(0)), dev)
    p = plan.phases[0]
    r = routes(plan)
    outs = [r[(f"c0_{k}", "o0")] for k in range(4)]
    ins = [r[(f"c0_{k}", "i0")] for k in range(4)]
    assert {x.kind for x in outs} == {"memtile_join"}
    assert {x.kind for x in ins} == {"memtile_distribute"}
    assert all(x.memtile_col == 0 and x.shim_col == 0 and not x.borrowed
               for x in outs + ins)
    assert [x.link_offset for x in outs] == [0, 64, 128, 192]
    assert [x.link_offset for x in ins] == [0, 64, 128, 192]
    assert [x.host_offset for x in outs] == [0, 64, 128, 192]
    assert len(p.groups) == 2
    join = next(g for g in p.groups if g.kind == "memtile_join")
    assert join.members == tuple((f"c0_{k}", "o0") for k in range(4))
    assert join.object_bytes == 256 and join.depth == 1 and join.memtile_bytes == 256
    assert (p.host_in_bytes, p.host_out_bytes) == (256, 256)


def test_staged_column_uses_one_shim_channel_per_direction(dev):
    plan = plan_channels(topo(*column_cores(0)), dev)
    shim = usage(plan, 0, 0)
    assert (shim.mm2s_used, shim.s2mm_used) == (1, 1)
    mem = usage(plan, 0, 1)
    # join of 4: 4 S2MM + 1 MM2S; distribute of 4: 1 S2MM + 4 MM2S
    assert (mem.s2mm_used, mem.mm2s_used) == (5, 5)
    assert (mem.s2mm_available, mem.mm2s_available) == (6, 6)
    assert mem.bytes_used == 512 and mem.bytes_available == dev.mem_tile_size
    core_u = usage(plan, 0, 2)
    assert (core_u.s2mm_used, core_u.mm2s_used) == (1, 1)
    assert (core_u.s2mm_available, core_u.mm2s_available) == (2, 2)


def test_memtile_channel_limit_is_respected(dev):
    # 4 cores x 2 outputs = 8 core MM2S, but a memtile has 6 S2MM: one
    # direct, a 6-way join, and the last output borrows another column.
    plan = plan_channels(topo(*column_cores(0, ins=(), outs=(32, 32))), dev)
    assert len(plan.phases[0].groups[0].members) == 6
    assert usage(plan, 0, 1).s2mm_used == 6


def test_partial_staging_when_memtile_is_too_small_for_all(dev):
    # 3 inputs, 2 channels: one group of 3 needs 192 B; memtile of 130 B only
    # fits 2 buffers, so the first goes direct and the other two are staged.
    small = tiny_memtile(dev, 130)
    cores = column_cores(0, n=3, outs=())
    plan = plan_channels(topo(*cores), small)
    kinds = [r.kind for r in plan.phases[0].routes]
    assert kinds == ["direct", "memtile_distribute", "memtile_distribute"]
    assert usage(plan, 0, 0).mm2s_used == 2


# -- rule 1: reserve ---------------------------------------------------------

def test_reserve_forces_staging(dev):
    two = [core("a", 0, 2, outs=(64,)), core("b", 0, 3, outs=(64,))]
    direct = plan_channels(topo(*two), dev)
    assert {r.kind for r in direct.phases[0].routes} == {"direct"}
    staged = plan_channels(topo(*two, reserve_shim_s2mm=1), dev)
    assert {r.kind for r in staged.phases[0].routes} == {"memtile_join"}
    shim = usage(staged, 0, 0)
    assert shim.s2mm_used == 1 and shim.s2mm_available == 1 and shim.s2mm_reserved == 1


def test_reserve_does_not_touch_inputs(dev):
    plan = plan_channels(topo(core("a", 0, 2, ins=(64, 64)), reserve_shim_s2mm=2), dev)
    assert {r.kind for r in plan.phases[0].routes} == {"direct"}


def test_reserving_every_s2mm_makes_outputs_unplaceable(dev):
    err = plan_channels(topo(core("a", 0, 2, outs=(64,)), reserve_shim_s2mm=2), dev)
    assert isinstance(err, PlanError) and err.code == "unplaceable"
    assert err.names == ("a",)


# -- rule 3: borrow ----------------------------------------------------------

def test_borrow_takes_nearest_column_with_lower_column_tiebreak(dev):
    small = tiny_memtile(dev)
    # Column 1 has 3 inputs, own shim has 2 MM2S. Staging is impossible, so the
    # third borrows: columns 0 and 2 are equally near, the lower wins.
    plan = plan_channels(topo(*column_cores(1, n=3, outs=())), small)
    rs = plan.phases[0].routes
    assert [(r.shim_col, r.borrowed) for r in rs] == [(1, False), (1, False), (0, True)]
    assert all(r.kind == "direct" for r in rs)
    assert usage(plan, 0, 0).mm2s_used == 1


def test_borrow_prefers_nearest_even_if_higher_index(dev):
    small = tiny_memtile(dev)
    # Column 0 over-subscribed and column 1 already full (2 inputs of its own):
    # nearest free shim is column 2.
    cores = column_cores(0, n=3, outs=(), prefix="a") + column_cores(1, n=2, outs=(), prefix="b")
    plan = plan_channels(topo(*cores), small)
    r = routes(plan)
    assert (r[("a0_2", "i0")].shim_col, r[("a0_2", "i0")].borrowed) == (2, True)


# -- rule 4: phases ----------------------------------------------------------

def sixteen(prefix="c", ins=(64,), outs=()):
    return [c for col in range(4) for c in column_cores(col, ins=ins, outs=outs, prefix=prefix)]


def test_whole_array_fits_in_one_phase_when_staging_is_possible(dev):
    plan = plan_channels(topo(*sixteen()), dev)
    assert len(plan.phases) == 1


def test_phase_split_when_array_is_oversubscribed(dev):
    small = tiny_memtile(dev)
    cs = sixteen()
    plan = plan_channels(topo(*cs), small)
    assert [len(p.cores) for p in plan.phases] == [8, 8]
    assert [c for p in plan.phases for c in p.cores] == [c.name for c in cs]
    assert plan.phases[1].index == 1
    # offsets restart per phase
    assert plan.phases[1].routes[0].host_offset == 0


def test_linked_cores_stay_in_one_phase(dev):
    small = tiny_memtile(dev)
    cs = sixteen()
    # Link core 7 and 8, which a plain 8-per-phase split would separate.
    t = topo(*cs, links=[CoreStream(cs[7].name, cs[8].name)])
    plan = plan_channels(t, small)
    where = {c: p.index for p in plan.phases for c in p.cores}
    assert where[cs[7].name] == where[cs[8].name]
    assert sorted(where) == sorted(c.name for c in cs)


def test_cascade_link_counts_as_a_group_without_dma_channels(dev):
    a, b = core("a", 0, 3, ins=(64,)), core("b", 0, 2, outs=(64,))
    plan = plan_channels(topo(a, b, links=[Cascade("a", "b")]), dev)
    assert len(plan.phases) == 1
    assert {r.kind for r in plan.phases[0].routes} == {"direct"}


def test_group_too_big_for_one_phase_is_a_plan_error(dev):
    small = tiny_memtile(dev)
    cs = sixteen()[:9]   # 9 input buffers, 8 shim MM2S channels in the array
    links = [CoreStream(cs[i].name, cs[i + 1].name) for i in range(8)]
    err = plan_channels(topo(*cs, links=links), small)
    assert isinstance(err, PlanError) and err.code == "unplaceable"
    assert set(err.names) == {c.name for c in cs}


# -- device resolution -------------------------------------------------------

@pytest.mark.parametrize("col,expected", [(0, "npu1_1col"), (1, "npu1_2col"),
                                          (2, "npu1_3col"), (3, "npu1")])
def test_auto_picks_smallest_covering_npu1_device(col, expected):
    t = topo(core("a", col, 2, ins=(64,)), device="auto")
    assert resolve_device(t).name == expected
    plan = plan_channels(t)
    assert isinstance(plan, ChannelPlan) and plan.device == expected


def test_auto_uses_widest_used_column_not_core_count():
    t = topo(core("a", 0, 2), core("b", 2, 2), device="auto")
    assert resolve_device(t).name == "npu1_3col"


def test_auto_never_picks_another_family():
    # npu2 variants have 1..7 columns too; family is chosen by name stem.
    t = topo(core("a", 0, 2), device="auto")
    assert resolve_device(t).name.startswith("npu1")


def test_auto_with_no_covering_device_is_an_error():
    t = topo(core("a", 9, 2), device="auto")
    err = resolve_device(t)
    assert isinstance(err, PlanError) and err.code == "no_device"
    assert plan_channels(t) == err


def test_explicit_device_by_name_without_model_argument():
    plan = plan_channels(topo(core("a", 0, 2, ins=(64,)), device="npu1_2col"))
    assert plan.device == "npu1_2col"


def test_unknown_device_is_an_error():
    err = plan_channels(topo(core("a", 0, 2), device="nope"))
    assert isinstance(err, PlanError) and err.code == "no_device"


# -- host layout / alignment ------------------------------------------------

def test_offsets_pack_in_core_then_buffer_order(dev):
    t = topo(core("a", 0, 2, ins=(8, 12), outs=(16,)),
             core("b", 1, 2, ins=(20,), outs=(4, 8)))
    r = routes(plan_channels(t, dev))
    assert [r[k].host_offset for k in [("a", "i0"), ("a", "i1"), ("b", "i0")]] == [0, 8, 20]
    assert [r[k].host_offset for k in [("a", "o0"), ("b", "o0"), ("b", "o1")]] == [0, 16, 20]


def test_offsets_align_to_dma_granularity(dev):
    wide = dataclasses.replace(dev, address_gen_granularity_bits=256)   # 32 bytes
    t = topo(core("a", 0, 2, ins=(36, 64)), core("b", 1, 2, ins=(4,)))
    plan = plan_channels(t, wide)
    r = routes(plan)
    assert plan.alignment_bytes == 32
    assert [r[k].host_offset for k in [("a", "i0"), ("a", "i1"), ("b", "i0")]] == [0, 64, 128]
    assert plan.phases[0].host_in_bytes == 132


def test_default_alignment_is_four_bytes(dev):
    assert plan_channels(topo(core("a", 0, 2, ins=(8,))), dev).alignment_bytes == 4


def test_join_group_is_contiguous_on_the_host(dev):
    # Core order a(col0) b(col1) c(col0): c0 and c2 are joined and so must be
    # adjacent in `out` even though b sits between them in core order.
    cs = [core("a0", 0, 2, outs=(8,)), core("b", 1, 2, outs=(8,)),
          core("a1", 0, 3, outs=(8,)), core("a2", 0, 4, outs=(8,))]
    r = routes(plan_channels(topo(*cs), dev))
    # col 0 has 3 outputs > 2 channels -> joined
    assert r[("a0", "o0")].kind == "memtile_join"
    assert [r[(n, "o0")].host_offset for n in ("a0", "a1", "a2")] == [0, 8, 16]
    assert r[("b", "o0")].host_offset == 24


def test_join_member_not_granule_multiple_is_an_error(dev):
    wide = dataclasses.replace(dev, address_gen_granularity_bits=256)
    cs = [core(f"c{k}", 0, 2 + k, outs=(36,)) for k in range(3)]
    err = plan_channels(topo(*cs), wide)
    assert isinstance(err, PlanError) and err.code == "join_member_misaligned"


# -- determinism / serialisation --------------------------------------------

def test_plan_is_deterministic_and_json_serialisable(dev):
    t = topo(*sixteen(outs=(64,)), reserve_shim_s2mm=1)
    a, b = plan_channels(t, dev), plan_channels(t, dev)
    assert a == b and hash(a) == hash(b)
    text = json.dumps(dataclasses.asdict(a), sort_keys=True)
    assert text == json.dumps(dataclasses.asdict(b), sort_keys=True)


def test_plan_does_not_depend_on_unused_columns(dev):
    t = topo(core("a", 1, 2, ins=(64,), outs=(64,)))
    assert plan_channels(t, dev).phases[0].routes == plan_channels(
        t, load_device_model("npu1_2col")).phases[0].routes


# -- partial staging (direct + staged + borrowed in one column) -------------

def test_sixteen_cores_two_in_two_out_take_two_phases(dev):
    cs = [c for col in range(4)
          for c in column_cores(col, ins=(64, 64), outs=(64, 64), prefix="c")]
    plan = plan_channels(topo(*cs), dev)
    assert isinstance(plan, ChannelPlan)
    assert len(plan.phases) == 2
    assert [c for p in plan.phases for c in p.cores] == [c.name for c in cs]
    for p in plan.phases:
        for u in p.usage:     # no tile ever exceeds its channels
            assert u.s2mm_used <= u.s2mm_available and u.mm2s_used <= u.mm2s_available
            assert u.bytes_used <= u.bytes_available


def test_stage_everything_when_one_group_holds_the_whole_column(dev):
    # 3 cores x 2 outputs in one column, one S2MM held back: one 6-way join.
    plan = plan_channels(topo(*column_cores(0, n=3, ins=(), outs=(32, 32)),
                              reserve_shim_s2mm=1), dev)
    p = plan.phases[0]
    assert {r.kind for r in p.routes} == {"memtile_join"}
    assert len(p.groups) == 1 and len(p.groups[0].members) == 6
    assert (usage(plan, 0, 1).s2mm_used, usage(plan, 0, 1).mm2s_used) == (6, 1)


def test_partial_staging_one_direct_rest_staged(dev):
    # 4 outputs, 2 shim channels, memtile fits only 2 buffers' worth: the
    # first goes direct, the next two are joined, the last borrows.
    small = tiny_memtile(dev, 130)
    plan = plan_channels(topo(*column_cores(0, ins=())), small)
    rs = plan.phases[0].routes
    assert [(r.kind, r.borrowed) for r in rs] == [
        ("direct", False), ("memtile_join", False), ("memtile_join", False),
        ("direct", True)]
    assert rs[3].shim_col == 1
    assert len(plan.phases[0].groups[0].members) == 2


def test_partial_staging_with_one_free_channel_borrows_the_remainder(dev):
    small = tiny_memtile(dev, 130)
    plan = plan_channels(topo(*column_cores(0, ins=()), reserve_shim_s2mm=1), small)
    rs = plan.phases[0].routes
    assert [(r.kind, r.borrowed) for r in rs] == [
        ("memtile_join", False), ("memtile_join", False),
        ("direct", True), ("direct", True)]


def test_staging_beats_borrowing(dev):
    plan = plan_channels(topo(*column_cores(0, ins=(), outs=(32, 32))), dev)
    # 8 outputs: 1 direct + 6-way join (memtile S2MM 6) + 1 borrowed
    rs = plan.phases[0].routes
    assert sum(1 for r in rs if r.kind == "memtile_join") == 6
    assert sum(1 for r in rs if r.borrowed) == 1


# -- first-fit phase packing -------------------------------------------------

def test_first_fit_backfills_an_earlier_phase(dev):
    small = tiny_memtile(dev)
    # Capacity 8 inputs per phase. A(5 cores) fills most of phase 0, B(4,
    # linked) cannot join it, C (single) can still backfill phase 0.
    A = [core(f"a{k}", 0, 2 + k % 4, ins=(64,)) for k in range(4)] + [core("a4", 1, 2, ins=(64,))]
    B = [core(f"b{k}", 2, 2 + k, ins=(64,)) for k in range(4)]
    C = [core("c", 3, 2, ins=(64,))]
    links = [CoreStream("b0", "b1"), CoreStream("b1", "b2"), CoreStream("b2", "b3")]
    plan = plan_channels(topo(*A, *B, *C, links=links), small)
    where = {c: p.index for p in plan.phases for c in p.cores}
    assert where["b0"] == where["b3"] == 1
    assert where["c"] == 0
    assert len(plan.phases) == 2
    # cores inside a phase stay in topology order
    assert plan.phases[0].cores == ("a0", "a1", "a2", "a3", "a4", "c")
