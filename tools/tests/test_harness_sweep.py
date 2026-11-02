"""aiecc sweep: every plan shape must package cleanly with the real toolchain.

Marked `aiecc` (minutes): deselect with `-m 'not aiecc'`. Skips when aiecc or
the Peano assembler is unavailable. Kernels are a bare `ret`: the sweep checks
that the emitted MLIR is accepted and routed, not what kernels compute.
"""
import dataclasses
import time
from pathlib import Path

import pytest

from harness.build import BuildError, aiecc_path, build_package
from harness.plan import plan_channels, resolve_device
from harness.topology import Buffer, Cascade, Core, CoreStream, TileRef, Topology

pytestmark = pytest.mark.aiecc

RET_KERNEL = """\
\t.text
\t.globl test_kernel
\t.type test_kernel,@function
test_kernel:
\tret lr
\tnop
\tnop
\tnop
\tnop
\tnop
\t.size test_kernel,.-test_kernel
"""


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    if aiecc_path() is None:
        pytest.skip("aiecc not available")
    p = tmp_path_factory.mktemp("kernel") / "ret.s"
    p.write_text(RET_KERNEL)
    return p


def mk(kernel, name, col, row, ins=(), outs=()):
    args = [Buffer(f"i{k}", "in", n) for k, n in enumerate(ins)]
    args += [Buffer(f"o{k}", "out", n) for k, n in enumerate(outs)]
    return Core(name, TileRef(col, row), kernel, args=tuple(args))


def column(kernel, col, n=4, ins=(64,), outs=(64,)):
    return [mk(kernel, f"c{col}_{r}", col, 2 + r, ins, outs) for r in range(n)]


def single(k):
    return Topology("single", (mk(k, "a", 0, 2, (64,), (64,)),), device="auto")


def column_join_distribute(k):
    return Topology("col4", tuple(column(k, 0)), device="auto")


def reserve_forces_staging(k):
    return Topology("reserve", tuple(column(k, 0, n=2)), device="auto", reserve_shim_s2mm=1)


# Partial staging and borrowing never happen at the real device's limits (the
# stream-switch capacity binds first), so these two are planned against a
# memtile too small to hold everything; the MLIR still targets the real device.
SMALL_MEMTILE = 130


def partial_staging(k):
    # 3 inputs vs 2 channels, memtile fits 2 buffers: 1 direct + a 2-way distribute.
    return Topology("partial", tuple(column(k, 0, n=3, outs=())), device="auto")


def borrowed_cross_column(k):
    # 4 outputs: 1 direct, a 2-way join, the last borrows column 1's shim.
    # "auto" would pick the one-column device, which has no neighbour to borrow from.
    return Topology("borrow", tuple(column(k, 0, ins=())), device="npu1")


def one_column_overflow(k):
    # 7 outputs in one column on the one-column device: more than the 4 south
    # channels allow, so the planner splits phases (aiecc used to fail this).
    cores = column(k, 0, ins=(), outs=(32, 32))
    cores[3] = mk(k, "c0_5", 0, 5, outs=(32,))
    return Topology("overflow", tuple(cores), device="npu1_1col")


def one_column_six_way_distribute(k):
    return Topology("distribute6", tuple(column(k, 0, n=3, ins=(32, 32), outs=())),
                    device="npu1_1col")


def sixteen_cores(k):
    cores = [c for col in range(4) for c in column(k, col, ins=(64, 64), outs=(64, 64))]
    return Topology("sixteen", tuple(cores))


def cascade_south(k):
    a, b = mk(k, "p", 0, 3, ins=(64,)), mk(k, "q", 0, 2, outs=(64,))
    return Topology("cascade_south", (a, b), links=(Cascade("p", "q"),), device="auto")


def cascade_east(k):
    a, b = mk(k, "p", 0, 2, ins=(64,)), mk(k, "q", 1, 2, outs=(64,))
    return Topology("cascade_east", (a, b), links=(Cascade("p", "q"),), device="auto")


def core_stream_pair(k):
    a, b = mk(k, "p", 0, 3, outs=(64,)), mk(k, "q", 0, 2, outs=(64,))
    return Topology("stream_pair", (a, b), links=(CoreStream("p", "q"),), device="auto")


def core_stream_cross_column(k):
    a, b = mk(k, "p", 0, 3, outs=(64,)), mk(k, "q", 2, 2, outs=(64,))
    return Topology("stream_cross", (a, b), links=(CoreStream("p", "q"),), device="auto")


def cascade_producer_without_args(k):
    a, b = mk(k, "p", 0, 3), mk(k, "q", 0, 2, outs=(64,))
    return Topology("cascade_noargs", (a, b), links=(Cascade("p", "q"),), device="auto")


def mixed_cascade_and_staging(k):
    a, b = mk(k, "p", 0, 3, ins=(64,)), mk(k, "q", 0, 2, outs=(64,))
    rest = [mk(k, "r", 0, 4, (64,), (64,)), mk(k, "s", 0, 5, (64,), (64,))]
    return Topology("mixed", (a, b, *rest), links=(Cascade("p", "q"),), device="auto",
                    reserve_shim_s2mm=1)


# name -> (factory, predicate over the plan proving the shape is what it claims)
def kinds(plan):
    return {r.kind for p in plan.phases for r in p.routes}


SHAPES = {
    "single": (single, lambda p: kinds(p) == {"direct"}),
    "col4_join_distribute": (column_join_distribute,
                             lambda p: {"memtile_join", "memtile_distribute"} <= kinds(p)),
    "reserve_forces_staging": (reserve_forces_staging, lambda p: "memtile_join" in kinds(p)),
    "partial_staging": (partial_staging,
                        lambda p: {"direct", "memtile_distribute"} <= kinds(p)
                        and not any(r.borrowed for r in p.phases[0].routes), SMALL_MEMTILE),
    "borrowed_cross_column": (borrowed_cross_column,
                              lambda p: any(r.borrowed for r in p.phases[0].routes)
                              and "memtile_join" in kinds(p), SMALL_MEMTILE),
    "one_column_overflow_phases": (one_column_overflow, lambda p: len(p.phases) > 1),
    "one_column_six_way_distribute": (
        one_column_six_way_distribute,
        lambda p: len(p.phases) == 1 and len(p.phases[0].groups[0].members) == 6),
    "sixteen_cores_two_phases": (sixteen_cores, lambda p: len(p.phases) == 2),
    "cascade_south": (cascade_south, lambda p: True),
    "cascade_east": (cascade_east, lambda p: True),
    "core_stream_pair": (core_stream_pair, lambda p: True),
    "core_stream_cross_column": (core_stream_cross_column,
                                 lambda p: any(b.kind == "horizontal" for b in p.phases[0].boundaries)),
    "cascade_producer_no_args": (cascade_producer_without_args, lambda p: True),
    "mixed_cascade_and_staging": (mixed_cascade_and_staging,
                                  lambda p: "memtile_join" in kinds(p)),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_shape_packages_with_aiecc(shape, kernel, tmp_path):
    factory, shape_holds, *shrink = SHAPES[shape]
    t = factory(kernel)
    device = None
    if shrink:
        device = dataclasses.replace(resolve_device(t), mem_tile_size=shrink[0])
    plan = plan_channels(t, device)
    assert not isinstance(plan, Exception) and shape_holds(plan), f"{shape}: plan changed shape"
    start = time.time()
    pkg = build_package(t, tmp_path / shape, device=device)
    elapsed = time.time() - start
    for d in pkg.phase_dirs:
        assert (d / "aie.xclbin").stat().st_size > 0 and (d / "insts.bin").stat().st_size > 0
    print(f"SWEEP {shape}: {len(pkg.phase_dirs)} phase(s) ok in {elapsed:.1f}s")
