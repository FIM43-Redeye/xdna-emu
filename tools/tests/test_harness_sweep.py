"""aiecc sweep: every plan shape must package cleanly with the real toolchain.

Marked `aiecc` (minutes): deselect with `-m 'not aiecc'`. Skips when aiecc or
the Peano assembler is unavailable. Kernels are a bare `ret`: the sweep checks
that the emitted MLIR is accepted and routed, not what kernels compute.
"""
import shutil
import time
from pathlib import Path

import pytest

from harness.build import BuildError, aiecc_path, build_package
from harness.plan import plan_channels
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


def partial_staging(k, device="npu1"):
    # 7 outputs vs 2 S2MM channels and 6 memtile S2MM: 1 direct + a 6-way join
    # (the planner stages everything it can, so partial staging needs overflow).
    cores = column(k, 0, ins=(), outs=(32, 32))
    cores[3] = mk(k, "c0_5", 0, 5, outs=(32,))
    return Topology("partial", tuple(cores), device=device)


def borrowed_cross_column(k):
    # 8 outputs in one column: 1 direct + 6-way join + 1 borrowed from column 1.
    # "auto" would pick the one-column device, which has no neighbour to borrow from.
    return Topology("borrow", tuple(column(k, 0, ins=(), outs=(32, 32))), device="npu1")


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
                        lambda p: {"direct", "memtile_join"} <= kinds(p)
                        and not any(r.borrowed for r in p.phases[0].routes)),
    "borrowed_cross_column": (borrowed_cross_column,
                              lambda p: any(r.borrowed for r in p.phases[0].routes)),
    "sixteen_cores_two_phases": (sixteen_cores, lambda p: len(p.phases) == 2),
    "cascade_south": (cascade_south, lambda p: True),
    "cascade_east": (cascade_east, lambda p: True),
    "core_stream_pair": (core_stream_pair, lambda p: True),
    "cascade_producer_no_args": (cascade_producer_without_args, lambda p: True),
    "mixed_cascade_and_staging": (mixed_cascade_and_staging,
                                  lambda p: "memtile_join" in kinds(p)),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_shape_packages_with_aiecc(shape, kernel, tmp_path):
    factory, shape_holds = SHAPES[shape]
    t = factory(kernel)
    plan = plan_channels(t)
    assert not isinstance(plan, Exception) and shape_holds(plan), f"{shape}: plan changed shape"
    start = time.time()
    pkg = build_package(t, tmp_path / shape)
    elapsed = time.time() - start
    for d in pkg.phase_dirs:
        assert (d / "aie.xclbin").stat().st_size > 0 and (d / "insts.bin").stat().st_size > 0
    print(f"SWEEP {shape}: {len(pkg.phase_dirs)} phase(s) ok in {elapsed:.1f}s")


@pytest.mark.xfail(strict=True, raises=BuildError, reason=(
    "known planner/router gap: on the one-column device the 7 flows of "
    "partial_staging cross row 2 -> row 1 but the column has 4 channels "
    "(aiecc: 'Unable to find a legal routing'); the full npu1 routes them "
    "through neighbouring columns. Coordinator decision pending."))
def test_partial_staging_on_one_column_device(kernel, tmp_path):
    build_package(partial_staging(kernel, device="npu1_1col"), tmp_path / "p1col")
