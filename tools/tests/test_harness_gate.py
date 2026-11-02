"""Gate topology construction (no hardware): the shapes the gates promise."""
import pytest

from harness import gate
from harness.plan import plan_channels

needs_recorded = pytest.mark.skipif(not (gate.RECORDED / "probe" / "manifest.json").is_file(),
                                    reason="recorded collision probes not present")


@needs_recorded
def test_gate_a_is_one_core_per_recorded_kernel():
    ks = gate.kernels()
    assert len(ks) == 31
    t = gate.single_topology(ks[0])
    assert len(t.cores) == 1 and t.cores[0].args[0].data == ks[0].input
    assert t.cores[0].args[1].size_bytes == ks[0].out_size


@needs_recorded
def test_gate_b_shapes_plan_without_error_and_cover_the_promised_cases():
    shapes = {t.name: (t, m) for t, m in gate.gate_b_topologies()}
    assert len(shapes["gateB_full16"][0].cores) == 16
    assert shapes["gateB_cols4_direct"][0].cores[1].tile.col == 1
    assert shapes["gateB_cols4x2_reserve1"][0].reserve_shim_s2mm == 1
    for t, mapping in shapes.values():
        plan = plan_channels(t)
        assert not hasattr(plan, "code"), plan
        assert set(mapping) == {c.name for c in t.cores}
    kinds = {r.kind for r in plan_channels(shapes["gateB_col0_stack4"][0]).phases[0].routes}
    assert kinds == {"memtile_join", "memtile_distribute"}
