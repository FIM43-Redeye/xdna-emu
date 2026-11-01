"""Tests for the `cascade` block in the device-model export.

The cascade width comes from mlir-aie's AIETargetModel.h (the Python binding
does not expose getAccumulatorCascadeSize); the adjacency rule comes from the
AIELowerCascadeFlows pass. Both are exported as data so that harness code
reads them rather than encoding them.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BRIDGE = REPO / "tools" / "mlir-aie-bridge.py"
MODELS = REPO / "tools" / "aie-device-models.json"

_spec = importlib.util.spec_from_file_location("mlir_aie_bridge", BRIDGE)
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)

requires_mlir_aie = pytest.mark.skipif(
    importlib.util.find_spec("aie._mlir_libs._aie") is None,
    reason="mlir-aie python bindings not importable",
)


def _check_cascade_block(cascade):
    assert cascade["width_bits"] == 512
    assert cascade["core_tiles_only"] is True
    offsets = {f["direction"]: tuple(f["src_offset"]) for f in cascade["adjacency"]}
    # src = dst + offset; flow South = src directly North of dst (row + 1).
    assert offsets == {"South": (0, 1), "East": (-1, 0)}
    assert "AIELowerCascadeFlows.cpp" in cascade["adjacency_source"]
    assert "AIETargetModel.h" in cascade["width_source"]


def test_committed_json_carries_cascade_for_every_aie2_family_device():
    devices = json.loads(MODELS.read_text())["devices"]
    for name, dev in devices.items():
        if name.startswith(("npu1", "npu2")):
            _check_cascade_block(dev["cascade"])
            assert dev["default_core_stack_size"] > 0


@requires_mlir_aie
@pytest.mark.parametrize("name", ["npu1", "npu2"])
def test_exporter_emits_cascade_block(name):
    from aie._mlir_libs._aie import get_target_model

    device_id = bridge.DEVICES[name]
    dev = bridge.dump_device(get_target_model(device_id), device_id, 1)
    _check_cascade_block(dev["cascade"])
    assert dev["default_core_stack_size"] == get_target_model(
        device_id).get_default_core_stack_size()


@requires_mlir_aie
def test_exporter_fails_loudly_on_unknown_arch():
    class FakeModel:
        def get_target_arch(self):
            return 99

    with pytest.raises(ValueError, match="cascade width"):
        bridge.cascade_block(FakeModel())
