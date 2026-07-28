import importlib.util
from pathlib import Path


BRIDGE_PATH = Path(__file__).with_name("mlir-aie-bridge.py")
SPEC = importlib.util.spec_from_file_location("mlir_aie_bridge", BRIDGE_PATH)
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


def test_physical_column_start_comes_from_mapped_driver_device(tmp_path):
    regs = tmp_path / "drivers" / "accel" / "amdxdna"
    regs.mkdir(parents=True)
    (regs / "npu1_regs.c").write_text(
        "const struct amdxdna_dev_info dev_npu1_info = {\n"
        "\t.first_col = 1,\n"
        "};\n"
    )
    (regs / "npu4_regs.c").write_text(
        "const struct amdxdna_dev_info dev_npu4_info = {\n"
        "\t.first_col = 0,\n"
        "};\n"
    )

    assert BRIDGE.driver_physical_column_start("npu1_2col", tmp_path) == 1
    assert BRIDGE.driver_physical_column_start("npu2_7col", tmp_path) == 0
