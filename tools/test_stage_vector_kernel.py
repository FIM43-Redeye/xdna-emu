"""Tests for stage_vector_kernel.py and the shared host harness contract."""

import re
from pathlib import Path

import pytest

import cpp_trace_patch
import stage_vector_kernel as stg
import vv_data

SOURCES = stg.data_sources()


@pytest.fixture
def mlir_aie(tmp_path):
    root = tmp_path / "mlir-aie"
    (root / "test" / "npu-xrt").mkdir(parents=True)
    return root


def knobs(run_lit_text):
    line = next(l for l in run_lit_text.splitlines() if l.startswith("// RUN: clang"))
    return {k: v for k, v in re.findall(r"-D(VV_\w+)=(\S+)", line)}


class TestSources:
    def test_every_generator_contributes(self):
        for name in ("vec_srs_i32", "vec_mac_i8", "vec_mac_bf16_forms",
                     "vec_acc_addsub_f32", "vec_vshift_wide",
                     "vec_macmode_elem_32_probe", "vec_macmode_conv_16x4_rand"):
            assert name in SOURCES

    def test_stageable_kernels_are_exactly_the_silicon_captured_ones(self):
        # Every kernel dir with a silicon golden stages from it, and nothing else
        # does: a new generator that forgets to register would leave its kernels
        # unstageable, and a stray source would stage data with no evidence.
        golden_dir = Path(stg.KERNEL_ROOT).parents[1] / "tools" / "golden" / "silicon"
        captured = {p.stem for p in golden_dir.glob("*.json")
                    if (Path(stg.KERNEL_ROOT) / p.stem).is_dir()}
        assert set(SOURCES) == captured

    def test_hand_written_kernels_are_not_stageable(self):
        for name in ("vec_eltwise_add", "vec_chain_state_probe", "vec_seed6159_clean",
                     "vec_nan_bf16_add", "vec_nan_fp32_dense_sub"):
            assert name not in SOURCES

    def test_every_stageable_kernel_has_a_committed_dir_without_data(self):
        for name in SOURCES:
            d = Path(stg.KERNEL_ROOT) / name
            assert d.is_dir(), name
            assert not (d / "test.cpp").exists(), name
            assert not list(d.glob("*.bin")), name

    def test_run_lit_knobs_match_the_data(self):
        for name, source in SOURCES.items():
            data = source()
            k = knobs((Path(stg.KERNEL_ROOT) / name / "run.lit").read_text())
            assert int(k["VV_NIN"]) == len(data.inputs), name
            width = vv_data.CTYPES[k["VV_OUT_T"]][1]
            assert int(k["VV_NOUT"]) * width == len(data.expected), name


class TestStage:
    def test_stage_assembles_kernel_host_and_data(self, mlir_aie):
        dest = Path(stg.stage("vec_mac_i8", SOURCES, str(mlir_aie)))
        assert dest == mlir_aie / "test" / "npu-xrt" / "vec_mac_i8"
        assert (dest / "run.lit").is_file() and (dest / "aie.mlir").is_file()
        assert (dest / "mac.cc").is_file()
        assert (dest / "test.cpp").read_text() == Path(stg.HOST_SOURCE).read_text()
        data = SOURCES["vec_mac_i8"]()
        assert (dest / "in0.bin").read_bytes() == data.inputs[0]
        assert (dest / "in1.bin").read_bytes() == data.inputs[1]
        assert (dest / "exp.bin").read_bytes() == data.expected

    def test_restaging_replaces_a_previous_copy(self, mlir_aie):
        dest = Path(stg.stage("vec_srs_i32", SOURCES, str(mlir_aie)))
        (dest / "stray.txt").write_text("x")
        stg.stage("vec_srs_i32", SOURCES, str(mlir_aie))
        assert not (dest / "stray.txt").exists()

    def test_uncaptured_kernel_is_staged_without_expected(self, mlir_aie):
        def uncaptured():
            return vv_data.KernelData(inputs=[b"\x00\x00\x00\x00"], expected=None)
        dest = Path(stg.stage("vec_srs_i32", {"vec_srs_i32": uncaptured}, str(mlir_aie)))
        assert (dest / "in0.bin").exists() and not (dest / "exp.bin").exists()

    def test_unknown_kernel_is_rejected(self, mlir_aie):
        with pytest.raises(SystemExit):
            stg.stage("vec_eltwise_add", SOURCES, str(mlir_aie))

    def test_stale_golden_leaves_the_old_staged_copy_alone(self, mlir_aie):
        dest = Path(stg.stage("vec_srs_i32", SOURCES, str(mlir_aie)))
        def stale():
            raise SystemExit("golden inputs are stale")
        with pytest.raises(SystemExit):
            stg.stage("vec_srs_i32", {"vec_srs_i32": stale}, str(mlir_aie))
        assert (dest / "exp.bin").exists()


class TestClean:
    def test_clean_removes_staged_copies_and_build_output(self, mlir_aie):
        stg.stage("vec_srs_i32", SOURCES, str(mlir_aie))
        build = mlir_aie / "build" / "test" / "npu-xrt" / "vec_srs_i32" / "chess"
        build.mkdir(parents=True)
        removed = stg.clean(SOURCES, str(mlir_aie))
        assert len(removed) == 2
        assert not (mlir_aie / "test" / "npu-xrt" / "vec_srs_i32").exists()
        assert not (mlir_aie / "build" / "test" / "npu-xrt" / "vec_srs_i32").exists()

    def test_clean_leaves_unrelated_tests_alone(self, mlir_aie):
        other = mlir_aie / "test" / "npu-xrt" / "vec_chain_state_probe"
        other.mkdir()
        (other / "test.cpp").write_text("x")
        stg.clean(SOURCES, str(mlir_aie))
        assert (other / "test.cpp").exists()


class TestMlirAieResolution:
    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("MLIR_AIE", "/a")
        monkeypatch.setenv("NPU_WORK_DIR", "/b")
        assert stg.mlir_aie_dir() == "/a"

    def test_work_dir_next(self, monkeypatch):
        monkeypatch.delenv("MLIR_AIE", raising=False)
        monkeypatch.setenv("NPU_WORK_DIR", "/b")
        assert stg.mlir_aie_dir() == "/b/mlir-aie"


class TestSharedHostContract:
    """The bridge patches the host source textually; the shared harness must survive."""

    HOST = Path(stg.HOST_SOURCE).read_text()

    def test_device_selection_forms_the_bdf_sed_patch_matches(self):
        assert "unsigned int device_index = 0;" in self.HOST
        assert "auto device = xrt::device(device_index);" in self.HOST

    def test_trace_patch_injects_into_the_shared_host(self):
        out = cpp_trace_patch.patch_test_cpp(self.HOST, 1048576, 5)
        assert "bo_trace" in out
        assert re.search(r"kernel\(opcode, bo_instr, instr_v\.size\(\), VV_IN_BOS, bo_out, bo_trace\)", out)
        # The trace buffer is declared after the last data BO.
        assert out.index("auto bo_trace") > out.index("auto bo_out")

    def test_knob_names_in_the_harness_are_the_ones_the_flags_emit(self):
        flags = vv_data.host_flags(1, "int8_t", 1)
        for name in re.findall(r"-D(VV_\w+)=", flags):
            assert name in self.HOST
