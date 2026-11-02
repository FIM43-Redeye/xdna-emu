import shutil
import subprocess

import pytest

import aiecc_artifacts as aa
import toolchain_paths


def test_finds_file_beside_the_build(tmp_path):
    (tmp_path / aa.FILENAME).write_text("x")
    assert aa.find_input_with_addresses(tmp_path) == tmp_path / aa.FILENAME


def test_finds_legacy_prj_location(tmp_path):
    prj = tmp_path / "aie_arch.mlir.prj"
    prj.mkdir()
    (prj / aa.FILENAME).write_text("x")
    assert aa.find_input_with_addresses(tmp_path) == prj / aa.FILENAME


def test_prefers_the_flag_location(tmp_path):
    (tmp_path / "aie_arch.mlir.prj").mkdir()
    (tmp_path / "aie_arch.mlir.prj" / aa.FILENAME).write_text("old")
    (tmp_path / aa.FILENAME).write_text("new")
    assert aa.find_input_with_addresses(tmp_path) == tmp_path / aa.FILENAME


def test_missing_is_none_and_require_names_the_flag(tmp_path):
    assert aa.find_input_with_addresses(tmp_path) is None
    with pytest.raises(aa.MissingInputWithAddresses) as e:
        aa.require_input_with_addresses(tmp_path)
    assert "--get-input-with-addresses" in str(e.value)


def test_installed_aiecc_writes_the_file_where_we_look(tmp_path):
    """The flag and landing spot are facts about the installed aiecc."""
    mlir_aie = toolchain_paths.try_resolve_mlir_aie()
    aiecc = shutil.which("aiecc")
    src = mlir_aie and next(
        (mlir_aie / "build/test/npu-xrt/add_one_objFifo/peano").glob("aie_arch.mlir"),
        None)
    if not (aiecc and src):
        pytest.skip("aiecc or the add_one_objFifo build is not available")
    subprocess.run(
        [aiecc, "--xchesscc=false", "--xbridge=false",
         aa.GET_INPUT_WITH_ADDRESSES, f"--output-dir={tmp_path}", str(src)],
        check=True, capture_output=True, timeout=600)
    assert aa.find_input_with_addresses(tmp_path, src.name) == tmp_path / aa.FILENAME
