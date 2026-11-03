"""Tests for tools/harness/build.py that need the Peano assembler but not aiecc."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from harness.build import BuildError, build_package
from harness.topology import Buffer, Core, TileRef, Topology

KERNEL = "\t.text\n\t.globl test_kernel\n\t.type test_kernel,@function\ntest_kernel:\n\tret lr\n\tnop\n\tnop\n\tnop\n\tnop\n\tnop\n\t.size test_kernel,.-test_kernel\n"


def peano(name):
    p = Path(os.environ.get("PEANO_INSTALL_DIR", "/nonexistent")) / "bin" / name
    return str(p) if p.exists() else shutil.which(name)


needs_peano = pytest.mark.skipif(not (peano("llvm-mc") and peano("llvm-objcopy")),
                                 reason="Peano llvm-mc/llvm-objcopy unavailable")


@pytest.fixture
def kernel(tmp_path):
    p = tmp_path / "ret.s"
    p.write_text(KERNEL)
    return p


def two_cores(kernel):
    return Topology("two", (
        Core("a", TileRef(0, 2), kernel, args=(Buffer("i", "in", 64),)),
        Core("b", TileRef(0, 3), kernel, args=(Buffer("o", "out", 64),))), device="auto")


def test_validation_error_stops_before_any_work(kernel, tmp_path):
    bad = Topology("bad", (Core("a", TileRef(0, 0), kernel),), device="auto")
    with pytest.raises(BuildError, match="validation failed"):
        build_package(bad, tmp_path / "out")
    assert not (tmp_path / "out" / "phase_0").exists()


def test_plan_error_stops_before_any_work(kernel, tmp_path):
    t = Topology("none", (), device="npu1")
    with pytest.raises(BuildError):
        build_package(t, tmp_path / "out")
    assert not (tmp_path / "out" / "phase_0").exists()


def test_missing_source_is_reported(tmp_path):
    t = Topology("t", (Core("a", TileRef(0, 2), tmp_path / "nope.s"),), device="auto")
    with pytest.raises(BuildError, match="does not exist"):
        build_package(t, tmp_path / "out", run_aiecc=False)


@needs_peano
def test_objects_symbols_and_manifest(kernel, tmp_path):
    pkg = build_package(two_cores(kernel), tmp_path / "out", run_aiecc=False)
    d = pkg.phase_dirs[0]
    assert (d / "aie.mlir").is_file()
    for core in ("a", "b"):
        syms = subprocess.run([peano("llvm-objcopy").replace("objcopy", "nm"), str(d / f"{core}.o")],
                              capture_output=True, text=True).stdout
        assert f"k_{core}" in syms and "test_kernel" not in syms
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["topology"]["name"] == "two"
    assert m["plan"]["device"] == "npu1_1col" and len(m["plan"]["phases"]) == 1
    assert m["phases"] == ["phase_0"] and m["git_rev"]
    assert not m["tools"]["llvm-mc"]["path"].startswith("/home")


# -- .ll sources ---------------------------------------------------------------

LL_KERNEL = "define void @test_kernel() {\n  ret void\n}\n"


@pytest.mark.skipif(not peano("llc"), reason="Peano llc unavailable")
def test_ll_source_is_compiled_with_llc(tmp_path):
    src = tmp_path / "k.ll"
    src.write_text(LL_KERNEL)
    t = Topology("ll", (Core("a", TileRef(0, 2), src, args=(Buffer("o", "out", 64),)),), device="auto")
    pkg = build_package(t, tmp_path / "out", run_aiecc=False)
    obj = pkg.phase_dirs[0] / "a.o"
    assert obj.is_file()
    log = (pkg.phase_dirs[0] / "a.o.log").read_text()
    assert "--issue-limit=1" in log and "llc" in log


def test_unknown_source_suffix_names_the_supported_kinds(tmp_path):
    src = tmp_path / "k.cc"
    src.write_text("")
    t = Topology("cc", (Core("a", TileRef(0, 2), src),), device="auto")
    with pytest.raises(BuildError, match=r"unsupported source type '\.cc'.*\.s.*\.ll.*\.o"):
        build_package(t, tmp_path / "out", run_aiecc=False)


# -- inputs hash and reuse -----------------------------------------------------

def test_inputs_hash_follows_source_content_not_path(kernel, tmp_path):
    from harness.build import compute_inputs_hash, _workspace
    other = tmp_path / "copy.s"
    other.write_text(KERNEL)
    base = two_cores(kernel)
    moved = Topology("two", (
        Core("a", TileRef(0, 2), other, args=(Buffer("i", "in", 64),)),
        Core("b", TileRef(0, 3), other, args=(Buffer("o", "out", 64),))), device="auto")
    root = _workspace()
    assert compute_inputs_hash(base, ["m"], root) == compute_inputs_hash(moved, ["m"], root)
    assert compute_inputs_hash(base, ["m"], root) != compute_inputs_hash(base, ["m2"], root)
    other.write_text(KERNEL + "\tnop\n")
    assert compute_inputs_hash(base, ["m"], root) != compute_inputs_hash(moved, ["m"], root)


def test_inputs_hash_follows_topology(kernel):
    from harness.build import compute_inputs_hash, _workspace
    import dataclasses
    t = two_cores(kernel)
    root = _workspace()
    assert (compute_inputs_hash(t, ["m"], root)
            != compute_inputs_hash(dataclasses.replace(t, seed=9), ["m"], root))


@pytest.mark.skipif(not shutil.which("aiecc") or not (peano("llvm-mc") and peano("llvm-objcopy")),
                    reason="aiecc or Peano unavailable")
def test_reuse_skips_an_unchanged_package_and_rebuilds_a_changed_one(kernel, tmp_path):
    t = two_cores(kernel)
    out = tmp_path / "out"
    first = build_package(t, out, reuse=True)
    assert not first.reused and first.inputs_hash
    stamp = (out / "phase_0" / "aie.xclbin").stat().st_mtime_ns
    again = build_package(t, out, reuse=True)
    assert again.reused and again.inputs_hash == first.inputs_hash
    assert (out / "phase_0" / "aie.xclbin").stat().st_mtime_ns == stamp
    kernel.write_text(KERNEL + "\tnop\n")
    third = build_package(t, out, reuse=True)
    assert not third.reused and third.inputs_hash != first.inputs_hash
    assert not build_package(t, out).reused          # reuse is opt-in
