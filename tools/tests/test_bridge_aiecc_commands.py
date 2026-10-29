"""Tests for emu-bridge-test.sh's aiecc command handling.

Upstream mlir-aie RUN lines invoke the C++ driver as `%aiecc` (the Python
`aiecc.py` passthrough was removed in July 2026), and our own generators emit a
bare `aiecc`. apply_lit_subs normalizes every spelling to the one $AIECC_BIN
path, so the per-compiler transforms can recognize compile commands exactly.
A miss there is silent and serious: the "Chess" build of a bridge test would
quietly compile with Peano.

The functions are extracted from the script by name (definition at column 0
through the closing `}` at column 0) and run in isolation.
"""

import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "emu-bridge-test.sh"
AIECC_BIN = "/opt/mlir-aie/build/bin/aiecc"
SRC_DIR = "/src/test_case"


def _function_source(name: str) -> str:
    text = SCRIPT.read_text()
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.M | re.S)
    assert match, f"{name} not found in {SCRIPT}"
    return match.group(0)


def _call(name: str, *args: str, env: dict | None = None) -> str:
    helpers = ["_cmd_has_flag", "_cmd_drop_flag", "_is_aiecc_cmd", "_aiecc_add_flag"]
    prelude = (
        # apply_lit_subs looks up the lit file; none is needed for these cases.
        "find_lit_file() { :; }\n"
        + "".join(_function_source(h) for h in helpers)
        + _function_source(name)
    )
    result = subprocess.run(
        ["bash", "-c", prelude + f'\n{name} "$@"', "bash", *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "AIECC_BIN": AIECC_BIN, **(env or {})},
    )
    return result.stdout.strip()


def _args(cmd: str) -> list[str]:
    return cmd.split()


# -- apply_lit_subs: every spelling becomes $AIECC_BIN ----------------------


def test_lit_aiecc_substitution_uses_aiecc_bin():
    cmd = _call("apply_lit_subs", SRC_DIR,
                "// RUN: %aiecc --get-xclbin --xclbin-name=aie.xclbin ./aie.mlir")
    assert _args(cmd)[0] == AIECC_BIN
    assert "--get-xclbin" in _args(cmd)


def test_bare_aiecc_from_our_generators_uses_aiecc_bin():
    cmd = _call("apply_lit_subs", SRC_DIR,
                "// RUN: aiecc --xchesscc=false --get-xclbin %S/aie.mlir")
    assert _args(cmd)[0] == AIECC_BIN
    assert f"{SRC_DIR}/aie.mlir" in _args(cmd)


def test_non_aiecc_commands_are_untouched_by_aiecc_normalization():
    cmd = _call("apply_lit_subs", SRC_DIR, "// RUN: aie-opt --help")
    assert cmd == "aie-opt --help"


# -- transform_for_chess ----------------------------------------------------


def test_chess_transform_selects_chess_and_drops_peano_opt_outs():
    cmd = _call("transform_for_chess",
                f"{AIECC_BIN} --xchesscc=false --xbridge=false --get-xclbin aie.mlir")
    args = _args(cmd)
    assert args[0] == AIECC_BIN
    assert {"--xchesscc", "--xbridge", "--unified"} <= set(args)
    assert "--xchesscc=false" not in args and "--xbridge=false" not in args


def test_chess_transform_requests_aiesim_with_current_flag():
    cmd = _call("transform_for_chess", f"{AIECC_BIN} --get-xclbin aie.mlir",
                env={"RUN_AIESIM": "true"})
    assert "--get-aiesim" in _args(cmd)
    assert "--aiesim" not in _args(cmd)


def test_chess_transform_leaves_other_commands_alone():
    cmd = "/usr/bin/clang++ test.cpp -o test.exe"
    assert _call("transform_for_chess", cmd) == cmd


# -- transform_for_peano ----------------------------------------------------


# The C++ driver has no --no-xchesscc / --no-xbridge; LLVM cl::opt bools
# negate as --flag=false. Peano is the default, but the opt-out is explicit so
# a Chess-when-available default can't sneak in.


def test_peano_transform_opts_out_of_chess():
    cmd = _call("transform_for_peano",
                f"{AIECC_BIN} --xchesscc --xbridge --get-xclbin aie.mlir")
    args = _args(cmd)
    assert args[0] == AIECC_BIN
    assert {"--xchesscc=false", "--xbridge=false"} <= set(args)
    assert "--xchesscc" not in args and "--xbridge" not in args


def test_peano_transform_keeps_existing_opt_outs_intact():
    cmd = _call("transform_for_peano",
                f"{AIECC_BIN} --xchesscc=false --xbridge=false --get-xclbin aie.mlir")
    args = _args(cmd)
    assert args.count("--xchesscc=false") == 1 and args.count("--xbridge=false") == 1


# -- requires_only_compiler -------------------------------------------------

# Since mlir-aie #3451/#3501 (Jul 31 / Aug 5 2026) every backend-neutral test
# says `REQUIRES: ..., peano`: upstream pins its default backend there. That
# is not a Peano-only marker -- these are the kernels Chess compiled until
# July. A test is Peano-only when a RUN step compiles with Peano directly, so
# its objects are Peano's and Chess cannot link them.

UPSTREAM_DEFAULT_LIT = """\
// REQUIRES: ryzen_ai, peano
// RUN: %aiecc --get-npu-insts --get-xclbin --xclbin-name=aie.xclbin ./aie_arch.mlir
"""

DIRECT_PEANO_LIT = """\
// REQUIRES: ryzen_ai_npu1, peano
// RUN: %PEANO_INSTALL_DIR/bin/clang --target=aie2-none-unknown-elf -c %S/kernel.cc -o kernel.o
// RUN: %aiecc --get-npu-insts --get-xclbin --xclbin-name=aie.xclbin ./aie_arch.mlir
"""

CHESS_LIT = """\
// REQUIRES: ryzen_ai_npu1, chess
// RUN: %aiecc --xchesscc --xbridge --get-xclbin ./aie_arch.mlir
"""


def _requires_only(tmp_path: Path, lit: str, compiler: str) -> bool:
    (tmp_path / "run.lit").write_text(lit)
    prelude = _function_source("find_lit_file") + _function_source("requires_only_compiler")
    result = subprocess.run(
        ["bash", "-c", prelude + '\nrequires_only_compiler "$@"', "bash",
         str(tmp_path), compiler],
        env={"PATH": "/usr/bin:/bin"},
    )
    return result.returncode == 0


def test_upstream_default_peano_marker_is_not_peano_only(tmp_path):
    assert not _requires_only(tmp_path, UPSTREAM_DEFAULT_LIT, "chess")
    assert not _requires_only(tmp_path, UPSTREAM_DEFAULT_LIT, "peano")


def test_direct_peano_compile_makes_a_test_peano_only(tmp_path):
    assert _requires_only(tmp_path, DIRECT_PEANO_LIT, "chess")
    assert not _requires_only(tmp_path, DIRECT_PEANO_LIT, "peano")


def test_direct_peano_compile_in_python_run_lines_is_peano_only(tmp_path):
    # Python-generated tests (matrix_transpose) carry `# RUN:` lines in aie2.py.
    (tmp_path / "aie2.py").write_text(
        "# REQUIRES: ryzen_ai_npu1, peano\n"
        "# RUN: %PEANO_INSTALL_DIR/bin/clang++ --target=aie2-none-unknown-elf -O2 \\\n"
        "# RUN:   -c %S/kernel.cc -o ./kernel.o\n"
        "# RUN: %aiecc --get-xclbin ./aie2.mlir\n")
    prelude = _function_source("find_lit_file") + _function_source("requires_only_compiler")
    result = subprocess.run(
        ["bash", "-c", prelude + '\nrequires_only_compiler "$@"', "bash", str(tmp_path), "chess"],
        env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 0


def test_chess_requirement_stays_chess_only(tmp_path):
    assert _requires_only(tmp_path, CHESS_LIT, "peano")
    assert not _requires_only(tmp_path, CHESS_LIT, "chess")


# -- lit-native tests: continuations, %t, runtime-arg variants --------------

# Shape of upstream's newer tests (dynamic_chain_passthrough, 2026): RUN lines
# continue with a trailing backslash, every artifact lives under lit's %t, the
# host includes an aiecc-generated header, and variants are host arguments.
LIT_NATIVE = r"""// REQUIRES: ryzen_ai, peano
// RUN: mkdir -p %t.prj
// RUN: cp %S/aie.mlir aie_arch.mlir
// RUN: %run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir
// RUN: %aiecc --get-xclbin --get-npu-cpp --npu-cpp-name=%t.gen.h \
// RUN:   --output-dir=%t.prj \
// RUN:   --xclbin-name=%t.xclbin --tmpdir=%t.prj aie_arch.mlir
// RUN: %host_clang %S/test.cpp -o %t.exe -std=c++17 -Wall \
// RUN:   -DGEN_HDR='"%t.gen.h"' -DXCLBIN='std::string("%/t.xclbin")' \
// RUN:   %test_utils_flags
// RUN: %run_on_npu1% %t.exe 1
// RUN: %run_on_npu2% %t.exe 1
// RUN: %run_on_npu1% %t.exe 2
// RUN: %run_on_npu2% %t.exe 2
// RUN: %run_on_npu1% %t.exe 8
// RUN: %run_on_npu2% %t.exe 8
"""

LIT_FUNCS = [
    "find_lit_file", "lit_run_lines", "apply_lit_subs", "requires_npu2",
    "extract_build_commands", "_variant_from_cmd", "_strip_trace_flags",
    "get_run_variants", "get_variant_run_cmd", "uses_lit_tmp", "_is_npu1_host_run",
    "_is_host_run_cmd", "_host_runs_only_on_npu2",
]


def _lit_call(tmp_path: Path, lit: str, func: str, *args: str) -> list[str]:
    (tmp_path / "run.lit").write_text(lit)
    prelude = "".join(_function_source(f) for f in LIT_FUNCS)
    result = subprocess.run(
        ["bash", "-c", prelude + f'\n{func} "$@"', "bash", *args],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "AIECC_BIN": AIECC_BIN,
             "TEST_UTILS_INCLUDE": "/tu/inc", "TEST_UTILS_LIB": "/tu/lib"},
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_continued_run_lines_join_into_one_command(tmp_path):
    lines = _lit_call(tmp_path, LIT_NATIVE, "lit_run_lines", str(tmp_path / "run.lit"))
    aiecc = [line for line in lines if "%aiecc" in line]
    assert len(aiecc) == 1
    assert "--npu-cpp-name=%t.gen.h --output-dir=%t.prj --xclbin-name=%t.xclbin" in " ".join(aiecc[0].split())
    assert not any(line.rstrip().endswith("\\") for line in lines)


def test_lit_tmp_maps_to_the_harness_test_prefix(tmp_path):
    (cmd,) = _lit_call(tmp_path, LIT_NATIVE, "apply_lit_subs", str(tmp_path),
                       "// RUN: %host_clang x.cpp -o %t.exe -DX='\"%/t.xclbin\"' %test_utils_flags")
    args = cmd.split()
    assert "./test.exe" in args
    assert "-DX='\"./test.xclbin\"'" in args
    # %t must not eat the prefix of %test_utils_flags.
    assert "-I/tu/inc" in args and "-ltest_utils" in args


def test_build_time_lit_tmp_is_the_absolute_build_prefix(tmp_path):
    # aiecc resolves a relative --npu-cpp-name against --output-dir, so a build
    # needs lit's absolute %t; LIT_TMP carries the build dir's prefix.
    (tmp_path / "run.lit").write_text(LIT_NATIVE)
    prelude = "".join(_function_source(f) for f in LIT_FUNCS)
    result = subprocess.run(
        ["bash", "-c", prelude + '\napply_lit_subs "$@"', "bash", str(tmp_path),
         "// RUN: %aiecc --npu-cpp-name=%t.gen.h --output-dir=%t.prj x.mlir"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "AIECC_BIN": AIECC_BIN, "LIT_TMP": "/b/peano/test"})
    assert "--npu-cpp-name=/b/peano/test.gen.h" in result.stdout.split()
    assert "--output-dir=/b/peano/test.prj" in result.stdout.split()


def test_lit_native_build_keeps_the_host_build_and_drops_runs(tmp_path):
    cmds = _lit_call(tmp_path, LIT_NATIVE, "extract_build_commands",
                     str(tmp_path / "run.lit"), str(tmp_path))
    assert any(c.startswith(AIECC_BIN) and "--xclbin-name=./test.xclbin" in c for c in cmds)
    assert any("clang++" in c and "-o ./test.exe" in c for c in cmds)
    assert not any(c.split()[0] == "./test.exe" for c in cmds)


def test_host_argument_runs_are_separate_variants(tmp_path):
    variants = _lit_call(tmp_path, LIT_NATIVE, "get_run_variants", str(tmp_path))
    assert variants == ["1", "2", "8"]
    (cmd,) = _lit_call(tmp_path, LIT_NATIVE, "get_variant_run_cmd", str(tmp_path), "8")
    assert cmd == "./test.exe 8"


def test_lit_tmp_marks_a_test_lit_native(tmp_path):
    (tmp_path / "run.lit").write_text(LIT_NATIVE)
    prelude = _function_source("find_lit_file") + _function_source("uses_lit_tmp")
    native = subprocess.run(["bash", "-c", prelude + '\nuses_lit_tmp "$1"', "bash", str(tmp_path)])
    assert native.returncode == 0
    (tmp_path / "run.lit").write_text(UPSTREAM_DEFAULT_LIT)
    classic = subprocess.run(["bash", "-c", prelude + '\nuses_lit_tmp "$1"', "bash", str(tmp_path)])
    assert classic.returncode != 0


# -- npu2-only by host runs -------------------------------------------------

# Shape of the pad_value tests (#3512): REQUIRES names no generation, but the
# device is npu2 and the host only runs under %run_on_npu2%. On NPU1 lit runs
# nothing, so the bridge must skip the test rather than run it as npu1.
NPU2_HOST_ONLY_LIT = """// REQUIRES: ryzen_ai
// RUN: cp %S/aie.mlir aie_arch.mlir
// RUN: sed 's/NPUDEVICE/npu2_1col/g' -i aie_arch.mlir
// RUN: %aiecc --get-npu-insts --get-xclbin --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch.mlir
// RUN: %host_clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags %host_link_flags %test_utils_flags
// RUN: %run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
"""


def _requires_npu2(tmp_path: Path, lit: str) -> bool:
    (tmp_path / "run.lit").write_text(lit)
    prelude = "".join(_function_source(f) for f in LIT_FUNCS)
    result = subprocess.run(
        ["bash", "-c", prelude + '\nrequires_npu2 "$1"', "bash", str(tmp_path)],
        env={"PATH": "/usr/bin:/bin", "AIECC_BIN": AIECC_BIN})
    return result.returncode == 0


def test_host_runs_only_under_run_on_npu2_make_a_test_npu2_only(tmp_path):
    assert _requires_npu2(tmp_path, NPU2_HOST_ONLY_LIT)


def test_host_runs_on_both_generations_are_not_npu2_only(tmp_path):
    assert not _requires_npu2(tmp_path, LIT_NATIVE)


def test_unprefixed_host_run_is_not_npu2_only(tmp_path):
    assert not _requires_npu2(tmp_path, UPSTREAM_DEFAULT_LIT
                              + "// RUN: ./test.exe -x aie.xclbin -k MLIR_AIE\n")


def test_compile_only_test_is_not_npu2_only(tmp_path):
    assert not _requires_npu2(tmp_path, UPSTREAM_DEFAULT_LIT)
