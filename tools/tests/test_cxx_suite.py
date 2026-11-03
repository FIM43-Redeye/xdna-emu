"""cxx_suite: the shared suite writer behind the instr-test, chess-test and shuffle-sweep generators."""
import json

import pytest

import cxx_suite
from harness import suite as S
from harness.run import prng_bytes


def kernel(name="vadd", in_size=8, out_size=4):
    return cxx_suite.CxxKernel(name, "// kernel\n", in_size, out_size, meta={"builtin": "b"})


def test_writes_one_case_per_kernel_with_compiler_and_inputs(tmp_path):
    cxx_suite.write_cxx_suite(tmp_path, "instr-peano", "gen.py", "peano",
                              [kernel("a", 8, 4), kernel("b", 64, 64)], seed=42)
    s = S.load_suite(tmp_path)
    assert s.name == "instr-peano" and s.cases == ("a", "b") and s.generator == "gen.py"
    t = s.topology("b")
    assert t.compiler == "peano" and t.device == "npu1"
    (core,) = t.cores
    assert core.source.name == "kernel.cc" and core.source.read_text() == "// kernel\n"
    assert core.symbol == "test_kernel"
    assert [(b.name, b.direction, b.size_bytes) for b in core.args] == [("in", "in", 64), ("out", "out", 64)]
    assert core.args[0].data == prng_bytes(64, 42)       # what `test_host --seed 42` fed the old tool
    assert s.case_meta("a") == {"builtin": "b", "in_size": 8, "out_size": 4}


def test_chess_suite_marks_the_package_compiler(tmp_path):
    cxx_suite.write_cxx_suite(tmp_path, "instr-chess", "gen.py", "chess", [kernel()])
    assert S.load_suite(tmp_path).topology("vadd").compiler == "chess"


def test_explicit_input_bytes_win_over_the_prng(tmp_path):
    k = cxx_suite.CxxKernel("sweep", "//\n", 128, 3072, input_bytes=bytes(range(128)))
    cxx_suite.write_cxx_suite(tmp_path, "shuffle", "gen.py", "chess", [k])
    assert S.load_suite(tmp_path).topology("sweep").cores[0].args[0].data == bytes(range(128))


def test_input_bytes_must_match_the_declared_size(tmp_path):
    k = cxx_suite.CxxKernel("bad", "//\n", 8, 4, input_bytes=b"\0" * 7)
    with pytest.raises(ValueError, match="input"):
        cxx_suite.write_cxx_suite(tmp_path, "x", "gen.py", "peano", [k])


def test_skipped_and_suite_meta_are_recorded(tmp_path):
    cxx_suite.write_cxx_suite(tmp_path, "x", "gen.py", "peano", [kernel()],
                              skipped=[{"name": "s", "reason": "r"}], meta={"k": 1})
    meta = S.load_suite(tmp_path).meta
    assert meta["skipped"] == [{"name": "s", "reason": "r"}] and meta["k"] == 1 and meta["seed"] == 42


def test_duplicate_case_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        cxx_suite.write_cxx_suite(tmp_path, "x", "gen.py", "peano", [kernel(), kernel()])


def test_sizes_round_up_to_the_word(tmp_path):
    """The harness buffers are whole i32 words; the case meta keeps the exact byte count."""
    cxx_suite.write_cxx_suite(tmp_path, "x", "gen.py", "peano", [kernel("odd", 6, 2)])
    core = S.load_suite(tmp_path).topology("odd").cores[0]
    assert [b.size_bytes for b in core.args] == [8, 4]
