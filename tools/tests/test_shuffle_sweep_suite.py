"""The shuffle-sweep generator writes a one-case Chess suite with the identity input."""
import importlib.util
import json
from pathlib import Path

from harness import suite as S

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("shuffle_gen_suite", ROOT / "tests/shuffle-sweep/gen_suite.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_generate_writes_the_chess_case(tmp_path):
    gen.generate(tmp_path)
    s = S.load_suite(tmp_path)
    assert s.cases == ("shuffle-sweep",)
    t = s.topology("shuffle-sweep")
    assert t.compiler == "chess"
    core = t.cores[0]
    assert core.source.read_text() == (ROOT / "tests/shuffle-sweep/shuffle_sweep.cc").read_text()
    assert core.args[0].data == bytes(range(128)) and core.args[1].size_bytes == 3072


def test_banked_capture_is_the_sweep_output_size():
    banked = ROOT / "tests/shuffle-sweep/silicon-capture-npu1.bin"
    assert banked.stat().st_size == gen.OUT_BYTES
