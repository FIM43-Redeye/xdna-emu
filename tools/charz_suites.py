#!/usr/bin/env python3
"""Suite generator for the characterizer kernels in tests/ (one suite, one case each).

    python3 tools/charz_suites.py <suite_dir>
    python3 -m harness suite build <suite_dir> && python3 -m harness suite hw <suite_dir> r1
    python3 -m harness suite emu <suite_dir> e1 && python3 -m harness suite compare <suite_dir> r1 e1
    python3 tests/<analyzer>.py --suite <suite_dir> r1 [e1]

Each characterizer is a hand-written assembly kernel that builds its own operands (or reads a
fixed input) and stores a table of results; its analyzer decodes the table. The cases:

    sparse-mask      sparse-mask-characterize.s   16 mask patterns -> active inner positions
    sparse-mac       sparse-mac-probe.s           i16xi8 sparse MAC, identity and counting
    bf16-elemwise    bf16_elemwise_characterizer.s  element permutation of the 16x2x1 bf16 multiply
    bf16-subprec     bf16_sub_precision_characterizer.s  fp32 accumulator bits below bf16
    vldb-4x          vldb_4x_characterizer.s      address decoding of the vldb.4x* gathers

Replaces the tests/run-*.sh scripts (shell plumbing around `aiecc.py`, removed upstream): the
generator writes inputs as buffer data, the runner assembles, packages, runs and compares, the
analyzers read `runs/<tag>/<case>/core.out.bin`.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness import suite as S
from harness.run import prng_bytes

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SEED = 42                       # the `test_host --seed 42` stream the retired scripts used
GENERATOR = "tools/charz_suites.py"


def load_test_module(filename: str):
    """Import a script from tests/ by file name (several have hyphens, so no plain import)."""
    path = TESTS / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Characterizer:
    case: str
    asm: str                            # file in tests/
    analyzer: str                       # file in tests/
    in_size: int
    out_size: int
    input: Callable[[], bytes]          # the exact input buffer
    what: str


def _prng(n: int) -> Callable[[], bytes]:
    return lambda: prng_bytes(n, SEED)


def _elemwise_input() -> bytes:
    return load_test_module("bf16_elemwise_analyze.py").build_input()


def _subprec_input() -> bytes:
    return load_test_module("bf16_sub_precision_gen.py").build_input()


CHARACTERIZERS: tuple[Characterizer, ...] = (
    # The kernel builds A and B itself and uses p0 as scratch: 32 bytes is q0's mask scratch.
    Characterizer("sparse-mask", "sparse-mask-characterize.s", "sparse-mask-characterize.py",
                  32, 1024, _prng(32), "16 sparse mask patterns x 64 bytes (32 x int16)"),
    Characterizer("sparse-mac", "sparse-mac-probe.s", "sparse-mac-analyze.py",
                  128, 128, _prng(128), "2 tests x 64 bytes (32 x int16): identity, counting"),
    Characterizer("bf16-elemwise", "bf16_elemwise_characterizer.s", "bf16_elemwise_analyze.py",
                  256, 192, _elemwise_input, "3 tests x (16 hi + 16 lo) x int16 SRS halves"),
    Characterizer("bf16-subprec", "bf16_sub_precision_characterizer.s", "bf16_sub_precision_analyze.py",
                  384, 512, _subprec_input, "4 tests x 128 bytes (32 hi + 32 lo used)"),
    Characterizer("vldb-4x", "vldb_4x_characterizer.s", "vldb_4x_analyze.py",
                  256, 256, _prng(256), "8 x 32 bytes: six gathers, the address vector, a repeat"),
)


def generate(suite_dir: str | Path) -> None:
    for c in CHARACTERIZERS:
        data = c.input()
        if len(data) != c.in_size:
            raise ValueError(f"{c.case}: input is {len(data)} bytes, expected {c.in_size}")
        S.write_single_core_case(
            suite_dir, c.case, (TESTS / c.asm).read_text(), input_bytes=data, out_bytes=c.out_size,
            meta={"analyzer": f"tests/{c.analyzer}", "kernel": f"tests/{c.asm}", "what": c.what,
                  "in_size": c.in_size, "out_size": c.out_size})
    S.write_suite_json(suite_dir, "charz", GENERATOR, [c.case for c in CHARACTERIZERS],
                       meta={"seed": SEED}, hw_checks=["aligned_labels"])


def read_outputs(argv: list[str], case: str) -> list[bytes]:
    """The result files an analyzer was asked about.

    `--suite DIR TAG [TAG2 ...]` reads the case's output of each tag from the runner's layout.
    """
    if len(argv) < 3 or argv[0] != "--suite":
        raise SystemExit("usage: --suite <suite_dir> <tag> [<tag> ...]")
    suite = S.load_suite(argv[1])
    return [suite.output(t, case).read_bytes() for t in argv[2:]]


def suite_input(argv: list[str], case: str) -> bytes:
    """The case's input buffer, for `argv` in `--suite` form."""
    return S.load_suite(argv[1]).topology(case).cores[0].args[0].data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    generate(sys.argv[1])
    print(f"wrote {sys.argv[1]} ({len(CHARACTERIZERS)} cases)")
