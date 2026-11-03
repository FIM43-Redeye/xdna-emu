#!/usr/bin/env python3
"""Write the shuffle-network sweep as a harness suite (one Chess C++ case).

    python3 tests/shuffle-sweep/gen_suite.py <suite_dir>
    python3 -m harness suite build <suite_dir>
    python3 -m harness suite hw    <suite_dir> r1
    python3 tests/shuffle-sweep/analyze.py <suite_dir>/runs/r1/shuffle-sweep/core.out.bin

Replaces run.sh: the input is the 128-byte identity pattern (gen_input.py), the output the
48 x 64-byte routing table analyze.py reads. The kernel is Chess-only (it declares the
`shuffle` intrinsic itself), so the case is a Chess package.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "tools")]

import gen_input                                  # noqa: E402
from cxx_suite import CxxKernel, write_cxx_suite  # noqa: E402

CASE = "shuffle-sweep"
MODES = 48
OUT_BYTES = MODES * 64


def generate(suite_dir: str | Path) -> None:
    kernel = CxxKernel(CASE, (HERE / "shuffle_sweep.cc").read_text(), in_size=128,
                       out_size=OUT_BYTES, input_bytes=gen_input.identity_input(),
                       meta={"modes": MODES, "analyzer": "tests/shuffle-sweep/analyze.py"})
    write_cxx_suite(suite_dir, "shuffle-sweep", "tests/shuffle-sweep/gen_suite.py", "chess", [kernel])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    generate(sys.argv[1])
    print(f"wrote {sys.argv[1]}")
