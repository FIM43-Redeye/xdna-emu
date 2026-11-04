#!/usr/bin/env python3
"""Analyze the sparse MAC probe (sparse-mac-probe.s).

    python3 tests/sparse-mac-analyze.py <output.bin> [<output2.bin>]
    python3 tests/sparse-mac-analyze.py --suite <suite_dir> <tag> [<tag2>]

Output: 2 tests x 64 bytes, each 32 x int16 (SRS shift 0 of the accumulator, i16xi8 sparse
rows=2 inner=16 cols=8):
    test 1  identity: A = i16 1, B = i8 1, mask 0x33..; the kernel header expects 8 in every lane
    test 2  counting: A = 1, B compressed = 1..8; column 0 expects 36
With two results the second is compared lane by lane against the first.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from charz_suites import read_outputs  # noqa: E402

CASE = "sparse-mac"
TESTS = ("Identity (A=1,B=1)", "Counting (B=1..8)")
EXPECT_IDENTITY_LANE = 8        # 4 inner groups x 2 active positions
EXPECT_COUNTING_COL0 = 36       # 1 + 2 + ... + 8


def decode(data: bytes) -> list[tuple[int, ...]]:
    """Per test, its 32 int16 lanes."""
    if len(data) < 128:
        raise ValueError(f"expected >= 128 bytes, got {len(data)}")
    return [struct.unpack_from("<32h", data, t * 64) for t in range(2)]


def header_expectations(tests: list[tuple[int, ...]]) -> dict[str, bool]:
    return {"identity: every lane == 8": all(v == EXPECT_IDENTITY_LANE for v in tests[0]),
            "counting: lane 0 == 36": tests[1][0] == EXPECT_COUNTING_COL0}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    outs = read_outputs(argv, CASE)
    tests = [decode(o) for o in outs]
    for n, vals in enumerate(tests):
        print(f"output {n + 1} ({len(outs[n])} bytes):")
        for t, lanes in enumerate(vals):
            print(f"  Test {t + 1} ({TESTS[t]}):")
            print(f"    Even lanes: {[lanes[i] for i in range(0, 32, 2)]}")
            print(f"    Odd lanes:  {[lanes[i] for i in range(1, 32, 2)]}")
        for what, ok in header_expectations(vals).items():
            print(f"  {what}: {ok}")
    if len(tests) > 1:
        bad = 0
        for t in range(2):
            diffs = [(i, tests[1][t][i], tests[0][t][i]) for i in range(32) if tests[1][t][i] != tests[0][t][i]]
            bad += len(diffs)
            print(f"Test {t + 1} ({TESTS[t]}): " + (f"{len(diffs)} lanes differ" if diffs else "MATCH"))
            for lane, got, ref in diffs[:8]:
                print(f"  lane {lane:2d}: second={got:7d}  first={ref:7d}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
