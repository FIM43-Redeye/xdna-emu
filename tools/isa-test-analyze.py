#!/usr/bin/env python3
"""Analyze ISA test results at the test-point level.

Reads an ISA suite (harness/suite.py layout, written by isa-test-gen.py) and two run
tags, one on hardware and one on the emulator, and compares each test point's slices of
the cases' output buffers. Produces a per-instruction, per-test-point pass/fail report,
grouped by instruction name and category.

Usage:
    python3 tools/isa-test-analyze.py SUITE_DIR [HW_TAG EMU_TAG]   # default: the `last` run
    python3 tools/isa-test-analyze.py SUITE_DIR --category vector-mac
    python3 tools/isa-test-analyze.py SUITE_DIR --failing           # only show failures
    python3 tools/isa-test-analyze.py SUITE_DIR --summary           # one-line-per-instruction
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from harness import suite as S


# Instruction name -> category mapping (prefix-based).
CATEGORY_PREFIXES = [
    # Vector MAC/matmul
    ("VMAC", "vector-mac"),
    ("VMSC", "vector-mac"),
    ("VMATMUL", "vector-mac"),
    # Vector SRS/UPS/convert
    ("VSRS", "vector-srs"),
    ("VSRSM", "vector-srs"),
    ("VUPS", "vector-ups"),
    ("VCONV", "vector-convert"),
    # Vector extract/insert/shuffle
    ("VEXTRACT", "vector-extract"),
    ("VEXTBCST", "vector-extract"),
    ("VINSERT", "vector-insert"),
    ("VPUSH", "vector-push"),
    ("VPOP", "vector-pop"),
    ("VSHUFFLE", "vector-shuffle"),
    ("VSHIFT", "vector-shift"),
    ("VUNPACK", "vector-unpack"),
    ("VPACK", "vector-pack"),
    # Vector arithmetic
    ("VADD", "vector-arith"),
    ("VSUB", "vector-arith"),
    ("VMUL", "vector-arith"),
    ("VMIN", "vector-arith"),
    ("VMAX", "vector-arith"),
    ("VNEG", "vector-arith"),
    ("VABS", "vector-arith"),
    ("VCMP", "vector-compare"),
    ("VSEL", "vector-select"),
    ("VBCST", "vector-broadcast"),
    ("VBAND", "vector-bitwise"),
    ("VBOR", "vector-bitwise"),
    ("VBNEG", "vector-bitwise"),
    # Vector load/store
    ("VLDA", "vector-load"),
    ("VLDB", "vector-load"),
    ("VST", "vector-store"),
    # Scalar load/store
    ("LDA", "scalar-load"),
    ("LDB", "scalar-load"),
    ("ST_", "scalar-store"),
    ("ST ", "scalar-store"),
    # Scalar ALU
    ("ADD", "scalar-alu"),
    ("SUB", "scalar-alu"),
    ("MUL", "scalar-alu"),
    ("ABS", "scalar-alu"),
    ("NEG", "scalar-alu"),
    ("AND", "scalar-alu"),
    ("OR_", "scalar-alu"),
    ("XOR", "scalar-alu"),
    ("NOT", "scalar-alu"),
    ("EQ", "scalar-compare"),
    ("NE", "scalar-compare"),
    ("GE", "scalar-compare"),
    ("GT", "scalar-compare"),
    ("LT", "scalar-compare"),
    ("DIVS", "scalar-alu"),
    ("MOD", "scalar-alu"),
    ("ADC", "scalar-alu"),
    ("CLB", "scalar-bits"),
    ("CLZ", "scalar-bits"),
    ("EXTEND", "scalar-extend"),
    ("ASHL", "scalar-shift"),
    ("LSHR", "scalar-shift"),
    ("ASHR", "scalar-shift"),
    ("MOV", "scalar-move"),
    # Pointer/address
    ("PADDA", "pointer"),
    ("PADDB", "pointer"),
    ("PADDS", "pointer"),
    # Branch/control
    ("J", "branch"),
    ("CALL", "branch"),
    ("RET", "branch"),
]


def categorize(name: str) -> str:
    """Map instruction name to category."""
    upper = name.upper()
    for prefix, cat in CATEGORY_PREFIXES:
        if upper.startswith(prefix):
            return cat
    return "other"


def compare_test_point(hw_slices: list[bytes], emu_slices: list[bytes]) -> Optional[bool]:
    """Compare one test point's slices (the same slice list for both backends).

    Returns True if every slice matches, False if any differs, None if a slice
    falls outside the available data (can't compare).
    """
    for hw, emu in zip(hw_slices, emu_slices):
        if hw is None or emu is None:
            return None
    return hw_slices == emu_slices


def _slice_of(data: bytes, offset: int, size: int) -> Optional[bytes]:
    return data[offset:offset + size] if offset + size <= len(data) else None


def _read(suite: S.Suite, tag: str, case: str, test: dict) -> Optional[list[Optional[bytes]]]:
    """The bytes of each of the test point's slices, or None if a file is missing."""
    out = []
    for sl in test["slices"]:
        p = S.output_path(suite.path, tag, case, sl["core"], sl["buffer"])
        if not p.is_file():
            return None
        out.append(_slice_of(p.read_bytes(), sl["offset"], sl["size"]))
    return out


def last_tags(suite_dir: Path) -> tuple[str, str]:
    """The (hw, emu) tags of the last run, from the `last` pointer scripts/isa-test.sh writes."""
    p = Path(suite_dir) / "last"
    if not p.is_file():
        raise S.SuiteError(f"{p} not found: pass HW_TAG and EMU_TAG, or run scripts/isa-test.sh")
    run = p.read_text().strip()
    return f"hw-{run}", f"emu-{run}"


@dataclass(frozen=True)
class Totals:
    passed: int
    failed: int
    skipped_cases: int


def analyze(suite: S.Suite, hw_tag: str, emu_tag: str, category_filter: str = None,
            failing_only: bool = False, summary_mode: bool = False) -> Totals:
    """Run the full analysis."""
    # instr_name -> {"pass": int, "fail": int, "total": int, "details": [...]}
    instr_stats = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0, "details": []})
    category_stats = defaultdict(lambda: {"pass": 0, "fail": 0})
    skipped_cases = 0
    total_points = 0

    for case in suite.cases:
        tests = suite.case_meta(case).get("tests", [])
        if not tests:
            continue
        batch_idx = suite.case_meta(case)["batch_index"]
        usable = True
        for test in tests:
            hw = _read(suite, hw_tag, case, test)
            emu = _read(suite, emu_tag, case, test)
            if hw is None or emu is None:
                usable = False
                break
        if not usable:
            skipped_cases += 1
            continue

        for test in tests:
            name = test["instruction"]
            cat = categorize(name)

            if category_filter and cat != category_filter:
                continue

            passed = compare_test_point(_read(suite, hw_tag, case, test),
                                        _read(suite, emu_tag, case, test))
            if passed is None:
                # Test point falls outside available data -- skip it.
                continue

            total_points += 1

            if passed:
                instr_stats[name]["pass"] += 1
                category_stats[cat]["pass"] += 1
            else:
                instr_stats[name]["fail"] += 1
                category_stats[cat]["fail"] += 1
                if not summary_mode:
                    instr_stats[name]["details"].append({
                        "batch": batch_idx,
                        "combo": test["combo_index"],
                        "operands": test.get("operands", {}),
                    })
            instr_stats[name]["total"] += 1

    # Sort instructions by category then name.
    sorted_instrs = sorted(instr_stats.items(),
                           key=lambda x: (categorize(x[0]), x[0]))

    # Print results.
    total_pass = sum(s["pass"] for s in instr_stats.values())
    total_fail = sum(s["fail"] for s in instr_stats.values())

    print(f"=== ISA Test-Point Analysis ===")
    print(f"Total test points: {total_points}")
    print(f"  PASS: {total_pass} ({100*total_pass/max(total_points,1):.1f}%)")
    print(f"  FAIL: {total_fail} ({100*total_fail/max(total_points,1):.1f}%)")
    if skipped_cases:
        print(f"  Skipped cases (missing data): {skipped_cases}")
    print()

    # Category summary.
    print("=== By Category ===")
    cat_sorted = sorted(category_stats.items(),
                        key=lambda x: x[1]["fail"], reverse=True)
    for cat, stats in cat_sorted:
        total = stats["pass"] + stats["fail"]
        pct = 100 * stats["pass"] / max(total, 1)
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {cat:20s}  {stats['pass']:4d}/{total:4d} ({pct:5.1f}%)  [{bar}]")
    print()

    # Per-instruction summary.
    if summary_mode:
        print("=== Per-Instruction Summary ===")
        current_cat = None
        for name, stats in sorted_instrs:
            cat = categorize(name)
            if failing_only and stats["fail"] == 0:
                continue
            if cat != current_cat:
                current_cat = cat
                print(f"\n  --- {cat} ---")
            status = "PASS" if stats["fail"] == 0 else "FAIL"
            pct = 100 * stats["pass"] / max(stats["total"], 1)
            print(f"    {status}  {name:40s}  {stats['pass']:3d}/{stats['total']:3d} ({pct:.0f}%)")
    else:
        # Detailed mode: show failing combos.
        print("=== Failing Instructions (with details) ===")
        current_cat = None
        for name, stats in sorted_instrs:
            if stats["fail"] == 0:
                continue
            cat = categorize(name)
            if cat != current_cat:
                current_cat = cat
                print(f"\n  --- {cat} ---")
            pct = 100 * stats["pass"] / max(stats["total"], 1)
            print(f"  {name:40s}  {stats['pass']:3d}/{stats['total']:3d} ({pct:.0f}%)")
            # Show up to 3 failing combos.
            for detail in stats["details"][:3]:
                ops = ", ".join(f"{k}={v}" for k, v in detail["operands"].items())
                print(f"    batch={detail['batch']} combo={detail['combo']}  {ops}")
            if len(stats["details"]) > 3:
                print(f"    ... and {len(stats['details']) - 3} more")

    return Totals(total_pass, total_fail, skipped_cases)


def main():
    parser = argparse.ArgumentParser(description="Analyze ISA test results at test-point level")
    parser.add_argument("suite_dir", help="the ISA suite directory (isa-test-gen.py --suite-dir)")
    parser.add_argument("hw_tag", nargs="?", help="hardware run tag (default: the last run)")
    parser.add_argument("emu_tag", nargs="?", help="emulator run tag")
    parser.add_argument("--category", default=None,
                        help="Filter by category (e.g., 'vector-mac', 'scalar-alu')")
    parser.add_argument("--failing", action="store_true",
                        help="Only show failing instructions")
    parser.add_argument("--summary", action="store_true",
                        help="One line per instruction (no combo details)")
    args = parser.parse_args()
    if bool(args.hw_tag) != bool(args.emu_tag):
        parser.error("give both HW_TAG and EMU_TAG, or neither")

    try:
        suite = S.load_suite(args.suite_dir)
        hw_tag, emu_tag = (args.hw_tag, args.emu_tag) if args.hw_tag else last_tags(suite.path)
    except S.SuiteError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Suite: {suite.path}")
    print(f"Tags:  {hw_tag} (hw) vs {emu_tag} (emu)")
    print()

    analyze(suite, hw_tag, emu_tag, args.category, args.failing, args.summary)


if __name__ == "__main__":
    main()
