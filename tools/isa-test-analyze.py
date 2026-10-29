#!/usr/bin/env python3
"""Analyze ISA test results at the test-point level.

Reads the manifest and per-batch binary outputs (hw vs emu) to produce
a per-instruction, per-test-point pass/fail report.  Groups results by
instruction name and category for actionable accuracy metrics.

Usage:
    python3 tools/isa-test-analyze.py [--results-dir DIR] [--manifest PATH]
    python3 tools/isa-test-analyze.py --category vector   # filter by category
    python3 tools/isa-test-analyze.py --failing            # only show failures
    python3 tools/isa-test-analyze.py --summary            # one-line-per-instruction
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


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


def compare_test_point(hw_data: bytes, emu_data: bytes,
                       out_offset: int, out_size: int) -> Optional[bool]:
    """Compare a single test point's output bytes.

    Returns True if match, False if mismatch, None if the test point
    falls outside the available data (can't compare).
    """
    end = out_offset + out_size
    if end > len(hw_data) or end > len(emu_data):
        return None  # out of bounds -- skip this test point
    hw_slice = hw_data[out_offset:end]
    emu_slice = emu_data[out_offset:end]
    return hw_slice == emu_slice


def load_batch_data(results_dir: str, batch_idx: int):
    """Load hw and emu binary output for a batch."""
    hw_path = os.path.join(results_dir, f"batch_{batch_idx}_hw.bin")
    emu_path = os.path.join(results_dir, f"batch_{batch_idx}_emu.bin")
    hw_data = open(hw_path, "rb").read() if os.path.exists(hw_path) else None
    emu_data = open(emu_path, "rb").read() if os.path.exists(emu_path) else None
    return hw_data, emu_data


def analyze(manifest_path: str, results_dir: str, category_filter: str = None,
            failing_only: bool = False, summary_mode: bool = False):
    """Run the full analysis."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Per-instruction aggregation.
    # instr_name -> {"pass": int, "fail": int, "total": int, "details": [...]}
    instr_stats = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0, "details": []})
    category_stats = defaultdict(lambda: {"pass": 0, "fail": 0})
    skipped_batches = 0
    total_points = 0

    for batch in manifest["batches"]:
        batch_idx = batch["batch_index"]
        tests = batch.get("tests", [])
        if not tests:
            continue

        # Pair batches (cascade/stream) require --multi-tile mode.
        # Don't count them as skipped in single-tile results.
        source_type = batch.get("source_type", "assembly")
        if source_type in ("cascade_pair", "stream_pair"):
            continue

        hw_data, emu_data = load_batch_data(results_dir, batch_idx)
        if hw_data is None or emu_data is None:
            skipped_batches += 1
            continue

        for test in tests:
            name = test["instruction"]
            cat = categorize(name)

            if category_filter and cat != category_filter:
                continue

            out_offset = test["out_offset"]
            out_size = test["out_size"]

            passed = compare_test_point(hw_data, emu_data, out_offset, out_size)

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
    if skipped_batches:
        print(f"  Skipped batches (missing data): {skipped_batches}")
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

    return total_pass, total_fail


def main():
    parser = argparse.ArgumentParser(description="Analyze ISA test results at test-point level")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with batch_N_{hw,emu}.bin files")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest.json")
    parser.add_argument("--category", default=None,
                        help="Filter by category (e.g., 'vector-mac', 'scalar-alu')")
    parser.add_argument("--failing", action="store_true",
                        help="Only show failing instructions")
    parser.add_argument("--summary", action="store_true",
                        help="One line per instruction (no combo details)")
    args = parser.parse_args()

    # Auto-detect paths.
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = args.manifest or os.path.join(project_dir, "build/isa-tests/manifest.json")

    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Prefer 'latest' directory (the working results folder).
        latest = os.path.join(project_dir, "build/isa-test-results/latest")
        if os.path.isdir(latest):
            results_dir = latest
        else:
            # Fall back to most recent dated archive.
            import glob
            candidates = sorted(glob.glob(
                os.path.join(project_dir, "build/isa-test-results/[0-9]*")))
            if candidates:
                results_dir = candidates[-1]
            else:
                print("ERROR: No results directory found. Run isa-test.sh first.",
                      file=sys.stderr)
                sys.exit(1)

    if not os.path.exists(manifest):
        print(f"ERROR: Manifest not found at {manifest}", file=sys.stderr)
        sys.exit(1)

    print(f"Manifest: {manifest}")
    print(f"Results:  {results_dir}")
    print()

    analyze(manifest, results_dir, args.category, args.failing, args.summary)


if __name__ == "__main__":
    main()
