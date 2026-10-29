#!/usr/bin/env python3
"""ISA test dependency analysis.

Reads the ISA test manifest and analysis results to determine which instruction
failures might be caused by broken supporting instructions (loads, stores,
vups, vsrs, etc.) versus genuine bugs in the target instruction.

Every ISA test point follows this pattern:
  1. Load inputs from memory (lda, vlda, vups)
  2. Execute the instruction under test
  3. Store outputs to memory (st, vst, vsrs)

If the supporting instructions are broken, the target instruction's test
results are unreliable. This tool maps those dependencies and flags which
failures we can trust vs which we need to investigate more carefully.

Usage:
  python3 tools/isa-dep-analysis.py [--manifest PATH] [--results-dir PATH]
"""

import json
import os
import re
import sys
from collections import defaultdict


def parse_analysis_results(results_dir: str) -> dict[str, tuple[int, int]]:
    """Parse analysis.log to get per-instruction pass/total counts.

    Returns: {instruction_name: (pass_count, total_count)}
    """
    analysis_log = os.path.join(results_dir, "analysis.log")
    if not os.path.isfile(analysis_log):
        print(f"ERROR: analysis.log not found at {analysis_log}", file=sys.stderr)
        sys.exit(1)

    results = {}
    with open(analysis_log) as f:
        for line in f:
            # Match lines like:
            #   PASS  VCLR_vclr                                   3/  3 (100%)
            #   FAIL  VADD_F                                      0/ 18 (0%)
            m = re.match(
                r'\s+(?:PASS|FAIL)\s+(\S+)\s+(\d+)/\s*(\d+)',
                line,
            )
            if m:
                name = m.group(1)
                passed = int(m.group(2))
                total = int(m.group(3))
                results[name] = (passed, total)
    return results


def get_operand_kinds(manifest: dict) -> dict[str, set[str]]:
    """For each instruction, collect all operand register kinds used.

    Returns: {instruction_name: set of (role, kind) tuples}
    where role is 'input' or 'output'.
    """
    # We need the ISA JSON to know operand types.  The manifest only has
    # register assignments.  Instead, we infer from the register names
    # which kind of load/store is used.
    pass


def infer_deps_from_register(reg_name: str) -> list[str]:
    """Given a concrete register name, infer what instructions are needed
    to load/store it in the test harness.

    Returns a list of dependency instruction categories.
    """
    deps = []
    if reg_name.startswith(("r", "s")):
        # Scalar: lda for load, st for store
        deps.append("scalar_load_store")
    elif reg_name.startswith(("wl", "wh")):
        # Vector256: vlda/vst
        deps.append("vector_load_store")
    elif reg_name.startswith("x"):
        # Vector512: 2x vlda/vst (via wl+wh)
        deps.append("vector_load_store")
    elif reg_name.startswith(("bml", "bmh", "cm", "am")):
        # Accumulator: vlda + vups (load), vsrs + vst (store)
        deps.append("vector_load_store")
        deps.append("vups")
        deps.append("vsrs")
    elif reg_name.startswith("y"):
        # Wide vector: 4x vlda
        deps.append("vector_load_store")
    elif reg_name.startswith("qx"):
        # Sparse: vlda only (mask not loaded)
        deps.append("vector_load_store")
    elif reg_name.startswith("q"):
        # Quad register: lda q
        deps.append("quad_load_store")
    elif reg_name.startswith("p"):
        # Pointer: lda + mov
        deps.append("scalar_load_store")
        deps.append("pointer_mov")
    elif reg_name.startswith("m") or reg_name.startswith("dj"):
        # Modifier: lda
        deps.append("scalar_load_store")
    elif reg_name.startswith("cr"):
        # Control register: lda + mov
        deps.append("scalar_load_store")
    return deps


# Maps dependency names to ISA test instructions that validate them.
# If ALL of these pass at 100%, the dependency is trustworthy.
DEP_VALIDATORS = {
    "scalar_load_store": [
        # These instructions validate that scalar lda/st work
        "LDA_dms_lda_idx",
        "ST_dms_sts_idx",
        "MOV_mv_scl",
    ],
    "vector_load_store": [
        # These validate vlda/vst
        "VLDA_dmw_lda_w_ag_idx",
        "VLDB_dmw_ldb_ag_idx",
    ],
    "vups": [
        # UPS is used to load accumulators.  If UPS itself fails,
        # all accumulator tests are suspect.
        # We can't directly check -- UPS is tested via vector-ups category.
    ],
    "vsrs": [
        # SRS is used to store accumulators.
    ],
    "quad_load_store": [
        "LDA_2D_dmv_lda_q",
        "ST_2D_dmv_sts_q",
    ],
    "pointer_mov": [
        "MOV_mv_scl",
    ],
}


def classify_instruction(name: str) -> str:
    """Classify an instruction name into a dependency category."""
    # MAC / matmul instructions
    if any(x in name for x in ("_vmac_", "MAC_", "MSC_", "MUL_vmac")):
        return "matmul"

    # Loads
    if name.startswith(("LDA_", "VLDA_", "VLDB_", "MOVA_")):
        return "load"

    # Stores
    if name.startswith(("ST_", "VST_")):
        return "store"

    # Pointer arithmetic
    if name.startswith("PADD"):
        return "pointer"

    # Branches
    if name in ("JL", "JL_IND", "JNZ", "JNZD", "JZ", "J_jump_imm",
                 "J_jump_ind", "RET"):
        return "branch"

    # Vector compute
    if name.startswith("V"):
        return "vector_compute"

    # Scalar
    return "scalar_compute"


def analyze_dependencies(manifest_path: str, results_dir: str):
    """Main analysis: map instruction failures to their dependencies."""

    with open(manifest_path) as f:
        manifest = json.load(f)

    results = parse_analysis_results(results_dir)

    # Build per-instruction dependency map from manifest operands.
    instr_deps = defaultdict(set)  # instruction_name -> set of dep categories
    instr_regs = defaultdict(set)  # instruction_name -> set of (role, reg_name)

    for batch in manifest["batches"]:
        for test in batch["tests"]:
            name = test["instruction"]
            operands = test.get("operands", {})
            for op_name, reg_name in operands.items():
                deps = infer_deps_from_register(reg_name)
                for d in deps:
                    instr_deps[name].add(d)
                instr_regs[name].add(reg_name)

    # Check dependency health using the test results.
    dep_health = {}
    for dep_name, validators in DEP_VALIDATORS.items():
        if not validators:
            dep_health[dep_name] = "untested"
            continue
        all_pass = True
        for v in validators:
            if v in results:
                passed, total = results[v]
                if passed < total:
                    all_pass = False
            else:
                all_pass = False
        dep_health[dep_name] = "healthy" if all_pass else "degraded"

    # Classify each failing instruction.
    print("=" * 72)
    print("ISA TEST DEPENDENCY ANALYSIS")
    print("=" * 72)
    print()

    # Summary of dependency health.
    print("--- Dependency Health ---")
    for dep_name, health in sorted(dep_health.items()):
        validators = DEP_VALIDATORS.get(dep_name, [])
        status_parts = []
        for v in validators:
            if v in results:
                p, t = results[v]
                status_parts.append(f"{v}: {p}/{t}")
            else:
                status_parts.append(f"{v}: NOT TESTED")
        detail = ", ".join(status_parts) if status_parts else "(no validators)"
        icon = "OK" if health == "healthy" else "??" if health == "untested" else "!!"
        print(f"  [{icon}] {dep_name}: {health}")
        for part in status_parts:
            print(f"        {part}")
    print()

    # Categorize failing instructions.
    trustworthy_failures = []   # Dependencies pass, failure is real
    suspect_failures = []       # Dependencies also fail, can't trust
    untested_deps = []          # Dependencies not tested

    for name, (passed, total) in sorted(results.items()):
        if passed == total:
            continue  # Passing, skip.

        deps = instr_deps.get(name, set())
        category = classify_instruction(name)

        broken_deps = []
        unknown_deps = []
        for d in deps:
            h = dep_health.get(d, "unknown")
            if h == "degraded":
                broken_deps.append(d)
            elif h in ("untested", "unknown"):
                unknown_deps.append(d)

        if broken_deps:
            suspect_failures.append((name, passed, total, broken_deps, category))
        elif unknown_deps:
            untested_deps.append((name, passed, total, unknown_deps, category))
        else:
            trustworthy_failures.append((name, passed, total, category))

    # Report trustworthy failures: these are REAL emulator bugs.
    print("--- TRUSTWORTHY FAILURES (dependencies healthy, bug is real) ---")
    print(f"    {len(trustworthy_failures)} instructions")
    print()

    # Group by category for readability.
    by_cat = defaultdict(list)
    for name, passed, total, cat in trustworthy_failures:
        by_cat[cat].append((name, passed, total))

    total_fixable = 0
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        cat_points = sum(t - p for _, p, t in items)
        total_fixable += cat_points
        print(f"  [{cat}] ({cat_points} fixable test points)")
        for name, passed, total in sorted(items, key=lambda x: x[1] / x[2]):
            pct = 100 * passed / total if total > 0 else 0
            print(f"    {name:50s} {passed:3d}/{total:3d} ({pct:5.1f}%)")
        print()

    print(f"  TOTAL FIXABLE: {total_fixable} test points")
    print()

    # Report suspect failures.
    print("--- SUSPECT FAILURES (dependencies also broken) ---")
    print(f"    {len(suspect_failures)} instructions")
    print("    Fix the dependencies first, then re-test these.")
    print()

    by_dep = defaultdict(list)
    for name, passed, total, broken, cat in suspect_failures:
        for d in broken:
            by_dep[d].append((name, passed, total))

    for dep in sorted(by_dep.keys()):
        items = by_dep[dep]
        dep_points = sum(t - p for _, p, t in items)
        print(f"  Blocked by [{dep}] ({dep_points} test points)")
        for name, passed, total in sorted(items, key=lambda x: x[1] / x[2]):
            pct = 100 * passed / total if total > 0 else 0
            print(f"    {name:50s} {passed:3d}/{total:3d} ({pct:5.1f}%)")
        print()

    # Report untested dependencies.
    if untested_deps:
        print("--- UNTESTED DEPENDENCIES ---")
        print(f"    {len(untested_deps)} instructions with untested deps")
        print()
        for name, passed, total, unknown, cat in untested_deps:
            pct = 100 * passed / total if total > 0 else 0
            deps_str = ", ".join(unknown)
            print(f"    {name:50s} {passed:3d}/{total:3d} ({pct:5.1f}%)  deps: {deps_str}")
        print()

    # Priority recommendations.
    print("=" * 72)
    print("PRIORITY RECOMMENDATIONS")
    print("=" * 72)
    print()

    # Find the biggest bang-for-buck trustworthy failures.
    all_trustworthy = []
    for name, passed, total, cat in trustworthy_failures:
        all_trustworthy.append((total - passed, name, passed, total, cat))
    all_trustworthy.sort(reverse=True)

    # Group by pattern (common prefixes suggest common root cause).
    print("Top 20 trustworthy failures by test point impact:")
    for i, (failing, name, passed, total, cat) in enumerate(all_trustworthy[:20]):
        pct = 100 * passed / total if total > 0 else 0
        print(f"  {i+1:2d}. {name:50s} {passed:3d}/{total:3d} ({pct:5.1f}%)  [{cat}]  -{failing}")
    print()

    # Pattern detection: group failures by common prefix.
    print("Failure patterns (common prefixes, likely same root cause):")
    prefix_groups = defaultdict(list)
    for name, passed, total, cat in trustworthy_failures:
        # Try progressively shorter prefixes.
        parts = name.split("_")
        if len(parts) >= 2:
            prefix = "_".join(parts[:2])
        else:
            prefix = parts[0]
        prefix_groups[prefix].append((name, passed, total))

    for prefix in sorted(prefix_groups.keys(),
                         key=lambda p: sum(t - pp for _, pp, t in prefix_groups[p]),
                         reverse=True):
        items = prefix_groups[prefix]
        if len(items) < 2:
            continue
        total_failing = sum(t - p for _, p, t in items)
        if total_failing < 5:
            continue
        pct_avg = 100 * sum(p for _, p, t in items) / sum(t for _, p, t in items) if sum(t for _, p, t in items) > 0 else 0
        print(f"  {prefix}* ({len(items)} instrs, {total_failing} failing points, avg {pct_avg:.0f}%)")
        for name, passed, total in sorted(items, key=lambda x: x[0]):
            pct = 100 * passed / total if total > 0 else 0
            print(f"    {name:50s} {passed:3d}/{total:3d} ({pct:5.1f}%)")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ISA test dependency analysis")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest.json")
    parser.add_argument("--results-dir", default=None,
                        help="Path to results directory")
    args = parser.parse_args()

    # Auto-detect paths.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    manifest = args.manifest or os.path.join(
        project_dir, "build/isa-tests/manifest.json")
    results_dir = args.results_dir

    if not results_dir:
        # Find the latest results directory.
        base = os.path.join(project_dir, "build/isa-test-results")
        if os.path.islink(os.path.join(base, "latest")):
            results_dir = os.path.realpath(os.path.join(base, "latest"))
        else:
            # Fall back to most recent date directory.
            dates = sorted(
                d for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d)) and d.isdigit()
            )
            if dates:
                results_dir = os.path.join(base, dates[-1])

    if not results_dir or not os.path.isdir(results_dir):
        print("ERROR: No results directory found", file=sys.stderr)
        sys.exit(1)

    print(f"Manifest:    {manifest}")
    print(f"Results dir: {results_dir}")
    print()

    analyze_dependencies(manifest, results_dir)


if __name__ == "__main__":
    main()
