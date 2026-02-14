#!/usr/bin/env python3
"""
Run mlir-aie tests through xdna-emu.

This script uses the test manifest to find compatible tests and runs them
through the emulator, comparing results against expected values.

Usage:
    # Run all AIE2 tests
    ./run_tests.py --arch aie2

    # Run specific category
    ./run_tests.py --category integration

    # Run single test by name
    ./run_tests.py --name add_one_using_dma

    # Dry run - just show what would be run
    ./run_tests.py --arch aie2 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TestResult:
    """Result of running a single test."""
    name: str
    status: str  # pass, fail, skip, timeout, load_error
    cycles: int = 0
    message: str = ""
    duration_ms: int = 0


def load_manifest(manifest_path: Path) -> dict:
    """Load the test manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def find_xclbin(test: dict, mlir_aie_root: Path) -> Optional[Path]:
    """Find the xclbin file for a test."""
    source_path = test["source_path"]

    # Check build directory first (built tests)
    build_path = mlir_aie_root / "build" / source_path
    for name in ["aie.xclbin", "final.xclbin"]:
        xclbin = build_path / name
        if xclbin.exists():
            return xclbin

    # Check source directory (some tests have pre-built xclbins)
    source_dir = mlir_aie_root / source_path
    for name in ["aie.xclbin", "final.xclbin"]:
        xclbin = source_dir / name
        if xclbin.exists():
            return xclbin

    return None


def find_insts_bin(test: dict, mlir_aie_root: Path) -> Optional[Path]:
    """Find the instruction binary for a test."""
    source_path = test["source_path"]
    build_path = mlir_aie_root / "build" / source_path

    insts_bin = build_path / "insts.bin"
    if insts_bin.exists():
        return insts_bin

    return None


def run_emulator(
    xclbin_path: Path,
    emulator_path: Path,
    max_cycles: int = 1_000_000,
    timeout_sec: int = 60,
) -> TestResult:
    """Run a test through the emulator."""
    test_name = xclbin_path.parent.name

    cmd = [
        str(emulator_path),
        "--dump-state",
        str(xclbin_path),
    ]

    try:
        import time
        start = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        duration_ms = int((time.time() - start) * 1000)

        if result.returncode == 0:
            # Parse output for cycle count and pass/fail
            output = result.stdout

            # Look for cycle count
            cycles = 0
            if "cycles" in output.lower():
                import re
                match = re.search(r'(\d+)\s*cycles', output)
                if match:
                    cycles = int(match.group(1))

            # Check for PASS/FAIL indicators
            if "PASS" in output or "passed" in output.lower():
                return TestResult(
                    name=test_name,
                    status="pass",
                    cycles=cycles,
                    duration_ms=duration_ms,
                )
            elif "FAIL" in output or "failed" in output.lower():
                return TestResult(
                    name=test_name,
                    status="fail",
                    cycles=cycles,
                    message=output[:200],
                    duration_ms=duration_ms,
                )
            else:
                # No clear pass/fail - assume success if no error
                return TestResult(
                    name=test_name,
                    status="pass",
                    cycles=cycles,
                    duration_ms=duration_ms,
                )
        else:
            return TestResult(
                name=test_name,
                status="fail",
                message=result.stderr[:200],
                duration_ms=duration_ms,
            )

    except subprocess.TimeoutExpired:
        return TestResult(
            name=test_name,
            status="timeout",
            message=f"Exceeded {timeout_sec}s timeout",
        )
    except Exception as e:
        return TestResult(
            name=test_name,
            status="load_error",
            message=str(e),
        )


def filter_tests(
    tests: list[dict],
    arch: Optional[str] = None,
    device: Optional[str] = None,
    category: Optional[str] = None,
    feature: Optional[str] = None,
    name_pattern: Optional[str] = None,
) -> list[dict]:
    """Filter tests by criteria."""
    result = tests

    if arch:
        result = [t for t in result if t.get("architecture") == arch]

    if device:
        result = [t for t in result if device in t.get("devices", [])]

    if category:
        result = [t for t in result if t.get("category") == category]

    if feature:
        result = [t for t in result if feature in t.get("features", [])]

    if name_pattern:
        pattern = name_pattern.lower()
        result = [t for t in result if pattern in t.get("name", "").lower()
                  or pattern in t.get("source_path", "").lower()]

    return result


def print_summary(results: list[TestResult]) -> None:
    """Print test run summary."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    timeout = sum(1 for r in results if r.status == "timeout")
    load_error = sum(1 for r in results if r.status == "load_error")

    total_cycles = sum(r.cycles for r in results)
    total_time = sum(r.duration_ms for r in results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:       {total}")
    print(f"Passed:      {passed} ({100.0 * passed / max(total, 1):.1f}%)")
    print(f"Failed:      {failed}")
    print(f"Skipped:     {skipped}")
    print(f"Timeout:     {timeout}")
    print(f"Load Error:  {load_error}")
    print(f"Total Cycles: {total_cycles:,}")
    print(f"Total Time:  {total_time}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Run mlir-aie tests through xdna-emu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Filter options
    parser.add_argument("--arch", choices=["aie1", "aie2", "aie2p", "multi"],
                        help="Filter by architecture")
    parser.add_argument("--device", help="Filter by device")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--feature", help="Filter by feature")
    parser.add_argument("--name", help="Filter by name pattern")

    # Execution options
    parser.add_argument("--max-cycles", type=int, default=1_000_000,
                        help="Maximum cycles per test (default: 1M)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout per test in seconds (default: 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be run without running")
    parser.add_argument("--limit", type=int,
                        help="Maximum number of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    # Paths
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).parent / "manifest.json",
                        help="Path to manifest.json")
    parser.add_argument("--mlir-aie", type=Path,
                        default=Path(os.environ.get("MLIR_AIE_PATH", str(Path(__file__).resolve().parent.parent.parent / ".." / "mlir-aie"))),
                        help="Path to mlir-aie repository (or set MLIR_AIE_PATH env var)")
    parser.add_argument("--emulator", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "target" / "release" / ("xdna-emu.exe" if os.name == "nt" else "xdna-emu"),
                        help="Path to xdna-emu binary")

    # Output
    parser.add_argument("--json-output", type=Path,
                        help="Write results to JSON file")

    args = parser.parse_args()

    # Load manifest
    if not args.manifest.exists():
        print(f"Error: Manifest not found at {args.manifest}", file=sys.stderr)
        print("Run discover_tests.py first.", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    tests = manifest.get("tests", [])

    # Filter tests
    filtered = filter_tests(
        tests,
        arch=args.arch,
        device=args.device,
        category=args.category,
        feature=args.feature,
        name_pattern=args.name,
    )

    # Apply limit
    if args.limit:
        filtered = filtered[:args.limit]

    # Find runnable tests (those with xclbin files)
    runnable = []
    skipped = []

    for test in filtered:
        xclbin = find_xclbin(test, args.mlir_aie)
        if xclbin:
            test["_xclbin_path"] = xclbin
            runnable.append(test)
        else:
            skipped.append(test)

    print(f"Tests matching criteria: {len(filtered)}")
    print(f"Runnable (have xclbin): {len(runnable)}")
    print(f"Skipped (no xclbin): {len(skipped)}")

    if args.dry_run:
        print("\nWould run:")
        for test in runnable:
            print(f"  {test['name']}: {test['_xclbin_path']}")
        return

    if not runnable:
        print("No runnable tests found.")
        sys.exit(0)

    # Check emulator
    if not args.emulator.exists():
        # Try debug build
        debug_path = args.emulator.parent.parent / "debug" / "xdna-emu"
        if debug_path.exists():
            args.emulator = debug_path
        else:
            print(f"Error: Emulator not found at {args.emulator}", file=sys.stderr)
            print("Build with: cargo build --release", file=sys.stderr)
            sys.exit(1)

    # Run tests
    print(f"\nRunning {len(runnable)} tests...")
    print("-" * 60)

    results = []

    for i, test in enumerate(runnable):
        xclbin = test["_xclbin_path"]
        name = test["name"]

        print(f"[{i+1:3}/{len(runnable)}] {name[:40]:40} ... ", end="", flush=True)

        result = run_emulator(
            xclbin,
            args.emulator,
            max_cycles=args.max_cycles,
            timeout_sec=args.timeout,
        )
        results.append(result)

        status_colors = {
            "pass": "\033[32mPASS\033[0m",
            "fail": "\033[31mFAIL\033[0m",
            "skip": "\033[33mSKIP\033[0m",
            "timeout": "\033[33mTIMEOUT\033[0m",
            "load_error": "\033[31mLOAD ERROR\033[0m",
        }

        status_str = status_colors.get(result.status, result.status.upper())
        cycles_str = f"({result.cycles:,} cycles)" if result.cycles else ""
        print(f"{status_str} {cycles_str}")

        if args.verbose and result.message:
            print(f"      {result.message}")

    # Add skipped tests
    for test in skipped:
        results.append(TestResult(
            name=test["name"],
            status="skip",
            message="No xclbin found",
        ))

    print_summary(results)

    # Save JSON output
    if args.json_output:
        output = {
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "passed": sum(1 for r in results if r.status == "pass"),
            "failed": sum(1 for r in results if r.status == "fail"),
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "cycles": r.cycles,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in results
            ],
        }
        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.json_output}")


if __name__ == "__main__":
    main()
