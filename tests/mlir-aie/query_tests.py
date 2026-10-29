#!/usr/bin/env python3
"""
Query and filter mlir-aie tests from the manifest.

This utility provides a simple interface to search and filter the test
manifest by architecture, category, features, and other criteria.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def load_manifest(manifest_path: Path) -> dict:
    """Load the test manifest."""
    with open(manifest_path) as f:
        return json.load(f)


def filter_tests(
    tests: list[dict],
    arch: Optional[str] = None,
    device: Optional[str] = None,
    category: Optional[str] = None,
    feature: Optional[str] = None,
    has_host_test: Optional[bool] = None,
    source_type: Optional[str] = None,
    name_pattern: Optional[str] = None,
) -> list[dict]:
    """Filter tests by various criteria."""
    result = tests

    if arch:
        result = [t for t in result if t.get("architecture") == arch]

    if device:
        result = [t for t in result if device in t.get("devices", [])]

    if category:
        result = [t for t in result if t.get("category") == category]

    if feature:
        result = [t for t in result if feature in t.get("features", [])]

    if has_host_test is not None:
        result = [t for t in result if t.get("has_host_test") == has_host_test]

    if source_type:
        result = [t for t in result if t.get("source_type") == source_type]

    if name_pattern:
        pattern = name_pattern.lower()
        result = [t for t in result if pattern in t.get("name", "").lower()
                  or pattern in t.get("source_path", "").lower()]

    return result


def print_summary(tests: list[dict]) -> None:
    """Print summary statistics for a set of tests."""
    if not tests:
        print("No tests match the criteria.")
        return

    # Count by architecture
    arch_counts: dict[str, int] = {}
    for t in tests:
        arch = t.get("architecture", "unknown")
        arch_counts[arch] = arch_counts.get(arch, 0) + 1

    # Count by category
    cat_counts: dict[str, int] = {}
    for t in tests:
        cat = t.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Count by device
    device_counts: dict[str, int] = {}
    for t in tests:
        for dev in t.get("devices", []):
            device_counts[dev] = device_counts.get(dev, 0) + 1

    # Feature counts
    feature_counts: dict[str, int] = {}
    for t in tests:
        for feat in t.get("features", []):
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    print(f"Total: {len(tests)} tests\n")

    print("By Architecture:")
    for arch, count in sorted(arch_counts.items()):
        print(f"  {arch}: {count}")

    print("\nBy Category:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    print("\nBy Device:")
    for dev, count in sorted(device_counts.items()):
        print(f"  {dev}: {count}")

    print("\nBy Feature:")
    for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {count}")

    # Host test stats
    with_host = sum(1 for t in tests if t.get("has_host_test"))
    with_python = sum(1 for t in tests if t.get("has_python_test"))
    print(f"\nWith host test (test.cpp): {with_host}")
    print(f"With Python test (test.py): {with_python}")


def print_list(tests: list[dict], verbose: bool = False) -> None:
    """Print a list of tests."""
    if not tests:
        print("No tests match the criteria.")
        return

    for t in tests:
        if verbose:
            print(f"\n{t['name']}")
            print(f"  Path: {t['source_path']}")
            print(f"  Architecture: {t['architecture']}")
            print(f"  Devices: {', '.join(t.get('devices', []))}")
            print(f"  Category: {t.get('category', 'unknown')}")
            print(f"  Source: {t.get('source_type', 'unknown')}")
            print(f"  Features: {', '.join(t.get('features', []))}")
            print(f"  Host test: {t.get('has_host_test', False)}")
            if t.get("requires_kernels"):
                print(f"  Kernels: {', '.join(t['requires_kernels'])}")
        else:
            # Compact format
            arch = t.get("architecture", "?")
            cat = t.get("category", "?")
            host = "H" if t.get("has_host_test") else "-"
            print(f"[{arch:5}] [{cat:11}] {host} {t['source_path']}")


def print_paths(tests: list[dict]) -> None:
    """Print just the source paths, one per line."""
    for t in tests:
        print(t["source_path"])


def main():
    parser = argparse.ArgumentParser(
        description="Query mlir-aie tests from the manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all AIE2 tests
  %(prog)s --arch aie2

  # Find integration tests with DMA feature
  %(prog)s --category integration --feature dma

  # Find tests for npu1 device
  %(prog)s --device npu1

  # Search by name
  %(prog)s --name passthrough

  # Show summary statistics
  %(prog)s --summary

  # Output just paths (for scripting)
  %(prog)s --arch aie2 --paths
""",
    )

    # Filter options
    parser.add_argument(
        "--arch",
        choices=["aie1", "aie2", "aie2p", "multi"],
        help="Filter by target architecture",
    )
    parser.add_argument(
        "--device",
        help="Filter by device (e.g., npu1, xcvc1902)",
    )
    parser.add_argument(
        "--category",
        choices=["unit_test", "integration", "example", "tutorial", "benchmark", "exercise"],
        help="Filter by test category",
    )
    parser.add_argument(
        "--feature",
        help="Filter by feature (e.g., dma, locks, objectfifo)",
    )
    parser.add_argument(
        "--source-type",
        choices=["mlir", "python", "both"],
        help="Filter by source type",
    )
    parser.add_argument(
        "--has-host-test",
        action="store_true",
        help="Only show tests with host test (test.cpp)",
    )
    parser.add_argument(
        "--no-host-test",
        action="store_true",
        help="Only show tests without host test",
    )
    parser.add_argument(
        "--name",
        help="Filter by name pattern (case-insensitive)",
    )

    # Output options
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with all test details",
    )
    parser.add_argument(
        "--paths",
        action="store_true",
        help="Output just paths (for scripting)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Just print the count of matching tests",
    )

    # Manifest location
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "manifest.json",
        help="Path to manifest.json",
    )

    args = parser.parse_args()

    # Load manifest
    if not args.manifest.exists():
        print(f"Error: Manifest not found at {args.manifest}", file=sys.stderr)
        print("Run discover_tests.py first to generate the manifest.", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    tests = manifest.get("tests", [])

    # Apply filters
    has_host_test = None
    if args.has_host_test:
        has_host_test = True
    elif args.no_host_test:
        has_host_test = False

    filtered = filter_tests(
        tests,
        arch=args.arch,
        device=args.device,
        category=args.category,
        feature=args.feature,
        has_host_test=has_host_test,
        source_type=args.source_type,
        name_pattern=args.name,
    )

    # Output
    if args.count:
        print(len(filtered))
    elif args.json:
        print(json.dumps(filtered, indent=2))
    elif args.paths:
        print_paths(filtered)
    elif args.summary:
        print_summary(filtered)
    else:
        print_list(filtered, verbose=args.verbose)


if __name__ == "__main__":
    main()
