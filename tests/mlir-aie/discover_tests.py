#!/usr/bin/env python3
"""
Discover and catalog mlir-aie tests for xdna-emu.

This script scans the mlir-aie repository and generates a manifest
describing all tests, their target architectures, and build requirements.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

# Device to architecture mapping
DEVICE_ARCH = {
    "xcvc1902": "aie1",
    "xcve2302": "aie2",
    "xcve2802": "aie2",
    "npu1": "aie2",
    "npu1_1col": "aie2",
    "npu1_2col": "aie2",
    "npu1_3col": "aie2",
    "npu2": "aie2p",
    "npu2_1col": "aie2p",
    "npu2_2col": "aie2p",
    "npu2_3col": "aie2p",
    "npu2_4col": "aie2p",
    "npu2_5col": "aie2p",
    "npu2_6col": "aie2p",
    "npu2_7col": "aie2p",
    "NPUDEVICE": "aie2",  # Placeholder defaults to AIE2/NPU
}

# Patterns to detect device targets in MLIR files
DEVICE_PATTERN = re.compile(r'aie\.device\((\w+)\)')

# Patterns to detect Python device selection
PYTHON_DEVICE_PATTERNS = [
    re.compile(r'AIEDevice\.(\w+)'),
    re.compile(r'dev\s*=\s*(\w+)\(\)'),
    re.compile(r'NPU1|NPU2|XCVC1902', re.IGNORECASE),
]


def get_git_commit(repo_path: Path) -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def detect_device_from_mlir(mlir_path: Path) -> list[str]:
    """Extract device target from MLIR file."""
    devices = []
    try:
        content = mlir_path.read_text()
        matches = DEVICE_PATTERN.findall(content)
        devices.extend(matches)
    except Exception:
        pass
    return list(set(devices))


def detect_devices_from_python(py_path: Path) -> list[str]:
    """Detect supported devices from Python generator."""
    devices = []
    try:
        content = py_path.read_text()

        # Check for direct device references
        if "npu2" in content.lower() or "NPU2" in content:
            devices.append("npu2")
        if "npu1" in content.lower() or "NPU1" in content or "npu" in content:
            devices.append("npu1")
        if "xcvc1902" in content.lower() or "XCVC1902" in content:
            devices.append("xcvc1902")
        if "xcve2802" in content.lower() or "XCVE2802" in content:
            devices.append("xcve2802")

    except Exception:
        pass
    return list(set(devices))


def detect_required_kernels(test_dir: Path) -> list[str]:
    """Detect kernel object files required by the test."""
    kernels = []

    # Check MLIR files for kernel references
    for mlir_file in test_dir.glob("*.mlir"):
        try:
            content = mlir_file.read_text()
            # Look for .o file references in linker scripts or MLIR
            kernel_matches = re.findall(r'(\w+\.o)', content)
            kernels.extend(kernel_matches)
        except Exception:
            pass

    # Check Makefile for kernel dependencies
    makefile = test_dir / "Makefile"
    if makefile.exists():
        try:
            content = makefile.read_text()
            # Look for kernel build targets
            kernel_matches = re.findall(r'build/(\w+\.o)', content)
            kernels.extend(kernel_matches)
        except Exception:
            pass

    return list(set(kernels))


def detect_features(test_dir: Path) -> list[str]:
    """Detect AIE features used by the test."""
    features = []

    # Read all MLIR and Python files
    content = ""
    for f in list(test_dir.glob("*.mlir")) + list(test_dir.glob("*.py")):
        try:
            content += f.read_text()
        except Exception:
            pass

    # Feature detection patterns
    feature_patterns = {
        "dma": r"aie\.dma|DMA|dma_",
        "locks": r"aie\.lock|acquire|release",
        "streams": r"stream|STREAM",
        "cascade": r"cascade|CASCADE",
        "packet_switch": r"packet|PACKET",
        "objectfifo": r"objectfifo|ObjectFifo",
        "memtile": r"memtile|mem_tile|MemTile",
        "shim_dma": r"shim.*dma|ShimTile",
        "core_kernel": r"aie\.core|@core",
        "multi_core": r"tile\([0-9]+,\s*[2-9]\).*tile\([0-9]+,\s*[2-9]\)",
        "multi_column": r"tile\([1-9],",
    }

    for feature, pattern in feature_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            features.append(feature)

    return features


def categorize_test(source_path: str) -> str:
    """Determine test category from path."""
    if "unit_tests" in source_path:
        return "unit_test"
    elif "benchmark" in source_path:
        return "benchmark"
    elif "npu-xrt" in source_path or "Integration" in source_path:
        return "integration"
    elif "programming_examples" in source_path:
        return "example"
    elif "programming_guide" in source_path or "tutorial" in source_path:
        return "tutorial"
    elif "exercise" in source_path:
        return "exercise"
    return "integration"


def discover_test(test_dir: Path, mlir_aie_root: Path) -> Optional[dict]:
    """Analyze a single test directory and return its metadata."""

    # Check for MLIR source
    mlir_files = list(test_dir.glob("aie.mlir")) + list(test_dir.glob("aie*.mlir"))
    py_files = [f for f in test_dir.glob("*.py")
                if not f.name.startswith("test") and "__pycache__" not in str(f)]

    if not mlir_files and not py_files:
        return None

    # Determine source type
    has_mlir = bool(mlir_files)
    has_python_gen = bool(py_files)

    if has_mlir and has_python_gen:
        source_type = "both"
    elif has_mlir:
        source_type = "mlir"
    else:
        source_type = "python"

    # Detect devices
    devices = []
    if has_mlir:
        for mlir_file in mlir_files:
            devices.extend(detect_device_from_mlir(mlir_file))
    if has_python_gen:
        for py_file in py_files:
            devices.extend(detect_devices_from_python(py_file))

    devices = list(set(devices)) if devices else ["npu1"]  # Default

    # Determine architecture
    archs = set()
    for device in devices:
        if device in DEVICE_ARCH:
            archs.add(DEVICE_ARCH[device])

    if len(archs) > 1:
        architecture = "multi"
    elif archs:
        architecture = list(archs)[0]
    else:
        architecture = "aie2"  # Default

    # Build relative path
    try:
        rel_path = test_dir.relative_to(mlir_aie_root)
    except ValueError:
        rel_path = test_dir

    source_path = str(rel_path)
    test_id = source_path.replace("/", "_").replace("\\", "_")

    # Check for test files
    has_host_test = (test_dir / "test.cpp").exists()
    has_python_test = (test_dir / "test.py").exists()

    return {
        "id": test_id,
        "name": test_dir.name,
        "source_path": source_path,
        "architecture": architecture,
        "devices": devices,
        "source_type": source_type,
        "has_host_test": has_host_test,
        "has_python_test": has_python_test,
        "requires_kernels": detect_required_kernels(test_dir),
        "category": categorize_test(source_path),
        "features": detect_features(test_dir),
        "build_status": "untested",
    }


def discover_all_tests(mlir_aie_root: Path) -> list[dict]:
    """Discover all tests in the mlir-aie repository."""
    tests = []

    # Directories to scan
    scan_dirs = [
        "test/npu-xrt",
        "test/unit_tests/aie",
        "test/unit_tests/aie2",
        "test/unit_tests/aie2p",
        "test/unit_tests/chess_compiler_tests",
        "test/unit_tests/chess_compiler_tests_aie2",
        "test/benchmarks",
        "programming_examples/basic",
        "programming_examples/ml",
        "programming_examples/vision",
        "programming_guide/section-2",
        "programming_guide/section-3",
        "programming_guide/section-4",
    ]

    for scan_dir in scan_dirs:
        dir_path = mlir_aie_root / scan_dir
        if not dir_path.exists():
            continue

        # Find test directories (those with aie.mlir or Python generators)
        for item in dir_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                test = discover_test(item, mlir_aie_root)
                if test:
                    tests.append(test)

                # Check subdirectories one level deep
                for subitem in item.iterdir():
                    if subitem.is_dir() and not subitem.name.startswith("."):
                        test = discover_test(subitem, mlir_aie_root)
                        if test:
                            tests.append(test)

    return tests


def main():
    parser = argparse.ArgumentParser(description="Discover mlir-aie tests")
    parser.add_argument(
        "--mlir-aie",
        type=Path,
        default=Path(os.environ.get("MLIR_AIE_PATH", str(Path(__file__).resolve().parent.parent.parent / ".." / "mlir-aie"))),
        help="Path to mlir-aie repository (or set MLIR_AIE_PATH env var)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "manifest.json",
        help="Output manifest file"
    )
    parser.add_argument(
        "--filter-arch",
        choices=["aie1", "aie2", "aie2p", "multi"],
        help="Only include tests for specific architecture"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )

    args = parser.parse_args()

    print(f"Scanning {args.mlir_aie}...")
    tests = discover_all_tests(args.mlir_aie)

    if args.filter_arch:
        tests = [t for t in tests if t["architecture"] == args.filter_arch]

    # Build manifest
    manifest = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "mlir_aie_commit": get_git_commit(args.mlir_aie),
        "tests": tests,
    }

    # Write manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(tests)} tests to {args.output}")

    if args.summary:
        print("\n=== Summary ===")

        # Count by architecture
        arch_counts = {}
        for t in tests:
            arch = t["architecture"]
            arch_counts[arch] = arch_counts.get(arch, 0) + 1

        print("\nBy Architecture:")
        for arch, count in sorted(arch_counts.items()):
            print(f"  {arch}: {count}")

        # Count by category
        cat_counts = {}
        for t in tests:
            cat = t.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        print("\nBy Category:")
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat}: {count}")

        # Count with host tests
        with_host = sum(1 for t in tests if t.get("has_host_test"))
        print(f"\nWith host test (test.cpp): {with_host}")


if __name__ == "__main__":
    main()
