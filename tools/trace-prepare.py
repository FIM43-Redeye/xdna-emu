#!/usr/bin/env python3
"""Standalone trace preparation tool for npu-xrt bridge tests.

Takes a test source directory and produces a traced build directory ready for
compilation by either Chess or Peano.  This tool is compiler-independent --
the same traced artifacts are used for both.

Steps:
1. Read aie.mlir, apply NPUDEVICE substitution -> aie_arch.mlir (implicit)
2. Call trace-inject.py functions as library -> aie_traced.mlir + manifest
3. Patch test.cpp via tree-sitter (cpp_trace_patch) + BDF patch -> test_traced.cpp
4. Write events.json from trace-inject manifest
5. Write prepare-status.txt (OK / FAIL / SKIP)

Usage:
    trace-prepare.py <test_source_dir> --output <dir> [--trace-size BYTES]
        [--device auto] [--skip-mlir]
        [--test-quarantine FILE] [--trace-quarantine FILE]
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Quarantine helpers
# ---------------------------------------------------------------------------

def load_quarantine(path: Path) -> set[str]:
    """Load a quarantine file: one test name per line, # comments.

    Returns an empty set if the file does not exist.
    """
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        entry = line.split("#")[0].strip()
        if entry:
            names.add(entry)
    return names


def is_test_quarantined(name: str, quarantine_path: Path) -> bool:
    """Check whether a test name appears in the test quarantine file."""
    return name in load_quarantine(quarantine_path)


def is_trace_quarantined(name: str, quarantine_path: Path) -> bool:
    """Check whether a test name appears in the trace quarantine file."""
    return name in load_quarantine(quarantine_path)


# ---------------------------------------------------------------------------
# BDF patch (same transform as the bridge script compile_one)
# ---------------------------------------------------------------------------

def apply_bdf_patch(source: str) -> str:
    """Replace hardcoded device_index with XRT_DEVICE_BDF env var lookup.

    This is the same transformation the bridge script applies via sed:
      unsigned int device_index = 0;  ->  const char* _bdf = ...
      auto device = xrt::device(device_index);  ->  auto device = _bdf ? ... : ...
    """
    source = source.replace(
        "unsigned int device_index = 0;",
        'const char* _bdf = std::getenv("XRT_DEVICE_BDF");',
    )
    source = source.replace(
        "auto device = xrt::device(device_index);",
        'auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);',
    )
    return source


# ---------------------------------------------------------------------------
# Status file writer
# ---------------------------------------------------------------------------

def write_status(output_dir: Path, status: str) -> None:
    """Write prepare-status.txt to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prepare-status.txt").write_text(status + "\n")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def prepare_trace(
    test_dir: Path,
    output_dir: Path,
    trace_size: int,
    device: str,
    skip_mlir: bool,
    test_quarantine_path: Path,
    trace_quarantine_path: Path,
) -> int:
    """Run the trace preparation pipeline.

    Returns 0 on success or SKIP, nonzero on failure.
    """
    test_name = test_dir.name

    # -- Quarantine checks --------------------------------------------------

    if is_test_quarantined(test_name, test_quarantine_path):
        write_status(output_dir, "SKIP test-quarantined")
        print(f"SKIP {test_name}: test-quarantined", file=sys.stderr)
        return 0

    if is_trace_quarantined(test_name, trace_quarantine_path):
        write_status(output_dir, "SKIP trace-quarantined")
        print(f"SKIP {test_name}: trace-quarantined", file=sys.stderr)
        return 0

    # -- MLIR preparation ---------------------------------------------------

    manifest_partial = None  # set by MLIR injection below; None when skipped

    if not skip_mlir:
        # Import trace_inject as a library.  The file is named with a
        # hyphen (trace-inject.py), so we use importlib for the import.
        # Module-level functions (detect_source_type, get_mlir_text) work
        # without the mlir-aie Python API.
        tools_dir = str(Path(__file__).parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "trace_inject",
            os.path.join(tools_dir, "trace-inject.py"),
        )
        trace_inject = importlib.util.module_from_spec(_spec)
        # Register in sys.modules so ProcessPoolExecutor workers can
        # pickle/unpickle functions defined in trace-inject.py.
        sys.modules["trace_inject"] = trace_inject
        _spec.loader.exec_module(trace_inject)

        try:
            source_type = trace_inject.detect_source_type(test_dir)
            mlir_text = trace_inject.get_mlir_text(
                test_dir, source_type, device,
            )
        except SystemExit:
            msg = "FAIL source detection or MLIR loading"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        # Check for already-traced MLIR.
        if trace_inject.has_existing_trace(mlir_text):
            write_status(output_dir, "SKIP already-traced")
            print(f"SKIP {test_name}: already has trace configuration",
                  file=sys.stderr)
            return 0

        # Plan trace route.
        plan = trace_inject.plan_trace_route(mlir_text)
        if not plan.feasible:
            msg = f"FAIL trace route infeasible: {plan.reason}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        # If the planner widened the device for trace routing room,
        # apply the same widening to the MLIR before injection.
        if plan.widened_device:
            mlir_text, _ = trace_inject.widen_device(mlir_text)
            print(f"  {test_name}: widened to {plan.widened_device} for trace",
                  file=sys.stderr)

        # Inject trace.
        try:
            traced_mlir, manifest_partial = trace_inject.inject_trace(
                mlir_text, trace_size, plan,
            )
        except Exception as e:
            msg = f"FAIL trace injection: {e}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        # Write traced MLIR.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "aie_traced.mlir").write_text(traced_mlir)

        # Write events.json from the manifest's tile trace info.
        events_data = {
            "trace_size": manifest_partial.get("trace_size", trace_size),
            "trace_ddr_id": manifest_partial.get("trace_ddr_id"),
            "tiles_traced": manifest_partial.get("tiles_traced", []),
        }
        (output_dir / "events.json").write_text(
            json.dumps(events_data, indent=2) + "\n"
        )

    # -- C++ patching -------------------------------------------------------

    test_cpp_path = test_dir / "test.cpp"
    if test_cpp_path.exists():
        # Ensure tools/ is on sys.path for cpp_trace_patch import.
        tools_dir = str(Path(__file__).parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

        cpp_source = test_cpp_path.read_text()

        # Apply BDF patch first (simple string replacement).
        cpp_source = apply_bdf_patch(cpp_source)

        # Apply tree-sitter trace buffer injection.
        try:
            from cpp_trace_patch import patch_test_cpp, PatchError
        except ImportError:
            msg = "FAIL cannot import cpp_trace_patch"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        # Compute the kernel argument index for the trace buffer from the
        # MLIR metadata.  trace_ddr_id is the runtime_sequence position;
        # the kernel arg index is offset by 3 (opcode, instr BO, instr count).
        # When --skip-mlir is used, manifest_partial is None -> falls back
        # to heuristic in the patcher.
        trace_arg_index = None
        if manifest_partial is not None:
            trace_ddr_id = manifest_partial.get("trace_ddr_id")
            if trace_ddr_id is not None:
                trace_arg_index = trace_ddr_id + 3

        try:
            cpp_source = patch_test_cpp(
                cpp_source, trace_size=trace_size,
                trace_arg_index=trace_arg_index,
            )
        except PatchError as e:
            msg = f"FAIL C++ patching: {e}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "test_traced.cpp").write_text(cpp_source)

    # -- Done ---------------------------------------------------------------

    write_status(output_dir, "OK")
    print(f"OK {test_name}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare trace-enabled build artifacts for npu-xrt tests",
        prog="trace-prepare.py",
    )
    parser.add_argument(
        "test_dir",
        type=Path,
        help="Path to npu-xrt test source directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for traced artifacts",
    )
    parser.add_argument(
        "--trace-size",
        type=int,
        default=1048576,
        help="Trace buffer size in bytes (default: 1MB)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device target for NPUDEVICE substitution (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-mlir",
        action="store_true",
        help="Skip MLIR trace injection (only patch test.cpp)",
    )
    parser.add_argument(
        "--test-quarantine",
        type=Path,
        default=None,
        help="Path to test quarantine file (default: scripts/test-quarantine.txt)",
    )
    parser.add_argument(
        "--trace-quarantine",
        type=Path,
        default=None,
        help="Path to trace quarantine file (default: scripts/trace-quarantine.txt)",
    )
    args = parser.parse_args()

    test_dir = args.test_dir.resolve()
    output_dir = args.output.resolve()

    if not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Resolve quarantine file paths with defaults.
    repo_root = Path(__file__).parent.parent
    test_quarantine = args.test_quarantine or (
        repo_root / "scripts" / "test-quarantine.txt"
    )
    trace_quarantine = args.trace_quarantine or (
        repo_root / "scripts" / "trace-quarantine.txt"
    )

    rc = prepare_trace(
        test_dir=test_dir,
        output_dir=output_dir,
        trace_size=args.trace_size,
        device=args.device,
        skip_mlir=args.skip_mlir,
        test_quarantine_path=test_quarantine,
        trace_quarantine_path=trace_quarantine,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
