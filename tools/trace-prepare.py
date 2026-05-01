#!/usr/bin/env python3
"""Standalone trace preparation tool for npu-xrt bridge tests.

Takes a test source directory and produces a traced build directory ready for
compilation by either Chess or Peano.  This tool is compiler-independent --
the same traced artifacts are used for both.

Steps:
1. Read aie.mlir, apply NPUDEVICE substitution -> aie_arch.mlir (implicit)
2. Inject trace via mlir-trace-inject.py (declarative aie.trace ops) -> aie_traced.mlir + manifest
3. Patch test.cpp via tree-sitter (cpp_trace_patch) + BDF patch -> test_traced.cpp
4. Write events.json from injected MLIR (tiles_traced; trace_ddr_id is
   derived later by aiecc lowering and patched in via cpp_trace_patch's
   heuristic when None here)
5. Write prepare-status.txt (OK / FAIL / SKIP)

Source-detect / route-planning utilities live in tools/deprecated/trace-inject.py
(loaded as a module). Their imports of mlir-aie's old per-tile-type setup
helpers were removed upstream, so we no longer call inject_trace from there;
all injection now goes through the declarative path in mlir-trace-inject.py.

Usage:
    trace-prepare.py <test_source_dir> --output <dir> [--trace-size BYTES]
        [--device auto] [--skip-mlir]
        [--test-quarantine FILE] [--trace-quarantine FILE]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
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
      xrt::device device = xrt::device(device_index);  ->  xrt::device device = _bdf ? ...
    """
    source = source.replace(
        "unsigned int device_index = 0;",
        'const char* _bdf = std::getenv("XRT_DEVICE_BDF");',
    )
    source = source.replace(
        "auto device = xrt::device(device_index);",
        'auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);',
    )
    source = source.replace(
        "xrt::device device = xrt::device(device_index);",
        "xrt::device device = "
        "_bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);",
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
# Declarative trace injection (replaces the deprecated imperative path)
# ---------------------------------------------------------------------------

# Per-tile aie.trace declaration in the injected MLIR. Symbol form is
# emitted by mlir-trace-inject._trace_sym as "trace_t<col>_<row>"; we match
# it back out here to build the tiles_traced manifest entry.
_TRACE_DECL_RE = re.compile(
    r"aie\.trace\s+@trace_t(\d+)_(\d+)\s*\(\s*%[A-Za-z0-9_]+\s*\)"
)


def inject_trace_declarative(
    mlir_text: str, trace_size: int
) -> tuple[str, dict]:
    """Run mlir-trace-inject.py on the MLIR text; return (traced, manifest).

    Replaces the deprecated trace-inject.py inject_trace() call. The
    declarative path lets aiecc lower the trace setup the same way it
    handles every other aie.trace.* op, which avoids the brittle
    aie.utils.trace.setup imports that broke when mlir-aie deprecated
    the per-tile-type configure_*_aie2 helpers.

    Manifest fields:
      - trace_size: passed through (aie.trace.host_config buffer_size)
      - trace_ddr_id: None. The trace BO arg index is added by aiecc
        during runtime-sequence lowering, after this step. Downstream
        consumers (cpp_trace_patch) fall back to a positional heuristic
        when this is None.
      - tiles_traced: derived from the injected aie.trace declarations
        by symbol-name parsing. Currently only compute cores (rows >= 2)
        get trace declarations from mlir-trace-inject.
    """
    inject_tool = Path(__file__).parent / "mlir-trace-inject.py"
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / "in.mlir"
        out_path = td_path / "out.mlir"
        in_path.write_text(mlir_text)
        proc = subprocess.run(
            [sys.executable, str(inject_tool),
             "--input", str(in_path),
             "--out", str(out_path),
             "--buffer-size", str(trace_size)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mlir-trace-inject failed (exit {proc.returncode}):\n"
                f"{proc.stderr.strip()}"
            )
        traced_text = out_path.read_text()

    tiles_traced = [
        {"col": int(col), "row": int(row), "tile_type": "core",
         "events": "default_core_8"}
        for col, row in _TRACE_DECL_RE.findall(traced_text)
    ]
    manifest = {
        "trace_size": trace_size,
        "trace_ddr_id": None,
        "tiles_traced": tiles_traced,
    }
    return traced_text, manifest


def lower_trace_ops_with_lateral_routing(mlir_text: str) -> str:
    """Run aiecc's trace lowering pipeline externally with lateral-routing on.

    Mirrors aiecc.cpp's `runTraceLoweringPipeline`: 4 device-level passes,
    but adds `lateral-routing=true` on the first one so trace destinations
    can move to spare shim columns when col 0's S2MM channels are taken.
    aiecc itself has no CLI knob for that option; doing it here keeps
    mlir-aie vanilla.

    aiecc skips its own trace lowering when `hasTraceOps()` is false, which
    holds after these passes consume every aie.trace op.
    """
    pipeline = (
        "builtin.module(aie.device("
        "aie-insert-trace-flows{lateral-routing=true},"
        "aie-trace-to-config,"
        "aie-trace-pack-reg-writes,"
        "aie-inline-trace-config"
        "))"
    )
    proc = subprocess.run(
        ["aie-opt", f"--pass-pipeline={pipeline}", "-"],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"aie-opt trace lowering failed (exit {proc.returncode}): "
            f"{proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}"
        )
    return proc.stdout


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
        # Source-type detection / MLIR loading / route planning still come
        # from the deprecated trace-inject module -- those helpers don't
        # touch the broken aie.utils.trace.setup imports. Only inject_trace
        # is migrated to the declarative path (mlir-trace-inject.py).
        tools_dir = str(Path(__file__).parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "trace_inject",
            os.path.join(tools_dir, "deprecated", "trace-inject.py"),
        )
        trace_inject = importlib.util.module_from_spec(_spec)
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

        # Plan trace route. With the declarative injection path, aiecc
        # picks the routing automatically so the planner is advisory --
        # but if it says infeasible, declarative injection is unlikely
        # to find a route either, so we still gate on it.
        plan = trace_inject.plan_trace_route(mlir_text)
        if not plan.feasible:
            msg = f"FAIL trace route infeasible: {plan.reason}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        if plan.widened_device:
            mlir_text, _ = trace_inject.widen_device(mlir_text)
            print(f"  {test_name}: widened to {plan.widened_device} for trace",
                  file=sys.stderr)

        try:
            traced_mlir, manifest_partial = inject_trace_declarative(
                mlir_text, trace_size,
            )
        except Exception as e:
            msg = f"FAIL trace injection: {e}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        # Lower the trace ops via aie-opt before handing the MLIR to aiecc.
        # We run the same 4-pass trace pipeline aiecc would have run in-memory,
        # but with `lateral-routing=true` so the AIEInsertTraceFlows pass can
        # redirect trace destinations to spare shim columns when the
        # application has saturated col 0's S2MM channels (ctrl-packet,
        # packet_flow_fanout, etc.). Vanilla aiecc has no CLI knob for this
        # option, so we lower externally; aiecc's `hasTraceOps` check sees
        # the ops are already gone and skips its own (un-lateral) lowering.
        try:
            traced_mlir = lower_trace_ops_with_lateral_routing(traced_mlir)
        except Exception as e:
            msg = f"FAIL trace lowering: {e}"
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
