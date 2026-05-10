#!/usr/bin/env python3
"""Standalone trace preparation tool for npu-xrt bridge tests.

Takes a test source directory and produces a traced build directory ready for
compilation by either Chess or Peano.  This tool is compiler-independent --
the same traced artifacts are used for both.

Steps:
1. Read aie.mlir, apply NPUDEVICE substitution -> aie_arch.mlir (implicit)
2. Inject trace via mlir-trace-inject.py -> aie_traced.mlir + trace_config.json
3. Patch test.cpp via tree-sitter (cpp_trace_patch) + BDF patch -> test_traced.cpp
4. Write prepare-status.txt (OK / FAIL / SKIP)

Source-detect / route-planning utilities live in tools/deprecated/trace-inject.py
(loaded as a module). Their imports of mlir-aie's old per-tile-type setup
helpers were removed upstream, so we no longer call inject_trace from there;
all injection now goes through the declarative path in mlir-trace-inject.py.

Single source of truth: ``trace_config.json`` written by mlir-trace-inject;
schema at ``tools/trace_config_schema.json``; spec at
``docs/superpowers/findings/2026-05-05-trace-config-schema.md``. Every
downstream tool (cpp_trace_patch, parse-trace, trace_compare, bridge script)
reads this file. There is no ``events.json`` and no stdout-passed arg_idx.

Usage:
    trace-prepare.py <test_source_dir> --output <dir> [--trace-size BYTES]
        [--device auto] [--skip-mlir]
        [--test-quarantine FILE] [--trace-quarantine FILE]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trace_config import load as trace_config_load  # noqa: E402


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

def inject_trace_declarative(
    mlir_text: str,
    trace_size: int,
    trace_config_path: Path,
    *,
    test_name: str,
    src_mlir: Path,
    shim_sweep_events: str | None = None,
    memtile_sweep_events: str | None = None,
    memtile_sel_channels: str | None = None,
    memmod_sweep_events: str | None = None,
    trace_mode: str | None = None,
) -> tuple[str, dict]:
    """Run mlir-trace-inject.py on the MLIR text.

    Returns ``(traced_mlir_text, trace_config_dict)``.  The injector writes
    the trace_config.json at *trace_config_path*; this function reads it
    back so the orchestrator can use it without re-parsing.

    Args:
        mlir_text: source MLIR (after device-name substitution and
            widening); may be munged from the on-disk source.
        trace_size: trace BO size in bytes.
        trace_config_path: where the injector should write trace_config.json.
        test_name: stable test identifier for the config's test_name field.
        src_mlir: original on-disk MLIR path for the config's src_mlir
            field. We pass the MLIR text through a tempfile (so widened/
            substituted variants don't write back to source), but the
            recorded path should still point at the real source so the
            cache invalidation upstream uses a stable identifier.

    The declarative path lets aiecc lower the trace setup the same way it
    handles every other aie.trace.* op, which avoids the brittle
    aie.utils.trace.setup imports that broke when mlir-aie deprecated
    the per-tile-type configure_*_aie2 helpers.
    """
    inject_tool = Path(__file__).parent / "mlir-trace-inject.py"
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / "in.mlir"
        out_path = td_path / "out.mlir"
        in_path.write_text(mlir_text)
        cmd = [sys.executable, str(inject_tool),
               "--input", str(in_path),
               "--out", str(out_path),
               "--buffer-size", str(trace_size),
               "--trace-config-out", str(trace_config_path),
               "--config-test-name", test_name,
               "--config-src-mlir", str(src_mlir.resolve())]
        # Opt-in shim trace injection. Only forward when set; the inject
        # tool's default (None) leaves row-0 tiles untraced, preserving
        # pre-stage-1 behavior for callers that don't pass this through.
        if shim_sweep_events is not None:
            cmd += ["--shim-sweep-events", shim_sweep_events]
        # Opt-in memtile trace injection (stage 2 / #373). Same forwarding
        # pattern as shim above: None leaves row-1 tiles untraced.
        if memtile_sweep_events is not None:
            cmd += ["--memtile-sweep-events", memtile_sweep_events]
        # Opt-in memtile DMA Event Channel Selection register override.
        # None leaves register 0xA06A0 at its reset value (every SEL slot
        # at channel 0). Used by #355a multi-channel memtile attribution
        # to redirect SEL slots at non-zero physical channels.
        if memtile_sel_channels is not None:
            cmd += ["--memtile-sel-channels", memtile_sel_channels]
        # Opt-in memmod trace injection (stage 3 / #374). Same pattern: None
        # leaves the compute tile's memory-module trace unit alone; any
        # non-None value emits a second aie.trace decl per compute tile
        # alongside the core decl.
        if memmod_sweep_events is not None:
            cmd += ["--memmod-sweep-events", memmod_sweep_events]
        # Opt-in trace mode override. Default (None) keeps mlir-trace-inject's
        # own default (event_pc / mode 1). For cycle-delta calibration
        # (#355a) callers pass "event_time" so the tile trace unit emits
        # cycle deltas alongside each event instead of PCs.
        if trace_mode is not None:
            cmd += ["--trace-mode", trace_mode]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mlir-trace-inject failed (exit {proc.returncode}):\n"
                f"{proc.stderr.strip()}"
            )
        traced_text = out_path.read_text()

    cfg = trace_config_load(trace_config_path)
    return traced_text, cfg


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
    shim_sweep_events: str | None = None,
    memtile_sweep_events: str | None = None,
    memtile_sel_channels: str | None = None,
    memmod_sweep_events: str | None = None,
    trace_mode: str | None = None,
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

    trace_config = None  # set by MLIR injection below; None when skipped

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

        output_dir.mkdir(parents=True, exist_ok=True)
        trace_config_path = output_dir / "trace_config.json"

        try:
            traced_mlir, trace_config = inject_trace_declarative(
                mlir_text, trace_size, trace_config_path,
                test_name=test_name,
                src_mlir=test_dir / "aie.mlir",
                shim_sweep_events=shim_sweep_events,
                memtile_sweep_events=memtile_sweep_events,
                memtile_sel_channels=memtile_sel_channels,
                memmod_sweep_events=memmod_sweep_events,
                trace_mode=trace_mode,
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

        # Write traced MLIR. (trace_config.json was already written by the
        # injector via --trace-config-out above.)
        (output_dir / "aie_traced.mlir").write_text(traced_mlir)

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

        # The kernel arg slot for the trace BO comes from trace_config.json,
        # the single source of truth written by mlir-trace-inject. When
        # --skip-mlir is set we have no config -- apply only the BDF patch
        # (already done above) and skip the trace BO injection.
        if trace_config is not None:
            try:
                cpp_source = patch_test_cpp(
                    cpp_source,
                    trace_size=trace_config["buffer"]["size_bytes"],
                    trace_arg_index=trace_config["buffer"]["kernel_arg_slot"],
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
    parser.add_argument(
        "--shim-sweep-events",
        default=None,
        help="forward through to mlir-trace-inject's --shim-sweep-events. "
             "Default (None) leaves row-0 tiles untraced. Pass 'all' (or a "
             "comma-separated event list) to inject shim DMA trace ops.",
    )
    parser.add_argument(
        "--memtile-sweep-events",
        default=None,
        help="forward through to mlir-trace-inject's --memtile-sweep-events. "
             "Default (None) leaves row-1 tiles untraced. Pass 'all' (or a "
             "comma-separated event list) to inject memtile DMA-port trace ops.",
    )
    parser.add_argument(
        "--memtile-sel-channels",
        default=None,
        help="forward through to mlir-trace-inject's --memtile-sel-channels. "
             "Default (None) leaves the DMA_Event_Channel_Selection register "
             "(0xA06A0) at its reset value (every SEL slot at channel 0). "
             "Pass 'SLOT:CHANNEL' pairs (e.g. 'S2MM_SEL1:1,MM2S_SEL1:1') to "
             "redirect SEL slots at non-zero physical channels.",
    )
    parser.add_argument(
        "--memmod-sweep-events",
        default=None,
        help="forward through to mlir-trace-inject's --memmod-sweep-events. "
             "Default (None) leaves the compute-tile memory-module trace "
             "unit alone. Pass 'all' (or a comma-separated event list) to "
             "inject a second aie.trace decl per compute tile (memmod) "
             "alongside its core trace.",
    )
    parser.add_argument(
        "--trace-mode",
        default=None,
        choices=("event_time", "event_pc", "inst_exec"),
        help="forward through to mlir-trace-inject's --trace-mode. Default "
             "(None) inherits mlir-trace-inject's default (event_pc / "
             "mode 1). Pass 'event_time' for cycle-delta-anchored traces "
             "(needed by tools/dma-fill-measure.py for #355a calibration).",
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
        shim_sweep_events=args.shim_sweep_events,
        memtile_sweep_events=args.memtile_sweep_events,
        memtile_sel_channels=args.memtile_sel_channels,
        memmod_sweep_events=args.memmod_sweep_events,
        trace_mode=args.trace_mode,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
