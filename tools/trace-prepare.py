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
``xdna-emu-priv: archive/findings/2026-05-05-trace-config-schema.md``. Every
downstream tool (cpp_trace_patch, parse-trace, trace_compare, bridge script)
reads this file. There is no ``events.json`` and no stdout-passed arg_idx.

Usage:
    trace-prepare.py <test_source_dir> --output <dir> [--trace-size BYTES]
        [--device auto] [--skip-mlir]
        [--test-quarantine FILE] [--trace-quarantine FILE]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trace_config import dump as trace_config_dump  # noqa: E402
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
    memmod_sel_locks: str | None = None,
    core_sweep_events: str | None = None,
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
        # Opt-in memtile event selection (DMA channel and lock behind each
        # SELn event). None leaves the selection registers at reset (every
        # selector at channel 0 / lock 0). See tools/trace_selectors.py.
        if memtile_sel_channels is not None:
            cmd += ["--memtile-sel-channels", memtile_sel_channels]
        # Opt-in memmod trace injection (stage 3 / #374). Same pattern: None
        # leaves the compute tile's memory-module trace unit alone; any
        # non-None value emits a second aie.trace decl per compute tile
        # alongside the core decl.
        if memmod_sweep_events is not None:
            cmd += ["--memmod-sweep-events", memmod_sweep_events]
        # Opt-in lock selector programming for the memory module's LOCK_n_REL
        # events (they follow Locks_Event_Selection_n, which resets to lock 0).
        if memmod_sel_locks is not None:
            cmd += ["--memmod-sel-locks", memmod_sel_locks]
        # Opt-in core sweep-event override. Default (None) keeps the injector's
        # own hard-coded core defaults (stall + lock-req events). Forwarding an
        # explicit list lets a caller trace core PORT_RUNNING_<N> (the S2MM/MM2S
        # switch-port cadence) -- needed for per-tile-type S2MM ingress-depth
        # measurement. Unlike shim/memtile, core trace is ALWAYS injected, so
        # this overrides rather than activates.
        if core_sweep_events is not None:
            cmd += ["--core-sweep-events", core_sweep_events]
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


def place_logical_tiles(mlir_text: str) -> str:
    """Resolve `aie.logical_tile` ops to physical `aie.tile` ops.

    The route planner and the injector only see physical tiles, so designs
    written with unplaced logical tiles (`aie.logical_tile<CoreTile>(?, ?)`)
    reported "no traceable tiles". aiecc runs the same sequential placer on
    the same input, so placing here does not change where the design lands.
    MLIR without logical tiles is returned untouched.
    """
    if "aie.logical_tile" not in mlir_text:
        return mlir_text
    proc = subprocess.run(
        ["aie-opt", "--aie-place-tiles", "-"],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"aie-opt tile placement failed (exit {proc.returncode}): "
            f"{proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}"
        )
    return proc.stdout


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


_PACKET_FLOW_RE = re.compile(
    r"aie\.packet_flow\s*\([^)]*\)\s*\{(?P<body>.*?)\n\s*\}", re.DOTALL)
_TRACE_SOURCE_RE = re.compile(r"aie\.packet_source<[^>]*\bTrace\s*:\s*\d+\s*>")
_PACKET_DEST_RE = re.compile(
    r"aie\.packet_dest<%\w*?_(?P<col>\d+)_(?P<row>\d+)\s*,\s*DMA\s*:\s*"
    r"(?P<ch>\d+)\s*>")


def read_trace_egress(lowered_mlir: str) -> tuple[int, int]:
    """Return (shim column, S2MM channel) that drains the trace packets.

    aie-insert-trace-flows picks the egress (it can move it off column 0 when
    the design saturates that shim), so the only truthful source is the
    lowered MLIR: the destination of every packet_flow whose source is a
    Trace port. Raises ValueError when there is none or they disagree.
    """
    egresses = set()
    for m in _PACKET_FLOW_RE.finditer(lowered_mlir):
        body = m.group("body")
        if not _TRACE_SOURCE_RE.search(body):
            continue
        for d in _PACKET_DEST_RE.finditer(body):
            if int(d.group("row")) != 0:
                raise ValueError(
                    f"trace packet_flow ends at non-shim tile row {d.group('row')}")
            egresses.add((int(d.group("col")), int(d.group("ch"))))
    if not egresses:
        raise ValueError("no trace packet_flow with a shim DMA packet_dest "
                         "in the lowered MLIR")
    if len(egresses) > 1:
        raise ValueError(
            "trace packet_flows reach more than one egress: "
            f"{sorted(egresses)}")
    return next(iter(egresses))


def apply_trace_egress(cfg: dict, lowered_mlir: str) -> dict:
    """Return cfg with routing.shim_col / shim_dma_channel read from the MLIR."""
    col, channel = read_trace_egress(lowered_mlir)
    out = dict(cfg)
    out["routing"] = {**cfg["routing"], "shim_col": col,
                      "shim_dma_channel": channel}
    return out


def _traced_devices(traced_mlir: str) -> list[tuple[str, int]]:
    """(symbol, trace buffer arg_index) of every device that declares a trace
    buffer, in source order."""
    traced = []
    for block in re.split(r"(?=aie\.device\()", traced_mlir):
        m = re.match(r"aie\.device\([^)]*\)\s*(?:@(\w+))?", block)
        t = re.search(r"#aie\.trace_buffer<arg_index = (\d+)", block)
        if m and t:
            traced.append((m.group(1) or "(unnamed)", int(t.group(1))))
    return traced


def unbound_trace_devices(traced_mlir: str, cpp_source: str) -> list[str]:
    """Devices that declare a trace buffer no kernel call of the host code binds.

    cpp_trace_patch binds a trace buffer to every kernel call it patches
    (``bo_trace``, ``bo_trace_1``, ... as the last call or set_arg argument),
    and kernel calls run the devices in order, so the first N traced devices
    are bound for N bound calls. A traced device past those names a host
    argument nothing binds: its trace BD would point at host address 0.
    Returns their symbols in source order.
    """
    traced = [name for name, _ in _traced_devices(traced_mlir)]
    bound = len(re.findall(r",\s*bo_trace(?:_\d+)?\s*\)", cpp_source))
    return traced[bound:]


def trace_arg_index_mismatch(traced_mlir: str) -> dict[str, int] | None:
    """None when every traced device declares its trace buffer at the same
    runtime-sequence argument; else {device: arg_index}. The host binds every
    call's trace buffer at the one slot trace_config.json names, so devices
    with different trace arguments would leave one of them unbound."""
    traced = _traced_devices(traced_mlir)
    if len({idx for _, idx in traced}) <= 1:
        return None
    return dict(traced)


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
    memmod_sel_locks: str | None = None,
    core_sweep_events: str | None = None,
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
            mlir_text = place_logical_tiles(mlir_text)
        except RuntimeError as e:
            msg = f"FAIL tile placement: {e}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1
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
                memmod_sel_locks=memmod_sel_locks,
                core_sweep_events=core_sweep_events,
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

        # The injector cannot know which shim the pass chose; read it back.
        try:
            trace_config = apply_trace_egress(trace_config, traced_mlir)
            trace_config_dump(trace_config, trace_config_path)
        except Exception as e:
            msg = f"FAIL trace egress: {e}"
            write_status(output_dir, msg)
            print(f"FAIL {test_name}: {msg}", file=sys.stderr)
            return 1

        # Write traced MLIR. (trace_config.json was written by the injector
        # and completed with the egress just above.)
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

        if trace_config is not None:
            mismatch = trace_arg_index_mismatch(traced_mlir)
            unbound = unbound_trace_devices(traced_mlir, cpp_source)
            if mismatch:
                msg = ("FAIL trace binding: traced devices declare their trace "
                       "buffer at different arguments: "
                       + ", ".join(f"@{d} {i}" for d, i in mismatch.items()))
            elif unbound:
                msg = ("FAIL trace binding: the host code binds no trace buffer "
                       f"for traced device(s) {', '.join('@' + d for d in unbound)}")
            if mismatch or unbound:
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
             "Default (None) leaves the memtile event selectors at reset "
             "(every DMA_*_SELn at channel 0, every LOCK_SELn at lock 0). "
             "Pass 'SLOT:CHANNEL' pairs for the DMA selectors (e.g. "
             "'S2MM_SEL1:1,MM2S_SEL1:1') and 'LOCK_SEL<n>:LOCK[@VALUE]' for "
             "the lock selectors (e.g. 'LOCK_SEL0:4'). The choice is recorded "
             "in trace_config.json.",
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
        "--memmod-sel-locks",
        default=None,
        help="forward through to mlir-trace-inject's --memmod-sel-locks: "
             "'LOCK_SEL<n>:LOCK[@VALUE]' pairs that program the compute-tile "
             "memory module's Locks_Event_Selection_<n>, so LOCK_n_REL / "
             "LOCK_SELn_* events can follow a lock other than lock 0. "
             "Needs --memmod-sweep-events.",
    )
    parser.add_argument(
        "--core-sweep-events",
        default=None,
        help="forward through to mlir-trace-inject's --core-sweep-events. "
             "Default (None) keeps the injector's hard-coded core defaults "
             "(stall + lock-request events). Core trace is always injected, "
             "so this OVERRIDES the default set rather than activating it. "
             "Pass a comma-separated event list (e.g. "
             "'PORT_RUNNING_0,PORT_RUNNING_1') to trace a compute tile's "
             "S2MM/MM2S switch-port cadence.",
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
        memmod_sel_locks=args.memmod_sel_locks,
        core_sweep_events=args.core_sweep_events,
        trace_mode=args.trace_mode,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
