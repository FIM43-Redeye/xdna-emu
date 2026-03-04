#!/usr/bin/env python3
"""Inject hardware tracing into mlir-aie npu-xrt test MLIR.

Takes a test source directory (containing aie.mlir or aie2.py) and produces
a trace-enabled MLIR variant + manifest JSON using the mlir-aie Python API.

The injection operates at the MLIR IR level -- no text manipulation of the
device/sequence structure.  The only text operation is appending the trace
buffer argument to the runtime_sequence signature after serialization, since
the Python API does not expose FuncOp argument mutation.

Usage:
    trace-inject.py <test_source_dir> --output <output_dir> \
        [--trace-size BYTES] [--device auto]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def resolve_events(names: list[str], event_enum) -> list:
    """Resolve event name strings to enum values.

    Accepts names like "TRUE", "INSTR_VECTOR", "LOCK_STALL".  Raises
    ValueError for unknown names.
    """
    result = []
    for name in names:
        name = name.strip().upper()
        if not name or name == "NONE":
            from aie.utils.trace.events import CoreEvent  # type: ignore
            result.append(event_enum(0))  # NONE = 0 for all enums
            continue
        try:
            result.append(event_enum[name])
        except KeyError:
            raise ValueError(
                f"Unknown event '{name}' for {event_enum.__name__}. "
                f"Available: {', '.join(e.name for e in event_enum)}"
            )
    return result


def load_events_config(path: Path) -> dict:
    """Load a JSON events configuration file.

    Expected format::

        {
            "core_events": ["TRUE", "INSTR_VECTOR", ...],
            "mem_events": ["TRUE", "DMA_S2MM_0_START_TASK", ...],
            "memtile_events": ["TRUE", "PORT_RUNNING_0", ...],
            "shim_events": ["TRUE", "DMA_S2MM_0_START_TASK", ...]
        }

    Each list has up to 8 entries (trace slot count).
    """
    with open(path) as f:
        return json.load(f)


def detect_source_type(test_dir: Path) -> str:
    """Detect whether this test uses raw MLIR or a Python generator.

    Returns "mlir" if aie.mlir exists, "python" if aie2.py exists.
    Raises SystemExit if neither is found.
    """
    if (test_dir / "aie.mlir").exists():
        return "mlir"
    if (test_dir / "aie2.py").exists():
        return "python"
    print(f"Error: {test_dir} has neither aie.mlir nor aie2.py", file=sys.stderr)
    sys.exit(1)


def parse_aie2_args(test_dir: Path) -> list[str]:
    """Extract aie2.py arguments from RUN directives.

    Reads the aie2.py file and parses the ``# RUN: %python %S/aie2.py <args>``
    line to determine the correct invocation arguments.  Falls back to ``npu``
    if no RUN directive is found.
    """
    aie2_path = test_dir / "aie2.py"
    if not aie2_path.exists():
        return ["npu"]

    text = aie2_path.read_text()
    for line in text.splitlines():
        m = re.match(r"#\s*RUN:\s*%python\s+%S/aie2\.py\s+(.*?)(?:\s*>\s*|$)", line)
        if m:
            raw = m.group(1).strip()
            # Strip any trailing redirections or pipes
            raw = re.sub(r"\s*[>|].*", "", raw)
            if raw:
                return raw.split()

    # No RUN directive found -- default to "npu"
    return ["npu"]


def parse_aiecc_extra_flags(test_dir: Path) -> list[str]:
    """Extract extra aiecc.py flags from RUN directives.

    Looks for flags like ``--dynamic-objFifos`` in the aiecc.py RUN line.
    Returns a list of extra flags to pass to aiecc.py during compilation.
    """
    extra = []
    # Check aie2.py and test.cpp for RUN lines
    for fname in ["aie2.py", "test.cpp"]:
        fpath = test_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text()
        for line in text.splitlines():
            if "aiecc.py" in line and "--dynamic-objFifos" in line:
                extra.append("--dynamic-objFifos")
                return extra
    return extra


def auto_detect_device(mlir_text: str) -> str:
    """Detect the required device target from tile column indices in MLIR.

    Scans for aie.tile declarations and picks the smallest device variant
    that fits all columns.
    """
    max_col = 0
    for m in re.finditer(r"aie\.tile\s*\(\s*(\d+)\s*,", mlir_text):
        col = int(m.group(1))
        max_col = max(max_col, col)

    if max_col == 0:
        return "npu1_1col"
    elif max_col == 1:
        return "npu1_2col"
    elif max_col == 2:
        return "npu1_3col"
    elif max_col == 3:
        return "npu1_4col"
    else:
        return "npu1"


def get_mlir_text(test_dir: Path, source_type: str, device: str) -> str:
    """Get MLIR text from either a static file or a Python generator.

    For raw MLIR: reads aie.mlir and substitutes the NPUDEVICE placeholder.
    For Python: runs ``python aie2.py <args>`` with arguments parsed from
    the test's RUN directives.
    """
    if source_type == "mlir":
        text = (test_dir / "aie.mlir").read_text()
        # Auto-detect device if needed, then substitute placeholder
        if device == "auto":
            device = auto_detect_device(text)
        text = text.replace("NPUDEVICE", device)
        return text

    # Python generator -- parse args from RUN directives
    aie2_args = parse_aie2_args(test_dir)

    result = subprocess.run(
        [sys.executable, str(test_dir / "aie2.py")] + aie2_args,
        capture_output=True,
        text=True,
        cwd=str(test_dir),
        timeout=60,
    )
    if result.returncode != 0:
        print(
            f"Error running aie2.py {' '.join(aie2_args)}:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout


def has_existing_trace(mlir_text: str) -> bool:
    """Check whether the MLIR already contains trace packet_flow ops.

    Looks for actual trace source ports (e.g. ``Trace : 0``), not just the
    word "Trace" which can appear in comments or unrelated context.
    """
    return "aie.packet_source" in mlir_text and "Trace :" in mlir_text


def inject_trace(
    mlir_text: str,
    trace_size: int,
    events_config: dict | None = None,
) -> tuple[str, dict]:
    """Inject trace configuration into parsed MLIR and return modified text.

    Uses the mlir-aie Python API to:
    1. Parse the MLIR into a live Module
    2. Locate device, tile, and runtime_sequence ops
    3. Insert packet flow routing for trace data
    4. Insert trace register configuration at sequence start
    5. Insert trace done/flush after the last DMA wait
    6. Serialize back to text
    7. Append trace buffer argument via regex on the canonical form

    Returns (modified_mlir_text, manifest_partial) where manifest_partial
    contains trace metadata (tile list, ddr_id, trace_size).
    """
    # Late imports -- only needed when actually injecting
    from aie.ir import Context, Location, Module, InsertionPoint  # type: ignore
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    import aie.dialects.aiex as aiexdialect  # type: ignore
    from aie.utils.trace.setup import (  # type: ignore
        configure_packet_tracing_flow,
        configure_packet_tracing_aie2,
        gen_trace_done_aie2,
    )

    with Context(), Location.unknown():
        module = Module.parse(mlir_text)

        # -- Locate key operations ------------------------------------------

        # DeviceOp: the top-level aie.device container
        device_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.opview, aiedialect.DeviceOp),
        )
        if not device_ops:
            raise RuntimeError("No aie.device op found in MLIR")

        # For multi-device modules, inject into the first device only.
        device_op = device_ops[0].opview
        device_block = device_op.body_region.blocks[0]

        # Scope all subsequent searches to THIS device (not the whole module)
        # to avoid cross-region SSA references in multi-device MLIR.
        search_root = device_op.operation

        # TileOps: tile declarations within this device
        tile_ops = find_ops(
            search_root,
            lambda o: isinstance(o.opview, aiedialect.TileOp),
        )

        # Classify tiles: one shim as trace destination, everything else traced.
        # Only trace tiles that are actually USED (have SSA references from
        # other ops).  Tests like matrix_transpose declare 24 tiles but only
        # use 2 -- tracing all of them overwhelms the router.
        shim_tile = None
        tiles_to_trace = []
        for t in tile_ops:
            top = t.opview
            col = top.col.value
            row = top.row.value
            if row == 0 and shim_tile is None:
                shim_tile = top  # First shim is the trace destination
                continue
            # Skip tiles with no uses (declared but not connected to anything)
            if len(list(top.result.uses)) == 0:
                continue
            tiles_to_trace.append(top)

        if shim_tile is None:
            raise RuntimeError("No shim tile (row 0) found")
        if not tiles_to_trace:
            raise RuntimeError("No used tiles to trace (all non-shim tiles are unused)")

        # RuntimeSequenceOp: the host instruction sequence (within this device)
        seq_ops = find_ops(
            search_root,
            lambda o: isinstance(o.opview, aiedialect.RuntimeSequenceOp),
        )
        if not seq_ops:
            raise RuntimeError("No aie.runtime_sequence op found")
        seq_op = seq_ops[0].opview
        seq_block = seq_op.body.blocks[0]

        # Count existing runtime_sequence arguments to know the trace DDR ID
        num_args = len(seq_block.arguments)
        trace_ddr_id = num_args

        # -- Insert trace packet flows at device level ----------------------

        with InsertionPoint.at_block_terminator(device_block):
            configure_packet_tracing_flow(tiles_to_trace, shim_tile)

        # -- Insert trace config at sequence start --------------------------

        # Build optional custom event kwargs from events_config.
        custom_event_kwargs: dict = {}
        if events_config:
            from aie.utils.trace.events import (  # type: ignore
                CoreEvent, MemEvent, MemTileEvent, ShimTileEvent,
            )
            if "core_events" in events_config:
                custom_event_kwargs["coretile_events"] = resolve_events(
                    events_config["core_events"], CoreEvent,
                )
            if "mem_events" in events_config:
                custom_event_kwargs["coremem_events"] = resolve_events(
                    events_config["mem_events"], MemEvent,
                )
            if "memtile_events" in events_config:
                custom_event_kwargs["memtile_events"] = resolve_events(
                    events_config["memtile_events"], MemTileEvent,
                )
            if "shim_events" in events_config:
                custom_event_kwargs["shimtile_events"] = resolve_events(
                    events_config["shim_events"], ShimTileEvent,
                )

        with InsertionPoint.at_block_begin(seq_block):
            configure_packet_tracing_aie2(
                tiles_to_trace,
                shim_tile,
                trace_size,
                ddr_id=trace_ddr_id,
                **custom_event_kwargs,
            )

        # -- Insert trace done after the last DMA wait ----------------------

        # Collect all wait/sync op types
        wait_op_types = [aiexdialect.NpuDmaWaitOp]
        if hasattr(aiexdialect, "DMAAwaitTaskOp"):
            wait_op_types.append(aiexdialect.DMAAwaitTaskOp)
        if hasattr(aiexdialect, "NpuSyncOp"):
            wait_op_types.append(aiexdialect.NpuSyncOp)

        wait_ops = find_ops(
            seq_op.operation,
            lambda o: isinstance(o.opview, tuple(wait_op_types)),
        )
        if wait_ops:
            last_wait = wait_ops[-1]
            with InsertionPoint.after(last_wait):
                gen_trace_done_aie2(shim_tile)
        else:
            # No waits found -- insert before the last operation in the block
            # (which may be an implicit terminator or the last instruction).
            ops = list(seq_block.operations)
            if ops:
                with InsertionPoint(ops[-1]):
                    gen_trace_done_aie2(shim_tile)
            else:
                with InsertionPoint.at_block_begin(seq_block):
                    gen_trace_done_aie2(shim_tile)

        # -- Serialize -------------------------------------------------------

        text = str(module)

    # -- Add trace buffer argument via regex on canonical form ---------------
    # The runtime_sequence signature in canonical MLIR looks like:
    #   aie.runtime_sequence(%arg0: memref<...>, %arg1: memref<...>) {
    # We append a trace buffer argument.
    trace_arg_name = f"%trace_buf"
    trace_words = trace_size // 4
    trace_memref = f"memref<{trace_words}xi32>"

    # Match the closing paren of the runtime_sequence arg list
    text = re.sub(
        r"(aie\.runtime_sequence\([^)]*)\)",
        rf"\1, {trace_arg_name}: {trace_memref})",
        text,
        count=1,
    )

    # Build manifest partial with per-tile type classification
    tiles_traced = []
    for tile in tiles_to_trace:
        row = tile.row.value
        if row == 0:
            tile_type = "shim"
            events = "default_shim_8"
        elif row == 1:
            tile_type = "memtile"
            events = "default_memtile_8"
        else:
            tile_type = "core"
            events = "default_core_8"
        tiles_traced.append({
            "col": tile.col.value,
            "row": row,
            "tile_type": tile_type,
            "events": events,
        })

    manifest_partial = {
        "trace_size": trace_size,
        "trace_ddr_id": trace_ddr_id,
        "tiles_traced": tiles_traced,
    }

    return text, manifest_partial


def parse_buffers_from_runtime_sequence(mlir_text: str) -> list[dict]:
    """Extract buffer specs from the runtime_sequence arguments.

    Parses memref types from the function signature to determine buffer
    count, element count, and element type.  Direction is inferred: the
    last non-trace buffer is output, all others are input.
    """
    buffers = []

    # Match the runtime_sequence arg list
    m = re.search(
        r"aie\.runtime_sequence\(([^)]+)\)",
        mlir_text,
    )
    if not m:
        return buffers

    arg_text = m.group(1)
    # Split on comma, parse each arg
    for i, arg in enumerate(arg_text.split(",")):
        arg = arg.strip()
        # Skip trace buffer (added by us)
        if "trace_buf" in arg:
            continue

        # Parse memref type: memref<NxTYPE>
        memref_match = re.search(r"memref<(\d+)x(\w+)>", arg)
        if not memref_match:
            continue

        count = int(memref_match.group(1))
        dtype = memref_match.group(2)

        # Map MLIR types to numpy-style names
        dtype_map = {
            "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
            "f16": "float16", "f32": "float32", "bf16": "bfloat16",
            "ui8": "uint8", "ui16": "uint16", "ui32": "uint32",
        }
        np_dtype = dtype_map.get(dtype, dtype)

        # Byte size per element
        size_map = {
            "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
            "int32": 4, "uint32": 4, "int64": 8,
            "float16": 2, "bfloat16": 2, "float32": 4,
        }
        elem_size = size_map.get(np_dtype, 4)

        buffers.append({
            "name": f"arg{i}",
            "size_bytes": count * elem_size,
            "dtype": np_dtype,
            "elements": count,
            "direction": "unknown",  # resolved below
        })

    # Heuristic: last buffer is output, rest are input
    if buffers:
        for b in buffers:
            b["direction"] = "input"
        buffers[-1]["direction"] = "output"

    return buffers


def build_manifest(
    test_name: str,
    test_dir: Path,
    output_dir: Path,
    mlir_text: str,
    manifest_partial: dict,
) -> dict:
    """Build a complete manifest JSON for trace-run.py."""
    buffers = parse_buffers_from_runtime_sequence(mlir_text)

    manifest = {
        "test_name": test_name,
        "xclbin": "aie.xclbin",
        "insts": "insts.bin",
        "kernel_name": "MLIR_AIE",
        "source_mlir": "aie_traced.mlir",
        "buffers": buffers,
        **manifest_partial,
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Inject hardware tracing into mlir-aie npu-xrt tests",
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
        help="Output directory for traced MLIR and manifest",
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
        "--events-json",
        type=Path,
        default=None,
        help="JSON file with custom event slot configuration (see load_events_config)",
    )
    args = parser.parse_args()

    test_dir = args.test_dir.resolve()
    output_dir = args.output.resolve()

    if not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Detect source type and get MLIR
    source_type = detect_source_type(test_dir)
    mlir_text = get_mlir_text(test_dir, source_type, args.device)

    # Check for already-traced MLIR
    if has_existing_trace(mlir_text):
        print(f"Skipping {test_dir.name}: already has trace configuration")
        # Still write a manifest so the runner knows to skip
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "test_name": test_dir.name,
            "skipped": True,
            "reason": "already_traced",
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        sys.exit(0)

    # Load custom events config if provided
    events_config = None
    if args.events_json:
        events_config = load_events_config(args.events_json)

    # Inject trace
    try:
        traced_mlir, manifest_partial = inject_trace(
            mlir_text, args.trace_size, events_config,
        )
    except Exception as e:
        print(f"Error injecting trace: {e}", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "aie_traced.mlir").write_text(traced_mlir)

    # Also write extra aiecc flags if any were detected
    extra_flags = parse_aiecc_extra_flags(test_dir)
    if extra_flags:
        (output_dir / ".aiecc-extra-flags").write_text("\n".join(extra_flags) + "\n")

    test_name = test_dir.name
    manifest = build_manifest(
        test_name, test_dir, output_dir, traced_mlir, manifest_partial,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"Injected trace for {test_name}:")
    print(f"  MLIR:     {output_dir / 'aie_traced.mlir'}")
    print(f"  Manifest: {output_dir / 'manifest.json'}")
    tile_counts = {}
    for t in manifest_partial["tiles_traced"]:
        tile_counts[t["tile_type"]] = tile_counts.get(t["tile_type"], 0) + 1
    tile_summary = ", ".join(f"{v} {k}" for k, v in sorted(tile_counts.items()))
    print(f"  Tiles:    {tile_summary}")
    print(f"  DDR ID:   {manifest_partial['trace_ddr_id']}")
    print(f"  Size:     {args.trace_size} bytes")
    if extra_flags:
        print(f"  Extra:    {' '.join(extra_flags)}")


if __name__ == "__main__":
    main()
