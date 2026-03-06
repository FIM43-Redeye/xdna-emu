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
from dataclasses import dataclass
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


def find_used_packet_ids(mlir_text: str) -> set[int]:
    """Scan MLIR text for all packet IDs already in use.

    Finds IDs from:
    - aie.packet_flow(0xN) declarations
    - pkt_id = N in controller_id attributes and dma_memcpy_nd packets
    - aie.packetflow arguments (decimal)

    Returns a set of integer packet IDs.
    """
    ids: set[int] = set()

    # aie.packet_flow(0x5) or aie.packet_flow(5)
    for m in re.finditer(r"aie\.packet_flow\s*\(\s*(0x[0-9a-fA-F]+|\d+)\s*\)", mlir_text):
        ids.add(int(m.group(1), 0))

    # pkt_id = 5 (in controller_id or packet attributes)
    for m in re.finditer(r"pkt_id\s*=\s*(\d+)", mlir_text):
        ids.add(int(m.group(1)))

    return ids


def choose_trace_id_start(used_ids: set[int], num_trace_ids: int) -> int:
    """Choose a starting packet ID for trace flows that avoids collisions.

    The stream switch uses 5-bit mask/value rules for packet routing.
    Rules of the form ``(pkt_id & mask) == value`` can falsely match
    IDs with similar bit patterns.  To avoid false matches, we place
    trace IDs in the highest available range (top of the 5-bit space),
    maximizing bit distance from existing low-numbered IDs.

    Falls back to the lowest gap if the top range is occupied.
    """
    max_id = 31  # 5-bit packet ID space

    if num_trace_ids > max_id + 1:
        # More trace IDs than the 5-bit space allows -- caller should
        # reduce trace targets.  Return 1 and let the pathfinder fail
        # with a clear error.
        return 1

    # Strategy: maximize bit distance from existing IDs.
    # Try placing trace IDs at the top first (far from typical low IDs),
    # then at the bottom (far from typical high IDs).  Pick whichever
    # has greater minimum bit distance from all used IDs.
    candidates = []

    # Top of range
    top_start = max_id - num_trace_ids + 1
    if top_start >= 0 and not set(range(top_start, max_id + 1)) & used_ids:
        candidates.append(top_start)

    # Bottom of range (starting from 1 to avoid ID 0)
    bot_start = 1
    if not set(range(bot_start, bot_start + num_trace_ids)) & used_ids:
        candidates.append(bot_start)

    if not candidates:
        # Both ends overlap -- find any contiguous gap.
        all_ids = sorted(used_ids | {-1, max_id + 1})
        for i in range(len(all_ids) - 1):
            gap_start = all_ids[i] + 1
            gap_end = all_ids[i + 1]
            if gap_end - gap_start >= num_trace_ids:
                candidates.append(gap_start)
                break
        if not candidates:
            return 1  # fallback, let pathfinder report the error

    # Score each candidate by minimum XOR distance to any used ID.
    # Higher minimum distance = better bit separation for mask/value rules.
    def min_bit_distance(start):
        trace_range = range(start, start + num_trace_ids)
        min_dist = 32
        for tid in trace_range:
            for uid in used_ids:
                # Count differing bits
                dist = bin(tid ^ uid).count("1")
                min_dist = min(min_dist, dist)
        return min_dist

    return max(candidates, key=min_bit_distance)



# ---------------------------------------------------------------------------
# Pathfinder-driven trace planner
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Pathfinder result for one (shim_col, channel) candidate."""
    shim_col: int
    channel: int
    success: bool                      # pathfinder routed trace successfully
    failure_reason: str | None = None
    # Tiebreaker metrics (set when success=True)
    trace_connections_on_test_cols: int = 0
    total_trace_connections: int = 0


@dataclass
class TracePlan:
    """Result of trace route planning."""
    feasible: bool
    reason: str
    shim_col: int | None = None
    trace_channel: int | None = None
    candidates: list[CandidateResult] | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON manifest output."""
        d = {
            "feasible": self.feasible,
            "reason": self.reason,
            "shim_col": self.shim_col,
            "trace_channel": self.trace_channel,
        }
        if self.candidates:
            d["candidates"] = [
                {
                    "shim_col": c.shim_col,
                    "channel": c.channel,
                    "success": c.success,
                    "failure_reason": c.failure_reason,
                    "trace_on_test_cols": c.trace_connections_on_test_cols,
                    "total_trace_connections": c.total_trace_connections,
                }
                for c in self.candidates
            ]
        return d


# ---------------------------------------------------------------------------
# IR-walking helpers -- shared by planner and injector
# ---------------------------------------------------------------------------

def find_device_with_sequence(module):
    """Find the DeviceOp containing a runtime_sequence.

    Multi-device modules (e.g. ctrl_packet_reconfig) have @base (empty)
    and @main (full design).  Returns the device with runtime_sequence,
    or the first device if none has one.
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    device_ops = find_ops(
        module.operation,
        lambda o: isinstance(o.opview, aiedialect.DeviceOp),
    )
    if not device_ops:
        raise RuntimeError("No aie.device op found in MLIR")

    for dop in device_ops:
        rs = find_ops(
            dop,
            lambda o: isinstance(o.opview, aiedialect.RuntimeSequenceOp),
        )
        if rs:
            return dop.opview
    return device_ops[0].opview


def find_tile_op(device_op, col: int, row: int):
    """Find an existing TileOp for (col, row) within a device, or None."""
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    tile_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.TileOp),
    )
    for t in tile_ops:
        top = t.opview
        if top.col.value == col and top.row.value == row:
            return top
    return None


def find_used_packet_ids_ir(device_op) -> set[int]:
    """Scan a parsed device op for all packet IDs in use.

    Walks PacketFlowOp operations in the IR instead of regex on text.
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    ids: set[int] = set()
    pkt_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.PacketFlowOp),
    )
    for pfop in pkt_ops:
        ids.add(pfop.opview.ID.value)
    return ids


def classify_tiles(device_op) -> tuple[list[int], list[tuple], set[int]]:
    """Classify tiles in a device into shim columns, trace targets, and test columns.

    Returns:
        shim_cols: list of column indices with shim tiles (row 0)
        trace_targets: list of (col, row, trace_port) tuples for traceable tiles
            Core tiles produce two entries: (col, row, 0) for Core, (col, row, 1) for Mem
        test_cols: set of column indices used by the test (non-shim tiles with uses)
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    from aie.dialects.aie import get_target_model as _gtm  # type: ignore

    tm = _gtm(device_op.operation.attributes["device"])

    tile_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.TileOp),
    )

    shim_cols = []
    trace_targets = []
    test_cols = set()

    for t in tile_ops:
        top = t.opview
        col, row = top.col.value, top.row.value
        if row == 0:
            if col not in shim_cols:
                shim_cols.append(col)
            continue
        if len(list(top.result.uses)) == 0:
            continue
        test_cols.add(col)
        if tm.is_core_tile(col, row):
            trace_targets.append((col, row, 0))  # Core trace port
            trace_targets.append((col, row, 1))  # Mem trace port
        elif tm.is_mem_tile(col, row):
            trace_targets.append((col, row, 0))  # MemTile trace port

    return shim_cols, trace_targets, test_cols


# ---------------------------------------------------------------------------
# Shared flow helper -- declares trace packet flows on a parsed module
# ---------------------------------------------------------------------------

def add_trace_flows(
    device_op,
    shim_col: int,
    trace_channel: int,
    trace_targets: list[tuple[int, int, int]],
    used_packet_ids: set[int],
) -> tuple[int, object]:
    """Declare trace packet flows from tiles to a target shim.

    Injects:
    - aie.tile declaration for the shim (if not already present)
    - aie.packet_flow ops routing each tile's Trace ports to the shim DMA

    Does NOT configure trace registers, timers, DMA, or buffers.

    Args:
        device_op: the DeviceOp to modify
        shim_col: target shim tile column
        trace_channel: S2MM DMA channel on the shim
        trace_targets: list of (col, row, trace_port) -- explicit, no guessing
        used_packet_ids: existing packet IDs to avoid

    Returns:
        (trace_id_start, shim_tile_op) -- first packet ID used and the shim TileOp
    """
    from aie.ir import InsertionPoint  # type: ignore
    from aie.dialects.aie import packetflow, tile, WireBundle  # type: ignore

    device_block = device_op.body_region.blocks[0]

    # Find or create shim tile
    shim_tile = find_tile_op(device_op, shim_col, 0)
    if shim_tile is None:
        with InsertionPoint.at_block_begin(device_block):
            shim_tile = tile(shim_col, 0)

    # Allocate packet IDs
    trace_id_start = choose_trace_id_start(used_packet_ids, len(trace_targets))

    # Build a lookup from (col, row) to existing TileOp
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    tile_ops = find_ops(
        device_op.operation,
        lambda o: isinstance(o.opview, aiedialect.TileOp),
    )
    tile_lookup = {}
    for t in tile_ops:
        top = t.opview
        tile_lookup[(top.col.value, top.row.value)] = top

    # Declare packet flows
    with InsertionPoint.at_block_terminator(device_block):
        for i, (col, row, trace_port) in enumerate(trace_targets):
            p_id = trace_id_start + i
            src_tile = tile_lookup.get((col, row))
            if src_tile is None:
                raise RuntimeError(
                    f"Tile ({col},{row}) referenced in trace_targets "
                    f"but not found in MLIR"
                )
            packetflow(
                p_id, src_tile, WireBundle.Trace, trace_port,
                dests={"dest": shim_tile, "port": WireBundle.DMA,
                       "channel": trace_channel},
                keep_pkt_header=True,
            )

    return trace_id_start, shim_tile


# ---------------------------------------------------------------------------
# Pathfinder runner and connection extractor
# ---------------------------------------------------------------------------

def run_pathfinder(module) -> bool:
    """Run the pathfinder pass on a module. Returns True if successful.

    Mutates the module in place: FlowOps and PacketFlowOps are replaced
    with concrete SwitchboxOp/ConnectOp routing.
    """
    from aie.passmanager import PassManager  # type: ignore

    pipeline = "builtin.module(aie.device(aie-create-pathfinder-flows))"
    try:
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        return True
    except Exception:
        return False


def extract_connections(module) -> set[tuple]:
    """Extract all switch connections from a routed module.

    After the pathfinder runs, the module contains SwitchboxOp with
    ConnectOp (circuit-switched) and PacketRulesOp/MasterSetOp
    (packet-switched) inside them.

    Each connection is extracted as a tuple:
        ("circuit", col, row, src_bundle, src_ch, dst_bundle, dst_ch)
        ("packet",  col, row, src_bundle, src_ch)     -- PacketRulesOp
        ("master",  col, row, dst_bundle, dst_ch)      -- MasterSetOp

    Returns a set for easy comparison via set subtraction.
    """
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore

    connections = set()

    switchbox_ops = find_ops(
        module.operation,
        lambda o: isinstance(o.opview, aiedialect.SwitchboxOp),
    )

    for sb_op in switchbox_ops:
        sb = sb_op.opview
        tile_op = sb.tile.owner.opview
        col = tile_op.col.value
        row = tile_op.row.value

        for child in sb_op.regions[0].blocks[0]:
            op = child.opview
            if isinstance(op, aiedialect.ConnectOp):
                connections.add((
                    "circuit", col, row,
                    int(op.source_bundle), int(op.source_channel),
                    int(op.dest_bundle), int(op.dest_channel),
                ))
            elif isinstance(op, aiedialect.PacketRulesOp):
                connections.add((
                    "packet", col, row,
                    int(op.source_bundle), int(op.source_channel),
                ))
            elif isinstance(op, aiedialect.MasterSetOp):
                connections.add((
                    "master", col, row,
                    int(op.dest_bundle), int(op.dest_channel),
                ))

    return connections


# ---------------------------------------------------------------------------
# Pathfinder-driven trace route planner
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    mlir_asm: str,
    shim_col: int,
    channel: int,
    trace_targets: list[tuple[int, int, int]],
    test_cols: set[int],
    data_conn_count: int,
) -> dict:
    """Evaluate one (shim_col, channel) candidate in a subprocess.

    Uses a two-pass pathfinder approach:
    1. Route the test's data flows first (FlowOp/PacketFlowOp -> SwitchboxOp)
    2. Add trace PacketFlowOps to the already-routed module
    3. Run pathfinder again -- it calls addFixedConnection() on existing
       SwitchboxOps, marking those ports as INVALID, then routes trace
       using only the leftover switch fabric.

    This guarantees zero interference: the trace router literally cannot
    touch ports used by existing data flows.

    Returns a plain dict (not CandidateResult) for pickling across the
    process boundary.
    """
    result = {
        "shim_col": shim_col,
        "channel": channel,
        "success": False,
        "failure_reason": None,
        "trace_connections_on_test_cols": 0,
        "total_trace_connections": 0,
    }
    try:
        from aie.ir import Context, Location, Module  # type: ignore

        with Context(), Location.unknown():
            module = Module.parse(mlir_asm)

            # Pass 1: route the test's own data flows.
            # This converts FlowOp/PacketFlowOp into SwitchboxOp/ConnectOp.
            if not run_pathfinder(module):
                result["failure_reason"] = "pathfinder failed on data flows"
                return result

            # Count data connections for scoring later
            data_conns = extract_connections(module)

            # Pass 2: add trace flows and route them on the locked fabric.
            # The existing SwitchboxOps from pass 1 will be picked up by
            # addFixedConnection() -- their ports become INVALID to the
            # trace router.
            device_op = find_device_with_sequence(module)
            used_ids = find_used_packet_ids_ir(device_op)

            add_trace_flows(
                device_op, shim_col, channel, trace_targets, used_ids,
            )

            if not run_pathfinder(module):
                result["failure_reason"] = "no route for trace on remaining fabric"
                return result

            # Score: connections added by trace routing
            all_conns = extract_connections(module)
            trace_conns = all_conns - data_conns
            on_test = sum(1 for c in trace_conns if c[1] in test_cols)
            result["trace_connections_on_test_cols"] = on_test
            result["total_trace_connections"] = len(trace_conns)

        result["success"] = True

    except Exception as exc:
        result["failure_reason"] = str(exc).splitlines()[0][:200]

    return result


def plan_trace_route(mlir_text: str) -> TracePlan:
    """Plan the best (shim_col, channel) for trace collection.

    Evaluates all (shim_col, channel) candidates in parallel.  Each worker
    uses a two-pass pathfinder: first routes the test's data flows (locking
    them via addFixedConnection), then routes trace on the remaining fabric.
    If the second pass succeeds, the candidate is viable -- guaranteed
    zero interference with existing data flows.

    Returns a TracePlan with the winning candidate or an infeasibility reason.
    """
    import concurrent.futures

    from aie.ir import Context, Location, Module  # type: ignore
    from aie.dialects.aie import get_target_model as _gtm  # type: ignore

    # -- Parse and classify tiles -----------------------------------------
    with Context(), Location.unknown():
        try:
            module = Module.parse(mlir_text)
        except Exception as e:
            return TracePlan(
                False,
                f"MLIR parse error: {str(e).splitlines()[0][:120]}",
            )

        device_op = find_device_with_sequence(module)
        tm = _gtm(device_op.operation.attributes["device"])
        shim_cols_list, trace_targets, test_cols = classify_tiles(device_op)

        if not trace_targets:
            return TracePlan(False, "no traceable tiles found")

    # -- Generate candidates: all device columns x 2 channels -------------
    num_cols = tm.columns()
    candidates_params = [
        (col, ch) for col in range(num_cols) for ch in range(2)
    ]

    # -- Evaluate in parallel via ProcessPoolExecutor ---------------------
    raw_results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(
                _evaluate_candidate,
                mlir_text,
                col,
                ch,
                trace_targets,
                test_cols,
                0,  # data_conn_count (unused, reserved)
            ): (col, ch)
            for col, ch in candidates_params
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                raw_results.append(future.result())
            except Exception as exc:
                col, ch = futures[future]
                raw_results.append({
                    "shim_col": col,
                    "channel": ch,
                    "success": False,
                    "failure_reason": f"subprocess error: {exc}",
                    "trace_connections_on_test_cols": 0,
                    "total_trace_connections": 0,
                })

    # -- Convert to CandidateResult objects -------------------------------
    candidates = [
        CandidateResult(
            shim_col=r["shim_col"],
            channel=r["channel"],
            success=r["success"],
            failure_reason=r.get("failure_reason"),
            trace_connections_on_test_cols=r.get(
                "trace_connections_on_test_cols", 0,
            ),
            total_trace_connections=r.get("total_trace_connections", 0),
        )
        for r in raw_results
    ]

    # -- Pick winner: viable = pathfinder succeeded -----------------------
    viable = [c for c in candidates if c.success]

    if not viable:
        return TracePlan(
            False,
            "no candidate found a route for trace",
            candidates=candidates,
        )

    # Tiebreak: fewer trace connections on test columns, then fewer total.
    # Lower is better -- less routing pressure on the columns the test uses.
    winner = min(
        viable,
        key=lambda c: (
            c.trace_connections_on_test_cols,
            c.total_trace_connections,
        ),
    )

    return TracePlan(
        feasible=True,
        reason=(
            f"shim col {winner.shim_col} ch {winner.channel}: "
            f"{winner.total_trace_connections} trace connections "
            f"({winner.trace_connections_on_test_cols} on test cols)"
        ),
        shim_col=winner.shim_col,
        trace_channel=winner.channel,
        candidates=candidates,
    )



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
    elif max_col <= 4:
        # npu1 is the full 5-column device (cols 0-4).
        # There is no npu1_4col variant -- use full npu1 for 4+ columns.
        return "npu1"
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
    plan: "TracePlan",
    events_config: dict | None = None,
    tile_filter: list[tuple[int, int, str]] | None = None,
) -> tuple[str, dict]:
    """Inject trace configuration into parsed MLIR and return modified text.

    Uses the mlir-aie Python API to:
    1. Parse the MLIR into a live Module
    2. Locate device, tile, and runtime_sequence ops
    3. Insert packet flow routing for trace data (via add_trace_flows)
    4. Insert trace register configuration at sequence start
    5. Insert trace done/flush after the last DMA wait
    6. Serialize back to text
    7. Append trace buffer argument via regex on the canonical form

    The ``plan`` argument is a TracePlan produced by plan_trace_route(),
    which specifies the shim column and DMA channel to use.

    Returns (modified_mlir_text, manifest_partial) where manifest_partial
    contains trace metadata (tile list, ddr_id, trace_size).
    """
    # Late imports -- only needed when actually injecting
    from aie.ir import Context, Location, Module, InsertionPoint  # type: ignore
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    import aie.dialects.aiex as aiexdialect  # type: ignore
    from aie.dialects.aie import get_target_model as _gtm  # type: ignore
    from aie.utils.trace.events import (  # type: ignore
        PacketType,
    )
    from aie.utils.trace.setup import (  # type: ignore
        configure_coretile_tracing_aie2,
        configure_coremem_tracing_aie2,
        configure_memtile_tracing_aie2,
        configure_shimtile_tracing_aie2,
        configure_shimtile_dma_aie2,
        configure_shim_trace_start_aie2,
        configure_timer_ctrl_coretile_aie2,
        configure_timer_ctrl_coremem_aie2,
        configure_timer_ctrl_memtile_aie2,
        configure_timer_ctrl_shimtile_aie2,
        gen_trace_done_aie2,
    )

    # Extract routing parameters from the plan.
    shim_col = plan.shim_col
    trace_channel = plan.trace_channel

    with Context(), Location.unknown():
        module = Module.parse(mlir_text)

        # -- Locate key operations ------------------------------------------

        device_op = find_device_with_sequence(module)
        search_root = device_op.operation

        tm = _gtm(device_op.operation.attributes["device"])

        # -- Build explicit trace targets -----------------------------------
        # classify_tiles() produces (col, row, trace_port) tuples: Core tiles
        # get two entries (port 0 for Core, port 1 for Mem), others get one.

        tile_ops = find_ops(
            search_root,
            lambda o: isinstance(o.opview, aiedialect.TileOp),
        )

        # Identify used non-shim tiles (same filter as classify_tiles)
        used_tile_ops = []
        for t in tile_ops:
            top = t.opview
            col, row = top.col.value, top.row.value
            if row == 0:
                continue
            if len(list(top.result.uses)) == 0:
                continue
            used_tile_ops.append(top)

        if not used_tile_ops:
            raise RuntimeError("No used tiles to trace (all non-shim tiles are unused)")

        # Build explicit trace targets and parallel config list
        trace_targets = []
        tiles_for_config = []  # (TileOp, trace_port) for register config

        for t_op in used_tile_ops:
            col, row = t_op.col.value, t_op.row.value
            if tm.is_core_tile(col, row):
                trace_targets.append((col, row, 0))
                tiles_for_config.append((t_op, 0))
                trace_targets.append((col, row, 1))
                tiles_for_config.append((t_op, 1))
            elif tm.is_mem_tile(col, row):
                trace_targets.append((col, row, 0))
                tiles_for_config.append((t_op, 0))

        # Apply tile filter if provided (from --tiles CLI flag).
        # Filter trace_targets and tiles_for_config in parallel.
        if tile_filter is not None:
            # Map filter module names to trace ports
            module_to_port = {"core": 0, "mem": 1, "memtile": 0}
            filter_set = set()
            for col, row, mod in tile_filter:
                port = module_to_port.get(mod, 0)
                filter_set.add((col, row, port))
            filtered_targets = []
            filtered_config = []
            for target, config in zip(trace_targets, tiles_for_config):
                if target in filter_set:
                    filtered_targets.append(target)
                    filtered_config.append(config)
            trace_targets = filtered_targets
            tiles_for_config = filtered_config
            if not trace_targets:
                raise RuntimeError("No valid tiles after applying --tiles filter")

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

        used_ids = find_used_packet_ids_ir(device_op)
        trace_id_start, shim_tile = add_trace_flows(
            device_op, shim_col, trace_channel, trace_targets, used_ids,
        )

        # -- Insert trace config at sequence start --------------------------
        #
        # Configure each tile's trace unit with events and packet IDs that
        # match the packet flows created by add_trace_flows() above.

        from aie.utils.trace.events import (  # type: ignore
            CoreEvent, MemEvent, MemTileEvent, ShimTileEvent,
            PortEvent, MemTilePortEvent,
        )

        # Resolve custom events from events_config, or use defaults.
        coretile_events = [
            CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1,
            CoreEvent.INSTR_VECTOR,
            PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),
            PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),
            CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
            CoreEvent.INSTR_LOCK_RELEASE_REQ, CoreEvent.LOCK_STALL,
        ]
        coremem_events = [
            MemEvent.DMA_S2MM_0_START_TASK, MemEvent.DMA_MM2S_0_START_TASK,
            MemEvent.CONFLICT_DM_BANK_0, MemEvent.CONFLICT_DM_BANK_1,
            MemEvent.CONFLICT_DM_BANK_2, MemEvent.CONFLICT_DM_BANK_3,
            MemEvent.EDGE_DETECTION_EVENT_0, MemEvent.EDGE_DETECTION_EVENT_1,
        ]
        memtile_events = [
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 14, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),
        ]
        # Shim events should monitor the trace channel specifically.
        if trace_channel == 1:
            shimtile_events = [
                ShimTileEvent.DMA_S2MM_0_START_TASK,
                ShimTileEvent.DMA_S2MM_1_START_TASK,
                ShimTileEvent.DMA_MM2S_0_START_TASK,
                ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
                ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
                ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
            ]
        else:
            # Channel 0 for trace -- swap focus to S2MM_0 events
            shimtile_events = [
                ShimTileEvent.DMA_S2MM_0_START_TASK,
                ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
                ShimTileEvent.DMA_S2MM_1_START_TASK,
                ShimTileEvent.DMA_MM2S_0_START_TASK,
                ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
                ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
                ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
            ]

        if events_config:
            if "core_events" in events_config:
                coretile_events = resolve_events(
                    events_config["core_events"], CoreEvent,
                )
            if "mem_events" in events_config:
                coremem_events = resolve_events(
                    events_config["mem_events"], MemEvent,
                )
            if "memtile_events" in events_config:
                memtile_events = resolve_events(
                    events_config["memtile_events"], MemTileEvent,
                )
            if "shim_events" in events_config:
                shimtile_events = resolve_events(
                    events_config["shim_events"], ShimTileEvent,
                )

        start_broadcast_num = 15
        stop_broadcast_num = 14
        start_user_event = ShimTileEvent.USER_EVENT_1
        stop_user_event = ShimTileEvent.USER_EVENT_0
        start_core_broadcast = CoreEvent(107 + start_broadcast_num)
        stop_core_broadcast = CoreEvent(107 + stop_broadcast_num)
        start_mem_broadcast = MemEvent(107 + start_broadcast_num)
        stop_mem_broadcast = MemEvent(107 + stop_broadcast_num)
        start_memtile_broadcast = MemTileEvent(142 + start_broadcast_num)
        stop_memtile_broadcast = MemTileEvent(142 + stop_broadcast_num)

        with InsertionPoint.at_block_begin(seq_block):
            for i, (tile_op, trace_port) in enumerate(tiles_for_config):
                p_id = trace_id_start + i
                tc, tr = int(tile_op.col), int(tile_op.row)

                if tm.is_shim_noc_or_pl_tile(tc, tr):
                    start_ev = start_user_event if tile_op == shim_tile else \
                        ShimTileEvent(110 + start_broadcast_num)
                    stop_ev = stop_user_event if tile_op == shim_tile else \
                        ShimTileEvent(110 + stop_broadcast_num)
                    configure_shimtile_tracing_aie2(
                        tile=tile_op, start=start_ev, stop=stop_ev,
                        events=shimtile_events, enable_packet=1,
                        packet_id=p_id, packet_type=PacketType.SHIMTILE,
                    )
                    configure_timer_ctrl_shimtile_aie2(tile_op, start_ev)
                elif tm.is_mem_tile(tc, tr):
                    configure_memtile_tracing_aie2(
                        tile=tile_op, start=start_memtile_broadcast,
                        stop=stop_memtile_broadcast,
                        events=memtile_events, enable_packet=1,
                        packet_id=p_id, packet_type=PacketType.MEMTILE,
                    )
                    configure_timer_ctrl_memtile_aie2(
                        tile_op, start_memtile_broadcast,
                    )
                elif tm.is_core_tile(tc, tr) and trace_port == 0:
                    configure_coretile_tracing_aie2(
                        tile=tile_op, start=start_core_broadcast,
                        stop=stop_core_broadcast,
                        events=coretile_events, enable_packet=1,
                        packet_id=p_id, packet_type=PacketType.CORE,
                    )
                    configure_timer_ctrl_coretile_aie2(
                        tile_op, start_core_broadcast,
                    )
                elif tm.is_core_tile(tc, tr) and trace_port == 1:
                    configure_coremem_tracing_aie2(
                        tile=tile_op, start=start_mem_broadcast,
                        stop=stop_mem_broadcast,
                        events=coremem_events, enable_packet=1,
                        packet_id=p_id, packet_type=PacketType.MEM,
                    )
                    configure_timer_ctrl_coremem_aie2(
                        tile_op, start_mem_broadcast,
                    )

            # Configure shim DMA for trace collection
            configure_shimtile_dma_aie2(
                shim=shim_tile, channel=trace_channel, bd_id=15,
                ddr_id=trace_ddr_id,
                size=trace_size // 4,  # convert to words
                offset=0, enable_token=1,
            )

            # Start trace broadcast
            configure_shim_trace_start_aie2(
                shim_tile, start_broadcast_num, start_user_event,
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
    trace_arg_name = "%trace_buf"
    trace_words = trace_size // 4
    trace_memref = f"memref<{trace_words}xi32>"

    # Match the closing paren of the runtime_sequence arg list
    text = re.sub(
        r"(aie\.runtime_sequence\([^)]*)\)",
        rf"\1, {trace_arg_name}: {trace_memref})",
        text,
        count=1,
    )

    # Build manifest partial with per-tile type classification.
    # tiles_for_config has explicit (tile_op, trace_port) pairs, so we can
    # classify directly without the old "seen" tracking hack.
    tiles_traced = []
    for tile_op, trace_port in tiles_for_config:
        col = int(tile_op.col)
        row = int(tile_op.row)
        if row == 0:
            tile_type = "shim"
            events = "default_shim_8"
        elif tm.is_mem_tile(col, row):
            tile_type = "memtile"
            events = "default_memtile_8"
        elif trace_port == 0:
            tile_type = "core"
            events = "default_core_8"
        else:
            tile_type = "mem"
            events = "default_mem_8"
        tiles_traced.append({
            "col": col,
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


def inject_trace_per_column(
    mlir_text: str,
    trace_size: int,
    events_config: dict | None = None,
) -> tuple[str, dict]:
    """Inject per-column trace routing for higher bandwidth multi-tile tracing.

    Instead of routing all trace packets to a single shim DMA channel, this
    groups tiles by column and routes each column's trace to that column's own
    shim DMA.  This provides up to 4x bandwidth on npu1 (4 columns).

    The implementation calls the same mlir-aie primitives as
    configure_packet_tracing_aie2() but decomposes the operation to support
    multiple shim destinations:

    1. Configure all tiles' trace units (events, packet IDs, timers)
    2. Create per-column packet flows (tile Trace port -> column shim DMA)
    3. Configure each column's shim DMA with its own DDR buffer
    4. Fire start broadcast once from the first shim

    Returns (modified_mlir_text, manifest_partial).
    """
    from aie.ir import Context, Location, Module, InsertionPoint  # type: ignore
    from aie.extras.util import find_ops  # type: ignore
    import aie.dialects.aie as aiedialect  # type: ignore
    import aie.dialects.aiex as aiexdialect  # type: ignore
    from aie.dialects.aie import packetflow, WireBundle, get_target_model  # type: ignore
    from aie.utils.trace.setup import (  # type: ignore
        configure_coretile_tracing_aie2,
        configure_coremem_tracing_aie2,
        configure_memtile_tracing_aie2,
        configure_shimtile_tracing_aie2,
        configure_timer_ctrl_coretile_aie2,
        configure_timer_ctrl_coremem_aie2,
        configure_timer_ctrl_memtile_aie2,
        configure_timer_ctrl_shimtile_aie2,
        configure_shimtile_dma_aie2,
        configure_shim_trace_start_aie2,
        gen_trace_done_aie2,
    )
    from aie.utils.trace.events import (  # type: ignore
        CoreEvent, MemEvent, MemTileEvent, ShimTileEvent,
        PortEvent, MemTilePortEvent,
    )
    from aie.utils.trace.setup import PacketType  # type: ignore

    with Context(), Location.unknown():
        module = Module.parse(mlir_text)

        # -- Locate key operations (same as inject_trace) ----------------------
        # For multi-device modules, find the device with runtime_sequence.

        device_ops = find_ops(
            module.operation,
            lambda o: isinstance(o.opview, aiedialect.DeviceOp),
        )
        if not device_ops:
            raise RuntimeError("No aie.device op found in MLIR")

        device_op = None
        for dop in device_ops:
            rs = find_ops(
                dop,
                lambda o: isinstance(o.opview, aiedialect.RuntimeSequenceOp),
            )
            if rs:
                device_op = dop.opview
                break
        if device_op is None:
            device_op = device_ops[0].opview
        device_block = device_op.body_region.blocks[0]
        search_root = device_op.operation

        tile_ops = find_ops(
            search_root,
            lambda o: isinstance(o.opview, aiedialect.TileOp),
        )

        # Collect all shim tiles (row 0) and all used non-shim tiles.
        shim_tiles = {}  # col -> TileOp
        tiles_to_trace = []
        for t in tile_ops:
            top = t.opview
            col = top.col.value
            row = top.row.value
            if row == 0:
                shim_tiles[col] = top
                continue
            if len(list(top.result.uses)) == 0:
                continue
            tiles_to_trace.append(top)

        if not shim_tiles:
            raise RuntimeError("No shim tile (row 0) found")
        if not tiles_to_trace:
            raise RuntimeError("No used tiles to trace")

        # Duplicate core tiles so both Core (Trace port 0) and Mem module
        # (Trace port 1) get traced.  The routing and config loops use a
        # "first visit = Core, second visit = Mem" pattern, so each core
        # tile must appear twice.  MemTiles and ShimTiles have only one
        # trace port and should not be duplicated.
        tm = get_target_model(device_block.owner.attributes["device"])
        expanded = []
        for tile in tiles_to_trace:
            expanded.append(tile)
            tc, tr = tile.col.value, tile.row.value
            if tm.is_core_tile(tc, tr):
                expanded.append(tile)  # second entry for Mem module
        tiles_to_trace = expanded

        # Group tiles by column
        tiles_by_col: dict[int, list] = {}
        for tile in tiles_to_trace:
            col = tile.col.value
            tiles_by_col.setdefault(col, []).append(tile)

        # Each active column needs a shim.  If a column has tiles but no shim
        # tile declared, route to the nearest column that has one.
        active_cols = sorted(tiles_by_col.keys())
        col_to_shim: dict[int, object] = {}
        for col in active_cols:
            if col in shim_tiles:
                col_to_shim[col] = shim_tiles[col]
            else:
                # Find nearest shim tile
                nearest = min(shim_tiles.keys(), key=lambda c: abs(c - col))
                col_to_shim[col] = shim_tiles[nearest]

        # Deduplicate: group columns that share a shim
        shim_to_cols: dict[int, list[int]] = {}
        for col in active_cols:
            shim_col = col_to_shim[col].col.value
            shim_to_cols.setdefault(shim_col, []).append(col)

        # The primary shim (for start/stop broadcast) is the first one
        primary_shim = col_to_shim[active_cols[0]]

        seq_ops = find_ops(
            search_root,
            lambda o: isinstance(o.opview, aiedialect.RuntimeSequenceOp),
        )
        if not seq_ops:
            raise RuntimeError("No aie.runtime_sequence op found")
        seq_op = seq_ops[0].opview
        seq_block = seq_op.body.blocks[0]

        num_args = len(seq_block.arguments)

        # -- Insert per-column packet flows at device level --------------------

        # Scan for existing packet IDs to avoid collisions.  Even though
        # per-column packets route to different shim DMAs, pathfinder still
        # validates globally and shared switchbox paths can cause false
        # packet ID matches.
        used_ids = find_used_packet_ids(mlir_text)
        total_trace = sum(len(tiles_by_col[c]) for c in active_cols)
        trace_id_start = choose_trace_id_start(used_ids, total_trace)
        max_trace_id = trace_id_start + total_trace - 1
        if used_ids:
            print(f"  Existing packet IDs: {sorted(used_ids)}, "
                  f"trace IDs: {trace_id_start}-{max_trace_id}",
                  file=sys.stderr)

        with InsertionPoint.at_block_terminator(device_block):
            for col in active_cols:
                shim = col_to_shim[col]
                col_tiles = tiles_by_col[col]
                seen = []
                for i, tile in enumerate(col_tiles):
                    p_id = trace_id_start + i
                    if tile not in seen:
                        packetflow(
                            p_id, tile, WireBundle.Trace, 0,
                            dests={"dest": shim, "port": WireBundle.DMA, "channel": 1},
                            keep_pkt_header=True,
                        )
                        seen.append(tile)
                    else:
                        # Second trace port (memory module) for same tile
                        packetflow(
                            p_id, tile, WireBundle.Trace, 1,
                            dests={"dest": shim, "port": WireBundle.DMA, "channel": 1},
                            keep_pkt_header=True,
                        )

        # -- Insert trace config at sequence start -----------------------------

        # Resolve event kwargs (same logic as inject_trace).
        custom_core = None
        custom_mem = None
        custom_memtile = None
        custom_shim = None
        if events_config:
            if "core_events" in events_config:
                custom_core = resolve_events(events_config["core_events"], CoreEvent)
            if "mem_events" in events_config:
                custom_mem = resolve_events(events_config["mem_events"], MemEvent)
            if "memtile_events" in events_config:
                custom_memtile = resolve_events(
                    events_config["memtile_events"], MemTileEvent,
                )
            if "shim_events" in events_config:
                custom_shim = resolve_events(
                    events_config["shim_events"], ShimTileEvent,
                )

        # Default events (match configure_packet_tracing_aie2 defaults)
        default_core = [
            CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1,
            CoreEvent.INSTR_VECTOR,
            PortEvent(CoreEvent.PORT_RUNNING_0, 1, True),
            PortEvent(CoreEvent.PORT_RUNNING_1, 1, False),
            CoreEvent.INSTR_LOCK_ACQUIRE_REQ,
            CoreEvent.INSTR_LOCK_RELEASE_REQ,
            CoreEvent.LOCK_STALL,
        ]
        default_mem = [
            MemEvent.DMA_S2MM_0_START_TASK, MemEvent.DMA_MM2S_0_START_TASK,
            MemEvent.CONFLICT_DM_BANK_0, MemEvent.CONFLICT_DM_BANK_1,
            MemEvent.CONFLICT_DM_BANK_2, MemEvent.CONFLICT_DM_BANK_3,
            MemEvent.EDGE_DETECTION_EVENT_0, MemEvent.EDGE_DETECTION_EVENT_1,
        ]
        default_memtile = [
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, 0, True),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, 14, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, 0, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, 1, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, 2, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, 3, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, 4, False),
            MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, 5, False),
        ]
        default_shim = [
            ShimTileEvent.DMA_S2MM_0_START_TASK,
            ShimTileEvent.DMA_S2MM_1_START_TASK,
            ShimTileEvent.DMA_MM2S_0_START_TASK,
            ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
            ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
            ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
            ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
            ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
        ]

        coretile_events = custom_core or default_core
        coremem_events = custom_mem or default_mem
        memtile_events = custom_memtile or default_memtile
        shimtile_events = custom_shim or default_shim

        # Broadcast event derivation (matches configure_packet_tracing_aie2)
        start_broadcast_num = 15
        stop_broadcast_num = 14
        start_core_broadcast = CoreEvent(107 + start_broadcast_num)
        stop_core_broadcast = CoreEvent(107 + stop_broadcast_num)
        start_mem_broadcast = MemEvent(107 + start_broadcast_num)
        stop_mem_broadcast = MemEvent(107 + stop_broadcast_num)
        start_memtile_broadcast = MemTileEvent(142 + start_broadcast_num)
        stop_memtile_broadcast = MemTileEvent(142 + stop_broadcast_num)
        start_user_event = ShimTileEvent.USER_EVENT_1
        stop_user_event = ShimTileEvent.USER_EVENT_0

        with InsertionPoint.at_block_begin(seq_block):
            # Phase 1: Configure all tiles' trace units across all columns
            for col in active_cols:
                col_tiles = tiles_by_col[col]
                seen_core = []
                for i, tile in enumerate(col_tiles):
                    p_id = trace_id_start + i
                    tm = get_target_model(tile.parent.attributes["device"])
                    tc, tr = int(tile.col), int(tile.row)

                    if tm.is_shim_noc_or_pl_tile(tc, tr):
                        start_ev = start_user_event if tile == primary_shim else \
                            ShimTileEvent(110 + start_broadcast_num)
                        stop_ev = stop_user_event if tile == primary_shim else \
                            ShimTileEvent(110 + stop_broadcast_num)
                        configure_shimtile_tracing_aie2(
                            tile=tile, start=start_ev, stop=stop_ev,
                            events=shimtile_events, enable_packet=1,
                            packet_id=p_id, packet_type=PacketType.SHIMTILE,
                        )
                        configure_timer_ctrl_shimtile_aie2(
                            tile, start_ev,
                        )
                    elif tm.is_mem_tile(tc, tr):
                        configure_memtile_tracing_aie2(
                            tile=tile,
                            start=start_memtile_broadcast,
                            stop=stop_memtile_broadcast,
                            events=memtile_events, enable_packet=1,
                            packet_id=p_id, packet_type=PacketType.MEMTILE,
                        )
                        configure_timer_ctrl_memtile_aie2(
                            tile, start_memtile_broadcast,
                        )
                    elif tm.is_core_tile(tc, tr):
                        if tile not in seen_core:
                            configure_coretile_tracing_aie2(
                                tile=tile,
                                start=start_core_broadcast,
                                stop=stop_core_broadcast,
                                events=coretile_events, enable_packet=1,
                                packet_id=p_id, packet_type=PacketType.CORE,
                            )
                            configure_timer_ctrl_coretile_aie2(
                                tile, start_core_broadcast,
                            )
                            seen_core.append(tile)
                        else:
                            configure_coremem_tracing_aie2(
                                tile=tile,
                                start=start_mem_broadcast,
                                stop=stop_mem_broadcast,
                                events=coremem_events, enable_packet=1,
                                packet_id=p_id, packet_type=PacketType.MEM,
                            )
                            configure_timer_ctrl_coremem_aie2(
                                tile, start_mem_broadcast,
                            )

            # Phase 2: Configure per-column shim DMAs.
            # Each active shim gets its own BD and DDR buffer.
            # ddr_id starts at num_args (first free XRT buffer slot).
            trace_ddr_ids = {}  # shim_col -> ddr_id
            unique_shim_cols = sorted(shim_to_cols.keys())
            for idx, shim_col in enumerate(unique_shim_cols):
                shim = shim_tiles[shim_col]
                ddr_id = num_args + idx
                trace_ddr_ids[shim_col] = ddr_id
                # BD IDs: use 15 for first shim, 14 for second, etc.
                bd_id = 15 - idx
                configure_shimtile_dma_aie2(
                    shim=shim, channel=1, bd_id=bd_id,
                    ddr_id=ddr_id, size=trace_size // 4,
                    offset=0, enable_token=1,
                )

            # Phase 3: Fire start broadcast from primary shim
            configure_shim_trace_start_aie2(
                primary_shim, start_broadcast_num, start_user_event,
            )

        # -- Insert trace done after the last DMA wait -------------------------

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
            with InsertionPoint.after(wait_ops[-1]):
                gen_trace_done_aie2(primary_shim)
        else:
            ops = list(seq_block.operations)
            if ops:
                with InsertionPoint(ops[-1]):
                    gen_trace_done_aie2(primary_shim)

        # -- Serialize ---------------------------------------------------------

        text = str(module)

    # -- Add per-column trace buffer arguments ---------------------------------

    trace_words = trace_size // 4
    trace_memref = f"memref<{trace_words}xi32>"
    for idx, shim_col in enumerate(unique_shim_cols):
        arg_name = f"%trace_buf_col{shim_col}"
        text = re.sub(
            r"(aie\.runtime_sequence\([^)]*)\)",
            rf"\1, {arg_name}: {trace_memref})",
            text,
            count=1,
        )

    # -- Build manifest --------------------------------------------------------

    tiles_traced = []
    seen_core_manifest = []
    for tile in tiles_to_trace:
        row = tile.row.value
        col = tile.col.value
        if row == 0:
            tile_type = "shim"
            events_label = "default_shim_8"
        elif row == 1:
            tile_type = "memtile"
            events_label = "default_memtile_8"
        elif (col, row) not in seen_core_manifest:
            tile_type = "core"
            events_label = "default_core_8"
            seen_core_manifest.append((col, row))
        else:
            tile_type = "mem"
            events_label = "default_mem_8"
        tiles_traced.append({
            "col": col,
            "row": row,
            "tile_type": tile_type,
            "events": events_label,
        })

    # Per-column trace info for downstream tools
    per_column = []
    for idx, shim_col in enumerate(unique_shim_cols):
        per_column.append({
            "shim_col": shim_col,
            "ddr_id": trace_ddr_ids[shim_col],
            "trace_size": trace_size,
            "tile_cols": shim_to_cols[shim_col],
        })

    manifest_partial = {
        "trace_size": trace_size,
        "trace_ddr_id": trace_ddr_ids[unique_shim_cols[0]],  # compat
        "tiles_traced": tiles_traced,
        "per_column_trace": per_column,
        "num_trace_buffers": len(unique_shim_cols),
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
    parser.add_argument(
        "--per-column",
        action="store_true",
        help="Route each column's trace to its own shim DMA for higher bandwidth",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Output a trace routing plan as JSON without injecting",
    )
    parser.add_argument(
        "--tiles",
        type=str,
        default=None,
        help="Trace only specific tiles (col.row:module,...). "
             "Example: '0.2:core,0.2:mem,0.3:core'",
    )
    args = parser.parse_args()

    test_dir = args.test_dir.resolve()
    output_dir = args.output.resolve()

    if not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Detect source type and resolve device
    device = args.device
    source_type = detect_source_type(test_dir)

    mlir_text = get_mlir_text(test_dir, source_type, device)

    # Plan-only mode: output JSON routing plan without injection.
    # Runs BEFORE the quarantine check so quarantined tests can still
    # be inspected for diagnostic purposes.
    if args.plan_only:
        plan = plan_trace_route(mlir_text)
        print(json.dumps(plan.to_dict(), indent=2))
        sys.exit(0)

    # Parse --tiles filter if provided.
    tile_filter = None
    if args.tiles:
        tile_filter = []
        for spec in args.tiles.split(","):
            parts = spec.strip().split(":")
            coord = parts[0].split(".")
            module = parts[1] if len(parts) > 1 else "core"
            tile_filter.append((int(coord[0]), int(coord[1]), module))

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

    # Hard guard: check trace quarantine list.
    # Some tests cause IOMMU page faults and full NPU wedge when trace
    # routes are injected (control packet stream collisions, shim DMA
    # port contention).  The quarantine file is the authoritative list
    # of known-dangerous tests.  This guard runs here (not just in the
    # bridge script) so that standalone trace-inject.py calls are also
    # protected.
    quarantine_file = Path(__file__).parent.parent / "scripts" / "trace-quarantine.txt"
    if quarantine_file.exists():
        quarantined = set()
        for line in quarantine_file.read_text().splitlines():
            entry = line.split("#")[0].strip()
            if entry:
                quarantined.add(entry)
        if test_dir.name in quarantined:
            reason = "routing_conflict"
            print(f"Skipping {test_dir.name}: trace-quarantined ({reason})",
                  file=sys.stderr)
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "test_name": test_dir.name,
                "skipped": True,
                "reason": reason,
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            sys.exit(0)

    # Pathfinder-driven feasibility analysis: evaluate all (shim_col, channel)
    # candidates in parallel and pick the best route.
    plan = plan_trace_route(mlir_text)

    if not plan.feasible:
        reason = f"infeasible: {plan.reason}"
        print(f"Skipping {test_dir.name}: {reason}", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "test_name": test_dir.name,
            "skipped": True,
            "reason": reason,
        }
        if plan.candidates:
            manifest["candidates"] = [
                {
                    "shim_col": c.shim_col,
                    "channel": c.channel,
                    "failure_reason": c.failure_reason or "failed",
                }
                for c in plan.candidates
            ]
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        sys.exit(0)

    # Print planner diagnostic table
    print(f"  Planner: shim col {plan.shim_col}, S2MM channel {plan.trace_channel}",
          file=sys.stderr)
    if plan.candidates:
        for c in plan.candidates:
            status = "WINNER" if (c.shim_col == plan.shim_col
                                  and c.channel == plan.trace_channel) else ""
            flag = ("ok" if c.success
                    else (c.failure_reason or "failed"))
            print(f"    col {c.shim_col} ch {c.channel}: {flag} "
                  f"(test_cols={c.trace_connections_on_test_cols}, "
                  f"total={c.total_trace_connections}) {status}",
                  file=sys.stderr)

    # Load custom events config if provided
    events_config = None
    if args.events_json:
        events_config = load_events_config(args.events_json)

    # Inject trace
    if args.per_column:
        if tile_filter is not None:
            print("Warning: --tiles is not supported with --per-column, "
                  "ignoring filter", file=sys.stderr)
        try:
            traced_mlir, manifest_partial = inject_trace_per_column(
                mlir_text, args.trace_size, events_config,
            )
        except Exception as e:
            print(f"Error injecting trace: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            traced_mlir, manifest_partial = inject_trace(
                mlir_text, args.trace_size, plan,
                events_config=events_config,
                tile_filter=tile_filter,
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
    # Add planner analysis to manifest for diagnostics.
    manifest["planner"] = plan.to_dict()
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
    if "per_column_trace" in manifest_partial:
        n = manifest_partial["num_trace_buffers"]
        cols = [pc["shim_col"] for pc in manifest_partial["per_column_trace"]]
        print(f"  Mode:     per-column ({n} shim DMAs: cols {cols})")
        for pc in manifest_partial["per_column_trace"]:
            print(f"    Col {pc['shim_col']}: ddr_id={pc['ddr_id']}, "
                  f"tiles from cols {pc['tile_cols']}")
    else:
        print(f"  DDR ID:   {manifest_partial['trace_ddr_id']}")
    print(f"  Size:     {args.trace_size} bytes per column")
    if extra_flags:
        print(f"  Extra:    {' '.join(extra_flags)}")


if __name__ == "__main__":
    main()
