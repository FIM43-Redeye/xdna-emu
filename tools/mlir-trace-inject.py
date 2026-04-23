#!/usr/bin/env python3
"""
mlir-trace-inject.py -- Programmatically add declarative trace ops to an
AIE MLIR design.

Uses mlir-aie's Python bindings to parse an existing .mlir, walk its
aie.device body, insert aie.trace declarations for each compute tile, and
insert aie.trace.host_config + aie.trace.start_config ops at the top of
the aie.runtime_sequence body.

Rationale: mlir-aie's declarative trace IRON API (aie.utils.trace) is
designed to be called from within an IRON Python design while a context
is active. We instead need to inject into pre-existing MLIR produced by
an arbitrary upstream tool. Rather than text-munging, we parse into the
MLIR IR and construct ops through the aie dialect Python bindings.

Dependencies (verified Task 2, Step 1 of the Phase B plan):
  - aie.ir (MLIR Python bindings from mlir-aie install/python)
  - aie.dialects.aie -- exposes the AIE dialect (aie.*) including:
      RuntimeSequenceOp, TileOp, CoreOp, BufferOp, TraceOp, ...
  - aie.dialects.aiex -- exposes AIEX dialect (aiex.*) and imports
      RuntimeSequenceOp from _aie_ops_gen into its own namespace.
      Importing aiex triggers the module-level register_dialect() call
      that makes both "aie.*" and "aiex.*" ops parseable.

Trace-related symbols found in aie.dialects.aie (from API probe):
  TraceComboEventOp, TraceComboEventOpAdaptor,
  TraceConfigOp, TraceConfigOpAdaptor,
  TraceEdgeEventOp, TraceEdgeEventOpAdaptor,
  TraceEventOp, TraceEventOpAdaptor,
  TraceHostConfigOp, TraceHostConfigOpAdaptor,
  TraceMode, TraceModeOp, TraceModeOpAdaptor,
  TraceOp, TraceOpAdaptor,
  TracePacketOp, TracePacketOpAdaptor, TracePacketType,
  TracePortOp, TracePortOpAdaptor,
  TraceRegOp, TraceRegOpAdaptor,
  TraceShimRouting,
  TraceStartConfigOp, TraceStartConfigOpAdaptor,
  TraceStartEventOp, TraceStartEventOpAdaptor,
  TraceStopEventOp, TraceStopEventOpAdaptor,
  _trace_event_attr, trace, trace_combo_event, trace_config,
  trace_edge_event, trace_event, trace_host_config, trace_mode,
  trace_packet, trace_port, trace_reg, trace_start, trace_start_config,
  trace_stop

Dialect registration note: importing aie.dialects.aiex is sufficient to
register both the "aie" and "aiex" dialects for MLIR text parsing.  The
aiex module runs register_dialect(get_dialect_registry()) at import time.
Using Context() without any explicit register_dialect() call then works,
provided aiex was imported before entering the context.

Usage:
  mlir-trace-inject.py --input design.mlir --out design-traced.mlir
  mlir-trace-inject.py --no-op --input design.mlir --out copy.mlir

Exit codes:
  0 -- success (output written)
  1 -- unexpected error (e.g. parse failure, argparse rejection)
  2 -- input already contains aie.trace ops; refusing to double-inject
       (output file is NOT written on exit 2)
"""
import argparse
import io
import sys
from pathlib import Path

# MLIR op name used by mlir-aie for the declarative trace op. If the
# aie dialect is ever renamed upstream, only this constant needs updating.
TRACE_OP_NAME = "aie.trace"

# Default trace configuration constants.  These mirror mlir-aie's own
# defaults in python/utils/trace/setup.py (configure_trace + _get_default_events_for_tile).
# Keeping them as named constants documents the derivation and makes upstream
# drift easy to track.
_TRACE_PACKET_ID_START = 1          # AIEInsertTraceFlows reassigns during lowering
_TRACE_BROADCAST_START = 15         # hardware broadcast channel that starts trace
_TRACE_BROADCAST_STOP = 14          # hardware broadcast channel that stops trace
_TRACE_DEFAULT_CORE_EVENTS = (
    "INSTR_VECTOR",      # vector-instruction issue
    "INSTR_EVENT_0",     # software event pin 0 (kernel boundary marker, if used)
    "INSTR_EVENT_1",     # software event pin 1 (kernel boundary marker, if used)
)


def _trace_sym(col: int, row: int) -> str:
    """Symbol name for the aie.trace op attached to tile (col, row).

    Single source of truth so a future rename only touches one place.
    Task 4 constructs the TraceOp with this name; Task 5's
    trace_start_config must reference the same name -- drift between the
    two sites would silently produce an unreachable trace config.
    """
    return f"trace_t{col}_{row}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--input", required=True, help="path to input .mlir")
    p.add_argument("--out", required=True, help="path to write traced .mlir")
    p.add_argument("--buffer-size", type=int, default=8192,
                   help="trace buffer size in bytes (default: 8192); "
                        "consumed by the host_config emission in Task 5")
    p.add_argument("--no-op", action="store_true",
                   help="parse and reserialize without injecting "
                        "(testing only -- skips injection)")
    return p.parse_args()


def _has_trace_ops(operation) -> bool:
    """Return True if any op in the subtree has name TRACE_OP_NAME.

    The MLIR Python bindings expose operation.walk() but its callback
    signature expects a C++ WalkCallback type that is not easily wrapped from
    Python (calling it raises std::bad_cast).  Manual traversal through
    operation.regions is reliable and fast enough for design-scale MLIR.
    """
    if operation.name == TRACE_OP_NAME:
        return True
    for region in operation.regions:
        for block in region:
            for child_op in block:
                if _has_trace_ops(child_op):
                    return True
    return False


def _inject_trace_ops(module, input_path: str, aied, buffer_size: int) -> int:
    """Walk module, find compute tiles (row >= 2), inject one aie.trace per tile,
    and prepend aie.trace.host_config + aie.trace.start_config ops to the
    aie.runtime_sequence body.

    Compute tile classification for npu1 / npu1_1col:
      row 0  -- shim tile (NoC interface)
      row 1  -- memory tile (shared SRAM, no core)
      row >= 2 -- compute tile (VLIW core + vector unit)

    Per-tile trace injection (Task 4):
    Construction uses the decorator-form lowercase builder aied.trace(tile, sym),
    which matches how mlir-aie's own configure_trace() (python/utils/trace/setup.py)
    constructs these ops.  The decorator creates the TraceOp with a proper region
    block, sets up the InsertionPoint for the body, calls the decorated function,
    and returns the completed op -- identical to how aied.core() and aied.device()
    work.  This is Path A (direct Python binding constructors).

    Runtime sequence injection (Task 5):
    After the per-tile traces are inserted, we find aie.runtime_sequence and
    prepend two classes of ops at block-begin (before any existing runtime ops):
      aie.trace.host_config buffer_size = <N>
      aie.trace.start_config @trace_t{col}_{row}   (one per compute tile)
    This mirrors mlir-aie's own configure_trace() in python/utils/trace/setup.py
    (lines 534-542).  mlir-aie's AIEInsertTraceFlows pass then lowers these
    declarative ops to register-write sequences during aiecc.py compilation.

    Output note: the caller must use module.operation.print(print_generic_op_form=False)
    rather than str(module) because the aie.trace region body lacks an explicit
    block terminator in MLIR generic form, which breaks Module.parse() on reload.
    The custom text-form printer handles this implicitly via aie.trace.stop.

    Returns 0 on success, non-zero on error.
    """
    from aie.ir import InsertionPoint, IntegerAttr

    # Locate the single aie.device op at the top level.
    device_op = None
    for op in module.body.operations:
        if op.operation.name == "aie.device":
            device_op = op
            break
    if device_op is None:
        print(
            f"error: no aie.device op found in {input_path}",
            file=sys.stderr,
        )
        return 1

    # Collect compute tile SSA values from the device body.
    # Keep them in declaration order so injected traces appear deterministically.
    device_body = device_op.operation.regions[0].blocks[0]
    compute_tiles = []  # list of (col: int, row: int, tile_ssa_value)
    for inner in device_body.operations:
        if inner.operation.name == "aie.tile":
            col = int(IntegerAttr(inner.operation.attributes["col"]).value)
            row = int(IntegerAttr(inner.operation.attributes["row"]).value)
            if row >= 2:
                compute_tiles.append((col, row, inner.operation.result))

    if not compute_tiles:
        print(
            f"warning: no compute tiles (row >= 2) found in {input_path}; "
            "writing unchanged",
            file=sys.stderr,
        )
        return 0

    # Insert trace ops before the device body's terminator.  The aie.device
    # block's last op is always its terminator (aie.end for well-formed input,
    # but using "last op" rather than matching name == 'aie.end' is resilient
    # to upstream renames and to blocks where the terminator differs).  If the
    # block is somehow empty, we bail rather than risk writing malformed MLIR.
    ops_list = list(device_body.operations)
    if not ops_list:
        print(
            f"error: aie.device body is empty in {input_path}; cannot inject",
            file=sys.stderr,
        )
        return 1
    terminator = ops_list[-1]

    with InsertionPoint(terminator.operation):
        for col, row, tile_val in compute_tiles:
            # The @aied.trace(tile, sym) decorator-form builder:
            #   1. Constructs TraceOp(tile=tile_val, sym_name=sym_name)
            #   2. Appends a block to the op's single region
            #   3. Sets the InsertionPoint to at_block_begin of that block
            #   4. Calls the decorated function synchronously (populates body)
            #   5. Restores the previous InsertionPoint
            # This exactly mirrors how mlir-aie's configure_trace() works.
            # The decorator invokes _trace_body synchronously before the next
            # loop iteration reassigns the name, so redefinition is safe.
            @aied.trace(tile_val, _trace_sym(col, row))
            def _trace_body():  # noqa: F811 -- redefinition is intentional (one per tile)
                aied.trace_mode(aied.TraceMode.EventTime)
                # packet id is the conventional starting id; AIEInsertTraceFlows
                # reassigns ids when it lowers the declarative ops.
                aied.trace_packet(_TRACE_PACKET_ID_START, aied.TracePacketType.Core)
                for event_name in _TRACE_DEFAULT_CORE_EVENTS:
                    aied.trace_event(event_name)
                aied.trace_start(broadcast=_TRACE_BROADCAST_START)
                aied.trace_stop(broadcast=_TRACE_BROADCAST_STOP)

    # Task 5: find aie.runtime_sequence and prepend host_config + start_config.
    # We must locate the op AFTER the per-tile traces are inserted (above), because
    # the device body is walked fresh here -- the earlier InsertionPoint context has
    # already closed, so the device body iterator reflects the completed state.
    rs_op = None
    for inner in device_body.operations:
        if inner.operation.name == "aie.runtime_sequence":
            rs_op = inner
            break
    if rs_op is None:
        print(
            f"warning: no aie.runtime_sequence in {input_path}; "
            "trace host config not emitted (trace decls still present)",
            file=sys.stderr,
        )
        return 0

    rs_block = rs_op.operation.regions[0].blocks[0]
    with InsertionPoint.at_block_begin(rs_block):
        # trace_host_config(buffer_size=N) emits:
        #   aie.trace.host_config buffer_size = N
        # Signature: trace_host_config(buffer_size, *, arg_idx=4,
        #                              routing=TraceShimRouting.Single, ...)
        # Mirror of mlir-aie python/utils/trace/setup.py line 534.
        aied.trace_host_config(buffer_size=buffer_size)
        # trace_start_config(name) emits:
        #   aie.trace.start_config @<name>
        # One per compute tile, matching the trace decl symbols inserted above.
        # Mirror of mlir-aie python/utils/trace/setup.py line 542.
        for col, row, _ in compute_tiles:
            aied.trace_start_config(_trace_sym(col, row))

    return 0


def main():
    args = parse_args()
    text = Path(args.input).read_text()
    # Import here so --help works without the mlir-aie env activated.
    # Importing aiex (before creating a Context) triggers the module-level
    # register_dialect() that makes both aie.* and aiex.* ops parseable.
    import aie.dialects.aie as aied   # noqa: F401 -- side-effect: registers aie dialect
    import aie.dialects.aiex as aiex  # noqa: F401 -- side-effect: registers aiex dialect
    from aie.ir import Context, Location, Module
    with Context(), Location.unknown():
        module = Module.parse(text)
        if not args.no_op:
            # Idempotency guard: refuse to inject if trace ops already exist.
            # --no-op is an identity round-trip and is always safe to re-run.
            if _has_trace_ops(module.operation):
                print(
                    f"error: {args.input} already contains aie.trace ops; "
                    "refusing to double-inject (exit 2)",
                    file=sys.stderr,
                )
                return 2
            rc = _inject_trace_ops(module, args.input, aied, args.buffer_size)
            if rc != 0:
                return rc
        buf = io.StringIO()
        # Use print_generic_op_form=False so the aie dialect's custom printers
        # emit text-form syntax (e.g. "aie.trace @sym(tile) { ... }") rather
        # than generic form ("aie.trace"(%0) <{sym_name = ...}> ({...})).
        # The generic form lacks explicit block terminators for aie.trace
        # regions and cannot be round-tripped through Module.parse().
        module.operation.print(print_generic_op_form=False, file=buf)
        Path(args.out).write_text(buf.getvalue())


if __name__ == "__main__":
    sys.exit(main() or 0)
