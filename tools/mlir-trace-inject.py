#!/usr/bin/env python3
"""
mlir-trace-inject.py -- Programmatically add declarative trace ops to an
AIE MLIR design.

Uses mlir-aie's Python bindings to parse an existing .mlir, walk its
aie.device body, insert aie.trace declarations for each compute tile, and
insert aie.trace_host_config + aie.trace_start_config ops at the top of
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
"""
import argparse
import sys
from pathlib import Path


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
    """Return True if any op in the subtree has name 'aie.trace'.

    The MLIR Python bindings expose operation.walk() but its callback
    signature expects a C++ WalkCallback type that is not easily wrapped from
    Python (calling it raises std::bad_cast).  Manual traversal through
    operation.regions is reliable and fast enough for design-scale MLIR.
    """
    if operation.name == "aie.trace":
        return True
    for region in operation.regions:
        for block in region:
            for child_op in block:
                if _has_trace_ops(child_op):
                    return True
    return False


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
            # TODO(Task 4-5): real injection logic
            raise NotImplementedError("injection not yet implemented")
        Path(args.out).write_text(str(module))


if __name__ == "__main__":
    sys.exit(main() or 0)
