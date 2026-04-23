#!/usr/bin/env python3
"""
trace-to-cycles.py -- Extract per-test HW cycle counts from a trace buffer.

Two input modes:

  --trace-bin   : raw bytes from bridge-trace-runner's --trace-out, paired
                  with --xclbin-mlir to locate the MLIR spec that describes
                  the trace layout.  Invokes mlir-aie's parse_trace() to
                  produce Perfetto JSON events, then computes cycle delta
                  from first-to-last event timestamp (max_ts - min_ts) across
                  all non-metadata events.

  --trace-json  : already-parsed Perfetto JSON (from parse_trace or a
                  compatible source).  Same cycle-delta computation.  Used
                  in unit tests with canned fixtures, and whenever the
                  caller has already parsed.

Output: a single line with an integer cycle count, written to --out.

Exit codes:
  0 -- success (cycles written to --out)
  1 -- unexpected error (missing input, bad JSON, mlir-aie not available)
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace-bin", help="raw trace bytes from bridge-trace-runner")
    src.add_argument("--trace-json", help="pre-parsed Perfetto JSON file")
    p.add_argument("--xclbin-mlir",
                   help="MLIR used to build the xclbin (required with --trace-bin)")
    p.add_argument("--out", required=True,
                   help="output path for single-integer cycle count")
    return p.parse_args()


def cycles_from_events(events):
    """Return max(ts) - min(ts) across all timestamped events.

    This is a coarse proxy for kernel duration.  It works whether or not the
    kernel emits explicit INSTR_EVENT_0/1 boundary markers: whichever events
    the core actually fires (INSTR_VECTOR, INSTR_EVENT_*, port events, etc.)
    will bracket the interval between kernel start and end.
    """
    ts_values = [
        e["ts"] for e in events
        if isinstance(e, dict) and "ts" in e and e.get("ph") in ("B", "E", "X", "i")
    ]
    if not ts_values:
        raise ValueError("trace has no timestamped events")
    return max(ts_values) - min(ts_values)


def main():
    args = parse_args()
    if args.trace_json:
        events = json.loads(Path(args.trace_json).read_text())
    else:
        if not args.xclbin_mlir:
            print("error: --trace-bin requires --xclbin-mlir", file=sys.stderr)
            return 1
        # Lazy import -- only needed for the --trace-bin path, and mlir-aie
        # env may not be active for --trace-json usage.
        try:
            import numpy as np
            from aie.utils.trace.parse import parse_trace
        except ImportError as e:
            print(
                f"error: mlir-aie trace module not importable: {e}\n"
                "  ensure PYTHONPATH includes mlir-aie/install/python "
                "and the ironenv Python is active",
                file=sys.stderr,
            )
            return 1
        raw = np.fromfile(args.trace_bin, dtype=np.uint32)
        mlir_text = Path(args.xclbin_mlir).read_text()
        events = parse_trace(raw, mlir_text)

    try:
        cycles = cycles_from_events(events)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    Path(args.out).write_text(f"{cycles}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
