#!/usr/bin/env python3
"""Generate sweep config JSONs programmatically.

Each helper produces a list of sweep groups in the format consumed by
run_sweep.py. Pipe to a file:

  python3 sweep_gen.py distance_npu1 --kind write32 --counts 64 256 1024 \\
      --out sweeps/distance_v1.json

Available probes:

  distance_npu1       Every (col, row) tile in the npu1 5x6 array.
  distance_col        All rows at one column (controls for column dimension).
  distance_row        All cols at one row (controls for row dimension).
  payload_dense       Many payloads x few counts for blockwrite.
  anchor_variation    Same target tile, varying anchor tile.
  read_vs_write       write32 alongside maskwrite at same target.
  dense_linearity     Contiguous counts 1..N at one target (modular structure).

Run `python3 sweep_gen.py <probe> --help` for probe-specific options.
"""

import argparse
import json
import sys
from pathlib import Path

# mlir-aie's `npu1` device exposes 4 columns x 6 rows. Note that Phoenix
# silicon physically has 5 columns of compute, but the npu1 mlir-aie target
# (TK_AIE2_NPU1_4Col) only exposes 4 of them. To target column 4 you would
# need a different device variant; for calibration we restrict to columns
# 0..3 since that's what the toolchain accepts.
NPU1_COLS = 4
NPU1_ROWS = 6  # row 0 shim, row 1 mem, rows 2..5 compute (4 deep).


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--counts", nargs="+", type=int, required=True,
                   help="Packet counts to sweep at each tile/payload")


def cmd_distance_npu1(args) -> list:
    """Every (col, row) tile in the 5x6 npu1 array, fixed anchor (0,2)."""
    groups = []
    for col in range(NPU1_COLS):
        for row in range(NPU1_ROWS):
            groups.append({
                "kind": args.kind,
                "device": "npu1",
                "anchor_col": 0, "anchor_row": 2,
                "target_col": col, "target_row": row,
                "counts": args.counts,
            })
    return groups


def cmd_distance_col(args) -> list:
    """All rows at one column."""
    return [{
        "kind": args.kind, "device": "npu1",
        "anchor_col": 0, "anchor_row": 2,
        "target_col": args.col, "target_row": row,
        "counts": args.counts,
    } for row in range(NPU1_ROWS)]


def cmd_distance_row(args) -> list:
    """All columns at one row."""
    return [{
        "kind": args.kind, "device": "npu1",
        "anchor_col": 0, "anchor_row": 2,
        "target_col": col, "target_row": args.row,
        "counts": args.counts,
    } for col in range(NPU1_COLS)]


def cmd_payload_dense(args) -> list:
    """Multiple blockwrite payloads at one target tile."""
    return [{
        "kind": "blockwrite",
        "device": "npu1_1col",
        "target_col": args.col, "target_row": args.row,
        "payload": payload,
        "counts": args.counts,
    } for payload in args.payloads]


def cmd_anchor_variation(args) -> list:
    """Fixed target tile, varying anchor tile across compute rows.

    Compares anchor on different compute tiles (rows 2..5) to test whether
    the per-packet baseline is anchor-side or target-side.
    """
    target_col, target_row = args.target_col, args.target_row
    return [{
        "kind": args.kind, "device": "npu1",
        "anchor_col": 0, "anchor_row": anchor_row,
        "target_col": target_col, "target_row": target_row,
        "counts": args.counts,
    } for anchor_row in range(2, NPU1_ROWS)]


def cmd_read_vs_write(args) -> list:
    """Write32 and maskwrite at the same target."""
    return [
        {"kind": "write32",   "target_col": args.col, "target_row": args.row,
         "counts": args.counts},
        {"kind": "maskwrite", "target_col": args.col, "target_row": args.row,
         "counts": args.counts},
    ]


def cmd_dense_linearity(args) -> list:
    """Contiguous counts from 1..max_n at a single target.

    Exposes modular structure in the cost-vs-count function. A least-squares
    slope fit over geometric counts collapses any periodic effect into a
    fractional-cycle slope; sweeping every contiguous count makes the period
    visible as a recurring delta in `cycles[N+1] - cycles[N]`.
    """
    counts = list(range(1, args.max_n + 1))
    group = {
        "kind": args.kind, "device": "npu1_1col",
        "target_col": args.col, "target_row": args.row,
        "anchor_col": 0, "anchor_row": 2,
        "counts": counts,
    }
    if args.kind == "blockwrite":
        group["payload"] = args.payload
    return [group]


PROBES = {
    "distance_npu1":     cmd_distance_npu1,
    "distance_col":      cmd_distance_col,
    "distance_row":      cmd_distance_row,
    "payload_dense":     cmd_payload_dense,
    "anchor_variation":  cmd_anchor_variation,
    "read_vs_write":     cmd_read_vs_write,
    "dense_linearity":   cmd_dense_linearity,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="probe", required=True)

    sp = sub.add_parser("distance_npu1")
    _common_args(sp)
    sp.add_argument("--kind", default="write32")

    sp = sub.add_parser("distance_col")
    _common_args(sp)
    sp.add_argument("--kind", default="write32")
    sp.add_argument("--col", type=int, required=True)

    sp = sub.add_parser("distance_row")
    _common_args(sp)
    sp.add_argument("--kind", default="write32")
    sp.add_argument("--row", type=int, required=True)

    sp = sub.add_parser("payload_dense")
    _common_args(sp)
    sp.add_argument("--payloads", nargs="+", type=int, required=True)
    sp.add_argument("--col", type=int, default=0)
    sp.add_argument("--row", type=int, default=2)

    sp = sub.add_parser("anchor_variation")
    _common_args(sp)
    sp.add_argument("--kind", default="write32")
    sp.add_argument("--target-col", type=int, required=True, dest="target_col")
    sp.add_argument("--target-row", type=int, required=True, dest="target_row")

    sp = sub.add_parser("read_vs_write")
    _common_args(sp)
    sp.add_argument("--col", type=int, default=0)
    sp.add_argument("--row", type=int, default=2)

    sp = sub.add_parser("dense_linearity")
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--kind", default="write32",
                    choices=["write32", "blockwrite", "maskwrite"])
    sp.add_argument("--max-n", type=int, default=64, dest="max_n",
                    help="Sweep contiguous counts 1..max_n")
    sp.add_argument("--col", type=int, default=0)
    sp.add_argument("--row", type=int, default=2)
    sp.add_argument("--payload", type=int, default=8,
                    help="Payload words (blockwrite only)")

    args = p.parse_args()
    groups = PROBES[args.probe](args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(groups, indent=2) + "\n")
    print(f"Wrote {args.out} ({len(groups)} groups, "
          f"{sum(len(g['counts']) for g in groups)} configurations)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
