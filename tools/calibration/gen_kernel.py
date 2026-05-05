#!/usr/bin/env python3
"""Generate parameterized calibration kernels for control-path cycle measurement.

A calibration kernel issues N copies of one specific control-packet kind
between two trace anchors and measures their cycle delta on real silicon.

# Design

The kernel has a minimal compute core (default tile (0,2)) that runs a tight
loop firing `aie.event(0)` (= INSTR_EVENT_0). This keeps the trace controller
warm: without a steady stream of trace events the controller never flushes its
packet buffer, and our anchor events alone (only two!) won't be enough to fill
a packet.

The runtime_sequence:
  1. Sets up trace on the anchor tile (mode=Event-Time, packet id=1).
     Trace listens for INSTR_EVENT_0 (background tick) plus USER_EVENT_0
     and USER_EVENT_1 (our anchors).
  2. Fires USER_EVENT_0 via Event_Generate on the anchor tile -> anchor A.
  3. Issues N calibration packets targeting `(target_col, target_row)`.
  4. Fires USER_EVENT_1 -> anchor B.
  5. Pads with extra anchor B firings so the trace BD's S2MM DMA has time
     to drain to the host buffer before the kernel exits.

Downstream parsing finds the FIRST occurrence of USER_EVENT_0 and USER_EVENT_1
in the events stream; the cycle delta is the calibration measurement.

The CMP (Control Microprocessor) issues all packets serially on the AXI bus,
so anchor B retires after the last calibration packet retires. The slope of
(delta vs N) reveals the per-packet contribution of the calibration target
to the CMP issue path.

Usage:
  # Single column, target compute tile (0,2):
  python3 gen_kernel.py --kind write32 --target-col 0 --target-row 2 \\
      --count 1024 --out build/calib/write32_compute_n1024/

  # Multi-column device, target memtile (3,1):
  python3 gen_kernel.py --kind write32 --device npu1 \\
      --target-col 3 --target-row 1 --count 1024 \\
      --out build/calib/write32_c3r1_n1024/

  # Backward-compat shorthand for column 0, default rows:
  python3 gen_kernel.py --kind write32 --target compute --count 1024 \\
      --out build/calib/write32_compute_n1024/

Output layout:
  <out>/aie.mlir       # generated MLIR
  <out>/params.json    # echo of the parameters for downstream tools
"""

import argparse
import json
import sys
from pathlib import Path

# AIE2 USER_EVENT IDs and Event_Generate offsets per aie-rt
# xaie_events_aieml.h and xaiemlgbl_params.h. Indexed by tile-type derived
# from row (row 0 = shim, row 1 = mem, row >= 2 = compute).
TILE_SPEC_BY_TYPE = {
    "compute": {
        "event_generate_offset": 0x34008,
        "user_event_0": 124,
        "user_event_1": 125,
        "calib_reg": 0x31000,  # Performance_Counter0 in CORE module
    },
    "mem": {
        "event_generate_offset": 0x94008,
        "user_event_0": 159,
        "user_event_1": 160,
        "calib_reg": 0x91000,  # Performance_Counter0 in MEM module
    },
    "shim": {
        "event_generate_offset": 0x34008,
        "user_event_0": 126,
        "user_event_1": 127,
        "calib_reg": 0x31000,  # Performance_Counter0 in PL module
    },
}

# Default row for the legacy --target shorthand.
DEFAULT_ROW_FOR_TYPE = {"shim": 0, "mem": 1, "compute": 2}


def tile_type_for_row(row: int) -> str:
    """Derive tile type from row index. AIE2 layout: row 0 shim, row 1 mem,
    row >= 2 compute (rows 2..5 on Phoenix, 4 compute tiles deep)."""
    if row == 0:
        return "shim"
    if row == 1:
        return "mem"
    return "compute"


def render_mlir(
    kind: str,
    count: int,
    payload: int,
    target_col: int,
    target_row: int,
    anchor_col: int,
    anchor_row: int,
    device: str,
    no_trace: bool = False,
    ticker_period: int = 256,
) -> str:
    """Generate a calibration kernel as MLIR text.

    The compute core that fires INSTR_EVENT_0 ticks always lives at
    (anchor_col, anchor_row). Anchors A/B fire on that same tile so the trace
    controller listens to a stable source. The calibration target may be any
    valid tile in the device.
    """
    target_type = tile_type_for_row(target_row)
    target_spec = TILE_SPEC_BY_TYPE[target_type]
    calib_reg = target_spec["calib_reg"]

    anchor_type = tile_type_for_row(anchor_row)
    anchor_spec = TILE_SPEC_BY_TYPE[anchor_type]
    anchor_event_gen = anchor_spec["event_generate_offset"]
    ev_a = anchor_spec["user_event_0"]
    ev_b = anchor_spec["user_event_1"]

    # The compute core that fires INSTR_EVENT_0 must be on a compute tile.
    # If anchor is shim/mem, we still need a compute tile elsewhere for the
    # core. Default: tile (anchor_col, 2) for the ticker if anchor isn't
    # itself compute.
    if anchor_type == "compute":
        ticker_col, ticker_row = anchor_col, anchor_row
    else:
        ticker_col, ticker_row = anchor_col, 2

    # aie.trace.packet's `type` enum values are core/mem/shimtile/memtile.
    # "mem" is the memory module of a compute tile (a separate trace unit on
    # row >= 2 tiles); "memtile" is the row-1 memtile; "shimtile" is row 0.
    trace_packet_type = {
        "compute": "core",
        "mem":     "memtile",
        "shim":    "shimtile",
    }[anchor_type]

    # INSTR_EVENT_0 is a CORE-module event (only valid when the trace
    # controller is on a compute tile). Memtile and shim trace controllers
    # have their own event namespaces; we simply omit the background tick
    # there and rely on the 256-entry anchor-B trail to fill the packet.
    if anchor_type == "compute":
        trace_event_lines = (
            '      aie.trace.event<"INSTR_EVENT_0">\n'
            '      aie.trace.event<"USER_EVENT_0">\n'
            '      aie.trace.event<"USER_EVENT_1">'
        )
    else:
        trace_event_lines = (
            '      aie.trace.event<"USER_EVENT_0">\n'
            '      aie.trace.event<"USER_EVENT_1">'
        )

    def write32_line(_i: int) -> str:
        return (
            f'      aiex.npu.write32 {{address = {hex(calib_reg)} : ui32, '
            f'column = {target_col} : i32, row = {target_row} : i32, '
            f'value = 0 : ui32}}'
        )

    def maskwrite_line(_i: int) -> str:
        return (
            f'      aiex.npu.maskwrite32 {{address = {hex(calib_reg)} : ui32, '
            f'column = {target_col} : i32, row = {target_row} : i32, '
            f'mask = 0xFFFFFFFF : ui32, value = 0 : ui32}}'
        )

    def blockwrite_line(_i: int) -> str:
        return (
            f'      aiex.npu.blockwrite(%blkdata) {{address = {hex(calib_reg)} : ui32, '
            f'column = {target_col} : i32, row = {target_row} : i32}} : '
            f'memref<{payload}xi32>'
        )

    def sync_line(_i: int) -> str:
        return (
            f'      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, '
            f'column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, '
            f'row_num = 1 : i32}}'
        )

    line_for = {
        "write32": write32_line,
        "maskwrite": maskwrite_line,
        "blockwrite": blockwrite_line,
        "sync": sync_line,
    }[kind]

    anchor_a = (
        f'      aiex.npu.write32 {{address = {hex(anchor_event_gen)} : ui32, '
        f'column = {anchor_col} : i32, row = {anchor_row} : i32, '
        f'value = {ev_a} : ui32}}'
    )
    anchor_b = (
        f'      aiex.npu.write32 {{address = {hex(anchor_event_gen)} : ui32, '
        f'column = {anchor_col} : i32, row = {anchor_row} : i32, '
        f'value = {ev_b} : ui32}}'
    )

    calib_body = "\n".join(line_for(i) for i in range(count))

    blockwrite_global = ""
    blockwrite_get_global = ""
    if kind == "blockwrite":
        zeros = ", ".join(["0"] * payload)
        blockwrite_global = (
            f'    memref.global "private" constant @blkdata_g : '
            f'memref<{payload}xi32> = dense<[{zeros}]>\n'
        )
        blockwrite_get_global = (
            f'      %blkdata = memref.get_global @blkdata_g : memref<{payload}xi32>\n'
        )

    # Trail: pad with USER_EVENT_1 firings to give the trace shim DMA time to
    # drain to the host buffer before the kernel exits. The first USER_EVENT_1
    # in the trace is anchor B; the rest are flush filler.
    # In no-trace mode the trail is unnecessary -- we measure runtime via
    # external wall-clock instead of the trace controller.
    TRAIL_PADDING = 0 if no_trace else 256
    trail_pad = "\n".join([anchor_b] * TRAIL_PADDING) if TRAIL_PADDING else ""

    # Tile declarations. Always declare shim (col, 0) for runtime infra. Then
    # declare ticker_tile (compute) and the trace/anchor tile if different,
    # and the calibration target if different from those.
    declared = set()
    tile_decls_list = []

    def declare(col: int, row: int) -> str:
        key = (col, row)
        if key in declared:
            return ""
        declared.add(key)
        return f'    %tile_{col}_{row} = aie.tile({col}, {row})'

    # Always declare shim tile (col, 0) for any column we touch.
    for c in {anchor_col, target_col, ticker_col}:
        tile_decls_list.append(declare(c, 0))
    tile_decls_list.append(declare(anchor_col, anchor_row))
    tile_decls_list.append(declare(ticker_col, ticker_row))
    tile_decls_list.append(declare(target_col, target_row))
    tile_decls = "\n".join(d for d in tile_decls_list if d)

    # Build the body parts conditionally so the no-trace kernel skips all
    # trace-related setup. We still emit the anchor A/B writes so the kernel's
    # control-packet stream is identical to the trace-on version (only
    # difference: no trace controller is listening).
    if no_trace:
        core_block = ""
        trace_block = ""
        rt_trace_setup = ""
    else:
        core_block = f"""    // Compute core on (ticker_col,ticker_row): tight loop firing
    // INSTR_EVENT_0 every iteration. Keeps the trace controller's stream
    // busy so anchor events flush to the shim DMA promptly. The
    // ticker_period inner loop sets the trace event rate -- shorter
    // period -> denser trace stream -> more trace controller activity.
    %core_{ticker_col}_{ticker_row} = aie.core(%tile_{ticker_col}_{ticker_row}) {{
      %c0 = arith.constant 0 : index
      %c_max = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_inner = arith.constant {ticker_period} : index
      scf.for %i = %c0 to %c_max step %c1 {{
        aie.event(0)
        scf.for %j = %c0 to %c_inner step %c1 {{
        }}
      }}
      aie.end
    }}
"""
        trace_block = f"""    // Trace config on the anchor tile.
    aie.trace @core_trace(%tile_{anchor_col}_{anchor_row}) {{
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type={trace_packet_type}
{trace_event_lines}
      aie.trace.start event=<"TRUE">
      aie.trace.stop broadcast=14
    }}
"""
        rt_trace_setup = """      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
"""

    return f"""// Auto-generated calibration kernel. Do not edit by hand.
// kind={kind} count={count} payload={payload}
// target=({target_col},{target_row}) anchor=({anchor_col},{anchor_row}) device={device}
// trace={'OFF' if no_trace else 'ON'}
module {{
  aie.device({device}) {{
{tile_decls}

{blockwrite_global}
{core_block}
{trace_block}
    aie.runtime_sequence @seq() {{
{rt_trace_setup}
{blockwrite_get_global}
      // Anchor A: fire USER_EVENT_0 on the anchor tile.
{anchor_a}

      // === Calibration packet train (count={count}) =====================
{calib_body}
      // === End of calibration train ====================================

      // Anchor B: fire USER_EVENT_1 on the anchor tile.
{anchor_b}

{trail_pad}
    }}
  }}
}}
"""


def resolve_target(args: argparse.Namespace) -> tuple:
    """Resolve target (col, row) from either explicit args or shorthand."""
    if args.target_col is not None or args.target_row is not None:
        col = args.target_col if args.target_col is not None else 0
        row = args.target_row if args.target_row is not None else 2
        return col, row
    if args.target:
        return 0, DEFAULT_ROW_FOR_TYPE[args.target]
    raise SystemExit("must supply either --target or --target-col/--target-row")


def resolve_anchor(args: argparse.Namespace) -> tuple:
    """Resolve anchor (col, row). Default is the canonical (0, 2) compute tile."""
    col = args.anchor_col if args.anchor_col is not None else 0
    row = args.anchor_row if args.anchor_row is not None else 2
    return col, row


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--kind", required=True,
                   choices=["write32", "blockwrite", "maskwrite", "sync"])
    p.add_argument("--count", type=int, required=True,
                   help="Number of calibration packets between anchors")
    p.add_argument("--payload", type=int, default=8,
                   help="Payload words for BlockWrite (ignored otherwise)")

    # Target tile.
    p.add_argument("--target", choices=["compute", "mem", "shim"],
                   help="Shorthand: target tile-type at column 0")
    p.add_argument("--target-col", type=int,
                   help="Explicit target column (overrides --target)")
    p.add_argument("--target-row", type=int,
                   help="Explicit target row (overrides --target)")

    # Anchor tile.
    p.add_argument("--anchor-col", type=int, default=None,
                   help="Anchor tile column (default 0)")
    p.add_argument("--anchor-row", type=int, default=None,
                   help="Anchor tile row (default 2 = compute)")

    p.add_argument("--device", default="npu1_1col",
                   help="aie.device target (npu1_1col, npu1, npu1_2col, etc.)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (created if needed)")
    p.add_argument("--no-trace", action="store_true",
                   help="Strip all trace controller setup and tracker core. "
                        "Use for external wall-clock measurement to "
                        "isolate trace-induced artifacts from real CMP cost.")
    p.add_argument("--ticker-period", type=int, default=256,
                   dest="ticker_period",
                   help="Inner-loop count per INSTR_EVENT_0 fire on the "
                        "compute core (default 256). Shorter = denser trace "
                        "event stream; useful for diagnosing trace-related "
                        "measurement artifacts.")
    args = p.parse_args()

    target_col, target_row = resolve_target(args)
    anchor_col, anchor_row = resolve_anchor(args)

    args.out.mkdir(parents=True, exist_ok=True)
    mlir_text = render_mlir(
        kind=args.kind,
        count=args.count,
        payload=args.payload,
        target_col=target_col,
        target_row=target_row,
        anchor_col=anchor_col,
        anchor_row=anchor_row,
        device=args.device,
        no_trace=args.no_trace,
        ticker_period=args.ticker_period,
    )
    (args.out / "aie.mlir").write_text(mlir_text)

    params = {
        "kind": args.kind,
        "count": args.count,
        "payload": args.payload,
        "target_col": target_col,
        "target_row": target_row,
        "anchor_col": anchor_col,
        "anchor_row": anchor_row,
        "device": args.device,
    }
    (args.out / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    print(f"Wrote {args.out / 'aie.mlir'} "
          f"({args.count} {args.kind} packets to ({target_col},{target_row}) "
          f"on {args.device}, anchor=({anchor_col},{anchor_row}))",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
