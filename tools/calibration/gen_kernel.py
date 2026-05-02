#!/usr/bin/env python3
"""Generate parameterized calibration kernels for control-path cycle measurement.

A calibration kernel issues N copies of one specific control-packet kind
between two trace anchors (USER_EVENT_0 at the start, USER_EVENT_1 at the
end). The trace tile captures both anchor cycle stamps; the slope of
(cycle_delta vs N) recovers the marginal per-packet cost.

Anchors target the same tile as the calibration packets so the CMP issue
queue can't pipeline anchor B ahead of the calibration train.

Usage:
  python3 gen_kernel.py --kind write32 --target compute --count 1024 \\
      --out build/calib/write32_compute_n1024/

Output layout per invocation:
  <out>/aie.mlir       # generated MLIR
  <out>/params.json    # echo of the parameters for downstream tools
"""

import argparse
import json
import sys
from pathlib import Path

# AIE2 USER_EVENT IDs and Event_Generate offsets per aie-rt xaie_events_aieml.h
# and xaiemlgbl_params.h.
TILE_SPEC = {
    "compute": {
        "row": 2,
        "event_generate_offset": 0x34008,
        "user_event_0": 124,
        "user_event_1": 125,
        "trace_packet_type": "core",
        "trace_id": 1,
    },
    "mem": {
        "row": 1,
        "event_generate_offset": 0x94008,
        "user_event_0": 159,
        "user_event_1": 160,
        "trace_packet_type": "memtile",
        "trace_id": 2,
    },
    "shim": {
        "row": 0,
        "event_generate_offset": 0x34008,
        "user_event_0": 126,
        "user_event_1": 127,
        "trace_packet_type": "shimtile",
        "trace_id": 4,
    },
}

# Calibration-target register within the destination tile. We pick a write-only
# scratch register that's safe to spam: the tile's Performance_Counter0 (offset
# 0x31000 on core, 0x91000 on memtile, 0x31000 on shim/PL). Writing a value to
# a perf-counter register is a benign control-path action that exercises the
# CMP -> fabric -> register-write path without disturbing the calibration tile's
# state (DMA, locks, core).
PERF_COUNTER_OFFSET = {
    "compute": 0x31000,
    "mem": 0x91000,
    "shim": 0x31000,
}


def render_mlir(kind: str, target: str, count: int, payload: int) -> str:
    """Build the .mlir text for a calibration kernel.

    Parameters
    ----------
    kind     : "write32" | "blockwrite" | "maskwrite" | "maskpoll" | "sync"
    target   : "compute" | "mem" | "shim" -- where the calibration packets land
    count    : number of calibration packets to issue
    payload  : payload word count (BlockWrite only; ignored for others)
    """
    spec = TILE_SPEC[target]
    target_col = 0
    target_row = spec["row"]
    event_gen_addr = spec["event_generate_offset"]
    ev_a = spec["user_event_0"]
    ev_b = spec["user_event_1"]
    trace_id = spec["trace_id"]
    pkt_type = spec["trace_packet_type"]
    calib_reg = PERF_COUNTER_OFFSET[target]

    # Calibration packet line generators.
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
        # BlockWrite needs a memref source. We synthesize a global with `payload`
        # words and reference it. One global shared across all N writes is fine
        # since we only care about issue-cost timing, not destination state.
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

    def maskpoll_line(_i: int) -> str:
        # Poll a register that's already at the expected value -> 1 iter.
        return (
            f'      aiex.npu.maskpoll {{address = {hex(calib_reg)} : ui32, '
            f'column = {target_col} : i32, row = {target_row} : i32, '
            f'mask = 0x0 : ui32, value = 0 : ui32}}'
        )

    line_for = {
        "write32": write32_line,
        "maskwrite": maskwrite_line,
        "blockwrite": blockwrite_line,
        "sync": sync_line,
        "maskpoll": maskpoll_line,
    }[kind]

    # Anchor packets fire USER_EVENT_0/_1 on the trace tile.
    anchor_a = (
        f'      aiex.npu.write32 {{address = {hex(event_gen_addr)} : ui32, '
        f'column = {target_col} : i32, row = {target_row} : i32, '
        f'value = {ev_a} : ui32}}'
    )
    anchor_b = (
        f'      aiex.npu.write32 {{address = {hex(event_gen_addr)} : ui32, '
        f'column = {target_col} : i32, row = {target_row} : i32, '
        f'value = {ev_b} : ui32}}'
    )

    calib_body = "\n".join(line_for(i) for i in range(count))

    # BlockWrite needs a global memref with `payload` zeros; everything else
    # doesn't.
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

    # Trace tile MLIR fragments. We always trace on the same tile we're hitting
    # so anchor events are observed by the same trace controller.
    if target == "compute":
        trace_decl = (
            f'    aie.trace @trace_target(%target_tile) {{\n'
            f'      aie.trace.mode "Event-Time"\n'
            f'      aie.trace.packet id={trace_id} type={pkt_type}\n'
            f'      aie.trace.event<"USER_EVENT_0">\n'
            f'      aie.trace.event<"USER_EVENT_1">\n'
            f'      aie.trace.start event=<"TRUE">\n'
            f'      aie.trace.stop event=<"NONE">\n'
            f'    }}\n'
        )
    elif target == "mem":
        trace_decl = (
            f'    aie.trace @trace_target(%target_tile) {{\n'
            f'      aie.trace.packet id={trace_id} type={pkt_type}\n'
            f'      aie.trace.event<"USER_EVENT_0">\n'
            f'      aie.trace.event<"USER_EVENT_1">\n'
            f'      aie.trace.start event=<"TRUE">\n'
            f'      aie.trace.stop event=<"NONE">\n'
            f'    }}\n'
        )
    else:  # shim
        trace_decl = (
            f'    aie.trace @trace_target(%target_tile) {{\n'
            f'      aie.trace.packet id={trace_id} type={pkt_type}\n'
            f'      aie.trace.event<"USER_EVENT_0">\n'
            f'      aie.trace.event<"USER_EVENT_1">\n'
            f'      aie.trace.start event=<"TRUE">\n'
            f'      aie.trace.stop event=<"NONE">\n'
            f'    }}\n'
        )

    target_tile_decl = f'%target_tile = aie.tile({target_col}, {target_row})'
    # Always need a shim tile for the trace DMA path.
    shim_tile_decl = '%shim_noc_tile_0_0 = aie.tile(0, 0)' if target != "shim" else ""

    return f"""// Auto-generated calibration kernel. Do not edit by hand.
// kind={kind} target={target} count={count} payload={payload}
module {{
  aie.device(npu1_1col) {{
    {target_tile_decl}
    {shim_tile_decl}

{blockwrite_global}
{trace_decl}

    aie.runtime_sequence @seq() {{
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @trace_target

{blockwrite_get_global}      // Anchor A: USER_EVENT_0 on trace tile.
{anchor_a}

      // === Calibration packets (count={count}) =====================
{calib_body}
      // === End of calibration train ================================

      // Anchor B: USER_EVENT_1 on trace tile.
{anchor_b}
    }}
  }}
}}
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True,
                   choices=["write32", "blockwrite", "maskwrite", "maskpoll", "sync"])
    p.add_argument("--target", required=True, choices=["compute", "mem", "shim"])
    p.add_argument("--count", type=int, required=True,
                   help="Number of calibration packets between anchors")
    p.add_argument("--payload", type=int, default=8,
                   help="Payload words for BlockWrite (ignored otherwise)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (created if needed)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    mlir_text = render_mlir(args.kind, args.target, args.count, args.payload)
    (args.out / "aie.mlir").write_text(mlir_text)

    params = {
        "kind": args.kind,
        "target": args.target,
        "count": args.count,
        "payload": args.payload,
    }
    (args.out / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    print(f"Wrote {args.out / 'aie.mlir'} ({args.count} {args.kind} packets to {args.target})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
