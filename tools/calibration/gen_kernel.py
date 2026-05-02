#!/usr/bin/env python3
"""Generate parameterized calibration kernels for control-path cycle measurement.

A calibration kernel issues N copies of one specific control-packet kind
between two trace anchors and measures their cycle delta on real silicon.

# Design

The kernel has a minimal compute core on tile (0,2) that runs a tight loop
firing `aie.event(0)` (= INSTR_EVENT_0). This keeps the trace controller
warm: without a steady stream of trace events the controller never flushes
its packet buffer, and our anchor events alone (only two!) won't be enough
to fill a packet.

The runtime_sequence:
  1. Sets up trace on the compute tile (mode=Event-Time, packet id=1).
     Trace listens for INSTR_EVENT_0 (background tick) plus USER_EVENT_0
     and USER_EVENT_1 (our anchors).
  2. Fires USER_EVENT_0 via Event_Generate on the compute tile -> anchor A.
  3. Issues N calibration packets targeting the compute tile.
  4. Fires USER_EVENT_1 -> anchor B.
  5. Pads with extra anchor B firings so the trace BD's S2MM DMA has time
     to drain to the host buffer before the kernel exits.

Downstream parsing finds the FIRST occurrence of USER_EVENT_0 and USER_EVENT_1
in the events stream; the cycle delta is the calibration measurement.

Anchor target == calibration target: this prevents the CMP issue queue from
pipelining anchor B ahead of the calibration train. With same-tile ordering,
anchor B retires after the last calibration packet retires.

Usage:
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
# xaie_events_aieml.h and xaiemlgbl_params.h.
TILE_SPEC = {
    "compute": {
        "row": 2,
        "event_generate_offset": 0x34008,
        "user_event_0": 124,
        "user_event_1": 125,
    },
    "mem": {
        "row": 1,
        "event_generate_offset": 0x94008,
        "user_event_0": 159,
        "user_event_1": 160,
    },
    "shim": {
        "row": 0,
        "event_generate_offset": 0x34008,
        "user_event_0": 126,
        "user_event_1": 127,
    },
}

# Calibration-target register: a write-only scratch we can spam without
# disturbing tile state. Performance_Counter0 lives at:
#  - 0x31000 on a compute tile (CORE module)
#  - 0x91000 on a memtile
#  - 0x31000 on a shim tile (PL module)
PERF_COUNTER_OFFSET = {
    "compute": 0x31000,
    "mem": 0x91000,
    "shim": 0x31000,
}


def render_mlir(kind: str, target: str, count: int, payload: int) -> str:
    """Generate a calibration kernel as MLIR text."""
    spec = TILE_SPEC[target]
    target_col = 0
    target_row = spec["row"]
    event_gen_addr = spec["event_generate_offset"]
    ev_a = spec["user_event_0"]
    ev_b = spec["user_event_1"]
    calib_reg = PERF_COUNTER_OFFSET[target]

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

    def maskpoll_line(_i: int) -> str:
        # Predicate `(reg & 0) == 0` is trivially true -> 1 iteration.
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
    TRAIL_PADDING = 256
    trail_pad = "\n".join([anchor_b] * TRAIL_PADDING)

    return f"""// Auto-generated calibration kernel. Do not edit by hand.
// kind={kind} target={target} count={count} payload={payload}
module {{
  aie.device(npu1_1col) {{
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

{blockwrite_global}
    // Compute core: tight loop firing INSTR_EVENT_0 every iteration. This
    // keeps the trace controller's stream busy so anchor events flush to
    // the shim DMA promptly. The core runs until the kernel terminates.
    %core_0_2 = aie.core(%tile_0_2) {{
      %c0 = arith.constant 0 : index
      %c_max = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_inner = arith.constant 256 : index
      // Outer loop fires INSTR_EVENT_0 occasionally (every ~256 inner steps)
      // so the trace stream stays warm without saturating the trace
      // controller's input queue. Each anchor event needs a timestamp slot;
      // a busy event source would queue them up and bias B-A measurement.
      scf.for %i = %c0 to %c_max step %c1 {{
        aie.event(0)
        scf.for %j = %c0 to %c_inner step %c1 {{
        }}
      }}
      aie.end
    }}

    // Trace config: capture INSTR_EVENT_0 (background tick) plus our two
    // anchor events. Trace runs from kernel start to broadcast 14 stop.
    aie.trace @core_trace(%tile_0_2) {{
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"USER_EVENT_0">
      aie.trace.event<"USER_EVENT_1">
      aie.trace.start event=<"TRUE">
      aie.trace.stop broadcast=14
    }}

    aie.runtime_sequence @seq() {{
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace

{blockwrite_get_global}
      // Anchor A: fire USER_EVENT_0 on the trace tile.
{anchor_a}

      // === Calibration packet train (count={count}) =====================
{calib_body}
      // === End of calibration train ====================================

      // Anchor B: fire USER_EVENT_1 on the trace tile.
{anchor_b}

      // Trace flush filler: extra USER_EVENT_1 firings give the trace
      // shim DMA time to drain to the host buffer before kernel exit.
{trail_pad}
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
