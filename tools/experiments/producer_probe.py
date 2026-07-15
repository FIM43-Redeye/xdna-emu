#!/usr/bin/env python3
"""producer_probe -- measure the compute-tile MM2S vs a DENSE core store loop,
steady-state, immune to Q=0 late-arming.

Sibling of `bankdisc.py` (frozen as the dma-bank-access-width finding's artifact).
Same skeleton -- a self-looping single-BD MM2S drains `dma_buf`, `STALLED_LOCK`
falling = transfer start, `FINISHED_BD` = transfer end -- but the concurrent core
body is a DENSE MARCH-STORE (`march_buf[i] = i`, one store per cycle, f_core ~= 1)
that mimics of_q0_rich's producer core (`eo[i] = i`), rather than bankdisc's sparse
load-load-store hammer (f_core = 0.15). This isolates the producer's core-vs-MM2S
bank-contention rate WITHOUT of_q0_rich's Q=0 late-arming, which caught only tail
reps and made the absolute stall count unmeasurable.

    variant   march_addr  dma_addr   march bank / dma bank      overlap
    idle      0x0400      0x2400     -- (no march)              drain floor
    apart     0x8000      0x2400     2 / 0                      none (control)
    collide   0x0400      0x2400     0 / 0                      SAME logical bank

collide - apart = the producer's core-vs-MM2S stall rate, per transfer, steady-state.

logical = (addr >> 14) & 3;  physical = 2*logical + ((addr >> 4) & 1).
The march runs MARCH_N single-word stores contiguously from march_addr; MARCH_N is
sized to stay inside one logical bank and to outlast the drain.
"""
import argparse

REPS = 16
OBJ = 256        # i32 words streamed per MM2S transfer (256 stream beats)
MARCH_ELEMS = 2048   # 8 KB buffer
MARCH_N = 1500       # dense stores per rep -- must outlast the 256-beat drain and
                     # stay inside one logical bank: 0x0400 + 1500*4 = 0x1780 < 0x4000
STACK_SIZE = 0x400

VARIANTS = {
    #            march_addr, dma_addr, march
    "idle":     (0x0400, 0x2400, False),
    "apart":    (0x8000, 0x2400, True),
    "collide":  (0x0400, 0x2400, True),
}

MARCH_BODY = f"""
        // DENSE MARCH-STORE: one store per cycle, marching contiguously, so it
        // alternates the two physical banks of its logical bank every 4 words --
        // exactly of_q0_rich's producer `eo[i] = i`. f_core ~= 1.
        scf.for %hi = %c0 to %cMARCH step %c1 {{
          %hv = arith.index_cast %hi : index to i32
          memref.store %hv, %march_buf[%hi] : memref<{MARCH_ELEMS}xi32>
        }}"""


def emit(variant: str) -> str:
    march_addr, dma_addr, do_march = VARIANTS[variant]
    total = REPS * OBJ
    return f"""//===- producer_probe {variant} -------------------------------*- MLIR -*-===//
// march_buf @ {march_addr:#06x} (logical bank {(march_addr >> 14) & 3}),
// dma_buf @ {dma_addr:#06x} (logical bank {(dma_addr >> 14) & 3}),
// march={'on' if do_march else 'off'}.
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %core_0_2 = aie.tile(0, 2)

    // Ascending address order: the bank-aware allocator rejects a pre-placed
    // buffer that lands below its per-bank cursor.
    %march_buf = aie.buffer(%core_0_2) {{sym_name = "march_buf", address = {march_addr} : i32}} : memref<{MARCH_ELEMS}xi32>
    %dma_buf   = aie.buffer(%core_0_2) {{sym_name = "dma_buf",   address = {dma_addr} : i32}} : memref<{OBJ}xi32>

    %lk_empty = aie.lock(%core_0_2, 0) {{init = 1 : i32, sym_name = "lk_empty"}}
    %lk_full  = aie.lock(%core_0_2, 1) {{init = 0 : i32, sym_name = "lk_full"}}

    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    %core = aie.core(%core_0_2) {{
      %c0     = arith.constant 0 : index
      %c1     = arith.constant 1 : index
      %cREPS  = arith.constant {REPS} : index
      %cOBJ   = arith.constant {OBJ} : index
      %cMARCH = arith.constant {MARCH_N} : index

      scf.for %r = %c0 to %cREPS step %c1 {{
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {{
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<{OBJ}xi32>
        }}
        // Release: the MM2S may now drain dma_buf.  The march below runs
        // CONCURRENTLY with that drain -- the measurement window.
        aie.use_lock(%lk_full, Release, 1){MARCH_BODY if do_march else ""}
      }}
      aie.end
    }} {{stack_size = {STACK_SIZE} : i32}}

    // Self-looping single-BD MM2S: fires once per lk_full release, REPS times.
    %mem = aie.mem(%core_0_2) {{
      %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dma_buf : memref<{OBJ}xi32>, 0, {OBJ}) {{bd_id = 0 : i32}}
      aie.use_lock(%lk_empty, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%comp_buf: memref<{total}xi32>) {{
      aiex.npu.dma_memcpy_nd(%comp_buf[0, 0, 0, 0][1, 1, 1, {total}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{total}xi32>
      aiex.npu.dma_wait {{symbol = @drain}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    print(emit(ap.parse_args().variant), end="")
