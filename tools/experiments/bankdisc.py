#!/usr/bin/env python3
"""bankdisc -- discriminate the AIE2 tile-DMA memory-bank access width.

Emits MLIR for a compute tile (0,2) whose MM2S DMA drains a fixed 256-word
buffer to the shim while the core hammers a separate scratch buffer.  The only
thing that changes between variants is WHICH LOGICAL BANK each buffer lives in,
so the MM2S transfer duration isolates bank contention.

    variant    hammer_buf   dma_buf    hammer logical bank / dma logical bank
    idle       0x0400       0x2400     0 / 0   (core does NOT hammer -- floor)
    apart      0x0400       0x8000     0 / 2   (core hammers, no bank overlap)
    collide    0x0400       0x2400     0 / 0   (core hammers the DMA's bank)
    collide2   0x8000       0xA000     2 / 2   (same, relocated to logical bank 2)

Logical bank = (addr >> 14) & 3.  Physical banks interleave every 16 B inside a
logical bank: physical = 2*logical + ((addr >> 4) & 1).  collide2 exists to test
that derivation: it should light CONFLICT_DM_BANK_4/5 and leave 0/1 silent.

The core stack is fixed at [0x0, 0x400) by AIEAssignBuffers (it allocates the
stack at address 0 and grows up), so buffers start at 0x400 and the core always
has some incidental logical-bank-0 traffic from spills.  That is identical
across variants and is why the floor is measured (idle), not assumed.

The core body is byte-for-byte identical between apart / collide / collide2 --
only the buffer addresses differ.  idle differs only by omitting the hammer loop.

Single-buffered by design: the core cannot touch dma_buf while the DMA drains it
(the lock forbids it), so during every measured transfer the core's ONLY memory
traffic is the hammer loop.  That is a cleaner isolation than a depth-2 fifo.
"""
import argparse

REPS = 16
OBJ = 256      # i32 words streamed per MM2S transfer (256 stream beats)
HAMMER_ELEMS = 2048   # 8 KB scratch buffer
HAMMER_N = 1536       # iterations per rep -- must outlast the drain
HAMMER_K = 4          # +4 words = +16 B = the sibling physical bank
STACK_SIZE = 0x400    # AIEAssignBuffers' default; pinned so the map is explicit

VARIANTS = {
    #            hammer_addr, dma_addr, hammer
    "idle":     (0x0400, 0x2400, False),
    "apart":    (0x0400, 0x8000, True),
    "collide":  (0x0400, 0x2400, True),
    "collide2": (0x8000, 0xA000, True),
}

HAMMER_BODY = f"""
        // HAMMER: two loads (sibling physical banks, 16 B apart) + one store.
        // Sustains demand on BOTH physical banks of its logical bank every iter.
        scf.for %hi = %c0 to %cHAM step %c1 {{
          %hj = arith.addi %hi, %cK : index
          %ha = memref.load %hammer_buf[%hi] : memref<{HAMMER_ELEMS}xi32>
          %hb = memref.load %hammer_buf[%hj] : memref<{HAMMER_ELEMS}xi32>
          %hs = arith.addi %ha, %hb : i32
          memref.store %hs, %hammer_buf[%hi] : memref<{HAMMER_ELEMS}xi32>
        }}"""


def emit(variant: str) -> str:
    hammer_addr, dma_addr, do_hammer = VARIANTS[variant]
    total = REPS * OBJ
    return f"""//===- bankdisc {variant} --------------------------------------*- MLIR -*-===//
// DMA bank-access-width discriminator.  hammer_buf @ {hammer_addr:#06x}
// (logical bank {(hammer_addr >> 14) & 3}), dma_buf @ {dma_addr:#06x}
// (logical bank {(dma_addr >> 14) & 3}), hammer={'on' if do_hammer else 'off'}.
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %core_0_2 = aie.tile(0, 2)

    // Declared in ASCENDING address order: the bank-aware allocator walks a
    // per-bank cursor and rejects a pre-allocated buffer that lands below it.
    %hammer_buf = aie.buffer(%core_0_2) {{sym_name = "hammer_buf", address = {hammer_addr} : i32}} : memref<{HAMMER_ELEMS}xi32>
    %dma_buf    = aie.buffer(%core_0_2) {{sym_name = "dma_buf",    address = {dma_addr} : i32}} : memref<{OBJ}xi32>

    %lk_empty = aie.lock(%core_0_2, 0) {{init = 1 : i32, sym_name = "lk_empty"}}
    %lk_full  = aie.lock(%core_0_2, 1) {{init = 0 : i32, sym_name = "lk_full"}}

    aie.flow(%core_0_2, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    %core = aie.core(%core_0_2) {{
      %c0    = arith.constant 0 : index
      %c1    = arith.constant 1 : index
      %cK    = arith.constant {HAMMER_K} : index
      %cREPS = arith.constant {REPS} : index
      %cOBJ  = arith.constant {OBJ} : index
      %cHAM  = arith.constant {HAMMER_N} : index

      scf.for %r = %c0 to %cREPS step %c1 {{
        aie.use_lock(%lk_empty, AcquireGreaterEqual, 1)
        scf.for %i = %c0 to %cOBJ step %c1 {{
          %iv = arith.index_cast %i : index to i32
          memref.store %iv, %dma_buf[%i] : memref<{OBJ}xi32>
        }}
        // Release: the MM2S may now drain dma_buf.  Everything below runs
        // CONCURRENTLY with that drain -- that is the measurement window.
        aie.use_lock(%lk_full, Release, 1){HAMMER_BODY if do_hammer else ""}
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
