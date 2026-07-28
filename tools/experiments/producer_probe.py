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

    variant          march_addr  dma_addr   march bank / dma bank      overlap
    idle             0x0400      0x2400     -- (no march)              drain floor
    apart            0x8000      0x2400     2 / 0                      none (control)
    collide          0x0400      0x2400     0 / 0                      SAME, K=4 dwell
    collide_sticky8  0x0400      0x2400     0 / 0                      SAME, K=8 dwell
    collide_sticky16 0x0400      0x2400     0 / 0                      SAME, K=16 dwell
    collide_sticky32 0x0400      0x2400     0 / 0                      SAME, K=32 dwell

collide - apart = the producer's core-vs-MM2S stall rate, per transfer, steady-state.

The collide_sticky<K> sweep probes the residual core-loss mechanism by varying the
core's physical-bank dwell K (consecutive same-bank stores before flipping) against
the fixed DMA granule period (4). collide is K=4 (matched). Sweeping K decouples the
CONFLICT count from the DMA-win count, and tests the FIFO-cover-threshold model: the
DMA fetches in stream order into a 12-word egress FIFO with the core holding per-cycle
bank priority, EXCEPT it force-grabs the bank to avoid FIFO underflow (each grab = one
core stall). That model predicts core stalls stay low while K < FIFO cover (~12) and
rise once the core camps a bank longer than the FIFO can bridge.

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
    #                   march_addr, dma_addr, march
    "idle":             (0x0400, 0x2400, False),
    "apart":            (0x8000, 0x2400, True),
    "collide":          (0x0400, 0x2400, True),   # K=4 (natural march, matched to granule)
    "collide_read":     (0x0400, 0x2400, True),   # core READS (loads) instead of writes -- port-model test
    "collide_read_dense": (0x0400, 0x2400, True),  # DENSE core READ (~2 loads/cy) -- density-vs-type discriminator
    # Density sweep: core WRITE-march throttled to ~S/G store-density by a
    # G-cycle loop-carried recurrence, to map CONFLICT_DM_BANK vs core density
    # and locate the knee (predicted ~75%, where core+DMA-25%-duty saturates
    # the bank). Names are the intended density; actual density is ELF-measured.
    "collide_d25":      (0x0400, 0x2400, True),   # S=1 G=4
    "collide_d50":      (0x0400, 0x2400, True),   # S=1 G=2
    "collide_d67":      (0x0400, 0x2400, True),   # S=2 G=3
    "collide_d75":      (0x0400, 0x2400, True),   # S=3 G=4
    "collide_d83":      (0x0400, 0x2400, True),   # S=5 G=6
    "collide_d90":      (0x0400, 0x2400, True),   # S=9 G=10
    "collide_sticky8":  (0x0400, 0x2400, True),
    "collide_sticky16": (0x0400, 0x2400, True),
    "collide_sticky32": (0x0400, 0x2400, True),
}

MARCH_BODY = f"""
        // DENSE MARCH-STORE: one store per cycle, marching contiguously, so it
        // alternates the two physical banks of its logical bank every 4 words --
        // exactly of_q0_rich's producer `eo[i] = i`. f_core ~= 1.
        scf.for %hi = %c0 to %cMARCH step %c1 {{
          %hv = arith.index_cast %hi : index to i32
          memref.store %hv, %march_buf[%hi] : memref<{MARCH_ELEMS}xi32>
        }}"""

# STICKY MARCH: dwell K cycles on one physical bank before flipping. bit4 selects the
# physical bank and flips every 16 B (= every 4 word-stores), so K same-bank stores must
# skip lines -- stride 8 words (32 B) keeps bit4 constant for K stores, then a +4-word
# base flips to the other bank for the next K. That sparse-in-address dwell would need
# tens of KB for a full march, so instead re-store the same small region MREPS times
# (data unchecked; the bank-contention timing is the observable). MREPS is sized to keep
# total march work ~constant across K (fair core-active duration). Two step-8 zero-
# overhead inner loops keep it ~1 store/cycle -- verify in the ELF before trusting the
# capture.
def sticky_body(K: int) -> str:
    hi0 = 8 * K          # bank0 run: idx 0,8,..,8(K-1)  -> K same-bank stores
    hi1 = 8 * K + 4      # bank1 run: idx 4,12,..,8(K-1)+4
    mreps = MARCH_N // (2 * K)
    return f"""
        %c4s   = arith.constant 4 : index
        %c8s   = arith.constant 8 : index
        %chi0  = arith.constant {hi0} : index
        %chi1  = arith.constant {hi1} : index
        %cMREPS = arith.constant {mreps} : index
        scf.for %rr = %c0 to %cMREPS step %c1 {{
          scf.for %i0 = %c0 to %chi0 step %c8s {{
            %v0 = arith.index_cast %i0 : index to i32
            memref.store %v0, %march_buf[%i0] : memref<{MARCH_ELEMS}xi32>
          }}
          scf.for %i1 = %c4s to %chi1 step %c8s {{
            %v1 = arith.index_cast %i1 : index to i32
            memref.store %v1, %march_buf[%i1] : memref<{MARCH_ELEMS}xi32>
          }}
        }}"""


# DENSE MARCH-LOAD: read march_buf densely instead of writing it. Same bank as the
# store-march collide, so it tests core-READ vs DMA-READ contention (vs collide's
# core-WRITE vs DMA-READ). Loads are independent (only the accumulator carries a
# dependency), so the compiler can pipeline them; the single final store sinks the
# accumulator so the loads are not dead-code-eliminated.
READ_MARCH_BODY = f"""
        %rz = arith.constant 0 : i32
        %racc = scf.for %hi = %c0 to %cMARCH step %c1 iter_args(%a = %rz) -> i32 {{
          %v = memref.load %march_buf[%hi] : memref<{MARCH_ELEMS}xi32>
          %n = arith.addi %a, %v : i32
          scf.yield %n : i32
        }}
        memref.store %racc, %march_buf[%c0] : memref<{MARCH_ELEMS}xi32>"""


# DENSE MARCH-LOAD: the density-vs-access-type discriminator. collide_read's single
# serial accumulator chain (load -> addi -> yield) throttles the loop to ~68% density
# (ISS-measured), which leaves the DMA read free cycles to defer into -> 0 conflicts.
# That 0 is consistent with BOTH "reads never conflict" and "the DMA just deferred into
# the gaps." This body removes the throttle: FOUR INDEPENDENT accumulator lanes over a
# step-4 loop, so the four loads per iteration have no cross-lane dependency and the two
# core load ports pipeline to ~2 loads/cy (no free cycles left). If HW still shows 0
# conflicts here, reads genuinely don't contend (access-type rule); if conflicts appear,
# the earlier 0 was pure deferral and the rule is density, not type. Four lane sums are
# folded and stored so nothing is dead-code-eliminated.
DENSE_READ_MARCH_BODY = f"""
        %rz0 = arith.constant 0 : i32
        %rz1 = arith.constant 0 : i32
        %rz2 = arith.constant 0 : i32
        %rz3 = arith.constant 0 : i32
        %c2d = arith.constant 2 : index
        %c3d = arith.constant 3 : index
        %c4d = arith.constant 4 : index
        %rr:4 = scf.for %hi = %c0 to %cMARCH step %c4d
                  iter_args(%a0 = %rz0, %a1 = %rz1, %a2 = %rz2, %a3 = %rz3) -> (i32, i32, i32, i32) {{
          %i1 = arith.addi %hi, %c1 : index
          %i2 = arith.addi %hi, %c2d : index
          %i3 = arith.addi %hi, %c3d : index
          %v0 = memref.load %march_buf[%hi] : memref<{MARCH_ELEMS}xi32>
          %v1 = memref.load %march_buf[%i1] : memref<{MARCH_ELEMS}xi32>
          %v2 = memref.load %march_buf[%i2] : memref<{MARCH_ELEMS}xi32>
          %v3 = memref.load %march_buf[%i3] : memref<{MARCH_ELEMS}xi32>
          %n0 = arith.addi %a0, %v0 : i32
          %n1 = arith.addi %a1, %v1 : i32
          %n2 = arith.addi %a2, %v2 : i32
          %n3 = arith.addi %a3, %v3 : i32
          scf.yield %n0, %n1, %n2, %n3 : i32, i32, i32, i32
        }}
        %s01 = arith.addi %rr#0, %rr#1 : i32
        %s23 = arith.addi %rr#2, %rr#3 : i32
        %ssum = arith.addi %s01, %s23 : i32
        memref.store %ssum, %march_buf[%c0] : memref<{MARCH_ELEMS}xi32>"""


# Density sweep parameters: variant -> (S stores per super-iter, G-cycle recurrence).
# Store density ~= S/G (II is forced to ~G by a length-G loop-carried scalar chain,
# and S stores ride inside it). ELF disassembly is the ground truth for the achieved
# density -- these are just the knobs.
DENSITY_PARAMS = {
    "collide_d25": (1, 4),
    "collide_d50": (1, 2),
    "collide_d67": (2, 3),
    "collide_d75": (3, 4),
    "collide_d83": (5, 6),
    "collide_d90": (9, 10),
}


def gap_body(S: int, G: int) -> str:
    """WRITE-march at ~S/G store-density on logical bank 0.

    Every super-iteration issues G stores on the ONE store port -- S to march_buf
    (logical bank 0, colliding with the DMA) and G-S to filler_buf (logical bank 2,
    never colliding), interleaved as evenly as possible. The store port is a hard
    resource: G stores force the pipeline initiation interval to >=G cycles, so the
    bank-0 store density is S/G. This is compiler-PROOF -- unlike a scalar +1
    recurrence, which LLVM folds to a single +G and collapses the throttle. Every
    store marches a NEW address (distinct per iter, no DCE, sweeps both physical
    banks of its logical bank). Addresses stay inside their memref (< 2048 words)."""
    mreps = max(1, 600 // G)
    n_b0, n_b2 = S * mreps, (G - S) * mreps
    assert n_b0 < MARCH_ELEMS and n_b2 < MARCH_ELEMS, f"march {n_b0} / filler {n_b2} overflow"
    # Even interleave: position p is a bank-0 store iff floor((p+1)S/G) > floor(pS/G).
    is_b0 = [((p + 1) * S) // G > (p * S) // G for p in range(G)]
    lines, b0_seen, b2_seen = [], 0, 0
    for p, b0 in enumerate(is_b0):
        if b0:
            lines.append(
                f"          %o{p} = arith.addi %b0base, %off{b0_seen} : index\n"
                f"          %ov{p} = arith.index_cast %o{p} : index to i32\n"
                f"          memref.store %ov{p}, %march_buf[%o{p}] : memref<{MARCH_ELEMS}xi32>")
            b0_seen += 1
        else:
            lines.append(
                f"          %o{p} = arith.addi %b2base, %off{b2_seen} : index\n"
                f"          %ov{p} = arith.index_cast %o{p} : index to i32\n"
                f"          memref.store %ov{p}, %filler_buf[%o{p}] : memref<{MARCH_ELEMS}xi32>")
            b2_seen += 1
    stores = "\n".join(lines)
    off_consts = "\n".join(
        f"        %off{k} = arith.constant {k} : index" for k in range(max(S, G - S)))
    return f"""
        %gcMREPS = arith.constant {mreps} : index
        %gcS = arith.constant {S} : index
        %gcGmS = arith.constant {G - S} : index
{off_consts}
        scf.for %sup = %c0 to %gcMREPS step %c1 {{
          %b0base = arith.muli %sup, %gcS : index
          %b2base = arith.muli %sup, %gcGmS : index
{stores}
        }}"""


def emit(variant: str) -> str:
    march_addr, dma_addr, do_march = VARIANTS[variant]
    filler_decl = ""
    if variant == "collide_read_dense":
        body = DENSE_READ_MARCH_BODY
    elif variant in DENSITY_PARAMS:
        body = gap_body(*DENSITY_PARAMS[variant])
        # filler_buf @ 0x8000 = logical bank 2 (never collides with the DMA on
        # bank 0); soaks the gap stores so the store port -- not a foldable
        # recurrence -- sets the initiation interval. Declared after dma_buf to
        # keep the buffer addresses ascending for the bank-aware allocator.
        filler_decl = (f'\n    %filler_buf = aie.buffer(%core_0_2) '
                       f'{{sym_name = "filler_buf", address = 32768 : i32}} : memref<{MARCH_ELEMS}xi32>')
    elif variant == "collide_read":
        body = READ_MARCH_BODY
    elif variant.startswith("collide_sticky"):
        body = sticky_body(int(variant[len("collide_sticky"):]))
    elif do_march:
        body = MARCH_BODY
    else:
        body = ""
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
    %dma_buf   = aie.buffer(%core_0_2) {{sym_name = "dma_buf",   address = {dma_addr} : i32}} : memref<{OBJ}xi32>{filler_decl}

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
        aie.use_lock(%lk_full, Release, 1){body}
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
