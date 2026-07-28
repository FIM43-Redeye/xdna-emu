#!/usr/bin/env python3
"""memtile_bankwidth -- discriminate the AIE2 MEMTILE bank-access width and
strided-channel parallelism (Experiment A, memtile arm).

Sibling of `bankdisc.py` (the compute-tile bank-width discriminator), moved to
row 1 (a memtile). A memtile has NO core -- so unlike bankdisc's core-vs-DMA
hammer, contention here is strictly between two memtile DMA channels: a
shim->memtile S2MM filling one buffer and a memtile->shim MM2S draining
another. Physical-bank map differs too: a memtile interleaves every 16 B over
16 banks (AM020 ch.5:105, `physical = (addr>>4)&0xF`; see
`src/device/banking.rs::BankLayout::MemTile`), NOT the compute tile's
8-physical/4-logical-bank map.

Two independent variant families:

  A1 CONTENTION (a1_collide, a1_apart, a1_solo) -- fill (S2MM, writes into
  the memtile) and drain (MM2S, reads out to the shim) are each a
  bankdisc-style SELF-LOOPING SINGLE BD (`aie.next_bd` jumping back to
  itself), gated by its OWN self-replenishing lock: `init = 1`,
  `AcquireGreaterEqual 1` before the `aie.dma_bd`, `Release 1` after. Neither
  channel's lock ever depends on the OTHER channel's release, so fill and
  drain run fully concurrently for the whole capture -- the actual
  contention window Experiment A wants to measure -- bounded only by the
  shim side's requested transfer length in the runtime sequence (the BD loop
  itself "free-runs, stream-bounded": it keeps looping after the host's
  `dma_wait` is satisfied, harmlessly, since the kernel invocation has
  already ended). Still matches the toolchain's bracket recipe (STALLED_LOCK
  falling edge to FINISHED_BD rising edge; `START_TASK`/`FINISHED_TASK` never
  fire for a self-looping single-BD chain, per the plan's global constraint).

  Each channel's `aie.dma_bd` is STRIDED to stay on a SINGLE physical bank --
  a 2D wrap `[<size = 64, stride = 64>, <size = 4, stride = 1>]` reads/writes
  one 16 B granule (4 words) every 256 B (64 words). 256 B is exactly one
  bank-interleave period, so every granule of every one of the 64 strided
  elements lands in the SAME physical bank `(base>>4)&0xF`. This replaces an
  earlier, broken A1 that pinned whole 256-word CONTIGUOUS buffers "to a
  bank": a contiguous 1 KB buffer spans all 16 banks, so a single traced
  bank saw almost no conflict (an actual HW capture read
  CONFLICT_DM_BANK_0 = 0). drain_buf is pinned to the SAME fixed address
  (bank 0) in every variant, so its OWN cadence is directly comparable
  across variants; fill_buf is pinned to the SAME physical bank (collide), a
  DIFFERENT physical bank (apart), or omitted entirely (solo = floor --
  isolates "does adding a contender slow the drain channel down").

  A2 STRIDE (a2_stride_{4,16,32,64,128,256}) -- a single memtile MM2S channel
  (no fill, no contention), same self-looping-single-BD-plus-ratchet shape,
  reads `word[i]` from `base + i*stride_bytes` for a fixed OBJ-word transfer,
  via a 2D `aie.dma_bd` wrap (`[<size=OBJ, stride=stride_words>, <size=1,
  stride=1>]` -- the inner size-1/stride-1 dim exists only to satisfy the
  dialect's "inner-most dimension's stride must be 1" rule; AIEOps.td:990).
  Sweeping the stride sweeps how many DISTINCT physical banks the transfer
  touches per 256 B period (16 banks * 16 B): stride=4 B (contiguous) and
  stride=16 B both cycle through all 16 banks; stride=256 B always lands in
  the SAME single bank (period 1). The span(strided)/span(contiguous) ratio
  is the "how many banks does the memtile serve in parallel per cycle"
  discriminator.

Both families emit REPS repeated transfers per channel (self-loop bounded by
the self-replenishing lock), mirroring bankdisc's statistical approach
without needing a core to pace it. Smoke-compiled through `aiecc.py`
(--no-aiesim, xclbin + npu-insts only; no hardware/NPU touched -- see task
report) for all nine variants.
"""
import argparse

REPS = 16
OBJ = 256   # i32 words per transfer (both A1 channels and each A2 BD)

# --- memtile physical bank map: interleave every 16 B over 16 banks, wraps
#     every 256 B (AM020 ch.5:105; src/device/banking.rs BankLayout::MemTile).
#     Derived here identically to the emulator's own model -- not a
#     placeholder constant.
BANK_INTERLEAVE_SHIFT = 4
NUM_BANKS = 16


def physical_bank(addr: int) -> int:
    return (addr >> BANK_INTERLEAVE_SHIFT) & (NUM_BANKS - 1)


# --- A1 contention variants: strided-to-a-single-bank geometry -------------
# Each element is one bank-interleave granule (GRANULE_WORDS, 16 B) spaced
# STRIDE_WORDS (one full 16-bank wrap, 256 B) apart, so every element of a
# transfer lands in the SAME physical bank (base>>4)&0xF regardless of which
# element is in flight -- see module docstring. N_ELEM is derived from OBJ so
# the total words moved per transfer still matches memtile_bankwidth_measure.
# py's TRANSFER_WORDS contract.
GRANULE_BYTES = 1 << BANK_INTERLEAVE_SHIFT                  # 16 B/granule
GRANULE_WORDS = GRANULE_BYTES // 4                          # 4 words
STRIDE_BYTES = NUM_BANKS * GRANULE_BYTES                    # 256 B/wrap
STRIDE_WORDS = STRIDE_BYTES // 4                            # 64 words
N_ELEM = OBJ // GRANULE_WORDS                               # 64 elements
BUF_ELEMS = (N_ELEM - 1) * STRIDE_WORDS + GRANULE_WORDS     # 4036 words:
                                                             # the strided
                                                             # extent one BD
                                                             # spans

_BUF_BYTES = BUF_ELEMS * 4
# Round up to the next 256 B (STRIDE_BYTES)-aligned slot past the drain
# buffer's extent -- guarantees no overlap (buffers must be declared in
# ASCENDING address order; AIEAssignBufferAddresses walks pre-addressed
# buffers in declaration order and rejects an out-of-order address) while
# staying bank-0-aligned (addr % STRIDE_BYTES == 0).
_FILL_SLOT = -(-_BUF_BYTES // STRIDE_BYTES) * STRIDE_BYTES

DRAIN_ADDR = 0                       # bank 0 in every A1 variant
FILL_COLLIDE_ADDR = _FILL_SLOT        # bank 0 -- SAME as drain (collide)
FILL_APART_ADDR = _FILL_SLOT + 128    # bank 8 -- DIFFERENT from drain (apart)

A1_VARIANTS = {
    #              fill_present, fill_addr,         drain_addr
    "a1_collide": (True,  FILL_COLLIDE_ADDR, DRAIN_ADDR),
    "a1_apart":   (True,  FILL_APART_ADDR,   DRAIN_ADDR),
    "a1_solo":    (False, None,              DRAIN_ADDR),
}
assert physical_bank(DRAIN_ADDR) == physical_bank(FILL_COLLIDE_ADDR)
assert physical_bank(DRAIN_ADDR) != physical_bank(FILL_APART_ADDR)

# --- A2 stride variants: byte stride swept, fixed OBJ word count ---
A2_STRIDES_BYTES = (4, 16, 32, 64, 128, 256)
A2_VARIANTS = {f"a2_stride_{s}": s for s in A2_STRIDES_BYTES}

VARIANTS = sorted(set(A1_VARIANTS) | set(A2_VARIANTS))


def _emit_a1(variant: str) -> str:
    fill_present, fill_addr, drain_addr = A1_VARIANTS[variant]
    total = REPS * OBJ
    buf_type = f"memref<{BUF_ELEMS}xi32>"
    dims = (f", [<size = {N_ELEM}, stride = {STRIDE_WORDS}>, "
            f"<size = {GRANULE_WORDS}, stride = 1>]")

    # Buffers declared in ASCENDING address order (drain_addr < fill_addr in
    # every variant): AIEAssignBufferAddresses walks pre-addressed buffers in
    # declaration order, tracking one monotonically-advancing next-free
    # address per allocator bank -- an out-of-order address trips its
    # "would override allocated address" check.
    drain_buf_decl = (
        f'    %drain_buf = aie.buffer(%mem_0_1) '
        f'{{sym_name = "drain_buf", address = {drain_addr} : i32}} : {buf_type}\n'
    )
    fill_buf_decl = (
        f'    %fill_buf = aie.buffer(%mem_0_1) '
        f'{{sym_name = "fill_buf", address = {fill_addr} : i32}} : {buf_type}\n'
        if fill_present else ""
    )
    drain_lock_decl = (
        f'    %lk_drain_go = aie.lock(%mem_0_1, 0) '
        f'{{init = 1 : i32, sym_name = "lk_drain_go"}}\n'
    )
    fill_lock_decl = (
        f'    %lk_fill_go = aie.lock(%mem_0_1, 1) '
        f'{{init = 1 : i32, sym_name = "lk_fill_go"}}'
        if fill_present else ""
    )
    drain_flow = "    aie.flow(%mem_0_1, DMA : 0, %shim_0_0, DMA : 0)\n"
    fill_flow = (
        "    aie.flow(%shim_0_0, DMA : 0, %mem_0_1, DMA : 0)\n"
        if fill_present else ""
    )
    drain_alloc = "    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)\n"
    fill_alloc = (
        "    aie.shim_dma_allocation @fill(%shim_0_0, MM2S, 0)"
        if fill_present else ""
    )

    if fill_present:
        seq_args = (f"%drain_dst: memref<{total}xi32>, "
                    f"%fill_src: memref<{total}xi32>")
        seq_body = (
            f"      aiex.npu.dma_memcpy_nd(%drain_dst[0, 0, 0, 0][1, 1, 1, {total}]"
            f"[0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain, issue_token = true}}"
            f" : memref<{total}xi32>\n"
            f"      aiex.npu.dma_memcpy_nd(%fill_src[0, 0, 0, 0][1, 1, 1, {total}]"
            f"[0, 0, 0, 1]) {{id = 1 : i64, metadata = @fill}} : memref<{total}xi32>"
        )
    else:
        seq_args = f"%drain_dst: memref<{total}xi32>"
        seq_body = (
            f"      aiex.npu.dma_memcpy_nd(%drain_dst[0, 0, 0, 0][1, 1, 1, {total}]"
            f"[0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain, issue_token = true}}"
            f" : memref<{total}xi32>"
        )

    drain_else = "^fill_start" if fill_present else "^end"
    drain_body = (
        f"    %0 = aie.dma_start(MM2S, 0, ^drain0, {drain_else})\n"
        "    ^drain0:\n"
        "      aie.use_lock(%lk_drain_go, AcquireGreaterEqual, 1)\n"
        f"      aie.dma_bd(%drain_buf : {buf_type}, 0, {OBJ}{dims})\n"
        "      aie.use_lock(%lk_drain_go, Release, 1)\n"
        "      aie.next_bd ^drain0\n"
    )
    if fill_present:
        drain_body += "    ^fill_start:\n"
        fill_body = (
            "    %1 = aie.dma_start(S2MM, 0, ^fill0, ^end)\n"
            "    ^fill0:\n"
            "      aie.use_lock(%lk_fill_go, AcquireGreaterEqual, 1)\n"
            f"      aie.dma_bd(%fill_buf : {buf_type}, 0, {OBJ}{dims})\n"
            "      aie.use_lock(%lk_fill_go, Release, 1)\n"
            "      aie.next_bd ^fill0\n"
        )
    else:
        fill_body = ""

    fill_note = (
        ' fill_buf @ ' + f'{fill_addr:#06x}' + f' (bank {physical_bank(fill_addr)}).'
        if fill_present else ' No fill channel (floor).'
    )
    return f"""//===- memtile_bankwidth {variant} ------------------------------*- MLIR -*-===//
// Memtile bank-access-width / contention discriminator (Experiment A1).
// Both channels strided to a single 16 B granule every 256 B (one bank
// wrap) -- see module docstring.
// drain_buf @ {drain_addr:#06x} (bank {physical_bank(drain_addr)}).{fill_note}
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %mem_0_1 = aie.tile(0, 1)

{drain_buf_decl}{fill_buf_decl}
{drain_lock_decl}{fill_lock_decl}

{drain_flow}{fill_flow}
{drain_alloc}{fill_alloc}

    aie.runtime_sequence({seq_args}) {{
{seq_body}
      aiex.npu.dma_wait {{symbol = @drain}}
    }}

    %mem_dma_0_1 = aie.memtile_dma(%mem_0_1) {{
{drain_body}{fill_body}    ^end:
      aie.end
    }}
  }}
}}
"""


def _emit_a2(variant: str) -> str:
    stride_bytes = A2_VARIANTS[variant]
    stride_words = stride_bytes // 4
    total = REPS * OBJ
    # Buffer must span the full strided extent: word (OBJ-1) sits at
    # (OBJ-1)*stride_words elements from base.
    buf_elems = (OBJ - 1) * stride_words + 1
    buf_type = f"memref<{buf_elems}xi32>"
    dims = f", [<size = {OBJ}, stride = {stride_words}>, <size = 1, stride = 1>]"

    return f"""//===- memtile_bankwidth {variant} ------------------------------*- MLIR -*-===//
// Memtile strided-channel-parallelism discriminator (Experiment A2).
// stride = {stride_bytes} B ({stride_words} words); OBJ = {OBJ} words/transfer;
// buffer spans {buf_elems} words to cover the full strided extent.
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %mem_0_1 = aie.tile(0, 1)

    %stride_buf = aie.buffer(%mem_0_1) {{sym_name = "stride_buf", address = 0 : i32}} : {buf_type}
    %lk_go = aie.lock(%mem_0_1, 0) {{init = 1 : i32, sym_name = "lk_go"}}

    aie.flow(%mem_0_1, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    aie.runtime_sequence(%drain_dst: memref<{total}xi32>) {{
      aiex.npu.dma_memcpy_nd(%drain_dst[0, 0, 0, 0][1, 1, 1, {total}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{total}xi32>
      aiex.npu.dma_wait {{symbol = @drain}}
    }}

    %mem_dma_0_1 = aie.memtile_dma(%mem_0_1) {{
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lk_go, AcquireGreaterEqual, 1)
      aie.dma_bd(%stride_buf : {buf_type}, 0, {OBJ}{dims})
      aie.use_lock(%lk_go, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }}
  }}
}}
"""


def emit(variant: str) -> str:
    if variant in A1_VARIANTS:
        return _emit_a1(variant)
    if variant in A2_VARIANTS:
        return _emit_a2(variant)
    raise KeyError(f"unknown variant {variant!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    print(emit(ap.parse_args().variant), end="")
