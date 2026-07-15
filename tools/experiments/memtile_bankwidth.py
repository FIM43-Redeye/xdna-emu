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

  A1 CONTENTION (a1_collide, a1_apart, a1_idle) -- fill (S2MM) and drain
  (MM2S) run as two separate, lockless BD chains (`aie.next_bd` self-chained
  REPS times; a memtile has no core to gate a producer/consumer lock handshake
  against, and `test/dialect/AIE/memtiledma.mlir` demonstrates that
  `aie.use_lock` is optional -- an unconditionally-chained BD sequence runs the
  chain back-to-back in hardware with no software pacing needed). Both
  channels are independent DMA engines, so they run fully concurrently for the
  whole capture -- exactly the contention window we want to measure.
  fill_buf is pinned at a fixed address; drain_buf is pinned to the SAME
  physical bank (collide), a DIFFERENT physical bank (apart), or fill is
  omitted entirely (idle = floor, no contention at all).

  A2 STRIDE (a2_stride_{4,16,32,64,128,256}) -- a single memtile MM2S channel
  (no fill, no contention) reads `word[i]` from `base + i*stride_bytes` for a
  fixed OBJ-word transfer, via a 2D `aie.dma_bd` wrap (`[<size=OBJ,
  stride=stride_words>, <size=1, stride=1>]` -- the inner size-1/stride-1 dim
  exists only to satisfy the dialect's "inner-most dimension's stride must be
  1" rule; AIEOps.td:990). Sweeping the stride sweeps how many DISTINCT
  physical banks the transfer touches per 256 B period (16 banks * 16 B):
  stride=4 B (contiguous) and stride=16 B both cycle through all 16 banks;
  stride=256 B always lands in the SAME single bank (period 1). The
  span(strided)/span(contiguous) ratio is the "how many banks does the memtile
  serve in parallel per cycle" discriminator.

Both families emit REPS repeated transfers per channel as a chain of
unconditionally-linked BDs (no locks) so a single capture yields REPS
transfers to bracket/pool, mirroring bankdisc's statistical approach without
needing a core to pace it.
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


# --- A1 contention variants ---
FILL_ADDR = 0x0000                                    # bank 0
A1_VARIANTS = {
    #              fill_present, drain_addr
    "a1_collide": (True,  0x1000),   # bank 0 -- SAME as fill (collide)
    "a1_apart":   (True,  0x1010),   # bank 1 -- DIFFERENT from fill (apart)
    "a1_idle":    (False, 0x1000),   # no fill at all -- floor
}
assert physical_bank(FILL_ADDR) == physical_bank(A1_VARIANTS["a1_collide"][1])
assert physical_bank(FILL_ADDR) != physical_bank(A1_VARIANTS["a1_apart"][1])

# --- A2 stride variants: byte stride swept, fixed OBJ word count ---
A2_STRIDES_BYTES = (4, 16, 32, 64, 128, 256)
A2_VARIANTS = {f"a2_stride_{s}": s for s in A2_STRIDES_BYTES}

VARIANTS = sorted(set(A1_VARIANTS) | set(A2_VARIANTS))


def _bd_chain(label: str, buf_ref: str, buf_type: str, dims: str, n: int,
              next_after: str) -> str:
    """N unconditionally-chained `aie.dma_bd` blocks, no locks.

    Block i is named ``^{label}{i}``; the last one's `next_bd` jumps to
    `next_after` (either the next channel's start label or `^end`).
    """
    blocks = []
    for i in range(n):
        dest = f"^{label}{i + 1}" if i + 1 < n else next_after
        blocks.append(
            f"    ^{label}{i}:\n"
            f"      aie.dma_bd({buf_ref} : {buf_type}, 0, {OBJ}{dims})\n"
            f"      aie.next_bd {dest}"
        )
    return "\n".join(blocks)


def _emit_a1(variant: str) -> str:
    fill_present, drain_addr = A1_VARIANTS[variant]
    total = REPS * OBJ
    buf_type = f"memref<{OBJ}xi32>"

    fill_buf_decl = (
        f'    %fill_buf = aie.buffer(%mem_0_1) '
        f'{{sym_name = "fill_buf", address = {FILL_ADDR} : i32}} : {buf_type}\n'
        if fill_present else ""
    )
    drain_buf_decl = (
        f'    %drain_buf = aie.buffer(%mem_0_1) '
        f'{{sym_name = "drain_buf", address = {drain_addr} : i32}} : {buf_type}'
    )
    fill_flow = (
        "    aie.flow(%shim_0_0, DMA : 0, %mem_0_1, DMA : 0)\n"
        if fill_present else ""
    )
    drain_flow = "    aie.flow(%mem_0_1, DMA : 0, %shim_0_0, DMA : 0)"
    fill_alloc = (
        "    aie.shim_dma_allocation @fill(%shim_0_0, MM2S, 0)\n"
        if fill_present else ""
    )
    drain_alloc = "    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)"

    if fill_present:
        seq_args = f"%fill_src: memref<{total}xi32>, %drain_dst: memref<{total}xi32>"
        seq_body = (
            f"      aiex.npu.dma_memcpy_nd(%fill_src[0, 0, 0, 0][1, 1, 1, {total}]"
            f"[0, 0, 0, 1]) {{id = 0 : i64, metadata = @fill}} : memref<{total}xi32>\n"
            f"      aiex.npu.dma_memcpy_nd(%drain_dst[0, 0, 0, 0][1, 1, 1, {total}]"
            f"[0, 0, 0, 1]) {{id = 1 : i64, metadata = @drain, issue_token = true}}"
            f" : memref<{total}xi32>"
        )
        drain_start = f"    %0 = aie.dma_start(S2MM, 0, ^fill0, ^drain_start)\n"
        drain_start += _bd_chain("fill", "%fill_buf", buf_type, "", REPS,
                                  "^drain_start") + "\n"
        drain_start += "    ^drain_start:\n"
        drain_start += "      %1 = aie.dma_start(MM2S, 0, ^drain0, ^end)\n"
    else:
        seq_args = f"%drain_dst: memref<{total}xi32>"
        seq_body = (
            f"      aiex.npu.dma_memcpy_nd(%drain_dst[0, 0, 0, 0][1, 1, 1, {total}]"
            f"[0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{total}xi32>"
        )
        drain_start = "    %1 = aie.dma_start(MM2S, 0, ^drain0, ^end)\n"

    drain_chain = _bd_chain("drain", "%drain_buf", buf_type, "", REPS, "^end")

    return f"""//===- memtile_bankwidth {variant} ------------------------------*- MLIR -*-===//
// Memtile bank-access-width / contention discriminator (Experiment A1).
// {'fill_buf @ ' + f'{FILL_ADDR:#06x}' + f' (bank {physical_bank(FILL_ADDR)}), ' if fill_present else 'no fill channel (floor), '}drain_buf @ {drain_addr:#06x} (bank {physical_bank(drain_addr)}).
//===----------------------------------------------------------------------===//
module {{
  aie.device(npu1_2col) {{
    %shim_0_0 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}}
    %mem_0_1 = aie.tile(0, 1)

{fill_buf_decl}{drain_buf_decl}

{fill_flow}{drain_flow}
{fill_alloc}{drain_alloc}

    aie.runtime_sequence({seq_args}) {{
{seq_body}
      aiex.npu.dma_wait {{symbol = @drain}}
    }}

    %mem_dma_0_1 = aie.memtile_dma(%mem_0_1) {{
{drain_start}{drain_chain}
    ^end:
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

    chain = _bd_chain("bd", "%stride_buf", buf_type, dims, REPS, "^end")

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

    aie.flow(%mem_0_1, DMA : 0, %shim_0_0, DMA : 0)
    aie.shim_dma_allocation @drain(%shim_0_0, S2MM, 0)

    aie.runtime_sequence(%drain_dst: memref<{total}xi32>) {{
      aiex.npu.dma_memcpy_nd(%drain_dst[0, 0, 0, 0][1, 1, 1, {total}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{total}xi32>
      aiex.npu.dma_wait {{symbol = @drain}}
    }}

    %mem_dma_0_1 = aie.memtile_dma(%mem_0_1) {{
    %0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
{chain}
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
