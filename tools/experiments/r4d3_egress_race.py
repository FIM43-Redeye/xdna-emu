#!/usr/bin/env python3
"""r4d3_egress_race -- MM2S egress staging depth by data race (Round 4, D3).

Idea: hold a consumer (an S2MM channel whose BD lock is withheld) so a source
MM2S channel fetches ahead until the stream backpressures it and then stalls.
While it is stalled, the host (NPU instruction stream, strictly ordered)
overwrites the source buffer B with a new pattern. When the consumer is then
released, every word the pipeline had already fetched carries the OLD pattern
and every later word carries the NEW pattern. The OLD-pattern count K is the
number of words absorbed between B and the held consumer:

    K = s (MM2S staging) + sum(stream-switch FIFOs on the route) + i (ingress)

Routes are chosen so the switch term differs by documented amounts (AM020
ch.2:78-92); the kernel-free design (no cores) keeps the ordering to the
command processor and the DMA hardware alone.

Variants (see build_mlir): `dma` routes tile (0,2) MM2S -> tile (0,2+hops) S2MM.
The held-consumer sink forwards the received buffer to the shim via its MM2S
channel 1 so the host can read it.
"""
import argparse

N = 128                      # words per transfer (must exceed any plausible K)
B_ADDR = 0x0400              # source buffer (logical bank 0)
C_ADDR = 0x4400              # sink buffer (logical bank 1)
SCRATCH_ADDR = 0x7000        # harmless write target for delay padding
OLD_TAG = 0x0A000000
NEW_TAG = 0x0B000000
MM2S_STATUS_0 = 0x1DF10      # DMA_MM2S_Status_0 (AM025), core-tile memory module
STALLED_STREAM_BP = 1 << 4   # Stalled_Stream_Backpressure
ROW0 = 2


def tile_base(col: int, row: int) -> int:
    return (col << 25) | (row << 20)


def build_mlir(hops: int, d1: int, d2: int, poll: bool, n: int = N, source: str = "dma",
                free_sink: bool = False, via_fifo: bool = False, align: int = 0) -> str:
    src = (0, ROW0)
    snk = (0, ROW0 + hops)
    same = hops == 0
    # bufB holds n+4 words so the BD can start `align` words in (128-bit granule
    # phase probe); words outside the BD are filler tagged 0x0C.
    old = ", ".join(
        str(OLD_TAG | (j - align)) if 0 <= j - align < n else str(0x0C000000 | j)
        for j in range(n + 4))
    new = ", ".join(str(NEW_TAG | i) for i in range(n))
    # lock ids: tile-local; when src == snk they share one lock space.
    lk_go, lk_spent = 0, 1
    lk_hold, lk_cfull, lk_cdone = (2, 3, 4) if same else (0, 1, 2)

    def pad(count):
        # Each write32 costs the command processor real time; they are the dwell.
        return "\n".join(
            "      aiex.npu.write32(%c_scr, %c_one) {buffer = @scratch} : i32, i32"
            for _ in range(count))

    snk_decl = "" if same else f"    %t_snk = aie.tile({snk[0]}, {snk[1]})\n"
    t_snk = "%t_src" if same else "%t_snk"
    lk_snk = {
        "hold": f"    %lk_hold  = aie.lock({t_snk}, {lk_hold}) {{init = 0 : i32, sym_name = \"lk_hold\"}}",
        "cfull": f"    %lk_cfull = aie.lock({t_snk}, {lk_cfull}) {{init = 0 : i32, sym_name = \"lk_cfull\"}}",
        "cdone": f"    %lk_cdone = aie.lock({t_snk}, {lk_cdone}) {{init = 0 : i32, sym_name = \"lk_cdone\"}}",
    }
    def dma_block(kind, ch, acq, buf, rel, off=0):
        return f"""      %d_{kind}{ch} = aie.dma({kind}, {ch}) [{{
        %ca_{kind}{ch} = arith.constant 1 : i32
        aie.use_lock({acq}, AcquireGreaterEqual, %ca_{kind}{ch})
        aie.dma_bd({buf} : memref<{n + 4}xi32> offset = {off} len = {n})
        %cr_{kind}{ch} = arith.constant 1 : i32
        aie.use_lock({rel}, Release, %cr_{kind}{ch})
      }}]"""

    src_dma = dma_block("MM2S", 0, "%lk_go", "%bufB", "%lk_spent", align)
    core_src = source == "core"
    src_core_op = ""
    if core_src:
        # Core streams bufB word by word; a load right before each put keeps
        # the loaded-but-unaccepted operand count at the blocked put's own.
        src_dma = ""
        # aie.put_stream must be a direct child of aie.core, so the stream is
        # emitted straight-line: one load then one put per word.
        puts = "\n".join(
            f"      %k{i} = arith.constant {i} : index\n"
            f"      %v{i} = memref.load %bufB[%k{i}] : memref<{n + 4}xi32>\n"
            f"      aie.put_stream(%ch : i32, %v{i} : i32)" for i in range(n))
        src_core_op = f"""
    %core_src = aie.core(%t_src) {{
      %ch = arith.constant 0 : i32
{puts}
      aie.end
    }}"""
    snk_dma = (dma_block("S2MM", 0, "%lk_hold", "%bufC", "%lk_cfull") + "\n"
               + dma_block("MM2S", 1, "%lk_cfull", "%bufC", "%lk_cdone"))
    if same:
        mems = f"""    %mem_src = aie.mem(%t_src) {{
{src_dma}
{snk_dma}
      aie.end
    }}"""
    else:
        src_mem = "" if core_src else f"""    %mem_src = aie.mem(%t_src) {{
{src_dma}
      aie.end
    }}
"""
        mems = f"""{src_mem}    %mem_snk = aie.mem({t_snk}) {{
{snk_dma}
      aie.end
    }}"""

    # Control: release the sink before starting the source, so nothing ever
    # stalls and the overwrite races a free-flowing transfer.
    free_op = "      aiex.set_lock(%lk_hold, %c_go)\n" if free_sink else ""
    rel_op = "" if free_sink else "      aiex.set_lock(%lk_hold, %c_go)\n"
    go_op = "" if core_src else "      aiex.set_lock(%lk_go, %c_go)\n"
    poll_op = ""
    if poll and not core_src:
        poll_op = f"""
      // Hardware-enforced precondition: source MM2S has stalled on the stream.
      %c_pa = arith.constant {MM2S_STATUS_0} : i32
      %c_pv = arith.constant {STALLED_STREAM_BP} : i32
      aiex.npu.maskpoll(%c_pa, %c_pv, %c_pv) {{column = {src[0]} : i32, row = {src[1]} : i32}} : i32, i32, i32"""

    src_port = "Core" if core_src else "DMA"
    if via_fifo:
        # Route through the source tile's stream-switch FIFO (AM020 ch.2:74,
        # 16-deep): src port -> master FIFO, slave FIFO -> sink DMA.
        flows = (f"    aie.flow(%t_src, {src_port} : 0, %t_src, FIFO : 0)\n"
                 f"    aie.flow(%t_src, FIFO : 0, {t_snk}, DMA : 0)")
    else:
        flows = f"    aie.flow(%t_src, {src_port} : 0, {t_snk}, DMA : 0)"
    return f"""// r4d3 egress race: hops={hops} d1={d1} d2={d2} poll={poll} n={n}
module {{
  aie.device(npu1_2col) {{
    memref.global "private" constant @new_data : memref<{n}xi32> = dense<[{new}]>
    %shim = aie.tile(0, 0)
    %t_src = aie.tile({src[0]}, {src[1]})
{snk_decl}
    %bufB = aie.buffer(%t_src) {{sym_name = "bufB", address = {B_ADDR} : i32, initial_value = dense<[{old}]> : tensor<{n + 4}xi32>}} : memref<{n + 4}xi32>
    %scratch = aie.buffer(%t_src) {{sym_name = "scratch", address = {SCRATCH_ADDR} : i32}} : memref<4xi32>
    %bufC = aie.buffer({t_snk}) {{sym_name = "bufC", address = {C_ADDR} : i32}} : memref<{n + 4}xi32>
    %lk_go    = aie.lock(%t_src, {lk_go}) {{init = 0 : i32, sym_name = "lk_go"}}
    %lk_spent = aie.lock(%t_src, {lk_spent}) {{init = 0 : i32, sym_name = "lk_spent"}}
{lk_snk['hold']}
{lk_snk['cfull']}
{lk_snk['cdone']}

{flows}
    aie.flow({t_snk}, DMA : 1, %shim, DMA : 0)
    aie.shim_dma_allocation @drain(%shim, S2MM, 0)

{src_core_op}
{mems}

    aie.runtime_sequence(%out: memref<{n}xi32>, %dummy: memref<4xi32>) {{
      %c_scr = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
      %c_go  = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%out[0, 0, 0, 0][1, 1, 1, {n}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @drain}} : memref<{n}xi32>
{free_op}{go_op}{poll_op}
{pad(d1)}
      %nd = memref.get_global @new_data : memref<{n}xi32>
      aiex.npu.blockwrite(%nd) {{address = {tile_base(*src) + B_ADDR + 4 * align} : ui32}} : memref<{n}xi32>
{pad(d2)}
{rel_op}      aiex.npu.dma_wait {{symbol = @drain}}
    }}
  }}
}}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hops", type=int, default=0)
    ap.add_argument("--d1", type=int, default=0)
    ap.add_argument("--d2", type=int, default=0)
    ap.add_argument("--no-poll", action="store_true")
    ap.add_argument("--source", choices=("dma", "core"), default="dma")
    ap.add_argument("--free-sink", action="store_true")
    ap.add_argument("--via-fifo", action="store_true")
    ap.add_argument("--align", type=int, default=0, choices=range(4))
    a = ap.parse_args()
    print(build_mlir(a.hops, a.d1, a.d2, not a.no_poll, source=a.source,
                     free_sink=a.free_sink, via_fifo=a.via_fifo, align=a.align), end="")
