#!/usr/bin/env python3
"""Pipeline-fill probe: how many reps does `of_q0_rich` run before its un-armed drain blocks it?

The design is `of_q0_rich` (mlir-aie test/npu-xrt/spike_bringup, the bank-arbitration capture):
producer (0,2) -> `src` -> memtile -> `d0`/`d1` -> consumers (0,3)/(1,3) -> `j0`/`j1` -> memtile join
-> `out` -> shim S2MM 0. The cores start at CDO enable, so the pipeline fills and blocks on `out`
long before the host arms the drain. Instead of inferring the depth from a trace window, this probe
reads the state of the blocked pipeline over control packets before the drain is armed:

  * each core stores its progress into a small buffer (`cnt`): reps released, and for the consumers
    the last rep whose `d` / `j` acquire returned;
  * the producer stamps the rep into its data (`rep << 16 | i`; the consumers add 1 / 2), so the
    first and last word of every buffer say which rep it holds;
  * every objectfifo lock value and the status of every DMA channel on the path.

The snapshot is taken twice (the pipeline must be settled: both equal), the drain is armed for all
16 reps, and a final snapshot follows. The trace units are dropped; the data path, the objectfifos
and the cores are the original's, but for the progress stores and the stamp.

Control packets go out on shim MM2S 0 and their responses come back on shim S2MM 1 (the drain
keeps S2MM 0, as in the original).

    gen_pipeline_fill_probe.py generate OUT       (writes OUT/q0fill/aie.mlir, a layout pass)
    gen_pipeline_fill_probe.py build OUT          (compiles, resolves buffer addresses, recompiles)
    gen_pipeline_fill_probe.py run OUT TAG --backend hw|emu
    gen_pipeline_fill_probe.py analyze OUT TAG
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import ctrlpkt_readback as cp  # noqa: E402

REPS = 16
OUT_WORDS = REPS * 64
CASE = "q0fill"

# tile -> (MLIR coordinates, AM025 module of its TileControl registers, ctrl packet id, response id)
TILES = {
    "shim": ((0, 0), "shim", 4, 1),
    "mem": ((0, 1), "memory_tile", 6, 3),
    "prod": ((0, 2), "memory", 5, 2),
    "consA": ((0, 3), "memory", 7, 9),
    "consB": ((1, 3), "memory", 8, 10),
}
VAR = {"shim": "%shim_noc_tile_0_0", "mem": "%mem_tile_0_1", "prod": "%tile_0_2",
       "consA": "%tile_0_3", "consB": "%tile_1_3"}
for _name, (_xy, _module, _pkt, _resp) in TILES.items():
    cp.TARGETS[_name] = cp.Target(_pkt, _resp, _module)

# Locks each tile's objectfifos use (from the lowered design: compute tiles 0..3, memtile 0..7).
LOCKS = {"prod": range(2), "consA": range(4), "consB": range(4), "mem": range(8), "shim": range(2)}
CHANNELS = {  # DMA status registers on the path
    "prod": ["DMA_MM2S_Status_0"],
    "consA": ["DMA_S2MM_Status_0", "DMA_MM2S_Status_0"],
    "consB": ["DMA_S2MM_Status_0", "DMA_MM2S_Status_0"],
    "mem": [f"DMA_S2MM_Status_{c}" for c in range(3)] + [f"DMA_MM2S_Status_{c}" for c in range(3)],
    "shim": ["DMA_S2MM_Status_0"],
}
# Buffers whose first / last word are read: (tile, symbol, words, offsets)
BUFFERS = [
    ("prod", "src__buf0", 64, (0, 63)), ("prod", "src__buf1", 64, (0, 63)),
    ("mem", "src__cons__buf0", 64, (0, 31, 32, 63)), ("mem", "src__cons__buf1", 64, (0, 31, 32, 63)),
    ("consA", "d0__cons__buf0", 32, (0, 31)), ("consA", "d0__cons__buf1", 32, (0, 31)),
    ("consA", "j0__buf0", 32, (0, 31)), ("consA", "j0__buf1", 32, (0, 31)),
    ("consB", "d1__cons__buf0", 32, (0, 31)), ("consB", "d1__cons__buf1", 32, (0, 31)),
    ("consB", "j1__buf0", 32, (0, 31)), ("consB", "j1__buf1", 32, (0, 31)),
    ("mem", "out__buf0", 64, (0, 31, 32, 63)), ("mem", "out__buf1", 64, (0, 31, 32, 63)),
]
COUNTERS = {"prod": ("released",), "consA": ("released", "got_d", "got_j"),
            "consB": ("released", "got_d", "got_j")}
CNT_ADDR = 0x3000          # tile-local address of each core's progress buffer (free in the original)
CORE_STATUS = cp.module_registers("core")["Core_Status"]


def snapshot(tag: str, addresses: dict[tuple[str, str], int] | None) -> list:
    out: list = []
    for tile, names in COUNTERS.items():
        out += [cp.Read(CNT_ADDR + 4 * i, f"{tag} {tile} cnt.{n}", tile) for i, n in enumerate(names)]
        out.append(cp.Read(CORE_STATUS, f"{tag} {tile} core_status", tile))
    for tile, locks in LOCKS.items():
        k = cp.TileRegs(tile)
        out += [cp.Read(k.value(i), f"{tag} {tile} lock{i}", tile) for i in locks]
        out += [cp.Read(k.regs[r], f"{tag} {tile} {r}", tile) for r in CHANNELS[tile]]
    if addresses:
        for tile, sym, _, offsets in BUFFERS:
            base = addresses[(tile, sym)]
            out += [cp.Read(base + 4 * o, f"{tag} {tile} {sym}[{o}]", tile) for o in offsets]
    return out


def steps(addresses) -> list:
    st = [*snapshot("s1", addresses), *snapshot("s2", addresses)]
    st.append(cp.Raw((f"aiex.npu.dma_memcpy_nd(%data[0, 0, 0, 0][1, 1, 1, {OUT_WORDS}][0, 0, 0, 1]) "
                      f"{{id = 4 : i64, issue_token = true, metadata = @out}} : memref<{OUT_WORDS}xi32>",)))
    st.append(cp.Wait("out"))
    st += snapshot("f", addresses)
    return st


def core_body(name: str) -> list[str]:
    """The original cores, plus the progress stores (`cnt`) and the producer's rep stamp."""
    if name == "prod":
        return [
            "      %c0 = arith.constant 0 : index", "      %c16 = arith.constant 16 : index",
            "      %c1 = arith.constant 1 : index", "      %c16i = arith.constant 16 : i32",
            "      %c1i = arith.constant 1 : i32",
            "      scf.for %r = %c0 to %c16 step %c1 {",
            "        %ri = arith.index_cast %r : index to i32",
            "        %stamp = arith.shli %ri, %c16i : i32",
            "        %1 = aie.objectfifo.acquire @src(Produce, 1) : memref<64xi32>",
            "        %c64 = arith.constant 64 : index",
            "        scf.for %i = %c0 to %c64 step %c1 {",
            "          %2 = arith.index_cast %i : index to i32",
            "          %3 = arith.ori %2, %stamp : i32",
            "          memref.store %3, %1[%i] : memref<64xi32>",
            "        }",
            "        aie.objectfifo.release @src(Produce, 1)",
            "        %n = arith.addi %ri, %c1i : i32",
            "        memref.store %n, %cnt_prod[%c0] : memref<4xi32>",
            "      }",
            "      aie.end",
        ]
    d, j, add, cnt = ("d0", "j0", 1, "%cnt_consA") if name == "consA" else ("d1", "j1", 2, "%cnt_consB")
    return [
        "      %c0 = arith.constant 0 : index", "      %c16 = arith.constant 16 : index",
        "      %c1 = arith.constant 1 : index", "      %c2 = arith.constant 2 : index",
        "      %c1i = arith.constant 1 : i32",
        "      scf.for %r = %c0 to %c16 step %c1 {",
        "        %ri = arith.index_cast %r : index to i32",
        "        %n = arith.addi %ri, %c1i : i32",
        f"        %1 = aie.objectfifo.acquire @{d}(Consume, 1) : memref<32xi32>",
        f"        memref.store %n, {cnt}[%c1] : memref<4xi32>",
        f"        %3 = aie.objectfifo.acquire @{j}(Produce, 1) : memref<32xi32>",
        f"        memref.store %n, {cnt}[%c2] : memref<4xi32>",
        "        %c32 = arith.constant 32 : index",
        "        scf.for %i = %c0 to %c32 step %c1 {",
        "          %4 = memref.load %1[%i] : memref<32xi32>",
        f"          %k = arith.constant {add} : i32",
        "          %5 = arith.addi %4, %k : i32",
        "          memref.store %5, %3[%i] : memref<32xi32>",
        "        }",
        f"        aie.objectfifo.release @{d}(Consume, 1)",
        f"        aie.objectfifo.release @{j}(Produce, 1)",
        f"        memref.store %n, {cnt}[%c0] : memref<4xi32>",
        "      }",
        "      aie.end",
    ]


def design(addresses) -> tuple[str, list, "cp.Layout"]:
    st = steps(addresses)
    layout = cp.encode(st)
    lines = ["module {", "  aie.device(npu1_2col) {"]
    for name, ((c, r), _m, pkt, _resp) in TILES.items():
        lines.append(f"    {VAR[name]} = aie.tile({c}, {r}) "
                     f"{{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {pkt}>}}")
    lines += [
        "    aie.objectfifo @src(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>",
        "    aie.objectfifo @d0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32xi32>>",
        "    aie.objectfifo @d1(%mem_tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32xi32>>",
        "    aie.objectfifo.link [@src] -> [@d0, @d1]([] [0, 32])",
        "    aie.objectfifo @j0(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>",
        "    aie.objectfifo @j1(%tile_1_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>",
        "    aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>",
        "    aie.objectfifo.link [@j0, @j1] -> [@out]([0, 32] [])",
    ]
    for name in COUNTERS:
        lines.append(f'    %cnt_{name} = aie.buffer({VAR[name]}) {{address = {CNT_ADDR} : i32, '
                     f'sym_name = "cnt_{name}"}} : memref<4xi32>')
    for name in COUNTERS:
        lines += [f"    %core_{name} = aie.core({VAR[name]}) {{", *core_body(name), "    }"]
    shim = VAR["shim"]
    for name, (_xy, _m, pkt, resp) in TILES.items():
        lines += [f"    aie.packet_flow({pkt}) {{", f"      aie.packet_source<{shim}, DMA : 0>",
                  f"      aie.packet_dest<{VAR[name]}, TileControl : 0>", "    }",
                  f"    aie.packet_flow({resp}) {{", f"      aie.packet_source<{VAR[name]}, TileControl : 0>",
                  f"      aie.packet_dest<{shim}, DMA : 1>", "    }"]
    lines += [f"    aie.shim_dma_allocation @ctrlin0 ({shim}, MM2S, 0)",
              f"    aie.shim_dma_allocation @ctrl0 ({shim}, S2MM, 1)",
              f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{OUT_WORDS}xi32>) {{",
              *cp.sequence_lines(st, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), st, layout


def write_case(d: Path, addresses) -> None:
    text, st, layout = design(addresses)
    d.mkdir(parents=True, exist_ok=True)
    (d / "aie.mlir").write_text(text)
    (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
    meta = {"case": CASE, "labels": list(layout.labels), "n_words": len(layout.words), "n_data": OUT_WORDS,
            "host_args": [], "addresses": {f"{t} {s}": a for (t, s), a in (addresses or {}).items()}}
    (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")


def lowered_addresses(d: Path) -> dict[tuple[str, str], int]:
    """Buffer addresses from aiecc's lowered design."""
    text = next(d.glob("*.prj/input_with_addresses.mlir")).read_text()
    by_var = {v: t for t, v in VAR.items()}
    out = {}
    for m in re.finditer(r"aie\.buffer\((%\w+)\) \{address = (\d+) : i32,[^}]*?sym_name = \"(\w+)\"\}", text):
        if m.group(1) in by_var:
            out[(by_var[m.group(1)], m.group(3))] = int(m.group(2))
    return out


def cmd_build(args) -> int:
    from harness.build import aiecc_command, aiecc_path
    d = Path(args.out_dir) / CASE

    def compile_() -> int:
        cmd = aiecc_command("peano", aiecc_path())
        # the lowered design (buffer addresses) is kept only with --dump-intermediates
        cmd.insert(-1, "--dump-intermediates")
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        (d / "aiecc.log").write_text(r.stdout + r.stderr)
        print(f"{d.name}: aiecc exit {r.returncode}", flush=True)
        return r.returncode

    if compile_():
        return 1
    addresses = lowered_addresses(d)
    missing = [b for b in BUFFERS if (b[0], b[1]) not in addresses]
    if missing:
        raise SystemExit(f"buffers not found in the lowered design: {missing}")
    write_case(d, addresses)
    if compile_():
        return 1
    if lowered_addresses(d) != addresses:
        raise SystemExit("buffer addresses moved between the layout pass and the probe build")
    return 0


def parse(d: Path, tag: str) -> dict[str, int]:
    meta = json.loads((d / "case.json").read_text())
    raw = (d / "runs" / tag / "out.bin").read_bytes()
    n = len(meta["labels"])
    words = struct.unpack(f"<{n + meta['n_data']}I", raw[:4 * (n + meta["n_data"])])
    return dict(zip(meta["labels"], words[:n])) | {"data": list(words[n:])}


def tag_text(w: int) -> str:
    return "----" if w == 0xEEEEEEEE else f"r{w >> 16}+{w & 0xFFFF}"


def report(r: dict) -> str:
    kinds = {"prod": "core", "consA": "core", "consB": "core", "mem": "mem", "shim": "shim"}
    lines = []
    for snap in ("s1", "s2", "f"):
        lines.append(f"[{snap}]")
        for tile in TILES:
            vals = {k.split(" ", 2)[2]: v for k, v in r.items() if k.startswith(f"{snap} {tile} ")}
            cnt = " ".join(f"{k}={v}" for k, v in vals.items() if k.startswith("cnt."))
            locks = " ".join(f"{v}" for k, v in vals.items() if k.startswith("lock"))
            lines.append(f"  {tile:5s} {cnt}  core_status={vals.get('core_status', 0):#x}  locks=[{locks}]")
            for k, v in vals.items():
                if k.startswith("DMA_"):
                    lines.append(f"        {k}: {status_text(v, kinds[tile])}")
            bufs = [f"{k}={tag_text(v)}" for k, v in vals.items() if "[" in k]
            if bufs:
                lines.append("        " + " ".join(bufs))
    data = r["data"]
    order = [data[i] >> 16 for i in range(0, len(data), 32)]
    lines.append(f"drained rep stamps per 32-word half: {order}")
    # rep k's half h holds the producer's words 32h..32h+31, plus 1 (consA) or 2 (consB)
    good = all(data[64 * k + 32 * h + i] == ((k << 16) | (32 * h + i)) + 1 + h
               for k in range(REPS) for h in (0, 1) for i in range(32))
    lines.append(f"drained data exact: {good}")
    return "\n".join(lines)


def status_text(word: int, kind: str) -> str:
    import gen_dma_queue_probe as q
    return q.status_text(word, kind)


# -- chain family: when does an MM2S BD finish while its consumer is held? -------------------
#
# A tile's MM2S 0 sends a chain of K BDs of L words each through its own switch to its S2MM 0,
# whose one BD is parked on a gate lock the host keeps at 0. Every MM2S BD releases a count
# lock when it finishes. Once the sender has settled, the count says how many BDs finished
# with the consumer held: a BD that finishes when its last word is fetched into the egress
# staging finishes floor(absorb / L) BDs; one that finishes when its last word leaves the
# port finishes floor((absorb - staging) / L). The host then opens the gate and every word
# must land in order.

CHAIN_K = 12
CHAIN_LENGTHS = (4, 8, 12, 16, 24)
CHAIN_SNAPS = 3


def chain_name(kind: str, length: int) -> str:
    return f"chain_{kind}_L{length}"


def chain_steps(kind: str, length: int) -> list:
    import gen_dma_queue_probe as q
    k = cp.TileRegs(kind)
    s2mm, mm2s = k.regs["DMA_S2MM_Status_0"], k.regs["DMA_MM2S_Status_0"]
    gate, count = k.value(q.GATE_LOCK[kind]), k.value(q.COUNT_LOCK[kind])
    total = CHAIN_K * length

    lock_o = k.value(q.GATE_LOCKS[kind][2])

    def snap(tag: str) -> list:
        return [cp.Read(mm2s, f"{tag} mm2s", kind), cp.Read(s2mm, f"{tag} s2mm", kind),
                cp.Read(count, f"{tag} C", kind), cp.Read(lock_o, f"{tag} O", kind)]

    out: list = [cp.Write(q.SRC_ADDR + 4 * j, q.SRC_MARK + j, f"src {j}", kind) for j in range(total)]
    out += [cp.Write(q.DST_ADDR + 4 * j, q.DST_EMPTY, f"dst {j}", kind) for j in range(total)]
    out += snap("s0")
    t = q.TILES[kind]
    recv = [f"%tr = aiex.dma_configure_task({t}, S2MM, 0) {{",
            "  %c_tr_a = arith.constant 1 : i32", "  aie.use_lock(%lG, AcquireGreaterEqual, %c_tr_a)",
            f"  aie.dma_bd(%dst : memref<{total}xi32> offset = 0 len = {total}) {{bd_id = 0 : i32}}",
            "  aie.use_lock(%lD, Release, %c_tr_a)", "  aie.end", "}", "aiex.dma_start_task(%tr)"]
    send = [f"%ts = aiex.dma_configure_task({t}, MM2S, 0) {{"]
    for j in range(CHAIN_K):
        if j:
            send.append(f"^bd{j}:")
        send += [f"  %c_ts{j}_a = arith.constant 1 : i32", f"  aie.use_lock(%lO, AcquireGreaterEqual, %c_ts{j}_a)",
                 f"  aie.dma_bd(%src : memref<{total}xi32> offset = {j * length} len = {length}) "
                 f"{{bd_id = {1 + j} : i32}}",
                 f"  %c_ts{j} = arith.constant 1 : i32", f"  aie.use_lock(%lC, Release, %c_ts{j})",
                 f"  aie.next_bd ^bd{j + 1}" if j + 1 < CHAIN_K else "  aie.end"]
    send += ["}", "aiex.dma_start_task(%ts)"]
    out += [cp.Raw(tuple(recv)), cp.Raw(tuple(send))]
    for i in range(CHAIN_SNAPS):
        out += snap(f"a{i + 1}")
    out += [cp.Write(gate, 1, "open the gate", kind), *snap("b1"), *snap("b2")]
    out += [cp.Read(q.DST_ADDR + 4 * j, f"dst {j}", kind) for j in range(total)]
    return out


def chain_design(kind: str, length: int) -> tuple[str, list, "cp.Layout"]:
    import gen_dma_queue_probe as q
    st = chain_steps(kind, length)
    layout = cp.encode(st)
    t = q.TILES[kind]
    total = CHAIN_K * length
    lines = ["module {", "  aie.device(npu1_1col) {",
             "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}",
             f"    {t} = aie.tile{q.TILE_COORDS[kind]} "
             f"{{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {q.PKT_IDS[kind]}>}}",
             *cp.packet_flows(t, "%t00", kind), *cp.shim_allocations("%t00")]
    # a BD takes one acquire and one release: the MM2S BDs acquire an always-open lock (lO), the
    # S2MM BD releases a spare one (lD) once the gate lets it run
    spare = q.GATE_LOCKS[kind][2:4]
    for sym, lid, init in (("lG", q.GATE_LOCK[kind], 0), ("lC", q.COUNT_LOCK[kind], 0),
                           ("lO", spare[0], CHAIN_K), ("lD", spare[1], 0)):
        lines.append(f'    %{sym} = aie.lock({t}, {lid}) {{init = {init} : i32, sym_name = "{sym}"}}')
    lines.append(f"    aie.flow({t}, DMA : 0, {t}, DMA : 0)")
    for sym, addr in (("src", q.SRC_ADDR), ("dst", q.DST_ADDR)):
        lines.append(f'    %{sym} = aie.buffer({t}) {{address = {addr} : i32, sym_name = "{sym}"}} '
                     f": memref<{total}xi32>")
    lines += [f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<4xi32>) {{",
              *cp.sequence_lines(st, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), st, layout


def chain_generate(out_dir: Path, kinds, lengths) -> None:
    for kind in kinds:
        for length in lengths:
            text, st, layout = chain_design(kind, length)
            d = out_dir / chain_name(kind, length)
            d.mkdir(parents=True, exist_ok=True)
            (d / "aie.mlir").write_text(text)
            (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
            meta = {"case": d.name, "family": "chain", "kind": kind, "length": length, "k": CHAIN_K,
                    "labels": list(layout.labels), "n_words": len(layout.words), "n_data": 4, "host_args": []}
            (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")


def chain_report(d: Path, tag: str) -> str:
    import gen_dma_queue_probe as q
    meta = json.loads((d / "case.json").read_text())
    r = parse(d, tag)
    kind, length, k = meta["kind"], meta["length"], meta["k"]
    lines = [f"{d.name}: {kind} MM2S 0 -> S2MM 0, {k} BDs of {length} words, consumer held"]
    for tag_ in [f"a{i + 1}" for i in range(CHAIN_SNAPS)] + ["b1", "b2"]:
        lines.append(f"  {tag_}: C={r[f'{tag_} C']:2d} O={r.get(f'{tag_} O', -1):2d}  mm2s {q.status_text(r[f'{tag_} mm2s'], kind)}"
                     f"  s2mm {q.status_text(r[f'{tag_} s2mm'], kind)}")
    got = sum(1 for j in range(k * length) if r[f"dst {j}"] == q.SRC_MARK + j)
    lines.append(f"  BDs finished while held: {r['a3 C']}  ({r['a3 C'] * length} words); "
                 f"landed in order after the gate opened: {got}/{k * length}")
    return "\n".join(lines)



# -- sink family: a memtile MM2S chain into a shim S2MM that holds no task -----------------
#
# The of_q0_rich drain path in isolation: the memtile's MM2S 0 sends K BDs of L words through
# the memtile and shim switches to shim S2MM 1, which has no task while the sender settles
# (the drain is armed late). The count lock says how many BDs finished, i.e. how many words
# the path from the memtile staging to the idle shim channel holds. The host then arms the
# S2MM for every word and checks they all land in order.

SINK_WORDS = 80
SINK_LENGTHS = (4, 8, 16)


def sink_name(length: int) -> str:
    return f"sink_mem_L{length}"


def sink_steps(length: int) -> list:
    import gen_dma_queue_probe as q
    kind = "mem"
    k = cp.TileRegs(kind)
    mm2s, count = k.regs["DMA_MM2S_Status_0"], k.value(q.COUNT_LOCK[kind])
    lock_o = k.value(q.GATE_LOCKS[kind][2])
    n = SINK_WORDS // length

    def snap(tag: str) -> list:
        return [cp.Read(mm2s, f"{tag} mm2s", kind), cp.Read(count, f"{tag} C", kind),
                cp.Read(lock_o, f"{tag} O", kind)]

    out: list = [cp.Write(q.SRC_ADDR + 4 * j, q.SRC_MARK + j, f"src {j}", kind) for j in range(SINK_WORDS)]
    out += snap("s0")
    send = [f"%ts = aiex.dma_configure_task({q.TILES[kind]}, MM2S, 0) {{"]
    for j in range(n):
        if j:
            send.append(f"^bd{j}:")
        send += [f"  %c_ts{j}_a = arith.constant 1 : i32", f"  aie.use_lock(%lO, AcquireGreaterEqual, %c_ts{j}_a)",
                 f"  aie.dma_bd(%src : memref<{SINK_WORDS}xi32> offset = {j * length} len = {length}) "
                 f"{{bd_id = {1 + j} : i32}}",
                 f"  %c_ts{j} = arith.constant 1 : i32", f"  aie.use_lock(%lC, Release, %c_ts{j})",
                 f"  aie.next_bd ^bd{j + 1}" if j + 1 < n else "  aie.end"]
    send += ["}", "aiex.dma_start_task(%ts)"]
    out.append(cp.Raw(tuple(send)))
    for i in range(CHAIN_SNAPS):
        out += snap(f"a{i + 1}")
    out.append(cp.Raw((f"aiex.npu.dma_memcpy_nd(%data[0, 0, 0, 0][1, 1, 1, {SINK_WORDS}][0, 0, 0, 1]) "
                       f"{{id = 4 : i64, issue_token = true, metadata = @sink}} : memref<{SINK_WORDS}xi32>",)))
    out.append(cp.Wait("sink"))
    out += snap("b1")
    return out


def sink_design(length: int) -> tuple[str, list, "cp.Layout"]:
    import gen_dma_queue_probe as q
    kind = "mem"
    st = sink_steps(length)
    layout = cp.encode(st)
    t = q.TILES[kind]
    lines = ["module {", "  aie.device(npu1_1col) {",
             "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}",
             f"    {t} = aie.tile{q.TILE_COORDS[kind]} "
             f"{{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {q.PKT_IDS[kind]}>}}",
             *cp.packet_flows(t, "%t00", kind), *cp.shim_allocations("%t00")]
    for sym, lid, init in (("lC", q.COUNT_LOCK[kind], 0), ("lO", q.GATE_LOCKS[kind][2], SINK_WORDS // length)):
        lines.append(f'    %{sym} = aie.lock({t}, {lid}) {{init = {init} : i32, sym_name = "{sym}"}}')
    lines += [f"    aie.flow({t}, DMA : 0, %t00, DMA : 1)",
              "    aie.shim_dma_allocation @sink (%t00, S2MM, 1)",
              f'    %src = aie.buffer({t}) {{address = {q.SRC_ADDR} : i32, sym_name = "src"}} : memref<{SINK_WORDS}xi32>',
              f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{SINK_WORDS}xi32>) {{",
              *cp.sequence_lines(st, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), st, layout


def sink_generate(out_dir: Path, lengths) -> None:
    for length in lengths:
        text, st, layout = sink_design(length)
        d = out_dir / sink_name(length)
        d.mkdir(parents=True, exist_ok=True)
        (d / "aie.mlir").write_text(text)
        (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
        meta = {"case": d.name, "family": "sink", "length": length, "k": SINK_WORDS // length,
                "labels": list(layout.labels), "n_words": len(layout.words), "n_data": SINK_WORDS, "host_args": []}
        (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")


def sink_report(d: Path, tag: str) -> str:
    import gen_dma_queue_probe as q
    meta = json.loads((d / "case.json").read_text())
    r = parse(d, tag)
    length, k = meta["length"], meta["k"]
    lines = [f"{d.name}: memtile MM2S 0 -> idle shim S2MM 1, {k} BDs of {length} words"]
    for tag_ in [f"a{i + 1}" for i in range(CHAIN_SNAPS)] + ["b1"]:
        lines.append(f"  {tag_}: C={r[f'{tag_} C']:2d} O={r[f'{tag_} O']:2d}  mm2s {q.status_text(r[f'{tag_} mm2s'], 'mem')}")
    got = sum(1 for j in range(SINK_WORDS) if r["data"][j] == q.SRC_MARK + j)
    lines.append(f"  BDs finished with the shim channel idle: {r['a3 C']}  ({r['a3 C'] * length} words); "
                 f"landed in order once armed: {got}/{SINK_WORDS}")
    return "\n".join(lines)


# -- race family: a path's capacity measured to the word by a data race -------------------
#
# One MM2S BD of RACE_WORDS words (pattern A) into a consumer that takes nothing: a shim S2MM
# that holds no task, or an S2MM whose one BD is parked on a gate lock. Once the sender has
# settled, the host overwrites the source with pattern B and only then lets the consumer run:
# every word that lands as A had been read from memory before the overwrite, so the A count
# is exactly the words the path holds (the R4-D3 method, which located the egress
# staging). Tiles are named as in TILES (prod = (0,2), consA = (0,3)).

RACE_WORDS = 96
RACE_MARK_B = 0xB0000000
RACE_SRC, RACE_DST = 0x1000, 0x2000
GATE = {"memory": (1, 4), "memory_tile": (42, 45)}    # (gate lock, spare release lock) per module
RACE_CASES = {                                        # name -> (source tile, consumer tile or "shim")
    "race_mem_shim": ("mem", "shim"),
    "race_prod_shim": ("prod", "shim"),
    "race_consA_shim": ("consA", "shim"),
    "race_mem_loop": ("mem", "mem"),
    "race_prod_loop": ("prod", "prod"),
    "race_prod_mem": ("prod", "mem"),
    "race_mem_prod": ("mem", "prod"),
}
# Start-phase variants of the loops: the BD starts PHASE words into a 128-bit granule. The
# staging fills to the largest count <= its depth congruent to -(phase + 1) mod 4 (R4-D3's
# signature), so the four phases pin the depth, not just its granule.
RACE_PHASES = (1, 2, 3)
for _loop in ("race_mem_loop", "race_prod_loop"):
    for _p in RACE_PHASES:
        RACE_CASES[f"{_loop}_p{_p}"] = RACE_CASES[_loop]


# Shim-end cases. `_turn`: the stream goes down into the shim switch and straight back north
# (shim North slave -> North master) to a held memtile S2MM, so it crosses the memtile -> shim
# hop without the shim DMA. `_shimq`: the shim S2MM holds a task parked on a shim lock instead of
# no task. Together they say whether the shim-north +4 lives in the hop or at the shim DMA.
RACE_CASES.update({
    "race_prod_turn": ("prod", "mem"), "race_mem_turn": ("mem", "mem"),
    "race_prod_shimq": ("prod", "shim"), "race_mem_shimq": ("mem", "shim"),
    # the same, channel swapped: the turnaround on channel 3 (the router's pick for the shim
    # cases) and the idle-shim case hand routed on channel 0 (the turnaround's)
    "race_prod_turn3": ("prod", "mem"), "race_prod_shim0": ("prod", "shim"),
})
# Start-phase variants of the shim-entry cases: phase 0 alone only brackets a path's size to its
# granule, the four phases pin it.
for _case in ("race_prod_shim0", "race_prod_shimq"):
    for _p in RACE_PHASES:
        RACE_CASES[f"{_case}_p{_p}"] = RACE_CASES[_case]
SHIM_GATE, SHIM_DONE = 4, 5        # shim locks of the parked S2MM task


def race_base(name: str) -> str:
    """The case name without its start-phase suffix."""
    return re.sub(r"_p\d$", "", name)


def race_turn_switchboxes(src: str, ch: int = 0, into_shim: bool = False) -> list[str]:
    """Circuit src DMA 0 -> ... -> shim North `ch`, hand routed on stream channel `ch` at every hop:
    back north to memtile DMA 0 (a `_turn` case), or into the shim S2MM 1 (South 3, `into_shim`)."""
    mem, shim = VAR["mem"], VAR["shim"]
    back = [] if into_shim else [f"aie.connect<South : {ch}, DMA : 0>"]
    if src == "mem":
        mem_conns = [f"aie.connect<DMA : 0, South : {ch}>", *back]
        out = []
    else:
        mem_conns = [f"aie.connect<North : {ch}, South : {ch}>", *back]
        out = [f"    %sb_src = aie.switchbox({VAR[src]}) {{", f"      aie.connect<DMA : 0, South : {ch}>", "    }"]
    shim_conn = f"aie.connect<North : {ch}, South : 3>" if into_shim else f"aie.connect<North : {ch}, North : {ch}>"
    out += [f"    %sb_mem = aie.switchbox({mem}) {{", *(f"      {c}" for c in mem_conns), "    }",
            f"    %sb_shim = aie.switchbox({shim}) {{", f"      {shim_conn}", "    }"]
    if into_shim:
        out += [f"    %mux_shim = aie.shim_mux({shim}) {{", "      aie.connect<North : 3, DMA : 1>", "    }"]
    return out


def race_route_channel(name: str) -> int | None:
    """Stream channel of a hand-routed case (`_turn` 0, `_turn3` 3, `_shim0` 0), None if auto-routed."""
    m = re.search(r"_(turn|shim)(\d?)$", race_base(name))
    if not m or (m.group(1) == "shim" and not m.group(2)):
        return None
    return int(m.group(2) or 0)


def race_words(name: str) -> int:
    """BD length: a parked shim S2MM holds more than RACE_WORDS on its own."""
    return 2 * RACE_WORDS if race_base(name).endswith("_shimq") else RACE_WORDS


def race_phase(name: str) -> int:
    m = re.search(r"_p(\d)$", name)
    return int(m.group(1)) if m else 0


def race_steps(src: str, dst: str, phase: int = 0, armed: bool = False, n: int = RACE_WORDS) -> list:
    import gen_dma_queue_probe as q
    mm2s = cp.TileRegs(src).regs["DMA_MM2S_Status_0"]
    at = RACE_SRC + 4 * phase
    out: list = [cp.Write(at + 4 * j, q.SRC_MARK + j, f"src {j}", src) for j in range(n)]
    if armed:
        out.append(cp.Raw(("%tq = aiex.dma_configure_task_for @sink {",
                           "  %c_tq = arith.constant 1 : i32", "  aie.use_lock(%lSG, AcquireGreaterEqual, %c_tq)",
                           f"  aie.dma_bd(%data : memref<{n}xi32> offset = 0 len = {n}) {{bd_id = 2 : i32}}",
                           "  aie.use_lock(%lSD, Release, %c_tq)", "  aie.end", "} {issue_token = true}",
                           "aiex.dma_start_task(%tq)")))
    if dst != "shim":
        out.append(cp.Raw((f"%tr = aiex.dma_configure_task({VAR[dst]}, S2MM, 0) {{",
                           "  %c_tr = arith.constant 1 : i32", "  aie.use_lock(%lG, AcquireGreaterEqual, %c_tr)",
                           f"  aie.dma_bd(%dst : memref<{n}xi32> offset = 0 len = {n}) {{bd_id = 0 : i32}}",
                           "  aie.use_lock(%lD, Release, %c_tr)", "  aie.end", "}", "aiex.dma_start_task(%tr)")))
    out.append(cp.Raw((f"%ts = aiex.dma_configure_task({VAR[src]}, MM2S, 0) {{",
                       f"  aie.dma_bd(%src : memref<{n + 4}xi32> offset = {phase} len = {n}) {{bd_id = 1 : i32}}",
                       "  aie.end", "}", "aiex.dma_start_task(%ts)")))
    out += [cp.Read(mm2s, f"a{i + 1} mm2s", src) for i in range(CHAIN_SNAPS)]
    out += [cp.Write(at + 4 * j, RACE_MARK_B + j, f"over {j}", src) for j in range(n)]
    out.append(cp.Read(mm2s, "o mm2s", src))
    if armed:
        out.append(cp.Raw(("%c_open = arith.constant 1 : i32", "aiex.set_lock(%lSG, %c_open)",
                           "aiex.dma_await_task(%tq)")))
    elif dst == "shim":
        out.append(cp.Raw((f"aiex.npu.dma_memcpy_nd(%data[0, 0, 0, 0][1, 1, 1, {n}][0, 0, 0, 1]) "
                           f"{{id = 4 : i64, issue_token = true, metadata = @sink}} : memref<{n}xi32>",)))
        out.append(cp.Wait("sink"))
    else:
        k = cp.TileRegs(dst)
        out.append(cp.Write(k.value(GATE[TILES[dst][1]][0]), 1, "open the gate", dst))
        out += [cp.Read(mm2s, "b0 mm2s", src)] * 2
        out += [cp.Read(RACE_DST + 4 * j, f"dst {j}", dst) for j in range(n)]
    out.append(cp.Read(mm2s, "b1 mm2s", src))
    return out


def race_design(name: str) -> tuple[str, list, "cp.Layout"]:
    src, dst = RACE_CASES[name]
    armed = race_base(name).endswith("_shimq")
    n = race_words(name)
    st = race_steps(src, dst, race_phase(name), armed, n)
    layout = cp.encode(st)
    shim = VAR["shim"]
    tiles = ["shim", src] + ([dst] if dst not in ("shim", src) else [])
    if race_route_channel(name) is not None and "mem" not in tiles:
        tiles.append("mem")                    # a hand-routed circuit names the memtile's switchbox
    lines = ["module {", "  aie.device(npu1_1col) {"]
    for t in tiles:
        (c, r), _m, pkt, _resp = TILES[t]
        lines.append(f"    {VAR[t]} = aie.tile({c}, {r}) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {pkt}>}}")
    for t in tiles[1:]:
        _xy, _m, pkt, resp = TILES[t]
        lines += [f"    aie.packet_flow({pkt}) {{", f"      aie.packet_source<{shim}, DMA : 0>",
                  f"      aie.packet_dest<{VAR[t]}, TileControl : 0>", "    }",
                  f"    aie.packet_flow({resp}) {{", f"      aie.packet_source<{VAR[t]}, TileControl : 0>",
                  f"      aie.packet_dest<{shim}, DMA : 0>", "    }"]
    lines += [f"    aie.shim_dma_allocation @ctrlin0 ({shim}, MM2S, 0)",
              f"    aie.shim_dma_allocation @ctrl0 ({shim}, S2MM, 0)",
              f'    %src = aie.buffer({VAR[src]}) {{address = {RACE_SRC} : i32, sym_name = "src"}} : memref<{n + 4}xi32>']
    if dst == "shim":
        ch = race_route_channel(name)
        lines += [*([f"    aie.flow({VAR[src]}, DMA : 0, {shim}, DMA : 1)"] if ch is None
                    else race_turn_switchboxes(src, ch, into_shim=True)),
                  f"    aie.shim_dma_allocation @sink ({shim}, S2MM, 1)"]
        if armed:
            lines += [f'    %lSG = aie.lock({shim}, {SHIM_GATE}) {{init = 0 : i32, sym_name = "lSG"}}',
                      f'    %lSD = aie.lock({shim}, {SHIM_DONE}) {{init = 0 : i32, sym_name = "lSD"}}']
    else:
        gate, spare = GATE[TILES[dst][1]]
        lines += [f'    %lG = aie.lock({VAR[dst]}, {gate}) {{init = 0 : i32, sym_name = "lG"}}',
                  f'    %lD = aie.lock({VAR[dst]}, {spare}) {{init = 0 : i32, sym_name = "lD"}}',
                  f'    %dst = aie.buffer({VAR[dst]}) {{address = {RACE_DST} : i32, sym_name = "dst"}} : memref<{n}xi32>',
                  *(race_turn_switchboxes(src, race_route_channel(name)) if race_route_channel(name) is not None
                    else [f"    aie.flow({VAR[src]}, DMA : 0, {VAR[dst]}, DMA : 0)"])]
    lines += [f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{n}xi32>) {{",
              *cp.sequence_lines(st, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), st, layout


def race_generate(out_dir: Path, names) -> None:
    for name in names or RACE_CASES:
        text, st, layout = race_design(name)
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "aie.mlir").write_text(text)
        (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
        src, dst = RACE_CASES[name]
        n = race_words(name)
        meta = {"case": name, "family": "race", "src": src, "dst": dst, "phase": race_phase(name),
                "labels": list(layout.labels), "n_words": len(layout.words), "n_data": n, "host_args": []}
        (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")


def race_report(d: Path, tag: str) -> str:
    import gen_dma_queue_probe as q
    meta = json.loads((d / "case.json").read_text())
    r = parse(d, tag)
    src, dst, n = meta["src"], meta["dst"], meta["n_data"]
    landed = r["data"] if dst == "shim" else [r[f"dst {j}"] for j in range(n)]
    old = sum(1 for j in range(n) if landed[j] == q.SRC_MARK + j)
    new = sum(1 for j in range(n) if landed[j] == RACE_MARK_B + j)
    prefix = next((j for j in range(n) if landed[j] != q.SRC_MARK + j), n)
    kind = "mem" if src == "mem" else "core"
    return (f"{d.name}: {src} MM2S 0 -> {dst} ({'idle shim S2MM 1' if dst == 'shim' else 'S2MM 0 held'}): "
            f"read before the overwrite {old} (prefix {prefix}), after {new}, other {n - old - new}; "
            f"held mm2s {q.status_text(r['a3 mm2s'], kind)}")

def main(argv=None) -> int:
    import gen_lock_stall_probe as lock
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("out_dir")
    gen.add_argument("--chain", nargs="*", metavar="KIND", help="the chain family for these tile kinds (core, mem)")
    gen.add_argument("--sink", action="store_true", help="the sink family (memtile MM2S into an idle shim S2MM)")
    gen.add_argument("--race", nargs="*", metavar="CASE", help="the race family (no names: every case)")
    gen.add_argument("--lengths", type=int, nargs="*", default=None)
    sub.add_parser("build").add_argument("out_dir")
    run = sub.add_parser("run")
    run.add_argument("out_dir")
    run.add_argument("tag")
    run.add_argument("--backend", choices=("hw", "emu"), required=True)
    run.add_argument("--runtime", choices=("debug", "release"), default="release")
    an = sub.add_parser("analyze")
    an.add_argument("out_dir")
    an.add_argument("tag")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        if args.race is not None:
            race_generate(Path(args.out_dir), args.race)
        elif args.sink:
            sink_generate(Path(args.out_dir), args.lengths or SINK_LENGTHS)
        elif args.chain:
            chain_generate(Path(args.out_dir), args.chain, args.lengths or CHAIN_LENGTHS)
        else:
            write_case(Path(args.out_dir) / CASE, None)
        return 0
    if args.cmd == "build":
        if (Path(args.out_dir) / CASE).exists():
            return cmd_build(args)
        return lock.frozen_build(argparse.Namespace(out_dir=args.out_dir))
    if args.cmd == "run":
        return lock.frozen_run(argparse.Namespace(out_dir=args.out_dir, tag=args.tag, backend=args.backend,
                                                  runtime=args.runtime, only=[]))
    root = Path(args.out_dir)
    if (root / CASE).exists():
        print(report(parse(root / CASE, args.tag)))
    for d in sorted(root.glob("chain_*"), key=lambda p: (p.name.split("_L")[0], int(p.name.split("_L")[1]))):
        if (d / "runs" / args.tag / "out.bin").exists():
            print(chain_report(d, args.tag))
    for d in sorted(root.glob("sink_*"), key=lambda p: int(p.name.split("_L")[1])):
        if (d / "runs" / args.tag / "out.bin").exists():
            print(sink_report(d, args.tag))
    for d in sorted(root.glob("race_*")):
        if (d / "runs" / args.tag / "out.bin").exists():
            print(race_report(d, args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
