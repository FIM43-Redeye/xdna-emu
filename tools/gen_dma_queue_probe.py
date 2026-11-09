#!/usr/bin/env python3
"""DMA task-queue probes: how deep is a channel's task queue, and what does a full queue do?

One design per (tile kind, direction). A channel of the tile under test gets up to PUSHES tasks
through its Start_Queue, one BD each; the first BD acquires a lock the host keeps at 0, so the
channel parks on it and every later push sits in the queue. The channel status (Task_Queue_Size,
Task_Queue_Overflow, Channel_Running, the stall bits) is read over a control packet after every
push. The host then opens the gate (the lock gets a large count) and watches the channel drain:
every BD that runs releases a counter lock and moves WORDS words, so the counter says how many
tasks ran and the destination words say which.

  shim_s2mm   shim S2MM 1 tasks, data from the shim's own MM2S 1 (a stream loop through its switch)
  shim_mm2s   shim MM2S 1 tasks, received by one plain S2MM 1 BD
  mem_*       memtile channel 0 of each direction, a loop through the memtile's switch
  core_*      compute tile channel 0 of each direction, likewise

A second family, `absorb`, asks where a stream's words sit when its consumer is held: the shim's MM2S 1
sends WORDS words through the switch to its own S2MM 1, whose one task is parked on a lock (or which has
no task at all, `idle`). The status of both channels is read after the sender has had time to run: an
MM2S that reads idle absorbed the whole transfer; one that reads Running with Stalled_Stream_Backpressure
did not. Sweeping the length finds the capacity.

Registers come from the AM025 database (ctrlpkt_readback.TileRegs); the BDs are written by the
runtime sequence (aiex.dma_configure_task), so a push is exactly what aie-rt's queue code does:
a BD write, then a Start_Queue write.

    gen_dma_queue_probe.py generate OUT [--cases ...] [--pushes N]
    gen_dma_queue_probe.py build OUT
    gen_dma_queue_probe.py run OUT TAG --backend hw|emu [--only ...]
    gen_dma_queue_probe.py analyze OUT TAG [--only ...]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import ctrlpkt_readback as cp  # noqa: E402

PUSHES = 9
WORDS = 2                                  # words one task moves
GATE_OPEN = 40                             # the gate lock's count once the host opens it
SRC_MARK, DST_EMPTY = 0xD0000000, 0xEEEEEEEE
SRC_ADDR, DST_ADDR = 0x1000, 0x2000        # tile-local buffer addresses (memtile, compute tile)
OVERFLOW_BIT = 1 << 18
# Time for the channel to drain after the gate opens: reads of an unrelated register.
SETTLE_SNAPS = 3

TILES = {"shim": "%t00", "mem": "%t01", "core": "%t02"}
TILE_COORDS = {"shim": (0, 0), "mem": (0, 1), "core": (0, 2)}
PKT_IDS = {"shim": 4, "mem": 6, "core": 5}
CHANNEL = {"shim": 1, "mem": 0, "core": 0}            # shim channel 0 carries the control packets
GATE_LOCKS = {"shim": (13, 11, 10, 9, 8, 7, 6, 5), "mem": (42, 44, 45, 46, 47, 48, 49, 50),
              "core": (1, 3, 4, 5, 6, 7, 8, 9)}   # one per chained BD
GATE_LOCK = {kind: locks[0] for kind, locks in GATE_LOCKS.items()}
DATA_CHECK_WORDS = 32                      # beyond this many words the memtile/compute tile data is not read back
COUNT_LOCK = {"shim": 12, "mem": 43, "core": 2}
# Shim BD ids 0..3 and 15 belong to the control packets; task BDs follow, then the partner's BD.
FIRST_BD = {"shim": 4, "mem": 0, "core": 0}
PARTNER_BD = {"shim": 14, "mem": 20, "core": 14}


@dataclass(frozen=True)
class QueueCase:
    kind: str                              # "shim" | "mem" | "core": the tile under test
    direction: str                         # "s2mm" | "mm2s": the channel whose queue is filled
    pushes: int = PUSHES
    gated: bool = True                     # False: no lock; the S2MM channel parks on stream starvation
    words: int = WORDS                     # words one BD moves
    chain: int = 1                         # BDs per task
    closed: int = 0                        # chain: index of the BD whose gate stays shut (the others start open)
    partner: bool = True                   # False: the other end of the loop never starts (a starved S2MM stays parked)

    @property
    def name(self) -> str:
        return (f"{self.kind}_{self.direction}" + ("" if self.gated else "_starve")
                + (f"_w{self.words}" if self.words != WORDS else "")
                + (f"_c{self.chain}k{self.closed}" if self.chain > 1 else "")
                + ("" if self.partner else "_np"))

    @property
    def span(self) -> int:
        """Words the partner BD moves: every BD's share."""
        return self.pushes * self.chain * self.words

    @property
    def checks_data(self) -> bool:
        return self.kind == "shim" or self.span <= DATA_CHECK_WORDS

    @property
    def data_words(self) -> int:
        """The host data BO: shim sources then destinations; the others do not use it."""
        return 2 * self.span if self.kind == "shim" else 4


def cases(pushes: int = PUSHES) -> dict[str, QueueCase]:
    out = [QueueCase(k, d, pushes) for k in TILES for d in ("s2mm", "mm2s")]
    # the same pipeline with the first task parked on its data instead of a lock
    out += [QueueCase(k, "s2mm", pushes, gated=False) for k in TILES]
    return {c.name: c for c in out}


def stall_cases() -> dict[str, QueueCase]:
    """One task parked at one point: BD `closed` of a chain of `chain` BDs (all but that gate open),
    and one long BD (`words`) parked on its lock."""
    out = []
    for k in TILES:
        for d in ("s2mm", "mm2s"):
            out += [QueueCase(k, d, 1, True, WORDS, n, c) for n, c in ((2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 2))]
            out += [QueueCase(k, d, 1, True, w) for w in (64, 128, 160, 256)]
    return {c.name: c for c in out}


def status_register(case: QueueCase) -> str:
    return f"DMA_{case.direction.upper()}_Status_{CHANNEL[case.kind]}"


def start_queue_register(case: QueueCase) -> str:
    return f"DMA_{case.direction.upper()}_{CHANNEL[case.kind]}_Start_Queue"


def _buffers(case: QueueCase) -> tuple[str, str]:
    """(source, destination) operand text of a dma_bd, up to the offset."""
    if case.kind == "shim":
        return "%data", "%data"
    return "%src", "%dst"


def _bd(case: QueueCase, role: str, first_word: int, n: int, bd_id: int) -> str:
    """A dma_bd that reads the source (role "src") or writes the destination ("dst")."""
    name = _buffers(case)[0 if role == "src" else 1]
    if case.kind == "shim":
        total = case.data_words
        first_word += 0 if role == "src" else case.span
    else:
        total = case.span
    return f"aie.dma_bd({name} : memref<{total}xi32> offset = {first_word} len = {n}) {{bd_id = {bd_id} : i32}}"


def _task(case: QueueCase, var: str, direction: str, bds: list[tuple[str, int | None]]) -> list[str]:
    """One task: a chain of BDs. Each is (dma_bd text, index of its gate lock | None for none)."""
    kind, ch = case.kind, CHANNEL[case.kind]
    out = [f"%{var} = aiex.dma_configure_task({TILES[kind]}, {direction.upper()}, {ch}) {{"]
    for j, (bd, gate) in enumerate(bds):
        if j:
            out.append(f"^bd{j}:")
        if gate is not None:
            out += [f"  %c_{var}{j}_a = arith.constant 1 : i32",
                    f"  aie.use_lock(%{gate_symbol(gate)}, AcquireGreaterEqual, %c_{var}{j}_a)"]
        out.append(f"  {bd}")
        if gate is not None:
            out += [f"  %c_{var}{j}_r = arith.constant 1 : i32",
                    f"  aie.use_lock(%lC, Release, %c_{var}{j}_r)"]
        out.append(f"  aie.next_bd ^bd{j + 1}" if j + 1 < len(bds) else "  aie.end")
    out.append("}")
    return out


def gate_symbol(j: int) -> str:
    return "lG" if j == 0 else f"lG{j}"


def push_lines(case: QueueCase, i: int) -> tuple[str, ...]:
    """Configure and start the i-th task on the channel under test."""
    role = "dst" if case.direction == "s2mm" else "src"
    bds = []
    for j in range(case.chain):
        n = i * case.chain + j
        bd = _bd(case, role, n * case.words, case.words, FIRST_BD[case.kind] + n)
        bds.append((bd, j if case.gated else None))
    return (*_task(case, f"t{i}", case.direction, bds), f"aiex.dma_start_task(%t{i})")


def partner_lines(case: QueueCase) -> tuple[str, ...]:
    """The plain BD on the other channel of the loop: it sends (or receives) every task's words."""
    role, direction = ("src", "mm2s") if case.direction == "s2mm" else ("dst", "s2mm")
    bd = _bd(case, role, 0, case.span, PARTNER_BD[case.kind])
    return (*_task(case, "tp", direction, [(bd, None)]), "aiex.dma_start_task(%tp)")


def steps(case: QueueCase) -> list:
    k = cp.TileRegs(case.kind)
    status = k.regs[status_register(case)]
    gate = k.value(GATE_LOCKS[case.kind][case.closed])
    count = k.value(COUNT_LOCK[case.kind])
    tile = case.kind

    def snap(tag: str) -> list:
        return [cp.Read(status, f"{tag} status", tile), cp.Read(count, f"{tag} C", tile),
                cp.Read(gate, f"{tag} G", tile)]

    out: list = []
    if case.kind != "shim" and case.checks_data:
        out += [cp.Write(SRC_ADDR + 4 * j, SRC_MARK + j, f"src {j}", tile) for j in range(case.span)]
        out += [cp.Write(DST_ADDR + 4 * j, DST_EMPTY, f"dst {j}", tile) for j in range(case.span)]
    # the other chained BDs start with their gates open
    out += [cp.Write(k.value(GATE_LOCKS[case.kind][j]), GATE_OPEN, f"open gate {j}", tile)
            for j in range(case.chain) if j != case.closed and case.gated]
    out += snap("s0")
    for i in range(case.pushes):
        out.append(cp.Raw(push_lines(case, i)))
        out.append(cp.Read(status, f"push{i + 1} status", tile))
    if case.partner:
        out.append(cp.Raw(partner_lines(case)))
    out += snap("q")
    out.append(cp.Write(gate, GATE_OPEN, "open the gate", tile))
    for i in range(SETTLE_SNAPS):
        out += snap(f"r{i + 1}")
    if case.kind != "shim" and case.checks_data:
        out += [cp.Read(DST_ADDR + 4 * j, f"dst {j}", tile) for j in range(case.span)]
    out += [cp.Write(status, OVERFLOW_BIT, "clear the overflow flag", tile), *snap("z")]
    return out


def design(case: QueueCase) -> tuple[str, list, "cp.Layout"]:
    st = steps(case)
    layout = cp.encode(st)
    kind = case.kind
    tile = TILES[kind]
    lines = ["module {", "  aie.device(npu1_1col) {",
             "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}"]
    if kind != "shim":
        lines.append(f"    {tile} = aie.tile{TILE_COORDS[kind]} "
                     f"{{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {PKT_IDS[kind]}>}}")
    lines += cp.packet_flows(tile, "%t00", kind)
    lines += cp.shim_allocations("%t00")
    for sym, lid in (*[(gate_symbol(j), GATE_LOCKS[kind][j]) for j in range(case.chain if case.gated else 1)],
                     ("lC", COUNT_LOCK[kind])):
        lines.append(f'    %{sym} = aie.lock({tile}, {lid}) {{init = 0 : i32, sym_name = "{sym}"}}')
    ch = CHANNEL[kind]
    lines.append(f"    aie.flow({tile}, DMA : {ch}, {tile}, DMA : {ch})")
    if kind == "shim":
        lines += [f"    aie.shim_dma_allocation @dout (%t00, MM2S, {ch})",
                  f"    aie.shim_dma_allocation @din (%t00, S2MM, {ch})"]
    else:
        for sym, addr in (("src", SRC_ADDR), ("dst", DST_ADDR)):
            lines.append(f'    %{sym} = aie.buffer({tile}) {{address = {addr} : i32, sym_name = "{sym}"}} '
                         f": memref<{case.span}xi32>")
    lines += [f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{case.data_words}xi32>) {{",
              *cp.sequence_lines(st, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), st, layout


# -- absorb family -------------------------------------------------------------------

ABSORB_LENGTHS = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256)
ABSORB_SNAPS = 3                           # status reads after the sender starts (time passes between)


@dataclass(frozen=True)
class AbsorbCase:
    words: int
    parked: bool = True                    # False: the S2MM channel gets no task at all

    @property
    def name(self) -> str:
        return f"absorb{'' if self.parked else '_idle'}_{self.words}"

    @property
    def data_words(self) -> int:
        return 2 * self.words


def absorb_cases(lengths=ABSORB_LENGTHS) -> dict[str, AbsorbCase]:
    return {c.name: c for c in (AbsorbCase(n, p) for p in (True, False) for n in lengths)}


def absorb_steps(case: AbsorbCase) -> list:
    k = cp.TileRegs("shim")
    s2mm, mm2s = k.regs["DMA_S2MM_Status_1"], k.regs["DMA_MM2S_Status_1"]
    gate, count = k.value(GATE_LOCK["shim"]), k.value(COUNT_LOCK["shim"])

    def snap(tag: str) -> list:
        return [cp.Read(mm2s, f"{tag} mm2s", "shim"), cp.Read(s2mm, f"{tag} s2mm", "shim"),
                cp.Read(count, f"{tag} C", "shim")]

    out = snap("s0")
    if case.parked:
        out.append(cp.Raw(_absorb_task_lines(case, "tr", "s2mm", True)))
    out.append(cp.Raw(_absorb_task_lines(case, "ts", "mm2s", False)))
    for i in range(ABSORB_SNAPS):
        out += snap(f"a{i + 1}")
    if case.parked:
        out += [cp.Write(gate, GATE_OPEN, "open the gate", "shim"), *snap("b1"), *snap("b2")]
    return out


def _absorb_task_lines(case: AbsorbCase, var: str, direction: str, gated: bool) -> tuple[str, ...]:
    n = case.words
    offset = n if direction == "s2mm" else 0
    bd = (f"aie.dma_bd(%data : memref<{case.data_words}xi32> offset = {offset} len = {n}) "
          f"{{bd_id = {FIRST_BD['shim'] if gated else PARTNER_BD['shim']} : i32}}")
    out = [f"%{var} = aiex.dma_configure_task(%t00, {direction.upper()}, 1) {{"]
    if gated:
        out += [f"  %c_{var}_a = arith.constant 1 : i32",
                f"  aie.use_lock(%lG, AcquireGreaterEqual, %c_{var}_a)"]
    out.append(f"  {bd}")
    if gated:
        out += [f"  %c_{var}_r = arith.constant 1 : i32", f"  aie.use_lock(%lC, Release, %c_{var}_r)"]
    out += ["  aie.end", "}", f"aiex.dma_start_task(%{var})"]
    return tuple(out)


def absorb_design(case: AbsorbCase) -> tuple[str, list, "cp.Layout"]:
    st = absorb_steps(case)
    layout = cp.encode(st)
    lines = ["module {", "  aie.device(npu1_1col) {",
             "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}",
             *cp.packet_flows("%t00", "%t00", "shim"), *cp.shim_allocations("%t00")]
    for sym, lid in (("lG", GATE_LOCK["shim"]), ("lC", COUNT_LOCK["shim"])):
        lines.append(f'    %{sym} = aie.lock(%t00, {lid}) {{init = 0 : i32, sym_name = "{sym}"}}')
    lines += ["    aie.flow(%t00, DMA : 1, %t00, DMA : 1)",
              "    aie.shim_dma_allocation @dout (%t00, MM2S, 1)",
              "    aie.shim_dma_allocation @din (%t00, S2MM, 1)",
              f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{case.data_words}xi32>) {{",
              *cp.sequence_lines(st, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), st, layout


def absorb_report(case: AbsorbCase, ctrl: dict[str, int], data: list[int]) -> str:
    lines = [f"{case.name}: MM2S 1 -> S2MM 1 on the shim, {case.words} words, "
             f"S2MM {'parked on a lock' if case.parked else 'without a task'}"]
    for tag in [f"a{i + 1}" for i in range(ABSORB_SNAPS)] + (["b1", "b2"] if case.parked else []):
        lines.append(f"  {tag}: mm2s {status_text(ctrl[f'{tag} mm2s'], 'shim')}")
        lines.append(f"      s2mm {status_text(ctrl[f'{tag} s2mm'], 'shim')}  C={ctrl[f'{tag} C']}")
    got = sum(1 for i in range(case.words) if data[case.words + i] == SRC_MARK + i)
    lines.append(f"  words landed in the destination: {got} of {case.words}")
    return "\n".join(lines)


# -- results -------------------------------------------------------------------------

STATUS_FIELDS = (("cur_bd", 24, 0xF), ("queue", 20, 0x7), ("running", 19, 1), ("overflow", 18, 1),
                 ("tct", 5, 1), ("starve", 4, 1), ("rel", 3, 1), ("acq", 2, 1), ("fsm", 0, 3))


def decode_status(word: int, kind: str) -> dict[str, int]:
    """The status fields this probe reads (Cur_BD is 4 bits except on the memtile, where it is 6)."""
    f = {n: (word >> lsb) & mask for n, lsb, mask in STATUS_FIELDS}
    if kind == "mem":
        f["cur_bd"] = (word >> 24) & 0x3F
    return f


def status_text(word: int, kind: str) -> str:
    f = decode_status(word, kind)
    stalls = "".join(c for c, n in (("A", "acq"), ("R", "rel"), ("S", "starve"), ("T", "tct")) if f[n])
    return (f"{word:#010x} q={f['queue']} run={f['running']} ovf={f['overflow']} fsm={f['fsm']} "
            f"bd={f['cur_bd']} stall={stalls or '-'}")


def parse_output(out: bytes, labels: list[str], data_words: int) -> tuple[dict[str, int], list[int]]:
    n = len(labels)
    words = struct.unpack(f"<{n + data_words}I", out[:4 * (n + data_words)])
    return dict(zip(labels, words[:n])), list(words[n:])


def report(case: QueueCase, ctrl: dict[str, int], data: list[int]) -> str:
    kind = case.kind
    lines = [f"{case.name}: {case.pushes} pushes, {WORDS} words each"]
    for i in range(case.pushes):
        lines.append(f"  push {i + 1}: {status_text(ctrl[f'push{i + 1} status'], kind)}")
    for tag in ("s0", "q", "r1", "r2", "r3", "z"):
        lines.append(f"  {tag:3s}: {status_text(ctrl[f'{tag} status'], kind)}  "
                     f"C={ctrl[f'{tag} C']} G={ctrl[f'{tag} G']}")
    if case.checks_data:
        dst = data[case.span:] if kind == "shim" else [ctrl[f"dst {j}"] for j in range(case.span)]
        # which source word landed in which slot (-1: untouched)
        placed = [(w - SRC_MARK) if (w & 0xFFF00000) == SRC_MARK else -1 for w in dst]
        lines.append("  landed (source index per destination word, -1 untouched): " + " ".join(map(str, placed)))
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------------

def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    table = cases(args.pushes) | stall_cases() | absorb_cases(args.lengths)
    for name in args.cases or cases(args.pushes):
        case = table[name]
        text, st, layout = absorb_design(case) if isinstance(case, AbsorbCase) else design(case)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "aie.mlir").write_text(text)
        (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
        meta = {"case": name, "labels": list(layout.labels), "n_words": len(layout.words),
                "n_data": case.data_words, "host_args": ["seq"]}
        if isinstance(case, AbsorbCase):
            meta |= {"family": "absorb", "words": case.words, "parked": case.parked}
        else:
            meta |= {"family": "queue", "kind": case.kind, "direction": case.direction,
                     "pushes": case.pushes, "gated": case.gated, "words": case.words,
                     "chain": case.chain, "closed": case.closed, "partner": case.partner}
        (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


def cmd_analyze(args) -> int:
    root = Path(args.out_dir)
    table = cases(args.pushes)
    for d in sorted(p for p in root.iterdir() if (p / "case.json").exists()):
        if args.only and d.name not in args.only:
            continue
        meta = json.loads((d / "case.json").read_text())
        out = d / "runs" / args.tag / "out.bin"
        if not out.exists():
            print(f"{d.name}: (no output)")
            continue
        ctrl, data = parse_output(out.read_bytes(), meta["labels"], meta["n_data"])
        if meta["family"] == "absorb":
            print(absorb_report(AbsorbCase(meta["words"], meta["parked"]), ctrl, data))
            continue
        case = QueueCase(meta["kind"], meta["direction"], meta["pushes"], meta.get("gated", True),
                         meta.get("words", WORDS), meta.get("chain", 1), meta.get("closed", 0),
                         meta.get("partner", True))
        print(report(case, ctrl, data))
    return 0


def main(argv=None) -> int:
    import gen_lock_stall_probe as lock
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("out_dir")
    gen.add_argument("--cases", nargs="*", default=[])
    gen.add_argument("--pushes", type=int, default=PUSHES)
    gen.add_argument("--lengths", type=int, nargs="*", default=list(ABSORB_LENGTHS),
                     help="absorb family: transfer lengths in words")
    sub.add_parser("build").add_argument("out_dir")
    run = sub.add_parser("run")
    run.add_argument("out_dir")
    run.add_argument("tag")
    run.add_argument("--backend", choices=("hw", "emu"), required=True)
    run.add_argument("--only", nargs="*", default=[])
    run.add_argument("--runtime", choices=("debug", "release"), default="release")
    an = sub.add_parser("analyze")
    an.add_argument("out_dir")
    an.add_argument("tag")
    an.add_argument("--only", nargs="*", default=[])
    an.add_argument("--pushes", type=int, default=PUSHES)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return lock.frozen_build(Namespace(out_dir=args.out_dir))
    if args.cmd == "run":
        return lock.frozen_run(args)
    return cmd_generate(args) if args.cmd == "generate" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
