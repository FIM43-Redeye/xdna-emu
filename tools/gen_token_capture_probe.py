#!/usr/bin/env python3
"""Task-complete token capture: the raw token words a tile emits, landed in a host buffer.

When a DMA task is pushed with Enable_Token_Issue set, the tile emits one word (the task-complete
token) on its tile-control stream port. Normally a packet flow carries it to the shim's South port
and the device's management controller consumes it. Here a packet flow routes it to a shim S2MM
channel instead, so the word lands in a host buffer and its fields (column, row, the channel's
actor identifier, parity, ...) can be read off silicon.

The routing is the one upstream's control-packet tests use for a tile's TileControl port
(mlir-aie test/npu-xrt/add_one_ctrl_packet: `packet_source<tile, TileControl : 0>` to
`packet_dest<shim, DMA : 0>`), with `keep_pkt_header = true` (as upstream's token flows to South
have it) so the header-only token packet is not stripped at the DMA. The token issue is hand
written the way upstream's `writebd_tokens` tests do it: a Controller_ID write to the channel's
Ctrl register, then a Start_Queue write with bit 31 set. The Controller_ID is the packet id of
the token, so each measured channel gets its own id and its own flow: a captured word is
attributed to its channel by the id the probe programmed, not by any field being measured.

Every measured task is a loop inside one tile: MM2S i feeds S2MM i through the tile's switch,
and only the measured end issues a token. Nothing waits on a diverted token (it never reaches the
controller): after each pair the sequence waits on a shim loop (shim MM2S 1 -> S2MM 1) whose
tokens go the normal way, started after the pair and much longer, so the pair has finished and
its token has been sent before the sequence moves on.

Termination, by construction. The capture S2MM gets a BD several times larger than any word count
the designs can produce and is never waited on, like the trace carrier of every traced bridge test:
its task does not finish and nothing depends on it, so no count of tokens, fewer or more, can stall
the run. All waits are on the shim loop whose tokens go the normal way. A shim S2MM writes host
memory in whole bursts (64 bytes on silicon per the trace work, 256 in the emulator's BD model) and never writes the trailing partial burst of an unfinished BD, so
after the last diverted token the sequence sends a recognisable filler (64 words, its own packet id)
from shim MM2S 0 into the capture channel and waits for that MM2S the normal way; the filler pushes
every token word into a completed burst. The decoder reads token words up to the first filler word
and reports how many it found per probe.

    gen_token_capture_probe.py list
    gen_token_capture_probe.py generate OUT [design ..]
    gen_token_capture_probe.py build OUT
    gen_token_capture_probe.py run OUT TAG --backend hw|emu [--only ..]
    gen_token_capture_probe.py decode OUT TAG [--backend hw|emu] [--only ..]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import ctrlpkt_readback as cp  # noqa: E402

# -- the fixed frame ------------------------------------------------------------------------

FILLER_ID = 2                                     # packet id of the filler MM2S -> capture S2MM
SHIM_CONTROLLER_ID = 4                            # the shim's own (normal-route) token id, as in the lock probes
SHIM1_CONTROLLER_ID = 6                           # shim (1, 0) in the two-column design
TOKEN_ID_BASE = 8                                 # probe token ids: 8 .. 31 (Controller_ID is 5 bits)
CONTROLLER_ID_MASK = 0x1F00                       # what mlir-aie's push lowering masks
CONTROLLER_ID_SHIFT = 8
ENABLE_TOKEN_ISSUE = 1 << 31

PAIR_WORDS = 16                                   # words one measured pair moves
MARKER_WORDS = 64                                 # the shim loop that follows every pair
FILLER_WORDS = 64                                 # a whole 256-byte burst (the emulator's shim BD burst); the tail may stay unwritten
FILLER_MARK = 0xF1110000                          # filler data word j is FILLER_MARK + j
HOST_MARK = 0xD0000000                            # every other host input word i is HOST_MARK + i
CAPTURE_WORDS = 256                               # the capture BD: 1 KiB, never finishes, never awaited
TILE_SRC_ADDR, TILE_DST_ADDR = 0x1000, 0x2000    # tile-local buffers of the loops (memtile, compute tile)
MARKER_BD_IN, MARKER_BD_OUT, FILLER_BD, CAPTURE_BD = 5, 6, 3, 15
MEM_ODD_CHANNEL_BD_BASE = 24                      # memtile: odd channels reach only BD ids 24..47

# kind -> (AM025 module, row, S2MM channels, MM2S channels)
KINDS = {"core": ("memory", 2, 2, 2), "mem": ("memory_tile", 1, 6, 6), "shim": ("shim", 0, 2, 2)}


# -- schema --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    """One measured channel: the end of a tile-local loop that issues a diverted token."""
    kind: str                # "core" | "mem" | "shim"
    col: int                 # mlir-aie column
    row: int
    direction: str           # "s2mm" | "mm2s"
    index: int               # channel number within the direction
    token_id: int

    @property
    def tile(self) -> str:
        return f"%t{self.col}{self.row}"

    @property
    def name(self) -> str:
        return f"{self.kind}{self.col}_{self.row}_{self.direction}{self.index}"


@dataclass(frozen=True)
class Design:
    name: str
    device: str
    probes: tuple[Probe, ...]                 # in issue order; a listed twice means two tasks
    suppressed: frozenset[int] = frozenset()  # probe indexes whose task issues no token (the (b) variants)

    @property
    def columns(self) -> int:
        return 1 + max(p.col for p in self.probes)

    @property
    def capture_words(self) -> int:
        return CAPTURE_WORDS

    @property
    def shim_probes(self) -> int:
        return sum(p.kind == "shim" for p in self.probes)

    @property
    def filler_offset(self) -> int:
        return MARKER_WORDS + PAIR_WORDS * self.shim_probes

    @property
    def host_in_words(self) -> int:
        return self.filler_offset + FILLER_WORDS

    @property
    def host_out_words(self) -> int:
        return self.filler_offset

    def expected(self) -> dict[int, int]:
        """Token words expected per token id if every issuing task sends one word."""
        out: dict[int, int] = {}
        for k, p in enumerate(self.probes):
            out[p.token_id] = out.get(p.token_id, 0) + (k not in self.suppressed)
        return out


def _ids(n: int, start: int = TOKEN_ID_BASE) -> range:
    r = range(start, start + n)
    assert r[-1] <= 31, "Controller_ID is a 5-bit field"
    return r


def _probe(kind: str, col: int, row: int, direction: str, index: int, token_id: int) -> Probe:
    return Probe(kind, col, row, direction, index, token_id)


def designs() -> dict[str, Design]:
    """Attribution is by token id; each design also varies something else (the tile for compute, the
    order elsewhere) so a word can be cross-checked."""
    def mem(directions, reverse):
        chans = [(d, i) for d in directions for i in range(6)]
        if reverse:
            chans.reverse()
        return tuple(_probe("mem", 0, 1, d, i, TOKEN_ID_BASE + (0 if d == "s2mm" else 6) + i) for d, i in chans)

    cmp_probes = tuple(
        _probe("core", 0, row, d, i, tid)
        for (row, d, i), tid in zip(((2, "s2mm", 0), (3, "s2mm", 1), (4, "mm2s", 0), (5, "mm2s", 1)), _ids(4)))
    shm_probes = tuple(
        _probe("shim", 1, 0, d, i, tid)
        for (d, i), tid in zip((("s2mm", 0), ("s2mm", 1), ("mm2s", 0), ("mm2s", 1)), _ids(4)))
    out = [Design("cmp", "npu1_1col", cmp_probes), Design("shm", "npu1_2col", shm_probes),
           Design("mems", "npu1_1col", mem(("s2mm",), False)), Design("memm", "npu1_1col", mem(("mm2s",), False)),
           Design("memsr", "npu1_1col", mem(("s2mm",), True)), Design("memmr", "npu1_1col", mem(("mm2s",), True))]
    out += [Design(f"{d.name}-half", d.device, d.probes, frozenset(range(1, len(d.probes), 2))) for d in list(out)]
    out += [Design(f"{d.name}-dbl", d.device, tuple(q for p in d.probes for q in (p, p))) for d in out[:6]]
    return {d.name: d for d in out}


BASE_DESIGNS = ("cmp", "shm", "mems", "memm", "memsr", "memmr")


# -- registers -------------------------------------------------------------------------------

_REGS: dict[str, dict[str, int]] = {}


def _regs(module: str) -> dict[str, int]:
    if module not in _REGS:
        _REGS[module] = cp.module_registers(module)
    return _REGS[module]


def channel_registers(probe: Probe) -> tuple[int, int]:
    """(Ctrl, Start_Queue) addresses of the probe's channel, row bits included as in the
    `aiex.npu.write32` operands of upstream's writebd_tokens tests."""
    module = KINDS[probe.kind][0]
    row = probe.row
    regs = _regs(module)
    d = probe.direction.upper()
    ctrl = regs[f"DMA_{d}_{probe.index}_Ctrl"]
    queue = regs.get(f"DMA_{d}_{probe.index}_Start_Queue", regs.get(f"DMA_{d}_{probe.index}_Task_Queue"))
    return (row << 20) | ctrl, (row << 20) | queue



# -- MLIR ------------------------------------------------------------------------------------

def _const(name: str, value: int) -> str:
    return f"      %{name} = arith.constant {value} : i32"


def _flow(pkt: int, src: str, src_port: str, dst: str, dst_port: str, keep: bool = False) -> list[str]:
    out = [f"    aie.packet_flow({pkt}) {{",
           f"      aie.packet_source<{src}, {src_port}>",
           f"      aie.packet_dest<{dst}, {dst_port}>",
           "    }" + (" {keep_pkt_header = true}" if keep else "")]
    return out


def _task(var: str, tile: str, direction: str, ch: int, bd: str, bd_id: int, token: bool = False) -> list[str]:
    return [f"      %{var} = aiex.dma_configure_task({tile}, {direction}, {ch}) {{",
            f"        {bd} {{bd_id = {bd_id} : i32}}",
            "        aie.end",
            "      }" + (" {issue_token = true}" if token else "")]


def probe_bd_ids(design: Design, k: int) -> tuple[int, int]:
    """(receiver, sender) BD ids of probe k. Ids are never reused on a tile (a raw-pushed task is
    not tracked by the compiler, so a reused id would be flagged in use); the memtile's odd
    channels reach only the upper half of its ids."""
    p = design.probes[k]
    same = [q for q in design.probes[:k] if q.tile == p.tile and (p.kind != "mem" or q.index % 2 == p.index % 2)]
    base = MEM_ODD_CHANNEL_BD_BASE * (p.index % 2) if p.kind == "mem" else 0
    return base + 2 * len(same), base + 2 * len(same) + 1


def pair_lines(design: Design, k: int) -> list[str]:
    """The k-th probe: a loop MM2S i -> S2MM i inside its tile, the measured end pushed by hand
    with Enable_Token_Issue and the probe's Controller_ID, the other end started plainly."""
    p = design.probes[k]
    i = p.index
    shim = p.kind == "shim"
    if shim:
        off = MARKER_WORDS + PAIR_WORDS * sum(q.kind == "shim" for q in design.probes[:k])
        src = f"aie.dma_bd(%din : memref<{design.host_in_words}xi32> offset = {off} len = {PAIR_WORDS})"
        dst = f"aie.dma_bd(%dout : memref<{design.host_out_words}xi32> offset = {off} len = {PAIR_WORDS})"
    else:
        src = f"aie.dma_bd(%src_{p.col}{p.row} : memref<{PAIR_WORDS}xi32> offset = 0 len = {PAIR_WORDS})"
        dst = f"aie.dma_bd(%dst_{p.col}{p.row} : memref<{PAIR_WORDS}xi32> offset = 0 len = {PAIR_WORDS})"
    ctrl, queue = channel_registers(p)
    row = p.row
    n = f"p{k}"
    bd_dst, bd_src = probe_bd_ids(design, k)
    out = [f"      // probe {k}: {p.name}, token id {p.token_id}"]
    out += _task(f"{n}_rx", p.tile, "S2MM", i, dst, bd_dst)
    out += _task(f"{n}_tx", p.tile, "MM2S", i, src, bd_src)
    plain = {"s2mm": f"{n}_tx", "mm2s": f"{n}_rx"}[p.direction]   # the end that is not measured
    if k in design.suppressed:     # the (b) variants: the same loop, no token issued
        measured = {"s2mm": f"{n}_rx", "mm2s": f"{n}_tx"}[p.direction]
        first, second = (measured, plain) if p.direction == "s2mm" else (plain, measured)
        return out + [f"      aiex.dma_start_task(%{first})", f"      aiex.dma_free_task(%{first})",
                      f"      aiex.dma_start_task(%{second})", f"      aiex.dma_free_task(%{second})"]
    out += [_const(f"{n}_a", ctrl), _const(f"{n}_v", p.token_id << CONTROLLER_ID_SHIFT),
            _const(f"{n}_m", CONTROLLER_ID_MASK), _const(f"{n}_q", queue)]
    out.append(f"      aiex.npu.maskwrite32(%{n}_a, %{n}_v, %{n}_m) "
               f"{{column = {p.col} : i32, row = {row} : i32}} : i32, i32, i32")
    bd_of = {"s2mm": bd_dst, "mm2s": bd_src}[p.direction]
    out.append(_const(f"{n}_w", ENABLE_TOKEN_ISSUE | bd_of))
    push = (f"      aiex.npu.write32(%{n}_q, %{n}_w) {{column = {p.col} : i32, row = {row} : i32}} : i32, i32")
    if p.direction == "s2mm":      # receiver first: it waits for the sender's words
        out += [push, f"      aiex.dma_start_task(%{plain})", f"      aiex.dma_free_task(%{plain})"]
    else:
        out += [f"      aiex.dma_start_task(%{plain})", f"      aiex.dma_free_task(%{plain})", push]
    return out


def marker_lines(design: Design, k: int) -> list[str]:
    """The shim loop that follows probe k; its S2MM token goes to the controller and is awaited."""
    n = f"mk{k}"
    src = f"aie.dma_bd(%din : memref<{design.host_in_words}xi32> offset = 0 len = {MARKER_WORDS})"
    dst = f"aie.dma_bd(%dout : memref<{design.host_out_words}xi32> offset = 0 len = {MARKER_WORDS})"
    return [f"      // marker {k}: shim MM2S 1 -> S2MM 1, token to the controller",
            *_task(f"{n}_rx", "%t00", "S2MM", 1, dst, MARKER_BD_OUT, token=True),
            *_task(f"{n}_tx", "%t00", "MM2S", 1, src, MARKER_BD_IN, token=True),
            f"      aiex.dma_start_task(%{n}_rx)",
            f"      aiex.dma_start_task(%{n}_tx)",
            f"      aiex.dma_await_task(%{n}_rx)",
            f"      aiex.dma_await_task(%{n}_tx)"]


def filler_lines(design: Design) -> list[str]:
    n = design.host_in_words
    return ["      // filler: shim MM2S 0 into the capture S2MM, behind the last token; awaited the normal way",
            f"      aiex.npu.dma_memcpy_nd(%din[0, 0, 0, {design.filler_offset}][1, 1, 1, {FILLER_WORDS}][0, 0, 0, 1], "
            f"packet = <pkt_id = {FILLER_ID}, pkt_type = 0>) "
            f"{{id = {FILLER_BD} : i64, issue_token = true, metadata = @filler0}} : memref<{n}xi32>",
            "      aiex.npu.dma_wait {symbol = @filler0}"]


def design_mlir(design: Design) -> str:
    L = design.capture_words
    tiles = sorted({(p.col, p.row) for p in design.probes})
    lines = ["module {", f"  aie.device({design.device}) {{",
             f"    %t00 = aie.tile(0, 0) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {SHIM_CONTROLLER_ID}>}}"]
    for c, r in tiles:
        if (c, r) == (0, 0):
            continue
        extra = (f" {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {SHIM1_CONTROLLER_ID}>}}"
                 if (c, r) == (1, 0) else "")
        lines.append(f"    %t{c}{r} = aie.tile({c}, {r}){extra}")
    for c, r in tiles:
        if r in (1, 2, 3, 4, 5):
            for sym, addr in (("src", TILE_SRC_ADDR), ("dst", TILE_DST_ADDR)):
                lines.append(f'    %{sym}_{c}{r} = aie.buffer(%t{c}{r}) {{address = {addr} : i32, '
                             f'sym_name = "{sym}_{c}{r}"}} : memref<{PAIR_WORDS}xi32>')
    loops = sorted({(p.tile, p.index) for p in design.probes})
    for tile, i in loops:
        lines.append(f"    aie.flow({tile}, DMA : {i}, {tile}, DMA : {i})")
    lines.append("    aie.flow(%t00, DMA : 1, %t00, DMA : 1)")
    lines += _flow(FILLER_ID, "%t00", "DMA : 0", "%t00", "DMA : 0", keep=True)
    for p in _unique(design):
        lines += _flow(p.token_id, p.tile, "TileControl : 0", "%t00", "DMA : 0", keep=True)
    lines += ["    aie.shim_dma_allocation @filler0 (%t00, MM2S, 0)",
              "    aie.shim_dma_allocation @ctrl0 (%t00, S2MM, 0)",
              f"    aie.runtime_sequence @seq(%din: memref<{design.host_in_words}xi32>, "
              f"%dout: memref<{design.host_out_words}xi32>, "
              f"%cap: memref<{L}xi32>) {{",
              f"      aiex.npu.dma_memcpy_nd(%cap[0, 0, 0, 0][1, 1, 1, {L}][0, 0, 0, 1]) "
              f"{{id = {CAPTURE_BD} : i64, issue_token = false, metadata = @ctrl0}} : memref<{L}xi32>"]
    for k in range(len(design.probes)):
        lines += pair_lines(design, k)
        lines += marker_lines(design, k)
    lines += filler_lines(design)
    lines += marker_lines(design, len(design.probes))     # settle: the filler's last bursts reach host memory
    lines += ["    }", "  }", "}", ""]
    return "\n".join(lines)


def _unique(design: Design) -> list[Probe]:
    """One flow per (tile, token id): the memtile design measures each channel twice."""
    seen: dict[tuple[str, int], Probe] = {}
    for p in design.probes:
        seen.setdefault((p.tile, p.token_id), p)
    return list(seen.values())


# -- decode -----------------------------------------------------------------------------------

# The emulator's token word, for comparison only: column << 21 | row << 16 | 6 << 12 | actor << 8 | id (actor bit 4
# lands on bit 12, inside the type field, so the decode reads the type's upper bits [14:13] and a 5-bit actor),
# bit 31 making the word's parity odd. Physical column = mlir column + the partition's start column.
EMU_SRC_COL_SHIFT, EMU_SRC_ROW_SHIFT, EMU_TYPE_SHIFT, EMU_ACTOR_SHIFT = 21, 16, 12, 8


def emu_fields(word: int) -> dict[str, int]:
    return {"parity": word >> 31, "col": (word >> EMU_SRC_COL_SHIFT) & 0x7F,
            "row": (word >> EMU_SRC_ROW_SHIFT) & 0x1F, "type_hi": (word >> 13) & 0x3,
            "actor": (word >> EMU_ACTOR_SHIFT) & 0x1F, "id": word & 0x1F}


def parity_odd(word: int) -> bool:
    return bin(word).count("1") % 2 == 1


@dataclass(frozen=True)
class Record:
    index: int
    word: int
    kind: str                 # "token" | "extra" | "filler" | "unattributed"
    probe: Probe | None = None
    seq: int | None = None    # which of the design's probes this word belongs to (issue order)


@dataclass(frozen=True)
class Decoded:
    records: tuple[Record, ...]        # the words before the filler (and the filler's first word)
    filler_at: int | None              # index of the first filler word
    found: dict                        # token id -> words found
    missing: tuple[int, ...]           # probe indexes expected to issue a token that have no word
    problems: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def token_words(self) -> int:
        return sum(self.found.values())


def find_filler(words: list[int]) -> int | None:
    """Index of the first filler word: a data word FILLER_MARK + j, or the filler's packet header
    (its id in the low bits) right before one."""
    for i, w in enumerate(words):
        if (w >> 16) == FILLER_MARK >> 16:
            return i - 1 if i and (words[i - 1] & 0x1F) == FILLER_ID else i
    return None


def decode_capture(words: list[int], design: Design) -> Decoded:
    """Attribute every word before the filler by its token id; do not assume a word count. A token id
    seen more often than the design's tasks issue it is listed as an extra word (a note, not an
    error); a word whose id is no probe's is unattributed. The run is invalid when the filler is not
    in the capture (words may be unflushed), a probe expected to issue a token has no word, or a
    word cannot be attributed."""
    filler = find_filler(words)
    body = words if filler is None else words[:filler]
    while body and body[-1] == 0:      # unwritten tail of the buffer (the buffer starts zeroed)
        body = body[:-1]
    by_id: dict[int, list[int]] = {}
    for k, p in enumerate(design.probes):
        if k not in design.suppressed:
            by_id.setdefault(p.token_id, []).append(k)
    all_ids = {p.token_id for p in design.probes}
    used: dict[int, int] = {}
    records: list[Record] = []
    problems: list[str] = []
    notes: list[str] = []
    for i, w in enumerate(body):
        ident = w & 0x1F
        if ident not in all_ids:
            records.append(Record(i, w, "unattributed"))
            problems.append(f"word {i} {w:#010x}: id {ident} is not a probe's token id")
            continue
        n = used.get(ident, 0)
        used[ident] = n + 1
        slots = by_id.get(ident, [])
        if n < len(slots):
            records.append(Record(i, w, "token", design.probes[slots[n]], slots[n]))
        else:
            first = next(p for p in design.probes if p.token_id == ident)
            records.append(Record(i, w, "extra", first))
    if filler is None:
        problems.append("the filler is not in the capture: token words may be unflushed")
    else:
        records.append(Record(filler, words[filler], "filler"))
    missing = tuple(k for k, p in enumerate(design.probes)
                    if k not in design.suppressed and used.get(p.token_id, 0) <
                    sum(1 for j in range(k + 1) if j not in design.suppressed and design.probes[j].token_id == p.token_id))
    for k in missing:
        problems.append(f"probe {k} {design.probes[k].name} (id {design.probes[k].token_id}): no word")
    for ident, n in sorted(used.items()):
        want = design.expected().get(ident, 0)
        if n != want:
            notes.append(f"id {ident}: {n} word(s), expected {want}")
    return Decoded(tuple(records), filler, dict(sorted(used.items())), missing, tuple(problems), tuple(notes))


def word_line(r: Record, partition_start_col: int | None = None) -> str:
    """One line for one captured word: raw, then the fields under the emulator's layout."""
    f = emu_fields(r.word)
    odd = "odd" if parity_odd(r.word) else "even"
    who = r.probe.name if r.probe else ""
    col = f"col {f['col']}" + (f" (mlir {f['col'] - partition_start_col})" if partition_start_col is not None else "")
    return (f"{r.index:3d} {r.word:#010x} {r.word:032b}  {r.kind:12s} {who:18s} "
            f"emu-layout: bit31 {f['parity']} {col} row {f['row']} type[14:13] {f['type_hi']} actor {f['actor']} "
            f"id {f['id']}  parity {odd}")


def report(decoded: Decoded, design: Design, partition_start_col: int | None = 1) -> str:
    lines = [word_line(r, partition_start_col) for r in decoded.records]
    tag = "filler begins at word " + str(decoded.filler_at) if decoded.filler_at is not None else "no filler found"
    lines.append(f"{tag}; token words found {decoded.token_words}; probes {len(design.probes)}; "
                 f"expected {sum(design.expected().values())}")
    lines += [f"  note: {n}" for n in decoded.notes]
    lines.append("VALID" if decoded.valid else "INVALID")
    lines += [f"  {p}" for p in decoded.problems]
    return "\n".join(lines)


def read_capture(path: Path, design: Design) -> list[int]:
    raw = path.read_bytes()
    n = min(design.capture_words, len(raw) // 4)
    return list(struct.unpack(f"<{n}I", raw[:4 * n]))


# -- CLI -------------------------------------------------------------------------------------

def host_in(design: Design) -> bytes:
    words = [HOST_MARK + i for i in range(design.host_in_words)]
    words[design.filler_offset:] = [FILLER_MARK + j for j in range(FILLER_WORDS)]
    return cp.pack_words(words)


def cmd_list(args) -> int:
    for d in designs().values():
        print(f"{d.name}: {d.device}, {len(d.probes)} tasks, {len(d.suppressed)} suppressed, "
              f"capture {d.capture_words} words")
        for k, p in enumerate(d.probes):
            print(f"  {k:2d} {p.name} id {p.token_id}")
    return 0


def cmd_generate(args) -> int:
    table = designs()
    for name in args.designs or table:
        d = table[name]
        out = Path(args.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "aie.mlir").write_text(design_mlir(d))
        (out / "din.bin").write_bytes(host_in(d))
        (out / "case.json").write_text(json.dumps(
            {"design": name, "device": d.device, "capture_words": d.capture_words,
             "host_words": d.host_in_words,
             "probes": [{"name": p.name, "token_id": p.token_id} for p in d.probes]}, indent=1) + "\n")
    return 0


def cmd_build(args) -> int:
    from harness.build import aiecc_command, aiecc_path
    for d in sorted(p for p in Path(args.out_dir).iterdir() if (p / "aie.mlir").exists()):
        r = subprocess.run(aiecc_command("peano", aiecc_path()), cwd=d, capture_output=True, text=True)
        (d / "aiecc.log").write_text(r.stdout + r.stderr)
        warnings = [ln for ln in (r.stdout + r.stderr).splitlines() if "warning" in ln]
        print(f"{d.name}: aiecc exit {r.returncode}, {len(warnings)} warnings")
        if r.returncode:
            return 1
    return 0


RUNNER = REPO / "bridge-runner" / "build" / "bridge-trace-runner"
HW_LOCK = REPO / "build" / "experiments" / ".npu-hw.lock"


def run_env(backend: str) -> dict[str, str]:
    env = dict(os.environ)
    if backend == "hw":
        for k in ("XDNA_EMU", "XDNA_EMU_LIB", "XDNA_EMU_DIR", "XDNA_BACKEND", "XDNA_EMU_RUNTIME"):
            env.pop(k, None)
    else:
        # The emulator backend and library are the caller's choice (XDNA_BACKEND, XDNA_EMU_LIB).
        env["XDNA_EMU"] = "1"
        env.setdefault("XDNA_EMU_RUNTIME", "release")
    return env


def runner_command(case: Path, rundir: Path, design: Design) -> list[str]:
    return [str(RUNNER), "--xclbin", str(case / "aie.xclbin"), "--instr", str(case / "insts.bin"),
            "--input", str(case / "din.bin"),
            "--output", str(rundir / "dout.bin"),
            "--trace-out", str(rundir / "cap.bin"), "--trace-size", str(4 * design.capture_words),
            "--trace-buf-idx", "2", *os.environ.get("BRIDGE_RUNNER_EXTRA_ARGS", "").split()]


def cmd_run(args) -> int:
    import fcntl
    table = designs()
    root = Path(args.out_dir)
    for name in args.only or [n for n in table if n in BASE_DESIGNS]:
        case, d = root / name, table[name]
        rd = case / f"{args.tag}.{args.backend}"
        if rd.exists():
            raise SystemExit(f"{rd} exists; pick another tag")
        rd.mkdir(parents=True)
        env = run_env(args.backend)
        if args.backend == "emu":
            env["RUST_LOG"] = "info"
        lock = None
        if args.backend == "hw":
            lock = open(HW_LOCK, "w")
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            p = subprocess.run(runner_command(case, rd, d), env=env, capture_output=True, text=True,
                               timeout=60 if args.backend == "hw" else 1800)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = -1, "", "TIMEOUT"
        finally:
            if lock:
                lock.close()
        keep = [ln for ln in err.splitlines()
                if not ("routing]" in ln or "stream switch" in ln or "xdna_emu::config" in ln)] if args.backend == "emu" else err.splitlines()
        (rd / "runner.log").write_text(out + "\n" + "\n".join(keep) + "\n")
        print(f"{name}: rc {rc}", flush=True)
        if rc:
            return 1
    return 0


def finish_check(log: str) -> tuple[list[str], bool]:
    """From an emulator run's log: ("tile ch BD" entries started more often than they finished,
    whether the run ended `completed`). Every DMA task must finish before the sequence ends,
    except the capture carrier (shim S2MM 0, BD 15), which by design never does."""
    import re
    from collections import Counter
    starts, ends = Counter(), Counter()
    for m in re.finditer(r"DMA tile\((\d+),(\d+)\) ch(\d+) BD(\d+) (start|FINISH)", log):
        key = f"tile({m.group(1)},{m.group(2)}) ch{m.group(3)} BD{m.group(4)}"
        (starts if m.group(5) == "start" else ends)[key] += 1
    carrier = f"BD{CAPTURE_BD}"          # the capture BD is the trace-carrier pattern: it never finishes
    unfinished = sorted(f"{k}: {starts[k]} started, {ends[k]} finished" for k in starts
                        if ends[k] < starts[k] and not (k.endswith(f",0) ch0 {carrier}")))
    return unfinished, "halt_reason=completed" in log


def partition_start_col(runner_log: Path) -> int | None:
    if runner_log.exists():
        for ln in runner_log.read_text().splitlines():
            if "start_col = " in ln:
                return int(ln.rsplit("start_col = ", 1)[1].split()[0])
    return None


def cmd_decode(args) -> int:
    table = designs()
    root = Path(args.out_dir)
    status = 0
    for name in args.only or [n for n in table if n in BASE_DESIGNS]:
        d = table[name]
        rd = root / name / f"{args.tag}.{args.backend}"
        cap = rd / "cap.bin"
        print(f"== {name} {args.tag}.{args.backend}")
        if not cap.exists():
            print("INVALID: no capture file")
            status = 2
            continue
        decoded = decode_capture(read_capture(cap, d), d)
        print(report(decoded, d, partition_start_col(rd / "runner.log")))
        if not decoded.valid:
            status = 2
        if args.backend == "emu":
            unfinished, completed = finish_check((rd / "runner.log").read_text())
            print(f"emulator run completed: {completed}; unfinished DMA tasks: {unfinished or 'none'}")
            if unfinished or not completed:
                status = 2
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("designs", nargs="*")
    sub.add_parser("build").add_argument("out_dir")
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--only", nargs="*", default=[])
    de = sub.add_parser("decode")
    de.add_argument("out_dir")
    de.add_argument("tag")
    de.add_argument("--backend", choices=("hw", "emu"), default="emu")
    de.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args(argv)
    return {"list": cmd_list, "generate": cmd_generate, "build": cmd_build, "run": cmd_run,
            "decode": cmd_decode}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
