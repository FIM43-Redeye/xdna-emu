#!/usr/bin/env python3
"""Gather-address probe: which bits of a `vldb.4x*` lane value pick the memory it reads.

`vldb.4x16/32/64.lo/hi wD, wS` reads four 8-byte elements, one per lane of `wS` (lanes 0..3
for lo, 4..7 for hi), each lane a full 32-bit value used as an address. The low five bits
are known (`lut_lane_address`, `docs/fidelity-gaps/addressing.md`). This probe sets the
other bits on purpose: bits 5..15 across the whole 64 KiB tile memory, bits 16..19 (the
cardinal direction of the 20-bit core address space: 4 south, 5 west, 6 north, 7 own
memory) to every value, and bits 20..31.

Buffers and regions
-------------------
  in:  [0,32)        marker 0xEE (pre-loaded into the destination: a gather that does not
                     write it shows the marker)
       [64,576)      window 1, halfwords (1 << 12) | j
       [576,1088)    window 2, halfwords (2 << 12) | j
       [1088,1600)   window 3, halfwords (3 << 12) | j
  The kernel copies window 2 to the top 512 bytes of its own data memory ("far",
  (in & ~0xFFFF) | 0xFE00) and window 3 into the north neighbour's memory at the in-buffer
  offset ("north", in ^ 0x10000). A lane that reads one of the three regions returns a
  halfword naming the window and the position, so the analysis decodes where each lane read
  from without assuming any rule.

  out: [0,64)         header, u32: in, out, far, north
       [64,..)        one 64-byte record per gather: the eight lane values, then the result

Cases
-----
  lut_alias       every target the kernel itself wrote: window 1 in place, far, north, each
                  with the lane value XORed by masks that flip bits 16..19 and 20..31
  ptr_alias       the same aliases through a pointer register (20 bits): `vlda`, `vldb`,
                  `lda` from window 1 and north, and `vst` into the record itself
  lut_unwritten   memory nobody in the kernel wrote: the west and south quadrants (no core
                  memory there on NPU1 row 2) and their aliases, each gather preceded by a
                  known `vldb` so a stale load-port value is recognisable
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from harness import suite                                                  # noqa: E402
from harness.topology import Buffer, Core, TileRef, Topology              # noqa: E402

CORE = "k"
MARKER = 0xEE
MARK_OFF = 0
WIN_BYTES = 512
WIN_OFF = {1: 64, 2: 64 + WIN_BYTES, 3: 64 + 2 * WIN_BYTES}
IN_BYTES = 64 + 3 * WIN_BYTES
HDR_BYTES = 64
REC_BYTES = 64
FAR_OFF = 0x10000 - WIN_BYTES     # far region: the top 512 bytes of the tile's data memory
NORTH_XOR = 0x10000               # in ^ this = the same offset in the north neighbour
NOPS = 8

# Per-lane offsets inside a 512-byte window: row starts and ends, both banks, odd granules,
# the first and the last 32-byte block.
LANE_OFFS = (0, 11, 78, 103, 201, 286, 489, 511)

GATHERS = tuple((w, h) for w in (16, 32, 64) for h in ("lo", "hi"))


def in_bytes() -> bytes:
    out = bytearray([MARKER] * 32 + [0] * 32)
    for w in (1, 2, 3):
        out += b"".join(struct.pack("<H", (w << 12) | j) for j in range(WIN_BYTES // 2))
    assert len(out) == IN_BYTES
    return bytes(out)


@dataclass(frozen=True)
class Record:
    """One gather: lane k = (target base + LANE_OFFS[k] + shift) ^ xor."""
    target: str                 # "w1" (in buffer), "far", "north", "west", "south"
    xor: int
    width: int                  # 16 / 32 / 64
    half: str                   # lo / hi
    shift: int = 0              # added to every lane offset (moves the lanes inside the window)
    pre_off: int = -1           # >= 0: a `vldb` from window 1 at this offset right before

    @property
    def name(self) -> str:
        s = f"{self.target}^{self.xor:08x}.4x{self.width}.{self.half}"
        return s + (f"+{self.shift}" if self.shift else "") + (
            f".pre{self.pre_off}" if self.pre_off >= 0 else "")


ALIAS_XORS_LOCAL = (0x00000000, 0x00040000, 0x00080000, 0x000C0000,
                    0x00100000, 0x00F00000, 0x80000000, 0xFFF00000)
ALIAS_XORS_FAR = (0x00000000, 0x00080000, 0x00040000, 0xFFF00000)
ALIAS_XORS_NORTH = (0x00000000, 0x00080000, 0x00040000, 0x000C0000, 0xFFF00000)
# from the in buffer (direction 7): 0x20000 -> 5 west, 0x30000 -> 4 south, 0x70000 -> 0,
# 0x60000 -> 1, 0xF0000 -> 8, 0xA0000 -> 0xD, plus the same with bits above 19
UNWRITTEN_XORS = (0x00020000, 0x00030000, 0x00070000, 0x00060000, 0x000F0000, 0x000A0000,
                  0x80020000, 0x80030000)


@dataclass(frozen=True)
class PtrRecord:
    """One pointer access at (target base + offset) ^ xor (the pointer register keeps 20 bits).
    Loads land in the record's result; a store writes window-1 bytes at the aliased result
    address."""
    kind: str                   # vlda / vldb / lda / vst
    target: str                 # w1 / north (loads); the record itself (stores)
    xor: int
    off: int = 0x48

    @property
    def name(self) -> str:
        return f"ptr.{self.kind}.{self.target}^{self.xor:05x}+{self.off}"


PTR_XORS = (0x00000, 0x40000, 0x80000, 0xC0000)


def build_ptr_records() -> list[PtrRecord]:
    recs = [PtrRecord(k, t, x) for k in ("vlda", "vldb", "lda") for t in ("w1", "north")
            for x in PTR_XORS]
    return recs + [PtrRecord("vst", "rec", x, off=0) for x in PTR_XORS]


def build_records() -> dict[str, list]:
    alias: list[Record] = []
    for x in ALIAS_XORS_LOCAL:
        gathers = GATHERS if x == 0 else ((64, "lo"), (16, "hi"))
        alias += [Record("w1", x, w, h) for w, h in gathers]
    alias += [Record("w1", 0, 32, "lo", shift=s) for s in (1, 2, 5, 17)]
    for x in ALIAS_XORS_FAR:
        alias += [Record("far", x, w, h) for w, h in ((64, "lo"), (16, "hi"))]
    for x in ALIAS_XORS_NORTH:
        alias += [Record("north", x, w, h) for w, h in ((64, "lo"), (16, "hi"))]
    unwritten: list[Record] = []
    for i, x in enumerate(UNWRITTEN_XORS):
        for j, (w, h) in enumerate(((64, "lo"), (16, "hi"))):
            unwritten.append(Record("w1", x, w, h, pre_off=64 * (2 * i + j) % WIN_BYTES))
    return {"lut_alias": alias, "ptr_alias": build_ptr_records(), "lut_unwritten": unwritten}


# -- assembly ---------------------------------------------------------------

def _nops(n: int) -> list[str]:
    return ["  nop"] * n


def _frame(body: list[str]) -> str:
    pro = [".text", ".globl test_kernel", "test_kernel:"]
    epi = ["  nop"] * 7 + ["  ret lr"] + ["  nop"] * 5
    return "\n".join(pro + body + epi) + "\n"


def _copy(src_reg_expr: list[str], dst_reg_expr: list[str], nbytes: int) -> list[str]:
    """p2 <- source, p3 <- destination (each built by the given lines into r1), then 32-byte
    load/store pairs with post-increment."""
    body = [*src_reg_expr, "  mov p2, r1", *dst_reg_expr, "  mov p3, r1", *_nops(2)]
    for _ in range(nbytes // 32):
        body += ["  vlda wl0, [p2], #32", *_nops(NOPS), "  vst wl0, [p3], #32"]
    return body + _nops(2)


def _base(target: str) -> str:
    """Register holding the target region's base address."""
    return {"w1": "r20", "far": "r21", "north": "r22"}[target]


def _ptr_record(rec: PtrRecord) -> list[str]:
    body = [f"  // {rec.name}", f"  movxm r7, #{rec.xor}", f"  movxm r6, #{rec.off}", *_nops(2)]
    if rec.kind == "vst":
        body += ["  mov r1, p4", *_nops(2), "  add r6, r1, r6", "  movxm r2, #32", *_nops(2),
                 "  add r6, r6, r2"]
    else:
        body += [f"  add r6, {_base(rec.target)}, r6"]
    body += [*_nops(1), "  xor r6, r6, r7", *_nops(1), "  st r6, [p4, #0]",
             f"  vlda wl2, [p0, #{MARK_OFF}]", *_nops(NOPS), "  vst wl2, [p4, #32]",
             "  mov p6, r6", *_nops(NOPS)]
    if rec.kind == "vst":
        body += ["  mov r1, r20", *_nops(2), "  mov p5, r1", *_nops(NOPS), "  vlda wl0, [p5, #0]",
                 *_nops(NOPS), "  vst wl0, [p6, #0]", *_nops(NOPS)]
    elif rec.kind == "lda":
        body += ["  lda r9, [p6, #0]", *_nops(NOPS), "  st r9, [p4, #32]"]
    else:
        body += [f"  {rec.kind} wl2, [p6, #0]", *_nops(NOPS), "  vst wl2, [p4, #32]"]
    return body + ["  padda [p4], #64", *_nops(2)]


def kernel_asm(records: list) -> str:
    in_plus = lambda off: ["  mov r1, p0", f"  movxm r2, #{off}", *_nops(2), "  add r1, r1, r2"]
    body: list[str] = []
    # r20 = window 1, r21 = far, r22 = north (window 3's offset in the neighbour = in ^ 0x10000)
    body += ["  mov r1, p0", f"  movxm r2, #{WIN_OFF[1]}", *_nops(2), "  add r20, r1, r2",
             "  movxm r3, #-65536", *_nops(2), "  and r4, r1, r3", f"  movxm r2, #{FAR_OFF}",
             *_nops(2), "  add r21, r4, r2",
             f"  movxm r3, #{NORTH_XOR}", *_nops(2), "  xor r22, r1, r3", *_nops(2)]
    body += ["  mov r2, p0", "  mov r3, p1", *_nops(3), "  st r2, [p1, #0]", "  st r3, [p1, #4]",
             "  st r21, [p1, #8]", "  st r22, [p1, #12]"]
    body += _copy(in_plus(WIN_OFF[2]), ["  mov r1, r21"], WIN_BYTES)
    body += _copy(in_plus(WIN_OFF[3]), ["  mov r1, r22"], WIN_BYTES)
    # p4 = record cursor
    body += ["  mov r1, p1", f"  movxm r2, #{HDR_BYTES}", *_nops(2), "  add r1, r1, r2",
             "  mov p4, r1", *_nops(4)]
    for rec in records:
        if isinstance(rec, PtrRecord):
            body += _ptr_record(rec)
            continue
        base = _base(rec.target)
        body += [f"  // {rec.name}", f"  movxm r7, #{rec.xor - (1 << 32) if rec.xor >= 1 << 31 else rec.xor}"]
        for k, off in enumerate(LANE_OFFS):
            body += [f"  movxm r6, #{off + rec.shift}", *_nops(2), f"  add r6, {base}, r6",
                     *_nops(1), "  xor r6, r6, r7", *_nops(1), f"  st r6, [p4, #{4 * k}]"]
        body += [*_nops(2), "  vlda wl4, [p4, #0]", f"  vlda wl2, [p0, #{MARK_OFF}]"]
        if rec.pre_off >= 0:
            body += ["  mov r1, p0", f"  movxm r2, #{WIN_OFF[1] + rec.pre_off}", *_nops(2),
                     "  add r1, r1, r2", "  mov p5, r1", *_nops(NOPS), "  vldb wl6, [p5, #0]",
                     *_nops(NOPS)]
        else:
            body += _nops(NOPS)
        body += [f"  vldb.4x{rec.width}.{rec.half} wl2, wl4", *_nops(NOPS),
                 "  vst wl2, [p4, #32]", "  padda [p4], #64", *_nops(2)]
    return _frame(body)


def out_bytes(records: list[Record]) -> int:
    return HDR_BYTES + REC_BYTES * len(records)


def case_topology(name: str, records: list[Record]) -> tuple[Topology, str]:
    return Topology(name, (Core(CORE, TileRef(0, 2), Path("kernel.s"), args=(
        Buffer("in", "in", IN_BYTES, data=in_bytes()),
        Buffer("out", "out", out_bytes(records)))),), device="auto"), kernel_asm(records)


def generate(out_dir: Path) -> list[str]:
    names = []
    for name, recs in build_records().items():
        topo, asm = case_topology(name, recs)
        suite.write_case(out_dir, name, topo, files={"kernel.s": asm},
                         meta=dict(name=name, records=[r.name for r in recs],
                                   result=dict(core=CORE, buffer="out")))
        names.append(name)
    suite.write_suite_json(out_dir, "lut-addr", "gen_lut_addr_probe.py", names)
    return names


# -- analysis ------------------------------------------------------------------

def lut_lane_address(v: int, lane: int, width: int) -> int:
    """The low-bit rule (silicon-fitted, `lut_lane_address` in memory/mod.rs)."""
    mask = {16: ~3, 32: ~7, 64: ~15}[width] & 0xFFFFFFFF
    return (v & ~31 & 0xFFFFFFFF) | (((v & 31) & mask) >> 1) | (0x10 if lane % 2 else 0)


@dataclass(frozen=True)
class Header:
    inp: int
    out: int
    far: int
    north: int


def parse_header(blob: bytes) -> Header:
    return Header(*struct.unpack_from("<4I", blob, 0))


def where(h: Header, half_word: int, lane_bytes: bytes) -> str | int:
    """Core address a lane read from, decoded from the window pattern; 'marker', 'zero' or
    'unknown:<hex>' otherwise."""
    if lane_bytes == bytes([MARKER]) * 8:
        return "marker"
    if lane_bytes == bytes(8):
        return "zero"
    w, j = half_word >> 12, half_word & 0xFFF
    if w in (1, 2, 3) and j < WIN_BYTES // 2:
        base = {1: h.inp + WIN_OFF[1], 2: h.far, 3: h.north}[w]
        return base + 2 * j
    return "unknown:" + lane_bytes.hex()


def expected_bytes(h: Header, addr: int) -> bytes:
    """The 8 bytes a gather lane at `addr` returns if it reads the region the kernel wrote
    there: bytes past the 16-byte row read zero."""
    regions = ((h.inp + WIN_OFF[1], 1), (h.far, 2), (h.north, 3))
    for base, w in regions:
        if base <= addr < base + WIN_BYTES:
            win = in_bytes()[WIN_OFF[w]:WIN_OFF[w] + WIN_BYTES]
            o = addr - base
            n = min(8, 16 - (addr & 15))
            return win[o:o + n] + bytes(8 - n)
    raise ValueError(f"{addr:#x} is in no written region")


def decode_records(blob: bytes, names: list[str]) -> tuple[Header, list[dict]]:
    h = parse_header(blob)
    rows = []
    for i, name in enumerate(names):
        rec = blob[HDR_BYTES + REC_BYTES * i:HDR_BYTES + REC_BYTES * (i + 1)]
        if name.startswith("ptr."):
            ptr = struct.unpack_from("<I", rec, 0)[0]
            n = 4 if ".lda." in name else 8
            lb = rec[32:32 + n] + bytes([MARKER]) * (8 - n)
            if ".vst." in name:     # the stored bytes landed (or not) at the record's result
                lb = rec[32:40]
                ok = lb == in_bytes()[WIN_OFF[1]:WIN_OFF[1] + 8]
                got = (ptr & ~0xC0000) | 0x40000 if ok else "marker" if lb[0] == MARKER else (
                    "unknown:" + lb.hex())
            else:
                got = where(h, struct.unpack_from("<H", lb, 0)[0], rec[32:40] if n == 8 else lb)
            width = 4 if ".lda." in name else 32     # every access floors to its width
            rows.append(dict(name=name, lanes=[dict(value=ptr, eff=ptr & ~(width - 1), got=got,
                                                    bytes=lb)],
                             ptr=True))
            continue
        lanes_all = struct.unpack_from("<8I", rec, 0)
        res = rec[32:64]
        width = int(name.split(".4x")[1][:2])
        half = name.split(".4x")[1].split(".")[1]
        first = 4 if half == "hi" else 0
        lanes = []
        for k in range(4):
            v = lanes_all[first + k]
            lb = res[8 * k:8 * k + 8]
            got = where(h, struct.unpack_from("<H", lb, 0)[0], lb)
            lanes.append(dict(value=v, eff=lut_lane_address(v, k, width), got=got, bytes=lb))
        rows.append(dict(name=name, width=width, half=half, lanes=lanes))
    return h, rows


def model_address(eff: int) -> int:
    """Hypothesis under test: bits 0..15 are the offset, bits 16..17 the direction (with 3 =
    own memory: 7 & 3), bits 18..31 ignored. Returns the 20-bit core address."""
    return 0x40000 | (eff & 0x3FFFF)


def report(blob: bytes, names: list[str]) -> list[str]:
    h, rows = decode_records(blob, names)
    out = [f"in={h.inp:#x} out={h.out:#x} far={h.far:#x} north={h.north:#x}"]
    for r in rows:
        out.append(r["name"])
        for k, ln in enumerate(r["lanes"]):
            g = ln["got"]
            gs = f"{g:#07x}" if isinstance(g, int) else g
            m = model_address(ln["eff"])
            ok = ""
            if isinstance(g, int):
                ok = "model" if g == m else f"MODEL {m:#07x}"
                if not r.get("ptr") and ln["bytes"] != expected_bytes(h, g):
                    ok += " BYTES"
            out.append(f"  lane{k} v={ln['value']:08x} eff={ln['eff']:08x} -> {gs} "
                       f"{ln['bytes'].hex()} {ok}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the suite")
    g.add_argument("out_dir")
    a = sub.add_parser("analyze", help="decode one case's output")
    a.add_argument("suite_dir")
    a.add_argument("tag")
    a.add_argument("case")
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        names = generate(Path(args.out_dir))
        print(f"{len(names)} case(s)")
        return 0
    recs = [r.name for r in build_records()[args.case]]
    blob = (Path(args.suite_dir) / "runs" / args.tag / args.case / f"{CORE}.out.bin").read_bytes()
    print("\n".join(report(blob, recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
