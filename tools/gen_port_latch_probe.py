#!/usr/bin/env python3
"""Port-latch probe: what a core read of a neighbour with no memory behind it returns.

On NPU1 a core in row 2 sits directly above the memory-tile row, so the south quadrant of its
20-bit address space (direction bits 16-17 = 0, `0x4xxxx`) has no core data memory behind it.
Silicon still returns data there (`load4x/wrap3` `lut_unwritten`: bytes of an earlier west read
or north store). This probe finds the rule: which earlier access the data comes from (unit,
direction, width, read or write, how long ago) and which of its bytes land where.

Every byte the kernel can move is self-describing: each window of the input is halfwords
`(tag << 12) | j` (j = halfword index inside the window), so the analysis names the window and
the offset of every output halfword without assuming a rule, and since every access in a case
uses its own 32-byte block, the offset names the access.

Buffers and regions
-------------------
  in:  [0,64)            marker 0xEE (pre-loaded into every probe register)
       OWN   tag 1       own-memory reads (and own stores land here)
       WC    tag 2       copied to the west neighbour at the same offset (west reads)
       NC    tag 3       copied to the north neighbour (north reads)
       SD    tag 4       store data: loaded (own) into a register, then stored by a prime
       SC    tag 5       copied to the south neighbour (rows 3..5 only: the control case)
  A region `R` in direction `D` is the address `(in + WIN_OFF[R] + off) ^ DIR_XOR[D]` (`in` is in
  the own quadrant, 0x7xxxx). Stores to a neighbour go to its SD offsets, which nothing reads.

  out: [0,64)            header, u32: in, out
       [64,..)           one record per entry: four 32-byte slots, marker-filled first.
                         slot 0 = what the prime read (or stored), slots 1..3 = probes 1..3
                         (a 32-bit probe fills the first word, a 128-bit probe the first 16 bytes)

Record = prime accesses, then a gap (nops and own accesses), then up to three south probes,
each `PROBE_GAP` nops apart, then the slots are stored. Every pointer and every probe register
is set up before the prime, so between the prime and the probes the core does nothing but what
the record names.
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

MARKER = 0xEE
MARK_BYTES = 64
WIN_BYTES = 2048
REGIONS = ("OWN", "WC", "NC", "SD", "SC")
TAG = {r: i + 1 for i, r in enumerate(REGIONS)}
WIN_OFF = {r: MARK_BYTES + i * WIN_BYTES for i, r in enumerate(REGIONS)}
IN_BYTES = MARK_BYTES + len(REGIONS) * WIN_BYTES
BLOCKS = WIN_BYTES // 32
# The copy of a window into a neighbour stores every block; records only use blocks below
# this, so the last setup store (block BLOCKS-1) is never confused with a record's access.
REC_BLOCKS = BLOCKS - 1
# Ring flush at kernel start: RING_DEPTH vlda then RING_DEPTH vldb of OWN lines from the top of
# the window, so no output depends on what ran on the core before (the rings survive launches).
RING_DEPTH = 6
FLUSH_A = tuple(REC_BLOCKS - 2 * RING_DEPTH + k for k in range(RING_DEPTH))
FLUSH_B = tuple(REC_BLOCKS - RING_DEPTH + k for k in range(RING_DEPTH))
OWN_LIMIT = FLUSH_A[0]          # records use OWN blocks below this
DIR_XOR = {"own": 0x00000, "N": 0x10000, "W": 0x20000, "S": 0x30000}
HDR_BYTES = 64
SLOT = 32
SLOTS = 4
REC_BYTES = SLOT * SLOTS
NOPS = 8            # a load's data is in its register NOPS bundles later
PROBE_GAP = NOPS

LOADS = {"vlda": 32, "vldb": 32, "lda": 4, "vldb128": 16, "vlda128": 16, "lda.u8": 1,
         "lda.s16": 2, "ldq": 16}
SCALAR_LOADS = ("lda", "lda.u8", "lda.s16")
# Loads with no destination the probe records: they only show what they do to the ring.
SIDE_LOADS = {"sparse": "vldb.sparse.reset.8 [{p}]",
              "unpack": "vldb.unpack.s16.s8 x2, [{p}, dj7]",
              "ups": "vlda.ups.s32.s16 bml0, s0, [{p}, #0]",
              "conv": "vlda.conv.fp32.bf16 bml0, [{p}, #0]"}
LOADS.update({k: 32 for k in SIDE_LOADS})
STORES = {"vst": 32, "st": 4, "vst128": 16}


def in_bytes() -> bytes:
    out = bytearray([MARKER] * MARK_BYTES)
    for r in REGIONS:
        out += b"".join(struct.pack("<H", (TAG[r] << 12) | j) for j in range(WIN_BYTES // 2))
    assert len(out) == IN_BYTES
    return bytes(out)


@dataclass(frozen=True)
class Access:
    """One memory access. `region` and `blk` give the offset: the block's 32 bytes inside the
    region's window, plus `sub` bytes; `dir` picks the quadrant the address points into."""
    kind: str                   # a key of LOADS or STORES
    dir: str                    # own / W / N / S
    region: str                 # a key of WIN_OFF
    blk: int
    sub: int = 0

    @property
    def nbytes(self) -> int:
        return {**LOADS, **STORES}[self.kind]

    @property
    def is_store(self) -> bool:
        return self.kind in STORES

    @property
    def win_off(self) -> int:
        """Offset inside the region's window."""
        return 32 * self.blk + self.sub

    def offset(self) -> int:
        """Offset inside the in buffer (and so inside any quadrant, relative to `in`)."""
        return WIN_OFF[self.region] + self.win_off

    def address(self, inp: int) -> int:
        return ((inp + self.offset()) ^ DIR_XOR[self.dir]) & 0xFFFFF

    @property
    def name(self) -> str:
        sub = f"+{self.sub}" if self.sub else ""
        return f"{self.kind}.{self.dir}.{self.region}{self.blk}{sub}"


@dataclass(frozen=True)
class Record:
    label: str
    primes: tuple[Access, ...]
    probes: tuple[Access, ...]
    gap: int = NOPS                         # nops after the last prime (before `between`)
    between: tuple[Access, ...] = ()        # own accesses between the prime and the probes
    probe_gap: int = PROBE_GAP

    def __post_init__(self):
        assert 1 <= len(self.probes) <= SLOTS - 1, self.label
        assert all(not p.is_store for p in self.probes), self.label
        assert all(b.dir == "own" for b in self.between), self.label


# -- the records ---------------------------------------------------------------

class Alloc:
    """Hands out a fresh 32-byte block per region, so every access in a case is unique."""

    def __init__(self):
        self.next = {r: 0 for r in REGIONS}

    def __call__(self, region: str) -> int:
        b = self.next[region]
        assert b < (OWN_LIMIT if region == "OWN" else REC_BLOCKS), f"region {region} out of blocks"
        self.next[region] = b + 1
        return b


def _src_region(kind: str, dir: str) -> str:
    """The region a prime of this kind and direction touches: reads read the content the
    setup put there, stores go to the neighbour's (unread) SD offsets."""
    if kind in STORES:
        return "OWN" if dir == "own" else "SD"
    return {"own": "OWN", "W": "WC", "N": "NC", "S": "SC"}[dir]


PRIMES = tuple((k, d) for d in ("W", "N") for k in ("vlda", "vldb", "lda", "vst", "st")) + (
    ("vlda", "own"), ("vst", "own"))
PROBE_KINDS = ("vlda", "vldb", "lda")


def unit_records() -> list[Record]:
    """Every prime unit x direction x read/write, each followed by one south probe of each
    load unit, and a probe-after-probe record per unit."""
    a = Alloc()
    recs = []
    for k, d in PRIMES:
        for q in PROBE_KINDS:
            p = Access(k, d, _src_region(k, d), a(_src_region(k, d)))
            recs.append(Record(f"{k}.{d}>{q}", (p,), (Access(q, "S", "SC", a("SC")),)))
    for q in PROBE_KINDS:
        recs.append(Record(f"none>{q}", (), (Access(q, "S", "SC", a("SC")),)))
    return recs


def bytes_records() -> list[Record]:
    """Which bytes of the latched value a south read returns: probe offsets inside the line,
    probe addresses equal to and away from the prime's, narrow primes."""
    a = Alloc()
    recs = []

    def wread(kind="vlda", sub=0):
        return Access(kind, "W", "WC", a("WC"), sub)

    for sub in range(0, 32, 4):
        recs.append(Record(f"vlda.W>lda+{sub}", (wread(),),
                           (Access("lda", "S", "SC", a("SC"), sub),)))
    for sub in (0, 16):
        recs.append(Record(f"vlda.W>vldb128+{sub}", (wread(),),
                           (Access("vldb128", "S", "SC", a("SC"), sub),)))
    # probe at the prime's own offset, the next line, 0x100 on, the next logical bank
    for rel, label in ((0, "same"), (32, "next"), (0x100, "+0x100"), (0x4000, "+0x4000")):
        p = wread()
        recs.append(Record(f"vlda.W>vlda.{label}", (p,),
                           (Access("vlda", "S", "WC", p.blk + rel // 32),)))
    # narrow primes
    for kind, sub in (("lda", 0), ("lda", 12), ("lda", 28), ("vldb128", 0), ("vldb128", 16)):
        recs.append(Record(f"{kind}.W+{sub}>vlda", (wread(kind, sub),),
                           (Access("vlda", "S", "SC", a("SC")),)))
    for kind, sub in (("st", 0), ("st", 8), ("vst128", 16)):
        recs.append(Record(f"{kind}.N+{sub}>vlda", (Access(kind, "N", "SD", a("SD"), sub),),
                           (Access("vlda", "S", "SC", a("SC")),)))
    # three probes in a row after one prime: does a probe disturb the latch?
    recs.append(Record("vlda.W>vlda,vldb,lda", (wread(),),
                       (Access("vlda", "S", "SC", a("SC")), Access("vldb", "S", "SC", a("SC")),
                        Access("lda", "S", "SC", a("SC"), 8))))
    return recs


def age_records() -> list[Record]:
    """How long a value stays: nops between prime and probe, own accesses in between, and two
    neighbour accesses in a row."""
    a = Alloc()
    recs = []
    for g in (0, 1, 2, 3, 4, 5, 6, 7, 16, 64):
        recs.append(Record(f"vlda.W~{g}>vlda", (Access("vlda", "W", "WC", a("WC")),),
                           (Access("vlda", "S", "SC", a("SC")),), gap=g))
    for g in (0, 2, 4, 6):
        recs.append(Record(f"vst.N~{g}>vlda", (Access("vst", "N", "SD", a("SD")),),
                           (Access("vlda", "S", "SC", a("SC")),), gap=g))
    for kinds in (("vlda",), ("vldb",), ("lda",), ("vst",), ("st",), ("vlda", "vldb", "vst", "lda")):
        between = tuple(Access(k, "own", "OWN", a("OWN")) for k in kinds)
        recs.append(Record(f"vlda.W~own:{'+'.join(kinds)}>vlda",
                           (Access("vlda", "W", "WC", a("WC")),),
                           (Access("vlda", "S", "SC", a("SC")),), between=between))
    pairs = ((("vlda", "W"), ("vlda", "N")), (("vlda", "N"), ("vlda", "W")),
             (("vst", "N"), ("vlda", "W")), (("vlda", "W"), ("vst", "N")),
             (("vlda", "W"), ("vldb", "W")), (("vldb", "W"), ("vlda", "W")),
             (("vlda", "W"), ("lda", "N")))
    for (k1, d1), (k2, d2) in pairs:
        p1 = Access(k1, d1, _src_region(k1, d1), a(_src_region(k1, d1)))
        p2 = Access(k2, d2, _src_region(k2, d2), a(_src_region(k2, d2)))
        for q in ("vlda", "vldb"):
            recs.append(Record(f"{k1}.{d1},{k2}.{d2}>{q}", (p1, p2),
                               (Access(q, "S", "SC", a("SC")),)))
            p1 = Access(k1, d1, _src_region(k1, d1), a(_src_region(k1, d1)))
            p2 = Access(k2, d2, _src_region(k2, d2), a(_src_region(k2, d2)))
    return recs


# (case name, tile (logical col, row), records, regions the setup copies to which neighbour)
SETUP_ROW2 = (("WC", "W"), ("NC", "N"))
SETUP_ROW3 = (("WC", "W"), ("NC", "N"), ("SC", "S"))
SETUP_TOP = (("WC", "W"), ("SC", "S"))     # row 5: nothing to the north


def build_cases() -> dict[str, tuple[tuple[int, int], list[Record], tuple]]:
    unit = unit_records()
    return {
        "unit": ((1, 2), unit, SETUP_ROW2),
        "bytes": ((1, 2), bytes_records(), SETUP_ROW2),
        "age": ((1, 2), age_records(), SETUP_ROW2),
        "ctrl": ((1, 3), unit, SETUP_ROW3),
    }


# -- assembly ------------------------------------------------------------------

def _nops(n: int) -> list[str]:
    return ["  nop"] * n


def _frame(body: list[str]) -> str:
    pro = [".text", ".globl test_kernel", "test_kernel:"]
    epi = ["  nop"] * 7 + ["  ret lr"] + ["  nop"] * 5
    return "\n".join(pro + body + epi) + "\n"


def _ptr(preg: str, off: int, dir: str) -> list[str]:
    """preg <- (in + off) ^ DIR_XOR[dir]; r20 holds in."""
    return [f"  movxm r2, #{off}", f"  movxm r3, #{DIR_XOR[dir]}", *_nops(2),
            "  add r1, r20, r2", *_nops(1), "  xor r1, r1, r3", *_nops(1), f"  mov {preg}, r1"]


_PROBE_VREG = ("wl2", "wl4", "wl6")
_PROBE_SREG = ("r10", "r11", "r12")
_PTR_PRIME = ("p5", "p6")       # up to two primes
_PTR_SRC = ("p2", "p3")         # store data source of each prime
_PTR_PROBE = ("p7", "p2", "p3")  # probe pointers (p2/p3 reused: set after the primes' data loads)
_PTR_BETWEEN = "p6"


def _load(kind: str, vreg: str, sreg: str, preg: str) -> str:
    if kind in SIDE_LOADS:
        return "  " + SIDE_LOADS[kind].format(p=preg)
    if kind in SCALAR_LOADS:
        return f"  {kind} {sreg}, [{preg}, #0]"
    if kind == "ldq":
        return f"  lda q0, [{preg}, #0]"
    if kind in ("vldb128", "vlda128"):
        # the 128-bit loads have no [p, #imm] form: index by dj7, which holds 0
        return f"  {kind[:4]}.128 {vreg}, [{preg}, dj7]"
    return f"  {kind} {vreg}, [{preg}, #0]"


def _store(kind: str, vreg: str, sreg: str, preg: str) -> str:
    if kind == "st":
        return f"  st {sreg}, [{preg}, #0]"
    return f"  {'vst.128' if kind == 'vst128' else 'vst'} {vreg}, [{preg}, #0]"


def _slots_asm(rec: Record) -> list[str]:
    """Marker-fill the four slots, then store the prime's data and each probe's result."""
    body = [*_nops(NOPS), "  vlda wl10, [p0, #0]", *_nops(NOPS)]
    body += [f"  vst wl10, [p4, #{SLOT * s}]" for s in range(SLOTS)]
    if rec.primes:
        p = rec.primes[0]
        body.append("  st r9, [p4, #0]" if p.kind in ("lda", "st") else "  vst wl0, [p4, #0]")
    for j, q in enumerate(rec.probes):
        if q.kind == "lda":
            body.append(f"  st {_PROBE_SREG[j]}, [p4, #{SLOT * (j + 1)}]")
        else:
            body.append(f"  vst {_PROBE_VREG[j]}, [p4, #{SLOT * (j + 1)}]")
    return body + ["  padda [p4], #128", *_nops(2)]


def record_asm(rec: Record, src_blocks: list[int]) -> list[str]:
    """One record. Pointer setup and every own preparatory load happen first; from the prime
    to the last probe the core issues only the record's accesses and nops."""
    body = [f"  // {rec.label}"]
    body += ["  vlda wl0, [p0, #0]", "  vlda wl8, [p0, #0]", "  vlda wl2, [p0, #0]",
             "  vlda wl4, [p0, #0]", "  vlda wl6, [p0, #0]", "  movxm r9, #-286331154",
             "  movxm r10, #-286331154", "  movxm r11, #-286331154", "  movxm r12, #-286331154",
             "  movxm r13, #-286331154", "  movxm r14, #-286331154", "  vlda wl10, [p0, #0]",
             *_nops(NOPS)]
    vreg = ("wl0", "wl8")
    sreg = ("r9", "r13")
    srcs = [Access("lda" if p.kind == "st" else "vlda", "own", "SD", src_blocks[i],
                   p.sub if p.is_store else 0)
            if p.is_store else None for i, p in enumerate(rec.primes)]
    for i, s in enumerate(srcs):
        if s is not None:
            body += _ptr(_PTR_SRC[i], s.offset(), "own")
    body += _nops(NOPS)
    for i, s in enumerate(srcs):
        if s is not None:
            body.append(_load("lda" if s.kind == "lda" else
                              ("vldb128" if rec.primes[i].kind == "vst128" else "vlda"),
                              vreg[i], sreg[i], _PTR_SRC[i]))
    body += _nops(NOPS)
    for i, p in enumerate(rec.primes):
        body += _ptr(_PTR_PRIME[i], p.offset(), p.dir)
    # probe pointers: p7, then p2 / p3 (free once the store data is loaded)
    for j, q in enumerate(rec.probes):
        body += _ptr(_PTR_PROBE[j], q.offset(), q.dir)
    if rec.between:
        assert len(rec.primes) == 1, rec.label
        body += _ptr(_PTR_BETWEEN, rec.between[0].offset(), "own")
    body += _nops(NOPS)
    for i, p in enumerate(rec.primes):
        if p.is_store:
            body.append(_store(p.kind, vreg[i], sreg[i], _PTR_PRIME[i]))
        else:
            body.append(_load(p.kind, vreg[i], sreg[i], _PTR_PRIME[i]))
        if i + 1 < len(rec.primes):
            body += _nops(NOPS)
    body += _nops(rec.gap)
    body += _between_asm(rec)
    for j, q in enumerate(rec.probes):
        body.append(_load(q.kind, _PROBE_VREG[j], _PROBE_SREG[j], _PTR_PROBE[j]))
        body += _nops(rec.probe_gap)
    return body + _slots_asm(rec)


def _between_asm(rec: Record) -> list[str]:
    """Own accesses between prime and probe, all through p6 with immediate offsets from the
    first one's block (the blocks are consecutive: the allocator hands them out in order)."""
    if not rec.between:
        return []
    first = rec.between[0]
    out = []
    for b in rec.between:
        imm = b.offset() - first.offset()
        if b.is_store:
            out.append(f"  {'st r14' if b.kind == 'st' else 'vst wl10'}, [p6, #{imm}]")
        else:
            reg = {"lda": "r14", "vlda": "wl10", "vldb": "wl10"}[b.kind]
            out.append(f"  {b.kind} {reg}, [p6, #{imm}]")
        out += _nops(NOPS)
    return out


def _copy(region: str, dir: str) -> list[str]:
    """Copy a window from the in buffer to the same offset in a neighbour, 8 lines at a time
    (eight loads in flight, then eight stores)."""
    body = _ptr("p2", WIN_OFF[region], "own") + _ptr("p3", WIN_OFF[region], dir) + _nops(NOPS)
    regs = ("wl0", "wl2", "wl4", "wl6", "wl8", "wl10", "wh0", "wh2")
    for _ in range(BLOCKS // len(regs)):
        body += [f"  vlda {r}, [p2], #32" for r in regs] + _nops(NOPS)
        body += [f"  vst {r}, [p3], #32" for r in regs]
    return body + _nops(NOPS)


def prologue(setup: tuple) -> list[str]:
    """Header, neighbour setup copies, ring flush, record cursor p4."""
    body = ["  movxm dj7, #0", "  mov r20, p0", "  mov r2, p0", "  mov r3, p1", *_nops(3),
            "  st r2, [p1, #0]", "  st r3, [p1, #4]"]
    for region, dir in setup:
        body += _copy(region, dir)
    body += _ptr("p2", WIN_OFF["OWN"] + 32 * FLUSH_A[0], "own") + _nops(NOPS)
    body += ["  vlda wl0, [p2], #32" for _ in FLUSH_A] + _nops(NOPS)
    body += ["  vldb wl0, [p2], #32" for _ in FLUSH_B] + _nops(NOPS)
    body += ["  mov r1, p1", f"  movxm r2, #{HDR_BYTES}", *_nops(2), "  add r1, r1, r2",
             "  mov p4, r1", *_nops(NOPS)]
    return body


def setup_events(setup: tuple, add) -> None:
    for region, dir in setup:
        for b in range(BLOCKS):
            add(-1, "setup", Access("vst", dir, region, b), (region, 32 * b))
    for b in (*FLUSH_A, *FLUSH_B):
        add(-1, "flush", Access("vlda" if b in FLUSH_A else "vldb", "own", "OWN", b),
            ("OWN", 32 * b))


def kernel_asm(records: list[Record], setup: tuple) -> str:
    body = prologue(setup)
    for rec, blocks in zip(records, store_blocks(records)):
        body += record_asm(rec, blocks)
    return _frame(body)


def store_blocks(records: list[Record]) -> list[list[int]]:
    """The own SD block each store prime loads its data from: counted down from the top of
    the window, so no neighbour store (which counts up) shares an offset with it."""
    out, n = [], 0
    for rec in records:
        bl = []
        for p in rec.primes:
            bl.append(REC_BLOCKS - 1 - n if p.is_store else -1)
            n += p.is_store
        out.append(bl)
    return out


def out_bytes(records: list[Record]) -> int:
    return HDR_BYTES + REC_BYTES * len(records)


def generate(out_dir: Path) -> list[str]:
    names = []
    for name, (tile, recs, setup) in build_cases().items():
        check_blocks(recs)
        suite.write_single_core_case(out_dir, name, kernel_asm(recs, setup), input_bytes=in_bytes(),
                                     out_bytes=out_bytes(recs), tile=tile,
                                     meta=dict(name=name, records=[r.label for r in recs]))
        names.append(name)
    for name, (tile, phases, setup) in sweep_cases().items():
        suite.write_single_core_case(out_dir, name, sweep_asm(phases, setup),
                                     input_bytes=in_bytes(),
                                     out_bytes=HDR_BYTES + 32 * sweep_probes(phases), tile=tile,
                                     meta=dict(name=name, phases=[str(p) for p in phases]))
        names.append(name)
    for name, blocks in ring_cases().items():
        tile, setup = ring_case_tile(name)
        suite.write_single_core_case(out_dir, name, ring_asm(blocks, setup),
                                     input_bytes=in_bytes(),
                                     out_bytes=HDR_BYTES + RING_REC * len(blocks), tile=tile,
                                     meta=dict(name=name, blocks=[b.label for b in blocks]))
        names.append(name)
    suite.write_suite_json(out_dir, "port-latch", "gen_port_latch_probe.py", names)
    return names


def check_blocks(records: list[Record]) -> None:
    """Every access in a case touches its own bytes (the analysis relies on it)."""
    seen: dict[tuple[str, str, int], str] = {}
    sd = [b for bl in store_blocks(records) for b in bl if b >= 0]
    for rec in records:
        for acc in (*rec.primes, *rec.between, *rec.probes):
            if acc.dir == "S":
                continue
            for o in range(acc.win_off, acc.win_off + acc.nbytes, 4):
                key = (acc.dir, acc.region, o)
                assert key not in seen, f"{rec.label}: {acc.name} reuses {seen[key]}"
                seen[key] = f"{rec.label}:{acc.name}"
            if acc.region == "SD":
                assert acc.blk not in sd or acc.dir != "own", rec.label
    assert min(sd, default=REC_BLOCKS) > max(
        (a.blk for r in records for a in r.primes if a.region == "SD"), default=-1)


# -- sweeps --------------------------------------------------------------------
#
# A sweep is a straight sequence of phases over consecutive 32-byte lines, each access 8 nops
# from the next: neighbour stores, neighbour reads, and south probes whose results go to
# consecutive 32-byte slots of the output (after the header). It shows how a south probe picks
# its stale data across many lines at once.

@dataclass(frozen=True)
class Phase:
    op: str                     # probe / store / read
    kind: str                   # load or store mnemonic key
    dir: str
    region: str
    first: int                  # first block
    n: int
    src: int = 0                # store: first own SD block of the data (counted from the top)


def _phase_asm(ph: Phase) -> list[str]:
    body = _ptr("p7", WIN_OFF[ph.region] + 32 * ph.first, ph.dir)
    if ph.op == "store":
        body += _ptr("p6", WIN_OFF["SD"] + 32 * (BLOCKS - ph.src - ph.n), "own")
    body += _nops(NOPS)
    if ph.op == "probe":
        for _ in range(ph.n):
            body += ["  vlda wl2, [p0, #0]", *_nops(NOPS), f"  {ph.kind} wl2, [p7], #32",
                     *_nops(NOPS), "  vst wl2, [p4], #32"]
        return body
    if ph.op == "read":
        for _ in range(ph.n):
            body += [f"  {ph.kind} wl0, [p7], #32", *_nops(NOPS)]
        return body
    regs = ("wl0", "wl4", "wl6", "wl8")
    for g in range(0, ph.n, len(regs)):
        k = min(len(regs), ph.n - g)
        body += [f"  vlda {r}, [p6], #32" for r in regs[:k]] + _nops(NOPS)
        for r in regs[:k]:
            body += [f"  vst {r}, [p7], #32", *_nops(NOPS)]
    return body


def sweep_asm(phases: tuple[Phase, ...], setup: tuple) -> str:
    body = prologue(setup)
    for ph in phases:
        body += [f"  // {ph}"] + _phase_asm(ph)
    return _frame(body)


def sweep_events(phases: tuple[Phase, ...], setup: tuple) -> list[Event]:
    ev: list[Event] = []

    def add(rec, role, acc, data):
        ev.append(Event(len(ev), rec, role, acc, data))

    setup_events(setup, add)
    for i, ph in enumerate(phases):
        for k in range(ph.n):
            acc = Access(ph.kind, ph.dir, ph.region, ph.first + k)
            if ph.op == "probe":
                add(i, "probe", acc, None)
            elif ph.op == "read":
                add(i, "read", acc, (ph.region, 32 * (ph.first + k)))
            else:
                add(i, "store", acc, ("SD", 32 * (BLOCKS - ph.src - ph.n + k)))
    return ev


def sweep_probes(phases) -> int:
    return sum(ph.n for ph in phases if ph.op == "probe")


def sweep_cases() -> dict[str, tuple[tuple[int, int], tuple[Phase, ...], tuple]]:
    P = lambda first, kind="vldb": Phase("probe", kind, "S", "SC", first, 32)
    phases = (P(0), P(0),
              Phase("store", "vst", "N", "SD", 0, 32, src=0), P(0), P(32),
              Phase("read", "vlda", "W", "WC", 0, 32), P(0),
              Phase("read", "vldb", "W", "WC", 32, 31), P(0), P(32))
    return {"sweep": ((1, 2), phases, SETUP_ROW2)}


def analyze_sweep(blob: bytes, phases, setup) -> list[str]:
    inp, outp = parse_header(blob)
    evs = sweep_events(phases, setup)
    probes = [e for e in evs if e.role == "probe"]
    out = [f"in={inp:#x} out={outp:#x}"]
    for k, e in enumerate(probes):
        slot = blob[HDR_BYTES + 32 * k:HDR_BYTES + 32 * (k + 1)]
        hws = struct.unpack("<16H", slot)
        descs = [describe(halfword_source(h), evs, e.seq) for h in hws]
        out.append(f"ph{e.rec} {e.acc.name}@{e.acc.address(inp):05x}: " + compress(descs))
    return out


# -- ring blocks ---------------------------------------------------------------
#
# A ring block: fill (RING_DEPTH vldb reads of fresh neighbour lines), then the operations
# under test, then RING_PROBES south vldb probes. If an operation neither writes the ring nor
# moves its pointer, the probes return the fill lines oldest first; one that moves the pointer
# rotates them; one that writes shows its own data in the slot it wrote.

RING_PROBES = 7
GATHER = {"g64lo": "vldb.4x64.lo", "g16hi": "vldb.4x16.hi", "g32lo": "vldb.4x32.lo"}


@dataclass(frozen=True)
class RingOp:
    kind: str                   # LOADS / STORES key, or a GATHER key
    dir: str
    sub: int = 0
    nops: int = NOPS            # nops after it
    # gathers: (direction, region) of each of the four lane lines; default all (dir, its region)
    lanes: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class RingBlock:
    label: str
    fill_dir: str = "W"
    ops: tuple[RingOp, ...] = ()
    fill_gap: int = NOPS
    probe_gap: int = NOPS       # nops before the first probe (after the last op or fill line)
    port: str = "B"             # the ring under test: fills and probes are vldb (B) or vlda (A)
    probe_dir: str = "S"        # the direction with no core memory behind it

    @property
    def unit(self) -> str:
        return "vldb" if self.port == "B" else "vlda"

    @property
    def aux(self) -> str:
        """The vector load the block's own bookkeeping uses: the other port's."""
        return "vlda" if self.port == "B" else "vldb"


def ring_block_plan(blk: RingBlock, a: Alloc):
    """Allocate every line the block touches. Returns (fill accesses, [(op, accesses)], probes)."""
    fill_region = _src_region(blk.unit, blk.fill_dir)
    fills = [Access(blk.unit, blk.fill_dir, fill_region, a(fill_region)) for _ in range(RING_DEPTH)]
    ops = []
    for op in blk.ops:
        if op.kind in GATHER:
            lanes = op.lanes or ((op.dir, _src_region("vlda", op.dir)),) * 4
            accs = [Access("vldb", d, r, a(r)) for d, r in lanes]
        else:
            region = _src_region(op.kind, op.dir)
            accs = [Access(op.kind, op.dir, region, a(region), op.sub)]
            src = Access("vlda", "own", "SD", REC_BLOCKS - 1 - a("SD_TOP"), 0) \
                if op.kind in STORES else None
            accs.append(src)
        ops.append((op, accs))
    probes = [Access(blk.unit, blk.probe_dir, "SC", a("SC")) for _ in range(RING_PROBES)]
    return fills, ops, probes


class RingAllocator(Alloc):
    """South lines have no memory behind them, so they need not be unique: they wrap."""

    def __init__(self):
        super().__init__()
        self.next["SD_TOP"] = 0

    def __call__(self, region: str) -> int:
        if region == "SC":
            b = self.next["SC"]
            self.next["SC"] = (b + 1) % REC_BLOCKS
            return b
        return super().__call__(region)


def ring_block_asm(blk: RingBlock, plan) -> list[str]:
    fills, ops, probes = plan
    body = [f"  // {blk.label}"]
    # store data and gather lanes first (own work, long before the fill)
    for i, (op, accs) in enumerate(ops):
        if op.kind in STORES:
            src = accs[1]
            off = src.offset() + (op.sub & ~3 if op.kind == "st" else op.sub & 16)
            if blk.port == "A" and op.kind == "st":
                # no port-A load in an A block: the word is built as an immediate
                word = struct.unpack_from("<i", in_bytes(), off)[0]
                body += [f"  movxm r13, #{word}"]
            else:
                body += _ptr("p6", off, "own") + _nops(NOPS)
                body.append("  lda r13, [p6, #0]" if op.kind == "st" else
                            ("  vldb.128 wl8, [p6, dj7]" if op.kind == "vst128" else
                             f"  {blk.aux} wl8, [p6, #0]"))
            body += _nops(NOPS)
        if op.kind in GATHER:
            # lanes: k = line k (k = 0..3), lanes 4..7 repeat them; written to the slot area of
            # the output (overwritten later by the slot stores) and loaded into wl4
            for k in range(8):
                acc = accs[k % 4]
                body += [f"  movxm r2, #{acc.offset()}", f"  movxm r3, #{DIR_XOR[acc.dir]}",
                         *_nops(2), "  add r1, r20, r2", *_nops(1), "  xor r1, r1, r3",
                         *_nops(1), f"  st r1, [p4, #{4 * k}]"]
            body += _nops(2) + [f"  {blk.aux} wl4, [p4, #0]"] + _nops(NOPS)
    assert sum(op.kind in STORES for op, _ in ops) <= 1
    assert sum(op.kind in GATHER for op, _ in ops) <= 1
    body += [f"  {blk.aux} wl6, [p0, #0]", "  movxm r12, #-286331154", *_nops(NOPS)]
    body += _ptr("p7", fills[0].offset(), fills[0].dir)
    body += _ptr("p5", probes[0].offset(), probes[0].dir)
    body += _nops(NOPS)
    for k in range(RING_DEPTH):
        body.append(f"  {blk.unit} wl0, [p7], #32")
        if k + 1 < RING_DEPTH:
            body += _nops(blk.fill_gap)
    for op, accs in ops:
        body += _nops(NOPS)
        if op.kind in GATHER:
            body.append(f"  {GATHER[op.kind]} wl6, wl4")
        else:
            acc = accs[0]
            body += _ptr("p6", acc.offset(), acc.dir) + _nops(NOPS)
            if op.kind in STORES:
                body.append(_store(op.kind, "wl8", "r13", "p6"))
            else:
                body.append(_load(op.kind, "wl6", "r12", "p6"))
    body += _nops(blk.probe_gap)
    first = True
    for k in range(RING_PROBES):
        if not first:
            body += [f"  {blk.aux} wl2, [p0, #0]", *_nops(NOPS)]
        body += [f"  {blk.unit} wl2, [p5], #32", *_nops(NOPS), f"  vst wl2, [p4, #{32 * (k + 1)}]"]
        first = False
    # slot 0: the op's result (marker if none), then advance
    if any(op.kind in SCALAR_LOADS for op, _ in ops):
        body += [f"  {blk.aux} wl6, [p0, #0]", *_nops(NOPS), "  vst wl6, [p4, #0]",
                 "  st r12, [p4, #0]"]
    elif any(op.kind == "ldq" for op, _ in ops):
        body += [f"  {blk.aux} wl6, [p0, #0]", *_nops(NOPS), "  vst wl6, [p4, #0]",
                 "  st q0, [p4, #0]"]
    else:
        body += ["  vst wl6, [p4, #0]"]
    return body + [f"  padda [p4], #{32 * (RING_PROBES + 1)}", *_nops(2)]


def ring_cases() -> dict[str, list[RingBlock]]:
    R = RingOp
    a = [RingBlock("none"),
         RingBlock("vldb.own", ops=(R("vldb", "own"),)),
         RingBlock("vlda.W", "N", ops=(R("vlda", "W"),)),
         RingBlock("vlda.own", ops=(R("vlda", "own"),)),
         RingBlock("vst.W", "N", ops=(R("vst", "W"),)),
         RingBlock("vst.own", ops=(R("vst", "own"),)),
         RingBlock("lda.W", "N", ops=(R("lda", "W"),)),
         RingBlock("st.N", ops=(R("st", "N"),)),
         RingBlock("vldb.N", ops=(R("vldb", "N"),))]
    b = [RingBlock("vldb128.W+0", "N", ops=(R("vldb128", "W"),)),
         RingBlock("vldb128.W+16", "N", ops=(R("vldb128", "W", 16),)),
         RingBlock("vldb.S", ops=(R("vldb", "S"),)),
         RingBlock("vldb128.S", "N", ops=(R("vldb128", "S"),)),
         RingBlock("g64lo.W", ops=(R("g64lo", "W"),)),
         RingBlock("g64lo.S", "N", ops=(R("g64lo", "S"),)),
         RingBlock("g16hi.S", ops=(R("g16hi", "S"),)),
         RingBlock("fill-back-to-back", "N", fill_gap=0),
         RingBlock("probe-gap0", probe_gap=0),
         RingBlock("vldb.own x2", "N", ops=(R("vldb", "own"), R("vldb", "own")))]
    SW, SO, OW = ("S", "WC"), ("S", "OWN"), ("own", "OWN")
    c = [RingBlock("g64lo.S@WC", ops=(R("g64lo", "S", lanes=(SW,) * 4),)),
         RingBlock("g64lo.S@OWN", "N", ops=(R("g64lo", "S", lanes=(SO,) * 4),)),
         RingBlock("g16hi.S@WC", ops=(R("g16hi", "S", lanes=(SW,) * 4),)),
         RingBlock("g32lo.S@WC", "N", ops=(R("g32lo", "S", lanes=(SW,) * 4),)),
         RingBlock("g64lo.own01,S23", ops=(R("g64lo", "S", lanes=(OW, OW, SW, SW)),)),
         RingBlock("g64lo.S01,own23", "N", ops=(R("g64lo", "S", lanes=(SW, SW, OW, OW)),)),
         RingBlock("g64lo.S0,own123", ops=(R("g64lo", "S", lanes=(SW, OW, OW, OW)),)),
         RingBlock("g64lo.own012,S3", "N", ops=(R("g64lo", "S", lanes=(OW, OW, OW, SW)),)),
         RingBlock("g64lo.W01,N23", ops=(R("g64lo", "W", lanes=(("W", "WC"),) * 2 +
                                                               (("N", "NC"),) * 2),))]
    A = lambda label, fill="W", ops=(): RingBlock(label, fill, ops, port="A")
    d = [A("A:none"),
         A("A:vlda.own", ops=(R("vlda", "own"),)),
         A("A:vldb.W", "N", ops=(R("vldb", "W"),)),
         A("A:lda.W+0", "N", ops=(R("lda", "W"),)),
         A("A:lda.W+20", ops=(R("lda", "W", 20),)),
         A("A:lda.own", ops=(R("lda", "own"),)),
         A("A:lda.u8.W+5", "N", ops=(R("lda.u8", "W", 5),)),
         A("A:lda.s16.W+22", ops=(R("lda.s16", "W", 22),)),
         A("A:ldq.W+0", "N", ops=(R("ldq", "W"),))]
    e = [A("A:ldq.W+16", ops=(R("ldq", "W", 16),)),
         A("A:vst.W", "N", ops=(R("vst", "W"),)),
         A("A:st.N", ops=(R("st", "N"),)),
         A("A:vlda.S", "N", ops=(R("vlda", "S"),)),
         A("A:lda.S+0", ops=(R("lda", "S"),)),
         A("A:lda.S+20", "N", ops=(R("lda", "S", 20),)),
         A("A:ldq.S+16", ops=(R("ldq", "S", 16),)),
         A("A:g64lo.S", "N", ops=(R("g64lo", "S", lanes=(SW,) * 4),)),
         A("A:g64lo.W", ops=(R("g64lo", "W"),))]
    top = [RingBlock("top:B.N", probe_dir="N"),
           RingBlock("top:A.N", port="A", probe_dir="N"),
           RingBlock("top:B.N own-fill", "own", probe_dir="N")]
    f = [RingBlock("B:sparse.own", ops=(R("sparse", "own"),)),
         RingBlock("B:unpack.own", "N", ops=(R("unpack", "own"),)),
         A("A:ups.own", ops=(R("ups", "own"),)),
         A("A:conv.own", "N", ops=(R("conv", "own"),)),
         RingBlock("B:unpack.S", ops=(R("unpack", "S"),)),
         A("A:ups.S", "N", ops=(R("ups", "S"),))]
    return {"ring_a": a, "ring_b": b, "ring_c": c, "ring_d": d, "ring_e": e, "ring_top": top,
            "ring_f": f}


RING_TILE = {"ring_top": ((1, 5), SETUP_TOP)}


def ring_case_tile(name: str):
    return RING_TILE.get(name, ((1, 2), SETUP_ROW2))


def ring_plans(blocks: list[RingBlock]):
    a = RingAllocator()
    return [ring_block_plan(b, a) for b in blocks]


def ring_asm(blocks: list[RingBlock], setup: tuple) -> str:
    body = prologue(setup)
    for blk, plan in zip(blocks, ring_plans(blocks)):
        body += ring_block_asm(blk, plan)
    return _frame(body)


RING_REC = 32 * (RING_PROBES + 1)


def ring_events(blocks, setup) -> list[Event]:
    ev: list[Event] = []

    def add(rec, role, acc, data):
        ev.append(Event(len(ev), rec, role, acc, data))

    setup_events(setup, add)
    for i, (blk, (fills, ops, probes)) in enumerate(zip(blocks, ring_plans(blocks))):
        for f in fills:
            add(i, "fill", f, (f.region, f.win_off))
        for op, accs in ops:
            if op.kind in GATHER:
                for acc in accs:
                    add(i, "op", acc, None if acc.dir == "S" else (acc.region, acc.win_off))
            elif op.kind in STORES:
                add(i, "op", accs[0], ("SD", accs[1].win_off + op.sub))
            else:
                acc = accs[0]
                add(i, "op", acc, None if acc.dir == "S" else (acc.region, acc.win_off))
        for q in probes:
            add(i, "probe", q, None)
    return ev


def analyze_ring(blob: bytes, blocks, setup) -> list[str]:
    inp, outp = parse_header(blob)
    evs = ring_events(blocks, setup)
    out = [f"in={inp:#x} out={outp:#x}"]
    for i, blk in enumerate(blocks):
        base = HDR_BYTES + RING_REC * i
        fills = [e for e in evs if e.rec == i and e.role == "fill"]
        last = [e for e in evs if e.rec == i][-1].seq + 1
        out.append(f"b{i} {blk.label}")

        for k in range(RING_PROBES + 1):
            slot = blob[base + 32 * k:base + 32 * (k + 1)]
            descs = [describe(halfword_source(h), evs, last) for h in struct.unpack("<16H", slot)]
            text = compress(descs)
            for kk, f in enumerate(fills):
                r, o = f.data
                text = text.replace(f"fillr{i}:{f.acc.kind}.{f.acc.dir}:{r}+{o}", f"F{kk}")
            out.append(f"  {'op ' if k == 0 else f'p{k} '}: {text}")
    return out


# -- analysis --------------------------------------------------------------------

def parse_header(blob: bytes) -> tuple[int, int]:
    return struct.unpack_from("<2I", blob, 0)


def halfword_source(h: int) -> str | tuple[str, int]:
    """'marker', or (region, byte offset inside its window)."""
    if h == MARKER * 0x101:
        return "marker"
    if h == 0:
        return "zero"
    tag, j = h >> 12, h & 0xFFF
    for r, t in TAG.items():
        if t == tag and j < WIN_BYTES // 2:
            return (r, 2 * j)
    return f"?{h:04x}"


@dataclass
class Event:
    """An access the kernel made, as the analysis sees it: what data crossed which port."""
    seq: int
    rec: int
    role: str                   # setup / prime / between / probe / data
    acc: Access
    data: tuple[str, int] | None    # (region, window offset) of the bytes moved; None: none


def events(records: list[Record], setup: tuple) -> list[Event]:
    ev: list[Event] = []

    def add(rec, role, acc, data):
        ev.append(Event(len(ev), rec, role, acc, data))

    setup_events(setup, add)
    for i, (rec, bl) in enumerate(zip(records, store_blocks(records))):
        for k, p in enumerate(rec.primes):
            if p.is_store:
                add(i, "prime", p, ("SD", 32 * bl[k] + p.sub))
            else:
                add(i, "prime", p, (p.region, p.win_off))
        for b in rec.between:
            add(i, "between", b, None if b.is_store else (b.region, b.win_off))
        for q in rec.probes:
            add(i, "probe", q, None)
    return ev


def describe(src, evs: list[Event], before: int) -> str:
    """Name the most recent access (sequence number below `before`) that moved this data."""
    if not isinstance(src, tuple):
        return src
    region, off = src
    for e in reversed(evs[:before]):
        if e.data is None:
            continue
        r, o = e.data
        if r == region and o <= off < o + e.acc.nbytes:
            age = "" if e.rec < 0 else f"r{e.rec}"
            return f"{e.role}{age}:{e.acc.kind}.{e.acc.dir}:{region}+{off}"
    return f"{region}+{off:#x}(never moved)"


def analyze(blob: bytes, records: list[Record], setup: tuple) -> list[str]:
    inp, outp = parse_header(blob)
    evs = events(records, setup)
    probe_seq = [e.seq for e in evs if e.role == "probe"]
    out = [f"in={inp:#x} out={outp:#x}"]
    k = 0
    for i, rec in enumerate(records):
        base = HDR_BYTES + REC_BYTES * i
        out.append(f"r{i} {rec.label}")
        for j, q in enumerate(rec.probes):
            slot = blob[base + SLOT * (j + 1):base + SLOT * (j + 2)]
            hws = struct.unpack("<16H", slot)
            descs = [describe(halfword_source(h), evs, probe_seq[k]) for h in hws]
            out.append(f"  {q.name}@{q.address(inp):05x}: " + compress(descs))
            k += 1
    return out


def compress(descs: list[str]) -> str:
    """Runs of halfwords from one access at consecutive offsets print as one item."""
    items, i = [], 0
    while i < len(descs):
        j = i + 1
        d = descs[i]
        if "+" in d and d.rsplit("+", 1)[1].isdigit():
            stem, o = d.rsplit("+", 1)
            o = int(o)
            while j < len(descs) and descs[j] == f"{stem}+{o + 2 * (j - i)}":
                j += 1
            items.append(f"[{2 * i}:{2 * j}]={stem}+{o}" if j - i > 1 else f"[{2 * i}]={d}")
        else:
            while j < len(descs) and descs[j] == d:
                j += 1
            items.append(f"[{2 * i}:{2 * j}]={d}")
        i = j
    return " ".join(items)


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
    blob = (Path(args.suite_dir) / "runs" / args.tag / args.case / "core.out.bin").read_bytes()
    if args.case in ring_cases():
        print("\n".join(analyze_ring(blob, ring_cases()[args.case], ring_case_tile(args.case)[1])))
    elif args.case in sweep_cases():
        tile, phases, setup = sweep_cases()[args.case]
        print("\n".join(analyze_sweep(blob, phases, setup)))
    else:
        tile, recs, setup = build_cases()[args.case]
        print("\n".join(analyze(blob, recs, setup)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
