#!/usr/bin/env python3
"""Load/store address-alignment probes for the topology harness (one core, hand-written asm).

What it measures
----------------
For each load/store instruction family and addressing form, the kernel issues the access at
a pointer swept byte by byte across a window and records what the hardware did with the
address:

  store  the destination region is pre-filled with a marker byte (0xEE); after the access
         the bytes that changed show where the data landed.
  load   the source window holds a distinct byte per position (byte = window index mod 256);
         the loaded register is stored back through an aligned store, so its bytes show
         which source bytes were read. The destination register is pre-loaded with the
         marker so a load that does not write it is visible.

Each iteration also records the pointer register after the access (the delta against the
value before it), so post-modify forms show whether the updated pointer keeps the
unaligned low bits.

A "case" is one family x addressing form x sweep (see `build_cases`). Cases are packed into
kernels (packages) by an output-size budget. The analysis side (`analyze_*`) is a pure
function of the output bytes; it needs no knowledge of the rule being tested.

Buffers
-------
  in  (read-only):  [0,64)    data bytes 1..64 (store source: wl0 = 1..32, wh0 = 33..64)
                    [64,128)  marker 0xEE (load destination pre-load)
                    [128,..)  load window, byte = index mod 256 (WIN_BYTES long)
  out: [0,64)        header, u32 words: in pointer, out pointer, then zero
       [64,..)       the cases' regions, each: data (n * stride bytes), pointer-delta table
                     (n u32, padded to 64 bytes); the whole area is first filled with the marker
Both buffers sit at 1 KiB-aligned addresses on silicon (0x74000 / 0x70400 observed); the header
records the real pointers, and `parse_output` rejects a 64-byte-misaligned base.

Pointer registers: p0 = in (never changed), p1 = out, p2 = access pointer (rebuilt every
iteration from the scalar r5), p3 = delta table cursor, p4 = load record cursor.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from harness.topology import Buffer, Core, TileRef, Topology, to_dict      # noqa: E402

MARKER = 0xEE
DATA_OFF, MARK_OFF, WIN_OFF = 0, 64, 128
WIN_BYTES = 1024
CENTER = 384                      # window offset where the swept pointer starts (loads)
IN_BYTES = WIN_OFF + WIN_BYTES
HDR_BYTES = 64
SLOT = 128                        # store case: bytes per iteration
STORE_BASE = 32                   # store case: first pointer offset inside a slot
OUT_BUDGET = 14 * 1024            # bytes of out buffer per kernel (data memory is 64 KiB)
SECONDARY_STEP = 5                # sweep stride of the secondary variants
NOPS = 6


def in_bytes() -> bytes:
    data = bytes(range(1, 65))
    marker = bytes([MARKER]) * 64
    window = bytes(i & 0xFF for i in range(WIN_BYTES))
    return data + marker + window


# -- the case catalogue -------------------------------------------------------

@dataclass(frozen=True)
class Family:
    """One instruction family. `mn` is the mnemonic (2D/3D forms insert the dimension after its
    first component), `ops` the operands before the address, `width` the bytes the access
    moves (0 = not predicted, measured)."""
    key: str
    kind: str                  # "st" or "ld"
    mn: str                    # e.g. "vst.srs.s16.s32"
    ops: str                   # e.g. "bml0, s1"
    width: int
    pre: tuple[str, ...] = ()  # per-iteration lines before the op (load: marker into dest; store: source)
    dump: tuple[str, ...] = () # per-iteration lines after the op (load: dest -> record, via p4)
    imm: int = 0               # post-modify immediate to use (and the expected delta)
    rec: int = 64              # load: bytes per record
    idx_imm: bool = True       # supports [p, #0]
    pstm_imm: bool = True
    pstm_m: bool = True
    idx_dj: bool = True
    dims: tuple[str, ...] = ()  # "2d" / "3d" forms that exist
    plain: bool = False        # only the bare [p] form exists
    sp_imm: int = 0            # nonzero: the stack-relative spill form [sp, #imm] exists
    sparse: bool = False       # reset the sparse stream at the swept pointer, then peek it
    lut: bool = False          # 4-way gather: the addresses come from a vector register
    conv: bool = False         # result bytes are a conversion of the source (no raw identity)

    def op(self, addr: str, dim: str = "") -> str:
        mn = self.mn
        if dim:
            head, _, rest = mn.partition(".")
            mn = f"{head}.{dim}" + (f".{rest}" if rest else "")
        if not addr:
            return f"{mn} {self.ops}"
        return f"{mn} {self.ops}, {addr}" if self.ops else f"{mn} {addr}"


# load destinations: how to pre-load the marker and how to dump the result to a record
LD_W = dict(pre=("vlda wl2, [p0, #64]",), dump=("vst wl2, [p4, #0]",), rec=32)
LD_X = dict(pre=("vlda wl2, [p0, #64]", "vlda wh2, [p0, #96]"),
            dump=("vst wl2, [p4, #0]", "vst wh2, [p4, #32]"))
LD_AM = dict(pre=("vlda amll0, [p0, #64]",), dump=("vst amll0, [p4, #0]",), rec=32)
LD_Q = dict(pre=("lda q0, [p0, #64]",), dump=("st q0, [p4, #0]",), rec=16)
LD_R = dict(pre=("mov r3, r9",), dump=("st r3, [p4, #0]",), rec=4)
LD_BM = dict(pre=("vmov bml0, x1",),
             dump=("vmov x2, bml0", "nop", "nop", "nop", "nop", "nop", "nop", "vst wl2, [p4, #0]", "vst wh2, [p4, #32]"))
LD_CM = dict(pre=("vmov bml0, x1", "vmov bmh0, x1"),
             dump=("vmov x2, bml0", "vmov x3, bmh0", "nop", "nop", "nop", "nop", "nop", "nop", "vst wl2, [p4, #0]", "vst wh2, [p4, #32]",
                   "vst wl3, [p4, #64]", "vst wh3, [p4, #96]"), rec=128)
# store sources for the fused stores
ST_BM = dict(pre=("vmov bml0, x0",))
ST_CM = dict(pre=("vmov bml0, x0", "vmov bmh0, x0"))
D23 = ("2d", "3d")


def families() -> list[Family]:
    F = Family
    out = [
        # scalar stores
        F("st_w", "st", "st", "r8", 4, imm=4, sp_imm=-64, dims=D23),
        F("st_s16", "st", "st.s16", "r8", 2, imm=2, dims=D23, pstm_m=True),
        F("st_s8", "st", "st.s8", "r8", 1, imm=1, dims=D23),
        # scalar loads
        F("lda_w", "ld", "lda", "r3", 4, imm=4, sp_imm=-64, dims=D23, **LD_R),
        F("lda_s16", "ld", "lda.s16", "r3", 2, imm=2, conv=True, dims=D23, **LD_R),
        F("lda_u16", "ld", "lda.u16", "r3", 2, imm=2, conv=True, dims=D23, **LD_R),
        F("lda_s8", "ld", "lda.s8", "r3", 1, imm=1, conv=True, dims=D23, **LD_R),
        F("lda_u8", "ld", "lda.u8", "r3", 1, imm=1, conv=True, dims=D23, **LD_R),
        # 128-bit
        F("st_q", "st", "st", "q1", 16, imm=16, sp_imm=-64, dims=D23),
        F("lda_q", "ld", "lda", "q0", 16, imm=16, sp_imm=-64, dims=D23, **LD_Q),
        F("vst_128", "st", "vst.128", "wl0", 16, imm=32, sp_imm=-64, dims=D23),
        F("vlda_128", "ld", "vlda.128", "wl2", 16, imm=32, idx_imm=False, pstm_imm=False,
          pstm_m=False, idx_dj=False, plain=True, **LD_W),
        F("vldb_128", "ld", "vldb.128", "wl2", 16, imm=32, idx_imm=False, pstm_imm=False,
          dims=D23, **LD_W),
        # 256-bit
        F("vst_w", "st", "vst", "wl0", 32, imm=32, sp_imm=-64, dims=D23),
        F("vst_am", "st", "vst", "amll0", 32, imm=32, sp_imm=-64, dims=D23),
        F("vlda_w", "ld", "vlda", "wl2", 32, imm=32, sp_imm=-64, dims=D23, **LD_W),
        F("vldb_w", "ld", "vldb", "wl2", 32, imm=32, dims=D23, **LD_W),
        F("vlda_am", "ld", "vlda", "amll0", 32, imm=32, sp_imm=-64, dims=D23, **LD_AM),
    ]
    # fused loads: vlda.ups.<acc>.<src> (all read 32 bytes), vlda.conv, vldb.unpack
    for acc, src, dst, ld in (("s32", "s16", "bml0", LD_BM), ("s32", "d16", "bml0", LD_BM),
                              ("s64", "s32", "bml0", LD_BM), ("s64", "d32", "bml0", LD_BM),
                              ("s32", "s8", "cm0", LD_CM), ("s32", "d8", "cm0", LD_CM),
                              ("s64", "s16", "cm0", LD_CM), ("s64", "d16", "cm0", LD_CM)):
        out.append(F(f"ups_{acc}_{src}", "ld", f"vlda.ups.{acc}.{src}", f"{dst}, s0", 32, imm=32,
                     conv=True, dims=D23, **ld))
    # 4-way LUT gathers: the source vector (x4) broadcasts the swept address to all four lanes
    for w in (16, 32, 64):
        for half in ("lo", "hi"):
            out.append(F(f"lut_4x{w}_{half}", "ld", f"vldb.4x{w}.{half}", "wl2, wl4", 8, lut=True,
                         idx_imm=False, pstm_imm=False, pstm_m=False, idx_dj=False, conv=True,
                         pre=("vlda wl2, [p0, #64]", "vbcst.32 x4, r5"), dump=("vst wl2, [p4, #0]",),
                         rec=32))
    # the same gathers with a different address in every lane (lane value = swept address + 37,
    # +70, +101 for lanes 1..3 and 5..7, so a lane depending on its neighbour shows)
    mix = ("movxm r14, #37", "movxm r15, #70", "movxm r13, #101", "nop", "nop", "nop", "nop",
           "add r14, r5, r14", "add r15, r5, r15", "add r13, r5, r13", "nop", "nop", "nop", "nop",
           "vbcst.32 x4, r5", *["nop"] * 6)
    for lane, reg in ((1, "r14"), (2, "r15"), (3, "r13"), (5, "r13"), (6, "r14"), (7, "r15")):
        mix += (f"mov r29, #{lane}", "nop", "nop", "nop", f"vinsert.32 x4, x4, r29, {reg}", *["nop"] * 6)
    for w in (16, 32, 64):
        for half in ("lo", "hi"):
            out.append(F(f"lutmix_4x{w}_{half}", "ld", f"vldb.4x{w}.{half}", "wl2, wl4", 8, lut=True,
                         idx_imm=False, pstm_imm=False, pstm_m=False, idx_dj=False, conv=True,
                         pre=("vlda wl2, [p0, #64]", *mix), dump=("vst wl2, [p4, #0]",), rec=32))
    # sparse stream: reset fetches the line at the swept pointer, a peek through a copy of the
    # pointer shows which bytes the unit read (qwl0 = {wl0, ql0}); the mask half is not dumped
    for w in (4, 8, 16):
        out.append(F(f"sparse_{w}", "ld", f"vldb.sparse.reset.{w}", "", 32, sparse=True, plain=True,
                     idx_imm=False, pstm_imm=False, pstm_m=False, idx_dj=False, conv=True,
                     pre=("vlda wl0, [p0, #64]", "vlda wh0, [p0, #96]", "mov p5, p2", *["nop"] * 6),
                     dump=(*["nop"] * 8, f"vldb.sparse.peek.{w} qwl0, [p5]", *["nop"] * 12,
                           "vst wl0, [p4, #0]", "vst wh0, [p4, #32]")))
    out.append(F("conv_ld", "ld", "vlda.conv.fp32.bf16", "bml0", 32, imm=32, conv=True,
                 dims=D23, **LD_BM))
    for dst, src, nbytes in (("s16", "s8", 32), ("d16", "d8", 32), ("s8", "s4", 16), ("d8", "d4", 16)):
        out.append(F(f"unpack_{dst}_{src}", "ld", f"vldb.unpack.{dst}.{src}", "x2", nbytes, imm=32,
                     idx_imm=False, pstm_imm=False, conv=True, dims=D23, **LD_X))
    # fused stores: vst.srs.<out>.<acc> (all write 32 bytes), vst.pack, vst.conv
    for o, acc, srcreg, st in (("s16", "s32", "bml0", ST_BM), ("d16", "s32", "bml0", ST_BM),
                               ("s32", "s64", "bml0", ST_BM), ("d32", "s64", "bml0", ST_BM),
                               ("s8", "s32", "cm0", ST_CM), ("d8", "s32", "cm0", ST_CM),
                               ("s16", "s64", "cm0", ST_CM), ("d16", "s64", "cm0", ST_CM)):
        out.append(F(f"srs_{o}_{acc}", "st", f"vst.srs.{o}.{acc}", f"{srcreg}, s1", 32, imm=32,
                     conv=True, dims=D23, **st))
    for kind, o, i in (("s", "s8", "s16"), ("d", "d8", "d16"), ("s", "s4", "s8"), ("d", "d4", "d8")):
        out.append(F(f"pack_{o}_{i}", "st", f"vst.pack.{o}.{i}", "x0", 0, imm=32, conv=True, dims=D23))
    out.append(F("conv_st", "st", "vst.conv.bf16.fp32", "bml0", 32, imm=32, conv=True, dims=D23, **ST_BM))
    return out


@dataclass(frozen=True)
class Case:
    name: str
    fam: Family
    form: str                  # "imm0" | "pstm_imm" | "pstm_m" | "dj_sweep" | "dj_mix"
    n: int
    ptr_step: int              # byte step of the pointer per iteration
    ptr_start: int             # pointer offset at iteration 0 (store: inside slot; load: window offset)
    mod_start: int             # modifier register value at iteration 0
    mod_step: int
    addr: str                  # the address operand text
    dim: str = ""              # "" | "2d" | "3d": which instruction variant
    sp_imm: int = 0            # nonzero: the access is [sp, #sp_imm]; sp (not p2) is swept

    @property
    def stride(self) -> int:
        return SLOT if self.fam.kind == "st" else self.fam.rec

    @property
    def data_bytes(self) -> int:
        return self.n * self.stride

    @property
    def table_bytes(self) -> int:
        return (self.n * 4 + 63) // 64 * 64

    @property
    def total_bytes(self) -> int:
        return self.data_bytes + self.table_bytes


def sweep_count(width: int) -> int:
    """Byte offsets worth sweeping for an access of `width` bytes (0..2*width, at least 8)."""
    return max(8, 2 * width) if width < 32 else 64


def build_cases(fams: list[Family] | None = None) -> list[Case]:
    cases: list[Case] = []
    # the sparse probes clobber wl0/wh0 (the store-data registers): they run last
    for f in sorted(fams if fams is not None else families(), key=lambda f: f.sparse):
        n_full = sweep_count(f.width)
        base = STORE_BASE if f.kind == "st" else CENTER
        sec_n = max(4, min(13, n_full // SECONDARY_STEP + 1))

        def add(form, addr, n, step, start, mod0, mods, dim="", sp_imm=0):
            cases.append(Case(f"{f.key}.{form}", f, form, n, step, base + start, mod0, mods, addr,
                              dim, sp_imm))

        if f.plain:
            add("plain", "[p2]", n_full, 1, 0, 0, 0)
        if f.sp_imm:   # stack-relative: sp is swept so that sp + imm walks the usual window
            add("spill", f"[sp, #{f.sp_imm}]", n_full, 1, -f.sp_imm, 0, 0, sp_imm=f.sp_imm)
        if f.lut:
            # the gather addresses are swept over 64 bytes at two window positions
            add("lut", "", 64, 1, 0, 0, 0)
            add("lut_far", "", 64, 1, 192, 0, 0)
        if f.idx_imm:
            add("imm0", "[p2, #0]", n_full, 1, 0, 0, 0)
        if f.pstm_imm:
            add("pstm_imm", f"[p2], #{f.imm}", sec_n, SECONDARY_STEP, 0, 0, 0)
        if f.pstm_m:
            if f.idx_imm or f.plain:
                add("pstm_m", "[p2], m0", sec_n, SECONDARY_STEP, 0, f.imm, 0)
            else:   # no [p, #0] form: the post-modify form carries the full byte sweep
                add("pstm_m", "[p2], m0", n_full, 1, 0, f.imm, 0)
        if f.idx_dj:
            # aligned base, index swept
            add("dj_sweep", "[p2, dj0]", sec_n, 0, 32 if f.kind == "st" else 0, 0, SECONDARY_STEP)
            # base swept, constant odd index
            add("dj_mix", "[p2, dj0]", sec_n, SECONDARY_STEP, 0, 5, 0)
        # multi-dimensional post-modify: counts and sizes zero, so the pointer steps by m0
        for dim in f.dims:
            add(f"pstm_{dim}", "[p2], d0", sec_n, SECONDARY_STEP, 0, f.imm, 0, dim)
    return cases


# -- assembly ---------------------------------------------------------------

def _nops(n: int) -> list[str]:
    return ["  nop"] * n


def _frame(body: list[str]) -> str:
    pro = [".text", ".globl test_kernel", "test_kernel:"]
    epi = ["  nop"] * 7 + ["  ret lr"] + ["  nop"] * 5
    return "\n".join(pro + body + epi) + "\n"


def layout(cases: list[Case]) -> list[int]:
    """Offset (from the out buffer start) of each case's region; first is after the header."""
    offs, cur = [], HDR_BYTES
    for c in cases:
        offs.append(cur)
        cur += c.total_bytes
    return offs


def out_bytes(cases: list[Case]) -> int:
    offs = layout(cases)
    return (offs[-1] + cases[-1].total_bytes) if cases else HDR_BYTES


def kernel_asm(cases: list[Case], tag: str = "k") -> str:
    body: list[str] = []
    # header: the real pointers
    body += ["  mov r2, p0", "  mov r3, p1", "  mov r28, sp", *_nops(3), "  st r2, [p1, #0]",
             "  st r3, [p1, #4]"]
    # constants and sources: data in x0, marker in x1, scalar data r8, scalar marker r9, zero r10
    body += ["  vlda wl0, [p0, #0]", "  vlda wh0, [p0, #32]",
             "  vlda wl1, [p0, #64]", "  vlda wh1, [p0, #96]",
             "  vlda amll0, [p0, #0]", "  lda q1, [p0, #0]",
             "  lda r8, [p0, #0]", "  lda r9, [p0, #64]", "  mov r10, #0", *_nops(NOPS + 4),
             "  mov s0, r10", "  mov s1, r10", *_nops(2),
             "  vmov bml0, x0", *_nops(NOPS)]
    # fill the case area with the marker (aligned 256-bit stores)
    area = out_bytes(cases) - HDR_BYTES
    body += ["  mov p2, p1", "  padda [p2], #64", f"  movxm r0, #{area // 32}", *_nops(3),
             ".p2align 4", f".L{tag}_fill:",
             "  vst wl1, [p2], #32", "  add r0, r0, #-1", *_nops(2),
             f"  jnz r0, #.L{tag}_fill", *_nops(5)]
    # restore the data registers clobbered by nothing (wl1 is marker; keep)
    for idx, (c, off) in enumerate(zip(cases, layout(cases))):
        f = c.fam
        label = f".L{tag}_c{idx}"
        data_off = off
        tab_off = off + c.data_bytes
        # r5 = absolute pointer for iteration 0
        if f.kind == "st":
            rel_ptr = data_off + c.ptr_start
            body += ["  mov r5, p1", f"  movxm r6, #{rel_ptr}", *_nops(2), "  add r5, r5, r6"]
        else:
            rel_ptr = WIN_OFF + c.ptr_start
            body += ["  mov r5, p0", f"  movxm r6, #{rel_ptr}", *_nops(2), "  add r5, r5, r6"]
        body += [f"  movxm r7, #{c.mod_start}", f"  movxm r6, #{tab_off}", "  mov r4, p1", *_nops(2),
                 "  add r4, r4, r6", f"  movxm r6, #{data_off}", *_nops(2), "  mov p3, r4",
                 "  mov r4, p1", *_nops(2), "  add r4, r4, r6", *_nops(2), "  mov p4, r4",
                 f"  movxm r0, #{c.n}", *_nops(4)]
        body += [".p2align 4", f"{label}:"]
        ptr = "sp" if c.sp_imm else "p2"
        body += [f"  mov {ptr}, r5", "  mov m0, r7", "  mov dj0, r7"]
        if c.dim:   # counts and sizes of the dimensions in use are zero: the step is the modifier
            body += ["  mov dn0, r10", "  mov dc0, r10", "  mov dn4, r10", "  mov dc4, r10"]
        body += _nops(NOPS)
        for ln in f.pre:
            body += [f"  {ln}"]
        if f.pre:
            body += _nops(NOPS + 2)
        body += [f"  {f.op(c.addr, c.dim)}", *_nops(NOPS + 6)]
        body += [f"  mov r6, {ptr}", *_nops(3), "  sub r6, r6, r5", *_nops(2),
                 "  st r6, [p3, #0]", "  padda [p3], #4"]
        for ln in f.dump:           # (dump lines that need latency carry their own nops)
            body += [f"  {ln}"]
        if f.kind == "ld":
            body += [f"  padda [p4], #{f.rec}"]
        body += _nops(3)
        # movxm into a register: `add` immediates are narrow and llvm-mc truncates silently
        total_step = c.ptr_step + (SLOT if f.kind == "st" else 0)
        body += [f"  movxm r11, #{total_step}", f"  movxm r12, #{c.mod_step}", *_nops(2),
                 "  add r5, r5, r11", "  add r7, r7, r12"]
        body += ["  add r0, r0, #-1", *_nops(2), f"  jnz r0, #{label}", *_nops(5)]
        if c.sp_imm:
            body += ["  mov sp, r28", *_nops(3)]
    return _frame(body)


# -- topology -----------------------------------------------------------------

def topology(name: str, cases: list[Case], out_dir: Path) -> Topology:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"{name}.s"
    src.write_text(kernel_asm(cases, tag=name.replace("-", "_")))
    return Topology(name, (Core("k", TileRef(0, 2), src, args=(
        Buffer("in", "in", IN_BYTES, data=in_bytes()),
        Buffer("out", "out", out_bytes(cases)))),), device="auto")


def pack(cases: list[Case], budget: int = OUT_BUDGET) -> list[list[Case]]:
    """Greedy packing of cases into kernels by out-buffer size."""
    groups: list[list[Case]] = []
    cur: list[Case] = []
    size = HDR_BYTES
    for c in cases:
        if cur and size + c.total_bytes > budget:
            groups.append(cur)
            cur, size = [], HDR_BYTES
        cur.append(c)
        size += c.total_bytes
    if cur:
        groups.append(cur)
    return groups


# -- analysis ------------------------------------------------------------------

@dataclass
class CaseResult:
    case: Case
    data: bytes                        # n * stride bytes
    deltas: list[int]                  # pointer after - pointer before, per iteration (signed)

    def record(self, i: int) -> bytes:
        s = self.case.stride
        return self.data[i * s:(i + 1) * s]


def parse_output(cases: list[Case], blob: bytes) -> tuple[dict[str, int], list[CaseResult]]:
    if len(blob) < out_bytes(cases):
        raise ValueError(f"output is {len(blob)} bytes, expected {out_bytes(cases)}")
    in_ptr, out_ptr = struct.unpack_from("<II", blob, 0)
    if out_ptr % 64 or in_ptr % 64:
        raise ValueError(f"buffers not 64-byte aligned: in {in_ptr:#x} out {out_ptr:#x}")
    results = []
    for c, off in zip(cases, layout(cases)):
        data = blob[off:off + c.data_bytes]
        raw = struct.unpack_from(f"<{c.n}i", blob, off + c.data_bytes)
        results.append(CaseResult(c, data, list(raw)))
    return {"in": in_ptr, "out": out_ptr}, results


def store_landing(res: CaseResult) -> list[dict]:
    """Per iteration: which slot bytes changed (offsets relative to the slot start)."""
    rows = []
    for i in range(res.case.n):
        rec = res.record(i)
        changed = [j for j, b in enumerate(rec) if b != MARKER]
        ptr = res.case.ptr_start + i * res.case.ptr_step
        mod = res.case.mod_start + i * res.case.mod_step
        rows.append(dict(i=i, ptr=ptr, mod=mod, addr=ptr + (mod if "dj" in res.case.form else 0) + res.case.sp_imm,
                         first=changed[0] if changed else None,
                         last=changed[-1] if changed else None,
                         count=len(changed), data=bytes(rec[j] for j in changed),
                         delta=res.deltas[i]))
    return rows


def load_landing(res: CaseResult) -> list[dict]:
    """Per iteration: the record's leading bytes and, for raw loads, the source window offset
    of its first byte (window byte = index mod 256, so it is the low byte of the offset)."""
    rows = []
    for i in range(res.case.n):
        rec = res.record(i)
        ptr = res.case.ptr_start + i * res.case.ptr_step
        mod = res.case.mod_start + i * res.case.mod_step
        rows.append(dict(i=i, ptr=ptr, mod=mod, addr=ptr + (mod if "dj" in res.case.form else 0) + res.case.sp_imm,
                         rec=rec, first=rec[0], delta=res.deltas[i]))
    return rows


def floor_to(addr: int, width: int) -> int:
    """The natural-alignment rule under test: an access of `width` bytes at `addr` touches
    the `width`-aligned block holding `addr` (low log2(width) address bits dropped)."""
    return addr - addr % width


def verify_natural(res: CaseResult) -> list[str]:  # noqa: C901
    """Check a raw (non-converting) case against the natural-alignment rule; returns problems."""
    c, w = res.case, res.case.fam.width
    bad = []
    if c.fam.kind == "st":
        want = bytes(range(1, w + 1))
        for row in store_landing(res):
            at = floor_to(row["addr"], w)
            if (row["first"], row["last"], row["count"]) != (at, at + w - 1, w) or row["data"] != want:
                bad.append(f"i={row['i']} addr={row['addr']}: touched {row['first']}..{row['last']} "
                           f"n={row['count']} data={row['data'].hex()}, want {at}..{at + w - 1}")
    elif not c.fam.conv:
        for row in load_landing(res):
            at = floor_to(row["addr"], w)
            want = bytes((at + k) & 0xFF for k in range(w))
            if row["rec"][:w] != want:
                bad.append(f"i={row['i']} addr={row['addr']}: loaded {row['rec'][:w].hex()}, want {want.hex()}")
    for i, d in enumerate(res.deltas):
        if c.form.startswith("pstm"):
            want_d = c.fam.imm
        else:
            want_d = 0
        if d != want_d:
            bad.append(f"i={i}: pointer delta {d}, want {want_d} (unmasked)")
    return bad


def lut_lane_offsets(fam_key: str) -> list[int]:
    """Offsets added to the swept address in the four lanes a 4x gather uses (lo: lanes 0..3,
    hi: lanes 4..7); the plain probe broadcasts the address, the `lutmix` probe offsets it."""
    if "lutmix" not in fam_key:
        return [0, 0, 0, 0]
    return [0, 101, 37, 70] if fam_key.endswith("_hi") else [0, 37, 70, 101]


def lut_expected_lane(v: int, lane: int, element_bits: int) -> bytes:
    """Silicon model of one 4x gather lane (fits every probe record on Phoenix): the lane's
    32-bit address `v` is reduced to the element granule (mask 4, 8, 16 bytes for 16, 32, 64
    bit), the low five bits are halved, odd lanes read the other 16-byte bank (bit 4 set), and
    the 8-byte read stops at its 16-byte row: bytes past it read as zero."""
    mask = {16: 3, 32: 7, 64: 15}[element_bits]
    eff = (v & ~31) | (((v & 31) & ~mask) >> 1) | (0x10 if lane % 2 else 0)
    return bytes(((eff + k) & 0xFF) if (eff & 15) + k < 16 else 0 for k in range(8))


def verify_lut(res: CaseResult) -> list[str]:
    c = res.case
    bits = int(c.fam.key.split("_4x")[1][:2])
    offs = lut_lane_offsets(c.fam.key)
    bad = []
    for row in load_landing(res):
        for lane in range(4):
            want = lut_expected_lane(row["addr"] + offs[lane], lane, bits)
            got = row["rec"][8 * lane:8 * lane + 8]
            if got != want:
                bad.append(f"i={row['i']} lane {lane} addr {row['addr'] + offs[lane]}: {got.hex()}, want {want.hex()}")
    return bad


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length()


def observed_granule(res: CaseResult) -> dict:
    """Granule of the address rounding, measured without assuming a width.

    store: `width` = bytes changed per access (constant across the sweep, else None) and
           `granule` = largest power of two dividing every landing address, with `ok` set
           when every landing address is the swept address rounded down to `granule`.
    load:  records are grouped by equal content over consecutive swept addresses; `granule`
           is the (constant) group length of the interior groups, `ok` when each group starts
           at a multiple of it, groups are distinct, and the rule floor(addr, granule) holds.
           Needs a step-1 sweep (the `imm0` / `plain` forms).
    """
    c = res.case
    out = dict(case=c.name, width=None, granule=None, ok=False)
    if c.fam.kind == "st":
        rows = store_landing(res)
        counts = {r["count"] for r in rows}
        if len(counts) != 1 or 0 in counts:
            out["note"] = f"counts {sorted(counts)}"
            return out
        out["width"] = counts.pop()
        g = 1
        while g < 128 and all(r["first"] % (g * 2) == 0 for r in rows):
            g *= 2
        out["granule"] = g
        out["ok"] = all(r["first"] == floor_to(r["addr"], g) for r in rows)
        return out
    if c.ptr_step != 1 or c.mod_step != 0:
        out["note"] = "needs a step-1 sweep"
        return out
    rows = load_landing(res)
    groups: list[tuple[int, int, bytes]] = []       # (first addr, length, record)
    for r in rows:
        if groups and groups[-1][2] == r["rec"]:
            a, n, rec = groups[-1]
            groups[-1] = (a, n + 1, rec)
        else:
            groups.append((r["addr"], 1, r["rec"]))
    # interior groups are complete; with only two groups, both must agree
    interior = {n for _, n, _ in (groups[1:-1] if len(groups) > 2 else groups)}
    if len(interior) != 1:
        out["note"] = f"group lengths {[n for _, n, _ in groups]}"
        return out
    g = interior.pop()
    out["granule"] = g
    out["ok"] = (all(a % g == 0 for a, _, _ in groups[1:])
                 and len({rec for _, _, rec in groups}) == len(groups))
    return out


def layout_fingerprint() -> str:
    """Hash of everything that fixes how a captured output must be read: the layout constants,
    the buffer contents, and every case (name, sweep, address text, record size, op text)
    with its packing. A manifest records it; readers refuse a mismatch."""
    import hashlib
    h = hashlib.sha256()
    h.update(repr((MARKER, DATA_OFF, MARK_OFF, WIN_OFF, WIN_BYTES, CENTER, HDR_BYTES, SLOT,
                   STORE_BASE, OUT_BUDGET, SECONDARY_STEP, NOPS)).encode())
    h.update(in_bytes())
    for c in build_cases():
        f = c.fam
        h.update(repr((c.name, c.form, c.n, c.ptr_step, c.ptr_start, c.mod_start, c.mod_step,
                       c.addr, c.dim, c.sp_imm, f.kind, f.op(c.addr, c.dim), f.width, f.pre, f.dump,
                       f.imm, f.rec, f.conv)).encode())
    return h.hexdigest()[:16]


def _git_rev() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class ManifestMismatch(RuntimeError):
    pass


def load_manifest(path: Path) -> list[dict]:
    """Kernel list of a generation's manifest; raises ManifestMismatch unless it was written by
    a generator with the same `layout_fingerprint` (old captures would be silently misread)."""
    m = json.loads(Path(path).read_text())
    if not isinstance(m, dict) or "layout_fingerprint" not in m:
        raise ManifestMismatch(f"{path}: no layout_fingerprint (written by an older generator); "
                               "regenerate and recapture, or stamp it after checking by hand")
    if m["layout_fingerprint"] != layout_fingerprint():
        raise ManifestMismatch(f"{path}: layout {m['layout_fingerprint']} (generator "
                               f"{m.get('generator_rev')}), current generator is "
                               f"{layout_fingerprint()} ({_git_rev()})")
    return m["kernels"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write topology JSON + asm per kernel")
    g.add_argument("out_dir")
    g.add_argument("--only", nargs="*", help="family keys (default all)")
    g.add_argument("--budget", type=int, default=OUT_BUDGET)
    a = ap.parse_args(argv)
    fams = [f for f in families() if not a.only or f.key in a.only]
    groups = pack(build_cases(fams), a.budget)
    out = Path(a.out_dir)
    manifest = []
    for gi, grp in enumerate(groups):
        name = f"ldst_{gi:02d}"
        t = topology(name, grp, out)
        (out / f"{name}.json").write_text(json.dumps(to_dict(t), indent=2) + "\n")
        manifest.append(dict(name=name, cases=[c.name for c in grp], out_bytes=out_bytes(grp)))
    (out / "manifest.json").write_text(json.dumps(
        dict(layout_fingerprint=layout_fingerprint(), generator_rev=_git_rev(), kernels=manifest),
        indent=2) + "\n")
    print(f"{len(groups)} kernel(s), {sum(len(g) for g in groups)} case(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
