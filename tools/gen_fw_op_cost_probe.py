#!/usr/bin/env python3
"""Runtime-sequence op cost probe: how long each host-sequence op takes, as a core sees it.

A compute core polls words of its own data memory in order. The host runtime sequence writes
them with the op under test (`aiex.npu.write32`, `blockwrite`, `maskwrite32`, optionally with
filler writes to other tiles in between). When a polled word changes, the core stamps the tile
cycle counter (`MOV_CNTR`, `mov r25:r24, cntr`, low word kept) and polls the next. Only the
differences between successive stamps mean anything: the counter is per tile and the core starts
at configuration, before the sequence.

Instrument
----------
The poll loop is a fixed 14-bundle loop (a load, its 7-cycle latency, the branch and its five
delay slots; the stamp sits in the first delay slot, so it is taken on every pass and the one from
the passing iteration is kept). Its period is the granularity of every arrival time: an arrival is
seen by the first pass whose load issues after it. Intervals between arrivals the host spaces
further apart than a pass are therefore good to one period; a burst faster than the loop reads as
the loop's own floor (stamp, two stores, pointer update, next pass), which the `control` case
(words pre-set by the configuration, no runtime writes) measures. Every word is bounded by a
timeout (a missing arrival stores value 0) so a lost write cannot hang the core.

Results buffer, little-endian u32: `n` stamps, then the `n` values the core saw (the marker of
the word, or 0 on timeout). The `spin` cases instead hold four words: stamp at release, stamp
after a counted spin, the release value, the spin count; compared with the host's wall time they
give the counter's rate in Hz.

  python3 tools/gen_fw_op_cost_probe.py generate <dir>   # one directory per case
  python3 tools/gen_fw_op_cost_probe.py build <dir> [--only CASE ..]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

GENERATOR = "tools/gen_fw_op_cost_probe.py"

POLL_WORDS_MAX = 512
BANK_BYTES = 0x2000
POLL_ADDR = 0x8000
TIMEOUT_FIRST = 1 << 26      # passes before the first word is given up on (~0.9 s)
TIMEOUT_NEXT = 1 << 20       # passes for every later word (~14 ms)
SPIN_CYCLES_PER_ITER = 7     # add + jnz + five delay slots
MASK_PARTIAL = 0x0000FFFF
FILLER_VALUE = 0x5A5A

FILLER_BUFFER = {"memtile": "mt_scratch", "column1": "c1_scratch"}


def marker(k: int) -> int:
    """Value written to polled word k: nonzero and distinct per word."""
    return 0x1000 + k + 1


@dataclass(frozen=True)
class Filler:
    target: str     # "memtile" (same column) or "column1" (another column's compute tile)
    count: int      # filler write32s before every polled write


@dataclass(frozen=True)
class Case:
    name: str
    kind: str                   # write32 | maskwrite32 | blockwrite | control | spin
    words: int                  # polled-region words written
    block: int = 1              # blockwrite: words per op
    poll_start: int = 0
    poll_stride: int = 1
    filler: Filler | None = None
    prefill: Filler | None = None   # untimed-by-design ops before the first polled write
    preset: bool = False        # words hold their markers from configuration
    spin_iters: int = 0


def all_cases() -> list[Case]:
    cases = [Case("control", "control", 64, preset=True)]
    for n in (8, 16, 32, 64):
        cases.append(Case(f"w32_n{n}", "write32", n))
    for n in (8, 16, 32, 64):
        cases.append(Case(f"mw32_n{n}", "maskwrite32", n))
    for l in (1, 4, 16, 64):
        cases.append(Case(f"bw_l{l}", "blockwrite", 128, block=l))
    cases.append(Case("bw_l16_last", "blockwrite", 128, block=16, poll_start=15, poll_stride=16))
    cases.append(Case("bw_l4_last", "blockwrite", 128, block=4, poll_start=3, poll_stride=4))
    cases.append(Case("bw_l64_last", "blockwrite", 512, block=64, poll_start=63, poll_stride=64))
    for k in (8, 64):
        cases.append(Case(f"pre_f{k}_w32_n64", "write32", 64, prefill=Filler("memtile", k)))
    # Long streams: ones shorter than 16 KiB of instruction stream, and ones whose polled range
    # contains a multiple of 16 KiB into the stream (the once-per-run stall probes).
    cases.append(Case("w32_n512", "write32", 512))
    cases.append(Case("mw32_n512", "maskwrite32", 512))   # ~14 KiB, ~110 us of ops
    for k in (400, 500, 1200, 1900):
        cases.append(Case(f"pre_f{k}_w32_n512", "write32", 512, prefill=Filler("memtile", k)))
    cases.append(Case("pre_f400_mw32_n512", "maskwrite32", 512, prefill=Filler("memtile", 400)))
    for target, tag in (("memtile", "mt"), ("column1", "c1")):
        for k in (1, 4):
            cases.append(Case(f"il_{tag}_k{k}", "write32", 16, filler=Filler(target, k)))
    cases.append(Case("spin_a", "spin", 1, spin_iters=20_000_000))
    cases.append(Case("spin_b", "spin", 1, spin_iters=60_000_000))
    return cases


def polled_indices(c: Case) -> list[int]:
    if c.kind == "spin":
        return [0]
    return list(range(c.poll_start, c.words, c.poll_stride))


def results_words(c: Case) -> int:
    return 4 if c.spin_iters else 2 * len(polled_indices(c))


# -- host op sequence -----------------------------------------------------------------------

def sequence_ops(c: Case) -> list[tuple]:
    """The timed host ops, in order (offsets are words into the named buffer: the lowering scales
    the `address` of a buffer-relative op by the element size): ("write32", target, word_off, value),
    ("maskwrite32", target, word_off, value, mask), ("blockwrite", target, word_off, [words])."""
    if c.kind == "control":
        return []
    if c.kind == "spin":
        return [("write32", "poll", 0, marker(0))]
    ops: list[tuple] = []
    if c.prefill:
        ops += [("write32", c.prefill.target, 0, FILLER_VALUE)] * c.prefill.count
    if c.kind == "blockwrite":
        for first in range(0, c.words, c.block):
            ops.append(("blockwrite", "poll", first,
                        [marker(k) for k in range(first, first + c.block)]))
        return ops
    for k in range(c.words):
        if c.filler:
            ops += [("write32", c.filler.target, 0, FILLER_VALUE)] * c.filler.count
        if c.kind == "maskwrite32":
            ops.append(("maskwrite32", "poll", k, marker(k), MASK_PARTIAL))
        else:
            ops.append(("write32", "poll", k, marker(k)))
    return ops


# -- kernel ---------------------------------------------------------------------------------

def _nops(n: int) -> list[str]:
    return ["  nop"] * n


def kernel_text(c: Case) -> str:
    """Assembly of `probe(poll, results)`: p0 = polled words, p1 = results.

    Registers: r1 word, r8 pass budget, r9 budget spent, r2 exit, r10 words left, r11/r12
    budgets, r24 stamp. All caller-saved (r16..r23, p6, p7 are not touched)."""
    idx = polled_indices(c)
    val_words = 2 if c.spin_iters else len(idx)
    lines = [f"// Auto-generated ({GENERATOR}): {c.name}", ".text", ".globl probe", "probe:",
             f"  movxm r11, #{TIMEOUT_FIRST}", f"  movxm r12, #{TIMEOUT_NEXT}",
             f"  movxm r10, #{len(idx)}",
             f"  movxm r13, #{4 * c.poll_stride}", "  mov m0, r13",
             f"  movxm r13, #{4 * c.poll_start}", "  mov m1, r13",
             f"  movxm r13, #{4 * val_words}", "  mov m2, r13",
             "  mov p2, p1", "  padda [p2], m2", "  padda [p0], m1"]
    lines += _nops(2)
    lines += [".p2align 4", ".Lword:", "  mov r8, r11", ".Lpoll:",
              "  lda r1, [p0, #0]", "  add r8, r8, #-1", "  eqz r9, r8"]
    lines += _nops(4)
    lines += ["  or r2, r1, r9", "  jz r2, #.Lpoll", "  mov r25:r24, cntr"]
    lines += _nops(4)
    lines += ["  st r24, [p1], #4", "  st r1, [p2], #4", "  padda [p0], m0", "  mov r11, r12",
              "  add r10, r10, #-1", "  nop", "  jnz r10, #.Lword"]
    lines += _nops(5)
    if c.spin_iters:
        lines += [f"  movxm r5, #{c.spin_iters}", "  mov r6, r5", "  nop",
                  ".p2align 4", ".Lspin:", "  add r5, r5, #-1", "  jnz r5, #.Lspin"]
        lines += _nops(5)
        lines += ["  mov r25:r24, cntr", "  nop", "  st r24, [p1], #4", "  st r6, [p2], #4"]
        lines += _nops(2)
    lines += ["  ret lr"] + _nops(5)
    return "\n".join(lines) + "\n"


# -- MLIR -----------------------------------------------------------------------------------

def mlir_text(c: Case) -> str:
    n_out = max(results_words(c), 1)
    ops = sequence_ops(c)
    targets = {o[1] for o in ops}
    out = ["module {", "  aie.device(npu1) {",
           "    %t00 = aie.tile(0, 0)", "    %t02 = aie.tile(0, 2)"]
    if "memtile" in targets:
        out += ["    %t01 = aie.tile(0, 1)",
                '    %mt_buf = aie.buffer(%t01) {sym_name = "mt_scratch"} : memref<16xi32>']
    if "column1" in targets:
        out += ["    %t12 = aie.tile(1, 2)",
                '    %c1_buf = aie.buffer(%t12) {sym_name = "c1_scratch"} : memref<16xi32>',
                "    aie.core(%t12) {", "      aie.end", "    }"]
    init = (f"dense<[{', '.join(str(marker(k)) for k in range(c.words))}]> : tensor<{c.words}xi32>"
            if c.preset else f"dense<0> : tensor<{c.words}xi32>")
    out += [f'    %poll = aie.buffer(%t02) {{sym_name = "poll", address = {POLL_ADDR} : i32, '
            f"initial_value = {init}}} : memref<{c.words}xi32>",
            f"    aie.objectfifo @of_out(%t02, {{%t00}}, 1 : i32) : !aie.objectfifo<memref<{n_out}xi32>>",
            f'    func.func private @probe(memref<{c.words}xi32>, memref<{n_out}xi32>) '
            f'attributes {{link_with = "kernel.o"}}',
            "    aie.core(%t02) {",
            f"      %o = aie.objectfifo.acquire @of_out(Produce, 1) : memref<{n_out}xi32>",
            f"      func.call @probe(%poll, %o) : (memref<{c.words}xi32>, memref<{n_out}xi32>) -> ()",
            "      aie.objectfifo.release @of_out(Produce, 1)", "      aie.end", "    }"]
    blocks = [o for o in ops if o[0] == "blockwrite"]
    for i, o in enumerate(blocks):
        out.append(f'    memref.global "private" constant @blk{i} : memref<{len(o[3])}xi32> = '
                   f"dense<[{', '.join(map(str, o[3]))}]>")
    out += [f"    aie.runtime_sequence(%in : memref<1xi32>, %buf : memref<1xi32>, "
            f"%out : memref<{n_out}xi32>) {{",
            "      %c0 = arith.constant 0 : i64", "      %c1 = arith.constant 1 : i64",
            f"      %cn = arith.constant {n_out} : i64",
            "      aiex.npu.dma_memcpy_nd(%out[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %cn]"
            "[%c0, %c0, %c0, %c1]) {metadata = @of_out, id = 0 : i64, issue_token = true} "
            f": memref<{n_out}xi32>"]
    nblk = 0
    for i, o in enumerate(ops):
        kind, target, off = o[0], o[1], o[2]
        attr = f"{{buffer = @{FILLER_BUFFER.get(target, target)}}}"
        if kind == "blockwrite":
            out += [f"      %g{i} = memref.get_global @blk{nblk} : memref<{len(o[3])}xi32>",
                    f"      aiex.npu.blockwrite(%g{i}) {{buffer = @{target}, address = {off} : ui32}} "
                    f": memref<{len(o[3])}xi32>"]
            nblk += 1
            continue
        out += [f"      %a{i} = arith.constant {off} : i32", f"      %v{i} = arith.constant {o[3]} : i32"]
        if kind == "maskwrite32":
            out += [f"      %m{i} = arith.constant {o[4]} : i32",
                    f"      aiex.npu.maskwrite32(%a{i}, %v{i}, %m{i}) {attr} : i32, i32, i32"]
        else:
            out.append(f"      aiex.npu.write32(%a{i}, %v{i}) {attr} : i32, i32")
    out += ["      aiex.npu.dma_wait {symbol = @of_out}", "    }", "  }", "}", ""]
    return "\n".join(out)


# -- analysis -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    words: list[int]
    stamps: list[int]
    values: list[int]
    intervals: list[int]
    missing: list[int]
    wrong: list[int]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.wrong


def analyze(c: Case, out: bytes) -> Result:
    idx = polled_indices(c)
    n = len(idx)
    vals = struct.unpack_from(f"<{2 * n}I", out)
    stamps, seen = list(vals[:n]), list(vals[n:])
    return Result(idx, stamps, seen,
                  [(b - a) & 0xFFFFFFFF for a, b in zip(stamps, stamps[1:])],
                  [k for k, v in zip(idx, seen) if v == 0],
                  [k for k, v in zip(idx, seen) if v not in (0, marker(k))])


def analyze_spin(c: Case, out: bytes) -> int:
    t0, t1, _, _ = struct.unpack_from("<4I", out)
    return (t1 - t0) & 0xFFFFFFFF


def summarize(xs: list[int]) -> dict:
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs)}


# -- generate / build -----------------------------------------------------------------------

def generate(root: Path, only: list[str] | None = None) -> list[str]:
    names = []
    for c in all_cases():
        if only and c.name not in only:
            continue
        d = root / c.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "aie.mlir").write_text(mlir_text(c))
        (d / "kernel.s").write_text(kernel_text(c))
        meta = {"name": c.name, "kind": c.kind, "words": c.words, "block": c.block,
                "polled": polled_indices(c), "results_words": results_words(c),
                "spin_iters": c.spin_iters, "preset": c.preset,
                "filler": None if c.filler is None else vars(c.filler),
                "prefill": None if c.prefill is None else vars(c.prefill)}
        (d / "case.json").write_text(json.dumps(meta, indent=1) + "\n")
        names.append(c.name)
    return names


def _tool(name: str) -> str:
    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    p = Path(peano) / "bin" / name
    return str(p) if p.exists() else (shutil.which(name) or name)


def build_case(d: Path) -> None:
    """Assemble the kernel, then run aiecc (Peano linker) in the case directory."""
    for cmd, log in (([_tool("llvm-mc"), "--triple=aie2", "--filetype=obj", "-o", "kernel.o",
                       "kernel.s"], "mc.log"),
                     (["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin",
                       "--xclbin-name=aie.xclbin", "--get-npu-insts", "--npu-insts-name=insts.bin",
                       "aie.mlir"], "aiecc.log")):
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        (d / log).write_text(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        if r.returncode:
            raise RuntimeError(f"{d.name}: {cmd[0]} failed, see {d / log}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("dir")
    g.add_argument("--only", nargs="+")
    b = sub.add_parser("build")
    b.add_argument("dir")
    b.add_argument("--only", nargs="+")
    a = ap.parse_args(argv)
    root = Path(a.dir)
    if a.cmd == "generate":
        print("\n".join(generate(root, a.only)))
    else:
        for d in sorted(p for p in root.iterdir() if (p / "aie.mlir").is_file()):
            if not a.only or d.name in a.only:
                build_case(d)
                print("built", d.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
