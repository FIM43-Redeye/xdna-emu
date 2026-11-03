#!/usr/bin/env python3
"""Cascade read-timing probes (where a `vmov x, SCD` stalls); writes a suite (harness/suite.py).

One producer, one consumer, one cascade flow (same placement as gen_cascade_probe).
The consumer runs a small program with `@ts` markers; each marker reads the core
cycle counter (`mov rB:rA, cntr`) into a register pair, and the low words are
stored to the output once the program is done (so the stores never perturb the
measured window). Timer differences within one core are the only timing used.

Two modes fix which side is late:
  absent   the producer waits `late_iters` loop iterations before writing, so the
           consumer is already waiting when the word arrives (read of an empty cascade)
  present  the consumer waits `late_iters` loop iterations before its program, so the
           word is already buffered when the program starts (read latency alone)

Consumer output layout (little-endian u32): TS_SLOTS timestamps (low words, unused
slots zero), padded to 32 bytes, then one 64-byte read-result record per `vst` pair
the program stores at its `@store` markers.
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_cascade_probe as base                                   # noqa: E402
from harness import suite                                           # noqa: E402
from harness.topology import Buffer                                # noqa: E402

TS_PAIRS = [(17, 16), (19, 18), (21, 20), (23, 22), (25, 24), (27, 26), (29, 28), (31, 30)]
TS_SLOTS = len(TS_PAIRS)
TS_BYTES = 32                      # TS_SLOTS words, already 32-byte aligned
WORD = base.WORD_BYTES


def _frame(body: list[str]) -> str:
    """A returning leaf function saving r16-r31 (the timestamp pairs use them)."""
    pro = [".text", ".globl test_kernel", "test_kernel:", "  paddb [sp], #64"]
    pro += [f"  st r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi = [f"  lda r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi += ["  nop"] * 7 + ["  paddb [sp], #-64", "  ret lr"] + ["  nop"] * 5
    return "\n".join(pro + body + epi) + "\n"


def expand_program(lines: list[str]) -> tuple[list[str], int, int]:
    """Replace `@ts` and `@store` markers; returns (asm, n_timestamps, n_stores).

    `@ts`    -> a counter read into the next pair
    `@store` -> store x0 (wl0/wh0) to the result record area at [p1], advance p1
    The timestamp stores are appended after the program (padding nops let the
    counter reads land first).
    """
    out: list[str] = []
    n_ts = n_st = 0
    for ln in lines:
        if ln.strip() == "@ts":
            if n_ts >= TS_SLOTS:
                raise ValueError(f"at most {TS_SLOTS} timestamps")
            hi, lo = TS_PAIRS[n_ts]
            out.append(f"  mov r{hi}:r{lo}, cntr")
            n_ts += 1
        elif ln.strip() == "@store":
            out += ["  vst wl0, [p1, #0]", "  vst wh0, [p1, #32]", f"  padda [p1], #{WORD}"]
            n_st += 1
        else:
            out.append(ln)
    out += ["  nop"] * 6
    for i in range(n_ts):
        out += [f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * i}]"]
    return out, n_ts, n_st


# -- cases ---------------------------------------------------------------------

N = "  nop"


def case_program(name: str) -> list[str]:
    """Consumer program for a named case (`family` or `family_param`)."""
    fam, _, par = name.partition("_")
    d = int(par) if par.isdigit() else 0
    if fam == "use":           # vmov x0,SCD; d nops; ts; vst (first use); ts
        return ["@ts", "  vmov x0, SCD", *[N] * d, "@ts", "@store", "@ts"]
    if fam == "nouse":         # no use of x0 at all: does the core ever wait?
        return ["@ts", "  vmov x0, SCD", *[N] * d, "@ts", *[N] * 8, "@ts"]
    if fam == "indep":         # an independent vector op between vmov and the use
        return ["@ts", "  vmov x0, SCD", "  vmov x2, x3", "@ts", *[N] * d, "@ts", "@store", "@ts"]
    if fam == "bm":            # accumulator-register destination, then a use
        return ["@ts", "  vmov bml1, SCD", *[N] * d, "@ts", "  vmov x0, bml1", "@ts", "@store", "@ts"]
    if fam == "waw":           # overwrite the destination before any read of it
        return ["@ts", "  vmov x0, SCD", "  vbcst.32 x0, r1", "@ts", *[N] * d, "@ts", "@store", "@ts"]
    if fam == "two":           # two back-to-back reads, d nops between them
        return ["@ts", "  vmov x0, SCD", *[N] * d, "  vmov x4, SCD", "@ts", "@store", "@ts",
                "  vmov x0, x4", "@ts", "@store"]
    raise ValueError(f"unknown case {name!r}")


def reads_per_case(name: str) -> int:
    return 2 if name.startswith("two") else 1


def producer_asm(words: int, late_iters: int) -> str:
    body = base._enable("crMCDEn")
    for j in range(base.DEPTH_WORD_REGS):
        body += [f"  mov r1, #{base.depth_word_value(j)}", *base._nops(2), f"  vbcst.32 x{j}, r1"]
    body += base._nops(8)
    if late_iters:
        body += base.delay_loop(late_iters, ".Lprod")
    for i in range(words):
        body += [f"  vmov MCD, x{i % base.DEPTH_WORD_REGS}"]
    return base._frame(body)


def consumer_asm(case: str, late_iters: int) -> str:
    prog, n_ts, n_st = expand_program(case_program(case))
    body = base._enable("crSCDEn")
    # Pre-load every destination with a value the producer never sends (7): the cores keep
    # registers across runs, so a stale read must not look like the previous run's data.
    body += ["  mov r1, #7", "  mov p1, p0", f"  padda [p1], #{TS_BYTES}", *base._nops(2),
             "  vbcst.32 x0, r1", "  vbcst.32 x4, r1", *base._nops(2),
             "  vmov bml1, x0", *base._nops(6)]
    if late_iters:
        body += base.delay_loop(late_iters, ".Lcons")
    body += prog
    return _frame(body)


def consumer_out_bytes(case: str) -> int:
    return TS_BYTES + reads_per_case(case) * WORD


@dataclass(frozen=True)
class ReadResult:
    ts: list[int]                  # raw low words, only the slots the case uses
    deltas: list[int]              # consecutive differences (mod 2^32)
    words: list[tuple[int, ...]]


def parse_result(case: str, consumer_out: bytes) -> ReadResult:
    _, n_ts, _ = expand_program(case_program(case))
    ts = list(struct.unpack_from(f"<{n_ts}I", consumer_out))
    words = [struct.unpack_from("<16I", consumer_out, TS_BYTES + i * WORD)
             for i in range(reads_per_case(case))]
    return ReadResult(ts, [(ts[i + 1] - ts[i]) & 0xFFFFFFFF for i in range(n_ts - 1)], words)


def read_case(case: str, mode: str, late_iters: int, direction: str = "south") -> base.Case:
    if mode not in ("absent", "present"):
        raise ValueError("mode is absent or present")
    prod_late = late_iters if mode == "absent" else 0
    cons_late = late_iters if mode == "present" else 0
    name = f"rd_{case}_{mode}_{direction}"
    return base._pair(name, direction,
                      producer_asm(reads_per_case(case), prod_late), consumer_asm(case, cons_late),
                      (Buffer("ts", "out", 16),), (Buffer("log", "out", consumer_out_bytes(case)),))


def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    names = []
    for case in args.cases:
        for mode in args.modes:
            topo, files = read_case(case, mode, args.late_iters, args.direction)
            suite.write_case(out, topo.name, topo, files=files,
                             meta={"case": case, "mode": mode, "late_iters": args.late_iters,
                                   "direction": args.direction})
            names.append(topo.name)
    suite.write_suite_json(out, "cascade-read", "gen_cascade_read_probe.py", names)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["absent", "present"])
    ap.add_argument("--late-iters", type=int, default=1500)
    ap.add_argument("--direction", default="south")
    return cmd_generate(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
