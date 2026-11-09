#!/usr/bin/env python3
"""Zero-overhead loop set-up probes: how many times does a hand-written loop run?

Each kernel runs up to 16 loops back to back; loop k counts its passes in r6 and stores the
count to word k of the output buffer, so the iteration count of every variant comes back as
data (no trace needed). The variants sweep what a set-up can differ in: the order of the
`add.nc lc` / `movxm ls` / `movxm le` writes, the bytes between the last of them and the loop
end (llvm-aie demands 112, `getZOLSupport` LoopSetupDistance), the alignment of LS, the LE
convention. All loops are bounded by LC, so a kernel always terminates.

    python3 tools/gen_zol_probe.py build OUT_DIR [--cases NAME ...]
    python3 tools/gen_zol_probe.py run OUT_DIR TAG --backend hw|emu [--only NAME ...]
    python3 tools/gen_zol_probe.py show OUT_DIR HW_TAG EMU_TAG [--only NAME ...]

`run --backend hw` takes the NPU lock. The kernels preserve r16/r17 and the link register.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_lock_pulse_probe as lp  # noqa: E402  (shared kernel scaffolding)
from gen_lock_pulse_probe import N, nops  # noqa: E402

LC_COUNT = 4          # passes a correctly set-up loop makes
W = 5                 # output words per loop: passes (r6), LC afterwards (r5), bundle after LE (r7), bundle before LS (r9), line after LE's (r8)


@dataclass(frozen=True)
class Loop:
    """One loop. `order` is the sequence of set-up bundles; `gap` nops sit between the last
    of them and the body; the body is `body` bundles (the first counts, the last is LE's)."""
    body: int = 12
    gap: int = 2
    order: tuple[str, ...] = ("lc", "ls", "le")
    align_ls: bool = True
    le_past: bool = False         # LE = the bundle after the last body bundle
    lc: int = LC_COUNT
    between: int = 0              # nops between consecutive set-up bundles
    align_le: bool = False        # `.p2align 4` before the LE bundle (LE at a fetch-line start)
    fill: str = N                 # the body's filler bundle (a 2-byte nop by default)
    ls_skew: int | None = None    # None: LS aligned; n: LS at line offset 4 + 2n, a counted bundle (r9) before it in its line
    escape: int = 0               # leave the loop through a `jz` after this many passes (for LC=0, which may not end)
    one_line: bool = False        # LS and LE are the same bundle, at a fetch-line start
    pre: int = 0                  # nops before the first set-up bundle (shifts the set-up against the fetch lines)
    le_nops: int = 0              # nops after the alignment and before the LE bundle (LE at line offset 2 * le_nops)


@dataclass(frozen=True)
class ZolCase:
    name: str
    loops: list[Loop] = field(default_factory=list)

    def consumer_asm(self) -> str:
        body: list[str] = ["  mov r4, #%d" % LC_COUNT, *nops(3)]
        for k, lo in enumerate(self.loops):
            body += loop_asm(k, lo)
        return lp.frame(body)


def loop_asm(k: int, lo: Loop) -> list[str]:
    t = f"z{k}_"
    setup = {"lc": f"  add.nc lc, r4, #0", "ls": f"  movxm ls, #{t}ls", "le": f"  movxm le, #{t}le"}
    # scrub: LC, LS and LE of the previous loop must not leak into this one
    # (a stale LS that a late-landing LC could redirect to must be harmless: the loop's exit)
    out = ["  mov r5, #0", f"  movxm ls, #{t}end", "  movxm le, #0", "  add.nc lc, r5, #0", *nops(40),
           f"  mov r4, #{lo.lc}", "  mov r6, #0", "  mov r7, #0", "  mov r8, #0", "  mov r9, #0", *nops(2), *nops(lo.pre)]
    for i, which in enumerate(lo.order):
        out += [f"{t}s{which}:", setup[which]]
        if i + 1 < len(lo.order):
            out += nops(lo.between)
    out += nops(lo.gap)
    if lo.ls_skew is not None:
        out += [".p2align 4", "  add r9, r9, #1", *nops(lo.ls_skew)]
    elif lo.align_ls:
        out += [".p2align 4"]
    if lo.one_line:
        out += [".p2align 4", f"{t}ls:", f"{t}le:", "  add r6, r6, #1"]
        out += ["  add r7, r7, #1", f"{t}end:", ".p2align 4", "  add r8, r8, #1", *nops(8), f"{t}out:", ".p2align 4", "  mov r5, lc", *nops(2)]
        out += [f"  st r{r}, [p0, #{4 * (W * k + i)}]" for i, r in enumerate((6, 5, 7, 9, 8))]
        return out + nops(4)
    out += [f"{t}ls:", "  add r6, r6, #1"]
    if lo.escape:
        # r10 = passes - escape; leave when it reaches zero (the delay slots are body bundles)
        out += [f"  add r10, r6, #{-lo.escape}", f"  jz r10, #{t}out"]
        out += [lo.fill] * (lo.body - 4)
    else:
        out += [lo.fill] * (lo.body - 2)
    if lo.align_le:
        out += [".p2align 4", *nops(lo.le_nops)]
    # r7 counts the bundle right after LE: once if the loop ends at LE, once per pass otherwise
    out += [f"{t}le:", N] if not lo.le_past else [N, f"{t}le:", N]
    out += ["  add r7, r7, #1", f"{t}end:", ".p2align 4", "  add r8, r8, #1", *nops(8), f"{t}out:", ".p2align 4", "  mov r5, lc", *nops(2)]
    out += [f"  st r{r}, [p0, #{4 * (W * k + i)}]" for i, r in enumerate((6, 5, 7, 9, 8))]
    return out + nops(4)


WIDE = "  nopa ; nopb ; nops ; nopxm ; nopv"      # a 16-byte empty bundle
MID = "  nopb ; mov r8, #5"                         # 6 bytes


def pack(add, prefix: str, loops: list[Loop], per: int = 1) -> None:
    """`loops` in kernels of `per` loops (the output buffer holds sixteen words, five per loop).
    One loop per kernel by default: an ill-formed loop leaves state in the program-control unit
    that changes how the next loop in the same kernel behaves (see the doc row)."""
    for i in range(0, len(loops), per):
        add(ZolCase(f"{prefix}_{i // per:02d}", loops[i:i + per]))


def cases() -> dict[str, ZolCase]:
    out: dict[str, ZolCase] = {}

    def add(c: ZolCase) -> None:
        out[c.name] = c

    far = dict(gap=0, pre=0, align_le=True)
    # llvm-aie's own shape: LE at a fetch-line start, the set-up 112+ bytes before it
    pack(add, "ref", [Loop(body=60, **far), Loop(body=12, **{**far, "gap": 2}), Loop(body=40, **{**far, "gap": 2})])
    # LC values on a well-formed loop (LC=0 bounded by the escape)
    pack(add, "lc", [Loop(body=40, lc=v, **far, escape=12 if v == 0 else 0) for v in (0, 1, 2, 3, 5, 8)])
    # what executes around LE and LS: the line after LE, the rest of LE's line, a bundle before LS in its line
    pack(add, "skew", [Loop(body=40, ls_skew=n, **far) for n in range(0, 7)])
    # which register's write the loop end has to clear, by how much: one register written last,
    # 140 bytes after the other two
    for order in (("lc", "ls", "le"), ("lc", "le", "ls"), ("ls", "le", "lc")):
        pack(add, "last_" + order[-1],
             [Loop(body=b, gap=0, pre=p, align_le=True, order=order, between=70) for b in (20, 28) for p in range(8)])
    pack(add, "lastls_far", [Loop(body=b, gap=0, pre=p, align_le=True, order=("lc", "le", "ls"), between=70)
                             for b in (36, 44, 52) for p in (0, 3)])
    # loops that fit one or two fetch lines (LS in the LE line; the fetch is still running ahead)
    pack(add, "short", [Loop(gap=60, one_line=True)] + [Loop(body=b, gap=60, align_le=True) for b in (2, 3, 8, 9, 10)])
    # LE off the line start
    pack(add, "le_off", [Loop(body=40, gap=0, align_le=True, le_nops=n) for n in range(0, 8)])
    return out


# -- build / run -----------------------------------------------------------------

def build_case(case: ZolCase, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(case.consumer_asm(), d / "consumer.o", "k_consumer")
    (d / "base.mlir").write_text(lp.design_mlir(lp.Case(case.name, [], {})))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_pc", "--core-sweep-events", "INSTR_EVENT_0",
           "--trace-config-out", "trace_config.json", "--config-test-name", case.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"], cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")


def elf_of(case_dir: Path) -> Path:
    return case_dir / "aie.mlir.prj" / "elfs_main_core_0_2" / "elfs_main_core_0_2.elf"


def label_addrs(elf: Path) -> dict[str, int]:
    syms = subprocess.run([lp.peano("llvm-nm"), str(elf)], capture_output=True, text=True, check=True).stdout
    return {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+) \w (\S+)$", syms, re.M)}


def counts(run_dir: Path) -> list[int] | None:
    f = run_dir / "out.bin"
    return np.fromfile(f, dtype=np.uint32).tolist()[:16] if f.exists() else None


def triple(words: list[int] | None, k: int) -> tuple[int, ...] | None:
    return tuple(words[W * k:W * k + W]) if words else None


def geometry(case: ZolCase, case_dir: Path, k: int) -> str:
    lab = label_addrs(elf_of(case_dir))
    lo = case.loops[k]
    t = f"z{k}_"
    end = lab[f"{t}le"]
    parts = [f"{w}->LE {end - lab[t + 's' + w]:3}B" for w in lo.order]
    return f"LS 0x{lab[t + 'ls']:x} LE 0x{end:x} " + " ".join(parts)


def show_table(out: Path, tag: str, only: list[str]) -> None:
    """One line per loop: what was set last, where it sat, and the result, for fitting a rule."""
    tab = cases()
    for n in [n for n in tab if (not only or n in only) and (out / n / tag).is_dir()]:
        words = counts(out / n / tag)
        lab = label_addrs(elf_of(out / n))
        for k, lo in enumerate(tab[n].loops):
            t = f"z{k}_"
            last = lo.order[-1]
            a = lab[t + "s" + last]
            ls, le = lab[t + "ls"], lab[t + "le"]
            res = triple(words, k)
            print(f"{n:12} {k} order={''.join(w[1] for w in lo.order)} last@0x{a:x}(+{a % 16:2}) "
                  f"last->LE {le - a:3} last->LS {ls - a:3} LS->LE {le - ls:3} LE+{le % 16:<2} -> {res}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--cases", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--only", nargs="*", default=[])
    s = sub.add_parser("show")
    s.add_argument("out_dir")
    s.add_argument("hw_tag", help="run directory name (TAG.hw)")
    s.add_argument("emu_tag", help="run directory name (TAG.emu)")
    s.add_argument("--only", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("tag", help="run directory name (TAG.hw)")
    t.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args(argv)
    out = Path(a.out_dir).resolve()
    if a.cmd == "table":
        show_table(out, a.tag, a.only)
        return 0
    table = cases()
    if a.cmd == "build":
        for name in a.cases or table:
            build_case(table[name], out)
            print("built", name)
        return 0
    names = [n for n in table if (not a.only or n in a.only) and (out / n).is_dir()]
    if a.cmd == "run":
        return 0 if all([lp.run_case_raw(out, n, a.tag, a.backend) for n in names]) else 1
    bad = 0
    for n in names:
        h = counts(out / n / a.hw_tag)
        e = counts(out / n / a.emu_tag)
        print(f"== {n}")
        for k, lo in enumerate(table[n].loops):
            hv = triple(h, k)
            ev = triple(e, k)
            flag = "" if hv == ev else "  DIFF"
            bad += hv != ev
            print(f"  {k:2} body={lo.body:3} gap={lo.gap} {''.join(w[1] for w in lo.order)} (passes, LC, post-LE, pre-LS, next-line) hw={hv} emu={ev}{flag}  "
                  f"{geometry(table[n], out / n, k)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
