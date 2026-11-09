#!/usr/bin/env python3
"""Taken-branch stall probes: cycles a `j` / `jnz` / `jl` + `ret` costs beyond its bundles.

Each site reads the core cycle counter (`mov r17:r16, cntr`) before and after a short code
sequence containing the branch and stores the difference to word k of the output buffer; the
expected cycle count of the executed bundles (one per bundle) is derived from the linked ELF's
disassembly, so `extra = measured - executed bundles` is the stall. A site is described by the
alignment of its bundles against the 16-byte fetch lines (`lead` nops before the first read, `pre`
between the read and the branch), the distance the branch skips, and the size of the bundles
around it. Sixteen sites per kernel; a kernel preserves r16/r17 (saved by the frame) and lr.

    python3 tools/gen_branch_stall_probe.py build OUT_DIR [--cases NAME ...]
    python3 tools/gen_branch_stall_probe.py run OUT_DIR TAG --backend hw|emu [--only NAME ...]
    python3 tools/gen_branch_stall_probe.py show OUT_DIR TAG [--only NAME ...]

`run --backend hw` takes the NPU lock.
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

WIDE = "  nopa ; nopb ; nops ; nopxm ; nopv"      # a 16-byte empty bundle
MID = "  nopb ; mov r8, #5"                         # 6 bytes
PAD = 24


@dataclass(frozen=True)
class Site:
    kind: str                     # straight | j | jnz_taken | jnz_not | jl
    lead: int = 0                 # nops before the first counter read (shifts everything against the lines)
    pre: int = 0                  # bundles between the first read and the branch
    post: int = 0                 # bundles between the branch target and the second read
    skip: int = 12                # fall-through bundles after the delay slots (not run when taken)
    fill: str = N                 # filler bundle of the pre / post / delay-slot / skip regions
    delay_fill: str | None = None # delay-slot filler when it differs
    align_target: bool = False    # `.p2align 4` before the target label


@dataclass(frozen=True)
class BranchCase:
    name: str
    sites: list[Site] = field(default_factory=list)
    chain: tuple[int, ...] = ()      # counter reads at these positions around a `j` (sites in sequence)

    def consumer_asm(self) -> tuple[str, list[str]]:
        if self.chain:
            return chain_asm(self.chain), []
        body: list[str] = ["  mov r5, #1", "  mov r6, #0", "  mov r10, lr", *nops(3)]
        funcs: list[str] = []
        for k, s in enumerate(self.sites):
            b, f = site_asm(k, s)
            body += b
            funcs += f
        asm = lp.frame(body).replace("  paddb [sp], #-64", "  mov lr, r10\n  nop\n  nop\n  paddb [sp], #-64", 1)
        return asm + "\n".join(["", *funcs]) + "\n", funcs


def site_asm(k: int, s: Site) -> tuple[list[str], list[str]]:
    t = f"b{k}_"
    dfill = s.delay_fill or s.fill
    out = [*nops(PAD), *nops(s.lead), f"{t}rd1:", "  mov r17:r16, cntr", *[s.fill] * s.pre, f"{t}br:"]
    funcs: list[str] = []
    if s.kind == "straight":
        out += [s.fill, *[dfill] * 5, *[s.fill] * s.skip]
    elif s.kind == "j":
        out += [f"  j #{t}tgt", *[dfill] * 5, *[s.fill] * s.skip]
    elif s.kind == "jnz_taken":
        out += [f"  jnz r5, #{t}tgt", *[dfill] * 5, *[s.fill] * s.skip]
    elif s.kind == "jnz_not":
        out += ["  jnz r6, #" + f"{t}tgt", *[dfill] * 5, *[s.fill] * s.skip]
    elif s.kind == "jl":
        out += [f"  jl #{t}fn", *[dfill] * 5]
        funcs += [".p2align 4", f"{t}fn:", f"{t}fe:", *[s.fill] * s.skip, f"{t}ret:", "  ret lr", *[dfill] * 5]
    else:
        raise ValueError(s.kind)
    if s.align_target:
        out += [".p2align 4"]
    out += [f"{t}tgt:", *[s.fill] * s.post, f"{t}rd2:", "  mov r19:r18, cntr", "  sub r20, r18, r16", *nops(2),
            f"  st r20, [p0, #{4 * k}]", *nops(2)]
    return out, funcs


def chain_asm(positions: tuple[int, ...]) -> str:
    """The `br_j` layout of gen_fetch_pc_probe (a taken `j` at bundle 5 of 11, twelve skipped
    bundles, a target and eight bundles), one site per entry of `positions`, the counter read
    in the bundle `5 + pos`. The reads land in r16..r31 pairs and are stored at the end."""
    body: list[str] = ["  mov r10, lr", *nops(3)]
    for i, pos in enumerate(positions):
        t = f"c{i}_"
        specs = [N] * 11
        specs[5] = f"  j #{t}tgt"
        specs[5 + pos] = f"  mov r{17 + 2 * i}:r{16 + 2 * i}, cntr"
        body += [*nops(PAD), *specs, *nops(12), f"{t}tgt:", N, *nops(8)]
    body += nops(6) + [f"  st r{16 + 2 * i}, [p0, #{4 * i}]" for i in range(len(positions))] + nops(4)
    return lp.frame(body).replace("  paddb [sp], #-64", "  mov lr, r10\n  nop\n  nop\n  paddb [sp], #-64", 1)


def cases() -> dict[str, BranchCase]:
    out: dict[str, BranchCase] = {}

    def add(name: str, sites: list[Site]) -> None:
        for i in range(0, len(sites), 16):
            n = name if len(sites) <= 16 else f"{name}_{i // 16}"
            out[n] = BranchCase(n, sites[i:i + 16])

    # calibration: the counter's own cost over straight-line code
    add("straight", [Site("straight", lead=l, pre=p, skip=q) for l in (0, 1) for p in (0, 3) for q in (4, 12)])
    # a `j` over 12 skipped bundles with its read-to-branch distance and line alignment swept
    add("j_align", [Site("j", lead=l, pre=p, post=2) for l in range(8) for p in (0, 4)])
    add("jnz_taken_align", [Site("jnz_taken", lead=l, pre=p, post=2) for l in range(8) for p in (0, 4)])
    add("jnz_not_align", [Site("jnz_not", lead=l, pre=p, post=2) for l in range(8) for p in (0, 4)])
    # bundle sizes around the branch, calls and returns, not-taken (taken `j` over 2-byte bundles costs nothing)
    # (wider bundles need a line-aligned target: a target truncated into the middle of a bundle
    # runs garbage and hangs the core; `jl` keeps to 2-byte bundles for the same reason with its
    # return address)
    # (the two `jl` placements kept have no bundle straddling the line the return resumes at: silicon
    # runs a return into the middle of a bundle as whatever it decodes to, the emulator stops)
    add("kinds_nop", [Site(k, lead=l, pre=p, post=2, skip=6) for k in ("j", "jnz_taken", "jnz_not")
                      for l in (0, 3) for p in (0, 3)] + [Site("jl", lead=0, pre=3, post=2, skip=6),
                                                          Site("jl", lead=3, pre=0, post=2, skip=6)])
    for tag, fill in (("wide", WIDE), ("mid", MID)):
        add(f"kinds_{tag}", [Site(k, lead=l, pre=p, post=2, fill=fill, skip=6, align_target=True)
                             for k in ("j", "jnz_taken", "jnz_not") for l in (0, 3) for p in (0, 3)])
    add("wide_delay", [Site(k, lead=l, pre=1, post=2, delay_fill=WIDE, skip=6, align_target=True)
                       for k in ("j", "jnz_taken") for l in range(0, 8)])
    # where after a taken branch do cycles go missing: the second counter read `post` bundles past the target
    add("j_post", [Site("j", lead=0, pre=0, post=q) for q in range(0, 16)])
    add("j_post_b", [Site("j", lead=0, pre=0, post=q) for q in range(16, 32)])
    add("j_post_c", [Site("j", lead=0, pre=0, post=q) for q in range(32, 48)])
    add("j_post_l3", [Site("j", lead=3, pre=0, post=q) for q in range(0, 16)])
    add("j_post_l3b", [Site("j", lead=3, pre=0, post=q) for q in range(16, 32)])
    add("straight_post", [Site("straight", lead=0, pre=0, skip=q) for q in range(0, 16)])
    out["chain_j"] = BranchCase("chain_j", chain=(-4, -3, -2, -1, 1, 2, 3, 4))
    out["chain_j2"] = BranchCase("chain_j2", chain=(-5, -4, -3, -2, -1, 1, 2, 3))
    return out


# -- build / run ---------------------------------------------------------------------

def build_case(case: BranchCase, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    asm, _ = case.consumer_asm()
    lp.assemble(asm, d / "consumer.o", "k_consumer")
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


def disassemble(elf: Path) -> tuple[list[int], dict[str, int]]:
    """Address of every bundle in .text order, and of every label."""
    text = subprocess.run([lp.peano("llvm-objdump"), "-d", str(elf)], capture_output=True, text=True, check=True).stdout
    addrs = [int(m.group(1), 16) for m in re.finditer(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )+", text, re.M)]
    syms = subprocess.run([lp.peano("llvm-nm"), str(elf)], capture_output=True, text=True, check=True).stdout
    labels = {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+) \w (\S+)$", syms, re.M)}
    return addrs, labels


def executed_bundles(site: Site, k: int, addrs: list[int], lab: dict[str, int]) -> int:
    """Bundles issued from the first counter read (inclusive) to the second (exclusive)."""
    t = f"b{k}_"
    import bisect

    class Ix(dict):
        def __missing__(self, a):          # a line start inside a bundle: the bundle that holds it
            return bisect.bisect_right(addrs, a) - 1

    ix = Ix({a: i for i, a in enumerate(addrs)})
    line = lambda a: a & ~15
    rd1, br, rd2 = ix[lab[t + "rd1"]], ix[lab[t + "br"]], ix[lab[t + "rd2"]]
    if site.kind in ("straight", "jnz_not"):
        return rd2 - rd1
    delays_end = br + 6                                   # the branch and its five delay slots
    first = delays_end - rd1
    if site.kind in ("j", "jnz_taken"):
        return first + (rd2 - ix[line(lab[t + "tgt"])])
    # jl: the callee from its line start to the ret and its delay slots, then back at the line of
    # the return address (the bundle after the caller's delay slots)
    ret = ix[lab[t + "ret"]]
    callee = ret + 6 - ix[line(lab[t + "fn"])]
    back = rd2 - ix[line(addrs[delays_end])]
    return first + callee + back


def deltas(run_dir: Path) -> list[int] | None:
    f = run_dir / "out.bin"
    return np.fromfile(f, dtype=np.uint32).astype(np.int64).tolist()[:16] if f.exists() else None


def table(out: Path, tag: str, only: list[str]) -> None:
    tab = cases()
    for n in [n for n in tab if (not only or n in only) and (out / n / tag).is_dir()]:
        d = deltas(out / n / tag)
        if tab[n].chain:
            print(f"== {n} {tag}: counter reads at positions {tab[n].chain}, differences between successive reads")
            print("   ", [int(b - a) & 0xFFFFFFFF for a, b in zip(d, d[1:])] if d else None)
            continue
        addrs, lab = disassemble(elf_of(out / n))
        print(f"== {n} {tag}")
        for k, s in enumerate(tab[n].sites):
            b = executed_bundles(s, k, addrs, lab)
            br = lab[f"b{k}_br"]
            tgt = lab[f"b{k}_tgt"]
            extra = None if d is None else d[k] - b
            print(f"  {k:2} {s.kind:10} lead={s.lead} pre={s.pre} post={s.post} skip={s.skip} br@+{br % 16:2} tgt@+{tgt % 16:2} "
                  f"bundles={b:3} measured={None if d is None else d[k]} extra={extra}")


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
    s.add_argument("tag", help="run directory name (TAG.hw / TAG.emu)")
    s.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args(argv)
    out = Path(a.out_dir).resolve()
    tab = cases()
    if a.cmd == "build":
        for name in a.cases or tab:
            build_case(tab[name], out)
            print("built", name)
        return 0
    names = [n for n in tab if (not a.only or n in a.only) and (out / n).is_dir()]
    if a.cmd == "run":
        return 0 if all([lp.run_case_raw(out, n, a.tag, a.backend) for n in names]) else 1
    table(out, a.tag, a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
