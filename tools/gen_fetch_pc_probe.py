#!/usr/bin/env python3
"""Mode-1 (EventPc) trace probes: which PC a core trace frame carries for an event.

A mode-1 frame holds the event mask and a 14-bit PC but no time. This probe lays out
straight-line, branch and loop code with a labelled site for every event under test, runs it
on silicon or the emulator in event-PC mode and prints each frame's PC as bundles from the
nearest site, using the linked ELF's disassembly for the bundle addresses.

    python3 tools/gen_fetch_pc_probe.py build OUT_DIR [--cases NAME ...]
    python3 tools/gen_fetch_pc_probe.py run OUT_DIR TAG --backend hw|emu [--only NAME ...]
    python3 tools/gen_fetch_pc_probe.py show OUT_DIR TAG [--only NAME ...]

`run --backend hw` takes the NPU lock. The kernels preserve r16/r17 (the aiecc main keeps
them live) and save/restore the link register around their own calls.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_lock_pulse_probe as lp  # noqa: E402  (shared kernel scaffolding)
from gen_lock_pulse_probe import N, OWN, A, nops  # noqa: E402

# grounding slots come first (PERF_CNT_2, INSTR_EVENT_0, INSTR_EVENT_1), the sweep follows
SWEEPS = {
    "ctl": "INSTR_LOCK_ACQUIRE_REQ,INSTR_LOCK_RELEASE_REQ,INSTR_CALL,INSTR_RETURN,INSTR_VECTOR",
    "mem": "INSTR_LOAD,INSTR_STORE,INSTR_VECTOR,LOCK_STALL,INSTR_LOCK_ACQUIRE_REQ",
}
MODE = "event_pc"                        # --mode event_time reads when each event fires instead
PAD = 24                                 # nops between sites, so frames and sites do not blur


def site(label: str, *slots: str) -> list[str]:
    """A labelled bundle: `label:` then the bundle's slots on one line."""
    return [f"{label}:", "  " + " ; ".join(slots) if slots else N]


@dataclass(frozen=True)
class FetchCase:
    name: str
    body: list[str]
    sweep: str = "ctl"
    inits: dict[int, int] = field(default_factory=dict)
    funcs: list[str] = field(default_factory=list)     # code placed after the epilogue
    # (event name, site label) in the order the events occur: `show` reports each such frame's
    # PC against its site
    sites: list[tuple[str, str]] = field(default_factory=list)

    def consumer_asm(self) -> str:
        setup = ["  mov r2, #1", "  mov r3, #-1", "  mov r10, lr", *nops(2)]
        main = lp.frame([*setup, *lp.delay_loop(lp.HEAD_ITERS, ".Lhead"), *nops(8), *self.body,
                         *nops(8), "  event #1", *nops(8), *lp.delay_loop(lp.TAIL_ITERS, ".Ltail")])
        # the saved link register goes back before the epilogue's `ret lr`
        main = main.replace("  paddb [sp], #-64", "  mov lr, r10\n  nop\n  nop\n  paddb [sp], #-64", 1)
        return main + "\n".join(["", *self.funcs]) + "\n"


def pad() -> list[str]:
    return nops(PAD)


def bundle_run(prefix: str, specs: list[str]) -> list[str]:
    """`specs` as bundles, every one carrying the label `prefix{index}` (so a PC reads off as a
    label); "nop" for an empty bundle."""
    out: list[str] = []
    for i, spec in enumerate(specs):
        out += [f"{prefix}{i}:", "  " + spec]
    return out


E0 = "event #0"


def at(specs: list[str], index: int, extra: str) -> list[str]:
    """`specs` with `extra` added to the slots of bundle `index`."""
    out = list(specs)
    out[index] = extra if out[index] == "nop" else f"{out[index]} ; {extra}"
    return out


def branch_variants(kind: str) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """E (event #0) at every bundle from 5 before to 5 after a taken `j`/`jl`/`ret`, one
    variant each, in one run. Returns (body, sites, callee code). For a branch the target is
    far from the fall-through code (12 skipped bundles between), so the fetch order after
    the delay slots shows whether the PC follows the taken branch."""
    body: list[str] = []
    sites: list[tuple[str, str]] = []
    callees: list[str] = []
    for pos in [p for p in range(-5, 6) if p != 0]:   # `event` does not share a bundle with a branch
        tag = f"{kind}{pos + 5:02d}_"
        specs = ["nop"] * 11
        if kind == "j":
            specs[5] = "j #" + tag + "tgt"
        elif kind == "jl":
            specs[5] = "jl #" + tag + "fn"
        else:
            raise ValueError(kind)
        specs = at(specs, 5 + pos, E0)
        body += [*pad(), *bundle_run(tag, specs)]
        sites.append(("INSTR_EVENT_0", f"{tag}{5 + pos}"))
        if kind == "j":
            body += [*bundle_run(tag + "skip", ["nop"] * 12), f"{tag}tgt:", *nops(1), *bundle_run(tag + "t", ["nop"] * 8)]
        else:
            body += [*bundle_run(tag + "r", ["nop"] * 8)]
            # the same E positions around the callee's `ret lr`
            cspecs = ["nop"] * 11
            cspecs[5] = "ret lr"
            callees += [f"{tag}fn:", *nops(1)]
            callees += bundle_run(tag + "c", at(cspecs, 5 + pos, E0))
            callees += [f"{tag}cx:", *nops(8)]
            sites.append(("INSTR_EVENT_0", f"{tag}c{5 + pos}"))
    return body, sites, callees


def loop_variants(set_lc: str = "add.nc lc, r4, #0", settle: int = 2, qs: range = range(7),
                  n_body: int = 12) -> tuple[list[str], list[tuple[str, str]]]:
    """A zero-overhead loop run 4 times with E at 0..6 bundles before its last bundle (LE),
    one loop each; `set_lc` loads the count, `settle` nops separate the set-up from the body."""
    body: list[str] = []
    sites: list[tuple[str, str]] = []
    for q in qs:
        tag = f"lp{q}_"
        specs = at(["nop"] * n_body, n_body - 1 - q, E0)
        body += [*pad(), "  mov r4, #4", *nops(3), "  " + set_lc, f"  movxm ls, #{tag}0", f"  movxm le, #{tag}{n_body - 1}",
                 *nops(settle), *bundle_run(tag, specs), *bundle_run(tag + "x", ["nop"] * 8)]
        sites += [("INSTR_EVENT_0", f"{tag}{n_body - 1 - q}")] * 4
    return body, sites


def cases() -> dict[str, FetchCase]:
    out: dict[str, FetchCase] = {}

    def add(c: FetchCase) -> None:
        out[c.name] = c

    # one of each event, in straight-line code; the callee sits after the epilogue
    callee = ["site_fn:", "  nop", "site_ret:", "  ret lr", *nops(5)]
    add(FetchCase(
        "straight",
        [*pad(), *site("site_ev0", "event #0"), *pad(),
         *site("site_acq", f"acq #{OWN + A}, r3"), *pad(),
         *site("site_rel", f"rel #{OWN + A}, r2"), *pad(),
         *site("site_call", "jl #site_fn"), *nops(5), *pad(),
         *site("site_vec", "vadd.32 x2, x0, x1"), *pad()],
        inits={A: 1}, funcs=callee,
        sites=[("INSTR_EVENT_0", "site_ev0"), ("INSTR_LOCK_ACQUIRE_REQ", "site_acq"),
               ("INSTR_LOCK_RELEASE_REQ", "site_rel"), ("INSTR_CALL", "site_call"),
               ("INSTR_RETURN", "site_ret")]))
    # event #0 sites a varying number of nops apart: does the PC lead depend on the alignment
    # of the event or on how far straight-line code has run since the last branch?
    scan = []
    for k in range(40):
        scan += [*site(f"site_s{k:02d}", "event #0"), *nops(1 + (k * 3) % 7)]
    add(FetchCase("scan", [*pad(), *scan, *pad()],
                  sites=[("INSTR_EVENT_0", f"site_s{k:02d}") for k in range(40)]))
    # the same, between bundles of other sizes: is the lead a number of bundles or of bytes?
    wide = ["  nopa ; nopb ; nops ; nopxm ; nopv"]
    mid = ["  nopb ; mov r8, #5"]
    for tag, filler in (("wide", wide), ("mid", mid)):
        sc = []
        for k in range(24):
            sc += [*site(f"site_s{k:02d}", "event #0"), *(filler * (1 + (k * 3) % 5)), *nops(k % 3)]
        add(FetchCase(f"scan_{tag}", [*pad(), *sc, *pad()],
                      sites=[("INSTR_EVENT_0", f"site_s{k:02d}") for k in range(24)]))
    for kind in ("j", "jl"):
        body, sites, callees = branch_variants(kind)
        add(FetchCase(f"br_{kind}", body, sites=sites, funcs=callees))
    lbody, lsites = loop_variants()
    add(FetchCase("loops", lbody, sites=lsites))
    for name, kw in (("loops_mov", {"set_lc": "mov lc, r4"}), ("loops_settle6", {"settle": 6}),
                     ("loops_long", {"n_body": 40})):
        lbody, lsites = loop_variants(qs=range(2), **kw)
        add(FetchCase(name, lbody, sites=lsites))
    add(FetchCase(
        "straight_mem",
        [*pad(), *site("site_ld", "lda r5, [p0, #0]"), *pad(),
         *site("site_st", "st r5, [p0, #0]"), *pad(),
         *site("site_vec", "vadd.32 x2, x0, x1"), *pad(),
         *site("site_acq", f"acq #{OWN + A}, r3"), *pad()],
        sweep="mem", inits={A: 1},
        sites=[("INSTR_LOAD", "site_ld"), ("INSTR_STORE", "site_st"), ("INSTR_VECTOR", "site_vec"),
               ("INSTR_LOCK_ACQUIRE_REQ", "site_acq")]))
    return out


# -- build / run -------------------------------------------------------------------

def lp_case(case: FetchCase) -> lp.Case:
    """The scaffolding case the MLIR generator reads (initial lock counts, no producer)."""
    return lp.Case(case.name, [], case.inits)


def build_case(case: FetchCase, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(case.consumer_asm(), d / "consumer.o", "k_consumer")
    (d / "base.mlir").write_text(lp.design_mlir(lp_case(case)))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", MODE, "--core-sweep-events", SWEEPS[case.sweep],
           "--trace-config-out", "trace_config.json", "--config-test-name", case.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"], cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")


def elf_of(case_dir: Path) -> Path:
    return case_dir / "aie.mlir.prj" / "elfs_main_core_0_2" / "elfs_main_core_0_2.elf"


def disassemble(elf: Path) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """(address, text) of every bundle in .text, and the address of every label."""
    objdump = lp.peano("llvm-objdump")
    text = subprocess.run([objdump, "-d", str(elf)], capture_output=True, text=True, check=True).stdout
    bundles = [(int(m.group(1), 16), m.group(2).strip())
               for m in re.finditer(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(.*)$", text, re.M)]
    syms = subprocess.run([lp.peano("llvm-nm"), str(elf)], capture_output=True, text=True, check=True).stdout
    labels = {m.group(2): int(m.group(1), 16) for m in re.finditer(r"^([0-9a-f]+) \w (\S+)$", syms, re.M)}
    return bundles, labels


def run_case(out: Path, name: str, tag: str, backend: str) -> bool:
    return (lp.run_case if MODE == "event_time" else lp.run_case_raw)(out, name, tag, backend)


def frames(trace_bin: Path, row: int = 2) -> list[tuple[int, int]]:
    """(slot mask, pc) of every EventPC frame of the core trace of `row`, repeats expanded."""
    import numpy as np
    from trace_decoder.decode import decode_words
    from trace_decoder.frame import EventCmd, RepeatCmd
    out: list[tuple[int, int]] = []
    cmds = decode_words(np.fromfile(trace_bin, dtype=np.uint32).tolist(), None)
    for (pkt_type, r, _col), cs in cmds.items():
        if pkt_type != 0 or r != row:
            continue
        for c in cs:
            if isinstance(c, EventCmd):
                out.append((c.event_bits, c.cycles))
            elif isinstance(c, RepeatCmd) and out:
                out.extend([out[-1]] * c.count)
    return out


def slot_names(case_dir: Path) -> list[str]:
    import json
    cfg = json.loads((case_dir / "trace_config.json").read_text())
    for tile in cfg["tiles_traced"]:
        if tile.get("row") == 2 and tile.get("kind") == "core":
            return tile["events"]
    return []


def where(pc: int, bundles: list[tuple[int, str]], labels: dict[str, int]) -> str:
    """`pc` as bundles after the nearest site label at or before it."""
    addrs = [a for a, _ in bundles]
    sites = sorted((a, n) for n, a in labels.items() if n.startswith("site_"))
    near = [s for s in sites if s[0] <= pc]
    if not near or pc not in addrs:
        return f"0x{pc:x}"
    a0, n0 = near[-1]
    k = addrs.index(pc) - addrs.index(a0)
    return f"{n0}+{k}b (0x{pc:x})"


def show_times(out: Path, name: str, tag: str, backend: str) -> None:
    """Event-time runs: the cycle of every INSTR_EVENT_0 and the gaps between them."""
    f = out / name / f"{tag}.{backend}" / "events.json"
    if not f.exists():
        return
    events = [e for e in lp.core_events(f) if e["name"] not in ("PERF_CNT_2",)]
    field = "ts" if backend == "hw" else "soc"
    swept = [e for e in events if e["name"] in {n for n, _ in cases()[name].sites}]
    print(f"== {name} {tag}.{backend}")
    prev = None
    for e in swept[:60]:
        print(f"  {e['name']:26} t={e[field]}" + ("" if prev is None else f"  +{e[field] - prev}"))
        prev = e[field]


def show(out: Path, name: str, tag: str, backend: str) -> None:
    if MODE == "event_time":
        return show_times(out, name, tag, backend)
    d = out / name
    path = d / f"{tag}.{backend}" / "trace_raw.bin"
    if not path.exists():
        return
    case = cases()[name]
    bundles, labels = disassemble(elf_of(d))
    addrs = [a for a, _ in bundles]
    names = slot_names(d)
    first = min(labels[lab] for _, lab in case.sites) if case.sites else 0
    seen: dict[str, list[int]] = {}
    for mask, pc in frames(path):
        for i in range(8):
            if mask & (1 << i) and names[i] != "PERF_CNT_2" and pc >= first:
                seen.setdefault(names[i], []).append(pc)
    print(f"== {name} {tag}.{backend}")
    cursor: dict[str, int] = {}
    for ev, lab in case.sites:
        k = cursor.get(ev, 0)
        cursor[ev] = k + 1
        pcs = seen.get(ev, [])
        if k >= len(pcs):
            print(f"  {ev:26} {lab:12} no frame")
            continue
        issue = labels[lab]
        pc = pcs[k]
        lead = addrs.index(pc) - addrs.index(issue) if pc in addrs else None
        at_pc = [n for n, a in labels.items() if a == pc and n[0] in "lsjtc"]
        print(f"  {ev:26} {lab:12} issue 0x{issue:x} pc 0x{pc:x} = {','.join(at_pc) or '?'}  +{pc - issue}B  +{lead} bundles")


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
    s.add_argument("tag", help="run directory name without the backend suffix")
    s.add_argument("--backend", choices=("hw", "emu"), required=True)
    s.add_argument("--only", nargs="*", default=[])
    for sp in (b, r, s):
        sp.add_argument("--mode", choices=("event_pc", "event_time"), default="event_pc")
    a = ap.parse_args(argv)
    global MODE
    MODE = a.mode
    out = Path(a.out_dir).resolve()
    table = cases()
    if a.cmd == "build":
        for name in a.cases or table:
            build_case(table[name], out)
            print("built", name)
        return 0
    names = [n for n in table if (not a.only or n in a.only) and (out / n).is_dir()]
    if a.cmd == "run":
        return 0 if all([run_case(out, n, a.tag, a.backend) for n in names]) else 1
    for n in names:
        show(out, n, a.tag, a.backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
