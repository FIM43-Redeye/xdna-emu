#!/usr/bin/env python3
"""Core load-port probe: when two of the core's own memory ports meet on one bank, is the conflict stamped?

The question (docs/fidelity-gaps/core-compute-timing.md, "Two core load ports on one bank"): on silicon a
scalar load (LoadA) and a vector load (LoadB, two banks) that reached the memory stage on one bank in one
cycle stalled the core (MEMORY_STALL) without raising CONFLICT_DM_BANK, while a core load meeting a core
store does raise it. One captured cycle cannot tell these readings apart:

  R1  a conflict between two core LOAD ports is never stamped (it only stalls);
  R2  the vector load's second bank is not stamped (its first is);
  R3  stamping depends on which port loses (modelled here as: stamped only when load A loses).

Every case is core-only: no DMA channel of the compute tile is configured, the core is started by the
runtime sequence's go lock (as the lock-fetch probes), and ONE bundle holds the two accesses, so they reach
their banks (issue + 4) in the same cycle. `event #0` / `event #1` bracket the meeting; a `rel` of lock 0
right after `event #0` gives the memory module a LOCK_0_REL anchor (its timer is not the core's), and a
second `rel` after the tail loop shows the core ran to its end (LOCK_0_REL x2). Memory-module CONFLICT_DM_BANK_0..3 and the core's MEMORY_STALL are traced.

Data memory is eight physical 16-byte-interleaved banks in four 16 KB logical banks; `ldbuf` (0x800) is
physical banks 0 / 1 and `far` (0x4800) 2 / 3. A 256-bit access is rounded down to 32 bytes, so its first
word is always on the even bank and its second on the odd one: a scalar load at +0 shares the vector's FIRST
bank (family a), at +16 its SECOND (family b).

Port facts: the `ldb` slot has no scalar load (only `vldb`, `paddb` and nops), so two scalar loads cannot
share a bundle; and every load reaches its bank at issue + 4, so loads of different bundles never meet.
Family c therefore uses the load pairs the ISA does allow (two 256-bit loads; a scalar load with a 128-bit
vector load, one bank each; two 128-bit loads). Who loses an emulated meeting is the arbiter's rotor:
`prime` contended cycles on the bank (load + store pairs, before `event #0`) move the priority, which is
how the "which port loses" axis is varied (`rotor_loser`).

    python3 tools/gen_loadport_probe.py build OUT_DIR [--names la0 ...]
    python3 tools/gen_loadport_probe.py run OUT_DIR TAG --backend hw|emu [--names ...]
    python3 tools/gen_loadport_probe.py table OUT_DIR RUN [RUN ...] [--names ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_lock_pulse_probe as lp                                    # noqa: E402
from gen_dma_lock_probe import (FETCH_LOCK_A, FETCH_LOCK_GO, FETCH_PAD_WRITES, FETCH_SCRATCH_ADDR,   # noqa: E402
                                FETCH_SETTLE_ITERS, FETCH_TAIL_ITERS, FETCH_ROW, PT_CORE, PT_MEM,
                                OWN, load_level_events)

N = "  nop"
LOCK_A, LOCK_GO = FETCH_LOCK_A, FETCH_LOCK_GO
PAD_WRITES = FETCH_PAD_WRITES
LDBUF_ADDR, FAR_ADDR = 0x0800, 0x4800
BUF_ADDR = {"ld": LDBUF_ADDR, "far": FAR_ADDR}
BUF_PTR = {"ld": "p0", "far": "p1"}
K_NOPS = 8                 # nops between the lock release and the meeting bundle
AFTER_NOPS = 4             # nops between the meeting and `event #1`
PRIME_GAP = 8              # nops after each priming bundle
CONFLICT_EVENTS = [f"CONFLICT_DM_BANK_{b}" for b in range(4)]
MEM_EVENTS = ["LOCK_0_REL", *CONFLICT_EVENTS]
CORE_EVENTS = ["INSTR_EVENT_0", "INSTR_EVENT_1", "MEMORY_STALL"]   # `event` reaches only INSTR_EVENT_0 / _1
CORE_GROUNDING = "PERF_CNT_2"
# A conflict this far before the release (memory-module timer) belongs to the priming meetings.
PRIME_SPLIT = -8

TWO_SCALAR_LOADS_IMPOSSIBLE = (
    "two scalar loads cannot meet: the `ldb` slot has no scalar load (only vldb, paddb, nop), so a bundle holds "
    "one scalar load, and loads of different bundles reach their banks at issue + 4, never in the same cycle")

# Rotor order of the emulator's arbiter (src/device/bank_arbiter.rs): requester ordinals, 7 in all.
ORDINAL = {"A": 0, "B": 1, "S": 2}
NUM_REQUESTERS = 7


# -- bank arithmetic ----------------------------------------------------------------------------------

def physical_bank(addr: int) -> int:
    """Compute-tile physical bank of a tile-local address (16 KB logical banks, 16-byte interleave)."""
    return 2 * ((addr >> 14) & 3) + ((addr >> 4) & 1)


def access_banks(addr: int, width: int) -> set[int]:
    """Banks an access of `width` bytes touches (rounded down to its width, one bank per 16-byte word)."""
    start = addr & ~(width - 1)
    return {physical_bank(w) for w in range(start, start + max(width, 1), 16)} if width >= 16 else {physical_bank(addr)}


def vector_banks_in_order(addr: int) -> tuple[int, int]:
    """(first, second) bank of a 256-bit access."""
    start = addr & ~31
    return physical_bank(start), physical_bank(start + 16)


# -- accesses and cases -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Access:
    port: str      # "A" (lda slot), "B" (ldb slot), "S" (st slot)
    kind: str      # scalar | vec256 | vec128 | store
    buf: str       # "ld" | "far"
    off: int

    @property
    def address(self) -> int:
        return BUF_ADDR[self.buf] + self.off

    @property
    def width(self) -> int:
        return {"scalar": 4, "store": 4, "vec256": 32, "vec128": 16}[self.kind]

    @property
    def is_load(self) -> bool:
        return self.kind != "store"

    def asm(self) -> str:
        p = BUF_PTR[self.buf]
        if self.kind == "scalar":
            return f"lda r26, [{p}, #{self.off}]"
        if self.kind == "store":
            return f"st r2, [{p}, #{self.off}]"
        if self.kind == "vec256":
            return f"vlda wl0, [{p}, #{self.off}]" if self.port == "A" else f"vldb wl4, [{p}, #{self.off}]"
        # 128-bit forms have no immediate index: [p0] on port A, [p2, dj0] (p2 = p0 + 32, dj0 = 0) on port B
        if self.port == "A":
            assert (self.buf, self.off) == ("ld", 0), "vlda.128 reads [p0]"
            return "vlda.128 wl2, [p0]"
        assert (self.buf, self.off) == ("ld", 32), "vldb.128 reads [p2] = p0 + 32"
        return "vldb.128 wl6, [p2, dj0]"


@dataclass(frozen=True)
class LoadPortCase:
    name: str
    family: str
    first: Access
    second: Access
    prime: int = 0         # contended load+store cycles on the meeting bank before `event #0`

    @property
    def meets(self) -> bool:
        return bool(meeting_banks(self))

    @property
    def ports(self) -> tuple[str, str]:
        return self.first.port, self.second.port


def meeting_banks(c: LoadPortCase) -> set[int]:
    return access_banks(c.first.address, c.first.width) & access_banks(c.second.address, c.second.width)


def plan() -> list[LoadPortCase]:
    sc = lambda off, buf="ld": Access("A", "scalar", buf, off)                  # noqa: E731
    v256 = lambda port, off, buf="ld": Access(port, "vec256", buf, off)         # noqa: E731
    st = lambda off, buf="ld": Access("S", "store", buf, off)                   # noqa: E731
    cases = [
        # a / b: scalar load + vector load on the vector's first / second bank; the primed twin (1) moves
        # the emulator's loser from the vector load to the scalar load
        LoadPortCase("la0", "a", sc(0), v256("B", 32)),
        LoadPortCase("la1", "a", sc(0), v256("B", 32), prime=1),
        LoadPortCase("lb0", "b", sc(16), v256("B", 32)),
        LoadPortCase("lb1", "b", sc(16), v256("B", 32), prime=1),
        # the scalar load inside the vector's own 16-byte word (same address as a half of the vector)
        LoadPortCase("lwa", "a", sc(32), v256("B", 32)),
        LoadPortCase("lwb", "b", sc(48), v256("B", 32)),
        # c: what the ISA allows instead of two scalar loads
        LoadPortCase("c1", "c", v256("A", 0), v256("B", 32)),
        LoadPortCase("c2", "c", sc(0), Access("B", "vec128", "ld", 32)),
        LoadPortCase("c3", "c", Access("A", "vec128", "ld", 0), Access("B", "vec128", "ld", 32)),
        # d: load against store (stamped on silicon: objectfifo_channel_turns), the second twin moves the loser
        LoadPortCase("d1_0", "d", sc(0), st(0)),
        LoadPortCase("d1_1", "d", sc(0), st(0), prime=1),
        LoadPortCase("d2_0", "d", v256("B", 32), st(0)),
        LoadPortCase("d2_2", "d", v256("B", 32), st(0), prime=2),
        LoadPortCase("d3_0", "d", v256("B", 32), st(16)),
        LoadPortCase("d3_2", "d", v256("B", 32), st(16), prime=2),
        # e: the same pairs on different banks
        LoadPortCase("e1", "e", sc(0, "far"), v256("B", 32)),
        LoadPortCase("e2", "e", sc(0), st(0, "far")),
        LoadPortCase("e3", "e", v256("B", 32), st(0, "far")),
        LoadPortCase("e4", "e", v256("A", 0), v256("B", 0, "far")),
    ]
    return cases


# -- kernel text --------------------------------------------------------------------------------------

def meeting_bundle(c: LoadPortCase) -> str:
    return "  " + " ; ".join(a.asm() for a in (c.first, c.second))


def _prime_offset(c: LoadPortCase) -> int:
    (bank,) = meeting_banks(c)
    assert bank in (0, 1), "priming uses ldbuf"
    return 16 * bank


def prime_bundle(c: LoadPortCase) -> str:
    off = _prime_offset(c)
    return f"  lda r27, [p0, #{off}] ; st r2, [p0, #{off}]"


def core_asm(c: LoadPortCase) -> str:
    """go, settle, [`prime` load+store bundles on the meeting bank], event #0, release lock A,
    K_NOPS nops, the meeting bundle, event #1, tail, a second release of lock A. p0 = ldbuf, p1 = far, p2 = p0 + 32, dj0 = 0,
    r2 = +1 (the stored value and the release), r3 = -1."""
    n = lp.nops
    prime = []
    if c.prime:
        for _ in range(c.prime):
            prime += [prime_bundle(c), *n(PRIME_GAP)]
    body = ["  mov r2, #1", "  mov r3, #-1", "  mov p2, p0", N, "  padda [p2], #32", N,
            "  movxm dj0, #0", *n(2),
            f"  acq #{OWN + LOCK_GO}, r3", *n(4),
            *lp.delay_loop(FETCH_SETTLE_ITERS, ".Lsettle"), *n(8),
            *prime,
            "  event #0", f"  rel #{OWN + LOCK_A}, r2", *n(K_NOPS),
            meeting_bundle(c), *n(AFTER_NOPS), "  event #1", *n(8),
            *lp.delay_loop(FETCH_TAIL_ITERS, ".Ltail"), f"  rel #{OWN + LOCK_A}, r2", *n(4)]
    return lp.frame(body)


def design_mlir() -> str:
    ld = ", ".join(str(0x0B000000 | j) for j in range(64))
    far = ", ".join(str(0x0C000000 | j) for j in range(16))
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(PAD_WRITES))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %ldbuf = aie.buffer(%t02) {{address = {LDBUF_ADDR} : i32, sym_name = "ldbuf", initial_value = dense<[{ld}]> : tensor<64xi32>}} : memref<64xi32>
    %far = aie.buffer(%t02) {{address = {FAR_ADDR} : i32, sym_name = "far", initial_value = dense<[{far}]> : tensor<16xi32>}} : memref<16xi32>
    %scratch = aie.buffer(%t02) {{address = {FETCH_SCRATCH_ADDR} : i32, sym_name = "scratch"}} : memref<4xi32>
    %lA = aie.lock(%t02, {LOCK_A}) {{init = 0 : i32, sym_name = "lA"}}
    %lGo = aie.lock(%t02, {LOCK_GO}) {{init = 0 : i32, sym_name = "lGo"}}
    func.func private @k_core(memref<64xi32>, memref<16xi32>) attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core(%ldbuf, %far) : (memref<64xi32>, memref<16xi32>) -> ()
      aie.end
    }}
    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
      aiex.set_lock(%lGo, %c_one)
{pad}
    }}
  }}
}}
"""


# -- the emulator's loser and what each reading predicts ----------------------------------------------

def rotor_loser(ports: tuple[str, str], rotor: int) -> str:
    """The port of `ports` that loses a contended bank when the bank's rotor stands at `rotor`: the arbiter
    serves the requester whose ordinal comes first at or after the rotor (7 requesters in all)."""
    winner = min(ports, key=lambda p: (ORDINAL[p] + NUM_REQUESTERS - rotor) % NUM_REQUESTERS)
    return next(p for p in ports if p != winner)


def predicted_losers(c: LoadPortCase) -> dict[int, str]:
    """Per meeting bank, the emulator's losing port. The rotor of a bank advances once per contended cycle
    on it; nothing but the priming meetings contends before the case's own, so rotor = prime."""
    return {b: rotor_loser(c.ports, c.prime % NUM_REQUESTERS) for b in sorted(meeting_banks(c))}


READINGS = ("R0", "R1", "R2", "R3")      # R0 = the emulator today (every contended bank stamps)


def _second_bank(acc: Access, bank: int) -> bool:
    return acc.kind == "vec256" and vector_banks_in_order(acc.address)[1] == bank


def predicted_stamps(c: LoadPortCase, reading: str) -> set[int]:
    """Banks that raise CONFLICT_DM_BANK at the meeting under a reading. A load against a store stamps
    under every reading (the positive control)."""
    shared = meeting_banks(c)
    load_load = c.first.is_load and c.second.is_load
    if reading == "R0" or not load_load:
        return set(shared)
    if reading == "R1":
        return set()
    if reading == "R2":
        return {b for b in shared if not any(_second_bank(a, b) for a in (c.first, c.second))}
    if reading == "R3":
        losers = predicted_losers(c)
        return {b for b in shared if losers[b] == "A"}
    raise ValueError(reading)


# -- decoding a run -----------------------------------------------------------------------------------

def _runs(stamps) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for t in sorted(set(stamps)):
        if out and t == out[-1][1] + 1:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def row(events) -> dict:
    """One run of tile (0,2). Core numbers are cycles after INSTR_EVENT_0 (core timer); conflicts are cycles
    after the first LOCK_0_REL (memory-module timer, another unit: the release follows `event #0` by about
    4 to 5 cycles). Conflicts before the release (PRIME_SPLIT) and stalls before INSTR_EVENT_0 are the priming's."""
    core: dict = {}
    mem: dict = {}
    for e in events:
        if e.row != FETCH_ROW:
            continue
        (mem if e.pkt_type == PT_MEM else core).setdefault(e.name, []).append(e.ts)
    rel_stamps = sorted(mem.get("LOCK_0_REL", []))
    ev0 = sorted(core.get("INSTR_EVENT_0", []))
    ev1 = sorted(core.get("INSTR_EVENT_1", []))
    r: dict = {"release": rel_stamps[0] if rel_stamps else None, "ev0": bool(ev0), "ev1": bool(ev1),
                              "window": ev1[0] - ev0[0] if ev0 and ev1 else None,
               "releases": len(rel_stamps), "memory_stalls": [], "prime_stalls": [], "conflicts": [],
               "prime_conflicts": []}
    if ev0:
        stalls = sorted(set(core.get("MEMORY_STALL", [])))
        r["memory_stalls"] = [t - ev0[0] for t in stalls if t >= ev0[0]]
        r["prime_stalls"] = [t - ev0[0] for t in stalls if t < ev0[0]]
    if r["release"] is not None:
        for name in CONFLICT_EVENTS:
            for t in sorted(set(mem.get(name, []))):
                (r["conflicts"] if t - r["release"] >= PRIME_SPLIT else r["prime_conflicts"]).append(
                    (name, t - r["release"]))
        r["conflicts"].sort(key=lambda c: (c[1], c[0]))
        r["prime_conflicts"].sort(key=lambda c: (c[1], c[0]))
    return r


def finish_missing(r: dict, c: LoadPortCase, halt: str | None) -> list[str]:
    """What a run lacks to count as finished: the emulator's halt reason (when known), the release anchor,
    the core's markers and its release after the tail loop (LOCK_0_REL x2)."""
    out = []
    if halt is not None and halt != "completed":
        out.append(f"halt_reason={halt}")
    if r.get("release") is None:
        out.append("LOCK_0_REL")
    for key, name in (("ev0", "INSTR_EVENT_0"), ("ev1", "INSTR_EVENT_1")):
        if not r.get(key):
            out.append(name)
    if r.get("release") is not None and r.get("releases") != 2:
        out.append(f"core end: LOCK_0_REL x{r.get('releases')} (want 2)")
    return out


def _fmt_runs(times) -> str:
    return ",".join(f"{a:+d}" if a == b else f"{a:+d}..{b:+d}" for a, b in _runs(times)) or "-"


def format_table(rows) -> str:
    """rows: (LoadPortCase, run, row dict with `check`)."""
    lines = ["case   run      stall(core, from ev0)  conflicts(mem, from LOCK_0_REL)  prime stalls / conflicts"
             "        window  emulator loser  finish"]
    for c, run, r in sorted(rows, key=lambda x: (x[0].name, x[1])):
        if r.get("release") is None and not r.get("ev0"):
            lines.append(f"{c.name:<6} {run:<8} no trace events")
            continue
        per_bank: dict[str, list[int]] = {}
        for name, t in r["conflicts"]:
            per_bank.setdefault(name[-1], []).append(t)
        conf = " ".join(f"B{b}@{_fmt_runs(ts)}" for b, ts in sorted(per_bank.items())) or "-"
        prime = (f"{_fmt_runs(r['prime_stalls'])} / "
                 f"{' '.join(f'{n[-1]}@{t:+d}' for n, t in r['prime_conflicts']) or '-'}") if c.prime else "n/a"
        losers = sorted(set(predicted_losers(c).values()))
        loser = ",".join(losers) if losers else "-"
        chk = "finish ok" if not r.get("check") else "finish MISSING " + "; ".join(r["check"])
        win = "?" if r.get("window") is None else str(r["window"])
        lines.append(f"{c.name:<6} {run:<8} stall {_fmt_runs(r['memory_stalls']):<14} {conf:<30} {prime:<28} "
                     f"{win:>6}  loser {loser:<8} {chk}")
    return "\n".join(lines)


# -- build / run / table ------------------------------------------------------------------------------

def inject_args() -> list[str]:
    return ["--memmod-sweep-events", ",".join(MEM_EVENTS), "--core-grounding", CORE_GROUNDING,
            "--core-sweep-events", ",".join(CORE_EVENTS)]


def build_case(c: LoadPortCase, out: Path) -> None:
    d = out / c.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(core_asm(c), d / "core.o", "k_core")
    (d / "base.mlir").write_text(design_mlir())
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *inject_args(),
           "--trace-config-out", "trace_config.json", "--config-test-name", c.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    (d / "case.json").write_text(json.dumps({"case": c.name, "family": c.family, "prime": c.prime,
                                             "bundle": meeting_bundle(c).strip()}) + "\n")


def selection(args) -> list[LoadPortCase]:
    return [c for c in plan() if not args.names or c.name in args.names]


def halt_reason(run_dir: Path) -> str | None:
    log = run_dir / "runner.log"
    m = re.search(r"halt_reason=(\w+)", log.read_text()) if log.exists() else None
    return m.group(1) if m else None


def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in selection(args):
        build_case(c, out)
        print("built", c.name)
    return 0


def cmd_run(args) -> int:
    out = Path(args.out_dir).resolve()
    for c in selection(args):
        if not (out / c.name / "aie.xclbin").exists():
            continue
        if not lp.run_case(out, c.name, args.tag, args.backend, output=False):
            print(f"{c.name}: run did not complete; stop")
            return 1
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows = []
    for c in selection(args):
        for run in args.runs:
            run_dir = out / c.name / run
            if not run_dir.is_dir():
                continue
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                rows.append((c, run, {"release": None, "check": ["no trace"]}))
                continue
            r = row(load_level_events(run_dir))
            r["check"] = finish_missing(r, c, halt_reason(run_dir) if run.endswith(".emu") else None)
            rows.append((c, run, r))
    print(format_table(rows))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--names", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--names", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+")
    t.add_argument("--names", nargs="*", default=[])
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
