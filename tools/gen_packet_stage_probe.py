#!/usr/bin/env python3
"""Packet stalls by stage: which switch holds a packet stream, and when it forgets.

Model under test. Slots count words from the packet header (slot 0); k is the number of hops from the
source switch (k = 0 is the source's own switch).
  S_k  slave side of switch k, slot 8k: the quiet-switch hold. No word crossed the switch for its abort
       period (128 cycles), so it holds the first word; k = 0 is the source port refusing its first offer
       (MM2S backpressure one cycle before the first word). Not at shim switches.
  P_k  packet stage of switch k, slot 8k + 4: every switch that routes the stream through packet rules,
       every packet, unconditionally (so far).
So a header-plus-16-word packet from (0,2) to (0,3) shows stalls at slots 4 (P_0), 8 (S_1) and 12 (P_1):
`r4 g1 r4 g1 r4 g1 r5` at the destination-bound slave port, with source backpressure at slots -1 4 8 12.

Open questions, one family each (case names in brackets):
  T  [t008 ... t1000] A then B through the same masters, N idle cycles between A's last word and B's header
     at the shared master. Does S_1 come back above the 128-cycle threshold, and does P_1 stay at every N?
  H  [hdrop, hkeep] the packet header kept against dropped at the destination DMA master: same stalls?
  W  [wp*, wc*] a packet W through another arbiter / master of the source switch (wp) or a circuit stream
     through it (wc), then A, N cycles after W's last word: does any crossing of the source switch remove
     A's S_0 (a switch-level idle detector), and does P_0 stay?
  C  [c] a circuit control of A's BD: r8 g1 r8 (S_1 only, no P), and the payload's first-word slot against
     the packet stream's.

Design (device npu1_1col). Shim (0,0); memtile (0,1) only in the W family (its sink); source (0,2);
destination (0,3). Only (0,2) is traced, so nothing but the probe's data crosses switch (0,3).
  A  MM2S 0 of (0,2), packet id 1, 16 words, to S2MM 0 of (0,3).     B  MM2S 1, id 2, 16 words, same sink.
  W  MM2S 1 of (0,2), id 3 (or a circuit), 16 words, to S2MM 0 of the memtile.
  Every source BD acquires its own lock (init 0) and releases a done lock nothing waits on. Each S2MM has
  one lock-free BD of exactly the words that reach it (t: 32, hdrop: 16, hkeep: 17, wp/wc: 16, c: 16).
  The core of (0,2) starts before the trace is armed, so it first waits on a go lock that the runtime
  sequence sets after the trace is armed, settles, raises `event #0`, releases the first stream's lock,
  waits a known number of cycles and releases the second stream's lock. The core of (0,3) is empty.

Why every case ends, whatever silicon does. Every lock a DMA or the core waits on is released by something
that always runs: the go lock by the runtime sequence, the stream locks by the core once it holds go. Every
S2MM BD has room for exactly the words sent to it and nothing waits for it to finish (a stream that is held
longer, or a header that is not kept, only leaves a BD unfinished). The runtime sequence holds the trace
setup, the go-lock write and plain padding writes, waits on nothing and ends.

Why the trace does not disturb the measurement. `check_routes` reads the lowered switchboxes: the trace
packets use neither the data masters nor their arbiters, A and B share one master and one arbiter, W has
its own, and no rule of the memtile collides with a trace packet id.

    python3 tools/gen_packet_stage_probe.py build OUT_DIR [--cases t112 hdrop ...]
    python3 tools/gen_packet_stage_probe.py run OUT_DIR TAG --backend hw|emu [--cases ...]
    python3 tools/gen_packet_stage_probe.py table OUT_DIR RUN [RUN ...] [--cases ...]
    python3 tools/gen_packet_stage_probe.py session      # the hardware order, repeats included

The emulator backend and library are the caller's choice (environment). `run --backend hw` removes every
emulator variable from the runner's environment.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_dma_lock_probe as dl                                     # noqa: E402
import gen_lock_pulse_probe as lp                                   # noqa: E402

# -- constants ---------------------------------------------------------------------------

LOCK_A, LOCK_X1, LOCK_GO, LOCK_DONE = 0, 1, 2, 3      # LOCK_X1: the lock of whatever MM2S 1 sends (B or W)
OWN = dl.OWN
WORDS = 16                         # payload words of every stream
HEADER_WORDS = 1
PKT_A, PKT_B, PKT_W = 1, 2, 3
SETTLE_ITERS = 40
TAIL_ITERS = 100                   # keeps the core running so PERF_CNT_2 fills its last trace packet
# A trace unit sends a packet only when it holds seven words, and the shim writes the trace buffer in whole
# bursts, so the events of the last stream sit in unsent packets unless many more frames follow. After the
# streams are over the core therefore pulses INSTR_EVENT_0 (core unit) and releases lock A (memory unit's
# LOCK_0_REL) FLUSH_PULSES times. The pulses come after every measured stream; the lock count they build
# (at most FLUSH_PULSES + 1) stays below the lock's 63 and nothing acquires it again.
FLUSH_WAIT_ITERS = 40              # about 10 cycles each: lets the second stream finish first
FLUSH_PULSES = 48
PAD_WRITES = dl.FETCH_PAD_WRITES
ADDR_A, ADDR_X1, ADDR_DST, ADDR_SCRATCH, ADDR_SINK = 0x0400, 0x4400, 0x8000, 0xC000, 0x0400
ROW, DST_ROW, MEM_ROW, PT_CORE, PT_MEM = 2, 3, 1, 0, 1
THRESHOLD = 128                    # the quiet-switch abort period, cycles
EDGE = 6                           # a count this close to the threshold cannot be called either way
STREAM_CYCLES = 20                 # first word to last word of a header + 16 words with three one-cycle stalls
# Cross-module timer alignment, memory-module stamp plus this = core-module stamp, per backend. Silicon: the
# earlier packet-gap runs put every MM2S backpressure pulse one cycle before the stall it reports and the
# first pulse two cycles before the first word, so +1 makes the pulses sit on the stall cycles and at -1.
# Emulator: its first-word refusal pulse is recorded one cycle before the first word with no shift.
MEM_TO_CORE = {"hw": 1, "emu": 0}
MEMMOD_EVENTS = ["DMA_MM2S_0_STREAM_BACKPRESSURE", "DMA_MM2S_1_STREAM_BACKPRESSURE",
                 "DMA_MM2S_0_FINISHED_BD", "DMA_MM2S_1_FINISHED_BD",
                 "DMA_MM2S_0_STALLED_LOCK", "DMA_MM2S_1_STALLED_LOCK", "LOCK_0_REL"]
CORE_GROUNDING = "PERF_CNT_2,INSTR_EVENT_0"
# What the router picked in the first builds. `build` reads the real masters from the lowered MLIR and
# rebuilds once with them when they differ, so these only save the second pass.
GUESS_SHARED = ("North", 5)
GUESS_W = ("South", 0)
# Delay between the two lock releases: nops below this many, a counted loop above it.
NOP_DELAY_MAX = 39
LOOP_BASE = 5                      # cycles of the loop code with zero iterations (set-up, padding, drain)
LOOP_ITER = 9                      # cycles per loop iteration


# -- cases --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Stream:
    name: str                      # A | B | W
    channel: int                   # MM2S channel of (0,2)
    pkt_id: int | None             # None: a circuit
    base: int                      # first data word (ramp)
    lock: int


A_STREAM = Stream("A", 0, PKT_A, 0x0A000000, LOCK_A)
B_STREAM = Stream("B", 1, PKT_B, 0x0B000000, LOCK_X1)
W_STREAM = Stream("W", 1, PKT_W, 0x0C000000, LOCK_X1)
STREAMS = {"A": A_STREAM, "B": B_STREAM, "W": W_STREAM}

T_TARGETS = (8, 64, 112, 124, 132, 144, 300, 1000)
WP_TARGETS = (16, 120, 136, 1000)
WC_TARGETS = (16, 136, 1000)
REPEATED = ("t112", "t144", "hkeep", "wp136")


@dataclass(frozen=True)
class Case:
    family: str                    # T | H | W | C
    variant: str = ""              # H: drop | keep; W: pkt | circ
    n: int | None = None           # target idle cycles N (T, W)

    @property
    def name(self) -> str:
        if self.family == "T":
            return f"t{self.n:03d}"
        if self.family == "H":
            return f"h{self.variant}"
        if self.family == "W":
            return f"w{self.variant[0]}{self.n:03d}"
        return "c"

    @property
    def order(self) -> list[str]:
        """Streams in the order the core releases them."""
        return {"T": ["A", "B"], "H": ["A"], "W": ["W", "A"], "C": ["A"]}[self.family]

    def stream(self, name: str) -> Stream:
        s = STREAMS[name]
        if self.family == "C" or (self.family == "W" and name == "W" and self.variant == "circ"):
            return Stream(s.name, s.channel, None, s.base, s.lock)
        return s

    @property
    def streams(self) -> list[Stream]:
        return [self.stream(n) for n in self.order]

    @property
    def dst_words(self) -> int:
        """Words that reach S2MM 0 of (0,3)."""
        if self.family == "T":
            return 2 * WORDS
        if self.family == "H" and self.variant == "keep":
            return WORDS + HEADER_WORDS
        return WORDS

    @property
    def keep_header(self) -> bool:
        return self.family == "H" and self.variant == "keep"

    @property
    def has_w(self) -> bool:
        return self.family == "W"

    @property
    def release_spacing(self) -> int | None:
        """Cycles from the first stream's lock release to the second's, for the target N."""
        return None if self.n is None else self.n + STREAM_CYCLES


def default_cases() -> list[Case]:
    return ([Case("T", n=n) for n in T_TARGETS] + [Case("H", "drop"), Case("H", "keep")]
            + [Case("W", "pkt", n) for n in WP_TARGETS] + [Case("W", "circ", n) for n in WC_TARGETS]
            + [Case("C")])


def case_from_name(name: str) -> Case:
    for c in default_cases():
        if c.name == name:
            return c
    raise ValueError(f"unknown case {name!r}")


def session_order() -> list[str]:
    """The hardware run order: controls, then the two cases next to the threshold, then the rest, each
    repeated case run twice in a row. One run at a time; the first run that does not complete ends it."""
    first = ["hdrop", "c", "t112", "t144"]
    rest = [c.name for c in default_cases() if c.name not in first]
    order = []
    for name in first + rest:
        order += [name] * (2 if name in REPEATED else 1)
    return order


# -- the core program ----------------------------------------------------------------------

def release_gap(spacing: int) -> list[str]:
    """Instructions between two `rel`s that put them `spacing` cycles apart: nops, or a counted loop with
    nops for the remainder when the spacing is long."""
    if spacing < 1:
        raise ValueError("spacing must be at least one cycle")
    nops_needed = spacing - 1
    if nops_needed <= NOP_DELAY_MAX:
        return lp.nops(nops_needed)
    iters, rest = divmod(nops_needed - LOOP_BASE, LOOP_ITER)
    return [*lp.delay_loop(max(iters, 1), ".Lgap"), *lp.nops(rest)]


def flush_loop() -> list[str]:
    """FLUSH_PULSES rounds of: a core event, a release of lock A (a memory-module event)."""
    return [f"  movxm r0, #{FLUSH_PULSES}", *lp.nops(2), ".p2align 4", ".Lflush:", "  add r0, r0, #-1",
            "  event #0", f"  rel #{OWN + LOCK_A}, r2", *lp.nops(1), "  jnz r0, #.Lflush", *lp.nops(5)]


def core_asm(case: Case) -> str:
    """go, settle, `event #0`, release the first stream, wait, release the second. The first release sits
    on a fetch-line boundary so the loop's alignment padding, and with it LOOP_BASE, never varies."""
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2),
            f"  acq #{OWN + LOCK_GO}, r3", *n(4),
            *lp.delay_loop(SETTLE_ITERS, ".Lsettle"), *n(8), "  event #0", ".p2align 4"]
    first, *more = case.streams
    body += [f"  rel #{OWN + first.lock}, r2"]
    if more:
        body += [*release_gap(case.release_spacing), f"  rel #{OWN + more[0].lock}, r2"]
    body += [*n(8), *lp.delay_loop(FLUSH_WAIT_ITERS, ".Lwait"), *flush_loop(), *lp.delay_loop(TAIL_ITERS, ".Ltail")]
    return lp.frame(body)


# -- the design ------------------------------------------------------------------------------

def _ramp(base: int, words: int) -> str:
    return ", ".join(str(base | j) for j in range(words))


def _packet_flow(s: Stream, dest: str, keep: bool) -> str:
    attr = " {keep_pkt_header = true}" if keep else ""
    return f"""    aie.packet_flow({s.pkt_id}) {{
      aie.packet_source<%t02, DMA : {s.channel}>
      aie.packet_dest<{dest}, DMA : 0>
    }}{attr}
"""


def design_mlir(case: Case) -> str:
    streams = {s.name: s for s in case.streams}
    flows, buffers, bds = "", "", {0: "", 1: ""}
    for s in case.streams:
        buf = f"buf{s.name}"
        addr = ADDR_A if s.channel == 0 else ADDR_X1
        buffers += (f"    %{buf} = aie.buffer(%t02) {{address = {addr} : i32, sym_name = \"{buf}\", "
                    f"initial_value = dense<[{_ramp(s.base, WORDS)}]> : tensor<{WORDS}xi32>}} : "
                    f"memref<{WORDS}xi32>\n")
        pkt = "" if s.pkt_id is None else f" {{packet = #aie.packet_info<pkt_type = 0, pkt_id = {s.pkt_id}>}}"
        bds[s.channel] = f"""      %s{s.channel} = aie.dma_start(MM2S, {s.channel}, ^bd{s.channel}, ^end)
    ^bd{s.channel}:
      %c_{s.name} = arith.constant 1 : i32
      aie.use_lock(%l{s.name}, AcquireGreaterEqual, %c_{s.name})
      aie.dma_bd(%{buf} : memref<{WORDS}xi32>){pkt}
      %d_{s.name} = arith.constant 1 : i32
      aie.use_lock(%lDone, Release, %d_{s.name})
      aie.next_bd ^end
"""
    # `mem` regions need the first dma_start to chain to the second channel's start, not to ^end.
    mm2s = ""
    chans = [c for c in (0, 1) if bds[c]]
    for i, c in enumerate(chans):
        text = bds[c]
        if i + 1 < len(chans):
            text = text.replace(f"^bd{c}, ^end)", f"^bd{c}, ^ch{chans[i + 1]})")
        mm2s += text
        if i + 1 < len(chans):
            mm2s += f"    ^ch{chans[i + 1]}:\n"
    for s in case.streams:
        if s.name == "W":
            if s.pkt_id is None:
                flows += "    aie.flow(%t02, DMA : 1, %t01, DMA : 0)\n"
            else:
                flows += _packet_flow(s, "%t01", False)
        elif case.family == "C":
            flows += "    aie.flow(%t02, DMA : 0, %t03, DMA : 0)\n"
        else:
            flows += _packet_flow(s, "%t03", case.keep_header)
    locks = "".join(f"    %l{n} = aie.lock(%t02, {i}) {{init = 0 : i32, sym_name = \"l{n}\"}}\n"
                    for n, i in (("A", LOCK_A), ("B", LOCK_X1), ("W", LOCK_X1), ("Go", LOCK_GO),
                                 ("Done", LOCK_DONE)) if n in streams or n in ("Go", "Done"))
    memtile = ""
    if case.has_w:
        memtile = f"""    %sink = aie.buffer(%t01) {{address = {ADDR_SINK} : i32, sym_name = "sink"}} : memref<{WORDS}xi32>
    %mt = aie.memtile_dma(%t01) {{
      %m0 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.dma_bd(%sink : memref<{WORDS}xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""
    wd = case.dst_words
    tile01 = "    %t01 = aie.tile(0, 1)\n" if case.has_w else ""
    pad = "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32"
                    for _ in range(PAD_WRITES))
    return f"""module {{
  aie.device(npu1_1col) {{
    %t00 = aie.tile(0, 0)
{tile01}    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
{buffers}    %dst = aie.buffer(%t03) {{address = {ADDR_DST} : i32, sym_name = "dst"}} : memref<{wd}xi32>
    %scratch = aie.buffer(%t02) {{address = {ADDR_SCRATCH} : i32, sym_name = "scratch"}} : memref<4xi32>
{locks}{flows}    func.func private @k_core() attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core() : () -> ()
      aie.end
    }}
    aie.core(%t03) {{
      aie.end
    }}
    %mem = aie.mem(%t02) {{
{mm2s}    ^end:
      aie.end
    }}
    %mem3 = aie.mem(%t03) {{
      %s2 = aie.dma_start(S2MM, 0, ^recv, ^end)
    ^recv:
      aie.dma_bd(%dst : memref<{wd}xi32>)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
{memtile}    aie.runtime_sequence(%dummy: memref<4xi32>) {{
      %c_zero = arith.constant 0 : i32
      %c_one = arith.constant 1 : i32
      aiex.set_lock(%lGo, %c_one)
{pad}
    }}
  }}
}}
"""


def port_slots(case: Case, shared: tuple[str, int], w: tuple[str, int] | None) -> str:
    """The trace unit's stream-switch event ports: 0 the shared master, 1 and 2 the two DMA slaves, 3 W's master."""
    slots = f"0={shared[0]}:{shared[1]}:master,1=DMA:0:slave,2=DMA:1:slave"
    if case.has_w and w is not None:
        slots += f",3={w[0]}:{w[1]}:master"
    return slots


def core_events(case: Case) -> list[str]:
    return ["PORT_RUNNING_0", "PORT_RUNNING_1", "PORT_RUNNING_2", "PORT_TLAST_0"] + (
        ["PORT_RUNNING_3"] if case.has_w else [])


def inject_args(case: Case, shared: tuple[str, int], w: tuple[str, int] | None) -> list[str]:
    return ["--trace-tiles", f"0:{ROW}", "--memmod-sweep-events", ",".join(MEMMOD_EVENTS),
            "--core-grounding", CORE_GROUNDING, "--core-sweep-events", ",".join(core_events(case)),
            "--port-slots", port_slots(case, shared, w)]


# -- route checks on the lowered MLIR -------------------------------------------------------------

_AMSEL = re.compile(r"(%\w+)\s*=\s*aie\.amsel<(\d+)>\s*\((\d+)\)")
_MASTERSET = re.compile(r"aie\.masterset\((\w+)\s*:\s*(\d+)((?:\s*,\s*%\w+)+)\)\s*(\{[^}]*\})?")
_RULES = re.compile(r"aie\.packet_rules\((\w+)\s*:\s*(\d+)\)\s*\{(.*?)\}", re.S)
_RULE = re.compile(r"aie\.rule\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(%\w+)\)")
_CONNECT = re.compile(r"aie\.connect<(\w+)\s*:\s*(\d+)\s*,\s*(\w+)\s*:\s*(\d+)>")
Port = tuple[str, int]


@dataclass
class Switchbox:
    arbiter: dict[str, int]                              # amsel name -> arbiter number
    masters: list[tuple[Port, tuple[str, ...], bool]]    # (master port, amsels, keep_pkt_header)
    rules: dict[Port, list[tuple[int, int, str]]]        # slave port -> (mask, id, amsel name)
    connects: list[tuple[Port, Port]]                    # circuit connections slave -> master

    def route(self, slave: Port, pkt_id: int) -> list[tuple[Port, int, bool]]:
        """(master, arbiter, keep) for every master a packet of this id leaves through."""
        out = []
        for mask, rid, name in self.rules.get(slave, []):
            if pkt_id & mask != rid & mask:
                continue
            for master, names, keep in self.masters:
                if name in names:
                    out.append((master, self.arbiter[name], keep))
        return out

    def circuit_masters(self, slave: Port) -> list[Port]:
        return [m for s, m in self.connects if s == slave]


def switchbox(mlir: str, tile: tuple[int, int]) -> Switchbox:
    col, row = tile
    m = re.search(rf"aie\.switchbox\(%(?:mem_)?tile_{col}_{row}\)\s*\{{(.*?)\n    \}}", mlir, re.S)
    if not m:
        raise ValueError(f"no switchbox for tile {tile}")
    box = m.group(1)
    arbiter = {name: int(a) for name, a, _ in _AMSEL.findall(box)}
    masters = [((b, int(ch)), tuple(re.findall(r"%\w+", names)), "keep_pkt_header = true" in (attr or ""))
               for b, ch, names, attr in _MASTERSET.findall(box)]
    rules = {(b, int(ch)): [(int(mk), int(i), n) for mk, i, n in _RULE.findall(body)]
             for b, ch, body in _RULES.findall(box)}
    connects = [((sb, int(sc)), (mb, int(mc))) for sb, sc, mb, mc in _CONNECT.findall(box)]
    return Switchbox(arbiter, masters, rules, connects)


def trace_ports(box: Switchbox) -> tuple[set[Port], set[int]]:
    """Masters and arbiter numbers the tile's own trace packets use (rules on the Trace slaves)."""
    masters, arbs = set(), set()
    for slave, rules in box.rules.items():
        if slave[0] != "Trace":
            continue
        for _, _, name in rules:
            arbs.add(box.arbiter[name])
            masters.update(m for m, names, _ in box.masters if name in names)
    return masters, arbs


@dataclass(frozen=True)
class Routes:
    shared: Port                   # the master A (and B) leave (0,2) through
    w: Port | None                 # the master W leaves (0,2) through


def check_routes(mlir: str, case: Case) -> tuple[list[str], Routes | None]:
    """Problems with the routed switchboxes for this case (empty: the case measures what it says), and the
    masters the trace's event ports must watch."""
    bad: list[str] = []
    src, dst = switchbox(mlir, (0, ROW)), switchbox(mlir, (0, DST_ROW))
    t_masters, t_arbs = trace_ports(src)
    shared = w_master = None
    arb: dict[str, int] = {}
    for s in case.streams:
        slave = ("DMA", s.channel)
        if s.pkt_id is None:
            if src.rules.get(slave):
                bad.append(f"{s.name} is a circuit but (0,{ROW}) has packet rules on DMA:{s.channel}")
            ms = src.circuit_masters(slave)
            if len(ms) != 1:
                bad.append(f"{s.name}: circuit from DMA:{s.channel} leaves through {ms}, want one master")
                continue
            master, a = ms[0], None
        else:
            routes = src.route(slave, s.pkt_id)
            if not src.rules.get(slave) or not routes:
                bad.append(f"{s.name}: no packet rule for id {s.pkt_id} on DMA:{s.channel} of (0,{ROW}) "
                           "(the router made it a circuit)")
                continue
            if len(routes) != 1:
                bad.append(f"{s.name}: id {s.pkt_id} leaves through {len(routes)} masters/arbiters")
                continue
            master, a, _ = routes[0]
            if a in t_arbs:
                bad.append(f"{s.name}: arbiter {a} is also a trace rule's")
        if master in t_masters:
            bad.append(f"{s.name}: master {master} also carries trace packets")
        arb[s.name] = a
        if s.name == "W":
            w_master = master
        else:
            shared = shared or master
            if shared != master:
                bad.append(f"{s.name} leaves through {master}, A through {shared}")
    if "A" in arb and "B" in arb and arb["A"] != arb["B"]:
        bad.append(f"A and B use arbiters {arb['A']} and {arb['B']}, not one")
    if "W" in arb and "A" in arb:
        if w_master == shared:
            bad.append(f"W and A share master {w_master}")
        if arb["W"] is not None and arb["W"] == arb["A"]:
            bad.append(f"W and A share arbiter {arb['A']}")
    bad += _check_destination(dst, case)
    if case.has_w:
        bad += _check_memtile(mlir, case, w_master)
    return bad, (Routes(shared, w_master) if shared and not bad else None)


def _check_destination(dst: Switchbox, case: Case) -> list[str]:
    bad = []
    dma0 = [(m, names, keep) for m, names, keep in dst.masters if m == ("DMA", 0)]
    if case.family == "C":
        if not any(m == ("DMA", 0) for _, m in dst.connects):
            bad.append("(0,3): no circuit connection into DMA:0")
        return bad
    if len(dma0) != 1:
        return [f"(0,3): {len(dma0)} DMA:0 master sets, want 1"]
    _, names, keep = dma0[0]
    if len({dst.arbiter[n] for n in names}) != 1:
        bad.append("(0,3): DMA:0 master fed through more than one arbiter")
    if keep != case.keep_header:
        bad.append(f"(0,3): DMA:0 keep_pkt_header is {keep}, want {case.keep_header}")
    return bad


def _check_memtile(mlir: str, case: Case, w_master: Port | None) -> list[str]:
    """W's sink: the memtile switch sends W's id to DMA:0 and no other rule of the slave port claims the id."""
    mt = switchbox(mlir, (0, MEM_ROW))
    if w_master is None:
        return []
    slave = ("North", w_master[1])
    if case.stream("W").pkt_id is None:
        return [] if any(m == ("DMA", 0) for _, m in mt.connects) else ["(0,1): no circuit into DMA:0"]
    hits = [(mask, rid, n) for mask, rid, n in mt.rules.get(slave, []) if PKT_W & mask == rid & mask]
    if len(hits) != 1:
        return [f"(0,1): {len(hits)} rules of {slave} match id {PKT_W}, want 1 (trace id collision?)"]
    ends = [m for m, names, _ in mt.masters if hits[0][2] in names]
    return [] if ends == [("DMA", 0)] else [f"(0,1): W's rule leads to {ends}, want DMA:0"]


# -- prediction ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    s0: bool
    stalls: tuple[int, ...]
    bp: tuple[int, ...]
    edge: bool = False             # N within EDGE of the threshold: either answer is consistent
    note: str = ""


def predict(case: Case, name: str, n: int | None) -> Prediction:
    """What the stage model gives silicon for stream `name` of `case`, N idle cycles after the first stream."""
    if case.stream(name).pkt_id is None:
        return Prediction(True, (8,), (-1, 8), note="circuit: S_1 only"
                          + ("; the memtile switch is on the trace route" if name == "W" else ""))
    first = case.order[0] == name
    p = (4, 12)
    if first or n is None:
        return Prediction(True, (4, 8, 12), (-1, 4, 8, 12),
                          note="W's S_1 is at the memtile switch, which the trace route crosses" if name == "W" else "")
    quiet = n > THRESHOLD
    edge = abs(n - THRESHOLD) <= EDGE
    if case.family == "T":                       # B: its own switch (0,2) and (0,3) both saw A
        stalls = tuple(sorted(p + ((8,) if quiet else ())))
    else:                                        # A after W: (0,3) never carried anything
        stalls = (4, 8, 12)
    return Prediction(quiet, stalls, ((-1,) if quiet else ()) + stalls, edge)


# -- the table --------------------------------------------------------------------------------------

def _stamps(events, pkt: int, name: str, row: int = ROW) -> list[int]:
    return sorted({e.ts for e in events if e.row == row and e.pkt_type == pkt and e.name == name})


def seq_text(stamps: list[int]) -> str:
    """`r4 g1 r4 ...`: runs of consecutive cycles and the idle gaps between them."""
    runs = dl._runs(stamps)
    if not runs:
        return "-"
    parts = [f"r{runs[0][1] - runs[0][0] + 1}"]
    for (_, e), (s, f) in zip(runs, runs[1:]):
        parts += [f"g{s - e - 1}", f"r{f - s + 1}"]
    return " ".join(parts)


def stall_slots(stamps: list[int]) -> list[tuple[int, int]]:
    """(slot, length) of every gap in the word run: the slot is the number of words before it."""
    out, words = [], 0
    runs = dl._runs(stamps)
    for i, (s, e) in enumerate(runs):
        words += e - s + 1
        if i + 1 < len(runs):
            out.append((words, runs[i + 1][0] - e - 1))
    return out


def pulse_slots(pulse_starts: list[int], words: list[int]) -> list[int]:
    """Slot of each backpressure pulse: the words that passed before it, -1 for a pulse before the first word."""
    first = words[0]
    return [-1 if t < first else sum(1 for w in words if w < t) for t in pulse_starts]


def stream_row(events, case: Case, name: str, backend: str = "hw") -> dict:
    """One stream as numbers, all on the core timer (the memory module's stamps shifted by MEM_TO_CORE[backend])."""
    s = case.stream(name)
    slave = _stamps(events, PT_CORE, f"PORT_RUNNING_{1 + s.channel}")
    want = WORDS + (0 if s.pkt_id is None else HEADER_WORDS)
    row = {"stream": name, "missing": [], "words": len(slave), "want": want}
    if len(slave) != want:
        row["missing"].append(f"slave port has {len(slave)} word cycles, expected {want}")
        row["first"] = slave[0] if slave else None
        return row
    ch = s.channel
    row["first"] = slave[0]
    row["seq"] = seq_text(slave)
    row["stalls"] = stall_slots(slave)
    bp = [a + MEM_TO_CORE[backend] for a, _ in dl._runs(_stamps(events, PT_MEM, f"DMA_MM2S_{ch}_STREAM_BACKPRESSURE"))]
    row["bp"] = pulse_slots(bp, slave)
    # The unit is configured for the event, so a stream with no pulse shows none; the same unit's
    # FINISHED_BD proves its packets arrived.
    row["bp_seen"] = any(e.name == f"DMA_MM2S_{ch}_FINISHED_BD" for e in events if e.row == ROW)
    locks = dl._runs(_stamps(events, PT_MEM, f"DMA_MM2S_{ch}_STALLED_LOCK"))
    row["grant"] = None if not locks else locks[-1][1] + 1 + MEM_TO_CORE[backend]
    row["lock_to_word"] = None if row["grant"] is None else slave[0] - row["grant"]
    fin = _stamps(events, PT_MEM, f"DMA_MM2S_{ch}_FINISHED_BD")
    row["finish_after_last"] = (fin[0] + MEM_TO_CORE[backend] - slave[-1]) if fin else None
    return row


def case_row(events, case: Case, backend: str = "hw") -> dict:
    """One run: each stream's row, the shared master's sequence and N."""
    rows = {n: stream_row(events, case, n, backend) for n in case.order}
    master = _stamps(events, PT_CORE, "PORT_RUNNING_0")
    wm = _stamps(events, PT_CORE, "PORT_RUNNING_3")
    tlast = _stamps(events, PT_CORE, "PORT_TLAST_0")
    out = {"streams": rows, "master": seq_text(master) if master else "-",
           "w_master": seq_text(wm) if wm else None, "n": None, "notes": [], "tlast_ok": None}
    if case.family == "T":
        want = 2 * (WORDS + HEADER_WORDS)
        if len(master) == want:
            end_a = WORDS + HEADER_WORDS - 1
            out["n"] = master[end_a + 1] - master[end_a] - 1
            out["tlast_ok"] = master[end_a] in tlast if tlast else None
        else:
            out["notes"].append(f"shared master has {len(master)} word cycles, expected {want}")
    elif case.family == "W":
        w_words = len(wm)
        want_w = WORDS + (0 if case.stream("W").pkt_id is None else HEADER_WORDS)
        a_first = rows["A"].get("first")
        if w_words == want_w and a_first is not None:
            out["n"] = a_first - wm[-1] - 1
        else:
            out["notes"].append(f"W's master has {w_words} word cycles, expected {want_w}")
    g = [rows[n].get("grant") for n in case.order]
    if len(g) == 2 and None not in g:
        out["release_spacing"] = g[1] - g[0]
    return out


def judge(case: Case, name: str, row: dict, n: int | None) -> tuple[Prediction | None, str]:
    """The prediction for a stream and whether its stall slots (and backpressure at -1) agree."""
    if "stalls" not in row:
        return None, "-"
    pred = predict(case, name, n)
    got = tuple(slot for slot, _ in row["stalls"])
    ok = got == pred.stalls and all(length == 1 for _, length in row["stalls"])
    if row.get("bp_seen"):
        ok = ok and ((-1 in row["bp"]) == pred.s0)
    return pred, ("ok" if ok else "DIFF") + ("~" if pred.edge else "")


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def format_table(rows) -> str:
    """rows: (case, run name, case_row dict)."""
    head = (f"{'case':<7} {'run':<8} {'strm':<4} {'N':>5} {'D':>5} {'first':>6} {'l>w1':>4} {'fin':>4}  "
            f"{'slave port sequence':<26} {'bp slots':<14} {'predicted stalls':<17} {'pred bp':<14} match\n"
            "  N: idle cycles between the streams (T: A's last word to B's header at the shared master; W: W's\n"
            "  last word at its master to A's first word at its slave); D: cycles between the two lock grants;\n"
            "  l>w1: lock grant to first word; fin: FINISHED_BD minus the last word. Backpressure slot -1 is the\n"
            "  pulse before the first word. match: the stall slots (and the -1 pulse where captured) agree with the\n"
            "  model; '~' marks N within a few cycles of the threshold, so either answer fits.")
    lines = [head]
    for case, run, r in rows:
        for name in case.order:
            sr = r["streams"][name]
            pred, match = judge(case, name, sr, r["n"] if name == case.order[-1] else None)
            if pred is None:
                note = "missing: " + "; ".join(sr["missing"] or ["slave port events"])
                lines.append(f"{case.name:<7} {run:<8} {name:<4} {_fmt(r['n']):>5} {'-':>5} {_fmt(sr.get('first')):>6}"
                             f" {'-':>4} {'-':>4}  {note}")
                continue
            bp = " ".join(map(str, sr["bp"])) if sr["bp"] else ("none" if sr["bp_seen"] else "no event")
            ps = " ".join(map(str, pred.stalls))
            pb = " ".join(map(str, pred.bp)) or "none"
            lines.append(f"{case.name:<7} {run:<8} {name:<4} {_fmt(r['n'] if name == case.order[-1] else None):>5} "
                         f"{_fmt(r.get('release_spacing') if name == case.order[-1] else None):>5} "
                         f"{sr['first']:>6} {_fmt(sr['lock_to_word']):>4} {_fmt(sr['finish_after_last']):>4}  "
                         f"{sr['seq']:<26} {bp:<14} {ps:<17} {pb:<14} {match}")
        tail = [f"master {r['master']}"]
        if r["w_master"]:
            tail.append(f"W master {r['w_master']}")
        if r["tlast_ok"] is False:
            tail.append("TLAST not on A's last word")
        lines.append(f"{'':<7} {run:<8} " + "; ".join(tail + r["notes"]))
    return "\n".join(lines)


# -- build / run / table CLI -------------------------------------------------------------------------

def _inject_and_compile(case: Case, d: Path, shared: Port, w: Port | None) -> str:
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir",
           "--out", "aie.mlir", "--trace-mode", "event_time", *inject_args(case, shared, w),
           "--trace-config-out", "trace_config.json", "--config-test-name", case.name,
           "--config-src-mlir", "base.mlir", "--buffer-size", "65536"],
          cwd=d, log=d / "inject.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
          cwd=d, log=d / "aiecc.log")
    return (d / "input_with_addresses.mlir").read_text()


def build_case(case: Case, out: Path) -> None:
    """Assemble, inject the trace with the guessed masters, compile, read the real masters from the lowered
    MLIR and compile again with them when the guess was wrong; fail unless every route check passes."""
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(core_asm(case), d / "core.o", "k_core")
    (d / "base.mlir").write_text(design_mlir(case))
    shared, w = GUESS_SHARED, (GUESS_W if case.has_w else None)
    for attempt in range(2):
        lowered = _inject_and_compile(case, d, shared, w)
        problems, routes = check_routes(lowered, case)
        if problems:
            raise SystemExit(f"{case.name}: " + "; ".join(problems))
        if (routes.shared, routes.w) == (shared, w):
            break
        shared, w = routes.shared, routes.w
    else:
        raise SystemExit(f"{case.name}: routes keep changing between builds")
    (d / "case.json").write_text(json.dumps(
        {"case": case.name, "family": case.family, "variant": case.variant, "n": case.n,
         "shared_master": list(routes.shared), "w_master": None if routes.w is None else list(routes.w)}) + "\n")


def select(out: Path, names) -> list[Case]:
    cases = [case_from_name(n) for n in names] if names else default_cases()
    return [c for c in cases if (out / c.name).is_dir()]


def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for case in ([case_from_name(n) for n in args.cases] if args.cases else default_cases()):
        build_case(case, out)
        print("built", case.name, flush=True)
    return 0


EMU_VARS = ("XDNA_EMU", "XDNA_EMU_LIB", "XDNA_EMU_DIR", "XDNA_BACKEND", "XDNA_EMU_RUNTIME")


def run_env(backend: str) -> dict[str, str]:
    """Hardware: no emulator variable survives. Emulator: on, release runtime unless the caller chose one;
    which backend and library load is the caller's environment."""
    env = dict(os.environ)
    if backend == "hw":
        for k in EMU_VARS:
            env.pop(k, None)
    else:
        env["XDNA_EMU"] = "1"
        env.setdefault("XDNA_EMU_RUNTIME", "release")
    return env


def runner_command(d: Path, r: Path) -> list[str]:
    return [str(lp.RUNNER), "--xclbin", str(d / "aie.xclbin"), "--instr", str(d / "insts.bin"),
            "--trace-out", str(r / "trace_raw.bin"), "--trace-size", "65536", *lp.runner_extra_args()]


def run_case(out: Path, case: Case, tag: str, backend: str) -> bool:
    d = out / case.name
    r = d / f"{tag}.{backend}"
    r.mkdir(parents=True, exist_ok=True)
    lockfile = None
    if backend == "hw":
        lockfile = open(lp.HW_LOCK, "w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{case.name}: waiting for {lp.HW_LOCK} (do not wrap this tool in flock)", file=sys.stderr, flush=True)
            fcntl.flock(lockfile, fcntl.LOCK_EX)
    try:
        p = subprocess.run(runner_command(d, r), env=run_env(backend), capture_output=True, text=True,
                           timeout=120 if backend == "hw" else 1800)
    except subprocess.TimeoutExpired:
        print(f"{case.name} {backend}: runner timed out")
        return False
    finally:
        if lockfile:
            lockfile.close()
    (r / "runner.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        print(f"{case.name} {backend}: runner failed rc={p.returncode}")
        return False
    fault = next((ln for ln in p.stderr.splitlines() if ln.startswith(lp.EMU_HOST_FAULT_MARKER)), None)
    if backend != "hw" and fault:
        print(f"{case.name} {backend}: shim DMA access outside every host buffer: {fault}")
        return False
    lp.sh([sys.executable, str(REPO / "tools" / "parse-trace.py"), "--trace-bin", str(r / "trace_raw.bin"),
           "--xclbin-mlir", str(d / "input_with_addresses.mlir"),
           "--trace-mode", "event_time", "--out-events", str(r / "events.json")], log=r / "parse.log")
    return True


def cmd_run(args) -> int:
    """Stops at the first run that does not complete; never retries."""
    out = Path(args.out_dir).resolve()
    for case in select(out, args.cases):
        if not run_case(out, case, args.tag, args.backend):
            print(f"{case.name}: run did not complete; stop")
            return 1
        print(f"{case.name}: ok", flush=True)
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows, absent = [], {}
    for case in select(out, args.cases):
        for run in args.runs:
            run_dir = out / case.name / run
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                absent.setdefault(run, []).append(case.name)
                continue
            rows.append((case, run, case_row(dl.load_level_events(run_dir), case, run.rsplit(".", 1)[-1])))
    print(format_table(rows))
    for run, names in absent.items():
        print(f"not in {run}: {' '.join(names)}")
    return 0


def cmd_session(args) -> int:
    print("\n".join(session_order()))
    return 0


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
    r.add_argument("--cases", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    t.add_argument("--cases", nargs="*", default=[])
    sub.add_parser("session")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "table": cmd_table, "session": cmd_session}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
