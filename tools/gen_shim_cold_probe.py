#!/usr/bin/env python3
"""What selects and sizes the first shim S2MM write of a run, and the slow drain after a long idle.

A shim S2MM channel writes the words of its stream to host memory. Its FINISHED_TASK follows
max(first word + F, last word + 150): F is 165 when the write path is "warm" and 526..9923 when it is
"cold". In the recorded silicon runs a first task is cold exactly when no shim S2MM write has finished
earlier in the run, which cannot be told from "first touch of the destination buffer or page"; some later
one-word tasks are cold at fixed positions in a sequence; and after a very long idle (88k cycles and
more) the channel drains a 4096-word burst slowly. This probe separates the explanations.

Hypotheses (HP page/buffer translation, HR run-first, HI idle path, HC per channel, HE translation-cache
eviction) and the 68 cases (families A..F) are listed by `cases`; each case's prediction per hypothesis is
in `predict`.

Data path of every task: a memtile MM2S (the source, one BD of N words per task, lock-gated) -> shim S2MM
channel 0 of column 0 -> a host buffer. The compute core of column 0 releases the source's lock after a
counted delay, so the idle times are known to the cycle; the runtime sequence only configures and starts
the S2MM tasks (all queued up front, one BD per task) and sets the core's go lock. Column 1 holds the
trace carrier (the shim S2MM channel the trace packets drain through) and, in families C, D and E, the
background traffic: another shim S2MM channel writing, or a shim MM2S channel reading, a separate buffer.

Trace configurations. TRACE-Q ("quiet", the default): only the shim units, the memtile unit(s) and no
compute core are traced, so the carrier writes only when a traced event completes a packet. TRACE-N
("noisy", family F): the compute core also traces its periodic PERF_CNT_2 event, as the recorded runs do.
Traced events: shim S2MM 0 START_TASK / FINISHED_TASK / STREAM_STARVATION, S2MM 1 STREAM_STARVATION (the
carrier's packet arrivals), MM2S 0 START_TASK / FINISHED_TASK, a software event used only to flush; memtile
PORT_RUNNING of the DMA 0 slave (the source's word clock) and master.

Flush. A trace unit sends a packet only when it holds seven words, so the last events of a unit stay in an
unsent packet. After the core's schedule the runtime sequence therefore writes the software-event
generator of each traced unit FLUSH_PULSES times; the pulses are traced events and fill the packets.

Why every case ends, whatever silicon does (the per-case line is in `termination`). Every lock a DMA
or the core waits on is released by something that always runs: the go lock by the runtime sequence, the
source lock by the core once it holds go. Every S2MM BD has room for exactly the words the source sends
it and nothing waits for a task to finish: the runtime sequence holds task configuration, lock sets and
padding writes only. A slow or cold host write only makes an event later; it cannot keep anything from
ending. The trace carrier's S2MM BD is oversized and never awaited, as in every traced run.

    python3 tools/gen_shim_cold_probe.py cases
    python3 tools/gen_shim_cold_probe.py build OUT_DIR [--cases a16d0 ...] [--set priority|rest]
    python3 tools/gen_shim_cold_probe.py run OUT_DIR TAG --backend hw|emu [--cases ...]
    python3 tools/gen_shim_cold_probe.py table OUT_DIR RUN [RUN ...] [--cases ...]
    python3 tools/gen_shim_cold_probe.py bounds OUT_DIR [--cases ...] [--set priority|rest]
    python3 tools/gen_shim_cold_probe.py session [--set priority|rest]

The emulator backend and library are the caller's choice (environment). `run --backend hw` removes
every emulator variable from the runner's environment.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import gen_dma_lock_probe as dl                                     # noqa: E402
import gen_lock_pulse_probe as lp                                   # noqa: E402
import gen_packet_stage_probe as ps                                 # noqa: E402

# -- constants ---------------------------------------------------------------------------

OWN = dl.OWN                       # raw lock-id base of a core's own memory module
LOCK_SRC, LOCK_DONE, LOCK_TRIG = 0, 1, 2   # memtile locks: gates a source BD; released by it (nothing waits on it);
                                   # trigger credits (one per transfer)
LOCK_GO = 0                        # lock of the column-0 compute tile that the runtime sequence sets
BG_LOCK, BG_DONE, BG_GO = 0, 1, 2  # locks of the column-1 compute tile (background traffic)
TRIGGER = "  mov ms, r2"           # the core writes one word to its stream port: that is the release
ADDR_SCRATCH = 0xC000
WORD = 4
PAGE = 4096
SRC_ALIGN = 64

# Columns: 0 holds the channel under test; 1 holds the trace carrier and the background traffic.
CARRIER_COL = 1
BG_LOCK_PERIOD = 1000              # cycles between background transfers (longer when the repeat count would exceed the BD limit)
BG_MAX_ROUNDS = 256                # memtile task repeat count limit (255) + 1
BG_WORDS = 16                      # words of one background transfer
READ_ROUNDS_E = 16

# Sequence pacing. One write of the runtime sequence costs OPS_CYCLES on silicon (about 108 measured with
# the launch probe); the padding is sized with the smaller 100 so that a faster command stream only
# leaves more margin, never less.
OPS_CYCLES = 100
SEQ_SLACK_CYCLES = 40000           # after the core's last release: cold writes (up to 10k), carrier, margin
FLUSH_PULSES = 96                  # software events per traced unit (2 bytes each: about 7 packets)
PRE_PUSH_PADS = 12                 # case E: ops between the go and the S2MM push, so the word is waiting at START

# Core program timing, cycles. The loop terms are those of the packet-stage generator.
NOP_DELAY_MAX = ps.NOP_DELAY_MAX
LOOP_BASE = ps.LOOP_BASE
LOOP_ITER = ps.LOOP_ITER
CHAIN_DELTA = 10                   # extra cycles between consecutive triggers beyond gap + words: a chained source BD
                                   # delivers its first word that much earlier than the first BD's (measured on the emulator)

# The software events the flush generates: (trace name, number written to Event_Generate). The carrier shim's
# trace unit uses the user events as its own start and stop, so the shim units flush with an event that never
# occurs there: the carrier channel's task never finishes.
SW_EVENT_SHIM = ("DMA_S2MM_1_FINISHED_TASK", 23)
SW_EVENT_MEMTILE = ("USER_EVENT_0", 159)
EVENT_GENERATE = {"shim": 0x34008, "memtile": 0x94008}

SHIM_EVENTS = ["DMA_S2MM_0_START_TASK", "DMA_S2MM_0_FINISHED_TASK", "DMA_S2MM_0_STREAM_STARVATION",
               "DMA_S2MM_1_STREAM_STARVATION", "DMA_MM2S_0_START_TASK", "DMA_MM2S_0_FINISHED_TASK",
               SW_EVENT_SHIM[0]]
MEMTILE_EVENTS = ["PORT_RUNNING_4", "PORT_RUNNING_0", SW_EVENT_MEMTILE[0]]
CORE_GROUNDING = "PERF_CNT_2"
CORE_SWEEP = "INSTR_EVENT_0"

COLD_X = 200                       # X above this is cold (F above 350 for a one-word task)
COLD_F_ONE_WORD = 350
WORD_PATH = 10                     # memtile port word -> shim stream arrival, cycles (report: +10, skew +-3)


# -- cases --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Transfer:
    """One S2MM task and the source BD that feeds it."""
    words: int
    buf: int = 0                   # runtime-sequence data argument: 0 main, 1 alternate
    offset: int = 0                # words from the buffer start
    gap: int = 0                   # idle cycles from the previous transfer's last word (the first: delay after go)
    stride: int = 0                # >0: `words` single words, `stride` words apart (pre-warm of several pages)
    role: str = ""

    @property
    def span_words(self) -> int:
        """Words from offset to one past the last word written."""
        return 1 + (self.words - 1) * self.stride if self.stride else self.words


@dataclass(frozen=True)
class Case:
    name: str
    family: str                    # A | B | C | D | E | F
    transfers: tuple[Transfer, ...]
    priority: bool = False
    trace: str = "Q"               # Q quiet | N noisy
    fill: str = ""                 # "" | s2mm | mm2s: background traffic during the idle before the last transfer
    paced_by: str = "core"         # core | sequence (case E: one task per round, spacing set by the sequence)
    spacing: int = 0               # case E: cycles per round
    reads: bool = False            # case E1: one background read per round
    note: str = ""

    @property
    def tasks(self) -> int:
        return len(self.transfers)

    @property
    def measured(self) -> Transfer:
        return self.transfers[-1]

    @property
    def background(self) -> bool:
        return bool(self.fill) or self.reads


def _t(words, gap=0, buf=0, offset=0, stride=0, role=""):
    return Transfer(words, buf, offset, gap, stride, role)


A_N = (1, 16, 256, 4096)
A_D = (0, 1000, 20000)
B_G = (0, 100, 1000, 10000)
B_VARIANTS = {"1": ("+64 B", dict(offset=16)), "2": ("+4 KiB", dict(offset=PAGE // WORD)),
              "3": ("+64 KiB", dict(offset=16 * PAGE // WORD)), "4": ("+2 MiB", dict(offset=512 * PAGE // WORD)),
              "5": ("other buffer", dict(buf=1))}
C_I = (1000, 10000, 100000, 300000)
D_I = (10000, 30000, 100000, 300000)
D_BURST = 4096
E_S = (20000, 80000)

P_A = {(16, 0), (16, 20000), (4096, 20000)}
P_B = {(v, g) for v in "135" for g in (100, 10000)}
P_C = {("0", 10000), ("0", 100000), ("0", 300000), ("1", 100000), ("2", 100000)}
P_D = {("0", 30000), ("0", 100000), ("0", 300000), ("1", 100000), ("2", 100000), ("3", 100000)}
P_E = {(0, 80000), (1, 80000)}


def _a_case(n, d):
    return Case(f"a{n}d{d}", "A", (_t(n, d, role="T1"),), priority=(n, d) in P_A)


def _b_case(v, g):
    label, kw = B_VARIANTS[v]
    return Case(f"b{v}g{g}", "B", (_t(16, 0, role="T1"), _t(16, g, role="T2", **kw)),
                priority=(v, g) in P_B, note=f"T2 at {label}")


def _c_case(fill, i, trace="Q", name=None, family="C", priority=None):
    kind = {"0": "", "1": "s2mm", "2": "mm2s"}[fill]
    return Case(name or f"c{fill}i{i}", family, (_t(16, 0, role="W"), _t(16, i, offset=16, role="T2")),
                priority=(fill, i) in P_C if priority is None else priority, trace=trace, fill=kind)


def _d_case(fill, i):
    burst_off = PAGE // WORD
    kind = {"0": "", "1": "s2mm", "2": "mm2s", "3": ""}[fill]
    if fill == "3":
        pre = _t(4, 0, offset=burst_off, stride=PAGE // WORD, role="PRE")
        ts = (_t(16, 0, role="W"), pre, _t(D_BURST, i, offset=burst_off, role="B"))
    else:
        ts = (_t(16, 0, role="W"), _t(D_BURST, i, offset=burst_off, role="B"))
    return Case(f"d{fill}i{i}", "D", ts, priority=(fill, i) in P_D, fill=kind)


def _e_case(reads, s):
    ts = tuple(_t(1, 0, offset=k, role=f"E{k}") for k in range(READ_ROUNDS_E))
    return Case(f"e{reads}s{s}", "E", ts, priority=(reads, s) in P_E, paced_by="sequence", spacing=s,
                reads=bool(reads))


def _f_cases():
    a = Case("fa16d1000q", "F", (_t(16, 1000, role="T1"),), priority=True, trace="Q")
    c = (_c_case("0", 100000, "Q", "fc0i100000q", "F", True))
    return [a, replace(a, name="fa16d1000n", trace="N"), c, replace(c, name="fc0i100000n", trace="N")]


def default_cases() -> list[Case]:
    out = [_a_case(n, d) for n in A_N for d in A_D]
    out += [_b_case(v, g) for v in B_VARIANTS for g in B_G]
    out += [_c_case(f, i) for f in "012" for i in C_I]
    out += [_d_case(f, i) for f in "0123" for i in D_I]
    out += [_e_case(r, s) for r in (0, 1) for s in E_S]
    out += _f_cases()
    return out


def case_from_name(name: str) -> Case:
    for c in default_cases():
        if c.name == name:
            return c
    raise ValueError(f"unknown case {name!r}")


def priority_cases() -> list[Case]:
    return [c for c in default_cases() if c.priority]


def rest_cases() -> list[Case]:
    return [c for c in default_cases() if not c.priority]


REPEATS = 3


def session_order(which: str = "priority") -> list[tuple[str, str]]:
    """(case, tag) in hardware order: REPEATS rounds over the set, the whole set once per round, so a
    drift between processes shows as a difference between rounds, not between cases."""
    cases = {"priority": priority_cases, "rest": rest_cases}[which]()
    return [(c.name, f"h{r}") for r in range(1, REPEATS + 1) for c in cases]


# -- layout and counts ----------------------------------------------------------------------

def source_layout(case: Case) -> list[tuple[int, int]]:
    """(byte address, words) of each transfer's source buffer in the memtile, packed and 64-byte aligned."""
    out, at = [], 0
    for t in case.transfers:
        out.append((at, t.words))
        at += -(-t.words * WORD // SRC_ALIGN) * SRC_ALIGN
    return out


def arg_words(case: Case) -> dict[int, int]:
    """Words each runtime-sequence data argument must hold, from the case alone (independent of the
    runner's own sizing, which the build compares with)."""
    need: dict[int, int] = {}
    for t in case.transfers:
        need[t.buf] = max(need.get(t.buf, 0), t.offset + t.span_words)
    if case.background:
        need[1] = max(need.get(1, 0), background_words(case))
    return need


def background_rounds(case: Case) -> int:
    """Background transfers: one per BG_LOCK_PERIOD of the idle before the last transfer (C, D), or one per
    round (E)."""
    if case.reads:
        return READ_ROUNDS_E
    return max(1, min(case.measured.gap // BG_LOCK_PERIOD, BG_MAX_ROUNDS)) if case.fill else 0


def background_period(case: Case) -> int:
    """Cycles between background transfers: BG_LOCK_PERIOD, stretched so the idle still holds all rounds."""
    return max(BG_LOCK_PERIOD, case.measured.gap // BG_MAX_ROUNDS)


def background_words(case: Case) -> int:
    """Words of the alternate buffer the background traffic touches (case E re-reads one chunk)."""
    return BG_WORDS if case.reads else background_rounds(case) * BG_WORDS


def release_spacings(case: Case) -> list[int]:
    """Cycles between consecutive source-lock releases by the core: the first is the delay after the go
    lock; the next adds the previous transfer's stream time."""
    out, prev = [], None
    for t in case.transfers:
        out.append(t.gap if prev is None else t.gap + prev.words + CHAIN_DELTA)
        prev = t
    return out


def core_cycles(case: Case) -> int:
    """Cycles from the go lock to the core's last release (nominal)."""
    if case.paced_by == "sequence":
        return case.spacing * case.tasks
    return sum(release_spacings(case))


def sequence_pads(case: Case) -> int:
    """Padding writes between the go lock and the flush: the core's schedule, the last transfer's stream
    time, and the slack for cold writes and the carrier, at OPS_CYCLES per write."""
    if case.paced_by == "sequence":
        return 0
    return math.ceil((core_cycles(case) + case.measured.words + SEQ_SLACK_CYCLES) / OPS_CYCLES)


def round_ops(case: Case) -> int:
    return case.spacing // OPS_CYCLES


# -- the core programs ---------------------------------------------------------------------------

def delay(cycles: int, label: str) -> list[str]:
    """Instructions that take `cycles` cycles: nops, or a counted loop with nops for the remainder."""
    if cycles < 0:
        raise ValueError("negative delay")
    if cycles <= NOP_DELAY_MAX:
        return lp.nops(cycles)
    iters, rest = divmod(cycles - LOOP_BASE, LOOP_ITER)
    return [*lp.delay_loop(max(iters, 1), label), *lp.nops(rest)]


def release_at(spacing: int, label: str) -> list[str]:
    """`spacing` cycles after the previous trigger (or the acquire), trigger the next source BD. The trigger
    sits on a fetch-line boundary so a loop's alignment padding never varies."""
    return [*delay(max(spacing - 1, 0), label), ".p2align 4", TRIGGER]


def core_asm(case: Case) -> str:
    """Column 0's core: wait for go, then write one word to its stream port at the nominal spacings. The word
    crosses to the memtile, where a one-word S2MM BD releases the lock of the next source BD (the core cannot
    reach a memtile lock itself). Case E: one go and one trigger per round."""
    n = lp.nops
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2)]
    if case.paced_by == "sequence":
        for k in range(case.tasks):
            body += [f"  acq #{OWN + LOCK_GO}, r3", *n(4), ".p2align 4", TRIGGER]
    else:
        body += [f"  acq #{OWN + LOCK_GO}, r3", *n(4)]
        for k, s in enumerate(release_spacings(case)):
            body += release_at(s, f".Lgap{k}")
    body += [*n(8)]
    if case.trace == "N":
        body += lp.delay_loop(ps.TAIL_ITERS, ".Ltail")
    return lp.frame(body)


def background_asm(case: Case) -> str | None:
    """Column 1's core: after go, `rounds` releases of the background DMA's lock, one period apart, starting
    when the last transfer's idle begins."""
    rounds = background_rounds(case)
    if not case.fill or case.paced_by != "core":
        return None
    n = lp.nops
    start = sum(release_spacings(case)[:-1]) + case.transfers[-2].words
    inner = max((background_period(case) - 14) // LOOP_ITER, 1)
    body = ["  mov r2, #1", "  mov r3, #-1", *n(2), f"  acq #{OWN + BG_GO}, r3", *n(4),
            *delay(start, ".Lbgstart"), f"  movxm r1, #{rounds}", *n(2), ".p2align 4", ".Lbground:",
            *lp.delay_loop(inner, ".Lbgwait"), f"  rel #{OWN + BG_LOCK}, r2", "  add r1, r1, #-1",
            *n(2), "  jnz r1, #.Lbground", *n(5), *n(8)]
    return lp.frame(body)


# -- the design ---------------------------------------------------------------------------------------

def _ramp(base: int, words: int) -> str:
    return ", ".join(str(base | j) for j in range(words))


def _pads(count: int) -> str:
    return "\n".join("      aiex.npu.write32(%c_zero, %c_one) {buffer = @scratch} : i32, i32" for _ in range(count))


def _dest_task(var: str, t: Transfer, bd_id: int, arg: str, arg_words_: int, tile: str = "%t00", channel: int = 0,
               direction: str = "S2MM") -> str:
    dims = f" sizes = [1, 1, {t.words}, 1] strides = [0, 0, {t.stride}, 1]" if t.stride else ""
    return f"""      %{var} = aiex.dma_configure_task({tile}, {direction}, {channel}) {{
        aie.dma_bd(%{arg} : memref<{arg_words_}xi32> offset = {t.offset} len = {t.words}{dims}) {{bd_id = {bd_id} : i32}}
        aie.end
      }}
      aiex.dma_start_task(%{var})
      aiex.dma_free_task(%{var})
"""


def flush_units(case: Case) -> list[tuple[int, int, str]]:
    """(column, row, kind) of every traced unit that gets software-event pulses."""
    return [(0, 0, "shim"), (CARRIER_COL, 0, "shim"), (0, 1, "memtile")]


def flush_block(case: Case) -> str:
    """FLUSH_PULSES rounds of one software event per traced unit. Equal frames would be folded into a
    repeat count and add no bytes, so the rounds are spaced irregularly (0, 1, 2 extra writes in turn)."""
    lines = []
    for j in range(FLUSH_PULSES):
        for col, row, kind in flush_units(case):
            ev = (SW_EVENT_SHIM if kind == "shim" else SW_EVENT_MEMTILE)[1]
            lines.append(f"      aiex.npu.write32(%c_gen_{kind}, %c_ev_{ev}) {{column = {col} : i32, row = {row} : i32}} "
                         ": i32, i32")
        lines.append(_pads(j % 3)) if j % 3 else None
    return "\n".join(lines)


def _src_bds(case: Case) -> tuple[str, str, int]:
    """The source buffers, and the memtile DMA program: MM2S 0 sends each transfer's words behind lSrc; S2MM 1
    takes one trigger word per transfer and releases lSrc for it. Returns (buffers, dma text, trigger address)."""
    layout = source_layout(case)
    bufs, bds, tbds = "", "", ""
    for k, (t, (addr, words)) in enumerate(zip(case.transfers, layout)):
        bufs += (f"    %src{k} = aie.buffer(%t01) {{address = {addr} : i32, sym_name = \"src{k}\", "
                 f"initial_value = dense<[{_ramp((k + 1) << 24, words)}]> : tensor<{words}xi32>}} : memref<{words}xi32>\n")
        nxt = f"^bd{k + 1}" if k + 1 < case.tasks else "^end"
        bds += f"""    ^bd{k}:
      %c_a{k} = arith.constant 1 : i32
      aie.use_lock(%lSrc, AcquireGreaterEqual, %c_a{k})
      aie.dma_bd(%src{k} : memref<{words}xi32>)
      %c_r{k} = arith.constant 1 : i32
      aie.use_lock(%lDone, Release, %c_r{k})
      aie.next_bd {nxt}
"""
        tnext = f"^tbd{k + 1}" if k + 1 < case.tasks else "^end"
        tbds += f"""    ^tbd{k}:
      %c_ta{k} = arith.constant 1 : i32
      aie.use_lock(%lTrig, AcquireGreaterEqual, %c_ta{k})
      aie.dma_bd(%trig : memref<{case.tasks}xi32> offset = {k} len = 1)
      %c_tr{k} = arith.constant 1 : i32
      aie.use_lock(%lSrc, Release, %c_tr{k})
      aie.next_bd {tnext}
"""
    taddr = layout[-1][0] + -(-layout[-1][1] * WORD // SRC_ALIGN) * SRC_ALIGN
    bufs += f"    %trig = aie.buffer(%t01) {{address = {taddr} : i32, sym_name = \"trig\"}} : memref<{case.tasks}xi32>\n"
    dma = f"""    %mt = aie.memtile_dma(%t01) {{
      %m0 = aie.dma_start(MM2S, 0, ^bd0, ^trig)
{bds}    ^trig:
      %m1 = aie.dma_start(S2MM, 1, ^tbd0, ^end)
{tbds}    ^end:
      aie.end
    }}
"""
    return bufs, dma, taddr


def _background(case: Case) -> tuple[str, str, str, str]:
    """Column-1 compute tile (1,2): the source or sink of the background traffic behind its own lock, released
    by its core one period apart (cases C, D), or a lock-free 256-word sink for the per-round reads (case E).
    Returns (locks, dma text, flows, shim allocations)."""
    rounds = background_rounds(case)
    locks = ""
    gated = case.paced_by == "core"
    if gated:
        locks = (f"    %lBg = aie.lock(%t12, {BG_LOCK}) {{init = 0 : i32, sym_name = \"lBg\"}}\n"
                 f"    %lBgDone = aie.lock(%t12, {BG_DONE}) {{init = 0 : i32, sym_name = \"lBgDone\"}}\n"
                 f"    %lBgGo = aie.lock(%t12, {BG_GO}) {{init = 0 : i32, sym_name = \"lBgGo\"}}\n")
    rep = f", repeat_count = {rounds - 1}" if rounds > 1 and gated else ""
    acq = ("      %c_ba = arith.constant 1 : i32\n      aie.use_lock(%lBg, AcquireGreaterEqual, %c_ba)\n" if gated else "")
    rel = ("      %c_br = arith.constant 1 : i32\n      aie.use_lock(%lBgDone, Release, %c_br)\n" if gated else "")
    if case.fill == "s2mm":
        buf = (f"    %bgbuf = aie.buffer(%t12) {{address = 1024 : i32, sym_name = \"bgbuf\", initial_value = "
               f"dense<[{_ramp(0x0F000000, BG_WORDS)}]> : tensor<{BG_WORDS}xi32>}} : memref<{BG_WORDS}xi32>\n")
        direction, words = "MM2S", BG_WORDS
        flow = "    aie.flow(%t12, DMA : 0, %t10, DMA : 0)\n"
        alloc = "    aie.shim_dma_allocation @bg (%t10, S2MM, 0)\n"
    else:
        words = BG_WORDS if gated else BG_WORDS * rounds
        buf = f"    %bgbuf = aie.buffer(%t12) {{address = 1024 : i32, sym_name = \"bgbuf\"}} : memref<{words}xi32>\n"
        direction = "S2MM"
        flow = "    aie.flow(%t10, DMA : 0, %t12, DMA : 0)\n"
        alloc = "    aie.shim_dma_allocation @bg (%t10, MM2S, 0)\n"
    mem = (f"{buf}    %bgmem = aie.mem(%t12) {{\n      %b0 = aie.dma_start({direction}, 0, ^bd0, ^end{rep})\n"
           f"    ^bd0:\n{acq}      aie.dma_bd(%bgbuf : memref<{words}xi32>)\n{rel}      aie.next_bd ^end\n"
           f"    ^end:\n      aie.end\n    }}\n")
    return locks, mem, flow, alloc


def design_mlir(case: Case) -> str:
    sizes = arg_words(case)
    main_words = sizes.get(0, 16)
    alt_words = sizes.get(1, 16)
    bg = case.background
    tiles = ["    %t00 = aie.tile(0, 0)", "    %t10 = aie.tile(1, 0)", "    %t01 = aie.tile(0, 1)",
             "    %t02 = aie.tile(0, 2)"] + (["    %t12 = aie.tile(1, 2)"] if bg else [])
    bufs, memtile, _ = _src_bds(case)
    locks = (f"    %lSrc = aie.lock(%t01, {LOCK_SRC}) {{init = 0 : i32, sym_name = \"lSrc\"}}\n"
             f"    %lDone = aie.lock(%t01, {LOCK_DONE}) {{init = 0 : i32, sym_name = \"lDone\"}}\n"
             f"    %lTrig = aie.lock(%t01, {LOCK_TRIG}) {{init = {case.tasks} : i32, sym_name = \"lTrig\"}}\n"
             f"    %lGo = aie.lock(%t02, {LOCK_GO}) {{init = 0 : i32, sym_name = \"lGo\"}}\n")
    flows = "    aie.flow(%t01, DMA : 0, %t00, DMA : 0)\n    aie.flow(%t02, Core : 0, %t01, DMA : 1)\n"
    allocs = "    aie.shim_dma_allocation @out (%t00, S2MM, 0)\n"
    bgmem, bgcore = "", ""
    if bg:
        bl, bgmem, bf, ba = _background(case)
        locks, flows, allocs = locks + bl, flows + bf, allocs + ba
        bgcore = ("    func.func private @k_bg() attributes {link_with = \"bg.o\"}\n"
                  "    aie.core(%t12) {\n      func.call @k_bg() : () -> ()\n      aie.end\n    }\n"
                  if case.paced_by == "core" else "    aie.core(%t12) {\n      aie.end\n    }\n")
    rounds = background_rounds(case)
    seq = ["      %c_zero = arith.constant 0 : i32", "      %c_one = arith.constant 1 : i32"]
    seq += [f"      %c_gen_{r} = arith.constant {EVENT_GENERATE[r]} : i32" for r in ("shim", "memtile")]
    seq += [f"      %c_ev_{SW_EVENT_SHIM[1]} = arith.constant {SW_EVENT_SHIM[1]} : i32",
            f"      %c_ev_{SW_EVENT_MEMTILE[1]} = arith.constant {SW_EVENT_MEMTILE[1]} : i32"]
    argname = {0: ("dst", main_words), 1: ("alt", alt_words)}
    body = ""
    if case.paced_by == "sequence":
        for k, t in enumerate(case.transfers):
            body += "      aiex.set_lock(%lGo, %c_one)\n" + _pads(PRE_PUSH_PADS) + "\n"
            name, w = argname[t.buf]
            body += _dest_task(f"task{k}", t, k % 4, name, w)
            used = PRE_PUSH_PADS + 4
            if case.reads:
                body += _pads((case.spacing // OPS_CYCLES) // 2 - used) + "\n"
                body += _dest_task(f"read{k}", _t(BG_WORDS, 0), k % 4, "alt", alt_words, "%t10", 0, "MM2S")
                used = (case.spacing // OPS_CYCLES) // 2 + 4
            body += _pads(round_ops(case) - used) + "\n"
    else:
        for k, t in enumerate(case.transfers):
            name, w = argname[t.buf]
            body += _dest_task(f"task{k}", t, k, name, w)
        if case.fill:
            body += _dest_task("bg", _t(BG_WORDS * rounds, 0, 1), 0, "alt", alt_words, "%t10", 0,
                               "S2MM" if case.fill == "s2mm" else "MM2S")
        body += "      aiex.set_lock(%lGo, %c_one)\n"
        if bg:
            body += "      aiex.set_lock(%lBgGo, %c_one)\n"
        body += _pads(sequence_pads(case)) + "\n"
    body += flush_block(case) + "\n"
    args = f"%dst: memref<{main_words}xi32>, %alt: memref<{alt_words}xi32>"
    return f"""module {{
  aie.device(npu1_2col) {{
{chr(10).join(tiles)}
{bufs}    %scratch = aie.buffer(%t02) {{address = {ADDR_SCRATCH} : i32, sym_name = "scratch"}} : memref<4xi32>
{locks}{flows}    func.func private @k_core() attributes {{link_with = "core.o"}}
    aie.core(%t02) {{
      func.call @k_core() : () -> ()
      aie.end
    }}
{bgcore}{memtile}{bgmem}{allocs}    aie.runtime_sequence @seq({args}) {{
{chr(10).join(seq)}
{body}    }}
  }}
}}
"""


# -- trace injection arguments ----------------------------------------------------------------------------

def inject_args(case: Case) -> list[str]:
    args = ["--shim-sweep-events", ",".join(SHIM_EVENTS), "--shim-grounding", "",
            "--memtile-sweep-events", ",".join(MEMTILE_EVENTS), "--memtile-grounding", ""]
    if case.trace == "N":
        return args + ["--trace-tiles", "0:2", "--core-grounding", CORE_GROUNDING, "--core-sweep-events", CORE_SWEEP]
    return args + ["--trace-tiles", ""]


# -- build -----------------------------------------------------------------------------------------------

TRACE_LOWERING = ("builtin.module(aie.device(aie-insert-trace-flows{lateral-routing=true lateral-target-col=%d},"
                  "aie-trace-to-config,aie-trace-pack-reg-writes,aie-inline-trace-config))")
TRACE_BUFFER_BYTES = 65536


def bounds_lines(insts: Path, sizes: dict[int, int]) -> tuple[int, str]:
    """The runner's device-free bounds check: (exit status, its report), every shim DMA task named."""
    args = [str(lp.RUNNER), "--check-bounds", str(insts)] + [f"{i}={b}" for i, b in sorted(sizes.items())]
    p = subprocess.run(args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def runner_arg_sizes(insts: Path) -> dict[int, int]:
    p = subprocess.run([str(lp.RUNNER), "--print-arg-sizes", str(insts)], capture_output=True, text=True, check=True)
    return {int(a): int(b) for a, b in (ln.split() for ln in p.stdout.splitlines() if ln.strip())}


TRACE_ARG = 2                      # the runtime sequence declares dst (0), alt (1); the trace buffer is appended


def expected_arg_bytes(case: Case) -> dict[int, int]:
    """Buffer sizes the runner must derive from the instruction stream: each data argument a task touches
    (to the end of its last byte), and the trace buffer."""
    sizes = {i: w * WORD for i, w in arg_words(case).items()}
    return {**sizes, TRACE_ARG: TRACE_BUFFER_BYTES}


def build_case(case: Case, out: Path) -> None:
    d = out / case.name
    d.mkdir(parents=True, exist_ok=True)
    lp.assemble(core_asm(case), d / "core.o", "k_core")
    if (bg := background_asm(case)) is not None:
        lp.assemble(bg, d / "bg.o", "k_bg")
    (d / "base.mlir").write_text(design_mlir(case))
    lp.sh([sys.executable, str(REPO / "tools" / "mlir-trace-inject.py"), "--input", "base.mlir", "--out", "aie.mlir",
           "--trace-mode", "event_time", *inject_args(case), "--trace-config-out", "trace_config.json",
           "--config-test-name", case.name, "--config-src-mlir", "base.mlir", "--buffer-size", str(TRACE_BUFFER_BYTES)],
          cwd=d, log=d / "inject.log")
    # The trace destination is forced to the carrier column, off the column under test. aiecc has no option
    # for it, so the trace lowering runs here and aiecc then finds no trace ops left.
    lp.sh(["aie-opt", f"--pass-pipeline={TRACE_LOWERING % CARRIER_COL}", "aie.mlir", "-o", "lowered.mlir"],
          cwd=d, log=d / "trace-lowering.log")
    lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
           "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "lowered.mlir"],
          cwd=d, log=d / "aiecc.log")
    got = runner_arg_sizes(d / "insts.bin")
    want = expected_arg_bytes(case)
    if got != want:
        raise SystemExit(f"{case.name}: the runner sizes the buffers {got}, the case needs {want}")
    rc, report = bounds_lines(d / "insts.bin", want)
    (d / "bounds.txt").write_text(f"{case.name}: buffers (arg: bytes) {want}\nrunner --check-bounds exit {rc}\n{report}")
    if rc != 0:
        raise SystemExit(f"{case.name}: a shim DMA task leaves its buffer:\n{report}")
    (d / "case.json").write_text(json.dumps({"case": case.name, "family": case.family, "buffers": want}) + "\n")


# -- run ---------------------------------------------------------------------------------------------------

def select(out: Path, names) -> list[Case]:
    cases = [case_from_name(n) for n in names] if names else default_cases()
    return [c for c in cases if (out / c.name).is_dir()]


def run_case(out: Path, case: Case, tag: str, backend: str) -> bool:
    d = out / case.name
    r = d / f"{tag}.{backend}"
    r.mkdir(parents=True, exist_ok=True)
    lockfile = None
    if backend == "hw":
        import fcntl
        lockfile = open(lp.HW_LOCK, "w")
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{case.name}: waiting for {lp.HW_LOCK} (do not wrap this tool in flock)", file=sys.stderr, flush=True)
            fcntl.flock(lockfile, fcntl.LOCK_EX)
    try:
        p = subprocess.run(ps.runner_command(d, r), env=ps.run_env(backend), capture_output=True, text=True,
                           timeout=120 if backend == "hw" else 3600)
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
           "--xclbin-mlir", str(d / "input_with_addresses.mlir"), "--trace-mode", "event_time",
           "--out-events", str(r / "events.json")], log=r / "parse.log")
    return True


# -- predictions ----------------------------------------------------------------------------------------------

HYPOTHESES = ("HP", "HR", "HI", "HC", "HE")
HYPOTHESIS_NAMES = {
    "HP": "page/buffer translation: cold = first access to the page (granule unknown) since the context was made",
    "HR": "run-first: cold = the first write of the run, any destination",
    "HI": "idle path: cold when no host access of any master happened within the last T cycles (T unknown)",
    "HC": "per channel: cold = the first task of each channel",
    "HE": "translation-cache eviction: warm entries are lost after enough other accesses or time",
}
SLOW_EXCESS = 500                  # drain window above the word count by more than this is the slow drain
SLOW_DRAIN_FROM = 88000            # corpus: slow at 88k and more idle, normal at 38k and less


def predict(case: Case) -> dict[str, str]:
    """What each hypothesis says about the measured task of a case: cold | warm | slow | normal | ? (no
    claim: the hypothesis leaves it open) | mixed (case E, a fraction)."""
    f = case.family
    t = case.measured
    p = {h: "?" for h in HYPOTHESES}
    if f == "A":
        p.update(HP="cold", HR="cold", HI="cold", HC="cold", HE="cold")
    elif f == "B":
        v, g = case.name[1], t.gap
        warm_always = dict(HR="warm", HC="warm")
        hp = "warm" if v == "1" else "cold"
        hi = "warm" if g <= 100 else ("cold" if g >= 10000 else "?")
        p.update(HP=hp, HI=hi, HE=("warm" if v == "1" and g <= 100 else "?"), **warm_always)
    elif f in ("C", "F") and case.tasks == 2 and t.words == 16:
        fill = case.fill
        i = t.gap
        if fill == "":
            hi = "warm" if i <= 1000 else ("cold" if i >= 100000 else "?")
            if case.trace == "N":
                hi = "warm"                  # periodic carrier writes keep the path alive within every idle
            he = "warm"
        else:
            hi, he = "warm", ("cold" if i >= 100000 else "?")
        p.update(HP="warm", HR="warm", HC="warm", HI=hi, HE=he)
        if f == "F" and case.trace == "N":
            p["HE"] = "cold"                 # the noisy carrier touches other pages
    elif f == "F":
        p.update(HP="cold", HR="cold", HI="cold", HC="cold", HE="cold")
    elif f == "D":
        i = t.gap
        fill = case.name[1]
        long_idle = i >= SLOW_DRAIN_FROM
        p.update(HR="n/a", HC="n/a")
        p["HI"] = "normal" if fill in ("1", "2") else ("slow" if long_idle else ("normal" if i <= 10000 else "?"))
        p["HP"] = "normal" if fill == "3" else "slow"
        p["HE"] = "slow" if (long_idle or fill in ("1", "2")) else "?"
    elif f == "E":
        p.update(HP="mixed:none-cold", HR="mixed:none-cold", HC="mixed:none-cold",
                 HI=("mixed:more-cold-than-reads" if not case.reads else "mixed:fewer-cold"),
                 HE=("mixed:fewer-cold" if not case.reads else "mixed:more-cold-than-quiet"))
    return p


def consistent(prediction: str, observed: str) -> str:
    """ok | CONTRADICTS | - (no claim)"""
    if prediction in ("?", "n/a") or observed in ("-", "?"):
        return "-"
    if prediction.startswith("mixed"):
        return "-"
    return "ok" if prediction == observed else "CONTRADICTS"


def termination(case: Case) -> str:
    """One line: why the case ends whatever silicon does."""
    ends = ("the go lock is set by the runtime sequence, the source lock released by the core once it holds go "
            f"(counted delays, {case.tasks} release{'s' if case.tasks != 1 else ''}), ")
    if case.paced_by == "sequence":
        ends = ("each round the runtime sequence sets the go lock, the core releases the source lock once, "
                f"and the sequence pushes one 1-word task ({case.tasks} rounds of padding writes), ")
    bds = f"every S2MM BD holds exactly the source's words ({', '.join(str(t.words) for t in case.transfers)}), "
    if case.fill:
        bds += ("the background source/sink BDs repeat a fixed number of times and the background core releases "
                f"its lock exactly that many times ({background_rounds(case)}), ")
    if case.reads:
        bds += "each background read is one 16-word MM2S task into a sink BD of 256 words that nothing waits for, "
    return ends + bds + "and the sequence waits on nothing (padding writes, then the flush pulses)."


# -- the table ---------------------------------------------------------------------------------------------------

@dataclass
class Unit:
    """Decoded events of one run, by traced unit. Columns are logical (the placement's origin removed)."""
    events: list
    origin: int

    def stamps(self, pkt: int, row: int, col: int, name: str) -> list[int]:
        return sorted(e.ts for e in self.events
                      if e.pkt_type == pkt and e.row == row and e.col == col + self.origin and e.name == name)


def runs_of(stamps: list[int]) -> list[tuple[int, int]]:
    return dl._runs(stamps)


def load_unit(run_dir: Path) -> Unit:
    doc = json.loads((run_dir / "events.json").read_text())
    return Unit(dl.load_level_events(run_dir), int(doc.get("placement", {}).get("origin_col", 0)))


def task_rows(u: Unit, case: Case) -> dict:
    """Per-task numbers of the S2MM channel under test, on the shim unit's clock, plus the covariates.
    fw (first word) is when the task's first word reaches the shim: the source port's first word of the
    transfer plus WORD_PATH, but not before START+6 (a word that was already waiting is taken at START+6).
    The starvation fall is the silicon cross-check (fw_starv): the two agree where a word arrives at a
    starved channel, and a queued task's START_TASK precedes its predecessor's FINISHED_TASK, so the
    starvation level alone cannot date the second word. The source port's word clock comes from the memtile."""
    starts = [a for a, _ in runs_of(u.stamps(2, 0, 0, "DMA_S2MM_0_START_TASK"))]
    fins = [a for a, _ in runs_of(u.stamps(2, 0, 0, "DMA_S2MM_0_FINISHED_TASK"))]
    starv = runs_of(u.stamps(2, 0, 0, "DMA_S2MM_0_STREAM_STARVATION"))
    words = u.stamps(3, 1, 0, "PORT_RUNNING_4")
    arrivals = [b + 1 for _, b in runs_of(u.stamps(2, 0, CARRIER_COL, "DMA_S2MM_1_STREAM_STARVATION"))]
    notes = []
    if len(starts) != case.tasks:
        notes.append(f"{len(starts)} START_TASK, expected {case.tasks}")
    if len(fins) != case.tasks:
        notes.append(f"{len(fins)} FINISHED_TASK, expected {case.tasks}")
    total = sum(t.words for t in case.transfers)
    if len(words) != total:
        notes.append(f"source port ran {len(words)} word cycles, expected {total}")
    rows, cum = [], 0
    prev_lw = prev_fin = None
    for k, t in enumerate(case.transfers):
        if k >= len(starts) or k >= len(fins):
            break
        st, fin = starts[k], fins[k]
        mem = words[cum:cum + t.words] if len(words) >= cum + t.words else []
        cum += t.words
        lw_mem = (mem[-1] + WORD_PATH) if mem else None
        fw_mem = (mem[0] + WORD_PATH) if mem else None
        fw = max(st + 6, fw_mem) if fw_mem is not None else st + 6
        onset = next((b + 1 for a, b in starv if a <= fw - 1 <= b), None)
        waiting = fw_mem is not None and fw_mem <= st + 6
        win = (mem[-1] - mem[0] + 1) if mem else None
        if onset is not None and abs(onset - fw) > 8:
            notes.append(f"task {k}: starvation fall {onset} vs first word by the source port {fw}")
        before = [a for a in arrivals if a <= fw]
        in_idle = [a for a in arrivals if prev_lw is not None and prev_lw < a <= fw]
        n = t.words
        x = fin - (fw + n - 1)
        f = fin - fw
        if n == 1:
            cls = "cold" if f > COLD_F_ONE_WORD else "warm"
        elif n > 300:
            cls = "cold" if x > COLD_X else "plain"
        else:
            cls = "cold" if x > COLD_X else "warm"
        rows.append({
            "k": k, "role": t.role, "words": n, "start": st, "fw": fw, "waiting": waiting, "fin": fin,
            "fw_starv": onset,
            "F": f, "X": x, "class": cls, "fw_mem": fw_mem, "lw": lw_mem,
            "window": win, "excess": None if win is None else win - n,
            "fin_after_last": None if lw_mem is None else fin - lw_mem,
            "idle_word": None if prev_lw is None or fw_mem is None else fw_mem - prev_lw - 1,
            "idle_fin": None if prev_fin is None else fw - prev_fin,
            "carrier_since": None if not before else fw - before[-1], "carrier_in_idle": len(in_idle),
        })
        prev_lw, prev_fin = lw_mem, fin
    return {"rows": rows, "notes": notes, "carrier_arrivals": len(arrivals)}


def observed(case: Case, rows: list[dict]) -> str:
    """The measured task as one word comparable with a prediction: cold | warm | slow | normal | - ."""
    if not rows or rows[-1]["k"] != case.tasks - 1:
        return "-"
    r = rows[-1]
    if case.family == "D":
        return "-" if r["excess"] is None else ("slow" if r["excess"] > SLOW_EXCESS else "normal")
    if case.family == "E":
        later = [x for x in rows[1:]]
        return f"cold {sum(x['class'] == 'cold' for x in later)} of {len(later)}"
    if r["class"] == "plain":
        return "-"
    return r["class"]


def _v(x) -> str:
    return "-" if x is None else str(x)


TABLE_HEAD = (
    "case          run      k role words   START      fw wait    FIN      F      X class  idle_w idle_f  carr  nca "
    "  window  excess fin-lw  observed   HP    HR    HI    HC    HE  verdict\n"
    "  fw: first word (START+6 when the word was waiting); X = FIN - (fw + words - 1), F = FIN - fw; class cold when\n"
    "  X > 200 (F > 350 for one word), 'plain' = a long task that hides the cold term; idle_w: previous task's last word\n"
    "  (source port clock, +10 to the shim) to this fw; idle_f: previous FINISHED to this fw; carr: cycles from the last\n"
    "  carrier packet arrival to fw; nca: arrivals inside the idle. verdict per hypothesis: ok / CONTRADICTS / -.")


def format_table(rows) -> str:
    """rows: (case, run name, task_rows dict)."""
    lines = [TABLE_HEAD]
    for case, run, tr in rows:
        pr = predict(case)
        obs = observed(case, tr["rows"])
        verdict = " ".join(f"{h}:{consistent(pr[h], obs)}" for h in HYPOTHESES)
        for r in tr["rows"]:
            last = r["k"] == case.tasks - 1
            lines.append(
                f"{case.name:<13} {run:<8} {r['k']:>2} {r['role']:<4} {r['words']:>5} {r['start']:>7} {r['fw']:>7} "
                f"{'y' if r['waiting'] else 'n':>4} {r['fin']:>6} {r['F']:>6} {r['X']:>6} {r['class']:<5} "
                f"{_v(r['idle_word']):>7} {_v(r['idle_fin']):>6} {_v(r['carrier_since']):>5} {r['carrier_in_idle']:>4} "
                f"{_v(r['window']):>8} {_v(r['excess']):>7} {_v(r['fin_after_last']):>6}  "
                + (f"{obs:<9} " + " ".join(f"{pr[h]:<5}" for h in HYPOTHESES) + f"  {verdict}" if last else ""))
        for n in tr["notes"]:
            lines.append(f"{'':<13} {run:<8} NOTE: {n}")
    return "\n".join(lines)


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    rows, absent = [], {}
    for case in select(out, args.cases):
        for run in args.runs:
            run_dir = out / case.name / run
            if not (run_dir / "trace_raw.bin").exists() or not (run_dir / "events.json").exists():
                absent.setdefault(run, []).append(case.name)
                continue
            rows.append((case, run, task_rows(load_unit(run_dir), case)))
    print(format_table(rows))
    for run, names in absent.items():
        print(f"not in {run}: {' '.join(names)}")
    return 0


# -- CLI ------------------------------------------------------------------------------------------------------

def _chosen(names, which) -> list[Case]:
    if names:
        return [case_from_name(n) for n in names]
    return {None: default_cases, "priority": priority_cases, "rest": rest_cases}[which]()


def cmd_build(args) -> int:
    out = Path(args.out_dir).resolve()
    for case in _chosen(args.cases, args.set):
        build_case(case, out)
        print("built", case.name, flush=True)
    return 0


def cmd_run(args) -> int:
    """Stops at the first run that does not complete; never retries."""
    out = Path(args.out_dir).resolve()
    for case in select(out, args.cases):
        if not run_case(out, case, args.tag, args.backend):
            print(f"{case.name}: run did not complete; stop")
            return 1
        print(f"{case.name}: ok", flush=True)
    return 0


def cmd_bounds(args) -> int:
    """Device-free bounds pre-flight of built cases: the runner's own sizing of every host buffer must equal
    what the case needs, and every shim DMA task must lie inside its buffer. Prints each case's buffers and
    tasks; exit status 1 if any case fails (or is not built)."""
    out = Path(args.out_dir).resolve()
    bad = 0
    for case in _chosen(args.cases, args.set):
        insts = out / case.name / "insts.bin"
        if not insts.exists():
            print(f"{case.name}: NOT BUILT")
            bad += 1
            continue
        want = expected_arg_bytes(case)
        got = runner_arg_sizes(insts)
        rc, report = bounds_lines(insts, want)
        ok = got == want and rc == 0
        bad += not ok
        print(f"{case.name}: {'ok' if ok else 'FAIL'} buffers {want}" + ("" if got == want else f" (runner sizes {got})"))
        print("".join(f"    {ln}\n" for ln in report.splitlines()), end="")
    print("ALL OK" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


def cmd_cases(args) -> int:
    for c in default_cases():
        pr = predict(c)
        print(f"{c.name:<13} {c.family} {'P' if c.priority else ' '} tasks={c.tasks} trace={c.trace} "
              + " ".join(f"{h}={pr[h]}" for h in HYPOTHESES))
    return 0


def cmd_session(args) -> int:
    for name, tag in session_order(args.set):
        print(name, tag)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases")
    b = sub.add_parser("build")
    b.add_argument("out_dir")
    b.add_argument("--cases", nargs="*", default=[])
    b.add_argument("--set", choices=("priority", "rest"), default=None)
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--cases", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+", help="run directory names, e.g. h1.hw e1.emu")
    t.add_argument("--cases", nargs="*", default=[])
    k = sub.add_parser("bounds")
    k.add_argument("out_dir")
    k.add_argument("--cases", nargs="*", default=[])
    k.add_argument("--set", choices=("priority", "rest"), default=None)
    s = sub.add_parser("session")
    s.add_argument("--set", choices=("priority", "rest"), default="priority")
    args = ap.parse_args(argv)
    return {"cases": cmd_cases, "build": cmd_build, "run": cmd_run, "table": cmd_table,
            "bounds": cmd_bounds, "session": cmd_session}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
