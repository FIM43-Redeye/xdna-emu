#!/usr/bin/env python3
"""Runtime-sequence probe: how fast does the controller work through task pushes and syncs?

The runtime sequence of a design is run by the device's command processor. This probe holds that
sequence as the independent variable and reads its effect from one tile: shim (0,0)'s own trace
unit (one timer) records START_TASK and FINISHED_TASK of all four of its DMA channels. The shim has
a fixed event per channel (only the memtile needs a channel selector), so no selector programming
is involved. The trace leaves through the next column's shim S2MM 1, which keeps all four channels
of the shim under test free; its Start frame gives the trace start.

Data paths are shim loopbacks through the shim's own stream switch: M0 -> S0 (MM2S 0 to S2MM 0) and
M1 -> S1. Tasks are short host transfers; every BD has its own slice of the one host buffer
(sources first, destinations after, the buffer sized from the largest offset + length).

Family P, push spacing: pushes (a `dma_memcpy_nd`: BD blockwrite, address patch, Task_Queue write32)
issued back to back with no sync between, then a sync on every awaited channel at the end.
  pN_mm / pN_ss / pN_il   N pushes, all MM2S first, all S2MM first, interleaved (M, S, M, S, ..)
  _kK                     K plain register writes before every push (they change nothing)
  p8_mmpair, p4_same_*    the same channel twice in a row, against the same pushes spread over channels
  p4_mm_notok, p8_mm_alltok   token only on the S2MM tasks / on every task (default: the last task of each channel)
One shim has four channels, so N = 8 needs two tasks per channel (queue depth is 4).

Family W, sync latency: a loopback of L words on M0 -> S0 (token on S0), a sync on S0 issued at once,
then a marker push (8 words on M1, whose sink S1 was armed first and takes every marker). The marker's
START_TASK stamps when the controller resumed: S0 FINISHED_TASK -> marker START_TASK is the quantity.
  wP_L<L>                 P transfer-sync pairs in a row, L = 300 / 3000 / 30000 words
  w1_late_L<L>            the sync follows 64 plain writes, so the task has finished before the sync is reached
  w1_nosync_L<L>          control: the marker is pushed with no sync before it

Probe 9d additions (plan_9d). Same W stream, with one more controlled variable each:
  w1_f<N>_L<L>, w1_late_L<L>   N plain writes between the S0 push and the sync (0 / 16 / 32 / 64 / 128 at L=3000)
  w1_latenosync_L<L>           the 64 writes, then the marker, and no sync before it
  w2_dummy_L3000               an 8-word S0 task and its sync ahead of the two pairs
  w2_nosync_L300               two pairs, both markers pushed before any sync
  <case>_br / <case>_tw        bracketed: a user event is generated on the traced shim immediately before and
                               immediately after every measured sync (USER_EVENT_0 before, USER_EVENT_1 after), so the
                               shim's own trace unit stamps when the wait is issued and when the sequence resumes.
                               The twin is the same stream without the marks (the marks cost time; the twin shows
                               how much). The two user events take the slots of the MM2S FINISHED events.

Closing tail (every case). After the measured section the sequence runs TAIL_ROUNDS rounds of one short
loopback task per channel (M0, S0, M1, S1, each with a token) followed by a sync on all four channels, then the
trailing writes. Every channel's last task is thus awaited by the sequence itself, so no data task can still be
running when the sequence ends, whatever the controller's latencies. The tail's events also carry the measured
events out of the trace's last packets: the trace S2MM writes the host buffer in 64-byte bursts and leaves a
final sub-burst packet unwritten, so without a tail the last measured events (and the stop frame) can be
missing on silicon. The tail does not change the measured section: its pushes come after the measured section's
own syncs, its buffer slices lie above the measured ones, and its BDs reuse ids whose tasks the measured syncs
have already seen finish.

Reading the tables. A push's START_TASK stamps the push only while its channel is idle or prefetching
a queued BD; a second task pushed to a busy channel can start later (same-channel cases). The `ops`
column lists the opcodes strictly between a push's Task_Queue write32 and the previous one's: a push
that carries a token has one more maskwrite32 than one that does not, which is why the token variants
exist. In the late-sync cases the quantity includes the LATE_FILL writes the sequence executes
between the task's end and the sync.

    gen_runtime_seq_probe.py list
    gen_runtime_seq_probe.py build OUT [--only CASE ..]
    gen_runtime_seq_probe.py run OUT TAG --backend hw|emu [--only CASE ..]
    gen_runtime_seq_probe.py table OUT RUN [RUN ..] [--only CASE ..]
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

PUSH_WORDS = 8
MARKER_WORDS = 8
LATE_FILL = 64
TAIL_FILL = 32                       # plain writes after the last sync: every task is long done
TAIL_ROUNDS = 3                      # closing rounds of one task per channel, each round synced on all four
TAIL_WORDS = 8
TAIL_ORDER = ("M0", "S0", "M1", "S1")
TAIL_BD_IDS = (0, 1, 2, 3)           # free again: the measured syncs cover every measured task
TAIL_MIN_EVENTS = 2 * 4 * TAIL_ROUNDS  # START + FINISHED per tail task: well over one trace packet of frames
SCRATCH_REG = 0x000340F0             # Timer_Trig_Event_Low_Value of the carrier shim: writing 0 changes nothing
DIM_LIMIT = 1000                     # inner dimension of a long BD stays under the hardware size field

CHANNELS = {"M0": ("MM2S", 0, "in0"), "M1": ("MM2S", 1, "in1"),
            "S0": ("S2MM", 0, "out0"), "S1": ("S2MM", 1, "out1")}
PATH_OF = {"M0": 0, "S0": 0, "M1": 1, "S1": 1}
# Task_Queue (Start_Queue) registers of a shim tile, AM025.
QUEUE_OFFSETS = {0x1D204: "S0", 0x1D20C: "S1", 0x1D214: "M0", 0x1D21C: "M1"}
TXN_WRITE, TXN_BLOCKWRITE, TXN_MASKWRITE, TXN_MASKPOLL, TXN_PATCH, TXN_SYNC = 0, 1, 3, 4, 0x81, 0x80
KIND = {TXN_WRITE: "write32", TXN_BLOCKWRITE: "blockwrite", TXN_MASKWRITE: "maskwrite32", TXN_MASKPOLL: "maskpoll",
        TXN_PATCH: "patch", TXN_SYNC: "sync"}
# The lowering inserts a maskpoll on a channel's status before a fifth task is queued on a channel the sequence has
# not synced (Task_Queue depth 4): it waits for queue space. It appears only in the closing tail of w4 cases, after
# the measured section.
EVENT_ORDER = ("M0", "M1", "S0", "S1")
# Event_Generate of a shim tile (NOC module), AM025: writing an event number makes the tile raise that event.
EVENT_GENERATE = 0x34008
# Marks: "A" is written immediately before a sync, "B" immediately after it. The shim module has two user events;
# the trace start and stop use the ones of the egress shim (column 1), the traced shim (column 0) has them free.
MARK_EVENTS = {"A": ("USER_EVENT_0", 126), "B": ("USER_EVENT_1", 127)}
MARKED_COL, MARKED_ROW = 0, 0
DEFAULT_EVENTS = tuple(f"DMA_{CHANNELS[ch][0]}_{CHANNELS[ch][1]}_{what}_TASK" for ch in EVENT_ORDER
                       for what in ("START", "FINISHED"))
# Eight trace slots only: a bracketed case gives up the FINISHED events of the two token-less MM2S channels (their
# STARTs, which stamp every push, stay) for the two user events.
BRACKET_EVENTS = tuple(e for e in DEFAULT_EVENTS if not (e.startswith("DMA_MM2S") and e.endswith("FINISHED_TASK"))) \
    + tuple(name for name, _ in MARK_EVENTS.values())


# -- plan -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Push:
    ch: str
    words: int
    token: bool = True
    tail: bool = False               # a closing-tail task, not part of the measured section
    dummy: bool = False              # a leading throwaway task: not one of the measured pairs


@dataclass(frozen=True)
class Fill:
    count: int


@dataclass(frozen=True)
class Sync:
    ch: str


@dataclass(frozen=True)
class Mark:
    which: str                       # "A": written just before a sync; "B": just after it


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    steps: tuple
    events: tuple = DEFAULT_EVENTS   # the eight trace slots, in slot order


def _p_case(name: str, order: list[str], k: int = 0, tokens: str = "last") -> Case:
    """Push order `order`, k fillers before every push, then a sync per awaited channel."""
    last = {ch: i for i, ch in enumerate(order)}          # index of the last push of each channel
    steps: list = []
    for i, ch in enumerate(order):
        if k:
            steps.append(Fill(k))
        token = {"last": last[ch] == i, "all": True,
                 "s2mm": ch[0] == "S" and last[ch] == i}[tokens]
        steps.append(Push(ch, PUSH_WORDS, token))
    awaited = [ch for ch in EVENT_ORDER if ch in last and (tokens != "s2mm" or ch[0] == "S")]
    steps += [Sync(ch) for ch in awaited]
    steps += drain_syncs(steps) + closing_tail() + [Fill(TAIL_FILL)]
    return Case(name, "P", tuple(steps))


def _w_case(name: str, pairs: int, words: int, late: bool = False, nosync: bool = False, *, fill: int = 0,
            marks: bool = False, dummy: bool = False, events: tuple = DEFAULT_EVENTS,
            tail_rounds: int = TAIL_ROUNDS) -> Case:
    """W stream. `fill` plain writes follow every S0 push (`late` is the 64-write case); `nosync` pushes the marker
    with no sync before it (the syncs follow all the pushes); `dummy` puts an 8-word S0 task and its sync ahead of
    the pairs; `marks` brackets every measured sync with the two user-event writes."""
    fill = LATE_FILL if late else fill
    steps: list = [Push("S1", MARKER_WORDS * pairs, True)]   # the marker sink, armed first
    if dummy:
        steps += [Push("M0", PUSH_WORDS, False, dummy=True), Push("S0", PUSH_WORDS, True, dummy=True), Sync("S0")]

    def sync(ch: str) -> list:
        return [Mark("A"), Sync(ch), Mark("B")] if marks else [Sync(ch)]

    for _ in range(pairs):
        steps += [Push("M0", words, False), Push("S0", words, True)]
        if fill:
            steps.append(Fill(fill))
        if nosync:
            steps.append(Push("M1", MARKER_WORDS, False))
            continue
        steps += sync("S0") + [Push("M1", MARKER_WORDS, False)]
    if nosync:
        for _ in range(pairs):
            steps += sync("S0")
    steps += sync("S1")
    steps += drain_syncs(steps) + closing_tail(tail_rounds) + [Fill(TAIL_FILL)]
    return Case(name, "W", tuple(steps), events)


def drain_syncs(steps: list) -> list:
    """One sync per token the measured section issued but did not wait for (a token that arrives before its sync is
    kept, so a surplus one would satisfy a later sync early): after these, every channel's tokens and syncs are
    level, so each tail sync waits for its own round's task."""
    surplus = {ch: 0 for ch in EVENT_ORDER}
    for s in steps:
        if isinstance(s, Push) and s.token:
            surplus[s.ch] += 1
        elif isinstance(s, Sync):
            surplus[s.ch] -= 1
    return [Sync(ch) for ch in EVENT_ORDER for _ in range(surplus[ch])]


def closing_tail(rounds: int = TAIL_ROUNDS) -> list:
    """`rounds` rounds of one token-carrying task per channel, each round followed by a sync on every
    channel: the sequence cannot end before every channel's last task has finished."""
    steps: list = []
    for _ in range(rounds):
        steps += [Push(ch, TAIL_WORDS, True, tail=True) for ch in TAIL_ORDER]
        steps += [Sync(ch) for ch in EVENT_ORDER]
    return steps


def plan() -> dict[str, Case]:
    mm8 = ["M0", "M1", "M0", "M1", "S0", "S1", "S0", "S1"]
    cases = [
        _p_case("p2_mm", ["M0", "S0"]), _p_case("p2_ss", ["S0", "M0"]),
        _p_case("p4_mm", ["M0", "M1", "S0", "S1"]), _p_case("p4_ss", ["S0", "S1", "M0", "M1"]),
        _p_case("p4_il", ["M0", "S0", "M1", "S1"]),
        _p_case("p4_mm_notok", ["M0", "M1", "S0", "S1"], tokens="s2mm"),
        _p_case("p8_mm", mm8), _p_case("p8_ss", ["S0", "S1", "S0", "S1", "M0", "M1", "M0", "M1"]),
        _p_case("p8_il", ["M0", "S0", "M1", "S1", "M0", "S0", "M1", "S1"]),
        _p_case("p8_mmpair", ["M0", "M0", "M1", "M1", "S0", "S0", "S1", "S1"]),
        _p_case("p8_mm_alltok", mm8, tokens="all"),
        _p_case("p4_same_mm", ["M0", "M0", "S0", "S0"]), _p_case("p4_same_ss", ["S0", "S0", "M0", "M0"]),
    ]
    for k in (4, 16):
        cases += [_p_case(f"p4_mm_k{k}", ["M0", "M1", "S0", "S1"], k),
                  _p_case(f"p4_ss_k{k}", ["S0", "S1", "M0", "M1"], k)]
    for pairs in (1, 2, 4):
        for words in (300, 3000, 30000):
            cases.append(_w_case(f"w{pairs}_L{words}", pairs, words))
    cases += [_w_case("w1_late_L300", 1, 300, late=True), _w_case("w1_late_L3000", 1, 3000, late=True),
              _w_case("w2_late_L300", 2, 300, late=True), _w_case("w1_nosync_L300", 1, 300, nosync=True)]
    return {c.name: c for c in cases}


def plan_9d() -> dict[str, Case]:
    """Probe 9d. The cases that also exist in probe 9c are the same plans (their instruction streams are
    byte-identical), so a session can read them against probe 9c's runs."""
    cases = [_w_case("w1_late_L300", 1, 300, late=True), _w_case("w1_L300", 1, 300), _w_case("w1_L3000", 1, 3000)]
    for name, pairs, words, late in (("w1_late_L300", 1, 300, True), ("w1_L300", 1, 300, False),
                                     ("w1_L3000", 1, 3000, False), ("w1_L30000", 1, 30000, False),
                                     ("w2_L30000", 2, 30000, False)):
        # four tail rounds: the six traced events per round of a bracketed case match the eight of a probe 9c case
        cases += [_w_case(f"{name}_tw", pairs, words, late, events=BRACKET_EVENTS, tail_rounds=4),
                  _w_case(f"{name}_br", pairs, words, late, marks=True, events=BRACKET_EVENTS, tail_rounds=4)]
    cases += [_w_case(f"w1_L{words}", 1, words) for words in (1000, 10000, 20000)]
    cases += [_w_case("w1_late_L30000", 1, 30000, late=True)]
    cases += [_w_case(f"w1_f{n}_L3000", 1, 3000, fill=n) for n in (16, 32)]
    cases += [_w_case("w1_late_L3000", 1, 3000, late=True), _w_case("w1_f128_L3000", 1, 3000, fill=128)]
    cases += [_w_case(f"w1_latenosync_L{words}", 1, words, late=True, nosync=True) for words in (300, 3000)]
    cases += [_w_case("w2_dummy_L3000", 2, 3000, dummy=True), _w_case("w2_nosync_L300", 2, 300, nosync=True)]
    out = {c.name: c for c in cases}
    if len(out) != len(cases):
        raise ValueError("plan_9d: duplicate case names")
    return out


# Probe 9e's grid: the wait lengths (words of the S0 task) between and beyond probe 9d's 300 / 3000 / 30000
SWEEP_9E = (600, 1000, 1500, 2000, 2500, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000)
DUMMY_9E = (1000, 3000, 8000, 20000)
TWIN_LS_9E = (2000, 8000)


def plan_9e() -> dict[str, Case]:
    """Probe 9e: how the resume time depends on how long the wait has been pending, mapped densely with marks.
    The first case is probe 9d's bracketed w1_L3000 (same plan, same instruction stream): the positive control."""
    def br(name, pairs, words, **kw):
        return _w_case(name, pairs, words, marks=True, events=BRACKET_EVENTS, tail_rounds=4, **kw)
    cases = [br("w1_L3000_br", 1, 3000)]
    cases += [br(f"w1_L{n}_br", 1, n) for n in SWEEP_9E]
    cases += [br(f"w1_f{n}_L3000_br", 1, 3000, fill=n) for n in (16, 32)]
    cases += [br(f"w2_dummy_L{n}_br", 2, n, dummy=True) for n in DUMMY_9E]
    cases += [_w_case(f"w1_L{n}_tw", 1, n, events=BRACKET_EVENTS, tail_rounds=4) for n in TWIN_LS_9E]
    out = {c.name: c for c in cases}
    if len(out) != len(cases):
        raise ValueError("plan_9e: duplicate case names")
    return out


# Probe 9f's grid: wait lengths (words of the S0 task) around the levels probe 9e found, as dense edge sweeps
EDGES_9F = (tuple(range(200, 701, 50)) + (800, 900, 1000)         # shortest waits
            + tuple(range(1900, 2151, 50))                        # around 2000
            + (1500, 2500, 2600, 2700, 2800, 2900, 3000)          # 1500 .. 3000
            + tuple(range(6250, 7051, 100))                       # the end of the plateau
            + tuple(range(8050, 8951, 100)))                      # the long waits


def prior_command_9f(name: str, long_words: int, long_path: str, sync_order: tuple[str, ...]) -> Case:
    """A command with no marks, run only to set up the state the next command starts from. After the opening marker
    pair and two ordinary closing rounds comes a last round of four token-carrying tasks, `long_words` words on
    the `long_path` channels (M0 and S0, or M1 and S1) and 8 words on the other path, awaited in `sync_order`; the command
    ends on 32 plain writes."""
    long_chs = (long_path, "S" + long_path[1])
    last = [Push(ch, long_words if ch in long_chs else 8, True, tail=True) for ch in TAIL_ORDER]
    steps = [Push("S1", 8, True), Push("M1", 8, True), Sync("M1"), Sync("S1")] + closing_tail(2) + last \
        + [Sync(ch) for ch in sync_order] + [Fill(TAIL_FILL)]
    return Case(name, "W", tuple(steps), BRACKET_EVENTS)


def plan_9f() -> dict[str, Case]:
    """Probe 9f: sampling points of the wait (bracketed length cases with dense edge sweeps), the effect of the prior
    command's last waits on the next command, and a fill case that locates a once-per-run stall.
    Length cases that also exist in probe 9e are the same plans (byte-identical instruction streams).
    Prior-command cases: k_a / k_e await the long M0 -> S0 pair last, M0 then S0 (3000 words in k_e);
    k_c awaits it S0 then M0; k_f has the long pair on the second path and awaits M0, S0, M1, S1 in order."""
    def br(name, words, **kw):
        return _w_case(name, 1, words, marks=True, events=BRACKET_EVENTS, tail_rounds=4, **kw)
    cases = [br("w1_L3000_br", 3000)]
    cases += [br(f"w1_L{n}_br", n) for n in sorted(set(EDGES_9F) - {3000})]
    cases += [prior_command_9f("k_a_L30000", 30000, "M0", ("M1", "S1", "M0", "S0")),
              prior_command_9f("k_c_L30000", 30000, "M0", ("M1", "S1", "S0", "M0")),
              prior_command_9f("k_f_L30000", 30000, "M1", ("M0", "S0", "M1", "S1")),
              prior_command_9f("k_e_L3000", 3000, "M0", ("M1", "S1", "M0", "S0")),
              br("w1_f128_L3000_br", 3000, fill=128)]
    out = {c.name: c for c in cases}
    if len(out) != len(cases):
        raise ValueError("plan_9f: duplicate case names")
    return out


# Probe 9g's grids: fine sweeps of the wait length around three of the levels probe 9f mapped
SWEEPS_9G = {"a": tuple(range(8440, 8641, 20)), "b": tuple(range(2600, 2801, 20)), "c": tuple(range(240, 601, 40))}


def plan_9g() -> dict[str, Case]:
    """Probe 9g: three fine sweeps of the bracketed one-pair case (no filler) around the levels of probe 9f: part a at the
    longest waits, part b between two plateaus, part c at the shortest waits. The first case is the positive control
    (probe 9e's w1_L3000_br, byte-identical instruction stream); cases that exist in probe 9f are the same plans."""
    cases = [_w_case("w1_L3000_br", 1, 3000, marks=True, events=BRACKET_EVENTS, tail_rounds=4)]
    for words in sorted({n for ns in SWEEPS_9G.values() for n in ns}):
        cases.append(_w_case(f"w1_L{words}_br", 1, words, marks=True, events=BRACKET_EVENTS, tail_rounds=4))
    out = {c.name: c for c in cases}
    if len(out) != len(cases):
        raise ValueError("plan_9g: duplicate case names")
    return out


# Probe 9h's grids: long waits that follow a prior-command case, and a fine sweep of the shortest steps just past one level
LONG_9H = tuple(range(7300, 8101, 100))
FINE_9H = tuple(range(8500, 8541, 4))


def plan_9h() -> dict[str, Case]:
    """Probe 9h: bracketed one-pair cases (no filler) at long waits, meant to run immediately after the unmarked
    prior-command case k_a, and a fine sweep of the transfer length in 4-word steps. The first case is the positive
    control; all plans are the same as in the earlier probes (byte-identical streams where a case exists there)."""
    def br(words):
        return _w_case(f"w1_L{words}_br", 1, words, marks=True, events=BRACKET_EVENTS, tail_rounds=4)
    cases = [br(3000), prior_command_9f("k_a_L30000", 30000, "M0", ("M1", "S1", "M0", "S0"))]
    cases += [br(n) for n in LONG_9H + FINE_9H]
    out = {c.name: c for c in cases}
    if len(out) != len(cases):
        raise ValueError("plan_9h: duplicate case names")
    return out


# -- layout and safety --------------------------------------------------------------------------------

def pushes(case: Case) -> list[Push]:
    return [s for s in case.steps if isinstance(s, Push)]


def measured_pushes(case: Case) -> list[Push]:
    return [p for p in pushes(case) if not p.tail]


def layout(case: Case) -> list[tuple[Push, int]]:
    """(push, word offset in the host buffer), in push order: the measured section's MM2S sources first and its
    S2MM destinations after (the offsets set 9 ran with), then the tail's sources and destinations above them."""
    out = {}
    base = 0
    for group in (measured_pushes(case), [p for p in pushes(case) if p.tail]):
        src_total = sum(p.words for p in group if p.ch[0] == "M")
        nxt = {"M": base, "S": base + src_total}
        for p in group:
            out[id(p)] = nxt[p.ch[0]]
            nxt[p.ch[0]] += p.words
        base = nxt["S"]
    return [(p, out[id(p)]) for p in pushes(case)]


def buffer_words(case: Case) -> int:
    return max(off + p.words for p, off in layout(case))


def validate(case: Case) -> None:
    """Every task finishes and nothing waits on something that cannot happen."""
    fed = {0: 0, 1: 0}          # MM2S words pushed per path so far
    sunk = {0: 0, 1: 0}         # S2MM words pushed per path so far
    tokens = {ch: 0 for ch in CHANNELS}
    syncs = {ch: 0 for ch in CHANNELS}
    last_push = {}
    awaited_after = {}
    for i, s in enumerate(case.steps):
        if isinstance(s, Push):
            if s.ch not in CHANNELS:
                raise ValueError(f"{case.name}: channel {s.ch}")
            (fed if s.ch[0] == "M" else sunk)[PATH_OF[s.ch]] += s.words
            tokens[s.ch] += s.token
            last_push[s.ch] = i
        elif isinstance(s, Sync):
            syncs[s.ch] += 1
            if syncs[s.ch] > tokens[s.ch]:
                raise ValueError(f"{case.name}: sync on {s.ch} with no token to wait for")
            path = PATH_OF[s.ch]
            if sunk[path] > fed[path]:
                raise ValueError(f"{case.name}: sync on {s.ch} before its sink's words can arrive "
                                 f"(sink {sunk[path]} words, fed {fed[path]})")
            awaited_after[s.ch] = i
    for path in (0, 1):
        if fed[path] != sunk[path]:
            raise ValueError(f"{case.name}: path {path} sink {sunk[path]} words but source {fed[path]}: "
                             f"a task is never fed or never drained")
    for ch, i in last_push.items():
        if awaited_after.get(ch, -1) < i:
            raise ValueError(f"{case.name}: the last task of {ch} is never awaited (no sync after it)")
        if not case.steps[i].token:
            raise ValueError(f"{case.name}: the last task of {ch} carries no token, so no sync can cover it")
    for ch in CHANNELS:
        if tokens[ch] != syncs[ch]:
            raise ValueError(f"{case.name}: {ch} issues {tokens[ch]} tokens but has {syncs[ch]} syncs; a surplus "
                             f"token would let a later sync return before its own task")
    syncs_at = [i for i, s in enumerate(case.steps) if isinstance(s, Sync)]
    if syncs_at and any(isinstance(s, Push) for s in case.steps[syncs_at[-1]:]):
        raise ValueError(f"{case.name}: a push after the last sync")
    for i, s in enumerate(case.steps):
        if isinstance(s, Mark):
            want = (case.steps[i + 1:i + 2], "A") if s.which == "A" else (case.steps[i - 1:i], "B")
            if s.which not in MARK_EVENTS or not (want[0] and isinstance(want[0][0], Sync)):
                raise ValueError(f"{case.name}: mark {s.which} at step {i} is not directly "
                                 f"{'before' if s.which == 'A' else 'after'} a sync")
    marks = [s.which for s in case.steps if isinstance(s, Mark)]
    for which, (name, _) in MARK_EVENTS.items():
        if marks.count(which) and name not in case.events:
            raise ValueError(f"{case.name}: marks need {name} among the traced events")
    if marks.count("A") != marks.count("B"):
        raise ValueError(f"{case.name}: {marks.count('A')} marks A but {marks.count('B')} marks B")
    if len(case.events) != 8 or len(set(case.events)) != 8:
        raise ValueError(f"{case.name}: the trace unit has eight distinct slots, got {len(case.events)}")
    tail = [s for s in case.steps if isinstance(s, Push) and s.tail]
    if 2 * len(tail) < TAIL_MIN_EVENTS:
        raise ValueError(f"{case.name}: closing tail of {len(tail)} tasks is too short to carry the trace out")
    first_tail = next((i for i, s in enumerate(case.steps) if isinstance(s, Push) and s.tail), len(case.steps))
    if any(isinstance(s, Mark) for s in case.steps[first_tail:]):
        raise ValueError(f"{case.name}: a mark inside the closing tail")
    if any(isinstance(s, Push) and not s.tail for s in case.steps[first_tail:]):
        raise ValueError(f"{case.name}: a measured push after the closing tail began")


# -- MLIR -----------------------------------------------------------------------------------------------

def _dims(words: int) -> tuple[int, int]:
    """(outer, inner) with outer * inner == words and inner <= DIM_LIMIT."""
    if words <= DIM_LIMIT:
        return 1, words
    inner = next(d for d in range(DIM_LIMIT, 0, -1) if words % d == 0)
    return words // inner, inner


def design_mlir(case: Case) -> str:
    nwords = buffer_words(case)
    m = f"memref<{nwords}xi32>"
    consts: dict[int, str] = {}
    has_marks = any(isinstance(s, Mark) for s in case.steps)

    def c(v: int) -> str:
        consts.setdefault(v, f"%c{v}")
        return consts[v]

    body: list[str] = []
    lay = iter(layout(case))
    ident = {"S": 0, "M": 0}
    next_bd = iter(range(0, 16))
    tail_bd = iter(TAIL_BD_IDS * (sum(1 for p in pushes(case) if p.tail) // len(TAIL_BD_IDS)))
    for s in case.steps:
        if isinstance(s, Fill):
            body += ["      aiex.npu.write32(%c_reg, %c_zero) {column = 1 : i32, row = 0 : i32} : i32, i32"] * s.count
        elif isinstance(s, Mark):
            body.append(f"      aiex.npu.write32(%c_evgen, %c_mark_{s.which}) "
                        f"{{column = {MARKED_COL} : i32, row = {MARKED_ROW} : i32}} : i32, i32")
        elif isinstance(s, Push):
            p, off = next(lay)
            direction, ch, sym = CHANNELS[s.ch]
            outer, inner = _dims(s.words)
            tok = ", issue_token = true" if s.token else ""
            body.append(f"      aiex.npu.dma_memcpy_nd(%buf[{c(0)}, {c(0)}, {c(0)}, {c(off)}] "
                        f"[{c(1)}, {c(1)}, {c(outer)}, {c(inner)}] [{c(0)}, {c(0)}, {c(inner)}, {c(1)}]) "
                        f"{{id = {next(tail_bd) if s.tail else next(next_bd)} : i64, metadata = @{sym}{tok}}} : {m}")
        else:
            body.append(f"      aiex.npu.dma_wait {{symbol = @{CHANNELS[s.ch][2]}}}")
    events = list(case.events)
    nl = "\n"
    const_lines = [f"      {name} = arith.constant {v} : i64" for v, name in sorted(consts.items())]
    mark_consts = (f"      %c_evgen = arith.constant {EVENT_GENERATE} : i32\n" +
                   "".join(f"      %c_mark_{w} = arith.constant {v} : i32\n" for w, (_, v) in MARK_EVENTS.items())
                   if has_marks else "")
    return f"""module {{
  aie.device(npu1_2col) {{
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    aie.flow(%t00, DMA : 0, %t00, DMA : 0)
    aie.flow(%t00, DMA : 1, %t00, DMA : 1)
    aie.shim_dma_allocation @in0 (%t00, MM2S, 0)
    aie.shim_dma_allocation @in1 (%t00, MM2S, 1)
    aie.shim_dma_allocation @out0 (%t00, S2MM, 0)
    aie.shim_dma_allocation @out1 (%t00, S2MM, 1)
    aie.trace @trace_shim(%t00) {{
      aie.trace.packet id = 1 type = shimtile
{nl.join(f'      aie.trace.event <"{e}">' for e in events)}
      aie.trace.start broadcast = 15
      aie.trace.stop broadcast = 14
    }}
    aie.runtime_sequence(%buf: {m}) {{
      aie.trace.host_config {{buffer_size = 65536 : i32, egress_shim_col = 1 : i32}}
      aie.trace.start_config @trace_shim
      %c_reg = arith.constant {SCRATCH_REG} : i32
      %c_zero = arith.constant 0 : i32
{mark_consts}{nl.join(const_lines)}
{nl.join(body)}
    }}
  }}
}}
"""


# -- instruction stream ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    kind: str
    col: int = 0
    row: int = 0
    offset: int = 0
    words: int = 0


def decode_ops(insts: bytes) -> list[Op]:
    """The instruction stream as a list of ops; ValueError on an unknown opcode or a stream that
    does not parse to its stated length."""
    n_ops = struct.unpack_from("<I", insts, 8)[0]
    off, out = 16, []
    for _ in range(n_ops):
        if off + 8 > len(insts):
            raise ValueError("stream ends inside an op")
        opcode = insts[off]
        if opcode in (TXN_WRITE, TXN_MASKWRITE, TXN_MASKPOLL):
            size = 24 if opcode == TXN_WRITE else 28
            addr = struct.unpack_from("<I", insts, off + 8)[0]
            out.append(Op(KIND[opcode], (addr >> 25) & 0x7F, (addr >> 20) & 0x1F, addr & 0xFFFFF))
        elif opcode == TXN_BLOCKWRITE:
            size = struct.unpack_from("<I", insts, off + 12)[0]
            addr = struct.unpack_from("<I", insts, off + 8)[0]
            out.append(Op("blockwrite", (addr >> 25) & 0x7F, (addr >> 20) & 0x1F, addr & 0xFFFFF,
                          (size - 16) // 4))
        elif opcode in (TXN_PATCH, TXN_SYNC):
            size = struct.unpack_from("<I", insts, off + 4)[0]
            out.append(Op(KIND[opcode]))
        else:
            raise ValueError(f"opcode {opcode:#x} at offset {off}")
        off += size
    if off != len(insts):
        raise ValueError(f"stream parses to {off} bytes, file has {len(insts)}")
    return out


@dataclass(frozen=True)
class PushOp:
    index: int
    channel: str


def push_ops(ops: list[Op], data_col: int = 0) -> list[PushOp]:
    """Task_Queue writes of the data shim, in order."""
    return [PushOp(i, QUEUE_OFFSETS[o.offset]) for i, o in enumerate(ops)
            if o.kind == "write32" and o.row == 0 and o.col == data_col and o.offset in QUEUE_OFFSETS]


def marks_in_ops(ops: list[Op]) -> int:
    """Event_Generate writes to the traced shim: the marks of the instruction stream."""
    return sum(1 for o in ops if o.kind == "write32" and (o.col, o.row, o.offset) == (MARKED_COL, MARKED_ROW, EVENT_GENERATE))


def between_pushes(ops: list[Op], pushes_: list[PushOp]) -> list[dict[str, int]]:
    """For each pair of consecutive pushes the opcode kinds and counts strictly between them."""
    out = []
    for a, b in zip(pushes_, pushes_[1:]):
        counts: dict[str, int] = {}
        for o in ops[a.index + 1:b.index]:
            counts[o.kind] = counts.get(o.kind, 0) + 1
        out.append(counts)
    return out


def op_summary(counts: dict[str, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "-"


# -- tables ---------------------------------------------------------------------------------------------

def _stamps(events: list[dict], ch: str, what: str, trace_start: int) -> list[int]:
    direction, num, _ = CHANNELS[ch]
    name = f"DMA_{direction}_{num}_{what}_TASK"
    return sorted(e["ts"] - trace_start for e in events if e["name"] == name)


def push_rows(case: Case, events: list[dict], trace_start: int) -> list[dict]:
    """One row per push in sequence order: the nth push of a channel takes the nth START_TASK of
    that channel, and `gap` is the difference to the previous push's START."""
    seen: dict[str, int] = {}
    rows, prev = [], None
    for p in measured_pushes(case):
        n = seen.get(p.ch, 0)
        seen[p.ch] = n + 1
        st = _stamps(events, p.ch, "START", trace_start)
        start = st[n] if n < len(st) else None
        rows.append({"ch": p.ch, "start": start,
                     "gap": None if start is None or prev is None else start - prev})
        prev = start if start is not None else prev
    return rows


def sync_rows(case: Case, events: list[dict], trace_start: int) -> list[dict]:
    """Per transfer-sync pair: S0 FINISHED_TASK to the marker's START_TASK on M1, and the transfer
    (S0 START to FINISHED)."""
    n_dummy = sum(1 for p in measured_pushes(case) if p.ch == "S0" and p.dummy)
    s0_start = _stamps(events, "S0", "START", trace_start)[n_dummy:]
    s0_fin = _stamps(events, "S0", "FINISHED", trace_start)[n_dummy:]
    m1_start = _stamps(events, "M1", "START", trace_start)
    rows = []
    n_pairs = sum(1 for p in measured_pushes(case) if p.ch == "S0" and not p.dummy)
    for i, (a, b) in enumerate(list(zip(s0_start, s0_fin))[:n_pairs]):
        marker = m1_start[i] if i < len(m1_start) else None
        rows.append({"pair": i, "s0_start": a, "s0_fin": b, "transfer": b - a,
                     "marker": marker, "latency": None if marker is None else marker - b})
    return rows


def _user_stamps(events: list[dict], name: str, trace_start: int) -> list[int]:
    return sorted(e["ts"] - trace_start for e in events if e["name"] == name)


def bracket_rows(case: Case, events: list[dict], trace_start: int) -> list[dict]:
    """One row per marked sync, in sequence order: the stamps of the mark before it (mark_a, the wait about to be
    issued) and after it (mark_b, the first thing the sequence does once it resumes), the FINISHED of the task the
    sync awaits, and the START of the next push. All relative to the trace start. A missing stamp is None."""
    marks_a = _user_stamps(events, MARK_EVENTS["A"][0], trace_start)
    marks_b = _user_stamps(events, MARK_EVENTS["B"][0], trace_start)
    pushed: dict[str, int] = {}          # pushes of each channel so far (the nth push takes the nth START / FINISHED)
    awaited: dict[str, int] = {}
    token_pushes: dict[str, list[int]] = {}
    rows, n_mark = [], 0
    steps = case.steps
    for i, s in enumerate(steps):
        if isinstance(s, Push):
            if s.token:
                token_pushes.setdefault(s.ch, []).append(pushed.get(s.ch, 0))
            pushed[s.ch] = pushed.get(s.ch, 0) + 1
        elif isinstance(s, Sync):
            k = awaited.get(s.ch, 0)
            awaited[s.ch] = k + 1
            if not (i and isinstance(steps[i - 1], Mark)):
                continue
            fins = _stamps(events, s.ch, "FINISHED", trace_start)
            idx = token_pushes[s.ch][k]
            nxt = next((p for p in steps[i + 1:] if isinstance(p, Push)), None)
            nxt_n = pushed.get(nxt.ch, 0) if nxt else 0
            starts = _stamps(events, nxt.ch, "START", trace_start) if nxt else []
            row = {"sync": s.ch, "mark_a": marks_a[n_mark] if n_mark < len(marks_a) else None,
                   "mark_b": marks_b[n_mark] if n_mark < len(marks_b) else None,
                   "fin": fins[idx] if idx < len(fins) else None,
                   "next_ch": nxt.ch if nxt else None,
                   "next": starts[nxt_n] if nxt_n < len(starts) else None}
            n_mark += 1

            def diff(a, b):
                return None if a is None or b is None else b - a
            row.update(a_to_fin=diff(row["mark_a"], row["fin"]), fin_to_b=diff(row["fin"], row["mark_b"]),
                       a_to_b=diff(row["mark_a"], row["mark_b"]), b_to_next=diff(row["mark_b"], row["next"]),
                       a_to_next=diff(row["mark_a"], row["next"]))
            rows.append(row)
    return rows


def finish_check(case: Case, events: list[dict], trace_stop: int | None) -> tuple[str, list[str]]:
    """ok when every measured task has its START and FINISHED event and the trace runs on past the last of them.

    With a closing tail the measured section ends before the tail's first push (M0) starts: the measured
    section's own syncs cover every measured task. So each channel must show exactly its measured count of
    START and FINISHED before that first tail START, and something must be traced after the last of them. The
    tail's own events may be cut short on silicon (the trace's final sub-burst packet is left unwritten, which
    is what the tail is for); the tail tasks' finish is proven by the sequence's syncs, not by the trace."""
    problems = []
    has_tail = any(p.tail for p in pushes(case))
    boundary = None
    if has_tail:
        first = TAIL_ORDER[0]
        n_first = sum(1 for p in measured_pushes(case) if p.ch == first)
        st = _stamps(events, first, "START", 0)
        boundary = st[n_first] if len(st) > n_first else None
    last = 0
    for ch in EVENT_ORDER:
        n = sum(1 for p in measured_pushes(case) if p.ch == ch)
        for what in ("START", "FINISHED"):
            if f"DMA_{CHANNELS[ch][0]}_{CHANNELS[ch][1]}_{what}_TASK" not in case.events:
                continue                     # this case does not trace it
            st = [t for t in _stamps(events, ch, what, 0) if boundary is None or t < boundary]
            if len(st) != n:
                problems.append(f"missing {ch} {what}: {len(st)} of {n}")
            last = max([last, *st])
    for which, (name, _) in MARK_EVENTS.items():
        n = sum(1 for m in case.steps if isinstance(m, Mark) and m.which == which)
        if n:
            st = [t for t in _user_stamps(events, name, 0) if boundary is None or t < boundary]
            if len(st) != n:
                problems.append(f"missing {name}: {len(st)} of {n}")
            last = max([last, *st])
    if not problems:
        if has_tail:
            if boundary is None and (trace_stop is None or trace_stop <= last):
                problems.append(f"tail: nothing traced after the last measured event {last}")
        elif trace_stop is None or trace_stop <= last:
            problems.append(f"tail: trace stop {trace_stop} not after the last FINISHED_TASK {last}")
    return ("finish ok" if not problems else "; ".join(problems)), problems


# -- trace decoding (run directories) -------------------------------------------------------------------

def trace_bounds(raw: bytes, col_pred=None) -> dict[tuple[int, int, int], tuple[int | None, int | None]]:
    """{(pkt_type, row, col): (Start timer, stop timer)} of every mode-0 stream in a raw trace."""
    from trace_decoder import decode_words, detect_per_tile_modes, TraceMode
    from trace_decoder.frame import StartCmd, StopCmd, SyncCmd, RepeatCmd, EventCmd
    sync = 1 << 18
    words = list(struct.unpack(f"<{len(raw) // 4}I", raw[:len(raw) // 4 * 4]))
    modes = detect_per_tile_modes(words)
    out = {}
    for key, cmds in decode_words(words, mode=None).items():
        if modes.get(key) != TraceMode.EVENT_TIME:
            continue
        timer = start = stop = prev = None
        timer, after_sync = 0, False
        for cmd in cmds:
            if isinstance(cmd, StartCmd):
                timer = start = cmd.timer_value
                prev = None
                continue
            if isinstance(cmd, SyncCmd):
                timer += sync
                after_sync = True
                continue
            if isinstance(cmd, RepeatCmd) and after_sync:
                timer += cmd.count * sync
                continue
            after_sync = False
            if isinstance(cmd, StopCmd):
                stop = timer + cmd.idle_cycles
            elif isinstance(cmd, EventCmd):
                timer += 1 + cmd.cycles
                prev = cmd
            elif isinstance(cmd, RepeatCmd) and prev is not None:
                timer += cmd.count if prev.cycles == 0 else cmd.count * (1 + prev.cycles)
        out[key] = (start, stop)
    return out


def load_run(case_dir: Path, run: str) -> tuple[list[dict], int, int | None]:
    """(shim events, trace Start timer, stop timer) of one run of one case."""
    r = case_dir / run
    events = json.loads((r / "events.json").read_text())["events"]
    user = {name for name, _ in MARK_EVENTS.values()}
    shim = [e for e in events if e["pkt_type"] == 2 and ((e["name"].startswith("DMA_")
            and e["name"].endswith(("START_TASK", "FINISHED_TASK"))) or e["name"] in user)]
    bounds = trace_bounds((r / "trace_raw.bin").read_bytes())
    cols = sorted({e["col"] for e in shim if e["name"].startswith("DMA_")})
    key = next(k for k in bounds if k[0] == 2 and (not cols or k[2] == cols[0]))
    start, stop = bounds[key]
    return shim, start or 0, stop


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def case_table(case: Case, case_dir: Path, runs: list[str]) -> str:
    meta = json.loads((case_dir / "case.json").read_text())
    loaded = {}
    for run in runs:
        try:
            loaded[run] = load_run(case_dir, run)
        except (FileNotFoundError, StopIteration, KeyError) as e:
            loaded[run] = None
    n_tail = len(pushes(case)) - len(measured_pushes(case))
    lines = [f"== {case.name}  ({case.family}; {len(measured_pushes(case))} pushes + {n_tail} tail; "
             f"{sum(isinstance(s, Sync) for s in case.steps)} syncs; host buffer {buffer_words(case)} words)"]
    ok = {run: finish_check(case, v[0], v[2])[0] if v else "no data" for run, v in loaded.items()}
    lines.append("finish: " + "; ".join(f"{run}: {t}" for run, t in ok.items()))
    if case.family == "P":
        between = meta.get("between", [])
        lines.append("  push  ch  ops between this push's queue write and the previous one's   " +
                     "  ".join(f"{r:>13}" for r in runs) + "   median")
        pr = {run: push_rows(case, v[0], v[1]) for run, v in loaded.items() if v}
        for i, p in enumerate(measured_pushes(case)):
            vals = [pr[run][i]["gap"] if run in pr else None for run in runs]
            starts = [pr[run][i]["start"] if run in pr else None for run in runs]
            ops = op_summary(between[i - 1]) if i and i - 1 < len(between) else "(first push)"
            good = [v for v in vals if v is not None]
            lines.append(f"  {i:>4}  {p.ch}  {ops:<60} " + "  ".join(f"{_fmt(v):>13}" for v in vals) +
                         f"   {statistics.median(good) if good else '-'}")
            lines.append(f"{'':9}START rel. trace start: " + "  ".join(f"{_fmt(v):>13}" for v in starts))
    else:
        sr = {run: sync_rows(case, v[0], v[1]) for run, v in loaded.items() if v}
        npairs = max((len(r) for r in sr.values()), default=0)
        lines.append("  pair  quantity (S0 FINISHED -> marker START)   " + "  ".join(f"{r:>9}" for r in runs) + "   median")
        for i in range(npairs):
            for label, key in (("latency", "latency"), ("transfer", "transfer"), ("S0 START", "s0_start")):
                vals = [sr[run][i][key] if run in sr and i < len(sr[run]) else None for run in runs]
                good = [v for v in vals if v is not None]
                lines.append(f"  {i:>4}  {label:<40} " + "  ".join(f"{_fmt(v):>9}" for v in vals) +
                             f"   {statistics.median(good) if good else '-'}")
        if any(isinstance(s, Mark) for s in case.steps):
            br = {run: bracket_rows(case, v[0], v[1]) for run, v in loaded.items() if v}
            n_syncs = max((len(r) for r in br.values()), default=0)
            lines.append("  sync  bracket (stamps relative to the trace start)   " +
                         "  ".join(f"{r:>9}" for r in runs) + "   median")
            for i in range(n_syncs):
                for key in ("mark_a", "fin", "mark_b", "next", "a_to_fin", "fin_to_b", "b_to_next", "a_to_next"):
                    vals = [br[run][i][key] if run in br and i < len(br[run]) else None for run in runs]
                    good = [v for v in vals if v is not None]
                    lab = f"{key} ({br[next(iter(br))][i]['sync']}" + (f"->{br[next(iter(br))][i]['next_ch']})" if key.endswith("next") else ")") \
                        if br else key
                    lines.append(f"  {i:>4}  {lab:<40} " + "  ".join(f"{_fmt(v):>9}" for v in vals) +
                                 f"   {statistics.median(good) if good else '-'}")
    return "\n".join(lines)


# -- commands -----------------------------------------------------------------------------------------------

def cmd_list(args) -> int:
    for c in _selected(argparse.Namespace(only=[], plan=getattr(args, "plan", "base"))):
        print(f"{c.name:<16} {c.family}  pushes={len(pushes(c))} syncs={sum(isinstance(s, Sync) for s in c.steps)} "
              f"buffer={buffer_words(c)}")
    return 0


def _selected(args) -> list[Case]:
    table = {"9d": plan_9d, "9e": plan_9e, "9f": plan_9f, "9g": plan_9g, "9h": plan_9h}.get(getattr(args, "plan", "base"), plan)()
    names = args.only or list(table)
    return [table[n] for n in names]


def check_runner_sizing(case: Case, insts: Path, runner: Path) -> None:
    """The runner sizes the host buffer from the compiled stream; it must arrive at this generator's own layout.
    Two derivations of one number: a disagreement is a bug in one of them, found before any run."""
    p = subprocess.run([str(runner), "--print-arg-sizes", str(insts)], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{case.name}: the runner cannot size {insts}: {p.stderr.strip()}")
    sizes = {int(a): int(b) for a, b in (ln.split() for ln in p.stdout.splitlines() if ln.strip())}
    if sizes.get(0) != 4 * buffer_words(case):
        raise SystemExit(f"{case.name}: the runner sizes the data buffer at {sizes.get(0)} bytes, the layout needs "
                         f"{4 * buffer_words(case)}")


def cmd_build(args) -> int:
    import gen_lock_pulse_probe as lp
    out = Path(args.out_dir).resolve()
    for case in _selected(args):
        validate(case)
        d = out / case.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "aie.mlir").write_text(design_mlir(case))
        lp.sh(["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin", "--xclbin-name=aie.xclbin",
               "--get-npu-insts", "--npu-insts-name=insts.bin", "--get-input-with-addresses", "aie.mlir"],
              cwd=d, log=d / "aiecc.log")
        ops = decode_ops((d / "insts.bin").read_bytes())
        pushes_ = push_ops(ops)
        want = [p.ch for p in pushes(case)]
        got = [p.channel for p in pushes_]
        if got != want:
            raise SystemExit(f"{case.name}: instruction stream pushes {got}, plan {want}")
        n_marks = sum(isinstance(s, Mark) for s in case.steps)
        if marks_in_ops(ops) != n_marks:
            raise SystemExit(f"{case.name}: instruction stream holds {marks_in_ops(ops)} marks, plan {n_marks}")
        between = between_pushes(ops, pushes_)
        check_runner_sizing(case, d / "insts.bin", lp.RUNNER)
        (d / "case.json").write_text(json.dumps(
            {"case": case.name, "family": case.family, "pushes": want, "between": between,
             "ops_total": len(ops), "buffer_words": buffer_words(case), "events": list(case.events),
             "marks": n_marks}, indent=1) + "\n")
        print(f"{case.name}: {len(want)} pushes, between consecutive pushes: " +
              "; ".join(op_summary(b) for b in between[:3]) + (" ..." if len(between) > 3 else ""))
    return 0


def cmd_run(args) -> int:
    import gen_lock_pulse_probe as lp
    out = Path(args.out_dir).resolve()
    for case in _selected(args):
        if not lp.run_case(out, case.name, args.tag, args.backend, output=False):
            print(f"{case.name}: run did not complete; stop")
            return 1
        if args.backend == "emu":
            log = (out / case.name / f"{args.tag}.emu" / "runner.log").read_text()
            if "halt_reason=completed" not in log:
                print(f"{case.name}: emulator did not complete")
                return 1
    return 0


def cmd_table(args) -> int:
    out = Path(args.out_dir).resolve()
    for case in _selected(args):
        print(case_table(case, out / case.name, args.runs))
        print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lst = sub.add_parser("list")
    lst.add_argument("--plan", choices=("base", "9d", "9e", "9f", "9g", "9h"), default="base")
    b = sub.add_parser("build")
    b.add_argument("--plan", choices=("base", "9d", "9e", "9f", "9g", "9h"), default="base")
    b.add_argument("out_dir")
    b.add_argument("--only", nargs="*", default=[])
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("tag")
    r.add_argument("--plan", choices=("base", "9d", "9e", "9f", "9g", "9h"), default="base")
    r.add_argument("--backend", choices=("hw", "emu"), required=True)
    r.add_argument("--only", nargs="*", default=[])
    t = sub.add_parser("table")
    t.add_argument("out_dir")
    t.add_argument("runs", nargs="+")
    t.add_argument("--plan", choices=("base", "9d", "9e", "9f", "9g", "9h"), default="base")
    t.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args(argv)
    return {"list": cmd_list, "build": cmd_build, "run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
