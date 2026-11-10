#!/usr/bin/env python3
"""Core lock-stall probes; writes a suite (harness/suite.py).

Two cores share one lock: the consumer at (0,2) owns lock LOCK_ID of its tile (raw ID
48 + LOCK_ID from its own core), and the producer at (0,3) reaches the same lock as
its south quadrant (raw ID LOCK_ID). The topology declares the lock with an init, so
every load starts it from a known count.

Families
--------
lk_D_MODE   the consumer issues `lda r6` and, D bundles after it, `acq` with change -1;
            r6 is sampled at every other bundle +1..+8 after the load (as in the
            cascade-read probe's lda/slda families). MODE `stall`: the lock starts at 0
            and the producer releases it after `late_iters` loop iterations, so the
            acquire blocks for about 10 cycles per iteration. MODE `free`: the producer
            releases at once and the consumer waits `late_iters` iterations before its
            program, so the acquire is granted on issue (the no-stall control).
vlk_KIND_D_MODE  the lk_D experiment for a vector load: KIND `vlda` (`vlda wl10`, load unit
            A), `vldb` (`vldb wl10`, the B load channel) or `am` (`vlda amll0`, an
            accumulator quarter), then `acq` at +D; the loaded register is copied into a
            fresh w register at every other bundle +1..+8 (`vmov wlN, wl10` /
            `vmov wlN, amll0`) and the copies are stored afterwards.
same_MODE   `lda r6 ; paddb [p2], #4 ; acq` in one bundle, then r6 sampled at +1..+8: is
            a load bundled with a blocked acquire issued once, and when; and is the
            pointer add applied once.
eq_NAME     acquires with a change >= 0. The consumer runs a list of lock requests,
            reading the cycle counter before the first and after each; the producer runs
            its own list, one request every `step_iters` loop iterations, reading the
            counter before each. When a consumer request is granted (its producer step)
            tells the acquire rule apart: `simulate` predicts it for each candidate rule,
            and every case completes under every rule (nothing can hang the NPU).

cond_KIND_STATE_V  `acq.cond` / `rel.cond` on LOCK_ID with the condition register r26 = V
            (COND_VALUES), KIND acq or rel. STATE `free` (acquire, lock holds 1: granted on
            issue if performed), `stall` (acquire, lock 0, the producer releases it late:
            a performed request waits), `rel` (release into lock 0). The counter is read
            in front of the request and in the next six bundles (cycles lost beyond one
            per bundle: the arbitration cycle at +4 or a refusal), then the lock's count
            is read back (0/1/2+): was the request performed, did it cost its timing.

Consumer output (`consumer.log`, 64 bytes, little-endian u32):
  lk/same: words 0..7 = r6 at +1..+8 (0 where bundle +D is the acquire), word 8/9 = the
           counter before the load / after the last sample, word 10 = p2 - p0 (same only),
           word 15 = the value the load reads.
  vlk:     (VLOG_BYTES) words 0/1 = the counter before the load / after the last copy, bytes
           32..63 = the loaded data (LDA_NEW in every word), bytes 64+32*(j-1) = the copy
           at +j (zero where +j is the acquire).
  eq:      word i = the counter low word before request 1 (i = 0) and after request i.
Producer output (`producer.ts`, 32 bytes): word k = the counter before its request k + 1.
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
from harness.topology import Buffer, Core, SharedLock, TileRef, Topology   # noqa: E402

LOCK_ID = 15
CONSUMER_TILE = TileRef(0, 2)
PRODUCER_TILE = TileRef(0, 3)
RAW_OWN = 48 + LOCK_ID             # the consumer's own memory module ("east" quadrant)
RAW_SOUTH = LOCK_ID                # the producer's south quadrant: the consumer's module
LDA_OLD, LDA_NEW = 111, 222
# Silicon (lk_D): a refused acquire stalls the core only after the three bundles that
# follow it have issued, so a counter read must sit at least four bundles after a lock
# request to see the request's stall.
LOCK_SETTLE_BUNDLES = 4
LOG_BYTES, TS_BYTES = 64, 32
TS_PAIRS = [(17, 16), (19, 18), (21, 20), (23, 22), (25, 24), (27, 26), (29, 28), (31, 30)]
N = "  nop"


def _frame(body: list[str]) -> str:
    """A returning leaf function saving r16-r31 (the counter pairs use them)."""
    pro = [".text", ".globl test_kernel", "test_kernel:", "  paddb [sp], #64"]
    pro += [f"  st r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi = [f"  lda r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi += [N] * 7 + ["  paddb [sp], #-64", "  ret lr"] + [N] * 5
    return "\n".join(pro + body + epi) + "\n"


def _ts(i: int) -> str:
    hi, lo = TS_PAIRS[i]
    return f"  mov r{hi}:r{lo}, cntr"


def _store_ts(n: int, first_word: int) -> list[str]:
    return [f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * (first_word + i)}]" for i in range(n)]


def _topology(name: str, cons: str, prod: str, init: int,
              log_bytes: int = LOG_BYTES) -> tuple[Topology, dict[str, str]]:
    topo = Topology(
        name=name,
        cores=(Core("producer", PRODUCER_TILE, Path("producer.s"), args=(Buffer("ts", "out", TS_BYTES),)),
               Core("consumer", CONSUMER_TILE, Path("consumer.s"), args=(Buffer("log", "out", log_bytes),))),
        links=(SharedLock("consumer", "producer", LOCK_ID, init=init),))
    return topo, {"producer.s": prod, "consumer.s": cons}


# -- lk / same: a load in flight across a lock stall ---------------------------

MODES = ("stall", "free")


def _releaser(late_iters: int) -> str:
    """Producer of lk/same: wait `late_iters` iterations (0: none), stamp, release +1."""
    body = base.delay_loop(late_iters, ".Lprod") if late_iters else []
    body += ["  mov r1, #1", *[N] * 2, _ts(0), f"  rel #{RAW_SOUTH}, r1", *[N] * 8, *_store_ts(1, 0)]
    return _frame(body)


def _load_setup() -> list[str]:
    # r6 holds LDA_OLD until the load lands; the load reads LDA_NEW from word 15.
    return [f"  mov r5, #{LDA_NEW}", f"  mov r6, #{LDA_OLD}", "  mov r4, #-1", "  mov p2, p0",
            *[N] * 2, "  st r5, [p0, #60]", *[N] * 8]


def lk_program(d: int) -> list[str]:
    """Consumer measured window of lk_D: the load, then bundles +1..+8 (+D = the acquire)."""
    if not 1 <= d <= 8:
        raise ValueError("lk_D needs 1 <= D <= 8")
    body = [_ts(0), "  lda r6, [p0, #60]"]
    for j in range(1, 9):
        body.append(f"  acq #{RAW_OWN}, r4" if j == d else f"  mov r{7 + j}, r6")
    return body + [_ts(1)]


def same_program() -> list[str]:
    body = [_ts(0), f"  lda r6, [p0, #60] ; paddb [p2], #4 ; acq #{RAW_OWN}, r4"]
    return body + [f"  mov r{8 + j}, r6" for j in range(8)] + [_ts(1)]


def _sampler_asm(program: list[str], captured: list[int], late_iters: int, with_ptr: bool) -> str:
    body = _load_setup()
    if late_iters:
        body += base.delay_loop(late_iters, ".Lcons")
    body += program + [*[N] * 8]
    body += [f"  st r{7 + j}, [p0, #{4 * (j - 1)}]" for j in captured]
    body += _store_ts(2, 8)
    if with_ptr:
        body += ["  mov r1, p2", "  mov r2, p0", *[N] * 2, "  sub r1, r1, r2", *[N] * 2,
                 "  st r1, [p0, #40]"]
    return _frame(body)


def lk_case(d: int, mode: str, late_iters: int):
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    captured = [j for j in range(1, 9) if j != d]
    stall = mode == "stall"
    cons = _sampler_asm(lk_program(d), captured, 0 if stall else late_iters, False)
    return _topology(f"lk_{d}_{mode}", cons, _releaser(late_iters if stall else 0), 0)


def same_case(mode: str, late_iters: int):
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    stall = mode == "stall"
    cons = _sampler_asm(same_program(), list(range(1, 9)), 0 if stall else late_iters, True)
    return _topology(f"same_{mode}", cons, _releaser(late_iters if stall else 0), 0)


# -- vlk: a vector load in flight across a lock stall --------------------------------

VLK_KINDS = {
    "vlda": ("  vlda wl10, [p0, #32]", "wl10"),
    "vldb": ("  vldb wl10, [p0, #32]", "wl10"),
    "am": ("  vlda amll0, [p0, #32]", "amll0"),
}
# Layout: the two counter reads, the loaded data, the eight copies (`vldb` and `st` take
# immediate offsets below 128 only).
VLK_TS_WORD, VLK_DATA_AT, VLK_COPY_AT = 0, 32, 64
VLOG_BYTES = VLK_COPY_AT + 8 * 32


def _vlk_setup(kind: str) -> list[str]:
    # The target holds LDA_OLD in every word until the load lands; the load reads LDA_NEW.
    old = (["  vmov amll0, wl11"] if kind == "am" else ["  vmov wl10, wl11"])
    return [f"  mov r5, #{LDA_NEW}", f"  mov r6, #{LDA_OLD}", "  mov r4, #-1", *[N] * 2,
            "  vbcst.32 x9, r5", "  vbcst.32 x11, r6", *[N] * 2,
            f"  vst wl9, [p0, #{VLK_DATA_AT}]", *old, *[N] * 8]


def vlk_program(kind: str, d: int) -> list[str]:
    """Consumer measured window of vlk_KIND_D: the load, then bundles +1..+8 (+D = acquire)."""
    if not 1 <= d <= 8:
        raise ValueError("vlk needs 1 <= D <= 8")
    load, target = VLK_KINDS[kind]
    body = [_ts(0), load]
    for j in range(1, 9):
        body.append(f"  acq #{RAW_OWN}, r4" if j == d else f"  vmov wl{j - 1}, {target}")
    return body + [_ts(1)]


def vlk_case(kind: str, d: int, mode: str, late_iters: int):
    if kind not in VLK_KINDS:
        raise ValueError(f"kind is one of {tuple(VLK_KINDS)}")
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    stall = mode == "stall"
    body = _vlk_setup(kind)
    if not stall:
        body += base.delay_loop(late_iters, ".Lcons")
    body += vlk_program(kind, d) + [*[N] * 8]
    body += [f"  vst wl{j - 1}, [p0, #{VLK_COPY_AT + 32 * (j - 1)}]" for j in range(1, 9) if j != d]
    body += [f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * (VLK_TS_WORD + i)}]" for i in range(2)]
    cons = _frame(body)
    return _topology(f"vlk_{kind}_{d}_{mode}", cons, _releaser(late_iters if stall else 0), 0,
                     VLOG_BYTES)


def parse_vlk(consumer_log: bytes) -> dict:
    """Word 0 of each copy (every word of a copy is equal unless a copy tore)."""
    copies = [struct.unpack_from("<8I", consumer_log, VLK_COPY_AT + 32 * j) for j in range(8)]
    caps = [c[0] if len(set(c)) == 1 else None for c in copies]
    first = next((i + 1 for i, v in enumerate(caps) if v == LDA_NEW), None)
    w = struct.unpack_from(f"<{VLOG_BYTES // 4}I", consumer_log)
    return {"caps": caps, "new_from": first,
            "window": (w[VLK_TS_WORD + 1] - w[VLK_TS_WORD]) & 0xFFFFFFFF}


# -- pos / relpos: where in the pipeline a lock request stalls the core ---------------

def pos_case(kind: str, mode: str, late_iters: int):
    """`pos_MODE`: a counter read, `acq` (change -1), then a counter read in each of the
    next seven bundles: the first difference above 1 is where the core stops issuing (the
    refused acquire's wait in `stall`, its one arbitration cycle, if any, in `free`).
    `relpos`: the same around a `rel` +1 (one mode: a release never waits)."""
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    stall = mode == "stall"
    req = f"  acq #{RAW_OWN}, r4" if kind == "pos" else f"  rel #{RAW_OWN}, r1"
    body = ["  mov r4, #-1", "  mov r1, #1", *[N] * 2]
    if not stall and kind == "pos":
        body += base.delay_loop(late_iters, ".Lcons")
    body += [_ts(0), req, *[_ts(i) for i in range(1, 8)], *[N] * 8, *_store_ts(8, 0)]
    if kind == "pos":
        prod = _releaser(late_iters if stall else 0)
        name = f"pos_{mode}"
    else:
        prod = _frame([*[N] * 2])
        name = "relpos"
    return _topology(name, _frame(body), prod, 0)


def parse_pos(consumer_log: bytes) -> list[int]:
    """Cycle differences between consecutive counter reads (read 0 before the request)."""
    ts = struct.unpack_from("<8I", consumer_log)
    return [(ts[i + 1] - ts[i]) & 0xFFFFFFFF for i in range(7)]


# -- b2b: a second lock request inside the first's window ------------------------------

# Request 1 is on lock B2B_FIRST, request 2 on LOCK_ID; both locks live in the consumer's
# memory module and the producer reaches both (to release whichever acquire is refused).
B2B_FIRST = 14
B2B_PAIRS = ("relacq", "acqacq", "acqrel", "relrel")
# Variants: "gg" every acquire granted on issue; "gr" the second acquire refused until the
# producer releases it; "rg" the first acquire refused; "same" (relacq only) both requests
# on LOCK_ID, count 0: the release is what lets the acquire through, unless silicon takes
# the acquire first, in which case the producer's release lets it through late.
B2B_VARIANTS = {"relacq": ("gg", "gr", "same"), "acqacq": ("gg", "gr", "rg"),
                "acqrel": ("gg", "rg"), "relrel": ("gg",)}
B2B_MAX_D = 5
# Counter reads co-issue with every bundle, the requests included, but only the eight
# eL pairs r16..r31 take `mov rH:rL, cntr`: each case reads the counter in front of
# request 1 and in seven consecutive bundles of one window (offsets from request 1).
# Window "a" covers +0..+6, "b" +6..+12; both read +6, a determinism check across cases.
B2B_WINDOWS = {"a": 0, "b": 6}
B2B_LAST = 12                      # the program runs to request 1 + B2B_LAST, then nops


def _b2b_requests(pair: str) -> tuple[str, str]:
    return pair[:3], pair[3:]


def b2b_layout(window: str) -> list[int]:
    """Bundle offset from request 1 of each counter read of a b2b case."""
    first = B2B_WINDOWS[window]
    return [-1, *range(first, first + 7)]


def b2b_program(pair: str, variant: str, d: int, window: str) -> list[str]:
    """Consumer window of b2b: request 1 at offset 0, request 2 at +d, nops to +B2B_LAST,
    the counter read at the offsets of `b2b_layout(window)`."""
    if pair not in B2B_PAIRS or variant not in B2B_VARIANTS[pair]:
        raise ValueError(f"no b2b case {pair}_{variant}")
    if not 1 <= d <= B2B_MAX_D:
        raise ValueError(f"b2b needs 1 <= d <= {B2B_MAX_D}")
    k1, k2 = _b2b_requests(pair)
    lock1 = LOCK_ID if variant == "same" else B2B_FIRST
    regs = {"acq": "r3", "rel": "r2"}
    slots: dict[int, list[str]] = {0: [f"{k1} #{48 + lock1}, {regs[k1]}"],
                                   d: [f"{k2} #{48 + LOCK_ID}, {regs[k2]}"]}
    for i, off in enumerate(b2b_layout(window)):
        hi, lo = TS_PAIRS[i]
        slots.setdefault(off, []).append(f"mov r{hi}:r{lo}, cntr")
    return [("  " + " ; ".join(slots[o])) if o in slots else N for o in range(-1, B2B_LAST + 1)]


def b2b_inits(pair: str, variant: str) -> tuple[int, int]:
    """Initial counts of (B2B_FIRST, LOCK_ID)."""
    k1, k2 = _b2b_requests(pair)
    if variant == "same":
        return 0, 0
    first = 1 if k1 == "acq" and variant != "rg" else 0
    second = 1 if k2 == "acq" and variant != "gr" else 0
    return first, second


def b2b_refused_lock(variant: str) -> int | None:
    """The lock the producer releases (the refused acquire's), if any."""
    return {"gr": LOCK_ID, "rg": B2B_FIRST, "same": LOCK_ID}.get(variant)


# After the window each lock's count is read back without risking a hang: the consumer
# takes each lock twice (-1, first B2B_FIRST then LOCK_ID) once the window's release has
# landed, and the producer releases each lock twice later on, B2B_SPACING_ITERS apart. A
# test acquire that completes before the producer's first fallback release of its lock was
# granted from the count the window left (0, 1, or 2 = "2 or more").
B2B_COUNT_LOCKS = (B2B_FIRST, LOCK_ID)
B2B_SPACING_ITERS = 1000


def _two_lock_case(name: str, setup: list[str], prog: list[str], pairs: list[tuple[int, int]],
                   refused: int | None, inits: tuple[int, int], late_iters: int):
    """A consumer window `prog` (counter reads into `pairs`) followed by the count read-back
    of B2B_COUNT_LOCKS; the producer releases `refused` (if any) after `late_iters`."""
    body = [*setup, *[N] * 2, *prog, *[N] * 8]
    body += [f"  st r{lo}, [p0, #{4 * i}]" for i, (_, lo) in enumerate(pairs)]
    body += base.delay_loop(late_iters + B2B_SPACING_ITERS // 2, ".Lcons")
    for i, lock in enumerate(l for l in B2B_COUNT_LOCKS for _ in range(2)):
        body += [f"  acq #{48 + lock}, r3", *[N] * LOCK_SETTLE_BUNDLES, _ts(i)]
    body += [*[N] * 8, *[f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * (8 + i)}]" for i in range(4)]]
    prod = ["  mov r1, #1", *[N] * 2, *base.delay_loop(late_iters, ".Lprod"), _ts(0)]
    prod += [f"  rel #{refused}, r1"] if refused is not None else [N]
    for k, fb in enumerate(l for l in B2B_COUNT_LOCKS for _ in range(2)):
        gap = late_iters + 3 * B2B_SPACING_ITERS // 2 if k == 0 else B2B_SPACING_ITERS
        prod += [*base.delay_loop(gap, f".Lfb{k}"), _ts(k + 1), f"  rel #{fb}, r1"]
    prod += [*[N] * 8, *_store_ts(5, 0)]
    topo = Topology(
        name=name,
        cores=(Core("producer", PRODUCER_TILE, Path("producer.s"), args=(Buffer("ts", "out", TS_BYTES),)),
               Core("consumer", CONSUMER_TILE, Path("consumer.s"), args=(Buffer("log", "out", LOG_BYTES),))),
        links=(SharedLock("consumer", "producer", B2B_FIRST, init=inits[0]),
               SharedLock("consumer", "producer", LOCK_ID, init=inits[1])))
    return topo, {"producer.s": _frame(prod), "consumer.s": _frame(body)}


def b2b_case(pair: str, variant: str, d: int, window: str, late_iters: int):
    prog = b2b_program(pair, variant, d, window)
    return _two_lock_case(f"b2b_{pair}_{variant}_d{d}{window}", ["  mov r2, #1", "  mov r3, #-1"],
                          prog, TS_PAIRS, b2b_refused_lock(variant), b2b_inits(pair, variant),
                          late_iters)


def parse_b2b_counts(consumer_log: bytes, producer_ts: bytes) -> dict[int, int]:
    """Count each of B2B_COUNT_LOCKS held after the window (2 = 2 or more)."""
    tests = struct.unpack_from("<4I", consumer_log, 32)
    fb = struct.unpack_from("<5I", producer_ts)[1:]
    out = {}
    for j, lock in enumerate(B2B_COUNT_LOCKS):
        first_release = fb[2 * j]
        out[lock] = sum(((first_release - t) & 0xFFFFFFFF) < 1 << 31 for t in tests[2 * j:2 * j + 2])
    return out


def parse_b2b(window: str, consumer_log: bytes, producer_ts: bytes | None) -> dict:
    """Per counter read: its bundle offset and the cycles lost beyond one per bundle since
    the read in front of request 1 (0 0 0 0 1 ... : one cycle lost in front of +4).
    `resume`: the first read after the producer's release minus the producer's read in
    front of its `rel` (as the lk/pos resume numbers)."""
    layout = b2b_layout(window)
    ts = struct.unpack_from("<8I", consumer_log)
    lost = [((ts[i] - ts[0]) & 0xFFFFFFFF) - (layout[i] - layout[0]) for i in range(8)]
    out = {"layout": layout, "lost": lost, "resume": None}
    if producer_ts is not None:
        (p,) = struct.unpack_from("<I", producer_ts)
        after = [(t - p) & 0xFFFFFFFF for t in ts if 0 < ((t - p) & 0xFFFFFFFF) < 1 << 31]
        out["resume"] = min(after) if after else None
    return out


# -- cond: acq.cond / rel.cond, the request performed or skipped by r26 ----------------

# Condition values: 0, 1, nonzero with bit 0 clear, a middle bit, the top bit, all ones,
# all ones but bit 0 -- whether the test is the whole register or one bit of it.
COND_VALUES = {"z": 0, "o": 1, "t": 2, "h": 0x100, "m": 0x80000000, "f": 0xFFFFFFFF,
               "e": 0xFFFFFFFE}
COND_STATES = {"acq": ("free", "stall"), "rel": ("rel",)}
# r26 holds the condition, so the counter pair (r27:r26) is not used by cond cases.
COND_PAIRS = [p for i, p in enumerate(TS_PAIRS) if i != 5]
COND_LAYOUT = [-1, 0, 1, 2, 3, 4, 5]     # bundle offset from the request of each read
COND_LAST = 8


def cond_layout(writer: tuple[int, int] | None = None) -> list[int]:
    """Bundle offsets of the cond counter reads: COND_LAYOUT, or with a r26 writer (its
    `mov` shares the move slot with a counter read) a baseline at -3 and the first six of
    -2..+5 that the writer's bundle leaves free."""
    if writer is None:
        return COND_LAYOUT
    return [-3] + [o for o in range(-2, 6) if o != writer[0]][:6]


def cond_program(kind: str, writer: tuple[int, int] | None = None) -> list[str]:
    """Consumer window of cond: `kind`.cond at offset 0, counter reads at COND_LAYOUT.
    `writer` = (offset, value): a `mov r26, #value` at that bundle (-2, -1, 0 = with the
    request): does the request see the old or the new r26?"""
    if kind not in COND_STATES:
        raise ValueError(f"cond kind is one of {tuple(COND_STATES)}")
    slots: dict[int, list[str]] = {0: [f"{kind}.cond #{48 + LOCK_ID}, {'r3' if kind == 'acq' else 'r2'}, r26"]}
    layout = cond_layout(writer)
    for (hi, lo), off in zip(COND_PAIRS, layout):
        slots.setdefault(off, []).append(f"mov r{hi}:r{lo}, cntr")
    if writer is not None:
        slots.setdefault(writer[0], []).append(f"mov r26, #{writer[1]}")
    return [("  " + " ; ".join(slots[o])) if o in slots else N for o in range(layout[0], COND_LAST + 1)]


def cond_case(kind: str, state: str, vkey: str, late_iters: int, writer: tuple[int, int] | None = None):
    if state not in COND_STATES.get(kind, ()):
        raise ValueError(f"no cond case {kind}_{state}")
    value = COND_VALUES[vkey]
    init = 1 if state == "free" else 0
    body = ["  mov r2, #1", "  mov r3, #-1", f"  movxm r26, #{value}", *[N] * 3,
            *cond_program(kind, writer), *[N] * 8]
    body += [f"  st r{lo}, [p0, #{4 * i}]" for i, (_, lo) in enumerate(COND_PAIRS)]
    body += base.delay_loop(late_iters + B2B_SPACING_ITERS // 2, ".Lcons")
    for i in range(2):
        body += [f"  acq #{RAW_OWN}, r3", *[N] * LOCK_SETTLE_BUNDLES, _ts(i)]
    body += [*[N] * 8, *[f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * (8 + i)}]" for i in range(2)]]
    prod = ["  mov r1, #1", *[N] * 2, *base.delay_loop(late_iters, ".Lprod"), _ts(0)]
    prod += [f"  rel #{RAW_SOUTH}, r1"] if state == "stall" else [N]
    for k in range(2):
        gap = late_iters + 3 * B2B_SPACING_ITERS // 2 if k == 0 else B2B_SPACING_ITERS
        prod += [*base.delay_loop(gap, f".Lfb{k}"), _ts(k + 1), f"  rel #{RAW_SOUTH}, r1"]
    prod += [*[N] * 8, *_store_ts(3, 0)]
    name = f"cond_{kind}_{state}_{vkey}" + (f"_w{-writer[0]}to{writer[1]}" if writer else "")
    return _topology(name, _frame(body), _frame(prod), init)


def parse_cond(consumer_log: bytes, producer_ts: bytes, layout: list[int] = COND_LAYOUT) -> dict:
    """Cycles lost beyond one per bundle at each read since the one in front of the request,
    and the count the lock held after the window (0, 1, or 2 = 2 or more)."""
    ts = struct.unpack_from("<7I", consumer_log)
    lost = [((ts[i] - ts[0]) & 0xFFFFFFFF) - (layout[i] - layout[0]) for i in range(7)]
    tests = struct.unpack_from("<2I", consumer_log, 32)
    first_release = struct.unpack_from("<3I", producer_ts)[1]
    count = sum(((first_release - t) & 0xFFFFFFFF) < 1 << 31 for t in tests)
    return {"lost": lost, "count": count}


# condwin: a skipped (or performed) conditional request next to a real one. B2B_FIRST holds
# the conditional request, LOCK_ID the plain release; does a skipped request open a window
# the other joins (the real release then resolves in front of +4), or is it nothing at all
# (in front of +d+4)? Reads: in front of the first request and six consecutive bundles.
CONDWIN_VARIANTS = {
    "sk_re": ("acq.cond", 0, "first"),    # skipped acq.cond, then a plain rel at +d
    "skr_re": ("rel.cond", 0, "first"),   # skipped rel.cond, then a plain rel at +d
    "cp_re": ("rel.cond", 1, "first"),    # performed rel.cond, then a plain rel at +d
    "re_sk": ("acq.cond", 0, "second"),   # a plain rel, then a skipped acq.cond at +d
}
CONDWIN_WINDOWS = {"a": [-1, 0, 1, 2, 3, 4, 5], "b": [-1, 4, 5, 6, 7, 8, 9]}
CONDWIN_MAX_D = 3


def condwin_program(variant: str, d: int, window: str) -> list[str]:
    cond, _, which = CONDWIN_VARIANTS[variant]
    reg = "r3" if cond == "acq.cond" else "r2"
    conditional = f"{cond} #{48 + B2B_FIRST}, {reg}, r26"
    plain = f"rel #{48 + LOCK_ID}, r2"
    first, second = (conditional, plain) if which == "first" else (plain, conditional)
    slots: dict[int, list[str]] = {0: [first], d: [second]}
    for (hi, lo), off in zip(COND_PAIRS, CONDWIN_WINDOWS[window]):
        slots.setdefault(off, []).append(f"mov r{hi}:r{lo}, cntr")
    return [("  " + " ; ".join(slots[o])) if o in slots else N for o in range(-1, B2B_LAST + 1)]


def condwin_case(variant: str, d: int, window: str, late_iters: int):
    if variant not in CONDWIN_VARIANTS or not 1 <= d <= CONDWIN_MAX_D:
        raise ValueError(f"no condwin case {variant} d{d}")
    cond, value, _ = CONDWIN_VARIANTS[variant]
    # an acq.cond that is performed would take lock 14 from 1 to 0; a rel.cond, 0 to 1
    inits = (1 if cond == "acq.cond" else 0, 0)
    setup = ["  mov r2, #1", "  mov r3, #-1", f"  movxm r26, #{value}", *[N] * 3]
    return _two_lock_case(f"condwin_{variant}_d{d}{window}", setup,
                          condwin_program(variant, d, window), COND_PAIRS, None, inits, late_iters)


# -- eq: acquires with a change >= 0 -----------------------------------------------

Op = tuple[str, int]               # ("acq" | "rel", change)


@dataclass(frozen=True)
class EqCase:
    init: int
    consumer: tuple[Op, ...]
    producer: tuple[Op, ...]


EQ_CASES: dict[str, EqCase] = {
    # count == change already: granted at once? and is the count left alone (then the
    # -1 acquire after it is granted at once too)?
    "c1v1": EqCase(1, (("acq", 1), ("acq", -1)), (("rel", 1), ("rel", 1))),
    # count below the change: granted only once releases raise it to the change
    "c2v1": EqCase(1, (("acq", 2), ("acq", -1)), (("rel", 1), ("rel", 1), ("rel", 1))),
    # change 0 on a free lock (count 0)
    "c0v0": EqCase(0, (("acq", 0), ("acq", -1)), (("rel", 1), ("rel", 1))),
    # change 0 on a held lock (count 1): granted only once the producer takes it to 0
    "c0v1": EqCase(1, (("acq", 0), ("rel", 1)), (("acq", -1),)),
    # count above the change
    "c1v2": EqCase(2, (("acq", 1), ("acq", -1), ("rel", 1)), (("acq", -1), ("rel", 1))),
}

MAX_LOCK = 63


def rule_grants(rule: str, count: int, change: int) -> int | None:
    """New count if an acquire of `change` on `count` is granted under `rule`, else None.

    eq     silicon's host Lock_Request path: change < 0 needs count >= -change and
           subtracts; change >= 0 needs count == change and leaves it unchanged
    ge     the EMU core path before this probe: every change is greater-or-equal
           (change > 0 needs count >= change, change 0 needs count >= 1), subtracting
    eqdec  the EMU DMA path: change > 0 needs count == change and subtracts it,
           change 0 is greater-or-equal 1
    """
    if change < 0:
        return count + change if count >= -change else None
    if rule == "eq":
        return count if count == change else None
    need = change if change > 0 else 1
    if rule == "ge":
        return count - need if count >= need else None
    if rule == "eqdec":
        if change > 0:
            return 0 if count == change else None
        return count - 1 if count >= 1 else None
    raise ValueError(f"unknown rule {rule!r}")


RULES = ("eq", "ge", "eqdec")


class Deadlock(Exception):
    pass


def simulate(case: EqCase, rule: str) -> list[int]:
    """Producer step at which each consumer request completes (0 = before the first).

    The producer's requests are `step_iters` loop iterations apart, so between two of
    them the consumer runs until a request of its own blocks. The consumer is then
    blocked whenever the producer issues, so a producer acquire that is refused can
    never be granted: that is a deadlock, and a case must have none under any rule.
    Releases must keep the count within 0..MAX_LOCK (an overflowed lock refuses
    everything after it).
    """
    count = case.init
    done: list[int] = []

    def request(kind: str, ch: int) -> bool:
        nonlocal count
        if kind == "rel":
            count += ch
            if not 0 <= count <= MAX_LOCK:
                raise ValueError(f"release leaves the lock range under {rule}")
            return True
        new = rule_grants(rule, count, ch)
        if new is None:
            return False
        count = new
        return True

    def run_consumer(step: int) -> None:
        while len(done) < len(case.consumer) and request(*case.consumer[len(done)]):
            done.append(step)

    run_consumer(0)
    for step, op in enumerate(case.producer, start=1):
        if not request(*op):
            raise Deadlock(f"producer request {step} refused under {rule}")
        run_consumer(step)
    if len(done) < len(case.consumer):
        raise Deadlock(f"consumer blocked at request {len(done) + 1} under {rule}")
    return done


def _change_regs(ops: tuple[Op, ...]) -> dict[int, str]:
    regs: dict[int, str] = {}
    for _, ch in ops:
        if ch not in regs:
            regs[ch] = f"r{len(regs) + 1}"
    if len(regs) > 7:
        raise ValueError("at most 7 distinct changes")
    return regs


def eq_consumer_asm(case: EqCase) -> str:
    if len(case.consumer) > len(TS_PAIRS) - 1:
        raise ValueError(f"at most {len(TS_PAIRS) - 1} consumer requests")
    regs = _change_regs(case.consumer)
    body = [f"  mov {r}, #{ch}" for ch, r in regs.items()] + [*[N] * 2, _ts(0)]
    for i, (kind, ch) in enumerate(case.consumer):
        body += [f"  {kind} #{RAW_OWN}, {regs[ch]}", *[N] * LOCK_SETTLE_BUNDLES, _ts(i + 1)]
    body += [*[N] * 8, *_store_ts(len(case.consumer) + 1, 0)]
    return _frame(body)


def eq_producer_asm(case: EqCase, step_iters: int) -> str:
    if len(case.producer) > len(TS_PAIRS):
        raise ValueError(f"at most {len(TS_PAIRS)} producer requests")
    regs = _change_regs(case.producer)
    body = [f"  mov {r}, #{ch}" for ch, r in regs.items()] + [*[N] * 2]
    for k, (kind, ch) in enumerate(case.producer):
        body += base.delay_loop(step_iters, f".Lstep{k}")
        body += [_ts(k), f"  {kind} #{RAW_SOUTH}, {regs[ch]}"]
    body += [*[N] * 8, *_store_ts(len(case.producer), 0)]
    return _frame(body)


def eq_case(name: str, step_iters: int):
    case = EQ_CASES[name]
    return _topology(f"eq_{name}", eq_consumer_asm(case), eq_producer_asm(case, step_iters), case.init)


# -- frozen: requests that overflow / underflow a lock, read back over control packets ---

# One core (0,2) issues lock requests on lock FZ_P, gated by FZ_G so the host reads a clean
# baseline first. A timer core (0,3), gated by FZ_GT, later releases FZ_C1 and FZ_C2, which
# let the tile DMA ship the core's `stamps` buffer to the host twice: pass 1 shows how far
# the core got (its progress marker and, once it finished, its counter reads) before the host
# reads the flags; pass 2, much later, after the host cleared them. Nothing here waits for the
# core, so a core held forever by a frozen lock cannot hang the run.
FZ_P, FZ_Q, FZ_G, FZ_C1, FZ_C2, FZ_D, FZ_GT = 15, 14, 13, 12, 11, 10, 9
FZ_D2, FZ_D3 = 8, 7                      # DMA cases: release targets of the second and third BD
FZ_SENTINEL = 0x5E5E5E5E
FZ_PROGRESS = (FZ_SENTINEL, 0xA1A1A1A1, 0xA2A2A2A2, 0xA3A3A3A3, 0x600D600D)
FZ_STAMP_WORDS = 16
FZ_PASS1_ITERS = 20000                   # timer: ~0.2 ms after the go write
FZ_PASS2_ITERS = 6 * 65536               # then ~4 ms (several maximal delay loops)
FZ_READ_OFFSETS = (-1, 3, 4, 5)          # counter reads around each request (bundles from it)


@dataclass(frozen=True)
class FrozenCase:
    init: int                            # FZ_P's count at load
    reqs: tuple[Op, ...]                 # one or two ("rel" | "acq", change) requests on FZ_P
    keep: bool = False                   # never clear the flags: do they outlive the run?
    dma: str = ""                        # "rel" | "acq": a BD chain on FZ_P instead of core requests
    init_q: int = 0                      # FZ_Q's count at load (b2b cases)
    b2b: tuple[str, str, int] | None = None  # (pair, variant, d): a lock-b2b window instead
    chain: int = 3                       # dma "rel": BDs in the chain (BDs 4.. gate on their own lock)


FROZEN_CASES: dict[str, FrozenCase] = {
    "base":   FrozenCase(5, ()),                              # no request: the plumbing alone
    "fit_hi": FrozenCase(62, (("rel", 1),)),                  # lands on 63: no flag
    "fit_lo": FrozenCase(5, (("rel", -5),)),                  # lands on 0: no flag
    "fit63":  FrozenCase(0, (("rel", 63),)),
    "ovf1":   FrozenCase(63, (("rel", 1),)),
    "ovf5":   FrozenCase(62, (("rel", 5),)),
    "unf1":   FrozenCase(0, (("rel", -1),)),
    "unf3":   FrozenCase(1, (("rel", -3),)),
    "ovf_rel": FrozenCase(63, (("rel", 1), ("rel", 1))),      # a release on the frozen lock
    "ovf_acq": FrozenCase(63, (("rel", 2), ("acq", -1))),     # count 1 and frozen: an acquire
    "unf_rel": FrozenCase(0, (("rel", -1), ("rel", 1))),
    "unf_acq": FrozenCase(0, (("rel", -1), ("acq", -1))),     # count 63 and frozen: an acquire
    # DMA BD path (three-BD chain, see frozen_design): BD1 overflows FZ_P with its release,
    # BD2's release meets the frozen lock, BD3 shows whether BD2 completed
    "dma_rel": FrozenCase(63, (), dma="rel"),
    # the core overflows FZ_P to count 1, BD2 acquires it (>= 1) while it is frozen
    "dma_acq": FrozenCase(63, (("rel", 2),), dma="acq"),
    # BDs behind the one that meets the frozen lock: how far does the channel run ahead?
    **{f"dma_rel_c{n}": FrozenCase(63, (), dma="rel", chain=n) for n in (4, 5, 6)},
    "ovf1_keep": FrozenCase(63, (("rel", 1),), keep=True),    # leaves the flag set after the run
    "unf1_keep": FrozenCase(0, (("rel", -1),), keep=True),
}


def _own(lock: int) -> int:
    return 48 + lock


def _b2b_frozen_cases() -> dict[str, "FrozenCase"]:
    """The refused-acquire-in-a-window cases of lock-b2b (window a, d = 1..3), read back."""
    out = {}
    for pair, variants in B2B_VARIANTS.items():
        for variant in variants:
            if b2b_refused_lock(variant) is None:
                continue
            q, p = b2b_inits(pair, variant)
            for d in (1, 2, 3):
                out[f"b2b_{pair}_{variant}_d{d}"] = FrozenCase(p, (), init_q=q, b2b=(pair, variant, d))
    return out


def frozen_core_asm(case: FrozenCase) -> str:
    """p0 = stamps. Words 0..7: counter reads at FZ_READ_OFFSETS around each request, word 8:
    the progress marker (FZ_PROGRESS). The marker moves before each request, so a core held by
    one is seen at the marker of that request."""
    if len(case.reqs) > 2:
        raise ValueError("at most two requests (eight counter reads)")
    body = ["  mov r4, #-1", *[f"  mov r{2 + k}, #{ch}" for k, (_, ch) in enumerate(case.reqs)],
            f"  movxm r5, #{FZ_SENTINEL}", *[N] * 3,
            *[f"  st r5, [p0, #{4 * i}]" for i in range(FZ_STAMP_WORDS)], *[N] * 8,
            f"  movxm r6, #{FZ_PROGRESS[1]}", *[N] * 3, "  st r6, [p0, #32]", *[N] * 8,
            f"  acq #{_own(FZ_G)}, r4", *[N] * LOCK_SETTLE_BUNDLES,
            f"  movxm r6, #{FZ_PROGRESS[2]}", *[N] * 3, "  st r6, [p0, #32]", *[N] * 8]
    if case.b2b:
        pair, variant, d = case.b2b
        body += ["  mov r2, #1", "  mov r3, #-1", *[N] * 2, *b2b_program(pair, variant, d, "a"), *[N] * 8,
                 *[f"  st r{lo}, [p0, #{4 * i}]" for i, (_, lo) in enumerate(TS_PAIRS)],
                 f"  movxm r6, #{FZ_PROGRESS[4]}", *[N] * 3, "  st r6, [p0, #32]", *[N] * 8]
        return _frame(body)
    for k, (kind, _) in enumerate(case.reqs):
        if k:
            body += [f"  movxm r6, #{FZ_PROGRESS[3]}", *[N] * 3, "  st r6, [p0, #32]", *[N] * 8]
        slots: dict[int, list[str]] = {0: [f"{kind} #{_own(FZ_P)}, r{2 + k}"]}
        for j, off in enumerate(FZ_READ_OFFSETS):
            hi, lo = TS_PAIRS[4 * k + j]
            slots.setdefault(off, []).append(f"mov r{hi}:r{lo}, cntr")
        body += [("  " + " ; ".join(slots[o])) if o in slots else N
                 for o in range(FZ_READ_OFFSETS[0], FZ_READ_OFFSETS[-1] + 1)]
        body += [*[N] * 6]
    body += [*[N] * 8, *[f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * i}]" for i in range(4 * len(case.reqs))],
             f"  movxm r6, #{FZ_PROGRESS[4]}", *[N] * 3, "  st r6, [p0, #32]", *[N] * 8]
    return _frame(body)


def _long_delay(iters: int, tag: str) -> list[str]:
    """`iters` loop iterations (~10 cycles each) in as many maximal delay loops as needed."""
    out, k = [], 0
    while iters > 0:
        n = min(iters, base.MAX_LOOP_ITERS)
        out += base.delay_loop(n, f".L{tag}{k}")
        iters -= n
        k += 1
    return out


def frozen_timer_asm(second: bool = True) -> str:
    """Core (0,3): gated by FZ_GT, releases FZ_C1 and, if `second`, much later FZ_C2 (south
    quadrant raw IDs)."""
    body = ["  mov r4, #-1", "  mov r1, #1", *[N] * 2,
            f"  acq #{FZ_GT}, r4", *[N] * LOCK_SETTLE_BUNDLES,
            *_long_delay(FZ_PASS1_ITERS, "pass1"), f"  rel #{FZ_C1}, r1", *[N] * 8]
    if second:
        body += [*_long_delay(FZ_PASS2_ITERS, "pass2"), f"  rel #{FZ_C2}, r1", *[N] * 8]
    return _frame(body)


FROZEN_CASES.update(_b2b_frozen_cases())


def frozen_passes(case: FrozenCase) -> int:
    """Stamps transfers the design makes: two timed passes, or one per BD of a DMA chain."""
    if case.dma == "rel":
        return case.chain
    return 3 if case.dma else 2


def frozen_steps(case: FrozenCase) -> list:
    """Host steps: baseline, go, pass 1, flags, clear, re-read, pass 2, flags, clear, re-read
    (DMA cases: see the comments in the branches)."""
    import ctrlpkt_readback as cp
    lv = cp.lock_value_addr
    extra = ((FZ_D, FZ_D2, FZ_D3) if case.dma else ()) + tuple(
        x for pair in FZ_EXTRA_BDS[:max(case.chain - 3, 0)] for x in pair)

    def snap(tag: str) -> list:
        return [cp.Read(cp.LOCKS_OVERFLOW, f"{tag} ovf"), cp.Read(cp.LOCKS_UNDERFLOW, f"{tag} unf"),
                cp.Read(lv(FZ_P), f"{tag} L{FZ_P}"), cp.Read(lv(FZ_Q), f"{tag} L{FZ_Q}"),
                cp.Read(lv(FZ_G), f"{tag} L{FZ_G}"),
                *[cp.Read(lv(x), f"{tag} L{x}") for x in extra],
                cp.Read(cp.MEM_EVENT_STATUS3, f"{tag} ev3")]

    def clear() -> list:
        return [cp.Write(cp.LOCKS_OVERFLOW, 0xFFFF), cp.Write(cp.LOCKS_UNDERFLOW, 0xFFFF)]

    go = [*snap("s0"), cp.Write(lv(FZ_G), 1), cp.Write(lv(FZ_GT), 1), cp.Wait("stamp0")]
    if case.b2b:
        # the window is over: what became of the refused acquire? s1 = the state it left (flags,
        # counts). Setting the refused lock's count to 1 lets a request still waiting on it
        # through (count 0 afterwards) while one that was dropped leaves the 1 (s2).
        refused = b2b_refused_lock(case.b2b[1])
        return [*go, *snap("s1"), cp.Write(lv(refused), 1), *snap("s2"), *clear(), cp.Wait("stamp0"),
                *snap("s3"), *clear(), *snap("s4")]
    if case.dma == "rel":
        # pass 1: BD1 sent and released FZ_P (+1 on 63: overflow). Open BD2 (host-written gate):
        # its data arrives, then its release meets the frozen lock; BD3 runs only if that
        # release completed (FZ_D3 counts it). Clear, then BD3 must run.
        return [*go, *snap("s1"), cp.Write(lv(FZ_C2), 1), cp.Wait("stamp0"), *snap("s2"),
                *clear(), *snap("s3"), *[cp.Wait("stamp0")] * (case.chain - 2), *snap("s4"), *clear(),
                *snap("s5")]
    if case.dma == "acq":
        # the core froze FZ_P at count 1; BD2 acquires it right after BD1 (timer-gated)
        return [*go, *snap("s1"), *clear(), *snap("s2"), cp.Wait("stamp0"), cp.Wait("stamp0"),
                *snap("s3"), *clear(), *snap("s4")]
    if case.keep:
        return [*go, *snap("s1"), *snap("s2"), cp.Wait("stamp0"), *snap("s3"), *snap("s4")]
    return [*go, *snap("s1"), *clear(), *snap("s2"), cp.Wait("stamp0"), *snap("s3"), *clear(), *snap("s4")]


FZ_X = 6                                 # DMA cases: a release target nobody reads
# BDs 4.. of a longer "rel" chain: (gate lock, witness lock) of BD n; gates start at 1, witnesses at 0
FZ_EXTRA_BDS = ((5, 2), (4, 1), (3, 0))


def frozen_chain(case: FrozenCase) -> list[tuple[str, str]]:
    """The tile DMA's BD chain as (acquire lock, release lock) symbols; each acquires >= 1 and
    releases +1. Core cases ship the stamps twice (timer-gated). DMA cases: see FrozenCase."""
    if case.dma == "rel":
        return [("lC1", "lP"), ("lC2", "lP"), ("lD", "lD3"),
                *[(f"lG{k}", f"lW{k}") for k in range(4, case.chain + 1)]]
    if case.dma == "acq":
        return [("lC1", "lX"), ("lP", "lD2"), ("lD", "lD3")]
    return [("lC1", "lD"), ("lC2", "lD")]


def frozen_design(case: FrozenCase) -> tuple[str, list, "object"]:
    """(aie.mlir text, host steps, packet layout)."""
    import ctrlpkt_readback as cp
    steps = frozen_steps(case)
    layout = cp.encode(steps)
    passes = frozen_passes(case)
    chain = frozen_chain(case)
    assert len(chain) == passes
    w = FZ_STAMP_WORDS
    m = f"memref<{w}xi32>"
    seq = [f"      aiex.npu.dma_memcpy_nd(%data[0, 0, 0, {w * i}][1, 1, 1, {w}][0, 0, 0, 1]) "
           f"{{id = {14 - i} : i64, issue_token = true, metadata = @stamp0}} : memref<{passes * w}xi32>"
           for i in range(passes)]
    seq += cp.sequence_lines(steps, layout, "ctrlout", "ctrlin")

    def bd(i: int, acq: str, rel: str, nxt: str) -> str:
        return f"""      %a{i} = arith.constant 1 : i32
      aie.use_lock(%{acq}, AcquireGreaterEqual, %a{i})
      aie.dma_bd(%stamps : {m})
      %r{i} = arith.constant 1 : i32
      aie.use_lock(%{rel}, Release, %r{i})
      aie.next_bd {nxt}"""
    blocks = []
    for i, (acq, rel) in enumerate(chain):
        blocks += [f"    ^b{i + 1}:", bd(i, acq, rel, f"^b{i + 2}" if i + 1 < len(chain) else "^end")]
    dma = bool(case.dma)
    lock_defs = [(FZ_P, "lP", case.init), (FZ_Q, "lQ", case.init_q), (FZ_G, "lG", 0), (FZ_GT, "lGT", 0),
                 (FZ_C1, "lC1", 0), (FZ_C2, "lC2", 0), (FZ_D, "lD", 1 if dma else 0)]
    if dma:
        lock_defs += [(FZ_D2, "lD2", 0), (FZ_D3, "lD3", 0), (FZ_X, "lX", 0)]
        for k, (gate, witness) in enumerate(FZ_EXTRA_BDS[:max(case.chain - 3, 0)], start=4):
            lock_defs += [(gate, f"lG{k}", 1), (witness, f"lW{k}", 0)]
    text = "\n".join([
        "module {",
        "  aie.device(npu1_1col) {",
        "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}",
        f"    %t02 = aie.tile(0, 2) {{controller_id = #aie.packet_info<pkt_type = 0, pkt_id = {cp.CTRL_PKT_ID}>}}",
        "    %t03 = aie.tile(0, 3)",
        f'    %stamps = aie.buffer(%t02) {{sym_name = "stamps"}} : {m}',
        *[f'    %{sym} = aie.lock(%t02, {lid}) {{init = {init} : i32, sym_name = "{sym}"}}'
          for lid, sym, init in lock_defs],
        "    aie.flow(%t02, DMA : 0, %t00, DMA : 1)",
        *cp.flow_lines("%t02", "%t00"),
        "    aie.shim_dma_allocation @stamp0 (%t00, S2MM, 1)",
        f"    func.func private @k_core({m}) attributes {{link_with = \"core.o\"}}",
        "    func.func private @k_timer() attributes {link_with = \"timer.o\"}",
        "    aie.core(%t02) {",
        f"      func.call @k_core(%stamps) : ({m}) -> ()",
        "      aie.end", "    }",
        "    aie.core(%t03) {", "      func.call @k_timer() : () -> ()", "      aie.end", "    }",
        "    %mem = aie.mem(%t02) {",
        "      %s0 = aie.dma_start(MM2S, 0, ^b1, ^end)",
        *blocks,
        "    ^end:", "      aie.end", "    }",
        f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
        f"%ctrlin: memref<{len(layout.words)}xi32>, %data: memref<{passes * w}xi32>) {{",
        *seq, "    }", "  }", "}", ""])
    return text, steps, layout


def parse_frozen(out: bytes, labels: list[str], passes: int = 2) -> dict:
    """Split a host output file: the labelled ctrl responses, then the stamps passes."""
    n, w = len(labels), FZ_STAMP_WORDS
    words = struct.unpack(f"<{n + passes * w}I", out[:4 * (n + passes * w)])
    r = {"ctrl": dict(zip(labels, words[:n]))}
    for i in range(passes):
        r[f"pass{i + 1}"] = list(words[n + i * w:n + (i + 1) * w])
    return r


def frozen_lost(stamps: list[int], n_reqs: int) -> list[list[int | None]]:
    """Per request: cycles beyond one per bundle at each of FZ_READ_OFFSETS[1:] since the read
    in front of it (None: that read was never made, the sentinel is still there)."""
    out = []
    for k in range(n_reqs):
        ts = stamps[4 * k:4 * k + 4]
        if FZ_SENTINEL in ts:
            out.append([None] * 3)
            continue
        out.append([((ts[j] - ts[0]) & 0xFFFFFFFF) - (FZ_READ_OFFSETS[j] - FZ_READ_OFFSETS[0])
                    for j in range(1, 4)])
    return out


# -- left: lock flags on the memtile and the shim, host requests and DMA BDs -----------------
#
# No cores. The host talks to the memtile's and the shim's own control port (the shim's by a
# packet flow from its DMA 0 into its TileControl, a loop through its switch): it sets lock
# values, issues lock requests (a read in the Lock_Request space), reads the flag registers,
# Event_Status and the DMA channel status. "host" cases need nothing else; "bd" cases also
# start one DMA task on a lock the host froze (overflowed or underflowed) beforehand, between
# two snapshots, so the channel's status says whether it is held on the lock.

LEFT_LOCKS = {"shim": (15, 14), "mem": (40, 5)}   # (overflow lock, underflow lock) of the host cases
LEFT_BD_LOCK = {"shim": 15, "mem": 40}
# A BD uses either no lock or one acquire and one release, so the frozen lock is paired with an
# auxiliary one: a release BD acquires it first (host-set to 1), an acquire BD releases it after
# (starts at 0): its value says how far the BD got.
LEFT_AUX_LOCK = {"shim": 14, "mem": 41}
# Release cases chain three more BDs behind the one that meets the frozen lock: each acquires a
# gate lock (host-set to 1) and releases a witness lock, so a gate at 0 says the channel moved
# on to that BD and a witness at 1 says its release was performed. LEFT_PROBE_LOCK is an
# unfrozen lock the host requests while a release is parked.
LEFT_GATE_LOCKS = {"shim": (13, 12, 11, 10, 9, 8), "mem": (42, 43, 44, 45, 46, 47)}
LEFT_WITNESS_LOCKS = {"shim": (6, 5, 4, 3, 2, 1), "mem": (48, 49, 50, 51, 52, 53)}
LEFT_PROBE_LOCK = {"shim": 7, "mem": 54}
LEFT_DMAERR_CHANNEL, LEFT_DMAERR_BD = 0, 3        # memtile S2MM 0, a BD no design configures
LEFT_WORDS = 16                                    # words a BD moves
LEFT_CHAIN = 4                                     # BDs of a release case's chain, by default
LEFT_MAX_CHAIN = 7
LEFT_MODULE = {"shim": "shim", "mem": "memory_tile"}
LEFT_DMA_STATUS = {"shim": ("DMA_S2MM_Status_1", "DMA_MM2S_Status_1"),
                   "mem": ("DMA_S2MM_Status_0", "DMA_MM2S_Status_0")}


@dataclass(frozen=True)
class LeftCase:
    kind: str                            # "shim" | "mem": the tile kind under test
    mode: str                            # "host" | "bd" | "dmaerr" (an invalid BD queued on a channel)
    direction: str = ""                  # bd: "mm2s" | "s2mm", the channel whose BD takes the lock
    act: str = ""                        # bd: "rel" (release meets the frozen lock) | "acq"
    chain: int = 1                       # bd: BDs on the locked channel (rel: the frozen one + witnesses)
    hold: int = 0                        # rel: index (>= 1) of the chain BD whose gate stays 0 until after the clear

    @property
    def data_words(self) -> int:
        """The data BO: source BDs, then room for them."""
        return 2 * self.chain * LEFT_WORDS


def _left_cases() -> dict[str, LeftCase]:
    out = {}
    for kind in ("shim", "mem"):
        out[f"{kind}_host"] = LeftCase(kind, "host")
        for direction in ("mm2s", "s2mm"):
            out[f"{kind}_{direction}_acq"] = LeftCase(kind, "bd", direction, "acq")
            out[f"{kind}_{direction}_rel"] = LeftCase(kind, "bd", direction, "rel", LEFT_CHAIN)
            # how many BDs run ahead of a parked release: sweep the chain length
            for n in range(1, LEFT_MAX_CHAIN + 1):
                if n != LEFT_CHAIN:
                    out[f"{kind}_{direction}_rel_c{n}"] = LeftCase(kind, "bd", direction, "rel", n)
            # where does the channel stop? one BD's gate is held shut until the flag is cleared
            for h in (2, 3, 4):
                out[f"{kind}_{direction}_rel_c5_hold{h}"] = LeftCase(kind, "bd", direction, "rel", 5, h)
    # Does an ordinary xclbin forward a DMA error event to the driver? Queue an unconfigured BD
    # on a memtile channel (the shim channel's error latches past a context load: not probed)
    out["mem_dmaerr"] = LeftCase("mem", "dmaerr", "s2mm")
    return out


LEFT_CASES = _left_cases()


def left_targets(case: LeftCase) -> list[str]:
    """Control targets the case reads: the tile under test, plus the shim for bd cases (its DMA
    status is the other end of every transfer)."""
    return [case.kind] if case.mode == "host" or case.kind == "shim" else [case.kind, "shim"]


def left_steps(case: LeftCase) -> list:
    import ctrlpkt_readback as cp
    regs = {t: cp.TileRegs(t) for t in left_targets(case)}
    k = regs[case.kind]
    status = {t: [(n, regs[t].regs[n]) for n in LEFT_DMA_STATUS[t]] for t in regs}

    def snap(tag: str, locks: tuple[int, ...]) -> list:
        out = []
        for t, r in regs.items():
            out += [cp.Read(a, f"{tag} {t} ovf{i}", t) for i, a in enumerate(r.overflow)]
            out += [cp.Read(a, f"{tag} {t} unf{i}", t) for i, a in enumerate(r.underflow)]
            out += [cp.Read(a, f"{tag} {t} ev{i}", t) for i, a in enumerate(r.events)]
        out += [cp.Read(k.value(n), f"{tag} {case.kind} L{n}", case.kind) for n in locks]
        if case.mode in ("bd", "dmaerr"):
            for t in regs:
                out += [cp.Read(a, f"{tag} {t} {n}", t) for n, a in status[t]]
        return out

    def clear() -> list:
        return [cp.Write(a, 0xFFFFFFFF, f"clear {case.kind}", case.kind) for a in (*k.overflow, *k.underflow)]

    def req(lock: int, acquire: bool, change: int, what: str) -> cp.Read:
        return cp.Read(k.request(lock, acquire, change),
                       f"req {case.kind} L{lock} {'acq' if acquire else 'rel'} {change:+d} {what}", case.kind)

    def setv(lock: int, v: int) -> cp.Write:
        return cp.Write(k.value(lock), v, f"set {case.kind} L{lock}", case.kind)

    if case.mode == "host":
        a, b = LEFT_LOCKS[case.kind]
        locks = (a, b)
        return [
            *snap("s0", locks),                                   # clean baseline
            setv(a, 63), setv(b, 0),
            req(a, False, 1, "(63: overflow)"), req(b, False, -1, "(0: underflow)"),
            *snap("s1", locks),
            req(a, False, 1, "(frozen)"), req(b, True, -1, "(frozen, 63)"),
            *snap("s2", locks),
            *clear(), *snap("s3", locks),
            req(a, False, 1, "(cleared)"), req(b, True, -1, "(cleared, 63)"),
            *snap("s4", locks),
            *clear(), *snap("s5", locks),                         # clear-and-reread
        ]
    if case.mode == "dmaerr":
        ch = LEFT_DMAERR_CHANNEL
        queue, ctrl = k.regs[f"DMA_S2MM_{ch}_Start_Queue"], k.regs[f"DMA_S2MM_{ch}_Ctrl"]
        return [
            *snap("s0", ()),
            cp.Write(queue, LEFT_DMAERR_BD, "start an unconfigured BD", case.kind),
            *snap("s1", ()), *snap("s2", ()),
            cp.Write(ctrl, 0x2, "channel reset", case.kind), cp.Write(ctrl, 0x0, "reset released", case.kind),
            *snap("s3", ()),
        ]
    lock, aux = LEFT_BD_LOCK[case.kind], LEFT_AUX_LOCK[case.kind]
    gates = LEFT_GATE_LOCKS[case.kind][:case.chain - 1]
    witnesses = LEFT_WITNESS_LOCKS[case.kind][:case.chain - 1]
    probe = LEFT_PROBE_LOCK[case.kind]
    rel = case.act == "rel"
    locks = (lock, aux, *(gates + witnesses if rel else ()), probe)
    held = gates[case.hold - 1] if case.hold else None
    freeze = [setv(lock, 63), setv(aux, 1 if rel else 0),
              *([setv(g, 1) for g in gates if g != held] if rel else []),
              req(lock, False, 1 if rel else 2, "(63: overflow)")]
    return [
        *snap("s0", locks),
        *freeze, *snap("s1", locks),
        cp.Raw(left_task_lines(case)),
        *snap("s2", locks), *snap("s3", locks),                   # time for the transfer: held?
        req(probe, False, 1, "(an unfrozen lock, requests pending)"), *snap("s3b", locks),
        *clear(), *snap("s4", locks),
        *([setv(held, 1)] if held is not None else []), *snap("s5", locks),
        *clear(), *snap("s6", locks),
    ]


def left_task_lines(case: LeftCase) -> tuple[str, ...]:
    """Runtime-sequence lines that configure and start the case's DMA tasks (receiver first).
    The lock end's first BD takes the frozen lock: a release after its data, or an acquire
    before it; a release case chains a witness BD behind it. The other end is a plain BD."""
    w, d = LEFT_WORDS, case.data_words
    m = f"memref<{d}xi32>"
    rel = case.act == "rel"
    # (acquire lock, release lock) of the locked end's BDs
    locked_bds = [("lx", "lk") if rel else ("lk", "lx")]
    if rel:
        locked_bds += [(f"ly{i}", f"lw{i}") for i in range(case.chain - 1)]

    def use(var: str, action: str, sym: str) -> list[str]:
        kind = "Release" if action == "rel" else "AcquireGreaterEqual"
        return [f"%c_{var}_{action} = arith.constant 1 : i32",
                f"aie.use_lock(%{sym}, {kind}, %c_{var}_{action})"]

    def task(var: str, tile: str, dir_: str, ch: int, bds: list[tuple[str, tuple | None]]) -> list[str]:
        """bds: (dma_bd text, (acquire sym, release sym) | None) per chained BD."""
        out = [f"%{var} = aiex.dma_configure_task(%{tile}, {dir_}, {ch}) {{"]
        for i, (bd, locks) in enumerate(bds):
            out.append(f"^bd{i}:" if i else "")
            if locks:
                out += [f"  {x}" for x in use(f"{var}{i}", "acq", locks[0])]
            out.append(f"  {bd}")
            if locks:
                out += [f"  {x}" for x in use(f"{var}{i}", "rel", locks[1])]
            out.append(f"  aie.next_bd ^bd{i + 1}" if i + 1 < len(bds) else "  aie.end")
        out.append("}")
        return [x for x in out if x]

    def shim_bd(off: int, n: int, bd_id: int) -> str:
        return f"aie.dma_bd(%data : {m} offset = {off} len = {n}) {{bd_id = {bd_id} : i32}}"

    def mem_bd(off: int, n: int, bd_id: int) -> str:
        return f"aie.dma_bd(%mbuf : memref<{case.chain * w}xi32> offset = {off} len = {n}) {{bd_id = {bd_id} : i32}}"

    nb = len(locked_bds)
    locked = lambda mk, base_off, base_id: [(mk(base_off + i * w, w, base_id + i), locked_bds[i])
                                             for i in range(nb)]
    plain = lambda mk, off, base_id: [(mk(off, nb * w, base_id), None)]
    dst_off, src_off = case.chain * w, 0
    locked_mm2s = case.direction == "mm2s"
    if case.kind == "shim":
        if locked_mm2s:
            recv = task("tr", "t00", "S2MM", 1, plain(shim_bd, dst_off, 12))
            send = task("ts", "t00", "MM2S", 1, locked(shim_bd, src_off, 4))
        else:
            recv = task("tr", "t00", "S2MM", 1, locked(shim_bd, dst_off, 4))
            send = task("ts", "t00", "MM2S", 1, plain(shim_bd, src_off, 12))
    elif not locked_mm2s:      # shim MM2S 1 -> memtile S2MM 0 (locked)
        recv = task("tr", "t01", "S2MM", 0, locked(mem_bd, 0, 0))
        send = task("ts", "t00", "MM2S", 1, plain(shim_bd, src_off, 12))
    else:                      # memtile MM2S 0 (locked) -> shim S2MM 1
        recv = task("tr", "t00", "S2MM", 1, plain(shim_bd, dst_off, 12))
        send = task("ts", "t01", "MM2S", 0, locked(mem_bd, 0, 0))
    return (*recv, *send, "aiex.dma_start_task(%tr)", "aiex.dma_start_task(%ts)")


def left_design(case: LeftCase) -> tuple[str, list, "object"]:
    """(aie.mlir text, host steps, packet layout) of a left case."""
    import ctrlpkt_readback as cp
    steps = left_steps(case)
    layout = cp.encode(steps)
    d = case.data_words
    m = f"memref<{d}xi32>"
    tiles = {"shim": "%t00", "mem": "%t01"}
    flows = []
    for t in left_targets(case):
        flows += cp.packet_flows(tiles[t], "%t00", t)
    lines = ["module {", "  aie.device(npu1_1col) {",
             "    %t00 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}",
             "    %t01 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}",
             *flows, *cp.shim_allocations("%t00")]
    if case.mode == "bd":
        t = case.kind
        tile = tiles[t]
        syms = [("lk", LEFT_BD_LOCK[t]), ("lx", LEFT_AUX_LOCK[t]), ("lz", LEFT_PROBE_LOCK[t]),
                *[(f"ly{i}", g) for i, g in enumerate(LEFT_GATE_LOCKS[t][:case.chain - 1])],
                *[(f"lw{i}", g) for i, g in enumerate(LEFT_WITNESS_LOCKS[t][:case.chain - 1])]]
        for sym, lid in syms:
            lines.append(f'    %{sym} = aie.lock({tile}, {lid}) {{init = 0 : i32, sym_name = "{sym}"}}')
        if t == "mem":
            lines += [f'    %mbuf = aie.buffer(%t01) {{sym_name = "mbuf"}} : memref<{case.chain * LEFT_WORDS}xi32>']
        # the receiver's and sender's stream flow
        src, dst = {("shim", "mm2s"): ("%t00, DMA : 1", "%t00, DMA : 1"),
                    ("shim", "s2mm"): ("%t00, DMA : 1", "%t00, DMA : 1"),
                    ("mem", "s2mm"): ("%t00, DMA : 1", "%t01, DMA : 0"),
                    ("mem", "mm2s"): ("%t01, DMA : 0", "%t00, DMA : 1")}[(t, case.direction)]
        lines += [f"    aie.flow({src}, {dst})",
                  "    aie.shim_dma_allocation @dout (%t00, MM2S, 1)",
                  "    aie.shim_dma_allocation @din (%t00, S2MM, 1)"]
    lines += [f"    aie.runtime_sequence @seq(%ctrlout: memref<{len(layout.labels)}xi32>, "
              f"%ctrlin: memref<{len(layout.words)}xi32>, %data: {m}) {{",
              *cp.sequence_lines(steps, layout, "ctrlout", "ctrlin"), "    }", "  }", "}", ""]
    return "\n".join(lines), steps, layout


# -- results ---------------------------------------------------------------------

def parse_sampler(consumer_log: bytes) -> dict:
    w = struct.unpack_from("<16I", consumer_log)
    caps = list(w[0:8])
    first = next((i + 1 for i, v in enumerate(caps) if v == LDA_NEW), None)
    return {"caps": caps, "new_from": first, "window": (w[9] - w[8]) & 0xFFFFFFFF,
            "ptr_delta": w[10]}


def parse_eq(name: str, consumer_log: bytes, producer_ts: bytes, step_cycles: float) -> dict:
    case = EQ_CASES[name]
    n = len(case.consumer) + 1
    ts = struct.unpack_from(f"<{n}I", consumer_log)
    waits = [(ts[i] - ts[0]) & 0xFFFFFFFF for i in range(1, n)]
    pts = struct.unpack_from(f"<{len(case.producer)}I", producer_ts)
    return {"waits": waits, "steps": [round(w / step_cycles) for w in waits],
            "producer_ts": list(pts),
            "match": [r for r in RULES if simulate(case, r) == [round(w / step_cycles) for w in waits]]}



# -- frozen CLI: generate / build / run / analyze one directory per case --------------------

def frozen_generate(args) -> int:
    import json
    import ctrlpkt_readback as cp
    out = Path(args.out_dir)
    for name in args.cases:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if name in LEFT_CASES:
            case = LEFT_CASES[name]
            text, steps, layout = left_design(case)
            (d / "aie.mlir").write_text(text)
            (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
            (d / "case.json").write_text(json.dumps(
                {"case": name, "family": "left", "kind": case.kind, "mode": case.mode,
                 "direction": case.direction, "act": case.act, "labels": list(layout.labels),
                 "n_words": len(layout.words), "passes": case.data_words // FZ_STAMP_WORDS, "chain": case.chain, "hold": case.hold,
                 "host_args": ["seq"]}, indent=1) + "\n")
            continue
        case = FROZEN_CASES[name]
        text, steps, layout = frozen_design(case)
        (d / "aie.mlir").write_text(text)
        (d / "core.s").write_text(frozen_core_asm(case))
        (d / "timer.s").write_text(frozen_timer_asm(second=not case.dma))
        (d / "pkts.bin").write_bytes(cp.pack_words(layout.words))
        (d / "case.json").write_text(json.dumps(
            {"case": name, "init": case.init, "reqs": case.reqs, "labels": list(layout.labels),
             "n_words": len(layout.words), "passes": frozen_passes(case), "dma": case.dma,
             "keep": case.keep, "b2b": case.b2b}, indent=1) + "\n")
    return 0


def frozen_build(args) -> int:
    import subprocess
    from harness.build import _peano_tool, aiecc_command, aiecc_path
    for d in sorted(p for p in Path(args.out_dir).iterdir() if (p / "aie.mlir").exists()):
        for k in ("core", "timer"):
            if not (d / f"{k}.s").exists():
                continue
            subprocess.run([_peano_tool("llvm-mc"), "--triple=aie2", "--filetype=obj",
                            "-o", str(d / f"{k}.o"), str(d / f"{k}.s")], check=True)
            subprocess.run([_peano_tool("llvm-objcopy"), "--redefine-sym", f"test_kernel=k_{k}",
                            str(d / f"{k}.o")], check=True)
        r = subprocess.run(aiecc_command("peano", aiecc_path()), cwd=d, capture_output=True, text=True)
        (d / "aiecc.log").write_text(r.stdout + r.stderr)
        print(f"{d.name}: aiecc exit {r.returncode}")
        if r.returncode:
            return 1
    return 0


def frozen_run(args) -> int:
    import json
    import subprocess
    import ctrlpkt_readback as cp
    from harness.run import KERNEL_NAME, _env, build_ffi
    root = Path(args.out_dir)
    host = cp.build_host(root)
    if args.backend == "emu":
        build_ffi(args.runtime)
    env = _env(args.backend, args.runtime)
    for d in sorted(p for p in root.iterdir() if (p / "aie.xclbin").exists()):
        if args.only and d.name not in args.only:
            continue
        rd = d / "runs" / args.tag
        if rd.exists():
            raise SystemExit(f"{rd} exists; pick another tag")
        rd.mkdir(parents=True)
        meta = json.loads((d / "case.json").read_text())
        cmd = [str(host), str(d / "aie.xclbin"), str(d / "insts.bin"), str(d / "pkts.bin"),
               str(len(meta["labels"])), str(meta.get("n_data", meta.get("passes", 2) * FZ_STAMP_WORDS)), str(rd / "out.bin"), *meta.get("host_args", [])]
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=60 if args.backend == "hw" else 1800)
            rc, text = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            rc, text = -1, "TIMEOUT"
        (rd / "log.txt").write_text(text)
        print(f"{d.name}: rc {rc}", flush=True)
        if rc != 0:
            return 1
    return 0


def left_report(name: str, meta: dict, r: dict) -> str:
    """One line per snapshot (flags, lock values, event words that are not 0, DMA status words),
    then the data halves."""
    c = r["ctrl"]
    lines = [f"{name:14s} kind={meta['kind']} mode={meta['mode']} {meta['direction']} {meta['act']}"]
    tags = list(dict.fromkeys(k.split()[0] for k in c))
    for t in tags:
        parts = []
        for k, v in c.items():
            tag, _, rest = k.partition(" ")
            if tag != t:
                continue
            tile, _, reg = rest.partition(" ")
            if reg.startswith("ev") and v == 0:
                continue
            if reg.startswith(("ovf", "unf")) and v == 0:
                continue
            parts.append(f"{tile}.{reg}={v:#x}")
        lines.append(f"   {t}: " + " ".join(parts))
    for k, v in c.items():
        if k.startswith("req"):
            lines.append(f"   {k}: {v:#x}")
    data = [x for i in range(meta["passes"]) for x in r[f"pass{i + 1}"]]
    lines.append("   dst per BD: " + " ".join(f"{x:#x}" for x in data[meta["chain"] * LEFT_WORDS::LEFT_WORDS]))
    return "\n".join(lines)


def frozen_analyze(args) -> int:
    import json
    root = Path(args.out_dir)
    for d in sorted(p for p in root.iterdir() if (p / "case.json").exists()):
        meta = json.loads((d / "case.json").read_text())
        if args.only and d.name not in args.only:
            continue
        out = d / "runs" / args.tag / "out.bin"
        if not out.exists():
            print(f"{d.name:8s} (no output)")
            continue
        r = parse_frozen(out.read_bytes(), meta["labels"], meta.get("passes", 2))
        c = r["ctrl"]
        if meta.get("family") == "left":
            print(left_report(d.name, meta, r))
            continue
        tags = sorted({k.split()[0] for k in c})
        locks = sorted({int(k.split()[1][1:]) for k in c if k.split()[1].startswith("L")})
        snaps = [f"{t}: ovf={c[t + ' ovf']:#x} unf={c[t + ' unf']:#x} "
                 + " ".join(f"L{x}={c[f'{t} L{x}']}" for x in locks)
                 + (f" ev3={c[t + ' ev3']:#x}" if t + ' ev3' in c else "") for t in tags]
        n = len(meta["reqs"])
        print(f"{d.name:8s} init={meta['init']} reqs={meta['reqs']} dma={meta.get('dma', '')}")
        for sn in snaps:
            print("   " + sn)
        for i in range(meta.get("passes", 2)):
            st = r[f"pass{i + 1}"]
            if meta.get("b2b"):
                layout = b2b_layout("a")
                lost = (None if FZ_SENTINEL in st[:8] else
                        [((st[j] - st[0]) & 0xFFFFFFFF) - (layout[j] - layout[0]) for j in range(1, 8)])
                print(f"   pass{i + 1}: marker={st[8]:#010x} lost at {layout[1:]}: {lost}")
                continue
            print(f"   pass{i + 1}: marker={st[8]:#010x} lost={frozen_lost(st, n)} "
                  f"raw={[hex(x) for x in st[:4 * n]]}")
    return 0


# -- CLI -----------------------------------------------------------------------------

def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    names = []
    for fam in args.cases:
        if fam.startswith("lk_"):
            d = int(fam.split("_")[1])
            made = [lk_case(d, m, args.late_iters) for m in args.modes]
            metas = [{"family": "lk", "d": d, "mode": m, "late_iters": args.late_iters} for m in args.modes]
        elif fam.startswith("vlk_"):
            # vlk_KIND (every D) or vlk_KIND_D
            parts = fam.split("_")[1:]
            kind = parts[0]
            ds = [int(parts[1])] if len(parts) > 1 else list(range(1, 9))
            made = [vlk_case(kind, d, m, args.late_iters) for d in ds for m in args.modes]
            metas = [{"family": "vlk", "kind": kind, "d": d, "mode": m, "late_iters": args.late_iters}
                     for d in ds for m in args.modes]
        elif fam == "same":
            made = [same_case(m, args.late_iters) for m in args.modes]
            metas = [{"family": "same", "mode": m, "late_iters": args.late_iters} for m in args.modes]
        elif fam == "pos":
            made = [pos_case("pos", m, args.late_iters) for m in args.modes]
            metas = [{"family": "pos", "mode": m, "late_iters": args.late_iters} for m in args.modes]
        elif fam == "relpos":
            made = [pos_case("relpos", "stall", 0)]
            metas = [{"family": "relpos"}]
        elif fam.startswith("b2b_"):
            # b2b_PAIR (every variant) or b2b_PAIR_VARIANT, at every distance 1..B2B_MAX_D
            parts = fam.split("_")
            pair = parts[1]
            if pair not in B2B_PAIRS:
                raise SystemExit(f"unknown b2b pair {pair!r}")
            variants = parts[2:] or list(B2B_VARIANTS[pair])
            made, metas = [], []
            for v in variants:
                for d in range(1, B2B_MAX_D + 1):
                    for w in B2B_WINDOWS:
                        made.append(b2b_case(pair, v, d, w, args.late_iters))
                        metas.append({"family": "b2b", "pair": pair, "variant": v, "d": d,
                                      "window": w, "late_iters": args.late_iters})
        elif fam == "condlat":
            # r26 written 2, 1 and 0 (same bundle) bundles before an acq.cond / rel.cond
            made, metas = [], []
            for kind, st in (("acq", "free"), ("rel", "rel")):
                for vk, new in (("z", 1), ("o", 0)):
                    for off in (-2, -1, 0):
                        made.append(cond_case(kind, st, vk, args.late_iters, (off, new)))
                        metas.append({"family": "cond", "kind": kind, "state": st, "v": vk,
                                      "value": COND_VALUES[vk], "late_iters": args.late_iters,
                                      "writer": [off, new], "layout": cond_layout((off, new))})
        elif fam == "cond" or fam.startswith("cond_"):
            # cond (every case), cond_KIND, cond_KIND_STATE, cond_KIND_STATE_V
            parts = fam.split("_")[1:]
            made, metas = [], []
            for kind, states in COND_STATES.items():
                for st in states:
                    for vk in COND_VALUES:
                        if [kind, st, vk][:len(parts)] != parts:
                            continue
                        made.append(cond_case(kind, st, vk, args.late_iters))
                        metas.append({"family": "cond", "kind": kind, "state": st, "v": vk,
                                      "value": COND_VALUES[vk], "late_iters": args.late_iters})
            if not made:
                raise SystemExit(f"no cond case matches {fam!r}")
        elif fam.startswith("condwin"):
            # condwin (every variant) or condwin_VARIANT, at every distance 1..CONDWIN_MAX_D
            parts = fam.split("_", 1)[1:]
            made, metas = [], []
            for v in CONDWIN_VARIANTS:
                if parts and v != parts[0]:
                    continue
                for d in range(1, CONDWIN_MAX_D + 1):
                    for w in CONDWIN_WINDOWS:
                        made.append(condwin_case(v, d, w, args.late_iters))
                        metas.append({"family": "condwin", "variant": v, "d": d, "window": w,
                                      "late_iters": args.late_iters})
            if not made:
                raise SystemExit(f"no condwin case matches {fam!r}")
        elif fam.startswith("eq_"):
            name = fam[3:]
            made = [eq_case(name, args.step_iters)]
            metas = [{"family": "eq", "case": name, "step_iters": args.step_iters,
                      "predictions": {r: simulate(EQ_CASES[name], r) for r in RULES}}]
        else:
            raise SystemExit(f"unknown family {fam!r}")
        for (topo, files), meta in zip(made, metas):
            suite.write_case(out, topo.name, topo, files=files, meta=meta)
            names.append(topo.name)
    suite.write_suite_json(out, "lock-stall", "gen_lock_stall_probe.py", names)
    return 0


def cmd_analyze(args) -> int:
    """One line per case of a run tag: lk/same samples, eq grant steps and matching rules."""
    sdir = Path(args.suite_dir)
    s = suite.load_suite(sdir)
    run = sdir / "runs" / args.tag
    for name in s.cases:
        meta = s.case_meta(name)
        cdir = run / name
        if not (cdir / "consumer.log.bin").exists():
            print(f"{name:14s} (no output)")
            continue
        log = (cdir / "consumer.log.bin").read_bytes()
        if meta["family"] == "b2b":
            pts = cdir / "producer.ts.bin"
            refused = b2b_refused_lock(meta["variant"]) is not None
            prod = pts.read_bytes()
            r = parse_b2b(meta["window"], log, prod if refused else None)
            cells = " ".join(f"{o:+d}:{x}" for o, x in zip(r["layout"][1:], r["lost"][1:]))
            res = f"  resume=P+{r['resume']}" if r["resume"] is not None else ""
            counts = parse_b2b_counts(log, prod)
            cnt = " ".join(f"L{k}={v}" for k, v in counts.items())
            print(f"{name:24s} lost {cells}  {cnt}{res}")
            continue
        if meta["family"] == "condwin":
            layout = CONDWIN_WINDOWS[meta["window"]]
            ts = struct.unpack_from("<8I", log)
            lost = [((ts[i] - ts[0]) & 0xFFFFFFFF) - (layout[i] - layout[0]) for i in range(1, 7)]
            counts = parse_b2b_counts(log, (cdir / "producer.ts.bin").read_bytes())
            cells = " ".join(f"{o:+d}:{x}" for o, x in zip(layout[1:], lost))
            print(f"{name:24s} lost {cells}  " + " ".join(f"L{k}={v}" for k, v in counts.items()))
            continue
        if meta["family"] == "cond":
            layout = meta.get("layout", COND_LAYOUT)
            r = parse_cond(log, (cdir / "producer.ts.bin").read_bytes(), layout)
            cells = " ".join(f"{o:+d}:{x}" for o, x in zip(layout[1:], r["lost"][1:]))
            w = f"  then mov r26,#{meta['writer'][1]} at {meta['writer'][0]:+d}" if "writer" in meta else ""
            print(f"{name:24s} r26={meta['value']:#010x} lost {cells}  count={r['count']}{w}")
            continue
        if meta["family"] == "vlk":
            r = parse_vlk(log)
            caps = " ".join("A" if meta["d"] == i + 1 else ("torn" if v is None else f"{v}")
                            for i, v in enumerate(r["caps"]))
            print(f"{name:20s} {caps}  new_from=+{r['new_from']} window={r['window']}")
            continue
        if meta["family"] in ("pos", "relpos"):
            print(f"{name:14s} deltas after the request: {parse_pos(log)}")
            continue
        if meta["family"] in ("lk", "same"):
            r = parse_sampler(log)
            caps = " ".join("A" if meta.get("d") == i + 1 else f"{v}" for i, v in enumerate(r["caps"]))
            extra = f" p2+={r['ptr_delta']}" if meta["family"] == "same" else ""
            print(f"{name:14s} {caps}  new_from=+{r['new_from']} window={r['window']}{extra}")
        else:
            step = args.step_cycles or 10.0 * meta["step_iters"]
            r = parse_eq(meta["case"], log, (cdir / "producer.ts.bin").read_bytes(), step)
            print(f"{name:14s} waits={r['waits']} steps={r['steps']} match={r['match']} "
                  f"predicted={meta['predictions']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("out_dir")
    gen.add_argument("--cases", nargs="+", required=True,
                     help="lk_1..lk_8, vlk_<vlda|vldb|am>[_D], same, pos, relpos, b2b_<pair>, condwin[_variant], cond[_kind[_state[_v]]], eq_<name> (" + ", ".join(EQ_CASES) + ")")
    gen.add_argument("--modes", nargs="+", default=list(MODES))
    gen.add_argument("--late-iters", type=int, default=1500)
    gen.add_argument("--step-iters", type=int, default=400)
    an = sub.add_parser("analyze")
    an.add_argument("suite_dir")
    an.add_argument("tag")
    an.add_argument("--step-cycles", type=float, default=None,
                    help="cycles per producer step (default: 10 per loop iteration)")
    fg = sub.add_parser("frozen-generate", help="overflow/underflow probes read back over control packets")
    fg.add_argument("out_dir")
    fg.add_argument("--cases", nargs="+", default=list(FROZEN_CASES))
    fb = sub.add_parser("frozen-build")
    fb.add_argument("out_dir")
    fr = sub.add_parser("frozen-run")
    fr.add_argument("out_dir")
    fr.add_argument("tag")
    fr.add_argument("--backend", choices=("hw", "emu"), required=True)
    fr.add_argument("--only", nargs="*", default=[])
    fr.add_argument("--runtime", choices=("debug", "release"), default="debug",
                    help="emulator .so profile (release is far faster; the 4 ms timer pass is slow in debug)")
    fa = sub.add_parser("frozen-analyze")
    fa.add_argument("out_dir")
    fa.add_argument("tag")
    fa.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args(argv)
    special = {"frozen-generate": frozen_generate, "frozen-build": frozen_build,
               "frozen-run": frozen_run, "frozen-analyze": frozen_analyze}
    if args.cmd in special:
        return special[args.cmd](args)
    return cmd_generate(args) if args.cmd == "generate" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
