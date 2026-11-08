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
same_MODE   `lda r6 ; paddb [p2], #4 ; acq` in one bundle, then r6 sampled at +1..+8: is
            a load bundled with a blocked acquire issued once, and when; and is the
            pointer add applied once.
eq_NAME     acquires with a change >= 0. The consumer runs a list of lock requests,
            reading the cycle counter before the first and after each; the producer runs
            its own list, one request every `step_iters` loop iterations, reading the
            counter before each. When a consumer request is granted (its producer step)
            tells the acquire rule apart: `simulate` predicts it for each candidate rule,
            and every case completes under every rule (nothing can hang the NPU).

Consumer output (`consumer.log`, 64 bytes, little-endian u32):
  lk/same: words 0..7 = r6 at +1..+8 (0 where bundle +D is the acquire), word 8/9 = the
           counter before the load / after the last sample, word 10 = p2 - p0 (same only),
           word 15 = the value the load reads.
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


def _topology(name: str, cons: str, prod: str, init: int) -> tuple[Topology, dict[str, str]]:
    topo = Topology(
        name=name,
        cores=(Core("producer", PRODUCER_TILE, Path("producer.s"), args=(Buffer("ts", "out", TS_BYTES),)),
               Core("consumer", CONSUMER_TILE, Path("consumer.s"), args=(Buffer("log", "out", LOG_BYTES),))),
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


# -- CLI -----------------------------------------------------------------------------

def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    names = []
    for fam in args.cases:
        if fam.startswith("lk_"):
            d = int(fam.split("_")[1])
            made = [lk_case(d, m, args.late_iters) for m in args.modes]
            metas = [{"family": "lk", "d": d, "mode": m, "late_iters": args.late_iters} for m in args.modes]
        elif fam == "same":
            made = [same_case(m, args.late_iters) for m in args.modes]
            metas = [{"family": "same", "mode": m, "late_iters": args.late_iters} for m in args.modes]
        elif fam == "pos":
            made = [pos_case("pos", m, args.late_iters) for m in args.modes]
            metas = [{"family": "pos", "mode": m, "late_iters": args.late_iters} for m in args.modes]
        elif fam == "relpos":
            made = [pos_case("relpos", "stall", 0)]
            metas = [{"family": "relpos"}]
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
                     help="lk_1..lk_8, same, pos, relpos, eq_<name> (" + ", ".join(EQ_CASES) + ")")
    gen.add_argument("--modes", nargs="+", default=list(MODES))
    gen.add_argument("--late-iters", type=int, default=1500)
    gen.add_argument("--step-iters", type=int, default=400)
    an = sub.add_parser("analyze")
    an.add_argument("suite_dir")
    an.add_argument("tag")
    an.add_argument("--step-cycles", type=float, default=None,
                    help="cycles per producer step (default: 10 per loop iteration)")
    args = ap.parse_args(argv)
    return cmd_generate(args) if args.cmd == "generate" else cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
