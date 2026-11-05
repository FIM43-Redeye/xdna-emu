#!/usr/bin/env python3
"""Cascade link timing probes, synchronized through the link itself; writes a suite.

The two cores of a cascade pair start a jittering ~1100-1800 cycles apart on silicon, so a
probe cannot place a producer write at a chosen cycle of the consumer by program layout alone.
Every probe here first passes a *sync word*: the consumer is already held in its first read
(the producer spins `--sync-iters` loop iterations first), so the release of that read ties the
two programs together, and everything after it is straight-line code on both cores. Each core
reads its cycle counter at `@ts` markers into a register pair; the low words are stored to the
core's output once its program is done, so the stores never perturb the measured window.

Timer differences within one core are offset-free. A difference between a consumer stamp and a
producer stamp also carries the fixed phase offset of the two tiles' timers.

Families (consumer reads hold at the fifth bundle after their `vmov x, SCD`; every consumer
`@ts` that matters sits on such a hold bundle):

  rel_n{N}_m{M}    read release: the producer writes the data word N bundles after the sync word;
                   the consumer issues its data read M bundles after its sync read released. The
                   consumer's stall against N shows where a word arriving just in time is still
                   missed and how long a held bundle takes to restart.
  wrel_n{N}_m{M}   write release: after the sync word the producer fills the link (two words) and
                   issues a third write N bundles later; the consumer pops one word M bundles
                   after its sync. The producer's stall against N and M is the backward path.
  rt_w{W}_g{G}_r{R}  round trip: the producer writes R words back to back after the sync word; the
                   consumer waits W bundles, then issues R reads G bundles apart and stamps every
                   hold bundle. Reads that find their word buffered do not stall; the first read
                   whose word was still stalled in the producer waits a full round trip.

Output layout (little-endian u32): 16 timestamp slots (low words, unused slots zero), then for
the consumer one 64-byte record per read, in read order.
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

# Register pairs for counter reads: `mov rH:rL, cntr` takes only the pairs of r16-r31, which the
# frame saves and restores.
TS_PAIRS = [(17, 16), (19, 18), (21, 20), (23, 22), (25, 24), (27, 26), (29, 28), (31, 30)]
TS_SLOTS = 16                      # 64 bytes keeps the record area 64-byte aligned
TS_BYTES = TS_SLOTS * 4
WORD = base.WORD_BYTES
READ_REGS = ["x0", "x2", "x4", "x6", "x8", "x10", "x1", "x3", "x5", "x7", "x9", "x11"]
SYNC_ITERS = 600                   # ~6000 cycles: longer than any start skew seen (East ~3500)
HOLD_AT = 5                        # a cascade read holds the fifth bundle after it
N = "  nop"


def _frame(body: list[str]) -> str:
    pro = [".text", ".globl test_kernel", "test_kernel:", "  paddb [sp], #64"]
    pro += [f"  st r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi = [f"  lda r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 32)]
    epi += [N] * 7 + ["  paddb [sp], #-64", "  ret lr"] + [N] * 5
    return "\n".join(pro + body + epi) + "\n"


def expand(lines: list[str], stores: list[str]) -> tuple[list[str], int]:
    """Replace `@ts` markers by counter reads; append the timestamp and read-record stores.

    `stores` names the x registers to store (in order) at [p1] after the timestamps.
    """
    out: list[str] = []
    n_ts = 0
    for ln in lines:
        if ln.strip() == "@ts":
            if n_ts >= len(TS_PAIRS):
                raise ValueError(f"at most {len(TS_PAIRS)} timestamps")
            hi, lo = TS_PAIRS[n_ts]
            out.append(f"  mov r{hi}:r{lo}, cntr")
            n_ts += 1
        else:
            out.append(ln)
    out += [N] * 8
    for i in range(n_ts):
        out.append(f"  st r{TS_PAIRS[i][1]}, [p0, #{4 * i}]")
    for x in stores:
        k = x[1:]
        out += [f"  vst wl{k}, [p1, #0]", f"  vst wh{k}, [p1, #32]", f"  padda [p1], #{WORD}"]
    return out, n_ts


def _word_value(i: int) -> int:
    return 0x100 + i


# Shared-memory words for the timer-phase probes (`mf*`/`mr*`): one in each core's own data
# memory, far above the stack (0x0-0x3FF) and the harness buffers (bank 0). The producer sits
# north of the consumer (South flow), so it sees the consumer's memory through its South window
# and the consumer sees the producer's through its North window.
MEM_CONS = 0xFF00                  # in the consumer's memory
MEM_PROD = 0xFF40                  # in the producer's memory
LOCAL, SOUTH, NORTH = 0x70000, 0x40000, 0x60000
MEM_VALUE = 0x5A
MEM_SLOT = TS_SLOTS - 1            # timestamp slot that receives the loaded value


def _mem_setup() -> list[str]:
    """Zero both shared words this core owns; r1 = the value a writer stores."""
    return ["  mov r0, #0", f"  movxm p4, #{LOCAL + MEM_CONS}", f"  movxm p5, #{LOCAL + MEM_PROD}",
            *base._nops(2), "  st r0, [p4, #0]", "  st r0, [p5, #0]", f"  mov r1, #{MEM_VALUE}",
            *base._nops(6)]


def _producer(lines: list[str], n_words: int, sync_iters: int, pre: list[str] = ()) -> str:
    body = base._enable("crMCDEn")
    for j in range(n_words):
        body += [f"  movxm r1, #{_word_value(j)}", *base._nops(2), f"  vbcst.32 x{j}, r1"]
    body += base._nops(8) + list(pre)
    body += base.delay_loop(sync_iters, ".Lsync")
    prog, _ = expand(lines, [])
    return _frame(body + prog)


def _consumer(lines: list[str], reads: list[str], pre: list[str] = ()) -> str:
    body = base._enable("crSCDEn")
    body += ["  mov r1, #7", "  mov p1, p0", f"  padda [p1], #{TS_BYTES}", *base._nops(2)]
    body += [f"  vbcst.32 {x}, r1" for x in reads] + base._nops(6) + list(pre)
    prog, _ = expand(lines, reads)
    return _frame(body + prog)


def _read(x: str) -> list[str]:
    """A cascade read and the four bundles before its hold bundle."""
    return [f"  vmov {x}, SCD", *[N] * (HOLD_AT - 1)]


# -- families ----------------------------------------------------------------

@dataclass(frozen=True)
class Programs:
    producer: list[str]
    consumer: list[str]
    words: int                     # words the producer sends (== consumer reads)
    pre: tuple[str, ...] = ()      # pre-sync setup on both cores


def _mem_load(after: int) -> list[str]:
    """`after` nops, a load of the shared word (p6), and the store of the loaded value."""
    return [*[N] * after, "  lda r20, [p6, #0]", *[N] * 8, f"  st r20, [p0, #{4 * MEM_SLOT}]"]


def mem_programs(kind: str, k: int, delay: int = 0) -> Programs:
    """Timer-phase probe through shared memory, synchronized by the cascade sync word.

    The writer stores `delay` bundles after its sync point (mf: right after the sync write; mr:
    the bundle after the sync hold bundle).

    mf1: producer stores through its South window after the sync write; the consumer
         loads its own word K bundles after its sync hold bundle.
    mf2: producer stores its own word; the consumer loads it through its North window.
    mr1: consumer stores its own word on the bundle after its sync hold; the producer loads it
         through its South window K bundles after the sync write.
    mr2: consumer stores through its North window; the producer loads its own word.
    """
    w_addr, r_addr = {"mf1": (SOUTH + MEM_CONS, LOCAL + MEM_CONS),
                      "mf2": (LOCAL + MEM_PROD, NORTH + MEM_PROD),
                      "mr1": (LOCAL + MEM_CONS, SOUTH + MEM_CONS),
                      "mr2": (NORTH + MEM_PROD, LOCAL + MEM_PROD)}[kind]
    store = "  st r1, [p7, #0]"
    if kind.startswith("mf"):
        prod_pre, cons_pre = [f"  movxm p7, #{w_addr}"], [f"  movxm p6, #{r_addr}"]
        prod = ["@ts", "  vmov MCD, x0", *[N] * delay, store, "@ts"]
        cons = ["@ts", *_read("x0"), "@ts", *_mem_load(k)]
    else:
        prod_pre, cons_pre = [f"  movxm p6, #{r_addr}"], [f"  movxm p7, #{w_addr}"]
        prod = ["@ts", "  vmov MCD, x0", *_mem_load(k), "@ts"]
        cons = ["@ts", *_read("x0"), "@ts", *[N] * delay, store]
    return Programs(prod, cons, 1, pre=(tuple(prod_pre), tuple(cons_pre)))


def rel_programs(n: int, m: int) -> Programs:
    prod = ["@ts", "  vmov MCD, x0", "@ts", *[N] * n, "  vmov MCD, x1", "@ts"]
    cons = ["@ts", *_read("x0"), "@ts", *[N] * m, *_read("x2"), "@ts"]
    return Programs(prod, cons, 2)


def wrel_programs(n: int, m: int) -> Programs:
    prod = ["@ts", "  vmov MCD, x0", "@ts", "  vmov MCD, x1", "  vmov MCD, x2", "@ts",
            *[N] * n, "  vmov MCD, x3", "@ts", "@ts"]
    cons = ["@ts", *_read("x0"), "@ts", *[N] * m, *_read("x2"), "@ts",
            *[N] * 40, "  vmov x4, SCD", "  vmov x6, SCD", *[N] * 8]
    return Programs(prod, cons, 4)


def rt_programs(w: int, g: int, r: int) -> Programs:
    """R writes back to back; R reads G bundles apart after W bundles; a stamp on every
    bundle from the first read's hold bundle to the last read's."""
    prod = ["@ts", "  vmov MCD, x0", "@ts"]
    for i in range(1, r + 1):
        prod += [f"  vmov MCD, x{i}"]
    prod += ["@ts"]
    # consumer bundle schedule after the sync hold: W nops, then reads at offsets k*(G+1);
    # stamps on each bundle from offset HOLD_AT to (R-1)*(G+1)+HOLD_AT.
    reads = READ_REGS[1:r + 1]
    first_hold, last_hold = HOLD_AT, (r - 1) * (g + 1) + HOLD_AT
    sched = []
    for off in range(last_hold + 1):
        k, rem = divmod(off, g + 1)
        if rem == 0 and k < r:
            sched.append(f"  vmov {reads[k]}, SCD")
        elif off >= first_hold:
            sched.append("@ts")
        else:
            sched.append(N)
    # A read bundle cannot also stamp; a hold that lands on a read bundle is measured by the
    # stamp on the bundle after it (the next one that exists).
    cons = ["@ts", *_read("x0"), "@ts", *[N] * w, *sched]
    return Programs(prod, cons, r + 1)


def _case(name: str, progs: Programs, direction: str, sync_iters: int) -> base.Case:
    reads = READ_REGS[:progs.words]
    prod_pre, cons_pre = progs.pre if progs.pre else ((), ())
    if progs.pre:
        prod_pre, cons_pre = _mem_setup() + list(prod_pre), _mem_setup() + list(cons_pre)
    return base._pair(name, direction,
                      _producer(progs.producer, progs.words, sync_iters, prod_pre),
                      _consumer(progs.consumer, reads, cons_pre),
                      (Buffer("ts", "out", TS_BYTES),),
                      (Buffer("log", "out", TS_BYTES + progs.words * WORD),))


def build_case(spec: str, direction: str, sync_iters: int = SYNC_ITERS) -> tuple[base.Case, dict]:
    """`spec` is a family name with parameters, e.g. `rel_n12_m0`, `rt_w60_g0_r4`."""
    fam, *params = spec.split("_")
    p = {s[0]: int(s[1:]) for s in params}
    if fam == "rel":
        progs = rel_programs(p["n"], p["m"])
    elif fam == "wrel":
        progs = wrel_programs(p["n"], p["m"])
    elif fam == "rt":
        progs = rt_programs(p["w"], p["g"], p["r"])
    elif fam in ("mf1", "mf2", "mr1", "mr2"):
        if direction != "south":
            raise ValueError("the shared-memory probes need the South placement")
        progs = mem_programs(fam, p["k"], p.get("s", 0))
    else:
        raise ValueError(f"unknown family in {spec!r}")
    name = f"{spec}_{direction}"
    meta = {"family": fam, "params": p, "direction": direction, "words": progs.words,
            "sync_iters": sync_iters,
            "producer_ts": sum(1 for x in progs.producer if x.strip() == "@ts"),
            "consumer_ts": sum(1 for x in progs.consumer if x.strip() == "@ts")}
    return _case(name, progs, direction, sync_iters), meta


# -- SEMAPHORE partners (three cores) -----------------------------------------
#
# TableGen's `II_VMOV_CASCADE_READ` reserves the SEMAPHORE unit for the six cycles after the read
# (`AvoidSemaphore<6>`); `vmov MCD` and the stream moves hold it too. The middle core B of a chain
# A -> B -> C (A at row 4, B row 3, C row 2) issues a cascade read and, D bundles from it, one of
# those partners; every word is checked downstream, and B stamps its timer around the pair.
#   semc_d{D}   B: `vmov x0, SCD`, D bundles later `vmov MCD, x5`          (C reads the cascade)
#   semcr_d{D}  B: `vmov MCD, x5`, D bundles later `vmov x0, SCD`
#   sems_d{D}   B: `vmov x0, SCD`, D bundles later `mov ms, r5`           (C reads the stream)
#   semsr_d{D}  B: `mov ms, r5`, D bundles later `vmov x0, SCD`
SEM_A, SEM_B, SEM_C = (0, 4), (0, 3), (0, 2)
SEM_X5, SEM_R5 = 0x200, 0x300


def sem_case(kind: str, d: int):
    from harness.topology import Cascade, Core, CoreStream, TileRef, Topology
    if not 1 <= d <= 12:
        raise ValueError("D must be 1..12")
    read = "  vmov x0, SCD"
    partner = {"semc": "  vmov MCD, x5", "semcr": "  vmov MCD, x5",
               "sems": "  mov ms, r5", "semsr": "  mov ms, r5"}[kind]
    first, second = (partner, read) if kind.endswith("r") else (read, partner)
    stream = kind.startswith("sems")
    a = _frame(base._enable("crMCDEn") + [f"  movxm r1, #{_word_value(0)}", *base._nops(2),
                                          "  vbcst.32 x0, r1", *base._nops(4), "  vmov MCD, x0"])
    b_pre = base._enable("crSCDEn") + ([] if stream else base._enable("crMCDEn"))
    b_pre += ["  mov p1, p0", f"  padda [p1], #{TS_BYTES}", "  mov r1, #7", f"  movxm r5, #{SEM_R5}",
              *base._nops(2), "  vbcst.32 x0, r1", f"  movxm r1, #{SEM_X5}", *base._nops(2),
              "  vbcst.32 x5, r1", *base._nops(6)]
    b_prog = ["@ts", first, *[N] * (d - 1), second, "@ts", *[N] * 12, "@ts"]
    if not stream:
        b_prog += ["  vmov MCD, x0"]          # forward the word B read, after the pair
    prog, _ = expand(b_prog, ["x0"])
    b = _frame(b_pre + prog)
    if stream:
        c_body = ["  mov r2, SS", *[N] * 8, f"  st r2, [p0, #{TS_BYTES}]"]
        c = _frame(c_body)
        c_out = TS_BYTES + 16
    else:
        c_body = base._enable("crSCDEn") + ["  mov p1, p0", f"  padda [p1], #{TS_BYTES}", "  mov r1, #7",
                                            *base._nops(2), "  vbcst.32 x0, r1", "  vbcst.32 x2, r1",
                                            *base._nops(4)]
        prog, _ = expand(["  vmov x0, SCD", *[N] * 8, "  vmov x2, SCD", *[N] * 8], ["x0", "x2"])
        c = _frame(c_body + prog)
        c_out = TS_BYTES + 2 * WORD
    links = [Cascade("a", "b"), CoreStream("b", "c") if stream else Cascade("b", "c")]
    name = f"{kind}_d{d}"
    topo = Topology(name=name, cores=(
        Core("a", TileRef(*SEM_A), Path("a.s"), args=(Buffer("ts", "out", 16),)),
        Core("b", TileRef(*SEM_B), Path("b.s"), args=(Buffer("log", "out", TS_BYTES + WORD),)),
        Core("c", TileRef(*SEM_C), Path("c.s"), args=(Buffer("log", "out", c_out),))),
        links=tuple(links))
    meta = {"family": kind, "params": {"d": d}, "direction": "chain"}
    return (topo, {"a.s": a, "b.s": b, "c.s": c}), meta


def analyze_sem(meta: dict, run: Path) -> dict:
    kind, d = meta["family"], meta["params"]["d"]
    b = (run / "b.log.bin").read_bytes()
    c = (run / "c.log.bin").read_bytes()
    ts = stamps(b, 3)
    r = {"b_word": hex(words(b, 1)[0]), "pair_cycles": ts[1] - ts[0] - (d + 1)}
    if kind.startswith("sems"):
        r["c"] = [hex(struct.unpack_from("<I", c, TS_BYTES)[0])]
        r["ok"] = r["b_word"] == hex(_word_value(0)) and r["c"] == [hex(SEM_R5)]
    else:
        r["c"] = [hex(w) for w in words(c, 2)]
        r["ok"] = r["b_word"] == hex(_word_value(0)) and r["c"] == [hex(SEM_X5), hex(_word_value(0))]
    return r


# -- analysis ----------------------------------------------------------------

def stamps(buf: bytes, n: int) -> list[int]:
    return list(struct.unpack_from(f"<{n}I", buf))


def words(buf: bytes, n: int) -> list[int]:
    """First lane of each read record (every lane carries the same value)."""
    return [struct.unpack_from("<I", buf, TS_BYTES + i * WORD)[0] for i in range(n)]


def analyze(meta: dict, prod: bytes, cons: bytes) -> dict:
    P = stamps(prod, meta["producer_ts"])
    C = stamps(cons, meta["consumer_ts"])
    W = words(cons, meta["words"])
    ok = W == [_word_value(i) for i in range(meta["words"])]
    d = lambda a, b: (a - b) & 0xFFFFFFFF if a >= b else -((b - a) & 0xFFFFFFFF)  # noqa: E731
    r = {"data_ok": ok, "sync_wait": d(C[1], C[0]), "fwd_sync": d(C[1], P[0])}
    fam, p = meta["family"], meta["params"]
    if fam == "rel":
        # natural spacing C1 -> C2: the stamp bundle, M nops, the read, 4 nops, the stamp
        r["cons_stall"] = d(C[2], C[1]) - (1 + p["m"] + HOLD_AT)
        r["prod"] = [d(P[i + 1], P[i]) for i in range(len(P) - 1)]
        r["fwd_data"] = d(C[2], P[2])            # consumer release - producer post-write stamp
    elif fam == "wrel":
        r["prod_stall"] = d(P[3], P[2]) - (p["n"] + 2)
        r["cons"] = [d(C[i + 1], C[i]) for i in range(len(C) - 1)]
        r["bwd"] = d(P[3], C[2])                 # producer release - consumer pop stamp
    elif fam == "rt":
        r["cons"] = [d(C[i + 1], C[i]) for i in range(1, len(C) - 1)]  # C1 -> first hold, then per bundle
        r["prod"] = [d(P[i + 1], P[i]) for i in range(len(P) - 1)]
    elif fam in ("mf1", "mf2"):
        r["seen"] = struct.unpack_from("<I", cons, 4 * MEM_SLOT)[0] == MEM_VALUE
    elif fam in ("mr1", "mr2"):
        r["seen"] = struct.unpack_from("<I", prod, 4 * MEM_SLOT)[0] == MEM_VALUE
    return r


# -- command line ------------------------------------------------------------

def expand_specs(specs: list[str]) -> list[str]:
    """`rel_n0..30_m0` style ranges expand to one spec per value."""
    out = []
    for s in specs:
        parts = s.split("_")
        for i, part in enumerate(parts):
            if ".." in part:
                lo, hi = part[1:].split("..")
                for v in range(int(lo), int(hi) + 1):
                    out += expand_specs(["_".join(parts[:i] + [f"{part[0]}{v}"] + parts[i + 1:])])
                break
        else:
            out.append(s)
    return out


def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    names = []
    for spec in expand_specs(args.specs):
        if spec.startswith("sem"):
            fam, dpart = spec.split("_")
            (topo, files), meta = sem_case(fam, int(dpart[1:]))
            suite.write_case(out, topo.name, topo, files=files, meta=meta)
            names.append(topo.name)
            continue
        for direction in args.directions:
            (topo, files), meta = build_case(spec, direction, args.sync_iters)
            suite.write_case(out, topo.name, topo, files=files, meta=meta)
            names.append(topo.name)
    suite.write_suite_json(out, "cascade-timing", "gen_cascade_timing_probe.py", names)
    print(f"{len(names)} cases -> {out}")
    return 0


def cmd_analyze(args) -> int:
    s = suite.load_suite(args.suite_dir)
    for case in s.cases:
        run = Path(args.suite_dir) / "runs" / args.tag / case
        meta = s.case_meta(case)
        if meta["family"].startswith("sem"):
            try:
                r = analyze_sem(meta, run)
            except FileNotFoundError:
                print(f"{case:28} (no output)")
                continue
            print(f"{case:28} " + " ".join(f"{k}={v}" for k, v in r.items()))
            continue
        try:
            prod = (run / "producer.ts.bin").read_bytes()
            cons = (run / "consumer.log.bin").read_bytes()
        except FileNotFoundError:
            print(f"{case:28} (no output)")
            continue
        r = analyze(s.case_meta(case), prod, cons)
        print(f"{case:28} " + " ".join(f"{k}={v}" for k, v in r.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("specs", nargs="+", help="e.g. rel_n0..24_m0 wrel_n0..10_m4 rt_w80_g0_r4")
    g.add_argument("--directions", nargs="+", default=["south"])
    g.add_argument("--sync-iters", type=int, default=SYNC_ITERS)
    g.set_defaults(fn=cmd_generate)
    a = sub.add_parser("analyze")
    a.add_argument("suite_dir")
    a.add_argument("tag")
    a.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
