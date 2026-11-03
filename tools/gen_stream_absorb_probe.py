#!/usr/bin/env python3
"""Core-to-core stream probes: how many words a held consumer lets the producer write, and
whether every word arrives. Writes a harness suite (harness/suite.py).

A producer core issues K blocking `mov ms, r` writes (distinct words 0x57000000 | i) with a
cycle-counter stamp after each; the consumer, one tile below, spins a known delay and then reads K
words with `mov r, SS`, storing each. Two routes per K: straight (`Core:0 -> Core:0`) and through the
producer tile's stream-switch FIFO 0 (`CoreStream.via_fifo`, AM020 ch.2:74).

What it pins (silicon vs emulator):
  - every word arrives, in order (the emulator dropped words a full Core slave port refused, then
    left the stalled producer asleep: the wedge of docs/fidelity-gaps/dma-stream-resources.md),
  - the words absorbed before the first producer stall (`analyze_depth` of the stamp deltas), with
    and without the FIFO: the FIFO route adds a fixed number of words.

Buffer layouts (little-endian u32)
  producer "ts":  [t_start, t_after_write_0, ..., t_after_write_{K-1}], zero padded to 16 bytes
  consumer "log": [t_start, t_after_delay, t_after_read_0, ...] padded to 16 bytes, then K words
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from harness import suite                                                  # noqa: E402
from harness.topology import Buffer, Core, CoreStream, TileRef, Topology  # noqa: E402
from gen_cascade_probe import (                                            # noqa: E402
    MAX_DELAY_NOPS, _frame, _nops, _round16, _timestamp_store, analyze_depth, delay_loop,
)

SS_NOPS = 8                     # `mov r, SS` result latency 7 (II_MOV_SS), plus one
WORD_TAG = 0x57000000
PRODUCER = (0, 3)               # directly north of the consumer: the flow crosses one tile boundary
CONSUMER = (0, 2)               # unless routed through the producer's FIFO


def word(i: int) -> int:
    return WORD_TAG | i


def ts_bytes(k: int) -> int:
    return _round16((k + 1) * 4)


def log_bytes(k: int) -> int:
    return _round16((k + 2) * 4) + k * 4


def producer_asm(k: int) -> str:
    body = [f"  movxm r1, #{word(0):#x}", *_nops(4), "  mov p1, p0", *_nops(2)]
    body += _timestamp_store()                              # t_start
    for _ in range(k):
        body += ["  mov ms, r1"]
        body += _timestamp_store()
        body += ["  add r1, r1, #1"]
    return _frame(body)


def consumer_asm(k: int, delay_nops: int, delay_iters: int = 0) -> str:
    if not 1 <= delay_nops <= MAX_DELAY_NOPS:
        raise ValueError(f"delay_nops must be 1..{MAX_DELAY_NOPS}")
    body = ["  mov p1, p0", f"  padda [p1], #{_round16((k + 2) * 4)}", *_nops(2)]
    body += _timestamp_store()                              # t_start
    body += _nops(delay_nops)
    if delay_iters:
        body += delay_loop(delay_iters, ".Ldelay")
    body += _timestamp_store()                              # t_after_delay
    for _ in range(k):
        body += ["  mov r2, SS", *_nops(SS_NOPS), "  st r2, [p1, #0]", "  padda [p1], #4",
                 "  mov r17:r16, cntr", *_nops(2), "  st r16, [p0, #0]", "  padda [p0], #4"]
    return _frame(body)


def case(k: int, delay_nops: int, via_fifo: bool, delay_iters: int = 0):
    name = f"{'fifo' if via_fifo else 'direct'}_k{k}_d{delay_nops}" + (f"_L{delay_iters}" if delay_iters else "")
    topo = Topology(
        name=name,
        cores=(Core("producer", TileRef(*PRODUCER), Path("producer.s"),
                    args=(Buffer("ts", "out", ts_bytes(k)),)),
               Core("consumer", TileRef(*CONSUMER), Path("consumer.s"),
                    args=(Buffer("log", "out", log_bytes(k)),))),
        links=(CoreStream("producer", "consumer", via_fifo=via_fifo),))
    return topo, {"producer.s": producer_asm(k), "consumer.s": consumer_asm(k, delay_nops, delay_iters)}


def consumer_words(consumer_log: bytes, k: int) -> list[int]:
    return list(struct.unpack_from(f"<{k}I", consumer_log, _round16((k + 2) * 4)))


def expected_words(k: int) -> list[int]:
    return [word(i) for i in range(k)]


def summarize(producer_ts: bytes, consumer_log: bytes, k: int) -> dict:
    d = analyze_depth(producer_ts, k)
    return {"absorbed": d.buffered, "pitch": d.pitch, "jump_cycles": d.jump_cycles,
            "all_words_in_order": consumer_words(consumer_log, k) == expected_words(k)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the suite")
    g.add_argument("out_dir")
    g.add_argument("--k", type=int, nargs="+", default=[96])
    g.add_argument("--delays", type=int, nargs="+", default=[1000],
                   help="consumer straight-line nop bundles before its first read")
    g.add_argument("--loop-iters", type=int, default=3000,
                   help="consumer counted-loop iterations after the nops (about 10 cycles each); the cores "
                        "start up to a few thousand cycles apart, so the hold must outlast that skew")
    g.set_defaults(fn=lambda a: generate(a))
    s = sub.add_parser("summarize", help="absorb count and integrity of one run directory")
    s.add_argument("run_dir")
    s.add_argument("k", type=int)
    s.set_defaults(fn=lambda a: print(summarize((Path(a.run_dir) / "producer.ts.bin").read_bytes(),
                                                (Path(a.run_dir) / "consumer.log.bin").read_bytes(), a.k)) or 0)
    args = ap.parse_args(argv)
    return args.fn(args)


def generate(args) -> int:
    out = Path(args.out_dir)
    names = []
    for k in args.k:
        for d in args.delays:
            for via_fifo in (False, True):
                topo, files = case(k, d, via_fifo, args.loop_iters)
                suite.write_case(out, topo.name, topo, files=files,
                                 meta={"k": k, "delay_nops": d, "loop_iters": args.loop_iters,
                                       "via_fifo": via_fifo})
                names.append(topo.name)
    suite.write_suite_json(out, "stream-absorb", "gen_stream_absorb_probe.py", names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
