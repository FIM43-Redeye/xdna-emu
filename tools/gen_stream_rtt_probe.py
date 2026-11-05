#!/usr/bin/env python3
"""Core-to-core stream round-trip probe: the stream-switch transit latency per hop.
Writes a harness suite (harness/suite.py).

A "ping" core at (0,2) and a "pong" core N tiles north of it (N = 1..3) exchange one word R
times over two circuit flows (`Core:0 -> Core:0` each way). Ping stamps the cycle counter after
each blocking read of the echo, so consecutive stamp deltas are the round trip:

    rtt(N) = core overheads (fixed) + 2 * transit(N)

and `rtt(N+1) - rtt(N)` is twice the latency one more tile boundary adds (one external slave ->
external master traversal plus the wire). The data-race absorb probes (R4-D3,
`gen_stream_absorb_probe.py`) pin the buffering per hop; this pins the latency, the other half of
AM020 ch.2:84-92 ("4-cycle latency and 8-deep FIFO" external to external).

`--via-fifo` routes ping's flow through ping's switch FIFO 0 (`CoreStream.via_fifo`), so the
difference against the direct case is the FIFO route's added latency.

Buffer layouts (little-endian u32)
  ping "ts":  [t_start, t_after_echo_0, ..., t_after_echo_{R-1}], zero padded to 16 bytes
  pong "log": [t_start, t_after_read_0, ..., t_after_read_{R-1}], zero padded to 16 bytes
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from harness import suite                                                  # noqa: E402
from harness.topology import Buffer, Core, CoreStream, TileRef, Topology  # noqa: E402
from gen_cascade_probe import _frame, _nops, _round16, _timestamp_store    # noqa: E402

SS_NOPS = 8                     # `mov r, SS` result latency 7 (II_MOV_SS), plus one
WORD_TAG = 0x52000000
PING = (0, 2)
ROUNDS = 16


def ts_bytes(rounds: int) -> int:
    return _round16((rounds + 1) * 4)


def ping_asm(rounds: int) -> str:
    body = [f"  movxm r1, #{WORD_TAG:#x}", *_nops(4)]
    body += _timestamp_store()                              # t_start
    for _ in range(rounds):
        body += ["  mov ms, r1", "  mov r2, SS", *_nops(SS_NOPS)]
        body += _timestamp_store()
        body += ["  add r1, r1, #1"]
    return _frame(body)


def pong_asm(rounds: int) -> str:
    body = list(_timestamp_store())                         # t_start
    for _ in range(rounds):
        body += ["  mov r2, SS", *_nops(SS_NOPS), "  mov ms, r2"]
        body += _timestamp_store()
    return _frame(body)


def case(hops: int, via_fifo: bool, rounds: int = ROUNDS):
    name = f"rtt_h{hops}" + ("_fifo" if via_fifo else "")
    pong = (PING[0], PING[1] + hops)
    topo = Topology(
        name=name,
        cores=(Core("ping", TileRef(*PING), Path("ping.s"), args=(Buffer("ts", "out", ts_bytes(rounds)),)),
               Core("pong", TileRef(*pong), Path("pong.s"), args=(Buffer("log", "out", ts_bytes(rounds)),))),
        links=(CoreStream("ping", "pong", via_fifo=via_fifo), CoreStream("pong", "ping")))
    return topo, {"ping.s": ping_asm(rounds), "pong.s": pong_asm(rounds)}


def round_trips(ping_ts: bytes, rounds: int = ROUNDS) -> list[int]:
    ts = struct.unpack_from(f"<{rounds + 1}I", ping_ts)
    return [(ts[i + 1] - ts[i]) & 0xFFFFFFFF for i in range(rounds)]


def summarize(run_dir: Path, rounds: int = ROUNDS) -> dict:
    """Steady round trip: the deltas after the first two (core start skew lands in those)."""
    rtt = round_trips((run_dir / "ping.ts.bin").read_bytes(), rounds)
    steady = rtt[2:]
    return {"rtt": rtt, "steady": sorted(set(steady))}


def generate(args) -> int:
    out = Path(args.out_dir)
    names = []
    for hops in args.hops:
        for via_fifo in ((False, True) if args.fifo else (False,)):
            topo, files = case(hops, via_fifo)
            suite.write_case(out, topo.name, topo, files=files,
                             meta={"hops": hops, "via_fifo": via_fifo, "rounds": ROUNDS})
            names.append(topo.name)
    suite.write_suite_json(out, "stream-rtt", "gen_stream_rtt_probe.py", names)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write the suite")
    g.add_argument("out_dir")
    g.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    g.add_argument("--fifo", action="store_true", help="also write the via-FIFO variants")
    g.set_defaults(fn=generate)
    s = sub.add_parser("summarize", help="round trips of every case in one run tag")
    s.add_argument("suite_dir")
    s.add_argument("tag")
    s.set_defaults(fn=lambda a: [print(c.name, json.dumps(summarize(c)))
                                 for c in sorted((Path(a.suite_dir) / "runs" / a.tag).iterdir())
                                 if (c / "ping.ts.bin").exists()] and 0)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
