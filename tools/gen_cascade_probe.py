#!/usr/bin/env python3
"""Cascade probes for the topology harness: smoke (data and order) and depth.

Writes harness topology JSON plus hand-written `.s` kernels; everything runs
through the harness CLI (`python3 -m harness build|hw|emu`). Two cores, one
`aie.cascade_flow`, producer on the north (flow South) or west (flow East) side.

Smoke
-----
The producer streams N distinct 512-bit words (read from its input buffer, so
every lane differs) with `vmov MCD, x0` (form "x") or through an accumulator
register (form "bm": `vmov bml0, x0` then `vmov MCD, bml0`); the consumer
stores each `vmov x0, SCD` result (form "bm": via `bml1`) to its output.
Expected consumer output == producer input, in order.

Depth (total cascade buffering between producer and consumer)
-------------------------------------------------------------
The consumer waits a known number of cycles (`delay_nops` straight-line nops) before its first
read, so the cascade can only drain after that. The producer issues K
back-to-back `vmov MCD` writes, cycling through registers x0..x11 (distinct
values), and reads the core cycle counter (`MOV_CNTR`: `mov r17:r16, cntr`,
low word kept) after each write. Writes that find room in the cascade buffers
complete at the issue pitch; the first write that finds them full stalls until
the consumer starts reading, so the timestamp after it jumps by about the
consumer delay. The number of writes before the jump is the total buffering.
Absolute counter values are meaningless (the counters are per tile); only
differences within one core are used.

Both sides always terminate: the consumer reads exactly as many words as the
producer writes (K), so a stalled producer is released once the delay ends. The
producer takes no input buffer, so it is not held back by a DMA fill and starts
at about the same time as the consumer.

Buffer layouts (little-endian u32 unless noted)
  producer "ts": [t_start, t_after_write_0, ..., t_after_write_{K-1}], zero padded
  consumer "log": [t_start, t_after_delay, t_after_read_0, ...] padded to 16 bytes,
                  then K x 64 bytes of read data
"""
from __future__ import annotations

import argparse
import statistics
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from harness.topology import (Buffer, Cascade, Core, TileRef, Topology,    # noqa: E402
                              to_dict)

WORD_BYTES = 64                 # one cascade word: 512 bits
DEPTH_WORD_REGS = 12            # x0..x11 hold distinct words in the depth probe
LOAD_NOPS = 6
SCD_NOPS = 8                    # VMOV_CASCADE_READ latency 7, plus one
MAX_DELAY_NOPS = 6000           # 2 bytes per nop bundle in 16 KiB of program memory
JUMP_MIN_CYCLES = 100           # a stall shorter than this is not distinguishable

# direction -> ((producer col, row), (consumer col, row)); the producer is the
# cascade source, directly North (flow South) or West (flow East) of the consumer.
PLACEMENT = {
    "south": ((0, 3), (0, 2)),
    "east": ((0, 2), (1, 2)),
}


# -- asm building blocks -----------------------------------------------------

def _frame(body: list[str]) -> str:
    """A returning function that saves r16-r19 (the counter pair uses them)."""
    pro = [".text", ".globl test_kernel", "test_kernel:",
           "  paddb [sp], #64"]
    pro += [f"  st r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 20)]
    epi = [f"  lda r{r}, [sp, #{-4 * (r - 15)}]" for r in range(16, 20)]
    epi += ["  nop"] * 7
    epi += ["  paddb [sp], #-64", "  ret lr"] + ["  nop"] * 5
    return "\n".join(pro + body + epi) + "\n"


def _nops(n: int) -> list[str]:
    return ["  nop"] * n


def _timestamp_store(offset_reg: str = "p0") -> list[str]:
    """Read the cycle counter and store its low word at [p0], then advance p0."""
    return ["  mov r17:r16, cntr", "  nop", "  nop",
            f"  st r16, [{offset_reg}, #0]", f"  padda [{offset_reg}], #4"]


def _enable(reg_name: str) -> list[str]:
    return ["  mov r0, #1", *_nops(2), f"  mov {reg_name}, r0", *_nops(2)]


# -- smoke -------------------------------------------------------------------

def producer_smoke_asm(n_words: int, form: str = "x") -> str:
    _check_form(form)
    body = _enable("crMCDEn")
    for _ in range(n_words):
        body += ["  vlda wl0, [p0, #0]", "  vlda wh0, [p0, #32]", *_nops(LOAD_NOPS)]
        if form == "bm":
            body += ["  vmov bml0, x0", *_nops(3), "  vmov MCD, bml0"]
        else:
            body += ["  vmov MCD, x0"]
        body += [f"  padda [p0], #{WORD_BYTES}"]
    return _frame(body)


def consumer_smoke_asm(n_words: int, form: str = "x") -> str:
    _check_form(form)
    body = _enable("crSCDEn")
    for _ in range(n_words):
        if form == "bm":
            body += ["  vmov bml1, SCD", *_nops(SCD_NOPS), "  vmov x0, bml1", *_nops(3)]
        else:
            body += ["  vmov x0, SCD", *_nops(SCD_NOPS)]
        body += ["  vst wl0, [p0, #0]", "  vst wh0, [p0, #32]",
                 f"  padda [p0], #{WORD_BYTES}"]
    return _frame(body)


def _check_form(form: str) -> None:
    if form not in ("x", "bm"):
        raise ValueError(f"unknown register form {form!r} (x or bm)")


def smoke_data(n_words: int) -> bytes:
    """N distinct words; every u32 lane of every word is distinct."""
    lanes = [0xC0000000 | (i << 8) | lane for i in range(n_words) for lane in range(16)]
    return struct.pack(f"<{len(lanes)}I", *lanes)


# -- depth -------------------------------------------------------------------

def depth_word_value(i: int) -> int:
    """Value broadcast in every lane of write i (registers are reused cyclically)."""
    return (i % DEPTH_WORD_REGS) + 1


def _round16(n: int) -> int:
    return (n + 15) // 16 * 16


def depth_producer_out_bytes(k: int) -> int:
    return _round16((k + 1) * 4)


def _consumer_ts_bytes(k: int) -> int:
    """Timestamp region size; the data after it must be 32-byte aligned because silicon
    clears the low 5 address bits of `vst` (the emulator does not: permissive-vs-hw.md)."""
    return (((k + 2) * 4) + 31) // 32 * 32


def depth_consumer_out_bytes(k: int) -> int:
    return _consumer_ts_bytes(k) + k * WORD_BYTES


def producer_depth_asm(k: int, pre_delay_nops: int = 0) -> str:
    """K back-to-back writes with a timestamp after each.

    `pre_delay_nops` straight-line nop bundles run before the first write, so a
    consumer with a short delay is already blocked in its first SCD read when the
    first word arrives (the forward-latency measurement).
    """
    if not 0 <= pre_delay_nops <= MAX_DELAY_NOPS:
        raise ValueError(f"pre_delay_nops must be 0..{MAX_DELAY_NOPS}")
    body = _enable("crMCDEn")
    for j in range(DEPTH_WORD_REGS):
        body += [f"  mov r1, #{depth_word_value(j)}", *_nops(2), f"  vbcst.32 x{j}, r1"]
    body += _nops(8 + pre_delay_nops)
    body += _timestamp_store()                      # t_start
    for i in range(k):
        body += [f"  vmov MCD, x{i % DEPTH_WORD_REGS}"]
        body += _timestamp_store()
    return _frame(body)


def consumer_depth_asm(k: int, delay_nops: int) -> str:
    """Read K words after `delay_nops` straight-line nop bundles (one cycle each).

    The delay is straight-line code, not a `jnz` loop: on silicon any cascade probe
    with a `jnz` loop in either core hung (docs/fidelity-gaps/core-compute-timing.md),
    while the same delay as nops runs. Program memory (16 KiB) bounds it.
    """
    if not 1 <= delay_nops <= MAX_DELAY_NOPS:
        raise ValueError(f"delay_nops must be 1..{MAX_DELAY_NOPS}")
    body = _enable("crSCDEn")
    body += ["  mov p1, p0", f"  padda [p1], #{_consumer_ts_bytes(k)}", *_nops(2)]
    body += _timestamp_store()                      # t_start
    body += _nops(delay_nops)
    body += _timestamp_store()                      # t_after_delay
    for _ in range(k):
        body += ["  vmov x0, SCD", "  mov r17:r16, cntr", *_nops(SCD_NOPS),
                 "  st r16, [p0, #0]", "  padda [p0], #4",
                 "  vst wl0, [p1, #0]", "  vst wh0, [p1, #32]",
                 f"  padda [p1], #{WORD_BYTES}"]
    return _frame(body)


@dataclass(frozen=True)
class DepthResult:
    pitch: int                  # smallest per-write delta: the unstalled issue pitch
    buffered: int | None        # writes before the first stall, None if no write stalled
    jump_cycles: int            # extra cycles of the stalled write over the pitch (0 if none)


def analyze_depth(producer_ts: bytes, k: int) -> DepthResult:
    """Find the stalled write in a producer "ts" buffer."""
    ts = struct.unpack_from(f"<{k + 1}I", producer_ts)
    deltas = [(ts[i + 1] - ts[i]) & 0xFFFFFFFF for i in range(k)]
    pitch = min(deltas)
    peak = max(deltas)
    if peak - pitch < JUMP_MIN_CYCLES:
        return DepthResult(pitch, None, 0)
    threshold = pitch + (peak - pitch) / 2
    first = next(i for i, d in enumerate(deltas) if d >= threshold)
    return DepthResult(pitch, first, deltas[first] - pitch)


def consumer_depth_view(consumer_log: bytes, k: int) -> dict:
    """Consumer log: the delay it actually spun and the word values it read."""
    ts = struct.unpack_from(f"<{k + 2}I", consumer_log)
    data_at = _consumer_ts_bytes(k)
    words = []
    for i in range(k):
        word = struct.unpack_from("<16I", consumer_log, data_at + i * WORD_BYTES)
        words.append(word)
    return {"delay_cycles": (ts[1] - ts[0]) & 0xFFFFFFFF,
            "read_deltas": [(ts[i + 1] - ts[i]) & 0xFFFFFFFF for i in range(1, k + 1)],
            "words": words}


# -- topologies --------------------------------------------------------------

def _placement(direction: str):
    if direction not in PLACEMENT:
        raise ValueError(f"unknown direction {direction!r} ({', '.join(PLACEMENT)})")
    return PLACEMENT[direction]


def _pair(name: str, direction: str, out_dir: Path, prod_asm: str, cons_asm: str,
          prod_args: tuple[Buffer, ...], cons_args: tuple[Buffer, ...]) -> Topology:
    (pc, pr), (cc, cr) = _placement(direction)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    psrc, csrc = out_dir / f"{name}.producer.s", out_dir / f"{name}.consumer.s"
    psrc.write_text(prod_asm)
    csrc.write_text(cons_asm)
    return Topology(
        name=name,
        cores=(Core("producer", TileRef(pc, pr), psrc, args=prod_args),
               Core("consumer", TileRef(cc, cr), csrc, args=cons_args)),
        links=(Cascade("producer", "consumer"),))


def smoke_topology(direction: str, n_words: int, out_dir: Path, form: str = "x",
                   name: str | None = None) -> Topology:
    _placement(direction)
    size = n_words * WORD_BYTES
    return _pair(name or f"smoke_{direction}", direction, out_dir,
                 producer_smoke_asm(n_words, form), consumer_smoke_asm(n_words, form),
                 (Buffer("words", "in", size, data=smoke_data(n_words)),),
                 (Buffer("words", "out", size),))


def depth_topology(direction: str, k: int, delay_nops: int, out_dir: Path,
                   name: str | None = None, pre_delay_nops: int = 0) -> Topology:
    _placement(direction)
    suffix = f"_p{pre_delay_nops}" if pre_delay_nops else ""
    return _pair(name or f"depth_{direction}_k{k}_d{delay_nops}{suffix}", direction, out_dir,
                 producer_depth_asm(k, pre_delay_nops), consumer_depth_asm(k, delay_nops),
                 (Buffer("ts", "out", depth_producer_out_bytes(k)),),
                 (Buffer("log", "out", depth_consumer_out_bytes(k)),))


# -- command line ------------------------------------------------------------

def _write(t: Topology, out_dir: Path) -> None:
    import json
    (out_dir / f"{t.name}.json").write_text(json.dumps(to_dict(t), indent=2) + "\n")


def cmd_generate(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for direction in PLACEMENT:
        _write(smoke_topology(direction, args.smoke_words, out), out)
    _write(smoke_topology("south", args.smoke_words, out, form="bm",
                          name="smoke_south_bm"), out)
    for direction in PLACEMENT:
        for k in args.depth_k:
            for d in args.delays:
                _write(depth_topology(direction, k, d, out, pre_delay_nops=args.pre), out)
    return 0


def cmd_analyze(args) -> int:
    r = analyze_depth(Path(args.ts_file).read_bytes(), args.k)
    print(f"pitch={r.pitch} buffered={r.buffered} jump_cycles={r.jump_cycles}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write topology JSON and kernels")
    g.add_argument("out_dir")
    g.add_argument("--smoke-words", type=int, default=8)
    g.add_argument("--depth-k", type=int, nargs="+", default=[24])
    g.add_argument("--delays", type=int, nargs="+", default=[1500, 3000])
    g.add_argument("--pre", type=int, default=0,
                   help="producer nop bundles before the first write (forward-latency probes)")
    g.set_defaults(fn=cmd_generate)
    a = sub.add_parser("analyze", help="stalled-write analysis of a producer ts file")
    a.add_argument("ts_file")
    a.add_argument("k", type=int)
    a.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
