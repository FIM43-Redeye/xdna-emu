#!/usr/bin/env python3
"""Core launch probe: where a core starts against the host runtime sequence.

Each core stamps its timer (`mov rH:rL, cntr`) at its fourth kernel instruction (after saving r16/r17), then
spins reading it until the value runs backwards (a timer reset) or a bounded number of
iterations pass. The runtime sequence resets every probe core's timer with a host write
(`Timer_Control.Reset`, bit 31) at a chosen position, with padding writes before and/or
after it, ahead of the output transfer. So per core:

- the reset arrived after the core started: `jumped` = 1, `before` is the last stamp read
  before the reset and `after` the first one after it; the core started
  `before - t0` (+ up to one spin iteration) cycles ahead of the reset write;
- the reset arrived before the core started: `jumped` = 0 and `t0` counts the cycles from
  the reset write to the kernel's first instruction.

Sweeping the padding before the reset (`pre<N>`) against the padding after it (`post<N>`)
tells whether the start is tied to the start of the runtime sequence or to its end
(the first transfer). Padding writes go to the core tile's Performance_Counter3, which
nothing reads.

Output words (8 x u32 per core): t0 low, t0 high, before, after low, after high, jumped,
iterations left, 0.

Run: python3 tools/gen_launch_probe.py <suite_dir>; then `python3 -m harness suite build/hw/emu`
(PYTHONPATH=tools); `python3 tools/gen_launch_probe.py <suite_dir> --analyze <tag>` prints each
core's lead (cycles it started ahead of the reset write) per repeat.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import suite  # noqa: E402
from harness.topology import Buffer, Core, SequenceWrite, TileRef, Topology  # noqa: E402

TIMER_CONTROL = 0x34000          # core module, AM025
TIMER_RESET = 1 << 31            # Timer_Control.Reset: "Reset Timer by writing 1"
PERFORMANCE_COUNTER3 = 0x3152C   # core module; padding target nothing reads
SPIN_ITERS = 60000               # about 20 cycles each: a bounded wait of ~1.2 M cycles
OUT_WORDS = 8

N = "  nop"


def kernel_asm(spin_iters: int = SPIN_ITERS) -> str:
    """The stamping kernel. p0 = the output buffer. The stamps use the r17:r16 / r19:r18
    pairs (`cntr` moves only into r16 and up), saved and restored around the kernel: the
    aiecc main keeps live values there (clobbering them hung the core in a first try).

    Branch targets sit after `.p2align 4` (silicon resumes a taken branch at the start of
    its target's 16-byte fetch line), and every branch is followed by its five delay slots
    as nops, so nothing in a delay slot changes the recorded values.
    """
    if not 1 <= spin_iters < 1 << 20:
        raise ValueError("spin iterations out of range")
    lines = [
        ".text", ".globl test_kernel", "test_kernel:",
        "  paddb [sp], #64",
        "  st r16, [sp, #-4]",
        "  st r17, [sp, #-8]",
        "  mov r17:r16, cntr", N, N,        # t0 (fourth instruction)
        "  st r18, [sp, #-12]",
        "  st r19, [sp, #-16]",
        "  mov r2, r16",                     # prev
        f"  movxm r3, #{spin_iters}", N, N,
        ".p2align 4", ".Lspin:",
        "  mov r19:r18, cntr", N, N,        # t
        "  ltu r6, r18, r2", N, N,          # t < prev: the timer was reset
        "  jnz r6, #.Ljump", *[N] * 5,
        "  mov r2, r18",
        "  add r3, r3, #-1", N, N,
        "  jnz r3, #.Lspin", *[N] * 5,
        "  mov r7, #0", N, N,
        "  j #.Lstore", *[N] * 5,
        ".p2align 4", ".Ljump:",
        "  mov r7, #1", N, N,
        ".p2align 4", ".Lstore:",
        "  mov r6, #0", N, N,
        "  st r16, [p0, #0]",
        "  st r17, [p0, #4]",
        "  st r2, [p0, #8]",
        "  st r18, [p0, #12]",
        "  st r19, [p0, #16]",
        "  st r7, [p0, #20]",
        "  st r3, [p0, #24]",
        "  st r6, [p0, #28]",
        "  lda r16, [sp, #-4]",
        "  lda r17, [sp, #-8]",
        "  lda r18, [sp, #-12]",
        "  lda r19, [sp, #-16]", *[N] * 7,
        "  paddb [sp], #-64",
        "  ret lr", *[N] * 5,
    ]
    return "\n".join(lines) + "\n"


def sequence(tiles: list[tuple[int, int]], pre: int, post: int) -> tuple[SequenceWrite, ...]:
    pad_tile = tiles[0]
    pad = [SequenceWrite(*pad_tile, PERFORMANCE_COUNTER3, k) for k in range(max(pre, post))]
    resets = [SequenceWrite(c, r, TIMER_CONTROL, TIMER_RESET) for c, r in tiles]
    return tuple(pad[:pre] + resets + pad[:post])


def write_suite(out: Path) -> list[str]:
    names: list[str] = []

    def case(name: str, tiles: list[tuple[int, int]], pre: int = 0, post: int = 0) -> None:
        cores = tuple(Core(f"c{c}{r}", TileRef(c, r), Path("k.s"),
                           args=(Buffer("ts", "out", 4 * OUT_WORDS),)) for c, r in tiles)
        topo = Topology(name, cores, sequence_writes=sequence(tiles, pre, post))
        suite.write_case(out, name, topo, files={"k.s": kernel_asm()},
                         meta={"tiles": tiles, "pre": pre, "post": post})
        names.append(name)

    one = [(0, 2)]
    case("base", one)
    for n in (32, 128, 512):
        case(f"pre{n}", one, pre=n)
        case(f"post{n}", one, post=n)
    case("col_23", [(0, 2), (0, 3)])
    case("col_2345", [(0, 2), (0, 3), (0, 4), (0, 5)])
    case("row_01", [(0, 2), (1, 2)])
    suite.write_suite_json(out, "launch", "tools/gen_launch_probe.py", names)
    return names


def lead(words: tuple[int, ...]) -> int:
    """Cycles a core started ahead of the reset write (negative: after it)."""
    t0, before, jumped = words[0], words[2], words[5]
    return (before - t0) & 0xFFFFFFFF if jumped else -t0


def analyze(suite_dir: Path, tag: str) -> list[str]:
    """One line per case: each core's lead over every repeat, last core first in the gaps."""
    meta = json.loads((suite_dir / "suite.json").read_text())
    lines = []
    for case in meta["cases"]:
        base = suite_dir / "runs" / tag / case
        runs = [base] + sorted(base.glob("repeat_*"))
        rows = []
        for r in runs:
            files = sorted(r.glob("*.ts.bin"))
            if files:
                rows.append({f.name.split(".")[0]: lead(struct.unpack("<8I", f.read_bytes()[:32]))
                             for f in files})
        if not rows:
            lines.append(f"{case:10} no output")
            continue
        cores = sorted(rows[0])
        text = " | ".join(f"{k}: {[r.get(k) for r in rows]}" for k in cores)
        if len(cores) > 1:
            gaps = [[r[cores[i]] - r[cores[i + 1]] for i in range(len(cores) - 1)]
                    for r in rows if all(k in r for k in cores)]
            text += f"  gaps {gaps}"
        lines.append(f"{case:10} {text}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite_dir", type=Path)
    ap.add_argument("--analyze", metavar="TAG",
                    help="print each core's start lead over the reset write for run TAG")
    a = ap.parse_args(argv)
    if a.analyze:
        print("\n".join(analyze(a.suite_dir, a.analyze)))
        return 0
    names = write_suite(a.suite_dir)
    print(f"{len(names)} cases in {a.suite_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
