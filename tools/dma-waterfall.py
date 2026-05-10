#!/usr/bin/env python3
"""dma-waterfall.py -- Per-channel DMA FSM phase residency timeline.

Parses an EMU bridge log (the `<test>.<compiler>.bridge.log` file produced
by `scripts/emu-bridge-test.sh`) and reconstructs a per-channel timeline of
DMA FSM phases, plus a summary table of cycles spent in each phase.

The DMA stepping logger emits one INFO line per FSM phase transition:

    DMA(c,r) ch<n>: <PhaseBefore> -> <PhaseAfter> cycle=<N>

Between two consecutive transitions for the same channel, the channel is in
the "after" state of the earlier line; we treat that span as time-in-state.

Use cases:
- Identify which channel/phase dominates pipeline fill for #355a calibration.
- Spot phases that complete in zero cycles (likely missing model coverage).
- Compare timing across compilers / changes to the FSM.

Usage:
    dma-waterfall.py <bridge.log>
    dma-waterfall.py <bridge.log> --tiles 1,0:0 1,1:0,1,7 1,2:0,2  # filter
    dma-waterfall.py <bridge.log> --max-cycle 3000                  # window
    dma-waterfall.py <bridge.log> --csv                             # raw CSV
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LINE_RE = re.compile(
    r"DMA\((?P<col>\d+),(?P<row>\d+)\)\s+ch(?P<ch>\d+):\s+"
    r"(?P<before>\w+)\s+->\s+(?P<after>\w+)\s+cycle=(?P<cycle>\d+)"
)


@dataclass
class Transition:
    col: int
    row: int
    ch: int
    before: str
    after: str
    cycle: int


def parse_log(path: Path) -> list[Transition]:
    out: list[Transition] = []
    with path.open() as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                out.append(Transition(
                    col=int(m.group("col")),
                    row=int(m.group("row")),
                    ch=int(m.group("ch")),
                    before=m.group("before"),
                    after=m.group("after"),
                    cycle=int(m.group("cycle")),
                ))
    return out


def per_channel_segments(
    transitions: Iterable[Transition], end_cycle: int
) -> dict[tuple[int, int, int], list[tuple[str, int, int]]]:
    """For each (col, row, ch), produce [(state, start_cycle, end_cycle), ...].

    Each transition's `after` state runs from its `cycle` until the channel's
    next transition's `cycle` (or `end_cycle` for the last segment).

    The pre-first-transition state is unknown (the channel may have been in
    Idle since cycle 0, or in some setup state). We emit it as "Pre" for
    transparency rather than guessing.
    """
    by_channel: dict[tuple[int, int, int], list[Transition]] = defaultdict(list)
    for t in transitions:
        by_channel[(t.col, t.row, t.ch)].append(t)

    segments: dict[tuple[int, int, int], list[tuple[str, int, int]]] = {}
    for key, ts in by_channel.items():
        ts.sort(key=lambda t: t.cycle)
        segs: list[tuple[str, int, int]] = []
        if ts[0].cycle > 0:
            # Span before first observed transition. The `before` field of the
            # first transition tells us what state the channel was in.
            segs.append((ts[0].before, 0, ts[0].cycle))
        for i, t in enumerate(ts):
            end = ts[i + 1].cycle if i + 1 < len(ts) else end_cycle
            segs.append((t.after, t.cycle, end))
        segments[key] = segs
    return segments


def phase_summary(
    segments: dict[tuple[int, int, int], list[tuple[str, int, int]]],
) -> dict[tuple[int, int, int], dict[str, int]]:
    """Total cycles per (col, row, ch) per phase."""
    summary: dict[tuple[int, int, int], dict[str, int]] = {}
    for key, segs in segments.items():
        ph: dict[str, int] = defaultdict(int)
        for state, start, end in segs:
            ph[state] += max(0, end - start)
        summary[key] = dict(ph)
    return summary


def parse_filter(spec: str) -> dict[tuple[int, int], set[int]]:
    """Parse `c,r:ch1,ch2 c,r:ch3` style filter into {(c,r): {ch...}}."""
    out: dict[tuple[int, int], set[int]] = {}
    for part in spec.split():
        loc, chs = part.split(":", 1)
        col, row = map(int, loc.split(","))
        out[(col, row)] = set(int(c) for c in chs.split(","))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("log", type=Path, help="bridge log file")
    p.add_argument("--tiles", default="",
                   help="filter spec, e.g. '1,0:0 1,1:0,1,7 1,2:0,2' "
                        "(tile col,row : channel list, space-separated tiles)")
    p.add_argument("--max-cycle", type=int, default=None,
                   help="restrict timeline to cycles <= this value")
    p.add_argument("--csv", action="store_true",
                   help="emit raw segment CSV instead of summary tables")
    args = p.parse_args(argv)

    if not args.log.exists():
        print(f"error: log not found: {args.log}", file=sys.stderr)
        return 2

    transitions = parse_log(args.log)
    if not transitions:
        print("warning: no DMA transitions found in log", file=sys.stderr)
        return 0

    end_cycle = (
        args.max_cycle
        if args.max_cycle is not None
        else max(t.cycle for t in transitions)
    )
    if args.max_cycle is not None:
        transitions = [t for t in transitions if t.cycle <= end_cycle]

    tile_filter = parse_filter(args.tiles) if args.tiles else None
    if tile_filter:
        transitions = [
            t for t in transitions
            if (t.col, t.row) in tile_filter and t.ch in tile_filter[(t.col, t.row)]
        ]

    segments = per_channel_segments(transitions, end_cycle)
    summary = phase_summary(segments)

    if args.csv:
        print("col,row,ch,phase,start,end,duration")
        for (col, row, ch), segs in sorted(segments.items()):
            for state, start, end in segs:
                print(f"{col},{row},{ch},{state},{start},{end},{end - start}")
        return 0

    # Print summary table.
    print(f"# DMA waterfall summary (end_cycle={end_cycle})")
    print(f"# {len(segments)} channels, {sum(len(s) for s in segments.values())} segments")
    print()
    all_phases = sorted({ph for ch_summary in summary.values() for ph in ch_summary})
    header = ["tile", "ch"] + all_phases + ["total"]
    widths = [max(len(h), 8) for h in header]

    def fmt_row(row: list[str]) -> str:
        return "  ".join(c.rjust(w) for c, w in zip(row, widths))

    print(fmt_row(header))
    print(fmt_row(["-" * w for w in widths]))
    for (col, row, ch) in sorted(summary):
        row_data = [f"({col},{row})", str(ch)]
        total = 0
        for ph in all_phases:
            v = summary[(col, row, ch)].get(ph, 0)
            row_data.append(str(v) if v else "")
            total += v
        row_data.append(str(total))
        print(fmt_row(row_data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
