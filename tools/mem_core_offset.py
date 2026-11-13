#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Memory-module minus core-module timestamp offset per compute tile, emulator against silicon.

The compute tile's memory trace timer is reset through a different broadcast path than
the core's, so a tile's two trace units disagree on time by a small fixed number of
cycles. That number cannot be read from one trace (the events of the two units are
separated in time by the workload too), but the *difference between the emulator and
silicon* of it can: for every event row that both sides saw the same number of times,
pair the occurrences in order and take the median of (emulator - silicon) timestamps.
A tile's core rows give one median, its memory rows another; the memory median minus
the core median is how many cycles later (positive) or earlier (negative) the
emulator's memory unit sits relative to its core unit, compared with silicon. Zero
means the two units agree with silicon with respect to each other. Silicon's absolute
timestamps differ from the emulator's by the host-side start time, which both
medians contain and the subtraction cancels.

Held-condition (level) rows are left out: their timestamps are interval starts and
move with other things (PERF_CNT, *STARVATION*, *RUNNING*, *STALL*, *BACKPRESSURE*).
Rows with fewer than 2 occurrences, or different counts on the two sides, are left
out too. Tiles with no usable row on either unit are not listed.

usage: mem_core_offset.py --emu TREE [TREE ...] --silicon DIR [DIR ...] [--only TEST ...]

TREE and DIR are bridge result directories (``<test>.<compiler>.{emu,hw}/events.json``).
Silicon samples are the ``.hw`` sides of the --silicon directories; emulator results are the
``.emu`` sides of the --emu directories. One line per (test, tile) and tree with the
median over silicon samples; then, per tree, the distribution over all pairs.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

LEVEL_MARKERS = ("PERF_CNT", "STARVATION", "RUNNING", "STALL", "BACKPRESSURE")
CORE, MEM = 0, 1  # events.json pkt_type of a compute tile's two trace units


def load_tiles(side_dir):
    """{(col, row): {(pkt_type, name): [ts...]}} with columns/rows relative to the placement."""
    path = Path(side_dir) / "events.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    pl = doc.get("placement") or {}
    oc, orow = pl.get("origin_col", 0), pl.get("origin_row", 0)
    tiles = {}
    for e in doc.get("events", []):
        if e.get("pkt_type") not in (CORE, MEM):
            continue
        tile = (max(e["col"] - oc, 0), max(e["row"] - orow, 0))
        tiles.setdefault(tile, {}).setdefault((e["pkt_type"], e["name"]), []).append(e["ts"])
    return tiles


def unit_offset(emu_rows, hw_rows, unit):
    """Median paired (emu - hw) over the unit's usable rows, or None."""
    diffs = []
    for (pkt, name), ts in emu_rows.items():
        if pkt != unit or any(m in name for m in LEVEL_MARKERS):
            continue
        hs = hw_rows.get((pkt, name))
        if not hs or len(hs) != len(ts) or len(ts) < 2:
            continue
        diffs.append(statistics.median(a - b for a, b in zip(sorted(ts), sorted(hs))))
    return statistics.median(diffs) if diffs else None


def tile_offsets(emu, hw):
    """{tile: memory-minus-core offset} for tiles usable on both units."""
    out = {}
    for tile, emu_rows in emu.items():
        hw_rows = hw.get(tile)
        if not hw_rows:
            continue
        core = unit_offset(emu_rows, hw_rows, CORE)
        mem = unit_offset(emu_rows, hw_rows, MEM)
        if core is not None and mem is not None:
            out[tile] = mem - core
    return out


def measure(emu_tree, silicon_dirs, only=()):
    """{(test.compiler, tile): median over silicon samples of the offset}."""
    result = {}
    for emu_dir in sorted(Path(emu_tree).glob("*.emu")):
        key = emu_dir.name[: -len(".emu")]
        if only and not any(o in key for o in only):
            continue
        emu = load_tiles(emu_dir)
        if not emu:
            continue
        per_tile = {}
        for sd in silicon_dirs:
            hw = load_tiles(Path(sd) / f"{key}.hw")
            if not hw:
                continue
            for tile, off in tile_offsets(emu, hw).items():
                per_tile.setdefault(tile, []).append(off)
        for tile, offs in per_tile.items():
            result[(key, tile)] = statistics.median(offs)
    return result


def summarise(values):
    if not values:
        return "no usable tiles"
    ints = sorted(values)
    exact0 = sum(1 for v in ints if abs(v) < 0.5)
    return (f"{len(ints)} test-tiles; median {statistics.median(ints):g}; "
            f"|offset| mean {sum(abs(v) for v in ints) / len(ints):.2f}; within 0.5 of silicon: {exact0}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emu", nargs="+", required=True)
    ap.add_argument("--silicon", nargs="+", required=True)
    ap.add_argument("--only", nargs="*", default=())
    args = ap.parse_args(argv)
    results = {t: measure(t, args.silicon, args.only) for t in args.emu}
    keys = sorted(set().union(*[set(r) for r in results.values()]))
    for key in keys:
        cells = ["%8s" % ("%g" % results[t][key] if key in results[t] else "-") for t in args.emu]
        print(f"{key[0]} tile{key[1]} " + " ".join(cells))
    for t in args.emu:
        print(f"{t}: {summarise(list(results[t].values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
