#!/usr/bin/env python3
"""Measure MM2S egress-FIFO depth (Experiment B) from an mm2s_egress_depth
capture.

The delay from DMA_MM2S_0_STALLED_LOCK onset to DMA_MM2S_0_MEMORY_STARVATION
onset is the egress occupancy in beats at stall-time -- the exact mirror of
the S2MM ingress finding (docs/superpowers/findings/2026-07-14-dma-memory-
pressure-event-semantics.md), which pinned ingress depth at ~15-16 beats via
the same onset-delay recipe: `BACKPRESSURE = [lock_stall_start + 15,
lock_stall_end + 1)`.

A capture can contain several STALLED_LOCK windows (e.g. `cold`'s pre-BD0
wait AND its post-BD0 permanent stall, or a multi-rep `fill_stall` capture).
Each stall window is paired with the FIRST STARVATION onset at-or-after its
own start and before the NEXT stall's start, so a later window's starvation
is never misattributed to an earlier stall. A window is flagged INVALID if
any STREAM_BACKPRESSURE interval overlaps [stall_start, starvation_onset) --
a starvation caused by the stream itself being blocked is not a genuinely
empty FIFO and must not count toward depth. Depth for one capture is the MAX
onset_delay over valid windows; the dwell-sweep ceiling (across many
captures/variants) is read the same way by the CLI driver below.

Tile-aware keying. bankdisc_measure.load_intervals (the mode-0 B/E interval
rebuild used elsewhere in this experiment family) keys intervals by
(module_type, event_name) alone -- it recovers the module string
("mem(2,0)") from the process_name metadata but discards the tile
coordinates, keeping only the leading type token. That's harmless for the
single-tile variants (fill_stall, fetch_starve, cold, never_stall), which
have exactly one MM2S-emitting tile. But stream_backpressure and
dwell_sweep_K are TWO-tile designs (source aie.tile(0, 2), sink
aie.tile(0, 3) -- see mm2s_egress_depth.py) whose sink emits an IDENTICAL
module="mem" entry with the SAME DMA_MM2S_0_* event names for its own
(unrelated) MM2S-0. Under the tile-blind key, the sink's STALLED_LOCK /
MEMORY_STARVATION / STREAM_BACKPRESSURE intervals get silently merged into
the source's, corrupting the depth reading for exactly the two variants that
solve the crux. _load_source_tile_intervals below is a tile-aware LOCAL
loader (kept local rather than folded into the shared bankdisc_measure.
load_intervals, which other tools depend on) that recovers (row, col) from
the module string and filters every measurement down to SOURCE_TILE -- the
MM2S under test in every variant, single- or two-tile.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# aie.tile(0, 2) (col=0, row=2) in mm2s_egress_depth.py: the MM2S under test
# in every variant, single-tile or two-tile. The two-tile designs' sink
# lives at aie.tile(0, 3) and must never be measured here.
SOURCE_TILE_ROW, SOURCE_TILE_COL = 2, 0

# Real capture module strings look like "mem(2,0)": pt_name then (row, col)
# -- see trace_decoder.decode.rebuild_perfetto_mode0, which builds
# process_name as f"{pt_name}({row},{col})" (docstring: "Returns a dict
# {(pkt_type, row, col): [...]}" in decode_words).
_TILE_RE = re.compile(r"^(\w+)\((\d+),(\d+)\)$")


def _load_source_tile_intervals(perfetto_path: Path, config_path: Path):
    """Tile-aware sibling of bankdisc_measure.load_intervals.

    -> {event_name: [(start, end), ...]}, restricted to SOURCE_TILE. Events
    from any other tile (the two-tile designs' sink) never reach the caller,
    instead of being merged in under a module-type-only key.
    """
    slots = {}
    for t in json.load(config_path.open())["tiles_traced"]:
        slots[t["module"]] = t["events"]

    ev = json.load(perfetto_path.open())
    pid_tile = {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            m = _TILE_RE.match(e["args"]["name"])
            if m:
                mod, row, col = m.group(1), int(m.group(2)), int(m.group(3))
                pid_tile[e["pid"]] = (mod, row, col)

    open_b = {}
    out = defaultdict(list)
    for e in ev:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        tile = pid_tile.get(e["pid"])
        if tile is None:
            continue
        mod, row, col = tile
        if (row, col) != (SOURCE_TILE_ROW, SOURCE_TILE_COL):
            continue
        names = slots.get(mod, [])
        tid = e["tid"]
        if tid >= len(names):
            continue
        key = names[tid]
        if ph == "B":
            open_b[key] = e["ts"]
        elif key in open_b:
            out[key].append((open_b.pop(key), e["ts"]))
    return out


def _pair_windows(stalls, starvations, backpressures):
    """stalls/starvations/backpressures: [(start, end), ...] intervals.

    -> one dict per stall window (ordered by stall_start): stall_start,
    stall_end, starvation_onset (or None), onset_delay (or None),
    backpressure_overlap, valid.
    """
    stalls = sorted(stalls)
    starv_starts = sorted(s for s, _ in starvations)
    windows = []
    for i, (s_start, s_end) in enumerate(stalls):
        domain_end = stalls[i + 1][0] if i + 1 < len(stalls) else float("inf")
        onset = next((t for t in starv_starts if s_start <= t < domain_end), None)
        delay = (onset - s_start) if onset is not None else None
        probe_end = onset if onset is not None else s_end
        overlap = any(a < probe_end and b > s_start for a, b in backpressures)
        windows.append({
            "stall_start": s_start,
            "stall_end": s_end,
            "starvation_onset": onset,
            "onset_delay": delay,
            "backpressure_overlap": overlap,
            "valid": delay is not None and not overlap,
        })
    return windows


def measure(build_dir: Path, rep: int) -> dict:
    iv = _load_source_tile_intervals(build_dir / f"perfetto_r{rep}.json",
                                      build_dir / "trace_config.json")
    windows = _pair_windows(
        iv["DMA_MM2S_0_STALLED_LOCK"],
        iv["DMA_MM2S_0_MEMORY_STARVATION"],
        iv["DMA_MM2S_0_STREAM_BACKPRESSURE"],
    )
    valid_delays = [w["onset_delay"] for w in windows if w["valid"]]
    return {
        "windows": windows,
        "n_windows": len(windows),
        "n_valid": len(valid_delays),
        "depth_estimate": max(valid_delays) if valid_delays else None,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="build/experiments/mm2s-egress-depth")
    ap.add_argument("--reps", type=int, default=3, help="HW run repeats to pool")
    args = ap.parse_args()
    root = Path(args.root)

    from mm2s_egress_depth import VARIANTS  # noqa: E402

    print(f"{'variant':20} {'run':>3} {'windows':>7} {'valid':>5} {'depth':>6}")
    print("-" * 50)
    sweep_depths = []
    for v in VARIANTS:
        d = root / f"build_mm2s_egress_depth_{v}"
        if not d.is_dir():
            continue
        pooled_depths = []
        for r in range(1, args.reps + 1):
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            m = measure(d, r)
            if m["depth_estimate"] is not None:
                pooled_depths.append(m["depth_estimate"])
            print(f"{v:20} {r:>3} {m['n_windows']:>7} {m['n_valid']:>5} "
                  f"{str(m['depth_estimate']):>6}")
        if pooled_depths:
            best = max(pooled_depths)
            sweep_depths.append(best)
            print(f"{v:20} {'ALL':>3} {'':>7} {'':>5} {best:>6}")
        print()

    if sweep_depths:
        print(f"DEPTH (ceiling across all variants/reps) = {max(sweep_depths)}")
