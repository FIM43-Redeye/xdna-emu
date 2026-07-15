#!/usr/bin/env python3
"""Measure MM2S egress-FIFO depth (Experiment B) from an mm2s_egress_depth
capture.

The delay from DMA_MM2S_0_STALLED_LOCK onset to DMA_MM2S_0_MEMORY_STARVATION
onset is the egress occupancy in beats at stall-time -- the exact mirror of
the S2MM ingress finding (docs/superpowers/findings/2026-07-14-dma-memory-
pressure-event-semantics.md), which pinned ingress depth at ~15-16 beats via
the same onset-delay recipe: `BACKPRESSURE = [lock_stall_start + 15,
lock_stall_end + 1)`.

Reuses bankdisc_measure.load_intervals (the mode-0 B/E interval rebuild --
STALLED_LOCK / MEMORY_STARVATION / STREAM_BACKPRESSURE are all LEVEL events,
so their onsets come from real intervals, never raw record counts; see that
module's docstring).

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
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bankdisc_measure import load_intervals  # noqa: E402


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
    iv = load_intervals(build_dir / f"perfetto_r{rep}.json",
                        build_dir / "trace_config.json")
    windows = _pair_windows(
        iv[("mem", "DMA_MM2S_0_STALLED_LOCK")],
        iv[("mem", "DMA_MM2S_0_MEMORY_STARVATION")],
        iv[("mem", "DMA_MM2S_0_STREAM_BACKPRESSURE")],
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
