#!/usr/bin/env python3
"""
compare-dispatch-overhead.py -- Side-by-side HW vs EMU comparison of the
per-(K, direction, gap_index) inter-task gap distributions produced by
aggregate-dispatch-overhead.py.

For each (K, direction) we tabulate, per gap_index:

    HW_median    EMU_median    delta (EMU - HW)    delta%

The point is to expose where the Q-aware EMU model matches or diverges
from HW so 2c.3 can decide whether to refine constants, add direction-
aware tiebreaking, or address deeper modeling gaps.

Usage:
  ./tools/compare-dispatch-overhead.py <HW-session-dir> <EMU-session-dir>

Both sessions must have been aggregated already
(aggregated.json exists).  The comparator emits:
  - A per-(K, direction) table to stdout
  - A JSON dump <EMU-session>/hw_vs_emu.json (HW path is also recorded)
"""

import argparse
import json
import sys
from pathlib import Path


def load_aggregated(p: Path) -> dict:
    f = p / "aggregated.json"
    if not f.exists():
        sys.exit(f"error: {f} missing; run aggregate-dispatch-overhead.py first")
    return json.loads(f.read_text())


def fmt_int(v) -> str:
    if v is None:
        return "    --"
    if isinstance(v, float):
        return f"{int(round(v)):>6d}"
    return f"{v:>6d}"


def fmt_pct(num, den) -> str:
    if num is None or den in (None, 0):
        return "      --"
    return f"{(num / den) * 100:+6.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("hw_session", type=Path, help="HW campaign session dir")
    ap.add_argument("emu_session", type=Path, help="EMU campaign session dir")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output path (default: <emu>/hw_vs_emu.json)")
    args = ap.parse_args()

    hw = load_aggregated(args.hw_session)
    emu = load_aggregated(args.emu_session)

    hw_summary = hw.get("summary", {})
    emu_summary = emu.get("summary", {})

    # Union of (K, direction) keys
    all_keys = sorted(set(hw_summary) | set(emu_summary))

    print(f"== HW vs EMU per-cell comparison ==")
    print(f"  HW  session: {hw.get('session', '?')}  "
          f"(N={hw.get('manifest', {}).get('n_runs', '?')})")
    print(f"  EMU session: {emu.get('session', '?')}  "
          f"(N={emu.get('manifest', {}).get('n_runs', '?')}; "
          f"EMU is deterministic)")
    print()

    out = {
        "hw_session": str(args.hw_session),
        "emu_session": str(args.emu_session),
        "cells": {},
    }

    for key in all_keys:
        if "." not in key:
            continue
        hw_s = hw_summary.get(key, {})
        emu_s = emu_summary.get(key, {})
        hw_gbi = hw_s.get("gaps_by_index", {})
        emu_gbi = emu_s.get("gaps_by_index", {})

        gaps = sorted({*hw_gbi.keys(), *emu_gbi.keys()}, key=int)

        print(f"  {key}")
        print(f"    {'cell':<10} {'HW med':>8} {'EMU med':>8} "
              f"{'delta':>8} {'pct':>8}    "
              f"{'HW p25':>7} {'HW p75':>7}  {'HW MAD':>6}")
        cell_out = {}
        for gi in gaps:
            hw_d = hw_gbi.get(gi, {})
            emu_d = emu_gbi.get(gi, {})
            hw_med = hw_d.get("median")
            emu_med = emu_d.get("median")
            delta = (
                emu_med - hw_med
                if (hw_med is not None and emu_med is not None) else None
            )
            print(f"    gap[{gi}]      "
                  f"{fmt_int(hw_med)}  {fmt_int(emu_med)}  "
                  f"{fmt_int(delta)}  {fmt_pct(delta, hw_med)}    "
                  f"{fmt_int(hw_d.get('p25'))} {fmt_int(hw_d.get('p75'))}  "
                  f"{fmt_int(hw_d.get('mad'))}")
            cell_out[gi] = {
                "hw_median": hw_med, "emu_median": emu_med,
                "delta": delta,
                "hw_p25": hw_d.get("p25"), "hw_p75": hw_d.get("p75"),
                "hw_mad": hw_d.get("mad"), "emu_mad": emu_d.get("mad"),
                "hw_n": hw_d.get("n"), "emu_n": emu_d.get("n"),
            }
        # Summary cells
        for kind in ("all_gaps", "first_gaps", "steady_gaps"):
            hw_d = hw_s.get(kind, {})
            emu_d = emu_s.get(kind, {})
            hw_med = hw_d.get("median")
            emu_med = emu_d.get("median")
            delta = (
                emu_med - hw_med
                if (hw_med is not None and emu_med is not None) else None
            )
            print(f"    {kind:<10} "
                  f"{fmt_int(hw_med)}  {fmt_int(emu_med)}  "
                  f"{fmt_int(delta)}  {fmt_pct(delta, hw_med)}    "
                  f"{fmt_int(hw_d.get('p25'))} {fmt_int(hw_d.get('p75'))}  "
                  f"{fmt_int(hw_d.get('mad'))}")
            cell_out[kind] = {
                "hw_median": hw_med, "emu_median": emu_med,
                "delta": delta,
            }
        print()
        out["cells"][key] = cell_out

    out_path = args.out or (args.emu_session / "hw_vs_emu.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
