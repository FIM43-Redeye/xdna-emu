#!/usr/bin/env python3
"""Analyze the producer BD-boundary probe: does the boundary MM2S fetch collide
with the core (item 1), and does FINISHED_BD fire on the last data cycle or the
cycle after (item 2)?

Both trace units live on tile (0,2) and share the tile timer, so core-unit and
mem-unit ts are directly comparable (proven by the arc's 220/220 result). We use
`ts` (never `soc`) per the timebase findings; for the discrete FINISHED_BD and
the PORT_RUNNING falling edge, the transition ts is what we compare.

Input: one events.json (schema_version 1) from parse-trace.py.
"""
import argparse
import json
from collections import defaultdict

# The producer in MLIR coords is (0,2); the HW capture shifts col0->col1, so the
# events land at (1,2). Accept whichever the decode emits.
PROD = {(0, 2), (1, 2)}


def load(path):
    d = json.load(open(path))
    evs = [e for e in d["events"] if (e["col"], e["row"]) in PROD]
    return evs, d.get("placement")


def level_intervals(evs, name):
    """Rebuild ON intervals for a level event from its record stream.

    schema-1 records are edge/level toggles; consecutive same-name records
    bracket an asserted window. We pair them as (rising, falling)."""
    ts = sorted(e["ts"] for e in evs if e["name"] == name)
    # A robust rebuild needs the B/E stream; the record stream alone can only
    # give rising edges reliably. We report rising-edge ts list and, where the
    # decoder emitted paired toggles, naive consecutive pairing.
    return ts


def discrete_ts(evs, name):
    return sorted(e["ts"] for e in evs if e["name"] == name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events_json")
    ap.add_argument("--window", type=int, default=6,
                    help="cycles around each FINISHED_BD to print")
    args = ap.parse_args()

    evs, placement = load(args.events_json)
    print(f"placement={placement}  producer events={len(evs)}")

    by_name = defaultdict(list)
    for e in evs:
        by_name[e["name"]].append(e["ts"])
    print("event name counts (producer tile):")
    for n in sorted(by_name):
        print(f"  {n:32} {len(by_name[n])}")

    finished = discrete_ts(evs, "DMA_MM2S_0_FINISHED_BD")
    port = sorted(by_name.get("PORT_RUNNING_1", []))
    conflict = sorted(by_name.get("CONFLICT_DM_BANK_0", []) +
                      by_name.get("CONFLICT_DM_BANK_1", []))
    mstall = sorted(by_name.get("MEMORY_STALL", []))
    lstall = sorted(by_name.get("LOCK_STALL", []))
    lreq = sorted(by_name.get("INSTR_LOCK_ACQUIRE_REQ", []))

    print(f"\nFINISHED_BD count = {len(finished)} at ts {finished}")

    # For each BD boundary, print every producer event within +/- window.
    all_ev = sorted(((e["ts"], e["name"]) for e in evs), key=lambda x: x[0])
    for i, f in enumerate(finished):
        lo, hi = f - args.window, f + args.window
        near = [(t, n) for t, n in all_ev if lo <= t <= hi]
        print(f"\n--- boundary {i}: FINISHED_BD @ {f} (window +/-{args.window}) ---")
        for t, n in near:
            mark = "  <== FINISHED_BD" if (t == f and n == "DMA_MM2S_0_FINISHED_BD") else ""
            print(f"    dt={t - f:+3d}  ts={t}  {n}{mark}")

    # Item 2 discriminator: nearest PORT_RUNNING_1 edge to each FINISHED_BD.
    print("\n=== item 2: FINISHED_BD vs nearest PORT_RUNNING_1 edge ===")
    for f in finished:
        if not port:
            print("  no PORT_RUNNING_1 events"); break
        nearest = min(port, key=lambda p: abs(p - f))
        print(f"  FINISHED_BD @ {f}: nearest PORT_RUNNING_1 edge @ {nearest} (dt={nearest - f:+d})")


if __name__ == "__main__":
    main()
