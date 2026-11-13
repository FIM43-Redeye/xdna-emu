#!/usr/bin/env python3
"""Per loop iteration lock and DMA timing of a compute tile whose core releases an input lock and an output lock.

Built for the objectfifo turn designs (objectfifo_turns_walkers, objectfifo_channel_turns): the core releases the
`a` input lock, later the `out` lock, once per iteration, and the tile's S2MM 0 and MM2S 0 each park on the lock
they are fed by. One traced run of the compute tile's memory module gives, on one timer:

    LOCK_<n>_REL of the `a` lock and of the `out` lock   (the selectors programmed so LOCK_n_REL follows them:
                                                         trace-prepare.py --memmod-sel-locks)
    DMA_S2MM_0_STALLED_LOCK, DMA_MM2S_0_STALLED_LOCK     (level: the grant is the cycle after it falls)
    DMA_S2MM_0_FINISHED_BD, DMA_MM2S_0_FINISHED_BD
    CONFLICT_DM_BANK_<n>

and the table prints, per iteration (an iteration runs from one `a` release to the next): the two releases, each
channel's grant minus the release it waited for (S2MM 0 on the latest `a` release, MM2S 0 on the latest `out`
release), each FINISHED_BD minus the latest earlier grant of its channel, and the conflict stamps per bank minus
the iteration's `out` release. A grant or finish with no earlier release or grant to pair with is printed `?`.
Channels that also wait on locks the trace does not name (a second input lock sharing S2MM 0) show those grants as
a larger delta from the latest `a` release; the summary groups the deltas so the usual values stand out.

    python3 tools/turns_lock_table.py RUN_DIR [RUN_DIR ...] [--col N] [--row N] [--a-event LOCK_1_REL]
                                      [--out-event LOCK_2_REL]

A RUN_DIR holds trace_raw.bin and events.json (the bridge suite's `<test>.<compiler>.emu` / `.hw` directories).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

PT_MEM = 1                                  # trace packet type of the memory module


def _runs(stamps) -> list[tuple[int, int]]:
    """(first, last) of each run of consecutive cycles."""
    out: list[tuple[int, int]] = []
    for ts in sorted(set(stamps)):
        if out and ts == out[-1][1] + 1:
            out[-1] = (out[-1][0], ts)
        else:
            out.append((ts, ts))
    return out


def _latest_before(stamps: list[int], t: int):
    """The latest stamp <= t (stamps ascending), or None."""
    best = None
    for s in stamps:
        if s > t:
            break
        best = s
    return best


def turns_rows(events, col: int | None = None, row: int = 2, a_event: str = "LOCK_1_REL",
               out_event: str = "LOCK_2_REL") -> list[dict]:
    """One dict per iteration of the compute tile's memory-module events (see the module note)."""
    mem: dict = {}
    for e in events:
        if e.pkt_type == PT_MEM and e.row == row and (col is None or e.col == col):
            mem.setdefault(e.name, []).append(e.ts)
    a_rels = sorted(set(mem.get(a_event, [])))
    out_rels = sorted(set(mem.get(out_event, [])))
    grants = {ch: sorted(last + 1 for _, last in _runs(mem.get(f"DMA_{ch}_0_STALLED_LOCK", [])))
              for ch in ("S2MM", "MM2S")}
    fins = {ch: sorted(set(mem.get(f"DMA_{ch}_0_FINISHED_BD", []))) for ch in ("S2MM", "MM2S")}
    waits_on = {"S2MM": a_rels, "MM2S": out_rels}
    rows = []
    for i, a in enumerate(a_rels):
        end = a_rels[i + 1] if i + 1 < len(a_rels) else float("inf")
        out = next((o for o in out_rels if a <= o < end), None)
        row_: dict = {"a_rel": a, "out_rel": out}
        for ch in ("S2MM", "MM2S"):
            gs = [g for g in grants[ch] if a <= g < end]
            row_[f"{ch.lower()}_grants"] = [g - r for g in gs
                                            if (r := _latest_before(waits_on[ch], g)) is not None]
            fs = [f for f in fins[ch] if a <= f < end]
            row_[f"{ch.lower()}_fins"] = [None if (g := _latest_before(grants[ch], f)) is None else f - g
                                          for f in fs]
        anchor = out if out is not None else a
        row_["conflicts"] = {b: [x - anchor for x in sorted(set(mem.get(f"CONFLICT_DM_BANK_{b}", [])))
                                 if a <= x < end] for b in (2, 3)}
        rows.append(row_)
    return rows


def _list(values, sign: bool = True) -> str:
    return ",".join("?" if v is None else (f"{v:+d}" if sign else str(v)) for v in values) or "-"


def _histogram(values) -> str:
    counts = Counter(v for v in values if v is not None)
    return "  ".join(f"{v:+d} x{n}" for v, n in sorted(counts.items())) or "-"


def format_turns_table(rows, label: str, a_name: str = "a", out_name: str = "out") -> str:
    """One line per iteration, then a summary of how often each delta occurred."""
    lines = [f"== {label} ==  (cycles; grant = cycle after STALLED_LOCK falls; conflicts after REL({out_name}))",
             "  it release              S2MM 0                         MM2S 0                         conflicts"]
    for i, r in enumerate(rows):
        off = "?" if r["out_rel"] is None else f"+{r['out_rel'] - r['a_rel']}"
        lines.append(
            f"{i:>4} {a_name} {r['a_rel']} {out_name} {off:<5} "
            f"S2MM grant {_list(r['s2mm_grants'])} fin {_list(r['s2mm_fins'])}   "
            f"MM2S grant {_list(r['mm2s_grants'])} fin {_list(r['mm2s_fins'])}   "
            f"B2 [{_list(r['conflicts'][2])}] B3 [{_list(r['conflicts'][3])}]")
    lines.append(f"summary over {len(rows)} iterations")
    lines.append(f"  S2MM grant after REL({a_name}):   " + _histogram(v for r in rows for v in r["s2mm_grants"]))
    lines.append(f"  S2MM FINISHED_BD after grant:  " + _histogram(v for r in rows for v in r["s2mm_fins"]))
    lines.append(f"  MM2S grant after REL({out_name}): " + _histogram(v for r in rows for v in r["mm2s_grants"]))
    lines.append(f"  MM2S FINISHED_BD after grant:  " + _histogram(v for r in rows for v in r["mm2s_fins"]))
    n2 = sum(len(r["conflicts"][2]) for r in rows)
    n3 = sum(len(r["conflicts"][3]) for r in rows)
    lines.append(f"  conflict stamps: B2 x{n2}, B3 x{n3}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (trace_raw.bin + events.json)")
    ap.add_argument("--col", type=int, default=None, help="trace column of the compute tile (default: any)")
    ap.add_argument("--row", type=int, default=2)
    ap.add_argument("--a-event", default="LOCK_1_REL", help="the memory-module event of the `a` lock's release")
    ap.add_argument("--out-event", default="LOCK_2_REL", help="the memory-module event of the `out` lock's release")
    args = ap.parse_args(argv)
    for run in args.runs:
        if not (Path(run) / "trace_raw.bin").exists() or not (Path(run) / "events.json").exists():
            print(f"{run}: no trace_raw.bin / events.json", file=sys.stderr)
            return 2
    from gen_dma_lock_probe import load_level_events
    for run in args.runs:
        rows = turns_rows(load_level_events(Path(run)), args.col, args.row, args.a_event, args.out_event)
        print(format_turns_table(rows, Path(run).name))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
