#!/usr/bin/env python3
"""Date a compute tile's S2MM 0 stream accepts against its core's lock stall, on one trace timer.

Built for the objectfifo turn designs (objectfifo_channel_turns, objectfifo_turns_walkers), where the
core's loop and the tile DMA's writes meet on memory banks and the bank conflict count depends on a
one-cycle phase between them. The core-module trace unit of the compute tile records, on ONE timer,

    LOCK_STALL        the core waiting on a lock
    PORT_RUNNING_0    the compute tile's stream switch master port to DMA S2MM 0 moving a word

so a run of PORT_RUNNING_0 can be dated against the nearest LOCK_STALL edges without crossing tiles. The
memory-module unit of the same tile (its own timer) adds the S2MM 0 task / BD events, the S2MM 0 lock
wait and the four bank conflict events. The bridge suite injects both through its ordinary trace
preparation; this tool only names the event lists and reads the result:

    eval "$(python3 tools/accept_probe_table.py env)"          # XDNA_TRACE_CORE_EVENTS / _MEMMOD_EVENTS
    ./scripts/emu-bridge-test.sh --no-hw '^objectfifo_channel_turns$'
    python3 tools/accept_probe_table.py table RUN_DIR [RUN_DIR ...] [--col N] [--row N]

A RUN_DIR holds trace_raw.bin and events.json (the suite's `<test>.<compiler>.emu` / `.hw` directories).
Per run the table prints the bank conflict counts and, for each PORT_RUNNING_0 run, its start and end
against the nearest LOCK_STALL rise and fall (signed cycles, later minus edge). A run is marked INVALID
when an event the question depends on is absent from its own trace (an absent unit is not a zero count),
and the dating lines are then not printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

PT_CORE, PT_MEM = 0, 1                      # trace packet types of the core and memory modules

# The stream switch port behind PORT_RUNNING_0 on a compute tile is the DMA bundle's channel 0 master
# port (data leaving the switch into S2MM 0): mlir-trace-inject's default binding for slot 0.
CORE_EVENTS = ("LOCK_STALL", "PORT_RUNNING_0", "MEMORY_STALL",
               "INSTR_LOCK_ACQUIRE_REQ", "INSTR_LOCK_RELEASE_REQ")
MEMMOD_EVENTS = ("DMA_S2MM_0_START_TASK", "DMA_S2MM_0_FINISHED_BD", "DMA_S2MM_0_FINISHED_TASK",
                 "DMA_S2MM_0_STALLED_LOCK",
                 "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1", "CONFLICT_DM_BANK_2", "CONFLICT_DM_BANK_3")
BANKS = (0, 1, 2, 3)

# Events whose absence makes the run unusable (the core unit's two, the memory unit's DMA pair).
REQUIRED_CORE = ("LOCK_STALL", "PORT_RUNNING_0")
REQUIRED_MEM = ("DMA_S2MM_0_START_TASK", "DMA_S2MM_0_FINISHED_BD")


def env_lines() -> str:
    """Shell exports the bridge suite reads to choose the traced events (see emu-bridge-test.sh)."""
    return (f"export XDNA_TRACE_CORE_EVENTS={','.join(CORE_EVENTS)}\n"
            f"export XDNA_TRACE_MEMMOD_EVENTS={','.join(MEMMOD_EVENTS)}\n")


def level_runs(stamps) -> list[tuple[int, int]]:
    """(first, last) of each run of consecutive cycles."""
    out: list[tuple[int, int]] = []
    for ts in sorted(set(stamps)):
        if out and ts == out[-1][1] + 1:
            out[-1] = (out[-1][0], ts)
        else:
            out.append((ts, ts))
    return out


def nearest(edges: list[int], t: int):
    """The edge closest to t (the earlier one on a tie), or None."""
    return min(edges, key=lambda e: (abs(t - e), e)) if edges else None


def analyse(events, col: int | None = None, row: int = 2) -> dict:
    """Per-run facts of one compute tile from level-expanded events (objects with pkt_type, row, col, name, ts)."""
    core: dict = {}
    mem: dict = {}
    for e in events:
        if e.row != row or (col is not None and e.col != col):
            continue
        if e.pkt_type == PT_CORE:
            core.setdefault(e.name, []).append(e.ts)
        elif e.pkt_type == PT_MEM:
            mem.setdefault(e.name, []).append(e.ts)
    missing = [f"core:{n}" for n in REQUIRED_CORE if not core.get(n)]
    missing += [f"mem:{n}" for n in REQUIRED_MEM if not mem.get(n)]
    stall = level_runs(core.get("LOCK_STALL", []))
    ports = level_runs(core.get("PORT_RUNNING_0", []))
    rises = [first for first, _ in stall]
    falls = [last + 1 for _, last in stall]          # first cycle after the stall
    rows = []
    for first, last in ports:
        end = last + 1                               # first cycle after the port run
        rows.append({
            "start": first, "end": end, "len": end - first,
            "start_vs_rise": None if (r := nearest(rises, first)) is None else first - r,
            "start_vs_fall": None if (f := nearest(falls, first)) is None else first - f,
            "end_vs_rise": None if (r := nearest(rises, end)) is None else end - r,
            "end_vs_fall": None if (f := nearest(falls, end)) is None else end - f,
        })
    conflicts = {b: sorted(set(mem.get(f"CONFLICT_DM_BANK_{b}", []))) for b in BANKS}
    return {
        "missing": missing, "stall_runs": len(stall), "port_runs": rows,
        "bd_finishes": len(set(mem.get("DMA_S2MM_0_FINISHED_BD", []))),
        "conflicts": conflicts,
    }


def _signed(v) -> str:
    return "?" if v is None else f"{v:+d}"


def format_run(info: dict, label: str) -> str:
    lines = [f"== {label} =="]
    if info["missing"]:
        lines.append("INVALID: missing " + ", ".join(info["missing"])
                     + " (an absent event is not a zero count; no verdict from this run)")
        return "\n".join(lines)
    lines.append(f"LOCK_STALL runs {info['stall_runs']}   PORT_RUNNING_0 runs {len(info['port_runs'])}"
                 f"   S2MM 0 BDs finished {info['bd_finishes']}")
    lines.append("bank conflicts (cycles): " + "  ".join(f"B{b} {len(info['conflicts'][b])}" for b in BANKS))
    lines.append("  run  start     len   start-vs-stall-rise fall   end-vs-stall-rise fall")
    for i, r in enumerate(info["port_runs"]):
        lines.append(f"{i:>5} {r['start']:>8} {r['len']:>5}   "
                     f"{_signed(r['start_vs_rise']):>8} {_signed(r['start_vs_fall']):>8}   "
                     f"{_signed(r['end_vs_rise']):>8} {_signed(r['end_vs_fall']):>8}")
    for key, name in (("start_vs_fall", "run start vs stall fall"), ("start_vs_rise", "run start vs stall rise"),
                      ("end_vs_fall", "run end vs stall fall"), ("end_vs_rise", "run end vs stall rise")):
        hist = Counter(r[key] for r in info["port_runs"] if r[key] is not None)
        lines.append(f"  {name:<26}" + ("  ".join(f"{v:+d} x{n}" for v, n in sorted(hist.items())) or "-"))
    return "\n".join(lines)


def load_events(run_dir: Path):
    from gen_dma_lock_probe import load_level_events
    return load_level_events(run_dir)


def cmd_table(args) -> int:
    for run in args.runs:
        if not (Path(run) / "trace_raw.bin").exists() or not (Path(run) / "events.json").exists():
            print(f"{run}: no trace_raw.bin / events.json", file=sys.stderr)
            return 2
    for run in args.runs:
        info = analyse(load_events(Path(run)), args.col, args.row)
        print(format_run(info, Path(run).name))
        print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="print the exports that select the traced events")
    t = sub.add_parser("table", help="print the dating table of one or more run directories")
    t.add_argument("runs", nargs="+", help="run directories (trace_raw.bin + events.json)")
    t.add_argument("--col", type=int, default=None, help="trace column of the compute tile (default: any)")
    t.add_argument("--row", type=int, default=2)
    args = ap.parse_args(argv)
    if args.cmd == "env":
        sys.stdout.write(env_lines())
        return 0
    return cmd_table(args)


if __name__ == "__main__":
    sys.exit(main())
