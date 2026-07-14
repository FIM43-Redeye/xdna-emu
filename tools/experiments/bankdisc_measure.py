#!/usr/bin/env python3
"""Measure MM2S transfer duration + bank-conflict area from a bankdisc capture.

Timebase rules (see docs/superpowers/findings/2026-07-13-memory-stall-bank-arbitration.md):
  * discrete events (FINISHED_BD): the `ts` of the rising edge is valid.
  * level events (CONFLICT_*, STALLED_LOCK, ...): decoded RECORD COUNTS are an
    encoding artifact -- compare INTERVAL AREA (cycles asserted).  We take the
    intervals straight from the mode-0 B/E rebuild, which integrates correctly
    across both the Event and Event+Repeat encodings.
  * never `soc` (decoder bug: it subtracts a physically-real +1).

Bracket: the MM2S has exactly one self-looping BD, so it is either stalled on
lk_full or transferring.  Transfer r starts at the FALLING edge of
STALLED_LOCK interval r and ends at FINISHED_BD[r].
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_intervals(perfetto_path: Path, config_path: Path):
    """-> {(module, event_name): [(start, end), ...]}, from the mode-0 B/E rebuild."""
    slots = {}
    for t in json.load(config_path.open())["tiles_traced"]:
        slots[t["module"]] = t["events"]

    ev = json.load(perfetto_path.open())
    pid_module = {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_module[e["pid"]] = e["args"]["name"].split("(")[0]

    open_b = {}
    out = defaultdict(list)
    for e in ev:
        ph = e.get("ph")
        if ph not in ("B", "E"):
            continue
        mod = pid_module[e["pid"]]
        names = slots.get(mod, [])
        tid = e["tid"]
        if tid >= len(names):
            continue
        key = (mod, names[tid])
        if ph == "B":
            open_b[key] = e["ts"]
        elif key in open_b:
            out[key].append((open_b.pop(key), e["ts"]))
    return out


def area(intervals):
    return sum(b - a for a, b in intervals)


def measure(build_dir: Path, rep: int):
    iv = load_intervals(build_dir / f"perfetto_r{rep}.json",
                        build_dir / "trace_config.json")

    finished = sorted(s for s, _ in iv[("mem", "DMA_MM2S_0_FINISHED_BD")])
    stalls = sorted(iv[("mem", "DMA_MM2S_0_STALLED_LOCK")])

    # Transfer r = [falling edge of the stall interval that precedes
    # FINISHED_BD[r], FINISHED_BD[r]).
    durations = []
    for f in finished:
        prior = [end for _, end in stalls if end <= f]
        if prior:
            durations.append(f - max(prior))

    res = {
        "n_finished_bd": len(finished),
        "n_bracketed": len(durations),
        "durations": durations,
        "median": statistics.median(durations) if durations else None,
        "mean": statistics.mean(durations) if durations else None,
        "min": min(durations) if durations else None,
        "max": max(durations) if durations else None,
    }
    for name in ("CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                 "CONFLICT_DM_BANK_4", "CONFLICT_DM_BANK_5",
                 "DMA_MM2S_0_STREAM_BACKPRESSURE",
                 "DMA_MM2S_0_MEMORY_STARVATION",
                 "DMA_MM2S_0_STALLED_LOCK"):
        res[name] = area(iv[("mem", name)])
    res["MEMORY_STALL"] = area(iv[("core", "MEMORY_STALL")])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="dir holding build_bankdisc_<variant>/")
    ap.add_argument("--reps", type=int, default=3, help="HW run repeats to pool")
    args = ap.parse_args()
    root = Path(args.root)

    print(f"{'variant':9} {'run':>3} {'n':>3} {'median':>7} {'mean':>7} "
          f"{'min':>5} {'max':>5} {'blk0':>5} {'blk1':>5} {'blk4':>5} {'blk5':>5} "
          f"{'strmBP':>6} {'memSTV':>6} {'coreSTL':>7}")
    print("-" * 96)
    medians = {}
    for v in ("idle", "apart", "collide", "collide2"):
        d = root / f"build_bankdisc_{v}"
        if not d.is_dir():
            continue
        pooled = []
        for r in range(1, args.reps + 1):
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            m = measure(d, r)
            pooled += m["durations"]
            print(f"{v:9} {r:>3} {m['n_bracketed']:>3} {m['median']:>7} "
                  f"{m['mean']:>7.1f} {m['min']:>5} {m['max']:>5} "
                  f"{m['CONFLICT_DM_BANK_0']:>5} {m['CONFLICT_DM_BANK_1']:>5} "
                  f"{m['CONFLICT_DM_BANK_4']:>5} {m['CONFLICT_DM_BANK_5']:>5} "
                  f"{m['DMA_MM2S_0_STREAM_BACKPRESSURE']:>6} "
                  f"{m['DMA_MM2S_0_MEMORY_STARVATION']:>6} {m['MEMORY_STALL']:>7}")
        if pooled:
            medians[v] = statistics.median(pooled)
            print(f"{v:9} {'ALL':>3} {len(pooled):>3} {medians[v]:>7}")
        print()

    if {"idle", "apart", "collide"} <= medians.keys():
        ti, ta, tc = medians["idle"], medians["apart"], medians["collide"]
        gate = abs(ta - ti) / ti
        print(f"VALIDITY GATE  T_APART/T_IDLE = {ta/ti:.3f}  (deviation {gate*100:.1f}%,"
              f" pass if <20%): {'PASS' if gate < 0.20 else 'FAIL'}")
        print(f"DISCRIMINATOR  T_COLLIDE/T_APART = {tc/ta:.3f}")
        print("   Model A (128-bit burst, 1 slot in 4) predicts 1.0 - 1.3")
        print("   Model B (32-bit per-beat, every cycle) predicts 2 - 4")
        if "collide2" in medians:
            t2 = medians["collide2"]
            print(f"REPLICATE      T_COLLIDE2/T_APART = {t2/ta:.3f} (same test, logical bank 2)")
