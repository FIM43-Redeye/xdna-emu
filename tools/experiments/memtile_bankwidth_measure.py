#!/usr/bin/env python3
"""Measure per-transfer memtile DMA duration + bank-conflict AREA from a
memtile_bankwidth capture (Experiment A).

Reuses bankdisc_measure's `load_intervals` (the mode-0 B/E interval rebuild --
see its docstring for why level events need INTERVAL AREA, not record counts)
and its bracket recipe, moved to the "memtile" trace module and MEM_TILE_*
event names (`parse-trace.py`'s pkt_type->module map: 0=core, 1=mem, 2=shim,
3=memtile).

Bracket: transfer r = [falling edge of the STALLED_LOCK interval preceding
MM2S_SEL0_FINISHED_BD[r], FINISHED_BD[r]). memtile_bankwidth.py's channels are
lockless BD chains (no core to gate a lock handshake against; see that
module's docstring), so in practice there is usually no STALLED_LOCK interval
between back-to-back transfers -- the bracket falls back to the previous
FINISHED_BD's rising edge, which is the same [prev_finish, this_finish)
window bankdisc's formula would produce when the stall duration is 0. Both
S2MM (fill) and MM2S (drain) channels are bracketed the same way, since A1's
width inversion needs the fill channel's OWN measured cadence (`f_contender`,
"derived from the second DMA's cadence, not fitted" per the task brief).
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bankdisc_measure import load_intervals, area  # noqa: E402

NUM_BANKS = 16
TRANSFER_WORDS = 256   # OBJ in memtile_bankwidth.py
TRANSFER_BYTES = TRANSFER_WORDS * 4


def _bracket(finished, stalls):
    """[(start, end), ...] transfer windows: STALLED_LOCK falling edge (or the
    previous FINISHED_BD, if no stall precedes -- the lockless-chain case) to
    FINISHED_BD's rising edge."""
    windows = []
    prev_finish = None
    for f in sorted(finished):
        prior = [end for _, end in stalls if end <= f]
        start = max(prior) if prior else prev_finish
        if start is not None:
            windows.append((start, f))
        prev_finish = f
    return windows


def measure(build_dir: Path, rep: int, channel: str = "MM2S") -> dict:
    """channel: "MM2S" (drain) or "S2MM" (fill) -- which memtile DMA channel's
    FINISHED_BD/STALLED_LOCK pair to bracket."""
    iv = load_intervals(build_dir / f"perfetto_r{rep}.json",
                        build_dir / "trace_config.json")
    fin = iv[("memtile", f"MEM_TILE_DMA_{channel}_SEL0_FINISHED_BD")]
    stalls = iv[("memtile", f"MEM_TILE_DMA_{channel}_SEL0_STALLED_LOCK")]
    windows = _bracket([s for s, _ in fin], stalls)
    durations = [b - a for a, b in windows]

    conflict_area = sum(
        area(iv[("memtile", f"CONFLICT_DM_BANK_{n}")]) for n in range(NUM_BANKS)
    )
    return {
        "n_finished": len(fin),
        "n_bracketed": len(durations),
        "durations": durations,
        "median": statistics.median(durations) if durations else None,
        "mean": statistics.mean(durations) if durations else None,
        "conflict_area": conflict_area,
    }


A1_VARIANTS = ("a1_idle", "a1_apart", "a1_collide")
A2_VARIANTS = tuple(f"a2_stride_{s}" for s in (4, 16, 32, 64, 128, 256))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="build/experiments/memtile-bankwidth",
                     help="dir holding build_memtile_bankwidth_<variant>/")
    ap.add_argument("--reps", type=int, default=3, help="HW run repeats to pool")
    args = ap.parse_args()
    root = Path(args.root)

    print(f"{'variant':14} {'run':>3} {'n':>3} {'median':>7} {'mean':>7} "
          f"{'conflict':>8}")
    print("-" * 50)
    for v in A1_VARIANTS + A2_VARIANTS:
        d = root / f"build_memtile_bankwidth_{v}"
        if not d.is_dir():
            continue
        pooled = []
        for r in range(1, args.reps + 1):
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            m = measure(d, r)
            pooled += m["durations"]
            print(f"{v:14} {r:>3} {m['n_bracketed']:>3} {str(m['median']):>7} "
                  f"{m['mean'] if m['mean'] is None else round(m['mean'], 1):>7} "
                  f"{m['conflict_area']:>8}")
        if pooled:
            print(f"{v:14} {'ALL':>3} {len(pooled):>3} "
                  f"{statistics.median(pooled):>7}")
        print()
