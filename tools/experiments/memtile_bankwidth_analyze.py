#!/usr/bin/env python3
"""Derive the AIE2 MEMTILE DMA bank-access width and strided-channel
parallelism from a memtile_bankwidth capture (Experiment A).

Two independent outputs, matching the A1/A2 variant families in
memtile_bankwidth.py:

  A1 WIDTH: invert the measured bank-CONFLICT area under the same single-port
  round-robin contention model bankdisc_analyze.py uses for the compute tile
  (AM020 ch.2:166) -- generalized here to a memtile fill-vs-drain DMA pair
  instead of core-vs-DMA. `f_contender` (the fill channel's per-cycle word
  rate) is MEASURED from the fill channel's own bracketed transfer duration,
  never fitted, per the task brief.

  A2 RATIO: span(a2_stride_S)/span(a2_stride_4) -- the "how many distinct
  physical banks does the memtile serve in parallel" discriminator (a
  256-byte stride always re-hits the SAME bank -- see memtile_bankwidth.py's
  docstring -- so its span should be the slowest; smaller strides spread
  across more banks and approach the contiguous baseline if the memtile can
  serve multiple banks per cycle).
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from memtile_bankwidth_measure import (  # noqa: E402
    TRANSFER_BYTES, TRANSFER_WORDS, measure,
)

A2_STRIDES_BYTES = (4, 16, 32, 64, 128, 256)


def invert_width_bytes(conflict_area: float, f_contender: float,
                        transfer_bytes: float = TRANSFER_BYTES):
    """Bytes moved per memtile bank access, inverted from measured conflict
    area (bankdisc_analyze's derivation, generalized).

    Round-robin, single-port-per-bank model: in any cycle where BOTH the fill
    and drain channels want the SAME physical bank, one CONFLICT_DM_BANK_n
    cycle is recorded. If the two channels' request streams are uncorrelated
    in phase,

        conflict_cycles = P(drain wants bank b) * f_contender * T

    (a single bank is "in play" for a1_collide -- both buffers are pinned to
    it -- so, unlike bankdisc's two-physical-bank compute-tile derivation,
    there is no factor of N banks to divide across). Solving for the drain's
    own per-cycle access rate:

        accesses_per_transfer = conflict_cycles / f_contender
        bytes_per_access      = transfer_bytes / accesses_per_transfer

    Returns None if there is nothing to invert (no conflict observed, or a
    contender with zero measured throughput).
    """
    if not f_contender or f_contender <= 0:
        return None
    accesses_per_transfer = conflict_area / f_contender
    if accesses_per_transfer <= 0:
        return None
    return transfer_bytes / accesses_per_transfer


def a1_width_bytes(build_root: Path, reps: int = 3):
    """Pool a1_collide's conflict area and a1_collide's OWN fill-channel
    cadence (the "second DMA", per the brief) across `reps` HW captures, then
    invert for the bytes-per-bank-access width."""
    conflict_pooled, fill_dur_pooled = [], []
    d = build_root / "build_memtile_bankwidth_a1_collide"
    for r in range(1, reps + 1):
        if not (d / f"perfetto_r{r}.json").exists():
            continue
        drain = measure(d, r, channel="MM2S")
        fill = measure(d, r, channel="S2MM")
        conflict_pooled.append(drain["conflict_area"])
        fill_dur_pooled += fill["durations"]
    if not conflict_pooled or not fill_dur_pooled:
        return None
    conflict_area = statistics.median(conflict_pooled)
    t_fill = statistics.median(fill_dur_pooled)
    f_contender = TRANSFER_WORDS / t_fill if t_fill else None
    return invert_width_bytes(conflict_area, f_contender)


def a2_strided_ratio(build_root: Path, reps: int = 3) -> dict:
    """{stride_bytes: span(stride)/span(4B contiguous)}, pooling medians
    across `reps` HW captures per stride."""
    medians = {}
    for s in A2_STRIDES_BYTES:
        d = build_root / f"build_memtile_bankwidth_a2_stride_{s}"
        pooled = []
        for r in range(1, reps + 1):
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            pooled += measure(d, r, channel="MM2S")["durations"]
        if pooled:
            medians[s] = statistics.median(pooled)
    baseline = medians.get(4)
    if not baseline:
        return {}
    return {s: v / baseline for s, v in medians.items() if s != 4}


def analyze(stats: dict) -> dict:
    """Pure, HW-free analysis core (the fixture-testable entry point).

    ``stats`` is ``{variant: {"conflict_area": float, "fill_median": float,
    "median": float}}`` -- the already-measured per-variant numbers (as
    `memtile_bankwidth_measure.measure()` produces, keyed by variant name).
    Returns ``{"width_bytes": float | None, "strided_ratio": {stride_bytes:
    ratio}}``.
    """
    out = {"width_bytes": None, "strided_ratio": {}}

    collide = stats.get("a1_collide")
    if collide and collide.get("fill_median"):
        f_contender = TRANSFER_WORDS / collide["fill_median"]
        out["width_bytes"] = invert_width_bytes(
            collide["conflict_area"], f_contender)

    baseline = stats.get("a2_stride_4", {}).get("median")
    if baseline:
        for s in A2_STRIDES_BYTES:
            if s == 4:
                continue
            v = stats.get(f"a2_stride_{s}", {}).get("median")
            if v is not None:
                out["strided_ratio"][s] = v / baseline

    return out


def main(root="."):
    root = Path(root)
    width = a1_width_bytes(root)
    ratios = a2_strided_ratio(root)

    print("=== A1: memtile bank-access width ===")
    if width is not None:
        print(f"  {width:.2f} B/access ({width / 4:.2f} words/access)")
    else:
        print("  (no capture data under", root, ")")

    print("\n=== A2: strided-channel-parallelism ratio (span/contiguous) ===")
    for s in A2_STRIDES_BYTES[1:]:
        r = ratios.get(s)
        print(f"  stride={s:4} B: {r:.3f}" if r is not None
              else f"  stride={s:4} B: (no data)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
