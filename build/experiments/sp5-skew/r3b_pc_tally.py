# build/experiments/sp5-skew/r3b_pc_tally.py
"""R3b-PC HW gate tally (#140 SP-5b Task 4): assert runnability + reproducibility
of the {d_h, d_v} extraction inputs across N HW runs of the hand-authored
sp5_skew_r3b_pc kernel. NO value assertion on d_h/d_v themselves -- SP-5c owns
the actual skew number. This is a reproducibility/runnability gate only,
mirroring r1_tally.py's per-pair skew range-0 check.

Per run, r3b_pc_gate.sh extracts a 24-byte counters.bin (6 x little-endian u32
Performance_Counter0 values, counter_index order) from the runner's
--trace-out sink -- see that script's header comment for why --trace-out
and not --output.

Checks (in order):
  1. observe_r3b(counters, geometry) -> extract_r3b(obs) succeeds for every
     run. A RankDeficientError (design matrix cannot identify {d_h, d_v} from
     this geometry) fails loud and immediately -- this is a static property
     of geometry.json, so if it fires it fires on run 1.
  2. Non-inversion: every raw counter, every run, is non-zero and below a
     sane ceiling. A zero counter means a missed START event or an s1/s2
     ordering inversion (STOP fired before START, so the counter never
     incremented); a value at or above GARBAGE_CEILING means a free-running,
     wrapped, or sentinel readback -- neither is a real bounded interval
     count.
  3. Range-0 b-vector: the per-tile *differenced* readings that extract_r3b
     solves against (r_i - r_reference, i != reference -- reproduced here
     rather than exposed by the merged extract_r3b, which returns only the
     fitted {d_h, d_v, fit_residual}) must be IDENTICAL across all N runs
     (max-min <= tol, default 0). This is the real correctness proxy: a
     systematically-contaminated-but-nonzero counter (e.g. picking up a
     stray broadcast, or a control-packet race) often passes non-inversion
     but is NOT reproducible run to run.

Printed for visibility only (never asserted): the per-run {d_h, d_v,
fit_residual} triple.
"""
import glob
import json
import sys

sys.path.insert(0, "tools")
from calibration.skew.r3b_observe import observe_r3b
from calibration.skew.r3b_extract import extract_r3b

# A raw Performance_Counter0 readback at or above this ceiling is treated as
# garbage (free-running/wrapped counter, or an all-ones-style reset/error
# sentinel) rather than a real bounded interval count. 2**31 is generous
# headroom above any plausible broadcast-arrival interval measured in cycles,
# while still catching wraparound/sentinel values riding the top of the u32
# range.
GARBAGE_CEILING = 2 ** 31


def _load_counters(path):
    return open(path, "rb").read()


def _b_vector(obs, reference=0):
    """Reproduce extract_r3b's internal differencing against `reference` as a
    standalone, inspectable vector. extract_r3b (merged, not modified here)
    returns only the fitted {d_h, d_v, fit_residual}, not the intermediate b
    it solved against -- the range-0 reproducibility check needs the raw
    per-tile differenced values, so this recomputes exactly what
    calibration/skew/r3b_extract.py's extract_r3b loop builds."""
    ref = obs[reference]
    return [o["r"] - ref["r"] for i, o in enumerate(obs) if i != reference]


def main(run_glob, geom_path, max_range=0):
    geom = json.load(open(geom_path))
    runs = sorted(glob.glob(run_glob))
    if not runs:
        raise ValueError(f"no runs matched glob: {run_glob!r}")

    per_tile_r = None   # raw counter word, per tile (counter_index), across runs
    per_tile_b = None   # differenced b-vector entry, per non-reference tile, across runs
    fits = []            # {d_h, d_v, fit_residual} per run, for visibility only

    for run in runs:
        counters = _load_counters(run)
        obs = observe_r3b(counters, geom)  # raises ValueError on short buffer

        raw = [o["r"] for o in obs]
        if per_tile_r is None:
            per_tile_r = [[] for _ in raw]
        for idx, v in enumerate(raw):
            per_tile_r[idx].append(v)

        b = _b_vector(obs)
        if per_tile_b is None:
            per_tile_b = [[] for _ in b]
        for idx, v in enumerate(b):
            per_tile_b[idx].append(v)

        fits.append(extract_r3b(obs))  # raises RankDeficientError loud

    # Non-inversion: every raw counter, every run, non-zero and sane.
    bad = [
        {"run": runs[r], "tile_counter_index": idx, "value": v}
        for idx, vals in enumerate(per_tile_r)
        for r, v in enumerate(vals)
        if v == 0 or v >= GARBAGE_CEILING
    ]
    assert not bad, (
        "non-inversion check failed -- zero or >= 2**31 raw counter "
        f"(missed START or s1/s2 inversion): {bad}"
    )

    # Range-0 (or within max_range) reproducibility of the b-vector across
    # runs -- the real correctness proxy (see module docstring).
    ranges = [max(v) - min(v) for v in per_tile_b]
    print(json.dumps({"n_runs": len(runs), "ranges": ranges}))
    assert all(r <= max_range for r in ranges), f"b-vector not range-0: {ranges}"

    # {d_h, d_v, fit_residual} per run, for visibility only. Printing is not
    # asserting -- SP-5c owns the actual skew value. Under range-0 above,
    # every run's fit is expected to be numerically identical; printed
    # per-run anyway so a real HW batch shows that agreement directly rather
    # than asking the reader to trust it.
    for run, fit in zip(runs, fits):
        print(json.dumps({"run": run, **fit}))

    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
