# build/experiments/sp5-skew/r1_tally.py
"""R1 HW gate tally (#140 SP-5b): assert runnability + range-0 reproducibility of
the per-pair skew across N HW runs. NO value assertion -- range-0 is a
reproducibility bound, mirroring SP-3's 20-run evidence."""
import json, sys, glob
sys.path.insert(0, "tools")
from calibration.skew.r1_observe import observe_r1
from calibration.skew.r1_diff_extract import extract_r1_diff


def _load(p):
    return json.load(open(p))["events"]


def main(run_glob, dwall_path, geom_path, max_range=0):
    geom = json.load(open(geom_path))
    dwall = _load(dwall_path)
    per_pair = None
    for run in sorted(glob.glob(run_glob)):
        obs = observe_r1(_load(run), dwall, geom)
        skews = [o["skew"] for o in obs]
        if per_pair is None:
            per_pair = [[] for _ in skews]
        for i, s in enumerate(skews):
            per_pair[i].append(s)
    # Non-degeneracy: >=3 distinct dn_v of kind core among the pairs' endpoints.
    dn_core = {ep["dn_v"] for p in geom["pairs"] for ep in (p["a"], p["b"])
               if ep["pkt_type"] == 0}
    assert len(dn_core) >= 3, f"degenerate: only {len(dn_core)} core dn_v"
    # Range-0 (or within max_range) reproducibility, per pair.
    ranges = [max(v) - min(v) for v in per_pair]
    print(json.dumps({"n_runs": len(per_pair[0]), "ranges": ranges,
                      "dn_core": sorted(dn_core)}))
    # The intra-contrast pair (last) is best-effort (Q2-A): report but do not fail.
    dv_ranges = ranges[:-1]
    assert all(r <= max_range for r in dv_ranges), f"d_v not range-0: {dv_ranges}"
    contrast_ok = ranges[-1] <= max_range
    print("contrast: " + ("range-0" if contrast_ok else "PROVISIONAL (Q2-A)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
