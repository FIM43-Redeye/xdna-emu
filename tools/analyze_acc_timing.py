#!/usr/bin/env python3
"""Classify the accumulator timing probe records and extract OLD/NEW thresholds.

For each (producer, consumer) pair the WAR reference point (d = REF_DISTANCE: the consumer reads
long before the producer lands) is the all-OLD record and the RAW reference point is the all-NEW
record. Every other record word is classified against them:

  O  the OLD word        N  the NEW word        .  OLD and NEW coincide (word carries no information)
  X  neither (a mixed or otherwise unexplained value)

The threshold of a word is read off the signed distance s: s = d for RAW points, s = -d for WAR
points (the consumer issues |s| bundles before the producer, 0 shares a bundle). The word sees NEW
exactly when s >= t. Pairs whose sweep is not monotone in s are reported as inconsistent.

  python3 tools/analyze_acc_timing.py <dir> <tag> [--json out.json] [-v]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_acc_timing_probe import RECORD_BYTES, RECORD_WORDS, REF_DISTANCE  # noqa: E402


def load_records(d, tag, kernel):
    with open(os.path.join(d, tag, kernel["name"] + ".bin"), "rb") as f:
        data = f.read()
    n = len(kernel["points"])
    return [struct.unpack_from(f"<{RECORD_WORDS}I", data, i * RECORD_BYTES) for i in range(n)]


def signed_distance(point):
    return point["d"] if point["dir"] == "RAW" else -point["d"]


def classify(rec, old, new):
    out = []
    for w, o, n in zip(rec, old, new):
        if o == n:
            out.append(".")
        elif w == o:
            out.append("O")
        elif w == n:
            out.append("N")
        else:
            out.append("X")
    return "".join(out)


def group_pairs(kernels, tag, d):
    """(producer, consumer) -> list of (point, record) in record order."""
    pairs: dict = {}
    for k in kernels:
        recs = load_records(d, tag, k)
        for pt, rec in zip(k["points"], recs):
            pairs.setdefault((pt["producer"], pt["consumer"]), []).append((pt, rec))
    return pairs


def analyze_pair(points):
    """Return dict with per-word thresholds and a consistency verdict."""
    old = new = None
    for pt, rec in points:
        if (pt["d"] == REF_DISTANCE and pt["dir"] == "WAR") or pt["dir"] == "CONLY":
            old = rec
        if pt["d"] == REF_DISTANCE and pt["dir"] == "RAW":
            new = rec
    if old is None or new is None:
        return {"error": "missing reference"}
    sweep = sorted(((signed_distance(pt), classify(rec, old, new))
                    for pt, rec in points
                    if pt["d"] != REF_DISTANCE and pt["dir"] != "CONLY"), key=lambda t: t[0])
    informative = [i for i in range(RECORD_WORDS) if old[i] != new[i]]
    words = {}
    for i in informative:
        col = [(s, cls[i]) for s, cls in sweep]
        # threshold: the smallest s from which every point sees NEW. Points below it that are not
        # OLD are anomalies (reported, not folded into the threshold).
        t = None
        for s, c in reversed(col):
            if c != "N":
                break
            t = s
        anomalies = [(s, c) for s, c in col if (t is None or s < t) and c != "O"]
        words[i] = {"t": t, "ok": t is not None and not anomalies, "anomalies": anomalies,
                    "col": "".join(c for _, c in col)}
    return {"sweep": sweep, "informative": informative, "words": words,
            "old": list(old), "new": list(new)}


def summarize(res):
    """Group thresholds: {t: [words]} for monotone words, plus the list of bad words."""
    groups: dict = {}
    bad = []
    for i, w in res["words"].items():
        if w["t"] is not None:
            groups.setdefault(w["t"], []).append(i)
        if not w["ok"]:
            bad.append(i)
    return groups, bad


def thresholds(points):
    res = analyze_pair(points)
    if "error" in res:
        return None
    return {i: w["t"] for i, w in res["words"].items()}


def compare(left, right, left_tag, right_tag):
    """Pairs whose per-word thresholds differ between two runs (single-point hazards ignored)."""
    differing = []
    for key in sorted(left):
        a, b = thresholds(left[key]), thresholds(right[key])
        if a is None or b is None:
            continue
        if a != b:
            differing.append((key, sorted(set(a.values()), key=str), sorted(set(b.values()), key=str)))
    for (p, c), ta, tb in differing:
        print(f"{p:18s} {c:24s} {left_tag} t={ta}  {right_tag} t={tb}")
    print(f"pairs with different thresholds: {len(differing)} of {len(left)}")
    return 1 if differing else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("tag")
    ap.add_argument("--json")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--compare", metavar="TAG", help="compare the thresholds with another run, e.g. the emulator")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.dir)
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    pairs = group_pairs(m["kernels"], args.tag, d)
    if args.compare:
        return compare(pairs, group_pairs(m["kernels"], args.compare, d), args.tag, args.compare)
    out = {}
    for (p, c), pts in pairs.items():
        res = analyze_pair(pts)
        if "error" in res:
            print(f"{p:20s} {c:24s} {res['error']}")
            continue
        groups, bad = summarize(res)
        desc = "  ".join(f"t={t}:{len(ws)}w" for t, ws in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])))
        print(f"{p:20s} {c:24s} {desc or 'no informative word'}" + (f"  BAD words {bad}" if bad else ""))
        if args.v:
            for s, cls in res["sweep"]:
                print(f"    s={s:+d} {cls}")
        out[f"{p}:{c}"] = {"groups": {str(t): ws for t, ws in groups.items()}, "bad": bad,
                           "informative": res["informative"]}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
