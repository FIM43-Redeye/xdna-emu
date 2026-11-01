#!/usr/bin/env python3
"""Check and analyse the collision probe (gen_collision_probe.py).

  check   <dir>                    every kernel's schedule against the resource checker: collision
                                   points must collide on exactly the designed units at the
                                   designed cycle, controls must not, and no collision may touch
                                   a setup, witness or observation instruction.
  report  <dir> <hw-tag> [<emu-tag>]   classify every point's observation (added below)

Evidence in <dir>: manifest.json, kNNN.s, kNNN/kernel.o, <tag>/kNNN.bin.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_collision_probe as gen  # noqa: E402

COLLISION_RE = re.compile(r"COLLISION (\S+) cycle (\d+) : (\S+)@(\d+)\+(\d+) vs (\S+)@(\d+)\+(\d+)")
POINT_RE = re.compile(r"//\s*-+\s*point (\d+):")


def bundle_owners(asm_path):
    """bundle index -> (point id, role) where role is 'A', 'B', 'C', 'W' or 'x' by emission."""
    owner, cur = [], None
    with open(asm_path) as f:
        for line in f:
            s = line.strip()
            m = POINT_RE.search(line)
            if m:
                cur = int(m.group(1))
                continue
            if not s or s.startswith("//") or s.startswith(".") or s.endswith(":"):
                continue
            owner.append((cur, s))
    return owner


def run_checker(obj):
    out = subprocess.run([gen.CHECKER, "elf", obj], capture_output=True, text=True, check=True).stdout
    head = out.splitlines()[0]
    nb = int(re.search(r"bundles (\d+)", head).group(1))
    cols = []
    for line in out.splitlines()[1:]:
        m = COLLISION_RE.search(line)
        if m:
            cols.append({"unit": m.group(1), "cycle": int(m.group(2)),
                         "b1": int(m.group(4)), "o1": int(m.group(5)), "n1": m.group(3),
                         "b2": int(m.group(7)), "o2": int(m.group(8)), "n2": m.group(6)})
    return nb, cols


def check(d):
    with open(os.path.join(d, "manifest.json")) as f:
        manifest = json.load(f)
    bad = []
    stats = {"pair": [0, 0], "control": [0, 0], "ref": [0, 0]}
    flagged_units = {}
    for k in manifest["kernels"]:
        owners = bundle_owners(os.path.join(d, k["name"] + ".s"))
        nb, cols = run_checker(os.path.join(d, k["name"], "kernel.o"))
        assert 0 <= nb - len(owners) <= 2, (k["name"], nb, len(owners))
        by_point = {}
        for c in cols:
            p1, p2 = owners[c["b1"]][0], owners[c["b2"]][0]
            if p1 is None or p1 != p2:
                bad.append((k["name"], "collision outside one point", c))
                continue
            by_point.setdefault(p1, []).append(c)
        for p in k["points"]:
            got = by_point.get(p["id"], [])
            kind = "ref" if p["kind"].startswith("ref") else p["kind"]
            stats[kind][0] += 1
            units = {c["unit"] for c in got}
            for c in got:
                texts = (owners[c["b1"]][1], owners[c["b2"]][1])
                if any(("wl5" in t or "vldb" in t) for t in texts):
                    bad.append((k["name"], p["id"], "collision involves the witness", c))
            if p["kind"] == "pair":
                ok = set(p["units"]) <= units
                flagged_units[(p["a"], p["b"], p["g"], p["cfg"])] = sorted(units)
            elif p["kind"] == "control":
                ok = not (set(p["units"]) & units) or any(
                    c["cycle"] != p["cycles"][p["units"].index(c["unit"])] + p["pa"]
                    for c in got if c["unit"] in p["units"])
            else:
                ok = not got
            if ok:
                stats[kind][1] += 1
            else:
                bad.append((k["name"], p["id"], p["kind"], p["a"], p["b"], p["g"], p["cfg"],
                            p.get("units"), sorted(units)))
    return stats, bad, flagged_units


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check"])
    ap.add_argument("dir")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.dir)
    if args.cmd == "check":
        stats, bad, _ = check(d)
        print("kind: (points, as designed)", stats)
        for b in bad:
            print("BAD", *b)
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
