#!/usr/bin/env python3
"""Which probe points does the itinerary resource checker flag, and where do EMU and silicon differ?

Runs `collision-check elf` over every kernel.o of a probe directory (manifest.json from
gen_acc_timing_probe.py), maps each collision back to the probe point whose assembly lines hold
the colliding bundles (one assembly line is one bundle, a `;`-joined line is still one), and
compares two output tags record by record.

  python3 tools/collision_points.py <dir> --hw r1 --emu e3 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_acc_timing_probe import RECORD_BYTES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(REPO, "target", "debug", "collision-check")
POINT_RE = re.compile(r"//\s*-+\s*point:\s*(\S+) -> (\S+) (\w+) d=(\d+)")
COLLISION_RE = re.compile(
    r"COLLISION (\S+) cycle (\d+) : (\S+)@(\d+)\+(\d+) vs (\S+)@(\d+)\+(\d+)")


def bundle_points(asm_path):
    """bundle index -> (producer, consumer, dir, d) of the point whose lines hold it, else None."""
    owner, cur, n = [], None, 0
    with open(asm_path) as f:
        for line in f:
            s = line.strip()
            m = POINT_RE.search(line)
            if m:
                cur = (m.group(1), m.group(2), m.group(3), int(m.group(4)))
                continue
            if s.startswith("// ==== pair"):
                cur = None
                continue
            if not s or s.startswith("//") or s.startswith(".") or s.endswith(":"):
                continue
            owner.append(cur)
            n += 1
    return owner


def read_records(d, tag, kernel):
    with open(os.path.join(d, tag, kernel["name"] + ".bin"), "rb") as f:
        data = f.read()
    return [data[i * RECORD_BYTES:(i + 1) * RECORD_BYTES] for i in range(len(kernel["points"]))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--hw", required=True)
    ap.add_argument("--emu", required=True)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.dir)
    with open(os.path.join(d, "manifest.json")) as f:
        manifest = json.load(f)
    flagged, differing, cross = {}, {}, []
    total = 0
    for k in manifest["kernels"]:
        name = k["name"]
        owner = bundle_points(os.path.join(d, name + ".s"))
        out = subprocess.run([CHECKER, "elf", os.path.join(d, name, "kernel.o")],
                             capture_output=True, text=True, check=True).stdout
        head = out.splitlines()[0]
        nb = int(re.search(r"bundles (\d+)", head).group(1))
        # The assembler may append padding bundles after the last instruction line.
        assert 0 <= nb - len(owner) <= 2, (name, nb, len(owner))
        owner += [None] * (nb - len(owner))
        for line in out.splitlines()[1:]:
            m = COLLISION_RE.search(line)
            if not m:
                continue
            unit, cycle = m.group(1), int(m.group(2))
            b1, b2 = int(m.group(4)), int(m.group(7))
            p1, p2 = owner[b1], owner[b2]
            if p1 is None or p2 is None or p1 != p2:
                cross.append((name, line.strip()))
                continue
            key = (name,) + p1
            flagged.setdefault(key, []).append(
                {"unit": unit, "cycle": cycle, "first": m.group(3), "second": m.group(6)})
        hw, emu = read_records(d, args.hw, k), read_records(d, args.emu, k)
        for pt, a, b in zip(k["points"], hw, emu):
            total += 1
            if a != b:
                differing[(name, pt["producer"], pt["consumer"], pt["dir"], pt["d"])] = pt
    both = sorted(set(flagged) & set(differing))
    only_flag = sorted(set(flagged) - set(differing))
    only_diff = sorted(set(differing) - set(flagged))
    print(f"points {total}; differing (hw {args.hw} vs emu {args.emu}) {len(differing)}; "
          f"flagged {len(flagged)}; flagged and differing {len(both)}; "
          f"flagged but EMU matched {len(only_flag)}; differing but unflagged {len(only_diff)}; "
          f"cross-point collisions {len(cross)}")
    for title, keys in (("FLAGGED AND DIFFERING", both), ("FLAGGED, EMU MATCHED", only_flag),
                        ("DIFFERING, NOT FLAGGED", only_diff)):
        print(f"\n{title} ({len(keys)})")
        for key in keys:
            c = flagged.get(key, [{}])[0]
            print("  ", *key, c.get("unit", ""), c.get("cycle", ""), c.get("first", ""), c.get("second", ""))
    if cross:
        print(f"\nCROSS-POINT COLLISIONS ({len(cross)})")
        for c in cross:
            print("  ", *c)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"flagged": [list(k) + [v] for k, v in flagged.items()],
                       "differing": [list(k) for k in differing]}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
