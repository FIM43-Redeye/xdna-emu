#!/usr/bin/env python3
"""Aggregate probe reports into outcome signatures per (unit group, config).

For each group the table gives how many pair points and how many distinct
(A, B) instruction pairs produced each normalised outcome, so a rule's
evidence base (distinct class pairs, not point count) is explicit.

Usage: collision_rules.py report_dir [report_dir ...] --tag r1
"""
import argparse
import collections
import csv
import os
import re


def normalise(label):
    label = re.sub(r"<-[a-z0-9_]+\.s\d", "<-MOVER", label)
    label = re.sub(r"\(.*", "(...)", label)
    return label


def load(dirs, tag):
    rows = []
    for d in dirs:
        path = os.path.join(d, f"report_{tag}.tsv")
        for r in csv.DictReader(open(path), delimiter="\t"):
            r["probe"] = os.path.basename(os.path.normpath(d))
            rows.append(r)
    return rows


def signatures(rows):
    table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, set()]))
    for r in rows:
        if r["kind"] != "pair":
            continue
        sig = (normalise(r["A"]), normalise(r["B"]))
        cell = table[(r["units"], r["cfg"])][sig]
        cell[0] += 1
        cell[1].add((r["a"], r["b"]))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--min-pairs", type=int, default=4,
                    help="flag outcomes resting on fewer distinct class pairs")
    args = ap.parse_args()
    table = signatures(load(args.dirs, args.tag))
    print("units\tcfg\tA_outcome\tB_outcome\tpoints\tdistinct_pairs\tweak")
    for (units, cfg), sigs in sorted(table.items()):
        for (a, b), (n, pairs) in sorted(sigs.items(), key=lambda kv: -kv[1][0]):
            weak = "WEAK" if len(pairs) < args.min_pairs and (a, b) != ("alone", "alone") else ""
            print(f"{units}\t{cfg}\t{a}\t{b}\t{n}\t{len(pairs)}\t{weak}")


if __name__ == "__main__":
    main()
