#!/usr/bin/env python3
"""Count the probe points whose EMU record differs from the silicon record.

  diff_probe_points.py <probe-dir> <hw-tag> <emu-tag> [--list]

Works on both probe layouts: the collision probes (manifest `kernels[].points[]`, fixed
RECORD_BYTES records, `kind` = pair | control | ref*) and the accumulator-timing probe
(`record_bytes` in the manifest). Prints `differ/total` per kind and in total.
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("hw")
    ap.add_argument("emu")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    d = os.path.abspath(a.dir)
    m = json.load(open(os.path.join(d, "manifest.json")))
    rec = m.get("record_bytes", 320)
    tot, bad = collections.Counter(), collections.Counter()
    for k in m["kernels"]:
        hw = open(os.path.join(d, a.hw, k["name"] + ".bin"), "rb").read()
        try:
            emu = open(os.path.join(d, a.emu, k["name"] + ".bin"), "rb").read()
        except FileNotFoundError:
            emu = b""
        for i, p in enumerate(k["points"]):
            kind = p.get("kind", p.get("dir", "pt"))
            kind = "ref" if str(kind).startswith("ref") else kind
            h, e = hw[i * rec:(i + 1) * rec], emu[i * rec:(i + 1) * rec]
            tot[kind] += 1
            if h != e:
                bad[kind] += 1
                if a.list:
                    print("DIFF", k["name"], i, p.get("id"), p.get("a"), p.get("b"), p.get("g"), p.get("cfg"), p.get("kind"))
    parts = ", ".join(f"{kd} {bad[kd]}/{tot[kd]}" for kd in sorted(tot))
    print(f"{os.path.basename(d)} {a.hw} vs {a.emu}: DIFFER {sum(bad.values())}/{sum(tot.values())} ({parts})")


if __name__ == "__main__":
    main()
