#!/usr/bin/env python3
"""Analyse MAC conf probe suites (gen_mac_conf_probe.py).

  analyze_mac_conf_probe.py diff <suite> <hw-tag> <emu-tag>   tiles where EMU != HW, by conf
  analyze_mac_conf_probe.py tiles <suite> <tag> <case>          dump a case's tiles (conf, out lanes)

`tile_data` gives the operands and outputs of every tile as plain integers, for ad-hoc models.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

TILE_IN, TILE_OUT = 448, 128


def tile_data(suite: Path, tag: str, case: str):
    topo = json.loads((suite / "cases" / case / "topology.json").read_text())
    data = bytes.fromhex(topo["cores"][0]["args"][0]["data"])
    meta = json.loads((suite / "cases" / case / "case.json").read_text())
    out = (suite / "runs" / tag / case / "core.out.bin").read_bytes()
    n = len(meta["confs"])
    tiles = []
    for t in range(n):
        d = data[t * TILE_IN:(t + 1) * TILE_IN]
        tiles.append(dict(a=d[0:64], b=d[64:128], acc1=d[128:256], acc2=d[256:384],
                          conf=struct.unpack("<I", d[384:388])[0],
                          out=out[t * TILE_OUT:(t + 1) * TILE_OUT]))
    return meta, tiles


def cases(suite: Path):
    return json.loads((suite / "suite.json").read_text())["cases"]


def cmd_diff(suite, hw, emu):
    bad = {}
    total = 0
    for c in cases(suite):
        try:
            meta, th = tile_data(suite, hw, c)
            _, te = tile_data(suite, emu, c)
        except FileNotFoundError:
            print(f"{c}: missing run")
            continue
        for a, b in zip(th, te):
            total += 1
            if a["out"] != b["out"]:
                bad.setdefault((meta["kind"], meta["plan"]), []).append(a["conf"])
    print(f"tiles {total}, differing {sum(len(v) for v in bad.values())}")
    for k, v in sorted(bad.items()):
        print(k, len(v), " ".join(f"{x:#010x}" for x in sorted(set(v))[:24]))


if __name__ == "__main__":
    cmd, suite = sys.argv[1], Path(sys.argv[2])
    if cmd == "diff":
        cmd_diff(suite, sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
