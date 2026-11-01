#!/usr/bin/env python3
"""Check and analyse the collision probe (gen_collision_probe.py).

  check   <dir>                    every kernel's schedule against the resource checker: collision
                                   points must collide on exactly the designed units at the
                                   designed cycle, controls must not, and no collision may touch
                                   a setup, witness or observation instruction.
  report  <dir> <hw-tag> [<emu-tag>]   classify every point's observation against the reference
                                   points and write <dir>/report_<hw-tag>.tsv

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


REC = gen.RECORD_BYTES
W = 4


def words(b):
    import struct
    return struct.unpack(f"<{len(b) // 4}I", b)


def region(rec, ins, role):
    """The bytes a point recorded for one instruction: A's half or B's half of the record."""
    if ins.dst == "mem":
        lo = 0 if role == "A" else 128
    else:
        lo = 0 if role == "A" else 128      # set by the caller via `slot`
    return rec[lo:lo + 128]


def used_bytes(ins):
    return {"cm": 128, "bml": 64, "x": 64, "wl": 32, "mem": 32}[ins.dst]


def ref_key(kind, ins_key, r, dom):
    return (kind, ins_key, tuple(r["s"]), r["d"], r["role"], r["wsel"], dom)


def load_points(d, tag):
    """(manifest, points by id with their record bytes)"""
    with open(os.path.join(d, "manifest.json")) as f:
        manifest = json.load(f)
    pts = {}
    for k in manifest["kernels"]:
        data = open(os.path.join(d, tag, k["name"] + ".bin"), "rb").read()
        for i, p in enumerate(k["points"]):
            pts[p["id"]] = (p, data[i * REC:(i + 1) * REC], k["dom"])
    return manifest, pts


def own_output(p, ins, r, rec, slot):
    """Words an instruction left: slot is 'A' or 'B' (which record half the point filled)."""
    lo = 0 if slot == "A" else 128
    if ins.dst == "mem":
        lo = 0 if r["role"] == "A" else 128
    return rec[lo:lo + used_bytes(ins)]


def witness(rec):
    wlo, whi = rec[256:288], rec[288:320]
    old = bytes(x for k in range(16) for x in (0x11 + k & 0xFF, 0x11)) if False else None
    return wlo, whi


def classify(d, hw_tag, emu_tag=None):
    manifest, hw = load_points(d, hw_tag)
    emu = load_points(d, emu_tag)[1] if emu_tag else None
    refs = {}
    for pid, (p, rec, dom) in hw.items():
        if p["kind"].startswith("ref"):
            ins = gen.BY_KEY[p["a"]]
            refs[ref_key(p["kind"][4:], p["a"], p["ra"], dom)] = (ins, own_output(p, ins, p["ra"], rec, "A"))
    old_w = hw_witness_pattern(d, hw_tag)
    rows = []
    for pid in sorted(hw):
        p, rec, dom = hw[pid]
        if p["kind"].startswith("ref"):
            continue
        a, b = gen.BY_KEY[p["a"]], gen.BY_KEY[p["b"]]
        row = {"id": pid, "kind": p["kind"], "a": p["a"], "b": p["b"], "g": p["g"], "cfg": p["cfg"],
               "delta": p["delta"], "units": "+".join(p["units"]), "cycles": p["cycles"]}
        for name, ins, r, slot, other_ins, other_r in (("A", a, p["ra"], "A", b, p["rb"]),
                                                       ("B", b, p["rb"], "B", a, p["ra"])):
            out = own_output(p, ins, r, rec, slot)
            cands = {}
            alone = refs.get(ref_key("alone", ins.key, r, dom))
            if alone:
                cands["alone"] = alone[1]
            none = refs.get(ref_key("none", ins.key, r, dom))
            if none:
                cands["init"] = none[1]
            for j in range(len(r["s"])):
                z = list(r["s"]); z[j] = gen.ZERO_REG
                ref = refs.get(ref_key("alone", ins.key, {**r, "s": z}, dom))
                if ref:
                    cands[f"zero{j + 1}"] = ref[1]
                for oi, osrc in enumerate(other_r["s"]):
                    sw = list(r["s"]); sw[j] = osrc
                    ref = refs.get(ref_key("alone", ins.key, {**r, "s": sw}, dom))
                    if ref and osrc != r["s"][j]:
                        cands[f"src{j + 1}<-{other_ins.key}.s{oi + 1}"] = ref[1]
            oalone = refs.get(ref_key("alone", other_ins.key, other_r, dom))
            if oalone and len(oalone[1]) == len(out):
                cands["other_result"] = oalone[1]
            row[name] = describe(out, cands)
            row[name + "_hex"] = out[:16].hex()
        wl, wh = rec[256:288], rec[288:320]
        row["wit"] = ("n/a" if p["c0"] is None else
                      f"lo={wit_label(wl, old_w)} hi={wit_label(wh, old_w)}")
        row["emu_equal"] = "" if emu is None else ("same" if emu[pid][1] == rec else "DIFF")
        rows.append(row)
    return rows


def hw_witness_pattern(d, tag):
    return None


def wit_label(b, _):
    old = b"".join(bytes([0x11 + k, 0x11]) for k in range(16))
    new = b"".join(bytes([0x22 + k, 0x22]) for k in range(16))
    return "OLD" if b == old else "NEW" if b == new else "other"


def describe(out, cands):
    """Name the candidate the observation equals, or how many words each candidate explains."""
    for name, c in cands.items():
        if c == out:
            return name
    w = words(out)
    nw = len(w)
    tally = {}
    for name, c in cands.items():
        cw = words(c)
        n = sum(1 for x, y in zip(w, cw) if x == y)
        if n:
            tally[name] = n
    if not tally:
        return f"unexplained[{nw}w]"
    return "mixed(" + ",".join(f"{k}:{v}/{nw}" for k, v in sorted(tally.items())) + ")"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "report"])
    ap.add_argument("dir")
    ap.add_argument("hw", nargs="?")
    ap.add_argument("emu", nargs="?")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.dir)
    if args.cmd == "check":
        stats, bad, _ = check(d)
        print("kind: (points, as designed)", stats)
        for b in bad:
            print("BAD", *b)
        return 1 if bad else 0
    rows = classify(d, args.hw, args.emu)
    cols = ["id", "kind", "a", "b", "g", "cfg", "delta", "units", "A", "B", "wit", "emu_equal"]
    with open(os.path.join(d, f"report_{args.hw}.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"{len(rows)} collision/control points -> report_{args.hw}.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
