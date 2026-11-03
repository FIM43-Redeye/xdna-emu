#!/usr/bin/env python3
"""Check and analyse the collision probe (gen_collision_probe.py).

  check   <dir>                    every kernel's schedule against the resource checker: collision
                                   points must collide on exactly the designed units at the
                                   designed cycle, controls must not, and no collision may touch
                                   a setup, witness or observation instruction.
  summary <dir> <hw-tag>           per resource group counts from report_<hw-tag>.tsv
  report  <dir> <hw-tag> [<emu-tag>]   classify every point's observation against the reference
                                   points and write <dir>/report_<hw-tag>.tsv

Evidence in <dir>, a suite (harness/suite.py): suite.json, cases/kNNN/{kernel.s,case.json},
pkgs/kNNN/phase_0/core.o, runs/<tag>/kNNN/core.out.bin.
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
from harness import suite  # noqa: E402

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
    s = suite.load_suite(d)
    manifest = s.manifest()
    bad = []
    stats = {"pair": [0, 0], "control": [0, 0], "ref": [0, 0]}
    flagged_units = {}
    for k in manifest["kernels"]:
        owners = bundle_owners(s.kernel_source(k["name"]))
        nb, cols = run_checker(s.kernel_object(k["name"]))
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
                # units named A~B (two write ports) or ending in * (a read the tables do not
                # reserve) are collisions the tables are silent about: the checker cannot flag them.
                table = {u for u in p["units"] if "~" not in u and "*" not in u}
                ok = table <= units
                stats.setdefault("pair_table_silent", [0, 0])
                if not table:
                    stats["pair_table_silent"][0] += 1
                    ok = ok and not units & set(p["units"])
                    stats["pair_table_silent"][1] += ok
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


def words(b):
    import struct
    return struct.unpack(f"<{len(b) // 4}I", b)



def used_bytes(ins):
    return {"cm": 128, "bml": 64, "bmh": 64, "amlh": 32, "amhl": 32, "amhh": 32, "x": 64, "wl": 32, "wh": 32, "mem": 64, "am": 32, "q": 16}[ins.dst]


def ref_key(kind, ins_key, r, dom):
    return (kind, ins_key, tuple(r["s"]), r["d"], r["role"], r["wsel"], dom)


def load_points(d, tag):
    """(manifest, points by id with their record bytes)"""
    s = suite.load_suite(d)
    manifest = s.manifest()
    pts = {}
    for k in manifest["kernels"]:
        data = s.output(tag, k["name"]).read_bytes()
        for i, p in enumerate(k["points"]):
            pts[p["id"]] = (p, data[i * REC:(i + 1) * REC], k["dom"])
    return manifest, pts


def own_output(p, ins, r, rec, slot):
    """Words an instruction left: slot is 'A' or 'B' (which record half the point filled)."""
    lo = 0 if slot == "A" else 128
    if ins.dst == "mem":
        lo = 0 if r["role"] == "A" else 128
    return rec[lo:lo + used_bytes(ins)]



def classify(d, hw_tag, emu_tag=None):
    manifest, hw = load_points(d, hw_tag)
    emu = load_points(d, emu_tag)[1] if emu_tag else None
    refs = {}
    for pid, (p, rec, dom) in hw.items():
        if p["kind"].startswith("ref"):
            ins = gen.BY_KEY[p["a"]]
            refs[ref_key(p["kind"][4:], p["a"], p["ra"], dom)] = (ins, own_output(p, ins, p["ra"], rec, "A"))
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
                        if alone and len(ref[1]) in (64, 128):   # a wide result: low part only
                            for bits, nbytes in (("lo512", 64), ("lo256", 32)):
                                if nbytes >= len(ref[1]):
                                    continue
                                cands[f"src{j + 1}.{bits}<-{other_ins.key}.s{oi + 1}"] = (
                                    ref[1][:nbytes] + alone[1][nbytes:])
            cands["zeros"] = bytes(len(out))
            oalone = refs.get(ref_key("alone", other_ins.key, other_r, dom))
            if oalone and alone:
                o = oalone[1]
                for at in range(0, max(0, len(out) - len(o)) + 1, 32):
                    for opname, fn in (("or", lambda x, y: x | y), ("and", lambda x, y: x & y)):
                        mix = bytearray(alone[1])
                        for i in range(min(len(o), len(mix) - at)):
                            mix[at + i] = fn(mix[at + i], o[i])
                        cands[f"{opname}_other@{at}"] = bytes(mix)
            if oalone:
                o = oalone[1]
                if len(o) == len(out):
                    cands["other_result"] = o
                elif len(o) > len(out):
                    cands["other_result_low"] = o[:len(out)]
                else:
                    if "init" in cands:
                        cands["other_result_low+init"] = o + cands["init"][len(o):]
                    if alone:
                        cands["other_result_low+own_high"] = o + alone[1][len(o):]
            # plain explanations first: a bitwise OR/AND only gets the label when nothing simpler fits
            cands = dict(sorted(cands.items(), key=lambda kv: kv[0].startswith(("or_", "and_"))))
            row[name] = describe(out, cands)
            row[name + "_hex"] = out[:16].hex()
        wl, wh = rec[256:288], rec[288:320]
        row["wit"] = ("n/a" if p["c0"] is None else
                      f"lo={wit_label(wl)} hi={wit_label(wh)}")
        row["emu_equal"] = "" if emu is None else ("same" if emu[pid][1] == rec else "DIFF")
        rows.append(row)
    return rows



def wit_label(b):
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


def summary(d, tag):
    import csv
    import collections
    rows = list(csv.DictReader(open(os.path.join(d, f"report_{tag}.tsv")), delimiter="\t"))
    lab = lambda x: re.sub(r"\(.*", "", x)
    out = []
    pairs = [r for r in rows if r["kind"] == "pair"]
    ctrl = [r for r in rows if r["kind"] == "control"]
    out.append(f"pair points {len(pairs)}, control points {len(ctrl)}")
    out.append(f"witness: " + ", ".join(f"{k}: {v}" for k, v in
                                        collections.Counter(r["wit"] for r in rows).items()))
    emu = [r for r in rows if r["emu_equal"]]
    if emu:
        out.append("EMU equals silicon: " + ", ".join(
            f"{k} {kind}: {sum(1 for r in emu if r['kind'] == kind and r['emu_equal'] == k)}"
            for kind in ("pair", "control") for k in ("same", "DIFF")))
    grp = collections.defaultdict(collections.Counter)
    for r in pairs:
        eff = (lab(r["A"]), lab(r["B"])) != ("alone", "alone")
        grp[(r["units"], r["cfg"])]["effect" if eff else "invisible"] += 1
        if emu:
            grp[(r["units"], r["cfg"])]["emu_diff" if r["emu_equal"] == "DIFF" else "emu_same"] += 1
    out.append("\nunits | cfg | points | effect | invisible | EMU differs from silicon")
    for (u, c), v in sorted(grp.items()):
        out.append(f"{u} | {c} | {v['effect'] + v['invisible']} | {v['effect']} | {v['invisible']} | {v['emu_diff']}")
    print("\n".join(out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "report", "summary"])
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
    if args.cmd == "summary":
        return summary(d, args.hw)
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
