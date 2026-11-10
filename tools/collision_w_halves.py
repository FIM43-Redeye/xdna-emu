#!/usr/bin/env python3
"""Who owns each 32-byte half of a W destination after two writes collide.

A W register x is {wl, wh}; a 512-bit writer fills both halves and a 256-bit writer one. For
every W-family collision this labels each half of the destination as written by A, by B, or
still the initial value, so the lane model (port fixed vs destination half) can be read off.

Usage: collision_w_halves.py probe_dir [--tag r1] [--detail]
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_collision_probe as an  # noqa: E402
import gen_collision_probe as gen     # noqa: E402

W = gen.W_FAMILY + ("cm", "bml", "am")
HALF = 32


def cover(dst):
    """Halves of the W register a destination kind covers."""
    return {"x": (0, 1), "wl": (0,), "wh": (1,),
            "cm": (0, 1, 2, 3), "bml": (0, 1), "am": (0,)}[dst]


def halves(b):
    return [b[i:i + HALF] for i in range(0, len(b), HALF)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    hw = an.load_points(args.dir, args.tag)
    refs = {}
    for pid, (p, rec, dom) in hw.items():
        if p["kind"].startswith("ref"):
            ins = gen.BY_KEY[p["a"]]
            refs[an.ref_key(p["kind"][4:], p["a"], p["ra"], dom)] = an.own_output(p, ins, p["ra"], rec, "A")
    table = collections.defaultdict(lambda: collections.defaultdict(set))
    for pid in sorted(hw):
        p, rec, dom = hw[pid]
        if p["kind"] == "ref" or p["kind"].startswith("ref"):
            continue
        a, b = gen.BY_KEY[p["a"]], gen.BY_KEY[p["b"]]
        if a.dst not in W or b.dst not in W:
            continue
        res = {}
        for slot, ins, r in (("A", a, p["ra"]), ("B", b, p["rb"])):
            alone = refs.get(an.ref_key("alone", ins.key, r, dom))
            none = refs.get(an.ref_key("none", ins.key, r, dom))
            if alone is None or none is None:
                break
            res[slot] = (ins, alone, none, an.own_output(p, ins, r, rec, slot))
        else:
            # per-half value of A's and B's own results and of the initial contents
            own = {s: dict(zip(cover(res[s][0].dst), halves(res[s][1]))) for s in res}
            init = {s: dict(zip(cover(res[s][0].dst), halves(res[s][2]))) for s in res}
            final = {}
            for s in res:
                for h, v in zip(cover(res[s][0].dst), halves(res[s][3])):
                    final.setdefault(h, {})[s] = v
            labels = []
            for h in (0, 1, 2, 3):
                if h not in final:
                    continue
                v = list(final[h].values())[0] if p["cfg"] == "dstshare" else None
                for s in res:
                    if p["cfg"] != "dstshare" and h in final and s in final[h]:
                        v = final[h][s]
                        lab = []
                        if h in own[s] and v == own[s][h]:
                            lab.append("own")
                        if h in init[s] and v == init[s][h]:
                            lab.append("init")
                        other = "B" if s == "A" else "A"
                        if any(v == ov for ov in own[other].values()):
                            lab.append("other")
                        labels.append(f"{s}.{h}={'|'.join(lab) or '?'}")
                if p["cfg"] == "dstshare":
                    lab = []
                    for s in res:
                        if h in own[s] and v == own[s][h]:
                            lab.append(s)
                    if h in init["A"] and v == init["A"][h]:
                        lab.append("init")
                    labels.append(f"{h}={'|'.join(lab) or '?'}")
            ports = "+".join(p["units"])
            key = (p["cfg"], a.dst, b.dst, p["kind"] if p["kind"] == "control" else "pair",
                   p["delta"], "W_WA/W_WM" if "~" in ports else ports.replace("CM_RM_PORT+UPPER_SRS+", ""))
            table[key][" ".join(labels)].add((p["a"], p["b"]))
            if args.detail and p["kind"] == "pair":
                print(pid, p["a"], p["b"], p["cfg"], " ".join(labels))
    if not args.detail:
        for key, outs in sorted(table.items()):
            if key[3] == "control" and key[4] != 0 and all("own" in o or "=A" in o or "=B" in o for o in outs) and False:
                continue
            for o, pairs in sorted(outs.items(), key=lambda kv: -len(kv[1])):
                print(key, "|", o, "|", len(pairs), "class pairs")


if __name__ == "__main__":
    main()
