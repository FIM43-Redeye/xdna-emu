#!/usr/bin/env python3
"""Check the W write-port lane model against a probe.

Model (A issued g bundles before B, both write the W register file in one cycle through one
port): two 256-bit data lanes D0 and D1 feed two 256-bit banks, bank0 = wl, bank1 = wh.
 - Every writer drives D0 (a 512-bit writer with its low half); only a 512-bit writer drives D1
   (with its high half). The later-issued writer (B) wins a lane both drive.
 - Each bank takes the register of the later writer that enables it; the bank captures D0
   (bank0), or D1 if some 512-bit writer drives it, else D0 (bank1).
 - A bank neither writer enables is untouched.
Prints, per (port set, config), how many pair points the model predicts exactly.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_collision_probe as an  # noqa: E402
import gen_collision_probe as gen     # noqa: E402
from collision_w_halves import cover, halves  # noqa: E402


FAMILIES = {
    "w": (("x", "wl", "wh"), ("W_WA_PORT", "W_WM_PORT"), "W_W"),
    "cm": (("cm", "bml", "am"), ("CM_WA_PORT", "CM_WM_PORT"), "CM_W"),
}


def predict(a_dst, b_dst, a_data, b_data, init, shared):
    """Final halves per (writer, bank) for the two destination registers."""
    nlanes = max(len(a_data), len(b_data))
    lanes = [b_data[i] if i < len(b_data) else (a_data[i] if i < len(a_data) else None)
             for i in range(nlanes)]
    d0 = lanes[0]
    reg = (lambda w: "R") if shared else (lambda w: w)      # one register or two
    state = {(reg("B"), h): init[("B", h)] for h in cover(b_dst)}
    state.update({(reg("A"), h): init[("A", h)] for h in cover(a_dst)})
    for bank in range(4):
        owners = [w for w, dst in (("A", a_dst), ("B", b_dst)) if bank in cover(dst)]
        if not owners:
            continue
        target = "B" if "B" in owners else "A"
        lane = lanes[bank] if bank < nlanes else None
        state[(reg(target), bank)] = lane if lane is not None else d0
    return state


def predict_cross(a_dst, b_dst, a_data, b_data, init, shared, winner):
    """Two ports: no lane sharing. A bank written by both goes to `winner` ("A" or "B") with that
    writer's own data for the bank; in separate registers nothing collides."""
    reg = (lambda w: "R") if shared else (lambda w: w)
    state = {(reg("B"), h): init[("B", h)] for h in cover(b_dst)}
    state.update({(reg("A"), h): init[("A", h)] for h in cover(a_dst)})
    own = {"A": dict(zip(cover(a_dst), a_data)), "B": dict(zip(cover(b_dst), b_data))}
    for w, dst in (("A", a_dst), ("B", b_dst)):
        for h in cover(dst):
            both = h in cover(a_dst) and h in cover(b_dst) and shared
            if not both or w == winner:
                state[(reg(w), h)] = own[w][h]
    return state


def swapped(state):
    """A state computed with the roles of A and B exchanged, relabelled."""
    flip = {"A": "B", "B": "A", "R": "R"}
    return {(flip[w], h): v for (w, h), v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--family", choices=sorted(FAMILIES), default="w")
    args = ap.parse_args()
    hw = an.load_points(args.dir, args.tag)
    refs = {}
    for pid, (p, rec, dom) in hw.items():
        if p["kind"].startswith("ref"):
            refs[an.ref_key(p["kind"][4:], p["a"], p["ra"], dom)] = an.own_output(p, gen.BY_KEY[p["a"]], p["ra"], rec, "A")
    dsts, port_names, prefix = FAMILIES[args.family]
    tally = collections.defaultdict(lambda: [0, 0])
    for pid in sorted(hw):
        p, rec, dom = hw[pid]
        if p["kind"] != "pair":
            continue
        cross = [u for u in p["units"] if "~" in u and u.split("~")[0].startswith(prefix)]
        if "".join(p["units"]).count("~") and not cross:
            continue
        a, b = gen.BY_KEY[p["a"]], gen.BY_KEY[p["b"]]
        ports = [u for u in p["units"] if u in port_names] or cross
        if a.dst not in dsts or b.dst not in dsts or len(ports) != 1:
            continue
        ra_alone = refs.get(an.ref_key("alone", a.key, p["ra"], dom))
        rb_alone = refs.get(an.ref_key("alone", b.key, p["rb"], dom))
        a_none = refs.get(an.ref_key("none", a.key, p["ra"], dom))
        b_none = refs.get(an.ref_key("none", b.key, p["rb"], dom))
        if None in (ra_alone, rb_alone, a_none, b_none):
            continue
        init = {}
        for w, dst, none in (("A", a.dst, a_none), ("B", b.dst, b_none)):
            for h, v in zip(cover(dst), halves(none)):
                init[(w, h)] = v
        shared = p["cfg"] == "dstshare"
        st = predict(a.dst, b.dst, halves(ra_alone), halves(rb_alone), init, shared)
        if cross:
            a_port = cross[0].split("~")[0]
            wa_writer = "A" if a_port.endswith("WA_PORT") else "B"
            args_ = (a.dst, b.dst, halves(ra_alone), halves(rb_alone), init, shared)
            st = predict_cross(*args_, winner="B")            # later-issued writer wins
            st_wa = predict_cross(*args_, winner=wa_writer)   # the W_WA writer wins
        obs = collections.defaultdict(list)
        for slot, ins, r in (("A", a, p["ra"]), ("B", b, p["rb"])):
            out = an.own_output(p, ins, r, rec, slot)
            for h, v in zip(cover(ins.dst), halves(out)):
                obs[("R" if p["cfg"] == "dstshare" else slot, h)].append(v)
        # in dstshare both slots observe the same register: they must agree with each other too
        fits = lambda state: all(all(o == v for o in obs[k]) for k, v in state.items() if k in obs)
        ok = fits(st)
        if cross:
            key = ("cross", p["cfg"])
            tally[key + ("later wins",)][0] += ok
            tally[key + ("later wins",)][1] += 1
            tally[key + ("W_WA wins",)][0] += fits(st_wa)
            tally[key + ("W_WA wins",)][1] += 1
            if args.show_misses and not (ok or fits(st_wa)):
                print("CROSS-MISS", pid, p["a"], p["b"], p["cfg"], cross)
            continue
        key = (ports[0], p["cfg"])
        tally[key][0] += ok
        tally[key][1] += 1
        if not ok and args.show_misses:
            print("MISS", pid, p["a"], p["b"], p["cfg"], [k for k, v in st.items() if k in obs and any(o != v for o in obs[k])])
    for key, (ok, n) in sorted(tally.items()):
        print(key, f"{ok}/{n} predicted")


if __name__ == "__main__":
    main()
