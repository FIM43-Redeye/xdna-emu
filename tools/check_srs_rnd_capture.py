#!/usr/bin/env python3
"""Check an srs-rnd run (gen_dynamic_sign_probe.py --family srs-rnd) against an independent
model of the crRnd table in docs/fidelity-gaps/vector-compute.md: round, then clamp (crSat 1).

  python3 tools/check_srs_rnd_capture.py <suite_dir>/peano <run_tag>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_dynamic_sign_probe as g  # noqa: E402

def rnd(v, s, mode):
    q = v >> s; d = v - (q << s); half = 1 << (s - 1) if s else 0
    inexact = d != 0
    up = {0: False, 1: inexact, 2: inexact and v < 0, 3: inexact and v >= 0}.get(mode)
    if mode in (4, 6): return q + (inexact and q % 2 == 1)
    if mode in (5, 7): return q + (inexact and q % 2 == 0)
    if up is not None: return q + up
    m = {14: 12, 15: 13}.get(mode, mode)
    if d != half: return q + (d > half)
    tie = {8: False, 9: True, 10: v < 0, 11: v >= 0, 12: q % 2 == 1, 13: q % 2 == 0}[m]
    return q + tie

D = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
tot = bad = 0
for c in sorted((D / "cases").iterdir()):
    case = json.load(open(c / "case.json")); m = case.get("meta", case)
    form = m["form"]; key = form.removeprefix("rnd_").rsplit("_", 1)[0]
    fn, aty, ab, lanes, rty, ob, wide = g.SRS_PEANO[key]
    ri, ro = m["rec_in"], m["rec_out"]
    inp = next((D / "runs" / tag).glob(f"{c.name}/phase_0.in.bin")).read_bytes()
    out = next((D / "runs" / tag).glob(f"{c.name}/phase_0.out.bin")).read_bytes()
    nb = ob * lanes // 8
    cb = bad
    for i, (sign, sh, mode) in enumerate(m["records"]):
        r = inp[i * ri:(i + 1) * ri]; o = out[i * ro:i * ro + nb]
        for L in range(lanes):
            v = int.from_bytes(r[L * ab // 8:(L + 1) * ab // 8], "little", signed=True)
            x = rnd(v, sh, mode)
            lo, hi = ((-(1 << (ob - 1)), (1 << (ob - 1)) - 1) if sign & 1 else (0, (1 << ob) - 1))
            x = min(max(x, lo), hi)
            got = int.from_bytes(o[L * ob // 8:(L + 1) * ob // 8], "little", signed=bool(sign & 1))
            tot += 1; bad += got != x
    print(c.name, "mismatch lanes:", bad - cb)
print("total lanes", tot, "bad", bad)
sys.exit(1 if bad else 0)
