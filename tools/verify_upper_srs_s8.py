#!/usr/bin/env python3
"""Check the byte placement of the UPPER_SRS OR for the s8-output partners (bit-pattern probe).

A vsrsm (reads the low 512 bits of its register) issued with a cm-source `vsrs.s8.s32` or
`vst.srs.s8.s32` ORs its source with the partner's high-half source (see verify_upper_srs.py).
For the s8 partners the OR'd 512-bit bus does not come out as 8-bit conversions: bytes 16..31 of
the partner's result hold, for m = 0..7, the low 16 bits (little endian) of 32-bit bus word 2m.
Bytes 0..15 are the partner's own (alone) result. The vsrsm's result is checked by
verify_upper_srs.py. Prints, per (config, partner, vsrsm shape), whether the prediction holds.
"""
import collections
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import analyze_collision_probe as an  # noqa: E402
import gen_collision_probe as gen     # noqa: E402

S8_PARTNERS = ("vsrs_s8_s32", "vst_srs_s8_s32")


def pattern_words(table, k, half):
    """32-bit accumulator words of the low (half 0) or high (half 1) 512 bits of cm<k>: the
    pattern's int16 elements widened by `vups.s32.s16`."""
    off = gen.TABLE_OFFSETS[f"PAT_i{k}"] + 32 * half
    return [struct.unpack_from("<h", table, off + 2 * w)[0] & 0xFFFFFFFF for w in range(16)] if False else \
        [struct.unpack_from("<h", table, gen.TABLE_OFFSETS[f"PAT_i{k}"] + 2 * (16 * half + w))[0] & 0xFFFFFFFF
         for w in range(16)]


def predict_hi(vsrsm_reg, partner_reg, table):
    low = pattern_words(table, vsrsm_reg, 0)
    high = pattern_words(table, partner_reg, 1)
    bus = [a | b for a, b in zip(low, high)]
    return b"".join(struct.pack("<H", bus[2 * m] & 0xFFFF) for m in range(8))


def main(d):
    table = open(f"{d}/input.bin", "rb").read()
    hw = an.load_points(d, "r1")
    refs = {}
    for pid, (p, rec, dom) in hw.items():
        if p["kind"].startswith("ref"):
            refs[an.ref_key(p["kind"][4:], p["a"], p["ra"], dom)] = an.own_output(p, gen.BY_KEY[p["a"]], p["ra"], rec, "A")
    tally = collections.defaultdict(lambda: [0, 0])
    for pid, (p, rec, dom) in sorted(hw.items()):
        if p["kind"] != "pair":
            continue
        for slot, other in (("A", "B"), ("B", "A")):
            key = p["a" if slot == "A" else "b"]
            if key not in S8_PARTNERS:
                continue
            vkey = p["b" if slot == "A" else "a"]
            if not vkey.startswith("vsrsm"):
                continue
            pr = p["ra" if slot == "A" else "rb"]
            vr = p["rb" if slot == "A" else "ra"]
            out = an.own_output(p, gen.BY_KEY[key], pr, rec, slot)
            alone = refs[an.ref_key("alone", key, pr, dom)]
            want = alone[:16] + predict_hi(vr["s"][0], pr["s"][0], table)
            tally[(p["cfg"], key, vkey)][0] += out[:32] == want
            tally[(p["cfg"], key, vkey)][1] += 1
    for k, (ok, n) in sorted(tally.items()):
        print(k, f"{ok}/{n}")
    print("total", sum(v[0] for v in tally.values()), "/", sum(v[1] for v in tally.values()))


if __name__ == "__main__":
    main(sys.argv[1])
