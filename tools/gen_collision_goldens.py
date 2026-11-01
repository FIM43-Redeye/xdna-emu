#!/usr/bin/env python3
"""Emit the silicon golden table of the register-port collision model.

Reads collision-probe evidence directories (their manifest, input table and silicon run), picks
the pair points that show an effect, one per distinct (port group, configuration, outcome
signature), and writes `src/interpreter/execute/collision_goldens_data.rs`: for every point the
two instructions' bytes, their bundle positions, the register state the probe kernel starts
the pair from and the silicon words the two destination registers hold afterwards.

  gen_collision_goldens.py OUT.rs  DIR:TAG:SRS_SHIFT [DIR:TAG:SRS_SHIFT ...] [--per-group N]

`DIR` is a probe directory, `TAG` its silicon run (r1) and `SRS_SHIFT` the shift the probe was
generated with (`COLLISION_SRS_SHIFT`, 0 when unset). Points that store to memory or write a q
register are skipped: the table checks register contents.
"""
import argparse
import collections
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_collision_probe as an  # noqa: E402
import collision_rules as rules       # noqa: E402
import gen_collision_probe as gen     # noqa: E402

base = gen.base
OBJCOPY = os.path.join(os.path.dirname(gen.LLVM_MC), "llvm-objcopy")
# destination kind -> (first 32-byte piece counted from the start of the cm / x register, number
# of pieces) for the kinds the table can read back
ACC_PIECES = {"cm": (0, 4), "bml": (0, 2), "bmh": (2, 2), "am": (0, 1), "amlh": (1, 1),
              "amhl": (2, 1), "amhh": (3, 1)}
W_PIECES = {"x": (0, 2), "wl": (0, 1), "wh": (1, 1)}


def assemble(asm: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        s, o, b = (os.path.join(tmp, n) for n in ("a.s", "a.o", "a.bin"))
        open(s, "w").write(".text\n  " + asm + "\n")
        subprocess.run([gen.LLVM_MC, "--triple=aie2", "--filetype=obj", "-o", o, s], check=True,
                       capture_output=True)
        subprocess.run([OBJCOPY, "-O", "binary", "--only-section=.text", o, b], check=True)
        return open(b, "rb").read()


def sext16(v):
    return v - 0x10000 if v & 0x8000 else v


def acc_image(table, k, dom, shift):
    """Sixteen u64 lanes of cm<k> after the prologue's vups (bml lanes 0..8, bmh 8..16)."""
    if k == gen.ZERO_REG:
        return [0] * 16
    off = offset(table, f"PAT_{dom}{k}")
    elems = struct.unpack_from("<32H", table, off)
    words = [(sext16(e) << shift) & 0xFFFFFFFF for e in elems]
    return [words[2 * j] | words[2 * j + 1] << 32 for j in range(16)]


def offset(table, name):
    """Byte offset of an item. The first two probes (1824-byte tables) predate the CARRY items
    that sit in front of WINIT / WOLD / WNEW; everything after them is 128 bytes earlier there."""
    off = gen.TABLE_OFFSETS[name]
    return off - 128 if len(table) == 1824 and off >= gen.TABLE_OFFSETS["CARRY_A"] else off


def table_slice(table, name, extra=0, n=32):
    off = offset(table, name) + extra
    return struct.unpack_from(f"<{n // 4}I", table, off)


def variant_state(table, dom, srs_shift):
    """The register state every point of this probe directory starts from."""
    shift = 0 if dom == "i" else 16
    acc = [acc_image(table, k, dom, shift) for k in range(gen.NUM_PAT + 1)]
    vec = {}
    for n, name in enumerate(("X0", "X1", "X2", "X3")):
        vec[2 * n], vec[2 * n + 1] = table_slice(table, name), table_slice(table, name, 32)
    for k in range(4):
        vec[12 + k] = table_slice(table, "Y3", 32 * k)
    vec[20], vec[21] = table_slice(table, f"NEWX_{dom}"), table_slice(table, f"NEWX_{dom}", 32)
    for reg in (16, 8):                            # x8 / x4 (and wl/wh of the same numbers)
        vec[reg], vec[reg + 1] = table_slice(table, "WINIT"), table_slice(table, "WINIT", 32)
    q = {1: struct.unpack_from("<4I", table, offset(table, "Q1")),
         3: struct.unpack_from("<4I", table, offset(table, "Q3"))}
    scalars = {i + 1: base.config_word(f"r{i + 1}") for i in range(4)}
    scalars.update({12: 9320, 29: 0, 40 + 3: shift, 40 + 1: srs_shift or shift})
    return {"acc": acc, "vec": vec, "q": q, "scalars": scalars, "table": table,
            "newx": offset(table, f"NEWX_{dom}"), "wnew": offset(table, "WNEW")}


def out_spec(ins, r):
    """(is_acc, first register, first quarter within it, quarter count) of a destination: the
    pieces are 256-bit quarters of consecutive bm registers, or consecutive w registers."""
    if ins.dst in ACC_PIECES:
        piece, n = ACC_PIECES[ins.dst]
        return True, 2 * r["d"] + piece // 2, piece % 2, n
    if ins.dst in W_PIECES:
        n0 = int(gen.W_DEST[r["wsel"]][0][1:])
        piece, n = W_PIECES[ins.dst]
        return False, 2 * n0 + piece, 0, n
    return None


def lanes_bytes(words):
    return b"".join(struct.pack("<I", w) for w in words)


def regs_of(d):
    return gen.Regs(s=tuple(d["s"]), d=d["d"], role=d["role"], wsel=d["wsel"])


def rust_bytes(b):
    return '"' + b.hex() + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--per-group", type=int, default=1)
    ap.add_argument("--extra", action="append", default=[],
                    help="KEY_A:KEY_B - also take the first colliding point of this pair even when its "
                         "outcome signature is already covered (reaches every deferred write path)")
    args = ap.parse_args()
    variants, goldens, states = [], [], {}
    seen = collections.Counter()
    for src in args.sources:
        d, tag, shift = src.split(":")
        shift = int(shift)
        d = os.path.abspath(d)
        table = open(os.path.join(d, "input.bin"), "rb").read()
        rows = an.classify(d, tag)
        _, hw = an.load_points(d, tag)
        for row in rows:
            if row["kind"] != "pair":
                continue
            if (rules.normalise(row["A"]), rules.normalise(row["B"])) == ("alone", "alone"):
                continue
            p, rec, dom = hw[row["id"]]
            a, b = gen.BY_KEY[p["a"]], gen.BY_KEY[p["b"]]
            sa, sb = out_spec(a, p["ra"]), out_spec(b, p["rb"])
            if sa is None or sb is None:
                continue
            sig = (shift, row["units"], row["cfg"], rules.normalise(row["A"]), rules.normalise(row["B"]))
            extra = (p["a"], p["b"], row["cfg"]) if f"{p['a']}:{p['b']}" in args.extra else None
            if seen[sig] >= args.per_group and (extra is None or seen[extra]):
                continue
            seen[sig] += 1
            if extra is not None:
                seen[extra] += 1
            key = (table, dom, shift)
            if key not in states:
                states[key] = len(variants)
                variants.append((os.path.basename(d), dom, variant_state(table, dom, shift)))
            ops = []
            for ins, r, spec, slot in ((a, p["ra"], sa, "A"), (b, p["rb"], sb, "B")):
                asm = gen.render(ins, regs_of(r))
                out = an.own_output(p, ins, r, rec, slot)
                ops.append((asm, assemble(asm), spec, out))
            goldens.append((os.path.basename(d), row["id"], row["units"], row["cfg"], states[key],
                            p["pa"], p["pb"], ops))
    with open(args.out, "w") as f:
        f.write("// Generated by tools/gen_collision_goldens.py from the collision probes' silicon\n"
                "// runs (build/experiments/collisions). Do not edit.\n\n")
        f.write("pub struct Variant {\n    pub name: &'static str,\n    pub acc: [[u64; 16]; 9],\n"
                "    pub vec: &'static [(u8, [u32; 8])],\n    pub q: &'static [(u8, [u32; 4])],\n"
                "    pub scalars: &'static [(u8, u32)],\n    pub table: &'static str,\n"
                "    pub newx: u32,\n    pub wnew: u32,\n}\n\n")
        f.write("pub struct Out {\n    pub asm: &'static str,\n    pub bytes: &'static str,\n"
                "    pub acc: bool,\n    pub reg: u8,\n    pub quarter: u8,\n    pub pieces: u8,\n"
                "    pub silicon: &'static str,\n}\n\n")
        f.write("pub struct Golden {\n    pub probe: &'static str,\n    pub id: u32,\n"
                "    pub units: &'static str,\n    pub cfg: &'static str,\n    pub variant: usize,\n"
                "    pub pa: u8,\n    pub pb: u8,\n    pub a: Out,\n    pub b: Out,\n}\n\n")
        f.write("pub const VARIANTS: &[Variant] = &[\n")
        for name, dom, st in variants:
            f.write(f"    Variant {{\n        name: \"{name}/{dom}\",\n        acc: [\n")
            for lanes in st["acc"]:
                f.write("            [" + ", ".join(f"0x{x:x}" for x in lanes) + "],\n")
            f.write("        ],\n        vec: &[\n")
            for reg, words in sorted(st["vec"].items()):
                f.write(f"            ({reg}, [" + ", ".join(f"0x{w:x}" for w in words) + "]),\n")
            f.write("        ],\n        q: &[\n")
            for reg, words in sorted(st["q"].items()):
                f.write(f"            ({reg}, [" + ", ".join(f"0x{w:x}" for w in words) + "]),\n")
            f.write("        ],\n        scalars: &[" +
                    ", ".join(f"({r}, {v})" for r, v in sorted(st["scalars"].items())) + "],\n")
            f.write(f"        table: {rust_bytes(st['table'])},\n        newx: {st['newx']},\n"
                    f"        wnew: {st['wnew']},\n    }},\n")
        f.write("];\n\npub const GOLDENS: &[Golden] = &[\n")
        for probe, pid, units, cfg, var, pa, pb, ops in goldens:
            f.write(f"    Golden {{\n        probe: \"{probe}\",\n        id: {pid},\n"
                    f"        units: \"{units}\",\n        cfg: \"{cfg}\",\n        variant: {var},\n"
                    f"        pa: {pa},\n        pb: {pb},\n")
            for name, (asm, bts, (is_acc, reg, piece, n), out) in zip("ab", ops):
                f.write(f"        {name}: Out {{\n            asm: \"{asm}\",\n")
                f.write(f"            bytes: {rust_bytes(bts)},\n            acc: {str(is_acc).lower()},\n"
                        f"            reg: {reg},\n            quarter: {piece},\n            pieces: {n},\n"
                        f"            silicon: {rust_bytes(out)},\n        }},\n")
            f.write("    },\n")
        f.write("];\n")
    print(f"{len(goldens)} goldens, {len(variants)} variants -> {args.out}")


if __name__ == "__main__":
    main()
