#!/usr/bin/env python3
"""Classify the loop-end (LE) bundle store of every fuzz seed's Peano core ELF.

For each build/fuzz/seed_*/aie.mlir.prj/main_core_0_2.elf:
  - disassemble with llvm-objdump --triple=aie2
  - locate the zero-overhead-loop end label (.L_LEnd0) and its bundle
  - if that bundle issues a partial-word store (st.s8 / st.s16), find the
    register it stores and walk backward to the instruction that produced
    that register (the data producer)
  - report the producer opcode, its latency class, and the bundle distance
    (number of issue slots between producer and the LE store)

This is the discriminator for BUG-B: on real silicon the LE-parked store is
flushed by the back-edge fetch redirect in some seeds and committed in others.
We tabulate the structural features here, then join against the real-HW
flush/commit labels to derive the exact rule. Pure observation tool -- no
emulator behavior is assumed.

Usage: tools/classify_le_store.py [build/fuzz]
"""
import os
import re
import subprocess
import sys

OBJDUMP = "/home/triple/npu-work/llvm-aie/install/bin/llvm-objdump"

# Mnemonics that do NOT write a scalar register destination (so they can't be
# the producer of a stored value). Stores write memory; control flow / nops
# write nothing relevant.
NON_PRODUCERS = {
    "nop", "nopa", "nopb", "nops", "nopm", "nopv", "nopx", "nopxm",
    "ret", "jl", "j", "b", "jnz", "jz", "done",
}

# Latency class for the producer opcode. Scalar multiply has 2-cycle result
# latency; loads are long (~E5, several cycles); most ALU/move ops are 1-cycle.
# (AM020 scalar pipeline; refine against aiesim only if the static rule is
# ambiguous.)
def latency_class(op):
    base = op.split(".")[0]  # strip .nc / .s8 style suffixes
    if base in ("mul", "mac", "macs", "msc"):
        return ("mul", 2)
    if base.startswith("ld") or base.startswith("lda") or base.startswith("vld"):
        return ("load", 6)
    return ("alu", 1)


# Bytes are hex pairs joined by single spaces; objdump then pads with spaces
# and/or a tab before the mnemonic. Match the byte pairs with a single-space
# separator (NOT \s) so the group never swallows the separating tab -- when the
# byte field is exactly column-width there is no padding and the last byte is
# followed directly by the tab.
LINE_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{2}(?: [0-9a-fA-F]{2})*)\s*\t(.*)$"
)


def is_store(op):
    base = op.split(".")[0]
    return base in ("st", "sta", "stb", "vst", "vsta")


def parse_func(lines, start_idx):
    """Return list of (addr, [ (op, dest, raw) ]) bundles until next label."""
    bundles = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        lbl = re.match(r"^[0-9a-fA-F]+ <([^>]+)>:", line)
        if lbl:
            # Local .L labels (.LBB, .L_LEnd0) are inside the function; only a
            # non-.L label marks the next real function -> stop there.
            if not lbl.group(1).startswith(".L"):
                break
            i += 1
            continue
        m = LINE_RE.match(line)
        if m:
            addr = int(m.group(1), 16)
            text = m.group(3)
            subs = []
            for piece in text.split(";"):
                piece = piece.strip()
                if not piece:
                    continue
                toks = piece.split()
                op = toks[0]
                dest = None
                if len(toks) > 1 and not is_store(op) and op not in NON_PRODUCERS:
                    dest = toks[1].rstrip(",")
                subs.append((op, dest, piece))
            bundles.append((addr, subs))
        i += 1
    return bundles


def store_reg(piece):
    """For 'st.s8 r2, [p1, dj0]' return the stored register 'r2'."""
    toks = piece.split()
    if len(toks) < 2:
        return None
    return toks[1].rstrip(",")


def classify(elf):
    try:
        out = subprocess.run(
            [OBJDUMP, "-d", "--triple=aie2", elf],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    lines = out.splitlines()

    # Find the .L_LEnd0 label and the function it belongs to.
    le_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^[0-9a-fA-F]+ <\.L_LEnd0>:", line):
            le_idx = i
            break
    if le_idx is None:
        return {"loop": False}

    # Find the function start (nearest preceding <name>: that's not a .L label
    # at the very top -- we want the bundle stream; parse from fuzz_kernel).
    func_start = None
    for i in range(le_idx, -1, -1):
        if re.match(r"^[0-9a-fA-F]+ <fuzz_kernel>:", lines[i]):
            func_start = i + 1
            break
    if func_start is None:
        # fall back: parse from just after the closest preceding label
        for i in range(le_idx, -1, -1):
            if re.match(r"^[0-9a-fA-F]+ <", lines[i]):
                func_start = i + 1
                break

    bundles = parse_func(lines, func_start)
    le_addr = int(re.match(r"^([0-9a-fA-F]+) <", lines[le_idx]).group(1), 16)

    # Recover the loop-start (LS) address from the prologue `movxm ls, #0x...`.
    ls_addr = None
    for (_a, subs) in bundles:
        for (op, dest, raw) in subs:
            if op.startswith("movxm") and dest == "ls":
                m = re.search(r"#(-?0x[0-9a-fA-F]+)", raw)
                if m:
                    ls_addr = int(m.group(1), 16)
        if ls_addr is not None:
            break

    # Locate the LE bundle in the parsed stream.
    le_pos = next((k for k, (a, _) in enumerate(bundles) if a == le_addr), None)
    if le_pos is None:
        return {"loop": True, "le_store": None}

    _, le_subs = bundles[le_pos]
    store_piece = next(
        (raw for (op, _d, raw) in le_subs if is_store(op) and "." in op), None
    )
    if store_piece is None:
        # full-word store or no store in LE bundle
        full = next((raw for (op, _d, raw) in le_subs if is_store(op)), None)
        return {"loop": True, "le_store": "full" if full else None}

    sreg = store_reg(store_piece)
    store_op = store_piece.split()[0]

    # Walk backward for the producer of sreg.
    prod_op = None
    prod_dist = None
    for k in range(le_pos - 1, -1, -1):
        _, subs = bundles[k]
        for (op, dest, _raw) in subs:
            if dest == sreg:
                prod_op = op
                prod_dist = le_pos - k
                break
        if prod_op:
            break

    # Loop body geometry: bundle count and byte span from LS to LE.
    ls_pos = next((k for k, (a, _) in enumerate(bundles) if a == ls_addr), None) \
        if ls_addr is not None else None
    body_bundles = (le_pos - ls_pos + 1) if ls_pos is not None else None
    body_bytes = (le_addr - ls_addr) if ls_addr is not None else None

    lat = latency_class(prod_op) if prod_op else ("?", None)
    return {
        "loop": True,
        "le_store": store_op,
        "sreg": sreg,
        "prod_op": prod_op,
        "prod_dist": prod_dist,
        "lat_class": lat[0],
        "lat": lat[1],
        "body_bundles": body_bundles,
        "body_bytes": body_bytes,
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "build/fuzz"
    seeds = sorted(
        d for d in os.listdir(root)
        if d.startswith("seed_")
        and os.path.exists(os.path.join(root, d, "aie.mlir.prj", "main_core_0_2.elf"))
    )
    rows = []
    for s in seeds:
        elf = os.path.join(root, s, "aie.mlir.prj", "main_core_0_2.elf")
        has_hw = os.path.exists(os.path.join(root, s, "npu_output.bin"))
        info = classify(elf)
        if not info or not info.get("loop"):
            continue
        rows.append((s, has_hw, info))

    # Only the seeds with a partial-word store in the LE bundle are interesting.
    part = [(s, h, i) for (s, h, i) in rows if i.get("le_store") and i["le_store"] != "full"]
    print(f"seeds with core ELF + ZOL: {len(rows)}")
    print(f"seeds with PARTIAL-WORD store in LE bundle: {len(part)}")
    print()
    print(f"{'seed':<13}{'hw?':<4}{'st':<7}{'sreg':<5}{'producer':<9}{'lat':<5}"
          f"{'dist':<5}{'body_b':<7}{'bytes':<6}")
    for s, h, i in part:
        print(f"{s:<13}{'Y' if h else '-':<4}{i['le_store']:<7}{i['sreg']:<5}"
              f"{str(i['prod_op']):<9}{i['lat_class']:<5}{str(i['prod_dist']):<5}"
              f"{str(i['body_bundles']):<7}{str(i['body_bytes']):<6}")


if __name__ == "__main__":
    main()
