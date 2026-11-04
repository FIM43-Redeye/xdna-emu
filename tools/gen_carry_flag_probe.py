#!/usr/bin/env python3
"""Generate the carry-flag probe suite: which scalar instructions set srCarry, to what, and
how adc / sbc consume it.

TableGen (`AIE2GenInstrInfo.td`) lists `Defs = [srCarry]` on `add` (register and immediate),
`sub`, `adc`, `sbc`, `abs` and `jnzd`, and `Uses = [srCarry]` on `adc` and `sbc`; the
itinerary puts the def at operand cycle 1. Each record writes srCarry, runs one instruction
on two operands from the input, and stores the result and the flag as read in the same
bundle and 1..4 bundles later. The operands cross edge values (0, 1, the signed and unsigned
extremes) from either initial carry. A second group chains a flag producer into adc / sbc 1
and 2 bundles later (carry-in timing); a third checks that instructions TableGen does not
list (mul, logic, shifts, compares, moves, `add.nc`, `padda`) leave the flag alone.

The kernels are unrolled assembly (no loop counter: `add` itself sets the flag).

  python3 tools/gen_carry_flag_probe.py --suite-dir <dir>
  python3 tools/gen_carry_flag_probe.py --analyze <dir> <tag>
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import suite as S  # noqa: E402

GENERATOR = "tools/gen_carry_flag_probe.py"
REC_IN = 16                     # a, b, initial carry, pad
REC_OUT = 32                    # result, flag at offsets 0..4, pad
READS = 5
RECORDS_PER_CASE = 96           # program memory: ~100 bytes of code per record

EDGES = (0, 1, 2, 0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, 0xFFFF_FFFE, 0xFFFF_FFFF, 0x1234_5678,
         0xEDCB_A988)


@dataclass(frozen=True)
class Op:
    name: str
    asm: str                    # `{d}` result, `{a}` `{b}` operands
    binary: bool = True
    pairs: bool = True          # shares a bundle with `mov rX, srCarry` (not an mv-slot op)


# The TableGen definers, then instructions it does not list (expected to keep the flag).
DEFINERS = (
    Op("add", "add {d}, {a}, {b}"),
    Op("sub", "sub {d}, {a}, {b}"),
    Op("adc", "adc {d}, {a}, {b}"),
    Op("sbc", "sbc {d}, {a}, {b}"),
    Op("abs", "abs {d}, {a}", False),
) + tuple(Op(f"add_imm{i}", f"add {{d}}, {{a}}, #{i}", False) for i in (1, -1, 63, -64))
OTHERS = (
    Op("mul", "mul {d}, {a}, {b}"),
    Op("or", "or {d}, {a}, {b}"),
    Op("and", "and {d}, {a}, {b}"),
    Op("xor", "xor {d}, {a}, {b}"),
    Op("lshl", "lshl {d}, {a}, {b}"),
    Op("ashl", "ashl {d}, {a}, {b}"),
    Op("ltu", "ltu {d}, {a}, {b}"),
    Op("lt", "lt {d}, {a}, {b}"),
    Op("eqz", "eqz {d}, {a}", False),
    Op("add_nc", "add.nc {d}, {a}, #-1", False, False),
)
# jnzd decrements its source into its destination and branches if the source was nonzero.
JNZD = Op("jnzd", "jnzd {d}, {a}, p5", False)


@dataclass(frozen=True)
class Record:
    op: str
    a: int
    b: int
    init: int
    chain: str = ""             # "": plain; "<producer>@<d>": producer, then `op` d bundles later

    def as_list(self) -> list:
        return [self.op, self.a, self.b, self.init, self.chain]


def records() -> list[Record]:
    recs = []
    for op in DEFINERS:
        bs = EDGES if op.binary else (0,)
        recs += [Record(op.name, a, b, init) for a in EDGES for b in bs for init in (0, 1)]
    recs += [Record(JNZD.name, a, 0, init) for a in EDGES for init in (0, 1)]
    for op in OTHERS:
        recs += [Record(op.name, a, b, init) for a in (0, 0x8000_0000, 0xFFFF_FFFF)
                 for b in (1, 0xFFFF_FFFF) for init in (0, 1)]
    # Carry-in timing: add (carry out of 0xffffffff + 1 = 1) or sub, then adc / sbc d later.
    for prod in ("add", "sub"):
        for cons in ("adc", "sbc"):
            for d in (1, 2, 3):
                for a in (0xFFFF_FFFF, 0, 5):
                    recs += [Record(cons, a, 1, init, f"{prod}@{d}") for init in (0, 1)]
    return recs


OPS = {op.name: op for op in DEFINERS + OTHERS + (JNZD,)}


def nops(n: int) -> list[str]:
    return ["  nop"] * n


def record_lines(rec: Record, k: int) -> list[str]:
    """r2 = a, r3 = b, r6 = initial carry; result in r4; flag reads in r8..r12."""
    op = OPS[rec.op]
    lines = [f"  // ---- record {k}: {rec.op} {rec.chain} ----",
             "  lda r2, [p0, #0]", "  lda r3, [p0, #4]", "  lda r6, [p0, #8]"] + nops(7)
    lines += ["  mov srCarry, r6"] + nops(3)
    body = op.asm.format(d="r4", a="r2", b="r3")
    if rec.chain:
        prod, d = rec.chain.split("@")
        # The producer computes a+b (or a-b) into r7; its carry feeds the consumer d later.
        lines.append("  " + OPS[prod].asm.format(d="r7", a="r2", b="r3"))
        lines += nops(int(d) - 1)
    if rec.op == "jnzd":
        label = f".Ljnzd{k}"
        lines += [f"  movxm p5, #{label}"] + nops(3)
        lines += [f"  {body} ; mov r8, srCarry", "  mov r9, srCarry", "  mov r10, srCarry",
                  "  mov r11, srCarry", "  mov r12, srCarry"]
        lines += [".p2align 4", f"{label}:"]
    elif op.pairs:
        lines += [f"  {body} ; mov r8, srCarry", "  mov r9, srCarry", "  mov r10, srCarry",
                  "  mov r11, srCarry", "  mov r12, srCarry"]
    else:
        # Reads at offsets 1..5 instead of 0..4.
        lines += [f"  {body}", "  mov r8, srCarry", "  mov r9, srCarry", "  mov r10, srCarry",
                  "  mov r11, srCarry", "  mov r12, srCarry"]
    lines += nops(2)
    lines += ["  st r4, [p1, #0]"] + [f"  st r{8 + i}, [p1, #{4 + 4 * i}]" for i in range(READS)]
    lines += nops(1) + [f"  padda [p0], #{REC_IN}", f"  padda [p1], #{REC_OUT}"]
    return lines


def kernel_text(recs: list[Record]) -> str:
    lines = [f"// Auto-generated ({GENERATOR}): {len(recs)} records", ".text", ".globl test_kernel",
             "test_kernel:", "  mov r14, srCarry"]
    for k, r in enumerate(recs):
        lines += record_lines(r, k)
    lines += nops(4) + ["  mov srCarry, r14"] + nops(2) + ["  ret lr"] + nops(5)
    return "\n".join(lines) + "\n"


def record_bytes(r: Record) -> bytes:
    return struct.pack("<4I", r.a, r.b, r.init, 0)


def generate(suite_dir: str | Path) -> list[str]:
    recs = records()
    names = []
    for k, start in enumerate(range(0, len(recs), RECORDS_PER_CASE)):
        chunk = recs[start:start + RECORDS_PER_CASE]
        name = f"carry__{k:02d}"
        S.write_single_core_case(suite_dir, name, kernel_text(chunk),
                                 input_bytes=b"".join(record_bytes(r) for r in chunk),
                                 out_bytes=REC_OUT * len(chunk),
                                 meta={"records": [r.as_list() for r in chunk], "rec_in": REC_IN,
                                       "rec_out": REC_OUT})
        names.append(name)
    S.write_suite_json(suite_dir, "carry", GENERATOR, names, hw_checks=["aligned_labels"])
    return names


# -- analysis ---------------------------------------------------------------------------------

M32 = 0xFFFF_FFFF


def _s7(imm: int) -> int:
    return imm & M32


def model(op: str, a: int, b: int, cin: int) -> tuple[int, int | None]:
    """(result, carry) on silicon; carry None = the instruction leaves srCarry alone.

    The add family sets the carry out of the 32-bit unsigned sum (an immediate is
    sign-extended; jnzd adds -1, so its carry is "the source was nonzero"); the subtract family
    sets the borrow (1 when the subtrahend plus borrow-in exceeds the minuend), and sbc
    subtracts the flag as borrow-in; abs sets the source's sign bit."""
    if op == "add":
        s = a + b
    elif op.startswith("add_imm"):
        s = a + _s7(int(op[7:]))
    elif op == "jnzd":
        s = a + M32
    elif op == "adc":
        s = a + b + cin
    elif op == "sub":
        return (a - b) & M32, int(a < b)
    elif op == "sbc":
        return (a - b - cin) & M32, int(a < b + cin)
    elif op == "abs":
        v = a - (1 << 32) if a >> 31 else a
        return abs(v) & M32, a >> 31
    else:
        return 0, None
    return s & M32, s >> 32


def analyze(suite_dir: str, tag: str, verbose: bool = False) -> int:
    """Check every record's flag reads (and the definers' results) against `model`."""
    suite = S.load_suite(suite_dir)
    total = bad = 0
    for case in suite.cases:
        meta = suite.case_meta(case)
        out = suite.output(tag, case).read_bytes()
        for i, r in enumerate(meta["records"]):
            op, a, b, init, chain = r
            res, *reads = struct.unpack_from("<6I", out, i * REC_OUT)
            cin = init
            if chain:
                prod, _ = chain.split("@")
                cin = model(prod, a, b, init)[1]
            want_res, c = model(op, a, b, cin)
            flag = cin if c is None else c
            # Same-bundle read sees the old flag; from the next bundle on the new one. An mv-slot
            # op (no same-bundle read) is read from offset 1.
            first = init if not chain else cin
            pairs = OPS[op].pairs
            want = ([first] if pairs else []) + [flag] * (READS - (1 if pairs else 0))
            ok = reads == want and (c is None or res == want_res)
            total += 1
            if not ok or verbose:
                bad += not ok
                print(f"{case} {r}: res {res:08x} (model {want_res:08x}) flags {reads} (model {want})")
    print(f"{total} records, {total - bad} match the model, {bad} differ")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite-dir")
    ap.add_argument("--analyze", nargs=2, metavar=("DIR", "TAG"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        sys.exit(1 if analyze(*args.analyze, verbose=args.verbose) else 0)
    if not args.suite_dir:
        ap.error("--suite-dir or --analyze")
    names = generate(args.suite_dir)
    print(f"{len(names)} cases -> {args.suite_dir}")


if __name__ == "__main__":
    main()
