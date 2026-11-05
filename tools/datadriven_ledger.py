#!/usr/bin/env python3
"""Data-driven ledger: where each AIE2 instruction's semantics come from.

For every encoding the decoder can produce (the build-time resolved encodings),
the ledger records which semantic layer assigned its SemanticOp, which rule
produced each fact the executor relies on, whether its executor is generic, and
whether runtime code keys on its name. It then computes the metrics defined in
METRICS below.

Inputs:
  * `$OUT_DIR/semantic_provenance.json`, written by the xdna-archspec build
    script (`build_helpers/codegen.rs::generate_semantic_provenance`). The
    script builds the crate to get it (or takes `--provenance PATH`).
  * A static scan of `src/interpreter/` for string literals that name an
    instruction or mnemonic (runtime name keys).
  * EXECUTOR_KIND: the executor classification per SemanticOp. This table is
    audit judgement (2026-11-04 data-driven audit), not measurement; update it
    when a package changes an executor.

Usage:
  tools/datadriven_ledger.py [--provenance FILE] [--out DIR]
  tools/datadriven_ledger.py --diff OLD_DIR NEW_DIR   # behaviour gate

Facts M1-M8 count what the interpreter consumes. The `schema` object of each provenance row
is the TableGen encoding schema (PIPE package); M9 reports its coverage and agreement with the
consumed values. A package that switches a consumer to a schema field also switches that
fact's rule in `fact_categories` to read the schema (and deletes the legacy field), so M5 moves
with it.

`--out` defaults to `build/experiments/harness/datadriven/ledger-<branch>/` in
the main checkout. `--diff` compares the final per-instruction values (never the
provenance tags, which a migration is meant to change) and the generated
tables copied beside each ledger; exit status 1 on any difference.
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Source categories. Every fact source maps to one of these.
#   tablegen        read from a TableGen record field, flag, DAG, or class lineage
#   tablegen+table  a TableGen relation (Pat<>, C++ selection) mapped through a
#                   hand table keyed on a TableGen name (intrinsic stem -> op)
#   name-parse      parsed by a hand rule out of a TableGen name/mnemonic string
#   hand            a hand table keyed on instruction identity, or a constant
#   n/a             the fact does not apply to this instruction
# ---------------------------------------------------------------------------
TABLEGEN, TG_TABLE, NAME_PARSE, HAND, NA = "tablegen", "tablegen+table", "name-parse", "hand", "n/a"

LAYER_CATEGORY = {
    "L0_move_flag": TABLEGEN,
    "L0_delay_slot_flags": TABLEGEN,
    "L0_mayload_maystore": TABLEGEN,
    "L0_done_parent_class": TABLEGEN,
    "L0_nop_mnemonic": NAME_PARSE,
    "L0_branch_mnemonic": NAME_PARSE,
    "L0_name_override_early": HAND,
    "L1_pat_sdnode": TABLEGEN,
    "L2_pseudo_sdnode": TABLEGEN,
    "L5a_cascade_implicit": TABLEGEN,
    "L1_pat_intrinsic": TG_TABLE,
    "L2_pseudo_intrinsic": TG_TABLE,
    "L3_cpp_switch": TG_TABLE,
    "L4_itinerary_table": HAND,
    "L5b_name_override": HAND,
}

OPERAND_TYPE_CATEGORY = {
    # The class name is TableGen's, but the class -> register-file mapping is a
    # hand match on that name (exact, prefix or composite table).
    "regclass_exact": NAME_PARSE,
    "regclass_prefix": NAME_PARSE,
    "composite_table": NAME_PARSE,
    "imm_classname": NAME_PARSE,
    "lockid_fieldname": HAND,
    "fieldname_fallback": HAND,
    "unknown": HAND,
    "unbound": HAND,
}

FACT_CATEGORY = {
    "tablegen_flags": TABLEGEN,
    "tablegen_defs_uses": TABLEGEN,
    "tablegen_itinerary": TABLEGEN,
    "tablegen_ag_class": TABLEGEN,
    "mnemonic_suffix": NAME_PARSE,
    "name_tokens": NAME_PARSE,
    "mnemonic_substring": NAME_PARSE,
    "name_substring": NAME_PARSE,
    "mnemonic_prefix": NAME_PARSE,
    "name_digits": NAME_PARSE,
    "tablegen_hw_encoding": TABLEGEN,
    "none": NA,
    "n/a": NA,
}

# Executor shape per SemanticOp (audit judgement; see module docstring).
#   generic      one body per SemanticOp, parameterised by decoded/TableGen data
#   role-scan    generic body, but operand roles found by scanning operand kinds
#                or by hard-coded register numbers (r27, r29, r31, "last scalar")
#   name-branch  generic body with a branch on the mnemonic string
#   type-match   a hand match over (from, to) element-type pairs
GENERIC, ROLE_SCAN, NAME_BRANCH, TYPE_MATCH = "generic", "role-scan", "name-branch", "type-match"
EXECUTOR_KIND = collections.defaultdict(lambda: GENERIC)
EXECUTOR_KIND.update({
    # MAC family: op kind from encoding_name stem / `contains("msc")`
    # (vector_matmul/mod.rs, sparse.rs::op_kind_of, archspec matmul.rs from_stem).
    **{op: NAME_BRANCH for op in ("Mac", "MatMul", "MatMulSub", "NegMul", "NegMatMul", "AddMac", "SubMac")},
    # Accumulator add/sub: NegAdd split by `contains("negsub")` (vector_arith.rs).
    "NegAdd": NAME_BRANCH, "AccumNegAdd": NAME_BRANCH, "AccumNegSub": NAME_BRANCH,
    # VBCSTSHFL detected by `contains("SHFL")` (vector_misc.rs) and remapped in slot_builder.rs.
    "VectorBroadcast": NAME_BRANCH,
    # Convert: hand match on (from, to); VFLOOR recognised by type pattern (vector_convert.rs).
    "Convert": TYPE_MATCH,
    # Operand roles by kind scan or fixed registers.
    **{op: ROLE_SCAN for op in ("VectorInsert", "VectorExtract", "VectorPush", "VectorPushHi", "Align",
                                "VectorSelect", "VectorClear", "Select", "DivStep", "SDiv")},
    # Stream ops: blocking / TLAST from mnemonic pieces `.nb` / `.tlast` (slot_builder.rs).
    "StreamRead": NAME_BRANCH, "StreamWrite": NAME_BRANCH, "StreamWritePacketHeader": NAME_BRANCH,
})

# Operand-role conventions that are positional hand code rather than TableGen operand names.
ROLE_SCAN_OPS = {op for op, k in EXECUTOR_KIND.items() if k == ROLE_SCAN} | {
    "Mac", "MatMul", "MatMulSub", "NegMul", "NegMatMul", "AddMac", "SubMac",
    "Accumulate", "AccumSub", "AccumNegAdd", "AccumNegSub", "Srs", "Ups", "VectorBroadcast",
}

# Ops whose timing uses a hand constant even though the itinerary states it.
HAND_LATENCY_OPS = {"StreamRead", "StreamWrite", "StreamWritePacketHeader", "CascadeRead"}

FACTS = ["semantic", "operand_roles", "register_classes", "element_type", "immediates",
         "implicit_regs", "latency_port", "side_effects", "mem_width", "addressing_mode"]

RUNTIME_SCAN_DIRS = ["src/interpreter"]
SKIP_FILE = re.compile(r"(_goldens?|_golden_data|_data|goldens)\.rs$|/tests?\.rs$")
NAME_CONTEXT = re.compile(r"encoding_name|mnemonic|\bname\b|enc_lower|\bstem\b|\bmn\b|\blower\b|\binstr\b|\|n\|")
# Files whose name-keyed literals are not instruction semantics: crossref.rs parses objdump text
# for diagnostics, register_widths.rs keys on register names.
SKIP_SCAN = {"src/interpreter/decode/crossref.rs", "src/interpreter/state/register_widths.rs",
             "src/interpreter/decode/register_map.rs"}
STRING_LIT = re.compile(r'"([A-Za-z][A-Za-z0-9_.]{1,40})"')
SUBSTRING_CALL = re.compile(r'\.(starts_with|ends_with|contains)\(\s*"([^"]+)"')


def repo_root():
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def main_checkout():
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          text=True).strip())
    return common.parent


def build_provenance(root):
    """Build xdna-archspec and return its OUT_DIR (holding semantic_provenance.json)."""
    proc = subprocess.run(["cargo", "build", "-p", "xdna-archspec", "--message-format=json"],
                          cwd=root, stdout=subprocess.PIPE, text=True)
    out_dir = None
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "build-script-executed" and "xdna-archspec" in msg.get("package_id", ""):
            out_dir = Path(msg["out_dir"])
    if proc.returncode != 0 or out_dir is None:
        sys.exit("cargo build -p xdna-archspec failed or reported no OUT_DIR")
    return out_dir


# ---------------------------------------------------------------------------
# Runtime name-key scan
# ---------------------------------------------------------------------------

TEST_MODULE = re.compile(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{")


def strip_tests(text):
    """Drop everything from the first `#[cfg(test)] mod ... {` on (test modules sit at the end)."""
    m = TEST_MODULE.search(text)
    return text if m is None else text[:m.start()]


def scan_runtime(root):
    """Return [(file, line, kind, literal)] for literals used as instruction-name keys."""
    hits = []
    for d in RUNTIME_SCAN_DIRS:
        for path in sorted((root / d).rglob("*.rs")):
            rel = path.relative_to(root).as_posix()
            if SKIP_FILE.search(rel) or rel in SKIP_SCAN:
                continue
            for n, line in enumerate(strip_tests(path.read_text()).splitlines(), 1):
                code = line.split("//")[0]
                if "log::" in code or "format!" in code or "panic!" in code or not NAME_CONTEXT.search(code):
                    continue
                subs = {lit for _, lit in SUBSTRING_CALL.findall(code)}
                for _, lit in SUBSTRING_CALL.findall(code):
                    hits.append((rel, n, "substring", lit))
                for lit in STRING_LIT.findall(code):
                    if lit not in subs:
                        hits.append((rel, n, "exact", lit))
    return hits


def match_hits(encodings, hits):
    """Map each encoding name to the runtime hits that key on it."""
    by_enc = collections.defaultdict(list)
    for rel, n, kind, lit in hits:
        low = lit.lower()
        for e in encodings:
            name, mn = e["name"].lower(), e["mnemonic"].lower()
            stem = mn.split(".")[0]
            if kind == "exact":
                # Exact name or mnemonic, or the op stem with or without the leading `v`
                # (`matches!(instr, "msc" | "addmac")` in the MAC code).
                ok = low in (name, mn, stem) or (stem.startswith("v") and low == stem[1:])
            else:
                # Two-letter tokens ("hi", "lo", "nb") are substrings of too many names to attribute.
                ok = len(low) >= 3 and (low in name or low in mn)
            if ok:
                by_enc[e["name"]].append(f"{rel}:{n}:{lit}")
    return by_enc


# ---------------------------------------------------------------------------
# Per-instruction classification
# ---------------------------------------------------------------------------

def short(op):
    return op.split("::")[-1] if op else None


def semantic_info(e):
    trail = e["semantic_trail"]
    if not trail:
        return "none", NA, None
    layer = trail[-1][0]
    cat = LAYER_CATEGORY[layer]
    note = None
    if cat == HAND:
        final = trail[-1][1]
        earlier = [(l, op) for l, op in trail[:-1] if LAYER_CATEGORY[l] in (TABLEGEN, TG_TABLE)]
        if not earlier:
            note = "sole"           # nothing derived proposed anything
        elif earlier[-1][1] == final:
            note = "redundant"      # the derived layers already said the same
        elif earlier[-1][0].startswith("L0_"):
            note = "refines-flags"  # overrides only the mayLoad/mayStore Load/Store guess
        else:
            note = "corrective"     # overrides a different Pat<>/C++-derived proposal
    return layer, cat, note


def fact_categories(e, name_keys):
    op = short(e["semantic"])
    fs = e["fact_sources"]
    layer, sem_cat, _ = semantic_info(e)
    facts = {"semantic": sem_cat}

    mem = e["may_load"] or e["may_store"] or e["is_ptr_arithmetic"]
    if mem:
        facts["operand_roles"] = HAND       # operand_extraction.rs positional conventions
    elif op in ROLE_SCAN_OPS:
        facts["operand_roles"] = HAND       # executor scans operand kinds / fixed regs
    else:
        facts["operand_roles"] = TABLEGEN   # MC operand order == TableGen ins/outs order

    worst = TABLEGEN
    rank = {TABLEGEN: 0, TG_TABLE: 1, NAME_PARSE: 2, HAND: 3}
    for f in e["operand_fields"]:
        c = OPERAND_TYPE_CATEGORY.get(f["type_source"], HAND)
        if rank[c] > rank[worst]:
            worst = c
    facts["register_classes"] = worst if e["operand_fields"] else NA

    facts["element_type"] = FACT_CATEGORY[fs["element_type"]]
    has_imm = any(f["type_source"] == "imm_classname" or "Immediate" in f["operand_type"]
                  for f in e["operand_fields"])
    # Runtime immediates come from the LLVM disassembler (decodeSImmOperand* etc.).
    facts["immediates"] = TABLEGEN if has_imm else NA

    if fs["implicit_regs"] == "none":
        facts["implicit_regs"] = NA
    elif op in ("Select", "DivStep", "SDiv", "VectorInsert", "VectorExtract"):
        facts["implicit_regs"] = HAND       # executor hard-codes r27 / r29 / r31
    else:
        facts["implicit_regs"] = TABLEGEN

    if fs["itinerary"] == "none":
        facts["latency_port"] = HAND        # timing_from_semantic fallback tier
    elif e["may_load"] or op in HAND_LATENCY_OPS or (op == "Mul" and not e["is_vector"]):
        facts["latency_port"] = HAND        # LATENCY_MEMORY=7, SS_* constants, MUL=2
    else:
        facts["latency_port"] = TABLEGEN

    facts["side_effects"] = TABLEGEN
    facts["mem_width"] = FACT_CATEGORY[fs["mem_width"]]
    facts["addressing_mode"] = FACT_CATEGORY[fs["addressing_mode"]] if mem else NA
    return facts


def executor_info(e, name_keys):
    op = short(e["semantic"])
    kind = EXECUTOR_KIND[op] if op else "none"
    if name_keys.get(e["name"]) and kind == GENERIC:
        kind = NAME_BRANCH
    return kind


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRICS = """\
Definitions (all over the N resolved encodings the decoder can produce):

- M1 semantic coverage: encodings with any SemanticOp / N.
- M2 semantic from TableGen: encodings whose winning semantic layer reads a TableGen
  record field, flag or generic SDNode (L0 move/delay-slot/mayLoad/mayStore/done
  flags, L1/L2 Pat<> with an SDNode operator, L5a cascade implicit Uses) / N.
- M3 semantic via TableGen relation + hand stem table: winning layer is a Pat<> or C++
  selection whose intrinsic was mapped through `semantic_from_intrinsic` / N.
- M4 semantic decided by hand: winning layer is ITINERARY_SEMANTICS, the nop/branch
  mnemonic rules, or `refine_encoding_semantic` / N; the hand layer is split into
  redundant (a derived layer already proposed the same op), refines-flags (overrides
  only the mayLoad/mayStore Load/Store guess, e.g. fused vst.srs), corrective
  (overrides a different Pat<>/C++-derived op) and sole (no derived proposal at all).
- M5 per fact: share of applicable encodings whose fact is `tablegen` (n/a excluded).
- M6 generic executor: encodings whose SemanticOp executor is `generic` (EXECUTOR_KIND)
  and that no runtime code keys on by name / N.
- M7 fully data-driven: encodings with every applicable fact `tablegen` or
  `tablegen+table` AND a generic executor / N. This is the number that replaces the
  undated "~33% data-driven".
- M7b no hand facts: like M7, but a fact parsed from a TableGen name (`name-parse`) also
  counts; only `hand` facts and non-generic executors disqualify.
- M8 runtime name keys: string literals in `src/interpreter/` (tests and goldens
  excluded) that name an instruction or mnemonic, and the encodings they touch.
- M9 schema availability: for the facts the TableGen encoding schema
  (`InstrEncoding::schema`, PIPE package) provides, the share of applicable encodings it
  covers, and how many agree with the legacy value the interpreter still consumes. M1-M8
  count what is consumed, so they move only when a consumer switches to the schema; M9 is
  what is ready to switch to. Every disagreement is listed in `schema_disagreements.tsv`.
"""


def pct(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else "0/0"


def compute(encodings, name_keys, hits):
    n = len(encodings)
    rows = []
    for e in encodings:
        layer, cat, note = semantic_info(e)
        facts = fact_categories(e, name_keys)
        ex = executor_info(e, name_keys)
        rows.append({
            "name": e["name"], "mnemonic": e["mnemonic"], "slot": e["slot"],
            "semantic": short(e["semantic"]), "semantic_layer": layer, "semantic_override": note or "",
            "semantic_trail": " > ".join(f"{l}:{short(o)}" for l, o in e["semantic_trail"]),
            **{f"fact_{k}": v for k, v in facts.items()},
            "executor": ex,
            "runtime_name_keys": ";".join(name_keys.get(e["name"], [])),
        })
    m = {}
    m["N"] = n
    m["M1"] = pct(sum(r["semantic"] is not None for r in rows), n)
    by_cat = collections.Counter(r["fact_semantic"] for r in rows)
    m["M2"] = pct(by_cat[TABLEGEN], n)
    m["M3"] = pct(by_cat[TG_TABLE], n)
    m["M4"] = pct(by_cat[HAND] + by_cat[NAME_PARSE], n)
    m["M4_split"] = dict(collections.Counter(r["semantic_override"] for r in rows if r["fact_semantic"] == HAND))
    m["M4_name_parse"] = by_cat[NAME_PARSE]
    m["M5"] = {}
    for f in FACTS:
        vals = [r[f"fact_{f}"] for r in rows if r[f"fact_{f}"] != NA]
        m["M5"][f] = {"tablegen": pct(sum(v == TABLEGEN for v in vals), len(vals)),
                      "mix": dict(collections.Counter(vals))}
    m["M6"] = pct(sum(r["executor"] == GENERIC for r in rows), n)
    m["M6_mix"] = dict(collections.Counter(r["executor"] for r in rows))
    full = 0
    for r in rows:
        ok = all(r[f"fact_{f}"] in (TABLEGEN, TG_TABLE, NA) for f in FACTS) and r["executor"] == GENERIC
        r["fully_data_driven"] = ok
        full += ok
    m["M7"] = pct(full, n)
    no_hand = 0
    for r in rows:
        ok = all(r[f"fact_{f}"] != HAND for f in FACTS) and r["executor"] == GENERIC
        r["no_hand_facts"] = ok
        no_hand += ok
    m["M7b"] = pct(no_hand, n)
    m["M8_literals"] = len(hits)
    m["M8_encodings"] = sum(1 for r in rows if r["runtime_name_keys"])
    m["layers"] = dict(collections.Counter(r["semantic_layer"] for r in rows))
    return rows, m


# ---------------------------------------------------------------------------
# Schema availability (M9): the TableGen schema beside the legacy values
# ---------------------------------------------------------------------------

# Legacy `RegisterKind` -> the schema register families it stands for.
LEGACY_KIND_FAMILIES = {
    "Scalar": {"Scalar"}, "Pointer": {"Bits20"}, "ModifierM": {"Bits20"}, "ModifierDN": {"Bits20"},
    "ModifierDJ": {"Bits20"}, "ModifierDC": {"Bits20"}, "Vector256": {"Vector256"},
    "Vector512": {"Vector512"}, "Vector1024": {"Vector1024"}, "SparseQx": {"Vector640"},
    "Accumulator": {"Acc256", "Acc512", "Acc1024"}, "Control": {"Other"}, "ScalarPair": {"Other"},
}
LEGACY_ADDRESSING = {"Indexed": "IndexedRegister", "IndexedImm": "IndexedImmediate",
                     "PostModify": "PostModifyRegister", "PostModifyImm": "PostModifyImmediate",
                     "TwoDim": "TwoDim", "ThreeDim": "ThreeDim", None: "Unknown"}
LEGACY_SUBWORD = {"Byte": 8, "HalfWord": 16}
REG_TYPE = re.compile(r"OperandType::(?:Register\(RegisterKind::(\w+)\)|RegisterWithOffset\(RegisterKind::(\w+), \d+\))")
IMM_TYPE = re.compile(r"OperandType::Immediate \{ signed: (\w+), scale: (-?\d+) \}")


def operand_type_agrees(legacy, kind):
    """Whether a legacy `OperandType` expression says what the schema operand kind says."""
    reg, imm = kind.get("register"), kind.get("immediate")
    m = REG_TYPE.fullmatch(legacy)
    if m:
        return reg is not None and reg["encoder"] is None and reg["family"] in LEGACY_KIND_FAMILIES[m.group(1) or m.group(2)]
    if legacy.startswith("OperandType::CompositeRegister("):
        return reg is not None and reg["encoder"] is not None
    m = IMM_TYPE.fullmatch(legacy)
    if m:
        return (imm is not None and (m.group(1) == "true") == imm["signed"] and int(m.group(2)) == imm["step"]
                and not imm["negative"])
    if legacy == "OperandType::LockId":
        return imm is not None and not imm["signed"]
    return False


def schema_report(encodings):
    """M9 counts and the disagreement rows (name, fact, item, legacy, schema)."""
    counts = collections.defaultdict(lambda: [0, 0, 0])  # fact -> [applicable, provided, agree]
    dis = []
    for e in encodings:
        sc = e["schema"]
        ops = {o["name"]: o for o in sc["operands"]}
        for f in e["operand_fields"]:
            c = counts["register_classes"]
            c[0] += 1
            o = ops.get(f["name"])
            if o is None:
                continue  # an encoding field with no DAG operand: the schema says nothing about it
            c[1] += 1
            if operand_type_agrees(f["operand_type"], o["kind"]):
                c[2] += 1
            else:
                dis.append((e["name"], "register_classes", f["name"], f["operand_type"], json.dumps(o["kind"], sort_keys=True)))
        c = counts["operand_roles"]
        c[0] += 1
        c[1] += 1
        c[2] += 1  # the schema operand table is new: nothing legacy to agree with
        mem = e["may_load"] or e["may_store"]
        if mem or sc["addressing"]:
            c = counts["addressing_mode"]
            c[0] += 1
            if sc["addressing"]:
                c[1] += 1
            want = LEGACY_ADDRESSING.get(sc["addressing"], "Unknown")
            if e["addressing_mode"].split("::")[-1] == want:
                c[2] += bool(sc["addressing"])
            else:
                dis.append((e["name"], "addressing_mode", "", e["addressing_mode"], str(sc["addressing"])))
        if mem:
            c = counts["mem_width"]
            c[0] += 1
            legacy = LEGACY_SUBWORD.get(e["mem_width"].split("::")[-1])
            if sc["mem_vt_bits"] is not None:
                c[1] += 1
            if legacy == sc["mem_vt_bits"]:
                c[2] += 1 if sc["mem_vt_bits"] is not None else 0
            else:
                dis.append((e["name"], "mem_width", "", e["mem_width"], str(sc["mem_vt_bits"])))
        if e["implicit_regs"]:
            c = counts["implicit_regs"]
            c[0] += 1
            c[1] += 1
            legacy = sorted((r["reg"], r["is_use"]) for r in e["implicit_regs"])
            if legacy == sorted((r["reg"], r["is_use"]) for r in sc["implicit_regs"]):
                c[2] += 1
            else:
                dis.append((e["name"], "implicit_regs", "", str(legacy), str(sc["implicit_regs"])))
        c = counts["format_class"]
        c[0] += 1
        c[1] += bool(sc["format_class"])
        c[2] += bool(sc["format_class"])
    m9 = {f: {"applicable": a, "schema": f"{p}/{a}", "agree_with_legacy": f"{g}/{p}"} for f, (a, p, g) in counts.items()}
    return m9, dis


def write_outputs(out, rows, m, hits, provenance_dir):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "runtime_name_keys.tsv", "w") as f:
        f.write("file\tline\tkind\tliteral\n")
        for h in hits:
            f.write("\t".join(map(str, h)) + "\n")
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    gen = out / "generated"
    if gen.exists():
        shutil.rmtree(gen)
    gen.mkdir()
    for p in sorted(provenance_dir.glob("gen_*.rs")) + [provenance_dir / "semantic_provenance.json"]:
        shutil.copy2(p, gen / p.name)
    lines = ["# Data-driven ledger metrics", "", METRICS, "## Values", ""]
    for k in ("N", "M1", "M2", "M3", "M4"):
        lines.append(f"- {k}: {m[k]}")
    lines.append(f"- M4 split (hand layer): {m['M4_split']}; name-parse layer: {m['M4_name_parse']}")
    lines.append("- M5 per fact (tablegen share; full mix):")
    for f, v in m["M5"].items():
        lines.append(f"  - {f}: {v['tablegen']}  {v['mix']}")
    lines.append(f"- M6: {m['M6']}  {m['M6_mix']}")
    lines.append(f"- M7: {m['M7']}")
    lines.append(f"- M7b: {m['M7b']}")
    lines.append(f"- M8: {m['M8_literals']} literals touching {m['M8_encodings']} encodings")
    lines.append(f"- winning layers: {m['layers']}")
    if "M9" in m:
        lines.append("- M9 schema availability (applicable / provided by the schema / agreeing with legacy):")
        for f, v in m["M9"].items():
            lines.append(f"  - {f}: schema {v['schema']}, agrees with legacy {v['agree_with_legacy']}")
        lines.append(f"  - disagreements: {m['M9_disagreements']} rows in schema_disagreements.tsv")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Behaviour gate: compare final values between two ledgers
# ---------------------------------------------------------------------------

VALUE_KEYS = ["slot", "mnemonic", "semantic", "element_type", "from_type", "mem_width", "addressing_mode",
              "branch_condition", "select_variant", "may_load", "may_store", "is_vector", "is_ptr_arithmetic",
              "is_sp_relative", "sched_class", "input_order", "output_order"]


def values(e):
    v = {k: e.get(k) for k in VALUE_KEYS}
    v["implicit_regs"] = sorted((r["reg"], r["is_use"]) for r in e["implicit_regs"])
    v["operand_types"] = sorted((f["name"], f["operand_type"]) for f in e["operand_fields"])
    return v


def diff(old_dir, new_dir):
    load = lambda d: {e["name"]: e for e in json.load(open(Path(d) / "generated" / "semantic_provenance.json"))["encodings"]}
    old, new = load(old_dir), load(new_dir)
    bad = 0
    for name in sorted(set(old) | set(new)):
        if name not in old or name not in new:
            print(f"{name}: only in {'new' if name in new else 'old'}")
            bad += 1
            continue
        vo, vn = values(old[name]), values(new[name])
        for k in vo:
            if vo[k] != vn[k]:
                print(f"{name}.{k}: {vo[k]!r} -> {vn[k]!r}")
                bad += 1
    gen_old, gen_new = Path(old_dir) / "generated", Path(new_dir) / "generated"
    for p in sorted(set(x.name for x in gen_old.glob("gen_*.rs")) | set(x.name for x in gen_new.glob("gen_*.rs"))):
        a, b = gen_old / p, gen_new / p
        if not a.exists() or not b.exists() or a.read_bytes() != b.read_bytes():
            print(f"generated table differs: {p}")
            bad += 1
    print(f"{bad} differences")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provenance", type=Path, help="semantic_provenance.json (skips the cargo build)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"))
    a = ap.parse_args()
    if a.diff:
        sys.exit(diff(*a.diff))
    root = repo_root()
    prov_dir = a.provenance.parent if a.provenance else build_provenance(root)
    encodings = json.load(open(prov_dir / "semantic_provenance.json"))["encodings"]
    hits = scan_runtime(root)
    name_keys = match_hits(encodings, hits)
    rows, m = compute(encodings, name_keys, hits)
    dis = []
    if all("schema" in e for e in encodings):
        m["M9"], dis = schema_report(encodings)
        m["M9_disagreements"] = len(dis)
    if a.out is None:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True).strip()
        a.out = main_checkout() / "build/experiments/harness/datadriven" / f"ledger-{branch.replace('/', '-')}"
    write_outputs(a.out, rows, m, hits, prov_dir)
    with open(a.out / "schema_disagreements.tsv", "w") as f:
        f.write("encoding\tfact\titem\tlegacy\tschema\n")
        for d in dis:
            f.write("\t".join(d) + "\n")
    print((a.out / "metrics.md").read_text())
    print(f"ledger: {a.out}")


if __name__ == "__main__":
    main()
