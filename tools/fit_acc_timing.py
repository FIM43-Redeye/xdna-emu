#!/usr/bin/env python3
"""Fit the accumulator read/write timing from the probe thresholds and compare it with TableGen.

Input: the thresholds JSON of tools/analyze_acc_timing.py (one OLD/NEW threshold t per
producer/consumer pair; t = the smallest issue distance, consumer after producer, at which the
consumer reads the producer's value).

Two checks:

1. Free fit. Unknowns: a landing cycle w_P per producer and a read cycle r_C per consumer, with
   t_PC = w_P - r_C. Every pair is one difference constraint; the system is solved by
   propagation over the bipartite graph from the anchor r(vst) = ANCHOR_READ (the TableGen use
   cycle of II_VST_AM) and every constraint is checked. The result is inconsistent exactly on the
   pairs whose producer and consumer share a forwarding class; those lie one cycle below the
   free fit (the forwarding path), see `forward` below.

2. TableGen prediction. The operand cycles and bypass classes are read from AIE2Schedule.td:
       t = def - use + 1 - [def bypass == use bypass and not NoBypass]
   (def/use are the itinerary operand cycles of the accumulator result and accumulator source).

Usage: python3 tools/fit_acc_timing.py <thresholds.json> [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque

NPU_WORK = os.environ.get("NPU_WORK_DIR", os.path.expanduser("~/npu-work"))
SCHEDULE_TD = os.path.join(NPU_WORK, "llvm-aie", "llvm", "lib", "Target", "AIE", "aie2",
                           "AIE2Schedule.td")
ANCHOR_CONSUMER = "vst_amll"
ANCHOR_READ = 1     # II_VST_AM operand cycle of the stored accumulator

# probe instruction -> TableGen itinerary class
PRODUCER_CLASS = {
    "vmul_i_dense": "II_VMUL", "vmul_i_sparse": "II_VMUL",
    "vmul_f_dense": "II_VMULf", "vmul_f_sparse": "II_VMULf",
    "vmac_i_dense": "II_VMAC", "vmac_i_sparse": "II_VMAC",
    "vmac_f_dense": "II_VMACf", "vmac_f_sparse": "II_VMACf",
    "vaddmac_i_dense": "II_VADDMAC", "vaddmac_i_sparse": "II_VADDMAC",
    "vaddmac_f_dense": "II_VADDMACf", "vaddmac_f_sparse": "II_VADDMACf",
    "vaddmsc_f_dense": "II_VADDMACf", "vaddmsc_f_sparse": "II_VADDMACf",
    "vmov_cm_cm": "II_VMOV_CM", "vmov_bm_bm": "II_VMOV_CM", "vmov_bm_x": "II_VMOV_X_BM_XM",
    "vclr_cm": "II_VCLR", "vadd_f": "II_VACCf", "vadd_i": "II_VACC",
    "vups": "II_VUPS", "vlda_ups": "II_VLDA_UPS",
}
# probe instruction -> (itinerary class, index of the accumulator source in operand_cycles)
CONSUMER_CLASS = {
    "vmac_i_dense_acc1": ("II_VMAC", 1), "vmac_i_sparse_acc1": ("II_VMAC", 1),
    "vmac_f_dense_acc1": ("II_VMACf", 1), "vmac_f_sparse_acc1": ("II_VMACf", 1),
    "vaddmac_i_dense_acc1": ("II_VADDMAC", 1), "vaddmac_i_sparse_acc1": ("II_VADDMAC", 1),
    "vaddmac_i_dense_acc2": ("II_VADDMAC", 2), "vaddmac_i_sparse_acc2": ("II_VADDMAC", 2),
    "vaddmac_f_dense_acc1": ("II_VADDMACf", 1), "vaddmac_f_sparse_acc1": ("II_VADDMACf", 1),
    "vaddmac_f_dense_acc2": ("II_VADDMACf", 2), "vaddmac_f_sparse_acc2": ("II_VADDMACf", 2),
    "vsrs_s16_s32": ("II_VSRS", 1), "vsrs_s32_s64": ("II_VSRS", 1), "vsrs_d16_s32": ("II_VSRS", 1),
    "vsrsm_s16_s32": ("II_VSRSM", 1),
    "vmov_x_bm": ("II_VMOV_X", 1), "vmov_cm_cm": ("II_VMOV_CM", 1), "vmov_bm_bm": ("II_VMOV_CM", 1),
    "vst_amll": ("II_VST_AM", 0), "vst_amlh": ("II_VST_AM", 0),
    "vst_amhl": ("II_VST_AM", 0), "vst_amhh": ("II_VST_AM", 0),
    "vst_srs_s16_s32": ("II_VST_SRS", 0),
    "vadd_f_acc1": ("II_VACCf", 1), "vadd_f_acc2": ("II_VACCf", 2),
    "vadd_i_acc1": ("II_VACC", 1), "vadd_i_acc2": ("II_VACC", 2),
    "vconv_bf16_fp32": ("II_VCONV", 1),
}


def parse_itineraries(path=SCHEDULE_TD):
    """class -> (operand_cycles, bypasses) for every InstrItinData / MemInstrItinData record."""
    text = open(path).read()
    out = {}
    for m in re.finditer(r"(?:Mem)?InstrItinData<\s*(II_\w+)\s*,(.*?)>,\s*(?=//|(?:Mem)?InstrItinData|\])",
                         text, re.S):
        name, body = m.group(1), m.group(2)
        # operand cycle list: the first bracket group made of numbers (comments allowed)
        lists = re.findall(r"\[([^\[\]]*)\]", body)
        cycles = bypass = None
        for lst in lists:
            clean = re.sub(r"/\*.*?\*/", "", lst)
            items = [x.strip() for x in clean.split(",") if x.strip()]
            if items and all(re.fullmatch(r"\d+", x) for x in items) and cycles is None:
                cycles = [int(x) for x in items]
            elif items and all(re.fullmatch(r"\w+_Bypass|NoBypass", x) for x in items):
                bypass = items
        if cycles is not None:
            out[name] = (cycles, bypass or [])
    return out


def itinerary_values(itins, cls, index):
    cycles, byp = itins[cls]
    b = byp[index] if index < len(byp) else "NoBypass"
    return cycles[index], b


TIED_CONSUMER_SUFFIX = "_acc1"
MIN_RAW_DISTANCE = 1


def is_censored(consumer):
    """A tied acc1 consumer (vaddmac family) overwrites the register it reads, so its WAR points
    carry no information and the sweep cannot see a threshold below the first RAW distance."""
    return consumer.startswith("vaddmac") and consumer.endswith(TIED_CONSUMER_SUFFIX)


# probe instruction -> LLVM opcode name (the name `decoder_ffi::opcode_name` returns)
PRODUCER_OPCODE = {
    "vmul_i_dense": "VMUL_vmac_cm_core_dense", "vmul_i_sparse": "VMUL_vmac_cm_core_sparse_narrow",
    "vmul_f_dense": "VMUL_F_vmac_bm_core_dense", "vmul_f_sparse": "VMUL_F_vmac_bm_core_sparse_wide",
    "vmac_i_dense": "VMAC_vmac_cm_core_dense", "vmac_i_sparse": "VMAC_vmac_cm_core_sparse_narrow",
    "vmac_f_dense": "VMAC_F_vmac_bm_core_dense", "vmac_f_sparse": "VMAC_F_vmac_bm_core_sparse_wide",
    "vaddmac_i_dense": "VADDMAC_vmac_bm_core_dense", "vaddmac_i_sparse": "VADDMAC_vmac_cm_core_sparse_narrow",
    "vaddmac_f_dense": "VADDMAC_F_vmac_bm_core_dense", "vaddmac_f_sparse": "VADDMAC_F_vmac_bm_core_sparse_wide",
    "vaddmsc_f_dense": "VADDMSC_F_vmac_bm_core_dense", "vaddmsc_f_sparse": "VADDMSC_F_vmac_bm_core_sparse_wide",
    "vmov_cm_cm": "VMOV_mv_cm", "vmov_bm_bm": "VMOV_mv_cm", "vmov_bm_x": "VMOV_mv_x",
    "vclr_cm": "VCLR_vclr", "vadd_f": "VADD_F", "vadd_i": "VADD",
    "vups": "VUPS_S32_S16_mv_ups_w2b", "vlda_ups": "VLDA_UPS_S32_S16_ag_idx_imm",
}
# probe instruction -> (LLVM opcode name, index of the accumulator among the instruction's uses)
CONSUMER_OPCODE = {
    "vmac_i_dense_acc1": ("VMAC_vmac_cm_core_dense", 0), "vmac_i_sparse_acc1": ("VMAC_vmac_cm_core_sparse_narrow", 0),
    "vmac_f_dense_acc1": ("VMAC_F_vmac_bm_core_dense", 0), "vmac_f_sparse_acc1": ("VMAC_F_vmac_bm_core_sparse_wide", 0),
    "vaddmac_i_dense_acc1": ("VADDMAC_vmac_bm_core_dense", 0), "vaddmac_i_sparse_acc1": ("VADDMAC_vmac_cm_core_sparse_narrow", 0),
    "vaddmac_i_dense_acc2": ("VADDMAC_vmac_bm_core_dense", 1), "vaddmac_i_sparse_acc2": ("VADDMAC_vmac_cm_core_sparse_narrow", 1),
    "vaddmac_f_dense_acc1": ("VADDMAC_F_vmac_bm_core_dense", 0), "vaddmac_f_sparse_acc1": ("VADDMAC_F_vmac_bm_core_sparse_wide", 0),
    "vaddmac_f_dense_acc2": ("VADDMAC_F_vmac_bm_core_dense", 1), "vaddmac_f_sparse_acc2": ("VADDMAC_F_vmac_bm_core_sparse_wide", 1),
    "vsrs_s16_s32": ("VSRS_S16_S32_mv_w_srs", 0), "vsrs_s32_s64": ("VSRS_S32_S64_mv_w_srs", 0),
    "vsrs_d16_s32": ("VSRS_D16_S32_mv_w_srs", 0), "vsrsm_s16_s32": ("VSRSM_S16_S32", 0),
    "vmov_x_bm": ("VMOV_mv_x", 0), "vmov_cm_cm": ("VMOV_mv_cm", 0), "vmov_bm_bm": ("VMOV_mv_cm", 0),
    "vst_amll": ("VST_dmw_sts_am_ag_idx_imm", 0), "vst_srs_s16_s32": ("VST_SRS_S16_S32_ag_idx_imm", 0),
    "vadd_f_acc1": ("VADD_F", 0), "vadd_f_acc2": ("VADD_F", 1),
    "vadd_i_acc1": ("VADD", 0), "vadd_i_acc2": ("VADD", 1),
    "vconv_bf16_fp32": ("VCONV_BF16_FP32", 0),
}


def rust_rows(pairs):
    """The silicon thresholds as rows for the Rust golden test (one row per opcode pair)."""
    rows = {}
    for (p, c), t in sorted(pairs.items()):
        if p not in PRODUCER_OPCODE or c not in CONSUMER_OPCODE:
            continue
        if is_censored(c) and t == MIN_RAW_DISTANCE:
            continue
        key = (PRODUCER_OPCODE[p], *CONSUMER_OPCODE[c])
        if rows.setdefault(key, t) != t:
            raise SystemExit(f"opcode pair {key} has two thresholds: {rows[key]} and {t} ({p}, {c})")
    return [f'    ("{p}", "{c}", {idx}, {t}),' for (p, c, idx), t in sorted(rows.items())]


def load_pairs(path):
    data = json.load(open(path))
    pairs = {}
    for key, v in data.items():
        p, c = key.split(":")
        ts = set(v["groups"].keys())
        if len(ts) == 1:
            pairs[(p, c)] = int(next(iter(ts)))
    return pairs


def predict(itins, p, c):
    d, db = itinerary_values(itins, PRODUCER_CLASS[p], 0)
    cls, idx = CONSUMER_CLASS[c]
    u, ub = itinerary_values(itins, cls, idx)
    match = db == ub and db != "NoBypass"
    return d - u + 1 - int(match), (d, db, u, ub)


def free_fit(pairs, forwarded):
    """Solve w_P - r_C = t over the non-forwarded pairs; returns (w, r, inconsistent edges)."""
    edges = [(p, c, t) for (p, c), t in pairs.items()
             if (p, c) not in forwarded and not (is_censored(c) and t == MIN_RAW_DISTANCE)]
    adj = defaultdict(list)
    for p, c, t in edges:
        adj[("c", c)].append((("p", p), lambda r, t=t: r + t))     # w = r + t
        adj[("p", p)].append((("c", c), lambda w, t=t: w - t))     # r = w - t
    val = {("c", ANCHOR_CONSUMER): ANCHOR_READ}
    queue = deque([("c", ANCHOR_CONSUMER)])
    while queue:
        node = queue.popleft()
        for nxt, f in adj[node]:
            if nxt not in val:
                val[nxt] = f(val[node])
                queue.append(nxt)
    bad = [(p, c, t, val[("p", p)] - val[("c", c)]) for p, c, t in edges
           if ("p", p) in val and ("c", c) in val and val[("p", p)] - val[("c", c)] != t]
    w = {k[1]: v for k, v in val.items() if k[0] == "p"}
    r = {k[1]: v for k, v in val.items() if k[0] == "c"}
    return w, r, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("thresholds")
    ap.add_argument("--md")
    ap.add_argument("--rust", action="store_true", help="print the golden rows for the Rust test")
    args = ap.parse_args(argv)
    pairs = {k: v for k, v in load_pairs(args.thresholds).items()
             if k[0] in PRODUCER_CLASS and k[1] in CONSUMER_CLASS}
    if args.rust:
        print("\n".join(rust_rows(load_pairs(args.thresholds))))
        return 0
    itins = parse_itineraries()

    # forwarding set from TableGen (bypass classes), used to split the free fit.
    forwarded = set()
    for (p, c) in pairs:
        _, (_, db, _, ub) = predict(itins, p, c)
        if db == ub and db != "NoBypass":
            forwarded.add((p, c))

    w, r, bad = free_fit(pairs, forwarded)
    lines = []
    lines.append(f"pairs: {len(pairs)}, forwarded (TableGen bypass match): {len(forwarded)}")
    lines.append(f"free fit over the {len(pairs) - len(forwarded)} non-forwarded pairs, anchor "
                 f"r({ANCHOR_CONSUMER}) = {ANCHOR_READ}: {len(bad)} inconsistent edges")
    for b in bad:
        lines.append(f"   INCONSISTENT {b}")
    lines.append("")
    lines.append("producer visible-from cycle w (fit, no forwarding) vs TableGen def + 1")
    for p in PRODUCER_CLASS:
        if p in w:
            d, db = itinerary_values(itins, PRODUCER_CLASS[p], 0)
            lines.append(f"  {p:18s} fit w={w[p]}  TableGen {PRODUCER_CLASS[p]:14s} def={d} {db:10s} "
                         f"{'AGREE' if w[p] == d + 1 else 'DIFFER'}")
    lines.append("consumer read cycle r (fit) vs TableGen use")
    for c in CONSUMER_CLASS:
        if c in r:
            cls, idx = CONSUMER_CLASS[c]
            u, ub = itinerary_values(itins, cls, idx)
            lines.append(f"  {c:24s} fit r={r[c]}  TableGen {cls:12s} use={u} {ub:10s} "
                         f"{'AGREE' if r[c] == u else 'DIFFER'}")
    # forwarded pairs against the free fit
    lines.append("forwarded pairs: measured t vs free-fit (w - r)")
    resid = defaultdict(int)
    for (p, c) in sorted(forwarded):
        if p in w and c in r:
            resid[pairs[(p, c)] - (w[p] - r[c])] += 1
    lines.append(f"   residual histogram (measured - free fit): {dict(resid)}")
    # full TableGen prediction
    miss = []
    for (p, c), t in sorted(pairs.items()):
        pred, info = predict(itins, p, c)
        if is_censored(c) and t == MIN_RAW_DISTANCE and pred <= MIN_RAW_DISTANCE:
            continue    # at or below the first RAW distance measured
        if pred != t:
            miss.append((p, c, t, pred, info))
    lines.append("")
    lines.append(f"TableGen prediction t = def - use + 1 - match: {len(pairs) - len(miss)} of "
                 f"{len(pairs)} pairs agree (censored tied-consumer pairs count as agreeing when "
                 f"the prediction is <= {MIN_RAW_DISTANCE})")
    for m in miss:
        lines.append(f"   MISMATCH {m}")
    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("```\n" + out + "\n```\n")
    return 1 if miss or bad else 0


if __name__ == "__main__":
    sys.exit(main())
