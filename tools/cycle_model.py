#!/usr/bin/env python3
"""Cycle-level discriminator for the AIE2 ZOL partial-word store flush (BUG-B).

Static body-size and data-recency are both *proxies* for one hidden quantity:
the number of cycles between the LE-bundle store issuing and the back-edge
fetch-redirect that can squash it before its E11 commit. AIE2 issues one bundle
per cycle in order (nops included -- AIE2Schedule.td single-bundle issue), so the
cycles in one loop iteration equal the *bundle count* from LS to LE inclusive --
NOT the byte span. Two 96-byte bodies with different nop density have different
cycle counts; that divergence is exactly what the body-size rule discards.

Hypothesis: a partial-word store parked at LE commits iff the iteration is long
enough (in cycles) for the store to reach E11 before the next back-edge squashes
it. So `cycles_per_iter` (bundle count LS..LE inclusive) should separate the
real-HW flush/commit labels where body-bytes alone does not.

This tool computes per store-at-LE kernel: body_bytes, body_bundles
(== cycles_per_iter), data_recency, and the HW label parsed from a fuzz run log,
then tabulates flush/commit vs the cycle feature so we can read the threshold and
its separation quality directly. Pure observation -- no emulator behavior assumed.

OUTCOME (2026-05-31): cycle-count is non-monotonic (at producer-distance 1 only
cycles=25 commits; 24/26/27 flush) -- a fetch-pipeline *phase* effect, not a
duration threshold. Combined with operand freshness (data + address producer
distance) an exhaustive fit separates the harvest set 0/104 but OVERFITS: it
loses to the shipped body-size rule on an independent natural corpus (5 vs 2
wrong; see the recency1b experiment's generalization_test.py). Conclusion: the
shipped `body_bytes <= 96` rule is the static-feature optimum; the ~2% residual
is irreducible to disassembly features. Full writeup in the BUG-B findings doc.

Usage: tools/cycle_model.py <fuzz_root> <hwrun_log>
  e.g. tools/cycle_model.py \
         build/experiments/2026-05-31-recency1b/build/fuzz \
         build/experiments/2026-05-31-recency1b/hwrun_recency1_2k.log
"""
import collections
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cls", os.path.join(HERE, "classify_le_store.py"))
cls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cls)


def parse_labels(logpath):
    """seed -> ('match'|'mismatch', emu, npu) from a fuzz --hw run log."""
    st = {}
    for ln in open(logpath):
        for x in ln.replace("\r", "\n").split("\n"):
            m = re.match(r"seed (\d+) MATCH", x)
            if m:
                st[int(m.group(1))] = ("match", None, None)
            m = re.match(
                r"seed (\d+) MISMATCH: element \[(\d+)\]: "
                r"emulator=(-?\d+), npu=(-?\d+)", x)
            if m:
                st[int(m.group(1))] = ("mismatch", int(m.group(3)), int(m.group(4)))
    return st


def hw_behavior(status, emu_flush_guess):
    """Map a (kind, emu, npu) status to flush / commit / computeDiff.

    On MATCH the EMU and HW agree, so the HW did whatever the current EMU model
    did -- we read that from the body-size guess (the shipped model). On
    MISMATCH the EMU and HW disagree: emu==0,npu!=0 means HW COMMITTED where EMU
    flushed; emu!=0,npu==0 means HW FLUSHED where EMU committed. Anything else is
    an unrelated compute divergence, excluded.
    """
    kind, emu, npu = status
    if kind == "match":
        return "flush" if emu_flush_guess else "commit"
    if emu == 0 and npu != 0:
        return "commit"
    if emu != 0 and npu == 0:
        return "flush"
    return "computeDiff"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    root, logpath = sys.argv[1], sys.argv[2]
    labels = parse_labels(logpath)

    rows = []
    for s in sorted(labels):
        elf = os.path.join(root, f"seed_{s}", "aie.mlir.prj", "main_core_0_2.elf")
        if not os.path.exists(elf):
            continue
        info = cls.classify(elf)
        if not info or not info.get("le_store") or info["le_store"] == "full":
            continue
        body_bytes = info.get("body_bytes")
        body_bundles = info.get("body_bundles")  # == cycles_per_iter
        rec = info.get("data_recency")
        emu_flush_guess = body_bytes is not None and body_bytes <= 96
        hw = hw_behavior(labels[s], emu_flush_guess)
        if hw == "computeDiff":
            continue
        rows.append((s, body_bytes, body_bundles, rec, hw))

    print(f"store-at-LE kernels with a clean flush/commit label: {len(rows)}")
    print()

    # Separation of HW label by cycles_per_iter (body_bundles).
    print("=== HW behavior vs cycles_per_iter (bundle count LS..LE incl) ===")
    by_cyc = collections.defaultdict(lambda: collections.Counter())
    for _s, _bb, cyc, _r, hw in rows:
        by_cyc[cyc][hw] += 1
    print(f"{'cycles':<8}{'flush':<8}{'commit':<8}")
    for cyc in sorted(k for k in by_cyc if k is not None):
        c = by_cyc[cyc]
        print(f"{cyc:<8}{c['flush']:<8}{c['commit']:<8}")

    # For comparison: separation by body_bytes (the shipped proxy).
    print()
    print("=== HW behavior vs body_bytes (shipped proxy) ===")
    by_b = collections.defaultdict(lambda: collections.Counter())
    for _s, bb, _cyc, _r, hw in rows:
        by_b[bb][hw] += 1
    print(f"{'bytes':<8}{'flush':<8}{'commit':<8}")
    for bb in sorted(k for k in by_b if k is not None):
        c = by_b[bb]
        print(f"{bb:<8}{c['flush']:<8}{c['commit']:<8}")

    # Best single threshold on cycles_per_iter: flush iff cyc <= T.
    print()
    print("=== best threshold on cycles_per_iter (flush iff cyc <= T) ===")
    cyc_vals = sorted({c for _s, _bb, c, _r, _hw in rows if c is not None})
    best = None
    for T in range(min(cyc_vals) - 1, max(cyc_vals) + 1):
        wrong = 0
        for _s, _bb, cyc, _r, hw in rows:
            if cyc is None:
                continue
            pred = "flush" if cyc <= T else "commit"
            if pred != hw:
                wrong += 1
        if best is None or wrong < best[1]:
            best = (T, wrong)
    total = sum(1 for r in rows if r[2] is not None)
    print(f"best T={best[0]}  mispredicts={best[1]}/{total}")

    # Same for body_bytes, head to head.
    b_vals = sorted({b for _s, b, _c, _r, _hw in rows if b is not None})
    bbest = None
    for T in range(min(b_vals) - 16, max(b_vals) + 16, 16):
        wrong = 0
        for _s, bb, _cyc, _r, hw in rows:
            if bb is None:
                continue
            pred = "flush" if bb <= T else "commit"
            if pred != hw:
                wrong += 1
        if bbest is None or wrong < bbest[1]:
            bbest = (T, wrong)
    print(f"body_bytes best T={bbest[0]}  mispredicts={bbest[1]}/{total}")

    # Dump the kernels the cycle threshold gets wrong, for inspection.
    print()
    print(f"=== mispredicts at cycles T={best[0]} ===")
    for s, bb, cyc, rec, hw in rows:
        if cyc is None:
            continue
        pred = "flush" if cyc <= best[0] else "commit"
        if pred != hw:
            print(f"  seed_{s:<6} bytes={bb} cycles={cyc} rec={rec} "
                  f"HW={hw} pred={pred}")


def table2d():
    """Auxiliary: 2D contingency of (cycles, recency-bucket) -> flush/commit.

    Invoked as: tools/cycle_model.py <root> <log> --2d
    """
    root, logpath = sys.argv[1], sys.argv[2]
    labels = parse_labels(logpath)
    rows = []
    for s in sorted(labels):
        elf = os.path.join(root, f"seed_{s}", "aie.mlir.prj", "main_core_0_2.elf")
        if not os.path.exists(elf):
            continue
        info = cls.classify(elf)
        if not info or not info.get("le_store") or info["le_store"] == "full":
            continue
        bb = info.get("body_bytes")
        cyc = info.get("body_bundles")
        rec = info.get("data_recency")
        emu_flush_guess = bb is not None and bb <= 96
        hw = hw_behavior(labels[s], emu_flush_guess)
        if hw == "computeDiff":
            continue
        rows.append((s, bb, cyc, rec, hw))

    def rbucket(r):
        if r == 1:
            return "rec1"
        if r in (2, 3, 4):
            return "rec2-4"
        if r is not None:
            return "rec>=5"
        return "stable"

    tab = collections.defaultdict(lambda: collections.Counter())
    for _s, _bb, cyc, rec, hw in rows:
        tab[(cyc, rbucket(rec))][hw] += 1
    print(f"{'cycles':<8}{'recency':<10}{'flush':<8}{'commit':<8}")
    for key in sorted(tab, key=lambda k: (k[0] or 0, k[1])):
        c = tab[key]
        print(f"{key[0]:<8}{key[1]:<10}{c['flush']:<8}{c['commit']:<8}")

    # Candidate unified quantity: dynamic cycles = static bundles + recency stall.
    # A recency-1 store stalls ~1 cycle for its just-written operand (ALU lat 1,
    # no store-unit bypass), adding a dynamic cycle the static count misses.
    print()
    print("=== dynamic-cycle hypothesis: dyn = bundles + (rec==1 ? 1 : 0) ===")
    dtab = collections.defaultdict(lambda: collections.Counter())
    for _s, _bb, cyc, rec, hw in rows:
        if cyc is None:
            continue
        dyn = cyc + (1 if rec == 1 else 0)
        dtab[dyn][hw] += 1
    print(f"{'dyn':<8}{'flush':<8}{'commit':<8}")
    for d in sorted(dtab):
        c = dtab[d]
        print(f"{d:<8}{c['flush']:<8}{c['commit']:<8}")
    # best threshold on dyn
    total = sum(1 for r in rows if r[2] is not None)
    best = None
    dvals = sorted(dtab)
    for T in range(dvals[0] - 1, dvals[-1] + 1):
        wrong = 0
        for _s, _bb, cyc, rec, hw in rows:
            if cyc is None:
                continue
            dyn = cyc + (1 if rec == 1 else 0)
            pred = "flush" if dyn <= T else "commit"
            if pred != hw:
                wrong += 1
        if best is None or wrong < best[1]:
            best = (T, wrong)
    print(f"best dyn T={best[0]}  mispredicts={best[1]}/{total}")
    print()
    print(f"=== dyn mispredicts at T={best[0]} ===")
    for s, bb, cyc, rec, hw in rows:
        if cyc is None:
            continue
        dyn = cyc + (1 if rec == 1 else 0)
        pred = "flush" if dyn <= best[0] else "commit"
        if pred != hw:
            print(f"  seed_{s:<6} bytes={bb} cycles={cyc} rec={rec} dyn={dyn} "
                  f"HW={hw} pred={pred}")


if __name__ == "__main__":
    if "--2d" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--2d"]
        table2d()
    else:
        main()
