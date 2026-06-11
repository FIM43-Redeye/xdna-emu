#!/usr/bin/env python3
"""Capture real-silicon output as the golden oracle for a bf16 edge kernel.

Phase-B edge inputs (denormal FTZ, NaN/Inf, overflow) are where the aietools
model diverges from NPU1 silicon, so the model cannot be the oracle. This reads
the raw output a HW run leaves in out.txt, pairs it with the corpus edge-slice
inputs + the model prediction, records every model-vs-silicon divergence (the
phase-B finding), and writes a provenance-stamped silicon golden to
tools/golden/silicon_edge/<kernel>.json. The generator then bakes EXP from it.

Flow per kernel:
  1. Generate (bootstrap model EXP) + compile via the bridge.
  2. Run test.exe on real HW once -> out.txt in the build dir.
  3. python3 capture_silicon_edge.py <kernel> --out-txt <build>/out.txt --date YYYY-MM-DD
  4. Regenerate (now EXP = silicon) and bridge-verify.
"""

import argparse
import json
import os

import gen_vector_kernel as gen
from vector_kernel_specs import SPECS, SWEEPS


def parse_out_txt(path):
    """Read out.txt (one integer per line) into a list of ints."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip() != ""]


def _golden_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden", "vector_ops.json")


def resolve_spec(name):
    """Find a kernel spec by name across SPECS and expanded SWEEPS."""
    if name in SPECS:
        return SPECS[name]
    for sw in SWEEPS.values():
        for pt in sw.expand():
            if pt.name == name:
                return pt
    raise KeyError(f"unknown kernel '{name}'")


def build_record(spec, golden, silicon, date="UNDATED"):
    """Assemble the silicon golden: inputs, model, silicon, divergences.

    `silicon` is the captured HW output (length must equal the output element
    count). Divergences are indices where silicon != model -- the phase-B record.
    """
    prov = (f"HW-observed: NPU1 Phoenix (real silicon), captured {date} via "
            "tools/capture_silicon_edge.py. Oracle tier: hardware observation "
            "(CLAUDE.md source #2). EXP for this edge kernel = `silicon`; `model` "
            "is the aietools prediction, kept for the divergence record only.")
    if spec.matmul is not None:
        a, b, c = gen.bake_matmul(golden[spec.golden["class"]], spec.golden["filt"],
                                  spec.matmul, predicate=spec.golden.get("predicate"))
        model = list(c); n = len(model)
        assert len(silicon) == n, f"silicon len {len(silicon)} != output count {n}"
        div = [{"i": i, "model": model[i], "silicon": silicon[i]}
               for i in range(n) if model[i] != silicon[i]]
        return {"kernel": spec.name, "class": spec.golden["class"], "n": n,
                "input_a": a, "input_b": b, "model": model,
                "silicon": list(silicon), "divergences": div, "provenance": prov}
    in_vals, model = gen._bake_io(gen.replace_silicon(spec, None), golden)
    n = spec.n
    assert len(silicon) == n, f"silicon len {len(silicon)} != n {n}"
    div = [{"i": i, "input": in_vals[i], "model": model[i], "silicon": silicon[i]}
           for i in range(n) if model[i] != silicon[i]]
    return {"kernel": spec.name, "class": spec.golden["class"], "n": n,
            "input": in_vals, "model": model, "silicon": list(silicon),
            "divergences": div, "provenance": prov}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capture silicon output as an edge golden.")
    ap.add_argument("kernel")
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--date", default="UNDATED")
    args = ap.parse_args(argv)

    spec = resolve_spec(args.kernel)
    golden = json.loads(open(_golden_path()).read())
    silicon = parse_out_txt(args.out_txt)
    rec = build_record(spec, golden, silicon, date=args.date)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "golden", "silicon_edge")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, args.kernel + ".json")
    with open(dest, "w") as f:
        json.dump(rec, f, indent=1); f.write("\n")
    print(f"wrote {dest}: {len(rec['divergences'])} model-vs-silicon divergences")
    for d in rec["divergences"][:20]:
        line = f"  [{d['i']}] model={d['model']:#x} silicon={d['silicon']:#x}"
        if "input" in d:
            line += f" input={d['input']:#x}"
        print(line)


if __name__ == "__main__":
    main()
