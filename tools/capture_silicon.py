#!/usr/bin/env python3
"""Capture real-silicon output as the expected-value oracle for a capture kernel.

The vector-verify kernels are fed designed inputs (tools/vector_inputs.py) and
their expected values are whatever NPU1 silicon produces. This reads the raw
output a HW run leaves in out.txt, pairs it with the kernel's inputs, and writes
a provenance-stamped record to tools/golden/silicon/<kernel>.json. The generator
(gen_vector_kernel.py) stages it as exp.bin (tools/stage_vector_kernel.py).

Record schema:
  {"kernel", "class", "n", "inputs": [one int array per input buffer],
   "silicon": [one int per output element], "provenance"}

Flow per kernel:
  1. Generate, stage with no exp.bin (bootstrap) and compile via the bridge.
  2. Run test.exe on real HW once -> out.txt in the build dir (the uncaptured
     run fails by design; out.txt is written before the compare).
  3. python3 capture_silicon.py <kernel> --out-txt <build>/out.txt --date YYYY-MM-DD
  4. Re-stage (now exp.bin = silicon) and bridge-verify HW (self-consistency)
     and EMU.
"""

import argparse
import json
import os

import gen_vector_kernel as gen
from vector_kernel_specs import SPECS, SWEEPS

_CTYPE_RANGE = {
    "int8_t": (-(2 ** 7), 2 ** 7 - 1), "uint8_t": (0, 2 ** 8 - 1),
    "int16_t": (-(2 ** 15), 2 ** 15 - 1), "uint16_t": (0, 2 ** 16 - 1),
    "int32_t": (-(2 ** 31), 2 ** 31 - 1), "uint32_t": (0, 2 ** 32 - 1),
}


def parse_out_txt(path):
    """Read out.txt (one integer per line) into a list of ints."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip() != ""]


def all_specs():
    """Every kernel spec by name: SPECS plus the expanded sweep points."""
    out = dict(SPECS)
    for sw in SWEEPS.values():
        for pt in sw.expand():
            out[pt.name] = pt
    return out


def resolve_spec(name):
    specs = all_specs()
    if name not in specs:
        raise KeyError(f"unknown kernel '{name}'")
    return specs[name]


def build_record(spec, silicon, date="UNDATED"):
    """Assemble the silicon golden: inputs and the observed output.

    `silicon` length must equal the output element count and each value must fit
    the output's host C type.
    """
    n = spec.n_out
    assert len(silicon) == n, f"silicon len {len(silicon)} != output count {n}"
    lo, hi = _CTYPE_RANGE[spec.output.ctype]
    bad = [v for v in silicon if not lo <= v <= hi]
    assert not bad, f"{len(bad)} silicon values outside {spec.output.ctype} range"
    prov = (f"HW-observed: NPU1 Phoenix silicon, captured {date} via "
            "tools/capture_silicon.py")
    return {"kernel": spec.name, "class": spec.io_class, "n": n,
            "inputs": gen.spec_inputs(spec), "silicon": list(silicon),
            "provenance": prov}


def write_record(rec, out_dir=None):
    out_dir = out_dir or os.path.dirname(gen._silicon_path("x"))
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, rec["kernel"] + ".json")
    with open(dest, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capture silicon output as a golden.")
    ap.add_argument("kernel")
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--date", required=True, help="capture date, YYYY-MM-DD")
    args = ap.parse_args(argv)

    spec = resolve_spec(args.kernel)
    rec = build_record(spec, parse_out_txt(args.out_txt), date=args.date)
    print(f"wrote {write_record(rec)}: {rec['n']} outputs")


if __name__ == "__main__":
    main()
