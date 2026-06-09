#!/usr/bin/env python3
"""Spec-driven generator for Half-B vector-compute capture kernels.

Each AIE2 vector-compute class (SRS, UPS, Pack, ...) needs a bridge kernel that
exercises the *actual* vector intrinsic end-to-end (shim DMA -> core vector op
-> shim DMA) so a bridge run can confirm the emulator matches real NPU1 silicon.
The four files such a kernel needs -- run.lit, aie.mlir, test.cpp, kernel.cc --
are ~80% boilerplate; the genuinely variable parts are the intrinsic body (the
IP, hand-written once per class) and the golden expected-output arrays.

This module turns a `KernelSpec` plus the Half-A golden corpus
(tools/golden/vector_ops.json) into those four files. The golden arrays are
baked from the corpus -- never transcribed by hand -- so a misread can't slip a
wrong expected value into a capture kernel. The correctness anchor is that
regenerating a known-good kernel (vec_srs_i32) reproduces its committed arrays.

Provenance: the golden corpus is itself derived from the genuine aietools model
(see tools/golden/README.md). This generator only *selects and bakes*; it
introduces no new oracle.
"""


import string
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Buf:
    """One DMA-connected buffer: host C type + MLIR element type."""

    name: str
    ctype: str
    mlir: str


@dataclass
class KernelSpec:
    """Everything needed to emit one vector-compute capture kernel.

    The `golden` dict selects a config slice of a vector_ops.json class:
    {"class": <key>, "filt": {field: value, ...}, "value_range": (lo, hi) | None}.
    Input values are baked from each record's `value`; expected outputs from
    `expected`. `body` is the hand-written intrinsic body (the IP), wrapped by
    the generated kernel.cc scaffold.
    """

    name: str
    func: str
    doc: str
    inputs: List[Buf]
    output: Buf
    n: int
    golden: dict
    body: str
    defines: List[Tuple[str, object]] = field(default_factory=list)
    stem: Optional[str] = None


def _stem(spec):
    """Basename for the kernel's .cc/.o (e.g. "srs" from func "srs_i32")."""
    return spec.stem if spec.stem else spec.func.split("_")[0]


def _format_c_array(name, ctype, values, per_line=8, indent="    "):
    """Emit `static const <ctype> <name>[N] = { ... };`, per_line values per row."""
    rows = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(v) for v in values[i:i + per_line])
        rows.append(f"{indent}{chunk},")
    body = "\n".join(rows)
    return (f"static const {ctype} {name}[{len(values)}] = {{\n"
            f"{body}\n}};")


def select_records(records, filt, value_range=None, value_field="value"):
    """Filter a golden class's records to a single config slice, in corpus order.

    `filt` is a dict of field==value constraints (e.g. {"shift": 4, "sat": True}).
    `value_range`, if given, is an inclusive (lo, hi) bound on `value_field` --
    used to keep only host-representable inputs (e.g. int32-representable
    accumulator values). Order is preserved so a baked array is reproducible.
    """
    out = []
    for r in records:
        if any(r.get(k) != v for k, v in filt.items()):
            continue
        if value_range is not None:
            lo, hi = value_range
            if not (lo <= r[value_field] <= hi):
                continue
        out.append(r)
    return out


def bake_array(records, field, n, pad=0):
    """Extract `field` from the first `n` records, padding to length n with `pad`.

    Asserts the slice fits (len(records) <= n) so silent truncation can't hide a
    larger-than-buffer golden slice behind a passing test.
    """
    vals = [r[field] for r in records[:n]]
    assert len(vals) <= n, f"{len(records)} records exceed buffer size {n}"
    vals.extend([pad] * (n - len(vals)))
    return vals


_TEST_CPP_TMPL = string.Template(
    """//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the Half-B $name capture kernel (GENERATED -- edit the spec
// in vector_kernel_specs.py, not this file). Feeds a baked input batch and
// checks the output against the Half-A golden ($gclass slice). PASS means the
// $name datapath ran correctly. Expected values are the genuine
// aietools-model outputs baked from tools/golden/vector_ops.json.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

static constexpr int N = $n;

$in_array
$exp_array

int main(int argc, const char *argv[]) {
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string xclbin_name = "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);
  std::string Node = "MLIR_AIE";

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 return k.get_name().rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, N * sizeof($in_ctype), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, N * sizeof($out_ctype), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  $in_ctype *bufIn = bo_in.map<$in_ctype *>();
  $out_ctype *bufOut = bo_out.map<$out_ctype *>();

  std::memcpy(bufIn, IN, N * sizeof($in_ctype));
  std::memset(bufOut, 0, N * sizeof($out_ctype));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Status: " << r << "\\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (bufOut[i] != EXP[i]) {
      if (errors < 10)
        std::cout << "Error [" << i << "]: in=" << IN[i]
                  << " got=" << bufOut[i] << " != exp=" << EXP[i] << "\\n";
      errors++;
    }
  }

  if (!errors) {
    std::cout << "\\nPASS!\\n\\n";
    return 0;
  }
  std::cout << "\\nfailed (" << errors << " errors).\\n\\n";
  return 1;
}
"""
)


def _bake_io(spec, golden):
    """Select the spec's golden slice and bake (input, expected) arrays to N."""
    g = spec.golden
    recs = select_records(
        golden[g["class"]], g["filt"], g.get("value_range")
    )
    in_vals = bake_array(recs, g.get("value_field", "value"), spec.n)
    exp_vals = bake_array(recs, g.get("expected_field", "expected"), spec.n)
    return in_vals, exp_vals


def render_test_cpp(spec, golden):
    """Render the host harness, baking golden input/expected arrays from the corpus."""
    in_vals, exp_vals = _bake_io(spec, golden)
    return _TEST_CPP_TMPL.substitute(
        name=spec.name,
        gclass=spec.golden["class"],
        n=spec.n,
        in_ctype=spec.inputs[0].ctype,
        out_ctype=spec.output.ctype,
        in_array=_format_c_array("IN", spec.inputs[0].ctype, in_vals),
        exp_array=_format_c_array("EXP", spec.output.ctype, exp_vals),
    )


_MLIR_TMPL = string.Template(
    """//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B $name capture design (GENERATED). Single compute tile (0,2) reads a
// $n-element batch from DDR via shim DMA, runs the $name kernel, writes the
// result back. Direct shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @$func(memref<${n}x${in_ty}>, memref<${n}x${out_ty}>)
        attributes {link_with = "$obj"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<${n}x${in_ty}>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<${n}x${out_ty}>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<${n}x${in_ty}>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<${n}x${in_ty}>> -> memref<${n}x${in_ty}>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<${n}x${out_ty}>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<${n}x${out_ty}>> -> memref<${n}x${out_ty}>

      func.call @$func(%a, %o) : (memref<${n}x${in_ty}>, memref<${n}x${out_ty}>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<${n}x${in_ty}>, %c: memref<${n}x${out_ty}>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, $n][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<${n}x${in_ty}>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, $n][0, 0, 0, 1]) {id = 1 : i64, metadata = @outC} : memref<${n}x${out_ty}>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
"""
)


def render_mlir(spec):
    """Render the single-tile shim<->core design for a scalar-elementwise kernel."""
    assert len(spec.inputs) == 1, "render_mlir currently models single-input kernels"
    return _MLIR_TMPL.substitute(
        name=spec.name,
        func=spec.func,
        obj=f"{_stem(spec)}.o",
        n=spec.n,
        in_ty=spec.inputs[0].mlir,
        out_ty=spec.output.mlir,
    )


def _cc_header(stem):
    """80-column LLVM-style file ruler for `<stem>.cc`, regardless of stem length."""
    prefix = f"//===- {stem}.cc "
    suffix = "*- C++ -*-===//"
    return prefix + "-" * (80 - len(prefix) - len(suffix)) + suffix


_KERNEL_TMPL = string.Template(
    """$header
//
// Half-B vector-compute capture kernel (GENERATED -- edit the spec in
// vector_kernel_specs.py, not this file). $doc
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>
$defines
extern "C" {

void $func($sig) {
$body}

} // extern "C"
"""
)


def _signature(spec):
    """`<ctype> *restrict <name>` for each input then the output, comma-joined."""
    bufs = list(spec.inputs) + [spec.output]
    return ", ".join(f"{b.ctype} *restrict {b.name}" for b in bufs)


def render_kernel(spec):
    """Wrap the hand-written intrinsic body in the standard kernel.cc scaffold."""
    defines = "".join(f"\n#define {k} {v}" for k, v in spec.defines)
    if defines:
        defines += "\n"
    return _KERNEL_TMPL.substitute(
        header=_cc_header(_stem(spec)),
        doc=spec.doc,
        defines=defines,
        func=spec.func,
        sig=_signature(spec),
        body=spec.body,
    )


_RUN_LIT_TMPL = string.Template(
    """// (c) Copyright 2026 -- xdna-emu Half-B vector-compute verification.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: ryzen_ai_npu1, chess
//
// RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/$cc -o ./$obj
// RUN: %python aiecc.py --xchesscc --xbridge --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.bin %S/aie.mlir
// RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
// RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
"""
)


def render_run_lit(spec):
    """Render the lit build/run recipe (Chess compile -> aiecc -> host -> run)."""
    stem = _stem(spec)
    return _RUN_LIT_TMPL.substitute(cc=f"{stem}.cc", obj=f"{stem}.o")


def generate(spec, golden, outdir):
    """Write the four kernel files into outdir/spec.name; return that directory.

    Files: run.lit, aie.mlir, test.cpp, <stem>.cc. The golden arrays in test.cpp
    are baked from `golden` (the loaded vector_ops.json corpus).
    """
    import os

    dest = os.path.join(str(outdir), spec.name)
    os.makedirs(dest, exist_ok=True)
    files = {
        "run.lit": render_run_lit(spec),
        "aie.mlir": render_mlir(spec),
        "test.cpp": render_test_cpp(spec, golden),
        f"{_stem(spec)}.cc": render_kernel(spec),
    }
    for fn, text in files.items():
        with open(os.path.join(dest, fn), "w") as f:
            f.write(text)
    return dest


def _default_golden_path():
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "golden", "vector_ops.json")


def _default_out_dir():
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "vector-verify")


def main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Generate Half-B vector-compute capture kernels from specs.")
    ap.add_argument("name", help="spec name (e.g. vec_ups_i32), or 'all'")
    ap.add_argument("--out", default=_default_out_dir(),
                    help="output root (default: tests/vector-verify)")
    ap.add_argument("--golden", default=_default_golden_path(),
                    help="vector_ops.json corpus (default: tools/golden/...)")
    args = ap.parse_args(argv)

    from vector_kernel_specs import SPECS
    golden = json.loads(open(args.golden).read())

    names = sorted(SPECS) if args.name == "all" else [args.name]
    for name in names:
        if name not in SPECS:
            ap.error(f"unknown spec '{name}'; known: {', '.join(sorted(SPECS))}")
        dest = generate(SPECS[name], golden, args.out)
        print(f"generated {name} -> {dest}")


if __name__ == "__main__":
    main()
