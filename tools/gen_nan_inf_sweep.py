#!/usr/bin/env python3
"""Generate the bf16/fp32 elementwise add/sub NaN/Inf-input silicon sweep.

This is an OFFLINE authoring tool. It emits, for each of the four
(type, op) combinations -- {bf16, fp32} x {add, sub} -- a self-contained
vector-verify kernel directory under tests/vector-verify/, plus an offline
"expected under the current emulator" dump and the shared input matrix under
build/experiments/.

The matrix is the full cross product of an operand-class list against itself
(both signs), so every output lane is exactly one (A_class, B_class) pair and
is self-localizing. The kernel is single-op: one aie::add / aie::sub repeated
across the matrix, so there is zero pipeline-adjacency confound.

What this tool does NOT do: it does not run hardware, and it does not touch the
interpreter. The "expected under current emulator" column is a faithful,
standalone re-implementation of the documented emulator code path
(src/interpreter/execute/vector_arith.rs vector_add/vector_sub, BFloat16 and
Float32 lanes; src/interpreter/execute/vector_helpers.rs bf16_to_f32 /
f32_to_bf16). It exists so that post-capture we can diff HW vs the present
emulator behavior per lane and read off the real datapath rule. The interpreter
fix waits on real silicon data -- see the design note
docs/superpowers/specs/2026-06-10-nan-inf-add-sub-sweep-design.md.

Usage:
    python3 tools/gen_nan_inf_sweep.py            # emit all four kernels + dumps
    python3 tools/gen_nan_inf_sweep.py --check     # regenerate + diff (CI-style)
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO, "tests", "vector-verify")
EXPERIMENTS_DIR = os.path.join(REPO, "build", "experiments", "nan-inf-add-sub-sweep")


# ---------------------------------------------------------------------------
# Operand-class representatives
# ---------------------------------------------------------------------------
#
# Sixteen classes, each a single bit-pattern representative, covering both
# signs of: zero, denormal, small normal, large normal, Inf, qNaN, sNaN, and a
# second NaN with a DISTINCT mantissa payload (to learn which operand's payload
# propagates and the sign rule). The cross product A x B (16x16 = 256) contains
# every adversarial discriminator pair explicitly:
#   Inf + (-Inf), Inf + finite, NaN + finite, NaN + NaN (distinct mantissas),
#   NaN + 0, denormal + denormal, large + large (overflow->Inf?), -0 + +0.
#
# bf16: 1 sign / 8 exp / 7 mantissa. exp all-ones = 255.
#   Inf  : exp=255, man=0
#   qNaN : exp=255, man MSB (bit 6, 0x40) set
#   sNaN : exp=255, man MSB clear, low bits set
# fp32: 1 sign / 8 exp / 23 mantissa. exp all-ones = 255.
#   qNaN : man MSB (bit 22, 0x400000) set
#   sNaN : man MSB clear, low bits set

# Class order is fixed and shared by every kernel so lane index -> (A,B) pair is
# stable across the suite. CLASS_NAMES[i] is the human label for class index i.
CLASS_NAMES = [
    "+0", "-0",
    "+denorm", "-denorm",
    "+nsmall", "-nsmall",
    "+nlarge", "-nlarge",
    "+inf", "-inf",
    "+qnan", "-qnan",
    "+snan", "-snan",
    "+qnan2", "-qnan2",
]
NCLASS = len(CLASS_NAMES)
assert NCLASS == 16

# bf16 representatives (uint16 bit patterns).
BF16_REPS = [
    0x0000, 0x8000,   # +0, -0
    0x0001, 0x8001,   # +denorm, -denorm (smallest subnormal)
    0x3880, 0xB880,   # +nsmall, -nsmall (exp=113, ~2^-14 normal)
    0x7F00, 0xFF00,   # +nlarge, -nlarge (exp=254, max finite)
    0x7F80, 0xFF80,   # +inf, -inf
    0x7FC0, 0xFFC0,   # +qnan, -qnan  (man=0x40, MSB set)
    0x7FA0, 0xFFA0,   # +snan, -snan  (man=0x20, MSB clear, low bits set)
    0x7FD5, 0xFFD5,   # +qnan2, -qnan2 (man=0x55, distinct payload)
]

# fp32 representatives (uint32 bit patterns). Mantissa payloads chosen so the
# distinct-payload NaN (0x355555) is unmistakable from the primary qNaN
# (0x400000) when reading propagation off silicon.
FP32_REPS = [
    0x00000000, 0x80000000,   # +0, -0
    0x00000001, 0x80000001,   # +denorm, -denorm (smallest subnormal)
    0x00800000, 0x80800000,   # +nsmall, -nsmall (exp=1, smallest normal)
    0x7F000000, 0xFF000000,   # +nlarge, -nlarge (exp=254, ~1.7e38)
    0x7F800000, 0xFF800000,   # +inf, -inf
    0x7FC00000, 0xFFC00000,   # +qnan, -qnan  (man MSB set)
    0x7FA00000, 0xFFA00000,   # +snan, -snan  (man MSB clear, low bits set)
    0x7FB55555, 0xFFB55555,   # +qnan2, -qnan2 (distinct payload, MSB clear*)
]
# *note: 0x7FB55555 has mantissa 0x355555 (bit22 clear, so technically an sNaN
#  by encoding) with a wholly distinct low payload; its purpose is payload-
#  tracking, not q/s classification, so the encoding class is secondary here.

assert len(BF16_REPS) == NCLASS
assert len(FP32_REPS) == NCLASS


# ---------------------------------------------------------------------------
# Offline "expected under current emulator" model
# ---------------------------------------------------------------------------
#
# Faithful re-implementation of the documented emulator datapath. NOT the
# proposed silicon model -- this is what the emulator computes TODAY, so we can
# diff it against the silicon capture.

# numpy float32 arithmetic is bit-for-bit IEEE-754 single precision -- the same
# operation Rust's `f32` add/sub performs (round-to-nearest-even, +/-Inf on
# overflow, host FPU NaN propagation). Using it (rather than Python double then
# repack) is what makes the offline column a faithful mirror of the emulator's
# native-f32 datapath, including the Inf-on-overflow that double-then-repack
# would instead raise as an OverflowError.

def _bits_to_f32(bits: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", bits & 0xFFFFFFFF), dtype="<f4")[0]


def _f32_to_bits(f: np.float32) -> int:
    return struct.unpack("<I", np.float32(f).tobytes())[0]


def _bf16_to_f32(bits: int) -> np.float32:
    """vector_helpers.rs bf16_to_f32: (bits as u32) << 16 reinterpreted as f32."""
    return _bits_to_f32((bits & 0xFFFF) << 16)


def _f32_to_bf16_truncate(f: np.float32) -> int:
    """vector_helpers.rs f32_to_bf16: (val.to_bits() >> 16). Raw truncate, no
    rounding, no NaN preservation -- exactly the elementwise add/sub path."""
    return (_f32_to_bits(f) >> 16) & 0xFFFF


def emu_bf16_op(a_bits: int, b_bits: int, op: str) -> int:
    """vector_arith.rs BFloat16 lane: host f32 add/sub then raw truncate."""
    fa = _bf16_to_f32(a_bits)
    fb = _bf16_to_f32(b_bits)
    # Inf/NaN arithmetic raises numpy RuntimeWarnings by design; those special
    # results are exactly what we capture, so silence the warnings locally.
    with np.errstate(all="ignore"):
        r = np.float32(fa + fb) if op == "add" else np.float32(fa - fb)
    return _f32_to_bf16_truncate(r)


def emu_fp32_op(a_bits: int, b_bits: int, op: str) -> int:
    """vector_arith.rs Float32 lane: plain host IEEE add/sub on the bits."""
    fa = _bits_to_f32(a_bits)
    fb = _bits_to_f32(b_bits)
    with np.errstate(all="ignore"):
        r = np.float32(fa + fb) if op == "add" else np.float32(fa - fb)
    return _f32_to_bits(r)


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

@dataclass
class Combo:
    name: str           # kernel dir name, e.g. "vec_nan_bf16_add"
    elem: str           # "bf16" or "fp32"
    op: str             # "add" or "sub"
    reps: list          # class representatives
    bitwidth: int       # 16 or 32
    mlir_ty: str        # "bf16" or "f32"
    host_ty: str        # "uint16_t" or "uint32_t"
    kern_ty: str        # "bfloat16" or "float"
    lanes: int          # vector lanes per op (32 for bf16, 8 for fp32)

    def build_matrix(self):
        """Return (A_bits[], B_bits[], emu_out_bits[]) for the full 256 pairs.
        Lane index = a_class * NCLASS + b_class."""
        a_arr, b_arr, e_arr = [], [], []
        op = emu_bf16_op if self.elem == "bf16" else emu_fp32_op
        # Inf/NaN arithmetic raises numpy RuntimeWarnings by design; the special
        # results are exactly what we want to capture, so silence the warnings.
        with np.errstate(all="ignore"):
            for ai in range(NCLASS):
                for bi in range(NCLASS):
                    a = self.reps[ai]
                    b = self.reps[bi]
                    a_arr.append(a)
                    b_arr.append(b)
                    e_arr.append(op(a, b, self.op))
        return a_arr, b_arr, e_arr


COMBOS = [
    Combo("vec_nan_bf16_add", "bf16", "add", BF16_REPS, 16, "bf16", "uint16_t", "bfloat16", 32),
    Combo("vec_nan_bf16_sub", "bf16", "sub", BF16_REPS, 16, "bf16", "uint16_t", "bfloat16", 32),
    Combo("vec_nan_fp32_add", "fp32", "add", FP32_REPS, 32, "f32", "uint32_t", "float", 8),
    Combo("vec_nan_fp32_sub", "fp32", "sub", FP32_REPS, 32, "f32", "uint32_t", "float", 8),
]

N = NCLASS * NCLASS  # 256 elements per input array


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def _fmt_array(name: str, host_ty: str, vals: list) -> str:
    lines = [f"static const {host_ty} {name}[{len(vals)}] = {{"]
    row = []
    for v in vals:
        row.append(str(v))
        if len(row) == 12:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def emit_kernel_cc(c: Combo) -> str:
    intr = "aie::add" if c.op == "add" else "aie::sub"
    # bf16 elementwise chains can trip Peano GlobalISel ("Register class not
    # set"); route the single op through a noinline helper (the fuzzer's
    # src/fuzzer/vector/lower.rs workaround) to keep it native and compile-clean.
    noinline_helper = ""
    body_call = f"  aie::vector<{c.kern_ty}, {c.lanes}> vc = {intr}(va, vb);"
    if c.elem == "bf16":
        noinline_helper = (
            f"using V = aie::vector<{c.kern_ty}, {c.lanes}>;\n"
            f"__attribute__((noinline)) static V vop(V a, V b) {{ return {intr}(a, b); }}\n\n"
        )
        body_call = "  aie::vector<bfloat16, %d> vc = vop(va, vb);" % c.lanes
    return f"""//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// NaN/Inf-input sweep capture kernel (GENERATED by tools/gen_nan_inf_sweep.py).
// {c.elem} elementwise vector {c.op} ({intr}) over a 256-element operand-class
// matrix. Single op, repeated across the matrix -- no pipeline-adjacency
// confound. Inputs/outputs are staged as raw bit patterns ({c.host_ty}); the
// kernel signature uses the real element type ({c.kern_ty}*) so the DMA moves
// bytes verbatim and the host compare is bit-exact (NaN!=NaN safe).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#define SWEEP_N {N}

{noinline_helper}extern "C" {{

void {c.op}_{c.elem}({c.kern_ty} *restrict a, {c.kern_ty} *restrict b,
                     {c.kern_ty} *restrict c) {{
  event0();
  for (int i = 0; i < SWEEP_N; i += {c.lanes}) {{
    aie::vector<{c.kern_ty}, {c.lanes}> va = aie::load_v<{c.lanes}>(a + i);
    aie::vector<{c.kern_ty}, {c.lanes}> vb = aie::load_v<{c.lanes}>(b + i);
{body_call}
    aie::store_v(c + i, vc);
  }}
  event1();
}}

}} // extern "C"
"""


def emit_aie_mlir(c: Combo) -> str:
    fn = f"{c.op}_{c.elem}"
    mt = f"memref<{N}x{c.mlir_ty}>"
    return f"""//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// NaN/Inf sweep design (GENERATED). Single compute tile (0,2) reads two
// {N}-element {c.mlir_ty} operand matrices from DDR via shim DMA, runs the
// {fn} kernel, writes the {N}-element {c.mlir_ty} result back. Direct
// shim<->core objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {{
  aie.device(npu1_1col) {{
    func.func private @{fn}({mt}, {mt}, {mt})
        attributes {{link_with = "kernel.o"}}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {{%core}}, 2 : i32) : !aie.objectfifo<{mt}>
    aie.objectfifo @inB(%shim, {{%core}}, 2 : i32) : !aie.objectfifo<{mt}>
    aie.objectfifo @outC(%core, {{%shim}}, 2 : i32) : !aie.objectfifo<{mt}>

    %core_0_2 = aie.core(%core) {{
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<{mt}>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<{mt}> -> {mt}
      %sb = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<{mt}>
      %b = aie.objectfifo.subview.access %sb[0] : !aie.objectfifosubview<{mt}> -> {mt}
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<{mt}>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<{mt}> -> {mt}

      func.call @{fn}(%a, %b, %o) : ({mt}, {mt}, {mt}) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @inB(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }}

    aie.runtime_sequence @sequence(%a: {mt}, %b: {mt}, %c: {mt}) {{
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, {N}][0, 0, 0, 1]) {{id = 0 : i64, metadata = @inA}} : {mt}
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, {N}][0, 0, 0, 1]) {{id = 1 : i64, metadata = @inB}} : {mt}
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, {N}][0, 0, 0, 1]) {{id = 2 : i64, metadata = @outC}} : {mt}
      aiex.npu.dma_wait {{symbol = @outC}}
    }}
  }}
}}
"""


def emit_run_lit(c: Combo) -> str:
    # Peano flow: compile kernel.cc -> kernel.o (linked in via the func-level
    # link_with on the aie.mlir func decl), then aiecc (--no-xchesscc) builds
    # the xclbin + npu insts. The bf16 kernels need C++20 + the aie_api include
    # (the noinline helper keeps them GlobalISel-clean). Verified compile-clean
    # offline (vadd.f / vsub.f native, zero errors) on 2026-06-10.
    return f"""// (c) Copyright 2026 -- xdna-emu NaN/Inf add/sub silicon sweep.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: ryzen_ai_npu1, peano
//
// RUN: %PEANO_INSTALL_DIR/bin/clang++ --target=aie2-none-unknown-elf -O2 -std=c++20 -I %aietools/include -c %S/kernel.cc -o ./kernel.o
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-xclbin --xclbin-name=aie.xclbin --aie-generate-npu-insts --npu-insts-name=insts.bin %S/aie.mlir
// RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
// RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
"""


def emit_test_cpp(c: Combo, a_arr, b_arr, e_arr) -> str:
    fn = f"{c.op}_{c.elem}"
    arr_a = _fmt_array("INA", c.host_ty, a_arr)
    arr_b = _fmt_array("INB", c.host_ty, b_arr)
    arr_e = _fmt_array("EMU", c.host_ty, e_arr)
    return f"""//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Host harness for the NaN/Inf {c.elem} elementwise {c.op} sweep (GENERATED by
// tools/gen_nan_inf_sweep.py). Stages the operand-class matrix as raw bit
// patterns, launches, and DUMPS per-lane {{A_bits, B_bits, hw_out_bits}} to
// out.txt for silicon capture. The baked EMU[] column is the CURRENT emulator
// expectation (host IEEE add/sub then raw bf16 truncate / plain fp32) -- it is
// NOT a golden; the run "PASSES" iff hw==emu, and a real silicon run is EXPECTED
// to diverge on special-operand lanes. The divergence is the finding.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

static constexpr int N = {N};

{arr_a}
{arr_b}
{arr_e}

int main(int argc, const char *argv[]) {{
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary("insts.bin");

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string xclbin_name = "aie.xclbin";
  xrt::xclbin xclbin(xclbin_name);
  std::string Node = "MLIR_AIE";

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {{
                                 return k.get_name().rfind(Node, 0) == 0;
                               }});
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, N * sizeof({c.host_ty}), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, N * sizeof({c.host_ty}), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, N * sizeof({c.host_ty}), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  {c.host_ty} *bufA = bo_a.map<{c.host_ty} *>();
  {c.host_ty} *bufB = bo_b.map<{c.host_ty} *>();
  {c.host_ty} *bufC = bo_c.map<{c.host_ty} *>();

  std::memcpy(bufA, INA, N * sizeof({c.host_ty}));
  std::memcpy(bufB, INB, N * sizeof({c.host_ty}));
  std::memset(bufC, 0, N * sizeof({c.host_ty}));

  void *bufInstr = bo_instr.map<void *>();
  std::memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {{
    std::cout << "Kernel did not complete. Status: " << r << "\\n";
    return 1;
  }}

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Per-lane silicon dump: A_bits B_bits hw_out_bits (decimal). This is the
  // ground-truth artifact -- one line per (A_class, B_class) pair.
  {{
    std::ofstream dump("out.txt");
    for (int i = 0; i < N; i++)
      dump << (uint32_t)bufA[i] << " "
           << (uint32_t)bufB[i] << " "
           << (uint32_t)bufC[i] << "\\n";
  }}

  int diffs = 0;
  for (int i = 0; i < N; i++) {{
    if (bufC[i] != EMU[i]) {{
      if (diffs < 16)
        std::cout << "Diff [" << i << "]: a=0x" << std::hex << (uint32_t)bufA[i]
                  << " b=0x" << (uint32_t)bufB[i] << " hw=0x" << (uint32_t)bufC[i]
                  << " emu=0x" << (uint32_t)EMU[i] << std::dec << "\\n";
      diffs++;
    }}
  }}

  std::cout << "\\n" << diffs << " lane(s) diverge from current emulator (out of "
            << N << ").\\n";
  if (!diffs) {{
    std::cout << "\\nPASS!\\n\\n";
    return 0;
  }}
  // A nonzero diff count is the EXPECTED, INFORMATIVE outcome on real silicon.
  // out.txt holds the ground truth; the model update is read off from it.
  std::cout << "\\n(divergence is expected on silicon; see out.txt)\\n\\n";
  return 1;
}}
"""


def emit_offline_dump(c: Combo, a_arr, b_arr, e_arr) -> str:
    """The build/experiments offline dump: per-lane class labels + bits +
    current-emulator expected. Same columns the silicon out.txt will have, plus
    the class labels so a human can read the discriminators directly."""
    lines = [
        f"# OFFLINE EXPECTED (current emulator) -- {c.name}",
        f"# {c.elem} elementwise {c.op}, {N} (A_class x B_class) lanes.",
        "# Columns: lane  A_class  B_class  A_bits  B_bits  emu_out_bits",
        "# emu_out_bits is what the PRESENT emulator computes (host IEEE add/sub",
        "# then raw truncate for bf16). Diff this against the silicon out.txt.",
        "#",
    ]
    width = max(len(n) for n in CLASS_NAMES)
    fmt_bits = "0x%04X" if c.bitwidth == 16 else "0x%08X"
    for lane in range(N):
        ai, bi = divmod(lane, NCLASS)
        lines.append(
            "%4d  %-*s  %-*s  %s  %s  %s"
            % (
                lane,
                width, CLASS_NAMES[ai],
                width, CLASS_NAMES[bi],
                fmt_bits % a_arr[lane],
                fmt_bits % b_arr[lane],
                fmt_bits % e_arr[lane],
            )
        )
    return "\n".join(lines) + "\n"


def write_if_changed(path: str, content: str, check: bool, changed: list):
    existing = None
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = f.read()
    if existing == content:
        return
    changed.append(path)
    if not check:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing (exit 1 if stale)")
    args = ap.parse_args()

    changed: list = []
    for c in COMBOS:
        a_arr, b_arr, e_arr = c.build_matrix()
        kdir = os.path.join(TESTS_DIR, c.name)
        write_if_changed(os.path.join(kdir, "kernel.cc"), emit_kernel_cc(c), args.check, changed)
        write_if_changed(os.path.join(kdir, "aie.mlir"), emit_aie_mlir(c), args.check, changed)
        write_if_changed(os.path.join(kdir, "run.lit"), emit_run_lit(c), args.check, changed)
        write_if_changed(os.path.join(kdir, "test.cpp"),
                         emit_test_cpp(c, a_arr, b_arr, e_arr), args.check, changed)
        write_if_changed(os.path.join(EXPERIMENTS_DIR, f"{c.name}.expected.txt"),
                         emit_offline_dump(c, a_arr, b_arr, e_arr), args.check, changed)

    if args.check:
        if changed:
            print("STALE -- regenerate with: python3 tools/gen_nan_inf_sweep.py")
            for p in changed:
                print("  " + os.path.relpath(p, REPO))
            return 1
        print("up to date")
        return 0

    if changed:
        print(f"wrote {len(changed)} file(s):")
        for p in changed:
            print("  " + os.path.relpath(p, REPO))
    else:
        print("no changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
