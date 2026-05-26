#!/usr/bin/env python3
"""Materialize K-variants of _diag_shim_chain_sweep.

For follow-up #2 of the cycle-accuracy mission: per-BD-dispatch cold-start
amortization calibration. Holds N (words per BD) fixed and sweeps K (number
of back-to-back dma_memcpy_nd dispatches per direction).

The k1 directory is the hand-written template (lock init = 1, single
dispatch per direction). This script generates k2, k4, k8, k16 by mutating
the template:

  - prod_lock init = 1 -> init = K
  - %arg0 / %arg2 memref<N x i32> -> memref<K*N x i32>
  - One dma_memcpy_nd per direction -> K back-to-back dma_memcpy_nd
    (with incrementing offsets and IDs, single dma_wait at the end)
  - test.cpp constants and bo_inA / bo_out sizes
  - run.lit and test.cpp comment headers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MLIR_AIE_TEST_DIR = (
    REPO_ROOT.parent / "mlir-aie" / "test" / "npu-xrt" / "_diag_shim_chain_sweep"
)

DEFAULT_K_LIST = [2, 4, 8, 16]
N = 64  # words per BD; fixed for this sweep


AIE_MLIR_TEMPLATE = """//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// xdna-emu shim BD-chain calibration kernel, K = {K} dispatches of N = {N} i32
// words each.  Used to characterize per-BD-dispatch cold-start amortization
// (or lack thereof) along the K axis.
//
// Data path: ddr -> shim MM2S 0 -> memtile S2MM 0 -> memtile buffer ->
// memtile MM2S 0 -> shim S2MM 0 -> ddr.  Memtile buffer is single-slot,
// reused across all K dispatches via depth-K lock semaphore (prod_lock
// init = K).  After all K MM2S dispatches drain, the buffer contains the
// LAST MM2S BD's data; all K subsequent S2MM dispatches read this same
// slot.  Verification is skipped (this is a calibration, not a correctness
// test) -- bo_out content is predictable but trivial.
//
// Trace anchors (default shim event set):
//   shim DMA_MM2S_0_START_TASK / FINISHED_TASK (fires K times per run)
//   shim DMA_S2MM_0_START_TASK / FINISHED_TASK (fires K times per run)
// Per-task duration = FINISHED_i - START_i.
// Inter-task gap     = START_(i+1) - FINISHED_i.
// Total span         = last FINISHED - first START.
//
//===----------------------------------------------------------------------===//

module {{
  aie.device(NPUDEVICE) {{
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    %loopback_buf = aie.buffer(%tile_0_1) {{sym_name = "loopback_buf"}} : memref<{N}xi32>

    %prod_lock = aie.lock(%tile_0_1, 0) {{init = {K} : i32, sym_name = "prod_lock"}}
    %cons_lock = aie.lock(%tile_0_1, 1) {{init = 0 : i32, sym_name = "cons_lock"}}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.shim_dma_allocation @in (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out (%tile_0_0, S2MM, 0)

    aie.runtime_sequence(%arg0: memref<{TOTAL}xi32>, %arg1: memref<32xi32>, %arg2: memref<{TOTAL}xi32>) {{
{CONSTANTS}
{MM2S_DISPATCHES}
      aiex.npu.dma_wait {{symbol = @in}}
{S2MM_DISPATCHES}
      aiex.npu.dma_wait {{symbol = @out}}
    }}

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {{
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // S2MM 0 receive loop
      aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%loopback_buf : memref<{N}xi32>, 0, {N})
      aie.use_lock(%cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // MM2S 0 send-back loop
      aie.use_lock(%cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%loopback_buf : memref<{N}xi32>, 0, {N})
      aie.use_lock(%prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }}
  }}
}}
"""


TEST_CPP_TEMPLATE = """//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// xdna-emu shim BD-chain calibration launcher (K = {K} dispatches of N = {N}
// i32 words each).  Verifies only that the kernel completed -- content
// verification is skipped (calibration test, not correctness test; the
// single-slot reuse pattern makes bo_out deterministic but trivial).
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int K = {K};
constexpr int N = {N};
constexpr int TOTAL = K * N;

int main(int argc, const char *argv[]) {{
  cxxopts::Options options("shim_chain_k{K}");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {{
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               }});
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, TOTAL * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_inB = xrt::bo(device, N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, TOTAL * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  uint32_t *bufInA = bo_inA.map<uint32_t *>();
  for (int i = 0; i < TOTAL; i++)
    bufInA[i] = static_cast<uint32_t>(i + 1);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {{
    std::cout << "Kernel did not complete. Returned status: " << r << "\\n";
    return 1;
  }}

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << "\\nPASS!\\n\\n";
  return 0;
}}
"""


RUN_LIT_TEMPLATE = """// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// xdna-emu shim BD-chain calibration, K = {K} dispatches of N = {N} i32 words.
// REQUIRES: ryzen_ai
//
// RUN: cp %S/aie.mlir aie_arch.mlir
// RUN: %run_on_npu1% sed 's/NPUDEVICE/npu1_1col/g' -i aie_arch.mlir
// RUN: %run_on_npu2% sed 's/NPUDEVICE/npu2_1col/g' -i aie_arch.mlir
// RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --alloc-scheme=basic-sequential --xclbin-name=aie.xclbin --npu-insts-name=insts.bin ./aie_arch.mlir
// RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
// RUN: %run_on_npu1% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
// RUN: %run_on_npu2% ./test.exe -x aie.xclbin -k MLIR_AIE -i insts.bin
"""


def emit_constants(k: int, n: int) -> str:
    """Emit the arith.constant declarations needed for K dispatches.

    We need 0, 1, n, and i*n for i in 1..K-1 as offsets.
    Duplicates are deduplicated; constants are emitted in sorted order.
    """
    needed = {0, 1, n}
    for i in range(1, k):
        needed.add(i * n)
    lines = [f"      %c{c}_i64 = arith.constant {c} : i64" for c in sorted(needed)]
    return "\n".join(lines)


def emit_dispatches(arg: str, symbol: str, k: int, n: int, id_start: int) -> str:
    """Emit K back-to-back dma_memcpy_nd lines with incrementing offsets/IDs."""
    total = k * n
    lines = []
    for i in range(k):
        offset = i * n
        offset_var = f"%c{offset}_i64"
        line = (
            f"      aiex.npu.dma_memcpy_nd({arg}"
            f"[%c0_i64, %c0_i64, %c0_i64, {offset_var}] "
            f"[%c1_i64, %c1_i64, %c1_i64, %c{n}_i64] "
            f"[%c0_i64, %c0_i64, %c0_i64, %c1_i64]) "
            f"{{id = {id_start + i} : i64, metadata = {symbol}, issue_token = true}} "
            f": memref<{total}xi32>"
        )
        lines.append(line)
    return "\n".join(lines)


def emit_kernel(k: int, n: int) -> str:
    return AIE_MLIR_TEMPLATE.format(
        K=k,
        N=n,
        TOTAL=k * n,
        CONSTANTS=emit_constants(k, n),
        MM2S_DISPATCHES=emit_dispatches("%arg0", "@in", k, n, 0),
        S2MM_DISPATCHES=emit_dispatches("%arg2", "@out", k, n, k),
    )


def emit_variant(k: int, n: int, base_dir: Path) -> Path:
    d = base_dir / f"k{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "aie.mlir").write_text(emit_kernel(k, n))
    (d / "test.cpp").write_text(TEST_CPP_TEMPLATE.format(K=k, N=n))
    (d / "run.lit").write_text(RUN_LIT_TEMPLATE.format(K=k, N=n))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=DEFAULT_K_LIST,
        help=f"K values to generate (default: {DEFAULT_K_LIST})",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=N,
        help=f"Words per BD (default: {N})",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=MLIR_AIE_TEST_DIR,
        help=f"Base directory (default: {MLIR_AIE_TEST_DIR})",
    )
    args = ap.parse_args()

    for k in args.k:
        d = emit_variant(k, args.n, args.base_dir)
        print(f"generated: {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
