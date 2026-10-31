#!/usr/bin/env python3
"""Assemble, package and run the accumulator timing probe kernels.

Mirrors the scripts/isa-test.sh single-tile path (llvm-mc, aiecc, the shared
instr-test-gen test_host) for the kernels written by gen_acc_timing_probe.py.

  python3 tools/run_acc_timing_probe.py build  <dir> [-j N]
  python3 tools/run_acc_timing_probe.py hw     <dir> <tag>     # silicon, one kernel at a time
  python3 tools/run_acc_timing_probe.py emu    <dir> <tag> [-j N]

Outputs land in <dir>/<tag>/<kernel>.bin. Never run two hardware jobs at once.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
instr_gen = importlib.import_module("instr-test-gen")

NPU_WORK = os.environ.get("NPU_WORK_DIR", os.path.expanduser("~/npu-work"))
LLVM_MC = os.path.join(NPU_WORK, "llvm-aie", "build", "bin", "llvm-mc")
MLIR_AIE = os.environ.get("MLIR_AIE", os.path.join(NPU_WORK, "mlir-aie"))
TEST_LIB = os.path.join(MLIR_AIE, "build", "runtime_lib", "x86_64", "test_lib")
XRT = "/opt/xilinx/xrt"
PROGRAM_MEMORY_BYTES = 16384


def load_manifest(d):
    with open(os.path.join(d, "manifest.json")) as f:
        return json.load(f)


def build_host(d) -> str:
    cpp, exe = os.path.join(d, "test_host.cpp"), os.path.join(d, "test_host")
    with open(cpp, "w") as f:
        f.write(instr_gen.generate_test_host_cpp())
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(cpp):
        subprocess.run(
            ["clang++", cpp, "-o", exe, "-std=c++17", "-Wall",
             "-I", f"{TEST_LIB}/include", "-I", f"{XRT}/include",
             "-L", f"{TEST_LIB}/lib", "-L", f"{XRT}/lib",
             "-ltest_utils", "-lxrt_coreutil", "-lrt", "-lstdc++"], check=True)
    return exe


def build_kernel(d, k) -> str:
    name = k["name"]
    src = os.path.join(d, name + ".s")
    kdir = os.path.join(d, name)
    os.makedirs(kdir, exist_ok=True)
    obj = os.path.join(kdir, "kernel.o")
    r = subprocess.run([LLVM_MC, "--triple=aie2", "--filetype=obj", "-o", obj, src],
                       capture_output=True, text=True)
    if r.returncode:
        return f"{name}: ASM FAIL {r.stderr.strip()[:300]}"
    with open(os.path.join(kdir, "aie.mlir"), "w") as f:
        f.write(instr_gen.generate_aie_mlir(k["in_size"], k["out_size"]))
    r = subprocess.run(
        ["aiecc", "--xchesscc=false", "--xbridge=false", "--get-xclbin",
         "--xclbin-name=aie.xclbin", "--get-npu-insts", "--npu-insts-name=insts.bin", "aie.mlir"],
        cwd=kdir, capture_output=True, text=True)
    with open(os.path.join(kdir, "aiecc.log"), "w") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode:
        return f"{name}: PKG FAIL (see {kdir}/aiecc.log)"
    return f"{name}: OK"


def run_kernel(d, k, tag, emu) -> str:
    name = k["name"]
    kdir = os.path.join(d, name)
    out_dir = os.path.join(d, tag)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".bin")
    env = dict(os.environ)
    if emu:
        env.update(XDNA_EMU="1", XDNA_EMU_RUNTIME=os.environ.get("XDNA_EMU_RUNTIME", "debug"),
                   XDNA_EMU_DIR=REPO)
    else:
        for var in ("XDNA_EMU", "XDNA_EMU_RUNTIME"):
            env.pop(var, None)
    cmd = [os.path.join(d, "test_host"), "-x", os.path.join(kdir, "aie.xclbin"), "-k", "MLIR_AIE",
           "-i", os.path.join(kdir, "insts.bin"), "--in-size", str(k["in_size"]),
           "--out-size", str(k["out_size"]), "--in-file", os.path.join(d, "input.bin"),
           "--out-file", out]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60 if emu else 30)
    except subprocess.TimeoutExpired:
        return f"{name}: TIMEOUT"
    with open(os.path.join(out_dir, name + ".log"), "w") as f:
        f.write(r.stdout + r.stderr)
    return f"{name}: {'OK' if r.returncode == 0 else 'FAIL rc=%d' % r.returncode}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "hw", "emu"])
    ap.add_argument("dir")
    ap.add_argument("tag", nargs="?")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="*", help="kernel names")
    args = ap.parse_args(argv)
    d = os.path.abspath(args.dir)
    m = load_manifest(d)
    kernels = [k for k in m["kernels"] if not args.only or k["name"] in args.only]
    build_host(d)
    if args.cmd == "build":
        with cf.ThreadPoolExecutor(args.j) as ex:
            for line in ex.map(lambda k: build_kernel(d, k), kernels):
                print(line, flush=True)
        return 0
    if not args.tag:
        ap.error("tag required")
    if args.cmd == "hw":
        for k in kernels:               # serial: the NPU is a single resource
            line = run_kernel(d, k, args.tag, emu=False)
            print(line, flush=True)
            if "OK" not in line:
                print("stopping at the first hardware failure", file=sys.stderr)
                return 1
        return 0
    with cf.ThreadPoolExecutor(args.j) as ex:
        for line in ex.map(lambda k: run_kernel(d, k, args.tag, emu=True), kernels):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
