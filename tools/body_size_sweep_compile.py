#!/usr/bin/env python3
"""Phase 1 of the BUG-B body-size sweep: compile a family of i8 kernels with a
controlled number of filler ops and measure the resulting loop-body byte span
(le - ls) of the Peano-unrolled core.

Goal: find filler counts that yield loop bodies at 80 / 96 / 112 / 128 bytes
(5 / 6 / 7 / 8 fetch packets), each still parking a partial-word store in the
LE bundle, so Phase 2 can run exactly those on real silicon and confirm the
fetch-packet flush boundary.

No hardware here -- pure compile + disassembly. Run with the NPU env sourced.
Usage: bash -c 'source toolchain-build/activate-npu-env.sh; \
                python3 xdna-emu/tools/body_size_sweep_compile.py'
"""
import os
import subprocess
import sys

ROOT = "/home/triple/npu-work"
EMU = f"{ROOT}/xdna-emu"
PEANO_CLANG = f"{ROOT}/llvm-aie/install/bin/clang"
AIECC = f"{ROOT}/mlir-aie/install/bin/aiecc.py"
PY = sys.executable
TEMPLATE = f"{EMU}/tools/fuzz_template.py"
OBJDUMP = f"{ROOT}/llvm-aie/install/bin/llvm-objdump"
OUTROOT = f"{EMU}/build/experiments/2026-05-31-body-sweep"

# Import the classifier's body-span logic.
sys.path.insert(0, f"{EMU}/tools")
import classify_le_store as cls  # noqa: E402


def kernel_src(n_pairs, ctype="int8_t", size=128):
    """Kernel shaped like seed_1826: repeated stores to buf_out[i] interleaved
    with updates. The final store lands in the LE bundle; n_pairs tunes the
    loop-body size. Each pair is `buf_out[i] = t0; t0 = t0 <op> (i+k);`."""
    ops = ["^", "+", "|", "-", "*", "^", "+", "|", "-", "*", "^", "+"]
    lines = [
        "#include <stdint.h>",
        f"extern \"C\" void fuzz_kernel({ctype}* __restrict buf_in, {ctype}* __restrict buf_out) {{",
        f"    for (int i = 0; i < {size}; i++) {{",
        f"        {ctype} t0 = buf_in[i];",
    ]
    for k in range(n_pairs):
        lines.append(f"        buf_out[i] = t0;")
        lines.append(f"        t0 = t0 {ops[k % len(ops)]} (i + {k + 1});")
    lines += [
        "        buf_out[i] = t0;",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def compile_case(case_dir, size=128, dtype="i8"):
    env = os.environ.copy()
    # Step 1: kernel.cc -> .o (Peano clang)
    r = subprocess.run(
        [PEANO_CLANG, "--target=aie2-none-unknown-elf", "-O2", "-c",
         "fuzz_kernel.cc", "-o", "fuzz_kernel.cc.o"],
        cwd=case_dir, env=env, capture_output=True, text=True,
    )
    if r.returncode:
        return f"clang failed: {r.stderr.strip().splitlines()[-3:]}"
    # Step 2: MLIR template
    r = subprocess.run(
        [PY, TEMPLATE, "--kernel", "fuzz_kernel.cc", "--size", str(size),
         "--dtype", dtype, "--outdir", case_dir, "--device", "npu1_1col", "--trace"],
        cwd=case_dir, env=env, capture_output=True, text=True,
    )
    if r.returncode:
        return f"template failed: {r.stderr.strip().splitlines()[-3:]}"
    # Step 3: aiecc.py -> xclbin (Peano, no chess)
    r = subprocess.run(
        [PY, AIECC, "--no-xchesscc", "--no-xbridge", "--no-aiesim",
         "--aie-generate-xclbin", "--aie-generate-npu-insts", "--no-compile-host",
         "--alloc-scheme=basic-sequential", "--xclbin-name=aie.xclbin",
         "--npu-insts-name=insts.bin", "aie.mlir"],
        cwd=case_dir, env=env, capture_output=True, text=True,
    )
    if r.returncode:
        return f"aiecc failed: {r.stderr.strip().splitlines()[-4:]}"
    if not os.path.exists(os.path.join(case_dir, "aie.xclbin")):
        return "no xclbin produced"
    return None


def main():
    os.makedirs(OUTROOT, exist_ok=True)
    variants = []  # (label, ctype, size, n_pairs)
    for ct, sz in [("int8_t", 128), ("int16_t", 128)]:
        for n in range(0, 12):
            variants.append((f"{'i8' if ct=='int8_t' else 'i16'}_n{n:02d}", ct, sz, n))

    print(f"{'label':<12}{'dtype':<7}{'body_bytes':<12}{'le_store':<9}{'sreg':<6}{'producer':<10}")
    harvest = {}  # body_bytes -> list of (label, dir)
    for label, ct, sz, n in variants:
        case = os.path.join(OUTROOT, label)
        os.makedirs(case, exist_ok=True)
        with open(os.path.join(case, "fuzz_kernel.cc"), "w") as f:
            f.write(kernel_src(n, ct, sz))
        err = compile_case(case, size=sz, dtype=("i8" if ct == "int8_t" else "i16"))
        if err:
            print(f"{label:<12}{'-':<7}{'COMPILE ERR':<12}{err}")
            continue
        elf = os.path.join(case, "aie.mlir.prj", "main_core_0_2.elf")
        info = cls.classify(elf) if os.path.exists(elf) else None
        if not info or not info.get("loop"):
            print(f"{label:<12}{'-':<7}{'no ZOL':<12}")
            continue
        bb = info.get("body_bytes")
        st = info.get("le_store") or "-"
        is_partial = st and st != "full"
        print(f"{label:<12}{('i8' if ct=='int8_t' else 'i16'):<7}{str(bb):<12}{st:<9}"
              f"{str(info.get('sreg') or '-'):<6}{str(info.get('prod_op')):<10}")
        if is_partial and bb is not None:
            harvest.setdefault(bb, []).append((label, case))

    print("\n=== store-at-LE harvest by body_bytes (fetch packets) ===")
    for bb in sorted(harvest):
        print(f"  {bb} bytes ({bb // 16} packets): {[l for l, _ in harvest[bb]]}")


if __name__ == "__main__":
    main()
