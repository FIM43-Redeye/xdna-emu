#!/usr/bin/env python3
"""Generate single-instruction test kernels from llvm-aie intrinsic definitions."""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ClassDef:
    """Parsed TableGen class definition."""
    name: str
    ret_types: list[str]
    arg_types: list[str]
    attrs: list[str]


def parse_class_defs(text: str) -> dict[str, ClassDef]:
    """Parse TableGen class definitions into ClassDef objects.

    Handles multi-line definitions by joining continuation lines before
    matching.  A class definition starts with 'class <Name>' and ends
    at the next ';'.
    """
    classes: dict[str, ClassDef] = {}

    # Collapse multi-line class defs into single lines.
    # Join lines that don't start with 'class ' or 'def ' to the previous line.
    lines = text.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("class ") or stripped.startswith("def ") or stripped.startswith("let "):
            merged.append(stripped)
        elif merged:
            merged[-1] += " " + stripped
        # else: standalone line before any class/def -- ignore

    pattern = re.compile(
        r"class\s+(\w+)\s*"
        r":\s*DefaultAttrsIntrinsic<"
        r"\[([^\]]*)\]"       # return types
        r"\s*,\s*\[([^\]]*)\]"  # arg types
        r"\s*,\s*\[([^\]]*)\]"  # attrs
    )

    for line in merged:
        m = pattern.search(line)
        if m:
            name = m.group(1)
            ret_types = [t.strip() for t in m.group(2).split(",") if t.strip()]
            arg_types = [t.strip() for t in m.group(3).split(",") if t.strip()]
            attrs = [a.strip() for a in m.group(4).split(",") if a.strip()]
            classes[name] = ClassDef(name=name, ret_types=ret_types,
                                     arg_types=arg_types, attrs=attrs)

    return classes


@dataclass
class IntrinsicDef:
    """Parsed intrinsic def entry."""
    name: str
    builtin: str | None
    class_name: str


def parse_intrinsic_defs(text: str) -> dict[str, IntrinsicDef]:
    """Parse 'def int_aie2_*' entries from TableGen source.

    Two forms:
      def NAME : ClangBuiltin<"BUILTIN">, CLASS;
      def NAME : CLASS;
    """
    defs: dict[str, IntrinsicDef] = {}

    # Pattern 1: with ClangBuiltin
    p_builtin = re.compile(
        r'def\s+(int_aie2_\w+)\s*:\s*'
        r'ClangBuiltin<"([^"]+)">\s*,\s*'
        r'(\w+)\s*;'
    )
    # Pattern 2: without ClangBuiltin (class only)
    p_class_only = re.compile(
        r'def\s+(int_aie2_\w+)\s*:\s*'
        r'(?!ClangBuiltin)'  # negative lookahead
        r'(\w+)\s*;'
    )

    for m in p_builtin.finditer(text):
        name, builtin, class_name = m.group(1), m.group(2), m.group(3)
        defs[name] = IntrinsicDef(name=name, builtin=builtin, class_name=class_name)

    for m in p_class_only.finditer(text):
        name, class_name = m.group(1), m.group(2)
        if name not in defs:  # don't overwrite builtin match
            defs[name] = IntrinsicDef(name=name, builtin=None, class_name=class_name)

    return defs


@dataclass
class TypeInfo:
    """Mapped type information for code generation."""
    llvm_type: str
    c_type: str
    size_bytes: int
    is_vector: bool
    # Alignment in int32_t units (for input buffer offset calculation)
    align_i32: int  # size_bytes // 4, minimum 1


# Complete LLVM IR -> Chess C type mapping from spec
TYPE_MAP: dict[str, TypeInfo] = {}

def _add(llvm: str, c: str, size: int, is_vec: bool):
    TYPE_MAP[llvm] = TypeInfo(llvm, c, size, is_vec, max(1, size // 4))

# Scalars
_add("llvm_i32_ty",      "int32_t",       4,  False)
_add("llvm_i64_ty",      "int64_t",       8,  False)
_add("llvm_v2i32_ty",    "int64_t",       8,  False)  # alias
_add("llvm_bfloat_ty",   "bfloat16",      2,  False)
_add("llvm_float_ty",    "float",         4,  False)

# 512-bit vectors (64 bytes)
_add("llvm_v64i8_ty",    "v64int8",       64, True)
_add("llvm_v32i16_ty",   "v32int16",      64, True)
_add("llvm_v16i32_ty",   "v16int32",      64, True)
_add("llvm_v32bf16_ty",  "v32bfloat16",   64, True)
_add("llvm_v16f32_ty",   "v16float",      64, True)
_add("llvm_v8i64_ty",    "v8acc64",       64, True)

# 256-bit vectors (32 bytes)
_add("llvm_v16bf16_ty",  "v16bfloat16",   32, True)
_add("llvm_v8bf16_ty",   "v8bfloat16",    16, True)
_add("llvm_v4i32_ty",    "v4int32",       16, True)
_add("llvm_v8i32_ty",    "v8int32",       32, True)
_add("llvm_v8f32_ty",    "v8float",       32, True)

# 1024-bit vectors (128 bytes)
_add("llvm_v32i32_ty",   "v32int32",      128, True)
_add("llvm_v64i16_ty",   "v64int16",      128, True)
_add("llvm_v128i8_ty",   "v128int8",      128, True)
_add("llvm_v16i64_ty",   "v16acc64",      128, True)
_add("llvm_v32f32_ty",   "v32float",      128, True)
_add("llvm_v64bf16_ty",  "v64bfloat16",   128, True)

# Small vectors
_add("llvm_v4i64_ty",    "v4acc64",       32, True)

del _add  # cleanup namespace


def map_llvm_type(llvm_type: str) -> TypeInfo | None:
    """Map an LLVM IR type string to Chess C type info. Returns None if unmapped."""
    return TYPE_MAP.get(llvm_type)


# Intrinsic name patterns that indicate cascade/stream/lock operations.
INFRA_PATTERNS = re.compile(
    r"int_aie2_(scd_|mcd_|get_ss|put_ms|put_wss|get_wss|"
    r"acquire|release|lock|event|done)"
)


def classify_intrinsic(
    defn: IntrinsicDef,
    class_def: ClassDef,
) -> tuple[str, str]:
    """Classify an intrinsic as 'generated' or 'skipped' with reason.

    Returns (status, reason) where status is 'generated' or 'skipped'.
    """
    # No ClangBuiltin -> cannot call from C
    if defn.builtin is None:
        return ("skipped", "no ClangBuiltin")

    # UND* class -> returns undefined
    if "UND" in defn.class_name:
        return ("skipped", "UND (returns undefined)")

    # Side effects
    if "IntrHasSideEffects" in class_def.attrs:
        return ("skipped", "IntrHasSideEffects (side effects)")

    # IntrInaccessibleMemOnly -> needs config register setup
    if "IntrInaccessibleMemOnly" in class_def.attrs:
        return ("skipped", "IntrInaccessibleMemOnly (implicit config regs)")

    # Cascade/stream/lock infrastructure
    if INFRA_PATTERNS.search(defn.name):
        return ("skipped", "cascade/stream/lock (needs hardware infrastructure)")

    # Multi-return -> out of scope
    if len(class_def.ret_types) > 1:
        return ("skipped", "multi-return (out of scope)")

    # No return type (void) -> nothing to compare
    if len(class_def.ret_types) == 0:
        return ("skipped", "void return (nothing to compare)")

    # Check all types are mapped
    ret_type = class_def.ret_types[0]
    if map_llvm_type(ret_type) is None:
        return ("skipped", f"unmapped return type: {ret_type}")

    for i, arg_type in enumerate(class_def.arg_types):
        if map_llvm_type(arg_type) is None:
            return ("skipped", f"unmapped arg type: {arg_type}")

    # Final guard: must be IntrNoMem (spec requirement)
    if "IntrNoMem" not in class_def.attrs:
        return ("skipped", "not IntrNoMem")

    return ("generated", "")


def generate_kernel_cc(
    builtin: str,
    ret_type: str,
    arg_types: list[str],
) -> str:
    """Generate kernel.cc source that calls one builtin intrinsic.

    Arguments are read from consecutive regions of the input buffer,
    with offsets computed from type sizes.  The result is written to
    the output buffer.
    """
    ret_info = map_llvm_type(ret_type)
    arg_infos = [map_llvm_type(t) for t in arg_types]

    # Build argument signature string for comment
    arg_sig = ", ".join(info.c_type for info in arg_infos)
    sig_comment = f"{ret_info.c_type} = f({arg_sig})" if arg_infos else f"{ret_info.c_type} = f()"

    lines = [
        f"// Auto-generated: tests {builtin}",
        f"// Signature: {sig_comment}",
        "#define __AIENGINE__ 2",
        "#define NOCPP",
        "#define __AIEARCH__ 20",
        "#include <stdint.h>",
        "",
        'extern "C" {',
        "void test_kernel(const int32_t *restrict in, int32_t *restrict out) {",
    ]

    # Generate argument reads from input buffer
    offset_i32 = 0  # current offset in int32_t units
    arg_names = []
    for i, (arg_type, info) in enumerate(zip(arg_types, arg_infos)):
        arg_name = f"arg{i}"
        arg_names.append(arg_name)

        if info.is_vector or info.size_bytes > 4:
            # Cast pointer for vector/large types
            lines.append(f"    {info.c_type} {arg_name} = "
                        f"*(const {info.c_type} *)(in + {offset_i32});")
        else:
            # Scalar read (int32_t or smaller)
            lines.append(f"    {info.c_type} {arg_name} = in[{offset_i32}];")

        offset_i32 += info.align_i32

    # Call intrinsic
    call_args = ", ".join(arg_names)
    lines.append(f"    {ret_info.c_type} result = {builtin}({call_args});")

    # Write result to output buffer
    lines.append(f"    {ret_info.c_type} *out_vec = ({ret_info.c_type} *)out;")
    lines.append("    *out_vec = result;")

    lines.append("}")
    lines.append('} // extern "C"')
    lines.append("")  # trailing newline

    return "\n".join(lines)


def generate_aie_mlir(in_size_bytes: int, out_size_bytes: int) -> str:
    """Generate single-tile MLIR that wraps test_kernel via link_with.

    Follows the pattern from add_one_func_link_with_chess/aie.mlir.
    Buffer sizes are parameterized per intrinsic.
    """
    in_elems = max(1, in_size_bytes // 4)   # memref<Nxi32>
    out_elems = max(1, out_size_bytes // 4)

    return f"""\
module {{
  aie.device(npu1_1col) {{
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {{%tile_0_2}}, 2 : i32) : !aie.objectfifo<memref<{in_elems}xi32>>
    aie.objectfifo @of_out(%tile_0_2, {{%tile_0_0}}, 2 : i32) : !aie.objectfifo<memref<{out_elems}xi32>>

    func.func private @test_kernel(memref<{in_elems}xi32>, memref<{out_elems}xi32>) attributes {{link_with = "kernel.o"}}

    aie.core(%tile_0_2) {{
      %sub_in  = aie.objectfifo.acquire @of_in(Consume, 1)  : !aie.objectfifosubview<memref<{in_elems}xi32>>
      %elem_in = aie.objectfifo.subview.access %sub_in[0]   : !aie.objectfifosubview<memref<{in_elems}xi32>> -> memref<{in_elems}xi32>
      %sub_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<{out_elems}xi32>>
      %elem_out = aie.objectfifo.subview.access %sub_out[0] : !aie.objectfifosubview<memref<{out_elems}xi32>> -> memref<{out_elems}xi32>

      func.call @test_kernel(%elem_in, %elem_out) : (memref<{in_elems}xi32>, memref<{out_elems}xi32>) -> ()

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }}

    aie.runtime_sequence(%in : memref<{in_elems}xi32>, %buf : memref<{in_elems}xi32>, %out : memref<{out_elems}xi32>) {{
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c_in  = arith.constant {in_elems} : i64
      %c_out = arith.constant {out_elems} : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c_out][%c0,%c0,%c0,%c1]) {{metadata = @of_out, id = 1 : i64}} : memref<{out_elems}xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c_in][%c0,%c0,%c0,%c1])  {{metadata = @of_in,  id = 0 : i64, issue_token = true}} : memref<{in_elems}xi32>
      aiex.npu.dma_wait {{symbol = @of_out}}
    }}
  }}
}}
"""


def generate_test_host_cpp() -> str:
    """Generate the shared test_host.cpp.

    Command-line interface:
      ./test_host -x aie.xclbin -i insts.bin \\
          --in-size 256 --out-size 64 --seed 42 --out-file result.bin

    The PRNG matches the spec: LCG with a=1103515245, c=12345, m=2^31.
    """
    return '''\
// Auto-generated host harness for instruction-level validation.
// Usage: ./test_host -x <xclbin> -i <insts> --in-size <N> --out-size <N>
//        --seed <S> --out-file <path>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Deterministic LCG matching the Python generator.
static void fill_prng(uint8_t *buf, size_t n, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        state = (state * 1103515245u + 12345u) & 0x7FFFFFFFu;
        buf[i] = (state >> 16) & 0xFF;
    }
}

int main(int argc, const char *argv[]) {
    cxxopts::Options options("instr-test-host", "Instruction-level validation harness");
    test_utils::add_default_options(options);
    options.add_options()
        ("in-size",  "Input buffer size in bytes",  cxxopts::value<int>())
        ("out-size", "Output buffer size in bytes",  cxxopts::value<int>())
        ("seed",     "PRNG seed",                    cxxopts::value<uint32_t>()->default_value("42"))
        ("out-file", "Output file path",             cxxopts::value<std::string>());

    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);

    int in_size  = vm["in-size"].as<int>();
    int out_size = vm["out-size"].as<int>();
    uint32_t seed = vm["seed"].as<uint32_t>();
    std::string out_file = vm["out-file"].as<std::string>();

    // Round up to i32 alignment for XRT buffer objects.
    int in_elems  = (in_size  + 3) / 4;
    int out_elems = (out_size + 3) / 4;

    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());

    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

    std::string Node = vm["kernel"].as<std::string>();
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
    auto bo_in  = xrt::bo(device, in_elems * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_buf = xrt::bo(device, in_elems * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, out_elems * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Fill input with deterministic PRNG data.
    uint8_t *buf_in = bo_in.map<uint8_t *>();
    fill_prng(buf_in, in_size, seed);

    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in, bo_buf, bo_out);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
        std::cerr << "Kernel did not complete. Status: " << r << std::endl;
        return 1;
    }

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint8_t *buf_out = bo_out.map<uint8_t *>();

    // Write raw output to file.
    std::ofstream ofs(out_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "Cannot open output file: " << out_file << std::endl;
        return 1;
    }
    ofs.write(reinterpret_cast<const char *>(buf_out), out_size);
    ofs.close();

    return 0;
}
'''


def short_name(intrinsic_name: str) -> str:
    """Derive a short directory name from the intrinsic def name.

    'int_aie2_vbroadcast32_I512' -> 'vbroadcast32_I512'
    """
    prefix = "int_aie2_"
    if intrinsic_name.startswith(prefix):
        return intrinsic_name[len(prefix):]
    return intrinsic_name


@dataclass
class GeneratedTest:
    """One generated test case."""
    name: str
    builtin: str
    in_size: int
    out_size: int


@dataclass
class SkippedIntrinsic:
    """One skipped intrinsic."""
    name: str
    reason: str


def generate_all(td_path: str, out_dir: str) -> tuple[list[GeneratedTest], list[SkippedIntrinsic]]:
    """Main entry point: parse TD file, generate all test artifacts.

    Returns (generated, skipped) lists for manifest construction.
    """
    td_text = Path(td_path).read_text()

    classes = parse_class_defs(td_text)
    defs = parse_intrinsic_defs(td_text)

    generated: list[GeneratedTest] = []
    skipped: list[SkippedIntrinsic] = []

    os.makedirs(out_dir, exist_ok=True)

    for name, defn in sorted(defs.items()):
        class_def = classes.get(defn.class_name)
        if class_def is None:
            skipped.append(SkippedIntrinsic(name, f"class {defn.class_name} not found"))
            continue

        status, reason = classify_intrinsic(defn, class_def)
        if status == "skipped":
            skipped.append(SkippedIntrinsic(name, reason))
            continue

        # Compute buffer sizes
        ret_info = map_llvm_type(class_def.ret_types[0])
        arg_infos = [map_llvm_type(t) for t in class_def.arg_types]
        in_size = sum(info.size_bytes for info in arg_infos)
        out_size = ret_info.size_bytes

        # Minimum 4 bytes for each buffer (at least one i32)
        in_size = max(4, in_size)
        out_size = max(4, out_size)

        sname = short_name(name)
        test_dir = os.path.join(out_dir, sname)
        os.makedirs(test_dir, exist_ok=True)

        # Write kernel.cc
        kernel_code = generate_kernel_cc(defn.builtin, class_def.ret_types[0],
                                          class_def.arg_types)
        Path(os.path.join(test_dir, "kernel.cc")).write_text(kernel_code)

        # Write aie.mlir
        mlir_code = generate_aie_mlir(in_size, out_size)
        Path(os.path.join(test_dir, "aie.mlir")).write_text(mlir_code)

        generated.append(GeneratedTest(
            name=sname, builtin=defn.builtin,
            in_size=in_size, out_size=out_size,
        ))

    # Write shared test_host.cpp
    host_code = generate_test_host_cpp()
    Path(os.path.join(out_dir, "test_host.cpp")).write_text(host_code)

    # Write manifest.json
    manifest = {
        "generated": [asdict(g) for g in generated],
        "skipped": [asdict(s) for s in skipped],
    }
    Path(os.path.join(out_dir, "manifest.json")).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-instruction test kernels from IntrinsicsAIE2.td")
    parser.add_argument("--td", required=True,
                        help="Path to IntrinsicsAIE2.td")
    parser.add_argument("--out-dir", default="build/instr-tests",
                        help="Output directory (default: build/instr-tests)")
    args = parser.parse_args()

    generated, skipped = generate_all(args.td, args.out_dir)

    print(f"Generated: {len(generated)} tests")
    print(f"Skipped:   {len(skipped)} intrinsics")
    for s in skipped:
        print(f"  SKIP {s.name}: {s.reason}")


if __name__ == "__main__":
    main()
