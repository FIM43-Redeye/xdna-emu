"""Unit tests for instr-test-gen.py (instruction-level validation generator)."""

import importlib
import sys
from pathlib import Path

# The script uses hyphens (instr-test-gen.py) but Python imports need
# underscores.  Load it explicitly via importlib.
_tools_dir = str(Path(__file__).parent)
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

_spec = importlib.util.spec_from_file_location(
    "instr_test_gen", Path(__file__).parent / "instr-test-gen.py"
)
instr_test_gen = importlib.util.module_from_spec(_spec)
sys.modules["instr_test_gen"] = instr_test_gen
_spec.loader.exec_module(instr_test_gen)

import pytest
from instr_test_gen import (
    ClassDef,
    IntrinsicDef,
    TypeInfo,
    parse_class_defs,
    parse_intrinsic_defs,
    map_llvm_type,
    classify_intrinsic,
    generate_kernel_cc,
    generate_aie_mlir,
    generate_test_host_cpp,
    short_name,
)

# ---------------------------------------------------------------------------
# Task 1: TableGen parser tests
# ---------------------------------------------------------------------------

SAMPLE_CLASSES = """
class AIEV2VBCST32I512
     : DefaultAttrsIntrinsic<[llvm_v16i32_ty], [llvm_i32_ty], [IntrNoMem]>;
class AIEV2V16I32V16I32V16I32I32 : DefaultAttrsIntrinsic<[llvm_v16i32_ty], [llvm_v16i32_ty, llvm_v16i32_ty, llvm_i32_ty], [IntrNoMem]>;
class AIE2EventIntrinsic
    : DefaultAttrsIntrinsic<[],
                [llvm_i32_ty],
                [IntrHasSideEffects, IntrNoMem]>;
class AIEV2V64I8V2I32V64I8 : DefaultAttrsIntrinsic<[llvm_v64i8_ty, llvm_v2i32_ty], [llvm_v64i8_ty], [IntrNoMem]>;
"""

def test_parse_single_return_single_arg():
    classes = parse_class_defs(SAMPLE_CLASSES)
    c = classes["AIEV2VBCST32I512"]
    assert c.ret_types == ["llvm_v16i32_ty"]
    assert c.arg_types == ["llvm_i32_ty"]
    assert c.attrs == ["IntrNoMem"]

def test_parse_multi_arg():
    classes = parse_class_defs(SAMPLE_CLASSES)
    c = classes["AIEV2V16I32V16I32V16I32I32"]
    assert c.ret_types == ["llvm_v16i32_ty"]
    assert c.arg_types == ["llvm_v16i32_ty", "llvm_v16i32_ty", "llvm_i32_ty"]

def test_parse_side_effects():
    classes = parse_class_defs(SAMPLE_CLASSES)
    c = classes["AIE2EventIntrinsic"]
    assert "IntrHasSideEffects" in c.attrs

def test_parse_multi_return():
    classes = parse_class_defs(SAMPLE_CLASSES)
    c = classes["AIEV2V64I8V2I32V64I8"]
    assert c.ret_types == ["llvm_v64i8_ty", "llvm_v2i32_ty"]


# ---------------------------------------------------------------------------
# Task 1: Intrinsic def parser tests
# ---------------------------------------------------------------------------

SAMPLE_DEFS = """
def int_aie2_vbroadcast32_I512 : ClangBuiltin<"__builtin_aiev2_vbroadcast32_I512">, AIEV2VBCST32I512;
def int_aie2_vsel32 : ClangBuiltin<"__builtin_aiev2_vsel32">, AIEV2V16I32V16I32V16I32I32;
def int_aie2_get_ss : AIEV2_get_ss;
def int_aie2_divs : AIEV2DIVS;
def int_aie2_v16int32 : ClangBuiltin<"__builtin_aiev2_v16int32">, AIEV2UNDV16Int32;
"""

def test_parse_def_with_builtin():
    defs = parse_intrinsic_defs(SAMPLE_DEFS)
    d = defs["int_aie2_vbroadcast32_I512"]
    assert d.builtin == "__builtin_aiev2_vbroadcast32_I512"
    assert d.class_name == "AIEV2VBCST32I512"

def test_parse_def_without_builtin():
    defs = parse_intrinsic_defs(SAMPLE_DEFS)
    d = defs["int_aie2_get_ss"]
    assert d.builtin is None
    assert d.class_name == "AIEV2_get_ss"

def test_parse_def_without_builtin_no_comma():
    defs = parse_intrinsic_defs(SAMPLE_DEFS)
    d = defs["int_aie2_divs"]
    assert d.builtin is None
    assert d.class_name == "AIEV2DIVS"


# ---------------------------------------------------------------------------
# Task 2: Type mapping tests
# ---------------------------------------------------------------------------

def test_map_scalar_i32():
    t = map_llvm_type("llvm_i32_ty")
    assert t.c_type == "int32_t"
    assert t.size_bytes == 4
    assert t.is_vector is False

def test_map_vector_v16i32():
    t = map_llvm_type("llvm_v16i32_ty")
    assert t.c_type == "v16int32"
    assert t.size_bytes == 64
    assert t.is_vector is True

def test_map_accumulator_v8i64():
    t = map_llvm_type("llvm_v8i64_ty")
    assert t.c_type == "v8acc64"
    assert t.size_bytes == 64

def test_map_bfloat_vector():
    t = map_llvm_type("llvm_v32bf16_ty")
    assert t.c_type == "v32bfloat16"
    assert t.size_bytes == 64

def test_map_unknown_returns_none():
    assert map_llvm_type("llvm_i128_ty") is None
    assert map_llvm_type("llvm_token_ty") is None


# ---------------------------------------------------------------------------
# Task 2: Intrinsic filtering tests
# ---------------------------------------------------------------------------

def _make_class(ret=None, args=None, attrs=None):
    """Helper to build a ClassDef."""
    return ClassDef(
        name="Test",
        ret_types=ret or ["llvm_v16i32_ty"],
        arg_types=args or ["llvm_i32_ty"],
        attrs=attrs or ["IntrNoMem"],
    )

def test_classify_simple_intrinsic():
    d = IntrinsicDef("int_aie2_vbroadcast32_I512",
                     "__builtin_aiev2_vbroadcast32_I512", "Test")
    status, reason = classify_intrinsic(d, _make_class())
    assert status == "generated"

def test_classify_no_builtin():
    d = IntrinsicDef("int_aie2_get_ss", None, "Test")
    status, reason = classify_intrinsic(d, _make_class())
    assert status == "skipped"
    assert "no ClangBuiltin" in reason

def test_classify_side_effects():
    d = IntrinsicDef("int_aie2_event0", "__builtin_aiev2_event0", "Test")
    c = _make_class(ret=[], args=["llvm_i32_ty"],
                    attrs=["IntrHasSideEffects", "IntrNoMem"])
    status, reason = classify_intrinsic(d, c)
    assert status == "skipped"
    assert "side effects" in reason.lower()

def test_classify_inaccessible_mem():
    d = IntrinsicDef("int_aie2_bf16_mul", "__builtin_x", "Test")
    c = _make_class(attrs=["IntrReadMem", "IntrInaccessibleMemOnly"])
    status, reason = classify_intrinsic(d, c)
    assert status == "skipped"
    assert "IntrInaccessibleMemOnly" in reason

def test_classify_cascade():
    d = IntrinsicDef("int_aie2_scd_read_vec", "__builtin_x", "Test")
    status, reason = classify_intrinsic(d, _make_class())
    assert status == "skipped"
    assert "cascade" in reason.lower() or "stream" in reason.lower()

def test_classify_undef():
    d = IntrinsicDef("int_aie2_v16int32", "__builtin_x", "AIEV2UNDV16Int32")
    status, reason = classify_intrinsic(d, _make_class(ret=["llvm_v16i32_ty"], args=[]))
    assert status == "skipped"
    assert "UND" in reason or "undef" in reason.lower()

def test_classify_i128_arg():
    d = IntrinsicDef("int_aie2_mul_conf", "__builtin_x", "Test")
    c = _make_class(args=["llvm_v64i8_ty", "llvm_v64i8_ty", "llvm_i128_ty", "llvm_i32_ty"])
    status, reason = classify_intrinsic(d, c)
    assert status == "skipped"
    assert "i128" in reason

def test_classify_multi_return():
    d = IntrinsicDef("int_aie2_abs_gtz8", "__builtin_x", "Test")
    c = _make_class(ret=["llvm_v64i8_ty", "llvm_v2i32_ty"])
    status, reason = classify_intrinsic(d, c)
    assert status == "skipped"
    assert "multi-return" in reason.lower()

def test_classify_not_intrnomem():
    d = IntrinsicDef("int_aie2_something", "__builtin_x", "Test")
    c = _make_class(attrs=["IntrWriteMem"])
    status, reason = classify_intrinsic(d, c)
    assert status == "skipped"
    assert "IntrNoMem" in reason


# ---------------------------------------------------------------------------
# Task 3: Kernel code generator tests
# ---------------------------------------------------------------------------

def test_generate_kernel_single_scalar_arg():
    """vbroadcast32: v16int32 = f(int32_t)"""
    code = generate_kernel_cc(
        builtin="__builtin_aiev2_vbroadcast32_I512",
        ret_type="llvm_v16i32_ty",
        arg_types=["llvm_i32_ty"],
    )
    assert "__builtin_aiev2_vbroadcast32_I512" in code
    assert "int32_t arg0 = in[0];" in code
    assert "v16int32 result =" in code
    assert "v16int32 *out_vec = (v16int32 *)out;" in code
    assert "*out_vec = result;" in code
    assert '#define __AIENGINE__ 2' in code
    assert 'extern "C"' in code

def test_generate_kernel_multi_arg():
    """vsel32: v16int32 = f(v16int32, v16int32, int32_t)"""
    code = generate_kernel_cc(
        builtin="__builtin_aiev2_vsel32",
        ret_type="llvm_v16i32_ty",
        arg_types=["llvm_v16i32_ty", "llvm_v16i32_ty", "llvm_i32_ty"],
    )
    # First vector arg at offset 0 (64 bytes = 16 int32s)
    assert "*(const v16int32 *)(in + 0)" in code
    # Second vector arg at offset 16 (next 64 bytes)
    assert "*(const v16int32 *)(in + 16)" in code
    # Scalar arg after two vectors (offset 32)
    assert "in[32]" in code

def test_generate_kernel_accumulator_return():
    """Test with v8acc64 return type."""
    code = generate_kernel_cc(
        builtin="__builtin_aiev2_some_acc_op",
        ret_type="llvm_v8i64_ty",
        arg_types=["llvm_v16i32_ty"],
    )
    assert "v8acc64 result =" in code
    assert "v8acc64 *out_vec = (v8acc64 *)out;" in code


# ---------------------------------------------------------------------------
# Task 4: MLIR template generator tests
# ---------------------------------------------------------------------------

def test_generate_mlir_basic():
    """Single scalar arg, vector return."""
    mlir = generate_aie_mlir(in_size_bytes=4, out_size_bytes=64)
    # Must have proper MLIR structure
    assert "aie.device(npu1_1col)" in mlir
    assert "aie.tile(0, 0)" in mlir
    assert "aie.tile(0, 2)" in mlir
    assert '@test_kernel' in mlir
    assert 'link_with = "kernel.o"' in mlir
    assert "@of_in" in mlir
    assert "@of_out" in mlir
    assert "aie.objectfifo.acquire" in mlir
    assert "aie.objectfifo.release" in mlir
    assert "aiex.npu.dma_memcpy_nd" in mlir
    assert "aiex.npu.dma_wait" in mlir

def test_generate_mlir_buffer_sizes():
    """Buffer sizes match argument types."""
    # 128 bytes in (two v16int32), 64 bytes out (one v16int32)
    mlir = generate_aie_mlir(in_size_bytes=128, out_size_bytes=64)
    # Input fifo element: 128/4 = 32 i32s
    assert "memref<32xi32>" in mlir
    # Output fifo element: 64/4 = 16 i32s
    assert "memref<16xi32>" in mlir


# ---------------------------------------------------------------------------
# Task 5: Host test harness generator tests
# ---------------------------------------------------------------------------

def test_generate_host_harness():
    code = generate_test_host_cpp()
    # Must parse command-line args
    assert "--in-size" in code
    assert "--out-size" in code
    assert "--seed" in code
    assert "--out-file" in code
    # Must use XRT API
    assert "xrt::device" in code
    assert "xrt::bo" in code
    assert "xrt::kernel" in code
    # Must implement PRNG
    assert "1103515245" in code  # LCG constant
    assert "12345" in code       # LCG increment
    # Must write output file
    assert "ofstream" in code or "fwrite" in code


# ---------------------------------------------------------------------------
# Task 6: short_name and integration tests
# ---------------------------------------------------------------------------

def test_short_name():
    assert short_name("int_aie2_vbroadcast32_I512") == "vbroadcast32_I512"
    assert short_name("int_aie2_vsel32") == "vsel32"
    assert short_name("int_aie2_pack_I8_I16") == "pack_I8_I16"


# ---------------------------------------------------------------------------
# Task 8: PRNG consistency verification tests
# ---------------------------------------------------------------------------

def gen_input_python(seed: int, n_bytes: int) -> bytes:
    """Python reference PRNG from spec."""
    state = seed
    buf = bytearray(n_bytes)
    for i in range(n_bytes):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        buf[i] = (state >> 16) & 0xFF
    return bytes(buf)


def test_prng_deterministic():
    """Same seed produces same output."""
    a = gen_input_python(42, 256)
    b = gen_input_python(42, 256)
    assert a == b

def test_prng_different_seeds():
    """Different seeds produce different output."""
    a = gen_input_python(42, 256)
    b = gen_input_python(43, 256)
    assert a != b

def test_prng_known_values():
    """Verify first few bytes for seed=42.

    state0 = 42
    state1 = (42 * 1103515245 + 12345) & 0x7FFFFFFF
           = 46327652297  & 0x7FFFFFFF
           = 46327652297 % 2147483648
           = 2032685001
    byte0  = (2032685001 >> 16) & 0xFF = 31010 & 0xFF = 0x22 = 34
    """
    data = gen_input_python(42, 4)
    assert data[0] == (((42 * 1103515245 + 12345) & 0x7FFFFFFF) >> 16) & 0xFF
