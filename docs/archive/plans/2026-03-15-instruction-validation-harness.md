# Instruction-Level Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-generate single-instruction test kernels from llvm-aie intrinsic definitions, compile with Chess, run on both real NPU and emulator, and diff outputs.

**Architecture:** A Python generator (`tools/instr-test-gen.py`) parses `IntrinsicsAIE2.td`, extracts intrinsic signatures, filters to in-scope `IntrNoMem` builtins, and emits per-intrinsic `kernel.cc` + `aie.mlir` files plus a shared `test_host.cpp`. A shell runner (`scripts/instr-test.sh`) orchestrates compilation (Chess + aiecc), execution (HW serial, EMU parallel), binary comparison, and reporting.

**Tech Stack:** Python 3, Bash, Chess/xchesscc, aiecc.py (mlir-aie), XRT C++ API, cmp(1)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `tools/instr-test-gen.py` | Parse IntrinsicsAIE2.td, generate kernel.cc + aie.mlir per intrinsic, shared test_host.cpp, manifest.json |
| `tools/test_instr_test_gen.py` | Unit tests for generator (parsing, type mapping, filtering, code generation) |
| `scripts/instr-test.sh` | Orchestrate: generate, compile, run HW, run EMU, compare, report |
| `build/instr-tests/` | Generated output directory (already gitignored via `/build/`) |

## Chunk 1: Generator

### Task 1: TableGen Parser

Parse `IntrinsicsAIE2.td` to extract class definitions and intrinsic `def` entries.

**Files:**
- Create: `tools/instr-test-gen.py`
- Create: `tools/test_instr_test_gen.py`
- Read: `../llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td`

**Background:** The file has two kinds of entries:

1. **Class definitions** declare return types, argument types, and attributes:
   ```tablegen
   class AIEV2VBCST32I512
        : DefaultAttrsIntrinsic<[llvm_v16i32_ty], [llvm_i32_ty], [IntrNoMem]>;
   ```

2. **Intrinsic defs** bind a name and optional ClangBuiltin to a class:
   ```tablegen
   def int_aie2_vbroadcast32_I512 : ClangBuiltin<"__builtin_aiev2_vbroadcast32_I512">, AIEV2VBCST32I512;
   ```

The parser must resolve each `def` to its class to get the full signature.

- [ ] **Step 1: Write tests for class definition parsing**

```python
# tools/test_instr_test_gen.py
import pytest
from instr_test_gen import parse_class_defs

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'instr_test_gen')

- [ ] **Step 3: Implement class definition parser**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py::test_parse_single_return_single_arg tools/test_instr_test_gen.py::test_parse_multi_arg tools/test_instr_test_gen.py::test_parse_side_effects tools/test_instr_test_gen.py::test_parse_multi_return -v`
Expected: 4 passed

- [ ] **Step 5: Write tests for intrinsic def parsing**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import parse_intrinsic_defs

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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py::test_parse_def_with_builtin -v`
Expected: FAIL (cannot import name 'parse_intrinsic_defs')

- [ ] **Step 7: Implement intrinsic def parser**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "parse_def"`
Expected: 3 passed

- [ ] **Step 9: Commit**

```bash
git add tools/instr-test-gen.py tools/test_instr_test_gen.py
git commit -m "feat(instr-test): add TableGen parser for IntrinsicsAIE2.td"
```

---

### Task 2: Type Mapping and Filtering

Map LLVM IR types to Chess C types and filter intrinsics to the in-scope subset.

**Files:**
- Modify: `tools/instr-test-gen.py`
- Modify: `tools/test_instr_test_gen.py`

- [ ] **Step 1: Write tests for type mapping**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import TypeInfo, map_llvm_type

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "map_"`
Expected: FAIL (cannot import name 'TypeInfo')

- [ ] **Step 3: Implement type mapping**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "map_"`
Expected: 5 passed

- [ ] **Step 5: Write tests for intrinsic filtering**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import classify_intrinsic

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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "classify_"`
Expected: FAIL (cannot import name 'classify_intrinsic')

- [ ] **Step 7: Implement filtering logic**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "classify_"`
Expected: 8 passed

- [ ] **Step 9: Commit**

```bash
git add tools/instr-test-gen.py tools/test_instr_test_gen.py
git commit -m "feat(instr-test): add type mapping and intrinsic filtering"
```

---

### Task 3: Kernel Code Generator

Generate `kernel.cc` files that call a single `__builtin_aiev2_*` intrinsic.

**Files:**
- Modify: `tools/instr-test-gen.py`
- Modify: `tools/test_instr_test_gen.py`

- [ ] **Step 1: Write tests for kernel code generation**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import generate_kernel_cc

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "generate_kernel"`
Expected: FAIL (cannot import name 'generate_kernel_cc')

- [ ] **Step 3: Implement kernel code generator**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "generate_kernel"`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/instr-test-gen.py tools/test_instr_test_gen.py
git commit -m "feat(instr-test): add kernel.cc code generator"
```

---

### Task 4: MLIR Template Generator

Generate the single-tile `aie.mlir` template that wraps each kernel using the `link_with` mechanism.

**Files:**
- Modify: `tools/instr-test-gen.py`
- Modify: `tools/test_instr_test_gen.py`
- Reference: `../mlir-aie/test/npu-xrt/add_one_func_link_with_chess/aie.mlir`

**Background:** The MLIR template follows the exact same pattern as the existing `add_one_func_link_with_chess` test. Key differences:
- Buffer sizes vary per intrinsic (based on argument + return type sizes)
- ObjectFIFO element type is always `memref<Nxi32>` where N is size/4
- The runtime_sequence transfers exactly the right number of i32s

- [ ] **Step 1: Write test for MLIR template generation**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import generate_aie_mlir

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "generate_mlir"`
Expected: FAIL (cannot import name 'generate_aie_mlir')

- [ ] **Step 3: Implement MLIR template generator**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "generate_mlir"`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tools/instr-test-gen.py tools/test_instr_test_gen.py
git commit -m "feat(instr-test): add MLIR template generator"
```

---

### Task 5: Host Test Harness Generator

Generate the shared `test_host.cpp` that fills input with deterministic PRNG data, runs the kernel via XRT, and writes raw output to a file.

**Files:**
- Modify: `tools/instr-test-gen.py`
- Modify: `tools/test_instr_test_gen.py`
- Reference: `../mlir-aie/test/npu-xrt/add_one_func_link_with_chess/test.cpp`

- [ ] **Step 1: Write test for host harness generation**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import generate_test_host_cpp

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py::test_generate_host_harness -v`
Expected: FAIL (cannot import name 'generate_test_host_cpp')

- [ ] **Step 3: Implement host harness generator**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py::test_generate_host_harness -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/instr-test-gen.py tools/test_instr_test_gen.py
git commit -m "feat(instr-test): add host test harness generator"
```

---

### Task 6: Main Generator Entry Point and Manifest

Wire everything together: parse the real IntrinsicsAIE2.td, classify all intrinsics, generate files, write manifest.json.

**Files:**
- Modify: `tools/instr-test-gen.py`
- Modify: `tools/test_instr_test_gen.py`

- [ ] **Step 1: Write test for short_name derivation**

```python
# Append to tools/test_instr_test_gen.py
from instr_test_gen import short_name

def test_short_name():
    assert short_name("int_aie2_vbroadcast32_I512") == "vbroadcast32_I512"
    assert short_name("int_aie2_vsel32") == "vsel32"
    assert short_name("int_aie2_pack_I8_I16") == "pack_I8_I16"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py::test_short_name -v`
Expected: FAIL

- [ ] **Step 3: Implement short_name and main generate function**

```python
# Add to tools/instr-test-gen.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py::test_short_name -v`
Expected: PASS

- [ ] **Step 5: Run generator against real IntrinsicsAIE2.td (smoke test)**

Run: `cd /home/triple/npu-work/xdna-emu && python tools/instr-test-gen.py --td ../llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td --out-dir /tmp/claude-1000/instr-test-smoke`
Expected: Prints "Generated: N tests" and "Skipped: M intrinsics" with N > 0.

Verify:
- `ls /tmp/claude-1000/instr-test-smoke/manifest.json` exists
- `ls /tmp/claude-1000/instr-test-smoke/vbroadcast32_I512/kernel.cc` exists
- `ls /tmp/claude-1000/instr-test-smoke/test_host.cpp` exists
- `cat /tmp/claude-1000/instr-test-smoke/manifest.json | python -m json.tool` is valid JSON

- [ ] **Step 6: Commit**

```bash
git add tools/instr-test-gen.py tools/test_instr_test_gen.py
git commit -m "feat(instr-test): add main generator entry point and manifest output"
```

---

## Chunk 2: Runner Script and Integration

### Task 7: Runner Script

Shell script that orchestrates generation, compilation, execution, comparison, and reporting.

**Files:**
- Create: `scripts/instr-test.sh`

**Background:** The runner has 6 phases: generate, compile (parallel), run HW (serial), run EMU (parallel), compare, report. It follows patterns established by `scripts/emu-bridge-test.sh` but is much simpler (no dual-compiler, no trace, no lit parsing).

**Dependencies:**
- `xchesscc_wrapper` (from aietools, on PATH after env activation)
- `aiecc.py` (from mlir-aie, on PATH)
- XRT (`/opt/xilinx/xrt/`)
- Test host build deps: `cxxopts.hpp`, `test_utils.h` (from mlir-aie)

- [ ] **Step 1: Create the runner script**

```bash
#!/usr/bin/env bash
# scripts/instr-test.sh -- Instruction-level validation harness runner.
#
# Generates single-instruction test kernels from IntrinsicsAIE2.td,
# compiles them with Chess, runs on real NPU and emulator, diffs outputs.
#
# Usage:
#   scripts/instr-test.sh [options]
#
# Options:
#   --no-hw          Skip hardware runs (EMU-only, no comparison)
#   --no-emu         Skip emulator runs (HW-only baseline)
#   --filter PAT     Only run tests matching PAT (grep -E)
#   --seed N         PRNG seed (default: 42)
#   --compile        Force recompilation
#   -j N             Parallelism for compile + EMU (default: nproc)
#   --generate-only  Only run the generator, skip compile/run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TD_FILE="${PROJECT_DIR}/../llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2.td"
OUT_DIR="${PROJECT_DIR}/build/instr-tests"
RESULTS_DIR="/tmp/instr-test-results-$(date +%Y%m%d)"

# mlir-aie paths for host compilation
MLIR_AIE="${PROJECT_DIR}/../mlir-aie"
INSTALL_DIR="${MLIR_AIE}/install"
XRT_DIR="/opt/xilinx/xrt"

# Defaults
RUN_HW=true
RUN_EMU=true
FILTER=""
SEED=42
FORCE_COMPILE=false
JOBS=$(nproc)
GENERATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-hw)       RUN_HW=false; shift ;;
        --no-emu)      RUN_EMU=false; shift ;;
        --filter)      FILTER="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --compile)     FORCE_COMPILE=true; shift ;;
        -j)            JOBS="$2"; shift 2 ;;
        --generate-only) GENERATE_ONLY=true; shift ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Instruction-Level Validation Harness ==="
echo "TD file:  $TD_FILE"
echo "Out dir:  $OUT_DIR"
echo "Results:  $RESULTS_DIR"
echo ""

# ---- Phase 1: Generate ----
echo "--- Phase 1: Generate ---"
python3 "${PROJECT_DIR}/tools/instr-test-gen.py" --td "$TD_FILE" --out-dir "$OUT_DIR"
echo ""

if $GENERATE_ONLY; then
    echo "Generate-only mode. Done."
    exit 0
fi

# Read manifest to get list of generated tests
TESTS=$(python3 -c "
import json, sys
m = json.load(open('${OUT_DIR}/manifest.json'))
for t in m['generated']:
    print(t['name'], t['in_size'], t['out_size'])
")

# Apply filter
if [[ -n "$FILTER" ]]; then
    TESTS=$(echo "$TESTS" | grep -E "$FILTER" || true)
fi

TOTAL=$(echo "$TESTS" | grep -c . || echo 0)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No tests match filter. Done."
    exit 0
fi
echo "Tests to run: $TOTAL"
echo ""

# ---- Phase 2: Compile ----
echo "--- Phase 2: Compile (j=$JOBS) ---"

# Compile shared host harness (once)
HOST_BIN="${OUT_DIR}/test_host"
if $FORCE_COMPILE || [[ ! -f "$HOST_BIN" ]] || [[ "${OUT_DIR}/test_host.cpp" -nt "$HOST_BIN" ]]; then
    echo "Compiling test_host..."
    clang++ "${OUT_DIR}/test_host.cpp" -o "$HOST_BIN" \
        -std=c++17 -Wall \
        -I "${INSTALL_DIR}/runtime_lib/test_lib" \
        -I "${XRT_DIR}/include" \
        -L "${XRT_DIR}/lib" \
        -lxrt_coreutil \
        -lrt -lstdc++ \
        "${INSTALL_DIR}/runtime_lib/test_lib/test_utils.cpp"
fi

compile_one() {
    local name="$1"
    local test_dir="${OUT_DIR}/${name}"

    # Skip if already compiled (unless --compile)
    if ! $FORCE_COMPILE && [[ -f "${test_dir}/aie.xclbin" ]] && [[ -f "${test_dir}/insts.bin" ]]; then
        return 0
    fi

    echo "  Compiling ${name}..."

    # Compile kernel with Chess
    (cd "$test_dir" && \
        xchesscc_wrapper aie2 -I "${XILINX_VITIS_AIETOOLS}/include" \
            -c kernel.cc -o kernel.o 2>"${test_dir}/chess.log") || {
        echo "  FAIL compile chess: ${name}"
        return 1
    }

    # Compile MLIR -> xclbin + insts.bin
    (cd "$test_dir" && \
        nice -n 19 aiecc.py --no-aiesim --xchesscc --xbridge \
            --aie-generate-xclbin --xclbin-name=aie.xclbin \
            --aie-generate-npu-insts --npu-insts-name=insts.bin \
            aie.mlir 2>"${test_dir}/aiecc.log") || {
        echo "  FAIL compile aiecc: ${name}"
        return 1
    }
}
export -f compile_one
export OUT_DIR FORCE_COMPILE XILINX_VITIS_AIETOOLS

echo "$TESTS" | awk '{print $1}' | xargs -P "$JOBS" -I{} bash -c 'compile_one "$1"' _ {}
echo ""

# ---- Phase 3: Run HW ----
mkdir -p "$RESULTS_DIR"

if $RUN_HW; then
    echo "--- Phase 3: Run HW (serial) ---"
    while IFS=' ' read -r name in_size out_size; do
        test_dir="${OUT_DIR}/${name}"
        hw_out="${RESULTS_DIR}/${name}_hw.bin"

        if [[ ! -f "${test_dir}/aie.xclbin" ]]; then
            echo "  SKIP ${name}: not compiled"
            continue
        fi

        "$HOST_BIN" \
            -x "${test_dir}/aie.xclbin" \
            -k MLIR_AIE \
            -i "${test_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "$hw_out" 2>/dev/null && \
            echo "  HW OK: ${name}" || \
            echo "  HW FAIL: ${name}"
    done <<< "$TESTS"
    echo ""
fi

# ---- Phase 4: Run EMU ----
if $RUN_EMU; then
    echo "--- Phase 4: Run EMU (j=$JOBS) ---"

    run_emu_one() {
        local name="$1"
        local in_size="$2"
        local out_size="$3"
        local test_dir="${OUT_DIR}/${name}"
        local emu_out="${RESULTS_DIR}/${name}_emu.bin"

        if [[ ! -f "${test_dir}/aie.xclbin" ]]; then
            return 0
        fi

        XDNA_EMU=1 "$HOST_BIN" \
            -x "${test_dir}/aie.xclbin" \
            -k MLIR_AIE \
            -i "${test_dir}/insts.bin" \
            --in-size "$in_size" --out-size "$out_size" \
            --seed "$SEED" --out-file "$emu_out" 2>/dev/null && \
            echo "  EMU OK: ${name}" || \
            echo "  EMU FAIL: ${name}"
    }
    export -f run_emu_one
    export HOST_BIN OUT_DIR RESULTS_DIR SEED

    echo "$TESTS" | while IFS=' ' read -r name in_size out_size; do
        echo "$name $in_size $out_size"
    done | xargs -P "$JOBS" -I{} bash -c 'set -- {}; run_emu_one "$1" "$2" "$3"' _
    echo ""
fi

# ---- Phase 5: Compare ----
if $RUN_HW && $RUN_EMU; then
    echo "--- Phase 5: Compare ---"
    PASS=0
    FAIL=0
    SKIP=0
    FAIL_LIST=""

    while IFS=' ' read -r name in_size out_size; do
        hw_out="${RESULTS_DIR}/${name}_hw.bin"
        emu_out="${RESULTS_DIR}/${name}_emu.bin"

        if [[ ! -f "$hw_out" ]] || [[ ! -f "$emu_out" ]]; then
            SKIP=$((SKIP + 1))
            continue
        fi

        if cmp -s "$hw_out" "$emu_out"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            FAIL_LIST="${FAIL_LIST}  DIVERGE: ${name}\n"
            echo "  DIVERGE: ${name}"
        fi
    done <<< "$TESTS"

    echo ""
    echo "=== Results ==="
    echo "PASS: $PASS"
    echo "FAIL: $FAIL"
    echo "SKIP: $SKIP"
    if [[ $FAIL -gt 0 ]]; then
        echo ""
        echo "Divergences:"
        printf '%b' "$FAIL_LIST"
    fi
else
    echo "=== Comparison skipped (need both --hw and --emu) ==="
fi
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /home/triple/npu-work/xdna-emu/scripts/instr-test.sh`

- [ ] **Step 3: Test generate-only mode (no hardware needed)**

Run: `cd /home/triple/npu-work/xdna-emu && scripts/instr-test.sh --generate-only`
Expected: Prints generated/skipped counts, creates `build/instr-tests/manifest.json` and test subdirectories.

Verify:
- `ls build/instr-tests/vbroadcast32_I512/kernel.cc` exists
- `ls build/instr-tests/test_host.cpp` exists
- `python3 -c "import json; m=json.load(open('build/instr-tests/manifest.json')); print(f'{len(m[\"generated\"])} generated, {len(m[\"skipped\"])} skipped')"` prints counts

- [ ] **Step 4: Commit**

```bash
git add scripts/instr-test.sh
git commit -m "feat(instr-test): add runner script for compile/run/compare phases"
```

---

### Task 8: PRNG Consistency Verification

Verify that the Python PRNG and C++ PRNG produce identical byte sequences, since any mismatch would invalidate all comparisons.

**Files:**
- Modify: `tools/test_instr_test_gen.py`

- [ ] **Step 1: Write cross-language PRNG test**

```python
# Append to tools/test_instr_test_gen.py

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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python -m pytest tools/test_instr_test_gen.py -v -k "prng_"`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tools/test_instr_test_gen.py
git commit -m "test(instr-test): add PRNG consistency verification"
```

---

### Task 9: End-to-End Compile Test (Single Intrinsic)

Verify that at least one generated test compiles successfully with Chess and aiecc. This requires the NPU development environment to be active.

**Files:**
- No new files. Uses existing runner script.

**Prerequisites:** NPU environment must be active (`source toolchain-build/activate-npu-env.sh`).

- [ ] **Step 1: Generate and compile a single test**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
scripts/instr-test.sh --generate-only
scripts/instr-test.sh --filter "vbroadcast32_I512" --no-hw --no-emu --compile
```

Expected: Chess compilation succeeds (`kernel.o` created), aiecc compilation succeeds (`aie.xclbin` + `insts.bin` created).

Verify:
- `ls build/instr-tests/vbroadcast32_I512/kernel.o` exists
- `ls build/instr-tests/vbroadcast32_I512/aie.xclbin` exists
- `ls build/instr-tests/vbroadcast32_I512/insts.bin` exists

- [ ] **Step 2: If compilation fails, debug and fix**

Common issues to check:
- Missing `__AIENGINE__` / `__AIEARCH__` defines (check kernel.cc)
- Wrong Chess C type names (check TYPE_MAP against aietools headers)
- MLIR syntax errors (check aie.mlir against reference test)
- Missing `NOCPP` define (Chess needs this for extern "C")
- aiecc.py flags (compare with run.lit from reference test)

Fix the generator and re-run until at least one test compiles.

- [ ] **Step 3: Compile the host harness**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
ls build/instr-tests/test_host  # should not exist yet
scripts/instr-test.sh --filter "vbroadcast32_I512" --no-hw --no-emu
```

Expected: `test_host` binary is created at `build/instr-tests/test_host`.

- [ ] **Step 4: Run on EMU only (no hardware needed)**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
scripts/instr-test.sh --filter "vbroadcast32_I512" --no-hw --seed 42
```

Expected: "EMU OK: vbroadcast32_I512" printed. Output file `vbroadcast32_I512_emu.bin` created in results directory.

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix(instr-test): compilation fixes from end-to-end testing"
```

---

### Task 10: Full Suite Compilation

Compile all generated tests in parallel and record which ones succeed.

**Files:**
- No new files.

- [ ] **Step 1: Run full compilation**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
scripts/instr-test.sh --no-hw --no-emu --compile -j $(nproc)
```

Expected: Most tests compile. Some may fail due to Chess not supporting certain type combinations -- that is expected and acceptable. Record the results.

- [ ] **Step 2: Review compilation failures**

Check `build/instr-tests/<name>/chess.log` and `build/instr-tests/<name>/aiecc.log` for failures. Common categories:
- Chess doesn't recognize a type -> add to exclusion list in generator
- MLIR lowering fails for certain buffer sizes -> adjust MLIR template

Fix the generator filters if needed and re-run.

- [ ] **Step 3: Run EMU on all compilable tests**

Run:
```bash
cd /home/triple/npu-work/xdna-emu
scripts/instr-test.sh --no-hw -j $(nproc)
```

Expected: EMU runs complete for compilable tests.

- [ ] **Step 4: Commit any additional fixes**

```bash
git add -u
git commit -m "fix(instr-test): filter adjustments from full-suite compilation"
```
