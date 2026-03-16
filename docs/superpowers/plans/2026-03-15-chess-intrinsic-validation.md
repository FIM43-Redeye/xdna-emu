# Chess Intrinsic Validation Path Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse Chess-native intrinsic declarations from `me_chess_opns.h` using clang.cindex, generate single-instruction test kernels compiled with xchesscc, and integrate into the existing instruction-level validation harness.

**Architecture:** Three-stage pipeline: (1) pre-process `me_chess_opns.h` to strip Chess extensions while preserving annotations, (2) parse cleaned C++ with clang.cindex using auto-generated type stubs, (3) filter and generate kernel.cc + aie.mlir per testable intrinsic. Reuses the existing runner script with new `--chess`/`--peano` flags.

**Tech Stack:** Python 3.13, clang.cindex (libclang Python bindings from amd-unified-software), xchesscc_wrapper, aiecc.py, XRT

**Spec:** `docs/superpowers/specs/2026-03-15-chess-intrinsic-validation-design.md`

**Existing Peano path (reference):** `tools/instr-test-gen.py`, `scripts/instr-test.sh`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `tools/chess_type_stubs.py` | Parse `chessTraitsOf<T>` from me_chess_types.h, emit `chess_type_stubs.h` |
| `tools/chess_preprocess.py` | Strip Chess extensions from me_chess_opns.h, return (clean_cpp, annotations) |
| `tools/chess-test-gen.py` | Main generator: orchestrates pipeline, walks AST, filters, generates output |
| `tools/test_chess_test_gen.py` | Unit tests for all three modules above |
| `scripts/instr-test.sh` | Modified: add `--chess`/`--peano`/`--both` flags, compiler-specific paths |

---

## Chunk 1: Type Stubs and Pre-processor

### Task 1: Type Stub Generator (`chess_type_stubs.py`)

**Files:**
- Create: `tools/chess_type_stubs.py`
- Test: `tools/test_chess_test_gen.py`
- Read: `../aietools/data/aie_ml/lib/isg/me_chess_types.h` (lines 17569-18170)

The `chessTraitsOf<T>` specializations have two formats:
```cpp
// Vector/custom types: literal bit count
template <> struct chessTraitsOf<v16int32> {
    static const unsigned bits = 512;
    static const unsigned elems = 16;
};

// Primitive C types: sizeof expression (with comment showing target size)
template <> struct chessTraitsOf<int> {
    static const unsigned bits = sizeof(int) * __CHAR_BIT__; // pertinent to host, may differ from target 32;
};
```

For primitive C types, we extract the target size from the trailing comment
(e.g., `// ... target 32;` -> 32 bits). For vector/custom types, we parse
the literal integer directly.

- [ ] **Step 1: Write failing tests for chessTraitsOf parser**

```python
# tools/test_chess_test_gen.py
import importlib
import pytest

# chess_type_stubs has a hyphen-free name, direct import works
from chess_type_stubs import parse_chess_traits, generate_stub_header

class TestChessTypeStubs:
    def test_parse_vector_type(self):
        """Parse a chessTraitsOf with literal bits."""
        text = """template <> struct chessTraitsOf<v16int32> {
    static const unsigned bits = 512;
    static const unsigned elems = 16;
};"""
        traits = parse_chess_traits(text)
        assert traits["v16int32"] == 512

    def test_parse_primitive_type_with_sizeof(self):
        """Parse a chessTraitsOf with sizeof expression + target comment."""
        text = """template <> struct chessTraitsOf<int> {
    static const unsigned bits = sizeof(int) * __CHAR_BIT__; // pertinent to host, may differ from target 32;
};"""
        traits = parse_chess_traits(text)
        assert traits["int"] == 32

    def test_skip_commented_out(self):
        """Commented-out //! traits should be skipped."""
        text = """//!template <> struct chessTraitsOf<void *> {
//!    static const unsigned bits = 64;
//!};
template <> struct chessTraitsOf<v8acc64> {
    static const unsigned bits = 512;
};"""
        traits = parse_chess_traits(text)
        assert "void *" not in traits
        assert traits["v8acc64"] == 512

    def test_generate_stub_header(self):
        """Stub header has correct struct sizes."""
        traits = {"v16int32": 512, "bfloat16": 16, "mask64": 64}
        header = generate_stub_header(traits)
        assert "struct v16int32 { char _data[64]; };" in header
        assert "struct bfloat16 { char _data[2]; };" in header
        assert "struct mask64 { char _data[8]; };" in header

    def test_skip_builtin_c_types(self):
        """Built-in C types (int, float, etc.) should not get struct stubs."""
        traits = {"int": 32, "float": 32, "v16int32": 512}
        header = generate_stub_header(traits)
        assert "struct int " not in header
        assert "struct float " not in header
        assert "struct v16int32" in header

    def test_parse_accumulator_type(self):
        """Accumulator types have bits but may not have elems."""
        text = """template <> struct chessTraitsOf<v1acc32> {
    static const unsigned bits = 32;
};"""
        traits = parse_chess_traits(text)
        assert traits["v1acc32"] == 32

    def test_parse_real_file(self):
        """Integration: parse the actual me_chess_types.h file."""
        from pathlib import Path
        types_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_types.h"
        if not types_path.exists():
            pytest.skip("aietools not available")
        text = types_path.read_text()
        traits = parse_chess_traits(text)
        # Spot-check known types
        assert traits["v16int32"] == 512
        assert traits["v8acc64"] == 512
        assert traits["bfloat16"] == 16
        assert traits["v64int8"] == 512
        assert len(traits) >= 100  # should have 150+ types
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessTypeStubs -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chess_type_stubs'`

- [ ] **Step 3: Implement chess_type_stubs.py**

```python
#!/usr/bin/env python3
"""Extract type sizes from me_chess_types.h and generate C struct stubs.

Parses chessTraitsOf<T> specializations to get bit widths, then emits a
minimal C header where each Chess type is a struct with the correct size.
This header is consumed by clang.cindex when parsing me_chess_opns.h.
"""

import re
import sys
from pathlib import Path

# C built-in types that should NOT get struct stubs (clang knows them).
BUILTIN_C_TYPES = frozenset({
    "bool", "char", "signed char", "unsigned char",
    "short", "unsigned short", "int", "unsigned",
    "long", "unsigned long", "long long", "unsigned long long",
    "float", "double", "long double",
})


def parse_chess_traits(text: str) -> dict[str, int]:
    """Parse chessTraitsOf<T> specializations, returning {type_name: bits}.

    Handles two formats:
    - Literal: `static const unsigned bits = 512;`
    - sizeof:  `static const unsigned bits = sizeof(int) * __CHAR_BIT__; // ... target 32;`

    Skips commented-out lines (//! prefix).
    """
    traits: dict[str, int] = {}

    # Match: template <> struct chessTraitsOf<TYPE> {
    #            static const unsigned bits = VALUE;
    # where VALUE is either a literal int or a sizeof expression with a comment.
    pattern = re.compile(
        r'^\s*template\s*<>\s*struct\s+chessTraitsOf<([^>]+)>\s*\{',
        re.MULTILINE,
    )
    bits_literal = re.compile(
        r'static\s+const\s+unsigned\s+bits\s*=\s*(\d+)\s*;'
    )
    bits_sizeof = re.compile(
        r'static\s+const\s+unsigned\s+bits\s*=\s*sizeof\([^)]+\)\s*\*\s*__CHAR_BIT__\s*;'
        r'\s*//.*?target\s+(\d+)\s*;'
    )

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip commented-out traits
        if line.lstrip().startswith('//!'):
            i += 1
            continue

        m = pattern.match(line)
        if m:
            type_name = m.group(1).strip()
            # Collect the block (up to closing brace)
            block = line
            j = i + 1
            while j < len(lines) and '};' not in block:
                block += '\n' + lines[j]
                j += 1

            # Try literal match first
            bm = bits_literal.search(block)
            if bm:
                traits[type_name] = int(bm.group(1))
            else:
                # Try sizeof with target comment
                bm = bits_sizeof.search(block)
                if bm:
                    traits[type_name] = int(bm.group(1))

            i = j + 1
        else:
            i += 1

    return traits


def generate_stub_header(traits: dict[str, int]) -> str:
    """Generate a C header with sized struct stubs for each Chess type.

    Built-in C types (int, float, etc.) are skipped -- clang knows them.
    """
    lines = [
        "// Auto-generated type stubs for clang.cindex parsing of me_chess_opns.h",
        "// Source: me_chess_types.h chessTraitsOf<T> specializations",
        "// DO NOT EDIT -- regenerate with: python3 tools/chess_type_stubs.py",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]

    for type_name, bits in sorted(traits.items()):
        if type_name in BUILTIN_C_TYPES:
            continue
        byte_size = max(1, bits // 8)
        lines.append(f"struct {type_name} {{ char _data[{byte_size}]; }};  "
                      f"// {bits} bits")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def main():
    """CLI: parse me_chess_types.h and emit chess_type_stubs.h."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate type stubs from me_chess_types.h")
    parser.add_argument("--types-header",
                        default="../aietools/data/aie_ml/lib/isg/me_chess_types.h",
                        help="Path to me_chess_types.h")
    parser.add_argument("--output", default=None,
                        help="Output path (default: stdout)")
    args = parser.parse_args()

    text = Path(args.types_header).read_text()
    traits = parse_chess_traits(text)
    header = generate_stub_header(traits)

    if args.output:
        Path(args.output).write_text(header)
        print(f"Wrote {len(traits)} type stubs to {args.output}")
    else:
        print(header)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessTypeStubs -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/chess_type_stubs.py tools/test_chess_test_gen.py
git commit -m "feat(instr-test): add Chess type stub generator from chessTraitsOf"
```

---

### Task 2: Chess Extension Pre-processor (`chess_preprocess.py`)

**Files:**
- Create: `tools/chess_preprocess.py`
- Test: `tools/test_chess_test_gen.py` (append new test class)

- [ ] **Step 1: Write failing tests for pre-processor**

Annotations are keyed by **function name** (not line number) to avoid
line-number drift between the pre-processing pass and the clang.cindex
AST walk. The pre-processor uses a two-pass approach: first pass extracts
annotations from the original text, second pass strips extensions.

```python
# Append to tools/test_chess_test_gen.py
from chess_preprocess import preprocess_chess_header, ChessAnnotation

class TestChessPreprocess:
    def test_strip_chess_property(self):
        """chess_property(...) is stripped, annotation recorded by func name."""
        text = 'mod_t undef_mod() chess_property(dont_care);\n'
        clean, annotations = preprocess_chess_header(text)
        assert "chess_property" not in clean
        assert "mod_t undef_mod();" in clean
        assert "undef_mod" in annotations
        assert "dont_care" in annotations["undef_mod"].properties

    def test_strip_chess_property_multiword(self):
        """Multi-word chess_property is split into word list."""
        text = 'void acquire_guarded(unsigned, unsigned) chess_property(guarded_memory_fence volatile output_stage_offset_7);\n'
        clean, annotations = preprocess_chess_header(text)
        assert "chess_property" not in clean
        ann = annotations["acquire_guarded"]
        assert "guarded_memory_fence" in ann.properties
        assert "volatile" in ann.properties

    def test_strip_chess_storage_in_param(self):
        """chess_storage on parameter type is stripped, recorded."""
        text = 'void acquire_equal_inner(const void *a, char chess_storage(TM) *mem);\n'
        clean, annotations = preprocess_chess_header(text)
        assert "chess_storage" not in clean
        ann = annotations["acquire_equal_inner"]
        assert "TM" in ann.storage_params

    def test_strip_if0_block(self):
        """#if 0//! ... #endif//! blocks are removed entirely."""
        text = '''some_func();
#if 0//!
namespace me_primitive {
    void hidden_func();
} //namespace me_primitive
#endif//!
another_func();
'''
        clean, _ = preprocess_chess_header(text)
        assert "hidden_func" not in clean
        assert "some_func" in clean
        assert "another_func" in clean

    def test_strip_bang_comment_lines(self):
        """Lines starting with //! are removed."""
        text = '//!v256uint4_sparse sparse_pop_aux(...);\nreal_func();\n'
        clean, _ = preprocess_chess_header(text)
        assert "sparse_pop_aux" not in clean
        assert "real_func" in clean

    def test_strip_chess_protect_access(self):
        """chess_protect_access is stripped."""
        text = 'extern chess_protect_access v16acc32 chess_storage(SCD) scd;\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_protect_access" not in clean

    def test_replace_vbitzconstexpr(self):
        """VBITzCONSTEXPR is replaced with inline."""
        text = 'VBITzCONSTEXPR inline cint32(int, int) chess_property(do_generate);\n'
        clean, _ = preprocess_chess_header(text)
        assert "VBITzCONSTEXPR" not in clean
        assert "inline" in clean

    def test_strip_chess_manifest(self):
        """chess_manifest(...) calls are stripped."""
        text = 'if (chess_manifest(idx < 0 || idx > 3)) ;\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_manifest" not in clean

    def test_strip_chess_dont_warn_dead(self):
        text = 'chess_dont_warn_dead(cmp);\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_dont_warn_dead" not in clean

    def test_strip_chess_memory_fence(self):
        text = 'chess_memory_fence();\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_memory_fence" not in clean

    def test_strip_chess_separator_scheduler(self):
        text = 'chess_separator_scheduler();\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_separator_scheduler" not in clean

    def test_strip_chess_unroll_loop(self):
        text = 'for (int n = 0; n < 3; n++) chess_unroll_loop(*)\n'
        clean, _ = preprocess_chess_header(text)
        assert "chess_unroll_loop" not in clean

    def test_property_word_splitting(self):
        """Properties are stored as word lists for filter matching."""
        text = 'void f() chess_property(functional loop_free);\n'
        _, annotations = preprocess_chess_header(text)
        ann = annotations["f"]
        assert "functional" in ann.properties
        assert "loop_free" in ann.properties

    def test_ifdef_chess_error_stripped(self):
        text = '''#ifdef __chess__
#error "generated native file not intended for compilation by chess"
#endif
void real_func();
'''
        clean, _ = preprocess_chess_header(text)
        assert '#error' not in clean
        assert "real_func" in clean

    def test_multiple_annotations_same_name(self):
        """Overloaded functions accumulate properties under same name."""
        text = '''int f(int) chess_property(volatile);
int f(int, int) chess_property(functional);
'''
        _, annotations = preprocess_chess_header(text)
        # Both properties should be recorded (accumulated)
        ann = annotations["f"]
        assert "volatile" in ann.properties
        assert "functional" in ann.properties

    def test_integration_real_file(self):
        """Integration: pre-process the actual me_chess_opns.h."""
        from pathlib import Path
        opns_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_opns.h"
        if not opns_path.exists():
            pytest.skip("aietools not available")
        text = opns_path.read_text()
        clean, annotations = preprocess_chess_header(text)
        assert "chess_property(" not in clean
        assert "chess_storage(" not in clean
        assert "chess_manifest(" not in clean
        assert "chess_memory_fence()" not in clean
        assert "VBITzCONSTEXPR" not in clean
        assert len(annotations) > 50
        # Verify dont_care annotations exist
        dont_care_funcs = [
            name for name, a in annotations.items()
            if "dont_care" in a.properties
        ]
        assert len(dont_care_funcs) >= 10  # many undef_* functions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessPreprocess -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chess_preprocess'`

- [ ] **Step 3: Implement chess_preprocess.py**

```python
#!/usr/bin/env python3
"""Pre-process me_chess_opns.h to strip Chess-specific extensions.

Uses a two-pass approach:
  Pass 1: Extract chess_property and chess_storage annotations from the
           ORIGINAL text, keyed by function name (not line number).
  Pass 2: Strip all Chess extensions to produce clean C++ for clang.cindex.

Annotations are keyed by function name to avoid line-number drift between
the pre-processed source and the clang.cindex AST.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ChessAnnotation:
    """Annotations extracted from a function declaration."""
    func_name: str
    properties: list[str] = field(default_factory=list)
    storage_params: list[str] = field(default_factory=list)


def _extract_func_name(text_before: str) -> str:
    """Extract the function name from text preceding chess_property/storage."""
    m = re.search(r'(\w+)\s*\([^)]*\)\s*$', text_before.rstrip())
    return m.group(1) if m else "_unknown"


def preprocess_chess_header(text: str) -> tuple[str, dict[str, ChessAnnotation]]:
    """Strip Chess extensions from header source.

    Returns (clean_cpp, annotations) where annotations is keyed by
    function name. Overloaded functions accumulate annotations under the
    same key.
    """
    annotations: dict[str, ChessAnnotation] = {}

    # -- Pass 1: Extract annotations from ORIGINAL text --

    # Extract chess_property annotations
    for m in re.finditer(r'\s*chess_property\(([^)]+)\)', text):
        prop_words = m.group(1).strip().split()
        func_name = _extract_func_name(text[:m.start()])
        ann = annotations.setdefault(
            func_name, ChessAnnotation(func_name=func_name),
        )
        ann.properties.extend(prop_words)

    # Extract chess_storage annotations (parameter-level only)
    # Look for chess_storage inside function parameter lists
    for m in re.finditer(r'chess_storage\(([^)]+)\)', text):
        storage_name = m.group(1).strip()
        # Only record if inside a function parameter list
        before = text[:m.start()]
        func_name = _extract_func_name(before)
        if func_name != "_unknown":
            ann = annotations.setdefault(
                func_name, ChessAnnotation(func_name=func_name),
            )
            ann.storage_params.append(storage_name)

    # -- Pass 2: Strip all Chess extensions --

    # Remove #if 0//! ... #endif//! blocks
    text = re.sub(
        r'#if\s+0\s*//!.*?#endif\s*//!',
        '',
        text,
        flags=re.DOTALL,
    )

    # Remove //! comment lines
    text = re.sub(r'^[ \t]*//!.*$', '', text, flags=re.MULTILINE)

    # Remove #ifdef __chess__ / #error / #endif guard
    text = re.sub(
        r'#ifdef\s+__chess__\s*\n#error\s+[^\n]*\n#endif\s*\n?',
        '',
        text,
    )

    # Strip chess_property(...), chess_storage(...)
    text = re.sub(r'\s*chess_property\([^)]+\)', '', text)
    text = re.sub(r'\s*chess_storage\([^)]+\)\s*', ' ', text)

    # Strip remaining Chess extensions
    text = re.sub(r'\bchess_protect_access\b', '', text)
    text = re.sub(r'\bchess_manifest\([^)]*\)', '(1)', text)
    text = re.sub(r'\bchess_memory_fence\(\)', '((void)0)', text)
    text = re.sub(r'\bchess_separator_scheduler\(\)', '((void)0)', text)
    text = re.sub(r'\bchess_dont_warn_dead\([^)]*\)', '((void)0)', text)
    text = re.sub(r'\bchess_unroll_loop\(\*\)', '', text)
    text = re.sub(r'\bVBITzCONSTEXPR\b', 'inline', text)

    return text, annotations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessPreprocess -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/chess_preprocess.py tools/test_chess_test_gen.py
git commit -m "feat(instr-test): add Chess extension pre-processor with annotation extraction"
```

---

### Task 3: Verify clang.cindex Can Parse Pre-processed Header

**Files:**
- Test: `tools/test_chess_test_gen.py` (append)
- Read: `tools/chess_type_stubs.py`, `tools/chess_preprocess.py`

This task validates that the pre-processor + type stubs produce valid C++
that clang.cindex can parse end-to-end. No generator code yet -- just
proving the parsing pipeline works.

- [ ] **Step 1: Write integration test for clang.cindex parsing**

```python
# Append to tools/test_chess_test_gen.py
import tempfile

class TestClangParsing:
    def test_parse_simple_stub_and_declaration(self):
        """clang.cindex can parse a type stub + function declaration."""
        import clang.cindex

        source = """
struct v16int32 { char _data[64]; };
struct v64int8 { char _data[64]; };

v16int32 broadcast_to_v16int32(int);
v64int8 some_vector_op(v16int32, v16int32, int);
"""
        index = clang.cindex.Index.create()
        tu = index.parse(
            "test.cpp", unsaved_files=[("test.cpp", source)],
            args=["-std=c++17", "-fsyntax-only"],
        )
        # Should have no errors
        errors = [d for d in tu.diagnostics
                  if d.severity >= clang.cindex.Diagnostic.Error]
        assert len(errors) == 0, f"Parse errors: {[d.spelling for d in errors]}"

        # Should find function declarations
        funcs = []
        for cursor in tu.cursor.get_children():
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                funcs.append(cursor.spelling)
        assert "broadcast_to_v16int32" in funcs
        assert "some_vector_op" in funcs

    def test_parse_namespace_declarations(self):
        """clang.cindex handles me_primitive namespace blocks."""
        import clang.cindex

        source = """
struct v512w8 { char _data[64]; };
struct v32w32 { char _data[128]; };
struct pmode_t { char _data[4]; };
struct smode_t { char _data[4]; };

namespace me_primitive {
v512w8 prmx_hw_prom(v32w32, pmode_t, smode_t);
} //namespace me_primitive
namespace me_primitive {
int some_other_prim(int, int);
} //namespace me_primitive
"""
        index = clang.cindex.Index.create()
        tu = index.parse(
            "test.cpp", unsaved_files=[("test.cpp", source)],
            args=["-std=c++17", "-fsyntax-only"],
        )
        errors = [d for d in tu.diagnostics
                  if d.severity >= clang.cindex.Diagnostic.Error]
        assert len(errors) == 0, f"Parse errors: {[d.spelling for d in errors]}"

    def test_parse_overloaded_functions(self):
        """clang.cindex distinguishes overloaded function signatures."""
        import clang.cindex

        source = """
struct v16int32 { char _data[64]; };
struct v16acc64 { char _data[128]; };
struct v32int16 { char _data[64]; };
struct v64uint8 { char _data[64]; };

v16acc64 mul_2x8_8x8(v32int16 a, v64uint8 b);
v16acc64 mul_2x8_8x8(v32int16 a, int sgn_x, v64uint8 b, int sgn_y);
"""
        index = clang.cindex.Index.create()
        tu = index.parse(
            "test.cpp", unsaved_files=[("test.cpp", source)],
            args=["-std=c++17", "-fsyntax-only"],
        )
        funcs = []
        for cursor in tu.cursor.get_children():
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                params = [c.type.spelling for c in cursor.get_children()
                          if c.kind == clang.cindex.CursorKind.PARM_DECL]
                funcs.append((cursor.spelling, params))

        mul_overloads = [f for f in funcs if f[0] == "mul_2x8_8x8"]
        assert len(mul_overloads) == 2
        # 2-arg and 4-arg variants
        param_counts = sorted(len(f[1]) for f in mul_overloads)
        assert param_counts == [2, 4]

    def test_parse_preprocessed_real_header(self):
        """Integration: pre-process + stub + parse the real me_chess_opns.h."""
        from pathlib import Path
        from chess_preprocess import preprocess_chess_header
        from chess_type_stubs import parse_chess_traits, generate_stub_header

        opns_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_opns.h"
        types_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_types.h"
        if not opns_path.exists() or not types_path.exists():
            pytest.skip("aietools not available")

        import clang.cindex

        # Generate stubs
        traits = parse_chess_traits(types_path.read_text())
        stubs = generate_stub_header(traits)

        # Pre-process header
        clean, annotations = preprocess_chess_header(opns_path.read_text())

        # Combine: stubs + cleaned header
        combined = stubs + "\n" + clean

        index = clang.cindex.Index.create()
        tu = index.parse(
            "me_chess_opns_clean.cpp",
            unsaved_files=[("me_chess_opns_clean.cpp", combined)],
            args=["-std=c++17", "-fsyntax-only"],
        )

        # Count errors (warnings are OK)
        errors = [d for d in tu.diagnostics
                  if d.severity >= clang.cindex.Diagnostic.Error]
        # We may get some errors from unhandled Chess extensions;
        # the test passes if we can extract a meaningful number of functions
        func_count = 0
        def walk(cursor):
            nonlocal func_count
            if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                func_count += 1
            for child in cursor.get_children():
                walk(child)
        walk(tu.cursor)

        # Should find hundreds of functions even with some parse errors
        assert func_count >= 500, (
            f"Only found {func_count} functions, expected 500+. "
            f"Errors: {len(errors)}"
        )
        print(f"Parsed {func_count} functions, {len(errors)} errors, "
              f"{len(annotations)} annotations")
```

- [ ] **Step 2: Run tests to verify they fail (or pass -- this is exploratory)**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestClangParsing -v -s`
Expected: The first 3 synthetic tests should PASS. The real-file integration
test may reveal pre-processor gaps that need fixing.

- [ ] **Step 3: Fix any pre-processor gaps revealed by the integration test**

If `test_parse_preprocessed_real_header` fails because clang.cindex hits
Chess extensions that survived pre-processing, add the missing patterns to
`chess_preprocess.py` and update its unit tests. Iterate until the
integration test finds 500+ functions.

- [ ] **Step 4: Run all tests**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/test_chess_test_gen.py tools/chess_preprocess.py
git commit -m "test(instr-test): verify clang.cindex parses pre-processed Chess header"
```

---

## Chunk 2: Generator, Filter, Runner Integration

### Task 4: AST Walker and Filter (`chess-test-gen.py`)

**Files:**
- Create: `tools/chess-test-gen.py`
- Test: `tools/test_chess_test_gen.py` (append)
- Read: `tools/chess_preprocess.py`, `tools/chess_type_stubs.py`

This is the core generator. It orchestrates the full pipeline: stub
generation -> pre-processing -> clang.cindex parsing -> AST walking ->
filtering -> code generation.

- [ ] **Step 1: Write failing tests for AST walker and filter**

```python
# Append to tools/test_chess_test_gen.py
import importlib
chess_test_gen = importlib.import_module("chess-test-gen")

class TestAnnotationJoining:
    def test_annotation_reaches_filter(self):
        """End-to-end: chess_property(dont_care) -> preprocess -> walk -> filter = skipped."""
        from chess_preprocess import preprocess_chess_header

        original = """
struct v16int32 { char _data[64]; };
v16int32 undef_v16int32() chess_property(dont_care);
v16int32 broadcast_to_v16int32(int);
"""
        clean, annotations = preprocess_chess_header(original)
        assert "undef_v16int32" in annotations
        assert "dont_care" in annotations["undef_v16int32"].properties

        # Prepend type stubs (already in source) and parse
        intrinsics = chess_test_gen.walk_ast(clean, annotations)
        assert len(intrinsics) == 2

        # undef should be skipped, broadcast should pass
        for i in intrinsics:
            status, reason = chess_test_gen.classify_chess_intrinsic(i)
            if i.name == "undef_v16int32":
                assert status == "skipped", f"Expected skipped, got {status}: {reason}"
                assert "dont_care" in reason
            elif i.name == "broadcast_to_v16int32":
                assert status == "generated", f"Expected generated, got {status}: {reason}"

    def test_volatile_annotation_reaches_filter(self):
        """chess_property(volatile) -> preprocess -> walk -> filter = skipped."""
        from chess_preprocess import preprocess_chess_header

        original = """
void acquire_guarded(unsigned, unsigned) chess_property(volatile);
"""
        clean, annotations = preprocess_chess_header(original)
        intrinsics = chess_test_gen.walk_ast(clean, annotations)
        assert len(intrinsics) == 1
        status, _ = chess_test_gen.classify_chess_intrinsic(intrinsics[0])
        assert status == "skipped"


class TestChessASTWalker:
    def test_walk_simple_function(self):
        """Extract a simple function from source."""
        source = """
struct v16int32 { char _data[64]; };
v16int32 broadcast_to_v16int32(int);
"""
        intrinsics = chess_test_gen.walk_ast(source)
        assert len(intrinsics) == 1
        i = intrinsics[0]
        assert i.name == "broadcast_to_v16int32"
        assert i.return_type == "v16int32"
        assert i.return_size == 64
        assert i.params == [("int", 4)]
        assert i.namespace == ""

    def test_walk_namespace(self):
        """Extract me_primitive namespace function."""
        source = """
struct v512w8 { char _data[64]; };
struct v32w32 { char _data[128]; };
struct pmode_t { char _data[4]; };
struct smode_t { char _data[4]; };
namespace me_primitive {
v512w8 prmx_hw_prom(v32w32, pmode_t, smode_t);
} //namespace me_primitive
"""
        intrinsics = chess_test_gen.walk_ast(source)
        assert len(intrinsics) == 1
        assert intrinsics[0].namespace == "me_primitive"

    def test_walk_overloads(self):
        """Overloaded functions get separate entries."""
        source = """
struct v16acc64 { char _data[128]; };
struct v32int16 { char _data[64]; };
struct v64uint8 { char _data[64]; };
v16acc64 mul_2x8_8x8(v32int16, v64uint8);
v16acc64 mul_2x8_8x8(v32int16, int, v64uint8, int);
"""
        intrinsics = chess_test_gen.walk_ast(source)
        muls = [i for i in intrinsics if i.name == "mul_2x8_8x8"]
        assert len(muls) == 2
        # Different overload indices
        assert muls[0].overload_index != muls[1].overload_index

    def test_walk_void_return(self):
        """Void-return functions are still extracted (filtered later)."""
        source = "void do_something(int);\n"
        intrinsics = chess_test_gen.walk_ast(source)
        assert len(intrinsics) == 1
        assert intrinsics[0].return_type == "void"


class TestChessFilter:
    def test_filter_dont_care(self):
        """Functions with dont_care property are skipped."""
        from chess_preprocess import ChessAnnotation
        i = chess_test_gen.ChessIntrinsic(
            name="undef_v16int32", namespace="", return_type="v16int32",
            return_size=64, params=[], is_inline=False,
            properties=["dont_care"], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "dont_care" in reason

    def test_filter_volatile(self):
        """Functions with volatile property are skipped."""
        i = chess_test_gen.ChessIntrinsic(
            name="acquire", namespace="me_primitive", return_type="void",
            return_size=0, params=[("unsigned", 4)], is_inline=False,
            properties=["volatile"], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "volatile" in reason

    def test_filter_void_return(self):
        """Void-return functions are skipped."""
        i = chess_test_gen.ChessIntrinsic(
            name="some_op", namespace="", return_type="void",
            return_size=0, params=[("int", 4)], is_inline=False,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "void" in reason

    def test_filter_operator(self):
        """Operator overloads are skipped."""
        i = chess_test_gen.ChessIntrinsic(
            name="operator+=", namespace="", return_type="v64int8",
            return_size=64, params=[("v64int8", 64)], is_inline=True,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "operator" in reason

    def test_filter_diagnostic(self):
        """Diagnostic functions (chess_report etc.) are skipped."""
        i = chess_test_gen.ChessIntrinsic(
            name="chess_report", namespace="", return_type="void",
            return_size=0, params=[("int", 4)], is_inline=True,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"

    def test_filter_storage_params(self):
        """Functions with chess_storage parameters are skipped."""
        i = chess_test_gen.ChessIntrinsic(
            name="some_mem_op", namespace="", return_type="v16int32",
            return_size=64, params=[("int", 4)], is_inline=False,
            properties=[], storage_params=["TM"],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "storage" in reason

    def test_filter_unsized_return(self):
        """Functions with 0-byte return type are skipped."""
        i = chess_test_gen.ChessIntrinsic(
            name="some_op", namespace="", return_type="unknown_type",
            return_size=0, params=[("int", 4)], is_inline=False,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "unsized" in reason

    def test_pass_pure_function(self):
        """A clean function with no skip signals passes the filter."""
        i = chess_test_gen.ChessIntrinsic(
            name="broadcast_to_v16int32", namespace="me_primitive",
            return_type="v16int32", return_size=64,
            params=[("int", 4)], is_inline=False,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "generated"


class TestChessDirectoryName:
    def test_simple_function(self):
        """Single-arg function gets signature-based dir name."""
        i = chess_test_gen.ChessIntrinsic(
            name="broadcast_to_v16int32", namespace="me_primitive",
            return_type="v16int32", return_size=64,
            params=[("int", 4)], is_inline=False,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        assert chess_test_gen.dir_name(i) == "broadcast_to_v16int32__int"

    def test_multi_arg_function(self):
        """Multi-arg function encodes all param types."""
        i = chess_test_gen.ChessIntrinsic(
            name="mul_2x8_8x8", namespace="",
            return_type="v16acc64", return_size=128,
            params=[("v32int16", 64), ("v64uint8", 64)], is_inline=False,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        assert chess_test_gen.dir_name(i) == "mul_2x8_8x8__v32int16_v64uint8"

    def test_no_args(self):
        """Zero-arg function gets name only (no __ suffix)."""
        i = chess_test_gen.ChessIntrinsic(
            name="get_something", namespace="",
            return_type="int", return_size=4,
            params=[], is_inline=False,
            properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        assert chess_test_gen.dir_name(i) == "get_something"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessASTWalker tools/test_chess_test_gen.py::TestChessFilter tools/test_chess_test_gen.py::TestChessDirectoryName -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement chess-test-gen.py**

```python
#!/usr/bin/env python3
"""Generate single-instruction test kernels from Chess intrinsic declarations.

Parses me_chess_opns.h using clang.cindex (after pre-processing to strip
Chess extensions), extracts function signatures, filters testable intrinsics,
and generates kernel.cc + aie.mlir per test.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import clang.cindex

# Import shared components
from chess_preprocess import preprocess_chess_header, ChessAnnotation
from chess_type_stubs import parse_chess_traits, generate_stub_header

# Import shared codegen from Peano path
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_peano_gen = import_module("instr-test-gen")
generate_aie_mlir = _peano_gen.generate_aie_mlir
generate_test_host_cpp = _peano_gen.generate_test_host_cpp


@dataclass
class ChessIntrinsic:
    """Parsed Chess intrinsic function declaration."""
    name: str
    namespace: str
    return_type: str
    return_size: int
    params: list[tuple[str, int]]
    is_inline: bool
    properties: list[str] = field(default_factory=list)
    storage_params: list[str] = field(default_factory=list)
    overload_index: int = 0
    source_line: int = 0


# Skip filter keywords (matched against chess_property words)
SKIP_PROPERTY_WORDS = frozenset({
    "volatile", "dont_care", "non_functional",
    "keep_with_operand", "arg_mem_only",
})

# Diagnostic/utility function name prefixes to skip
SKIP_NAME_PREFIXES = (
    "chess_report", "chess_assert", "chess_error", "chess_warning",
    "chess_exit", "chess_stop", "chess_message", "chess_cycle_count",
    "chess_return_address", "chess_dont_care",
    "keep_in_registers_wrapper",
)


def walk_ast(source: str, annotations: dict[str, ChessAnnotation] | None = None) -> list[ChessIntrinsic]:
    """Parse source with clang.cindex and extract function declarations.

    Returns a list of ChessIntrinsic records. Annotations (from the
    pre-processor) are joined by **function name** -- not line number --
    so the annotation lookup is stable across pre-processing transforms.
    """
    if annotations is None:
        annotations = {}

    index = clang.cindex.Index.create()
    tu = index.parse(
        "chess_opns_clean.cpp",
        unsaved_files=[("chess_opns_clean.cpp", source)],
        args=["-std=c++17", "-fsyntax-only"],
    )

    intrinsics: list[ChessIntrinsic] = []
    name_counts: dict[str, int] = {}  # for overload indexing

    def visit(cursor, namespace=""):
        nonlocal intrinsics, name_counts

        if cursor.kind == clang.cindex.CursorKind.NAMESPACE:
            ns_name = cursor.spelling
            for child in cursor.get_children():
                visit(child, namespace=ns_name)
            return

        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            name = cursor.spelling
            ret_type = cursor.result_type

            # Extract parameter types and sizes
            params = []
            for child in cursor.get_children():
                if child.kind == clang.cindex.CursorKind.PARM_DECL:
                    ptype = child.type.get_canonical()
                    params.append((child.type.spelling, ptype.get_size()))

            # Overload index
            overload_idx = name_counts.get(name, 0)
            name_counts[name] = overload_idx + 1

            # Check if inline (has a body / compound statement child)
            is_inline = any(
                child.kind == clang.cindex.CursorKind.COMPOUND_STMT
                for child in cursor.get_children()
            )

            # Join annotations by function name
            ann = annotations.get(name)
            props = list(ann.properties) if ann else []
            storage = list(ann.storage_params) if ann else []

            ret_canonical = ret_type.get_canonical()
            intrinsics.append(ChessIntrinsic(
                name=name,
                namespace=namespace,
                return_type=ret_type.spelling,
                return_size=max(0, ret_canonical.get_size()),
                params=params,
                is_inline=is_inline,
                properties=props,
                storage_params=storage,
                overload_index=overload_idx,
                source_line=cursor.location.line,
            ))

        for child in cursor.get_children():
            if child.kind != clang.cindex.CursorKind.NAMESPACE:
                # Don't double-visit namespaces
                if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
                    visit(child, namespace)

    for child in tu.cursor.get_children():
        visit(child)

    return intrinsics


def classify_chess_intrinsic(i: ChessIntrinsic) -> tuple[str, str]:
    """Classify a Chess intrinsic as 'generated' or 'skipped' with reason."""
    # Property-based skips (word-level matching)
    prop_words = set(i.properties)
    for skip_word in SKIP_PROPERTY_WORDS:
        if skip_word in prop_words:
            return ("skipped", f"chess_property({skip_word})")

    # chess_storage parameters
    if len(i.storage_params) > 0:
        return ("skipped", f"chess_storage params: {i.storage_params}")

    # Void return
    if i.return_type == "void" or i.return_size <= 0:
        return ("skipped", "void return or unsized return type")

    # Unsized parameters
    for ptype, psize in i.params:
        if psize <= 0:
            return ("skipped", f"unsized param type: {ptype}")

    # Operator overloads
    if i.name.startswith("operator"):
        return ("skipped", "operator overload")

    # Diagnostic/utility functions
    for prefix in SKIP_NAME_PREFIXES:
        if i.name.startswith(prefix):
            return ("skipped", f"diagnostic/utility: {prefix}")

    return ("generated", "")


def dir_name(i: ChessIntrinsic) -> str:
    """Generate a stable directory name encoding the function signature.

    Format: {name}__{param_types} for functions with args,
            {name} for zero-arg functions.
    """
    if not i.params:
        return i.name
    param_types = "_".join(ptype for ptype, _ in i.params)
    return f"{i.name}__{param_types}"


def generate_chess_kernel_cc(
    func_name: str,
    namespace: str,
    return_type: str,
    params: list[tuple[str, int]],
) -> str:
    """Generate kernel.cc that calls one Chess intrinsic.

    Arguments are read from consecutive regions of the input buffer.
    The result is written to the output buffer.
    """
    param_sig = ", ".join(ptype for ptype, _ in params)
    sig_comment = f"{return_type} = f({param_sig})" if params else f"{return_type} = f()"

    # Check if any sub-4-byte scalar params exist (need string.h for memcpy)
    has_sub4_scalar = any(psize < 4 for _, psize in params)

    lines = [
        f"// Auto-generated: tests {func_name} (Chess)",
        f"// Signature: {sig_comment}",
    ]
    if namespace:
        lines.append(f"// Namespace: {namespace}")
    lines.append("#define NOCPP")
    lines.append("#include <stdint.h>")
    if has_sub4_scalar:
        lines.append("#include <string.h>")
    lines += [
        "",
        'extern "C" {',
        "void test_kernel(const int32_t *restrict in, int32_t *restrict out) {",
    ]

    # Generate argument reads
    offset_bytes = 0
    offset_i32 = 0
    arg_names = []
    for idx, (ptype, psize) in enumerate(params):
        arg_name = f"arg{idx}"
        arg_names.append(arg_name)

        if psize >= 4:
            lines.append(f"    {ptype} {arg_name} = "
                          f"*({ptype} *)(in + {offset_i32});")
        else:
            lines.append(f"    {ptype} {arg_name};")
            lines.append(f"    memcpy(&{arg_name}, "
                          f"(const char *)in + {offset_bytes}, "
                          f"sizeof({ptype}));")

        align_i32 = max(1, psize // 4)
        offset_i32 += align_i32
        offset_bytes += max(4, psize)

    # Call intrinsic
    call_args = ", ".join(arg_names)
    lines.append(f"    {return_type} result = {func_name}({call_args});")

    # Write result
    lines.append(f"    {return_type} *out_vec = ({return_type} *)out;")
    lines.append("    *out_vec = result;")
    lines.append("}")
    lines.append('} // extern "C"')
    lines.append("")

    return "\n".join(lines)


@dataclass
class GeneratedTest:
    """One generated Chess test case."""
    name: str
    func_name: str
    namespace: str
    compiler: str
    in_size: int
    out_size: int
    params: list[str]
    properties: list[str]


@dataclass
class SkippedIntrinsic:
    """One skipped Chess intrinsic."""
    name: str
    func_name: str
    reason: str


def generate_all(
    opns_path: str,
    types_path: str,
    out_dir: str,
) -> tuple[list[GeneratedTest], list[SkippedIntrinsic]]:
    """Main entry point: full pipeline from header files to test artifacts."""

    # Step 1: Generate type stubs
    types_text = Path(types_path).read_text()
    traits = parse_chess_traits(types_text)
    stubs = generate_stub_header(traits)

    # Step 2: Pre-process header
    opns_text = Path(opns_path).read_text()
    clean, annotations = preprocess_chess_header(opns_text)

    # Step 3: Combine stubs + cleaned header
    combined = stubs + "\n" + clean

    # Step 4: Walk AST
    intrinsics = walk_ast(combined, annotations)

    # Step 5: Filter and generate
    os.makedirs(out_dir, exist_ok=True)

    generated: list[GeneratedTest] = []
    skipped: list[SkippedIntrinsic] = []

    for i in intrinsics:
        status, reason = classify_chess_intrinsic(i)
        dname = dir_name(i)

        if status == "skipped":
            skipped.append(SkippedIntrinsic(
                name=dname, func_name=i.name, reason=reason,
            ))
            continue

        # Compute buffer sizes
        in_size = sum(max(4, psize) for _, psize in i.params)
        in_size = max(4, in_size)
        out_size = max(4, i.return_size)

        test_dir = os.path.join(out_dir, dname)
        os.makedirs(test_dir, exist_ok=True)

        # Write kernel.cc
        kernel = generate_chess_kernel_cc(
            i.name, i.namespace, i.return_type, i.params,
        )
        Path(os.path.join(test_dir, "kernel.cc")).write_text(kernel)

        # Write aie.mlir (reuse Peano path template)
        mlir = generate_aie_mlir(in_size, out_size)
        Path(os.path.join(test_dir, "aie.mlir")).write_text(mlir)

        generated.append(GeneratedTest(
            name=dname, func_name=i.name, namespace=i.namespace,
            compiler="chess", in_size=in_size, out_size=out_size,
            params=[ptype for ptype, _ in i.params],
            properties=i.properties,
        ))

    # Write type stubs header (for reference)
    Path(os.path.join(out_dir, "chess_type_stubs.h")).write_text(stubs)

    # Write shared test_host.cpp
    host_code = generate_test_host_cpp()
    Path(os.path.join(out_dir, "test_host.cpp")).write_text(host_code)

    # Write manifest
    manifest = {
        "compiler": "chess",
        "generated": [asdict(g) for g in generated],
        "skipped": [asdict(s) for s in skipped],
    }
    Path(os.path.join(out_dir, "manifest.json")).write_text(
        json.dumps(manifest, indent=2) + "\n",
    )

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chess intrinsic test kernels from me_chess_opns.h")
    parser.add_argument("--opns",
                        default="../aietools/data/aie_ml/lib/isg/me_chess_opns.h",
                        help="Path to me_chess_opns.h")
    parser.add_argument("--types",
                        default="../aietools/data/aie_ml/lib/isg/me_chess_types.h",
                        help="Path to me_chess_types.h")
    parser.add_argument("--out-dir", default="build/instr-tests-chess",
                        help="Output directory")
    args = parser.parse_args()

    generated, skipped = generate_all(args.opns, args.types, args.out_dir)

    print(f"Generated: {len(generated)} tests")
    print(f"Skipped:   {len(skipped)} intrinsics")
    for s in skipped:
        print(f"  SKIP {s.name}: {s.reason}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessASTWalker tools/test_chess_test_gen.py::TestChessFilter tools/test_chess_test_gen.py::TestChessDirectoryName -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/chess-test-gen.py tools/test_chess_test_gen.py
git commit -m "feat(instr-test): add Chess intrinsic test generator with clang.cindex AST walker"
```

---

### Task 5: Integration Test -- Generate and Compile One Chess Intrinsic

**Files:**
- Test: `tools/test_chess_test_gen.py` (append)
- Read: `tools/chess-test-gen.py`

This task validates the full generate -> xchesscc compile cycle for at least
one real Chess intrinsic. This is where we discover if bare function names
work or if we need `using namespace me_primitive;`.

- [ ] **Step 1: Write integration test for generation from real header**

```python
# Append to tools/test_chess_test_gen.py
class TestChessIntegration:
    def test_generate_from_real_header(self):
        """Integration: generate tests from actual me_chess_opns.h."""
        from pathlib import Path
        import tempfile

        opns_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_opns.h"
        types_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_chess_types.h"
        if not opns_path.exists() or not types_path.exists():
            pytest.skip("aietools not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            generated, skipped = chess_test_gen.generate_all(
                str(opns_path), str(types_path), tmpdir,
            )

            # Should generate a meaningful number of tests
            assert len(generated) >= 100, (
                f"Only {len(generated)} tests generated, expected 100+. "
                f"Skipped: {len(skipped)}"
            )

            # Should have manifest
            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()
            manifest = json.load(manifest_path.open())
            assert manifest["compiler"] == "chess"
            assert len(manifest["generated"]) == len(generated)

            # Should have test_host.cpp
            assert (Path(tmpdir) / "test_host.cpp").exists()

            # Spot-check a generated kernel.cc
            # Find broadcast_to_v16int32 if it exists
            broadcast_tests = [g for g in generated
                               if "broadcast_to_v16int32" in g.name]
            if broadcast_tests:
                kernel_path = Path(tmpdir) / broadcast_tests[0].name / "kernel.cc"
                assert kernel_path.exists()
                kernel_src = kernel_path.read_text()
                assert "broadcast_to_v16int32" in kernel_src
                assert "test_kernel" in kernel_src
                assert 'extern "C"' in kernel_src

            print(f"Generated {len(generated)} tests, skipped {len(skipped)}")
```

- [ ] **Step 2: Run the integration test**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessIntegration -v -s`
Expected: PASS with 100+ tests generated

- [ ] **Step 3: Try compiling one generated kernel with xchesscc**

Run manually to verify xchesscc accepts the generated code:
```bash
cd /home/triple/npu-work/xdna-emu
python3 tools/chess-test-gen.py --out-dir build/instr-tests-chess
# Pick a simple test (broadcast if available)
cd build/instr-tests-chess
ls  # find a test directory
# Try compiling
cd <test_dir>
xchesscc_wrapper aie2 -c kernel.cc -o kernel.o 2>&1
```

If bare function names fail, add `using namespace me_primitive;` to the
kernel template and regenerate. Update `generate_chess_kernel_cc()` and
its tests accordingly.

- [ ] **Step 4: Fix any compilation issues found in step 3**

Common issues and fixes:
- **"undeclared identifier"**: Add `using namespace me_primitive;` after
  the `extern "C"` line in the kernel template.
- **Type mismatch**: The type stub sizes may not match what xchesscc
  expects. Fix type stubs or add exclusions.

- [ ] **Step 5: Commit**

```bash
git add tools/test_chess_test_gen.py tools/chess-test-gen.py
git commit -m "test(instr-test): integration test -- Chess intrinsic generation from real header"
```

---

### Task 6: Runner Script Integration (`instr-test.sh`)

**Files:**
- Modify: `scripts/instr-test.sh`

- [ ] **Step 1: Add --chess/--peano/--both flags and compiler-specific paths**

Key changes to `scripts/instr-test.sh`:

1. Add `COMPILER` variable with `--chess`/`--peano`/`--both` flag parsing
2. Set `OUT_DIR` and `RESULTS_DIR` based on compiler choice
3. Switch kernel compile command (xchesscc vs Peano clang)
4. Switch aiecc.py flags (`--xchesscc --xbridge` vs `--no-xchesscc --no-xbridge`)
5. For `--both`, run each compiler path sequentially (two full passes)

Changes at the top of the script (argument parsing):
```bash
# Add after existing defaults
COMPILER="peano"  # default

# Add to the case statement
--chess)       COMPILER="chess"; shift ;;
--peano)       COMPILER="peano"; shift ;;
--both)        COMPILER="both"; shift ;;
```

Changes to `OUT_DIR` and `RESULTS_DIR`:
```bash
OUT_DIR="${PROJECT_DIR}/build/instr-tests-${COMPILER}"
RESULTS_DIR="/tmp/instr-test-results-$(date +%Y%m%d)-${COMPILER}"
```

Changes to Phase 1 (Generate):
```bash
if [[ "$COMPILER" == "chess" ]]; then
    python3 "${PROJECT_DIR}/tools/chess-test-gen.py" \
        --opns "${AIETOOLS}/data/aie_ml/lib/isg/me_chess_opns.h" \
        --types "${AIETOOLS}/data/aie_ml/lib/isg/me_chess_types.h" \
        --out-dir "$OUT_DIR"
else
    python3 "${PROJECT_DIR}/tools/instr-test-gen.py" --td "$TD_FILE" --out-dir "$OUT_DIR"
fi
```

Changes to `compile_one()`:
```bash
if [[ "$COMPILER" == "chess" ]]; then
    # Chess kernel compile
    (cd "$test_dir" && \
        nice -n 19 xchesscc_wrapper aie2 -c kernel.cc -o kernel.o \
        2>"${test_dir}/chess.log") || { ... }
    # Chess MLIR compile
    (cd "$test_dir" && \
        nice -n 19 aiecc.py --no-aiesim --xchesscc --xbridge \
            --aie-generate-xclbin --xclbin-name=aie.xclbin \
            --aie-generate-npu-insts --npu-insts-name=insts.bin \
            aie.mlir 2>"${test_dir}/aiecc.log") || { ... }
else
    # Peano kernel compile (existing code)
    ...
fi
```

For `--both`, the script simply re-invokes itself twice:
```bash
if [[ "$COMPILER" == "both" ]]; then
    "$0" --peano "${PASSTHROUGH_ARGS[@]}"
    "$0" --chess "${PASSTHROUGH_ARGS[@]}"
    exit $?
fi
```
where `PASSTHROUGH_ARGS` captures all flags except `--both` (collected
during argument parsing). This avoids restructuring the script body.

- [ ] **Step 2: Test with --chess flag (generate-only first)**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/instr-test.sh --chess --generate-only`
Expected: Generates Chess test artifacts in `build/instr-tests-chess/`

- [ ] **Step 3: Test with --peano flag (verify no regression)**

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/instr-test.sh --peano --generate-only`
Expected: Generates Peano test artifacts in `build/instr-tests-peano/`
(same content as before, just different directory)

- [ ] **Step 4: Commit**

```bash
git add scripts/instr-test.sh
git commit -m "feat(instr-test): add --chess/--peano/--both flags to runner script"
```

---

### Task 7: End-to-End Validation

**Files:** None (manual validation)

This task runs the full pipeline: generate -> compile -> HW -> EMU -> compare
for at least one Chess intrinsic.

- [ ] **Step 1: Run Chess path EMU-only**

```bash
cd /home/triple/npu-work/xdna-emu
./scripts/instr-test.sh --chess --no-hw --filter broadcast_to_v16int32
```

Expected: One test compiles, runs on EMU, produces output.

- [ ] **Step 2: Run Chess path with HW**

```bash
./scripts/instr-test.sh --chess --filter broadcast_to_v16int32
```

Expected: HW run, EMU run, comparison. PASS or DIVERGE (both are valid
outcomes -- the harness is working either way).

- [ ] **Step 3: Run broader Chess set**

```bash
./scripts/instr-test.sh --chess --no-hw -j$(nproc)
```

Expected: Compile results for many Chess intrinsics. Some may fail to
compile (xchesscc may reject certain generated patterns). Record failure
categories for future filter additions.

- [ ] **Step 4: Commit any filter fixes discovered during validation**

```bash
git add tools/chess-test-gen.py tools/test_chess_test_gen.py
git commit -m "fix(instr-test): refine Chess filter based on xchesscc compilation results"
```
