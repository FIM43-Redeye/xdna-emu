# Chess Kernel Codegen Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all four root causes of xchesscc compile failures so every generated Chess test kernel compiles cleanly.

**Architecture:** Four targeted fixes to the existing Chess pipeline: (1) parse me_iss_types.h property comments for missing type stubs, (2) restore namespace-conditional qualification, (3) detect reference output params via clang TypeKind and write them to the output buffer, (4) replace all pointer-cast buffer access with memcpy.

**Tech Stack:** Python 3.13, clang.cindex, xchesscc_wrapper, existing chess-test-gen.py pipeline

**Spec:** `docs/superpowers/specs/2026-03-16-chess-codegen-fixes-design.md`

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `tools/chess_type_stubs.py` | Modify | Add `parse_iss_property_comments()` for me_iss_types.h; fix ceiling division; add `--iss-types-header` CLI flag |
| `tools/chess-test-gen.py` | Modify | Namespace qualification; reference param handling; memcpy buffer access; pointer param classification; `--iss-types-header` passthrough |
| `tools/test_chess_test_gen.py` | Modify | Tests for all new behavior |
| `scripts/instr-test.sh` | Modify | Pass `--iss-types-header` to Chess generator |

---

## Chunk 1: ISS Type Stubs and Ceiling Division

### Task 1: Parse me_iss_types.h Property Comments

**Files:**
- Modify: `tools/chess_type_stubs.py`
- Modify: `tools/test_chess_test_gen.py`

The property comments in me_iss_types.h have two forms:
```cpp
class u2;               // property(  2 bit unsigned );
class v8w64;            // property( vector w64[8] );
```

Scalar types give bit width directly. Vector types reference an element
type and count -- resolve element width from the same table, multiply.

- [ ] **Step 1: Write failing tests for ISS property comment parser**

```python
# Append to tools/test_chess_test_gen.py
from chess_type_stubs import parse_iss_property_comments

class TestISSTypeStubs:
    def test_parse_scalar_type(self):
        """Parse a scalar property comment."""
        text = 'class u2;               // property(  2 bit unsigned );'
        types = parse_iss_property_comments(text)
        assert types["u2"] == 2

    def test_parse_signed_scalar(self):
        """Parse a signed scalar."""
        text = 'class w32;              // property( 32 bit   signed );'
        types = parse_iss_property_comments(text)
        assert types["w32"] == 32

    def test_parse_large_scalar(self):
        """Parse a large scalar like smode_t (768 bits)."""
        text = 'class smode_t;          // property( 768 bit unsigned );'
        types = parse_iss_property_comments(text)
        assert types["smode_t"] == 768

    def test_parse_vector_type(self):
        """Parse a vector property comment with element resolution."""
        text = '''class w64;              // property( 64 bit   signed );
class v8w64;            // property( vector w64[8] );'''
        types = parse_iss_property_comments(text)
        assert types["w64"] == 64
        assert types["v8w64"] == 512  # 64 * 8

    def test_parse_nested_vector(self):
        """Vector of vectors: v16w256 = 16 * w256, w256 = 256 bits."""
        text = '''class w256;             // property( 256 bit   signed );
class v16w256;          // property( vector w256[16] );'''
        types = parse_iss_property_comments(text)
        assert types["v16w256"] == 4096

    def test_parse_vector_of_small_element(self):
        """v5u1 = 5 * u1 = 5 bits."""
        text = '''class u1;               // property(  1 bit unsigned );
class v5u1;             // property( vector u1[5] );'''
        types = parse_iss_property_comments(text)
        assert types["v5u1"] == 5

    def test_skip_non_property_lines(self):
        """Lines without property comments are ignored."""
        text = '''#ifndef ME_ISS_TYPES_H
#define ME_ISS_TYPES_H
class u2;               // property(  2 bit unsigned );
// some random comment
class w64;              // property( 64 bit   signed );'''
        types = parse_iss_property_comments(text)
        assert len(types) == 2

    def test_parse_real_file(self):
        """Integration: parse the actual me_iss_types.h."""
        from pathlib import Path
        iss_path = Path(__file__).parent.parent.parent / "aietools/data/aie_ml/lib/isg/me_iss_types.h"
        if not iss_path.exists():
            pytest.skip("aietools not available")
        types = parse_iss_property_comments(iss_path.read_text())
        # Spot-check known types
        assert types["u1"] == 1
        assert types["u2"] == 2
        assert types["w32"] == 32
        assert types["w64"] == 64
        assert types["w128"] == 128
        assert types["pmode_t"] == 26
        assert types["smode_t"] == 768
        assert types["mmode_t"] == 8
        assert types["v5u1"] == 5
        assert types["v8w64"] == 512
        assert types["v16w32"] == 512
        assert types["v32w32"] == 1024
        assert types["v512w8"] == 4096
        assert len(types) >= 100  # hundreds of types
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestISSTypeStubs -v`
Expected: FAIL with `ImportError: cannot import name 'parse_iss_property_comments'`

- [ ] **Step 3: Implement parse_iss_property_comments**

Add to `tools/chess_type_stubs.py`:

```python
# Matches: class NAME; // property( N bit [un]signed );
_ISS_SCALAR = re.compile(
    r'class\s+(\w+)\s*;\s*//\s*property\(\s*(\d+)\s+bit\s+(?:un)?signed\s*\)'
)

# Matches: class NAME; // property( vector ELEM[COUNT] );
_ISS_VECTOR = re.compile(
    r'class\s+(\w+)\s*;\s*//\s*property\(\s*vector\s+(\w+)\[(\d+)\]\s*\)'
)


def parse_iss_property_comments(text: str) -> dict[str, int]:
    """Parse ISS type property comments, returning {type_name: bits}.

    Two forms:
    - Scalar: ``class u2; // property( 2 bit unsigned );`` -> 2
    - Vector: ``class v8w64; // property( vector w64[8] );`` -> w64.bits * 8

    Vector element types are resolved from the same table (scalars parsed
    first, then vectors in dependency order).
    """
    types: dict[str, int] = {}

    # First pass: collect all scalar types.
    for m in _ISS_SCALAR.finditer(text):
        name, bits = m.group(1), int(m.group(2))
        types[name] = bits

    # Second pass: resolve vector types (may need multiple passes for
    # vectors-of-vectors, though in practice one pass suffices).
    unresolved: list[tuple[str, str, int]] = []
    for m in _ISS_VECTOR.finditer(text):
        name, elem, count = m.group(1), m.group(2), int(m.group(3))
        if elem in types:
            types[name] = types[elem] * count
        else:
            unresolved.append((name, elem, count))

    # Resolve remaining vectors (dependency chains).
    max_passes = 10
    for _ in range(max_passes):
        if not unresolved:
            break
        still_unresolved = []
        for name, elem, count in unresolved:
            if elem in types:
                types[name] = types[elem] * count
            else:
                still_unresolved.append((name, elem, count))
        unresolved = still_unresolved

    return types
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestISSTypeStubs -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Fix ceiling division bug**

In `generate_stub_header`, change line 129:
```python
# Before (truncates):
byte_size = max(1, bits // 8)
# After (ceiling):
byte_size = max(1, (bits + 7) // 8)
```

Add test:
```python
class TestChessTypeStubs:
    # ... (append to existing class)
    def test_ceiling_division_non_byte_aligned(self):
        """Non-byte-aligned types get ceiling division."""
        traits = {"v5u1": 5, "w9_step4": 9, "u1": 1}
        header = generate_stub_header(traits)
        assert "struct v5u1 { char _data[1]; };" in header   # ceil(5/8) = 1
        assert "struct w9_step4 { char _data[2]; };" in header  # ceil(9/8) = 2
        assert "struct u1 { char _data[1]; };" in header      # ceil(1/8) = 1
```

- [ ] **Step 6: Wire ISS types into generate_all and CLI**

In `chess_type_stubs.py`, update `main()` to accept `--iss-types-header`.

In `chess-test-gen.py`:
- Add `--iss-types-header` CLI argument
- In `generate_all`, call `parse_iss_property_comments` on the ISS header,
  merge with traits from `parse_chess_traits`, then generate stubs from
  the merged table.

```python
# In generate_all():
traits = parse_chess_traits(types_text)
if iss_path:
    iss_text = Path(iss_path).read_text()
    iss_types = parse_iss_property_comments(iss_text)
    # ISS types fill gaps; chessTraitsOf takes precedence
    for name, bits in iss_types.items():
        traits.setdefault(name, bits)
stubs = generate_stub_header(traits)
```

In `scripts/instr-test.sh`, update the Chess generator invocation:
```bash
python3 "${PROJECT_DIR}/tools/chess-test-gen.py" \
    --opns-header "${AIETOOLS_DIR}/data/aie_ml/lib/isg/me_chess_opns.h" \
    --types-header "${AIETOOLS_DIR}/data/aie_ml/lib/isg/me_chess_types.h" \
    --iss-types-header "${AIETOOLS_DIR}/data/aie_ml/lib/isg/me_iss_types.h" \
    --out-dir "$OUT_DIR"
```

- [ ] **Step 7: Run all tests**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py -v`
Expected: All tests PASS (existing 51 + new ~9)

- [ ] **Step 8: Verify clang.cindex parse errors drop to near-zero**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestClangParsing::test_parse_preprocessed_real_header -v -s`
Expected: Error count drops from 20 to near 0 (all ISS types now have stubs)

- [ ] **Step 9: Commit**

```bash
git add tools/chess_type_stubs.py tools/chess-test-gen.py tools/test_chess_test_gen.py scripts/instr-test.sh
git commit -m "feat(instr-test): parse me_iss_types.h for complete Chess type coverage"
```

---

## Chunk 2: Namespace, Reference Params, and Memcpy

### Task 2: Restore Namespace Qualification

**Files:**
- Modify: `tools/chess-test-gen.py:311`
- Modify: `tools/test_chess_test_gen.py`

- [ ] **Step 1: Write failing test**

```python
# Append to TestChessASTWalker or create new class
class TestChessNamespaceQualification:
    def test_me_primitive_gets_qualified(self):
        """me_primitive functions get me_primitive:: prefix."""
        chess_test_gen = importlib.import_module("chess-test-gen")
        kernel = chess_test_gen.generate_chess_kernel_cc(
            "ext_xl", "me_primitive", "v64uint4",
            [("v128uint4", 64)],
        )
        assert "me_primitive::ext_xl(" in kernel

    def test_global_stays_unqualified(self):
        """Global-namespace functions have no prefix."""
        chess_test_gen = importlib.import_module("chess-test-gen")
        kernel = chess_test_gen.generate_chess_kernel_cc(
            "broadcast_elem", "", "v16int32",
            [("v16int32", 64), ("int", 4)],
        )
        assert "broadcast_elem(" in kernel
        assert "me_primitive::" not in kernel
```

- [ ] **Step 2: Run test, verify it fails**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessNamespaceQualification -v`
Expected: FAIL -- `me_primitive::ext_xl` not found (currently emits bare `ext_xl`)

- [ ] **Step 3: Fix the qualification in chess-test-gen.py**

Change line 311:
```python
# Before:
qualified_name = func_name
# After:
qualified_name = f"{namespace}::{func_name}" if namespace else func_name
```

- [ ] **Step 4: Run test, verify it passes**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessNamespaceQualification -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/chess-test-gen.py tools/test_chess_test_gen.py
git commit -m "fix(instr-test): restore namespace qualification for me_primitive functions"
```

---

### Task 3: Reference Output Parameters and Memcpy Buffer Access

**Files:**
- Modify: `tools/chess-test-gen.py` (classify + codegen)
- Modify: `tools/test_chess_test_gen.py`

This task handles Root Causes 3 and 4 together since both affect
`generate_chess_kernel_cc` and the changes are intertwined.

- [ ] **Step 1: Write failing tests for reference param detection**

```python
class TestChessReferenceParams:
    def test_detect_lvalue_ref_output(self):
        """Non-const lvalue reference is an output parameter."""
        import clang.cindex
        chess_test_gen = importlib.import_module("chess-test-gen")

        source = """
struct v5u1 { char _data[1]; };
struct v16int32 { char _data[64]; };
int some_func(v16int32, v5u1 &);
"""
        intrinsics = chess_test_gen.walk_ast(source)
        assert len(intrinsics) == 1
        i = intrinsics[0]
        # v5u1 & should be detected as reference output
        # walk_ast normalizes ref params to (type, size, is_ref_output=True)
        # by checking TypeKind.LVALUEREF, not string matching
        assert any("v5u1" in p[0] for p in i.params)
        # Classify should pass (it has a non-void return)
        status, _ = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "generated"

    def test_const_ref_is_input(self):
        """const T & is an input parameter, not output."""
        chess_test_gen = importlib.import_module("chess-test-gen")
        source = """
struct v16int32 { char _data[64]; };
v16int32 some_func(const v16int32 &);
"""
        intrinsics = chess_test_gen.walk_ast(source)
        assert len(intrinsics) == 1
        # const ref should be treated as an input, not skipped
        status, _ = chess_test_gen.classify_chess_intrinsic(intrinsics[0])
        assert status == "generated"

    def test_kernel_with_ref_output(self):
        """Generated kernel writes ref output to output buffer."""
        chess_test_gen = importlib.import_module("chess-test-gen")
        kernel = chess_test_gen.generate_chess_kernel_cc(
            "some_func", "", "int",
            [("v16int32", 64), ("v5u1 &", 1)],
        )
        # Should NOT read ref param from input buffer
        assert "arg1_out" in kernel or "ref" in kernel.lower()
        # Should write ref output to output buffer after return value
        assert "memcpy" in kernel
        # Should have both result and ref output written
        assert kernel.count("memcpy((char *)out") >= 1 or kernel.count("memcpy(out") >= 1

    def test_kernel_memcpy_for_all_reads(self):
        """All buffer reads use memcpy, not pointer casts."""
        chess_test_gen = importlib.import_module("chess-test-gen")
        kernel = chess_test_gen.generate_chess_kernel_cc(
            "broadcast_elem", "", "v16int32",
            [("v16int32", 64), ("int", 4)],
        )
        # Should use memcpy for reading, not pointer cast
        assert "memcpy(&arg0" in kernel
        # Should use memcpy for writing result
        assert "memcpy(out" in kernel or "memcpy((char *)out" in kernel
        # Should NOT have pointer cast reads
        assert "*(const" not in kernel

    def test_pointer_param_skipped(self):
        """Functions with pointer parameters are classified as untestable."""
        chess_test_gen = importlib.import_module("chess-test-gen")
        i = chess_test_gen.ChessIntrinsic(
            name="load_lut", namespace="", return_type="v16int32",
            return_size=64, params=[("const void *", 8), ("int", 4)],
            is_inline=False, properties=[], storage_params=[],
            overload_index=0, source_line=1,
        )
        status, reason = chess_test_gen.classify_chess_intrinsic(i)
        assert status == "skipped"
        assert "pointer" in reason
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py::TestChessReferenceParams -v`
Expected: Multiple failures

- [ ] **Step 3: Update classify_chess_intrinsic for pointer params**

Add pointer parameter detection to the classifier, after the existing
storage_params check:

```python
# In classify_chess_intrinsic, after storage_params check:

# Pointer parameters cannot be meaningfully filled from an input buffer.
for ptype, psize, _is_ref in i.params:
    if '*' in ptype:
        return ("skipped", f"pointer parameter: {ptype}")
```

- [ ] **Step 4: Rewrite generate_chess_kernel_cc with memcpy and ref output support**

Replace the entire function body. Key changes:
- ALL reads use `memcpy(&argN, (const char *)in + byte_offset, sizeof(type))`
- Reference output params become output locals, not read from input
- ALL writes use `memcpy((char *)out + byte_offset, &val, sizeof(type))`
- `#include <string.h>` is unconditional
- Output buffer: return value first, then reference outputs

**Important**: Reference params are detected by `walk_ast` using
`clang.cindex.TypeKind.LVALUEREF`, NOT by string matching on `&`.
The walker normalizes param tuples to `(type_name, size, is_ref_output)`
where `is_ref_output` is True for non-const lvalue references.

First, update `walk_ast` to detect and normalize reference params.
Change the param extraction in `_visit`:

```python
# In walk_ast's _visit, replace the param extraction:
params: list[tuple[str, int, bool]] = []  # (type, size, is_ref_output)
for child in cursor.get_children():
    if child.kind == clang.cindex.CursorKind.PARM_DECL:
        ptype = child.type
        is_ref_output = False
        # Detect non-const lvalue reference = output parameter
        if ptype.kind == clang.cindex.TypeKind.LVALUEREF:
            pointee = ptype.get_pointee()
            if not pointee.is_const_qualified():
                is_ref_output = True
            ptype = pointee  # use the referent type for name/size
        type_name = ptype.spelling
        type_size = max(0, ptype.get_canonical().get_size())
        params.append((type_name, type_size, is_ref_output))
```

Update the `ChessIntrinsic` dataclass to use the new param tuple:
```python
params: list[tuple[str, int, bool]]  # [(type_name, size, is_ref_output), ...]
```

Then the codegen uses the `is_ref_output` flag directly:

```python
def generate_chess_kernel_cc(
    func_name: str,
    namespace: str,
    return_type: str,
    params: list[tuple[str, int, bool]],
) -> str:
    qualified_name = f"{namespace}::{func_name}" if namespace else func_name

    # Separate input params from reference output params.
    input_params = []   # (arg_name, type, size)
    ref_outputs = []    # (arg_name, base_type, size)
    for idx, (ptype, psize, is_ref) in enumerate(params):
        if is_ref:
            ref_outputs.append((f"arg{idx}_out", ptype, psize))
        else:
            input_params.append((f"arg{idx}", ptype, psize))

    # Build signature comment.
    all_types = ", ".join(ptype for ptype, _ in params)
    sig = f"{return_type} = {qualified_name}({all_types})"

    lines = [
        f"// Auto-generated: tests {qualified_name}",
        f"// Signature: {sig}",
        "#define NOCPP",
        "#include <stdint.h>",
        "#include <string.h>",
        "",
        'extern "C" {',
        "void test_kernel(const int32_t *restrict in, int32_t *restrict out) {",
    ]

    # Read input params from buffer via memcpy.
    in_byte_offset = 0
    for arg_name, ptype, psize in input_params:
        lines.append(f"    {ptype} {arg_name};")
        lines.append(f"    memcpy(&{arg_name}, (const char *)in + {in_byte_offset}, sizeof({ptype}));")
        in_byte_offset += psize

    # Declare reference output locals (zero-initialized).
    for arg_name, base_type, psize in ref_outputs:
        lines.append(f"    {base_type} {arg_name} = {{}};")

    # Build call argument list in original parameter order.
    call_args = []
    in_idx = 0
    ref_idx = 0
    for idx, (ptype, psize, is_ref) in enumerate(params):
        if is_ref:
            call_args.append(ref_outputs[ref_idx][0])
            ref_idx += 1
        else:
            call_args.append(input_params[in_idx][0])
            in_idx += 1

    lines.append(f"    {return_type} result = {qualified_name}({', '.join(call_args)});")

    # Write return value to output buffer.
    lines.append(f"    memcpy(out, &result, sizeof({return_type}));")

    # Write reference outputs after return value.
    out_byte_offset_expr = f"sizeof({return_type})"
    for arg_name, base_type, psize in ref_outputs:
        lines.append(f"    memcpy((char *)out + {out_byte_offset_expr}, &{arg_name}, sizeof({base_type}));")
        out_byte_offset_expr = f"{out_byte_offset_expr} + sizeof({base_type})"

    lines.append("}")
    lines.append('} // extern "C"')
    lines.append("")

    return "\n".join(lines)
```

Also update `generate_all` to compute `out_size` including reference
outputs:

```python
# In generate_all, after filtering:
ref_out_sizes = [psize for _, psize, is_ref in i.params if is_ref]
in_params = [(ptype, psize) for ptype, psize, is_ref in i.params if not is_ref]
in_size = max(4, sum(psize for _, psize in in_params))
out_size = max(4, i.return_size + sum(ref_out_sizes))
```

**TODO (future)**: The manifest should record per-field byte offsets
and sizes for the output buffer (return value + each ref output) so
the host-side reader can extract and compare them individually. For
now, binary comparison of the entire output buffer suffices.

- [ ] **Step 5: Run tests, verify they pass**

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_chess_test_gen.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tools/chess-test-gen.py tools/test_chess_test_gen.py
git commit -m "feat(instr-test): memcpy buffer access and reference output parameters"
```

---

### Task 4: End-to-End Validation

**Files:** None (manual validation)

- [ ] **Step 1: Regenerate with all fixes**

```bash
cd /home/triple/npu-work/xdna-emu
rm -rf build/instr-tests-chess
./scripts/instr-test.sh --chess --generate-only
```

Expected: Test count should increase (more types resolved = fewer skips).

- [ ] **Step 2: Spot-check previously failing kernels**

```bash
# me_primitive function (was failing: undeclared identifier)
xchesscc_wrapper aie2 -c build/instr-tests-chess/ext_xl__v128uint4/kernel.cc \
    -o /tmp/claude-1000/ext_xl.o

# Function with ISS types (was failing: degraded type info)
xchesscc_wrapper aie2 -c build/instr-tests-chess/prmx_hw_prom__v32w32_pmode_t_smode_t/kernel.cc \
    -o /tmp/claude-1000/prmx.o

# Previously working function (regression check)
xchesscc_wrapper aie2 -c build/instr-tests-chess/broadcast_elem__v16int32_int/kernel.cc \
    -o /tmp/claude-1000/broadcast.o
```

Expected: All three compile cleanly.

- [ ] **Step 3: Run fail-fast compile on first 100 tests**

```bash
./scripts/instr-test.sh --chess --no-hw --no-emu --fail-fast --filter "." -j 1
```

If any test fails, inspect the error, fix the generator or add a
classification rule, and re-run. Iterate until fail-fast passes.

- [ ] **Step 4: Commit any additional fixes**

```bash
git add tools/chess-test-gen.py tools/test_chess_test_gen.py
git commit -m "fix(instr-test): additional Chess codegen fixes from validation"
```

- [ ] **Step 5: Full EMU-only run**

```bash
nice -n 19 ./scripts/instr-test.sh --chess --no-hw -j 16
```

This will take a long time. Run in background and check results when done.
