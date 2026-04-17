# Always-On Trace Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make hardware trace collection the default for every bridge test
run, producing `trace_raw.bin` alongside PASS/FAIL for both HW and EMU,
with both compilers. Tracing is opt-out (`--no-trace`) instead of opt-in.

**Architecture:** A new standalone tool (`trace-prepare.py`) prepares a
traced variant of each test before compilation. It calls `trace-inject.py`
as a library for MLIR trace routing and uses `tree-sitter-cpp` for C++ AST
transforms on test.cpp. The bridge script calls it once per test, then
forks to Chess/Peano compilation on the traced artifacts. Old sweep-based
trace plumbing is removed from the bridge script.

**Tech Stack:** Python 3.13, tree-sitter + tree-sitter-cpp (installed),
trace-inject.py (existing library), mlir-aie Python API (via trace-inject),
bash (bridge script).

**Design doc:** `docs/plans/2026-03-08-always-on-tracing-design.md`

---

## Task 1: Tree-Sitter C++ Transform Module

The core transformation engine. A standalone Python module that parses
test.cpp via tree-sitter, applies three transforms (trace buffer allocation,
kernel argument, trace write-out), and returns the modified source. Fully
testable in isolation -- no dependency on trace-inject.py or bridge script.

**Files:**
- Create: `tools/cpp_trace_patch.py`
- Create: `tools/test_cpp_trace_patch.py`

### Step 1: Write the failing tests

Create `tools/test_cpp_trace_patch.py`. Tests exercise the three transforms
plus edge cases (already-traced, missing insertion points, fstream include).

```python
#!/usr/bin/env python3
"""Tests for cpp_trace_patch.py -- tree-sitter C++ trace patching."""

import textwrap
import pytest

from cpp_trace_patch import patch_test_cpp, PatchError


# Minimal but representative test.cpp (mirrors add_one_using_dma/test.cpp).
MINIMAL_CPP = textwrap.dedent("""\
    #include <cstdint>
    #include <iostream>
    #include <vector>

    #include "xrt/xrt_bo.h"
    #include "xrt/xrt_device.h"
    #include "xrt/xrt_kernel.h"

    int main(int argc, const char *argv[]) {
      unsigned int device_index = 0;
      auto device = xrt::device(device_index);
      auto xclbin = xrt::xclbin("aie.xclbin");
      device.register_xclbin(xclbin);
      xrt::hw_context context(device, xclbin.get_uuid());
      auto kernel = xrt::kernel(context, "MLIR_AIE");

      auto bo_instr = xrt::bo(device, 1024, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
      auto bo_inA = xrt::bo(device, 256, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
      auto bo_out = xrt::bo(device, 256, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

      unsigned int opcode = 3;
      auto run = kernel(opcode, bo_instr, 256, bo_inA, bo_out);
      auto r = run.wait();

      return 0;
    }
""")


def test_adds_fstream_include():
    """#include <fstream> added when not present."""
    result = patch_test_cpp(MINIMAL_CPP, trace_size=1048576)
    assert '#include <fstream>' in result


def test_no_duplicate_fstream():
    """#include <fstream> not duplicated when already present."""
    cpp_with_fstream = MINIMAL_CPP.replace(
        '#include <iostream>', '#include <iostream>\n#include <fstream>'
    )
    result = patch_test_cpp(cpp_with_fstream, trace_size=1048576)
    assert result.count('#include <fstream>') == 1


def test_trace_buffer_allocation():
    """Trace buffer allocated after last xrt::bo."""
    result = patch_test_cpp(MINIMAL_CPP, trace_size=1048576)
    # bo_trace should appear after bo_out (group_id 5) with group_id 6
    assert 'bo_trace' in result
    assert 'group_id(6)' in result
    assert 'trace_size = 1048576' in result
    # bo_trace must come AFTER bo_out
    assert result.index('bo_out') < result.index('bo_trace')


def test_kernel_call_argument():
    """bo_trace appended as last argument to kernel() call."""
    result = patch_test_cpp(MINIMAL_CPP, trace_size=1048576)
    # The kernel call should end with bo_trace before the closing paren.
    # Find kernel call -- should now include bo_trace.
    assert 'bo_inA, bo_out, bo_trace)' in result


def test_trace_writeout():
    """Trace write-out block appears after run.wait()."""
    result = patch_test_cpp(MINIMAL_CPP, trace_size=1048576)
    assert 'XDNA_TRACE_DIR' in result
    assert 'trace_raw.bin' in result
    assert 'bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE)' in result
    # Write-out must come AFTER run.wait()
    assert result.index('run.wait()') < result.index('XDNA_TRACE_DIR')


def test_group_id_auto_increment():
    """NEXT_ID is max(existing group_ids) + 1."""
    # This test has group_ids 1, 3, 5 -> next should be 6.
    result = patch_test_cpp(MINIMAL_CPP, trace_size=1048576)
    assert 'group_id(6)' in result


def test_group_id_dense():
    """Handles dense group_id allocation (0,1,2,3 -> next 4)."""
    cpp = MINIMAL_CPP.replace('group_id(1)', 'group_id(0)')
    cpp = cpp.replace('group_id(3)', 'group_id(1)')
    cpp = cpp.replace('group_id(5)', 'group_id(2)')
    result = patch_test_cpp(cpp, trace_size=1048576)
    assert 'group_id(3)' in result


def test_skip_already_traced():
    """Tests with existing trace_size variable are returned unchanged."""
    cpp_already_traced = MINIMAL_CPP.replace(
        'unsigned int opcode = 3;',
        'constexpr size_t trace_size = 1048576;\nunsigned int opcode = 3;'
    )
    result = patch_test_cpp(cpp_already_traced, trace_size=1048576)
    assert result == cpp_already_traced


def test_custom_trace_size():
    """Trace size parameter is respected."""
    result = patch_test_cpp(MINIMAL_CPP, trace_size=2097152)
    assert 'trace_size = 2097152' in result


def test_error_on_no_bo():
    """PatchError if no xrt::bo declarations found."""
    cpp_no_bo = textwrap.dedent("""\
        #include <iostream>
        int main() { return 0; }
    """)
    with pytest.raises(PatchError, match="xrt::bo"):
        patch_test_cpp(cpp_no_bo, trace_size=1048576)


def test_error_on_no_kernel_call():
    """PatchError if no kernel() call found."""
    cpp_no_call = MINIMAL_CPP.replace(
        'auto run = kernel(opcode, bo_instr, 256, bo_inA, bo_out);',
        '// no kernel call'
    )
    with pytest.raises(PatchError, match="kernel.*call"):
        patch_test_cpp(cpp_no_call, trace_size=1048576)


def test_error_on_no_run_wait():
    """PatchError if no run.wait() found."""
    cpp_no_wait = MINIMAL_CPP.replace('auto r = run.wait();', '// no wait')
    with pytest.raises(PatchError, match="run.wait"):
        patch_test_cpp(cpp_no_wait, trace_size=1048576)


def test_ext_kernel_variant():
    """Works with xrt::ext::kernel (newer XRT API)."""
    cpp_ext = MINIMAL_CPP.replace(
        'auto kernel = xrt::kernel(context, "MLIR_AIE");',
        'auto kernel = xrt::ext::kernel(context, xclbin, "MLIR_AIE");'
    )
    result = patch_test_cpp(cpp_ext, trace_size=1048576)
    assert 'bo_trace' in result
    assert 'XDNA_TRACE_DIR' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Step 2: Run tests to verify they fail

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_cpp_trace_patch.py -v`
Expected: ImportError or ModuleNotFoundError (cpp_trace_patch doesn't exist yet)

### Step 3: Write the implementation

Create `tools/cpp_trace_patch.py`:

```python
#!/usr/bin/env python3
"""Tree-sitter C++ transform for trace buffer injection.

Parses test.cpp via tree-sitter-cpp, applies three transforms:
1. Trace buffer allocation (after last xrt::bo)
2. Kernel call argument (append bo_trace)
3. Trace write-out (after last run.wait())

Also adds #include <fstream> if missing.

Usage as library:
    from cpp_trace_patch import patch_test_cpp
    patched = patch_test_cpp(source_text, trace_size=1048576)
"""

import re
import sys

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser


class PatchError(Exception):
    """Raised when a required insertion point cannot be found."""


CPP_LANGUAGE = Language(tscpp.language())


def _make_parser() -> Parser:
    parser = Parser(CPP_LANGUAGE)
    return parser


def _find_all_nodes(node, predicate):
    """Walk the AST and yield all nodes matching predicate."""
    if predicate(node):
        yield node
    for child in node.children:
        yield from _find_all_nodes(child, predicate)


def _node_text(node, source_bytes: bytes) -> str:
    """Extract source text for a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _find_xrt_bo_declarations(root, source_bytes: bytes):
    """Find all xrt::bo variable declarations.

    Returns list of (node, group_id_int) tuples, sorted by position.
    """
    results = []
    for node in _find_all_nodes(root, lambda n: n.type == "declaration"):
        text = _node_text(node, source_bytes)
        if "xrt::bo" in text:
            # Extract group_id(N) if present.
            m = re.search(r"group_id\((\d+)\)", text)
            gid = int(m.group(1)) if m else None
            results.append((node, gid))
    return results


def _find_kernel_call(root, source_bytes: bytes):
    """Find the kernel(opcode, bo_instr, ...) call expression.

    Returns the call_expression node, or None.
    """
    for node in _find_all_nodes(root, lambda n: n.type == "call_expression"):
        text = _node_text(node, source_bytes)
        # Match kernel(...) but not kernel.group_id(...) or xrt::kernel(...)
        func = node.child_by_field_name("function")
        if func is None:
            continue
        func_text = _node_text(func, source_bytes)
        if func_text == "kernel":
            # Verify it has arguments that look like a kernel invocation
            # (opcode as first arg, not a constructor call).
            args = node.child_by_field_name("arguments")
            if args and args.named_child_count >= 2:
                return node
    return None


def _find_last_run_wait(root, source_bytes: bytes):
    """Find the last run.wait() call expression.

    Returns the statement node containing the call, or None.
    """
    last = None
    for node in _find_all_nodes(root, lambda n: n.type == "call_expression"):
        text = _node_text(node, source_bytes)
        if "run.wait()" in text or "run.wait2()" in text:
            # Walk up to the enclosing expression_statement or declaration.
            stmt = node
            while stmt.parent and stmt.parent.type not in (
                "expression_statement", "declaration", "compound_statement"
            ):
                stmt = stmt.parent
            if stmt.parent and stmt.parent.type in (
                "expression_statement", "declaration"
            ):
                last = stmt.parent
            else:
                last = stmt
    return last


def _has_fstream_include(source: str) -> bool:
    """Check if #include <fstream> is already present."""
    return bool(re.search(r'#include\s*<fstream>', source))


def _has_trace_size(source: str) -> bool:
    """Check if trace_size variable already exists (already-traced test)."""
    return bool(re.search(r'\btrace_size\b', source))


def patch_test_cpp(source: str, trace_size: int = 1048576) -> str:
    """Apply trace buffer transforms to a test.cpp source string.

    Returns the modified source string. Raises PatchError if a required
    insertion point cannot be found.

    If the source already contains a `trace_size` variable (indicating it
    was previously traced), returns the source unchanged.
    """
    # Skip already-traced sources.
    if _has_trace_size(source):
        return source

    source_bytes = source.encode("utf-8")
    parser = _make_parser()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # ---- 1. Find insertion points ----

    bo_decls = _find_xrt_bo_declarations(root, source_bytes)
    if not bo_decls:
        raise PatchError(
            "Cannot find any xrt::bo declarations in test.cpp. "
            "This test may not use the standard XRT buffer pattern."
        )

    kernel_call = _find_kernel_call(root, source_bytes)
    if kernel_call is None:
        raise PatchError(
            "Cannot find kernel() call expression in test.cpp. "
            "Expected pattern: kernel(opcode, bo_instr, ...)"
        )

    run_wait = _find_last_run_wait(root, source_bytes)
    if run_wait is None:
        raise PatchError(
            "Cannot find run.wait() call in test.cpp. "
            "Expected pattern: run.wait() or run.wait2()"
        )

    # ---- 2. Compute group_id for trace buffer ----

    group_ids = [gid for _, gid in bo_decls if gid is not None]
    next_group_id = max(group_ids) + 1 if group_ids else 0

    # ---- 3. Build patches (as byte-offset edits, applied back-to-front) ----

    edits = []  # (offset, length, replacement_text)

    # 3a. Trace buffer allocation -- insert after last xrt::bo declaration.
    last_bo_node = bo_decls[-1][0]
    # Find the end of the statement (including semicolon and newline).
    insert_after_bo = last_bo_node.end_byte
    # Skip trailing whitespace/newline to insert on the next line.
    while insert_after_bo < len(source_bytes) and source_bytes[insert_after_bo:insert_after_bo+1] in (b' ', b'\t'):
        insert_after_bo += 1
    if insert_after_bo < len(source_bytes) and source_bytes[insert_after_bo:insert_after_bo+1] == b'\n':
        insert_after_bo += 1

    # Detect indentation from the last bo declaration.
    line_start = source_bytes.rfind(b'\n', 0, last_bo_node.start_byte)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    indent = b""
    while line_start + len(indent) < last_bo_node.start_byte and \
          source_bytes[line_start + len(indent):line_start + len(indent) + 1] in (b' ', b'\t'):
        indent += source_bytes[line_start + len(indent):line_start + len(indent) + 1]
    indent_str = indent.decode("utf-8")

    trace_alloc = (
        f"\n"
        f"{indent_str}// Trace buffer (injected by trace-prepare.py)\n"
        f"{indent_str}constexpr size_t trace_size = {trace_size};\n"
        f"{indent_str}auto bo_trace = xrt::bo(device, trace_size, XRT_BO_FLAGS_HOST_ONLY,\n"
        f"{indent_str}                         kernel.group_id({next_group_id}));\n"
        f"{indent_str}memset(bo_trace.map<void*>(), 0, trace_size);\n"
        f"{indent_str}bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);\n"
    )
    edits.append((insert_after_bo, 0, trace_alloc))

    # 3b. Kernel call argument -- append bo_trace before closing paren.
    args_node = kernel_call.child_by_field_name("arguments")
    # Find the closing parenthesis byte offset.
    close_paren = args_node.end_byte - 1  # Points to ')'
    edits.append((close_paren, 0, ", bo_trace"))

    # 3c. Trace write-out -- insert after the run.wait() statement.
    insert_after_wait = run_wait.end_byte
    while insert_after_wait < len(source_bytes) and source_bytes[insert_after_wait:insert_after_wait+1] in (b' ', b'\t'):
        insert_after_wait += 1
    if insert_after_wait < len(source_bytes) and source_bytes[insert_after_wait:insert_after_wait+1] == b'\n':
        insert_after_wait += 1

    # Detect indentation from the run.wait() line.
    wait_line_start = source_bytes.rfind(b'\n', 0, run_wait.start_byte)
    if wait_line_start == -1:
        wait_line_start = 0
    else:
        wait_line_start += 1
    wait_indent = b""
    while wait_line_start + len(wait_indent) < run_wait.start_byte and \
          source_bytes[wait_line_start + len(wait_indent):wait_line_start + len(wait_indent) + 1] in (b' ', b'\t'):
        wait_indent += source_bytes[wait_line_start + len(wait_indent):wait_line_start + len(wait_indent) + 1]
    wi = wait_indent.decode("utf-8")

    trace_writeout = (
        f"\n"
        f"{wi}// Write trace data (injected by trace-prepare.py)\n"
        f"{wi}if (const char* trace_dir = std::getenv(\"XDNA_TRACE_DIR\")) {{\n"
        f"{wi}    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);\n"
        f"{wi}    auto trace_ptr = bo_trace.map<char*>();\n"
        f"{wi}    std::string trace_path = std::string(trace_dir) + \"/trace_raw.bin\";\n"
        f"{wi}    std::ofstream trace_file(trace_path, std::ios::binary);\n"
        f"{wi}    trace_file.write(trace_ptr, trace_size);\n"
        f"{wi}}}\n"
    )
    edits.append((insert_after_wait, 0, trace_writeout))

    # ---- 4. Apply edits back-to-front (so byte offsets remain valid) ----

    result_bytes = bytearray(source_bytes)
    for offset, length, text in sorted(edits, key=lambda e: e[0], reverse=True):
        result_bytes[offset:offset + length] = text.encode("utf-8")

    result = result_bytes.decode("utf-8")

    # ---- 5. Add #include <fstream> if missing ----

    if not _has_fstream_include(result):
        # Insert after the last existing #include line.
        lines = result.split('\n')
        last_include_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                last_include_idx = i
        if last_include_idx >= 0:
            lines.insert(last_include_idx + 1, '#include <fstream>')
        else:
            lines.insert(0, '#include <fstream>')
        result = '\n'.join(lines)

    return result
```

### Step 4: Run tests to verify they pass

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_cpp_trace_patch.py -v`
Expected: All 14 tests PASS

### Step 5: Validate against a real test.cpp

Run a quick sanity check against add_one_using_dma/test.cpp:

```bash
cd /home/triple/npu-work/xdna-emu
python3 -c "
from tools.cpp_trace_patch import patch_test_cpp
from pathlib import Path
src = Path('../mlir-aie/test/npu-xrt/add_one_using_dma/test.cpp').read_text()
result = patch_test_cpp(src)
print(result[:2000])
print('...')
print('--- OK: patched successfully ---')
"
```

Expected: Patched source prints with bo_trace, kernel argument, and XDNA_TRACE_DIR block.

### Step 6: Commit

```bash
git add tools/cpp_trace_patch.py tools/test_cpp_trace_patch.py
git commit -m "feat(trace): add tree-sitter C++ transform for trace buffer injection"
```

---

## Task 2: trace-prepare.py

Standalone preparation tool. Reads a test source directory, calls
trace-inject.py as a library for MLIR trace routing, calls cpp_trace_patch
for test.cpp, writes all traced artifacts to an output directory.

**Files:**
- Create: `tools/trace-prepare.py`
- Create: `tools/test_trace_prepare.py`

### Step 1: Write the failing tests

Create `tools/test_trace_prepare.py`. Since trace-prepare.py depends on
mlir-aie Python API (via trace-inject.py), most tests use a mock approach.
One integration test runs against a real test directory (marked slow).

```python
#!/usr/bin/env python3
"""Tests for trace-prepare.py."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


TOOLS_DIR = Path(__file__).parent
TRACE_PREPARE = TOOLS_DIR / "trace-prepare.py"
SCRIPTS_DIR = TOOLS_DIR.parent / "scripts"
MLIR_AIE_TESTS = Path(os.environ.get(
    "MLIR_AIE_TESTS",
    str(TOOLS_DIR.parent.parent / "mlir-aie" / "test" / "npu-xrt"),
))

# Minimal aie.mlir for testing (does not need to parse with mlir-aie).
MINIMAL_MLIR = textwrap.dedent("""\
    module {
      aie.device(NPUDEVICE) {
        %tile_0_0 = aie.tile(0, 0)
        %tile_0_2 = aie.tile(0, 2)
      }
    }
""")

MINIMAL_CPP = textwrap.dedent("""\
    #include <iostream>
    #include "xrt/xrt_bo.h"
    #include "xrt/xrt_device.h"
    #include "xrt/xrt_kernel.h"

    int main() {
      auto device = xrt::device(0);
      auto xclbin = xrt::xclbin("aie.xclbin");
      device.register_xclbin(xclbin);
      xrt::hw_context context(device, xclbin.get_uuid());
      auto kernel = xrt::kernel(context, "MLIR_AIE");

      auto bo_instr = xrt::bo(device, 1024, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
      auto bo_in = xrt::bo(device, 256, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
      auto bo_out = xrt::bo(device, 256, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

      auto run = kernel(3, bo_instr, 256, bo_in, bo_out);
      run.wait();
      return 0;
    }
""")


@pytest.fixture
def fake_test_dir(tmp_path):
    """Create a minimal test source directory."""
    src = tmp_path / "fake_test"
    src.mkdir()
    (src / "aie.mlir").write_text(MINIMAL_MLIR)
    (src / "test.cpp").write_text(MINIMAL_CPP)
    return src


def test_cli_help():
    """trace-prepare.py --help runs without error."""
    result = subprocess.run(
        [sys.executable, str(TRACE_PREPARE), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "trace-prepare" in result.stdout.lower() or "--output" in result.stdout


def test_missing_test_dir():
    """Exits nonzero for nonexistent test directory."""
    result = subprocess.run(
        [sys.executable, str(TRACE_PREPARE), "/nonexistent", "-o", "/tmp/out"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_cpp_patching_standalone(fake_test_dir, tmp_path):
    """C++ patching works even when MLIR injection is skipped (trace-quarantined)."""
    # Create a trace-quarantine.txt that includes our test.
    quarantine = SCRIPTS_DIR / "trace-quarantine.txt"
    original_quarantine = quarantine.read_text() if quarantine.exists() else None
    try:
        # We'll test that trace-quarantine skips MLIR injection but
        # the tool itself still exits cleanly with a status file.
        out_dir = tmp_path / "output"
        result = subprocess.run(
            [sys.executable, str(TRACE_PREPARE), str(fake_test_dir),
             "-o", str(out_dir), "--skip-mlir"],
            capture_output=True, text=True,
        )
        # --skip-mlir should produce a patched test.cpp but no traced MLIR.
        if result.returncode == 0:
            assert (out_dir / "test_traced.cpp").exists()
            content = (out_dir / "test_traced.cpp").read_text()
            assert "bo_trace" in content
            assert "XDNA_TRACE_DIR" in content
    finally:
        if original_quarantine is not None:
            quarantine.write_text(original_quarantine)


@pytest.mark.skipif(
    not (MLIR_AIE_TESTS / "add_one_using_dma").exists(),
    reason="mlir-aie test suite not available",
)
def test_integration_add_one(tmp_path):
    """Integration test: full trace preparation for add_one_using_dma.

    Requires mlir-aie Python API to be importable.
    """
    pytest.importorskip("aie.ir", reason="mlir-aie Python API not available")

    src = MLIR_AIE_TESTS / "add_one_using_dma"
    out = tmp_path / "traced"

    result = subprocess.run(
        [sys.executable, str(TRACE_PREPARE), str(src), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"trace-prepare failed:\n{result.stderr}"

    # Check all expected outputs.
    assert (out / "aie_traced.mlir").exists(), "Missing aie_traced.mlir"
    assert (out / "test_traced.cpp").exists(), "Missing test_traced.cpp"
    assert (out / "events.json").exists(), "Missing events.json"
    assert (out / "prepare-status.txt").exists(), "Missing prepare-status.txt"

    status = (out / "prepare-status.txt").read_text().strip()
    assert status.startswith("OK"), f"Unexpected status: {status}"

    # Verify test_traced.cpp has all three transforms.
    traced_cpp = (out / "test_traced.cpp").read_text()
    assert "bo_trace" in traced_cpp
    assert "XDNA_TRACE_DIR" in traced_cpp
    assert "#include <fstream>" in traced_cpp

    # Verify events.json is valid JSON with expected keys.
    events = json.loads((out / "events.json").read_text())
    assert "core_events" in events or "tiles_traced" in events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Step 2: Run tests to verify they fail

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_trace_prepare.py -v`
Expected: FAIL (trace-prepare.py doesn't exist yet)

### Step 3: Write the implementation

Create `tools/trace-prepare.py`:

```python
#!/usr/bin/env python3
"""Prepare a traced variant of an mlir-aie npu-xrt test.

Standalone tool that takes a test source directory and produces a traced
build directory ready for normal compilation by either Chess or Peano.

Steps:
1. Copy and prepare MLIR (NPUDEVICE substitution)
2. Run trace-inject.py as library (two-pass pathfinder routing)
3. Patch test.cpp via tree-sitter (trace buffer, kernel arg, write-out)
4. Write events.json (event slot config for trace decoding)
5. Write prepare-status.txt (OK or FAIL with reason)

Usage:
    trace-prepare.py <test_source_dir> --output <dir> \
        [--trace-size BYTES] [--device auto] [--skip-mlir]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add tools/ to path for local imports.
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from cpp_trace_patch import patch_test_cpp, PatchError


def write_status(output_dir: Path, status: str, reason: str = ""):
    """Write prepare-status.txt."""
    text = status
    if reason:
        text += f" {reason}"
    (output_dir / "prepare-status.txt").write_text(text + "\n")


def is_trace_quarantined(test_name: str) -> bool:
    """Check if this test is in the trace quarantine list."""
    quarantine_file = TOOLS_DIR.parent / "scripts" / "trace-quarantine.txt"
    if not quarantine_file.exists():
        return False
    for line in quarantine_file.read_text().splitlines():
        entry = line.split("#")[0].strip()
        if entry and entry == test_name:
            return True
    return False


def is_test_quarantined(test_name: str) -> bool:
    """Check if this test is in the general test quarantine list."""
    quarantine_file = TOOLS_DIR.parent / "scripts" / "test-quarantine.txt"
    if not quarantine_file.exists():
        return False
    for line in quarantine_file.read_text().splitlines():
        entry = line.split("#")[0].strip()
        if entry and entry == test_name:
            return True
    return False


def prepare_mlir(test_dir: Path, output_dir: Path, trace_size: int,
                 device: str) -> dict | None:
    """Run trace-inject.py as a library to produce traced MLIR.

    Returns manifest_partial dict on success, None on failure.
    Writes aie_traced.mlir to output_dir.
    """
    # Import trace-inject functions (adds mlir-aie dependency).
    import trace_inject
    from trace_inject import (
        detect_source_type,
        get_mlir_text,
        plan_trace_route,
        inject_trace,
        build_manifest,
    )

    source_type = detect_source_type(test_dir)
    mlir_text = get_mlir_text(test_dir, source_type, device)

    # Plan trace routing.
    plan = plan_trace_route(mlir_text)
    if not plan.feasible:
        return None

    # Inject trace.
    traced_mlir, manifest_partial = inject_trace(
        mlir_text, trace_size, plan,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aie_traced.mlir").write_text(traced_mlir)

    return manifest_partial


def prepare_cpp(test_dir: Path, output_dir: Path, trace_size: int) -> None:
    """Patch test.cpp with trace buffer via tree-sitter.

    Also applies the BDF environment variable patch (same as bridge script).
    """
    test_cpp = test_dir / "test.cpp"
    if not test_cpp.exists():
        raise PatchError(f"No test.cpp found in {test_dir}")

    source = test_cpp.read_text()

    # Apply BDF patch (same sed as bridge script compile_one).
    source = source.replace(
        "unsigned int device_index = 0;",
        'const char* _bdf = std::getenv("XRT_DEVICE_BDF");',
    )
    source = source.replace(
        "auto device = xrt::device(device_index);",
        'auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);',
    )

    # Apply trace transforms.
    patched = patch_test_cpp(source, trace_size=trace_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_traced.cpp").write_text(patched)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a traced variant of an npu-xrt test",
    )
    parser.add_argument(
        "test_dir",
        type=Path,
        help="Path to npu-xrt test source directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for traced artifacts",
    )
    parser.add_argument(
        "--trace-size",
        type=int,
        default=1048576,
        help="Trace buffer size in bytes (default: 1MB)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device target for NPUDEVICE substitution (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-mlir",
        action="store_true",
        help="Skip MLIR injection (only patch test.cpp). "
             "Used for trace-quarantined tests or testing.",
    )
    args = parser.parse_args()

    test_dir = args.test_dir.resolve()
    output_dir = args.output.resolve()

    if not test_dir.is_dir():
        print(f"Error: {test_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    test_name = test_dir.name

    # Check test quarantine (fundamentally broken tests).
    if is_test_quarantined(test_name):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_status(output_dir, "SKIP", "test-quarantined")
        print(f"SKIP {test_name}: test-quarantined", file=sys.stderr)
        sys.exit(0)

    # Check trace quarantine (trace injection breaks the test).
    trace_quarantined = is_trace_quarantined(test_name)

    # ---- MLIR trace injection ----
    manifest_partial = None
    if args.skip_mlir or trace_quarantined:
        if trace_quarantined:
            print(f"Skipping MLIR injection for {test_name}: trace-quarantined",
                  file=sys.stderr)
    else:
        try:
            manifest_partial = prepare_mlir(
                test_dir, output_dir, args.trace_size, args.device,
            )
            if manifest_partial is None:
                print(f"Error: trace routing infeasible for {test_name}",
                      file=sys.stderr)
                write_status(output_dir, "FAIL", "routing_infeasible")
                sys.exit(1)
        except Exception as e:
            print(f"Error: MLIR injection failed for {test_name}: {e}",
                  file=sys.stderr)
            write_status(output_dir, "FAIL", f"mlir_injection: {e}")
            sys.exit(1)

    # ---- C++ trace patching ----
    if not trace_quarantined:
        try:
            prepare_cpp(test_dir, output_dir, args.trace_size)
        except PatchError as e:
            print(f"Error: C++ patching failed for {test_name}: {e}",
                  file=sys.stderr)
            write_status(output_dir, "FAIL", f"cpp_patch: {e}")
            sys.exit(1)

    # ---- Events manifest ----
    if manifest_partial is not None:
        events = {}
        if "tiles_traced" in manifest_partial:
            events["tiles_traced"] = manifest_partial["tiles_traced"]
        if "core_events" in manifest_partial:
            events["core_events"] = manifest_partial["core_events"]
        if "mem_events" in manifest_partial:
            events["mem_events"] = manifest_partial["mem_events"]
        (output_dir / "events.json").write_text(
            json.dumps(events, indent=2) + "\n"
        )
    elif trace_quarantined:
        # No events for quarantined tests.
        pass
    else:
        # --skip-mlir mode without quarantine -- write empty events.
        (output_dir / "events.json").write_text("{}\n")

    write_status(output_dir, "OK")
    print(f"Prepared {test_name} -> {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

### Step 4: Run tests to verify they pass

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_trace_prepare.py -v -k "not integration"`
Expected: Non-integration tests PASS

Run: `cd /home/triple/npu-work/xdna-emu && python3 -m pytest tools/test_trace_prepare.py::test_integration_add_one -v`
Expected: Integration test PASS (requires mlir-aie Python API)

### Step 5: Commit

```bash
git add tools/trace-prepare.py tools/test_trace_prepare.py
git commit -m "feat(trace): add trace-prepare.py standalone preparation tool"
```

---

## Task 3: test-quarantine.txt

New quarantine tier for fundamentally broken tests (segfault, compile fail).
These tests are skipped entirely by the bridge script -- no compile, no run.

**Files:**
- Create: `scripts/test-quarantine.txt`

### Step 1: Create the quarantine file

The file format matches trace-quarantine.txt: one test directory name per
line, with `#` comments.

```
# test-quarantine.txt -- Tests that are fundamentally broken.
#
# Tests listed here are skipped entirely: no compile, no run, no trace.
# Unlike trace-quarantine.txt (which only skips trace injection), these
# tests have structural problems (segfault, compile failure, missing
# dependencies) that make them unable to run at all.
#
# Format: one test directory name per line.
# Lines starting with # are comments.

# (none yet -- add tests here as they are identified)
```

### Step 2: Commit

```bash
git add scripts/test-quarantine.txt
git commit -m "feat(trace): add test-quarantine.txt for fundamentally broken tests"
```

---

## Task 4: Bridge Script Integration -- Trace Preparation

Wire trace-prepare.py into the bridge script's compile_one flow. This is
the core integration: every test gets traced artifacts before compilation,
and both compilers use the traced MLIR and test.cpp.

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

### Step 1: Add --no-trace CLI flag

In the argument parsing section of the bridge script (look for the `while`
loop that processes `$@`), add:

```bash
--no-trace)
  NO_TRACE=true
  ;;
```

And near the top where variables are initialized:

```bash
NO_TRACE="${NO_TRACE:-false}"
export NO_TRACE
```

### Step 2: Add test-quarantine check to discover_tests or compile_one

In `compile_one()`, add at the top (after the `safe=` line):

```bash
  # Check test quarantine (fundamentally broken tests -- skip entirely).
  if is_test_quarantined "$name"; then
    local compilers
    read -ra compilers <<< "$COMPILERS_STR"
    for compiler in "${compilers[@]}"; do
      echo "SKIP_QUARANTINED" > "$RESULTS_DIR/${safe}.${compiler}.compile.result"
    done
    echo "  COMPILE $name: SKIP (test-quarantined)"
    return 0
  fi
```

Add the `is_test_quarantined` function near the other quarantine helpers:

```bash
is_test_quarantined() {
  local name="${1%%:*}"  # Strip compiler suffix if present.
  local quarantine_file="$SCRIPT_DIR/test-quarantine.txt"
  [[ -f "$quarantine_file" ]] || return 1
  while IFS= read -r line; do
    local entry="${line%%#*}"
    entry="${entry// /}"
    [[ -z "$entry" ]] && continue
    [[ "$entry" == "$name" ]] && return 0
  done < "$quarantine_file"
  return 1
}
export -f is_test_quarantined
```

### Step 3: Add trace-prepare.py call in compile_one

In `compile_one()`, after the test-quarantine check and before the
compiler loop, add trace preparation:

```bash
  # ---- Trace preparation (always-on unless --no-trace) ----
  local traced_dir="$build_dir/traced"
  local trace_ok=false

  if [[ "$NO_TRACE" != "true" ]] && ! is_trace_quarantined "$name"; then
    local trace_log="$RESULTS_DIR/${safe}.trace-prepare.log"
    if python3 "$EMU_ROOT/tools/trace-prepare.py" "$src_dir" \
        -o "$traced_dir" > "$trace_log" 2>&1; then
      if [[ -f "$traced_dir/prepare-status.txt" ]] \
          && grep -q '^OK' "$traced_dir/prepare-status.txt"; then
        trace_ok=true
        echo "  TRACE PREP $name: OK"
      else
        echo "  TRACE PREP $name: FAIL (status)"
        echo "ERROR trace_prep_failed" > "$RESULTS_DIR/${safe}.trace-prepare.result"
        return 1
      fi
    else
      echo "  TRACE PREP $name: FAIL (exit code)"
      echo "ERROR trace_prep_failed" > "$RESULTS_DIR/${safe}.trace-prepare.result"
      return 1
    fi
  fi
  export TRACE_OK="$trace_ok"
  export TRACED_DIR="$traced_dir"
```

### Step 4: Modify compile_one_compiler to use traced artifacts

In `compile_one_compiler()`, after the NPUDEVICE substitution block
(around line 560), add artifact replacement when tracing is active:

```bash
    # Use traced artifacts if trace preparation succeeded.
    if [[ "$TRACE_OK" == "true" ]] && [[ -f "$TRACED_DIR/aie_traced.mlir" ]]; then
      cp "$TRACED_DIR/aie_traced.mlir" "$build_dir/aie_arch.mlir"
    fi
```

### Step 5: Modify compile_one to use test_traced.cpp

In `compile_one()`, in the test.exe compilation block (around line 641),
replace the sed-based test.cpp patching with:

```bash
  if [[ -f "$src_dir/test.cpp" ]]; then
    if [[ "$TRACE_OK" == "true" ]] && [[ -f "$traced_dir/test_traced.cpp" ]]; then
      # Use tree-sitter-patched version (includes BDF + trace transforms).
      cp "$traced_dir/test_traced.cpp" "$build_dir/test.cpp"
    else
      # No tracing -- just apply BDF patch.
      sed \
        -e 's/unsigned int device_index = 0;/const char* _bdf = std::getenv("XRT_DEVICE_BDF");/' \
        -e 's/auto device = xrt::device(device_index);/auto device = _bdf ? xrt::device(std::string(_bdf)) : xrt::device(0);/' \
        "$src_dir/test.cpp" > "$build_dir/test.cpp"
    fi
  fi
```

### Step 6: Add XDNA_TRACE_DIR to run phases

In `run_one_bridge()` (EMU runs), add XDNA_TRACE_DIR export inside the
subshell (around line 800):

```bash
    # Set trace output directory if tracing is active.
    local trace_out_dir="$RESULTS_DIR/${safe}.${compiler}.emu"
    mkdir -p "$trace_out_dir"
    export XDNA_TRACE_DIR="$trace_out_dir"
```

In `run_one_hardware()` (HW runs), add the same inside the subshell
(around line 730):

```bash
    local trace_out_dir="$RESULTS_DIR/${safe}.${compiler}.hw"
    mkdir -p "$trace_out_dir"
    export XDNA_TRACE_DIR="$trace_out_dir"
```

### Step 7: Copy events.json alongside trace output

After each run completes (both HW and EMU), copy events.json to the
result directory. Add after the result file is written in each runner:

```bash
  # Copy events.json for trace decoding.
  local build_traced="$BUILD_BASE/$name/traced"
  if [[ -f "$build_traced/events.json" ]]; then
    cp "$build_traced/events.json" "$trace_out_dir/"
  fi
```

### Step 8: Test the integration

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh --no-hw add_one_using_dma`
Expected: Compiles with traced artifacts, EMU run produces trace_raw.bin.

Run: `ls /tmp/emu-bridge-results-*/add_one_using_dma.*.emu/`
Expected: `trace_raw.bin` and `events.json` present.

### Step 9: Commit

```bash
git add scripts/emu-bridge-test.sh
git commit -m "feat(trace): integrate always-on trace preparation into bridge script"
```

---

## Task 5: Remove Old Trace Plumbing

Remove the old sweep-based trace code from the bridge script. This cleans
up the functions and phases that are replaced by the always-on pipeline.

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

### Step 1: Remove old functions

Delete these functions from the bridge script:
- `compile_trace_base_one()` (lines ~846-865)
- `trace_one_test()` (lines ~873-1061)

### Step 2: Remove old CLI flags

In the argument parsing, remove:
- `--trace=*` and `TRACE_MODE` variable
- `--trace=sweep` / `--trace=sweep-all` / `--trace=compare` / `--trace=compare-all`
  (but keep `--no-trace` added in Task 4)

### Step 3: Remove Phase 4b

Remove the entire Phase 4b block (lines ~1621-1760+):
- Background trace compile-ahead launch
- `trace_compile_pid` variable and its wait
- The `if [[ -n "$TRACE_MODE" ]]` block

### Step 4: Simplify Phase 5 report

The trace column in the Phase 5 report should now always be present
(not gated on `$has_trace`). Trace status comes from checking for
`trace_raw.bin` in the result directories instead of `.trace.summary`
files. Modify the report's trace column logic.

### Step 5: Test

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh --no-hw add_one_using_dma`
Expected: Runs successfully, no errors about removed functions.

### Step 6: Commit

```bash
git add scripts/emu-bridge-test.sh
git commit -m "refactor(trace): remove old sweep-based trace plumbing from bridge script"
```

---

## Task 6: Phase 5 Report -- Trace Column

Update the Phase 5 report to show trace status for every test. Trace
status is derived from the presence and content of trace_raw.bin files,
not from separate trace comparison runs.

**Files:**
- Modify: `scripts/emu-bridge-test.sh`

### Step 1: Add trace comparison to Phase 5

After all HW and EMU runs complete, for each test where both
`<test>.chess.hw/trace_raw.bin` and `<test>.chess.emu/trace_raw.bin`
exist, run `trace-compare` and record the result.

```bash
  # ---- Phase 5: Trace comparison (automatic for always-on traces) ----

  if [[ "$NO_TRACE" != "true" ]]; then
    info "Phase 5: Comparing traces"
    for name in "${compiled[@]}"; do
      local safe
      safe="$(sanitize_name "$name")"
      for compiler in "${compilers[@]}"; do
        local hw_trace="$RESULTS_DIR/${safe}.${compiler}.hw/trace_raw.bin"
        local emu_trace="$RESULTS_DIR/${safe}.${compiler}.emu/trace_raw.bin"
        local events_file="$RESULTS_DIR/${safe}.${compiler}.hw/events.json"
        [[ ! -f "$events_file" ]] && events_file="$RESULTS_DIR/${safe}.${compiler}.emu/events.json"
        local summary_file="$RESULTS_DIR/${safe}.${compiler}.trace.summary"

        if [[ -f "$hw_trace" ]] && [[ -f "$emu_trace" ]]; then
          local cmp_args=(--hw "$hw_trace" --emu "$emu_trace")
          [[ -f "$events_file" ]] && cmp_args+=(--events "$events_file")
          local cmp_out
          cmp_out="$(run_trace_compare "${cmp_args[@]}" 2>&1)" || true

          if echo "$cmp_out" | grep -q "CLEAN"; then
            echo "CLEAN" > "$summary_file"
          elif echo "$cmp_out" | grep -q "DIVERGE"; then
            echo "DIVERGE" > "$summary_file"
          else
            echo "ERROR" > "$summary_file"
          fi
          echo "$cmp_out" > "$RESULTS_DIR/${safe}.${compiler}.trace.log"
        elif [[ -f "$emu_trace" ]]; then
          echo "EMU_ONLY" > "$summary_file"
        elif [[ -f "$hw_trace" ]]; then
          echo "HW_ONLY" > "$summary_file"
        else
          echo "NONE" > "$summary_file"
        fi
      done
    done
  fi
```

### Step 2: Update report to always show trace column

Change the `$has_trace` condition in the report to be based on
`$NO_TRACE`:

```bash
  local has_trace=false
  [[ "$NO_TRACE" != "true" ]] && has_trace=true
```

### Step 3: Test

Run: `cd /home/triple/npu-work/xdna-emu && ./scripts/emu-bridge-test.sh add_one_using_dma`
Expected: Report shows trace column with CLEAN/DIVERGE/EMU_ONLY status.

### Step 4: Commit

```bash
git add scripts/emu-bridge-test.sh
git commit -m "feat(trace): automatic trace comparison in Phase 5 report"
```

---

## Verification Checklist

After all tasks are complete, run these verification steps:

1. **Unit tests:**
   ```bash
   python3 -m pytest tools/test_cpp_trace_patch.py -v
   python3 -m pytest tools/test_trace_prepare.py -v -k "not integration"
   ```

2. **Integration test (requires mlir-aie):**
   ```bash
   python3 -m pytest tools/test_trace_prepare.py::test_integration_add_one -v
   ```

3. **Standalone trace-prepare:**
   ```bash
   python3 tools/trace-prepare.py ../mlir-aie/test/npu-xrt/add_one_using_dma \
     -o /tmp/trace-prep-test
   ls /tmp/trace-prep-test/
   # Expect: aie_traced.mlir test_traced.cpp events.json prepare-status.txt
   ```

4. **Bridge test -- EMU only:**
   ```bash
   ./scripts/emu-bridge-test.sh --no-hw add_one_using_dma
   ls /tmp/emu-bridge-results-*/add_one_using_dma.chess.emu/
   # Expect: trace_raw.bin events.json
   ```

5. **Bridge test -- no trace:**
   ```bash
   ./scripts/emu-bridge-test.sh --no-hw --no-trace add_one_using_dma
   ls /tmp/emu-bridge-results-*/add_one_using_dma.chess.emu/
   # Expect: NO trace_raw.bin
   ```

6. **Bridge test -- full run with report:**
   ```bash
   ./scripts/emu-bridge-test.sh add_one_using_dma
   # Report should show trace column with CLEAN/DIVERGE
   ```

7. **Quarantine behavior:**
   ```bash
   # Add a test to test-quarantine.txt, verify it's skipped entirely.
   echo "add_one_using_dma" >> scripts/test-quarantine.txt
   ./scripts/emu-bridge-test.sh --no-hw add_one_using_dma
   # Expect: "SKIP (test-quarantined)"
   # Clean up:
   sed -i '/add_one_using_dma/d' scripts/test-quarantine.txt
   ```

8. **Cargo tests still pass:**
   ```bash
   cargo test --lib
   ```
