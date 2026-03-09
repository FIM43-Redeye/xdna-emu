"""Tree-sitter C++ transform module for trace buffer injection.

Parses test.cpp via tree-sitter-cpp and applies three transforms to inject
trace buffer allocation, kernel call argument, and trace data write-out.

Public API:
    patch_test_cpp(source, trace_size=1048576) -> str
    PatchError -- raised when insertion points cannot be found

Designed to be used by trace-prepare.py as the core patching engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node


class PatchError(Exception):
    """Raised when a required insertion point cannot be found in the source."""
    pass


# Tree-sitter setup (module-level, initialized once).
_LANGUAGE = Language(tscpp.language())
_PARSER = Parser(_LANGUAGE)


# ---------------------------------------------------------------------------
# Injected code templates
# ---------------------------------------------------------------------------

_TRACE_BO_TEMPLATE = """\

  // Trace buffer (injected by trace-prepare.py)
  constexpr size_t trace_size = {trace_size};  // {trace_size_comment}
  auto bo_trace = xrt::bo(device, trace_size, XRT_BO_FLAGS_HOST_ONLY,
                           kernel.group_id({next_id}));
  memset(bo_trace.map<void*>(), 0, trace_size);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);"""

_TRACE_WRITEOUT_TEMPLATE = """\

  // Write trace data (injected by trace-prepare.py)
  if (const char* trace_dir = std::getenv("XDNA_TRACE_DIR")) {
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto trace_ptr = bo_trace.map<char*>();
      std::string trace_path = std::string(trace_dir) + "/trace_raw.bin";
      std::ofstream trace_file(trace_path, std::ios::binary);
      trace_file.write(trace_ptr, trace_size);
  }"""


def _human_size(nbytes: int) -> str:
    """Format a byte count as a human-readable string (e.g., '1MB')."""
    if nbytes >= 1048576 and nbytes % 1048576 == 0:
        return f"{nbytes // 1048576}MB"
    if nbytes >= 1024 and nbytes % 1024 == 0:
        return f"{nbytes // 1024}KB"
    return f"{nbytes} bytes"


# ---------------------------------------------------------------------------
# Tree-sitter query helpers
# ---------------------------------------------------------------------------

def _walk_all(node: Node):
    """Yield all descendant nodes in pre-order."""
    yield node
    for child in node.children:
        yield from _walk_all(child)


def _is_xrt_bo_declaration(node: Node, source_bytes: bytes) -> bool:
    """Check if a node is a declaration containing an xrt::bo(...) or
    xrt::ext::bo{...} constructor call."""
    if node.type != "declaration":
        return False
    text = source_bytes[node.start_byte:node.end_byte].decode()
    # Match both xrt::bo(...) and xrt::ext::bo{...} patterns
    return bool(re.search(r'xrt::(?:ext::)?bo[({]', text))


def _find_last_xrt_bo_declaration(root: Node, source_bytes: bytes) -> Optional[Node]:
    """Find the last xrt::bo declaration in the translation unit."""
    last = None
    for node in _walk_all(root):
        if _is_xrt_bo_declaration(node, source_bytes):
            last = node
    return last


def _extract_group_ids(root: Node, source_bytes: bytes) -> list[int]:
    """Extract all integer arguments to kernel.group_id(N) calls.

    Only matches leaf-level group_id call expressions (not their parents)
    to avoid double-counting when a group_id call is nested inside an
    xrt::bo constructor call.
    """
    ids: set[int] = set()
    for node in _walk_all(root):
        if node.type == "call_expression":
            # Only match nodes whose callee ends with .group_id
            func = node.child_by_field_name("function")
            if func is None:
                continue
            func_text = source_bytes[func.start_byte:func.end_byte].decode()
            if not func_text.endswith(".group_id"):
                continue
            # Extract the integer argument
            args = node.child_by_field_name("arguments")
            if args is None:
                continue
            args_text = source_bytes[args.start_byte:args.end_byte].decode()
            m = re.search(r'\((\d+)\)', args_text)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def _find_kernel_call(root: Node, source_bytes: bytes) -> Optional[Node]:
    """Find the kernel(...) call expression that produces a 'run' variable.

    Looks for: auto run = kernel(opcode, bo_instr, ...)
    The kernel call is the call_expression whose callee is 'kernel'.
    """
    for node in _walk_all(root):
        if node.type != "declaration":
            continue
        text = source_bytes[node.start_byte:node.end_byte].decode()
        # Must declare 'run' and call 'kernel'
        if "run" not in text:
            continue
        # Find the call_expression child
        for desc in _walk_all(node):
            if desc.type == "call_expression":
                callee = desc.children[0] if desc.children else None
                if callee is None:
                    continue
                callee_text = source_bytes[callee.start_byte:callee.end_byte].decode()
                if callee_text == "kernel":
                    return desc
    return None


def _find_last_run_wait(root: Node, source_bytes: bytes) -> Optional[Node]:
    """Find the last run.wait() or run.wait2() call.

    Returns the statement node (declaration or expression_statement) that
    contains the call, so we can insert after the complete statement.
    """
    last_stmt = None
    for node in _walk_all(root):
        if node.type != "call_expression":
            continue
        # Check if it's run.wait() or run.wait2()
        callee = node.children[0] if node.children else None
        if callee is None or callee.type != "field_expression":
            continue
        callee_text = source_bytes[callee.start_byte:callee.end_byte].decode()
        if not re.match(r'^run\.wait\d*$', callee_text):
            continue
        # Walk up to find the enclosing statement.
        # run.wait() may be in:
        # - declaration: ert_cmd_state r = run.wait();
        # - expression_statement: run.wait2();
        stmt = _find_enclosing_statement(node, root, source_bytes)
        if stmt is not None:
            last_stmt = stmt
    return last_stmt


def _find_enclosing_statement(node: Node, root: Node, source_bytes: bytes) -> Optional[Node]:
    """Find the nearest enclosing statement (declaration or expression_statement)."""
    # Walk all nodes and build a parent map, then walk up from our node.
    # Since tree-sitter Python doesn't have parent pointers, we search
    # for the statement that contains this node's byte range.
    best = None
    for candidate in _walk_all(root):
        if candidate.type in ("declaration", "expression_statement"):
            if (candidate.start_byte <= node.start_byte and
                    candidate.end_byte >= node.end_byte):
                # Prefer the tightest enclosing statement
                if best is None or (candidate.end_byte - candidate.start_byte) < (best.end_byte - best.start_byte):
                    best = candidate
    return best


def _has_fstream_include(source: str) -> bool:
    """Check if #include <fstream> is already present."""
    return bool(re.search(r'#\s*include\s*<fstream>', source))


def _add_fstream_include(source: str) -> str:
    """Add #include <fstream> after the last system #include."""
    if _has_fstream_include(source):
        return source
    # Find the last #include <...> line and insert after it
    lines = source.split('\n')
    last_sys_include_idx = -1
    for i, line in enumerate(lines):
        if re.match(r'\s*#\s*include\s*<', line):
            last_sys_include_idx = i
    if last_sys_include_idx >= 0:
        lines.insert(last_sys_include_idx + 1, '#include <fstream>')
        return '\n'.join(lines)
    # No system includes found -- insert at the top
    return '#include <fstream>\n' + source


@dataclass
class _InsertionPoints:
    """Byte positions for all three transforms."""
    # Byte offset of the end of the last xrt::bo declaration (insert after)
    last_bo_end: int
    # Byte offset of the closing ')' in the kernel call argument list
    kernel_call_close_paren: int
    # Byte offset of the last character of the run.wait() statement (insert after)
    run_wait_end: int
    # Next group_id to use
    next_group_id: int


def _find_insertion_points(source_bytes: bytes) -> _InsertionPoints:
    """Parse the source and locate all three insertion points.

    Raises PatchError if any required point cannot be found.
    """
    tree = _PARSER.parse(source_bytes)
    root = tree.root_node

    # 1. Last xrt::bo declaration
    last_bo = _find_last_xrt_bo_declaration(root, source_bytes)
    if last_bo is None:
        raise PatchError(
            "Cannot find any xrt::bo declaration. "
            "The source does not follow the expected test.cpp pattern."
        )

    # 2. Group IDs
    group_ids = _extract_group_ids(root, source_bytes)
    next_id = max(group_ids) + 1 if group_ids else 1

    # 3. Kernel call
    kernel_call = _find_kernel_call(root, source_bytes)
    if kernel_call is None:
        raise PatchError(
            "Cannot find kernel(...) call expression. "
            "Expected: auto run = kernel(opcode, bo_instr, ...);"
        )

    # Find the closing paren of the argument list
    arg_list = None
    for child in kernel_call.children:
        if child.type == "argument_list":
            arg_list = child
            break
    if arg_list is None:
        raise PatchError(
            "Cannot find argument list in kernel call expression."
        )
    # The close paren is the last child of argument_list
    close_paren = arg_list.children[-1]
    assert close_paren.type == ")", f"Expected ')' but got {close_paren.type}"

    # 4. Last run.wait()
    run_wait_stmt = _find_last_run_wait(root, source_bytes)
    if run_wait_stmt is None:
        raise PatchError(
            "Cannot find run.wait() or run.wait2() call. "
            "The source does not follow the expected test.cpp pattern."
        )

    return _InsertionPoints(
        last_bo_end=last_bo.end_byte,
        kernel_call_close_paren=close_paren.start_byte,
        next_group_id=next_id,
        run_wait_end=run_wait_stmt.end_byte,
    )


def patch_test_cpp(source: str, trace_size: int = 1048576) -> str:
    """Apply trace buffer injection transforms to a test.cpp source.

    Applies three transforms:
    1. Insert trace buffer allocation after the last xrt::bo declaration
    2. Append bo_trace as the last argument to the kernel() call
    3. Insert trace data write-out after the last run.wait()

    Also adds #include <fstream> if not already present.

    Args:
        source: The C++ source code as a string.
        trace_size: Size of the trace buffer in bytes (default: 1MB).

    Returns:
        The transformed source code.

    Raises:
        PatchError: If required insertion points cannot be found.
    """
    # Skip if already traced (look for the specific declaration pattern)
    if re.search(r'\btrace_size\b', source):
        return source

    source_bytes = source.encode("utf-8")
    points = _find_insertion_points(source_bytes)

    # Build the trace buffer allocation snippet
    trace_bo_snippet = _TRACE_BO_TEMPLATE.format(
        trace_size=trace_size,
        trace_size_comment=_human_size(trace_size),
        next_id=points.next_group_id,
    )

    # Apply transforms in reverse byte order to preserve offsets.
    # Order: run_wait_end > kernel_call_close_paren > last_bo_end
    # (as long as they don't overlap, which they shouldn't).

    result = bytearray(source_bytes)

    # 3. Insert trace write-out after run.wait() statement
    writeout_snippet = _TRACE_WRITEOUT_TEMPLATE.encode("utf-8")
    result[points.run_wait_end:points.run_wait_end] = writeout_snippet

    # 2. Insert ", bo_trace" before the closing paren of kernel call
    # Adjust offset since run_wait_end > kernel_call_close_paren
    # (the run.wait() comes after the kernel call), so the insertion at
    # run_wait_end does not affect kernel_call_close_paren.
    arg_insert = b", bo_trace"
    result[points.kernel_call_close_paren:points.kernel_call_close_paren] = arg_insert

    # 1. Insert trace buffer allocation after last xrt::bo declaration
    # Adjust offset: last_bo_end < kernel_call_close_paren, so no adjustment needed.
    trace_bo_bytes = trace_bo_snippet.encode("utf-8")
    result[points.last_bo_end:points.last_bo_end] = trace_bo_bytes

    result_str = result.decode("utf-8")

    # Add #include <fstream> if needed
    result_str = _add_fstream_include(result_str)

    return result_str
