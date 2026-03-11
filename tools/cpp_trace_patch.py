"""Tree-sitter C++ transform module for trace buffer injection.

Parses test.cpp via tree-sitter-cpp and applies three transforms to inject
trace buffer allocation, kernel call argument, and trace data write-out.

Supports three XRT API patterns:

1. **Direct call**: ``auto run = kernel(opcode, bo_instr, ..., bo_out);``
   -> appends ``, bo_trace`` before the closing paren.

2. **set_arg**: ``auto run = xrt::run(kernel); run.set_arg(N, bo_out);``
   -> inserts ``run.set_arg(N+1, bo_trace);`` after the last set_arg.

3. **Multi-kernel**: ``kernel0(opcode, ...); kernel1(opcode, ...);``
   -> traces the first kernel (pattern 1 or 2).

All patterns also handle ``runlist.wait()`` in addition to ``run.wait()``.

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
                           {kernel_name}.group_id({next_id}));
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


def _extract_group_ids(
    root: Node, source_bytes: bytes, kernel_name: str = "kernel"
) -> list[int]:
    """Extract integer arguments to <kernel_name>.group_id(N) calls.

    When *kernel_name* is provided, only extracts group_ids for that
    specific kernel variable (e.g., ``kernel0.group_id(3)`` but not
    ``kernel1.group_id(3)``).  This is important for multi-kernel tests
    where different kernels reuse the same group_id numbers.

    Only matches leaf-level group_id call expressions (not their parents)
    to avoid double-counting when a group_id call is nested inside an
    xrt::bo constructor call.
    """
    expected_suffix = f"{kernel_name}.group_id"
    ids: set[int] = set()
    for node in _walk_all(root):
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if func is None:
                continue
            func_text = source_bytes[func.start_byte:func.end_byte].decode()
            if not func_text.endswith(".group_id"):
                continue
            # For multi-kernel tests, only match the specific kernel
            if not func_text.endswith(expected_suffix):
                continue
            args = node.child_by_field_name("arguments")
            if args is None:
                continue
            args_text = source_bytes[args.start_byte:args.end_byte].decode()
            m = re.search(r'\((\d+)\)', args_text)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def _find_kernel_call(
    root: Node, source_bytes: bytes
) -> tuple[Optional[Node], str]:
    """Find the first kernel call expression that produces a run variable.

    Matches single-kernel patterns (``auto run = kernel(...)``), and
    multi-kernel patterns (``auto run0 = kernel0(...)``).

    Returns ``(call_node, kernel_name)`` where *kernel_name* is the
    callee identifier (e.g., ``"kernel"`` or ``"kernel0"``).  Returns
    ``(None, "")`` if no kernel call is found.
    """
    # First pass: look in declarations (auto run = kernel(...))
    for node in _walk_all(root):
        if node.type != "declaration":
            continue
        text = source_bytes[node.start_byte:node.end_byte].decode()
        if "run" not in text:
            continue
        for desc in _walk_all(node):
            if desc.type == "call_expression":
                callee = desc.children[0] if desc.children else None
                if callee is None:
                    continue
                callee_text = source_bytes[callee.start_byte:callee.end_byte].decode()
                if re.match(r'^kernel\w*$', callee_text):
                    return desc, callee_text

    # Second pass: look in expression statements (kernel0(...).wait2())
    for node in _walk_all(root):
        if node.type != "expression_statement":
            continue
        for desc in _walk_all(node):
            if desc.type == "call_expression":
                callee = desc.children[0] if desc.children else None
                if callee is None:
                    continue
                callee_text = source_bytes[callee.start_byte:callee.end_byte].decode()
                if re.match(r'^kernel\w*$', callee_text):
                    return desc, callee_text

    return None, ""


def _find_xrt_run_constructor(
    root: Node, source_bytes: bytes
) -> tuple[Optional[str], Optional[str]]:
    """Find ``auto run = xrt::run(kernel)`` and extract both variable names.

    Returns ``(run_var_name, kernel_name)`` or ``(None, None)`` if not found.
    E.g., for ``auto run1 = xrt::run(kernel0)`` returns ``("run1", "kernel0")``.
    """
    for node in _walk_all(root):
        if node.type != "declaration":
            continue
        text = source_bytes[node.start_byte:node.end_byte].decode()
        if "xrt::run" not in text:
            continue
        # Match: auto <run_var> = xrt::run(<kernel_var>)
        # or:    xrt::run <run_var> = xrt::run(<kernel_var>)
        m = re.search(r'(?:auto|xrt::run)\s+(run\w*)\s*=\s*xrt::run\((\w+)\)', text)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _find_last_set_arg(
    root: Node, source_bytes: bytes, run_var: str
) -> tuple[Optional[Node], int]:
    """Find the last ``run_var.set_arg(N, ...)`` call and extract N.

    Returns ``(statement_node, max_arg_index)`` or ``(None, -1)`` if
    no set_arg calls are found.
    """
    last_stmt = None
    max_idx = -1
    for node in _walk_all(root):
        if node.type != "call_expression":
            continue
        func = node.child_by_field_name("function")
        if func is None:
            continue
        func_text = source_bytes[func.start_byte:func.end_byte].decode()
        if func_text != f"{run_var}.set_arg":
            continue
        # Extract the first argument (the index)
        args = node.child_by_field_name("arguments")
        if args is None:
            continue
        args_text = source_bytes[args.start_byte:args.end_byte].decode()
        m = re.search(r'\((\d+)', args_text)
        if m:
            idx = int(m.group(1))
            # Always track the last set_arg by source position
            stmt = _find_enclosing_statement(node, root, source_bytes)
            if stmt is not None:
                if idx >= max_idx:
                    max_idx = idx
                    last_stmt = stmt
    return last_stmt, max_idx


def _find_last_wait(root: Node, source_bytes: bytes) -> Optional[Node]:
    """Find the last wait call in any form.

    Matches:
    - ``run.wait()`` / ``run.wait2()`` / ``run0.wait()``
    - ``runlist.wait()``
    - ``kernel0(...).wait2()`` (chained call, no run variable)

    Returns the statement node (declaration or expression_statement) that
    contains the call, so we can insert after the complete statement.
    """
    last_stmt = None
    for node in _walk_all(root):
        if node.type != "call_expression":
            continue
        # Look at the full call text for a .wait pattern
        call_text = source_bytes[node.start_byte:node.end_byte].decode()
        if ".wait" not in call_text:
            continue
        # The callee must be a field_expression ending with .wait
        callee = node.children[0] if node.children else None
        if callee is None or callee.type != "field_expression":
            continue
        callee_text = source_bytes[callee.start_byte:callee.end_byte].decode()
        if not re.search(r'\.wait\d*$', callee_text):
            continue
        stmt = _find_enclosing_statement(node, root, source_bytes)
        if stmt is not None:
            last_stmt = stmt
    return last_stmt


def _find_enclosing_statement(node: Node, root: Node, source_bytes: bytes) -> Optional[Node]:
    """Find the nearest enclosing statement (declaration or expression_statement)."""
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


# ---------------------------------------------------------------------------
# Injection modes
# ---------------------------------------------------------------------------

@dataclass
class _DirectCallPoints:
    """Injection via direct kernel call: kernel(opcode, ..., bo_trace)."""
    last_bo_end: int
    kernel_call_close_paren: int
    wait_end: int
    next_group_id: int
    kernel_name: str


@dataclass
class _SetArgPoints:
    """Injection via set_arg: run.set_arg(N+1, bo_trace)."""
    last_bo_end: int
    last_set_arg_end: int
    next_arg_index: int
    wait_end: int
    next_group_id: int
    kernel_name: str
    run_var: str


def _find_insertion_points(
    source_bytes: bytes,
) -> _DirectCallPoints | _SetArgPoints:
    """Parse the source and locate injection points.

    Tries direct-call pattern first, falls back to set_arg pattern.
    Raises PatchError if neither pattern is found.
    """
    tree = _PARSER.parse(source_bytes)
    root = tree.root_node

    # 1. Last xrt::bo declaration (needed for both patterns).
    last_bo = _find_last_xrt_bo_declaration(root, source_bytes)
    if last_bo is None:
        raise PatchError(
            "Cannot find any xrt::bo declaration. "
            "The source does not follow the expected test.cpp pattern."
        )

    # 2. Last wait (run.wait, runlist.wait, etc.) -- needed for both.
    wait_stmt = _find_last_wait(root, source_bytes)
    if wait_stmt is None:
        raise PatchError(
            "Cannot find any wait() call (run.wait, runlist.wait, etc.)."
        )

    # 3. Try direct-call pattern first: kernel(opcode, ...)
    kernel_call, kernel_name = _find_kernel_call(root, source_bytes)
    if kernel_call is not None:
        group_ids = _extract_group_ids(root, source_bytes, kernel_name)
        next_id = max(group_ids) + 1 if group_ids else 1

        arg_list = None
        for child in kernel_call.children:
            if child.type == "argument_list":
                arg_list = child
                break
        if arg_list is None:
            raise PatchError(
                "Cannot find argument list in kernel call expression."
            )
        close_paren = arg_list.children[-1]
        assert close_paren.type == ")", f"Expected ')' but got {close_paren.type}"

        return _DirectCallPoints(
            last_bo_end=last_bo.end_byte,
            kernel_call_close_paren=close_paren.start_byte,
            wait_end=wait_stmt.end_byte,
            next_group_id=next_id,
            kernel_name=kernel_name,
        )

    # 4. Try set_arg pattern: xrt::run(kernel) + run.set_arg(N, ...)
    run_var, kernel_name = _find_xrt_run_constructor(root, source_bytes)
    if run_var is not None and kernel_name is not None:
        group_ids = _extract_group_ids(root, source_bytes, kernel_name)
        next_id = max(group_ids) + 1 if group_ids else 1

        last_set_arg_stmt, max_idx = _find_last_set_arg(
            root, source_bytes, run_var
        )
        if last_set_arg_stmt is None:
            raise PatchError(
                f"Found xrt::run({kernel_name}) but no "
                f"{run_var}.set_arg() calls."
            )

        return _SetArgPoints(
            last_bo_end=last_bo.end_byte,
            last_set_arg_end=last_set_arg_stmt.end_byte,
            next_arg_index=max_idx + 1,
            wait_end=wait_stmt.end_byte,
            next_group_id=next_id,
            kernel_name=kernel_name,
            run_var=run_var,
        )

    raise PatchError(
        "Cannot find kernel invocation. Expected one of:\n"
        "  - auto run = kernel(opcode, bo_instr, ...);\n"
        "  - auto run = xrt::run(kernel); run.set_arg(N, ...);"
    )


def patch_test_cpp(source: str, trace_size: int = 1048576) -> str:
    """Apply trace buffer injection transforms to a test.cpp source.

    Applies three transforms:
    1. Insert trace buffer allocation after the last xrt::bo declaration
    2. Pass bo_trace to the kernel (direct call or set_arg)
    3. Insert trace data write-out after the last wait()

    Also adds #include <fstream> if not already present.

    Args:
        source: The C++ source code as a string.
        trace_size: Size of the trace buffer in bytes (default: 1MB).

    Returns:
        The transformed source code.

    Raises:
        PatchError: If required insertion points cannot be found.
    """
    # Skip if already patched by us -- the injected BO declaration is the marker.
    # We do NOT skip on the bare name 'trace_size', because some tests define
    # their own trace_size variable for the legacy --trace_sz mechanism.  That
    # legacy path is now voided at the bridge level (--trace_sz stripped from
    # test.exe commands), so those tests need our injection like any other.
    if "injected by trace-prepare.py" in source:
        return source

    # Skip xrt::ext::kernel tests.  The ext API uses a different BO mapping
    # model (positional, no group_id) that is incompatible with our trace BO
    # injection.  Allocating with kernel.group_id(N) creates a bank mismatch
    # when XRT maps the BO by argument position, causing IOMMU page faults.
    if "xrt::ext::kernel" in source:
        return source

    source_bytes = source.encode("utf-8")
    points = _find_insertion_points(source_bytes)

    # Build the trace buffer allocation snippet (same for both modes).
    trace_bo_snippet = _TRACE_BO_TEMPLATE.format(
        trace_size=trace_size,
        trace_size_comment=_human_size(trace_size),
        next_id=points.next_group_id,
        kernel_name=points.kernel_name,
    )

    result = bytearray(source_bytes)

    if isinstance(points, _DirectCallPoints):
        # Apply in reverse byte order: wait_end > close_paren > last_bo_end.

        # 3. Trace write-out after wait
        writeout = _TRACE_WRITEOUT_TEMPLATE.encode("utf-8")
        result[points.wait_end:points.wait_end] = writeout

        # 2. Append ", bo_trace" before closing paren
        arg_insert = b", bo_trace"
        result[points.kernel_call_close_paren:points.kernel_call_close_paren] = arg_insert

        # 1. Trace buffer allocation after last xrt::bo
        result[points.last_bo_end:points.last_bo_end] = trace_bo_snippet.encode("utf-8")

    elif isinstance(points, _SetArgPoints):
        # Apply in reverse byte order: wait_end > last_set_arg_end > last_bo_end.

        # 3. Trace write-out after wait
        writeout = _TRACE_WRITEOUT_TEMPLATE.encode("utf-8")
        result[points.wait_end:points.wait_end] = writeout

        # 2. Insert set_arg for trace buffer after last set_arg
        set_arg_line = (
            f"\n  {points.run_var}.set_arg({points.next_arg_index}, bo_trace);"
        ).encode("utf-8")
        result[points.last_set_arg_end:points.last_set_arg_end] = set_arg_line

        # 1. Trace buffer allocation after last xrt::bo
        result[points.last_bo_end:points.last_bo_end] = trace_bo_snippet.encode("utf-8")

    result_str = result.decode("utf-8")

    # Add #include <fstream> if needed
    result_str = _add_fstream_include(result_str)

    return result_str
