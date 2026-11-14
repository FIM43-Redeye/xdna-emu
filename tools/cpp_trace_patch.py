"""Tree-sitter C++ transform module for trace buffer injection.

Parses test.cpp via tree-sitter-cpp and applies three transforms to inject
trace buffer allocation, kernel call argument, and trace data write-out.

Supports three XRT API patterns:

1. **Direct call**: ``auto run = kernel(opcode, bo_instr, ..., bo_out);``
   -> appends ``, bo_trace`` before the closing paren.

2. **set_arg**: ``auto run = xrt::run(kernel); run.set_arg(N, bo_out);``
   -> inserts ``run.set_arg(N+1, bo_trace);`` after the last set_arg.

3. **Multi-kernel**: ``kernel0(opcode, ...); kernel1(opcode, ...);`` or
   several ``xrt::run`` objects -> every call gets a trace buffer of its own
   (``bo_trace``, ``bo_trace_1``, ...), because every call's instruction
   stream may patch the trace argument; only ``bo_trace`` (the first call's)
   is written out.

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
  constexpr size_t _xdna_trace_size = {trace_size};  // {trace_size_comment}
  auto bo_trace = xrt::bo(device, _xdna_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                           {kernel_name}.group_id({next_id}));
  memset(bo_trace.map<void*>(), 0, _xdna_trace_size);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);"""

# A further kernel call's trace buffer. Bound so that call's trace BD writes
# into a buffer of its own instead of naming an unbound argument; it is not
# written out (the capture is the first call's buffer).
_EXTRA_TRACE_BO_TEMPLATE = """\

  auto {bo_name} = xrt::bo(device, _xdna_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                           {kernel_name}.group_id({next_id}));
  memset({bo_name}.map<void*>(), 0, _xdna_trace_size);
  {bo_name}.sync(XCL_BO_SYNC_BO_TO_DEVICE);"""

_TRACE_WRITEOUT_TEMPLATE = """\

  // Write trace data (injected by trace-prepare.py)
  if (const char* trace_dir = std::getenv("XDNA_TRACE_DIR")) {
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      auto trace_ptr = bo_trace.map<char*>();
      std::string trace_path = std::string(trace_dir) + "/trace_raw.bin";
      std::ofstream trace_file(trace_path, std::ios::binary);
      trace_file.write(trace_ptr, _xdna_trace_size);
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


def _kernel_callee(node: Node, source_bytes: bytes) -> Optional[str]:
    """The callee name of a ``kernel*(...)`` call expression, else None."""
    if node.type != "call_expression" or not node.children:
        return None
    callee = node.children[0]
    callee_text = source_bytes[callee.start_byte:callee.end_byte].decode()
    return callee_text if re.match(r'^kernel\w*$', callee_text) else None


def _find_kernel_calls(
    root: Node, source_bytes: bytes
) -> list[tuple[Node, str]]:
    """Find every kernel call expression, in source order.

    Matches calls in declarations that produce a run variable
    (``auto run = kernel(...)``, ``auto run0 = kernel0(...)``) and calls in
    expression statements (``kernel0(...).wait2()``).

    Returns ``[(call_node, kernel_name), ...]`` where *kernel_name* is the
    callee identifier (e.g., ``"kernel"`` or ``"kernel0"``).
    """
    found: dict[int, tuple[Node, str]] = {}
    for node in _walk_all(root):
        if node.type == "declaration":
            text = source_bytes[node.start_byte:node.end_byte].decode()
            if "run" not in text:
                continue
        elif node.type != "expression_statement":
            continue
        for desc in _walk_all(node):
            name = _kernel_callee(desc, source_bytes)
            if name is not None:
                found.setdefault(desc.start_byte, (desc, name))
    return [found[k] for k in sorted(found)]


def _find_xrt_run_constructors(
    root: Node, source_bytes: bytes
) -> list[tuple[str, str]]:
    """Find every ``auto run = xrt::run(kernel)``, in source order.

    Returns ``[(run_var_name, kernel_name), ...]``; e.g. for
    ``auto run1 = xrt::run(kernel0)`` the entry is ``("run1", "kernel0")``.
    """
    runs = []
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
            runs.append((m.group(1), m.group(2)))
    return runs


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
    # An if_statement counts too: tests commonly guard the wait itself,
    # `if (run.wait() != ERT_CMD_STATE_COMPLETED) {...}`, so the call lives in
    # the condition and no declaration or expression statement contains it.
    for candidate in _walk_all(root):
        if candidate.type in ("declaration", "expression_statement", "if_statement"):
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
class _Binding:
    """One kernel call that gets a trace buffer: insert the buffer at
    *insert_at*, as the last call argument (*run_var* None) or as a
    ``run_var.set_arg`` statement."""
    insert_at: int
    kernel_name: str
    run_var: Optional[str] = None


@dataclass
class _InsertionPoints:
    last_bo_end: int
    wait_end: int
    trace_arg_index: int
    bindings: list[_Binding]


def _find_insertion_points(
    source_bytes: bytes,
    trace_arg_index: int,
) -> _InsertionPoints:
    """Parse the source and locate injection points.

    Tries the direct-call pattern first, falls back to the set_arg pattern;
    either way every call found is bound. Raises PatchError if neither
    pattern is found, or if a call that binds a trace buffer comes before
    the point where the buffers are declared.

    Args:
        trace_arg_index: Kernel argument slot for the trace BO. Comes from
            ``trace_config.json``'s ``buffer.kernel_arg_slot``. Used as
            both the ``kernel.group_id(N)`` argument and the set_arg index.
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

    bindings: list[_Binding] = []

    # 3. Try direct-call pattern first: kernel(opcode, ...)
    for kernel_call, kernel_name in _find_kernel_calls(root, source_bytes):
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
        bindings.append(_Binding(close_paren.start_byte, kernel_name))

    # 4. Else the set_arg pattern: xrt::run(kernel) + run.set_arg(N, ...)
    if not bindings:
        for run_var, kernel_name in _find_xrt_run_constructors(root, source_bytes):
            last_set_arg_stmt, _max_idx = _find_last_set_arg(
                root, source_bytes, run_var
            )
            if last_set_arg_stmt is None:
                raise PatchError(
                    f"Found xrt::run({kernel_name}) but no "
                    f"{run_var}.set_arg() calls."
                )
            bindings.append(
                _Binding(last_set_arg_stmt.end_byte, kernel_name, run_var)
            )

    if not bindings:
        raise PatchError(
            "Cannot find kernel invocation. Expected one of:\n"
            "  - auto run = kernel(opcode, bo_instr, ...);\n"
            "  - auto run = xrt::run(kernel); run.set_arg(N, ...);"
        )

    for b in bindings:
        if b.insert_at < last_bo.end_byte:
            raise PatchError(
                f"A {b.kernel_name} call binds a trace buffer before the last "
                "xrt::bo declaration, where the trace buffers are declared."
            )

    return _InsertionPoints(
        last_bo_end=last_bo.end_byte,
        wait_end=wait_stmt.end_byte,
        trace_arg_index=trace_arg_index,
        bindings=bindings,
    )


def _trace_bo_name(call_index: int) -> str:
    """``bo_trace`` for the first call (the one written out), then
    ``bo_trace_1``, ``bo_trace_2``, ..."""
    return "bo_trace" if call_index == 0 else f"bo_trace_{call_index}"


def patch_test_cpp(
    source: str,
    trace_size: int,
    trace_arg_index: int,
) -> str:
    """Apply trace buffer injection transforms to a test.cpp source.

    Applies three transforms:
    1. Insert trace buffer allocation(s) after the last xrt::bo declaration
    2. Pass a trace buffer to every kernel call (direct call or set_arg)
    3. Insert the first call's trace data write-out after the last wait()

    Also adds #include <fstream> if not already present.

    Args:
        source: The C++ source code as a string.
        trace_size: Size of the trace buffer in bytes
            (from ``trace_config.json``'s ``buffer.size_bytes``).
        trace_arg_index: Kernel argument slot for the trace BO
            (from ``trace_config.json``'s ``buffer.kernel_arg_slot``).
            Used as both the ``kernel.group_id(N)`` argument and the
            ``run.set_arg(N, ...)`` index, for every call.

    Returns:
        The transformed source code.

    Raises:
        PatchError: If required insertion points cannot be found.
    """
    # Skip if already patched by us -- the injected BO declaration is the marker.
    # The injected symbol is `_xdna_trace_size` (underscore-prefixed) so it
    # never collides with a test's own `trace_size` from the legacy --trace_sz
    # path (which is voided at the bridge level anyway).
    if "injected by trace-prepare.py" in source:
        return source

    source_bytes = source.encode("utf-8")
    points = _find_insertion_points(source_bytes, trace_arg_index)

    # Trace buffer allocations, all at one point: the first call's buffer
    # (with the size constant), then one per further call.
    allocations = _TRACE_BO_TEMPLATE.format(
        trace_size=trace_size,
        trace_size_comment=_human_size(trace_size),
        next_id=points.trace_arg_index,
        kernel_name=points.bindings[0].kernel_name,
    )
    for i, b in enumerate(points.bindings[1:], start=1):
        allocations += _EXTRA_TRACE_BO_TEMPLATE.format(
            bo_name=_trace_bo_name(i),
            kernel_name=b.kernel_name,
            next_id=points.trace_arg_index,
        )

    edits: list[tuple[int, str]] = [
        (points.last_bo_end, allocations),
        (points.wait_end, _TRACE_WRITEOUT_TEMPLATE),
    ]
    for i, b in enumerate(points.bindings):
        name = _trace_bo_name(i)
        if b.run_var is None:
            # Append ", bo_trace" before the call's closing paren.
            edits.append((b.insert_at, f", {name}"))
        else:
            # A set_arg for the trace buffer after the run's last set_arg.
            # If a padding set_arg already exists at this index, the later
            # call overrides it (XRT uses the last value for each index).
            edits.append((
                b.insert_at,
                f"\n  {b.run_var}.set_arg({points.trace_arg_index}, {name});",
            ))

    # Apply in reverse byte order so earlier offsets stay valid.
    result = bytearray(source_bytes)
    for offset, text in sorted(edits, key=lambda e: e[0], reverse=True):
        result[offset:offset] = text.encode("utf-8")

    result_str = result.decode("utf-8")

    # Add #include <fstream> if needed
    result_str = _add_fstream_include(result_str)

    return result_str
