#!/usr/bin/env python3
"""Generate single-instruction test kernels from Chess intrinsic definitions.

This is the Chess-native counterpart to instr-test-gen.py (the Peano path).
The full pipeline:

  1. Parse me_chess_types.h  ->  sized struct stubs (chess_type_stubs.py)
  2. Pre-process me_chess_opns.h -> clean C++ + annotation dict
     (chess_preprocess.py)
  3. Combine stubs + clean source, walk AST with clang.cindex -> ChessIntrinsic
     records
  4. Classify each intrinsic (skip/generate)
  5. Emit kernel.cc + aie.mlir per generated intrinsic

The aie.mlir and test_host.cpp templates are shared with the Peano path --
they are imported from instr-test-gen.py via importlib (the module name
contains a hyphen, so a normal import statement cannot be used).
"""

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Add amd-unified-software site-packages for clang.cindex if needed.
_clang_site = Path("/home/triple/npu-work/amd-unified-software/tps/lnx64"
                   "/python-3.13.0/lib/python3.13/site-packages")
if _clang_site.is_dir() and str(_clang_site) not in sys.path:
    sys.path.append(str(_clang_site))

import clang.cindex

# Load the Peano generator to reuse generate_aie_mlir and generate_test_host_cpp.
# The module name has a hyphen so we must use importlib rather than 'import'.
_peano = importlib.import_module("instr-test-gen")
generate_aie_mlir = _peano.generate_aie_mlir
generate_test_host_cpp = _peano.generate_test_host_cpp


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChessIntrinsic:
    """One function declaration extracted from the Chess operations header.

    Attributes
    ----------
    name:
        The bare function name (e.g. ``broadcast_to_v16int32``).
    namespace:
        Enclosing C++ namespace, or empty string for global scope.
        The only non-empty value seen in me_chess_opns.h is ``"me_primitive"``.
    return_type:
        The spelling of the return type as seen by clang (e.g. ``"v16int32"``).
    return_size:
        Byte size of the return type per the canonical AST type, clamped to
        ``max(0, ...)`` because clang returns -1 for unknown/incomplete types.
    params:
        List of ``(type_name, byte_size)`` tuples, one per parameter.
        ``type_name`` is the spelling from clang; ``byte_size`` is the
        canonical size (max(0, ...) guard applied).
    is_inline:
        True when the cursor has a definition body (inline function), False for
        declaration-only.
    properties:
        Property words gathered from ``chess_property(...)`` annotations for
        this function name, merged across all overloads (see chess_preprocess).
    storage_params:
        Storage class names from ``chess_storage(...)`` parameter qualifiers,
        merged across all overloads.
    overload_index:
        Zero-based index distinguishing overloads that share the same name.
        The first encountered overload gets index 0; each subsequent one
        increments.
    source_line:
        Line number in the (pre-processed) source for traceability.
    """

    name: str
    namespace: str
    return_type: str
    return_size: int
    params: list[tuple[str, int]]
    is_inline: bool
    properties: list[str]
    storage_params: list[str]
    overload_index: int
    source_line: int


# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------

def walk_ast(
    source: str,
    annotations: dict,
) -> list[ChessIntrinsic]:
    """Parse *source* with clang.cindex and extract ChessIntrinsic records.

    Parameters
    ----------
    source:
        Pre-processed C++ source suitable for clang (structs already defined,
        Chess extensions stripped).
    annotations:
        Dict mapping function-name -> ChessAnnotation, as produced by
        ``chess_preprocess.preprocess_chess_header``.  Keyed by bare function
        name so overloads share the same annotation entry.

    Returns
    -------
    List of ChessIntrinsic, one per function declaration found in the AST,
    in source order.  Overloads are distinguished by ``overload_index``.
    """
    index = clang.cindex.Index.create()
    tu = index.parse(
        "chess_opns.cpp",
        unsaved_files=[("chess_opns.cpp", source)],
        args=["-std=c++17", "-fsyntax-only"],
    )

    results: list[ChessIntrinsic] = []
    # Track how many overloads we have seen per (namespace, name) pair.
    overload_counter: dict[tuple[str, str], int] = {}

    def _visit(cursor: clang.cindex.Cursor, namespace: str) -> None:
        """Recursively walk the AST, collecting function declarations."""
        if cursor.kind == clang.cindex.CursorKind.NAMESPACE:
            ns_name = cursor.spelling
            for child in cursor.get_children():
                _visit(child, ns_name)
            return

        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            name = cursor.spelling
            ret_type = cursor.result_type.spelling
            ret_size = max(0, cursor.result_type.get_canonical().get_size())
            is_inline = cursor.is_definition()
            line = cursor.location.line or 0

            params: list[tuple[str, int]] = []
            for child in cursor.get_children():
                if child.kind == clang.cindex.CursorKind.PARM_DECL:
                    ptype = child.type.spelling
                    psize = max(0, child.type.get_canonical().get_size())
                    params.append((ptype, psize))

            ann = annotations.get(name)
            props = list(ann.properties) if ann else []
            storage = list(ann.storage_params) if ann else []

            key = (namespace, name)
            ov_idx = overload_counter.get(key, 0)
            overload_counter[key] = ov_idx + 1

            results.append(ChessIntrinsic(
                name=name,
                namespace=namespace,
                return_type=ret_type,
                return_size=ret_size,
                params=params,
                is_inline=is_inline,
                properties=props,
                storage_params=storage,
                overload_index=ov_idx,
                source_line=line,
            ))
            return  # do not descend into function body

        # For any other cursor kind at the top level, recurse into children.
        # This handles unnamed namespaces and similar constructs.
        for child in cursor.get_children():
            _visit(child, namespace)

    for child in tu.cursor.get_children():
        _visit(child, "")

    return results


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

# Function name prefixes that indicate diagnostic / helper routines that
# should never be tested as data-path intrinsics.
_SKIP_PREFIXES = (
    "operator",
    "chess_report",
    "chess_assert",
    "chess_error",
    "chess_warning",
    "chess_exit",
    "chess_stop",
    "chess_message",
    "chess_cycle_count",
    "chess_return_address",
    "chess_dont_care",
    "keep_in_registers_wrapper",
    "__promo_",     # internal compiler promotion primitives
    "__promy_",     # internal compiler promotion primitives
)

# Property words from chess_property(...) that flag a function as untestable.
_SKIP_PROPERTY_WORDS = frozenset({
    "dont_care",
    "volatile",
    "non_functional",
    "keep_with_operand",
    "arg_mem_only",
})


def classify_chess_intrinsic(i: ChessIntrinsic) -> tuple[str, str]:
    """Decide whether to generate a test for *i*.

    Returns ``("generated", "")`` or ``("skipped", reason_str)``.

    Skip criteria (in order):
    1. Any property word in *_SKIP_PROPERTY_WORDS*.
    2. Has chess_storage parameter qualifiers.
    3. void return, or return_size <= 0.
    4. Any parameter with size <= 0.
    5. Name starts with a skip prefix.
    """
    # 1. Property-based skip.
    for word in i.properties:
        if word in _SKIP_PROPERTY_WORDS:
            return ("skipped", f"property: {word}")

    # 2. Storage-class parameter qualifiers prevent clean calling.
    if i.storage_params:
        return ("skipped", "has chess_storage parameter(s)")

    # 3. Void or unknowable return type.
    if i.return_type == "void" or i.return_size <= 0:
        return ("skipped", f"void/unknown return type ({i.return_type}, size={i.return_size})")

    # 4. Any parameter whose size clang could not determine.
    for ptype, psize in i.params:
        if psize <= 0:
            return ("skipped", f"unsized parameter: {ptype}")

    # 5. Name prefix skip list.
    for prefix in _SKIP_PREFIXES:
        if i.name.startswith(prefix):
            return ("skipped", f"name prefix: {prefix}")

    return ("generated", "")


# ---------------------------------------------------------------------------
# Directory name
# ---------------------------------------------------------------------------

def dir_name(i: ChessIntrinsic) -> str:
    """Return a stable filesystem directory name for the test.

    Pattern: ``{func_name}__{p1type_p2type_...}``
    For zero-argument functions: just ``{func_name}``.

    The double-underscore separator makes it easy to split the name
    back into function and parameter parts without ambiguity.
    """
    if not i.params:
        return _sanitize_dirname(i.name)
    param_part = "_".join(ptype for ptype, _ in i.params)
    return _sanitize_dirname(f"{i.name}__{param_part}")


def _sanitize_dirname(name: str) -> str:
    """Replace characters that are problematic in filesystem paths."""
    # "unsigned char" -> "unsigned_char", etc.
    return name.replace(" ", "_")


# ---------------------------------------------------------------------------
# kernel.cc generator (Chess-native)
# ---------------------------------------------------------------------------

def generate_chess_kernel_cc(
    func_name: str,
    namespace: str,
    return_type: str,
    params: list[tuple[str, int]],
) -> str:
    """Generate kernel.cc that calls one Chess-native intrinsic.

    The generated kernel follows the same input/output buffer convention as the
    Peano path (instr-test-gen.py:generate_kernel_cc) so that the same
    test_host.cpp and aie.mlir can be reused:

    - Input buffer: ``const int32_t *restrict in``
    - Output buffer: ``int32_t *restrict out``

    Arguments are read from consecutive regions of the input buffer.
    The return value is cast-written to the output buffer.

    Sub-4-byte scalar arguments use memcpy to avoid implicit narrowing
    conversions.  This check is done BEFORE building the lines list to
    keep the logic clean (not insert-in-loop).
    """
    # Call functions unqualified -- xchesscc knows Chess intrinsics natively
    # regardless of their namespace in me_chess_opns.h.
    qualified_name = func_name

    # Compute sizes and check sub-4-byte scalar flag BEFORE building lines.
    has_sub4_scalar = any(
        psize < 4 and not _is_vector_type(ptype)
        for ptype, psize in params
    )

    arg_sig = ", ".join(ptype for ptype, _ in params)
    sig_comment = (
        f"{return_type} = {qualified_name}({arg_sig})"
        if params
        else f"{return_type} = {qualified_name}()"
    )

    lines = [
        f"// Auto-generated: tests {qualified_name}",
        f"// Signature: {sig_comment}",
        "#define NOCPP",
        "#include <stdint.h>",
    ]
    if has_sub4_scalar:
        lines.append("#include <string.h>")
    lines += [
        "",
        'extern "C" {',
        "void test_kernel(const int32_t *restrict in, int32_t *restrict out) {",
    ]

    # Generate argument reads from input buffer.
    offset_i32 = 0    # current offset in int32_t units
    byte_offset = 0   # current byte offset (for sub-4-byte reads)
    arg_names: list[str] = []

    for idx, (ptype, psize) in enumerate(params):
        arg_name = f"arg{idx}"
        arg_names.append(arg_name)

        is_vector = _is_vector_type(ptype)
        if is_vector or psize > 4:
            # Cast the int32_t pointer to the vector/large type.
            lines.append(
                f"    {ptype} {arg_name} = "
                f"*(const {ptype} *)(in + {offset_i32});"
            )
        elif psize < 4:
            # Sub-4-byte scalar: memcpy to avoid implicit conversion/narrowing.
            lines.append(f"    {ptype} {arg_name};")
            lines.append(
                f"    memcpy(&{arg_name}, "
                f"(const char *)in + {byte_offset}, "
                f"sizeof({ptype}));"
            )
        else:
            # 4-byte scalar (int, unsigned, float, etc.).
            lines.append(f"    {ptype} {arg_name} = in[{offset_i32}];")

        # Advance offsets: always at least one i32 slot per argument.
        align_i32 = max(1, psize // 4)
        offset_i32 += align_i32
        byte_offset += max(4, psize)

    call_args = ", ".join(arg_names)
    lines.append(f"    {return_type} result = {qualified_name}({call_args});")
    lines.append(f"    {return_type} *out_vec = ({return_type} *)out;")
    lines.append("    *out_vec = result;")
    lines.append("}")
    lines.append('} // extern "C"')
    lines.append("")  # trailing newline

    return "\n".join(lines)


def _is_vector_type(type_name: str) -> bool:
    """Heuristic: a type is vector/struct if it is not a primitive C scalar.

    Chess types such as v16int32, v64uint8, pmode_t, etc. are all struct stubs
    with sizeof > 4 typically, but pmode_t has sizeof==4 and is still a struct.
    We use the naming convention: if the name starts with 'v' and contains a
    digit, or is otherwise not a known C scalar, treat it as a struct.

    In practice, for Chess intrinsics the only scalar types encountered are
    ``int``, ``unsigned``, ``unsigned int``, ``short``, ``char``, ``float``,
    and their signed/const variants.  Everything else is a struct.
    """
    base = type_name.replace("const ", "").replace("restrict ", "").strip()
    _SCALAR_BASES = frozenset({
        "int", "unsigned", "unsigned int", "short", "unsigned short",
        "char", "signed char", "unsigned char", "long", "unsigned long",
        "long long", "unsigned long long", "float", "double",
        "bool", "int8_t", "uint8_t", "int16_t", "uint16_t",
        "int32_t", "uint32_t", "int64_t", "uint64_t",
        "bfloat16",  # 2-byte scalar on AIE
    })
    return base not in _SCALAR_BASES


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

@dataclass
class GeneratedChessTest:
    """One generated Chess test case."""
    name: str           # dir_name
    func_name: str      # bare function name
    namespace: str
    in_size: int
    out_size: int


@dataclass
class SkippedChessIntrinsic:
    """One skipped Chess intrinsic."""
    name: str
    namespace: str
    reason: str


def generate_all(
    opns_path: str,
    types_path: str,
    out_dir: str,
    iss_types_path: str | None = None,
) -> tuple[list[GeneratedChessTest], list[SkippedChessIntrinsic]]:
    """Full pipeline: stubs -> preprocess -> walk -> filter -> generate files.

    Parameters
    ----------
    opns_path:
        Path to me_chess_opns.h (the Chess operations header).
    types_path:
        Path to me_chess_types.h (chessTraitsOf specializations).
    out_dir:
        Root output directory.  One subdirectory per generated test.
    iss_types_path:
        Optional path to me_iss_types.h.  When provided, ISS property
        comments are parsed and merged with chessTraitsOf types.  ISS fills
        gaps only -- chessTraitsOf takes precedence via ``setdefault``.

    Returns
    -------
    (generated, skipped) lists suitable for manifest construction.
    """
    # Import here rather than at module top to keep the dependency optional
    # when running unit tests that do not need the full pipeline.
    from chess_type_stubs import (
        parse_chess_traits, parse_iss_property_comments, generate_stub_header,
    )
    from chess_preprocess import preprocess_chess_header

    types_text = Path(types_path).read_text()
    traits = parse_chess_traits(types_text)
    if iss_types_path:
        iss_text = Path(iss_types_path).read_text()
        iss_types = parse_iss_property_comments(iss_text)
        for name, bits in iss_types.items():
            traits.setdefault(name, bits)
    stubs = generate_stub_header(traits)

    opns_text = Path(opns_path).read_text()
    clean, annotations = preprocess_chess_header(opns_text)

    combined = stubs + "\n" + clean
    intrinsics = walk_ast(combined, annotations)

    generated: list[GeneratedChessTest] = []
    skipped: list[SkippedChessIntrinsic] = []

    os.makedirs(out_dir, exist_ok=True)

    for i in intrinsics:
        status, reason = classify_chess_intrinsic(i)
        if status == "skipped":
            skipped.append(SkippedChessIntrinsic(
                name=i.name, namespace=i.namespace, reason=reason,
            ))
            continue

        dname = dir_name(i)
        test_dir = os.path.join(out_dir, dname)
        os.makedirs(test_dir, exist_ok=True)

        # Compute buffer sizes from parameter and return type sizes.
        in_size = sum(psize for _, psize in i.params)
        out_size = i.return_size
        # Minimum 4 bytes for each buffer (at least one i32 slot).
        in_size = max(4, in_size)
        out_size = max(4, out_size)

        # Write kernel.cc (Chess-native calling convention).
        kernel_code = generate_chess_kernel_cc(
            i.name, i.namespace, i.return_type, i.params,
        )
        Path(os.path.join(test_dir, "kernel.cc")).write_text(kernel_code)

        # Write aie.mlir (shared template from Peano path).
        mlir_code = generate_aie_mlir(in_size, out_size)
        Path(os.path.join(test_dir, "aie.mlir")).write_text(mlir_code)

        generated.append(GeneratedChessTest(
            name=dname,
            func_name=i.name,
            namespace=i.namespace,
            in_size=in_size,
            out_size=out_size,
        ))

    # Write shared test_host.cpp (identical to Peano path).
    host_code = generate_test_host_cpp()
    Path(os.path.join(out_dir, "test_host.cpp")).write_text(host_code)

    # Write manifest.json.
    manifest = {
        "generated": [asdict(g) for g in generated],
        "skipped": [asdict(s) for s in skipped],
    }
    Path(os.path.join(out_dir, "manifest.json")).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    return generated, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate single-instruction Chess test kernels from me_chess_opns.h"
    )
    parser.add_argument(
        "--opns-header",
        default="../aietools/data/aie_ml/lib/isg/me_chess_opns.h",
        help="Path to me_chess_opns.h (default: %(default)s)",
    )
    parser.add_argument(
        "--types-header",
        default="../aietools/data/aie_ml/lib/isg/me_chess_types.h",
        help="Path to me_chess_types.h (default: %(default)s)",
    )
    parser.add_argument(
        "--iss-types-header",
        default=None,
        help="Path to me_iss_types.h for additional ISS type coverage (optional)",
    )
    parser.add_argument(
        "--out-dir",
        default="build/chess-instr-tests",
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    generated, skipped = generate_all(
        args.opns_header, args.types_header, args.out_dir,
        iss_types_path=args.iss_types_header,
    )

    print(f"Generated: {len(generated)} tests")
    print(f"Skipped:   {len(skipped)} intrinsics")


if __name__ == "__main__":
    main()
