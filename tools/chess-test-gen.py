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
  5. Emit one suite case (kernel .cc) per generated intrinsic via cxx_suite
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Add amd-unified-software site-packages for clang.cindex if needed.
_npu_work = Path(os.environ.get("NPU_WORK_DIR") or Path(__file__).resolve().parents[2])
_clang_site = (_npu_work / "amd-unified-software/tps/lnx64"
               "/python-3.13.0/lib/python3.13/site-packages")
if _clang_site.is_dir() and str(_clang_site) not in sys.path:
    sys.path.append(str(_clang_site))

import clang.cindex

from chess_preprocess import (
    compiled_in_function_names, parse_emulated_accuracy, undefined_accuracy_variant,
)



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
        List of ``(type_name, byte_size, is_ref_output)`` tuples, one per
        parameter.  ``type_name`` is the spelling from clang (with the
        reference stripped for lvalue-ref params); ``byte_size`` is the
        canonical size (max(0, ...) guard applied); ``is_ref_output`` is
        True for non-const lvalue reference parameters (output params).
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
    params: list[tuple[str, int, bool]]
    is_inline: bool
    properties: list[str]
    storage_params: list[str]
    overload_index: int
    source_line: int
    # clang diagnostics (errors) located in the declaration's signature. clang recovers from
    # an unknown type name by substituting int, so without this an ISS-internal operation
    # (v32w32, pmode_t, ...) is indistinguishable from a real int operation.
    signature_errors: list[str] = field(default_factory=list)


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
        # clang stops reporting after 20 errors by default; the header has thousands.
        args=["-std=c++17", "-fsyntax-only", "-ferror-limit=0"],
    )
    errors_by_line: dict[int, list[str]] = {}
    for d in tu.diagnostics:
        if d.severity >= clang.cindex.Diagnostic.Error and d.location.line:
            errors_by_line.setdefault(d.location.line, []).append(d.spelling)

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

            params: list[tuple[str, int, bool]] = []
            for child in cursor.get_children():
                if child.kind == clang.cindex.CursorKind.PARM_DECL:
                    ptype_obj = child.type
                    is_ref_output = False
                    if ptype_obj.kind == clang.cindex.TypeKind.LVALUEREFERENCE:
                        pointee = ptype_obj.get_pointee()
                        if not pointee.is_const_qualified():
                            is_ref_output = True
                        ptype_obj = pointee
                    ptype = ptype_obj.spelling
                    psize = max(0, ptype_obj.get_canonical().get_size())
                    params.append((ptype, psize, is_ref_output))

            sig_first = cursor.extent.start.line
            sig_last = cursor.extent.end.line
            if is_inline:
                # a definition: the signature ends where the body opens
                brace = next((t for t in cursor.get_tokens() if t.spelling == "{"), None)
                if brace is not None:
                    sig_last = brace.location.line
            sig_errors = [msg for n in range(sig_first, sig_last + 1)
                          for msg in errors_by_line.get(n, [])]

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
                signature_errors=sig_errors,
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

# Blocking stream / cascade reads: with no producer feeding the core they never return, and
# every one costs a TDR on silicon (24 in the first sweep). The Peano generator skips the same
# family (INFRA_PATTERNS). A producer-fed version belongs in a two-core pair, not here.
_BLOCKING_READ_PREFIXES = ("get_ss", "get_scd")

# set_xl / set_yhl / set_cll ...: write one part of a wider register and return the whole. The
# other parts are whatever the register held, so the result is not a function of the arguments
# (silicon: they differ with the case that ran before). Not testable as pure intrinsics.
_PARTIAL_REGISTER_SET = re.compile(r"^set_[xybc][lh]{1,2}$")

# Property words from chess_property(...) that flag a function as untestable.
_SKIP_PROPERTY_WORDS = frozenset({
    "dont_care",
    "volatile",
    "non_functional",
    "keep_with_operand",
    "arg_mem_only",
})


@dataclass(frozen=True)
class HeaderFacts:
    """What the other Chess headers say about the declarations in me_chess_opns.h."""

    # function names me_chess_llvm.h declares (what Chess compiles against); None = unknown
    compiled_in: frozenset[str] | None = None
    # the <op>_<mode>_accuracy_<level> family me_vmult_float_emulated.h defines
    accuracy_ops: frozenset[str] = frozenset()
    accuracy_modes: frozenset[str] = frozenset()


def classify_chess_intrinsic(i: ChessIntrinsic,
                             facts: HeaderFacts | None = None) -> tuple[str, str]:
    """Decide whether to generate a test for *i*.

    Returns ``("generated", "")`` or ``("skipped", reason_str)``.

    *facts* carries what the other Chess headers say (see HeaderFacts); None skips those checks.

    Skip criteria (in order):
    1. Any property word in *_SKIP_PROPERTY_WORDS*.
    2. Has chess_storage parameter qualifiers.
    3. void return, or return_size <= 0.
    4. Any parameter with size <= 0.
    5. Name starts with a skip prefix.
    """
    # 0. The declaration does not parse as written (unknown types: ISS-internal operations).
    if i.signature_errors:
        return ("skipped", f"unresolved declaration: {i.signature_errors[0]}")

    if facts is not None:
        if facts.compiled_in is not None and i.name not in facts.compiled_in:
            return ("skipped", "not declared in me_chess_llvm.h (compiled out there, "
                               "e.g. #if 0//not yet): Chess cannot call it")
        if undefined_accuracy_variant(i.name, facts.accuracy_ops, facts.accuracy_modes):
            return ("skipped", "declared //?? UNDEFINED and no vendor header or library "
                               "defines it (me_vmult_float_emulated.h defines only "
                               f"{'/'.join(sorted(facts.accuracy_ops))} for this mode): "
                               "link fails with an undefined symbol")

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
    for ptype, psize, _is_ref in i.params:
        if psize <= 0:
            return ("skipped", f"unsized parameter: {ptype}")

    # 4b. Pointer parameters cannot be meaningfully filled from an input buffer.
    for ptype, _psize, _is_ref in i.params:
        if '*' in ptype:
            return ("skipped", f"pointer parameter: {ptype}")

    # 5. Name prefix skip list.
    for prefix in _SKIP_PREFIXES:
        if i.name.startswith(prefix):
            return ("skipped", f"name prefix: {prefix}")

    if _PARTIAL_REGISTER_SET.match(i.name):
        return ("skipped", "partial register set: the unwritten parts are undefined")

    for prefix in _BLOCKING_READ_PREFIXES:
        if i.name.startswith(prefix):
            return ("skipped", f"blocking stream/cascade read needs a producer: {prefix}")

    # 6. _conf functions require immediate (compile-time constant) operands
    # that cannot be loaded from a buffer at runtime.  The non-_conf variants
    # test the same hardware operations with default config values.
    # Future: hardcode common config values (0, 1) to test these.
    if "_conf" in i.name:
        return ("skipped", "immediate operand required (_conf function)")

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
    param_part = "_".join(ptype for ptype, _, _ in i.params)
    return _sanitize_dirname(f"{i.name}__{param_part}")


def unique_name(taken: dict[str, str], name: str, namespace: str, code: str) -> str | None:
    """Case name for a kernel whose natural name may already be `taken` (name -> kernel text).

    The header declares some overloads twice (once more inside me_primitive); the retired
    layout let the later directory overwrite the earlier one. An identical repeat gets None
    (dropped); a different kernel under the same name is kept, qualified by its namespace.
    """
    if name not in taken:
        return name
    if taken[name] == code:
        return None
    qualified = _sanitize_dirname(f"{namespace or 'global'}__{name}")
    if qualified in taken:
        raise ValueError(f"chess test name {qualified!r} still collides")
    return qualified


def _sanitize_dirname(name: str) -> str:
    """Replace characters that are problematic in filesystem paths."""
    # "unsigned char" -> "unsigned_char", etc.
    return name.replace(" ", "_").replace("::", "_")


# ---------------------------------------------------------------------------
# Immediate operands and recorded Chess verdicts
# ---------------------------------------------------------------------------

OUTCOMES_PATH = Path(__file__).with_name("chess_gen_outcomes.json")

# What Chess did with a signature when it was compiled:
#   no-instruction  no instruction implements this signature, with constants either
#   internal-error  Chess's own instruction table lacks an operation its header declares
OUTCOME_KINDS = ("no-instruction", "internal-error")


def load_outcomes(path: Path) -> dict[str, tuple[str, str]]:
    """Read the recorded Chess verdicts: {case name: (outcome, reason)}."""
    import json
    raw = json.loads(Path(path).read_text())
    out = {}
    for name, entry in raw.items():
        if entry.get("outcome") not in OUTCOME_KINDS:
            raise ValueError(f"{name}: outcome must be one of {OUTCOME_KINDS}")
        if not entry.get("reason", "").strip():
            raise ValueError(f"{name}: a recorded verdict needs a reason")
        out[name] = (entry["outcome"], entry["reason"])
    return out


# ---------------------------------------------------------------------------
# kernel.cc generator (Chess-native)
# ---------------------------------------------------------------------------

def generate_chess_kernel_cc(
    func_name: str,
    namespace: str,
    return_type: str,
    params: list[tuple[str, int, bool]],
) -> str:
    """Generate kernel.cc that calls one Chess-native intrinsic.

    The generated kernel follows the same input/output buffer convention as the
    Peano path (instr-test-gen.py:generate_kernel_cc) so that the same
    suite host program and topology apply:

    - Input buffer: ``const int32_t *restrict in``
    - Output buffer: ``int32_t *restrict out``

    All buffer reads and writes use ``memcpy`` to avoid type-punning issues
    with xchesscc.  Non-const lvalue reference parameters (``is_ref_output``)
    are zero-initialized locals passed by reference; their values are written
    to the output buffer after the return value.
    """
    qualified_name = f"{namespace}::{func_name}" if namespace else func_name

    # Separate input params from reference output params.
    input_params = []   # (arg_name, type, size)
    ref_outputs = []    # (arg_name, base_type, size)
    for idx, (ptype, psize, is_ref) in enumerate(params):
        if is_ref:
            ref_outputs.append((f"arg{idx}_out", ptype, psize))
        else:
            input_params.append((f"arg{idx}", ptype, psize))

    arg_sig = ", ".join(ptype for ptype, _, _ in params)
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
    for idx, (ptype, _psize, is_ref) in enumerate(params):
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
    case: str = ""   # the case name the declaration would have had (overloads share `name`)


def header_facts(opns_path: str, llvm_path: str | None = None) -> HeaderFacts:
    """Read the sibling Chess headers the classifier needs; missing files leave a check off."""
    opns = Path(opns_path)
    llvm = Path(llvm_path) if llvm_path else opns.with_name("me_chess_llvm.h")
    compiled_in = (frozenset(compiled_in_function_names(llvm.read_text()))
                   if llvm.is_file() else None)
    emulated = opns.parent.parent / "me_vmult_float_emulated.h"
    ops, modes = (parse_emulated_accuracy(emulated.read_text())
                  if emulated.is_file() else (set(), set()))
    return HeaderFacts(compiled_in, frozenset(ops), frozenset(modes))


def collect_chess_kernels(
    opns_path: str,
    types_path: str,
    iss_types_path: str | None = None,
    llvm_path: str | None = None,
    outcomes_path: str | Path | None = None,
) -> tuple[list[tuple[GeneratedChessTest, str]], list[SkippedChessIntrinsic]]:
    """Full pipeline: stubs -> preprocess -> walk -> filter -> generate kernel text.

    Parameters
    ----------
    opns_path:
        Path to me_chess_opns.h (the Chess operations header).
    types_path:
        Path to me_chess_types.h (chessTraitsOf specializations).
    iss_types_path:
        Optional path to me_iss_types.h.  When provided, ISS property
        comments are parsed and merged with chessTraitsOf types.  ISS fills
        gaps only -- chessTraitsOf takes precedence via ``setdefault``.

    llvm_path:
        Path to me_chess_llvm.h, the header Chess compiles against (default: beside
        *opns_path*); declarations it does not carry are skipped. The emulation header
        me_vmult_float_emulated.h (one directory up) is read the same way.

    outcomes_path:
        Recorded Chess verdicts (OUTCOMES_PATH by default; "" for none).

    Returns
    -------
    ([(test, kernel.cc text)], skipped).
    """
    # Import here rather than at module top to keep the dependency optional
    # when running unit tests that do not need the full pipeline.
    from chess_type_stubs import parse_all_types, parse_types_header, generate_stub_header
    from chess_preprocess import preprocess_chess_header

    types_text = Path(types_path).read_text()
    iss_text = Path(iss_types_path).read_text() if iss_types_path else None
    traits = parse_all_types(types_text, iss_text)
    _native_traits, aliases = parse_types_header(types_text)
    stubs = generate_stub_header(traits, aliases)

    opns_text = Path(opns_path).read_text()
    clean, annotations = preprocess_chess_header(opns_text)

    combined = stubs + "\n" + clean
    intrinsics = walk_ast(combined, annotations)
    facts = header_facts(opns_path, llvm_path)

    kernels: list[tuple[GeneratedChessTest, str]] = []
    skipped: list[SkippedChessIntrinsic] = []

    # Types that xchesscc knows natively: chessTraitsOf types + C builtins.
    # Functions using types outside this set (ISS-internal types like w32,
    # mmode_t, pmode_t) cannot be compiled by xchesscc and are skipped.
    from chess_type_stubs import BUILTIN_C_TYPES
    chess_native_types = set(_native_traits) | set(aliases) | BUILTIN_C_TYPES

    outcomes = load_outcomes(OUTCOMES_PATH) if outcomes_path is None else (
        load_outcomes(outcomes_path) if outcomes_path else {})

    def skip(i: ChessIntrinsic, reason: str) -> None:
        skipped.append(SkippedChessIntrinsic(
            name=i.name, namespace=i.namespace, reason=reason, case=dir_name(i)))

    for i in intrinsics:
        status, reason = classify_chess_intrinsic(i, facts)
        if status == "skipped":
            skip(i, reason)
            continue

        # Skip functions using ISS-internal types that xchesscc cannot
        # compile (w32, mmode_t, pmode_t, etc.).  These are hardware
        # primitives not callable from user code.  Future: dedicated
        # ISS-level test harness with proper type support.
        all_types = [ptype for ptype, _, _ in i.params] + [i.return_type]
        iss_only = [t for t in all_types
                    if t not in chess_native_types and t != "void"]
        if iss_only:
            skip(i, f"ISS-internal type(s): {', '.join(sorted(set(iss_only)))}")
            continue

        dname = dir_name(i)
        verdict = outcomes.get(dname)
        if verdict:
            skip(i, f"{verdict[0]}: {verdict[1]}")
            continue

        # Buffer sizes: ref outputs are not inputs but are outputs.
        in_size = sum(psize for _, psize, is_ref in i.params if not is_ref)
        out_size = i.return_size + sum(psize for _, psize, is_ref in i.params if is_ref)
        # Minimum 4 bytes for each buffer (at least one i32 slot).
        in_size = max(4, in_size)
        out_size = max(4, out_size)

        # kernel.cc text (Chess-native calling convention).
        kernel_code = generate_chess_kernel_cc(i.name, i.namespace, i.return_type, i.params)

        case_name = unique_name(dict((g.name, code) for g, code in kernels), dname,
                                i.namespace, kernel_code)
        if case_name is None:
            skip(i, "duplicate declaration")
            continue

        kernels.append((GeneratedChessTest(
            name=case_name,
            func_name=i.name,
            namespace=i.namespace,
            in_size=in_size,
            out_size=out_size,
        ), kernel_code))

    return kernels, skipped


def generate_suite(opns_path: str, types_path: str, suite_dir: str,
                   iss_types_path: str | None = None, *, seed: int = 42,
                   outcomes_path: str | Path | None = None) -> tuple[int, int]:
    """Write the Chess instruction-test suite (one case per intrinsic) for the suite runner.

    Returns (generated, skipped) counts. Never calls a compiler or XRT.
    """
    from cxx_suite import CxxKernel, write_cxx_suite
    kernels, skipped = collect_chess_kernels(
        opns_path, types_path, iss_types_path, outcomes_path=outcomes_path)
    write_cxx_suite(
        suite_dir, "instr-chess", "chess-test-gen.py", "chess",
        [CxxKernel(g.name, code, g.in_size, g.out_size,
                   meta={"func_name": g.func_name, "namespace": g.namespace})
         for g, code in kernels],
        seed=seed, skipped=[asdict(x) for x in skipped])
    return len(kernels), len(skipped)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate single-instruction Chess test kernels from me_chess_opns.h"
    )
    parser.add_argument(
        "--opns-header",
        default="../amd-unified-software/aietools/data/aie_ml/lib/isg/me_chess_opns.h",
        help="Path to me_chess_opns.h (default: %(default)s)",
    )
    parser.add_argument(
        "--types-header",
        default="../amd-unified-software/aietools/data/aie_ml/lib/isg/me_chess_types.h",
        help="Path to me_chess_types.h (default: %(default)s)",
    )
    parser.add_argument(
        "--iss-types-header",
        default=None,
        help="Path to me_iss_types.h for additional ISS type coverage (optional)",
    )
    parser.add_argument(
        "--outcomes", default=None,
        help="recorded Chess verdicts (default: tools/chess_gen_outcomes.json)",
    )
    parser.add_argument("--suite-dir", required=True, help="Suite directory to write")
    parser.add_argument("--seed", type=int, default=42, help="input PRNG seed")
    args = parser.parse_args()

    n, n_skipped = generate_suite(args.opns_header, args.types_header, args.suite_dir,
                                  iss_types_path=args.iss_types_header, seed=args.seed,
                                  outcomes_path=args.outcomes)
    print(f"Suite {args.suite_dir}: {n} cases, {n_skipped} intrinsics skipped")


if __name__ == "__main__":
    main()
