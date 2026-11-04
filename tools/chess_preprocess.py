#!/usr/bin/env python3
"""Pre-process me_chess_opns.h to strip Chess-specific extensions.

Uses a two-pass approach:
  Pass 1: Extract chess_property and chess_storage annotations from the
           ORIGINAL text, keyed by function name (not line number).
  Pass 2: Strip all Chess extensions to produce clean C++ for clang.cindex.

Design note -- annotation keys are function names, not line numbers:
  The pre-processed text will have different line numbers than the original
  because entire blocks are removed.  Keying by function name lets the AST
  walker (which sees the pre-processed text) still look up annotations for
  any declaration it finds.  Overloaded functions accumulate all their
  annotations under the same key; that is intentional -- callers that care
  about per-overload detail can refine later.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ChessAnnotation:
    """Annotations extracted from a function declaration."""

    func_name: str
    # Words from chess_property(...) -- may be multiple space-separated tokens.
    properties: list[str] = field(default_factory=list)
    # Register/memory class names from chess_storage(...) parameter qualifiers.
    storage_params: list[str] = field(default_factory=list)


def _extract_func_name(text_before: str) -> str:
    """Extract the function name from text preceding a Chess annotation.

    Handles two cases:

    1. Annotation is AFTER the closing paren of the parameter list (the
       common ``func(...) chess_property(...)`` pattern).  The text ends
       with ``name(args)`` (possibly followed by whitespace) and we grab
       the last such name.

    2. Annotation is INSIDE the parameter list (the
       ``func(..., type chess_storage(REG) *param, ...)`` pattern).  The
       text ends without a closing paren for the outer call, so we look
       for the last ``name(`` pattern that precedes the current position.

    Returns "_unknown" when no function name can be identified (e.g. on
    a global variable declaration).
    """
    stripped = text_before.rstrip()

    # Case 1: annotation follows ``func(args)`` -- text ends with ')'
    m = re.search(r'(\w+)\s*\([^)]*\)\s*$', stripped)
    if m:
        return m.group(1)

    # Case 2: annotation is inside the parameter list -- text ends inside
    # an open '('.  Find the last identifier followed by '(' that is not
    # itself a Chess keyword or a type cast.
    #
    # We scan right-to-left for ``word(`` patterns.  The first one we find
    # (from the right) that is not a Chess keyword is the enclosing function.
    _CHESS_KEYWORDS = frozenset({
        'chess_storage', 'chess_property', 'chess_manifest',
        'chess_memory_fence', 'chess_separator_scheduler', 'chess_separator',
        'chess_dont_warn_dead', 'chess_protect_access',
    })
    for m2 in reversed(list(re.finditer(r'(\w+)\s*\(', stripped))):
        name = m2.group(1)
        if name not in _CHESS_KEYWORDS:
            return name

    return "_unknown"


def preprocess_chess_header(text: str) -> tuple[str, dict[str, ChessAnnotation]]:
    """Strip Chess extensions from header source.

    Returns ``(clean_cpp, annotations)`` where:
    - ``clean_cpp`` is valid C++ that clang.cindex can parse.
    - ``annotations`` maps function name -> ChessAnnotation.  Overloaded
      functions accumulate annotations under the same key.

    The two passes operate on the ORIGINAL text (pass 1) and then produce
    the stripped version (pass 2).  This preserves annotation accuracy
    regardless of how much text is deleted in pass 2.
    """
    annotations: dict[str, ChessAnnotation] = {}

    # ------------------------------------------------------------------
    # Pass 1: Extract annotations from ORIGINAL text.
    # We must run this before any stripping so that the text before each
    # annotation still contains the function name.
    # ------------------------------------------------------------------

    # chess_property(word word ...) -- one or more space-separated tokens.
    for m in re.finditer(r'\s*chess_property\(([^)]+)\)', text):
        prop_words = m.group(1).strip().split()
        func_name = _extract_func_name(text[:m.start()])
        ann = annotations.setdefault(
            func_name, ChessAnnotation(func_name=func_name),
        )
        ann.properties.extend(prop_words)

    # chess_storage(REGISTER_CLASS) -- a single identifier naming the
    # physical storage class (TM, DM_bankA, SCD, …).
    for m in re.finditer(r'chess_storage\(([^)]+)\)', text):
        storage_name = m.group(1).strip()
        func_name = _extract_func_name(text[:m.start()])
        if func_name != "_unknown":
            ann = annotations.setdefault(
                func_name, ChessAnnotation(func_name=func_name),
            )
            ann.storage_params.append(storage_name)

    # ------------------------------------------------------------------
    # Pass 2: Strip all Chess extensions to produce clean C++.
    # Order matters: block-level removals first, then inline ones.
    # ------------------------------------------------------------------

    # Remove ``#if 0//! ... #endif//!`` blocks (Chess-disabled code).
    # These blocks use the exact markers ``#if 0//!`` and ``#endif//!``
    # on their own lines.
    text = re.sub(r'#if\s+0\s*//!.*?#endif\s*//!', '', text, flags=re.DOTALL)

    # Remove lines beginning with ``//!`` (Chess-only declaration comments).
    text = re.sub(r'^[ \t]*//!.*$', '', text, flags=re.MULTILINE)

    # Remove ``#ifdef __chess__ / #error ... / #endif`` guard at file top.
    text = re.sub(
        r'#ifdef\s+__chess__\s*\n#error\s+[^\n]*\n#endif\s*\n?', '', text,
    )

    # Remove function-level Chess annotations (handled in pass 1 above).
    text = re.sub(r'\s*chess_property\([^)]+\)', '', text)
    # chess_storage(...) appears inline in parameter types; replace with a
    # single space so the surrounding tokens remain valid C++.
    text = re.sub(r'\s*chess_storage\([^)]+\)\s*', ' ', text)

    # Remove Chess keyword qualifiers that are not valid C++.
    text = re.sub(r'\bchess_protect_access\b', '', text)

    # chess_output is a parameter qualifier (marks output-only parameters);
    # it has no C++ equivalent -- strip it so the parameter list is clean.
    text = re.sub(r'\bchess_output\b', '', text)

    # chess_error("msg") is a Chess compile-time assertion that causes a
    # compile error with the given message.  Replace with a no-op expression
    # so the surrounding if-statement bodies still compile.
    text = re.sub(r'\bchess_error\([^)]*\)', '((void)0)', text)

    # chess_dont_care(TypeName) returns an uninitialized value of TypeName.
    # Replace with TypeName{} (value-initialization), which is the closest
    # valid C++ equivalent.  The argument is always a single identifier.
    text = re.sub(r'\bchess_dont_care\((\w+)\)', r'\1{}', text)

    # chess_manifest(expr) is a Chess compile-time assertion; replace with
    # a no-op expression that is always true so surrounding if-statements
    # still compile.
    text = re.sub(r'\bchess_manifest\([^)]*\)', '(1)', text)

    # Scheduling / pipeline hints -- replace with cast-to-void no-ops so
    # the statement structure (semicolons, etc.) is preserved.
    text = re.sub(r'\bchess_memory_fence\(\)', '((void)0)', text)
    # chess_separator_scheduler may carry an optional numeric argument.
    text = re.sub(r'\bchess_separator_scheduler\([^)]*\)', '((void)0)', text)
    # chess_separator() is the no-argument variant of the above.
    text = re.sub(r'\bchess_separator\(\)', '((void)0)', text)
    text = re.sub(r'\bchess_dont_warn_dead\([^)]*\)', '((void)0)', text)

    # chess_unroll_loop(*) is a loop pragma; remove it entirely (it trails
    # the closing parenthesis of the for-statement header).
    text = re.sub(r'\bchess_unroll_loop\(\*\)', '', text)

    # VBITzCONSTEXPR is a Chess macro that expands to something like
    # ``inline constexpr``; replace with plain ``inline`` so the
    # declaration is still valid.
    text = re.sub(r'\bVBITzCONSTEXPR\b', 'inline', text)

    return text, annotations


_CONDITIONAL = re.compile(r'#\s*(if|ifdef|ifndef|else|elif|endif)\b(.*)')


def compiled_in_text(text: str) -> str:
    """The part of a Chess C++ header that a C++ compile actually sees.

    Drops `#if 0 ... #endif` regions (the generator parks not-yet-supported operations
    there: `#if 0//not yet`) and `#ifndef __cplusplus` regions. Every other conditional is
    kept on both sides: over-approximating the visible text only ever keeps a declaration.
    """
    out: list[str] = []
    # frame: [branch is live, the opening condition is statically false]
    stack: list[list[bool]] = []
    for line in text.split('\n'):
        m = _CONDITIONAL.match(line.strip())
        if m:
            directive = m.group(1)
            arg = re.sub(r'/\*.*?\*/|//.*', '', m.group(2)).strip()
            if directive in ('if', 'ifdef', 'ifndef'):
                static_false = ((directive == 'if' and arg == '0')
                                or (directive == 'ifndef' and arg == '__cplusplus'))
                stack.append([not static_false, static_false])
            elif directive == 'else' and stack:
                # only the else of a statically-false condition becomes live, and an
                # unknown condition stays live on both sides
                stack[-1][0] = True
            elif directive == 'elif' and stack:
                stack[-1][0] = True
            elif directive == 'endif' and stack:
                stack.pop()
            continue
        if all(frame[0] for frame in stack):
            out.append(line)
    return '\n'.join(out)


def compiled_in_function_names(text: str) -> set[str]:
    """Every identifier followed by `(` in the compiled-in part of the header."""
    return set(re.findall(r'\b(\w+)\s*\(', compiled_in_text(text)))


def parse_emulated_accuracy(text: str) -> tuple[set[str], set[str]]:
    """What me_vmult_float_emulated.h defines for the `<op>_<mode>_accuracy_<level>` family.

    The header builds the family from a macro, `_MUL_MAC_TYPE(RET, X, Y, MODE, level)`: its
    body names the operations it defines (`mul_##MODE##_accuracy_##TYPE`), and each use
    names a mode. Returns ({operations}, {modes}). me_chess_opns.h declares more operations
    per mode than the macro defines (negmul, addmac, addmsc), marked `//?? UNDEFINED`.
    """
    macro = re.search(r'#define _MUL_MAC_TYPE\(.*?\)(.*?)(?=\n#(?:if|undef|define|endif)|\n_INSTANTIATE)',
                      text, re.DOTALL)
    ops = set(re.findall(r'\binline\s+\w+\s+(\w+?)_##MODE##_accuracy_##TYPE\(',
                         macro.group(1))) if macro else set()
    modes = set(re.findall(r'_MUL_MAC_TYPE\(\s*\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*(\w+)\s*,\s*ACCURACY\s*\)',
                           text))
    return ops, modes


def undefined_accuracy_variant(name: str, ops: set[str], modes: set[str]) -> bool:
    """True when *name* is `<op>_<mode>_accuracy_<level>` for an emulated mode but an
    operation the emulation header does not define."""
    m = re.fullmatch(r'(\w+?)_(\w+)_accuracy_(?:low|fast|safe)', name)
    return bool(m) and m.group(2) in modes and m.group(1) not in ops


# ---------------------------------------------------------------------------
# Address operands (me.h + me_chess.lib)
# ---------------------------------------------------------------------------

_PV_OPEN = re.compile(r"^\s*#\s*(?:ifdef\s+__programmers_view__|if\s+defined\s*\(\s*__programmers_view__\s*\))")
_PV_OPEN_NOT = re.compile(r"^\s*#\s*ifndef\s+__programmers_view__")
_PRIM_DECL = re.compile(r"^\s*(\w+)\s+(\w+)\s*\(([^)]*)\)")


def _param_types(params: str) -> list[str]:
    """`v8w32 a, u1 off` -> ['v8w32', 'u1']: the type is the first word of each parameter."""
    return [p.split()[0] for p in params.split(",") if p.strip()]


def address_operand_primitives(me_h: str) -> dict[str, int]:
    """Primitives whose operand is a vector of data-memory addresses, as the processor model
    me.h states it: a `__programmers_view__` block whose hardware branch declares an address
    primitive `<ret> <stem>_address(T, ...)` (the address-generation step) holds, in its
    programmer's-view branch, the operations that take that operand T. Returns the mangled
    primitive name (`<ret>_<name>_<param types>`, the me_chess.lib `prm` spelling) -> the
    index of the T parameter."""
    out: dict[str, int] = {}
    branch: str | None = None          # None outside a block, else "pv" / "hw"
    decls: dict[str, list[tuple[str, str, list[str]]]] = {"pv": [], "hw": []}

    def close() -> None:
        operand_types = {ps[0] for ret, name, ps in decls["hw"] if name.endswith("_address") and ps}
        for ret, name, ps in decls["pv"]:
            for idx, t in enumerate(ps):
                if t in operand_types:
                    out["_".join([ret, name, *ps])] = idx
                    break

    for line in me_h.splitlines():
        if _PV_OPEN.match(line) or _PV_OPEN_NOT.match(line):
            branch = "pv" if _PV_OPEN.match(line) else "hw"
            decls = {"pv": [], "hw": []}
        elif branch and re.match(r"^\s*#\s*else\b", line):
            branch = "hw" if branch == "pv" else "pv"
        elif branch and re.match(r"^\s*#\s*endif\b", line):
            close()
            branch = None
        elif branch:
            m = _PRIM_DECL.match(line)
            if m:
                decls[branch].append((m.group(1), m.group(2), _param_types(m.group(3))))
    return out


_LIB_ENTRY = re.compile(r"fnm\s*:\s*\"[^\"]*\"\s*'([^']*)'\s*;.*?prm\s*:\s*\(([^)]*)\)", re.S)


def intrinsic_primitives(lib: str) -> dict[str, tuple[str, ...]]:
    """me_chess.lib: intrinsic signature (`v32int8 load_4x16_lo(v8int32)`) -> the primitives it
    promotes to (its `prm` list)."""
    out: dict[str, tuple[str, ...]] = {}
    for entry in lib.split("\n}"):
        m = _LIB_ENTRY.search(entry)
        if m:
            out[" ".join(m.group(1).split())] = tuple(m.group(2).split())
    return out


def address_operands(me_h: str, lib: str) -> dict[str, int]:
    """Intrinsic signature -> index of its address operand, for every intrinsic that promotes
    to exactly one address-operand primitive (a promotion maps the operands in order)."""
    prims = address_operand_primitives(me_h)
    return {sig: prims[prm[0]] for sig, prm in intrinsic_primitives(lib).items()
            if len(prm) == 1 and prm[0] in prims}
