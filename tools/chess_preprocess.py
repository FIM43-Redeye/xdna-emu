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
