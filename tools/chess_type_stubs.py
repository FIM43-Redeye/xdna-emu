#!/usr/bin/env python3
"""Extract type sizes from me_chess_types.h and generate C struct stubs.

The chessTraitsOf<T> specializations come in two forms:

  Vector/custom types: literal bit count
    template <> struct chessTraitsOf<v16int32> {
        static const unsigned bits = 512;
        static const unsigned elems = 16;
    };

  Primitive C types: sizeof expression with a target-size comment
    template <> struct chessTraitsOf<int> {
        static const unsigned bits = sizeof(int) * __CHAR_BIT__; // pertinent to host, may differ from target 32;
    };

Commented-out blocks (lines prefixed with //!) are ignored.

Generated stubs give clang.cindex the size information it needs when parsing
me_chess_opns.h for the Chess-native intrinsic validation path.
"""

import re
import sys
from pathlib import Path

# C built-in types that the stub header must not redefine.
BUILTIN_C_TYPES: frozenset[str] = frozenset({
    "bool", "char", "signed char", "unsigned char",
    "short", "unsigned short", "int", "unsigned",
    "long", "unsigned long", "long long", "unsigned long long",
    "float", "double", "long double",
})

# Matches the opening line of a chessTraitsOf specialization.
_SPEC_OPEN = re.compile(
    r'^\s*template\s*<>\s*struct\s+chessTraitsOf<([^>]+)>\s*\{',
    re.MULTILINE,
)

# Matches a literal bit-count field: static const unsigned bits = NNN;
_BITS_LITERAL = re.compile(r'static\s+const\s+unsigned\s+bits\s*=\s*(\d+)\s*;')

# Matches a sizeof-based field with a target-size hint in the comment:
#   static const unsigned bits = sizeof(T) * __CHAR_BIT__; // ... target NNN;
_BITS_SIZEOF = re.compile(
    r'static\s+const\s+unsigned\s+bits\s*=\s*sizeof\([^)]+\)\s*\*\s*__CHAR_BIT__\s*;'
    r'\s*//.*?target\s+(\d+)\s*;'
)


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

    # Second pass: resolve vector types.
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


def parse_all_types(
    types_text: str,
    iss_text: str | None = None,
) -> dict[str, int]:
    """Parse chessTraitsOf types and optionally merge ISS property types.

    ISS types fill gaps only -- chessTraitsOf takes precedence.
    """
    traits = parse_chess_traits(types_text)
    if iss_text is not None:
        iss_types = parse_iss_property_comments(iss_text)
        for name, bits in iss_types.items():
            traits.setdefault(name, bits)
    return traits


def parse_chess_traits(text: str) -> dict[str, int]:
    """Parse chessTraitsOf<T> specializations, returning {type_name: bits}.

    Handles:
    - Literal ``bits = NNN`` assignments (vectors, accumulators, masks).
    - ``sizeof(T) * __CHAR_BIT__`` assignments with a trailing comment that
      ends in ``target NNN;`` (primitive C types whose size is host-dependent
      but annotated with the AIE target size).

    Lines starting with ``//!`` are treated as commented-out and skipped.
    """
    traits: dict[str, int] = {}

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip lines that are commented out with the Chess //! convention.
        if line.lstrip().startswith('//!'):
            i += 1
            continue

        m = _SPEC_OPEN.match(line)
        if m:
            type_name = m.group(1).strip()

            # Accumulate lines until the closing '};' so we can search the
            # full block for a bits field.
            block_lines = [line]
            j = i + 1
            while j < len(lines):
                block_lines.append(lines[j])
                if '};' in lines[j]:
                    break
                j += 1

            block = '\n'.join(block_lines)

            bm = _BITS_LITERAL.search(block)
            if bm:
                traits[type_name] = int(bm.group(1))
            else:
                bm = _BITS_SIZEOF.search(block)
                if bm:
                    traits[type_name] = int(bm.group(1))

            i = j + 1
        else:
            i += 1

    return traits


def generate_stub_header(traits: dict[str, int]) -> str:
    """Generate a C header with sized struct stubs for each Chess type.

    Built-in C types (int, float, etc.) are skipped because the compiler
    already knows their sizes.  The struct layout is a char array of the
    correct byte length so that sizeof(T) returns the right value when clang
    parses signatures that contain these types.
    """
    lines = [
        "// Auto-generated type stubs for clang.cindex parsing of me_chess_opns.h",
        "// DO NOT EDIT -- regenerate with: python3 tools/chess_type_stubs.py",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]

    skip = BUILTIN_C_TYPES

    for type_name, bits in sorted(traits.items()):
        if type_name in skip:
            continue
        # Round up to at least 1 byte; most types are byte-aligned but
        # ISS types like u1 (1 bit) and pmode_t (26 bits) are not.
        byte_size = max(1, (bits + 7) // 8)
        lines.append(f"struct {type_name} {{ char _data[{byte_size}]; }};  // {bits} bits")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate C type stubs from me_chess_types.h chessTraitsOf specializations."
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
        "--output",
        default=None,
        help="Output path for generated header (default: print to stdout)",
    )
    args = parser.parse_args()

    types_text = Path(args.types_header).read_text()
    iss_text = Path(args.iss_types_header).read_text() if args.iss_types_header else None
    traits = parse_all_types(types_text, iss_text)
    header = generate_stub_header(traits)

    if args.output:
        Path(args.output).write_text(header)
        print(f"Wrote {len(traits)} type stubs to {args.output}")
    else:
        print(header)


if __name__ == "__main__":
    main()
