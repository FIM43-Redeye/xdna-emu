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
        "// Source: me_chess_types.h chessTraitsOf<T> specializations",
        "// DO NOT EDIT -- regenerate with: python3 tools/chess_type_stubs.py",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]

    for type_name, bits in sorted(traits.items()):
        if type_name in BUILTIN_C_TYPES:
            continue
        # Round up to at least 1 byte; bits must be a multiple of 8 in
        # practice, but guard against pathological entries.
        byte_size = max(1, bits // 8)
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
        "--output",
        default=None,
        help="Output path for generated header (default: print to stdout)",
    )
    args = parser.parse_args()

    text = Path(args.types_header).read_text()
    traits = parse_chess_traits(text)
    header = generate_stub_header(traits)

    if args.output:
        Path(args.output).write_text(header)
        print(f"Wrote {len(traits)} type stubs to {args.output}")
    else:
        print(header)


if __name__ == "__main__":
    main()
