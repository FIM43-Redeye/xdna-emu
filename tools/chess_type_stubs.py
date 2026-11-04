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
    r'class\s+(\w+)\s*;\s*//\s*property\(\s*(\d+)\s+bit\b'
)

# Matches: class NAME; // property( complex ELEM );  (two ELEMs)
_ISS_COMPLEX = re.compile(
    r'class\s+(\w+)\s*;\s*//\s*property\(\s*complex\s+(\w+)\s*\)'
)

# Target widths of the C types a complex property can be built from.
_C_ELEMENT_BITS = {"short": 16, "int": 32, "float": 32, "char": 8, "long long": 64}

# `namespace NS { struct T ... { unsigned low; int high; ... }; }` and `typedef struct NS::T A;`
_NS_STRUCT = re.compile(
    r'namespace\s+(\w+)\s*\{\s*struct\s+(\w+)[^{;]*\{(.*?)\n\s*\};', re.DOTALL)
_STRUCT_MEMBER = re.compile(r'^\s*(unsigned|int|short|unsigned short|float)\s+\w+\s*;', re.MULTILINE)
_TYPE_ALIAS = re.compile(r'^typedef\s+struct\s+([\w:]+)\s+(\w+)\s*;', re.MULTILINE)
_MEMBER_BITS = {"unsigned": 32, "int": 32, "short": 16, "unsigned short": 16, "float": 32}

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

    # Complex scalars are two elements of a C type or of an already-sized type.
    for m in _ISS_COMPLEX.finditer(text):
        name, elem = m.group(1), m.group(2)
        bits = _C_ELEMENT_BITS.get(elem, types.get(elem))
        if bits is not None:
            types[name] = 2 * bits

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


def parse_namespaced_structs(text: str) -> dict[str, int]:
    """Parse plain structs declared inside a namespace, returning {"ns::T": bits}.

    Only the data members count (the BitType typedef and constructors do not); a struct with
    a member of a type outside _MEMBER_BITS is left out rather than guessed at.
    """
    out: dict[str, int] = {}
    for m in _NS_STRUCT.finditer(text):
        members = [mm.group(1) for mm in _STRUCT_MEMBER.finditer(m.group(3))]
        if members:
            out[f"{m.group(1)}::{m.group(2)}"] = sum(_MEMBER_BITS[t] for t in members)
    return out


def parse_type_aliases(text: str) -> dict[str, str]:
    """Parse `typedef struct NS::T Alias;` lines, returning {Alias: "NS::T"}."""
    return {m.group(2): m.group(1) for m in _TYPE_ALIAS.finditer(text)}


def parse_types_header(types_text: str) -> tuple[dict[str, int], dict[str, str]]:
    """Every type the Chess types header itself defines: ({name: bits}, {alias: target}).

    chessTraitsOf covers most; `class T; // property(...)` declarations fill the scalars it
    leaves out (bfloat16, the complex types); plain structs in a namespace (dint) and the
    typedefs naming them (float64) come from their definitions.
    """
    traits = parse_chess_traits(types_text)
    for name, bits in parse_iss_property_comments(types_text).items():
        traits.setdefault(name, bits)
    traits.update(parse_namespaced_structs(types_text))
    return traits, parse_type_aliases(types_text)


def parse_all_types(
    types_text: str,
    iss_text: str | None = None,
) -> dict[str, int]:
    """Parse the Chess types header and optionally merge ISS property types.

    ISS types fill gaps only -- the types header takes precedence.
    """
    traits, _aliases = parse_types_header(types_text)
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


def generate_stub_header(traits: dict[str, int], aliases: dict[str, str] | None = None) -> str:
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
        if "::" in type_name:
            ns, bare = type_name.rsplit("::", 1)
            lines.append(f"namespace {ns} {{ struct {bare} {{ char _data[{byte_size}]; }}; }}"
                         f"  // {bits} bits")
        else:
            lines.append(f"struct {type_name} {{ char _data[{byte_size}]; }};  // {bits} bits")

    for alias, target in sorted((aliases or {}).items()):
        lines.append(f"typedef {target} {alias};")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate C type stubs from me_chess_types.h chessTraitsOf specializations."
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
