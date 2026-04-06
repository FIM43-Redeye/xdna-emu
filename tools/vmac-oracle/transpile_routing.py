#!/usr/bin/env python3
"""Transpile C++ red8_prmx_hw/red8_prmy_hw to Rust functions.

Instead of extracting routing tables (which can't handle negated selections),
this script generates Rust code that directly evaluates the same expressions
as the C++ model, including negations.

Output: Rust source code with two functions:
  - red8_prmx_hw(a: &[u8; 128], m: &[u64; 13]) -> [u8; 512]
  - red8_prmy_hw(a: &[u8; 128], m: u32) -> [u8; 512]
"""

import re
import sys
from pathlib import Path


def transpile_expr(expr: str, elem_width: int) -> str:
    """Transpile a C++ VBit expression to Rust.

    Handles:
    - m.extract(0xNN) -> m_bit(m, 0xNN)
    - a.elem(0xNN).val -> a[0xNN]
    - VBit<W, true>(sel) -> mask_W(sel)
    - VBit<W, false>(...) -> (...)  (just a cast, passthrough)
    - ~ operator -> !
    - | operator -> |
    - & operator -> &
    - Named variables: pass through
    """
    w = str(elem_width)

    # Replace m.extract(0xNNu) with m_bit(m, 0xNN)
    expr = re.sub(r'VBit<1,\s*false>\(m\.extract\(0x([0-9A-Fa-f]+)u?\)\)', r'mb(m, 0x\1)', expr)
    expr = re.sub(r'm\.extract\(0x([0-9A-Fa-f]+)u?\)', r'mb(m, 0x\1)', expr)

    # Replace a.elem(0xNN).val with a[0xNN]
    expr = re.sub(r'a\.elem\(0x([0-9A-Fa-f]+)u?\)\.val', r'a[0x\1]', expr)

    # Replace VBit<W, true>(sel) used as mask -> sel (bool, will be used with & on u8)
    # The & VBit<W, true>(sel) pattern means "mask byte/nibble with 1-bit sel"
    expr = re.sub(r'VBit<' + w + r',\s*true>\(([^)]+)\)', r'bm(\1)', expr)

    # Replace VBit<1, true>(expr) -> (expr) (just a cast to 1-bit signed)
    expr = re.sub(r'VBit<1,\s*true>\(([^)]*(?:\([^)]*\))*[^)]*)\)', r'(\1)', expr)

    # Replace VBit<W, false>(expr) -> (expr) (unsigned cast, passthrough)
    expr = re.sub(r'VBit<' + w + r',\s*false>\(([^)]*(?:\([^)]*\))*[^)]*)\)', r'(\1)', expr)
    # Handle deeper nesting
    expr = re.sub(r'VBit<\d+,\s*(?:true|false)>\(([^)]*(?:\([^)]*\))*[^)]*)\)', r'(\1)', expr)

    # Replace .val suffix (both NAME.val and ).val)
    expr = re.sub(r'\.val\b', '', expr)

    # Replace ~ with ! (bitwise not -> logical not for bools, bitwise for u8)
    expr = expr.replace('~', '!')

    # Replace || with | (C++ logical OR used in bit expressions)
    expr = expr.replace('||', '|')

    # Clean up: me_primitive::uW(...) -> (...)
    expr = re.sub(r'me_primitive::u' + w + r'\(([^)]*)\)', r'(\1)', expr)
    expr = re.sub(r'me_primitive::w' + w + r'\(([^)]*)\)', r'(\1)', expr)

    return expr.strip()


def transpile_function(lines: list[str], func_name: str, elem_width: int,
                       input_count: int, output_count: int, ctrl_bits: int) -> list[str]:
    """Transpile one routing function to Rust."""
    w = str(elem_width)

    rust_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#ifdef') or line.startswith('#endif') or line.startswith('std::cerr'):
            continue

        # Skip debug blocks entirely
        if 'PDG_DEBUG' in line or 'std::cerr' in line:
            continue

        # Selection signal: VBit<1, true> NAME = VBit<1, true>((EXPR));
        sel_match = re.match(r'VBit<1,\s*true>\s+(\w+)\s*=\s*VBit<1,\s*true>\((.+)\);', line)
        if sel_match:
            name = sel_match.group(1)
            expr = transpile_expr(sel_match.group(2), elem_width)
            rust_lines.append(f'    let {name}: bool = ({expr});')
            continue

        # Value assignment: me_primitive::uW NAME = ...;
        val_match = re.match(r'me_primitive::u' + w + r'\s+(\w+)\s*=\s*(.+);', line)
        if val_match:
            name = val_match.group(1)
            expr = transpile_expr(val_match.group(2), elem_width)
            rust_lines.append(f'    let {name}: u8 = ({expr});')
            continue

        # VBit<W, false> intermediate (treated like value variable)
        vbit_match = re.match(r'VBit<' + w + r',\s*false>\s+(\w+)\s*=\s*(.+);', line)
        if vbit_match:
            name = vbit_match.group(1)
            expr = transpile_expr(vbit_match.group(2), elem_width)
            rust_lines.append(f'    let {name}: u8 = ({expr});')
            continue

        # Output assignment: r.elem(0xNN) = me_primitive::wW(NAME.val).val;
        out_match = re.match(r'r\.elem\(0x([0-9A-Fa-f]+)u?\)\s*=\s*me_primitive::w' + w + r'\((\w+)\.val\)\.val;', line)
        if out_match:
            out_idx = out_match.group(1)
            var_name = out_match.group(2)
            rust_lines.append(f'    r[0x{out_idx}] = {var_name};')
            continue

        # Initialization (v512w8 r{...}) - skip
        if 'VBitZeroInitializeTag' in line:
            continue

        # Return statement - skip
        if line.startswith('return'):
            continue

    return rust_lines


def main():
    inc_path = Path(__file__).parent / "vmac_functions.inc"
    all_lines = inc_path.read_text().splitlines()

    # Extract function bodies
    def extract_body(sig):
        start = None
        for i, line in enumerate(all_lines):
            if sig in line:
                start = i
                break
        depth = 0
        body_start = None
        for i in range(start, len(all_lines)):
            for ch in all_lines[i]:
                if ch == '{':
                    if depth == 0: body_start = i + 1
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return all_lines[body_start:i]
        raise ValueError(f"Unmatched braces for {sig}")

    prmx_body = extract_body("red8_prmx_hw")
    prmy_body = extract_body("red8_prmy_hw")

    prmx_rust = transpile_function(prmx_body, "red8_prmx_hw", 8, 128, 512, 789)
    prmy_rust = transpile_function(prmy_body, "red8_prmy_hw", 4, 128, 512, 26)

    print("// Auto-generated from vmac_functions.inc by transpile_routing.py")
    print("// DO NOT EDIT MANUALLY.")
    print("")
    print("#[inline(always)]")
    print("fn mb(m: &[u64], bit: usize) -> bool {")
    print("    (m[bit / 64] >> (bit % 64)) & 1 != 0")
    print("}")
    print("")
    print("#[inline(always)]")
    print("fn bm(sel: bool) -> u8 {")
    print("    if sel { 0xFF } else { 0 }")
    print("}")
    print("")
    print("/// Crossbar routing: 128 input bytes -> 512 output bytes.")
    print("/// m: 789-bit control word = concat(smode[768], pmode[0:20])")
    print("pub fn eval_prmx(a: &[u8; 128], m: &[u64; 13]) -> [u8; 512] {")
    print("    let mut r = [0u8; 512];")
    for line in prmx_rust:
        print(line)
    print("    r")
    print("}")
    print("")
    print("/// Y-perm routing: 128 input nibbles -> 512 output nibbles.")
    print("/// m: 26-bit pmode value stored as [u64; 1]")
    print("pub fn eval_prmy(a_nibbles: &[u8; 128], pmode: u32) -> [u8; 512] {")
    print("    let m: &[u64] = &[pmode as u64];")
    print("    let a = a_nibbles;")
    print("    let mut r = [0u8; 512];")
    for line in prmy_rust:
        print(line)
    print("    r")
    print("}")

    print(f"\n// prmx: {len(prmx_rust)} lines, prmy: {len(prmy_rust)} lines", file=sys.stderr)


if __name__ == "__main__":
    main()
