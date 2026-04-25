#!/usr/bin/env python3
"""
Parse red8_prmx_hw from me_inline_primitives.h to extract the permute crossbar
routing table, then validate against known dense permutation.

The function maps 128 input bytes a[0..127] to 512 output bytes r[0..511]
based on a 789-bit control word m. Each output is a bitwise-OR of masked
inputs: r[i] = OR(a[j] & mask_j) where mask_j is derived from m bits.

When exactly one mask evaluates to all-1s, the corresponding a[j] passes
through to r[i]. When all masks are 0, r[i] = 0.

Previous regex parser failed because VBit<8, true>(VBit<1, true>(m.extract(...)))
has 3 levels of nested parens, but the regex only handled 2. This version uses
balanced paren matching instead.
"""

import re
import json
import sys
from collections import defaultdict

SOURCE_FILE = "/home/triple/npu-work/amd-unified-software/aietools/data/aie_ml/lib/isg/me_inline_primitives.h"
FUNC_START = 5189
FUNC_END = 9047


def read_function():
    """Read the function source lines."""
    lines = []
    with open(SOURCE_FILE, "r") as f:
        for i, line in enumerate(f, 1):
            if FUNC_START <= i <= FUNC_END:
                lines.append(line.rstrip())
    return lines


def find_matching_paren(s, start):
    """Find the index of the closing paren matching the open paren at s[start].

    Returns the index of the matching ')' or -1 if not found.
    """
    assert s[start] == '(', f"Expected '(' at position {start}, got '{s[start]}'"
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def extract_m_bits(expr):
    """Extract all m.extract(0xNu) bit indices from an expression."""
    return [int(x, 16) for x in re.findall(r'm\.extract\(0x([0-9A-Fa-f]+)u\)', expr)]


def extract_a_and_selector(expr):
    """Extract (a_index, selector_substring) pairs from an expression.

    Finds patterns like: a.elem(0xNu).val & VBit<8, true>(SELECTOR)
    Uses balanced paren matching to correctly capture SELECTOR regardless
    of nesting depth.
    """
    terms = []
    # Find all a.elem( occurrences
    for m in re.finditer(r'a\.elem\(0x([0-9A-Fa-f]+)u\)\.val\s*&\s*VBit<8,\s*true>\(', expr):
        a_idx = int(m.group(1), 16)
        # The open paren is at the end of the match, right before the selector
        paren_start = m.end() - 1  # position of '(' in VBit<8, true>(
        paren_end = find_matching_paren(expr, paren_start)
        if paren_end > paren_start:
            selector = expr[paren_start + 1:paren_end]
            terms.append((a_idx, selector))
    return terms


def parse_expr_terms(expr, sel_bits, var_terms):
    """Parse an expression to extract all (a_index, m_bits) terms.

    Handles:
    - Direct a.elem(N).val & VBit<8, true>(selector) terms
    - References to intermediate variables (ptN.val, rN.val)
    - OR combinations of the above
    """
    terms = []

    # Find direct a.elem terms with their selectors (balanced paren matching)
    for a_idx, selector in extract_a_and_selector(expr):
        # Check if selector is a variable reference
        sel_var_match = re.match(r'^(sel\d+)$', selector.strip())
        if sel_var_match and sel_var_match.group(1) in sel_bits:
            bits = sel_bits[sel_var_match.group(1)]
        else:
            # Inline selector - extract m bits from full substring
            bits = set(extract_m_bits(selector))

        if bits:
            terms.append((a_idx, bits))

    # Find references to intermediate variables (ptN.val, rN.val etc.)
    # These are ORed in, so we include their terms
    for m in re.finditer(r'\b(\w+)\.val\b', expr):
        var_name = m.group(1)
        if var_name in var_terms:
            # Make sure it's not part of a.elem(...).val
            start = m.start()
            prefix = expr[max(0, start - 10):start]
            if 'a.elem' not in prefix and '.elem' not in prefix:
                terms.extend(var_terms[var_name])

    return terms


def parse_function():
    """Parse the function and extract the routing table.

    Returns:
        routing: dict mapping output_idx -> [(a_idx, set_of_m_bit_indices), ...]
        Each entry means: output gets a[a_idx] masked by OR of those m bits.
    """
    lines = read_function()

    sel_bits = {}     # sel variable name -> set of m-bit indices
    var_terms = {}    # variable name -> [(a_idx, set of m_bits), ...]
    routing = {}      # output_idx -> [(a_idx, set of m_bits), ...]

    # Join lines into statements (each ending with ;)
    statements = []
    current = ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line == "{" or line == "}":
            continue
        current += " " + line
        if ";" in current:
            statements.append(current.strip())
            current = ""

    for stmt in statements:
        # Skip boilerplate
        if "inline" in stmt and "red8_prmx_hw" in stmt:
            continue
        if stmt.startswith("return"):
            continue
        if "VBitZeroInitializeTag" in stmt:
            continue

        # Pattern 1: Selector variable
        # VBit<1, true> selN = VBit<1, true>(EXPR);
        sel_match = re.match(r'VBit<1,\s*true>\s+(sel\d+)\s*=\s*VBit<1,\s*true>\(', stmt)
        if sel_match:
            name = sel_match.group(1)
            # Find the matching paren for the outer VBit<1, true>(
            paren_start = stmt.index('=')
            # Find VBit<1, true>( after the =
            rhs_start = stmt.index('(', paren_start + 1)
            # But actually there are two levels: VBit<1, true>(VBit<1, true>(...))
            # We want the inner expression, so find the outer ( of the assignment
            # Actually, just extract all m.extract bits from the entire RHS
            rhs = stmt[stmt.index('=') + 1:]
            bits = extract_m_bits(rhs)
            sel_bits[name] = set(bits)
            continue

        # Pattern 2: u8 variable (ptN, rN, etc.)
        # me_primitive::u8 NAME = EXPR;
        pt_match = re.match(r'me_primitive::u8\s+(\w+)\s*=\s*(.*);', stmt)
        if pt_match:
            name = pt_match.group(1)
            expr = pt_match.group(2)
            terms = parse_expr_terms(expr, sel_bits, var_terms)
            var_terms[name] = terms
            continue

        # Pattern 3: VBit<8, false> intermediate
        vbit8_match = re.match(r'VBit<8,\s*false>\s+(\w+)\s*=\s*(.*);', stmt)
        if vbit8_match:
            name = vbit8_match.group(1)
            expr = vbit8_match.group(2)
            terms = parse_expr_terms(expr, sel_bits, var_terms)
            var_terms[name] = terms
            continue

        # Pattern 4: Output assignment
        # r.elem(0xNu) = me_primitive::w8(EXPR).val;
        out_match = re.match(
            r'r\.elem\(0x([0-9A-Fa-f]+)u\)\s*=\s*me_primitive::w8\((.*)\)\.val;',
            stmt
        )
        if out_match:
            out_idx = int(out_match.group(1), 16)
            expr = out_match.group(2)
            # The expr references a variable: rN.val or ptN.val
            terms = parse_expr_terms(expr, sel_bits, var_terms)
            routing[out_idx] = terms
            continue

    return routing


def consolidate_routing(routing):
    """Consolidate routing: merge terms with same a_index."""
    consolidated = {}
    for out_idx, terms in routing.items():
        merged = defaultdict(set)
        for a_idx, bits in terms:
            merged[a_idx] |= bits
        consolidated[out_idx] = [
            (a_idx, sorted(bits)) for a_idx, bits in sorted(merged.items())
        ]
    return consolidated


def evaluate_routing(routing, m_bits_set):
    """Evaluate routing for a given set of m bits that are 1.

    Returns: dict mapping output_idx -> a_idx (or None/list).
    A term is active if ANY of its control bits are set (OR semantics).
    Active terms OR together at the output.
    """
    result = {}
    for out_idx in range(512):
        if out_idx not in routing:
            result[out_idx] = None
            continue

        active_inputs = []
        for a_idx, bits in routing[out_idx]:
            if any(b in m_bits_set for b in bits):
                active_inputs.append(a_idx)

        if len(active_inputs) == 0:
            result[out_idx] = None
        elif len(active_inputs) == 1:
            result[out_idx] = active_inputs[0]
        else:
            result[out_idx] = active_inputs  # Multiple active -> OR

    return result


# ---------- Control word construction ----------

def decode_mask(m4):
    """4-bit mask nibble -> (sel0, sel1) selection codes."""
    table = {
        0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (1, 1), 4: (0, 2),
        5: (1, 2), 6: (2, 2), 8: (0, 4), 9: (1, 4), 10: (2, 4), 12: (4, 4),
    }
    return table.get(m4, (0, 0))


def mask2sel_pmode23(mask128):
    """Build r16x8 array (64 entries of 3-bit values) from 128-bit mask."""
    r16x8 = [0] * 64
    for ig in range(4):       # inner group
        for c in range(8):    # column
            g = ig + c * 4    # mask nibble index (column-major)
            m4 = (mask128 >> (4 * g)) & 0xF
            sel0, sel1 = decode_mask(m4)
            r16x8[c + 8 * (2 * ig)] = sel0
            r16x8[c + 8 * (2 * ig + 1)] = sel1
    return r16x8


def build_pmode_bits(pmode_num):
    """Build the lower 21 bits of pmode for a given pmode number.

    In the hardware, pmode is a register with individual bits set for
    each active mode. The lower 21 bits are passed to the crossbar.
    """
    if pmode_num < 21:
        return 1 << pmode_num
    else:
        return 0  # Bits 21+ are NOT in the lower 21 bits


def build_m_word(pmode_num, mask128=0):
    """Build the 789-bit m word for a given pmode and mask.

    m = concat(smode[768 bits], pmode_lower[21 bits])
    Bit layout: m[0..20] = pmode_lower, m[21..788] = smode
    smode = concat(rbfxbf[96], r16x16[96], r16x8[192], r8x8[192], r8x4[192])
    """
    pmode_lower = build_pmode_bits(pmode_num)

    smode = 0
    if pmode_num == 23 and mask128 != 0:
        r16x8 = mask2sel_pmode23(mask128)
        for i in range(64):
            smode |= (r16x8[i] & 0x7) << (384 + 3 * i)

    m = (smode << 21) | pmode_lower
    return m


def m_to_bit_set(m_val, num_bits=789):
    """Convert an integer m value to a set of bit indices that are 1."""
    return {i for i in range(num_bits) if (m_val >> i) & 1}


def validate_dense(routing):
    """Validate the parsed routing against known dense permutation tables.

    For dense pmode 4 (i16 x i8, no mask), the X-side permutation is
    fully specified by aietools constants.py permute_x[4]. Every output
    should map to exactly one input with no ambiguity.
    """
    sys.path.insert(0, '/home/triple/npu-work/xdna-emu/tools/aietools-py3')
    import constants as C_module
    C = C_module.C

    print("=" * 70)
    print("VALIDATION: Dense pmode 4 (i16 x i8)")
    print("=" * 70)

    m_val = build_m_word(4)  # pmode 4: bit 4 set in lower 21
    m_bits = m_to_bit_set(m_val)
    print(f"Control word: pmode_lower=0x{m_val & 0x1FFFFF:06x}, active m bits: {sorted(m_bits)}")

    result = evaluate_routing(routing, m_bits)
    xt = C.permute_x[4]

    match = mismatch = zero = multi = 0
    first_mismatches = []
    first_zeros = []
    for i in range(512):
        r = result.get(i)
        expected = xt[i]
        if r is None:
            zero += 1
            if len(first_zeros) < 5:
                first_zeros.append((i, expected))
        elif isinstance(r, list):
            multi += 1
            if expected in r:
                pass  # Correct input is among the OR'd set
        else:
            if r == expected:
                match += 1
            else:
                mismatch += 1
                if len(first_mismatches) < 5:
                    first_mismatches.append((i, r, expected))

    print(f"Results: {match} match, {mismatch} mismatch, {zero} zero, {multi} multi")
    if first_mismatches:
        for i, got, exp in first_mismatches:
            print(f"  MISMATCH output[{i}]: got a[{got}], expected a[{exp}]")
    if first_zeros:
        for i, exp in first_zeros:
            print(f"  ZERO     output[{i}]: expected a[{exp}]")

    return match == 512


def main():
    print("Parsing red8_prmx_hw (balanced paren matching)...")
    routing_raw = parse_function()
    routing = consolidate_routing(routing_raw)

    parsed_outputs = len(routing)
    total_terms = sum(len(terms) for terms in routing.values())
    print(f"Parsed {parsed_outputs} output assignments with {total_terms} total terms")

    # Validate against dense pmode 4
    ok = validate_dense(routing)

    if ok:
        print("\nVALIDATION PASSED -- parser is correct")
    else:
        print("\nVALIDATION FAILED -- parser still has issues")

    # Save routing table
    json_routing = {}
    for out_idx, terms in sorted(routing.items()):
        json_routing[str(out_idx)] = [[a_idx, bits] for a_idx, bits in terms]

    import os
    out_dir = "/home/triple/npu-work/xdna-emu/tools/aietools-py3"
    out_path = os.path.join(out_dir, "crossbar_routing.json")
    with open(out_path, "w") as f:
        json.dump(json_routing, f, indent=2)
    print(f"\nRouting table saved to {out_path}")

    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
