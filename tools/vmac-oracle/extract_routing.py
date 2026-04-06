#!/usr/bin/env python3
"""Extract routing tables from prmx_hw/prmy_hw C++ OR-mux expressions.

Parses the machine-generated C++ in vmac_functions.inc and extracts:
- For each output element: list of (input_index, [control_bit_indices])
- This represents: output = OR of (input[idx] & ANY(m[bits])) for each entry

Output: JSON routing tables that can be used to generate Rust code.

The C++ has these patterns (machine-generated, very regular):
  1. Selection signals: VBit<1,true> selN = VBit<1,true>((m.extract(X) | m.extract(Y) | ...))
  2. Masked inputs:    u8 ptN = u8((a.elem(X).val & VBit<8,true>(sel_expr))).val
  3. OR combinations:  u8 rN = (term1 | term2 | ...)
  4. Output assigns:   r.elem(0xNN) = w8(rN.val).val
"""

import re
import json
import sys
from collections import defaultdict
from pathlib import Path


def extract_m_bits(expr: str) -> list[int]:
    """Extract all m.extract(0xNN) bit indices from an expression string."""
    return [int(m, 16) for m in re.findall(r'm\.extract\(0x([0-9A-Fa-f]+)u?\)', expr)]


def find_masked_inputs(rhs: str, w_str: str, sel_vars: dict, neg_sel_vars: dict) -> list[tuple[int, frozenset, bool]]:
    """Find all (input_idx, m_bit_set, is_negated) entries.

    Matches patterns like: a.elem(0xNN).val & VBit<W, true>(...)
    where ... can have arbitrary nesting depth.

    is_negated=True means the selection fires when NONE of the bits are set.
    """
    entries = []
    search_str = 'a.elem('
    pos = 0
    while True:
        idx = rhs.find(search_str, pos)
        if idx == -1:
            break
        pos = idx + len(search_str)

        hex_match = re.match(r'0x([0-9A-Fa-f]+)u?\)', rhs[pos:])
        if not hex_match:
            continue
        input_idx = int(hex_match.group(1), 16)
        pos += hex_match.end()

        amp_match = re.match(r'\.val\s*&\s*VBit<' + w_str + r',\s*true>\(', rhs[pos:])
        if not amp_match:
            continue
        pos += amp_match.end()

        depth = 1
        sel_start = pos
        while pos < len(rhs) and depth > 0:
            if rhs[pos] == '(':
                depth += 1
            elif rhs[pos] == ')':
                depth -= 1
            pos += 1
        sel_expr = rhs[sel_start:pos - 1]

        bits = set(extract_m_bits(sel_expr))
        is_negated = False
        for name in sel_vars:
            if name in sel_expr:
                bits |= sel_vars[name]
                # Check if this named sel is negated
                if name in neg_sel_vars:
                    _, neg = neg_sel_vars[name]
                    if neg:
                        is_negated = True
        # Check for inline negation: ~(bits)
        if sel_expr.strip().startswith('~') or sel_expr.strip().startswith('!'):
            is_negated = True
        if bits:
            entries.append((input_idx, frozenset(bits), is_negated))
    return entries


def find_var_refs(rhs: str, val_vars: dict) -> list[tuple[int, frozenset, bool]]:
    """Find all variable references (ptNNN.val, rNNN.val) that aren't a.elem().val."""
    entries = []
    for ref_match in re.finditer(r'\b((?:pt|r)\w+)\.val', rhs):
        ref_name = ref_match.group(1)
        start = ref_match.start()
        prefix = rhs[max(0, start - 30):start]
        if 'a.elem(' in prefix:
            continue
        if ref_name in val_vars:
            entries.extend(val_vars[ref_name])
    return entries


def parse_routing_function(lines: list[str], elem_width: int) -> dict:
    """Parse a red8_prmx_hw or red8_prmy_hw function body.

    elem_width: 8 for prmx (u8/w8), 4 for prmy (u4/w4)

    Returns: dict mapping output_index -> [(input_index, [m_bit_indices])]
    """
    w_str = str(elem_width)

    # Symbol tables
    sel_vars: dict[str, set[int]] = {}
    val_vars: dict[str, list[tuple[int, frozenset[int], bool]]] = {}
    output_table: dict[int, list[tuple[int, frozenset[int], bool]]] = {}

    # Negated selection vars: name -> (positive_bits, is_negated)
    # For negated: sel = ~(bit_a | bit_b | ...) → fires when NONE of bits are set
    neg_sel_vars: dict[str, tuple[set[int], bool]] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Pattern 1: Selection signal assignment
        sel_match = re.match(r'VBit<1,\s*true>\s+(\w+)\s*=\s*VBit<1,\s*true>\((.+)\);', line)
        if sel_match:
            name = sel_match.group(1)
            expr = sel_match.group(2)
            # Check for negation: ~(expr)
            is_negated = expr.strip().startswith('~')
            bits = set(extract_m_bits(expr))
            for other_name in list(sel_vars.keys()):
                if other_name != name and other_name in expr:
                    bits |= sel_vars[other_name]
            sel_vars[name] = bits
            neg_sel_vars[name] = (bits, is_negated)
            continue

        # Pattern 2: Value variable assignment
        val_match = re.match(r'me_primitive::u' + w_str + r'\s+(\w+)\s*=\s*(.+);', line)
        if val_match:
            name = val_match.group(1)
            rhs = val_match.group(2)
            entries = find_masked_inputs(rhs, w_str, sel_vars, neg_sel_vars)
            entries.extend(find_var_refs(rhs, val_vars))
            if entries:
                val_vars[name] = entries
            continue

        # Pattern 2c: VBit<W, false> wrapper (intermediate)
        vbit_val_match = re.match(r'VBit<' + w_str + r',\s*false>\s+(\w+)\s*=\s*(.+);', line)
        if vbit_val_match:
            name = vbit_val_match.group(1)
            rhs = vbit_val_match.group(2)
            entries = find_masked_inputs(rhs, w_str, sel_vars, neg_sel_vars)
            entries.extend(find_var_refs(rhs, val_vars))
            if entries:
                val_vars[name] = entries
            continue

        # Pattern 3: Output assignment
        out_match = re.match(r'r\.elem\(0x([0-9A-Fa-f]+)u?\)\s*=\s*me_primitive::w' + w_str + r'\((\w+)\.val\)\.val;', line)
        if out_match:
            out_idx = int(out_match.group(1), 16)
            var_name = out_match.group(2)
            if var_name in val_vars:
                output_table[out_idx] = val_vars[var_name]
            else:
                output_table[out_idx] = []
            continue

    return output_table


def extract_function_body(all_lines: list[str], func_sig: str) -> list[str]:
    """Extract the body of a C++ inline function (between first { and matching })."""
    start = None
    for i, line in enumerate(all_lines):
        if func_sig in line:
            start = i
            break
    if start is None:
        raise ValueError(f"Function not found: {func_sig}")

    # Find matching braces
    depth = 0
    body_start = None
    for i in range(start, len(all_lines)):
        for ch in all_lines[i]:
            if ch == '{':
                if depth == 0:
                    body_start = i + 1
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return all_lines[body_start:i]
    raise ValueError(f"Unmatched braces for {func_sig}")


def main():
    inc_path = Path(__file__).parent / "vmac_functions.inc"
    all_lines = inc_path.read_text().splitlines()

    print("Parsing red8_prmx_hw (crossbar, 128->512 bytes)...", file=sys.stderr)
    prmx_body = extract_function_body(all_lines, "red8_prmx_hw")
    prmx_table = parse_routing_function(prmx_body, 8)
    print(f"  Extracted {len(prmx_table)} output routes", file=sys.stderr)

    # Validate: should have 512 outputs
    missing_prmx = [i for i in range(512) if i not in prmx_table]
    if missing_prmx:
        print(f"  WARNING: Missing prmx outputs: {missing_prmx[:20]}...", file=sys.stderr)

    # Check for outputs with no entries
    empty_prmx = [i for i in range(512) if i in prmx_table and not prmx_table[i]]
    if empty_prmx:
        print(f"  WARNING: Empty prmx outputs: {empty_prmx[:20]}...", file=sys.stderr)

    print("\nParsing red8_prmy_hw (Y-perm, 128->512 nibbles)...", file=sys.stderr)
    prmy_body = extract_function_body(all_lines, "red8_prmy_hw")
    prmy_table = parse_routing_function(prmy_body, 4)
    print(f"  Extracted {len(prmy_table)} output routes", file=sys.stderr)

    missing_prmy = [i for i in range(512) if i not in prmy_table]
    if missing_prmy:
        print(f"  WARNING: Missing prmy outputs: {missing_prmy[:20]}...", file=sys.stderr)

    empty_prmy = [i for i in range(512) if i in prmy_table and not prmy_table[i]]
    if empty_prmy:
        print(f"  WARNING: Empty prmy outputs: {empty_prmy[:20]}...", file=sys.stderr)

    # Statistics
    prmx_entry_counts = [len(prmx_table.get(i, [])) for i in range(512)]
    prmy_entry_counts = [len(prmy_table.get(i, [])) for i in range(512)]
    print(f"\n  prmx entries per output: min={min(prmx_entry_counts)}, "
          f"max={max(prmx_entry_counts)}, avg={sum(prmx_entry_counts)/512:.1f}", file=sys.stderr)
    print(f"  prmy entries per output: min={min(prmy_entry_counts)}, "
          f"max={max(prmy_entry_counts)}, avg={sum(prmy_entry_counts)/512:.1f}", file=sys.stderr)

    # Collect all referenced m-bits
    all_prmx_bits = set()
    for entries in prmx_table.values():
        for _, bits, _ in entries:
            all_prmx_bits |= bits
    all_prmy_bits = set()
    for entries in prmy_table.values():
        for _, bits, _ in entries:
            all_prmy_bits |= bits
    print(f"  prmx uses m-bits: {min(all_prmx_bits)}-{max(all_prmx_bits)} "
          f"({len(all_prmx_bits)} unique)", file=sys.stderr)
    print(f"  prmy uses m-bits: {min(all_prmy_bits)}-{max(all_prmy_bits)} "
          f"({len(all_prmy_bits)} unique)", file=sys.stderr)

    # Convert to serializable format
    # Each entry: [input_idx, [bit_indices], is_negated]
    def table_to_json(table):
        result = {}
        for out_idx in range(512):
            entries = table.get(out_idx, [])
            result[str(out_idx)] = [[inp, sorted(bits), neg] for inp, bits, neg in entries]
        return result

    # Count negated entries
    neg_prmx = sum(1 for entries in prmx_table.values() for _, _, neg in entries if neg)
    neg_prmy = sum(1 for entries in prmy_table.values() for _, _, neg in entries if neg)
    print(f"  prmx negated entries: {neg_prmx}", file=sys.stderr)
    print(f"  prmy negated entries: {neg_prmy}", file=sys.stderr)

    output = {
        "prmx": {
            "description": "Crossbar routing: 128 input bytes -> 512 output bytes",
            "control_bits": 789,
            "control_layout": "concat(smode[768], pmode[0:20])",
            "input_count": 128,
            "output_count": 512,
            "routes": table_to_json(prmx_table),
        },
        "prmy": {
            "description": "Y-perm routing: 128 input nibbles -> 512 output nibbles",
            "control_bits": 26,
            "control_layout": "pmode[0:25]",
            "input_count": 128,
            "output_count": 512,
            "routes": table_to_json(prmy_table),
        },
    }

    json.dump(output, sys.stdout, indent=1)
    print(file=sys.stderr)
    print("Done. Pipe stdout to a .json file.", file=sys.stderr)


if __name__ == "__main__":
    main()
