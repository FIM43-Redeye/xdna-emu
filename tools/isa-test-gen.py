#!/usr/bin/env python3
"""ISA test generator for xdna-emu instruction validation.

Reads aie2-isa.json (exported from the emulator's TableGen parser) and
generates assembly test programs that exercise individual instructions.

Tasks:
  - Classify instructions as testable or skipped (with reason).
  - Map register kinds to concrete register names.
  - Generate assembly test points: load inputs, execute, store outputs.
  - Build mega-programs combining multiple test points.

Usage:
  python3 tools/isa-test-gen.py [--isa tools/aie2-isa.json] [--out /tmp/tests]
"""

import json
import os
import re
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mnemonics that are branch/jump/call/return -- never testable in isolation.
BRANCH_MNEMONICS = frozenset({
    "j", "jl", "jnz", "jnzd", "jz",
    "ret",
    # These are not in the current ISA JSON but guard against future additions.
    "call", "bnez", "beqz",
})

# Mnemonics that are lock acquire/release -- need real lock hardware.
LOCK_MNEMONICS = frozenset({
    "acq", "acq.cond", "rel", "rel.cond",
})

# Mnemonics that are NOPs or side-effect-only (no testable output).
SIDE_EFFECT_MNEMONICS = frozenset({
    "nopx", "nopa", "nopb", "nopm", "nops", "nopv", "nopxm",
    "done", "event",
})

# Mnemonics that interact with the stream switch -- need live streams.
STREAM_MNEMONICS = frozenset({
    "mov.nb", "mov.nb.tlast", "mov.tlast",
    "mov.ph", "mov.ph.nb", "mov.ph.nb.tlast", "mov.ph.tlast",
    "mov.cph", "mov.cph.nb", "mov.cph.nb.tlast", "mov.cph.tlast",
    "mov.d1", "mov.d2", "mov.d3", "mov.d4", "mov.d5", "mov.d6",
})

# Register kinds we know how to load/store.
KNOWN_REGISTER_KINDS = frozenset({
    "scalar", "pointer", "vector256", "vector512", "accumulator",
    "control", "modifier_m", "modifier_dj",
})

# Operand types that are safe to handle.
SAFE_OPERAND_TYPES = frozenset({
    "register", "immediate",
})

# Size in bytes for each register kind (for offset calculations).
REGISTER_SIZES = {
    "scalar": 4,
    "pointer": 4,
    "vector256": 32,
    "vector512": 64,
    "accumulator": 64,  # loaded as 2x vector256
    "control": 4,
    "modifier_m": 4,
    "modifier_dj": 4,
}

# Maximum positive immediate offset for lda/st (6-bit signed, step 4).
# Range: [-128, 124].  We use only positive offsets in generated code.
MAX_SCALAR_OFFSET = 124

# Maximum positive immediate offset for vlda/vst (6-bit signed, step 32).
# Range: [-1024, 992].
MAX_VECTOR_OFFSET = 992


# ---------------------------------------------------------------------------
# Task 2: Instruction Classification
# ---------------------------------------------------------------------------

def classify_instruction(instr: dict) -> tuple[str, str]:
    """Classify an instruction as testable or skipped.

    Returns:
        ("testable", "") or ("skipped", reason)
    """
    mnemonic = instr["mnemonic"]

    # 1. Load/store instructions need valid addresses -- skip.
    if instr.get("may_load", False):
        return ("skipped", "load instruction")
    if instr.get("may_store", False):
        return ("skipped", "store instruction")

    # 2. Branch/jump/call/return -- skip.
    if mnemonic in BRANCH_MNEMONICS:
        return ("skipped", "branch/jump instruction")

    # 3. Lock acquire/release -- skip.
    if mnemonic in LOCK_MNEMONICS:
        return ("skipped", "lock instruction")

    # 4. Stream switch instructions -- need live streams.
    if mnemonic in STREAM_MNEMONICS:
        return ("skipped", "stream instruction")

    # 5. NOPs and side-effect-only -- skip.
    if mnemonic in SIDE_EFFECT_MNEMONICS:
        return ("skipped", "no output (nop/side-effect)")

    operands = instr.get("operands", [])

    # 6. No operands at all -> no output.
    if not operands:
        return ("skipped", "no operands")

    # 7. Check each operand for unsupported types.
    for op in operands:
        op_type = op.get("operand_type", "unknown")
        if op_type == "composite_register":
            return ("skipped", "composite register operand")
        if op_type == "unknown":
            return ("skipped", "unknown operand type")
        if op_type == "register+16":
            return ("skipped", "register+16 operand type")
        if op_type == "register":
            kind = op.get("register_kind")
            if kind and kind not in KNOWN_REGISTER_KINDS:
                return ("skipped", f"unknown register kind: {kind}")

    # 8. Must have at least one register output.
    outputs = detect_output_operands(instr)
    if not outputs:
        return ("skipped", "no output operands detected")

    return ("testable", "")


def detect_output_operands(instr: dict) -> list[dict]:
    """Detect which operands are outputs (destinations).

    Since is_output is unreliable (always False in current data), we use
    the asm_string to determine output order: the first $-operand in the
    asm_string is the destination for ALU/vector instructions.

    For instructions with explicit 'dst' in the operand name, that is
    always an output regardless of position.

    Returns a list of operand dicts that are outputs.
    """
    operands = instr.get("operands", [])
    if not operands:
        return []

    asm_string = instr.get("asm_string", "")

    # Extract ordered operand names from asm_string.
    # Pattern: $name where name is alphanumeric + underscore.
    asm_op_names = re.findall(r'\$(\w+)', asm_string)
    if not asm_op_names:
        return []

    # Build a lookup from operand name to operand dict.
    op_by_name = {op["name"]: op for op in operands}

    # The first operand placeholder in asm_string is the destination,
    # IF it is a register (not an immediate).
    outputs = []
    first_name = asm_op_names[0]
    if first_name in op_by_name:
        first_op = op_by_name[first_name]
        if first_op.get("operand_type") == "register":
            outputs.append(first_op)

    # Also check for any operand explicitly named "dst".
    for op in operands:
        if op["name"] == "dst" and op not in outputs:
            if op.get("operand_type") == "register":
                outputs.append(op)

    return outputs


# ---------------------------------------------------------------------------
# Task 2: Register Name Mapping
# ---------------------------------------------------------------------------

def register_names(kind: str) -> list[str]:
    """Return a list of representative register names for a register kind.

    Reserves p0/p1 for buffer base pointers (input/output).
    """
    if kind == "scalar":
        # r0-r15 are general purpose.  Use a spread of values.
        return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r15"]

    if kind == "pointer":
        # p0, p1 reserved for input/output buffers.
        return ["p2", "p3", "p4", "p5"]

    if kind == "vector256":
        # wl0, wl2, wl4 ... (even indices for 256-bit halves).
        return ["wl0", "wl2", "wl4", "wl6"]

    if kind == "vector512":
        # x0, x2, x4 ... (512-bit registers).
        return ["x0", "x2", "x4"]

    if kind == "accumulator":
        # cm0, cm2, cm4 ... (accumulator registers).
        return ["cm0", "cm2", "cm4"]

    if kind == "control":
        # Control registers -- limited set.
        return ["r16", "r17"]

    if kind == "modifier_m":
        return ["m0", "m1"]

    if kind == "modifier_dj":
        return ["dj0", "dj1"]

    # Unknown kind -- return empty.
    return []


# ---------------------------------------------------------------------------
# Task 2: Immediate Value Generation
# ---------------------------------------------------------------------------

def immediate_values(bit_width: int, signed: bool = True) -> list[int]:
    """Generate boundary test values for an immediate field.

    Returns a deduplicated, sorted list of boundary values.
    """
    if signed:
        lo = -(1 << (bit_width - 1))
        hi = (1 << (bit_width - 1)) - 1
    else:
        lo = 0
        hi = (1 << bit_width) - 1

    # Boundary values: min, max, zero, one, minus-one (if signed).
    vals = {lo, hi, 0}
    if signed:
        vals.add(-1)
    if hi > 0:
        vals.add(1)

    # Filter to valid range.
    vals = {v for v in vals if lo <= v <= hi}

    return sorted(vals)


# ---------------------------------------------------------------------------
# Task 3: Assembly Test Point Generation
# ---------------------------------------------------------------------------

def _scalar_load(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate scalar load with pointer arithmetic for large offsets.

    lda offset range: [-128, 124] (6-bit signed, step 4).
    For offsets beyond this, copy base to p6 and use padda.
    """
    if 0 <= offset <= MAX_SCALAR_OFFSET:
        return [f"  lda {reg_name}, [{ptr}, #{offset}]"]
    # Copy base to scratch pointer p6, advance, then load.
    return [
        f"  mov p6, {ptr}",
        f"  padda [p6], #{offset}",
        f"  lda {reg_name}, [p6, #0]",
    ]


def _vector_load(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate vector load with pointer arithmetic for large offsets.

    vlda offset range: [-1024, 992] (6-bit signed, step 32).
    """
    if 0 <= offset <= MAX_VECTOR_OFFSET:
        return [f"  vlda {reg_name}, [{ptr}, #{offset}]"]
    return [
        f"  mov p6, {ptr}",
        f"  padda [p6], #{offset}",
        f"  vlda {reg_name}, [p6, #0]",
    ]


def _load_instruction(reg_name: str, kind: str, ptr: str, offset: int) -> list[str]:
    """Generate load instructions for a register from memory.

    Args:
        reg_name: Target register name (e.g., "r3", "wl2", "cm0").
        kind: Register kind (e.g., "scalar", "vector256").
        ptr: Pointer register for base address (e.g., "p0").
        offset: Byte offset from base.

    Returns:
        List of assembly lines.
    """
    if kind == "scalar":
        return _scalar_load(reg_name, ptr, offset)
    if kind == "pointer":
        return _scalar_load(reg_name, ptr, offset)
    if kind == "vector256":
        return _vector_load(reg_name, ptr, offset)
    if kind == "vector512":
        # x0 = {wl0, wh0} -- load low and high halves with SAME index.
        idx = reg_name[1:]  # "x2" -> "2"
        return (
            _vector_load(f"wl{idx}", ptr, offset)
            + _vector_load(f"wh{idx}", ptr, offset + 32)
        )
    if kind == "accumulator":
        # Load as vector256 then upshift.  For test scaffolding, load
        # zero into the accumulator by loading a vector and using vups.
        # This is complex -- for now, load the low half as vector256.
        # The actual accumulator content will be whatever vups produces.
        idx = reg_name[2:]  # "cm0" -> "0"
        return [
            f"  // load accumulator {reg_name}: load wl{idx} then ups",
        ] + _vector_load(f"wl{idx}", ptr, offset)
    if kind == "control":
        return _scalar_load(reg_name, ptr, offset)
    if kind in ("modifier_m", "modifier_dj"):
        return _scalar_load(reg_name, ptr, offset)
    return [f"  // unsupported load for kind={kind}"]


def _scalar_store(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate scalar store with pointer arithmetic for large offsets."""
    if 0 <= offset <= MAX_SCALAR_OFFSET:
        return [f"  st {reg_name}, [{ptr}, #{offset}]"]
    return [
        f"  mov p7, {ptr}",
        f"  padda [p7], #{offset}",
        f"  st {reg_name}, [p7, #0]",
    ]


def _vector_store(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate vector store with pointer arithmetic for large offsets."""
    if 0 <= offset <= MAX_VECTOR_OFFSET:
        return [f"  vst {reg_name}, [{ptr}, #{offset}]"]
    return [
        f"  mov p7, {ptr}",
        f"  padda [p7], #{offset}",
        f"  vst {reg_name}, [p7, #0]",
    ]


def _store_instruction(reg_name: str, kind: str, ptr: str, offset: int) -> list[str]:
    """Generate store instructions for a register to memory.

    Args:
        reg_name: Source register name.
        kind: Register kind.
        ptr: Pointer register for base address.
        offset: Byte offset from base.

    Returns:
        List of assembly lines.
    """
    if kind == "scalar":
        return _scalar_store(reg_name, ptr, offset)
    if kind == "pointer":
        # Store pointer as scalar -- mov to scalar first, then store.
        return [
            f"  mov r15, {reg_name}",
        ] + _scalar_store("r15", ptr, offset)
    if kind == "vector256":
        return _vector_store(reg_name, ptr, offset)
    if kind == "vector512":
        idx = reg_name[1:]  # "x2" -> "2"
        return (
            _vector_store(f"wl{idx}", ptr, offset)
            + _vector_store(f"wh{idx}", ptr, offset + 32)
        )
    if kind == "accumulator":
        # Use vsrs to shift down to vector256, then store.
        # vsrs needs a 512-bit accumulator half (bml/bmh), not the 1024-bit cm.
        # The shift operand is s0 (shift register), not a general scalar.
        # cm0 = {bml0, bmh0}.  We store the low half via bml.
        idx = reg_name[2:]  # "cm0" -> "0"
        return [
            f"  // store accumulator {reg_name}: srs bml{idx} to wl{idx} then vst",
            f"  vsrs.s16.s32 wl{idx}, bml{idx}, s0",
        ] + _nop_sled(5) + _vector_store(f"wl{idx}", ptr, offset)
    if kind == "control":
        return _scalar_store(reg_name, ptr, offset)
    if kind in ("modifier_m", "modifier_dj"):
        return _scalar_store(reg_name, ptr, offset)
    return [f"  // unsupported store for kind={kind}"]


def _nop_sled(count: int = 5) -> list[str]:
    """Generate a NOP sled for pipeline safety."""
    return [f"  nop" for _ in range(count)]


def _substitute_asm(asm_string: str, regs: dict[str, str]) -> str:
    """Substitute operand placeholders in asm_string with register names.

    The asm_string looks like "add\\t$mRx, $mRx0, $mRy".
    We replace each $name with the corresponding value from regs.
    We also replace \\t with a real tab.
    """
    result = asm_string
    # Sort by length descending to avoid partial substitution
    # (e.g., $mRx before $mRx0 would be wrong -- but $mRx0 before $mRx is fine).
    for name in sorted(regs.keys(), key=len, reverse=True):
        result = result.replace(f"${name}", regs[name])
    return result


def generate_test_point(
    instr: dict,
    regs: dict[str, str],
    in_offset: int,
    out_offset: int,
) -> str:
    """Generate an assembly test point for a single instruction.

    Args:
        instr: Instruction dict from aie2-isa.json.
        regs: Mapping from operand name to concrete register/value string.
        in_offset: Byte offset into input buffer (p0) for loading inputs.
        out_offset: Byte offset into output buffer (p1) for storing outputs.

    Returns:
        Assembly string for the test point (multiple lines).
    """
    lines = []
    name = instr["name"]
    lines.append(f"  // ---- test: {name} ----")

    # Build lookup: operand name -> operand dict.
    op_by_name = {op["name"]: op for op in instr.get("operands", [])}

    # Detect outputs.
    outputs = detect_output_operands(instr)
    output_names = {op["name"] for op in outputs}

    # Determine inputs: all operands that are NOT outputs.
    # Separate register inputs from immediate inputs.
    asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))

    # Load input registers from [p0, #offset].
    cur_in_offset = in_offset
    for op_name in asm_op_names:
        if op_name in output_names:
            continue
        if op_name not in op_by_name:
            continue
        op = op_by_name[op_name]
        if op["operand_type"] != "register":
            continue
        kind = op.get("register_kind", "")
        if kind not in KNOWN_REGISTER_KINDS:
            continue
        reg = regs.get(op_name, "r0")
        load_lines = _load_instruction(reg, kind, "p0", cur_in_offset)
        lines.extend(load_lines)
        cur_in_offset += REGISTER_SIZES.get(kind, 4)

    # NOP sled before instruction.
    lines.extend(_nop_sled(5))

    # The instruction itself.
    asm_line = "  " + _substitute_asm(instr["asm_string"], regs)
    lines.append(asm_line)

    # NOP sled after instruction.
    lines.extend(_nop_sled(5))

    # Store output registers to [p1, #offset].
    cur_out_offset = out_offset
    for op in outputs:
        kind = op.get("register_kind", "")
        reg = regs.get(op["name"], "r0")
        store_lines = _store_instruction(reg, kind, "p1", cur_out_offset)
        lines.extend(store_lines)
        cur_out_offset += REGISTER_SIZES.get(kind, 4)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 3: Mega-program Assembly
# ---------------------------------------------------------------------------

def build_mega_program(test_points: list[str]) -> str:
    """Build a complete .s file from a list of test point assembly strings.

    Args:
        test_points: List of assembly strings from generate_test_point().

    Returns:
        Complete assembly program as a string.
    """
    lines = []
    lines.append("// Auto-generated by isa-test-gen.py")
    lines.append("// Do not edit manually.")
    lines.append("")
    lines.append(".text")
    lines.append(".globl test_kernel")
    lines.append("test_kernel:")
    lines.append("")

    for tp in test_points:
        lines.append(tp)

    # Return sequence: ret lr + NOP sled.
    lines.append("  ret lr")
    lines.extend([f"  nop" for _ in range(4)])
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 4: Operand Combination Generation
# ---------------------------------------------------------------------------

def generate_operand_combos(instr: dict) -> list[dict[str, str]]:
    """Generate multiple operand assignments for an instruction.

    For each operand, picks a default register/value, then varies one
    operand at a time through its alternatives.  This produces a baseline
    combo plus N combos (one per operand variation).

    Returns:
        List of dicts mapping operand name -> concrete register/value string.
    """
    operands = instr.get("operands", [])
    asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))
    op_by_name = {op["name"]: op for op in operands}
    outputs = detect_output_operands(instr)
    output_names = {op["name"] for op in outputs}

    # Build default assignment and per-operand alternatives.
    defaults: dict[str, str] = {}
    alternatives: dict[str, list[str]] = {}

    for op_name in asm_op_names:
        if op_name not in op_by_name:
            continue
        op = op_by_name[op_name]
        op_type = op.get("operand_type", "")

        if op_type == "register":
            kind = op.get("register_kind", "")
            names = register_names(kind)
            if not names:
                defaults[op_name] = "r0"
                alternatives[op_name] = []
                continue
            defaults[op_name] = names[0]
            alternatives[op_name] = names[1:]

        elif op_type == "immediate":
            bit_width = op.get("bit_width", 8)
            signed = op.get("signed", True)
            vals = immediate_values(bit_width, signed)
            if not vals:
                defaults[op_name] = "0"
                alternatives[op_name] = []
                continue
            defaults[op_name] = str(vals[0])
            alternatives[op_name] = [str(v) for v in vals[1:]]

        else:
            defaults[op_name] = "0"
            alternatives[op_name] = []

    # Baseline combo: all defaults.
    combos = [dict(defaults)]

    # One-at-a-time variations: for each operand, try each alternative
    # while keeping all others at default.
    for op_name, alts in alternatives.items():
        for alt_val in alts:
            combo = dict(defaults)
            combo[op_name] = alt_val
            combos.append(combo)

    return combos


# ---------------------------------------------------------------------------
# Task 4: Test Point Metadata
# ---------------------------------------------------------------------------

def _compute_input_size(instr: dict, regs: dict[str, str]) -> int:
    """Compute total input buffer size for a test point."""
    operands = instr.get("operands", [])
    op_by_name = {op["name"]: op for op in operands}
    outputs = detect_output_operands(instr)
    output_names = {op["name"] for op in outputs}
    asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))

    total = 0
    for op_name in asm_op_names:
        if op_name in output_names:
            continue
        if op_name not in op_by_name:
            continue
        op = op_by_name[op_name]
        if op["operand_type"] != "register":
            continue
        kind = op.get("register_kind", "")
        if kind not in KNOWN_REGISTER_KINDS:
            continue
        total += REGISTER_SIZES.get(kind, 4)
    return max(4, total)


def _compute_output_size(instr: dict) -> int:
    """Compute total output buffer size for a test point."""
    outputs = detect_output_operands(instr)
    total = 0
    for op in outputs:
        kind = op.get("register_kind", "")
        total += REGISTER_SIZES.get(kind, 4)
    return max(4, total)


# ---------------------------------------------------------------------------
# Task 4: Full Pipeline
# ---------------------------------------------------------------------------

# Maximum test points per batch.  16KB program memory / 16-byte VLIW bundle
# gives 1024 bundles.  Each test point uses ~15 bundles (5 NOP + 1 instr +
# 5 NOP + ~2 load + ~2 store).  Conservative limit: 70.
MAX_POINTS_PER_BATCH = 70


def generate_all(isa_json_path: str, out_dir: str) -> dict:
    """Full pipeline: classify, generate combos, batch, write .s + manifest.

    Args:
        isa_json_path: Path to aie2-isa.json.
        out_dir: Directory to write batch_NNN.s files and manifest.json.

    Returns:
        Summary dict with counts and batch info.
    """
    with open(isa_json_path) as f:
        isa_data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    # Flatten across slots.
    all_instrs = []
    for slot, instrs in isa_data.items():
        for instr in instrs:
            all_instrs.append(instr)

    # Classify and generate combos.
    testable_count = 0
    skipped_count = 0
    skip_reasons: dict[str, int] = {}
    test_points_meta: list[dict] = []  # per-test-point metadata
    test_points_asm: list[str] = []    # generated assembly strings

    # Track running offsets across all test points (reset per batch later).
    global_in_offset = 0
    global_out_offset = 0

    for instr in all_instrs:
        status, reason = classify_instruction(instr)
        if status != "testable":
            skipped_count += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        testable_count += 1
        combos = generate_operand_combos(instr)

        for combo_idx, regs in enumerate(combos):
            in_size = _compute_input_size(instr, regs)
            out_size = _compute_output_size(instr)

            asm = generate_test_point(
                instr, regs,
                in_offset=global_in_offset,
                out_offset=global_out_offset,
            )
            test_points_asm.append(asm)
            test_points_meta.append({
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": regs,
                "in_offset": global_in_offset,
                "in_size": in_size,
                "out_offset": global_out_offset,
                "out_size": out_size,
            })

            global_in_offset += in_size
            global_out_offset += out_size

    # Batch test points into mega-programs.
    batches = []
    batch_idx = 0
    i = 0
    while i < len(test_points_asm):
        end = min(i + MAX_POINTS_PER_BATCH, len(test_points_asm))
        batch_asm = test_points_asm[i:end]
        batch_meta = test_points_meta[i:end]

        # Compute batch-level buffer sizes from constituent test points.
        batch_in_size = 0
        batch_out_size = 0
        for m in batch_meta:
            batch_in_size = max(batch_in_size, m["in_offset"] + m["in_size"])
            batch_out_size = max(batch_out_size, m["out_offset"] + m["out_size"])

        # Write .s file.
        filename = f"batch_{batch_idx:03d}.s"
        filepath = os.path.join(out_dir, filename)
        program = build_mega_program(batch_asm)
        with open(filepath, "w") as f:
            f.write(program)

        batches.append({
            "batch_index": batch_idx,
            "filename": filename,
            "test_count": len(batch_asm),
            "in_size": batch_in_size,
            "out_size": batch_out_size,
            "tests": batch_meta,
        })

        batch_idx += 1
        i = end

    # Write manifest.json.
    manifest = {
        "testable_instructions": testable_count,
        "skipped_instructions": skipped_count,
        "total_test_points": len(test_points_asm),
        "total_batches": len(batches),
        "skip_reasons": skip_reasons,
        "batches": batches,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for ISA test generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate assembly test programs from aie2-isa.json",
    )
    parser.add_argument(
        "--isa-json",
        default=os.path.join(os.path.dirname(__file__), "aie2-isa.json"),
        help="Path to aie2-isa.json (default: tools/aie2-isa.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="build/isa-tests",
        help="Output directory for .s files and manifest (default: build/isa-tests)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print classification summary only (no file generation)",
    )
    args = parser.parse_args()

    if args.summary:
        with open(args.isa_json) as f:
            isa_data = json.load(f)

        testable_count = 0
        skipped_count = 0
        skip_reasons: dict[str, int] = {}
        testable_instrs: list[dict] = []

        for slot, instrs in isa_data.items():
            for instr in instrs:
                status, reason = classify_instruction(instr)
                if status == "testable":
                    testable_count += 1
                    testable_instrs.append(instr)
                else:
                    skipped_count += 1
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        print(f"Testable: {testable_count}")
        print(f"Skipped:  {skipped_count}")
        print(f"Total:    {testable_count + skipped_count}")
        print()
        print("Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
        print()
        print("Testable instructions:")
        for instr in testable_instrs:
            outputs = detect_output_operands(instr)
            out_names = [o["name"] for o in outputs]
            print(f"  {instr['name']:40s} [{instr['slot']:4s}] outputs={out_names}")
        return

    manifest = generate_all(args.isa_json, args.out_dir)

    print(f"Testable instructions: {manifest['testable_instructions']}")
    print(f"Skipped instructions:  {manifest['skipped_instructions']}")
    print(f"Total test points:     {manifest['total_test_points']}")
    print(f"Batches:               {manifest['total_batches']}")
    print(f"Output:                {args.out_dir}")
    print()
    print("Skip reasons:")
    for reason, count in sorted(manifest["skip_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
