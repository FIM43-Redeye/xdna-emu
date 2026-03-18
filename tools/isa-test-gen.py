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

# Mnemonic substrings for conversion load/store variants that llvm-mc
# cannot assemble (they combine memory access with data conversion in
# a single instruction with no direct assembly syntax).
CONVERSION_SUFFIXES = (".ups.", ".srs.", ".conv.", ".pack.", ".unpack.")

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
    # quad: 128-byte (1024-bit) quadword vector registers q0-q3.
    # These appear in DMV_Q load/store instructions as "accumulator" bw=2
    # in the ISA JSON (exporter misclassification); we promote them here.
    "quad",
})

# Operand types that are safe to handle.
SAFE_OPERAND_TYPES = frozenset({
    "register", "immediate",
})

# Operand types that are equivalent to "register" with an encoding offset.
# "register+16" maps to eRS8 (r16-r23) or eL (register pairs) in the ISA.
REGISTER_LIKE_TYPES = frozenset({"register", "register+16"})

# Composite register kinds that map cleanly to a concrete register class.
# These appear as operand_type="composite_register" in the ISA JSON because
# their hardware encoding is non-trivial, but functionally they are just a
# subset of a standard register class that the assembler can name directly.
#
# Derivation (from AIE2GenRegisterInfo.td and AIE2Disassembler.cpp):
#   mShflDst = AIE2Vector512RegisterClass(add mXm, mBMSm)
#     -- broadcast-shuffle destination: x* (vector512) or bml*/bmh* (acc-half).
#     For test generation we use the x* subset (simplest concrete names).
#   mWm_1 = AIE2Vector256RegisterClass(add mWm)
#     -- vector256 with rearranged encoding (wl0/wh0/wl1/... non-monotonic).
#     The assembler accepts the same wl*/wh* names as for plain vector256.
#   mMvSclSrc / MvSclSrc (bw=7, 128 entries)
#     -- source operand class for mv-slot scalar moves (MOV_mv_scl, ADD_NC,
#     MOVXM, VEXTRACT).  Encodes scalars, pointers, shift regs, modifiers, and
#     control regs.  For testing we use the plain scalar (r*) subset.
#   LdaCg (bw=7, 128 entries)
#     -- constant-generator destination in the lda slot (MOVA_lda_cg).
#     Encodes the same register space as MvSclSrc.  Map to scalar for testing.
#   AluCg (bw=6, 64 entries)
#     -- constant-generator destination in the alu slot (MOVX_alu_cg).
#     Subset of the scalar register space.  Map to scalar for testing.
#   ERS4 (bw=2, 4 entries: r24-r27)
#     -- narrow scalar subclass used as an index operand in VEXTRACT.
#     Concrete names: r24, r25, r26, r27.
#   MvBMXDst (bw=6)
#     -- vector/accumulator destination for cascade-read vmov (VMOV_mv_scd,
#     VMOV_mv_x, VMOV_mv_mcd).  Encodes x* (vector512) or bml*/bmh* (acc-half).
#     Map to vector512 for testing.
#   MvBMXSrc (bw=9)
#     -- wide vector/accumulator source for vmov (VMOV_mv_x).
#     Map to vector512 for testing.
#   MvAMWQDst (bw=7)
#     -- 256-bit vector destination for wmov (VMOV_mv_w).
#     Map to vector256 for testing.
#   MvAMWQSrc (bw=9)
#     -- wide 256-bit vector source for wmov (VMOV_mv_w).
#     Map to vector256 for testing.
#
# Value: the effective register kind to use for load/store/naming purposes.
TESTABLE_COMPOSITE_KINDS: dict[str, str] = {
    "ShflDst": "vector512",
    "Wm1": "vector256",
    # Scalar-family composite kinds.
    "MvSclSrc": "scalar",
    "LdaCg": "scalar",
    "AluCg": "scalar",
    # Narrow scalar subclass (r24-r27); marked as scalar so load/store work.
    "ERS4": "scalar",
    # Vector-family composite kinds.
    "MvBMXDst": "vector512",
    "MvBMXSrc": "vector512",
    "MvAMWQDst": "vector256",
    "MvAMWQSrc": "vector256",
    # DMS scalar load/store class (mLdaScl / mSclSt).
    # AIE2ScalarRegisterClass(add eP, eR, eDC, eDJ, eDN, eM, lr): spans
    # pointers, general scalars, loop counters, DJ indices, modifiers, LR.
    # We use the eR (r*) subset as the simplest concrete names; the
    # assembler accepts r* names for both mLdaScl and mSclSt operands.
    # Source: AIE2GenRegisterInfo.td mLdaScl / mSclSt definitions.
    "LdaScl": "scalar",
}


# Operand names that can carry a cm-class (1024-bit full accumulator) encoding.
# These are the names that appear in VUPS x2c, VSRS x_srs, VMOV_mv_cm, and
# VNEG where the ISA exporter emits operand_type="unknown" + bit_width=4
# instead of operand_type="register" + register_kind="accumulator".
#
# Heuristic derivation: the cm register class (eCM/mCMm in TableGen) uses a
# 4-bit field encoding cm0..cm7 (8 entries).  Every non-dontcare operand with
# operand_type="unknown" and bit_width=4 that appears with one of these names
# in a vector instruction is a cm-class register.  The 4-bit width distinguishes
# cm (4 bits) from bm-class (5 bits) and am-class (6 bits).
#
# This is a workaround for a known limitation of the Rust ISA exporter: it
# cannot yet tag cm-class operands correctly.  The proper fix is to add cm
# register class recognition to the OperandInfo export in src/tablegen/.
_CM_CLASS_OPERAND_NAMES = frozenset({"dst", "src", "acc1", "acc2"})


def _resolve_unknown_operand(op: dict) -> dict:
    """Attempt to resolve an unknown-typed operand to a concrete register class.

    Workaround: the ISA exporter does not yet tag cm-class (1024-bit full
    accumulator, eCM/mCMm TableGen class) operands correctly.  They appear
    with operand_type="unknown", register_kind=None, bit_width=4 instead of
    operand_type="register", register_kind="accumulator", bit_width=4.

    When an operand matches this signature AND its name is one of the known
    cm-class operand names (dst, src, acc1, acc2), return a corrected copy
    with operand_type="register" and register_kind="accumulator".

    All other unknown operands are returned unchanged.
    """
    if op.get("operand_type") != "unknown":
        return op
    if op.get("register_kind"):
        return op
    if op.get("bit_width") != 4:
        return op
    if op.get("name", "") not in _CM_CLASS_OPERAND_NAMES:
        return op
    # Matches the cm-class heuristic: treat as accumulator register.
    resolved = dict(op)
    resolved["operand_type"] = "register"
    resolved["register_kind"] = "accumulator"
    return resolved


def _effective_kind(op: dict) -> str:
    """Return the effective register kind for an operand.

    For plain register operands, returns register_kind directly.
    For composite_register operands whose kind is in TESTABLE_COMPOSITE_KINDS,
    returns the mapped concrete kind (e.g., "ShflDst" -> "vector512").
    For unknown operands that match the cm-class heuristic (see
    _resolve_unknown_operand), returns "accumulator".
    Returns "" for all other operand types.
    """
    # Apply cm-class resolution before the type dispatch below.
    op = _resolve_unknown_operand(op)
    op_type = op.get("operand_type", "")
    kind = op.get("register_kind", "") or ""
    if op_type in REGISTER_LIKE_TYPES:
        return kind
    if op_type == "composite_register":
        return TESTABLE_COMPOSITE_KINDS.get(kind, "")
    return ""

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
    # quad: 128-byte (1024-bit) quadword vector registers q0-q3.
    # mQQa = AIE2Vector128RegisterClass(add q0, q1, q2, q3) per AIE2GenRegisterInfo.td.
    "quad": 128,
}

# Maximum positive immediate offset for lda/st (6-bit signed, step 4).
# Range: [-128, 124].  We use only positive offsets in generated code.
MAX_SCALAR_OFFSET = 124

# Maximum positive immediate offset for vlda/vst (6-bit signed, step 32).
# Range: [-1024, 992].
MAX_VECTOR_OFFSET = 992

# Load pipeline latency: cycles from lda/vlda issue to register availability.
# All lda/vlda variants have latency 7 in AIE2Schedule.td.
LOAD_LATENCY = 7

# Default result latency when sched_class is not in the latency map.
DEFAULT_RESULT_LATENCY = 7

# Minimum result latency enforced regardless of scheduling model.
# The scheduling model's II_ values represent initiation intervals
# (pipeline occupation), NOT actual write-back latencies.  Multi-cycle
# operations like DIVS report II=1 but need many cycles to commit.
# Since this is a test harness (correctness, not performance), we use
# a generous minimum to guarantee the result is committed before we
# store it.  The store sequence itself adds ~3 more cycles (mov + padda),
# so the effective delay is MIN_RESULT_LATENCY + ~3.
MIN_RESULT_LATENCY = 5

# Scheduling model latencies: sched_class -> result latency in cycles.
# Loaded from aie2-sched-latencies.json (extracted from AIE2Schedule.td).
_SCHED_LATENCIES: dict[str, int] = {}


def _load_sched_latencies() -> dict[str, int]:
    """Load scheduling latencies from JSON, caching the result."""
    global _SCHED_LATENCIES
    if _SCHED_LATENCIES:
        return _SCHED_LATENCIES
    lat_path = os.path.join(os.path.dirname(__file__), "aie2-sched-latencies.json")
    if os.path.exists(lat_path):
        with open(lat_path) as f:
            _SCHED_LATENCIES = json.load(f)
    return _SCHED_LATENCIES


def result_latency(instr: dict) -> int:
    """Get the result latency for an instruction from the scheduling model.

    Returns the number of NOP cycles to insert between instruction issue
    and result store, ensuring the pipeline has committed the result.
    Uses the scheduling model where available, with a floor of
    MIN_RESULT_LATENCY to guard against incomplete latency data.
    """
    sched_class = instr.get("sched_class", "")
    lats = _load_sched_latencies()
    model_latency = lats.get(sched_class, DEFAULT_RESULT_LATENCY)
    return max(model_latency, MIN_RESULT_LATENCY)


# ---------------------------------------------------------------------------
# Strategy Interface
# ---------------------------------------------------------------------------

class TestStrategy:
    """Base class for instruction test strategies."""

    def can_test(self, instr: dict) -> tuple[bool, str]:
        """Return (True, "") if this strategy can test the instruction,
        or (False, skip_reason) if not."""
        raise NotImplementedError

    def generate_test_point(self, instr: dict, regs: dict[str, str],
                            in_offset: int, out_offset: int) -> str:
        """Generate assembly for one test point."""
        raise NotImplementedError

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Generate operand combos for this instruction.
        Default delegates to the shared generate_operand_combos()."""
        return generate_operand_combos(instr)

    def compute_input_size(self, instr: dict, regs: dict[str, str]) -> int:
        """Compute total input buffer size for a test point."""
        return _compute_input_size(instr, regs)

    def compute_output_size(self, instr: dict) -> int:
        """Compute total output buffer size for a test point."""
        return _compute_output_size(instr)


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
    #
    # Exception: some instructions are tagged may_load in TableGen but have no
    # pointer operand -- they are register-to-register shuffles that use load
    # slot resources but don't actually access memory (e.g. vldb.4x16.hi,
    # vldb.4x32.lo, vldb.4x64.hi).  Without a pointer they cannot reach memory,
    # so we treat them as compute instructions and fall through to the normal
    # classify path.
    operands = instr.get("operands", [])
    has_pointer = any(
        op.get("register_kind") == "pointer" for op in operands
    )
    if instr.get("may_load", False) and has_pointer:
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

    # 5b. Hardware counter/stream reads hang without active hardware.
    asm = instr.get("asm_string", "")
    if "cntr" in asm:
        return ("skipped", "hardware counter (register pair)")
    if "SS" in asm and "mov" in mnemonic:
        return ("skipped", "stream switch status read (hangs without streams)")
    # 5c. Cascade read/write instructions need active cascade connections.
    # SCD = stream cascade data (read from cascade), MCD = main cascade data
    # (write to cascade).  Without live cascade setup these stall.
    if "SCD" in asm or "MCD" in asm:
        return ("skipped", "cascade instruction (needs active cascade)")

    operands = instr.get("operands", [])

    # 6. No operands at all -> no output.
    if not operands:
        return ("skipped", "no operands")

    # Resolve cm-class unknowns before all subsequent checks.  The ISA
    # exporter emits operand_type="unknown" for cm-class (1024-bit full
    # accumulator, 4-bit encoding) operands.  _resolve_unknown_operand
    # corrects this for operands named dst/src/acc1/acc2 with bit_width=4.
    operands = [_resolve_unknown_operand(op) for op in operands]

    # 7. Check each operand for unsupported types.
    for op in operands:
        op_type = op.get("operand_type", "unknown")
        if op_type == "composite_register":
            kind = op.get("register_kind", "")
            if kind not in TESTABLE_COMPOSITE_KINDS:
                return ("skipped", "composite register operand")
        if op_type == "unknown":
            if not op.get("name", "").startswith("dontcare"):
                return ("skipped", "unknown operand type")
        if op_type in REGISTER_LIKE_TYPES:
            kind = op.get("register_kind")
            if kind and kind not in KNOWN_REGISTER_KINDS:
                return ("skipped", f"unknown register kind: {kind}")

    # 8. Detect outputs.  Control register destinations are testable: after the
    # instruction executes, the control register value is read back into a
    # general-purpose scalar via 'mov r14, <ctrl_reg>' (MOV_mv_scl, mv slot),
    # then stored to the output buffer.
    outputs = detect_output_operands(instr)

    # 9. Must have at least one register output.
    if not outputs:
        return ("skipped", "no output operands detected")

    # 10. Skip instructions with unusual accumulator subclasses:
    #   - bit_width=2: composite/sparse register classes (mQQXw) misclassified
    #     by the exporter.  Real accumulators use 4 bits (cm) or 5 bits (bm).
    #   - bit_width=6: 256-bit accumulator quarters (amll/amlh/amhl/amhh).
    #     These can't be loaded/stored via vsrs and only appear in VFLOOR.
    for op in operands:
        if op.get("register_kind") == "accumulator":
            bw = op.get("bit_width", 0)
            if bw == 2:
                return ("skipped", "composite sparse register (bw=2)")
            if bw == 6:
                return ("skipped", "accumulator quarter register (bw=6)")

    return ("testable", "")


def detect_output_operands(instr: dict) -> list[dict]:
    """Detect which operands are outputs (destinations).

    Since is_output is unreliable (always False in current data), we use
    the asm_string to determine output order: the first $-operand in the
    asm_string is the destination for ALU/vector instructions.

    For instructions with explicit 'dst' in the operand name, that is
    always an output regardless of position.

    Tied-destination fallback (VADDMAC/VADDMSC/VSUBMAC/VSUBMSC family):
    Some instructions show $dst first in the asm_string but have NO operand
    named "dst".  In these instructions $dst is a write alias for one of the
    accumulator inputs (the operation accumulates in-place).  The hardware
    convention is that the first accumulator appearing after $dst in the
    asm_string order is the tied register: it is both read as input AND
    written as output.  When this pattern is detected we return that
    accumulator as the output so the test harness can load it before the
    instruction and store it afterward to capture the result.

    Returns a list of operand dicts that are outputs.
    """
    operands = instr.get("operands", [])
    if not operands:
        return []

    # Resolve cm-class unknowns so that dst/src operands with the cm-class
    # signature are treated as register operands below.
    operands = [_resolve_unknown_operand(op) for op in operands]

    asm_string = instr.get("asm_string", "")

    # Extract ordered operand names from asm_string.
    # Pattern: $name where name is alphanumeric + underscore.
    asm_op_names = re.findall(r'\$(\w+)', asm_string)
    if not asm_op_names:
        return []

    # Build a lookup from operand name to resolved operand dict.
    op_by_name = {op["name"]: op for op in operands}

    # The first operand placeholder in asm_string is the destination,
    # IF it is a register (not an immediate).
    outputs = []
    first_name = asm_op_names[0]
    if first_name in op_by_name:
        first_op = op_by_name[first_name]
        op_type = first_op.get("operand_type", "")
        is_register_like = op_type in REGISTER_LIKE_TYPES
        is_testable_composite = (
            op_type == "composite_register"
            and first_op.get("register_kind", "") in TESTABLE_COMPOSITE_KINDS
        )
        if is_register_like or is_testable_composite:
            outputs.append(first_op)
    elif first_name == "dst":
        # Tied-destination pattern: $dst appears first in asm_string but no
        # operand named "dst" exists in the operands list.  The write
        # destination is tied to the first accumulator operand that follows
        # $dst in asm_string order.  That register serves as both input
        # (pre-loaded) and output (stored after execution), implementing
        # in-place accumulation.
        for name in asm_op_names[1:]:
            if name not in op_by_name:
                continue
            candidate = op_by_name[name]
            if (candidate.get("operand_type") in REGISTER_LIKE_TYPES
                    and candidate.get("register_kind") == "accumulator"):
                outputs.append(candidate)
                break

    # Also check for any operand explicitly named "dst" or "d".
    for op in operands:
        if op["name"] in ("dst", "d") and op not in outputs:
            op_type = op.get("operand_type", "")
            is_register_like = op_type in REGISTER_LIKE_TYPES
            is_testable_composite = (
                op_type == "composite_register"
                and op.get("register_kind", "") in TESTABLE_COMPOSITE_KINDS
            )
            if is_register_like or is_testable_composite:
                outputs.append(op)

    return outputs


# ---------------------------------------------------------------------------
# Task 2: Register Name Mapping
# ---------------------------------------------------------------------------

def _needs_register_pair(instr_name: str) -> bool:
    """Check if a register+16 operand needs a register pair (eL) vs single (eRS8).

    8-bit element and 64-bit element variants use eL (64-bit register pairs)
    because the comparison bitmask or data value exceeds 32 bits.
    The vec_size=0b00 (8-bit) encoding uses eL in the TableGen definitions.
    """
    name = instr_name.upper()
    # Explicit 8-bit element variants: _D8, _S8, _8, or trailing 8
    # (e.g., VNEG_GTZ8, VEQZ_8, VABS_GTZ_D8)
    if name.endswith(("_D8", "_S8", "_8", "8")):
        # Exclude names ending in 16/32/BF16 etc.
        if not name.endswith(("16", "32", "BF16")):
            return True
    # 64-bit element variants
    if name.endswith("_64"):
        return True
    # Special cases
    if name in ("MOV_CNTR",):
        return True
    return False


def register_names(kind: str, bit_width: int = 0,
                   operand_type: str = "register",
                   instr_name: str = "") -> list[str]:
    """Return a list of representative register names for a register kind.

    Uses bit_width and operand_type to distinguish register subclasses:
      - accumulator, 4 bits -> cm0-cm8 (1024-bit full accumulators)
      - accumulator, 5 bits -> bml0-bml8, bmh0-bmh8 (512-bit halves)
      - accumulator, 6 bits -> amll0, amlh0, amhl0, amhh0 (256-bit quarters)
      - scalar, 2 bits -> s0-s3 (shift registers)
      - scalar, 5 bits -> r0-r31 (general purpose)
      - register+16, scalar, 3 bits -> r16-r23 (eRS8 high scalar class)

    Reserves p0/p1 for buffer base pointers (input/output).
    """
    if kind == "scalar":
        if operand_type == "register+16":
            if _needs_register_pair(instr_name):
                # eL register pair class: l1 = r19:r18, l2 = r21:r20, etc.
                # Avoid l0 (r17:r16) -- r16 is callee-saved in the
                # mlir-aie runtime and clobbering it can deadlock DMA.
                return ["r19:r18", "r21:r20", "r23:r22",
                        "r25:r24", "r27:r26", "r29:r28", "r31:r30"]
            # eRS8 high scalar class.  Avoid r16 (callee-saved by
            # mlir-aie runtime, clobbering causes lock release failure).
            return ["r17", "r18", "r19", "r20", "r21", "r22", "r23"]
        if bit_width == 2:
            # Shift registers (mSm/mSs class).
            # s3 is reserved for UPS/SRS infrastructure (accumulator
            # load/store uses it as the shift amount).
            return ["s0", "s1", "s2"]
        # General purpose scalars (eR class).
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
        if bit_width == 2:
            # DMV_Q quad registers: mQQa = AIE2Vector128RegisterClass(add q0,q1,q2,q3).
            # The ISA exporter misclassifies these as "accumulator" with bw=2 because
            # the 2-bit encoding maps to 4 entries.  The real names are q0-q3.
            # Source: AIE2GenRegisterInfo.td eQQEs / eQQOs / mQQa definitions.
            return ["q0", "q1", "q2", "q3"]
        if bit_width == 5:
            # 512-bit accumulator halves (mBMa/mBMm class).
            return ["bml0", "bml2", "bmh0", "bmh2"]
        if bit_width == 6:
            # 256-bit accumulator quarters (mAMs class).
            return ["amll0", "amlh0", "amhl0", "amhh0"]
        # Default: 1024-bit full accumulators (eCM/mCMm class, 4-bit encoding).
        return ["cm0", "cm2", "cm4"]

    if kind == "control":
        # Control registers (crSat, crRnd, etc.) -- not general r16/r17.
        return ["crSat", "crRnd"]

    if kind == "modifier_m":
        return ["m0", "m1"]

    if kind == "modifier_dj":
        return ["dj0", "dj1"]

    # Testable composite kinds: mapped to concrete register names.
    # ShflDst (mShflDst) is AIE2Vector512RegisterClass -- use x* names.
    # The composite encoding allows either x* or bml*/bmh*, but x* is the
    # simplest subset and is what the Peano compiler generates.
    if kind == "ShflDst":
        return ["x0", "x2", "x4"]

    # Wm1 (mWm_1) is AIE2Vector256RegisterClass with non-monotonic encoding.
    # The assembler accepts the same wl*/wh* names as for plain vector256.
    if kind == "Wm1":
        return ["wl0", "wl2", "wl4", "wl6"]

    # MvSclSrc (mMvSclSrc) is the mv-slot scalar source class.
    # Encodes scalars, pointers, shift regs, modifiers, and control regs.
    # For testing we use plain general-purpose scalars (r0-r7).
    if kind == "MvSclSrc":
        return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]

    # LdaCg: constant-generator destination in the lda slot.
    # Encodes the same register space as MvSclSrc; map to scalar for testing.
    if kind == "LdaCg":
        return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]

    # AluCg: constant-generator destination in the alu slot.
    # Subset of the scalar register space; map to scalar for testing.
    if kind == "AluCg":
        return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"]

    # ERS4: narrow scalar subclass r24-r27, used as index in VEXTRACT.
    # These are the only 4 registers in this class (2-bit encoding).
    if kind == "ERS4":
        return ["r24", "r25", "r26", "r27"]

    # MvBMXDst / MvBMXSrc: vector/accumulator registers for cascade vmov.
    # Encoding covers x* (vector512) and bml*/bmh* (acc-half).
    # Use x* names as the simplest concrete subset.
    if kind in ("MvBMXDst", "MvBMXSrc"):
        return ["x0", "x2", "x4"]

    # MvAMWQDst / MvAMWQSrc: 256-bit vector registers for wmov.
    # Use the same wl* names as plain vector256.
    if kind in ("MvAMWQDst", "MvAMWQSrc"):
        return ["wl0", "wl2", "wl4", "wl6"]

    # LdaScl: DMS scalar load/store composite class (mLdaScl / mSclSt).
    # Encodes eR, eP, eDC, eDJ, eDN, eM, and lr -- we use eR (r*) subset.
    if kind == "LdaScl":
        return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r15"]

    # quad: 128-byte (1024-bit) quadword registers q0-q3.
    # mQQa = AIE2Vector128RegisterClass(add eQQEs, eQQOs) per AIE2GenRegisterInfo.td.
    # eQQEs = {q0, q2}, eQQOs = {q1, q3}.
    if kind == "quad":
        return ["q0", "q1", "q2", "q3"]

    # Unknown kind -- return empty.
    return []


# ---------------------------------------------------------------------------
# Task 2: Immediate Value Generation
# ---------------------------------------------------------------------------

def immediate_values(bit_width: int, signed: bool = True,
                     scale: int = 1) -> list[int]:
    """Generate boundary test values for an immediate field.

    Args:
        bit_width: Number of bits in the field (before scaling).
        signed: Whether the field is signed.
        scale: Scale factor -- values must be multiples of this.  For
            example, PADDB has scale=4 (immediates must be multiples of 4).

    Returns a deduplicated, sorted list of boundary values (already scaled).
    """
    if scale is None or scale < 1:
        scale = 1

    if signed:
        lo = -(1 << (bit_width - 1))
        hi = (1 << (bit_width - 1)) - 1
    else:
        lo = 0
        hi = (1 << bit_width) - 1

    # Apply scale: raw field values are multiplied by scale to get the
    # actual immediate value accepted by the assembler.
    lo_scaled = lo * scale
    hi_scaled = hi * scale

    # Boundary values: min, max, zero, one step, minus-one step (if signed).
    vals = {lo_scaled, hi_scaled, 0}
    if signed:
        vals.add(-scale)
    if hi > 0:
        vals.add(scale)

    # Filter to valid range.
    vals = {v for v in vals if lo_scaled <= v <= hi_scaled}

    return sorted(vals)


# ---------------------------------------------------------------------------
# Task 3: Assembly Test Point Generation
# ---------------------------------------------------------------------------

def _align(offset: int, alignment: int) -> int:
    """Round up offset to the next multiple of alignment."""
    return (offset + alignment - 1) & ~(alignment - 1)


def _is_sp_relative(instr: dict) -> bool:
    """Return True if the instruction uses SP-relative addressing.

    SP-relative loads/stores use [sp, $imm] in the asm_string.  SP is
    pointer register 6 (p6) in AIE2.  These instructions have no explicit
    pointer operand -- SP is implicit in the encoding.

    Examples: VLDA_dmw_lda_w_ag_spill, VST_dmw_sts_w_ag_spill.
    """
    asm = instr.get("asm_string", "").lower()
    return "[sp," in asm or "[sp]" in asm


def _padda_sequence(ptr_reg: str, base_ptr: str, offset: int) -> list[str]:
    """Generate pointer arithmetic sequence to reach a large offset.

    PADDA immediate range is 12-bit signed: [-4096, 4095].
    For larger offsets, emit multiple PADDA instructions.
    """
    max_step = 1024  # PADDA accepts up to ~1536; use 1024 for safety
    lines = [f"  mov {ptr_reg}, {base_ptr}"]
    remaining = offset
    while remaining > max_step:
        lines.append(f"  padda [{ptr_reg}], #{max_step}")
        remaining -= max_step
    if remaining > 0:
        lines.append(f"  padda [{ptr_reg}], #{remaining}")
    return lines


def _scalar_load(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate scalar load with pointer arithmetic for large offsets.

    lda offset range: [-128, 124] (6-bit signed, step 4).
    For offsets beyond this, copy base to p6 and use padda.
    """
    if 0 <= offset <= MAX_SCALAR_OFFSET:
        return [f"  lda {reg_name}, [{ptr}, #{offset}]"]
    return _padda_sequence("p6", ptr, offset) + [
        f"  lda {reg_name}, [p6, #0]",
    ]


def _vector_load(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate vector load with pointer arithmetic for large offsets.

    vlda offset range: [-1024, 992] (6-bit signed, step 32).
    """
    if 0 <= offset <= MAX_VECTOR_OFFSET:
        return [f"  vlda {reg_name}, [{ptr}, #{offset}]"]
    return _padda_sequence("p6", ptr, offset) + [
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
        # Shift registers (s0-s3) can't be loaded directly via lda.
        # Load into scratch scalar r14, then mov to shift register.
        if reg_name.startswith("s"):
            return _scalar_load("r14", ptr, offset) + [
                f"  mov {reg_name}, r14",
            ]
        # Register pairs (r17:r16, etc.) need two loads.
        if ":" in reg_name:
            hi, lo = reg_name.split(":")
            return (
                _scalar_load(lo, ptr, offset)
                + _scalar_load(hi, ptr, offset + 4)
            )
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
        # Load data INTO the accumulator register file using UPS.
        # The vector and accumulator register files are separate in AIE2.
        # Simply loading wl{idx} does NOT populate the accumulator -- we must
        # use vups.s32.s16 to convert 16-bit vector elements into 32-bit
        # accumulator elements.
        #
        # Round-trip: vlda -> vups -> [instruction] -> vsrs -> vst
        # UPS with s0=0 zero-extends 16->32 bit, SRS with s0=0 truncates
        # 32->16 bit.  Lossless when values fit in 16 bits (they do, since
        # we loaded 16-bit data from the input buffer).
        #
        # s0 must be initialized to 0 before vups (shift amount).
        init_s3 = ["  mov r14, #0", "  mov s3, r14"]
        if reg_name.startswith("cm"):
            # cm0 = {bml0, bmh0} -- load both halves via UPS.
            idx = reg_name[2:]
            return (
                [f"  // load accumulator {reg_name}: vlda + vups both halves"]
                + init_s3
                + _vector_load(f"wl{idx}", ptr, offset)
                + _vector_load(f"wh{idx}", ptr, offset + 32)
                + _nop_sled(LOAD_LATENCY)
                + [f"  vups.s32.s16 bml{idx}, wl{idx}, s3",
                   f"  vups.s32.s16 bmh{idx}, wh{idx}, s3"]
            )
        elif reg_name.startswith("bml") or reg_name.startswith("bmh"):
            idx = reg_name[3:]
            return (
                [f"  // load accumulator {reg_name}: vlda + vups"]
                + init_s3
                + _vector_load(f"wl{idx}", ptr, offset)
                + _nop_sled(LOAD_LATENCY)
                + [f"  vups.s32.s16 {reg_name}, wl{idx}, s3"]
            )
        elif reg_name.startswith("am"):
            # am quarters (amll, amlh, amhl, amhh) can't be UPS'd directly.
            # Load the parent half-accumulator instead:
            #   amll/amlh -> bml,  amhl/amhh -> bmh
            idx = reg_name[-1]
            if reg_name.startswith("amh"):
                parent = f"bmh{idx}"
            else:
                parent = f"bml{idx}"
            return (
                [f"  // load accumulator {reg_name}: vlda + vups via {parent}"]
                + init_s3
                + _vector_load(f"wl{idx}", ptr, offset)
                + _nop_sled(LOAD_LATENCY)
                + [f"  vups.s32.s16 {parent}, wl{idx}, s3"]
            )
        else:
            idx = "0"
            return (
                [f"  // load accumulator {reg_name}: vlda + vups"]
                + init_s3
                + _vector_load(f"wl{idx}", ptr, offset)
                + _nop_sled(LOAD_LATENCY)
                + [f"  vups.s32.s16 bml{idx}, wl{idx}, s3"]
            )
    if kind == "control":
        return _scalar_load(reg_name, ptr, offset)
    if kind in ("modifier_m", "modifier_dj"):
        return _scalar_load(reg_name, ptr, offset)
    if kind == "quad":
        # 128-byte (1024-bit) quadword register load via scalar 'lda' slot.
        # Syntax: lda q0, [ptr, #imm]  where imm has scale=16.
        # We use #0 (zero offset) and advance the pointer with padda if needed.
        # LDA_dmv_lda_q_ag_idx_imm: immediate range [-512, 496] (6-bit * 16).
        MAX_QUAD_OFFSET = 496  # 6-bit signed * scale 16
        if 0 <= offset <= MAX_QUAD_OFFSET:
            return [f"  lda {reg_name}, [{ptr}, #{offset}]"]
        return _padda_sequence("p6", ptr, offset) + [
            f"  lda {reg_name}, [p6, #0]",
        ]
    return [f"  // unsupported load for kind={kind}"]


def _scalar_store(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate scalar store with pointer arithmetic for large offsets."""
    if 0 <= offset <= MAX_SCALAR_OFFSET:
        return [f"  st {reg_name}, [{ptr}, #{offset}]"]
    return _padda_sequence("p7", ptr, offset) + [
        f"  st {reg_name}, [p7, #0]",
    ]


def _vector_store(reg_name: str, ptr: str, offset: int) -> list[str]:
    """Generate vector store with pointer arithmetic for large offsets."""
    if 0 <= offset <= MAX_VECTOR_OFFSET:
        return [f"  vst {reg_name}, [{ptr}, #{offset}]"]
    return _padda_sequence("p7", ptr, offset) + [
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
        # Shift registers (s0-s3) can't be stored directly via st.
        # mov to scratch scalar r14, then store.
        if reg_name.startswith("s"):
            return [f"  mov r14, {reg_name}"] + _scalar_store("r14", ptr, offset)
        # Register pairs (r17:r16, etc.) need two stores.
        if ":" in reg_name:
            hi, lo = reg_name.split(":")
            return (
                _scalar_store(lo, ptr, offset)
                + _scalar_store(hi, ptr, offset + 4)
            )
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
        # s0 must be initialized to 0 (shift amount) before use.
        init_s3 = ["  mov r14, #0", "  mov s3, r14"]
        if reg_name.startswith("cm"):
            # cm0 = {bml0, bmh0}.  Store BOTH halves via SRS.
            idx = reg_name[2:]
            return (
                [f"  // store accumulator {reg_name}: srs both halves"]
                + init_s3
                + [f"  vsrs.s16.s32 wl{idx}, bml{idx}, s3"]
                + _nop_sled(4)
                + _vector_store(f"wl{idx}", ptr, offset)
                + [f"  vsrs.s16.s32 wh{idx}, bmh{idx}, s3"]
                + _nop_sled(4)
                + _vector_store(f"wh{idx}", ptr, offset + 32)
            )
        elif reg_name.startswith("bml") or reg_name.startswith("bmh"):
            idx = reg_name[3:]
            return (
                [f"  // store accumulator {reg_name}: srs to wl{idx} then vst"]
                + init_s3
                + [f"  vsrs.s16.s32 wl{idx}, {reg_name}, s3"]
                + _nop_sled(4)
                + _vector_store(f"wl{idx}", ptr, offset)
            )
        elif reg_name.startswith("am"):
            # am quarters can't be SRS'd directly -- use parent half.
            idx = reg_name[-1]
            if reg_name.startswith("amh"):
                parent = f"bmh{idx}"
            else:
                parent = f"bml{idx}"
            return (
                [f"  // store accumulator {reg_name}: srs via {parent}"]
                + init_s3
                + [f"  vsrs.s16.s32 wl{idx}, {parent}, s3"]
                + _nop_sled(4)
                + _vector_store(f"wl{idx}", ptr, offset)
            )
        else:
            idx = "0"
            return (
                [f"  // store accumulator {reg_name}: srs bml{idx} then vst"]
                + init_s3
                + [f"  vsrs.s16.s32 wl{idx}, bml{idx}, s3"]
                + _nop_sled(4)
                + _vector_store(f"wl{idx}", ptr, offset)
            )
    if kind == "control":
        # Control registers (crSat, crRnd, etc.) cannot be stored directly
        # via 'st' -- 'st' only accepts general-purpose scalar registers (eR
        # class).  Read the control register back into r14 via 'mov r14,
        # <ctrl_reg>', then store r14.  The 'mov' instruction uses the
        # MOV_mv_scl variant (mv slot) which accepts mMvSclSrc as source,
        # and mMvSclSrc includes mCRm (control regs).
        return [f"  mov r14, {reg_name}"] + _scalar_store("r14", ptr, offset)
    if kind in ("modifier_m", "modifier_dj"):
        return _scalar_store(reg_name, ptr, offset)
    if kind == "quad":
        # 128-byte (1024-bit) quadword register store via scalar 'st' slot.
        # Syntax: st q0, [ptr, #imm]  where imm has scale=16.
        # We use #0 (zero offset) and advance the pointer with padda if needed.
        # ST_dmv_sts_q_ag_idx_imm: immediate range [-512, 496] (6-bit * 16).
        MAX_QUAD_OFFSET = 496  # 6-bit signed * scale 16
        if 0 <= offset <= MAX_QUAD_OFFSET:
            return [f"  st {reg_name}, [{ptr}, #{offset}]"]
        return _padda_sequence("p7", ptr, offset) + [
            f"  st {reg_name}, [p7, #0]",
        ]
    return [f"  // unsupported store for kind={kind}"]


def _nop_sled(count: int = 5) -> list[str]:
    """Generate a NOP sled for pipeline safety."""
    return [f"  nop" for _ in range(count)]


# Fixed-register operands in TableGen that aren't exported as variable
# operands (e.g., eR29 = "always r29").  These appear as unsubstituted
# $name tokens after normal operand substitution.
FIXED_REGISTER_MAP = {
    "idx": "r29",
    "sel": "r28",
}


# Mnemonic suffixes that llvm-mc does not accept when a modifier_m
# operand is present.  llvm-mc infers the addressing mode from the
# operand types, so ".2d", ".3d", and trailing data-type suffixes
# (e.g., ".s16", ".u8") must be stripped from the asm_string.
#
# Pattern: strip ".2d" or ".3d" and everything after it from the mnemonic.
# Examples: "lda.2d.s16" -> "lda", "vlda.3d" -> "vlda", "padda.2d" -> "padda"
_MODIFIER_SUFFIX_RE = re.compile(r'\.(2d|3d)\b[.\w]*')


def _has_modifier_operand(instr: dict) -> bool:
    """Check if instruction has a modifier_m operand."""
    return any(
        op.get("register_kind") == "modifier_m"
        for op in instr.get("operands", [])
    )


def _is_postmodify_immediate(asm_string: str, op_name: str) -> bool:
    """Check if an immediate operand is a post-modify update amount.

    Post-modify:  lda r0, [ptr], $imm   -- ], $imm  (update after access)
    Basic offset: lda r0, [ptr, $imm]   -- , $imm]  (offset within brackets)

    Post-modify immediates must NOT be zeroed -- they are the pointer
    update amount, not an address offset.
    """
    # Pattern: closing bracket followed by the operand reference.
    return f"], ${op_name}" in asm_string


def _normalize_mnemonic(asm_string: str, has_modifier: bool) -> str:
    """Strip internal mnemonic suffixes that llvm-mc doesn't accept.

    When an instruction has a modifier_m operand, llvm-mc expects the
    base mnemonic (e.g., "lda" not "lda.2d.s16") and infers the
    addressing mode from the operands.
    """
    if not has_modifier:
        return asm_string
    # The mnemonic is everything before the first tab or space.
    parts = re.split(r'(\t| )', asm_string, maxsplit=1)
    if len(parts) < 2:
        return asm_string
    mnemonic = parts[0]
    rest = ''.join(parts[1:])
    # Strip .2d/.3d and everything after in the mnemonic.
    normalized = _MODIFIER_SUFFIX_RE.sub('', mnemonic)
    return normalized + rest


def _substitute_asm(asm_string: str, regs: dict[str, str],
                    has_modifier: bool = False) -> str:
    """Substitute operand placeholders in asm_string with register names.

    The asm_string looks like "add\\t$mRx, $mRx0, $mRy".
    We replace each $name with the corresponding value from regs.
    We also replace \\t with a real tab.

    If has_modifier is True, strip .2d/.3d suffixes that llvm-mc
    doesn't accept when a modifier_m operand is present.
    """
    result = _normalize_mnemonic(asm_string, has_modifier)
    # Sort by length descending to avoid partial substitution
    # (e.g., $mRx before $mRx0 would be wrong -- but $mRx0 before $mRx is fine).
    for name in sorted(regs.keys(), key=len, reverse=True):
        value = regs[name]
        # Immediate values (numeric) need # prefix for llvm-mc syntax.
        if value.lstrip("-").isdigit():
            value = f"#{value}"
        result = result.replace(f"${name}", value)

    # Replace any remaining $name tokens with known fixed registers.
    for name, fixed_reg in FIXED_REGISTER_MAP.items():
        result = result.replace(f"${name}", fixed_reg)

    # Zero-fill any remaining dontcare operands.
    result = re.sub(r'\$dontcare\w*', '#0', result)

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
        # Accept both plain register types and testable composite kinds.
        kind = _effective_kind(op)
        if not kind:
            continue
        if kind not in KNOWN_REGISTER_KINDS:
            continue
        # Align offset for this operand's natural alignment.
        align = 32 if kind in ("vector256", "vector512", "accumulator") else 4
        cur_in_offset = _align(cur_in_offset, align)
        reg = regs.get(op_name, "r0")
        load_lines = _load_instruction(reg, kind, "p0", cur_in_offset)
        lines.extend(load_lines)
        cur_in_offset += _operand_size(op, name)

    # NOP sled before instruction: must cover load pipeline latency.
    # AIE2 lda/vlda latency is 7 cycles (from AIE2Schedule.td).
    lines.extend(_nop_sled(LOAD_LATENCY))

    # The instruction itself.
    asm_line = "  " + _substitute_asm(instr["asm_string"], regs,
                                      has_modifier=_has_modifier_operand(instr))
    lines.append(asm_line)

    # NOP sled after instruction: must cover result latency before store.
    # Per-instruction from the scheduling model (II_ABS=1, II_VMAC=5, etc.).
    lines.extend(_nop_sled(result_latency(instr)))

    # Store output registers to [p1, #offset].
    cur_out_offset = out_offset
    for op in outputs:
        # Use effective kind so testable composite kinds store correctly.
        kind = _effective_kind(op) or op.get("register_kind", "")
        # Align offset for this operand's natural alignment.
        align = 32 if kind in ("vector256", "vector512", "accumulator") else 4
        cur_out_offset = _align(cur_out_offset, align)
        reg = regs.get(op["name"], "r0")
        store_lines = _store_instruction(reg, kind, "p1", cur_out_offset)
        lines.extend(store_lines)
        cur_out_offset += _operand_size(op, name)

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
    # Resolve cm-class unknowns so that combo generation assigns cm register
    # names (cm0, cm2, ...) instead of falling through to the "else: 0" branch.
    operands = [_resolve_unknown_operand(op) for op in operands]
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

        if op_type in REGISTER_LIKE_TYPES:
            kind = op.get("register_kind", "")
            bw = op.get("bit_width", 0)
            names = register_names(kind, bw, operand_type=op_type,
                                   instr_name=instr.get("name", ""))
            if not names:
                defaults[op_name] = "r0"
                alternatives[op_name] = []
                continue
            defaults[op_name] = names[0]
            alternatives[op_name] = names[1:]

        elif op_type == "composite_register":
            # Testable composite kinds have a known register name mapping.
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
            scale = op.get("scale") or 1
            vals = immediate_values(bit_width, signed, scale)
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
# ComputeStrategy: wraps existing classify + generate logic
# ---------------------------------------------------------------------------

class ComputeStrategy(TestStrategy):
    """Tests compute instructions (ALU, vector, moves).

    This wraps the original classify_instruction() and generate_test_point()
    with no behavioral change.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        status, reason = classify_instruction(instr)
        return (status == "testable", reason)

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        return generate_test_point(instr, regs, in_offset, out_offset)


# ---------------------------------------------------------------------------
# LoadStrategy: tests load instructions
# ---------------------------------------------------------------------------

class LoadStrategy(TestStrategy):
    """Tests load instructions by pointing at input buffer and capturing result.

    Handles Tier 1 (ptr+offset, ptr+index) and Tier 2 (ptr+modifier 2D/3D).
    Also handles SP-relative spill loads: [sp, $imm] addressing where SP (p6)
    is initialized to the input buffer for the test.

    Rejects: may_store=True, no pointer operand (non-SP-relative), composite
    destinations, compressed/sparse loads.
    """

    _SKIP_PREFIXES = ("vldb.compr", "vldb.sparse")

    def can_test(self, instr: dict) -> tuple[bool, str]:
        if not instr.get("may_load", False):
            return (False, "not a load instruction")
        if instr.get("may_store", False):
            return (False, "load+store combo (actually a store)")

        mnemonic = instr.get("mnemonic", "")
        if any(mnemonic.startswith(p) for p in self._SKIP_PREFIXES):
            return (False, "compressed/sparse load (needs FIFO init)")

        # Reject conversion loads (ups/conv/unpack) -- llvm-mc has no
        # single-instruction syntax for these.
        if any(s in mnemonic for s in CONVERSION_SUFFIXES):
            return (False, "conversion load (no llvm-mc syntax)")

        operands = instr.get("operands", [])
        sp_relative = _is_sp_relative(instr)

        # Must have a pointer operand, unless SP-relative (SP is implicit).
        # SP-relative instructions like VLDA_dmw_lda_w_ag_spill encode the
        # base pointer as the SP register and list no pointer in operands.
        has_pointer = any(
            op.get("register_kind") == "pointer"
            for op in operands
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )
        if not has_pointer and not sp_relative:
            return (False, "load with no pointer operand (register shuffle)")

        # Reject composite destinations unless they are in TESTABLE_COMPOSITE_KINDS.
        # LDA_dms_* instructions use mLdaScl (kind="LdaScl") which maps to scalar.
        for op in operands:
            if op.get("operand_type") == "composite_register":
                ck = op.get("register_kind", "")
                if ck not in TESTABLE_COMPOSITE_KINDS:
                    return (False, "composite register destination (deferred)")

        # Detect the output (loaded register).
        dest = self._detect_load_dest(instr)
        if dest is None:
            return (False, "no detectable load destination")

        kind = _effective_kind(dest)
        if not kind:
            kind = dest.get("register_kind", "")
        if kind not in KNOWN_REGISTER_KINDS:
            return (False, f"unsupported load destination kind: {kind}")

        # Reject accumulator-dest scalar loads: llvm-mc doesn't accept
        # "lda cm0, [ptr]" syntax even though AIE2 hardware supports it.
        # Exception: accumulator bw=2 is actually the quad register class
        # (q0-q3, mQQa in TableGen) which uses the same scalar 'lda' slot
        # and IS accepted by llvm-mc as "lda q0, [ptr, #imm]".
        mnemonic = instr.get("mnemonic", "")
        if kind == "accumulator" and not mnemonic.startswith(("vlda", "vldb")):
            bw = dest.get("bit_width", 0)
            if bw != 2:
                return (False, "accumulator dest with scalar lda (unsupported by llvm-mc)")
            # bw=2: quad register (q0-q3) -- fall through, handled below.

        if kind == "accumulator":
            bw = dest.get("bit_width", 0)
            if bw == 2:
                # Quad registers: handled as "quad" kind in generate_test_point.
                pass
            elif bw == 6:
                return (False, "accumulator quarter register (bw=6)")

        # Reject unknown operand types (except dontcare and cm-class resolvable).
        for op in operands:
            op = _resolve_unknown_operand(op)
            op_type = op.get("operand_type", "unknown")
            if op_type == "unknown":
                if not op.get("name", "").startswith("dontcare"):
                    return (False, "unknown operand type")

        return (True, "")

    def _detect_load_dest(self, instr: dict) -> Optional[dict]:
        """For loads, the destination is the first register in asm_string."""
        outputs = detect_output_operands(instr)
        return outputs[0] if outputs else None

    def compute_input_size(self, instr, regs):
        dest = self._detect_load_dest(instr)
        if dest is None:
            return 4
        return _operand_size(dest, instr.get("name", ""))

    def compute_output_size(self, instr):
        dest = self._detect_load_dest(instr)
        if dest is None:
            return 4
        return _operand_size(dest, instr.get("name", ""))

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        lines = []
        name = instr["name"]
        lines.append(f"  // ---- test (load): {name} ----")

        dest = self._detect_load_dest(instr)
        dest_name = dest["name"]
        # Use effective kind: resolves composite (LdaScl -> scalar) and
        # accumulator bw=2 (DMV_Q quad -> "quad") for correct load/store dispatch.
        dest_kind = _effective_load_store_kind(dest)
        dest_reg = regs.get(dest_name, "r0")

        # Align input offset for this destination type.
        align = _kind_alignment(dest_kind)
        in_offset = _align(in_offset, align)

        op_by_name = {op["name"]: op for op in instr.get("operands", [])}
        sp_relative = _is_sp_relative(instr)

        if sp_relative:
            # SP-relative load: SP (p6) is the implicit base pointer.
            # Initialize SP to point at the input data region.  All
            # address-offset immediates are zeroed so data is read from SP.
            lines.extend(_padda_sequence("p6", "p0", in_offset))
        else:
            # Normal pointer load: copy p0 to p6 and advance to data region.
            lines.extend(_padda_sequence("p6", "p0", in_offset))

        # For modifier operands (2D/3D loads), zero the modifier.
        for op_name, op in op_by_name.items():
            kind = op.get("register_kind", "")
            if kind == "modifier_m":
                mod_reg = regs.get(op_name, "m0")
                lines.append(f"  mov {mod_reg}, #0")
            elif kind == "modifier_dj":
                dj_reg = regs.get(op_name, "dj0")
                lines.append(f"  mov {dj_reg}, #0")

        # NOP sled to cover setup latency.
        lines.extend(_nop_sled(2))

        # Execute: the load instruction itself.
        # For pointer loads: override pointer operand to use p6.
        # For SP-relative loads: no pointer operand to override; SP is already
        #   set to the data address above.
        # Zero address-offset immediates (data is already at p6 / sp).
        # Preserve post-modify immediates (they update the pointer, not
        # the address -- zeroing them makes the test degenerate).
        asm_string = instr["asm_string"]
        load_regs = dict(regs)
        for op_name, op in op_by_name.items():
            if op.get("register_kind") == "pointer":
                load_regs[op_name] = "p6"
            if op.get("operand_type") == "immediate":
                if _is_postmodify_immediate(asm_string, op_name):
                    pass  # Keep combo-specified value.
                else:
                    load_regs[op_name] = "0"

        asm_line = "  " + _substitute_asm(instr["asm_string"], load_regs,
                                          has_modifier=_has_modifier_operand(instr))
        lines.append(asm_line)

        # NOP sled for load result latency.
        lines.extend(_nop_sled(result_latency(instr)))

        # Capture: store loaded value to output buffer.
        out_offset = _align(out_offset, align)
        store_lines = _store_instruction(dest_reg, dest_kind, "p1", out_offset)
        lines.extend(store_lines)

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load/store kind helpers
# ---------------------------------------------------------------------------

def _effective_load_store_kind(op: dict) -> str:
    """Resolve the effective register kind for a load/store operand.

    This extends _effective_kind() to also handle the accumulator-bw=2
    (DMV_Q quad) case: the ISA exporter labels q0-q3 registers as
    'accumulator' with bit_width=2 because of a 2-bit encoding for 4
    entries.  In load/store context these are 128-byte quad registers.

    Returns the kind string suitable for _load_instruction / _store_instruction.
    """
    # First try the composite_register -> TESTABLE_COMPOSITE_KINDS path.
    eff = _effective_kind(op)
    if eff:
        return eff
    kind = op.get("register_kind", "") or ""
    if kind == "accumulator" and op.get("bit_width", 0) == 2:
        # DMV_Q quad registers (q0-q3) misclassified by exporter as bw=2 accum.
        return "quad"
    return kind


def _kind_alignment(kind: str) -> int:
    """Return the natural memory alignment (bytes) for a register kind."""
    if kind in ("vector256", "accumulator"):
        return 32
    if kind in ("vector512",):
        return 32  # two 32-byte vlda instructions
    if kind == "quad":
        return 128  # 128-byte quadword register
    return 4


# ---------------------------------------------------------------------------
# StoreStrategy: tests store instructions
# ---------------------------------------------------------------------------

class StoreStrategy(TestStrategy):
    """Tests store instructions by loading known data then executing the store.

    The store writes to the output buffer via p7 (normal) or via SP/p6
    (SP-relative spill stores).  The verification is that HW and EMU produce
    identical output bytes.

    Also handles SP-relative spill stores: [sp, $imm] addressing where SP (p6)
    is initialized to the output buffer for the test.
    """

    def can_test(self, instr: dict) -> tuple[bool, str]:
        if not instr.get("may_store", False):
            return (False, "not a store instruction")

        # Reject conversion stores (srs/conv/pack) -- llvm-mc has no
        # single-instruction syntax for these.
        mnemonic = instr.get("mnemonic", "")
        if any(s in mnemonic for s in CONVERSION_SUFFIXES):
            return (False, "conversion store (no llvm-mc syntax)")

        operands = instr.get("operands", [])
        sp_relative = _is_sp_relative(instr)

        # Must have a pointer operand, unless SP-relative (SP is implicit).
        # SP-relative spill stores encode the destination address as the SP
        # register and list no pointer in the operand table.
        has_pointer = any(
            op.get("register_kind") == "pointer"
            for op in operands
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )
        if not has_pointer and not sp_relative:
            return (False, "store with no pointer operand")

        # Reject composite operands unless they are in TESTABLE_COMPOSITE_KINDS.
        # ST_dms_* instructions use mSclSt (kind="LdaScl") which maps to scalar.
        for op in operands:
            if op.get("operand_type") == "composite_register":
                ck = op.get("register_kind", "")
                if ck not in TESTABLE_COMPOSITE_KINDS:
                    return (False, "composite register operand (deferred)")

        # Detect the source (data to be stored).
        source = self._detect_store_source(instr)
        if source is None:
            return (False, "no detectable store source")

        kind = _effective_load_store_kind(source)
        if not kind:
            kind = source.get("register_kind", "")
        if kind not in KNOWN_REGISTER_KINDS:
            return (False, f"unsupported store source kind: {kind}")

        # Reject accumulator-source scalar stores: llvm-mc doesn't accept
        # "st cm0, [ptr]" syntax even though AIE2 hardware supports it.
        # Exception: accumulator bw=2 is actually the quad register class (q0-q3)
        # which uses scalar 'st' slot and IS accepted as "st q0, [ptr, #imm]".
        mnemonic = instr.get("mnemonic", "")
        if kind == "accumulator" and not mnemonic.startswith("vst"):
            bw = source.get("bit_width", 0)
            if bw != 2:
                return (False, "accumulator source with scalar st (unsupported by llvm-mc)")
            # bw=2: quad register (q0-q3) -- fall through, handled below.

        # Reject unknown operand types (except dontcare and cm-class resolvable).
        for op in operands:
            op = _resolve_unknown_operand(op)
            op_type = op.get("operand_type", "unknown")
            if op_type == "unknown":
                if not op.get("name", "").startswith("dontcare"):
                    return (False, "unknown operand type")

        return (True, "")

    def _detect_store_source(self, instr: dict) -> Optional[dict]:
        """For stores, the source is the first non-pointer, non-modifier
        register in asm_string.

        Accepts both plain register operands (operand_type in REGISTER_LIKE_TYPES)
        and testable composite operands (operand_type="composite_register" with
        kind in TESTABLE_COMPOSITE_KINDS), so that DMS scalar stores (mSclSt)
        are detected correctly.
        """
        asm_string = instr.get("asm_string", "")
        asm_op_names = re.findall(r'\$(\w+)', asm_string)
        op_by_name = {op["name"]: op for op in instr.get("operands", [])}

        for name in asm_op_names:
            if name not in op_by_name:
                continue
            op = op_by_name[name]
            op_type = op.get("operand_type", "")
            is_register_like = op_type in REGISTER_LIKE_TYPES
            is_testable_composite = (
                op_type == "composite_register"
                and op.get("register_kind", "") in TESTABLE_COMPOSITE_KINDS
            )
            if not is_register_like and not is_testable_composite:
                continue
            kind = op.get("register_kind", "")
            if kind not in ("pointer", "modifier_m", "modifier_dj"):
                return op
        return None

    def _compute_store_width(self, instr: dict) -> int:
        """Determine how many bytes the store instruction actually writes."""
        source = self._detect_store_source(instr)
        if source is None:
            kind = "scalar"
        else:
            kind = _effective_load_store_kind(source) or source.get("register_kind", "scalar")

        if kind == "vector256":
            return 32
        if kind == "vector512":
            return 64
        if kind == "accumulator":
            return 64
        if kind == "quad":
            return 128

        # Scalar stores: check mnemonic for data width.
        mnemonic = instr.get("mnemonic", "")
        if ".s8" in mnemonic or ".u8" in mnemonic:
            return 1
        if ".s16" in mnemonic or ".u16" in mnemonic:
            return 2
        return 4

    def compute_input_size(self, instr, regs):
        source = self._detect_store_source(instr)
        if source is None:
            return 4
        return _operand_size(source, instr.get("name", ""))

    def compute_output_size(self, instr):
        return self._compute_store_width(instr)

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        lines = []
        name = instr["name"]
        lines.append(f"  // ---- test (store): {name} ----")

        source = self._detect_store_source(instr)
        src_name = source["name"]
        # Use effective kind: resolves composite (LdaScl -> scalar) and
        # accumulator bw=2 (DMV_Q quad -> "quad") for correct load dispatch.
        src_kind = _effective_load_store_kind(source)
        src_reg = regs.get(src_name, "r0")

        op_by_name = {op["name"]: op for op in instr.get("operands", [])}
        sp_relative = _is_sp_relative(instr)

        # Align and load known data into the source register.
        align = _kind_alignment(src_kind)
        in_offset = _align(in_offset, align)
        load_lines = _load_instruction(src_reg, src_kind, "p0", in_offset)
        lines.extend(load_lines)

        # NOP sled for load latency.
        lines.extend(_nop_sled(LOAD_LATENCY))

        out_offset = _align(out_offset, align)
        if sp_relative:
            # SP-relative store: SP (p6) is the implicit destination pointer.
            # Initialize SP to point at the output data region.  The immediate
            # is zeroed so the store writes to exactly [sp, #0].
            lines.extend(_padda_sequence("p6", "p1", out_offset))
        else:
            # Normal pointer store: copy p1 to p7 and advance to output region.
            lines.extend(_padda_sequence("p7", "p1", out_offset))

        # For modifier operands, zero the modifier.
        for op_name, op in op_by_name.items():
            kind = op.get("register_kind", "")
            if kind == "modifier_m":
                mod_reg = regs.get(op_name, "m0")
                lines.append(f"  mov {mod_reg}, #0")
            elif kind == "modifier_dj":
                dj_reg = regs.get(op_name, "dj0")
                lines.append(f"  mov {dj_reg}, #0")

        # Execute: the store instruction.
        # For pointer stores: override pointer to p7.
        # For SP-relative stores: no pointer operand to override; SP is already
        #   set to the output address above.
        # Zero address-offset immediates (data is at p7 / sp).
        # Preserve post-modify immediates (pointer update amount).
        asm_string = instr["asm_string"]
        store_regs = dict(regs)
        for op_name, op in op_by_name.items():
            if op.get("register_kind") == "pointer":
                store_regs[op_name] = "p7"
            if op.get("operand_type") == "immediate":
                if _is_postmodify_immediate(asm_string, op_name):
                    pass  # Keep combo-specified value.
                else:
                    store_regs[op_name] = "0"

        asm_line = "  " + _substitute_asm(instr["asm_string"], store_regs,
                                          has_modifier=_has_modifier_operand(instr))
        lines.append(asm_line)

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BranchStrategy: tests branch instructions with marker-based verification
# ---------------------------------------------------------------------------

class BranchStrategy(TestStrategy):
    """Tests branch instructions using marker-based verification.

    Stores marker values to the output buffer to verify which path executed.
    Conditional branches get two test points: taken and not-taken.
    """

    _TESTABLE = frozenset({"j", "jl", "jnz", "jz"})
    _DEFERRED = frozenset({"ret", "jnzd"})

    _label_counter = 0

    @classmethod
    def _next_label_id(cls) -> int:
        cls._label_counter += 1
        return cls._label_counter

    @classmethod
    def reset_labels(cls):
        """Reset label counter (call at start of each batch)."""
        cls._label_counter = 0

    def can_test(self, instr: dict) -> tuple[bool, str]:
        mnemonic = instr.get("mnemonic", "")
        if mnemonic in self._DEFERRED:
            return (False, f"deferred branch: {mnemonic}")
        if mnemonic not in self._TESTABLE:
            return (False, "not a branch instruction")

        # Reject register-indirect branches (j $mPm, jl $mPm).
        operands = instr.get("operands", [])
        has_pointer = any(
            op.get("register_kind") == "pointer"
            for op in operands
            if op.get("operand_type") in REGISTER_LIKE_TYPES
        )
        if has_pointer:
            return (False, "register-indirect branch (deferred)")

        return (True, "")

    def _is_conditional(self, instr: dict) -> bool:
        return instr.get("mnemonic", "") in ("jnz", "jz")

    def compute_input_size(self, instr, regs):
        return 4 if self._is_conditional(instr) else 0

    def compute_output_size(self, instr):
        return 8  # 4-byte "before" marker + 4-byte "path" marker

    def generate_combos(self, instr: dict) -> list[dict[str, str]]:
        """Conditional branches get 2 combos, unconditional get 1."""
        base_combo = {}
        for op in instr.get("operands", []):
            op_name = op["name"]
            if op_name == "cpmaddr" or op_name.startswith("cpmaddr"):
                base_combo[op_name] = "0"  # placeholder, overridden
            elif op.get("operand_type") in REGISTER_LIKE_TYPES:
                kind = op.get("register_kind", "")
                bw = op.get("bit_width", 0)
                names = register_names(kind, bw,
                                       operand_type=op.get("operand_type", ""),
                                       instr_name=instr.get("name", ""))
                base_combo[op_name] = names[0] if names else "r0"
            else:
                base_combo[op_name] = "0"

        if not self._is_conditional(instr):
            return [base_combo]

        # Two combos: nonzero + zero condition.
        return [dict(base_combo), dict(base_combo)]

    def generate_test_point(self, instr, regs, in_offset, out_offset):
        lines = []
        name = instr["name"]
        mnemonic = instr.get("mnemonic", "")
        lid = self._next_label_id()
        lines.append(f"  // ---- test (branch): {name} ----")

        BRANCH_DELAY = 5

        # For conditional branches, load the condition value from input.
        if self._is_conditional(instr):
            in_offset = _align(in_offset, 4)
            lines.extend(_scalar_load("r0", "p0", in_offset))
            lines.extend(_nop_sled(LOAD_LATENCY))

        # Store "before" marker (0xAA = 170).
        out_offset = _align(out_offset, 4)
        lines.append(f"  mov r14, #170")
        lines.extend(_scalar_store("r14", "p1", out_offset))

        # The branch instruction with label target.
        if mnemonic == "j":
            lines.append(f"  j .Ltaken_{lid}")
        elif mnemonic == "jl":
            lines.append(f"  jl .Ltaken_{lid}")
        elif mnemonic == "jnz":
            lines.append(f"  jnz r0, .Ltaken_{lid}")
        elif mnemonic == "jz":
            lines.append(f"  jz r0, .Ltaken_{lid}")

        # Branch delay slots.
        lines.extend(_nop_sled(BRANCH_DELAY))

        # Fall-through path (branch NOT taken).
        lines.append(f"  mov r14, #187")  # 0xBB
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))
        lines.append(f"  j .Ldone_{lid}")
        lines.extend(_nop_sled(BRANCH_DELAY))

        # Taken path.
        lines.append(f".Ltaken_{lid}:")
        lines.append(f"  mov r14, #204")  # 0xCC
        lines.extend(_scalar_store("r14", "p1", out_offset + 4))

        # Convergence point.
        lines.append(f".Ldone_{lid}:")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy Dispatch
# ---------------------------------------------------------------------------

STRATEGIES: list[TestStrategy] = [
    # BranchStrategy disabled: llvm-mc doesn't support label-based branch
    # targets for AIE2 (uses absolute instruction word addresses, not
    # relocatable labels). The strategy class is preserved for future use
    # when a linker with relocation support is available.
    # BranchStrategy(),
    LoadStrategy(),
    StoreStrategy(),
    ComputeStrategy(),
]


def classify_with_strategies(instr: dict) -> tuple[Optional[TestStrategy], str]:
    """Try each strategy in order. Return (strategy, "") or (None, reason)."""
    last_reason = "no strategy matched"
    for strategy in STRATEGIES:
        can, reason = strategy.can_test(instr)
        if can:
            return (strategy, "")
        if reason:
            last_reason = reason
    return (None, last_reason)


# ---------------------------------------------------------------------------
# Task 4: Test Point Metadata
# ---------------------------------------------------------------------------

def _operand_size(op: dict, instr_name: str = "") -> int:
    """Compute the storage size in bytes for a single register operand."""
    # Resolve cm-class unknowns before inspecting the kind.
    op = _resolve_unknown_operand(op)
    kind = op.get("register_kind", "")
    # For testable composite kinds, use the effective kind for sizing.
    effective = _effective_kind(op)
    if op.get("operand_type") == "composite_register" and effective:
        kind = effective
    # For accumulator bw=2 (DMV_Q quad registers q0-q3), use quad size.
    if kind == "accumulator" and op.get("bit_width", 0) == 2:
        kind = "quad"
    base = REGISTER_SIZES.get(kind, 4)
    # Register pairs (eL class) need double the scalar size.
    if (op.get("operand_type") == "register+16"
            and _needs_register_pair(instr_name)):
        return 8
    return base


def _compute_input_size(instr: dict, regs: dict[str, str]) -> int:
    """Compute total input buffer size for a test point."""
    operands = instr.get("operands", [])
    op_by_name = {op["name"]: op for op in operands}
    outputs = detect_output_operands(instr)
    output_names = {op["name"] for op in outputs}
    asm_op_names = re.findall(r'\$(\w+)', instr.get("asm_string", ""))
    instr_name = instr.get("name", "")

    total = 0
    for op_name in asm_op_names:
        if op_name in output_names:
            continue
        if op_name not in op_by_name:
            continue
        op = op_by_name[op_name]
        # Accept both plain register types and testable composite kinds.
        kind = _effective_kind(op)
        if not kind:
            continue
        if kind not in KNOWN_REGISTER_KINDS:
            continue
        align = 32 if kind in ("vector256", "vector512", "accumulator") else 4
        total = _align(total, align)
        total += _operand_size(op, instr_name)
    return max(4, total)


def _compute_output_size(instr: dict) -> int:
    """Compute total output buffer size for a test point."""
    outputs = detect_output_operands(instr)
    instr_name = instr.get("name", "")
    total = 0
    for op in outputs:
        # Use effective kind so testable composite kinds are sized correctly.
        kind = _effective_kind(op) or op.get("register_kind", "")
        align = 32 if kind in ("vector256", "vector512", "accumulator") else 4
        total = _align(total, align)
        total += _operand_size(op, instr_name)
    return max(4, total)


# ---------------------------------------------------------------------------
# Task 4: Full Pipeline
# ---------------------------------------------------------------------------

# Program memory limit: 16 KB = 16384 bytes.  Each assembly line produces
# one VLIW bundle = 16 bytes.  The mega-program header (4 lines) and
# return sequence (5 lines) consume 9 * 16 = 144 bytes of overhead.
# We use a 10% safety margin: (16384 - 144) * 0.9 = ~14616 bytes = 913 bundles.
PROG_MEM_BYTES = 16384
BUNDLE_SIZE = 16
PROG_OVERHEAD_BUNDLES = 9  # .text + .globl + label + blank + ret + 4 nops
SAFETY_MARGIN = 0.90
MAX_BUNDLES_PER_BATCH = int(
    (PROG_MEM_BYTES / BUNDLE_SIZE - PROG_OVERHEAD_BUNDLES) * SAFETY_MARGIN
)


def _count_bundles(asm_text: str) -> int:
    """Count the number of assembly bundles (non-empty, non-comment lines)."""
    count = 0
    for line in asm_text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            count += 1
    return count


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

    # Classify and generate combos.  First pass: collect test point specs
    # (instruction + operand combos) without generating assembly yet.
    # Assembly generation happens per-batch so offsets reset to zero.
    testable_count = 0
    skipped_count = 0
    skip_reasons: dict[str, int] = {}
    # (instr, combo_idx, regs, strategy)
    test_point_specs: list[tuple[dict, int, dict, TestStrategy]] = []

    for instr in all_instrs:
        strategy, reason = classify_with_strategies(instr)
        if strategy is None:
            skipped_count += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        testable_count += 1
        combos = strategy.generate_combos(instr)

        for combo_idx, regs in enumerate(combos):
            test_point_specs.append((instr, combo_idx, regs, strategy))

    # First pass: generate assembly for each test point at offset 0 to
    # measure its code size (bundle count).  This lets us bin-pack into
    # batches that fit within program memory.
    measured_specs = []  # (instr, combo_idx, regs, bundle_count, strategy)
    for instr, combo_idx, regs, strategy in test_point_specs:
        asm = strategy.generate_test_point(instr, regs, in_offset=0, out_offset=0)
        bundles = _count_bundles(asm)
        measured_specs.append((instr, combo_idx, regs, bundles, strategy))

    # Bin-pack into batches by measured code size.
    batches = []
    batch_idx = 0
    i = 0
    while i < len(measured_specs):
        # Greedily fill this batch until we'd exceed the program memory limit.
        batch_bundle_total = 0
        end = i
        while end < len(measured_specs):
            next_bundles = measured_specs[end][3]
            if batch_bundle_total + next_bundles > MAX_BUNDLES_PER_BATCH:
                break
            batch_bundle_total += next_bundles
            end += 1
        # Must include at least one test point per batch.
        if end == i:
            end = i + 1

        batch_specs = measured_specs[i:end]

        # Reset branch labels per batch to avoid cross-file collisions.
        BranchStrategy.reset_labels()

        # Second pass: regenerate assembly with correct per-batch offsets.
        batch_in_offset = 0
        batch_out_offset = 0
        batch_asm = []
        batch_meta = []

        for instr, combo_idx, regs, _bundles, strategy in batch_specs:
            in_size = strategy.compute_input_size(instr, regs)
            out_size = strategy.compute_output_size(instr)
            # Align offsets to avoid vlda/vst alignment assertions.
            batch_in_offset = _align(batch_in_offset, 32)
            batch_out_offset = _align(batch_out_offset, 32)

            asm = strategy.generate_test_point(
                instr, regs,
                in_offset=batch_in_offset,
                out_offset=batch_out_offset,
            )
            batch_asm.append(asm)
            batch_meta.append({
                "instruction": instr["name"],
                "slot": instr.get("slot", ""),
                "combo_index": combo_idx,
                "operands": regs,
                "in_offset": batch_in_offset,
                "in_size": in_size,
                "out_offset": batch_out_offset,
                "out_size": out_size,
            })

            batch_in_offset += in_size
            batch_out_offset += out_size

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
            "in_size": batch_in_offset,
            "out_size": batch_out_offset,
            "tests": batch_meta,
        })

        batch_idx += 1
        i = end

    # Write manifest.json.
    manifest = {
        "testable_instructions": testable_count,
        "skipped_instructions": skipped_count,
        "total_test_points": len(test_point_specs),
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
        strategy_counts: dict[str, int] = {}
        testable_instrs: list[tuple[dict, str]] = []

        for slot, instrs in isa_data.items():
            for instr in instrs:
                strategy, reason = classify_with_strategies(instr)
                if strategy is not None:
                    testable_count += 1
                    sname = type(strategy).__name__
                    strategy_counts[sname] = strategy_counts.get(sname, 0) + 1
                    testable_instrs.append((instr, sname))
                else:
                    skipped_count += 1
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        print(f"Testable: {testable_count}")
        print(f"Skipped:  {skipped_count}")
        print(f"Total:    {testable_count + skipped_count}")
        print()
        print("Strategy breakdown:")
        for sname, count in sorted(strategy_counts.items()):
            print(f"  {sname}: {count}")
        print()
        print("Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
        print()
        print("Testable instructions:")
        for instr, sname in testable_instrs:
            outputs = detect_output_operands(instr)
            out_names = [o["name"] for o in outputs]
            print(f"  {instr['name']:40s} [{instr['slot']:4s}] {sname:20s} outputs={out_names}")
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
